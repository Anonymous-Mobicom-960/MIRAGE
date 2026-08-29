"""
segbench.run_config -- run ONE benchmarked configuration end-to-end over the
test clip and emit: (1) the grey-masked 1264x1264 output video, (2) a per-frame
record archive (union mask packed, boxes, ids, scores, per-stage timings), and
(3) a timing summary. Metrics are computed in a SEPARATE step (segbench.metrics)
against the cached pseudo-GT, so this module never spends timed cycles on
evaluation math -- the timers wrap only the real pipeline stages.
"""
import os
import sys
import time
import json
from dataclasses import dataclass, field, asdict

import numpy as np
import cv2

from .core import UltralyticsSeg, build_union_mask, apply_grey, box_centroid
from .semantic import SelfieSegNet
from .face import FaceGuard

# Reuse the SHIPPED per-person tracker from the reference repo (read-only import;
# we never write into the cloned tier1/). PersonIdentityTracker = unbounded ids,
# any number of people, greedy nearest-centroid with grace frames through
# occlusion -- exactly the "id every person, left stays left, any count" need.
_REF_SRC = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(
    os.path.abspath(__file__)))), "tier1", "src")
if _REF_SRC not in sys.path:
    sys.path.insert(0, _REF_SRC)
from mirage.export_tracking import PersonIdentityTracker  # noqa: E402


@dataclass
class Config:
    name: str
    weights: str
    family: str            # "yolo11" | "yolo26" | ...
    method: str            # "baseline" | "smooth" | "guided"
    imgsz: int = 320
    close_k: int = 5       # smooth/guided: morphological-close kernel (px), 0=off
    dilate_px: int = 8     # smooth/guided: OUTWARD privacy dilation (px), 0=off
    guided_radius: int = 12
    guided_eps: float = 1e-2
    guided_scale: float = 0.25
    work_scale: float = 1.0
    safety_net: bool = False   # union a semantic person mask (catches missed people)
    face_guard: bool = False   # detect + grey-fill faces (absolute face-coverage guarantee)
    conf: float = 0.4
    device: str = "cpu"
    fmt: str = "onnx"      # bookkeeping only (onnx/pt/ncnn/openvino/int8)
    note: str = ""


def _pack(mask_bool):
    return np.packbits(mask_bool.reshape(-1))


def run_one(cfg: Config, video_path: str, out_dir: str,
            warmup: int = 3, draw_hud: bool = False) -> dict:
    os.makedirs(out_dir, exist_ok=True)
    seg = UltralyticsSeg(cfg.weights, conf=cfg.conf)
    semnet = SelfieSegNet() if cfg.safety_net else None
    faceguard = FaceGuard() if cfg.face_guard else None

    cap = cv2.VideoCapture(video_path)
    W = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    H = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = cap.get(cv2.CAP_PROP_FPS) or 10.0
    out_hw = (H, W)

    writer = cv2.VideoWriter(os.path.join(out_dir, "masked.mp4"),
                             cv2.VideoWriter_fourcc(*"mp4v"), fps, (W, H))
    tracker = PersonIdentityTracker(W, H)

    # warm the model (JIT/first-call allocations) so frame-0 doesn't skew timing
    ok, f0 = cap.read()
    if ok:
        for _ in range(warmup):
            seg.infer(f0, cfg.imgsz, device=cfg.device)
    cap.set(cv2.CAP_PROP_POS_FRAMES, 0)

    rec = dict(t_seg=[], t_post=[], t_track=[], t_fill=[], t_total=[],
               n_persons=[], ids=[], boxes=[], scores=[], union_packed=[])
    frame_idx = 0
    while True:
        ok, frame = cap.read()
        if not ok:
            break
        t0 = time.perf_counter()

        t_seg, instances = seg.infer(frame, cfg.imgsz, device=cfg.device)

        t1 = time.perf_counter()
        guide = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY) if cfg.method == "guided" else None
        union = build_union_mask(instances, out_hw, method=cfg.method,
                                 close_k=cfg.close_k, dilate_px=cfg.dilate_px,
                                 guide_gray=guide, guided_radius=cfg.guided_radius,
                                 guided_eps=cfg.guided_eps, guided_scale=cfg.guided_scale,
                                 work_scale=cfg.work_scale)
        if semnet is not None:
            sem = semnet.person_mask(frame)
            if cfg.dilate_px > 0:
                kk = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2 * cfg.dilate_px + 1, 2 * cfg.dilate_px + 1))
                sem = cv2.dilate(sem.astype(np.uint8), kk).astype(bool)
            union = union | sem
        if faceguard is not None:
            union = union | faceguard.face_fill(frame)   # absolute face guarantee
        boxes = [ins["box"] for ins in instances]
        scores = [ins["score"] for ins in instances]
        t_post = time.perf_counter() - t1

        t2 = time.perf_counter()
        dets = [(i, box_centroid(boxes[i])) for i in range(len(boxes))]
        matched, _departed = tracker.assign(dets)
        det2id = {di: pid for pid, di in matched.items()}
        ids = [det2id.get(i, -1) for i in range(len(boxes))]
        t_track = time.perf_counter() - t2

        t3 = time.perf_counter()
        out = apply_grey(frame, union)
        t_fill = time.perf_counter() - t3

        t_total = time.perf_counter() - t0

        writer.write(out)
        rec["t_seg"].append(t_seg)
        rec["t_post"].append(t_post)
        rec["t_track"].append(t_track)
        rec["t_fill"].append(t_fill)
        rec["t_total"].append(t_total)
        rec["n_persons"].append(len(boxes))
        rec["ids"].append(ids)
        rec["boxes"].append(np.array(boxes, dtype=np.float32) if boxes else np.zeros((0, 4), np.float32))
        rec["scores"].append(np.array(scores, dtype=np.float32))
        rec["union_packed"].append(_pack(union))

        if frame_idx in (0, 25, 50, 75):
            cv2.imwrite(os.path.join(out_dir, f"sample_{frame_idx:03d}.png"), out)
        frame_idx += 1

    cap.release()
    writer.release()

    # timing stats over post-warmup frames (frame 0 already primed above, but
    # drop the first `warmup` recorded frames too for a clean steady-state read)
    def stat(key):
        arr = np.array(rec[key][warmup:]) * 1000.0  # ms
        if arr.size == 0:
            arr = np.array(rec[key]) * 1000.0
        return dict(median=float(np.median(arr)), mean=float(np.mean(arr)),
                    p90=float(np.percentile(arr, 90)), std=float(np.std(arr)))

    total_ms = stat("t_total")
    summary = dict(
        name=cfg.name, config=asdict(cfg), frames=frame_idx,
        fps_from_total=1000.0 / total_ms["median"] if total_ms["median"] else 0.0,
        t_seg_ms=stat("t_seg"), t_post_ms=stat("t_post"),
        t_track_ms=stat("t_track"), t_fill_ms=stat("t_fill"),
        t_total_ms=total_ms,
        mean_persons=float(np.mean(rec["n_persons"])),
        device=cfg.device,
    )

    # persist per-frame record (union masks packed) for the metrics pass
    np.savez_compressed(
        os.path.join(out_dir, "frames.npz"),
        union_packed=np.array(rec["union_packed"], dtype=object),
        boxes=np.array(rec["boxes"], dtype=object),
        ids=np.array(rec["ids"], dtype=object),
        scores=np.array(rec["scores"], dtype=object),
        n_persons=np.array(rec["n_persons"]),
        hw=np.array(out_hw),
        t_seg=np.array(rec["t_seg"]), t_post=np.array(rec["t_post"]),
        t_track=np.array(rec["t_track"]), t_fill=np.array(rec["t_fill"]),
        t_total=np.array(rec["t_total"]),
    )
    with open(os.path.join(out_dir, "summary.json"), "w") as fh:
        json.dump(summary, fh, indent=2)
    return summary
