#!/usr/bin/env python3
"""Extract the NATIVE-PIXEL box geometry of the EMITTED mask, per clip, per frame.

WHY THIS EXISTS (review finding W9, 2026-08-28)
-----------------------------------------------
The shipped silhouette defence is `MASK_SHAPE_MODE="bbox"`: every person's mask is replaced by a
filled rectangle. Every silhouette number in the ledger prices it with an adversary that consumes a
(T,64,64) SIZE-NORMALISED silhouette -- GEI nearest-neighbour (Class 1) and the frozen GaitBase
(Class 2). Both were designed for real silhouettes, and normalisation throws away the one thing a
rectangle still carries: how big it is.

So `bbox` has never been measured against an adversary built for it. That is W9. This script
extracts what such an adversary actually holds.

WHAT AN ATTACKER HOLDS
----------------------
They hold the released mask video. In `bbox` mode that is, per frame, a rectangle. Everything that
survives the defence is therefore in (x0, y0, x1, y1) over time -- height, width, aspect, area,
centroid track, and how those vary. Height in particular is close to a direct read of stature.

PROTOCOL (PROTOCOL.md R1-R5)
----------------------------
R5 is the binding one here: the defence is applied WHERE IT IS DEPLOYED. `mask_mitigate` is
imported from tree/mirage_edge_deploy/tier1_raspberry_pi5 and run on the native-resolution mask,
exactly as evaluation/privacy/appearance/extract_arm.py does it (setup_mirage_mask_config is mirrored
line for line, including the per-CLIP seed). Boxes are then measured on the EMITTED mask in NATIVE
pixels. Nothing is normalised anywhere in this file -- normalising is the very step that hid the
channel.

Two arms are written so the defence can be priced by LIFT OVER ITS OWN NULL rather than by raw
accuracy:
  raw     - the box of the undefended YOLO silhouette   (positive control / no defence)
  mirage  - the box of the emitted `bbox` mask          (SHIPPED)

  python extract_boxes.py --video-dir corpus_10fps --manifest corpus_10fps.json \n                          --annotations <fullbody-annotations> --out boxes.npz
"""
import argparse
import hashlib
import json
import os
import subprocess
import sys
import time

import cv2
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.normpath(os.path.join(HERE, "..", "..", ".."))          # repository root
EDGE = os.path.join(ROOT, "tier1", "src", "edge_runner_pi5")
sys.path.insert(0, EDGE)

# The capture corpus is not redistributable, so --video-dir, --manifest and --annotations must be
# supplied. They are the SAME inputs evaluation/privacy/appearance/extract_arm.py takes, and the
# annotations are the same hand-verified subject boxes; see that file's docstring.

MAXF = 120        # the published frame budget, identical to extract_arm.MAXF
WIN = 2           # MASK_TEMPORAL_WIN, pinned (it derives from a DURATION and would be 1 at 10 fps)
EPS = 0.01        # MASK_SIMPLIFY_EPS
STALE_MAX_S = 0.20
YOLO_W = os.path.join(ROOT, "models", "yolo11s-seg.pt")


def ann_key(relpath):
    return relpath.replace(" ", "").replace("/", "__") + ".json"


def src_fps(path):
    p = subprocess.run(["ffprobe", "-v", "error", "-select_streams", "v:0",
                        "-show_entries", "stream=r_frame_rate", "-of", "csv=p=0", path],
                       capture_output=True, text=True)
    a, b = p.stdout.strip().split("/")
    return float(a) / float(b)


def iou(a, b):
    ix0, iy0 = max(a[0], b[0]), max(a[1], b[1])
    ix1, iy1 = min(a[2], b[2]), min(a[3], b[3])
    iw, ih = max(0.0, ix1 - ix0), max(0.0, iy1 - iy0)
    inter = iw * ih
    ua = (a[2] - a[0]) * (a[3] - a[1]) + (b[2] - b[0]) * (b[3] - b[1]) - inter
    return inter / ua if ua > 0 else 0.0


def setup_mirage_mask_config(seed):
    """Verbatim from evaluation/privacy/appearance/extract_arm.py, the same setup Classes 1 and 2 use.

    Every knob this study touches is reset to its shipped default before `bbox` is selected, so arm
    state cannot leak between clips. The seed is PER CLIP -- per-identity seeding leaks 6-8x chance
    on every arm (ledger A.2j-2) and would silently invalidate the measurement.
    """
    import config as C
    C.MASK_TEMPORAL_WIN = WIN
    C._MASK_DISPLACE_SEED = seed
    C._MASK_DISPLACE_PHASE = 0.0
    C.MASK_BBOX_MERGE, C.MASK_BBOX_PAD_FRAC = True, 0.0
    C.MASK_ELLIPSE_INFLATE = 1.15
    C.MASK_CLOSE_KERNEL_FRAC = 0.25
    C.MASK_RADIALLP_KEEP, C.MASK_RADIALLP_BINS = 4, 180
    C.MASK_DISPLACE_AMP_FRAC = 0.25
    C.MASK_DISPLACE_PHASE_STEP = 0.35
    C.MASK_DIRBAND_AMP = (0.10, 0.40)
    C.MASK_DIRBAND_N = (4, 7)
    C.MASK_BAND_AMP = (0.10, 0.40)
    C.MASK_BAND_N = (4, 7)
    C.MASK_SHAPE_MODE = "bbox"
    return C


def box_of(mask):
    """Tight native-pixel box of a binary mask, or None when the mask is empty."""
    ys, xs = np.nonzero(mask)
    if xs.size == 0:
        return None
    return np.array([xs.min(), ys.min(), xs.max(), ys.max()], np.float32)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default=os.path.join(HERE, "boxes.npz"))
    ap.add_argument("--video-dir", required=True,
                    help="directory of the 10 fps corpus clips (same as extract_arm.py)")
    ap.add_argument("--manifest", required=True, help="corpus_10fps.json")
    ap.add_argument("--annotations", required=True,
                    help="hand-verified per-frame subject boxes (fullbody annotation dir)")
    ap.add_argument("--limit", type=int, default=0)
    a = ap.parse_args()

    import mirage_tier1 as ST
    from ultralytics import YOLO
    yolo = YOLO(YOLO_W)

    man = json.load(open(a.manifest))
    if a.limit:
        man = man[:a.limit]

    out, t0 = {}, time.time()
    for ci, m in enumerate(man, 1):
        vid = os.path.join(a.video_dir, m["clip"])
        if not os.path.exists(vid):
            print("MISSING VIDEO " + vid, flush=True)
            continue

        d = json.load(open(os.path.join(a.annotations, ann_key(m["source_relpath"]))))
        byf = {}
        for f in d["frames"]:
            for p in f.get("people", []):
                if p["identity"] == m["identity"]:
                    byf[f["frame"]] = p["bbox_xyxy"]
                    break
        if not byf:
            print("NO SUBJECT BOXES " + m["clip"], flush=True)
            continue
        keys = np.array(sorted(byf))
        f_src = src_fps(os.path.join(a.video_dir, m["clip"]))

        seed = int(hashlib.sha256(m["clip"].encode()).hexdigest()[:8], 16) & 0x7FFFFFFF
        C = setup_mirage_mask_config(seed)

        cap = cv2.VideoCapture(vid)
        raw_b, mir_b, hist, n = [], [], [], -1
        H = W = 0
        while n + 1 < MAXF:
            ok, fr = cap.read()
            if not ok:
                break
            n += 1
            s = int(round(n * f_src / 10.0)) + 1
            j = int(np.argmin(np.abs(keys - s)))
            if abs(int(keys[j]) - s) / max(f_src, 1e-6) > STALE_MAX_S:
                continue                      # the box no longer contains the subject
            box = np.asarray(byf[int(keys[j])], np.float32)
            H, W = fr.shape[:2]

            r = yolo.predict(fr, classes=[0], verbose=False, device=0, half=True, imgsz=640)[0]
            mask = np.zeros((H, W), np.uint8)
            if r.masks is not None and len(r.masks) > 0:
                bxs = r.boxes.xyxy.cpu().numpy()
                k = int(np.argmax([iou(box, b) for b in bxs]))
                if iou(box, bxs[k]) > 0.10:
                    poly = np.asarray(r.masks.xy[k], np.int32)
                    if poly.size >= 6:
                        cv2.fillPoly(mask, [poly], 1)

            C._MASK_DISPLACE_PHASE = float(C.MASK_DISPLACE_PHASE_STEP) * n
            hist.append(mask)
            if len(hist) > WIN:
                hist.pop(0)
            em = ST.mask_mitigate(hist, mask, EPS)

            rb, mb = box_of(mask), box_of(em)
            if rb is not None and mb is not None:
                raw_b.append(rb)
                mir_b.append(mb)
        cap.release()

        if len(raw_b) < 20:
            print("[%d/%d] %s  DROPPED, only %d usable frames"
                  % (ci, len(man), m["clip"], len(raw_b)), flush=True)
            continue
        out[m["clip"]] = dict(raw=np.stack(raw_b), mirage=np.stack(mir_b),
                              identity=m["identity"], condition=m["condition"],
                              source=m["source_file"], frame_hw=np.array([H, W], np.float32))
        if ci % 10 == 0 or ci == len(man):
            print("[%d/%d] %s  %df  %.0fs elapsed"
                  % (ci, len(man), m["clip"], len(raw_b), time.time() - t0), flush=True)

    np.savez_compressed(
        a.out,
        clips=np.array(sorted(out)),
        **{"%s__%s" % (k, f): v[f] for k, v in out.items()
           for f in ("raw", "mirage", "identity", "condition", "source", "frame_hw")})
    print("\nwrote %s: %d clips, %d identities, %.0fs"
          % (a.out, len(out), len({v["identity"] for v in out.values()}), time.time() - t0),
          flush=True)


if __name__ == "__main__":
    main()
