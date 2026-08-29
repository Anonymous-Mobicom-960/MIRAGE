"""
segbench.metrics -- score one config's per-frame output (runs/<name>/frames.npz)
against the cached pseudo-GT. Every metric here is PLATFORM-INDEPENDENT (pixels,
not milliseconds), so these numbers are valid regardless of x86 vs ARM -- unlike
latency, which is only indicative locally.

Metric groups
  quality : mask IoU + boundary-F vs the strong-model foreground person mask
  privacy : body reveal-rate, and the hard FACE-LEAK GATE over visible faces
  utility : over-mask rate (background pixels needlessly greyed by dilation)
  counting: person-count error vs GT foreground
  identity: leftmost/rightmost ID switches, unique foreground IDs
  temporal: frame-to-frame mask IoU (flicker / stability)
"""
import os
import json
import numpy as np
import cv2

from .pseudo_gt import unpack


def _iou(a, b):
    inter = np.logical_and(a, b).sum(dtype=np.int64)
    union = np.logical_or(a, b).sum(dtype=np.int64)
    return float(inter) / float(union) if union else 1.0


def _boundary(mask, width=2):
    m = mask.astype(np.uint8)
    er = cv2.erode(m, cv2.getStructuringElement(cv2.MORPH_RECT, (2 * width + 1, 2 * width + 1)))
    return (m - er).astype(bool)


def _boundary_f(cand, gt, tol=5):
    cb, gb = _boundary(cand), _boundary(gt)
    if cb.sum() == 0 and gb.sum() == 0:
        return 1.0
    if cb.sum() == 0 or gb.sum() == 0:
        return 0.0
    dt_gt = cv2.distanceTransform((~gb).astype(np.uint8), cv2.DIST_L2, 3)
    dt_cand = cv2.distanceTransform((~cb).astype(np.uint8), cv2.DIST_L2, 3)
    prec = float((cb & (dt_gt <= tol)).sum()) / float(cb.sum())
    rec = float((gb & (dt_cand <= tol)).sum()) / float(gb.sum())
    return 2 * prec * rec / (prec + rec) if (prec + rec) else 0.0


def _fg_boxes(boxes, min_h=220):
    if boxes is None or len(boxes) == 0:
        return np.zeros((0, 4), np.float32)
    boxes = np.asarray(boxes, np.float32).reshape(-1, 4)
    return boxes[(boxes[:, 3] - boxes[:, 1]) >= min_h]


def evaluate_run(run_dir: str, gt_path: str, face_leak_thresh: float = 0.995,
                 boundary_tol: int = 5, min_person_h: int = 220) -> dict:
    fr = np.load(os.path.join(run_dir, "frames.npz"), allow_pickle=True)
    gt = np.load(gt_path, allow_pickle=True)
    hw = tuple(fr["hw"].tolist())
    N = len(fr["union_packed"])

    cand_packed = fr["union_packed"]
    cand_boxes = fr["boxes"]
    cand_ids = fr["ids"]
    n_persons_cand = np.asarray(fr["n_persons"])

    gt_fg = gt["person_union"]         # foreground-only strong mask
    gt_all = gt["person_union_all"]
    gt_boxes = gt["person_boxes"]
    face_boxes = gt["face_boxes"]

    ious, bfs, reveals, overmasks = [], [], [], []
    face_cov_min_per_frame, face_leak_frames, face_leak_px_total, face_px_total = [], 0, 0, 0
    count_err = []
    prev_cand = None
    flick = []

    left_ids, right_ids = [], []

    for i in range(N):
        cand = unpack(cand_packed[i], hw)
        gfg = unpack(gt_fg[i], hw)
        gall = unpack(gt_all[i], hw)

        # --- quality (only where GT has foreground people) ---
        if gfg.any():
            ious.append(_iou(cand, gfg))
            bfs.append(_boundary_f(cand, gfg, tol=boundary_tol))
            # --- privacy: body reveal = GT person px left uncovered ---
            leaked = np.logical_and(gfg, ~cand).sum(dtype=np.int64)
            reveals.append(float(leaked) / float(gfg.sum(dtype=np.int64)))

        # --- utility: background px needlessly greyed ---
        bg = ~gall
        if bg.any():
            overmasks.append(float(np.logical_and(cand, bg).sum(dtype=np.int64)) / float(bg.sum(dtype=np.int64)))

        # --- privacy GATE: visible faces must be fully covered ---
        fbs = np.asarray(face_boxes[i]).reshape(-1, 5) if len(face_boxes[i]) else np.zeros((0, 5))
        frame_min_cov = 1.0
        frame_has_leak = False
        for fb in fbs:
            x1, y1, x2, y2 = [int(v) for v in fb[:4]]
            # face-on-person region = face box intersected with the real person mask
            region = np.zeros(hw, bool)
            region[y1:y2, x1:x2] = True
            region &= gall
            tot = int(region.sum())
            if tot == 0:
                continue
            cov = int(np.logical_and(region, cand).sum())
            frac = cov / tot
            frame_min_cov = min(frame_min_cov, frac)
            face_px_total += tot
            face_leak_px_total += (tot - cov)
            if frac < face_leak_thresh:
                frame_has_leak = True
        face_cov_min_per_frame.append(frame_min_cov)
        if frame_has_leak:
            face_leak_frames += 1

        # --- counting ---
        gfgb = _fg_boxes(gt_boxes[i], min_person_h)
        count_err.append(abs(int(n_persons_cand[i]) - len(gfgb)))

        # --- identity: leftmost/rightmost fg person id ---
        cb = np.asarray(cand_boxes[i]).reshape(-1, 4) if len(cand_boxes[i]) else np.zeros((0, 4))
        cids = list(cand_ids[i]) if len(cand_ids[i]) else []
        fg_idx = [j for j in range(len(cb)) if (cb[j][3] - cb[j][1]) >= min_person_h]
        if fg_idx:
            fg_idx.sort(key=lambda j: 0.5 * (cb[j][0] + cb[j][2]))
            left_ids.append(cids[fg_idx[0]] if fg_idx[0] < len(cids) else -1)
            right_ids.append(cids[fg_idx[-1]] if fg_idx[-1] < len(cids) else -1)

        # --- temporal flicker ---
        if prev_cand is not None:
            flick.append(_iou(cand, prev_cand))
        prev_cand = cand

    def _sw(seq):
        seq = [s for s in seq if s != -1]
        return int(sum(1 for a, b in zip(seq, seq[1:]) if a != b))

    m = dict(
        # quality
        mask_iou=float(np.mean(ious)) if ious else None,
        boundary_f=float(np.mean(bfs)) if bfs else None,
        # privacy
        body_reveal_rate=float(np.mean(reveals)) if reveals else None,
        face_leak_frames=int(face_leak_frames),
        face_leak_px_total=int(face_leak_px_total),
        face_leak_px_frac=float(face_leak_px_total) / float(face_px_total) if face_px_total else 0.0,
        min_face_coverage=float(np.min(face_cov_min_per_frame)) if face_cov_min_per_frame else 1.0,
        privacy_gate_pass=bool(face_leak_frames == 0),
        # utility
        overmask_rate=float(np.mean(overmasks)) if overmasks else None,
        # counting
        count_mae=float(np.mean(count_err)) if count_err else None,
        # identity
        left_id_switches=_sw(left_ids),
        right_id_switches=_sw(right_ids),
        n_unique_fg_ids=len(set([s for s in (left_ids + right_ids) if s != -1])),
        # temporal
        temporal_iou=float(np.mean(flick)) if flick else None,
        frames=N,
    )
    with open(os.path.join(run_dir, "metrics.json"), "w") as fh:
        json.dump(m, fh, indent=2)
    return m
