#!/usr/bin/env python3
"""
check_sidecars.py - turn the sidecar agreement claim into a reproducible artifact.
==================================================================================
`ALPHA_MATTE.md` and `PHONE_APP_STATUS.md` state that the person MASK (derived from
`silhouette.mp4` by fill-colour detection) and the character ALPHA (derived from
`character.mp4` by yolo11x-seg) "agree at IoU 0.83-0.95 - the strongest correctness check
available here". That claim has **no artifact on disk**: no JSON, no log, no way to re-run it.
This script produces one, so it can act as a regression gate rather than folklore.

    python check_sidecars.py --mask input/sets/mask_A.mp4 \
                             --alpha input/sets/synthetic_alpha_p1_A.mp4 --tag setA

WHY THE AGREEMENT IS MEANINGFUL: the two videos come from completely independent sources - one
from where Tier-1 painted grey, the other from what a segmentation model finds in the generated
character. They share no code path. High IoU therefore cross-validates BOTH; a divergence means
at least one is wrong, and the per-frame series says exactly where.

WHAT IS ALSO CHECKED (the failure modes that actually bit this project):
  * NOTCH / see-through - matte pixels that should be person but read as background. Session 6
    measured 9.57 % on the derived path and found 100 % of it was BOUNDARY-CONNECTED, not
    interior holes, so `fillInteriorHoles` was never the lever. Reported both ways here.
  * MISSING DETECTIONS - frames where the alpha is empty ("0 of 153 missing" was claimed).
  * FRAME/GEOMETRY PARITY - differing length, size or fps between the two sidecars silently
    misaligns everything downstream.
"""
import argparse
import json
import os
import sys

import cv2
import numpy as np


def read_binary(path, thresh=127, max_frames=0):
    cap = cv2.VideoCapture(path)
    if not cap.isOpened():
        raise SystemExit(f"cannot open {path}")
    fps = cap.get(cv2.CAP_PROP_FPS)
    out = []
    while True:
        ok, f = cap.read()
        if not ok:
            break
        g = f[..., 0] if f.ndim == 3 else f
        out.append((g > thresh).astype(np.uint8))
        if max_frames and len(out) >= max_frames:
            break
    cap.release()
    if not out:
        raise SystemExit(f"no frames decoded from {path}")
    return np.stack(out), float(fps or 0)


def boundary_connected_share(m):
    """Of the background pixels INSIDE the matte's bounding box, how many touch the border of
    that box (i.e. are notches eating in from the edge) vs are fully enclosed holes?"""
    ys, xs = np.nonzero(m)
    if not len(ys):
        return None, None
    y0, y1, x0, x1 = ys.min(), ys.max() + 1, xs.min(), xs.max() + 1
    sub = m[y0:y1, x0:x1]
    bg = (sub == 0).astype(np.uint8)
    if not bg.any():
        return 0.0, 0.0
    n, lab = cv2.connectedComponents(bg, connectivity=4)
    border = set(lab[0, :]) | set(lab[-1, :]) | set(lab[:, 0]) | set(lab[:, -1])
    border.discard(0)
    tot = int(bg.sum())
    bc = int(np.isin(lab, list(border)).sum()) if border else 0
    return bc / tot, (tot - bc) / tot


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--mask", required=True, help="mask.mp4 (person silhouette from Tier-1/2A)")
    ap.add_argument("--alpha", required=True, nargs="+",
                    help="synthetic_alpha_p*.mp4 (character matte). Pass ALL persons' alphas for a "
                         "multi-person clip - mask.mp4 is the UNION of everyone, so comparing it "
                         "against one person's alpha understates agreement (set B: 0.46 vs 0.91).")
    ap.add_argument("--tag", default="set")
    ap.add_argument("--max-frames", type=int, default=0)
    ap.add_argument("--out", default=os.path.join(os.path.dirname(os.path.abspath(__file__)),
                                                  "reports"))
    a = ap.parse_args()

    M, fps_m = read_binary(a.mask, max_frames=a.max_frames)
    print(f"[sc] mask  {a.mask}: {M.shape[0]} frames {M.shape[2]}x{M.shape[1]} @{fps_m:g}")
    parts = []
    for ap_path in a.alpha:
        P, fps_a = read_binary(ap_path, max_frames=a.max_frames)
        print(f"[sc] alpha {ap_path}: {P.shape[0]} frames {P.shape[2]}x{P.shape[1]} @{fps_a:g}")
        parts.append(P)
    if len(parts) > 1:                       # mask.mp4 is the UNION over all people
        k = min(p.shape[0] for p in parts)
        A = parts[0][:k].copy()
        for P in parts[1:]:
            A |= P[:k]
        print(f"[sc] unioned {len(parts)} person alphas over {k} frames")
    else:
        A = parts[0]

    parity = {"frames_equal": M.shape[0] == A.shape[0], "size_equal": M.shape[1:] == A.shape[1:],
              "fps_equal": abs(fps_m - fps_a) < 0.51}
    if not parity["size_equal"]:
        print(f"[sc] !! SIZE MISMATCH - resizing alpha to the mask grid for comparison")
        A = np.stack([cv2.resize(f, (M.shape[2], M.shape[1]),
                                 interpolation=cv2.INTER_NEAREST) for f in A])
    n = min(M.shape[0], A.shape[0])
    if M.shape[0] != A.shape[0]:
        print(f"[sc] !! FRAME-COUNT MISMATCH {M.shape[0]} vs {A.shape[0]} - comparing first {n}")
    M, A = M[:n], A[:n]

    ious, cov_m, cov_a, notch_bc, notch_int = [], [], [], [], []
    empty_alpha = empty_mask = 0
    for i in range(n):
        m, al = M[i] > 0, A[i] > 0
        u = int((m | al).sum())
        ious.append(int((m & al).sum()) / u if u else 1.0)
        cov_m.append(float(m.mean())); cov_a.append(float(al.mean()))
        if not al.any():
            empty_alpha += 1
        if not m.any():
            empty_mask += 1
        bc, it = boundary_connected_share(al)
        if bc is not None:
            notch_bc.append(bc); notch_int.append(it)
    io = np.array(ious)

    rep = {"tag": a.tag, "mask": os.path.abspath(a.mask), "alpha": [os.path.abspath(x) for x in a.alpha],
           "alpha_union_of_n_persons": len(a.alpha),
           "frames_compared": n, "parity": parity,
           "iou": {"mean": float(io.mean()), "median": float(np.median(io)),
                   "min": float(io.min()), "max": float(io.max()),
                   "p05": float(np.percentile(io, 5)), "p95": float(np.percentile(io, 95))},
           "coverage": {"mask_mean_pct": float(np.mean(cov_m) * 100),
                        "alpha_mean_pct": float(np.mean(cov_a) * 100)},
           "empty_alpha_frames": empty_alpha, "empty_mask_frames": empty_mask,
           "alpha_background_inside_bbox": {
               "boundary_connected_mean_pct": float(np.mean(notch_bc) * 100) if notch_bc else None,
               "interior_hole_mean_pct": float(np.mean(notch_int) * 100) if notch_int else None},
           "worst_frames_by_iou": [int(x) for x in np.argsort(io)[:5]]}

    print(f"\n  IoU(mask, alpha)  mean={io.mean():.4f}  median={np.median(io):.4f}  "
          f"range=[{io.min():.4f}, {io.max():.4f}]  p05={np.percentile(io,5):.4f}")
    print(f"  coverage          mask {np.mean(cov_m)*100:.2f}%   alpha {np.mean(cov_a)*100:.2f}%")
    print(f"  empty frames      alpha {empty_alpha}/{n}   mask {empty_mask}/{n}")
    if notch_bc:
        print(f"  alpha bg inside bbox   boundary-connected {np.mean(notch_bc)*100:.2f}%  "
              f"interior holes {np.mean(notch_int)*100:.2f}%")
        print("     (session 6 found 100% of see-through was boundary-connected - "
              "fillInteriorHoles was never the lever)")
    print(f"  worst frames by IoU: {rep['worst_frames_by_iou']}")

    os.makedirs(a.out, exist_ok=True)
    p = os.path.join(a.out, f"SIDECAR_CHECK_{a.tag}.json")
    json.dump(rep, open(p, "w"), indent=2)
    print(f"\n[sc] wrote {p}")


if __name__ == "__main__":
    main()
