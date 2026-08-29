#!/usr/bin/env python3
"""
measure_local_masks.py -- what the SHIPPED mask mitigation actually does to REAL MIRAGE masks.
==============================================================================================
This is NOT a re-identification result and never becomes one: it measures the PERTURBATION the
mitigation applies (coverage, area growth, IoU, contour complexity), not what an attacker can
recover. The re-ID number for this channel needs a validated silhouette adversary and the
CASIA-B silhouette dataset -- see README.md and MASTER_EVAL_LEDGER §P.

It is still worth measuring, because it bounds two things we DO claim:
  * §2 coverage: the emitted mask must be a SUPERSET of the detected mask on every frame
    (violations must be exactly 0 -- the hard guarantee mask_mitigate() is built around);
  * cost: how much extra gray the mitigation paints, which is what shows up as ghosting behind
    fast motion (ATTENTION_AND_RISKS §2.3) and what Tier-2's inpainting has to fill.

Input: a mask.mkv produced by a Tier-1 run with config.MASK_ANON_ON=False (the RAW arm). The
mitigated arm is then produced here by the REAL mirage_tier1.mask_mitigate(), so both arms come
from one segmentation pass and differ only by the shipped transform.

  python measure_local_masks.py --raw-mask out_rawmask/mask.mkv --tag 2person_1264
"""
import argparse
import json
import os
import sys
import time

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
sys.path.insert(0, os.path.dirname(HERE))

import silhouette_harness as H                                  # noqa: E402

OUT_DIR = os.path.join(H.LAB, "reports", "reid")


def read_mask_video(path, max_frames=0):
    """mask.mkv (lossless FFV1, single channel) -> (T,H,W) uint8 {0,1}."""
    import cv2
    cap = cv2.VideoCapture(path)
    frames = []
    while True:
        ok, f = cap.read()
        if not ok:
            break
        if f.ndim == 3:
            f = f[..., 0]
        frames.append((f >= 128).astype(np.uint8))
        if max_frames and len(frames) >= max_frames:
            break
    cap.release()
    if not frames:
        raise RuntimeError(f"no frames decoded from {path}")
    return np.stack(frames)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--raw-mask", required=True,
                    help="mask.mkv from a Tier-1 run with MASK_ANON_ON=False")
    ap.add_argument("--tag", default="local")
    ap.add_argument("--max-frames", type=int, default=0)
    a = ap.parse_args()

    raw = read_mask_video(a.raw_mask, a.max_frames)
    T, Hh, Ww = raw.shape
    print(f"[mask] {a.raw_mask}: {T} frames {Ww}x{Hh}, "
          f"mean person-pixel fraction {float(raw.mean()):.4f}")

    t0 = time.perf_counter()
    mit = H.apply_mask_mitigation(raw)
    ms_per_frame = (time.perf_counter() - t0) * 1000.0 / T
    print(f"[mask] mitigation (win={H.MASK_WIN}, eps={H.MASK_EPS}): "
          f"{ms_per_frame:.2f} ms/frame at {Ww}x{Hh} (laptop reference)")

    cov = H.coverage_report({"clip": raw}, {"clip": mit})
    cov["ms_per_frame_laptop"] = ms_per_frame
    cov["resolution"] = [int(Ww), int(Hh)]
    cov["source"] = os.path.abspath(a.raw_mask)

    print(f"[mask] §2 superset violations: {cov['superset_violations_px']} px "
          f"over {cov['frames']} frames (MUST be 0)")
    ar, iou = cov["area_ratio_mitigated_over_raw"], cov["iou_rawbin_vs_mitigated"]
    vr, vm = cov["contour_vertices_raw"], cov["contour_vertices_mitigated"]
    print(f"[mask] area ratio mitigated/raw  mean={ar['mean']:.4f} p50={ar['p50']:.4f} "
          f"p95={ar['p95']:.4f}   (>1 = extra gray painted)")
    print(f"[mask] IoU raw vs mitigated      mean={iou['mean']:.4f} p50={iou['p50']:.4f}")
    print(f"[mask] contour vertices          {vr['p50']:.0f} -> {vm['p50']:.0f} "
          f"(p50; shape-detail reduction)")
    assert cov["superset_violations_px"] == 0, "§2 VIOLATION on real masks"

    os.makedirs(OUT_DIR, exist_ok=True)
    path = os.path.join(OUT_DIR, f"MASK_MITIGATION_shape_{a.tag}.json")
    with open(path, "w") as f:
        json.dump(cov, f, indent=2)
    print(f"\n[mask] wrote {path}")
    print("[mask] NOTE: shape-perturbation statistics only -- what the mitigation DOES to the "
          "mask, not what an attacker recovers. The re-ID number for this channel is measured "
          "separately by run_silhouette_reid.py (ledger §A.6b: 96.55 -> 62.27 % NM rank-1).")


if __name__ == "__main__":
    main()
