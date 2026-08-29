#!/usr/bin/env python3
"""NAIVE pose-space defences, scored by the Class-3 unlearned gait adversary on CASIA-B.

WHY THESE AND NOT A PUBLISHED METHOD
No published skeleton-space anonymiser could be evaluated. ReGenHuman released neither code nor
weights. Skeleton-anonymization (AAAI 2023) released code, but its pretrained anonymisers live on a
Google Drive folder that now 404s, and the only checkpoints left in its repo are NTU-25 Shift-GCN
CLASSIFIERS which cannot ingest COCO-17 poses. Training their UNet from random init for a few
minutes and calling the output "the AAAI 2023 method" would report my training run, not their paper.

So these are NAIVE baselines, defined here and labelled as such -- never as a published method:

  noise_XX    isotropic Gaussian jitter on every joint, sigma = XX% of that tracklet's own median
              torso length (mid-shoulder to mid-hip), so the perturbation is scale-invariant
  quant_N     joint coordinates rounded to an N x N grid spanning the tracklet's bounding box
  tdown_K     temporal decimation: keep every K-th frame and hold, destroying fine dynamics while
              leaving static shape untouched

PROTOCOL IDENTITY BY IMPORT, not re-implementation: `embed`, `null_cmc` come from
`untrained_reid`, scoring from `protocol.reid_metrics` (NM#5-6, all-gallery, same-view excluded) --
the same calls that produced the published Class-3 row, so these arms sit in the same table.

  python naive_pose_baselines.py --arms noise_05,noise_10,quant_16,tdown_3 --reps 10
"""
import argparse
import io
import json
import os
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)

from casia_loader import load_tracklets            # noqa: E402
import protocol as P                               # noqa: E402
import untrained_reid as U                         # noqa: E402

DATA = os.path.join(HERE, "data")


def torso_scale(a):
    """Median mid-shoulder -> mid-hip distance; the same body-scale unit the descriptor uses."""
    xy = np.asarray(a)[:, :17, :2]
    ms = 0.5 * (xy[:, 5] + xy[:, 6])
    mh = 0.5 * (xy[:, 11] + xy[:, 12])
    d = np.linalg.norm(ms - mh, axis=1)
    d = d[np.isfinite(d) & (d > 1e-6)]
    return float(np.median(d)) if d.size else 1.0


def apply_arm(a, arm, rng):
    x = np.array(a, dtype=np.float32, copy=True)
    xy = x[:, :17, :2]
    if arm.startswith("noise_"):
        pct = float(arm.split("_")[1]) / 100.0
        s = torso_scale(a) * pct
        xy += rng.normal(0.0, s, size=xy.shape).astype(np.float32)
    elif arm.startswith("quant_"):
        n = int(arm.split("_")[1])
        lo = np.nanmin(xy, axis=(0, 1), keepdims=True)
        hi = np.nanmax(xy, axis=(0, 1), keepdims=True)
        rngxy = np.maximum(hi - lo, 1e-6)
        xy[:] = lo + np.round((xy - lo) / rngxy * (n - 1)) / (n - 1) * rngxy
    elif arm.startswith("tdown_"):
        k = int(arm.split("_")[1])
        idx = (np.arange(xy.shape[0]) // k) * k
        idx = np.clip(idx, 0, xy.shape[0] - 1)
        xy[:] = xy[idx]
    else:
        raise SystemExit(f"unknown arm {arm}")
    x[:, :17, :2] = xy
    return x


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--arms", default="noise_05,noise_10,noise_20,quant_16,tdown_3")
    ap.add_argument("--reps", type=int, default=10)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out", default=os.path.join(HERE, "..", "reports", "reid",
                                                  "NAIVE_POSE_BASELINES.json"))
    a = ap.parse_args()

    test = load_tracklets(os.path.join(DATA, "casia-b_pose_test.csv"), split="test")
    print(f"[naive] {len(test)} CASIA-B test tracklets", flush=True)

    def score(tracks, tag):
        emb = U.embed(tracks)
        m = P.reid_metrics({k + (i,): v for i, (k, v) in enumerate(emb.items())},
                           bootstrap=0, verbose=False)["NM#5-6"]
        n1, n5 = U.null_cmc(emb, a.reps)
        t1, t5 = m["rank1"] * 100, m["rank5"] * 100
        n1, n5 = n1 * 100, n5 * 100
        print(f"[naive] {tag:12s} top1 {t1:6.2f} (null {n1:5.2f}, lift {t1-n1:+7.2f}) | "
              f"top5 {t5:6.2f} (null {n5:5.2f}, lift {t5-n5:+7.2f})", flush=True)
        return dict(top1=t1, top1_null=n1, top1_lift=t1 - n1,
                    top5=t5, top5_null=n5, top5_lift=t5 - n5,
                    reps=a.reps, n_tracklets=len(tracks))

    res = {"RAW-control": score({k: np.asarray(v) for k, v in test.items()}, "RAW-control")}
    for arm in [x.strip() for x in a.arms.split(",") if x.strip()]:
        rng = np.random.RandomState(a.seed)
        res[arm] = score({k: apply_arm(v, arm, rng) for k, v in test.items()}, arm)

    os.makedirs(os.path.dirname(a.out), exist_ok=True)
    json.dump({"attacker": "UNTRAINED hand-crafted descriptor + cosine NN (Class 3), imported",
               "protocol": "protocol.reid_metrics NM#5-6, all-gallery, same-view excluded",
               "note": "NAIVE baselines defined in this file. NOT published methods. Do not "
                       "attribute them to any paper.",
               "results": res}, io.open(a.out, "w", encoding="utf-8"), indent=1)
    print(f"\nwrote {a.out}")


if __name__ == "__main__":
    main()
