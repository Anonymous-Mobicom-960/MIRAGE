#!/usr/bin/env python3
"""UNTRAINED re-identification attacker: top-1 / top-5 with a measured null, NO learning anywhere.

WHY THIS EXISTS. Owner constraint (2026-08-13): the paper reports MODEL-FREE metrics only, so the
TM3 numbers -- which retrain a ResGCN adversary -- cannot appear in it. This gives the SAME six
columns (top-1, its guess, its lift, top-5, its guess, its lift) from an attacker that fits
nothing: hand-crafted gait descriptors plus nearest-neighbour matching under the same protocol.

WHAT THE ATTACKER IS. Per tracklet, a descriptor built only from the emitted keypoints:
  * STATIC SHAPE   median bone length / spine over the 9 bone groups (scale-free), plus the
                   shoulder:spine and leg:torso ratios. This is the k-same channel.
  * SIZE           median visible height in px (the channel `seeded_global_scale_max` randomises).
  * DYNAMICS       per-joint speed and jerk percentiles, normalised by visible height, plus the
                   dominant limb-swing frequency from an FFT of the wrist/ankle offset tracks.
                   This is the channel the time-warps attack.
  * PHASE          cross-correlation lags between the four limb chains -- the inter-limb phase
                   relationships `_limb_phase_offset` / `_limb_phase_warp` attack.
Each block is z-scored across the corpus (so no block dominates by unit), concatenated, and
matched by cosine distance. Nothing is trained: there are no weights, no fitting, no data split.

PROTOCOL parity with the trained path: the same `protocol.reid_metrics` cell (NM#5-6, all-gallery,
same-view exclusion) and the same label-permutation null (`reps` draws), so top-1/top-5 and the
nulls are directly comparable in KIND to the TM3-side numbers in §A.2w -- though NOT in value,
because a hand-crafted attacker is weaker than a retrained network by construction. The lift over
the arm's OWN measured null is the quantity to compare across arms.

  python untrained_reid.py --configs RAW-control,qsize_m10_smooth,r1,r2,r3 --reps 5
"""
import argparse
import io
import json
import os
import sys
import time

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
from casia_loader import load_tracklets            # noqa: E402
import protocol as P                               # noqa: E402
import anon_adapter as A                           # noqa: E402
import tm3_retrain as R                            # noqa: E402

DATA = os.path.join(HERE, "data")
_GROUPS = [("thighL", 11, 13), ("shinL", 13, 15), ("thighR", 12, 14), ("shinR", 14, 16),
           ("uarmL", 5, 7), ("farmL", 7, 9), ("uarmR", 6, 8), ("farmR", 8, 10), ("clav", 5, 6)]
_CHAINS = [(5, 9), (6, 10), (11, 15), (12, 16)]     # shoulder->wrist, hip->ankle, per side


def _log(m):
    print(f"[{time.strftime('%H:%M:%S')}] {m}", flush=True)


def _visible_h(xy):
    live = np.abs(xy).sum(2) > 0
    hs = []
    for t in range(xy.shape[0]):
        y = xy[t][live[t]][:, 1]
        if y.size >= 2:
            hs.append(float(y.max() - y.min()))
    return float(np.median(hs)) if hs else 1.0


def descriptor(xy):
    """(T,17,2) -> 1-D feature vector. Hand-crafted; nothing fitted."""
    h = max(_visible_h(xy), 1.0)
    ms = 0.5 * (xy[:, 5] + xy[:, 6])
    mh = 0.5 * (xy[:, 11] + xy[:, 12])
    spine = np.linalg.norm(ms - mh, axis=1)
    spine[spine < 1] = np.nan

    shape = []
    for _n, a, b in _GROUPS:
        A_, B_ = xy[:, a], xy[:, b]
        ok = (np.abs(A_).sum(1) > 0) & (np.abs(B_).sum(1) > 0)
        d = np.linalg.norm(B_ - A_, axis=1)
        d[~ok] = np.nan
        shape.append(np.nanmedian(d / spine) if np.isfinite(d).any() else 0.0)

    v = np.diff(xy, axis=0)
    sp = np.linalg.norm(v, axis=2) / h
    ac = np.linalg.norm(np.diff(v, axis=0), axis=2) / h
    dyn = [np.nanmedian(sp), np.nanpercentile(sp, 90), np.nanmedian(ac),
           np.nanpercentile(ac, 90), np.nanstd(sp)]

    # dominant limb-swing frequency + inter-chain phase lag, from the offset tracks
    freqs, lags, tracks = [], [], []
    for anchor, tip in _CHAINS:
        off = xy[:, tip] - xy[:, anchor]
        s = off[:, 0] - np.nanmean(off[:, 0])
        tracks.append(s)
        if s.size >= 8 and np.isfinite(s).all() and np.std(s) > 1e-6:
            mag = np.abs(np.fft.rfft(s - s.mean()))
            mag[0] = 0.0
            freqs.append(float(np.argmax(mag)) / max(len(s), 1))
        else:
            freqs.append(0.0)
    for i in range(len(tracks)):
        for j in range(i + 1, len(tracks)):
            a_, b_ = tracks[i], tracks[j]
            if a_.size < 8 or np.std(a_) < 1e-6 or np.std(b_) < 1e-6:
                lags.append(0.0)
                continue
            a0 = (a_ - a_.mean()) / (np.std(a_) + 1e-9)
            b0 = (b_ - b_.mean()) / (np.std(b_) + 1e-9)
            c = np.correlate(a0, b0, mode="same") / len(a0)
            lags.append(float(np.argmax(c) - len(a0) // 2) / len(a0))

    f = np.array(shape + [h] + dyn + freqs + lags, dtype=np.float64)
    return np.nan_to_num(f, nan=0.0, posinf=0.0, neginf=0.0)


def embed(tracks):
    keys = list(tracks)
    F = np.stack([descriptor(np.asarray(tracks[k])[:, :17, :2]) for k in keys])
    mu, sd = F.mean(0), F.std(0)
    sd[sd < 1e-9] = 1.0
    F = (F - mu) / sd                                     # z-score per feature, corpus-wide
    F /= (np.linalg.norm(F, axis=1, keepdims=True) + 1e-9)
    return {k: F[i] for i, k in enumerate(keys)}


def null_cmc(emb, reps, seed=0):
    rng = np.random.RandomState(seed)
    keys = list(emb)
    subs = np.array([k[0] for k in keys])
    r1, r5 = [], []
    for _ in range(reps):
        pm = rng.permutation(len(keys))
        e2 = {(subs[pm[i]],) + tuple(keys[i][1:]) + (i,): emb[keys[i]] for i in range(len(keys))}
        m = P.reid_metrics(e2, bootstrap=0, verbose=False)["NM#5-6"]
        r1.append(m["rank1"]); r5.append(m["rank5"])
    return float(np.mean(r1)), float(np.mean(r5))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--configs", default="RAW-control,qsize_m10_smooth,r1,r2,r3")
    ap.add_argument("--reps", type=int, default=5)
    ap.add_argument("--out", default=os.path.join(HERE, "..", "reports", "reid",
                                                  "UNTRAINED_reid.json"))
    a = ap.parse_args()

    test = load_tracklets(os.path.join(DATA, "casia-b_pose_test.csv"), split="test")
    train = load_tracklets(os.path.join(DATA, "casia-b_pose_train_valid.csv"), split="train_valid")
    tmpl = A.build_template(train)                 # same public TRAIN-id template tm3_retrain uses
    _log(f"{len(test)} test tracklets; template from {len(train)} train tracklets")

    rows, results = [], {}
    for name in [c.strip() for c in a.configs.split(",") if c.strip()]:
        cfg = R.CONFIGS[name]
        _log(f"--- {name} ---")
        if cfg.get("__raw__"):
            anon = {k: np.asarray(v) for k, v in test.items()}
        else:
            c = dict(cfg)
            c.pop("__raw__", None)
            sf = c.pop("scale_from", None)
            for k in ("strip_conf", "binarize_conf", "deployment_postprocess",
                      "eval_seed_map_master"):
                c.pop(k, None)
            anon = A.transform_v2(test, c, tmpl, seed_mode="per_sequence", scale_from=sf)
        emb = embed(anon)
        m = P.reid_metrics({k + (i,): v for i, (k, v) in enumerate(emb.items())},
                           bootstrap=0, verbose=False)["NM#5-6"]
        n1, n5 = null_cmc(emb, a.reps)
        t1, t5 = m["rank1"] * 100, m["rank5"] * 100
        n1, n5 = n1 * 100, n5 * 100
        results[name] = dict(top1=t1, top1_null=n1, top1_lift=t1 - n1,
                             top5=t5, top5_null=n5, top5_lift=t5 - n5, reps=a.reps,
                             n_tracklets=len(anon))
        rows.append((name, t1, n1, t1 - n1, t5, n5, t5 - n5))
        _log(f"    top1 {t1:6.2f} (null {n1:5.2f}, lift {t1-n1:+7.2f}) | "
             f"top5 {t5:6.2f} (null {n5:5.2f}, lift {t5-n5:+7.2f})")

    print()
    print(f"{'arm':22s} | {'top1':>7s} {'guess':>6s} {'lift':>8s} | "
          f"{'top5':>7s} {'guess':>6s} {'lift':>8s}")
    for r in rows:
        print(f"{r[0]:22s} | {r[1]:7.2f} {r[2]:6.2f} {r[3]:+8.2f} | "
              f"{r[4]:7.2f} {r[5]:6.2f} {r[6]:+8.2f}")
    os.makedirs(os.path.dirname(a.out), exist_ok=True)
    json.dump({"attacker": "UNTRAINED hand-crafted descriptor + cosine NN; nothing fitted",
               "protocol": "protocol.reid_metrics NM#5-6, all-gallery, same-view excluded",
               "results": results},
              io.open(a.out, "w", encoding="utf-8"), indent=1)
    print(f"\nwrote {a.out}")


if __name__ == "__main__":
    main()
