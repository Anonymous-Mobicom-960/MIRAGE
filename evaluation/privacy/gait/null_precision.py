#!/usr/bin/env python3
"""MEASURE THE PERMUTATION NULL PRECISELY, AND TEST WHETHER IT DEPENDS ON THE ARM AT ALL.

Every table this project has published carries a DIFFERENT null per arm (gait TM2sq ranged
1.70-2.25 % at top-1; silhouette 10.63-11.93 %). Those differences were never real:

  After the labels are permuted, the probe's label lands on a RANDOM tracklet, so the chance that
  the top-ranked gallery entry carries it depends only on how many gallery entries share that label
  -- i.e. on gallery composition. It does NOT depend on where the arm put the embeddings. The true
  null is therefore one number per (threat model x rank), identical for every arm, and the observed
  spread is sampling noise from 5 reps.

That is an argument. This script MEASURES it, two ways:

  1. PRECISION -- many reps on one arm, so the null has a tight standard error rather than a 5-rep
     wobble.
  2. GEOMETRY-INDEPENDENCE -- the same high-rep null on arms whose embedding geometry is as
     different as this corpus offers: CLEAN (95 % rank-1, tight identity clusters), the shipped
     defended arm (3.5 %), and the undefended-dynamics floor. If the argument is right, all three
     agree inside their standard errors. If they do not, the argument is wrong and per-arm nulls
     must stay.

  python null_precision.py [--reps 40]
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
from casia_loader import load_tracklets                        # noqa: E402
from adversary import GaitGraphAdversary                       # noqa: E402
import protocol as P                                           # noqa: E402
import anon_adapter as A                                       # noqa: E402
import tm12_gait0807 as T                                      # noqa: E402

OUT = os.path.join(HERE, "..", "reports", "reid", "NULL_PRECISION.json")


def _log(m):
    print(f"[{time.strftime('%H:%M:%S')}] {m}", flush=True)


def null_draws(emb, reps, seed=0):
    """Return the per-rep rank-1 / rank-5 null draws (not just their mean)."""
    rng = np.random.RandomState(seed)
    keys = list(emb)
    subs = np.array([k[0] for k in keys])
    r1, r5 = [], []
    for i in range(reps):
        pm = rng.permutation(len(keys))
        e2 = {(subs[pm[j]],) + tuple(keys[j][1:]) + (j,): emb[keys[j]] for j in range(len(keys))}
        m = P.reid_metrics(e2, bootstrap=0, verbose=False)["NM#5-6"]
        r1.append(m["rank1"] * 100); r5.append(m["rank5"] * 100)
        if (i + 1) % 10 == 0:
            _log(f"      {i+1}/{reps} reps: running mean r1 {np.mean(r1):.3f}  r5 {np.mean(r5):.3f}")
    return r1, r5


def summarise(r1, r5):
    n = len(r1)
    return {"reps": n,
            "rank1_mean": float(np.mean(r1)), "rank1_sd": float(np.std(r1, ddof=1)),
            "rank1_sem": float(np.std(r1, ddof=1) / np.sqrt(n)),
            "rank5_mean": float(np.mean(r5)), "rank5_sd": float(np.std(r5, ddof=1)),
            "rank5_sem": float(np.std(r5, ddof=1) / np.sqrt(n)),
            "rank1_draws": r1, "rank5_draws": r5}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--reps", type=int, default=40)
    ap.add_argument("--out", default=OUT)
    a = ap.parse_args()

    t0 = time.time()
    tr = load_tracklets(T.CSV_TEST, split="test", min_frames=61)
    adv = GaitGraphAdversary(T.WEIGHTS, use_flip=True)
    _log(f"{len(tr)} tracklets; embedding CLEAN ...")
    clean = adv.embed_dict(tr)
    template = A.build_template(tr)   # matches tm12_gait0807.py, the harness these nulls belong to

    embs = {"clean": clean}
    for name in ("g18_armangles", "L0-floor"):
        _log(f"building {name} (per-sequence, deployed seeding) ...")
        embs[name] = adv.embed_dict(
            A.transform_v2(tr, dict(T.CONFIGS[name]), template, seed_mode="per_sequence"))

    res = {}
    for name, e in embs.items():
        _log(f"--- {name}: {a.reps}-rep null ---")
        r1, r5 = null_draws(e, a.reps)
        res[name] = summarise(r1, r5)
        s = res[name]
        _log(f"    rank1 {s['rank1_mean']:.3f} +/- {s['rank1_sem']:.3f} (sem) | "
             f"rank5 {s['rank5_mean']:.3f} +/- {s['rank5_sem']:.3f}")

    # ---- the test: do the three arms' nulls agree inside their standard errors? ----
    names = list(res)
    verdict = {}
    for rank in ("rank1", "rank5"):
        means = np.array([res[n][f"{rank}_mean"] for n in names])
        sems = np.array([res[n][f"{rank}_sem"] for n in names])
        spread = float(means.max() - means.min())
        # worst-case pairwise z between any two arms
        zmax = 0.0
        for i in range(len(names)):
            for j in range(i + 1, len(names)):
                z = abs(means[i] - means[j]) / np.sqrt(sems[i] ** 2 + sems[j] ** 2)
                zmax = max(zmax, float(z))
        pooled = float(np.concatenate([res[n][f"{rank}_draws"] for n in names]).mean())
        pooled_sem = float(np.concatenate([res[n][f"{rank}_draws"] for n in names]).std(ddof=1)
                           / np.sqrt(sum(res[n]["reps"] for n in names)))
        verdict[rank] = {"means": {n: res[n][f"{rank}_mean"] for n in names},
                         "spread_pp": spread, "max_pairwise_z": zmax,
                         "pooled_mean": pooled, "pooled_sem": pooled_sem,
                         "geometry_independent": bool(zmax < 3.0)}
        tag = ("GEOMETRY-INDEPENDENT (use one null for every arm)" if zmax < 3.0
               else "ARM-DEPENDENT (per-arm nulls required)")
        print(f"\n  {rank}: " + "  ".join(f"{n}={res[n][f'{rank}_mean']:.3f}" for n in names))
        print(f"    spread {spread:.3f} pp | max pairwise z {zmax:.2f} | "
              f"POOLED {pooled:.3f} +/- {pooled_sem:.3f}  -> {tag}")

    os.makedirs(os.path.dirname(a.out), exist_ok=True)
    json.dump({"note": __doc__.strip(), "reps_per_arm": a.reps, "arms": res, "verdict": verdict,
               "analytic_chance": {"rank1": 100.0 / 50, "rank5": 500.0 / 50},
               "wall_s": round(time.time() - t0, 1)},
              io.open(a.out, "w", encoding="utf-8"), indent=1)
    print(f"\n  wrote {a.out}  ({time.time()-t0:.0f} s)")


if __name__ == "__main__":
    main()
