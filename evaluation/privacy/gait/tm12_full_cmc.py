#!/usr/bin/env python3
"""TOP-1 + TOP-5 + A MEASURED NULL FOR ALL THREE FROZEN THREAT MODELS, ON ONE PROTOCOL.

What was missing before this script (see §A.2m-2b-2 / §A.2q):
  * TM1 had a rank-1 point estimate and NOTHING else -- no null, no top-5.
  * TM2id likewise.
  * TM2sq had a published top-5 but a null for only ONE arm (g18), and its top-1 and top-5
    columns came from DIFFERENT gallery constructions (§B.21 violation, inherited from §A.2j-2).

Every number here comes from `protocol.reid_metrics` -- all-gallery, same-view exclusion, cell
NM#5-6 -- so within this file top-1 and top-5 are the SAME experiment and may be read together.
Because of that, values will NOT equal §A.2j-2's canonical-protocol figures (there RAW reads
87.64 %, here 95.39 %; g18 TM2sq 3.64 vs 3.51). That is the point: one protocol, internally
consistent, rather than two spliced.

Nulls are measured per arm AND per threat model -- they are not interchangeable (§A.2q measured a
0.74 pp spread between clean and g18 on the same protocol).

ARMS RUN IN PRIORITY ORDER (shipped first) so an early stop still leaves the decision-relevant rows.
Results are flushed to disk after EVERY arm.

  python tm12_full_cmc.py [--configs a,b,c] [--reps 5]
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

OUT = os.path.join(HERE, "..", "reports", "reid", "TM12_full_cmc.json")

# shipped first, then the near-neighbours, then the rejected head-anchor arms, then the floor
PRIORITY = ["g18_armangles", "shipped", "g10_boost", "g14_mirror", "g16_outward",
            "g12_headanchor", "g19_armangles_headanchor", "L0-floor"]


def _log(m):
    print(f"[{time.strftime('%H:%M:%S')}] {m}", flush=True)


def cmc(emb):
    r = P.reid_metrics(emb, bootstrap=0, verbose=False)["NM#5-6"]
    return {"rank1": r["rank1"], "rank5": r["rank5"], "n_probes": r["n_probes"]}


def null_cmc(emb, reps=5, seed=0):
    """Label-permutation null for BOTH ranks, same rekey as tm12_gait0807.permuted_null()."""
    rng = np.random.RandomState(seed)
    keys = list(emb)
    subs = np.array([k[0] for k in keys])
    r1, r5 = [], []
    for _ in range(reps):
        pm = rng.permutation(len(keys))
        e2 = {(subs[pm[i]],) + tuple(keys[i][1:]) + (i,): emb[keys[i]] for i in range(len(keys))}
        m = P.reid_metrics(e2, bootstrap=0, verbose=False)["NM#5-6"]
        r1.append(m["rank1"]); r5.append(m["rank5"])
    return {"rank1": float(np.mean(r1)), "rank1_sd": float(np.std(r1)),
            "rank5": float(np.mean(r5)), "rank5_sd": float(np.std(r5)), "reps": reps}


def scored(emb, reps):
    a, n = cmc(emb), null_cmc(emb, reps)
    return {"top1": a["rank1"] * 100, "top1_null": n["rank1"] * 100,
            "top1_lift": (a["rank1"] - n["rank1"]) * 100,
            "top5": a["rank5"] * 100, "top5_null": n["rank5"] * 100,
            "top5_lift": (a["rank5"] - n["rank5"]) * 100,
            "n_probes": a["n_probes"], "null_sd": [n["rank1_sd"] * 100, n["rank5_sd"] * 100]}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--configs", default=",".join(PRIORITY))
    ap.add_argument("--reps", type=int, default=5)
    ap.add_argument("--out", default=OUT)
    a = ap.parse_args()
    want = [c for c in a.configs.split(",") if c.strip() in T.CONFIGS]

    t0 = time.time()
    tr = load_tracklets(T.CSV_TEST, split="test", min_frames=61)
    adv = GaitGraphAdversary(T.WEIGHTS, use_flip=True)
    _log(f"{len(tr)} tracklets / {len({k[0] for k in tr})} subjects; embedding CLEAN ...")
    clean = adv.embed_dict(tr)
    template = A.build_template(tr)

    res = {"_RAW_control": scored(clean, a.reps)}
    r = res["_RAW_control"]
    _log(f"RAW positive control: top1 {r['top1']:.2f} (null {r['top1_null']:.2f}, "
         f"lift {r['top1_lift']:+.2f})  top5 {r['top5']:.2f} (null {r['top5_null']:.2f}, "
         f"lift {r['top5_lift']:+.2f})")

    def flush():
        os.makedirs(os.path.dirname(a.out), exist_ok=True)
        json.dump({"note": __doc__.strip(), "protocol":
                   "protocol.reid_metrics, all-gallery, same-view exclusion, cell NM#5-6, "
                   "bootstrap=0; nulls by label permutation, per arm AND per threat model",
                   "n_test_tracklets": len(tr), "reps": a.reps, "results": res,
                   "wall_s": round(time.time() - t0, 1)},
                  io.open(a.out, "w", encoding="utf-8"), indent=1)

    flush()
    for name in want:
        cfg = T.CONFIGS[name]
        _log(f"--- {name} --- transforming + embedding (2 seed modes) ...")
        aID = A.transform_v2(tr, dict(cfg), template, seed_mode="per_identity")
        aSQ = A.transform_v2(tr, dict(cfg), template, seed_mode="per_sequence")
        eID, eSQ = adv.embed_dict(aID), adv.embed_dict(aSQ)
        tm1 = {k: (clean[k] if T.is_gallery(k) else eID[k]) for k in clean}
        _log(f"    scoring TM1 / TM2id / TM2sq with {a.reps}-rep nulls each ...")
        res[name] = {"TM1": scored(tm1, a.reps), "TM2id": scored(eID, a.reps),
                     "TM2sq": scored(eSQ, a.reps)}
        for tm in ("TM1", "TM2id", "TM2sq"):
            v = res[name][tm]
            _log(f"    {tm:<6} top1 {v['top1']:6.2f} (null {v['top1_null']:5.2f}, "
                 f"lift {v['top1_lift']:+6.2f})   top5 {v['top5']:6.2f} "
                 f"(null {v['top5_null']:5.2f}, lift {v['top5_lift']:+6.2f})")
        flush()
        _log(f"    flushed; {time.time() - t0:.0f} s elapsed")

    # -------- report --------
    print(f"\n{'arm':<26} {'TM':<6} | {'top1':>7} {'null':>7} {'lift':>7} | "
          f"{'top5':>7} {'null':>7} {'lift':>7}")
    v = res["_RAW_control"]
    print(f"{'RAW positive control':<26} {'--':<6} | {v['top1']:7.2f} {v['top1_null']:7.2f} "
          f"{v['top1_lift']:+7.2f} | {v['top5']:7.2f} {v['top5_null']:7.2f} {v['top5_lift']:+7.2f}")
    for n in want:
        if n not in res:
            continue
        for tm in ("TM1", "TM2id", "TM2sq"):
            v = res[n][tm]
            print(f"{n:<26} {tm:<6} | {v['top1']:7.2f} {v['top1_null']:7.2f} {v['top1_lift']:+7.2f} "
                  f"| {v['top5']:7.2f} {v['top5_null']:7.2f} {v['top5_lift']:+7.2f}")
    flush()
    print(f"\n  wrote {a.out}   ({time.time() - t0:.0f} s)")


if __name__ == "__main__":
    main()
