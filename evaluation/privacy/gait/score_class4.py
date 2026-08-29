#!/usr/bin/env python3
"""Score the ADAPTIVE (Class 4) adversary checkpoints: rank-1/5 with a MEASURED null.

`tm3_retrain.py` records only the canonical CASIA-B rank-1 during training. The paper's Class 4
table instead reports `protocol.reid_metrics` (all-gallery, same-view exclusion, cell NM#5-6)
with a label-permutation null, so this script re-scores the trained adversaries: nothing is
retrained. Adapted from the internal `tm3_eval_cmc.py` that produced the published numbers; the
embedding path, the template rule and the scoring protocol are unchanged, and the internal
ledger-constant gates are replaced by self-gates against the values `tm3_retrain.py` recorded
for THIS run's own checkpoints.

Per config: rebuild the ResGCN, load each seed's checkpoint, re-derive the anonymised test set
from the cfg STORED IN THE CHECKPOINT (so the arm cannot drift from the one that was trained
against), embed, then score reid_metrics plus the permutation null.

THE TEMPLATE MUST COME FROM THE TRAIN IDS. `tm3_retrain.py` builds the public anthropometric
template from the train split; a template built from the test split produces a DIFFERENT
anonymised test set from the one the checkpoints were trained against. This drifted a seed by
0.17 pp when it was once done wrong, and the gate caught it.

GATE: each seed's canonical rank-1 must reproduce the value `tm3_retrain.py` wrote into
reports/reid/TM3_<config>_<tag>_s<seed>.json for that same checkpoint. A drift means the
anonymisation or the eval path changed and the top-5 would be measured on a different quantity.
Gate failures are reported per seed and the run exits non-zero.

  python score_class4.py --configs RAW-control,qsize_m10_smooth,e2 --tag repro --seeds 0,1,2
"""
import argparse
import io
import json
import os
import sys
import time

import numpy as np
import torch

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
sys.path.insert(0, os.path.join(HERE, "vendor", "GaitGraph", "src"))
from casia_loader import load_tracklets                        # noqa: E402
import protocol as P                                           # noqa: E402
import anon_adapter as A                                       # noqa: E402
import tm3_retrain as R                                        # noqa: E402
from datasets.graph import Graph                               # noqa: E402

DATA = os.path.join(HERE, "data")
CSV_TEST = os.path.join(DATA, "casia-b_pose_test.csv")
OUT_DIR = os.path.join(HERE, "..", "reports", "reid")   # where tm3_retrain.py writes its records
GATE_TOL = 0.15  # pp; deterministic given the same checkpoint + the same anonymised test set


def _log(m):
    print(f"[{time.strftime('%H:%M:%S')}] {m}", flush=True)


def null_cmc(emb, reps=5, seed=0):
    rng = np.random.RandomState(seed)
    keys = list(emb)
    subs = np.array([k[0] for k in keys])
    r1, r5 = [], []
    for _ in range(reps):
        pm = rng.permutation(len(keys))
        e2 = {(subs[pm[i]],) + tuple(keys[i][1:]) + (i,): emb[keys[i]] for i in range(len(keys))}
        m = P.reid_metrics(e2, bootstrap=0, verbose=False)["NM#5-6"]
        r1.append(m["rank1"]); r5.append(m["rank5"])
    return {"rank1": float(np.mean(r1)), "rank5": float(np.mean(r5)),
            "rank1_sd": float(np.std(r1)), "rank5_sd": float(np.std(r5)), "reps": reps}


def recorded_rank1(config, tag, seed):
    """The canonical rank-1 tm3_retrain.py recorded for this exact checkpoint, if present."""
    p = os.path.join(OUT_DIR, f"TM3_{config}_{tag}_s{seed}.json")
    if os.path.exists(p):
        try:
            return float(json.load(open(p))["final"]["NM"]) * 100.0
        except Exception:
            return None
    return None


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--configs", default="RAW-control,qsize_m10_smooth,e2")
    ap.add_argument("--tag", default="repro",
                    help="the --tag the checkpoints were trained with")
    ap.add_argument("--ckpt-pattern", default="tm3_adv_{config}_{tag}_s{seed}.pth",
                    help="checkpoint filename pattern under data/")
    ap.add_argument("--reps", type=int, default=5)
    ap.add_argument("--seeds", default="0,1,2")
    ap.add_argument("--out", default=os.path.join(HERE, "..", "reports", "reid", "CLASS4.json"))
    a = ap.parse_args()
    want = [c.strip() for c in a.configs.split(",") if c.strip()]
    seeds = [int(s) for s in a.seeds.split(",")]

    t0 = time.time()
    device = "cuda" if torch.cuda.is_available() else "cpu"
    graph = Graph("coco")
    test_tr = load_tracklets(CSV_TEST, split="test", min_frames=61)
    _log("loading TRAIN split to build the public template (as tm3_retrain does) ...")
    train_tr = load_tracklets(os.path.join(DATA, "casia-b_pose_train_valid.csv"),
                              split="train_valid", min_frames=61)
    P.assert_identity_disjoint(train_tr, test_tr)
    template = A.build_template(train_tr)
    _log(f"{len(test_tr)} test tracklets / {len({k[0] for k in test_tr})} subjects; "
         f"template from {len({k[0] for k in train_tr})} TRAIN ids; device={device}")

    res, gates = {}, []
    for name in want:
        anon_test = None
        rows = []
        for si in seeds:
            ck_path = os.path.join(DATA, a.ckpt_pattern.format(config=name, tag=a.tag, seed=si))
            if not os.path.exists(ck_path):
                _log(f"  !! missing checkpoint {os.path.basename(ck_path)}, skipping")
                continue
            ck = torch.load(ck_path, map_location=device, weights_only=False)
            cfg = ck["cfg"]
            if anon_test is None:
                if cfg.get("__raw__"):
                    _log(f"--- {name} --- RAW positive control, no anonymisation")
                    anon_test = test_tr
                else:
                    _log(f"--- {name} --- anonymising test set once "
                         f"(per-sequence, scale_from={cfg.get('scale_from')!r}) ...")
                    anon_test = A.transform_v2(test_tr, cfg, template, seed_mode="per_sequence",
                                               scale_from=cfg.get("scale_from"))
            model = R.build_model(graph, device)
            model.load_state_dict(ck["model"])
            emb = R.embed_eval(model, anon_test, graph, device)
            m = P.reid_metrics(emb, bootstrap=0, verbose=False)["NM#5-6"]
            nul = null_cmc(emb, a.reps)
            # gate on the CANONICAL rank-1, the quantity tm3_retrain recorded for this checkpoint
            canon = P.canonical_casia_b_rank1(emb)["NM#5-6"] * 100
            rec = recorded_rank1(name, a.tag, si)
            if rec is not None:
                d = abs(canon - rec)
                ok = bool(d <= GATE_TOL)
                gates.append({"arm": name, "seed": si, "canonical_rank1": canon,
                              "recorded": rec, "delta": d, "pass": ok})
                gtxt = f"recorded {rec:.2f}, d={d:.2f} {'PASS' if ok else 'FAIL'}"
            else:
                gates.append({"arm": name, "seed": si, "canonical_rank1": canon,
                              "recorded": None, "delta": None, "pass": True})
                gtxt = "no recorded value found (gate skipped)"
            rows.append({"seed": si, "top1": m["rank1"] * 100, "top5": m["rank5"] * 100,
                         "top1_null": nul["rank1"] * 100, "top5_null": nul["rank5"] * 100,
                         "canonical_rank1": canon})
            _log(f"    s{si}: canonical r1 {canon:6.2f} ({gtxt}) | "
                 f"reid_metrics top1 {m['rank1']*100:6.2f} (null {nul['rank1']*100:5.2f})  "
                 f"top5 {m['rank5']*100:6.2f} (null {nul['rank5']*100:5.2f})")
            del model
            if device == "cuda":
                torch.cuda.empty_cache()
        if rows:
            agg = {k: float(np.mean([r[k] for r in rows])) for k in
                   ("top1", "top5", "top1_null", "top5_null")}
            agg["top1_sd"] = float(np.std([r["top1"] for r in rows]))
            agg["top5_sd"] = float(np.std([r["top5"] for r in rows]))
            agg["top1_lift"] = agg["top1"] - agg["top1_null"]
            agg["top5_lift"] = agg["top5"] - agg["top5_null"]
            res[name] = {"mean": agg, "seeds": rows}
            _log(f"    MEAN top1 {agg['top1']:.2f}+/-{agg['top1_sd']:.2f} "
                 f"(null {agg['top1_null']:.2f}, lift {agg['top1_lift']:+.2f}) | "
                 f"top5 {agg['top5']:.2f}+/-{agg['top5_sd']:.2f} "
                 f"(null {agg['top5_null']:.2f}, lift {agg['top5_lift']:+.2f})")
        os.makedirs(os.path.dirname(a.out), exist_ok=True)
        json.dump({"gates": gates, "results": res,
                   "protocol": "protocol.reid_metrics, all-gallery, same-view exclusion, NM#5-6",
                   "wall_s": round(time.time() - t0, 1)},
                  io.open(a.out, "w", encoding="utf-8"), indent=1)

    print(f"\n{'arm':<18} | {'top1':>7} {'null':>6} {'lift':>7} | {'top5':>7} {'null':>6} {'lift':>7}")
    for n, v in res.items():
        m = v["mean"]
        print(f"{n:<18} | {m['top1']:7.2f} {m['top1_null']:6.2f} {m['top1_lift']:+7.2f} | "
              f"{m['top5']:7.2f} {m['top5_null']:6.2f} {m['top5_lift']:+7.2f}")
    bad = [g for g in gates if not g["pass"]]
    print(f"\n  GATES: {len(gates) - len(bad)}/{len(gates)} pass"
          + ("" if not bad else "  FAILED: " + ", ".join(f"{g['arm']}s{g['seed']}" for g in bad)))
    print(f"  wrote {a.out}   ({time.time() - t0:.0f} s)")
    sys.exit(0 if not bad else 2)


if __name__ == "__main__":
    main()
