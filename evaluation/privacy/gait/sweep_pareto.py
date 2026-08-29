"""
sweep_pareto.py -- privacy<->utility Pareto for the redesigned anonymizer, with
PER-SEQUENCE seeding as the deployment default (per-identity seeding re-creates a
pseudo-identity, so it is retired).

Dials the dynamic channel from 'canon only' up to aggressive, plus two off-axis
points (cadence-only / angle-only) to attribute the effect. Every point reports:
  TM1  gallery CLEAN / probe ANON (per-seq)   -- naive cross-domain attacker
  TM2  gallery ANON  / probe ANON (per-seq)   -- cross-clip linkage attacker
  utility  MPJPE(px) + mean bone-angle err(deg) vs original  (sampled)
all rank-1 % against the 2.0% chance line. Frozen (clean-trained) adversary = a
LOWER BOUND on identity leakage; TM3 adaptive-retrain on the chosen knee comes next.
"""
import os
import sys
import json
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
sys.path.insert(0, os.path.join(HERE, ".."))
from casia_loader import load_tracklets
from adversary import GaitGraphAdversary
import protocol as P
import anon_adapter as A
from segbench.pose_anon import EDGES, _virtual

DATA = os.path.join(HERE, "data")
CSV_TEST = os.path.join(DATA, "casia-b_pose_test.csv")
WEIGHTS = os.path.join(DATA, "gaitgraph_resgcn-n39-r8_coco_seq_60.pth")
OUT_DIR = os.path.join(HERE, "..", "reports", "reid")


def C(cad, ac, ad, asym):
    return dict(do_canon=True, cadence_amp=cad, angle_const_deg=ac,
               angle_drift_deg=ad, asymmetry_sigma=asym)


LADDER = {
    "L0 canon":      C(0.00, 0, 0, 0.00),
    "L1":            C(0.15, 4, 3, 0.03),
    "L2":            C(0.25, 8, 6, 0.05),
    "L3":            C(0.40, 14, 10, 0.08),
    "L4":            C(0.60, 20, 15, 0.12),
    "L5 aggressive": C(0.80, 28, 20, 0.16),
    "cad-only":      C(0.60, 0, 0, 0.00),
    "ang-only":      C(0.00, 20, 15, 0.12),
}


def is_gallery(k):
    return k[1] == 0 and k[2] <= 4


def cell(emb):
    c = P.canonical_casia_b_rank1(emb)
    m = P.reid_metrics(emb, bootstrap=0, verbose=False)
    return {"NM": c["NM#5-6"], "BG": c["BG#1-2"], "CL": c["CL#1-2"], "mAP": m["overall"]["mAP"]}


def utility(orig, trans, keys):
    mp, ang = [], []
    for k in keys:
        o = orig[k][..., :2]; a = trans[k][..., :2]
        T = min(len(o), len(a)); o = o[:T]; a = a[:T]
        mp.append(float(np.mean(np.linalg.norm(o - a, axis=-1))))
        ea = []
        for t in range(0, T, 3):
            vo = _virtual(o[t]); va = _virtual(a[t])
            for p, c, _ in EDGES:
                bo = vo[c] - vo[p]; ba = va[c] - va[p]
                no = np.linalg.norm(bo); na = np.linalg.norm(ba)
                if no > 1e-6 and na > 1e-6:
                    ea.append(np.degrees(np.arccos(np.clip(np.dot(bo, ba) / (no * na), -1, 1))))
        ang.append(float(np.mean(ea)) if ea else 0.0)
    return round(float(np.mean(mp)), 1), round(float(np.mean(ang)), 1)


def main():
    os.makedirs(OUT_DIR, exist_ok=True)
    tracklets = load_tracklets(CSV_TEST, split="test", min_frames=61)
    adv = GaitGraphAdversary(WEIGHTS, use_flip=True)
    clean = adv.embed_dict(tracklets)
    template = A.build_template(tracklets)
    rng = np.random.RandomState(0)
    ukeys = [list(tracklets)[i] for i in rng.permutation(len(tracklets))[:300]]

    rows = {"raw": {"TM1": cell(clean), "TM2": cell(clean), "util": (0.0, 0.0)}}
    for name, cfg in LADDER.items():
        print(f"--- {name} {cfg} ---", flush=True)
        t = A.transform_v2(tracklets, cfg, template, seed_mode="per_sequence")
        e = adv.embed_dict(t)
        tm1 = {k: (clean[k] if is_gallery(k) else e[k]) for k in clean}
        rows[name] = {"TM1": cell(tm1), "TM2": cell(e), "util": utility(tracklets, t, ukeys)}

    order = ["raw"] + list(LADDER)
    sub = (f"{'config':14s} | {'TM1 clean/anon NM/BG/CL mAP':>28s} | "
           f"{'TM2 anon/anon NM/BG/CL mAP':>27s} | {'MPJPEpx':>7s} {'angDeg':>6s}")
    lines = [sub, "-" * len(sub)]
    for n in order:
        r = rows[n]; u = r["util"]
        lines.append(
            f"{n:14s} | "
            f"{r['TM1']['NM']*100:6.1f}{r['TM1']['BG']*100:6.1f}{r['TM1']['CL']*100:6.1f}{r['TM1']['mAP']*100:7.1f} | "
            f"{r['TM2']['NM']*100:6.1f}{r['TM2']['BG']*100:6.1f}{r['TM2']['CL']*100:6.1f}{r['TM2']['mAP']*100:7.1f} | "
            f"{u[0]:7.1f} {u[1]:6.1f}")
    tail = "(rank-1 %, chance=2.0%. per-SEQUENCE seeding. utility vs original, n=300 sample.)"
    print("\n" + "\n".join(lines)); print(tail)

    with open(os.path.join(OUT_DIR, "SWEEP_pareto.json"), "w") as f:
        json.dump({"template": template, "ladder": LADDER, "rows": rows}, f, indent=2)
    with open(os.path.join(OUT_DIR, "SWEEP_pareto.md"), "w") as f:
        f.write("# Privacy/utility Pareto -- pose_anon_v2, per-sequence seeding\n\n```\n")
        f.write("\n".join(lines)); f.write("\n" + tail + "\n```\n")
    print("\nwrote reports/reid/SWEEP_pareto.{json,md}")


if __name__ == "__main__":
    main()
