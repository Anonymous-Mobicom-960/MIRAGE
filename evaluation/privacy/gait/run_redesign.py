"""
run_redesign.py -- evaluate the REDESIGNED anonymizer (pose_anon_v2) against the
frozen, validated GaitGraph adversary, and expose the privacy<->utility trade.

Ablation (all vs the 2.0% chance line):
  raw            reference upper bound
  v1-ours        the pseudonymization design we are replacing
  v2-canon       identity-collapse only (snap limbs to shared template)
  v2-canon+cad   + cadence/rhythm warp
  v2-canon+ang   + joint-angle offsets + asymmetry
  v2-full        canon + cadence + angle + asymmetry

Threat framings (no retrain): TM1 gal-clean/probe-anon, TM2id both-anon per-id,
TM2sq both-anon per-seq.  Utility proxies (sampled): MPJPE(px) and mean bone-angle
error(deg) vs original -- privacy is bought with motion distortion, shown explicitly.
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

V2_CONFIGS = {
    "v2-canon":     dict(do_canon=True),
    "v2-canon+cad": dict(do_canon=True, cadence_amp=0.25),
    "v2-canon+ang": dict(do_canon=True, angle_const_deg=8, angle_drift_deg=6, asymmetry_sigma=0.05),
    "v2-full":      dict(do_canon=True, cadence_amp=0.25, angle_const_deg=8,
                         angle_drift_deg=6, asymmetry_sigma=0.05),
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
                    cs = np.clip(np.dot(bo, ba) / (no * na), -1, 1)
                    ea.append(np.degrees(np.arccos(cs)))
        ang.append(float(np.mean(ea)) if ea else 0.0)
    return float(np.mean(mp)), float(np.mean(ang))


def main():
    os.makedirs(OUT_DIR, exist_ok=True)
    tracklets = load_tracklets(CSV_TEST, split="test", min_frames=61)
    adv = GaitGraphAdversary(WEIGHTS, use_flip=True)
    clean = adv.embed_dict(tracklets)
    template = A.build_template(tracklets)
    print("template (group->len):", {g: round(v, 1) for g, v in template.items()})
    rng = np.random.RandomState(0)
    ukeys = [list(tracklets)[i] for i in rng.permutation(len(tracklets))[:300]]

    rows = {}
    util = {}

    # references
    rows["raw"] = {"TM1": cell(clean), "TM2id": cell(clean), "TM2sq": cell(clean)}
    util["raw"] = (0.0, 0.0)
    msig, _ = A.estimate_matched_sigma(tracklets)
    tID = A.transform_tracklets(tracklets, "ours", "per_identity", matched_sigma=msig)
    tSQ = A.transform_tracklets(tracklets, "ours", "per_sequence", matched_sigma=msig)
    eID, eSQ = adv.embed_dict(tID), adv.embed_dict(tSQ)
    rows["v1-ours"] = {"TM1": cell({k: (clean[k] if is_gallery(k) else eID[k]) for k in clean}),
                       "TM2id": cell(eID), "TM2sq": cell(eSQ)}
    util["v1-ours"] = utility(tracklets, tID, ukeys)

    for name, cfg in V2_CONFIGS.items():
        print(f"--- {name} ---")
        tID = A.transform_v2(tracklets, cfg, template, "per_identity")
        tSQ = A.transform_v2(tracklets, cfg, template, "per_sequence")
        eID, eSQ = adv.embed_dict(tID), adv.embed_dict(tSQ)
        rows[name] = {"TM1": cell({k: (clean[k] if is_gallery(k) else eID[k]) for k in clean}),
                      "TM2id": cell(eID), "TM2sq": cell(eSQ)}
        util[name] = utility(tracklets, tID, ukeys)

    order = ["raw", "v1-ours"] + list(V2_CONFIGS)
    sub = (f"{'config':14s} | {'TM1 NM/BG/CL mAP':>22s} | {'TM2id NM/BG/CL':>19s} | "
           f"{'TM2sq NM/BG/CL':>19s} | {'MPJPEpx':>7s} {'angDeg':>6s}")
    lines = [sub, "-" * len(sub)]
    for n in order:
        r = rows[n]; u = util[n]
        lines.append(
            f"{n:14s} | "
            f"{r['TM1']['NM']*100:5.1f}{r['TM1']['BG']*100:5.1f}{r['TM1']['CL']*100:5.1f}{r['TM1']['mAP']*100:6.1f} | "
            f"{r['TM2id']['NM']*100:6.1f}{r['TM2id']['BG']*100:6.1f}{r['TM2id']['CL']*100:6.1f} | "
            f"{r['TM2sq']['NM']*100:6.1f}{r['TM2sq']['BG']*100:6.1f}{r['TM2sq']['CL']*100:6.1f} | "
            f"{u[0]:7.1f} {u[1]:6.1f}")
    tail = "(rank-1 %, chance=2.0%. utility: MPJPE px & mean bone-angle err deg vs original, sampled n=300.)"
    print("\n" + "\n".join(lines)); print(tail)

    with open(os.path.join(OUT_DIR, "REDESIGN_tm12.json"), "w") as f:
        json.dump({"template": template, "v2_configs": V2_CONFIGS,
                   "rows": rows, "utility": util}, f, indent=2)
    with open(os.path.join(OUT_DIR, "REDESIGN_tm12.md"), "w") as f:
        f.write("# Redesign (pose_anon_v2) vs frozen GaitGraph adversary\n\n```\n")
        f.write("\n".join(lines)); f.write("\n" + tail + "\n```\n")
    print("\nwrote reports/reid/REDESIGN_tm12.{json,md}")


if __name__ == "__main__":
    main()
