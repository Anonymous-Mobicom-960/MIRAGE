"""TM1 / TM2id / TM2sq for EVERY live gait arm - with the two controls the protocol was missing.

§A.2i-4 and the 2026-08-03 boost run measured TM3 only, against an ASSUMED 2.0 % chance floor and
with no same-harness upper bound. That is the exact gap the 2026-08-03 silhouette audit closed on
the other channel (§A.6n-8: "a plain white rectangle scored level with a real silhouette" - only a
positive control revealed the metric was not measuring identity). This script adds both:

  RAW positive control   the frozen adversary on UNTOUCHED tracklets, through this harness's own
                         embedding and this harness's own protocol. If the defended arms are to be
                         read as "identity removed", this row is what they are removed FROM.
  measured null          identity labels permuted at the TRACKLET level and pushed through the
                         identical scoring path. 2.0 % is what 1/50 predicts; this is what the
                         pipeline actually returns when there is no identity to find.

THREAT FRAMINGS (unchanged from tm1_gait0802.py)
  TM1    gallery CLEAN, probe ANON - a clean-enrolled adversary meeting an anonymised probe.
  TM2id  gallery+probe ANON, per-IDENTITY seed - pseudonymity: does a constant per-person transform
         just relabel people into separable clusters?
  TM2sq  gallery+probe ANON, per-SEQUENCE seed - the DEPLOYED seeding.

🔴 TM1 IS NOT A SUBSTITUTE FOR TM3. §A.2c measured TM1 under-pricing the angle knob by 4.1x. It is
reported because a knob that looks good to both framings is better understood than one measured
against either alone.

  python tm12_gait0807.py [--configs a,b,c]
"""
import argparse
import io
import json
import os
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
from casia_loader import load_tracklets                        # noqa: E402
from adversary import GaitGraphAdversary                       # noqa: E402
import protocol as P                                           # noqa: E402
import anon_adapter as A                                       # noqa: E402

DATA = os.path.join(HERE, "data")
CSV_TEST = os.path.join(DATA, "casia-b_pose_test.csv")
WEIGHTS = os.path.join(DATA, "gaitgraph_resgcn-n39-r8_coco_seq_60.pth")
OUT = os.path.join(HERE, "..", "reports", "reid", "TM12_gait0807.json")

_A1410 = dict(do_canon=True, cadence_amp=0.60, angle_const_deg=14, angle_drift_deg=10,
              asymmetry_sigma=0.12, binarize_conf=0.5)
_L = ("uarm", "farm", "thigh", "shin")
_G6 = dict(_A1410, angle_groups=_L)

CONFIGS = {
    "shipped":            dict(_A1410),                                        # = L4-bin-a1410
    "g6tq":               dict(_G6),
    "g8proj":             dict(_A1410, projection_fit=True),
    "g9proj":             dict(_G6, projection_fit=True),                      # = a1410-tqproj
    "g6tq-limb090":       dict(_G6, limb_phase_amp=0.90),
    "g6tq-limb090-cad090": dict(_G6, limb_phase_amp=0.90, cadence_amp=0.90),
    "g6tq-limb120-cad090": dict(_G6, limb_phase_amp=1.20, cadence_amp=0.90),
    "g10_boost":          dict(_G6, limb_phase_amp=1.20, cadence_amp=0.90, projection_fit=True),
    # the ceiling: canonicalisation + conf binarisation only, every dynamic knob off
    "L0-floor":           dict(do_canon=True, cadence_amp=0.0, angle_const_deg=0,
                               angle_drift_deg=0, asymmetry_sigma=0.0, binarize_conf=0.5),

    # ---- the 2026-08-07 knobs, added in response to the owner's two visual reports -----------
    # Each is g10_boost plus ONE change, so the frozen adversary prices the change and nothing
    # else. `angle_mirror`/`head_anchor` reach anonymize_v2 unchanged: anon_adapter.transform_v2
    # pops only strip_conf/binarize_conf and forwards the rest verbatim.
    "g14_mirror":   dict(_G6, limb_phase_amp=1.20, cadence_amp=0.90, projection_fit=True,
                         angle_mirror="mirror"),
    "g16_outward":  dict(_G6, limb_phase_amp=1.20, cadence_amp=0.90, projection_fit=True,
                         angle_mirror="outward"),
    # legs exempt from the ANGLE knob: they keep the real joint angles and are perturbed only in
    # TIME by limb_phase + cadence. Geometrically it is the only option with no lean, no splay and
    # no crossing; whether it keeps the privacy is exactly what this row measures.
    "g18_armangles": dict(_A1410, angle_groups=("uarm", "farm"), limb_phase_amp=1.20,
                          cadence_amp=0.90, projection_fit=True),
    # the HEAD ANCHOR, priced on its own on top of g10 and on top of g18
    "g12_headanchor": dict(_G6, limb_phase_amp=1.20, cadence_amp=0.90, projection_fit=True,
                           head_anchor="nose"),
    "g19_armangles_headanchor": dict(_A1410, angle_groups=("uarm", "farm"), limb_phase_amp=1.20,
                                     cadence_amp=0.90, projection_fit=True, head_anchor="nose"),
    "g13_shipped_headanchor": dict(_A1410, head_anchor="nose"),
}


def is_gallery(k):
    return k[1] == 0 and k[2] <= 4


def cell(emb):
    c = P.canonical_casia_b_rank1(emb)
    return {"NM": c["NM#5-6"], "BG": c["BG#1-2"], "CL": c["CL#1-2"]}


def permuted_null(emb, reps=5, seed=0):
    """Measured null: reassign each tracklet a random subject id, score through the SAME protocol.

    A 5-tuple key keeps every index the protocol uses (k[0..3]) meaningful while guaranteeing the
    relabelled keys stay unique - permuting the subject field in place would collide.
    """
    rng = np.random.RandomState(seed)
    keys = list(emb)
    subs = np.array([k[0] for k in keys])
    out = []
    for r in range(reps):
        pm = rng.permutation(len(keys))
        e2 = {(subs[pm[i]],) + tuple(keys[i][1:]) + (i,): emb[keys[i]] for i in range(len(keys))}
        c = P.canonical_casia_b_rank1(e2)
        out.append([c["NM#5-6"], c["BG#1-2"], c["CL#1-2"]])
    m = np.mean(out, 0)
    return {"NM": float(m[0]), "BG": float(m[1]), "CL": float(m[2]),
            "reps": reps, "NM_sd": float(np.std([o[0] for o in out]))}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--configs", default=",".join(CONFIGS))
    ap.add_argument("--out", default=OUT)
    a = ap.parse_args()
    want = [c for c in a.configs.split(",") if c.strip() in CONFIGS]

    tr = load_tracklets(CSV_TEST, split="test", min_frames=61)
    adv = GaitGraphAdversary(WEIGHTS, use_flip=True)
    print(f"{len(tr)} test tracklets, {len({k[0] for k in tr})} subjects; embedding CLEAN ...",
          flush=True)
    clean = adv.embed_dict(tr)
    template = A.build_template(tr)

    res = {}
    # ---- the two controls, first, so a failure here stops the run before 9 arms are spent ----
    res["_RAW_control"] = {"TM1": cell(clean), "TM2id": cell(clean), "TM2sq": cell(clean),
                           "full": P.reid_metrics(clean, verbose=False)}
    res["_measured_null"] = permuted_null(clean)
    print(f"RAW positive control  NM {res['_RAW_control']['TM1']['NM']*100:.2f} %   "
          f"measured null NM {res['_measured_null']['NM']*100:.2f} % "
          f"(1/50 predicts {P.CHANCE_RANK1_50*100:.1f} %)\n", flush=True)

    for name in want:
        cfg = CONFIGS[name]
        print(f"--- {name} ---", flush=True)
        aID = A.transform_v2(tr, dict(cfg), template, seed_mode="per_identity")
        aSQ = A.transform_v2(tr, dict(cfg), template, seed_mode="per_sequence")
        eID, eSQ = adv.embed_dict(aID), adv.embed_dict(aSQ)
        tm1 = {k: (clean[k] if is_gallery(k) else eID[k]) for k in clean}
        res[name] = {"TM1": cell(tm1), "TM2id": cell(eID), "TM2sq": cell(eSQ),
                     "full_TM2sq": P.reid_metrics(eSQ, verbose=False),
                     "null_TM2sq": permuted_null(eSQ)}
        r = res[name]
        print(f"    TM1 {r['TM1']['NM']*100:6.2f}  TM2id {r['TM2id']['NM']*100:6.2f}  "
              f"TM2sq {r['TM2sq']['NM']*100:6.2f}  (null {r['null_TM2sq']['NM']*100:.2f})",
              flush=True)

    print(f"\n  {'config':22s} | {'TM1':>7s} {'TM2id':>7s} {'TM2sq':>7s} {'null':>6s} | "
          f"{'TM2sq rank5':>11s} {'mAP':>6s} {'EER':>6s}")
    print(f"  {'RAW positive control':22s} | {res['_RAW_control']['TM1']['NM']*100:7.2f} "
          f"{' - ':>7s} {' - ':>7s} {res['_measured_null']['NM']*100:6.2f} |")
    for n in want:
        r = res[n]
        f = r["full_TM2sq"]["NM#5-6"]
        print(f"  {n:22s} | {r['TM1']['NM']*100:7.2f} {r['TM2id']['NM']*100:7.2f} "
              f"{r['TM2sq']['NM']*100:7.2f} {r['null_TM2sq']['NM']*100:6.2f} | "
              f"{f['rank5']*100:11.2f} {f['mAP']*100:6.2f} {f['EER']*100:6.2f}")

    os.makedirs(os.path.dirname(a.out), exist_ok=True)
    json.dump({"note": __doc__.strip(), "chance_rank1_predicted": P.CHANCE_RANK1_50,
               "n_test_tracklets": len(tr),
               "configs": {k: {kk: (list(vv) if isinstance(vv, tuple) else vv)
                               for kk, vv in CONFIGS[k].items()} for k in want},
               "results": res}, io.open(a.out, "w", encoding="utf-8"), indent=1)
    print(f"\n  wrote {OUT}")


if __name__ == "__main__":
    main()
