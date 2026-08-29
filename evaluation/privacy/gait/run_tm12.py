"""
run_tm12.py -- the first CREDIBLE anonymization numbers (replaces privacy-by-
construction). No retraining: our anonymizer + strawman baselines are pushed
through the frozen, validated GaitGraph adversary under three threat framings.

  TM1   naive cross-domain : gallery CLEAN, probe ANON (per-identity)   -> can a
        clean-enrolled adversary still match an anonymized probe?
  TM2id pseudonymity        : gallery+probe ANON, per-IDENTITY seed      -> does a
        constant per-person transform just relabel into separable clusters?
  TM2sq linkage-broken      : gallery+probe ANON, per-SEQUENCE seed      -> the
        pseudonymity gap when the transform is not a fixed per-person bijection.

Privacy is good only when the adversary COLLAPSES TOWARD chance (rank-1 -> 2.0%).
"""
import os
import sys
import json
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
from casia_loader import load_tracklets
from adversary import GaitGraphAdversary
import protocol as P
import anon_adapter as A

DATA = os.path.join(HERE, "data")
CSV_TEST = os.path.join(DATA, "casia-b_pose_test.csv")
WEIGHTS = os.path.join(DATA, "gaitgraph_resgcn-n39-r8_coco_seq_60.pth")
OUT_DIR = os.path.join(HERE, "..", "reports", "reid")
KINDS = ["raw", "jitter", "jitter_lowpass", "ours"]


def is_gallery(k):
    return k[1] == 0 and k[2] <= 4


def cell_metrics(emb):
    c = P.canonical_casia_b_rank1(emb)
    full = P.reid_metrics(emb, bootstrap=0, verbose=False)
    return {"NM": c["NM#5-6"], "BG": c["BG#1-2"], "CL": c["CL#1-2"],
            "mAP": full["overall"]["mAP"], "EER_nm": full["NM#5-6"]["EER"]}


def main():
    os.makedirs(OUT_DIR, exist_ok=True)
    tracklets = load_tracklets(CSV_TEST, split="test", min_frames=61)
    adv = GaitGraphAdversary(WEIGHTS, use_flip=True)

    print("\nembedding CLEAN reference ...")
    clean = adv.embed_dict(tracklets)

    msig, rms = A.estimate_matched_sigma(tracklets)
    print(f"matched-jitter sigma={msig:.3f}px per-coord (ours RMS joint disp={rms:.3f}px)\n")

    results = {}
    for kind in KINDS:
        print(f"--- condition: {kind} ---")
        if kind == "raw":
            eID = eSQ = clean
        else:
            tID = A.transform_tracklets(tracklets, kind, "per_identity", matched_sigma=msig)
            tSQ = A.transform_tracklets(tracklets, kind, "per_sequence", matched_sigma=msig)
            eID = adv.embed_dict(tID)
            eSQ = adv.embed_dict(tSQ)
        tm1 = {k: (clean[k] if is_gallery(k) else eID[k]) for k in clean}
        results[kind] = {"TM1": cell_metrics(tm1), "TM2id": cell_metrics(eID),
                         "TM2sq": cell_metrics(eSQ)}

    # ---- table ----
    chance = P.CHANCE_RANK1_50 * 100
    hdr = (f"\n{'':16s} | {'TM1 gal-clean/probe-anon':>26s} | "
           f"{'TM2id both-anon per-id':>24s} | {'TM2sq both-anon per-seq':>24s}")
    sub = (f"{'condition':16s} | {'NM':>6s}{'BG':>6s}{'CL':>6s}{'mAP':>7s} | "
           f"{'NM':>6s}{'BG':>6s}{'CL':>6s} | {'NM':>6s}{'BG':>6s}{'CL':>6s}")
    print(hdr); print(sub); print("-" * len(sub))
    lines = [hdr, sub, "-" * len(sub)]
    for kind in KINDS:
        r = results[kind]
        line = (f"{kind:16s} | "
                f"{r['TM1']['NM']*100:6.1f}{r['TM1']['BG']*100:6.1f}{r['TM1']['CL']*100:6.1f}"
                f"{r['TM1']['mAP']*100:7.1f} | "
                f"{r['TM2id']['NM']*100:6.1f}{r['TM2id']['BG']*100:6.1f}{r['TM2id']['CL']*100:6.1f} | "
                f"{r['TM2sq']['NM']*100:6.1f}{r['TM2sq']['BG']*100:6.1f}{r['TM2sq']['CL']*100:6.1f}")
        print(line); lines.append(line)
    tail = (f"(chance rank-1 = {chance:.1f}%.  rank-1 in %, canonical per-angle protocol; "
            f"mAP = all-gallery overall.)")
    print(tail); lines.append(tail)

    with open(os.path.join(OUT_DIR, "BASELINE_tm12.json"), "w") as f:
        json.dump({"matched_sigma_px": msig, "ours_rms_disp_px": rms,
                   "knobs": A.DEFAULT_KNOBS, "results": results}, f, indent=2)
    with open(os.path.join(OUT_DIR, "BASELINE_tm12.md"), "w") as f:
        f.write("# TM1/TM2 baseline (frozen GaitGraph adversary, no retrain)\n\n```")
        f.write("\n".join(lines)); f.write("\n```\n")
    print(f"\nwrote reports/reid/BASELINE_tm12.{{json,md}}")


if __name__ == "__main__":
    main()
