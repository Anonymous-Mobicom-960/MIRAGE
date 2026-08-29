"""
reproduce_baseline.py -- MILESTONE 1: validate the adversary BEFORE any anonymizer.

Loads the CASIA-B test split (ids 75-124), embeds every tracklet with the frozen
GaitGraph ResGCN, and reports:
  (A) the canonical GaitGraph per-angle rank-1 (must land near the published
      ~NM 87.7 / BG 74.8 / CL 66.3) -- this is the go/no-go that the adversary works;
  (B) the publication-grade suite (rank-1/5, mAP, EER, CMC + bootstrap CIs).

If (A) reproduces, the adversary is trustworthy and we can put the anonymizer in
front of it. If it doesn't, everything downstream is meaningless -- so this gate
comes first.
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

DATA = os.path.join(HERE, "data")
CSV_TEST = os.path.join(DATA, "casia-b_pose_test.csv")
WEIGHTS = os.path.join(DATA, "gaitgraph_resgcn-n39-r8_coco_seq_60.pth")
OUT_DIR = os.path.join(HERE, "..", "reports", "reid")
PUBLISHED = {"NM#5-6": 0.877, "BG#1-2": 0.748, "CL#1-2": 0.663}  # GaitGraph ICIP'21


def main():
    os.makedirs(OUT_DIR, exist_ok=True)
    print("=" * 72)
    print("MILESTONE 1 -- reproduce GaitGraph clean CASIA-B baseline (adversary sanity)")
    print("=" * 72)

    tracklets = load_tracklets(CSV_TEST, split="test", min_frames=61)
    ids = sorted({k[0] for k in tracklets})
    assert min(ids) >= 75 and max(ids) <= 124, f"test ids out of range: {min(ids)}..{max(ids)}"
    print(f"[gate] identity range {min(ids)}..{max(ids)} ({len(ids)} subjects) -- disjoint from train 1..74 OK\n")

    adv = GaitGraphAdversary(WEIGHTS, network_name="resgcn-n39-r8",
                             sequence_length=60, embedding_size=128, use_flip=True)
    print(f"\nembedding {len(tracklets)} tracklets ...")
    import time
    t0 = time.time()
    emb = adv.embed_dict(tracklets, batch_size=256)
    dt = time.time() - t0
    d = len(next(iter(emb.values())))
    print(f"done in {dt:.1f}s  ({len(emb)} embeddings, dim={d}, "
          f"{1000*dt/len(emb):.1f} ms/tracklet incl. flip-TTA)\n")

    print("-" * 72)
    print("(A) CANONICAL GaitGraph per-angle rank-1  vs  published")
    print("-" * 72)
    canon = P.canonical_casia_b_rank1(emb)
    ok = True
    for k in ["NM#5-6", "BG#1-2", "CL#1-2"]:
        pub = PUBLISHED[k]; got = canon[k]; delta = (got - pub) * 100
        flag = "OK" if abs(delta) <= 3.0 else "CHECK"
        if abs(delta) > 3.0:
            ok = False
        print(f"  {k:8s}  ours {got*100:5.2f}%   published {pub*100:5.1f}%   d={delta:+5.2f}pp   [{flag}]")
    print(f"  {'mean':8s}  ours {canon['mean']*100:5.2f}%")
    print(f"\n  => adversary {'REPRODUCES' if ok else 'DEVIATES FROM'} published baseline "
          f"(tolerance ±3pp)\n")

    print("-" * 72)
    print("(B) PUBLICATION-GRADE SUITE (all-gallery, same-view excluded, bootstrap CI)")
    print("-" * 72)
    full = P.reid_metrics(emb, max_rank=20, bootstrap=1000, seed=0, verbose=True)

    out = {"milestone": "reproduce_clean_baseline",
           "n_subjects": len(ids), "n_tracklets": len(emb),
           "embed_ms_per_tracklet": 1000 * dt / len(emb),
           "adversary": {"network": "resgcn-n39-r8", "num_input": adv.num_input,
                         "num_channel": adv.num_channel, "use_flip": adv.use_flip},
           "canonical_rank1": canon, "published": PUBLISHED,
           "reproduces_within_3pp": ok, "full_suite": full}
    outp = os.path.join(OUT_DIR, "BASELINE_clean.json")
    with open(outp, "w") as f:
        json.dump(out, f, indent=2)
    print(f"\nwrote {os.path.relpath(outp, HERE)}")
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
