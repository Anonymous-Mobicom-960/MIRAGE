#!/usr/bin/env python3
"""
eval_fid_pooled.py - POOLED FID for the Tier-2 anonymized output, at a real (large) n.
=====================================================================================
The per-clip visual-utility FID was n=100 - below the ~1000-frame bar where the 2048-d Inception
covariance is well-conditioned (the per-clip harness documents that n=100 yields nonsense). This pools
EVERY available on-device final-output frame (the anonymized result) into one "generated" set and a
large sample of ORIGINAL real-video frames into the "real" set, and computes ONE Fréchet distance - 
the only FID worth citing. Reuses eval_tier2_visual_utility's Inception-pool3 + Frechet, so both
harnesses compute the metric identically. `fid_reliable` = both sides >= --min-frames (default 1000).
"""
import argparse, glob, json, os, sys
import cv2, numpy as np, torch

HERE = os.path.dirname(os.path.abspath(__file__)); REPO = os.path.dirname(HERE)
sys.path.insert(0, REPO)
sys.path.insert(0, HERE)
# These live in the per-clip harness beside this file. The import previously named a module
# (`eval_pool_all`) that is not in this repository, so this script could not even be imported.
from eval_tier2_visual_utility import get_inception_features, compute_fid


def load(path, stride=1, cap=100000, size=299):
    c = cv2.VideoCapture(path); out = []
    i = 0
    while len(out) < cap:
        ok, f = c.read()
        if not ok: break
        if i % stride == 0: out.append(cv2.resize(f, (size, size)))
        i += 1
    c.release(); return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--min-frames", type=int, default=1000)
    ap.add_argument("--gen", nargs="*", default=[],
                    help="Generated (anonymised) videos to pool over.")
    ap.add_argument("--real-glob",
                    default=os.path.join(REPO, "examples", "inputs", "*.mp4"),
                    help="Glob for the reference (real) clips.")
    a = ap.parse_args()
    dev = "cuda" if torch.cuda.is_available() else "cpu"

    # GENERATED (anonymized final outputs, on-device)
    # Pass the on-device final outputs with --gen. The paper pooled three device runs:
    # set A alley (300 f, 1264^2), set B garden (300 f, 640^2), on-phone e2e (100 f, 512^2).
    gen_paths = list(a.gen)
    gen = []
    for p in gen_paths:
        if os.path.exists(p):
            fr = load(p); gen += fr; print(f"[gen] {os.path.basename(os.path.dirname(os.path.dirname(p)))}/{os.path.basename(p)}: {len(fr)}")
    # REAL (original clips - real people in real scenes)
    real_clips = sorted(glob.glob(a.real_glob))
    # target ~1800 real frames pooled across clips
    per = max(1, 1800 // max(1, len(real_clips)))
    real = []
    for p in real_clips:
        fr = load(p, stride=3, cap=per); real += fr
    print(f"[real] {len(real_clips)} clips -> {len(real)} frames pooled")
    print(f"[fid] device={dev}  gen={len(gen)}  real={len(real)}  (Inception pool3 …)")

    gf = get_inception_features(gen, dev)
    rf = get_inception_features(real, dev)
    fid = compute_fid(rf, gf)
    reliable = len(gen) >= a.min_frames and len(real) >= a.min_frames
    out = {"pooled_fid": round(fid, 2), "n_generated": len(gen), "n_real": len(real),
           "fid_reliable_ge_%d" % a.min_frames: reliable,
           "generated_sources": ["setA_alley_final(300)", "setB_garden_final(300)", "onphone_e2e_final(100)"],
           "real_source": "input/reid_dataset/*.mp4 (original real clips, stride-3 pooled)",
           "note": "Pooled FID (anonymized Tier-2 output distribution vs real-video distribution). "
                   "Generated side is the limiting n; %s the %d-frame reliability bar." %
                   ("clears" if reliable else "below", a.min_frames)}
    print(json.dumps(out, indent=2))
    op = os.path.join(HERE, "reports", "TIER2_FID_POOLED.json"); os.makedirs(os.path.dirname(op), exist_ok=True)
    json.dump(out, open(op, "w"), indent=2); print("wrote", op)


if __name__ == "__main__":
    main()
