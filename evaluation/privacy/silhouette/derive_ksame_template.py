#!/usr/bin/env python3
"""
derive_ksame_template.py -- build the POPULATION boundary-radius template for
`MASK_SHAPE_MODE="ksame"` (mirage_tier1._shape_polys).
=============================================================================

WHAT IT PRODUCES
    T(theta): a scale-free canonical silhouette boundary profile on a uniform angular grid,
    normalised to mean radius 1. `ksame` emits max(r_person(theta), s*T(theta)) with s the
    person's own mean radius, so every subject's width profile collapses toward ONE shared shape
    instead of carrying their own build -- the mask analogue of pose_anon_edge._TEMPLATE_RATIOS.

IDENTITY HYGIENE (hard rule)
    Derived from CASIA-B **TRAIN** identities 001..074 ONLY. The 50 held-out TEST identities
    (075..124) that the re-ID evaluation scores on are never read here, so the template cannot
    leak information about the subjects it is later evaluated against. Asserted in code.

    python derive_ksame_template.py --silh-root "<...>/CASIA-B Dataset/output"
"""
import argparse
import glob
import json
import os
import sys
import time

import cv2
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
OUT = os.path.join(HERE, "reports")
TRAIN_IDS = set(range(1, 75))
TEST_IDS = set(range(75, 125))
BINS = 180


def profile(mask, bins=BINS):
    """Largest external contour -> max-radius-per-angular-bin profile, normalised to mean 1."""
    cnts, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    if not cnts:
        return None
    c = max(cnts, key=cv2.contourArea)
    p = c.reshape(-1, 2).astype(np.float32)
    if p.shape[0] < 16:
        return None
    ctr = p.mean(0)
    v = p - ctr
    r = np.linalg.norm(v, axis=1)
    a = np.arctan2(v[:, 1], v[:, 0])
    idx = np.minimum(((a + np.pi) * (bins / (2.0 * np.pi))).astype(np.int32), bins - 1)
    rb = np.zeros(bins, np.float32)
    np.maximum.at(rb, idx, r)
    hit = rb > 0
    if hit.sum() < bins * 0.5:
        return None
    if not hit.all():
        ii = np.nonzero(hit)[0].astype(np.float32)
        xs = np.concatenate([ii - bins, ii, ii + bins])
        ys = np.tile(rb[hit], 3)
        rb = np.interp(np.arange(bins, dtype=np.float32), xs, ys).astype(np.float32)
    m = float(rb.mean())
    return rb / m if m > 1e-6 else None


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--silh-root", required=True)
    ap.add_argument("--frame-stride", type=int, default=4)
    ap.add_argument("--subject-stride", type=int, default=1)
    a = ap.parse_args()
    os.makedirs(OUT, exist_ok=True)

    subs = sorted(TRAIN_IDS)[::a.subject_stride]
    assert not (set(subs) & TEST_IDS), "TEST identities must never enter the template"
    profs, n_frames, t0 = [], 0, time.time()
    for i, sid in enumerate(subs, 1):
        d = os.path.join(a.silh_root, f"{sid:03d}")
        if not os.path.isdir(d):
            continue
        for cond in sorted(os.listdir(d)):
            for ang in sorted(os.listdir(os.path.join(d, cond))):
                pngs = sorted(glob.glob(os.path.join(d, cond, ang, "*.png")))[::a.frame_stride]
                for p in pngs:
                    im = cv2.imread(p, cv2.IMREAD_GRAYSCALE)
                    if im is None:
                        continue
                    n_frames += 1
                    pr = profile((im > 127).astype(np.uint8))
                    if pr is not None:
                        profs.append(pr)
        print(f"  [{i}/{len(subs)}] sid {sid:03d}: {len(profs)} profiles "
              f"({time.time()-t0:.0f}s)", flush=True)
    if not profs:
        raise SystemExit("no profiles built -- check --silh-root")
    P = np.stack(profs)
    out = {"n_profiles": int(P.shape[0]), "n_frames_read": n_frames, "bins": BINS,
           "train_subjects": subs, "frame_stride": a.frame_stride,
           "source": os.path.abspath(a.silh_root),
           "templates": {}}
    for q in (50, 60, 65, 75, 85):
        t = np.percentile(P, q, axis=0).astype(np.float64)
        t = t / t.mean()                                    # scale-free
        out["templates"][f"p{q}"] = [round(float(x), 4) for x in t]
    path = os.path.join(OUT, "KSAME_TEMPLATE.json")
    with open(path, "w") as f:
        json.dump(out, f, indent=1)
    print(f"\n[ksame] {P.shape[0]} profiles from {len(subs)} TRAIN subjects -> {path}")
    for q in (50, 60, 65, 75, 85):
        t = np.asarray(out["templates"][f"p{q}"])
        print(f"  p{q}: min {t.min():.3f} max {t.max():.3f}")


if __name__ == "__main__":
    main()
