#!/usr/bin/env python3
"""
make_sidecars.py - generate the two SIDECAR inputs the MIRAGE Tier-2 phone app prefers, on a laptop.

    python make_sidecars.py --silhouette silhouette.mp4 --character character.mp4 --out ./sidecars

Produces:
    mask.mp4               - white = the person hole in the Tier-1 silhouette video
    synthetic_alpha_p1.mp4 - white = the synthetic character in the cloud's character video

WHY THIS EXISTS
---------------
The phone app can DERIVE both of these when they are absent (HoleMask.kt / MosaicKeyer.kt), but both
derivations are lossy reconstructions of information the pipeline already had. The keyer in particular
cannot separate "garment that happens to match the lightmap block behind it" from "background" - no
purely colour-based method can - which is what leaves see-through notches in the composite.

Running here removes that ceiling for the ALPHA, because a laptop can afford a real segmentation model:
this uses YOLO11-seg, which decides "person vs not-person" from learned shape and semantics rather than
from a colour difference, so a character whose clothing matches its background is still segmented.
That is a genuine information gain, not the same computation moved off-device.

For the MASK the gain is different: the silhouette fill is a flat solid colour, so detection is already
near-exact. What this adds is (a) a TEMPORAL consistency pass the phone's per-frame detector cannot do
cheaply, and (b) taking the guess off the device entirely, so an unusual scene cannot mis-trigger it.

GENERAL BY DESIGN (project rule: never fit to one clip)
------------------------------------------------------
Nothing here is hardcoded to the bench footage. The fill colour is auto-detected per clip by voting over
NEUTRAL and FLAT pixels; the mosaic grid is not assumed at all (the segmentation model does not need it);
frame size, length and fps are taken from the inputs. Every stage prints what it detected so an odd clip
is visible rather than silent.
"""

import argparse
import os
import sys

import cv2
import numpy as np

# ---------------------------------------------------------------- mask (silhouette -> person hole)

FLAT_R, FLAT_T = 2, 4          # 5x5 local range below this = "solid fill"
SPREAD_STRONG, SPREAD_WEAK = 6, 14
TOL_STRONG, TOL_WEAK = 8, 18
GEO_ITERS = 14                 # bounded growth, so the seed cannot flood into same-coloured scenery
MIN_AREA_FRAC = 0.002
CLOSE_R, FINAL_DILATE = 2, 3


def local_range(gray, r=FLAT_R):
    k = np.ones((2 * r + 1,) * 2, np.uint8)
    return cv2.dilate(gray, k).astype(np.int16) - cv2.erode(gray, k).astype(np.int16)


def detect_fill_colour(cap, n, samples=7):
    """Vote for the fill colour over NEUTRAL and FLAT pixels only, so texture cannot win the ballot."""
    votes = np.zeros(256, np.int64)
    total = 0
    for t in np.linspace(0, max(0, n - 1), min(samples, max(1, n))).astype(int):
        cap.set(cv2.CAP_PROP_POS_FRAMES, int(t))
        ok, f = cap.read()
        if not ok:
            continue
        b, g, r = [f[:, :, i].astype(np.int16) for i in range(3)]
        spread = np.maximum.reduce([r, g, b]) - np.minimum.reduce([r, g, b])
        gray = cv2.cvtColor(f, cv2.COLOR_BGR2GRAY)
        cand = (spread <= SPREAD_STRONG) & (local_range(gray) < FLAT_T)
        vals = ((r + g + b) // 3)[cand]
        if vals.size:
            votes += np.bincount(vals.astype(int), minlength=256)
            total += vals.size
    if total < 1000:
        return None, 0.0
    v = int(votes.argmax())
    share = votes[max(0, v - 3):v + 4].sum() / total
    return v, float(share)


def hole_mask(frame, V):
    """Hysteresis: flat on-colour seed -> distance-bounded geodesic growth into a loose on-colour set."""
    h, w = frame.shape[:2]
    b, g, r = [frame[:, :, i].astype(np.int16) for i in range(3)]
    spread = np.maximum.reduce([r, g, b]) - np.minimum.reduce([r, g, b])
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    rng = local_range(gray)
    strong = ((abs(r - V) <= TOL_STRONG) & (abs(g - V) <= TOL_STRONG) & (abs(b - V) <= TOL_STRONG)
              & (spread <= SPREAD_STRONG) & (rng < FLAT_T)).astype(np.uint8)
    weak = ((abs(r - V) <= TOL_WEAK) & (abs(g - V) <= TOL_WEAK) & (abs(b - V) <= TOL_WEAK)
            & (spread <= SPREAD_WEAK)).astype(np.uint8)

    n, lab, st, _ = cv2.connectedComponentsWithStats(strong, 8)
    seeds = np.zeros_like(strong)
    for k in range(1, n):
        if st[k, cv2.CC_STAT_AREA] >= MIN_AREA_FRAC * h * w:
            seeds[lab == k] = 1
    if seeds.sum() == 0:
        return np.zeros((h, w), np.uint8)

    m, k3 = seeds, np.ones((3, 3), np.uint8)
    for _ in range(GEO_ITERS):
        m2 = cv2.dilate(m, k3) & weak
        if (m2 == m).all():
            break
        m = m2
    m = cv2.morphologyEx(m, cv2.MORPH_CLOSE, np.ones((2 * CLOSE_R + 1,) * 2, np.uint8))
    ff = m.copy()
    cv2.floodFill(ff, np.zeros((h + 2, w + 2), np.uint8), (0, 0), 1)
    m[ff == 0] = 1
    return cv2.dilate(m, np.ones((2 * FINAL_DILATE + 1,) * 2, np.uint8))


def temporal_smooth(masks, radius=1):
    """Majority vote across +-radius frames: kills single-frame dropouts the per-frame test cannot see."""
    out = []
    n = len(masks)
    for i in range(n):
        lo, hi = max(0, i - radius), min(n, i + radius + 1)
        s = np.sum([masks[j].astype(np.uint16) for j in range(lo, hi)], axis=0)
        out.append(((s * 2) >= (hi - lo)).astype(np.uint8))
    return out


# ---------------------------------------------------- alpha (character -> synthetic person matte)

def solidify(mask, min_dirs=6, max_dist=None):
    """8-direction inside-ness fill (same rule as Compositor.solidify): close notches that open onto
    the boundary. A gap that is genuinely open toward the ground scores 5, so 6-of-8 preserves it."""
    m = mask.astype(bool)
    if not m.any():
        return mask
    ys = np.nonzero(m.any(axis=1))[0]
    if max_dist is None:
        max_dist = max(8, int(0.25 * (ys.max() - ys.min())))
    h, w = m.shape
    hits = np.zeros((h, w), np.uint8)
    for dy, dx in ((0, -1), (0, 1), (-1, 0), (1, 0), (-1, -1), (-1, 1), (1, -1), (1, 1)):
        d = np.full((h, w), 1 << 20, np.int32)
        ys2 = range(h) if dy == -1 else range(h - 1, -1, -1)
        for y in ys2:
            xs2 = range(w) if dx == -1 else range(w - 1, -1, -1)
            py = y + dy
            row_ok = 0 <= py < h
            for x in xs2:
                if m[y, x]:
                    d[y, x] = 0
                    continue
                px = x + dx
                if row_ok and 0 <= px < w:
                    pv = d[py, px]
                    d[y, x] = pv + 1 if pv < (1 << 20) else (1 << 20)
        hits += (d <= max_dist).astype(np.uint8)
    out = m | ((hits >= min_dirs) & ~m)
    return out.astype(np.uint8)


def character_alpha(model, frame, conf=0.25):
    """Person masks from YOLO11-seg, unioned. Learned semantics, so clothing that matches the
    background is still segmented - the exact case a colour keyer cannot handle."""
    res = model.predict(frame, conf=conf, classes=[0], verbose=False, retina_masks=True)[0]
    h, w = frame.shape[:2]
    if res.masks is None or len(res.masks.data) == 0:
        return np.zeros((h, w), np.uint8), 0
    acc = np.zeros((h, w), np.float32)
    for m in res.masks.data.cpu().numpy():
        if m.shape != (h, w):
            m = cv2.resize(m, (w, h), interpolation=cv2.INTER_LINEAR)
        acc = np.maximum(acc, m)
    return (acc > 0.5).astype(np.uint8), int(len(res.masks.data))


def clean_alpha(m):
    m = cv2.morphologyEx(m, cv2.MORPH_CLOSE, np.ones((9, 9), np.uint8))
    n, lab, st, _ = cv2.connectedComponentsWithStats(m, 8)
    keep = np.zeros_like(m)
    for k in range(1, n):
        if st[k, cv2.CC_STAT_AREA] >= 0.003 * m.size:
            keep[lab == k] = 1
    ff = keep.copy()
    cv2.floodFill(ff, np.zeros((m.shape[0] + 2, m.shape[1] + 2), np.uint8), (0, 0), 1)
    keep[ff == 0] = 1
    return solidify(keep)


# ------------------------------------------------------------------------------------ io helpers

def writer(path, w, h, fps):
    # mp4v is universally readable by Android MediaMetadataRetriever; a binary matte compresses fine.
    return cv2.VideoWriter(path, cv2.VideoWriter_fourcc(*"mp4v"), fps, (w, h), True)


def probe(path):
    cap = cv2.VideoCapture(path)
    if not cap.isOpened():
        sys.exit(f"cannot open {path}")
    n = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    fps = cap.get(cv2.CAP_PROP_FPS) or 10.0
    w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    return cap, n, fps, w, h


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--silhouette", required=True, help="Tier-1 video: real scene, person as a solid fill")
    ap.add_argument("--character", required=True, help="cloud video: synthetic character over a lightmap")
    ap.add_argument("--out", default="sidecars", help="output directory")
    ap.add_argument("--model", default=None, help="YOLO*-seg weights (default: search the repo)")
    ap.add_argument("--conf", type=float, default=0.25)
    args = ap.parse_args()
    os.makedirs(args.out, exist_ok=True)

    # ---------------- mask.mp4 ----------------
    cap, n, fps, w, h = probe(args.silhouette)
    V, share = detect_fill_colour(cap, n)
    if V is None:
        sys.exit("no flat neutral fill detected in the silhouette video - is this the right input?")
    print(f"[mask ] {n} frames {w}x{h} @{fps:g} - fill colour auto-detected: gray({V}), band share {share:.2f}")
    masks = []
    cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
    for i in range(n):
        ok, f = cap.read()
        if not ok:
            break
        masks.append(hole_mask(f, V))
        if i % 20 == 0:
            print(f"[mask ] {i}/{n}", end="\r")
    cap.release()
    masks = temporal_smooth(masks)
    mp = os.path.join(args.out, "mask.mp4")
    vw = writer(mp, w, h, fps)
    cov = []
    for m in masks:
        cov.append(100.0 * m.mean())
        vw.write(cv2.cvtColor(m * 255, cv2.COLOR_GRAY2BGR))
    vw.release()
    print(f"[mask ] wrote {mp} - coverage min {min(cov):.2f}% mean {np.mean(cov):.2f}% max {max(cov):.2f}%")

    # ---------------- synthetic_alpha_p1.mp4 ----------------
    mdl = args.model
    if mdl is None:
        here = os.path.dirname(os.path.abspath(__file__))
        for cand in ("models/yolo11x-seg.pt", "models/yolo11s-seg.pt", "models/yolo11n-seg.pt"):
            p = os.path.join(os.path.dirname(here), cand)
            if os.path.exists(p):
                mdl = p
                break
    if mdl is None:
        sys.exit("no YOLO*-seg weights found; pass --model")
    from ultralytics import YOLO
    import torch
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"[alpha] segmentation model: {os.path.basename(mdl)} on {dev}")
    model = YOLO(mdl)
    model.to(dev)

    cap, n2, fps2, w2, h2 = probe(args.character)
    print(f"[alpha] {n2} frames {w2}x{h2} @{fps2:g}")
    ap_path = os.path.join(args.out, "synthetic_alpha_p1.mp4")
    vw = writer(ap_path, w2, h2, fps2)
    cov2, misses = [], 0
    for i in range(n2):
        ok, f = cap.read()
        if not ok:
            break
        m, k = character_alpha(model, f, args.conf)
        if k == 0:
            misses += 1
        else:
            m = clean_alpha(m)
        cov2.append(100.0 * m.mean())
        vw.write(cv2.cvtColor(m * 255, cv2.COLOR_GRAY2BGR))
        if i % 20 == 0:
            print(f"[alpha] {i}/{n2}", end="\r")
    cap.release()
    vw.release()
    print(f"[alpha] wrote {ap_path} - coverage min {min(cov2):.2f}% mean {np.mean(cov2):.2f}% "
          f"max {max(cov2):.2f}%; frames with no person detected: {misses}")
    if misses:
        print("[alpha] WARNING: some frames had no detection - lower --conf or use a larger model.")

    print("\nStage on the phone with:")
    print(f'  adb push "{mp}" "/sdcard/Download/Project MIRAGE/input/mask.mp4"')
    print(f'  adb push "{ap_path}" "/sdcard/Download/Project MIRAGE/input/synthetic_alpha_p1.mp4"')


if __name__ == "__main__":
    main()
