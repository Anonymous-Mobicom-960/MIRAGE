#!/usr/bin/env python3
"""
verify_bundle.py -- check a cloud bundle against the live contract, and MEASURE it.
===================================================================================
Written for the bundles `build_cloud_bundle.py` produces, but it reads nothing but the bundle, so
it also runs against the reference bundle on the network volume
(`runpod-slim/ComfyUI/input/100826_twoperson/`) for a side-by-side.

WHAT IT CHECKS
  1. FILE SET     -- every name the contract requires is present (missing / extra both reported).
  2. GEOMETRY     -- every video is the same frame count / size / fps as the union mask.
  3. PER-SLOT MASKS ARE DISTINCT. 🔴 This is the B.57e regression test. The failure mode is not a
     crash: the two per-slot masks come back BYTE-IDENTICAL when the people touch, both cloud
     renders are then told to paint the same region, and the matte -- derived by DETECTING a person
     in the generated video -- binds to the wrong one. Measured here as identical-frame count plus
     the per-frame overlap in pixels.
  4. STICK CONTAINMENT -- the fraction of the DRAWN skeleton's ink that lands inside the mask the
     generator is allowed to repaint. Ink outside the hole is ink the sampler is told to obey and
     forbidden to paint, which is exactly how run3's c2 lost the figure (B.61a/b: 57.75 % mean,
     49/140 frames at zero). Reported per slot against BOTH the slot's own mask and the union mask,
     at full render resolution -- not on pose.json, which the generator never sees (B.61b).

     🔴 This is an ENGINEERING measurement of where ink falls. It is not a privacy figure and it
     says nothing about re-identifiability.

    python verify_bundle.py --bundle <dir> [--expect-frames 50] [--json out.json]
"""
import argparse
import io
import json
import os
import sys

import cv2
import numpy as np

_SHARED = ["MANIFEST.json", "masked_video_00002.mp4", "mask_00002.mp4", "light_map.mp4",
           "pose_sticks_both_00002.mp4"]


def required_files(n_slots):
    """The contract for a bundle with `n_slots` people.

    DERIVED FROM THE SLOT COUNT, not hardcoded to two. The two-person contract was baked in
    here and a single-person bundle failed with four "missing" files that are correctly
    absent (2026-08-15) -- the live single-person reference on the volume,
    `runpod-slim/ComfyUI/input/100826_p03c02_e2/`, carries exactly 7 objects for one person.
    `pose_sticks_both` stays in the shared set: it is reference-only and is emitted for any
    slot count, and it must NEVER be fed to a slot-bound render (the B.57e condition).
    """
    out = list(_SHARED)
    for s in range(1, n_slots + 1):
        out += ["mask_p%d_00002.mp4" % s, "pose_sticks_p%d_00002.mp4" % s,
                "facemesh_p%d_00002.mp4" % s]
    out += ["reference_p%d_640.png" % s for s in range(1, n_slots + 1)]
    return out


REQUIRED = required_files(2)      # back-compat default; main() re-derives from the MANIFEST
INK_THRESH = 16          # a pixel counts as drawn ink above this luminance on the black canvas
MASK_THRESH = 127


def read_gray(path):
    cap = cv2.VideoCapture(path)
    out = []
    while True:
        ok, f = cap.read()
        if not ok:
            break
        out.append(cv2.cvtColor(f, cv2.COLOR_BGR2GRAY))
    cap.release()
    return out


def probe(path):
    c = cv2.VideoCapture(path)
    d = dict(frames=int(c.get(cv2.CAP_PROP_FRAME_COUNT)), w=int(c.get(3)), h=int(c.get(4)),
             fps=round(float(c.get(cv2.CAP_PROP_FPS)), 4))
    c.release()
    return d


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--bundle", required=True)
    ap.add_argument("--expect-frames", type=int, default=0)
    ap.add_argument("--expect-size", default="")
    ap.add_argument("--expect-fps", type=float, default=0.0)
    ap.add_argument("--json", default="")
    a = ap.parse_args()
    B = a.bundle
    rep = {"bundle": os.path.abspath(B), "checks": {}}

    have = set(os.listdir(B))
    rep["files_present"] = sorted(have)
    # The contract follows THIS bundle's slot count, read from its own MANIFEST.
    n_slots = 2
    try:
        with io.open(os.path.join(B, "MANIFEST.json"), encoding="utf-8") as fh:
            n_slots = max(1, len(json.load(fh).get("slots") or [0, 1]))
    except Exception:
        pass
    REQUIRED = required_files(n_slots)
    rep["slots_expected"] = n_slots
    rep["files_missing"] = [f for f in REQUIRED if f not in have]
    rep["files_extra"] = sorted(f for f in have if f not in REQUIRED)
    rep["checks"]["file_set_complete"] = not rep["files_missing"]

    # ---- geometry -----------------------------------------------------------------------------
    vids = sorted(f for f in have if f.endswith(".mp4"))
    rep["video_probe"] = {v: probe(os.path.join(B, v)) for v in vids}
    ref = rep["video_probe"].get("mask_00002.mp4")
    want_n = a.expect_frames or (ref or {}).get("frames")
    want_wh = ([int(x) for x in a.expect_size.split("x")] if a.expect_size
               else [(ref or {}).get("w"), (ref or {}).get("h")])
    want_fps = a.expect_fps or (ref or {}).get("fps")
    bad = {v: d for v, d in rep["video_probe"].items()
           if not (d["frames"] == want_n and [d["w"], d["h"]] == want_wh
                   and abs(d["fps"] - want_fps) < 1e-3)}
    rep["expected"] = {"frames": want_n, "size": want_wh, "fps": want_fps}
    rep["video_mismatches"] = bad
    rep["checks"]["all_videos_same_geometry"] = not bad

    # ---- per-slot masks distinct (the B.57e regression test) ----------------------------------
    m1p, m2p = os.path.join(B, "mask_p1_00002.mp4"), os.path.join(B, "mask_p2_00002.mp4")
    if os.path.exists(m1p) and os.path.exists(m2p):
        with open(m1p, "rb") as f1, open(m2p, "rb") as f2:
            file_identical = f1.read() == f2.read()
        m1, m2 = read_gray(m1p), read_gray(m2p)
        n = min(len(m1), len(m2))
        ident, over, a1, a2 = 0, [], [], []
        for t in range(n):
            b1, b2 = m1[t] > MASK_THRESH, m2[t] > MASK_THRESH
            ident += bool(np.array_equal(b1, b2))
            over.append(int((b1 & b2).sum()))
            a1.append(int(b1.sum())); a2.append(int(b2.sum()))
        rep["per_slot_masks"] = {
            "file_bytes_identical": bool(file_identical),
            "frames_compared": n,
            "frames_binary_identical": int(ident),
            "overlap_px_total": int(sum(over)), "overlap_px_max_frame": int(max(over)),
            "mask_p1_area_px_mean": float(np.mean(a1)), "mask_p2_area_px_mean": float(np.mean(a2)),
        }
        rep["checks"]["per_slot_masks_not_identical"] = (not file_identical) and ident == 0

    # ---- stick containment --------------------------------------------------------------------
    union = read_gray(os.path.join(B, "mask_00002.mp4")) if "mask_00002.mp4" in have else []
    cont = {}
    for si, sl in enumerate(("p1", "p2")):
        sp = os.path.join(B, "pose_sticks_%s_00002.mp4" % sl)
        mp = os.path.join(B, "mask_%s_00002.mp4" % sl)
        if not (os.path.exists(sp) and os.path.exists(mp)):
            continue
        st, ms = read_gray(sp), read_gray(mp)
        n = min(len(st), len(ms), len(union) or len(st))
        f_own, f_uni, ink_px, zero_own, zero_uni, noink = [], [], [], 0, 0, 0
        for t in range(n):
            ink = st[t] > INK_THRESH
            tot = int(ink.sum())
            ink_px.append(tot)
            if tot == 0:
                noink += 1
                continue
            own = float((ink & (ms[t] > MASK_THRESH)).sum()) / tot
            uni = float((ink & (union[t] > MASK_THRESH)).sum()) / tot if union else float("nan")
            f_own.append(own); f_uni.append(uni)
            zero_own += (own == 0.0); zero_uni += (uni == 0.0)
        cont[sl] = {
            "frames": n, "frames_with_no_ink": noink,
            "ink_px_mean": float(np.mean(ink_px)),
            "in_own_slot_mask_mean": round(float(np.mean(f_own)), 6) if f_own else None,
            "in_own_slot_mask_min": round(float(np.min(f_own)), 6) if f_own else None,
            "in_union_mask_mean": round(float(np.mean(f_uni)), 6) if f_uni else None,
            "in_union_mask_min": round(float(np.min(f_uni)), 6) if f_uni else None,
            "zero_containment_frames_own": int(zero_own),
            "zero_containment_frames_union": int(zero_uni),
        }
    rep["stick_containment"] = cont
    rep["stick_containment_note"] = (
        "fraction of DRAWN stick ink (luminance > %d on the black conditioning canvas) inside the "
        "mask, measured at full render resolution on the emitted videos. Engineering only -- not a "
        "privacy figure." % INK_THRESH)

    rep["all_checks_pass"] = all(v is True for v in rep["checks"].values())
    print(json.dumps(rep, indent=1))
    if a.json:
        json.dump(rep, io.open(a.json, "w", encoding="utf-8"), indent=1)
    return 0 if rep["all_checks_pass"] else 1


if __name__ == "__main__":
    sys.exit(main())
