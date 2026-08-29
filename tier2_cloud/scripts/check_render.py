#!/usr/bin/env python3
"""Three gates that separate a bundle the generator can animate from one it cannot (§B.57/§B.60).

c2's cloud render came back `status: success`, the phone ran all three phases without an error, and
429 s of device time produced a composite with no character in it. Nothing between the sampler and
the final video asserted that a person had been generated, and nothing before the upload asserted
that a person COULD be. These are the cheapest assertions that separate the observed cases.

  BEFORE queuing (--pre):  can the generator possibly draw a person here?
      The decisive quantity is NOT how tall the skeleton is. It is whether the skeleton is INSIDE
      the hole the plate allows the sampler to repaint. Measured on the artifact the generator
      actually receives - the rendered `pose_sticks` video, not `pose.json`:

          stick ink inside the grown+blockified mask      outcome
          c1   /p1   99.99 %  (p05 100.0 %)               PERSON
          c3   /p1   99.13 %  (p05  92.8 %)               people
          c3   /p2  100.00 %  (p05 100.0 %)               people
          c2   /p1   57.75 %  (p05   0.0 %)               a metal cylinder, every frame

      On 49 of c2's 140 frames NOT ONE stick pixel lands inside that hole (and on 51 of 140 the
      skeleton's horizontal span misses the raw mask box entirely). The generator
      obeyed the mask perfectly (99.3 % of what it drew is inside it) and had no person-shaped
      conditioning in there to obey, so it inpainted the rectangle as scene furniture.

  AFTER rendering (--post): did the RIGHT person actually come out?
      The alpha is computed by DETECTING A PERSON IN THE GENERATED VIDEO (WanVideoDecode ->
      PoseAndFaceDetection -> Sam2Segmentation -> GrowMaskWithBlur), so when generation fails the
      alpha degenerates rather than erroring, and when generation produces the wrong number of
      people it picks one with no notion of which is meant. Coverage catches the first (c1 8.0 %
      of frame, c2 89.2 %); the identity check catches the second (§B.57e).

  --selftest re-measures every arm on disk and asserts the recorded verdict. A gate with no
  positive control measures nothing (§A.6o-1b); this one carries three positives and one negative
  and refuses to be trusted without them.

  python check_render.py --selftest
  python check_render.py --pre  --clip c2_p08_single --slot p1
  python check_render.py --post --clip c1_p05_single
"""
# cp1252 consoles cannot encode the status glyphs below; without this the script dies on a
# print AFTER doing its work, which reads as a failed run. See scripts/utilities/console_safe.py.
import os as _os, sys as _sys
_sys.path.insert(0, _os.path.join(_os.path.dirname(_os.path.abspath(__file__)),
                                  "..", "..", "scripts", "utilities"))
import console_safe  # noqa: F401,E402
import argparse
import glob
import io
import json
import os
import sys

import cv2
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))

# --- post-render: plausible character coverage, as a fraction of frame -------------------------
# c1 (good) = 0.080, c2 (dead) = 0.892. The band is deliberately wide: it is there to catch a
# degenerate matte, not to police composition.
ALPHA_MIN, ALPHA_MAX = 0.005, 0.400

# --- pre-render: is the conditioning inside the repaintable hole? ------------------------------
# HARD REFUSAL. Set from four measured arms with a 0.35 absolute margin between the worst PERSON
# arm (0.9913) and the best OBJECT arm (0.6423 across five c2 Tier-1 draws). Not a guess and not
# a single-sided threshold: both controls are re-measured by --selftest.
INK_MIN, INK_P05_MIN = 0.95, 0.75
# --- pre-render: is it big enough? -------------------------------------------------------------
# WARN ONLY, and honestly uncalibrated: the drawn skeleton filled 0.414 / 0.619 / 0.729 of its
# mask box on the three arms that produced people and 0.285-0.324 on the five that did not, but
# every failing arm ALSO failed the registration gate, so scale has never been varied on its own.
# n=1 either side of the boundary. §B.60 names the render that would calibrate it.
SCALE_WARN = 0.40

# GrowMaskWithBlur #224 (expand 10, blur 4) -> BlockifyMask #225 (16), as wired in
# `V9_GAIT_A_g9proj` / `RUN3_c*.api.json`. The sampler may only repaint inside this.
GROW_EXPAND, GROW_BLUR, BLOCK = 10, 4, 16

# Measured 2026-08-08 from the artifacts in this directory. `--selftest` recomputes every number
# and asserts the verdict, so a code change that silently breaks a gate cannot pass review.
CONTROLS = [
    # clip, slot, expected verdict, what the render actually did
    ("c1_p05_single", "p1", "PASS", "rendered a person; alpha 7.77 % of frame; delivered"),
    ("c3_p20p21_two", "p1", "PASS", "generated people (the matte then picked the wrong one, "
                                    "which is the POST identity gate's job, not this one)"),
    ("c3_p20p21_two", "p2", "PASS", "same clip, other slot"),
    ("c2_p08_single", "p1", "REFUSE", "generated a metal cylinder in all 140 frames"),
]


# ------------------------------------------------------------------ shared readers
def read_gray(path, cap=400):
    c = cv2.VideoCapture(path)
    o = []
    while len(o) < cap:
        ok, f = c.read()
        if not ok:
            break
        o.append(cv2.cvtColor(f, cv2.COLOR_BGR2GRAY))
    c.release()
    return o


def grow_blockify(m):
    """The mask the SAMPLER sees: GrowMaskWithBlur(expand,blur) -> BlockifyMask(block)."""
    k = np.ones((2 * GROW_EXPAND + 1, 2 * GROW_EXPAND + 1), np.uint8)
    g = cv2.dilate(m.astype(np.uint8), k)
    g = cv2.GaussianBlur(g.astype(np.float32), (0, 0), GROW_BLUR) > 0.5
    h, w = g.shape
    gb = g[:h // BLOCK * BLOCK, :w // BLOCK * BLOCK]
    gb = gb.reshape(h // BLOCK, BLOCK, w // BLOCK, BLOCK).max(axis=(1, 3))
    out = np.zeros_like(g)
    out[:h // BLOCK * BLOCK, :w // BLOCK * BLOCK] = np.kron(gb, np.ones((BLOCK, BLOCK), bool))
    return out


def alpha_coverage(path, cap=400):
    fr = [float((g > 127).mean()) for g in read_gray(path, cap)]
    return np.array(fr) if fr else None


def centroid_x(path, thr, cap=400):
    xs = []
    for g in read_gray(path, cap):
        m = g > thr
        xs.append(float(np.where(m)[1].mean()) if m.sum() > 200 else np.nan)
    return np.array(xs)


# ------------------------------------------------------------------ PRE
def pre_measure(clip, slot, bundle=""):
    """Everything the pre-gate decides on. Pure measurement, no verdict."""
    d = bundle or os.path.join(HERE, clip, "to_cloud")
    sticks = sorted(glob.glob(os.path.join(d, f"pose_sticks_{slot}_*.mp4")))
    face = sorted(glob.glob(os.path.join(d, f"facemesh_{slot}_*.mp4")))
    mask_p = os.path.join(HERE, "phone_in", clip, "input", "mask.mp4")
    if not sticks:
        return dict(error=f"no pose_sticks_{slot}_*.mp4 in {d}")
    if not os.path.exists(mask_p):
        return dict(error=f"no emitted mask at {mask_p}")
    mk = read_gray(mask_p)
    st = read_gray(sticks[0])
    fm = read_gray(face[0]) if face else []
    ink, scale, fink = [], [], []
    for i in range(min(len(mk), len(st))):
        m = mk[i] > 127
        s = st[i] > 25
        if m.sum() < 50 or s.sum() < 50:
            continue
        gb = grow_blockify(m)
        ink.append(float((s & gb).sum()) / float(s.sum()))
        sy, _ = np.where(s)
        my, _ = np.where(m)
        scale.append((sy.max() - sy.min() + 1) / float(my.max() - my.min() + 1))
        if i < len(fm):
            f = fm[i] > 25
            if f.sum() >= 50:
                fink.append(float((f & gb).sum()) / float(f.sum()))
    if not ink:
        return dict(error="no frame had both a mask and a skeleton")
    ink, scale = np.array(ink), np.array(scale)
    r = dict(n=len(ink), ink_in_hole=float(ink.mean()), ink_p05=float(np.percentile(ink, 5)),
             frames_zero_overlap=int((ink <= 0.001).sum()),
             scale=float(scale.mean()), scale_p05=float(np.percentile(scale, 5)))
    if fink:
        r["face_in_hole"] = float(np.mean(fink))
        r["face_p05"] = float(np.percentile(fink, 5))
    return r


def pre(clip, slot="p1", bundle="", quiet=False):
    """0 = go, 1 = REFUSE (do not upload), 2 = warn."""
    r = pre_measure(clip, slot, bundle)
    if "error" in r:
        print(f"  PRE GATE: FAIL - {r['error']}")
        return 1
    hard = r["ink_in_hole"] >= INK_MIN and r["ink_p05"] >= INK_P05_MIN
    warn = r["scale"] >= SCALE_WARN
    if not quiet:
        print(f"{clip}/{slot}  (n={r['n']} frames)")
        print(f"  REGISTRATION  stick ink inside the repaintable hole: "
              f"mean {100*r['ink_in_hole']:.2f}%  p05 {100*r['ink_p05']:.1f}%  "
              f"-> {'PASS' if hard else 'REFUSE'}   "
              f"(need >= {100*INK_MIN:.0f}% / {100*INK_P05_MIN:.0f}%; "
              f"person arms 99.99 / 99.13 / 100.00%, the c2 object arm 57.75%)")
        if r["frames_zero_overlap"]:
            print(f"     {r['frames_zero_overlap']}/{r['n']} frames have ZERO overlap between the "
                  f"skeleton and the mask - on those frames the plate tells the sampler to repaint "
                  f"a rectangle it has no person-shaped conditioning for (§B.60)")
        if "face_in_hole" in r:
            print(f"  FACE          facemesh ink inside the hole: mean {100*r['face_in_hole']:.1f}%"
                  f"  p05 {100*r['face_p05']:.1f}%   (person arms 100.0 / 99.7 / 100.0%)")
        print(f"  SCALE (warn)  drawn skeleton height / mask box height: {100*r['scale']:.1f}%"
              f"  -> {'ok' if warn else 'WARN'}   "
              f"(person arms 41.4 / 61.9 / 72.9%, the c2 object arm 31.3%; "
              f"UNCALIBRATED - never varied independently of registration)")
        if not hard:
            print("     🔴 DO NOT UPLOAD. The generator will be asked to fill a hole it has no "
                  "person in. §B.57 is what that costs: a successful render, a confident matte, "
                  "and no character.")
    return 0 if (hard and warn) else (1 if not hard else 2)


# ------------------------------------------------------------------ POST
def identity(alpha_path, clip, slot):
    """Does the matte follow the person the CONDITIONING drove, or a different one?

    §B.57 again, one layer up. The alpha comes from running a person detector over the GENERATED
    video, and nothing ties that detection back to the slot whose sticks drove the sampler. On a
    two-person plate the generator fills BOTH boxes, the detector picks one, and it picked the
    WRONG one on RUN3_c3_male in 140/140 frames - the composite would have pasted an unconditioned
    hallucination and dropped the pose-driven character entirely.

    Skipped (returns True) when the clip has one slot, where there is nothing to confuse.
    """
    d = os.path.join(HERE, clip, "to_cloud")
    sticks = {os.path.basename(p).split("_")[2]: p
              for p in sorted(glob.glob(os.path.join(d, "pose_sticks_p*_00002.mp4")))}
    if len(sticks) < 2:
        return True
    a = centroid_x(alpha_path, 127)
    cs = {s: centroid_x(p, 25) for s, p in sticks.items()}
    n = min([len(a)] + [len(v) for v in cs.values()])
    a = a[:n]
    ok_frames, tot = 0, 0
    for i in range(n):
        ds = {s: abs(a[i] - v[i]) for s, v in cs.items() if not (np.isnan(a[i]) or np.isnan(v[i]))}
        if len(ds) < 2:
            continue
        tot += 1
        ok_frames += (min(ds, key=ds.get) == slot)
    if not tot:
        return True
    frac = ok_frames / tot
    print(f"  IDENTITY GATE: matte follows the driven slot {slot} in {ok_frames}/{tot} frames "
          f"({100*frac:.0f}%) -> {'PASS' if frac >= 0.7 else 'FAIL'}")
    if frac < 0.7:
        others = {s: float(np.nanmean(v)) for s, v in cs.items()}
        print(f"     alpha mean x {np.nanmean(a):.1f}; sticks mean x " +
              ", ".join(f"{s}={v:.1f}" for s, v in sorted(others.items())))
        print("     the composite would paste the WRONG character - the one the sticks did not "
              "drive (§B.57e)")
    return frac >= 0.7


def matte_selects_what_was_drawn(clip, alpha_path, person_path):
    """Diagnostic (not a gate, n=3): does the matte select the pixels the generator CHANGED?

    The generated video's background IS the lightmap it was handed, so |generated - lightmap|
    isolates what the model actually painted. Measured:
        c1        0.919 white / 0.001 black - the matte IS the character
        c2        0.000 white / 0.543 black - the matte is its COMPLEMENT. SAM2, given a
                                               degenerate bbox, returned the BACKGROUND as the
                                               segment. "89 % white" is not merely too big: it
                                               is inverted.
        c3_male   0.346 white / 0.131 black - the matte is ONE of the two painted characters,
                                               which is §B.57e's wrong-person failure seen from
                                               the pixel side.
    Not promoted to a gate: three points, and the c3 value depends on how many characters the
    generator invented, so its band is unknown.
    """
    lm = os.path.join(HERE, clip, "to_cloud", "light_map.mp4")
    if not (os.path.exists(lm) and os.path.exists(person_path)):
        return None
    cp, cl, ca = (cv2.VideoCapture(person_path), cv2.VideoCapture(lm), cv2.VideoCapture(alpha_path))
    iw, ib, k = [], [], 0
    while k < 400:
        okp, p = cp.read()
        okl, l = cl.read()
        oka, a = ca.read()
        if not (okp and okl and oka):
            break
        k += 1
        if l.shape[:2] != p.shape[:2]:
            l = cv2.resize(l, (p.shape[1], p.shape[0]))
        obj = np.abs(p.astype(int) - l.astype(int)).mean(2) > 28
        if obj.sum() < 200:
            continue
        w = cv2.cvtColor(a, cv2.COLOR_BGR2GRAY) > 127
        iw.append((obj & w).sum() / (obj | w).sum())
        ib.append((obj & ~w).sum() / (obj | ~w).sum())
    for c in (cp, cl, ca):
        c.release()
    if not iw:
        return None
    return float(np.mean(iw)), float(np.mean(ib))


def post(alpha_path, quiet=False, clip="", slot="p1"):
    """True if this render plausibly contains the RIGHT character. Prints the numbers either way."""
    cov = alpha_coverage(alpha_path)
    if cov is None:
        print(f"  ALPHA GATE: FAIL - {os.path.basename(alpha_path)} has no readable frames")
        return False
    m = float(cov.mean())
    ok = ALPHA_MIN <= m <= ALPHA_MAX
    if not quiet:
        print(f"  ALPHA GATE: mean coverage {100*m:.2f}% of frame "
              f"(band {100*ALPHA_MIN:.1f}-{100*ALPHA_MAX:.1f}%; c1 8.0% good, c2 89.2% dead) "
              f"-> {'PASS' if ok else 'FAIL'}")
        if not ok:
            why = ("almost the whole frame is marked as character - the generator very likely "
                   "produced no person and the matte degenerated (§B.57)" if m > ALPHA_MAX else
                   "almost nothing is marked as character - the generator produced nothing to paste")
            print(f"     {why}")
    if clip:
        pp = sorted(glob.glob(os.path.join(HERE, clip, "from_cloud*", "synthetic_person_p1*.mp4")))
        r = matte_selects_what_was_drawn(clip, alpha_path, pp[0]) if pp else None
        if r and not quiet:
            print(f"  MATTE/OBJECT (diagnostic, n=2): IoU(painted pixels, matte WHITE) {r[0]:.3f}, "
                  f"with matte BLACK {r[1]:.3f}   (c1 0.919 / 0.001; c2 0.000 / 0.543 = INVERTED)")
    if ok and clip:
        ok = identity(alpha_path, clip, slot) and ok
    return ok


# ------------------------------------------------------------------ SELFTEST
def selftest():
    """Re-measure every arm on disk and assert the recorded verdict - positives AND negatives."""
    print("PRE-GATE SELFTEST - three positive controls, one negative, all measured from disk\n")
    print(f"{'arm':22s} {'ink_in_hole':>12} {'p05':>8} {'0-ovlp':>7} {'scale':>7} "
          f"{'verdict':>8} {'expected':>9}  reality")
    bad = 0
    for clip, slot, want, note in CONTROLS:
        r = pre_measure(clip, slot)
        if "error" in r:
            print(f"{clip+'/'+slot:22s} MISSING - {r['error']}")
            bad += 1
            continue
        code = pre(clip, slot, quiet=True)
        got = {0: "PASS", 1: "REFUSE", 2: "WARN"}[code]
        okmark = "ok " if got == want else "🔴 "
        bad += got != want
        print(f"{clip+'/'+slot:22s} {100*r['ink_in_hole']:11.2f}% {100*r['ink_p05']:7.1f}% "
              f"{r['frames_zero_overlap']:>3}/{r['n']:<3} {100*r['scale']:6.1f}% "
              f"{got:>8} {want:>9}  {okmark}{note}")
    print()
    if bad:
        print(f"🔴 {bad} control(s) disagree with the recorded verdict - the gate is NOT validated.")
    else:
        print("All controls reproduce. The gate separates the arms that produced people from the "
              "one that produced an object, with both directions exercised.")
    return 1 if bad else 0


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--pre", action="store_true")
    ap.add_argument("--post", action="store_true")
    ap.add_argument("--selftest", action="store_true")
    ap.add_argument("--clip", default="")
    ap.add_argument("--alpha", default="")
    ap.add_argument("--bundle", default="", help="override the to_cloud dir the pre-gate reads")
    ap.add_argument("--slot", default="p1", help="which slot's sticks drove this render")
    a = ap.parse_args()
    if a.selftest:
        return selftest()
    if a.pre:
        return pre(a.clip, a.slot, a.bundle)          # 1 = REFUSE, 2 = warn, callers may proceed
    if a.post:
        p = a.alpha or (sorted(glob.glob(os.path.join(HERE, a.clip, "from_cloud*",
                                                      "synthetic_alpha_p1*.mp4")) or [""])[0])
        if not p:
            sys.exit("no synthetic_alpha_p1*.mp4 found")
        return 0 if post(p, clip=a.clip, slot=a.slot) else 1
    sys.exit("pass --pre, --post or --selftest")


if __name__ == "__main__":
    sys.exit(main())
