#!/usr/bin/env python3
"""PARITY: the released code still is the code the reported numbers describe.

    python -m pytest tier1/tests/test_shipped_parity.py -q
    python tier1/tests/test_shipped_parity.py            # same checks, no pytest needed

WHY THIS FILE EXISTS
--------------------
Every privacy number in this repository was measured on a specific configuration of a specific
defence. Nothing bound those numbers to the code that shipped, so the two could drift apart
silently -- and they did, twice, in ways that were caught by luck rather than by a test:

  * The angle knob was measured, decided and reported at 14/10 while `LEVELS["L4"]` still said
    20/15, so every artifact was produced by an env-var override and no bundle could say which
    configuration built it (ledger A.2d).
  * `GAIT_PRESETS["g18"]` was approved as "the projection fit", but that phrase named two different
    mechanisms. All eleven decision arms carried one; the preset carried the other. The shipped
    output was 1.74x more different from the real person than what was approved, and bought
    nothing (ledger B.62).

Both are the same failure: the reported arm and the committed arm diverged, and no test noticed.
This file is that test. It is deliberately dumb -- it pins values -- because the failure mode is
not subtle logic, it is a number quietly changing.

WHAT IT CHECKS
--------------
  A. CONFIG PARITY: the shipped constants are the ones the papers and docs name.
  B. GOLDEN VECTORS: both shipped defences, run on deterministic synthetic input, still produce
     bit-identical output.

There is no real footage here and none is needed. The fixtures are generated from a fixed seed, so
this runs anywhere, in CI, with no models, no GPU, no dataset and no privacy surface.

WHEN A GOLDEN TEST FAILS
------------------------
It is not automatically a bug -- an intentional improvement fails it too. But it means the
published numbers no longer describe this code. Re-measure the affected arm, update the ledger, and
then update the digest here IN THE SAME COMMIT, so the pin and the numbers move together.
"""
import hashlib
import os
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.dirname(os.path.dirname(HERE))
EDGE = os.path.join(REPO, "tier1", "src", "edge_runner_pi5")
sys.path.insert(0, EDGE)

MASK_TEMPORAL_WIN = 2      # the SHIPPED frame count; it derives from a duration, so it must be
                           # pinned rather than recomputed (it resolves to 1 at 10 fps)
MASK_SIMPLIFY_EPS = 0.01


# --------------------------------------------------------------------------------------------
# Deterministic fixtures. Nothing here touches the corpus.
# --------------------------------------------------------------------------------------------
def synthetic_masks(n=24, h=240, w=180, seed=20260828):
    """A walking-ish blob: a torso ellipse plus two swinging legs, on a fixed seed.

    It does not need to look like a person. It needs to be non-convex, to move, and to be
    IDENTICAL on every machine -- non-convexity is what makes contour simplification and the
    bbox collapse observable at all.
    """
    import cv2
    rng = np.random.RandomState(seed)
    phase = rng.uniform(0, 2 * np.pi)
    out = []
    for t in range(n):
        m = np.zeros((h, w), np.uint8)
        cx = int(w * 0.5 + 6.0 * np.sin(0.4 * t + phase))
        cy = int(h * 0.42)
        cv2.ellipse(m, (cx, cy), (26, 54), 0, 0, 360, 1, -1)          # torso
        sw = 18.0 * np.sin(0.8 * t + phase)
        for side in (-1, 1):
            x2 = int(cx + side * 14 + sw * side)
            cv2.line(m, (cx + side * 10, cy + 48), (x2, int(h * 0.92)), 1, 11)   # legs
        cv2.circle(m, (cx, cy - 68), 17, 1, -1)                        # head
        out.append(m)
    return out


def synthetic_pose_log(n=40, seed=20260828):
    """A 133-keypoint log in the shape `anonymize_pose_log` expects, one person, fixed seed."""
    rng = np.random.RandomState(seed)
    base = rng.uniform(0.2, 0.8, (17, 2)).astype(np.float32) * np.array([180.0, 240.0], np.float32)
    log = []
    for t in range(n):
        kp = base + np.array([0.6 * t, 3.0 * np.sin(0.5 * t)], np.float32)
        kp = np.concatenate([kp, np.zeros((133 - 17, 2), np.float32)])
        sc = np.concatenate([np.full(17, 0.9, np.float32), np.zeros(133 - 17, np.float32)])
        log.append([{"kp": kp.tolist(), "score": sc.tolist()}])
    return log


def digest(arrays):
    """A stable digest of a list of arrays. Rounded to 4 dp so it is not float-noise fragile."""
    hsh = hashlib.sha256()
    for a in arrays:
        a = np.asarray(a)
        if a.dtype.kind == "f":
            a = np.round(a.astype(np.float64), 4) + 0.0        # +0.0 normalises -0.0 to 0.0
        hsh.update(np.ascontiguousarray(a).tobytes())
        hsh.update(str(a.shape).encode())
    return hsh.hexdigest()[:32]


# --------------------------------------------------------------------------------------------
# A. CONFIG PARITY
# --------------------------------------------------------------------------------------------
def test_shipped_gait_preset_is_e2():
    """`e2` is the owner's shipped arm (2026-08-14) and the DEFAULT, not merely available."""
    import pose_anon_edge as PA
    assert PA.SHIPPED_PRESET == "e2", PA.SHIPPED_PRESET
    got = PA.gait_preset()
    want = PA.GAIT_PRESETS["e2"]
    assert got == want, ("gait_preset() no longer returns the shipped preset.\n"
                         "got:  %r\nwant: %r" % (got, want))


def test_e2_carries_the_mechanisms_it_was_approved_with():
    """B.62 in one assertion.

    `g18` was approved as "the projection fit" and shipped without the mechanism that phrase
    referred to, because two different mechanisms shared the name. Naming each required key
    explicitly is what makes that impossible to repeat: a preset that silently loses one of these
    fails here rather than in a render three weeks later.
    """
    import pose_anon_edge as PA
    e2 = PA.GAIT_PRESETS["e2"]
    want = {"projection_fit": True, "scale_from": "extent", "cadence_root_lock": True,
            "limb_swing_amp": 0.25, "limb_phase_offset_s": 0.35, "limb_phase_amp": 1.80,
            "cadence_amp": 0.0, "angle_groups": ("uarm", "farm")}
    for k, v in want.items():
        assert k in e2, "e2 lost the %r mechanism entirely" % k
        assert e2[k] == v, "e2[%r] is %r, was approved as %r" % (k, e2[k], v)


def test_l4_angles_are_14_10():
    """14/10 is the decided arm: 0.00 % invented hands at 14.13 % gait top-1 (A.2c / B.44).

    20/15 does not fix the hands (29.73 %) and 0/0 costs 14.4 pp for no hands benefit, so both
    neighbours on this knob are known-bad. A silent change here invalidates every gait number.
    """
    import pose_anon_edge as PA
    l4 = PA.LEVELS["L4"]
    assert l4["angle_const_deg"] == 14, l4
    assert l4["angle_drift_deg"] == 10, l4


def test_shipped_mask_shape_mode_is_bbox():
    """`bbox` took the frozen gait model to chance; `displace`, which it replaced, removed ~none
    of the available lift (A.6o). The default must not drift back."""
    import config as C
    assert C.MASK_SHAPE_MODE == "bbox", C.MASK_SHAPE_MODE


# --------------------------------------------------------------------------------------------
# B. GOLDEN VECTORS
# --------------------------------------------------------------------------------------------
GOLDEN_MASK = "5d95ea21a6f02680f4e14266701c706b"      # placeholder, filled by --bless
GOLDEN_POSE = "f3e6199f92476d67199da28efb710500"      # placeholder, filled by --bless


def emitted_masks():
    import mirage_tier1 as ST
    hist, out = [], []
    for i, m in enumerate(synthetic_masks()):
        hist.append(m)
        if len(hist) > MASK_TEMPORAL_WIN:
            hist.pop(0)
        out.append(ST.mask_mitigate(hist, m, MASK_SIMPLIFY_EPS))
    return out


def anonymised_pose(seed="20260828"):
    """The gait defence under a PINNED seed.

    It has to be pinned, because the shipped path deliberately is not deterministic: `new_clip_seed`
    draws fresh OS randomness PER SEQUENCE, so two runs of the same clip differ. That is the
    property that stops the perturbation becoming a linkable pseudo-identity, and
    `test_gait_seed_is_not_deterministic_by_default` below asserts it still holds.

    `MIRAGE_TEST_FIXED_SEED` is the mechanism the module itself provides for controlled A/B work.
    It prints a loud warning and stamps `test_fixed_seed` into the artifact, so an output built
    this way is self-identifying and can never be mistaken for a shipped one.
    """
    import pose_anon_edge as PA
    old = os.environ.get("MIRAGE_TEST_FIXED_SEED")
    os.environ["MIRAGE_TEST_FIXED_SEED"] = seed
    try:
        anon = PA.anonymize_pose_log(synthetic_pose_log(), "L4", fps=10.0)
    finally:
        if old is None:
            os.environ.pop("MIRAGE_TEST_FIXED_SEED", None)
        else:
            os.environ["MIRAGE_TEST_FIXED_SEED"] = old
    return [np.asarray(fr[0]["kp"][:17], np.float32) for fr in anon]


def test_gait_seed_is_not_deterministic_by_default():
    """A PRIVACY property, not a style one.

    If the perturbation were stable across clips, an adversary could characterise it once and
    subtract it forever -- the defence would hand out a linkable pseudo-identity instead of
    removing one. Per-IDENTITY seeding was measured to leak 6-8x chance on every arm (A.2j-2).
    So: with no seed pinned, two runs of identical input MUST differ.
    """
    import pose_anon_edge as PA
    os.environ.pop("MIRAGE_TEST_FIXED_SEED", None)
    assert PA.test_fixed_seed() is None
    a = digest([np.asarray(f[0]["kp"][:17], np.float32)
                for f in PA.anonymize_pose_log(synthetic_pose_log(), "L4", fps=10.0)])
    b = digest([np.asarray(f[0]["kp"][:17], np.float32)
                for f in PA.anonymize_pose_log(synthetic_pose_log(), "L4", fps=10.0)])
    assert a != b, ("two unseeded runs produced IDENTICAL output (%s). The per-sequence seed is "
                    "not being drawn, which re-creates the linkable pseudo-identity that "
                    "new_clip_seed exists to prevent." % a)


def test_pinned_seed_is_reproducible():
    """The pin must actually pin, or the golden below is meaningless."""
    assert digest(anonymised_pose()) == digest(anonymised_pose())


def test_mask_mitigate_golden():
    """The silhouette defence, on fixed input, still emits exactly what it emitted."""
    got = digest(emitted_masks())
    assert got == GOLDEN_MASK, (
        "mask_mitigate output changed: %s != %s.\n"
        "If this was intentional, re-measure the silhouette arms, update the ledger, and re-bless "
        "in the SAME commit:  python tier1/tests/test_shipped_parity.py --bless" % (got, GOLDEN_MASK))


def test_gait_anon_golden():
    """The gait defence at the shipped preset, on fixed input, still emits exactly what it emitted."""
    got = digest(anonymised_pose())
    assert got == GOLDEN_POSE, (
        "anonymize_pose_log output changed: %s != %s.\n"
        "If this was intentional, re-measure the gait arms, update the ledger, and re-bless in the "
        "SAME commit:  python tier1/tests/test_shipped_parity.py --bless" % (got, GOLDEN_POSE))


def test_mask_mitigate_is_inclusion_biased():
    """The hard section-2 guarantee, checked rather than trusted: the emitted mask CONTAINS the
    input on every frame. Mitigation may only add grey. A digest cannot express this, and it is
    the property that actually protects a face."""
    for i, (m, e) in enumerate(zip(synthetic_masks(), emitted_masks())):
        assert np.all(e[m > 0] > 0), "frame %d: mitigation removed coverage" % i


def _bless():
    m, p = digest(emitted_masks()), digest(anonymised_pose())
    src = open(__file__, encoding="utf-8").read()
    src = src.replace('GOLDEN_MASK = "%s"' % GOLDEN_MASK, 'GOLDEN_MASK = "%s"' % m)
    src = src.replace('GOLDEN_POSE = "%s"' % GOLDEN_POSE, 'GOLDEN_POSE = "%s"' % p)
    open(__file__, "w", encoding="utf-8", newline="\n").write(src)
    print("blessed:\n  GOLDEN_MASK = %s\n  GOLDEN_POSE = %s" % (m, p))


if __name__ == "__main__":
    if "--bless" in sys.argv:
        _bless()
        raise SystemExit(0)
    fails = 0
    for name, fn in sorted(globals().items()):
        if name.startswith("test_") and callable(fn):
            try:
                fn()
                print("PASS  " + name)
            except AssertionError as exc:
                fails += 1
                print("FAIL  %s\n      %s" % (name, str(exc).replace("\n", "\n      ")))
    print("\n%d failed" % fails if fails else "\nall parity checks pass")
    raise SystemExit(1 if fails else 0)
