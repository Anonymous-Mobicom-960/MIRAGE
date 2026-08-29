"""PARITY + CONTRACT tests for the two vendored MIRAGE re-ID defences on this host.

Run standalone (exits non-zero on any failure) or under pytest:

    python scripts/test_gait_mask_anon.py       # from tier1/
    pytest scripts/test_gait_mask_anon.py -q

WHAT THIS FILE IS FOR, AND HOW IT DIFFERS FROM `test_gait_anon.py`
-----------------------------------------------------------------
`test_gait_anon.py` tests the gait ADAPTER's own API surface in depth. This file
tests the five properties that make the integration as a whole trustworthy - 
each one is a claim someone will otherwise take on faith:

  (a) PROVENANCE   every vendored file is byte-identical to its source under
                   `tier1/src/edge_runner_pi5/`, and the sha256
                   that `provenance.vendor_sha256()` stamps into an artifact is
                   the digest of those same bytes. `mask_shape.py` is a header +
                   config shim followed by a VERBATIM BYTE SLICE of
                   `mirage_tier1.py`, so its check is byte-containment of that
                   slice, located by its own banner rather than by a hardcoded
                   line range.
  (b) POISONING    the 1e10 px regression. Checked WITH A POSITIVE CONTROL: the
                   naive call is asserted to still blow up, so "the adapter is
                   fine" cannot be a test that measures nothing. Dense zero rows,
                   short slots and the partial-limb degeneracy (which compaction
                   does NOT fix) are all exercised.
  (c) PER-SEQUENCE SEED
                   two runs over the same clip differ, and NOT by a constant
                   offset - a shared translation would leave the walk itself
                   intact and is exactly what a linkable pseudo-identity looks
                   like. Controlled by a pinned-seed run that IS bit-identical,
                   so the difference is attributable to the seed and not to
                   ambient nondeterminism.
  (d) §2 SUPERSET  the emitted mask never loses a pixel of the mask handed in,
                   and the caller's array is not mutated (this host propagates
                   `last_seg_mask` across frames; mutating it would compound the
                   running max frame over frame).
  (e) OFF == OFF   with both defences off nothing runs and nothing changes:
                   the defence imports are lazy AND guarded, the gait adapter's
                   passthrough returns the SAME array objects (bit-identity, not
                   an equal copy), and the mask wrapper at a null configuration
                   reproduces its input exactly - so every difference an enabled
                   run shows is attributable to a knob, not to the wrapper.

🔴 NO PRIVACY NUMBER MEASURED ON THE MIRAGE HOST DESCRIBES THIS HOST. Nothing
here measures re-ID. Every figure printed below is an engineering property of
the code on SYNTHETIC input.

THRESHOLDS. There is exactly one numeric tolerance in this file
(`_EMIT_QUANTUM_PX` x `_QUANTUM_MARGIN`) and it is derived from the vendored
code's own 2-decimal output rounding, not fitted to a clip.
"""
from __future__ import annotations

import ast
import hashlib
import inspect
import os
import subprocess
import sys

import cv2
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
TIER1 = os.path.dirname(HERE)
SRC = os.path.join(TIER1, "src")
sys.path.insert(0, SRC)

#: Where the MIRAGE Raspberry-Pi-5 edge runner lives -- the upstream of the vendored defence
#: modules. In this repository it ships at tier1/src/edge_runner_pi5/. Override with
#: MIRAGE_SOURCE_DIR for a checkout laid out differently. If it is absent the provenance tests
#: SKIP loudly rather than pass -- an unverifiable copy must never look verified.
MIRAGE_SRC = os.environ.get("MIRAGE_SOURCE_DIR") or os.path.join(
    SRC, "edge_runner_pi5")

VENDOR_DIR = os.path.join(SRC, "mirage", "vendor", "mirage_edge")

from mirage.gait_anon import (  # noqa: E402
    GaitAnonError,
    anonymize_export_rows,
)
from mirage.mask_anon import MaskAnonymizer  # noqa: E402
from mirage.vendor import mirage_edge as V  # noqa: E402

# --------------------------------------------------------------------------- fixtures
#: A plausible standing COCO-17 skeleton in native pixels on a 640x640 canvas.
BASE = np.array([
    [300, 100], [295, 95], [305, 95], [288, 98], [312, 98],
    [270, 150], [330, 150], [255, 210], [345, 210], [250, 265], [350, 265],
    [285, 260], [315, 260], [283, 350], [317, 350], [281, 440], [319, 440],
], dtype=np.float64)
FRAME_WH = (640, 640)
FPS = 30.0
SKELETON_H_PX = float(np.ptp(BASE[:, 1]))       # 345 px - reporting scale only

#: The vendored transform rounds every emitted coordinate to 2 dp
#: (`np.round(kp, 2)`), so 0.01 px is the finest difference it can express.
_EMIT_QUANTUM_PX = 0.01
#: How far above that quantum a difference must sit before it counts as
#: STRUCTURE rather than rounding. 100x is a margin, not a fitted constant: it
#: is two orders of magnitude, and the observed run-to-run residual on this
#: fixture is ~10 px, i.e. another order above the bound.
_QUANTUM_MARGIN = 100.0
_STRUCTURE_PX = _EMIT_QUANTUM_PX * _QUANTUM_MARGIN      # 1.0 px

_LEGS = [13, 14, 15, 16]        # knees + ankles: the `shin`/`thigh` groups


def present_row(t, conf=0.9):
    """One present (17, 3) float32 row - the base skeleton walking and swaying."""
    xy = BASE + np.array([0.7 * t, 0.0])
    xy[[9, 10], 1] += 6.0 * np.sin(t / 4.0)          # wrists
    xy[[15, 16], 0] += 4.0 * np.cos(t / 4.0)         # ankles
    row = np.zeros((17, 3), dtype=np.float32)
    row[:, :2] = xy
    row[:, 2] = conf
    return row


def absent_row():
    """Exactly what `pipeline.py:1172` appends for an unoccupied export slot."""
    return np.zeros((17, 3), dtype=np.float32)


def legless_row(t):
    """Present and confident, but knees+ankles unplaced at (0, 0).

    Routine when the lower body leaves the shot - and the case presence
    compaction cannot fix, because the median that collapses is taken per GROUP
    over frames, not per frame.
    """
    row = present_row(t)
    row[_LEGS, :2] = 0.0
    return row


def pack_person(row):
    """One host row -> one MIRAGE emit person dict (the NAIVE packing).

    Used only by the positive controls, which deliberately hand the vendored
    function what the adapter exists to stop it receiving.
    """
    kp = [[float(row[j, 0]), float(row[j, 1])] if j < 17 else [0.0, 0.0]
          for j in range(133)]
    score = [float(row[j, 2]) if j < 17 else 0.0 for j in range(133)]
    return {"kp": kp, "score": score}


def raw_vendored_max_coord(rows):
    """Max |coordinate| the VENDORED function emits for a dense row list.

    `frame_wh=None` on purpose: with a canvas the vendored code clips to it
    (`pose_anon_edge.py:2426`) and a blow-up becomes silent corruption instead
    of a visible one. The control has to be able to SEE the failure.
    """
    log = [[pack_person(r)] for r in rows]
    out = V.anonymize_pose_log(log, "L4", FPS, frame_wh=None,
                               slot_log=[[0] for _ in rows])
    return max(abs(c) for fr in out for p in fr for xy in p["kp"] for c in xy)


def moving_blob(t, H=160, W=200):
    """A NON-convex, hole-free silhouette that walks across the frame.

    Non-convex on purpose: a rectangle emitted for a rectangular input would
    prove nothing about the shape op.
    """
    m = np.zeros((H, W), np.uint8)
    cx, cy = 40 + 3 * t, 80 + int(10 * np.sin(t / 3.0))
    cv2.ellipse(m, (cx, cy), (14, 40), 0, 0, 360, 1, -1)      # torso
    cv2.circle(m, (cx, cy - 46), 9, 1, -1)                    # head
    cv2.line(m, (cx, cy), (cx + 26, cy + 30), 1, 5)           # an outstretched arm
    return m.astype(bool)


def _sha256(blob):
    return hashlib.sha256(blob).hexdigest()


def _read(path):
    with open(path, "rb") as fh:
        return fh.read()


def _report(name, **kv):
    print(f"    {name:<44}" + "  ".join(f"{k}={v}" for k, v in kv.items()))


class _Skipped(Exception):
    """Standalone-runner equivalent of `pytest.skip`."""


def _skip(msg):
    try:
        import pytest
    except ImportError:
        raise _Skipped(msg)
    pytest.skip(msg)


# =========================================================================== (a)
#: vendored filename -> (source filename, "whole" | "slice")
#:
#: `mask_shape.py` is NOT a whole-file copy: `mirage_tier1.py` is a 2 567-line
#: CLI runner, and only its silhouette section is vendored, appended verbatim
#: under a locally-authored header + config shim (VENDOR.md, "What it is").
VENDORED_FILES = {
    "pose_anon_edge.py": ("pose_anon_edge.py", "whole"),
    "person_slots.py": ("person_slots.py", "whole"),
    "mask_shape.py": ("mirage_tier1.py", "slice"),
}
#: The vendored slice starts at this banner and runs to EOF, so the check needs
#: no hardcoded line range - the range is DERIVED and reported.
SLICE_BANNER = b"# ------------------------- silhouette mitigation"
#: The only file in the vendor package that is locally authored.
NOT_VENDORED = {"__init__.py"}


def test_a1_vendored_files_are_byte_identical_to_source():
    """(a) Every vendored file sha256-matches its source. No file escapes the table.

    The second half matters as much as the first: a vendored file that nobody
    added to `VENDORED_FILES` would sail through a per-entry loop while being
    completely unchecked, so the directory listing is compared to the table.
    """
    found = {f for f in os.listdir(VENDOR_DIR)
             if f.endswith(".py") and f not in NOT_VENDORED}
    unlisted = sorted(found - set(VENDORED_FILES))
    assert not unlisted, (
        f"vendored file(s) {unlisted} are in {VENDOR_DIR} but not in this test's "
        f"VENDORED_FILES table, so their provenance is unchecked. Add them with "
        f"their source file and kind.")
    missing = sorted(set(VENDORED_FILES) - found)
    assert not missing, f"VENDORED_FILES lists {missing}, which are not on disk"

    if not os.path.isdir(MIRAGE_SRC):
        _skip(f"MIRAGE source mirror not found at {MIRAGE_SRC} - set "
              f"MIRAGE_SOURCE_DIR to check the vendored copies. NOT verified.")

    for vname, (sname, kind) in sorted(VENDORED_FILES.items()):
        vpath, spath = os.path.join(VENDOR_DIR, vname), os.path.join(MIRAGE_SRC, sname)
        assert os.path.isfile(spath), f"source {spath} for vendored {vname} is missing"
        vblob, sblob = _read(vpath), _read(spath)

        if kind == "whole":
            assert _sha256(vblob) == _sha256(sblob), (
                f"{vname} DIVERGED from {sname}: vendored {_sha256(vblob)} vs source "
                f"{_sha256(sblob)}. Never edit a vendored file - fix it upstream in "
                f"{MIRAGE_SRC} and re-vendor. Every privacy number quotable for this "
                f"module was measured against the SOURCE bytes.")
            _report(vname, sha256=_sha256(vblob)[:16], bytes=len(vblob), kind="whole-file")
            continue

        # "slice": everything from the banner to EOF must appear VERBATIM,
        # contiguously, inside the source file.
        i = vblob.find(SLICE_BANNER)
        assert i >= 0, (f"{vname} does not contain the slice banner {SLICE_BANNER!r}; "
                        f"its provenance cannot be located")
        tail = vblob[i:]
        j = sblob.find(tail)
        assert j >= 0, (
            f"{vname}'s lifted region is NOT a verbatim byte slice of {sname} any more "
            f"(slice sha256 {_sha256(tail)}, {len(tail)} bytes). Either the vendored "
            f"copy was edited - never do that - or the source moved and the copy must "
            f"be re-taken.")
        first_line = sblob[:j].count(b"\n") + 1
        last_line = first_line + tail.count(b"\n") - (1 if tail.endswith(b"\n") else 0)
        # The header above the banner is local, so it must not smuggle in any of
        # the lifted logic: assert it defines nothing the source also defines.
        header = vblob[:i]
        assert b"def mask_mitigate" not in header and b"def _shape_polys" not in header, (
            f"{vname}'s locally-authored header redefines a lifted symbol; the vendored "
            f"logic would be shadowed by host code")
        _report(vname, slice_sha256=_sha256(tail)[:16], bytes=len(tail),
                source=f"{sname}:{first_line}-{last_line}")


def test_a2_provenance_stamps_the_digest_of_those_same_bytes():
    """(a) The sha256 an artifact records is the digest of the file just checked.

    `TIER1_CONFIG.json` / `manifest.json` name the defence code by digest. If
    that digest came from a DIFFERENT file (e.g. a `MIRAGE_VENDOR_DIR` override
    on someone's shell) the artifact names code that did not run - the §A.2d
    divergence failure mode, one level down.
    """
    from mirage import provenance as P

    for vname in sorted(VENDORED_FILES):
        rec = P.vendor_sha256(vname)
        assert rec["status"] == "ok", f"provenance cannot find vendored {vname}: {rec}"
        disk = os.path.join(VENDOR_DIR, vname)
        assert os.path.samefile(rec["path"], disk), (
            f"provenance would stamp {rec['path']} for {vname}, but the package copy is "
            f"{disk}. An env override must never be able to silently swap the defence "
            f"code out from under a run.")
        assert rec["sha256"] == _sha256(_read(disk)), (
            f"provenance digest for {vname} does not match the file on disk")
        _report(f"provenance/{vname}", sha256=rec["sha256"][:16], bytes=rec["bytes"])

    # And the modules actually imported are the package copies, not an override.
    for mod in ("pose_anon_edge", "mask_shape"):
        src = P.vendor_source(mod)
        if src is not None:
            assert src == "package", (
                f"vendored {mod} was loaded via {src}, not from the package - a "
                f"development path, never a shipping one")


# =========================================================================== (b)
def test_b1_positive_control_dense_zero_rows_still_poison():
    """(b) THE CONTROL. The naive call really does emit ~1e10 px - assert it.

    Without this, every "the adapter is fine" assertion below could be passing
    because the bug is gone, because the fixture is wrong, or because nothing is
    being measured at all. `group_lengths()` (`pose_anon_edge.py:115`) takes an
    UNWEIGHTED median over every frame and never reads confidence, so zero rows
    drive a group median to exactly 0.0 and
    `len_factors = target / (0 + 1e-6)` follows.
    """
    T = 60
    dense = [present_row(t) if t < 20 else absent_row() for t in range(T)]
    worst = raw_vendored_max_coord(dense)
    assert worst > 1e6, (
        f"the poisoning control did NOT reproduce (max |coord| {worst:.4g} px). Either "
        f"the vendored code changed or this fixture no longer drives a group median to "
        f"zero - fix the control before trusting anything below it.")
    _report("raw vendored, 20/60 present", max_coord_px=f"{worst:.4g}")


def test_b2_adapter_survives_dense_zero_rows():
    """(b) The same buffer through the adapter: bounded output, zero slot untouched."""
    T = 60
    slot0 = [present_row(t) if t < 20 else absent_row() for t in range(T)]
    slot1 = [absent_row() for _ in range(T)]        # export_people=2, one person
    out, prov = anonymize_export_rows([slot0, slot1], None, FPS, frame_wh=FRAME_WH)

    arr = np.stack([r[:, :2] for r in out[0]])
    worst = float(np.abs(arr).max())
    assert np.isfinite(arr).all()
    assert worst <= max(FRAME_WH) * 4.0, f"coordinates escaped the canvas bound: {worst}"
    assert prov["applied"] is True and prov["transformed_frames"] == 20
    assert prov["per_slot_present_frames"] == [20, 0]
    assert prov["per_slot_absent_frames"] == [40, 60]
    # The absent frames - and the whole 100 %-zero slot - come back as the SAME
    # objects: nothing was invented for a person who was not there.
    assert all(out[0][t] is slot0[t] for t in range(20, T))
    assert all(out[1][t] is slot1[t] for t in range(T))
    assert prov["raw_passthrough_frames"] == 0, (
        "an all-zero row holds nothing and must not be counted as a raw leak")
    _report("adapter, 20/60 present + empty slot",
            max_coord_px=f"{worst:.1f}", transformed=prov["transformed_frames"],
            absent_rows_identical=True)


def test_b3_adapter_skips_short_slots():
    """(b) A sequence too short to transform is left RAW, and SAID to be raw.

    `DEFAULT_MIN_PRESENT_FRAMES = 4` is the vendored code's own floor
    (`_cadence_warp` :222 / `_limb_phase_warp` :254 both return the input below
    T=4), so a shorter sequence cannot receive the time-domain half of the
    defence anyway - while still being long enough to skew a median bone length.
    """
    T, n_present = 60, 3
    rows = [present_row(t) if t < n_present else absent_row() for t in range(T)]
    out, prov = anonymize_export_rows([rows], None, FPS, frame_wh=FRAME_WH)

    assert prov["applied"] is False
    assert prov["skipped"] and prov["skipped"][0]["reason"] == "below_min_present_frames"
    assert prov["raw_passthrough_frames"] == n_present
    assert prov["raw_passthrough_breakdown"]["short_sequences"] == n_present
    assert all(out[0][t] is rows[t] for t in range(T)), (
        "a skipped sequence must be returned as the SAME array objects - bit-identity, "
        "not a copy that merely compares equal")
    _report("short slot (3 present frames)", applied=prov["applied"],
            raw_frames=prov["raw_passthrough_frames"],
            breakdown=prov["raw_passthrough_breakdown"]["short_sequences"])


def test_b4_adapter_refuses_degenerate_bone_lengths():
    """(b) The case COMPACTION DOES NOT FIX: present on every frame, legs unplaced.

    Compaction removes the all-zero ROW; this degeneracy is per GROUP. The
    control below shows the naive call blows up on exactly this input, so the
    guard is doing work rather than decorating a path that was already safe.
    """
    T = 60
    rows = [legless_row(t) if t % 5 < 3 else present_row(t) for t in range(T)]

    worst = raw_vendored_max_coord(rows)             # POSITIVE CONTROL
    assert worst > 1e6, (
        f"the partial-limb control did NOT reproduce ({worst:.4g} px); the guard below "
        f"would then be testing nothing")

    try:
        anonymize_export_rows([rows], None, FPS, frame_wh=FRAME_WH)
    except GaitAnonError as exc:
        assert "shin" in str(exc) or "thigh" in str(exc), (
            f"the refusal must name the degenerate group; got: {exc}")
    else:
        raise AssertionError(
            "adapter accepted a sequence whose leg joints are unplaced on 60 % of "
            "frames. With frame_wh set the vendored code CLIPS the 1e9 px result to the "
            "canvas, so this is silent corruption, not a crash.")

    out, prov = anonymize_export_rows([rows], None, FPS, frame_wh=FRAME_WH,
                                      on_degenerate="skip")
    assert prov["applied"] is False
    assert prov["raw_passthrough_breakdown"]["degenerate_sequences"] == T
    assert all(out[0][t] is rows[t] for t in range(T))
    _report("degenerate legs (raise / skip)", control_px=f"{worst:.4g}",
            raised=True, skip_raw_frames=prov["raw_passthrough_frames"])


# =========================================================================== (c)
def _run_slot(rows, ids=None, **kw):
    out, prov = anonymize_export_rows([rows], None if ids is None else [ids],
                                      FPS, frame_wh=FRAME_WH, **kw)
    assert prov["applied"] is True
    return np.stack([r[:, :2] for r in out[0]]), prov


def test_c1_two_runs_differ_and_not_by_a_constant_offset():
    """(c) Per-SEQUENCE seed: the same clip twice is not the same skeleton twice.

    "Different" is not enough. A constant translation would leave the walk
    itself - limb phase, swing, bone lengths - bit-identical between the two
    runs, which is precisely the linkable pseudo-identity the per-sequence draw
    exists to destroy. So the single best global translation is SUBTRACTED and
    the residual must still be structural.

    The pinned-seed control makes the attribution sound: with the seed fixed the
    two runs are bit-identical, so the difference measured here comes from the
    seed and not from ambient nondeterminism in the pipeline around it.
    """
    T = 60
    rows = [present_row(t) for t in range(T)]

    a, _ = _run_slot(rows)
    b, _ = _run_slot(rows)
    d = a - b
    assert np.abs(d).max() > 0.0, (
        "two runs produced BIT-IDENTICAL keypoints. The vendored per-sequence "
        "`new_clip_seed()` draw is not reaching the transform - every clip would carry "
        "the same perturbation, which is a stable pseudo-identity.")

    offset = d.reshape(-1, 2).mean(0)                 # the best constant translation
    residual = d - offset
    rms = float(np.sqrt((residual ** 2).sum(-1).mean()))
    assert rms > _STRUCTURE_PX, (
        f"after removing a constant offset of {offset} px the two runs differ by only "
        f"{rms:.3f} px RMS, i.e. within {_QUANTUM_MARGIN:.0f}x the vendored emit quantum "
        f"({_EMIT_QUANTUM_PX} px). The runs differ by a TRANSLATION, not by a different "
        f"gait - the walk itself would still be linkable.")

    # Control: pin the seed and the same call is reproducible to the byte.
    p1, _ = _run_slot(rows, seed=20260814)
    p2, _ = _run_slot(rows, seed=20260814)
    assert np.array_equal(p1, p2), (
        "pinned seed did not reproduce - the run-to-run difference above cannot be "
        "attributed to the seed")

    _report("run-to-run", max_delta_px=f"{np.abs(d).max():.2f}",
            const_offset_px=f"({offset[0]:.2f}, {offset[1]:.2f})",
            residual_rms_px=f"{rms:.2f}",
            frac_of_skeleton_height=f"{rms / SKELETON_H_PX:.3f}")


def test_c2_two_sequences_in_one_call_get_different_perturbations():
    """(c) The draw is per SEQUENCE, not per call - two tracklets never share one.

    A per-call (or per-identity) seed leaks 6-8x chance on every measured MIRAGE
    arm. Two identical input slots are the sharpest test: identical input, so
    any difference in the output is the seed and nothing else.
    """
    T = 60
    rows = [present_row(t) for t in range(T)]
    out, prov = anonymize_export_rows([rows, rows], [["A"] * T, ["B"] * T], FPS,
                                      frame_wh=FRAME_WH)
    assert len(prov["sequences"]) == 2 and prov["calls_to_anonymize_pose_log"] == 1
    s0 = np.stack([r[:, :2] for r in out[0]])
    s1 = np.stack([r[:, :2] for r in out[1]])
    d = s0 - s1
    residual = d - d.reshape(-1, 2).mean(0)
    rms = float(np.sqrt((residual ** 2).sum(-1).mean()))
    assert rms > _STRUCTURE_PX, (
        f"two DIFFERENT people with identical input came out {rms:.3f} px RMS apart "
        f"after removing a constant offset - they are sharing one seed and one collapse "
        f"template inside a single call")
    _report("sequence-to-sequence (one call)", max_delta_px=f"{np.abs(d).max():.2f}",
            residual_rms_px=f"{rms:.2f}", sequences=len(prov["sequences"]))


# =========================================================================== (d)
def test_d_emit_mask_is_strict_superset_and_input_is_not_mutated():
    """(d) §2 on this host: emitted ⊇ handed-in, on every frame, input untouched.

    The non-mutation half is not pedantry. `last_seg_mask` is PROPAGATED across
    frames here (warped forward on skip frames and reused). If `apply()` wrote
    into it, the temporal running-max would compound frame over frame and the
    grey region would grow without bound - which is why the pipeline keeps a
    separate `emit_mask`.

    STRICT superset is asserted because the input is deliberately non-convex: a
    `bbox` emit that added nothing would mean the shape op did not run.
    """
    ma = MaskAnonymizer(shape_mode="bbox", temporal_win=2, seed=7)
    ratios = []
    for t in range(12):
        m = moving_blob(t)
        before = m.tobytes()
        out = ma.apply(m)

        assert out is not m, "apply() returned the caller's own array"
        assert m.tobytes() == before, (
            f"frame {t}: apply() MUTATED the caller's mask. On this host that array is "
            f"propagated to the next frame, so the running max would compound.")
        lost = int(np.count_nonzero(m & ~out))
        assert lost == 0, (
            f"frame {t}: {lost} px of the detected silhouette are NOT covered by the "
            f"emitted mask. §2 says the emitted mask can only ever ADD grey.")
        assert int(np.count_nonzero(out & ~m)) > 0, (
            f"frame {t}: emitted mask equals the input exactly - the bbox shape op did "
            f"not run on a non-convex silhouette")
        # Every emitted component must be a filled axis-aligned rectangle.
        n, _, stats, _ = cv2.connectedComponentsWithStats(out.astype(np.uint8), 8)
        for i in range(1, n):
            assert (stats[i, cv2.CC_STAT_AREA]
                    == stats[i, cv2.CC_STAT_WIDTH] * stats[i, cv2.CC_STAT_HEIGHT]), (
                f"frame {t}: emitted component {i} is not a filled rectangle")
        ratios.append(out.sum() / m.sum())

    st = ma.stats()
    assert st["s2_superset_violations"] == 0 and st["s2_superset_violation_px"] == 0
    assert st["frames"] == 12
    _report("bbox mask, 12 frames", lost_px=0,
            area_ratio=f"{np.mean(ratios):.3f}x [{min(ratios):.3f}, {max(ratios):.3f}]",
            adapter_violations=st["s2_superset_violations"])

    # A detection gap must not retract coverage either: the window carries it over.
    gap = ma.apply(None)
    assert gap is not None and gap.any(), (
        "a drop frame emitted nothing; the temporal running-max should carry the "
        "previous frames' coverage across the gap")
    _report("drop frame (mask=None)", emitted_px=int(gap.sum()),
            counted_as=ma.stats()["frames_none_as_empty"])


# =========================================================================== (e)
DEFENCE_MODULES = ("gait_anon", "mask_anon", "provenance", "vendor")


def test_e1_pipeline_defaults_are_off_and_defence_imports_are_lazy():
    """(e) With mask off and gait_anon left at its default, defence imports stay lazy.

    gait_anon defaults to True as of the paper-alignment change (2026-08):
    the paper's Section 4 states pose anonymization runs
    before egress unconditionally, so the shipped default now matches that
    and the gait defence import IS expected to fire on a defaults-only run.
    mask_shape_mode stays off by default; a bare `process_video()` call
    still must not import mask_anon/vendor.mirage_edge for the mask half.

    Two independent checks, because either alone is weak:
      * STRUCTURAL - every import of a defence module inside `process_video()`
        is nested under an `if` whose test names `gait_anon` or `mask_enabled`.
      * RUNTIME - a fresh subprocess that imports `mirage.pipeline` ends up
        with none of those modules in `sys.modules` (module-scope import only;
        process_video() is never called by importing the module). Run
        out-of-process because this test file has already imported them.
    """
    import mirage.pipeline as PL

    sig = inspect.signature(PL.process_video)
    assert sig.parameters["gait_anon"].default is True, (
        "gait_anon's default changed to True (paper-alignment, 2026-08) -- "
        "if this fails, either the default reverted or this test is stale")
    assert str(sig.parameters["mask_shape_mode"].default).lower() in ("none", "", "off")
    assert sig.parameters["score_binarize"].default is None
    assert sig.parameters["score_binarize_thresh"].default == 0.5   # owner decision

    tree = ast.parse(_read(PL.__file__).decode("utf-8"))
    fn = next(n for n in ast.walk(tree)
              if isinstance(n, ast.FunctionDef) and n.name == "process_video")

    imports = []

    def walk(node, guards):
        if isinstance(node, (ast.Import, ast.ImportFrom)):
            mod = getattr(node, "module", None) or ",".join(a.name for a in node.names)
            if any(t in str(mod) for t in DEFENCE_MODULES):
                imports.append((node.lineno, str(mod), list(guards)))
            return
        if isinstance(node, ast.If):
            test = ast.unparse(node.test)
            for c in node.body:
                walk(c, guards + [test])
            for c in node.orelse:
                walk(c, guards + [f"not({test})"])
            return
        for c in ast.iter_child_nodes(node):
            walk(c, guards)

    for stmt in fn.body:
        walk(stmt, [])

    assert imports, ("found no defence-module import inside process_video() - either "
                     "the wiring is gone or this AST walk stopped matching it")
    for lineno, mod, guards in imports:
        assert any(("gait_anon" in g) or ("mask_enabled" in g) for g in guards), (
            f"pipeline.py:{lineno} imports {mod!r} under guards {guards}, none of which "
            f"mentions gait_anon or mask_enabled. A defaults-only run would execute the "
            f"defence code path.")
        _report(f"import {mod}", line=lineno,
                guarded_by=" && ".join(f"({g})" for g in guards))

    # Also assert nothing at module scope drags the defence in.
    for n in tree.body:
        if isinstance(n, (ast.Import, ast.ImportFrom)):
            mod = getattr(n, "module", None) or ""
            assert not any(t in str(mod) for t in DEFENCE_MODULES), (
                f"pipeline.py imports {mod!r} at module scope; the defence code would "
                f"load on every run")

    probe = (
        "import sys; import mirage.pipeline; "
        "print(','.join(m for m in ('mirage.gait_anon','mirage.mask_anon',"
        "'mirage.provenance','mirage.vendor.mirage_edge','pose_anon_edge') "
        "if m in sys.modules) or 'NONE')")
    env = dict(os.environ, PYTHONPATH=SRC)
    res = subprocess.run([sys.executable, "-c", probe], capture_output=True, text=True,
                         env=env, cwd=TIER1)
    assert res.returncode == 0, f"probe subprocess failed: {res.stderr[-500:]}"
    loaded = res.stdout.strip().splitlines()[-1]
    assert loaded == "NONE", (
        f"importing mirage.pipeline pulled in {loaded} - the defence imports are "
        f"not lazy any more")
    _report("fresh subprocess import", defence_modules_loaded=loaded)


def test_e2_gait_passthrough_is_bit_identity():
    """(e) When the gait transform does not run, the rows that come back ARE the rows.

    This is the property the pipeline's OFF path relies on: with `gait_anon=False`
    it never calls the adapter at all and `kp_src IS export_kp_rows`. Here the
    same identity is asserted where the adapter DOES run but has nothing it may
    transform - same objects, and an input buffer that is byte-for-byte unchanged
    afterwards.
    """
    T = 40
    rows = [absent_row() for _ in range(T)]
    before = [r.tobytes() for r in rows]
    out, prov = anonymize_export_rows([rows], None, FPS, frame_wh=FRAME_WH)

    assert prov["applied"] is False and prov["calls_to_anonymize_pose_log"] == 0
    assert all(out[0][t] is rows[t] for t in range(T))
    assert [r.tobytes() for r in rows] == before, "input buffer was mutated"

    # And the caller's outer list is a NEW list, so writing into the returned
    # structure cannot reach back into the pipeline's buffer.
    assert out is not [rows] and out[0] is not rows
    _report("all-absent buffer", applied=prov["applied"],
            vendored_calls=prov["calls_to_anonymize_pose_log"], rows_identical=True)

    # A present clip must NOT be identity - otherwise the assertion above would
    # pass for a wrapper that never does anything at all.
    live = [present_row(t) for t in range(T)]
    live_before = [r.tobytes() for r in live]
    out2, prov2 = anonymize_export_rows([live], None, FPS, frame_wh=FRAME_WH)
    assert prov2["applied"] is True
    assert any(not np.array_equal(out2[0][t], live[t]) for t in range(T))
    assert [r.tobytes() for r in live] == live_before, (
        "the adapter mutated the caller's buffer on the transforming path; "
        "`_write_export_arrays()` re-transforms it every 30 frames, so that would "
        "double-transform on the next flush")
    # Confidence is copied verbatim, by contract: binarisation is a later stage.
    assert all(np.array_equal(out2[0][t][:, 2], live[t][:, 2]) for t in range(T))
    _report("present buffer (control)", applied=prov2["applied"],
            input_unmutated=True, confidence_untouched=True)


def test_e3_mask_null_config_is_bit_identity():
    """(e) The mask wrapper adds NOTHING of its own at a null configuration.

    `shape_mode="none"`, window 1, no EDGE_EXPAND, eps 0 - every knob off. The
    emitted mask must then equal the input exactly, which is what makes the
    enabled run's growth attributable to the knobs rather than to the wrapper
    (the ring buffer, the uint8 round-trip, the `| sm | cur` re-OR).

    The input is hole-free on purpose: `mask_mitigate` uses RETR_EXTERNAL, so a
    silhouette with an interior hole is legitimately hole-FILLED, and that is a
    property of the vendored code, not of this wrapper.
    """
    ma = MaskAnonymizer(shape_mode="none", temporal_win=1, edge_expand_px=0,
                        simplify_eps=0.0, seed=1)
    for t in range(8):
        m = moving_blob(t)
        out = ma.apply(m)
        assert out.dtype == np.bool_
        assert np.array_equal(out, m), (
            f"frame {t}: the null configuration changed the mask by "
            f"{int(np.count_nonzero(out ^ m))} px - the wrapper is not neutral, so an "
            f"enabled run's area cost cannot be attributed to the shape mode")
    st = ma.stats()
    assert st["area_in_px"] == st["area_out_px"] and st["area_ratio_mean"] == 1.0
    _report("null config, 8 frames", area_ratio=st["area_ratio_mean"],
            in_px=st["area_in_px"], out_px=st["area_out_px"])


# =========================================================================== runner
_SKIP_EXC = [_Skipped]
try:                                            # pytest.skip raises a BaseException
    import pytest as _pytest

    _SKIP_EXC.append(_pytest.skip.Exception)
except Exception:                               # noqa: BLE001 - pytest is optional
    _pytest = None
_SKIP_EXC = tuple(_SKIP_EXC)


def _main():
    # Assertion messages are printed by this runner, and they carry em dashes and
    # section signs. On Windows a piped stdout is cp1252, which cannot encode all
    # of that -- and a UnicodeEncodeError inside the runner would mask the very
    # failure it is reporting. Degrade the character, never the report.
    for stream in (sys.stdout, sys.stderr):
        if hasattr(stream, "reconfigure"):
            try:
                stream.reconfigure(errors="backslashreplace")
            except Exception:                             # noqa: BLE001
                pass

    tests = [(n, f) for n, f in sorted(globals().items())
             if n.startswith("test_") and callable(f)]
    print(f"gait+mask parity/contract suite: {len(tests)} tests")
    print(f"  vendor dir  : {VENDOR_DIR}")
    print(f"  MIRAGE src  : {MIRAGE_SRC}"
          f"{'' if os.path.isdir(MIRAGE_SRC) else '   <- NOT FOUND'}\n")
    failed = skipped = 0
    for name, fn in tests:
        print(f"{name}")
        try:
            fn()
        except _SKIP_EXC as exc:                          # noqa: BLE001
            skipped += 1
            print(f"  SKIP {exc}")
        except Exception as exc:                          # noqa: BLE001 - test runner
            failed += 1
            print(f"  FAIL {type(exc).__name__}: {exc}")
        else:
            print("  PASS")
    print(f"\n{len(tests) - failed - skipped}/{len(tests)} passed, "
          f"{skipped} skipped, {failed} failed")
    if skipped:
        print("*** a SKIP is not a pass: the skipped property is UNVERIFIED.")
    return 1 if failed else 0


if __name__ == "__main__":
    sys.exit(_main())
