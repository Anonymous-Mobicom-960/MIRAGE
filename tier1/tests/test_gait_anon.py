"""Durable tests for mirage.gait_anon -- the MIRAGE gait-defence adapter.

Run with pytest, or standalone:

    python scripts/test_gait_anon.py          # from tier1/
    pytest scripts/test_gait_anon.py -q

The first two tests are REGRESSION tests for a crash that was reproduced on this
host before the adapter existed: dense zero rows reaching the vendored
`group_lengths()` drove a median bone length to 0.0, `len_factors =
target / (0 + 1e-6)` followed, and the emitted coordinates came out at 1.02e10 px
(see `test_raw_vendored_call_still_poisons`, which asserts the bug is real rather
than assuming it).
"""
import os
import sys

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "src"))

from mirage.gait_anon import (  # noqa: E402
    GaitAnonError,
    anonymize_export_rows,
)
from mirage.vendor import mirage_edge as V  # noqa: E402

# A plausible standing COCO-17 skeleton in native pixels (640x640 canvas).
BASE = np.array([
    [300, 100], [295, 95], [305, 95], [288, 98], [312, 98],
    [270, 150], [330, 150], [255, 210], [345, 210], [250, 265], [350, 265],
    [285, 260], [315, 260], [283, 350], [317, 350], [281, 440], [319, 440],
], dtype=np.float64)
FRAME_WH = (640, 640)
FPS = 30.0


def make_row(t, dx=0.0, conf=0.9, walk=True):
    """One present (17, 3) float32 row: the base skeleton, walking and swaying."""
    off = np.array([t * 0.7 + dx, 0.0]) if walk else np.array([dx, 0.0])
    xy = BASE + off
    if walk:                                    # a little limb motion to warp
        xy[[9, 10], 1] += 6.0 * np.sin(t / 4.0)
        xy[[15, 16], 0] += 4.0 * np.cos(t / 4.0)
    row = np.zeros((17, 3), dtype=np.float32)
    row[:, :2] = xy
    row[:, 2] = conf
    return row


def absent_row():
    """Exactly what pipeline.py:1172 appends for an unoccupied slot."""
    return np.zeros((17, 3), dtype=np.float32)


def build(n_frames, present_ranges, ident="A"):
    """One slot's rows: present inside any (lo, hi) range, absent elsewhere."""
    rows, ids = [], []
    for t in range(n_frames):
        if any(lo <= t < hi for lo, hi in present_ranges):
            rows.append(make_row(t))
            ids.append(ident)
        else:
            rows.append(absent_row())
            ids.append(None)
    return rows, ids


def _report(name, **kv):
    print(f"  {name:<46}" + "  ".join(f"{k}={v}" for k, v in kv.items()))


def _assert_bound_where_it_is_live(kp_rows, id_rows, seed=None, **kw):
    """Assert "no coordinate blow-up" IN THE ONLY CONFIGURATION WHERE IT CAN FAIL.

    🔴 `assert max |coord| <= 1e4` is VACUOUS while `frame_wh` is passed. The vendored code
    `np.clip()`s to the canvas before returning (pose_anon_edge.py:2426), so that assertion
    holds just as well for a 1.0e10 px blow-up -- which is not a theory, it is measured by
    `test_frame_wh_clipping_hides_the_poisoning_signature` below (1.0e10 px with no canvas
    becomes exactly 640.0 px with one). Four tests here asserted precisely that while passing
    FRAME_WH, so they would have gone on passing with the present-frame compaction deleted.

    This runs the SAME rows again with `frame_wh=None`, where nothing clips first and the
    adapter's own output bound is live -- checked explicitly via `output_bound_effective`, so
    the test fails loudly if that ever stops being the live configuration. Returns the max
    |coord| measured there.
    """
    out, prov = anonymize_export_rows(kp_rows, id_rows, FPS, frame_wh=None, seed=seed, **kw)
    assert prov["output_bound_effective"] is True, \
        "frame_wh=None must leave the adapter's coordinate bound live, or this is vacuous again"
    worst = max(float(np.abs(np.stack(rows)[:, :, :2]).max()) for rows in out)
    assert worst <= 1e4, f"coordinate blow-up (no canvas to hide it, so this is real): {worst}"
    return worst


# --------------------------------------------------------------------------- (0) the bug
def test_raw_vendored_call_still_poisons():
    """The crash the adapter exists to prevent is REAL: assert it, do not assume it.

    Calls the vendored function the naive way -- dense rows, zeros for absent
    frames -- and asserts the emitted coordinates blow up.
    """
    T, S = 60, 1
    pose_log = []
    for t in range(T):
        if 20 <= t < 40:
            row = make_row(t)
            kp = [[0.0, 0.0]] * 133
            sc = [0.0] * 133
            for j in range(17):
                kp[j] = [float(row[j, 0]), float(row[j, 1])]
                sc[j] = float(row[j, 2])
        else:
            kp, sc = [[0.0, 0.0]] * 133, [0.0] * 133
        pose_log.append([{"kp": kp, "score": sc}])
    old = os.environ.get("MIRAGE_GAIT_PRESET")
    os.environ["MIRAGE_GAIT_PRESET"] = "e2"
    try:
        out = V.anonymize_pose_log(pose_log, "L4", FPS)
    finally:
        if old is None:
            os.environ.pop("MIRAGE_GAIT_PRESET", None)
        else:
            os.environ["MIRAGE_GAIT_PRESET"] = old
    worst = float(np.abs(np.array([[p["kp"] for p in f] for f in out])).max())
    _report("naive dense call, max |coord|", px=f"{worst:.3e}", slots=S)
    assert worst > 1e6, f"expected the poisoning blow-up, got {worst}"


# --------------------------------------------------------------------------- (a)
def test_zero_slot_is_untouched_and_nothing_explodes():
    """VERIFY (a): a 100 % zero slot beside a real one.

    No exception, the zero slot comes back bit-identical, and no coordinate
    anywhere exceeds 1e4 px.
    """
    T = 60
    s0, id0 = build(T, [(0, T)])
    s1 = [absent_row() for _ in range(T)]
    id1 = [None] * T
    out, prov = anonymize_export_rows([s0, s1], [id0, id1], FPS,
                                      frame_wh=FRAME_WH, seed=1234)

    worst = _assert_bound_where_it_is_live([s0, s1], [id0, id1], seed=1234)
    for t in range(T):
        assert out[1][t] is s1[t], "zero slot rows must be returned untouched"
        assert np.array_equal(out[1][t], np.zeros((17, 3), np.float32))
        assert out[1][t].dtype == np.float32
    assert not np.array_equal(np.stack(out[0]), np.stack(s0)), "slot 0 must change"
    assert prov["per_slot_present_frames"] == [T, 0]
    assert len(prov["sequences"]) == 1 and prov["sequences"][0]["slot"] == 0
    assert prov["skipped"] == []
    assert prov["transformed_frames"] == T
    assert prov["raw_passthrough_frames"] == 0
    _report("2 slots, slot1 all-zero: max |coord|", px=f"{worst:.1f}",
            seqs=len(prov["sequences"]), kinds=prov["template_kinds"])


def test_partial_presence_does_not_poison():
    """The exact configuration that produced 1.02e10 px, now through the adapter."""
    T = 60
    rows, ids = build(T, [(20, 40)])
    out, prov = anonymize_export_rows([rows], [ids], FPS, frame_wh=FRAME_WH, seed=7)
    worst = _assert_bound_where_it_is_live([rows], [ids], seed=7)
    _report("20/60 present frames: max |coord|", px=f"{worst:.1f}",
            present=prov["per_slot_present_frames"][0])


def test_three_slot_default_export_shape():
    """export_people defaults to 3, so a single-person clip has TWO zero slots."""
    T = 120
    s0, id0 = build(T, [(0, T)])
    zeros = [absent_row() for _ in range(T)]
    nones = [None] * T
    out, prov = anonymize_export_rows([s0, list(zeros), list(zeros)],
                                      [id0, nones, nones], FPS, frame_wh=FRAME_WH)
    worst = _assert_bound_where_it_is_live([s0, list(zeros), list(zeros)],
                                           [id0, nones, nones])
    assert prov["per_slot_present_frames"] == [T, 0, 0]
    assert len(prov["sequences"]) == 1
    _report("export_people=3, 1 real person", px=f"{worst:.1f}",
            present=prov["per_slot_present_frames"])


def test_presence_gaps_stay_one_sequence_and_are_reported():
    """A dropout inside ONE identity is one sequence -- with the gap declared.

    Compaction makes the subsequence non-uniformly sampled in time, which the
    vendored cadence/limb-phase knobs assume it is not. That is the same
    behaviour MIRAGE's own deployment has (slot_groups groups the frames a slot
    exists in), so it is reproduced rather than "fixed" -- but it is REPORTED,
    so an audit can see when the assumption was strained.
    """
    T = 90
    rows, ids = build(T, [(0, 20), (35, 60), (61, 90)])
    out, prov = anonymize_export_rows([rows], [ids], FPS, frame_wh=FRAME_WH, seed=8)
    assert len(prov["sequences"]) == 1
    seq = prov["sequences"][0]
    assert seq["n_present"] == 20 + 25 + 29
    assert seq["n_gaps"] == 2 and seq["max_gap_frames"] == 15
    for t in list(range(20, 35)) + [60]:
        assert np.count_nonzero(out[0][t]) == 0
    _assert_bound_where_it_is_live([rows], [ids], seed=8)
    _report("two dropouts, one identity", seqs=1,
            gaps=seq["n_gaps"], max_gap=seq["max_gap_frames"])


# --------------------------------------------------------------------------- (b)
def test_middle_third_absent_frames_stay_exactly_zero():
    """VERIFY (b): present only in the middle third."""
    T = 90
    rows, ids = build(T, [(30, 60)])
    out, prov = anonymize_export_rows([rows], [ids], FPS, frame_wh=FRAME_WH, seed=99)

    for t in list(range(0, 30)) + list(range(60, 90)):
        assert out[0][t] is rows[t], f"absent frame {t} must be untouched"
        assert np.count_nonzero(out[0][t]) == 0, f"absent frame {t} must stay zero"
    changed = sum(1 for t in range(30, 60)
                  if not np.array_equal(out[0][t][:, :2], rows[t][:, :2]))
    assert changed == 30, f"only {changed}/30 present frames changed"
    assert prov["sequences"][0]["first_frame"] == 30
    assert prov["sequences"][0]["last_frame"] == 59
    assert prov["sequences"][0]["n_present"] == 30
    _report("middle third: absent zero / present changed",
            absent_nonzero=0, changed=f"{changed}/30")


# --------------------------------------------------------------------------- (c)
def test_two_seeds_differ_but_confidence_is_identical():
    """VERIFY (c): different seeds -> different coordinates, identical confidence."""
    T = 60
    rows, ids = build(T, [(0, T)])
    a, _ = anonymize_export_rows([rows], [ids], FPS, frame_wh=FRAME_WH, seed=11)
    b, _ = anonymize_export_rows([rows], [ids], FPS, frame_wh=FRAME_WH, seed=22)
    A, B, R = np.stack(a[0]), np.stack(b[0]), np.stack(rows)

    assert not np.array_equal(A[:, :, :2], B[:, :, :2]), "seeds produced identical poses"
    assert np.array_equal(A[:, :, 2], R[:, :, 2]), "seed 11 altered confidence"
    assert np.array_equal(B[:, :, 2], R[:, :, 2]), "seed 22 altered confidence"
    d = float(np.abs(A[:, :, :2] - B[:, :, :2]).max())
    _report("seed 11 vs 22: max |delta| / conf equal", px=f"{d:.2f}", conf="identical")

    # ...and the same seed twice is reproducible (the pin actually pins).
    c, _ = anonymize_export_rows([rows], [ids], FPS, frame_wh=FRAME_WH, seed=11)
    assert np.array_equal(A, np.stack(c[0])), "pinned seed is not reproducible"


def test_unpinned_calls_use_a_fresh_per_sequence_seed():
    """Default (seed=None) must NOT be deterministic, and must not leak the seed."""
    T = 60
    rows, ids = build(T, [(0, T)])
    a, pa = anonymize_export_rows([rows], [ids], FPS, frame_wh=FRAME_WH)
    b, _ = anonymize_export_rows([rows], [ids], FPS, frame_wh=FRAME_WH)
    assert not np.array_equal(np.stack(a[0]), np.stack(b[0]))
    assert pa["deterministic_seed"] is False
    assert pa["seed_source"] == "vendored_new_clip_seed_per_sequence"
    flat = repr(pa)
    assert "seed" not in {k for k in pa if k.endswith("_value")}
    assert "MIRAGE_TEST_FIXED_SEED" not in pa["env_snapshot"] or \
        pa["env_snapshot"].get("MIRAGE_TEST_FIXED_SEED_set") is False
    _report("unpinned: two calls differ, no seed in provenance",
            deterministic=pa["deterministic_seed"], chars=len(flat))


def test_two_people_in_one_slot_get_two_sequences():
    """A re-let slot must never share one collapse template / seed between people."""
    T = 80
    rows, ids = [], []
    for t in range(T):
        rows.append(make_row(t))
        ids.append("personA" if t < 40 else "personB")
    _, prov = anonymize_export_rows([rows], [ids], FPS, frame_wh=FRAME_WH)
    idents = [s["identity"] for s in prov["sequences"]]
    assert idents == ["personA", "personB"], idents
    assert [s["n_present"] for s in prov["sequences"]] == [40, 40]
    _report("re-let slot split by identity", sequences=idents)


# --------------------------------------------------------------------------- (d)
def test_identity_transform_round_trips_bit_for_bit():
    """VERIFY (d): with the transform replaced by identity, output == input."""
    T = 45
    s0, id0 = build(T, [(0, 20), (25, T)])
    s1, id1 = build(T, [(10, 40)], ident="B")
    ident_fn = (lambda pose_log, level, fps, frame_wh=None, slot_log=None: pose_log)

    out, prov = anonymize_export_rows([s0, s1], [id0, id1], FPS, frame_wh=FRAME_WH,
                                      _anonymize_fn=ident_fn)
    for s, src in enumerate([s0, s1]):
        for t in range(T):
            assert out[s][t].dtype == src[t].dtype
            assert out[s][t].shape == src[t].shape
            assert np.array_equal(out[s][t], src[t]), f"slot {s} frame {t} drifted"
    assert prov["applied"] is True and prov["calls_to_anonymize_pose_log"] == 1
    _report("identity transform round trip", slots=2, frames=T, drift="0 bits")


# --------------------------------------------------------------------------- guards
def test_short_sequences_are_skipped_and_reported():
    T = 40
    rows, ids = build(T, [(5, 8)])           # 3 present frames, floor is 4
    out, prov = anonymize_export_rows([rows], [ids], FPS, frame_wh=FRAME_WH)
    for t in range(T):
        assert out[0][t] is rows[t]
    assert prov["applied"] is False
    assert prov["sequences"] == []
    assert prov["skipped"][0]["reason"] == "below_min_present_frames"
    assert prov["skipped"][0]["n_present"] == 3
    assert prov["raw_passthrough_frames"] == 3
    _report("3-frame sequence skipped", reason=prov["skipped"][0]["reason"],
            raw_frames=prov["raw_passthrough_frames"])


def test_min_present_frac_guard():
    T = 100
    rows, ids = build(T, [(0, 20)])
    _, prov = anonymize_export_rows([rows], [ids], FPS, frame_wh=FRAME_WH,
                                    min_present_frac=0.5)
    assert prov["skipped"][0]["reason"] == "below_min_present_frac"
    assert prov["skipped"][0]["floor_frames"] == 50
    _report("min_present_frac=0.5 on 20/100", reason=prov["skipped"][0]["reason"])


def test_low_confidence_rows_are_not_present():
    T = 40
    rows = [make_row(t, conf=0.1) for t in range(T)]
    out, prov = anonymize_export_rows([rows], None, FPS, frame_wh=FRAME_WH)
    assert prov["per_slot_present_frames"] == [0]
    assert prov["applied"] is False

    # ...and they reach the caller RAW. These rows carry a real person's real
    # coordinates -- only their confidence is low -- so they must be COUNTED, not
    # quietly bucketed with the all-zero absent rows.
    for t in range(T):
        assert out[0][t] is rows[t]
        assert np.count_nonzero(rows[t][:, :2]) > 0
    assert prov["per_slot_low_confidence_frames"] == [T]
    assert prov["per_slot_absent_frames"] == [0]
    assert prov["raw_passthrough_frames"] == T, prov["raw_passthrough_frames"]
    assert prov["raw_passthrough_breakdown"]["low_confidence_rows"] == T
    _report("low-conf rows counted as raw passthrough",
            present=0, raw=prov["raw_passthrough_frames"])


def test_absent_rows_are_not_counted_as_raw_passthrough():
    """An all-zero row holds nothing, so it must NOT inflate the leak count."""
    T = 30
    rows = [absent_row() for _ in range(T)]
    _, prov = anonymize_export_rows([rows], None, FPS, frame_wh=FRAME_WH)
    assert prov["per_slot_absent_frames"] == [T]
    assert prov["per_slot_low_confidence_frames"] == [0]
    assert prov["raw_passthrough_frames"] == 0
    _report("all-zero rows are not a leak", absent=T, raw=0)


# --------------------------------------------------------------------------- degeneracy
def _partial_limb_rows(T=60, degenerate_frac_mod=(10, 6)):
    """Present and CONFIDENT on every frame, but knees+ankles unplaced on most.

    This is what a detector emits when the lower body leaves the shot. Presence
    compaction cannot help: every frame IS present.
    """
    m, k = degenerate_frac_mod
    rows = []
    for t in range(T):
        r = make_row(t)
        if t % m < k:
            r[[13, 14, 15, 16], :2] = 0.0
        rows.append(r)
    return rows


def test_partial_limb_degeneracy_is_caught_under_production_conditions():
    """🔴 The case COMPACTION DOES NOT FIX -- caught with frame_wh set.

    Without the pre-flight `group_lengths` guard this call returns quietly with
    `applied=True` and max |coord| 640.0 px, because the vendored code clips the
    ~1e9 px blow-up to the canvas before the adapter can see it. Production
    always passes frame_wh, so this is the configuration that matters.
    """
    rows = _partial_limb_rows()
    try:
        anonymize_export_rows([rows], None, FPS, frame_wh=FRAME_WH)
    except GaitAnonError as exc:
        msg = str(exc)
        assert "shin" in msg, msg
        assert "MEDIAN length of 0 px" in msg, msg
        _report("partial-limb degeneracy caught (frame_wh SET)",
                groups="shin", raised="GaitAnonError")
    else:
        raise AssertionError(
            "a sequence whose shin median is 0 px was accepted -- the vendored "
            "clip would have hidden a ~1e9 px blow-up behind a 640 px maximum")


def test_degenerate_sequence_can_be_skipped_and_is_reported_as_raw():
    """on_degenerate='skip' keeps the run alive -- and declares the raw leak."""
    T = 60
    rows = _partial_limb_rows(T)
    out, prov = anonymize_export_rows([rows], None, FPS, frame_wh=FRAME_WH,
                                      on_degenerate="skip")
    for t in range(T):
        assert out[0][t] is rows[t], "a skipped sequence must be returned untouched"
    assert prov["applied"] is False
    assert prov["sequences"] == []
    sk = prov["skipped"][0]
    assert sk["reason"] == "degenerate_bone_lengths"
    assert sk["degenerate_groups"] == ["shin"], sk["degenerate_groups"]
    assert sk["min_group_len_px"] == 0.0
    assert prov["raw_passthrough_frames"] == T
    assert prov["raw_passthrough_breakdown"]["degenerate_sequences"] == T
    _report("degenerate seq skipped + declared raw",
            groups=sk["degenerate_groups"], raw=prov["raw_passthrough_frames"])


def test_healthy_sequence_reports_its_margin():
    """A good clip records how close it came, so a near-miss is auditable."""
    T = 60
    rows, ids = build(T, [(0, T)])
    _, prov = anonymize_export_rows([rows], [ids], FPS, frame_wh=FRAME_WH, seed=2)
    seq = prov["sequences"][0]
    assert seq["min_group_len_px"] > 0.0
    assert prov["degenerate_groups_checked"] == 1
    assert "degenerate_groups" not in seq
    _report("healthy sequence margin reported",
            min_group_px=seq["min_group_len_px"], checked=1)


def test_on_degenerate_argument_is_validated():
    rows, ids = build(12, [(0, 12)])
    try:
        anonymize_export_rows([rows], [ids], FPS, on_degenerate="ignore")
    except GaitAnonError as exc:
        assert "on_degenerate" in str(exc)
        _report("bad on_degenerate rejected", value="ignore")
    else:
        raise AssertionError("on_degenerate='ignore' must be rejected")


# --------------------------------------------------------------------------- positive controls
def _naive_dense_pose_log(T=60, lo=20, hi=40):
    """The pose_log a caller builds WITHOUT compaction: dense, zeros when absent."""
    log = []
    for t in range(T):
        kp = [[0.0, 0.0] for _ in range(133)]
        sc = [0.0] * 133
        if lo <= t < hi:
            r = make_row(t)
            for j in range(17):
                kp[j] = [float(r[j, 0]), float(r[j, 1])]
                sc[j] = float(r[j, 2])
        log.append([{"kp": kp, "score": sc}])
    return log


def _vendored_max_coord(pose_log, frame_wh):
    old = os.environ.get("MIRAGE_GAIT_PRESET")
    os.environ["MIRAGE_GAIT_PRESET"] = "e2"
    try:
        out = V.anonymize_pose_log(pose_log, "L4", FPS, frame_wh=frame_wh)
    finally:
        if old is None:
            os.environ.pop("MIRAGE_GAIT_PRESET", None)
        else:
            os.environ["MIRAGE_GAIT_PRESET"] = old
    return float(np.abs(np.array([[p["kp"] for p in f] for f in out])).max())


def test_frame_wh_clipping_hides_the_poisoning_signature():
    """🔴 POSITIVE CONTROL for the vacuity of any output-coordinate assertion.

    The SAME poisoned input measured twice. With no canvas the signature is
    ~1e10 px; with a canvas the vendored clip turns it into exactly 640.0 px.
    Any test that asserts "max |coord| <= 1e4" while passing frame_wh is
    therefore asserting nothing -- it would pass with the compaction removed.
    This test exists so that never gets re-introduced unnoticed.
    """
    unclipped = _vendored_max_coord(_naive_dense_pose_log(), None)
    clipped = _vendored_max_coord(_naive_dense_pose_log(), FRAME_WH)
    assert unclipped > 1e6, unclipped
    assert clipped <= max(FRAME_WH), clipped
    _report("clipping hides the blow-up", no_canvas=f"{unclipped:.3e}",
            with_canvas=f"{clipped:.1f}")


def test_compaction_regression_measured_where_it_is_visible():
    """VERIFY (a), NON-VACUOUSLY: adapter vs naive, both with frame_wh=None.

    Same clip, same preset, same level, same canvas setting -- the only
    difference is the adapter's present-frame compaction. The naive path is the
    positive control: it MUST blow up, or this test is measuring nothing.
    """
    T = 60
    rows, ids = build(T, [(20, 40)])
    naive = _vendored_max_coord(_naive_dense_pose_log(T, 20, 40), None)
    out, prov = anonymize_export_rows([rows], [ids], FPS, frame_wh=None, seed=7)
    adapted = float(np.abs(np.stack(out[0])[:, :, :2]).max())

    assert naive > 1e6, f"positive control failed to poison: {naive}"
    assert adapted <= 1e4, f"adapter blew up: {adapted}"
    assert prov["output_bound_effective"] is True
    _report("compaction, no canvas to hide it", naive=f"{naive:.3e}",
            adapter=f"{adapted:.1f}")


def test_output_bound_is_declared_dead_when_a_canvas_is_passed():
    T = 60
    rows, ids = build(T, [(0, T)])
    _, with_wh = anonymize_export_rows([rows], [ids], FPS, frame_wh=FRAME_WH, seed=1)
    _, no_wh = anonymize_export_rows([rows], [ids], FPS, frame_wh=None, seed=1)
    assert with_wh["output_bound_effective"] is False
    assert "DEAD" in with_wh["output_bound_note"]
    assert no_wh["output_bound_effective"] is True
    _report("output bound self-declares", with_canvas=False, without=True)


def test_seed_value_never_appears_anywhere_in_provenance():
    """Requirement: the seed is used in-process and NEVER logged."""
    import json
    T = 40
    rows, ids = build(T, [(0, T)])
    secret = 1928374651                       # distinctive: no accidental substring
    _, prov = anonymize_export_rows([rows], [ids], FPS, frame_wh=FRAME_WH, seed=secret)
    blob = json.dumps(prov)
    assert str(secret) not in blob, "the pinned seed leaked into the provenance"
    assert prov["deterministic_seed"] is True
    assert prov["env_snapshot"]["MIRAGE_TEST_FIXED_SEED_set"] is True
    assert "MIRAGE_TEST_FIXED_SEED" not in prov["env_snapshot"]
    _report("seed absent from provenance", seed_chars=len(str(secret)),
            provenance_chars=len(blob))


def test_bad_width_raises():
    rows = [np.zeros((17, 2), np.float32) for _ in range(10)]
    try:
        anonymize_export_rows([rows], None, FPS)
    except GaitAnonError as exc:
        assert "confidence" in str(exc)
        _report("(17,2) input rejected", msg=str(exc)[:52] + "...")
    else:
        raise AssertionError("a (17, 2) row must be rejected")

    rows = [np.zeros((133, 3), np.float32) for _ in range(10)]
    try:
        anonymize_export_rows([rows], None, FPS)
    except GaitAnonError as exc:
        assert "expected (17, 3)" in str(exc)
        _report("(133,3) input rejected", msg=str(exc)[:52] + "...")
    else:
        raise AssertionError("a (133, 3) row must be rejected")


def test_bad_dtype_and_nonfinite_and_ragged_raise():
    ok, _ = build(12, [(0, 12)])
    for bad, needle in (
        ([np.zeros((17, 3), np.int32) for _ in range(12)], "floating"),
        ([np.full((17, 3), np.nan, np.float32) for _ in range(12)], "non-finite"),
    ):
        try:
            anonymize_export_rows([bad], None, FPS)
        except GaitAnonError as exc:
            assert needle in str(exc), str(exc)
        else:
            raise AssertionError(f"expected a {needle} rejection")
    try:
        anonymize_export_rows([ok, ok[:5]], None, FPS)
    except GaitAnonError as exc:
        assert "ragged" in str(exc)
    else:
        raise AssertionError("ragged slots must be rejected")
    _report("dtype / nan / ragged rejected", cases=3)


def test_unknown_preset_and_level_raise():
    rows, ids = build(12, [(0, 12)])
    for kwargs, needle in (({"preset": "not_a_preset"}, "not a known preset"),
                           ({"level": "L9"}, "LEVELS ladder")):
        try:
            anonymize_export_rows([rows], [ids], FPS, **kwargs)
        except GaitAnonError as exc:
            assert needle in str(exc), str(exc)
        else:
            raise AssertionError(f"expected a rejection for {kwargs}")
    _report("unknown preset / level rejected", cases=2)


def test_input_is_never_mutated():
    T = 50
    rows, ids = build(T, [(0, T)])
    before = np.stack(rows).copy()
    anonymize_export_rows([rows], [ids], FPS, frame_wh=FRAME_WH, seed=5)
    assert np.array_equal(np.stack(rows), before), "the input buffer was mutated"
    _report("input buffer unmutated", frames=T)


def test_provenance_shape_and_content():
    T = 60
    rows, ids = build(T, [(0, T)])
    _, prov = anonymize_export_rows([rows], [ids], FPS, frame_wh=FRAME_WH, seed=3)
    import json
    json.dumps(prov)                                  # must be JSON-safe

    assert prov["preset"] == "e2" and prov["level"] == "L4"
    assert prov["template_kinds"] == ["anatomical_extent"], prov["template_kinds"]
    assert prov["collapse_target_kwargs"] == {"scale_from": "extent"}
    assert len(prov["preset_kwargs"]) == 9, prov["preset_kwargs"]
    assert len(prov["anonymize_v2_kwargs"]) == 12, prov["anonymize_v2_kwargs"]
    assert prov["deterministic_seed"] is True
    assert "3" not in json.dumps(prov["env_snapshot"])   # the pinned seed is redacted
    assert prov["env_snapshot"]["MIRAGE_TEST_FIXED_SEED_set"] is True
    assert prov["fps"] == FPS and prov["frame_wh"] == [640, 640]
    _report("provenance keys", n=len(prov),
            kinds=prov["template_kinds"], v2_kwargs=len(prov["anonymize_v2_kwargs"]))


def test_env_is_restored_exactly():
    rows, ids = build(20, [(0, 20)])
    os.environ.pop("MIRAGE_GAIT_PRESET", None)
    os.environ.pop("MIRAGE_TEST_FIXED_SEED", None)
    anonymize_export_rows([rows], [ids], FPS, frame_wh=FRAME_WH, seed=1)
    assert "MIRAGE_GAIT_PRESET" not in os.environ
    assert "MIRAGE_TEST_FIXED_SEED" not in os.environ

    os.environ["MIRAGE_GAIT_PRESET"] = "q1s"
    try:
        _, prov = anonymize_export_rows([rows], [ids], FPS, frame_wh=FRAME_WH)
        assert prov["preset"] == "e2"                       # the argument wins
        assert os.environ["MIRAGE_GAIT_PRESET"] == "q1s"    # ...and is restored
        _, prov2 = anonymize_export_rows([rows], [ids], FPS, frame_wh=FRAME_WH,
                                         preset=None)
        assert prov2["preset"] == "q1s"                     # None => ambient
        assert prov2["preset_source"] == "ambient_env"
    finally:
        os.environ.pop("MIRAGE_GAIT_PRESET", None)
    _report("env pinned then restored exactly", ambient="q1s -> preserved")


def test_empty_buffer_is_a_no_op():
    out, prov = anonymize_export_rows([[], [], []], None, FPS, frame_wh=FRAME_WH)
    assert out == [[], [], []]
    assert prov["n_frames"] == 0 and prov["applied"] is False
    _report("empty buffer", slots=prov["n_slots"], frames=prov["n_frames"])


def test_no_frame_wh_still_bounds_the_output():
    T = 60
    rows, ids = build(T, [(0, T)])
    out, _ = anonymize_export_rows([rows], [ids], FPS, seed=4)
    worst = float(np.abs(np.stack(out[0])[:, :, :2]).max())
    assert worst <= 1e4
    _report("frame_wh=None: bounded by input extent", px=f"{worst:.1f}")


def _main():
    tests = [(n, f) for n, f in sorted(globals().items())
             if n.startswith("test_") and callable(f)]
    print(f"gait_anon: running {len(tests)} tests\n")
    failed = 0
    for name, fn in tests:
        try:
            fn()
            print(f"PASS {name}")
        except Exception as exc:                      # noqa: BLE001 -- test runner
            failed += 1
            print(f"FAIL {name}: {type(exc).__name__}: {exc}")
    print(f"\n{len(tests) - failed}/{len(tests)} passed")
    return 1 if failed else 0


if __name__ == "__main__":
    sys.exit(_main())
