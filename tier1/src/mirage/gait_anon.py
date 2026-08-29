"""gait_anon -- host adapter for the vendored MIRAGE gait defence (preset `e2`).

WHAT THIS IS
------------
`src/mirage/vendor/mirage_edge/pose_anon_edge.py` is a BYTE-IDENTICAL copy of the
MIRAGE edge anonymiser and must never be edited. It expects MIRAGE's own emit
format; this pipeline buffers something else. Everything host-specific therefore
lives here:

    MIRAGE emit format                       this host's export buffer
    ------------------------------           --------------------------------
    pose_log[frame][person] = {              export_kp_rows[slot][frame] =
        "kp":    [[x, y]] * 133,                 np.ndarray (17, 3) float32
        "score": [s] * 133,                      columns = [x, y, confidence]
    }                                        absent slot -> np.zeros((17, 3))
    grouping by slot_log (stable ids)        slot re-let after repeated misses

`anonymize_export_rows()` converts one direction, calls the vendored
`anonymize_pose_log` EXACTLY ONCE, and converts back.

🔴 THE ONE THING THIS FILE EXISTS TO PREVENT: BONE-LENGTH POISONING
-------------------------------------------------------------------
`export_people` defaults to 3, so a single-person clip leaves two slots that are
100 % `np.zeros((17, 3))`, and any slot is zero on every frame its person was not
detected. The vendored `group_lengths()` (pose_anon_edge.py:115) takes an
UNWEIGHTED MEDIAN OVER EVERY FRAME IT IS GIVEN and never reads confidence, and
`anonymize_v2` then forms (pose_anon_edge.py:866, applied at :152)

    len_factors[g] = target[g] / (group_lengths[g] + 1e-6)
    emitted_bone   = observed_bone_this_frame * len_factors[g]

so any group whose MEDIAN length is driven to 0.0 amplifies that group's bones by
~1e6 and emits coordinates around 1e10 px. Measured on this host, dense rows with
20 present frames out of 60: **1.2045e10 px**.

Compaction to the present frames (`_present_frames()` below) removes the
all-zero-ROW path. TWO MEASUREMENTS SAY THAT IS NOT ENOUGH, and both are why the
pre-flight guard exists:

  1. 🔴 COMPACTION DOES NOT CLOSE THE CHANNEL. The median is over FRAMES, but the
     degeneracy is per GROUP. A slot present and confident on EVERY frame, whose
     knees+ankles are unplaced `(0, 0)` on 60 % of them -- routine when the lower
     body leaves the shot -- takes `shin` to a median of exactly 0.0. Measured on
     this host, 60/60 present frames: **1.382e9 px**. Nothing about presence
     compaction touches this.
  2. 🔴 PASSING `frame_wh` HIDES IT. The vendored code clips to the canvas
     (pose_anon_edge.py:2426) BEFORE returning, so the output-side coordinate
     bound below cannot see a blow-up whenever a canvas is supplied -- which is
     how production calls this. Same degenerate input as (1), with
     `frame_wh=(640, 640)`: no exception, `applied: True`, max |coord| 640.0 px,
     only 6.9 % of coordinates pinned to a border. SILENT CORRUPTION, not a crash.

So the real guard is `_degenerate_groups()`: before each sequence is sent, the
adapter calls the VENDORED `group_lengths()` on it and refuses any group whose
median is <= 0. That is mechanism-exact (it is the divisor above), it needs no
threshold at all (the degenerate value is EXACTLY 0.0 -- two joints at the same
point), and it is unaffected by clipping. The output-side bound is kept as a
second net, but it is documented as DEAD whenever `frame_wh` is passed; do not
rely on it.

  3. 🔴 AND `_degenerate_groups` IS STILL NARROWER THAN THE FAILURE, so in the
     PRODUCTION configuration (`frame_wh` set) NOTHING refuses a small-but-
     positive median. A shin median of 4 px -- a seated or end-on subject, not an
     exotic input -- emits 2 635 px coordinates on a 640 px canvas, which the
     vendored clip then squeezes to exactly 640: `applied: True`, no exception,
     no warning. Full measured table in `_degenerate_groups`. The adapter
     therefore RECORDS `group_lengths_px` for every transformed sequence and
     MEASURES `canvas_clipped_joints` (0 on a clean sequence), so an audit can
     see it. Choosing a refusal cutoff on `min_group_len_px` is an OWNER POLICY
     CALL and is deliberately not made here.

WHAT IS DELIBERATELY *NOT* DONE HERE
------------------------------------
* No pixel arithmetic of our own. Every coordinate the caller gets back was
  produced by the vendored transform (which rounds to 2 dp, as MIRAGE ships) or
  is the caller's own untouched input.
* No seeding of our own. The vendored `anonymize_pose_log` draws one fresh
  `new_clip_seed()` PER SEQUENCE internally; that per-sequence draw is
  load-bearing (a per-identity seed leaks 6-8x chance on every measured arm), so
  the adapter must not collapse it to one seed per call. The `seed=` argument is
  a TEST-ONLY pin (see its docstring) and its value is never logged or returned.
* No confidence editing. Requirement: the transform does not touch scores, so
  column 2 of every returned row is copied verbatim from the input row. The
  {0,1} binarisation at POSE_THRESH is a separate, later stage.

CALLER CONTRACT (important)
---------------------------
`anonymize_export_rows` is PURE: it never mutates `kp_rows`. The caller MUST keep
its raw buffer and use the returned rows only for writing, because
`_write_export_arrays()` runs every 30 frames (pipeline.py:1203) and re-transforms
the whole buffer each time. Assigning the result back over the raw buffer would
double-transform on the next flush.

Rows that were skipped (too short to transform safely) are returned as the SAME
array objects, so "unchanged" is bit-identity, not a copy that merely compares
equal. 🔴 Those frames therefore reach disk carrying RAW gait; the count is
reported as `provenance["raw_passthrough_frames"]` precisely so an audit can see
it rather than assume zero.

🔴 RAW PASSTHROUGH HAS THREE SOURCES, NOT ONE. A row that is NOT all-zero but
whose every joint sits below `conf_thresh` is a real detection of a real person
carrying real coordinates. It is not "present" (the vendored code would zero
those joints anyway, `_unobs` at pose_anon_edge.py:2309), so it is never
transformed -- and it is returned untouched. Measured on this host: 40 rows at
confidence 0.10 produced 40 frames of real pose on disk. `raw_passthrough_frames`
counts these too, alongside short sequences and (when `on_degenerate="skip"`)
degenerate ones; `raw_passthrough_breakdown` says which is which. An all-zero row
is NOT counted -- there is nothing in it to leak.

NO PRIVACY NUMBER MEASURED ON THE MIRAGE HOST DESCRIBES THIS HOST. The preset
name is recorded so a bundle is self-identifying; that is a provenance claim, not
a re-ID claim.
"""
from __future__ import annotations

import os
import threading

import numpy as np

from .vendor import mirage_edge as _mirage

__all__ = [
    "GaitAnonError",
    "anonymize_export_rows",
    "DEFAULT_PRESET",
    "DEFAULT_LEVEL",
    "DEFAULT_MIN_PRESENT_FRAMES",
    "DEFAULT_MIN_PRESENT_FRAC",
    "DEFAULT_MAX_COORD_MULT",
    "DEFAULT_ON_DEGENERATE",
    "POSE_THRESH",
]

# --------------------------------------------------------------------------- constants
#: Confidence at/above which a joint counts as observed. 0.5 is MIRAGE's own
#: POSE_THRESH and is also the threshold the vendored code uses internally for
#: its `_unobs` gate (pose_anon_edge.py:2309) and for `_observed_shoulder_px`.
#: Keeping one number for all three is what makes "present" here mean the same
#: thing as "observed" down there.
POSE_THRESH = 0.5

#: The measured arm (owner decision 2026-08-14): preset `e2`, level `L4`.
DEFAULT_PRESET = "e2"
DEFAULT_LEVEL = "L4"

#: Shortest subsequence the vendored transform is willing to treat as a gait
#: sequence. NOT fitted to any clip: 4 is the vendored code's OWN floor --
#: `_cadence_warp` (pose_anon_edge.py:222) and `_limb_phase_warp` (:254) both
#: return the input untouched below T=4, so a shorter sequence cannot receive the
#: time-domain half of the defence anyway, while still being long enough to skew
#: a median bone length.
DEFAULT_MIN_PRESENT_FRAMES = 4

#: Fraction-of-clip floor, disabled by default. A person who walks through for a
#: second is a legitimate short tracklet, and skipping them means leaving their
#: RAW gait in the export -- so the default must not silently do that. Hosts that
#: prefer the opposite trade can raise it.
DEFAULT_MIN_PRESENT_FRAC = 0.0

#: Sanity bound on coordinates, as a multiple of the frame size (or, with no
#: frame size given, of the input's own observed extent). Generous on purpose:
#: detector coordinates may legitimately sit a little outside the canvas, but the
#: poisoning signature is 1e10 px, seven orders of magnitude past this.
#: 🔴 THIS BOUND IS DEAD WHENEVER `frame_wh` IS PASSED -- the vendored code clips
#: to the canvas before returning (pose_anon_edge.py:2426), so a poisoned
#: sequence arrives here already squeezed to <= max(w, h). `_degenerate_groups`
#: is the guard that actually holds; this is only a second net for the
#: `frame_wh=None` path.
DEFAULT_MAX_COORD_MULT = 4.0

#: What to do when a sequence's bone lengths are degenerate (see
#: `_degenerate_groups`). "raise" is the default because the alternative is to
#: write a skeleton nobody has validated, or to leave a real person's real gait
#: raw on disk -- both silent. 🔴 NEITHER OPTION IS FREE and this is a policy
#: call, not a technical one: "raise" aborts an export on input that is merely
#: awkward (a subject whose legs leave the shot for most of a flush window),
#: while "skip" keeps the run alive and leaks that sequence RAW -- counted in
#: `raw_passthrough_frames`, never silently.
DEFAULT_ON_DEGENERATE = "raise"

N_BODY = 17            # COCO-17 body block the host buffers
N_WHOLEBODY = 133      # RTMPose wholebody layout the vendored code expects
FACE_SLICE = slice(23, 91)   # 68 face points -- must stay zeroed end to end

#: Preset keys `anonymize_pose_log` POPS before building the `anonymize_v2`
#: kwargs (pose_anon_edge.py:2188-2262). Mirrored here for provenance ONLY -- so
#: the report can separate "went to anonymize_v2" from "selected the collapse
#: target one level up". Nothing is computed from it.
_HOST_LEVEL_PRESET_KEYS = (
    "scale_from", "height_mult", "arm_mult", "leg_mult", "neck_mult",
    "stature_ratio", "height_anchor",
)

#: Environment variables the vendored module reads that can change the emitted
#: pose. Snapshotted into provenance so a run that was steered by ambient env is
#: self-identifying. `MIRAGE_TEST_FIXED_SEED` is reported as a BOOLEAN only.
_ENV_OF_INTEREST = (
    "MIRAGE_GAIT_PRESET", "MIRAGE_POSE_SCALE_FROM", "MIRAGE_HEIGHT_MULT",
    "MIRAGE_ARM_MULT", "MIRAGE_LEG_MULT", "MIRAGE_NECK_MULT",
    "MIRAGE_STATURE_RATIO", "MIRAGE_HEIGHT_ANCHOR", "MIRAGE_HEAD_ANCHOR",
    "MIRAGE_ANATOMICAL_TEMPLATE", "MIRAGE_RESTORE_TORSO_ROLL",
    "MIRAGE_LOWFREQ_AMP", "MIRAGE_ANGLE_CONST", "MIRAGE_ANGLE_DRIFT",
    "MIRAGE_CADENCE_AMP", "MIRAGE_CADENCE_PERIOD_S", "MIRAGE_ASYM_SIGMA",
    "MIRAGE_LR_ASYM_SIGMA", "MIRAGE_LIMB_PHASE_AMP", "MIRAGE_LIMB_PHASE_PERIOD_S",
)

#: The vendored provenance globals, by their real names. They are process-global
#: lists that the vendored code only ever APPENDS to (one entry per transformed
#: sequence) and never resets, so the only correct way to read them is to
#: snapshot the length before the call and slice after it -- which is what
#: `_snapshot_globals` / `_slice_globals` do, under `_CALL_LOCK`.
_PROVENANCE_GLOBALS = {
    "template_kinds": "_TEMPLATE_KIND_USED",
    "stature_scales": "_STATURE_SCALE_USED",
}

#: Serialises env mutation + global-list slicing + the vendored call. The host
#: runs detection/pose/segmentation on a thread pool; without this, a concurrent
#: caller could observe our pinned env or steal our provenance entries.
_CALL_LOCK = threading.RLock()


class GaitAnonError(ValueError):
    """Raised on any input the adapter refuses, or any output it does not trust.

    Subclasses ValueError so a caller that already catches ValueError around the
    vendored code keeps working.
    """


# --------------------------------------------------------------------------- small helpers
def _jsonable(obj):
    """Make provenance safe for `json.dump` (tuples -> lists, numpy -> python)."""
    if isinstance(obj, dict):
        return {str(k): _jsonable(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [_jsonable(v) for v in obj]
    if isinstance(obj, np.generic):
        return obj.item()
    if isinstance(obj, np.ndarray):
        return _jsonable(obj.tolist())
    return obj


class _PinnedEnv:
    """Set env vars for the duration of a call and restore them EXACTLY.

    Restoring exactly means a key that was absent before is deleted again, not
    set to "" -- `gait_preset()` treats "" as "bare LEVELS", which is a different
    config from "unset" (which means the shipped default).
    """

    def __init__(self, **overrides):
        self._overrides = {k: v for k, v in overrides.items() if v is not None}
        self._saved = {}

    def __enter__(self):
        for k, v in self._overrides.items():
            self._saved[k] = os.environ.get(k)
            os.environ[k] = str(v)
        return self

    def __exit__(self, *exc):
        for k, old in self._saved.items():
            if old is None:
                os.environ.pop(k, None)
            else:
                os.environ[k] = old
        return False


def _snapshot_globals():
    pae = _mirage.pose_anon_edge
    return {key: len(getattr(pae, name)) for key, name in _PROVENANCE_GLOBALS.items()}


def _slice_globals(marks):
    pae = _mirage.pose_anon_edge
    return {key: list(getattr(pae, name)[marks[key]:])
            for key, name in _PROVENANCE_GLOBALS.items()}


# --------------------------------------------------------------------------- validation
def _validate_rows(kp_rows, identity_rows, frame_wh, max_coord_mult, conf_thresh):
    """Hard validation. Returns (n_slots, n_frames, coord_limit).

    Nothing here is a warning: a wrong keypoint width or a missing confidence
    column silently produces a plausible-looking but meaningless anonymisation,
    which is exactly the failure mode that must never be accepted quietly.
    """
    if kp_rows is None:
        raise GaitAnonError("kp_rows is None; expected a list of per-slot row lists")
    try:
        n_slots = len(kp_rows)
    except TypeError as exc:
        raise GaitAnonError(f"kp_rows is not a sequence: {type(kp_rows).__name__}") from exc

    lengths = {len(rows) for rows in kp_rows}
    if len(lengths) > 1:
        raise GaitAnonError(
            f"ragged export buffer: slots have different frame counts "
            f"{[len(r) for r in kp_rows]}; every slot must be appended to on every "
            f"exported frame (pipeline.py:1146 present / :1172 absent)")
    n_frames = lengths.pop() if lengths else 0

    if identity_rows is not None:
        if len(identity_rows) != n_slots:
            raise GaitAnonError(
                f"identity_rows has {len(identity_rows)} slots but kp_rows has {n_slots}")
        for s, ids in enumerate(identity_rows):
            if len(ids) != n_frames:
                raise GaitAnonError(
                    f"identity_rows[{s}] has {len(ids)} frames but kp_rows[{s}] has "
                    f"{n_frames}; the two buffers must be appended to together")

    if not (0.0 <= float(conf_thresh) <= 1.0):
        raise GaitAnonError(f"conf_thresh must be in [0, 1], got {conf_thresh!r}")

    if frame_wh is not None:
        try:
            w, h = float(frame_wh[0]), float(frame_wh[1])
        except Exception as exc:
            raise GaitAnonError(f"frame_wh must be (width, height), got {frame_wh!r}") from exc
        if not (w > 0 and h > 0):
            raise GaitAnonError(f"frame_wh must be positive, got {frame_wh!r}")
        coord_limit = max(w, h) * float(max_coord_mult)
    else:
        coord_limit = None

    observed = 0.0
    for s, rows in enumerate(kp_rows):
        for t, row in enumerate(rows):
            if not isinstance(row, np.ndarray):
                raise GaitAnonError(
                    f"kp_rows[{s}][{t}] is {type(row).__name__}, expected np.ndarray")
            if row.ndim != 2 or row.shape[0] != N_BODY:
                raise GaitAnonError(
                    f"kp_rows[{s}][{t}] has shape {row.shape}, expected ({N_BODY}, 3) "
                    f"-- this adapter takes the COCO-17 body block only")
            if row.shape[1] != 3:
                raise GaitAnonError(
                    f"kp_rows[{s}][{t}] has shape {row.shape}: column 2 (confidence) is "
                    f"missing. The template and the unobserved-joint gate both read "
                    f"confidence; a (17, 2) row cannot be anonymised correctly")
            if not np.issubdtype(row.dtype, np.floating):
                raise GaitAnonError(
                    f"kp_rows[{s}][{t}].dtype is {row.dtype}, expected a floating dtype")
            if not np.isfinite(row).all():
                raise GaitAnonError(
                    f"kp_rows[{s}][{t}] contains non-finite values "
                    f"(nan/inf); refusing to anonymise")
            m = float(np.abs(row[:, :2]).max()) if row.size else 0.0
            if coord_limit is not None and m > coord_limit:
                raise GaitAnonError(
                    f"kp_rows[{s}][{t}] has |coord| {m:.1f} px > {coord_limit:.1f} px "
                    f"({max_coord_mult}x the frame size {frame_wh}); input is not in "
                    f"native pixel coordinates for this frame")
            observed = max(observed, m)

    if coord_limit is None:
        # No canvas given: bound the OUTPUT against the input's own extent instead,
        # so the 1e10 px signature is still caught either way.
        coord_limit = max(1.0, observed) * float(max_coord_mult)
    return n_slots, n_frames, coord_limit


# --------------------------------------------------------------------------- sequencing
def _present_frames(rows, conf_thresh):
    """Classify a slot's frames. Returns `(present, n_absent, n_low_conf)`.

    Present == the row is not all-zero AND at least one joint is at/above
    `conf_thresh`. Both halves are needed: `np.zeros((17, 3))` is the host's
    "absent" marker (pipeline.py:1172), and a row of real coordinates with every
    confidence below threshold is a detection the vendored code would zero out
    joint-by-joint anyway (`_unobs`, pose_anon_edge.py:2309) -- feeding it in
    would only skew the median bone lengths.

    The two rejected classes are counted SEPARATELY because they are not morally
    the same. An absent row is all zeros: excluding it leaks nothing. A
    low-confidence row holds a real person's real coordinates and is returned
    UNTRANSFORMED, so it is raw gait reaching disk and the caller is owed the
    number (see the module docstring).
    """
    present, n_absent, n_low = [], 0, 0
    for t, row in enumerate(rows):
        if not row.any():
            n_absent += 1
            continue
        if float(row[:, 2].max()) < conf_thresh:
            n_low += 1
            continue
        present.append(t)
    return present, n_absent, n_low


def _identity_runs(present, ids):
    """Split a slot's present frames into maximal runs of ONE identity.

    A slot is re-let after repeated misses, so one slot block can carry two
    different people. Grouping the whole block as one sequence would hand them a
    SHARED collapse template and a SHARED seed -- which is the linkable
    pseudo-identity the per-sequence seed exists to prevent.

    `None` (the host's `det_idx_to_identity.get(di)` miss) is treated as its own
    sentinel identity: it never merges with a named identity, and consecutive
    `None`s stay together. So an identity-free caller (identity_rows=None) gets
    exactly one run per slot, and a lookup miss splits rather than merges --
    the conservative direction.

    Returns [(identity, [frame, ...]), ...] in frame order.
    """
    runs = []
    prev = object()          # sentinel that equals nothing, including None
    for t in present:
        ident = ids[t] if ids is not None else None
        if not runs or not _same_identity(ident, prev):
            runs.append((ident, []))
        runs[-1][1].append(t)
        prev = ident
    return runs


def _same_identity(a, b):
    if a is b:
        return True
    try:
        return bool(a == b)
    except Exception:
        return False


def _degenerate_groups(kp_rows_slot, frames):
    """Bone-length degeneracy check for ONE compacted sequence, before it is sent.

    Runs the VENDORED `group_lengths()` -- the exact function whose output becomes
    the divisor at pose_anon_edge.py:866 -- on the (T, 17, 2) body sequence this
    call is about to hand down, and returns
    `(degenerate_group_names, min_median_px, all_medians)`.

    THE TEST IS `median <= 0`, WITH NO TOLERANCE, and that is deliberate: the
    failure value is EXACTLY 0.0 because it comes from two joints occupying the
    same point (an unplaced joint pair at `(0, 0)`), so no threshold has to be
    chosen and none is fitted to any clip. A near-zero-but-positive median is a
    real if unusual geometry and is REPORTED (`min_group_len_px`) rather than
    judged here -- inventing a cutoff for it would be exactly the kind of
    clip-fitted constant this project forbids.

    🔴 SO THIS GUARD IS NARROWER THAN THE FAILURE IT IS NAMED AFTER, and the gap
    is reachable on ordinary footage. MEASURED on this host (60 frames, one slot,
    the lower leg foreshortened on 51 of them -- a seated or end-on subject --
    `frame_wh=(640, 640)` exactly as the pipeline calls it):

        shin median   emitted max|coord| with frame_wh=None   with frame_wh set
        110.0 px      554 px  (healthy)                       554 px, 0.00 % on border
         12.0 px      1 461 px                                640 px, 1.5 % on border
          4.0 px      2 635 px                                640 px, 0.9 % on border
          0.25 px     19 930 px                               640 px, 2.6 % on border

    Every row below 110 px returned `applied: True` with NO exception and NO
    warning, because the vendored canvas clip (pose_anon_edge.py:2426) ran first.
    `min_group_len_px` was recorded correctly (4.0, 1.0, 0.25) -- the signal is in
    the provenance, nothing acts on it. The `frame_wh=None` column is the same
    input refused by the output bound, which is the bound this adapter's own
    docstring calls a dead second net in production. IT IS THE ONLY LIVE ONE for
    a small-but-positive median, and production never gets it.

    What the adapter now does instead of pretending otherwise: it records
    `group_lengths_px` for EVERY transformed sequence (not only refused ones) and
    MEASURES how many emitted joints the vendored clip moved onto the canvas
    border from an input that was inside it (`canvas_clipped_joints`). Both are
    facts, not thresholds. Choosing a refusal cutoff on `min_group_len_px` is an
    owner policy call and is deliberately NOT made here.
    """
    body = np.stack([np.asarray(kp_rows_slot[t], dtype=np.float64)[:, :2]
                     for t in frames], axis=0)              # (T, 17, 2)
    med = _mirage.pose_anon_edge.group_lengths(body)
    bad = sorted(g for g, v in med.items() if not (float(v) > 0.0))
    return bad, (min(float(v) for v in med.values()) if med else 0.0), \
        {g: round(float(v), 4) for g, v in sorted(med.items())}


def _gap_stats(frames):
    """(number of gaps, longest gap in frames) inside one present-frame run."""
    n_gaps, max_gap = 0, 0
    for a, b in zip(frames, frames[1:]):
        d = b - a - 1
        if d > 0:
            n_gaps += 1
            max_gap = max(max_gap, d)
    return n_gaps, max_gap


# --------------------------------------------------------------------------- packing
def _pack_person(row):
    """One host row (17, 3) -> one MIRAGE emit person dict.

    Body 0..16 carry the host's coordinates; 17..132 are zero with score 0.0, so
    the vendored feet/hand blocks translate rigidly by a zero-anchored delta and
    stay at the origin, and the face block 23..90 is zero before and after (the
    vendored code re-asserts `kp[:, 23:91] = 0.0` regardless).
    """
    kp = [[0.0, 0.0] for _ in range(N_WHOLEBODY)]
    score = [0.0] * N_WHOLEBODY
    xy = row[:, :2]
    conf = row[:, 2]
    for j in range(N_BODY):
        kp[j] = [float(xy[j, 0]), float(xy[j, 1])]
        score[j] = float(conf[j])
    return {"kp": kp, "score": score}


# --------------------------------------------------------------------------- public API
def anonymize_export_rows(kp_rows, identity_rows, fps,
                          preset=DEFAULT_PRESET,
                          level=DEFAULT_LEVEL,
                          seed=None,
                          min_present_frames=DEFAULT_MIN_PRESENT_FRAMES,
                          min_present_frac=DEFAULT_MIN_PRESENT_FRAC,
                          frame_wh=None,
                          conf_thresh=POSE_THRESH,
                          max_coord_mult=DEFAULT_MAX_COORD_MULT,
                          on_degenerate=DEFAULT_ON_DEGENERATE,
                          _anonymize_fn=None):
    """Apply the vendored MIRAGE gait defence to this host's export buffers.

    Parameters
    ----------
    kp_rows : list[list[np.ndarray]]
        Per slot, per exported frame, an (17, 3) float array `[x, y, confidence]`
        in NATIVE PIXEL coordinates -- i.e. `export_kp_rows` exactly as
        `_write_export_arrays()` sees it. An absent slot-frame is
        `np.zeros((17, 3))`. Never mutated.
    identity_rows : list[list] or None
        Parallel to `kp_rows`: the STABLE identity of the person occupying that
        slot on that frame (`det_idx_to_identity.get(di)`, pipeline.py:1142), or
        None where the slot was empty / the lookup missed. Pass None to treat
        each slot as a single identity for the whole clip.
    fps : float
        Sampling rate of the exported rows. Export mode forces dense export and
        SKIP_N == 1 (pipeline.py:104/123), so every buffered row is one real
        frame and this is the input clip's own fps -- not fps/skip.
    preset : str or None
        Gait preset name. "e2" is the measured arm. "" selects the bare LEVELS
        behaviour with no preset kwargs. None leaves `MIRAGE_GAIT_PRESET` alone
        and takes whatever the ambient environment selects (recorded in the
        provenance either way).
    level : str
        LEVELS ladder entry, "L4" for the shipped amplitudes.
    seed : int or None
        🔴 TEST ONLY. None (the default) is the shipped behaviour: the vendored
        `anonymize_pose_log` draws a fresh `new_clip_seed()` PER SEQUENCE, so no
        two tracklets and no two runs share a perturbation. Passing an int pins
        `MIRAGE_TEST_FIXED_SEED` for the duration of the call, which makes the
        output deterministic AND identical across every sequence in the call --
        a linkable pseudo-identity, never privacy-safe to ship. The value is
        never logged and never appears in the returned provenance; only the
        boolean `deterministic_seed` does.
    min_present_frames, min_present_frac : int, float
        Guards against zero-row poisoning residue. A sequence shorter than
        `min_present_frames` frames, or than `min_present_frac` of the clip, is
        left UNTRANSFORMED and reported in `provenance["skipped"]`.
    frame_wh : (int, int) or None
        Canvas size. When given it is passed straight to the vendored function,
        which clips the emitted coordinates to it, and it also sets the input /
        output sanity bound.
    conf_thresh : float
        Confidence at/above which a joint counts as observed (POSE_THRESH).
    max_coord_mult : float
        Sanity bound, as a multiple of the frame size. 🔴 Cannot fire when
        `frame_wh` is given -- the vendored code clips first. See
        `DEFAULT_MAX_COORD_MULT`.
    on_degenerate : {"raise", "skip"}
        What to do with a sequence whose bone lengths are degenerate
        (`_degenerate_groups`). "raise" (default) stops rather than emit an
        unvalidated skeleton; "skip" leaves that sequence RAW and counts it in
        `raw_passthrough_frames`. Both are lossy -- see `DEFAULT_ON_DEGENERATE`.
    _anonymize_fn : callable or None
        Test seam. Defaults to the vendored `anonymize_pose_log`. Must accept
        `(pose_log, level, fps, frame_wh=..., slot_log=...)`.

    Returns
    -------
    (new_kp_rows, provenance) : (list[list[np.ndarray]], dict)
        `new_kp_rows` has the same shape and dtype as the input. Transformed
        frames are NEW arrays; untransformed frames are the SAME array objects as
        the input (bit-identity, not a copy). `provenance` is JSON-safe and
        carries no seed value.
    """
    if fps is None or not np.isfinite(float(fps)) or float(fps) <= 0:
        raise GaitAnonError(f"fps must be a positive finite number, got {fps!r}")
    fps = float(fps)
    if level not in _mirage.LEVELS:
        raise GaitAnonError(
            f"level {level!r} is not in the vendored LEVELS ladder "
            f"{sorted(_mirage.LEVELS)}")
    if seed is not None:
        try:
            seed = int(seed)
        except Exception as exc:
            raise GaitAnonError(f"seed must be an int or None, got {seed!r}") from exc
    if on_degenerate not in ("raise", "skip"):
        raise GaitAnonError(
            f"on_degenerate must be 'raise' or 'skip', got {on_degenerate!r}")

    n_slots, n_frames, coord_limit = _validate_rows(
        kp_rows, identity_rows, frame_wh, max_coord_mult, conf_thresh)

    # Resolve the preset EARLY (pure function, raises on a typo) so a bad name
    # fails before anything is packed -- and so the provenance can report the
    # resolved kwargs even if the clip turns out to have nothing to transform.
    try:
        preset_kwargs = _mirage.gait_preset(preset)
    except ValueError as exc:
        raise GaitAnonError(str(exc)) from exc
    effective_preset = (
        preset if preset is not None
        else os.environ.get("MIRAGE_GAIT_PRESET", _mirage.SHIPPED_PRESET))

    level_knobs = dict(_mirage.LEVELS[level])
    host_level_kwargs = {k: v for k, v in preset_kwargs.items()
                         if k in _HOST_LEVEL_PRESET_KEYS}
    v2_preset_kwargs = {k: v for k, v in preset_kwargs.items()
                        if k not in _HOST_LEVEL_PRESET_KEYS}
    v2_kwargs = dict(level_knobs, **v2_preset_kwargs)

    provenance = {
        "defence": "gait",
        "vendored_module": getattr(_mirage.pose_anon_edge, "__file__", None),
        "preset": effective_preset,
        "preset_source": "argument" if preset is not None else "ambient_env",
        "level": level,
        "fps": fps,
        "frame_wh": list(frame_wh) if frame_wh is not None else None,
        "n_slots": n_slots,
        "n_frames": n_frames,
        "conf_thresh": float(conf_thresh),
        "min_present_frames": int(min_present_frames),
        "min_present_frac": float(min_present_frac),
        "max_coord_mult": float(max_coord_mult),
        "level_knobs": level_knobs,
        "preset_kwargs": preset_kwargs,
        "anonymize_v2_kwargs": v2_kwargs,
        "collapse_target_kwargs": host_level_kwargs,
        "seed_source": ("explicit_pinned_test_seed" if seed is not None
                        else "vendored_new_clip_seed_per_sequence"),
        "deterministic_seed": seed is not None,
        "provenance_globals": {k: f"pose_anon_edge.{v}"
                               for k, v in _PROVENANCE_GLOBALS.items()},
        "on_degenerate": on_degenerate,
        "per_slot_present_frames": [],
        "per_slot_absent_frames": [],
        "per_slot_low_confidence_frames": [],
        "sequences": [],
        "skipped": [],
        "transformed_frames": 0,
        "raw_passthrough_frames": 0,
        # Which kind of raw passthrough, because they have different causes and
        # different fixes. `low_confidence_rows` are real coordinates below
        # `conf_thresh`; `short_sequences` failed the min-present guard;
        # `degenerate_sequences` failed the bone-length guard under "skip".
        # All-zero rows are absent, hold nothing, and are counted in NONE of them.
        "raw_passthrough_breakdown": {
            "low_confidence_rows": 0,
            "short_sequences": 0,
            "degenerate_sequences": 0,
        },
        "calls_to_anonymize_pose_log": 0,
        "applied": False,
    }
    # The output bound is unreachable when the vendored clip runs first. Say so in
    # the artifact rather than letting a reader infer protection that is not there.
    provenance["output_bound_effective"] = frame_wh is None
    provenance["output_bound_note"] = (
        "coordinate sanity bound is DEAD: frame_wh was passed, so the vendored code "
        "clipped to the canvas (pose_anon_edge.py:2426) before this adapter saw the "
        "result. `degenerate_groups_checked` is the guard that held."
        if frame_wh is not None else
        "coordinate sanity bound is live (frame_wh=None: no vendored clip).")

    # --- who is present, where, and as whom ---------------------------------
    frac_floor = int(np.ceil(float(min_present_frac) * n_frames)) if n_frames else 0
    floor = max(int(min_present_frames), frac_floor)
    seq_of = [[None] * n_frames for _ in range(n_slots)]   # seq_of[slot][frame] = seq id
    seq_id = 0
    n_checked = 0
    for s in range(n_slots):
        present, n_absent, n_low = _present_frames(kp_rows[s], conf_thresh)
        provenance["per_slot_present_frames"].append(len(present))
        provenance["per_slot_absent_frames"].append(n_absent)
        provenance["per_slot_low_confidence_frames"].append(n_low)
        # A low-confidence row is real coordinates going to disk untransformed.
        provenance["raw_passthrough_frames"] += n_low
        provenance["raw_passthrough_breakdown"]["low_confidence_rows"] += n_low
        ids = identity_rows[s] if identity_rows is not None else None
        for ident, frames in _identity_runs(present, ids):
            n_gaps, max_gap = _gap_stats(frames)
            record = {
                "seq_id": seq_id,
                "slot": s,
                "identity": _jsonable(ident),
                "n_present": len(frames),
                "first_frame": frames[0],
                "last_frame": frames[-1],
                "n_gaps": n_gaps,
                "max_gap_frames": max_gap,
            }
            if len(frames) < floor:
                # A short sequence is exactly the shape that collapses a median
                # bone length; it is left raw and counted, never quietly
                # transformed.
                reason = ("below_min_present_frames" if len(frames) < int(min_present_frames)
                          else "below_min_present_frac")
                record["reason"] = reason
                record["floor_frames"] = floor
                provenance["skipped"].append(record)
                provenance["raw_passthrough_frames"] += len(frames)
                provenance["raw_passthrough_breakdown"]["short_sequences"] += len(frames)
                continue

            # 🔴 THE POISONING GUARD. Runs on the COMPACTED sequence, using the
            # vendored `group_lengths` -- the very divisor that blows up. This is
            # the check that catches the partial-limb case compaction cannot (see
            # the module docstring), and unlike the output bound it is unaffected
            # by `frame_wh` clipping.
            bad, min_len, medians = _degenerate_groups(kp_rows[s], frames)
            n_checked += 1
            record["min_group_len_px"] = round(float(min_len), 4)
            # Recorded for EVERY sequence, not only the refused ones: the divisor at
            # pose_anon_edge.py:866 is `median + 1e-6`, so a small-but-positive median
            # is the amplification factor in plain sight. See _degenerate_groups for
            # the measured table of what that does behind the canvas clip.
            record["group_lengths_px"] = medians
            if bad:
                record["reason"] = "degenerate_bone_lengths"
                record["degenerate_groups"] = bad
                record["group_lengths_px"] = medians
                if on_degenerate == "raise":
                    raise GaitAnonError(
                        f"slot {s} sequence {record['first_frame']}..{record['last_frame']} "
                        f"({len(frames)} present frames, identity {ident!r}): bone group(s) "
                        f"{bad} have a MEDIAN length of 0 px over the sequence. The vendored "
                        f"transform would divide by that (pose_anon_edge.py:866, "
                        f"len_factors = target / (median + 1e-6)) and emit ~1e9 px "
                        f"coordinates -- which `frame_wh` would then silently clip to the "
                        f"canvas. Cause: those joints are unplaced (0, 0) in more than half "
                        f"the sequence's frames. Group medians: {medians}. Pass "
                        f"on_degenerate='skip' to leave this sequence RAW and continue "
                        f"(it is then counted in raw_passthrough_frames).")
                provenance["skipped"].append(record)
                provenance["raw_passthrough_frames"] += len(frames)
                provenance["raw_passthrough_breakdown"]["degenerate_sequences"] += len(frames)
                continue

            for t in frames:
                seq_of[s][t] = seq_id
            provenance["sequences"].append(record)
            provenance["transformed_frames"] += len(frames)
            seq_id += 1
    provenance["degenerate_groups_checked"] = n_checked

    out_rows = [list(rows) for rows in kp_rows]
    if seq_id == 0:
        # Nothing survived the guard (an empty buffer, or a clip of slots too
        # short to transform). Report it exactly as such rather than pretending
        # the defence ran.
        provenance["template_kinds"] = []
        provenance["stature_scales"] = []
        provenance["canvas_clipped_joints"] = 0
        provenance["canvas_clip_note"] = "nothing was transformed, so nothing was clipped."
        # Under the same lock as the real path: `_PinnedEnv` mutates process-global
        # os.environ, so doing it outside would let this no-op call clobber a
        # concurrent call's pinned preset (the host runs det/pose/seg on a pool).
        with _CALL_LOCK:
            with _PinnedEnv(**({"MIRAGE_GAIT_PRESET": preset} if preset is not None else {})):
                provenance["env_snapshot"] = _env_snapshot()
        return out_rows, _jsonable(provenance)

    # --- pack: variable-length frames, so absent cells are never handed down --
    # `slot_groups()` (person_slots.py:52) walks `range(len(frame))` for every
    # frame, so ANY cell present in `pose_log` joins a group. Emitting only the
    # present cells is therefore both the compaction and the identity split: the
    # group id IS the sequence id, and one call covers every slot.
    pose_log, slot_log, cells = [], [], []
    for t in range(n_frames):
        frame_people, frame_slots, frame_cells = [], [], []
        for s in range(n_slots):
            sid = seq_of[s][t]
            if sid is None:
                continue
            frame_people.append(_pack_person(kp_rows[s][t]))
            frame_slots.append(sid)
            frame_cells.append(s)
        pose_log.append(frame_people)
        slot_log.append(frame_slots)
        cells.append(frame_cells)

    fn = _anonymize_fn or _mirage.anonymize_pose_log
    with _CALL_LOCK:
        env = {}
        if preset is not None:
            env["MIRAGE_GAIT_PRESET"] = preset
        if seed is not None:
            env["MIRAGE_TEST_FIXED_SEED"] = seed
        with _PinnedEnv(**env):
            provenance["env_snapshot"] = _env_snapshot()
            marks = _snapshot_globals()
            out_log = fn(pose_log, level, fps, frame_wh=frame_wh, slot_log=slot_log)
            provenance.update(_slice_globals(marks))
    provenance["calls_to_anonymize_pose_log"] = 1

    # --- unpack -------------------------------------------------------------
    if len(out_log) != n_frames:
        raise GaitAnonError(
            f"vendored anonymize_pose_log returned {len(out_log)} frames for "
            f"{n_frames} in -- refusing to scatter a misaligned result")
    clipped_by_seq = {}
    for t in range(n_frames):
        if len(out_log[t]) != len(cells[t]):
            raise GaitAnonError(
                f"frame {t}: vendored result has {len(out_log[t])} people for "
                f"{len(cells[t])} sent -- refusing to scatter a misaligned result")
        for k, s in enumerate(cells[t]):
            src = kp_rows[s][t]
            kp = np.asarray(out_log[t][k]["kp"], dtype=np.float64)[:N_BODY, :2]
            if not np.isfinite(kp).all():
                raise GaitAnonError(
                    f"slot {s} frame {t}: anonymised coordinates are not finite")
            if frame_wh is not None:
                # 🔴 DID THE VENDORED CANVAS CLIP BIND? (pose_anon_edge.py:2426.)
                # A joint whose INPUT was strictly inside the canvas but whose
                # OUTPUT sits exactly on an edge was not placed by the transform,
                # it was placed by np.clip. That is the visible half of the
                # bone-length amplification the median<=0 guard cannot see, and
                # on a clean sequence it is 0 (measured: 0.00 % vs 0.8-2.6 % for
                # the foreshortened-shin sequences tabulated in
                # `_degenerate_groups`). MEASURED, never raised: a subject
                # genuinely walking off the frame edge produces some of these
                # honestly, so the threshold is an owner call, not ours.
                w, h = float(frame_wh[0]), float(frame_wh[1])
                inside = ((src[:, 0] > 0) & (src[:, 0] < w) &
                          (src[:, 1] > 0) & (src[:, 1] < h))
                pinned = ((kp[:, 0] <= 0.0) | (kp[:, 0] >= w) |
                          (kp[:, 1] <= 0.0) | (kp[:, 1] >= h))
                n_clipped = int(np.count_nonzero(inside & pinned))
                # 🔴 SPLIT BY CONFIDENCE (2026-08-14). The raw total counts joints the
                # detector never actually placed -- an unobserved joint sits wherever the
                # rebuild put it and carries no information -- so it OVERSTATES the signal
                # badly. MEASURED on the 2-person end-to-end run (d07, 50 f @1264²): the
                # raw total was 463 while only 42 CONFIDENT joints touched an edge, and all
                # 42 belonged to the one subject whose real detector box already sat on the
                # frame bottom in 50/50 frames. Reading the raw total alone reads as a
                # transform defect; reading the confident count reads as what it is --
                # framing. Both are kept: the raw one stays for continuity, the confident
                # one is the number to act on.
                conf_ok = src[:, 2] >= conf_thresh
                n_clipped_conf = int(np.count_nonzero(inside & pinned & conf_ok))
                # Was the SUBJECT already cut off by the camera? Derived from the INPUT
                # only (no detector box needed here): a confident input joint sitting on
                # the border means the person genuinely leaves the frame there, which
                # produces honest clipping the transform did not cause.
                src_at_edge = ((src[:, 0] <= 1.0) | (src[:, 0] >= w - 1.0) |
                               (src[:, 1] <= 1.0) | (src[:, 1] >= h - 1.0))
                n_src_edge = int(np.count_nonzero(src_at_edge & conf_ok))
                if n_clipped or n_clipped_conf or n_src_edge:
                    sid = seq_of[s][t]
                    a, b, c = clipped_by_seq.get(sid, (0, 0, 0))
                    clipped_by_seq[sid] = (a + n_clipped, b + n_clipped_conf, c + n_src_edge)
            m = float(np.abs(kp).max())
            if m > coord_limit:
                raise GaitAnonError(
                    f"slot {s} frame {t}: anonymised |coord| {m:.4g} px exceeds the "
                    f"{coord_limit:.1f} px sanity bound. This is the ZERO-ROW POISONING "
                    f"signature (group_lengths median driven to ~0 => len_factors ~1e6). "
                    f"The present-frame compaction should make it unreachable; do not "
                    f"widen the bound, find the degenerate sequence.")
            new = np.empty_like(src)
            new[:, :2] = kp                 # transform output (2 dp, as MIRAGE emits)
            new[:, 2] = src[:, 2]           # confidence: untouched, by contract
            out_rows[s][t] = new

    # Attribute the clip measurement back to the sequences that produced it, so an
    # auditor can pair `canvas_clipped_joints` with that sequence's own
    # `min_group_len_px` / `group_lengths_px` and see the mechanism, not just a total.
    for rec in provenance["sequences"]:
        n_cl, n_cl_conf, n_src_edge = clipped_by_seq.get(rec["seq_id"], (0, 0, 0))
        n_cl, n_cl_conf, n_src_edge = int(n_cl), int(n_cl_conf), int(n_src_edge)
        rec["canvas_clipped_joints"] = n_cl
        rec["canvas_clipped_joints_confident"] = n_cl_conf
        # A sequence whose own INPUT already touches the border is cut off by the
        # camera; its clipping is not evidence of a transform defect.
        rec["input_joints_on_frame_edge_confident"] = n_src_edge
        rec["subject_cut_off_by_framing"] = bool(n_src_edge)
        rec["canvas_clipped_frac"] = (round(n_cl / float(rec["n_present"] * N_BODY), 6)
                                      if rec["n_present"] else None)
        rec["canvas_clipped_frac_confident"] = (
            round(n_cl_conf / float(rec["n_present"] * N_BODY), 6)
            if rec["n_present"] else None)
    provenance["canvas_clipped_joints"] = int(sum(v[0] for v in clipped_by_seq.values()))
    provenance["canvas_clipped_joints_confident"] = int(
        sum(v[1] for v in clipped_by_seq.values()))
    provenance["input_joints_on_frame_edge_confident"] = int(
        sum(v[2] for v in clipped_by_seq.values()))
    # THE DETERMINISTIC HALF of the amplification signal, and the one to read first.
    # `canvas_clipped_joints` depends on the per-sequence seed (which direction the
    # blown-up bone happened to point), so it is 0 on some draws of a genuinely
    # degenerate sequence -- measured: shin median 4.0 px gave 39, 0 and 12 clipped
    # joints on three draws of the same input. The smallest median bone length does
    # NOT depend on the seed: it is computed from the input, before any draw.
    provenance["min_group_len_px_over_sequences"] = (
        min((s["min_group_len_px"] for s in provenance["sequences"]), default=None))
    provenance["canvas_clip_note"] = (
        "joints whose INPUT was strictly inside frame_wh but whose emitted position sits "
        "exactly on a canvas edge, i.e. placed by the vendored clip (pose_anon_edge.py:2426) "
        "rather than by the transform. 0 on a clean sequence. A NON-ZERO count with a small "
        "`min_group_len_px` is the bone-length amplification signature that the median<=0 "
        "degeneracy guard cannot see -- see gait_anon._degenerate_groups for the measured "
        "table. Reported, never raised: a subject genuinely walking off the frame edge also "
        "produces some of these."
        if frame_wh is not None else
        "not measured: frame_wh was None, so the vendored code did no canvas clip and the "
        "coordinate sanity bound was live instead.")
    provenance["applied"] = True
    return out_rows, _jsonable(provenance)


def _env_snapshot():
    """Ambient values of the vendored module's env knobs, seed value redacted."""
    snap = {k: os.environ.get(k) for k in _ENV_OF_INTEREST}
    snap["MIRAGE_TEST_FIXED_SEED_set"] = bool(
        (os.environ.get("MIRAGE_TEST_FIXED_SEED") or "").strip())
    return snap
