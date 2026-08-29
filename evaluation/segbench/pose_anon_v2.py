"""
segbench.pose_anon_v2 -- anti-re-ID pose anonymiser, REDESIGNED after the v1
evaluation proved v1 was only *pseudonymisation* (a deterministic per-identity
bijection: TM2-id rank-1 >= raw; dynamic gait channel untouched: TM2-seq floored
~18% >> 2% chance).

Two fixes, matching the two measured failures:

1. IDENTITY-COLLAPSE (k-same on the static channel).  v1 rescaled each person's
   limbs by a per-person CONSTANT -> an invertible bijection that preserves the
   identity manifold (relabelling, not anonymisation).  v2 instead SNAPS every
   skeleton's limb proportions to a SHARED population template (K=1 => the static
   anthropometry -- limb-length ratios, height -- carries ZERO identity, like
   greying every face to the same patch).  Information is DESTROYED, so it is not
   invertible and not a per-identity code.  Optional K>1: snap to the nearest of K
   prototypes for a k-anonymity/diversity trade.

2. DYNAMIC-CHANNEL perturbation.  v1 left joint-angle dynamics, cadence and
   asymmetry untouched (where most gait identity lives).  v2 adds, per clip:
    - cadence/rhythm warp  : smooth non-uniform time re-sampling (perturbs step
                              timing / stride rhythm; preserves clip duration),
    - joint-ANGLE offsets  : per-limb constant + slow-drift rotation of each bone
                              direction (perturbs angular dynamics & coordination),
    - asymmetry            : independent L/R so real gait asymmetry is broken.

Seeding is PER-CLIP random by default (NOT a per-identity constant) so no stable
pseudo-identity is emitted; the collapse to the template is deterministic and
identity-free regardless of seed.  A low-frequency positional drift is kept for
naturalness/robustness.
"""
import os
import numpy as np
from .pose_anon import EDGES, _GROUPS, _virtual, _body_scale, _rng, _lowfreq_noise

# groups that have a left/right partner (for asymmetry); "spine/neck/face/hip" are axial
_LR_GROUPS = ["thigh", "shin", "clav", "uarm", "farm"]


# --------------------------------------------------------------- template (k-same)
def group_lengths(kp_seq):
    """Per-GROUP median bone length over a clip (T,17,2). {group: length}."""
    per = {g: [] for g in _GROUPS}
    for t in range(kp_seq.shape[0]):
        o = _virtual(kp_seq[t])
        for p, c, g in EDGES:
            per[g].append(float(np.linalg.norm(o[c] - o[p])))
    return {g: float(np.median(v)) if v else 1.0 for g, v in per.items()}


# ---------------------------------------------------------------- standard skeleton
# ANATOMICAL template, expressed as multiples of SHOULDER WIDTH (biacromial breadth).
#
# WHY THIS EXISTS (measured 2026-07-26, v3/a10, N=100 frames):
#   the emitted skeleton was anatomically impossible. Against shoulder span = 1.00:
#       neck 1.027 (human 0.55-0.65)   upper arm 1.17/1.28 (0.85)
#       forearm 0.62/0.21 (0.75)       thigh 1.09/1.64 (1.05)
#       shin 3.85/3.30 (0.95)  <- FOUR TIMES too long
#   The user spotted it as "why is neck same length as of arms and legs".
#
# ROOT CAUSE: the collapse target came from the CLIP's own median bone lengths, and
# `_body_scale` normalises by the shoulder->mid-hip torso. On a bust/3-4 framing the legs are
# NOT OBSERVED at all -- measured binarised confidence on a10: nose/shoulders/elbows 1.000,
# wrists 0.000-0.160, hips 0.840/0.860 (18 %/51 % off-frame), knees 0.000, ankles 0.000
# (23 %/82 % off-frame). So every limb was being scaled by a reference derived from the one
# unreliable landmark pair, and unobserved joints were snapped to garbage lengths.
#
# THE FIX: collapse to a FIXED anatomical skeleton scaled by SHOULDER WIDTH, which is the
# landmark pair that is actually observed (confidence 1.000, never off-frame).
#
# Ratios below are stature-normalised anthropometry re-expressed over biacromial breadth
# (biacromial ~ 0.234 H): upper arm 0.186 H, forearm 0.146 H, thigh 0.245 H, shin 0.246 H,
# shoulder->hip 0.288 H, nose->mid-shoulder ~0.140 H.
#
# PRIVACY: this is MORE anonymous than the previous target, not less. Every subject now
# collapses onto ONE identical set of proportions instead of a per-clip median that still
# carried some of the subject's own build. Only the overall SIZE (shoulder width) survives,
# and size is already disclosed to the recipient by the mask.
_ANATOMICAL_OVER_SHOULDER = {
    "clav":  0.500,   # mid-shoulder -> each shoulder = half the span, by definition
    "neck":  0.600,   # mid-shoulder -> nose
    "uarm":  0.795,
    "farm":  0.624,
    "spine": 1.231,   # mid-hip -> mid-shoulder
    "hip":   0.360,   # mid-hip -> each hip
    "thigh": 1.047,
    "shin":  1.051,
    "face":  0.130,   # nose -> eye / eye -> ear, small and fixed
}


def anatomical_template(shoulder_px):
    """The standard skeleton, in pixels, for a given observed shoulder width.

    Use this INSTEAD of population_template() when the clip's own limb lengths cannot be
    trusted -- which is whenever the lower body is out of frame. Returns {group: length_px}.
    """
    s = float(max(shoulder_px, 1.0))
    return {g: r * s for g, r in _ANATOMICAL_OVER_SHOULDER.items()}


def observed_shoulder_px(kp_seq, scores=None, conf_thresh=0.5):
    """Median shoulder width over the frames where BOTH shoulders are confidently seen.

    Shoulder width is the only scale reference measured to be reliable on real framings
    (confidence 1.000, 0 % off-frame vs hips 0.84/0.86 and up to 51 % off-frame)."""
    vals = []
    for t in range(kp_seq.shape[0]):
        if scores is not None:
            sc = scores[t]
            if min(float(sc[5]), float(sc[6])) < conf_thresh:
                continue
        d = float(np.linalg.norm(kp_seq[t][5] - kp_seq[t][6]))
        if d > 1.0:
            vals.append(d)
    return float(np.median(vals)) if vals else 1.0


def population_template(kp_seqs):
    """Median per-group bone length across a POPULATION of clips -> the shared
    collapse target. kp_seqs: iterable of (T,17,2). Aggregate statistic (not
    identity-specific): everyone is mapped to this one skeleton."""
    acc = {g: [] for g in _GROUPS}
    for seq in kp_seqs:
        gl = group_lengths(seq)
        for g in _GROUPS:
            acc[g].append(gl[g])
    return {g: float(np.median(v)) for g, v in acc.items()}


# ------------------------------------------------- PER-CLIP SCALE SOURCES (mirrored 2026-08-08)
# 🔴 WHY THESE ARE HERE, AND WHY NOTHING IN THIS DIRECTORY CALLS THEM YET.
#
# Every reID harness in `evaluation/privacy/gait/` builds ONE `population_template(...)` and hands the
# same absolute-size skeleton to every tracklet (`anon_adapter.build_template` ->
# `transform_v2(..., template)`), so in the lab the emitted figure's SIZE is a population constant.
# DEPLOYMENT does not do that: `pose_anon_edge.anatomical_template_for_clip` scales the same shared
# proportion set by a quantity observed on THIS clip. Consequence, stated plainly:
#
#     NO TM1/TM2/TM3 NUMBER ON RECORD PRICES THE DEPLOYED PER-CLIP SIZE CHANNEL, under ANY scale
#     source - not `shoulder` (what ships), not `projected` (what was approved), not `extent`.
#
# These are verbatim mirrors of the edge copy so that arm can be run through the SAME code the
# device runs, rather than a reimplementation of it. Numeric parity with `pose_anon_edge` is
# asserted by `_e2e/run3_20260807/sticks_fix/parity_check.py`.
def observed_spine_px(kp_seq, scores=None, conf_thresh=0.5):
    """Median mid-hip -> mid-shoulder length over frames where all four trunk joints are seen."""
    vals = []
    for t in range(kp_seq.shape[0]):
        k = kp_seq[t]
        if scores is not None:
            sc = scores[t]
            if min(float(sc[5]), float(sc[6]), float(sc[11]), float(sc[12])) < conf_thresh:
                continue
        d = float(np.linalg.norm(0.5 * (k[5] + k[6]) - 0.5 * (k[11] + k[12])))
        if d > 1.0:
            vals.append(d)
    return float(np.median(vals)) if vals else 0.0


def stature_scale(kp_seq, scores=None, template=None, conf_thresh=0.5, min_joints=6):
    """Least-squares VERTICAL scale that makes the canonical skeleton match this person's stature.

    Verbatim mirror of `pose_anon_edge._stature_scale` - see that docstring for the derivation
    (k is exact, not iterative, because the rebuild is linear in the template)."""
    if template is None:
        return None
    gl = group_lengths(kp_seq)
    lf = {g: template[g] / (gl[g] + 1e-6) for g in _GROUPS}
    zero = {g: 0.0 for g in _GROUPS}
    ks = []
    for t in range(kp_seq.shape[0]):
        obs = np.asarray(kp_seq[t], np.float64)
        ok = np.abs(obs).sum(1) > 0
        if scores is not None:
            ok &= np.asarray(scores[t], np.float64) >= conf_thresh
        if int(ok.sum()) < min_joints:
            continue
        rb = _rebuild_frame(kp_seq[t], lf, zero)
        oy = obs[ok, 1] - 0.5 * (obs[11, 1] + obs[12, 1])
        ry = rb[ok, 1] - 0.5 * (rb[11, 1] + rb[12, 1])
        den = float((ry * ry).sum())
        if den <= 1e-6:
            continue
        ks.append(float((oy * ry).sum() / den))
    if not ks:
        return None
    k = float(np.median(ks))
    return k if 0.05 < k < 20.0 else None


def anatomical_template_for_clip(kp_seq, scores=None, src=None):
    """Verbatim mirror of `pose_anon_edge.anatomical_template_for_clip`.

    src: "shoulder" (deployment default) | "spine" | "projected" | "extent".
    Returns (template, kind)."""
    src = (src if src is not None
           else os.environ.get("MIRAGE_POSE_SCALE_FROM", "shoulder")).strip().lower()
    if src == "extent":
        sh0 = observed_shoulder_px(kp_seq, scores)
        if sh0 > 1.0:
            base = {g: _ANATOMICAL_OVER_SHOULDER[g] * sh0 for g in _GROUPS}
            k = stature_scale(kp_seq, scores, base)
            if k is not None:
                return {g: v * k for g, v in base.items()}, "anatomical_extent"
    if src in ("spine", "projected"):
        sp = observed_spine_px(kp_seq, scores)
        if sp > 1.0:
            sh_equiv = sp / _ANATOMICAL_OVER_SHOULDER["spine"]
            return ({g: _ANATOMICAL_OVER_SHOULDER[g] * sh_equiv for g in _GROUPS},
                    "anatomical_spine" if src == "spine" else "anatomical_projected")
    sh = observed_shoulder_px(kp_seq, scores)
    return {g: _ANATOMICAL_OVER_SHOULDER[g] * sh for g in _GROUPS}, "anatomical_shoulder"


# ------------------------------------------------------------------ frame rebuild
def _rebuild_frame(kp, len_factors, angle_off, edge_scale=None, angle_sign=None,
                   angle_abs=False):
    """Rebuild 17 joints from mid-hip, applying per-group length scale AND a 2D
    rotation of each bone direction by angle_off[group] (radians). Mid-hip fixed.

    `angle_sign` (optional, 2026-08-07) is a PER-EDGE multiplier on the ROTATION, aligned with
    EDGES, used by the ANTI-SYMMETRIC angle mode (`angle_mirror`). VERBATIM port of the edge copy.

    `edge_scale` (optional, added 2026-07-31) is a PER-EDGE multiplier aligned with EDGES, used
    by the true left/right asymmetry break (`lr_asymmetry_sigma`). `len_factors` is keyed by
    GROUP and both members of an L/R pair share the group name, so a per-group dict structurally
    cannot express "left thigh 3 % longer than right". Default None keeps the exact original
    arithmetic (no extra multiply), so parity with pose_anon_edge is untouched."""
    o = _virtual(kp)
    n = {"MH": o["MH"].copy()}
    for ei, (parent, child, grp) in enumerate(EDGES):
        bone = o[child] - o[parent]
        a = angle_off.get(grp, 0.0)
        if a and angle_sign is not None:
            # angle_abs => OUTWARD-ONLY: the magnitude is whatever was drawn, the DIRECTION is
            # forced away from the body midline (see _mirror_signs).
            a = (abs(a) if angle_abs else a) * angle_sign[ei]
        if a:
            ca, sa = np.cos(a), np.sin(a)
            bone = np.array([ca * bone[0] - sa * bone[1], sa * bone[0] + ca * bone[1]])
        bone = bone * len_factors[grp] if edge_scale is None else bone * (len_factors[grp] * edge_scale[ei])
        n[child] = n[parent] + bone
        if child == "MS":
            n["MS"] = n[parent] + bone
    return np.stack([n[i] for i in range(17)], axis=0)


# ------------------------------------------------------------------- cadence warp
def _cadence_warp(seq, wobble_amp_frac, period_s, fps, seed):
    """Non-uniform time re-sampling: local speed = 1 + smooth low-freq wobble,
    re-normalised to preserve clip endpoints (perturbs cadence/rhythm, not length).
    seq:(T,17,2). wobble_amp_frac is fractional speed deviation amplitude."""
    T = seq.shape[0]
    if wobble_amp_frac <= 0 or T < 4:
        return seq
    sigma = max(1.0, period_s * fps / 3.0)
    w = _lowfreq_noise(T, sigma, 1.0, seed)          # unit-std smooth track
    speed = 1.0 + wobble_amp_frac * (w / (np.abs(w).max() + 1e-9))
    speed = np.clip(speed, 0.4, 1.6)
    cum = np.cumsum(speed); cum = cum - cum[0]
    t_src = cum / (cum[-1] + 1e-9) * (T - 1)         # maps [0,T-1]->[0,T-1] warped
    i0 = np.floor(t_src).astype(int); i1 = np.minimum(i0 + 1, T - 1)
    frac = (t_src - i0)[:, None, None]
    return (1 - frac) * seq[i0] + frac * seq[i1]


# ------------------------------------ per-limb cadence decorrelation (ADDITIVE, 2026-07-31)
# `_cadence_warp` re-times the WHOLE skeleton on one clock, so every inter-limb phase
# relationship -- left-arm vs right-arm swing, arm vs contralateral leg -- survives it intact,
# and those relationships are a large part of what a gait recogniser reads. `limb_phase_amp`
# gives each limb chain its OWN independent time warp.
#
# Each entry is (anchor_joint, [joints hanging off it]). Only the OFFSETS from the anchor are
# re-sampled; the anchor stays on the global clock. By construction: the skeleton never comes
# apart, no bone length changes (the offset at t' carries the length the collapse already
# fixed, so `do_canon`'s identity collapse is untouched), and it is a re-SAMPLE of a smooth
# track rather than additive noise -- the same operation family as `cadence_amp`, measured at
# jerk 0.45 against the real input's 0.50 (POSE_KNOB_ATTRIBUTION_20260726.json).
# Default 0.0 => the block is skipped and output is bit-identical to before.
_LIMB_CHAINS = [(5, [7, 9]), (6, [8, 10]), (11, [13, 15]), (12, [14, 16])]


def _limb_phase_warp(seq, amp, period_s, fps, seed):
    """Give each limb chain its own cadence warp, relative to its anchor joint. seq:(T,17,2)."""
    if amp <= 0 or seq.shape[0] < 4:
        return seq
    out = seq.copy()
    for ci, (anchor, joints) in enumerate(_LIMB_CHAINS):
        off = seq[:, joints, :] - seq[:, anchor:anchor + 1, :]          # (T,nj,2)
        off = _cadence_warp(off, amp, period_s, fps, seed * 7919 + 101 + ci * 37)
        out[:, joints, :] = seq[:, anchor:anchor + 1, :] + off
    return out


# ------------------------- bounded seeded retargeting (ADDITIVE, 2026-08-11)
# VERBATIM numeric mirror of pose_anon_edge.py. Keep the separate RNG stream and operation order:
# a TM3 number measured here must describe the edge code that actually ships.
_SEEDED_SCALE_REGIONS = (
    ("head", ("neck", "face")),
    ("trunk", ("spine", "clav", "hip")),
    ("arms", ("uarm", "farm")),
    ("legs", ("thigh", "shin")),
)
_PROXIMITY_WEIGHTS = np.array([
    0.60, 0.60, 0.60, 0.60, 0.60,
    0.65, 0.65,
    0.85, 0.85, 1.00, 1.00,
    0.65, 0.65, 0.85, 0.85, 1.00, 1.00,
], np.float64)


def _bounded_seeded_scales(len_factors, seed, global_max=0.0, region_max=0.0):
    """Multiply canonical bone targets by bounded, clip-static seeded factors."""
    gm, rm = float(global_max), float(region_max)
    if not (0.0 <= gm < 0.5 and 0.0 <= rm < 0.5):
        raise ValueError("seeded scale maxima must be in [0, 0.5)")
    if gm == 0.0 and rm == 0.0:
        return len_factors
    out = dict(len_factors)
    rs = _rng(seed * 104729 + 7919)
    global_mult = 1.0 + (rs.uniform(-gm, gm) if gm else 0.0)
    for g in out:
        out[g] *= global_mult
    if rm:
        for _name, groups in _SEEDED_SCALE_REGIONS:
            mult = 1.0 + rs.uniform(-rm, rm)
            for g in groups:
                out[g] *= mult
    return out


def _observed_pose_heights(kp_seq, scores=None):
    """Per-frame visible heights using only confidence-valid detector joints.

    Missing/cropped frames inherit the clip median. This keeps the bound content-scaled without
    letting a low-confidence coordinate outlier enlarge every frame's proximity envelope.
    """
    k = np.asarray(kp_seq, np.float64)
    if scores is None:
        score_live = np.ones(k.shape[:2], dtype=bool)
    else:
        s = np.asarray(scores, np.float64)
        if s.shape != k.shape[:2]:
            raise ValueError(f"proximity scores shape {s.shape} != keypoints {k.shape[:2]}")
        score_live = np.isfinite(s) & (s >= 0.5)
    heights = np.full(len(k), np.nan, np.float64)
    for i, frame in enumerate(k):
        live = score_live[i] & np.isfinite(frame).all(1) & (np.abs(frame).sum(1) > 0)
        if int(live.sum()) >= 2:
            h = float(np.ptp(frame[live, 1]))
            if h > 1.0:
                heights[i] = h
    finite = np.isfinite(heights)
    fallback = float(np.median(heights[finite])) if finite.any() else 1.0
    heights[~finite] = max(1.0, fallback)
    return heights


def _observed_pose_height(kp_seq, scores=None):
    """Robust clip-median visible height; retained for reports and compatibility."""
    return float(np.median(_observed_pose_heights(kp_seq, scores=scores)))


def _apply_proximity_envelope(out, kp_seq, max_frac, scores=None):
    """Strict sequence-wide joint envelope; mirrors the deployed implementation."""
    frac = float(max_frac)
    if frac == 0.0:
        return out
    if not (0.0 < frac <= 0.25):
        raise ValueError("proximity_cap_frac must be in (0, 0.25]")
    ref = np.asarray(kp_seq, np.float64)
    cand = np.asarray(out, np.float64)
    live = (np.isfinite(ref).all(2) & np.isfinite(cand).all(2) &
            (np.abs(ref).sum(2) > 0))
    if scores is not None:
        s = np.asarray(scores, np.float64)
        if s.shape != live.shape:
            raise ValueError(f"proximity scores shape {s.shape} != keypoints {live.shape}")
        live &= np.isfinite(s) & (s >= 0.5)
    dist = np.linalg.norm(cand - ref, axis=2)
    caps = (_observed_pose_heights(ref, scores=scores)[:, None] * frac *
            _PROXIMITY_WEIGHTS[None, :])
    moving = live & (dist > 1e-12)
    if not moving.any():
        return cand
    ratios = caps[moving] / dist[moving]
    alpha = min(1.0, float(np.min(ratios)))
    blended = ref + alpha * (cand - ref)
    return np.where(live[:, :, None], blended, cand)


# ------------------------------------------------------------------- main entry
def _observed_shoulder_px(kp_seq, scores=None, conf_thresh=0.5):
    """Median shoulder width over frames where BOTH shoulders are confidently observed.

    Vectorised 2026-08-02: this is called twice per tracklet by the projection fit, and at
    CASIA-B scale (13 640 tracklets x 2 for train+test) the per-frame Python loop cost ~12 min
    of pure CPU per TM3 run. Numerically identical -- verified max|delta| = 0 against the loop.
    """
    k = np.asarray(kp_seq)
    d = np.linalg.norm(k[:, 5, :2] - k[:, 6, :2], axis=-1)
    ok = d > 1.0
    if scores is not None:
        sc = np.asarray(scores)
        ok &= np.minimum(sc[:, 5], sc[:, 6]) >= conf_thresh
    return float(np.median(d[ok])) if ok.any() else 0.0


_AXIAL_GROUPS = ("spine", "clav", "hip", "neck", "face")
_LIMB_ANGLE_GROUPS = ("uarm", "farm", "thigh", "shin")


def _angle_group_mask(angle_groups):
    """{group: does it take the angle offset}. None => all of them (shipped behaviour).

    Env fallback for build scripts: MIRAGE_TORSO_QUIET=1 selects the four LIMB groups;
    MIRAGE_ANGLE_GROUPS="a,b,c" overrides that selection explicitly (used by the ablation).
    Unknown names raise rather than being ignored -- a typo that silently reproduced the shipped
    arm would be scored as the new one, which is how the MIRAGE_ANON_ANGLE_CONST_DEG typo nearly
    shipped duplicate arms.
    """
    if angle_groups is None:
        ov = os.environ.get("MIRAGE_ANGLE_GROUPS", "").strip()
        quiet = os.environ.get("MIRAGE_TORSO_QUIET", "0") not in ("0", "", "false", "off")
        if not (ov or quiet):
            return {g: True for g in _GROUPS}
        angle_groups = tuple(s for s in (x.strip() for x in ov.split(",")) if s) if ov \
            else _LIMB_ANGLE_GROUPS
    bad = [g for g in angle_groups if g not in _GROUPS]
    if bad:
        raise ValueError(f"angle_groups: unknown bone group(s) {bad}; valid are {sorted(_GROUPS)}")
    return {g: g in angle_groups for g in _GROUPS}


def _limb_phase_offset(seq, max_s, fps, seed):
    """Give each limb chain a CONSTANT time offset instead of a time WARP. seq:(T,17,2).

    PORTED VERBATIM from `tree/mirage_edge_deploy/tier1_raspberry_pi5/pose_anon_edge.py`
    (2026-08-13) -- the two copies must stay in numeric parity or a TM3 number describes code that
    does not ship. See that file for the rationale; in short, `_limb_phase_warp` decorrelates
    inter-limb phase by RE-SAMPLING each chain, and a re-sample changes local speed, which is the
    owner-reported "sudden" motion. A pure SHIFT destroys the same relative-timing information
    while leaving every limb's own trajectory, and therefore its smoothness, untouched.
    """
    if max_s <= 0 or seq.shape[0] < 4:
        return seq
    T = seq.shape[0]
    r = _rng(seed * 6151 + 17)
    out = seq.copy()
    idx = np.arange(T)
    for ci, (anchor, joints) in enumerate(_LIMB_CHAINS):
        dt = float(r.uniform(-max_s, max_s) * fps)
        t = idx + dt
        per = 2 * (T - 1) if T > 1 else 1
        t = np.abs(np.mod(t, per))
        t = np.where(t > T - 1, per - t, t)
        i0 = np.floor(t).astype(int)
        i1 = np.minimum(i0 + 1, T - 1)
        f = (t - i0)[:, None, None]
        off = seq[:, joints, :] - seq[:, anchor:anchor + 1, :]
        shifted = (1 - f) * off[i0] + f * off[i1]
        out[:, joints, :] = seq[:, anchor:anchor + 1, :] + shifted
    return out


def _root_speed_warp(seq, amp, period_s, fps, seed, max_frac):
    """Re-time the ROOT along its OWN path -- the walking-SPEED channel, bounded for containment.

    WHY (2026-08-14). Every arm to date leaves the body travelling at the subject's real pace: the
    root is deliberately locked to the true trajectory so the emitted figure stays inside the mask
    the recipient is given. But the untrained descriptor computes speed/acceleration on ABSOLUTE
    coordinates, so that untouched root motion is a live identity channel, and walking speed is one
    of the strongest gait cues there is. §A.2x could not see it separately because it sits inside
    the same DYNAMICS block the limb knobs also feed.

    The path is NOT changed -- only the rate at which it is traversed -- and the whole skeleton is
    translated by the resulting root delta, so limbs keep their exact relationship to the body and
    no bone length or joint angle moves. The modulation is smooth and low-frequency, so it adds no
    kink; and it is HARD-CLAMPED to `max_px` of displacement from the true root, which is what
    keeps the figure inside the grown mask. A clamp, not a hope: the shift is rescaled until the
    measured worst-case displacement fits, so containment cannot regress silently.
    """
    T = seq.shape[0]
    if amp <= 0 or T < 4 or max_frac <= 0:
        return seq
    r = _rng(seed * 49979687 + 15485867)
    root = 0.5 * (seq[:, 11, :] + seq[:, 12, :])
    idx = np.arange(T)
    ph = r.uniform(0, 2 * np.pi)
    dt = amp * float(fps) * np.sin(2 * np.pi * idx / max(period_s * fps, 2.0) + ph)
    t = np.clip(idx + dt, 0, T - 1)
    i0 = np.floor(t).astype(int)
    i1 = np.minimum(i0 + 1, T - 1)
    f = (t - i0)[:, None]
    shift = ((1 - f) * root[i0] + f * root[i1]) - root
    # 🔴 THE CLAMP MUST BE SCALE-FREE (2026-08-14). It was first written as an absolute pixel
    # bound, which is a cross-config bug: CASIA-B subjects are ~115.6 px tall and our own footage
    # ~629.1 px, so one number means a 5.4x DIFFERENT perturbation in the lab than on device --
    # the same class of error as parameterising a low-pass in seconds (§A.2x-5). Expressed as a
    # fraction of the subject's own visible height, one value means one thing everywhere.
    live = np.abs(seq).sum(2) > 0
    hs = [float(seq[t][live[t]][:, 1].ptp()) for t in range(T) if live[t].sum() >= 2]
    h = float(np.median(hs)) if hs else 1.0
    cap = max(max_frac * h, 1e-6)
    worst = float(np.abs(shift).max()) if shift.size else 0.0
    if worst > cap:                          # hard clamp -- containment is not negotiable
        shift *= cap / worst
    return seq + shift[:, None, :]


def _limb_swing_amp(seq, amp, seed):
    """Scale each limb chain's ANGULAR SWING about its own mean pose. seq:(T,17,2).

    WHY THIS EXISTS (2026-08-14). §A.2x showed the only channel still leaking is DYNAMICS -- the
    five speed/acceleration statistics. Every knob tried against it so far reaches it by RE-TIMING
    (`cadence_amp`, `limb_phase_amp/offset`, `pose_rate_jitter`), and re-timing under
    `cadence_root_lock` is measured to cost 40-46 % jerk (§A.2y-5) because the re-timed limbs beat
    against a root still travelling at the real speed. That cost survived both a low-pass and a C1
    resampler, so it is inherent to re-timing, not to how the re-timing is interpolated.

    Swing AMPLITUDE reaches the same statistics by a different route: joint speed is (angular rate x
    radius), so scaling the angular excursion scales every speed and acceleration in the block
    WITHOUT touching the clock. Nothing is re-sampled, so there is no interpolation kink and no beat
    against the root -- the limb traces the same path shape, just a wider or narrower arc, which is
    also a real between-person difference rather than an artefact.

    BONE LENGTHS ARE EXACTLY PRESERVED: each bone is rebuilt as parent + L(t) * unit(angle), where
    L(t) is the ORIGINAL per-frame length, so this cannot undo the k-same collapse the template
    performs. Chains are walked parent-first so each bone hangs off its already-corrected parent.
    One draw per CHAIN per SEQUENCE, from its own RNG stream.
    """
    if amp <= 0 or seq.shape[0] < 4:
        return seq
    r = _rng(seed * 27644437 + 6291469)
    out = seq.copy()
    for anchor, joints in _LIMB_CHAINS:
        k = 1.0 + float(r.uniform(-amp, amp))
        parent = anchor
        for j in joints:
            p = out[:, parent, :]
            v = seq[:, j, :] - seq[:, parent, :]
            L = np.linalg.norm(v, axis=1)
            th = np.unwrap(np.arctan2(v[:, 1], v[:, 0]))
            th = th.mean() + k * (th - th.mean())        # scale the excursion, keep the mean pose
            out[:, j, 0] = p[:, 0] + L * np.cos(th)
            out[:, j, 1] = p[:, 1] + L * np.sin(th)
            parent = j
    return out


def _resample_at(seq, t, kind="linear"):
    """Sample `seq` at fractional frame positions `t`, linearly or with Catmull-Rom.

    WHY CUBIC IS AN OPTION (2026-08-14, §A.2y-5). Linear interpolation is C0: the second derivative
    is impulsive at every knot, so a re-timed track carries a kink at each source frame. Measured,
    that is what makes `pose_rate_jitter` cost +40 % jerk at the phone's 10 fps while costing only
    12 % at the lab's 25 fps -- the sparser the samples, the larger each kink. Catmull-Rom is C1 and
    removes the kink itself rather than blurring it away afterwards, so it should not damp limb
    swing the way a low-pass does. Both are offered because a previous attempt to swap linear for
    Catmull-Rom (on a DIFFERENT knob, the cadence warp) measured WORSE, so this must be measured
    here rather than assumed.
    """
    T = seq.shape[0]
    i0 = np.floor(t).astype(int)
    f = (t - i0)[:, None, None]
    if kind == "linear":
        return (1 - f) * seq[np.clip(i0, 0, T - 1)] + f * seq[np.clip(i0 + 1, 0, T - 1)]
    p0 = seq[np.clip(i0 - 1, 0, T - 1)]
    p1 = seq[np.clip(i0, 0, T - 1)]
    p2 = seq[np.clip(i0 + 1, 0, T - 1)]
    p3 = seq[np.clip(i0 + 2, 0, T - 1)]
    f2, f3 = f * f, f * f * f
    return (0.5 * ((-f3 + 2 * f2 - f) * p0 + (3 * f3 - 5 * f2 + 2) * p1 +
                   (-3 * f3 + 4 * f2 + f) * p2 + (f3 - f2) * p3))


def _pose_rate_warp(seq, rate, fps, kind="linear"):
    """Re-time the POSE at a CONSTANT rate. PORTED VERBATIM from the edge copy (2026-08-13).

    Moves the emitted step FREQUENCY off the subject's own. Applied root-relative when the root is
    locked, so the body keeps the real path and real ground speed and only the limb cycling
    changes. NOTE, measured: this is smooth in the TIME MAPPING but scales every derivative, so
    jerk goes as roughly `rate^3` and the limbs beat against a body still moving at the real speed
    -- rate 1.18 measured jerk 1.87x real, rate 0.85 measured 1.68x, against `q1s`'s 1.19x.
    """
    if rate == 1.0 or rate <= 0 or seq.shape[0] < 4:
        return seq
    T = seq.shape[0]
    t = np.arange(T) * float(rate)
    per = 2 * (T - 1) if T > 1 else 1
    t = np.abs(np.mod(t, per))
    t = np.where(t > T - 1, per - t, t)
    return _resample_at(seq, t, kind)


def _pose_smooth(seq, sigma_frames):
    """Temporal Gaussian low-pass on the pose -- randomised PEAKINESS of the speed profile.

    WHY (2026-08-14). The descriptor ablation (ledger SS A.2x) shows the shipped arm's whole
    residual lift lives in the five speed/acceleration statistics, and that a UNIFORM rate change
    cannot remove all of it: rate scales median/p90 speed together, so ratios like p90/median
    survive it untouched. Low-passing by a per-sequence random amount moves exactly those ratios,
    because it attenuates the high-frequency tail (p90 acceleration, std speed) far more than the
    median. It is the second of the two independent directions in that subspace.

    It is also the only anti-reID knob measured so far that cannot make the render WORSE on the
    axis the owner rejected `cadence_amp` for: a low-pass is monotonically smoothing, so jerk after
    it is bounded above by jerk before it. Sigma is drawn in SECONDS and converted with fps so the
    same preset means the same thing at 10 fps on the phone and at 25 fps in the lab corpus.
    Applied root-relative by the caller, so the travelled path is untouched.
    """
    if sigma_frames <= 0 or seq.shape[0] < 3:
        return seq
    rad = max(1, int(np.ceil(3.0 * sigma_frames)))
    off = np.arange(-rad, rad + 1)
    k = np.exp(-0.5 * (off / float(sigma_frames)) ** 2)
    k /= k.sum()
    T = seq.shape[0]
    idx = np.clip(np.arange(T)[:, None] + off[None, :], 0, T - 1)   # edge-clamp, no wrap
    return np.einsum("w,twjc->tjc", k, seq[idx])


def _apply_projection_fit(out, kp_seq, jitter=0.0, seed=0):
    """Squeeze the emitted skeleton's WIDTH back to the subject's observed shoulder span.

    The collapse target is ONE canonical body defined in FRONTAL anthropometry, but the subject is
    a 2-D PROJECTION of a real body, so foreshortening compresses whichever axis points away from
    the camera and a frontal template cannot match both axes at once. Forcing
    spine = 1.231 x shoulder therefore breaks whichever axis it does not measure. MEASURED, both
    directions, on the two pinned cloud clips:
        A_corridor (side-on)      shoulder scale -> 0.49x height    spine scale -> ~2x shoulder width
        B_atrium   (bust-cropped) shoulder scale -> 1.87x height    spine scale -> 0.53x shoulder width
    and the cost lands on JOINT CONTAINMENT -- the fraction of emitted joints inside the mask hole,
    the only registration signal the generator gets, since a joint outside the hole instructs it to
    draw a limb where it has no canvas:
        A_corridor  shoulder 92.8 %  spine 62.7 %      B_atrium  shoulder 62.7 %  spine 92.3 %
    Each mode wins one clip and loses the other, so NEITHER generalises.

    Taking the height from the observed spine and the width from the observed shoulder span renders
    the SAME canonical body under the subject's own projection: corridor 93.1 % / atrium 96.7 %,
    the only arm >= shipped on both. Privacy is unchanged in kind -- both scalars come from the
    same already-composited body, the k-same collapse is untouched (every subject still lands on
    one shared shape), and the mask discloses the silhouette's aspect anyway.
    """
    obs = _observed_shoulder_px(kp_seq)
    emi = _observed_shoulder_px(out)
    if not (obs > 1.0 and emi > 1.0):
        return out                                   # never confidently seen -> leave it alone
    sx = obs / emi
    # ASPECT JITTER (2026-08-13, ported verbatim from the edge copy). `sx` is a per-clip readout of
    # how broad THIS subject is, and because it scales X only it re-injects subject-specific bone
    # ratios -- the measured mechanism behind "projection_fit leaves 97 % of the between-subject
    # shape difference intact". Randomising it per SEQUENCE keeps the correction (posture) while
    # destroying the precision of what it reveals. Own RNG stream; bit-identical at jitter=0.
    if jitter:
        sx *= float(_rng(seed * 3571 + 29).uniform(1.0 - jitter, 1.0 + jitter))
    # Vectorised over frames (2026-08-02). Squeeze about EACH frame's own trunk midline so the
    # walking trajectory is untouched; frames with no visible trunk joint are left alone, exactly
    # as the per-frame loop did. Verified max|delta| = 0 against that loop on both pinned clips.
    live = np.abs(out).sum(2) > 0                                   # (T,17)
    tj = np.array([5, 6, 11, 12])
    tl = live[:, tj]                                                # (T,4)
    cnt = tl.sum(1)                                                 # (T,)
    cx = (out[:, tj, 0] * tl).sum(1) / np.maximum(cnt, 1)           # (T,)
    hit = (cnt > 0)[:, None] & live                                 # (T,17)
    cxb = cx[:, None]
    out[:, :, 0] = np.where(hit, cxb + (out[:, :, 0] - cxb) * sx, out[:, :, 0])
    return out


_HEAD_BLOCK = (0, 1, 2, 3, 4)          # nose, eyes, ears - the joints the face rig is placed from

_ANGLE_MIRROR_ENV = os.environ.get("MIRAGE_ANGLE_MIRROR", "off")


def _mirror_signs(kp_seq, mode):
    """VERBATIM port of pose_anon_edge._mirror_signs - see that docstring for the measurements
    (A_corridor common-mode lean, and the 57.9 % knee-crossing that ruled plain "mirror" out)
    and for the warning that the privacy effect is unmeasured until an arm is run.

    Per-EDGE +/-1 that makes a paired-limb rotation ANTI-SYMMETRIC instead of common-mode.

    WHY (2026-08-07, owner report: "the feet are towards left/centre of screen while the hips area
    is more towards right/away from centre, which is not natural").

    `angle_off` is keyed by GROUP and both members of a pair share the group name - `(11,13,
    "thigh")` and `(12,14,"thigh")` are rotated by the SAME angle in `_rebuild_frame`. So the
    perturbation lands almost entirely in COMMON mode: the whole lower body pivots about the hips
    like a leaning tower, instead of the stance opening or closing. Measured on the pinned
    A_corridor clip (`_e2e/gait_close_20260807/leg_geometry.py`), thigh angle from vertical:

        arm                        COMMON (both legs)      DIFF (stance)
        real person                     -1.67 deg              +3.21 deg
        g6tq / g9proj / g10       -12.89 / -13.42 / -12.92    -0.42 / -0.40 / -0.18

    i.e. ~13 deg of common-mode lean added and essentially nothing added to the stance. It is the
    identical failure the axial exemption fixed for the shoulder line (§A.2i-1: one `clav` offset
    rotates MS->5 and MS->6 together and tilts the whole shoulder line).

    TWO MODES, because the obvious fix is not sufficient:
      "mirror"   negate the rotation on the RIGHT member of every pair. Removes the lean - measured
                 A_corridor common-mode -13.04 deg -> -0.92 deg off the real person - but the drawn
                 angle is random-signed, so half the time the pair converges instead of opening and
                 🔴 THE KNEES CROSS: measured 57.9 % of frames crossed, knee separation +0.892 ->
                 -0.236 hip-widths (real +1.109). That trades one unnatural artifact for a worse one.
      "outward"  anti-symmetric AND outward-only: `|theta|` with the direction forced AWAY from the
                 body midline, so the stance can only WIDEN, never cross. This is the same
                 outward-only discipline the mask ops already follow (displace / radiallp / ksame
                 are all outward-only, for the §2 superset guarantee).

    The outward DIRECTION is derived per clip from the subject's own facing - `sign(mean(x_Lsh -
    x_Rsh))`, using the shoulders because they are the landmark pair measured to be reliable
    (confidence 1.000, never off-frame, vs hips up to 51 % off-frame). No fitted constant, no
    per-clip tuning: a subject facing the other way gets the mirrored sign automatically.

    Only the five LEFT/RIGHT groups are signed (thigh, shin, uarm, farm, clav); the axial groups
    (spine, neck, hip, face) have no partner to be anti-symmetric with and keep sign +1.

    🔴 THE PRIVACY EFFECT IS UNMEASURED UNTIL AN ARM IS RUN. "mirror" preserves the per-limb
    rotation magnitude exactly (verified to 1e-14) but changes the RELATIVE leg geometry, which is
    closer to what a gait recogniser reads than a global lean is. "outward" additionally removes
    the sign entropy of the perturbation, which can only help an attacker. Default OFF; any arm
    shipping with either needs its own TM1 and TM3 row.
    """
    sgn = np.zeros(len(EDGES)) + 1.0
    if str(mode).strip().lower() not in ("mirror", "outward"):
        return sgn
    kp = np.asarray(kp_seq, np.float64)
    ok = (np.abs(kp[:, 5]).sum(1) > 0) & (np.abs(kp[:, 6]).sum(1) > 0)
    face = float(np.sign(np.mean((kp[ok, 5, 0] - kp[ok, 6, 0]))) or 1.0)   # +1 => left is at +x
    for ei, (_p, c, g) in enumerate(EDGES):
        if g not in _LR_GROUPS or not isinstance(c, int):
            continue
        left = (c % 2 == 1)                      # COCO: odd = left, even = right for 5..16
        # A positive rotation moves a DOWNWARD-pointing bone's tip towards -x (see _rebuild_frame:
        # x' = ca*x - sa*y with y > 0). Outward for a left-side joint therefore means a positive
        # rotation when "left" sits at -x, and a negative one when the subject faces the other way.
        sgn[ei] = (1.0 if left else -1.0) * (-face)
    return sgn


def _apply_head_anchor(out, kp_seq, mode):
    """VERBATIM port of pose_anon_edge._apply_head_anchor - see that docstring for the mechanism,
    the measurement that motivated it and the privacy warning.

    Translate the emitted HEAD BLOCK so the emitted nose sits on the REAL nose.

    WHY (2026-08-07, owner request: "keep face position same for all cases, vertically too").
    The synthetic face is placed from the emitted head joints, so where the face lands is
    `shoulder_mid + neck_len * neck_dir` of the EMITTED skeleton. Canonicalisation rebuilds the
    neck from the anatomical template, not from this person, and that alone moves the face:
    measured on the pinned clips, `g9proj`/`g10` emit a neck of 1.13x real on A_corridor and
    0.66x on B_atrium, putting the face 29 px high and 48 px low respectively (0.33 / 0.19
    shoulder widths). The body registration is already good on those arms (shoulder midpoint
    within 3 px); it is specifically the head that is misplaced.

    WHAT IT DOES. A rigid per-frame translation of joints 0..4 only. Bone lengths inside the head
    block, the emitted shoulder width, and every limb are untouched, so the face RIG's size
    (io = 0.26 x emitted shoulder) does not change either.

    🔴 IT IS A PRIVACY COST AND MUST NOT BE ENABLED ON A PRICE THAT WAS ASSUMED. Placing the
    emitted nose on the real nose hands back the real nose TRAJECTORY, and - because these arms
    already put the shoulders within ~3 px of the real ones - very nearly the real NECK VECTOR,
    which is an anthropometric identity channel. §B.11-gate measured the legacy full
    re-registration (uniform scale + rigid head anchor + head template) at TM1 NM 8.97 %, i.e.
    +2.99 pp over the anonymiser alone. This knob is a strict SUBSET of that change, so its cost
    is bounded above by that figure and is not known until measured. Default "off"; every arm
    that ships with it on needs its own TM1 and TM3 row.

    Frames where the real nose was never observed are left exactly as canonicalisation produced
    them - a (0,0) "not detected" marker must never be used as an anchor.
    """
    if str(mode).strip().lower() not in ("nose", "on", "1", "true"):
        return out
    real = np.asarray(kp_seq, np.float64)[:, 0, :]                    # (T,2) real nose
    ok = np.abs(real).sum(1) > 0
    if not ok.any():
        return out
    d = np.zeros_like(real)
    d[ok] = real[ok] - out[ok, 0, :]
    out = out.copy()
    out[:, _HEAD_BLOCK, :] += d[:, None, :]
    return out


def anonymize_v2(kp_seq, template, seed=0, fps=25.0,
                 k_prototypes=None,          # None/1 => single template (max privacy)
                 len_jitter=0.0,             # small per-clip length noise (naturalness)
                 asymmetry_sigma=0.0,        # independent L/R length break
                 angle_const_deg=0.0,        # per-limb constant angular offset
                 angle_drift_deg=0.0,        # per-limb slow-drift angular offset
                 cadence_amp=0.0,            # cadence/rhythm warp amplitude (frac)
                 cadence_period_s=0.7,       # gait-band period for the warp
                 lowfreq_amp_frac=0.03,      # positional drift (naturalness)
                 lowfreq_period_s=1.2,
                 do_canon=True,
                 center_root=False,          # subtract per-frame mid-hip (kill global trajectory)
                 # ---- ADDITIVE knobs (2026-07-31). Both default 0.0 => byte-identical output
                 # and identical RandomState consumption to every number measured before today.
                 lr_asymmetry_sigma=0.0,     # TRUE independent left/right length factors (static)
                 limb_phase_amp=0.0,         # per-limb-chain cadence decorrelation amplitude
                 limb_phase_period_s=0.7,
                 # ---- ADDITIVE (2026-08-02). Ported VERBATIM from the edge copy so a TM3
                 # number measured here describes the code that actually ships.
                 angle_groups=None,          # None => EVERY group rotates (shipped)
                 projection_fit=False,       # width -> observed shoulder span
                 # ---- ADDITIVE (2026-08-13). Ported VERBATIM from the edge copy. All three are
                 # strict no-ops at their defaults (own RNG streams), so every arm measured before
                 # today is bit-identical. Each opens a channel the q-family does not currently
                 # touch -- required because neither `seeded_global_scale_max` nor `cadence_amp`
                 # moves top-5 lift beyond seed noise (§A.2w-3).
                 projection_fit_jitter=0.0,  # randomise the ASPECT correction => shape channel
                 limb_phase_offset_s=0.0,    # constant per-limb time SHIFT => inter-limb phase
                 pose_rate=1.0,              # uniform pose re-timing => step frequency
                 # ---- ADDITIVE knobs (2026-08-14). Both attack the DYNAMICS block, which the
                 # descriptor ablation (SS A.2x) shows is where the shipped arm's entire residual
                 # lift now lives -- shape and inter-limb phase already score AT OR BELOW chance,
                 # so nothing is left to win there. Defaults 0.0 are strict no-ops that consume no
                 # RandomState, so every number recorded before 2026-08-14 reproduces bit-for-bit.
                 pose_rate_jitter=0.0,       # PER-SEQUENCE rate draw. A fixed `pose_rate` is a
                                             #   constant the corpus-wide z-score removes entirely;
                                             #   only a per-clip DRAW adds variance to the channel.
                 pose_smooth_max_s=0.0,      # per-sequence temporal low-pass, sigma ~ U[0, max]
                 pose_rate_two_sided=False,  # draw in [1-j, 1+j] instead of [1-j, 1]: same width,
                                             #   centred at 1.0, so limbs are not only ever slowed
                 # ---- ADDITIVE (2026-08-14, after §A.2y-5 measured `pose_rate_jitter` costing
                 # +40 % jerk at the shipped 10 fps). Two independent ways to pay that back:
                 pose_smooth_max_frames=0.0,  # low-pass with sigma in FRAMES, not seconds. The
                                              #   seconds form is INERT at 10 fps (§A.2x-5); this
                                              #   one bites equally at any rate. Blurs the kink.
                 root_speed_amp=0.0,          # walking-SPEED modulation, in seconds of
                                              #   re-timing along the root's OWN path
                 root_speed_period_s=2.5,     # modulation period
                 root_speed_max_frac=0.016,   # HARD clamp on root displacement, as a
                                              #   FRACTION of visible body height (scale-free)
                 limb_swing_amp=0.0,          # per-chain angular-excursion scale, U[1-a, 1+a].
                                              #   Reaches DYNAMICS without re-timing, so it has
                                              #   no root beat and no interpolation kink.
                 pose_rate_kind="linear",     # "catmull": C1 resampling, removing the kink at
                                              #   source instead of blurring it after the fact
                 # ---- ADDITIVE (2026-08-07). Ported VERBATIM from the edge copy.
                 head_anchor="off",          # "nose": put the emitted face on the REAL face
                 angle_mirror=False,         # anti-symmetric paired-limb rotation
                 # ---- ADDITIVE knob (2026-08-08). Default False reproduces every previous number
                 # bit-for-bit and consumes the RandomState identically.
                 cadence_root_lock=False,    # re-time the POSE without moving the BODY
                 # ---- ADDITIVE bounded retargeting (2026-08-11). Defaults are strict no-ops.
                 seeded_global_scale_max=0.0,
                 seeded_region_scale_max=0.0,
                 proximity_cap_frac=0.0,
                 proximity_scores=None):
    """kp_seq:(T,17,2) -> anonymised (T,17,2). `template`: {group: target length}.
    Defaults reproduce a near-identity pass-through with only positional drift; the
    privacy knobs are opt-in so ablations are clean.

    The two 2026-07-31 knobs draw from a SEPARATE RandomState (`_rng(seed*7919+13)`), never
    from `r`. Deliberate: switching them on must not shift `const_ang`/`len_factors` downstream,
    or an A/B of the new knob would also be an A/B of a different angle draw."""
    kp_seq = np.asarray(kp_seq, np.float64)
    if center_root and float(proximity_cap_frac) != 0.0:
        raise ValueError("center_root and proximity_cap_frac are incompatible: root centering "
                         "changes coordinate frames, so a detector-coordinate cap is undefined")
    T = kp_seq.shape[0]
    r = _rng(seed)

    # (1) static collapse: per-group length factors snapping clip -> template
    if do_canon:
        gl = group_lengths(kp_seq)
        tgt = dict(template)
        if k_prototypes and k_prototypes > 1:
            # jitter the target within a small band and snap to the nearest of K
            # deterministic prototypes (k-anonymity diversity); simplest realisation:
            protos = [ {g: template[g] * (1 + (2*(j/(k_prototypes-1))-1)*0.12) for g in _GROUPS}
                       for j in range(k_prototypes) ]
            j = r.randint(0, k_prototypes)
            tgt = protos[j]
        len_factors = {g: tgt[g] / (gl[g] + 1e-6) for g in _GROUPS}
    else:
        len_factors = {g: 1.0 for g in _GROUPS}

    if len_jitter:
        for g in _GROUPS:
            len_factors[g] *= float(1 + r.randn() * len_jitter)
    if asymmetry_sigma:
        for g in _LR_GROUPS:
            len_factors[g] *= float(1 + r.randn() * asymmetry_sigma)

    # (1a) BOUNDED SEEDED STANDARD-SKELETON SCALE. Separate RNG stream, static for the clip.
    len_factors = _bounded_seeded_scales(len_factors, seed, seeded_global_scale_max,
                                         seeded_region_scale_max)

    # (1b) TRUE left/right length break (2026-07-31). Per-EDGE, so the left and right member of
    # a pair get INDEPENDENT factors -- which `asymmetry_sigma` above structurally cannot do
    # (one factor per group name, shared by both sides; the misnomer documented in
    # pose_anon_edge.py:255-263). Drawn ONCE per clip => completely static => zero added
    # temporal jitter, which is the point: the cloud's invented-hands defect is driven by
    # per-frame jitter (§B.34/§B.37) and a static shape perturbation is invisible to it.
    edge_scale = None
    if lr_asymmetry_sigma:
        r2 = _rng(seed * 7919 + 13)
        edge_scale = np.ones(len(EDGES))
        for ei, (_p, _c, g) in enumerate(EDGES):
            if g in _LR_GROUPS:
                edge_scale[ei] = float(1 + r2.randn() * lr_asymmetry_sigma)

    # (2a) per-limb angular offsets: constant + slow drift, per group (asymmetric)
    ac = np.deg2rad(angle_const_deg)
    ad = np.deg2rad(angle_drift_deg)
    _rot = _angle_group_mask(angle_groups)
    # Draw for EVERY group and zero afterwards, never skip the draw: skipping would shift the RNG
    # stream and change the LIMB angles too, so a quiet-vs-shipped A/B would no longer be
    # single-variable -- and with angle_groups=None the stream must stay bit-identical to every
    # number measured before 2026-08-02.
    const_ang = {g: (r.randn() * ac) * (1.0 if _rot[g] else 0.0) for g in _GROUPS}
    drift_ang = {g: (_lowfreq_noise(T, max(1.0, lowfreq_period_s * fps / 3.0), ad, seed * 17 + gi * 101 + 3)
                     if ad > 0 else np.zeros(T)) * (1.0 if _rot[g] else 0.0)
                 for gi, g in enumerate(_GROUPS)}

    # positional low-freq drift (naturalness), per joint/axis
    scale = np.median([_body_scale(kp_seq[t]) for t in range(T)])
    amp = lowfreq_amp_frac * scale
    if amp > 0:
        nse = np.stack([[_lowfreq_noise(T, max(1.0, lowfreq_period_s * fps / 3.0), amp, seed * 131 + j * 7 + a)
                         for a in range(2)] for j in range(17)], axis=0)  # (17,2,T)
    else:
        nse = np.zeros((17, 2, T))

    _am = str(angle_mirror if angle_mirror else _ANGLE_MIRROR_ENV).strip().lower()
    _am = {'true': 'mirror', '1': 'mirror', 'false': 'off', '0': 'off',
           '': 'off', 'none': 'off'}.get(_am, _am)
    ang_sign = _mirror_signs(kp_seq, _am) if _am in ('mirror', 'outward') else None
    ang_abs = (_am == 'outward')

    out = np.empty_like(kp_seq)
    for t in range(T):
        angle_off = {g: const_ang[g] + drift_ang[g][t] for g in _GROUPS}
        rt = _rebuild_frame(kp_seq[t], len_factors, angle_off, edge_scale, ang_sign,
                            ang_abs)
        out[t] = rt + nse[:, :, t]

    # (2b) cadence / rhythm warp (temporal)
    # CADENCE_ROOT_LOCK (2026-08-08, default OFF). Verbatim mirror of the edge copy - see
    # `pose_anon_edge.anonymize_v2` step 2b for the measurement. In one line: `_cadence_warp` is a
    # time re-sample of ABSOLUTE coordinates, so it drags the whole figure along its own walking
    # path (run3 c2: 123.7 px mean mid-hip error, all of it this one knob). The lock warps the
    # root-relative pose and re-attaches it to the true per-frame mid-hip.
    if cadence_root_lock is False:
        cadence_root_lock = os.environ.get("MIRAGE_CADENCE_ROOT_LOCK", "0") \
            not in ("0", "", "false", "False", "off")
    if cadence_amp > 0:
        if cadence_root_lock:
            _root = 0.5 * (out[:, 11:12, :] + out[:, 12:13, :])
            out = _cadence_warp(out - _root, cadence_amp, cadence_period_s, fps,
                                seed * 991 + 1) + _root
        else:
            out = _cadence_warp(out, cadence_amp, cadence_period_s, fps, seed * 991 + 1)
    # (2c) per-limb cadence decorrelation (2026-07-31) -- AFTER the global warp, so it perturbs
    # exactly the inter-limb PHASE relationships the global warp leaves intact. No-op at amp 0.
    if limb_phase_amp > 0:
        out = _limb_phase_warp(out, limb_phase_amp, limb_phase_period_s, fps, seed)
    # (2c-ii) CONSTANT per-limb phase SHIFT (2026-08-13, ported verbatim). Same channel as (2c),
    # by shifting rather than re-sampling, so it costs no smoothness. No-op at 0.
    if limb_phase_offset_s > 0:
        out = _limb_phase_offset(out, limb_phase_offset_s, fps, seed)
    # (2c-iii) UNIFORM pose re-timing (2026-08-13, ported verbatim): moves the emitted STEP
    # FREQUENCY off the subject's own. Root-relative when the root is locked. No-op at rate 1.0.
    # (2c-iv) PER-SEQUENCE rate DRAW (2026-08-14). A FIXED `pose_rate` is a corpus-wide constant
    # that the attacker's z-score removes entirely -- which is why r3 barely moved this channel.
    # Only a per-clip DRAW adds variance, and variance is what breaks a channel.
    # 🔴 Drawn one-sided in [1-j, 1] because rate^3 scales the limb-relative jerk -- but MEASURED,
    # that does NOT make the knob free: d2 costs 1.12x q1s's jerk. The reasoning was wrong because
    # under the root lock the slowed limbs BEAT against a root still travelling at the real speed,
    # so absolute jerk rises even though the root-relative part falls. Same mechanism as r3's
    # 1.68x. Pair it with `pose_smooth_max_s`, which more than pays the cost back (d3 = 0.94x).
    _rate_eff = float(pose_rate)
    if pose_rate_jitter > 0:
        if not 0.0 <= pose_rate_jitter < 1.0:
            raise ValueError("pose_rate_jitter must be in [0, 1)")
        _rj = _rng(seed * 15485863 + 32452843)
        # ONE draw either way, same stream, so two_sided=False stays bit-identical to §A.2y's d2.
        _rate_eff *= (1.0 + _rj.uniform(-pose_rate_jitter, pose_rate_jitter)
                      if pose_rate_two_sided else 1.0 - _rj.uniform(0.0, pose_rate_jitter))
    if _rate_eff != 1.0:
        if cadence_root_lock:
            _root2 = 0.5 * (out[:, 11:12, :] + out[:, 12:13, :])
            out = _pose_rate_warp(out - _root2, _rate_eff, fps, pose_rate_kind) + _root2
        else:
            out = _pose_rate_warp(out, _rate_eff, fps, pose_rate_kind)
    # (2c-v) PER-SEQUENCE temporal LOW-PASS (2026-08-14) -- the r-invariant direction of the same
    # dynamics block. Root-relative under the root lock so the travelled path is bit-untouched.
    if pose_smooth_max_s > 0 or pose_smooth_max_frames > 0:
        _rs = _rng(seed * 22801763 + 15486071)
        # ONE draw from the SAME stream either way, so the seconds form stays bit-identical to d1/d3.
        _sig = (_rs.uniform(0.0, float(pose_smooth_max_frames)) if pose_smooth_max_frames > 0
                else _rs.uniform(0.0, float(pose_smooth_max_s)) * float(fps))
        if cadence_root_lock:
            _root3 = 0.5 * (out[:, 11:12, :] + out[:, 12:13, :])
            out = _pose_smooth(out - _root3, _sig) + _root3
        else:
            out = _pose_smooth(out, _sig)
    # (2c-vii) ROOT SPEED (2026-08-14) -- the walking-pace channel, clamped for containment.
    if root_speed_amp > 0:
        out = _root_speed_warp(out, root_speed_amp, root_speed_period_s, fps, seed,
                               root_speed_max_frac)
    # (2c-vi) LIMB SWING AMPLITUDE (2026-08-14) -- the only dynamics knob that does not
    # re-time anything. No-op at 0.0 and consumes no RandomState there.
    if limb_swing_amp > 0:
        out = _limb_swing_amp(out, limb_swing_amp, seed)
    # (2d) PROJECTION FIT (2026-08-02) -- see _apply_projection_fit for the measurement.
    if projection_fit or             os.environ.get("MIRAGE_POSE_SCALE_FROM", "").strip().lower() == "projected":
        out = _apply_projection_fit(out, kp_seq, jitter=projection_fit_jitter, seed=seed)
    # (2e) HEAD ANCHOR (2026-08-07) -- put the emitted FACE where the real face is.
    out = _apply_head_anchor(out, kp_seq, head_anchor)
    # (2f) STRICT PROXIMITY ENVELOPE (2026-08-11).
    out = _apply_proximity_envelope(out, kp_seq, proximity_cap_frac,
                                    scores=proximity_scores)

    # (3) optional: remove the global walking trajectory (per-frame root-centering).
    #     Tests whether the residual adaptive-adversary identity lives in the
    #     preserved mid-hip path rather than in the perturbed pose dynamics.
    if center_root:
        mh = 0.5 * (out[:, 11:12, :] + out[:, 12:13, :])   # (T,1,2)
        out = out - mh
    return out
