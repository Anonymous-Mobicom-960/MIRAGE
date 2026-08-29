"""
pose_anon_edge -- on-device anti-re-ID pose anonymiser for MIRAGE Tier-1.
=========================================================================
SELF-CONTAINED numpy-only port of the lab-validated anonymiser
(tier1_lab/segbench/pose_anon_v2.anonymize_v2 + the helpers it pulls from
pose_anon). The lab tree does NOT exist on the Pi, so everything needed at
runtime is replicated here VERBATIM -- numeric parity with the lab code is
asserted by test_pose_anon_edge.py (same input/template/seed => allclose).

WHY (threat model §2, gait channel): the emitted pose.json carries body-17
keypoint dynamics -- gait, limb proportions, posture -- all re-identifiable
soft biometrics even with the face landmarks zeroed. The lab evaluation
(tier1_lab/reid_eval, GaitGraph2 adversary on CASIA-B) showed:
  * static collapse (do_canon): every skeleton's limb proportions snap to ONE
    shared population template -> the anthropometric channel carries zero
    identity (k-same, non-invertible);
  * L4 dynamic knobs (cadence warp .60, angle const 20deg, drift 15deg,
    asymmetry .12): perturb the gait dynamics where most identity lives;
  * PER-SEQUENCE random seed (secrets, never identity/content-derived): no
    stable pseudo-identity across clips;
  * confidence BINARIZED to {0,1}: kills the per-joint confidence trace as an
    identity side-channel (lab strip_conf ablation);
  * center_root=False: the global mid-hip trajectory is KEPT -- mask.mkv
    carries the silhouette position anyway and the Tier-2 composite needs it.

Deployment semantics vs the lab: the lab template is ABSOLUTE bone lengths in
CASIA-B pixels. On device the subject's on-frame size is arbitrary (camera
distance), so we bake the template as unit-free RATIOS (bone length / body
scale) and rebuild an absolute template per clip from the subject's own
auto-computed median body scale (torso length). Proportions still collapse to
the population skeleton (identity destroyed); on-frame size/position stay
correct for the composite. No per-clip constant is hardcoded anywhere --
the scale is auto-detected per clip, the ratios are population-level.
"""
import secrets
import os
import sys
import numpy as np
from person_slots import slot_groups   # sibling edge module (pure stdlib); slot-aware grouping

# ----------------------------------------------------------------------------
# COCO-17 kinematic tree (verbatim from tier1_lab/segbench/pose_anon.py):
# 0 nose 1 leye 2 reye 3 lear 4 rear 5 lsh 6 rsh 7 lelb 8 relb 9 lwr
# 10 rwr 11 lhip 12 rhip 13 lknee 14 rknee 15 lank 16 rank
# Rooted at a virtual mid-hip ('MH'); 'MS' = virtual mid-shoulder.
# Each edge: (parent, child, sym_group) -- symmetric groups share a scale factor.
EDGES = [
    ("MH", "MS", "spine"), ("MH", 11, "hip"), ("MH", 12, "hip"),
    (11, 13, "thigh"), (13, 15, "shin"), (12, 14, "thigh"), (14, 16, "shin"),
    ("MS", 5, "clav"), ("MS", 6, "clav"), ("MS", 0, "neck"),
    (5, 7, "uarm"), (7, 9, "farm"), (6, 8, "uarm"), (8, 10, "farm"),
    (0, 1, "face"), (1, 3, "face"), (0, 2, "face"), (2, 4, "face"),
]
_GROUPS = sorted({e[2] for e in EDGES})
# groups that have a left/right partner (for asymmetry); "spine/neck/face/hip" are axial
_LR_GROUPS = ["thigh", "shin", "clav", "uarm", "farm"]

# RTMPose wholebody-133 layout (mirrors mirage_tier1.py's own slicing:
# body 0..22 kept, face 23..90 zeroed, hands 91..132 kept):
#   0..16 COCO body | 17..19 L foot | 20..22 R foot | 23..90 face (68)
#   91..111 L hand (21) | 112..132 R hand (21)
_FACE = slice(23, 91)
_ATTACH = [  # (wholebody slice, body-17 anchor joint whose delta it follows)
    (slice(17, 20), 15),    # left foot  -> left ankle
    (slice(20, 23), 16),    # right foot -> right ankle
    (slice(91, 112), 9),    # left hand  -> left wrist
    (slice(112, 133), 10),  # right hand -> right wrist
]


# --------------------------------------------------------------------- helpers
# (verbatim ports from tier1_lab/segbench/pose_anon.py)
def _virtual(kp):
    """kp: (17,2). Return dict of positions incl virtual MH, MS."""
    pos = {i: kp[i].astype(np.float64) for i in range(17)}
    pos["MH"] = 0.5 * (kp[11] + kp[12])
    pos["MS"] = 0.5 * (kp[5] + kp[6])
    return pos


def _body_scale(kp):
    """Rough body size (shoulder->hip torso length) for normalising amplitudes."""
    ms = 0.5 * (kp[5] + kp[6]); mh = 0.5 * (kp[11] + kp[12])
    return max(1.0, float(np.linalg.norm(ms - mh)))


def _rng(seed):
    return np.random.RandomState(seed & 0x7fffffff)


def _lowfreq_noise(T, sigma_frames, amp, seed):
    """Band-limited noise: white noise smoothed by a Gaussian temporal kernel ->
    slowly varying (no jitter), and itself low-frequency (not low-pass removable)."""
    r = _rng(seed)
    w = r.randn(T)
    rad = int(max(1, round(3 * sigma_frames)))
    k = np.exp(-0.5 * (np.arange(-rad, rad + 1) / sigma_frames) ** 2); k /= k.sum()
    s = np.convolve(w, k, mode="same")
    # np.convolve(..., "same") returns max(len(w), len(k)) -- NOT len(w). When a person-slot is
    # shorter than the kernel (at fps=15 the cadence kernel is 21 samples, the drift kernel 37)
    # this silently returned MORE samples than the clip has: _cadence_warp then produced a
    # different frame count than it was given and anonymize_pose_log died on `anon - body`
    # (measured: a slot present for 8 frames -> 21 frames out). A person walking briefly through
    # frame is enough to take down the whole clip's emit. Keep the CENTRED T samples; this is a
    # no-op whenever T >= len(k), so long-clip output stays bit-identical (parity test asserts it).
    if s.size != T:
        off = (s.size - T) // 2
        s = s[off:off + T]
    sd = s.std()
    return (s / sd * amp) if sd > 1e-9 else s


# ----------------------------------------------------------- template (k-same)
# (verbatim ports from tier1_lab/segbench/pose_anon_v2.py)
def group_lengths(kp_seq):
    """Per-GROUP median bone length over a clip (T,17,2). {group: length}."""
    per = {g: [] for g in _GROUPS}
    for t in range(kp_seq.shape[0]):
        o = _virtual(kp_seq[t])
        for p, c, g in EDGES:
            per[g].append(float(np.linalg.norm(o[c] - o[p])))
    return {g: float(np.median(v)) if v else 1.0 for g, v in per.items()}


def _rebuild_frame(kp, len_factors, angle_off, edge_scale=None, angle_sign=None,
                   angle_abs=False):
    """Rebuild 17 joints from mid-hip, applying per-group length scale AND a 2D
    rotation of each bone direction by angle_off[group] (radians). Mid-hip fixed.

    `edge_scale` (optional, added 2026-07-31) is a PER-EDGE multiplier aligned with EDGES,
    used by the true left/right asymmetry break (`lr_asymmetry_sigma`). `len_factors` is keyed
    by GROUP and the left and right member of a pair share the group name, so a per-group dict
    can never express "left thigh 3 % longer than right". Default None keeps the exact original
    arithmetic (no extra multiply) so numeric parity with the lab code is untouched.

    `angle_sign` (optional, added 2026-08-07) is a PER-EDGE multiplier on the ROTATION, aligned
    with EDGES, used by the ANTI-SYMMETRIC angle mode (`angle_mirror`). Same reason as above:
    `angle_off` is keyed by GROUP, so the left and right member of a pair are rotated by the SAME
    angle and the pair swings as a unit. Default None is the original arithmetic exactly."""
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




# Restore the SUBJECT'S OWN torso orientation after the collapse (2026-07-27).
# Measured before enabling: emitted shoulder tilt +0.35 deg +- 0.35 vs the real person's
# +5.03 deg +- 1.06, and head roll -3.04 vs +1.95 -- the character rendered too upright.
# Orientation is POSE, not SHAPE: limb lengths and ratios are untouched, so this cannot weaken
# the identity collapse. Set MIRAGE_RESTORE_TORSO_ROLL=0 to disable for an A/B.
# 🔴 MEASURED A NO-OP -- DEFAULT OFF (2026-07-27). The hypothesis was that the collapse levels the
# torso on a bust framing. It does not. Shoulder tilt through the pipeline, same clip, n=100:
#     RAW detector (anon OFF)  +0.38 +- 0.33
#     FIXED sticks (0/0)       +0.34 +- 0.36    <- already reproduces the detector
#     L4 shipped (old sticks) +10.53 +- 14.23   <- THIS was the visible tilt, already fixed by 0/0
# Enabling this changed the emitted tilt by 0.02 deg, because the emitted line ALREADY matches the
# observed one. The residual gap to MediaPipe's +5.03 +- 1.06 is RTMPose-vs-MediaPipe estimator
# disagreement on the same footage, NOT something the anonymiser introduced -- so "correcting" it
# here would be fitting our own skeleton to a second estimator's opinion.
# Kept (off) because it is the correct mechanism should a future framing genuinely need it.
RESTORE_TORSO_ROLL = os.environ.get("MIRAGE_RESTORE_TORSO_ROLL", "0") not in ("0", "false", "False")

def _restore_torso_roll(rebuilt, kp_obs):
    """Rotate the rebuilt skeleton so its SHOULDER LINE matches the OBSERVED one.

    🔴 WHY. On a bust/chest-up framing the hips are 0 % observed (measured on a10), so `_virtual`
    SYNTHESISES mid-hip and the whole chain is rebuilt from a fabricated root. The result is a torso
    snapped to canonical level: measured shoulder tilt **+0.35 deg +- 0.35** against the real
    person's steady **+5.03 deg +- 1.06**, and head roll -3.04 deg against the real +1.95 deg. The
    character therefore renders too upright and slightly counter-rotated -- the "head tilt".

    THE FIX IS A RIGID ROTATION of the whole frame about mid-shoulder, by the angle between the
    emitted and the observed shoulder line.

    PRIVACY: this restores ORIENTATION (a pose attribute), not SHAPE. Every limb length and ratio
    stays exactly as the collapse set it, and `do_canon` is untouched -- so the identity channel the
    anonymiser destroys is unaffected. A rigid rotation cannot re-introduce body proportions.

    No-ops unless BOTH shoulders are genuinely observed, so it can never invent an orientation.
    """
    l, r = kp_obs[5][:2], kp_obs[6][:2]
    if not (np.isfinite(l).all() and np.isfinite(r).all()):
        return rebuilt
    if abs(r[0] - l[0]) < 1e-6 and abs(r[1] - l[1]) < 1e-6:
        return rebuilt
    obs = np.arctan2(r[1] - l[1], r[0] - l[0])
    el, er = rebuilt[5], rebuilt[6]
    cur = np.arctan2(er[1] - el[1], er[0] - el[0])
    d = obs - cur
    d = (d + np.pi) % (2 * np.pi) - np.pi          # shortest signed rotation
    if abs(d) < 1e-9:
        return rebuilt
    ms = 0.5 * (rebuilt[5] + rebuilt[6])
    ca, sa = np.cos(d), np.sin(d)
    v = rebuilt - ms
    out = np.empty_like(rebuilt)
    out[:, 0] = ca * v[:, 0] - sa * v[:, 1]
    out[:, 1] = sa * v[:, 0] + ca * v[:, 1]
    return out + ms


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
# relationship -- left-arm vs right-arm swing, arm vs contralateral leg -- survives it intact.
# Those relationships are a large part of what a gait recogniser reads. `limb_phase_amp` gives
# each limb chain its OWN independent time warp.
#
# Each entry is (anchor_joint, [joints hanging off it]). Only the OFFSETS from the anchor are
# re-sampled in time; the anchor stays on the global clock. Consequences, by construction:
#   * the skeleton never comes apart (the chain is re-attached to the anchor's true position);
#   * no bone length changes -- the offset taken at t' has the length the collapse already fixed,
#     so `do_canon`'s identity collapse is untouched;
#   * it is a re-SAMPLE of a smooth track, not additive noise: the same family of operation as
#     `cadence_amp`, which POSE_KNOB_ATTRIBUTION measured at jerk 0.45 vs the real input's 0.50.
# Default 0.0 => this whole block is skipped and the output is bit-identical to before.
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
# The target template is already a STANDARD skeleton.  These knobs multiply that standard target
# by fresh per-SEQUENCE values before forward-kinematic reconstruction.  The factors are static for
# the whole clip (no breathing/jitter), symmetric left/right, and come from a separate RNG stream so
# every historical configuration remains bit-identical when the maxima are zero.
_SEEDED_SCALE_REGIONS = (
    ("head", ("neck", "face")),
    ("trunk", ("spine", "clav", "hip")),
    ("arms", ("uarm", "farm")),
    ("legs", ("thigh", "shin")),
)

# Distal joints may move a little farther than the face/trunk, but every value is still multiplied
# by the caller's strict fraction-of-observed-height cap.
_PROXIMITY_WEIGHTS = np.array([
    0.60, 0.60, 0.60, 0.60, 0.60,   # nose / eyes / ears
    0.65, 0.65,                       # shoulders
    0.85, 0.85, 1.00, 1.00,           # elbows / wrists
    0.65, 0.65, 0.85, 0.85, 1.00, 1.00,  # hips / knees / ankles
], np.float64)


def _bounded_seeded_scales(len_factors, seed, global_max=0.0, region_max=0.0):
    """Multiply canonical bone targets by bounded, clip-static seeded factors.

    `global_max=.03` means one isotropic draw in [0.97, 1.03]. `region_max=.04` adds one
    independent draw per HEAD/TRUNK/ARMS/LEGS region in [0.96, 1.04]. Left and right share the
    same group factor, so this cannot create unequal paired limbs. The seed must be fresh per clip;
    a person-stable seed would create a linkable pseudonym.
    """
    gm, rm = float(global_max), float(region_max)
    if not (0.0 <= gm < 0.5 and 0.0 <= rm < 0.5):
        raise ValueError("seeded scale maxima must be in [0, 0.5)")
    if gm == 0.0 and rm == 0.0:
        return len_factors
    out = dict(len_factors)
    rs = _rng(seed * 104729 + 7919)   # independent of every historical draw in `r`
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
    """Keep every emitted joint within a strict fraction of the detected visible height.

    A single blend coefficient is used for the WHOLE sequence. That avoids frame-to-frame clipping
    switches (jitter) and preserves the seeded transform's temporal coherence. Head/trunk caps are
    tighter via `_PROXIMITY_WEIGHTS`; wrists/ankles receive the full allowance.
    """
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


# ------------------------------------------------------------------ main entry
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

    WHY THIS EXISTS (2026-08-13). `_limb_phase_warp` decorrelates inter-limb phase by re-SAMPLING
    each chain on its own wobbling clock, and a re-sample changes local speed, which is what shows
    up as the owner-reported "sudden" motion (§A.2v: cadence 1.78x real jerk). A pure SHIFT does
    not: each limb keeps its own trajectory and smoothness exactly, it merely arrives early or
    late. The inter-limb phase relationships a gait recogniser reads -- arm vs contralateral leg,
    left vs right swing -- are destroyed just as thoroughly by a shift as by a warp, because what
    the recogniser reads is the RELATIVE timing, not the absolute clock.

    Offsets are drawn fresh per SEQUENCE from a separate RNG stream, uniform in [-max_s, +max_s],
    and applied to each chain's OFFSETS FROM ITS ANCHOR, so the skeleton never comes apart and no
    bone length changes (identical guarantees to `_limb_phase_warp`). Edges are handled by
    REFLECTION rather than clamping, so a shifted limb never freezes at the clip boundary.
    """
    if max_s <= 0 or seq.shape[0] < 4:
        return seq
    T = seq.shape[0]
    r = _rng(seed * 6151 + 17)
    out = seq.copy()
    idx = np.arange(T)
    for ci, (anchor, joints) in enumerate(_LIMB_CHAINS):
        dt = float(r.uniform(-max_s, max_s) * fps)            # frames, can be fractional
        t = idx + dt
        # reflect into [0, T-1] so the limb keeps moving at the ends instead of holding a pose
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
    """Re-time the POSE at a CONSTANT rate: the limbs cycle at a different frequency, smoothly.

    WHY (2026-08-13). A subject's own step frequency is an identity cue, and `_cadence_warp`
    attacks it only with a random wobble whose local speed changes are what produce jerk. A
    UNIFORM rate change has, by construction, no local speed variation at all -- it is a single
    linear remap of the timeline -- so it adds no acceleration artefact while moving the emitted
    cadence off the subject's own toward a shared constant.

    Applied to the pose only. The caller passes the root-relative sequence when `cadence_root_lock`
    is on, so the body still travels the real path at the real speed and only the limb cycling
    changes; without the root lock this would also change the walking speed and drag the figure
    off its own trajectory. Reflection at the ends, as in `_limb_phase_offset`.
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
    same preset means the same thing at 10 fps on the phone and at the corpus rate in the lab.
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
    # ASPECT JITTER (2026-08-13). `sx` is the subject's OWN observed shoulder span over the
    # canonical emitted one, so it is a direct, per-clip readout of how broad this person is -- and
    # because it scales X only, it changes every non-vertical bone ratio and hands the emitted
    # proportion set back its subject-specific character. That is the measured mechanism behind
    # "projection_fit leaves 97 % of the between-subject shape difference intact" (§B.63b), i.e.
    # the single largest untapped channel in the q-family. Removing the fit entirely fixes the leak
    # (0.0000 cross-subject difference) but costs posture badly -- every PF=False render came back
    # hunched (q2/q2w/q3, §A.2w renders). So instead of deleting the aspect correction, randomise
    # it: keep the correction (quality) and destroy the precision of what it reveals (privacy),
    # the same trade `seeded_global_scale_max` already makes for SIZE. Fresh per SEQUENCE, its own
    # RNG stream, so every existing arm stays bit-identical at jitter=0.
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


_HEAD_BLOCK = (0, 1, 2, 3, 4)          # nose, eyes, ears — the joints the face rig is placed from

_ANGLE_MIRROR_ENV = os.environ.get("MIRAGE_ANGLE_MIRROR", "off")


def _mirror_signs(kp_seq, mode):
    """Per-EDGE +/-1 that makes a paired-limb rotation ANTI-SYMMETRIC instead of common-mode.

    WHY (2026-08-07, owner report: "the feet are towards left/centre of screen while the hips area
    is more towards right/away from centre, which is not natural").

    `angle_off` is keyed by GROUP and both members of a pair share the group name — `(11,13,
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
      "mirror"   negate the rotation on the RIGHT member of every pair. Removes the lean — measured
                 A_corridor common-mode -13.04 deg -> -0.92 deg off the real person — but the drawn
                 angle is random-signed, so half the time the pair converges instead of opening and
                 🔴 THE KNEES CROSS: measured 57.9 % of frames crossed, knee separation +0.892 ->
                 -0.236 hip-widths (real +1.109). That trades one unnatural artifact for a worse one.
      "outward"  anti-symmetric AND outward-only: `|theta|` with the direction forced AWAY from the
                 body midline, so the stance can only WIDEN, never cross. This is the same
                 outward-only discipline the mask ops already follow (displace / radiallp / ksame
                 are all outward-only, for the §2 superset guarantee).

    The outward DIRECTION is derived per clip from the subject's own facing — `sign(mean(x_Lsh -
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
    """Translate the emitted HEAD BLOCK so the emitted nose sits on the REAL nose.

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
    emitted nose on the real nose hands back the real nose TRAJECTORY, and — because these arms
    already put the shoulders within ~3 px of the real ones — very nearly the real NECK VECTOR,
    which is an anthropometric identity channel. §B.11-gate measured the legacy full
    re-registration (uniform scale + rigid head anchor + head template) at TM1 NM 8.97 %, i.e.
    +2.99 pp over the anonymiser alone. This knob is a strict SUBSET of that change, so its cost
    is bounded above by that figure and is not known until measured. Default "off"; every arm
    that ships with it on needs its own TM1 and TM3 row.

    Frames where the real nose was never observed are left exactly as canonicalisation produced
    them — a (0,0) "not detected" marker must never be used as an anchor.
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
                 # ---- ADDITIVE knobs (2026-07-31). Both default to 0.0 => byte-identical output
                 # and identical RandomState consumption to every number measured before today.
                 lr_asymmetry_sigma=0.0,     # TRUE independent left/right length factors (static)
                 limb_phase_amp=0.0,         # per-limb-chain cadence decorrelation amplitude
                 limb_phase_period_s=0.7,
                 # ---- ADDITIVE knobs (2026-08-02). Both defaults reproduce the previous output
                 # bit-for-bit and consume the RandomState identically (verified max|delta| = 0).
                 # They are KWARGS, not env-only, so the lab harness can sweep them through the
                 # same call the edge makes -- the env vars below merely set these defaults for
                 # build scripts. The edge and lab copies must stay in numeric parity (see the
                 # docstring); an env-only feature would have been unreachable from the lab and a
                 # TM3 number would then have described code that does not ship.
                 angle_groups=None,          # None => EVERY group rotates (shipped). A collection
                                             # => only those groups take the angle offset.
                 projection_fit=False,       # squeeze width back to the observed shoulder span
                 # ---- ADDITIVE knobs (2026-08-13). All three default to a no-op, so every arm
                 # measured before today is bit-identical and consumes the RandomState identically
                 # (each uses its OWN stream). Each attacks a DIFFERENT residual channel while
                 # leaving the three things that make the q-family render well untouched --
                 # projection_fit (upright posture), scale_from=extent (stature), root lock
                 # (registration). See the r1/r2/r3 presets for the per-arm rationale.
                 projection_fit_jitter=0.0,  # randomise the ASPECT correction => shape channel
                 limb_phase_offset_s=0.0,    # constant per-limb time SHIFT => inter-limb phase,
                                             #   with no local speed change and so no added jerk
                 pose_rate=1.0,              # uniform re-timing of the POSE => step frequency
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
                 # ---- ADDITIVE knobs (2026-08-07). Both defaults reproduce every previous
                 # number bit-for-bit and consume the RandomState identically.
                 head_anchor="off",          # "nose": put the emitted face on the REAL face
                 angle_mirror=False,         # anti-symmetric paired-limb rotation (see below)
                 # ---- ADDITIVE knob (2026-08-08). Default False reproduces every previous number
                 # bit-for-bit and consumes the RandomState identically (parity gate section F).
                 cadence_root_lock=False,    # re-time the POSE without moving the BODY
                 # ---- ADDITIVE bounded retargeting (2026-08-11). Defaults are strict no-ops.
                 seeded_global_scale_max=0.0,  # isotropic target multiplier, uniform +/- max
                 seeded_region_scale_max=0.0,  # HEAD/TRUNK/ARMS/LEGS multipliers, symmetric L/R
                 proximity_cap_frac=0.0,       # strict max joint delta / observed visible height
                 proximity_scores=None):       # raw confidence for cap scale/live-joint gating
    """kp_seq:(T,17,2) -> anonymised (T,17,2). `template`: {group: target length}.
    VERBATIM port of tier1_lab/segbench/pose_anon_v2.anonymize_v2 -- do not
    "improve" anything here: identical RandomState call ORDER is what keeps
    numeric parity with the validated lab implementation.

    The two 2026-07-31 knobs draw from a SEPARATE RandomState (`_rng(seed*7919+13)`), never
    from `r`. That is deliberate: turning them on must not shift `const_ang`/`len_factors`
    downstream, otherwise an A/B of the new knob would also be an A/B of a different angle
    draw and nothing could be attributed."""
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
        # ⚠️ MISNOMER, found 2026-07-26 (behaviour deliberately NOT changed — every published
        # privacy number was measured on this code): the docstring says "independent L/R length
        # break", but there is ONE factor per group and the L and R edges of a pair share the
        # group name (e.g. ("MS",5,"clav") and ("MS",6,"clav") both read len_factors["clav"]),
        # so left and right are always scaled IDENTICALLY. What this knob actually does is add
        # per-clip length jitter to the five paired groups only.
        for g in _LR_GROUPS:
            len_factors[g] *= float(1 + r.randn() * asymmetry_sigma)

    # (1a) BOUNDED SEEDED STANDARD-SKELETON SCALE. Separate RNG stream, static for the clip.
    len_factors = _bounded_seeded_scales(len_factors, seed, seeded_global_scale_max,
                                         seeded_region_scale_max)

    # (1b) TRUE left/right length break (2026-07-31). Per-EDGE, so the left and right member of
    # a pair get INDEPENDENT factors -- which `asymmetry_sigma` above structurally cannot do
    # (one factor per group name, shared by both sides). Drawn ONCE per clip => completely
    # static => it adds no temporal jitter at all, which is the whole point: the generator's
    # invented-hands defect is driven by temporal jitter (§B.34/§B.37), and a static shape
    # perturbation is invisible to it. Separate RNG stream so `r` is untouched.
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
    # TORSO-QUIET option (2026-08-02, MIRAGE_TORSO_QUIET=1; default OFF = unchanged).
    #
    # WHY. The angle knob is applied UNIFORMLY to every group, so it perturbs the TRUNK
    # (spine/clav/neck) exactly as hard as the LIMBS. Measured consequence on the emitted
    # skeleton vs the real person, mask held byte-identical:
    #     spine tilt   3.4x (corridor) / 7.0x (atrium)
    #     shoulder roll 2.3x          / 4.4x
    # which is the user's "weird posture / shoulder asymmetry too strong" report. The clav
    # offset rotates MS->5 and MS->6 by the SAME angle, tilting the whole shoulder line and
    # holding it tilted for the clip.
    #
    # But the gait identity an adversary reads is LIMB dynamics -- arm swing and leg cadence --
    # not trunk lean. The two are only coupled because one scalar drives all groups. Zeroing the
    # angle on the trunk groups alone should recover posture WITHOUT giving up the limb
    # perturbation that §A.2c actually priced.
    #
    # WHICH groups is not a guess -- it was measured by group-wise ablation (2026-08-01, both
    # pinned clips, 8 seeds, identical RNG consumption so it is single-variable):
    #   spine -> ALL of the trunk tilt (4.78x/7.83x real -> 1.05x/1.02x when exempted alone)
    #   clav  -> ALL of the shoulder roll (2.09x/6.80x -> 1.02x/1.00x) and most of the asymmetry
    #   hip   -> the remainder of the asymmetry
    #   neck+face -> the head-lean (atrium 1.62 deg real -> 14.13 deg shipped = 8.7x; the
    #                "weird posture"); exempting them returns it to 2.58 deg
    # The four LIMB groups cause none of it. Exempting the five axial groups removes 100% of the
    # measured posture damage while RETAINING 69-75% of the limb-joint perturbation -- whereas
    # descending the angle ladder to 4/3 keeps only 29-30% of it and still leaves tilt at
    # 1.6-2.4x with head-lean untouched. That is why the ladder is the wrong axis.
    #
    # Expressed as an ALLOW-list of the groups that keep the perturbation, not a deny-list of the
    # ones that lose it: a bone group added to EDGES later must not silently start rotating.
    #
    # 🔴 PREDICTED, NOT MEASURED on privacy. Requires a fresh TM3 adaptive run before any privacy
    # claim -- §A.2c's ladder does not describe this configuration, and borrowing its numbers would
    # be the §B.21 cross-config error. Lowering angles is also NOT monotone in quality: at 0/0 the
    # atrium torso moves LESS than the real person (tilt 0.45x real), i.e. it over-corrects into
    # the stiffness the change is meant to remove.
    # The allow-list is overridable so the group-wise ablation can be run through THIS code path
    # rather than a monkey-patched copy of it -- an ablation that measures a reimplementation is
    # evidence about the reimplementation. Unknown names are rejected loudly rather than silently
    # ignored: a typo here would silently produce the shipped arm and be scored as the new one,
    # which is exactly how the MIRAGE_ANON_ANGLE_CONST_DEG typo nearly shipped duplicate arms.
    _rot = _angle_group_mask(angle_groups)
    # Draw for EVERY group and zero afterwards, never skip the draw. Skipping would shift the RNG
    # stream and change the LIMB angles too, so a quiet-vs-shipped comparison would no longer be
    # single-variable -- and with the flag off the stream must stay bit-identical to HEAD, which is
    # what keeps §A.2c's 14.13 +/- 3.07 quotable for the shipped path.
    const_ang = {g: (r.randn() * ac) * (1.0 if _rot[g] else 0.0) for g in _GROUPS}
    drift_ang = {g: (_lowfreq_noise(T, max(1.0, lowfreq_period_s * fps / 3.0), ad,
                                    seed * 17 + gi * 101 + 3) if ad > 0 else np.zeros(T))
                    * (1.0 if _rot[g] else 0.0)
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
        if RESTORE_TORSO_ROLL:
            rt = _restore_torso_roll(rt, kp_seq[t])
        out[t] = rt + nse[:, :, t]

    # (2b) cadence / rhythm warp (temporal)
    #
    # 🔴 CADENCE_ROOT_LOCK (2026-08-08, default OFF = every previous number unchanged).
    # `_cadence_warp` is a time RE-SAMPLE, and `out` at this point holds ABSOLUTE image
    # coordinates including the subject's walking trajectory. So frame t is filled with the pose
    # the person had at t' — AND with the position they were standing in at t'. The whole figure
    # is dragged along its own path by (t' - t) x on-screen speed, which is a defect that grows
    # with how fast the subject crosses the frame and is invisible on a slow clip.
    #
    # MEASURED, run3 c2 (subject crosses at 28.1 px per emitted frame at EMIT_FPS=10), shipped
    # `g18` knobs, emitted mid-hip vs the REAL mid-hip, n=140 frames, seed pinned:
    #     cadence_amp 0.9 (shipped)      123.7 px mean   (= 4.4 frames of walking, 19.8 % of the
    #                                                     subject's own skeleton height)
    #     cadence_amp 0.6 (L4 default)    91.5 px
    #     cadence_amp 0.0                  0.1 px   <- the ENTIRE displacement is this one knob
    #     limb_phase_amp 0.0             123.7 px   <- and none of it is the per-limb warp, which
    #                                                  already re-attaches each chain to its own
    #                                                  un-warped anchor
    # It is the mechanism behind the owner's *"[the sticks] are not completely on the person"*,
    # and it is independent of the scale defect: it is present identically on the shipped and the
    # 2026-08-07-approved arms.
    #
    # The lock warps the ROOT-RELATIVE pose and re-attaches it to the TRUE per-frame mid-hip, so
    # the cadence/rhythm perturbation — the thing the knob is there to buy — is untouched, while
    # the global trajectory stops being re-sampled. Bone lengths are unaffected (a rigid
    # translation), the RandomState is consumed identically, and `center_root` below would remove
    # the trajectory entirely anyway.
    # 🔴 PRICE NOT MEASURED. `_cadence_warp`'s effect on absolute position is part of what every
    # TM1/TM2/TM3 row scored, so this knob needs its own adaptive row before any privacy number
    # may be quoted with it. See `_e2e/run3_20260807/FIX3_STICKS.md`.
    if cadence_root_lock is False:
        cadence_root_lock = os.environ.get("MIRAGE_CADENCE_ROOT_LOCK", "0") \
            not in ("0", "", "false", "False", "off")
    if cadence_amp > 0:
        if cadence_root_lock:
            _root = 0.5 * (out[:, 11:12, :] + out[:, 12:13, :])          # (T,1,2) TRUE trajectory
            out = _cadence_warp(out - _root, cadence_amp, cadence_period_s, fps,
                                seed * 991 + 1) + _root
        else:
            out = _cadence_warp(out, cadence_amp, cadence_period_s, fps, seed * 991 + 1)
    # (2c) per-limb cadence decorrelation (2026-07-31) -- AFTER the global warp, so it perturbs
    # the inter-limb PHASE relationships the global warp leaves intact. No-op at amp 0.
    if limb_phase_amp > 0:
        out = _limb_phase_warp(out, limb_phase_amp, limb_phase_period_s, fps, seed)
    # (2c-ii) CONSTANT per-limb phase SHIFT (2026-08-13). Same channel as (2c) -- inter-limb phase
    # -- but by shifting rather than re-sampling, so it costs no smoothness. No-op at 0.
    if limb_phase_offset_s > 0:
        out = _limb_phase_offset(out, limb_phase_offset_s, fps, seed)
    # (2c-iii) UNIFORM pose re-timing (2026-08-13): moves the emitted STEP FREQUENCY off the
    # subject's own. Root-relative when the root is locked, so the body keeps the real path and the
    # real walking speed and only the limb cycling changes. No-op at rate 1.0.
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
    if projection_fit or \
            os.environ.get("MIRAGE_POSE_SCALE_FROM", "").strip().lower() == "projected":
        out = _apply_projection_fit(out, kp_seq, jitter=projection_fit_jitter, seed=seed)
    # (2e) HEAD ANCHOR (2026-08-07) -- put the emitted FACE where the real face is.
    out = _apply_head_anchor(out, kp_seq, head_anchor)
    # (2f) STRICT PROXIMITY ENVELOPE (2026-08-11). Applied after every spatial/temporal knob so
    # the stated bound covers the actual emitted pose, not an intermediate skeleton.
    out = _apply_proximity_envelope(out, kp_seq, proximity_cap_frac,
                                    scores=proximity_scores)

    # (3) optional: remove the global walking trajectory (per-frame root-centering)
    if center_root:
        mh = 0.5 * (out[:, 11:12, :] + out[:, 12:13, :])   # (T,1,2)
        out = out - mh
    return out


# ------------------------------------------------- baked population template
# PROVENANCE: computed 2026-07-22 by the lab's own population_template()
# (tier1_lab/segbench/pose_anon_v2.py) over the CASIA-B GaitGraph TRAIN pool
# (tier1_lab/reid_eval/data/casia-b_pose_train_valid.csv, subject ids 1..74,
# min_frames=61), sampled to 500 tracklets with RandomState(0) -- exactly
# reproducing tier1_lab/reid_eval/anon_adapter.build_template(sample_n=500,
# seed=0). Each per-group population-median bone length (px) was divided by
# the population-median body scale (_body_scale: mid-shoulder..mid-hip torso
# length, median-per-clip then median-over-clips) to make it unit-free.
# These are POPULATION-level constants (aggregate over 74 identities -- not
# per-clip, not per-identity); the per-clip quantity (body scale) is
# auto-detected at runtime by template_for_clip().
# Derivation stats: 7966 train tracklets, 500 sampled, pop body scale
# 37.45170331303146 px; "spine" == 1.0 exactly because the spine bone
# (MH->MS) IS the body-scale segment -- a built-in consistency check.
_TEMPLATE_RATIOS = {
    "clav":  0.237332,
    "face":  0.087087,
    "farm":  0.434120,
    "hip":   0.150783,
    "neck":  0.430378,
    "shin":  0.688716,
    "spine": 1.000000,
    "thigh": 0.725496,
    "uarm":  0.534984,
}


# ---------------------------------------- STANDARD ANATOMICAL SKELETON (2026-07-26)
# The baked _TEMPLATE_RATIOS above are expressed over the SPINE (torso), and they were derived
# from CASIA-B -- a distant walking camera that foreshortens the shoulder span. Re-expressed
# over shoulder width they are systematically 1.4-1.5x too long:
#     group   template x shoulder   human anthropometry
#     neck            0.905                0.60         <- 1.51x
#     uarm            1.126                0.795        <- 1.42x
#     farm            0.914                0.624        <- 1.46x
#     thigh           1.527                1.047        <- 1.46x
#     shin            1.450                1.051        <- 1.38x
#
# And `template_for_clip` scales them by _body_scale = the shoulder->mid-hip TORSO, which on a
# bust/3-4 framing is measured from hips that are barely there. Binarised confidence, a10 N=100:
#     nose/shoulders/elbows 1.000 | wrists 0.000-0.160 | hips 0.840/0.860 (18 %/51 % off-frame)
#     knees 0.000 | ankles 0.000 (23 %/82 % off-frame)
# Result on the emitted skeleton (shoulder span = 1.00): neck 1.027, uarm 1.17/1.28,
# farm 0.62/0.21, thigh 1.09/1.64, shin 3.85/3.30 -- anatomically impossible, and the reason
# the head/neck read wrong. The user's report: "why is neck same length as of arms and legs".
#
# FIX: collapse to a fixed STANDARD skeleton whose proportions are real anthropometry, scaled by
# SHOULDER WIDTH -- the only landmark pair measured to be reliable (confidence 1.000, never
# off-frame). Stature-normalised source: biacromial 0.234 H, uarm 0.186 H, farm 0.146 H,
# thigh 0.245 H, shin 0.246 H, shoulder->hip 0.288 H, nose->mid-shoulder 0.140 H.
#
# PRIVACY: strictly MORE anonymous than before. Every subject now collapses onto ONE identical
# proportion set instead of ratios scaled by their own torso measurement. Only overall SIZE
# survives, and size is already disclosed to the recipient by the mask.
_TEMPLATE_KIND_USED = []          # provenance: which collapse target each slot used
_STATURE_SCALE_USED = []          # provenance: the per-clip auto size scale actually applied
_ANATOMICAL_OVER_SHOULDER = {
    "clav":  0.500,   # mid-shoulder -> each shoulder (half the span, by definition)
    "neck":  0.600,   # mid-shoulder -> nose
    "uarm":  0.795,
    "farm":  0.624,
    "spine": 1.231,   # mid-hip -> mid-shoulder
    "hip":   0.360,   # mid-hip -> each hip
    "thigh": 1.047,
    "shin":  1.051,
    "face":  0.130,
}


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


def _observed_spine_px(kp_seq, scores=None, conf_thresh=0.5):
    """Median mid-hip -> mid-shoulder length over frames where all four trunk joints are seen."""
    vals = []
    for t in range(kp_seq.shape[0]):
        k = kp_seq[t]
        if scores is not None:
            sc = scores[t]
            if min(float(sc[5]), float(sc[6]), float(sc[11]), float(sc[12])) < conf_thresh:
                continue
        smid = 0.5 * (k[5] + k[6])
        hmid = 0.5 * (k[11] + k[12])
        d = float(np.linalg.norm(smid - hmid))
        if d > 1.0:
            vals.append(d)
    return float(np.median(vals)) if vals else 0.0


def _stature_scale(kp_seq, scores=None, template=None, conf_thresh=0.5, min_joints=6):
    """Per-clip uniform factor k that makes the canonical skeleton's STATURE match this person's.

    ADDITIVE (2026-08-08), reached only by `MIRAGE_POSE_SCALE_FROM=extent`; every other source is
    bit-identical to before. Default is still "shoulder".

    WHY IT EXISTS. `shoulder` and `spine` both set the emitted size from ONE observed bone and then
    trust a frontal-anthropometry ratio to carry it to every other bone. Both are therefore exact
    on the bone they measure and wrong by the projection error everywhere else — which is why
    `shoulder` emits 0.49x height on a side-on subject and 1.89x on a bust crop (see the docstring
    below), and why `spine` inherits the mirror-image failure. Neither derivation looks at the
    quantity the owner is actually complaining about ("the sticks look like a small person"), which
    is the figure's overall VERTICAL EXTENT against the person's.

    So measure that quantity directly and solve for it. Rebuild the canonical skeleton at a base
    template with the angles at zero (the collapse and nothing else), then take the least-squares
    vertical scale between the rebuilt joint offsets and the observed ones, about each frame's own
    mid-hip, over the joints that frame actually observed:

        k_t = sum_j (o_yj * r_yj) / sum_j (r_yj ^ 2)          k = median_t k_t

    `k` is exact rather than iterative because the rebuild is LINEAR in the template: every joint
    is mid-hip + offset, and scaling every template length by k scales every offset by k. The X
    axis is deliberately NOT fitted here — `projection_fit` already sets the width to the observed
    shoulder span, and fitting both axes with one scalar is the very over-constraint that makes
    `shoulder` and `spine` each fail on one clip.

    GENERALISABLE BY CONSTRUCTION: both inputs are measured on this clip, per frame, and the
    median over frames is the only aggregation. There is no baked constant and no per-clip tuning.

    Returns None when the clip never shows enough joints to fit, so the caller falls back to the
    shipped path rather than guessing.

    PRIVACY: this changes SIZE only. The collapse target is still the one shared proportion set —
    every subject still lands on ONE identical skeleton shape — so the k-same collapse is untouched.
    But size that tracks the subject's own stature is a size channel, and 🔴 no TM row on record
    prices it: the lab harness collapses every tracklet onto ONE population template
    (`anon_adapter.build_template`), so no TM1/TM2/TM3 number has ever exercised a per-clip scale
    source at all. See `_e2e/run3_20260807/FIX3_STICKS.md` for the arm that must be run.
    """
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
    """Standard-skeleton collapse target for THIS clip.

    SCALE SOURCE (`src`, else MIRAGE_POSE_SCALE_FROM, default "shoulder" = shipped behaviour):

      "shoulder"  scale = observed biacromial width. The original choice, because shoulders are
                  the only landmark pair measured at confidence 1.000 and never off-frame.
      "spine"     scale = observed torso / 1.231, i.e. the emitted spine MATCHES the subject's
                  own observed torso instead of being re-derived from shoulder width.
      "extent"    scale = the least-squares VERTICAL fit of the whole canonical skeleton to this
                  person's observed joints (see `_stature_scale`). Added 2026-08-08, default OFF.

    🔴 WHY "spine" EXISTS (2026-08-02). The shoulder scaling forces
    spine = 1.231 x shoulder, a ratio taken from FRONTAL anthropometry (biacromial 0.234 H /
    shoulder-to-hip 0.288 H). Shoulder width is only a faithful scale when the subject is frontal
    AND fully in frame. Measured on our two cloud-A/B clips:
        A_corridor  real torso 217.9 px / shoulder 89.5 px  = ratio 2.435  -> emitted 0.49x height
        B_atrium    real torso 167.9 px / shoulder 254.8 px = ratio 0.659  -> emitted 1.89x height
    The corridor subject walks SIDE-ON, so the shoulder span projects short and the ratio inflates;
    the atrium subject is BUST-CROPPED, so the visible torso is short and it collapses. The emitted
    figure came out at half height on one clip and double on the other, which is what drove the
    "person is too short" / distorted-posture reports.

    PRIVACY IS UNCHANGED by this option. The collapse target is the same shared proportion set --
    every subject still lands on ONE identical skeleton shape. Only the overall SIZE differs, and
    the header above already states that size survives by design and is disclosed by the mask
    anyway. What changes is WHICH observed quantity sets that size.
    """
    src = (src if src is not None
           else os.environ.get("MIRAGE_POSE_SCALE_FROM", "shoulder")).strip().lower()
    if src == "extent":
        sh0 = _observed_shoulder_px(kp_seq, scores)
        if sh0 > 1.0:
            base = {g: _ANATOMICAL_OVER_SHOULDER[g] * sh0 for g in _GROUPS}
            k = _stature_scale(kp_seq, scores, base)
            if k is not None:
                return {g: v * k for g, v in base.items()}, "anatomical_extent"
        # never enough observed geometry to fit -> fall through to the shipped path, never guess
    if src in ("spine", "projected"):
        # "projected" takes the same spine-derived HEIGHT and additionally squeezes the width back
        # to the observed shoulder span in anonymize_v2 step (2d) -- see the rationale there.
        sp = _observed_spine_px(kp_seq, scores)
        if sp > 1.0:
            sh_equiv = sp / _ANATOMICAL_OVER_SHOULDER["spine"]
            return ({g: _ANATOMICAL_OVER_SHOULDER[g] * sh_equiv for g in _GROUPS},
                    "anatomical_spine" if src == "spine" else "anatomical_projected")
        # trunk never confidently seen -> fall through to the shipped path rather than guess
    sh = _observed_shoulder_px(kp_seq, scores)
    if sh <= 1.0:
        return template_for_clip(kp_seq), "legacy_torso_fallback"
    return {g: _ANATOMICAL_OVER_SHOULDER[g] * sh for g in _GROUPS}, "anatomical_shoulder"


def template_for_clip(kp_seq):
    """Absolute collapse template for THIS clip: baked population RATIOS scaled
    by the subject's own auto-computed median torso length. kp_seq:(T,17,2).
    Keeps the anonymised skeleton at the subject's on-frame size/position
    (the composite needs it) while its PROPORTIONS collapse to the shared
    population skeleton (the identity channel)."""
    T = kp_seq.shape[0]
    clip_scale = float(np.median([_body_scale(kp_seq[t]) for t in range(T)]))
    return {g: _TEMPLATE_RATIOS[g] * clip_scale for g in _GROUPS}


# ------------------------------------------------------------ deployment glue
# Knob ladder from the lab Pareto sweep (tier1_lab/reid_eval/sweep_pareto.py
# LADDER): the four dynamic-channel knobs per level; do_canon is always True.
LEVELS = {
    "L0": dict(cadence_amp=0.00, angle_const_deg=0, angle_drift_deg=0, asymmetry_sigma=0.00),
    "L2": dict(cadence_amp=0.25, angle_const_deg=8, angle_drift_deg=6, asymmetry_sigma=0.05),
    # 🔴 SHIPPED ANGLES ARE 14/10 (user decision, 2026-07-27) — was 20/15.
    # This is the single knob that decides whether the generator invents hands, and it was the
    # source of a long-running divergence: every corrected render (fixv_a10/a15/a30, the whole c4
    # sweep) was produced with 0/0 forced by env var while this dict still said 20/15, so the
    # committed source and the artifacts disagreed and no bundle could say which built it (§A.2d).
    # 14/10 is now the DEFAULT so committed == tested == shipped.
    # WHY 14/10 and not 0/0 or 20/15 — all three measured, see §A.2c / §B.44:
    #   20/15  gait top-1 10.34 %  but invents hands in 29.73 % of frames  -> not shippable
    #   14/10  gait top-1 14.13 %  and 0.00 % invented hands               -> SHIPPED
    #    0/0   gait top-1 28.53 %  and 0.00 % invented hands               -> 14.4 pp worse for
    #                                                                         no hands benefit,
    #                                                                         and within 1.3 sd of
    #                                                                         the anonymiser-off floor
    "L4": dict(cadence_amp=0.60, angle_const_deg=14, angle_drift_deg=10, asymmetry_sigma=0.12,
               lowfreq_amp_frac=0.0),
}
# lowfreq_amp_frac=0.0 on L4 (2026-07-26). MEASURED both sides before changing it:
#   QUALITY  (POSE_KNOB_ATTRIBUTION_20260726.json, 5 seeds/arm, v3 dynamics, N=100)
#            canon only      boneCV 0.0474  jerk 0.50   (= the real input, 0.0475 / 0.50)
#            +lowfreq alone  boneCV 0.1649  jerk 2.17   <- essentially the WHOLE L4 cost
#            L4 full         boneCV 0.1688  jerk 3.50
#   PRIVACY  (REID_REREGISTER_GATE.json, CASIA-B + GaitGraph2, 50 subjects, chance 2.0 %)
#            L4              TM1 NM rank-1 5.99 %
#            L4 lowfreq=0    TM1 NM rank-1 6.29 %       <- +0.30 pp, inside run-to-run noise
# So the limb "breathing" that made the emitted skeleton unusable as conditioning was bought
# for almost no privacy at all. lowfreq is a "naturalness" noise, not one of the four Pareto
# knobs, and it never appeared in the §A.2 ladder. Set MIRAGE_LOWFREQ_AMP to override.
try:
    _lf = os.environ.get("MIRAGE_LOWFREQ_AMP")
    if _lf is not None:
        LEVELS["L4"]["lowfreq_amp_frac"] = float(_lf)
except Exception:
    pass

# ANGLE overrides (added 2026-07-26). MEASURED motivation, not a guess: with lowfreq already
# zeroed, the angle knobs are the ENTIRE remaining temporal-noise source.
#   POSE_KNOB_ATTRIBUTION_20260726.json, 5 seeds/arm, N=100:
#     input_real 0.50 jerk · canon only 0.50 (free) · canon+cadence 0.45 (free)
#     canon+angles 2.41  <- 4.8x the real jerk, and the dominant term
#   Independently, pose.json vs MediaPipe on the aligned real clip (§B.35) measured the emitted
#   SHOULDERS -- the only genuinely visible joints -- travelling 5.6-9.4x the real per-frame
#   distance, and §B.34 found generated-motion excess predicts the invented-hand rate at
#   Spearman +0.886 while pose_strength manages only +0.371.
# 🔴 THESE ARE PARETO PRIVACY KNOBS, unlike lowfreq. Two of the four in the §A.2 ladder are
# angle_const/angle_drift, so lowering them HAS a re-ID cost that lowfreq did not (+0.30 pp).
# ✅ THE ANGLE-ONLY MEASUREMENT THIS COMMENT USED TO DEMAND NOW EXISTS — it is what set the 14/10
# default above. Both threat models, angles swept with every other knob held at the deployed point:
#   FROZEN adversary  (TM1, §B.38, sweep_angle_only.py, 50 subjects)
#     20/15 5.9 %  ·  14/10 7.7 %  ·  8/6 9.8 %  ·  0/0 10.3 %
#   ADAPTIVE adversary (TM3 retrained from scratch, §A.2c, 5 configs x 3 seeds x 100 ep)
#     20/15 10.34 +/- 0.66  ·  14/10 14.13 +/- 3.07  ·  8/6 20.63 +/- 5.15  ·  0/0 28.53 +/- 4.90
#     floor (no dynamic anonymisation) 34.94 +/- 13.74
# The frozen number UNDERSTATES the cost of lowering the knob by 4.1x (it prices 20/15 -> 0/0 at
# +4.41 pp where the adaptive adversary measures +18.19 pp), which is exactly why this gate existed.
# Overriding these for a quality experiment is still fine; changing the DEFAULT again needs a fresh
# adaptive measurement, not a frozen one.
for _env, _key in (("MIRAGE_ANGLE_CONST", "angle_const_deg"),
                   ("MIRAGE_ANGLE_DRIFT", "angle_drift_deg")):
    try:
        _v = os.environ.get(_env)
        if _v is not None:
            LEVELS["L4"][_key] = float(_v)
    except Exception:
        pass

# ---------------------------------------------------------------- 2026-07-31 knob overrides
# The gait-channel strengthening sweep (GAIT-SWEEP) needs cadence / asymmetry / the two NEW
# knobs settable without a source edit -- the angle knobs silently diverging from what shipped
# (§A.2d) is exactly what happens when an experiment has to patch the file. Every default below
# is UNCHANGED: L4 keeps cadence 0.60 / asymmetry 0.12, and the two new knobs stay 0.0 (off),
# so `LEVELS["L4"]` semantics are identical to yesterday's until a value is flipped here.
#   lr_asymmetry_sigma  -- TRUE independent L/R length break; STATIC per clip (zero added jitter)
#   limb_phase_amp      -- per-limb cadence decorrelation; a time re-sample, same family as
#                          cadence_amp, which measured jerk 0.45 vs the real input's 0.50
for _env, _key in (("MIRAGE_CADENCE_AMP", "cadence_amp"),
                   ("MIRAGE_CADENCE_PERIOD_S", "cadence_period_s"),
                   ("MIRAGE_ASYM_SIGMA", "asymmetry_sigma"),
                   ("MIRAGE_LR_ASYM_SIGMA", "lr_asymmetry_sigma"),
                   ("MIRAGE_LIMB_PHASE_AMP", "limb_phase_amp"),
                   ("MIRAGE_LIMB_PHASE_PERIOD_S", "limb_phase_period_s")):
    try:
        _v = os.environ.get(_env)
        if _v is not None:
            LEVELS["L4"][_key] = float(_v)
    except Exception:
        pass


def test_fixed_seed():
    """TEST-ONLY seed pin, or None. Read from `MIRAGE_TEST_FIXED_SEED`.

    🔴 THIS EXISTS FOR CONTROLLED A/B EXPERIMENTS ONLY AND MUST NEVER BE SET IN A SHIPPED RUN.
    A stable seed is precisely the "linkable pseudo-identity" failure `new_clip_seed` was written
    to avoid: every clip would carry the same learnable perturbation, so an adversary could invert
    it once and reuse it forever.

    WHY IT IS NEEDED AT ALL (2026-08-01, ledger §B.51). Both the gait seed here and the mask seed
    in `mirage_tier1.py` are drawn from `secrets` per run. That is right for deployment, but it
    means two Tier-1 invocations of the SAME clip differ in the perturbation as well as in
    whatever you meant to vary. The A20-vs-shipped cloud A/B was confounded exactly this way --
    the arms differed by 104-122 px of gait on top of the intended mask change -- so the
    invented-person result could not be attributed to the mask. Pinning both seeds is the only way
    to isolate one variable.

    SAFETY. Anything built with this set records `"test_fixed_seed": <value>` in the artifact's
    own `anon` block (see `anonymize_pose_log`), so a pinned-seed artifact is self-identifying and
    cannot later be mistaken for a shipped one -- the §A.2d divergence, where committed source and
    rendered artifact silently disagreed, is exactly what that guards against.
    """
    v = os.environ.get("MIRAGE_TEST_FIXED_SEED")
    if v is None or str(v).strip() == "":
        return None
    try:
        return int(v) & 0x7FFFFFFF
    except ValueError:
        return None


def new_clip_seed():
    """PER-SEQUENCE seed: fresh OS randomness, NEVER derived from identity or
    content (a stable seed would re-create a linkable pseudo-identity). The
    seed is used in-process only and must never be emitted/logged."""
    pinned = test_fixed_seed()
    if pinned is not None:
        sys.stderr.write(
            "🔴 MIRAGE_TEST_FIXED_SEED IS SET — gait perturbation is DETERMINISTIC. "
            "This output is a TEST ARTIFACT and is NOT privacy-safe to ship.\n")
        return pinned
    return secrets.randbits(31)


# ---------------------------------------------------------------- GAIT PRESETS (2026-08-07)
# A NAMED config, because "four env vars set by a build script" is exactly the shape of the §A.2d
# divergence: the committed source said one thing, every rendered artifact was produced with
# another, and no bundle could say which built it. A preset is one documented value that the
# emitted `anon` block records, so an artifact is self-identifying.
#
# Selected with MIRAGE_GAIT_PRESET=<name>. The DEFAULT is SHIPPED_PRESET ("e2") since
# 2026-08-14 -- NOT the empty string. Setting MIRAGE_GAIT_PRESET="" explicitly is what now
# restores the bare LEVELS behaviour. (This line said the opposite until 2026-08-28; the
# code at gait_preset() has defaulted to SHIPPED_PRESET since e2 was adopted.)
# A preset only supplies kwargs to anonymize_v2; it does not touch LEVELS, so `level` still
# chooses the amplitude ladder and the two compose.
#
# A preset key may also carry `scale_from` (2026-08-08). It is NOT an `anonymize_v2` kwarg — it is
# popped in `anonymize_pose_log` and handed to `anatomical_template_for_clip`, because the scale
# source lives one level ABOVE `anonymize_v2` and a preset that could not express it is exactly how
# the 2026-08-07 decision came apart. See the 🔴 note on `g18` below.
GAIT_PRESETS = {
    # ---- g18: the 2026-08-07 recommendation (ledger §A.2m-2/-2c) --------------------------------
    # = shipped L4 (14/10) + the LEGS EXEMPT from the angle knob + the projection fit + the
    #   2026-08-03 time-domain boost. Measured against BOTH adversaries:
    #     adaptive TM3  11.4 ± 0.57 %   (reference g10_boost 11.10 ± 0.85 — free)
    #     frozen TM1     2.98 %, TM2sq 3.64 %  (reference 3.27 / 3.65 — free)
    #   and it FIXES the owner-reported leg lean: thigh common-mode −0.08° vs the real person's
    #   −1.67°, where the un-exempted arm emitted −12.92°, at +3.32 pp containment on A_corridor.
    #   Exempting thigh+shin costs 1.5 % of total limb perturbation because `limb_phase_amp` and
    #   `cadence_amp` — TIME re-samples — were carrying the leg channel already.
    # 🔴🔴 `g18` DOES NOT REPRODUCE THE ARM THAT WAS APPROVED. Found 2026-08-08 from the owner's
    # report *"the body sticks look like a small person and are not completely on the person"*.
    #
    # "the projection fit" names TWO DIFFERENT MECHANISMS in this codebase and the preset took the
    # smaller one:
    #   `projection_fit=True`              an `anonymize_v2` kwarg. Squeezes the emitted WIDTH back
    #                                      to the observed shoulder span. X axis only.
    #   `MIRAGE_POSE_SCALE_FROM=projected` an `anatomical_template_for_clip` source. Takes the
    #                                      emitted HEIGHT from the observed torso *and* implies the
    #                                      width squeeze (see step 2d of anonymize_v2).
    # Every arm the 2026-08-07 decision measured, screened and put in front of the owner was built
    # by `_e2e/gait_close_20260807/build_gait_arms.py` with the ENV VAR — its own comment says
    # "`MIRAGE_POSE_SCALE_FROM=projected` is the projection fit" — so every one of them emitted
    # `template_kinds: ["anatomical_projected"]`. This preset sets only the kwarg, so it emits
    # `anatomical_shoulder`, the source the docstring of `anatomical_template_for_clip` already
    # records as collapsing to 0.49x height on a side-on subject.
    #
    # MEASURED, the two decision clips, `rawnoanon` as the real reference (emitted stature / real):
    #     A_corridor (side-on)   approved g18 1.021x   ·  what `g18` emits  0.537x
    #     B_atrium   (bust crop) approved g18 0.724x   ·  what `g18` emits  1.161x
    # and mean per-joint distance from the real body roughly DOUBLES (0.41 -> 0.81 shoulder widths
    # on A_corridor), which is the owner's "only slightly different from the person, not as much as
    # you have done it now" — nearly all of the extra difference is scale error, not perturbation.
    #
    # `g18` is LEFT BIT-IDENTICAL so every artifact and number recorded before 2026-08-08 still
    # describes committed code. Use `g18p` (what was approved) or `g18e` (the 2026-08-08 candidate).
    # 🔴 height_mult=3.0 (2026-08-10, owner direction): the "shoulder"-sourced collapse target
    # this preset uses (see the note above) was measured emitting as little as 0.537x stature, and
    # the owner judged the rendered character on p03_c02/p04_c04 as too short by roughly this
    # factor. Applied as a uniform post-scale on every bone-length group (see `height_mult` in
    # `anonymize_pose_log`), so arms/neck/legs stay proportioned to the now-taller torso -- it is
    # not a vertical-only stretch. Gait knobs (angle/cadence/limb_phase) are untouched.
    # NOT RE-MEASURED for privacy or mask containment -- a taller emitted skeleton may draw outside
    # the bbox mask sized for the originally-detected person; re-check containment after the next
    # render before trusting any prior g18 privacy number for this config. Confirmed on p03_c02
    # (§ COMPARE_p03_c02.mp4, 2026-08-10): height_mult=2.1 + hip anchor tracks the real person
    # closely, but the cloud render still shows missing feet -- the bbox mask never grew to match,
    # see the fix note attached to `height_mult` above.
    # arm_mult=0.93 (2026-08-10, owner direction): arms only, -7%. Composes with height_mult (see
    # `arm_mult` in `anonymize_pose_log`) -- torso/neck/legs untouched.
    # leg_mult=0.714 (2026-08-10, owner direction: "grow the mask slightly, cap the leg growth,
    # keep proportions natural"): legs end up at height_mult*leg_mult overall, short of the body's
    # main height_mult on purpose, paired with grow_mask_to_pose.py growing the mask a bounded
    # amount to cover what the capped legs need. ONE multiplier for both thigh and shin keeps the
    # leg's own proportions (and its ratio to the rest of the scaled body) natural rather than
    # shrinking one segment more than the other.
    # height_mult=1.9 (2026-08-11, owner direction, down from 2.1): the mask-growth fix measured
    # sufficient (0/13 grown frames hit the cap) and the pose data itself has zero missing
    # ankle/foot joints (checked all 50 frames), so 2.1x's feet/leg fade in the CLOUD render is not
    # a local containment or data-completeness issue -- most likely the reference image's own
    # implied body proportions not supporting a figure asked to be 2.1x taller. Lowering the
    # requested multiplier is the lever that's actually in scope (owner: cloud graph stays
    # untouched). Re-check COMPARE + a cloud render before trusting this value either.
    # 🔴 2026-08-12 REWORK of the three size knobs, after the emitted character's FEET/lower legs
    # were dissolving in every cloud render since the knobs landed. Two independent investigations
    # plus a code read agreed on the mechanism, and it was NOT the mask (measured: the drawn
    # skeleton never once fell below the mask, 0/50 frames, 76-158 px of clearance):
    #   1. `height_mult` was applied to the TEMPLATE, where `_apply_projection_fit` promptly
    #      divided it back out of X -- a pure VERTICAL stretch, shoulder/stature 0.234 -> 0.145.
    #      FIXED: it is now an isotropic post-scale about the mid-hip (see anonymize_pose_log).
    #   2. `leg_mult=0.714` (added to "cap leg growth") crushed leg/torso to 1.14-1.26 against an
    #      anthropometric 1.705 and a canonical-template 1.96 -- stubby legs on a long torso. The
    #      cleanest single-variable evidence: same height_mult, mask NOT grown, only the mults
    #      differing, foot contrast fell 22 %. REMOVED (leg_mult=1.0).
    # Both defects make the body anatomically impossible, and the generator answers an impossible
    # body with a smooth confidence falloff toward the extremities rather than a hard clip.
    # `arm_mult=0.93` is the owner's explicit "arms -7 %" and is kept.
    # `height_mult` re-derived for the now-isotropic scale so the emitted stature stays near what
    # the owner accepted (~580 px on p03_c02); the old 1.9 was a vertical-only figure and its
    # number does not carry over.
    # height_mult=1.70 CHOSEN BY MEASUREMENT, not by eye (p03_c02, 50 f, sweep 1.40/1.55/1.70 with
    # the isotropic scale + leg_mult 1.0). It is the arm that lands on the anthropometric
    # shoulder/stature ratio exactly -- 0.234 vs the 0.234 target, where the broken vertical-stretch
    # arm sat at 0.184 -- with leg/torso 1.83 (target 1.71, canonical 1.96) and an emitted stature
    # of 567 px = 0.93x the real subject's 608 px, i.e. a believable adult body rather than the
    # 0.537x dwarfed one that started this. Ink falling outside the repaintable hole is 0.00 % once
    # the mask is grown in all four directions (_grow_mask_to_pose2.py) -- cleaner than the 1.41 %
    # of the original arm whose feet rendered correctly.
    # 2026-08-12 (owner): back to 2.1 after 1.70 rendered a figure the owner read as short and
    # hunched. NOTE THE SEMANTICS CHANGED under the isotropy fix -- the OLD 2.1 stretched Y only
    # (X was divided back out by _apply_projection_fit), so it was tall-and-narrow; this 2.1 scales
    # BOTH axes, so the figure is both taller and correspondingly wider at the same number.
    # height_anchor="ankle" (2026-08-12) -- MEASURED NECESSITY, not a preference. Scaling about the
    # mid-hip grows the figure symmetrically, so at 2.1 the feet land at y=1267 on a 1264-row canvas
    # -- off the bottom of the picture, on every frame, with the mask bottom already pinned to the
    # last row (50/50). No amount of mask growth can buy room that the canvas does not have, which
    # is why enlarging the hole alone could never restore the feet. Anchoring the ANKLE keeps the
    # character standing where the real subject stands and spends the multiplier upward into the
    # empty headroom instead: feet y 1061 (166 px of canvas to spare), head y 373. Angles are
    # untouched by this -- it is a rigid per-frame translation, not a rotation.
    # 🔴 THE LIMB MULTS ARE EMPIRICAL AND TUNED AGAINST THE RENDER, NOT AGAINST ANATOMY (2026-08-12).
    # Measured on v9, the EMITTED SKELETON already carried leg/torso 2.140 (126 % of the 1.704
    # template target) and arm/torso 0.994 (86 % of the 1.153 target), yet the rendered character
    # reads differently -- the generator only partly follows the sticks (pose_strength 0.8) and
    # pulls limb proportions back toward the reference image's own body. So these are set from what
    # the OUTPUT looks like, and the skeleton will not match anthropometry on its own.
    # v11 (owner, from the v10 render): legs -10 %, neck +10-15 %, everything else as v10.
    #   leg_mult 1.00 -> 0.90   neck_mult 1.125   arm_mult 0.93 (v10 value, unchanged)
    #
    # v12 (owner, 2026-08-12): "generalized ... automatically adjusts according to any given case no
    # matter how close or far the person is". The ONLY clip-specific number in v11 was
    # `height_mult=2.1`, an absolute multiplier fitted to a subject ~629 px tall; every other knob
    # here is already dimensionless. It is replaced by `stature_ratio=1.055`, which solves the
    # multiplier per clip so the emitted figure is 1.055x THIS subject's own observed height --
    # 1.055 being exactly what v11 achieved on p03_c02 (663.8 px emitted / 629.1 px observed), so
    # v12 reproduces the approved v11 look here while adapting itself to any other framing.
    # `height_mult` is kept only as the fallback if the stature fit is ever degenerate.
    "g18": dict(angle_groups=("uarm", "farm"), projection_fit=True,
                limb_phase_amp=1.20, cadence_amp=0.90, stature_ratio=1.055, height_mult=2.1,
                arm_mult=0.93, leg_mult=0.90, neck_mult=1.125, height_anchor="ankle"),
    # ---- g18p: `g18` AS APPROVED — the preset now carries the scale source, so the arm is one
    #      documented name instead of a preset plus an env var a build script has to remember.
    #      Bit-identical to `_e2e/gait_close_20260807` `g18_boost_armangles`.
    "g18p": dict(angle_groups=("uarm", "farm"), projection_fit=True,
                 limb_phase_amp=1.20, cadence_amp=0.90, scale_from="projected"),
    # ---- g18e: 2026-08-08 candidate. Same gait knobs; the stature is FITTED to the person instead
    #      of re-derived from one bone, because `projected` is exact on the torso and inherits the
    #      projection error everywhere else (1.021x on A_corridor but 0.724x on B_atrium).
    #      🔴 Its gait knobs are `g18`'s and are priced; its SIZE channel is NOT priced — and
    #      neither is `g18`'s or `g18p`'s (see `_stature_scale`). Awaiting owner approval.
    "g18e": dict(angle_groups=("uarm", "farm"), projection_fit=True,
                 limb_phase_amp=1.20, cadence_amp=0.90, scale_from="extent"),
    # ---- g20: the full 2026-08-08 answer to the owner's report — `g18e` + `cadence_root_lock`.
    #      The two defects behind *"not completely on the person"* are INDEPENDENT and each needs
    #      its own knob: `scale_from` fixes the figure's SIZE, `cadence_root_lock` stops the time
    #      warp from dragging the figure off its own body (run3 c2: 123.7 px of mid-hip error,
    #      100 % of it `cadence_amp`). Gait knobs are `g18`'s, unchanged.
    #      🔴 NEITHER new knob is priced. Both change what the adversary sees. Awaiting owner
    #      approval AND the TM3 arm named in `_e2e/run3_20260807/FIX3_STICKS.md`.
    "g20": dict(angle_groups=("uarm", "farm"), projection_fit=True,
                limb_phase_amp=1.20, cadence_amp=0.90, scale_from="extent",
                cadence_root_lock=True),
    # ---- seedclose*: bounded privacy/conditioning trade-offs (2026-08-11) -------------------
    # Common pipeline: canonical standard skeleton -> robust observed-extent scale -> fresh
    # per-SEQUENCE bounded multipliers -> FK rebuild -> strict joint proximity envelope. Seeds are
    # never identity-derived/reused. These are NEW arms: historical g18/fix7 percentages do not
    # apply; each needs its own adaptive TM3 measurement before a privacy claim.
    #
    # S1: maximum pose fidelity. Static canonical collapse plus only a +/-3 % global scale draw.
    "seedclose1": dict(scale_from="extent", projection_fit=False, cadence_root_lock=True,
                       cadence_amp=0.0, limb_phase_amp=0.0, angle_const_deg=0.0,
                       angle_drift_deg=0.0, asymmetry_sigma=0.0, lowfreq_amp_frac=0.0,
                       seeded_global_scale_max=0.03, seeded_region_scale_max=0.0,
                       proximity_cap_frac=0.04),
    # S2: static/dynamic middle point. Symmetric regional shape draws plus bounded limb motion.
    "seedclose2": dict(scale_from="extent", projection_fit=False, cadence_root_lock=True,
                       cadence_amp=0.25, limb_phase_amp=0.30, angle_const_deg=8.0,
                       angle_drift_deg=6.0, angle_groups=("uarm", "farm", "thigh", "shin"),
                       asymmetry_sigma=0.0, lowfreq_amp_frac=0.0,
                       seeded_global_scale_max=0.025, seeded_region_scale_max=0.04,
                       proximity_cap_frac=0.06),
    # S3: strongest of the close-pose family. Uses the validated g18 dynamic amplitudes, but a
    # root lock and an 8 %-of-visible-height envelope prevent the large off-person excursions.
    "seedclose3": dict(scale_from="extent", projection_fit=False, cadence_root_lock=True,
                       cadence_amp=0.90, limb_phase_amp=1.20, angle_const_deg=14.0,
                       angle_drift_deg=10.0, angle_groups=("uarm", "farm"),
                       asymmetry_sigma=0.0, lowfreq_amp_frac=0.0,
                       seeded_global_scale_max=0.02, seeded_region_scale_max=0.06,
                       proximity_cap_frac=0.08),
    # ---- lite1: "keep the sticks as the original, change things only very lightly" ------------
    # Owner spec (2026-08-12): sticks as close to the detected pose as possible; lengths changed
    # only slightly; angles "very extremely lightly"; a slight cadence change; and a little
    # randomness on each of those.
    #
    # 🔴 THE SPEC AS LITERALLY STATED HAS ALREADY BEEN MEASURED AND FAILS. `seedclose1` is
    # essentially it -- zero angle, zero cadence, a +/-3 % scale draw, joints held near observed --
    # and it scores adaptive TM3 69.27 +/- 0.82 % against a 20.63 % bar and a 77.09 % undefended
    # ceiling (checkpoints in tier1_lab/reports/reid/). Holding the pose and sprinkling randomness
    # does not remove identity, because the identity is IN the pose.
    #
    # What makes this arm different is WHERE the pose is allowed to stay identical. The canonical
    # collapse in `_rebuild_frame` keeps each bone's OBSERVED DIRECTION and substitutes the
    # CANONICAL LENGTH, so the emitted figure still traces the subject's own posture and motion --
    # measured cost only 0.066-0.110 H of joint displacement -- while every subject is forced onto
    # ONE proportion set. With `projection_fit=False` that collapse is exact: cross-subject
    # proportion difference 0.0000 (COLLAPSE_FLOOR.json). It is removal, not obfuscation: emitted
    # proportions become a constant function of the subject, so no attacker, adaptive or not, can
    # recover them. `projection_fit=True` would silently undo this -- it rescales X per clip, which
    # makes every non-vertical bone ratio subject-specific again (97 % of the undefended difference
    # survives) -- which is why it is OFF here despite costing some width fidelity.
    #
    # So: shape identity is removed exactly and cheaply; the light angle/cadence/randomness knobs
    # below are the owner's requested perturbation on top, and are what the ADAPTIVE measurement
    # has to price. 🔴 UNPRICED until that run lands -- do not ship on this comment.
    "lite1": dict(
        projection_fit=False,          # REQUIRED for an exact shape collapse (see above)
        scale_from="extent",           # size from the subject's own visible extent: no fixed
                                       # multiplier, so it generalises to any camera distance
        cadence_root_lock=True,        # re-time the POSE without translating the BODY: keeps the
                                       # figure ON the person (root error 0.27 H -> 0.002 H)
        angle_groups=("uarm", "farm"), # legs exempt -- measured free (g18 11.37 vs g10_boost 11.10)
        angle_const_deg=3.0,           # "very extremely lightly": g18 ships 14
        angle_drift_deg=2.0,           #   ” : g18 ships 10
        cadence_amp=0.15,              # "slight cadence": g18 ships 0.90
        limb_phase_amp=0.20,           #   ” : g18 ships 1.20
        asymmetry_sigma=0.0,           # OFF deliberately: this is the +/-12 % per-run L/R draw that
                                       # moves leg/torso 1.512 -> 1.904 on seed alone, i.e. it makes
                                       # any deliberate tuning unconvergeable. Randomness is instead
                                       # taken through the two BOUNDED, symmetric draws below.
        lowfreq_amp_frac=0.0,
        seeded_global_scale_max=0.10,  # slight randomness on overall size, fresh per SEQUENCE
        seeded_region_scale_max=0.05,  # slight randomness on regional lengths, fresh per SEQUENCE
    ),
    # ---- q2: quality-first arm whose PRIVACY ARGUMENT IS UNLEARNED -----------------------------
    # Owner constraint (2026-08-12): the paper may report only MODEL-FREE metrics, so the privacy
    # claim has to rest on something provable rather than on beating a particular recogniser.
    #
    # THE ONE STRUCTURAL CLAIM AVAILABLE, and this arm is built to earn it: with
    # `projection_fit=False` the canonical collapse makes the emitted bone proportions a CONSTANT
    # function of the subject -- cross-subject |proportion difference| measured at exactly 0.0000
    # (COLLAPSE_FLOOR.json, PF off; the same collapse with PF on leaves 97 % of the undefended
    # difference intact, because a per-clip X rescale changes every non-vertical bone ratio and
    # makes shape subject-specific again). That is REMOVAL of the shape channel, not obfuscation:
    # there is nothing left for any adversary, learned or not, to recover. It costs only
    # d50 0.066-0.110 H of joint displacement, because `_rebuild_frame` keeps each bone's OBSERVED
    # DIRECTION and substitutes only its length -- the figure still traces the subject's posture.
    #
    # WHY `seeded_global_scale_max` AND NOT `seeded_region_scale_max`: the global draw scales every
    # group UNIFORMLY, so it randomises SIZE while leaving the ratios -- and therefore the exact
    # k-same collapse -- untouched. A regional draw scales groups differently and destroys it:
    # measured on `lite1` (region 0.05), cross-subject difference came back at 0.0804, WORSE than
    # the undefended 0.0744. Size randomisation and exact shape collapse are compatible; regional
    # randomisation and exact shape collapse are not.
    #
    # The dynamic knobs are g18's, unchanged, because those are the ones with a priced TM3 number
    # (11.37 +/- 0.38) and, per §A.2c, they -- not the shape channel -- are what actually protects
    # the shipped arm. `asymmetry_sigma` is OFF: it is the +/-12 % per-run L/R draw that moves
    # leg/torso 1.512 -> 1.904 on seed alone, making any deliberate tuning unconvergeable.
    #
    # 🔴 SCOPE OF THE CLAIM, to be stated in the paper exactly this way: the shape channel is
    # removed exactly and provably; the DYNAMICS and TRAJECTORY channels are only PERTURBED, and no
    # model-free metric can show a walking rhythm is unrecoverable. Do not write that this arm
    # defeats re-identification -- that would need the learned measurement the paper is excluding.
    "q2": dict(
        projection_fit=False,          # REQUIRED for the exact 0.0000 collapse (the paper claim)
        scale_from="extent",           # stature from the subject's own extent: 0.57x -> ~1.05x,
                                       # and generalises to any camera distance with no constant
        cadence_root_lock=True,        # re-time the POSE without translating the BODY:
                                       # root error 0.27 H -> 0.002 H, keeps the figure ON the person
        angle_groups=("uarm", "farm"), # legs exempt -- measured free (g18 11.37 vs g10_boost 11.10)
        angle_const_deg=14.0,          # g18's priced dynamic knobs, unchanged
        angle_drift_deg=10.0,          #   "
        cadence_amp=0.90,              #   "
        limb_phase_amp=1.20,           #   "
        asymmetry_sigma=0.0,           # OFF: unconvergeable +/-12 % per-run jitter (see above)
        lowfreq_amp_frac=0.0,
        seeded_global_scale_max=0.10,  # randomise the SIZE channel (uniform => ratios preserved)
    ),
    # ---- q2w / q3: the two open questions after q1/q2/q1w (2026-08-13) ------------------------
    # q2w = q2 + the wider size draw. q1 vs q1w measured the draw at PF=True and found the channel
    # SATURATES (m 0.10 -> 0.20 bought only 1.0 pp after the first 2.82 pp). This asks whether the
    # same is true with the shape collapse active, where the residual should be smaller.
    "q2w": dict(
        projection_fit=False, scale_from="extent", cadence_root_lock=True,
        angle_groups=("uarm", "farm"), angle_const_deg=14.0, angle_drift_deg=10.0,
        cadence_amp=0.90, limb_phase_amp=1.20,
        asymmetry_sigma=0.0, lowfreq_amp_frac=0.0,
        seeded_global_scale_max=0.20,
    ),
    # q3 = q2 with the TIME warps halved. `cadence_root_lock` moves the warp out of whole-body
    # translation and into the pose itself, which measured emitted jerk at 1.56x the real
    # skeleton's (0.037 vs 0.0237). §B.34 links motion excess to the invented-hands defect
    # (Spearman +0.886), so the arm that fixes registration may be buying a generation-quality
    # problem. This halves cadence/limb_phase to test that directly.
    # 🔴 Those two knobs are the ones carrying the gait channel, so this WILL cost privacy margin;
    # it is a quality probe, and its own TM3 is required before it could ship.
    "q3": dict(
        projection_fit=False, scale_from="extent", cadence_root_lock=True,
        angle_groups=("uarm", "farm"), angle_const_deg=14.0, angle_drift_deg=10.0,
        cadence_amp=0.45, limb_phase_amp=0.60,
        asymmetry_sigma=0.0, lowfreq_amp_frac=0.0,
        seeded_global_scale_max=0.10,
    ),
    # ---- q1s ✅ SHIPPED (owner decision, 2026-08-14, "unless we find something better") -------
    # Selected over `q1` and `r2` from a three-way render (`100826_runs/_Q1_Q1S_R2.mp4`). The
    # untrained attacker does NOT separate the three (q1s top-1 lift +2.56 / top-5 +6.77, vs r2's
    # +2.36 / +5.96 and q1's +2.84 / +5.65 -- all inside the nulls' own +/-0.3 pp wobble), so this
    # is a visual choice, which is the right basis when the measurement cannot discriminate.
    #
    # 🔴 WHAT SHIPPING THIS COMMITS TO, stated plainly because it is the one arm of the three that
    # fails the trained bar. Adaptive TM3, n=3: q1s 20.14 +/- 1.11 against a 20.63 % bar, with
    # 1/3 seeds (21.37) OVER it and not statistically below (one-sided t = 0.77, p ~ 0.26).
    # `q1` measured 18.43 +/- 1.02 (passes, 2.20 pp margin) and `q1sw` 18.78 +/- 0.45 (passes,
    # t = 7.17). `r2` has NO trained number at all. So on the LEARNED adversary the shipped arm is
    # the weakest of those measured; on the UNTRAINED attacker the paper reports it removes 92.71 %
    # of top-1 lift and 87.44 % of top-5 lift (§A.2x). Any privacy claim must be scoped to the
    # untrained adversary, and must not be phrased as resistance to re-identification generally.
    #
    # ---- (mechanism) q1 with the SUDDEN MOTION removed (owner report, 2026-08-12) -------------
    # The owner judged q1's render the only acceptable one but reported the movement as "a bit
    # sudden in between". That is measurable, and it localises to ONE knob.
    #
    # MEASURED (p03_c02, 50 f, seed pinned 777 so every row is single-variable; jerk and p95
    # acceleration are median/95th-percentile per-joint, normalised by observed visible height;
    # d50 is median per-joint displacement from the anon-OFF skeleton = how much the pose was
    # actually perturbed):
    #     config                          jerk/real   p95 acc   d50
    #     REAL (anon off)                     1.00     0.0450   0.000
    #     q1 shipped: cad 0.90 limb 1.20      1.78     0.0574   0.065
    #     cadence ONLY (limb 0)               1.74     0.0601   0.063
    #     limb ONLY (cadence 0)               1.06     0.0472   0.063
    #     cad 0.0  limb 1.8                   1.13     0.0496   0.063
    #     cad 0.0  limb 2.4                   1.14     0.0511   0.063
    #     cad 0.3  limb 1.8                   1.70     0.0523   0.063
    #     neither warp                        0.98     0.0411   0.058
    # Three things fall out. (1) `cadence_amp` causes ESSENTIALLY ALL of the excess jerk and adds
    # almost no perturbation over limb-phase alone (0.065 vs 0.063). (2) It is not a dial to turn
    # down -- 0.3 already restores 1.70x. (3) Perturbation SATURATES at 0.063, so raising
    # `limb_phase_amp` past ~1.8 buys nothing; 1.8 is chosen as a modest compensation for dropping
    # cadence, at 1.13x jerk instead of 1.78x.
    #
    # WHY THIS SHOULD ALSO BE THE BETTER PRIVACY TRADE, per `_limb_phase_warp`'s own docstring:
    # `_cadence_warp` re-times the WHOLE skeleton on ONE clock, so every inter-limb phase
    # relationship -- arm vs contralateral leg, left vs right swing -- "survives it intact", and
    # those relationships are what a gait recogniser reads. `_limb_phase_warp` is the knob that
    # actually breaks them. So cadence is the expensive, low-value half of the pair.
    #
    # 🔴 PRIVACY MUST BE RE-MEASURED. `q1`'s 18.43 +/- 1.02 % was measured WITH cadence_amp=0.90;
    # dropping it changes the arm, and §A.2k priced the two time re-samples together at -8.23 pp
    # without separating them. Do not inherit q1's number. Settling command:
    #   python tm3_retrain.py --configs qsize_m10_smooth --tags q10s --epochs 100 --seeds 0,1,2
    "q1s": dict(angle_groups=("uarm", "farm"), projection_fit=True,
                limb_phase_amp=1.80,           # the knob that breaks inter-limb phase: KEPT, raised
                cadence_amp=0.0,               # the knob that causes the suddenness: OFF
                scale_from="extent", cadence_root_lock=True,
                seeded_global_scale_max=0.10),
    # ---- d1/d2/d3 (2026-08-14): SAME GEOMETRY AS q1s, attacking only what still leaks ---------
    # The owner picked q1s on RENDER quality and asked to keep it while improving the defence. So
    # these change NOTHING q1s renders with -- same angle groups, same projection_fit, same
    # scale_from, same root lock, same size draw, same limb_phase_amp -- and add only per-sequence
    # TIME knobs. Every static proportion the generator sees is bit-identical to q1s.
    #
    # WHY ONLY TIME. Descriptor ablation of q1s (ledger §A.2x, N=5375): the dynamics block ALONE
    # scores +2.71 top-1 / +6.24 top-5 lift, i.e. it reproduces the whole full-descriptor attack
    # (+2.56 / +6.77), while shape scores -0.20 / -0.07 and inter-limb phase -1.09 / -0.20 -- both
    # AT OR BELOW chance. The collapse template and limb_phase_amp=1.80 have already taken those
    # two channels to the floor; more of either buys nothing. Dynamics is the only channel left.
    #
    # The two arms are the two INDEPENDENT directions of that block, which is why d1 and d2 are
    # separated before being combined: a uniform rate scales median and p90 speed together, so
    # ratios such as p90/median are invariant to it and need the low-pass instead.
    "d1": dict(angle_groups=("uarm", "farm"), projection_fit=True,
               limb_phase_amp=1.80, cadence_amp=0.0,
               scale_from="extent", cadence_root_lock=True,
               seeded_global_scale_max=0.10,
               pose_smooth_max_s=0.08),        # peakiness only; jerk bounded BELOW q1s's
    "d2": dict(angle_groups=("uarm", "farm"), projection_fit=True,
               limb_phase_amp=1.80, cadence_amp=0.0,
               scale_from="extent", cadence_root_lock=True,
               seeded_global_scale_max=0.10,
               pose_rate_jitter=0.18),         # tempo only; one-sided so jerk cannot rise
    "d3": dict(angle_groups=("uarm", "farm"), projection_fit=True,
               limb_phase_amp=1.80, cadence_amp=0.0,
               scale_from="extent", cadence_root_lock=True,
               seeded_global_scale_max=0.10,
               pose_smooth_max_s=0.08, pose_rate_jitter=0.18),   # both
    # ---- d4/d5/d6 (2026-08-14): push the RATE draw, the only knob that both works and transfers.
    # §A.2y: the draw carries the whole win (d2 +1.01 top-1 lift vs q1s's +2.56) while the low-pass
    # alone bought nothing (d1 +2.21). §A.2x-5: the low-pass is largely inert at the phone's 10 fps
    # because its sigma is in SECONDS; a uniform re-time is frame-rate independent. d6 is the same
    # draw WIDTH as d4 but centred at 1.0, so limbs are not only ever slowed against a real-speed
    # root -- the beat that drove r3's 1.68x jerk.
    "d4": dict(angle_groups=("uarm", "farm"), projection_fit=True,
               limb_phase_amp=1.80, cadence_amp=0.0,
               scale_from="extent", cadence_root_lock=True,
               seeded_global_scale_max=0.10, pose_rate_jitter=0.30),
    "d5": dict(angle_groups=("uarm", "farm"), projection_fit=True,
               limb_phase_amp=1.80, cadence_amp=0.0,
               scale_from="extent", cadence_root_lock=True,
               seeded_global_scale_max=0.10, pose_rate_jitter=0.45),
    "d6": dict(angle_groups=("uarm", "farm"), projection_fit=True,
               limb_phase_amp=1.80, cadence_amp=0.0,
               scale_from="extent", cadence_root_lock=True,
               seeded_global_scale_max=0.10, pose_rate_jitter=0.30,
               pose_rate_two_sided=True),
    # ---- d2s / d2c (2026-08-14): "d2 but smoothed out". §A.2y-5 measured d2 at +40 % jerk vs q1s
    # at the shipped 10 fps (n=5, t=5.03). Cause: `_pose_rate_warp` interpolates LINEARLY, so each
    # source frame is a kink and sparser frames mean bigger kinks (12 % at 25 fps, 40 % at 10 fps).
    # d2s BLURS the kink with a low-pass whose sigma is in FRAMES (the seconds form is inert at
    # 10 fps, §A.2x-5); d2c REMOVES it with C1 Catmull-Rom resampling.
    "d2s": dict(angle_groups=("uarm", "farm"), projection_fit=True,
                limb_phase_amp=1.80, cadence_amp=0.0,
                scale_from="extent", cadence_root_lock=True,
                seeded_global_scale_max=0.10, pose_rate_jitter=0.18,
                pose_smooth_max_frames=1.0),
    "d2s2": dict(angle_groups=("uarm", "farm"), projection_fit=True,
                limb_phase_amp=1.80, cadence_amp=0.0,
                scale_from="extent", cadence_root_lock=True,
                seeded_global_scale_max=0.10, pose_rate_jitter=0.18,
                pose_smooth_max_frames=2.0),
    # 🔴 "d2c" DELETED 2026-08-14 (owner: discard fully). It held the best untrained top-1 ever
    # measured (+0.70, 98.0 % of RAW's lift removed) but was NEVER RENDERED, and it cost +46 % jerk
    # (1.46 x q1s) with a top-5 no better than the arm it would have replaced. Ledger §A.3a/§A.3b-7.
    # Do not re-add: an unrendered arm is not a candidate (§B.64-2).
    "d2sc": dict(angle_groups=("uarm", "farm"), projection_fit=True,
                limb_phase_amp=1.80, cadence_amp=0.0,
                scale_from="extent", cadence_root_lock=True,
                seeded_global_scale_max=0.10, pose_rate_jitter=0.18,
                pose_rate_kind="catmull", pose_smooth_max_frames=1.0),
    # ---- e-family (2026-08-14): reach DYNAMICS without re-timing anything. Every d-arm pays
    # +40..46 % jerk because re-timed limbs beat against a root the mask pins to the real path;
    # that cost survived both a low-pass and a C1 resampler, so it is inherent to re-timing.
    # Swing amplitude scales the same statistics via the angular-rate factor instead, and keeps
    # bone lengths exact. See `_limb_swing_amp`.
    "e1": dict(angle_groups=("uarm", "farm"), projection_fit=True,
               limb_phase_amp=1.80, cadence_amp=0.0,
               scale_from="extent", cadence_root_lock=True,
               seeded_global_scale_max=0.10, limb_swing_amp=0.25),
    "e2": dict(angle_groups=("uarm", "farm"), projection_fit=True,
               limb_phase_amp=1.80, cadence_amp=0.0,
               scale_from="extent", cadence_root_lock=True,
               seeded_global_scale_max=0.10, limb_swing_amp=0.25, limb_phase_offset_s=0.35),
    "e3": dict(angle_groups=("uarm", "farm"), projection_fit=True,
               limb_phase_amp=1.80, cadence_amp=0.0,
               scale_from="extent", cadence_root_lock=True,
               seeded_global_scale_max=0.10, limb_swing_amp=0.25, limb_phase_offset_s=0.35,
               pose_rate_jitter=0.18, pose_rate_kind="catmull"),
    # swing-amplitude sweep: no amplitude on this pipeline may be assumed monotone (§A.2z).
    "e4": dict(angle_groups=("uarm", "farm"), projection_fit=True,
               limb_phase_amp=1.80, cadence_amp=0.0,
               scale_from="extent", cadence_root_lock=True,
               seeded_global_scale_max=0.10, limb_swing_amp=0.15),
    "e5": dict(angle_groups=("uarm", "farm"), projection_fit=True,
               limb_phase_amp=1.80, cadence_amp=0.0,
               scale_from="extent", cadence_root_lock=True,
               seeded_global_scale_max=0.10, limb_swing_amp=0.40),
    # f-family: the WALKING-SPEED channel, untouched by every arm on record because the root is
    # locked to the true path for containment -- but the descriptor reads speed on ABSOLUTE
    # coordinates, so the subject's own pace has been passing through intact. Hard px clamp keeps
    # the figure inside the grown mask. f3 = every channel EXCEPT the jerk-costly re-timing.
    "f1": dict(angle_groups=("uarm", "farm"), projection_fit=True,
               limb_phase_amp=1.80, cadence_amp=0.0,
               scale_from="extent", cadence_root_lock=True,
               seeded_global_scale_max=0.10, root_speed_amp=0.20),
    "f2": dict(angle_groups=("uarm", "farm"), projection_fit=True,
               limb_phase_amp=1.80, cadence_amp=0.0,
               scale_from="extent", cadence_root_lock=True,
               seeded_global_scale_max=0.10, root_speed_amp=0.20, limb_swing_amp=0.25),
    # 🔴 "f3" DELETED 2026-08-14 (owner: discard fully). It won EVERY pose-space metric -- smoothest
    # arm ever measured (0.92 x q1s, t = -3.39), best untrained top-1 of any no-cost arm (+1.96),
    # tied best top-5, containment 0.00 %, bone lengths exact -- AND LOST THE FEET on the render:
    # figure 11.1 % shorter, feet-band alpha -30 %. Ledger §B.64. Do not re-add, and do not ship
    # `root_speed_amp`: it is also worthless alone (§A.3b-6, f1 +2.42/+6.92 vs shipped +2.56/+6.77).
    # e6/e7: the swing sweep says top-1 improves with amplitude at ZERO smoothness cost, but e2
    # pairs the phase shift with 0.25. These pair it with 0.40; e7 adds the root-speed channel.
    "e6": dict(angle_groups=("uarm", "farm"), projection_fit=True,
               limb_phase_amp=1.80, cadence_amp=0.0,
               scale_from="extent", cadence_root_lock=True,
               seeded_global_scale_max=0.10, limb_swing_amp=0.40, limb_phase_offset_s=0.35),
    "e7": dict(angle_groups=("uarm", "farm"), projection_fit=True,
               limb_phase_amp=1.80, cadence_amp=0.0,
               scale_from="extent", cadence_root_lock=True,
               seeded_global_scale_max=0.10, limb_swing_amp=0.40, limb_phase_offset_s=0.35,
               root_speed_amp=0.20),
    # ---- q1sw: SMOOTH motion AND the most reliable privacy pass measured so far ---------------
    # = q1s (cadence off, limb-phase up) + the WIDER size draw, to buy back what smoothing cost.
    #
    # THE THREE-ARM STORY, all adaptive TM3, n=3 seeds, bar 20.63, control 77.09, null 2.06:
    #     arm    cadence  limb  size draw   TM3 mean +/- sd   margin   one-sided t   jerk/real
    #     q1      0.90    1.20    0.10      18.43 +/- 1.02    2.20 pp     3.73         1.78x
    #     q1s     0.0     1.80    0.10      20.14 +/- 1.11    0.49 pp     0.77  FAIL   1.13x
    #     q1sw    0.0     1.80    0.20      18.78 +/- 0.45    1.85 pp     7.17         1.13x
    # `q1s` refuted the prediction that cadence contributes little defence: dropping it cost
    # +1.71 pp and put the arm ON the bar (1/3 seeds over). The wider draw then bought back
    # -1.36 pp of that, so the two effects partially compose.
    #
    # WHY THIS IS THE BEST ARM ON THE EVIDENCE, despite a smaller nominal margin than `q1`: its
    # seed sd is less than half (0.45 vs 1.02), so the pass is roughly twice as statistically
    # confident (t 7.17 vs 3.73) and 0/3 seeds land over the bar. It also has the smooth motion the
    # owner asked for (jerk 1.13x real vs `q1`'s 1.78x).
    #
    # 🔴 THE COST, and it is a QUALITY cost the owner must judge from a render: the +/-20 % draw
    # spreads emitted stature to 0.929-1.172x (vs 1.013-1.155x at +/-10 %). `q1w`, the sudden-motion
    # arm at the same amplitude, rendered visibly less stable in size. NOT YET RENDERED at these
    # warp settings -- that render is what decides whether this ships.
    "q1sw": dict(angle_groups=("uarm", "farm"), projection_fit=True,
                 limb_phase_amp=1.80, cadence_amp=0.0,
                 scale_from="extent", cadence_root_lock=True,
                 seeded_global_scale_max=0.20),
    # ============================================================================================
    # r1 / r2 / r3 — THREE DIRECTIONS FROM q1s (2026-08-13, owner brief)
    # ============================================================================================
    # WHAT MAKES q1 / q1s RENDER BETTER THAN EVERY OTHER ARM, measured, so the new arms keep it:
    #   projection_fit=True   the aspect correction. Every PF=False arm (q2/q2w/q3) rendered
    #                         HUNCHED -- this is what holds the figure upright.
    #   scale_from="extent"   stature ~1.0-1.1x instead of `shoulder`'s 0.57x "small person".
    #   cadence_root_lock     root error 0.002 H instead of 0.27 H: the figure stays ON the person.
    #   cadence_amp=0 (q1s)   jerk 1.13x real instead of 1.78x: the "sudden" motion, fixed.
    # ALL THREE ARMS BELOW HOLD ALL FOUR FIXED and change exactly one NEW thing, so whichever the
    # owner prefers identifies a direction rather than a lucky combination.
    #
    # WHY NEW CHANNELS AT ALL (§A.2w): the q-family removes ~74 % of top-1 lift but only ~46 % of
    # top-5 lift, and NEITHER existing knob moves top-5 -- the size draw shifts top-5 lift by
    # 0.7 pp (inside seed sd) and cadence by 1.6 pp (inside `q1`'s own 2.56 sd). The residual is
    # not reachable from `seeded_global_scale_max` or `cadence_amp`, so more of either cannot help.
    # Each arm below therefore opens a channel that is currently UNTOUCHED.
    #
    # 🔴 ALL THREE ARE UNPRICED. Settling command, per arm:
    #   python tm3_retrain.py --configs <cfg> --tags <tag> --epochs 100 --seeds 0,1,2
    # then `tm3_eval_cmc.py` for top-1/top-5 with measured nulls. Bar 20.63 (canonical rank-1),
    # control 77.09, null 2.06. Reference to beat: q1s 20.14 +/- 1.11 rank-1, top-5 lift +47.52.

    # ---- r1 "SHAPE": randomise the aspect correction ------------------------------------------
    # THE BIGGEST UNTAPPED CHANNEL. `_apply_projection_fit` scales X by
    # sx = observed_shoulder / emitted_shoulder -- a direct per-clip readout of how broad THIS
    # person is -- and because it touches X only it re-injects subject-specific bone ratios,
    # measured as 97 % of the undefended between-subject shape difference surviving (§B.63b).
    # Deleting the fit removes the leak exactly (0.0000) but wrecks posture. So randomise sx per
    # sequence instead: the figure still gets AN aspect correction (upright, rendering well) but
    # the adversary cannot read the subject's true breadth from it. Same trade
    # `seeded_global_scale_max` makes for size -- and that knob is measured to work (-2.4 pp top-1).
    # PREDICTION: should move top-5 as well as top-1, because shape is a static cue that persists
    # across every frame of a tracklet, which is exactly the kind of evidence a rank-5 list is
    # built from. If it does not move top-5, the residual is dynamic, not static -- which is itself
    # the answer to where to go next.
    "r1": dict(angle_groups=("uarm", "farm"), projection_fit=True,
               limb_phase_amp=1.80, cadence_amp=0.0,
               scale_from="extent", cadence_root_lock=True,
               seeded_global_scale_max=0.10,
               projection_fit_jitter=0.15),

    # ---- r2 "PHASE" — was shipped 2026-08-13, SUPERSEDED BY q1s 2026-08-14 --------------------
    # Owner reverted to `q1s` after a three-way render (`100826_runs/_Q1_Q1S_R2.mp4`, q1 vs q1s vs
    # r2). r2 remains the best arm on the measured quality knobs (d50 0.040 H, jerk 1.10x) and on
    # untrained top-1 lift (+2.36), but the three arms sit within 0.48 pp (top-1) and 1.12 pp
    # (top-5) of each other against nulls that themselves move +/-0.3 pp, so no untrained number
    # separates them and the choice was visual. Kept intact and measurable.
    # Chosen from a four-way render (q1s baseline vs r1/r2/r3, `100826_runs/_R_ARMS_COMPARE.mp4`).
    # It also carries the best measured quality of any arm this session -- d50 0.040 H (vs q1s
    # 0.050, g18 0.181) and jerk 1.10x real (vs q1s 1.19x, g18 1.78x) -- and the best untrained
    # re-ID numbers of the four (top-1 lift +2.36, top-5 lift +5.96, vs q1s's +2.56 / +6.77),
    # though those margins are inside the null wobble and did NOT decide the choice; the render did.
    # 🔴 UNPRICED AGAINST A TRAINED ADVERSARY. The owner scoped the paper to model-free metrics and
    # the TM3 runs were stopped, so this arm has NO learned number. Its siblings measured
    # q1s 20.14 +/- 1.11 (ON the 20.63 bar, 1/3 seeds over) and q1 18.43 +/- 1.02; r2 is q1s plus
    # one knob and has no reason to be assumed better. Do not state a privacy claim beyond what the
    # untrained attacker in §A.2x supports.
    # ---- (mechanism) constant per-limb time shift, on top of the warp -------------------------
    # `_limb_phase_warp` is saturated -- perturbation caps at d50 0.063 and raising the amplitude
    # past ~1.8 buys nothing (§A.2v-1). But it saturates because a WARP of a smooth track can only
    # move a limb so far before it doubles back. A constant SHIFT has no such ceiling: a limb can
    # be a full half-cycle out of phase with its contralateral partner, which no warp amplitude
    # achieves. And a shift changes no local speed, so it adds no jerk -- it keeps q1s's smoothness
    # exactly. 0.35 s at a ~1 s gait cycle is up to a third of a cycle per chain, drawn per
    # sequence per chain.
    # PREDICTION: strongest on the DYNAMIC channel; should move top-5 if the residual is gait
    # rhythm rather than static shape. Risk to watch in the render: a limb far out of phase with
    # the torso can look uncoordinated even though each limb is individually smooth.
    "r2": dict(angle_groups=("uarm", "farm"), projection_fit=True,
               limb_phase_amp=1.80, cadence_amp=0.0,
               scale_from="extent", cadence_root_lock=True,
               seeded_global_scale_max=0.10,
               limb_phase_offset_s=0.35),

    # ---- r3 "RHYTHM": move the emitted step frequency off the subject's own -------------------
    # Step frequency is a strong, well-documented gait cue and NOTHING in the q-family currently
    # touches it: `cadence_amp` only wobbles the clock around the subject's own rate (and is off in
    # q1s), while `limb_phase_*` shifts phase without changing frequency. A UNIFORM rate change has
    # no local speed variation by construction, so unlike `cadence_amp` it costs no smoothness --
    # it is the one way to attack rhythm that q1s can afford. 1.18 is chosen to sit outside typical
    # within-subject cadence variation while staying inside plausible human walking rates; with the
    # root locked the BODY still travels the real path at the real speed, so only the limb cycling
    # changes and the figure stays registered on the person.
    # PREDICTION: if the top-5 residual is the subject's walking rhythm, this is the arm that moves
    # it. Risk to watch in the render: limbs cycling at a rate that disagrees with the body's
    # ground speed can read as sliding/skating.
    #
    # 🔴 RATE IS BELOW 1.0, AND THE SIGN MATTERS — a claim of mine that MEASUREMENT refuted.
    # I first set 1.18 on the reasoning that a uniform rate change has no local speed variation and
    # so adds no jerk. Measured on p03_c02: jerk 1.87x real, WORSE than `q1` (1.78x) and far worse
    # than `q1s` (1.19x) -- i.e. it reintroduced exactly the "sudden" motion this family exists to
    # remove. The error: a uniform remap is smooth in the TIME MAPPING but scales every derivative,
    # so joint velocity goes as `rate` and jerk as `rate^3` (1.18^3 = 1.64, matching the observed
    # 1.57x rise over q1s). SLOWING the pose moves the emitted step frequency off the subject just
    # as effectively and scales jerk DOWN (0.85^3 = 0.61), so it buys smoothness instead of
    # spending it. Never raise this above 1.0 without re-measuring jerk.
    "r3": dict(angle_groups=("uarm", "farm"), projection_fit=True,
               limb_phase_amp=1.80, cadence_amp=0.0,
               scale_from="extent", cadence_root_lock=True,
               seeded_global_scale_max=0.10,
               pose_rate=0.85),
    # ---- g10: the 2026-08-03 arm it replaces, kept so the A/B stays expressible ----------------
    "g10": dict(angle_groups=("uarm", "farm", "thigh", "shin"), projection_fit=True,
                limb_phase_amp=1.20, cadence_amp=0.90),
    # ---- g9proj: the 2026-08-02 candidate (§A.2i-4), superseded by g18 -------------------------
    "g9proj": dict(angle_groups=("uarm", "farm", "thigh", "shin"), projection_fit=True),
    # ---- q1 / q1w: QUALITY-FIRST (2026-08-12) — measure the size channel, then RANDOMISE it ----
    # THE TRADE-OFF THESE SOLVE. The owner's two quality complaints have known prices (FIX7, §A.2p):
    #   `extent` (right SIZE)                 20.12 ± 0.59      `shoulder`+root lock  15.87 ± 0.27
    #   `extent` + root lock (`g20`, BOTH)    21.07 ± 1.04  🔴 over the 20.63 % bar
    # so fixing size AND position together does not fit the budget — which is why every quality arm
    # so far has failed. The cost is not the size ERROR, it is the size MEASUREMENT: `extent` makes
    # emitted stature a faithful readout of THIS person's stature (§A.2p-3 prices a genuinely second
    # body dimension at ~15× the already-disclosed shoulder span).
    #
    # These arms keep the measurement — fidelity needs it — and destroy its INFORMATION with a fresh
    # per-SEQUENCE draw in [1−m, 1+m] on the collapse target. The adversary sees stature × noise; the
    # generator sees a figure whose typical size error is m instead of the 0.54×–1.16× that
    # `shoulder` produces. Because `_apply_projection_fit` re-squeezes X onto the observed shoulder
    # span AFTER the template scaling, the draw lands entirely on Y — i.e. on the +8.15 pp channel —
    # while X stays shoulder-derived, already priced at +0.60 pp. That axis asymmetry is the design.
    #
    # 🔴 DO NOT ADD `stature_ratio` OR `height_mult` TO THESE. `stature_ratio` RE-FITS emitted height
    # to a multiple of the observed height, which divides the random draw straight back out and
    # silently restores the unrandomised size channel — the same class of defect as `height_mult`
    # being cancelled by the projection fit (see the g18 note above). `scale_from="extent"` already
    # puts the figure at ~1.0× the subject's own height, so neither knob is needed here.
    # Owner cosmetic knobs (arms −7 %, neck +12.5 %) are the SAME constant for every subject, so they
    # cannot carry identity; layer them with MIRAGE_ARM_MULT / MIRAGE_NECK_MULT rather than baking
    # them in, so the preset stays bit-identical to the arm the adversary was trained against.
    #
    # 🔴 PRIVACY IS BEING MEASURED, NOT ASSUMED: TM3 arms `qsize_m10` / `qsize_m20` in
    # `tier1_lab/reid_eval/tm3_retrain.py` (R5 + this one knob, FIX7 protocol, 3 seeds, 100 ep).
    # Until those land NEITHER PRESET MAY BE SHIPPED and no privacy number may be quoted for them.
    "q1":  dict(angle_groups=("uarm", "farm"), projection_fit=True,
                limb_phase_amp=1.20, cadence_amp=0.90,
                scale_from="extent", cadence_root_lock=True,
                seeded_global_scale_max=0.10),
    "q1w": dict(angle_groups=("uarm", "farm"), projection_fit=True,
                limb_phase_amp=1.20, cadence_amp=0.90,
                scale_from="extent", cadence_root_lock=True,
                seeded_global_scale_max=0.20),
}
# 🔴 `head_anchor` is deliberately in NO preset. It delivers a 0.0 px emitted-face position and the
# owner asked for it, but the ADAPTIVE adversary prices it at +13.45 pp (g18 11.4 → g19 24.85),
# which breaks the pre-registered 20.63 % bar — while the FROZEN adversary priced the same knob at
# +1.32 pp, a 10.2× under-pricing (§A.2m-2c). It stays reachable via MIRAGE_HEAD_ANCHOR for
# experiments, and shipping it needs the bar re-negotiated, not quietly exceeded.


# ---------------------------------------------------------------- THE SHIPPED ARM (2026-08-14)
# ✅ OWNER DECISION: **e2 ships from now on.** Set here as the DEFAULT rather than left to an env
# var, for the §A.2d reason this whole preset mechanism exists: when the committed source says one
# thing and every rendered artifact was produced with another, no bundle can say what built it.
# With this default, source == tested == shipped, and the emitted `anon` block records
# `gait_preset: "e2"` so an artifact is self-identifying.
#
# `e2` = q1s + limb swing amplitude 0.25 + limb phase offset 0.35 s. It reaches the DYNAMICS
# channel -- the only channel still leaking (§A.2x) -- WITHOUT re-timing anything, which is why it
# alone avoids the +40..46 % jerk every rate-draw arm pays (§A.2y-5). Measured against the previous
# shipped `q1s`, untrained attacker N=5375 and jerk over n=5 independent Tier-1 runs at 10 fps:
#     jerk   0.95 x q1s (t = -2.28, statistically SMOOTHER)
#     top-1  +2.10 lift vs q1s's +2.56      (RAW +35.17)
#     top-5  +4.98 lift vs q1s's +6.77      (RAW +53.93)  -- 26.4 % better
# and it is the arm the owner selected from a RENDER (§B.64): feet grounded in all 49 frames.
#
# 🔴 WHY NOT `f3`, which beat it on every metric: f3 = e2 + `root_speed_amp` was smoother still
# (0.92 x) and better on top-1 (+1.96), and it LOST THE FEET on the render -- figure 11.1 % shorter,
# feet-band alpha -30 %. No pose-space metric predicted that. See §B.64-2.
# 🔴 `d2c` (+0.70 top-1, the best privacy measured) was never rendered, so it is NOT a candidate.
#
# To reproduce the pre-2026-08-14 bare behaviour, set MIRAGE_GAIT_PRESET="" explicitly.
SHIPPED_PRESET = "e2"


def gait_preset(name=None):
    """kwargs for `anonymize_v2` from the named preset, or {} for the bare LEVELS behaviour.

    Unknown names raise rather than silently falling back — a typo that quietly ships the default
    is the failure this whole mechanism exists to prevent.

    The default is `SHIPPED_PRESET` (2026-08-14), NOT the empty string: the owner's shipped arm has
    to be what runs when nobody sets anything. An explicit empty value still selects bare LEVELS.
    """
    n = (name if name is not None else os.environ.get("MIRAGE_GAIT_PRESET", SHIPPED_PRESET)).strip()
    if not n:
        return {}
    if n not in GAIT_PRESETS:
        raise ValueError(f"MIRAGE_GAIT_PRESET={n!r} is not a known preset; "
                         f"valid are {sorted(GAIT_PRESETS)} (or empty for the shipped default)")
    return dict(GAIT_PRESETS[n])


def anonymize_pose_log(pose_log, level, fps, frame_wh=None, slot_log=None):
    """Anonymise a whole clip's pose_log IN THE EMIT FORMAT of mirage_tier1.py:
    [frames][persons]{"kp": [[x,y]*133], "score": [s*133]}. Returns a NEW log
    (kp replaced; scores untouched -- binarize separately).

    Per person-SLOT, over the frames where that slot exists:
      * body 0..16 -> anonymize_v2 with do_canon + LEVELS[level] knobs,
        per-clip template (template_for_clip) and a fresh per-slot secrets seed;
      * feet 17..22 / hands 91..132 translate rigidly with their own ankle's /
        wrist's anonymisation delta (their pose relative to the anchor is kept);
      * face 23..90 stays zeroed (re-asserted here).

    `slot_log` ([frames][persons] of slot ids, from mirage_tier1's SlotTracker)
    defines the grouping. Omit it and grouping falls back to the old positional
    proxy (person p == index p) -- kept so pre-slot callers and the parity tests
    still work, but the tracked ids are what deployment passes: a single missed
    detection shifts positional indices and would splice two people into one
    "person", giving them ONE shared collapse template and ONE cadence warp.

    SEEDING: one fresh per-SEQUENCE secrets seed per slot. Never derived from
    identity/content/slot id -- a stable seed re-creates a linkable pseudo-
    identity (lab TM2id 88.5%, ledger A.2).
    """
    knobs = LEVELS[level]
    # The preset is resolved ONCE for the clip. `scale_from` is popped out here because it is not
    # an `anonymize_v2` kwarg — it selects the COLLAPSE TARGET, one level up. An explicit
    # MIRAGE_POSE_SCALE_FROM still wins so an experiment can override a preset, and the effective
    # source is declared in the artifact by `template_kinds`.
    preset = gait_preset()
    _proximity_cap = float(preset.get("proximity_cap_frac", 0.0))
    _preset_src = preset.pop("scale_from", None)
    _env_src = os.environ.get("MIRAGE_POSE_SCALE_FROM")
    scale_src = _env_src if (_env_src is not None and _env_src.strip()) else _preset_src
    # `height_mult` (2026-08-10): a UNIFORM post-scale on every bone-length group returned by the
    # collapse target, popped here for the same reason `scale_from` is -- it is not an
    # `anonymize_v2` kwarg, it rescales the COLLAPSE TARGET one level up. Applying it to every
    # group equally (not just the vertical axis) is what keeps arms/neck/legs in natural
    # proportion to the now-taller torso, since `anonymize_v2` rebuilds each bone as
    # anchor + factor * observed_direction -- a uniform factor on every target length scales the
    # whole chained skeleton isotropically rather than stretching it along one axis.
    _height_mult = preset.pop("height_mult", None)
    # Env overrides for the three 2026-08-10/11 size knobs, same pattern as MIRAGE_POSE_SCALE_FROM
    # above: an experiment must be able to sweep these through THIS code path rather than by
    # editing the preset, because a preset edit that is forgotten silently becomes what ships.
    _e = os.environ.get("MIRAGE_HEIGHT_MULT")
    if _e is not None and _e.strip():
        _height_mult = float(_e)
    # `arm_mult` (2026-08-10, owner direction): a post-scale on ONLY the "uarm"/"farm" template
    # groups, popped here for the same reason `height_mult` is. Applied AFTER `height_mult` so it
    # composes: final arm length = base * height_mult * arm_mult.
    _arm_mult = preset.pop("arm_mult", None)
    _e = os.environ.get("MIRAGE_ARM_MULT")
    if _e is not None and _e.strip():
        _arm_mult = float(_e)
    # `leg_mult` (2026-08-10): a post-scale on ONLY "thigh"/"shin", multiplying whatever
    # `height_mult` already applied -- e.g. height_mult=2.1, leg_mult=0.714 => legs end up at
    # 2.1*0.714 = 1.5x overall, capped well short of the rest of the body's 2.1x.
    _leg_mult = preset.pop("leg_mult", None)
    _e = os.environ.get("MIRAGE_LEG_MULT")
    if _e is not None and _e.strip():
        _leg_mult = float(_e)
    # `neck_mult` (2026-08-12, owner: "increase the neck length by 15-20%"): scales ONLY the "neck"
    # group, which is the MS->nose edge, i.e. how high the head rides above the shoulder line.
    # Nothing else moves; the head BLOCK (eyes/ears, group "face") keeps its own proportions and is
    # carried along by the chain, so this lengthens the neck rather than inflating the head.
    _neck_mult = preset.pop("neck_mult", None)
    _e = os.environ.get("MIRAGE_NECK_MULT")
    if _e is not None and _e.strip():
        _neck_mult = float(_e)
    # `stature_ratio` (2026-08-12) -- THE GENERAL FORM OF `height_mult`.
    #
    # `height_mult` is an ABSOLUTE multiplier, so a value tuned on one clip is wrong the moment the
    # subject stands closer to or further from the camera: 2.1 was fitted to a person ~629 px tall
    # and would make a near subject enormous (feet off-canvas) and a distant one still tiny. Every
    # OTHER knob in this preset is already dimensionless -- leg/arm/neck are ratios on template
    # groups, the mask margin is a fraction of the drawn figure's own height, the ankle anchor is a
    # positioning rule -- so `height_mult` was the single thing standing between this config and
    # working on any framing.
    #
    # `stature_ratio` replaces "scale by 2.1" with "make the emitted figure this multiple of THIS
    # subject's own observed height", and the multiplier is then solved per clip:
    #     scale = (median observed visible height x stature_ratio) / median emitted visible height
    # Both heights come from `_observed_pose_heights`, which already takes the per-frame vertical
    # extent over confidence-valid joints and back-fills missing frames with the clip median, so
    # occlusion and a cropped frame cannot blow the fit up. No pixel constant anywhere.
    #
    # 🔴 PRIVACY, STATED PLAINLY: this makes emitted SIZE track the individual, which a fixed
    # population constant did not. It is the same channel `scale_from="extent"` opens, and FIX7
    # priced that at 20.12 % -- ON the 20.63 % bar. The k-same SHAPE collapse is untouched (every
    # subject still lands on one proportion set), and the mask already discloses the silhouette's
    # size, but this MUST be re-measured on the adaptive adversary before any privacy claim.
    _stature_ratio = preset.pop("stature_ratio", None)
    _e = os.environ.get("MIRAGE_STATURE_RATIO")
    if _e is not None and _e.strip():
        _stature_ratio = float(_e)
    # `height_anchor` (2026-08-10, owner direction): WHICH joint stays close to its natural
    # (height_mult=1) position while the rest of the body grows around it -- "hip" (default) is
    # the plain symmetric scale with no correction at all. This is a STABLE per-frame formula (it
    # only reads the CURRENT frame's own rebuilt joints, never the frame boundary), so it tracks
    # the person's real motion smoothly and does not jitter frame to frame the way a
    # containment/frame-boundary based correction does.
    _preset_anchor = preset.pop("height_anchor", "hip")
    _env_anchor = os.environ.get("MIRAGE_HEIGHT_ANCHOR")
    height_anchor = (_env_anchor if (_env_anchor is not None and _env_anchor.strip())
                     else _preset_anchor)
    out = [[dict(person) for person in frame] for frame in pose_log]
    for _slot, cells in sorted(slot_groups(pose_log, slot_log).items()):
        kp = np.array([pose_log[i][j]["kp"] for i, j in cells], np.float64)   # (T,133,2)
        body = kp[:, :17, :]
        # per-joint confidence for the SAME cells, so the template can ignore frames where the
        # shoulders were not actually seen (scores are untouched by this function)
        sc_arr = np.array([pose_log[i][j].get("score", [1.0] * 133) for i, j in cells],
                          np.float64)[:, :17]
        # STANDARD SKELETON by default (2026-07-26). The legacy torso-scaled population
        # template produced anatomically impossible limbs on any framing where the lower body
        # is out of shot -- see _ANATOMICAL_OVER_SHOULDER above for the measured numbers.
        # Set MIRAGE_ANATOMICAL_TEMPLATE=0 to fall back to the legacy behaviour for an A/B.
        if os.environ.get("MIRAGE_ANATOMICAL_TEMPLATE", "1") not in ("0", "false", "False"):
            tmpl, tmpl_kind = anatomical_template_for_clip(body, scores=sc_arr, src=scale_src)
        else:
            tmpl, tmpl_kind = template_for_clip(body), "legacy_torso"
        # NOTE: `height_mult` is deliberately NOT applied to the template -- see the isotropic
        # post-scale after `anonymize_v2` below for why. `arm_mult`/`leg_mult` DO belong here:
        # they set limb proportions RELATIVE to the canonical body, which is a template property.
        if _arm_mult:
            # `arm_mult` (2026-08-10, owner direction): ONLY "uarm"/"farm" -- upper arm and
            # forearm. Nothing else in `tmpl` is touched, so torso/neck/legs/height_mult are
            # unaffected.
            for _g in ("uarm", "farm"):
                if _g in tmpl:
                    tmpl[_g] = tmpl[_g] * _arm_mult
        if _leg_mult:
            # `leg_mult` (2026-08-10, owner direction): caps how much of `height_mult` reaches the
            # legs, so they do not overshoot the mask by as much as the rest of the body -- paired
            # with growing the mask itself (see grow_mask_to_pose.py) rather than either fix alone.
            # ONE multiplier for BOTH "thigh" and "shin" keeps their ratio to each other (and to
            # the now height_mult*arm_mult-scaled rest of the body) proportional -- not a
            # different shrink per segment, which is what would look unnatural.
            for _g in ("thigh", "shin"):
                if _g in tmpl:
                    tmpl[_g] = tmpl[_g] * _leg_mult
        if _neck_mult and "neck" in tmpl:
            tmpl["neck"] = tmpl["neck"] * _neck_mult
        _TEMPLATE_KIND_USED.append(tmpl_kind)
        # DROP joints that were never actually observed. The collapse rebuilds every bone from
        # its measured direction, so a joint the detector never saw (binarised confidence 0.0)
        # gets a plausible LENGTH attached to a meaningless DIRECTION -- measured on a10 that
        # produced shins at 2.1-2.3x human and a right forearm at 0.24x, all on joints whose
        # confidence is exactly 0.000. Inventing limb geometry is worse than omitting it: the
        # renderer already skips (0,0) joints, and a pose-conditioned generator handed a
        # 2.3x-long shin is being told the person has one.
        _unobs = sc_arr < 0.5 if sc_arr is not None else None
        # HEAD ANCHOR is env-selected here (default off) so a Tier-1 build can turn it on without
        # a source edit — the knob itself is a kwarg of anonymize_v2 so the lab harness reaches
        # exactly the same code path. See _apply_head_anchor for the privacy warning.
        anon = anonymize_v2(body, tmpl, seed=new_clip_seed(),
                            fps=fps, do_canon=True, center_root=False,
                            head_anchor=os.environ.get("MIRAGE_HEAD_ANCHOR", "off"),
                            proximity_scores=sc_arr,
                            **dict(knobs, **preset))
        _ANCHOR_JOINTS = {"ankle": (15, 16), "knee": (13, 14), "shoulder": (5, 6),
                          "neck": (5, 6), "head": (0, 0), "nose": (0, 0)}
        # Resolve the size scale for THIS clip. `stature_ratio` (auto, generalises to any framing)
        # wins over `height_mult` (absolute, clip-specific) when both are present.
        _eff_scale = _height_mult
        if _stature_ratio:
            _obs_h = float(np.median(_observed_pose_heights(body, sc_arr)))
            _emi_h = float(np.median(_observed_pose_heights(anon, None)))
            _cand = (_obs_h * _stature_ratio) / _emi_h if (_obs_h > 1.0 and _emi_h > 1.0) else None
            # Never guess: a degenerate fit (subject barely visible, near-zero extent) falls back to
            # the explicit multiplier rather than emitting an arbitrarily scaled body.
            if _cand is not None and 0.2 < _cand < 8.0:
                _eff_scale = _cand
            else:
                _eff_scale = _height_mult or 1.0
                print(f"[size] stature fit degenerate (obs {_obs_h:.1f}px, emitted {_emi_h:.1f}px) "
                      f"-> falling back to height_mult {_eff_scale}", flush=True)
            _STATURE_SCALE_USED.append(round(float(_eff_scale), 4))
            print(f"[size] stature_ratio {_stature_ratio}: observed {_obs_h:.1f}px, emitted "
                  f"{_emi_h:.1f}px -> auto scale x{_eff_scale:.3f}", flush=True)
        if _eff_scale and _eff_scale != 1:
            # 🔴 ISOTROPIC SIZE SCALE -- APPLIED HERE, NOT ON THE TEMPLATE (fixed 2026-08-12).
            #
            # THE BUG THIS REPLACES. `height_mult` used to multiply the collapse target, i.e. the
            # bone LENGTHS, before `anonymize_v2` ran. But `anonymize_v2` step (2d) calls
            # `_apply_projection_fit`, which computes `sx = observed_shoulder_px(kp_seq) /
            # observed_shoulder_px(out)` and rescales the X axis so the emitted shoulder span
            # equals the REAL subject's -- and it runs AFTER the template multiply, so it divided
            # the multiplier straight back out of X. `height_mult` therefore acted as a PURE
            # VERTICAL STRETCH: measured, emitted shoulder width stayed pinned at 87-93 px while
            # stature went 367 -> 553 -> 741 px across height_mult none/1.9/2.1, collapsing
            # shoulder/stature from the anthropometric 0.234 to 0.145-0.160 -- a figure ~1.5x too
            # narrow for its height. That is an anatomically impossible body, and the generator,
            # bound only weakly by pose_strength 0.8 and carrying a strong human prior from the
            # reference image, resolves the contradiction by not committing the lower limbs: a
            # SMOOTH contrast falloff (not a hard clip) that reads as the feet dissolving. Measured
            # foot-vs-peak contrast fell 0.39 (good arm) -> 0.25 -> 0.16 as the stretch worsened.
            #
            # Scaling the FINISHED skeleton about each frame's own mid-hip is isotropic by
            # construction -- X and Y take the same factor and nothing downstream re-pins either --
            # so the emitted body keeps the canonical template's proportions at every size. The
            # mid-hip is also exactly the "hip anchor" the owner chose on 2026-08-10, so the
            # figure still grows symmetrically about the subject's own hip.
            # PRIVACY: unchanged in kind -- ONE population constant applied to every subject alike,
            # so the k-same collapse still lands every subject on one shared shape (unlike
            # `scale_from=extent`, which fits size to the INDIVIDUAL and is a per-clip size channel;
            # FIX7_PRIVACY priced that at 20.12 %, ON the 20.63 % bar).
            # 🔴 STILL UNPRICED: the emitted stature moves and no TM row covers this constant yet.
            _hip0 = 0.5 * (anon[:, 11] + anon[:, 12])                      # (T,2)
            _live = np.abs(anon).sum(2) > 0                                # never move a (0,0)
            _scaled = _hip0[:, None, :] + (anon - _hip0[:, None, :]) * _eff_scale
            anon = np.where(_live[:, :, None], _scaled, anon)
        if _eff_scale and _eff_scale != 1 and height_anchor not in (None, "", "hip"):
            # Re-anchor to a joint OTHER than the hip (owner direction, 2026-08-10). The uniform
            # bone-length scale above already fixes every group's proportion relative to every
            # other -- that part is correct and untouched. What differs per anchor is WHERE that
            # correctly-shaped rigid figure sits: this keeps `anchor_joint` close to where it
            # would sit at height_mult=1 (the un-multiplied canonical proportions) and lets the
            # multiplier's extra length show up on the OTHER side of that joint instead.
            # STABLE BY CONSTRUCTION: the shift is a closed-form function of THIS frame's own
            # hip/anchor positions only -- never the frame boundary -- so it tracks the person's
            # real per-frame motion exactly as smoothly as the hip anchor does, with no jitter.
            if height_anchor not in _ANCHOR_JOINTS:
                raise ValueError(f"MIRAGE_HEIGHT_ANCHOR={height_anchor!r} unknown; "
                                 f"valid: hip, {sorted(_ANCHOR_JOINTS)}")
            ja, jb = _ANCHOR_JOINTS[height_anchor]
            hip = 0.5 * (anon[:, 11] + anon[:, 12])                        # (T,2)
            anchor_pt = 0.5 * (anon[:, ja] + anon[:, jb])                  # (T,2)
            shift = -(1.0 - 1.0 / _eff_scale) * (anchor_pt - hip)
            anon = anon + shift[:, None, :]
        if frame_wh and _proximity_cap:
            # `anonymize_v2` operates in detector coordinates, which may legitimately extend a
            # few pixels beyond the image. Deployment emits canvas-clipped coordinates. Clipping
            # only the candidate can increase its distance from an off-canvas raw joint, so make
            # the final guarantee in the emitted coordinate system against the equally clipped
            # raw reference. This second pass also covers any wrapper-level height-anchor shift.
            visible_ref = body.copy()
            visible_anon = anon.copy()
            for arr in (visible_ref, visible_anon):
                arr[:, :, 0] = np.clip(arr[:, :, 0], 0, frame_wh[0])
                arr[:, :, 1] = np.clip(arr[:, :, 1], 0, frame_wh[1])
            anon = _apply_proximity_envelope(visible_anon, visible_ref, _proximity_cap,
                                             scores=sc_arr)
        delta = anon - body                                                # (T,17,2)
        kp[:, :17] = anon
        for seg, anchor in _ATTACH:
            kp[:, seg] += delta[:, anchor:anchor + 1]
        kp[:, _FACE] = 0.0
        # zero the never-observed body joints (see _unobs above). (0,0) is the pipeline's
        # existing "not detected" marker, which every consumer already skips.
        if _unobs is not None and _unobs.any():
            kp[:, :17][_unobs] = 0.0
            # ...AND the attached sub-skeletons of any anchor we just zeroed. `_ATTACH` translates
            # the 21-point hand blocks (91..132) and the foot blocks by their wrist's / ankle's
            # delta, so zeroing only `kp[:, :17]` left a full hand floating at a position derived
            # from a joint the pipeline had just declared unobserved. Today the renderer happens to
            # hide it (it needs >= 8 valid hand points and a10 has 0/200 person-instances at that
            # bar), so this was latent rather than live — but "a hand is emitted only because a
            # detector was confidently wrong about a wrist that was never in shot" is exactly the
            # class of assertion this pass exists to remove, and a downstream consumer that reads
            # the hand block directly would have believed it. Found 2026-07-26 while chasing the
            # hallucinated hands; the defect it fixes is NOT the one being chased (§B.23-CORRECTION).
            for _seg, _anchor in _ATTACH:
                _bad = _unobs[:, _anchor]
                if _bad.any():
                    kp[_bad, _seg] = 0.0
        if frame_wh:
            kp[:, :, 0] = np.clip(kp[:, :, 0], 0, frame_wh[0])
            kp[:, :, 1] = np.clip(kp[:, :, 1], 0, frame_wh[1])
        for t, (i, j) in enumerate(cells):
            out[i][j] = dict(out[i][j], kp=np.round(kp[t], 2).tolist())
    return out


def binarize_pose_scores(pose_log, thresh):
    """Binarize every emitted confidence to {0,1} at `thresh` (the pipeline's
    keypoint threshold): the raw per-joint confidence trace is an identity
    side-channel (lab strip_conf ablation). Returns a NEW log."""
    return [[dict(person, score=[1 if s >= thresh else 0 for s in person["score"]])
             for person in frame] for frame in pose_log]
