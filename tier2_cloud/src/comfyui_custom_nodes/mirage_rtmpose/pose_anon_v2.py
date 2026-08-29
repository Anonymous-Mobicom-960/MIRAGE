"""
mirage_rtmpose.pose_anon_v2 -- CLOUD-side anti-re-ID pose anonymiser (V9 Phase 1).
===================================================================================
SELF-CONTAINED numpy-only port of the lab-validated anonymiser
(evaluation/segbench/pose_anon_v2.anonymize_v2 + the helpers it pulls from
evaluation/segbench/pose_anon). The lab tree does NOT exist on the pod, so
everything needed at runtime is replicated here VERBATIM -- numeric parity with
the lab implementation (same input/template/seed => equal output) is asserted by
the scratchpad test that accompanied this change.

WHY (threat model, gait channel): RTMPoseTinyPoseAndFace.detect() builds BOTH
the exported pose.json AND the WanAnimate driving pose (pose_metas) from the
same raw kp2ds_seq -- so the rendered character replays the real person's gait
and the export is gait-reID-able even with face landmarks zeroed. The lab
evaluation (evaluation/privacy/gait, GaitGraph2 adversary on CASIA-B) validated:
  * static collapse (do_canon): every skeleton's limb proportions snap to ONE
    shared population template -> the anthropometric channel (limb ratios,
    relative height) carries zero identity (k-same, non-invertible);
  * L4 dynamic knobs (cadence warp .60, angle const 20deg, drift 15deg,
    asymmetry .12 -- evaluation/privacy/gait/sweep_pareto.py LADDER): perturb the
    gait dynamics where most identity lives;
  * PER-RUN random seed (secrets, never identity/content-derived): no stable
    pseudo-identity across clips;
  * confidence BINARIZED to {0,1}: kills the per-joint confidence trace as an
    identity side-channel (lab strip_conf ablation);
  * center_root=False: the global mid-hip trajectory is KEPT -- the composite
    needs the character placed on the real person's path.

Deployment semantics vs the lab: the lab template is ABSOLUTE bone lengths in
CASIA-B pixels. In a real clip the subject's on-frame size is arbitrary (camera
distance), so the template is baked as unit-free RATIOS (bone length / body
scale) and rebuilt per clip from the subject's own auto-computed median body
scale (torso length). Proportions still collapse to the population skeleton
(identity destroyed); on-frame size/position stay correct for the composite.
No per-clip constant is hardcoded anywhere -- the scale is auto-detected per
clip, the ratios are population-level. (Identical constants + semantics as the
Tier-1 edge port, tier1/src/edge_runner_pi5/pose_anon_edge.py.)
"""
import secrets
import numpy as np

# ----------------------------------------------------------------------------
# COCO-17 kinematic tree (verbatim from evaluation/segbench/pose_anon.py):
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

# RTMPose wholebody-133 layout (matches RTMPoseTinyPoseAndFace's own slicing:
# FACE_KP_START..FACE_KP_END = 23..90):
#   0..16 COCO body | 17..19 L foot | 20..22 R foot | 23..90 face (68)
#   91..111 L hand (21) | 112..132 R hand (21)
# Appendage blocks ride RIGIDLY on their body-17 anchor's anonymisation delta:
# their pose RELATIVE to the anchor is kept, only their placement follows the
# anonymised skeleton. CLOUD-specific: the face block (23..90) is attached to
# the nose too -- unlike the Tier-1 edge port (which zeroes it, its log IS the
# export), here kp2ds_seq also drives WanAnimate via pose_metas, so the driving
# head must stay attached to the anonymised body. A pure translation preserves
# the face-landmark GEOMETRY exactly, so nothing new leaks vs the status quo --
# and pose.json still zeroes 23..90 downstream in detect() (untouched).
_ATTACH = [  # (wholebody slice, body-17 anchor joint whose delta it follows)
    (slice(17, 20), 15),    # left foot  -> left ankle
    (slice(20, 23), 16),    # right foot -> right ankle
    (slice(91, 112), 9),    # left hand  -> left wrist
    (slice(112, 133), 10),  # right hand -> right wrist
    (slice(23, 91), 0),     # face       -> nose (cloud-only, see note above)
]

# detect() has no fps input and ComfyUI gives the node no frame-rate signal, so
# the knob periods (cadence_period_s=0.7, lowfreq_period_s=1.2) are converted
# with an ASSUMED rate. 16 fps = WanAnimate's native rate and mid-band for the
# pipeline's driving clips (10-30 fps); the dynamic-knob AMPLITUDES (the privacy
# dose) are fps-independent, a mismatch only shifts the wobble band by <2x.
FPS_ASSUMED = 16.0


# --------------------------------------------------------------------- helpers
# (verbatim ports from evaluation/segbench/pose_anon.py)
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
    # np.convolve(..., "same") returns max(len(w), len(k)) -- NOT len(w). A sequence shorter than
    # the kernel therefore came back LONGER than it went in, which crashed the deployed edge
    # anonymiser on short person-slots (see pose_anon_edge._lowfreq_noise). Keep the CENTRED T
    # samples; no-op whenever T >= len(k), so all existing (long-clip) output is bit-identical.
    if s.size != T:
        off = (s.size - T) // 2
        s = s[off:off + T]
    sd = s.std()
    return (s / sd * amp) if sd > 1e-9 else s


# ----------------------------------------------------------- template (k-same)
# (verbatim ports from evaluation/segbench/pose_anon_v2.py)
def group_lengths(kp_seq):
    """Per-GROUP median bone length over a clip (T,17,2). {group: length}."""
    per = {g: [] for g in _GROUPS}
    for t in range(kp_seq.shape[0]):
        o = _virtual(kp_seq[t])
        for p, c, g in EDGES:
            per[g].append(float(np.linalg.norm(o[c] - o[p])))
    return {g: float(np.median(v)) if v else 1.0 for g, v in per.items()}


def _rebuild_frame(kp, len_factors, angle_off):
    """Rebuild 17 joints from mid-hip, applying per-group length scale AND a 2D
    rotation of each bone direction by angle_off[group] (radians). Mid-hip fixed."""
    o = _virtual(kp)
    n = {"MH": o["MH"].copy()}
    for parent, child, grp in EDGES:
        bone = o[child] - o[parent]
        a = angle_off.get(grp, 0.0)
        if a:
            ca, sa = np.cos(a), np.sin(a)
            bone = np.array([ca * bone[0] - sa * bone[1], sa * bone[0] + ca * bone[1]])
        bone = bone * len_factors[grp]
        n[child] = n[parent] + bone
        if child == "MS":
            n["MS"] = n[parent] + bone
    return np.stack([n[i] for i in range(17)], axis=0)


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


# ------------------------------------------------------------------ main entry
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
                 center_root=False):         # subtract per-frame mid-hip (kill global trajectory)
    """kp_seq:(T,17,2) -> anonymised (T,17,2). `template`: {group: target length}.
    VERBATIM port of evaluation/segbench/pose_anon_v2.anonymize_v2 -- do not
    "improve" anything here: identical RandomState call ORDER is what keeps
    numeric parity with the validated lab implementation."""
    kp_seq = np.asarray(kp_seq, np.float64)
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
        for g in _LR_GROUPS:
            len_factors[g] *= float(1 + r.randn() * asymmetry_sigma)

    # (2a) per-limb angular offsets: constant + slow drift, per group (asymmetric)
    ac = np.deg2rad(angle_const_deg)
    ad = np.deg2rad(angle_drift_deg)
    const_ang = {g: (r.randn() * ac) for g in _GROUPS}
    drift_ang = {g: (_lowfreq_noise(T, max(1.0, lowfreq_period_s * fps / 3.0), ad, seed * 17 + gi * 101 + 3)
                     if ad > 0 else np.zeros(T)) for gi, g in enumerate(_GROUPS)}

    # positional low-freq drift (naturalness), per joint/axis
    scale = np.median([_body_scale(kp_seq[t]) for t in range(T)])
    amp = lowfreq_amp_frac * scale
    if amp > 0:
        nse = np.stack([[_lowfreq_noise(T, max(1.0, lowfreq_period_s * fps / 3.0), amp, seed * 131 + j * 7 + a)
                         for a in range(2)] for j in range(17)], axis=0)  # (17,2,T)
    else:
        nse = np.zeros((17, 2, T))

    out = np.empty_like(kp_seq)
    for t in range(T):
        angle_off = {g: const_ang[g] + drift_ang[g][t] for g in _GROUPS}
        rt = _rebuild_frame(kp_seq[t], len_factors, angle_off)
        out[t] = rt + nse[:, :, t]

    # (2b) cadence / rhythm warp (temporal)
    if cadence_amp > 0:
        out = _cadence_warp(out, cadence_amp, cadence_period_s, fps, seed * 991 + 1)
    # (3) optional: remove the global walking trajectory (per-frame root-centering)
    if center_root:
        mh = 0.5 * (out[:, 11:12, :] + out[:, 12:13, :])   # (T,1,2)
        out = out - mh
    return out


# ------------------------------------------------- baked population template
# PROVENANCE: computed 2026-07-22 by the lab's own population_template()
# (evaluation/segbench/pose_anon_v2.py) over the CASIA-B GaitGraph TRAIN pool
# (evaluation/privacy/gait/data/casia-b_pose_train_valid.csv, subject ids 1..74,
# min_frames=61), sampled to 500 tracklets with RandomState(0) -- exactly
# reproducing evaluation/privacy/gait/anon_adapter.build_template(sample_n=500,
# seed=0). Each per-group population-median bone length (px) was divided by
# the population-median body scale (_body_scale: mid-shoulder..mid-hip torso
# length, median-per-clip then median-over-clips) to make it unit-free.
# These are POPULATION-level constants (aggregate over 74 identities -- not
# per-clip, not per-identity); the per-clip quantity (body scale) is
# auto-detected at runtime by template_for_clip().
# Derivation stats: 7966 train tracklets, 500 sampled, pop body scale
# 37.45170331303146 px; "spine" == 1.0 exactly because the spine bone
# (MH->MS) IS the body-scale segment -- a built-in consistency check.
# IDENTICAL constants are baked in the Tier-1 edge port (pose_anon_edge.py):
# edge and cloud collapse every skeleton to the SAME population skeleton.
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
# Knob ladder from the lab Pareto sweep (evaluation/privacy/gait/sweep_pareto.py
# LADDER): the four dynamic-channel knobs per level; do_canon is always True.
LEVELS = {
    "L0": dict(cadence_amp=0.00, angle_const_deg=0, angle_drift_deg=0, asymmetry_sigma=0.00),
    "L2": dict(cadence_amp=0.25, angle_const_deg=8, angle_drift_deg=6, asymmetry_sigma=0.05),
    "L4": dict(cadence_amp=0.60, angle_const_deg=20, angle_drift_deg=15, asymmetry_sigma=0.12),
}


def new_clip_seed():
    """PER-RUN seed: fresh OS randomness, NEVER derived from identity or
    content (a stable seed would re-create a linkable pseudo-identity). The
    seed is used in-process only and must never be emitted/logged."""
    return secrets.randbits(31)


def anonymize_kp133_seq(kp2ds_seq, anon_level="L4", conf_binarize=True,
                        pose_thresh=0.3, fps=FPS_ASSUMED, seed=None):
    """Anonymise a whole clip's kp2ds_seq IN detect()'s OWN format: a sequence
    of (133,3) float arrays [x, y, score] per frame (one tracked person).
    Returns a NEW list of float32 (133,3) -- the input is not mutated.

    Applied ONCE, BEFORE pose_json and pose_metas are built, so the export and
    the WanAnimate driving pose carry the SAME anonymised skeleton:
      * scores -> BINARIZED to {0,1} at `pose_thresh` (the node's keypoint
        threshold). MANDATORY whenever anon_level != "off" (continuous
        confidence is an identity side-channel -- lab strip_conf ablation);
        `conf_binarize=False` is only honoured in the raw "off" debug mode.
      * body 0..16 -> anonymize_v2 with do_canon + LEVELS[anon_level] knobs,
        per-clip template (template_for_clip) and a fresh secrets seed
        (`seed` is an override for TESTS only);
      * feet 17..22 / hands 91..132 / face 23..90 translate rigidly with their
        anchor joint's delta (_ATTACH; face rides the nose -- see note there);
      * all-zero placeholder PREFIX frames (track not yet visible; detect()
        freezes later dropouts, so blanks can only be a prefix) are left
        untouched -- anonymising across them would blend zeros into real
        frames via the cadence warp's temporal interpolation.
    anon_level: "L4" (lab-validated deployment default) | "L2" | "L0" (canon
    only) | "off" (raw passthrough, debug only)."""
    frames = [np.asarray(f, np.float64) for f in kp2ds_seq]
    if not frames:
        return list(kp2ds_seq)
    kp3 = np.stack(frames, axis=0).copy()                       # (T,133,3)
    T = kp3.shape[0]
    if conf_binarize or anon_level != "off":
        kp3[:, :, 2] = (kp3[:, :, 2] >= pose_thresh).astype(np.float64)
    if anon_level != "off":
        knobs = LEVELS[anon_level]
        valid = np.abs(kp3[:, :17, :2]).reshape(T, -1).max(axis=1) > 1e-6
        if valid.any():
            first = int(np.argmax(valid))
            body = kp3[first:, :17, :2].copy()                  # (Tv,17,2) raw
            if seed is None:
                seed = new_clip_seed()
            anon = anonymize_v2(body, template_for_clip(body), seed=seed,
                                fps=fps, do_canon=True, center_root=False,
                                **knobs)
            delta = anon - body                                 # (Tv,17,2)
            seg = kp3[first:]                                   # view into kp3
            for sl, anchor in _ATTACH:
                seg[:, sl, :2] += delta[:, anchor:anchor + 1]
            seg[:, :17, :2] = anon
    return [f.astype(np.float32) for f in kp3]
