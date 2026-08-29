"""
segbench.pose_anon -- anti-re-identification perturbation for 2D pose (COCO-17)
body sticks / keypoints, so the pose signal that leaves Tier-1 can't be used to
re-identify the real person by GAIT, LIMB-LENGTH, HEIGHT, or POSTURE (biometrics
that survive body-appearance masking -- see Sensors 2026 "Privacy Beyond the
Face"; arXiv 2405.05428 skeleton retargeting; GaitRef on why naive jitter fails).

Design principle (the whole point): anonymize by a STRUCTURAL, temporally-
CONSISTENT transformation, NOT additive per-frame jitter. Per-frame Gaussian
noise is both visibly jittery AND trivially removed by a low-pass filter (which
recovers the original gait). Instead:

  1. Bone-length retargeting -- rescale each limb's length by a per-person factor
     held CONSTANT over the whole clip. Changes height / limb ratios / proportions
     (strong biometrics) with ZERO added jitter (it's a constant reparametrisation
     of every frame). Symmetric L/R factors keep the body natural-looking.
  2. Low-frequency trajectory perturbation -- band-limited (Gaussian-smoothed)
     noise added to each joint. Slowly varying => no tremor, and being itself
     low-frequency it CANNOT be low-pass filtered away (unlike high-freq jitter).
  3. Constant posture bias -- a small fixed lean/rotation.

All three are deterministic in a per-identity seed, so a given person is
transformed the same way every frame (temporal coherence) but differently from
everyone else.
"""
import numpy as np

# COCO-17: 0 nose 1 leye 2 reye 3 lear 4 rear 5 lsh 6 rsh 7 lelb 8 relb 9 lwr
# 10 rwr 11 lhip 12 rhip 13 lknee 14 rknee 15 lank 16 rank
# Kinematic tree rooted at a virtual mid-hip ('MH'); 'MS' = virtual mid-shoulder.
# Each edge: (parent, child, sym_group) -- symmetric groups share a scale factor.
EDGES = [
    ("MH", "MS", "spine"), ("MH", 11, "hip"), ("MH", 12, "hip"),
    (11, 13, "thigh"), (13, 15, "shin"), (12, 14, "thigh"), (14, 16, "shin"),
    ("MS", 5, "clav"), ("MS", 6, "clav"), ("MS", 0, "neck"),
    (5, 7, "uarm"), (7, 9, "farm"), (6, 8, "uarm"), (8, 10, "farm"),
    (0, 1, "face"), (1, 3, "face"), (0, 2, "face"), (2, 4, "face"),
]
_GROUPS = sorted({e[2] for e in EDGES})


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


def retarget_frame(kp, factors):
    """Rebuild a frame's 17 joints from the mid-hip outward, scaling each bone's
    LENGTH by factors[group] but keeping its direction (so motion/pose is
    preserved, only proportions change). Mid-hip stays put (person doesn't move)."""
    o = _virtual(kp)
    n = {"MH": o["MH"].copy()}
    for parent, child, grp in EDGES:
        bone = o[child] - o[parent]
        n[child] = n[parent] + bone * factors[grp]
        if child == "MS":
            n["MS"] = n[parent] + bone * factors[grp]
    out = np.stack([n[i] for i in range(17)], axis=0)
    return out


def _rng(seed):
    return np.random.RandomState(seed & 0x7fffffff)


def make_identity_params(seed, bone_sigma=0.12, posture_deg=4.0):
    """Per-identity constant transform: bone-length factors + a posture lean."""
    r = _rng(seed)
    factors = {g: float(np.clip(1.0 + r.randn() * bone_sigma, 1 - 2 * bone_sigma, 1 + 2 * bone_sigma))
               for g in _GROUPS}
    factors["face"] = 1.0  # leave the (already-canonicalised) face geometry alone
    lean = np.deg2rad(r.uniform(-posture_deg, posture_deg))
    return dict(factors=factors, lean=lean)


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


def anonymize_sequence(kp_seq, seed=1234, bone_sigma=0.12, posture_deg=4.0,
                       lowfreq_amp_frac=0.03, lowfreq_period_s=1.2, fps=10.0):
    """kp_seq: (T,17,2). Returns anonymised (T,17,2).
    - bone retargeting + posture lean: constant per identity (no jitter)
    - low-freq perturbation: amplitude = lowfreq_amp_frac * body_scale, smoothed
      over ~lowfreq_period_s so it never tremors."""
    T = kp_seq.shape[0]
    p = make_identity_params(seed, bone_sigma, posture_deg)
    scale = np.median([_body_scale(kp_seq[t]) for t in range(T)])
    sigma_frames = max(1.0, lowfreq_period_s * fps / 3.0)
    amp = lowfreq_amp_frac * scale
    # one smooth noise track per joint per axis
    nse = np.stack([[_lowfreq_noise(T, sigma_frames, amp, seed * 131 + j * 7 + a)
                     for a in range(2)] for j in range(17)], axis=0)  # (17,2,T)
    c, s = np.cos(p["lean"]), np.sin(p["lean"])
    R = np.array([[c, -s], [s, c]])
    out = np.empty_like(kp_seq, dtype=np.float64)
    for t in range(T):
        rt = retarget_frame(kp_seq[t], p["factors"])
        mh = 0.5 * (kp_seq[t][11] + kp_seq[t][12])
        rt = (rt - mh) @ R.T + mh          # constant posture lean about mid-hip
        rt = rt + nse[:, :, t]             # + slow low-freq drift
        out[t] = rt
    return out, p


# ---------------------------------------------------------------------------
# Metrics: (A) JITTER must not rise (temporal smoothness preserved),
#          (B) IDENTITY features must shift (biometric destroyed).
# ---------------------------------------------------------------------------
def jitter(kp_seq):
    """Mean per-joint frame-to-frame acceleration magnitude (2nd difference).
    Lower = smoother. The anon output should be ~equal to the input (no tremor)."""
    a = kp_seq[2:] - 2 * kp_seq[1:-1] + kp_seq[:-2]
    return float(np.mean(np.linalg.norm(a, axis=-1)))


def bone_lengths(kp_seq):
    v = []
    for t in range(kp_seq.shape[0]):
        o = _virtual(kp_seq[t])
        v.append([np.linalg.norm(o[c] - o[p]) for p, c, _ in EDGES])
    return np.mean(v, axis=0)


def identity_shift(orig, anon):
    """Relative change in the mean bone-length signature (height/limb biometric)
    and in a gait spectrum (ankle vertical trajectory dominant frequency band)."""
    bo, ba = bone_lengths(orig), bone_lengths(anon)
    bone_rel = float(np.linalg.norm(ba - bo) / (np.linalg.norm(bo) + 1e-9))

    def gait_spec(seq):
        y = 0.5 * (seq[:, 15, 1] + seq[:, 16, 1])   # mean ankle height over time
        y = y - y.mean()
        f = np.abs(np.fft.rfft(y * np.hanning(len(y))))
        return f / (f.sum() + 1e-9)
    so, sa = gait_spec(orig), gait_spec(anon)
    L = min(len(so), len(sa))
    gait_l1 = float(np.abs(so[:L] - sa[:L]).sum())    # 0=identical .. up to 2
    return dict(bonelen_rel_shift=bone_rel, gait_spectrum_l1=gait_l1)
