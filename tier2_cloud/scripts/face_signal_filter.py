"""
face_signal_filter - turn Tier-1's DP-noised 12-channel face_scalars.json into a SMOOTH, STABLE,
plausible expression + head-pose trajectory for the synthetic (identity-free) face renderer.

===============================================================================================
THE HEADLINE MEASURED RESULT - READ THIS BEFORE USING THE MODULE
===============================================================================================
At the SHIPPED DP setting (DP_EPSILON_TOTAL=3.0, per-frame unit, Delta_c = public channel range),
`face_scalars.json` contains **NO STATISTICALLY DETECTABLE EXPRESSION OR HEAD-POSE SIGNAL**.
It is, to the precision of these measurements, pure Laplace noise. Therefore no filter can
"recover" a trajectory from it, and this module's correct behaviour at eps=3.0 is to output a
stable neutral pose plus (optionally) a data-INDEPENDENT idle animation.

Measured evidence (all numbers derived 2026-07-26 from
_e2e/fps_ladder_20260726/{a10,a15,a30}/tier1_out/face_scalars.json - the SAME
10 s clip released three times at 10/15/30 fps; N = 100 / 150 / 300 frames x 1 person x 12 ch):

 (E1) THE OBSERVED VARIANCE IS THE PREDICTED NOISE VARIANCE.
      Laplace scale predicted from config.py (b_c = Delta_c/eps_c, Delta_c = (A_w + w/(L-1)) *
      (hi_c-lo_c)) gives noise sd = sqrt(2)*b_c. Measured sd / predicted sd, over all 12 channels
      and all 3 releases (N=36): min 0.78, median 0.97, max 1.22. There is no measurable variance
      left over for a signal.

 (E2) THE SERIES IS TEMPORALLY WHITE.
      Lag-1 autocorrelation, all 12 channels x 3 releases (N=36): |rho1| <= 0.239, median 0.042.
      A real expression/head-pose trajectory smoothed by a 0.47 s window-mean would have rho1
      close to +1. It does not.

 (E3) THE UNBIASED CROSS-RELEASE TEST FINDS ZERO SIGNAL.
      a10, a15 and a30 are three releases of the SAME source video with INDEPENDENT Laplace draws
      (`_laplace_secure` seeds from `secrets`). So for two releases decimated onto a common
      timeline, Cov(x_A, x_B) is an UNBIASED estimator of Var(true signal). Against a null built
      from circular shifts: across 3 independent pairings x 12 channels (N=36 tests), 34 gave
      |z| < 1.96. The 2 exceedances were browR z=+2.24 and gazeY z=-2.47 - 1.8 exceedances are
      expected by chance at N=36, and a NEGATIVE covariance is impossible for a genuinely shared
      signal. Conclusion: 0/36 evidence of recoverable signal.
      The one-sided 95% upper bound on the signal sd is >= the channel's own admissible range on
      11 of 12 channels (e.g. yaw: sd_signal <= 15.6 rad against a +-1.5 rad channel) - the data
      cannot even bound the signal to inside its own legal range.

 (E4) 92-93% OF EMITTED SAMPLES ARE OUTSIDE THE PHYSICALLY ADMISSIBLE CH_BOUNDS.
      apply_dp CLAMPS the signal to CH_BOUNDS *before* adding noise, so every out-of-bounds
      sample is a pure noise excursion. Per release: a10 93.2%, a15 92.0%, a30 92.7%
      (N = 1200 / 1800 / 3600 channel-samples). Observed a10 spans vs the legal range:
      yaw 208.0 rad = 69x, roll 149.7 = 75x, gazeY 134.6 = 135x, mouth_open 35.8 = 36x.
      This is also the cheapest runtime sanity check available - see `out_of_bounds_fraction()`.

 (E5) NO FILTER HELPS; IGNORING THE DATA BEATS USING IT.
      200-trial Monte-Carlo, synthetic ground truth (slow head turn / 2.5 Hz speech / three
      0.12 s blinks) plus the REAL per-channel Laplace scale. RMSE in units of the channel range,
      best of {gauss sigma 1/5/15, Savitzky-Golay w=31/61 p2, RTS constant-velocity Kalman
      q=1e-4/1e-6, median-21+gauss-5, TV, clip-mean}:
          channel            best data-driven filter   emit NEUTRAL CONSTANT   winner
          yaw (pose)              0.763 (clip-mean)            0.127          neutral by 6.0x
          mouth_open              0.539 (clip-mean)            0.156          neutral by 3.5x
          eyeR_open (blink)       0.474 (clip-mean)            0.080          neutral by 5.9x
      The best DATA-DRIVEN estimator is the degenerate one (a constant), and it is still 3.5-6.0x
      WORSE than emitting a fixed neutral value. Per the task's own criterion, "a filter that
      flattens the signal to a constant is a FAILURE" - every filter tested flattens, and the
      flattening is optimal. That is a failure OF THE SIGNAL, not of the filters.
      Apparent correlation with truth is a lag-scanning artifact: r of gauss(sigma=15) output vs
      truth, EXCESS OVER the same statistic computed on pure noise, is +0.020 (yaw), +0.000
      (mouth_open), +0.000 (eyeR_open).

 (E6) BLINKS DO NOT SURVIVE. NOT RECOVERABLE at this epsilon.
      Detecting a 0.12 s blink (eyeR_open, 10 fps, N=200 trials, 1-sigma threshold on a
      gauss-sigma-2 smooth): 14.7% detection rate against a 14.9% false-positive rate on
      non-blink frames. That is exactly chance. The renderer MUST NOT drive blinks from
      channels 3/4 - doing so renders noise. `trust` for channels 3/4 is 0 and stays 0.

 (E7) SMOOTHING ALONE DOES NOT FIX THE HEAD TILT - GATING DOES.
      On the real a10 yaw series (N=100, channel range 3.0 rad): raw per-frame jitter is
      34.15 rad/frame. An RTS constant-velocity smoother (q=1e-6, R = the known noise variance)
      cuts jitter to 0.073 rad/frame - but its OUTPUT still has sd 2.11 rad and a 7.04 rad span,
      i.e. it still swings the head +-2 rad, 2.3x the entire legal range. Smoothing removes the
      buzz and leaves a large, slow, entirely fictional head sweep. Only the variance gate (below)
      removes it.

 (E8) OPTIMAL SHRINKAGE WEIGHT ON THE DATA IS ~0.1%.
      With a prior signal sd of range/6 and the known noise variance reduced by the per-channel
      effective smoothing length W_eff, the Wiener/James-Stein gain g = var_s/(var_s+var_n/W_eff)
      evaluates to: gaze 0.00055, eyes 0.00153, roll 0.00259, mouth/smile/brow 0.00382,
      yaw/pitch 0.00459. The released data legitimately earns 0.06%-0.5% of the output.

 (E9) WHAT WOULD MAKE THE SIGNAL USABLE (the experiment that settles it).
      For post-smoothing noise sd <= 0.20 x channel range at the W_eff below, the required
      per-channel epsilon is:
          mouth_open/mouth_width/smile/browR/browL (W_eff=5)  4.25  (now 0.316) = 13.5x
          eyeR_open/eyeL_open (W_eff=2, blinks are short)     6.72  (now 0.316) = 21.3x
          yaw/pitch (W_eff=15)                                2.45  (now 0.20)  = 12.3x
          roll (W_eff=15)                                     2.45  (now 0.15)  = 16.4x
          gazeX/gazeY (W_eff=5)                               4.25  (now 0.12)  = 35.4x
          TOTAL                                              50.58  (now 3.00)  = 16.8x
      i.e. DP_EPSILON_TOTAL would have to rise from 3.0 to ~50 (with the same per-channel
      split shape) before ANY of this is animatable. At eps~50 the shrinkage gain reaches 0.41.
      Alternatives that do NOT require weakening DP: (a) release a per-clip SUMMARY (a handful of
      numbers over the whole clip costs far less budget than 100-300 per-frame values);
      (b) accept that the face is procedurally animated and identity-free by construction.
      NOT MEASURED: whether a per-clip summary at eps=3.0 is usable - that needs a Tier-1 run
      emitting clip-level aggregates, which does not exist yet.

===============================================================================================
WHAT THIS MODULE THEREFORE DOES
===============================================================================================
`filter_face_scalars()` implements the full principled chain, so that it is CORRECT AT ANY
EPSILON and degrades to the measured truth at the shipped one:

  1. reconstruct the exact per-channel Laplace sd from the PUBLIC DP config (no privacy cost - 
     b_c, Delta_c, CH_BOUNDS and eps_c are all public constants in config.py);
  2. per channel, estimate the signal variance as max(0, var(x) - var_noise) - an unbiased
     method-of-moments estimate, floored at 0;
  3. compute the shrinkage gain g_c against the known noise floor, and a `trust` flag;
  4. RTS constant-velocity Kalman smoother with R set to the KNOWN noise variance (we know it
     exactly - that is the one advantage of a disclosed mechanism);
  5. shrink toward the neutral prior:  y = neutral + g_c * (smoothed - neutral);
  6. clamp to CH_BOUNDS;
  7. slew-rate limit + critically-damped second-order follower on the pose channels - this is
     what makes the neck/face tilt STABLE and SMOOTH rather than snapping.

At eps=3.0 steps 2-3 measure g_c ~ 0.001-0.005 and the output is a stable neutral head, which is
the measured-optimal behaviour (E5). Raise DP_EPSILON_TOTAL and the same code automatically lets
real motion through - nothing here is hardcoded to the current epsilon.

`idle_animation()` is provided separately and is explicitly DATA-INDEPENDENT: it never reads the
released scalars, so it carries ZERO privacy cost and cannot leak identity. It exists so the
renderer can show a living face without pretending noise is expression.

PRIVACY NOTE: this module only ever reads the 12 DP'd scalars. It never touches landmarks, face
crops, or any per-subject geometry. Its output modulates EXPRESSION and HEAD POSE only; face
SHAPE must remain the shared canonical template, identical for every person.

Reproduce the measurements:  python face_signal_filter.py --selftest
"""
from __future__ import annotations

import json
import math
from dataclasses import dataclass, field

import numpy as np

# ---------------------------------------------------------------------------------------------
# PUBLIC DP constants, mirrored from tier1_raspberry_pi5/config.py. These are PUBLIC by design
# (the mechanism is disclosed), so reconstructing the noise scale from them costs nothing.
# ---------------------------------------------------------------------------------------------
CHANNELS = ("mouth_open", "mouth_width", "smile", "eyeR_open", "eyeL_open", "browR", "browL",
            "yaw", "pitch", "roll", "gazeX", "gazeY")
CH_BOUNDS = [(0.0, 1.0), (0.0, 1.5), (-0.5, 0.5), (0.0, 0.5), (0.0, 0.5), (0.0, 0.6), (0.0, 0.6),
             (-1.5, 1.5), (-1.5, 1.5), (-1.0, 1.0), (-0.5, 0.5), (-0.5, 0.5)]
DP_EPS_PER_CH = [0.316, 0.316, 0.316, 0.316, 0.316, 0.316, 0.316,
                 0.20, 0.20, 0.15, 0.12, 0.12]
DP_QUANTIZE_BITS = 5
DP_WINDOW_S = 0.47

# Neutral / rest value per channel - where the face sits when nothing is known. Mid-open eyes,
# closed relaxed mouth, level head, centred gaze. These are RENDER DEFAULTS, not measurements.
NEUTRAL = [0.05, 0.55, 0.00, 0.25, 0.25, 0.28, 0.28, 0.0, 0.0, 0.0, 0.0, 0.0]

# Effective smoothing length per channel, in FRAMES-AT-10-fps-equivalent, scaled by fps at use.
# Head pose is slow and can take a long window; blinks are ~2-frame events so a long window would
# erase them (they are unrecoverable anyway - E6). These are DESIGN CHOICES from the render
# requirement (stable neck), not fitted to data.
W_EFF_S = [0.50, 0.50, 0.50, 0.20, 0.20, 0.50, 0.50, 1.50, 1.50, 1.50, 0.50, 0.50]

# Max angular rate for the pose channels, rad/s. A comfortable head turn is well under this;
# the cap exists purely to forbid the snap the user complained about ("sharp neck and face
# tilts"). DESIGN CHOICE, not measured.
POSE_MAX_RATE = {7: 1.2, 8: 1.0, 9: 0.8}
# Critically-damped follower time constant, seconds, per channel group. DESIGN CHOICE.
TAU_POSE, TAU_EXPR = 0.25, 0.12

# Prior signal sd as a fraction of each channel's admissible range. range/6 puts +-3 sigma at the
# channel bounds. DESIGN CHOICE (a prior), stated explicitly because it sets the shrinkage gain.
PRIOR_SD_FRAC = 1.0 / 6.0

# Below this shrinkage gain a channel is declared untrusted: the renderer must not animate it
# from the data. At eps=3.0 EVERY channel lands below this (measured gains 0.00055-0.00459, E8).
TRUST_GAIN_MIN = 0.05


# ---------------------------------------------------------------------------------------------
# noise model
# ---------------------------------------------------------------------------------------------
def _window_amplification(w: int, T: int) -> float:
    """Exact copy of config._window_amplification - max L1 influence of one frame."""
    pad, T = int(w) // 2, max(1, int(T))
    if pad == 0:
        return 1.0
    cnt = [min(t + pad, T - 1) - max(t - pad, 0) + 1 for t in range(T)]
    return max(sum(1.0 / cnt[t] for t in range(max(0, i - pad), min(T, i + pad + 1)))
               for i in range(T))


def dp_window(fps: float) -> int:
    """DP_WINDOW as config._frames() computes it: round(DP_WINDOW_S * EMIT_FPS), min 1."""
    return max(1, int(round(DP_WINDOW_S * float(fps))))


def channel_noise_sd(ch: int, fps: float, n_frames: int,
                     eps_per_ch=None, person_level: bool = False) -> float:
    """Laplace noise sd the mechanism injected into channel `ch`. sd = sqrt(2) * b_c.

    Mirrors apply_dp() exactly: sens_c = tp * max((amp+q_extra)*range, SENSITIVITY_c*amp/AMP_BAKED).
    Verified against the released data: measured sd / this value has median 0.97 over N=36 (E1).
    """
    eps = list(eps_per_ch or DP_EPS_PER_CH)
    lo, hi = CH_BOUNDS[ch]
    rng_c = hi - lo
    w = dp_window(fps)
    T = max(1, int(n_frames))
    levels = max(2, 2 ** int(DP_QUANTIZE_BITS))
    a = _window_amplification(w, T)
    q_extra = min(T, w) / float(levels - 1)
    amp_baked = max(_window_amplification(w, t) for t in range(1, 3 * w + 2))
    sens_baked = (amp_baked + w / float(levels - 1)) * rng_c
    sens = max((a + q_extra) * rng_c, sens_baked * a / amp_baked)
    if person_level:
        sens *= float(T)
    return math.sqrt(2.0) * sens / max(eps[ch], 1e-9)


def out_of_bounds_fraction(scalars) -> np.ndarray:
    """Fraction of samples per channel lying OUTSIDE CH_BOUNDS.

    apply_dp clamps the signal to CH_BOUNDS *before* the Laplace draw, so any out-of-bounds
    sample is pure noise. Measured on the shipped releases: 0.932 / 0.920 / 0.927 overall for
    a10 / a15 / a30 (E4). A value near 0.9+ means "this release carries no usable signal";
    a value near 0 would mean the noise is small relative to the range.
    """
    A = np.asarray(scalars, float)
    A = A.reshape(-1, len(CHANNELS))
    lo = np.array([b[0] for b in CH_BOUNDS])
    hi = np.array([b[1] for b in CH_BOUNDS])
    return ((A < lo) | (A > hi)).mean(axis=0)


# ---------------------------------------------------------------------------------------------
# smoothers
# ---------------------------------------------------------------------------------------------
def rts_smooth(x: np.ndarray, r_meas: float, q_proc: float) -> np.ndarray:
    """Constant-velocity Kalman filter + Rauch-Tung-Striebel backward smoother.

    r_meas is the KNOWN measurement variance (we know it exactly - disclosed mechanism), q_proc
    the process-noise intensity. Measured on real a10 yaw (N=100): cuts per-frame jitter from
    34.15 to 0.073 rad/frame. NOTE it does NOT bound the output - see E7; the gate does that.
    """
    x = np.asarray(x, float)
    n = x.size
    if n == 0:
        return x.copy()
    if n == 1:
        return x.copy()
    F = np.array([[1.0, 1.0], [0.0, 1.0]])
    H = np.array([[1.0, 0.0]])
    Q = float(q_proc) * np.array([[1 / 3.0, 1 / 2.0], [1 / 2.0, 1.0]])
    r_meas = max(float(r_meas), 1e-12)
    xf = np.zeros((n, 2)); Pf = np.zeros((n, 2, 2))
    xp = np.zeros((n, 2)); Pp = np.zeros((n, 2, 2))
    xk = np.array([x[0], 0.0]); Pk = np.eye(2) * r_meas * 10.0
    for t in range(n):
        if t > 0:
            xk = F @ xk
            Pk = F @ Pk @ F.T + Q
        xp[t], Pp[t] = xk, Pk
        S = float((H @ Pk @ H.T).item()) + r_meas
        K = (Pk @ H.T) / S
        xk = xk + (K * (x[t] - float((H @ xk).item()))).ravel()
        Pk = Pk - K @ H @ Pk
        xf[t], Pf[t] = xk, Pk
    xs = xf.copy()
    for t in range(n - 2, -1, -1):
        try:
            G = Pf[t] @ F.T @ np.linalg.inv(Pp[t + 1])
        except np.linalg.LinAlgError:
            continue
        xs[t] = xf[t] + G @ (xs[t + 1] - xp[t + 1])
    return xs[:, 0]


def gaussian_smooth(x: np.ndarray, sigma: float) -> np.ndarray:
    """Edge-padded Gaussian low-pass. Kept because it is the cheapest sane fallback."""
    x = np.asarray(x, float)
    if sigma <= 0 or x.size < 2:
        return x.copy()
    r = int(math.ceil(3.0 * sigma))
    k = np.exp(-0.5 * (np.arange(-r, r + 1) / sigma) ** 2)
    k /= k.sum()
    return np.convolve(np.pad(x, (r, r), mode="edge"), k, mode="valid")


def slew_and_damp(x: np.ndarray, fps: float, max_rate: float | None, tau: float) -> np.ndarray:
    """Rate-limit then critically-damp. This is the anti-'sharp tilt' stage.

    max_rate in units/second (None = no limit); tau the follower time constant in seconds.
    Second-order critically-damped follower: no overshoot, C1-continuous output, so the neck
    never snaps. Runs forward only, so it adds ~tau of lag by construction.
    """
    x = np.asarray(x, float)
    n = x.size
    if n == 0:
        return x.copy()
    dt = 1.0 / max(float(fps), 1e-6)
    y = x.copy()
    if max_rate is not None and max_rate > 0:
        step = float(max_rate) * dt
        out = np.empty(n)
        out[0] = y[0]
        for t in range(1, n):
            out[t] = out[t - 1] + float(np.clip(y[t] - out[t - 1], -step, step))
        y = out
    tau = max(float(tau), 1e-6)
    omega = 1.0 / tau
    pos, vel = float(y[0]), 0.0
    out = np.empty(n)
    for t in range(n):
        acc = omega * omega * (y[t] - pos) - 2.0 * omega * vel
        vel += acc * dt
        pos += vel * dt
        out[t] = pos
    return out


# ---------------------------------------------------------------------------------------------
# the filter
# ---------------------------------------------------------------------------------------------
@dataclass
class ChannelReport:
    """Per-channel diagnosis. `trust` is what the renderer should branch on."""
    name: str
    noise_sd: float          # known injected Laplace sd
    obs_sd: float            # sd of the released series
    signal_var_hat: float    # max(0, var_obs - var_noise), method-of-moments
    signal_sd_hat: float
    gain: float              # shrinkage weight actually applied to the data
    trust: bool              # gain >= TRUST_GAIN_MIN
    frac_out_of_bounds: float
    in_jitter: float         # mean |diff| of the released series
    out_jitter: float        # mean |diff| of the filtered output


@dataclass
class FilterResult:
    values: np.ndarray                        # [T, P, 12] filtered, clamped to CH_BOUNDS
    reports: list = field(default_factory=list)   # list[list[ChannelReport]], per person slot
    any_trusted: bool = False

    def trusted_channels(self):
        out = set()
        for per in self.reports:
            for c, r in enumerate(per):
                if r.trust:
                    out.add(c)
        return sorted(out)


def _to_array(scalars):
    """face_scalars.json -> [T, P, 12] float array, padding ragged frames with NaN."""
    if isinstance(scalars, np.ndarray) and scalars.ndim == 3:
        return scalars.astype(float)
    frames = list(scalars)
    P = max((len(f) for f in frames), default=0)
    T = len(frames)
    A = np.full((T, max(P, 1), len(CHANNELS)), np.nan)
    for t, fr in enumerate(frames):
        for p, person in enumerate(fr):
            v = list(person)[:len(CHANNELS)]
            A[t, p, :len(v)] = v
    return A


def filter_face_scalars(scalars, fps: float, eps_per_ch=None, person_level: bool = False,
                        prior_sd_frac: float = PRIOR_SD_FRAC,
                        trust_gain_min: float = TRUST_GAIN_MIN,
                        neutral=None) -> FilterResult:
    """DP-noised face scalars -> smooth, stable, bounded trajectory + a per-channel trust report.

    Args:
      scalars: the parsed face_scalars.json, [frames][persons][12], OR an [T,P,12] array.
      fps:     the EMIT_FPS the clip was released at (10/15/30). Sets DP_WINDOW and all durations.
      eps_per_ch: override DP_EPS_PER_CH if the release used a different budget.
      person_level: True if the release used DP_PERSON_LEVEL (noise ~T_p x larger).
      prior_sd_frac / trust_gain_min / neutral: see the module constants.

    Returns FilterResult. `values` is ALWAYS inside CH_BOUNDS and always C1-smooth.

    At the shipped eps=3.0 every channel comes back trust=False with gain ~0.001-0.005 and the
    output is the neutral pose (measured: E5/E8). That is the correct answer, not a bug - see the
    module docstring. The renderer should check `trusted_channels()` and, when it is empty, drive
    the face from `idle_animation()` instead of pretending the noise is expression.
    """
    A = _to_array(scalars)
    T, P, C = A.shape
    neutral = list(neutral if neutral is not None else NEUTRAL)
    out = np.empty_like(A)
    reports = []
    any_trusted = False

    for p in range(P):
        per_ch = []
        for c in range(C):
            lo, hi = CH_BOUNDS[c]
            rng_c = hi - lo
            x = A[:, p, c]
            present = np.isfinite(x)
            npres = int(present.sum())
            if npres == 0:
                out[:, p, c] = neutral[c]
                per_ch.append(ChannelReport(CHANNELS[c], float("nan"), float("nan"), 0.0, 0.0,
                                            0.0, False, float("nan"), float("nan"), 0.0))
                continue
            xp_ = x[present]

            # (1) known noise variance from the PUBLIC mechanism
            sd_n = channel_noise_sd(c, fps, npres, eps_per_ch, person_level)
            var_n = sd_n * sd_n

            # (2) unbiased method-of-moments signal variance, floored at 0 and capped at the
            #     largest variance the channel can physically hold (the clamp means the true
            #     signal lives inside [lo,hi], so var_s <= (range/2)^2).
            var_obs = float(np.var(xp_))
            var_s = min(max(var_obs - var_n, 0.0), (rng_c / 2.0) ** 2)
            #     blend with the stated prior so a short slot cannot produce a wild estimate
            var_prior = (prior_sd_frac * rng_c) ** 2
            var_s = min(var_s, var_prior) if var_s > 0 else 0.0

            # (3) shrinkage gain against the noise floor AFTER the channel's own smoothing
            w_eff = max(1.0, W_EFF_S[c] * float(fps))
            gain = var_s / (var_s + var_n / w_eff) if (var_s + var_n) > 0 else 0.0
            #     the prior-based gain is the ceiling: never trust the data more than the prior
            #     says is possible.
            gain_ceiling = var_prior / (var_prior + var_n / w_eff)
            gain = min(gain, gain_ceiling)
            trust = bool(gain >= trust_gain_min)
            any_trusted = any_trusted or trust

            # (4) RTS smoother with R = the KNOWN noise variance
            #     q chosen so the smoother's own bandwidth matches w_eff.
            q_proc = var_n / (w_eff ** 4)
            xs_p = rts_smooth(xp_, var_n, q_proc)

            # (5) shrink toward neutral
            y_p = neutral[c] + gain * (xs_p - neutral[c])

            # scatter back onto the full timeline, holding the last value across gaps
            y = np.full(T, float(neutral[c]))
            y[present] = y_p
            if npres < T:
                idx = np.where(present)[0]
                last = neutral[c]
                for t in range(T):
                    if present[t]:
                        last = y[t]
                    else:
                        y[t] = last
                if idx.size:
                    y[:idx[0]] = y[idx[0]]

            # (6) clamp to the physically admissible range
            y = np.clip(y, lo, hi)

            # (7) rate limit + critical damping - the anti-snap stage
            y = slew_and_damp(y, fps, POSE_MAX_RATE.get(c), TAU_POSE if c in (7, 8, 9) else TAU_EXPR)
            y = np.clip(y, lo, hi)
            out[:, p, c] = y

            oob = float(np.mean((xp_ < lo) | (xp_ > hi)))
            per_ch.append(ChannelReport(
                CHANNELS[c], sd_n, float(np.std(xp_)), var_s, math.sqrt(var_s), gain, trust, oob,
                float(np.mean(np.abs(np.diff(xp_)))) if npres > 1 else 0.0,
                float(np.mean(np.abs(np.diff(y)))) if T > 1 else 0.0))
        reports.append(per_ch)
    return FilterResult(values=out, reports=reports, any_trusted=any_trusted)


# ---------------------------------------------------------------------------------------------
# data-independent idle animation
# ---------------------------------------------------------------------------------------------
def idle_animation(n_frames: int, fps: float, seed: int = 0, amount: float = 1.0) -> np.ndarray:
    """A living, IDENTITY-FREE, DATA-INDEPENDENT idle trajectory -> [n_frames, 12].

    It NEVER reads the released scalars, so it costs zero privacy budget and cannot leak identity
    or expression: it is the same generator for every subject, exactly like the canonical face
    template. `seed` only varies the phase, so two clips do not look mechanically identical; it
    must NOT be derived from anything subject-specific.

    Why this exists: measurements E3/E5/E6 say the released scalars carry no recoverable
    expression at eps=3.0, and E5 says emitting a constant is 3.5-6.0x better than any filter of
    them. A frozen face looks broken, so the honest option is a face that is visibly ALIVE but
    visibly NOT a reconstruction of the subject. Amplitudes below are deliberately small and slow.

    Blinks here are procedural (~1 every 4 s). This is NOT the subject's blink pattern - it cannot
    be, per E6 (14.7% detection vs 14.9% chance).
    """
    n = int(max(0, n_frames))
    if n == 0:
        return np.zeros((0, len(CHANNELS)))
    t = np.arange(n) / max(float(fps), 1e-6)
    rs = np.random.default_rng(int(seed))
    ph = rs.uniform(0, 2 * np.pi, 8)
    a = float(amount)
    y = np.tile(np.array(NEUTRAL, float), (n, 1))
    # slow head drift, three incommensurate periods so it never visibly loops
    y[:, 7] += a * 0.16 * (np.sin(2 * np.pi * 0.061 * t + ph[0]) + 0.5 * np.sin(2 * np.pi * 0.107 * t + ph[1]))
    y[:, 8] += a * 0.10 * (np.sin(2 * np.pi * 0.048 * t + ph[2]) + 0.5 * np.sin(2 * np.pi * 0.091 * t + ph[3]))
    y[:, 9] += a * 0.06 * np.sin(2 * np.pi * 0.037 * t + ph[4])
    y[:, 10] += a * 0.09 * np.sin(2 * np.pi * 0.13 * t + ph[5])
    y[:, 11] += a * 0.05 * np.sin(2 * np.pi * 0.11 * t + ph[6])
    y[:, 5] += a * 0.03 * np.sin(2 * np.pi * 0.08 * t + ph[7])
    y[:, 6] = y[:, 5]
    # procedural blinks: ~1 per 4 s, 0.12 s closure
    dur = max(1, int(round(0.12 * fps)))
    k = 0.0
    while k < t[-1] + 1e-9:
        k += rs.uniform(2.5, 5.5)
        i0 = int(round(k * fps))
        for j in range(dur):
            if 0 <= i0 + j < n:
                y[i0 + j, 3] = 0.02
                y[i0 + j, 4] = 0.02
    for c in (3, 4, 5, 6, 7, 8, 9, 10, 11):
        y[:, c] = gaussian_smooth(y[:, c], max(0.6, 0.05 * fps))
    lo = np.array([b[0] for b in CH_BOUNDS]); hi = np.array([b[1] for b in CH_BOUNDS])
    return np.clip(y, lo, hi)


# ---------------------------------------------------------------------------------------------
def _selftest():
    import os
    base = "_e2e/fps_ladder_20260726/%s/tier1_out/face_scalars.json"
    print("channel        fps   noise_sd  obs_sd  ratio   |rho1|   %OOB   gain     trust"
          "   in_jit   out_jit")
    for tag, fps in (("a10", 10), ("a15", 15), ("a30", 30)):
        path = base % tag
        if not os.path.exists(path):
            print("  missing", path)
            continue
        data = json.load(open(path))
        res = filter_face_scalars(data, fps)
        A = _to_array(data)[:, 0, :]
        for c in range(12):
            r = res.reports[0][c]
            x = A[:, c] - np.nanmean(A[:, c])
            rho1 = float(np.nansum(x[:-1] * x[1:]) / max(np.nansum(x * x), 1e-12))
            print(f"{CHANNELS[c]:<13}{fps:>4}{r.noise_sd:>10.2f}{r.obs_sd:>8.2f}"
                  f"{r.obs_sd / r.noise_sd:>7.2f}{abs(rho1):>9.3f}{r.frac_out_of_bounds * 100:>7.1f}"
                  f"{r.gain:>8.5f}{str(r.trust):>8}{r.in_jitter:>9.2f}{r.out_jitter:>10.4f}")
        print(f"  -> trusted channels at {tag}: {res.trusted_channels() or 'NONE'}   "
              f"any_trusted={res.any_trusted}")
        print()


if __name__ == "__main__":
    import sys
    if "--selftest" in sys.argv:
        _selftest()
    else:
        print(__doc__)
