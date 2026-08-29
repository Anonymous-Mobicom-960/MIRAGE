#!/usr/bin/env python3
"""SYNTHETIC HUMAN SILHOUETTE - emit a DEFAULT BODY instead of the subject's own outline.

WHAT THIS IS. Today Tier-1 emits the real person's silhouette, perturbed (`MASK_SHAPE_MODE`).
Every perturbation mode starts from the subject's own boundary, so their build stays recoverable
underneath it (§A.6e: the shipped `displace 0.25` still scores 55.03 % NM). This module instead
DRAWS A NEW BODY: a canonical, population-constant human figure, posed by the person's REAL
joints, sized to contain them, and unioned so it is a strict superset of the real mask. The
attacker downstream sees the same build for every subject; only motion survives.

It is the mask analogue of `pose_anon_edge._ANATOMICAL_OVER_SHOULDER`, and it uses that module's
scale convention deliberately: every dimension here is a fraction of BIACROMIAL (shoulder) WIDTH,
the one landmark pair measured to be reliable on our framings (confidence 1.000, never off-frame).

WHY IT IS DRAWN ON THE *REAL* JOINTS. A 2026-08-01 probe built the same capsules on the
ANONYMISED skeleton and never reached zero leak at any dilation ≤ 80 px (commit d334ec3): the
anonymiser retargets bone lengths and adds 14°/10° of angle noise, so its limbs simply are not
where the person's limbs are. Real joints keep MOTION exact (the requirement) and cost nothing in
privacy that the mask does not already disclose - the emitted outline carries canonical WIDTHS,
canonical head, canonical hands/feet, and joint POSITIONS, which is what "same build, real
motion" means. The anonymised skeleton is still what leaves in pose.json; the two are independent
channels and this module never touches the emitted pose.

WHAT LEAKS, STATED PLAINLY.
  * overall SIZE - one scalar per clip, log-quantised into ~7 % buckets (`SYNTH_SCALE_QUANT`).
    Size is already disclosed to the recipient by the mask's extent; quantising makes it k-same.
  * the COVERAGE factor alpha - one scalar per clip (running max, quantised to 0.05 steps). A
    subject who is bulkier than the canonical build pushes it up a bucket.
  * joint positions (= motion), by design.
  * BACKSTOP PIXELS: anything the synthetic body still fails to cover (a rucksack, an umbrella,
    a coat that is not body-shaped) is OR-ed back in, because a §2 reveal is never acceptable.
    Those pixels ARE the subject's own outline. The union is counted and reported per clip
    (`synth_body` block in pose.json) - if it is not ~0, the defence is not doing its job on that
    clip and the number says so rather than hiding it.

§2 GUARANTEE. `apply()` returns `synth | real`. The output is a superset of its input on every
frame by construction, so every §2 property the pipeline already has is preserved unchanged, and
`mask_mitigate`'s own `| cur` union downstream becomes a no-op rather than a re-leak.

NO PIXEL CONSTANTS. Every length is a fraction of the measured shoulder width, so the model is
the same proportion of the person at 64x64 and at native 1264².
"""
import math

import cv2
import numpy as np

# ---------------------------------------------------------------- canonical build (population)
# Half-widths / lengths as fractions of BIACROMIAL WIDTH S (= |L shoulder - R shoulder|).
# Sources are stature-normalised anthropometry re-expressed over biacromial = 0.234 H, i.e.
# H = 4.27 S, the same conversion `pose_anon_edge._ANATOMICAL_OVER_SHOULDER` documents.
# These are POPULATION CONSTANTS. Nothing here is ever fitted to a subject or to a clip.
CANON = dict(
    sh_half     = 0.560,   # deltoid outer edge, wider than the shoulder JOINT (0.5) - the joint
                           #   sits inside the arm, the silhouette does not.
    waist_half  = 0.430,   # narrowest trunk, at WAIST_AT down the spine
    waist_at    = 0.62,    # fraction of mid-shoulder -> mid-hip
    hip_half    = 0.470,   # widest pelvis (> the 0.360 hip JOINT half-span, same reason)
    crotch_at   = 1.10,    # trunk bottom, past the hip joints toward the crotch
    # NECK/TRAPEZIUS (2026-08-01, measured). A thin cylindrical neck was the single largest
    # residual on B_atrium: seen from behind, the slope from the head down to the shoulders is
    # filled by the trapezius and by hair, and a 0.185 S neck left two red wedges either side of
    # it. Drawn as a taper from the head down to a WIDE base at the shoulder line - universal
    # anatomy, so it is a population constant, and it is exactly the region where a per-subject
    # outline would otherwise have shown hair length.
    neck_half   = 0.240,   # at the head end
    neck_base   = 0.400,   # at the shoulder line
    # HEAD (2026-08-01 v2, user direction: "the head is too big of a blob, make it slightly near
    # to the person, possibly destroying the real silhouette but still keeping the person's
    # gender"). v1 drew a fixed 0.345/0.375 S ellipse and then the coverage margin puffed it, which
    # on a close framing reads as a balloon. v2 makes the canonical ellipse a FLOOR and lets the
    # outline follow the subject's own head+hair region, radially low-passed to `head_lp_keep`
    # harmonics about the head centre.
    # 🔴 THIS IS A DELIBERATE PRIVACY REDUCTION, made on the user's explicit instruction. Head
    # outline and hair length are identifying and this hands a coarse version of them back. What it
    # buys is the gender cue - long hair / short hair / a bun survive 3 harmonics, the fine
    # boundary does not. `head_mode="canon"` restores v1's identity-free head.
    head_rx      = 0.300,  # canonical FLOOR (was 0.345): head breadth 0.155 H = 0.66 S
    head_ry      = 0.320,  # canonical FLOOR (was 0.375)
    head_lp_keep = 3,      # angular harmonics kept of the real head/hair profile
    head_band    = 2.10,   # hair region searched this far from the head centre, in units of head_ry
    head_up     = 0.660,   # mid-shoulder -> head CENTRE along the torso axis
    face_up     = 0.190,   # face-keypoint centroid -> head centre, along the torso axis
    uarm_r0     = 0.170,   # shoulder end
    uarm_r1     = 0.135,   # elbow end
    farm_r0     = 0.130,
    farm_r1     = 0.095,   # wrist
    hand_r      = 0.150,   # hand disc at the wrist ...
    hand_ext    = 0.260,   # ... pushed this far past the wrist along the forearm
    thigh_r0    = 0.230,   # hip end
    thigh_r1    = 0.155,   # knee
    shin_r0     = 0.145,
    shin_r1     = 0.100,   # ankle
    foot_rx     = 0.400,   # foot ellipse at the ankle: SYMMETRIC left/right on purpose - which way
    foot_ry     = 0.150,   #   the foot points is a gait cue, so it is not drawn.
    smooth      = 0.090,   # closing radius: joins the parts into one organic outline
    # ---- YAW (2026-08-01, after the first measured run). A trunk drawn at full frontal width on
    # every frame is wrong in both directions: it costs 2.56x the real area when the subject turns
    # to profile (measured, A_corridor f31), and it means the emitted silhouette does NOT rotate
    # when the person does - one of the three things this design is supposed to preserve.
    # Model the trunk cross-section as an ellipse of breadth 1 and depth `torso_depth`, viewed at
    # yaw t:  width(t) = sqrt(cos^2 t + depth^2 sin^2 t), with cos t read off the OBSERVED
    # shoulder span / the clip's own scale. That is a MOTION quantity (it is already in the pose
    # sticks), not a build quantity - the breadth and depth it scales are both canonical.
    torso_depth = 0.620,   # chest depth / biacromial breadth
    head_depth  = 1.250,   # head is LONGER than it is wide: 0.195 H deep vs 0.155 H broad
    yaw_min_c   = 0.000,   # no floor needed - the ellipse model already bottoms out at `depth`
)

# COCO-17 body indices used (the wholebody 133 layout shares them).
NOSE, LEYE, REYE, LEAR, REAR = 0, 1, 2, 3, 4
LSH, RSH, LEL, REL, LWR, RWR = 5, 6, 7, 8, 9, 10
LHIP, RHIP, LKN, RKN, LAN, RAN = 11, 12, 13, 14, 15, 16


def _ok(kp, sc, i, thr):
    return (i < len(kp) and (sc is None or sc[i] >= thr)
            and (abs(kp[i][0]) + abs(kp[i][1])) > 0)


def _mid(a, b):
    return (a + b) * 0.5


def _taper(m, p, q, r0, r1):
    """Filled tapered capsule from p (radius r0) to q (radius r1) - a cone frustum plus its two
    end discs. cv2.line only does constant thickness, and a constant-thickness limb reads as a
    tube; real limbs taper, and the taper is most of what makes the outline look human."""
    p = np.asarray(p, np.float32); q = np.asarray(q, np.float32)
    d = q - p
    L = float(np.hypot(d[0], d[1]))
    if L < 1e-3:
        cv2.circle(m, (int(round(p[0])), int(round(p[1]))), int(max(1, r0)), 1, -1)
        return
    n = np.array([-d[1] / L, d[0] / L], np.float32)
    poly = np.array([p + n * r0, q + n * r1, q - n * r1, p - n * r0], np.float32)
    cv2.fillConvexPoly(m, np.round(poly).astype(np.int32), 1)
    cv2.circle(m, (int(round(p[0])), int(round(p[1]))), int(max(1, round(r0))), 1, -1)
    cv2.circle(m, (int(round(q[0])), int(round(q[1]))), int(max(1, round(r1))), 1, -1)


def body_scale(kp, sc, thr=0.35):
    """Per-frame shoulder-width estimate, with rotation-robust fallbacks.

    Shoulder width COLLAPSES toward 0 when the subject turns to profile, so a raw per-frame value
    is unusable as a size. Take the largest of three mutually-redundant estimates and let the
    caller take a robust statistic over the clip:
        biacromial directly | spine / 1.231 | bihip / 0.720   (ratios from _ANATOMICAL_OVER_SHOULDER)
    Returns 0.0 when nothing usable is visible."""
    kp = np.asarray(kp, np.float32)
    c = []
    if _ok(kp, sc, LSH, thr) and _ok(kp, sc, RSH, thr):
        c.append(float(np.linalg.norm(kp[LSH] - kp[RSH])))
    if (_ok(kp, sc, LSH, thr) and _ok(kp, sc, RSH, thr)
            and _ok(kp, sc, LHIP, thr) and _ok(kp, sc, RHIP, thr)):
        c.append(float(np.linalg.norm(_mid(kp[LSH], kp[RSH]) - _mid(kp[LHIP], kp[RHIP]))) / 1.231)
    if _ok(kp, sc, LHIP, thr) and _ok(kp, sc, RHIP, thr):
        c.append(float(np.linalg.norm(kp[LHIP] - kp[RHIP])) / 0.720)
    return max(c) if c else 0.0


def radial_lp(mask, keep=4, bins=180):
    """Outward-only radial low-pass, per connected component - the POSE-FREE fallback.

    WHY IT IS HERE. The body can only be drawn where the pose model succeeds. On the 108-clip
    corpus it did not: several clips annotate a DISTANT person behind foliage, RTMPose returns
    every body-joint score at 0.10-0.21, no figure is drawn at all, and the §2 backstop then
    emits the subject's own outline untouched - 73 % of the mask on `p11_c02_head`, measured.
    Passing the real outline through is the one outcome this whole design exists to prevent, so
    when there is no body the mask still gets a defence: keep only the DC term plus `keep`
    angular harmonics of r(theta) about each component's centroid, and take the max with the
    original so the op is outward-only and §2 holds by construction.

    Same mechanism as `mirage_tier1._shape_polys("radiallp")`, reimplemented here (~20 lines) so
    that `synth_body` stays importable on its own and cannot create an import cycle with the
    Tier-1 module that imports IT."""
    out = np.zeros_like(mask)
    cnts, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not cnts:
        return mask
    polys = []
    for c in cnts:
        p = c.reshape(-1, 2).astype(np.float32)
        if p.shape[0] < 8:
            polys.append(c)
            continue
        ctr = p.mean(0)
        v = p - ctr
        r = np.linalg.norm(v, axis=1)
        a = (np.arctan2(v[:, 1], v[:, 0]) + 2 * np.pi) % (2 * np.pi)
        b = np.clip((a / (2 * np.pi) * bins).astype(np.int32), 0, bins - 1)
        rb = np.zeros(bins, np.float32)
        np.maximum.at(rb, b, r.astype(np.float32))
        if not rb.any():
            polys.append(c)
            continue
        idx = np.nonzero(rb)[0]                      # fill empty bins circularly
        rb = np.interp(np.arange(bins), idx, rb[idx], period=bins).astype(np.float32)
        F = np.fft.rfft(rb)
        F[int(keep) + 1:] = 0
        rs = np.maximum(rb, np.fft.irfft(F, n=bins).astype(np.float32))   # OUTWARD-ONLY
        th = (np.arange(bins) + 0.5) / bins * 2 * np.pi
        polys.append(np.round(np.stack([ctr[0] + rs * np.cos(th),
                                        ctr[1] + rs * np.sin(th)], 1)).astype(np.int32))
    cv2.fillPoly(out, [q.reshape(-1, 1, 2) for q in polys], 1)
    return out | mask


def clip_jitter(seed, canon=CANON, amp=0.06, asym=0.5):
    """A per-CLIP wobble of the canonical build.

    WHY (user direction 2026-08-01, "add slight randomness"). A build that is identical for every
    clip is a template the attacker can subtract: they know the figure exactly, so anything that
    is NOT the template is signal. Redrawing the proportions per clip from a fixed distribution
    removes that. The important property is that the draw is keyed to the CLIP, never to identity
    or content - the same subject filmed twice gets two different builds, which is what breaks
    linkage, and two different subjects can land on the same build. It is the mask analogue of the
    per-clip pose reseed, which measured +1-3 pp against a gallery-adapting attacker.

    `amp` is the half-width of the multiplicative wobble (0.06 = +-6 %). `asym` scales a second,
    smaller draw applied to the LEFT limbs only - a real body is not perfectly symmetric, and a
    perfectly symmetric one is itself a tell that the outline is synthetic."""
    rng = np.random.default_rng(int(seed) & 0x7FFFFFFF)
    out = dict(canon)
    for k in ("sh_half", "waist_half", "hip_half", "neck_half", "neck_base", "head_rx", "head_ry",
              "uarm_r0", "uarm_r1", "farm_r0", "farm_r1", "hand_r", "hand_ext",
              "thigh_r0", "thigh_r1", "shin_r0", "shin_r1", "foot_rx", "foot_ry"):
        out[k] = float(canon[k]) * float(1.0 + amp * rng.uniform(-1.0, 1.0))
    out["waist_at"] = float(canon["waist_at"]) * float(1.0 + 0.5 * amp * rng.uniform(-1.0, 1.0))
    out["_asym"] = float(1.0 + asym * amp * rng.uniform(-1.0, 1.0))
    return out


def _head_profile(real, hc, rx, ry, keep, band, bins=72):
    """Outline of the subject's own head+hair about `hc`, low-passed to `keep` harmonics.

    Returns a (bins,) radius profile, or None when there is nothing to read. The region searched
    is a disc of `band` head-heights about the head centre, so shoulders and torso do not pull the
    profile down the body; the radius per angular bin is the FARTHEST real pixel in that bin, and
    the profile is floored at the canonical ellipse so the head can never come out smaller than
    the population head."""
    H, W = real.shape[:2]
    R = float(band * max(rx, ry))
    x0, x1 = int(max(0, hc[0] - R)), int(min(W, hc[0] + R + 1))
    y0, y1 = int(max(0, hc[1] - R)), int(min(H, hc[1] + R + 1))
    if x1 - x0 < 4 or y1 - y0 < 4:
        return None
    sub = real[y0:y1, x0:x1]
    ys, xs = np.nonzero(sub)
    if ys.size < 16:
        return None
    dx = (xs + x0) - hc[0]
    dy = (ys + y0) - hc[1]
    r = np.hypot(dx, dy)
    inside = r <= R
    if inside.sum() < 16:
        return None
    dx, dy, r = dx[inside], dy[inside], r[inside]
    b = ((np.arctan2(dy, dx) + 2 * np.pi) % (2 * np.pi)) / (2 * np.pi) * bins
    b = np.clip(b.astype(np.int32), 0, bins - 1)
    prof = np.zeros(bins, np.float32)
    np.maximum.at(prof, b, r.astype(np.float32))
    # fill empty bins from the canonical ellipse so the FFT never sees a hole
    th = (np.arange(bins) + 0.5) / bins * 2 * np.pi
    ell = 1.0 / np.sqrt((np.cos(th) / max(1e-3, rx)) ** 2 + (np.sin(th) / max(1e-3, ry)) ** 2)
    prof = np.where(prof > 0, prof, ell)
    F = np.fft.rfft(prof)
    F[int(keep) + 1:] = 0
    lp = np.fft.irfft(F, n=bins).astype(np.float32)
    return np.maximum(lp, ell)                      # canonical head is the FLOOR


def quantise_scale(s, q=0.10):
    """Log-ladder quantisation of the body scale, anchored ABSOLUTELY (at 1 px) so the bucket a
    subject lands in cannot be reverse-engineered from a per-subject reference. q=0.10 in log2
    means ~7.2 % buckets."""
    if s <= 1.0:
        return 0.0
    return float(2.0 ** (round(math.log2(s) / q) * q))


def draw(kp, sc, H, W, S, alpha=1.0, thr=0.35, canon=CANON, real=None, head_mode="hair"):
    """Rasterise ONE canonical body at this frame's joint positions.

    kp: (>=17, 2) native-coord keypoints (RAW detector output - see the module docstring).
    S : the clip's quantised body scale in px.  alpha: uniform width multiplier (coverage).
    Returns a uint8 {0,1} HxW mask, or None if the pose is too incomplete to place a body."""
    kp = np.asarray(kp, np.float32)
    if S <= 1.0:
        return None
    m = np.zeros((H, W), np.uint8)
    C = canon
    U = S * float(alpha)                       # every dimension below is a multiple of U

    have = lambda i: _ok(kp, sc, i, thr)
    sh = [i for i in (LSH, RSH) if have(i)]
    hp = [i for i in (LHIP, RHIP) if have(i)]
    if not sh:
        return None

    smid = np.mean([kp[i] for i in sh], axis=0)
    if hp:
        hmid = np.mean([kp[i] for i in hp], axis=0)
    else:
        # No hips (bust framing / occluded): synthesise them DOWN the image at the canonical
        # spine length. Better than skipping the trunk, which would leave the torso to the backstop.
        hmid = smid + np.array([0.0, 1.231 * U], np.float32)
    axis = smid - hmid
    an = float(np.linalg.norm(axis))
    axis = axis / an if an > 1e-3 else np.array([0.0, -1.0], np.float32)
    lat = np.array([-axis[1], axis[0]], np.float32)      # trunk lateral direction
    spine = max(an, 0.6 * U)

    # YAW. cos(t) from the OBSERVED shoulder span against this clip's scale; both shoulders must
    # be confident or we assume frontal (the safe, wider case).
    c = 1.0
    if len(sh) == 2:
        c = float(np.clip(np.linalg.norm(kp[LSH] - kp[RSH]) / max(1e-6, S), 0.0, 1.0))
    s2 = max(0.0, 1.0 - c * c)
    w_trunk = math.sqrt(c * c + (C["torso_depth"] ** 2) * s2)     # <=1: narrows in profile
    w_head = math.sqrt(c * c + (C["head_depth"] ** 2) * s2)       # >=1: the head is deeper

    # ---- TRUNK: one tapered polygon shoulders -> waist -> hips -> crotch. A quad between the
    # four joints would be far too narrow (joints sit INSIDE the body) and would show the
    # subject's own shoulder-to-hip taper; this shape is fixed for everyone.
    waist = smid - axis * (C["waist_at"] * spine)
    crotch = smid - axis * (C["crotch_at"] * spine)
    Ut = U * w_trunk                                  # trunk lateral scale, yaw-corrected
    trunk = [smid + lat * (C["sh_half"] * Ut), waist + lat * (C["waist_half"] * Ut),
             hmid + lat * (C["hip_half"] * Ut), crotch + lat * (C["hip_half"] * 0.92 * Ut),
             crotch - lat * (C["hip_half"] * 0.92 * Ut), hmid - lat * (C["hip_half"] * Ut),
             waist - lat * (C["waist_half"] * Ut), smid - lat * (C["sh_half"] * Ut)]
    cv2.fillPoly(m, [np.round(np.array(trunk, np.float32)).astype(np.int32)], 1)
    # rounded shoulder caps so the trunk corners are not square
    for i in sh:
        cv2.circle(m, tuple(np.round(kp[i]).astype(int)), int(max(2, C["uarm_r0"] * U)), 1, -1)

    # ---- HEAD + NECK. Placed from the FACE keypoints when they are confident (that is where the
    # head actually is), else purely from the torso axis. Either way its SIZE and SHAPE are
    # canonical: head outline and hair are strongly identifying, so drawing a default head is a
    # privacy feature, not a compromise.
    fc = [i for i in (NOSE, LEYE, REYE, LEAR, REAR) if have(i)]
    if len(fc) >= 2:
        hc = np.mean([kp[i] for i in fc], axis=0) + axis * (C["face_up"] * U)
    else:
        hc = smid + axis * (C["head_up"] * U)
    ang = math.degrees(math.atan2(axis[0], -axis[1]))    # head tilts with the trunk
    _taper(m, smid, hc, C["neck_base"] * U * w_trunk, C["neck_half"] * U)
    hrx, hry = C["head_rx"] * U * w_head, C["head_ry"] * U
    prof = None
    if head_mode == "hair" and real is not None:
        prof = _head_profile(real, hc, hrx, hry, C.get("head_lp_keep", 3),
                             C.get("head_band", 2.10))
    if prof is None:
        cv2.ellipse(m, (int(round(hc[0])), int(round(hc[1]))),
                    (int(max(2, hrx)), int(max(2, hry))), ang, 0, 360, 1, -1)
    else:
        n = prof.shape[0]
        th = (np.arange(n) + 0.5) / n * 2 * np.pi
        pts = np.stack([hc[0] + prof * np.cos(th), hc[1] + prof * np.sin(th)], 1)
        cv2.fillPoly(m, [np.round(pts).astype(np.int32)], 1)

    # ---- LIMBS. Each segment is drawn only if BOTH ends are confident; a missing segment falls
    # to the backstop rather than being invented somewhere wrong.
    asym = float(C.get("_asym", 1.0))          # per-clip left/right wobble (see clip_jitter)
    for a, b, r0, r1, sd in ((LSH, LEL, "uarm_r0", "uarm_r1", asym),
                             (RSH, REL, "uarm_r0", "uarm_r1", 1.0),
                             (LEL, LWR, "farm_r0", "farm_r1", asym),
                             (REL, RWR, "farm_r0", "farm_r1", 1.0),
                             (LHIP, LKN, "thigh_r0", "thigh_r1", asym),
                             (RHIP, RKN, "thigh_r0", "thigh_r1", 1.0),
                             (LKN, LAN, "shin_r0", "shin_r1", asym),
                             (RKN, RAN, "shin_r0", "shin_r1", 1.0)):
        if have(a) and have(b):
            _taper(m, kp[a], kp[b], C[r0] * U * sd, C[r1] * U * sd)

    # ---- LIMBS THAT LEAVE THE PICTURE (2026-08-01, measured). On a close framing the legs are
    # simply not in shot: no knee, no ankle, so nothing was drawn below the crotch - and the real
    # mask runs all the way to the frame edge. That single gap was the worst residual on
    # A_corridor (bottom decile, 91 k px, 49.7 px) and it was what dragged the global coverage
    # margin to 0.32 S, i.e. it inflated the WHOLE body to fix the bottom of the frame.
    # Continue the chain along the trunk axis at canonical length, but ONLY when the continuation
    # lands off-canvas - that is the "the limb is out of shot" case. A missing joint that would
    # land INSIDE the picture is a detection failure, not a framing one, and inventing a straight
    # limb there could put canonical geometry somewhere the person is not.
    off = lambda p: not (0 <= p[0] < W and 0 <= p[1] < H)
    # arms first: on a bust framing the elbows are below the frame and the arms hang beside the
    # trunk, so the real silhouette WIDENS toward the cut while a full-body trunk TAPERS to the
    # waist. That mismatch was B_atrium's whole residual (chest/waist/hip deciles, 47 px).
    for sh_i, el_i, wr_i in ((LSH, LEL, LWR), (RSH, REL, RWR)):
        if not have(sh_i) or have(el_i):
            continue
        elb = kp[sh_i] - axis * (0.795 * U)          # canonical upper arm, hanging at rest
        if off(elb):
            _taper(m, kp[sh_i], elb, C["uarm_r0"] * U, C["uarm_r1"] * U)
            wri = elb - axis * (0.624 * U)
            if not have(wr_i) and off(wri):
                _taper(m, elb, wri, C["farm_r0"] * U, C["farm_r1"] * U)
    for hip_i, kn_i, an_i, side in ((LHIP, LKN, LAN, +1.0), (RHIP, RKN, RAN, -1.0)):
        root = kp[hip_i] if have(hip_i) else (hmid + lat * (0.360 * U * w_trunk * side))
        knee = kp[kn_i] if have(kn_i) else root - axis * (1.047 * U)
        if not have(kn_i):
            if not off(knee):
                continue
            _taper(m, root, knee, C["thigh_r0"] * U, C["thigh_r1"] * U)
        ank = kp[an_i] if have(an_i) else knee - axis * (1.051 * U)
        if not have(an_i) and off(ank):
            _taper(m, knee, ank, C["shin_r0"] * U, C["shin_r1"] * U)

    # ---- HANDS: a disc pushed past the wrist along the forearm. The wrist keypoint is the joint,
    # the hand extends ~0.11 H beyond it and is a large part of what a limb-following mask misses.
    for el, wr in ((LEL, LWR), (REL, RWR)):
        if not have(wr):
            continue
        d = kp[wr] - kp[el] if have(el) else np.array([0.0, 1.0], np.float32)
        n = float(np.linalg.norm(d))
        d = d / n if n > 1e-3 else np.array([0.0, 1.0], np.float32)
        _taper(m, kp[wr], kp[wr] + d * (C["hand_ext"] * U), C["farm_r1"] * U, C["hand_r"] * U)

    # ---- FEET: a horizontally symmetric ellipse at the ankle. Deliberately NOT pointed along the
    # direction of travel - foot orientation is a gait cue and the ellipse covers either way.
    for an_i in (LAN, RAN):
        if have(an_i):
            cv2.ellipse(m, tuple(np.round(kp[an_i]).astype(int)),
                        (int(max(2, C["foot_rx"] * U)), int(max(2, C["foot_ry"] * U))),
                        0, 0, 360, 1, -1)

    # ---- ORGANIC FINISH: close the seams so the union reads as one body rather than assembled
    # parts (a lumpy outline is also an inpaint hazard - see the A20 invented-person failure).
    k = int(max(3, round(C["smooth"] * U)) | 1)
    m = cv2.morphologyEx(m, cv2.MORPH_CLOSE,
                         cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (k, k)))
    return m


class SynthBody:
    """Per-clip state: the quantised body scale and the coverage factor alpha.

    Both are single scalars held for the whole clip (alpha as a running max), so the emitted
    outline does not breathe frame to frame - a per-frame fitted width would itself be a signal.
    """

    def __init__(self, scale_quant=0.10, alpha_step=0.05, alpha_max=2.20, thr=0.35,
                 canon=CANON, backstop=True, grow_step=0.04, grow_max=0.60, alpha=1.0,
                 grow_tol=0.002, debug_dir=None, seed=0, jitter=0.06, head_mode="hair",
                 work_res=512, grow_win=60, grow_pct=0.90, fallback="radiallp",
                 fallback_frac=0.02, fallback_keep=4, max_body=2.5):
        self.scale_quant = float(scale_quant)
        self.alpha_step = float(alpha_step)
        self.alpha_max = float(alpha_max)
        self.thr = float(thr)
        self.canon = dict(canon)
        self.backstop = bool(backstop)
        # COVERAGE KNOB (2026-08-01, after the first measured run). The first version reached
        # coverage by scaling every canonical width by `alpha`, and alpha ratcheted to its 2.20
        # ceiling on clip A - a 2.2x-wide figure is a blob, not "a plausible ordinary human", and
        # it cost 2.29x the real area. The residual it was chasing is a roughly UNIFORM outward
        # margin (hair, shoes, loose clothing, the EDGE_EXPAND dilate), and a multiplicative knob
        # is the wrong instrument for an additive gap: covering +12 px on a 17 px-radius forearm
        # needs alpha 1.7, which then also doubles the trunk. `grow` is that additive margin, in
        # units of S, applied as an isotropic dilation of the finished body - it keeps the
        # PROPORTIONS exactly canonical and only puffs the figure.
        self.grow_step = float(grow_step)
        self.grow_max = float(grow_max)
        # RESIDUAL TOLERANCE (2026-08-01, from the measured frontier - grow_frontier.py).
        # Demanding ZERO residual on the WORST frame of a clip is what makes this expensive: on
        # A_corridor the last 0.18 % of coverage costs +0.48x area (1.407x at grow 0.12 vs 1.888x
        # at 0.28). Tolerating a small residual is NOT a §2 reveal - the backstop ORs those pixels
        # back in, so coverage is still complete. What it costs is that this fraction of the
        # emitted boundary is the subject's OWN outline rather than the synthetic one, which is a
        # privacy cost to be priced, not a correctness one.
        self.grow_tol = float(grow_tol)
        self.alpha = float(alpha)          # build multiplier; fixed, NOT fitted per subject
        # GROW AS A WINDOWED PERCENTILE, NOT A RUNNING MAX (2026-08-01 v2, forced by the 108-clip
        # corpus). The monotone ratchet let ONE bad frame pin an entire ~900-frame clip at the
        # ceiling: 41.7 % of clips ended at grow_max, area mean hit 2.674x (max 14.967x), and only
        # 1 of 108 clips landed inside the measured-safe 1.358x (ledger §A.6l-2). A trailing
        # percentile can come back DOWN, so a bad detection costs the frames around it and nothing
        # more. It does mean the margin varies slowly over a clip, which is a weak shape signal - 
        # quantising it to the `grow_step` ladder is what keeps that to a handful of levels.
        self.grow_win = int(grow_win)
        self.grow_pct = float(grow_pct)
        self._gwin = []                    # trailing per-frame REQUIRED margins
        self.grow = 0.0                    # margin actually used on the last frame
        self.head_mode = str(head_mode)
        # POSE-FREE FALLBACK. Triggered when no figure could be drawn, or when the drawn figure
        # still leaves more than `fallback_frac` of the mask to the backstop. Without it those
        # frames emit the subject's own outline verbatim - measured at up to 73 % of the mask on
        # the corpus (see radial_lp's docstring).
        self.fallback = str(fallback)
        self.fallback_frac = float(fallback_frac)
        self.fallback_keep = int(fallback_keep)
        self.max_body = float(max_body)
        self.work_res = int(work_res)      # 0 = full res; see apply() for why this exists
        # per-CLIP build. seed=0 means "no jitter" so old runs stay reproducible.
        self.canon = clip_jitter(seed, self.canon, float(jitter)) if seed and jitter > 0 else \
            dict(self.canon)
        self.seed = int(seed)
        self.debug_dir = debug_dir         # dump the PURE synthetic body (pre-backstop) here
        self._scales = {}            # slot -> running scale samples (a distant person must not
                                     # inherit a near person's size; the BUILD is shared, the
                                     # overall size cannot be)
        self.stats = dict(frames=0, drawn=0, no_pose=0, real_px=0, synth_px=0, body_px=0,
                          backstop_px=0, frames_clean=0, alpha_final=1.0, grow_final=0.0,
                          grow_mean=0.0, grow_capped=0, scale_px=0.0, fallback_frames=0, oversize_frames=0, jitter=float(jitter),
                          head_mode=str(head_mode), work_res=int(work_res))
        self._gsum = 0.0

    def _scale_for(self, key, kp, sc):
        buf = self._scales.setdefault(key, [])
        s = body_scale(kp, sc, self.thr)
        if s > 1.0:
            buf.append(s)
        if not buf:
            return 0.0
        # median over everything seen so far: robust to the profile-view collapse and to a single
        # bad detection, and it converges within a few frames.
        return quantise_scale(float(np.median(buf)), self.scale_quant)

    def _no_body(self, real):
        """No figure could be placed this frame. Emit the pose-free fallback rather than the
        subject's own outline - the mask still has to be defended when the pose model fails."""
        self.stats["no_pose"] += 1
        out = radial_lp(real, self.fallback_keep) if self.fallback == "radiallp" else real
        self.stats["real_px"] += int(real.sum())
        self.stats["synth_px"] += int(out.sum())
        self.stats["backstop_px"] += int(real.sum())
        self.stats["fallback_frames"] += 1
        return out, dict(drawn=0, fallback=self.fallback, backstop_px=int(real.sum()))

    def apply(self, real, people):
        """real: HxW uint8 {0,1} emitted mask so far.  people: list of {"kp":..., "score":...}
        with RAW native-coord keypoints for this frame.
        Returns (mask, info). The mask is ALWAYS a superset of `real`."""
        H, W = real.shape[:2]
        self.stats["frames"] += 1
        if not people:
            return self._no_body(real)

        bodies = []
        for p in people:
            kp = np.asarray(p.get("kp", ()), np.float32)
            sc = np.asarray(p.get("score", ()), np.float32) if p.get("score") is not None else None
            if kp.ndim != 2 or kp.shape[0] < 17:
                continue
            S = self._scale_for(p.get("slot", 0), kp, sc)
            if S <= 1.0:
                continue
            bodies.append((kp, sc, S))
        if not bodies:
            return self._no_body(real)

        # WORK RESOLUTION (2026-08-01 v2). Drawing, the closing op and the distance transform are
        # all O(H·W) or worse, so the stage cost tracked the CAPTURE resolution: 36.7 ms at 1264²
        # against 7.1 ms at 640×480 (measured). The body is a smooth shape and the margin ladder is
        # coarse, so none of that detail survives anyway. Everything below runs on a copy whose
        # long side is `work_res`; the result is upsampled and OR-ed with the FULL-RES real mask,
        # so §2 is untouched - the union is still exact at native resolution.
        ws = 1.0
        if self.work_res and max(H, W) > self.work_res:
            ws = self.work_res / float(max(H, W))
        wh, ww = max(8, int(round(H * ws))), max(8, int(round(W * ws)))
        rw = real if ws == 1.0 else (cv2.resize(real, (ww, wh),
                                                interpolation=cv2.INTER_NEAREST))
        base = np.zeros((wh, ww), np.uint8)
        drawn = 0
        for kp, sc, S in bodies:
            b = draw(kp * ws, sc, wh, ww, S * ws, self.alpha, self.thr, self.canon,
                     real=rw, head_mode=self.head_mode)
            if b is not None:
                base |= b
                drawn += 1
        Smax = max(b[2] for b in bodies) * ws

        # DILATION BY DISTANCE TRANSFORM. Dilating with an elliptical structuring element costs
        # O(H·W·k²) and k reaches ~2·0.6·S - on a 1264² frame that was ~7 min for ONE clip in the
        # re-ID harness, and the coverage search repeats it at every ladder step. The Euclidean
        # dilation of `base` by radius r is exactly {d ≤ r} for d = distance-to-`base`, so ONE
        # transform prices the whole ladder: the residual at radius r is the count of real pixels
        # with d > r. Same result, one pass, and a true disc rather than a polygonal approximation.
        # SANITY GUARD: the figure must be roughly the size of the person it replaces. A pose that
        # is confident but wrong (a distant bystander behind foliage, a spurious spine) yields a
        # correct-looking skeleton at the wrong SCALE, and the figure drawn from it swamped the
        # frame - measured 4.07x mean area on p10_c01 where every other arm sits at 1.47x. When
        # the drawn body exceeds `max_body` times the real mask, it is not on the person: drop it
        # and let the pose-free fallback defend the frame.
        if base.sum() > self.max_body * max(1.0, float(rw.sum())):
            self.stats["oversize_frames"] += 1
            return self._no_body(real)

        dist = cv2.distanceTransform((1 - base).astype(np.uint8), cv2.DIST_L2, 5)
        dreal = dist[rw > 0]

        # THIS frame's REQUIRED margin: the smallest radius that leaves at most `grow_tol` of the
        # real mask uncovered. Read straight off the sorted distances - no ladder walk.
        tol = int(self.grow_tol * float(rw.sum()))
        if dreal.size == 0:
            g_need = 0.0
        else:
            k = dreal.size - 1 - min(tol, dreal.size - 1)
            g_need = float(np.partition(dreal, k)[k]) / max(1e-6, Smax)
        # A frame the figure does not fit (occluded/distant subject) demands an enormous margin,
        # and letting that demand into the window inflates every neighbouring frame too: measured
        # on the corpus, those clips sat at grow 0.52-0.59 AND fired the fallback, costing 4.94x
        # area for coverage the fallback was already providing. Such frames are handled by the
        # fallback and are kept OUT of the window.
        will_fallback = (self.fallback == "radiallp"
                         and g_need > self.grow_max + 1e-9)
        if not will_fallback:
            self._gwin.append(g_need)
            if len(self._gwin) > self.grow_win:
                self._gwin.pop(0)
        if not self._gwin:
            self._gwin.append(min(g_need, self.grow_max))
        # WINDOWED PERCENTILE, not a running max - see __init__ for what the max cost on the
        # corpus. Rounded UP to the ladder so the emitted margin takes only a few discrete values.
        gq = float(np.quantile(self._gwin, self.grow_pct))
        g = min(self.grow_max, math.ceil(gq / self.grow_step - 1e-9) * self.grow_step)
        if gq > self.grow_max:
            self.stats["grow_capped"] += 1
        self.grow = g
        self._gsum += g
        syn = base if g <= 0 else (dist <= g * Smax).astype(np.uint8)
        if ws != 1.0:
            up = lambda a: (cv2.resize(a * 255, (W, H), interpolation=cv2.INTER_LINEAR) > 96
                            ).astype(np.uint8)
            syn, base_full = up(syn), up(base)
        else:
            base_full = base

        # BACKSTOP: whatever is still uncovered (a bag, a coat, a limb whose joints were not
        # confident) is OR-ed back in. §2 first, always.
        leak = int((real & ~syn).sum())
        # PARTIAL FAILURE: the figure was drawn but does not contain the person (occlusion, a
        # second body in the component, a limb the pose missed). Defending only the part the body
        # covers and passing the rest through is the worst of both worlds, so the whole mask goes
        # through the pose-free fallback as well.
        if self.fallback == "radiallp" and leak > self.fallback_frac * max(1.0, float(real.sum())):
            # The low-pass does the covering here. The figure is DROPPED rather than unioned in:
            # a body that misses this much of the mask is not on the person, so keeping it only
            # adds area and draws a limb where there is none (measured: keeping it cost 4.94x on
            # p10_c01 against 4.07x without the margin, and the figure was misplaced in both).
            syn = radial_lp(real, self.fallback_keep)
            leak = int((real & ~syn).sum())
            self.stats["fallback_frames"] += 1
        out = (syn | real) if self.backstop else syn
        if self.debug_dir:
            # the PURE synthetic body, before the backstop - this is what the defence actually
            # produces, and the only way to see what it fails to cover.
            import os as _os
            _os.makedirs(self.debug_dir, exist_ok=True)
            cv2.imwrite(_os.path.join(self.debug_dir, "synth_%05d.png" % self.stats["frames"]),
                        syn * 255)
            # ALSO the un-grown body: the difference between the two is what the coverage margin
            # is paying for, and where the residual sits is what says which canonical constant is
            # too small (hair? shoes? hands?) - a question a single global margin cannot answer.
            cv2.imwrite(_os.path.join(self.debug_dir, "base_%05d.png" % self.stats["frames"]),
                        base * 255)
            cv2.imwrite(_os.path.join(self.debug_dir, "real_%05d.png" % self.stats["frames"]),
                        real * 255)
        self.stats["drawn"] += drawn
        self.stats["real_px"] += int(real.sum())
        self.stats["synth_px"] += int(out.sum())
        self.stats["body_px"] += int(syn.sum())
        self.stats["backstop_px"] += leak
        self.stats["frames_clean"] += int(leak == 0)
        self.stats["alpha_final"] = float(self.alpha)
        self.stats["grow_final"] = round(float(self.grow), 4)
        self.stats["grow_mean"] = round(self._gsum / max(1, self.stats["frames"]), 4)
        self.stats["scale_px"] = float(bodies[0][2])
        return out, dict(drawn=drawn, grow=self.grow, backstop_px=leak)

    def report(self):
        s = dict(self.stats)
        r = max(1, s["real_px"])
        s["area_ratio_vs_real"] = round(s["synth_px"] / r, 4)          # emitted (incl. backstop)
        s["body_ratio_vs_real"] = round(s["body_px"] / r, 4)           # the synthetic body alone
        s["backstop_frac_of_real"] = round(s["backstop_px"] / r, 6)
        return s
