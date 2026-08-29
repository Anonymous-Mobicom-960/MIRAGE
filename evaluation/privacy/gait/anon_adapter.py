"""
anon_adapter -- route CASIA-B tracklets through our anti-reID anonymizer and the
strawman baselines, so every condition hits the SAME frozen adversary through the
identical preprocessing (avoids a domain-fingerprint confound).

Conditions:
  raw            -- passthrough (upper bound on adversary accuracy)
  ours           -- segbench.pose_anon.anonymize_sequence at default knobs
  jitter         -- additive Gaussian noise, magnitude-MATCHED to `ours`
  jitter_lowpass -- jitter then a temporal low-pass (the attacker removing the
                    high-freq jitter) -- shows naive jitter is non-robust

Seed strategy:
  per_identity   -- transform is a fixed function of the TRUE subject id (our
                    design: constant per person). Gallery+probe of one person get
                    the same transform => tests pseudonymity.
  per_sequence   -- fresh seed per (id,status,seq,angle) => breaks cross-sequence
                    linkage; historical deterministic arithmetic retained for
                    old arms only.
  evaluation map -- independent random seed stored per clip for reproducible
                    deployment-parity arms; never use this map in production.

Tracklets are (T,17,3) [x,y,conf]; we transform x,y and carry conf through.
"""
import os
import sys
from collections.abc import Mapping
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, "..", ".."))          # for segbench
sys.path.insert(0, os.path.join(HERE, ".."))
from segbench.pose_anon import anonymize_sequence            # noqa: E402
from segbench.pose_anon_v2 import (anonymize_v2, population_template,   # noqa: E402
                                   anatomical_template_for_clip,
                                   _apply_proximity_envelope)

FPS = 25.0   # CASIA-B nominal frame rate

# default operating knobs of our anonymizer (as shipped in pose_anon)
DEFAULT_KNOBS = dict(bone_sigma=0.12, posture_deg=4.0, lowfreq_amp_frac=0.03,
                     lowfreq_period_s=1.2)


def seed_for(key, mode):
    sub, status, seq, angle = key
    if mode == "per_identity":
        return 1000 + int(sub) * 7
    elif mode == "per_sequence":
        return 1000 + (((int(sub) * 97 + status) * 89 + seq) * 83 + angle)
    raise ValueError(mode)


def build_eval_sequence_seed_map(*tracklet_sets, master_seed=0):
    """Return reproducible, subject-independent random seeds for evaluation clips.

    This is deliberately an *evaluation-only* substitute for deployment's fresh
    ``secrets`` seed.  Values are sampled independently from a PRNG and merely
    stored under each tracklet key; the subject id is not arithmetically folded
    into the seed as it is in :func:`seed_for`.  Sorting only makes the generated
    map invariant to dictionary insertion order.

    Pass every split used by an evaluation in one call (normally train + test),
    then pass the resulting map to both :func:`transform_v2` calls.  This avoids
    accidentally assigning the same deterministic seed schedule independently
    to two splits.  Production must continue to use one fresh cryptographic seed
    per clip and must never use this reproducibility helper.
    """
    keys = sorted({key for tracklets in tracklet_sets for key in tracklets})
    rng = np.random.RandomState(int(master_seed) & 0x7fffffff)
    used = set()
    seed_map = {}
    for key in keys:
        # Collision-free values make it explicit that two clips never share a
        # transform seed in the evaluated corpus.  The retry is effectively
        # free at CASIA-B scale, but avoids relying on probability in a test.
        while True:
            value = int(rng.randint(1, 0x7fffffff))
            if value not in used:
                break
        used.add(value)
        seed_map[key] = value
    return seed_map


def _frame_wh_for(key, frame_wh):
    """Resolve either one global ``(width,height)`` or a per-tracklet mapping."""
    if frame_wh is None:
        return None
    value = frame_wh.get(key) if isinstance(frame_wh, Mapping) else frame_wh
    if value is None:
        return None
    if len(value) != 2:
        raise ValueError("deployment_frame_wh values must be (width, height)")
    width, height = float(value[0]), float(value[1])
    if not np.isfinite(width) or not np.isfinite(height) or width <= 0 or height <= 0:
        raise ValueError("deployment frame width/height must be finite and positive")
    return width, height


def _lowpass(seq, sigma_frames=2.0):
    """Temporal Gaussian smooth over frames (attack on high-freq jitter). seq:(T,17,2)."""
    rad = int(max(1, round(3 * sigma_frames)))
    k = np.exp(-0.5 * (np.arange(-rad, rad + 1) / sigma_frames) ** 2); k /= k.sum()
    T = seq.shape[0]
    pad = np.pad(seq, ((rad, rad), (0, 0), (0, 0)), mode="edge")
    out = np.zeros_like(seq)
    for i in range(len(k)):
        out += k[i] * pad[i:i + T]
    return out


def estimate_matched_sigma(tracklets, sample_n=400, fps=FPS, knobs=DEFAULT_KNOBS, seed=0):
    """Global per-coordinate noise std so `jitter`'s RMS joint displacement matches
    `ours`. RMS_disp(ours) = sqrt(mean ||a-x||^2); per-coord std = RMS/sqrt(2)."""
    keys = list(tracklets.keys())
    rng = np.random.RandomState(seed)
    pick = [keys[i] for i in rng.permutation(len(keys))[:min(sample_n, len(keys))]]
    sqs = []
    for key in pick:
        xy = tracklets[key][..., :2].astype(np.float64)
        a, _ = anonymize_sequence(xy, seed=seed_for(key, "per_identity"), fps=fps, **knobs)
        sqs.append(np.mean(np.sum((a - xy) ** 2, axis=-1)))   # mean ||d||^2 per joint
    rms = float(np.sqrt(np.mean(sqs)))
    return rms / np.sqrt(2.0), rms


def transform_tracklets(tracklets, kind, seed_mode="per_identity",
                        matched_sigma=None, fps=FPS, knobs=DEFAULT_KNOBS):
    """Return {key:(T,17,3)} transformed under `kind` (v1 baselines)."""
    out = {}
    for key, tr in tracklets.items():
        xy = tr[..., :2].astype(np.float64); conf = tr[..., 2:]
        if kind == "raw":
            a = xy
        elif kind == "ours":
            a, _ = anonymize_sequence(xy, seed=seed_for(key, seed_mode), fps=fps, **knobs)
        elif kind in ("jitter", "jitter_lowpass"):
            rng = np.random.RandomState(seed_for(key, seed_mode) & 0x7fffffff)
            a = xy + rng.randn(*xy.shape) * matched_sigma
            if kind == "jitter_lowpass":
                a = _lowpass(a, sigma_frames=2.0)
        else:
            raise ValueError(kind)
        out[key] = np.concatenate([a.astype(np.float32), conf], axis=-1)
    return out


# --------------------------------------------------------------------- v2 (redesign)
def build_template(tracklets, sample_n=500, seed=0):
    """Shared collapse target = population-median per-group bone length. Sampled
    for speed; it is an aggregate statistic (not identity-specific)."""
    keys = list(tracklets.keys())
    rng = np.random.RandomState(seed)
    pick = [keys[i] for i in rng.permutation(len(keys))[:min(sample_n, len(keys))]]
    return population_template([tracklets[k][..., :2].astype(np.float64) for k in pick])


def transform_v2(tracklets, config, template, seed_mode="per_identity", fps=FPS,
                 scale_from=None, sequence_seed_map=None, deployment_frame_wh=None):
    """Return {key:(T,17,3)} through the redesigned anonymizer with `config` knobs.

    `scale_from` (2026-08-08, default None) selects the COLLAPSE TARGET, one level up from
    `anonymize_v2`'s kwargs - exactly as `pose_anon_edge.anonymize_pose_log` does:

      None                    every tracklet collapses onto the SHARED population `template`
                              passed in. This is what every reID number on record was measured
                              with, and the branch is byte-identical to the pre-2026-08-08 code
                              (asserted by `fix7_parity.py`, max|Δ| 0.0).
      "shoulder"|"spine"|     PER-TRACKLET: `anatomical_template_for_clip(xy, scores, src=...)`,
      "projected"|"extent"    i.e. the shared PROPORTION set scaled by a quantity observed on
                              THIS tracklet - which is what DEPLOYMENT does. The `template`
                              argument is then unused for the collapse target and is kept only
                              so callers need not branch.

    🔴 Why this exists: in the lab the emitted figure's absolute SIZE was a population constant,
    so the height channel was fully destroyed and NO TM1/TM2/TM3 number on record priced the
    deployed per-clip size channel under ANY source (`_e2e/run3_20260807/FIX3_STICKS.md` §5.1).

    `scores` handed to the scale source are the RAW per-joint confidences, matching
    `anonymize_pose_log` (which reads `pose_log[i][j]["score"]` before any binarisation).
    Note this is not a free choice that could confound the arms: every scale source gates on
    `conf >= 0.5` and `binarize_conf` is 0.5, so raw and binarised confidences select exactly
    the same frames/joints.

    Adapter-only keys are popped here (never passed to anonymize_v2); conf is otherwise
    carried through byte-identical across all configs:
      strip_conf    -- True: replace the per-joint confidence channel with a constant 1.0.
                       ABLATION ONLY: it measures what conf contributes, but it is not what
                       any deployment emits (it destroys occlusion/visibility information
                       the WanAnimate retarget needs).
      binarize_conf -- float threshold: emit conf as {0,1} at that threshold. This is what
                       the SHIPPED pipeline actually does (mirage_tier1.py /
                       pose_anon_edge.binarize_pose_scores at config.POSE_THRESH=0.5, and
                       the same in the cloud mirage_rtmpose node), so this is the mode that
                       measures the DEPLOYED artifact rather than a lab analogue.
      deployment_postprocess -- opt-in exact body-joint emit semantics: after anonymization,
                       zero x/y wherever the RAW score is below ``binarize_conf``; clip x to
                       [0,width] and y to [0,height] when ``deployment_frame_wh`` supplies
                       dimensions; then round x/y to two decimals. CASIA-B's pose CSV and
                       loader contain no source frame dimensions, so its TM3 runs reproduce
                       zeroing + rounding but must skip clipping rather than invent a canvas.
      eval_seed_map_master -- provenance for the deterministic evaluation-only seed map. If
                       ``sequence_seed_map`` is omitted, a map for this one input dictionary
                       is built from this master. TM3 should build one map over train + test
                       and pass it to both calls. Production must use fresh secrets seeds.

    All deployment behavior is opt-in. Configs without ``deployment_postprocess`` and without
    ``sequence_seed_map`` retain the historical arithmetic seed and coordinate behavior.
    """
    cfg = dict(config)
    strip_conf = cfg.pop("strip_conf", False)
    binarize_conf = cfg.pop("binarize_conf", None)
    deployment_postprocess = bool(cfg.pop("deployment_postprocess", False))
    eval_seed_map_master = cfg.pop("eval_seed_map_master", None)
    if strip_conf and binarize_conf is not None:
        raise ValueError("strip_conf and binarize_conf are mutually exclusive")
    if deployment_postprocess and binarize_conf is None:
        raise ValueError("deployment_postprocess requires binarize_conf")
    if deployment_frame_wh is not None and not deployment_postprocess:
        raise ValueError("deployment_frame_wh requires deployment_postprocess")
    if sequence_seed_map is not None and seed_mode != "per_sequence":
        raise ValueError("sequence_seed_map is valid only with seed_mode='per_sequence'")
    if sequence_seed_map is None and eval_seed_map_master is not None:
        if seed_mode != "per_sequence":
            raise ValueError("eval_seed_map_master requires seed_mode='per_sequence'")
        sequence_seed_map = build_eval_sequence_seed_map(
            tracklets, master_seed=eval_seed_map_master)
    cfg.pop("scale_from", None)        # never an anonymize_v2 kwarg; see the arg of the same name
    out = {}
    kinds = {}
    for key, tr in tracklets.items():
        xy = tr[..., :2].astype(np.float64); conf = tr[..., 2:].astype(np.float32)
        raw_scores = tr[..., 2].astype(np.float64)
        if strip_conf:
            conf = np.ones_like(conf)
        elif binarize_conf is not None:
            conf = (conf >= float(binarize_conf)).astype(np.float32)
        tmpl = template
        if scale_from is not None:
            tmpl, kind = anatomical_template_for_clip(xy, scores=raw_scores, src=scale_from)
            kinds[kind] = kinds.get(kind, 0) + 1
        if sequence_seed_map is None:
            transform_seed = seed_for(key, seed_mode)
        else:
            try:
                transform_seed = int(sequence_seed_map[key])
            except KeyError as exc:
                raise KeyError(f"sequence_seed_map has no seed for tracklet {key!r}") from exc
        anon_cfg = dict(cfg, proximity_scores=raw_scores) if deployment_postprocess else cfg
        a = anonymize_v2(xy, tmpl, seed=transform_seed, fps=fps, **anon_cfg)
        if deployment_postprocess:
            # Match pose_anon_edge.anonymize_pose_log ordering for COCO-17:
            # anonymize -> final visible-coordinate cap -> zero never-observed joints ->
            # optional canvas clip -> round(2).
            a = np.asarray(a, dtype=np.float64).copy()
            wh = _frame_wh_for(key, deployment_frame_wh)
            proximity_cap = float(cfg.get("proximity_cap_frac", 0.0))
            if wh is not None and proximity_cap:
                visible_ref = xy.copy()
                visible_a = a.copy()
                for arr in (visible_ref, visible_a):
                    arr[..., 0] = np.clip(arr[..., 0], 0.0, wh[0])
                    arr[..., 1] = np.clip(arr[..., 1], 0.0, wh[1])
                a = _apply_proximity_envelope(
                    visible_a, visible_ref, proximity_cap, scores=raw_scores)
            a[raw_scores < float(binarize_conf)] = 0.0
            if wh is not None:
                a[..., 0] = np.clip(a[..., 0], 0.0, wh[0])
                a[..., 1] = np.clip(a[..., 1], 0.0, wh[1])
            a = np.round(a, 2)
        out[key] = np.concatenate([a.astype(np.float32), conf], axis=-1)
    if scale_from is not None:
        # Declare the RESOLVED source the way mirage_tier1 records `anon.template_kinds`, so
        # "the operator asked for extent and the trunk was never seen" is visible rather than
        # silently falling back to `shoulder`.
        print(f"[anon_adapter] scale_from={scale_from!r} -> template_kinds={kinds}", flush=True)
    return out
