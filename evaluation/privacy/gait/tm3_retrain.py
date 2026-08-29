"""
tm3_retrain.py -- TM3 ADAPTIVE adversary (Kerckhoffs' worst case): retrain the
GaitGraph ResGCN FROM SCRATCH on anonymized train-split poses, then evaluate
anon-gallery vs anon-probe on the disjoint test ids. This is the decisive number:
the frozen clean adversary is only a LOWER BOUND; a retrained one can exploit any
residual structure the anonymizer leaves behind.

Recipe faithfully mirrors GaitGraph/src/train.py: SupCon loss, two augmented views
(mirror/flip/random-crop/point+joint noise), Adam + OneCycleLR + AMP.

Usage:
  python tm3_retrain.py --config v2-full --epochs 200
The anonymizer config is applied ONCE per sequence (fixed per-clip transform =
what actually leaves Tier-1); the adversary then augments + trains on that.
Template is built from TRAIN ids (public, disjoint from the test identities).
"""
import os
import sys
import json
import time
import argparse
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
sys.path.insert(0, os.path.join(HERE, ".."))
sys.path.insert(0, os.path.join(HERE, "vendor", "GaitGraph", "src"))

from casia_loader import load_tracklets                       # noqa: E402
import protocol as P                                          # noqa: E402
import anon_adapter as A                                      # noqa: E402
from adversary import multi_input, select_sequence_center     # noqa: E402
# vendored (pure torch/numpy)
from datasets.augmentation import (MirrorPoses, FlipSequence, RandomSelectSequence,   # noqa: E402
                                   ShuffleSequence, PointNoise, JointNoise, MultiInput,
                                   ToTensor, TwoNoiseTransform)
from datasets.graph import Graph                              # noqa: E402
import models.ResGCNv1 as ResGCNv1                            # noqa: E402
from losses import SupConLoss                                 # noqa: E402

DATA = os.path.join(HERE, "data")
OUT_DIR = os.path.join(HERE, "..", "reports", "reid")

# candidate configs (keep in sync with sweep_pareto.LADDER); knee set after sweep
CONFIGS = {
    "v2-canon":      dict(do_canon=True),
    "v2-full":       dict(do_canon=True, cadence_amp=0.25, angle_const_deg=8,     # = L2
                          angle_drift_deg=6, asymmetry_sigma=0.05),
    "L3":            dict(do_canon=True, cadence_amp=0.40, angle_const_deg=14,
                          angle_drift_deg=10, asymmetry_sigma=0.08),
    "L4":            dict(do_canon=True, cadence_amp=0.60, angle_const_deg=20,
                          angle_drift_deg=15, asymmetry_sigma=0.12),
    "L5":            dict(do_canon=True, cadence_amp=0.80, angle_const_deg=28,
                          angle_drift_deg=20, asymmetry_sigma=0.16),
    # confound test: L4 + kill the global walking trajectory (per-frame root-centering)
    "L4c":           dict(do_canon=True, cadence_amp=0.60, angle_const_deg=20,
                          angle_drift_deg=15, asymmetry_sigma=0.12, center_root=True),
    # control: root-centering alone (no dynamic perturb) - isolates the trajectory channel
    "canon-c":       dict(do_canon=True, center_root=True),
    # conf-channel ablation (audit): L4 + strip the per-joint confidence side-channel
    "L4-nc":         dict(do_canon=True, cadence_amp=0.60, angle_const_deg=20,
                          angle_drift_deg=15, asymmetry_sigma=0.12, strip_conf=True),
    # BOTH side-channels removed (conf + trajectory) -> the TRUE intrinsic-dynamics floor
    "L4c-nc":        dict(do_canon=True, cadence_amp=0.60, angle_const_deg=20, angle_drift_deg=15,
                          asymmetry_sigma=0.12, center_root=True, strip_conf=True),
    # ***** AS-RENDERED 2026-07-27: the config the CORRECTED arms actually used *****
    # Identical to L4-bin except the ANGLE knobs are ZERO. B.37 measured that those two knobs are
    # the entire remaining source of emitted-skeleton jitter (8.7x -> 1.6x the real person), and
    # B.41/B.44 rendered a10/a15/a30 with them at 0/0. B.38 priced it at +4.41 pp against the
    # FROZEN adversary (TM1); this row is the ADAPTIVE number, which is what the posture is
    # actually quoted against.
    # NOTE the shipped DEFAULT in pose_anon_edge.LEVELS is still 20/15 -- 0/0 was applied via
    # MIRAGE_ANGLE_CONST/DRIFT when the bundles were built, so this measures the ARTIFACT that
    # left Tier-1, not the committed default.
    "L4-bin-a0":     dict(do_canon=True, cadence_amp=0.60, angle_const_deg=0,
                          angle_drift_deg=0, asymmetry_sigma=0.12, binarize_conf=0.5),
    # ***** ANGLE CURVE UNDER THE ADAPTIVE ADVERSARY (2026-07-27) *****
    # B.38 swept angle_const/angle_drift against the FROZEN adversary (TM1) at 20/15, 14/10, 8/6,
    # 0/0 and concluded 0/0 Pareto-dominates 8/6 at a price of +4.41 pp. TM1 is a LOWER BOUND and
    # tells you nothing about what a retrained adversary does with the same knob, so the same four
    # points are measured here against TM3. Everything except the two angle knobs is held at the
    # deployed L4-bin operating point, so the curve isolates exactly one dimension.
    "L4-bin-a1410":  dict(do_canon=True, cadence_amp=0.60, angle_const_deg=14,
                          angle_drift_deg=10, asymmetry_sigma=0.12, binarize_conf=0.5),
    "L4-bin-a86":    dict(do_canon=True, cadence_amp=0.60, angle_const_deg=8,
                          angle_drift_deg=6, asymmetry_sigma=0.12, binarize_conf=0.5),
    # The FLOOR for that curve: canonicalisation + conf-binarisation only, every dynamic knob off.
    # This is "what the adaptive adversary gets if the dynamic anonymiser is switched off entirely",
    # and it bounds the whole ladder from above. B.38's TM1 analogue scored 18.9 %.
    "L0-bin":        dict(do_canon=True, cadence_amp=0.0, angle_const_deg=0, angle_drift_deg=0,
                          asymmetry_sigma=0.0, binarize_conf=0.5),
    # ***** THE DEPLOYED OPERATING POINT (2026-07-22 edge + cloud) *****
    # canon + L4 dynamics + conf BINARIZED to {0,1} at the pipeline's own POSE_THRESH=0.5
    # + trajectory KEPT (no center_root) + per-SEQUENCE seed (seed_mode below).
    # Every other row is a lab analogue; this one is the artifact that actually leaves
    # Tier-1. Its closest analogues bracket it: L4 (conf intact) 33.2% and L4-nc (conf
    # stripped) 10.0% NM. Same 100-epoch budget/schedule as those two, for comparability.
    "L4-bin":        dict(do_canon=True, cadence_amp=0.60, angle_const_deg=20, angle_drift_deg=15,
                          asymmetry_sigma=0.12, binarize_conf=0.5),
}

# ***** GAIT-SWEEP 2026-07-31: strengthen the gait channel WITHOUT touching the angle knobs *****
# The angle pair is frozen at the shipped 14/10 (user decision 2026-07-27) because it is the
# dominant per-frame-jitter source and per-frame joint travel is the causal driver of the cloud's
# invented-hands defect (§B.34/§B.35/§B.44). These arms buy privacy from knobs that are free or
# near-free on that jitter metric instead: cadence amplitude/period (a time RE-SAMPLE, measured
# jerk 0.45 vs the real input's 0.50), a NEW static independent L/R length break, and a NEW
# per-limb cadence decorrelation. Registered from ONE definition (gaitsweep_arms.ARMS) so the
# TM3 arms, the TM1 ranking arms and the jitter arms cannot silently drift apart.
# They appear as `gs_<key>`; `gs_base_1410` is the shipped baseline and must be run in the SAME
# session as any challenger (§B.21: never pair numbers from two measurement configs).
# ***** PROJECTION-FIT / AXIAL-EXEMPT ARMS (2026-08-02) *****
# §A.2i attributed ALL the posture damage to 5 of the 9 bone groups (spine->tilt, clav->roll,
# hip->asymmetry, neck+face->head-lean 10.2x) and none of it to the limbs, and showed that the
# fixed frontal ratio spine = 1.231 x shoulder cannot fit a PROJECTED body -- each scale source
# wins one clip and loses the other on joint containment. These arms price the two resulting
# changes against the ADAPTIVE adversary, which is the only adversary that has ever priced this
# knob correctly (TM1 under-priced the angle knob 4.1x, §A.2c).
#
# Baseline is L4-bin-a1410 (the shipped operating point, 14.13 +/- 3.07). It does NOT need
# re-running this session despite §B.21: the lab anonymiser was proven bit-identical to its own
# HEAD with both new kwargs at their defaults, AND bit-identical to the edge copy in all six
# configurations (max|delta| = 0), so the baseline and these arms are the same measurement config.
# PRE-REGISTERED SHIP BAR: mean <= 20.63 % (the 8/6 rung that would otherwise be taken).
_A1410 = dict(do_canon=True, cadence_amp=0.60, angle_const_deg=14, angle_drift_deg=10,
              asymmetry_sigma=0.12, binarize_conf=0.5)
_LIMBS = ("uarm", "farm", "thigh", "shin")
CONFIGS["a1410-tq"]     = dict(_A1410, angle_groups=_LIMBS)                        # axial-exempt
CONFIGS["a1410-proj"]   = dict(_A1410, projection_fit=True)                        # projection fit
CONFIGS["a1410-tqproj"] = dict(_A1410, angle_groups=_LIMBS, projection_fit=True)   # = g9_proj_tq

# ***** g6tq BOOST (2026-08-03) - recover g6tq's privacy WITHOUT giving back the axial angles *****
# g6tq fixes the visible posture (B.54) but costs +5.20 pp and put one seed at 26.75 %, above the
# 20.63 % bar. Restoring the axial angles would undo the quality win, so the perturbation is bought
# from knobs that a posture screen (tier1_lab/gait_g6tq_boost.py -> GAIT_G6TQ_BOOST_SCREEN.json,
# 6 seeds x 2 clips) showed leave tilt/roll/head-lean untouched:
#   limb_phase_amp  per-limb-chain time warp: perturbs inter-limb PHASE, a large part of what a
#                   gait recogniser reads. tilt/roll/head all 0.99-1.01x. +40-51 % limb travel.
#   cadence_amp     global time re-sample, same family. +31 % limb travel, head 0.96-1.00x.
# 🔴 lr_asymmetry_sigma was the obvious third candidate and is DISQUALIFIED: independent L/R
# factors on the clav edges tilt the shoulder line and the head follows -- B_atrium head-lean
# 0.97x -> 6.00x at sigma 0.30. It reintroduces the exact defect the axial exemption removes.
_G6 = dict(_A1410, angle_groups=_LIMBS)
CONFIGS["g6tq-limb090"]        = dict(_G6, limb_phase_amp=0.90)
CONFIGS["g6tq-limb090-cad090"] = dict(_G6, limb_phase_amp=0.90, cadence_amp=0.90)
CONFIGS["g6tq-limb120-cad090"] = dict(_G6, limb_phase_amp=1.20, cadence_amp=0.90)

# ALL THREE (2026-08-03). The boost arms fix g6tq's privacy but leave the STATURE error the
# projection fit corrects (shipped emits 0.54x real height on the corridor, 1.30x on the atrium).
# This is the config that should get everything: axial-exempt posture + boosted privacy +
# corrected scale. Its posture is guaranteed by the same screen -- limb_phase and cadence do not
# touch tilt/roll/head-lean, and projection_fit only rescales.
CONFIGS["g6tq-limb120-cad090-proj"] = dict(_G6, limb_phase_amp=1.20, cadence_amp=0.90,
                                           projection_fit=True)

# ***** THE MISSING POSITIVE CONTROL (2026-08-07) *****
# Passthrough: the adaptive adversary trained and evaluated on UNTOUCHED CASIA-B poses, through
# the identical 100-epoch recipe every defended arm uses. Without it "11.10 %" has a denominator
# of 2.0 % (assumed) and no numerator at all - there is no evidence in this harness that the
# adversary can see identity when it IS present. See the `__raw__` branch in main().
CONFIGS["RAW-control"] = dict(__raw__=True)

# ***** THE 2026-08-07 KNOBS (owner's two visual reports) *****
# Each is g10_boost plus ONE change, so TM3 prices the change and nothing else. Both knobs are
# additive kwargs of anonymize_v2 with defaults that reproduce every earlier number bit-for-bit
# (_e2e/gait_close_20260807/parity_check.py asserts it), and anon_adapter.transform_v2 forwards
# them verbatim.
#   angle_mirror="outward"  anti-symmetric paired-limb rotation, outward-only: fixes the
#                           common-mode leg lean without letting the knees cross (plain "mirror"
#                           crossed them in 57.9 % of frames).
#   angle_groups=(uarm,farm) legs exempt from the ANGLE knob entirely; they keep the real joint
#                           angles and are perturbed only in TIME by limb_phase + cadence. Costs
#                           1.5 % of total limb perturbation geometrically -- this row is whether
#                           it costs privacy.
#   head_anchor="nose"      puts the emitted face on the REAL face (owner request). Hands back the
#                           real neck vector; bounded above by SS B.11-gate's +2.99 pp, unpriced.
_G10 = dict(_G6, limb_phase_amp=1.20, cadence_amp=0.90, projection_fit=True)
CONFIGS["g16_outward"]      = dict(_G10, angle_mirror="outward")
CONFIGS["g18_armangles"]    = dict(_A1410, angle_groups=("uarm", "farm"), limb_phase_amp=1.20,
                                   cadence_amp=0.90, projection_fit=True)
CONFIGS["g19_armangles_ha"] = dict(_A1410, angle_groups=("uarm", "farm"), limb_phase_amp=1.20,
                                   cadence_amp=0.90, projection_fit=True, head_anchor="nose")
CONFIGS["g12_headanchor"]   = dict(_G10, head_anchor="nose")

# ***** THE 2026-08-08 ROWS - FIX7: the deployed PER-CLIP SIZE CHANNEL, and the root lock *****
# `_e2e/run3_20260807/FIX3_STICKS.md` §5.1: every harness in this directory builds ONE
# `population_template(...)` and hands the SAME absolute-size skeleton to every tracklet, so in the
# lab the emitted figure's SIZE is a population constant and the height channel is fully destroyed.
# DEPLOYMENT calls `anatomical_template_for_clip`, which scales the shared PROPORTION set by a
# quantity observed on THIS clip. Consequence: no TM1/TM2/TM3 number on record prices the deployed
# size channel under ANY source - not `shoulder` (ships), not `projected` (approved), not `extent`.
# `anon_adapter.transform_v2(..., scale_from=...)` routes each tracklet through the SAME function
# the device calls (verbatim lab mirror, parity max|Δ| 0.0), so these rows measure the deployed
# configuration rather than a lab analogue of it.
#
# Every row below is `g18_armangles` plus ONE change, so TM3 prices the change and nothing else.
# Reference (R0, must reproduce this session): `g18_armangles` = 11.37 ± 0.38 % (§A.2m-2c).
# PRE-REGISTERED SHIP BAR: mean ≤ 20.63 % (§A.2c).
_G18 = dict(_A1410, angle_groups=("uarm", "farm"), limb_phase_amp=1.20,
            cadence_amp=0.90, projection_fit=True)
# R1 - the root lock, ISOLATED. Population template, so the ONLY difference from R0 is that the
#      cadence time-warp is applied to the ROOT-RELATIVE pose and re-attached to the true
#      per-frame mid-hip instead of dragging the whole figure along its own walking path.
#      Measurable here precisely because `casia_loader` applies no normalisation and
#      `multi_input(..., enabled=False)` feeds raw x,y,conf: the adversary DOES see absolute
#      position over time, which is what this knob restores.
CONFIGS["fix7_R1_rootlock"]   = dict(_G18, cadence_root_lock=True)
# R2-R5 - PER-TRACKLET template. `scale_from` is popped by transform_v2, never passed to
#         anonymize_v2 (same contract as `anonymize_pose_log`).
CONFIGS["fix7_R2_pc_shoulder"]  = dict(_G18, scale_from="shoulder")    # what SHIPS today
CONFIGS["fix7_R3_pc_projected"] = dict(_G18, scale_from="projected")   # `g18p` = what was APPROVED
CONFIGS["fix7_R4_pc_extent"]    = dict(_G18, scale_from="extent")      # `g18e` = the scale fix
CONFIGS["fix7_R5_g20"]          = dict(_G18, scale_from="extent",      # `g20` = both fixes
                                       cadence_root_lock=True)
# R6 (added 2026-08-08 mid-grid, once R3 came in ABOVE the 20.63 % bar at 21.15 +/- 0.50) - the
# DEPLOYABLE combination the original R0..R5 list has no row for. The owner's report has two
# independent halves: "looks like a small person" (SIZE) and "not completely on the person"
# (POSITION). R3/R4/R5 all change SIZE, which is where the privacy cost turned out to live. This
# arm changes POSITION ONLY: the shipped `shoulder` scale source, plus `cadence_root_lock`.
# R1 prices the same knob but on the POPULATION template, which is not a deployable configuration;
# this is R1's deployable twin and the natural fallback if the scale arms fail the bar.
CONFIGS["fix7_R6_pc_shoulder_lock"] = dict(_G18, scale_from="shoulder", cadence_root_lock=True)

# ***** 2026-08-11 BOUNDED SEEDED CLOSE-POSE ARMS *****
# Exact lab twins of pose_anon_edge.GAIT_PRESETS["seedclose1".."seedclose3"]. All three first
# collapse onto the same canonical anatomy, recover visible size through the per-tracklet extent
# fit, then apply fresh per-SEQUENCE bounded multipliers. The proximity envelope is part of the
# attacked output, so these rows measure the privacy price of keeping sticks close enough for
# pose-conditioned generation. RAW-control + the measured null remain the denominator.
CONFIGS["seedclose1"] = dict(
    do_canon=True, scale_from="extent", projection_fit=False, cadence_root_lock=True,
    cadence_amp=0.0, limb_phase_amp=0.0, angle_const_deg=0.0, angle_drift_deg=0.0,
    asymmetry_sigma=0.0, lowfreq_amp_frac=0.0, binarize_conf=0.5,
    seeded_global_scale_max=0.03, seeded_region_scale_max=0.0,
    proximity_cap_frac=0.04, deployment_postprocess=True,
    eval_seed_map_master=20260811)
CONFIGS["seedclose2"] = dict(
    do_canon=True, scale_from="extent", projection_fit=False, cadence_root_lock=True,
    cadence_amp=0.25, limb_phase_amp=0.30, angle_const_deg=8.0, angle_drift_deg=6.0,
    angle_groups=("uarm", "farm", "thigh", "shin"), asymmetry_sigma=0.0,
    lowfreq_amp_frac=0.0, binarize_conf=0.5,
    seeded_global_scale_max=0.025, seeded_region_scale_max=0.04,
    proximity_cap_frac=0.06, deployment_postprocess=True,
    eval_seed_map_master=20260811)
CONFIGS["seedclose3"] = dict(
    do_canon=True, scale_from="extent", projection_fit=False, cadence_root_lock=True,
    cadence_amp=0.90, limb_phase_amp=1.20, angle_const_deg=14.0, angle_drift_deg=10.0,
    angle_groups=("uarm", "farm"), asymmetry_sigma=0.0, lowfreq_amp_frac=0.0,
    binarize_conf=0.5, seeded_global_scale_max=0.02,
    seeded_region_scale_max=0.06, proximity_cap_frac=0.08,
    deployment_postprocess=True, eval_seed_map_master=20260811)

# ***** 2026-08-12 QUALITY-FIRST ARMS: RANDOMISE THE SIZE CHANNEL INSTEAD OF SUPPRESSING IT *****
# THE PROBLEM THESE ROWS EXIST TO SOLVE. The owner's two standing quality complaints are SIZE
# ("looks like a small person") and POSITION ("not completely on the person"). FIX7 measured the
# two fixes and they do not fit in the privacy budget together:
#     R2  per-clip `shoulder`                       11.97 +/- 0.06   (ships; wrong size)
#     R4  per-clip `extent`            = `g18e`     20.12 +/- 0.59   (right size, ON the bar)
#     R6  `shoulder` + root lock                    15.87 +/- 0.27   (right position, wrong size)
#     R5  `extent` + root lock         = `g20`      21.07 +/- 1.04   🔴 BOTH fixes -> FAILS the bar
# So "fix the quality" costs +9.10 pp and busts the 20.63 % bar. That is why every quality arm so
# far has been a failure.
#
# THE MECHANISM OF THE COST is not the size ERROR, it is the size MEASUREMENT: `extent` makes the
# emitted stature a faithful per-clip readout of THIS person's stature, which is a body dimension
# and therefore an identity channel (§A.2p-3: a genuinely second body dimension costs ~15x more
# than propagating the already-disclosed shoulder span).
#
# THE FIX IS NOT TO STOP MEASURING IT -- fidelity needs the measurement -- BUT TO DESTROY ITS
# INFORMATION CONTENT AFTER MEASURING IT. `seeded_global_scale_max` multiplies the collapse target
# by a fresh per-SEQUENCE draw in [1-m, 1+m], so the emitted stature is (this person's stature) x
# (independent noise). The adversary sees stature + noise; the generator sees a figure whose
# TYPICAL size error is m, not the 0.46x-1.16x it gets from `shoulder`.
#
# 🔴 THE AXIS MATTERS, AND IT IS THE POINT. `_apply_projection_fit` (step 2d) runs AFTER the
# template scaling and re-squeezes X back onto the observed shoulder span, so it DIVIDES THE DRAW
# BACK OUT OF X. That is not a bug here, it is the design: X stays shoulder-derived, a channel
# FIX7 already priced at +0.60 pp (R2 vs R0), while the draw lands entirely on Y -- exactly the
# +8.15 pp channel `extent` opens. The randomisation is aimed at the expensive dimension only.
#
# PREDICTED BOUNDS, so this is falsifiable rather than hopeful:
#     m -> 0    must reproduce R5   = 21.07 (the draw vanishes)
#     m -> big  emitted size becomes independent of the subject, so the size channel carries no
#               more than a population constant does -> should approach R1/R6 (~15.9-16.1)
# Two values bracket the knee. If NEITHER lands under the bar, the design is refuted and `extent`
# cannot be bought back by randomisation at any quality-acceptable amplitude.
#
# PROTOCOL: these are `_G18` + `scale_from` + `cadence_root_lock` + ONE knob, i.e. R5 plus one
# change, deliberately WITHOUT `deployment_postprocess`/`eval_seed_map_master` so they stay in the
# FIX7 measurement config and are directly comparable to R0-R6 (§B.21: never pair numbers from two
# measurement configs). Reference to reproduce: R5 = 21.07 +/- 1.04, R0 = 11.37 +/- 0.38.
_G20 = dict(_G18, scale_from="extent", cadence_root_lock=True)          # == fix7_R5_g20
CONFIGS["qsize_m10"] = dict(_G20, seeded_global_scale_max=0.10)         # +/-10 % (sd 5.8 %)
CONFIGS["qsize_m20"] = dict(_G20, seeded_global_scale_max=0.20)         # +/-20 % (sd 11.5 %)

# 🔴 FALLBACK ARM - and a correction the whole `projection_fit` family needs.
# MEASURED 2026-08-12 (`100826_runs/_defence_opus/collapse_floor.py` -> COLLAPSE_FLOOR.json), pure
# collapse with every perturbation knob at zero, two subjects (p03_c02, p04_c04), mean absolute
# difference in bone-length/spine proportions over 7 groups:
#     raw, no defence                       0.0559
#     collapse @ extent, projection_fit ON  0.0542   <- 97 % of the undefended difference SURVIVES
#     collapse @ extent, projection_fit OFF 0.0000   <- exact k-same, by construction
# `_apply_projection_fit` rescales X by a PER-CLIP factor, and a uniform X scale changes the length
# of every non-vertical bone, so the emitted proportion set becomes subject-specific again. The
# claim "every subject still lands on ONE identical skeleton shape", which the docstrings of
# `anatomical_template_for_clip` and `_stature_scale` use to argue that scale-source changes are
# privacy-neutral, is therefore FALSE for every arm carrying `projection_fit=True` -- which is
# `g18` (ships), `g10_boost`, `g18p`, `g18e` and `g20`. Their measured TM3 numbers stand; what does
# not stand is the reason given for them. The shape channel is not what protects those arms.
# This arm drops the projection fit, so the collapse is exact, and keeps the size randomisation.
# It costs pose fidelity (d50 0.066/0.110 H vs 0.051/0.062 H with the fit) and is the fallback if
# the two arms above fail the bar.
CONFIGS["qsize_m10_nopf"] = dict(_G20, seeded_global_scale_max=0.10, projection_fit=False)

# 🔴 SMOOTH ARM - owner reported `qsize_m10`'s render as "a bit sudden in between" (2026-08-12).
# The suddenness localises to ONE knob. MEASURED single-variable on p03_c02, 50 f, seed pinned 777
# (jerk = median per-joint 3rd difference, p95 acc = 95th-pct 2nd difference, both normalised by
# observed visible height; d50 = median per-joint displacement from the anon-OFF skeleton):
#     config                       jerk/real   p95 acc   d50
#     REAL (anon off)                  1.00     0.0450   0.000
#     qsize_m10: cad 0.90 limb 1.20    1.78     0.0574   0.065
#     cadence ONLY (limb 0)            1.74     0.0601   0.063
#     limb ONLY (cadence 0)            1.06     0.0472   0.063
#     cad 0.0 limb 1.8  <- THIS ARM    1.13     0.0496   0.063
#     cad 0.3 limb 1.8                 1.70     0.0523   0.063
# `cadence_amp` causes essentially ALL the excess jerk while adding almost nothing over limb-phase
# alone (0.065 vs 0.063), and it is not a dial: 0.3 already restores 1.70x. Perturbation SATURATES
# at 0.063, so `limb_phase_amp` above ~1.8 buys nothing.
#
# WHY THIS MAY ALSO BE THE BETTER PRIVACY TRADE, from `_limb_phase_warp`'s own docstring:
# `_cadence_warp` re-times the WHOLE skeleton on ONE clock, so every inter-limb phase relationship
# survives it intact -- and those are what a gait recogniser reads. `_limb_phase_warp` is the knob
# that breaks them. So cadence is the expensive, low-value half of the pair. PREDICTION, stated in
# advance so it is falsifiable: this arm should land at or BELOW `qsize_m10`'s 18.43 +/- 1.02 %.
# If it lands materially ABOVE, the prediction is wrong and cadence was carrying real defence.
#
# 🔴 It must NOT inherit `qsize_m10`'s number: §A.2k priced the two time re-samples together
# (-8.23 pp) and never separated them. This arm is `qsize_m10` with cadence off and limb-phase up.
CONFIGS["qsize_m10_smooth"] = dict(_G20, seeded_global_scale_max=0.10,
                                   cadence_amp=0.0, limb_phase_amp=1.80)

# 🔴 RESULT FOR THE ARM ABOVE, AND THE PREDICTION IT REFUTED (measured 2026-08-12, n=3):
#     qsize_m10_smooth  19.79 / 21.37 / 19.27  ->  20.14 +/- 1.11
# The bar is 20.63, so this is NOT a pass: 1/3 seeds is over it and 20.14 +/- 1.11 is not
# statistically below the bar (one-sided t = 0.77, p ~ 0.26). Against `qsize_m10`'s 18.43 +/- 1.02
# it is 1.71 pp WORSE. The prediction recorded above -- that dropping `cadence_amp` would cost
# little because `_cadence_warp` leaves inter-limb phase intact -- is therefore WRONG. Cadence
# carries roughly 1.71 pp of real defence, and the docstring's phase argument does not describe
# what the adaptive adversary actually reads. Smoothness and privacy genuinely trade on this knob.
#
# THIS ARM tests whether the two effects COMPOSE, so the smooth motion can be bought back with the
# size channel instead. `qsize_m20` measured 16.50 (n=2) vs `qsize_m10`'s 18.43, i.e. about
# -1.9 pp for the wider draw -- close to the +1.71 pp that removing cadence cost.
# PREDICTION, again stated in advance: if the effects are additive this lands near
# 20.14 - 1.9 = ~18.2, i.e. back under the bar with roughly `qsize_m10`'s margin but with the
# smooth motion. If it lands near 20 the effects do NOT compose and the smooth family is dead.
# COST IF IT WORKS: the wider draw spreads emitted stature (0.929-1.172 vs 1.013-1.155 at m10),
# which is a visible quality cost the owner has to accept -- q1w's render was visibly less stable.
CONFIGS["qsize_m20_smooth"] = dict(_G20, seeded_global_scale_max=0.20,
                                   cadence_amp=0.0, limb_phase_amp=1.80)

# ---- r1 / r2 / r3: THREE NEW CHANNELS (2026-08-13, owner brief) ----------------------------
# Each is `qsize_m10_smooth` (= q1s, the arm the owner accepted on quality) plus EXACTLY ONE new
# knob, so a difference in the result is attributable to that channel and nothing else.
#
# WHY NEW CHANNELS: §A.2w-3 measured that neither existing knob reaches the rank-5 residual --
# the size draw moves top-5 lift 0.7 pp and cadence 1.6 pp, both inside seed noise, while the
# family removes ~74 % of top-1 lift but only ~46 % of top-5. More of either cannot help.
#
# 🔴 REQUIRED BEFORE THESE MEAN ANYTHING: the three mechanisms were ported VERBATIM into
# `segbench/pose_anon_v2.py` (the copy this harness imports) and edge-vs-lab parity was asserted
# at max|delta| = 0.0 on all four cases including knobs-off. Without that port the lab would have
# silently ignored the kwargs and measured `qsize_m10_smooth` three times.
_Q1S = dict(_G20, seeded_global_scale_max=0.10, cadence_amp=0.0, limb_phase_amp=1.80)
# r1 SHAPE: randomise the aspect correction. `_apply_projection_fit` scales X by
# observed_shoulder/emitted_shoulder -- a per-clip readout of the subject's breadth that re-injects
# subject-specific bone ratios and leaves 97 % of the between-subject shape difference intact
# (§B.63b). Deleting the fit removes the leak exactly but wrecks posture (every PF=False render
# came back hunched). Randomising it keeps the correction and destroys its precision.
# PREDICTION: shape is a STATIC cue present in every frame of a tracklet, which is the kind of
# evidence a rank-5 list is built from -- so if anything moves top-5, this should.
CONFIGS["r1"] = dict(_Q1S, projection_fit_jitter=0.15)
# r2 PHASE: constant per-limb time SHIFT. `limb_phase_amp` saturates (d50 caps at 0.063) because a
# warp of a smooth track can only displace a limb so far; a shift has no such ceiling and can put a
# limb a third of a gait cycle out of phase with its contralateral partner. Costs no smoothness --
# measured jerk 1.10x real, the best of any arm (q1s 1.19x).
CONFIGS["r2"] = dict(_Q1S, limb_phase_offset_s=0.35)
# r3 RHYTHM: uniform pose re-timing. Step frequency is a strong gait cue that NOTHING in the family
# currently touches. 🔴 Measured jerk 1.68x real (q1s 1.19x) -- it re-introduces some of the
# "sudden" motion, because slowed limbs beat against a body still travelling at the real speed.
# Kept as the only probe of this channel; if it wins on privacy, find a smoother way to reach it.
CONFIGS["r3"] = dict(_Q1S, pose_rate=0.85)

# ---- d1/d2/d3 (2026-08-14): the owner shipped q1s on render quality and asked to keep it while
# improving the defence. Descriptor ablation of q1s (§A.2x, N=5375) says where the room is: the
# DYNAMICS block alone reproduces the entire untrained attack (+2.71 t1 / +6.24 t5 vs the full
# descriptor's +2.56 / +6.77), while shape (-0.20 / -0.07) and inter-limb phase (-1.09 / -0.20)
# are already AT OR BELOW chance. So these three touch time only; every static proportion the
# generator sees is bit-identical to q1s. r3 already probed this channel and barely moved it --
# because a FIXED `pose_rate` is a corpus-wide constant that the attacker's z-score removes
# entirely. Only a per-sequence DRAW adds variance, which is what actually breaks a channel.
CONFIGS["d1"] = dict(_Q1S, pose_smooth_max_s=0.08)                        # peakiness direction
CONFIGS["d2"] = dict(_Q1S, pose_rate_jitter=0.18)                         # tempo direction
CONFIGS["d3"] = dict(_Q1S, pose_smooth_max_s=0.08, pose_rate_jitter=0.18)  # both

# ---- d4/d5/d6 (2026-08-14): PUSH THE ONE KNOB THAT BOTH WORKS AND TRANSFERS. §A.2y measured the
# rate DRAW carrying the whole win (d2 +1.01 t1 vs q1s's +2.56) while the low-pass bought nothing
# alone (d1 +2.21 / +6.81), and §A.2x-5 measured that low-pass largely INERT at the phone's 10 fps
# because sigma is in seconds. A uniform re-time is frame-rate independent, so it is the only knob
# in the family whose corpus number should transfer on-device. This sweeps its magnitude to find
# where privacy saturates and where the smoothness cost starts, plus a CENTRED variant: two-sided
# gives the same draw WIDTH without only ever slowing the limbs, which is what makes them beat
# against a root still travelling at the real speed (the mechanism behind r3's 1.68x jerk).
CONFIGS["d4"] = dict(_Q1S, pose_rate_jitter=0.30)
CONFIGS["d5"] = dict(_Q1S, pose_rate_jitter=0.45)
CONFIGS["d6"] = dict(_Q1S, pose_rate_jitter=0.30, pose_rate_two_sided=True)

# ---- d2s / d2c (2026-08-14): "d2 but smoothed out", the owner's ask, two ways.
# §A.2y-5 measured d2 at +40 % jerk vs q1s at the shipped 10 fps (n=5, t=5.03) -- a real trade, not
# the free win §A.2y first reported. The cause is that `_pose_rate_warp` interpolates LINEARLY, so
# every source frame becomes a kink; the sparser the frames, the bigger the kink (12 % cost at
# 25 fps, 40 % at 10 fps). Two independent repairs, deliberately separated:
#   d2s* BLUR THE KINK -- a low-pass with sigma in FRAMES. d1/d3's sigma is in SECONDS and §A.2x-5
#        measured it degenerating to identity at 10 fps, which is why d3 never paid d2 back.
#   d2c  REMOVE THE KINK -- Catmull-Rom (C1) resampling, so no kink is created to begin with. This
#        should not damp limb swing the way a blur does. NOT assumed to win: a previous swap of
#        linear->Catmull-Rom on the CADENCE warp measured worse.
CONFIGS["d2s"] = dict(_Q1S, pose_rate_jitter=0.18, pose_smooth_max_frames=1.0)
CONFIGS["d2s2"] = dict(_Q1S, pose_rate_jitter=0.18, pose_smooth_max_frames=2.0)
# 🔴 d2c DELETED 2026-08-14 (owner). Best untrained top-1 measured (+0.70) but NEVER RENDERED
# and +46 % jerk. An unrendered arm is not a candidate -- §B.64-2.
CONFIGS["d2sc"] = dict(_Q1S, pose_rate_jitter=0.18, pose_rate_kind="catmull",
                       pose_smooth_max_frames=1.0)

# ---- e-family (2026-08-14): DYNAMICS WITHOUT RE-TIMING. Owner's live set is q1/q1s/r2/d2c; the
# d-family's whole cost is that every one of them reaches dynamics by re-timing, and §A.2y-5
# measured re-timing under the root lock at +40..46 % jerk -- a cost that survived BOTH a low-pass
# (d2s 1.47x) and a C1 resampler (d2c 1.46x), so it is inherent to re-timing itself. Swing
# AMPLITUDE scales the same speed/acceleration statistics through the OTHER factor of
# (angular rate x radius), touching no clock: no interpolation kink, no beat against the root, and
# bone lengths exactly preserved so the k-same collapse is untouched.
# `q1s` BARE, registered 2026-08-15 so the arm it is can be measured rather than inferred from its
# derivatives. It was the SHIPPED arm before e2 (owner decision 2026-08-14) yet only r1/r2/e1/e2/e3
# were ever runnable here, so every q1s number in the ledger came from a run that added something.
# Nothing about the recipe changes -- this is exactly `_Q1S`, the same dict the others build on.
CONFIGS["q1s"] = dict(_Q1S)
CONFIGS["e1"] = dict(_Q1S, limb_swing_amp=0.25)                       # the new channel, isolated
CONFIGS["e2"] = dict(_Q1S, limb_swing_amp=0.25, limb_phase_offset_s=0.35)   # + r2's phase shift
CONFIGS["e3"] = dict(_Q1S, limb_swing_amp=0.25, limb_phase_offset_s=0.35,
                     pose_rate_jitter=0.18, pose_rate_kind="catmull")  # + d2c, everything on

# ---- swing-amplitude SWEEP. §A.2z measured the rate-draw curve to be NON-MONOTONIC (0.30 twice as
# leaky as 0.18), and §P item 1 records the same shape for mask displacement, so no amplitude on
# this pipeline may be assumed monotone. 0.25 is a guess until the curve is measured.
CONFIGS["e4"] = dict(_Q1S, limb_swing_amp=0.15)
CONFIGS["e5"] = dict(_Q1S, limb_swing_amp=0.40)

# ---- f-family: the WALKING-SPEED channel, which no arm on record touches. Every previous arm locks
# the root to the true trajectory for containment, but the untrained descriptor reads speed on
# ABSOLUTE coordinates, so the subject's own pace has been passing through intact the whole time.
# Bounded by a hard px clamp so the figure cannot leave the grown mask. f1 isolates it; f2 stacks it
# on the two smooth limb knobs; f3 is the all-smooth arm -- every channel EXCEPT the re-timing that
# §A.2y-5 measured at +40..46 % jerk.
CONFIGS["f1"] = dict(_Q1S, root_speed_amp=0.20)
CONFIGS["f2"] = dict(_Q1S, root_speed_amp=0.20, limb_swing_amp=0.25)
# 🔴 f3 DELETED 2026-08-14 (owner). Won every pose-space metric and LOST THE FEET on the
# render (figure -11.1 %, feet-band alpha -30 %) -- §B.64. Do not ship root_speed_amp.

# ---- e6/e7 (2026-08-14): the swing SWEEP says top-1 improves monotonically with amplitude at ZERO
# smoothness cost (0.15 -> +2.47, 0.25 -> +2.45, 0.40 -> +2.07; jerk 0.99/0.98/1.00 x q1s), but the
# best combined arm so far (e2) pairs the PHASE shift with the middle amplitude 0.25. These pair it
# with 0.40 instead. e7 adds the root-speed channel at the CORRECTED scale-free clamp.
# Not assumed to win: stacking has been ANTI-ADDITIVE twice already (e3 worse than e2 on both
# ranks; d4 worse than d2), so this is measured, not argued.
CONFIGS["e6"] = dict(_Q1S, limb_swing_amp=0.40, limb_phase_offset_s=0.35)
CONFIGS["e7"] = dict(_Q1S, limb_swing_amp=0.40, limb_phase_offset_s=0.35, root_speed_amp=0.20)


try:
    from gaitsweep_arms import ARMS as _GS_ARMS, BINARIZE_CONF as _GS_BIN   # noqa: E402
    for _k, (_knobs, _why) in _GS_ARMS.items():
        CONFIGS["gs_" + _k] = dict(_knobs, binarize_conf=_GS_BIN)
except Exception as _e:                                          # pragma: no cover
    print(f"[TM3] gaitsweep arms not registered: {_e}")


class AnonDS(Dataset):
    """Anonymized tracklets -> two augmented views + subject-id label (SupCon)."""
    def __init__(self, tracklets, transform):
        self.items = [(int(k[0]), v.astype(np.float32)) for k, v in tracklets.items()]
        self.transform = transform

    def __len__(self):
        return len(self.items)

    def __getitem__(self, i):
        sid, arr = self.items[i]
        return self.transform(arr), sid


def build_model(graph, device):
    opt = argparse.Namespace(network_name="resgcn-n39-r8", embedding_layer_size=128,
                             use_multi_branch=False)
    model, _ = _get_model(graph, opt)
    return model.to(device)


def _get_model(graph, opt):
    args = dict(A=torch.tensor(graph.A, dtype=torch.float32, requires_grad=False),
                num_class=opt.embedding_layer_size,
                num_input=1 if not opt.use_multi_branch else 3,
                num_channel=3 if not opt.use_multi_branch else 6, parts=graph.parts)
    return ResGCNv1.create(opt.network_name, **args), args


@torch.no_grad()
def embed_eval(model, tracklets, graph, device, seq_len=60, bs=256):
    model.eval()
    keys = list(tracklets)
    emb = {}
    for i in range(0, len(keys), bs):
        ck = keys[i:i + bs]
        x = np.stack([multi_input(select_sequence_center(tracklets[k][..., :2] if tracklets[k].shape[-1] == 2
                                                          else tracklets[k], seq_len),
                                  graph.connect_joint, False).astype(np.float32) for k in ck])
        x = torch.from_numpy(x).to(device)
        x = torch.cat([x, torch.flip(x, dims=[1])], 0)
        out = model(x); b = len(ck)
        f1, f2 = torch.split(out, [b, b], 0); out = torch.mean(torch.stack([f1, f2]), 0)
        for j, k in enumerate(ck):
            emb[k] = out[j].cpu().numpy()
    return emb


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="v2-full", choices=list(CONFIGS))
    ap.add_argument("--epochs", type=int, default=200)
    ap.add_argument("--batch_size", type=int, default=128)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--eval_every", type=int, default=25)
    ap.add_argument("--seq_len", type=int, default=60)
    # Nothing in this script was ever seeded, so every run drew a different model init, shuffle
    # order and augmentation stream -- fine for one number, useless for a curve, because a 2 pp
    # run-to-run wobble is indistinguishable from a 2 pp knob effect. --seed makes a run
    # reproducible AND lets the same config be repeated at seeds 0/1/2 to get a real spread.
    # Default None reproduces the historical unseeded behaviour exactly.
    ap.add_argument("--seed", type=int, default=None)
    # --seeds 0,1,2 repeats the TRAINING at each seed inside ONE process, reusing a single
    # anonymisation pass. Legitimate because the anonymiser seed is fixed by the TRACKLET KEY for
    # historical arms, or by one frozen evaluation seed map for deployment-parity arms -- --seed
    # only touches torch/numpy/random, i.e. model init, shuffling and augmentation. Verified: the
    # anonymised arrays are bit-identical across training seeds. Anonymising the 2026-08-02
    # configs costs ~8-10 min per pass at
    # CASIA-B scale, so this saves ~2/3 of that for a 3-seed spread. NOT a disk cache: a cache
    # keyed on the config would silently serve stale arrays after an anonymiser edit, which is
    # precisely the source != tested trap that produced A.2d.
    ap.add_argument("--seeds", default=None,
                    help="comma-separated training seeds, e.g. 0,1,2 (reuses one anon pass)")
    # Repeats would otherwise overwrite TM3_<config>.json each time.
    ap.add_argument("--tag", default="", help="suffix for the output json / ckpt filenames")
    # --configs a,b,c runs SEVERAL arms in one process, reusing the single CSV load (~7 min for the
    # 820 MB of pose CSVs). The anonymisation pass is still redone per arm -- it MUST be, the config
    # is what it depends on. Each arm still gets its own --tag suffix (`<tag><CONFIG_TAG>`), its own
    # artifact json and its own jsonl line written by the SAME run_one, so a 5-arm process is
    # byte-for-byte the same set of artifacts as 5 one-arm processes. Default None => the historical
    # single-config path, unchanged.
    ap.add_argument("--configs", default=None,
                    help="comma-separated configs to run in ONE process, reusing the CSV load")
    ap.add_argument("--tags", default=None,
                    help="comma-separated tags, one per --configs entry (default: --tag for all)")
    args = ap.parse_args()
    multi = [c.strip() for c in args.configs.split(",") if c.strip()] if args.configs else None
    if multi:
        for c in multi:
            if c not in CONFIGS:
                ap.error(f"--configs: {c!r} is not a known config")
        multi_tags = ([t.strip() for t in args.tags.split(",")] if args.tags
                      else [args.tag] * len(multi))
        if len(multi_tags) != len(multi):
            ap.error("--tags must have one entry per --configs entry")
    cfg = CONFIGS[multi[0] if multi else args.config]
    if args.seed is not None:
        import random
        random.seed(args.seed)
        np.random.seed(args.seed)
        torch.manual_seed(args.seed)
        torch.cuda.manual_seed_all(args.seed)
    suffix = f"_{args.tag}" if args.tag else ""
    device = "cuda" if torch.cuda.is_available() else "cpu"
    os.makedirs(OUT_DIR, exist_ok=True)
    graph = Graph("coco")

    _run_label = ",".join(multi) if multi else args.config
    print(f"[TM3] config={_run_label} first_cfg={cfg}  epochs={args.epochs} device={device}")
    train_tr = load_tracklets(os.path.join(DATA, "casia-b_pose_train_valid.csv"),
                              split="train_valid", min_frames=args.seq_len + 1)
    test_tr = load_tracklets(os.path.join(DATA, "casia-b_pose_test.csv"),
                             split="test", min_frames=args.seq_len + 1)
    P.assert_identity_disjoint(train_tr, test_tr)

    template = A.build_template(train_tr)          # public template from TRAIN ids

    if multi:
        # Multi-arm mode: everything above (CSV load, split assertion, public template) is
        # config-independent and is reused; everything below is per-arm and is NOT.
        train_aug_m = transforms.Compose([
            MirrorPoses(0.5), FlipSequence(0.5), RandomSelectSequence(args.seq_len),
            ShuffleSequence(False), PointNoise(0.05), JointNoise(0.1),
            MultiInput(graph.connect_joint, False), ToTensor()])
        seeds_m = ([int(x) for x in args.seeds.split(",") if x.strip()] if args.seeds
                   else [args.seed])
        import gc as _gc
        for _ci, _cname in enumerate(multi):
            _cfg = CONFIGS[_cname]
            _tag = multi_tags[_ci]
            print(f"\n[TM3] ########## ARM {_ci + 1}/{len(multi)}: {_cname} tag={_tag} "
                  f"{_cfg} ##########", flush=True)
            _src = _cfg.get("scale_from")
            _seed_map = None
            if _cfg.get("deployment_postprocess"):
                _master = _cfg.get("eval_seed_map_master")
                if _master is None:
                    raise ValueError(f"{_cname}: deployment_postprocess requires "
                                     "eval_seed_map_master for reproducible TM3")
                _seed_map = A.build_eval_sequence_seed_map(
                    train_tr, test_tr, master_seed=_master)
                print(f"[TM3] deployment postprocess: raw-score zero + round(2); "
                      f"clip skipped (CASIA-B CSV has no frame dimensions); "
                      f"subject-independent eval seed map master={_master}", flush=True)
            print(f"[TM3] anonymizing train + test (per-sequence) scale_from={_src!r} ...",
                  flush=True)
            _t0 = time.time()
            a_tr = A.transform_v2(train_tr, _cfg, template, seed_mode="per_sequence",
                                  scale_from=_src, sequence_seed_map=_seed_map)
            a_te = A.transform_v2(test_tr, _cfg, template, seed_mode="per_sequence",
                                  scale_from=_src, sequence_seed_map=_seed_map)
            print(f"[TM3] anon pass {(time.time() - _t0) / 60:.1f} min", flush=True)
            _margs = argparse.Namespace(**vars(args))
            _margs.config, _margs.tag = _cname, _tag
            for _si, _seed in enumerate(seeds_m):
                if _seed is not None:
                    import random as _random
                    _random.seed(_seed); np.random.seed(_seed)
                    torch.manual_seed(_seed); torch.cuda.manual_seed_all(_seed)
                _sfx = (f"_{_tag}_s{_seed}" if _tag else f"_s{_seed}") if args.seeds \
                    else (f"_{_tag}" if _tag else "")
                print(f"[TM3] ===== {_cname} seed {_seed} ({_si + 1}/{len(seeds_m)}) =====",
                      flush=True)
                try:
                    run_one(_margs, _cfg, graph, device, a_tr, a_te, train_aug_m, _seed, _sfx)
                except Exception as _e:                      # one arm must not cost the grid
                    import traceback
                    traceback.print_exc()
                    print(f"[TM3] 🔴 {_cname} seed {_seed} FAILED: {_e}", flush=True)
            del a_tr, a_te
            _gc.collect()
            if device == "cuda":
                torch.cuda.empty_cache()
        return 0

    if cfg.get("__raw__"):
        # POSITIVE CONTROL (2026-08-07). Every TM3 number on this project has been quoted against
        # an ASSUMED 2.0 % chance floor with no same-recipe upper bound. That is exactly the gap
        # the 2026-08-03 silhouette audit closed (§A.6n-8/-9: "a plain white rectangle scored level
        # with a real silhouette" - the metric was not measuring identity, and only a positive
        # control could show it). `L0-bin` is NOT that control: it still canonicalises the skeleton
        # and binarises confidence. This arm passes the tracklets through UNTOUCHED, so the SAME
        # 100-epoch SupCon recipe, the same augmentations and the same eval protocol produce the
        # number the defended arms have to be read against.
        print("[TM3] RAW POSITIVE CONTROL - no anonymisation applied")
        anon_train, anon_test = train_tr, test_tr
    else:
        # `scale_from` selects the collapse TARGET one level above anonymize_v2's kwargs - see
        # anon_adapter.transform_v2. None => the historical shared population template.
        _src = cfg.get("scale_from")
        _seed_map = None
        if cfg.get("deployment_postprocess"):
            _master = cfg.get("eval_seed_map_master")
            if _master is None:
                raise ValueError(f"{args.config}: deployment_postprocess requires "
                                 "eval_seed_map_master for reproducible TM3")
            _seed_map = A.build_eval_sequence_seed_map(
                train_tr, test_tr, master_seed=_master)
            print(f"[TM3] deployment postprocess: raw-score zero + round(2); "
                  f"clip skipped (CASIA-B CSV has no frame dimensions); "
                  f"subject-independent eval seed map master={_master}", flush=True)
        print(f"[TM3] anonymizing train + test (per-sequence) scale_from={_src!r} ...")
        anon_train = A.transform_v2(train_tr, cfg, template, seed_mode="per_sequence",
                                    scale_from=_src, sequence_seed_map=_seed_map)
        anon_test = A.transform_v2(test_tr, cfg, template, seed_mode="per_sequence",
                                   scale_from=_src, sequence_seed_map=_seed_map)

    train_aug = transforms.Compose([
        MirrorPoses(0.5), FlipSequence(0.5), RandomSelectSequence(args.seq_len),
        ShuffleSequence(False), PointNoise(0.05), JointNoise(0.1),
        MultiInput(graph.connect_joint, False), ToTensor()])
    seeds = ([int(x) for x in args.seeds.split(",") if x.strip()] if args.seeds
             else [args.seed])
    for _si, _seed in enumerate(seeds):
        if _seed is not None:
            import random as _random
            _random.seed(_seed); np.random.seed(_seed)
            torch.manual_seed(_seed); torch.cuda.manual_seed_all(_seed)
        _sfx = f"_{args.tag}" if args.tag else ""
        if args.seeds:
            _sfx = f"_{args.tag}_s{_seed}" if args.tag else f"_s{_seed}"
        print(f"[TM3] ===== seed {_seed} ({_si + 1}/{len(seeds)}) =====", flush=True)
        run_one(args, cfg, graph, device, anon_train, anon_test, train_aug, _seed, _sfx)
    return 0


def run_one(args, cfg, graph, device, anon_train, anon_test, train_aug, seed, suffix):
    ds = AnonDS(anon_train, TwoNoiseTransform(train_aug))
    dl = DataLoader(ds, batch_size=args.batch_size, shuffle=True, num_workers=0,
                    drop_last=True, pin_memory=(device == "cuda"))
    print(f"[TM3] train tracklets={len(ds)} steps/epoch={len(dl)}")

    model = build_model(graph, device)
    crit = SupConLoss(temperature=0.07).to(device)
    opt = torch.optim.Adam(model.parameters(), lr=args.lr, weight_decay=1e-5)
    sched = torch.optim.lr_scheduler.OneCycleLR(opt, args.lr, epochs=args.epochs,
                                                steps_per_epoch=len(dl))
    scaler = torch.cuda.amp.GradScaler(enabled=(device == "cuda"))

    hist = []
    for epoch in range(1, args.epochs + 1):
        model.train(); t0 = time.time(); tot = 0.0
        for points, target in dl:
            points = torch.cat([points[0], points[1]], 0).to(device, non_blocking=True)
            labels = target.to(device, non_blocking=True)
            bsz = labels.shape[0]
            with torch.cuda.amp.autocast(enabled=(device == "cuda")):
                feats = model(points)
                f1, f2 = torch.split(feats, [bsz, bsz], 0)
                feats = torch.cat([f1.unsqueeze(1), f2.unsqueeze(1)], 1)
                loss = crit(feats, labels)
            scaler.scale(loss).backward(); scaler.step(opt); sched.step()
            scaler.update(); opt.zero_grad()
            tot += loss.item()
        if epoch % args.eval_every == 0 or epoch == args.epochs:
            emb = embed_eval(model, anon_test, graph, device, args.seq_len)
            r = P.canonical_casia_b_rank1(emb)
            hist.append({"epoch": epoch, "loss": tot / len(dl),
                         "NM": r["NM#5-6"], "BG": r["BG#1-2"], "CL": r["CL#1-2"]})
            print(f"[TM3] ep{epoch:3d} loss={tot/len(dl):.3f} "
                  f"anon-test rank1 NM={r['NM#5-6']*100:.1f} BG={r['BG#1-2']*100:.1f} "
                  f"CL={r['CL#1-2']*100:.1f}  ({time.time()-t0:.1f}s/ep)", flush=True)

    ckpt = os.path.join(DATA, f"tm3_adv_{args.config}{suffix}.pth")
    torch.save({"model": model.state_dict(), "config": args.config, "cfg": cfg}, ckpt)
    final = hist[-1]
    out = {"threat": "TM3_adaptive_retrain", "config": args.config, "cfg": cfg,
           "epochs": args.epochs, "seed": seed, "tag": args.tag,
           "n_train_tracklets": len(ds), "n_test_tracklets": len(anon_test),
           "final": final, "history": hist,
           "chance_rank1": P.CHANCE_RANK1_50}
    with open(os.path.join(OUT_DIR, f"TM3_{args.config}{suffix}.json"), "w") as f:
        json.dump(out, f, indent=2)
    print(f"\n[TM3] FINAL adaptive adversary on anon-{args.config}: "
          f"NM={final['NM']*100:.1f} BG={final['BG']*100:.1f} CL={final['CL']*100:.1f} "
          f"(chance {P.CHANCE_RANK1_50*100:.1f}%)")
    print(f"[TM3] wrote reports/reid/TM3_{args.config}{suffix}.json ; ckpt {os.path.basename(ckpt)}")

    # Append-only run log. The per-run json is the artifact of record, but a grid of 15 runs is
    # only readable as one table, and a crashed/overwritten json must not silently erase a
    # measurement that actually happened. One line per completed run, never rewritten.
    with open(os.path.join(OUT_DIR, "TM3_RUNS.jsonl"), "a") as f:
        f.write(json.dumps({"config": args.config, "tag": args.tag, "seed": seed,
                            "epochs": args.epochs, "cfg": cfg,
                            "NM": final["NM"], "BG": final["BG"], "CL": final["CL"],
                            "loss": final["loss"], "chance_rank1": P.CHANCE_RANK1_50,
                            "n_train": len(ds), "n_test": len(anon_test)}) + "\n")

    # Release the model/optimiser before the next seed. With --seeds the process now spans
    # several trainings, so without this each one's GPU allocation would accumulate -- and this
    # machine has already lost one 9-run session to memory exhaustion.
    del model, crit, opt, sched, scaler, dl, ds
    import gc
    gc.collect()
    if device == "cuda":
        torch.cuda.empty_cache()


if __name__ == "__main__":
    main()
