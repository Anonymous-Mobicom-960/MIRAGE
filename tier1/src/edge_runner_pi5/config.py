# MIRAGE Tier-1 (Raspberry Pi 5) config - EDIT PATHS FOR YOUR DEVICE.
import os

# Pre-rename compatibility: every knob was SITARA_* before 2026-08-29, and configurations
# recorded under those names must keep working or a rename becomes a reproducibility break.
# Fill in each MIRAGE_* from its SITARA_* twin unless the new name is already set.
for _k in list(os.environ):
    if _k.startswith('SITARA_'):
        os.environ.setdefault('MIRAGE_' + _k[len('SITARA_'):], os.environ[_k])
from pathlib import Path
BASE   = Path(__file__).resolve().parent
MODELS = BASE / "models"          # put the onnx models here (see README)
YOLO11N   = str(MODELS / "yolo11n.onnx")           # export int8 for the real device
RTMPOSE_T = str(MODELS / "rtmpose-t-wholebody.onnx")
RVM_ONNX  = str(MODELS / "rvm_mobilenetv3_fp32.onnx")
INPUT_VIDEO = str(BASE / "input.mp4")   # or pass --source 0 for the camera
OUT_DIR   = str(BASE / "out_tier1")

# ------------------------------ resolution ------------------------------
# NATIVE-RESOLUTION policy (privacy + quality): the captured frame is KEPT at its own (H,W).
# Detection / pose / matte / face all run on a WORK_RES letterbox COPY (cheap inference); the
# binary person mask is upscaled WORK_RES->native and gray-fill happens on the NATIVE frame.
# RATIONALE: the RGB background NEVER round-trips through WORK_RES - only the binary mask is
# upscaled, so the emitted masked_video/mask keep native detail.
WORK_RES = 640                    # inference resolution (the letterbox copy fed to all 4 models).
                                  # Knob: raise toward native where compute allows, e.g. Meta Ray-Ban SoC.
OUT_RES  = None                   # output video resolution. None = NATIVE source size (do NOT force 640).
                                  # Set to a (w, h) tuple ONLY to force a specific emit size
                                  # (the mask is resized with INTER_NEAREST so it stays strictly binary).
FRAME_SIZE = 640                  # DEPRECATED (pre-native-res): superseded by WORK_RES (inference) +
                                  # OUT_RES (output). Kept only for reference / back-compat.

# ------------------------------ emit fps --------------------------------
EMIT_FPS = float(os.environ.get("MIRAGE_EMIT_FPS", 30))
                                  # emit rate. The pipeline contract is 30 fps in -> 30 fps out, so
                                  # the default emits every frame of a 30 fps source (stride = 1);
                                  # if source fps is unknown (live camera) every frame is emitted.
                                  # ⚠️ OVERRIDE VIA ENV, NOT BY EDITING THIS LINE. Every _frames()
                                  # value below (MASK_TEMPORAL_WIN, SUBJECT_LOCK_FRAMES, TRACK_TTL,
                                  # …) is derived from EMIT_FPS **at import time**, so setting
                                  # `C.EMIT_FPS` AFTER importing config silently leaves all of
                                  # them at their old values - the mask window would not follow.
                                  # `MIRAGE_EMIT_FPS` is applied before any of them are computed,
                                  # so the whole config stays self-consistent.
                                  # Cloud hand-off: the mask stream must stay at the same rate as
                                  # its video or the two desynchronise (ledger §P P1).

# ------------------------------ detection / pose / matte ----------------
DET_THRESH = 0.5

# §2 RECALL - detect on NATIVE-resolution tiles as well as the downscaled work copy.
# Detection runs on letterbox(frame, WORK_RES), so at 1264 native a person is shrunk 1.97x
# before YOLO sees them, and EVERY §2 mechanism downstream can only iterate boxes the
# detector returned. Measured 2026-07-23: detection rate 15% at 0.31% of frame, 5% at 0.08%,
# with the bystander 100% OUTSIDE the emitted mask on 47/60 frames. Tiling makes the
# smallest detectable person a fixed fraction of NATIVE pixels, independent of frame size
# and aspect ratio. Tile count is DERIVED (ceil(H/WORK_RES) x ceil(W/WORK_RES)), not tuned.
DETECT_TILED = False              # ⚠️ OFF since 2026-07-25 - it was DEAD CODE, not a safety net.
                                  # `det_s2_extra` (the native-res tiled pass) has exactly ONE
                                  # consumer: the box-guarantee fill loop, which `BOX_GUARANTEE_FILL`
                                  # has gated OFF since 2026-07-24. PROVEN by a bit-exact A/B on
                                  # d01_bystander4 (the bystander clip, i.e. tiling's best case),
                                  # 60 frames, identical config but this flag: **0 differing mask
                                  # pixels** (ledger §A.1l). So it cost detector passes per frame and
                                  # emitted nothing. ⚠️ RE-ENABLE THIS TOGETHER WITH
                                  # `BOX_GUARANTEE_FILL=True` - tiling is what reaches distant
                                  # bystanders (§A.1c), and it only reaches them THROUGH that fill.
DETECT_TILE_OVERLAP = 0.20      # so a person on a tile seam is whole in at least one tile
DETECT_TILE_PX = 640            # tile-grid target = the DETECTOR'S fixed onnx input (640x640), NOT
                               # WORK_RES. Tiling exists to feed the 640 detector NATIVE-scale crops;
                               # keying the grid off 640 (not WORK_RES) keeps tiling effective even if
                               # WORK_RES is raised toward native. Measured 2026-07-24: raising WORK_RES
                               # does NOT help small-person detection (onnx input is a hard 640), and
                               # the old ceil(H/WORK_RES) grid silently disabled tiling at WORK_RES=native.
# Recall-first, matching the justification already used for SEG_CONF=0.10: a missed person
# is a §2 reveal, a false box only paints extra grey. Tiles are the recall pass, so they run
# looser than the full-frame pass.
DET_THRESH_TILE = 0.25
# How fine the tile grid is, as a multiple of the no-downscale grid. MEASURED REACH of the default
# (ceil(H/WORK_RES) x ceil(W/WORK_RES), i.e. 2x2 at 1264 native), on ground-truth composites:
#     1.2 % of frame : 11 -> 3  frames fully uncovered
#    0.34 % of frame : 37 -> 19 frames fully uncovered, median outside 1.000 -> 0.000
#   0.084 % of frame : 51 -> 51 frames fully uncovered  <-- NO IMPROVEMENT
# The floor is structural, not a threshold: 2x2 tiling puts a 51 px person at ~51 px in the
# detector's 640 input, which is below what YOLO11n resolves. A finer grid does buy reach, and the
# cost is steep -- MEASURED on that 51 px bystander (20 frames, laptop CPU):
#     scale 1 ( 4 tiles): found on  8/20 frames   100 ms/frame
#     scale 2 (16 tiles): found on 10/20 frames   407 ms/frame
#     scale 3 (36 tiles): found on 15/20 frames   919 ms/frame
# So 9x the compute buys 8->15 of 20 frames and still does NOT fully protect them. Whether that is
# worth paying is a per-device decision (it is plainly unaffordable on a Pi5 already at ~3 FPS),
# which is why this is a KNOB with its curve stated rather than a default someone silently pays.
# HONEST LIMIT: below roughly 0.1 % of frame area, Tier-1 does not reliably protect a bystander at
# any setting here. Closing it needs a HIGHER-INPUT DETECTOR MODEL (the onnx input is a hard 640,
# so raising WORK_RES does NOT help - measured 2026-07-24, detection flat across WORK_RES 640/960/
# 1264) or finer tiling (DETECT_TILE_GRID_SCALE). NOT WORK_RES.
DETECT_TILE_GRID_SCALE = 1
POSE_THRESH = 0.5
RVM_DOWNSAMPLE = 0.25             # RVM speed/quality knob (lower = faster on Pi)
BINARIZE = 0.5                    # alpha -> binary (Manager hardening: no soft-edge bg bleed)
EDGE_EXPAND = 1                   # EXPAND (dilate) the person silhouette OUTWARD at NATIVE res so the real
                                  # person is FULLY covered by gray - matches the V8.5 cloud sim (#182
                                  # expand=1). USER (2026-07-18): keep this the SAME as V8.5 - too MUCH
                                  # expand -> hazy/grey halo around the output character; too LITTLE ->
                                  # the real person shows through at points. NOT a cloud leak: gray-fill
                                  # grays this margin, so Tier-2's 2A ships GRAY (never real background).
                                  # (Replaces the earlier inward EDGE_ERODE, which UNDER-covered the person.)
MAX_PEOPLE = 3                    # per-frame EMIT budget for pose/face (the lowest MAX_PEOPLE slots,
                                  # i.e. the leftmost people). Not a privacy cap: the §2 mask net
                                  # uses the FULL untruncated detector output.
                                  # ⚠️ THIS COMMENT USED TO SAY "NEVER a privacy cap" FULL STOP, AND
                                  # THAT WAS FALSE. It was true of the BOX net and false of the
                                  # head-keypoint FACE guarantee, which was built inside the pose
                                  # loop over the TRUNCATED box list - so person #4+ got the box net
                                  # but not the face guarantee, the very mechanism that exists
                                  # because a whole-box mean cannot catch a partially-dropped head.
                                  # Fixed 2026-07-23 (ledger §A.1d): every detection now gets a head
                                  # guarantee. Keep it that way - if you add a §2 mechanism, drive it
                                  # from det_all_native/det_s2_extra, never from `boxes`.

# ------------------------------ auto person count + person slots --------
# Tier-1 derives the clip's person count from ITS OWN detector stream (never a hardcoded N) and
# assigns every detection to a STABLE left-to-right slot, so the whole-clip per-person stages
# (pose anonymization, face-scalar DP) each see one real person's trajectory. See person_slots.py
# (module docstring = the authoritative policy) and pose.json's "person_count"/"slot_policy" meta.
# ⚠️ THESE ARE DURATIONS, STORED AS FRAME COUNTS - derive them from EMIT_FPS, never re-type them.
# Every value below is documented as a TIME ("~0.5 s at EMIT_FPS=15") but was stored as a frame
# count fitted to EMIT_FPS=15. EMIT_FPS is an explicit knob, so changing it silently changed all of
# these to the wrong duration - DP_WINDOW's own comment conceded the coupling ("was 5 @ 10 fps - 
# Revisit if EMIT_FPS changes") without handling it. Deriving them keeps the documented semantics
# true at any frame rate, which is what the project's generalisation rule requires.
def _frames(seconds, lo=1):
    return max(int(lo), int(round(float(seconds) * float(EMIT_FPS))))


PCOUNT_SUSTAIN_S = 0.55           # a detection count must hold this long to register as real
SLOT_HYSTERESIS_S = 0.35          # how long an inverted slot order must persist before committing
PREV_BOX_MAX_AGE_S = 0.20         # how long a stale detector box may be carried through a dropout
MASK_TEMPORAL_S = 0.14            # silhouette running-max window -> 2 frames @15fps (user 2026-07-25:
                                  # lowered again 3f->2f to further cut the visible temporal trail, after
                                  # visual review of the guided output. HISTORY: 0.35/5f -> 0.20/3f
                                  # (2026-07-24) -> 0.14/2f (2026-07-25).
                                  # ⚠️ PRIVACY COST, measured (§A.6c, CASIA-B, chance 2.0 %): silhouette
                                  # NM re-ID 62.27 % @win5 -> 71.44 % @win3 -> 88.66 % @win1. win=2 has
                                  # NO measured value yet - it sits between 71.44 and 88.66 and MUST be
                                  # measured on the CASIA-B silhouettes (still on disk at
                                  # tier1_lab/reports/reid/CASIA-B Dataset/) before any paper number is
                                  # quoted for the shipped config. Tracked in the ledger §P.
DP_WINDOW_S = 0.47                # face-scalar smoothing window

PCOUNT_SUSTAIN_FRAMES = _frames(PCOUNT_SUSTAIN_S)     # 8 at EMIT_FPS=15 (unchanged from the fitted value)
                                  # A per-frame detection count of n only registers as "n people in
                                  # this clip" once it holds for this many CONSECUTIVE frames.
                                  # Rejects 1-2 frame spurious blobs /
                                  # double-detections while still counting a real person who only
                                  # walks through briefly. Clips shorter than the window use their
                                  # own length. Raise if the detector is noisy, lower to catch
                                  # very brief entrants.
SLOT_HYSTERESIS_FRAMES = _frames(SLOT_HYSTERESIS_S)   # 5 at EMIT_FPS=15
                                  # A slot renumbering (people appearing to
                                  # cross) is only COMMITTED after the inverted order persists for
                                  # this many consecutive frames - so detector x-jitter around a
                                  # near-crossing cannot flip slot identity frame-to-frame (which
                                  # would splice two people's pose/face streams together). Higher =
                                  # stickier slots but a real crossing takes longer to register.
SLOT_SWAP_MARGIN_FRAC = 0.10      # …unless the inverted pair is separated by more than this
                                  # fraction of the FRAME WIDTH (10% = an unambiguous, already-
                                  # completed crossing), which commits immediately without waiting
                                  # out the hysteresis. Fraction-of-width, never absolute px, so it
                                  # transfers across resolutions/clips.
# DETECTION-BACKED §2 fallback: if the RVM matte covers < COVER_MIN of a detected person's bbox, gray
# the WHOLE bbox so the person is NEVER revealed even when RVM drops them (measured failure on real
# footage: full reveal during a person's entrance). Over-covers (box shape) only on matte-failure frames.

# Persistent-track association (person_slots.SlotTracker). These were hardcoded DEFAULTS in
# SlotTracker.__init__ and never passed at the construction site, so they were invisible to
# anyone tuning the system - and track_ttl is another duration stored as a frame count.
TRACK_GATE_FRAC = 0.15            # max x-movement between frames, as a fraction of FRAME WIDTH,
                                  # for a detection to associate with an existing track. Fraction,
                                  # never absolute px, so it transfers across resolutions.
TRACK_TTL_S = 0.55                # how long a track survives without being matched
TRACK_TTL = _frames(TRACK_TTL_S)  # 8 at EMIT_FPS=15
COVER_MIN = 0.40
# Horizontal bands the COVER_MIN test is evaluated over. A single box-wide mean can be satisfied by
# an OVERLAPPING person's mask pixels, leaving the second person uncovered; per-band means make
# that impossible to hide. min(bands) <= mean(box), so this only ever fires MORE often.
COVER_BANDS = 4
# user 2026-07-24: REMOVED the COVER_MIN box-guarantee fallback FILL (it painted a hard box/ellipse
# over a person the seg dropped -> destructive). Set True to restore. ⚠️ PRIVACY: with this OFF, a
# person the guided seg drops (e.g. white clothing on a bright bg) is covered ONLY by seg + FaceGuard,
# so a seg-drop can partially reveal them. The person-shaped alternative is temporal mask carry.
BOX_GUARANTEE_FILL = False
# Head share of a person box, for the §2 face guarantee on detections the pose stage does not
# reach (beyond MAX_PEOPLE). ANATOMICAL ratio (head ~1/7-1/8 of standing height), with
# margin for seated/partial boxes - not a value fitted to any clip.
HEAD_FRAC_OF_BOX = 0.30
# §2 box-guarantee hardening (adversary review C1/C3/D5):
GUARANTEE_MARGIN = None           # px margin added around a filled guarantee box so a box that clips
                                  # hair/fingertips still covers the real edge. None => EDGE_EXPAND+2.
PREV_BOX_MAX_AGE = _frames(PREV_BOX_MAX_AGE_S)        # 3 at EMIT_FPS=15
                                  # carry stale detector boxes for at most this many consecutive
                                  # detection-gap frames, then EXPIRE - so a walking person's OLD
                                  # position isn't grayed for long, and no permanent phantom gray
                                  # rectangle is left after a person leaves frame.
GRAY = 128                        # solid-gray silhouette fill value

# ---- native-res mask upscale (WORK_RES binary mask -> native) ----
# The binary mask is upsampled with an edge-aware guided filter (cv2.ximgproc.guidedFilter, using
# the NATIVE gray frame as the guide) so the mask edge snaps to the real image edge, THEN it is
# re-binarized + eroded at native res. Requires opencv-contrib-python (cv2.ximgproc); if that module
# is absent the code falls back to a plain cv2.resize(INTER_LINEAR) upsample automatically.
GUIDED_RADIUS = 8                 # guidedFilter window radius (native px) - tune on device.
GUIDED_EPS    = 500.0             # guidedFilter regularization (guide is 8-bit, so eps ~ intensity^2).

# ------------------------------ model swaps (measure-then-adopt) --------
# READY-TO-ENABLE alternatives. Defaults reproduce the current, working onnx+rvm path EXACTLY.
# Flip a flag ONLY after measuring latency+quality on the real Pi5 (see README swap notes).
DETECT_BACKEND = "onnx"           # 'onnx' (default, working) | 'ncnn' (YOLO11n int8 via NCNN - measure first)
MATTE_MODEL    = "rvm"            # 'rvm'  (default, working) | 'pphumanseg' (PP-HumanSegV2-Lite - measure first)
SEG_BACKEND    = "yolo11n"        # NOW CANONICAL (DECISION.md guided-seg winner, 2026-07-24): union a
                                  #         person-SHAPED yolo11n-seg silhouette (§2-COMPLETE, verified path)
                                  #         edge-aligned by the guided post below. 'none' = legacy RVM +
                                  #         head-kp + detection-box fallback only.
                                  # ⚠️ §2 completeness: the 'none' path guarantees the FACE/identity every
                                  # frame (head-kp region) + catches TOTAL matte drops (box-guarantee), but
                                  # a whole-box-mean gate cannot catch a thin BODY-EDGE strip on a rare
                                  # motion-blur frame where RVM covers >COVER_MIN of the box yet misses an
                                  # edge sliver (measured: 4_bystander f388). Only the person-SHAPED seg
                                  # closes that (the local eval that passed 0 reveals on all 7 clips uses
                                  # seg). yolo11n-seg is HEAVY on a Pi5 CPU (person-detect + proto decode)
                                  # -> MEASURE latency on device (LD2 15fps) before making it the default;
                                  # enable where compute allows (e.g. Meta Ray-Ban SoC). Residual on
                                  # 'none' is a NON-identifying body sliver, not a face leak.
SEG_CONF       = 0.15             # yolo11n-seg person confidence (user 2026-07-24: raised a bit from
                                  # 0.10 to cut low-conf spurious blobs/specks). At 0.25 seg drops a
                                  # motion-blurred person => reveal; 0.15 is a modest step. LOW=recall.
# ---- seg backend FORMAT, per device: 'onnx' (PC/x86 local + S25 Ultra INT8) | 'ncnn' (Pi5 fp16, ~2x) ----
# DECISION.md §C: Pi5 NCNN-fp16 ~2x ONNX (INT8 is SLOWER than fp32 on the A76); S25 = ONNX-INT8 (NNAPI
# rejects seg's Split op). LOCAL-FIRST testing runs 'onnx' (YOLO11nSeg). 'ncnn' needs YOLO11nSegNCNN +
# the exported models/yolo11n-seg_ncnn_* blobs.
SEG_FORMAT     = "onnx"           # 'onnx' (default, portable) | 'ncnn' (Pi5 canonical)
# ---- guided-seg mask post = the tier1_lab/reports/DECISION.md WINNER (0/900 face-leaks, 98.5% cov) ----
# The person-shaped yolo11n-seg silhouette is edge-aligned to the native person boundary with a guided
# filter (radius 12, eps 0.01 on a [0,1]-normalized guide), morph-closed (k=5), then dilated OUTWARD d4 px.
# Applied to the SEG component ONLY; the RVM/box/head-kp union and EDGE_EXPAND (anti-halo, user-set) stay
# unchanged - the seg carries its own d4 privacy margin.
SEG_GUIDED         = True
SEG_GUIDED_RADIUS  = 12
SEG_GUIDED_EPS     = 0.01
SEG_GUIDED_CLOSE   = 5
SEG_GUIDED_DILATE  = 4            # d4 - d0/1/2 leak; d4 is the first to pass 0 leaks (DECISION.md §6)
# ---- FaceGuard (DECISION.md "the key privacy piece"): MediaPipe FaceDetection full-range, expanded-
# ellipse fill of every detected face, UNIONed into the §2 mask. Reuses the MediaPipe already loaded for
# the 12 face scalars, so ~no extra model cost. ----
FACE_GUARD         = True
FACE_GUARD_CONF    = 0.40
FACE_GUARD_EXPAND  = 0.20         # face ellipse = box * 1.20; clamped to seg + FACE_GUARD_CLAMP_PX so it
                                  # covers a face-edge sliver without bulging into a floating oval
                                  # (user 2026-07-24; A/B verify showed 0.2 vs 0.3 identical on coverage).
FACE_GUARD_CLAMP_PX = 15          # FaceGuard may extend at most this many px past the seg silhouette.
FACE_GUARD_MODEL   = 1            # mediapipe model_selection: 1 = full-range (far/profile/small faces)
POSE_CKPT      = RTMPOSE_T        # pose model path knob (swap in a different RTMPose ckpt here)
# Extra model files needed ONLY if you flip the flags above (not required for the defaults):
# NCNN export paths - these are the file names ULTRALYTICS actually produces:
#   python -c "from ultralytics import YOLO; YOLO('models/yolo11n.pt').export(format='ncnn', imgsz=640)"
# writes models/yolo11n_ncnn_model/{model.ncnn.param, model.ncnn.bin}. (The previous
# yolo11n.param / yolo11n.bin names matched no exporter output.) Blob names verified from the
# generated .param: input "in0", output "out0" - exactly what YOLO11nNCNN assumes.
YOLO11N_NCNN_PARAM = str(MODELS / "yolo11n_ncnn_model" / "model.ncnn.param")
YOLO11N_NCNN_BIN   = str(MODELS / "yolo11n_ncnn_model" / "model.ncnn.bin")
PPHUMANSEG_ONNX    = str(MODELS / "pphumansegv2_lite.onnx")   # PP-HumanSegV2-Lite export
YOLO11N_SEG_ONNX   = str(MODELS / "yolo11n-seg.onnx")         # yolo11n-seg export (SEG_FORMAT='onnx', default)
# Pi5 SEG_FORMAT='ncnn' path (fp16, ~2x ONNX on the A76): the .param/.bin of an ncnn export at the same
# imgsz. Export: `yolo export model=yolo11n-seg.pt format=ncnn imgsz=256 half=True`. ⚠️ confirm the blob
# names on device (netron model.param) - YOLO11nSegNCNN assumes in0/out0/out1. Only needed if SEG_FORMAT='ncnn'.
YOLO11N_SEG_NCNN_PARAM = str(MODELS / "yolo11n-seg_ncnn_256" / "model.ncnn.param")
YOLO11N_SEG_NCNN_BIN   = str(MODELS / "yolo11n-seg_ncnn_256" / "model.ncnn.bin")

# ---------------- Anti-re-ID pose anonymizer (whole-clip, before pose.json emit) ----------------
# Lab-validated (evaluation/privacy/gait: GaitGraph2 adversary on CASIA-B): the emitted body-17
# keypoints carry gait + limb-proportion biometrics - re-identifiable even with the face landmarks
# zeroed. pose_anon_edge.py collapses every skeleton's proportions to a baked POPULATION template
# (k-same; rescaled by the clip's own auto-detected body scale, so on-frame size/position are kept
# for the composite) and perturbs the dynamic gait channel per the lab Pareto ladder. Seed is fresh
# per-clip OS randomness (secrets) - NEVER identity/content-derived, never emitted.
POSE_ANON_ON = True               # default ON - raw gait must not leave the device (§2).
POSE_ANON_LEVEL = os.environ.get("MIRAGE_POSE_ANON_LEVEL", "L4")
                                  # 'off' | 'L0' (canon only) | 'L2' | 'L4'. L4 = the lab operating
                                  # point (cadence .60, angle 20° const + 15° drift, asymmetry .12);
                                  # keep unless a new adversary eval says otherwise.
                                  # ⚠️ THE LEVEL DOES NOT TOUCH THE SKELETON'S GEOMETRY. `do_canon`
                                  # is passed True at every level (pose_anon_edge.py), so L4→L2→L0
                                  # only weakens the DYNAMIC gait channel (cadence/angle/asymmetry).
                                  # The static collapse that sets limb proportions -- and therefore
                                  # where the hips land, which §B.23 measured as the driver of the
                                  # hallucinated hands -- is identical at L0..L4 and only disappears
                                  # at 'off'. Env override added 2026-07-26 to sweep this without
                                  # editing config between runs.
POSE_CONF_BINARIZE = True         # emit every per-joint confidence as {0,1} at POSE_THRESH - the raw
                                  # confidence trace is an identity side-channel (lab strip_conf).

# ---------------- Crop-aware joint gating (§B.23/§B.24 - the hallucinated-hands fix) -------------
# 🔴 THE PROBLEM, MEASURED. On a chest-up clip the real right elbow is visible in 0 % of frames,
# the wrists in 0 % and the hips in 0 % - yet Tier-1 emitted ELBOWS in 100/100 frames and HIPS in
# 84/100. The anti-reID collapse rebuilds every bone at a TEMPLATE length along its measured
# direction, and a template upper arm (0.795 x shoulder) is shorter than this subject's real one, so
# the elbow is dragged from below the crop UP INTO frame. The conditioning image then shows an arm
# ending in mid-picture and the generator completes it into a hand - 98-100 % of frames (§B.23).
#
# WHY THE EXISTING GUARD DOES NOT CATCH IT: pose_anon_edge already drops joints with confidence
# < 0.5, but RTMPose-t returns CONFIDENT predictions for joints outside a top-down crop. A
# confidence gate cannot detect a confidently-wrong joint.
#
# WHY NOT JUST WEAKEN THE ANONYMISER: measured and ruled out (§B.24). L4/L2/L0 are geometrically
# IDENTICAL (do_canon runs at every level; spine spread 3.2 %), so lowering the level cannot fix
# this and would cost +5.5 to +20.8 pp to the adaptive adversary for nothing. 'off' moves the
# geometry the WRONG way (wrist-in-frame 10 % -> 91 %) and ships raw gait.
#
# THE GATE: a below-shoulder joint is emitted only if it lands on the person's OWN emitted mask - 
# already computed per frame and the ground truth for what the camera actually saw. Head, neck and
# shoulders are never gated. PRIVACY: strictly information-REMOVING, so it cannot weaken §2 or the
# anti-reID posture; it only stops the pipeline asserting body parts that were never observed.
# GENERALISES: the tolerance is a fraction of the silhouette's OWN scale (sqrt of its area), never a
# fitted pixel constant, so it adapts to subject size and resolution per clip.
CROP_AWARE_JOINTS = os.environ.get("MIRAGE_CROP_AWARE_JOINTS", "0") not in ("0", "false", "False")
                                  # ⚠️ DEFAULT OFF - pixel-affecting, awaiting a render + sign-off.
CROP_GATE_DILATE_FRAC = 0.02      # tolerance as a fraction of sqrt(mask area): a joint merely NEAR
                                  # the silhouette edge survives; one well outside it does not.
CROP_GATE_MIN_IDX = 7             # gate COCO-17 joints >= this (elbows/wrists/hips/knees/ankles).
                                  # 0-6 = nose/eyes/ears/shoulders are never gated: the head is the
                                  # one thing always in shot, and gating shoulders would fight the
                                  # anonymiser's own deliberate displacement.
# 🔴 MEASURED 2026-07-26 - CROP_AWARE_JOINTS DOES NOT REMOVE THE DANGLING TERMINUS, and its
# decisions are a random draw. Two runs of the IDENTICAL config on a10's clip zeroed completely
# different joints (run A: Relb 60x, Lelb 0x; run B: Lelb 100x, Relb 5x) and the max-per-arm
# dangling rate went 100 % -> 100 % and 96 % -> 79 %. Three reasons, all structural:
#   (a) the test "the joint lands on the person's mask" is vacuous on a TRUNCATED framing. On a10
#       the silhouette touches the bottom frame edge in 100/100 frames and covers 26.3 % of it;
#       the horizontal mask run-length through the emitted elbow's own row is 0.52 x W, i.e. ~1.5x
#       the shoulder span. Any invented below-shoulder joint lands on the chest and passes.
#   (b) the mask it tests against is not the observed silhouette: it is EDGE_EXPAND-dilated, then
#       temporal running-max'd, then MASK_DISPLACE-bulged OUTWARD by up to MASK_DISPLACE_AMP_FRAC
#       (0.25) of the component's mean radius with a per-clip CSPRNG phase field, and the gate then
#       dilates it again by CROP_GATE_DILATE_FRAC. Every one of those steps is one-directional
#       (the emitted mask is a strict SUPERSET of the detected one, by design), so the test is
#       systematically permissive.
#   (c) it is a PER-JOINT test with no notion of the kinematic chain - and zeroing a wrist while
#       keeping its elbow is what CREATES a dangling terminus. A per-joint gate cannot fix a
#       property of a chain. That is what POSE_FREE_END_PRUNE below is for.
# Left in place (default OFF) because its stated goal - not asserting joints off the person - is
# still sound; it is simply not the fix for a limb that ends in open space.

# ---------- Open-chain FREE-END prune (the dangling-terminus fix, 2026-07-26) ----------------
# THE DEFECT. `pose_sticks_*.mp4` is the conditioning image the cloud generator actually sees
# (build_cloud_bundle renders it from THIS artifact). A bone drawn from an observed shoulder to an
# elbow that is the last drawn point INSIDE the picture shows the generator a limb that stops dead
# in open space, and a pose-conditioned model completes it. Measured on a10: real elbows/wrists/
# hips visible in 0/100 frames (MediaPipe, visibility > 0.5 AND in frame) while Tier-1 emits elbows
# 100/100 and hips 84-86/100, giving a max-per-arm dangling rate of 96-100 %.
#
# THE RULE, and why it is not another heuristic: a joint is dropped only when ALL THREE hold - 
#   (1) it is an INTERMEDIATE joint of an open limb chain (elbow, knee, hip). Wrists and ankles are
#       anatomical termini and are never dropped; head and shoulders are never touched.
#   (2) its distal child is NOT emitted, so nothing continues the chain past it, and
#   (3) it lies STRICTLY INSIDE the frame. A joint OUTSIDE the frame is kept deliberately: the bone
#       then exits the canvas, which is the honest "the body continues out of shot" cue and exactly
#       what the frame_wh=None decision above exists to preserve.
# Applied distal-first (ankle -> knee -> hip) so one pass settles a whole chain.
#
# GENERALISES: pure graph + frame-bounds logic. No threshold, no tolerance, no measured constant,
# nothing derived from this clip. It is a NO-OP on any clip where the distal joints are observed - 
# a full-body walking shot emits wrists and ankles, so nothing is ever dropped there.
# PRIVACY: strictly information-REMOVING (joints are only ever zeroed, to the pipeline's existing
# (0,0) "not detected" marker), so it cannot weaken §2 and cannot weaken the anti-reID posture.
# `do_canon` stays on and the anonymiser is untouched - this changes WHICH joints are emitted, not
# whether the geometry is anonymised.
POSE_FREE_END_PRUNE = os.environ.get("MIRAGE_FREE_END_PRUNE", "1") not in ("0", "false", "False")
# ✅ DEFAULT ON since 2026-07-26 - user signed off after the before/after render
# (`_e2e/A100_20260726/_jitter/STICKS_FINAL.png`: the shipped column shows bones ending in coloured
# dots mid-picture at f45/f80; the pruned column shows only bones that exit the canvas).
# Measured at sign-off: dangling 96/84 % → 0.0 % on a10, and a NO-OP-sized 10 joints/200 f on
# d09 and 16 joints/302 f on d04, i.e. it fires only on frames that genuinely dangle.
# 🔴 THE HIP IS DELIBERATELY EXCLUDED, and this was measured before deciding. A hip whose knee is
# absent LOOKS like the same defect, but it is not: the hip is a corner of the TORSO QUAD (the
# emitted COCO skeleton carries 11-12, 5-11 and 6-12), so it is not a free end of the skeleton - 
# only of the OpenPose-18 *render* graph, which happens to omit the pelvis bone. Including it was
# measured on `d09_driving_video` (a waist-up clip whose hips ARE in shot but whose knees are not,
# 200 person-frames): it removed 192/200 left hips and 197/200 right hips, i.e. it deleted the
# whole torso from the conditioning image on a clip where the torso is genuinely visible - a real
# loss, for zero movement on the arm dangling-terminus rate this exists to fix. Turn it on only for
# a deliberate A/B; the honest fix for the render-graph free end is renderer-side, not here.
POSE_FREE_END_INCLUDE_HIPS = os.environ.get("MIRAGE_FREE_END_HIPS", "0") not in ("0", "false", "False")

# ---------------- Silhouette mitigation (mask.mkv / masked_video.mkv shape channel) --------------
# INFORMATION-REDUCTION mitigation, NOT a validated silhouette-reID defense (no lab adversary eval,
# unlike the pose anonymizer): temporal running-max over MASK_TEMPORAL_WIN frames + contour
# simplification (approxPolyDP at MASK_SIMPLIFY_EPS of each contour's perimeter) on the DILATED
# mask, then UNION with the per-frame original - so the emitted mask ⊇ the detected mask on EVERY
# frame (§2 coverage can only GROW) and the EDGE_EXPAND margin is untouched.
# ---------------- Background-person scope (USER DECISION 2026-07-25) --------------------------
# ⚠️ THIS IS A DELIBERATE PRIVACY SCOPE REDUCTION, NOT A BUG FIX. Distant background people are
# NOT masked at all. The user reviewed the d01_bystander4 / d04_final_full output on 2026-07-25
# and chose to exclude them ("better to ignore that person too") rather than mask them
# consistently. Consequence, stated plainly so it can never be quoted away: a background person
# whose FACE IS RESOLVABLE is knowingly left unmasked, so any "0 reveals" figure measured with
# this ON means "0 reveals AMONG IN-SCOPE PEOPLE", and the paper must state the exclusion rule.
# Set MASK_IGNORE_BACKGROUND=False to restore full coverage.
#
# The threshold is RELATIVE, never a fitted constant (project rule: generalize, never hardcode a
# measured value). Measured on d04_final_full 2026-07-25, the component-area distribution is
# strongly BIMODAL - subject 19.5-32.8 % of frame, background people 0.28-1.53 %, a ~13x gap with
# nothing between, stable over all 302 frames. So a person is dropped only when BOTH hold:
#   (a) its area < MASK_IGNORE_REL x the LARGEST person component in that frame  (auto-scales), and
#   (b) its area < MASK_IGNORE_ABS_FRAC of the frame                             (absolute guard)
# Both conditions => a 2-person clip keeps both, a 5-face crowd keeps all five, and a clip whose
# only subject is small is not silently emptied.
MASK_IGNORE_BACKGROUND = True
MASK_IGNORE_REL        = 0.10     # < 10 % of the largest person's area
MASK_IGNORE_ABS_FRAC   = 0.02     # AND < 2 % of the frame

# ---------------- SUBJECT LOCK (user decision 2026-07-25) --------------------------------------
# "The person at the START of the video is the subject; never detect, mask or count anyone who
# appears later." Enrollment runs over the first SUBJECT_LOCK_S seconds and binds to the
# PERSISTENT track ids from person_slots.SlotTracker (not positional slots, which renumber). After
# the window closes, a track id that was never enrolled contributes NOTHING: no mask pixels, no
# person_count, no slot, no pose, no face scalars.
#
# ⚠️ THIS IS THE WIDEST PRIVACY SCOPE REDUCTION IN THE PIPELINE, and it is deliberate. A person who
# walks in after enrollment is left FULLY VISIBLE no matter how close or large they become - this
# is strictly broader than the size rule above, which at least kept large people masked. It exists
# because an unenrolled person reaching the CLOUD tier causes a real failure: the cloud runs its
# multi-person WanAnimate branch and can generate a character for a bystander or place one at the
# wrong position. Excluding them from person_count/slots fixes that; the user additionally chose
# to leave them unmasked. Any "0 reveals" measured with SUBJECT_LOCK on means "0 reveals AMONG
# ENROLLED SUBJECTS" and the paper must state the enrollment rule.
# Set SUBJECT_LOCK=False to restore full multi-person coverage.
SUBJECT_LOCK   = True
SUBJECT_LOCK_S = 1.0              # enrollment window from clip start (seconds)
# Enrollment is COMMITTED at the end of the window, not per-frame, and a candidate must pass BOTH
# tests below. Rationale: accepting every track seen in the window means a clip that OPENS on a
# passer-by enrolls the passer-by, and the real subject arriving a second later is refused and ships
# FULLY UNMASKED - the worst failure this feature can have, and a normal smart-glasses case (you
# start recording before the subject settles). Both thresholds are RELATIVE to what this clip
# actually contains, so nothing is fitted to one video.
SUBJECT_LOCK_MIN_REL    = 0.25    # must reach >=25 % of the largest person seen during the window
SUBJECT_LOCK_MIN_FRAMES = 3       # and be present on >=3 window frames (rejects 1-frame spurious IDs)
SUBJECT_LOCK_FRAMES = _frames(SUBJECT_LOCK_S)         # 15 at EMIT_FPS=15

# ---- SYNTHETIC BODY (2026-08-01, user proposal). Replace the subject's own outline with a
# CANONICAL human figure posed by their REAL joints: same build for every subject, real motion.
# See synth_body.py for the full rationale, the leak list and the §2 argument. OFF by default;
# it runs BEFORE mask_mitigate, and its output is a superset of the mask it replaces, so every
# existing shape mode still composes on top (set MIRAGE_MASK_SHAPE_MODE=none to see it bare).
MASK_SYNTH_BODY       = os.environ.get("MIRAGE_SYNTH_BODY", "0") not in ("0", "", "false", "off")
SYNTH_SCALE_QUANT     = float(os.environ.get("MIRAGE_SYNTH_SCALE_QUANT", 0.10))
                                  # log2 quantisation step of the per-clip body scale (~7.2 % buckets)
SYNTH_ALPHA           = float(os.environ.get("MIRAGE_SYNTH_ALPHA", 1.0))
                                  # BUILD multiplier on the canonical widths. A population
                                  # constant - never fitted to a subject. Coverage is bought with
                                  # SYNTH_GROW instead (see synth_body.py for why the first
                                  # version's multiplicative coverage knob was the wrong tool).
SYNTH_GROW_STEP       = float(os.environ.get("MIRAGE_SYNTH_GROW_STEP", 0.04))
SYNTH_GROW_MAX        = float(os.environ.get("MIRAGE_SYNTH_GROW_MAX", 0.60))
                                  # additive coverage margin, in units of shoulder width
SYNTH_GROW_TOL        = float(os.environ.get("MIRAGE_SYNTH_GROW_TOL", 0.002))
                                  # residual (fraction of the real mask) the margin is allowed to
                                  # leave to the §2 backstop. NOT a reveal - the backstop still
                                  # covers it; it is the fraction of the emitted boundary that is
                                  # the subject's own outline. MEASURED frontier (§A.6l):
                                  #   A_corridor  tol 0 -> 1.888x area | tol 0.002 -> 1.407x
                                  #   B_atrium    tol 0 -> 1.354x      | tol 0.002 -> 1.246x
                                  # Set 0.0 for a purely synthetic boundary at full area cost.
SYNTH_POSE_THRESH     = float(os.environ.get("MIRAGE_SYNTH_POSE_THRESH", 0.35))
SYNTH_BACKSTOP        = os.environ.get("MIRAGE_SYNTH_BACKSTOP", "1") not in ("0", "false", "off")
                                  # OR the uncovered residual back in. Turning this OFF makes the
                                  # emitted mask PURELY synthetic and can produce a §2 REVEAL - 
                                  # it exists only to MEASURE what the synthetic body misses.
SYNTH_DEBUG_DIR       = os.environ.get("MIRAGE_SYNTH_DEBUG_DIR", "") or None
                                  # dump the PURE synthetic body (pre-backstop) per frame
# ---- v2 knobs (2026-08-01, after the 108-clip corpus measurement, ledger §A.6l-2) -------------
SYNTH_JITTER          = float(os.environ.get("MIRAGE_SYNTH_JITTER", 0.06))
                                  # per-CLIP wobble of the canonical build, ±fraction. Keyed to
                                  # the clip seed, NEVER to identity or content, so the same
                                  # subject filmed twice gets two different builds. 0 = off.
SYNTH_HEAD_MODE       = os.environ.get("MIRAGE_SYNTH_HEAD_MODE", "hair")
                                  # "hair" = canonical head as a FLOOR, outline follows the
                                  #   subject's own head+hair low-passed to 3 harmonics.
                                  #   🔴 DELIBERATE privacy reduction (user direction): it hands
                                  #   back a coarse hair silhouette to keep the gender cue.
                                  # "canon" = identity-free canonical ellipse (v1 behaviour).
SYNTH_WORK_RES        = int(os.environ.get("MIRAGE_SYNTH_WORK_RES", 512))
                                  # draw/close/distance-transform at this long side, then upsample
                                  # and OR with the FULL-RES real mask (§2 unaffected). 0 = native.
SYNTH_GROW_WIN        = int(os.environ.get("MIRAGE_SYNTH_GROW_WIN", 60))
SYNTH_GROW_PCT        = float(os.environ.get("MIRAGE_SYNTH_GROW_PCT", 0.90))
SYNTH_FALLBACK        = os.environ.get("MIRAGE_SYNTH_FALLBACK", "radiallp")
SYNTH_FALLBACK_FRAC   = float(os.environ.get("MIRAGE_SYNTH_FALLBACK_FRAC", 0.02))
SYNTH_FALLBACK_KEEP   = int(os.environ.get("MIRAGE_SYNTH_FALLBACK_KEEP", 4))
SYNTH_MAX_BODY        = float(os.environ.get("MIRAGE_SYNTH_MAX_BODY", 2.5))
                                  # the drawn figure must be at most this many times the real
                                  # mask, else the pose was confident but wrong and the frame
                                  # falls back (measured: without it, 4.07x mean area on
                                  # p10_c01 where every other arm sits at 1.47x).
                                  # POSE-FREE FALLBACK. The body can only be drawn where the pose
                                  # model succeeds; on the corpus several clips annotate a DISTANT
                                  # person behind foliage and every body-joint score comes back at
                                  # 0.10-0.21, so NO figure was drawn and the backstop emitted the
                                  # subject's own outline verbatim (73 % of the mask on
                                  # p11_c02_head). "radiallp" defends those frames anyway; "none"
                                  # restores the pass-through. FRAC also triggers it when a figure
                                  # WAS drawn but still misses more than that fraction of the mask.
                                  # the margin is the GROW_PCT quantile of the last GROW_WIN
                                  # frames' requirements, not a running max. The running max let
                                  # one bad frame pin a whole clip at the ceiling: 41.7 % of the
                                  # 108 corpus clips ended at grow_max and area hit 2.674x mean.

MASK_ANON_ON = True
MASK_TEMPORAL_WIN = int(os.environ.get("MIRAGE_MASK_TEMPORAL_WIN", 0)) or _frames(MASK_TEMPORAL_S)          # 2 @15fps, 1 @10fps (MASK_TEMPORAL_S=0.14).
# 🔴 The SHIPPED decision (ledger, user 2026-07-25) is win=2 as a FRAME COUNT, worth -9.41 pp of
# silhouette re-ID (88.66 -> 79.25 % NM). Because this is derived from a DURATION, a cloud-bound run
# at EMIT_FPS=10 silently resolves to 1 and gives that back. Pin it with MIRAGE_MASK_TEMPORAL_WIN.
                                  # running-max window in frames.
# SHAPE CANONICALISATION of the emitted silhouette (2026-07-25). "none" = the historic behaviour
# (contour simplification only). "hull" = convex hull per person component. "ellipse" = fitted,
# inflated ellipse -- a canonical blob preserving only position and scale. Both destroy limb
# articulation and the body-width profile, which is what the CASIA-B gait adversary reads.
# MASK-ONLY by design: the silhouette adversary consumes bare binary masks, so a pose-driven
# canonical body could not be evaluated against it. Privacy/area cost: MEASURED, see ledger.
# 🔴 DEFAULT CHANGED 2026-08-08: "displace" -> "bbox". SOURCE != SHIPPED was live here.
# Every recent runner sets MIRAGE_MASK_SHAPE_MODE=bbox explicitly and calls it "the shipped
# silhouette mode" (e.g. `_e2e/run3_20260807/build_run3.py:50`), while this default still handed
# `displace` to anyone who did NOT set it - including any fresh checkout, any Pi5 bring-up, and any
# eval that forgot the env var. §A.6o re-measured both against each arm's OWN null: raw silhouette
# carries **+31.88 pp** of re-ID lift; **bbox +7.16** (78 % of the lift removed, and the frozen gait
# model lands AT CHANCE), **displace +29.24** (8 % removed - i.e. effectively no protection). So the
# default was the WORST mode measured, and it was the one a careless run would get.
# Found 2026-08-08 by the §2 face audit (FIX10), which noticed source and documentation disagreeing.
MASK_SHAPE_MODE = os.environ.get("MIRAGE_MASK_SHAPE_MODE", "bbox")
                                  # "none" | "hull" | "ellipse" | "displace" | "bbox"
                                  # Env-overridable 2026-07-27, same pattern as MIRAGE_ANGLE_CONST:
                                  # §A.6g measured hull at 20.33 % NM / area ×1.470 vs the shipped
                                  # displace 0.25's 55.03 % / ×1.358 - hull STRICTLY PARETO-DOMINATES
                                  # on privacy-per-area. What is NOT measured is what a convex blob
                                  # does to Tier-2 inpaint quality, so the override exists to let a
                                  # render answer that WITHOUT editing this file (an uncommitted edit
                                  # is exactly how the angle knobs silently diverged from what shipped).
                                  # SHIPPED "displace" 2026-07-25 at the user's direction.
                                  # §2 SAFE (outward-only; superset verified: 0 violation px).
                                  # ✅ PRIVACY NOW MEASURED 2026-07-26 (ledger §A.6e), CASIA-B,
                                  # 50 identity-disjoint subjects, 3298 probes, gate 0.35 pp:
                                  #   control raw_bin        96.55 % NM
                                  #   win2, shape=none       79.25 %  (−17.30 pp)
                                  #   win2 + displace 0.25   55.03 %  (−41.52 pp)  <= SHIPPED
                                  # The shape term contributes −24.22 pp BEYOND the trail.
                                  # Supersedes the old "quote 79.25 %" note: that figure is the
                                  # temporal union WITHOUT the shape term and understates ship.
# ---- "bbox" knobs (added 2026-08-03 at the user's direction: "fill the whole bounding box with
# the mask, full rectangle rather than the person's silhouette"). NOT the shipped mode; NOT yet
# measured against the CASIA-B silhouette adversary. It emits the axis-aligned bounding rectangle
# of each person component, so it keeps position + extent and destroys everything else -- and it
# is the most §2-conservative mode in the file (a rect is a superset of its own contour before
# mask_mitigate's `| sm | cur` is applied at all). Its cost is AREA, which is the cloud's
# inpainting bill; measure it per run rather than assuming a ratio.
MASK_BBOX_MERGE = os.environ.get("MIRAGE_MASK_BBOX_MERGE", "1") not in ("0", "false", "False")
                                  # merge rects that INTERSECT (one person broken into blobs by
                                  # occlusion / the erode margin -> one rect). Non-intersecting
                                  # rects are left alone, so two separate people are never fused.
MASK_BBOX_PAD_FRAC = float(os.environ.get("MIRAGE_MASK_BBOX_PAD", 0.0))
                                  # optional outward pad as a FRACTION of each rect's own w/h
                                  # (never absolute px -- the no-fitted-constants rule). 0.0 =
                                  # the exact bounding box, which is the default and what shipped
                                  # in the 2026-08-03 p11_c08 run.
# "displace" knobs: outward-only boundary perturbation (the §2-safe form of the published
# silhouette-deformation method). AMP is the max outward bulge in px; HARMONICS sets how many
# angular frequencies compose the field (more = less regular, harder to invert). The phase and
# seed are set PER CLIP by mirage_tier1 (never derived from identity or content).
MASK_DISPLACE_AMP_FRAC  = float(os.environ.get("MIRAGE_MASK_DISPLACE_AMP", 0.25))
                                  # max outward bulge as a FRACTION of the component's mean
                                  # radius (resolution- and distance-independent by design)
                                  # 0.15 -> 0.25 on 2026-07-26 at the user's direction
                                  # ("lets use displace 0.25 and do evals later").
                                  # MEASURED Tier-2 COST (d07, 100 f @1264^2, win 2,
                                  # WINDOW_INPAINT_COST_SHAPE025b_2026-07-25.json):
                                  #   none@2      hole 24.47 % | core 14.239 % | blob 9.138 %
                                  #   displace .15 hole 27.10 % | core 15.873 % | blob 9.996 %
                                  #   displace .25 hole 29.03 % | core 16.821 % | blob 10.497 %
                                  # so .15 -> .25 costs +1.93 pp hole, +0.95 pp never-revealed
                                  # core, +0.50 pp largest blob. For scale, the whole win 2->5
                                  # trail (worth -16.98 pp of silhouette re-ID) costs +0.42 pp
                                  # of core, so this step costs 2.3x that trade's core.
                                  # ✅ THE PRIVACY HALF IS NOW MEASURED (2026-07-26, §A.6e).
                                  # CASIA-B, 50 identity-disjoint subjects, 3298 probes, all
                                  # amplitudes in ONE embedding pass, control raw_bin 96.55 %:
                                  #   none   79.25 %  (−17.30 pp)   area 1.1313  IoU 0.8876
                                  #   0.05   85.35 %  (−11.19 pp)   area 1.1218  IoU 0.8943
                                  #   0.15   66.05 %  (−30.50 pp)   area 1.2491  IoU 0.8032
                                  #   0.25   55.03 %  (−41.52 pp)   area 1.3579  IoU 0.7388
                                  #   0.35   45.42 %  (−51.13 pp)   area 1.4691  IoU 0.6829
                                  # 0 superset-violation px at EVERY amplitude (446 551 frames).
                                  # 🔴 NON-MONOTONIC AT THE HEAD: 0.05 is WORSE THAN NO
                                  # DISPLACEMENT AT ALL (+6.11 pp to the attacker) while still
                                  # costing area. NEVER ship a small amplitude - a weak
                                  # perturbation regularises the contour without destroying the
                                  # shape cue. Any new amplitude must be measured, not
                                  # interpolated, until the inversion point is bracketed.

# HEAD-vs-BODY AMPLITUDE SPLIT (2026-07-26) - OFF BY DEFAULT, pending measurement + user sign-off.
# Rationale: the torso and limbs carry nearly all of the silhouette's gait signal and want a large
# bulge, but the head is small and is the first thing a viewer looks at, so the same bulge there
# reads as a deformed skull. Setting AMP_HEAD below AMP_FRAC spends the perturbation where it buys
# privacy and spares the part that costs perceived quality.
#
# Set MASK_DISPLACE_AMP_HEAD to None for the shipped uniform behaviour (bit-identical to before);
# set it to a float to enable the split. The head band is the top HEAD_FRAC of each component's OWN
# bounding-box height - a PROPORTION of the person, so it transfers between a 64x64 CASIA-B crop
# and a native 1264^2 mask with no pixel constant. Derived from the MASK ALONE (never the pose), so
# it remains evaluable against the CASIA-B silhouette adversary.
#
# ✅ NOW MEASURED (2026-07-26, ledger §A.6f) - this supersedes the old "DO NOT ENABLE, NOT MEASURED"
# warning that used to sit here. The head/body arms went through the CASIA-B harness:
#   head 0.10 / body 0.25   58.49 % NM   (+3.46 pp WORSE than shipped - the head carries real signal)
#   head 0.10 / body 0.30   54.73 % NM   (statistically the SHIPPED privacy, at a third the head
#                                         amplitude, for +2.0 % area: 1.3845 vs 1.3579)
#   uniform 0.30            50.13 % NM   (−4.90 pp BETTER than shipped, for +4.1 % area)
# So sparing the head is NOT free - it must be PAID FOR on the body. `head 0.10 / body 0.30` is the
# configuration that buys a gentler head at today's privacy; it is a real option, not a hazard.
# ⚠️ STILL TRUE: never set a low amplitude UNIFORMLY - §A.6g pins the harmful inversion below ~0.10.
MASK_DISPLACE_AMP_HEAD  = (lambda v: None if v is None or str(v).strip().lower() in ("", "none")
                           else float(v))(os.environ.get("MIRAGE_MASK_DISPLACE_AMP_HEAD"))
                                  # None = uniform (SHIPPED). Set to e.g. 0.10 together with
                                  # MIRAGE_MASK_DISPLACE_AMP=0.30 for the measured split above.
MASK_DISPLACE_HEAD_FRAC = 0.15    # head band as a fraction of the component's bbox height.
                                  # Population anthropometry (head ≈ 13 % of stature, the same
                                  # constant as pose_anon_edge._ANATOMICAL_OVER_SHOULDER["face"]),
                                  # rounded up slightly so the neck falls inside the blend zone.
MASK_DISPLACE_HEAD_BLEND = 0.10   # linear ramp head->body over the next 10 % of height. A hard
                                  # step would leave a visible ledge at the neck.
MASK_DISPLACE_UPRIGHT_MIN = 1.2   # only taper when bbox height/width >= this. "Top of the bbox is
                                  # the head" is false for a person lying down, a torso-only sliver
                                  # at the frame edge, or two people merged into one blob; those
                                  # fall back to the uniform amplitude instead of perturbing
                                  # whatever happens to be at the top.
MASK_DISPLACE_HARMONICS = 3
MASK_DISPLACE_PHASE_STEP = float(os.environ.get("MIRAGE_MASK_PHASE_STEP", 0.35))
                                  # phase advance per EMITTED frame. This is what perturbs walking
                                  # RHYTHM (as opposed to static shape); 0 disables the temporal half.
MASK_DISPLACE_RESEED_PHASE = float(os.environ.get("MIRAGE_MASK_RESEED_PHASE", 0.0))
                                  # 2026-07-31. Re-draw the displacement field's coefficients every
                                  # this many radians of accumulated phase (with PHASE_STEP 0.35,
                                  # 3.5 = every 10 emitted frames). 0.0 = never = SHIPPED, and the
                                  # code path is bit-identical in that case (epoch is always 0).
                                  # Costs ZERO area - only the field's direction changes - which is
                                  # why it is the kind of lever worth having under an area budget.
# RUNTIME state, SET PER CLIP / PER FRAME by mirage_tier1.main() -- declared here so the config
# contract test can see them. They were previously read via getattr defaults and never set, so
# the shipped pipeline ran with phase=0 (NO walking-rhythm perturbation -- half the published
# method's effect missing) and seed=0 (the SAME field on every clip, i.e. a fixed learnable
# transform -- the exact 'stable seed = linkable pseudo-identity' failure the pose anonymiser
# avoids). Caught by test_config_contract.py on 2026-07-25.
_MASK_DISPLACE_SEED  = 0          # overwritten per clip from the OS CSPRNG
_MASK_DISPLACE_PHASE = 0.0        # overwritten per emitted frame
MASK_ELLIPSE_INFLATE = 1.15       # ellipse mode: inflate so the blob contains what it replaces

# ---- 2026-07-31: two ADDITIVE non-convex shape modes (hull/ellipse are banned by user decision).
# Both are OFF unless MASK_SHAPE_MODE names them. ⚠️ This line used to end "the shipped default is
# still 'displace'" - that was true and is no longer: the default is **"bbox"** as of 2026-08-08
# (§A.6o; see the note at MASK_SHAPE_MODE). Corrected here because a stale comment asserting the old
# default is exactly how source and documentation drifted apart in the first place.
# "radiallp"  outward-only radial low-pass: resample the boundary radius r(theta) about the
#             component centroid onto a uniform angular grid (max per bin), keep only the DC term
#             + MASK_RADIALLP_KEEP harmonics, then take max(r_binned, r_smoothed). Kills the
#             body-width/limb-articulation profile a gait recogniser reads WITHOUT going convex.
# "close"     morphological close per component with an ELLIPSE kernel sized as a fraction of that
#             component's own equivalent radius sqrt(area/pi). Fills inter-limb gaps.
# Composable with the existing modes via "+", e.g. MASK_SHAPE_MODE="displace+close".
# NO PIXEL CONSTANTS in either: the radial grid is angular and the kernel is component-relative,
# so both are the same proportion of the person on a 64x64 CASIA-B crop and a native 1264^2 mask.
MASK_RADIALLP_KEEP  = int(os.environ.get("MIRAGE_MASK_RADIALLP_KEEP", 4))
                                  # angular harmonics retained. 0 = a pure disc of the mean radius
                                  # (max'd with the real profile, so still outward-only); higher
                                  # keeps more of the person's real outline.
MASK_RADIALLP_BINS  = int(os.environ.get("MIRAGE_MASK_RADIALLP_BINS", 180))
                                  # angular resolution of the resample (2 deg). Resolution-
                                  # independent by construction -- it is an ANGLE grid, not pixels.
MASK_CLOSE_KERNEL_FRAC = float(os.environ.get("MIRAGE_MASK_CLOSE_FRAC", 0.25))
                                  # close kernel radius as a fraction of the component's own
                                  # equivalent radius sqrt(area/pi).

# ---- 2026-07-31: "mounds" mode - travelling raised-cosine swellings of the boundary radius
# profile ("goo wiggling on the body", the user's accepted geometry; supersedes the rejected
# disc "blobs" - discs notch the outline and barely change body WIDTH, which is what a
# silhouette recogniser reads). Composable, e.g. MASK_SHAPE_MODE="radiallp+mounds". OFF unless
# MASK_SHAPE_MODE names it. All randomness from the per-CLIP seed (never identity/content).
# NO PIXEL CONSTANTS: height is a fraction of the component's mean radius, width is an ANGLE.
MASK_MOUND_N        = int(os.environ.get("MIRAGE_MASK_MOUND_N", 4))
                                  # concurrent mounds per person component.
MASK_MOUND_HEIGHT   = float(os.environ.get("MIRAGE_MASK_MOUND_HEIGHT", 0.13))
                                  # peak swelling as a fraction of the component's mean radius
                                  # (before the per-mound amp_k in [0.5,1] jitter).
MASK_MOUND_WIDTH_RAD = float(os.environ.get("MIRAGE_MASK_MOUND_WIDTH", 0.55))
                                  # angular half-span of a mound (radians): narrow = local
                                  # blister, wide = broad swelling.
MASK_MOUND_DRIFT    = float(os.environ.get("MIRAGE_MASK_MOUND_DRIFT", 0.020))
                                  # centre drift scale in rad/EMITTED-frame (the "goo" wiggle);
                                  # per-mound speed = ±U[0.5,1]*this. 0 = static mounds.
                                  # ⚠️ frame-based like MASK_TEMPORAL_WIN: EMIT_FPS-coupled.

# ---- 2026-07-31: "bands" mode (user-specified) - random horizontal bands, per-band amplitude.
# The component is cut into N random horizontal sections (random boundaries, fractions of the
# component's OWN bbox height - no pixel constants) and each band runs its own displacement
# amplitude, all drawn from the per-CLIP seed. Composable: e.g. MASK_SHAPE_MODE="bands+blobs".
# OFF unless MASK_SHAPE_MODE names it. Rationale: GEI reads width-as-a-function-of-height; a
# per-clip random band profile corrupts it UNPREDICTABLY (a gallery-adapting attacker cannot
# replicate a layout they have never seen - the property the deterministic modes lack).
MASK_BAND_N        = (int(os.environ.get("MIRAGE_MASK_BAND_N_LO", 4)),
                      int(os.environ.get("MIRAGE_MASK_BAND_N_HI", 7)))
                                  # inclusive range for the random band count. 4-7: at 3 it
                                  # degenerates toward the fixed head/body split (measured
                                  # 0.00 pp); at 8 thin bands' blend zones overlap and the
                                  # profile averages back toward uniform.
MASK_BAND_AMP      = (float(os.environ.get("MIRAGE_MASK_BAND_AMP_LO", 0.10)),
                      float(os.environ.get("MIRAGE_MASK_BAND_AMP_HI", 0.40)))
                                  # per-band amplitude range (fraction of mean radius, same
                                  # units as MASK_DISPLACE_AMP_FRAC). Default [0.10,0.40]:
                                  # mean 0.25 == shipped, so any gain is attributable to
                                  # VARIATION alone, not magnitude/area. ⚠️ the 0.10 lower
                                  # edge borders the known non-monotonic dead zone (§A.6e:
                                  # 0.05 scored worse than nothing) - see the band_matched
                                  # lab arm before narrowing.
# ---- RANDOM DIRECTIONAL BANDS (2026-07-31, user proposal; ledger A.6k-8) --------------------
# MASK_SHAPE_MODE="dirbands". Same band machinery as "bands", but the band coordinate is the
# projection onto a PER-CLIP RANDOM AXIS u = x*sin(theta) + y*cos(theta) instead of height.
# The adversary reads the body as 16 HORIZONTAL strips, so horizontal bands hand each strip one
# consistent amplitude -- a clean offset, which is what a recogniser normalises away. Angled
# bands cut ACROSS strips, so each strip gets a MIXTURE and its width estimate is BLURRED
# rather than displaced. theta is per clip, so the banding axis itself is unpredictable.
#
# MEASURED: 56.07 % -> 30.84 % top-1, -25.23 pp, McNemar p < 0.0001, area 1.302x
# (108 clips / 107 probes, GEI vs a DEFENDED gallery). Best arm in the sweep that fits the
# strict 1.358x area ceiling, at an area BELOW the shipped mask's own 1.337x.
#
# OFF unless MASK_SHAPE_MODE names it -- adding these knobs changes nothing by itself.
MASK_DIRBAND_N     = (int(os.environ.get("MIRAGE_MASK_DIRBAND_N_LO", 4)),
                      int(os.environ.get("MIRAGE_MASK_DIRBAND_N_HI", 7)))
MASK_DIRBAND_AMP   = (float(os.environ.get("MIRAGE_MASK_DIRBAND_AMP_LO", 0.10)),
                      float(os.environ.get("MIRAGE_MASK_DIRBAND_AMP_HI", 0.30)))
                                  # the measured 30.84 % configuration. Tuning sweep running.
# ---- COMPACT: pull the perturbed outline back toward its centre (2026-08-01, user idea) ----
# Every shape mode pushes the outline OUTWARD and the added area is the cloud's inpainting
# bill. COMPACT contracts the perturbed polygon toward its own centroid before it is filled.
# Safe by construction: mask_mitigate OR-s the result with the real person mask, so the
# contraction can only remove ADDED area, never true-person area -- S2 is untouched.
#
# Not equivalent to simply lowering the amplitude: lowering amplitude shrinks the angular
# VARIATION too, and A.6k-1 measured that variation (not magnitude) is what buys privacy.
# A contraction preserves the variation pattern and only reduces overall size.
MASK_COMPACT_SCALE  = float(os.environ.get("MIRAGE_MASK_COMPACT_SCALE", 1.0))
                                  # 1.0 = OFF. Fixed contraction factor, e.g. 0.90.
MASK_COMPACT_TARGET = (lambda v: None if v is None or str(v).strip().lower() in ("", "none")
                       else float(v))(os.environ.get("MIRAGE_MASK_COMPACT_TARGET"))
                                  # target emitted-area ratio (e.g. 1.30). Solved per FRAME by
                                  # bisection, so each frame lands ON the budget rather than
                                  # averaging there -- the privacy/area frontier is convex
                                  # (increasing returns), so that should beat spending the mean.
                                  # Takes precedence over MASK_COMPACT_SCALE. None = OFF.
MASK_COMPACT_ITERS  = int(os.environ.get("MIRAGE_MASK_COMPACT_ITERS", 6))
                                  # bisection steps; each costs one extra fill+OR per frame.

MASK_DIRBAND_UPRIGHT_MIN = float(os.environ.get("MIRAGE_MASK_DIRBAND_UPRIGHT_MIN", 0.0))
                                  # 0.0 = NO upright guard, deliberately. "bands" needs one
                                  # because it bands by HEIGHT; dirbands bands along a RANDOM
                                  # AXIS, so no orientation is privileged. Measured: the
                                  # inherited h/w<1.2 guard fired on 45.6 % of components on
                                  # our corpus (median h/w 1.22) and caused the 5-17 pp
                                  # lab-vs-shipped gap in ledger A.6k-13.
MASK_DIRBAND_THETA = (lambda v: None if v is None or str(v).strip().lower() in ("", "none")
                      else float(v))(os.environ.get("MIRAGE_MASK_DIRBAND_THETA"))
                                  # None => band axis drawn per clip (the measured config).
                                  # Setting a fixed angle makes the layout DETERMINISTIC, which
                                  # the band_static control measured at -0.93 pp, p=1.0 -- i.e.
                                  # essentially nothing. For debugging only, never for shipping.

MASK_BAND_AMP_STEP = float(os.environ.get("MIRAGE_MASK_BAND_AMP_STEP", 0.0025))
                                  # amplitude quantisation step inside MASK_BAND_AMP.
MASK_BAND_ALPHA    = float(os.environ.get("MIRAGE_MASK_BAND_ALPHA", 2.5))
                                  # Dirichlet concentration for the random band WIDTHS: random
                                  # fractions summing to 1 (bands always tile the body);
                                  # 2.5 = meaningfully uneven without degenerate slivers.
MASK_BAND_MIN_W    = float(os.environ.get("MIRAGE_MASK_BAND_MIN_W", 0.10))
                                  # minimum band width (fraction of bbox height); the width
                                  # draw is resampled until satisfied - a sliver band cannot
                                  # carry an amplitude difference, it just becomes a ledge.
MASK_BAND_BLEND_FRAC = float(os.environ.get("MIRAGE_MASK_BAND_BLEND", 0.25))
                                  # blend zone at each band edge as a fraction of the NARROWER
                                  # neighbour (hard steps = visible ledge = inpaint hazard and
                                  # a potential fingerprint of its own).

# ---- "ksame": k-SAME SILHOUETTE COLLAPSE (2026-07-31). The mask analogue of the pose
# anonymiser's `_TEMPLATE_RATIOS`. Every person's boundary is pushed OUT to a shared population
# profile: r_out(theta) = max(r_person(theta), s * T(theta)), with s the person's OWN mean radius,
# so the template carries shape but never size -> no pixel constant, transfers between a 64x64
# crop and a native 1264^2 mask. Outward-only, so §2 holds by construction.
#
# PROVENANCE of MASK_KSAME_TEMPLATE: per-angular-bin percentile of the mean-radius-normalised
# boundary profile over CASIA-B **TRAIN** identities 001..074 ONLY (the 50 held-out TEST ids the
# re-ID eval scores on are never read). Rebuild with
#   evaluation/privacy/silhouette/derive_ksame_template.py
# ⚠️ DOMAIN CAVEAT, stated rather than hidden: CASIA-B is a full-body walking corpus, while our own
# footage is a chest-up subject at 1264^2. The MECHANISM is domain-free but this TEMPLATE is not - 
# re-derive it on target-domain masks before shipping ksame, or the area cost will not match what
# the lab measured.
MASK_KSAME_SCALE = float(os.environ.get("MIRAGE_MASK_KSAME_SCALE", 1.0))
                                  # multiplies the template after it is scaled to the person.
                                  # >1 collapses harder and costs more area; <1 the reverse.
MASK_KSAME_TEMPLATE = (
    0.5824, 0.5817, 0.5824, 0.5929, 0.6108, 0.6269, 0.6405, 0.6531, 0.6656, 0.6777, 0.6913,
    0.7046, 0.7181, 0.7303, 0.7452, 0.7589, 0.7751, 0.7910, 0.8075, 0.8253, 0.8426, 0.8668,
    0.8939, 0.9122, 0.9369, 0.9658, 0.9978, 1.0323, 1.0710, 1.1141, 1.1616, 1.2153, 1.2732,
    1.3367, 1.4016, 1.4636, 1.5166, 1.5642, 1.6124, 1.6802, 1.8187, 2.0979, 2.3009, 2.3927,
    2.4230, 2.4160, 2.3719, 2.2800, 2.1326, 1.9426, 1.8055, 1.7225, 1.6695, 1.6241, 1.5778,
    1.5280, 1.4733, 1.4125, 1.3479, 1.2863, 1.2288, 1.1767, 1.1297, 1.0864, 1.0460, 1.0081,
    0.9744, 0.9457, 0.9130, 0.8822, 0.8574, 0.8325, 0.8103, 0.7888, 0.7682, 0.7503, 0.7316,
    0.7166, 0.7021, 0.6879, 0.6748, 0.6625, 0.6506, 0.6403, 0.6281, 0.6139, 0.5975, 0.5880,
    0.5910, 0.5947, 0.5854, 0.5651, 0.5490, 0.5405, 0.5366, 0.5330, 0.5286, 0.5243, 0.5210,
    0.5179, 0.5159, 0.5144, 0.5144, 0.5154, 0.5170, 0.5200, 0.5246, 0.5302, 0.5378, 0.5479,
    0.5590, 0.5726, 0.5902, 0.6062, 0.6274, 0.6508, 0.6773, 0.7087, 0.7466, 0.7892, 0.8397,
    0.9009, 0.9736, 1.0594, 1.1522, 1.2431, 1.3382, 1.4448, 1.5591, 1.6732, 1.7771, 1.8654,
    1.9344, 1.9767, 2.0043, 2.0275, 2.0365, 2.0382, 2.0078, 1.9294, 1.8256, 1.7043, 1.5594,
    1.4303, 1.3234, 1.2360, 1.1571, 1.0831, 1.0150, 0.9537, 0.8977, 0.8485, 0.8032, 0.7642,
    0.7306, 0.7021, 0.6758, 0.6555, 0.6332, 0.6150, 0.5989, 0.5839, 0.5712, 0.5604, 0.5509,
    0.5429, 0.5367, 0.5318, 0.5286, 0.5268, 0.5259, 0.5260, 0.5273, 0.5290, 0.5309, 0.5333,
    0.5367, 0.5438, 0.5566, 0.5738)
# ^ p50 of 149820 boundary profiles from CASIA-B TRAIN ids 001..074 (180 angular bins,
#   mean radius normalised to 1). Regenerate: evaluation/privacy/silhouette/derive_ksame_template.py
MASK_SIMPLIFY_EPS = 0.01          # approxPolyDP epsilon as a fraction of each contour's perimeter.

# ---------------- Differential privacy on the 12 face scalars (WHOLE-CLIP) ----------------
# Applied once in mirage_tier1.apply_dp() over the full clip (smooth -> quantize -> Laplace),
# NOT per frame. See apply_dp() for the mechanism + budgeting.
DP_ON = True                      # default ON. Set False only for debugging the raw scalars.
# UNIT OF PRIVACY (user decision 2026-07-24). Two policies, one flag:
#   False (SHIPPED, "option a") - unit = one FRAME. Delivered eps = 3.0 PER FRAME (verified, §I.1).
#                                 An attacker re-identifying a person from their whole T_p-frame
#                                 expression trajectory is NOT covered; that is stated as a known
#                                 limitation, not claimed away.
#   True  ("option b")         - unit = one PERSON (group privacy over that person's T_p frames):
#                                 Δ_c scales by T_p, so the Laplace noise is ~T_p× larger (T_p~109
#                                 here). Delivers person-level eps ≈ DP_EPSILON_TOTAL, but visibly
#                                 noisier face scalars → coarser character face animation.
# The two are meant to be A/B-compared VISUALLY (generate a character each, judge face quality)
# before the final paper wording is chosen - that comparison needs a cloud render, so it is pending.
# Default stays False so nothing changes until that call is made.
DP_PERSON_LEVEL = False
DP_EPSILON_TOTAL = 3.0            # PLACEHOLDER whole-CLIP epsilon budget, split evenly across the 12
                                  # channels => eps_c = 3.0/12 = 0.25 each; b_c = SENSITIVITY[c]/eps_c.
                                  # Smaller = more privacy / more noise. 3.0 is provisional - calibrate.
DP_WINDOW = _frames(DP_WINDOW_S)                      # 7 at EMIT_FPS=15
                                  # temporal window-mean length in FRAMES (
                                  # was 5 @ 10 fps - bumped to keep the same ~0.5 s window after the
                                  # 30->15 fps decimation. Revisit if EMIT_FPS changes.)
DP_QUANTIZE_BITS = 5              # quantize each channel to 2**5 = 32 levels over its observed range.
# Per-channel L1 sensitivity Δ_c. ORDER matches face_scalars():
#   0 mouth_open 1 mouth_width 2 smile 3 eyeR_open 4 eyeL_open 5 browR 6 browL
#   7 yaw 8 pitch 9 roll(rad) 10 gazeX 11 gazeY
#
# ✅ CORRECTED 2026-07-23 (tier1_lab/reports/DP_CALIBRATION_2026-07-23.json, ledger §I.1).
# Δ_c is a PROPERTY OF THE MECHANISM, not a tuning knob - it is the largest L1 change one unit of
# privacy can cause in the released vector, and apply_dp's derivation gives exactly:
#     one frame enters `DP_WINDOW` window-means at weight 1/w each  =>  ΣL1 = the clip range,
# and the mechanism clips to the PUBLIC CH_BOUNDS, so  Δ_c = hi_c - lo_c.
#
# The previous values were the p1..p99 SPAN of observed data (2026-07-19). Observed spread is not
# sensitivity: it is 1.20x-10.75x SMALLER than the range the mechanism actually admits (worst:
# gazeY 0.093 vs 1.0), so the Laplace scale was that many times too small and the delivered
# epsilon was 9.24, not the configured 3.0. Understating Δ to get less noise does not buy privacy,
# it just makes the guarantee false. If the added noise costs too much animation quality, raise
# DP_EPSILON_TOTAL - that is an honest, declarable policy choice; shrinking Δ is not.
# Δ_c = WINDOW_AMPLIFICATION * (hi_c - lo_c). The range alone is NOT enough: a centred moving
# average gives frames near the clip EDGE more than unit influence, because they sit in the short
# windows where each sample carries a larger weight (1/4, 1/5, 1/6 rather than 1/w). Measured
# 2026-07-23 with an adversarial probe: Δ = range was violated at 1.129x. The factor below is the
# exact geometry of _window_mean - max over i of sum_{t in window(i)} 1/count(t) - so it stays
# correct if DP_WINDOW changes, instead of being a magic number:
#   w=3 -> 1.1667   w=5 -> 1.1833   w=7 -> 1.1881   w=9 -> 1.1901   w=11 -> 1.1911
# (It converges to ~1.19; the edge effect does not vanish with a longer window.)
def _window_amplification(w, T=4096):
    pad, T = int(w) // 2, max(1, int(T))
    if pad == 0:
        return 1.0
    cnt = [min(t + pad, T - 1) - max(t - pad, 0) + 1 for t in range(T)]
    return max(sum(1.0 / cnt[t] for t in range(max(0, i - pad), min(T, i + pad + 1)))
               for i in range(T))


# ⚠️ A_w is NOT MONOTONE IN T - it PEAKS AT T == w, then decays to the long-clip value:
#   T =  5      6      7      8      9     10   ...  4096
#     1.100  1.233  1.376  1.269  1.212  1.188  ...  1.188      (w = 7)
# Evaluating it at one long length therefore UNDER-bounds every short slot. Measured 2026-07-23:
# a 7-frame person-slot (0.47 s at EMIT_FPS=15 - a bystander crossing frame) delivered
# eps = 3.421 against a configured 3.00. apply_dp now derives the factor from the slot it is
# actually releasing; this module constant is the SUPREMUM over all T, so anything reading
# SENSITIVITY directly is conservative rather than subtly wrong.
WINDOW_AMPLIFICATION = max(_window_amplification(DP_WINDOW, t)
                           for t in range(1, 3 * int(DP_WINDOW) + 2))   # 1.3762 at DP_WINDOW=7
# Quantization runs BEFORE the Laplace draw, so it is pre-processing and its sensitivity counts:
# rounding to the public grid can add one step per changed coordinate, and one frame moves at
# most `w` coordinates. Caught at T=2/T=4, where A_w is exactly 1.0 yet the measured L1 was
# 1.0323x range - the excess was precisely one step.
QUANTIZE_SENSITIVITY_EXTRA = float(DP_WINDOW) / float(max(2, 2 ** int(DP_QUANTIZE_BITS)) - 1)
_CH_RANGE = [1.0, 1.5, 1.0, 0.5, 0.5, 0.6, 0.6, 3.0, 3.0, 2.0, 1.0, 1.0]   # hi - lo per channel
SENSITIVITY = [round((WINDOW_AMPLIFICATION + QUANTIZE_SENSITIVITY_EXTRA) * r, 6)
               for r in _CH_RANGE]
# Kept for provenance / A-B against the earlier artifacts (NOT a valid sensitivity):
SENSITIVITY_LEGACY_P1P99 = [0.316, 0.897, 0.170, 0.167, 0.161, 0.500,
                            0.444, 1.557, 0.965, 0.939, 0.240, 0.093]
# ⚠️ STILL OPEN - the UNIT OF PRIVACY. The above makes the PER-FRAME claim true (eps_total = 3.0
# as configured). It does NOT give person-level DP: an attacker re-identifies someone from their
# whole expression trajectory, and one person contributes T_p frames (measured 84..199 locally),
# so person-level epsilon is ~T_p x larger (~1000 at these settings). Closing that needs eps_c/T_p
# or a per-person cap on contributed frames - ~T_p x more noise, a design decision with a real
# utility cost, deliberately left to the user. Until it is closed, describe this as
# "smoothed + quantized + noised", NOT as "eps-differentially private".

# FIXED PUBLIC per-channel bounds for DP quantization (adversary D2 - must NOT be data-dependent).
# A-priori bounds for eye-distance-normalised scalars (O(1); roll in radians). Widen if a real clip
# clips a channel. ORDER matches face_scalars(): mouth_open, mouth_w, smile, eyeR, eyeL, browR, browL,
# yaw, pitch, roll, gazeX, gazeY.
CH_BOUNDS = [(0.0, 1.0), (0.0, 1.5), (-0.5, 0.5), (0.0, 0.5), (0.0, 0.5), (0.0, 0.6), (0.0, 0.6),
             (-1.5, 1.5), (-1.5, 1.5), (-1.0, 1.0), (-0.5, 0.5), (-0.5, 0.5)]
# OPTIONAL non-uniform per-channel epsilon (adversary A3): spend LESS eps (=> MORE noise) on the
# channels that carry the most re-identification signal and least animation value - yaw/pitch/roll
# (7,8,9) and gazeX/gazeY (10,11). None => even split of DP_EPSILON_TOTAL. Calibrate with real footage
# (dp_calibration.md) before shipping; these are provisional weights that still sum to DP_EPSILON_TOTAL.
# ENABLED with a principled split (sums to DP_EPSILON_TOTAL=3.0): the 7 EXPRESSION channels (mouth,
# smile, eyes, brows - the animation signal, low re-id value) each get eps≈0.316; the 5 POSE/GAZE
# channels get LESS eps (=> MORE noise) because head-pose + gaze carry the most re-identification signal,
# with gaze the tightest: yaw/pitch 0.20, roll 0.15, gazeX/gazeY 0.12. Provisional - re-tune with the
# on-device + per-person calibration (D3) alongside SENSITIVITY.
DP_EPS_PER_CH = [0.316, 0.316, 0.316, 0.316, 0.316, 0.316, 0.316, 0.20, 0.20, 0.15, 0.12, 0.12]

ORT_PROVIDERS = ["CPUExecutionProvider"]   # Pi5 has no NPU. Use CPU. (int8 models strongly advised)
