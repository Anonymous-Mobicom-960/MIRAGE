#!/usr/bin/env python3
"""
build_cloud_bundle.py -- assemble the PER-SLOT cloud bundle for a MIRAGE Tier-1 two-person run.
====================================================================================================

Adapted from `100826_runs/_build_two_person.py` (itself adapted from
`_e2e/run3_20260807/build_bundles3.py`). The difference is only the INPUT: that script reads a
MIRAGE edge run's `tier1_out/{pose.json, face_scalars.json, mask.mp4}` directly, while this one
reads the MIRAGE host's `.npy` export and converts it first with `npy_to_mirage_emit.py`.
Everything downstream of that conversion -- the plate, the per-slot split, the sticks, the facemesh
-- is the same logic, and the split rule below is carried over VERBATIM together with the reason it
exists.

  🔴 THE PER-SLOT MASK IS THE PIECE THAT BROKE N-CHARACTER COMPOSITING (2026-08-08, ledger B.57e).
  The V9 graph has TWO mask loaders, and when both pointed at the UNION mask,
  `WanVideoAnimateEmbeds.mask` told the sampler to generate inside BOTH boxes. It obliged -- every
  render came back with two characters in it -- and the matte, which is derived by DETECTING a
  person in the generated video, then had two candidates and picked the wrong one on both arms,
  0/140 frames each.

So each slot gets its own mask, sticks and facemesh, and the two cloud renders are each bound to
ONE slot.

  🔴 PER-PIXEL NEAREST-POSE ASSIGNMENT, not connected components (2026-08-13).
  The component rule assigns each whole blob to one slot, which is correct ONLY while the two
  people are separated. When they touch, the mask has ONE component, both slots select it, and the
  two emitted masks come out BYTE-IDENTICAL -- measured on `two_person_b`: 14/50 frames, 898 076 px
  of overlap. That is precisely the B.57e defect. Assigning every mask pixel to the slot whose OWN
  keypoints are nearest degrades gracefully: while the people are apart it reproduces the component
  split exactly, and when they touch it cuts the merged blob along the perpendicular bisector
  between the two bodies. It is also free of the left/right assumption that B.57e-2 showed is what
  fails here -- the rule reads each slot's own pose, never screen position.

OUTPUTS (the contract, matched against the live reference bundle
`runpod-slim/ComfyUI/input/100826_twoperson/` read off the network volume):

    masked_video_00002.mp4     PLATE: the emitted mask filled flat grey over the PHONE'S LIGHTMAP.
                               The real scene never leaves the device -- this is what makes the
                               upload privacy-safe.
    mask_00002.mp4             the emitted UNION mask, passed through byte-for-byte
    light_map.mp4              the phone's lightmap (reaches the sampler as bg_images)
    mask_p1/p2_00002.mp4       the per-slot masks (see the split rule above)
    pose_sticks_p1/p2/both_00002.mp4   anonymised skeleton on BLACK
    facemesh_p1/p2_00002.mp4   the shared synthetic canonical head on BLACK
    reference_p1/p2_640.png    NOT PRODUCIBLE FROM THIS CLIP -- see --refs and the note below
    MANIFEST.json              every stream, plus THIS run's real config

REFERENCE IMAGES. Inspected on the volume, `reference_p{1,2}_640.png` are OWNER-SUPPLIED SYNTHETIC
CHARACTER SHEETS (AI-generated stock people: a male portrait sheet and a female full-body shot).
They are the REPLACEMENT identity WanAnimate paints into the hole. They are NOT crops of anybody in
the footage and they cannot be derived from a clip -- deriving one from the clip would carry the
real subject's appearance to the cloud, which is exactly what Tier-1 exists to prevent. So this
script never generates them. `--refs <dir>` copies a pair the owner already has, and records in
MANIFEST.json that they were CARRIED OVER and from where; without `--refs` the bundle is written
without them and MANIFEST.json says so.

    python build_cloud_bundle.py --tier1 <out_final> --tier2 <tier2_out> --out <to_cloud> \
                                 [--refs <dir with reference_p1_640.png/reference_p2_640.png>]
"""
import argparse
import io
import json
import os
import shutil
import sys

import cv2
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.dirname(os.path.dirname(HERE))
sys.path.insert(0, HERE)   # tier1_viz.py + npy_to_mirage_emit.py ship beside this script
import tier1_viz as VIZ                      # noqa: E402
from npy_to_mirage_emit import convert        # noqa: E402

FILL = 128            # the flat grey the plate paints inside the mask (the repaintable hole)
MASK_CROP_MARGIN_FRAC = 0.12   # of the emitted skeleton height; see the crop block below
# ---------------------------------------------------------------- MISSING FEET (2026-08-16)
# THE DEFECT, MEASURED on both 7 s walking clips: the generated character's alpha stops SHORT of
# the bottom of the repaint hole -- median 18 px (clip 1) / 17 px (clip 2), worst 66 px, on
# 32/69 and 26/69 frames. The composite then shows inpainted ground where the feet belong.
# It is the GENERATOR under-filling, not a mask error: the hole already reaches the subject's
# real foot line (hole bottom median 1217 vs character bottom median 1206).
#
# THE FIX BEING TESTED: extend the per-slot hole BELOW the foot line so the sampler has room to
# place feet instead of ending exactly where it must stop. Expressed as a fraction of the
# emitted skeleton's own height, so it scales with framing and subject distance -- never a pixel
# constant. 0.04 of a ~560 px figure is ~22 px, just past the measured 18 px median shortfall.
# 🔴 UNVERIFIED: no render has used this yet. It costs mask area (the cloud's inpaint bill) and
# whether the generator actually fills the extra room is exactly what the next render settles.
# Set 0.0 to reproduce every bundle built before today.
# 🔴 REFUTED AND DISABLED 2026-08-16. Measured on BOTH clips: the hole bottom moved
# 1217->1245 (clip 1) and 1099->1122 (clip 2) while the CHARACTER bottom moved 0 px on each
# (1206->1206, 1097->1097), so the shortfall got WORSE (18->46 px, 17->38 px) and the frames
# >20 px short went 32/69->50/69 and 26/69->64/69. The character extent is set by the
# CONDITIONING -- the sticks end at the ankle and the generator draws to them -- not by the
# hole, so extra room is never filled and only costs ~3 %% mask area (the cloud inpaint bill).
# The fix that DOES work is a composite-time vertical stretch of the character to the ground
# line (scratchpad/stretch_char.py): 2.5 %% mean / 8.4 %% max on clip 1, 3.4 %% on clip 2.
MASK_FOOT_MARGIN_FRAC = 0.0
# ---------------------------------------------------------------- VERTICAL ENLARGEMENT (owner, 2026-08-16)
# OWNER INSTRUCTION: "increase the mask we feed to cloud for a given character by 20-25 %
# vertically" -- the PER-CHARACTER cloud masks only, explicitly NOT the phone-side mask.
# So this touches `mask_p{N}_00002.mp4` and nothing else: the Tier-1 union mask
# (`mask_00002.mp4`) is still passed through byte-identical, and the plate is still greyed from
# that union mask, so neither the anonymisation nor the phone pipeline changes by one pixel.
#
# It is applied AFTER the skeleton-extent crop, so it enlarges the hole the crop just settled
# rather than fighting it, and it scales the mask's OWN measured height -- never a pixel constant.
# Growth direction is set by MASK_VSCALE_DOWN_FRAC: 1.0 = all of it downward (the feet end, which
# is where the defect is), 0.5 = symmetric. Downward by default because the crown of the head is
# already covered and pushing the hole UP only buys empty sky, whereas §G.9-3 measured that
# disturbing the TOP edge is what smears faces.
#
# 🔴 UNVERIFIED AND, ON THE PRIOR EVIDENCE, UNLIKELY TO MOVE THE FEET. §G.10-1 rendered a +29 px
# (clip 1) / +23 px (clip 2) downward extension on both clips and the character bottom moved
# EXACTLY 0 px on each -- the extent is set by the CONDITIONING (the sticks stop at the ankle),
# not by the hole. This knob is ~4-5x bigger, so it is a fair test of whether the effect is
# merely threshold-like, but nothing measured so far predicts that it is. It costs mask area,
# i.e. cloud inpaint bill and more of the scene replaced by generated content.
# Set 1.0 to reproduce every bundle built before 2026-08-16.
#
# 🔴 THE CANVAS CAN REFUSE IT, AND ON CLIP 1 IT DOES. Measured 2026-08-16: clip 1's hole already
# ends at y 1217 of 1264, so only 46 px of downward room exist and a requested +22 % delivers
# +5.8 %. Clip 2's hole ends at 1087 and takes the full +21.1 %. `MASK_VSCALE_SPILL_UP` decides
# what happens to the remainder: False (default) accepts the smaller increase, True puts the
# leftover above the head.
# DEFAULT IS FALSE ON EVIDENCE, not caution. Extending the hole where no conditioning describes a
# body is what §G.9 measured producing an invented crouch. BELOW the feet that mismatch is proven
# benign -- §G.10-1 grew the hole downward on both clips and the generator simply left inpainted
# ground (character bottom moved 0 px). ABOVE the head the same mismatch has NEVER been measured,
# and that is the region where §G.9-3 measured a disturbed top edge smearing the face. So the
# unmeasured direction stays off until a render prices it.
# 🔴 SET TO 1.0 BECAUSE THE FLOOR EXTENSION SUBSUMES IT -- proven, not assumed: building both
# clips at 1.22+floor and at 1.0+floor produced BYTE-IDENTICAL mask_p1 streams
# (509d7e8cb2cc5bba / 08bd948a4dccf1e4). The shipped silhouette mode is `bbox`, so the mask is a
# rectangle; stretching a rectangle down and then filling to the floor lands in the same place as
# filling to the floor directly. Leaving it at 1.22 would advertise a 22 % enlargement that
# changes nothing, which is worse than not having the knob. It is NOT redundant under a shaped
# mode (`hull`, `ellipse`), where the scale genuinely stretches the silhouette -- so the knob
# stays, at its no-op value, for whoever changes shape mode.
MASK_VSCALE = 1.0             # owner band was 1.20-1.25; see above for why it is 1.0
MASK_VSCALE_DOWN_FRAC = 1.0   # 1.0 = grow downward only; 0.5 = split evenly top/bottom
MASK_VSCALE_SPILL_UP = False  # put canvas-refused downward growth above the head instead
# FLOOR EXTENSION (owner, 2026-08-16): run the hole all the way to the BOTTOM OF THE FRAME.
# Without it the enlargement is uneven between clips purely because of where each subject stands:
# clip 1's hole reaches the canvas edge on 69/70 frames (which is why it could only grow 5.8 %),
# clip 2's on 16/70, mean bottom 1204 of 1263. With it, "the character has room down to the
# ground" is true on every frame of every clip instead of being an accident of framing.
# Applied ONLY under the columns the mask already occupies, so it is a strip beneath the subject,
# never a band across the whole frame -- the same rule the (refuted) foot-room margin used.
MASK_EXTEND_TO_FLOOR = True
MIN_KP_FOR_SPLIT = 4  # a slot needs >4 drawable joints before it may claim mask pixels
MIN_COMPONENT_PX = 3000


def slot_filter(frames, slot):
    if slot is None:
        return frames
    return [[p for p in fr if int(p.get("slot", 0)) == slot] for fr in frames]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--tier1", required=True, help="Tier-1 out dir (keypoints_p*.npy, mask.mp4)")
    ap.add_argument("--tier2", required=True, help="Tier-2 out dir (light_map.mp4)")
    ap.add_argument("--out", required=True, help="bundle dir to write")
    ap.add_argument("--refs", default="", help="dir holding reference_p{1,2}_640.png to CARRY OVER")
    ap.add_argument("--emit", default="", help="where to write the converted pose/face json "
                                               "(default <out>/../tier1_emit)")
    a = ap.parse_args()
    out = a.out
    os.makedirs(out, exist_ok=True)
    emit_dir = a.emit or os.path.join(os.path.dirname(os.path.abspath(out)), "tier1_emit")

    # ---- 0. ADAPT: .npy arrays -> the MIRAGE emit format this pipeline consumes ---------------
    adapter_report = convert(a.tier1, emit_dir)
    if not adapter_report["all_checks_pass"]:
        sys.exit("adapter validation FAILED -- refusing to build:\n%s"
                 % json.dumps(adapter_report["checks"], indent=1))
    pose = json.load(io.open(os.path.join(emit_dir, "pose.json"), encoding="utf-8"))
    frames = pose["frames"]
    scal = json.load(io.open(os.path.join(emit_dir, "face_scalars.json"), encoding="utf-8"))
    slots = sorted({int(p.get("slot", 0)) for fr in frames for p in fr})
    t1man = json.load(io.open(os.path.join(a.tier1, "manifest.json"), encoding="utf-8"))
    FPS = float(t1man.get("fps") or 10.0)
    print("slots present: %s | fps %.4g" % (slots, FPS))

    mask_p = os.path.join(a.tier1, "mask.mp4")
    lm_p = os.path.join(a.tier2, "light_map.mp4")
    for p in (mask_p, lm_p):
        if not os.path.exists(p):
            sys.exit("MISSING %s -- refusing to build a partial bundle" % p)

    # ---- 1. PLATE: emitted mask over the phone's LIGHTMAP (the real scene never leaves device)
    cm, cl = cv2.VideoCapture(mask_p), cv2.VideoCapture(lm_p)
    n = min(int(cm.get(cv2.CAP_PROP_FRAME_COUNT)), int(cl.get(cv2.CAP_PROP_FRAME_COUNT)))
    w, h = int(cm.get(cv2.CAP_PROP_FRAME_WIDTH)), int(cm.get(cv2.CAP_PROP_FRAME_HEIGHT))
    vw = cv2.VideoWriter(os.path.join(out, "masked_video_00002.mp4"),
                         cv2.VideoWriter_fourcc(*"mp4v"), FPS, (w, h))
    mask_frames, k = [], 0
    while k < n:
        okm, fm = cm.read()
        okl, fl = cl.read()
        if not (okm and okl):
            break
        if fl.shape[:2] != (h, w):
            fl = cv2.resize(fl, (w, h), interpolation=cv2.INTER_LINEAR)
        g = cv2.cvtColor(fm, cv2.COLOR_BGR2GRAY)
        mask_frames.append(g)
        fr = fl.copy()
        fr[g > 127] = FILL
        vw.write(fr)
        k += 1
    cm.release(); cl.release(); vw.release()

    # ---- 2. union mask + lightmap pass through, byte for byte
    for src, dst in ((mask_p, "mask_00002.mp4"), (lm_p, "light_map.mp4")):
        with open(src, "rb") as r, open(os.path.join(out, dst), "wb") as wf:
            wf.write(r.read())

    # ---- 3. PER-SLOT masks, per-pixel nearest-pose assignment (see the header) ----------------
    per_slot = {s: [] for s in slots}
    crop_stats = {s: {"frames_cropped": 0, "px_removed": 0} for s in slots}
    vscale_stats = {s: {"frames": 0, "px_added": 0, "h_before": 0, "h_after": 0, "clipped": 0}
                    for s in slots}
    floor_stats = {s: {"frames": 0, "px_added": 0} for s in slots}
    split_ok = 0
    for t, g in enumerate(mask_frames):
        b = g > 127
        pts = {}
        for p in (frames[t] if t < len(frames) else []):
            kp = np.array(p["kp"], float)
            sc = np.array(p["score"], float)
            v = kp[(sc > 0) & (np.abs(kp).sum(1) > 0)]
            if len(v) > MIN_KP_FOR_SPLIT:
                pts[int(p.get("slot", 0))] = v
        dists = {}
        for s in slots:
            seed = np.ones(b.shape, np.uint8) * 255
            if s in pts:
                for x, y in pts[s]:
                    xi, yi = int(round(x)), int(round(y))
                    if 0 <= xi < b.shape[1] and 0 <= yi < b.shape[0]:
                        seed[yi, xi] = 0
            dists[s] = cv2.distanceTransform(seed, cv2.DIST_L2, 3)
        ncc, _lab, st, _ce = cv2.connectedComponentsWithStats(b.astype(np.uint8), 8)
        split_ok += (len([i for i in range(1, ncc) if st[i, 4] >= MIN_COMPONENT_PX]) == len(slots))
        stack = np.stack([dists[s] for s in slots])          # (S,H,W)
        winner = np.argmin(stack, axis=0)
        for si, s in enumerate(slots):
            m = np.zeros(b.shape, np.uint8)
            m[b & (winner == si)] = 255
            # 🔴 CROP THE REPAINT HOLE TO WHAT THE CONDITIONING ACTUALLY DESCRIBES.
            # THE DEFECT (measured on p20-p21_c01_2face, 2026-08-15): the hole is a bbox of
            # the REAL silhouette and runs to the frame bottom, but the framing cuts the legs
            # off -- ankles confident in 0 % of frames for BOTH subjects, knees 0 % for slot 1.
            # So the sticks describe a torso while the hole demands a whole body, and the
            # generator fills the difference with an invented crouch (that is the leaning
            # half-body figure in the first render).
            #
            # The hole is therefore limited to the emitted skeleton's own extent plus a margin
            # DERIVED FROM THAT SKELETON'S HEIGHT -- never a pixel constant, so it scales with
            # framing and subject distance. `MASK_CROP_MARGIN_FRAC` sits inside the 0.10-0.13
            # bracket §B measured as rendering cleanly (65-83 px on a ~629 px figure); smaller
            # blurred the body, larger let the generator invent to fill empty space. It is a
            # population value from that bracket, NOT fitted to this clip.
            #
            # WHY THIS DOES NOT WEAKEN §2: the ANONYMISATION mask (mask_00002.mp4, the union
            # written above) is untouched and still covers the whole person. Only the per-slot
            # REPAINT region shrinks. The plate is the lightmap with the hole greyed, so the
            # area no longer repainted shows the INPAINTED BACKGROUND, never real pixels.
            # 🔴 BOTTOM ONLY, AND NEVER THE TOP. A symmetric crop around the skeleton was tried
            # first and MEASURED HARMFUL (2026-08-15): it pulled the hole's top DOWN 39-61 px on
            # p20-p21 because the skeleton's topmost point is the NOSE (y 632-666) while the real
            # head reaches y 517-541. The crown of the head fell outside the repaint region and the
            # render came back with a smeared, truncated face. A skeleton cannot say where the crown
            # is -- only the mask knows that -- so the top, left and right edges are left alone.
            #
            # The bottom is also measured against the lowest GENUINELY OBSERVED joint. A joint
            # sitting on the canvas edge is not an observation: `anonymize_pose_log` ends with
            # np.clip(kp, 0, frame_wh), so a joint the transform pushed out of shot is parked
            # exactly on the border (kneeR reads y = 1264.0 here). Treating that as "the body
            # reaches the frame bottom" is what let the hole keep demanding a full figure while the
            # conditioning described a torso. Those joints are excluded from the extent.
            if s in pts and len(pts[s]):
                ys = pts[s][:, 1]
                on_edge = (pts[s][:, 1] >= b.shape[0] - 1.5) | (pts[s][:, 1] <= 0.5) | \
                          (pts[s][:, 0] >= b.shape[1] - 1.5) | (pts[s][:, 0] <= 0.5)
                real = ys[~on_edge]
                if len(real) >= MIN_KP_FOR_SPLIT:
                    sk_h = float(real.max() - real.min())
                    if sk_h > 1:
                        y1 = min(b.shape[0] - 1,
                                 int(round(real.max() + MASK_CROP_MARGIN_FRAC * sk_h)))
                        before = int((m > 0).sum())
                        m[y1 + 1:, :] = 0
                        # FOOT ROOM: grow the hole downward past the real foot line, only
                        # under the columns the person already occupies, so the extra area is
                        # a strip beneath the subject and not a rectangle across the frame.
                        if MASK_FOOT_MARGIN_FRAC > 0:
                            pad = int(round(MASK_FOOT_MARGIN_FRAC * sk_h))
                            if pad > 0:
                                ys_m, xs_m = np.nonzero(m > 0)
                                if len(xs_m):
                                    lo, hi = int(xs_m.min()), int(xs_m.max())
                                    foot = int(ys_m.max())
                                    y2f = min(b.shape[0] - 1, foot + pad)
                                    m[foot + 1:y2f + 1, lo:hi + 1] = 255
                        removed = before - int((m > 0).sum())
                        if removed:
                            crop_stats[s]["frames_cropped"] += 1
                            crop_stats[s]["px_removed"] += removed
            # VERTICAL ENLARGEMENT of THIS slot's cloud mask (see MASK_VSCALE at the top).
            # Resize the mask's occupied band, so a bbox hole simply gets taller and a shaped
            # hole keeps its silhouette; then paste it back with the growth placed per
            # MASK_VSCALE_DOWN_FRAC and clipped to the canvas.
            if MASK_VSCALE != 1.0 and (m > 0).any():
                ys_v = np.nonzero(m > 0)[0]
                top0, bot0 = int(ys_v.min()), int(ys_v.max())
                h0 = bot0 - top0 + 1
                h1 = int(round(h0 * MASK_VSCALE))
                if h1 > h0:
                    band = cv2.resize(m[top0:bot0 + 1], (m.shape[1], h1),
                                      interpolation=cv2.INTER_NEAREST)
                    grow = h1 - h0
                    top1 = top0 - int(round(grow * (1.0 - MASK_VSCALE_DOWN_FRAC)))
                    if MASK_VSCALE_SPILL_UP:
                        # the canvas bottom refuses part of the downward growth -> lift the band
                        # so the FULL requested height still lands, as far as the top allows
                        over = (top1 + h1) - m.shape[0]
                        if over > 0:
                            top1 = max(0, top1 - over)
                    src_lo = max(0, -top1)                      # clip whatever leaves the canvas
                    dst_lo = max(0, top1)
                    n_rows = min(h1 - src_lo, m.shape[0] - dst_lo)
                    if n_rows > 0:
                        m2 = np.zeros_like(m)
                        m2[dst_lo:dst_lo + n_rows] = band[src_lo:src_lo + n_rows]
                        vs = vscale_stats[s]
                        vs["frames"] += 1
                        vs["px_added"] += int((m2 > 0).sum()) - int((m > 0).sum())
                        vs["h_before"] += h0
                        if (m2 > 0).any():
                            yv2 = np.nonzero(m2 > 0)[0]
                            vs["h_after"] += int(yv2.max() - yv2.min() + 1)
                        vs["clipped"] += 1 if (src_lo or n_rows < h1) else 0
                        m = m2
            if MASK_EXTEND_TO_FLOOR and (m > 0).any():
                ys_f, xs_f = np.nonzero(m > 0)
                lo, hi, foot = int(xs_f.min()), int(xs_f.max()), int(ys_f.max())
                if foot < m.shape[0] - 1:
                    m[foot + 1:, lo:hi + 1] = 255
                    floor_stats[s]["frames"] += 1
                    floor_stats[s]["px_added"] += (m.shape[0] - 1 - foot) * (hi - lo + 1)
            per_slot[s].append(m)
    for s in slots:
        vws = cv2.VideoWriter(os.path.join(out, "mask_p%d_00002.mp4" % (s + 1)),
                              cv2.VideoWriter_fourcc(*"mp4v"), FPS, (w, h))
        for m in per_slot[s]:
            vws.write(cv2.cvtColor(m, cv2.COLOR_GRAY2BGR))
        vws.release()

    # ---- 4. PER-SLOT sticks + facemesh on black, plus a combined sticks stream ----------------
    blank = [np.zeros((h, w, 3), np.uint8)] * max(len(frames), len(mask_frames))
    made = []
    for slot, name in [(s, "p%d" % (s + 1)) for s in slots] + [(None, "both")]:
        ff = slot_filter(frames, slot)
        sp = os.path.join(out, "pose_sticks_%s_00002.mp4" % name)
        VIZ.make_pose_sticks(blank, ff, sp, FPS, on_black=True)
        made.append(os.path.basename(sp))
        if slot is not None:
            fp = os.path.join(out, "facemesh_%s_00002.mp4" % name)
            VIZ.make_synthetic_face(blank, mask_frames, fp, FPS, on_black=True,
                                    pose_frames=ff, face_scalars=scal)
            made.append(os.path.basename(fp))

    # ---- 5. REFERENCE IMAGES -- carried over only, never generated (see the header) -----------
    refs = {"producible_from_this_clip": False,
            "what_they_are": ("owner-supplied SYNTHETIC character sheets -- the REPLACEMENT "
                              "identity WanAnimate paints into the hole. Not a crop of anybody in "
                              "the footage; deriving one from the clip would put the real "
                              "subject's appearance in the cloud upload."),
            "present": [], "provenance": None}
    if a.refs:
        for s in slots:
            nm = "reference_p%d_640.png" % (s + 1)
            src = os.path.join(a.refs, nm)
            if os.path.exists(src):
                shutil.copyfile(src, os.path.join(out, nm))
                refs["present"].append(nm)
        refs["provenance"] = {
            "status": "CARRIED OVER -- NOT derived from this clip",
            "from": os.path.abspath(a.refs),
            "owner_confirmation_required": True,
            # Over the slots THIS run actually has. Hardcoding two indexed past the end on a
            # single-person clip (IndexError, 2026-08-15) -- the builder was written for the
            # two-person case and silently assumed it everywhere.
            "slot_binding_evidence": {
                "slot%d->p%d" % (s, s + 1): (t1man["slots"][s].get("gender"),
                                             t1man["slots"][s].get("gender_confidence"))
                for s in slots if s < len(t1man.get("slots", []))
            },
        }
        with io.open(os.path.join(out, "REFERENCE_IMAGES_README.txt"), "w", encoding="utf-8") as f:
            f.write("reference_p1_640.png / reference_p2_640.png in this bundle were COPIED from\n"
                    "%s\nThey are owner-supplied synthetic character sheets belonging to a "
                    "DIFFERENT clip with DIFFERENT subjects.\nNothing in this bundle derived them "
                    "and nothing verifies they are the characters you want here.\nConfirm or "
                    "replace them before queueing a render.\n" % os.path.abspath(a.refs))

    # ---- 6. MANIFEST: THIS run's real config, read from THIS run's artifacts ------------------
    anon = pose["anon"]
    man = dict(
        clip=t1man.get("clip_id"),
        source_clip=os.path.abspath(os.path.join(a.tier1, "..", "clip_5s_10fps_1264.mp4")),
        fps=FPS, frames=k, size=[w, h], slots=slots,
        person_count=pose["person_count"], emitted_slots=pose["emitted_slots"],
        config=anon,
        tier1_host=anon.get("host"),
        tier1_manifest_anon=t1man.get("anon"),
        tier1_manifest_mask=t1man.get("mask"),
        adapter=dict(script="tier2_cloud/scripts/npy_to_mirage_emit.py",
                     role="format ADAPTER: MIRAGE .npy -> MIRAGE pose.json/face_scalars.json",
                     checks=adapter_report["checks"], per_slot=adapter_report["per_slot"]),
        mask_split=dict(frames=len(mask_frames), clean_split_frames=split_ok,
                        rule="per-pixel nearest slot pose keypoint, per frame"),
        mask_vscale=dict(
            factor=MASK_VSCALE, down_frac=MASK_VSCALE_DOWN_FRAC,
            applies_to="mask_p{N}_00002.mp4 ONLY -- NOT mask_00002.mp4, NOT the plate, "
                       "NOT the phone-side mask",
            owner_instruction="increase the per-character cloud mask 20-25 % vertically "
                              "(2026-08-16)",
            per_slot={("p%d" % (s + 1)): dict(
                frames=vscale_stats[s]["frames"],
                mean_height_before=round(vscale_stats[s]["h_before"] /
                                         max(1, vscale_stats[s]["frames"]), 1),
                mean_height_after=round(vscale_stats[s]["h_after"] /
                                        max(1, vscale_stats[s]["frames"]), 1),
                px_added=vscale_stats[s]["px_added"],
                frames_clipped_by_canvas=vscale_stats[s]["clipped"]) for s in slots}),
        mask_extend_to_floor=dict(
            enabled=MASK_EXTEND_TO_FLOOR,
            rule="fill from the mask bottom to the canvas bottom, only under the columns the "
                 "mask already occupies",
            per_slot={("p%d" % (s + 1)): floor_stats[s] for s in slots}),
        reference_images=refs,
        streams={
            "masked_video_00002.mp4": "PLATE -- emitted mask filled %d over the phone lightmap" % FILL,
            "mask_00002.mp4": "emitted UNION mask (Tier-1 mask.mp4, byte-identical)",
            "light_map.mp4": "phone lightmap (bg_images)",
            "mask_p{1,2}_00002.mp4": "per-slot masks, per-pixel nearest-pose split",
            "pose_sticks_p{1,2}_00002.mp4": "anonymised skeleton on black (pose_images)",
            "pose_sticks_both_00002.mp4": "both slots, reference only -- do not feed one render",
            "facemesh_p{1,2}_00002.mp4": "shared synthetic canonical head on black (face_images)",
        },
        note=("TWO-PERSON run through the MIRAGE Tier-1 host with BOTH MIRAGE defences on: "
              "gait preset '%s' (level %s, the SHIPPED default, owner decision 2026-08-14) and "
              "silhouette shape_mode '%s' at temporal_win %s. Per-slot mask/sticks/facemesh are "
              "MANDATORY: with both graph mask loaders on the UNION mask the sampler generates a "
              "character in EVERY box and the cloud matte binds to the wrong one (0/140 frames, "
              "run3 B.57e). Each cloud render must be bound to ONE slot. "
              "PRIVACY: 🔴 NO privacy number measured on the MIRAGE edge host describes THIS host. "
              "The e2 and bbox figures in the ledger (A.2x, A.6o) were measured on the MIRAGE edge "
              "pipeline; this run is the same CODE on a different host and carries no measured "
              "privacy figure of its own. See MASTER_EVAL_LEDGER.md G.1-G.7 and the P0 row in "
              "section P."
              % (anon.get("gait_preset"), anon.get("level"),
                 (anon.get("mask") or {}).get("shape_mode"),
                 (anon.get("mask") or {}).get("temporal_win"))),
    )
    json.dump(man, io.open(os.path.join(out, "MANIFEST.json"), "w", encoding="utf-8"), indent=1)
    print("%d f %dx%d, slots %s, mask split cleanly in %d/%d frames"
          % (k, w, h, slots, split_ok, len(mask_frames)))
    print("  wrote %d files to %s" % (len(os.listdir(out)), out))
    return 0


if __name__ == "__main__":
    sys.exit(main())
