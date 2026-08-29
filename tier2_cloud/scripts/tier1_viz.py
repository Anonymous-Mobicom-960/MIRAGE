#!/usr/bin/env python3
"""
tier1_viz.py -- render the Tier-1 (edge) stage as viewable sidecar videos for the phone app.
============================================================================================
The phone app only ever sees the FINISHED Tier-1 mask; the interesting *pipeline* signals
(anonymised pose, the identity-collapse, the face-normalisation) are invisible. This renders
three sidecars the app's "Tier 1 (edge)" card can display:

    tier1_pose_sticks.mp4    -- the EMITTED (anonymised) 17-body + hands skeleton drawn over the
                                masked frame. This is the exact pose.json that leaves the device.
    tier1_canonical.mp4      -- the SAME skeleton root-centred + drawn on a neutral panel: the
                                "canonicaliser" view. Every identity's limb proportions have been
                                collapsed to ONE shared population template (pose_anon_edge), so
                                this is literally what the gait re-ID adversary sees -- no identity.
    tier1_synthetic_face.mp4 -- a SYNTHETIC canonical face mesh (a fixed generic template, NOT the
                                person's geometry) placed on each head using only the emitted,
                                identity-free head keypoints (nose/eyes/ears). PRIVACY-CLEAN: no real
                                face pixels and no real landmarks are used or drawn -- the shipped
                                pipeline zeroes the 68 face landmarks and emits only 12 expression
                                scalars, and this sidecar honours that (a placeholder face for the
                                normalisation stage, tracking head pose/scale). --no-synthetic-face omits it.

PRIVACY: nothing here ever touches raw real pixels -- only the anonymised masked frames, the
emitted mask, and the identity-free pose.json.

RENDER TARGET (user 2026-07-25): pose_sticks and synthetic_face now render on a BLACK canvas by
default, because they are not debug views -- they are the CONDITIONING SIGNALS fed to
WanVideoAnimateEmbeds.pose_images / .face_images. Drawn over the masked frame they were 0.653 %
and 0.684 % skeleton/face against ~98 % scene content plus a baked-in text banner. Pass
on_black=False to get the old human-readable overlay for the app's Tier-1 tiles.

    python tier1_viz.py --clip input/reid_dataset/d07_2person_1264.mp4 --out <stage_dir> \
                        [--work <existing_bridge_work_dir>] [--max-frames 100] [--no-synthetic-face]
"""
import argparse
import colorsys
import json
import math
import os
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.dirname(os.path.dirname(HERE))
EDGE = os.path.join(REPO, "tier1", "src", "edge_runner_pi5")

# COCO-17 skeleton (the body-17 block of the wholebody-133 kp). Kinematic pairs for drawing sticks.
SKELETON = [(0, 1), (0, 2), (1, 3), (2, 4),                 # head: nose-eyes-ears
            (5, 7), (7, 9), (6, 8), (8, 10),                # arms
            (5, 6), (5, 11), (6, 12), (11, 12),             # shoulders + torso
            (11, 13), (13, 15), (12, 14), (14, 16)]         # legs
# per-person colours (BGR) so multiple people are distinguishable
PCOLORS = [(0, 235, 0), (0, 200, 255), (255, 120, 0), (200, 0, 255), (0, 255, 255), (255, 255, 0)]
L_HAND = list(range(91, 112))
R_HAND = list(range(112, 133))
FACE_CLR = (120, 235, 120)   # synthetic-face wireframe colour (BGR)

# ---------------- OpenPose/DWPose-style rendering (2026-07-26) ----------------
# Pose-conditioned video models are trained on OpenPose-format skeleton renders (colored limbs
# on black, filled-ellipse bones, per-joint colored circles, hand bones as colored lines), not
# on a monochrome 3 px wireframe. The cloud-native V9 graph feeds WanVideoAnimateEmbeds from a
# `DrawViTPose` node (ComfyUI-WanAnimatePreprocess); its source is not in this mirror, so an
# exact style match is NOT verifiable locally - this implements the standard OpenPose BODY-18
# render (the format that family of preprocessors emits). Verify against one pod render.
#
# MEASURED motivation (v3 uploaded sticks, N=100 @1264²): the legacy render covers 1.101 % of
# pixels (>16) and a 3 px stroke is 0.375 latent px after the VAE's 8× downsample.
# COCO-17 -> OpenPose-18: neck (17) = mid-shoulder, then reorder.
_OP_FROM_COCO = [0, 17, 6, 8, 10, 5, 7, 9, 12, 14, 16, 11, 13, 15, 2, 1, 4, 3]
_OP_LIMBS = [(1, 2), (1, 5), (2, 3), (3, 4), (5, 6), (6, 7), (1, 8), (8, 9), (9, 10),
             (1, 11), (11, 12), (12, 13), (1, 0), (0, 14), (14, 16), (0, 15), (15, 17)]
_OP_COLORS = [(255, 0, 0), (255, 85, 0), (255, 170, 0), (255, 255, 0), (170, 255, 0),
              (85, 255, 0), (0, 255, 0), (0, 255, 85), (0, 255, 170), (0, 255, 255),
              (0, 170, 255), (0, 85, 255), (0, 0, 255), (85, 0, 255), (170, 0, 255),
              (255, 0, 255), (255, 0, 170), (255, 0, 85)]                     # RGB
_HAND_EDGES = [(0, 1), (1, 2), (2, 3), (3, 4), (0, 5), (5, 6), (6, 7), (7, 8),
               (0, 9), (9, 10), (10, 11), (11, 12), (0, 13), (13, 14), (14, 15), (15, 16),
               (0, 17), (17, 18), (18, 19), (19, 20)]


def _kp18(kp, sc):
    """COCO-17 (+scores) -> OpenPose-18 points + validity. Neck = mid-shoulder."""
    pts = np.zeros((18, 2)); ok = np.zeros(18, bool)
    def good(i):
        return sc[i] > 0 and (abs(kp[i][0]) > 1e-6 or abs(kp[i][1]) > 1e-6)
    ext = list(kp[:17]) + [0.5 * (np.asarray(kp[5], float) + np.asarray(kp[6], float))]
    extok = [good(i) for i in range(17)] + [good(5) and good(6)]
    for oi, ci in enumerate(_OP_FROM_COCO):
        pts[oi] = ext[ci]; ok[oi] = extok[ci]
    return pts, ok


def _draw_person_dwpose(img, kp, sc, base, bone_gate=None):
    """OpenPose-style render of one wholebody-133 person. `base` scales stroke widths to the
    canvas (1.0 at 512 px). `bone_gate(a17, b17) -> bool` optionally suppresses a COCO bone
    (the temporal debouncer); head/neck limbs are never gated."""
    import cv2
    kp = np.asarray(kp, float)
    if len(kp) < 17:
        return
    stick = max(3, int(round(4 * base)))
    joint_r = max(3, int(round(4 * base)))
    hand_t = max(2, int(round(2 * base)))
    pts, ok = _kp18(kp, sc)
    # ---- OPEN-CHAIN POLICY, decided BEFORE anything is drawn (2026-07-26) ----
    # Must run ahead of the limb loop: dropping a chain has to suppress the shoulder->elbow BONE as
    # well as the elbow circle, and the limb loop reads `ok` as it goes. Deciding this afterwards
    # would have left the bone drawn and removed only the dot.
    _mode = os.environ.get("MIRAGE_STICK_OPEN_CHAIN", "drop").lower()
    if _mode == "drop":
        for (_p, _m, _d) in ((2, 3, 4), (5, 6, 7), (8, 9, 10), (11, 12, 13)):
            if (ok[_p] and ok[_m]) and not ok[_d]:
                ok[_m] = False        # elbow/knee whose distal joint was never seen: omit it entirely
    # limbs as filled ellipses at 0.6 intensity (the OpenPose convention)
    _OP2COCO = {oi: ci for oi, ci in enumerate(_OP_FROM_COCO)}
    _bone_drawn = np.zeros(18, bool)      # which joints ended up with at least one DRAWN bone
    for li, (a, b) in enumerate(_OP_LIMBS):
        if not (ok[a] and ok[b]):
            continue
        ca, cb = _OP2COCO[a], _OP2COCO[b]
        if bone_gate is not None and ca < 17 and cb < 17 and not bone_gate(ca, cb):
            continue
        _bone_drawn[a] = _bone_drawn[b] = True
        ax, ay = pts[a]; bx, by = pts[b]
        L = math.hypot(bx - ax, by - ay)
        if L < 1:
            continue
        ang = math.degrees(math.atan2(by - ay, bx - ax))
        poly = cv2.ellipse2Poly((int((ax + bx) / 2), int((ay + by) / 2)),
                                (max(1, int(L / 2)), stick), int(ang), 0, 360, 1)
        r, g, bl = _OP_COLORS[li % 18]
        cv2.fillConvexPoly(img, poly, (int(bl * 0.6), int(g * 0.6), int(r * 0.6)), cv2.LINE_AA)
    # ---- OPEN-CHAIN EXTENSION (2026-07-26) -- the fix for hallucinated hands ----
    # When a limb's DISTAL joint is unobserved (hands below a bust framing -> wrist confidence under
    # the gate -> zeroed), the forearm bone above is simply skipped and the ELBOW is still drawn as a
    # joint. The conditioning image then shows an arm that stops dead in open space, and a
    # pose-conditioned generator does the reasonable thing: it completes the limb, inventing a hand.
    #
    # MEASURED (§B.16): hands appear only where one arm dangles like this in ~every frame --
    # a10 100 % dangling -> 100 % hands; ab_posefix 54 % -> 0 %; a15 9.3 % -> 0 %. And the A100
    # re-render still shows 98 % of frames with hands while the ground truth has NONE.
    #
    # FIX: continue the chain along its own last-known direction until it LEAVES the canvas. That is
    # not an invention -- the limb really does continue out of view, and an arm crossing the frame
    # edge is the cue for "occluded/out of shot" rather than "ends here". No joint circle is drawn at
    # the extrapolated end, so nothing reads as an anatomical terminus.
    #
    # GENERALISES: the direction comes from the limb's own two observed joints and the distance from
    # the canvas size. No fitted constant, no per-clip tuning, and it is a no-op whenever the distal
    # joint IS observed.
    #
    # 🔴 2026-07-26 - THIS EXTENSION DID NOT WORK AND IS NOW OFF BY DEFAULT.
    # `a10f` re-rendered a10 with it (same GPU, same seeds, only the sticks changed): the fix DID
    # apply (sticks differ in 100/100 frames, canvas-border ink +58 %) and the invented-hand rate went
    # **98 % -> 100 %**. The mechanism it was built on is withdrawn - see ledger §B.23-CORRECTION:
    # the elbow sits BELOW its shoulder in 100/100 frames, and the hands cluster at y=0.61, on the
    # SHOULDER (0.589, 0.587), ~450 px ABOVE the elbows. They are not completing a truncated arm.
    #
    # `STICK_OPEN_CHAIN` selects what to do when a limb's distal joint is unobserved:
    #   "drop"   (default) - omit the whole open chain AND its terminal joint circle. This is the
    #                        community convention for OpenPose-style conditioning: make the keypoint
    #                        invisible so its segments do not render, rather than inventing geometry
    #                        (sd-webui-openpose-editor; kijai added `pose_draw_threshold` to
    #                        ComfyUI-WanAnimatePreprocess for the same reason). Information-REMOVING,
    #                        so it cannot weaken §2 or the anti-reID posture.
    #   "extend"          - the 2026-07-26 behaviour kept for reproducing the a10f A/B.
    #   "keep"            - the pre-2026-07-26 behaviour: bone skipped, elbow circle still drawn.
    H_img, W_img = img.shape[:2]
    for (p_, m_, d_) in (() if _mode != "extend" else
                         ((2, 3, 4), (5, 6, 7), (8, 9, 10), (11, 12, 13))):
        if not (ok[p_] and ok[m_]) or ok[d_]:
            continue
        px, py = pts[p_]; mx, my = pts[m_]
        dx, dy = mx - px, my - py
        L = math.hypot(dx, dy)
        if L < 1:
            continue
        dx, dy = dx / L, dy / L
        # smallest t that puts (m + t*dir) outside the canvas, then a little past it
        ts = []
        if abs(dx) > 1e-6:
            ts += [((0 if dx < 0 else W_img - 1) - mx) / dx]
        if abs(dy) > 1e-6:
            ts += [((0 if dy < 0 else H_img - 1) - my) / dy]
        ts = [t for t in ts if t > 0]
        if not ts:
            continue
        t = min(ts) + max(H_img, W_img) * 0.02          # overshoot so it clearly crosses the edge
        ex_, ey_ = mx + dx * t, my + dy * t
        li = _OP_LIMBS.index((m_, d_)) if (m_, d_) in _OP_LIMBS else 0
        r, g, bl = _OP_COLORS[li % 18]
        cv2.line(img, (int(mx), int(my)), (int(ex_), int(ey_)),
                 (int(bl * 0.6), int(g * 0.6), int(r * 0.6)), stick * 2, cv2.LINE_AA)

    # ---- ORPHAN-DOT SUPPRESSION (2026-07-26, config/env MIRAGE_FREE_END_PRUNE) ----
    # The joint loop below draws a circle for every `ok` joint, INDEPENDENTLY of whether any bone
    # touching it survived. Two things routinely leave a joint with no bone: `_bone_debounce`
    # (which suppresses a bone whose run is shorter than debounce_min_run but not its endpoints)
    # and the "drop" open-chain policy. MEASURED on a10 (100 f): the right wrist is emitted in 16
    # frames in runs of [1,1,6,1,2,5], so the debouncer removes the forearm on 5 of them and the
    # render shows an ELBOW DOT and a WRIST DOT floating in open space with nothing between them
    # (visible at f45 in FREE_END_STICKS.png). A pose-conditioned generator has no way to read an
    # isolated dot as anything but a body part it must produce.
    # Head joints are exempt for the same reason they are exempt everywhere else in this pipeline
    # (the head is the one thing always in shot), and a lone person with NO bones at all still
    # renders, so this can never blank a frame that previously carried a skeleton.
    # Information-REMOVING, so it cannot weaken §2 or the anti-reID posture.
    # DEFAULT "1" since 2026-07-26, to MATCH config.POSE_FREE_END_PRUNE, which the user signed off
    # after the before/after render. These two halves must agree: Tier-1 prunes the joint out of
    # pose.json, this prunes the orphaned DOT out of the rendered conditioning image, and the cloud
    # only ever sees the render (build_cloud_bundle renders the stick video locally and uploads
    # that, never pose.json). A default mismatch between them silently ships half the fix.
    if os.environ.get("MIRAGE_FREE_END_PRUNE", "1") not in ("0", "false", "False") \
            and _bone_drawn.any():
        for _oi in range(18):
            if _oi in (0, 1, 14, 15, 16, 17):     # nose, neck, eyes, ears
                continue
            if ok[_oi] and not _bone_drawn[_oi]:
                ok[_oi] = False
    for oi in range(18):
        if ok[oi]:
            r, g, bl = _OP_COLORS[oi % 18]
            cv2.circle(img, (int(pts[oi][0]), int(pts[oi][1])), joint_r, (bl, g, r), -1, cv2.LINE_AA)
    # hands: bone LINES (the legacy 2 px dots are 0.25 latent px after the VAE - invisible)
    for hand in (L_HAND, R_HAND):
        hv = [i for i in hand if i < len(kp) and sc[i] > 0
              and (abs(kp[i][0]) > 1e-6 or abs(kp[i][1]) > 1e-6)]
        if len(hv) < 8:                       # a partial hand draws spider legs - skip
            continue
        b0 = hand[0]
        for ei, (u, v) in enumerate(_HAND_EDGES):
            iu, iv = b0 + u, b0 + v
            if iu in hv and iv in hv:
                r, g, bl = [int(255 * c) for c in colorsys.hsv_to_rgb(ei / len(_HAND_EDGES), 1.0, 1.0)]
                cv2.line(img, (int(kp[iu][0]), int(kp[iu][1])), (int(kp[iv][0]), int(kp[iv][1])),
                         (bl, g, r), hand_t, cv2.LINE_AA)


def _load_pose(work):
    d = json.load(open(os.path.join(work, "pose.json"), encoding="utf-8"))
    return d.get("frames", []), d.get("anon", {}), d


def read_video(path):
    import cv2
    cap = cv2.VideoCapture(path)
    out, fps = [], cap.get(cv2.CAP_PROP_FPS)
    while True:
        ok, f = cap.read()
        if not ok:
            break
        out.append(f)
    cap.release()
    return out, float(fps or 10.0)


def _writer(path, wh, fps):
    import cv2
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    return cv2.VideoWriter(path, fourcc, max(1.0, fps), wh)


def _draw_skeleton(img, kp, score, color, thick=3, rad=4):
    """Draw the body-17 sticks + hand dots for one person onto img (in place)."""
    import cv2
    def ok(i):
        return score[i] > 0 and (kp[i][0] > 0 or kp[i][1] > 0)
    for a, b in SKELETON:
        if a < len(kp) and b < len(kp) and ok(a) and ok(b):
            cv2.line(img, (int(kp[a][0]), int(kp[a][1])), (int(kp[b][0]), int(kp[b][1])), color, thick, cv2.LINE_AA)
    # NECK. COCO-17 has no neck joint, so the head block draws disconnected from the body --
    # a floating cluster above a torso. DWPose/OpenPose renders, which pose-conditioned video
    # models are trained on, DO show a neck, so add nose <-> shoulder-midpoint. Purely a
    # drawing edge: no keypoint is invented or emitted.
    if len(kp) > 6 and ok(0) and ok(5) and ok(6):
        mx = int(round((kp[5][0] + kp[6][0]) / 2)); my = int(round((kp[5][1] + kp[6][1]) / 2))
        cv2.line(img, (int(kp[0][0]), int(kp[0][1])), (mx, my), color, thick, cv2.LINE_AA)
    for i in range(17):
        if ok(i):
            cv2.circle(img, (int(kp[i][0]), int(kp[i][1])), rad, (30, 30, 30), -1, cv2.LINE_AA)
            cv2.circle(img, (int(kp[i][0]), int(kp[i][1])), rad - 1, color, -1, cv2.LINE_AA)
    for hand in (L_HAND, R_HAND):
        for i in hand:
            if i < len(kp) and ok(i):
                cv2.circle(img, (int(kp[i][0]), int(kp[i][1])), 2, color, -1, cv2.LINE_AA)


def _banner(img, text, sub=None):
    import cv2
    cv2.rectangle(img, (0, 0), (img.shape[1], 34), (18, 18, 22), -1)
    cv2.putText(img, text, (10, 24), cv2.FONT_HERSHEY_SIMPLEX, 0.62, (235, 235, 235), 2, cv2.LINE_AA)
    if sub:
        cv2.putText(img, sub, (10, img.shape[0] - 12), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (150, 220, 150), 1, cv2.LINE_AA)


def _bone_debounce(frames, min_run=3):
    """Per (slot, COCO bone) timeline gate: suppress drawable runs SHORTER than `min_run`.

    MEASURED motivation (v3, N=100, STICK_QUALITY_v3.json): the right forearm was drawable in
    16/100 frames across 12 valid<->invalid transitions - a stick that pops in and out of
    existence for 1-3 frames at a time is a temporal artifact in the conditioning stream.
    The debouncer only ever REMOVES short apparitions (never invents a joint), so it cannot
    add information; head bones are exempt (they are template-stamped and stable)."""
    ok = {}                                       # (slot, joint) -> [T] validity
    T = len(frames)
    for e in range(T):
        for pi, person in enumerate(frames[e]):
            slot = person.get("slot", pi)
            kp = person.get("kp", []); sc = person.get("score", [1] * len(kp))
            for j in range(min(17, len(kp))):
                key = (slot, j)
                if key not in ok:
                    ok[key] = np.zeros(T, bool)
                ok[key][e] = sc[j] > 0 and (abs(kp[j][0]) > 1e-6 or abs(kp[j][1]) > 1e-6)
    gate = {}                                     # (slot, a, b) -> [T] draw?
    slots = sorted({s for (s, _) in ok})
    for slot in slots:
        for a, b in SKELETON:
            if a < 5 and b < 5:                   # head bones exempt
                continue
            va = ok.get((slot, a)); vb = ok.get((slot, b))
            if va is None or vb is None:
                continue
            drawable = va & vb
            g = drawable.copy()
            t = 0
            while t < T:
                if drawable[t]:
                    t0 = t
                    while t < T and drawable[t]:
                        t += 1
                    if t - t0 < min_run:
                        g[t0:t] = False
                else:
                    t += 1
            gate[(slot, a, b)] = g
    return gate


def make_pose_sticks(masked, frames, out_path, fps, on_black=True, style="dwpose",
                     debounce_min_run=3):
    """Anonymised skeleton -- the exact pose.json that leaves the device.

    `on_black` (DEFAULT, user 2026-07-25): render the skeleton on a BLACK canvas.

    WHY: this video is not a debug view, it is the CONDITIONING SIGNAL for
    WanVideoAnimateEmbeds.pose_images. Measured on the 900 f / 1264^2 video-1 sidecar while
    it still drew over the masked frame: the skeleton was **0.653 % of the frame** and
    **97.9 % was scene content**, plus a baked-in text banner across the top 34 rows.
    Pose-conditioned video models are trained on skeleton-on-black, so we were spending
    >99 % of the conditioning image on pixels the encoder has to learn to ignore -- and at
    8x VAE downsampling what little skeleton there was became a handful of latent pixels.

    `style` (2026-07-26, ⚠️ PIXEL-AFFECTING - render + user approval before shipping):
      "dwpose"  (default) OpenPose-format render: colored filled-ellipse limbs, per-joint
                colored circles, hand BONES as colored lines, stroke width scaled to the
                canvas (4 px at 512 -> ~10 px at 1264; the legacy 3 px stroke is 0.375
                latent px after the VAE's 8× downsample). Also debounces 1-2-frame bone
                apparitions (see _bone_debounce).
      "legacy"  the previous monochrome 3 px wireframe (per-person PCOLORS, hand dots).

    Pass on_black=False for the human-readable overlay used in the app's Tier-1 tiles.
    """
    H, W = masked[0].shape[:2]
    base = min(H, W) / 512.0
    vw = _writer(out_path, (W, H), fps)
    n = min(len(masked), len(frames))
    gate = _bone_debounce(frames[:n], debounce_min_run) if (style == "dwpose" and debounce_min_run > 1) else {}
    for e in range(n):
        img = np.zeros_like(masked[e]) if on_black else masked[e].copy()
        for pi, person in enumerate(frames[e]):
            kp = person.get("kp", []); sc = person.get("score", [1] * len(kp))
            if style == "dwpose":
                slot = person.get("slot", pi)
                if gate:
                    def bg(a, b, _s=slot, _e=e):
                        g = gate.get((_s, a, b))
                        return True if g is None else bool(g[_e])
                else:
                    bg = None
                _draw_person_dwpose(img, kp, sc, base, bone_gate=bg)
            else:
                _draw_skeleton(img, kp, sc, PCOLORS[pi % len(PCOLORS)])
        if not on_black:
            # the banner is baked INTO the pixels, so it must never appear in a
            # conditioning image -- debug overlay only.
            _banner(img, "TIER-1 - anonymised pose (pose.json, emitted)",
                    "identity-free - face landmarks zeroed")
        vw.write(img)
    vw.release()
    return n


def make_canonical(frames, out_path, fps, wh):
    """The canonicaliser view: each person's emitted skeleton root-centred + scaled onto a neutral
    panel. Proportions have already collapsed to the population template (pose_anon_edge)."""
    import cv2
    W, H = wh
    vw = _writer(out_path, (W, H), fps)
    cx, target_torso = W // 2, H * 0.32
    for e in range(len(frames)):
        img = np.full((H, W, 3), 22, np.uint8)
        for gx in range(0, W, 60):
            cv2.line(img, (gx, 34), (gx, H), (32, 32, 38), 1)
        for gy in range(34, H, 60):
            cv2.line(img, (0, gy), (W, gy), (32, 32, 38), 1)
        people = frames[e]
        for pi, person in enumerate(people):
            kp = np.array(person.get("kp", []), np.float64)
            sc = person.get("score", [1] * len(kp))
            if len(kp) < 17:
                continue
            body = kp[:17]
            mh = 0.5 * (body[11] + body[12]); ms = 0.5 * (body[5] + body[6])
            torso = float(np.linalg.norm(ms - mh)) or 1.0
            s = target_torso / torso
            slot_x = cx if len(people) == 1 else int(W * (pi + 1) / (len(people) + 1))
            disp = (body - mh) * s + np.array([slot_x, H * 0.55])
            k2 = kp.copy(); k2[:17] = disp
            _draw_skeleton(img, k2.tolist(), sc, PCOLORS[pi % len(PCOLORS)], thick=3, rad=5)
        _banner(img, "TIER-1 - canonicaliser (identity-collapsed)",
                "limb proportions -> ONE population template - non-invertible (k-same)")
        vw.write(img)
    vw.release()
    return len(frames)


# ---------------------------------------------------------- synthetic canonical face template
# A FIXED generic face in local coords: origin at the eye-midpoint, x = right, y = down, and the
# inter-ocular distance = 1.0. Nothing here is derived from any real person -- it is a placeholder
# "mesh" for the face-normalisation stage, positioned with the identity-free head keypoints only.
def _face_template():
    ring = [(1.05 * math.cos(t), 0.40 + 1.40 * math.sin(t))
            for t in np.linspace(0, 2 * math.pi, 26, endpoint=False)]
    inner = [(0.62 * x, 0.40 + 0.62 * (y - 0.40)) for (x, y) in ring]
    polylines = [ring + [ring[0]], inner + [inner[0]]]                 # oval + inner ring (mesh feel)
    mesh = [(ring[i], inner[i]) for i in range(len(ring))]             # radial rungs oval<->inner
    center = (0.0, 0.40)
    spokes = [(center, inner[i]) for i in range(0, len(inner), 2)]     # spokes to inner ring
    # features (each a closed/open polyline in local coords)
    def ellipse(cx, cy, rx, ry, nn=16, close=True):
        pts = [(cx + rx * math.cos(t), cy + ry * math.sin(t)) for t in np.linspace(0, 2 * math.pi, nn)]
        return pts + ([pts[0]] if close else [])
    eyes = [ellipse(-0.5, 0.0, 0.22, 0.12), ellipse(0.5, 0.0, 0.22, 0.12)]
    iris = [ellipse(-0.5, 0.0, 0.06, 0.06), ellipse(0.5, 0.0, 0.06, 0.06)]
    brows = [[(-0.74, -0.24), (-0.5, -0.30), (-0.26, -0.24)], [(0.26, -0.24), (0.5, -0.30), (0.74, -0.24)]]
    nose = [[(0.0, 0.02), (0.0, 0.52)], [(-0.14, 0.55), (0.0, 0.62), (0.14, 0.55)]]
    mouth = [ellipse(0.0, 0.92, 0.34, 0.11)]
    return polylines + eyes + iris + brows + nose + mouth, mesh + spokes


_FACE_POLYS, _FACE_MESH = _face_template()


def _draw_face_at(img, mid, ex, io, bold=1):
    """Draw the fixed synthetic template at eye-mid `mid`, axis `ex`, inter-ocular `io` (px).

    `bold` scales stroke width. This image is CONDITIONING for
    WanVideoAnimateEmbeds.face_images, and the VAE downsamples 8x before the model sees it,
    so 1 px hairlines on a 1264^2 canvas survive as sub-pixel noise. Thicker strokes cost
    nothing and are what actually reaches the encoder.
    """
    import cv2
    ey = np.array([-ex[1], ex[0]])                            # perpendicular, pointing "down"
    def tf(local):
        return [(int(round((mid + io * (lx * ex + ly * ey))[0])),
                 int(round((mid + io * (lx * ex + ly * ey))[1]))) for (lx, ly) in local]
    for a, b in _FACE_MESH:                                   # faint mesh rungs first
        (ax, ay), (bx, by) = tf([a, b])
        cv2.line(img, (ax, ay), (bx, by), (60, 140, 60), max(1, bold), cv2.LINE_AA)
    for poly in _FACE_POLYS:                                  # feature polylines on top
        cv2.polylines(img, [np.array(tf(poly), np.int32)], False, FACE_CLR,
                      max(1, 2 * bold), cv2.LINE_AA)


def _smooth1d(x, sigma):
    """Gaussian smooth a 1-D trace (edge-padded). sigma in FRAMES; <=0 is a no-op."""
    x = np.asarray(x, float)
    if sigma <= 0 or x.size < 3:
        return x
    rad = max(1, int(round(3 * sigma)))
    k = np.exp(-0.5 * (np.arange(-rad, rad + 1) / sigma) ** 2)
    k /= k.sum()
    return np.convolve(np.pad(x, rad, mode="edge"), k, mode="valid")


def _tf_poly(mid, ex, io, local):
    """Transform local face coords (eye-mid origin, io=1) -> int32 pixel polygon."""
    ey = np.array([-ex[1], ex[0]])
    return np.array([[int(round((mid + io * (lx * ex + ly * ey))[0])),
                      int(round((mid + io * (lx * ex + ly * ey))[1]))] for lx, ly in local],
                    np.int32)


def _ellipse_local(cx, cy, rx, ry, n=32, a0=0.0, a1=2 * math.pi):
    return [(cx + rx * math.cos(t), cy + ry * math.sin(t)) for t in np.linspace(a0, a1, n)]


def _draw_face_cartoon(img, mid, ex, io, bold=3):
    """A FLAT, FILLED cartoon face (fixed generic template - no real geometry). Replaces the
    green wireframe 'mesh' style: filled opaque shapes read as a face after the VAE's 8×
    downsample, where 1-2 px green hairlines average away against the black canvas. The
    template is a shared constant for every subject - identity-free by construction."""
    import cv2
    SKIN, SKIN_EDGE = (150, 190, 240), (70, 100, 150)
    HAIR = (60, 70, 90)
    WHITE, PUPIL = (245, 245, 245), (35, 35, 35)
    BROW, NOSE_C = (50, 65, 90), (90, 120, 170)
    MOUTH = (70, 70, 190)
    b = max(1, bold)
    # head
    head = _tf_poly(mid, ex, io, _ellipse_local(0.0, 0.35, 1.30, 1.65))
    cv2.fillConvexPoly(img, head, SKIN, cv2.LINE_AA)
    cv2.polylines(img, [head], True, SKIN_EDGE, b, cv2.LINE_AA)
    # hair cap - arc only above the brow line: the closing chord sits at y=-0.45, above the
    # brows (-0.30), so it can never cut across the eyes (the first draft's chord at +0.05
    # read as a bandit mask across the face).
    hair = _ellipse_local(0.0, -0.45, 1.24, 0.92, a0=math.pi, a1=2 * math.pi)
    cv2.fillPoly(img, [_tf_poly(mid, ex, io, hair + [hair[0]])], HAIR, cv2.LINE_AA)
    # eyes
    for sx in (-0.5, 0.5):
        cv2.fillConvexPoly(img, _tf_poly(mid, ex, io, _ellipse_local(sx, 0.0, 0.27, 0.17)),
                           WHITE, cv2.LINE_AA)
        cv2.fillConvexPoly(img, _tf_poly(mid, ex, io, _ellipse_local(sx, 0.02, 0.09, 0.09, 16)),
                           PUPIL, cv2.LINE_AA)
    # brows
    for sx in (-1, 1):
        brow = [(sx * 0.76, -0.30), (sx * 0.50, -0.38), (sx * 0.24, -0.30)]
        cv2.polylines(img, [_tf_poly(mid, ex, io, brow)], False, BROW,
                      max(2, 2 * b), cv2.LINE_AA)
    # nose
    cv2.polylines(img, [_tf_poly(mid, ex, io, [(0.0, 0.10), (0.0, 0.52)])], False, NOSE_C,
                  max(2, b), cv2.LINE_AA)
    cv2.polylines(img, [_tf_poly(mid, ex, io, [(-0.13, 0.56), (0.0, 0.63), (0.13, 0.56)])],
                  False, NOSE_C, max(2, b), cv2.LINE_AA)
    # smiling mouth (filled lower arc + chord)
    smile = _ellipse_local(0.0, 0.82, 0.40, 0.24, n=24, a0=0.15 * math.pi, a1=0.85 * math.pi)
    cv2.fillPoly(img, [_tf_poly(mid, ex, io, smile + [smile[0]])], MOUTH, cv2.LINE_AA)


# =============================================================================================
# SYNTHETIC CANONICAL HEAD - 2.5D/3D RIG  (rebuild 2026-07-26, style="head3d")
# =============================================================================================
# WHY THE REBUILD (user verdict): "the canonicaliser face is shit, it has no lips, eyes or
# expression movement and that sharp neck and face tilts".
#
# The three things that were wrong, and what replaces each:
#
#  1. NO EXPRESSION. `_draw_face_cartoon` is 100 % hardcoded constants and `make_synthetic_face`
#     never read face_scalars.json at all - verified by signature + grep. The rig below is
#     parameterised by the SAME 12 channels Tier-1 emits, so lips open/widen/curve, eyelids close
#     over an iris, brows lift and the irises track gaze.
#
#  2. A SPRITE, NOT A HEAD. The old template was a 2-D polygon set rotated by one screen angle.
#     Here every vertex has a real z on a shared canonical ELLIPSOID skull; the head is posed by a
#     true 3x3 R = Rz(roll)·Ry(yaw)·Rx(pitch) and projected weak-perspective, the silhouette is the
#     exact projected ellipsoid outline (closed form via the SVD of the 2x3 projection matrix, so
#     it narrows correctly as the head turns), and features are back-face culled by their surface
#     normal. Turning the head therefore looks like a head turning.
#
#  3. SHARP TILTS. Measured on the shipped facemesh_p1_00002.mp4 (N=100 @10 fps): rendered roll
#     moved 1.49 deg/frame mean, 4.70 deg/frame peak, and the face centre 16.84 px/frame mean,
#     54.07 px peak. Roll was taken straight from the anonymised eye vector, whose own jitter is
#     1.79 deg/frame - sigma=2.0 Gaussian only removed ~17 % of it. Here roll/position/scale are
#     Gaussian-smoothed AND passed through a rate limiter + critically-damped 2nd-order follower
#     (no overshoot, C1-continuous), and the body-lean term is attenuated (see `lean_gain`).
#
# ---------------------------------------------------------------------------------------------
# PRIVACY - this rig is identity-free BY CONSTRUCTION, and the drive is gated twice.
# ---------------------------------------------------------------------------------------------
#   * ONE SHARED TEMPLATE. Every geometric constant below is a module-level literal. Nothing is
#     derived from the subject: not the skull axes, not the eye separation, not the mouth width.
#     Two different people render byte-identical geometry for the same drive vector.
#   * SIZE still comes from the anonymised SKELETON (io = 0.26 x shoulder width, a fixed
#     anatomical constant) - never from eye separation or any face scalar.
#   * The 12 channels may only modulate EXPRESSION and HEAD POSE. They never touch face SHAPE.
#   * GATE 1 (privacy allow-list, `_ALLOW_AC`): only mouth_open(0), smile(2), and the SYMMETRIC
#     means of eye_open(3,4) and brow(5,6) may ever be data-driven. mouth_width(1), yaw(7),
#     pitch(8), roll(9), gazeX(10), gazeY(11) are HARD DENIED as drives: 1/7/8/10/11 are ratios in
#     the subject's own face frame (mouth-corner separation / inter-ocular; the nose-TIP coordinate
#     in the eye frame; iris offset / inter-ocular) i.e. anthropometric shape welded to motion, and
#     9 is a redundant second head-orientation channel the pose path already carries. The L/R
#     DIFFERENCES (3-4, 5-6) are denied too - facial asymmetry is a strong per-subject constant - 
#     so the rig forcibly symmetrises both eyes and both brows.
#   * GATE 2 (utility trust gate, measured): even an allow-listed channel is used only if
#     `face_signal_filter` reports trust=True for it, i.e. the shrinkage gain against the KNOWN
#     Laplace noise floor is >= 0.05. MEASURED at the shipped DP_EPSILON_TOTAL=3.0 on a10/a15/a30:
#     trusted channels = NONE, gains 0.00055-0.00518 (N = 36 channel x release cells). So TODAY
#     100 % of the drive is `idle_animation` + `_idle_expression`, which are DATA-INDEPENDENT: they
#     never read the released scalars, cost zero privacy budget, and cannot leak identity,
#     expression or speech. Raise DP_EPSILON_TOTAL and the same code starts using real motion with
#     no edit - but that is a declarable policy change for the user, not an agent.
#   * The procedural generators are seeded from a CONSTANT (`idle_seed`, default 0). The seed must
#     NEVER be derived from the slot id, clip hash, subject count or any data-derived quantity - a
#     data-seeded "random" motion is a fingerprint.
#   * Black background stays mandatory (`on_black=True`), and build_cloud_bundle still hands the
#     renderer an all-zero stand-in so no real pixel is ever decoded while this draws.
# ---------------------------------------------------------------------------------------------

# --- the shared canonical skull: an ellipsoid, origin at the eye-midpoint, x right, y DOWN,
#     z out of the face, unit = inter-ocular distance. Fixed literals, identical for everyone.
_SK_A, _SK_B, _SK_C = 1.30, 1.62, 1.18          # semi-axes
_SK_CX, _SK_CY, _SK_CZ = 0.0, 0.35, -0.16       # centre
_FOCAL = 8.0                                    # weak-perspective focal length, in io units

# palette (BGR). Chosen for SEPARATION after the encoder's downsample, not for realism: the face
# branch maps the image x*2-1, so only large opaque high-chroma fills register. Each part gets a
# unique exact colour so the render can be MEASURED back out of the mp4 by palette matching.
_C_SKIN = (150, 190, 240)
_C_SHADE = (118, 155, 205)
_C_HAIR = (58, 66, 88)
_C_WHITE = (245, 245, 245)
_C_IRIS = (155, 95, 45)
_C_PUPIL = (22, 22, 22)
_C_BROW = (48, 62, 88)
_C_LIP = (78, 78, 205)
_C_CAVITY = (36, 24, 62)          # mouth interior - the aperture measurement key
_C_TEETH = (218, 228, 250)      # deliberately separable from _C_WHITE so the render is measurable
_C_NECK = (104, 138, 186)

# privacy allow-list: the ONLY channels that may ever be data-driven (AC component only, and
# symmetrised). Everything else is procedural for all time. See the header block above.
_ALLOW_AC = (0, 2, 3, 4, 5, 6)


def _skull_z(x, y, infl=1.0):
    """z of the canonical skull surface at local (x, y). `infl` inflates it (features that sit
    slightly proud of the surface). Scalars or arrays."""
    a, b, c = _SK_A * infl, _SK_B * infl, _SK_C * infl
    r2 = 1.0 - ((np.asarray(x, float) - _SK_CX) / a) ** 2 - ((np.asarray(y, float) - _SK_CY) / b) ** 2
    return _SK_CZ + c * np.sqrt(np.maximum(r2, 0.04))


def _lift(pts2d, infl=1.0, dz=0.0):
    """local 2-D face-plane points -> 3-D points on (or dz in front of) the canonical skull."""
    p = np.asarray(pts2d, float).reshape(-1, 2)
    z = _skull_z(p[:, 0], p[:, 1], infl) + dz
    return np.concatenate([p, z[:, None]], axis=1)


def _skull_normal(p3):
    """outward normal of the ellipsoid at a surface point (un-normalised is fine for a sign test)."""
    p = np.asarray(p3, float).reshape(-1, 3)
    n = np.stack([(p[:, 0] - _SK_CX) / _SK_A ** 2,
                  (p[:, 1] - _SK_CY) / _SK_B ** 2,
                  (p[:, 2] - _SK_CZ) / _SK_C ** 2], axis=1)
    return n / np.maximum(np.linalg.norm(n, axis=1, keepdims=True), 1e-9)


def _rot3(yaw, pitch, roll):
    """R = Rz(roll) @ Ry(yaw) @ Rx(pitch). y is DOWN, so this is a real head rotation, not a
    screen-space sprite spin."""
    cy, sy = math.cos(yaw), math.sin(yaw)
    cp, sp = math.cos(pitch), math.sin(pitch)
    cr, sr = math.cos(roll), math.sin(roll)
    Ry = np.array([[cy, 0.0, sy], [0.0, 1.0, 0.0], [-sy, 0.0, cy]])
    Rx = np.array([[1.0, 0.0, 0.0], [0.0, cp, -sp], [0.0, sp, cp]])
    Rz = np.array([[cr, -sr, 0.0], [sr, cr, 0.0], [0.0, 0.0, 1.0]])
    return Rz @ Ry @ Rx


def _project(p3, R, mid, io):
    """rotate + weak-perspective project local 3-D points to pixel coords."""
    q = np.asarray(p3, float).reshape(-1, 3) @ R.T
    k = _FOCAL / np.maximum(_FOCAL - q[:, 2], 1e-3)
    return np.asarray(mid, float)[None, :] + io * (q[:, :2] * k[:, None])


def _ipoly(pts):
    return np.round(np.asarray(pts, float).reshape(-1, 2)).astype(np.int32)


def _head_outline(R, mid, io, n=72):
    """EXACT projected silhouette of the rotated ellipsoid, in closed form.

    Surface points are C + R·diag(a,b,c)·u with |u|=1, so the projected set is the image of the
    unit ball under the 2x3 matrix M = P·R·diag(a,b,c); its boundary is U·diag(s)·w, |w|=1, from
    the SVD M = U·diag(s)·Vt. This is why the head NARROWS correctly when it turns - a rotated
    2-D template cannot do that.
    """
    M = (R @ np.diag([_SK_A, _SK_B, _SK_C]))[:2, :]
    U, s, _ = np.linalg.svd(M, full_matrices=False)
    C = R @ np.array([_SK_CX, _SK_CY, _SK_CZ])
    k = _FOCAL / max(_FOCAL - float(C[2]), 1e-3)
    t = np.linspace(0.0, 2.0 * math.pi, n, endpoint=False)
    w = np.stack([np.cos(t), np.sin(t)], axis=0)          # [2, n]
    bnd = (U @ (s[:, None] * w)).T                        # [n, 2]
    return np.asarray(mid, float)[None, :] + io * k * (C[:2][None, :] + bnd)


def _hair_poly(R, mid, io, outline):
    """Hair cap = the front hairline arc (a real 3-D curve on the skull) closed over the top by
    the projected silhouette arc. Built this way (rather than as a 2-D chord) so the hairline
    swings with yaw/pitch and can never cut across the eyes."""
    phi = np.linspace(0.0, math.pi, 25)
    psi = 1.14 - 0.13 * np.sin(phi)        # arched: higher over the centre of the forehead
    a, b, c = _SK_A * 1.02, _SK_B * 1.02, _SK_C * 1.02
    P = np.stack([_SK_CX + a * np.sin(psi) * np.cos(phi),
                  _SK_CY - b * np.cos(psi),
                  _SK_CZ + c * np.sin(psi) * np.sin(phi)], axis=1)
    hl = _project(P, R, mid, io)
    n = len(outline)
    i0 = int(np.argmin(np.linalg.norm(outline - hl[0], axis=1)))
    i1 = int(np.argmin(np.linalg.norm(outline - hl[-1], axis=1)))

    def arc(a_i, b_i):
        idx, i = [], a_i
        for _ in range(n + 1):
            idx.append(i)
            if i == b_i:
                break
            i = (i + 1) % n
        return outline[idx]

    # BOTH candidates must run i1 -> i0 so the polygon closes on hl[0] without a chord across
    # the face (a self-intersecting ring made fillPoly punch a black lens through the forehead).
    # They differ only in which half of the silhouette they traverse; keep the UPPER one.
    up = -(R @ np.array([0.0, 1.0, 0.0]))[:2]             # local "up" in screen coords
    c1, c2 = arc(i1, i0), arc(i0, i1)[::-1]
    pick = c1 if float((c1.mean(0) - mid) @ up) > float((c2.mean(0) - mid) @ up) else c2
    return np.vstack([hl, pick])


def _qcurve(p0, p1, p2, n=13):
    """quadratic Bezier through p0 -> p2 with control p1 (the lip / eyelid contour primitive)."""
    t = np.linspace(0.0, 1.0, n)[:, None]
    p0, p1, p2 = (np.asarray(v, float) for v in (p0, p1, p2))
    return (1 - t) ** 2 * p0 + 2 * (1 - t) * t * p1 + t ** 2 * p2


def _lip_contours(mo, mw, sm, fs):
    """Local 2-D lip geometry from mouth_open / mouth_width / smile. Returns
    (upper_outer, upper_inner, lower_inner, lower_outer), each left -> right.

    This is the feature the user said was missing entirely ("it has no lips"). Aperture,
    corner curl and width are all continuous functions of the drive, so the lips genuinely
    open, widen and curve rather than being a fixed painted arc.
    """
    ym = 0.86
    w = 0.355 * fs * (0.80 + 0.36 * float(np.clip(mw / 1.5, 0.0, 1.0)))
    ap = 0.42 * fs * float(np.clip(mo, 0.0, 1.0))
    corner = ym - 0.30 * float(np.clip(sm, -0.5, 0.5))
    t_up, t_dn = 0.17 * fs, 0.22 * fs
    L, Rp = (-w, corner), (w, corner)
    hi_in = ym - 0.5 * ap - 0.015
    lo_in = ym + 0.5 * ap + 0.015
    up_in = _qcurve(L, (0.0, 2 * hi_in - corner), Rp)
    lo_in_c = _qcurve(L, (0.0, 2 * lo_in - corner), Rp)
    up_out = _qcurve(L, (0.0, 2 * (hi_in - t_up) - corner), Rp)
    lo_out = _qcurve(L, (0.0, 2 * (lo_in + t_dn) - corner), Rp)
    # cupid's bow: a shallow dip at the centre of the upper outer edge
    m = len(up_out) // 2
    up_out[m, 1] += 0.045 * fs
    return up_out, up_in, lo_in_c, lo_out


def _eye_contour(sx, opn, fs):
    """Almond eyelid contour for one eye. `opn` is eye_open in channel units (bounds 0..0.5)."""
    hw = 0.315 * fs
    k = float(np.clip(opn / 0.30, 0.0, 1.30))
    up, dn = 0.215 * fs * k, 0.120 * fs * k
    L, Rp = (sx - hw, 0.0), (sx + hw, 0.0)
    top = _qcurve(L, (sx, -2.0 * up), Rp)
    bot = _qcurve(Rp, (sx, 2.0 * dn), L)
    return np.vstack([top, bot]), up, dn


def _fill3(img, cv2m, pts2d, color):
    p = _ipoly(pts2d)
    if len(p) >= 3:
        cv2m.fillPoly(img, [p], color, cv2m.LINE_AA)


def _draw_head3d(img, mid, io, ypr, expr, bold=3, fs=1.0):
    """Draw ONE shared canonical 3-D head, posed by `ypr` and deformed by `expr` (the 12-channel
    drive vector). `mid` = eye-midpoint in px, `io` = inter-ocular distance in px, `fs` scales the
    features inside the head.

    Nothing here reads the subject. `expr` may only change EXPRESSION and POSE; every length,
    radius and colour is a module constant shared by all people.
    """
    import cv2
    yaw, pitch, roll = float(ypr[0]), float(ypr[1]), float(ypr[2])
    R = _rot3(yaw, pitch, roll)
    mid = np.asarray(mid, float)
    b = max(1, int(round(bold * io / 60.0)))              # stroke scales with the head, not the frame

    mo, mw, sm = expr[0], expr[1], expr[2]
    eye_o = 0.5 * (expr[3] + expr[4])                     # SYMMETRIC mean only (L/R diff denied)
    brow = 0.5 * (expr[5] + expr[6])                      # SYMMETRIC mean only
    gx, gy = expr[10], expr[11]

    def vis(p_local3, thr=0.06):
        return float((_skull_normal(p_local3)[0] @ R.T)[2]) > thr

    # ---- neck (behind the head): a short shared stub so the head is not a floating ball ----
    nk = []
    for s in np.linspace(0.0, 1.0, 5):
        for sx in (-1.0, 1.0):
            nk.append((sx * (0.44 - 0.06 * s), 1.55 + 1.05 * s, -0.10))
    # projected WITHOUT R: the neck belongs to the body, so it must not swing with the head - 
    # that rigid head+neck unit is what read as the "sharp neck ... tilts".
    nkp = _project(np.array(nk, float), np.eye(3), mid, io)
    cv2.fillPoly(img, [cv2.convexHull(_ipoly(nkp))], _C_NECK, cv2.LINE_AA)

    # ---- ears (drawn before the head so the skull overlaps their inner edge) ----
    for sx in (-1.0, 1.0):
        if float((R @ np.array([sx, 0.0, 0.0]))[2]) < -0.20:
            continue
        th = np.linspace(0.0, 2.0 * math.pi, 20, endpoint=False)
        ear = np.stack([np.full_like(th, sx * 1.20),
                        0.34 + 0.30 * np.cos(th),
                        -0.12 + 0.22 * np.sin(th)], axis=1)
        _fill3(img, cv2, _project(ear, R, mid, io), _C_SHADE)

    # ---- skull ----
    outline = _head_outline(R, mid, io)
    cv2.fillPoly(img, [_ipoly(outline)], _C_SKIN, cv2.LINE_AA)
    cv2.polylines(img, [_ipoly(outline)], True, _C_SHADE, b, cv2.LINE_AA)

    # ---- hair cap ----
    try:
        cv2.fillPoly(img, [_ipoly(_hair_poly(R, mid, io, outline))], _C_HAIR, cv2.LINE_AA)
    except Exception:
        pass

    # ---- everything from here on is CLIPPED to the head silhouette: at a strong yaw the far
    #      brow/eye would otherwise project outside the skull outline and float in mid-air.
    hx0 = max(0, int(outline[:, 0].min()) - 2); hx1 = min(img.shape[1], int(outline[:, 0].max()) + 3)
    hy0 = max(0, int(outline[:, 1].min()) - 2); hy1 = min(img.shape[0], int(outline[:, 1].max()) + 3)
    clip_roi = None
    if hx1 > hx0 and hy1 > hy0:
        hmask = np.zeros((hy1 - hy0, hx1 - hx0), np.uint8)
        cv2.fillPoly(hmask, [_ipoly(outline - np.array([hx0, hy0], float))], 255)
        clip_roi = (hx0, hx1, hy0, hy1, hmask, img[hy0:hy1, hx0:hx1].copy())

    # ---- brows: raised/lowered by the symmetric brow channel ----
    dy = -(float(brow) - 0.28) * 0.50
    for sx in (-1.0, 1.0):
        p3 = _lift([[sx * 0.30, -0.30 + dy]], 1.0, 0.01)
        if not vis(p3):
            continue
        arc = _qcurve((sx * 0.78, -0.27 + dy), (sx * 0.52, -0.42 + dy), (sx * 0.24, -0.29 + dy))
        cv2.polylines(img, [_ipoly(_project(_lift(arc, 1.0, 0.012), R, mid, io))],
                      False, _C_BROW, max(2, 3 * b), cv2.LINE_AA)

    # ---- eyes: eyelid closure over a drawn iris + pupil, irises track gaze ----
    for sx in (-0.5, 0.5):
        p3c = _lift([[sx, 0.0]], 1.0, 0.01)
        if not vis(p3c):
            continue
        cont, up, dn = _eye_contour(sx, eye_o, fs)
        eye_px = _project(_lift(cont, 1.0, 0.012), R, mid, io)
        cv2.fillPoly(img, [_ipoly(eye_px)], _C_WHITE, cv2.LINE_AA)
        if up + dn > 0.035:
            # iris/pupil, clipped to the eye white inside its own bbox ROI
            ix = sx + 0.24 * float(np.clip(gx, -0.5, 0.5))
            iy = 0.02 + 0.16 * float(np.clip(gy, -0.5, 0.5))
            th = np.linspace(0.0, 2.0 * math.pi, 24, endpoint=False)
            iris = np.stack([ix + 0.145 * fs * np.cos(th), iy + 0.145 * fs * np.sin(th)], axis=1)
            pup = np.stack([ix + 0.070 * fs * np.cos(th), iy + 0.070 * fs * np.sin(th)], axis=1)
            ip = _project(_lift(iris, 1.0, 0.016), R, mid, io)
            pp = _project(_lift(pup, 1.0, 0.018), R, mid, io)
            allp = np.vstack([eye_px, ip])
            x0 = max(0, int(allp[:, 0].min()) - 2); x1 = min(img.shape[1], int(allp[:, 0].max()) + 3)
            y0 = max(0, int(allp[:, 1].min()) - 2); y1 = min(img.shape[0], int(allp[:, 1].max()) + 3)
            if x1 > x0 and y1 > y0:
                off = np.array([x0, y0], float)
                m = np.zeros((y1 - y0, x1 - x0), np.uint8)
                cv2.fillPoly(m, [_ipoly(eye_px - off)], 255, cv2.LINE_AA)
                lay = np.zeros((y1 - y0, x1 - x0, 3), np.uint8)
                cv2.fillPoly(lay, [_ipoly(ip - off)], _C_IRIS, cv2.LINE_AA)
                cv2.fillPoly(lay, [_ipoly(pp - off)], _C_PUPIL, cv2.LINE_AA)
                sel = (m > 0) & lay.any(axis=2)
                roi = img[y0:y1, x0:x1]
                roi[sel] = lay[sel]
        cv2.polylines(img, [_ipoly(eye_px)], True, _C_BROW, max(1, b), cv2.LINE_AA)

    # ---- nose: a real 3-D wedge, so yaw gives it a profile ----
    npts = np.array([[0.0, 0.06], [0.14, 0.50], [0.0, 0.62], [-0.14, 0.50]], float)
    ndz = np.array([0.05, 0.20, 0.30, 0.20])
    n3 = _lift(npts, 1.0, 0.0)
    n3[:, 2] += ndz
    _fill3(img, cv2, _project(n3, R, mid, io), _C_SHADE)
    for sx in (-1.0, 1.0):
        th = np.linspace(0.0, 2.0 * math.pi, 12, endpoint=False)
        nos = np.stack([sx * 0.085 + 0.040 * np.cos(th), 0.565 + 0.024 * np.sin(th)], axis=1)
        n3b = _lift(nos, 1.0, 0.22)
        _fill3(img, cv2, _project(n3b, R, mid, io), _C_BROW)

    # ---- lips + mouth cavity + teeth ----
    up_out, up_in, lo_in, lo_out = _lip_contours(mo, mw, sm, fs)
    P = lambda a2, dz: _project(_lift(a2, 1.005, dz), R, mid, io)
    cav = np.vstack([up_in, lo_in[::-1]])
    ap_px = float(np.clip(mo, 0.0, 1.0))
    if ap_px > 0.06:
        _fill3(img, cv2, P(cav, 0.008), _C_CAVITY)
        # teeth strip just under the upper inner edge - high-contrast, only when clearly open
        if ap_px > 0.22:
            band = np.vstack([up_in, (up_in + np.array([0.0, 0.30 * 0.42 * ap_px]))[::-1]])
            _fill3(img, cv2, P(band, 0.010), _C_TEETH)
    _fill3(img, cv2, P(np.vstack([up_out, up_in[::-1]]), 0.014), _C_LIP)
    _fill3(img, cv2, P(np.vstack([lo_in, lo_out[::-1]]), 0.014), _C_LIP)

    # ---- restore every pixel a feature painted OUTSIDE the skull silhouette ----
    if clip_roi is not None:
        x0, x1, y0, y1, hmask, before = clip_roi
        out = hmask == 0
        img[y0:y1, x0:x1][out] = before[out]


# ---------------------------------------------------------------------------------------------
# the drive: DATA-INDEPENDENT by default, data-driven only where MEASURED to be trustworthy
# ---------------------------------------------------------------------------------------------
def _import_face_filter():
    """face_signal_filter lives in _e2e/tools (repo-local analysis code). Imported lazily so
    tier1_viz still works on the phone / in a stripped checkout where _e2e is absent."""
    try:
        import face_signal_filter as F                     # already importable?
        return F
    except Exception:
        pass
    p = HERE            # face_signal_filter.py ships beside this module
    if p not in sys.path:
        sys.path.insert(0, p)
    try:
        import face_signal_filter as F
        return F
    except Exception:
        return None


_FALLBACK_NEUTRAL = [0.05, 0.55, 0.00, 0.25, 0.25, 0.28, 0.28, 0.0, 0.0, 0.0, 0.0, 0.0]
_FALLBACK_BOUNDS = [(0.0, 1.0), (0.0, 1.5), (-0.5, 0.5), (0.0, 0.5), (0.0, 0.5), (0.0, 0.6),
                    (0.0, 0.6), (-1.5, 1.5), (-1.5, 1.5), (-1.0, 1.0), (-0.5, 0.5), (-0.5, 0.5)]


def _idle_expression(n, fps, seed=0, F=None):
    """DATA-INDEPENDENT idle drive -> [n, 12]. Head drift + blinks come from
    face_signal_filter.idle_animation; this adds the MOUTH/SMILE motion the user asked for
    ("no lips ... or expression movement").

    PRIVACY: reads nothing. It is the same generator for every subject - exactly like the shared
    face template - so it costs zero DP budget and carries zero identity. The mouth schedule is
    deliberately UNCORRELATED with the subject and band-limited well below speech rates, so it is
    not a viseme sequence and is not lip-readable (the audio channel is out of scope by user
    decision and must not sneak back in through the lips).
    """
    n = int(max(0, n))
    if n == 0:
        return np.zeros((0, 12))
    if F is not None:
        y = np.asarray(F.idle_animation(n, fps, seed=seed), float).copy()
        lo = np.array([b[0] for b in F.CH_BOUNDS]); hi = np.array([b[1] for b in F.CH_BOUNDS])
    else:
        y = np.tile(np.array(_FALLBACK_NEUTRAL, float), (n, 1))
        lo = np.array([b[0] for b in _FALLBACK_BOUNDS]); hi = np.array([b[1] for b in _FALLBACK_BOUNDS])
    t = np.arange(n) / max(float(fps), 1e-6)
    rs = np.random.default_rng(int(seed) + 7717)
    ph = rs.uniform(0, 2 * np.pi, 6)
    # MOUTH RATE (user 2026-07-26: "moving too slowly ... it should be natural").
    # Human relaxed speech runs 4-6 syllables/s; the previous 0.55 Hz carrier was ~8x too slow
    # and read as a slow-motion chew. But the rate cannot simply be raised to 4.5 Hz: the drive
    # is sampled at EMIT_FPS, so above fps/4 there are fewer than 4 samples per open-close cycle
    # and the mouth STROBES instead of speaking (at 10 fps, 4.5 Hz is 2.2 frames/cycle -- past
    # Nyquist-usable). So the carrier is the natural rate CAPPED by the sampling limit, which
    # generalises to any EMIT_FPS instead of hardcoding one clip's number:
    #     10 fps -> 2.50 Hz | 15 fps -> 3.75 Hz | 30 fps -> 4.50 Hz
    SYLLABLE_HZ = 4.5                      # relaxed conversational speech
    carrier_hz = min(SYLLABLE_HZ, float(fps) / 4.0)
    PHRASE_HZ = 0.30                       # speaking bursts ~3.3 s apart (was 0.068 = every 15 s)
    # DUTY CYCLE. The squared envelope left the mouth active in only 39-43 % of frames
    # (measured), so even at the right carrier rate it read as long silences with occasional
    # chewing. A person in conversation is articulating roughly two thirds of the time. Keep the
    # phrase structure but put a floor under it and soften the shaping.
    env = 0.30 + 0.70 * np.clip(0.5 + 0.5 * np.sin(2 * np.pi * PHRASE_HZ * t + ph[0]),
                                0.0, 1.0) ** 1.4
    car = 0.5 + 0.5 * np.sin(2 * np.pi * carrier_hz * t + ph[1])
    y[:, 0] = 0.04 + 0.62 * env * car
    y[:, 1] = 0.55 + 0.16 * np.sin(2 * np.pi * 0.28 * t + ph[2])
    y[:, 2] = 0.26 * np.sin(2 * np.pi * 0.17 * t + ph[3]) + 0.08 * np.sin(2 * np.pi * 0.41 * t + ph[4])
    # Smoothing must not flatten the carrier it is smoothing: keep sigma well under a quarter
    # period. The old 0.10*fps was 1.0 frame at 10 fps, which was fine for 0.55 Hz and would
    # erase 2.5 Hz.
    sig = max(0.5, min(0.06 * float(fps), 0.20 / max(carrier_hz, 1e-6) * float(fps)))
    for c in (0, 1, 2):
        y[:, c] = _smooth1d(y[:, c], sig)
    # BLINKS: idle_animation uses a 0.12 s closure, which at EMIT_FPS=10 is ONE frame and its own
    # sigma=0.6 smoothing leaves the lid only ~39 % closed (measured: eye_open min 0.097 against a
    # 0.25 neutral). Re-inject a >=2-frame closure so the eyelid visibly moves - the user's
    # complaint was specifically that the eyes never move. Still fully data-independent.
    dur = max(2, int(round(0.16 * float(fps))))
    k = rs.uniform(1.0, 3.0)
    while k < (t[-1] if n > 1 else 0.0) + 1e-9:
        i0 = int(round(k * fps))
        for j in range(dur):
            if 0 <= i0 + j < n:
                y[i0 + j, 3] = y[i0 + j, 4] = 0.015
        k += rs.uniform(2.5, 5.0)
    for c in (3, 4):
        y[:, c] = _smooth1d(y[:, c], 0.45)
    return np.clip(y, lo, hi)


def _damp1d(x, fps, tau, max_rate=None):
    """rate limit + critically-damped 2nd-order follower. No overshoot, C1-continuous output.
    THIS is the anti-'sharp neck and face tilt' stage; a Gaussian alone was measured to remove
    only ~17 % of the roll jitter (1.79 -> 1.49 deg/frame)."""
    x = np.asarray(x, float)
    if x.size == 0:
        return x.copy()
    dt = 1.0 / max(float(fps), 1e-6)
    y = x.copy()
    if max_rate:
        step = float(max_rate) * dt
        o = np.empty_like(y); o[0] = y[0]
        for i in range(1, y.size):
            o[i] = o[i - 1] + float(np.clip(y[i] - o[i - 1], -step, step))
        y = o
    w = 1.0 / max(float(tau), 1e-6)
    pos, vel = float(y[0]), 0.0
    o = np.empty_like(y)
    for i in range(y.size):
        vel += (w * w * (y[i] - pos) - 2.0 * w * vel) * dt
        pos += vel * dt
        o[i] = pos
    return o


def _build_drive(n, fps, face_scalars, slot, idle_seed):
    """[n,12] drive vector for one person slot + a note on what was actually used.

    GATE 1 privacy allow-list, then GATE 2 measured trust. Anything that fails either gate is
    filled from the data-independent procedural generator.
    """
    F = _import_face_filter()
    drive = _idle_expression(n, fps, seed=idle_seed, F=F)
    used, gains = [], {}
    if face_scalars is not None and F is not None:
        try:
            res = F.filter_face_scalars(face_scalars, fps)
            vals = res.values                                     # [T,P,12], clamped + C1-smooth
            p = min(int(slot), vals.shape[1] - 1)
            m = min(n, vals.shape[0])
            for c in _ALLOW_AC:                                   # GATE 1
                r = res.reports[p][c]
                gains[F.CHANNELS[c]] = round(float(r.gain), 6)
                if r.trust:                                       # GATE 2 (measured)
                    drive[:m, c] = vals[:m, p, c]
                    used.append(F.CHANNELS[c])
        except Exception as ex:                                   # never fail the render on this
            gains["_error"] = repr(ex)
    # enforce symmetry: L/R DIFFERENCES are denied outright (facial asymmetry is per-subject)
    drive[:, 4] = drive[:, 3]
    drive[:, 6] = drive[:, 5]
    # mouth_width / yaw / pitch / roll / gaze are never data-driven; they are already procedural.
    return drive, used, gains


def make_synthetic_face(masked, mask_frames, out_path, fps, on_black=True,
                        pose_frames=None, bold=3, style="head3d", smooth_sigma=3.0,
                        face_scalars=None, idle_seed=0, lean_gain=0.35, pos_tau=0.20,
                        feature_scale=1.0, crop512_path=None, debug_json=None):
    """Draw ONE SHARED synthetic canonical head per person, posed in 3-D and animated.

    REBUILD 2026-07-26 (⚠️ PIXEL-AFFECTING - render + user approval before shipping). What
    changed and why, against the user's verdict "no lips, eyes or expression movement and that
    sharp neck and face tilts":

      * `style="head3d"` (NEW DEFAULT) - the 2.5D/3D rig above: a shared canonical ELLIPSOID
        skull, posed by a real R = Rz(roll)·Ry(yaw)·Rx(pitch) with weak-perspective projection and
        a closed-form projected silhouette, back-face culled features, a neck stub and ears.
        LIPS (upper + lower contour that opens, widens and curls with smile), EYELIDS that close
        over an IRIS + PUPIL, BROWS that lift, and irises that track gaze - all continuous
        functions of the 12-channel drive. `style="cartoon"` / `"mesh"` keep the two old looks.
      * THE FACE IS NOW DRIVEN. `make_synthetic_face` previously ignored all 12 channels (verified:
        the old signature had no scalar argument and `_draw_face_cartoon` was 100 % literals),
        which is exactly why there was no expression. Pass `face_scalars` (the parsed
        face_scalars.json or its path) to feed the rig.
      * ROLL/YAW/PITCH come from the SMOOTHED DRIVE, not from the shoulder line and not raw from
        the eye vector. The body only contributes an attenuated, heavily smoothed LEAN
        (`lean_gain`, default 0.35 - measured real head/shoulder roll correlation is ~0.50, and
        the legacy shoulder basis welded them at corr 1.0000).
      * ANTI-SNAP. Position, scale and roll each get a Gaussian pre-smooth (`smooth_sigma`, frames)
        AND a rate limiter + critically-damped 2nd-order follower (`pos_tau`, seconds; no
        overshoot, C1-continuous). Measured on the shipped render this replaces: roll 1.49
        deg/frame mean / 4.70 peak, centre 16.84 px/frame mean / 54.07 peak, N=100 @10 fps.

    🔴 PRIVACY (see the block above `_draw_head3d` for the full rules). Geometry is a shared
    constant - nothing is derived from the subject. SIZE stays io = 0.26 x shoulder width off the
    ANONYMISED skeleton. The 12 channels may only modulate expression/pose, and are gated twice:
    a hard allow-list (only mouth_open, smile, and the SYMMETRIC eye/brow means may ever be
    data-driven - mouth_width, yaw, pitch, roll, gaze and all L/R differences are permanently
    denied as anthropometric or redundant), then a measured trust gate from `face_signal_filter`.
    MEASURED at the shipped DP_EPSILON_TOTAL=3.0 (a10/a15/a30, N=36 channel x release cells):
    trusted = NONE, so the drive is 100 % `_idle_expression` - data-independent, zero DP budget,
    zero identity. `debug_json` records which channels were actually used for every render.

    `on_black` (DEFAULT, user 2026-07-25): render on a BLACK canvas - this is the CONDITIONING
    signal for WanVideoAnimateEmbeds.face_images, not a debug view.

    `crop512_path` (OPT-IN, default None): additionally write a 512x512 per-frame CENTRED FACE
    CROP. Measured contract gap (kijai's WanAnimate preprocessor: get_face_bboxes(scale=1.3) ->
    cv2.resize to 512, landmark hull = 87.7 % of the canvas): the full-frame sidecar delivers the
    face at 22.7 % of the 512 encoder input, 0.259x linear / 0.067x area of the contract, and
    off-centre in 82/100 frames. The crop stream hits the contract. It is NOT wired into the cloud
    graph here - switching the uploaded stream is a user decision.
    """
    import cv2
    H, W = masked[0].shape[:2]
    vw = _writer(out_path, (W, H), fps)
    n = min(len(masked), len(mask_frames))
    min_area = 0.004 * H * W
    if isinstance(face_scalars, str):
        face_scalars = json.load(open(face_scalars, encoding="utf-8"))

    # ---- pass 1: per-slot placement traces from the emitted (anonymised) pose ----
    IO_OVER_SHOULDER = 0.26
    traces = {}                    # slot -> {"e": [], "mid": [], "io": [], "lean": []}
    if pose_frames is not None:
        for e in range(min(n, len(pose_frames))):
            for pi, person in enumerate(pose_frames[e]):
                kp = np.array(person.get("kp", []), dtype=float)
                sc = person.get("score", [1] * len(kp))
                if len(kp) < 7:
                    continue
                nose, lsh, rsh = kp[0], kp[5], kp[6]
                if not (sc[0] > 0 and (abs(nose[0]) > 1e-6 or abs(nose[1]) > 1e-6)):
                    continue
                sh_w = float(abs(lsh[0] - rsh[0]))
                if sh_w < 8:
                    continue
                io = IO_OVER_SHOULDER * sh_w
                # BODY LEAN only (attenuated later by `lean_gain`): the head's own roll comes
                # from the drive. Taking roll straight off the eye vector was the measured
                # source of the sharp tilt (1.79 deg/frame raw).
                ev = kp[1] - kp[2]
                if sc[1] > 0 and sc[2] > 0 and np.linalg.norm(ev) > 1e-6:
                    lean = float(np.arctan2(ev[1], ev[0]))
                else:
                    shv = lsh - rsh
                    lean = float(np.arctan2(shv[1], shv[0])) if np.linalg.norm(shv) > 1 else 0.0
                mid = nose - np.array([0.0, 1.0]) * (0.45 * io)   # eye line just above the nose
                t = traces.setdefault(person.get("slot", pi),
                                      {"e": [], "mid": [], "io": [], "lean": []})
                t["e"].append(e); t["mid"].append(mid); t["io"].append(io); t["lean"].append(lean)

    # ---- pass 2: smooth + critically damp every placement trace, then build the drive ----
    placed = {}                    # e -> [(mid, io, (yaw,pitch,roll), expr12)]
    dbg = {"style": style, "fps": float(fps), "frames": int(n), "slots": {},
           "privacy": {"allow_list_ac_only": list(_ALLOW_AC),
                       "denied_always": [1, 7, 8, 9, 10, 11],
                       "lr_differences": "denied (eyes/brows forcibly symmetrised)"}}
    for slot, t in traces.items():
        k = len(t["e"])
        mx = _damp1d(_smooth1d([m[0] for m in t["mid"]], smooth_sigma), fps, pos_tau)
        my = _damp1d(_smooth1d([m[1] for m in t["mid"]], smooth_sigma), fps, pos_tau)
        io_s = _damp1d(_smooth1d(t["io"], smooth_sigma), fps, pos_tau)
        lean = _damp1d(_smooth1d(np.unwrap(t["lean"]), max(smooth_sigma, 0.8 * fps)),
                       fps, 0.45, max_rate=0.5)
        drive, used, gains = _build_drive(k, fps, face_scalars, slot, idle_seed)
        yaw = _damp1d(drive[:, 7], fps, 0.25, max_rate=1.2)
        pitch = _damp1d(drive[:, 8], fps, 0.25, max_rate=1.0)
        roll = _damp1d(drive[:, 9] + float(lean_gain) * lean, fps, 0.30, max_rate=0.8)
        dbg["slots"][str(slot)] = {
            "n": k, "data_driven_channels": used, "shrinkage_gain_per_allowed_channel": gains,
            "drive_source": "face_scalars (gated)" if used else "procedural (data-independent)",
            "lean_gain": float(lean_gain),
            "roll_deg_per_frame_mean": float(np.mean(np.abs(np.diff(np.degrees(roll))))) if k > 1 else 0.0,
            "centre_px_per_frame_mean": float(np.mean(np.hypot(np.diff(mx), np.diff(my)))) if k > 1 else 0.0,
            "mouth_open_mean": float(np.mean(drive[:, 0])), "mouth_open_sd": float(np.std(drive[:, 0])),
            "eye_open_mean": float(np.mean(drive[:, 3])), "eye_open_min": float(np.min(drive[:, 3])),
        }
        dbg.setdefault("_traces", {})[str(slot)] = {
            "e": [int(v) for v in t["e"]],
            "mouth_open": [float(v) for v in drive[:, 0]],
            "eye_open": [float(v) for v in drive[:, 3]],
            "roll": [float(v) for v in roll], "yaw": [float(v) for v in yaw],
            "cx": [float(v) for v in mx], "cy": [float(v) for v in my],
        }
        for i, e in enumerate(t["e"]):
            placed.setdefault(e, []).append((np.array([mx[i], my[i]]), float(io_s[i]),
                                             (yaw[i], pitch[i], roll[i]), drive[i]))

    # crop-512 companion stream (opt-in): the head centred and filling the canvas to the
    # measured WanAnimate contract (landmark hull = 87.7 % of a 512 canvas).
    cw = None
    if crop512_path:
        cw = _writer(crop512_path, (512, 512), fps)
    IO_CROP = 133.0
    MID_CROP = np.array([256.0, 256.0 - 0.315 * IO_CROP])

    legacy = None if style == "head3d" else (_draw_face_cartoon if style == "cartoon" else _draw_face_at)
    for e in range(n):
        img = np.zeros_like(masked[e]) if on_black else masked[e].copy()
        m = mask_frames[e]
        th = ((m[..., 0] if m.ndim == 3 else m) > 127).astype(np.uint8)
        if e in placed:
            for mid, io, ypr, expr in placed[e]:
                if legacy is None:
                    _draw_head3d(img, mid, io, ypr, expr, bold=bold, fs=feature_scale)
                else:
                    ex = np.array([math.cos(ypr[2]), math.sin(ypr[2])])
                    legacy(img, mid, ex, io, bold=bold)
            if cw is not None:
                cimg = np.zeros((512, 512, 3), np.uint8)
                mid, io, ypr, expr = placed[e][0]
                _draw_head3d(cimg, MID_CROP, IO_CROP, ypr, expr, bold=bold, fs=feature_scale)
                cw.write(cimg)
            vw.write(img)
            continue
        # no pose for this frame: fall back to the mask silhouette, upright, neutral drive
        expr = np.array(_FALLBACK_NEUTRAL, float)
        num, _lab, stats, _cent = cv2.connectedComponentsWithStats(th, 8)
        for c in range(1, num):
            x, y, w, h, area = stats[c]
            if area < min_area or h < 40:
                continue
            io = max(6.0, 0.14 * w)                           # inter-ocular px from body width
            mid = np.array([x + w * 0.5, y + h * 0.10 + io * 0.9])   # eye line just below head top
            if legacy is None:
                _draw_head3d(img, mid, io, (0.0, 0.0, 0.0), expr, bold=bold, fs=feature_scale)
            else:
                legacy(img, mid, np.array([1.0, 0.0]), io, bold=bold)
        if cw is not None:
            cw.write(np.zeros((512, 512, 3), np.uint8))
        if not on_black:
            _banner(img, "TIER-1 - synthetic canonical face",
                    "generic template on the silhouette head - NOT the real face")
        vw.write(img)
    vw.release()
    if cw is not None:
        cw.release()
    if debug_json:
        json.dump(dbg, open(debug_json, "w", encoding="utf-8"), indent=1)
    return n


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--clip", required=True)
    ap.add_argument("--out", required=True, help="dir to write the 3 sidecar mp4s into")
    ap.add_argument("--work", default=None, help="existing bridge work dir (masked_video.mkv + pose.json); "
                                                 "if absent, Tier-1 is run fresh into a temp work dir")
    ap.add_argument("--max-frames", type=int, default=100)
    ap.add_argument("--no-synthetic-face", action="store_true")
    a = ap.parse_args()
    os.makedirs(a.out, exist_ok=True)

    work = a.work
    if not work or not os.path.exists(os.path.join(work, "pose.json")):
        sys.path.insert(0, os.path.join(REPO, "tier2_phone", "companion_scripts"))
        from tier1_to_tier2 import run_tier1
        work = os.path.join(a.out, "_t1_work")
        print(f"[viz] running Tier-1 (deployed guided config) on {os.path.basename(a.clip)} ...")
        ok, err = run_tier1(a.clip, work, a.max_frames)
        if not ok:
            raise SystemExit(f"Tier-1 failed:\n{err}")

    frames, anon, _ = _load_pose(work)
    masked, fps = read_video(os.path.join(work, "masked_video.mkv"))
    if not masked or not frames:
        raise SystemExit("no masked frames or pose frames to draw")
    print(f"[viz] anon={anon} | masked frames={len(masked)} pose frames={len(frames)} @ {fps:g} fps")

    ps = os.path.join(a.out, "tier1_pose_sticks.mp4")
    cn = os.path.join(a.out, "tier1_canonical.mp4")
    n1 = make_pose_sticks(masked, frames, ps, fps); print(f"[viz] wrote {ps} ({n1} f)")
    n2 = make_canonical(frames, cn, fps, (masked[0].shape[1], masked[0].shape[0])); print(f"[viz] wrote {cn} ({n2} f)")
    if not a.no_synthetic_face:
        mask_frames, _ = read_video(os.path.join(work, "mask.mkv"))
        sf = os.path.join(a.out, "tier1_synthetic_face.mp4")
        n3 = make_synthetic_face(masked, mask_frames, sf, fps, pose_frames=frames)
        print(f"[viz] wrote {sf} ({n3} f)  [synthetic template - no real face geometry]")
    print("[viz] done.")


if __name__ == "__main__":
    main()
