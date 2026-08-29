#!/usr/bin/env python3
"""
audit_bundle.py -- INDEPENDENT adversarial audit of a two-person cloud bundle.
==============================================================================
This is NOT `verify_bundle.py`. That script was written by the agent that BUILT the bundle and shares
its assumptions; this one re-derives every quantity from the raw files, adds the checks that one does
not make, and carries a POSITIVE CONTROL for every measurement that could silently return "clean"
because it is measuring nothing (the §A.6o lesson: an adversary with no positive control may be
measuring nothing at all).

WHAT IT ADDS OVER verify_bundle.py
  * DECODE-TO-END frame counts (cv2's CAP_PROP_FRAME_COUNT is a container hint, not a decode).
  * UNION SUPERSET: mask_p1 | mask_p2 vs the union mask, BOTH directions, in pixels.
  * MANIFEST vs the Tier-1 manifest, field by field -- a bundle that misdescribes its own run is a
    hard fail (the whole provenance discipline of this project exists because a rendered artifact
    once disagreed with the committed source).
  * RAW-FACE SWEEP: YuNet over every frame of every video in the bundle and both reference PNGs,
    with a positive control on the SOURCE clip so a "0 faces" result is known to be a measurement.
  * PLATE INTEGRITY: inside the mask the plate must be flat fill; outside it must equal the lightmap.
  * LIGHTMAP LEAK: does the uploaded lightmap still carry the real scene where the people are?

🔴 EVERYTHING HERE IS ENGINEERING GEOMETRY. Not one number is a privacy figure, and no number
measured on the MIRAGE edge host describes this host.

    python audit_bundle.py --bundle <dir> --tier1 <out_final> --tier2 <tier2_out> \
                           --source <clip.mp4> [--json report.json]
"""
import argparse
import io
import json
import os
import sys

import cv2
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.dirname(os.path.dirname(HERE))
YUNET = os.environ.get("MIRAGE_YUNET_ONNX") or os.path.join(
    REPO, "models", "face_detection_yunet_2023mar.onnx")
# YuNet is an INDEPENDENT cross-check detector, never in the runtime path. It is not
# redistributed: fetch it from the OpenCV Zoo and either drop it at the path above or
# point MIRAGE_YUNET_ONNX at it (models/README.md). The previous hardcoded path pointed
# into a development tree that does not exist in this release.

# the two-person contract, as read off the live reference bundle on the network volume
CONTRACT = ["MANIFEST.json", "masked_video_00002.mp4", "mask_00002.mp4", "light_map.mp4",
            "mask_p1_00002.mp4", "mask_p2_00002.mp4",
            "pose_sticks_p1_00002.mp4", "pose_sticks_p2_00002.mp4", "pose_sticks_both_00002.mp4",
            "facemesh_p1_00002.mp4", "facemesh_p2_00002.mp4",
            "reference_p1_640.png", "reference_p2_640.png"]
INK = 16
MTH = 127
FILL = 128


def decode(path, gray=True, limit=0):
    """Decode to the END. Returns (frames, meta) -- meta['decoded'] is the REAL frame count."""
    c = cv2.VideoCapture(path)
    meta = dict(container_frames=int(c.get(cv2.CAP_PROP_FRAME_COUNT)),
                w=int(c.get(3)), h=int(c.get(4)), fps=round(float(c.get(cv2.CAP_PROP_FPS)), 4))
    out, sizes = [], set()
    while True:
        ok, f = c.read()
        if not ok:
            break
        sizes.add(f.shape[:2])
        out.append(cv2.cvtColor(f, cv2.COLOR_BGR2GRAY) if gray else f)
        if limit and len(out) >= limit:
            break
    c.release()
    meta["decoded"] = len(out)
    meta["distinct_frame_shapes"] = sorted("%dx%d" % (s[1], s[0]) for s in sizes)
    return out, meta


def yunet_sweep(frames_bgr, tag, thresh=0.5):
    """Run YuNet over BGR frames. Returns per-frame detection counts + the best score seen."""
    if not os.path.exists(YUNET):
        return {"tag": tag, "error": "yunet model missing at %s" % YUNET}
    h, w = frames_bgr[0].shape[:2]
    det = cv2.FaceDetectorYN.create(YUNET, "", (w, h), thresh, 0.3, 5000)
    n_hits, best, hit_frames = 0, 0.0, []
    for i, f in enumerate(frames_bgr):
        det.setInputSize((f.shape[1], f.shape[0]))
        _, faces = det.detect(f)
        if faces is not None and len(faces):
            n_hits += len(faces)
            hit_frames.append(i)
            best = max(best, float(faces[:, -1].max()))
    return {"tag": tag, "frames": len(frames_bgr), "threshold": thresh,
            "detections": int(n_hits), "frames_with_a_face": len(hit_frames),
            "first_hit_frames": hit_frames[:10], "best_score": round(best, 4)}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--bundle", required=True)
    ap.add_argument("--tier1", required=True)
    ap.add_argument("--tier2", required=True)
    ap.add_argument("--source", default="")
    ap.add_argument("--json", default="")
    a = ap.parse_args()
    B = a.bundle
    R = {"bundle": os.path.abspath(B), "fail": [], "warn": []}

    def FAIL(m):
        R["fail"].append(m); print("FAIL  " + m)

    def WARN(m):
        R["warn"].append(m); print("warn  " + m)

    # ---------------------------------------------------------------- a. FILE SET ---------------
    have = sorted(os.listdir(B))
    R["files_present"] = have
    R["files_missing"] = [f for f in CONTRACT if f not in have]
    R["files_extra"] = [f for f in have if f not in CONTRACT]
    R["file_sizes"] = {f: os.path.getsize(os.path.join(B, f)) for f in have}
    if R["files_missing"]:
        FAIL("missing contract files: %s" % R["files_missing"])
    if R["files_extra"]:
        WARN("files beyond the 13-object contract: %s" % R["files_extra"])
    zero = [f for f, s in R["file_sizes"].items() if s == 0]
    if zero:
        FAIL("zero-byte files: %s" % zero)

    # ---------------------------------------------------------------- b. DECODE -----------------
    vids = [f for f in have if f.endswith(".mp4")]
    G, meta = {}, {}
    for v in sorted(vids):
        fr, m = decode(os.path.join(B, v))
        G[v] = fr
        meta[v] = m
        if m["decoded"] != m["container_frames"]:
            WARN("%s: container says %d frames, decoded %d"
                 % (v, m["container_frames"], m["decoded"]))
    R["video_meta"] = meta
    want = (50, 1264, 1264, 10.0)
    for v, m in meta.items():
        if (m["decoded"], m["w"], m["h"], m["fps"]) != want:
            FAIL("%s geometry %s != expected 50f 1264x1264 10fps"
                 % (v, (m["decoded"], m["w"], m["h"], m["fps"])))
        if len(m["distinct_frame_shapes"]) != 1:
            FAIL("%s decodes to more than one frame size: %s" % (v, m["distinct_frame_shapes"]))

    # ---------------------------------------------------------------- c/d. MASKS ----------------
    u, p1, p2 = G["mask_00002.mp4"], G["mask_p1_00002.mp4"], G["mask_p2_00002.mp4"]
    b1 = open(os.path.join(B, "mask_p1_00002.mp4"), "rb").read()
    b2 = open(os.path.join(B, "mask_p2_00002.mp4"), "rb").read()
    n = min(len(u), len(p1), len(p2))
    ov, ident, esc, hole, a1, a2, au = [], 0, [], [], [], [], []
    for t in range(n):
        B1, B2, BU = p1[t] > MTH, p2[t] > MTH, u[t] > MTH
        ov.append(int((B1 & B2).sum()))
        ident += bool(np.array_equal(B1, B2))
        esc.append(int(((B1 | B2) & ~BU).sum()))     # per-slot px OUTSIDE the union  -> superset break
        hole.append(int((BU & ~(B1 | B2)).sum()))    # union px claimed by NEITHER slot
        a1.append(int(B1.sum())); a2.append(int(B2.sum())); au.append(int(BU.sum()))
    R["per_slot_masks"] = {
        "file_bytes_identical": b1 == b2,
        "frames": n, "frames_binary_identical": ident,
        "overlap_px_total": int(sum(ov)), "overlap_px_max_frame": int(max(ov)),
        "overlap_px_mean_per_frame": float(np.mean(ov)),
        "area_mean": {"p1": float(np.mean(a1)), "p2": float(np.mean(a2)),
                      "union": float(np.mean(au))},
        "union_frame_coverage_pct": round(100.0 * float(np.mean(au)) / (1264 * 1264), 4),
    }
    R["union_superset"] = {
        "px_in_slot_masks_but_not_union_total": int(sum(esc)),
        "px_in_slot_masks_but_not_union_max_frame": int(max(esc)),
        "frames_violating": int(sum(1 for x in esc if x > 0)),
        "px_in_union_claimed_by_no_slot_total": int(sum(hole)),
        "px_in_union_claimed_by_no_slot_max_frame": int(max(hole)),
    }
    if b1 == b2:
        FAIL("mask_p1 and mask_p2 are BYTE-IDENTICAL -- this is the B.57e condition")
    if ident:
        FAIL("%d frames where the two per-slot masks are binary-identical" % ident)
    if sum(ov) > 0:
        WARN("per-slot masks overlap on %d px total" % sum(ov))

    # ---------------------------------------------------------------- e. CONTAINMENT ------------
    cont = {}
    for si, sl in enumerate(("p1", "p2")):
        st = G["pose_sticks_%s_00002.mp4" % sl]
        own = p1 if sl == "p1" else p2
        oth = p2 if sl == "p1" else p1
        f_own, f_uni, f_oth, ink, z = [], [], [], [], 0
        worst = (1.0, -1, 0)
        for t in range(min(len(st), n)):
            I = st[t] > INK
            tot = int(I.sum())
            ink.append(tot)
            if tot == 0:
                continue
            o = float((I & (own[t] > MTH)).sum()) / tot
            v = float((I & (u[t] > MTH)).sum()) / tot
            x = float((I & (oth[t] > MTH)).sum()) / tot
            f_own.append(o); f_uni.append(v); f_oth.append(x)
            z += (o == 0.0)
            if o < worst[0]:
                worst = (o, t, tot - int((I & (own[t] > MTH)).sum()))
        cont[sl] = {
            "frames": len(f_own), "ink_px_mean": float(np.mean(ink)),
            "in_own_mask_mean": round(float(np.mean(f_own)), 6),
            "in_own_mask_min": round(float(np.min(f_own)), 6),
            "in_union_mask_mean": round(float(np.mean(f_uni)), 6),
            "in_union_mask_min": round(float(np.min(f_uni)), 6),
            "in_OTHER_slot_mask_mean": round(float(np.mean(f_oth)), 6),   # POSITIVE CONTROL
            "zero_containment_frames": int(z),
            "worst_frame": {"containment": round(worst[0], 6), "frame": worst[1],
                            "escaped_px": worst[2]},
        }
    R["stick_containment"] = cont
    R["stick_containment_note"] = ("fraction of drawn stick ink inside the mask, full render "
                                   "resolution, on the EMITTED mp4s. ENGINEERING ONLY.")
    for sl, d in cont.items():
        if d["in_OTHER_slot_mask_mean"] > 0.02:
            FAIL("%s sticks land in the OTHER slot's mask (%.4f) -- the split is wrong"
                 % (sl, d["in_OTHER_slot_mask_mean"]))
        if d["zero_containment_frames"]:
            FAIL("%s has %d zero-containment frames" % (sl, d["zero_containment_frames"]))

    # ---------------------------------------------------------------- f. MANIFEST ---------------
    man = json.load(io.open(os.path.join(B, "MANIFEST.json"), encoding="utf-8"))
    t1 = json.load(io.open(os.path.join(a.tier1, "manifest.json"), encoding="utf-8"))
    cfg, mk = man.get("config", {}), (man.get("config", {}) or {}).get("mask", {})
    want_man = {
        "gait_preset==e2": cfg.get("gait_preset") == "e2",
        "level==L4": cfg.get("level") == "L4",
        "mask.shape_mode==bbox": mk.get("shape_mode") == "bbox",
        "mask.temporal_win==2": mk.get("temporal_win") == 2,
        "frames==50": man.get("frames") == 50,
        "size==1264x1264": man.get("size") == [1264, 1264],
        "fps==10": float(man.get("fps", 0)) == 10.0,
        "clip_id_matches_tier1": man.get("clip") == t1.get("clip_id"),
        "anon_block_equals_tier1": man.get("tier1_manifest_anon") == t1.get("anon"),
        "mask_block_equals_tier1": man.get("tier1_manifest_mask") == t1.get("mask"),
        "preset_not_r2": cfg.get("gait_preset") != "r2",
        "tier1_anon_preset==e2": (t1.get("anon") or {}).get("preset") == "e2",
        "tier1_mask_shape==bbox": (t1.get("mask") or {}).get("shape_mode") == "bbox",
        "tier1_mask_win==2": (t1.get("mask") or {}).get("temporal_win") == 2,
        "slots_match_tier1_present": man.get("slots") == [
            i for i, v in enumerate((t1.get("anon") or {}).get("runtime", {})
                                    .get("per_slot_present_frames", [])) if v],
        "host_is_not_mirage_edge": "NOT the MIRAGE edge host" in str(man.get("tier1_host")),
    }
    R["manifest_checks"] = want_man
    for k, v in want_man.items():
        if not v:
            FAIL("MANIFEST check failed: %s" % k)
    R["manifest_config_echo"] = {
        "gait_preset": cfg.get("gait_preset"), "level": cfg.get("level"),
        "shape_mode": mk.get("shape_mode"), "temporal_win": mk.get("temporal_win"),
        "pose_scale_from": cfg.get("pose_scale_from"), "head_anchor": cfg.get("head_anchor"),
        "template_kinds": cfg.get("template_kinds"),
        "reference_images_carried_over": (man.get("reference_images") or {})
        .get("provenance", {}) and (man["reference_images"]["provenance"] or {}).get("status"),
    }

    # ---------------------------------------------------------------- g. RAW FACE SWEEP --------
    face = {}
    if a.source and os.path.exists(a.source):
        src, _ = decode(a.source, gray=False)
        face["POSITIVE_CONTROL_source_clip"] = yunet_sweep(src, "source clip")
        R["source_meta"] = _
    for v in sorted(vids):
        fr, _m = decode(os.path.join(B, v), gray=False)
        face[v] = yunet_sweep(fr, v)
    for png in [f for f in have if f.endswith(".png")]:
        img = cv2.imread(os.path.join(B, png))
        face[png] = yunet_sweep([img], png)
    R["face_sweep_yunet"] = face
    pc = face.get("POSITIVE_CONTROL_source_clip", {})
    if pc and pc.get("detections", 0) == 0:
        FAIL("the face detector found NOTHING in the source clip -- the sweep is measuring nothing")
    for v in sorted(vids):
        if face[v]["detections"]:
            FAIL("%s: YuNet found %d face detections (best %.3f) -- possible raw face pixels"
                 % (v, face[v]["detections"], face[v]["best_score"]))

    # ---------------------------------------------------------------- h. PLATE INTEGRITY -------
    plate, _ = decode(os.path.join(B, "masked_video_00002.mp4"), gray=False)
    lmv, _ = decode(os.path.join(B, "light_map.mp4"), gray=False)
    dev_in, dev_out, mx_in = [], [], []
    for t in range(min(len(plate), len(lmv), n)):
        M = u[t] > MTH
        pg = plate[t].astype(np.int16)
        lg = lmv[t].astype(np.int16)
        dev_in.append(float(np.abs(pg[M] - FILL).mean()))
        mx_in.append(int(np.abs(pg[M] - FILL).max()))
        dev_out.append(float(np.abs(pg[~M] - lg[~M]).mean()))
    R["plate"] = {"mean_abs_dev_from_%d_inside_mask" % FILL: float(np.mean(dev_in)),
                  "max_abs_dev_inside_mask": int(max(mx_in)),
                  "mean_abs_diff_vs_lightmap_outside_mask": float(np.mean(dev_out)),
                  "note": "mp4v is lossy; a few counts of deviation is codec noise, not a fill error"}
    if np.mean(dev_in) > 12:
        FAIL("the plate is NOT flat-filled inside the mask (mean |px-%d| = %.2f)"
             % (FILL, np.mean(dev_in)))

    # ---------------------------------------------------------------- i. LIGHTMAP LEAK ---------
    if a.source and os.path.exists(a.source):
        srcf, _ = decode(a.source, gray=False)
        rtm, _ = decode(os.path.join(a.tier1, "output_rtm.mp4"), gray=False)
        din, dout, dr = [], [], []
        for t in range(min(len(srcf), len(lmv), len(rtm), n)):
            M = u[t] > MTH
            s = cv2.cvtColor(srcf[t], cv2.COLOR_BGR2GRAY).astype(np.int16)
            l = cv2.cvtColor(lmv[t], cv2.COLOR_BGR2GRAY).astype(np.int16)
            r = cv2.cvtColor(rtm[t], cv2.COLOR_BGR2GRAY).astype(np.int16)
            din.append(float(np.abs(s[M] - l[M]).mean()))
            dout.append(float(np.abs(s[~M] - l[~M]).mean()))
            dr.append(float(np.abs(r[M] - l[M]).mean()))
        R["lightmap_leak"] = {
            "mean_abs_diff_source_vs_lightmap_INSIDE_mask": float(np.mean(din)),
            "mean_abs_diff_source_vs_lightmap_OUTSIDE_mask": float(np.mean(dout)),
            "mean_abs_diff_tier1_greyfilled_vs_lightmap_INSIDE_mask": float(np.mean(dr)),
            "note": ("if the lightmap still carried the real people, the INSIDE-mask distance to "
                     "the SOURCE would be small and the distance to the GREY-FILLED Tier-1 video "
                     "would be large. Both are reported so neither can be read alone."),
        }

    R["pass"] = not R["fail"]
    print(json.dumps({k: v for k, v in R.items() if k != "face_sweep_yunet"}, indent=1)[:6000])
    print("\nFACE SWEEP:")
    print(json.dumps(R["face_sweep_yunet"], indent=1))
    print("\nFAIL=%d  WARN=%d  PASS=%s" % (len(R["fail"]), len(R["warn"]), R["pass"]))
    if a.json:
        json.dump(R, io.open(a.json, "w", encoding="utf-8"), indent=1)
    return 0 if R["pass"] else 1


if __name__ == "__main__":
    sys.exit(main())
