#!/usr/bin/env python3
"""
npy_to_mirage_emit.py -- an ADAPTER between two artifact formats. It measures nothing and
decides nothing; it re-expresses what Tier-1 already emitted.
=============================================================================================

WHAT THIS BRIDGES
-----------------
The MIRAGE Tier-1 host (`tier1/`, the groupmate clone carrying
`_e2e/tier1_reid_20260814/tier1_reid_integration.patch`) exports its per-person signals as NUMPY
ARRAYS, one file per person SLOT:

    keypoints_p{s}.npy   (T, 17, 3) float32   [x, y, confidence], NATIVE PIXEL coords, COCO-17 order
                                              an ABSENT slot-frame is exactly np.zeros((17, 3))
                                              🔴 the confidence column is BINARIZED to {0, 1} at 0.5
    face_params_p{s}.npy (T, 12)    float32   the identity-free expression scalars
    identities_p{s}.npy  (T,)       int32     the persistent track id (-1 = slot never used)

Every downstream consumer in THIS repo -- `tier2_cloud/scripts/tier1_viz.py`,
`100826_runs/_build_two_person.py`, `tier2_cloud/scripts/face_signal_filter.py` -- was written against the
MIRAGE EDGE EMIT FORMAT instead, i.e. the two JSON files that
`tier1/src/edge_runner_pi5/mirage_tier1.py` writes at its own emit seam
(`mirage_tier1.py:2352` for pose.json, `:2517` for face_scalars.json):

    pose.json          {"anon": {...}, "person_count": N, "slot_policy": str,
                        "emitted_slots": M,
                        "frames": [ [ {"slot": int, "track": int,
                                       "kp":    [[x, y]] * 133,
                                       "score": [s]      * 133}, ... ], ... ]}
    face_scalars.json  [frames][persons][12]   (`face_signal_filter._to_array`, :356)

This file converts the first into the second. Nothing else.

THE 133-WIDE KEYPOINT VECTOR (wholebody), and what each block gets
-----------------------------------------------------------------
    0..16    BODY, COCO-17          <- copied verbatim from keypoints_p{s}.npy
    17..22   FEET                   <- ZERO. This host runs a COCO-17 pose model and never
                                       produced feet. Emitting anything here would be an
                                       invented joint.
    23..90   FACE (68 landmarks)    <- ZERO, MANDATORY. `mirage_tier1.py:1925` zeroes this block
                                       because face-landmark geometry is a re-identifiable soft
                                       biometric that must not leave the device. The face is
                                       carried identity-free by face_scalars.json instead.
    91..132  HANDS (2 x 21)         <- ZERO. Same reason as feet: not produced by this host.

A zero keypoint with score 0 is how "not observed" is spelled in this format -- every consumer
gates on `score[j] > 0 and (|x| > 0 or |y| > 0)` (tier1_viz.py:268, :86, :227). So zeroing the
blocks this host never produced makes them INVISIBLE, it does not fabricate a joint at the origin.

RULES THIS ADAPTER OBEYS
------------------------
* NEVER INVENT A JOINT. An absent slot-frame -- `np.zeros((17, 3))` -- produces NO person entry in
  that frame. It is not interpolated, not held over from the previous frame, not filled with a
  template. Presence is read off the array, and the per-slot present-frame count is asserted
  against the array itself and cross-checked against Tier-1's own `manifest.json`
  (`anon.runtime.per_slot_present_frames`).
* NEVER HARDCODE A CONSTANT FITTED TO ONE CLIP. T, the slot set, the frame size, fps, the track
  ids and every config value are read from the run's own artifacts. The only literals here are the
  FORMAT constants (133, the 23..90 face block, 12 channels), which belong to the contract.
* COORDINATES PASS THROUGH UNCHANGED, in native pixels, float. Rounding matches the real emitter
  (kp 2 dp, score 3 dp, `mirage_tier1.py:1934`) so a diff against a real pose.json is meaningful.
* A LOW-CONFIDENCE JOINT KEEPS ITS COORDINATE and carries score 0, exactly as the real emitter
  does. The confidence column arrives already binarized to {0, 1}; that is the HOST's choice
  (`TIER1_CONFIG.json: export.binarize_thresh = 0.5`), not this adapter's, and it is recorded in
  the emitted `anon` block as `conf_binarized`.

face_scalars.json -- SHAPE, AND WHY IT IS SLOT-INDEXED
------------------------------------------------------
The consumer is `tier1_viz._build_drive` (:1044), which does `p = min(int(slot), P - 1)` and then
`vals[:, p, c]` -- it indexes the PERSON axis BY SLOT ID. The real emitter builds each frame's row
positionally from the frame's detections, so the two agree only while every slot is present in
every frame. This adapter emits the SLOT-INDEXED form (person axis index == slot id) because that
is what the consumer actually reads, and writes JSON `null` for a slot that is absent on a frame.
`face_signal_filter._to_array` (:363) turns that into NaN and `filter_face_scalars` (:404) drops it
via `np.isfinite`, so an absent person contributes nothing rather than a fake neutral row.
When every slot is present on every frame the two forms are identical, and the adapter asserts
that equivalence and reports it (`face_scalars.positional_equals_slot_indexed`).

⚠️ These scalars are NOT differentially private on this host: `TIER1_CONFIG.json` records no DP
stage, and all 1200 emitted cells sit inside `face_signal_filter.CH_BOUNDS`. `filter_face_scalars`
prices them AS IF they carried the shipped Laplace noise, so its trust gate is pessimistic and the
face rig falls back to its data-independent procedural idle. That is the safe direction (no real
expression reaches the cloud), but it means the emitted face motion is NOT this person's.

🔴 NO PRIVACY NUMBER measured on the MIRAGE edge host describes this host. This adapter changes no
pixel and adds no defence; it only re-expresses Tier-1's output.

USAGE
-----
    python npy_to_mirage_emit.py --src <tier1 out dir> --out <dir> [--report report.json]

    # or as a library
    from npy_to_mirage_emit import convert
    report = convert(src_dir, out_dir)
"""
import argparse
import io
import json
import os
import sys

import numpy as np

# ---- FORMAT CONSTANTS (the wholebody-133 contract, not tuning) ------------------------------
N_WHOLEBODY = 133
BODY_SLICE = slice(0, 17)          # COCO-17, the only block this host produces
FEET_SLICE = slice(17, 23)
FACE_SLICE = slice(23, 91)         # MUST stay zero -- privacy
HAND_SLICE = slice(91, 133)
N_FACE_CHANNELS = 12
SLOT_POLICY = "x-sorted+hysteresis (vendored person_slots; slot ids assigned by the Tier-1 host)"


def _find_slots(src):
    """Slot ids present as keypoints_p{s}.npy files, ascending. Derived, never assumed."""
    out = []
    for name in os.listdir(src):
        if name.startswith("keypoints_p") and name.endswith(".npy"):
            try:
                out.append(int(name[len("keypoints_p"):-len(".npy")]))
            except ValueError:
                pass
    return sorted(out)


def _anon_block(manifest, tier1_config):
    """The `anon` dict pose.json carries: THIS run's effective config, read from THIS run's
    artifacts. Shape mirrors what `_build_two_person.py` copies into its MANIFEST (`pose["anon"]`).
    """
    a = dict(manifest.get("anon", {}))
    rt = dict(a.get("runtime", {}))
    call = dict(a.get("call_site_kwargs", {}))
    cfg = (tier1_config or {}).get("config", {})
    tk = list(rt.get("template_kinds", []))
    out = {
        "level": a.get("level"),
        "canon": call.get("do_canon"),
        "conf_binarized": bool(cfg.get("extra", {}).get("score_binarize", False)),
        "conf_binarize_thresh": cfg.get("export", {}).get("binarize_thresh"),
        "level_knobs": dict(a.get("anonymize_v2_kwargs", {})),
        "gait_preset": a.get("preset"),
        "gait_preset_is_shipped_default": a.get("preset_is_shipped_default"),
        "pose_scale_from": (a.get("wrapper_knobs") or {}).get("scale_from"),
        "head_anchor": call.get("head_anchor"),
        "template_kinds": sorted(set(tk)),
        "template_kind_counts": {k: tk.count(k) for k in sorted(set(tk))},
        "seed_policy": rt.get("seed_source"),
        "per_slot_present_frames": rt.get("per_slot_present_frames"),
        # the mask arm is part of "what anonymisation this artifact carries"
        "mask": {k: v for k, v in (manifest.get("mask") or {}).items() if k != "runtime"},
        # 🔴 provenance of the host, so nobody reads a MIRAGE-host privacy number onto this
        "host": (cfg.get("host") or "mirage tier1 (groupmate clone) -- NOT the MIRAGE edge host"),
        "privacy_note": ("No privacy figure measured on the MIRAGE edge host describes this "
                         "pipeline. See MASTER_EVAL_LEDGER.md sections G.1-G.7."),
    }
    return out


def free_end_prune(frames, fw, fh, include_hips=False):
    """Zero INTERMEDIATE limb joints left dangling inside the frame.

    PORTED VERBATIM in rule and order from `mirage_tier1.py:2300-2337` (the 2026-07-26
    dangling-terminus fix, `config.POSE_FREE_END_PRUNE`, default ON there and signed off
    after a before/after render). This host never had it, because the prune lives in
    `mirage_tier1.py` and only `pose_anon_edge.py` was vendored - which is exactly the
    defect it exists to prevent:

      MEASURED on p20-p21_c01_2face (2026-08-15): ankles are confident in 0 % of frames for
      BOTH subjects and knees in 0 % for slot 1 / 40-94 % for slot 0, because the framing
      cuts the legs off. Tier-1 still emitted hip->knee bones ENDING DEAD IN OPEN SPACE, and
      the pose-conditioned cloud model completed them into a crouched half-body figure.

    THE RULE - a joint is dropped only when ALL THREE hold:
      (1) it is an INTERMEDIATE joint of an open chain. Wrists (9/10) and ankles (15/16) are
          anatomical termini and are NEVER dropped; head and shoulders are never touched.
          Hips (11/12) are opt-in: a hip is a corner of the torso quad, not a free end.
      (2) its distal child is NOT emitted, so nothing continues the chain past it.
      (3) it lies STRICTLY INSIDE the frame. A joint outside the canvas is KEPT so the bone
          exits the picture, which is the honest "the body continues out of shot" cue.
    Applied distal-first so one pass settles a whole chain.

    GENERALISES: pure graph + frame-bounds logic. No threshold, no tolerance, no constant
    fitted to any clip. A NO-OP on full-body footage where the distal joints are observed.
    PRIVACY: strictly information-REMOVING (joints are only ever zeroed to the pipeline's
    own "(0,0) = not detected" marker), so it cannot weaken §2 or the anti-reID posture.
    """
    stats = {"on": True, "include_hips": bool(include_hips), "frames": len(frames)}
    if not fw or not fh:
        stats.update({"on": False, "reason": "no frame size; cannot test the inside-frame rule"})
        return stats
    fw, fh = float(fw), float(fh)
    chain = [(13, 15), (14, 16)]                       # knee -> ankle, distal first
    if include_hips:
        chain += [(11, 13), (12, 14)]
    chain += [(7, 9), (8, 10)]                         # elbow -> wrist
    # 🔴 "EMITTED" MUST BE TESTED THE WAY THIS FORMAT SPELLS IT, NOT THE WAY MIRAGE DOES.
    # mirage_tier1 zeroes the COORDINATE of an unobserved joint, so upstream the rule can ask
    # `child == (0,0)`. This adapter deliberately KEEPS the coordinate and zeroes only the
    # score (see the module docstring), so that test is blind here: it sees a coordinate and
    # concludes the chain continues. MEASURED on the first run of this port - it zeroed 70
    # elbows and 2 knees while leaving every dangling hip->knee bone standing, which is the
    # exact defect it was ported to fix. A joint counts as emitted iff the RENDERER would draw
    # it, and tier1_viz gates on `score > 0 and off-origin`.
    def emitted(kp, sc, j):
        x, y = kp[j][0], kp[j][1]
        s = sc[j] if j < len(sc) else 1.0
        return (s > 0) and not (x == 0 and y == 0)

    drop, tot = {}, 0
    for fr in frames:
        for p in fr:
            kp, sc = p["kp"], p.get("score", [])
            for j, child in chain:
                if j >= len(kp) or child >= len(kp):
                    continue
                if not emitted(kp, sc, j):
                    continue                            # already not drawn
                x, y = kp[j][0], kp[j][1]
                if not (0.0 <= x < fw and 0.0 <= y < fh):
                    continue                            # outside the canvas: the bone exits - keep
                if emitted(kp, sc, child):
                    continue                            # the chain continues past it - keep
                kp[j] = [0.0, 0.0]
                if j < len(sc):
                    sc[j] = 0.0
                drop[j] = drop.get(j, 0) + 1
                tot += 1
    stats.update({"joints_zeroed": tot,
                  "per_joint": {str(k): v for k, v in sorted(drop.items())}})
    return stats


def convert(src, out, wholebody=N_WHOLEBODY):
    """npy arrays -> pose.json + face_scalars.json. Returns a validation report dict."""
    os.makedirs(out, exist_ok=True)
    manifest = json.load(io.open(os.path.join(src, "manifest.json"), encoding="utf-8"))
    tcfg_p = os.path.join(src, "TIER1_CONFIG.json")
    tcfg = json.load(io.open(tcfg_p, encoding="utf-8")) if os.path.exists(tcfg_p) else {}

    slots = _find_slots(src)
    if not slots:
        raise SystemExit("no keypoints_p*.npy under %s" % src)

    kps, faces, ids = {}, {}, {}
    for s in slots:
        kps[s] = np.load(os.path.join(src, "keypoints_p%d.npy" % s))
        fp = os.path.join(src, "face_params_p%d.npy" % s)
        faces[s] = np.load(fp) if os.path.exists(fp) else None
        ip = os.path.join(src, "identities_p%d.npy" % s)
        ids[s] = np.load(ip) if os.path.exists(ip) else None
    T = int(max(k.shape[0] for k in kps.values()))

    # ---- presence, read off the array. An ABSENT slot-frame is exactly np.zeros((17,3)). ----
    present = {s: (np.abs(kps[s]).reshape(kps[s].shape[0], -1).sum(1) > 0) for s in slots}
    present_counts = {s: int(present[s].sum()) for s in slots}
    live_slots = [s for s in slots if present_counts[s] > 0]

    # ---- pose.json --------------------------------------------------------------------------
    frames = []
    for t in range(T):
        row = []
        for s in live_slots:                       # ascending slot order, like the real emitter
            if t >= kps[s].shape[0] or not present[s][t]:
                continue                            # ABSENT -> no entry. Never interpolated.
            k = np.zeros((wholebody, 2), np.float64)
            sc = np.zeros(wholebody, np.float64)
            body = kps[s][t]
            k[BODY_SLICE] = body[:, :2]             # coordinates pass through unchanged
            sc[BODY_SLICE] = body[:, 2]
            # FEET / FACE / HANDS stay exactly zero -- see the module docstring.
            ent = {"slot": int(s),
                   "kp": np.round(k, 2).tolist(),
                   "score": np.round(sc, 3).tolist()}
            if ids[s] is not None and t < len(ids[s]):
                ent["track"] = int(ids[s][t])
            row.append(ent)
        frames.append(row)

    free_end = free_end_prune(frames, manifest.get("width"), manifest.get("height"))

    emitted_slots = max((len(fr) for fr in frames), default=0)
    pose = {
        "adapter": {
            "name": "npy_to_mirage_emit.py",
            "role": ("ADAPTER between two artifact formats: MIRAGE Tier-1 .npy arrays -> the "
                     "MIRAGE edge emit format (pose.json / face_scalars.json). Adds no signal."),
            "source_dir": os.path.abspath(src),
            "clip_id": manifest.get("clip_id"),
            "wholebody_n": wholebody,
            "free_end_prune": free_end,
            "blocks": {"body_coco17": "0..16 from keypoints_p*.npy",
                       "feet": "17..22 ZERO (not produced by this host)",
                       "face": "23..90 ZERO (privacy, mandatory)",
                       "hands": "91..132 ZERO (not produced by this host)"},
        },
        "anon": _anon_block(manifest, tcfg),
        "person_count": len(live_slots),
        "slot_policy": SLOT_POLICY,
        "emitted_slots": emitted_slots,
        "fps": manifest.get("fps"),
        "size": [manifest.get("width"), manifest.get("height")],
        "frames": frames,
    }
    json.dump(pose, io.open(os.path.join(out, "pose.json"), "w", encoding="utf-8"))

    # ---- face_scalars.json: [T][P][12], person axis indexed BY SLOT (see the docstring) ------
    P = (max(live_slots) + 1) if live_slots else 0
    scal, positional_equal = [], True
    for t in range(T):
        row = [None] * P
        pos = []
        for s in live_slots:
            if t >= kps[s].shape[0] or not present[s][t]:
                continue
            v = (faces[s][t].tolist() if (faces[s] is not None and t < faces[s].shape[0])
                 else [None] * N_FACE_CHANNELS)
            v = [None if x is None else float(x) for x in v[:N_FACE_CHANNELS]]
            row[s] = v
            pos.append(v)
        # The two forms coincide exactly when the present slots are 0..len(pos)-1, i.e. no slot is
        # missing before a present one. `pos` is built in ascending slot order, so comparing it
        # against the leading window of the slot-indexed row is the whole test.
        if pos != row[:len(pos)]:
            positional_equal = False
        scal.append(row)
    json.dump(scal, io.open(os.path.join(out, "face_scalars.json"), "w", encoding="utf-8"))

    # ---- VALIDATION (asserted, not assumed) --------------------------------------------------
    rep = {"src": os.path.abspath(src), "out": os.path.abspath(out), "T": T,
           "slots_found": slots, "live_slots": live_slots,
           "person_count": len(live_slots), "emitted_slots": emitted_slots,
           "checks": {}, "per_slot": {}}
    ck = rep["checks"]

    ck["every_person_has_133_kp"] = all(len(p["kp"]) == wholebody and len(p["score"]) == wholebody
                                        for fr in frames for p in fr)
    ck["face_block_23_90_all_zero"] = all(
        all(x == 0.0 for xy in p["kp"][FACE_SLICE] for x in xy)
        and all(x == 0.0 for x in p["score"][FACE_SLICE]) for fr in frames for p in fr)
    ck["feet_block_17_22_all_zero"] = all(
        all(x == 0.0 for xy in p["kp"][FEET_SLICE] for x in xy) for fr in frames for p in fr)
    ck["hand_block_91_132_all_zero"] = all(
        all(x == 0.0 for xy in p["kp"][HAND_SLICE] for x in xy) for fr in frames for p in fr)

    ok_counts = True
    for s in slots:
        n_json = sum(1 for fr in frames for p in fr if p["slot"] == s)
        n_npy = present_counts[s]
        ok_counts &= (n_json == n_npy)
        body_kp = kps[s][present[s]] if n_npy else np.zeros((0, 17, 3))
        rep["per_slot"][str(s)] = {
            "present_frames_npy": n_npy,
            "person_entries_pose_json": n_json,
            "match": n_json == n_npy,
            "track_ids": sorted({int(v) for v in ids[s]}) if ids[s] is not None else None,
            "conf_values_in_npy": sorted(float(v) for v in np.unique(kps[s][..., 2])),
            "confident_joints": int((body_kp[..., 2] > 0).sum()) if n_npy else 0,
            "joints_scored_but_at_origin": int(((body_kp[..., 2] > 0) &
                                                (np.abs(body_kp[..., :2]).sum(-1) == 0)).sum())
            if n_npy else 0,
            "joints_zero_score_off_origin": int(((body_kp[..., 2] == 0) &
                                                 (np.abs(body_kp[..., :2]).sum(-1) > 0)).sum())
            if n_npy else 0,
        }
    ck["per_slot_counts_match_npy"] = bool(ok_counts)

    # absent frames really are absent
    ck["absent_slot_frames_have_no_entry"] = all(
        not any(p["slot"] == s for p in frames[t])
        for s in slots for t in range(T) if t < kps[s].shape[0] and not present[s][t])

    # cross-check against Tier-1's OWN accounting
    t1_counts = ((manifest.get("anon") or {}).get("runtime") or {}).get("per_slot_present_frames")
    rep["tier1_manifest_per_slot_present_frames"] = t1_counts
    ck["agrees_with_tier1_manifest"] = (
        None if t1_counts is None
        else [present_counts.get(s, 0) for s in range(len(t1_counts))] == list(t1_counts))

    ck["face_scalars_shape"] = [len(scal), P, N_FACE_CHANNELS]
    ck["face_scalars_rows_match_pose_entries"] = all(
        len([r for r in scal[t] if r is not None]) == len(frames[t]) for t in range(T))
    ck["face_scalars_positional_equals_slot_indexed"] = bool(positional_equal)

    rep["all_checks_pass"] = all(v is True for v in ck.values()
                                 if isinstance(v, bool)) and ck["per_slot_counts_match_npy"]
    return rep


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[1])
    ap.add_argument("--src", required=True, help="Tier-1 output dir holding keypoints_p*.npy")
    ap.add_argument("--out", required=True, help="dir to write pose.json + face_scalars.json")
    ap.add_argument("--report", default="", help="optional path for the validation report json")
    a = ap.parse_args()
    rep = convert(a.src, a.out)
    print(json.dumps(rep, indent=1))
    if a.report:
        json.dump(rep, io.open(a.report, "w", encoding="utf-8"), indent=1)
    return 0 if rep["all_checks_pass"] else 1


if __name__ == "__main__":
    sys.exit(main())
