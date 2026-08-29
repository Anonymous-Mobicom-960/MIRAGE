#!/usr/bin/env python3
"""
face_coverage_eval.py - does the emitted mask actually cover every REAL face?
=============================================================================
This is the §2 guarantee measured against human-annotated ground truth instead of against the
pipeline's own detector. Every other face number we have (DECISION.md: 0 reveals / 900 frames,
98.5 % coverage) was scored against detections produced by the same YOLO/FaceGuard stack that
draws the mask - so a face BOTH models miss counts as a success. Annotated GT removes that
circularity.

    python face_coverage_eval.py --gt input/reid_dataset/_annotations \
                                 --videos <corpus dir> --out reports/

GT layout (110 videos x 150 frames, 10 subjects x 11 scenarios):
    _annotations/Annotated JSONs/<video>_frame<N>_output.json   [{"bbox_xyxy":[x1,y1,x2,y2]},...]
    _annotations/video_frames_mapping.csv                       video_name,frame_numbers,total

FRAME ALIGNMENT (the easy thing to get wrong): Tier-1 decimates source->EMIT_FPS, so emitted
frame e corresponds to SOURCE frame e*stride. GT frame numbers are SOURCE indices. We therefore
score only GT frames that the pipeline actually emits (f % stride == 0) - measuring what really
ships at the deployed config, rather than re-running at stride=1 and measuring a configuration
nobody deploys.

REVEAL CRITERION: a GT face box is "covered" when the fraction of its pixels inside the emitted
mask is >= --cover-thresh. Results are reported across several thresholds because the strict
100 % reading is dominated by single-pixel boundary effects, while the loose reading hides real
partial exposure. The headline number is FULL coverage (100 %) and any-exposure count.
"""
import argparse
import csv
import glob
import json
import os
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.dirname(HERE)
# The internal development tree still carries this directory under its pre-rename name, and that
# repository cannot simply be renamed: `tree/<x>` maps 1:1 onto the pod's `/workspace/<x>`, so
# moving it would create a second copy on the pod rather than rename the first. Accept either
# name, newest first, so the two repositories do not have to change in the same commit.
def _edge_root(repo, *tail):
    for _name in ("mirage_edge_deploy", "sitara_edge_deploy"):
        _p = os.path.join(repo, "tree", _name, "tier1_raspberry_pi5", *tail)
        if os.path.exists(_p):
            return _p
    return os.path.join(repo, "tree", "mirage_edge_deploy", "tier1_raspberry_pi5", *tail)


EDGE = _edge_root(REPO)


def load_gt(gt_dir):
    """-> {video_name: {frame_no: [ [x1,y1,x2,y2], ... ]}}"""
    jdir = os.path.join(gt_dir, "Annotated JSONs")
    gt = {}
    for p in glob.glob(os.path.join(jdir, "*_output.json")):
        base = os.path.basename(p)[:-len("_output.json")]
        if "_frame" not in base:
            continue
        vid, _, fno = base.rpartition("_frame")
        if not fno.isdigit():
            continue
        try:
            boxes = [b["bbox_xyxy"] for b in json.load(open(p, encoding="utf-8"))]
        except Exception:
            continue
        gt.setdefault(vid, {})[int(fno)] = boxes
    return gt


def read_masks(path, max_frames=0):
    import cv2
    cap = cv2.VideoCapture(path)
    out = []
    while True:
        ok, f = cap.read()
        if not ok:
            break
        out.append(((f[..., 0] if f.ndim == 3 else f) >= 128).astype(np.uint8))
        if max_frames and len(out) >= max_frames:
            break
    cap.release()
    return out


def run_tier1(video, outdir, max_frames=0):
    """Run the DEPLOYED Tier-1 config on one video; returns (masks, stride, native_wh)."""
    import subprocess
    drv = os.path.join(outdir, "_run_one.py")
    os.makedirs(outdir, exist_ok=True)
    with open(drv, "w", encoding="utf-8") as f:
        f.write(f'''
import os, sys, json
sys.path.insert(0, r"{EDGE}")
import config as C
C.OUT_DIR = sys.argv[2]
os.makedirs(C.OUT_DIR, exist_ok=True)
import cv2, mirage_tier1 as ST
cap = cv2.VideoCapture(sys.argv[1]); fps = cap.get(cv2.CAP_PROP_FPS); cap.release()
stride = max(1, int(round(fps / float(C.EMIT_FPS)))) if fps and fps > 0 else 1
json.dump({{"stride": stride, "src_fps": fps}}, open(os.path.join(C.OUT_DIR, "_meta.json"), "w"))
sys.argv = ["mirage_tier1.py", "--source", sys.argv[1], "--max-frames", sys.argv[3]]
ST.main()
''')
    r = subprocess.run([sys.executable, drv, video, outdir, str(max_frames)],
                       capture_output=True, text=True, encoding="utf-8", errors="replace")
    mp = os.path.join(outdir, "mask.mkv")
    if r.returncode != 0 or not os.path.exists(mp):
        return None, None, (r.stderr or "")[-500:]
    meta = json.load(open(os.path.join(outdir, "_meta.json")))
    return read_masks(mp), int(meta["stride"]), None


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--gt", default=os.path.join(REPO, "input", "reid_dataset", "_annotations"))
    ap.add_argument("--videos", default=os.path.join(REPO, "input", "reid_dataset"))
    ap.add_argument("--work", default=os.path.join(REPO, "input", "reid_dataset", "_fc_work"))
    ap.add_argument("--out", default=os.path.join(HERE, "reports"))
    ap.add_argument("--max-frames", type=int, default=0)
    ap.add_argument("--cover-thresh", type=float, default=1.0)
    a = ap.parse_args()

    gt = load_gt(a.gt)
    print(f"[fc] GT: {len(gt)} videos, {sum(len(v) for v in gt.values())} annotated frames, "
          f"{sum(len(b) for v in gt.values() for b in v.values())} face boxes")

    # which GT videos do we actually have footage for?
    man_p = os.path.join(a.videos, "MANIFEST.json")
    have = {}
    if os.path.exists(man_p):
        for rec in json.load(open(man_p, encoding="utf-8"))["files"]:
            name = rec.get("gt_video_name")
            if name and name in gt:
                have.setdefault(name, os.path.join(a.videos, rec["new_name"]))
    # FALLBACK (2026-07-24): the MANIFEST maps only a few videos, but the systematic annotated set
    # (Size/Num/Movement of Faces) has GT and footage with matching BASENAMES. Match any remaining GT
    # video to a footage file by basename anywhere under --videos, so the full corpus becomes scorable.
    if len(have) < len(gt):
        cache = {}
        for p in glob.glob(os.path.join(a.videos, "**", "*.mp4"), recursive=True):
            cache.setdefault(os.path.splitext(os.path.basename(p))[0], p)
        for name in gt:
            if name not in have and name in cache:
                have[name] = cache[name]
    if not have:
        raise SystemExit(f"no GT video is present in {a.videos} (MANIFEST gt_video_name). "
                         f"Supply the source videos; annotations alone cannot be scored.")
    print(f"[fc] footage available for {len(have)}/{len(gt)} annotated videos: {sorted(have)}")

    rows, per_video = [], []
    for vid, path in sorted(have.items()):
        masks, stride, err = run_tier1(path, os.path.join(a.work, vid), a.max_frames)
        if masks is None:
            print(f"  {vid}: TIER-1 FAILED - {err}")
            continue
        H, W = masks[0].shape
        scored = skipped = 0
        covs = []
        for fno, boxes in sorted(gt[vid].items()):
            if fno % stride:
                skipped += 1
                continue
            e = fno // stride
            if e >= len(masks):
                skipped += 1
                continue
            m = masks[e]
            for b in boxes:
                x1, y1, x2, y2 = [int(round(v)) for v in b]
                x1, y1 = max(0, x1), max(0, y1); x2, y2 = min(W, x2), min(H, y2)
                if x2 <= x1 or y2 <= y1:
                    continue
                covs.append(float(m[y1:y2, x1:x2].mean()))
                scored += 1
        if not covs:
            print(f"  {vid}: no scorable GT faces"); continue
        c = np.array(covs)
        rec = {"video": vid, "stride": stride, "frames_emitted": len(masks),
               "faces_scored": int(c.size), "gt_frames_skipped_by_decimation": skipped,
               "mean_coverage": float(c.mean()),
               "fully_covered_pct": float((c >= 1.0).mean() * 100),
               "cov_ge_99_pct": float((c >= 0.99).mean() * 100),
               "cov_ge_95_pct": float((c >= 0.95).mean() * 100),
               "any_exposure_n": int((c < 1.0).sum()),
               "cov_lt_50_n": int((c < 0.50).sum()),
               "uncovered_n": int((c <= 0.0).sum())}
        per_video.append(rec); rows.append(c)
        print(f"  {vid:26s} stride={stride} faces={rec['faces_scored']:4d}  "
              f"full={rec['fully_covered_pct']:6.2f}%  >=99%={rec['cov_ge_99_pct']:6.2f}%  "
              f"mean={rec['mean_coverage']:.4f}  any-exposure={rec['any_exposure_n']:4d}  "
              f"<50%={rec['cov_lt_50_n']:3d}  zero={rec['uncovered_n']:3d}")

    C = np.concatenate(rows)
    agg = {"videos": len(per_video), "faces_scored": int(C.size),
           "mean_coverage": float(C.mean()),
           "fully_covered_pct": float((C >= 1.0).mean() * 100),
           "cov_ge_99_pct": float((C >= 0.99).mean() * 100),
           "cov_ge_95_pct": float((C >= 0.95).mean() * 100),
           "any_exposure_n": int((C < 1.0).sum()), "cov_lt_50_n": int((C < 0.50).sum()),
           "uncovered_n": int((C <= 0.0).sum())}
    print("\n" + "=" * 92)
    print(f"FACE COVERAGE vs ANNOTATED GT - {agg['faces_scored']} face boxes over "
          f"{agg['videos']} video(s)")
    print(f"  fully covered (100%) : {agg['fully_covered_pct']:6.2f}%")
    print(f"  coverage >= 99%      : {agg['cov_ge_99_pct']:6.2f}%")
    print(f"  coverage >= 95%      : {agg['cov_ge_95_pct']:6.2f}%")
    print(f"  mean coverage        : {agg['mean_coverage']:.4f}")
    print(f"  ANY exposure         : {agg['any_exposure_n']} faces")
    print(f"  <50% covered         : {agg['cov_lt_50_n']} faces")
    print(f"  completely uncovered : {agg['uncovered_n']} faces")
    print("=" * 92)

    os.makedirs(a.out, exist_ok=True)
    p = os.path.join(a.out, "FACE_COVERAGE_GT.json")
    json.dump({"aggregate": agg, "per_video": per_video,
               "note": "Scored against HUMAN-ANNOTATED face boxes, not the pipeline's own "
                       "detections - a face missed by both the annotator's model and ours would "
                       "still be missed, but a face our detector misses is now counted as a "
                       "reveal instead of a success.",
               "gt_videos_total": len(gt), "gt_videos_scored": len(per_video)},
              open(p, "w"), indent=2)
    print(f"\n[fc] wrote {p}")
    if len(per_video) < len(gt):
        print(f"[fc] {len(gt) - len(per_video)} annotated videos have no footage here - "
              f"supply them to make this the full-corpus number.")


if __name__ == "__main__":
    main()
