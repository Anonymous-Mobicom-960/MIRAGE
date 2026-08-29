#!/usr/bin/env python3
"""AUDIT: how many REAL people does the emitted mask leave uncovered?
Same rule as _e2e/run3_20260807/audit_unmasked_people.py (COVER_MIN 0.50), run independently of
the pipeline's own refused_boxes log -- that log is a SAMPLE the pipeline chose to record, not a
measurement. Detector is the same yolo11n.onnx Tier-1 uses, swept over the SOURCE frames.

WHY IT MATTERS: to the CLOUD an uncovered person is safe (the plate is mask-over-lightmap), but in
the FINAL VIDEO Phase 1 reconstructs the real background, so an uncovered real person appears AS
THEMSELVES."""
import json, os, sys
import cv2, numpy as np
EDGE = os.environ.get("MIRAGE_EDGE_DIR", os.path.join(
    os.path.dirname(os.path.abspath(__file__)), "..", "..", "..",
    "tier1", "src", "edge_runner_pi5"))
sys.path.insert(0, EDGE)
import mirage_tier1 as ST
COVER_MIN, DET_THRESH = 0.50, 0.5
src, mask_p, out = sys.argv[1], sys.argv[2], sys.argv[3]
det = ST.YOLO11n(os.path.join(EDGE, "models", "yolo11n.onnx"), ["CPUExecutionProvider"])
cs, cm = cv2.VideoCapture(src), cv2.VideoCapture(mask_p)
frames = 0; dets = 0; uncovered = []; covs = []
while True:
    oks, fs = cs.read(); okm, fm = cm.read()
    if not (oks and okm): break
    g = cv2.cvtColor(fm, cv2.COLOR_BGR2GRAY) > 127
    for b in det(fs, DET_THRESH):
        x1, y1, x2, y2 = [int(round(v)) for v in b]
        x1, y1 = max(x1, 0), max(y1, 0); x2, y2 = min(x2, g.shape[1]), min(y2, g.shape[0])
        if x2 <= x1 or y2 <= y1: continue
        dets += 1
        cov = float(g[y1:y2, x1:x2].mean()); covs.append(cov)
        if cov < COVER_MIN:
            uncovered.append(dict(frame=frames, box=[x1, y1, x2, y2], coverage=round(cov, 4),
                                  box_area_frac=round((x2-x1)*(y2-y1)/(g.shape[0]*g.shape[1]), 5)))
    frames += 1
cs.release(); cm.release()
res = dict(source=os.path.basename(src), mask=os.path.basename(mask_p), frames=frames,
           person_detections=dets, cover_min=COVER_MIN,
           coverage_mean=round(float(np.mean(covs)), 4) if covs else None,
           coverage_min=round(float(np.min(covs)), 4) if covs else None,
           uncovered_detections=len(uncovered),
           uncovered_frames=len(sorted({u["frame"] for u in uncovered})),
           uncovered=uncovered[:60])
json.dump(res, open(out, "w"), indent=1)
print(json.dumps({k: v for k, v in res.items() if k != "uncovered"}, indent=1))
if uncovered:
    print(f"\nfirst 10 uncovered: {[(u['frame'], u['coverage'], u['box_area_frac']) for u in uncovered[:10]]}")
