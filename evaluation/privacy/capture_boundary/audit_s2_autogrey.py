#!/usr/bin/env python3
"""§2 face-reveal audit with an AUTO-DETECTED fill level.

WHY THIS EXISTS. `evaluation/privacy/capture_boundary/audit_s2_faces.py` hardcodes `GRAY=128, GRAY_TOL=1`, which is
correct for the MIRAGE edge runner's FFV1 LOSSLESS mkv. The MIRAGE host writes a LOSSY H.264
mp4, where the flat fill lands at ~124 -- so every filled pixel fails |px-128|<=1 and the auditor
reports 100 % un-greyed on a frame that is in fact fully covered. That is a FALSE POSITIVE of the
auditor, not a reveal.

This version reads the fill level out of the video itself (modal grey inside the emitted mask) and
uses a tolerance sized to the encoder's own spread, then reports the same three quantities. It also
keeps the DETECTOR result, which needs no fill value at all and is the stronger evidence."""
import json, os, sys
import cv2, numpy as np
src, masked_p, mask_p, out = sys.argv[1], sys.argv[2], sys.argv[3], sys.argv[4]
STRIDE = 1
import mediapipe as mp

def read(p, n=None):
    c, out_ = cv2.VideoCapture(p), []
    while True:
        ok, f = c.read()
        if not ok: break
        out_.append(f)
        if n and len(out_) >= n: break
    c.release(); return out_

S, M, K = read(src), read(masked_p), read(mask_p)
N = min(len(S), len(M), len(K))
print(f"frames {N}", flush=True)

# ---- 1. auto-detect the fill: modal grey INSIDE the emitted mask, and the encoder's spread ----
lv, sd = [], []
for i in range(0, N, max(N // 20, 1)):
    g = cv2.cvtColor(K[i], cv2.COLOR_BGR2GRAY) > 127
    if g.sum() < 100: continue
    px = M[i][g].astype(np.int16)
    lv.append(float(np.median(px))); sd.append(float(px.std()))
GRAY = int(round(float(np.median(lv)))); TOL = max(2, int(np.ceil(3 * float(np.median(sd)))))
print(f"auto fill GRAY={GRAY} TOL=+/-{TOL} (encoder sd {np.median(sd):.2f})", flush=True)

# ---- 2. detectors ------------------------------------------------------------------------------
def mp_faces(frames):
    det = mp.solutions.face_detection.FaceDetection(model_selection=1, min_detection_confidence=0.40)
    r = []
    for f in frames:
        h, w = f.shape[:2]
        res = det.process(cv2.cvtColor(f, cv2.COLOR_BGR2RGB))
        b = []
        for d in (res.detections or []):
            bb = d.location_data.relative_bounding_box
            x1, y1 = int(bb.xmin * w), int(bb.ymin * h)
            b.append((max(x1,0), max(y1,0), min(x1+int(bb.width*w), w), min(y1+int(bb.height*h), h),
                      float(d.score[0])))
        r.append(b)
    det.close(); return r

def yn_faces(frames):
    # OpenCV YuNet, used ONLY as an independent cross-check of the pipeline's own masking --
    # it is never in the runtime path. Download it from the OpenCV Zoo (models/README.md)
    # and point MIRAGE_YUNET_ONNX at it.
    mdl = os.environ.get("MIRAGE_YUNET_ONNX", "face_detection_yunet_2023mar.onnx")
    h, w = frames[0].shape[:2]
    d = cv2.FaceDetectorYN.create(mdl, "", (w, h), 0.60, 0.3, 5000)
    r = []
    for f in frames:
        _, ff = d.detect(f)
        b = []
        for row in (ff if ff is not None else []):
            x, y, ww, hh, sc = row[0], row[1], row[2], row[3], row[-1]
            b.append((max(int(x),0), max(int(y),0), min(int(x+ww), w), min(int(y+hh), h), float(sc)))
        r.append(b)
    return r

raw_mp, msk_mp = mp_faces(S[:N]), mp_faces(M[:N])
raw_yn, msk_yn = yn_faces(S[:N]), yn_faces(M[:N])

def ungrey(fr, box):
    x1, y1, x2, y2 = box[:4]
    sub = fr[y1:y2, x1:x2].astype(np.int16)
    if sub.size == 0: return 0.0
    return float((np.abs(sub - GRAY).max(axis=2) > TOL).mean())

rows = []
for i in range(N):
    for det, dets in (("mediapipe", raw_mp[i]), ("yunet", raw_yn[i])):
        for b in dets:
            rows.append(dict(frame=i, det=det, h=b[3]-b[1], score=round(b[4],3),
                             ungrey=round(ungrey(M[i], b), 4)))
def blk(name):
    rs = [r for r in rows if r["det"] == name]
    u = np.array([r["ungrey"] for r in rs]) if rs else np.zeros(0)
    return dict(gt_faces_scored=len(rs),
                mean_ungrey=round(float(u.mean()),5) if u.size else None,
                max_ungrey=round(float(u.max()),4) if u.size else None,
                reveals_gt_5pct=int((u > 0.05).sum()), severe_gt_50pct=int((u > 0.50).sum()),
                worst=sorted(rs, key=lambda r: -r["ungrey"])[:5])
res = dict(source=os.path.basename(src), frames=N, auto_gray=GRAY, tol=TOL,
           positive_control_raw_source=dict(
               frames_with_a_face_mediapipe=int(sum(1 for f in raw_mp if f)),
               frames_with_a_face_yunet=int(sum(1 for f in raw_yn if f)),
               face_h_px_median_mp=float(np.median([b[3]-b[1] for f in raw_mp for b in f])) if any(raw_mp) else None),
           faces_found_in_EMITTED=dict(
               frames_with_a_face_mediapipe=int(sum(1 for f in msk_mp if f)),
               frames_with_a_face_yunet=int(sum(1 for f in msk_yn if f))),
           ungrey_mediapipe=blk("mediapipe"), ungrey_yunet=blk("yunet"))
json.dump(res, open(out, "w"), indent=1)
print(json.dumps(res, indent=1))
