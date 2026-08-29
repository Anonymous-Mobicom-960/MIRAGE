#!/usr/bin/env python3
"""§2 FACE audit of the SHIPPED guided Tier-1 config, on RUN3 - with a POSITIVE CONTROL.

Closes the internal runbook's open item #2:

    "The §2 audit of the shipped guided config FAILS - 2 reveals / 1232 frames (§A.1j). Every doc
     quoting '0 in-scope reveals / 1172 instances' is describing the retired SEG_BACKEND='none'
     config."

WHAT THIS MEASURES, and why it is a FACE audit and not a person audit.
  The privacy claim under test is "a real face never survives Tier-1". The project's §2 auditor
  (`test_tier1_eval.audit`) scores PERSON PIXELS and uses a face only as a scope gate. That is a
  different quantity: a person can be 10 % ungreyed at the shoe and score a reveal while their face
  is entirely hidden. This script scores the FACE REGION directly, on the artifact Tier-1 actually
  emits (`tier1_out/masked_video.mkv`, FFV1 lossless, GRAY=128).

  A FACE REVEAL = a face box found in the SOURCE frame whose pixels are more than REVEAL_FRAC
  un-greyed in the corresponding emitted frame. 0.05 is used so the threshold matches the project's
  existing §2 reveal rule (`test_tier1_eval.audit`: ">5 % of that instance's pixels not grayed").

🔴 THE POSITIVE CONTROL IS THE POINT.
  A "0 reveals" produced by a detector that cannot see faces at this scale measures nothing - the
  exact failure §A.6o-1b punished (an adversary scoring 18.50 % on UNDEFENDED data against its own
  18.49 % null). So every detector here is run on the RAW SOURCE frames first, and its raw count and
  the face sizes it found are reported ALONGSIDE the masked-video count. If the raw count is ~0 the
  detector is blind and its masked count is void.

TWO INDEPENDENT DETECTORS, deliberately:
  * mediapipe FaceDetection model_selection=1 @ conf 0.40 - this is EXACTLY the detector the
    pipeline's own FaceGuard uses (config.py FACE_GUARD_MODEL=1, FACE_GUARD_CONF=0.40). It is
    circular as evidence that FaceGuard works, but it is the strongest possible end-to-end statement:
    "the pipeline's own face detector cannot find a face in the pipeline's own output."
  * YuNet face_detection_yunet_2023mar @ conf 0.60 - INDEPENDENT of the pipeline (it is the probe the
    §2 auditor's scope gate uses, and is not in the MediaPipe stack under test).

ATTRIBUTION (enrolled vs refused). SUBJECT_LOCK deliberately leaves unenrolled people unmasked
(ledger §A.1k / §A.1m), so a face reveal must be attributed before it is counted. Every face is
matched to a yolo11n person box (the same detector Tier-1 uses) and that person's mask coverage is
measured: a revealed face on a person the mask covers >= COVER_MIN is a genuine COVERAGE FAILURE on
an enrolled subject; a revealed face on a person the mask barely covers is a person Tier-1 REFUSED,
and is reported in its own bucket, never merged into the headline.

REDACTION. Any visual written to disk has every detected face gaussian-blurred BEFORE the file is
written. An audit artifact must not become the leak it is auditing.

  python audit_s2_faces.py                    # all three clips
  python audit_s2_faces.py --clip c3_p20p21_two --dump
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
YOLO_N = os.path.join(EDGE, "models", "yolo11n.onnx")
YUNET = os.environ.get("MIRAGE_YUNET_ONNX") or os.path.join(
    REPO, "models", "face_detection_yunet_2023mar.onnx")
# YuNet is an INDEPENDENT cross-check detector, never in the runtime path. It is not
# redistributed: fetch it from the OpenCV Zoo and either drop it at the path above or
# point MIRAGE_YUNET_ONNX at it (models/README.md). The previous hardcoded path pointed
# into a development tree that does not exist in this release.
OUT = os.path.join(HERE, "fix10")

CLIPS = ["c1_p05_single", "c2_p08_single", "c3_p20p21_two"]

# --- thresholds, all stated rather than tuned -------------------------------------------------
GRAY = 128          # config.GRAY - the silhouette fill value
GRAY_TOL = 1        # same tolerance the project auditor uses: |px-128| <= 1 on all 3 channels
REVEAL_FRAC = 0.05  # >5 % of the face box un-greyed = a reveal (project §2 rule)
SEVERE_FRAC = 0.50  # >50 % un-greyed = the face is substantially present, not an edge sliver
MP_CONF = 0.40      # config.FACE_GUARD_CONF
MP_MODEL = 1        # config.FACE_GUARD_MODEL (full-range)
YUNET_CONF = 0.60   # test_tier1_eval.FACE_CONF
COVER_MIN = 0.50    # a person the mask covers less than this is not meaningfully masked
PERSON_CONF = 0.35  # audit_unmasked_people.py
IMGSZ = 640


# ------------------------------------------------------------------ detectors
def mp_faces(frames):
    """[[(x1,y1,x2,y2,score), ...] per frame] from the pipeline's OWN FaceGuard detector."""
    import mediapipe as mp
    det = mp.solutions.face_detection.FaceDetection(model_selection=MP_MODEL,
                                                    min_detection_confidence=MP_CONF)
    out = []
    for f in frames:
        h, w = f.shape[:2]
        res = det.process(cv2.cvtColor(f, cv2.COLOR_BGR2RGB))
        boxes = []
        for d in (res.detections or []):
            r = d.location_data.relative_bounding_box
            x1, y1 = int(r.xmin * w), int(r.ymin * h)
            x2, y2 = int((r.xmin + r.width) * w), int((r.ymin + r.height) * h)
            x1, y1 = max(0, x1), max(0, y1)
            x2, y2 = min(w, x2), min(h, y2)
            if x2 > x1 and y2 > y1:
                boxes.append((x1, y1, x2, y2, float(d.score[0])))
        out.append(boxes)
    det.close()
    return out


def yunet_faces(frames):
    """[[(x1,y1,x2,y2,score), ...] per frame] from YuNet - INDEPENDENT of the pipeline stack."""
    h, w = frames[0].shape[:2]
    det = cv2.FaceDetectorYN.create(YUNET, "", (w, h), YUNET_CONF, 0.3, 5000)
    out = []
    for f in frames:
        det.setInputSize((f.shape[1], f.shape[0]))
        _, faces = det.detect(f)
        boxes = []
        for d in (faces if faces is not None else []):
            x, y, bw, bh = float(d[0]), float(d[1]), float(d[2]), float(d[3])
            if bw <= 0 or bh <= 0:
                continue
            x1, y1 = max(0, int(x)), max(0, int(y))
            x2, y2 = min(f.shape[1], int(x + bw)), min(f.shape[0], int(y + bh))
            if x2 > x1 and y2 > y1:
                boxes.append((x1, y1, x2, y2, float(d[-1])))
        out.append(boxes)
    return out


def person_boxes(frames):
    """yolo11n person boxes - the SAME detector Tier-1 runs (audit_unmasked_people.py settings)."""
    import onnxruntime as ort
    ses = ort.InferenceSession(YOLO_N, providers=["CPUExecutionProvider"])
    iname = ses.get_inputs()[0].name
    out = []
    for f in frames:
        h, w = f.shape[:2]
        s = min(IMGSZ / w, IMGSZ / h)
        nw, nh = int(w * s), int(h * s)
        canvas = np.full((IMGSZ, IMGSZ, 3), 114, np.uint8)
        canvas[:nh, :nw] = cv2.resize(f, (nw, nh))
        blob = canvas[:, :, ::-1].transpose(2, 0, 1)[None].astype(np.float32) / 255.0
        p = ses.run(None, {iname: blob})[0][0]
        if p.shape[0] < p.shape[1]:
            p = p.T
        boxes = []
        for row in p:
            cls = row[4:]
            c = float(cls[0])
            if c < PERSON_CONF or (len(cls) > 1 and cls.argmax() != 0):
                continue
            cx, cy, bw, bh = row[:4]
            boxes.append((max(0, int((cx - bw / 2) / s)), max(0, int((cy - bh / 2) / s)),
                          min(w, int((cx + bw / 2) / s)), min(h, int((cy + bh / 2) / s)), c))
        boxes.sort(key=lambda b: -b[4])
        keep = []
        for b in boxes:
            if all(_iou(b, k) < 0.45 for k in keep):
                keep.append(b)
        out.append(keep)
    return out


def _iou(a, b):
    ix1, iy1 = max(a[0], b[0]), max(a[1], b[1])
    ix2, iy2 = min(a[2], b[2]), min(a[3], b[3])
    iw, ih = max(0, ix2 - ix1), max(0, iy2 - iy1)
    inter = iw * ih
    ua = (a[2] - a[0]) * (a[3] - a[1]) + (b[2] - b[0]) * (b[3] - b[1]) - inter
    return inter / ua if ua > 0 else 0.0


# ------------------------------------------------------------------ io helpers
def read(path, n=None, step=1, start=0):
    cap = cv2.VideoCapture(path)
    out, i = [], 0
    while True:
        ok, f = cap.read()
        if not ok:
            break
        if i >= start and (i - start) % step == 0:
            out.append(f)
            if n is not None and len(out) >= n:
                break
        i += 1
    cap.release()
    return out


def ungrey(masked_frame, box, tol=GRAY_TOL):
    """Fraction of the box's pixels that are NOT the grey fill in the emitted frame."""
    x1, y1, x2, y2 = box[:4]
    roi = masked_frame[y1:y2, x1:x2]
    if roi.size == 0:
        return 0.0
    g = np.all(np.abs(roi.astype(np.int16) - GRAY) <= tol, axis=2)
    return float((~g).mean())


def redact(img, face_boxes, person_boxes_=()):
    """Blur every face (and, for safety, every person box) BEFORE the image is ever written."""
    out = img.copy()
    for b in list(person_boxes_) + list(face_boxes):
        x1, y1, x2, y2 = [int(v) for v in b[:4]]
        roi = out[y1:y2, x1:x2]
        if roi.size:
            k = max(6.0, (y2 - y1) / 6.0)
            out[y1:y2, x1:x2] = cv2.GaussianBlur(roi, (0, 0), k)
    return out


# ------------------------------------------------------------------ the audit
def audit_clip(clip, dump=False, src_p=None, t1_out=None):
    """`src_p` / `t1_out` override the RUN3 layout so any Tier-1 run directory can be audited."""
    src_p = src_p or os.path.join(HERE, clip, "src_cut.mp4")
    t1_out = t1_out or os.path.join(HERE, clip, "tier1_out")
    mkv_p = os.path.join(t1_out, "masked_video.mkv")
    mp4_p = os.path.join(HERE, "phone_in", clip, "input", "masked_video.mp4")
    pose_p = os.path.join(t1_out, "pose.json")
    for p in (src_p, mkv_p, pose_p):
        if not os.path.exists(p):
            return {"clip": clip, "error": f"missing {p}"}

    pose = json.load(io.open(pose_p, encoding="utf-8"))
    stride = int(pose.get("stride", 3))
    masked = read(mkv_p)
    n = len(masked)
    srcs = read(src_p, n=n, step=stride)          # frames 0, stride, 2*stride, ... - as Tier-1 read them
    n = min(n, len(srcs))
    masked, srcs = masked[:n], srcs[:n]
    H, W = srcs[0].shape[:2]
    masked_mp4 = read(mp4_p)[:n] if os.path.exists(mp4_p) else []

    print(f"  [{clip}] {n} emitted frames, {W}x{H}, stride {stride}", flush=True)

    # ---- POSITIVE CONTROL: both detectors on the RAW source ----------------------------------
    print("    raw source: mediapipe ...", flush=True)
    raw_mp = mp_faces(srcs)
    print("    raw source: yunet ...", flush=True)
    raw_yn = yunet_faces(srcs)
    # ---- THE TEST: the same detectors on the EMITTED masked video -----------------------------
    print("    emitted mkv: mediapipe ...", flush=True)
    msk_mp = mp_faces(masked)
    print("    emitted mkv: yunet ...", flush=True)
    msk_yn = yunet_faces(masked)
    msk_mp4_mp = mp_faces(masked_mp4) if masked_mp4 else []
    msk_mp4_yn = yunet_faces(masked_mp4) if masked_mp4 else []
    # ---- persons, for attribution -------------------------------------------------------------
    print("    persons (yolo11n) ...", flush=True)
    pers = person_boxes(srcs)

    def sizes(dets):
        hs = [b[3] - b[1] for fr in dets for b in fr]
        if not hs:
            return {}
        hs = np.array(hs, float)
        return {"n": int(hs.size), "h_px_min": float(hs.min()), "h_px_med": float(np.median(hs)),
                "h_px_max": float(hs.max()),
                "frac_of_frame_med": float(np.median(hs) / H)}

    # ---- per-face coverage on the emitted artifact --------------------------------------------
    rows = []
    for i in range(n):
        # person boxes + how much of each the emitted mask covers (for attribution)
        pb = []
        for (x1, y1, x2, y2, c) in pers[i]:
            cov = 1.0 - ungrey(masked[i], (x1, y1, x2, y2))
            pb.append((x1, y1, x2, y2, c, cov))
        for src_name, dets in (("mediapipe", raw_mp[i]), ("yunet", raw_yn[i])):
            for b in dets:
                u = ungrey(masked[i], b)
                cx, cy = (b[0] + b[2]) / 2.0, (b[1] + b[3]) / 2.0
                owner_cov, owner = None, None
                for (x1, y1, x2, y2, c, cov) in pb:
                    if x1 <= cx <= x2 and y1 <= cy <= y2:
                        if owner_cov is None or cov > owner_cov:
                            owner_cov, owner = cov, [x1, y1, x2, y2]
                rows.append(dict(frame=i, det=src_name, box=[int(v) for v in b[:4]],
                                 score=round(float(b[4]), 3), h=int(b[3] - b[1]),
                                 ungrey=round(u, 4),
                                 owner_person_box=owner,
                                 owner_mask_cover=None if owner_cov is None else round(owner_cov, 4)))

    def bucket(rs):
        """Split reveals into ENROLLED coverage failures vs REFUSED (deliberately unmasked)."""
        rev = [r for r in rs if r["ungrey"] > REVEAL_FRAC]
        enrolled, refused, unknown = [], [], []
        for r in rev:
            c = r["owner_mask_cover"]
            if c is None:
                unknown.append(r)
            elif c >= COVER_MIN:
                enrolled.append(r)
            else:
                refused.append(r)
        return rev, enrolled, refused, unknown

    res = {"clip": clip, "frames": n, "res": [W, H], "stride": stride,
           "subject_lock": {k: v for k, v in (pose.get("subject_lock") or {}).items()
                            if k != "refused_boxes"},
           "refused_boxes": (pose.get("subject_lock") or {}).get("refused_boxes", []),
           "positive_control_raw_source": {
               "mediapipe_full_range@0.40": sizes(raw_mp),
               "yunet@0.60": sizes(raw_yn),
               "frames_with_a_face_mediapipe": int(sum(1 for f in raw_mp if f)),
               "frames_with_a_face_yunet": int(sum(1 for f in raw_yn if f))},
           "faces_found_in_EMITTED_mkv": {
               "mediapipe": sizes(msk_mp), "yunet": sizes(msk_yn),
               "frames_with_a_face_mediapipe": int(sum(1 for f in msk_mp if f)),
               "frames_with_a_face_yunet": int(sum(1 for f in msk_yn if f))},
           "faces_found_in_UPLOADED_mp4": {
               "mediapipe": sizes(msk_mp4_mp), "yunet": sizes(msk_mp4_yn)} if masked_mp4 else None,
           "persons_detected": int(sum(len(p) for p in pers))}

    for det in ("mediapipe", "yunet"):
        rs = [r for r in rows if r["det"] == det]
        rev, enrolled, refused, unknown = bucket(rs)
        severe = [r for r in rev if r["ungrey"] > SEVERE_FRAC]
        ug = np.array([r["ungrey"] for r in rs]) if rs else np.zeros(0)
        res[f"faces_{det}"] = {
            "gt_faces_scored": len(rs),
            "mean_ungrey": round(float(ug.mean()), 5) if ug.size else None,
            "max_ungrey": round(float(ug.max()), 4) if ug.size else None,
            f"reveals_gt_{int(REVEAL_FRAC*100)}pct": len(rev),
            f"severe_gt_{int(SEVERE_FRAC*100)}pct": len(severe),
            "reveals_ENROLLED_coverage_failures": len(enrolled),
            "reveals_REFUSED_deliberately_unmasked": len(refused),
            "reveals_unattributed": len(unknown),
            "reveal_frames_enrolled": len(sorted({r["frame"] for r in enrolled})),
            "worst_enrolled": sorted(enrolled, key=lambda r: -r["ungrey"])[:6],
            "worst_refused": sorted(refused, key=lambda r: -r["ungrey"])[:6],
            "worst_unattributed": sorted(unknown, key=lambda r: -r["ungrey"])[:6],
        }

    # ---- redacted visual of the worst case ----------------------------------------------------
    if dump and rows:
        worst = max(rows, key=lambda r: r["ungrey"])
        i = worst["frame"]
        allf = [b for b in raw_mp[i]] + [b for b in raw_yn[i]]
        left = redact(srcs[i], allf, [p[:4] for p in pers[i]])
        right = masked[i].copy()   # already grey where masked; blur any face the detector still sees
        right = redact(right, [b for b in msk_mp[i]] + [b for b in msk_yn[i]], [])
        x1, y1, x2, y2 = worst["box"]
        for im, tag in ((left, "SOURCE (redacted)"), (right, "EMITTED masked_video")):
            cv2.rectangle(im, (x1, y1), (x2, y2), (0, 0, 255), 3)
            cv2.putText(im, tag, (24, 48), cv2.FONT_HERSHEY_SIMPLEX, 1.2, (0, 0, 255), 3, cv2.LINE_AA)
        vis = np.hstack([cv2.resize(left, (640, 640)), cv2.resize(right, (640, 640))])
        cv2.putText(vis, f"f{i} worst face: {worst['ungrey']*100:.1f}% un-greyed, {worst['h']}px",
                    (20, 620), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2, cv2.LINE_AA)
        os.makedirs(OUT, exist_ok=True)
        p = os.path.join(OUT, f"S2FACE_{clip}_REDACTED.png")
        cv2.imwrite(p, vis)
        res["redacted_visual"] = p

    os.makedirs(OUT, exist_ok=True)
    json.dump(rows, io.open(os.path.join(OUT, f"S2FACE_rows_{clip}.json"), "w", encoding="utf-8"))
    return res


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--clip", default="")
    ap.add_argument("--dump", action="store_true")
    ap.add_argument("--src", default="", help="override the source video path")
    ap.add_argument("--tier1-out", default="", help="override the Tier-1 output directory")
    a = ap.parse_args()
    clips = [a.clip] if a.clip else CLIPS
    allres = {}
    for c in clips:
        print(f"=== {c} ===", flush=True)
        allres[c] = audit_clip(c, a.dump, a.src or None, a.tier1_out or None)
        os.makedirs(OUT, exist_ok=True)
        json.dump({c: allres[c]}, io.open(os.path.join(OUT, f"S2_FACE_{c.split('_')[0]}.json"),
                                          "w", encoding="utf-8"), indent=1)
        print(json.dumps(allres[c], indent=1)[:2600], flush=True)
    os.makedirs(OUT, exist_ok=True)
    json.dump(allres, io.open(os.path.join(OUT, "S2_FACE_AUDIT.json"), "w", encoding="utf-8"),
              indent=1)
    print(f"\nwrote {os.path.join(OUT, 'S2_FACE_AUDIT.json')}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
