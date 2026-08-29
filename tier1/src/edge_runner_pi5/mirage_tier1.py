#!/usr/bin/env python3
"""
MIRAGE Tier-1 (Smart glasses / Raspberry Pi 5 proxy) — on-device trusted layer.
================================================================================
TORCH-FREE. Uses onnxruntime (CPU), rtmlib, mediapipe, opencv, numpy — nothing that
needs a GPU or torch, so it installs and runs on a Pi 5 (Cortex-A76, no NPU).

This is the VX "edge-viable" model set, ported out of ComfyUI to a real device:
  * DETECT : YOLO11n  (onnx; export int8 for the real device)
  * POSE   : RTMPose-t wholebody (rtmlib; 133 kp -> body-stick + hand + face kp)
  * SEGMENT: Robust Video Matting (RVM MobileNetV3 + ConvGRU) -> temporally-stable
             human alpha; binarized+eroded; per-person split via the tracked boxes.
  * FACE   : MediaPipe FaceMesh -> 12 identity-free expression scalars
             (whole-clip differential privacy applied before emit — see apply_dp() / config.py).

PRIVACY (Tier-1 boundary): the ONLY things this script emits downstream are
  1) masked_video.mkv  — bystanders replaced by a solid gray (128) silhouette;
                         background is real (it will stay on the PHONE, never cloud). LOSSLESS (FFV1).
  2) mask.mkv          — the binary person silhouette (so the phone can extract/fill). LOSSLESS (FFV1):
                         a lossy codec would let DCT ringing re-expand the mask past the erode margin,
                         leaking a ring of real background into the cloud-bound person region (§2).
  3) pose.json         — body + hand keypoints per frame (identity-free; the 68 face landmarks
                         are ZEROED — face-landmark geometry is a re-identifiable soft biometric),
                         anonymized whole-clip per person-slot, plus meta: the clip's AUTO-detected
                         person_count and the stable left-to-right "slot" id of each person entry.
  4) face_scalars.json — 12 expression scalars per person per frame (identity-free), DP'd whole-clip
                         per person-slot (shape unchanged: [frames][persons][12], in slot order)
The bystander's RAW frames never leave this device.

Run:  python3 mirage_tier1.py            (uses config.py)
      python3 mirage_tier1.py --source 0 (camera)   --source clip.mp4
"""
import os, sys, json, argparse, math, time, secrets
import numpy as np
import cv2
import onnxruntime as ort
import config as C
import pose_anon_edge as PA
import person_slots as PS

# These two degrade SILENTLY if absent, which is inconsistent with how this file treats every
# other privacy-relevant dependency (the FFV1 writer fails LOUDLY on purpose). A privacy tool that
# quietly ships a weaker guarantee because a wheel is missing is the worst failure mode available:
# nothing looks wrong. `main()` reports the degradation loudly and records it in pose.json meta, so
# an artifact produced without them is self-identifying rather than indistinguishable.
# NOTE (2026-07-23): the §2 FACE guarantee no longer depends on RTMPose. Since every detection now
# gets a box-derived head rectangle (ledger §A.1d), losing pose costs the tighter pose-derived
# region and the emitted keypoints — not the guarantee itself.
try:
    from rtmlib import RTMPose
except Exception as e:
    RTMPose = None
try:
    import mediapipe as mp
except Exception:
    mp = None
DEGRADED = [n for n, ok in (("rtmlib.RTMPose (pose keypoints + tight head region)", RTMPose is not None),
                            ("mediapipe (12 face scalars)", mp is not None)) if not ok]
try:
    import ncnn                       # optional — only needed if config.DETECT_BACKEND == 'ncnn'
except Exception:
    ncnn = None

# opencv-contrib provides cv2.ximgproc.guidedFilter (edge-aware mask upscale). Detect once; if the
# contrib module is absent we fall back to a plain cv2.resize(INTER_LINEAR) upsample (still correct,
# just a softer WORK_RES->native mask edge). Install opencv-contrib-python to enable the guided path.
try:
    cv2.ximgproc.guidedFilter        # attribute probe (raises if contrib is not installed)
    HAS_XIMGPROC = True
except Exception:
    HAS_XIMGPROC = False


def letterbox(bgr, R, pad_val=114):
    """Native BGR frame -> RxR letterbox COPY for inference (top-left placement, pad=114 to match
    the YOLO convention). Returns (work, scale, nh, nw):
      * native (x, y) maps to work (x*scale, y*scale); invert a work coord with  / scale.
      * only work[:nh, :nw] holds real pixels — the rest is padding (strip it before upscaling a mask).
    The native frame itself is never modified; this is a throwaway copy fed to the models."""
    H, W = bgr.shape[:2]
    scale = min(R / float(H), R / float(W))
    nh, nw = int(round(H * scale)), int(round(W * scale))
    work = np.full((R, R, 3), pad_val, np.uint8)
    work[:nh, :nw] = cv2.resize(bgr, (nw, nh))
    return work, scale, nh, nw


# ----------------------------- YOLO11n detector -----------------------------
class YOLO11n:
    def __init__(self, path, providers):
        self.s = ort.InferenceSession(path, providers=providers)
        self.inp = self.s.get_inputs()[0].name

    @staticmethod
    def _nms(b, sc, thr=0.5):
        if len(b) == 0:
            return []
        x1, y1, x2, y2 = b[:, 0], b[:, 1], b[:, 2], b[:, 3]
        ar = (x2 - x1) * (y2 - y1); order = sc.argsort()[::-1]; keep = []
        while order.size:
            i = order[0]; keep.append(int(i))
            xx1 = np.maximum(x1[i], x1[order[1:]]); yy1 = np.maximum(y1[i], y1[order[1:]])
            xx2 = np.minimum(x2[i], x2[order[1:]]); yy2 = np.minimum(y2[i], y2[order[1:]])
            w = np.maximum(0, xx2 - xx1); h = np.maximum(0, yy2 - yy1); inter = w * h
            iou = inter / (ar[i] + ar[order[1:]] - inter + 1e-9)
            order = order[1:][iou <= thr]
        return keep

    def __call__(self, bgr, thr):
        H, W = bgr.shape[:2]; inp = 640
        sc = min(inp / H, inp / W); nh, nw = int(round(H * sc)), int(round(W * sc))
        pad = np.full((inp, inp, 3), 114, np.uint8); pad[:nh, :nw] = cv2.resize(bgr, (nw, nh))
        rgb = cv2.cvtColor(pad, cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0
        out = self.s.run(None, {self.inp: rgb.transpose(2, 0, 1)[None]})[0]
        out = np.squeeze(out, 0)
        if out.shape[0] < out.shape[1]:
            out = out.T
        person = out[:, 4]; keep = person >= thr
        if not keep.any():
            return np.empty((0, 4), np.float32)
        xywh = out[keep, :4]; s = person[keep]
        cx, cy, w, h = xywh.T
        box = np.stack([cx - w / 2, cy - h / 2, cx + w / 2, cy + h / 2], 1) / sc
        box[:, 0::2] = np.clip(box[:, 0::2], 0, W); box[:, 1::2] = np.clip(box[:, 1::2], 0, H)
        idx = self._nms(box, s, 0.5)
        return box[idx].astype(np.float32)


# ------------------- YOLO11n (NCNN) detector — READY-TO-ENABLE --------------
# TODO(on-device): requires an NCNN export of yolo11n (yolo11n.param + yolo11n.bin, int8 for the Pi5)
#   and `pip install ncnn`. MEASURE latency + quality on the Pi5 before switching DETECT_BACKEND to
#   'ncnn' as the default — README notes int8 NCNN as the recommended real-HW detect path.
# Same interface as YOLO11n: __call__(bgr, thr) -> (N, 4) xyxy boxes in the INPUT frame's coords.
class YOLO11nNCNN:
    def __init__(self, param_path, bin_path, inp=640):
        if ncnn is None:
            raise RuntimeError("DETECT_BACKEND='ncnn' but the 'ncnn' python package is not installed "
                               "(pip install ncnn).")
        self.inp = inp
        self.net = ncnn.Net()
        # Pi5 tuning knobs (mirror run.sh OMP threads); adjust on device:
        self.net.opt.num_threads = 4
        self.net.opt.use_vulkan_compute = False       # Pi5 GPU path is not reliable for ncnn; CPU.
        self.net.load_param(param_path)
        self.net.load_model(bin_path)
        # TODO(on-device): confirm the exported blob names — Ultralytics NCNN exports commonly use
        #   input "in0" and output "out0". Set these to match `netron yolo11n.param`.
        self.in_blob = "in0"
        self.out_blob = "out0"

    def __call__(self, bgr, thr):
        H, W = bgr.shape[:2]; inp = self.inp
        sc = min(inp / H, inp / W); nh, nw = int(round(H * sc)), int(round(W * sc))
        pad = np.full((inp, inp, 3), 114, np.uint8); pad[:nh, :nw] = cv2.resize(bgr, (nw, nh))
        rgb = cv2.cvtColor(pad, cv2.COLOR_BGR2RGB)
        mat = ncnn.Mat.from_pixels(rgb, ncnn.Mat.PixelType.PIXEL_RGB, inp, inp)
        mat.substract_mean_normalize([0.0, 0.0, 0.0], [1 / 255.0, 1 / 255.0, 1 / 255.0])
        ex = self.net.create_extractor()
        ex.input(self.in_blob, mat)
        _, out = ex.extract(self.out_blob)
        out = np.array(out)                            # (C, A) or (A, C) depending on the export
        if out.ndim == 1:
            out = out.reshape(1, -1)
        if out.shape[0] < out.shape[1]:
            out = out.T                                # -> (anchors, channels), matches the onnx path
        person = out[:, 4]; keep = person >= thr
        if not keep.any():
            return np.empty((0, 4), np.float32)
        xywh = out[keep, :4]; s = person[keep]
        cx, cy, w, h = xywh.T
        box = np.stack([cx - w / 2, cy - h / 2, cx + w / 2, cy + h / 2], 1) / sc
        box[:, 0::2] = np.clip(box[:, 0::2], 0, W); box[:, 1::2] = np.clip(box[:, 1::2], 0, H)
        idx = YOLO11n._nms(box, s, 0.5)
        return box[idx].astype(np.float32)


# ----------------------------- RVM video matte -----------------------------
class RVM:
    def __init__(self, path, providers):
        self.s = ort.InferenceSession(path, providers=providers)
        ins = {i.name: i for i in self.s.get_inputs()}
        outs = [o.name for o in self.s.get_outputs()]
        pk = lambda cs, pool: next((n for c in cs for n in pool if c in n.lower()), None)
        self.src = pk(["src", "input", "image"], ins) or list(ins)[0]
        self.ri = [pk([f"r{k}i", f"r{k}"], ins) for k in (1, 2, 3, 4)]
        self.dsr = pk(["downsample", "ratio"], ins)
        self.dsr_shape = list(ins[self.dsr].shape) if self.dsr else None
        self.pha = pk(["pha", "alpha"], outs) or (outs[1] if len(outs) > 1 else outs[0])
        self.ro = [n for n in outs if "r" in n.lower() and "o" in n.lower()][:4]
        self.rec = None

    def reset(self):
        self.rec = [np.zeros([1, 1, 1, 1], np.float32) for _ in range(4)]

    def _dsr(self, v):
        if self.dsr_shape == [1]:
            return np.array([v], np.float32)
        return np.array(v, np.float32)

    def alpha(self, rgb01, dsr):
        """rgb01: HxWx3 float 0..1 RGB -> HxW alpha 0..1 (carries temporal state)."""
        if self.rec is None:
            self.reset()
        H, W = rgb01.shape[:2]
        feed = {self.src: rgb01.transpose(2, 0, 1)[None].astype(np.float32)}
        for k, r in zip(self.ri, self.rec):
            if k is not None:
                feed[k] = r
        if self.dsr:
            feed[self.dsr] = self._dsr(dsr)
        outs = self.s.run(None, feed)
        om = {o.name: v for o, v in zip(self.s.get_outputs(), outs)}
        if self.ro:
            self.rec = [om[r] for r in self.ro]
        a = np.asarray(om[self.pha]).squeeze()
        if a.shape != (H, W):
            a = cv2.resize(a.astype(np.float32), (W, H))
        return a.astype(np.float32)


# ------------------- PP-HumanSegV2-Lite matte — READY-TO-ENABLE -------------
# TODO(on-device): requires the PP-HumanSegV2-Lite onnx export (config.PPHUMANSEG_ONNX). MEASURE
#   latency + quality on the Pi5 before switching MATTE_MODEL to 'pphumanseg' as the default.
#   README: ~15.86 ms / Cortex-A76 CPU (proven), at a small temporal-stability cost vs RVM (this is
#   a per-frame segmenter — no ConvGRU recurrence — so reset() is a no-op).
# Same interface as RVM: reset() + alpha(rgb01, dsr) -> HxW float alpha in 0..1 (dsr is ignored).
class PPHumanSeg:
    def __init__(self, path, providers, in_size=(192, 192)):
        self.s = ort.InferenceSession(path, providers=providers)
        self.inp = self.s.get_inputs()[0].name
        self.out = self.s.get_outputs()[0].name
        self.in_size = in_size          # TODO(on-device): confirm export input HxW (Lite is 192x192)

    def reset(self):
        pass                            # stateless per-frame segmenter (kept for a uniform matte API)

    def alpha(self, rgb01, dsr=None):
        """rgb01: HxWx3 float 0..1 RGB -> HxW float alpha 0..1 (no temporal state)."""
        H, W = rgb01.shape[:2]
        iw, ih = self.in_size
        x = cv2.resize(rgb01.astype(np.float32), (iw, ih))
        # TODO(on-device): PP-HumanSeg preprocessing is (img-0.5)/0.5 on RGB 0..1; confirm vs the export.
        x = (x - 0.5) / 0.5
        out = self.s.run([self.out], {self.inp: x.transpose(2, 0, 1)[None].astype(np.float32)})[0]
        out = np.squeeze(out, 0)                          # (C, h, w) logits, C=2 (bg, person)
        if out.ndim == 3 and out.shape[0] == 2:
            e = np.exp(out - out.max(0, keepdims=True))
            a = (e[1] / e.sum(0))                         # softmax -> person-channel probability
        else:
            a = out.squeeze()                             # single-channel sigmoid export
        a = cv2.resize(a.astype(np.float32), (W, H))
        return a.astype(np.float32)


# ------------------- YOLO11n-seg person instances — READY-TO-ENABLE ----------
# TORCH-FREE ONNX decode of the YOLO11-seg head (proto masks + per-detection coeffs). Produces a
# PERSON-SHAPED silhouette (covers the face pixel-accurately) that is UNION-ed into the RVM matte so
# a single-foreground matte dropout can never reveal the person. §2 is ALREADY guaranteed without this
# by the detection-box fallback in main(); seg only makes the covered region tighter/person-shaped.
# Default OFF (config.SEG_BACKEND='none') because yolo11n-seg is heavy on a Pi5 CPU — MEASURE on device
# (enable where compute allows, e.g. the Meta Ray-Ban SoC). Validated locally vs ultralytics (see
# tests/validate_seg_onnx.py: IoU ~0.98 on the real clips).
# Output0 (1, 4+nc+32, A) detections; Output1 proto (1, 32, mh, mw).
class YOLO11nSeg:
    def __init__(self, path, providers, inp=640, conf=0.25, iou=0.5, person_cls=0, nc=80):
        self.s = ort.InferenceSession(path, providers=providers)
        self.in_name = self.s.get_inputs()[0].name
        self.inp = inp; self.conf = conf; self.iou = iou; self.pc = person_cls; self.nc = nc

    def person_mask(self, bgr):
        """bgr HxWx3 -> (mask HxW uint8 {0,1} = union of person instances, boxes[list xyxy native])."""
        H, W = bgr.shape[:2]; inp = self.inp
        sc = min(inp / H, inp / W); nh, nw = int(round(H * sc)), int(round(W * sc))
        pad = np.full((inp, inp, 3), 114, np.uint8); pad[:nh, :nw] = cv2.resize(bgr, (nw, nh))
        rgb = cv2.cvtColor(pad, cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0
        outs = self.s.run(None, {self.in_name: rgb.transpose(2, 0, 1)[None]})
        proto = next(o for o in outs if o.ndim == 4)[0]                 # (32, mh, mw)
        det = next(o for o in outs if o.ndim == 3)[0]                   # (C, A) with C = 4+nc+32
        if det.shape[0] > det.shape[1]:
            det = det.T                                                 # ensure (C, A)
        nm = proto.shape[0]                                             # 32 mask coeffs
        box = det[:4].T                                                 # (A,4) cxcywh @ inp
        cls = det[4:4 + self.nc].T                                      # (A,nc)
        coef = det[4 + self.nc:4 + self.nc + nm].T                      # (A,32)
        score = cls[:, self.pc]
        keep = score >= self.conf
        m = np.zeros((H, W), np.uint8); boxes = []
        if not keep.any():
            return m, boxes
        box, coef, score = box[keep], coef[keep], score[keep]
        cx, cy, bw, bh = box.T
        xyxy = np.stack([cx - bw / 2, cy - bh / 2, cx + bw / 2, cy + bh / 2], 1)   # @ inp coords
        idx = YOLO11n._nms(np.clip(xyxy, 0, inp), score, self.iou)
        mh, mw = proto.shape[1:]
        pflat = proto.reshape(nm, -1).astype(np.float32)               # (32, mh*mw)
        for i in idx:
            mk = 1.0 / (1.0 + np.exp(-(coef[i] @ pflat)))              # sigmoid -> (mh*mw,)
            mk = mk.reshape(mh, mw)
            mk = cv2.resize(mk, (inp, inp), interpolation=cv2.INTER_LINEAR)
            x1, y1, x2, y2 = [int(v) for v in xyxy[i]]                  # crop mask to its box (YOLO conv.)
            crop = np.zeros((inp, inp), np.float32)
            x1, y1 = max(0, x1), max(0, y1); x2, y2 = min(inp, x2), min(inp, y2)
            crop[y1:y2, x1:x2] = mk[y1:y2, x1:x2]
            crop = crop[:nh, :nw]                                       # strip letterbox padding
            crop = cv2.resize(crop, (W, H), interpolation=cv2.INTER_LINEAR)  # -> native
            m |= (crop >= 0.5).astype(np.uint8)
            bx = xyxy[i] / sc                                           # box -> native
            boxes.append([float(np.clip(bx[0], 0, W)), float(np.clip(bx[1], 0, H)),
                          float(np.clip(bx[2], 0, W)), float(np.clip(bx[3], 0, H))])
        return m, boxes


# ------------------- YOLO11n-seg (NCNN) — Pi5 canonical path, READY-TO-ENABLE -------------
# NCNN-fp16 decode of the YOLO11-seg head for SEG_FORMAT="ncnn" (~2× ONNX on the Pi5 A76, ledger §C).
# SAME person_mask() contract + decode math as YOLO11nSeg above (kept a faithful COPY — if you touch the
# ONNX decode, mirror it here). ⚠️ NOT YET VALIDATED ON DEVICE: the input/output blob names below are the
# usual Ultralytics NCNN seg export names but MUST be confirmed with `netron model.param` on the Pi before
# trusting this path; the ONNX path is the tested default. Needs `pip install ncnn`.
class YOLO11nSegNCNN:
    def __init__(self, param_path, bin_path, inp=256, conf=0.10, iou=0.5, person_cls=0, nc=80, nm=32):
        if ncnn is None:
            raise RuntimeError("SEG_FORMAT='ncnn' but the 'ncnn' python package is not installed "
                               "(pip install ncnn).")
        self.net = ncnn.Net()
        self.net.opt.num_threads = 4
        self.net.opt.use_vulkan_compute = False           # Pi5 GPU path unreliable for ncnn; CPU.
        self.net.load_param(param_path); self.net.load_model(bin_path)
        self.inp = inp; self.conf = conf; self.iou = iou; self.pc = person_cls; self.nc = nc; self.nm = nm
        # TODO(on-device): confirm these against `netron model.param`. Ultralytics seg exports commonly
        # use input "in0", detection output "out0", proto output "out1".
        self.in_blob = "in0"; self.det_blob = "out0"; self.proto_blob = "out1"

    def person_mask(self, bgr):
        """bgr -> (mask HxW {0,1} = union of persons, boxes[xyxy native]). Same contract as YOLO11nSeg."""
        H, W = bgr.shape[:2]; inp = self.inp
        sc = min(inp / H, inp / W); nh, nw = int(round(H * sc)), int(round(W * sc))
        pad = np.full((inp, inp, 3), 114, np.uint8); pad[:nh, :nw] = cv2.resize(bgr, (nw, nh))
        rgb = cv2.cvtColor(pad, cv2.COLOR_BGR2RGB)
        mat = ncnn.Mat.from_pixels(rgb, ncnn.Mat.PixelType.PIXEL_RGB, inp, inp)
        mat.substract_mean_normalize([0.0, 0.0, 0.0], [1 / 255.0, 1 / 255.0, 1 / 255.0])
        ex = self.net.create_extractor(); ex.input(self.in_blob, mat)
        _, det = ex.extract(self.det_blob); _, proto = ex.extract(self.proto_blob)
        det = np.array(det); proto = np.array(proto)
        if det.ndim == 1:
            det = det.reshape(4 + self.nc + self.nm, -1)
        if det.shape[0] != 4 + self.nc + self.nm and det.shape[1] == 4 + self.nc + self.nm:
            det = det.T                                       # -> (C, A)
        if proto.ndim == 2:                                   # (nm, mh*mw) -> (nm, s, s), square proto
            s = int(round(proto.shape[1] ** 0.5)); proto = proto.reshape(self.nm, s, s)
        # ---- identical decode to YOLO11nSeg.person_mask (COPY — keep in sync) ----
        nm = proto.shape[0]
        box = det[:4].T; cls = det[4:4 + self.nc].T; coef = det[4 + self.nc:4 + self.nc + nm].T
        score = cls[:, self.pc]; keep = score >= self.conf
        m = np.zeros((H, W), np.uint8); boxes = []
        if not keep.any():
            return m, boxes
        box, coef, score = box[keep], coef[keep], score[keep]
        cx, cy, bw, bh = box.T
        xyxy = np.stack([cx - bw / 2, cy - bh / 2, cx + bw / 2, cy + bh / 2], 1)
        idx = YOLO11n._nms(np.clip(xyxy, 0, inp), score, self.iou)
        mh, mw = proto.shape[1:]; pflat = proto.reshape(nm, -1).astype(np.float32)
        for i in idx:
            mk = (1.0 / (1.0 + np.exp(-(coef[i] @ pflat)))).reshape(mh, mw)
            mk = cv2.resize(mk, (inp, inp), interpolation=cv2.INTER_LINEAR)
            x1, y1, x2, y2 = [int(v) for v in xyxy[i]]
            crop = np.zeros((inp, inp), np.float32)
            x1, y1 = max(0, x1), max(0, y1); x2, y2 = min(inp, x2), min(inp, y2)
            crop[y1:y2, x1:x2] = mk[y1:y2, x1:x2]
            crop = cv2.resize(crop[:nh, :nw], (W, H), interpolation=cv2.INTER_LINEAR)
            m |= (crop >= 0.5).astype(np.uint8)
            bx = xyxy[i] / sc
            boxes.append([float(np.clip(bx[0], 0, W)), float(np.clip(bx[1], 0, H)),
                          float(np.clip(bx[2], 0, W)), float(np.clip(bx[3], 0, H))])
        return m, boxes


def build_seg(cfg):
    """The SHIPPED primary person-mask segmenter (`SEG_BACKEND="yolo11n"`). SEG_FORMAT selects the
    backend: 'onnx' (default, PC/S25 — tested) | 'ncnn' (Pi5 fp16 — Pi-validate blob names first)."""
    if getattr(cfg, "SEG_BACKEND", "none") == "yolo11n":
        conf = float(getattr(cfg, "SEG_CONF", 0.10))          # low = recall (0.25 drops motion-blur, C6)
        if getattr(cfg, "SEG_FORMAT", "onnx") == "ncnn":
            return YOLO11nSegNCNN(getattr(cfg, "YOLO11N_SEG_NCNN_PARAM", ""),
                                  getattr(cfg, "YOLO11N_SEG_NCNN_BIN", ""), conf=conf)
        return YOLO11nSeg(cfg.YOLO11N_SEG_ONNX, cfg.ORT_PROVIDERS, conf=conf)
    return None


def guided_seg_post(seg_m, frame, C):
    """Edge-align the person-shaped seg silhouette to the real image boundary — the
    tier1_lab/reports/DECISION.md guided WINNER: guided filter (radius 12, eps 0.01 on a [0,1]-
    normalized gray guide) -> re-binarize -> morph-close (k5) -> OUTWARD d4 privacy dilation. seg_m is
    a native {0,1} uint8 mask; frame is the native BGR. The guided edge-align needs cv2.ximgproc
    (opencv-contrib); if it is absent we still apply close+d4 (a valid, if softer, silhouette)."""
    if not getattr(C, "SEG_GUIDED", True):
        return seg_m
    m = seg_m.astype(np.float32)
    if HAS_XIMGPROC:
        try:
            guide = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY).astype(np.float32) / 255.0
            m = cv2.ximgproc.guidedFilter(guide, m, int(getattr(C, "SEG_GUIDED_RADIUS", 12)),
                                          float(getattr(C, "SEG_GUIDED_EPS", 0.01)))
        except Exception:
            pass
    b = (m > 0.5).astype(np.uint8)
    ck = int(getattr(C, "SEG_GUIDED_CLOSE", 5))
    if ck > 0:
        b = cv2.morphologyEx(b, cv2.MORPH_CLOSE,
                             cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (ck, ck)))
    dp = int(getattr(C, "SEG_GUIDED_DILATE", 4))
    if dp > 0:
        b = cv2.dilate(b, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2 * dp + 1, 2 * dp + 1)))
    return b


class FaceGuard:
    """DECISION.md 'the key privacy piece': a dedicated face detector (MediaPipe FaceDetection, full-
    range) whose expanded-ellipse fill of every detected face is UNIONed into the §2 mask — so a face
    the body seg misses (a nose/mouth in profile, a face at the mask edge) is still grayed, without
    inflating the whole-body mask into a halo. Ported from tier1_lab/segbench/face.py; reuses the
    mediapipe already imported for the 12 face scalars."""
    def __init__(self, min_conf=0.4, expand=0.45, model_selection=1):
        self._fd = mp.solutions.face_detection.FaceDetection(
            model_selection=model_selection, min_detection_confidence=min_conf)
        self.expand = float(expand)

    def face_fill(self, frame_bgr):
        H, W = frame_bgr.shape[:2]
        res = self._fd.process(cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB))
        m = np.zeros((H, W), np.uint8)
        if res.detections:
            for d in res.detections:
                bb = d.location_data.relative_bounding_box
                w = bb.width * W; h = bb.height * H
                cx = bb.xmin * W + w / 2; cy = bb.ymin * H + h / 2
                ax = int(max(2, w / 2 * (1 + self.expand)))   # optimized face ellipse (box * (1+expand))
                ay = int(max(2, h / 2 * (1 + self.expand)))
                cv2.ellipse(m, (int(cx), int(cy)), (ax, ay), 0, 0, 360, 1, -1)
        return m


def build_faceguard(cfg):
    """FaceGuard iff config.FACE_GUARD (default on) and mediapipe is available; else None."""
    if getattr(cfg, "FACE_GUARD", False) and mp is not None:
        return FaceGuard(min_conf=float(getattr(cfg, "FACE_GUARD_CONF", 0.4)),
                         expand=float(getattr(cfg, "FACE_GUARD_EXPAND", 0.45)),
                         model_selection=int(getattr(cfg, "FACE_GUARD_MODEL", 1)))
    return None


# ------------------------- MediaPipe 12 face scalars -------------------------
FACEMESH = None
def face_scalars(face_bgr):
    """Return the 12 identity-free expression scalars from a face crop (or zeros).

    RAW here (only rounded). Differential privacy is NOT applied per-frame: an iid per-frame
    Laplace draw is the wrong mechanism for a temporally-correlated signal. DP is instead applied
    ONCE over the whole clip in apply_dp() (smooth -> quantize -> Laplace), before json.dump.
    """
    global FACEMESH
    if mp is None:
        return [0.0] * 12
    if FACEMESH is None:
        FACEMESH = mp.solutions.face_mesh.FaceMesh(
            static_image_mode=False, max_num_faces=1, refine_landmarks=True,
            min_detection_confidence=0.5, min_tracking_confidence=0.5)
    res = FACEMESH.process(cv2.cvtColor(face_bgr, cv2.COLOR_BGR2RGB))
    if not res.multi_face_landmarks:
        return [0.0] * 12
    lm = res.multi_face_landmarks[0]
    h, w = face_bgr.shape[:2]
    P = np.array([[p.x * w, p.y * h] for p in lm.landmark], np.float32)
    eR, eL = P[33], P[263]; d = np.linalg.norm(eL - eR) + 1e-6
    ec = (eR + eL) / 2.0; roll = math.atan2(eL[1] - eR[1], eL[0] - eR[0])
    c, s = math.cos(-roll), math.sin(-roll)
    Q = (P - ec) @ np.array([[c, -s], [s, c]], np.float32).T / d
    g = lambda i: Q[i]
    mo = max(0.0, g(14)[1] - g(13)[1]); mw = float(np.linalg.norm(g(291) - g(61)))
    sm = ((g(13)[1] + g(14)[1]) / 2) - ((g(61)[1] + g(291)[1]) / 2)
    eRo = max(0.0, g(145)[1] - g(159)[1]); eLo = max(0.0, g(374)[1] - g(386)[1])
    bR = g(159)[1] - g(105)[1]; bL = g(386)[1] - g(334)[1]
    yaw, pitch = g(1)[0], g(1)[1]; gaze = np.zeros(2, np.float32)
    if len(P) >= 478:
        cR = (g(33) + g(133)) / 2; cL = (g(362) + g(263)) / 2
        gaze = ((Q[468] - cR) + (Q[473] - cL)) / 2
    # Channel ORDER (must be preserved everywhere; apply_dp + downstream rely on it):
    #  0 mouth_open  1 mouth_width  2 smile  3 eyeR_open  4 eyeL_open  5 browR  6 browL
    #  7 yaw  8 pitch  9 roll (rad)  10 gazeX  11 gazeY
    v = [mo, mw, float(sm), eRo, eLo, float(bR), float(bL),
         float(yaw), float(pitch), float(roll), float(gaze[0]), float(gaze[1])]
    # float() EVERY channel, not just some: mo/eRo/eLo come out of max(0.0, np.float32) and stay
    # numpy scalars, and round(np.float32) is still np.float32 — which json.dump REFUSES. With
    # DP_ON that was masked because apply_dp rewrites every value through round(float(...)); with
    # DP_ON=False the emit crashed on face_scalars.json. That is the exact path dp_calibrate.py
    # step 1 tells you to run, so the documented calibration workflow could never complete.
    return [round(float(x), 5) for x in v]   # RAW scalars — DP happens whole-clip in apply_dp()


def face_crop_from_box(frame, box, pad=0.15):
    """Upper-body -> face region crop from a person box (fallback when no face kp)."""
    x1, y1, x2, y2 = [int(v) for v in box[:4]]; bw, bh = x2 - x1, y2 - y1
    fx1, fy1 = x1, y1; fx2, fy2 = x2, y1 + int(bh * 0.35)     # top ~35% ≈ head/shoulders
    fx1 = max(0, fx1 - int(bw * pad)); fx2 = min(frame.shape[1], fx2 + int(bw * pad))
    fy2 = min(frame.shape[0], fy2 + int(bh * pad))
    crop = frame[fy1:fy2, fx1:fx2]
    return crop if crop.size else np.zeros((64, 64, 3), np.uint8)


# ------------------------- whole-clip differential privacy -------------------------
def _iou_min(a, b):
    """Intersection over the SMALLER area — so a box nested inside another scores ~1.0."""
    x1, y1 = max(a[0], b[0]), max(a[1], b[1])
    x2, y2 = min(a[2], b[2]), min(a[3], b[3])
    inter = max(0.0, x2 - x1) * max(0.0, y2 - y1)
    ar = min(max(0.0, a[2] - a[0]) * max(0.0, a[3] - a[1]),
             max(0.0, b[2] - b[0]) * max(0.0, b[3] - b[1]))
    return inter / max(1e-6, ar)


def _box_nms(boxes, iou_thr=0.55):
    """Greedy NMS over [x1,y1,x2,y2] native-coord boxes (largest first). Pure numpy."""
    if len(boxes) <= 1:
        return list(boxes)
    b = np.asarray(boxes, np.float32)
    area = np.maximum(0, b[:, 2] - b[:, 0]) * np.maximum(0, b[:, 3] - b[:, 1])
    order = np.argsort(-area)
    keep = []
    while order.size:
        i = order[0]; keep.append(i)
        if order.size == 1:
            break
        r = order[1:]
        xx1 = np.maximum(b[i, 0], b[r, 0]); yy1 = np.maximum(b[i, 1], b[r, 1])
        xx2 = np.minimum(b[i, 2], b[r, 2]); yy2 = np.minimum(b[i, 3], b[r, 3])
        inter = np.maximum(0, xx2 - xx1) * np.maximum(0, yy2 - yy1)
        # IoU against the SMALLER box: a tile detection nested inside a full-frame detection of the
        # same person must be treated as a duplicate, and plain IoU would score that pair low.
        iou = inter / np.maximum(1e-6, np.minimum(area[i], area[r]))
        order = r[iou <= iou_thr]
    return [boxes[i] for i in keep]


def detect_native_tiled(frame, det, thr, work_res):
    """§2 RECALL: detect on NATIVE-resolution tiles, not only on the downscaled work copy.

    THE FAILURE THIS EXISTS TO FIX (measured 2026-07-23, adversarial red team). Detection runs on
    a single `letterbox(frame, WORK_RES)` copy, so at 1264 native a person is shrunk 1.97x before
    YOLO ever sees them. Every §2 mechanism downstream — the box net, the head-keypoint face
    guarantee, the prev_boxes carry — can only iterate boxes the detector returned, so a person
    the detector misses receives NO protection at all: measured detection rate fell to 15% at
    0.31% of frame and 5% at 0.08%, with the bystander's pixels 100% OUTSIDE the emitted mask on
    47 of 60 frames. That is the normal case for smart glasses: a bystander at distance.

    THE RULE IS DERIVED, NOT TUNED. Tile the frame so no tile is downscaled below the DETECTOR'S
    INPUT SIZE — ceil(H/tile_px) x ceil(W/tile_px), tile_px = the model's fixed 640² input — which
    makes the smallest detectable person a fixed fraction of NATIVE pixels regardless of frame size
    or aspect ratio. A 16:9 frame gets more tiles across than down, automatically.

    ⚠️ tile_px is the DETECTOR INPUT (`DETECT_TILE_PX`, =640), NOT `WORK_RES`. Original code used
    ceil(H/work_res), which was a latent footgun: the WORK_RES comment invites "raise toward native
    where compute allows", and at WORK_RES=native the grid collapses to 1x1 → tiling silently OFF →
    small-bystander §2 protection lost. MEASURED 2026-07-24: at 0.338 % of frame, tiling caught the
    bystander on 18/30 frames at WORK_RES 640, but only 2/30 at WORK_RES 1264 (tiling disabled).
    Keying off the fixed detector input decouples tiling from WORK_RES entirely. Also measured:
    raising WORK_RES does NOT help detection at all (the onnx input is a hard 640², so det()
    re-letterboxes everything to 640 regardless) — the sub-0.1 %-of-frame limit is NOT a WORK_RES
    problem; the levers are finer tiling (DETECT_TILE_GRID_SCALE) or a higher-input detector model.
    Returns native-coord boxes; the caller unions these with the full-frame pass.
    """
    H, W = frame.shape[:2]
    gs = max(1, int(getattr(C, "DETECT_TILE_GRID_SCALE", 1)))
    tile_px = float(getattr(C, "DETECT_TILE_PX", 640))       # detector input, NOT work_res
    ny = gs * int(math.ceil(H / tile_px)); nx = gs * int(math.ceil(W / tile_px))
    if nx * ny <= 1:
        return []                                   # frame already fits: tiling adds nothing
    ov = float(getattr(C, "DETECT_TILE_OVERLAP", 0.20))
    th, tw = int(math.ceil(H / ny)), int(math.ceil(W / nx))
    pad_y, pad_x = int(th * ov), int(tw * ov)       # overlap so a person on a seam is whole in one tile
    out = []
    for iy in range(ny):
        for ix in range(nx):
            y0 = max(0, iy * th - pad_y); y1 = min(H, (iy + 1) * th + pad_y)
            x0 = max(0, ix * tw - pad_x); x1 = min(W, (ix + 1) * tw + pad_x)
            crop = frame[y0:y1, x0:x1]
            if crop.size == 0:
                continue
            for b in det(crop, thr).tolist():
                out.append([b[0] + x0, b[1] + y0, b[2] + x0, b[3] + y0])
    return out


def _laplace_secure(scale, shape):
    """Laplace noise drawn from the OS CSPRNG, not numpy's global MT19937.

    The DP guarantee assumes the noise is unpredictable. `np.random.laplace` uses the global
    Mersenne Twister, which is (a) not seeded from any entropy source here, so it is reproducible
    across runs, and (b) fully reconstructible from ~624 outputs — an adversary who recovers the
    state can SUBTRACT the noise and recover the exact smoothed scalars, which is the whole thing
    epsilon is supposed to prevent. The pose anonymizer already seeds from `secrets` for exactly
    this reason; this path did not, which made the weakest link the RNG rather than epsilon.

    Inverse-CDF of the Laplace from uniforms in (0,1) taken from secrets.
    """
    n = int(np.prod(shape)) if shape else 1
    if scale <= 0 or n == 0:
        return np.zeros(shape, np.float64)
    # 53-bit uniforms in (0,1); the open interval matters — log(0) at u=0 would be -inf.
    u = np.array([(secrets.randbits(53) + 0.5) / float(1 << 53) for _ in range(n)], np.float64)
    return (-float(scale) * np.sign(u - 0.5) * np.log1p(-2.0 * np.abs(u - 0.5))).reshape(shape)


def _window_amplification(w, T):
    """Max L1 influence ONE frame can have on the released vector of a length-T slot.

    Delegates to config so there is a single definition, with an inline fallback for stub configs
    (the probe passes one). NOT monotone in T: it peaks at T == w, so a long-clip value is NOT a
    safe bound for a short slot — see the call site in apply_dp.
    """
    f = getattr(C, "_window_amplification", None)
    if callable(f):
        return f(w, T)
    pad, T = int(w) // 2, max(1, int(T))
    if pad == 0:
        return 1.0
    cnt = [min(t + pad, T - 1) - max(t - pad, 0) + 1 for t in range(T)]
    return max(sum(1.0 / cnt[t] for t in range(max(0, i - pad), min(T, i + pad + 1)))
               for i in range(T))


# The value config.SENSITIVITY was baked with, so a hand-raised SENSITIVITY can be rescaled to
# this slot's amplification instead of being silently discarded.
_AMP_BAKED = float(getattr(C, "WINDOW_AMPLIFICATION", 1.0)) or 1.0


def _window_mean(x, w):
    """Centered moving average over the IN-BOUNDS samples only (length preserved).

    NOT edge-replication padding. `mode="edge"` replicates x[0] into pad+1 slots of the first
    window and into every window that reaches back past 0, so frame 0's total influence on the
    released series is (pad+1 + pad + ... + 1)/w = 10/7 for w=7 — i.e. an EDGE frame moves the
    output 1.43x as much as an interior one, and the DP sensitivity bound Δ = range is violated
    exactly there. Measured 2026-07-23 with an adversarial probe (baseline at `lo`, one frame
    driven to `hi`): 1.4194x range at frame 0. An earlier probe of mine missed it by starting
    from the MIDPOINT, which caps a single frame's excursion at range/2 and so under-reported by
    2x — the bug was in the probe, not only in the code.

    Averaging only the real samples is the more honest estimate — edge replication biases the
    smoothed value toward the boundary sample — and it removes the 1.43x replication spike.

    ⚠️ It does NOT make every frame's influence <= 1, and an earlier version of this docstring
    claimed it did. That reasoning covered frame 0 and interior frames and skipped index `pad`,
    which is exactly where the real peak is: frame `pad` sits inside the SHORT windows near the
    boundary, where it is averaged over 4, 5, 6 samples instead of w, so its weights are 1/4 +
    1/5 + 1/6 + 4x(1/7) = 1.188 > 1. Worse, the peak is not monotone in clip length — it is
    largest at T == w (1.376 at w=7). That is why the bound is computed by
    `config._window_amplification(w, T)` per slot rather than assumed. See apply_dp.
    """
    x = np.asarray(x, np.float64)
    w = int(w)
    if w <= 1 or x.size <= 1:
        return x
    pad = w // 2
    ones = np.ones(x.size, np.float64)
    num = np.convolve(np.pad(x, (pad, pad), mode="constant"), np.ones(w, np.float64), mode="valid")
    cnt = np.convolve(np.pad(ones, (pad, pad), mode="constant"), np.ones(w, np.float64), mode="valid")
    return (num / np.maximum(cnt, 1.0))[:x.size]


def apply_dp(face_log, cfg, slot_log=None):
    """Whole-clip differential privacy on the 12 face scalars, applied ONCE before emit.

    Input/return shape is preserved exactly: [frames][persons][12], channel order unchanged
    (see face_scalars()). We NEVER emit landmarks or blendshapes — only the 12 DP'd scalars.

    Why whole-clip instead of per-frame: the expression signal is temporally correlated, so an
    iid per-frame Laplace draw both (a) fails to bound information a correlated stream leaks and
    (b) adds far more noise than needed. Instead, per person-slot, per channel c we:
      (1) window-mean smooth over cfg.DP_WINDOW frames (~0.5 s at the emit fps) — kills per-frame
          jitter that carries no real expression information;
      (2) quantize to cfg.DP_QUANTIZE_BITS bits over a FIXED PUBLIC per-channel range (cfg.CH_BOUNDS)
          — NOT the data-dependent observed [min,max], which would leak the range and break the eps claim;
      (3) add Laplace noise with per-channel scale b_c = SENSITIVITY[c] / eps_c.

    Budgeting: eps_c is either an explicit NON-UNIFORM allocation (cfg.DP_EPS_PER_CH — tighter on the
    gaze/roll channels that carry the most re-id signal) or DP_EPSILON_TOTAL/12. SENSITIVITY (Δ_c) is a
    PLACEHOLDER and MUST be calibrated on real RPi5 clips (see config.py + plan §8 + dp_calibration.md).

    KNOWN LIMITATION (adversary D3 — unit of privacy): the Laplace scale bounds a SINGLE-FRAME change.
    The re-identification threat is a person's whole expression TRAJECTORY; one person contributing all
    their frames can move the released vector by up to T·Δ. True person-level DP needs group privacy
    (scale by the per-person frame count) — the whole-clip window-mean reduces, but does not fully close,
    this. Calibration must set Δ_c and eps_c against the per-PERSON influence, not per-frame.

    "Person" here means the STABLE person SLOT (person_slots.SlotTracker, policy
    "x-sorted+hysteresis"), passed in as `slot_log` ([frames][persons] of slot ids). Omitting
    slot_log falls back to the old positional proxy (person p == index p in every frame) — kept
    only for back-compat: under that proxy one missed detection shifts every later index, so the
    smoothing window would run straight across an identity boundary and the per-person budget
    would be spent on a spliced pseudo-person.
    """
    if not face_log or not getattr(cfg, "DP_ON", True):
        return face_log
    if max((len(fr) for fr in face_log), default=0) == 0:
        return face_log                                    # every frame empty -> nothing to DP
    NCH = 12
    # Per-channel eps (adversary A3): use an explicit NON-UNIFORM budget when provided (tighter eps /
    # more noise on the highest-re-id channels — gaze/roll), else fall back to the even split.
    eps_ch = getattr(cfg, "DP_EPS_PER_CH", None)
    if eps_ch and len(eps_ch) == NCH:
        eps_vec = [float(e) for e in eps_ch]
    else:
        eps_vec = [float(cfg.DP_EPSILON_TOTAL) / NCH] * NCH        # even split (legacy)
    win = max(1, int(cfg.DP_WINDOW))
    levels = max(2, 2 ** int(cfg.DP_QUANTIZE_BITS))
    sens = list(cfg.SENSITIVITY)
    # FIXED PUBLIC per-channel bounds for quantization (adversary D2): quantizing over the observed
    # [min,max] is DATA-DEPENDENT pre-processing (NOT covered by DP post-processing immunity) and its
    # grid leaks the range — one new-extremum frame shifts every value, so the true sensitivity exceeds
    # Δ_c. Quantize over a-priori PUBLIC bounds instead. Fall back to a wide default if unset.
    bounds = getattr(cfg, "CH_BOUNDS", None)
    if not bounds or len(bounds) != NCH:
        bounds = [(-2.0, 2.0)] * NCH
    # deep copy so the raw shape/order is preserved and untouched slots pass through unchanged
    out = [[list(person) for person in frame] for frame in face_log]
    # group cells by STABLE slot id (positional fallback when slot_log is None) so each budget is
    # spent on one real person's trajectory, N-generic in the number of slots.
    for _slot, cells in sorted(PS.slot_groups(face_log, slot_log).items()):
        arr = np.array([face_log[i][j] for i, j in cells], np.float64)   # [T, 12]
        # Skip no-face rows (adversary D4): face_scalars() returns all-zeros when no face was found.
        # DP'ing those would fabricate a non-zero expression for a person whose face was never detected
        # (corrupting the artifact) and waste budget on a deterministic sentinel. Keep them zero.
        present = np.abs(arr).sum(axis=1) > 1e-9
        pres_cells = [cells[t] for t in range(len(cells)) if present[t]]
        if not pres_cells:
            continue                                                # all no-face -> leave the zeros as-is
        arrp = arr[present]
        # PER-SLOT Δ. `config.WINDOW_AMPLIFICATION` is evaluated at one long length, but the
        # amplification is NOT monotone in T: it peaks at T == DP_WINDOW (1.3762 at w=7, i.e.
        # 1.1583x the long-clip 1.1881) because that is where a single frame lands inside the
        # largest number of *short* windows at once. A person-slot of 6-9 face-present frames --
        # a bystander crossing frame for half a second, and the exact shape this file's own
        # short-slot test uses -- was therefore getting a Laplace scale too small to deliver the
        # configured epsilon. Measured 2026-07-23: eps 3.42 at T=7 against a configured 3.00.
        # Derive it from the slot ACTUALLY being released so it is right at every length.
        # Sound under this mechanism's neighbour relation: the unit of privacy is one frame's
        # VALUE, so neighbouring inputs share the same T and Δ carries no information about the
        # data. (It would NOT be sound if adding/removing a frame were the neighbour relation.)
        # ... plus a QUANTIZATION term. Rounding to the public grid happens BEFORE the Laplace
        # draw, so it is pre-processing, not post-processing, and its immunity does not apply:
        # |q(x)-q(y)| <= |x-y| + step per coordinate. One frame can move at most min(T, win)
        # coordinates, so the grid can add up to that many steps. Caught 2026-07-23 at T=2 and
        # T=4, where the window covers the whole slot (amp exactly 1.0) yet the measured L1 was
        # 1.0323x range — the excess was exactly one step.
        amp = _window_amplification(win, len(arrp))
        q_extra = min(len(arrp), win) / float(levels - 1)
        # UNIT OF PRIVACY (config.DP_PERSON_LEVEL). Frame-level (default): one frame's value is the
        # unit, Δ is the per-frame sensitivity below. Person-level ("option b"): the unit is this
        # whole person's trajectory, so under group privacy adding/removing the person can move all
        # T_p of their released values by the full range — Δ_c(person) = T_p × Δ_c(frame) — i.e. the
        # noise scales by T_p. This is the honest ~T_p× utility cost of person-level DP.
        tp_scale = float(len(arrp)) if bool(getattr(cfg, "DP_PERSON_LEVEL", False)) else 1.0
        for c in range(NCH):
            lo, hi = float(bounds[c][0]), float(bounds[c][1])
            # (0) CLAMP THE RAW INPUT FIRST — this is what makes SENSITIVITY = the public range a
            # VALID bound. Measured 2026-07-23: clipping only AFTER the window-mean left the raw
            # value unbounded, so one frame could drive up to `win` consecutive window-means from
            # lo to hi and the true L1 sensitivity was 3.61x the range (channel 0, w=7) — i.e. the
            # Laplace scale was 3.61x too small and eps was that much larger than configured.
            # With the clamp, the worst one-frame L1 is 0.774x range (edge frame; interior 0.677x,
            # the 1.14x gap being edge-padding replication), so Delta = range is conservative and
            # the configured eps holds.
            raw = np.clip(arrp[:, c], lo, hi)                       # (0) bound the record itself
            xs = _window_mean(raw, win)                             # (1) temporal smoothing
            xs = np.clip(xs, lo, hi)                                # (2) quantize over PUBLIC range
            if hi > lo:
                step = (hi - lo) / (levels - 1)
                xs = np.round((xs - lo) / step) * step + lo
            # (3) Laplace noise. Δ_c = amp(T) x public range, with amp from THIS slot's length --
            # not config.SENSITIVITY, which is baked at one length and under-scales short slots.
            # max() keeps it a true upper bound if a config ships a hand-raised SENSITIVITY.
            sens_c = tp_scale * max((amp + q_extra) * (hi - lo), float(sens[c]) * amp / _AMP_BAKED)
            b_c = sens_c / max(eps_vec[c], 1e-9)
            xs = xs + _laplace_secure(b_c, xs.shape)
            for t, (i, j) in enumerate(pres_cells):
                out[i][j][c] = round(float(xs[t]), 5)
    return out


# ------------------------- silhouette mitigation (mask shape channel) -------------------------
def _radial_profile(p, bins):
    """Boundary radius profile r(theta) of ONE contour about its own centroid, resampled onto a
    uniform angular grid of `bins` bins by taking the MAX radius per bin.

    -> (centroid[2], r[bins]) or None if the contour is degenerate.

    The max-per-bin resample is deliberately OUTWARD-BIASED: along any ray from the centroid it
    keeps the farthest boundary point, so the profile describes the person's outer envelope and
    the ops built on it can only ever bulge. It is also what makes those ops resolution-
    independent -- the grid is an ANGLE grid, so it carries no pixel constant.
    """
    if p.shape[0] < 3:
        return None
    ctr = p.mean(0)
    v = p - ctr
    r = np.linalg.norm(v, axis=1)
    a = np.arctan2(v[:, 1], v[:, 0])
    idx = np.minimum(((a + np.pi) * (bins / (2.0 * np.pi))).astype(np.int32), bins - 1)
    rb = np.zeros(bins, np.float32)
    np.maximum.at(rb, idx, r)
    hit = rb > 0
    if not hit.any():
        return None
    if not hit.all():
        # circularly interpolate the empty bins (a contour rarely covers all of them)
        ii = np.nonzero(hit)[0].astype(np.float32)
        xs = np.concatenate([ii - bins, ii, ii + bins])
        ys = np.tile(rb[hit], 3)
        rb = np.interp(np.arange(bins, dtype=np.float32), xs, ys).astype(np.float32)
    return ctr, rb


def _radial_poly(ctr, r, bins):
    """Uniform-angular-grid radius profile -> an OpenCV polygon."""
    th = (np.arange(bins, dtype=np.float32) + 0.5) * (2.0 * np.pi / bins) - np.pi
    pts = np.stack([ctr[0] + r * np.cos(th), ctr[1] + r * np.sin(th)], 1)
    return np.round(pts).astype(np.int32).reshape(-1, 1, 2)


def _shape_polys(cnts, mode, eps_frac):
    """ONE shape-canonicalisation op: external contours -> the polygons to fill.

    Split out of mask_mitigate() on 2026-07-31 so that modes can be COMPOSED ("displace+close")
    without duplicating any of them. Every branch below is the code that used to live inline in
    mask_mitigate(), moved VERBATIM — the pre-existing modes are bit-identical (verified against a
    hash snapshot of 60 sequence/window/mode combinations taken before the move).

    §2 is NOT this function's job: whatever it returns, mask_mitigate() OR-s `sm` and `cur` back
    in, so the emitted mask is a superset of the detected one no matter what a mode does.
    """
    if mode == "hull":
        return [cv2.convexHull(c) for c in cnts]
    if mode == "bbox":
        # FULL AXIS-ALIGNED BOUNDING BOX per person component (user request 2026-08-03: "fill the
        # whole bounding box with the mask, full rectangle rather than the person's silhouette").
        #
        # This is the STRONGEST shape canonicalisation in this function: unlike hull/ellipse, which
        # still leak the body's aspect and lean, a rectangle preserves ONLY the component's position
        # and its width/height extent. Every trace of limb articulation and of the width-as-a-
        # function-of-height profile -- which is the bulk of what a silhouette-gait recogniser reads
        # -- is gone by construction. ⚠️ NOT MEASURED against the CASIA-B adversary; hull is the
        # strongest MEASURED mode (20.33 % NM, §A.6g). Do not quote a privacy number for this mode
        # until an arm is run; the arm that would settle it is the §A.6g harness with mode="bbox".
        #
        # §2 IS SAFE BY CONSTRUCTION AND STRICTLY MORE SO THAN ANY OTHER MODE: boundingRect(c) is a
        # superset of c for every contour, so the filled rect ⊇ the component before mask_mitigate's
        # `| sm | cur` is even applied. A rectangle can only ever ADD gray, never reveal.
        #
        # COST, which is the real trade here: a rectangle over an upright person is roughly the
        # inverse of the person's box-fill ratio -- order 2-2.5x the silhouette area -- and that
        # area is the cloud's inpainting bill. build_run.py MEASURES the emitted coverage rather
        # than assuming it, and writes it to the run's AREA.json.
        #
        # MERGE (MASK_BBOX_MERGE, default on): when one person's mask breaks into several blobs
        # (occlusion, a limb separated by the erode margin) per-component rects would emit several
        # overlapping rectangles instead of one. Rects that INTERSECT are merged, iterated to a
        # fixed point. Rects that do not intersect are left alone, so two genuinely separate people
        # are never fused into one box spanning the gap between them.
        rects = [list(cv2.boundingRect(c)) for c in cnts]        # x, y, w, h
        if bool(getattr(C, "MASK_BBOX_MERGE", True)) and len(rects) > 1:
            changed = True
            while changed:
                changed = False
                for i in range(len(rects)):
                    for j in range(i + 1, len(rects)):
                        ax, ay, aw, ah = rects[i]
                        bx, by, bw, bh = rects[j]
                        if ax < bx + bw and bx < ax + aw and ay < by + bh and by < ay + ah:
                            nx, ny = min(ax, bx), min(ay, by)
                            rects[i] = [nx, ny, max(ax + aw, bx + bw) - nx,
                                        max(ay + ah, by + bh) - ny]
                            rects.pop(j)
                            changed = True
                            break
                    if changed:
                        break
        # PAD as a FRACTION of the rect's own size, never absolute px -- the project rule against
        # fitted constants. 0.0 (the default) is the exact bounding box.
        pad = float(getattr(C, "MASK_BBOX_PAD_FRAC", 0.0))
        polys = []
        for x, y, w, h in rects:
            if pad > 0:
                dx, dy = int(round(w * pad)), int(round(h * pad))
                x, y, w, h = x - dx, y - dy, w + 2 * dx, h + 2 * dy
            x2, y2 = x + w - 1, y + h - 1
            polys.append(np.array([[[x, y]], [[x2, y]], [[x2, y2]], [[x, y2]]], np.int32))
        return polys
    if mode == "displace":
        # OUTWARD-ONLY DISPLACEMENT FIELD (2026-07-25). The §2-safe form of the published
        # silhouette-deformation approach (Anonymization of Human Gait in Video Based on
        # Silhouette Deformation and Texture Transfer, IEEE 2022), which reports CNN gait
        # recognition collapsing 100 % -> 1.57 % by perturbing static body shape AND walking
        # rhythm. A GENERAL displacement field moves boundary points both ways and would break
        # our "emitted ⊇ detected" guarantee, so every displacement here is >= 0 along the
        # outward normal: the boundary can only bulge, never cut in.
        #   * low-frequency angular field  -> perturbs STATIC body shape (build, width profile)
        #   * phase advances per frame     -> perturbs WALKING RHYTHM across the sequence
        #   * per-clip seed                -> never derived from identity or content, so two
        #                                     clips of the same person get different fields
        # Far cheaper in area than hull/ellipse (a few px of bulge vs a whole convex blob),
        # which is what makes it viable against the cloud's halo constraint.
        # RELATIVE amplitude: a fraction of each component's own mean radius, NOT absolute px.
        # An absolute value cannot generalise -- 6 px is ~10 % of a 64x64 CASIA-B crop and
        # negligible on a 1264^2 native mask, so a CASIA-B measurement taken with absolute px
        # would not transfer to device footage at all (and would hardcode a fitted constant,
        # which the project rule forbids). Scaling by the component radius makes the perturbation
        # the same PROPORTION of the person at every resolution and distance.
        amp_frac = float(getattr(C, "MASK_DISPLACE_AMP_FRAC", 0.10))
        nh = int(getattr(C, "MASK_DISPLACE_HARMONICS", 3))
        ph = float(getattr(C, "_MASK_DISPLACE_PHASE", 0.0))
        # PER-EPOCH RE-SEEDING (2026-07-31). The phase advance already perturbs walking RHYTHM,
        # but the field's SHAPE (coef/off) is fixed for the whole clip, so a long sequence still
        # carries one consistent deformation an attacker can average over. Re-drawing the field
        # every `MASK_DISPLACE_RESEED_PHASE` radians of phase breaks that: the deformation becomes
        # a piecewise-constant random process instead of a single per-clip transform. It costs
        # ZERO extra area (the amplitude is unchanged; only the field's direction changes), which
        # is why it is worth measuring against an area budget.
        # 0.0 = never re-seed = the SHIPPED behaviour, bit-identical (epoch is then always 0).
        _reseed = float(getattr(C, "MASK_DISPLACE_RESEED_PHASE", 0.0))
        _epoch = int(ph / _reseed) if _reseed > 0 else 0
        rng = np.random.default_rng(int(getattr(C, "_MASK_DISPLACE_SEED", 0))
                                    + 1000003 * _epoch)
        coef = rng.uniform(0.3, 1.0, nh); off = rng.uniform(0, 2 * np.pi, nh)

        # HEAD-vs-BODY AMPLITUDE SPLIT (2026-07-26). A single amplitude has to serve two very
        # different jobs: the torso/limbs carry nearly all of the silhouette's gait signal and want
        # a LARGE bulge, while the head is small, high-contrast and the thing a viewer looks at
        # first, so the same bulge there reads as a deformed skull. This lets the head run a lower
        # amplitude than the body.
        #
        # DERIVED FROM THE MASK ALONE — no pose. That is a hard constraint, not an oversight: the
        # CASIA-B silhouette adversary consumes bare binary masks, so a pose-driven head region
        # could never be evaluated against the only real gait attacker we have (see the block
        # comment above). The head band is therefore the top `MASK_DISPLACE_HEAD_FRAC` of the
        # COMPONENT'S OWN bounding-box height — a proportion of the person, so it transfers between
        # a 64x64 CASIA-B crop and a native 1264^2 mask without hardcoding any pixel constant.
        #
        # UPRIGHT GUARD: "top of the bbox = head" only holds for an upright person. For a component
        # wider than it is tall (someone lying down, a torso-only sliver at the frame edge, two
        # people merged into one blob) the assumption fails, so those fall back to the uniform
        # amplitude rather than silently perturbing whatever happens to be at the top.
        #
        # §2 IS UNAFFECTED: every per-vertex amplitude is still >= 0 and the displacement is still
        # outward-only along the normal, so the emitted mask remains a superset of the detected one.
        amp_head = getattr(C, "MASK_DISPLACE_AMP_HEAD", None)
        amp_head = None if amp_head is None else float(amp_head)
        head_frac = float(getattr(C, "MASK_DISPLACE_HEAD_FRAC", 0.15))
        head_blend = float(getattr(C, "MASK_DISPLACE_HEAD_BLEND", 0.10))
        upright_min = float(getattr(C, "MASK_DISPLACE_UPRIGHT_MIN", 1.2))

        polys = []
        for c in cnts:
            p = c.reshape(-1, 2).astype(np.float32)
            ctr = p.mean(0)
            v = p - ctr
            r = np.linalg.norm(v, axis=1) + 1e-6
            ang = np.arctan2(v[:, 1], v[:, 0])
            # sum of harmonics in [0,1] -> displacement is non-negative by construction
            d = np.zeros_like(ang)
            for k in range(nh):
                d += coef[k] * (0.5 + 0.5 * np.sin((k + 1) * ang + off[k] + ph))
            d = d / max(1e-6, coef.sum())                     # normalise to [0,1]
            rmean = float(r.mean())

            if amp_head is None:
                amp = amp_frac * rmean                # scalar: original uniform behaviour
            else:
                y = p[:, 1]; x = p[:, 0]
                y0 = float(y.min()); h = max(1e-6, float(y.max()) - y0)
                w = max(1e-6, float(x.max()) - float(x.min()))
                if h / w >= upright_min:
                    # y grows DOWNWARD, so y0 is the crown. t ramps 0 (head) -> 1 (body) across
                    # a blend zone; a hard step would leave a visible ledge at the neck.
                    t = np.clip(((y - y0) / h - head_frac) / max(1e-6, head_blend), 0.0, 1.0)
                    frac = amp_head + (amp_frac - amp_head) * t
                else:
                    frac = np.full(p.shape[0], amp_frac, dtype=np.float32)
                amp = frac * rmean                    # per-VERTEX amplitude
            newp = ctr + v / r[:, None] * (r + amp * d)[:, None]
            polys.append(np.round(newp).astype(np.int32).reshape(-1, 1, 2))
        return polys
    if mode == "ellipse":
        polys = []
        for c in cnts:
            fit = None
            if len(c) >= 5:
                # cv2.fitEllipse needs >=5 points but STILL returns NaN on degenerate (near-
                # collinear) contours — a 1-px sliver of a person at the frame edge is enough.
                # Guard on the VALUES, not just the point count: an unguarded int(NaN) killed a
                # 40-minute CASIA-B sweep at subject 34/50 on 2026-07-25.
                try:
                    _f = cv2.fitEllipse(c)
                    if all(np.isfinite(v) for v in (_f[0][0], _f[0][1], _f[1][0], _f[1][1], _f[2])):
                        fit = _f
                except cv2.error:
                    fit = None
            if fit is not None:
                (ex, ey), (ea, eb), eang = fit
                # Inflate so the blob CONTAINS the component it replaces; the union below makes
                # this a correctness backstop rather than a requirement, but a tight ellipse would
                # leave the real contour poking out and re-leak the very shape we are hiding.
                f = float(getattr(C, "MASK_ELLIPSE_INFLATE", 1.15))
                pts = cv2.ellipse2Poly((int(ex), int(ey)),
                                       (max(1, int(ea * f / 2)), max(1, int(eb * f / 2))),
                                       int(eang), 0, 360, 5)
                polys.append(pts.reshape(-1, 1, 2))
            else:
                polys.append(cv2.convexHull(c))
        return polys
    if mode == "radiallp":
        # OUTWARD-ONLY RADIAL LOW-PASS (2026-07-31). What a silhouette-gait recogniser reads is the
        # body-WIDTH profile and limb articulation — i.e. the HIGH angular frequencies of the
        # boundary radius r(theta) about the component centroid. `hull` destroys those by going
        # convex, which is why it scores 20.33 % NM (§A.6g) — but convexity is a blunt instrument
        # and the user has banned it. This mode attacks the same cue WITHOUT going convex:
        #   1. resample r(theta) onto a uniform angular grid, taking the MAX radius per bin. That
        #      alone is a "radial hull": along any ray from the centroid everything up to the
        #      farthest boundary point is filled, so inter-limb gaps close, but the shape may still
        #      be non-convex (a bent arm, a lunging stride).
        #   2. circular low-pass: keep only the DC term + the first MASK_RADIALLP_KEEP harmonics of
        #      that profile. This is what erases the width/articulation signature.
        #   3. take max(r_binned, r_smoothed) so the result only ever bulges OUTWARD.
        # RESOLUTION-INDEPENDENT BY CONSTRUCTION: the grid is angular and the retained harmonic
        # count is a pure integer — there is no pixel constant anywhere, so the op is the same
        # proportion of the person on a 64x64 CASIA-B crop and on a native 1264^2 mask.
        keep = max(0, int(getattr(C, "MASK_RADIALLP_KEEP", 4)))
        bins = max(16, int(getattr(C, "MASK_RADIALLP_BINS", 180)))
        polys = []
        for c in cnts:
            prof = _radial_profile(c.reshape(-1, 2).astype(np.float32), bins)
            if prof is None:
                polys.append(c)
                continue
            ctr, rb = prof
            F = np.fft.rfft(rb)
            F[keep + 1:] = 0                                  # circular low-pass
            rs = np.fft.irfft(F, n=bins).astype(np.float32)
            polys.append(_radial_poly(ctr, np.maximum(rb, rs), bins))   # OUTWARD-ONLY
        return polys
    if mode == "ksame":
        # k-SAME SILHOUETTE COLLAPSE (2026-07-31) — the mask analogue of what the pose anonymiser
        # already does to bone lengths with `_TEMPLATE_RATIOS`. Instead of perturbing each person's
        # own outline (which leaves their build recoverable), every silhouette is pushed OUT to a
        # shared POPULATION TEMPLATE boundary profile: the emitted radius is
        #       r_out(theta) = max( r_person(theta), s * T(theta) )
        # where T is a scale-free canonical profile (mean radius 1) and `s` scales it to this
        # person's own mean radius, so it carries no absolute size and no pixel constant. Where the
        # person is NARROWER than the template their own width profile is replaced by the shared
        # one; where they are wider they stick out (which is what keeps the area cost small and
        # keeps §2 trivially satisfied — the op is outward-only by construction).
        # The template is a POPULATION CONSTANT, exactly like the pose template. See
        # config.MASK_KSAME_TEMPLATE for its provenance and its domain caveat.
        bins = max(16, int(getattr(C, "MASK_RADIALLP_BINS", 180)))
        tmpl = np.asarray(getattr(C, "MASK_KSAME_TEMPLATE", ()), dtype=np.float32)
        if tmpl.size < 8:
            return [cv2.approxPolyDP(c, eps_frac * cv2.arcLength(c, True), True) for c in cnts]
        if tmpl.size != bins:                                 # resample the constant, circularly
            src = (np.arange(tmpl.size, dtype=np.float32) + 0.5) / tmpl.size
            dst = (np.arange(bins, dtype=np.float32) + 0.5) / bins
            tmpl = np.interp(dst, np.concatenate([src - 1, src, src + 1]),
                             np.tile(tmpl, 3)).astype(np.float32)
        tmpl = tmpl / max(1e-6, float(tmpl.mean()))           # scale-free: mean radius == 1
        scale = float(getattr(C, "MASK_KSAME_SCALE", 1.0))
        polys = []
        for c in cnts:
            prof = _radial_profile(c.reshape(-1, 2).astype(np.float32), bins)
            if prof is None:
                polys.append(c)
                continue
            ctr, rb = prof
            polys.append(_radial_poly(ctr, np.maximum(rb, tmpl * (scale * float(rb.mean()))),
                                      bins))
        return polys
    if mode == "bands":
        # RANDOMISED HORIZONTAL BANDS, PER-BAND AMPLITUDE (2026-07-31, user-specified mechanism).
        # The silhouette is cut by imaginary horizontal lines into N sections (N random in
        # MASK_BAND_N), at RANDOMLY PLACED boundaries, and each band displaces with ITS OWN
        # amplitude drawn from MASK_BAND_AMP in MASK_BAND_AMP_STEP steps. Every draw comes from
        # the per-CLIP seed (never identity/content), so each clip presents a DIFFERENT width-
        # profile distortion — aimed at the two audited findings at once:
        #   * GEI (the strongest measured attacker) reads body width as a function of height;
        #     per-band random amplitudes corrupt that profile differently per clip;
        #   * every DETERMINISTIC arm hands a gallery-adapting attacker 9-38 pp back; a band
        #     layout the attacker has never seen cannot be replicated offline (the property the
        #     pose channel's per-clip reseed already demonstrates at +1-3 pp).
        # Geometry: band boundaries are FRACTIONS of the component's own bbox height (no pixel
        # constants); the amplitude profile is blended across band edges over MASK_BAND_BLEND of
        # the height (a hard step would leave a visible ledge = inpaint hazard + fingerprint);
        # the harmonic field/phase machinery is displace's own, so walking-rhythm perturbation
        # is preserved; upright guard falls back to a UNIFORM amplitude = the mean of this
        # clip's drawn band amplitudes (stays random per clip, stays inside MASK_BAND_AMP).
        # Outward-only along the normal, so §2 holds by construction.
        nh = int(getattr(C, "MASK_DISPLACE_HARMONICS", 3))
        ph = float(getattr(C, "_MASK_DISPLACE_PHASE", 0.0))
        _reseed = float(getattr(C, "MASK_DISPLACE_RESEED_PHASE", 0.0))
        _epoch = int(ph / _reseed) if _reseed > 0 else 0
        seed = int(getattr(C, "_MASK_DISPLACE_SEED", 0))
        rng = np.random.default_rng(seed + 1000003 * _epoch)
        coef = rng.uniform(0.3, 1.0, nh); off = rng.uniform(0, 2 * np.pi, nh)
        n_lo, n_hi = (int(x_) for x_ in getattr(C, "MASK_BAND_N", (4, 7)))
        a_lo, a_hi = (float(x_) for x_ in getattr(C, "MASK_BAND_AMP", (0.10, 0.40)))
        a_step = float(getattr(C, "MASK_BAND_AMP_STEP", 0.0025))
        alpha = float(getattr(C, "MASK_BAND_ALPHA", 2.5))
        min_w = float(getattr(C, "MASK_BAND_MIN_W", 0.10))
        blendf = float(getattr(C, "MASK_BAND_BLEND_FRAC", 0.25))
        upright_min = float(getattr(C, "MASK_DISPLACE_UPRIGHT_MIN", 1.2))
        GRID = 512                                       # band-profile sample grid over height
        polys = []
        for ci_, c in enumerate(cnts):
            p = c.reshape(-1, 2).astype(np.float32)
            if p.shape[0] < 8:
                polys.append(c)
                continue
            ctr = p.mean(0)
            v = p - ctr
            r = np.linalg.norm(v, axis=1) + 1e-6
            ang = np.arctan2(v[:, 1], v[:, 0])
            d = np.zeros_like(ang)
            for k in range(nh):
                d += coef[k] * (0.5 + 0.5 * np.sin((k + 1) * ang + off[k] + ph))
            d = d / max(1e-6, coef.sum())
            rmean = float(r.mean())
            # per-CLIP, per-COMPONENT draws — stable across frames by construction.
            # Count uniform{4..7}; widths Dirichlet(alpha=2.5) (random fractions that sum to 1
            # = bands always tile the body), RESAMPLED until every band >= MASK_BAND_MIN_W of
            # the height (a sliver band cannot carry an amplitude difference — it just becomes
            # a ledge); amplitudes on the a_step grid within MASK_BAND_AMP.
            rb = np.random.default_rng(seed + 424243 * (ci_ + 1))
            nb = int(rb.integers(n_lo, n_hi + 1))
            widths = None
            for _try in range(500):
                cand = rb.dirichlet(np.full(nb, alpha))
                if float(cand.min()) >= min_w:
                    widths = cand
                    break
            if widths is None:                            # pathological draw streak: equal split
                widths = np.full(nb, 1.0 / nb)
            steps = int(round((a_hi - a_lo) / max(1e-9, a_step)))
            amps = a_lo + a_step * rb.integers(0, steps + 1, nb)
            cuts = np.cumsum(widths)[:-1]
            y = p[:, 1]; x = p[:, 0]
            y0 = float(y.min()); h = max(1e-6, float(y.max()) - y0)
            w = max(1e-6, float(x.max()) - float(x.min()))
            if h / w >= upright_min:
                tg = (np.arange(GRID, dtype=np.float32) + 0.5) / GRID
                prof = amps[np.searchsorted(cuts, tg)].astype(np.float32)
                # blend each edge over 25 % of the NARROWER neighbour: linear lerp on the grid.
                # Blend half-widths are <= 12.5 % of each band's own width, so zones never
                # overlap and every band keeps a flat plateau at its drawn amplitude.
                for e in range(nb - 1):
                    b = blendf * float(min(widths[e], widths[e + 1]))
                    lo, hi = float(cuts[e]) - b / 2, float(cuts[e]) + b / 2
                    m_ = (tg >= lo) & (tg <= hi)
                    if m_.any():
                        prof[m_] = amps[e] + (amps[e + 1] - amps[e]) * \
                            ((tg[m_] - lo) / max(1e-9, hi - lo))
                frac = np.interp((y - y0) / h, tg, prof).astype(np.float32)
            else:
                # upright guard: uniform fallback at the MEAN of this clip's drawn amplitudes
                # (stays per-clip random, stays inside MASK_BAND_AMP)
                frac = np.full(p.shape[0], float(amps.mean()), np.float32)
            amp = frac * rmean
            newp = ctr + v / r[:, None] * (r + amp * d)[:, None]
            polys.append(np.round(newp).astype(np.int32).reshape(-1, 1, 2))
        return polys
    if mode == "dirbands":
        # RANDOM DIRECTIONAL BANDS (2026-07-31, user proposal 17:17; ledger §A.6k-8).
        # Identical band machinery to "bands" above, with ONE change: the band coordinate is
        # not height, it is the projection onto a PER-CLIP RANDOM AXIS
        #     u = x*sin(theta) + y*cos(theta),  theta ~ U(0, pi)
        # normalised over the component's own extent along that axis.
        #
        # WHY. The silhouette adversary reads the body as 16 HORIZONTAL strips
        # (PART_DISCRIM_OURS.json). Horizontal bands align with those strips, so each strip
        # receives ONE consistent amplitude -- its width is shifted, but cleanly, and a clean
        # offset is exactly what a recogniser normalises away. Bands at an angle CUT ACROSS
        # strips, so every strip gets a MIXTURE of amplitudes and its width estimate is
        # BLURRED rather than displaced. theta is drawn per clip, so a gallery-adapting
        # attacker cannot even assume the banding axis.
        #
        # MEASURED (108 clips, 107 probes, GEI vs a DEFENDED gallery, paired McNemar):
        #   56.07 % -> 30.84 % top-1, -25.23 pp, p < 0.0001, area 1.302x.
        # That is the best arm in the whole sweep that fits the strict 1.358x area ceiling,
        # at an area BELOW the shipped mask's own 1.337x, and at -83.5 pp per unit excess
        # area it is 1.35x more efficient than the next best arm measured.
        #
        # NOTE it carries NO time term in its angular profile. Per §A.6k-5 that is not
        # incidental: GEI is the time-average, so a perturbation that ROTATES averages into a
        # uniform offset the attacker normalises away, while one held FIXED within the clip
        # survives the average. Do NOT add a phase-step term here.
        #
        # Outward-only along the radius, so §2 holds by construction.
        nh = int(getattr(C, "MASK_DISPLACE_HARMONICS", 3))
        seed = int(getattr(C, "_MASK_DISPLACE_SEED", 0))
        rng = np.random.default_rng(seed)
        coef = rng.uniform(0.3, 1.0, nh); off = rng.uniform(0, 2 * np.pi, nh)
        n_lo, n_hi = (int(x_) for x_ in getattr(C, "MASK_DIRBAND_N", getattr(C, "MASK_BAND_N", (4, 7))))
        a_lo, a_hi = (float(x_) for x_ in getattr(C, "MASK_DIRBAND_AMP", (0.10, 0.30)))
        alpha = float(getattr(C, "MASK_BAND_ALPHA", 2.5))
        min_w = float(getattr(C, "MASK_BAND_MIN_W", 0.10))
        # UPRIGHT GUARD -- OFF BY DEFAULT FOR THIS MODE, and that is deliberate.
        # The guard exists for "bands", where the band coordinate is HEIGHT: on a wide/short
        # component (person lying down, a fragment) height-banding is meaningless, so it falls
        # back to a uniform swell. `dirbands` bands along a RANDOM AXIS, so there is no
        # privileged orientation to be wrong about and the guard has no rationale here.
        # It was inherited by copying the "bands" branch, and it cost a great deal:
        # censused over 1341 components of our corpus, h/w < 1.2 fires on 45.6 % of them
        # (median h/w is 1.22 -- the threshold sits right on the mode of the distribution).
        # Nearly half of all components were therefore getting the weak uniform fallback on
        # the shipped path while the lab implementation banded every one, which is what
        # produced the systematic 5-17 pp lab-vs-shipped offset in A.6k-13.
        # Set MASK_DIRBAND_UPRIGHT_MIN > 0 to re-enable it.
        upright_min = float(getattr(C, "MASK_DIRBAND_UPRIGHT_MIN", 0.0))
        theta_env = getattr(C, "MASK_DIRBAND_THETA", None)      # None => random per clip
        # DENSIFY FIRST. mask_mitigate simplifies the contour with approxPolyDP at
        # eps_frac * perimeter BEFORE any shape op runs: measured on a real 1264^2 clip that
        # takes 1220 boundary points down to 453 (CHAIN_APPROX_SIMPLE) and then to TWELVE.
        # You cannot place 4-11 random directional bands, each with its own angular phase, on
        # a 12-gon -- the mechanism is crushed before it executes. That is exactly what the
        # port-validation gate caught: 46.73 % (p=0.0525, not significant) against the lab
        # implementation's 30.84 %, with emitted area 1.183x instead of 1.303x.
        #
        # This mode's value is the structure it ADDS, not detail it preserves, so resampling
        # the simplified polygon back up to a workable number of points recovers the mechanism
        # without touching eps_frac -- which is itself a privacy mitigation and must not be
        # weakened to make this mode work.
        def _densify(poly, target=512):
            q = poly.reshape(-1, 2).astype(np.float32)
            if q.shape[0] >= target:
                return q
            closed = np.vstack([q, q[:1]])
            seg = np.linalg.norm(np.diff(closed, axis=0), axis=1)
            total = float(seg.sum())
            if total <= 1e-6:
                return q
            cum = np.concatenate([[0.0], np.cumsum(seg)])
            t = np.linspace(0.0, total, target, endpoint=False)
            xs = np.interp(t, cum, closed[:, 0])
            ys = np.interp(t, cum, closed[:, 1])
            return np.stack([xs, ys], 1).astype(np.float32)

        polys = []
        for ci_, c in enumerate(cnts):
            p = _densify(c)
            if p.shape[0] < 8:
                polys.append(c)
                continue
            ctr = p.mean(0)
            v = p - ctr
            r = np.linalg.norm(v, axis=1) + 1e-6
            ang = np.arctan2(v[:, 1], v[:, 0])
            rmean = float(r.mean())
            rb = np.random.default_rng(seed + 424243 * (ci_ + 1))
            th = float(rb.uniform(0, np.pi)) if theta_env is None else float(theta_env)
            ct_, st_ = np.cos(th), np.sin(th)
            u = p[:, 0] * st_ + p[:, 1] * ct_                   # projection onto the band axis
            u0 = float(u.min()); span = max(1e-6, float(u.max()) - u0)
            h = max(1e-6, float(p[:, 1].max() - p[:, 1].min()))
            w = max(1e-6, float(p[:, 0].max() - p[:, 0].min()))
            # ANGULAR PROFILE -- PER BAND, NOT GLOBAL. This is the whole mechanism and the
            # first port of it got this wrong: mirroring the "bands" mode above, it computed
            # ONE global multi-harmonic field and merely SCALED it per band, so every band
            # bulged at the same angles. Measured, that port scored 52.34 % (-3.74 pp,
            # p = 0.4807, not significant) against the lab implementation's 30.84 % -- a
            # 21.5 pp miss, i.e. it did essentially nothing.
            #
            # What actually works: each band draws its OWN phase, so band k bulges left while
            # band k+1 bulges right. Combined with the random band AXIS, that is a per-region
            # DIRECTIONAL displacement rather than a globally coherent lobe pattern, and it is
            # what blurs the strip-wise width profile the recogniser reads. Do not "unify"
            # this with the bands mode's shared field.
            if h / w < upright_min:                             # upright guard: uniform fallback
                a_mean = 0.5 * (a_lo + a_hi)
                amp = a_mean * 0.5 * (1.0 + np.sin(nh * ang + float(rb.uniform(0, 2 * np.pi))))
            else:
                nb = int(rb.integers(n_lo, n_hi + 1))
                widths = None
                for _try in range(500):
                    cand = rb.dirichlet(np.full(nb, alpha))
                    if float(cand.min()) >= min_w:
                        widths = cand
                        break
                if widths is None:
                    widths = np.full(nb, 1.0 / nb)
                edges = np.concatenate([[0.0], np.cumsum(widths)])
                amps = rb.uniform(a_lo, a_hi, nb)
                phase = rb.uniform(0, 2 * np.pi, nb)            # PER-BAND phase
                un = (u - u0) / span
                amp = np.zeros(p.shape[0], np.float32)
                for i in range(nb):
                    sel = (un >= edges[i]) & (un <= edges[i + 1])
                    if not sel.any():
                        continue
                    amp[sel] = amps[i] * 0.5 * (1.0 + np.sin(nh * ang[sel] + phase[i]))
                k_ = max(1, int(p.shape[0] * 0.01))             # smooth across band seams
                amp = np.convolve(np.r_[amp[-k_:], amp, amp[:k_]],
                                  np.ones(2 * k_ + 1) / (2 * k_ + 1), "same")[k_:-k_][:p.shape[0]]
            newp = ctr + v / r[:, None] * (r + amp * rmean)[:, None]
            polys.append(np.round(newp).astype(np.int32).reshape(-1, 1, 2))
        return polys
    if mode == "mounds":
        # TRAVELLING RAISED-COSINE MOUNDS ON THE RADIUS PROFILE (2026-07-31). Supersedes the
        # rejected "blobs" discs (user: mounds must be "semi-circles or even less ... like goo
        # wiggling on the body", not balls). A disc half-out of the outline decorates the edge
        # with concave notches (perimeter x1.124) and barely changes body WIDTH — the quantity a
        # silhouette recogniser reads. A raised-cosine swelling of r(theta) changes width
        # smoothly, adds no notches (perimeter x1.031 at the reference params) and is far
        # cheaper in area (x1.077 vs x1.283 measured on a real clip at native 1264^2):
        #     r'(th) = r(th) + height*rmean * sum_k amp_k * 1/2(1+cos(pi*dth_k/width_rad))
        #     for |dth_k| < width_rad,   dth_k = wrapped(th - centre_k(t))
        #     centre_k(t) = phase_k + speed_k * t          # slow drift = the "goo" wiggle
        # phase_k / amp_k / speed_k come from the per-CLIP seed (never identity/content);
        # per-component seed offset so two people get different goo. Outward-only (the bump is
        # >= 0) so §2 holds by construction. NO PIXEL CONSTANTS: height is a fraction of the
        # component's mean radius, width is an ANGLE, drift is rad/frame (⚠️ fps-coupled like
        # MASK_TEMPORAL_WIN — pin via env if EMIT_FPS changes).
        n_m = max(0, int(getattr(C, "MASK_MOUND_N", 4)))
        hgt = float(getattr(C, "MASK_MOUND_HEIGHT", 0.13))
        wid = max(1e-3, float(getattr(C, "MASK_MOUND_WIDTH_RAD", 0.55)))
        drift = float(getattr(C, "MASK_MOUND_DRIFT", 0.020))
        ph = float(getattr(C, "_MASK_DISPLACE_PHASE", 0.0))
        _st = float(getattr(C, "MASK_DISPLACE_PHASE_STEP", 0.35))
        tidx = int(round(ph / _st)) if _st > 0 else 0
        seed = int(getattr(C, "_MASK_DISPLACE_SEED", 0))
        polys = []
        for ci_, c in enumerate(cnts):
            p = c.reshape(-1, 2).astype(np.float32)
            if p.shape[0] < 8 or n_m == 0:
                polys.append(c)
                continue
            ctr = p.mean(0)
            v = p - ctr
            r = np.linalg.norm(v, axis=1) + 1e-6
            ang = np.arctan2(v[:, 1], v[:, 0])
            rmean = float(r.mean())
            rng = np.random.default_rng(seed + 104729 * (ci_ + 1))
            phase = rng.uniform(0.0, 2.0 * np.pi, n_m)
            amp = rng.uniform(0.5, 1.0, n_m)
            # speed: magnitude in [0.5,1]*drift with random sign — avoids accidental statics
            # while keeping `drift` the meaningful scale; drift=0 => static mounds (control).
            speed = drift * rng.uniform(0.5, 1.0, n_m) * rng.choice((-1.0, 1.0), n_m)
            bump = np.zeros_like(ang)
            for k in range(n_m):
                ck = phase[k] + speed[k] * tidx
                dth = np.angle(np.exp(1j * (ang - ck)))
                m_ = np.abs(dth) < wid
                if m_.any():
                    bump[m_] += amp[k] * 0.5 * (1.0 + np.cos(np.pi * dth[m_] / wid))
            newp = ctr + v / r[:, None] * (r + hgt * rmean * bump)[:, None]
            polys.append(np.round(newp).astype(np.int32).reshape(-1, 1, 2))
        return polys
    if mode == "close":
        # MORPHOLOGICAL CLOSE with a component-relative kernel (2026-07-31). Fills the gaps a gait
        # recogniser reads most directly — between the legs, between arm and torso — without
        # touching the outer envelope the way hull does. The kernel is sized as a fraction of the
        # component's OWN equivalent radius sqrt(area/pi), so it is the same proportion of the
        # person at any resolution or camera distance (project rule: no fitted pixel constants).
        # Each component is closed in its own padded bbox buffer, so two people standing close
        # together can never be merged into one blob by this op.
        frac = float(getattr(C, "MASK_CLOSE_KERNEL_FRAC", 0.25))
        polys = []
        for c in cnts:
            x, y, w, h = cv2.boundingRect(c)
            rad = float(np.sqrt(max(1.0, float(cv2.contourArea(c))) / np.pi))
            half = max(1, int(round(frac * rad)))
            k = 2 * half + 1
            pad = k
            sub = np.zeros((h + 2 * pad, w + 2 * pad), np.uint8)
            cv2.drawContours(sub, [c - np.array([[[x - pad, y - pad]]], np.int32)], -1, 1, -1)
            ker = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (k, k))
            sub = cv2.morphologyEx(sub, cv2.MORPH_CLOSE, ker)
            cs2, _ = cv2.findContours(sub, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            if not cs2:
                polys.append(c)
                continue
            for cc in cs2:
                polys.append(cc + np.array([[[x - pad, y - pad]]], np.int32))
        return polys
    return [cv2.approxPolyDP(c, eps_frac * cv2.arcLength(c, True), True) for c in cnts]


def mask_mitigate(hist, cur, eps_frac):
    """Information-REDUCTION on the SHAPE of the emitted silhouette (mask.mkv +
    masked_video.mkv): the per-frame contour dynamics are a soft biometric
    (silhouette/gait shape). Two inclusion-biased ops on the DILATED mask:
      (1) temporal running-MAX over `hist` (the last MASK_TEMPORAL_WIN dilated
          masks) — union, so coverage only grows; smears boundary dynamics;
      (2) contour simplification: cv2.approxPolyDP at eps_frac of each external
          contour's perimeter (also fills interior holes via RETR_EXTERNAL).
    HARD §2 GUARANTEE: the result is OR-ed with BOTH the temporal union `sm` and the
    CURRENT frame's mask, so the emitted mask ⊇ sm ⊇ cur on EVERY frame — mitigation
    can only ADD gray, never reveal, and it is MONOTONE in its input (a larger input
    union can never produce a smaller output).

    ⚠️ WHY `| sm` (added 2026-07-23). The docstring used to claim `| cur` alone made
    coverage "only grow". It did not. `approxPolyDP` at eps = eps_frac·perimeter
    RETRACTS from `sm` at concave corners — and eps SCALES WITH PERIMETER, so when the
    native-res tiled-detection pass enlarged the union (longer contour), the polygon
    cut DEEPER and the emitted mask lost real person pixels the temporal smear had
    covered. Measured as a regression against the pre-tiling baseline (ledger §A.1g).
    `simp` is not a superset of `sm`, so OR-ing `sm` back in restores those bands. The
    area added back is **0.3–0.6 % of the mask per frame** (mean; up to ~5 % on
    high-motion frames — measured; an earlier ~0.09 % estimate was wrong). It is a pure
    ADDITION of temporal-union pixels, so the fix is provably monotone (fixed output ⊇
    old output on 1232/1232 frames) and can only make the silhouette perturbation
    STRONGER, never weaker — §A.6c shows more temporal-union area LOWERS re-ID. Hole-fill
    and running-max are preserved. NOTE: §A.6b's 62.27 % NM was measured on the OLD
    function; re-measure before re-quoting it against the fixed code.
    NOTE: information-reduction mitigation only — NOT a validated silhouette-reID
    defense (unlike the pose anonymizer, this has no lab adversary eval)."""
    sm = np.max(np.stack(hist, 0), 0) if len(hist) > 1 else cur
    cnts, _ = cv2.findContours(sm, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not cnts:
        return sm | cur                                # still ⊇ the temporal union
    simp = np.zeros_like(cur)
    mode = str(getattr(C, "MASK_SHAPE_MODE", "none")).lower() if 'C' in globals() else "none"
    # SHAPE CANONICALISATION (2026-07-25, config.MASK_SHAPE_MODE). The contour simplification
    # above only nibbles at the outline; it leaves limb articulation and the body-width profile
    # intact, which is most of what a silhouette-gait recogniser actually reads (§A.6d: the
    # shipped setting still scores 79.25 % NM = 39.6x chance). The modes replace the PERSON'S OWN
    # SHAPE with a less identifying one at the same position and scale -- see _shape_polys() for
    # each ("hull", "ellipse", "displace", and 2026-07-31's "radiallp" / "close"). All are
    # MASK-ONLY (no pose), which is deliberate: the CASIA-B silhouette adversary consumes bare
    # binary masks, so a pose-driven canonical body could not be evaluated against it at all.
    # §2 is untouched -- the result is still OR-ed with `sm` and `cur` below, so it can only ADD.
    #
    # COMPOSITION (2026-07-31): a "+"-joined mode such as "displace+close" runs the ops in order.
    # Each intermediate stage is rasterised and OR-ed with the temporal union `sm` before its
    # contours are handed to the next op, so every stage is itself a superset of `sm` and the
    # pipeline stays monotone end to end.
    ops = [m for m in mode.split("+") if m]
    if len(ops) > 1:
        work = cnts
        _seed0 = int(getattr(C, "_MASK_DISPLACE_SEED", 0))
        for _i, _op in enumerate(ops):
            # STAGE SALT. Every band-family op derives its per-component RNG from
            # _MASK_DISPLACE_SEED, with no notion of which stage it is. Without a salt,
            # "dirbands+dirbands" draws the SAME axis, layout and amplitudes twice and
            # degenerates into amplitude-doubling of one pattern instead of crossing two;
            # "bands+dirbands" gets correlated layouts. No arm measured so far stacks two
            # band-family stages, so nothing already recorded is affected -- but the first
            # crossed-band config anyone writes would silently underperform.
            C._MASK_DISPLACE_SEED = _seed0 + 7919 * _i
            try:
                polys = _shape_polys(work, _op, eps_frac)
            finally:
                C._MASK_DISPLACE_SEED = _seed0
            if _i == len(ops) - 1:
                break
            _tmp = np.zeros_like(cur)
            cv2.fillPoly(_tmp, polys, 1)
            _tmp |= sm
            work, _ = cv2.findContours(_tmp, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            if not work:
                break
    else:
        polys = _shape_polys(cnts, mode, eps_frac)
    # ---- COMPACT: pull the perturbed outline back toward its own centre ----------------
    # Every shape op above pushes the outline OUTWARD, and the area it adds is the cloud's
    # inpainting bill. This pulls the perturbed polygon back toward its centroid by a factor
    # c <= 1 before it is filled.
    #
    # WHY THIS IS SAFE. The return below OR-s with `sm` and `cur`, so wherever the contraction
    # would cut into the real person the union puts those pixels straight back. S2 therefore
    # still holds by construction, exactly as it does without this step. The contraction can
    # only ever remove ADDED area, never true-person area.
    #
    # WHY IT IS NOT THE SAME AS JUST LOWERING THE AMPLITUDE. Lowering amplitude shrinks the
    # angular VARIATION as well as the size, and A.6k-1 measured that variation - not
    # magnitude - is what buys privacy (raising amplitude made an arm WORSE). A uniform
    # contraction keeps the variation pattern intact and only reduces the overall size, so it
    # should trade area for privacy on better terms. NOT MEASURED YET.
    #
    # Two modes:
    #   MASK_COMPACT_SCALE  - fixed factor, e.g. 0.90.
    #   MASK_COMPACT_TARGET - target emitted-area ratio; c is solved per frame by bisection so
    #                         the frame lands ON the budget instead of averaging there. The
    #                         frontier is convex (A.6k-4/-6, p ~ 1.5-1.8, increasing returns),
    #                         so spending every frame's full allowance beats spending the mean.
    # TARGET wins if both are set. Costs <= _COMPACT_ITERS extra fill+OR per frame.
    _ctgt = getattr(C, "MASK_COMPACT_TARGET", None)
    _csc = float(getattr(C, "MASK_COMPACT_SCALE", 1.0))
    if (_ctgt is not None or _csc < 1.0) and polys:
        def _shrink(ps, c):
            out = []
            for q in ps:
                a = q.reshape(-1, 2).astype(np.float32)
                ctr = a.mean(0)
                out.append(np.round(ctr + (a - ctr) * c).astype(np.int32).reshape(-1, 1, 2))
            return out

        def _emit(ps):
            t = np.zeros_like(cur)
            cv2.fillPoly(t, ps, 1)
            return t | sm | cur

        base_area = float(cur.sum()) or 1.0
        if _ctgt is not None:
            lo, hi = 0.30, 1.0                       # c below 0.30 is pointless: the union dominates
            if float(_emit(polys).sum()) / base_area > float(_ctgt):
                for _ in range(int(getattr(C, "MASK_COMPACT_ITERS", 6))):
                    mid = 0.5 * (lo + hi)
                    if float(_emit(_shrink(polys, mid)).sum()) / base_area > float(_ctgt):
                        hi = mid
                    else:
                        lo = mid
                polys = _shrink(polys, 0.5 * (lo + hi))
        else:
            polys = _shrink(polys, _csc)
    cv2.fillPoly(simp, polys, 1)
    return simp | sm | cur                             # ⊇ sm ⊇ cur: monotone, never retracts


# ------------------------- per-stage timing + peak RSS (--stats) -------------------------
# Tail latency, not the mean, decides the >=15 fps Tier-1 claim (V9_BUILD_SPEC P7.8), so the
# runner reports p50 AND p95 per stage plus peak resident memory. Disabled by default and a
# no-op when off, so the shipping path pays nothing.
class StageTimer:
    """Accumulate per-stage wall-clock. `with T('detect'): ...`. NOT re-entrant (never nest)."""
    def __init__(self, enabled=False):
        self.enabled = bool(enabled)
        self.ms = {}
        self._name = None
        self._t0 = 0.0

    def __call__(self, name):
        self._name = name
        return self

    def __enter__(self):
        if self.enabled:
            self._t0 = time.perf_counter()
        return self

    def __exit__(self, *exc):
        if self.enabled:
            self.ms.setdefault(self._name, []).append((time.perf_counter() - self._t0) * 1000.0)
        return False

    def summary(self):
        out = {}
        for k, v in self.ms.items():
            a = np.asarray(v, np.float64)
            out[k] = {"n": int(a.size), "mean_ms": float(a.mean()),
                      "p50_ms": float(np.percentile(a, 50)),
                      "p95_ms": float(np.percentile(a, 95)),
                      "max_ms": float(a.max()), "total_ms": float(a.sum())}
        return out


def peak_rss_mb():
    """Peak resident set size in MiB, or None if unavailable. psutil where present;
    else /proc/self/status VmHWM (Linux/Pi5) -- both report the PEAK, not the current."""
    try:
        import psutil                                  # optional, dev/laptop convenience
        p = psutil.Process()
        mi = p.memory_info()
        return float(getattr(mi, "peak_wset", getattr(mi, "rss", 0))) / (1024.0 * 1024.0)
    except Exception:
        pass
    try:
        with open("/proc/self/status") as f:
            for line in f:
                if line.startswith("VmHWM:"):
                    return float(line.split()[1]) / 1024.0
    except Exception:
        pass
    return None


# ------------------------- model factories (flag-guarded swaps) -------------------------
# Defaults reproduce the working onnx YOLO11n + RVM path EXACTLY. The alternatives are wired but
# only constructed when the corresponding config flag is flipped (measure-then-adopt).
def build_detector(cfg):
    backend = getattr(cfg, "DETECT_BACKEND", "onnx")
    if backend == "ncnn":
        return YOLO11nNCNN(cfg.YOLO11N_NCNN_PARAM, cfg.YOLO11N_NCNN_BIN)   # READY-TO-ENABLE
    return YOLO11n(cfg.YOLO11N, cfg.ORT_PROVIDERS)                          # DEFAULT (working)


def build_matte(cfg):
    model = getattr(cfg, "MATTE_MODEL", "rvm")
    if model == "pphumanseg":
        return PPHumanSeg(cfg.PPHUMANSEG_ONNX, cfg.ORT_PROVIDERS)           # READY-TO-ENABLE
    return RVM(cfg.RVM_ONNX, cfg.ORT_PROVIDERS)                             # DEFAULT (working)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--source", default=None, help="video path or camera index (default config.INPUT_VIDEO)")
    ap.add_argument("--max-frames", type=int, default=0)
    ap.add_argument("--stats", action="store_true",
                    help="report per-stage p50/p95 latency, loop FPS and peak RSS")
    ap.add_argument("--stats-json", default=None, help="also write the --stats report to this path")
    a = ap.parse_args()
    T = StageTimer(a.stats or bool(a.stats_json))
    t_load0 = time.perf_counter()

    src = a.source if a.source is not None else C.INPUT_VIDEO
    cap = cv2.VideoCapture(int(src) if str(src).isdigit() else src)
    os.makedirs(C.OUT_DIR, exist_ok=True)

    # ---- emit-fps decimation: keep 1 of every `stride` source frames so we emit ~EMIT_FPS ----
    emit_fps = float(getattr(C, "EMIT_FPS", 15))
    src_fps = cap.get(cv2.CAP_PROP_FPS)
    if src_fps and src_fps > 0:
        stride = max(1, int(round(src_fps / emit_fps)))   # 30 -> 15 gives stride 2
    else:
        stride = 1                                        # unknown fps (live camera): emit every frame
    # TRUE output fps = what we ACTUALLY emit, which is src/stride — NOT the EMIT_FPS target.
    # They differ whenever the source is SLOWER than EMIT_FPS: stride = round(10/15) = 0 -> 1, so no
    # frame is dropped, yet stamping EMIT_FPS=15 on 10 fps content plays it back 1.5x too fast and
    # every downstream consumer inherits the wrong timebase. Measured 2026-07-23: a 10 fps source
    # produced mask.mkv/masked_video.mkv labelled 15 fps with the same 100 frames.
    out_fps = (src_fps / stride) if (src_fps and src_fps > 0) else emit_fps
    work_res = int(getattr(C, "WORK_RES", 640))
    out_res = getattr(C, "OUT_RES", None)                 # None => NATIVE source size (do NOT force 640)

    # PRIVACY / MUST-FIX: emit the mask AND the masked video LOSSLESS. With a lossy codec, DCT
    # ringing re-expands the decoded binary mask past the erode margin, so Tier-2's 2A would keep a
    # ring of REAL-BACKGROUND pixels in the person region it ships to the cloud — a §2 leak.
    # FFV1 is a mathematically lossless intra codec (ffmpeg-backed OpenCV) in a Matroska (.mkv) container.
    # Writers run at EMIT_FPS and at the NATIVE (or OUT_RES) frame size — created lazily on the first
    # emitted frame so the writer size exactly matches the native capture (cap props can lie).
    # TODO(on-device verify): confirm this OpenCV/ffmpeg build actually ships the FFV1 encoder
    #   (VideoWriter.isOpened() must be True after construction). If a device build lacks FFV1, fall
    #   back to a LOSSLESS PNG frame-sequence (cv2.imwrite per frame) — never fall back to a lossy mp4.
    fourcc = cv2.VideoWriter_fourcc(*"FFV1")
    masked_w = mask_w = None

    det = build_detector(C)                               # DEFAULT: onnx YOLO11n (flag-guarded, §swaps)
    matte = build_matte(C); matte.reset()                 # DEFAULT: RVM (flag-guarded, §swaps)
    seg = build_seg(C)                                     # OPTIONAL person-instance seg (default None)
    faceguard = build_faceguard(C)                         # DECISION.md FaceGuard (default ON; None if off)
    pose = RTMPose(onnx_model=C.POSE_CKPT, model_input_size=(192, 256),
                   backend="onnxruntime", device="cpu") if RTMPose else None
    model_load_s = time.perf_counter() - t_load0      # excluded from the loop FPS below

    # AUTO PERSON COUNT + STABLE SLOTS (person_slots.py — its docstring is the policy):
    #   * det_counts collects the UNTRUNCATED per-frame detection count; the clip's person count is
    #     estimated from it at emit time (never hardcoded, never a per-clip constant).
    #   * slots assigns each frame's detections to stable left-to-right slot ids with swap
    #     hysteresis; slot_log records the id of every emitted person cell so the whole-clip
    #     per-person stages (pose anon, face DP) group by real person instead of by list position.
    slots = PS.SlotTracker(hysteresis_frames=int(getattr(C, "SLOT_HYSTERESIS_FRAMES", 5)),
                           swap_margin_frac=float(getattr(C, "SLOT_SWAP_MARGIN_FRAC", 0.10)),
                           # pass these explicitly: they were SlotTracker defaults that the call
                           # site never set, so two identity-tracking knobs were invisible to
                           # anyone tuning config.py (and track_ttl is an fps-coupled duration)
                           track_gate_frac=float(getattr(C, "TRACK_GATE_FRAC", 0.15)),
                           track_ttl=int(getattr(C, "TRACK_TTL", 8)))
    det_counts = []; slot_log = []; track_log = []
    pose_log, face_log = [], []; prev_boxes = []; prev_boxes_age = 0
    # Background-person scope accounting (config.MASK_IGNORE_BACKGROUND). Counted, never silent:
    # this is the only stage that REMOVES person pixels, so the totals ship in pose.json.
    ignored_px_total = 0; ignored_blob_total = 0
    # SUBJECT LOCK state (config.SUBJECT_LOCK): persistent track ids enrolled during the opening
    # window, and every track id refused afterwards. Both ship in pose.json.
    enrolled = set(); dropped_tracks = set(); reenroll_events = 0
    # [emitted_frame, track_id, x1, y1, x2, y2] for every REFUSED person, so the auditor
    # can attribute a reveal to the exclusion policy rather than to a coverage failure.
    refused_boxes_log = []
    # track -> [first_frame_seen, last_frame_seen, frames_with_a_box]. See the comment
    # at the append site: a refused person is only logged on frames the DETECTOR found
    # them, so boxes alone understate presence and cannot be read as an exposure count.
    refused_spans = {}
    # Enrollment candidates {track: [max_box_area, n_window_frames]}, committed once.
    enroll_stats = {}; lock_committed = False
    # PER-CLIP displacement seed from the OS CSPRNG. Never derived from identity or content: a
    # stable field would be a learnable transform and would re-create a linkable pseudo-identity
    # across clips, the same trap pose_anon_edge.new_clip_seed() exists to avoid.
    # 🔴 TEST-ONLY OVERRIDE, same guard as pose_anon_edge.new_clip_seed(). Pinning this makes the
    # mask perturbation deterministic, which is the same linkable-pseudo-identity failure the
    # comment above warns about — it exists solely so a controlled A/B can vary ONE thing.
    # Ledger §B.51: the A20-vs-shipped render differed in mask mode AND mask seed AND gait seed,
    # so its invented-person finding could not be attributed. Never set in a shipped run.
    _pinned = PA.test_fixed_seed() if hasattr(PA, "test_fixed_seed") else None
    if _pinned is not None:
        sys.stderr.write("🔴 MIRAGE_TEST_FIXED_SEED IS SET — mask perturbation is DETERMINISTIC. "
                         "This output is a TEST ARTIFACT and is NOT privacy-safe to ship.\n")
        C._MASK_DISPLACE_SEED = _pinned ^ 0x5EED  # decorrelate from the gait seed, still fixed
    else:
        C._MASK_DISPLACE_SEED = secrets.randbits(31)
    mask_hist = []; native_wh = None                  # silhouette-mitigation window + native frame size
    # SYNTHETIC BODY (config.MASK_SYNTH_BODY): per-clip state (quantised size + coverage alpha).
    synth = None
    if getattr(C, "MASK_SYNTH_BODY", False):
        import synth_body as SB
        synth = SB.SynthBody(scale_quant=float(getattr(C, "SYNTH_SCALE_QUANT", 0.10)),
                             alpha=float(getattr(C, "SYNTH_ALPHA", 1.0)),
                             grow_step=float(getattr(C, "SYNTH_GROW_STEP", 0.04)),
                             grow_max=float(getattr(C, "SYNTH_GROW_MAX", 0.60)),
                             grow_tol=float(getattr(C, "SYNTH_GROW_TOL", 0.002)),
                             thr=float(getattr(C, "SYNTH_POSE_THRESH", 0.35)),
                             backstop=bool(getattr(C, "SYNTH_BACKSTOP", True)),
                             debug_dir=getattr(C, "SYNTH_DEBUG_DIR", None),
                             # the per-CLIP build wobble rides the SAME seed as the mask
                             # perturbation, salted so the two draws are independent. Never
                             # identity- or content-derived; pinned only by MIRAGE_TEST_FIXED_SEED.
                             seed=int(getattr(C, "_MASK_DISPLACE_SEED", 0)) ^ 0x5B0D,
                             jitter=float(getattr(C, "SYNTH_JITTER", 0.06)),
                             head_mode=str(getattr(C, "SYNTH_HEAD_MODE", "hair")),
                             work_res=int(getattr(C, "SYNTH_WORK_RES", 512)),
                             grow_win=int(getattr(C, "SYNTH_GROW_WIN", 60)),
                             grow_pct=float(getattr(C, "SYNTH_GROW_PCT", 0.90)),
                             fallback=str(getattr(C, "SYNTH_FALLBACK", "radiallp")),
                             fallback_frac=float(getattr(C, "SYNTH_FALLBACK_FRAC", 0.02)),
                             fallback_keep=int(getattr(C, "SYNTH_FALLBACK_KEEP", 4)),
                             max_body=float(getattr(C, "SYNTH_MAX_BODY", 2.5)))
    ek = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2 * C.EDGE_EXPAND + 1,) * 2) if C.EDGE_EXPAND else None
    read_i = 0     # source-frame counter (for decimation)
    emitted = 0    # emitted-frame counter (== len(pose_log)/rows written)
    # crop-aware joint gate (§B.23) — read once, not per frame
    crop_gate_on = bool(getattr(C, "CROP_AWARE_JOINTS", False))
    crop_gate_frac = float(getattr(C, "CROP_GATE_DILATE_FRAC", 0.02))
    crop_gate_min_idx = int(getattr(C, "CROP_GATE_MIN_IDX", 7))
    crop_gate_masks = []          # (downscaled binary mask, scale, native_w, native_h) per emitted frame
    t_loop0 = time.perf_counter()
    while True:
        with T("read"):
            ok, frame = cap.read()
        if not ok:
            break
        take = (read_i % stride == 0)
        read_i += 1
        if not take:
            continue                                      # decimated: drop this source frame entirely
        if a.max_frames and emitted >= a.max_frames:
            break

        # NATIVE resolution is kept — `frame` is NEVER downscaled. Detection / pose / matte / face all
        # run on a WORK_RES letterbox COPY; only the binary mask is upscaled back to native. RATIONALE:
        # the RGB background never round-trips through WORK_RES — it stays native in the emitted video.
        H, W = frame.shape[:2]; native_wh = (W, H)
        with T("letterbox"):
            work, scale, nh, nw = letterbox(frame, work_res)
            work_rgb01 = cv2.cvtColor(work, cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0

        # DETECT on the work copy -> boxes in WORK coords (used by pose/face, which also run on work).
        with T("detect"):
            boxes_all = det(work, C.DET_THRESH)
        # §2 NET uses the FULL, UNTRUNCATED detector output (native coords): the MAX_PEOPLE cap is a
        # pose/face EMIT budget only — it must NEVER shrink the privacy guarantee, or person #4+ (whom
        # the matte may drop) is revealed. (adversary C4)
        det_all_native = [[c / scale for c in b[:4]] for b in boxes_all.tolist()]
        # ...UNION with a native-resolution tiled pass. The work copy is downscaled, so a distant
        # bystander is below the detector's operating point there and would get no protection at
        # all (see detect_native_tiled). This only ADDS boxes, so it can only widen the mask.
        det_s2_extra = []
        if getattr(C, "DETECT_TILED", True):
            with T("detect_tiled"):
                extra = detect_native_tiled(frame, det, float(getattr(C, "DET_THRESH_TILE",
                                                                      C.DET_THRESH)), work_res)
            if extra:
                # NEVER drop a full-frame detection. Running NMS over the combined list could
                # discard a real box in favour of a larger overlapping tile box, and a box that
                # disappears takes its §2 guarantee with it — a recall pass that can REMOVE
                # coverage is worse than no recall pass. So dedupe only among the ADDED boxes,
                # and only against each other and the originals, leaving det_all_native a strict
                # superset of what the full-frame pass found.
                base = list(det_all_native)
                fresh = [b for b in _box_nms(extra)
                         if not any(_iou_min(b, o) > 0.55 for o in base)]
                # §2-ONLY. Deliberately NOT merged into det_all_native: that list drives slot
                # assignment and the MAX_PEOPLE emit budget, so adding boxes to it changes WHICH
                # people the pose stage reaches — and a person who drops out of the budget loses
                # their precise pose-derived head region, costing coverage. Detecting more people
                # must never make the emitted mask smaller, so the recall pass feeds the privacy
                # net only and the emit path keeps exactly its previous behaviour.
                det_s2_extra = fresh
        # det_counts is appended AFTER the subject-lock filter below (not here), because
        # person_count is a CLOUD-FACING field: it decides whether the cloud runs its multi-person
        # WanAnimate branch. Counting refused people here made pose.json report person_count=2 on
        # d04 while emitting a single slot — the exact mismatch that can make the cloud generate a
        # character for a bystander. Still UNTRUNCATED by MAX_PEOPLE; only unenrolled people drop.

        # STABLE SLOTS: assign every detection to a left-to-right slot id with swap hysteresis
        # (person_slots.SlotTracker). x-centres and the margin width are in NATIVE coords so the
        # SLOT_SWAP_MARGIN_FRAC threshold is a true fraction of the emitted frame width.
        # order[j] = index (into the detector's list) of the person occupying slot j.
        with T("slots"):
            order, tracks_all = slots.assign_tracks(
                [(b[0] + b[2]) / 2.0 for b in det_all_native], W)
        # The MAX_PEOPLE cap is applied AFTER slot assignment, so it keeps the LOWEST slot ids
        # (the leftmost people) and the same people stay kept frame to frame. It is an emit budget
        # only — det_all_native above (untruncated) is what the §2 mask net uses.
        # ---- SUBJECT LOCK (config.SUBJECT_LOCK, user 2026-07-25) -----------------------------
        # Enroll the people present in the first SUBJECT_LOCK_FRAMES, by PERSISTENT track id.
        # Afterwards an unenrolled track contributes nothing downstream — no slot, no pose, no
        # face scalars, no person_count — which is what stops the cloud tier from generating a
        # character for a passer-by or placing one at the wrong position. Enrollment is by track
        # id, not position, so a subject who crosses the frame keeps their enrollment.
        if getattr(C, "SUBJECT_LOCK", False):
            if emitted < int(getattr(C, "SUBJECT_LOCK_FRAMES", 15)):
                # ACCUMULATE candidates; the decision is COMMITTED at window close (below), because
                # enrolling every track on sight lets a passer-by in the opening second claim the
                # subject slot and banish the real subject for the whole clip.
                for _d, _t in zip(order, tracks_all):
                    _b = det_all_native[_d]
                    _a = max(0.0, _b[2] - _b[0]) * max(0.0, _b[3] - _b[1])
                    _st = enroll_stats.setdefault(_t, [0.0, 0])
                    _st[0] = max(_st[0], _a); _st[1] += 1
                enrolled.update(tracks_all)             # provisional; replaced at commit
            else:
                if not lock_committed:
                    lock_committed = True
                    if enroll_stats:
                        _big = max(v[0] for v in enroll_stats.values())
                        _rel = float(getattr(C, "SUBJECT_LOCK_MIN_REL", 0.25))
                        _mnf = int(getattr(C, "SUBJECT_LOCK_MIN_FRAMES", 3))
                        _sel = {t for t, (a, n) in enroll_stats.items()
                                if a >= _rel * _big and n >= _mnf}
                        # Never commit an EMPTY enrollment — that would refuse everyone and emit a
                        # blank clip. If both tests reject all candidates (very short window, heavy
                        # churn), fall back to the single largest person seen.
                        if not _sel:
                            _sel = {max(enroll_stats.items(), key=lambda kv: kv[1][0])[0]}
                        enrolled = _sel
                kt = [(d, t) for d, t in zip(order, tracks_all) if t in enrolled]
                # RE-ENROLL ON LOSS (fix 2026-07-25). SlotTracker recycles track ids when a person
                # is briefly lost, so binding enrollment to the id alone banished the SUBJECT the
                # first time they blinked out: measured on d01_bystander4, track 0 was lost, the
                # same human returned as track 1, and pose was emitted for only 103 of 450 frames
                # (347 frames with NO person at all) -- which would hand the cloud an unusable
                # driving-pose stream. If NO enrolled track is present this frame but somebody is,
                # re-adopt the LARGEST current detection as the subject. A genuine late arrival
                # appears ALONGSIDE the enrolled subject, so it never reaches this branch and is
                # still refused; only a total dropout re-opens enrollment.
                if not kt and order:
                    areas = [max(0.0, (det_all_native[d][2] - det_all_native[d][0])) *
                             max(0.0, (det_all_native[d][3] - det_all_native[d][1])) for d in order]
                    j = int(np.argmax(areas))
                    enrolled.add(tracks_all[j]); reenroll_events += 1
                    kt = [(order[j], tracks_all[j])]
                dropped_tracks.update(t for t in tracks_all if t not in enrolled)
                # Record WHERE each refused person was, per frame, so the §2 auditor can ATTRIBUTE a
                # reveal to a deliberately-excluded person instead of reporting an unexplained count.
                # Without this the exclusion policy is unfalsifiable: "0 reveals, some people ignored"
                # is exactly the shape of claim that caused the 2026-07-25 audit mess.
                for d, t in zip(order, tracks_all):
                    if t not in enrolled:
                        b = det_all_native[d]
                        refused_boxes_log.append([emitted, int(t),
                                                  int(b[0]), int(b[1]), int(b[2]), int(b[3])])
                        # 🔴 2026-08-08 — WHY THE SPAN EXISTS, and why "log every frame" is not the
                        # fix it sounds like. A refused person only reaches this loop if the
                        # DETECTOR found them THAT frame. c3's bystander is 66 px at conf ~0.44,
                        # right on yolo11n's 640-input floor, so the pipeline logged 2 boxes while
                        # an independent sweep found the same person uncovered in 11 frames
                        # (§A.1m) — and the §2 auditor then mis-attributed a reveal to a mask
                        # failure because the track was logged in 1 frame of >=15 (§A.1o-3).
                        # The log was never a sample of presence; it is detection-limited, and it
                        # LOOKED like presence. The pipeline cannot log a frame it did not detect,
                        # so instead the span records the first/last frame the track was ever seen
                        # and how many frames carried a box. A consumer can then see directly that
                        # `boxes < span` means "present, not detected" rather than "absent".
                        _sp = refused_spans.setdefault(int(t), [emitted, emitted, 0])
                        _sp[0] = min(_sp[0], emitted)
                        _sp[1] = max(_sp[1], emitted)
                        _sp[2] += 1
                order = [d for d, _ in kt]; tracks_all = [t for _, t in kt]
        # UNTRUNCATED by MAX_PEOPLE, but AFTER subject lock -> person_count counts only enrolled
        # people. With SUBJECT_LOCK off this equals len(det_all_native), i.e. the old behaviour.
        det_counts.append(len(order))

        keep = order[:C.MAX_PEOPLE]
        # which NATIVE boxes the pose stage will reach — used below so every detection the
        # pose loop does NOT reach still gets a face guarantee (see the §2 FACE-COVERAGE block)
        boxes_native_kept = [det_all_native[d] for d in keep]
        # Derive the WORK-coord list FROM det_all_native so the two are index-aligned by
        # construction. `order`/`tracks_all` index det_all_native, and once the tiled pass adds
        # boxes that list is longer than the full-frame `boxes_all` — indexing the latter with a
        # slot order computed from the former is an IndexError at best and the WRONG PERSON's box
        # at worst. One source of truth avoids both.
        all_work_boxes = [[c * scale for c in b[:4]] for b in det_all_native]
        boxes = [all_work_boxes[d] for d in keep]                   # WORK coords, in SLOT order
        # `slot` is POSITIONAL (left-to-right) — that is what downstream wants (the cloud graph's
        # p1=RIGHT / p2=LEFT). `track` is the PERSISTENT identity and is what the whole-clip
        # per-person stages must group by: on a count change the positional slot renumbers, so
        # grouping by it splices two people (measured 2026-07-23 — emitted rows were only
        # "01"/"012", never a gap, i.e. slot ids carried no identity at all).
        frame_slots = list(range(len(keep)))
        frame_tracks = tracks_all[:C.MAX_PEOPLE]                    # aligned with `keep`
        boxes_native = [[c / scale for c in b[:4]] for b in boxes]  # truncated set, in SLOT order

        # POSE on the work copy -> keypoints scaled back to NATIVE coords for emit.
        frame_pose = []; head_regions = []
        if pose is not None and boxes:
            with T("pose"):
                kps, scr = pose(work, bboxes=np.array([b[:4] for b in boxes], np.float32))
            for si, (k, sc) in enumerate(zip(kps, scr)):
                k = np.array(k, np.float32).copy(); sc = np.array(sc, np.float32).copy()
                k[:, :2] = k[:, :2] / scale               # WORK -> NATIVE coords
                # EMIT the UNCLIPPED keypoints; clip only the LOCAL copy used for head boxes.
                # WHY (measured, ledger §B.10 + 2026-07-26): clipping BEFORE anonymisation pins
                # below-frame hips/knees to y=H on bust/¾ framings, so do_canon's clip-median
                # torso (mid-shoulder→mid-hip) is UNDERESTIMATED and every limb group scales
                # down from the bad reference — emitted/real shoulder width 0.350/0.422/0.442
                # across three e2e runs. Off-frame floats are legal in pose.json (every consumer
                # is float-safe; cv2 clips lines at the canvas edge when drawing), and privacy is
                # unchanged: only the ANONYMISED skeleton leaves, clipped or not.
                k_emit = k.copy()
                k[:, 0] = np.clip(k[:, 0], 0, W); k[:, 1] = np.clip(k[:, 1], 0, H)
                # §2 FACE-COVERAGE GUARANTEE (adversary C2): the head keypoints nose/eyes/ears (body
                # idx 0..4, NOT the zeroed face-landmark block) localise the FACE every frame. We record
                # a head box from the confident ones so the mask ALWAYS covers the face even if the
                # matte partially drops the head while the torso stays matted (a whole-box mean can't
                # catch that). These kps are used LOCALLY for masking only — never emitted.
                hk = k[0:5][sc[0:5] > C.POSE_THRESH]
                if len(hk) >= 2:
                    hx1, hy1 = hk[:, 0].min(), hk[:, 1].min(); hx2, hy2 = hk[:, 0].max(), hk[:, 1].max()
                    pad = 0.6 * max(hx2 - hx1, hy2 - hy1, 8.0)   # expand ear-to-ear span to whole head
                    head_regions.append([hx1 - pad, hy1 - 1.2 * pad, hx2 + pad, hy2 + pad])
                # PRIVACY: zero the 68 face landmarks (wholebody idx 23..90) — face-landmark
                # geometry is a re-identifiable soft biometric and must not leave this device.
                # Body (0..22) + hands (91..132) are kept; the face is carried identity-free by
                # face_scalars.json. Mirrors the cloud mirage_rtmpose node (133-length preserved).
                k_emit[23:91] = 0.0; sc[23:91] = 0.0
                # "slot" = the STABLE left-to-right person id (person_slots policy
                # "x-sorted+hysteresis"), so a consumer can follow one person across frames even
                # when a detection drops out and the list indices shift. Position-derived only —
                # it carries no appearance/identity information.
                frame_pose.append({"slot": frame_slots[si],
                                   "track": frame_tracks[si],
                                   "kp": np.round(k_emit, 2).tolist(),
                                   "score": np.round(sc, 3).tolist()})
        pose_log.append(frame_pose)
        # one slot row per emitted frame, parallel to BOTH pose_log and face_log (both are built in
        # slot order from `boxes`; pose_log may be shorter when the pose model is unavailable).
        slot_log.append(list(frame_slots))
        track_log.append(list(frame_tracks))     # persistent ids -> what per-person stages group by

        # 12 face scalars per person (identity-free) — the only face rep that leaves. Run on the work
        # copy (scalars are eye-distance normalised, so they are resolution-independent).
        frame_faces = []
        with T("face_scalars"):
            for b in boxes:
                frame_faces.append(face_scalars(face_crop_from_box(work, b)))
        face_log.append(frame_faces)

        # MATTE on the work copy -> binary mask, then upscale WORK_RES->native. Order matters for
        # privacy: binarize @ work -> upscale -> re-binarize @ native -> EXPAND (the LAST mask op).
        with T("matte"):
            alpha = matte.alpha(work_rgb01, C.RVM_DOWNSAMPLE)
        binm_work = (alpha >= C.BINARIZE).astype(np.uint8)
        binm_content = binm_work[:nh, :nw]                # strip the letterbox padding first
        # upscale the BINARY mask to native. Edge-aware guidedFilter (native gray frame as guide)
        # snaps the mask edge to the real image edge; falls back to plain INTER_LINEAR if opencv-contrib
        # (cv2.ximgproc) is not installed. Only the mask is upscaled — the background stays native.
        with T("mask_upscale"):
            mask_up = cv2.resize((binm_content * 255).astype(np.uint8), (W, H),
                                 interpolation=cv2.INTER_LINEAR)
            if HAS_XIMGPROC:
                try:
                    guide = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
                    mask_up = cv2.ximgproc.guidedFilter(guide, mask_up, C.GUIDED_RADIUS, C.GUIDED_EPS)
                except Exception:
                    pass                                  # any guided-filter failure -> keep INTER_LINEAR
            binm = (mask_up >= 128).astype(np.uint8)      # RE-binarize at native (soft interp -> hard)

        # OPTIONAL person-instance SEG union (flag-guarded, default OFF). RVM is a SINGLE-foreground
        # matte that intermittently drops the person; a person-shaped seg mask (covers the face pixel-
        # accurately) union-ed in closes that gap with a tighter silhouette than the box fallback.
        # seg_boxes (when present) are the native-space person boxes used by the §2 guarantee below.
        seg_boxes = []
        if seg is not None:
            with T("seg"):
                seg_m, seg_boxes = seg.person_mask(frame)  # native-res person mask + boxes
                seg_m = guided_seg_post(seg_m, frame, C)   # DECISION.md guided edge-align + close + d4
                binm |= seg_m

        # FaceGuard (DECISION.md): union an expanded ellipse over every detected face — a face the body
        # seg misses (a nose/mouth in profile, a face at the mask edge) is still grayed, without a
        # whole-body halo. Runs BEFORE the COVER_MIN snapshot and the EXPAND dilate so faces also get
        # the outward margin.
        if faceguard is not None:
            with T("faceguard"):
                fg = faceguard.face_fill(frame)
                # Clamp the face oval to the CURRENT person silhouette (seg|RVM) + a small margin, so it
                # reinforces the face WITHIN the person and never bulges past the mask outline (user
                # 2026-07-24). Where the whole person is dropped, the box guarantee below still covers it.
                cpx = int(getattr(C, "FACE_GUARD_CLAMP_PX", 15))
                clamp = cv2.dilate(binm, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2 * cpx + 1,) * 2))
                binm |= (fg & clamp)

        # SNAPSHOT for the COVER_MIN decision, taken here — after the matte/seg, BEFORE any
        # detection-derived fill. COVER_MIN asks one question: "does the MATTE cover this person?"
        # Answering it against a binm that already contains head rectangles and other boxes' fills
        # makes the answer depend on how many people were detected, so detecting MORE people
        # SUPPRESSES fills and removes coverage. Measured 2026-07-23: 12 370 px, of which 89 were
        # real person pixels by an independent segmenter. The matte is detector-independent, so
        # deciding against it makes every box's fill independent of every other and of the box
        # count -- the guarantee is then strictly additive in the detector, as §2 requires.
        binm_matte = binm.copy()

        # §2 FACE-COVERAGE: union the head-keypoint regions (nose/eyes/ears) so the face is grayed even
        # when the matte drops the head with the torso still matted (adversary C2). Head is always part
        # of the bystander -> always safe to gray; only additive on matte-failure frames.
        # ...and for EVERY detection the pose stage did not reach. head_regions is built inside the
        # pose loop, which iterates `boxes` — truncated to MAX_PEOPLE. So person #4+ got the box
        # net but NOT the face guarantee, contradicting config's claim that MAX_PEOPLE is "NEVER a
        # privacy cap": it is not a cap on the BOX net, but it was one on this. It also made §2
        # NON-MONOTONE in the detector — adding tiled detections changed which people `keep`
        # selected, and 13 150 previously-grayed pixels went UNCOVERED (measured 2026-07-23).
        # A §2 mechanism must never lose coverage when the detector finds MORE people.
        # Fallback shape: the top HEAD_FRAC of the person box. That ratio is ANATOMICAL (head ≈
        # 1/7–1/8 of standing height; 0.30 leaves margin for a seated/partial box), not a value
        # fitted to any clip, and it is applied unconditionally exactly as the pose-derived region
        # above is — so people beyond the emit budget are treated the same way, not worse.
        # Applied to EVERY detection, not only the ones the pose stage missed. Restricting it to
        # non-kept boxes still left §2 coupled to the emit budget: when the detector finds more
        # people, MAX_PEOPLE keeps a different subset, and whoever drops out loses their precise
        # pose-derived head region and falls back to this coarser one — a net LOSS of coverage
        # caused purely by detecting more. Applying it unconditionally makes head coverage a
        # function of the detections alone, so it can only grow with them. The pose-derived
        # regions below are unioned on top, and remain the tighter signal where available.
        # MINIMAL TIGHT HEAD OVAL (user 2026-07-24): draw ONLY the POSE-DERIVED head regions (a real
        # head position from the pose stage) as a tight, head-proportioned ELLIPSE (width capped to
        # ~1.2x its height so it hugs the head, not shoulder-to-shoulder). The coarse box-derived
        # top-HEAD_FRAC-of-every-box fill (`extra_heads`) is DROPPED: it produced the shoulder-wide
        # square blocks and is redundant now that the guided seg covers the whole body and FaceGuard
        # covers the detected face — both run untruncated, so people beyond MAX_PEOPLE are still
        # covered by seg + FaceGuard, not by this backstop.
        for hr in list(head_regions):
            hx1, hy1 = max(0, int(hr[0])), max(0, int(hr[1])); hx2, hy2 = min(W, int(hr[2])), min(H, int(hr[3]))
            if hx2 > hx1 and hy2 > hy1:
                cx, cy = (hx1 + hx2) // 2, (hy1 + hy2) // 2
                # TIGHT VERTICAL face-shaped ellipse (user 2026-07-24): taller than wide (ax < ay),
                # and much smaller than the head box, so it hugs the face rather than the whole head.
                ay = max(1, int((hy2 - hy1) * 0.42))       # vertical half-axis (taller)
                ax = max(1, int(ay * 0.62))                # narrower than tall = face shape, tight
                cv2.ellipse(binm, (cx, cy), (ax, ay), 0, 0, 360, 1, -1)

        # DETECTION-BACKED §2 GUARANTEE — runs BEFORE the EXPAND dilate (adversary C1) so any box we
        # fill also gets the outward margin, and coverage is measured on the UN-dilated mask (dilation
        # would inflate the mean and make the guarantee under-fire). The box set is the UNION of ALL
        # detector boxes (untruncated, C4) AND seg boxes (C5) — seg must ADD to, never REPLACE, the
        # detector net. RVM is a single-foreground matte that can drop a person entirely (measured:
        # full reveal during a person's entrance ~8-9s); any person box the mask covers < COVER_MIN is
        # grayed with an EDGE_EXPAND+ margin so a box that clips hair/fingertips still covers the real
        # edge. prev_boxes carry is AGE-CAPPED (C3/D5): a stale box is used for at most PREV_BOX_MAX_AGE
        # frames, then expired, so we neither gray the wrong (old) position for long nor leave a
        # permanent phantom gray rectangle after a person leaves.
        cur_boxes = list(det_all_native) + list(det_s2_extra) + list(seg_boxes)   # UNION, untruncated
        # The carry is a UNION, not an either/or. It used to be `cur_boxes if cur_boxes else
        # prev_boxes`, which is all-or-nothing: a frame where 3 people drop to 1 detection got NO
        # carry at all, even though two people had just vanished — precisely the partial-dropout
        # case the carry exists for. It also made §2 non-monotone in the detector: finding ONE
        # more person could turn an empty frame into a non-empty one and thereby DISCARD the
        # carried boxes, losing coverage (measured 2026-07-23: 12 370 px). Unioning fixes both,
        # and the age cap still expires the carry so no permanent phantom rectangle is left.
        max_age = int(getattr(C, "PREV_BOX_MAX_AGE", 3))
        use_boxes = list(cur_boxes)
        if prev_boxes and prev_boxes_age < max_age:
            use_boxes += list(prev_boxes)
        if cur_boxes:
            prev_boxes = cur_boxes; prev_boxes_age = 0
        elif prev_boxes and prev_boxes_age < max_age:
            prev_boxes_age += 1                                # carry the last boxes briefly (correlated gap)
        else:
            prev_boxes = []; prev_boxes_age = 0                # expire -> no permanent phantom rectangle
        _gmv = getattr(C, "GUARANTEE_MARGIN", None)
        gm = int(_gmv) if _gmv is not None else (int(C.EDGE_EXPAND or 0) + 2)
        cmin = float(getattr(C, "COVER_MIN", 0.40))
        # DECIDE against a SNAPSHOT taken before any guarantee fill, then apply. Measuring coverage
        # on the live `binm` made this loop ORDER-DEPENDENT and NON-MONOTONE: filling one box raises
        # the mean inside every box overlapping it, which can push a later box ABOVE cmin and
        # SUPPRESS its fill. So adding a detection could REMOVE coverage — measured 2026-07-23, when
        # the new tiled pass cost 18 881 previously-grayed pixels. A §2 guarantee must not depend on
        # the order the detector happened to return boxes in, and must never shrink when the
        # detector finds MORE people. Deciding on the snapshot makes every box independent, so the
        # result is the union of all fills and is strictly additive in the box set.
        for bx in use_boxes:
            x1, y1, x2, y2 = int(bx[0]), int(bx[1]), int(bx[2]), int(bx[3])
            x1, y1 = max(0, x1), max(0, y1); x2, y2 = min(W, x2), min(H, y2)
            # Coverage is judged per horizontal BAND, not as one box-wide mean. A single mean over
            # the whole box can be satisfied by ANOTHER PERSON'S mask pixels: when two people
            # overlap, a well-matted neighbour lifts the mean inside the second person's box above
            # COVER_MIN, so the fill never fires and the overlapped person stays revealed. Bands
            # make that impossible to hide — a person whose legs are matted but whose head is not
            # has a failing band even if the box mean looks fine.
            # Strictly more conservative: min(bands) <= mean(box), so this fires wherever the old
            # test did, and additionally where a sub-region is bare. §2 stays monotone.
            bands = max(1, int(getattr(C, "COVER_BANDS", 4)))
            reg = binm_matte[y1:y2, x1:x2]
            if getattr(C, "BOX_GUARANTEE_FILL", False) and x2 > x1 and y2 > y1 and min(
                    float(b.mean()) for b in np.array_split(reg, bands, axis=0) if b.size) < cmin:
                gy1, gy2 = max(0, y1 - gm), min(H, y2 + gm); gx1, gx2 = max(0, x1 - gm), min(W, x2 + gm)
                # PERSON-SHAPED fallback (user 2026-07-24): a vertical ELLIPSE inscribed in the box, NOT
                # a solid rectangle — covers the centred person (detectors centre the subject) and drops
                # the empty box corners, so a seg-drop no longer paints a hard grey block. The face sits
                # at the top-centre and stays inside the ellipse; FaceGuard reinforces it.
                ecx, ecy = (gx1 + gx2) // 2, (gy1 + gy2) // 2
                cv2.ellipse(binm, (ecx, ecy), (max(1, (gx2 - gx1) // 2), max(1, (gy2 - gy1) // 2)),
                            0, 0, 360, 1, -1)

        # Advance the displacement PHASE every emitted frame so the perturbation varies over the
        # sequence -- that is what perturbs WALKING RHYTHM, as opposed to static body shape. With a
        # constant phase only the static half of the mitigation is active.
        C._MASK_DISPLACE_PHASE = float(getattr(C, 'MASK_DISPLACE_PHASE_STEP', 0.35)) * emitted

        # ---- BACKGROUND-PERSON SCOPE (user decision 2026-07-25) -------------------------------
        # ⚠️ DELIBERATE PRIVACY SCOPE REDUCTION — see config.MASK_IGNORE_BACKGROUND. Drops small
        # person components so distant background people are not masked at all. Runs BEFORE the
        # EXPAND dilate (so a dropped blob is not re-grown) and BEFORE mask_mitigate (so the
        # temporal running-max cannot resurrect it from an earlier frame). This is the ONE place
        # in the pipeline where the mask is allowed to SHRINK, so it is gated, logged, and the
        # dropped area is reported in pose.json for the auditor to account for.
        if getattr(C, "MASK_IGNORE_BACKGROUND", False):
            nlab, lab, st, _ = cv2.connectedComponentsWithStats(binm, 8)
            if nlab > 2:                       # 1 background label + >1 person component
                areas = st[1:, cv2.CC_STAT_AREA]
                big = int(areas.max())
                rel = float(getattr(C, "MASK_IGNORE_REL", 0.10))
                absf = float(getattr(C, "MASK_IGNORE_ABS_FRAC", 0.02))
                frame_px = binm.shape[0] * binm.shape[1]
                # Drop only if BOTH the relative and the absolute test say "background".
                drop = [i + 1 for i, a in enumerate(areas)
                        if a < rel * big and a < absf * frame_px]
                if drop:
                    killed = np.isin(lab, drop)
                    ignored_px_total += int(killed.sum())
                    ignored_blob_total += len(drop)
                    binm[killed] = 0

        if ek is not None:
            # EXPAND (dilate) outward so the real person is FULLY covered by gray (matches V8.5 #182
            # expand=1). The grayed margin is NOT a cloud leak — gray-fill grays it, so 2A ships gray.
            # Per user: keep == V8.5 (too much -> character halo; too little -> person shows). LAST mask op.
            with T("dilate"):
                binm = cv2.dilate(binm, ek)

        # ---- SYNTHETIC BODY (2026-08-01, config.MASK_SYNTH_BODY, default OFF) -----------------
        # Replace the person's own outline with a CANONICAL figure posed by their REAL joints.
        # See synth_body.py. Placed HERE deliberately:
        #   * AFTER the EXPAND dilate and the background-scope drop, so what must be covered is
        #     exactly the mask that would otherwise have been emitted;
        #   * BEFORE mask_mitigate, so the temporal running-max and every shape mode operate on
        #     the SYNTHETIC outline. Putting it inside mask_mitigate would be useless: that
        #     function OR-s `cur` back in as its §2 guarantee, which would restore the real
        #     outline on every frame and defeat the whole point.
        # `frame_pose` here still holds the RAW detector keypoints — pose anonymisation is a
        # whole-clip stage that runs after this loop. That is required (the anonymised skeleton is
        # not where the person's limbs are) and costs nothing: only widths/head/hands/feet are
        # canonicalised, positions are the real motion, and the emitted pose.json is untouched.
        if synth is not None:
            with T("synth_body"):
                binm, _sinfo = synth.apply(binm, frame_pose)
        # SILHOUETTE MITIGATION (flag-guarded, default ON): reduce the shape information the mask
        # channel leaks (temporal running-max + polygon simplification — see mask_mitigate). Runs
        # AFTER the EXPAND dilate (so every history frame carries the margin) and ORs the current
        # mask back in, so §2 coverage is a SUPERSET of the un-mitigated mask on every frame.
        if getattr(C, "MASK_ANON_ON", True):
            with T("mask_mitigate"):
                mask_hist.append(binm)                # pre-mitigation dilated mask (no feedback growth)
                if len(mask_hist) > max(1, int(getattr(C, "MASK_TEMPORAL_WIN", 5))):
                    mask_hist.pop(0)
                binm = mask_mitigate(mask_hist, binm, float(getattr(C, "MASK_SIMPLIFY_EPS", 0.01)))

        # ---- CROP-AWARE JOINT GATE, part 1: KEEP THE MASK (§B.23; CROP_AWARE_JOINTS, default OFF) --
        # The gate must test the ANONYMISED joint positions, because the collapse is what invents the
        # geometry (a template-length upper arm drags a below-crop elbow up into frame). Anonymisation
        # is whole-clip and runs after this loop, so the mask has to survive until then.
        #
        # ⚠️ Measured the hard way: a first version tested the joints available HERE, which are the
        # RAW detector outputs — and those legitimately sit on the person, so the gate fired on only
        # 33 joint-instances (wrists/ankles) and left elbows at 100 % and hips at 84 %. Emitted-vs-real
        # honesty was UNCHANGED at 30.8 pp. Testing pre-collapse coordinates measures the wrong thing.
        #
        # Stored DOWNSCALED (max dim ~256) so the cost is bounded on a Pi5: ~64 KB/frame instead of
        # 1.6 MB at 1264², i.e. ~19 MB for a 300-frame clip. Precision loss is immaterial — the test
        # already carries a tolerance far larger than one downscaled pixel.
        if crop_gate_on:
            _mh0, _mw0 = binm.shape[:2]
            _ds = max(1.0, max(_mh0, _mw0) / 256.0)
            _small = cv2.resize(binm, (max(1, int(round(_mw0 / _ds))), max(1, int(round(_mh0 / _ds)))),
                                interpolation=cv2.INTER_NEAREST)
            crop_gate_masks.append((_small, _ds, _mw0, _mh0))

        # gray-silhouette FILL on the NATIVE frame: replace bystander pixels with 128 (bg kept real)
        with T("gray_fill"):
            masked = frame.copy(); masked[binm > 0] = (C.GRAY, C.GRAY, C.GRAY)
        out_masked = masked
        out_mask = (binm * 255).astype(np.uint8)
        if out_res:                                       # optional forced emit size (default None=native)
            ow, oh = int(out_res[0]), int(out_res[1])
            # §2 (adversary D6): resize the BINARY mask with NEAREST, then RE-FILL gray from it, so
            # masked_video and mask.mkv stay pixel-aligned and INTER_AREA never blends gray(128) with
            # real background at the silhouette edge (which would leak a partial-background boundary ring).
            out_mask = cv2.resize(out_mask, (ow, oh), interpolation=cv2.INTER_NEAREST)
            # ...then GROW it by one destination pixel. A destination pixel is an average over a
            # block of source pixels, so a block straddling the silhouette is PART PERSON. NEAREST
            # samples one point of that block and can land outside the mask, leaving the blended
            # pixel un-grayed. One pixel of dilation makes the emitted mask cover every destination
            # pixel any person pixel contributed to.
            out_mask = cv2.dilate(out_mask, np.ones((3, 3), np.uint8))
            # Downscale the ALREADY-GRAYED frame, not the raw one. This used to resize `frame`,
            # so INTER_AREA averaged REAL PERSON pixels into boundary pixels and any the re-gray
            # missed shipped a partial person. The comment justified it as avoiding gray/background
            # blending — but that blend is the SAFE direction (a grey-ish ring is over-coverage);
            # person/background blending is the leak. Fixed 2026-07-23. OUT_RES defaults to None,
            # so this path was latent, not live.
            out_masked = cv2.resize(masked, (ow, oh), interpolation=cv2.INTER_AREA)
            out_masked[out_mask > 0] = (C.GRAY, C.GRAY, C.GRAY)                       # re-gray from nearest mask
        if masked_w is None:                              # lazy writer creation at the true emit size
            eh, ew = out_masked.shape[:2]
            masked_w = cv2.VideoWriter(os.path.join(C.OUT_DIR, "masked_video.mkv"), fourcc, out_fps,
                                       (ew, eh), isColor=True)
            mask_w = cv2.VideoWriter(os.path.join(C.OUT_DIR, "mask.mkv"), fourcc, out_fps,
                                     (ew, eh), isColor=False)
            # §2 (adversary D1): a device OpenCV build lacking the FFV1 encoder opens an UN-opened
            # writer whose .write() is a silent no-op -> the run "succeeds" with EMPTY privacy artifacts.
            # Fail LOUDLY instead so the operator installs ffmpeg/FFV1 rather than shipping empty files.
            if not (masked_w.isOpened() and mask_w.isOpened()):
                raise RuntimeError("FFV1 VideoWriter failed to open — this OpenCV build lacks the FFV1 "
                                   "encoder. Install an ffmpeg-enabled OpenCV, or add a lossless PNG-sequence "
                                   "fallback. Refusing to emit silently-empty masked_video.mkv/mask.mkv.")
        with T("write"):
            masked_w.write(out_masked); mask_w.write(out_mask)
        emitted += 1

    loop_s = time.perf_counter() - t_loop0
    cap.release()
    if masked_w is not None: masked_w.release()
    if mask_w is not None: mask_w.release()
    # use `with` so the handle is flushed+closed even if json.dump raises — an unclosed
    # handle could leave a TRUNCATED pose.json / face_scalars.json (both are Tier-2 contract files).
    # ---- anti-re-ID pose anonymization (whole-clip, per person-slot) + conf binarization ----
    # The raw body-17 dynamics are a gait biometric (§2): collapse limb proportions to the baked
    # population template + the L-level dynamic perturbation, with a fresh per-slot secrets seed
    # (never identity/content-derived, never emitted). Feet/hands ride their ankle's/wrist's
    # delta; face stays zeroed. See pose_anon_edge.py (numeric-parity port of the lab pose_anon_v2).
    anon_level = str(getattr(C, "POSE_ANON_LEVEL", "L4"))
    anon_on = bool(getattr(C, "POSE_ANON_ON", True)) and anon_level != "off"
    if anon_on:
        # frame_wh=None (2026-07-26): do NOT clip the anonymised output to the frame. The clip
        # pinned template-consistent below-frame joints (hips/knees on a bust framing) onto the
        # border, distorting the very geometry the template had just canonicalised — and the
        # emitted skeleton is CONDITIONING, where an off-frame joint is correct (the body
        # continues below frame) and a border-pinned one is wrong. Privacy-neutral: the clip
        # only ever REMOVED spatial precision from already-anonymised, template-proportioned
        # points; parity tests always ran with frame_wh=None.
        pose_log = PA.anonymize_pose_log(pose_log, anon_level, fps=out_fps, frame_wh=None,
                                         slot_log=track_log)
    # ---- CROP-AWARE JOINT GATE (§B.23) — applied AFTER anonymisation, by design ----
    # The collapse is what invents the geometry (a template-length upper arm drags a below-crop
    # elbow up into frame), so the gate has to run on its output. `_unseen` was recorded during the
    # loop against the final emitted mask. Zeroing uses (0,0) + score 0 — the pipeline's existing
    # "not detected" marker, which every consumer already skips, so no downstream contract changes.
    crop_gate_stats = {"on": bool(crop_gate_on)}
    if crop_gate_on and crop_gate_masks:
        # Test the ANONYMISED positions against the frame's own emitted mask. A below-shoulder joint
        # that does not land on the person the camera saw was not observed — it is an artefact of the
        # template collapse — so it is zeroed rather than asserted to the generator.
        _dropped = {}
        _total = 0
        _n = min(len(pose_log), len(crop_gate_masks))
        for _i in range(_n):
            _small, _ds, _mw0, _mh0 = crop_gate_masks[_i]
            _sh, _sw = _small.shape[:2]
            _tol = max(1, int(round(crop_gate_frac * math.sqrt(max(float(_small.sum()), 1.0)))))
            _grown = cv2.dilate(_small, cv2.getStructuringElement(
                cv2.MORPH_ELLIPSE, (2 * _tol + 1, 2 * _tol + 1)))
            for _p in pose_log[_i]:
                _kp = _p["kp"]
                for _j in range(crop_gate_min_idx, min(17, len(_kp))):
                    _x, _y = _kp[_j][0], _kp[_j][1]
                    if _x <= 0 and _y <= 0:
                        continue                       # already "not detected"
                    _xi, _yi = int(round(_x / _ds)), int(round(_y / _ds))
                    if 0 <= _xi < _sw and 0 <= _yi < _sh and _grown[_yi, _xi] > 0:
                        continue                       # on the person we actually saw — keep
                    _kp[_j] = [0.0, 0.0]
                    if _j < len(_p.get("score", [])):
                        _p["score"][_j] = 0.0
                    _dropped[_j] = _dropped.get(_j, 0) + 1
                    _total += 1
        crop_gate_stats.update({"joints_zeroed": _total,
                                "per_joint": {str(k): v for k, v in sorted(_dropped.items())},
                                "frames": _n, "dilate_frac": crop_gate_frac,
                                "min_idx": crop_gate_min_idx})
        print("[crop-gate] zeroed %d joint-instances not on the observed person over %d frames: %s"
              % (_total, _n, crop_gate_stats["per_joint"]), flush=True)
    crop_gate_masks = []          # release the retained masks as soon as the gate has run
    # ---- OPEN-CHAIN FREE-END PRUNE (config.POSE_FREE_END_PRUNE — its comment is the policy) ----
    # Runs AFTER anonymisation and AFTER the crop gate, on the joints that will actually be drawn.
    # Drops an INTERMEDIATE limb joint (elbow/knee/hip) that is the last drawn point of its chain
    # INSIDE the picture — the "limb ends in open space" cue a pose-conditioned generator completes.
    # A joint outside the frame is kept: that bone exits the canvas, which is the honest cue.
    free_end_stats = {"on": bool(getattr(C, "POSE_FREE_END_PRUNE", False))}
    if free_end_stats["on"] and native_wh:
        _fw, _fh = float(native_wh[0]), float(native_wh[1])
        # (joint, its distal child), DISTAL FIRST so ankle->knee(->hip) settles in a single pass.
        # Wrists (9/10) and ankles (15/16) are absent from this list on purpose: they are the
        # anatomical END of their chain, so drawing one is correct, not a dangling terminus.
        # The HIPS (11/12) are opt-in only — see config.POSE_FREE_END_INCLUDE_HIPS for the measured
        # reason (a hip is a corner of the torso quad, not a free end of the emitted skeleton).
        _CHAIN = [(13, 15), (14, 16)]
        if getattr(C, "POSE_FREE_END_INCLUDE_HIPS", False):
            _CHAIN += [(11, 13), (12, 14)]
        _CHAIN += [(7, 9), (8, 10)]
        free_end_stats["include_hips"] = bool(getattr(C, "POSE_FREE_END_INCLUDE_HIPS", False))
        _fdrop = {}
        _ftot = 0
        for _fr in pose_log:
            for _p in _fr:
                _kp = _p["kp"]
                _sc = _p.get("score", [])
                for _j, _child in _CHAIN:
                    if _j >= len(_kp) or _child >= len(_kp):
                        continue
                    _x, _y = _kp[_j][0], _kp[_j][1]
                    if _x == 0 and _y == 0:
                        continue                       # already "not detected"
                    if not (0.0 <= _x < _fw and 0.0 <= _y < _fh):
                        continue                       # outside the canvas: the bone exits — keep
                    _cx, _cy = _kp[_child][0], _kp[_child][1]
                    if not (_cx == 0 and _cy == 0):
                        continue                       # the chain continues past it — keep
                    _kp[_j] = [0.0, 0.0]
                    if _j < len(_sc):
                        _sc[_j] = 0.0
                    _fdrop[_j] = _fdrop.get(_j, 0) + 1
                    _ftot += 1
        free_end_stats.update({"joints_zeroed": _ftot,
                               "per_joint": {str(k): v for k, v in sorted(_fdrop.items())},
                               "frames": len(pose_log)})
        print("[free-end] zeroed %d intermediate limb joints left dangling inside the frame over "
              "%d frames: %s" % (_ftot, len(pose_log), free_end_stats["per_joint"]), flush=True)
    conf_bin = bool(getattr(C, "POSE_CONF_BINARIZE", True))
    if conf_bin:
        # the raw per-joint confidence trace is an identity side-channel (lab strip_conf ablation):
        # emit {0,1} at the pipeline's own keypoint threshold.
        pose_log = PA.binarize_pose_scores(pose_log, C.POSE_THRESH)
    # AUTO person count for THIS clip, from the clip's own UNTRUNCATED detector stream (never a
    # hardcoded N, never a constant measured on one clip). May legitimately EXCEED the emitted slot
    # count when more people were detected than MAX_PEOPLE — the count describes the scene, the
    # slots describe what was emitted; downstream reads both.
    person_count = PS.estimate_person_count(det_counts,
                                            int(getattr(C, "PCOUNT_SUSTAIN_FRAMES", 8)))
    emitted_slots = max((len(r) for r in slot_log), default=0)
    n_tracks = len({t for r in track_log for t in r})
    with open(os.path.join(C.OUT_DIR, "pose.json"), "w") as _f:
        # CONTRACT: pose.json is {"anon": {...}, "person_count": N, "slot_policy": "...",
        # "emitted_slots": M, "frames": [frames]}; each person entry additionally carries "slot".
        #   * "anon"          — what anonymization the artifact carries (NEVER the seed).
        #   * "person_count"  — people estimated in the clip (person_slots.estimate_person_count
        #                       over the untruncated per-frame detection counts, sustain window
        #                       PCOUNT_SUSTAIN_FRAMES). Replaces every hardcoded person count
        #                       downstream (the cloud graph's INTConstant, etc.).
        #   * "slot_policy"   — how "slot" ids are assigned (person_slots module docstring).
        #   * "emitted_slots" — max slots actually present in a frame (<= MAX_PEOPLE emit budget).
        # No code consumer existed for pose.json when the "anon" key was added (2026-07-22); these
        # keys are additive, so any consumer written against that shape keeps working.
        # SKELETON-TEMPLATE PROVENANCE (2026-07-26). `anatomical_template_for_clip()` returns which
        # collapse target it used and silently falls back to the legacy torso-scaled template when the
        # shoulders were never confidently seen -- but the returned kind was only ever appended to a
        # module-level list that NOTHING read, so no artifact recorded which path produced it.
        # That mattered: on the 2026-07-26 fps ladder the emitted shoulder width came out 497.0 /
        # 353.8 / 445.9 px for the SAME subject and clip at stride 3 / 2 / 1, while the source-side
        # estimate is stable to 0.00 % across those strides -- and with no provenance in the artifact
        # there was no way to tell a fallback from a mis-scaled anatomical run. Record it.
        try:
            from pose_anon_edge import _TEMPLATE_KIND_USED as _TK
            _tk = sorted(set(_TK))
            _tk_counts = {k: _TK.count(k) for k in _tk}
        except Exception:
            _tk, _tk_counts = [], {}

        # The EFFECTIVE dynamic knobs, after any MIRAGE_ANGLE_CONST / MIRAGE_ANGLE_DRIFT /
        # MIRAGE_LOWFREQ_AMP env override has been folded into LEVELS at import time.
        # Recording only "level": "L4" was not enough: on 2026-07-27 every corrected arm
        # (fixv_a10/a15/a30 and the whole c4 sweep) was rendered with angle_const/angle_drift
        # forced to 0/0 by env var, while the committed LEVELS["L4"] default still said 20/15 --
        # so the artifacts and the source DISAGREED and no bundle could say which one built it.
        # It had to be recovered forensically from bone-angle statistics (5.3-7.3x less angular
        # travel per frame; torso range 42-66 deg -> 3.3-3.9 deg). Same reasoning as
        # template_kinds above: if a knob can change the emitted skeleton, the artifact declares it.
        # 🔴 THE KNOBS RECORDED HERE MUST BE THE EFFECTIVE ONES (fixed 2026-08-07). This block
        # used to record LEVELS[level] alone, which is WRONG the moment a gait PRESET is selected:
        # `MIRAGE_GAIT_PRESET=g18` sets cadence_amp 0.9, and the artifact declared the L4 default of
        # 0.6 — a bundle that actively MISSTATED the config it was built with, and did not record
        # `angle_groups` / `projection_fit` / `limb_phase_amp` at all. That is the §A.2d divergence
        # in miniature (committed source vs rendered artifact, with no way to tell which built it),
        # and it is exactly what the preset mechanism exists to prevent. Caught on the first real
        # run that used a preset, before any bundle left the laptop.
        _preset_name = ""
        _scale_req = None
        try:
            from pose_anon_edge import LEVELS as _LV, gait_preset as _gp, SHIPPED_PRESET as _SP
            # 🔴 Read the env with the SAME default `gait_preset()` uses (2026-08-14). Reading it
            # with a "" default recorded `gait_preset: ""` on artifacts that were in fact built
            # with the shipped preset — the artifact claiming "no preset" while carrying one is
            # the §A.2d divergence exactly, and it appeared the moment the default stopped being
            # empty. Caught by verifying the emitted provenance, not by reading the diff.
            _preset_name = os.environ.get("MIRAGE_GAIT_PRESET", _SP).strip()
            _knobs = {k: v for k, v in _LV.get(anon_level, {}).items()} if anon_on else {}
            if anon_on:
                # the preset OVERRIDES the level knobs at the call site, so it must override here
                _p = _gp()
                # `scale_from` is NOT an anonymize_v2 knob — it selects the COLLAPSE TARGET one
                # level up — so it is recorded separately rather than misfiled among the knobs.
                _scale_req = _p.pop("scale_from", None)
                for _k, _v in _p.items():
                    _knobs[_k] = list(_v) if isinstance(_v, tuple) else _v
            _e = os.environ.get("MIRAGE_POSE_SCALE_FROM")
            if _e is not None and _e.strip():
                _scale_req = _e.strip()
            # The three SIZE knobs (2026-08-10/11) are env-overridable in `anonymize_pose_log`
            # exactly like MIRAGE_POSE_SCALE_FROM above, so the same override has to be reflected
            # here or the artifact under-reports what actually built it. Caught 2026-08-12: a
            # bundle generated at MIRAGE_HEIGHT_MULT=1.70 recorded the preset's 1.55, which is
            # precisely the §A.2d/§FIX3 failure this block exists to prevent -- an arm whose
            # provenance says something other than what was rendered.
            for _env, _key in (("MIRAGE_HEIGHT_MULT", "height_mult"),
                               ("MIRAGE_ARM_MULT", "arm_mult"),
                               ("MIRAGE_LEG_MULT", "leg_mult")):
                _v = os.environ.get(_env)
                if _v is not None and _v.strip():
                    _knobs[_key] = float(_v)
        except Exception:
            _knobs = {}

        json.dump({"anon": {"level": anon_level if anon_on else "off",
                            "canon": anon_on,
                            "conf_binarized": conf_bin,
                            # the actual numeric knobs behind `level`, env overrides included
                            "level_knobs": _knobs,
                            # WHICH NAMED PRESET produced those knobs ("" = the shipped default).
                            # An artifact must be able to say what built it without anyone
                            # reconstructing the env from a build script (§A.2d).
                            "gait_preset": _preset_name,
                            # WHICH SCALE SOURCE WAS REQUESTED (None = the "shoulder" default).
                            # `template_kinds` below records what was RESOLVED; recording the
                            # request too is what separates "the operator asked for spine and the
                            # trunk was never seen" from "nobody asked". The 2026-08-07 gait
                            # decision came apart precisely here: the approved arm was built with
                            # `MIRAGE_POSE_SCALE_FROM=projected` and the preset that replaced it
                            # carried only the same-sounding `projection_fit` kwarg (§FIX3).
                            "pose_scale_from": _scale_req,
                            # and the head anchor, which is REJECTED on privacy (§A.2m-2c) and so
                            # must be visible in any artifact that turned it on anyway
                            "head_anchor": os.environ.get("MIRAGE_HEAD_ANCHOR", "off"),
                            # which collapse target each slot actually used; a
                            # "legacy_torso_fallback" here means the shoulders were never
                            # confidently observed and the proportions are NOT the standard skeleton
                            "template_kinds": _tk,
                            "template_kind_counts": _tk_counts,
                            # §B.23: which below-shoulder joints were dropped because they did not
                            # land on the person the camera actually saw. Declared in the artifact
                            # so a consumer can tell a genuinely-absent joint from a gated one.
                            "crop_gate": crop_gate_stats,
                            # the open-chain free-end prune: which INTERMEDIATE limb joints were
                            # dropped because nothing continued their chain inside the picture.
                            # Declared so a consumer can tell a pruned joint from an absent one.
                            "free_end_prune": free_end_stats,
                            # 🔴 SELF-IDENTIFYING TEST MARKER. Present ONLY when
                            # MIRAGE_TEST_FIXED_SEED pinned the gait and mask perturbations, which
                            # makes them deterministic and therefore NOT privacy-safe. Emitted so a
                            # test artifact can never be mistaken for a shipped one downstream —
                            # §A.2d is the precedent, where committed source and rendered artifact
                            # silently disagreed and no bundle could say which built it. Absent on
                            # every real run, so its mere presence is the alarm.
                            **({"test_fixed_seed": PA.test_fixed_seed(),
                                "PRIVACY": "TEST ARTIFACT — deterministic perturbation, DO NOT SHIP"}
                               if getattr(PA, "test_fixed_seed", lambda: None)() is not None
                               else {})},
                   "person_count": person_count,
                   # SYNTHETIC BODY: declared so a consumer knows the emitted silhouette is a
                   # canonical figure, not this subject's outline — and, crucially, how much of
                   # the real outline the BACKSTOP had to put back (backstop_frac_of_real). A
                   # non-zero value is the honest measure of what the defence failed to cover.
                   **({"synth_body": synth.report()} if synth is not None else {}),
                   # Self-identifying artifact: if an optional dependency was missing, the
                   # file says so rather than looking identical to a full-strength run.
                   "degraded_components": list(DEGRADED),
                   "slot_policy": "x-sorted+hysteresis",
                   # ⚠️ DELIBERATE SCOPE REDUCTION, declared in the artifact so a downstream
                   # auditor cannot mistake an ignored background person for full coverage.
                   "background_scope": {
                       "enabled": bool(getattr(C, "MASK_IGNORE_BACKGROUND", False)),
                       "rel": float(getattr(C, "MASK_IGNORE_REL", 0.10)),
                       "abs_frac": float(getattr(C, "MASK_IGNORE_ABS_FRAC", 0.02)),
                       "ignored_blobs": ignored_blob_total,
                       "ignored_px": ignored_px_total},
                   "subject_lock": {
                       "enabled": bool(getattr(C, "SUBJECT_LOCK", False)),
                       "window_frames": int(getattr(C, "SUBJECT_LOCK_FRAMES", 15)),
                       "enrolled_tracks": sorted(enrolled),
                       "refused_tracks": sorted(dropped_tracks),
                       "reenroll_events": reenroll_events,
                       "refused_boxes": refused_boxes_log,
                       # DETECTION-LIMITED: boxes < frames_present means the track was
                       # there but below the detector floor on those frames. Read the
                       # span, not the box count, when attributing a reveal.
                       "refused_spans": {str(k): {"first": v[0], "last": v[1],
                                                  "frames_present": v[1] - v[0] + 1,
                                                  "frames_with_box": v[2],
                                                  "detection_limited": v[2] < (v[1] - v[0] + 1)}
                                         for k, v in sorted(refused_spans.items())}},
                   "emitted_slots": emitted_slots,
                   "distinct_tracks": n_tracks,
                   "emit_fps_target": emit_fps, "output_fps": out_fps, "stride": stride,
                   "frames": pose_log}, _f)
    # whole-clip DP on the 12 face scalars (per person-SLOT, per channel) BEFORE emit
    if getattr(C, "DP_ON", True):
        face_log = apply_dp(face_log, C, slot_log=track_log)
    with open(os.path.join(C.OUT_DIR, "face_scalars.json"), "w") as _f:
        json.dump(face_log, _f)
    print(f"[Tier-1] {emitted} frames @ {out_fps:g}fps (target {emit_fps:g}, stride {stride}) -> {C.OUT_DIR}/ "
          f"(masked_video.mkv, mask.mkv, pose.json, face_scalars.json)")
    # ASCII-only (a Pi console under a C/cp1252 locale raises UnicodeEncodeError on box drawing).
    print(f"[Tier-1] person_count={person_count} (auto, sustain "
          f"{int(getattr(C, 'PCOUNT_SUSTAIN_FRAMES', 8))}f) | emitted slots={emitted_slots} "
          f"(budget MAX_PEOPLE={C.MAX_PEOPLE}) | slot policy x-sorted+hysteresis")
    if DEGRADED:
        # ASCII-only, like every other print here: a Pi console under a C/cp1252 locale raises
        # UnicodeEncodeError on non-ASCII, and a privacy warning that crashes the run is useless.
        print("[Tier-1] !! DEGRADED RUN - missing: " + "; ".join(DEGRADED))
        print("[Tier-1] !! recorded in pose.json meta.degraded_components. The section-2 mask net")
        print("[Tier-1] !! still holds (every detection gets a box-derived head guarantee), but")
        print("[Tier-1] !! the emitted artifacts are NOT full strength.")
    print("[Tier-1] privacy: raw frames NEVER written/emitted; only masked video + mask "
          "+ pose keypoints + 12 face scalars leave this device.")

    if T.enabled:
        # Loop FPS EXCLUDES model load (a one-off) and the whole-clip emit stages (anon + DP +
        # json), because the >=15 fps claim is about the streaming per-frame loop. Both are
        # reported separately so nothing is hidden.
        stages = T.summary()
        fps = (emitted / loop_s) if loop_s > 0 else float("nan")
        rss = peak_rss_mb()
        rep = {"backend": {"detect": getattr(C, "DETECT_BACKEND", "onnx"),
                           "matte": getattr(C, "MATTE_MODEL", "rvm"),
                           "seg": getattr(C, "SEG_BACKEND", "none")},
               "work_res": work_res, "native_wh": list(native_wh) if native_wh else None,
               "frames_emitted": emitted, "frames_read": read_i, "stride": stride,
               "emit_fps_target": emit_fps, "loop_seconds": loop_s,
               "loop_fps": fps, "model_load_seconds": model_load_s,
               "peak_rss_mb": rss, "person_count": person_count,
               "stages_ms": stages}
        w = max(len(k) for k in stages) if stages else 8
        print(f"\n[Tier-1 --stats] loop {emitted} frames in {loop_s:.2f}s = {fps:.2f} FPS "
              f"| model load {model_load_s:.2f}s | peak RSS "
              f"{('%.0f MiB' % rss) if rss is not None else 'n/a'}")
        print(f"  {'stage'.ljust(w)}    n    p50 ms    p95 ms    max ms   share")
        tot = sum(s["total_ms"] for s in stages.values()) or 1.0
        for k, s in sorted(stages.items(), key=lambda kv: -kv[1]["total_ms"]):
            print(f"  {k.ljust(w)} {s['n']:4d} {s['p50_ms']:9.2f} {s['p95_ms']:9.2f} "
                  f"{s['max_ms']:9.2f} {100.0*s['total_ms']/tot:6.1f}%")
        if a.stats_json:
            with open(a.stats_json, "w") as _f:
                json.dump(rep, _f, indent=2)
            print(f"  -> {a.stats_json}")


if __name__ == "__main__":
    main()
