import sys
import json
import numpy as np
import torch
import torch.nn.functional as F
import cv2
from pathlib import Path
import onnxruntime as ort
import mediapipe as mp

try:
    import mediapipe as mp
    _HAS_MEDIAPIPE = True
except ImportError:
    _HAS_MEDIAPIPE = False

# ComfyUI-WanAnimatePreprocess is a SIBLING custom-node package; resolve it relative to this
# file so the node works wherever ComfyUI is installed (it used to be an absolute server path).
import os as _os
_SIBLING = _os.path.join(_os.path.dirname(_os.path.dirname(_os.path.abspath(__file__))),
                         "ComfyUI-WanAnimatePreprocess")
if _SIBLING not in sys.path:
    sys.path.insert(0, _SIBLING)

# ComfyUI exposes its models root via folder_paths; fall back to the documented
# deployment layout when this module is imported outside a ComfyUI process.
try:
    import folder_paths as _fp
    _DEFAULT_DETECTION_DIR = _os.path.join(_fp.models_dir, "detection")
except Exception:
    _DEFAULT_DETECTION_DIR = _os.path.join("models", "detection")
from pose_utils.pose2d_utils import load_pose_metas_from_kp2ds_seq, AAPoseMeta

# V9 anti-reID (Phase 1): lab-validated gait/anthropometry anonymiser, applied to
# kp2ds_seq BEFORE pose_json + pose_metas are built (see pose_anon_v2.py).
from .pose_anon_v2 import anonymize_kp133_seq

try:
    from rtmlib import RTMPose
    _HAS_RTMLIB = True
except Exception:
    _HAS_RTMLIB = False

# --- ORIGINAL 8 MIRAGE NODES ---
class MIRAGEReferenceCollage:
    @classmethod
    def INPUT_TYPES(cls):
        return {"required": {"face_image": ("IMAGE",), "output_size": ("INT", {"default": 512}), "layout": (["grid", "horizontal"], {"default": "grid"}), "padding": ("INT", {"default": 0})}, "optional": {"pose_front": ("IMAGE",), "pose_left": ("IMAGE",), "pose_right": ("IMAGE",), "pose_back": ("IMAGE",)}}
    RETURN_TYPES, RETURN_NAMES, FUNCTION, CATEGORY = (("IMAGE",), ("reference_collage",), "process", "MIRAGE/V5")
    def process(self, face_image, output_size=512, layout="grid", padding=0, pose_front=None, pose_left=None, pose_right=None, pose_back=None):
        images_to_concat = [img for img in [face_image, pose_front, pose_left, pose_right, pose_back] if img is not None]
        target_h, target_w = face_image.shape[1], face_image.shape[2]
        processed_images = [F.interpolate(img.movedim(-1, 1), size=(target_h, target_w), mode="bilinear").movedim(1, -1) for img in images_to_concat]
        return (torch.cat(processed_images, dim=2),)

class MIRAGEAlphaComposite:
    @classmethod
    def INPUT_TYPES(cls): return {"required": {"rgb_images": ("IMAGE",), "alpha": ("MASK",), "premultiplied": ("BOOLEAN", {"default": True}), "edge_feather": ("INT", {"default": 2})}}
    RETURN_TYPES, RETURN_NAMES, FUNCTION, CATEGORY = (("IMAGE", "MASK"), ("rgba", "alpha_out"), "process", "MIRAGE/V5")
    def process(self, rgb_images, alpha, premultiplied=True, edge_feather=2):
        if len(alpha.shape) == 2: alpha = alpha.unsqueeze(0)
        if len(alpha.shape) == 3: alpha = alpha.unsqueeze(-1)
        if alpha.shape[0] < rgb_images.shape[0]: alpha = alpha.repeat(rgb_images.shape[0], 1, 1, 1)
        return (torch.cat((rgb_images, alpha), dim=-1), alpha.squeeze(-1))

class MIRAGEMultiPersonRouter:
    @classmethod
    def INPUT_TYPES(cls): return {"required": {"mask": ("MASK",), "max_instances": ("INT", {"default": 4}), "area_threshold": ("FLOAT", {"default": 0.005}), "temporal_smooth": ("FLOAT", {"default": 0.2})}}
    RETURN_TYPES, RETURN_NAMES, FUNCTION, CATEGORY = (("MASK", "INT", "STRING"), ("instance_mask", "person_count", "track_report"), "process", "MIRAGE/V5")
    def process(self, mask, max_instances=4, area_threshold=0.005, temporal_smooth=0.2): return (mask, 1, "Tracked")

class MIRAGEInstanceMaskExport:
    @classmethod
    def INPUT_TYPES(cls): return {"required": {"instance_mask": ("MASK",), "max_channels": ("INT", {"default": 4}), "encoding": (["grey_levels", "rgb_channels"], {"default": "grey_levels"}), "step_size": ("INT", {"default": 40})}}
    RETURN_TYPES, RETURN_NAMES, FUNCTION, CATEGORY = (("IMAGE", "MASK"), ("mask_video", "binary_union"), "process", "MIRAGE/V5")
    def process(self, instance_mask, max_channels=4, encoding="grey_levels", step_size=40):
        if len(instance_mask.shape) == 2: instance_mask = instance_mask.unsqueeze(0)
        return (instance_mask.unsqueeze(-1).repeat(1, 1, 1, 3), instance_mask)

class MIRAGECloudMobilePackager:
    @classmethod
    def INPUT_TYPES(cls): return {"required": {"generated_frames": ("IMAGE",), "bitrate_target": ("INT", {"default": 16}), "crf": ("INT", {"default": 20}), "container": (["ffv1_mkv", "h265_mp4", "prores_mov"], {"default": "ffv1_mkv"})}, "optional": {"explicit_mask": ("MASK",), "instance_mask": ("IMAGE",)}}
    RETURN_TYPES, RETURN_NAMES, FUNCTION, CATEGORY = (("STRING", "STRING"), ("rgb_video_path", "mask_video_path"), "process", "MIRAGE/V5")
    def process(self, generated_frames, bitrate_target=16, crf=20, container="ffv1_mkv", explicit_mask=None, instance_mask=None): return ("/output/rgb.mp4", "/output/mask.mkv")

# --- YOUR PROVIDED RTMPose NODE ---
class RTMPoseTinyPoseAndFace:
    _detector = None
    _rtmpose  = None

    @classmethod
    def INPUT_TYPES(cls):
        return {"required": {
            "images":      ("IMAGE",),
            "width":       ("INT",   {"default": 832,  "min": 64, "max": 4096}),
            "height":      ("INT",   {"default": 480,  "min": 64, "max": 4096}),
            "det_thresh":  ("FLOAT", {"default": 0.3,  "min": 0.0, "max": 1.0, "step": 0.05}),
            "pose_thresh": ("FLOAT", {"default": 0.3,  "min": 0.0, "max": 1.0, "step": 0.05}),
            "face_pad":    ("FLOAT", {"default": 0.4,  "min": 0.0, "max": 2.0, "step": 0.05}),
            # Default resolved from ComfyUI's own models dir when available; the widget value
            # saved inside a workflow always wins over this default.
            "model_dir":   ("STRING", {"default": _DEFAULT_DETECTION_DIR}),
            "person_index": ("INT",  {"default": 0, "min": 0, "max": 7}),
        },
        # V9 anti-reID knobs. OPTIONAL and APPENDED AT THE END on purpose: saved
        # workflows (V9.json stores exactly 7 widgets_values for this node) still
        # load, because ComfyUI restores widgets_values POSITIONALLY over the
        # current widget list -- a shorter saved array leaves these two trailing
        # widgets at their defaults, and API prompts that omit optional inputs
        # fall back to detect()'s keyword defaults. Never insert/reorder above.
        "optional": {
            "anon_level":    (["L4", "L2", "L0", "off"], {"default": "L4"}),
            "conf_binarize": ("BOOLEAN", {"default": True}),
            # Real clip fps for the anonymiser's cadence band. APPENDED AFTER the two
            # V9 knobs above, same load-safe rule: saved workflows keep loading and simply
            # leave this at its default. Replaces pose_anon_v2.FPS_ASSUMED, which had to
            # guess because detect() gets frames, not a frame rate (ATTENTION_AND_RISKS
            # §2.2). Wire the source clip's real fps (e.g. VHS_LoadVideo's frame_rate) to
            # make cadence-warp exact; the default reproduces the previous behaviour
            # EXACTLY, so leaving it alone changes nothing.
            "fps":           ("FLOAT", {"default": 16.0, "min": 1.0, "max": 240.0,
                                        "step": 0.5}),
        }}

    # +STRING (V8.5): expose this track's 133 wholebody keypoints per frame as JSON, so the
    # cloud pipeline can emit pose.json drop-in for mirage_edge_deploy. The raw kp2ds_seq (x,y,
    # score per keypoint) is already computed for pose_metas; POSEDATA itself is live Python
    # objects (not serializable), hence a dedicated STRING slot. Schema matches mirage_tier1.py's
    # PER-PERSON element: {"kp": [[x,y] x133], "score": [x133]} per frame (133-length preserved).
    # PRIVACY: the 68 face landmarks are ZEROED before export (see detect()) - face geometry is a
    # re-identifiable soft biometric and must not leave; the face is carried by face_scalars.json.
    # (slot 3 type is "BBOX" - a pre-existing "BBOX," typo was fixed here; slot is unconsumed.)
    RETURN_TYPES  = ("POSEDATA", "IMAGE", "BBOX", "BBOX", "STRING")
    RETURN_NAMES  = ("pose_data", "face_images", "bboxes", "face_bboxes", "pose_json")
    FUNCTION      = "detect"
    CATEGORY      = "MIRAGE/pose"

    @classmethod
    def _load_models(cls, model_dir):
        if cls._detector is not None and cls._rtmpose is not None:
            return
        det_path  = str(Path(model_dir) / "yolox_nano.onnx")
        pose_path = str(Path(model_dir) / "rtmpose-t-wholebody.onnx")
        providers     = ["CUDAExecutionProvider", "CPUExecutionProvider"]
        cls._detector = ort.InferenceSession(det_path, providers=providers)
        cls._rtmpose  = RTMPose(
            onnx_model       = pose_path,
            model_input_size = (192, 256),
            backend          = "onnxruntime",
            device           = "cuda",
        )

    def _detect_yolov10(self, frame_bgr, thresh):
        H, W     = frame_bgr.shape[:2]
        inp_size = 640
        scale    = min(inp_size / H, inp_size / W)
        nh, nw   = int(H * scale), int(W * scale)
        resized  = cv2.resize(frame_bgr, (nw, nh))
        padded   = np.full((inp_size, inp_size, 3), 114, dtype=np.uint8)
        padded[:nh, :nw] = resized
        blob     = padded.astype(np.float32) / 255.0
        blob     = blob.transpose(2, 0, 1)[None]
        inp_name = self._detector.get_inputs()[0].name
        outputs  = self._detector.run(None, {inp_name: blob})[0]
        dets     = outputs[0]
        scores   = dets[:, 4]
        classes  = dets[:, 5].astype(int)
        keep     = (scores >= thresh) & (classes == 0)
        dets     = dets[keep]
        if len(dets) == 0:
            return np.empty((0, 5), dtype=np.float32)
        boxes = dets[:, :4] / scale
        boxes[:, 0] = np.clip(boxes[:, 0], 0, W)
        boxes[:, 1] = np.clip(boxes[:, 1], 0, H)
        boxes[:, 2] = np.clip(boxes[:, 2], 0, W)
        boxes[:, 3] = np.clip(boxes[:, 3], 0, H)
        return np.concatenate([boxes, dets[:, 4:5]], axis=1)

    FACE_KP_START = 23
    FACE_KP_END   = 90

    def _extract_face_crop(self, frame_rgb, kps, scores, face_pad, out_size=512):
        H, W        = frame_rgb.shape[:2]
        face_kps    = kps[self.FACE_KP_START : self.FACE_KP_END + 1]
        face_scores = scores[self.FACE_KP_START : self.FACE_KP_END + 1]
        valid       = face_kps[face_scores > 0.2]
        if len(valid) < 4:
            valid = kps[[0, 1, 2]]
        x1 = int(np.clip(valid[:, 0].min(), 0, W))
        y1 = int(np.clip(valid[:, 1].min(), 0, H))
        x2 = int(np.clip(valid[:, 0].max(), 0, W))
        y2 = int(np.clip(valid[:, 1].max(), 0, H))
        bw, bh = max(x2 - x1, 1), max(y2 - y1, 1)
        x1 = max(0, x1 - int(bw * face_pad))
        y1 = max(0, y1 - int(bh * face_pad))
        x2 = min(W, x2 + int(bw * face_pad))
        y2 = min(H, y2 + int(bh * face_pad))
        side = max(x2 - x1, y2 - y1)
        cx, cy = (x1 + x2) // 2, (y1 + y2) // 2
        x1 = max(0, cx - side // 2)
        y1 = max(0, cy - side // 2)
        x2 = min(W, x1 + side)
        y2 = min(H, y1 + side)
        crop = frame_rgb[y1:y2, x1:x2]
        if crop.size == 0:
            crop = np.zeros((out_size, out_size, 3), dtype=np.uint8)
        else:
            crop = cv2.resize(crop, (out_size, out_size), interpolation=cv2.INTER_LINEAR)
        return crop, (x1, y1, x2, y2)

    @staticmethod
    def _box_iou(a, b):
        ix0, iy0 = max(a[0], b[0]), max(a[1], b[1])
        ix1, iy1 = min(a[2], b[2]), min(a[3], b[3])
        if ix1 <= ix0 or iy1 <= iy0:
            return 0.0
        inter = (ix1 - ix0) * (iy1 - iy0)
        ua = ((a[2] - a[0]) * (a[3] - a[1]) + (b[2] - b[0]) * (b[3] - b[1]) - inter)
        return float(inter / max(ua, 1.0))

    def detect(self, images, width, height, det_thresh, pose_thresh, face_pad, model_dir, person_index=0,
               anon_level="L4", conf_binarize=True, fps=16.0):
        if not _HAS_RTMLIB:
            raise RuntimeError("rtmlib not installed.")
        self._load_models(model_dir)
        imgs_np = (images.cpu().numpy() * 255).astype(np.uint8)
        B       = imgs_np.shape[0]
        kp2ds_seq       = []
        all_face_crops  = []
        all_bboxes      = []
        all_face_bboxes = []
        blank_kps3      = np.zeros((133, 3), dtype=np.float32)
        blank_face      = np.zeros((512, 512, 3), dtype=np.float32)

        # Temporal identity tracks. person_index selects a TRACK (0 = the
        # LEFTMOST person when people first appear), not a per-frame
        # confidence rank, so each branch keeps following the SAME real person
        # even when detector ordering flips or the two people touch/overlap.
        tracks = []          # {"bbox": np[4], "kp3": last kp3, "face": last crop}
        EARLY  = 5           # first frames: re-sort tracks left->right as people appear

        for i in range(B):
            frame      = imgs_np[i]
            frame_bgr  = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
            bboxes_det = self._detect_yolov10(frame_bgr, det_thresh)
            assign     = {}
            kps_list, scores_list = [], []
            if len(bboxes_det) > 0:
                kps_list, scores_list = self._rtmpose(frame_bgr, bboxes=bboxes_det[:, :4])
                dets = bboxes_det[:, :4]
                if not tracks:
                    for d in np.argsort([float(b[0] + b[2]) for b in dets]):
                        tracks.append({"bbox": dets[int(d)].copy(), "kp3": None, "face": None})
                        assign[len(tracks) - 1] = int(d)
                else:
                    # greedy IoU matching against each track's last box
                    pairs = sorted(
                        ((self._box_iou(tr["bbox"], dets[d]), t, d)
                         for t, tr in enumerate(tracks) for d in range(len(dets))),
                        reverse=True)
                    used_t, used_d = set(), set()
                    for iou, t, d in pairs:
                        if iou < 0.10 or t in used_t or d in used_d:
                            continue
                        assign[t] = d; used_t.add(t); used_d.add(d)
                    # nearest-centre fallback for fast motion, gated to 25% of diag
                    diag = float(np.hypot(frame.shape[0], frame.shape[1]))
                    for t, tr in enumerate(tracks):
                        if t in used_t:
                            continue
                        cx = (tr["bbox"][0] + tr["bbox"][2]) / 2.0
                        cy = (tr["bbox"][1] + tr["bbox"][3]) / 2.0
                        best, bd = None, 0.25 * diag
                        for d in range(len(dets)):
                            if d in used_d:
                                continue
                            dist = float(np.hypot((dets[d][0] + dets[d][2]) / 2.0 - cx,
                                                  (dets[d][1] + dets[d][3]) / 2.0 - cy))
                            if dist < bd:
                                best, bd = d, dist
                        if best is not None:
                            assign[t] = best; used_t.add(t); used_d.add(best)
                    # unmatched detections spawn new tracks
                    new_spawned = False
                    for d in range(len(dets)):
                        if d not in used_d:
                            tracks.append({"bbox": dets[d].copy(), "kp3": None, "face": None})
                            assign[len(tracks) - 1] = d
                            new_spawned = True
                    if new_spawned and i <= EARLY:
                        # stabilise identity order while people are first appearing
                        order = np.argsort([float(t["bbox"][0] + t["bbox"][2]) for t in tracks])
                        remap  = {int(o): new_t for new_t, o in enumerate(order)}
                        tracks = [tracks[int(o)] for o in order]
                        assign = {remap[t]: d for t, d in assign.items()}
                    for t, d in assign.items():
                        tracks[t]["bbox"] = dets[d].copy()

            t_sel = int(person_index)
            if t_sel < len(tracks) and t_sel in assign:
                d              = assign[t_sel]
                primary_kps    = np.array(kps_list[d])
                primary_scores = np.array(scores_list[d])
                kp3 = np.concatenate([primary_kps, primary_scores[:, None]], axis=1)
                face_crop, face_bbox = self._extract_face_crop(
                    frame, primary_kps, primary_scores, face_pad, out_size=512)
                face_f = face_crop.astype(np.float32) / 255.0
                tracks[t_sel]["kp3"]  = kp3
                tracks[t_sel]["face"] = face_f
                kp2ds_seq.append(kp3)
                all_face_crops.append(face_f)
                all_face_bboxes.append([list(face_bbox)])
            else:
                # this track is not visible this frame (occlusion/merged boxes):
                # FREEZE its last pose/face instead of jumping to another person
                tr = tracks[t_sel] if t_sel < len(tracks) else None
                kp2ds_seq.append(tr["kp3"].copy() if tr is not None and tr["kp3"] is not None
                                 else blank_kps3.copy())
                all_face_crops.append(tr["face"].copy() if tr is not None and tr["face"] is not None
                                      else blank_face.copy())
                all_face_bboxes.append([])
            all_bboxes.append(bboxes_det[:, :4].tolist() if len(bboxes_det) > 0 else [])
        # V9 anti-reID (Phase 1): this ONE call rewrites kp2ds_seq BEFORE both
        # pose_json and pose_metas are built below, so the export AND the
        # WanAnimate driving pose carry the SAME anonymised skeleton -- the raw
        # gait/anthropometry never reaches either consumer. Body-17 collapses to
        # the baked population template with the lab-ladder dynamics; feet/hands/
        # face ride rigidly on ankle/wrist/nose deltas; scores are BINARIZED to
        # {0,1} at pose_thresh (continuous confidence is an identity side-
        # channel). Fresh secrets seed per run -- never identity/content-derived.
        # `fps` feeds the cadence-warp band only; a bad value degrades to a slightly
        # mis-tuned warp, never to a privacy loss, so clamp rather than fail the render.
        try:
            _fps = float(fps)
        except (TypeError, ValueError):
            _fps = 16.0
        if not (_fps > 0.0) or _fps != _fps:                 # non-positive or NaN
            _fps = 16.0
        kp2ds_seq = anonymize_kp133_seq(kp2ds_seq, anon_level=anon_level,
                                        conf_binarize=conf_binarize,
                                        pose_thresh=pose_thresh, fps=_fps)
        # identity-free pose keypoints per frame (this track) -> JSON for pose.json.
        # kp2ds_seq[i] is [133,3] = (x, y, score) in input-frame pixel coords.
        # PRIVACY: the 68 face landmarks (idx FACE_KP_START..FACE_KP_END = 23..90) are a
        # re-identifiable soft biometric of the REAL person -> ZERO them out. Body (0..22) + hands
        # (91..132) are kept (that is what the body-stick pose drives); the face is carried
        # identity-free by the MIRAGEFaceCanonicalizer scalars (face_scalars.json). 133-length is
        # preserved so the wholebody schema stays intact. Mirrored in mirage_edge_deploy tier1.
        pose_frames = []
        for f in kp2ds_seq:
            f = np.asarray(f).astype(np.float32).copy()
            f[self.FACE_KP_START:self.FACE_KP_END + 1, :] = 0.0     # strip 68 face landmarks
            pose_frames.append({"kp":    np.round(f[:, :2], 2).tolist(),
                                "score": np.round(f[:, 2], 3).tolist()})
        pose_json = json.dumps(pose_frames)
        pose_metas = load_pose_metas_from_kp2ds_seq(kp2ds_seq, width=width, height=height)
        pose_metas = [AAPoseMeta.from_humanapi_meta(m) for m in pose_metas]
        pose_data  = {
            "pose_metas":          pose_metas,
            "pose_metas_original": pose_metas,
            "retarget_image":      None,
        }
        face_tensor = torch.from_numpy(np.stack(all_face_crops, axis=0))
        return (pose_data, face_tensor, all_bboxes, all_face_bboxes, pose_json)

class MIRAGEMaskPersonCount:
    """Auto person-count from the bystander MASK stream (user req 2026-07-22).

    Counts connected components per frame on the (optionally OR-combined) binary mask and
    returns the largest count sustained for >= sustain_frames CONSECUTIVE frames - a 1-frame
    spurious blob cannot inflate the count, but a real person present briefly still counts.
    min_area_frac filters speckle as a FRACTION of frame area (never absolute pixels), so the
    node generalizes across resolutions/clips. Feed both per-person masks (or the union mask
    twice) - replaces the manual person_count INT in V9_CLOUD_ONLY."""

    @classmethod
    def INPUT_TYPES(cls):
        return {"required": {
                    "mask": ("MASK",),
                    "min_area_frac": ("FLOAT", {"default": 0.005, "min": 0.0, "max": 0.5, "step": 0.001}),
                    "sustain_frames": ("INT", {"default": 8, "min": 1, "max": 300}),
                    "threshold": ("FLOAT", {"default": 0.5, "min": 0.0, "max": 1.0, "step": 0.05}),
                },
                "optional": {"mask_b": ("MASK",)}}

    RETURN_TYPES = ("INT",)
    RETURN_NAMES = ("person_count",)
    FUNCTION = "count"
    CATEGORY = "MIRAGE"

    def count(self, mask, min_area_frac=0.005, sustain_frames=8, threshold=0.5, mask_b=None):
        m = mask.cpu().numpy() if hasattr(mask, "cpu") else np.asarray(mask)
        if m.ndim == 2:
            m = m[None]
        if mask_b is not None:
            b = mask_b.cpu().numpy() if hasattr(mask_b, "cpu") else np.asarray(mask_b)
            if b.ndim == 2:
                b = b[None]
            t = min(len(m), len(b))
            m = np.maximum(m[:t], b[:t])                       # union of the two person streams
        counts = []
        for fr in m:
            binm = (fr >= threshold).astype(np.uint8)
            n_lbl, _, stats, _ = cv2.connectedComponentsWithStats(binm, connectivity=8)
            min_area = min_area_frac * binm.shape[0] * binm.shape[1]
            counts.append(int(sum(1 for i in range(1, n_lbl) if stats[i, cv2.CC_STAT_AREA] >= min_area)))
        best, run_val, run_len = 0, -1, 0
        for c in counts:                                        # largest level sustained >= sustain_frames
            run_val, run_len = (c, run_len + 1) if c == run_val else (c, 1)
            if run_len >= sustain_frames and run_val > best:
                best = run_val
        if best == 0 and counts:
            best = int(np.median(counts))                       # very short clip fallback
        return (max(best, 1),)


# --- Node Registration for ComfyUI ---

NODE_CLASS_MAPPINGS = {
    "MIRAGEMaskPersonCount": MIRAGEMaskPersonCount,
    "MIRAGEReferenceCollage": MIRAGEReferenceCollage,
    "MIRAGEAlphaComposite": MIRAGEAlphaComposite,
    "MIRAGEMultiPersonRouter": MIRAGEMultiPersonRouter,
    "MIRAGEInstanceMaskExport": MIRAGEInstanceMaskExport,
    "MIRAGECloudMobilePackager": MIRAGECloudMobilePackager,
    # MIRAGEBackgroundFill + MIRAGEFaceCanonicalizer are intentionally NOT provided here.
    # They were DUPLICATE stubs (a no-op fill, and a REAL-FACE affine-warp) colliding with the
    # real identity-free classes in mirage_facenorm. ComfyUI resolves duplicate node names by
    # last-write-wins over an UNSORTED os.listdir, so providing these made the winner depend on
    # filesystem order - and if a stub won, real bystander faces could reach an output (privacy
    # breach) and the V8.5 STRING taps break. Their class bodies (and the unused REAL-FACE
    # affine-warp node EnhancedMIRAGECanonicalizer) have now been DELETED from this file so they
    # can never load; mirage_facenorm is the sole provider. (See STATE.md privacy-landmine note.)
    "RTMPoseTinyPoseAndFace": RTMPoseTinyPoseAndFace,
}

NODE_DISPLAY_NAME_MAPPINGS = {k: k for k in NODE_CLASS_MAPPINGS}
NODE_DISPLAY_NAME_MAPPINGS["RTMPoseTinyPoseAndFace"] = "RTMPose-T + YOLOv10m Pose & Face (MIRAGE lightweight)"

# ---------------------------------------------------------------------------------------------
# PRE-RENAME COMPATIBILITY (2026-08-29). A saved ComfyUI graph stores each node's registry key in
# its `class_type` field, so every workflow exported before the rename says "SITARA...". Loading
# one against a registry that only knows "MIRAGE..." fails with an unhelpful missing-node error.
# Each node is therefore registered a SECOND time under its old key. The alias maps to the SAME
# class object, so the two names cannot drift apart, and new graphs save under the new name.
_LEGACY = {k.replace("MIRAGE", "SITARA", 1): v
           for k, v in NODE_CLASS_MAPPINGS.items() if k.startswith("MIRAGE")}
NODE_CLASS_MAPPINGS.update(_LEGACY)
NODE_DISPLAY_NAME_MAPPINGS.update({
    k: NODE_DISPLAY_NAME_MAPPINGS.get(k.replace("SITARA", "MIRAGE", 1), k) + " (legacy name)"
    for k in _LEGACY
})
