import cv2
import numpy as np
import mediapipe as mp
from mediapipe.tasks import python
from mediapipe.tasks.python import vision

CANONICAL_SIZE = 512

# ── Landmark index groups ──────────────────────────────────────────────────────

_FACE_OVAL = [10, 338, 297, 332, 284, 251, 389, 356, 454, 323, 361, 288,
              397, 365, 379, 378, 400, 377, 152, 148, 176, 149, 150, 136,
              172, 58, 132, 93, 234, 127, 162, 21, 54, 103, 67, 109]

# Full eye contours (upper + lower lid, closed loop)
_LEFT_EYE  = [33, 246, 161, 160, 159, 158, 157, 173,
               133, 155, 154, 153, 145, 144, 163, 7]
_RIGHT_EYE = [362, 398, 384, 385, 386, 387, 388, 466,
               263, 249, 390, 373, 374, 380, 381, 382]

_LEFT_BROW  = [70, 63, 105, 66, 107, 55, 65, 52, 53, 46]
_RIGHT_BROW = [300, 293, 334, 296, 336, 285, 295, 282, 283, 276]

_LIPS_OUTER = [61, 185, 40, 39, 37, 0, 267, 269, 270, 409,
                291, 375, 321, 405, 314, 17, 84, 181, 91, 146]
_LIPS_INNER = [78, 191, 80, 81, 82, 13, 312, 311, 310, 415,
                308, 324, 318, 402, 317, 14, 87, 178, 88, 95]

_NOSE_BRIDGE = [168, 6, 197, 195, 5, 4]
# Single nostril landmark indices (used as circle centres)
_LEFT_NOSTRIL  = 48
_RIGHT_NOSTRIL = 278
_NOSE_TIP      = 4

# ── Diffusion-friendly color palette (BGR) ─────────────────────────────────────
_BG         = (228, 225, 222)   # light warm-gray background
_SKIN       = (185, 190, 198)   # neutral face fill (no ethnicity bias)
_OUTLINE    = (70,  65,  62)    # face / eyelid outline
_BROW       = (38,  44,  58)    # dark brows
_SCLERA     = (250, 250, 248)   # eye whites
_IRIS       = (48,  58,  78)    # dark iris
_PUPIL      = (8,   8,   12)    # near-black pupil
_HIGHLIGHT  = (255, 255, 255)   # specular highlight on iris
_LIP        = (90,  82, 195)    # dusty-rose lips (R=195, G=82, B=90)
_LIP_IN     = (48,  36, 145)    # darker mouth-cavity / inner lip
_NOSE_SHD   = (162, 168, 178)   # subtle nose shadow (slightly darker than skin)


def _px(landmarks, size=CANONICAL_SIZE):
    return np.array([[int(lm.x * size), int(lm.y * size)]
                     for lm in landmarks], dtype=np.int32)


def yaw_from_transform(matrix: np.ndarray) -> float:
    """Head yaw in degrees from a FaceLandmarker facial_transformation_matrix
    (needs output_facial_transformation_matrixes=True on the options).
    0 = facing camera, +-90 = full profile. FaceLandmarker's mesh output has
    no usable per-detection confidence score (NormalizedLandmark.presence is
    unset for this model, confirmed by direct inspection) -- yaw from the
    real head-pose rotation is used instead, as a face-quality proxy for
    PersonState.update_best()'s "best crop" ranking: a confidently-tracked
    body pose with the face turned away is a bad embedding source, and raw
    keypoint confidence alone can't tell the two apart (see tracking.py)."""
    r = matrix[:3, :3]
    return float(np.degrees(np.arctan2(-r[2, 0], np.sqrt(r[2, 1] ** 2 + r[2, 2] ** 2))))


def face_quality_from_yaw(yaw_deg: float) -> float:
    """1.0 at frontal, ->0 approaching a 90-degree profile. Simple cosine
    falloff -- not claimed to be a precise perceptual model, just enough to
    stop a confidently-tracked profile/turned-away frame from winning the
    "best embedding source" ranking purely on body-pose confidence."""
    return max(0.0, float(np.cos(np.radians(yaw_deg))))


def _eye_center_and_height(px_all, indices):
    pts = px_all[indices]
    cx  = int(pts[:, 0].mean())
    cy  = int(pts[:, 1].mean())
    eh  = int(pts[:, 1].max() - pts[:, 1].min())
    return pts, cx, cy, eh


def _render_canonical(px_all):
    canvas = np.full((CANONICAL_SIZE, CANONICAL_SIZE, 3), _BG, dtype=np.uint8)

    # 1. Face oval - filled skin tone
    cv2.fillPoly(canvas, [px_all[_FACE_OVAL]], _SKIN)

    # 2. Lips outer fill, then inner (cavity shows mouth-open/close)
    cv2.fillPoly(canvas, [px_all[_LIPS_OUTER]], _LIP)
    cv2.fillPoly(canvas, [px_all[_LIPS_INNER]], _LIP_IN)
    cv2.polylines(canvas, [px_all[_LIPS_OUTER]], isClosed=True,
                  color=_OUTLINE, thickness=1, lineType=cv2.LINE_AA)

    # 3. Eyes: white sclera fill → iris → pupil → specular dot
    for eye_idx in (_LEFT_EYE, _RIGHT_EYE):
        pts, cx, cy, eh = _eye_center_and_height(px_all, eye_idx)
        cv2.fillPoly(canvas, [pts], _SCLERA)
        if eh > 5:                                   # only draw iris when eye is open
            iris_r = max(int(eh * 0.44), 3)
            cv2.circle(canvas, (cx, cy), iris_r,     _IRIS,      -1)
            cv2.circle(canvas, (cx, cy), max(iris_r // 2, 2), _PUPIL, -1)
            # tiny specular highlight (upper-left of pupil)
            hx, hy = cx - max(iris_r // 4, 1), cy - max(iris_r // 4, 1)
            cv2.circle(canvas, (hx, hy), max(iris_r // 5, 1), _HIGHLIGHT, -1)
        # eyelid outline drawn on top of fill
        cv2.polylines(canvas, [pts], isClosed=True,
                      color=_OUTLINE, thickness=2, lineType=cv2.LINE_AA)

    # 4. Eyebrows - thick polyline reads as a filled brow stripe
    for brow_idx in (_LEFT_BROW, _RIGHT_BROW):
        cv2.polylines(canvas, [px_all[brow_idx]], isClosed=False,
                      color=_BROW, thickness=7, lineType=cv2.LINE_AA)

    # 5. Nose: bridge line + nostril circles
    cv2.polylines(canvas, [px_all[_NOSE_BRIDGE]], isClosed=False,
                  color=_NOSE_SHD, thickness=2, lineType=cv2.LINE_AA)
    nt = tuple(px_all[_NOSE_TIP])
    for n_idx in (_LEFT_NOSTRIL, _RIGHT_NOSTRIL):
        cv2.circle(canvas, tuple(px_all[n_idx]), 7, _NOSE_SHD, 2, cv2.LINE_AA)
    cv2.circle(canvas, nt, 4, _NOSE_SHD, -1, cv2.LINE_AA)

    # 6. Face oval outline (drawn last so it's clean over everything)
    cv2.polylines(canvas, [px_all[_FACE_OVAL]], isClosed=True,
                  color=_OUTLINE, thickness=2, lineType=cv2.LINE_AA)

    # 7. Gentle blur to smooth pixel-level jaggies
    canvas = cv2.GaussianBlur(canvas, (3, 3), 0)
    return canvas


class FaceCanonicalizer:
    def __init__(self, model_path: str = 'face_landmarker.task',
                 infer_size: int = 512,
                 min_detection_conf: float = 0.7,
                 min_presence_conf: float = 0.45):
        base_options = python.BaseOptions(model_asset_path=model_path)
        options = vision.FaceLandmarkerOptions(
            base_options=base_options,
            num_faces=1,
            running_mode=vision.RunningMode.IMAGE,
            min_face_detection_confidence=min_detection_conf,
            min_face_presence_confidence=min_presence_conf,
            output_facial_transformation_matrixes=True,
        )
        self._landmarker = vision.FaceLandmarker.create_from_options(options)
        self._infer_size = infer_size

    def get_canonical_face(self, face_crop: np.ndarray) -> np.ndarray | None:
        canonical, _yaw = self.get_canonical_face_and_yaw(face_crop)
        return canonical

    def get_canonical_face_and_yaw(self, face_crop: np.ndarray) -> tuple[np.ndarray | None, float | None]:
        small  = cv2.resize(face_crop, (self._infer_size, self._infer_size))
        rgb    = cv2.cvtColor(small, cv2.COLOR_BGR2RGB)
        result = self._landmarker.detect(
            mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)
        )
        if not result.face_landmarks:
            return None, None
        canonical = _render_canonical(_px(result.face_landmarks[0]))
        yaw = (yaw_from_transform(result.facial_transformation_matrixes[0])
               if result.facial_transformation_matrixes else None)
        return canonical, yaw

    def close(self):
        self._landmarker.close()
