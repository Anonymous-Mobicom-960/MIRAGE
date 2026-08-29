"""
Parametric face canonicalizer (v2).

Instead of re-plotting raw landmark positions every frame (v1's approach),
this extracts ~12 identity-free expression scalars -- mouth open/width,
smile, eye openness L/R, brow raise L/R, gaze x/y, head roll/yaw/pitch --
in a coordinate frame normalized by inter-eye distance and rotated upright,
then renders them onto a fixed synthetic template. Only dynamic expression
change crosses into the signal; a person's resting geometry (including
their neutral mouth curvature, itself a soft identity cue) does not, once
a clip-level smile baseline has been subtracted via set_smile_baseline().

Visual rendering reuses the v1 template's drawing style (filled lips with
an inner cavity, sclera/iris/pupil/highlight, thick brow polylines, nose
bridge + nostrils, warm-gray background) so the two are visually
comparable side by side; only the *drive signal* differs.
"""

import cv2
import numpy as np
import mediapipe as mp
from mediapipe.tasks import python
from mediapipe.tasks.python import vision

from .face_canonical import (
    CANONICAL_SIZE, _BG, _SKIN, _OUTLINE, _BROW, _SCLERA, _IRIS, _PUPIL,
    _HIGHLIGHT, _LIP, _LIP_IN, _NOSE_SHD,
)

# Landmark indices, MediaPipe 468/478-pt face mesh topology.
_EYE_OUTER_R, _EYE_OUTER_L = 33, 263      # roll axis + scale reference
_EYE_CENTER_R, _EYE_CENTER_L = 133, 362   # gaze reference centers
_EYELID_TOP_R, _EYELID_BOT_R = 159, 145
_EYELID_TOP_L, _EYELID_BOT_L = 386, 374
_BROW_R, _BROW_L = 105, 334
_MOUTH_TOP_IN, _MOUTH_BOT_IN = 13, 14
_MOUTH_R, _MOUTH_L = 61, 291
_NOSE_TIP = 1
_IRIS_R, _IRIS_L = 468, 473

N_PARAMS = 12
# param vector indices, for readability at call sites
P_MOUTH_OPEN, P_MOUTH_W, P_SMILE = 0, 1, 2
P_EYE_R, P_EYE_L = 3, 4
P_BROW_R, P_BROW_L = 5, 6
P_YAW, P_PITCH, P_ROLL = 7, 8, 9
P_GAZE_X, P_GAZE_Y = 10, 11


def _extract_scalars(landmarks, w: int, h: int) -> np.ndarray:
    P = np.array([[lm.x * w, lm.y * h] for lm in landmarks], dtype=np.float32)
    eR_o, eL_o = P[_EYE_OUTER_R], P[_EYE_OUTER_L]
    d = np.linalg.norm(eL_o - eR_o) + 1e-6
    ec = (eR_o + eL_o) / 2.0
    roll = np.arctan2(eL_o[1] - eR_o[1], eL_o[0] - eR_o[0])
    c, s = np.cos(-roll), np.sin(-roll)
    R = np.array([[c, -s], [s, c]], dtype=np.float32)
    Q = (P - ec) @ R.T / d  # upright, scale-normalized

    def g(i):
        return Q[i]

    mouth_open = max(0.0, g(_MOUTH_BOT_IN)[1] - g(_MOUTH_TOP_IN)[1])
    mouth_w    = np.linalg.norm(g(_MOUTH_L) - g(_MOUTH_R))
    smile      = ((g(_MOUTH_TOP_IN)[1] + g(_MOUTH_BOT_IN)[1]) / 2.0) \
                 - ((g(_MOUTH_R)[1] + g(_MOUTH_L)[1]) / 2.0)
    eye_r      = max(0.0, g(_EYELID_BOT_R)[1] - g(_EYELID_TOP_R)[1])
    eye_l      = max(0.0, g(_EYELID_BOT_L)[1] - g(_EYELID_TOP_L)[1])
    brow_r     = g(_EYELID_TOP_R)[1] - g(_BROW_R)[1]
    brow_l     = g(_EYELID_TOP_L)[1] - g(_BROW_L)[1]
    nose       = g(_NOSE_TIP)
    yaw, pitch = nose[0], nose[1]

    gaze = np.zeros(2, dtype=np.float32)
    if len(P) > _IRIS_L:
        ir_r, ir_l = Q[_IRIS_R], Q[_IRIS_L]
        c_r = (g(_EYE_OUTER_R) + g(_EYE_CENTER_R)) / 2.0
        c_l = (g(_EYE_CENTER_L) + g(_EYE_OUTER_L)) / 2.0
        gaze = ((ir_r - c_r) + (ir_l - c_l)) / 2.0

    return np.array([mouth_open, mouth_w, smile, eye_r, eye_l,
                      brow_r, brow_l, yaw, pitch, roll,
                      gaze[0], gaze[1]], dtype=np.float32)


def _rot_fn(center, roll):
    a = roll * 0.7
    ca, sa = np.cos(a), np.sin(a)

    def rot(px, py):
        dx, dy = px - center[0], py - center[1]
        return (int(center[0] + ca * dx - sa * dy),
                int(center[1] + sa * dx + ca * dy))
    return rot


def _render_canonical_v2(q: np.ndarray, gain: float,
                          size: int = CANONICAL_SIZE) -> np.ndarray:
    S = size
    canvas = np.full((S, S, 3), _BG, dtype=np.uint8)

    mo, mw, sm  = q[P_MOUTH_OPEN], q[P_MOUTH_W], q[P_SMILE]
    eR, eL      = q[P_EYE_R], q[P_EYE_L]
    bR, bL      = q[P_BROW_R], q[P_BROW_L]
    yaw, pitch, roll = q[P_YAW], q[P_PITCH], q[P_ROLL]
    gx, gy      = q[P_GAZE_X], q[P_GAZE_Y]

    shx = int(np.clip(yaw * gain, -0.25, 0.25) * S * 0.6)
    shy = int(np.clip((pitch - 0.45) * gain, -0.2, 0.2) * S * 0.5)
    fc  = (S // 2 + shx // 2, int(S * 0.52) + shy // 2)
    rot = _rot_fn(fc, roll)

    face_axes = (int(S * 0.30), int(S * 0.40))
    cv2.ellipse(canvas, fc, face_axes, 0, 0, 360, _SKIN, -1)

    # eyes: sclera fill, iris/pupil/highlight when open, eyelid outline
    ey = fc[1] - int(S * 0.08)
    ew = int(S * 0.075)
    for side, eopen, brow_off in ((-1, eR, bR), (1, eL, bL)):
        exc = fc[0] + side * int(S * 0.12)
        eh  = int(np.clip(eopen * gain * S * 0.6, 2, S * 0.055))
        c0  = rot(exc, ey)
        eye_pts = cv2.ellipse2Poly(c0, (ew, eh), 0, 0, 360, 10)
        cv2.fillPoly(canvas, [eye_pts], _SCLERA)
        if eh > int(S * 0.012):
            iris_r = max(int(eh * 0.85), 3)
            pup = (c0[0] + int(np.clip(gx * gain, -0.08, 0.08) * S * 1.2),
                   c0[1] + int(np.clip(gy * gain, -0.06, 0.06) * S * 1.2))
            cv2.circle(canvas, pup, iris_r, _IRIS, -1)
            cv2.circle(canvas, pup, max(iris_r // 2, 2), _PUPIL, -1)
            hx, hy = pup[0] - max(iris_r // 4, 1), pup[1] - max(iris_r // 4, 1)
            cv2.circle(canvas, (hx, hy), max(iris_r // 5, 1), _HIGHLIGHT, -1)
        cv2.polylines(canvas, [eye_pts], isClosed=True,
                      color=_OUTLINE, thickness=2, lineType=cv2.LINE_AA)

        boff = int(S * 0.055 + np.clip((brow_off - 0.10) * gain, -0.06, 0.10) * S)
        b0 = rot(exc, ey - boff)
        cv2.ellipse(canvas, b0, (ew + 4, max(3, S // 90)), 0, 200, 340,
                    _BROW, max(4, S // 70), cv2.LINE_AA)

    # nose: bridge + nostrils, fixed (subtle, doesn't carry identity signal)
    nx, ny = rot(fc[0], fc[1] + int(S * 0.05))
    cv2.line(canvas, (nx, ny - int(S * 0.05)), (nx, ny + int(S * 0.03)),
              _NOSE_SHD, 2, cv2.LINE_AA)
    for nside in (-1, 1):
        ncx, ncy = rot(fc[0] + nside * int(S * 0.03), ny + int(S * 0.03))
        cv2.circle(canvas, (ncx, ncy), 7, _NOSE_SHD, 2, cv2.LINE_AA)

    # mouth: filled lips with inner cavity when open, else a closed-lip curve
    #
    # mw/mo scaling below is calibrated against observed value ranges on
    # real test clips (scripts/test_face_canon_v2.py), not guessed: over
    # 200 sampled frames, mouth_open ranged 0.005-0.142 (median 0.048) and
    # mouth_width ranged 0.51-0.71. Mapping those through gain and pixel
    # multipliers picked without that data saturated both at their clamp
    # ceilings for nearly every frame regardless of actual mouth state --
    # width was pinned at max width the whole clip, and the closed-mouth
    # median already exceeded the open-mouth render threshold.
    my = fc[1] + int(S * 0.17)
    m0 = rot(fc[0], my)
    MW_LO, MW_HI = 0.48, 0.78
    MO_LO, MO_HI = 0.02, 0.16
    mw_t = np.clip((mw - MW_LO) / (MW_HI - MW_LO), 0.0, 1.0)
    mhw  = int(S * (0.08 + 0.09 * mw_t))
    curve = int(np.clip(sm * gain, -0.08, 0.08) * S * 1.5)
    if mo >= MO_LO:
        mo_t = np.clip((mo - MO_LO) / (MO_HI - MO_LO) * gain, 0.0, 1.0)
        mhh  = int(S * (0.02 + 0.10 * mo_t))
        outer_pts = cv2.ellipse2Poly(m0, (mhw, mhh), 0, 0, 360, 10)
        inner_pts = cv2.ellipse2Poly(m0, (int(mhw * 0.7), int(mhh * 0.55)), 0, 0, 360, 10)
        cv2.fillPoly(canvas, [outer_pts], _LIP)
        cv2.fillPoly(canvas, [inner_pts], _LIP_IN)
        cv2.polylines(canvas, [outer_pts], isClosed=True,
                      color=_OUTLINE, thickness=1, lineType=cv2.LINE_AA)
    else:
        pts = np.array([[m0[0] - mhw, m0[1] - curve],
                         [m0[0], m0[1] + curve],
                         [m0[0] + mhw, m0[1] - curve]], np.int32)
        cv2.polylines(canvas, [pts], False, _LIP, max(3, S // 90), cv2.LINE_AA)

    cv2.ellipse(canvas, fc, face_axes, 0, 0, 360, _OUTLINE, 2, cv2.LINE_AA)
    canvas = cv2.GaussianBlur(canvas, (3, 3), 0)
    return canvas


class FaceCanonicalizerV2:
    def __init__(self, model_path: str = 'face_landmarker.task',
                 infer_size: int = 512,
                 min_detection_conf: float = 0.7,
                 min_presence_conf: float = 0.45,
                 expression_gain: float = 1.2,
                 smoothing: float = 0.5):
        base_options = python.BaseOptions(model_asset_path=model_path)
        options = vision.FaceLandmarkerOptions(
            base_options=base_options,
            num_faces=1,
            running_mode=vision.RunningMode.IMAGE,
            min_face_detection_confidence=min_detection_conf,
            min_face_presence_confidence=min_presence_conf,
        )
        self._landmarker = vision.FaceLandmarker.create_from_options(options)
        self._infer_size = infer_size
        self._gain = expression_gain
        self._smoothing = smoothing
        self._smile_baseline = 0.0
        self._prev_raw = None
        self._prev_smoothed = None

    def set_smile_baseline(self, value: float):
        self._smile_baseline = float(value)

    def extract_params(self, face_crop: np.ndarray) -> np.ndarray | None:
        small  = cv2.resize(face_crop, (self._infer_size, self._infer_size))
        rgb    = cv2.cvtColor(small, cv2.COLOR_BGR2RGB)
        result = self._landmarker.detect(
            mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)
        )
        if not result.face_landmarks:
            return None
        return _extract_scalars(result.face_landmarks[0], self._infer_size, self._infer_size)

    def render(self, params: np.ndarray) -> np.ndarray:
        q = params.copy()
        q[P_SMILE] -= self._smile_baseline
        if self._prev_smoothed is not None and self._smoothing > 0:
            q = self._smoothing * self._prev_smoothed + (1 - self._smoothing) * q
        self._prev_smoothed = q
        return _render_canonical_v2(q, self._gain)

    def get_canonical_face(self, face_crop: np.ndarray) -> np.ndarray | None:
        """Single-pass convenience path (streaming use, no clip-wide smile
        baseline available yet -- smile is rendered uncorrected unless
        set_smile_baseline() was called beforehand)."""
        params = self.extract_params(face_crop)
        if params is None:
            if self._prev_raw is None:
                return None
            params = self._prev_raw
        else:
            self._prev_raw = params
        return self.render(params)

    def close(self):
        self._landmarker.close()
