import numpy as np
import cv2
import mediapipe as mp

COCO_NOSE      = 0
COCO_LEFT_EYE  = 1
COCO_RIGHT_EYE = 2

BODY_KPT_INDICES = list(range(0, 17))
BODY_CROP_PADDING = 20

HEAD_SCALE = 2.5

LK_PARAMS = dict(
    winSize  = (21, 21),
    maxLevel = 2,
    criteria = (cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 10, 0.03)
)


def euclidean(p1, p2) -> float:
    return float(np.linalg.norm(np.array(p1) - np.array(p2)))


def get_face_size_tier(inter_eye_px: float, far_thr: int, med_thr: int) -> str:
    if inter_eye_px < far_thr:
        return "far"
    elif inter_eye_px < med_thr:
        return "medium"
    return "close"


def get_movement_tier(disp: float, slow_thr: int, fast_thr: int) -> str:
    if disp < slow_thr:
        return "slow"
    elif disp < fast_thr:
        return "medium"
    return "fast"


def derive_face_crop(frame, kpts, scores, kpt_thr=0.3):
    h, w = frame.shape[:2]
    if (scores[COCO_NOSE] < kpt_thr or
            scores[COCO_LEFT_EYE] < kpt_thr or
            scores[COCO_RIGHT_EYE] < kpt_thr):
        return None, None, None, None, 0.0
    inter_eye_px = euclidean(kpts[COCO_LEFT_EYE], kpts[COCO_RIGHT_EYE])
    face_radius  = max(int(inter_eye_px * 2.2), 20)
    cx, cy = int(kpts[COCO_NOSE][0]), int(kpts[COCO_NOSE][1])
    x1 = max(cx - face_radius, 0)
    y1 = max(cy - face_radius, 0)
    x2 = min(cx + face_radius, w)
    y2 = min(cy + face_radius, h)
    crop = frame[y1:y2, x1:x2]
    if crop.size == 0:
        return None, None, None, None, 0.0
    return crop, x1, y1, (x2 - x1, y2 - y1), inter_eye_px


def derive_body_crop(frame, kpts, scores, kpt_thr=0.3, detector_bbox=None):
    """detector_bbox, if given, is (x1, y1, x2, y2) in frame coordinates --
    the person-detector's own box for this same person this frame (YOLOX-Nano
    det_model(), or yolo_seg's own detect head when a yoloseg* anonymizer is
    active -- see pipeline.py's last_scaled_bboxes, which is populated from
    whichever detector actually ran that frame). COCO-17 keypoints have no
    point above eye/nose level, so a keypoint-only box's top edge sits at
    eyebrow height -- confirmed on real restored-video output as a grey
    silhouette left over the head/hair, since that region was never captured
    at all. The detector's box DOES extend to the top of the head (a person
    detector has to box the whole visible person to detect one), so it's
    used outright in place of the keypoint-derived box when available --
    not unioned. This only affects the encrypted archive's crop quality
    (never the actual blur/anonymization region, which is driven separately
    by the segmentation mask/convex hull), so an occasional smaller crop
    from a partial/low-confidence detector box on some frame is an
    acceptable, low-stakes tradeoff for simpler logic."""
    h, w = frame.shape[:2]

    if detector_bbox is not None:
        x1, y1, x2, y2 = (int(v) for v in detector_bbox)
    else:
        visible_pts = []
        for idx in BODY_KPT_INDICES:
            if scores[idx] > kpt_thr:
                visible_pts.append([kpts[idx][0], kpts[idx][1]])
        if len(visible_pts) < 2:
            return None, 0, 0, 0, 0
        pts = np.array(visible_pts)
        x1 = int(pts[:, 0].min()) - BODY_CROP_PADDING
        y1 = int(pts[:, 1].min()) - BODY_CROP_PADDING
        x2 = int(pts[:, 0].max()) + BODY_CROP_PADDING
        y2 = int(pts[:, 1].max()) + BODY_CROP_PADDING

    x1 = max(x1, 0)
    y1 = max(y1, 0)
    x2 = min(x2, w)
    y2 = min(y2, h)
    crop = frame[y1:y2, x1:x2]
    if crop.size == 0:
        return None, 0, 0, 0, 0
    return crop, x1, y1, x2, y2


def compute_frame_confidence(scores, kpt_thr=0.3) -> float:
    visible = scores[scores > kpt_thr]
    return float(visible.mean()) if len(visible) > 0 else 0.0


def project_landmarks(landmarks_list, x_off, y_off, crop_w, crop_h):
    return [
        (int(x_off + lm.x * crop_w), int(y_off + lm.y * crop_h))
        for lm in landmarks_list
    ]


def draw_face_mesh_pts(frame, pts, color=(0, 255, 180), radius=1):
    for (x, y) in pts:
        cv2.circle(frame, (x, y), radius, color, -1)
