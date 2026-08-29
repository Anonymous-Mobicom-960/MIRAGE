import numpy as np
import cv2
from .pose import euclidean, BODY_KPT_INDICES, COCO_NOSE, COCO_LEFT_EYE, COCO_RIGHT_EYE

FACE_OVAL_INDICES = [
    10,  338, 297, 332, 284, 251, 389, 356, 454,
    323, 361, 288, 397, 365, 379, 378, 400, 377,
    152, 148, 176, 149, 150, 136, 172,  58, 132,
     93, 234, 127, 162,  21,  54, 103,  67, 109
]

HULL_COLOR = (127, 127, 127)
HEAD_SCALE = 2.5


def _pad_hull(hull, center, padding_px):
    pts        = hull.astype(float).reshape(-1, 2)
    directions = pts - center
    norms      = np.linalg.norm(directions, axis=1, keepdims=True)
    norms[norms == 0] = 1
    padded = pts + (directions / norms) * padding_px
    return padded.astype(np.int32).reshape(-1, 1, 2)


def compute_adaptive_padding(kpts, scores, frame_h, frame_w, kpt_thr=0.3):
    LEFT_SHOULDER  = 5
    RIGHT_SHOULDER = 6
    LEFT_HIP       = 11
    RIGHT_HIP      = 12

    if (scores[LEFT_SHOULDER] > kpt_thr and scores[RIGHT_SHOULDER] > kpt_thr):
        shoulder_width = euclidean(kpts[LEFT_SHOULDER], kpts[RIGHT_SHOULDER])
        return int(np.clip(int(shoulder_width * 0.40), 20, 100))

    visible_pts = []
    for idx in BODY_KPT_INDICES:
        if scores[idx] > kpt_thr:
            visible_pts.append([kpts[idx][0], kpts[idx][1]])

    if len(visible_pts) >= 2:
        pts  = np.array(visible_pts)
        diag = euclidean(
            [pts[:, 0].min(), pts[:, 1].min()],
            [pts[:, 0].max(), pts[:, 1].max()]
        )
        return int(np.clip(int(diag * 0.12), 20, 100))

    return 30


def blur_all_persons(frame, all_keypoints, all_scores,
                     all_face_mesh_pts=None, kpt_thr=0.3):
    if all_keypoints is None or len(all_keypoints) == 0:
        return frame

    h, w = frame.shape[:2]

    LEFT_SHOULDER  = 5
    RIGHT_SHOULDER = 6
    LEFT_HIP       = 11
    RIGHT_HIP      = 12

    for i in range(len(all_keypoints)):
        kpts = all_keypoints[i]
        scrs = all_scores[i]

        padding = compute_adaptive_padding(kpts, scrs, h, w, kpt_thr)
        visible_pts = []

        for idx in BODY_KPT_INDICES:
            if scrs[idx] > kpt_thr:
                x = int(np.clip(kpts[idx][0], 0, w - 1))
                y = int(np.clip(kpts[idx][1], 0, h - 1))
                visible_pts.append([x, y])

        face_mesh_pts = (
            all_face_mesh_pts[i]
            if all_face_mesh_pts is not None and i < len(all_face_mesh_pts)
            else None
        )

        if face_mesh_pts is not None and len(face_mesh_pts) >= 468:
            for idx in FACE_OVAL_INDICES:
                x = int(np.clip(face_mesh_pts[idx][0], 0, w - 1))
                y = int(np.clip(face_mesh_pts[idx][1], 0, h - 1))
                visible_pts.append([x, y])
        else:
            nose_s  = scrs[COCO_NOSE]
            leye_s  = scrs[COCO_LEFT_EYE]
            reye_s  = scrs[COCO_RIGHT_EYE]
            if nose_s > kpt_thr and leye_s > kpt_thr and reye_s > kpt_thr:
                inter_eye = float(np.linalg.norm(
                    np.array([kpts[COCO_LEFT_EYE][0], kpts[COCO_LEFT_EYE][1]]) -
                    np.array([kpts[COCO_RIGHT_EYE][0], kpts[COCO_RIGHT_EYE][1]])
                ))
                head_top_x = int(np.clip(kpts[COCO_NOSE][0], 0, w - 1))
                head_top_y = int(np.clip(
                    kpts[COCO_NOSE][1] - inter_eye * HEAD_SCALE, 0, h - 1
                ))
                visible_pts.append([head_top_x, head_top_y])

        if scrs[LEFT_SHOULDER] > kpt_thr and scrs[RIGHT_SHOULDER] > kpt_thr:
            shoulder_width = euclidean(kpts[LEFT_SHOULDER], kpts[RIGHT_SHOULDER])
            expand = shoulder_width * 0.25
            mid_y  = int((kpts[LEFT_SHOULDER][1] + kpts[RIGHT_SHOULDER][1]) / 2)
            visible_pts.append([max(int(kpts[LEFT_SHOULDER][0]  - expand), 0), mid_y])
            visible_pts.append([min(int(kpts[RIGHT_SHOULDER][0] + expand), w - 1), mid_y])

        if scrs[LEFT_HIP] > kpt_thr and scrs[RIGHT_HIP] > kpt_thr:
            hip_width = euclidean(kpts[LEFT_HIP], kpts[RIGHT_HIP])
            expand = hip_width * 0.20
            mid_y  = int((kpts[LEFT_HIP][1] + kpts[RIGHT_HIP][1]) / 2)
            visible_pts.append([max(int(kpts[LEFT_HIP][0]  - expand), 0), mid_y])
            visible_pts.append([min(int(kpts[RIGHT_HIP][0] + expand), w - 1), mid_y])

        if len(visible_pts) >= 4:
            pts         = np.array(visible_pts, dtype=np.int32)
            hull        = cv2.convexHull(pts.reshape(-1, 1, 2))
            center      = pts.mean(axis=0)
            hull_padded = _pad_hull(hull, center, padding)

            hull_area  = cv2.contourArea(hull_padded)
            frame_area = h * w
            if hull_area < frame_area * 0.005:
                x1 = max(int(pts[:, 0].min()) - padding, 0)
                y1 = max(int(pts[:, 1].min()) - padding, 0)
                x2 = min(int(pts[:, 0].max()) + padding, w)
                y2 = min(int(pts[:, 1].max()) + padding, h)
                cv2.rectangle(frame, (x1, y1), (x2, y2), HULL_COLOR, -1)
            else:
                cv2.fillConvexPoly(frame, hull_padded, HULL_COLOR)

        elif len(visible_pts) >= 1:
            pts = np.array(visible_pts)
            x1  = max(int(pts[:, 0].min()) - padding, 0)
            y1  = max(int(pts[:, 1].min()) - padding, 0)
            x2  = min(int(pts[:, 0].max()) + padding, w)
            y2  = min(int(pts[:, 1].max()) + padding, h)
            cv2.rectangle(frame, (x1, y1), (x2, y2), HULL_COLOR, -1)

    return frame
