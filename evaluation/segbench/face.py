"""
segbench.face -- FaceGuard: a hard guarantee that a detected face is NEVER
revealed. Priority per the threat model: FACE must never show (absolute);
clothes/body may be slightly revealed (acceptable). So rather than inflating the
whole-body mask (halo everywhere), we detect faces directly and grey-fill each
one, unioned into the output. Cheap (MediaPipe FaceDetection ~few ms), and it
covers faces the body segmenter's edge misses (e.g. a nose/mouth protruding in
profile). The same detector used to AUDIT face leaks is used to CLOSE them.

In production this can reuse RTMPose head keypoints (nose/eyes/ears) instead of a
separate detector, at zero extra model cost -- but MediaPipe FaceDetection is used
here for a self-contained, strong-recall guard.
"""
import numpy as np
import cv2


class FaceGuard:
    def __init__(self, min_conf: float = 0.4, expand: float = 0.45, model_selection: int = 1):
        import mediapipe as mp
        # model_selection=1 = full-range model (better for varied/profile/far faces)
        self._fd = mp.solutions.face_detection.FaceDetection(
            model_selection=model_selection, min_detection_confidence=min_conf)
        self.expand = expand

    def face_fill(self, frame_bgr: np.ndarray) -> np.ndarray:
        """Boolean mask (H,W) covering every detected face as an expanded ellipse."""
        H, W = frame_bgr.shape[:2]
        res = self._fd.process(cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB))
        m = np.zeros((H, W), np.uint8)
        if res.detections:
            for d in res.detections:
                bb = d.location_data.relative_bounding_box
                w = bb.width * W; h = bb.height * H
                cx = bb.xmin * W + w / 2; cy = bb.ymin * H + h / 2
                ax = int(max(2, w / 2 * (1 + self.expand)))
                ay = int(max(2, h / 2 * (1 + self.expand)))
                cv2.ellipse(m, (int(cx), int(cy)), (ax, ay), 0, 0, 360, 1, -1)
        return m.astype(bool)
