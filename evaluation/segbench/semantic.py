"""
segbench.semantic -- a cheap SEMANTIC person mask used as a privacy SAFETY-NET,
unioned with the instance-seg mask. Rationale (research defense-in-depth): a nano
instance segmenter MISSES people (small/occluded/low-conf), and a missed person's
face is a hard-gate violation. A semantic person segmenter can't give per-person
IDs, but it doesn't need to -- it just guarantees person-pixels get greyed even
when the instance model drops them. Torch-free (MediaPipe TFLite), CPU, fast.
"""
import numpy as np
import cv2


class SelfieSegNet:
    """MediaPipe SelfieSegmentation -> full-res boolean person mask.
    model_selection=0 is the general (256x256) model; better for full bodies than
    the landscape model. Output is upsampled to the frame size by MediaPipe."""

    def __init__(self, model_selection: int = 0, thresh: float = 0.5):
        import mediapipe as mp
        self._seg = mp.solutions.selfie_segmentation.SelfieSegmentation(
            model_selection=model_selection)
        self.thresh = thresh

    def person_mask(self, frame_bgr: np.ndarray) -> np.ndarray:
        res = self._seg.process(cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB))
        m = res.segmentation_mask
        if m is None:
            return np.zeros(frame_bgr.shape[:2], dtype=bool)
        if m.shape[:2] != frame_bgr.shape[:2]:
            m = cv2.resize(m, (frame_bgr.shape[1], frame_bgr.shape[0]))
        return m > self.thresh
