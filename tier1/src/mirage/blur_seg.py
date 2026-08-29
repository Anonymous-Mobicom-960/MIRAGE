import cv2
import numpy as np
import mediapipe as mp
from mediapipe.tasks import python
from mediapipe.tasks.python import vision

HULL_COLOR = (127, 127, 127)


class SelfieSegBlur:
    """
    MediaPipe ImageSegmenter (Tasks API, mediapipe>=0.10) drop-in replacement
    for convex hull blur. Uses selfie segmentation tflite models.

    model_path: path to .tflite model file
     - selfie_segmenter.tflite          (general, faster)
     - selfie_segmenter_landscape.tflite (landscape/full-body, more accurate)
    """

    def __init__(self, model_path: str, threshold: float = 0.5,
                 morph_kernel: int = 5):
        base_options = python.BaseOptions(model_asset_path=model_path)
        options = vision.ImageSegmenterOptions(
            base_options=base_options,
            output_category_mask=False,
            output_confidence_masks=True,
        )
        self._segmenter = vision.ImageSegmenter.create_from_options(options)
        self._threshold = threshold
        # Cleans up the mask post-threshold: opening strips small isolated
        # oversegmented blobs/protrusions (e.g. background texture wrongly
        # included), closing fills small holes/gaps in undersegmented
        # regions (e.g. a ragged sleeve edge). Kernel size trades cleanup
        # strength against eroding genuinely thin real structures (fingers,
        # hair wisps).
        self._morph_kernel = (
            cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (morph_kernel, morph_kernel))
            if morph_kernel > 0 else None
        )

    def get_mask(self, frame: np.ndarray, infer_size: int = None):
        """
        Run selfie segmentation on frame, return bool mask (H×W) at
        original frame resolution. infer_size (if given) downsizes the
        frame before inference; the float confidence field (not an
        already-thresholded mask) is then upsampled with bilinear
        interpolation before thresholding, so the boundary stays smooth
        instead of picking up nearest-neighbor blockiness. A morphological
        open+close pass then removes small speckle and fills small gaps.
        Returns None if no confidence mask was produced.
        """
        h, w = frame.shape[:2]
        small = cv2.resize(frame, (infer_size, infer_size)) if infer_size else frame
        rgb = cv2.cvtColor(small, cv2.COLOR_BGR2RGB)
        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)
        result = self._segmenter.segment(mp_image)
        if not result.confidence_masks:
            return None
        conf = result.confidence_masks[0].numpy_view().squeeze()
        if infer_size:
            conf = cv2.resize(conf, (w, h), interpolation=cv2.INTER_LINEAR)
        mask = (conf > self._threshold).astype(np.uint8)
        if self._morph_kernel is not None:
            mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, self._morph_kernel)
            mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, self._morph_kernel)
        return mask.astype(bool)

    def apply_mask(self, frame: np.ndarray, mask: np.ndarray) -> np.ndarray:
        out = frame.copy()
        out[mask] = HULL_COLOR
        return out

    def blur_frame(self, frame: np.ndarray) -> np.ndarray:
        mask = self.get_mask(frame)
        return self.apply_mask(frame, mask) if mask is not None else frame

    def close(self):
        self._segmenter.close()


def bbox_region_mask(bboxes, frame_h: int, frame_w: int, padding: int = 40) -> np.ndarray:
    """
    Build a bool mask (H×W) that is True only inside the union of the given
    [x1,y1,x2,y2] boxes, each expanded by `padding` px. Used to constrain a
    whole-frame segmentation mask (which has no notion of "where a person
    was actually detected") to the regions RTMPose actually found people in
   - this stops the segmenter from painting over background clutter
    (tree branches, textured walls, furniture) that it mistakes for a person.
    """
    region = np.zeros((frame_h, frame_w), dtype=bool)
    if bboxes is None:
        return region
    for bbox in bboxes:
        x1 = max(int(bbox[0]) - padding, 0)
        y1 = max(int(bbox[1]) - padding, 0)
        x2 = min(int(bbox[2]) + padding, frame_w)
        y2 = min(int(bbox[3]) + padding, frame_h)
        if x2 > x1 and y2 > y1:
            region[y1:y2, x1:x2] = True
    return region
