"""
Patches a threshold bug in rtmlib's YOLOX postprocessing.

rtmlib.tools.object_detection.yolox.YOLOX.postprocess() has two code paths:
 - ONNX without baked-in NMS: correctly filters by `self.score_thr`.
 - ONNX with baked-in NMS (our yolox_nano_8xb8-300e_humanart export):
    hardcodes `final_scores > 0.3`, silently ignoring the score_thr passed
    to Body(det_score_thr=...). This lets low-confidence spurious boxes
    (e.g. background clutter scoring ~0.3-0.5) through as "detected people".

This module monkeypatches YOLOX.postprocess so the baked-in-NMS branch
also respects self.score_thr, and applies_detector_patch() must be called
once before constructing rtmlib.Body().
"""

import numpy as np
from rtmlib.tools.object_detection import yolox as _yolox_mod

_PATCHED = False


def _patched_postprocess(self, outputs, ratio=1.0):
    if outputs.shape[-1] == 4 or outputs.shape[-1] > 5:
        return _ORIGINAL_POSTPROCESS(self, outputs, ratio)

    elif outputs.shape[-1] == 5:
        # onnx contains nms module - respect self.score_thr instead of
        # the upstream hardcoded 0.3.
        final_boxes, final_scores = outputs[0, :, :4], outputs[0, :, 4]
        final_boxes = final_boxes / ratio
        keep = final_scores > self.score_thr
        final_boxes = final_boxes[np.asarray(keep)]

    if self.mode == 'multiclass':
        final_cls_inds = np.zeros(len(final_boxes), dtype=int)
        return final_boxes, final_cls_inds
    elif self.mode == 'human':
        return final_boxes
    else:
        raise NotImplementedError(
            f"Mode must be 'human' or 'multiclass': {self.mode} is not supported."
        )


_ORIGINAL_POSTPROCESS = _yolox_mod.YOLOX.postprocess


def apply_detector_patch():
    """Idempotent. Call once before constructing rtmlib.Body()."""
    global _PATCHED
    if _PATCHED:
        return
    _yolox_mod.YOLOX.postprocess = _patched_postprocess
    _PATCHED = True
