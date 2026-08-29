import os
import cv2
import numpy as np

GENDER_ONNX_PATH = os.path.join("models", "insightface", "genderage.onnx")

GENDER_LABELS = ["Female", "Male"]  # index order verified against insightface/app/common.py: gender==1 -> Male

# InsightFace's real preprocessing (insightface.utils.face_align.norm_crop)
# warps 5 detected landmarks (both eyes, nose, both mouth corners) onto
# these fixed destination points in a 112x112 reference frame via a fitted
# similarity transform -- this is what genderage.onnx was actually trained
# on. RTMPose only gives us 3 of those 5 points (both eyes + nose, COCO
# indices 0/1/2), not the mouth corners, but 3 point-correspondences are
# still enough to fit a similarity transform (rotation + uniform scale +
# translation, 4 degrees of freedom) -- see estimate_align_matrix() below.
# Values taken directly from insightface.utils.face_align.arcface_dst,
# rows [left_eye, right_eye, nose] only.
_ARCFACE_DST_3PT = np.array([
    [38.2946, 51.6963],
    [73.5318, 51.5014],
    [56.0252, 71.7366],
], dtype=np.float32)


class GenderClassifier:
    """InsightFace genderage.onnx (buffalo_l pack, MobileNet-0.25 backbone),
    run via InsightFace's own insightface.model_zoo.attribute.Attribute
    class rather than a hand-rolled preprocessing pipeline.

    Chosen over the earlier Levi & Hassner (gender_googlenet.onnx) model
    after an empirical comparison (scripts/diag_gender_model_compare.py,
    scratch/meeting_prep/diag_gender_model_compare_results.json): this
    model is ~1-6ms/crop vs. 90-2000ms/crop for Levi & Hassner on the same
    hardware, and gender_googlenet.onnx turned out to be a documented
    misnomer -- it's the Levi & Hassner 2015 3-conv AlexNet-derivative, not
    GoogLeNet, per the original OpenCV age-gender sample.

    Reusing Attribute directly (not a reimplementation) sidesteps two
    subtle bugs already found and fixed once during diagnostics: (1) a
    double-translation bug in a hand-rolled alignment matrix that produced
    an all-black warped crop, and (2) assuming the wrong input_mean/
    input_std normalization constants -- Attribute.__init__ inspects the
    ONNX graph itself to pick the right constants per checkpoint (0.0/1.0
    for this file, not the commonly-assumed 127.5/128.0).

    Attribute.get() only returns (argmax_gender, age) -- no confidence.
    predict() below duplicates its forward pass (same alignment, same
    input_mean/input_std/input_size Attribute.__init__ already resolved)
    only to keep the pre-argmax logits long enough to softmax them into a
    real confidence, which the manifest's per-slot vote-aggregation
    (pipeline.py) weights by.
    """

    def __init__(self, model_path: str = GENDER_ONNX_PATH):
        from insightface.model_zoo.attribute import Attribute
        if not os.path.isfile(model_path):
            raise FileNotFoundError(
                f"Gender classifier ONNX not found at '{model_path}'.\n"
                "Expected models/insightface/genderage.onnx."
            )
        self._attr = Attribute(model_file=model_path)
        print(f"[GenderClassifier] Loaded {model_path} (InsightFace genderage, "
              f"input_mean={self._attr.input_mean} input_std={self._attr.input_std})")

    def predict_from_keypoints(self, frame: np.ndarray, left_eye, right_eye, nose) -> tuple[str, float] | None:
        """Preferred entry point: aligns directly off the 3 raw RTMPose
        keypoints (COCO left_eye/right_eye/nose) against the ORIGINAL frame,
        via a proper similarity transform (rotation + scale + translation)
        fit to _ARCFACE_DST_3PT -- not the crude bbox-center-and-scale
        fallback predict() below uses on an already-cropped, axis-aligned
        square. Rotates a tilted head upright and frames the face the same
        way InsightFace's own 5-point norm_crop does (minus the 2 mouth
        points we don't have), instead of relying on derive_face_crop's
        nose-centered square, which was tuned for "reliably contains the
        whole face" rather than "matches training-time alignment."

        left_eye/right_eye here are COCO ANATOMICAL left/right (the
        subject's own left/right, so "left_eye" appears on the image's
        RIGHT side when facing the camera) -- same convention derive_face_
        crop/pose.py already use. _ARCFACE_DST_3PT's row order is IMAGE-
        left/right (row 0 = smaller x = appears on the image's left side),
        confirmed empirically: feeding COCO order straight through
        (anatomical-left -> dst row 0) fit a transform with a ~83 degree
        rotation component for an upright, roughly-frontal face -- an
        obviously-wrong result that reproduced as a ~90-degree-rotated,
        mis-cropped output image during verification (scripts/
        verify_gender_100.py). Swapping fixed it: rotation term drops to
        near-zero for the same frontal-face input. Do not "simplify" this
        swap away without re-verifying against real images -- it looks like
        it shouldn't matter but empirically does.

        Requires the ORIGINAL, uncropped frame (not a pre-cropped output of
        derive_face_crop) since it needs the keypoints' true pixel
        coordinates to fit the transform."""
        if frame is None or frame.size == 0:
            return None

        attr = self._attr
        target = attr.input_size[0]
        # _ARCFACE_DST_3PT is calibrated for a 112x112 reference frame (see
        # insightface.utils.face_align.estimate_norm's own ratio scaling) --
        # rescale to whatever size this checkpoint actually declares
        # (genderage.onnx uses 96x96, not 112x112) so alignment lands
        # correctly regardless of the model's input size.
        dst = _ARCFACE_DST_3PT * (target / 112.0)
        src = np.array([right_eye, left_eye, nose], dtype=np.float32)  # swapped -- see docstring
        M, _ = cv2.estimateAffinePartial2D(src, dst, method=cv2.LMEDS)
        if M is None:
            return None
        aligned = cv2.warpAffine(frame, M, (target, target), borderValue=0.0)

        return self._infer(aligned)

    def predict(self, face_crop_bgr: np.ndarray, bbox_in_crop=None) -> tuple[str, float] | None:
        """Fallback entry point for callers that only have an already-cropped
        face image and no raw keypoints (e.g. tracking.py's flush_to_disk(),
        which only retains best_face_crop, not the keypoints that produced
        it). Uses a crude bbox-center-and-scale alignment with NO rotation
        correction -- prefer predict_from_keypoints() wherever the caller
        has the original frame + keypoints available; see that method's
        docstring for why this one is weaker."""
        if face_crop_bgr is None or face_crop_bgr.size == 0:
            return None

        from insightface.utils import face_align

        h, w = face_crop_bgr.shape[:2]
        if bbox_in_crop is None:
            bbox_in_crop = (0, 0, w, h)
        bbox = np.array(bbox_in_crop, dtype=np.float32)

        attr = self._attr
        bw, bh = (bbox[2] - bbox[0]), (bbox[3] - bbox[1])
        center = (bbox[2] + bbox[0]) / 2, (bbox[3] + bbox[1]) / 2
        scale = attr.input_size[0] / (max(bw, bh) * 1.5)
        aimg, _ = face_align.transform(face_crop_bgr, center, attr.input_size[0], scale, 0)

        return self._infer(aimg)

    def _infer(self, aligned_bgr: np.ndarray) -> tuple[str, float] | None:
        """Shared forward pass for an already-aligned image (any size --
        blobFromImage below reads its actual shape, doesn't assume
        attr.input_size)."""
        attr = self._attr
        input_size = tuple(aligned_bgr.shape[0:2][::-1])
        blob = cv2.dnn.blobFromImage(
            aligned_bgr, 1.0 / attr.input_std, input_size,
            (attr.input_mean, attr.input_mean, attr.input_mean), swapRB=True,
        )
        pred = attr.session.run(attr.output_names, {attr.input_name: blob})[0][0]
        if pred is None or len(pred) != 3:
            return None

        logits = pred[:2].astype(np.float64)
        probs = np.exp(logits - logits.max())
        probs = probs / probs.sum()
        idx = int(np.argmax(probs))
        return GENDER_LABELS[idx], float(probs[idx])
