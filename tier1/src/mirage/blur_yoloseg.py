import cv2
import numpy as np
import os

HULL_COLOR = (127, 127, 127)

_DEFAULT_PT   = "yolov8n-seg.pt"
_DEFAULT_ONNX = "yolov8n-seg.onnx"

_DEFAULT_INFER_SIZE = 320  # existing cached models (pre-dating per-size filenames) are this size


def _size_suffixed_path(model_name: str, infer_size: int) -> str:
    """
    yolo11n-seg-int8.onnx @ 320 -> yolo11n-seg-int8.onnx (unchanged, matches
    already-cached files from before this suffixing existed). Any other size
    -> yolo11n-seg-int8_256.onnx, so different infer_size values never share
    (and silently collide on) the same cache file.
    """
    if infer_size == _DEFAULT_INFER_SIZE or not model_name.endswith(".onnx"):
        return model_name
    return model_name[:-len(".onnx")] + f"_{infer_size}.onnx"


class YOLOSegBlur:
    """
    YOLO-seg (v8n or 11n) instance segmentation.
    Defaults to the ONNX export: measured ~2.5-3.4x faster than the
    PyTorch .pt checkpoint on CPU (onnxruntime CPUExecutionProvider vs
    torch CPU), with pixel-identical mask output (0.000% disagreement
    across a 300-frame comparison, both single- and multi-person clips,
    skip-n warped and full-inference frames alike) -- same weights, just
    a different execution graph. Same get_mask / apply_mask interface as
    SelfieSegBlur.
    """

    def __init__(self, model_name: str = None, infer_size: int = 320, conf: float = 0.4):
        from ultralytics import YOLO

        if model_name is None:
            model_name = _DEFAULT_ONNX
            print(f"  [YOLOSeg] Using ONNX model: {_DEFAULT_ONNX}")

        # ONNX-exported YOLO-seg has a STATIC input shape baked in at export
        # time (imgsz is not a runtime-adjustable param the way it is for the
        # .pt checkpoint) -- confirmed directly: passing a different imgsz to
        # a model exported at 320 raises InvalidArgument ("Got: 256 Expected:
        # 320"), it does not silently resize. So a model file exported at one
        # size cannot serve another; the requested infer_size is baked into
        # the cache filename below, so different sizes never collide on (or
        # silently reuse a stale) the same cache path.
        #
        # is_int8/pt_name are derived on the ORIGINAL (un-suffixed) name --
        # the .pt checkpoint is size-independent (imgsz is just an export-time
        # arg to it), so there's exactly one .pt per architecture regardless
        # of how many differently-sized .onnx exports get cached from it.
        is_int8 = model_name.endswith("-int8.onnx")
        pt_name = (model_name[:-len("-int8.onnx")] if is_int8 else model_name[:-len(".onnx")]) + ".pt"
        base_onnx = _size_suffixed_path(
            model_name[:-len("-int8.onnx")] + ".onnx" if is_int8 else model_name, infer_size
        )
        model_name = _size_suffixed_path(model_name, infer_size)

        # ONNX files aren't fetchable by URL the way .pt weights are (ultralytics
        # auto-downloads .pt from its GitHub releases); if the requested .onnx
        # is missing but the matching .pt is present or downloadable, export it
        # once so first-run on a fresh machine (e.g. the Pi) doesn't hard-fail.
        if model_name.endswith(".onnx") and not os.path.exists(model_name):
            if not os.path.exists(base_onnx):
                print(f"  [YOLOSeg] {base_onnx} not found -- exporting from {pt_name} (imgsz={infer_size}) ...")
                pt_model = YOLO(pt_name)  # auto-downloads .pt if missing
                exported_path = pt_model.export(format="onnx", imgsz=infer_size, simplify=True)
                if os.path.abspath(exported_path) != os.path.abspath(base_onnx):
                    os.replace(exported_path, base_onnx)
                print(f"  [YOLOSeg] Exported -> {base_onnx}")

            if is_int8:
                # Dynamic weight-only INT8 quantization. Measured slower than
                # FP32 ONNX on x86 (dequant/requant overhead outweighs
                # bandwidth savings without static activation calibration),
                # but ARM's INT8 SIMD path (NEON dot-product on ARMv8.2+,
                # e.g. Pi 5's Cortex-A76) can behave very differently --
                # worth measuring per-platform, not assuming from x86 results.
                from onnxruntime.quantization import quantize_dynamic, QuantType
                print(f"  [YOLOSeg] Quantizing {base_onnx} -> {model_name} (INT8, dynamic)...")
                quantize_dynamic(base_onnx, model_name, weight_type=QuantType.QUInt8)
                print(f"  [YOLOSeg] Quantized -> {model_name}")

        self._model      = YOLO(model_name)
        self._infer_size = infer_size
        self._conf       = conf

    def get_mask(self, frame: np.ndarray, infer_size: int = None):
        """
        Run YOLO-seg on frame, return combined bool mask (H×W).
        infer_size arg accepted for API compatibility but ignored.
        Returns None if no persons detected.
        """
        mask, _ = self.get_mask_and_boxes(frame)
        return mask

    def get_mask_and_boxes(self, frame: np.ndarray):
        """
        Same inference as get_mask(), but also returns the detector boxes
        YOLO-seg found internally (its own detect head runs before the mask
        head regardless -- this just surfaces that result instead of
        discarding it). Lets a caller reuse this one detection pass for
        pose estimation too, instead of running a second, separate detector
        (e.g. rtmlib's YOLOX-Nano) purely to get boxes RTMPose-T needs --
        confirmed via direct timing this was pure duplicated detection cost
        (YOLOX-Nano ~18ms + YOLO-seg's own internal detect+mask ~33ms, when
        one detection pass is all that's actually needed).

        Returns (mask_or_None, boxes) where boxes is an (N,4) float array
        [x1,y1,x2,y2] in full-frame pixel space, or an empty array if none
        detected.
        """
        h, w = frame.shape[:2]
        results = self._model(
            frame,
            imgsz=self._infer_size,
            conf=self._conf,
            classes=[0],
            verbose=False,
        )
        combined = np.zeros((h, w), dtype=bool)
        boxes = np.empty((0, 4), dtype=float)
        for r in results:
            if r.boxes is not None and len(r.boxes) > 0:
                boxes = r.boxes.xyxy.cpu().numpy()
            if r.masks is None:
                continue
            for mask_tensor in r.masks.data:
                m = mask_tensor.cpu().numpy()
                m_up = cv2.resize(m, (w, h), interpolation=cv2.INTER_NEAREST).astype(bool)
                combined |= m_up
        mask = combined if combined.any() else None
        return mask, boxes

    def apply_mask(self, frame: np.ndarray, mask: np.ndarray) -> np.ndarray:
        out = frame.copy()
        out[mask] = HULL_COLOR
        return out

    def blur_frame(self, frame: np.ndarray) -> np.ndarray:
        mask = self.get_mask(frame)
        return self.apply_mask(frame, mask) if mask is not None else frame
