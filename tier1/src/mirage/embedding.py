import os
import numpy as np
import cv2
import onnxruntime as ort

EDGEFACE_ONNX_PATH = "models/edgeface_xs_gamma_06.onnx"


class EmbeddingExtractor:
    """EdgeFace-XS (gamma = 0.6) face embedding via ONNX Runtime.

    Upstream: https://github.com/otroshi/edgeface -> checkpoints/edgeface_xs_gamma_06.pt,
    exported to ONNX here. Weights are CC BY-NC-SA 4.0 (non-commercial); see models/README.md.

    George, A., Ecabert, C., Otroshi Shahreza, H., Kotwal, K. and Marcel, S.
    "EdgeFace: Efficient Face Recognition Model for Edge Devices."
    IEEE Transactions on Biometrics, Behavior, and Identity Science, 2024.

    WHY THIS VARIANT AND NOT s-gamma-05: the paper's efficiency table reports 1.77 M parameters,
    and counted off the ONNX graphs xs-gamma-06 is 1,770,492 against s-gamma-05's 3,652,520, so
    1.77 M can only be xs-gamma-06. Upstream reports the same 1.77 M for XS (gamma = 0.6).
    """

    INPUT_SIZE = (112, 112)

    def __init__(self, model_path: str = EDGEFACE_ONNX_PATH):
        if not os.path.isfile(model_path):
            raise FileNotFoundError(
                f"EdgeFace ONNX not found at '{model_path}'.\n"
                "Run export_edgeface_onnx.py on a dev machine first."
            )
        sess_opts = ort.SessionOptions()
        sess_opts.intra_op_num_threads = 2
        sess_opts.inter_op_num_threads = 1
        sess_opts.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL

        self._session = ort.InferenceSession(
            model_path,
            sess_options=sess_opts,
            providers=["CPUExecutionProvider"],
        )
        self._input_name = self._session.get_inputs()[0].name
        print(f"[EdgeFace] Loaded {model_path} - "
              f"input '{self._input_name}' {self._session.get_inputs()[0].shape}")

    @staticmethod
    def preprocess(crop_bgr: np.ndarray) -> np.ndarray:
        img = cv2.resize(crop_bgr, EmbeddingExtractor.INPUT_SIZE,
                         interpolation=cv2.INTER_LINEAR)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0
        img = (img - 0.5) / 0.5
        img = img.transpose(2, 0, 1)[np.newaxis, ...]
        return np.ascontiguousarray(img, dtype=np.float32)

    @staticmethod
    def _l2_normalize(vec: np.ndarray) -> np.ndarray:
        norm = np.linalg.norm(vec)
        return vec if norm < 1e-10 else vec / norm

    def extract(self, crop_bgr: np.ndarray) -> np.ndarray | None:
        if crop_bgr is None or crop_bgr.size == 0:
            return None
        if crop_bgr.shape[0] < 20 or crop_bgr.shape[1] < 20:
            return None
        tensor = self.preprocess(crop_bgr)
        raw = self._session.run(None, {self._input_name: tensor})[0]
        return self._l2_normalize(raw[0])
