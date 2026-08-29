import numpy as np
import cv2
import urllib.request
import os

HULL_COLOR = (127, 127, 127)
MOBILESAM_WEIGHTS_URL = (
    "https://github.com/ChaoningZhang/MobileSAM/raw/master/weights/mobile_sam.pt"
)
MOBILESAM_DEFAULT_PATH = "models/mobile_sam.pt"


def download_weights(dest: str = MOBILESAM_DEFAULT_PATH):
    os.makedirs(os.path.dirname(dest), exist_ok=True)
    print(f"  Downloading MobileSAM weights -> {dest} ...")
    urllib.request.urlretrieve(MOBILESAM_WEIGHTS_URL, dest)
    print("  Done.")


class MobileSAMBlur:
    """
    MobileSAM (ViT-Tiny) with bounding-box prompts derived from RTMPose keypoints.
    One set_image() call per frame; one predict() call per detected person.
    """

    def __init__(self, checkpoint_path: str = MOBILESAM_DEFAULT_PATH, device: str = "cpu"):
        import torch
        from mobile_sam import sam_model_registry, SamPredictor

        if not os.path.exists(checkpoint_path):
            download_weights(checkpoint_path)

        sam = sam_model_registry["vit_t"](checkpoint=checkpoint_path)
        sam.to(device=device)
        sam.eval()
        self._predictor = SamPredictor(sam)

    def blur_frame(self, frame: np.ndarray, bboxes_xyxy) -> np.ndarray:
        """
        frame        : BGR frame (original resolution)
        bboxes_xyxy  : list/array of [x1, y1, x2, y2] in frame pixel coordinates
        """
        if bboxes_xyxy is None or len(bboxes_xyxy) == 0:
            return frame

        h, w = frame.shape[:2]
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        self._predictor.set_image(rgb)

        out = frame.copy()
        for bbox in bboxes_xyxy:
            x1 = int(np.clip(bbox[0], 0, w - 1))
            y1 = int(np.clip(bbox[1], 0, h - 1))
            x2 = int(np.clip(bbox[2], 0, w - 1))
            y2 = int(np.clip(bbox[3], 0, h - 1))
            if x2 <= x1 or y2 <= y1:
                continue
            masks, scores, _ = self._predictor.predict(
                box=np.array([x1, y1, x2, y2]),
                multimask_output=True,
            )
            best = int(np.argmax(scores))
            out[masks[best]] = HULL_COLOR

        return out


def bboxes_from_keypoints(all_keypoints, all_scores, frame_h, frame_w,
                           kpt_thr=0.3, padding=40):
    """
    Derive per-person bounding boxes from RTMPose keypoints.
    Used on both full and skip (optical-flow) frames so MobileSAM always
    gets a prompt even when the detector didn't run.
    """
    bboxes = []
    for kpts, scrs in zip(all_keypoints, all_scores):
        visible = [
            [kpts[j][0], kpts[j][1]]
            for j in range(len(kpts))
            if scrs[j] > kpt_thr
        ]
        if not visible:
            continue
        pts = np.array(visible)
        x1 = max(0,          pts[:, 0].min() - padding)
        y1 = max(0,          pts[:, 1].min() - padding)
        x2 = min(frame_w - 1, pts[:, 0].max() + padding)
        y2 = min(frame_h - 1, pts[:, 1].max() + padding)
        bboxes.append([x1, y1, x2, y2])
    return bboxes
