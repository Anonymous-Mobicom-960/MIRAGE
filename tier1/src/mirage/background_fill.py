"""
Tier 2B-1 background fill: replaces the grey-filled person-shaped hole in
the anonymized video with an inpainted reconstruction of what the
background likely looks like behind them.

Per the plan's corrected design (no temporal-median pre-pass): each frame
is inpainted independently. Two implementations:

  - BackgroundFiller: cv2.inpaint, CPU-trivial, no model file. Guaranteed
    fast (milliseconds/frame) but visibly lower quality on large holes
    (streaking/smearing -- no learned prior, just extends nearby
    texture/color along gradients). The shipping fallback per the plan.

  - LamaBackgroundFiller: the real big-lama model, verified this session
    against the exact algorithm the project's own ComfyUI
    INPAINT_InpaintWithModel node uses (Acly/comfyui-inpaint-nodes),
    confirmed against the live RunPod pod's installed source. Genuinely
    good quality once configured correctly -- LaMa's native/trained
    working resolution is 256x256 (confirmed via the WACV paper: trained
    on 256x256 crops), NOT an arbitrary speed shortcut; running it at
    other resolutions measurably degrades output. ~1.5-2s/frame on a
    desktop CPU at 256x256 -- not yet verified on mobile hardware or
    exported to a mobile-runnable format (ONNX/TFLite); see
    scripts/lama_onnx_export.py for that spike.

Runs on-device in the real system (Tier 2B-1, Android/Kotlin); these
Python classes are the reference implementation, used for testing/
evaluation and as the source of truth before porting/exporting.
"""
import os

import cv2
import numpy as np


class BackgroundFiller:
    """
    Wraps cv2.inpaint for per-frame background reconstruction.

    method: cv2.INPAINT_TELEA (fast marching, generally better edge
        continuation) or cv2.INPAINT_NS (Navier-Stokes, sometimes better
        for smooth/uniform regions). TELEA is the default -- our masks are
        large single-connected-region holes (a whole person), not scattered
        small defects, and TELEA tends to handle sizeable regions more
        cleanly.
    radius: inpainting neighborhood radius in pixels. Larger radius pulls
        from a wider surrounding area per pixel -- more context, but
        blurrier/costlier. 3-5 is cv2's own typical default range; larger
        masks (a whole person silhouette) benefit from a slightly larger
        radius than cv2's default of 3.
    """

    def __init__(self, method: int = cv2.INPAINT_TELEA, radius: int = 5):
        self._method = method
        self._radius = radius

    def fill_frame(self, frame_bgr: np.ndarray, mask_bool: np.ndarray) -> np.ndarray:
        """
        frame_bgr: H×W×3 uint8, the anonymized (grey-filled) frame.
        mask_bool: H×W bool, True where the person (grey fill) is -- the
            region to reconstruct.
        Returns: H×W×3 uint8, background reconstructed where the mask was.
        """
        if not mask_bool.any():
            return frame_bgr.copy()
        mask_u8 = (mask_bool.astype(np.uint8)) * 255
        return cv2.inpaint(frame_bgr, mask_u8, self._radius, self._method)

    def fill_video(self, video_path: str, mask_path: str, output_path: str,
                    progress_callback=None) -> None:
        """
        Fills every frame of video_path using the corresponding frame of
        mask_path (same frame count/resolution, as produced by
        pipeline.py's dense export: output_rtm.mp4 + mask.mp4), writes the
        result to output_path.

        progress_callback, if given, is called as (frame_idx, total_frames)
        after each frame -- lets a caller (e.g. an Android bridge, or a
        future concurrent-with-cloud-generation scheduler per the plan's
        corrected 2B-1/2B-2 concurrency design) report progress without
        this function needing to know about UI or threading.
        """
        cap_v = cv2.VideoCapture(video_path)
        cap_m = cv2.VideoCapture(mask_path)
        if not cap_v.isOpened() or not cap_m.isOpened():
            raise IOError(f"Could not open {video_path} or {mask_path}")

        fps = cap_v.get(cv2.CAP_PROP_FPS) or 30.0
        w = int(cap_v.get(cv2.CAP_PROP_FRAME_WIDTH))
        h = int(cap_v.get(cv2.CAP_PROP_FRAME_HEIGHT))
        total = int(cap_v.get(cv2.CAP_PROP_FRAME_COUNT))

        writer = cv2.VideoWriter(output_path, cv2.VideoWriter_fourcc(*"mp4v"), fps, (w, h))

        frame_idx = 0
        try:
            while True:
                ok_v, frame = cap_v.read()
                ok_m, mask_frame = cap_m.read()
                if not (ok_v and ok_m):
                    break

                mask_gray = cv2.cvtColor(mask_frame, cv2.COLOR_BGR2GRAY)
                mask_bool = mask_gray > 127

                filled = self.fill_frame(frame, mask_bool)
                writer.write(filled)

                frame_idx += 1
                if progress_callback is not None:
                    progress_callback(frame_idx, total)
        finally:
            cap_v.release()
            cap_m.release()
            writer.release()


def _pad_reflect_once(x, padding):
    import torch.nn.functional as F
    _, _, h, w = x.shape
    padding_arr = np.array(padding)
    size = np.array([w, w, h, h])
    initial_padding = np.minimum(padding_arr, size - 1)
    additional_padding = padding_arr - initial_padding
    x = F.pad(x, tuple(int(v) for v in initial_padding), mode="reflect")
    if np.any(additional_padding > 0):
        x = F.pad(x, tuple(int(v) for v in additional_padding), mode="constant")
    return x


def _resize_square(image, mask, size):
    import torch.nn.functional as F
    _, _, h, w = image.shape
    pad_w, pad_h, prev_size = 0, 0, w

    if w == size and h == size:
        return image, mask, (pad_w, pad_h, prev_size)

    if w < h:
        pad_w = h - w
        prev_size = h
    elif h < w:
        pad_h = w - h
        prev_size = w

    image = _pad_reflect_once(image, (0, pad_w, 0, pad_h))
    mask = _pad_reflect_once(mask, (0, pad_w, 0, pad_h))

    if image.shape[-1] != size:
        image = F.interpolate(image, size=size, mode="nearest-exact")
        mask = F.interpolate(mask, size=size, mode="nearest-exact")

    return image, mask, (pad_w, pad_h, prev_size)


def _undo_resize_square(image, original_size):
    pad_w, pad_h, prev_size = original_size
    _, _, h, w = image.shape
    if prev_size != w or prev_size != h:
        img_np = image[0].permute(1, 2, 0).cpu().numpy()
        img_np = cv2.resize(img_np, (prev_size, prev_size), interpolation=cv2.INTER_LANCZOS4)
        import torch
        image = torch.from_numpy(img_np).permute(2, 0, 1).unsqueeze(0).to(image.device)
    return image[:, :, 0:prev_size - pad_h, 0:prev_size - pad_w]


def _mask_floor(mask, threshold: float = 0.99):
    return (mask >= threshold).to(mask.dtype)


class LamaBackgroundFiller:
    """
    Real big-lama model, matching the exact algorithm used by this
    project's own ComfyUI INPAINT_InpaintWithModel node
    (Acly/comfyui-inpaint-nodes), verified against the live RunPod pod's
    installed source this session, plus two validated deviations found to
    improve quality on hard scenes (dense/fine background texture):

      - grow_mask_px: dilates the raw mask by this many pixels before the
        256px resize, excluding a contaminated boundary ring (anti-aliased/
        compressed person-color bleeding just past the mask edge) that the
        model would otherwise sample as if it were real background --
        without this, LaMa can hallucinate person-like structure in the
        fill on some frames. Not part of the upstream node's default
        behavior.
      - Lanczos upsample (instead of the upstream node's bilinear) when
        scaling the 256px result back to source resolution -- measurably
        sharper at no extra cost, still not a full fix for detail lost at
        the 256px bottleneck but a clear improvement.

    LaMa's 256px working resolution is NOT an arbitrary speed/quality
    shortcut -- it's the model's actual trained resolution (WACV 2022
    paper: trained on 256x256 crops). Running the model at other
    resolutions was measured (this session) to produce worse output, not
    just slower -- do not "fix" this by changing required_size.
    """

    def __init__(self, model_path: str = "models/big-lama.pt",
                 required_size: int = 256, grow_mask_px: int = 10,
                 device: str = "cpu"):
        import torch
        self._torch = torch
        self._device = torch.device(device)
        self._model = torch.jit.load(model_path, map_location=self._device)
        self._model.eval()
        self._required_size = required_size
        self._grow_mask_px = grow_mask_px
        self._dilate_kernel = (
            cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (grow_mask_px * 2 + 1,) * 2)
            if grow_mask_px > 0 else None
        )

    def fill_frame(self, frame_bgr: np.ndarray, mask_bool: np.ndarray, seed: int = 0) -> np.ndarray:
        """
        frame_bgr: H×W×3 uint8, the anonymized (grey-filled) frame.
        mask_bool: H×W bool, True where the person (grey fill) is.
        Returns: H×W×3 uint8, background reconstructed where the mask was.
        """
        torch = self._torch
        if not mask_bool.any():
            return frame_bgr.copy()

        frame_rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
        mask_u8 = mask_bool.astype(np.uint8) * 255
        if self._dilate_kernel is not None:
            mask_u8 = cv2.dilate(mask_u8, self._dilate_kernel)

        image_t = torch.from_numpy(frame_rgb).float().div(255.0).unsqueeze(0).permute(0, 3, 1, 2)
        mask_t = torch.from_numpy(mask_u8).float().div(255.0).unsqueeze(0).unsqueeze(0)

        work_image, work_mask, original_size = _resize_square(image_t, mask_t, self._required_size)
        work_mask_floored = _mask_floor(work_mask)

        with torch.inference_mode():
            torch.manual_seed(seed)
            result = self._model(work_image.to(self._device), work_mask_floored.to(self._device))

        result = _undo_resize_square(result, original_size)
        final = image_t + (result - image_t) * _mask_floor(mask_t)

        final_np = final[0].permute(1, 2, 0).clamp(0, 1).mul(255).byte().cpu().numpy()
        return cv2.cvtColor(final_np, cv2.COLOR_RGB2BGR)

    def fill_video(self, video_path: str, mask_path: str, output_path: str,
                    progress_callback=None) -> None:
        cap_v = cv2.VideoCapture(video_path)
        cap_m = cv2.VideoCapture(mask_path)
        if not cap_v.isOpened() or not cap_m.isOpened():
            raise IOError(f"Could not open {video_path} or {mask_path}")

        fps = cap_v.get(cv2.CAP_PROP_FPS) or 30.0
        w = int(cap_v.get(cv2.CAP_PROP_FRAME_WIDTH))
        h = int(cap_v.get(cv2.CAP_PROP_FRAME_HEIGHT))
        total = int(cap_v.get(cv2.CAP_PROP_FRAME_COUNT))

        writer = cv2.VideoWriter(output_path, cv2.VideoWriter_fourcc(*"mp4v"), fps, (w, h))

        frame_idx = 0
        try:
            while True:
                ok_v, frame = cap_v.read()
                ok_m, mask_frame = cap_m.read()
                if not (ok_v and ok_m):
                    break

                mask_gray = cv2.cvtColor(mask_frame, cv2.COLOR_BGR2GRAY)
                mask_bool = mask_gray > 127

                filled = self.fill_frame(frame, mask_bool)
                writer.write(filled)

                frame_idx += 1
                if progress_callback is not None:
                    progress_callback(frame_idx, total)
        finally:
            cap_v.release()
            cap_m.release()
            writer.release()
