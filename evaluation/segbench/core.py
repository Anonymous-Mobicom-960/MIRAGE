"""
segbench.core -- apple-to-apple building blocks for the Tier-1 segmentation
comparison. Everything that is *shared and held constant* across every
benchmarked config lives here, so the only thing that varies between runs is
the factor under test (seg model / mask-upsampling method / infer size / etc).

Frame convention: the Tier-1 input AND output are both 1264x1264 (a hard
project constraint). The input clip is square, so the seg network's square
letterbox has NO padding -- a mask returned at imgsz x imgsz maps back to the
full frame by a single resize with no unletterboxing needed. (Verified: input
is exactly 1264x1264; ultralytics masks.data is imgsz x imgsz.)
"""
import time
import numpy as np
import cv2

# The grey fill colour the shipped pipeline uses (blur_yoloseg.HULL_COLOR).
GREY = np.array([127, 127, 127], dtype=np.uint8)


# ---------------------------------------------------------------------------
# Mask upsampling methods -- THE factor the "downscale + smooth" idea is about.
# Both take a low-res binary mask (imgsz x imgsz, values {0,1}) and produce a
# full-res boolean mask (out_h x out_w).
# ---------------------------------------------------------------------------
_HAS_XIMGPROC = hasattr(cv2, "ximgproc")


def upscale_mask(mask_native: np.ndarray, out_hw, *,
                 interp: str = "nearest", close_k: int = 0, dilate_px: int = 0,
                 guided: bool = False, guide_gray: np.ndarray = None,
                 guided_radius: int = 12, guided_eps: float = 1e-3) -> np.ndarray:
    """Unified low-res -> full-res mask upsampler. Every mask-post factor the
    ablation sweeps is a keyword here, so a config is data, not a code branch.

    Pipeline: resize (nearest|linear) -> optional guided-filter edge-align
    (He/Sun fast guided filter, image as guidance) -> threshold -> morphological
    close (seal pinholes) -> OUTWARD dilation (privacy margin so the true
    silhouette sits strictly inside the grey region -- never a boundary leak).
    Dilation is applied LAST and only outward, per the structural-privacy rule.
    """
    out_h, out_w = out_hw
    inter = cv2.INTER_NEAREST if interp == "nearest" else cv2.INTER_LINEAR
    m = cv2.resize(mask_native.astype(np.float32), (out_w, out_h), interpolation=inter)

    if guided and _HAS_XIMGPROC and guide_gray is not None:
        # edge-align the coarse mask to the real person boundary using the
        # full-res frame as guidance; fast guided filter subsamples internally.
        m = cv2.ximgproc.guidedFilter(guide_gray, m.astype(np.float32),
                                      guided_radius, guided_eps)

    b = (m > 0.5).astype(np.uint8)
    if close_k and close_k > 0:
        k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (close_k, close_k))
        b = cv2.morphologyEx(b, cv2.MORPH_CLOSE, k)
    if dilate_px and dilate_px > 0:
        k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE,
                                      (2 * dilate_px + 1, 2 * dilate_px + 1))
        b = cv2.dilate(b, k)
    return b.astype(bool)


# Named presets -> upscale_mask kwargs. "baseline" is faithful to the shipped
# blur_yoloseg path (nearest, no smoothing, no dilation). The others are the
# proposed "downscale + smart upscale + structural privacy dilation" methods.
def preset_kwargs(method: str, close_k: int, dilate_px: int) -> dict:
    if method == "baseline":
        return dict(interp="nearest", close_k=0, dilate_px=0, guided=False)
    if method == "smooth":          # bilinear + morph + outward dilation
        return dict(interp="linear", close_k=close_k, dilate_px=dilate_px, guided=False)
    if method == "guided":          # guided-filter edge-align + outward dilation
        return dict(interp="linear", close_k=close_k, dilate_px=dilate_px, guided=True)
    raise ValueError(f"unknown method {method}")


def _fast_guided(guide_gray, src_float, radius, eps, scale=0.25):
    """He/Sun FAST guided filter: run the guided filter at `scale` resolution,
    then bilinearly upsample the result -- cost ~scale^2 of full, near-invisible
    quality loss (research: s=0.25 sweet spot, >10x speedup). Guidance = full-res
    gray frame, downsampled to match. The guide MUST be normalized to [0,1] --
    eps is defined on that scale; a raw 0-255 guide with eps~1e-3 makes the
    covariance term underflow and the filter returns NaN (learned the hard way)."""
    h, w = src_float.shape[:2]
    sw, sh = max(1, int(w * scale)), max(1, int(h * scale))
    g = cv2.resize(guide_gray, (sw, sh), interpolation=cv2.INTER_LINEAR)
    if g.dtype != np.float32:
        g = g.astype(np.float32) / 255.0
    s = cv2.resize(src_float.astype(np.float32), (sw, sh), interpolation=cv2.INTER_LINEAR)
    r = max(1, int(round(radius * scale)))
    q = cv2.ximgproc.guidedFilter(g, s, r, float(eps))
    q = np.nan_to_num(q, nan=0.0, posinf=1.0, neginf=0.0)
    return cv2.resize(q, (w, h), interpolation=cv2.INTER_LINEAR)


def build_union_mask(instances, out_hw, *, method="baseline",
                     close_k=5, dilate_px=8, guide_gray=None,
                     guided_radius=12, guided_eps=1e-2, guided_scale=0.25,
                     work_scale=1.0):
    """Full-res boolean UNION person mask for grey-fill + privacy. Grey-fill
    doesn't need per-person mask separation (all persons -> same grey), so the
    soft union is built once and refined once per frame -- N-times cheaper than
    per-instance guided filtering, and over-coverage where masks fuse is a
    privacy PLUS, not a bug.

    method: baseline (nearest, no smoothing/dilation -- shipped behaviour)
          | smooth   (bilinear + morph-close + OUTWARD dilation)
          | guided   (bilinear soft-union + fast guided-filter edge-align
                      + morph-close + OUTWARD dilation)

    work_scale<1.0: run the ENTIRE mask pipeline (union build, guided, morph,
    dilate) at work_scale*out resolution, then nearest-upscale the final binary
    mask to full res. All heavy 1264^2 ops shrink ~work_scale^2 -- the big ARM
    cost. Since the source mask is a coarse imgsz-res seg (e.g. 256) anyway, the
    quality cost is small; measured, not assumed. Kernel sizes scale with
    work_scale so the effective full-res morphology/dilation is preserved.
    """
    out_h, out_w = out_hw
    if not instances:
        return np.zeros(out_hw, dtype=bool)

    ws = float(work_scale)
    wh, ww = (max(1, int(out_h * ws)), max(1, int(out_w * ws))) if ws < 1.0 else (out_h, out_w)
    inter = cv2.INTER_NEAREST if method == "baseline" else cv2.INTER_LINEAR
    union = np.zeros((wh, ww), dtype=np.float32)
    for ins in instances:
        up = cv2.resize(ins["mask"].astype(np.float32), (ww, wh), interpolation=inter)
        np.maximum(union, up, out=union)

    if method == "guided" and _HAS_XIMGPROC and guide_gray is not None:
        g = guide_gray if (wh, ww) == guide_gray.shape[:2] else cv2.resize(guide_gray, (ww, wh))
        union = _fast_guided(g, union, max(1, int(guided_radius * ws)) if ws < 1 else guided_radius,
                             guided_eps, guided_scale)

    b = (union > 0.5).astype(np.uint8)
    if method != "baseline":
        ck = max(1, int(round(close_k * ws))) if ws < 1 else close_k
        dp = max(1, int(round(dilate_px * ws))) if (ws < 1 and dilate_px > 0) else dilate_px
        if close_k and close_k > 0:
            k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (ck, ck))
            b = cv2.morphologyEx(b, cv2.MORPH_CLOSE, k)
        if dilate_px and dilate_px > 0:
            k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2 * dp + 1, 2 * dp + 1))
            b = cv2.dilate(b, k)

    if (wh, ww) != (out_h, out_w):
        b = cv2.resize(b, (out_w, out_h), interpolation=cv2.INTER_NEAREST)
    return b.astype(bool)


def apply_grey(frame: np.ndarray, union_mask: np.ndarray) -> np.ndarray:
    """Grey-fill every pixel under the union of all person masks."""
    out = frame.copy()
    out[union_mask] = GREY
    return out


def box_centroid(box):
    x1, y1, x2, y2 = box
    return (0.5 * (x1 + x2), 0.5 * (y1 + y2))


# ---------------------------------------------------------------------------
# Unified instance-segmentation wrapper. One .infer() returns per-instance
# low-res masks + boxes + scores, and the pure model-forward time. Works for
# any ultralytics-loadable weights (.pt / .onnx / ncnn dir / openvino dir),
# so model + export-format become just config, not code branches.
# ---------------------------------------------------------------------------
class UltralyticsSeg:
    def __init__(self, weights: str, conf: float = 0.4):
        from ultralytics import YOLO
        self.model = YOLO(weights)
        self.conf = conf
        self.weights = weights

    def infer(self, frame: np.ndarray, imgsz: int, device: str = "cpu"):
        """Returns (t_seg_seconds, instances) where instances is a list of
        dicts {mask: (imgsz,imgsz) uint8 {0,1}, box: [x1,y1,x2,y2] float in
        full-frame px, score: float}. Only the model forward pass is timed."""
        t0 = time.perf_counter()
        r = self.model.predict(frame, imgsz=imgsz, conf=self.conf, classes=[0],
                               device=device, verbose=False, retina_masks=False)[0]
        t_seg = time.perf_counter() - t0
        instances = []
        if r.masks is not None and r.boxes is not None and len(r.boxes) > 0:
            md = r.masks.data.cpu().numpy()          # (N, imgsz, imgsz)
            bx = r.boxes.xyxy.cpu().numpy()          # (N, 4) full-frame px
            sc = r.boxes.conf.cpu().numpy()          # (N,)
            for i in range(md.shape[0]):
                instances.append(dict(
                    mask=(md[i] > 0.5).astype(np.uint8),
                    box=bx[i].astype(float),
                    score=float(sc[i]),
                ))
        return t_seg, instances
