"""
segbench.prep_models -- export/cache the exact model artifact each config needs.

ONNX-exported YOLO-seg has a STATIC input shape baked in at export time (a 320
export rejects a 256 input -- confirmed InvalidArgument), so every infer size
needs its own export. Filenames are size-suffixed so sizes never collide, same
convention as the shipped blur_yoloseg. All artifacts land in the lab models/
dir (never in the reference repo).
"""
import os


def export_onnx(pt_path: str, imgsz: int, out_dir: str = None,
                half: bool = False, simplify: bool = False) -> str:
    """Static-shape ONNX export at `imgsz`, cached & size-suffixed. Returns path."""
    out_dir = out_dir or os.path.dirname(pt_path)
    base = os.path.splitext(os.path.basename(pt_path))[0]         # e.g. yolo11n-seg
    suffix = f"_{imgsz}" + ("_fp16" if half else "")
    onnx_path = os.path.join(out_dir, f"{base}{suffix}.onnx")
    if os.path.exists(onnx_path):
        return onnx_path
    from ultralytics import YOLO
    exported = YOLO(pt_path).export(format="onnx", imgsz=imgsz,
                                    simplify=simplify, half=half)
    exported = str(exported)
    if os.path.abspath(exported) != os.path.abspath(onnx_path):
        os.replace(exported, onnx_path)
    return onnx_path


def quantize_onnx_int8_dynamic(onnx_fp32: str) -> str:
    """Dynamic weight-only INT8 (matches blur_yoloseg). Cheap, no calibration
    set; on x86 often ~parity/slower, but ARM NEON dot-product can differ --
    kept for the ablation, measured not assumed. Returns path (cached)."""
    out = onnx_fp32[:-len(".onnx")] + "_int8.onnx"
    if os.path.exists(out):
        return out
    from onnxruntime.quantization import quantize_dynamic, QuantType
    quantize_dynamic(onnx_fp32, out, weight_type=QuantType.QUInt8)
    return out
