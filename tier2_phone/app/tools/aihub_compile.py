#!/usr/bin/env python3
"""
aihub_compile.py - compile a PUBLIC MI-GAN.onnx into a QNN-context ONNX
for the Snapdragon 8 Elite (SM8750) Hexagon HTP, at W8A16, using Qualcomm AI Hub.

The output .onnx (a QNN-context model) is PUSHED to the installed app at runtime (v0.3+,
NO rebuild/reinstall - the app's ModelStore prefers files/models/ over bundled assets):
    adb push migan_qnn_ctx.onnx /sdcard/Android/data/com.mirage.npu/files/models/
(Requires the -Pmirage.enableOrt=true APK flavour for the HTP.)

╔══════════════════════════════════════════════════════════════════════════════════════════════╗
║  🔒 PRIVACY - READ BEFORE RUNNING (V9_PLAN §2 / §9)                                             ║
║  • This step runs OFF-DEVICE on Qualcomm AI Hub and MUST process ONLY PUBLIC model weights     ║
║    (the released MI-GAN checkpoint). NEVER feed it user frames, real backgrounds, or            ║
║    any MIRAGE capture.                                                                          ║
║  • Quantization calibration data MUST be PUBLIC/synthetic (e.g. random tensors or a public      ║
║    dataset). Do NOT calibrate on captured footage - that would bake real-scene statistics into  ║
║    an artifact that leaves the device.                                                          ║
║  • Do NOT run AI Hub *on-device profiling* (submit_profile_job / submit_inference_job) with     ║
║    real user footage. Profiling uploads inputs to Qualcomm's device farm. Use synthetic/public  ║
║    inputs only. This script does NOT submit a profile job; a commented example shows how, gated ║
║    behind an explicit synthetic-data flag.                                                      ║
║  • The compiled model has no telemetry, but the RUNTIME app must still be audited for egress    ║
║    (it ships with NO INTERNET permission - see the app's AndroidManifest.xml).                  ║
╚══════════════════════════════════════════════════════════════════════════════════════════════╝

STATUS: REAL script, but UNVERIFIED here (needs a `qai-hub` account + API token + network, none of
which exist in this scaffold's environment). The exact qai-hub option strings and the model's input
names/shapes MUST be confirmed against your installed qai-hub version and your actual ONNX file - 
spots that need that are marked `VERIFY`.

Setup (host, x86-64 Linux/macOS/WSL recommended):
    pip install qai-hub qai-hub-models onnx onnxruntime numpy
    qai-hub configure --api_token <YOUR_TOKEN>     # from https://aihub.qualcomm.com (free account)

Obtain PUBLIC model ONNX first (see README "Obtain the models"):
    # MI-GAN: export from github.com/Picsart-AI-Research/MI-GAN (MI-GAN-512) to ONNX

Run:
    python aihub_compile.py --model migan.onnx --kind migan --out ../app/src/main/assets/models/migan_qnn_ctx.onnx
"""
import argparse
import sys

CHIPSET = "snapdragon-8-elite"     # SM8750 (S25 Ultra). AI Hub device attribute.


def log(msg: str) -> None:
    print(f"[aihub_compile] {msg}", flush=True)


def build_input_specs(kind: str, size: int):
    """
    VERIFY against your real ONNX (open it in netron / print session.get_inputs()).
    MI-GAN takes an image + a 1-channel mask. These are the shapes the compiled HTP context
    is frozen to.
    """
    if kind == "migan":
        return {
            "image": ((1, 3, size, size), "float32"),
            "mask": ((1, 1, size, size), "float32"),
        }
    raise ValueError(f"unknown --kind {kind!r} (expected migan)")


def public_calibration_data(input_specs, num_samples: int = 8):
    """
    PUBLIC/synthetic calibration ONLY (random tensors here). For better PTQ accuracy, substitute a
    PUBLIC dataset (e.g. a few COCO/Vimeo90K frames you are licensed to use) - but NEVER user frames.
    Returns the dict shape qai-hub's quantize job expects: {input_name: [np.ndarray, ...]}.
    """
    import numpy as np
    data = {}
    for name, (shape, dtype) in input_specs.items():
        data[name] = [np.random.rand(*shape).astype(dtype) for _ in range(num_samples)]
    log(f"calibration = {num_samples} PUBLIC/synthetic random samples (NOT user data)")
    return data


def main() -> int:
    ap = argparse.ArgumentParser(description="Compile a PUBLIC model to a QNN-context ONNX for the 8 Elite HTP.")
    ap.add_argument("--model", required=True, help="path to the PUBLIC source .onnx (MI-GAN)")
    ap.add_argument("--kind", required=True, choices=["migan"])
    ap.add_argument("--out", required=True, help="output QNN-context .onnx path (adb push to files/models/, no reinstall)")
    ap.add_argument("--size", type=int, default=512, help="square input size (multiple of 32)")
    ap.add_argument("--precision", default="w8a16", choices=["w8a16", "fp16"],
                    help="w8a16 = 8-bit weights / 16-bit activations (Qualcomm's accuracy/perf balance); fp16 = no PTQ")
    ap.add_argument("--calib-samples", type=int, default=8)
    args = ap.parse_args()

    try:
        import qai_hub as hub
    except ImportError:
        log("ERROR: qai-hub not installed. `pip install qai-hub qai-hub-models` and `qai-hub configure`.")
        return 2

    input_specs = build_input_specs(args.kind, args.size)
    # AI Hub device selected by chipset attribute (works without naming a specific phone model).
    device = hub.Device(attributes=[f"chipset:{CHIPSET}"])
    log(f"target device attribute: chipset:{CHIPSET}  precision={args.precision}  size={args.size}")

    source_model = args.model
    compile_options = "--target_runtime precompiled_qnn_onnx"   # VERIFY flag name for your qai-hub version

    # ---- Step 1: quantize to W8A16 (skip for fp16) - PUBLIC calibration only ----
    if args.precision == "w8a16":
        log("submitting QUANTIZE job (W8A16) with PUBLIC/synthetic calibration ...")
        calib = public_calibration_data(input_specs, args.calib_samples)
        try:
            quantize_job = hub.submit_quantize_job(
                model=source_model,
                calibration_data=calib,
                weights_dtype="int8",        # W8   (VERIFY the exact enum/string for your qai-hub)
                activations_dtype="int16",   # A16  -> together = W8A16
            )
            source_model = quantize_job.get_target_model()   # feed the quantized model into compile
            log("quantize job complete -> quantized model handle obtained")
        except Exception as e:  # noqa: BLE001 - surface the real API error for the operator
            log(f"ERROR: quantize job failed ({e}).")
            log("       VERIFY submit_quantize_job signature + dtype strings for your qai-hub version.")
            return 3

    # ---- Step 2: compile to a QNN-context ONNX for the 8 Elite HTP ----
    log("submitting COMPILE job -> precompiled QNN-context ONNX ...")
    try:
        compile_job = hub.submit_compile_job(
            model=source_model,
            device=device,
            input_specs={k: v[0] for k, v in input_specs.items()},   # {name: shape}
            options=compile_options,
        )
        target_model = compile_job.get_target_model()
        target_model.download(args.out)
    except Exception as e:  # noqa: BLE001
        log(f"ERROR: compile job failed ({e}).")
        log("       VERIFY --target_runtime value + input_specs against your qai-hub version and ONNX.")
        return 4

    log(f"DONE -> {args.out}")
    log("Push to the installed app (no reinstall): adb push <out> /sdcard/Android/data/com.mirage.npu/files/models/")
    log("(Needs the -Pmirage.enableOrt=true APK flavour for HTP execution; see INSTALL.md §3/§4.)")
    log("REMINDER: this compiled PUBLIC-weight model is safe to ship; do NOT profile it on AI Hub with real frames.")

    # ---- OPTIONAL profiling (DISABLED). Only with SYNTHETIC/PUBLIC inputs, NEVER user footage. ----
    # if args.allow_synthetic_profile:
    #     hub.submit_profile_job(model=target_model, device=device)   # uploads inputs to Qualcomm's device farm
    return 0


if __name__ == "__main__":
    sys.exit(main())
