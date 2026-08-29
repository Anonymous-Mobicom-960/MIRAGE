# Edge runner deployment (Raspberry Pi 5)

This directory is the Tier-1 edge runner: the `config.py`-driven implementation that runs on the
Raspberry Pi 5 wearable proxy. It owns the guided-segmentation mask source, FaceGuard, and the
differential-privacy stage for the expression channel. The capture service used for the reported
end-to-end runs is its sibling, `../mirage/`; the two share the vendored re-identification
defences byte-for-byte (see `../mirage/vendor/mirage_edge/VENDOR.md`).

## Models

Place in `models/` (nothing is downloaded automatically; sources and licences in
[`../../../models/README.md`](../../../models/README.md)):

```
yolo11n.onnx  or  yolo11n_ncnn_model/     person detection (NCNN pair is the Pi-canonical path)
yolo11n-seg.onnx                          instance segmentation for the guided mask source
rtmpose-t-wholebody.onnx                  pose
```

The historical Robust Video Matting backend (`rvm_mobilenetv3_fp32.onnx`) survives only as an
optional union backstop; the shipped mask source is the guided YOLO11n-seg.

## Run

```bash
pip3 install -r requirements.txt
bash run.sh --source input.mp4          # or --source 0 for a camera
```

Configuration is entirely `config.py`: 107 knobs, each overridable by a `MIRAGE_*` environment
variable. `test_config_contract.py` asserts the contract between the config surface and the
runner. A run's emitted `pose.json` records the *effective* configuration, which is what should
be quoted; several environment variables mutate the level dictionaries at import time.

## Notes for the Pi specifically

* Prefer the NCNN detector backend; on the Cortex-A76 it measured about twice as fast as ONNX
  Runtime for the same model, and ONNX INT8 is *slower* than fp32 there (the opposite of the
  phone). Reference numbers: `../../../evaluation/performance/pi_bench/PI_RESULTS.md`.
* Install exactly one of `opencv-python` / `opencv-contrib-python`; contrib provides the
  guided-filter mask edge and the runner falls back to bilinear resize without it.
* A Pi 5 under a sustained 4-thread inference load needs a genuine 5 V / 5 A supply; an
  inadequate one brown-out-resets the board mid-run.
