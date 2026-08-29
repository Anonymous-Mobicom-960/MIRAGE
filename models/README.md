# Model Weights

**No model weights are redistributed in this repository.** Every model below is third party. Several
carry licences that either forbid redistribution or restrict downstream use, so each must be obtained
from its own upstream source.

Download each file, place it at the path given, and check the licence column before using the system
for anything beyond research.

---

## Tier 1: capture service

| Model | Expected filename | Expected location | Source | Licence |
|---|---|---|---|---|
| YOLO11n (person detection) | `yolo11n.onnx`, `yolo11n.pt`, or the NCNN pair `model.ncnn.param` / `model.ncnn.bin` | `tier1/src/edge_runner_pi5/models/` (`yolo11n_ncnn_model/` for the NCNN pair) | [Ultralytics](https://github.com/ultralytics/ultralytics) | **AGPL-3.0** |
| YOLO11n-seg (instance segmentation, edge runner) | `yolo11n-seg.onnx` | `tier1/src/edge_runner_pi5/models/` | Ultralytics | **AGPL-3.0** |
| RTMPose-Tiny wholebody | `rtmpose-t-wholebody.onnx` | `tier1/src/edge_runner_pi5/models/` | [MMPose / RTMPose](https://github.com/open-mmlab/mmpose/tree/main/projects/rtmpose) | Apache-2.0 |
| RTMPose-Tiny body7 (capture service) | fetched automatically by `rtmlib` into its own cache | `~/.cache/rtmlib/` | [rtmlib](https://github.com/Tau-J/rtmlib) | Apache-2.0 |
| MediaPipe FaceLandmarker | `face_landmarker.task` | alongside the capture service; path is a constructor argument | [Google MediaPipe](https://developers.google.com/mediapipe/solutions/vision/face_landmarker) | Apache-2.0 |
| MediaPipe Selfie Segmenter (alternative anonymiser backends) | `selfie_segmenter.tflite`, `selfie_segmenter_landscape.tflite` | as configured | Google MediaPipe | Apache-2.0 |
| InsightFace genderage | `genderage.onnx` (from the `buffalo_l` pack) | `models/insightface/` | [InsightFace](https://github.com/deepinsight/insightface) | **Non-commercial research use only** |
| EdgeFace-XS (gamma = 0.6) | `edgeface_xs_gamma_06.onnx`, exported from the upstream `edgeface_xs_gamma_06.pt` | `models/` (the path in `embedding.py`, overridable) | [otroshi/edgeface](https://github.com/otroshi/edgeface) - `checkpoints/edgeface_xs_gamma_06.pt`, also `torch.hub.load('otroshi/edgeface', 'edgeface_xs_gamma_06')`; model card [Idiap/EdgeFace-XS-GAMMA](https://huggingface.co/Idiap/EdgeFace-XS-GAMMA) | **CC BY-NC-SA 4.0** - non-commercial, share-alike |
| Robust Video Matting (legacy edge backend) | `rvm_mobilenetv3_fp32.onnx` | `tier1/src/edge_runner_pi5/models/` | [PeterL1n/RobustVideoMatting](https://github.com/PeterL1n/RobustVideoMatting) | GPL-3.0 |

**EdgeFace citation.** The recovery embedding is produced by EdgeFace-XS (gamma = 0.6), the compact
variant that won the IJCB-2023 Efficient Face Recognition Competition compact track. Cite the
authors, not this repository:

```bibtex
@article{edgeface,
  title   = {EdgeFace: Efficient Face Recognition Model for Edge Devices},
  author  = {George, Anjith and Ecabert, Christophe and Otroshi Shahreza, Hatef
             and Kotwal, Ketan and Marcel, Sebastien},
  journal = {IEEE Transactions on Biometrics, Behavior, and Identity Science},
  year    = {2024}
}
```

We export the upstream `.pt` to ONNX and redistribute neither. The variant is XS (gamma = 0.6)
rather than S (gamma = 0.5) because the reported 1.77 M parameter count is xs-gamma-06's:
counted off the ONNX graphs, xs-gamma-06 is 1,770,492 parameters against s-gamma-05's 3,652,520,
and upstream reports the same 1.77 M for XS (gamma = 0.6).

## Tier 2: phone

| Model | Expected filename | Expected location | Source | Licence |
|---|---|---|---|---|
| LaMa (background inpainting) | `lama.onnx` | `/sdcard/Android/data/com.mirage.npu/files/models/` on the device | [advimman/lama](https://github.com/advimman/lama) (`big-lama`) | Code Apache-2.0; **`big-lama` weights are CC BY-NC-SA 4.0, non-commercial** |
| YOLO11n-seg, MoveNet, RVM (optional on-device Tier-1 preview) | `seg.onnx`, `movenet.onnx`, `rvm.onnx` | same device directory | Ultralytics / Google / PeterL1n | AGPL-3.0 / Apache-2.0 / GPL-3.0 |
| MI-GAN-512 (optional faster inpaint) | `migan.onnx` | same device directory | [Picsart-AI-Research/MI-GAN](https://github.com/Picsart-AI-Research/MI-GAN) | **research/non-commercial** |

`migan.onnx` is entirely optional. `MiGanNpu.fromModelsDirOrNull` returns null when the file is
absent and the app falls back to LaMa, so a device without it runs the full pipeline unchanged.

Models are loaded at runtime from the device directory, which takes precedence over anything bundled
into the APK, so a model can be replaced with `adb push`, without reinstalling.

```bash
adb push <model>.onnx /sdcard/Android/data/com.mirage.npu/files/models/
```

## Tier 2: cloud

Place these where the ComfyUI installation expects them.

| Model | Expected filename | Expected location | Source | Licence |
|---|---|---|---|---|
| Wan 2.2 Animate 14B (generator) | `Wan2_2-Animate-14B_fp8_e4m3fn_scaled_KJ.safetensors` | `<ComfyUI>/models/diffusion_models/` | [Wan-Video/Wan2.2](https://github.com/Wan-Video/Wan2.2); the fp8 scaled repack is distributed by Kijai | Apache-2.0 (check the repack's own card) |
| CFG step-distill LoRA | `lightx2v_I2V_14B_480p_cfg_step_distill_rank64_bf16.safetensors` | `<ComfyUI>/models/loras/` | lightx2v | Check the model card |
| HighRes textures LoRA | `Wan2.1_HighRes_Textures.safetensors` | `<ComfyUI>/models/loras/` | community | Check the model card |
| Realism boost LoRA | `Wan14B_RealismBoost.safetensors` | `<ComfyUI>/models/loras/WanVideo/` | community | Check the model card |
| Camera-rotation LoRA | `Wan21_Camera_Rotation.safetensors` | `<ComfyUI>/models/loras/WanVideo/` | community | Loaded at strength **0.0**, present but disabled |
| SAM 2.1 (video segmentation) | `sam2.1_hiera_base_plus.safetensors` | `<ComfyUI>/models/sam2/` | [facebookresearch/sam2](https://github.com/facebookresearch/sam2) | Apache-2.0 |
| ViTPose-L wholebody (matte re-detection) | `vitpose-l-wholebody.onnx` | `<ComfyUI>/models/detection/` | ViTPose | Apache-2.0 |
| YOLOv10-M (matte re-detection) | `yolov10m.onnx` | `<ComfyUI>/models/detection/` | [THU-MIG/yolov10](https://github.com/THU-MIG/yolov10) | AGPL-3.0 |

The detection directory is resolved from ComfyUI's own `folder_paths` when the node is imported
inside a ComfyUI process, and falls back to a relative `models/detection` otherwise.

## Evaluation

| Model / data | Expected location | Source |
|---|---|---|
| GaitGraph ResGCN checkpoint | `evaluation/privacy/gait/data/gaitgraph_resgcn-n39-r8_coco_seq_60.pth` | [tteepe/GaitGraph](https://github.com/tteepe/GaitGraph/releases/tag/v0.1) |
| GaitGraph / GaitGraph2 source | `evaluation/privacy/gait/vendor/GaitGraph{,2}/` | same |
| CASIA-B COCO-17 pose CSVs | `evaluation/privacy/gait/data/casia-b_pose_{test,train_valid}.csv` | same release |
| OpenGait + a GaitBase checkpoint | `evaluation/privacy/silhouette/vendor/`, `.../data/` | [ShiqiYu/OpenGait](https://github.com/ShiqiYu/OpenGait) |
| YOLO11s-seg (appearance-harness extractor) | `evaluation/privacy/appearance/models/yolo11s-seg.pt` | [Ultralytics](https://github.com/ultralytics/ultralytics) (AGPL-3.0) |
| DeepPrivacy2 (baseline defence) | `evaluation/privacy/appearance/baselines_dp2/repo/` at commit `f4d8f09` + its own checkpoints, in its own container | [hukkelas/deep_privacy2](https://github.com/hukkelas/deep_privacy2) |
| YuNet face detector (independent audit cross-check) | anywhere; point `MIRAGE_YUNET_ONNX` at it | [OpenCV Zoo](https://github.com/opencv/opencv_zoo/tree/main/models/face_detection_yunet) |

Exact commands are in [`../docs/reproduction.md`](../docs/reproduction.md).

## Checksums

Checksums are not published here, because none of the files above are distributed by this project and
several upstream sources re-cut their artifacts. Verify against the checksum the upstream source
publishes at download time.

## Licence caution

Four constraints propagate into anything built on this repository:

* **Ultralytics YOLO11 is AGPL-3.0.** Network use of a derived work triggers the source-provision
  obligation, and the AGPL is incompatible with a permissive licence for the combined work. This is
  the single largest constraint on what licence this repository can adopt.
* **The `big-lama` weights are CC BY-NC-SA 4.0**: non-commercial, share-alike.
* **InsightFace models are licensed for non-commercial research only.**
* **EdgeFace weights are CC BY-NC-SA 4.0**: non-commercial, share-alike. The recovery embedding
  is produced by this model, so the constraint reaches the Tier-1 capture path, not just evaluation.

See [`../THIRD_PARTY.md`](../THIRD_PARTY.md).
