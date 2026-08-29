# Third-Party Components

Every external component this system depends on, what it is used for, and under what terms. Nothing
listed here is redistributed by this repository; code is installed from upstream, weights are
downloaded per [`models/README.md`](models/README.md).

Licence identifiers are given as published upstream at the time of curation. **Verify each one at
the version you install**; several of these projects have changed licence between releases.

---

## Code included in this repository

One piece of third-party-derived code is vendored, and it is fully attributed rather than absorbed.

| Path | What it is | Provenance |
|---|---|---|
| `tier1/src/mirage/vendor/mirage_edge/` | The two re-identification defence modules (`pose_anon_edge.py`, `person_slots.py`, and a verbatim byte slice of the silhouette mitigation as `mask_shape.py`) | Copied **byte-identical** from `tier1/src/edge_runner_pi5/`. `VENDOR.md` in that directory records the source path, the originating commit, the SHA-256 of every file, and the copy date. `tier1/tests/test_gait_mask_anon.py` re-verifies those digests. |

The copy exists so that the privacy measurements and the deployed artifact provably describe the same
bytes; **do not edit files in that directory**; fix them upstream and re-vendor.

Everything else under `tier1/`, `tier2_phone/`, `tier2_cloud/`, `tier3_restoration/` and
`evaluation/` is original to this project.

---

## Runtime libraries

| Component | Used for | Licence |
|---|---|---|
| ONNX Runtime (Python and Android, incl. the QNN build) | All CPU/NPU inference | MIT |
| OpenCV (`opencv-python`, OpenCV Android SDK) | Image and video processing throughout | Apache-2.0 |
| NumPy, SciPy, scikit-learn, scikit-image, matplotlib | Numerics and evaluation | BSD-3-Clause / BSD-3-Clause / BSD-3-Clause / BSD-3-Clause / PSF-style |
| MediaPipe | Face landmarks, selfie segmentation | Apache-2.0 |
| `rtmlib` | RTMPose runner | Apache-2.0 |
| `cryptography` | AES-128-GCM, RSA-4096-OAEP | Apache-2.0 **or** BSD-3-Clause (dual) |
| `insightface` | Apparent-gender classifier | MIT (**code**); its *models* are non-commercial research only |
| Ultralytics | YOLO11 export/tooling | **AGPL-3.0** |
| NCNN | ARM inference backend on the Raspberry Pi | BSD-3-Clause |
| PyTorch, torchvision | Evaluation adversaries and the cloud generator | BSD-3-Clause |
| LPIPS | Perceptual distance in the quality harness | BSD-2-Clause |
| FastAPI, Uvicorn, `zeroconf` | Optional LAN handoff service | MIT / BSD-3-Clause / LGPL-2.1 |
| `requests` | TTP public-key fetch | Apache-2.0 |
| Pillow | Image I/O in the cloud nodes | MIT-CMU (HPND) |
| FFmpeg | Transcoding and clip assembly | LGPL-2.1+ / GPL-2.0+ depending on build |

## Android

| Component | Licence |
|---|---|
| AndroidX (`core-ktx`, `appcompat`, `activity-ktx`), Material Components | Apache-2.0 |
| Kotlin stdlib, Gradle, the Android Gradle Plugin | Apache-2.0 |
| Qualcomm QNN runtime / ONNX Runtime QNN execution provider | Qualcomm proprietary SDK terms; review before redistribution |

`gradle/wrapper/gradle-wrapper.jar` is redistributed as part of the Gradle wrapper convention
(Apache-2.0).

## Cloud graph

The ComfyUI graphs in `workflows/tier2_cloud/` reference node types provided by third-party packs.
The packs themselves are **not** included; install them through ComfyUI-Manager.

| Component | Licence |
|---|---|
| ComfyUI | GPL-3.0 |
| `ComfyUI-WanVideoWrapper`, `ComfyUI-WanAnimatePreprocess` (Kijai) | Apache-2.0 |
| `ComfyUI-KJNodes` | GPL-3.0 |
| `ComfyUI-VideoHelperSuite` | GPL-3.0 |
| `ComfyUI-segment-anything-2` | Apache-2.0 (wraps SAM 2, Apache-2.0) |

`tier2_cloud/src/comfyui_custom_nodes/mirage_rtmpose/nodes.py` imports `pose_utils` from
`ComfyUI-WanAnimatePreprocess` at runtime, resolved as a sibling custom-node package.

## Models and weights

Full table with filenames, expected locations and sources:
[`models/README.md`](models/README.md). The licence-critical entries:

| Model | Licence | Consequence |
|---|---|---|
| **YOLO11n / YOLO11n-seg** (Ultralytics) | **AGPL-3.0** | Strong copyleft with a network-use clause. The binding constraint on this repository's own licence; see [`LICENSE_PENDING.md`](LICENSE_PENDING.md). |
| **`big-lama` inpainting weights** | **CC BY-NC-SA 4.0** | Non-commercial, share-alike. The LaMa *code* is Apache-2.0; the *weights* are not. |
| **InsightFace `genderage.onnx` / `buffalo_l`** | **Non-commercial research use only** | Bars commercial deployment of any build that includes them. |
| Robust Video Matting | GPL-3.0 | Legacy backend, not on the shipped path. |
| YOLOv10-M (cloud matte re-detection) | AGPL-3.0 | Same family of obligation as YOLO11. |
| RTMPose / MMPose, SAM 2, ViTPose | Apache-2.0 | Permissive. |
| EdgeFace-S-GAMMA (Idiap) | See the model card | Check before any non-research use. |
| Wan 2.2 Animate 14B and the LoRA stack | Apache-2.0 upstream; **each repack and each community LoRA carries its own terms** | Check every model card individually; LoRA licensing in this ecosystem is inconsistent. |

## Datasets

| Dataset | Used for | Terms |
|---|---|---|
| CASIA-B (COCO-17 pose CSVs, via the GaitGraph release) | Gait re-identification evaluation | Research use; obtain from the original distributors |
| MS-COCO | Referenced protocol and pretraining | CC BY 4.0 (annotations); images under Flickr terms |
| The project's own capture corpus | Coverage audits, silhouette evaluation, end-to-end runs | **Not redistributed.** It shows real people. |

## Attribution notes

* Upstream copyright headers in vendored files are preserved verbatim. `VENDOR.md` is part of the
  attribution and must travel with the files it describes.
* Removing or altering any notice in this document, in `VENDOR.md`, or in a vendored source file is
  a licensing defect, not a cleanup.
