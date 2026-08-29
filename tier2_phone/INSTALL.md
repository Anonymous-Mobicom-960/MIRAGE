# Installing the Tier 2 phone app

The app is the Tier 2 mobile stage: it reconstructs the background behind the removed person,
derives the illumination abstraction that is the only pixel stream sent to the cloud, and
composites the returned character back into the scene.

Measured on a **Samsung Galaxy S25 Ultra** (SM-S938B, Snapdragon 8 Elite, Android 16). Other
Snapdragon devices with a Hexagon NPU should work; nothing else has been tested.

---

## 1. Get the APK

A prebuilt debug APK is attached to the repository's Releases page (the URL is withheld while
the paper is under double-blind review; build from source with section 2 in the meantime):

```
mirage-tier2-s25ultra-<version>-qnn-debug.apk
```

It is debug-signed, so it installs directly without a developer account. It contains **no model
weights** — see section 3, and the licence note there for why.

```bash
adb install -r mirage-tier2-s25ultra-<version>-qnn-debug.apk
```

## 2. Or build it yourself

Requires JDK 17 and an Android SDK with API 35.

```bash
cd tier2_phone/app
./gradlew assembleDebug -Pmirage.enableOrt=true
# app/build/outputs/apk/debug/app-debug.apk
```

🔴 **`-Pmirage.enableOrt=true` is required.** It selects `onnxruntime-android:1.24.3` plus the
Qualcomm QNN packages. The default flavour resolves ONNX Runtime **1.22.0**, which does not have
`registerExecutionProviderLibrary` or `epDevices` — the API `OrtRunner` uses to place work on the
Hexagon NPU — so the default build **does not compile**. The QNN flavour is also the only one that
reaches the NPU at all, so it is what you want regardless.

## 3. Add the model weights after installing

**No weights ship with this project, and none is bundled in the APK.** Every model is third party
and several carry licences that forbid redistribution or restrict downstream use
(`big-lama` is CC BY-NC-SA 4.0, YOLO11 is AGPL-3.0, RVM is GPL-3.0). Each must be obtained from its
own upstream source, by you, under the licence that applies to your use.

> ⚠️ **Upgrading from a build made before 2026-08-29.** The application id changed with the
> project rename, `com.sitara.npu` -> `com.mirage.npu`, so this installs as a NEW app beside the
> old one rather than over it.
>
> * **Models must be pushed again.** Android scoped storage lets an app read only its OWN
>   `/sdcard/Android/data/<id>/` directory, so the new build cannot see the files under
>   `com.sitara.npu`. There is no fallback that could work here. Re-run the `adb push` block below,
>   then `adb uninstall com.sitara.npu` once the new build runs.
> * **Your clips are NOT affected.** The shared project folder moved from
>   `Project Body Sitara` to `Project MIRAGE`, but that one lives in `Download/`, which any app can
>   read, so the app falls back to the old folder when the new one is absent
>   (`MiragePaths.LEGACY_PROJECT_DIR_NAME`). Rename it when convenient; nothing breaks either way.

The app loads models **at runtime** from one directory on the device. That directory takes
precedence over anything baked into the APK, so models can be added or swapped with `adb push` and
**no reinstall**.

```
/sdcard/Android/data/com.mirage.npu/files/models/
```

| File | What it does | Needed for | Source | Licence |
|---|---|---|---|---|
| `lama.onnx` | background inpainting (Phase 1) | **required** | [advimman/lama](https://github.com/advimman/lama) (`big-lama`) | code Apache-2.0; **weights CC BY-NC-SA 4.0, non-commercial** |
| `seg.onnx` | person segmentation | on-device Tier-1 preview only | [Ultralytics YOLO11n-seg](https://github.com/ultralytics/ultralytics) | **AGPL-3.0** |
| `movenet.onnx` | pose estimation | on-device Tier-1 preview only | [Google MoveNet](https://www.kaggle.com/models/google/movenet) | Apache-2.0 |
| `rvm.onnx` | matting | on-device Tier-1 preview only | [PeterL1n/RobustVideoMatting](https://github.com/PeterL1n/RobustVideoMatting) | **GPL-3.0** |

Only `lama.onnx` is needed to run the Tier 2 pipeline. The other three are for the optional
on-device Tier-1 preview.

```bash
adb shell mkdir -p /sdcard/Android/data/com.mirage.npu/files/models/
adb push lama.onnx /sdcard/Android/data/com.mirage.npu/files/models/
```

Each ONNX must be exported to the filename in the table — the app looks the names up literally.

## 4. Stage a clip and run

The app reads its inputs from, and writes outputs to:

```
/sdcard/Download/Project MIRAGE/input/
/sdcard/Download/Project MIRAGE/output/
```

Push the Tier-1 artifacts, then the generated character once the cloud stage has returned it:

```bash
adb push masked_video.mp4        "/sdcard/Download/Project MIRAGE/input/"
adb push mask.mp4                "/sdcard/Download/Project MIRAGE/input/"
adb push synthetic_person_p1.mp4 "/sdcard/Download/Project MIRAGE/input/"
adb push synthetic_alpha_p1.mp4  "/sdcard/Download/Project MIRAGE/input/"
```

Then, in the app's Tier 2 tab, in order:

| | Phase | Produces |
|---|---|---|
| 1 | inpaint | `background_reconstructed.mp4` |
| 1b | lightmap | `light_map.mp4` — the only pixels that go to the cloud |
| 2 | composite | `composite.mp4`, and the final output |

Leave the **NPU** toggle on. It is on by default and is what places work on the Hexagon.

`synthetic_alpha_pK.mp4` is **required** alongside each `synthetic_person_pK.mp4`. Phase 2 refuses
to composite a layer that has no explicit alpha rather than falling back to a keyer — on a lightmap
plate a keyer keys almost the whole frame.

## 5. If something goes wrong

| Symptom | Cause |
|---|---|
| Phase 1 fails immediately | `lama.onnx` is missing from the models directory, or misnamed |
| Phase 2 throws `MissingAlphaException` | a character layer was pushed without its `synthetic_alpha_pK.mp4` |
| Build fails with `Unresolved reference 'epDevices'` | `-Pmirage.enableOrt=true` was omitted (section 2) |
| Everything runs but slowly | the NPU toggle is off, so ORT fell back to CPU |

Model provenance for every tier, with licences, is in [`../models/README.md`](../models/README.md).
