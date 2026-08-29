# Tier 2: Companion Phone

## Purpose

The companion phone is the second trusted domain. It turns the Tier-1 egress into a watchable video
**entirely on device**: it reconstructs the person-free background, abstracts it into a
low-frequency illumination stream that is safe to send to an untrusted service, and composites the
synthetic character the cloud returns over the reconstructed plate. The
real background never leaves the phone, and no raw bystander pixel ever reaches it in the first
place.

![Tier 2 on the phone: masked input, reconstructed background, illumination abstraction](../assets/gifs/tier2_phone_demo.gif)

## Input

The Tier-1 egress plus, for Phase 2, the cloud's return:

| File | Produced by | Content |
|---|---|---|
| `masked_video.mp4` | Tier 1 | Grey-filled anonymised video (H.264; Tier 1 writes lossless FFV1 and it is transcoded for the app's contract) |
| `mask.mp4` | Tier 1 | Binary occupancy map |
| `pose.json`, `face_scalars.json` | Tier 1 | Control sidecars, used for the Tier-1 preview cards |
| `synthetic_person_pK.mp4` | Tier 2 cloud | The generated character, one file per person slot |
| `synthetic_alpha_pK.mp4` | Tier 2 cloud | That character's alpha matte |

The bundle contract used for the reported runs is **1264 × 1264 at 30 fps** - 30 fps in, 30 fps out.

## Output

| File | Content |
|---|---|
| `background_reconstructed.mp4` | The person-free background plate. **Stays on the phone.** |
| `light_map.mp4` | The low-frequency illumination stream. This is the only pixel product allowed across the cloud boundary. |
| `composite.mp4` | Character alpha-composited over the reconstructed background |
| `final_output.mp4` | The protected video (H.264/AVC), the published copy of the composite |

## Dependencies

**App:** Android/Kotlin, Gradle 8.x, `compileSdk`/`targetSdk` 35, `minSdk` 31, ONNX Runtime
(`com.microsoft.onnxruntime:onnxruntime-android`; the QNN build additionally pulls
`com.qualcomm.qti:onnxruntime-android-qnn` and `qnn-runtime`). No `INTERNET` permission.

**Companion scripts:** [`requirements.txt`](requirements.txt): NumPy, OpenCV, and `ffmpeg` on
`PATH`.

## Models

Not redistributed; see [`../models/README.md`](../models/README.md).

| Role | Model | Runs on |
|---|---|---|
| Background inpainting | LaMa (`big-lama` family), ONNX | CPU execution provider |
| On-device Tier-1 preview (optional) | YOLO11n-seg, MoveNet, RVM | CPU / NNAPI |

Models are loaded at runtime from
`/sdcard/Android/data/com.mirage.npu/files/models/`, which takes precedence over anything bundled
into the APK, so a model can be replaced with `adb push`, without reinstalling.
`app/app/src/main/assets/models/MODELS_README.txt` records what each bundled slot expects.

## Configuration

| Where | What it controls |
|---|---|
| `app/app/build.gradle.kts` | Build flavour. `-Pmirage.enableOrt=true` selects the QNN build (`0.17-qnn`), which is the one that can reach the Hexagon NPU. |
| `RuntimeConfig.kt` | Runtime parameters (working resolution, lightmap and audit knobs). Overridable at runtime from `.../files/config.json`. |
| `BackgroundInpaint.kt` | The camera-motion branch thresholds (`R_MAX_BASE`, `SKIP_ALIGN_PX`) and `MOSAIC_ENABLED`. |
| `NCompositor.kt` | `requireExplicitAlpha` (default `true`) and `allowKeyerFallback` (default `false`). |
| App UI cards | Which phase to run, and the Phase-1 sharpness / seam-match sliders. Both sliders are pixel-affecting and are recorded on every Phase-1 evaluation row. |

## Usage

**Build and install:**

```bash
cd app
./gradlew assembleDebug -Pmirage.enableOrt=true          # QNN build
adb install -r app/build/outputs/apk/debug/app-debug.apk
adb push <model>.onnx /sdcard/Android/data/com.mirage.npu/files/models/
```

**Run a clip:**

```bash
# 1. Package a Tier-1 export into the app's input contract.
#    Tier 1 writes lossless FFV1 .mkv, which Android's MediaCodec cannot decode, so both streams
#    are re-encoded to H.264 High / yuv420p. The script then MEASURES that the mask survived the
#    transcode: a mask that SHRANK would reveal a sliver of a real person, so the check is on
#    added-vs-dropped pixels, not on IoU alone, and it falls back to lossless on any violation.
python companion_scripts/tier1_to_tier2.py --clip <input.mp4> --tag A

# 2. Push the bundle to the device.
adb push <bundle_dir> /sdcard/Android/data/com.mirage.npu/files/

# 3. On the phone: run Phase 1 (background) and Phase 1b (lightmap) from the Tier-2 card,
#    then pull light_map.mp4 back and send it to the cloud with the rest of the bundle.
adb pull /sdcard/Android/data/com.mirage.npu/files/out/light_map.mp4

# 4. After the cloud returns the character, author the authoritative alpha:
python companion_scripts/alpha_from_tier1.py --selftest
python companion_scripts/alpha_from_tier1.py --clip <clip_dir> --slot p1 \
    --from-cloud <from_cloud_dir> --out <alpha_dir>

# 5. Push the character + alpha back, then run Phase 2 (composite)
#    from the app's cards, and pull final_output.mp4.
```

The phases are driven from the app's own UI cards; there is no headless runner, and the
device-automation harness used for the reported runs is tied to one app build's accessibility tree
and is not part of this release. `make_sidecars.py` produces the silhouette/alpha sidecars used when
testing the compositor in isolation.

> ⚠️ **Never `-c copy` and never `-crf 0` when preparing the mask.** `-c copy` preserves FFV1 inside
> an `.mp4` wrapper, which desktop tools read happily and the phone cannot open at all. `-crf 0`
> emits High 4:4:4 Predictive, which the device decoder rejects; every frame then decodes as null
> and the background stage silently falls back to a whole-frame colour guess. `tier1_to_tier2.py`
> uses `-profile:v high -crf 6 -g 1 -bf 0`, verified bit-identical after the consumer's own
> threshold. Verifying a phone-bound file with a desktop decoder proves nothing about the phone.

## Pipeline

```text
Phase 1  BackgroundInpaint.kt
         global-motion pre-pass -> devC (worst trajectory deviation), compared with R_MAX_BASE
           devC < SKIP_ALIGN_PX      STATIC   no alignment
           SKIP_ALIGN_PX..R_MAX_BASE JITTER   3-level pyramidal alignment
           devC > R_MAX_BASE         DYNAMIC  8-DOF Gauss-Newton homography + pan-sized mosaic
         -> per-pixel temporal trimmed mean over REAL pixels only
         -> LaMa fills the never-revealed core, once
         => background_reconstructed.mp4                       (never leaves the device)

Phase 1b LightmapPhase.kt
         downsample -> Gaussian -> upscale; low-frequency luminance only
         => light_map.mp4                                      (the only pixel product sent to the cloud)

Phase 2  NCompositor.kt
         character over the reconstructed plate, using the EXPLICIT alpha matte
         => composite.mp4, published as final_output.mp4       (H.264/AVC)
```

## Expected Files

```text
tier2_phone/
├── app/                                Android Studio project
│   ├── app/src/main/java/com/mirage/npu/    the whole pipeline, one package
│   ├── app/src/main/assets/models/          MODELS_README.txt (weights are pushed, not bundled)
│   └── gradlew, settings.gradle.kts, ...
├── companion_scripts/
│   ├── tier1_to_tier2.py               Tier-1 export -> the app's input contract, with a §2 check
│   ├── alpha_from_tier1.py             authors the authoritative alpha (Tier-1 domain ∧ generated shape)
│   └── make_sidecars.py                silhouette/alpha sidecars for testing the compositor alone
└── docs/ALPHA_MATTE.md                 why the explicit matte is required, and how it is sourced
```

## Notes

* **The explicit alpha is required, not preferred.** `requireExplicitAlpha = true` is the default: a
  character layer without its own `synthetic_alpha_pK.mp4` raises `MissingAlphaException` *before a
  single frame is decoded*, rather than silently falling back to a keyer. On a lightmap plate a keyer
  keys almost the entire frame. The keyer survives only as a logged, default-off opt-out.
* **The authoritative alpha is authored off-device**, by `companion_scripts/alpha_from_tier1.py`:
  Tier-1's per-slot mask supplies the spatial **domain**, the generated pixels supply the **shape**.
  The cloud's own SAM2 matte is one candidate shape input, never used bare.
* **The DYNAMIC background branch has not been verified on a device.** It compiles and is measured
  against local references. The STATIC/JITTER branch is the one exercised on hardware.
* **`MiGanNpu.kt` is present but unreachable.** `NpuFactory.createMiGanOrNull()` has no callers. It
  is not part of the shipped pipeline; LaMa is.
* `companion_scripts/alpha_from_tier1.py` imports the blockify constants from
  `../../tier2_cloud/scripts/check_render.py` so that the sampler's repaintable-hole geometry is
  defined in exactly one place, shared with the cloud-side render check.
