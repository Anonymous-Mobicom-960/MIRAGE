# Pipeline

Every stage in execution order, with what it consumes, what it does, what it produces, and what
consumes that next. Field names, shapes and formats below are read from the implementation and from
the real artifacts in [`../examples/outputs/`](../examples/outputs/); anything not verifiable from
the code is marked *implementation dependent*.

---

## Stage 1: Detection and tracking

| | |
|---|---|
| **Runs on** | Wearable (Raspberry Pi 5 prototype) |
| **Code** | `tier1/src/mirage/pipeline.py`, `blur_yolo11n.py`, `export_tracking.py` |
| **Input** | Decoded BGR frames from a video file or camera |
| **Processing** | YOLO11n person detection (confidence threshold 0.50) every *n* frames (default `--skip-n 5`); a greedy nearest-centroid tracker with grace frames assigns a stable slot per person; skipped frames propagate boxes and keypoints by pyramidal Lucas-Kanade optical flow |
| **Output** | Per-frame, per-slot bounding boxes and stable identity indices, in memory |
| **Next** | Stages 2, 3, 4, 5 all consume the tracked boxes |

Read, process and write run on separate threads with bounded queues. Queued frames flush after
capture ends, so inference back-pressure never drops an incoming frame.

---

## Stage 2: Fail-closed redaction

| | |
|---|---|
| **Runs on** | Wearable |
| **Code** | `tier1/src/mirage/mask_anon.py`, `vendor/mirage_edge/mask_shape.py` |
| **Input** | Tracked boxes; the current frame |
| **Processing** | Per-person region becomes an axis-aligned bounding rectangle (`mask_shape_mode=bbox`), unioned across a 2-frame temporal window, merged on intersection; the region is filled with uniform grey |
| **Output** | `masked_video.mkv` (FFV1, lossless) and `mask.mp4` (binary occupancy) |
| **Next** | Stage 8 (phone background reconstruction); the mask also reaches Stage 10 (cloud) and Stage 12 (alpha authoring) |

The mitigation returns `simplified ∪ smeared ∪ current`, so the emitted mask is a **superset** of
the frame's own mask. Mitigation can add grey; it can never retract and reveal. The masked video is
written lossless on purpose: a lossy codec lets DCT ringing re-expand the decoded mask past the
erode margin and leak a ring of real background into the cloud-bound region.

---

## Stage 3: Pose extraction

| | |
|---|---|
| **Runs on** | Wearable |
| **Code** | `tier1/src/mirage/pose.py` (RTMPose via `rtmlib`) |
| **Input** | Frame + tracked box per person |
| **Processing** | RTMPose-Tiny, model input 192 x 256, ONNX Runtime CPU |
| **Output** | 17 COCO body keypoints per person per frame as `(x, y, confidence)` in native pixel coordinates |
| **Next** | Stage 4 |

---

## Stage 4: Identity neutralisation of the skeleton

| | |
|---|---|
| **Runs on** | Wearable |
| **Code** | `tier1/src/mirage/gait_anon.py` calling `vendor/mirage_edge/pose_anon_edge.py` |
| **Input** | The raw keypoint sequence for one slot, compacted to the frames in which that slot is actually present |
| **Processing** | Whole-clip transform: least-squares fit of a canonical anthropometric template to the subject's vertical extent, seeded global-scale jitter of at most 0.10, 14 deg / 10 deg joint-angle perturbation on the arm chains only, limb-swing amplitude rescale 0.25 about each chain's mean pose, per-chain constant time shift of at most 0.35 s, free-end keypoint pruning. Cadence re-timing disabled; the root trajectory stays locked to the subject's true path. Seed drawn fresh per sequence with `secrets`. |
| **Output** | A **new** keypoint sequence; confidences untouched, then binarised to `{0, 1}` at 0.5 |
| **Next** | Stage 7 (`pose.json`) |

Absent slots must be compacted out before the transform: a dense all-zero row drives a median bone
length to 0, and the length factor `target / (0 + 1e-6)` then emits coordinates around 1e10 px. The
adapter compacts, transforms and scatters back; `tier1/tests/test_gait_anon.py` keeps a positive
control that the raw call still exhibits the failure, so the guard cannot silently stop mattering.

---

## Stage 5: Expression extraction and DP release

| | |
|---|---|
| **Runs on** | Wearable |
| **Code** | `tier1/src/mirage/face_canonical_v2.py`; DP stage in `tier1/src/edge_runner_pi5/config.py` + `mirage_tier1.py` |
| **Input** | Face crop localised from the nose and eye keypoints |
| **Processing** | MediaPipe FaceLandmarker (468 points) reduced to 12 scalars: mouth openness, mouth width, smile intensity, left/right eye openness, left/right brow raise, head yaw, pitch, roll, gaze x, gaze y. Calibrated Laplace noise is then added under a whole-clip budget split non-uniformly across the 12 channels. |
| **Output** | `face_scalars.json`, a JSON array shaped `[frames][persons][12]`, floats, in slot order |
| **Next** | Stage 11 (rendered onto a generic canonical face template for the cloud) |

At the shipped budget the released series is, to the precision of the available measurements,
indistinguishable from pure Laplace noise. The downstream renderer's correct behaviour is therefore
a stable neutral pose plus a data-independent idle animation, not an attempt to recover a
trajectory that is not there.

---

## Stage 6: Attribute flag, embedding and encryption

| | |
|---|---|
| **Runs on** | Wearable |
| **Code** | `tier1/src/mirage/gender.py`, `embedding.py`, `tracking.py`, `encryption.py` |
| **Input** | The highest-rank-score face crop per person stream (detection confidence x face quality), and the person crops |
| **Processing** | InsightFace `genderage.onnx` produces one categorical apparent-gender label per stream. EdgeFace-XS (gamma = 0.6) produces one 512-d L2-normalised embedding per stream, 112 x 112 input. Person crops are JPEG-encoded at quality 70, the rate-distortion knee identified in the paper's storage evaluation. Crops and embedding are encrypted with AES-128-GCM (128-bit key, 96-bit nonce, 128-bit tag); the AES key is wrapped with RSA-4096-OAEP-SHA256 under the TTP's public key, fetched over HTTP(S) at start-up. |
| **Output** | `crypto/stream_<uuid>.packet`, `crypto/stream_<uuid>.key` |
| **Next** | Stage 14 (Tier 3), only if a bystander consents |

---

## Stage 7: Egress

| | |
|---|---|
| **Runs on** | Wearable |
| **Output** | The complete set of things allowed off the device |

`pose.json`:

```jsonc
{
  "adapter":       { ... },              // which writer produced this file, and its role
  "anon":          { ... },              // effective anonymisation config: preset, level, every kwarg,
                                         // template kind per slot, seed policy, mask settings
  "person_count":  1,
  "slot_policy":   "x-sorted+hysteresis (vendored person_slots; slot ids assigned by the Tier-1 host)",
  "emitted_slots": 1,
  "fps":           30.0,
  "size":          [1264, 1264],
  "frames": [                            // [frames][persons]
    [ { "slot": 0,
        "kp":    [[x, y], ...],          // 133 rows, native pixel coords, float
        "score": [s, ...],               // 133 values, binarised to {0.0, 1.0}
        "track": 1 } ]
  ]
}
```

The 133 rows are the wholebody layout: body `0..16`, feet `17..22`, face `23..90` **deliberately
zeroed**, hands `91..132`. In the example artifact only 8 of the 133 rows are non-zero, because the
free-end prune drops unsupported termini rather than inventing them.

`manifest.json` carries per slot: `stream_id`, `gender` and `gender_confidence`,
`frames_with_face`, `packet_file`, `key_file` and skeleton-containment diagnostics, plus the full
effective configuration.

**Next:** Stage 8.

---

## Stage 8: Background reconstruction (phone)

| | |
|---|---|
| **Runs on** | Companion phone |
| **Code** | `tier2_phone/app/.../BackgroundInpaint.kt` |
| **Input** | `masked_video.mp4` + `mask.mp4`, transcoded to H.264 by `tier1_to_tier2.py` |
| **Processing** | Global-motion pre-pass gives `devC` (worst trajectory deviation), which selects a branch: no alignment (static), 3-level pyramidal alignment (jitter), or 8-DOF Gauss-Newton homography with a pan-sized mosaic canvas (dynamic). Per-pixel temporal trimmed mean over real pixels only; LaMa fills the never-revealed core once. |
| **Output** | `background_reconstructed.mp4`, which **stays on the device** |
| **Next** | Stage 9 |

---

## Stage 9: Illumination abstraction (phone)

| | |
|---|---|
| **Runs on** | Companion phone |
| **Code** | `tier2_phone/app/.../LightmapPhase.kt` |
| **Input** | The reconstructed background |
| **Processing** | Downsample, Gaussian spatial filter, rescale to original resolution. Retains global colour distribution, ambient gradient and lighting direction; attenuates object boundaries. |
| **Output** | `light_map.mp4`, the only pixel product permitted across the cloud boundary |
| **Next** | Stage 10 |

---

## Stage 10: Bundle assembly (the local privacy firewall)

| | |
|---|---|
| **Runs on** | Workstation / companion |
| **Code** | `tier2_cloud/scripts/build_cloud_bundle.py`, `npy_to_mirage_emit.py`, `tier1_viz.py`, `face_signal_filter.py` |
| **Input** | The Tier-1 export and `light_map.mp4` |
| **Processing** | Convert the dense `.npy` export to `pose.json` / `face_scalars.json`; composite the Tier-1 grey fill over the lightmap; split the union mask **per pixel by nearest pose** into per-slot masks; render the anonymised skeleton and the canonical face mesh to video; copy the supplied reference sheet |
| **Output** | `to_cloud/{masked_video_*, mask_*, mask_pK_*, pose_sticks_pK_*, facemesh_pK_*, light_map, reference_pK_640.png, MANIFEST.json, REFERENCE_IMAGES_README.txt}` |
| **Next** | Stage 11 |

The mask split is **per pixel by nearest keypoint**, not by connected component. A component rule
assigns a whole blob to one slot, which is correct only while two people are separated; when they
touch, the mask has one component, both slots select it, and the two emitted masks come out
byte-identical, which is exactly the condition that makes the cloud generate two characters and
bind the matte to the wrong one.

`verify_bundle.py` checks the contract and `audit_bundle.py` re-checks it against the Tier-1 and
Tier-2 sources before anything is uploaded.

---

## Stage 11: Cloud synthesis

| | |
|---|---|
| **Runs on** | GPU server (ComfyUI) |
| **Code** | `workflows/tier2_cloud/*.json` + `tier2_cloud/src/comfyui_custom_nodes/` |
| **Input** | Exactly five loaded videos (plate, occupancy mask, anonymised skeleton, canonical face mesh, lightmap) plus the reference PNG |
| **Processing** | `GrowMaskWithBlur(expand 10, blur 4)` then `BlockifyMask(16)` defines the repaintable hole; the video-diffusion sampler runs at steps 5 / cfg 1.0 / `dpm++_sde` / shift 5.0 with a 77-frame window and a 5-slot LoRA stack; the generated video is then re-detected and segmented with SAM2 to produce a matte |
| **Output** | `synthetic_person_pK.mp4`, `synthetic_alpha_pK.mp4` |
| **Next** | Stage 12 |

Resolution and rate are read at runtime from the loaded Tier-1 artifact. There is no frame-rate
conversion at any stage: whatever rate Tier-1 emits is the rate the cloud renders and the phone
composites, so a 30 fps capture is 30 fps end to end.

🔴 **The artifacts in this release are 10 fps, not 30.** `examples/outputs/tier1/pose.json` and
`manifest.json` both record `fps: 10.0` over 100 frames, and the appearance corpus is decimated to
10 fps by design. Read every reported latency, energy and dynamics number as measured at that rate
unless it says otherwise.

This is not merely cosmetic. `MASK_TEMPORAL_WIN` is derived from a **duration**
(`MASK_TEMPORAL_S = 0.14`), so it resolves to 2 frames at 15 fps, **1 at 10 fps**, and 4 at 30 fps.
The shipped decision is a frame count of 2, worth -9.41 pp of silhouette re-identification, so any
run that changes the emit rate must pin `MIRAGE_MASK_TEMPORAL_WIN=2` or silently give that back.
`tier1/src/edge_runner_pi5/config.py:511` carries the same warning at the definition.

---

## Stage 12: Alpha authoring

| | |
|---|---|
| **Runs on** | Workstation |
| **Code** | `tier2_phone/companion_scripts/alpha_from_tier1.py` |
| **Input** | Tier-1's per-slot mask, the generated character, the cloud's matte, the lightmap |
| **Processing** | `DOMAIN = grow_blockify(mask_pK)`, the exact hole the sampler was allowed to repaint. `SHAPE` is the cloud matte's components that live inside `DOMAIN`, else a painted-difference key `abs(generated - light_map) > threshold`. `alpha = union(selected components) ∧ DOMAIN`. Which source was used is recorded per layer. |
| **Output** | `synthetic_alpha_pK.mp4` (authored) and a JSON report including `tier1_boundary_frac`, the fraction of the matte's own boundary that lies on the domain edge |
| **Next** | Stage 13 |

Refuses rather than guesses: if nothing survives, if coverage leaves the plausible band, or if the
matte is simply the box, it fails loudly.

---

## Stage 13: Compositing (phone)

| | |
|---|---|
| **Runs on** | Companion phone |
| **Code** | `tier2_phone/app/.../NCompositor.kt` |
| **Input** | `background_reconstructed.mp4`, `synthetic_person_pK.mp4`, `synthetic_alpha_pK.mp4` |
| **Processing** | Depth-ordered alpha composite over the reconstructed plate. `requireExplicitAlpha = true`: a layer without its own matte raises `MissingAlphaException` before any decode. Encode as H.264/AVC (not HEVC), bitrate `max(12 Mbps, w*h*fps/3)` capped at 60 Mbps. |
| **Output** | `composite.mp4`, published as `final_output.mp4` - the protected video |
| **Next** | Delivery, or Stage 14 on consent |

---

## Stage 14: Consent-based restoration

| | |
|---|---|
| **Runs on** | TTP, then the companion phone |
| **Code** | `tier3_restoration/scripts/ttp_stub.py`; primitives in `tier1/src/mirage/encryption.py` |
| **Input** | `crypto/stream_<uuid>.{packet,key}` over TLS, never media |
| **Processing** | RSA-4096 unwrap; cosine-similarity match against the pre-registered template database (conservative threshold 0.65) with fragmented tracks merged; consent prompt disclosing only a session identifier and non-sensitive metadata; release of only the AES keys for that track and interval; AES-128-GCM decryption and re-compositing on the phone |
| **Output** | The authorised bystander's original appearance, restored over their synthetic track. Everyone else stays synthetic. |
| **Next** | n/a |

The matching, consent and release steps are **not implemented in this release**; see
[`../tier3_restoration/README.md`](../tier3_restoration/README.md).
