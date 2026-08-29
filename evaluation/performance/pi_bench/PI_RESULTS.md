# Raspberry Pi 5 - YOLO11n-seg inference latency (NCNN vs ONNX)

Closes the deferred "NCNN + Pi-5 latency" question left open by the Tier-1 segmentation-model
selection study: which runtime is fastest for the shipped segmenter on the wearable target.

**Device:** Raspberry Pi 5 Model B Rev 1.1 · Cortex-A76 ×4 @ 2.4 GHz · 8 GB · Debian 13
(trixie) / Python 3.13 / aarch64. **Governor:** performance. **Power:** official-grade PSU
(after an inadequate supply brown-out-reset the board mid-run - see note). Throttle `0x0`
before AND after (temp 48→65 °C) → **numbers are valid, not power-limited.**

**Method:** pure model-inference latency (the runtime-dependent part; mask pre/post is
runtime-independent and excluded). Random input, 8 warmup + 60 timed iters, 4 threads
(= core count). NCNN fp16 (use_fp16 packed/storage/arith, Vulkan off); onnxruntime CPU-EP.
`pi_bench/bench_pi.py`.

## Result (median ms, lower = faster)

| runtime            | @192  | @256  |
|--------------------|-------|-------|
| ONNX-fp32 (ORT CPU)| 16.4  | 26.6  |
| ONNX-int8 (ORT CPU)| 17.8  | 29.2  |
| **NCNN-fp16**      | **8.1** | **14.6** |

## Findings
- **NCNN-fp16 is the clear winner: ~2× faster than ONNX** (192: 2.0×, 256: 1.8×). Confirms the
  model-selection hypothesis that NCNN is the ARM speed-leader for the Pi-5 glasses target.
  → **Pi-5 seg should run NCNN-fp16**, not ONNX.
- **ONNX INT8 is SLOWER than fp32 on the Pi (17.8 > 16.4; 29.2 > 26.6)** - the OPPOSITE of the
  S25 Ultra, where dynamic INT8 was ~1.4× faster (see `reports/ARM_RESULTS.md`). onnxruntime's
  weight-only dynamic INT8 does not pay off on the A76: the dequant overhead outweighs any gain.
  → per-device runtime choice differs: **S25 → ONNX-INT8, Pi-5 → NCNN-fp16.**
- NCNN-fp16 @192 = 8.1 ms (~123 inf/s pure), @256 = 14.6 ms (~68 inf/s). End-to-end FPS is
  gated by the mask post-processing (~65-80 ms on ARM per ARM_RESULTS), not inference.

## Caveats
- Pure inference only; full pipeline adds the cv2 guided-filter + morphology post-proc.
- fp16 NCNN vs fp32/int8 ONNX is the deployable comparison, but not a numerical-parity A/B;
  mask-quality parity of the NCNN fp16 export vs the ONNX path is unverified (latency study).
- The A76 result is a Pi-5-specific runtime choice; it does not change the platform-independent
  privacy config (guided mask, d4, FaceGuard) - that is runtime-agnostic.
- **Power lesson:** a Pi 5 at `performance` under a 4-thread NN load needs a real 5 V/5 A PSU;
  an inadequate supply brown-out-resets it (and the unclean shutdown corrupted the venv + pushed
  files, which had to be reinstalled/re-pushed with size verification). Always fsync-verify pushes.

---

## FULL Tier-1 pipeline end-to-end (Pi 5 A76, 4 threads, 1264² frame, seg@256)

Throttle `0x0` throughout (valid, 2.4 GHz, 51→58 °C). `bench_tier1_pi.py`.

| stage | median ms |
|---|---|
| 1. seg (NCNN yolo11n-seg @256, forward) | 15.0 |
| 2. mask post-proc (guided fast-filter + morph + d4 @1264) | 19.5 |
| 3. grey-fill (1264²) | 16.3 |
| 4. FaceGuard (keypoint ellipse-fill) | 0.4 |
| 5. RTMPose (rtmlib lightweight = rtmpose-**s**) | 50.7 |
| 6. anti-reID anonymize (60-frame clip) | 25.7 |
| **END-TO-END** | **127.7 → 7.8 FPS** |
| END-TO-END fast-post (mask ws0.5/d8 = 6.0 ms) | 114.2 → **8.8 FPS** |

**Read:** ~**7.8 FPS** full pipeline (~8.8 with fast-post). Dominant cost is **RTMPose (50.7 ms)**, not
the mask (19.5 ms) - but rtmlib's "lightweight" loaded rtmpose-**s**, not the spec'd rtmpose-**t**; -t is
~2× smaller, so real pose ≈ 20-25 ms and end-to-end ≈ **~100 ms → ~10 FPS**. Other optimization targets:
grey-fill (16 ms - a full-frame copy+mask; in-place `np.putmask` would cut it) and anonymize (25.7 ms - 
Python loops in `anonymize_v2`, vectorizable).

**Caveats:** seg = NCNN forward only (decode/NMS adds a few ms); FaceGuard measured as the keypoint
ellipse-fill (mediapipe has no py3.13/aarch64 wheel - production reuses RTMPose head kpts, so no separate
model); pose is rtmpose-s not -t (overcounts). A `-t` re-measure + grey/anon vectorization would tighten it.

---

## OPTIMIZED for >=15 FPS (masking + pose sticks + canon) -- MEASURED

Change: RTMPose(onnxruntime, rtmpose-s) -> **YOLO11n-pose NCNN-fp16** (single-shot, COCO-17); grey-fill
vectorized (np.where); canon-only anonymize (dynamic-perturb/reID deferred). `bench_tier1_opt.py`.

| stage | ms |
|---|---|
| seg NCNN @256 | 14.7 |
| pose YOLO11n-pose NCNN @256 | 10.5  (was RTMPose 50.7) |
| mask post-proc guided d4 | 19.4 |
| grey-fill vectorized | 6.3  (was 16.3) |
| FaceGuard | 0.2 |
| canon-only (per-clip) | 20.5 / 60f = 0.34 ms/frame amortized |
| **MEASURED END-TO-END** | **60.5 ms -> 16.5 FPS** |

pose NCNN forward: @192 6.5ms, @256 10.5ms (vs RTMPose-s onnxruntime 50.7ms = 4.8x).
Caveat: seg+pose = NCNN forward only; decode/NMS (~5-8ms) not yet in loop -> ~15 FPS with decode.
Comfortable margin: seg@192 / pose@192 / fast-post mask each add headroom -> ~18-22 FPS with decode.
Further headroom (research pending): Pi5 VideoCore-VII NCNN-Vulkan, int8, frame-skip/tracking. (Hailo excluded.)
