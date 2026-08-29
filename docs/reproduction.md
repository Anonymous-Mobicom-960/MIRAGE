# Reproduction

What this repository lets you reproduce, what it does not, and how to do the former.

For a checked, single-command path through the computational metrics (re-identification and
quality), see [`../reproduce/`](../reproduce/): it pins the shipped configuration in
`config.yaml`, records the paper's numbers with tolerances in `expected_results.json`, and
compares your measured results against them.

## Summary of what is reproducible

| Result class | Reproducible from this repository? |
|---|---|
| Running the full pipeline end to end on your own footage | **Yes**, given the three platforms and the model downloads |
| Pose/gait re-identification results (CASIA-B, frozen and retrained adversaries) | **Yes**, after downloading the dataset and the adversary checkpoints |
| Silhouette re-identification results | **Yes**, after obtaining OpenGait and its checkpoint |
| Capture-boundary coverage audits on your own footage | **Yes** |
| Visual-utility and FID metrics | **Yes**, on your own runs |
| Phone energy / thermal / sustained-throughput measurements | **Yes**, with a compatible device |
| Raspberry Pi 5 throughput | **Yes**, with the hardware |
| **The reported detection AP / AR numbers** | **No**, see [Gaps](#gaps) |
| **The reported numbers on the authors' own capture corpus** | **No**, that footage shows real people and is not bundled with this code repository |

## Environment

Python 3.10 was used throughout. Tier 1 and the companion tooling are torch-free; the evaluation
harnesses are not.

```bash
python -m venv .venv && source .venv/bin/activate
pip install -r tier1/requirements.txt
pip install -r tier2_cloud/requirements.txt
pip install -r tier2_phone/requirements.txt
pip install -r evaluation/requirements.txt          # heavy: torch, torchvision, lpips
```

`ffmpeg` must be on `PATH`.

For the evaluation harnesses a CUDA GPU is strongly recommended, because the adaptive adversary is
*retrained*, not just evaluated. The reported retraining ran on an RTX 4060.

## Model setup

No weights are redistributed. Follow [`../models/README.md`](../models/README.md), which
gives, for every model: the upstream source, the exact filename the code opens, the directory it
must sit in, and the licence.

Sanity check before running anything:

```bash
python tier1/tests/test_gait_anon.py                    # 30 checks, no models needed
python tier1/tests/test_gait_mask_anon.py               # 12 checks, no models needed
python tier1/src/edge_runner_pi5/test_config_contract.py
```

`test_gait_mask_anon.py` also verifies that the vendored re-identification defence modules are
still byte-identical to their upstream copies in `tier1/src/edge_runner_pi5/`. If it reports a
digest mismatch, the defences in your checkout are **not** the ones the published numbers were
measured on.

## Dataset setup

### Pose / gait re-identification (CASIA-B)

Not redistributed. From `evaluation/privacy/gait/`:

```bash
mkdir -p data vendor
curl -L -o data/data.zip \
  https://github.com/tteepe/GaitGraph/releases/download/v0.1/data.zip
cd data && unzip -o data.zip casia-b_pose_test.csv casia-b_pose_train_valid.csv && cd ..

curl -L -o data/gaitgraph_resgcn-n39-r8_coco_seq_60.pth \
  https://github.com/tteepe/GaitGraph/releases/download/v0.1/gaitgraph_resgcn-n39-r8_coco_seq_60.pth

git clone --depth 1 https://github.com/tteepe/GaitGraph.git  vendor/GaitGraph
git clone --depth 1 https://github.com/tteepe/GaitGraph2.git vendor/GaitGraph2
```

`data/` and `vendor/` are gitignored.

### Silhouette re-identification

`evaluation/privacy/silhouette/` expects OpenGait and a GaitBase checkpoint under its own `vendor/`
and `data/`. See `evaluation/privacy/silhouette/README.md` and `PROTOCOL.md`.

### Your own footage

`examples/inputs/` is empty by design. The capture corpus used for the reported results shows real
people and is not bundled here. Any MP4 of people walking will exercise the pipeline; the
end-to-end runbook assumes 30 fps.

## Execution

### Privacy: pose / gait channel

Run in this order from `evaluation/privacy/gait/`:

```bash
python reproduce_baseline.py        # GATE: the frozen adversary must reproduce published CASIA-B
python run_tm12.py                  # TM1 / TM2: naive cross-domain, and per-identity vs per-sequence seeding
python run_redesign.py              # ablation of the shipped transform
python sweep_pareto.py              # privacy/utility ladder against the frozen adversary
python tm3_retrain.py --config <c> --epochs <N>    # TM3: the adversary RETRAINED on protected output
python plot_results.py
```

**Do not stop at the frozen adversary.** TM3, retraining on protected output, is the decisive
measurement, and the two have repeatedly disagreed by large factors in both directions. A number
from one bounds the other in neither direction.

The threat models are: **TM1** clean gallery / protected probe; **TM2-id** and **TM2-seq** both
protected, seeded per identity vs per sequence (this is what separates pseudonymity from real
protection); **TM3** the retrained adversary. In the paper's terms, the unlearned hand-crafted
attacker is adversary Class 3 and the TM3 retrained attacker is Class 4.

### Privacy: silhouette channel

```bash
cd evaluation/privacy/silhouette
python run_silhouette_reid.py       # see the module header for arm selection
python cmc_opengait.py
```

In the paper's terms, the GEI nearest-neighbour attacker is adversary Class 1 and the frozen
GaitBase attacker is Class 2. Rank arms by lift over **each arm's own measured null**, not by raw
accuracy. A protocol that normalises the canvas from the clean person flatters exactly the smooth,
outward-only family of defences it then ranks highest; `PROTOCOL.md` records why the harness does
not do that.

### Privacy: capture boundary

Run against your own clip and its Tier-1 output:

```bash
python evaluation/privacy/capture_boundary/audit_unmasked.py <clip.mp4> <mask.mp4> AUDIT_UNMASKED.json
python evaluation/privacy/capture_boundary/audit_s2_autogrey.py            # needs MIRAGE_YUNET_ONNX
python evaluation/privacy/capture_boundary/face_coverage_eval.py
```

Reference outputs from a real run are in [`../examples/outputs/tier1/`](../examples/outputs/tier1/).

`audit_s2_autogrey.py` reads the fill level out of the video rather than assuming grey 128. That
matters: the edge runner writes lossless FFV1 whose flat fill is exactly 128, while the capture
service writes lossy H.264 whose flat fill lands near 124. An audit hardcoding 128 ± 1 reports
severe face reveals on fully covered frames.

### Quality

```bash
python evaluation/quality/eval_tier2_visual_utility.py --help
python evaluation/quality/eval_fid_pooled.py --gen <final_output.mp4> ... --real-glob '<clips>/*.mp4'
python evaluation/quality/leak_audit.py --help
```

### Performance

```bash
# Phone (needs adb; set MIRAGE_ADB if adb is not on PATH)
python evaluation/performance/phone_thermal.py
python evaluation/performance/phone_battery_report.py
python evaluation/performance/phone_sustained_report.py

# Raspberry Pi 5
python evaluation/performance/pi_bench/bench_tier1_pi.py
```

## Expected output directories

| Harness | Writes to |
|---|---|
| Tier 1 | `--export-dir` |
| Bundle builder | `--out` |
| Cloud render | the ComfyUI server's `output/` |
| Phone phases | `/sdcard/Android/data/com.mirage.npu/files/out/` |
| Gait evaluation | `evaluation/privacy/gait/reports/` |
| Silhouette evaluation | `evaluation/privacy/silhouette/reports/` |
| Phone performance | beside the script, or `MIRAGE_OUT_JSON` / `MIRAGE_POWER_CSV` |

## Non-determinism and seeds

* **The anonymisation seed is deliberately non-deterministic.** `new_clip_seed()` draws fresh
  entropy from `secrets` for every clip, so two runs on the same input produce different, and both
  correct, outputs. `MIRAGE_TEST_FIXED_SEED` makes it deterministic **for tests only**; output
  produced under it is a test artifact and is not privacy-safe to ship. The code says so at
  runtime.
* **The diffusion sampler is seeded per render** in the workflow JSON. The published graphs carry
  the seeds the reported renders used, but generation is not bit-reproducible across GPU models,
  driver versions or attention backends.
* **Adversary retraining** (TM3 / Class 4) is stochastic. Report a mean over seeds with its
  spread, not a single run. The differences between candidate configurations are frequently
  smaller than the between-seed spread.
* Phone thermal and energy measurements depend on ambient temperature, screen state and charge
  state. The harnesses tag every row with the screen and power-save flags Android reports, because
  a screen-on figure and a screen-off figure are not comparable.

## Device-specific requirements

* **The Raspberry Pi 5 path uses NCNN** for detection. On the Pi, prefer
  `opencv-contrib-python` if you want the guided-filter mask edge; install exactly one of
  `opencv-python` / `opencv-contrib-python`.
* The cloud graph was run on ComfyUI 0.18.2 with an A100 80 GB. Any CUDA GPU that can host the
  video-diffusion model should work; the operating point is asserted by the queue script, not
  inferred from the hardware.

## Gaps

These are stated so nobody spends time looking for them.

1. **The detection AP / AR evaluation is not in this repository.** The capture-service source
   refers to `results/tier1_detection_eval/` for the reported average-precision and average-recall
   figures and their ground-truth annotations. That directory is not present in the working tree
   this release was curated from. The numbers cannot be reproduced from this repository; refer to
   the paper.
2. **The consent server does not exist here.** Identity matching, consent dispatch and scoped key
   release are design-only. `tier3_restoration/scripts/ttp_stub.py` stands in for the one endpoint
   Tier 1 calls, and the TTP-side unwrap and decrypt primitives are present, so the chain can be
   exercised. The protocol itself is not implemented.
3. **The capture corpus is not bundled here.** It shows real people and is distributed separately
   from this code repository. Results measured on it, including the silhouette-channel numbers and
   the end-to-end coverage audits, can be *re-run* on your own footage, and reproduced exactly only
   with access to that dataset.
4. **The iOS companion benchmark is not in this release.** The paper reports the companion stages
   on an iPhone 15 Pro Max for cross-platform comparison; only the Android implementation ships
   here.
5. **The dynamic-camera background branch has never run on a device.** It compiles and is measured
   against local references; the static/jitter branch is the one exercised on hardware. A device
   run of it is a test of that branch, not a demonstration of it.
6. **Model weights are third-party and several are not redistributable.** Some carry licences that
   restrict commercial use; see [`../THIRD_PARTY.md`](../THIRD_PARTY.md).
