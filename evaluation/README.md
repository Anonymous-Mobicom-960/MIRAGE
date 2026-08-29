# Evaluation

The harnesses behind the reported privacy, quality and system-cost results. This is the smallest
set that reproduces them, not the full experiment workspace.

```text
evaluation/
├── privacy/
│   ├── appearance/          appearance-channel re-ID (paper Classes 1 and 2) on the capture corpus
│   ├── gait/                pose-channel re-ID (paper Classes 3 and 4) against a real gait adversary (CASIA-B)
│   ├── silhouette/          the earlier CASIA-B silhouette-mitigation study and the frozen GaitBase adversary
│   └── capture_boundary/    independent audits of what the emitted mask actually covers
├── quality/                 visual utility, FID, leak audit, sidecar contract checks
├── performance/             phone energy / thermal / sustained throughput, and Raspberry Pi 5 bench
└── segbench/                shared segmentation + pose-anonymisation library used by the above
```

Setup (datasets, adversary checkpoints, run order) is in
[`../docs/reproduction.md`](../docs/reproduction.md). For a checked path through the paper's
numbers, see [`../reproduce/`](../reproduce/).

---

## `privacy/appearance/`

The paper's appearance-channel table: adversary **Class 1** (GEI + nearest neighbour, unlearned)
and **Class 2** (frozen GaitBase, learned) attack what each defence actually releases, on the
capture corpus (103 clips, 20 identities). Includes the common extractor, the vendored protocol
module, and the DeepPrivacy2 baseline runner. This is where the headline "at the measured chance
floor" result comes from. See [`privacy/appearance/README.md`](privacy/appearance/README.md).

---

## `privacy/gait/`

Measures whether the exported COCO-17 track still carries gait identity, against a **real gait
re-identification model**, not a metric of our own. This is the paper's motion-channel table:
adversary **Class 3** (hand-crafted descriptors + cosine nearest neighbour, unlearned) and
**Class 4** (a ResGCN retrained from scratch on protected output with supervised contrastive
loss, the adaptive case), on CASIA-B (5,375 test sequences, 50 identity-disjoint subjects).

| Module | Role |
|---|---|
| `casia_loader.py` | Streams the CASIA-B pose CSVs into per-sequence `(T, 17, 3)` tracklets with an identity-split filter |
| `adversary.py` | The frozen GaitGraph ResGCN as a 128-d embedding extractor |
| `protocol.py` | Canonical CASIA-B rank-1, plus the full suite (rank-1/5, mAP, CMC, EER, bootstrap CI) with an identity-disjointness assertion |
| `anon_adapter.py` | Routes tracklets through the anonymisation configurations; per-identity vs per-sequence seeding; matched-jitter controls; template construction |
| `reproduce_baseline.py` | **Gate.** The frozen adversary must reproduce the published CASIA-B numbers before anything else is believed |
| `run_tm12.py`, `run_redesign.py`, `sweep_pareto.py` | Frozen-adversary evaluation, the transform ablation, and the frozen privacy-utility ladder |
| `tm3_retrain.py`, `cmc_tm3.py` | The **Class 4** adversary: retrained on protected output |
| `untrained_reid.py`, `naive_pose_baselines.py` | The **Class 3** model-free attacker |
| `null_precision.py`, `tm12_full_cmc.py` | Measured nulls and full CMC curves |
| `score_class4.py` | Re-scores trained Class 4 checkpoints under the paper's protocol, with a measured null and a self-gate |

**Threat models.** TM1: clean gallery, protected probe. TM2-id / TM2-seq: both protected, seeded
per identity vs per sequence, which is what separates pseudonymity from protection. TM3: the
retrained adversary (the paper's Class 4). The paper's Class 3 is the unlearned attacker in
`untrained_reid.py`.

### Three rules this harness exists to enforce

1. **Always run the adaptive adversary.** A frozen adversary is a lower bound and nothing more.
   On this system, frozen and adaptive measurements of the *same* mechanism have disagreed by
   large factors, in both directions: a frozen number under-pricing a cost, and an untrained one
   over-pricing a benefit. A number from one adversary bounds the other in neither direction.
2. **Always measure the null.** Report lift over each arm's own measured null, never raw
   accuracy. An attacker with no positive control may be measuring nothing at all: a retrained
   silhouette attacker that scores at its own null on *undefended* data is not evidence of
   protection.
3. **Seed per sequence, never per identity.** Identity-derived seeding makes the perturbation
   itself a stable signature, and TM2-id will show it.

---

## `privacy/silhouette/`

The earlier CASIA-B study of the silhouette mitigation, plus `silhouette_harness.py`, which wraps
the frozen OpenGait GaitBase recogniser that `privacy/appearance/` imports as its Class 2
adversary. `silhouette_harness.py` imports the **deployed** mitigation and its **deployed**
parameters directly from `tier1/src/edge_runner_pi5/`, rather than reimplementing them, so the
arm measured is the arm that ships. `PROTOCOL.md` records the binding attack protocol and,
specifically, why the defence must never be applied inside a canvas normalised from the clean
person: doing so flatters exactly the smooth, outward-only family of defences it then ranks
highest, and inverted a ranking when it was corrected.

Rank by lift over each arm's own measured null. Note that amplitude curves in this channel are
**non-monotonic**: a small displacement amplitude has measured *worse* than no displacement at
all. Never assume a smaller perturbation is a safer perturbation.

---

## `privacy/capture_boundary/`

Checks what the emitted mask actually covers, independently of the pipeline's own logs.

| Script | What it answers |
|---|---|
| `audit_unmasked.py` | How many real people does the emitted mask leave uncovered? Sweeps the same detector Tier 1 uses over the **source** frames. |
| `audit_s2_autogrey.py` | Does a real face survive into the emitted video? Reads the fill level out of the video instead of assuming it. |
| `audit_s2_faces.py` | The same question with an independent detector, for cross-checking. |
| `face_coverage_eval.py` | Per-instance face-coverage fraction. |
| `sticks_check.py` | How far did anonymisation actually move each joint, normalised by torso length? |

Two things these scripts are built around:

* **The pipeline's own refusal log is a sample, not a measurement.** It records boxes it chose to
  log, not presence. An independent detector sweep found several times more uncovered frames than
  the log did. That is why these audits re-read the source rather than trusting the pipeline.
* **Do not hardcode the fill value.** The edge runner writes lossless FFV1 whose flat fill is
  exactly grey 128; the capture service writes lossy H.264 whose flat fill lands near 124. An
  audit asserting `128 ± 1` reports severe face reveals on frames that are completely covered.

Run a **positive control first**: confirm the detector finds real faces in the raw frames before
reporting that it found none in the protected ones.

---

## `quality/`

| Script | Measures |
|---|---|
| `eval_tier2_visual_utility.py` | Per-clip perceptual utility of the protected output against a reference |
| `eval_fid_pooled.py` | FID between pooled protected frames (`--gen`) and pooled real frames (`--real-glob`) |
| `leak_audit.py` | Whether any real-person pixel survived into a composited or final frame |
| `check_sidecars.py` | That a bundle's sidecars satisfy the contract before a run |
| `eval_tier1_models.py` | Tier-1 model ablation |

**Never compare across measurement configurations.** Frame rate, resolution, crop and the
reference used all move these numbers; a comparison between two differently configured runs is
not a measurement of anything.

---

## `performance/`

| Script | Measures |
|---|---|
| `phone_battery_report.py` | Device energy, tagged with the screen and power-save flags Android reports |
| `phone_thermal.py` | Thermal trajectory during a sustained run |
| `phone_sustained_report.py` | Throughput under sustained load |
| `phone_arm_report.py`, `phone_static_metrics.py` | Per-arm timings; which execution provider the session actually opened |
| `window_inpaint_cost.py` | Inpainting cost as a function of the masked region |
| `pi_bench/` | Raspberry Pi 5 Tier-1 throughput (`PI_RESULTS.md` records the reference platform) |

`adb` is located via `MIRAGE_ADB` (default: `adb` on `PATH`). Outputs go beside the script unless
`MIRAGE_OUT_JSON` / `MIRAGE_POWER_CSV` are set.

A screen-on figure and a screen-off figure are not comparable; every row is tagged accordingly.
`phone_static_metrics.py` deliberately reports **which execution provider opened** rather than a
sampled utilisation figure. Over a short phase, `top` sampled across adb is dominated by sampling
artefacts, whereas "did the work reach the NPU or silently fall back to CPU" is both answerable
and the question the metric exists for.

---

## `segbench/`

The shared library the harnesses above import: segmentation backends, the pose-anonymisation
reference implementations (`pose_anon.py`, `pose_anon_v2.py`), metrics and configuration
plumbing. `run_config.py` reuses the shipped per-person tracker from `tier1/src/mirage/`
rather than reimplementing it, so evaluation and deployment agree on how people are assigned to
slots.

---

## Dependencies

[`requirements.txt`](requirements.txt). Heavier than the pipeline itself, because the adaptive
adversary is retrained here; a CUDA GPU is strongly recommended. `data/` and `vendor/`
subdirectories are gitignored and are populated by the setup commands in
[`../docs/reproduction.md`](../docs/reproduction.md).
