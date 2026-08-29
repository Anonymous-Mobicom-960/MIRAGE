# Gait re-identification harness (adversary Classes 3 and 4)

Measures whether the COCO-17 pose track that Tier 1 exports still carries gait identity, against
a **real gait re-identification adversary**, not a metric of our own. This harness produces the
paper's motion-channel privacy table.

## The two adversaries

| Class | Attacker | What it is |
|---|---|---|
| **Class 3** | Unlearned gait adversary | Hand-crafted geometric pose descriptors matched by cosine nearest neighbour. Training-free. |
| **Class 4** | Learned adaptive adversary | A spatial-temporal ResGCN **retrained from scratch on anonymised output** with supervised contrastive loss (100 epochs). The Kerckhoffs case: the attacker knows and exploits the defence. |

The harness also keeps the frozen published GaitGraph ResGCN as a lower-bound reference
(`adversary.py`), gated by `reproduce_baseline.py`: the frozen adversary must reproduce the
published CASIA-B baseline before any defended number is believed.

## The arms

| Arm | What it is |
|---|---|
| Raw poses | The unmodified CASIA-B keypoint tracks. The positive control and the attacker's ceiling. |
| Quantized pose baseline | Canonical-template fitting and confidence binarisation with every dynamic perturbation disabled. Isolates what the pose representation alone removes, before any anonymisation runs. |
| MIRAGE pose anonymiser | The full shipped transform: canonical anthropometric template, seeded scale jitter, arm-chain angle perturbation, limb-swing rescale, per-chain phase offsets, free-end pruning, per-sequence seeding. Applied by the real vendored defence code. |

## Dataset and expected results

CASIA-B pose CSVs (from the GaitGraph release): 5,375 test sequences, 50 identity-disjoint
subjects. Chance is a single pooled permutation null per rank, verified independent of embedding
geometry. Rank-1, with lift over the measured 1.93 % floor:

| Arm | Class 3 | Class 4 (adaptive) |
|---|---|---|
| Raw poses (positive control) | 37.18 % (+35.25) | 90.25 ± 1.14 % (+88.32) |
| Quantized pose baseline | 4.43 % (+2.50) | 27.31 % (+25.38) |
| MIRAGE pose anonymiser | **3.78 % (+1.85)** | **26.20 ± 0.57 % (+24.27)** |

Read honestly: the anonymiser removes 94.8 % of the unlearned lift and 72.5 % of the adaptive
adversary's advantage. The Class 4 residual is strong mitigation, not anonymisation, and most of
the reduction on this channel comes from the pose representation itself (compare the quantized
baseline) rather than from the dynamic perturbation. Both facts are stated in the paper.
`reproduce/expected_results.json` at the repository root carries these numbers with tolerances.

## Environment and one-time setup

Python 3.10, torch with CUDA (the Class 4 adversary is retrained; the reported runs used an
RTX 4060), numpy/scipy/scikit-learn/matplotlib.

```bash
mkdir -p data vendor
curl -L -o data/data.zip https://github.com/tteepe/GaitGraph/releases/download/v0.1/data.zip
cd data && unzip -o data.zip casia-b_pose_test.csv casia-b_pose_train_valid.csv && cd ..
curl -L -o data/gaitgraph_resgcn-n39-r8_coco_seq_60.pth \
  https://github.com/tteepe/GaitGraph/releases/download/v0.1/gaitgraph_resgcn-n39-r8_coco_seq_60.pth
git clone --depth 1 https://github.com/tteepe/GaitGraph.git  vendor/GaitGraph
git clone --depth 1 https://github.com/tteepe/GaitGraph2.git vendor/GaitGraph2
```

`data/` and `vendor/` are gitignored.

## Modules

| File | Role |
|---|---|
| `casia_loader.py` | Streams the CASIA-B pose CSVs into per-sequence (T,17,3) tracklets with an identity-split filter |
| `adversary.py` | The frozen GaitGraph ResGCN as a 128-d embedding extractor |
| `protocol.py` | Canonical CASIA-B rank-1 plus the full suite (rank-1/5, mAP, CMC, EER, bootstrap CI), with an identity-disjointness assertion |
| `anon_adapter.py` | Routes tracklets through the anonymisation configurations; per-identity vs per-sequence seeding; matched-jitter controls; template construction |
| `untrained_reid.py`, `naive_pose_baselines.py` | The Class 3 attacker |
| `tm3_retrain.py`, `cmc_tm3.py` | The Class 4 attacker (retraining + canonical CMC) |
| `score_class4.py` | Re-scores trained Class 4 checkpoints with rank-1/5 and a measured permutation null (the paper's protocol), self-gated against each checkpoint's recorded training-time rank-1 |
| `null_precision.py`, `tm12_full_cmc.py`, `tm12_gait0807.py` | Measured nulls and full CMC curves |
| `plot_results.py` | Figures |

## Run order

```bash
python reproduce_baseline.py        # GATE: frozen adversary must reproduce published CASIA-B
python run_tm12.py                  # frozen-adversary threat models, incl. the seeding A/B
python run_redesign.py              # ablation of the shipped transform
python sweep_pareto.py              # frozen privacy/utility ladder
python tm3_retrain.py --config <c> --epochs 100    # Class 4: retrain on protected output
python plot_results.py
```

## Threat-model taxonomy (harness names)

The harness scripts predate the paper's Class numbering and use threat-model names:

* **TM1**: clean gallery, protected probe. A frozen lower bound and nothing more.
* **TM2-id / TM2-seq**: both gallery and probe protected, seeded per identity vs per sequence.
  This pair is what separates pseudonymity from protection: an identity-derived seed makes the
  perturbation itself a stable signature, and TM2-id shows it.
* **TM3**: the adversary retrained on protected output. This is the paper's **Class 4** and the
  decisive measurement.
* The paper's **Class 3** is the unlearned attacker in `untrained_reid.py`.

Three findings this harness established, which the code now enforces:

1. The frozen adversary and the retrained adversary have disagreed by large factors, in both
   directions, on the same mechanism. Never price a defence with only one of them.
2. Report lift over the measured null, never raw accuracy.
3. Class 4 retraining is stochastic: report a mean over seeds with its spread, because
   between-seed spread is frequently larger than the gap between candidate configurations.
