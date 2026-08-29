# Appearance-based re-identification (adversary Classes 1 and 2)

This harness produces the paper's appearance-channel privacy table: how well an attacker can
re-identify people from what each defence actually releases, using silhouette shape and aggregate
motion. It is the study in which MIRAGE's released bounding-box mask drives the learned adversary
to its measured chance floor while DeepPrivacy2, which replaces appearance but keeps the person's
real outline, barely moves it.

## The two adversaries

| Class | Attacker | What it is |
|---|---|---|
| **Class 1** | Unlearned appearance adversary | Gait Energy Images (the temporal mean silhouette) matched by nearest neighbour. Training-free. |
| **Class 2** | Learned appearance adversary | The published, frozen OpenGait **GaitBase** CASIA-B checkpoint used as an embedding extractor. |

Both consume (T, 64, 64) binary silhouettes in OpenGait's own CASIA-B input domain, produced by
one common extractor from whatever each defence releases.

## The arms

Arm names are directory names under `arms/`; each is one defence's released output.

| Arm | What the attacker sees |
|---|---|
| `raw` | The real person's true silhouette, extracted from the unprotected video. The positive control: it proves the attackers work before any defence is scored. |
| `mirage` | What MIRAGE releases at the device boundary: the grey-filled axis-aligned bounding box emitted by the shipped silhouette mitigation (2-frame temporal window), produced by the real shipped code, never a re-implementation. |
| `dp2` | DeepPrivacy2's released video (full-body synthetic replacement), silhouetted by the same extractor as every other arm. |

The asymmetry between the arms is real and deliberate: each defence is attacked on what it
actually hands out. MIRAGE releases a mask and a skeleton, not a photorealistic video, so its
silhouette is the emitted mask. DeepPrivacy2 releases a photorealistic video, so its silhouette
is extracted from that video.

## Protocol (enforced by `reid_modes.py`, shared by import)

* Gallery is exactly one clip per identity, redrawn 12 times per probe.
* Condition-matched and collection-matched gallery draws, so framing and capture batch cannot
  single anyone out.
* The probe's own clip and its whole source recording are excluded; content-level duplicates are
  banned.
* Equal frame budget per clip (70 frames after resampling), so length cannot be matched on.
* **Every chance floor is measured**, by pushing permuted labels through the identical pipeline,
  never assumed. Results are reported as lift over that measured floor.

## Expected results (the paper's numbers)

103 scored clips, 20 identities. Rank-1, with the measured chance floor in brackets:

| Arm | Class 1 (GEI + NN) | Class 2 (frozen GaitBase) |
|---|---|---|
| Raw silhouette (positive control) | 43.67 % (floor 11.10) | 38.24 % (floor 11.12) |
| DeepPrivacy2 | 42.76 % | 37.10 % |
| MIRAGE bounding box | **17.42 %** | **10.86 % (at the floor)** |

MIRAGE removes 80.6 % of the Class 1 lift and all of the Class 2 lift (lift -0.27 pp, i.e. at
chance). DeepPrivacy2 removes 2.8 % and 4.2 % respectively. `reproduce/expected_results.json` at
the repository root carries these with tolerances.

## Running it

Requires: the evaluation dataset (distributed separately from this code repository; it shows
real people), a CUDA GPU for the extractor and GaitBase, `ffmpeg`, and the weights listed in
[`../../../models/README.md`](../../../models/README.md) (YOLO11s-seg for the
extractor, the GaitBase checkpoint under `../silhouette/data/`, RTMPose-t under the Tier-1 edge
runner's `models/`).

```bash
# 1. Build the shared 10 fps corpus and its manifest from the dataset.
python make_corpus_manifest.py --dataset-root <reid_dataset_flat>

# 2. Extract the raw control and the MIRAGE arm (the shipped defence code runs inside this step).
python extract_arm.py --arm raw --dataset-root <reid_dataset_flat> \
    --video-dir corpus_10fps --also-mirage

# 3. (Optional baseline) Run DeepPrivacy2 over the same corpus, then extract its arm.
(cd baselines_dp2 && ./run_dp2.sh)
python extract_arm.py --arm dp2 --dataset-root <reid_dataset_flat> \
    --video-dir baselines_dp2/out --no-pose

# 4. Score Classes 1 and 2.
python class12_silhouette.py --arms raw,mirage,dp2 --reps 40
```

Outputs land in `reports/CLASS12.json`. `corpus_10fps/`, `arms/`, `models/` and `reports/` are
gitignored; they hold either derived data of real people or large binaries.

## Provenance notes

* `reid_modes.py` is the protocol module vendored from the internal harness that produced the
  published numbers; `class12_silhouette.py` imports the protocol from it rather than
  re-implementing anything.
* `casia_domain.py` carries the OpenGait CASIA-B pretreatment and the pose domain registration,
  copied verbatim from the internal modules, with the reasoning in its docstring.
* The MIRAGE arm is produced by the real shipped defence modules in
  `tier1/src/edge_runner_pi5/` (the same bytes the vendored-provenance test pins).
* The DeepPrivacy2 baseline runs the authors' own implementation, unmodified, at commit
  `f4d8f09d1eb8f758c89cf1795ae41f20533942d3`, in their own container; `baselines_dp2/` holds the
  exact in-container runner and its settings, including why multi-modal truncation is mandatory.
* The subject box comes from the clean annotation and is identical across arms, so detector
  localisation differences can never masquerade as privacy differences; detector recall per arm
  is reported separately by the extractor.
