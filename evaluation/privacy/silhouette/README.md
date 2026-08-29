# Silhouette channel: the frozen adversary and the CASIA-B mitigation study

This directory serves two roles.

1. **It hosts the learned appearance adversary.** `silhouette_harness.py` wraps the published,
   frozen OpenGait **GaitBase** CASIA-B checkpoint as an embedding extractor. The paper's
   appearance-channel table (adversary Class 2) imports it from here; see
   [`../appearance/README.md`](../appearance/README.md) for that measurement.
2. **It records the earlier CASIA-B study** of the silhouette mitigation, which validated the
   adversary against its published baseline and measured how much a temporal-union mask defence
   costs a gait recogniser. Its numbers are a development-time study on CASIA-B, not the paper's
   capture-corpus results; they are kept because the methodological lessons in them still bind.

## The adversary and its gate

| | |
|---|---|
| **Model** | GaitBase, OpenGait's CASIA-B baseline; published weights only (`GaitBase_DA-60000.pt` from [`opengait/OpenGait`](https://huggingface.co/opengait/OpenGait)) |
| **Input domain** | (T, 64, 64) binary silhouettes in OpenGait's own CASIA-B pretreatment |
| **Validation gate** | The undefended arm must reproduce the published CASIA-B baseline before any defended number is believed. The gate is enforced in code: a raw arm more than 3 pp off marks the whole report `passed: false`. This run passed at 0.35 pp. |

`require_real_adversary()` blocks any report produced by an unvalidated embedder. An adversary
that has never demonstrated a positive control may be measuring nothing.

## Two constraints every silhouette defence must satisfy

* **The fail-closed superset rule.** The emitted mask must contain the detected mask, every
  frame, because a pixel of the real person outside the mask is a privacy reveal. All mechanisms
  are outward-only, and the study verified 0 violation pixels across 446,551 frames.
* **An area ceiling.** The mask is also the cloud's inpainting budget: growing it measurably
  degrades generation quality, so a defence cannot simply inflate the mask without paying for it
  in the released video.

The shipped configuration is the axis-aligned bounding box with a 2-frame temporal window,
applied by `mask_mitigate()` in the deployed edge code. The harness imports the deployed function
and its deployed parameters directly from `tier1/src/edge_runner_pi5/`, never a re-implementation,
so the arm measured is the arm that ships.

## What the CASIA-B study found

On CASIA-B (5,485 sequences, 50 identity-disjoint test subjects, measured chance 2.0 %), passing
the true silhouette through the temporal-union mitigation cut the frozen recogniser's rank-1 from
96.55 % to 62.27 %. Three lessons from that study still bind anything done in this channel:

1. **Set-theoretic reasoning about per-frame masks predicts nothing about a sequence attacker.**
   The pre-study analysis argued that a superset mask "cannot reduce silhouette information".
   Per frame that is true; the measured sequence-level effect was still a 34 pp drop, because the
   temporal union destroys the boundary dynamics that carry the identity signal.
2. **This is mitigation, not anonymisation.** 62.27 % is 31 times chance. The silhouette channel
   still identified the subject most of the time, which is why the released mask is a bounding
   box rather than a person-shaped contour at all.
3. **The protocol must not normalise the canvas from the clean person.** A harness that registers
   the defended mask inside a crop derived from the undefended person flatters exactly the
   smooth, outward-only family of defences, and correcting this inverted an entire ranking.
   Registration order matters and is recorded with every result.

The study also measured that the effect size tracks inter-frame motion (continuous walking gives
the temporal union more to cover than near-static footage), so its magnitude is corpus-dependent
and was never quoted as a deployment constant.

## Attack protocol

[`BOXNATIVE.md`](BOXNATIVE.md) reports **Class 5**, an adversary built for the shape the shipped
defence actually emits. Classes 1 and 2 consume a size-normalised silhouette, and the shipped
defence emits a rectangle, so after normalisation a defended clip is a white square -- 10 of 103
collapse to bit-identical GEIs. Read natively instead, the defence removes 32 % of the available
lift rather than 80.6 %, and the residual is carried by box height alone. Reproduce with
`extract_boxes.py` then `attack_boxnative.py`.

[`PROTOCOL.md`](PROTOCOL.md) records the five binding rules for every silhouette re-identification
measurement: what the attacker may train on, what they are tested on, that raw video appears
nowhere, and that a probe never matches against its own source clip. It also records the
single-seed variance lesson: a learned attacker's seed-to-seed range can exceed the gap between
defences, so single-seed numbers are never quoted.

## Reproducing

```bash
git clone https://github.com/ShiqiYu/OpenGait vendor/OpenGait      # model code
huggingface-cli download opengait/OpenGait \
  "CASIA-B/Baseline/GaitBase_DA/checkpoints/GaitBase_DA-60000.pt" --local-dir data
python run_silhouette_reid.py --silh-root "<CASIA-B silhouette root>" \
  --ckpt data/CASIA-B/Baseline/GaitBase_DA/checkpoints/GaitBase_DA-60000.pt
python run_silhouette_reid.py --from-cache          # re-analyse in seconds
```

The CASIA-B silhouette dataset is about 1.4 GB of individual PNGs, so PNG loading dominates the
run; subjects are streamed (load one subject, build the arms, embed, discard) and embeddings are
cached to `data/emb_cache.npz` so a re-analysis takes seconds.

## Files

| File | Purpose |
|---|---|
| `silhouette_harness.py` | Data loaders, the deployed-mitigation import, coverage stats, the GaitBase adversary wrapper, `require_real_adversary()` |
| `run_silhouette_reid.py` | The CASIA-B evaluation; `--selftest` exercises the plumbing on synthetic silhouettes and never writes a result |
| `cmc_opengait.py` | Full CMC curves for the frozen adversary |
| `measure_local_masks.py` | Shape/coverage A/B on real emitted masks |
| `build_arms_20260731.py`, `derive_ksame_template.py`, `prescreen_area.py`, `summarize_20260731.py` | Tooling from the development-time mechanism ablation; kept for completeness, not needed to reproduce any paper number |
| `data/` | Gitignored; datasets and checkpoints live here |
