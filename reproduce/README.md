# Reproduce

One pinned configuration, the paper's numbers with tolerances, and a runner that checks a
reproduction against them. This directory exists so that "we reproduced the privacy results" is a
command with a PASS/CHECK verdict, not a judgement call.

| File | What it is |
|---|---|
| [`config.yaml`](config.yaml) | The complete shipped operating point (every knob of the capture boundary, the anti-re-identification defences, the cloud generation settings, the phone pipeline), plus one block per reproduction target: dataset, adversary, commands, runtime. |
| [`expected_results.json`](expected_results.json) | The paper's numbers, machine-readable, each with a tolerance that reflects that pipeline's own run-to-run variability. Chance floors included, because a rank accuracy without its measured floor is uninformative. |
| [`reproduce.py`](reproduce.py) | The runner: executes a target's harness at the pinned configuration, parses the produced JSON, compares against the expected values, and appends a verdict table to `REPRODUCTION_REPORT.md`. |

## Quick start

```bash
pip install -r evaluation/requirements.txt      # torch etc.; a CUDA GPU for the adversaries
python reproduce/reproduce.py --list

# 1. No data needed: static tests + vendored-byte provenance.
python reproduce/reproduce.py --run checkout

# 2. After the CASIA-B setup in docs/reproduction.md (~15-30 min):
python reproduce/reproduce.py --run gait-class3

# 3. The decisive (and expensive) one: retrains the adaptive adversary,
#    3 arms x 3 seeds x 100 epochs (~30-40 min each on an RTX 4060):
python reproduce/reproduce.py --run gait-class4

# 4. With the separately distributed evaluation dataset:
python reproduce/reproduce.py --run appearance-class12 --dataset-root <reid_dataset_flat>
```

Each target compares only the shipped arms and the paper's controls: the raw positive control,
the quantized pose baseline, the MIRAGE anonymiser, and (where applicable) the DeepPrivacy2
baseline. `--check <target>` re-compares an existing output without re-running anything;
`--run gait-class4 --skip-train` scores existing checkpoints.

## What a verdict means

* **PASS**: every compared value landed within tolerance of the paper's number.
* **CHECK**: at least one value missed its tolerance or was missing. That is a signal to compare
  protocol, data setup and configuration, not something to average away. The adaptive-adversary
  target also self-gates: each scored checkpoint must reproduce the canonical rank-1 recorded at
  its own training time, so a silent drift in the anonymisation or the eval path is caught before
  it can masquerade as a result.
* **BLOCKED**: a prerequisite (dataset, checkpoint, vendor checkout) is missing; the runner names
  it exactly.

## What is deliberately not automated

* **Visual utility** needs both generation arms re-rendered on the same clips with the same seeds
  and matched settings; diffusion output is also not bit-reproducible across GPU models and
  attention backends. The runner prints the commands and the reference values instead of
  pretending.
* **System cost** needs the physical hardware (Raspberry Pi 5 with a shunt-resistor power rig,
  Galaxy S25 Ultra). Reference values are in `expected_results.json`.
* **Detection AP / AR** cannot be reproduced from this repository at all; its ground truth is not
  in this release (see [`../docs/reproduction.md`](../docs/reproduction.md), Gaps).

## Honesty rules inherited from the evaluation harness

1. Always run the adaptive adversary; a frozen adversary is a lower bound and nothing more.
2. Always measure the null; report lift over it, never raw accuracy alone.
3. Never compare numbers across two different measurement configurations.
4. Adversary retraining is stochastic: compare means over the same number of seeds, never a
   single seed against a mean.
