# Workflows

Runnable definitions for the pipeline, grouped by the tier they drive.

```text
workflows/
├── end_to_end/RUNBOOK.md          the full three-tier sequence, step by step, device by device
├── tier1/run_tier1.sh             Tier 1 at its shipped configuration
└── tier2_cloud/                   the ComfyUI graphs and the script that derives them
```

## `end_to_end/RUNBOOK.md`

The three tiers run on three separate machines (wearable / phone / GPU server), so there is no single
command that executes the system. The runbook gives the order of operations, the exact command for
each step, which device it runs on, and what each step must have produced before the next one can
start. **Start here.**

## `tier1/run_tier1.sh`

```bash
./run_tier1.sh <input.mp4> <output_dir> [n_people] [ttp_url]
```

A thin wrapper around `tier1/scripts/run_tier1.py` that pins the shipped anti-re-identification
configuration: the shipped pose-anonymisation preset (`e2` in the code), bounding-box masking
(`bbox`), a 2-frame temporal window, and binarised
confidences. It changes no behaviour; it only fixes the flags so a run cannot silently drift off the
configuration the reported results were measured at. See
[`../tier1/README.md`](../tier1/README.md) for what every flag means.

## `tier2_cloud/`

| File | What it is |
|---|---|
| `V9_CLOUD_ONLY.json` | The base ComfyUI graph. Carries a complete second-character subgraph, muted; `ui2api.py` drops muted nodes during conversion so it never reaches the server. |
| `V9_GAIT_A_g9proj.json` | The graph the reported end-to-end renders actually used, at the shipped generation settings (operating point `c4_van`). |
| `phase_v9_a20_manualref.py` | Derives the manual-reference variant from the base graph: un-bypasses the fixed reference-image loader and mutes the auto-reference branch entirely, so no external API is called. |

These are **UI-format** ComfyUI workflows. Convert them to an API prompt with
`tier2_cloud/scripts/ui2api.py`, using an `/object_info` fetched from the running server, and queue
them with `tier2_cloud/scripts/queue_render.py`, which asserts the sampler, LoRA and window settings
against the recorded operating point and refuses to POST on any mismatch.

> **The API-key widget on the optional auto-reference node has been emptied** in both graphs. Supply
> your own key if you enable that branch; the reported runs used the manual-reference path, which
> makes no external call.

## Tiers without a workflow file

* **Tier 2 phone**: the phases are driven from the app's own UI cards; the device-automation harness
  used for the reported runs is bound to one app build's accessibility tree and is not part of this
  release. The companion-side steps are commands in the runbook and in
  [`../tier2_phone/README.md`](../tier2_phone/README.md).
* **Tier 3**: the only runnable component is `tier3_restoration/scripts/ttp_stub.py`, a single
  command. The consent protocol it stands in for is not implemented here.
