# Examples

Real artifacts from one end-to-end run, kept as a concrete statement of each interface contract.
They are small, identity-free JSON files, not media.

```text
examples/
├── inputs/                          (empty by design; see below)
└── outputs/
    ├── tier1/
    │   ├── manifest.json            per-slot stream id, apparent-gender flag, packet/key names,
    │   │                            skeleton-containment diagnostics, effective configuration
    │   ├── pose.json                the anonymised skeleton actually emitted, 100 frames x 1 person
    │   ├── face_scalars.json        the 12 DP-noised expression scalars, [frames][persons][12]
    │   ├── TIER1_CONFIG.json        the full effective Tier-1 configuration for that run
    │   ├── AUDIT_UNMASKED.json      independent detector sweep: real people the mask left uncovered
    │   ├── AUDIT_S2.json            face-reveal audit of the emitted video
    │   └── STICKS_ANONYMISATION.json  measured displacement between the raw and emitted skeletons
    └── tier2_cloud/
        ├── BUNDLE_MANIFEST.json     exactly what was assembled for the cloud, and its provenance
        └── REFERENCE_IMAGES_README.txt
```

## Why `inputs/` is empty

The capture corpus used for the reported results shows real people and is not redistributed. Any MP4
of one or more people walking will exercise the pipeline; the runbook assumes **30 fps**, which is
what the cloud graph and the phone app both expect.

## Why these particular files

Each one is the answer to a question someone will otherwise have to read source code to answer.

* **`pose.json`**: what "anonymised pose" actually means on the wire. 133 keypoint rows in the
  wholebody layout, with face rows `23..90` zeroed and confidences binarised to `{0, 1}`. In this
  run only 8 of the 133 rows are non-zero: the free-end prune drops unsupported termini rather than
  inventing joints. The `anon` block records every effective knob, so a consumer can tell what
  produced the file rather than trusting a default.
* **`face_scalars.json`**: the shape of the expression channel, and its scale. At the shipped
  privacy budget these values are, to the precision of the available measurements, indistinguishable
  from Laplace noise.
* **`manifest.json`**: the per-person recovery envelope index, and the only place the apparent-gender
  flag appears. It carries no identity-bearing content: a random stream UUID, counters, and
  configuration.
* **`BUNDLE_MANIFEST.json`**: the complete list of what crossed the cloud boundary, including the
  explicit record that the reference character sheet was *carried over* and needs human confirmation.
* **The three audit JSONs**: what an independent check of the capture boundary produces, so a new
  audit run has something to be compared against.

Absolute paths in these files were replaced with `<input>` during publication. Nothing else was
changed.

## Visual examples

Video output is published as GIFs under [`../assets/gifs/`](../assets/gifs/) rather than as source
media, to keep the repository small.
