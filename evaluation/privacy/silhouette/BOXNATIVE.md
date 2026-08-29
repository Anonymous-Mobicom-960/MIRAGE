# Class 5: a box-native adversary against the shipped `bbox` silhouette defence

**Measured 2026-08-28 · laptop (RTX 4060) · 103 scored clips, 20 identities · local-first**
Source: `evaluation/privacy/silhouette/` — `extract_boxes.py`, `attack_boxnative.py`,
`BOXNATIVE_RESULTS.json`.

---

## Why this measurement exists

The shipped silhouette defence is `MASK_SHAPE_MODE="bbox"`: each person's mask becomes a filled
rectangle. Every published silhouette number prices it with an adversary that consumes a
**(T,64,64) size-normalised** silhouette — GEI nearest-neighbour (Class 1) and the frozen GaitBase
(Class 2).

After normalisation, a rectangle is a **white square**. That is not a rhetorical point:
`class12_silhouette.py --tiebreak` documents that 10 of 103 MIRAGE clips collapse to *bit-identical*
GEIs, creating exact distance ties in ~32 % of draws. Both published adversaries therefore discard
the one property a rectangle still carries — **how big it is** — and a tall person emits a tall box.

So `bbox` had never been measured against an adversary built for it. This is that measurement.

## Protocol

Scoring is `reid_modes.evaluate` / `.metrics` / `.bootstrap_ci` **imported, not reimplemented**, so
the gallery construction is bit-identical to the published Class 1/2 numbers: condition-matched,
same-collection, one clip per identity, probe condition ≠ gallery condition, same-source excluded
(R4), `BAN` list honoured. Nulls are **measured** per arm by label permutation through the identical
pipeline, 40 reps.

The defence is applied **where it is deployed** (PROTOCOL.md R5): `mask_mitigate` imported from
`tier1/src/edge_runner_pi5`, run on the native-resolution mask with the shipped
config and a per-**clip** seed, mirroring `evaluation/privacy/appearance/extract_arm.py` line for line.
Boxes are then measured on the **emitted** mask in native pixels. Nothing is normalised anywhere —
normalising is the step that hid this channel.

> ✅ **Positive control on the protocol itself.** The measured null lands at **11.00–11.15 %** across
> four independent runs, reproducing the published chance floor (Class 1 `11.1`, Class 2 `11.12`)
> without being given it. The harness is scoring the same task.

## Results

Two feature sets, because they answer different questions. `full` includes absolute pixel size —
what a real attacker holding the released masks can actually compute. `scalefree` deletes the four
size features, so whatever it finds cannot be explained by the subject standing closer to the lens.

| Feature set | Arm | Rank-1 | 95 % CI | Measured null | **Lift** |
|---|---|---|---|---|---|
| full | raw (undefended) | 23.08 % | [18.06, 39.80] | 11.15 % | **+11.93 pp** |
| full | **mirage (SHIPPED)** | **19.23 %** | [14.51, 32.29] | 11.12 % | **+8.11 pp** |
| scalefree | raw (undefended) | 18.10 % | [13.87, 30.94] | 11.06 % | +7.04 pp |
| scalefree | **mirage (SHIPPED)** | **14.48 %** | [10.38, 20.71] | 11.00 % | +3.48 pp |

n = 5 304 scored draws per cell.

### 🔴 The defence removes 32.0 % of the box-native lift

Against the adversary it was priced by, MIRAGE removes **80.6 %** of the Class 1 rank-1 lift and
sits **at the measured chance floor** for Class 2 (lift −0.27 pp). Against an adversary built for
the shape it actually emits, it removes **32.0 %** (`full`) — and:

* **The defended arm leaks significantly above its own null.** 19.23 % with a 95 % CI of
  [14.51, 32.29] against a null of 11.12 %: the interval excludes chance.
* **The defended arm is not separable from no defence at all.** Its CI overlaps the undefended
  arm's throughout. On this channel the measurement cannot show that the defence helps.

### The channel is stature, and only stature

Per-feature, on the **defended** arm, ranked:

| Feature | Rank-1 alone | | Feature | Rank-1 alone |
|---|---|---|---|---|
| **`med_h` (median box height)** | **21.72 %** | | `drift` | 13.35 % |
| `p90_h` | 20.36 % | | `gait_hz` | 12.67 % |
| `med_w` | 17.42 % | | `aspect_iqr` | 11.99 % |
| `sqrt_area` | 16.52 % | | `cv_h` | 11.54 % |
| `cv_w` | 14.93 % | | `bob` | 11.31 % |
| `aspect` | 14.25 % | | `gait_pow` | 10.41 % |

**Median box height alone (21.72 %) beats the entire twelve-feature set (19.23 %)** on the defended
arm. The other eleven features are, on balance, noise that dilutes it.

And the confirmation: once the four size features are deleted, the residual is **no longer
significant** — `scalefree` mirage is 14.48 % with CI [10.38, 20.71], which **contains** its 11.00 %
null. Remove size and there is nothing left to measure.

So the mechanism is not gait, not outline dynamics, not cadence. It is **how tall the emitted
rectangle is**, which the defence preserves almost exactly: measured on the smoke set, emitted box
height is within **0.3–0.7 %** of the true silhouette height (`ratio 1.0030 / 1.0035 / 1.0066`).

## What this does and does not establish

**Does:** the shipped silhouette defence protects the *shape* channel and leaves the *size* channel
substantially intact, and no previously published number could have revealed that, because both
published adversaries normalise size away before they look.

**Does not:** this is not a claim that the system is broadly re-identifiable from masks. The
box-native attacker is *weak in absolute terms* — 23.08 % on **undefended** silhouettes, against the
82.58 % a Class 1 attacker reaches on the same undefended data. Box geometry is a far poorer signal
than a silhouette. The finding is about the defence's **coverage**, not about a large absolute risk.

**Honest bound.** Apparent size mixes stature with camera geometry, and the two cannot be separated
here. The gallery is cross-condition, so a match must generalise across the `close`/`medium`/`far`
conditions, which is real evidence — but conditions within a collection may still share camera
distance. Take `full` (+8.11 pp) as the upper bound of what is attributable to the person and
`scalefree` (+3.48 pp, not significant) as the lower. **The truth is bracketed, not pinned.**

**Precision.** The bootstrap CIs are wide because there are only 20 identities. Tightening this
needs more identities, not more reps.

## What would close it

Not another adversary — a defence change. The obvious candidate is quantising or jittering the
emitted box's height per sequence, the way the gait channel already jitters, so stature stops being
readable off the rectangle. That has an obvious cost: the mask must remain inclusion-biased
(`emitted ⊇ current` on every frame, the §2 guarantee), so height may only ever be *added*, never
trimmed. A per-sequence upward-only height pad is therefore feasible and would need to be priced
against both this adversary and the coverage audit.

**Not attempted here, and not recommended without measurement**: this report ends at the finding.
