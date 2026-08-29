# Vendored MIRAGE Tier-1 defence modules

These files are **BYTE-IDENTICAL** copies taken from this project's internal source
mirror. They implement the two measured re-ID defences this host
integrates:

* **GAIT** - gait preset `e2` (the shipped arm), applied as a whole-clip
  transform by `anonymize_pose_log()` in `pose_anon_edge.py`.
* the slot-grouping bridge `slot_groups()` in `person_slots.py`, which
  `pose_anon_edge.py` imports as a sibling.

* **SILHOUETTE** - mask shape mode `bbox` (`_shape_polys` / `mask_mitigate`,
  lifted from `mirage_tier1.py`) in `mask_shape.py`. Added by a separate change;
  its record is the **`mask_shape.py`** section at the end of this file, not the
  "Vendored files" block below, which covers only the two gait files.

---

## 🔴 THE RULE

**NEVER edit a file in this directory.**
Fix it upstream in `tier1/src/edge_runner_pi5/`, then re-vendor
(re-copy + re-hash + update this file). All host-specific logic belongs in
adapters *outside* this directory. The only non-vendored file here is
`__init__.py`, which is a pure import shim and touches no defence logic.

Rationale: every privacy number we can quote for these modules
(ledger §A.2x, §A.3a-e, §B.64 for `e2`; §A.6o for the silhouette arm) was
measured against *these exact bytes*. A local edit silently voids the
provenance chain from measurement → source → deployed artifact - the
`§A.2d` divergence failure mode.

---

## Vendored files

Copy date: **2026-08-14**
Vendored from repo: `<repository root>` (this project's internal source mirror)
Repo `HEAD` at copy time: **`c6d82dd59b26616ca9f1e144412978a298368a15`**
(both source files were **clean at HEAD** - `git status --porcelain` empty,
`git diff HEAD` empty - so `HEAD` describes the copied bytes exactly.)

### `pose_anon_edge.py`

| field | value |
|---|---|
| source path | `tier1/src/edge_runner_pi5/pose_anon_edge.py` |
| taken from commit | `c6d82dd59b26616ca9f1e144412978a298368a15` |
| last commit touching it | `439f467` (2026-08-14) - *"discard f3 and d2c fully: deleted from presets, CONFIGS and the queue registry"* |
| sha256 | `df88f9edfac2c17786c1746228a289772fcc945e667e94cf8d75f8f27a2411e9` |
| size / lines | 160 826 bytes / 2 440 lines |
| copy date | 2026-08-14; re-verified 2026-08-29 |
|   (history) | `85a9c598...` (160 571 B) was recorded at the 2026-08-14 copy. It went stale at the re-vendor that corrected the default-preset comment, when this table was not updated - the digest on disk was `c148a6c4...` while this row still said `85a9c598...`. Nothing depended on the row: `test_a1` compares the vendored file to its SOURCE live and never reads this file, which is why the drift ran silent. The current value is the 2026-08-29 rename. |

### `person_slots.py`

| field | value |
|---|---|
| source path | `tier1/src/edge_runner_pi5/person_slots.py` |
| taken from commit | `c6d82dd59b26616ca9f1e144412978a298368a15` |
| last commit touching it | `0ceda13` (2026-07-23) - *"§2: OUT_RES downscale leaked real person pixels; document person_count's real semantics"* |
| sha256 | `5f932cbd4d8eb9ff3256319e2636b34ad20c29f4a2f873d6c10dc0fc257cd0ec` |
| size / lines | 13 068 bytes / 230 lines |
| copy date | 2026-08-14; re-verified 2026-08-29 |
|   (history) | `d4f76882...` before the 2026-08-29 rename. The byte COUNT is unchanged because `SITARA_` and `MIRAGE_` are both seven characters - a digest change with no size change is expected here, not a red flag. |

### Re-verify the copies

```bash
cd <repository root>
sha256sum tier1/src/edge_runner_pi5/pose_anon_edge.py \
          tier1/src/edge_runner_pi5/person_slots.py
sha256sum tier1/src/mirage/vendor/mirage_edge/pose_anon_edge.py \
          tier1/src/mirage/vendor/mirage_edge/person_slots.py
```

The two blocks must match, digest for digest.

---

## Dependencies

`pose_anon_edge.py` imports only `secrets`, `os`, `sys`, `numpy`, and its
sibling `person_slots` (which is pure stdlib). **No torch, no cv2, no
`config.py`, no `mirage_tier1`.** Nothing else needs installing to use it.

`__init__.py` inserts this directory at the **front** of `sys.path` before
importing, because `pose_anon_edge.py:39` is a top-level absolute import of its
sibling (`from person_slots import slot_groups`) - on device both files sit
loose in one directory. The insert is idempotent. It also means
`pose_anon_edge` occupies its bare top-level name in `sys.modules`, the same
name MIRAGE's own `mirage_tier1.py` uses, so there is exactly one module object
in the process.

---

## Re-exported API (`from mirage.vendor.mirage_edge import ...`)

| name | kind | note |
|---|---|---|
| `anonymize_pose_log(pose_log, level, fps, frame_wh=None, slot_log=None)` | function | whole-clip gait transform; returns a **new** log, scores untouched. Do not change its return type - 8 in-repo call sites assign it directly. |
| `gait_preset(name=None)` | function | kwargs for `anonymize_v2`; defaults to `SHIPPED_PRESET`. `""` selects bare `LEVELS`. |
| `GAIT_PRESETS` | dict | the preset registry. |
| `LEVELS` | dict | the base dynamic-knob levels (`L4` etc.). |
| `SHIPPED_PRESET` | str | `"e2"` at this commit. |
| `new_clip_seed()` | function | fresh per-SEQUENCE `secrets` seed. **Never** seed per identity. |
| `test_fixed_seed()` | function | deterministic seed for parity tests only. |
| `binarize_pose_scores(pose_log, thresh)` | function | confidence → {0,1}; host threshold is 0.5 (MIRAGE `POSE_THRESH`), flag-overridable. |
| `pose_anon_edge` | module | the module object itself - read provenance off this. |
| `person_slots` | module | sibling module (`slot_groups`, `estimate_person_count`, `SlotTracker`). |
| `VENDOR_DIR` | str | absolute path of this directory. |

`gait_preset()` at this commit resolves `e2` to **9 keys**:

```
angle_groups=('uarm','farm')  cadence_amp=0.0  cadence_root_lock=True
limb_phase_amp=1.8  limb_phase_offset_s=0.35  limb_swing_amp=0.25
projection_fit=True  scale_from='extent'  seeded_global_scale_max=0.1
```

`scale_from` is **popped one level up** (`pose_anon_edge.py:2188`) - it is not
an `anonymize_v2` kwarg; it selects the collapse target. Same for
`height_mult` / `arm_mult` when a preset carries them.

---

## Provenance globals a host may read

`anonymize_pose_log()` returns **no** provenance block - upstream builds that
inside `mirage_tier1.main()` from its own locals plus these two module-level
lists. Read them off the module (`from mirage.vendor.mirage_edge import
pose_anon_edge`), never by copying the list out:

| exact name | type | written at | appended | meaning |
|---|---|---|---|---|
| `pose_anon_edge._TEMPLATE_KIND_USED` | `list[str]` | `pose_anon_edge.py:2301` (`.append(tmpl_kind)`); declared `:1146` | **unconditionally**, once per transformed slot | which collapse target each transformed slot actually used - distinguishes a real anatomical run from the silent legacy torso-scaled fallback taken when the shoulders were never confidently seen. Under preset `e2` (`scale_from='extent'`) the value is `anatomical_extent`. |
| `pose_anon_edge._STATURE_SCALE_USED` | `list[float]` | `pose_anon_edge.py:2335` (`.append(round(float(_eff_scale), 4))`); declared `:1147` | 🔴 **only when `stature_ratio` is truthy** (`:2323`) | the per-clip auto size scale actually applied. |

Both are **process-global, append-only, and never reset by the vendored code**
(there is no `global` statement anywhere in the module - the lists are mutated
in place). A host that wants *per-clip* provenance must snapshot
`len(...)` before the call and slice after it. Upstream's own reader is
`mirage_tier1.py:2372-2377`, which does `sorted(set(_TK))` plus per-kind counts.

🔴 **THE TWO LISTS ARE NOT INDEX-ALIGNED, and a host must not assume they are.**
`_STATURE_SCALE_USED.append` sits *inside* `if _stature_ratio:` (`:2323`), so it
is written only when a preset key or `MIRAGE_STATURE_RATIO` supplies one.
**The shipped `e2` preset carries no `stature_ratio`, so under the shipped
config `_STATURE_SCALE_USED` stays EMPTY while `_TEMPLATE_KIND_USED` grows by
one entry per slot.** Verified by running the vendored copy 2026-08-14: a
60-frame 1-person synthetic clip through `anonymize_pose_log(..., "L4", 25.0)`
gave `_TEMPLATE_KIND_USED` `0 → 1` (`['anatomical_extent']`) and
`_STATURE_SCALE_USED` `0 → 0`. Zip the two lists together and you will silently
mis-attribute a scale to the wrong slot; an empty `_STATURE_SCALE_USED` means
"no stature fit ran", **not** "scale 1.0 was applied".

No other module-level global in `pose_anon_edge.py` is provenance; the rest
(`EDGES`, `_GROUPS`, `_TEMPLATE_RATIOS`, `_ANATOMICAL_OVER_SHOULDER`,
`LEVELS`, `GAIT_PRESETS`, `SHIPPED_PRESET`, `RESTORE_TORSO_ROLL`,
`_ANGLE_MIRROR_ENV`, …) are constants or env-resolved config read once at
import time.

---

## Environment variables the vendored code reads at import time

Four are resolved **once, at import time**, so a host that wants a non-default
must set them *before* importing this package. Verified by reading the actual
scope of each `os.environ.get` site:

| var | site | effect |
|---|---|---|
| `MIRAGE_RESTORE_TORSO_ROLL` | `:176` (module scope) | sets `RESTORE_TORSO_ROLL` |
| `MIRAGE_ANGLE_MIRROR` | `:660` (module scope) | sets `_ANGLE_MIRROR_ENV` |
| `MIRAGE_LOWFREQ_AMP` | `:1358` (module-scope `try:`) | **mutates `LEVELS["L4"]["lowfreq_amp_frac"]` in place** |
| `MIRAGE_ANGLE_CONST` / `MIRAGE_ANGLE_DRIFT` | `:1386` (module-scope `for`) | **mutates `LEVELS["L4"]["angle_const_deg"]` / `["angle_drift_deg"]` in place** |

🔴 The last two rows matter: `LEVELS["L4"]` is the object this package re-exports
as `LEVELS`, so an env var set before import silently changes what a host reads
back as "the shipped level". That is exactly the §A.2d divergence (committed
source vs rendered artifact) - a host that records config must record the
*effective* dict, not the source default. Shipped defaults at this commit are
`angle_const_deg=14`, `angle_drift_deg=10`, `lowfreq_amp_frac=0.0`.

Resolved **per call** (safe to set later): `MIRAGE_GAIT_PRESET` (`:2149`),
`MIRAGE_POSE_SCALE_FROM` (`:1075`, `:1289`, `:2189`), `MIRAGE_ANGLE_GROUPS` /
`MIRAGE_TORSO_QUIET` (`:392-393`), `MIRAGE_CADENCE_ROOT_LOCK` (`:1011`),
`MIRAGE_TEST_FIXED_SEED` (`:1439`), `MIRAGE_HEIGHT_MULT` / `MIRAGE_ARM_MULT` /
`MIRAGE_LEG_MULT` / `MIRAGE_NECK_MULT` (`:2202-2224`),
`MIRAGE_STATURE_RATIO` (`:2250`), `MIRAGE_HEIGHT_ANCHOR` (`:2260`),
`MIRAGE_ANATOMICAL_TEMPLATE` (`:2275`), `MIRAGE_HEAD_ANCHOR` (`:2315`).

Leaving `MIRAGE_GAIT_PRESET` unset selects `SHIPPED_PRESET` (`e2`) - **not**
bare `LEVELS`. Only an explicit empty value selects bare `LEVELS`.

---

## Known integration hazards (established by a verified study - do not re-derive)

1. **Zero-row poisoning is a real crash.** `group_lengths()`
   (`pose_anon_edge.py:115`) takes an *unweighted median over all frames and
   never reads scores*. An absent export slot appends
   `np.zeros((17,3), float32)` (`pipeline.py:1172`), so a single-person clip
   with `export_people=3` leaves two 100 %-zero slots; every group length
   collapses to 0.0 and `len_factors = target/(0+1e-6)` emits coordinates
   around 1e10 px. **The adapter must compact each slot to its present frames
   only, transform that subsequence, and scatter back** - never pass dense zero
   rows in.
2. **Pass `slot_log`.** Export slots are re-let after repeated misses, so one
   slot block can splice two different people onto one collapse template and
   one seed. The stable id already exists at the append site as
   `det_idx_to_identity.get(di)` (`pipeline.py:1142`).
3. **Input format.** `anonymize_pose_log` expects the MIRAGE emit format - 
   `[frames][persons]` of `{"kp": [[x,y]] * 133, "score": [s] * 133}`, float64,
   **native pixel** coords. Body is `kp[:, :17, :]`; feet 17..22; face 23..90
   stays zeroed; hands 91..132.
4. **Never quote a MIRAGE-host privacy number as if it described this host.**

---

# `mask_shape.py` - the SILHOUETTE defence (change #2)

Added by a separate change from the two files above; this section is its record.
Everything in "🔴 THE RULE" applies to it unchanged.

## What it is, and how it differs from the other two

`mask_shape.py` is **not a whole-file copy** - the source, `mirage_tier1.py`, is a
2 567-line CLI runner that pulls in torch-free but still heavy device machinery.
Only its silhouette section is needed, so the file is a **locally-authored header
+ config shim followed by a verbatim byte slice** of that section. The slice is
appended, never retyped.

| field | value |
|---|---|
| source path | `tier1/src/edge_runner_pi5/mirage_tier1.py` |
| source lines | **789-1520 inclusive** (732 lines, 46 877 bytes) |
| taken from commit | `c6d82dd59b26616ca9f1e144412978a298368a15` |
| last commit touching it | `317c937` (2026-08-14) - *"SHIP e2: make it the default preset, and fix the provenance divergence that created"* |
| source file sha256 | `350950a267212b2b1265f79856af31e6342dab7efa09572d76e6184b1209c552` (170 488 bytes / 2 567 lines; was `b890a403...` before the 2026-08-29 rename) |
| **slice sha256** | `ad36b38658b9edfd30e8ac7a3f02b48a58bb9fbe8f815fb3689ed8eabfc8ffc0` |
| vendored file sha256 | `ba8aaee6f7b481a1748009e5fadaa15bacb97b40ad3935d7705fb6aabcbfcf5d` (59 523 bytes / 923 lines) |
|   (was) | `c59b90de...` -> `d9d8d028...` -> the current value, both steps on 2026-08-29: first the source-mirror rename, then the project rename. Both rewrote only the LOCALLY-AUTHORED header. |
|   🔴 **the line that matters** | the **slice sha256 above is UNCHANGED** through both. The lifted region carries no renamed identifier, so the bytes every published silhouette number was measured against are the same bytes. `test_a1` re-derives this from the banner on every run. |
| copy date | 2026-08-14 |

The source file was **clean at HEAD** when copied (`git status --porcelain` empty),
so the commit describes the copied bytes exactly.

Lifted symbols, with their line numbers in the source:

| symbol | source lines | note |
|---|---|---|
| `_radial_profile` | 790-819 | helper; used by `radiallp` + `ksame` |
| `_radial_poly` | 822-826 | helper; used by `radiallp` + `ksame` |
| `_shape_polys(cnts, mode, eps_frac)` | 829-1388 | one shape-canonicalisation op |
| `mask_mitigate(hist, cur, eps_frac)` | 1391-1520 | temporal union + shape op + the §2 re-OR |

### Re-verify the copy

The slice starts at the banner line
`# ------------------------- silhouette mitigation (mask shape channel) -------------------------`
and runs to EOF, so it can be checked without knowing the header's length:

```bash
cd <repository root>
python - <<'PY'
import hashlib
src = open('tier1/src/edge_runner_pi5/mirage_tier1.py','rb').read()
sl  = b'\r\n'.join(src.split(b'\r\n')[788:1520]) + b'\r\n'          # lines 789..1520
v   = open('tier1/src/mirage/vendor/mirage_edge/mask_shape.py','rb').read()
tail = v[v.find(b'# ------------------------- silhouette mitigation'):]
print(hashlib.sha256(sl).hexdigest())
print(hashlib.sha256(tail).hexdigest())
print('IDENTICAL' if sl == tail else 'DIVERGED')
PY
```

Both digests must read `ad36b386…` and the last line must print `IDENTICAL`.

## The config shim `C`

`_shape_polys` / `mask_mitigate` read MIRAGE's `config` module as
`getattr(C, "NAME", default)`. Importing `config.py` is not an option (it drags in
the whole device config and resolves ~40 `MIRAGE_*` env vars at import), so the
module defines its own **mutable** `C = _MaskShapeConfig()`.

* It carries **all 36** names the lifted region actually reads - the set was taken
  by grepping `getattr(C, "…"` over source lines 790-1520, not chosen by eye.
* Each value is the **shipped default from `config.py`** with no `MIRAGE_*` env
  var set, and each carries its `config.py` line number.
* 🔴 The inline `getattr` fallbacks are **not** always the shipped value - 
  e.g. `MASK_DISPLACE_AMP_FRAC` ships at **0.25** while the inline default is
  `0.10`. A name missing from `C` would silently select the wrong arm, which is
  why the set is exhaustive rather than "the ones bbox needs".
* It is a plain instance, so a host adapter assigns to it
  (`C.MASK_SHAPE_MODE = "bbox"`) without patching anything global. `mask_mitigate`
  itself writes `C._MASK_DISPLACE_SEED` while composing `"+"`-joined modes and
  restores it in a `finally`.

## The measured arm

```
MASK_SHAPE_MODE   = "bbox"      # axis-aligned per-component bounding rectangle
MASK_TEMPORAL_WIN = 2           # frames of running-max history the CALLER supplies as `hist`
MASK_SIMPLIFY_EPS = 0.01        # passed as mask_mitigate()'s `eps_frac`
```

🔴 **`MASK_TEMPORAL_WIN = 2` is the value ledger §A.6o measured `bbox` at - the
rectangle alone is not the measured arm.** §A.6o ran `bbox` *through*
`mask_mitigate` at a 2-frame window and scored **+7.16 pp** of re-ID lift over that
arm's own measured null (raw silhouette = +31.88 pp), with the frozen gait model
landing at chance. Quote it only for `bbox` **and** win = 2, and only for the
**MIRAGE host it was measured on** - it does not describe this host.

`mask_mitigate()` never reads `MASK_TEMPORAL_WIN`; the caller owns the history
deque. The knob lives in `C` so the adapter has one authoritative place to read it
from. It is **fps-coupled upstream** (`_frames(MASK_TEMPORAL_S=0.14)` at
`EMIT_FPS=15` → 2; at `EMIT_FPS=10` → **1**, the weakest window). Pin the 2; never
re-derive it from this host's frame rate.

## Dependencies

**`cv2` + `numpy` only** - no `config`, no `mirage_tier1`, no torch. Verified by
walking the AST of the lifted region: the only free module-level names are
`cv2`, `np` and `C`. That is also why the slice needed **zero** import
adjustment, and why the byte diff above is expected to be exactly empty.

## §2 - the superset guarantee

The guarantee is an **emergent property of the exact body**, not a wrapper around
it. `mask_mitigate` ends with

```python
return simp | sm | cur          # ⊇ sm ⊇ cur: monotone, never retracts
```

so the emitted mask is a superset of both the temporal union and the current
frame: mitigation can only ever ADD grey, never reveal. For `bbox` it holds twice
over, since `cv2.boundingRect(c)` already contains `c`. Do **not** touch the
return, the `| sm`, or the bbox merge loop - each was measured or audited as
written. (The `| sm` term specifically fixes a measured regression, ledger §A.1g:
`approxPolyDP` retracts at concave corners and was losing real person pixels the
temporal smear had covered.)

## Integration hazard specific to this module

🔴 **Keep a SEPARATE emit mask.** The host propagates `last_seg_mask` across
frames (it is warped forward on skip frames and reused). Writing the *mitigated*
mask back into `last_seg_mask` makes the temporal running-max compound frame over
frame and the grey region grows without bound. Feed the mitigated mask to the two
consumers - `applier.apply_mask(annotated, last_seg_mask)` (`pipeline.py:1406-1408`)
and `export_mask_writer.write` (`pipeline.py:1424-1427`) - and leave
`last_seg_mask` itself untouched.
