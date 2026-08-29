# Tier 2, cloud: the generation graph

`V9_CLOUD_ONLY.json` is the ComfyUI workflow the reported cloud renders used. Load it in the UI, or
convert it with `tier2_cloud/scripts/ui2api.py` and POST it to `/prompt`.

## The operating point

Every generation figure and video in this repository was rendered at `c4_van`, and the graph is set
to it:

| Knob | Value | Why this value |
|---|---|---|
| `steps` | 5 | 12 steps regressed the hands to 100 % defective |
| `cfg` | 1.0 | interacts with the LoRA stack; an earlier "keep 1.2" was confounded and withdrawn |
| `shift` / scheduler | 5.0 / `dpm++_sde` | |
| `frame_window_size` | 77 | |
| `lightx2v` cfg-step distill | 1.0 | |
| `Wan2.1_HighRes_Textures` | 0.3 | |
| `Wan14B_RealismBoost` | 0.35 | |
| `Wan21_Camera_Rotation` | **0.0** | load-bearing, do not raise |

Resolution and frame count are **not** widgets. They arrive over Set/Get links from the loaded
video, so the graph follows whatever Tier 1 emitted rather than needing to be edited per clip.

🔴 **`Camera_Rotation` at 0 is a fix, not a default.** The hands defect (40.54 % of frames) was
traced to the Tier-1 pose sticks, and the shipped remedy is the free-end prune plus this generation
config. Raising camera rotation reintroduces motion the pose sticks cannot support.

Do not re-test these: `pose_strength` (non-monotonic, not the lever), motion excess (refuted at
1.93x ground-truth motion), sampler steps, and the negative prompt (works, but strictly worse than
the control).

## 🔴 One known gap: the relight LoRA

The renders that produced the assets in this repository ran `WanAnimate_relight_lora_fp16` at
strength 1.0. It has since been removed from the pipeline, and `lora_0` in this graph is `none`.

**So this graph will not reproduce the shipped videos exactly.** Everything else matches; this one
slot does not. If you are trying to reproduce a published frame rather than run the current
pipeline, set `lora_0` to `WanAnimate_relight_lora_fp16.safetensors` at 1.0.

## Privacy: the conversion is audited, and the audit is the point

`ui2api.py` taint-tracks the graph during conversion and refuses to emit a prompt that routes
`face_images` to any save or preview node. That is not defensive decoration. On a real run, a node
labelled `vitpose` turned out to be fed by `SetNode[face_images]` -- raw face crops, one link away
from being written to disk. The audit caught it independently of the human check.

Never wire `face_images` to a preview or save node. The identity-free face signals are the
synthetic ones.
