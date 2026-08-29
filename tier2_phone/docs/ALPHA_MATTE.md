# The character alpha matte: why it is required, and how it is authored

**Summary:** shipping an explicit `synthetic_alpha_pK.mp4` alongside each character video removes
the single largest quality defect in Tier 2 (see-through characters), makes the compositing phase
2.6x faster, and drives the pixel-leak audit to 0.000 %. The matte is required, not preferred,
and it is authored from Tier-1's own mask geometry rather than trusted blindly from the cloud.

Measurements in this note are from a real clip (1264 x 1264, 10 fps) on a Galaxy S25 Ultra.

---

## 1. The explicit alpha is REQUIRED

`NCompositor.run(..., requireExplicitAlpha = true)` is the default. A character layer without its
own `synthetic_alpha_pK.mp4` throws `NCompositor.MissingAlphaException` **before a single frame
is decoded**, naming the missing file. The colour keyers are not a silent safety net; they
survive only as an explicit opt-out on the Phase 2 card (*allow keyer fallback*), and taking that
opt-out stamps the run:

```
[composite] 140 frames @ 1264x1264 · 2 character(s) · MATTE OK 2/2 EXPLICIT
[composite] 140 frames @ 1264x1264 · 2 character(s) · MATTE !! 1/2 EXPLICIT, 1 KEYED (DOWNGRADE)
```

Evaluation rows carry `explicit_alpha = N/N` for the same reason. Anything other than `N/N` is a
downgrade, and its numbers must not be pooled with an explicit-alpha run.

Why it matters, measured. The character arrives composited over the lightmap, so a colour keyer
can recover the background and key on difference; that works until the character's clothing
happens to match the block behind it, which no colour-based method can resolve by definition:

| | derived keyer | + geometric solidify | **explicit alpha** |
|---|---|---|---|
| see-through (notch % of matte) | 9.57 % | 2.38 % | **0.37 %** |
| worst frame | 26.90 % | 3.49 % | **~0.4 %** |
| Phase-2 composite | 531 ms/frame | 864 ms/frame | **328 ms/frame** |
| pixel-leak audit | 0.158 % | 0.001 % | **0.000 %** |

## 2. The alpha is AUTHORED from Tier 1: Tier 1 owns the DOMAIN, the generator owns the SHAPE

The cloud graph does export its own matte (a fresh SAM2 detection over the generated video), but
that matte has nothing tying it to the driving slot, and it was measured wrong on three of four
test renders: on one it covered 88.66 % of the frame because no person was generated; on two it
followed the character the skeleton did *not* drive. So the shipped path authors the matte
off-device (`companion_scripts/alpha_from_tier1.py`):

```
DOMAIN_K = grow_blockify(Tier-1 mask_pK)     # GrowMaskWithBlur(10,4) -> BlockifyMask(16)
                                             # = exactly the hole the sampler could repaint
SHAPE    = the cloud's SAM2 matte where it lands in DOMAIN_K, else the painted-difference key
           |generated - light_map| > 24      # decided ONCE per clip and printed, never per frame
alpha_K  = union(components of SHAPE that live in DOMAIN_K) ∩ DOMAIN_K
REFUSE     if nothing survives, if coverage leaves 0.5-40 %, or if the matte is simply the box
```

Verified across four rendered arms (n = 140 frames each, `alpha_from_tier1.py --selftest`
exercises the machinery): on a render whose cloud matte was already correct the gate is a no-op
(authored coverage 7.76 % vs the cloud's 7.77 %); on the renders whose cloud matte was wrong, the
authored matte tracked the correct character on every frame where the cloud's own tracked it on
none. The painted-difference key's positive control against SAM2, on frames where SAM2 is right,
is IoU 0.91-0.98.

**What Tier 1 contributes, and why that is privacy-safe.** The domain is an *intersection* with
`mask_pK`, which the cloud already holds (it is the uploaded mask, and the plate is built from
it), so it can disclose nothing new; and at the shipped bounding-box mask mode that set is an
axis-aligned rectangle, so the only boundary Tier 1 can contribute is a rectangle edge. Measured:
0.59-0.96 % of the emitted matte's boundary lies on it. Never replace the SHAPE term with
Tier-1's own person silhouette: pasting the real outline into the output is exactly the leak the
silhouette defence exists to prevent.

The painted-difference key works because the generated frame *is* the lightmap outside the
character (the background the generation was conditioned on is known and shipped): max-channel
`|generated - light_map|` has p99 of 11-13 over non-mask pixels versus an in-mask mean near 90,
so a threshold of 24 recovers exactly what the model painted. That is the closest thing to "the
generator's own matte" that the shipped graph makes available.

## 3. The contract the phone app expects

Placed in the app's input directory:

| file | meaning | notes |
|---|---|---|
| `masked_video.mp4` | real scene, person as a solid fill | required |
| `mask.mp4` | white = person hole | optional; without it `HoleMask` derives it |
| `synthetic_person_pK.mp4` | character K | required, K = 1..N |
| `synthetic_alpha_pK.mp4` | white = character K | **required**; without it Phase 2 refuses (see 1) |

Format: same width, height and frame count as its partner, any fps the partner uses. The app
binarises at > 127 and then applies `Compositor.haloAlpha` (binarise, 1 px erode, 3x3
anti-alias), so mild codec loss is harmless; there is no need for lossless. Greyscale written as
an ordinary H.264 video is fine (~2 MB for 153 frames at 1264 x 1264).

## 4. Laptop tool: `companion_scripts/make_sidecars.py`

```bash
python companion_scripts/make_sidecars.py \
    --silhouette silhouette.mp4 --character character.mp4 --out ./sidecars
# -> sidecars/mask.mp4, sidecars/synthetic_alpha_p1.mp4  (+ the adb push lines to run)
```

Produces silhouette/alpha sidecars for testing the compositor in isolation, using YOLO
person segmentation on the character video. Needs `ultralytics`, `torch`, `opencv-python`. It is
general, not fitted to one clip: the silhouette fill colour is auto-detected per clip by voting
over pixels that are both neutral and flat; nothing assumes a grid, resolution, length or fps;
and every stage prints what it detected, so an unusual clip is visible rather than silent.

A useful regression signal with no ground truth: the segmentation-derived alpha (from the
character video) and the mask (from the silhouette video) come from completely independent
sources, yet agree at IoU 0.83-0.95 per frame.

## 5. What the matte does NOT solve

An over-exposed character face is a *generation* defect, not a matte defect: in the measured
clips 31-69 % of the blown pixels are hard-clipped at the source, so the detail does not exist
and no post-process can invent it. `HighlightRecovery.kt` repairs the look (brightness and colour
reconstructed from a plane fit to the region's surroundings), but the real cure is exposing the
generation correctly.
