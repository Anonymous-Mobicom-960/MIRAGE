# Architecture

This document describes the components, the trust domains they sit in, and what crosses each
boundary. It is a technical companion to the code, not a restatement of the paper.

## Trust domains

The system is split across three independent domains so that no single one holds enough to
reconstruct an identity.

| Domain | Holds | Does **not** hold |
|---|---|---|
| **Wearable (Tier 1)** | Raw frames, briefly; the AES key for each person, briefly | Any private key that can unwrap what it emitted |
| **Companion phone (Tier 2, trusted half)** | The protected stream, the reconstructed background, the encrypted envelopes | Any key material that opens an envelope |
| **Cloud (Tier 2, untrusted half)** | De-identified controls and a low-frequency illumination stream | Any identity-bearing pixel, any encrypted material |
| **TTP (Tier 3)** | The private key, and the pre-registered embedding database | Any media, ever |
| **Third-party VLM (optional, AutoRef)** | Grey silhouettes only, when automatic reference-sheet generation is enabled | Any photographic pixel, any control signal, any envelope |

![What crosses each trust boundary](../assets/figures/trust_boundaries.png)

The fourth row is the one that is easy to miss, so it is stated explicitly. `MIRAGE_AutoRefPrompt`
optionally asks a hosted vision model (Gemini or OpenAI) to describe the subject so a reference
sheet can be generated automatically, and that is a **network egress to a party outside all three
domains**. It is off unless an API key is configured, and what leaves is constrained by an egress
guard that judges **every frame individually** and admits only grey silhouettes.

🔴 The guard exists because both plausible failures were real and measured (2026-07-23): batch
statistics let a photographic frame ride along 1-in-8 undetected, and a second upload path had no
check at all while a comment claimed otherwise. `test_egress_guard.py` pins both, and fails if any
new node calls the upload helpers without the guard in the same function. Even so, the party on the
other end is outside the threat model: a silhouette is not identity-free (it is the channel Class 1
and 2 attack), and enabling AutoRef sends it to someone whose retention you do not control. Run it
only when that is acceptable, or supply the reference sheet by hand, which is the default.


Two invariants follow, and every design decision below serves one of them:

* **Architectural privacy.** Unencrypted bystander pixels never leave the wearable's execution
  space. Masking is fail-closed and does not depend on generation succeeding: if the cloud is
  unreachable, the bystander is still covered, just not replaced.
* **Cloud blindness.** The synthesis service is an appearance engine driven by structure. It
  receives no texture, no face geometry, and no high-frequency background.

## Tier 1: capture service

Implementation: `tier1/src/mirage/` (the capture service used for the reported runs) and
`tier1/src/edge_runner_pi5/` (the Raspberry-Pi-5 runner, `config.py`-driven, which owns the
differential-privacy stage for the expression channel).

Per frame:

1. **Detect and track.** YOLO11n localises people; a greedy nearest-centroid tracker with grace
   frames assigns stable per-person slots through occlusion and re-entry. Detection and pose run
   every *n* frames (default 5); intermediate frames propagate boxes, keypoints and face mesh with
   pyramidal Lucas-Kanade optical flow. Ingestion, processing and writing are separate threads with
   bounded queues, so inference back-pressure never drops an incoming frame.
2. **Redact, fail-closed.** Each person region is replaced with a uniform grey fill over an
   axis-aligned bounding box. The coarse rectangle is a privacy decision, not a simplification: the
   fine silhouette contour is itself a strong re-identification channel, and rectangularising it
   removes that channel at the source while still giving Tier 2 deterministic spatial bounds. The
   mitigation is **monotone by construction**: it returns the union of the shape operation, the
   temporal smear and the current frame's mask, so it can only ever add grey, never retract and
   reveal.
3. **Derive controls.** RTMPose-Tiny yields 17 COCO keypoints per person. MediaPipe FaceLandmarker
   yields a 468-point mesh, from which 12 identity-free scalars are derived (mouth openness and
   width, smile intensity, per-eye openness, per-brow raise, head yaw/pitch/roll, gaze x/y) and
   released smoothed, quantised and noised at a per-frame epsilon of 3.0.
4. **Neutralise identity in the controls.** The skeleton is fitted to a canonical anthropometric
   template and perturbed under a cryptographically fresh per-clip seed. This is the difference
   between releasing *a* skeleton and releasing *this person's* skeleton; see
   [Identity neutralisation](#identity-neutralisation-of-the-skeletal-channel).
5. **Preserve recovery material.** A JPEG-encoded person crop and a 512-d EdgeFace embedding are
   encrypted per person under AES-128-GCM; the AES key is wrapped under the TTP's RSA-4096 public
   key. Tier 1 fetches that public key over the network and **refuses to generate a keypair
   locally**, because generating one would put the private key next to the data it protects.
6. **Emit a coarse attribute.** A single categorical apparent-gender label per person stream, from
   an InsightFace MobileNet-0.25 classifier applied on-device to the best face crop. It steers the
   choice of a synthetic appearance prior; it is not a visual signal and cannot be inverted.

Everything else is dropped inside the device.

## The capture boundary

The complete egress is: the grey-filled video, the occupancy mask, the anonymised pose, the
DP-noised expression scalars, the apparent-gender flag, and the encrypted envelopes. See
[`pipeline.md`](pipeline.md) for the exact file formats.

Two deliberate scope reductions live at this boundary and shape how every coverage number must be
read:

* **Enrolment.** Tracks are enrolled during a short window from clip start. A track never enrolled
  contributes nothing downstream and is **not covered by the emitted mask**. Coverage results are
  therefore statements about enrolled, in-scope subjects.
* **Detection-limited logging.** The refusal log records the frames in which a refused person was
  *detected*, not the frames in which they were *present*. `refused_spans` carries a
  `detection_limited` flag so a consumer can see that "boxes < span" means "present, not detected"
  rather than "absent".

## Tier 2: companion phone (trusted)

Implementation: `tier2_phone/app/` (Kotlin, ONNX Runtime), `tier2_phone/companion_scripts/`.

The phone must restore scene context without letting the real background reach the cloud.

* **Background reconstruction** aligns neighbouring frames and takes a per-pixel temporal trimmed
  mean over *real* pixels only, recovering background that the subject's own motion progressively
  reveals. LaMa then fills the never-revealed core once. The alignment strategy is chosen per clip
  from a global-motion pre-pass: no alignment (static), pyramidal alignment (jitter), or an 8-DOF
  Gauss-Newton homography with a pan-sized mosaic canvas (dynamic).
* **Illumination abstraction** downsamples the reconstructed plate, applies a Gaussian, and
  rescales. What survives is the global colour distribution, ambient gradient and lighting
  direction; what does not is object structure. The Tier-1 grey fill is composited onto that plate,
  and *that* is what the cloud receives.
* **The local privacy firewall** is the bundle builder (`tier2_cloud/scripts/build_cloud_bundle.py`):
  it constructs the cloud payload from exactly the five permitted signals. It is the enforcement
  point, and `verify_bundle.py` / `audit_bundle.py` check its output before anything is uploaded.

## Tier 2: cloud (untrusted)

Implementation: `tier2_cloud/src/comfyui_custom_nodes/` plus the graphs in
`workflows/tier2_cloud/`.

A video-diffusion backend animates a synthetic reference identity under the skeletal and expression
controls, restricted to the repaintable hole the occupancy mask defines. A second pass re-detects a
person in the **generated** video and segments it to produce a per-character alpha matte.

Two structural properties matter:

* **One render is bound to one person slot.** If both mask loaders point at the union mask, the
  sampler is told to generate inside every box, obliges, and the matte (derived by detection) then
  has two candidates and binds to the wrong one. The bundle builder writes per-slot masks and the
  queue script binds them.
* **The reference identity is supplied, never derived.** Deriving it from the footage would carry
  the real subject's appearance across the trust boundary. The bundle builder therefore never
  generates one; it copies a supplied synthetic character sheet and records the carry-over for
  human confirmation. An optional node can infer coarse build attributes from grey silhouette
  frames alone, behind a per-frame egress guard that refuses to transmit any frame that is not
  statistically a flat silhouette.

## Model boundary

The generator is used strictly as an **appearance engine**. Identity comes from the reference
sheet; behaviour comes from the controls. Nothing in the graph can read the subject's appearance,
because nothing carrying it is loadable: the five video inputs are all Tier-1 or Tier-2 artifacts.

## The alpha-matte boundary

Compositing is where a leak would be easiest to reintroduce, so it is handled explicitly:

* **Domain** comes from Tier 1: the exact hole the sampler was allowed to repaint. Intersecting
  with a set the recipient already holds discloses nothing new, and at the shipped bounding-box
  mode that set is a rectangle, so the only boundary Tier 1 can contribute is a rectangle edge.
* **Shape** comes from the generated pixels: the cloud's own matte where it lands inside the
  domain, otherwise a painted-difference key against the lightmap the frame was conditioned on.
* Using Tier-1's *silhouette* as the matte is exactly what the anonymiser exists to prevent, since
  it would paste the real outline straight back into the output.

The compositor treats an explicit alpha as **required and refusable**. A layer without one raises
before a single frame is decoded, rather than silently falling back to a keyer.

## Tier 3: restoration boundary

The companion sends encrypted envelopes, never media. The TTP unwraps the symmetric keys with its
private key, merges fragmented tracks by cosine similarity between embeddings, matches against a
pre-registered template database under a conservative threshold, and asks the matched person for
consent, disclosing only a session identifier and non-sensitive metadata, never the wearer, the
content, or the embedding. On approval it releases only the AES keys bound to that track and
interval. Decryption and re-compositing happen on the phone.

The envelope side is implemented; the matching, consent and release steps are not part of this
release. See [`../tier3_restoration/README.md`](../tier3_restoration/README.md).

## Identity neutralisation of the skeletal channel

![The three body-centric re-identification channels](../assets/figures/reid_channels.png)

A skeleton with no appearance still carries identity, in three separable ways, and each is
addressed separately:

| Channel | What leaks | What the design does |
|---|---|---|
| **Static proportions** | Limb-length ratios and absolute body size are stable per person | Least-squares fit of a canonical template to the subject's vertical extent, plus a seeded global-scale jitter |
| **Dynamics** | Swing amplitude and inter-limb phase are a gait signature | Per-limb-chain amplitude rescale about that chain's mean pose, and a per-chain constant time shift |
| **Contour** | The silhouette outline is itself a biometric | The emitted mask is an axis-aligned rectangle, not an instance mask |

Two properties of the seeding are load-bearing:

* The seed is drawn **per sequence**, never per identity. An identity-derived seed turns the
  perturbation itself into a stable signature, which is the same failure that makes naive
  pseudonymisation useless against a linkage adversary.
* Cadence re-timing is **disabled** and the root trajectory stays locked to the subject's true
  path. Re-timing the walk both moved the figure off the mask it is supposed to occupy and made the
  motion measurably less natural, for no privacy gain that survived an adversary retrained on the
  output.

## Shared components

Deliberately few. The two that are shared are shared because a divergence would be a correctness
problem:

* `tier1/src/mirage/vendor/mirage_edge/` is a **byte-identical** copy of the defence modules from
  `tier1/src/edge_runner_pi5/`, with digests recorded in `VENDOR.md` and enforced by a test. Every
  privacy number quoted for those modules was measured against exactly those bytes.
* `tier2_phone/companion_scripts/alpha_from_tier1.py` imports the repaintable-hole geometry from
  `tier2_cloud/scripts/check_render.py`, so the phone's idea of that hole and the cloud's cannot
  drift apart.

## Where a claim can go wrong

Three failure modes recur often enough in this system that the code guards against them, and anyone
extending it should know about them:

1. **A number from one adversary bounds another in neither direction.** Mechanisms priced against a
   frozen adversary have repeatedly been mispriced relative to an adversary retrained on protected
   output, in both directions and by large factors. Only compare within one adversary and one
   protocol.
2. **An adversary with no positive control may be measuring nothing.** A retrained attacker that
   scores at its own measured null on *undefended* data is not evidence of protection.
3. **A defence is not a candidate until it has been rendered.** Configurations have won every
   pose-space metric and then produced visibly broken output. Pose-space metrics rank arms; they do
   not qualify them.
