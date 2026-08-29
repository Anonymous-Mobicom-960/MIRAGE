# End-to-End Runbook

The three tiers run on three different machines, so there is no single-process entry point. This is
the order of operations, the command for each step, and which device it runs on. It is the sequence
the reported end-to-end runs followed.

The example uses a single-person clip. For a two-person clip, everything below is per **slot**: two
cloud renders, each bound to one slot's mask, sticks and face mesh.

---

## 0. Prepare the clip

| where | what |
|---|---|
| workstation | Cut the segment at **30 fps**; the cloud graph and the phone app both assume it (30 fps in, 30 fps out). |

```bash
ffmpeg -i source.mp4 -ss <start> -t <dur> -vf "fps=30" -c:v libx264 -crf 16 -pix_fmt yuv420p clip.mp4
```

## 1. Start a TTP public-key endpoint

| where | what |
|---|---|
| workstation (or the real TTP) | Serves `GET /v1/public-key`. Tier 1 will not run without one. |

```bash
python tier3_restoration/scripts/ttp_stub.py ttp_private_TESTONLY.pem 8843 &
```

Generating a local test key is described in [`../../tier3_restoration/README.md`](../../tier3_restoration/README.md).

## 2. Tier 1: capture boundary

| where | what |
|---|---|
| Raspberry Pi 5 (or any CPU host) | Detect, track, mask, anonymise the pose, derive expression scalars, encrypt the recovery envelopes. |

```bash
bash workflows/tier1/run_tier1.sh clip.mp4 out_t1 1 http://127.0.0.1:8843
```

Produces `out_t1/`, the egress bundle described in [`../../tier1/README.md`](../../tier1/README.md).
**Everything downstream reads only this directory.** The source clip is not needed again except by
the audits in step 3, which deliberately re-read it to check the mask from outside the pipeline.

## 3. Audit the capture boundary (optional but recommended)

| where | what |
|---|---|
| workstation | Independent checks that do not trust the pipeline's own logs. |

```bash
python evaluation/privacy/capture_boundary/audit_unmasked.py  clip.mp4 out_t1/mask.mp4 AUDIT_UNMASKED.json
python evaluation/privacy/capture_boundary/audit_s2_autogrey.py               # face-reveal audit
python evaluation/privacy/capture_boundary/sticks_check.py                    # anonymisation magnitude
```

`audit_unmasked.py` sweeps the same detector Tier 1 uses over the **source** frames and reports how
many real people the emitted mask leaves uncovered, independently of the pipeline's own
`refused_boxes` log, which records a sample of what it chose to log, not a measurement of presence.
Reference outputs are in [`../../examples/outputs/tier1/`](../../examples/outputs/tier1/).

## 4. Tier 2 phone: background and illumination

| where | what |
|---|---|
| workstation | Transcode the Tier-1 output into the app's contract and verify the mask survived it. |
| **phone** | Phase 1 (background reconstruction) and Phase 1b (lightmap), from the app's Tier-2 card. |

```bash
python tier2_phone/companion_scripts/tier1_to_tier2.py --clip clip.mp4 --tag A
adb push <bundle_dir> /sdcard/Android/data/com.mirage.npu/files/
# run Phase 1 and Phase 1b in the app, then:
adb pull /sdcard/Android/data/com.mirage.npu/files/out/light_map.mp4 tier2_out/
```

`background_reconstructed.mp4` stays on the device. `light_map.mp4` is the only pixel product that is
allowed to continue.

## 5. Build and verify the cloud bundle

| where | what |
|---|---|
| workstation | Assemble exactly the five permitted signals, plus the supplied synthetic reference identity. |

```bash
python tier2_cloud/scripts/build_cloud_bundle.py \
    --tier1 out_t1 --tier2 tier2_out --out to_cloud --refs refs
python tier2_cloud/scripts/verify_bundle.py --bundle to_cloud
python tier2_cloud/scripts/audit_bundle.py  --bundle to_cloud --tier1 out_t1 --tier2 tier2_out
```

`--refs` copies a **supplied** synthetic character sheet. Nothing in the bundle derives it, and
nothing verifies that it is the right one; `MANIFEST.json` and `REFERENCE_IMAGES_README.txt` record
the carry-over so a human confirms it before a render is queued.

## 6. Tier 2 cloud: render

| where | what |
|---|---|
| GPU server | Upload the bundle into ComfyUI's `input/`, convert the graph, assert the operating point, queue. |

```bash
# upload to_cloud/ into <ComfyUI>/input/<bundle_name>/
curl -s http://<host>:8188/object_info > object_info.json

python tier2_cloud/scripts/queue_render.py --object-info object_info.json --url http://<host>:8188
python tier2_cloud/scripts/queue_render.py --object-info object_info.json --url http://<host>:8188 --queue
```

The first invocation is a dry run: it converts, verifies and prints `VERIFIED` or `FAILED` per arm
and refuses to queue if anything mismatches. Collect `synthetic_person_pK.mp4` and
`synthetic_alpha_pK.mp4` from the server's `output/`.

## 7. Author the alpha matte

| where | what |
|---|---|
| workstation | Tier-1's mask supplies the domain, the generated pixels supply the shape. |

```bash
python tier2_phone/companion_scripts/alpha_from_tier1.py --selftest
python tier2_phone/companion_scripts/alpha_from_tier1.py \
    --clip <clip_dir> --slot p1 --from-cloud <from_cloud_dir> --out <alpha_dir>
```

The script refuses rather than guesses: if nothing survives the intersection, if coverage leaves the
plausible band, or if the matte is simply the box, it fails loudly.

## 8. Tier 2 phone: composite

| where | what |
|---|---|
| **phone** | Phase 2 (composite over the reconstructed plate). |

Push `synthetic_person_pK.mp4` and the authored `synthetic_alpha_pK.mp4` to the device, run the
phase from the app's card, and pull `final_output.mp4`. Phase 2 **throws** rather than keying if a
layer has no explicit alpha.

## 9. Tier 3: restoration, only if a bystander consents

| where | what |
|---|---|
| TTP | Unwrap the AES key, match the embedding, obtain consent, release only that track's key. |
| **phone** | Decrypt that track's crops and composite them back over its synthetic avatar. |

The matching, consent and release steps are not implemented in this release; see
[`../../tier3_restoration/README.md`](../../tier3_restoration/README.md), which shows how to open one
envelope with the TTP-side primitives so the rest of the chain can be exercised.

---

## What each step must produce

| after | expect |
|---|---|
| 2 | `out_t1/{masked_video.mkv, mask.mp4, pose.json, face_scalars.json, manifest.json, crypto/}` |
| 4 | `tier2_out/{background_reconstructed.mp4 (on device), light_map.mp4}` |
| 5 | `to_cloud/{masked_video, mask, mask_pK, pose_sticks_pK, facemesh_pK, light_map, reference_pK, MANIFEST.json}` |
| 6 | `synthetic_person_pK.mp4`, `synthetic_alpha_pK.mp4` |
| 8 | `composite.mp4`, `final_output.mp4` |

## Ordering constraints

* Step 5 needs **both** step 2 and step 4: the bundle is Tier-1's mask and controls composited over
  the phone's lightmap. It cannot be built from Tier-1 output alone.
* Step 7 needs step 6's output **and** step 2's per-slot mask.
* Steps 3 and 9 are optional; every other step is required.
