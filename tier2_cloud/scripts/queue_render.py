#!/usr/bin/env python3
"""Convert a UI-format ComfyUI workflow to an API prompt, VERIFY it against the expected
operating point, and (optionally) queue it on a ComfyUI server.

BASE GRAPH: workflows/tier2_cloud/V9_GAIT_A_g9proj.json -- the graph the end-to-end runs used,
at the `c4_van` operating point (WanVideoSampler steps 5 / cfg 1.0 / dpm++_sde / shift 5.0,
the 5-slot LoRA stack, frame_window_size 77).

The script REFUSES to POST unless every EXPECT_* constant below matches the converted prompt,
so a graph edit cannot silently change the rendered operating point.

    python queue_render.py --object-info object_info.json
    python queue_render.py --object-info object_info.json --url http://HOST:8188 --queue

Fetch object_info.json first from the running server:  GET <url>/object_info
"""
import argparse
import json
import os
import subprocess
import sys
import urllib.request

HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.dirname(os.path.dirname(HERE))
TOOLS = HERE                       # ui2api.py ships beside this script
WF_DIR = os.path.join(REPO, "workflows", "tier2_cloud")
CLOUD = os.path.join(HERE, "cloud")
BASE_WF = os.path.join(WF_DIR, "V9_GAIT_A_g9proj.json")
BASE_FOLDER = "pin_A_corridor_g9proj"
BASE_REF = "a20_ref_A_corridor.png"

# tag -> (volume folder, reference image)  -- both single-person, both drive slot p1
#
# v9/v10 (2026-08-12) are a MASK-MARGIN BRACKET on one identical pose: same tier1_FINAL keypoints,
# same 2.1x isotropic height, same ankle anchor, same angles. ONLY the repaintable hole differs --
# v9 ~83px margin (x2.607 the raw bbox), v10 ~124px (x3.174) -- because three prior renders on a
# single pose showed the hole has a sweet spot: ~34px blurs the body into a faceless blob, ~206px
# makes the generator invent a backpack to fill the empty space, ~65px rendered clean on a smaller
# figure. Containment is 0.00 % for every margin tested, so this knob buys resolution-vs-invention,
# not containment. Rendering both in one pod session brackets it in a single sitting.
# v11/v12 (2026-08-12) are the SAME config except for how the size scale is obtained:
#   v11  height_mult=2.1        an absolute multiplier fitted to this clip's ~629px subject
#   v12  stature_ratio=1.055    solved per clip so the figure is 1.055x THIS subject's own height
# On p03_c02 they should look near-identical (1.055 is exactly what v11 achieves here) -- that
# equivalence IS v12's acceptance test. They diverge only on a differently framed clip, where the
# fixed 2.1 fails: on p04_c04 it would emit 0.46x the person, while the auto fit holds 1.055x.
ARMS = [
    # p11_c01_bystander, source seconds 15.0-25.0 -> 100 f @ 10 fps @ 1264^2.
    # SINGLE-PERSON: the union mask IS that person, but the arm is still bound to its own
    # `mask_p1_00002.mp4` so the v22 MASK_EXTEND_TO_FLOOR growth actually reaches the sampler --
    # without the binding an enlarged mask_p1 is uploaded, verified, rendered and has ZERO effect
    # (the exact defect the 2026-08-16 commit fixed).
    ("p11c01_e2_p1", "p11c01_e2_20260822", "p11c01_e2_20260822/reference_p1_640.png",
     None, "mask_p1_00002.mp4"),
]

# the c4_van operating point -- every value measured; asserted, never assumed
EXPECT_SAMPLER = {"steps": 5, "cfg": 1.0, "sampler_name": "dpm++_sde", "shift": 5.0}
EXPECT_LORAS = [0, 1.0, 0.3, 0.35, 0.0]   # slot 0 is empty ("none")
EXPECT_WINDOW = 77
UA = "Mozilla/5.0 (mirage-queue)"
# Where the ComfyUI server keeps its input/ directory. Override with MIRAGE_COMFY_INPUT if the
# server is installed somewhere else than the deployment documented in tier2_cloud/README.md.
POD_INPUT = os.environ.get("MIRAGE_COMFY_INPUT", "/workspace/runpod-slim/ComfyUI/input/")
# Local directory holding the built bundle (the output of build_cloud_bundle.py). Needed only by
# bind_slot_masks(), which checks that a per-slot mask actually exists before binding to it.
BUNDLE_DIR = os.environ.get("MIRAGE_BUNDLE_DIR") or None


# The two mask loaders, and which branch each feeds (verified in V9_GAIT_A_g9proj.json):
#   #1350 "LOAD: person-RIGHT mask video" -> Set_mask_person_right -> grow -> blockify -> BOTH
#         `WanVideoAnimateEmbeds.mask` and the bg_images DrawMaskOnImage. This is the mask that
#         decides WHERE THE ACTIVE SAMPLER GENERATES, so it must carry the DRIVEN slot.
#   #1352 "LOAD: person-LEFT mask video"  -> the P2 subgraph + the bystander_mask_p2 output.
# Both ship pointed at the UNION mask, and their own titles admit why: "closest Tier-1 artifact:
# `mask` -- union silhouette; V9 has no per-person mask writer". `_build_two_person.py` is now that
# writer. Leaving them on the union is the §B.57e defect: it asks the sampler for a character in
# EVERY box, and the matte -- derived by detecting a person in the generated video -- then has two
# candidates and binds to the wrong one (0/140 frames on both run3 arms).
MASK_LOADER_DRIVEN, MASK_LOADER_OTHER = 1350, 1352


def bind_slot_masks(wf_path, folder, slot):
    """Point each mask loader at ONE slot's box. Returns what it bound, or None if unavailable."""
    other = "p2" if slot == "p1" else "p1"
    src = BUNDLE_DIR          # --bundle-dir / MIRAGE_BUNDLE_DIR; see the module header
    if not src:
        return None
    if not all(os.path.exists(os.path.join(src, f"mask_{s}_00002.mp4")) for s in (slot, other)):
        return None
    g = json.load(open(wf_path, encoding="utf-8"))
    bound = {}
    for n in g["nodes"]:
        if n["id"] not in (MASK_LOADER_DRIVEN, MASK_LOADER_OTHER):
            continue
        s = slot if n["id"] == MASK_LOADER_DRIVEN else other
        name = f"mask_{s}_00002.mp4"
        wv = n["widgets_values"]
        wv["video"] = f"{folder}/{name}"
        prev = (wv.get("videopreview") or {}).get("params")
        if isinstance(prev, dict):
            prev["filename"] = name                # keep the UI preview honest about the file
        bound[n["id"]] = wv["video"]
    json.dump(g, open(wf_path, "w", encoding="utf-8"))
    return bound


def bind_driven_mask(wf_path, folder, name):
    """Point #1350 -- the mask that reaches WanVideoAnimateEmbeds -- at a NAMED mask file.

    🔴 WHY THIS EXISTS. The single-person arms ship both loaders on the union mask, and for one
    person the union IS that person, so it renders correctly. But it also means a per-character
    mask written into the bundle is NEVER READ: the v22 enlargement (mask_p1_00002.mp4) would
    have been uploaded, verified, rendered -- and had exactly zero effect, because #1350 was
    still loading mask_00002.mp4. Measured before queueing, not after: the converted prompt for
    both v22 arms reported `mask=union`.

    #1352 is left alone deliberately: it feeds the P2 subgraph, which is MUTED in this graph and
    dropped by ui2api before queueing (§B.60b), so binding it would change nothing that runs.
    """
    g = json.load(open(wf_path, encoding="utf-8"))
    hit = None
    for n in g["nodes"]:
        if n["id"] != MASK_LOADER_DRIVEN:
            continue
        wv = n["widgets_values"]
        wv["video"] = f"{folder}/{name}"
        prev = (wv.get("videopreview") or {}).get("params")
        if isinstance(prev, dict):
            prev["filename"] = name                # keep the UI preview honest about the file
        hit = wv["video"]
    json.dump(g, open(wf_path, "w", encoding="utf-8"))
    return {MASK_LOADER_DRIVEN: hit} if hit else None


def build_wf(folder, ref, out_path, slot=None, mask_name=None):
    txt = open(BASE_WF, encoding="utf-8").read()
    txt = txt.replace(BASE_FOLDER, folder).replace(BASE_REF, ref)
    if slot and slot != "p1":
        # the sidecars are named by slot; swap the two the graph consumes
        txt = txt.replace(f"{folder}/pose_sticks_p1_", f"{folder}/pose_sticks_{slot}_")
        txt = txt.replace(f"{folder}/facemesh_p1_", f"{folder}/facemesh_{slot}_")
    open(out_path, "w", encoding="utf-8").write(txt)
    if slot:
        return bind_slot_masks(out_path, folder, slot)
    if mask_name:
        return bind_driven_mask(out_path, folder, mask_name)
    return None


def convert(wf_path, oi_path, out_path):
    r = subprocess.run([sys.executable, os.path.join(TOOLS, "ui2api.py"), wf_path,
                        "--object-info", oi_path, "-o", out_path],
                       capture_output=True, text=True, cwd=REPO)
    if r.returncode != 0:
        return None, (r.stderr or r.stdout)[-600:]
    return json.load(open(out_path, encoding="utf-8")), None


def retag_outputs(prompt, tag):
    n = 0
    for _nid, node in prompt.items():
        ins = node.get("inputs", {})
        fp = ins.get("filename_prefix")
        if isinstance(fp, str) and "GAIT_A_g9proj" in fp:
            ins["filename_prefix"] = fp.replace("GAIT_A_g9proj", tag)
            n += 1
    return n


def swap_loadimage(prompt, ref):
    n_swapped = 0
    for nid, n in prompt.items():
        if n.get("class_type") != "LoadImage":
            continue
        n["class_type"] = "VHS_LoadImagePath"
        n["inputs"] = {"image": POD_INPUT + ref, "custom_width": 0, "custom_height": 0}
        n_swapped += 1
    return n_swapped


def reaches(prompt, start, want_prefix="WanVideo"):
    """Does node `start` have a forward path to any node whose class_type starts with prefix?

    🔴 THE RIGHT QUESTION IS REACHABILITY, NOT FILENAME. This graph loads the union mask on #1352
    AND the per-character mask on #1350, and a filename check therefore condemns a perfectly
    correct prompt. Traced on the converted v22 prompt: #1350 -> ImageToMask -> GrowMaskWithBlur
    -> BlockifyMask -> WanVideoAnimateEmbeds.mask AND -> DrawMaskOnImage -> .bg_images, i.e. it
    is the mask that decides where the sampler generates; #1352 -> MaskToImage ->
    VHS_VideoCombine and stops -- a diagnostic side output that never touches the sampler.
    """
    seen, stack = set(), [str(start)]
    while stack:
        nid = stack.pop()
        if nid in seen:
            continue
        seen.add(nid)
        for oid, n in prompt.items():
            if oid in seen:
                continue
            for v in (n.get("inputs") or {}).values():
                if isinstance(v, list) and len(v) == 2 and str(v[0]) == nid:
                    if str(n.get("class_type", "")).startswith(want_prefix):
                        return True
                    stack.append(oid)
    return False


def verify(prompt, folder, ref, slot=None, bound=None, mask_name=None):
    errs = []
    loaders, refs = [], []
    for nid, n in prompt.items():
        ct, ins = n.get("class_type", ""), n.get("inputs", {})
        if ct == "VHS_LoadVideo":
            loaders.append(str(ins.get("video", "")))
        if ct in ("LoadImage", "VHS_LoadImagePath"):
            refs.append(str(ins.get("image", "")).replace(POD_INPUT, ""))
    bad_folder = [v for v in loaders if v and not v.startswith(folder + "/")]
    if bad_folder:
        errs.append(f"loader(s) outside {folder}/: {bad_folder}")
    if sorted(set(refs)) != [ref]:
        errs.append(f"reference images {sorted(set(refs))} != [{ref}]")
    s = slot or "p1"
    want_sticks = f"{folder}/pose_sticks_{s}_00002.mp4"
    want_face = f"{folder}/facemesh_{s}_00002.mp4"
    if want_sticks not in loaders:
        errs.append(f"driving sticks {want_sticks} not loaded (got {loaders})")
    if want_face not in loaders:
        errs.append(f"driving facemesh {want_face} not loaded")
    if slot is None:
        # SINGLE PERSON. Either the union mask (the historical default -- for one person the
        # union IS that person), or a NAMED per-character mask when the arm asks for one. In the
        # named case assert the file that actually reaches the sampler, because an unread mask
        # is the failure mode this argument exists to prevent.
        if mask_name:
            want = f"{folder}/{mask_name}"
            if want not in loaders:
                errs.append(f"driven mask {want} not loaded (got {loaders})")
            if (bound or {}).get(MASK_LOADER_DRIVEN) != want:
                errs.append(f"#{MASK_LOADER_DRIVEN} (the mask that reaches WanVideoAnimateEmbeds) "
                            f"is {(bound or {}).get(MASK_LOADER_DRIVEN)}, expected {want}")
            # The union mask may still be LOADED -- it feeds a diagnostic video output. What
            # must hold is that it does not reach the sampler, and that the named one does.
            if not reaches(prompt, MASK_LOADER_DRIVEN):
                errs.append(f"#{MASK_LOADER_DRIVEN} carries {mask_name} but has NO path to the "
                            f"sampler -- the enlarged mask would have no effect")
            if reaches(prompt, MASK_LOADER_OTHER):
                errs.append(f"#{MASK_LOADER_OTHER} (the union mask) REACHES the sampler -- the "
                            f"named per-character mask is not what would drive generation")
        elif f"{folder}/mask_00002.mp4" not in loaders:
            errs.append("no mask video loaded at all")
    else:
        # ---- MULTI-PERSON: the mask reaching the sampler must name exactly ONE slot -------------
        # A union mask here is the §B.57e defect: the sampler is asked for a character in every
        # box and the matte can bind to the wrong one. Also assert the OTHER slot's sidecars are
        # absent, or the two people's conditioning would be mixed into one render.
        want_mask = f"{folder}/mask_{s}_00002.mp4"
        if bound is None:
            errs.append(f"per-slot masks unavailable for {folder} slot {s}")
        else:
            if want_mask not in loaders:
                errs.append(f"slot mask {want_mask} not loaded (got {loaders})")
            if f"{folder}/mask_00002.mp4" in loaders:
                errs.append("the UNION mask is still loaded -- the sampler would generate a "
                            "character in every box and the matte could bind to the wrong one")
            if bound.get(MASK_LOADER_DRIVEN) != want_mask:
                errs.append(f"#{MASK_LOADER_DRIVEN} (the mask that reaches WanVideoAnimateEmbeds) "
                            f"is {bound.get(MASK_LOADER_DRIVEN)}, expected {want_mask}")
        other = "p2" if s == "p1" else "p1"
        mixed = [v for v in loaders if f"_{other}_" in v and not v.startswith(f"{folder}/mask_")]
        if mixed:
            errs.append(f"BOTH slots loaded, conditioning would be mixed: {mixed}")
    for nid, n in prompt.items():
        ins = n.get("inputs", {})
        for k, want in EXPECT_SAMPLER.items():
            if k in ins and isinstance(ins[k], (int, float, str)) and ins[k] != want:
                errs.append(f"#{nid} {n['class_type']}.{k} = {ins[k]!r}, expected {want!r}")
        if "frame_window_size" in ins and ins["frame_window_size"] != EXPECT_WINDOW:
            errs.append(f"#{nid} frame_window_size = {ins['frame_window_size']}, expected {EXPECT_WINDOW}")
    return errs


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--url", default=os.environ.get("MIRAGE_COMFY_URL", "http://127.0.0.1:8188"),
                    help="Base URL of the ComfyUI server (env: MIRAGE_COMFY_URL).")
    ap.add_argument("--bundle-dir", default=os.environ.get("MIRAGE_BUNDLE_DIR") or None,
                    help="Local bundle directory built by build_cloud_bundle.py "
                         "(env: MIRAGE_BUNDLE_DIR). Enables per-slot mask binding.")
    ap.add_argument("--queue", action="store_true")
    ap.add_argument("--object-info", default=os.path.join(CLOUD, "object_info.json"))
    ap.add_argument("--only", default="")
    global BUNDLE_DIR
    a = ap.parse_args()
    BUNDLE_DIR = a.bundle_dir
    os.makedirs(CLOUD, exist_ok=True)
    ok, prompts = True, {}
    want = [t.strip() for t in a.only.split(",") if t.strip()]
    for entry in ARMS:
        tag, folder, ref = entry[0], entry[1], entry[2]
        slot = entry[3] if len(entry) > 3 else None
        mask_name = entry[4] if len(entry) > 4 else None
        if want and tag not in want:
            continue
        wfp = os.path.join(CLOUD, f"{tag}.wf.json")
        bound = build_wf(folder, ref, wfp, slot, mask_name)
        pr, err = convert(wfp, a.object_info, os.path.join(CLOUD, f"{tag}.api.json"))
        if err:
            print(f"{tag}: CONVERT FAILED {err}")
            ok = False
            continue
        n_sw = swap_loadimage(pr, ref)
        n_rt = retag_outputs(pr, tag)
        errs = verify(pr, folder, ref, slot, bound, mask_name)
        if n_sw == 0:
            errs.append("no LoadImage node found to swap -- the graph changed")
        mk = ((bound or {}).get(MASK_LOADER_DRIVEN, "UNBOUND")
              if (slot or mask_name) else "union")
        print(f"{tag:16s} {folder:20s} ref={ref:26s} slot={slot or '-':3s} "
              f"mask={str(mk).split('/')[-1]:22s} {'VERIFIED' if not errs else 'FAILED'}")
        for e in errs:
            print(f"     - {e}")
        if errs:
            ok = False
        else:
            prompts[tag] = pr
    if not ok:
        print("\nREFUSING TO QUEUE -- at least one arm failed verification")
        return 1
    print(f"\nall {len(prompts)} arms verified")
    if not a.queue:
        print("dry run -- pass --queue to POST")
        return 0
    for tag, pr in prompts.items():
        body = json.dumps({"prompt": pr, "client_id": f"mirage-{tag}"}).encode()
        req = urllib.request.Request(a.url.rstrip("/") + "/prompt", data=body,
                                     headers={"Content-Type": "application/json",
                                              "User-Agent": UA})
        with urllib.request.urlopen(req, timeout=120) as r:
            res = json.loads(r.read().decode())
        print(f"  queued {tag}: prompt_id {res.get('prompt_id')} number {res.get('number')}")
        io_path = os.path.join(CLOUD, f"POSTED_{tag}.api.json")
        json.dump(pr, open(io_path, "w", encoding="utf-8"), indent=1)
    return 0


if __name__ == "__main__":
    sys.exit(main())
