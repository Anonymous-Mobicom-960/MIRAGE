#!/usr/bin/env python3
"""V9_CLOUD_ONLY.json -> V9_A20_<tag>.json : the A20-mask cloud A/B, with a MANUAL reference image.

WHY THIS EXISTS
---------------
Ledger §A.6k-23 picked `SHIP_r4d_a20_t135` on privacy (worst-case attacker 56.07 % -> 9.35 %), but
its mask area sits above 1.358x - the only ratio whose PICTURE-QUALITY cost has ever been measured
(§B.48). This builds the cloud half of the render that prices it.

THREE THINGS IT CHANGES, and nothing else:

  1. INPUT FOLDER. The six VHS_LoadVideo nodes point at the uploaded bundle.

  2. MANUAL REFERENCE IMAGE, replacing MIRAGE_GeminiAutoRef. The workflow already ships
     `LoadImage` nodes titled "MANUAL REF (bypassed fallback)" for exactly this; this un-bypasses
     one, points it at the chosen PNG, and re-origins the two links that feed the reference
     resizers. The AutoRef SaveImage nodes are muted so the AutoRef branch becomes unreachable and
     ComfyUI never executes it - no OpenAI call, no generation cost, and the reference is a fixed
     input rather than a per-run sample. The build FAILS if AutoRef is still REQUIRED by a live
     output, or if any required node is muted -- in ComfyUI a muted-but-required node is an
     execution error, not a silent skip. An earlier version of that check skipped muted nodes
     while walking, which made it unfalsifiable; it is now a plain requirement walk with the mode
     test applied afterwards, and it caught a real breakage the first time it ran.

  3. THE SHIPPED PRE-HQ GENERATION SETTINGS. 🔴 The committed V9_CLOUD_ONLY.json is NOT at them - 
     it still carries the older `c4_ctrl` values (cfg 1.2, LoRAs 0.75/0.9/0.6/0.9/0.3). The
     shipped arm is `c4_van` (§B.44: keypoint 0.0267, expression 0.0500, yaw 0.0327 - the best of
     any arm measured, and 0.00 % invented hands). This script writes the c4_van values in, so the
     render measures the shipped generator and not a superseded one.

`frame_window_size` stays at 77 (node #62) - the window the whole cost model is built on: cost is
linear in ceil(frames/77) windows at ~103 Wh each (§B.10). A 50-frame clip is ONE window.

Everything else - model, VAE, block swap, sampler steps/scheduler/shift, pose_strength 0.8,
face_strength 0.85, CLIP-vision strengths 0.25/1.0 - is left exactly as the shipped run had it.
"""
import argparse
import copy
import io
import json
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
SRC = os.path.join(HERE, "ComfyUI", "user", "default", "workflows", "V9_CLOUD_ONLY.json")

# ---- the shipped pre-HQ generation config (c4_van, ledger §B.44) -----------------------------
C4_VAN_SAMPLER = dict(steps=5, cfg=1.0, shift=5.0, scheduler="dpm++_sde")
C4_VAN_LORAS = [("none", 0),
                ("lightx2v_I2V_14B_480p_cfg_step_distill_rank64_bf16.safetensors", 1.0),
                ("Wan2.1_HighRes_Textures.safetensors", 0.3),
                ("WanVideo/Wan14B_RealismBoost.safetensors", 0.35),
                ("WanVideo/Wan21_Camera_Rotation.safetensors", 0.0)]
FRAME_WINDOW = 77

LOADERS = {                       # node id -> file inside the uploaded folder
    1348: "masked_video_00002.mp4",
    1350: "mask_00002.mp4",
    1352: "mask_00002.mp4",
    1354: "pose_sticks_p1_00002.mp4",
    1355: "facemesh_p1_00002.mp4",
    1357: "light_map.mp4",
}
MANUAL_REF_NODE = 1339            # a "MANUAL REF (bypassed fallback)" LoadImage
REF_RESIZERS = {405: 1937,        # primary  (CLIP-vision image_1, strength 1.00) <- link 1937
                403: 1939}        # secondary(CLIP-vision image_2, strength 0.25) <- link 1939
AUTOREF_NODES = (943, 944)
MUTED = 2                         # LiteGraph: 0 active, 2 muted (never executes), 4 bypass
BYPASS = 4                        # passes inputs straight through; never executes itself

# Output nodes to mute, and why each one. Derived by diffing the committed workflow's live
# outputs against what the shipped `c4_van` run ACTUALLY executed (_e2e/check4s/history/c4_van.json)
# -- not by guesswork:
#   1134/1135  autoref_p1_refsheet / _fullbody -- c4_van RAN these, but they exist only to save the
#              GENERATED reference. With a manual reference there is nothing to save, and leaving
#              them live is precisely what keeps AutoRef #943 required.
#   1136/1137  autoref_p2_*  -- c4_van MUTED these.
#   1175/1177  synthetic_person_p2 / synthetic_alpha_p2 -- c4_van MUTED these. Both clips here are
#              single-person, so the P2 branch has no subject to render.
# The last four are the whole reason #944 was still required: the committed workflow leaves them
# live, the shipped run did not. Muting #944 without them is an execution error, which is exactly
# what the validator now catches.
MUTE_OUTPUTS = (1134, 1135, 1136, 1137, 1175, 1177)

# Nodes that are NOT required by any live output but whose CLASS must still resolve in
# /object_info for ui2api.py to convert the graph. #1379 MIRAGEMaskPersonCount is dead weight --
# its only consumer is an unused SetNode -- but the 2026-08-01 second pod did not have that
# custom-node package registered, so the converter hard-errored on a node the graph never
# executes. Muting it drops it before conversion and changes nothing about what runs.
# (The first pod that day DID have the class, which is why the same workflow converted fine
# there: the converter could resolve-then-prune it. Same volume, different registered nodes --
# so do not assume two pods expose the same /object_info.)
MUTE_UNUSED = (1379,)


def nodes_by_id(g):
    return {int(n["id"]): n for n in g["nodes"]}


def links_by_id(g):
    return {int(l[0]): l for l in g["links"]}


def required_by_outputs(g):
    """Every node ComfyUI must execute to satisfy the live output nodes.

    Two things this gets right that the obvious version does not:

      * SET/GET ARE VIRTUAL. KJNodes' GetNode has NO input link - it resolves to the SetNode with
        the same key at runtime. A walk that only follows links stops dead at every GetNode; this
        graph has 103 of them, so such a walk sees 24 nodes where the real execution is 44.

      * MODE IS NOT CONSULTED DURING THE WALK. An earlier version skipped muted nodes while
        walking, which made "is the AutoRef still needed?" unfalsifiable: muting the node was
        itself what made it look unreachable. The walk must record what is REQUIRED, and the
        caller then checks whether anything required happens to be muted - in ComfyUI that is an
        execution error, not a silent skip.
    """
    N = nodes_by_id(g)
    L = links_by_id(g)
    setters = {}
    for i, n in N.items():
        if n.get("type") == "SetNode":
            wv = n.get("widgets_values")
            key = wv[0] if isinstance(wv, list) and wv else (
                list(wv.values())[0] if isinstance(wv, dict) and wv else None)
            if key is not None:
                setters[str(key)] = i
    out_types = ("SaveImage", "VHS_VideoCombine", "PreviewImage", "SaveAnimatedWEBP")
    req, stack = set(), [i for i, n in N.items()
                         if n.get("type") in out_types and n.get("mode", 0) != MUTED]
    while stack:
        i = stack.pop()
        if i in req or i not in N:
            continue
        req.add(i)
        n = N[i]
        for inp in (n.get("inputs") or []):
            lid = inp.get("link")
            if lid is not None and int(lid) in L:
                stack.append(int(L[int(lid)][1]))
        if n.get("type") == "GetNode":
            wv = n.get("widgets_values")
            key = wv[0] if isinstance(wv, list) and wv else (
                list(wv.values())[0] if isinstance(wv, dict) and wv else None)
            if key is not None and str(key) in setters:
                stack.append(setters[str(key)])
    # A BYPASSED node passes data through but is never EXECUTED, so it must not appear in the
    # executed set -- otherwise it shows up as a spurious diff against a real API graph
    # (#35 WanVideoTorchCompileSettings did exactly that). Its upstream is still required, which
    # the link walk above has already collected.
    return {i for i in req if N[i].get("mode", 0) != BYPASS}


def widget_prefix(n):
    """SaveImage stores filename_prefix as a LIST in the UI format and a dict in the API format.
    Reading only the dict form silently skipped every SaveImage and left the AutoRef branch live.
    """
    wv = n.get("widgets_values")
    if isinstance(wv, dict):
        return wv.get("filename_prefix")
    if isinstance(wv, list) and wv and isinstance(wv[0], str):
        return wv[0]
    return None


def build(folder, ref, tag, fps, out_path):
    g = json.load(io.open(SRC, encoding="utf-8"))
    N, L = nodes_by_id(g), links_by_id(g)
    log = []

    # ---- 1. inputs ---------------------------------------------------------------------------
    for nid, fn in LOADERS.items():
        assert nid in N and N[nid]["type"] == "VHS_LoadVideo", f"#{nid} is not a VHS_LoadVideo"
        N[nid]["widgets_values"]["video"] = f"{folder}/{fn}"
        log.append(f"  #{nid} video = {folder}/{fn}")

    # ---- 2. manual reference image, AutoRef made unreachable -----------------------------------
    m = N[MANUAL_REF_NODE]
    assert m["type"] == "LoadImage", f"#{MANUAL_REF_NODE} is not a LoadImage"
    m["mode"] = 0                                            # un-bypass
    wv = m.get("widgets_values")
    if isinstance(wv, dict):
        wv["image"] = ref
    else:
        m["widgets_values"] = [ref, "image"]
    m.setdefault("outputs", [])
    m["outputs"][0].setdefault("links", [])
    m["outputs"][0]["links"] = list(m["outputs"][0].get("links") or [])
    log.append(f"  #{MANUAL_REF_NODE} LoadImage = {ref!r}  (un-bypassed)")

    for resizer, lid in REF_RESIZERS.items():
        assert lid in L, f"link {lid} not found"
        old_src = int(L[lid][1])
        L[lid][1] = MANUAL_REF_NODE                          # re-origin: AutoRef -> LoadImage
        L[lid][2] = 0
        if old_src in N:
            for o in (N[old_src].get("outputs") or []):
                if o.get("links") and lid in o["links"]:
                    o["links"].remove(lid)
        m["outputs"][0]["links"].append(lid)
        log.append(f"  link {lid}: #{old_src} -> #{MANUAL_REF_NODE}  (feeds #{resizer})")

    for nid in AUTOREF_NODES:
        if nid in N:
            N[nid]["mode"] = MUTED
            log.append(f"  #{nid} MIRAGE_GeminiAutoRef MUTED")
    for nid in MUTE_OUTPUTS:
        assert nid in N, f"#{nid} not in graph"
        N[nid]["mode"] = MUTED
        log.append(f"  #{nid} {N[nid]['type']} {widget_prefix(N[nid])!r} MUTED")
    for nid in MUTE_UNUSED:
        if nid in N:
            N[nid]["mode"] = MUTED
            log.append(f"  #{nid} {N[nid]['type']} MUTED (unused; class may be absent on a pod)")

    # ---- 3. the shipped pre-HQ generation settings --------------------------------------------
    for nid, n in N.items():
        t, wv = n.get("type"), n.get("widgets_values")
        if t == "WanVideoSampler" and isinstance(wv, list):
            wv[0], wv[1], wv[2] = C4_VAN_SAMPLER["steps"], C4_VAN_SAMPLER["cfg"], \
                C4_VAN_SAMPLER["shift"]
            wv[6] = C4_VAN_SAMPLER["scheduler"]
            log.append(f"  #{nid} sampler -> steps {wv[0]}, cfg {wv[1]}, shift {wv[2]}, {wv[6]}")
        elif t == "WanVideoLoraSelectMulti" and isinstance(wv, list):
            for k, (name, st) in enumerate(C4_VAN_LORAS):
                assert wv[2 * k] == name, f"LoRA slot {k} is {wv[2*k]!r}, expected {name!r}"
                wv[2 * k + 1] = st
            log.append(f"  #{nid} LoRAs -> {[s for _, s in C4_VAN_LORAS]}")
        elif t == "WanVideoAnimateEmbeds" and isinstance(wv, list):
            assert wv[4] == FRAME_WINDOW, f"frame_window_size is {wv[4]}, expected {FRAME_WINDOW}"
            log.append(f"  #{nid} frame_window_size {wv[4]} (unchanged), "
                       f"pose_strength {wv[6]}, face_strength {wv[7]}")
        elif t == "VHS_VideoCombine" and isinstance(wv, dict):
            wv["frame_rate"] = fps
            wv["filename_prefix"] = str(wv["filename_prefix"]).replace("CLOUDONLY", tag)

    # ---- 4. validate ---------------------------------------------------------------------------
    req = required_by_outputs(g)
    muted_but_required = sorted(i for i in req if N[i].get("mode", 0) == MUTED)
    if muted_but_required:
        detail = "\n  ".join(f"#{i} {N[i].get('type')} "
                             f"{str(N[i].get('widgets_values'))[:60]}"
                             for i in muted_but_required)
        sys.exit("REFUSING: these nodes are MUTED but still REQUIRED by a live output, which is "
                 "an execution error in ComfyUI, not a silent skip:\n  " + detail)
    still = [i for i in AUTOREF_NODES if i in req]
    if still:
        sys.exit(f"REFUSING: AutoRef {still} is still REQUIRED - the OpenAI call would fire and "
                 f"the reference would not be the manual one.")
    ids = [int(n["id"]) for n in g["nodes"]]
    assert len(ids) == len(set(ids)), "duplicate node ids"
    lids = [int(l[0]) for l in g["links"]]
    assert len(lids) == len(set(lids)), "duplicate link ids"
    known = set(ids)
    for l in g["links"]:
        assert int(l[1]) in known and int(l[3]) in known, f"dangling link {l[0]}"
    # The strongest available check: the set of nodes this will execute should be the set the
    # SHIPPED run executed, minus the AutoRef branch, plus the manual LoadImage. Anything else
    # means the committed workflow has drifted from the shipped one in a way we have not noticed.
    ref_hist = os.path.join(HERE, "..", "..", "_e2e", "check4s", "history", "c4_van.json")
    if os.path.exists(ref_hist):
        shipped = {int(k) for k in
                   list(json.load(io.open(ref_hist, encoding="utf-8")).values())[0]["prompt"][2]}
        # #1146 MaskToImage is the AutoRef branch's own feeder (it supplied silhouette_frames),
        # so it correctly drops out with the branch rather than indicating drift.
        expected_gone = set(AUTOREF_NODES) | set(MUTE_OUTPUTS) | {1146}
        missing = (shipped - expected_gone) - req
        extra = req - shipped - {MANUAL_REF_NODE}
        # Set/Get are virtual: they appear in the UI walk but never in an executed API graph.
        virt = {i for i in (missing | extra) if N[i].get("type") in ("SetNode", "GetNode")}
        missing, extra = missing - virt, extra - virt
        if missing or extra:
            print(f"  NOTE vs shipped c4_van: missing={sorted(missing)} extra={sorted(extra)}")
        else:
            print("  executed set == shipped c4_van set, minus AutoRef, plus the manual LoadImage")
    srcN = nodes_by_id(json.load(io.open(SRC, encoding="utf-8")))
    assert {n["type"] for n in g["nodes"]} == {n["type"] for n in srcN.values()}, \
        "node TYPES changed - that would need a ComfyUI server restart"

    io.open(out_path, "w", encoding="utf-8").write(json.dumps(g, indent=1, ensure_ascii=False))
    print(f"{os.path.basename(out_path)}   ({len(g['nodes'])} nodes, "
          f"{len(req)} executed, AutoRef NOT required)")
    for line in log:
        print(line)
    return out_path


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--folder", required=True, help="input folder name on the volume")
    ap.add_argument("--ref", required=True, help="reference image filename in ComfyUI/input")
    ap.add_argument("--tag", required=True, help="output filename prefix, replaces CLOUDONLY")
    ap.add_argument("--fps", type=float, default=30.0)
    ap.add_argument("--out", default=None)
    a = ap.parse_args()
    out = a.out or os.path.join(os.path.dirname(SRC), f"V9_A20_{a.tag}.json")
    build(a.folder, a.ref, a.tag, a.fps, out)


if __name__ == "__main__":
    main()
