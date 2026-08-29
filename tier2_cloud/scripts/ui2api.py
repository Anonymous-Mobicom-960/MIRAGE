#!/usr/bin/env python3
"""
ui2api.py - convert a ComfyUI *UI* (litegraph) workflow into an *API* prompt.

Why this exists: `POST /prompt` only accepts the API format. The ComfyUI frontend
normally produces it (graphToPrompt), but we queue headlessly, so we have to do the
same three jobs it does:

  1. FLATTEN SUBGRAPHS   V9_CLOUD_ONLY has one ("2nd Character Generation", 14 nodes).
                         Subgraphs are a frontend-only concept; the backend never sees
                         them. Inner links reference two pseudo-nodes: the input
                         boundary (id -10) and the output boundary (id -20).
  2. RESOLVE VIRTUAL NODES  KJNodes SetNode/GetNode are frontend-only. A GetNode's
                         output is really whatever feeds the same-named SetNode.
                         103 of this graph's 171 nodes are Set/Get.
  3. APPLY MODES         mode 2 = MUTE (node vanishes), mode 4 = BYPASS (each output
                         passes through the first input of a matching type).

Then widget values have to be bound to input *names*. `widgets_values` is positional,
so we walk the node's declared inputs from /object_info in order and consume one value
each -- plus ONE EXTRA for any input carrying `control_after_generate` (the seed
control combo is a frontend-only widget that still occupies a slot). VHS nodes
serialise `widgets_values` as a name-keyed dict instead, which needs no positional
guessing.

Usage:
    python ui2api.py WORKFLOW.json --object-info object_info.json -o prompt_api.json
    python ui2api.py WORKFLOW.json --structure-only     # offline: flattening checks only
"""
import argparse
import json
import random
import sys
from collections import defaultdict

VIRTUAL = {"SetNode", "GetNode"}
NON_EXEC = {"Note", "MarkdownNote", "Reroute"}  # Reroute handled as passthrough below
MODE_MUTE, MODE_BYPASS = 2, 4

# Widget-bearing primitive types. Anything else is a link-only input.
WIDGET_SCALARS = {"INT", "FLOAT", "STRING", "BOOLEAN", "COMBO"}


class Converter:
    def __init__(self, wf, object_info=None):
        self.wf = wf
        self.oi = object_info or {}
        self.nodes = {}       # key -> node dict (key is str; inner nodes are "936_718")
        self.raw_src = {}     # (key, in_slot) -> (src_key, src_slot)
        self.pseudo = {}      # subgraph boundary bookkeeping
        self.warnings = []
        self.errors = []
        self.defaulted = []
        self._resolve_memo = {}
        self._build()

    # ---------------- graph assembly ----------------
    def _build(self):
        subgraph_defs = {s["id"]: s for s in ((self.wf.get("definitions") or {}).get("subgraphs") or [])}

        for n in self.wf.get("nodes", []):
            self.nodes[str(n["id"])] = dict(n, _key=str(n["id"]), _ns=None)

        for link in self.wf.get("links", []):
            lid, oid, oslot, tid, tslot = link[0], link[1], link[2], link[3], link[4]
            self.raw_src[(str(tid), tslot)] = (str(oid), oslot)

        # Flatten every subgraph instance found at the top level.
        for n in list(self.nodes.values()):
            sg = subgraph_defs.get(n["type"])
            if sg is None:
                continue
            self._flatten(n, sg)

    def _flatten(self, inst, sg):
        """Inline one subgraph instance. Inner ids are namespaced '<instid>_<innerid>'."""
        inst_id = inst["_key"]
        ns = lambda i: f"{inst_id}_{i}"

        # Boundary name/index sanity: the instance's slot order must match the definition's.
        inst_in_names = [i.get("name") for i in inst.get("inputs", [])]
        def_in_names = [i.get("name") for i in sg.get("inputs", [])]
        if inst_in_names != def_in_names:
            self.errors.append(
                f"#{inst_id}: subgraph INPUT slot order differs from its definition\n"
                f"    instance  : {inst_in_names}\n    definition: {def_in_names}")
        inst_out_names = [o.get("name") for o in inst.get("outputs", [])]
        def_out_names = [o.get("name") for o in sg.get("outputs", [])]
        if inst_out_names != def_out_names:
            self.errors.append(
                f"#{inst_id}: subgraph OUTPUT slot order differs from its definition\n"
                f"    instance  : {inst_out_names}\n    definition: {def_out_names}")

        for n in sg.get("nodes", []):
            self.nodes[ns(n["id"])] = dict(n, _key=ns(n["id"]), _ns=inst_id)

        in_key, out_key = f"{inst_id}:IN", f"{inst_id}:OUT"
        self.pseudo[in_key] = inst_id
        self.pseudo[out_key] = inst_id

        for link in sg.get("links", []):
            if isinstance(link, dict):
                oid, oslot = link["origin_id"], link["origin_slot"]
                tid, tslot = link["target_id"], link["target_slot"]
            else:
                oid, oslot, tid, tslot = link[1], link[2], link[3], link[4]
            src = in_key if oid == -10 else ns(oid)
            dst = out_key if tid == -20 else ns(tid)
            self.raw_src[(dst, tslot)] = (src, oslot)

    # ---------------- source resolution ----------------
    def resolve(self, key, slot, _seen=None):
        """Follow a (node, output-slot) reference down to a real executable node."""
        memo_k = (key, slot)
        if memo_k in self._resolve_memo:
            return self._resolve_memo[memo_k]
        _seen = _seen or set()
        if memo_k in _seen:
            self.errors.append(f"CYCLE while resolving {key}:{slot}")
            return None
        _seen = _seen | {memo_k}

        out = self._resolve_uncached(key, slot, _seen)
        self._resolve_memo[memo_k] = out
        return out

    def _resolve_uncached(self, key, slot, seen):
        # subgraph input boundary -> the instance's own input at that slot
        if key.endswith(":IN"):
            inst = self.pseudo[key]
            up = self.raw_src.get((inst, slot))
            return self.resolve(*up, _seen=seen) if up else None

        node = self.nodes.get(key)
        if node is None:
            self.errors.append(f"missing node {key}")
            return None

        # a subgraph INSTANCE appearing as a source -> its output boundary
        if node["type"] in {s["id"] for s in ((self.wf.get("definitions") or {}).get("subgraphs") or [])}:
            up = self.raw_src.get((f"{key}:OUT", slot))
            return self.resolve(*up, _seen=seen) if up else None

        t, mode = node["type"], node.get("mode", 0)

        if t == "GetNode":
            name = (node.get("widgets_values") or [None])[0]
            setter = self.setnode_by_name.get(name)
            if setter is None:
                self.errors.append(f"GetNode #{key} wants '{name}' but no SetNode defines it")
                return None
            up = self.raw_src.get((setter, 0))
            return self.resolve(*up, _seen=seen) if up else None

        if t == "SetNode":                      # output chains through its input
            up = self.raw_src.get((key, 0))
            return self.resolve(*up, _seen=seen) if up else None

        if t == "Reroute":
            up = self.raw_src.get((key, 0))
            return self.resolve(*up, _seen=seen) if up else None

        if mode == MODE_MUTE:
            return None

        if mode == MODE_BYPASS:
            # pass through the first input whose type matches this output's type
            otype = (node.get("outputs") or [{}])[slot].get("type") if slot < len(node.get("outputs") or []) else None
            for i, inp in enumerate(node.get("inputs") or []):
                if inp.get("type") == otype:
                    up = self.raw_src.get((key, i))
                    return self.resolve(*up, _seen=seen) if up else None
            return None

        return (key, slot)

    @property
    def setnode_by_name(self):
        if not hasattr(self, "_setmap"):
            m = {}
            for k, n in self.nodes.items():
                if n["type"] == "SetNode":
                    nm = (n.get("widgets_values") or [None])[0]
                    if nm in m:
                        self.warnings.append(f"duplicate SetNode name '{nm}' (#{m[nm]} and #{k})")
                    m[nm] = k
            self._setmap = m
        return self._setmap

    # ---------------- widget binding ----------------
    def _declared_inputs(self, ctype):
        """Ordered [(name, spec)] for a class, required first then optional."""
        info = self.oi.get(ctype)
        if info is None:
            return None
        inp = info.get("input", {})
        req, opt = inp.get("required", {}) or {}, inp.get("optional", {}) or {}
        order = info.get("input_order") or {}
        names = list(order.get("required") or req.keys()) + list(order.get("optional") or opt.keys())
        merged = {**req, **opt}
        return [(n, merged[n]) for n in names if n in merged]

    @staticmethod
    def _opts(spec):
        if isinstance(spec, (list, tuple)) and len(spec) > 1 and isinstance(spec[1], dict):
            return spec[1]
        return {}

    @classmethod
    def _is_widget(cls, spec):
        # forceInput inputs are link-only sockets even though their type is scalar,
        # so they never consume a widgets_values slot (e.g. Sam2Segmentation's
        # coordinates_*, MIRAGEPersonGate's person_count).
        if cls._opts(spec).get("forceInput"):
            return False
        tp = spec[0] if isinstance(spec, (list, tuple)) and spec else spec
        if isinstance(tp, list):          # combo box
            return True
        return tp in WIDGET_SCALARS

    @classmethod
    def _has_seed_control(cls, name, spec):
        # The frontend appends a control_after_generate combo either when the input
        # declares it, or purely by name -- WanVideoSampler's seed carries no flag but
        # still serialises a trailing "fixed"/"randomize" value.
        if cls._opts(spec).get("control_after_generate"):
            return True
        tp = spec[0] if isinstance(spec, (list, tuple)) and spec else spec
        return name in ("seed", "noise_seed") and tp == "INT"

    randomize_seeds = True      # honour control_after_generate="randomize" headlessly
    reseeded = []               # audit trail of which seeds this conversion re-rolled

    def bind_widgets(self, node):
        """Return {input_name: value} for a node's widget inputs."""
        ctype = node["type"]
        wv = node.get("widgets_values")
        decl = self._declared_inputs(ctype)
        if decl is None:
            self.errors.append(f"#{node['_key']}: class '{ctype}' NOT in /object_info")
            return {}

        widget_inputs = [(n, s) for n, s in decl if self._is_widget(s)]

        if isinstance(wv, dict):
            # VHS-style, name-keyed. Keep keys that are NOT declared inputs: VHS marks
            # its hidden dict with ContainsAll, so ComfyUI forwards any extra name into
            # **kwargs -- that is how pix_fmt/crf/save_metadata actually reach the
            # encoder. Only `videopreview` is a pure DOM widget and must go.
            out = {k: v for k, v in wv.items() if k != "videopreview"}
            return out

        wv = list(wv or [])
        out, i = {}, 0
        for name, spec in widget_inputs:
            if i < len(wv):
                out[name] = wv[i]; i += 1
                if self._has_seed_control(name, spec):
                    # The trailing combo is control_after_generate. In the UI, "randomize"
                    # bumps the seed on every queue; headless /prompt has no frontend, so a
                    # `randomize` seed would otherwise be FROZEN at whatever value happened to
                    # be saved in the .json -- every headless render reusing one seed.
                    # MEASURED CONSEQUENCE (2026-07-26): MIRAGE_GeminiAutoRef seed was pinned at
                    # 261698084363409 across every e2e run, so the generated character came out
                    # with the same race and the same clothing every single time. Honour the
                    # widget's own mode instead.
                    mode = wv[i] if i < len(wv) else None
                    if isinstance(mode, str) and mode in ("randomize", "increment", "decrement"):
                        if mode == "randomize" and self.randomize_seeds:
                            out[name] = random.randrange(0, 0xffffffffffffffff)
                            self.reseeded.append(f"#{node['_key']}.{name}")
                        elif mode == "increment":
                            out[name] = int(out[name]) + 1
                        elif mode == "decrement":
                            out[name] = int(out[name]) - 1
                    i += 1                               # consume control_after_generate
            else:
                # Widget added to the node AFTER this workflow was saved: the frontend
                # would show the declared default, so send that.
                default = self._opts(spec).get("default")
                if default is None and isinstance(spec, (list, tuple)) and isinstance(spec[0], list) and spec[0]:
                    default = spec[0][0]                 # combo with no explicit default
                out[name] = default
                self.defaulted.append(f"#{node['_key']} {ctype}.{name} = {default!r} (not in saved workflow)")

            # BOOLEAN widgets can carry a stale non-bool from an older node version
            # (e.g. "" where a widget was inserted since). Coerce, matching how the
            # node itself would evaluate it.
            tp = spec[0] if isinstance(spec, (list, tuple)) and spec else spec
            if tp == "BOOLEAN" and not isinstance(out[name], bool):
                self.warnings.append(
                    f"#{node['_key']} {ctype}.{name}: {out[name]!r} is not a bool -> {bool(out[name])}")
                out[name] = bool(out[name])

        if i != len(wv):
            self.warnings.append(
                f"#{node['_key']} {ctype}: consumed {i}/{len(wv)} widget values "
                f"(expected inputs: {[n for n, _ in widget_inputs]})")
        return out

    # ---------------- emission ----------------
    def to_prompt(self):
        subgraph_types = {s["id"] for s in ((self.wf.get("definitions") or {}).get("subgraphs") or [])}
        prompt = {}
        for key, node in self.nodes.items():
            t = node["type"]
            if t in VIRTUAL or t in NON_EXEC or t in subgraph_types:
                continue
            if node.get("mode", 0) in (MODE_MUTE, MODE_BYPASS):
                continue

            entry = {"class_type": t, "_meta": {"title": node.get("title") or t}}
            inputs = self.bind_widgets(node) if self.oi else {}

            for i, inp in enumerate(node.get("inputs") or []):
                name = (inp.get("widget") or {}).get("name") or inp.get("name")
                up = self.raw_src.get((key, i))
                if up is None:
                    continue
                r = self.resolve(*up)
                if r is None:
                    self.warnings.append(
                        f"#{key} {t}: input '{name}' resolves to nothing (bypassed/muted upstream)")
                    inputs.pop(name, None)
                    continue
                inputs[name] = [self.api_id(r[0]), r[1]]

            entry["inputs"] = inputs
            prompt[self.api_id(key)] = entry
        return prompt

    @staticmethod
    def api_id(key):
        return key.replace(":", "_")

    # ---------------- pruning + privacy ----------------
    def output_nodes(self, prompt):
        """Node ids ComfyUI will treat as execution roots."""
        roots = []
        for k, v in prompt.items():
            info = self.oi.get(v["class_type"])
            if info is not None:
                if info.get("output_node"):
                    roots.append(k)
            elif v["class_type"] in ("VHS_VideoCombine", "PreviewImage", "SaveImage"):
                roots.append(k)          # offline fallback
        return roots

    @staticmethod
    def subtree(prompt, root, seen=None):
        seen = seen if seen is not None else set()
        if root in seen:
            return seen
        seen.add(root)
        for v in prompt[root]["inputs"].values():
            if isinstance(v, list) and len(v) == 2 and isinstance(v[0], str) and v[0] in prompt:
                Converter.subtree(prompt, v[0], seen)
        return seen

    def prune(self, prompt):
        keep = set()
        for r in self.output_nodes(prompt):
            self.subtree(prompt, r, keep)
        return {k: v for k, v in prompt.items() if k in keep}

    @staticmethod
    def dedupe_loaders(prompt, classes=("VHS_LoadVideo",)):
        """Collapse loader nodes whose inputs are byte-identical, rewiring consumers.

        V9_CLOUD_ONLY loads the SAME union mask twice (#1350 and #1352) because it has no
        per-person mask writer. At 900 frames x 1264^2 each VHS_LoadVideo materialises a
        float32 IMAGE batch of ~17.3 GB, so the duplicate is 17 GB of CPU RAM for a
        bit-identical tensor -- a real OOM risk on a long, high-res clip.

        Safe by construction: nodes are only merged when their class AND their entire
        resolved input dict match exactly, so the outputs cannot differ.
        """
        sig = {}
        canon = {}
        for k, v in prompt.items():
            if v["class_type"] not in classes:
                continue
            s = (v["class_type"], json.dumps(v["inputs"], sort_keys=True, default=str))
            if s in sig:
                canon[k] = sig[s]           # k is a duplicate of sig[s]
            else:
                sig[s] = k
        if not canon:
            return prompt, {}
        for v in prompt.values():
            for name, val in list(v["inputs"].items()):
                if isinstance(val, list) and len(val) == 2 and val[0] in canon:
                    v["inputs"][name] = [canon[val[0]], val[1]]
        for dup in canon:
            prompt.pop(dup, None)
        return prompt, canon

    # Nodes whose whole job is to destroy identity: taint STOPS at their outputs.
    # The internal runbook names #192:0 (P1) and #710:0 (P2) as the only identity-free face
    # signals, and both are this class's output.
    SANITIZERS = {"MIRAGEFaceCanonicalizer"}

    def raw_face_taps(self, prompt):
        """{(api_node_id, slot): why} for every output slot carrying REAL face pixels.

        Found by output NAME from /object_info rather than by hardcoded node ids, so a
        renumbered graph still gets audited. Falls back to the documented ids offline.
        """
        taps = {}
        for key, node in self.nodes.items():
            info = self.oi.get(node["type"])
            if not info:
                continue
            for slot, name in enumerate(info.get("output_name") or []):
                if name == "face_images":
                    taps[(self.api_id(key), slot)] = f"#{key} {node['type']}:{slot} face_images"
        # subgraph boundaries re-export it under the same name
        for s in ((self.wf.get("definitions") or {}).get("subgraphs") or []):
            for inst_key, n in self.nodes.items():
                if n["type"] != s["id"]:
                    continue
                for slot, o in enumerate(s.get("outputs", [])):
                    if o.get("name") == "face_images":
                        r = self.resolve(inst_key, slot)
                        if r:
                            taps[(self.api_id(r[0]), r[1])] = \
                                f"#{inst_key} subgraph face_images (slot {slot})"
        if not self.oi and "273" in self.nodes:
            taps[("273", 1)] = "#273:1 face_images (documented tap)"
        return taps

    def privacy_audit(self, prompt):
        """HARD RULE (internal runbook): raw real-face crops must never reach a preview/save node.

        Slot-precise. An earlier node-level version flagged any writer that merely
        DEPENDED on the node producing face_images, which is a false positive: that same
        node also emits pose_data, and the face slot legitimately feeds the canonicalizer.
        What actually matters is whether a writer consumes the tainted SLOT, directly or
        through nodes that pass identity along -- and taint dies at a SANITIZER.
        """
        taps = self.raw_face_taps(prompt)

        # propagate taint forward: a node's outputs are tainted if any input is tainted,
        # unless the node is a sanitizer.
        tainted = {t for t in taps}
        for _ in range(len(prompt) + 2):                 # fixpoint; graph is a DAG
            grew = False
            for k, v in prompt.items():
                if prompt[k]["class_type"] in self.SANITIZERS:
                    continue
                if any(isinstance(val, list) and len(val) == 2 and tuple(val) in tainted
                       for val in v["inputs"].values()):
                    for slot in range(8):                # cover every plausible out slot
                        if (k, slot) not in tainted:
                            tainted.add((k, slot)); grew = True
            if not grew:
                break

        violations = []
        for root in self.output_nodes(prompt):
            for name, val in prompt[root]["inputs"].items():
                if isinstance(val, list) and len(val) == 2 and tuple(val) in tainted:
                    src = taps.get(tuple(val), f"tainted via {val[0]}:{val[1]}")
                    violations.append(
                        f"{prompt[root]['class_type']} #{root} "
                        f"('{prompt[root]['_meta']['title']}') input '{name}' carries RAW FACE "
                        f"pixels from {src}")
        return taps, violations


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("workflow")
    ap.add_argument("--object-info")
    ap.add_argument("--structure-only", action="store_true")
    ap.add_argument("--baseline-raw-video", action="store_true",
                    dest="baseline_raw_video",
                    help="ACKNOWLEDGE that this graph sends raw face pixels to a "
                         "writer. ONLY for a deliberate no-privacy BASELINE render "
                         "(vanilla WanAnimate has no canonicaliser). Never for a "
                         "MIRAGE pipeline graph.")
    ap.add_argument("--drop", default="", help="comma-separated node ids to omit (kills their subtree)")
    ap.add_argument("--no-prune", action="store_true")
    ap.add_argument("-o", "--out")
    a = ap.parse_args()

    wf = json.load(open(a.workflow, encoding="utf-8"))
    oi = json.load(open(a.object_info, encoding="utf-8")) if a.object_info else None
    c = Converter(wf, oi)
    prompt = c.to_prompt()

    n_raw = len(prompt)
    for d in [x.strip() for x in a.drop.split(",") if x.strip()]:
        if d in prompt:
            print(f"dropping #{d} {prompt[d]['class_type']} "
                  f"({prompt[d]['_meta']['title']})".encode("ascii", "replace").decode())
            del prompt[d]
        else:
            print(f"!! --drop {d}: not an executable node")
    if not a.no_prune:
        prompt = c.prune(prompt)

    banned, violations = c.privacy_audit(prompt)
    print("\n--- PRIVACY AUDIT (raw face crops must not reach any writer) ---")
    for (nid, slot), why in banned.items():
        print(f"  raw-face tap: {nid}:{slot}  <- {why}")
    print(f"  sanitizers (taint stops here): {sorted(Converter.SANITIZERS)}")
    print(f"  violations: {len(violations)}")
    for v in violations:
        print("  X", v.encode("ascii", "replace").decode())
    if violations and not getattr(a, "baseline_raw_video", False):
        c.errors.append(f"{len(violations)} PRIVACY VIOLATION(S) - refusing to emit")
    elif violations and getattr(a, "baseline_raw_video", False):
        # DELIBERATE, and only for a no-privacy BASELINE reference render. A vanilla
        # WanAnimate graph has no MIRAGEFaceCanonicalizer, so face_images is raw by
        # construction -- that is precisely what the baseline measures against. The flag is
        # explicit and loud so a baseline artifact can never be mistaken for a pipeline
        # output, nor quoted in support of any privacy claim.
        bar = "!" * 78
        print("")
        print(bar)
        print("  BASELINE OVERRIDE: this prompt sends RAW FACE PIXELS to the cloud.")
        print("  %d violation(s) acknowledged and INTENTIONAL for this render." % len(violations))
        print("  Artifacts are a NO-PRIVACY REFERENCE: never use them to support a privacy")
        print("  claim, and always label them as baseline.")
        print(bar)
        print("")


    print(f"\nnodes in file        : {len(wf.get('nodes', []))}")
    print(f"nodes after flatten  : {len(c.nodes)}")
    print(f"executable API nodes : {n_raw} -> {len(prompt)} after drop+prune")
    kept = {k for k in prompt}
    dflt = [d for d in c.defaulted if d.split()[0].lstrip("#") in kept]
    if dflt:
        print(f"\n--- {len(dflt)} WIDGET(S) FILLED FROM /object_info DEFAULTS ---")
        for d in dflt:
            print("  =", d)
    if c.warnings:
        print(f"\n--- {len(c.warnings)} WARNING(S) ---")
        for w in c.warnings:
            print("  !", w)
    if c.errors:
        print(f"\n--- {len(c.errors)} ERROR(S) ---")
        for e in c.errors:
            print("  X", e)

    if a.out and not c.errors:
        clean = {k: {kk: vv for kk, vv in v.items() if kk != "_meta"} | {"_meta": v["_meta"]}
                 for k, v in prompt.items()}
        json.dump(clean, open(a.out, "w", encoding="utf-8"), indent=1)
        print(f"\nwrote {a.out}")
    return 1 if c.errors else 0


if __name__ == "__main__":
    sys.exit(main())
