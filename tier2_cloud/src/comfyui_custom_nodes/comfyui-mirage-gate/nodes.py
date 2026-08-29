"""
MIRAGE Person Gate - dynamic per-person branch skip (V8).

ComfyUI graphs are static, so node `mode` cannot switch on a runtime value. The
ONLY runtime-conditional mechanism is LAZY evaluation: a node may declare inputs
`"lazy": True` and, in `check_lazy_status`, request only the input(s) it actually
needs - ComfyUI then never executes the subtree feeding an un-requested lazy input.

MIRAGEPersonGate is an IMAGE switch built on that:
    person_count >= min_people  -> output `multi`   (the 2-person stream)
    person_count <  min_people  -> output `single`  (the 1-person stream)
and it requests ONLY the selected branch. Wired at the points where Person-B
enters the main graph (character stream, matte stream, and the two Person-B
preview taps), a 1-person detection therefore skips the ENTIRE 2nd-person path - 
the Person-B WanAnimate subgraph AND the Person-B auto-reference image - because
nothing downstream requests them.

Privacy/location: this node moves no pixels off-device and adds no cloud input; it
only *reduces* what the cloud stage generates. It is IMAGE-typed (no wildcard) so
it drops straight into the existing IMAGE streams.

Correctness note: `run()` returns the correct stream regardless, so even if a host
build did not honour lazy skipping, the OUTPUT stays correct - only the compute
saving depends on lazy evaluation (a core ComfyUI feature since the execution-model
rewrite; this pod runs 0.18.2, which supports it).
"""


class MIRAGEPersonGate:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                # detected count (wire from MIRAGESam2VideoMultiBox.person_count)
                "person_count": ("INT", {"default": 2, "min": 1, "max": 8, "forceInput": True}),
                "min_people": ("INT", {"default": 2, "min": 1, "max": 8}),
                # 2-person stream - LAZY: only evaluated (and its whole upstream run)
                # when person_count >= min_people. This is what prunes Person-B.
                "multi": ("IMAGE", {"lazy": True}),
            },
            "optional": {
                # 1-person stream (eager). Wire the P1 stream for the composite gates;
                # leave UNWIRED for preview gates -> a small black image is returned
                # (so a 1-person run shows a blank Person-B preview and never triggers
                # the Person-B AutoRef image).
                "single": ("IMAGE",),
            },
        }

    RETURN_TYPES = ("IMAGE",)
    RETURN_NAMES = ("image",)
    FUNCTION = "run"
    CATEGORY = "MIRAGE/V8"

    def check_lazy_status(self, person_count, min_people, multi=None, single=None):
        # Only request 'multi' when we will actually use it. When we won't, ComfyUI
        # never evaluates it, so the Person-B subgraph / AutoRef upstream is skipped.
        return ["multi"] if int(person_count) >= int(min_people) else []

    def run(self, person_count, min_people, multi=None, single=None):
        if int(person_count) >= int(min_people):
            return (multi,)
        if single is not None:
            return (single,)
        import torch
        return (torch.zeros((1, 64, 64, 3), dtype=torch.float32),)   # blank P2 placeholder


NODE_CLASS_MAPPINGS = {"MIRAGEPersonGate": MIRAGEPersonGate}
NODE_DISPLAY_NAME_MAPPINGS = {
    "MIRAGEPersonGate": "MIRAGE Person Gate (skip 2nd person if 1 detected)"
}

# ---------------------------------------------------------------------------------------------
# PRE-RENAME COMPATIBILITY (2026-08-29). A saved ComfyUI graph stores each node's registry key in
# its `class_type` field, so every workflow exported before the rename says "SITARA...". Loading
# one against a registry that only knows "MIRAGE..." fails with an unhelpful missing-node error.
# Each node is therefore registered a SECOND time under its old key. The alias maps to the SAME
# class object, so the two names cannot drift apart, and new graphs save under the new name.
_LEGACY = {k.replace("MIRAGE", "SITARA", 1): v
           for k, v in NODE_CLASS_MAPPINGS.items() if k.startswith("MIRAGE")}
NODE_CLASS_MAPPINGS.update(_LEGACY)
NODE_DISPLAY_NAME_MAPPINGS.update({
    k: NODE_DISPLAY_NAME_MAPPINGS.get(k.replace("SITARA", "MIRAGE", 1), k) + " (legacy name)"
    for k in _LEGACY
})
