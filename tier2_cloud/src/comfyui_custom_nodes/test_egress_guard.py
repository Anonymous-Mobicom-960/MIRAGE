#!/usr/bin/env python3
"""
test_egress_guard.py - the §2 egress guard, tested against the real function
============================================================================
    python3 test_egress_guard.py

Runs on a laptop with numpy+Pillow only: it imports `mirage_autoref_node` by file path and never
touches ComfyUI, torch, or an API key. Ledger §A.5 / §A.5b cite this file, so those rows have a
re-runnable source instead of a scratchpad script that no longer exists.

WHAT THIS EXISTS TO PREVENT (measured 2026-07-23, both real)
------------------------------------------------------------
The guard's job is to ensure the only images this node uploads to a third-party API are grey
silhouettes. Two ways it failed:

  * DILUTION. The statistics were aggregated over the BATCH - concatenated chroma, and the MEAN
    of per-frame tone mass. A greyscale photographic frame is refused on its own, but hidden
    1-in-8 among masks its tone mass averaged against seven 1.000s to 0.891 and cleared the 0.85
    threshold. Every frame is uploaded individually, so every frame must be judged individually.
    Case `dilution_*` below pins this at several ratios, because a fix that only handled 1-in-8
    would be a fix tuned to the one ratio someone happened to test.

  * A SECOND, UNGUARDED PATH. `MIRAGE_AutoRefPrompt.run` posts the same frames to the VLM and had
    no check at all, while the guarded node's audit comment claimed no other tensor reached an
    API call. `test_all_upload_paths_are_guarded` fails if any node in the file calls `_analyze`
    or `_genimg` without `_egress_check` in the same function - so the next path added is caught
    by the test rather than by an incident.
"""
import ast
import os
import sys
import importlib.util
import types

import numpy as np
from PIL import Image

HERE = os.path.dirname(os.path.abspath(__file__))
NODE = os.path.join(HERE, "mirage_autoref_node.py")


#: Modules the node imports at module level that are absent outside a ComfyUI install. They are
#: STUBBED rather than installed: the guard operates on PIL images and numpy, and the only torch
#: call in the whole node is one `torch.from_numpy` in a function this test never reaches. Pulling
#: a ~200 MB CPU torch wheel into CI to satisfy an unused import would be waste.
_STUB_MODULES = ("torch", "comfy", "folder_paths", "nodes")


def _load():
    """Import the node by file path, with the ComfyUI-only modules stubbed.

    🔴 The stubs are why this test can claim to run on numpy+Pillow alone. Without them the node
    dies on `import torch` at line 47, BEFORE `_egress_check` is defined, and the swallow-and-warn
    below hands back a module missing the very function under test -- which then surfaces as a
    bare AttributeError several frames away instead of "a dependency is missing". That happened:
    the suite passed on a workstation with torch installed and failed in CI without it, for
    reasons the output did not explain.
    """
    saved = {n: sys.modules.get(n) for n in _STUB_MODULES}
    for n in _STUB_MODULES:
        sys.modules.setdefault(n, types.ModuleType(n))
    spec = importlib.util.spec_from_file_location("mirage_autoref_node", NODE)
    mod = importlib.util.module_from_spec(spec)
    try:
        spec.loader.exec_module(mod)
    except Exception as ex:                      # optional ComfyUI-only imports may be absent
        print(f"  (note: module raised {type(ex).__name__} on import; using partial module)")
    finally:
        for n, prev in saved.items():
            if prev is None:
                sys.modules.pop(n, None)
            else:
                sys.modules[n] = prev

    # Fail LOUDLY and specifically rather than letting a partial module reach the assertions.
    for need in ("_egress_check", "_analyze"):
        if not hasattr(mod, need):
            raise SystemExit(
                f"FATAL: {need} is missing from the loaded module. The node did not finish "
                f"importing, so this run would test nothing. Add whatever it failed on to "
                f"_STUB_MODULES, or install it.")
    return mod


def mask_frame(seed=0):
    """What MaskToImage(person_mask) produces: R==G==B, essentially two-valued."""
    a = np.zeros((256, 256), np.uint8)
    a[60:220, 90:170] = 200
    return Image.fromarray(np.stack([a] * 3, -1))


def grey_photo(rng):
    """A REAL camera frame, desaturated. chroma == 0, so only the tone test can catch it."""
    y, x = np.mgrid[0:256, 0:256]
    g = (120 + 40 * np.sin(x / 9.0) + 50 * np.cos(y / 13.0)
         + rng.normal(0, 18, (256, 256))).clip(0, 255).astype(np.uint8)
    return Image.fromarray(np.stack([g] * 3, -1))


def colour_photo(rng):
    x = rng.integers(0, 255, (256, 256, 3), dtype=np.uint8)
    return Image.fromarray(np.clip(x * 0.6 + 80, 0, 255).astype(np.uint8))


def main():
    m = _load()
    rng = np.random.default_rng(0)
    cases = [("masks only (MUST PASS)", [mask_frame() for _ in range(8)], True),
             ("single grey photo", [grey_photo(rng)], False),
             ("single colour photo", [colour_photo(rng)], False)]
    # Dilution at several ratios: no batch size may launder one bad frame.
    for k in (8, 16, 32, 64):
        cases.append((f"dilution_{k - 1}masks+1grey", [mask_frame() for _ in range(k - 1)]
                      + [grey_photo(rng)], False))
    cases.append(("dilution_7masks+1colour",
                  [mask_frame() for _ in range(7)] + [colour_photo(rng)], False))

    ok = True
    print("§2 EGRESS GUARD - strict mode")
    for name, pils, want_pass in cases:
        try:
            m._egress_check(pils, strict=True)
            got = True
        except RuntimeError:
            got = False
        good = (got == want_pass)
        ok &= good
        print(f"  [{'OK  ' if good else 'FAIL'}] {name:28} -> "
              f"{'passed' if got else 'REFUSED'}  (want {'pass' if want_pass else 'refuse'})")

    print("\nEvery upload path calls the guard")
    ok &= _all_upload_paths_guarded()
    print("\nALL TESTS PASS" if ok else "\nFAILURES ABOVE")
    return 0 if ok else 1


def _all_upload_paths_guarded():
    """Static check: no function may reach _analyze/_genimg without _egress_check.

    This is the finding that a call-site audit missed - it enumerated paths inside ONE node and
    concluded something about the file. Checking every function body instead means a newly added
    node cannot quietly reopen the hole.
    """
    tree = ast.parse(open(NODE, encoding="utf-8").read())
    uploads = {"_analyze", "_analyze_attrs", "_gemini_image", "_openai_image"}
    ok = True
    for fn in [n for n in ast.walk(tree) if isinstance(n, (ast.FunctionDef, ast.AsyncFunctionDef))]:
        called = {c.func.id for c in ast.walk(fn)
                  if isinstance(c, ast.Call) and isinstance(c.func, ast.Name)}
        # lambdas bound to an upload helper count as reaching it
        hits = (called & uploads)
        if not hits:
            continue
        if "_egress_check" not in called:
            print(f"  [FAIL] {fn.name}() calls {sorted(hits)} with NO _egress_check")
            ok = False
        else:
            print(f"  [OK  ] {fn.name}() calls {sorted(hits)} behind _egress_check")
    return ok


if __name__ == "__main__":
    sys.exit(main())
