#!/usr/bin/env python3
"""
test_config_contract.py - every knob the runner reads must actually EXIST in config.py
======================================================================================
    python3 test_config_contract.py

WHY THIS EXISTS (a real break, 2026-07-23)
------------------------------------------
An edit to config.py swallowed a newline, turning

    TRACK_TTL = _frames(TRACK_TTL_S)  # 8 at EMIT_FPS=15
    COVER_MIN = 0.40

into a single line where `COVER_MIN = 0.40` sat INSIDE the trailing comment. `config.COVER_MIN`
stopped existing. Nothing failed: the runner reads it as `getattr(C, "COVER_MIN", 0.40)`, and the
fallback is the same number, so every test passed and every render was byte-identical. The break
surfaced only because an unrelated script happened to read `C.COVER_MIN` directly.

That is the dangerous shape: a defensive `getattr` default silently becomes the ACTUAL value, so
the config file stops being the source of truth and edits to it do nothing. On a privacy knob
(`COVER_MIN` governs the §2 box fill) that means someone can tighten the setting, see no error,
and ship the old behaviour.

So this test asserts the CONTRACT rather than the values: every name the runner reads via getattr
must be defined in config.py. It deliberately does not check what the values are - that is policy,
and policy belongs in the ledger, not in an assertion.
"""
import ast
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import config as C                                                    # noqa: E402

SOURCES = ["mirage_tier1.py", "person_slots.py", "pose_anon_edge.py"]

# Names read via getattr that are genuinely OPTIONAL - absent by design on some builds.
OPTIONAL = {
    "ORT_PROVIDERS",        # set by callers/harnesses, not shipped in config
    "OUT_RES",              # documented as possibly-None
    "GUARANTEE_MARGIN",     # documented as possibly-None
    "SENSITIVITY_LEGACY_P1P99",
    "DP_EPS_PER_CH",
    "CH_BOUNDS",
    "MASK_ANON_ON",
    "WINDOW_AMPLIFICATION",
}


def getattr_names(path):
    """-> {name: default_repr} for every getattr(C, "NAME", default) in the file."""
    out = {}
    tree = ast.parse(open(path, encoding="utf-8").read())
    for n in ast.walk(tree):
        if not (isinstance(n, ast.Call) and isinstance(n.func, ast.Name)
                and n.func.id == "getattr" and len(n.args) >= 2):
            continue
        obj, key = n.args[0], n.args[1]
        if not (isinstance(obj, ast.Name) and obj.id in ("C", "cfg", "config")):
            continue
        if isinstance(key, ast.Constant) and isinstance(key.value, str):
            default = ast.unparse(n.args[2]) if len(n.args) > 2 else "<no default>"
            out[key.value] = default
    return out


def main():
    missing, checked = [], 0
    for src in SOURCES:
        p = os.path.join(HERE, src)
        if not os.path.exists(p):
            continue
        for name, default in sorted(getattr_names(p).items()):
            if name in OPTIONAL:
                continue
            checked += 1
            if not hasattr(C, name):
                missing.append((src, name, default))

    print(f"config contract: {checked} required knob(s) read via getattr across {len(SOURCES)} files")
    if missing:
        print("\nMISSING FROM config.py - the getattr default is silently acting as the real value:")
        for src, name, default in missing:
            print(f"  {src}: C.{name}  (falling back to {default})")
        print("\nThis is the failure mode that hides: no error, no test failure, and edits to "
              "config.py have NO EFFECT on the shipped behaviour.")
        raise SystemExit(1)
    print("ALL TESTS PASS")


if __name__ == "__main__":
    main()
