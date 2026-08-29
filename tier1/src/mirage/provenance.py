"""ONE description of what actually ran - the single source of truth for a defended run.

WHY THIS FILE EXISTS
--------------------
On the MIRAGE host, a rendered artifact once declared ``gait_preset: ""`` while carrying `e2`
knobs, because two places computed "the config" independently and disagreed (ledger §A.2d, and
again in RUN3 / §B.62 where ``GAIT_PRESETS["g18"]`` was not what had been approved). The fix is
structural, not documentary: ``resolve_config()`` is computed ONCE per run and the result is
passed around. A second, independent computation is the bug this module exists to make
impossible, so a recompute that disagrees with the active config RAISES.

WHAT IT IS NOT
--------------
It is not a privacy claim. Every number quoted in the comments here was measured on the MIRAGE
host (`tier1/src/edge_runner_pi5/`), NOT on this one. A provenance block names
the code and the knobs; it never asserts what they achieved here.

SEEDS
-----
The per-clip gait and mask seeds are drawn from the OS CSPRNG per sequence and MUST NEVER be
emitted (`pose_anon_edge.new_clip_seed`: "used in-process only and must never be emitted/logged"
 - a stable, recorded seed re-creates the linkable pseudo-identity the anonymiser exists to
avoid). The only seed this module will ever carry is ``MIRAGE_TEST_FIXED_SEED``, and only when
``test_artifact`` is True - which is precisely the flag that says the output is a TEST ARTIFACT
and is not privacy-safe to ship. ``_assert_no_seed_leak()`` enforces that on every call.

Vendored-code resolution and hashing also live here, so "which config" and "which code" are
answered by one module. ``mask_anon.py`` imports :func:`vendor_module` from here rather than
duplicating the resolver - one resolver, one answer.
"""

from __future__ import annotations

import hashlib
import importlib
import inspect
import json
import os
import sys
import types
from typing import Any, Dict, Iterable, Optional

__all__ = [
    "resolve_config",
    "active",
    "reset_active",
    "vendor_module",
    "vendor_path",
    "vendor_sha256",
    "SCHEMA",
]

SCHEMA = "mirage_defence_provenance/1"

# ---------------------------------------------------------------------------------------------
# Vendored-code resolution
# ---------------------------------------------------------------------------------------------
# The vendored MIRAGE files are BYTE-IDENTICAL copies (owner decision: never edit a vendored
# file; all host-specific logic lives in adapters like this one). They live in the package at
# ``mirage/vendor/mirage_edge/``.
#
# ORDER MATTERS AND IS DELIBERATE: the in-package module wins, ALWAYS. ``MIRAGE_VENDOR_DIR`` is
# a development/verification fallback used only when the package copy is absent - an env var
# must never be able to silently swap the defence code out from under a shipped run. Whichever
# source was used is recorded in the provenance block, so an artifact says which it was.
_VENDOR_PKG = "vendor.mirage_edge"                 # relative to this package
_VENDOR_ENV = "MIRAGE_VENDOR_DIR"
_VENDOR_FILES = ("pose_anon_edge.py", "person_slots.py", "mask_shape.py")

_MODCACHE: Dict[str, types.ModuleType] = {}
_SRCCACHE: Dict[str, str] = {}                # module name -> "package" | "MIRAGE_VENDOR_DIR"


def _module_dir(mod: types.ModuleType) -> Optional[str]:
    f = getattr(mod, "__file__", None)
    return os.path.normcase(os.path.dirname(os.path.abspath(f))) if f else None


def vendor_module(name: str) -> types.ModuleType:
    """Import a vendored MIRAGE module by bare name (e.g. ``"mask_shape"``).

    Raises ImportError with an actionable message if neither source has it - never falls back
    to a re-implementation, because a silent re-implementation is how a "vendored byte-identical"
    claim stops being true.

    🔴 THE BARE NAME IS LOAD-BEARING - DO NOT "TIDY" IT INTO A DOTTED IMPORT.
    ``vendor/mirage_edge/__init__.py`` puts VENDOR_DIR at the FRONT of ``sys.path`` and imports its
    modules under their BARE top-level names, because ``pose_anon_edge.py:39`` is a top-level
    absolute import of its sibling. Importing ``mirage.vendor.mirage_edge.pose_anon_edge``
    instead EXECUTES THE FILE A SECOND TIME and creates a SECOND module object - and importlib
    then rebinds the parent package's attribute to that second copy, while
    ``mirage.anonymize_pose_log`` (bound by ``from pose_anon_edge import ...`` at package-init
    time) still belongs to the first. MEASURED on this host before this was fixed: the vendored
    call appended ``'anatomical_extent'`` to the FIRST copy's ``_TEMPLATE_KIND_USED`` while
    ``gait_anon`` read the SECOND copy's, so ``manifest.json -> anon.runtime.template_kinds``
    came back ``[]`` on every real run - silently losing the one signal that distinguishes a
    real anatomical collapse from the legacy torso-scaled fallback (VENDOR.md, "Provenance
    globals"). The tests never saw it because they import ``gait_anon`` first; the PIPELINE
    imports ``provenance`` first, which is why only a real run was affected.

    So: resolve through the package (which owns the ``sys.path`` insert), then import by the
    BARE name - ``sys.modules`` already holds it - and verify the file really came from
    VENDOR_DIR before trusting it.
    """
    if name in _MODCACHE:
        return _MODCACHE[name]
    err_pkg = None
    try:
        pkg = importlib.import_module("." + _VENDOR_PKG, __package__)
        vdir = os.path.normcase(os.path.abspath(getattr(pkg, "VENDOR_DIR", "") or ""))
        mod = importlib.import_module(name)        # BARE name -> exactly one module object
        if not vdir or _module_dir(mod) != vdir:
            raise ImportError(
                f"top-level module {name!r} resolved to {getattr(mod, '__file__', None)!r}, "
                f"which is not the vendored copy in {vdir!r}; something else on sys.path is "
                f"shadowing it")
        if getattr(pkg, name, None) is not mod:
            # Repair a parent-attribute rebind left behind by someone else's dotted import, so
            # `mirage.pose_anon_edge` and `mirage.anonymize_pose_log` describe ONE module again.
            setattr(pkg, name, mod)
        _MODCACHE[name], _SRCCACHE[name] = mod, "package"
        return mod
    except ImportError as e:                       # noqa: PERF203 - two distinct sources
        err_pkg = e
    vdir = os.environ.get(_VENDOR_ENV, "").strip()
    if vdir:
        if vdir not in sys.path:
            sys.path.insert(0, vdir)
        mod = importlib.import_module(name)
        _MODCACHE[name], _SRCCACHE[name] = mod, _VENDOR_ENV
        return mod
    raise ImportError(
        f"vendored MIRAGE module {name!r} not found. Expected "
        f"mirage/{_VENDOR_PKG.replace('.', '/')}/{name}.py (byte-identical copy of "
        f"tier1/src/edge_runner_pi5/{name}.py), or set {_VENDOR_ENV} to a "
        f"directory containing it. Underlying error: {err_pkg}")


def vendor_source(name: str) -> Optional[str]:
    """``"package"`` / ``"MIRAGE_VENDOR_DIR"`` for an already-imported vendored module."""
    return _SRCCACHE.get(name)


def _vendor_dirs() -> Iterable[str]:
    here = os.path.dirname(os.path.abspath(__file__))
    yield os.path.join(here, *_VENDOR_PKG.split("."))
    vdir = os.environ.get(_VENDOR_ENV, "").strip()
    if vdir:
        yield vdir


def vendor_path(filename: str) -> Optional[str]:
    """Absolute path of a vendored file, package copy first. None if it is not there."""
    for d in _vendor_dirs():
        p = os.path.join(d, filename)
        if os.path.isfile(p):
            return p
    return None


def vendor_sha256(filename: str) -> Dict[str, Any]:
    """``{status, sha256, bytes, path}`` for one vendored file.

    A MISSING file is reported as ``status="MISSING"`` rather than omitted: an artifact that
    cannot name the code that produced it must SAY so, loudly, not quietly leave the field out.
    """
    p = vendor_path(filename)
    if p is None:
        return {"status": "MISSING", "sha256": None, "bytes": None, "path": None}
    with open(p, "rb") as fh:
        blob = fh.read()
    return {"status": "ok", "sha256": hashlib.sha256(blob).hexdigest(),
            "bytes": len(blob), "path": os.path.abspath(p)}


# ---------------------------------------------------------------------------------------------
# Seconds -> frames
# ---------------------------------------------------------------------------------------------
# Every ``*_s`` kwarg of ``anonymize_v2`` is a DURATION, and its frame-domain magnitude depends
# on the fps the transform is told about. Two very different things share that suffix, and
# conflating them is how a run gets described wrongly:
#
#   BOUND   the code draws uniformly INSIDE the value; the applied shift is never the value
#           itself. ``limb_phase_offset_s=0.35`` -> each limb chain draws U[-0.35, +0.35] s
#           (pose_anon_edge._limb_phase_offset: ``r.uniform(-max_s, max_s) * fps``), so at
#           10 fps the applied shift is somewhere in +/-3.5 frames, NOT 3.5 frames.
#   PERIOD  an applied oscillation period, used as-is.
#
# The map is structural (it names which amplitude gates which duration, read off the call sites
# in ``anonymize_v2``), not a fitted constant, so it does not violate the no-clip-constants rule.
_SECONDS_SEMANTICS = {
    "limb_phase_offset_s":  ("bound", "limb_phase_offset_s",
                             "per-chain constant time shift, drawn U[-v, +v] s per SEQUENCE"),
    "pose_smooth_max_s":    ("bound", "pose_smooth_max_s",
                             "temporal low-pass sigma, drawn U[0, v] s per SEQUENCE"),
    "cadence_period_s":     ("period", "cadence_amp", "gait-band period of the cadence warp"),
    "limb_phase_period_s":  ("period", "limb_phase_amp", "period of the per-limb phase warp"),
    "lowfreq_period_s":     ("period", "lowfreq_amp_frac", "period of the positional drift"),
    "root_speed_period_s":  ("period", "root_speed_amp", "period of the walking-speed modulation"),
}


def _seconds_block(effective: Dict[str, Any], fps: float) -> Dict[str, Any]:
    out: Dict[str, Any] = {}
    for k, v in sorted(effective.items()):
        if not k.endswith("_s") or not isinstance(v, (int, float)):
            continue
        kind, gate, sem = _SECONDS_SEMANTICS.get(
            k, ("unknown", None, "NOT CLASSIFIED - a new seconds-knob was added to "
                                 "anonymize_v2; classify it in _SECONDS_SEMANTICS"))
        gate_val = effective.get(gate) if gate else None
        entry = {
            "seconds": float(v),
            "frames": float(v) * float(fps),
            "kind": kind,                      # "bound" | "period" | "unknown"
            "semantics": sem,
            "gated_by": gate,
            "gate_value": gate_val,
            # A knob whose gating amplitude is 0.0 does nothing at all this run.
            "active": (None if kind == "unknown"
                       else bool(v) and bool(gate_val) if gate else bool(v)),
        }
        if kind == "bound":
            entry["applied_frames"] = f"drawn within +/-{float(v) * float(fps):.4g}" \
                if k == "limb_phase_offset_s" else \
                f"drawn within [0, {float(v) * float(fps):.4g}]"
        out[k] = entry
    return out


# ---------------------------------------------------------------------------------------------
# Wrapper knobs
# ---------------------------------------------------------------------------------------------
# Keys a preset may carry that are NOT ``anonymize_v2`` kwargs: ``anonymize_pose_log`` POPS them
# and uses them one level up, on the collapse TEMPLATE (pose_anon_edge.py:2188+). Reproduced here
# in the same order as the pops so the merged kwargs are exactly the dict that reaches the call.
# ``proximity_cap_frac`` is deliberately NOT here: the source ``get``s it (does not pop), and it
# IS an ``anonymize_v2`` kwarg, so it stays in the merged dict.
_POPPED_WRAPPER_KEYS = ("scale_from", "height_mult", "arm_mult", "leg_mult",
                        "neck_mult", "stature_ratio", "height_anchor")

# Env vars that override a wrapper knob at the same call site, recorded so an artifact shows the
# EFFECTIVE value rather than the preset's.
_WRAPPER_ENV = {
    "scale_from": "MIRAGE_POSE_SCALE_FROM",
    "height_mult": "MIRAGE_HEIGHT_MULT",
    "arm_mult": "MIRAGE_ARM_MULT",
}

# MIRAGE_* env vars that change the gait/mask transform. Recorded (name -> value) when set, so a
# run whose numbers came from an env override is self-describing.
_TRACKED_ENV = (
    "MIRAGE_GAIT_PRESET", "MIRAGE_POSE_SCALE_FROM", "MIRAGE_HEIGHT_MULT", "MIRAGE_ARM_MULT",
    "MIRAGE_ANGLE_CONST", "MIRAGE_ANGLE_DRIFT", "MIRAGE_LOWFREQ_AMP", "MIRAGE_CADENCE_AMP",
    "MIRAGE_CADENCE_PERIOD_S", "MIRAGE_ASYM_SIGMA", "MIRAGE_LR_ASYM_SIGMA",
    "MIRAGE_LIMB_PHASE_AMP", "MIRAGE_LIMB_PHASE_PERIOD_S", "MIRAGE_HEAD_ANCHOR",
    "MIRAGE_ANGLE_MIRROR", "MIRAGE_MASK_SHAPE_MODE", "MIRAGE_MASK_TEMPORAL_WIN",
    "MIRAGE_MASK_BBOX_MERGE", "MIRAGE_MASK_BBOX_PAD", "MIRAGE_EMIT_FPS",
    "MIRAGE_TEST_FIXED_SEED",
)

# Keys whose VALUE is a seed. ``seeded_global_scale_max`` contains the substring "seed" and is
# NOT one of these - it is an amplitude - which is exactly why the check is an exact-name
# allowlist and not a substring match.
_SEED_KEYS = frozenset({"seed", "clip_seed", "gait_seed", "mask_seed", "_MASK_DISPLACE_SEED",
                        "test_fixed_seed", "fixed_seed"})


def _jsonable(v: Any) -> Any:
    if isinstance(v, dict):
        return {str(k): _jsonable(x) for k, x in v.items()}
    if isinstance(v, (list, tuple, set, frozenset)):
        return [_jsonable(x) for x in v]
    if isinstance(v, (str, int, float, bool)) or v is None:
        return v
    return repr(v)


def _assert_no_seed_leak(cfg: Dict[str, Any]) -> None:
    """Hard gate: a non-test provenance block carries NO seed value, anywhere, at any depth."""
    if cfg.get("test_artifact"):
        return

    def walk(node, path=""):
        if isinstance(node, dict):
            for k, v in node.items():
                if k in _SEED_KEYS and v is not None:
                    raise AssertionError(
                        f"provenance would leak a seed at {path}/{k} = {v!r}. A per-clip seed is "
                        "in-process only (pose_anon_edge.new_clip_seed); recording it re-creates "
                        "a linkable pseudo-identity.")
                walk(v, f"{path}/{k}")
        elif isinstance(node, list):
            for i, v in enumerate(node):
                walk(v, f"{path}[{i}]")

    walk(cfg)


def _test_fixed_seed():
    """The pinned TEST seed, or None. Delegates to the vendored reader when it is importable so
    there is one parse of ``MIRAGE_TEST_FIXED_SEED``; falls back to the same rule verbatim."""
    try:
        return vendor_module("pose_anon_edge").test_fixed_seed()
    except Exception:
        v = os.environ.get("MIRAGE_TEST_FIXED_SEED")
        if v is None or str(v).strip() == "":
            return None
        try:
            return int(v) & 0x7FFFFFFF
        except ValueError:
            return None


def _digest(cfg: Dict[str, Any]) -> str:
    return hashlib.sha256(
        json.dumps(cfg, sort_keys=True, separators=(",", ":")).encode("utf-8")).hexdigest()


# ---------------------------------------------------------------------------------------------
# The one call
# ---------------------------------------------------------------------------------------------
_ACTIVE: Optional[Dict[str, Any]] = None


def resolve_config(gait_preset_name: Optional[str] = None,
                   level: str = "L4",
                   mask_shape_mode: str = "bbox",
                   temporal_win: int = 2,
                   fps: float = 10.0,
                   *,
                   gait_enabled: bool = False,
                   mask_enabled: bool = False,
                   mask_fps: Optional[float] = None,
                   edge_expand_px: Optional[int] = None,
                   simplify_eps: Optional[float] = None,
                   bbox_merge: Optional[bool] = None,
                   bbox_pad_frac: Optional[float] = None,
                   mask_route: str = "segmentation backend + MIRAGE bbox",
                   binarize_thresh: float = 0.5,
                   export_people: Optional[int] = None,
                   deterministic_seed_pinned: bool = False,
                   extra: Optional[Dict[str, Any]] = None,
                   allow_recompute: bool = False) -> Dict[str, Any]:
    """Resolve EVERYTHING about a defended run into one JSON-safe dict, and remember it.

    Parameters mirror the CLI: ``gait_preset_name`` (None => the vendored ``SHIPPED_PRESET``,
    ``""`` => bare LEVELS with no preset), ``level`` the LEVELS amplitude ladder, and the mask
    knobs. ``gait_enabled`` / ``mask_enabled`` default False because both defences default OFF
    at the function level; the CLI turns them on.

    Called twice with a DIFFERENT result, this RAISES: a second independent computation is how
    an artifact ended up declaring one config while carrying another. Pass the returned dict
    around instead. ``allow_recompute=True`` (or :func:`reset_active`) is the deliberate escape
    hatch for a process that really does run two clips at two configs.
    """
    global _ACTIVE
    PA = vendor_module("pose_anon_edge")
    warnings = []

    # ---- gait -------------------------------------------------------------------------------
    shipped = getattr(PA, "SHIPPED_PRESET", "")
    preset_name = shipped if gait_preset_name is None else str(gait_preset_name).strip()
    preset = PA.gait_preset(preset_name)          # raises on a typo - never silently defaults
    preset_n_keys = len(preset)

    if level not in PA.LEVELS:
        raise ValueError(f"level {level!r} is not one of {sorted(PA.LEVELS)}")
    levels_knobs = dict(PA.LEVELS[level])         # already carries any import-time env override

    wrapper: Dict[str, Any] = {}
    for k in _POPPED_WRAPPER_KEYS:
        if k in preset:
            wrapper[k] = preset.pop(k)
    # Same precedence as the call site: an explicit env override beats the preset.
    for k, env in _WRAPPER_ENV.items():
        ev = os.environ.get(env)
        if ev is not None and str(ev).strip():
            wrapper[k] = float(ev) if k.endswith("_mult") else ev.strip()

    merged = dict(levels_knobs, **preset)         # == the dict(knobs, **preset) at the call site

    # The merged dict must be exactly what ``anonymize_v2`` accepts. Validated against the live
    # signature rather than a hardcoded list, so a preset that grows a new wrapper-only key fails
    # HERE (loudly, before a render) instead of at the call.
    sig = inspect.signature(PA.anonymize_v2)
    accepted = set(sig.parameters)
    unknown = sorted(set(merged) - accepted)
    if unknown:
        raise ValueError(
            f"preset {preset_name!r} carries {unknown} which are NOT anonymize_v2 kwargs and are "
            f"not in _POPPED_WRAPPER_KEYS. Classify them before running.")

    # Effective values = anonymize_v2's own defaults, overridden by what we pass. Needed so a
    # period whose amplitude is on is reported even when the preset never mentions the period.
    effective = {n: p.default for n, p in sig.parameters.items()
                 if p.default is not inspect.Parameter.empty}
    effective.update(merged)

    # Fixed kwargs the call site adds itself (pose_anon_edge.py:2313). NOT including ``seed``
    # (per-clip CSPRNG) or ``proximity_scores`` (per-clip data).
    call_site = {"do_canon": True, "center_root": False, "fps": float(fps),
                 "head_anchor": os.environ.get("MIRAGE_HEAD_ANCHOR", "off")}

    gait = {
        "enabled": bool(gait_enabled),
        "preset": preset_name,
        "preset_is_shipped_default": preset_name == shipped,
        "shipped_preset": shipped,
        "preset_keys_raw": preset_n_keys,
        "level": level,
        "fps": float(fps),
        "anonymize_v2_kwargs": _jsonable(merged),
        "anonymize_v2_kwargs_n": len(merged),
        "wrapper_knobs": _jsonable(wrapper),
        "call_site_kwargs": _jsonable(call_site),
        "seconds_knobs_at_fps": _seconds_block(effective, float(fps)),
        "levels_source": f"vendored LEVELS[{level}] after import-time env overrides",
        # The privacy-rejected knob (§A.2m-2c: +13.3 pp adaptively). Recorded because it is
        # env-selected and therefore invisible in the preset.
        "head_anchor_on": call_site["head_anchor"] not in ("off", "", None),
    }
    if gait["head_anchor_on"]:
        warnings.append("MIRAGE_HEAD_ANCHOR is ON - rejected on privacy on the MIRAGE host "
                        "(§A.2m-2c). This run is not the shipped arm.")

    # ---- mask -------------------------------------------------------------------------------
    # Defaults come from the vendored config shim when it exposes them, otherwise the documented
    # MIRAGE values (config.py:95 EDGE_EXPAND=1, :829 MASK_SIMPLIFY_EPS=0.01, :558 BBOX_MERGE,
    # :562 BBOX_PAD_FRAC). Never invented here.
    C = getattr(vendor_module("mask_shape"), "C", None) if _have("mask_shape") else None
    eep = int(_first(edge_expand_px, getattr(C, "EDGE_EXPAND", None), 1))
    eps = float(_first(simplify_eps, getattr(C, "MASK_SIMPLIFY_EPS", None), 0.01))
    mrg = bool(_first(bbox_merge, getattr(C, "MASK_BBOX_MERGE", None), True))
    pad = float(_first(bbox_pad_frac, getattr(C, "MASK_BBOX_PAD_FRAC", None), 0.0))
    mfps = float(_first(mask_fps, fps))
    win = int(temporal_win)

    mask = {
        "enabled": bool(mask_enabled),
        "shape_mode": str(mask_shape_mode),
        "temporal_win": win,
        # The window is a FRAME COUNT on the MIRAGE host but derived from a DURATION, so the same
        # config at a different fps is a different defence (§A.6 / config.py:512). Both stated.
        "mask_fps": mfps,
        "temporal_win_s": (win / mfps) if mfps else None,
        "edge_expand_px": eep,
        "edge_expand_kernel": f"MORPH_ELLIPSE {2 * eep + 1}x{2 * eep + 1}" if eep else "none",
        "simplify_eps": eps,
        "bbox_merge": mrg,
        "bbox_pad_frac": pad,
        "route": mask_route,
        "order": ["binarize", "edge_expand_dilate", "ring_buffer_append", "mask_mitigate"],
        "measured_arm_note": "§A.6o measured `bbox` THROUGH mask_mitigate at temporal win=2 on "
                             "the MIRAGE host. The rectangle alone is not the measured arm, and "
                             "no §A.6 number describes THIS host.",
    }
    if mask["enabled"] and win != 2:
        warnings.append(f"mask temporal_win={win} is not the measured 2 (§A.6o).")

    # ---- test artifact ----------------------------------------------------------------------
    # TWO ways to be a test artifact, and reading only the first one made this module lie.
    #
    #   (a) MIRAGE_TEST_FIXED_SEED is set in the AMBIENT environment when the config is
    #       resolved -- what `_test_fixed_seed()` sees.
    #   (b) the CALLER will pin a seed later, for the duration of each transform call.
    #
    # (b) is how `pipeline.py`'s `gait_pin_run_seed` works: `gait_anon._PinnedEnv` sets
    # MIRAGE_TEST_FIXED_SEED around the vendored call and restores it after, and resolve_config
    # runs long before the first flush. So the env is CLEAN here and (a) is False -- MEASURED:
    # a run with gait_pin_run_seed=True wrote a TIER1_CONFIG.json whose top-level PRIVACY said
    # "TEST ARTIFACT - DO NOT SHIP" while `config.test_artifact` inside said `false` and
    # `config.warnings` was EMPTY. One artifact, two answers to "is this shippable", and the
    # machine-readable half said the wrong one. That is §A.2d in the file whose whole job is to
    # prevent §A.2d, so the caller now declares its intent as part of the ONE resolution and it
    # lands in the digest.
    #
    # Only the FLAG crosses this boundary, never a seed value: `test_fixed_seed` is still written
    # only for case (a), where the value is already in the environment for anyone to read.
    pinned = _test_fixed_seed()
    test_artifact = (pinned is not None) or bool(deterministic_seed_pinned)
    if pinned is not None:
        warnings.append("MIRAGE_TEST_FIXED_SEED is set: gait and mask perturbations are "
                        "DETERMINISTIC. This output is a TEST ARTIFACT and is NOT privacy-safe "
                        "to ship.")
    if deterministic_seed_pinned:
        warnings.append("the caller pins a fixed gait seed for the whole run "
                        "(deterministic_seed_pinned=True): the perturbation is deterministic AND "
                        "SHARED by every tracklet in the clip, which is a linkable "
                        "pseudo-identity. This output is a TEST ARTIFACT and is NOT privacy-safe "
                        "to ship.")

    # ---- vendored code ----------------------------------------------------------------------
    vend = {f: vendor_sha256(f) for f in _VENDOR_FILES}
    vend["_source"] = {n: s for n, s in _SRCCACHE.items()}
    vend["_complete"] = all(v.get("status") == "ok"
                            for k, v in vend.items() if not k.startswith("_"))
    if not vend["_complete"]:
        missing = [k for k, v in vend.items()
                   if not k.startswith("_") and v.get("status") != "ok"]
        warnings.append(f"vendored file(s) not found: {missing} - this artifact cannot fully "
                        f"name the code that produced it.")
    if any(s == _VENDOR_ENV for s in _SRCCACHE.values()):
        warnings.append(f"one or more vendored modules were loaded via {_VENDOR_ENV}, not from "
                        f"the package. Development/verification path, not a shipping path.")

    cfg: Dict[str, Any] = {
        "schema": SCHEMA,
        "host": "mirage tier1 (groupmate clone) - NOT the MIRAGE edge host",
        "gait": gait,
        "mask": mask,
        "export": {
            "binarize_thresh": float(binarize_thresh),   # owner decision: 0.5 == MIRAGE POSE_THRESH
            "export_people": export_people,
            "dense_export_forced": True if export_people is not None else None,
        },
        "test_artifact": bool(test_artifact),
        "test_artifact_cause": ([c for c, on in
                                 (("MIRAGE_TEST_FIXED_SEED_in_env", pinned is not None),
                                  ("caller_pins_run_seed", bool(deterministic_seed_pinned)))
                                 if on] or None),
        "env_overrides": {k: os.environ[k] for k in _TRACKED_ENV if os.environ.get(k)},
        "vendor": vend,
        "warnings": warnings,
    }
    if test_artifact:
        cfg["test_fixed_seed"] = int(pinned)
    if extra:
        cfg["extra"] = _jsonable(extra)

    _assert_no_seed_leak(cfg)
    cfg["digest"] = _digest(cfg)

    if _ACTIVE is not None and _ACTIVE.get("digest") != cfg["digest"] and not allow_recompute:
        raise RuntimeError(
            "resolve_config() was called a second time with a DIFFERENT result "
            f"({_ACTIVE.get('digest')[:12]} -> {cfg['digest'][:12]}). Compute the config ONCE "
            "and pass it around; a second independent computation is exactly how an artifact "
            "declared gait_preset:\"\" while carrying e2 knobs (§A.2d). Use "
            "allow_recompute=True or reset_active() if two configs really are intended.")
    _ACTIVE = cfg
    return cfg


def active() -> Dict[str, Any]:
    """The config resolved for this process. Raises if nothing has been resolved yet."""
    if _ACTIVE is None:
        raise RuntimeError("no active config: call resolve_config() once at startup.")
    return _ACTIVE


def reset_active() -> None:
    """Forget the active config (tests, or a process deliberately running two configs)."""
    global _ACTIVE
    _ACTIVE = None


def _have(name: str) -> bool:
    try:
        vendor_module(name)
        return True
    except ImportError:
        return False


def _first(*vals):
    for v in vals:
        if v is not None:
            return v
    return None
