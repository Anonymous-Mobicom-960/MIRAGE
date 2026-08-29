"""Code vendored from MIRAGE (tier1/src/edge_runner_pi5/), byte-identical.

  mask_shape.py -- silhouette shape canonicalisation (`_shape_polys`) and the temporal
                   running-max mitigation (`mask_mitigate`), lifted from mirage_tier1.py
                   lines 789-1520 @ internal source mirror c6d82dd.

  pose_anon_edge.py -- the gait defence: whole-clip pose anonymiser (`anonymize_pose_log`,
                   `gait_preset`, shipped preset `e2`), WHOLE FILE @ internal source mirror c6d82dd.
  person_slots.py -- `slot_groups` / `estimate_person_count` / `SlotTracker`; a sibling
                   `pose_anon_edge.py` imports by bare name, WHOLE FILE @ same commit.

Do not edit these files. See ../__init__.py for the rule. Per-file source paths, commits
and sha256 digests are in VENDOR.md beside this module.

WHY THIS FILE CONTAINS CODE AT ALL
----------------------------------
`pose_anon_edge.py` line 39 is a TOP-LEVEL absolute import of its sibling::

    from person_slots import slot_groups   # sibling edge module

because on the Pi both files sit loose in one directory that is the process cwd. Rewriting
it to a relative import would break byte-identity, so this package instead puts its OWN
directory at the FRONT of `sys.path` before importing. Front insertion is deliberate: it
guarantees our `person_slots` wins over any same-named module elsewhere on the path. The
insert is idempotent, and only the module names present in this directory are exposed.

One consequence worth knowing: `pose_anon_edge` also lands in `sys.modules` under its bare
top-level name -- the same name MIRAGE's own `mirage_tier1.py` looks it up by -- so there is
exactly ONE module object in the process and the provenance lists below are shared with any
other importer.

PROVENANCE GLOBALS a host may read (off the re-exported `pose_anon_edge` module, never
copied out -- they are appended to, one entry per transformed slot):

  * `pose_anon_edge._TEMPLATE_KIND_USED`  list[str]   -- which collapse target each slot used
                                                        (declared :1146, appended :2301
                                                        UNCONDITIONALLY, once per slot)
  * `pose_anon_edge._STATURE_SCALE_USED`  list[float] -- the per-clip auto size scale applied
                                                        (declared :1147, appended :2335 ONLY
                                                        when `stature_ratio` is set, :2323)

Both are process-global and append-only; the vendored code never resets them. For per-clip
provenance, snapshot `len(...)` before the call and slice after. `anonymize_pose_log` emits
no provenance block of its own -- upstream builds that in `mirage_tier1.main()`.

🔴 THE TWO LISTS ARE NOT INDEX-ALIGNED. The shipped `e2` preset carries no `stature_ratio`,
so `_STATURE_SCALE_USED` stays EMPTY while `_TEMPLATE_KIND_USED` grows one entry per slot
(verified 2026-08-14: 60-frame 1-person clip -> _TEMPLATE_KIND_USED ['anatomical_extent'],
_STATURE_SCALE_USED []). Never zip them; an empty scale list means "no stature fit ran",
NOT "scale 1.0".

ENV VARS READ AT IMPORT TIME (set them BEFORE importing this package, or they do nothing):
`MIRAGE_RESTORE_TORSO_ROLL` (:176), `MIRAGE_ANGLE_MIRROR` (:660), `MIRAGE_LOWFREQ_AMP`
(:1358) and `MIRAGE_ANGLE_CONST` / `MIRAGE_ANGLE_DRIFT` (:1386) -- the last three mutate
`LEVELS["L4"]` in place at module scope. Everything else (`MIRAGE_GAIT_PRESET`,
`MIRAGE_POSE_SCALE_FROM`, the `*_MULT` family, ...) is re-read per call.
"""

# The vendored defences read their knobs straight from the environment, and this package
# is importable on its own, so the pre-rename SITARA_* aliases are applied here too rather
# than relying on a parent package having been imported first. Idempotent.
try:
    from ..compat_env import apply_legacy_env_aliases as _apply_legacy_env_aliases
except ImportError:  # loaded by path, outside the mirage package
    import os as _os

    def _apply_legacy_env_aliases():
        for _k in list(_os.environ):
            if _k.startswith("SITARA_"):
                _n = "MIRAGE_" + _k[len("SITARA_"):]
                _os.environ.setdefault(_n, _os.environ[_k])

_apply_legacy_env_aliases()

import os as _os
import sys as _sys

#: Absolute path of this directory -- the one put on sys.path for the bare-name sibling import.
VENDOR_DIR = _os.path.dirname(_os.path.abspath(__file__))


def _ensure_on_path():
    """Put VENDOR_DIR at the FRONT of sys.path, at most once. Idempotent; returns the dir."""
    if VENDOR_DIR not in _sys.path:
        _sys.path.insert(0, VENDOR_DIR)
    return VENDOR_DIR


_ensure_on_path()

# Imported only AFTER the path insert -- otherwise pose_anon_edge.py:39's
# `from person_slots import slot_groups` raises ModuleNotFoundError. Import order is
# load-bearing here; do not let a formatter hoist these.
import pose_anon_edge  # noqa: E402
import person_slots    # noqa: E402

from pose_anon_edge import (  # noqa: E402
    anonymize_pose_log,   # whole-clip gait transform; returns a NEW log, scores untouched
    gait_preset,          # kwargs for anonymize_v2; defaults to SHIPPED_PRESET
    GAIT_PRESETS,         # the preset registry
    LEVELS,               # base dynamic-knob levels (L0/L2/L4)
    SHIPPED_PRESET,       # "e2" at this commit
    new_clip_seed,        # fresh per-SEQUENCE secrets seed -- never seed per identity
    test_fixed_seed,      # deterministic seed, parity tests only
    binarize_pose_scores,  # confidence -> {0,1} at the host threshold (0.5)
)

__all__ = [
    "VENDOR_DIR",
    "pose_anon_edge",
    "person_slots",
    "anonymize_pose_log",
    "gait_preset",
    "GAIT_PRESETS",
    "LEVELS",
    "SHIPPED_PRESET",
    "new_clip_seed",
    "test_fixed_seed",
    "binarize_pose_scores",
]
