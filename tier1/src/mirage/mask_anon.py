"""Host wrapper around the vendored MIRAGE silhouette mitigation (`mask_shape.mask_mitigate`).

WHAT THIS IS
------------
The measured MIRAGE silhouette arm is NOT "draw a rectangle". §A.6o measured `bbox` THROUGH
`mask_mitigate` at temporal window 2: a temporal running-max over the last N *dilated* masks,
contour simplification, the shape op, and then a hard OR with both the temporal union and the
current frame. This class reproduces that pipeline on this host WITHOUT touching the vendored
code - the ordering below is copied from `mirage_tier1.main()`, not invented:

    mirage_tier1.py:2140   binm = cv2.dilate(binm, ek)        # EDGE_EXPAND, ellipse 3x3
    mirage_tier1.py:2164   mask_hist.append(binm)             # PRE-mitigation, post-dilate
    mirage_tier1.py:2166   mask_hist.pop(0) if len > MASK_TEMPORAL_WIN
    mirage_tier1.py:2167   binm = mask_mitigate(mask_hist, binm, MASK_SIMPLIFY_EPS)

`EDGE_EXPAND = 1` is `config.py:95` (an ellipse kernel of `2*E+1` square, `mirage_tier1.py:1719`);
`MASK_SIMPLIFY_EPS = 0.01` is `config.py:829`. Neither is invented here, and neither is fitted to
a clip.

WHAT IT DELIBERATELY DOES NOT DO
--------------------------------
* It does NOT write back into the caller's mask. `apply()` returns a NEW array and never mutates
  its input. On this host `last_seg_mask` is PROPAGATED across frames (warped forward on skip
  frames and reused, `pipeline.py:1395-1408`); feeding a mitigated mask back into it would make
  the temporal running-max compound frame over frame and the grey region grow without bound.
  Keep the emitted mask separate from the propagated one.
* It keeps its OWN ring buffer. The buffer holds the PRE-mitigation dilated masks, so no output
  ever re-enters the history - same no-feedback property as the vendored host loop.
* It carries no privacy claim. Every §A.6 number was measured on the MIRAGE host; nothing here
  describes what this host achieves.

SEEDS
-----
One per-clip seed is drawn from `secrets.randbits(31)` (the CSPRNG, per sequence, never derived
from identity or content). It is used in-process only and is NEVER returned by `stats()` - a
recorded seed is a linkable pseudo-identity. `bbox` does not consume it at all; it is drawn and
supplied so a run at any other shape mode is correct without a code change.
"""

from __future__ import annotations

import inspect
import re
import secrets
from collections import deque
from typing import Any, Dict, Optional, Tuple

import cv2
import numpy as np

from .provenance import vendor_module

__all__ = ["MaskAnonymizer", "supported_shape_modes", "check_shape_mode"]

# Documented MIRAGE defaults, used only when the vendored config shim does not carry them.
_DEFAULT_EDGE_EXPAND = 1        # config.py:95
_DEFAULT_SIMPLIFY_EPS = 0.01    # config.py:829
_DEFAULT_PHASE_STEP = 0.35      # config.py:638 (MASK_DISPLACE_PHASE_STEP)
_DEFAULT_BBOX_MERGE = True      # config.py:558
_DEFAULT_BBOX_PAD_FRAC = 0.0    # config.py:562

# --------------------------------------------------------------------------------------------
# 🔴 SHAPE-MODE VALIDATION -- an UNKNOWN MODE IS A SILENT NO-OP IN THE VENDORED CODE.
#
# `_shape_polys` (mask_shape.py:232) dispatches on `mode` with a chain of `if mode == "...":
# return ...` and NO else-clause. Its final statement is the fall-through
#
#     return [cv2.approxPolyDP(c, eps_frac * cv2.arcLength(c, True), True) for c in cnts]
#
# so ANY name it does not recognise degrades to plain contour simplification. That is not a
# crash and it is not logged: the run would emit a nearly-unmitigated silhouette while the
# manifest declared `shape_mode: "<typo>", enabled: true`. MEASURED on this host, one synthetic
# 200x200 two-blob mask at temporal_win=2 (emitted/input area):
#
#     "bbox"             2.326        <- the measured arm
#     "hull"             1.672
#     "bbxo"             1.082        <- a one-character typo
#     "totally_made_up"  1.082        <- identical to the typo: the fall-through
#
# 1.082 is the temporal union plus approxPolyDP, i.e. the shape defence did not run at all.
# That is exactly the "declared one config while carrying another" failure `provenance.py`
# exists to prevent, so the name is checked before a clip is processed.
#
# The valid set is DERIVED FROM THE VENDORED SOURCE (regex over `_shape_polys`'s own dispatch),
# never a list maintained here -- a hand-copied list is the next thing to drift out of date, and
# `mask_mitigate` supports "+"-composed modes ("displace+close", :843) whose parts are the same
# branch names, so one derived set covers both forms.
_MODE_BRANCH_RE = re.compile(r"""^\s*if\s+mode\s*==\s*['"]([a-z0-9_]+)['"]\s*:""", re.M)
_OFF_MODES = frozenset({"", "none", "off"})


def supported_shape_modes() -> Tuple[str, ...]:
    """Shape-mode names the VENDORED `_shape_polys` actually branches on, read from its source."""
    src = inspect.getsource(vendor_module("mask_shape")._shape_polys)
    return tuple(sorted(set(_MODE_BRANCH_RE.findall(src))))


def check_shape_mode(mode: str) -> str:
    """Return the normalised mode, or raise ValueError naming the supported set.

    Accepts the off spellings ("", "none", "off") and "+"-composed modes, matching
    `mask_mitigate`'s own `mode.split("+")` (mask_shape.py:843) and its `.lower()` (:828).
    """
    norm = str(mode or "none").strip().lower()
    if norm in _OFF_MODES:
        return "none"
    ok = supported_shape_modes()
    bad = [op for op in norm.split("+") if op and op not in ok]
    if bad or not [op for op in norm.split("+") if op]:
        raise ValueError(
            f"mask_shape_mode={mode!r}: unknown shape op(s) {bad or [mode]}. The vendored "
            f"_shape_polys has no else-branch, so an unrecognised name SILENTLY falls through "
            f"to plain contour simplification -- the shape defence would not run while the "
            f"manifest declared it enabled (measured: emitted/input area 1.082x vs 2.326x for "
            f"'bbox' on the same input).\n"
            f"  Supported: {', '.join(ok)} (or 'none'/'off' to disable; '+'-compose with e.g. "
            f"'displace+close').\n"
            f"  The measured arm is 'bbox' (owner decision).")
    return norm


class MaskAnonymizer:
    """Per-clip silhouette mitigation. One instance per clip; `apply()` once per emitted frame.

    NOT THREAD-SAFE ACROSS INSTANCES. `mask_mitigate` reads its configuration off a module-level
    shim (`mask_shape.C`), so two instances calling `apply()` concurrently would race on it. The
    intended use - one instance, called from the frame loop - is safe.

    Args:
        shape_mode: vendored `MASK_SHAPE_MODE`. `"bbox"` is the default mask route (owner
            decision). `"none"` leaves the contour simplification and temporal union only.
        temporal_win: running-max window IN FRAMES. Default 2 - the window §A.6o measured.
            ⚠️ On the MIRAGE host this is derived from a DURATION and silently becomes 1 at
            10 fps, which is the weakest window. Here it is an explicit frame count: pass what
            you mean.
        edge_expand_px: EDGE_EXPAND dilate radius, applied FIRST. None => the vendored
            `C.EDGE_EXPAND`, else 1. 0 disables the dilate (as `mirage_tier1.py:1719` does).
        seed: per-clip seed. None => `secrets.randbits(31)`. Never exposed by `stats()`.
        simplify_eps: `approxPolyDP` epsilon as a fraction of each contour's perimeter.
            None => vendored `C.MASK_SIMPLIFY_EPS`, else 0.01.
        mask_shape_module: test hook - inject the vendored module instead of importing it.
    """

    def __init__(self,
                 shape_mode: str = "bbox",
                 temporal_win: int = 2,
                 edge_expand_px: Optional[int] = None,
                 seed: Optional[int] = None,
                 simplify_eps: Optional[float] = None,
                 phase_step: Optional[float] = None,
                 bbox_merge: Optional[bool] = None,
                 bbox_pad_frac: Optional[float] = None,
                 mask_shape_module: Any = None):
        self._ms = mask_shape_module if mask_shape_module is not None else vendor_module("mask_shape")
        self._C = getattr(self._ms, "C", None)
        if self._C is None:
            raise RuntimeError("vendored mask_shape module exposes no config shim `C`; "
                               "mask_mitigate reads MASK_SHAPE_MODE and friends off it.")

        # Refuse an unrecognised mode rather than silently emitting an unmitigated silhouette
        # under its name -- see check_shape_mode above for the measured fall-through.
        self.shape_mode = check_shape_mode(shape_mode)
        self.temporal_win = max(1, int(temporal_win))
        self.edge_expand_px = int(self._pick(edge_expand_px, "EDGE_EXPAND", _DEFAULT_EDGE_EXPAND))
        self.simplify_eps = float(self._pick(simplify_eps, "MASK_SIMPLIFY_EPS",
                                             _DEFAULT_SIMPLIFY_EPS))
        self.phase_step = float(self._pick(phase_step, "MASK_DISPLACE_PHASE_STEP",
                                           _DEFAULT_PHASE_STEP))
        self.bbox_merge = bool(self._pick(bbox_merge, "MASK_BBOX_MERGE", _DEFAULT_BBOX_MERGE))
        self.bbox_pad_frac = float(self._pick(bbox_pad_frac, "MASK_BBOX_PAD_FRAC",
                                              _DEFAULT_BBOX_PAD_FRAC))

        # PER-CLIP, from the OS CSPRNG. Held privately; never logged, never in stats().
        self._seed = int(secrets.randbits(31)) if seed is None else int(seed) & 0x7FFFFFFF
        self.seed_supplied = seed is not None

        self._kernel = (cv2.getStructuringElement(
            cv2.MORPH_ELLIPSE, (2 * self.edge_expand_px + 1,) * 2)
            if self.edge_expand_px else None)

        # PRE-mitigation, POST-dilate masks. maxlen == the window, so the oldest drops exactly
        # where the vendored loop pops it.
        self._hist: deque = deque(maxlen=self.temporal_win)
        self._shape: Optional[tuple] = None

        self._n = 0                    # apply() calls that produced a mask
        self._n_none_emitted = 0       # mask=None handled as an empty frame
        self._n_none_skipped = 0       # mask=None before any shape was known -> returned None
        self._n_empty_in = 0           # input had no set pixels
        self._n_resets = 0             # frame-size change -> ring buffer cleared
        self._s2_viol = 0              # frames where emitted NOT superset of the CALLER's mask
        self._s2_viol_px = 0
        self._dil_viol = 0             # frames where emitted NOT superset of the dilated mask
        self._area_in = 0
        self._area_out = 0
        self._ratio_sum = 0.0
        self._ratio_n = 0

    # ---------------------------------------------------------------- internals
    def _pick(self, explicit, cname, fallback):
        if explicit is not None:
            return explicit
        v = getattr(self._C, cname, None)
        return fallback if v is None else v

    def _push_config(self) -> None:
        """Point the vendored config shim at THIS instance's knobs.

        `mask_mitigate` / `_shape_polys` read their configuration off the module-level `C`
        (`getattr(C, "MASK_SHAPE_MODE", ...)` etc.), so the adapter sets it on every call rather
        than once: two MaskAnonymizers in one process would otherwise silently share whichever
        was constructed last. This is host glue - the vendored file itself is untouched.
        """
        C = self._C
        C.MASK_SHAPE_MODE = self.shape_mode
        C.MASK_SIMPLIFY_EPS = self.simplify_eps
        C.MASK_BBOX_MERGE = self.bbox_merge
        C.MASK_BBOX_PAD_FRAC = self.bbox_pad_frac
        C._MASK_DISPLACE_SEED = self._seed
        # The host advances the displacement phase every EMITTED frame (mirage_tier1.py:2109)
        # so the perturbation varies over the sequence; with a constant phase only the static
        # half of the mitigation is active. Unused by `bbox`, correct for every other mode.
        C._MASK_DISPLACE_PHASE = self.phase_step * self._n

    # ---------------------------------------------------------------- public
    def apply(self, mask_bool_HxW: Optional[np.ndarray]) -> Optional[np.ndarray]:
        """Emit the mitigated mask for one frame. Returns a NEW bool array (or None).

        The input is never modified and is never retained. Any non-zero value counts as set, so
        a bool mask, a 0/1 mask and a 0/255 mask all behave identically.

        `mask_bool_HxW is None` means "no mask this frame" and is handled EXPLICITLY:

        * Before any frame has been seen (the frame size is still unknown) there is nothing to
          emit and nothing to remember: `apply()` returns **None** and the caller should emit
          whatever it emits when it has no mask (on this host, an all-zero frame,
          `pipeline.py:1426`). The ring buffer is untouched.
        * Once a size is known, None is treated as an **all-zero frame** and goes through the
          full path. This is faithful to the vendored host, where a detection drop yields an
          all-zero `binm` that is still appended and still mitigated - so the temporal
          running-max carries the previous frames' coverage over the gap. The emitted mask is
          therefore NOT empty on a drop frame; it is the union of the window. That is the
          measured behaviour and it is the conservative direction (coverage only grows).
        """
        if mask_bool_HxW is None:
            if self._shape is None:
                self._n_none_skipped += 1
                return None
            self._n_none_emitted += 1
            raw = np.zeros(self._shape, np.uint8)
        else:
            arr = np.asarray(mask_bool_HxW)
            if arr.ndim != 2:
                raise ValueError(f"mask must be HxW, got shape {arr.shape}")
            # `!= 0` allocates a fresh array, so nothing downstream can alias the caller's data.
            raw = (arr != 0).astype(np.uint8)

        if self._shape is not None and raw.shape != self._shape:
            # A frame-size change invalidates the running max (np.stack would raise). Start the
            # window again rather than silently mixing resolutions.
            self._hist.clear()
            self._n_resets += 1
        self._shape = raw.shape

        # (1) EDGE_EXPAND first - every history frame then carries the margin, exactly as in the
        #     vendored loop, and the mitigation operates on the dilated silhouette.
        cur = cv2.dilate(raw, self._kernel) if self._kernel is not None else raw.copy()

        # (2) ring buffer of PRE-mitigation masks (no output feedback).
        self._hist.append(cur)

        # (3) the vendored mitigation. `hist` is a list, current frame last, len <= window.
        self._push_config()
        out = self._ms.mask_mitigate(list(self._hist), cur, self.simplify_eps)
        out = np.ascontiguousarray(out).astype(bool)

        # (4) §2 accounting. The vendored function guarantees emitted >= sm >= cur by
        #     construction; we MEASURE it rather than trust it, because "the mask is a superset"
        #     is the whole safety argument.
        rawb = raw.astype(bool)
        viol = int(np.count_nonzero(rawb & ~out))
        if viol:
            self._s2_viol += 1
            self._s2_viol_px += viol
        if np.count_nonzero(cur.astype(bool) & ~out):
            self._dil_viol += 1

        a_in, a_out = int(rawb.sum()), int(out.sum())
        self._area_in += a_in
        self._area_out += a_out
        if a_in:
            self._ratio_sum += a_out / a_in
            self._ratio_n += 1
        else:
            self._n_empty_in += 1
        self._n += 1
        return out

    def reset(self) -> None:
        """Clear the temporal window (new clip / discontinuity). Counters are kept."""
        self._hist.clear()
        self._shape = None

    def stats(self) -> Dict[str, Any]:
        """Factual counters for this instance. Contains NO seed value, by design."""
        return {
            "frames": self._n,
            "frames_none_as_empty": self._n_none_emitted,
            "frames_none_skipped_no_shape": self._n_none_skipped,
            "frames_empty_input": self._n_empty_in,
            "window_resets": self._n_resets,
            # §2: the emitted mask must be a SUPERSET of the mask the caller handed in.
            "s2_superset_violations": self._s2_viol,
            "s2_superset_violation_px": self._s2_viol_px,
            "s2_superset_violations_vs_dilated": self._dil_viol,
            "area_in_px": self._area_in,
            "area_out_px": self._area_out,
            "area_ratio_mean": (self._ratio_sum / self._ratio_n) if self._ratio_n else None,
            "shape_mode": self.shape_mode,
            "temporal_win": self.temporal_win,
            "edge_expand_px": self.edge_expand_px,
            "simplify_eps": self.simplify_eps,
            "bbox_merge": self.bbox_merge,
            "bbox_pad_frac": self.bbox_pad_frac,
            "seed_supplied_by_caller": self.seed_supplied,
        }
