#!/usr/bin/env python3
"""
window_inpaint_cost.py - what MASK_TEMPORAL_WIN costs TIER-2.
==============================================================
Ledger §A.6c measured the PRIVACY half of this knob (silhouette re-ID 88.66 % at win=1 down to
52.65 % at win=9). This measures the other half: how much harder each window makes Tier-2's job.

    python window_inpaint_cost.py --raw-mask <mask.mkv from a MASK_ANON_ON=False run> --tag C

THE METRIC THAT MATTERS is not hole area, it is the **NEVER-REVEALED CORE**: pixels the person
covers in EVERY frame. For those, Tier-2's reveal-and-fill has no real pixel to borrow from any
frame, so they must be HALLUCINATED by the neural core-fill (LaMa) - which is where the
"black border marks / flat dark bands" artefact comes from (PHONE_APP_STATUS session 2 measured
21.6 % core on a dynamic pan). A wider mitigation window grows the union every frame, so it can
only grow the core. Hole area matters too (it is inpaint work per frame) but the core is what
determines whether the output looks invented.

Reported per window: hole area %, never-revealed core % (of frame and of mean hole), and the
size of the largest connected core component - a single big blob is far worse to hallucinate
than the same area scattered.
"""
import argparse
import json
import os
import sys

import cv2
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.dirname(HERE)
sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(HERE)),
                                "tier1", "src", "edge_runner_pi5"))
import mirage_tier1 as ST                                       # noqa: E402  (the REAL mitigation)


def read_mask(path, max_frames=0):
    cap = cv2.VideoCapture(path)
    out = []
    while True:
        ok, f = cap.read()
        if not ok:
            break
        out.append(((f[..., 0] if f.ndim == 3 else f) >= 128).astype(np.uint8))
        if max_frames and len(out) >= max_frames:
            break
    cap.release()
    if not out:
        raise SystemExit(f"no frames in {path}")
    return np.stack(out)


def mitigate(raw, win, eps, shape=None, amp_frac=None):
    """Reproduce mirage_tier1.main()'s loop exactly (history holds the PRE-mitigation mask).

    `shape` selects a config.MASK_SHAPE_MODE ("hull" | "ellipse" | "displace"); None keeps
    whatever the live config has. Added 2026-07-25 to price the TIER-2 cost of the shape
    canonicalisation modes -- the privacy half is measured on CASIA-B, but the halo risk lives
    here: a mask much larger than the generated character reintroduces the ring V8.3 removed,
    and the never-revealed core is what has to be hallucinated outright."""
    import config as _C
    prev_mode = getattr(_C, "MASK_SHAPE_MODE", "none")
    prev_amp = getattr(_C, "MASK_DISPLACE_AMP_FRAC", 0.10)
    if shape is not None:
        _C.MASK_SHAPE_MODE = shape
    if amp_frac is not None:
        _C.MASK_DISPLACE_AMP_FRAC = float(amp_frac)
    try:
        hist, out = [], np.empty_like(raw)
        for t in range(raw.shape[0]):
            cur = np.ascontiguousarray(raw[t])
            hist.append(cur)
            if len(hist) > max(1, int(win)):
                hist.pop(0)
            # per-frame phase + per-clip seed, matching the device path
            _C._MASK_DISPLACE_PHASE = 0.35 * t
            _C._MASK_DISPLACE_SEED = 12345
            out[t] = ST.mask_mitigate(hist, cur, float(eps))
        return out
    finally:
        _C.MASK_SHAPE_MODE = prev_mode
        _C.MASK_DISPLACE_AMP_FRAC = prev_amp


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--raw-mask", required=True,
                    help="mask.mkv from a Tier-1 run with MASK_ANON_ON=False")
    ap.add_argument("--tag", default="C")
    ap.add_argument("--windows", default="1,3,5,7,9")
    ap.add_argument("--shapes", default="",
                    help="also price shape modes, as mode@win[@amp], e.g. "
                         "displace@2@0.10,hull@2,ellipse@2")
    ap.add_argument("--eps", type=float, default=0.01)
    ap.add_argument("--max-frames", type=int, default=0)
    a = ap.parse_args()

    raw = read_mask(a.raw_mask, a.max_frames)
    T, H, W = raw.shape
    frame_px = H * W
    print(f"[win] {a.raw_mask}: {T} frames {W}x{H}")

    # ⚠️ The plain --windows rows inherit the LIVE config's MASK_SHAPE_MODE (mitigate(shape=None)
    # keeps it). Found 2026-07-26: after displace@0.15 shipped, a "win=2" row silently measured
    # win2+displace and did NOT reproduce §B.2b's win2+none (27.10/15.873 vs 24.47/14.239 on the
    # same mask). Record the effective mode in every row so the two can never be conflated again.
    import config as _CC
    live_shape = str(getattr(_CC, "MASK_SHAPE_MODE", "none"))
    live_amp = float(getattr(_CC, "MASK_DISPLACE_AMP_FRAC", 0.10))
    print(f"[win] NOTE: plain-window rows use the LIVE config shape mode: {live_shape}"
          + (f"@{live_amp}" if live_shape == "displace" else ""))
    rows = []
    base_core = None
    for win in [int(w) for w in a.windows.split(",")]:
        m = mitigate(raw, win, a.eps)
        hole = float(m.mean())                              # mean per-frame hole area
        core = (m.min(axis=0) > 0).astype(np.uint8)         # masked in EVERY frame
        core_px = int(core.sum())
        n, lab, stats, _ = cv2.connectedComponentsWithStats(core, connectivity=8)
        big = int(stats[1:, cv2.CC_STAT_AREA].max()) if n > 1 else 0
        rec = {"win": win, "live_shape_mode": live_shape,
               "live_amp_frac": live_amp if live_shape == "displace" else None,
               "hole_area_pct": hole * 100,
               "never_revealed_core_pct_of_frame": 100 * core_px / frame_px,
               "never_revealed_core_pct_of_hole": (100 * core_px / (hole * frame_px)) if hole else 0.0,
               "core_components": int(n - 1), "largest_core_component_px": big,
               "largest_core_component_pct_of_frame": 100 * big / frame_px}
        if base_core is None:
            base_core = core_px
        rec["core_growth_vs_win1"] = (core_px / base_core) if base_core else float("nan")
        rows.append(rec)
        print(f"  win={win:2d}  hole {hole*100:6.2f}%  core {rec['never_revealed_core_pct_of_frame']:6.3f}% "
              f"of frame ({rec['never_revealed_core_pct_of_hole']:5.2f}% of hole)  "
              f"largest blob {big:7d}px  x{rec['core_growth_vs_win1']:.2f} vs win1")

    # ---- SHAPE-MODE arms (2026-07-25): the Tier-2 price of shape canonicalisation ----
    shape_rows = []
    for spec in [e for e in a.shapes.split(",") if e.strip()]:
        parts = spec.split("@")
        mode = parts[0].strip()
        win = int(parts[1]) if len(parts) > 1 else 2
        amp = float(parts[2]) if len(parts) > 2 else None
        m = mitigate(raw, win, a.eps, shape=mode, amp_frac=amp)
        hole = float(m.mean())
        core = (m.min(axis=0) > 0).astype(np.uint8)
        core_px = int(core.sum())
        n, lab, stats, _ = cv2.connectedComponentsWithStats(core, connectivity=8)
        big = int(stats[1:, cv2.CC_STAT_AREA].max()) if n > 1 else 0
        rec = {"arm": spec, "mode": mode, "win": win, "amp_frac": amp,
               "hole_area_pct": hole * 100,
               "never_revealed_core_pct_of_frame": 100 * core_px / frame_px,
               "never_revealed_core_pct_of_hole": (100 * core_px / (hole * frame_px)) if hole else 0.0,
               "core_components": int(n - 1), "largest_core_component_px": big,
               "largest_core_component_pct_of_frame": 100 * big / frame_px}
        shape_rows.append(rec)
        print(f"  {spec:>18}  hole {hole*100:6.2f}%  core {rec['never_revealed_core_pct_of_frame']:6.3f}% "
              f"of frame  largest blob {big:7d}px")

    # pair with the privacy half (ledger §A.6c) so the trade-off is readable in one place
    PRIVACY = {1: 88.66, 3: 71.44, 5: 62.27, 7: 56.65, 9: 52.65}
    print("\n  TRADE-OFF (privacy from ledger §A.6c, cost measured here)")
    print(f"  {'win':>4}  {'silhouette reID NM %':>21}  {'hole %':>8}  {'core % frame':>13}  {'largest blob %':>15}")
    for r in rows:
        p = PRIVACY.get(r["win"])
        print(f"  {r['win']:>4}  {(f'{p:.2f}' if p else ' - '):>21}  {r['hole_area_pct']:8.2f}  "
              f"{r['never_revealed_core_pct_of_frame']:13.3f}  "
              f"{r['largest_core_component_pct_of_frame']:15.3f}")

    if shape_rows:
        print("\n  SHAPE MODES (privacy half NOT yet measured -- CASIA-B sweep is deferred)")
        print(f"  {'arm':>18}  {'hole %':>8}  {'core % frame':>13}  {'largest blob %':>15}")
        for r in shape_rows:
            print(f"  {r['arm']:>18}  {r['hole_area_pct']:8.2f}  "
                  f"{r['never_revealed_core_pct_of_frame']:13.3f}  "
                  f"{r['largest_core_component_pct_of_frame']:15.3f}")

    out = os.path.join(HERE, "reports", f"WINDOW_INPAINT_COST_{a.tag}.json")
    os.makedirs(os.path.dirname(out), exist_ok=True)
    json.dump({"tag": a.tag, "source": os.path.abspath(a.raw_mask), "frames": T,
               "resolution": [W, H], "eps": a.eps, "rows": rows,
               "shape_modes": shape_rows,
               "privacy_reid_nm_pct_from_ledger_A6c": PRIVACY}, open(out, "w"), indent=2)
    print(f"\n[win] wrote {out}")


if __name__ == "__main__":
    main()
