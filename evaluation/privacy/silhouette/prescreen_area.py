#!/usr/bin/env python3
"""
prescreen_area.py -- AREA-ONLY pre-screen of candidate silhouette-mitigation configs.
=====================================================================================
Area is the budget (ledger §B.48: `hull` at ×1.470 cost SSIM -0.0215 / LPIPS +0.0157 on the
composite), so an arm above the SHIPPED ×1.358 is not recommendable on privacy alone. Embedding
5485 sequences through GaitBase costs minutes per arm; measuring the AREA of an arm costs
seconds. This screens the candidate grid on area + §2 first, so the expensive re-ID pass only
carries arms that could actually be shipped.

It runs the REAL `mirage_tier1.mask_mitigate()` through the REAL harness loop, so the area
ratios here are the same quantity `run_silhouette_reid.py` reports -- just on a subject subset.
NOT a privacy measurement: it produces no re-ID number at all.

    python prescreen_area.py --silh-root "<...>/CASIA-B Dataset/output" --subjects 4
"""
import argparse
import json
import os
import sys
import time

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import silhouette_harness as H                                        # noqa: E402

OUT = os.path.join(HERE, "reports")
TEST_IDS = list(range(75, 125))


def arms_grid(tmpl_path):
    """The candidate grid. `cfg` keys are edge-config attributes."""
    D = "displace"
    g = []
    def add(name, win=2, **cfg):
        g.append({"name": name, "win": win, "cfg": cfg})

    # --- anchors -----------------------------------------------------------------
    add("mit_w2_a25", 2, MASK_SHAPE_MODE=D, MASK_DISPLACE_AMP_FRAC=0.25)       # SHIPPED
    add("none_w2", 2, MASK_SHAPE_MODE="none")
    add("hull_w2", 2, MASK_SHAPE_MODE="hull")                                  # banned; scale only
    # --- window ------------------------------------------------------------------
    for w in (1, 3, 4, 6, 8):
        add(f"mit_w{w}_a25", w, MASK_SHAPE_MODE=D, MASK_DISPLACE_AMP_FRAC=0.25)
    # --- amplitude ---------------------------------------------------------------
    for a in (0.35, 0.45, 0.60):
        add(f"mit_w2_a{int(a*100)}", 2, MASK_SHAPE_MODE=D, MASK_DISPLACE_AMP_FRAC=a)
    # --- harmonics (area-neutral by design) --------------------------------------
    for h in (5, 8, 12):
        add(f"mit_w2_a25_h{h}", 2, MASK_SHAPE_MODE=D, MASK_DISPLACE_AMP_FRAC=0.25,
            MASK_DISPLACE_HARMONICS=h)
    # --- phase-advance rate (area-neutral) ---------------------------------------
    for s in (0.0, 1.0, 2.0, 3.0):
        add(f"mit_w2_a25_ps{s}", 2, MASK_SHAPE_MODE=D, MASK_DISPLACE_AMP_FRAC=0.25,
            MASK_DISPLACE_PHASE_STEP=s)
    # --- per-epoch re-seed (area-neutral) ----------------------------------------
    for r in (3.5, 1.4, 0.34):
        add(f"mit_w2_a25_rs{r}", 2, MASK_SHAPE_MODE=D, MASK_DISPLACE_AMP_FRAC=0.25,
            MASK_DISPLACE_RESEED_PHASE=r)
    # --- radial low-pass ---------------------------------------------------------
    for k in (2, 4, 6, 8, 12):
        add(f"rlp{k}_w2", 2, MASK_SHAPE_MODE="radiallp", MASK_RADIALLP_KEEP=k)
    # --- morphological close -----------------------------------------------------
    for f in (0.10, 0.15, 0.25, 0.35):
        add(f"close{int(f*100)}_w2", 2, MASK_SHAPE_MODE="close", MASK_CLOSE_KERNEL_FRAC=f)
    # --- k-same collapse ---------------------------------------------------------
    if os.path.isfile(tmpl_path):
        T = json.load(open(tmpl_path))["templates"]
        for q in ("p50", "p65", "p75", "p85"):
            add(f"ksame_{q}_w2", 2, MASK_SHAPE_MODE="ksame",
                MASK_KSAME_TEMPLATE=tuple(T[q]))
        # --- compositions ---------------------------------------------------------
        for q in ("p50", "p65", "p75"):
            add(f"ksame_{q}+d25_w2", 2, MASK_SHAPE_MODE="ksame+displace",
                MASK_KSAME_TEMPLATE=tuple(T[q]), MASK_DISPLACE_AMP_FRAC=0.25)
            add(f"ksame_{q}+d15_w2", 2, MASK_SHAPE_MODE="ksame+displace",
                MASK_KSAME_TEMPLATE=tuple(T[q]), MASK_DISPLACE_AMP_FRAC=0.15)
        for q in ("p50", "p65"):
            add(f"ksame_{q}+rlp6_w2", 2, MASK_SHAPE_MODE="ksame+radiallp",
                MASK_KSAME_TEMPLATE=tuple(T[q]), MASK_RADIALLP_KEEP=6)
            add(f"ksame_{q}+cl15+d15_w2", 2, MASK_SHAPE_MODE="ksame+close+displace",
                MASK_KSAME_TEMPLATE=tuple(T[q]), MASK_CLOSE_KERNEL_FRAC=0.15,
                MASK_DISPLACE_AMP_FRAC=0.15)
    for f, a in ((0.15, 0.15), (0.15, 0.25), (0.25, 0.15)):
        add(f"cl{int(f*100)}+d{int(a*100)}_w2", 2, MASK_SHAPE_MODE="close+displace",
            MASK_CLOSE_KERNEL_FRAC=f, MASK_DISPLACE_AMP_FRAC=a)
    for k, a in ((6, 0.15), (6, 0.25), (10, 0.25)):
        add(f"rlp{k}+d{int(a*100)}_w2", 2, MASK_SHAPE_MODE="radiallp+displace",
            MASK_RADIALLP_KEEP=k, MASK_DISPLACE_AMP_FRAC=a)
    return g


_MISSING = object()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--silh-root", required=True)
    ap.add_argument("--subjects", type=int, default=4)
    ap.add_argument("--tmpl", default=os.path.join(OUT, "KSAME_TEMPLATE.json"))
    ap.add_argument("--tag", default="")
    a = ap.parse_args()
    os.makedirs(OUT, exist_ok=True)

    subs = TEST_IDS[:a.subjects]
    # PNG loading is minutes and dominates; cache the binarised masks so re-running the grid
    # (e.g. once the k-same template exists) costs seconds instead of another full read.
    cpath = os.path.join(H.DATA, f"prescreen_masks_{len(subs)}.npz")
    if os.path.isfile(cpath):
        z = np.load(cpath)
        base = {tuple(int(x) for x in n.split("|")): z[n] for n in z.files}
        print(f"[pre] loaded {len(base)} cached sequences from {cpath}", flush=True)
    else:
        print(f"[pre] loading {len(subs)} TEST subjects ...", flush=True)
        tr = H.load_casia_b_silhouettes(a.silh_root, subjects=set(subs))
        base = {k: (v > 127).astype(np.uint8) for k, v in tr.items()}
        os.makedirs(H.DATA, exist_ok=True)
        np.savez_compressed(cpath, **{"|".join(str(int(x)) for x in k): v
                                      for k, v in base.items()})
        print(f"[pre] cached -> {cpath}", flush=True)
    nfr = sum(v.shape[0] for v in base.values())
    print(f"[pre] {len(base)} sequences, {nfr} frames", flush=True)

    grid = arms_grid(a.tmpl)
    rows = []
    for spec in grid:
        prev = {}
        for k, v in spec["cfg"].items():
            prev[k] = getattr(H.EDGE_CFG, k, _MISSING)
            setattr(H.EDGE_CFG, k, v)
        t0 = time.perf_counter()
        try:
            areas, viol = [], 0
            for k, v in base.items():
                o = H.apply_mask_mitigation(v, win=int(spec["win"]))
                for t in range(v.shape[0]):
                    x, y = v[t] > 0, o[t] > 0
                    viol += int(np.count_nonzero(x & ~y))
                    if x.sum():
                        areas.append(y.sum() / x.sum())
        finally:
            for k, v in prev.items():
                if v is _MISSING:
                    if hasattr(H.EDGE_CFG, k):
                        delattr(H.EDGE_CFG, k)
                else:
                    setattr(H.EDGE_CFG, k, v)
        dt = time.perf_counter() - t0
        rows.append({"name": spec["name"], "win": spec["win"],
                     "cfg": {k: (v if not isinstance(v, tuple) else f"<{len(v)} floats>")
                             for k, v in spec["cfg"].items()},
                     "area_ratio_mean": float(np.mean(areas)),
                     "superset_violations_px": int(viol),
                     "ms_per_frame": dt / max(1, nfr) * 1000.0})
        r = rows[-1]
        print(f"  {r['name']:26s} area x{r['area_ratio_mean']:.4f}  viol {r['superset_violations_px']:d}"
              f"  {r['ms_per_frame']:.3f} ms/f", flush=True)

    ship = next(r["area_ratio_mean"] for r in rows if r["name"] == "mit_w2_a25")
    for r in rows:
        r["within_shipped_budget"] = bool(r["area_ratio_mean"] <= ship)
    path = os.path.join(OUT, f"PRESCREEN_AREA{('_' + a.tag) if a.tag else ''}.json")
    json.dump({"subjects": subs, "n_sequences": len(base), "n_frames": nfr,
               "shipped_area_ratio": ship, "rows": rows}, open(path, "w"), indent=1)
    print(f"\n[pre] shipped (displace .25 @w2) area = x{ship:.4f}; "
          f"{sum(r['within_shipped_budget'] for r in rows)}/{len(rows)} arms within budget")
    print(f"[pre] wrote {path}")


if __name__ == "__main__":
    main()
