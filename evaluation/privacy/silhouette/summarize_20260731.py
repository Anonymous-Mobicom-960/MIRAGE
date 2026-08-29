#!/usr/bin/env python3
"""
summarize_20260731.py -- turn a run_silhouette_reid.py report into the two tables the ledger
wants: the privacy/area pareto ranking, and the full protocol suite (rank-1/5, mAP, EER) per arm.

    python summarize_20260731.py [path/to/SILHOUETTE_reid_*.json]
"""
import json
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
DEF = os.path.join(os.path.dirname(os.path.dirname(HERE)), "tier1_lab", "reports", "reid",
                   "SILHOUETTE_reid_AREABUDGET_20260731.json")


def g(d, *ks, default=None):
    for k in ks:
        if not isinstance(d, dict):
            return default
        d = d.get(k)
        if d is None:
            return default
    return d


def main():
    path = sys.argv[1] if len(sys.argv) > 1 else DEF
    R = json.load(open(path))
    ctrl = R["arms"]["raw_bin"]["opengait_rank1"]["NM#5-6"] * 100
    ship_area = g(R, "pareto", "shipped_area_ratio")
    gate = R.get("gate", {})
    print(f"file      : {path}")
    print(f"gate      : passed={gate.get('passed')}  "
          f"max|delta|={max(abs(v) for v in gate['delta_pp'].values()):.2f} pp")
    print(f"N         : {R['n_sequences']} sequences, {R['n_subjects']} subjects, "
          f"chance rank-1 {R['chance_rank1']*100:.1f} %")
    print(f"control   : raw_bin NM {ctrl:.2f} %   raw NM "
          f"{R['arms']['raw']['opengait_rank1']['NM#5-6']*100:.2f} %")
    print(f"shipped   : mitigated NM {R['arms']['mitigated']['opengait_rank1']['NM#5-6']*100:.2f} %"
          f"  area x{ship_area:.4f}\n")

    rows = g(R, "pareto", "rows", default=[])
    print(f"{'arm':22s} {'NM%':>7} {'BG%':>7} {'CL%':>7} {'dNM':>8} {'area x':>8} {'IoU':>6} "
          f"{'pp/%area':>9} {'viol':>5} {'budget':>7}")
    for r in rows:
        print(f"{r['arm']:22s} {r['NM']:7.2f} {r['BG']:7.2f} {r['CL']:7.2f} "
              f"{r['delta_NM_pp']:+8.2f} {r['area_ratio']:8.4f} {r['iou']:6.3f} "
              f"{r['pp_per_pct_area']:9.3f} {r['superset_violations_px']:5d} "
              f"{'OK' if r['within_shipped_area_budget'] else 'OVER':>7}")

    # protocol.py's own suite. NOTE: a DIFFERENT convention from `opengait_rank1` above (it does
    # not exclude the identical view and uses the flattened-vector distance), so its rank-1 is much
    # higher and the two must never be mixed in one column. Reported because it is the suite the
    # pose channel is scored with in ledger §A.2.
    print(f"\nprotocol.py suite (NOT comparable to the OpenGait rank-1 above)")
    print(f"{'arm':22s} {'NM r1':>7} {'NM r5':>7} {'NM mAP':>7} {'NM EER':>7} {'probes':>7} "
          f"{'all r1':>7} {'all mAP':>8}")
    for r in [{"arm": "raw"}, {"arm": "raw_bin"}] + rows:
        m = g(R, "arms", r["arm"], "reid_metrics", default={})
        nm, ov = m.get("NM#5-6", {}), m.get("overall", {})
        def f(d, k, s=100.0):
            v = d.get(k)
            return f"{v*s:7.2f}" if isinstance(v, (int, float)) else f"{'-':>7}"
        print(f"{r['arm']:22s} {f(nm,'rank1')} {f(nm,'rank5')} {f(nm,'mAP')} {f(nm,'EER')} "
              f"{nm.get('n_probes','-'):>7} {f(ov,'rank1')} {f(ov,'mAP')}")


if __name__ == "__main__":
    main()
