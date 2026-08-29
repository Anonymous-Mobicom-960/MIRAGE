#!/usr/bin/env python3
"""
run_silhouette_reid.py -- silhouette-channel re-ID evaluation of the SHIPPED mask mitigation.
=============================================================================================
THREE arms, all derived from ONE load so each comparison isolates one variable:
  raw        silhouettes as released (anti-aliased) -- the domain GaitBase was trained on, and
             the ONLY arm comparable to the published baseline (the validation GATE).
  raw_bin    raw hard-binarised at 127 -- the CONTROL for the mitigation, because
             mask_mitigate() consumes a binary mask.
  mitigated  raw_bin through the REAL mirage_tier1.mask_mitigate() at deployed parameters.
So `raw -> raw_bin` is the cost of binarising, and `raw_bin -> mitigated` is the shipped
mitigation's actual effect on an attacker.

Adversary: OpenGait GaitBase_DA, the PUBLISHED CASIA-B checkpoint. Reported with OpenGait's own
distance (mean over the 16 parts) so the gate is meaningful, AND with protocol.py's suite
(rank1/rank5/mAP/EER + bootstrap CIs) so the numbers sit beside the pose channel in ledger §A.2.

  python run_silhouette_reid.py --silh-root "<...>/CASIA-B Dataset/output" \
      --ckpt data/CASIA-B/Baseline/GaitBase_DA/checkpoints/GaitBase_DA-60000.pt

PNG loading dominates (~45 s/subject), so subjects are STREAMED: load one subject, build all
three arms, embed, discard the frames. Embeddings are cached to data/emb_cache.npz -- rerunning
the analysis costs seconds.
"""
import argparse
import json
import os
import sys
import time

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
sys.path.insert(0, os.path.dirname(HERE))                       # reid_eval/ (protocol.py)

import protocol as P                                            # noqa: E402
import silhouette_harness as H                                  # noqa: E402

OUT_DIR = os.path.join(H.LAB, "reports", "reid")
ARMS = ("raw", "raw_bin", "mitigated")
TRAIN_IDS = set(range(1, 75))       # the 74 identities the published checkpoint was trained on
TEST_IDS = set(range(75, 125))      # the 50 held-out identities (chance rank-1 = 2.0%)


def key_str(k):
    return "|".join(str(int(x)) for x in k)


def key_parse(s):
    return tuple(int(x) for x in s.split("|"))


def embed_all(silh_root, ckpt, opengait_root, subjects, max_frames, cache, contour_budget,
              windows=(), shapes=(), specs=(), pool=None, ckpt_every=10):
    """Stream subjects: load -> arms -> embed -> drop frames. Returns (emb, coverage, arm_names).

    `windows` adds one extra arm per MASK_TEMPORAL_WIN value (`mit_w<N>`), all computed in the
    SAME pass - PNG loading costs ~45 s/subject and dominates everything else, so sweeping the
    knob for free here beats N separate 40-minute runs.

    `specs` (2026-07-31) is the GENERAL form of the same idea: a list of
    `{"name": str, "win": int, "cfg": {EDGE_CONFIG_ATTR: value}}`. Any knob `mask_mitigate()`
    reads can be swept without adding a positional field to `--shape-sweep` every time - which is
    what the harmonics / phase-step / new-shape-mode sweeps need."""
    rec = H.OpenGaitRecognizer(ckpt, opengait_root)
    names = (list(ARMS) + [f"mit_w{w}" for w in windows]
             + [f"mit_{m}_w{w}" + (f"_a{amp}" if amp is not None else "")
                + (f"_h{ha}" if ha is not None else "")
                for (m, w, amp, ha) in shapes]
             + [s["name"] for s in specs])
    emb = {a: {} for a in names}
    cov = {}                                    # per-window coverage accumulators
    subs = sorted(subjects)
    t_all = time.time()
    for i, sid in enumerate(subs, 1):
        t0 = time.time()
        tr = H.load_casia_b_silhouettes(silh_root, subjects={sid}, max_frames=max_frames)
        if not tr:
            print(f"  [{i}/{len(subs)}] subject {sid}: NO sequences found", flush=True)
            continue
        arms = H.build_arms(tr, arms=ARMS)
        for a in ARMS:
            emb[a].update(rec.embed(arms[a]))
        base = {k: v // 255 for k, v in arms["raw_bin"].items()}     # {0,1} control
        # Key the deployed arm by the window it ACTUALLY used. It was hardcoded to "5", which
        # silently mislabels the coverage row the moment config.MASK_TEMPORAL_WIN changes - the
        # `mitigated` arm follows the config, so the label has to as well.
        cov[str(H.MASK_WIN)] = H.coverage_accumulate(
            base, {k: v // 255 for k, v in arms["mitigated"].items()},
            cov.get(str(H.MASK_WIN)), contour_budget=contour_budget)
        for w in windows:                        # extra MASK_TEMPORAL_WIN arms, same source
            mit = {k: H.apply_mask_mitigation(v, win=w) for k, v in base.items()}
            emb[f"mit_w{w}"].update(rec.embed({k: v * 255 for k, v in mit.items()}))
            cov[str(w)] = H.coverage_accumulate(base, mit, cov.get(str(w)),
                                                contour_budget=contour_budget)
        # SHAPE-CANONICALISATION arms (2026-07-25). mask_mitigate() reads the shape mode off the
        # edge config, so each arm is produced by flipping that global around the call and putting
        # it back. Same source silhouettes as every other arm, same pass -- PNG loading dominates,
        # so extra arms are nearly free here and MUST NOT be run as separate jobs (a second run
        # would be a different embedding pass and no longer a controlled comparison).
        _prev_mode = getattr(H.EDGE_CFG, "MASK_SHAPE_MODE", "none")
        _prev_amp = getattr(H.EDGE_CFG, "MASK_DISPLACE_AMP_FRAC", 0.10)
        # The head knob must be saved/restored exactly like the other two, or a split arm would
        # leak its head amplitude into every LATER arm in the same pass and silently corrupt the
        # comparison -- the arms share one process and one config module.
        _prev_head = getattr(H.EDGE_CFG, "MASK_DISPLACE_AMP_HEAD", None)
        for (mode, w, amp, ha) in shapes:
            H.EDGE_CFG.MASK_SHAPE_MODE = mode
            if amp is not None:
                H.EDGE_CFG.MASK_DISPLACE_AMP_FRAC = amp
            H.EDGE_CFG.MASK_DISPLACE_AMP_HEAD = ha      # None => uniform, explicitly
            try:
                mit = {k: H.apply_mask_mitigation(v, win=w) for k, v in base.items()}
            finally:
                H.EDGE_CFG.MASK_SHAPE_MODE = _prev_mode
                H.EDGE_CFG.MASK_DISPLACE_AMP_FRAC = _prev_amp
                H.EDGE_CFG.MASK_DISPLACE_AMP_HEAD = _prev_head
            key = (f"mit_{mode}_w{w}" + (f"_a{amp}" if amp is not None else "")
                   + (f"_h{ha}" if ha is not None else ""))
            emb[key].update(rec.embed({k: v * 255 for k, v in mit.items()}))
            cov[key] = H.coverage_accumulate(base, mit, cov.get(key),
                                             contour_budget=contour_budget)
        # GENERAL arms (2026-07-31): arbitrary edge-config overrides per arm, same source
        # silhouettes, same pass, same embedding model -- so every arm below is a controlled
        # comparison against `mitigated` and against each other.
        for spec in specs:
            mit = H.mitigate_batch(base, int(spec.get("win", H.MASK_WIN)),
                                   spec.get("cfg", {}), pool=pool)
            nm = spec["name"]
            emb[nm].update(rec.embed({k: v * 255 for k, v in mit.items()}))
            cov[nm] = H.coverage_accumulate(base, mit, cov.get(nm),
                                            contour_budget=contour_budget)
        print(f"  [{i}/{len(subs)}] subject {sid}: {len(tr):3d} seqs, "
              f"{time.time()-t0:5.1f}s  (elapsed {(time.time()-t_all)/60:.1f} min)", flush=True)
        del tr, arms
        # CHECKPOINT (2026-07-31). A 2.5-hour pass that dies at subject 40 used to lose
        # everything. Dump the embeddings so far every `ckpt_every` subjects - uncompressed,
        # because compression on a ~500 MB dict costs minutes and this is a crash net, not an
        # artifact. Re-analysing a checkpoint needs --from-cache and a smaller subject set.
        if cache and ckpt_every and i % ckpt_every == 0 and i < len(subs):
            np.savez(cache + f".ckpt{i}.npz", **{f"{a}::{key_str(k)}": v
                                                 for a in names for k, v in emb[a].items()})
            print(f"      [checkpoint {i}/{len(subs)} -> {cache}.ckpt{i}.npz]", flush=True)
    if cache:
        np.savez_compressed(cache, **{f"{a}::{key_str(k)}": v
                                      for a in names for k, v in emb[a].items()})
        print(f"[silh] cached embeddings -> {cache}")
    return emb, {k: H.coverage_finalize(v) for k, v in cov.items()}, names


def load_cache(path):
    """Arm names come from the CACHE, not from ARMS.

    A `--win-sweep` run writes extra `mit_w<N>` arms alongside the three fixed ones. Seeding the
    dict from ARMS made `--from-cache` raise KeyError on exactly those caches, so the §A.6c sweep
    could not be re-analysed the way this file documents. Discover the arms instead.
    """
    z = np.load(path)
    emb = {}
    for name in z.files:
        a, _, ks = name.partition("::")
        emb.setdefault(a, {})[key_parse(ks)] = z[name]
    for a in ARMS:
        emb.setdefault(a, {})
    return emb


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--silh-root", default=None)
    ap.add_argument("--ckpt", default=None)
    ap.add_argument("--opengait-root", default=None)
    ap.add_argument("--subjects", type=int, default=0, help="limit to the first N TEST subjects")
    ap.add_argument("--max-frames", type=int, default=0)
    ap.add_argument("--cache", default=os.path.join(H.DATA, "emb_cache.npz"))
    ap.add_argument("--from-cache", action="store_true", help="skip loading; re-analyse the cache")
    ap.add_argument("--contour-budget", type=int, default=20000)
    ap.add_argument("--win-sweep", default="",
                    help="extra MASK_TEMPORAL_WIN values to evaluate, e.g. 1,3,7,9 "
                         "(5 = the shipped default, always included as `mitigated`)")
    ap.add_argument("--shape-sweep", default="",
                    help="shape-canonicalisation arms as mode@win, e.g. hull@1,ellipse@1,hull@2. "
                         "mode is a config.MASK_SHAPE_MODE value (hull|ellipse|none).")
    ap.add_argument("--arms-json", default="",
                    help="path to a JSON list of general arms: "
                         "[{\"name\":..., \"win\":2, \"cfg\":{EDGE_CONFIG_ATTR: value}}, ...]")
    ap.add_argument("--workers", type=int, default=0,
                    help="processes for the CPU-bound mitigation step (0 = serial). Bit-identical "
                         "either way - verify with --check-parallel.")
    ap.add_argument("--check-parallel", action="store_true",
                    help="assert the parallel mitigation is bit-identical to the serial one on a "
                         "small real batch, then exit")
    ap.add_argument("--ckpt-every", type=int, default=10,
                    help="dump the embedding cache every N subjects (0 = never)")
    ap.add_argument("--tag", default="")
    a = ap.parse_args()
    os.makedirs(OUT_DIR, exist_ok=True)
    specs = []
    if a.arms_json:
        with open(a.arms_json) as f:
            specs = json.load(f)
        seen = set()
        for s in specs:
            assert s["name"] not in seen and s["name"] not in ARMS, f"duplicate arm {s['name']}"
            seen.add(s["name"])
        print(f"[silh] {len(specs)} general arms from {a.arms_json}")

    if a.check_parallel:
        import concurrent.futures as _cf
        tr = H.load_casia_b_silhouettes(a.silh_root, subjects={TEST_IDS.copy().pop()},
                                        max_frames=40)
        base = {k: (v > 127).astype(np.uint8) for k, v in list(tr.items())[:12]}
        with _cf.ProcessPoolExecutor(max_workers=max(2, a.workers or 4)) as ex:
            for spec in (specs or [{"name": "shipped", "win": 2, "cfg": {}}]):
                w, cfg = int(spec.get("win", H.MASK_WIN)), spec.get("cfg", {})
                s = H.mitigate_batch(base, w, cfg, pool=None)
                p = H.mitigate_batch(base, w, cfg, pool=ex)
                same = all(np.array_equal(s[k], p[k]) for k in s)
                print(f"  {spec['name']:24s} serial == parallel: {same}")
                assert same, f"parallel mitigation diverged for {spec['name']}"
        print("[silh] --check-parallel PASSED")
        return

    windows = tuple(int(w) for w in a.win_sweep.split(",") if w.strip()) if a.win_sweep else ()
    def _spec(e):
        """mode[@win][@amp][@head_amp]

        The 4th field enables the HEAD/BODY amplitude split (2026-07-26): the head band runs at
        `head_amp` while the rest of the body runs at `amp`. Omit it for the uniform behaviour.
        "none" means literally no head override, so `displace@2@0.30@none` is a way to write the
        uniform 0.30 control alongside a split arm without ambiguity.
        """
        q = e.split("@")
        def _f(i):
            if len(q) <= i:
                return None
            t = q[i].strip().lower()
            return None if t in ("", "none", "-") else float(t)
        return (q[0].strip(), int(q[1]) if len(q) > 1 else 2, _f(2), _f(3))
    shapes = tuple(_spec(e) for e in a.shape_sweep.split(",") if e.strip()) if a.shape_sweep else ()
    subjects = sorted(TEST_IDS)[:a.subjects] if a.subjects else sorted(TEST_IDS)
    if a.from_cache:
        emb = load_cache(a.cache)
        cov = None
        names = [k for k in emb if emb[k]]
        print(f"[silh] loaded cached embeddings from {a.cache}: arms {names}")
    else:
        if not (a.silh_root and a.ckpt):
            raise SystemExit("--silh-root and --ckpt are required (or use --from-cache)")
        print(f"[silh] streaming {len(subjects)} TEST subjects from {a.silh_root}"
              + (f"; MASK_TEMPORAL_WIN sweep {windows}" if windows else ""))
        pool, ex = None, None
        if a.workers and specs:
            import concurrent.futures as _cf
            ex = _cf.ProcessPoolExecutor(max_workers=a.workers)
            pool = ex
            print(f"[silh] mitigation fanned out over {a.workers} processes "
                  f"(bit-identical to serial; see mitigate_batch)")
        try:
            emb, cov, names = embed_all(a.silh_root, a.ckpt, a.opengait_root, set(subjects),
                                        a.max_frames, a.cache, a.contour_budget, windows,
                                        shapes, specs, pool, a.ckpt_every)
        finally:
            if ex is not None:
                ex.shutdown()

    ids = {k[0] for k in emb["raw"]}
    P.assert_identity_disjoint(TRAIN_IDS, ids)
    print(f"[silh] {len(emb['raw'])} sequences, {len(ids)} subjects; identity-disjoint from the "
          f"{len(TRAIN_IDS)} TRAIN ids the checkpoint saw: OK")

    report = {"adversary": "OpenGait GaitBase_DA (published CASIA-B checkpoint)",
              "is_real_adversary": True, "n_sequences": len(emb["raw"]),
              "n_subjects": len(ids), "subjects": sorted(ids),
              "chance_rank1": P.CHANCE_RANK1_50,
              "mitigation_params": {"MASK_TEMPORAL_WIN": H.MASK_WIN,
                                    "MASK_SIMPLIFY_EPS": H.MASK_EPS},
              "coverage": cov, "arms": {}}

    # ---- GATE: raw arm vs the published baseline, using OpenGait's own metric ----
    pub = H.OpenGaitRecognizer.PUBLISHED
    gate = H.opengait_rank1(emb["raw"])
    deltas = {p: gate[k] * 100 - pub[p] for k, p in
              (("NM#5-6", "NM"), ("BG#1-2", "BG"), ("CL#1-2", "CL"))}
    print("\n=== VALIDATION GATE - raw silhouettes vs published GaitBase (identical view excluded)")
    for k, p in (("NM#5-6", "NM"), ("BG#1-2", "BG"), ("CL#1-2", "CL")):
        print(f"    {p}: reproduced {gate[k]*100:6.2f}   published {pub[p]:5.1f}   "
              f"delta {deltas[p]:+6.2f} pp")
    report["gate"] = {"reproduced": gate, "published": pub, "delta_pp": deltas}
    ok = max(abs(v) for v in deltas.values()) <= 3.0
    report["gate"]["passed"] = bool(ok)
    print(f"    => GATE {'PASSED' if ok else 'FAILED'} (max |delta| "
          f"{max(abs(v) for v in deltas.values()):.2f} pp, tolerance 3.0)")
    if not ok:
        print("    !! The adversary does not reproduce its published baseline. Arm numbers below "
              "are NOT trustworthy and must not enter the ledger until this is resolved.")

    # ---- all arms ----
    for arm in [n for n in names if emb.get(n)]:
        og = H.opengait_rank1(emb[arm])
        flat = {k: v.reshape(-1) for k, v in emb[arm].items()}
        print(f"\n--- arm: {arm} ---")
        print(f"  OpenGait metric rank-1  NM={og['NM#5-6']*100:6.2f}  "
              f"BG={og['BG#1-2']*100:6.2f}  CL={og['CL#1-2']*100:6.2f}")
        full = P.reid_metrics(flat, seed=0)
        report["arms"][arm] = {"opengait_rank1": og, "reid_metrics": full}

    # ---- MASK_TEMPORAL_WIN privacy/cost curve ----
    sweep = sorted({5} | set(windows))
    if len(sweep) > 1:
        print("\n" + "=" * 88)
        print("MASK_TEMPORAL_WIN sweep - privacy vs extra-gray cost (control = raw_bin)")
        print(f"  {'win':>4}  {'NM':>7} {'BG':>7} {'CL':>7}   {'d_NM':>7}   {'area':>6} "
              f"{'IoU':>6}   arm")
        ctrl = report["arms"]["raw_bin"]["opengait_rank1"]["NM#5-6"] * 100
        rows = []
        for w in sweep:
            arm = "mitigated" if w == 5 else f"mit_w{w}"
            if arm not in report["arms"]:
                continue
            o = report["arms"][arm]["opengait_rank1"]
            c = (cov or {}).get(str(w)) or {}
            ar = (c.get("area_ratio_mitigated_over_raw") or {}).get("mean", float("nan"))
            iou = (c.get("iou_rawbin_vs_mitigated") or {}).get("mean", float("nan"))
            rows.append({"win": w, "NM": o["NM#5-6"] * 100, "BG": o["BG#1-2"] * 100,
                         "CL": o["CL#1-2"] * 100, "delta_NM_pp": o["NM#5-6"] * 100 - ctrl,
                         "area_ratio_mean": ar, "iou_mean": iou, "arm": arm})
            print(f"  {w:>4}  {rows[-1]['NM']:7.2f} {rows[-1]['BG']:7.2f} {rows[-1]['CL']:7.2f}"
                  f"   {rows[-1]['delta_NM_pp']:+7.2f}   {ar:6.3f} {iou:6.3f}   {arm}")
        print(f"  {'ctrl':>4}  {ctrl:7.2f}     -     -        0.00    1.000  1.000   raw_bin")
        print("=" * 88)
        report["win_sweep"] = {"control_NM": ctrl, "rows": rows}

    # ---- PRIVACY-PER-AREA ranking (2026-07-31) ----
    # Area is the real currency: it is what the cloud halo / background-fill budget pays (ledger
    # §B.48). An arm is only recommendable if it sits at or below the SHIPPED area ratio, so rank
    # by NM and show the area ratio and the pp-of-NM bought per %-of-area-added beside it.
    report["arm_specs"] = specs
    if cov:
        ctrl = report["arms"]["raw_bin"]["opengait_rank1"]["NM#5-6"] * 100
        ship_area = ((cov.get("mitigated") or cov.get(str(H.MASK_WIN)) or {})
                     .get("area_ratio_mitigated_over_raw") or {}).get("mean")
        # `mitigated` is keyed in `cov` by the window it actually used, and the `--win-sweep`
        # arms by their window number; everything else is keyed by its own arm name.
        ckey = {"mitigated": str(H.MASK_WIN)}
        ckey.update({f"mit_w{w}": str(w) for w in windows})
        rows = []
        for arm in [n for n in names if emb.get(n) and ckey.get(n, n) in cov]:
            o = report["arms"][arm]["opengait_rank1"]
            c = cov[ckey.get(arm, arm)]
            ar = (c.get("area_ratio_mitigated_over_raw") or {}).get("mean", float("nan"))
            nm = o["NM#5-6"] * 100
            rows.append({"arm": arm, "NM": nm, "BG": o["BG#1-2"] * 100,
                         "CL": o["CL#1-2"] * 100, "delta_NM_pp": nm - ctrl,
                         "area_ratio": ar, "iou": (c.get("iou_rawbin_vs_mitigated")
                                                   or {}).get("mean", float("nan")),
                         "pp_per_pct_area": ((ctrl - nm) / ((ar - 1.0) * 100.0)
                                             if ar and ar > 1.0 else float("nan")),
                         "superset_violations_px": c.get("superset_violations_px"),
                         "within_shipped_area_budget": (bool(ar <= ship_area)
                                                        if ship_area else None)})
        rows.sort(key=lambda r: r["NM"])
        print("\n" + "=" * 104)
        print(f"PRIVACY-PER-AREA RANKING  (control raw_bin NM {ctrl:.2f} %"
              + (f", shipped area x{ship_area:.4f})" if ship_area else ")"))
        print(f"  {'NM':>7} {'BG':>7} {'CL':>7} {'dNM':>8} {'area x':>7} {'IoU':>6} "
              f"{'pp/%area':>9} {'viol':>5}  arm")
        for r in rows:
            print(f"  {r['NM']:7.2f} {r['BG']:7.2f} {r['CL']:7.2f} {r['delta_NM_pp']:+8.2f} "
                  f"{r['area_ratio']:7.4f} {r['iou']:6.3f} {r['pp_per_pct_area']:9.3f} "
                  f"{r['superset_violations_px']:5d}  "
                  f"{'' if r['within_shipped_area_budget'] else 'OVER-BUDGET '}{r['arm']}")
        print("=" * 104)
        report["pareto"] = {"control_NM": ctrl, "shipped_area_ratio": ship_area, "rows": rows}

    r0 = report["arms"]["raw"]["opengait_rank1"]["NM#5-6"] * 100
    rb = report["arms"]["raw_bin"]["opengait_rank1"]["NM#5-6"] * 100
    rm = report["arms"]["mitigated"]["opengait_rank1"]["NM#5-6"] * 100
    print("\n" + "=" * 88)
    print(f"SILHOUETTE CHANNEL (NM rank-1, chance {P.CHANCE_RANK1_50*100:.1f}%)")
    print(f"  raw (published domain) {r0:6.2f}%")
    print(f"  raw_bin  (control)     {rb:6.2f}%   <- cost of binarising: {rb-r0:+.2f} pp")
    print(f"  mitigated              {rm:6.2f}%   <- SHIPPED MITIGATION: {rm-rb:+.2f} pp vs control")
    print("=" * 88)
    report["headline"] = {"raw_NM": r0, "raw_bin_NM": rb, "mitigated_NM": rm,
                          "binarise_cost_pp": rb - r0, "mitigation_effect_pp": rm - rb}

    path = os.path.join(OUT_DIR, f"SILHOUETTE_reid{('_' + a.tag) if a.tag else ''}.json")
    with open(path, "w") as f:
        json.dump(report, f, indent=2)
    print(f"\n[silh] wrote {path}")


if __name__ == "__main__":
    main()
