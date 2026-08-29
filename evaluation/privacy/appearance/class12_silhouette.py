#!/usr/bin/env python3
"""CLASS 1 (GEI + nearest neighbour) and CLASS 2 (frozen GaitBase) over the defence arms.

This is the script behind the paper's appearance-based re-identification table. Protocol identity
is guaranteed by IMPORT, not by re-implementation: `gei`, `evaluate`, `metrics`, `dist_l2`,
`dist_parts`, `bootstrap_ci`, `L_FRAMES`, `REPEAT` and the content-duplicate `BAN` list all come
from the vendored `reid_modes.py` beside this file, the module that produced the published
numbers. The only thing that changes between arms is where the (T,64,64) silhouettes come from:
the common extractor (`extract_arm.py`) run on what each defence actually releases.

Arms (build them with extract_arm.py first):
  raw     the undefended silhouette of the real person (positive control)
  mirage  what MIRAGE releases: the bounding-box mask emitted by the shipped silhouette
          mitigation (the paper's "MIRAGE bounding box" row)
  dp2     DeepPrivacy2's released video, silhouetted by the common extractor

Nulls are MEASURED by label permutation through the identical pipeline, pooled over --reps
repetitions and over all arms, because after permutation the hit rate is fixed by gallery
composition and cannot depend on the arm. The per-arm nulls that justify pooling are recorded
too, and pooling is flagged if they spread beyond a few SEM.

  python class12_silhouette.py --arms raw,mirage,dp2 --reps 40
"""
import argparse
import json
import os
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
LAB = os.path.normpath(os.path.join(HERE, "..", "silhouette"))
sys.path.insert(0, HERE)
sys.path.insert(0, LAB)

import reid_modes as RM                                            # noqa: E402


def collection(relpath):
    """Which capture COLLECTION a clip belongs to (verbatim from the internal cache builder)."""
    return "faces" if relpath.split("/")[0] in ("Movement of Faces", "Num of Faces",
                                                "Size of Faces") else "pxx"


def load_items(arms, manifest, arms_dir):
    man = {m["clip"]: m for m in json.load(open(manifest))}
    per_arm = {}
    for arm in arms:
        d = os.path.join(arms_dir, arm)
        if not os.path.isdir(d):
            raise SystemExit(f"arm directory missing: {d}  (build it with extract_arm.py)")
        per_arm[arm] = {f[:-4]: os.path.join(d, f)
                        for f in sorted(os.listdir(d)) if f.endswith(".npz")}
    common = set.intersection(*[set(v) for v in per_arm.values()])
    print(f"[class12] clips present in ALL arms: {len(common)} "
          f"(per-arm: {[(a, len(v)) for a, v in per_arm.items()]})")

    items = []
    for clip in sorted(common):
        m = man[clip]
        if m["source_file"] in RM.BAN:
            continue
        it = dict(ident=m["identity"], clip=clip, source=m["source_file"],
                  cond=m["condition"], coll=collection(m["source_relpath"]))
        ok = True
        for arm in arms:
            z = np.load(per_arm[arm][clip], allow_pickle=True)
            s = z["sil"]
            if len(s) < 24:
                ok = False; break
            idx = np.linspace(0, len(s) - 1, RM.L_FRAMES).astype(int)
            it[arm] = (s[idx] > 0).astype(np.uint8)
        if ok:
            items.append(it)
    print(f"[class12] scored clips: {len(items)}  identities: {len({i['ident'] for i in items})}")
    return items


def pooled_null(items, arms, feat_fn, dist, reps, tag, per_arm_out=None):
    """Pooled null, PLUS the per-arm nulls that justify pooling it.

    Pooling across arms rests on one claim: after label permutation the hit rate is fixed by
    gallery composition and CANNOT depend on the arm. The whole study inherits that; every lift,
    and therefore every "% of lift removed", is measured against this one number. The per-arm
    nulls are recorded and their spread printed: if they differ by more than a few SEM, pooling
    is wrong and every arm needs its own null. This only ADDS reporting; the returned pooled
    values are unchanged by it.
    """
    vals1, vals5, by_arm = [], [], {}
    for arm in arms:
        F = feat_fn(items, arm)
        a1, a5 = [], []
        for r in range(reps):
            rk, _ = RM.evaluate(items, F, dist, permute=True, seed=1000 + r)
            m = RM.metrics(rk)
            a1.append(m["rank1"]); a5.append(m["rank5"])
        by_arm[arm] = {"rank1": float(np.mean(a1)), "rank5": float(np.mean(a5)),
                       "sem1": float(np.std(a1) / np.sqrt(len(a1)))}
        vals1 += a1; vals5 += a5
    n1, n5 = float(np.mean(vals1)), float(np.mean(vals5))
    s1 = float(np.std(vals1) / np.sqrt(len(vals1)))
    s5 = float(np.std(vals5) / np.sqrt(len(vals5)))
    print(f"[class12] {tag} POOLED NULL over {len(vals1)} draws "
          f"({len(arms)} arms x {reps} reps): rank1 {n1:.3f} +/- {s1:.3f}   "
          f"rank5 {n5:.3f} +/- {s5:.3f}")
    spread = max(v["rank1"] for v in by_arm.values()) - min(v["rank1"] for v in by_arm.values())
    flag = "OK" if spread <= 6 * s1 else "!! POOLING ASSUMPTION VIOLATED"
    print(f"[class12] {tag} per-arm null rank1: "
          + "  ".join(f"{a.split('_v')[0]} {v['rank1']:.3f}" for a, v in by_arm.items()))
    print(f"[class12] {tag} per-arm null SPREAD {spread:.3f} pp vs pooled SEM {s1:.3f} -> {flag}")
    if per_arm_out is not None:
        per_arm_out.update({"per_arm_null": by_arm, "spread_rank1": spread,
                            "pooled_sem1": s1, "pooling_ok": bool(spread <= 6 * s1)})
    return n1, n5, s1, s5


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--arms", default="raw,mirage,dp2")
    ap.add_argument("--reps", type=int, default=40)
    ap.add_argument("--no-gaitbase", action="store_true")
    ap.add_argument("--manifest", default=os.path.join(HERE, "manifests", "corpus_10fps.json"),
                    help="corpus manifest written by make_corpus_manifest.py")
    ap.add_argument("--arms-dir", default=os.path.join(HERE, "arms"),
                    help="directory holding one subdirectory of npz files per arm")
    ap.add_argument("--ckpt", default=os.path.join(LAB, "data", "CASIA-B", "Baseline",
                                                   "GaitBase_DA", "checkpoints",
                                                   "GaitBase_DA-60000.pt"),
                    help="frozen GaitBase checkpoint for Class 2")
    ap.add_argument("--tiebreak", default="argsort", choices=["argsort", "random"],
                    help="MIRAGE's bounding-box arm collapses some clips to solid-white 64x64 "
                         "squares whose GEIs are bit-identical, creating exact distance ties. "
                         "np.argsort is stable, so ties resolve ALPHABETICALLY BY IDENTITY, a "
                         "determinism artefact that touches one arm only. 'random' breaks them "
                         "uniformly (measured: the artefact flatters the ATTACKER, so argsort is "
                         "the conservative choice and the published one).")
    ap.add_argument("--tiebreak-seed", type=int, default=20260823,
                    help="noise seed for --tiebreak random; vary it to get a sd")
    ap.add_argument("--out", default=None)
    a = ap.parse_args()
    arms = [x.strip() for x in a.arms.split(",") if x.strip()]
    items = load_items(arms, a.manifest, a.arms_dir)

    # --- tie-breaking policy -------------------------------------------------------------
    # `--tiebreak random` adds uniform noise of 1e-9, far below any genuine distance gap, which
    # leaves every real ordering untouched and breaks exact ties uniformly at random.
    _tb_rng = np.random.RandomState(a.tiebreak_seed)

    def _rand_tie(dist):
        def f(G, q):
            d = np.asarray(dist(G, q), dtype=np.float64)
            return d + _tb_rng.uniform(0.0, 1e-9, size=d.shape)
        return f

    wrap = _rand_tie if a.tiebreak == "random" else (lambda d: d)
    D_L2, D_PARTS = wrap(RM.dist_l2), wrap(RM.dist_parts)
    print(f"[class12] tie-breaking: {a.tiebreak}")
    res = {"_scope": {"tiebreak": a.tiebreak, "n_clips": len(items),
                      "n_identities": len({i["ident"] for i in items}),
                      "arms": arms, "null_reps_per_arm": a.reps,
                      "frame_budget": RM.L_FRAMES, "repeat": RM.REPEAT}}

    # ---------------- CLASS 1 : GEI + nearest neighbour ----------------
    gei_feats = lambda its, arm: {i["clip"]: RM.gei(i[arm]) for i in its}          # noqa: E731
    _pa1 = {}
    n1, n5, s1, s5 = pooled_null(items, arms, gei_feats, D_L2, a.reps, "CLASS1 GEI", _pa1)
    res["class1"] = {"null_rank1": n1, "null_rank5": n5, "null_sem1": s1, "null_sem5": s5,
                     **_pa1, "arms": {}}
    for arm in arms:
        F = gei_feats(items, arm)
        rk, per = RM.evaluate(items, F, D_L2)
        m = RM.metrics(rk)
        res["class1"]["arms"][arm] = dict(**m, ci95=RM.bootstrap_ci(per),
                                          lift_rank1=m["rank1"] - n1, lift_rank5=m["rank5"] - n5)
        print(f"[class12] CLASS1 {arm:8s} r1 {m['rank1']:6.2f} (lift {m['rank1']-n1:+6.2f})   "
              f"r5 {m['rank5']:6.2f} (lift {m['rank5']-n5:+6.2f})   n={m['n']} gal={m['gallery']:.2f}")

    # ---------------- CLASS 2 : frozen GaitBase (OpenGait GaitBase_DA) ----------------
    if not a.no_gaitbase:
        import silhouette_harness as H
        gb = H.OpenGaitRecognizer(a.ckpt)
        cache = {}

        def gb_feats(its, arm):
            if arm not in cache:
                cache[arm] = gb.embed({i["clip"]: i[arm] for i in its})
            return cache[arm]

        _pa2 = {}
        n1b, n5b, s1b, s5b = pooled_null(items, arms, gb_feats, D_PARTS, a.reps,
                                         "CLASS2 GaitBase", _pa2)
        res["class2"] = {"null_rank1": n1b, "null_rank5": n5b, "null_sem1": s1b,
                         "null_sem5": s5b, **_pa2, "arms": {}}
        for arm in arms:
            F = gb_feats(items, arm)
            rk, per = RM.evaluate(items, F, D_PARTS)
            m = RM.metrics(rk)
            res["class2"]["arms"][arm] = dict(**m, ci95=RM.bootstrap_ci(per),
                                              lift_rank1=m["rank1"] - n1b,
                                              lift_rank5=m["rank5"] - n5b)
            print(f"[class12] CLASS2 {arm:8s} r1 {m['rank1']:6.2f} (lift {m['rank1']-n1b:+6.2f})   "
                  f"r5 {m['rank5']:6.2f} (lift {m['rank5']-n5b:+6.2f})")

    out = a.out or os.path.join(HERE, "reports", "CLASS12.json")
    os.makedirs(os.path.dirname(out), exist_ok=True)
    json.dump(res, open(out, "w"), indent=1)
    print(f"[class12] wrote {out}")


if __name__ == "__main__":
    main()
