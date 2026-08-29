"""
protocol -- CASIA-B gait re-identification evaluation.

Two views of the same embeddings:
  (A) canonical_casia_b_rank1()  -- verbatim port of GaitGraph/src/evaluate.py's
      _evaluate_casia_b: per-gallery-angle nearest-neighbour, cross-view averaged
      excluding identical view. This is the number to MATCH the published baseline.
  (B) reid_metrics()             -- publication-grade suite (rank-1, rank-5, mAP,
      full CMC, EER) over the standard all-NM#1-4 gallery with same-view exclusion,
      plus bootstrap 95% CIs over probes. "Never rank-1 alone."

Embeddings are {(subject, status, seq, angle): vec}. status: nm=0, bg=1, cl=2.
CHANCE for the 50-id test split: rank-1 = 1/50 = 2.0%, EER = 50%.

Privacy reading: these are ADVERSARY numbers. Anonymization works only when they
COLLAPSE TOWARD CHANCE. A drop from 88% to 30% has anonymized almost nothing.
"""
import numpy as np

ANGLES = list(range(0, 181, 18))            # 11 CASIA-B views
PROBE_DEF = {"NM#5-6": (0, lambda s: s >= 5), "BG#1-2": (1, None), "CL#1-2": (2, None)}
CHANCE_RANK1_50 = 1.0 / 50.0


def assert_identity_disjoint(train_emb_or_ids, test_emb_or_ids):
    """Hard gate: the sets of subject ids in train and test must not overlap."""
    def ids(x):
        return {(k[0] if isinstance(k, tuple) else k) for k in
                (x.keys() if isinstance(x, dict) else x)}
    a, b = ids(train_emb_or_ids), ids(test_emb_or_ids)
    inter = a & b
    assert not inter, f"IDENTITY LEAK: {len(inter)} subjects in both train and test: {sorted(inter)[:10]}"
    return True


# ------------------------------------------------------------------ (A) canonical
def canonical_casia_b_rank1(embeddings):
    """Verbatim GaitGraph protocol. Returns {'NM#5-6','BG#1-2','CL#1-2','mean'} rank-1."""
    gallery = {k: v for k, v in embeddings.items() if k[1] == 0 and k[2] <= 4}
    gpa = {a: {k: v for k, v in gallery.items() if k[3] == a} for a in ANGLES}
    probe_nm = {k: v for k, v in embeddings.items() if k[1] == 0 and k[2] >= 5}
    probe_bg = {k: v for k, v in embeddings.items() if k[1] == 1}
    probe_cl = {k: v for k, v in embeddings.items() if k[1] == 2}

    correct = np.zeros((3, 11, 11)); total = np.zeros((3, 11, 11))
    for ga in ANGLES:
        g_emb = np.array(list(gpa[ga].values())); g_keys = list(gpa[ga].keys())
        if len(g_emb) == 0:
            continue
        gpos = ga // 18
        for pnum, probe in enumerate([probe_nm, probe_bg, probe_cl]):
            for tgt, emb in probe.items():
                ppos = tgt[3] // 18
                d = np.linalg.norm(g_emb - emb, ord=2, axis=1)
                if g_keys[int(np.argmin(d))][0] == tgt[0]:
                    correct[pnum, gpos, ppos] += 1
                total[pnum, gpos, ppos] += 1
    acc = np.divide(correct, total, out=np.zeros_like(correct), where=total > 0)
    for i in range(3):
        acc[i] -= np.diag(np.diag(acc[i]))          # exclude same view
    acc_flat = np.sum(acc, 1) / 10.0
    sub = np.mean(acc_flat, 1)
    return {"NM#5-6": float(sub[0]), "BG#1-2": float(sub[1]),
            "CL#1-2": float(sub[2]), "mean": float(np.mean(sub))}


# ------------------------------------------------------------- (B) full reid suite
def _gallery_probe(embeddings):
    gal = [(k, v) for k, v in embeddings.items() if k[1] == 0 and k[2] <= 4]
    probes = {name: [(k, v) for k, v in embeddings.items()
                     if k[1] == st and (f is None or f(k[2]))]
              for name, (st, f) in PROBE_DEF.items()}
    return gal, probes


def _rank_ap_for_probe(pk, pv, g_ids, g_views, g_emb):
    """Cross-view: exclude gallery of same view as probe. Returns (rank_of_first_hit
    or None, average_precision, genuine_dists, impostor_dists)."""
    m = g_views != pk[3]
    if not m.any():
        return None, None, np.array([]), np.array([])
    ids, emb = g_ids[m], g_emb[m]
    d = np.linalg.norm(emb - pv, axis=1)
    order = np.argsort(d, kind="stable")
    hit = (ids[order] == pk[0])
    first = np.argmax(hit) if hit.any() else None
    if hit.any():
        ranks = np.where(hit)[0]
        prec = (np.arange(1, len(ranks) + 1)) / (ranks + 1.0)
        ap = float(prec.mean())
    else:
        ap = 0.0
    gd = d[ids == pk[0]]; imd = d[ids != pk[0]]
    return (int(first) if first is not None else None), ap, gd, imd


def _eer(genuine, impostor):
    if len(genuine) == 0 or len(impostor) == 0:
        return float("nan")
    thr = np.unique(np.concatenate([genuine, impostor]))
    thr = np.linspace(thr.min(), thr.max(), 512)
    far = np.array([(impostor <= t).mean() for t in thr])   # impostor accepted
    frr = np.array([(genuine > t).mean() for t in thr])     # genuine rejected
    i = np.argmin(np.abs(far - frr))
    return float((far[i] + frr[i]) / 2.0)


def reid_metrics(embeddings, max_rank=20, bootstrap=1000, seed=0, verbose=True):
    """Standard all-gallery reID metrics with same-view exclusion + bootstrap CIs.
    Returns nested dict per probe-type and 'overall' with rank1/rank5/mAP/CMC/EER."""
    gal, probes = _gallery_probe(embeddings)
    g_ids = np.array([k[0] for k, _ in gal]); g_views = np.array([k[3] for k, _ in gal])
    g_emb = np.array([v for _, v in gal])
    rng = np.random.RandomState(seed)
    results = {}
    all_ranks, all_aps = [], []
    for name, plist in probes.items():
        ranks, aps, gds, imds = [], [], [], []
        for pk, pv in plist:
            r, ap, gd, imd = _rank_ap_for_probe(pk, pv, g_ids, g_views, g_emb)
            if ap is None:
                continue
            ranks.append(r); aps.append(ap); gds.append(gd); imds.append(imd)
        ranks = np.array([r if r is not None else max_rank + 999 for r in ranks])
        aps = np.array(aps)
        cmc = np.array([(ranks <= k).mean() for k in range(max_rank)])
        gd = np.concatenate(gds) if gds else np.array([])
        imd = np.concatenate(imds) if imds else np.array([])

        def _stat(idx):
            rr, aa = ranks[idx], aps[idx]
            return (rr <= 0).mean(), (rr <= 4).mean(), aa.mean()
        n = len(ranks)
        if bootstrap and n:
            boot = np.array([_stat(rng.randint(0, n, n)) for _ in range(bootstrap)])
            lo, hi = np.percentile(boot, [2.5, 97.5], axis=0)
        else:
            lo = hi = [float("nan")] * 3
        results[name] = {
            "n_probes": int(n),
            "rank1": float((ranks <= 0).mean()), "rank1_ci": [float(lo[0]), float(hi[0])],
            "rank5": float((ranks <= 4).mean()), "rank5_ci": [float(lo[1]), float(hi[1])],
            "mAP": float(aps.mean()), "mAP_ci": [float(lo[2]), float(hi[2])],
            "EER": _eer(gd, imd), "CMC": cmc.tolist(),
        }
        all_ranks.append(ranks); all_aps.append(aps)
    ar = np.concatenate(all_ranks); aa = np.concatenate(all_aps)
    results["overall"] = {
        "n_probes": int(len(ar)),
        "rank1": float((ar <= 0).mean()), "rank5": float((ar <= 4).mean()),
        "mAP": float(aa.mean()),
        "CMC": [float((ar <= k).mean()) for k in range(max_rank)],
    }
    results["_chance"] = {"rank1": CHANCE_RANK1_50, "EER": 0.5}
    if verbose:
        print("  probe      n   rank1[95%CI]         rank5    mAP      EER")
        for name in list(PROBE_DEF) + ["overall"]:
            r = results[name]
            ci = r.get("rank1_ci", [float('nan')] * 2)
            print(f"  {name:9s} {r['n_probes']:4d}  {r['rank1']*100:5.1f} "
                  f"[{ci[0]*100:4.1f},{ci[1]*100:4.1f}]   {r['rank5']*100:5.1f}   "
                  f"{r['mAP']*100:5.1f}   {r.get('EER', float('nan'))*100:5.1f}")
        print(f"  (chance rank-1 = {CHANCE_RANK1_50*100:.1f}%, EER = 50.0%)")
    return results
