#!/usr/bin/env python3
"""The silhouette re-identification PROTOCOL module.

Vendored from the internal harness that produced the published numbers
(`_e2e/sil_modes_20260807/reid_modes.py`); the protocol functions (`gei`, `metrics`, `evaluate`,
`eer`, `bootstrap_ci`, `dist_l2`, `dist_parts`, `learned`) and the constants (`L_FRAMES`,
`REPEAT`, `BAN`) are unchanged. `class12_silhouette.py` imports the protocol from here so that
protocol identity is guaranteed by IMPORT, not by re-implementation. Only the internal
mask-transform arm registry (`arms.py`, an ablation zoo that is not part of this release) and the
internal cache paths were removed; the standalone `main()` below needs that registry and refuses
to run without it.

PROTOCOL
  * gallery = exactly ONE clip per identity, redrawn REPEAT times
  * CONDITION-MATCHED: all gallery entries from one condition B, probe from condition A != B. The
    framing is then a constant of the gallery and cannot single anyone out.
  * SAME-COLLECTION galleries. The corpus is two capture batches whose silhouette widths do not
    overlap at all (19-29 px vs 36-64 px of a 64-wide canvas); a mixed gallery would let any probe
    discard a whole block of identities on CAMERA, which the threat model forbids. Blocking it
    inside the gallery draw (rather than dropping a batch) keeps all identities in play.
  * the probe's own clip and its whole SOURCE RECORDING are excluded
  * CONTENT-LEVEL duplicates banned, not just filenames
  * equal frame budget per clip, so length and fps cannot be matched on
  * every NULL MEASURED by label permutation through the identical pipeline
"""
import argparse
import io
import json
import os
import sys
import time
from collections import defaultdict

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
LAB = os.path.normpath(os.path.join(HERE, "..", "silhouette"))
CACHE = os.path.join(HERE, "cache")
sys.path.insert(0, HERE)
try:
    from arms import ARMS                                                   # noqa: E402
except ImportError:                          # the internal ablation registry is not shipped
    ARMS = {}

BAN = {"p11_c08_input_video.mp4", "p11_c01_bystander.mp4",
       "p11_c07_input7s.mp4", "p11_c04_1face.mp4"}
L_FRAMES = 70
REPEAT = 12


def load(scope="all"):
    items = []
    for f in sorted(os.listdir(CACHE)):
        if not f.endswith(".npz"):
            continue
        z = np.load(os.path.join(CACHE, f), allow_pickle=True)
        clip = str(z["clip"])
        if clip in BAN:
            continue
        coll = str(z["coll"])
        if scope != "all" and coll != scope:
            continue
        idx = np.linspace(0, len(z["clean"]) - 1, L_FRAMES).astype(int)
        it = dict(ident=str(z["identity"]), clip=clip, source=str(z["source"]),
                  cond=str(z["condition"]), coll=coll)
        for arm in ARMS:
            it[arm] = (z[arm][idx] > 0).astype(np.uint8)
        items.append(it)
    return items


def gei(seq):
    g = seq.astype(np.float32).mean(0).ravel()
    return g / (np.linalg.norm(g) + 1e-12)


def metrics(ranks):
    r = np.array([x[0] for x in ranks], float)
    g = np.array([x[1] for x in ranks], float)
    return dict(n=int(len(r)), gallery=float(g.mean()),
                rank1=100.0 * float((r == 1).mean()),
                rank5=100.0 * float((r <= 5).mean()),
                mAP=100.0 * float((1.0 / r).mean()),
                norm_rank=float(((r - 1) / (g - 1)).mean()))


def evaluate(items, feats, dist, permute=False, seed=0):
    """Condition-matched, one-clip-per-identity, SAME-COLLECTION gallery."""
    rng = np.random.RandomState(seed)
    by = defaultdict(dict)
    for it in items:
        by[(it["coll"], it["cond"])][it["ident"]] = it
    ranks, per = [], defaultdict(list)
    for (cA, A), pool_a in by.items():
        for (cB, B), pool_b in by.items():
            if cB != cA or B == A or len(pool_b) < 4:
                continue
            for pid, probe in pool_a.items():
                if pid not in pool_b:
                    continue
                for _ in range(REPEAT):
                    gal = [g for g in pool_b.values() if g["source"] != probe["source"]]
                    if len(gal) < 4 or not any(g["ident"] == pid for g in gal):
                        continue
                    lab = np.array([g["ident"] for g in gal])
                    if permute:
                        lab = lab.copy(); rng.shuffle(lab)
                        if pid not in lab:
                            continue
                    Gm = np.stack([feats[g["clip"]] for g in gal])
                    d = dist(Gm, feats[probe["clip"]])
                    rk = int(np.where(lab[np.argsort(d)] == pid)[0][0]) + 1
                    ranks.append((rk, len(gal)))
                    per[pid].append(1.0 if rk == 1 else 0.0)
    return ranks, per


def eer(items, feats, dist):
    gen, imp = [], []
    for i, a in enumerate(items):
        for b in items[i + 1:]:
            if a["cond"] == b["cond"] or a["source"] == b["source"] or a["coll"] != b["coll"]:
                continue
            d = float(dist(feats[b["clip"]][None], feats[a["clip"]])[0])
            (gen if a["ident"] == b["ident"] else imp).append(d)
    gen, imp = np.array(gen), np.array(imp)
    if not len(gen) or not len(imp):
        return float("nan"), 0, 0
    th = np.unique(np.concatenate([gen, imp]))
    far = np.array([(imp <= t).mean() for t in th])
    frr = np.array([(gen > t).mean() for t in th])
    k = int(np.argmin(np.abs(far - frr)))
    return 100.0 * float((far[k] + frr[k]) / 2), len(gen), len(imp)


def bootstrap_ci(per, reps=2000, seed=0):
    """95 % CI on rank-1, resampling IDENTITIES (draws are clustered within identity)."""
    rng = np.random.RandomState(seed)
    ids = [k for k, v in per.items() if v]
    if len(ids) < 3:
        return (float("nan"), float("nan"))
    means = np.array([np.mean(per[i]) for i in ids])
    boot = [means[rng.randint(0, len(ids), len(ids))].mean() for _ in range(reps)]
    return (100.0 * float(np.percentile(boot, 2.5)), 100.0 * float(np.percentile(boot, 97.5)))


def dist_l2(G, q):
    return np.linalg.norm(G - q[None], axis=1)


def dist_parts(G, q):
    return np.linalg.norm(G - q[None], axis=2).mean(axis=1)


def learned(items, train_key, test_key, seeds=3):
    """LEARNED attacker, identity-disjoint train/test split. train==test=='<arm>' is TM3."""
    import torch
    import torch.nn as nn
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    ids = sorted({it["ident"] for it in items})
    ntr = max(3, int(round(len(ids) * 0.65)))
    tr_ids, te_ids = set(ids[:ntr]), set(ids[ntr:])
    tr = [it for it in items if it["ident"] in tr_ids]
    te = [it for it in items if it["ident"] in te_ids]
    if len(te_ids) < 3:
        return None
    idmap = {v: i for i, v in enumerate(sorted(tr_ids))}

    def st(it, k):
        idx = np.linspace(0, len(it[k]) - 1, 16).astype(int)
        return it[k][idx].astype(np.float32)

    X = torch.tensor(np.stack([st(t, train_key) for t in tr])).to(dev)
    Y = torch.tensor([idmap[t["ident"]] for t in tr]).to(dev)

    class Net(nn.Module):
        def __init__(s, nc):
            super().__init__()
            s.f = nn.Sequential(nn.Conv2d(16, 32, 3, 2, 1), nn.BatchNorm2d(32), nn.ReLU(),
                                nn.Conv2d(32, 64, 3, 2, 1), nn.BatchNorm2d(64), nn.ReLU(),
                                nn.Conv2d(64, 128, 3, 2, 1), nn.BatchNorm2d(128), nn.ReLU(),
                                nn.AdaptiveAvgPool2d(1), nn.Flatten(), nn.Linear(128, 128))
            s.c = nn.Linear(128, nc)

        def forward(s, x):
            e = s.f(x)
            return e, s.c(e)

    accs, a5, nulls = [], [], []
    for sd in range(seeds):
        torch.manual_seed(sd); np.random.seed(sd)
        net = Net(len(idmap)).to(dev)
        opt = torch.optim.Adam(net.parameters(), 1e-3)
        lo = nn.CrossEntropyLoss()
        net.train()
        for _ in range(300):
            opt.zero_grad(); _, lg = net(X); lo(lg, Y).backward(); opt.step()
        net.eval()
        with torch.no_grad():
            E = net(torch.tensor(np.stack([st(t, test_key) for t in te])).to(dev))[0].cpu().numpy()
        E = E / (np.linalg.norm(E, axis=1, keepdims=True) + 1e-12)
        feats = {t["clip"]: E[i] for i, t in enumerate(te)}
        rk, _ = evaluate(te, feats, dist_l2, seed=sd)
        rkn, _ = evaluate(te, feats, dist_l2, permute=True, seed=100 + sd)
        if rk:
            m = metrics(rk)
            accs.append(m["rank1"]); a5.append(m["rank5"])
        if rkn:
            nulls.append(metrics(rkn)["rank1"])
    if not accs:
        return None
    return dict(rank1=float(np.mean(accs)), rank1_sd=float(np.std(accs)),
                rank5=float(np.mean(a5)), rank5_sd=float(np.std(a5)),
                null_rank1=float(np.mean(nulls)) if nulls else float("nan"),
                seeds=[round(x, 2) for x in accs],
                n_train_ids=len(tr_ids), n_test_ids=len(te_ids),
                train_on=train_key, test_on=test_key)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--scope", default="all", choices=["all", "faces", "pxx"])
    ap.add_argument("--no-learned", action="store_true")
    ap.add_argument("--out", default=None)
    a = ap.parse_args()

    if not ARMS:
        raise SystemExit("reid_modes.main() drives the internal mask-transform ablation registry, "
                         "which is not part of this release. Use class12_silhouette.py, which "
                         "imports the protocol functions from this module.")

    items = load(a.scope)
    ids = sorted({it["ident"] for it in items})
    colls = sorted({it["coll"] for it in items})
    print(f"scope={a.scope}: {len(items)} clips, {len(ids)} identities, collections {colls}")
    print(f"frame budget {L_FRAMES}, gallery = 1 clip/identity, condition- AND collection-matched\n")

    gb = None
    try:
        sys.path.insert(0, LAB)
        import silhouette_harness as H
        gb = H.OpenGaitRecognizer(os.path.join(LAB, "data", "CASIA-B", "Baseline", "GaitBase_DA",
                                               "checkpoints", "GaitBase_DA-60000.pt"))
        print("TM1 frozen adversary: GaitBase_DA-60000.pt loaded\n")
    except Exception as e:
        print(f"TM1 frozen adversary UNAVAILABLE ({type(e).__name__}: {e})\n")

    rows = {}
    for arm in ARMS:
        t0 = time.time()
        r = {}
        F = {it["clip"]: gei(it[arm]) for it in items}
        rk, per = evaluate(items, F, dist_l2)
        rkn, _ = evaluate(items, F, dist_l2, permute=True, seed=1)
        m, mn = metrics(rk), metrics(rkn)
        e, ng, ni = eer(items, F, dist_l2)
        r["GEI"] = dict(**m, null_rank1=mn["rank1"], null_rank5=mn["rank5"],
                        null_norm_rank=mn["norm_rank"], ci95=bootstrap_ci(per),
                        EER=e, eer_pairs=[ng, ni])
        if gb is not None:
            emb = gb.embed({it["clip"]: it[arm] for it in items})
            rk1, per1 = evaluate(items, emb, dist_parts)
            rk1n, _ = evaluate(items, emb, dist_parts, permute=True, seed=1)
            m1, m1n = metrics(rk1), metrics(rk1n)
            r["TM1"] = dict(**m1, null_rank1=m1n["rank1"], null_rank5=m1n["rank5"],
                            null_norm_rank=m1n["norm_rank"], ci95=bootstrap_ci(per1))
        if not a.no_learned:
            lt = learned(items, arm, arm)
            if lt:
                r["TM3" if arm != "clean" else "LEARNED"] = lt
            if arm != "clean":
                ln = learned(items, "clean", arm)
                if ln:
                    r["NAIVE"] = ln
        rows[arm] = r
        g = r["GEI"]
        print(f"{arm:11s} GEI {g['rank1']:6.2f}% (null {g['null_rank1']:5.2f}) "
              f"CI[{g['ci95'][0]:5.1f},{g['ci95'][1]:5.1f}]  "
              f"TM1 {r.get('TM1',{}).get('rank1',float('nan')):6.2f}%  "
              f"TM3 {r.get('TM3',{}).get('rank1',float('nan')):6.2f}%  "
              f"({time.time()-t0:.0f}s)", flush=True)

    rows["_scope"] = dict(scope=a.scope, n_clips=len(items), n_identities=len(ids),
                          collections=colls, frame_budget=L_FRAMES, repeat=REPEAT,
                          banned=sorted(BAN))
    out = a.out or os.path.join(HERE, f"REID_MODES_{a.scope}.json")
    json.dump(rows, io.open(out, "w", encoding="utf-8"), indent=1)
    print(f"\nwrote {out}")


if __name__ == "__main__":
    main()
