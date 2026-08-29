#!/usr/bin/env python3
"""cmc_opengait.py -- rank-1/3/5 under the VALIDATED OpenGait protocol.

WHY THIS EXISTS. `silhouette_harness.opengait_rank1()` takes an argmin, so it can only report
rank-1 -- and rank-1 under that protocol is the number the whole ledger quotes (raw 97.88 %
reproduces published GaitBase 97.6 %, which is what validates the adversary at all).
`protocol.reid_metrics()` DOES report a CMC, but under a DIFFERENT gallery/distance protocol
whose NM rank-1 for the shipped arm is 96.73 %, not 55.03 %. Those two numbers are not
comparable and must never be mixed in one table.

So when a paper table asks for top-1 / top-3 / top-5, the honest answer needs a CMC computed
under the SAME protocol as the headline rank-1. That is what this does: identical gallery
(NM#1-4), probes (NM#5-6 / BG#1-2 / CL#1-2), identical-view exclusion and OpenGait's own
mean-over-16-parts euclidean distance -- but ranking the gallery IDENTITIES instead of taking
the argmin, so rank-k falls out.

Rank of a probe = position of its true identity in the gallery ranked by that identity's BEST
(minimum) distance -- the standard CMC convention for a multi-shot gallery.

    python cmc_opengait.py --cache <emb.npz> --arms raw,raw_bin,mitigated,mit_hull_w2
"""
import argparse
import json
import os
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
sys.path.insert(0, os.path.dirname(HERE))

import silhouette_harness as H                                   # noqa: E402

ANGLES = list(range(0, 181, 18))
OUT_DIR = os.path.join(H.LAB, "reports", "reid")


def key_parse(s):
    return tuple(int(x) for x in s.split("|"))


def load_cache(path):
    z = np.load(path)
    emb = {}
    for name in z.files:
        a, _, ks = name.partition("::")
        emb.setdefault(a, {})[key_parse(ks)] = z[name]
    return emb


def cmc_opengait(embeddings, max_rank=5, angles=ANGLES):
    """Returns {probe: [rank1..rank_max]} averaged exactly like opengait_rank1 does:
    per (gallery-view, probe-view) cell, then identical-view removed, then meaned."""
    gal = {k: v for k, v in embeddings.items() if k[1] == 0 and k[2] <= 4}
    probes = [{k: v for k, v in embeddings.items() if k[1] == 0 and k[2] >= 5},
              {k: v for k, v in embeddings.items() if k[1] == 1},
              {k: v for k, v in embeddings.items() if k[1] == 2}]
    n = len(angles)
    hit = np.zeros((3, n, n, max_rank))
    tot = np.zeros((3, n, n))
    for gi, ga in enumerate(angles):
        gk = [k for k in gal if k[3] == ga]
        if not gk:
            continue
        G = np.stack([gal[k] for k in gk])                       # [N, parts, ch]
        gids = np.array([k[0] for k in gk])
        uids = np.unique(gids)
        for pn, probe in enumerate(probes):
            for pk, pv in probe.items():
                if pk[3] not in angles:
                    continue
                d = np.linalg.norm(G - pv[None], axis=2).mean(axis=1)   # OpenGait distance
                # best distance per gallery IDENTITY, then rank identities
                best = np.array([d[gids == u].min() for u in uids])
                order = uids[np.argsort(best, kind="stable")]
                pos = int(np.where(order == pk[0])[0][0]) if pk[0] in order else max_rank + 1
                pi = angles.index(pk[3])
                for r in range(max_rank):
                    if pos <= r:
                        hit[pn, gi, pi, r] += 1
                tot[pn, gi, pi] += 1
    out = {}
    names = ["NM#5-6", "BG#1-2", "CL#1-2"]
    for i, nm in enumerate(names):
        acc = np.divide(hit[i], tot[i][..., None],
                        out=np.zeros_like(hit[i]), where=tot[i][..., None] > 0)
        for r in range(max_rank):                                # exclude identical view
            acc[..., r] -= np.diag(np.diag(acc[..., r]))
        per = np.sum(acc, 1) / float(len(angles) - 1)            # [probe-view, rank]
        out[nm] = (np.mean(per, 0) * 100).tolist()
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--cache", required=True)
    ap.add_argument("--arms", default="raw,raw_bin,mitigated,mit_hull_w2,mit_ellipse_w2")
    ap.add_argument("--tag", default="20260727")
    a = ap.parse_args()
    emb = load_cache(a.cache)
    arms = [x for x in a.arms.split(",") if x.strip() in emb]
    rep = {"protocol": "OpenGait mean-over-16-parts euclidean, gallery NM#1-4, identical view "
                       "excluded - the SAME protocol as the ledger's headline rank-1",
           "chance_rank1_pct": 2.0, "arms": {}}
    print(f"{'arm':20s} {'probe':9s} {'rank1':>7s} {'rank3':>7s} {'rank5':>7s}")
    for arm in arms:
        c = cmc_opengait(emb[arm])
        rep["arms"][arm] = c
        for nm, v in c.items():
            print(f"{arm:20s} {nm:9s} {v[0]:7.2f} {v[2]:7.2f} {v[4]:7.2f}")
        print()
    p = os.path.join(OUT_DIR, f"SILHOUETTE_CMC_{a.tag}.json")
    json.dump(rep, open(p, "w"), indent=2)
    print(f"wrote {p}")


if __name__ == "__main__":
    main()
