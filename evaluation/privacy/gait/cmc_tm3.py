#!/usr/bin/env python3
"""cmc_tm3.py -- rank-1/3/5 for the POSE/GAIT channel under the SAME protocol as §A.2c's rank-1.

`protocol.canonical_casia_b_rank1()` takes an argmin, so it only yields rank-1, and that rank-1 is
what the ledger quotes. A paper table asking for top-1/top-3/top-5 therefore needs a CMC computed
under the identical gallery/probe/view rules and distance -- not `reid_metrics`, which uses a
different protocol and would silently report a different rank-1 for the same arm.

Loads each saved TM3 adversary checkpoint, re-embeds the anonymised test split exactly as
tm3_retrain.embed_eval does, and ranks gallery IDENTITIES by their best (minimum) distance.

    python cmc_tm3.py --configs L4-bin,L4-bin-a1410,L4-bin-a0 --seed-tag s0
"""
import argparse
import json
import os
import sys

import numpy as np
import torch

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
sys.path.insert(0, os.path.join(HERE, "vendor", "GaitGraph", "src"))

from casia_loader import load_tracklets                          # noqa: E402
import protocol as P                                             # noqa: E402
import anon_adapter as A                                         # noqa: E402
import tm3_retrain as T                                          # noqa: E402
from datasets.graph import Graph                                 # noqa: E402

DATA = os.path.join(HERE, "data")
OUT_DIR = os.path.join(HERE, "..", "reports", "reid")
ANGLES = P.ANGLES


def cmc(embeddings, max_rank=5):
    gal = {k: v for k, v in embeddings.items() if k[1] == 0 and k[2] <= 4}
    probes = [{k: v for k, v in embeddings.items() if k[1] == 0 and k[2] >= 5},
              {k: v for k, v in embeddings.items() if k[1] == 1},
              {k: v for k, v in embeddings.items() if k[1] == 2}]
    hit = np.zeros((3, 11, 11, max_rank)); tot = np.zeros((3, 11, 11))
    for ga in ANGLES:
        gk = [k for k in gal if k[3] == ga]
        if not gk:
            continue
        G = np.stack([gal[k] for k in gk]); gids = np.array([k[0] for k in gk])
        uids = np.unique(gids); gpos = ga // 18
        for pn, probe in enumerate(probes):
            for tgt, e in probe.items():
                d = np.linalg.norm(G - e, ord=2, axis=1)
                best = np.array([d[gids == u].min() for u in uids])
                order = uids[np.argsort(best, kind="stable")]
                pos = int(np.where(order == tgt[0])[0][0]) if tgt[0] in order else max_rank + 1
                ppos = tgt[3] // 18
                for r in range(max_rank):
                    if pos <= r:
                        hit[pn, gpos, ppos, r] += 1
                tot[pn, gpos, ppos] += 1
    out = {}
    for i, nm in enumerate(["NM#5-6", "BG#1-2", "CL#1-2"]):
        acc = np.divide(hit[i], tot[i][..., None],
                        out=np.zeros_like(hit[i]), where=tot[i][..., None] > 0)
        for r in range(max_rank):
            acc[..., r] -= np.diag(np.diag(acc[..., r]))
        out[nm] = (np.mean(np.sum(acc, 0) / 10.0, 0) * 100).tolist()
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--configs", default="L4-bin,L4-bin-a1410,L4-bin-a0")
    ap.add_argument("--seed-tag", default="s0")
    ap.add_argument("--seq_len", type=int, default=60)
    a = ap.parse_args()
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    graph = Graph("coco")
    train = load_tracklets(os.path.join(DATA, "casia-b_pose_train_valid.csv"),
                           split="train_valid", min_frames=a.seq_len + 1)
    test = load_tracklets(os.path.join(DATA, "casia-b_pose_test.csv"),
                          split="test", min_frames=a.seq_len + 1)
    template = A.build_template(train)
    rep = {"protocol": "GaitGraph canonical CASIA-B, gallery NM#1-4, identical view excluded - "
                       "the SAME protocol as §A.2c's rank-1", "chance_rank1_pct": 2.0,
           "seed_tag": a.seed_tag, "arms": {}}
    print(f"{'config':16s} {'probe':9s} {'rank1':>7s} {'rank3':>7s} {'rank5':>7s}")
    for cfg_name in [c for c in a.configs.split(",") if c.strip()]:
        ck = os.path.join(DATA, f"tm3_adv_{cfg_name}_{a.seed_tag}.pth")
        if not os.path.exists(ck):
            print(f"{cfg_name}: no checkpoint {os.path.basename(ck)}"); continue
        cfg = T.CONFIGS[cfg_name]
        anon_test = A.transform_v2(test, cfg, template, seed_mode="per_sequence")
        model = T.build_model(graph, dev)
        model.load_state_dict(torch.load(ck, map_location=dev)["model"])
        emb = T.embed_eval(model, anon_test, graph, dev, a.seq_len)
        c = cmc(emb)
        rep["arms"][cfg_name] = {"cfg": cfg, "cmc": c}
        for nm, v in c.items():
            print(f"{cfg_name:16s} {nm:9s} {v[0]:7.2f} {v[2]:7.2f} {v[4]:7.2f}")
        print()
        del model
        torch.cuda.empty_cache()
    p = os.path.join(OUT_DIR, f"TM3_CMC_{a.seed_tag}_20260727.json")
    json.dump(rep, open(p, "w"), indent=2)
    print(f"wrote {p}")


if __name__ == "__main__":
    main()
