#!/usr/bin/env python3
"""CLASS 5: a BOX-NATIVE adversary -- the attacker `bbox` was never priced against (W9).

THE GAP THIS CLOSES
-------------------
Classes 1 and 2 consume a (T,64,64) SIZE-NORMALISED silhouette. The shipped defence replaces every
person with a filled rectangle, so after normalisation a defended clip is a WHITE SQUARE. That is
not a figure of speech: `class12_silhouette.py --tiebreak` documents that 10/103 MIRAGE clips
collapse to bit-identical GEIs. An attacker who normalises is measuring nothing, and both published
silhouette adversaries normalise.

But the released mask still carries the rectangle's SIZE, and size is identity-bearing -- a tall
person emits a tall box. This adversary reads exactly that, in native pixels, with no normalisation
anywhere.

PROTOCOL
--------
Scoring is `reid_modes.evaluate` / `.metrics` / `.bootstrap_ci` IMPORTED, not reimplemented, so the
gallery construction (condition-matched, same-collection, one clip per identity, R4 same-source
exclusion) and the permutation null are bit-identical to the published Class 1/2 numbers. The BAN
list of content-duplicate sources is honoured. The only thing that differs is the feature.

TWO FEATURE SETS, because they answer different questions
---------------------------------------------------------
  full       every feature, including absolute pixel size. This is what a real attacker holding the
             released masks can compute, so it is the honest threat number.
  scalefree  absolute size dropped; only aspect, relative variability and gait periodicity remain.
             Apparent size confounds stature with camera distance, so this is the conservative
             lower bound: whatever `scalefree` finds cannot be explained by the subject standing
             closer to the lens.

Reporting is by LIFT OVER EACH ARM'S OWN MEASURED NULL (PROTOCOL.md, ledger A.6o), never by raw
accuracy.

  python attack_boxnative.py --boxes boxes.npz --manifest corpus_10fps.json --reps 40
"""
import argparse
import json
import os
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
APP = os.path.normpath(os.path.join(HERE, "..", "appearance"))
sys.path.insert(0, APP)

import reid_modes as RM                                            # noqa: E402

ARMS = ("raw", "mirage")
FPS = 10.0


def collection(relpath):
    """Which capture COLLECTION a clip belongs to -- verbatim from class12_silhouette.py:35."""
    return "faces" if relpath.split("/")[0] in ("Movement of Faces", "Num of Faces",
                                                "Size of Faces") else "pxx"


def _dominant(sig, fps=FPS, lo_hz=0.5, hi_hz=3.0):
    """Dominant frequency in the human gait band, and its share of detrended power.

    Walking widens and narrows the bounding box once per stride, so the box width carries the
    cadence even when the shape inside it is gone. Returned as (hz, relative power); a signal with
    no periodic content lands near (0, 0) rather than picking up a spurious peak, because the
    search is restricted to the band and the power is expressed as a FRACTION of total.
    """
    x = np.asarray(sig, np.float64)
    if len(x) < 16:
        return 0.0, 0.0
    x = x - np.polyval(np.polyfit(np.arange(len(x)), x, 1), np.arange(len(x)))   # detrend
    if x.std() < 1e-9:
        return 0.0, 0.0
    w = np.hanning(len(x))
    p = np.abs(np.fft.rfft(x * w)) ** 2
    f = np.fft.rfftfreq(len(x), 1.0 / fps)
    band = (f >= lo_hz) & (f <= hi_hz)
    if not band.any() or p[1:].sum() <= 0:
        return 0.0, 0.0
    k = int(np.argmax(np.where(band, p, 0.0)))
    return float(f[k]), float(p[k] / p[1:].sum())


# Feature layout. The split is load-bearing: SCALE_IDX are the features that carry absolute pixel
# size, and `scalefree` is defined by deleting exactly these.
FEATS = ["med_h", "p90_h", "med_w", "sqrt_area",              # scale (indices 0-3)
         "aspect", "aspect_iqr",                              # shape
         "cv_h", "cv_w",                                      # relative variability
         "gait_hz", "gait_pow",                               # cadence from box width
         "bob", "drift"]                                      # centroid dynamics
SCALE_IDX = [0, 1, 2, 3]


def featurise(box):
    """(T,4) native-pixel boxes -> one feature vector. No normalisation of any kind."""
    x0, y0, x1, y1 = box[:, 0], box[:, 1], box[:, 2], box[:, 3]
    h, w = np.maximum(y1 - y0, 1.0), np.maximum(x1 - x0, 1.0)
    cx, cy = (x0 + x1) / 2.0, (y0 + y1) / 2.0
    asp = w / h
    mh = float(np.median(h))
    hz, pw = _dominant(w)
    return np.array([
        mh,
        float(np.percentile(h, 90)),
        float(np.median(w)),
        float(np.sqrt(np.median(h * w))),
        float(np.median(asp)),
        float(np.subtract(*np.percentile(asp, [75, 25]))),
        float(h.std() / mh),
        float(w.std() / max(np.median(w), 1.0)),
        hz,
        pw,
        float(cy.std() / mh),
        float(np.abs(np.diff(cx)).mean() / mh) if len(cx) > 1 else 0.0,
    ], np.float64)


def load_items(npz, manifest):
    z = np.load(npz, allow_pickle=True)
    man = {m["clip"]: m for m in json.load(open(manifest))}
    items, raw_feats = [], {a: {} for a in ARMS}
    for c in z["clips"]:
        clip = str(c)
        m = man[clip]
        if m["source_file"] in RM.BAN:
            continue
        ok = True
        for arm in ARMS:
            b = z[clip + "__" + arm]
            if len(b) < 24:
                ok = False
                break
            raw_feats[arm][clip] = featurise(b)
        if ok:
            items.append(dict(ident=m["identity"], clip=clip, source=m["source_file"],
                              cond=m["condition"], coll=collection(m["source_relpath"])))
    keep = {it["clip"] for it in items}
    for arm in ARMS:
        raw_feats[arm] = {k: v for k, v in raw_feats[arm].items() if k in keep}
    print("[boxnative] scored clips: %d  identities: %d"
          % (len(items), len({i["ident"] for i in items})))
    return items, raw_feats


def standardise(feats, idx):
    """Select feature columns and z-score them ACROSS THE CORPUS.

    Without this the L2 distance is dominated by whichever feature happens to be measured in the
    largest units (pixels of height, ~10^3) and the dynamics terms (~10^-1) contribute nothing.
    Z-scoring is unsupervised -- it never sees an identity label -- so it leaks nothing to the
    attacker that they could not compute themselves from the released masks.
    """
    clips = sorted(feats)
    M = np.stack([feats[c] for c in clips])[:, idx]
    sd = M.std(0)
    sd[sd < 1e-12] = 1.0
    M = (M - M.mean(0)) / sd
    return {c: M[i] for i, c in enumerate(clips)}


def run(items, feats, idx, reps, tag):
    out = {}
    per_arm_null = {}
    for arm in ARMS:
        F = standardise(feats[arm], idx)
        rk, per = RM.evaluate(items, F, RM.dist_l2)
        m = RM.metrics(rk)
        lo, hi = RM.bootstrap_ci(per)
        n1, n5 = [], []
        for r in range(reps):
            nrk, _ = RM.evaluate(items, F, RM.dist_l2, permute=True, seed=1000 + r)
            nm = RM.metrics(nrk)
            n1.append(nm["rank1"])
            n5.append(nm["rank5"])
        null1, null5 = float(np.mean(n1)), float(np.mean(n5))
        sem1 = float(np.std(n1) / np.sqrt(len(n1)))
        per_arm_null[arm] = null1
        out[arm] = dict(rank1=m["rank1"], rank5=m["rank5"], ci95=[lo, hi],
                        null_rank1=null1, null_rank5=null5, null_sem1=sem1,
                        lift_rank1=m["rank1"] - null1, lift_rank5=m["rank5"] - null5,
                        n_draws=len(rk))
        print("[%s] %-7s rank1 %6.2f  (null %5.2f +/- %.2f)  LIFT %+7.2f pp   "
              "rank5 %6.2f  LIFT %+7.2f pp   n=%d"
              % (tag, arm, m["rank1"], null1, sem1, m["rank1"] - null1,
                 m["rank5"], m["rank5"] - null5, len(rk)))
    if out["raw"]["lift_rank1"] > 1e-9:
        frac = out["mirage"]["lift_rank1"] / out["raw"]["lift_rank1"]
        out["_removed_frac_rank1"] = 1.0 - frac
        print("[%s] defence removes %.1f %% of the available rank-1 lift "
              "(raw lift %+.2f -> mirage lift %+.2f)"
              % (tag, 100.0 * (1.0 - frac), out["raw"]["lift_rank1"],
                 out["mirage"]["lift_rank1"]))
    spread = abs(per_arm_null["raw"] - per_arm_null["mirage"])
    print("[%s] per-arm null spread %.3f pp (pooling sanity; nulls are used PER ARM here)"
          % (tag, spread))
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--boxes", default=os.path.join(HERE, "boxes.npz"))
    ap.add_argument("--manifest", required=True, help="corpus_10fps.json")
    ap.add_argument("--reps", type=int, default=40)
    ap.add_argument("--out", default=os.path.join(HERE, "BOXNATIVE_RESULTS.json"))
    a = ap.parse_args()

    items, feats = load_items(a.boxes, a.manifest)
    res = {"_meta": {"features": FEATS, "scale_idx": SCALE_IDX, "reps": a.reps,
                     "n_clips": len(items),
                     "n_identities": len({i["ident"] for i in items}),
                     "scoring": "reid_modes.evaluate/metrics/bootstrap_ci, imported",
                     "gallery": "condition-matched, same-collection, one clip per identity, "
                                "same-source excluded (R4)",
                     "null": "measured by label permutation through the identical pipeline"}}
    res["full"] = run(items, feats, list(range(len(FEATS))), a.reps, "full")
    print()
    res["scalefree"] = run(items, feats,
                           [i for i in range(len(FEATS)) if i not in SCALE_IDX],
                           a.reps, "scalefree")

    # Which single feature carries it? Reported because "the box leaks" is not actionable, but
    # "the box leaks HEIGHT" names the knob to turn.
    print("\n[per-feature] mirage arm, each feature alone:")
    singles = {}
    for i, name in enumerate(FEATS):
        F = standardise(feats["mirage"], [i])
        rk, _ = RM.evaluate(items, F, RM.dist_l2)
        m = RM.metrics(rk)
        singles[name] = m["rank1"]
    for name, v in sorted(singles.items(), key=lambda kv: -kv[1]):
        print("    %-10s rank1 %6.2f" % (name, v))
    res["per_feature_mirage_rank1"] = singles

    json.dump(res, open(a.out, "w"), indent=2)
    print("\nwrote " + a.out)


if __name__ == "__main__":
    main()
