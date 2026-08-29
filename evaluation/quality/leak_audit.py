#!/usr/bin/env python3
"""
leak_audit.py - INDEPENDENT §2 audit of Tier-2 output. Does not trust the app.
==============================================================================
The shipped "§2 leak 0.000 %" figures come from the app's own `HoleMask` tripwire
(`HoleMask.kt:243-273`) - the same codebase being audited, gated to a 12 px ring. That is a
self-check, not an audit. This is the outside check: it takes only the INPUTS and the OUTPUT and
asks the two questions that matter, using nothing from the app.

    python leak_audit.py --masked-video masked_video.mp4 --mask mask.mp4 --output composite.mp4
    python leak_audit.py --selftest                       # prove the auditor can actually catch one

TWO INDEPENDENT QUESTIONS
  Q1 FILL SURVIVAL (privacy). Tier-1 paints the person solid grey. If that grey is still visible
     in the Tier-2 output, the inpaint failed to replace it and the frame ships a person-shaped
     hole - the silhouette leak. We detect the fill colour from the INPUT (it is whatever Tier-1
     used; measured 122 on one clip, 128 nominally - never assume) and count how much of the
     hole region still carries it in the OUTPUT.
  Q2 BACKGROUND FIDELITY (correctness, and a privacy signal). OUTSIDE the hole, Tier-2 must not
     alter the real background. Large changes there mean the compositor is writing where it
     should not - which can both corrupt the image and move real pixels around. Reported as
     per-frame mean |Δ| restricted to the non-hole region.

The self-test PLANTS a leak (pastes fill pixels back into the output) and requires the auditor to
catch it. An auditor that has never been shown to fire is not evidence.
"""
import argparse
import json
import os

import cv2
import numpy as np


def frames(path, max_frames=0):
    cap = cv2.VideoCapture(path)
    out = []
    while True:
        ok, f = cap.read()
        if not ok:
            break
        out.append(f)
        if max_frames and len(out) >= max_frames:
            break
    cap.release()
    if not out:
        raise SystemExit(f"no frames decoded from {path}")
    return out


def local_range(img, k=5):
    """5x5 local (max-min) of luma - a texture measure. Tier-1's fill is a SOLID flat
    colour, so its local range is ~0; real grey background (paving stones, concrete) is
    textured (measured median 18 on the alley clip). This is the SAME cue the app's own
    HoleMask uses to tell fill from same-coloured pavement, and it is why the app's tripwire
    reads ~0 where a colour-only test reads several percent. Added 2026-07-24 after the first
    device audit over-counted grey pavement as a leak on a grey-background scene."""
    g = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    ker = np.ones((k, k), np.uint8)
    return cv2.dilate(g, ker).astype(np.int16) - cv2.erode(g, ker).astype(np.int16)


def detect_fill(masked, masks, samples=9):
    """The fill colour Tier-1 actually used = the modal colour INSIDE the hole. Never assumed."""
    vals = []
    idx = np.linspace(0, len(masked) - 1, min(samples, len(masked))).astype(int)
    for i in idx:
        m = masks[min(i, len(masks) - 1)]
        h = (m[..., 0] if m.ndim == 3 else m) > 127
        if h.sum() < 50:
            continue
        px = masked[i][h]
        vals.append(np.median(px, axis=0))
    if not vals:
        return None, None
    v = np.median(np.stack(vals), axis=0)
    spread = float(np.mean([np.abs(x - v).mean() for x in vals]))
    return v, spread


def audit(masked, masks, out, fill, tol=12, max_frames=0, alphas=None, flat_max=4):
    """`alphas` = list of per-person character alpha videos (frame lists).

    WHY THEY MATTER - measured 2026-07-23 and it inverted the verdict. Colour-matching alone
    cannot tell "Tier-1 fill that survived into the output" from "the character is WEARING
    grey". On the cloud composite this read 5.59 % fill survival, which looks like a serious
    leak; 99.98 % of those pixels turned out to be INSIDE a character alpha, and the characters
    were themselves 4.19 % / 10.16 % near-grey-128. The real leak test is fill-coloured pixels
    in the hole that NO character covers - those are places where the inpaint failed and
    nothing was drawn over it. Both numbers are reported so the distinction stays visible."""
    n = min(len(masked), len(masks), len(out))
    if max_frames:
        n = min(n, max_frames)
    fill_survive, fill_survive_excl, bg_delta, hole_frac = [], [], [], []
    fill_under_alpha = []
    fill_survive_flat, leak_flat_frac = [], []
    blob_frac, blob_worst = [], (-1, 0.0, (0, 0))
    worst = (-1, 0.0)
    for i in range(n):
        m = masks[i]
        hole = ((m[..., 0] if m.ndim == 3 else m) > 127)
        cover = None
        if alphas:
            for A in alphas:
                if i < len(A):
                    f = A[i]
                    c = ((f[..., 0] if f.ndim == 3 else f) > 127)
                    if c.shape != hole.shape:
                        c = cv2.resize(c.astype(np.uint8), (hole.shape[1], hole.shape[0]),
                                       interpolation=cv2.INTER_NEAREST) > 0
                    cover = c if cover is None else (cover | c)
        o = out[i]
        if o.shape[:2] != hole.shape:
            o = cv2.resize(o, (hole.shape[1], hole.shape[0]), interpolation=cv2.INTER_NEAREST)
        mv = masked[i]
        if mv.shape[:2] != hole.shape:
            mv = cv2.resize(mv, (hole.shape[1], hole.shape[0]), interpolation=cv2.INTER_NEAREST)
        hole_frac.append(float(hole.mean()))
        # Q1 - fill colour still present inside the hole region of the OUTPUT
        if hole.sum():
            near = np.abs(o.astype(np.int16) - fill.astype(np.int16)).max(axis=2) <= tol
            frac = float(near[hole].mean())
            fill_survive.append(frac)
            # the number that actually means "leak": fill-coloured AND not covered by any
            # character. Grey clothing lives in the difference between these two.
            uncovered = hole & ~cover if cover is not None else hole
            fe = float((near & uncovered).sum() / hole.sum())
            fill_survive_excl.append(fe)
            # TEXTURE-DISCRIMINATED leak: Tier-1 fill is a solid FLAT colour; real grey
            # background (pavement/concrete) is textured. Gate the colour leak by local
            # flatness so a grey-background scene stops reading its own pavement as a leak.
            # This is the honest estimate on such scenes; `fe` (colour-only) is the
            # conservative upper bound kept beside it. On a non-grey scene the two coincide.
            flat = local_range(o) <= flat_max
            fl_mask = (near & uncovered & flat)
            ff = float(fl_mask.sum() / hole.sum())
            fill_survive_flat.append(ff)
            # SILHOUETTE SIGNAL: a real reveal is ONE connected, person-shaped blob; grey
            # pavement is many scattered specks. The largest connected flat-leak component
            # (as a fraction of the hole) discriminates the two - a scattered 0.5% is benign,
            # a connected 0.5% spanning the hole height is not. Added 2026-07-24.
            ncc, _, st, _ = cv2.connectedComponentsWithStats(fl_mask.astype(np.uint8), 8)
            if ncc > 1:
                areas = st[1:, cv2.CC_STAT_AREA]
                k = 1 + int(areas.argmax())
                bf = float(st[k, cv2.CC_STAT_AREA] / hole.sum())
                bbox = (int(st[k, cv2.CC_STAT_WIDTH]), int(st[k, cv2.CC_STAT_HEIGHT]))
            else:
                bf, bbox = 0.0, (0, 0)
            blob_frac.append(bf)
            if bf > blob_worst[1]:
                blob_worst = (i, bf, bbox)
            leak_pix = int((near & uncovered).sum())
            leak_flat_frac.append(float((near & uncovered & flat).sum() / leak_pix)
                                  if leak_pix else 0.0)
            if fe > worst[1]:
                worst = (i, fe)
            # The mass this exclusion DISCARDS, reported instead of thrown away. Everything
            # under a character alpha is assumed to be grey clothing - but a real leak sitting
            # under a claiming alpha would be excluded by exactly the same rule and become
            # invisible. It cannot be disambiguated from colour alone, so quantify it and let
            # the reader see how much room the assumption is being given. If this is large,
            # `fill_survival_mean_pct` is resting on it.
            if cover is not None:
                fill_under_alpha.append(float((near & hole & cover).sum() / hole.sum()))
        # Q2 - background outside the hole must be untouched
        bg = ~hole
        if bg.sum():
            bg_delta.append(float(np.abs(o[bg].astype(np.int16) -
                                         mv[bg].astype(np.int16)).mean()))
    fs = np.array(fill_survive) if fill_survive else np.array([0.0])
    fe = np.array(fill_survive_excl) if fill_survive_excl else fs
    ffa = np.array(fill_survive_flat) if fill_survive_flat else np.array([0.0])
    bd = np.array(bg_delta) if bg_delta else np.array([0.0])
    return {"frames": n, "mean_hole_frac_pct": float(np.mean(hole_frac) * 100),
            "character_alphas_supplied": bool(alphas),
            "fill_survival_raw_mean_pct": float(fs.mean() * 100),
            "fill_survival_mean_pct": float(fe.mean() * 100),
            "fill_survival_max_pct": float(fe.max() * 100),
            "fill_survival_worst_frame": int(worst[0]),
            # texture-discriminated leak (grey AND flat AND uncovered) - the honest number
            # on a scene whose real background is near the fill colour. flat_max is the 5x5
            # local-range ceiling; leak pixels that are textured (pavement) are dropped.
            "flat_max": flat_max,
            "fill_survival_flat_mean_pct": float(ffa.mean() * 100),
            "fill_survival_flat_max_pct": float(ffa.max() * 100),
            "frames_over_1pct_flat": int((ffa > 0.01).sum()),
            "leak_pixels_flat_frac_pct": (float(np.mean(leak_flat_frac) * 100)
                                          if leak_flat_frac else None),
            # largest CONNECTED flat-leak blob (silhouette signal). worst_bbox is (w,h) px.
            "largest_connected_flat_blob_mean_pct": (float(np.mean(blob_frac) * 100)
                                                     if blob_frac else 0.0),
            "largest_connected_flat_blob_max_pct": (float(np.max(blob_frac) * 100)
                                                    if blob_frac else 0.0),
            "largest_blob_worst_frame": int(blob_worst[0]),
            "largest_blob_worst_bbox_wh": list(blob_worst[2]),
            # How much fill-coloured area the alpha exclusion absorbed. Assumed to be grey
            # clothing; a leak under a claiming alpha is indistinguishable from it here, so this
            # is the size of the blind spot, not a clean bill of health.
            "fill_under_alpha_mean_pct": (float(np.mean(fill_under_alpha) * 100)
                                          if fill_under_alpha else None),
            "fill_under_alpha_max_pct": (float(np.max(fill_under_alpha) * 100)
                                         if fill_under_alpha else None),
            "frames_over_1pct_fill": int((fe > 0.01).sum()),
            "bg_mean_abs_delta": float(bd.mean()), "bg_max_abs_delta": float(bd.max()),
            "fill_colour_bgr": [float(x) for x in fill], "tolerance": tol}


def selftest():
    """Plant a leak and require the auditor to catch it. Also require a clean case to pass."""
    H = W = 128
    rng = np.random.RandomState(0)
    fill = np.array([122, 122, 122], np.uint8)
    masked, masks, clean, leaky = [], [], [], []
    for t in range(30):
        bg = (rng.rand(H, W, 3) * 200 + 20).astype(np.uint8)
        m = np.zeros((H, W), np.uint8)
        cv2.circle(m, (40 + t, 64), 22, 1, -1)
        mv = bg.copy(); mv[m > 0] = fill                      # Tier-1 output
        ch = bg.copy()
        ch[m > 0] = (rng.rand(int((m > 0).sum()), 3) * 255).astype(np.uint8)   # good inpaint
        masked.append(mv); masks.append((m * 255).astype(np.uint8)); clean.append(ch)
        lk = ch.copy()
        ys, xs = np.nonzero(m)
        k = ys.size // 5                                       # 20 % of the hole left as fill
        lk[ys[:k], xs[:k]] = fill
        leaky.append(lk)
    f, _ = detect_fill(masked, masks)
    print(f"  detected fill colour BGR {f} (planted {fill})")
    a_clean = audit(masked, masks, clean, f)
    a_leak = audit(masked, masks, leaky, f)
    print(f"  CLEAN output : fill survival {a_clean['fill_survival_mean_pct']:.3f}% "
          f"| bg delta {a_clean['bg_mean_abs_delta']:.3f}")
    print(f"  LEAKY output : fill survival {a_leak['fill_survival_mean_pct']:.3f}% "
          f"| bg delta {a_leak['bg_mean_abs_delta']:.3f}")
    ok = (a_clean["fill_survival_mean_pct"] < 1.0 and a_leak["fill_survival_mean_pct"] > 10.0
          and a_clean["bg_mean_abs_delta"] < 0.5)
    print(f"  => auditor {'CATCHES the planted leak and passes the clean case' if ok else 'FAILED'}")
    if not ok:
        raise SystemExit("self-test FAILED - auditor is not trustworthy")
    return {"clean": a_clean, "leaky": a_leak, "passed": True}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--selftest", action="store_true")
    ap.add_argument("--masked-video"); ap.add_argument("--mask"); ap.add_argument("--output")
    ap.add_argument("--alpha", nargs="*", default=None,
                    help="character alpha video(s). STRONGLY RECOMMENDED: without them a "
                         "grey-clothed character reads as a leak (measured: 5.59%% vs 0.02%%).")
    ap.add_argument("--tag", default="run"); ap.add_argument("--max-frames", type=int, default=0)
    ap.add_argument("--flat-max", type=int, default=4,
                    help="5x5 local-range ceiling for the flatness-gated leak (Tier-1 fill is "
                         "flat ~0; real grey pavement is textured ~18). Default 4.")
    ap.add_argument("--out", default=os.path.join(os.path.dirname(os.path.abspath(__file__)),
                                                  "reports"))
    a = ap.parse_args()
    os.makedirs(a.out, exist_ok=True)

    if a.selftest:
        print("[audit] SELF-TEST - planting a known leak")
        rep = {"selftest": selftest()}
        p = os.path.join(a.out, "LEAK_AUDIT_selftest.json")
        json.dump(rep, open(p, "w"), indent=2)
        print(f"\n[audit] wrote {p}")
        return

    if not (a.masked_video and a.mask and a.output):
        raise SystemExit("need --masked-video --mask --output (or --selftest)")
    mv = frames(a.masked_video, a.max_frames)
    mk = frames(a.mask, a.max_frames)
    ou = frames(a.output, a.max_frames)
    fill, spread = detect_fill(mv, mk)
    if fill is None:
        raise SystemExit("could not detect a fill colour - is the mask empty?")
    print(f"[audit] fill colour detected BGR {fill} (sample spread {spread:.2f})")
    al = [frames(x, a.max_frames) for x in (a.alpha or [])] or None
    if al:
        print(f"[audit] {len(al)} character alpha(s) supplied - grey clothing will be excluded")
    else:
        print("[audit] !! no --alpha given: a grey-clothed character will read as a leak")
    rep = audit(mv, mk, ou, fill, max_frames=a.max_frames, alphas=al, flat_max=a.flat_max)
    print(f"  frames {rep['frames']}  mean hole {rep['mean_hole_frac_pct']:.2f}%")
    print(f"  Q1 raw fill-colour in hole   : mean {rep['fill_survival_raw_mean_pct']:.4f}% "
          f"(includes grey clothing)")
    print(f"  Q1 LEAK colour-only (fill & no char): mean {rep['fill_survival_mean_pct']:.4f}%  "
          f"max {rep['fill_survival_max_pct']:.4f}% (frame {rep['fill_survival_worst_frame']})")
    print(f"     frames over 1% fill       : {rep['frames_over_1pct_fill']}")
    print(f"  Q1 LEAK flatness-gated (fill & FLAT & no char): mean "
          f"{rep['fill_survival_flat_mean_pct']:.4f}%  max {rep['fill_survival_flat_max_pct']:.4f}%  "
          f"({rep['frames_over_1pct_flat']} frames >1%)")
    if rep.get("leak_pixels_flat_frac_pct") is not None:
        print(f"     ^ only {rep['leak_pixels_flat_frac_pct']:.1f}% of colour-leak pixels are "
              f"flat; the rest are textured background (pavement/concrete near the fill colour)")
    bw, bh = rep["largest_blob_worst_bbox_wh"]
    print(f"  Q1 largest CONNECTED flat-leak blob (silhouette signal): mean "
          f"{rep['largest_connected_flat_blob_mean_pct']:.4f}%  max "
          f"{rep['largest_connected_flat_blob_max_pct']:.4f}% of hole "
          f"(frame {rep['largest_blob_worst_frame']}, bbox {bw}x{bh}px) - "
          f"a person reveal would be ONE blob spanning the hole")
    if rep.get("fill_under_alpha_mean_pct") is not None:
        print(f"     ^ excluded as grey clothing: mean "
              f"{rep['fill_under_alpha_mean_pct']:.4f}%  max "
              f"{rep['fill_under_alpha_max_pct']:.4f}%  <- a leak UNDER an alpha hides in here")
    print(f"  Q2 background |delta| outside : mean {rep['bg_mean_abs_delta']:.4f}  "
          f"max {rep['bg_max_abs_delta']:.4f}")
    # Record WHICH files produced these numbers. Without this the report cannot be traced back
    # to a clip, so a ledger row citing one is unverifiable from the artifact alone.
    rep["inputs"] = {"masked_video": os.path.abspath(a.masked_video),
                     "mask": os.path.abspath(a.mask),
                     "output": os.path.abspath(a.output),
                     "alphas": [os.path.abspath(x) for x in (a.alpha or [])],
                     "max_frames": int(a.max_frames), "tag": a.tag}
    p = os.path.join(a.out, f"LEAK_AUDIT_{a.tag}.json")
    json.dump(rep, open(p, "w"), indent=2)
    print(f"\n[audit] wrote {p}")


if __name__ == "__main__":
    main()
