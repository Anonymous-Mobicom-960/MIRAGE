#!/usr/bin/env python3
"""
build_arms_20260731.py -- writes the arm spec consumed by `run_silhouette_reid.py --arms-json`.
================================================================================================
THE OBJECTIVE THIS GRID SERVES (user, 2026-07-31): find the strongest SILHOUETTE-channel
anti-re-ID configuration **at or below the shipped mask-area ratio** (displace 0.25 @ win 2 =
x1.358 vs the raw binary mask, ledger §A.6g). Area is the budget -- it is what the cloud
background-fill / inpaint pays (§B.48: `hull` at x1.470 cost the composite SSIM -0.0215 /
LPIPS +0.0157, essentially all of it in the background plate). `hull` and `ellipse` are BANNED by
user decision and appear nowhere here.

The grid is therefore weighted toward mechanisms that REPLACE shape information rather than ADD
coverage:
  (a) k-SAME COLLAPSE (`ksame`)  -- push every silhouette out to ONE shared population profile;
      the mask analogue of the pose anonymiser's `_TEMPLATE_RATIOS`. Template from CASIA-B TRAIN
      ids 001..074 only (derive_ksame_template.py).
  (b) TEMPORAL DE-CORRELATION at constant area -- phase-advance rate and per-epoch re-seeding of
      the displacement field. Neither adds a pixel of mean area.
  (c) OUTWARD-ONLY RADIAL LOW-PASS (`radiallp`) and per-component morphological CLOSE.

Every amplitude here was AREA-SCREENED first on 40 real CASIA-B sequences, so the grid does not
waste GPU minutes on arms already known to breach the ceiling. The two `w6`/`w8` arms are the
exception: they are included only to complete the window question the brief asked, and are known
in advance to be over budget.
"""
import json
import os

HERE = os.path.dirname(os.path.abspath(__file__))
TMPL = os.path.join(HERE, "reports", "KSAME_TEMPLATE.json")
OUT = os.path.join(HERE, "reports", "ARMS_20260731.json")
D = "displace"


def main():
    T = json.load(open(TMPL))["templates"]
    K = lambda q: list(T[q])
    arms = []

    def add(name, win=2, **cfg):
        arms.append({"name": name, "win": win, "cfg": cfg})

    # ---- (b) TEMPORAL DE-CORRELATION, area-neutral by construction ----------------------
    for s in (0.0, 1.0, 2.0):                       # phase-advance rate per emitted frame
        add(f"d25_ps{s}", 2, MASK_SHAPE_MODE=D, MASK_DISPLACE_AMP_FRAC=0.25,
            MASK_DISPLACE_PHASE_STEP=s)
    for r, lbl in ((3.5, "10f"), (1.4, "4f"), (0.34, "1f")):    # re-seed every N frames
        add(f"d25_rs{lbl}", 2, MASK_SHAPE_MODE=D, MASK_DISPLACE_AMP_FRAC=0.25,
            MASK_DISPLACE_RESEED_PHASE=r)
    for h in (8, 12):                               # harmonic richness at the same amplitude
        add(f"d25_h{h}", 2, MASK_SHAPE_MODE=D, MASK_DISPLACE_AMP_FRAC=0.25,
            MASK_DISPLACE_HARMONICS=h)
    add("d25_h8_rs4f_ps1", 2, MASK_SHAPE_MODE=D, MASK_DISPLACE_AMP_FRAC=0.25,
        MASK_DISPLACE_HARMONICS=8, MASK_DISPLACE_RESEED_PHASE=1.4,
        MASK_DISPLACE_PHASE_STEP=1.0)               # all three area-neutral levers together

    # ---- (c) OUTWARD-ONLY RADIAL LOW-PASS (screened: keep<=4 breaches the ceiling) ------
    for k in (6, 10):
        add(f"rlp{k}", 2, MASK_SHAPE_MODE="radiallp", MASK_RADIALLP_KEEP=k)

    # ---- per-component morphological CLOSE (screened at x1.12-x1.16: very cheap) --------
    for f in (0.25, 0.35, 0.50):
        add(f"close{int(f*100)}", 2, MASK_SHAPE_MODE="close", MASK_CLOSE_KERNEL_FRAC=f)

    # ---- (a) k-SAME COLLAPSE (scale<1 needed to sit under the ceiling) -----------------
    add("ksame_p50_s85", 2, MASK_SHAPE_MODE="ksame", MASK_KSAME_TEMPLATE=K("p50"),
        MASK_KSAME_SCALE=0.85)
    add("ksame_p50_s90", 2, MASK_SHAPE_MODE="ksame", MASK_KSAME_TEMPLATE=K("p50"),
        MASK_KSAME_SCALE=0.90)
    add("ksame_p85_s85", 2, MASK_SHAPE_MODE="ksame", MASK_KSAME_TEMPLATE=K("p85"),
        MASK_KSAME_SCALE=0.85)

    # ---- COMPOSITIONS that fit the budget ----------------------------------------------
    add("close25+d20", 2, MASK_SHAPE_MODE="close+displace", MASK_CLOSE_KERNEL_FRAC=0.25,
        MASK_DISPLACE_AMP_FRAC=0.20)
    add("close35+d15", 2, MASK_SHAPE_MODE="close+displace", MASK_CLOSE_KERNEL_FRAC=0.35,
        MASK_DISPLACE_AMP_FRAC=0.15)
    add("close25+rlp10", 2, MASK_SHAPE_MODE="close+radiallp", MASK_CLOSE_KERNEL_FRAC=0.25,
        MASK_RADIALLP_KEEP=10)
    add("close25+ksame50s85", 2, MASK_SHAPE_MODE="close+ksame", MASK_CLOSE_KERNEL_FRAC=0.25,
        MASK_KSAME_TEMPLATE=K("p50"), MASK_KSAME_SCALE=0.85)

    # ---- OVER-BUDGET references: the window question the brief asked -------------------
    for w in (6, 8):
        add(f"d25_w{w}", w, MASK_SHAPE_MODE=D, MASK_DISPLACE_AMP_FRAC=0.25)

    json.dump(arms, open(OUT, "w"), indent=1)
    print(f"[arms] {len(arms)} arms -> {OUT}")
    for a in arms:
        print(f"   {a['name']:22s} win{a['win']} "
              + " ".join(f"{k}={'<tmpl>' if k == 'MASK_KSAME_TEMPLATE' else v}"
                         for k, v in a["cfg"].items()))


if __name__ == "__main__":
    main()
