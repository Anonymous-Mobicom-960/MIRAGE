#!/usr/bin/env python3
"""AUTHOR the character alpha FROM TIER-1 - owner decision, 2026-08-08.

    "keep alpha as primarily turned on ... alpha must be created from tier 1"

WHAT WAS WRONG. Until now the matte was produced ENTIRELY inside the cloud, by running a fresh
person detector over the GENERATED video:

    #28 WanVideoDecode -> #779 ImageFromBatch -> #259 PoseAndFaceDetection
      -> #262 Sam2Segmentation -> #235 GrowMaskWithBlur -> #786 MaskToImage -> synthetic_alpha_pK

Nothing in that chain is tied to the slot whose sticks drove the sampler, and both failure modes
are measured:

    RUN3_c2         no person generated  -> the matte degenerated to 88.66 % of frame (c1: 7.77 %)
                                            and Phase 2 pasted the lightmap over the composite
    RUN3_c3_male    drove p1 (sticks cx 366.4) -> alpha cx 964.1   -> 0/140 frames correct
    RUN3_c3_female  drove p2 (sticks cx 970.6) -> alpha cx 353.6   -> 0/140 frames correct

THE DESIGN THAT SHIPS - Tier-1 authors the matte's DOMAIN, the generated pixels author its SHAPE.

    DOMAIN_K = grow_blockify(Tier-1 mask_pK)     the EXACT hole the sampler was allowed to repaint
                                                 (GrowMaskWithBlur #224 expand 10 / blur 4 ->
                                                  BlockifyMask #225 block 16, as wired in the graph)
    SHAPE    = a segmentation of the GENERATED character:
                 1. the cloud's SAM2 matte, but only the components that LIVE IN DOMAIN_K;
                 2. else the painted-difference key |generated - light_map| > KEY_T.
               Which one was used is RECORDED PER LAYER and printed. There is no silent choice.
    alpha_K  = union(selected components) & DOMAIN_K
    REFUSE     if nothing survives, if coverage leaves the plausible band, or if the matte is
               simply the box (i.e. it is the hole, not a character).

🔴 WHY NOT "USE TIER-1's SILHOUETTE AS THE MATTE". Because that is the one thing the anonymiser
exists to prevent. §A.6o measures the undefended silhouette at +31.88 pp of re-ID lift over its own
null, and `bbox` is the SHIPPED mode precisely because it cuts that to +7.16. Cutting the generated
character to the real outline would paste the real shape straight back into the output.

WHAT TIER-1 CONTRIBUTES HERE, EXACTLY. A spatial RESTRICTION - an intersection with `mask_pK`,
which the cloud already holds (it is the uploaded `mask_*_00002.mp4`, and it is what the plate is
built from). Intersecting with a set the recipient already has cannot disclose anything new. And at
the shipped `MASK_SHAPE_MODE=bbox` that set is an axis-aligned rectangle, so the only boundary the
gate can ever contribute is a rectangle edge - zero real-silhouette shape, by construction. The
module MEASURES this rather than asserting it: `tier1_boundary_frac` is the fraction of the emitted
matte's own boundary that lies on the DOMAIN edge.

  python alpha_from_tier1.py --selftest
  python alpha_from_tier1.py --clip c3_p20p21_two --slot p1 \
      --from-cloud c3_p20p21_two/from_cloud_male --out c3_p20p21_two/alpha_t1
"""
# cp1252 consoles cannot encode the status glyphs below; without this the script dies on a
# print AFTER doing its work, which reads as a failed run. See scripts/utilities/console_safe.py.
import os as _os, sys as _sys
_sys.path.insert(0, _os.path.join(_os.path.dirname(_os.path.abspath(__file__)),
                                  "..", "..", "scripts", "utilities"))
import console_safe  # noqa: F401,E402
import argparse
import glob
import io
import json
import os
import sys

import cv2
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))

# ---- the sampler's repaintable hole, as wired in V9_GAIT_A_g9proj / RUN3_c*.api.json -----------
# #224 GrowMaskWithBlur(expand 10, blur 4) -> #225 BlockifyMask(16). Same constants as
# check_render.py, deliberately duplicated nowhere: imported from there.
# check_render.py is the cloud-render QC tool; the blockify constants are imported from it so
# they are defined in exactly one place (see tier2_cloud/scripts/check_render.py).
sys.path.insert(0, os.path.join(HERE, "..", "..", "tier2_cloud", "scripts"))
from check_render import GROW_EXPAND, GROW_BLUR, BLOCK, grow_blockify   # noqa: E402,F401

# ---- painted-difference key -------------------------------------------------------------------
# Outside the character the generated frame IS the lightmap it was conditioned on (measured on c3:
# max-channel |gen - light_map| p99 = 11..13 over non-mask pixels vs an in-mask mean of ~90).
# KEY_T swept 12/18/24/30/40 against the cloud's own SAM2 matte; 24 is the peak of a flat 18..30
# plateau (§B.60d). Kept identical to split_layers.KEY_T so the two agree by construction.
KEY_T = 24
MIN_COMP = 3000          # px - the generated characters are ~150 000 px each

# A candidate component belongs to slot K only if at least this much of IT is inside DOMAIN_K.
# Deliberately a fraction OF THE COMPONENT, not of the domain: that is what rejects c2's inverted
# 88.66 %-of-frame matte (a huge component with a small slice inside the box) while accepting c1's
# character (99 % of it inside the box).
INSIDE_MIN = 0.50

# SAM2 is the preferred shape source, but only if it lands in this slot's domain on essentially
# every frame. Below this it is not this slot's matte at all (c3: 0/140) or it has degenerated
# (c2), and the painted key takes over - for the WHOLE clip, so the source cannot flip mid-clip.
SOURCE_MIN_FRAC = 0.90

# ---- refusal thresholds ------------------------------------------------------------------------
COV_MIN, COV_MAX = 0.005, 0.400      # same band as check_render.ALPHA_MIN/MAX
EMPTY_FRAC_MAX = 0.20                # > 20 % of frames with no character in the slot's domain
DOMAIN_FILL_MAX = 0.95               # the matte is the hole, not a character


# ------------------------------------------------------------------ small helpers
def _cap(path):
    c = cv2.VideoCapture(path)
    if not c.isOpened():
        raise SystemExit(f"cannot open {path}")
    return c


def fill_holes(b):
    """Background not connected to the frame border is interior - fill it.

    Same rule as Compositor.fillHoles on the phone and split_layers.fill_holes, so a dark jacket
    cannot show the background through the character.
    """
    h, w = b.shape
    ff = np.zeros((h + 2, w + 2), np.uint8)
    inv = (1 - b).astype(np.uint8).copy()
    cv2.floodFill(inv, ff, (0, 0), 2)
    return ((b > 0) | (inv == 1)).astype(np.uint8)


def painted_key(gen, lm, t=KEY_T):
    """What the generator actually PAINTED: |gen - light_map| with morphological cleanup."""
    if lm.shape[:2] != gen.shape[:2]:
        lm = cv2.resize(lm, (gen.shape[1], gen.shape[0]), interpolation=cv2.INTER_LINEAR)
    d = np.abs(gen.astype(np.int16) - lm.astype(np.int16)).max(2)
    k = (d > t).astype(np.uint8)
    k = cv2.morphologyEx(k, cv2.MORPH_OPEN, np.ones((5, 5), np.uint8))
    k = cv2.morphologyEx(k, cv2.MORPH_CLOSE, np.ones((9, 9), np.uint8))
    return fill_holes(k)


def select_in_domain(binary, domain, inside_min=INSIDE_MIN, min_comp=MIN_COMP):
    """Union of the components of `binary` that LIVE IN `domain`. Returns (mask, n_kept, n_seen)."""
    n, lab, st, _ = cv2.connectedComponentsWithStats(binary.astype(np.uint8), 8)
    out = np.zeros(binary.shape, bool)
    kept = seen = 0
    for i in range(1, n):
        if st[i, 4] < min_comp:
            continue
        seen += 1
        comp = lab == i
        if float((comp & domain).sum()) / float(comp.sum()) >= inside_min:
            out |= comp
            kept += 1
    return out, kept, seen


def boundary(m):
    """1 px inner boundary of a boolean mask."""
    u = m.astype(np.uint8)
    return (u - cv2.erode(u, np.ones((3, 3), np.uint8))).astype(bool)


# ------------------------------------------------------------------ the authoring pass
def author(clip, slot, from_cloud, bundle="", out_dir="", cap=10 ** 9, quiet=False,
           write_refused=False):
    """Author `synthetic_alpha_<slot>.mp4` for one slot. Returns a report dict.

    `from_cloud` is one cloud arm's fetch directory (its files are named `..._p1_...` whatever slot
    actually drove the render - the cloud names them by NODE, not by slot).
    """
    d = bundle or os.path.join(HERE, clip, "to_cloud")
    gen_p = sorted(glob.glob(os.path.join(from_cloud, "synthetic_person_p*.mp4")))
    cal_p = sorted(glob.glob(os.path.join(from_cloud, "synthetic_alpha_p*.mp4")))
    lm_p = os.path.join(d, "light_map.mp4")
    # Tier-1's authority for this slot: the per-slot mask if the clip has one, else the union - on a
    # single-slot clip the union IS that slot's mask, and there is nothing to confuse.
    mk_p = os.path.join(d, f"mask_{slot}_00002.mp4")
    per_slot = os.path.exists(mk_p)
    if not per_slot:
        mk_p = os.path.join(d, "mask_00002.mp4")
    for p, what in ((lm_p, "light_map.mp4"), (mk_p, os.path.basename(mk_p))):
        if not os.path.exists(p):
            return dict(error=f"missing Tier-1 artifact {what} in {d}")
    if not gen_p:
        return dict(error=f"no synthetic_person_p*.mp4 in {from_cloud}")

    cg, cl, cm = _cap(gen_p[0]), _cap(lm_p), _cap(mk_p)
    ca = _cap(cal_p[0]) if cal_p else None
    sizes = dict(gen=int(cg.get(cv2.CAP_PROP_FRAME_COUNT)),
                 light_map=int(cl.get(cv2.CAP_PROP_FRAME_COUNT)),
                 mask=int(cm.get(cv2.CAP_PROP_FRAME_COUNT)))
    fps = cg.get(cv2.CAP_PROP_FPS) or 10.0

    if out_dir:
        os.makedirs(out_dir, exist_ok=True)

    # ---- pass 1: compute BOTH candidate mattes per frame, store them packed --------------------
    # Both are kept so the SHAPE SOURCE can be decided ONCE for the whole clip rather than per
    # frame. A source that flips mid-clip is a temporal artifact, and `NCompositor` already sets
    # the precedent on the phone: "decide each layer's matte source ONCE, from frame 0, and log it".
    # np.packbits keeps 140 frames of two 1264² masks in ~56 MB.
    dom_k, sam_k, key_k, shape = [], [], [], None
    ctrl_iou = []           # the cloud's SAM2 matte scored against OUR key, where both land here
    t = 0
    while t < cap:
        okg, gen = cg.read()
        okl, lm = cl.read()
        okm, mk = cm.read()
        oka, cav = (ca.read() if ca else (False, None))
        if not (okg and okl and okm):
            break
        h, w = gen.shape[:2]
        shape = (h, w)
        if mk.shape[:2] != (h, w):
            mk = cv2.resize(mk, (w, h), interpolation=cv2.INTER_NEAREST)
        domain = grow_blockify(cv2.cvtColor(mk, cv2.COLOR_BGR2GRAY) > 127)

        sam = np.zeros((h, w), bool)
        if oka and cav is not None:
            if cav.shape[:2] != (h, w):
                cav = cv2.resize(cav, (w, h), interpolation=cv2.INTER_NEAREST)
            sam, _kept, _seen = select_in_domain(cv2.cvtColor(cav, cv2.COLOR_BGR2GRAY) > 127, domain)
        key, _kept, _seen = select_in_domain(painted_key(gen, lm), domain)
        if sam.any() and key.any():
            ctrl_iou.append(float((sam & key).sum()) / float((sam | key).sum()))
        dom_k.append(np.packbits(domain))
        sam_k.append(np.packbits(sam))
        key_k.append(np.packbits(key))
        t += 1
    for c in (cg, cl, cm, ca):
        if c is not None:
            c.release()
    if not dom_k:
        return dict(error="no overlapping frames between the render and the Tier-1 bundle")

    def unpack(a):
        return np.unpackbits(a)[:shape[0] * shape[1]].reshape(shape).astype(bool)

    n = len(dom_k)
    sam_hit = sum(1 for a in sam_k if a.any())
    key_hit = sum(1 for a in key_k if a.any())
    # SAM2 is preferred whenever it lands in this slot's domain on essentially every frame:
    # ALPHA_MATTE.md §3 measures semantic segmentation as categorically better than colour keying
    # (a character whose clothing matches the block behind it is still segmented). The painted key
    # takes over only when SAM2's matte is not this slot's - which is exactly c2 and c3.
    source = "sam2" if sam_hit >= SOURCE_MIN_FRAC * n else ("painted_key" if key_hit else "NONE")
    primary, secondary = ((sam_k, key_k) if source == "sam2" else (key_k, sam_k))

    # ---- pass 2: compose --------------------------------------------------------------------
    frames, fallback, empty = [], 0, 0
    cov, dom_fill, contain, t1b = [], [], [], []
    cm2 = _cap(mk_p)
    for i in range(n):
        domain = unpack(dom_k[i])
        sel = unpack(primary[i])
        used = source
        if not sel.any():
            sel = unpack(secondary[i])
            if sel.any():
                used = "sam2" if source == "painted_key" else "painted_key"
                fallback += 1
        alpha = sel & domain
        okm, mk = cm2.read()
        m = (cv2.cvtColor(cv2.resize(mk, (shape[1], shape[0]), interpolation=cv2.INTER_NEAREST)
                          if okm and mk.shape[:2] != shape else mk,
                          cv2.COLOR_BGR2GRAY) > 127) if okm else domain
        if not alpha.any():
            empty += 1
            used = "none"
        else:
            cov.append(float(alpha.mean()))
            dom_fill.append(float(alpha.sum()) / float(max(domain.sum(), 1)))
            contain.append(float((alpha & m).sum()) / float(alpha.sum()))
            # how much of the emitted matte's own boundary was CUT BY TIER-1 rather than drawn by
            # the generator: boundary pixels that sit on the domain edge
            b = boundary(alpha)
            db = cv2.dilate(boundary(domain).astype(np.uint8), np.ones((3, 3), np.uint8)).astype(bool)
            t1b.append(float((b & db).sum()) / float(max(b.sum(), 1)))
        frames.append((alpha, used))
    cm2.release()

    r = dict(
        clip=clip, slot=slot, n=n, fps=round(float(fps), 3),
        tier1_mask=os.path.basename(mk_p), per_slot_mask=per_slot,
        from_cloud=os.path.basename(from_cloud.rstrip("/\\")),
        shape_source=source,
        source_frames=dict(sam2_lands_in_domain=sam_hit, painted_key_lands_in_domain=key_hit,
                           per_frame_fallbacks=fallback, none=empty),
        coverage_mean=round(float(np.mean(cov)) if cov else 0.0, 6),
        domain_fill_mean=round(float(np.mean(dom_fill)) if dom_fill else 0.0, 6),
        containment_in_raw_box_mean=round(float(np.mean(contain)) if contain else 0.0, 6),
        tier1_boundary_frac_mean=round(float(np.mean(t1b)) if t1b else 0.0, 6),
        tier1_boundary_frac_max=round(float(np.max(t1b)) if t1b else 0.0, 6),
        frames_source_counts=sizes,
    )
    if ctrl_iou:
        r["control_iou_sam2_vs_painted_key"] = dict(
            n=len(ctrl_iou), mean=round(float(np.mean(ctrl_iou)), 4),
            p05=round(float(np.percentile(ctrl_iou, 5)), 4), min=round(float(np.min(ctrl_iou)), 4))

    # ---- identity, on the AUTHORED matte, by check_render's own rule ---------------------------
    idr = identity_check(frames, clip, slot, d)
    if idr:
        r["identity_gate"] = f"{idr[0]}/{idr[1]}"
        r["identity_pass"] = idr[0] >= 0.7 * idr[1]

    # ---- REFUSAL -------------------------------------------------------------------------------
    why = []
    if empty / float(n) > EMPTY_FRAC_MAX:
        why.append(f"{empty}/{n} frames have NO generated character inside {slot}'s Tier-1 box "
                   f"(> {100*EMPTY_FRAC_MAX:.0f} %)")
    if cov and not (COV_MIN <= float(np.mean(cov)) <= COV_MAX):
        why.append(f"mean coverage {100*float(np.mean(cov)):.2f} % outside the plausible band "
                   f"{100*COV_MIN:.1f}-{100*COV_MAX:.1f} %")
    if dom_fill and float(np.mean(dom_fill)) > DOMAIN_FILL_MAX:
        why.append(f"the matte fills {100*float(np.mean(dom_fill)):.1f} % of the repaintable hole - "
                   f"that is the BOX, not a character (§B.57's 88.66 % failure)")
    r["refused"] = bool(why)
    r["refusal"] = why

    if not quiet:
        src = r["shape_source"]
        print(f"  {clip}/{slot}: {n} f · shape from {src.upper()} "
              f"(SAM2 lands here on {sam_hit}/{n}, painted key on {key_hit}/{n}; "
              f"{fallback} per-frame fallback(s), {empty} empty) · "
              f"domain = {r['tier1_mask']}{'' if per_slot else '  [union - single-slot clip]'}")
        print(f"      coverage {100*r['coverage_mean']:.2f} % of frame · fills "
              f"{100*r['domain_fill_mean']:.1f} % of the hole · "
              f"{100*r['containment_in_raw_box_mean']:.2f} % inside the raw Tier-1 box")
        print(f"      TIER-1 SHAPE CONTRIBUTION: {100*r['tier1_boundary_frac_mean']:.2f} % of the "
              f"matte boundary lies on the domain edge (max {100*r['tier1_boundary_frac_max']:.2f} %)"
              f" - and the domain is an axis-aligned bbox")
        if ctrl_iou:
            c = r["control_iou_sam2_vs_painted_key"]
            print(f"      CONTROL  SAM2 vs painted key, where both land in this domain: "
                  f"IoU {c['mean']:.4f} (p05 {c['p05']:.4f}, n={c['n']})")
        if "identity_gate" in r:
            print(f"      IDENTITY the authored matte is nearest slot {slot} in "
                  f"{r['identity_gate']} frames -> {'PASS' if r['identity_pass'] else 'FAIL'}")
        for x in why:
            print(f"      🔴 REFUSE: {x}")

    # ---- write ---------------------------------------------------------------------------------
    # A refusal writes NOTHING by default: a refused matte on disk is a matte that will eventually be
    # composited by someone who did not read the log. `write_refused` exists only for `--force`.
    if out_dir and r["refused"] and write_refused and not quiet:
        print("      ⚠ WRITING A REFUSED MATTE because --force was given - do not use it for any "
              "quality or privacy number")
    if out_dir and (not r["refused"] or write_refused):
        ap = os.path.join(out_dir, f"synthetic_alpha_{slot}.mp4")
        pp = os.path.join(out_dir, f"synthetic_person_{slot}.mp4")
        h, w = frames[0][0].shape
        vw = cv2.VideoWriter(ap, cv2.VideoWriter_fourcc(*"mp4v"), fps, (w, h))
        for a, _ in frames:
            vw.write(cv2.cvtColor((a * 255).astype(np.uint8), cv2.COLOR_GRAY2BGR))
        vw.release()
        cg = _cap(gen_p[0])
        vw = cv2.VideoWriter(pp, cv2.VideoWriter_fourcc(*"mp4v"), fps, (w, h))
        for _ in range(n):
            ok, f = cg.read()
            if not ok:
                break
            vw.write(f)
        cg.release()
        vw.release()
        r["written"] = [os.path.basename(ap), os.path.basename(pp)]
        if not quiet:
            print(f"      -> {ap}")
    elif out_dir:
        r["written"] = []
    return r


# ------------------------------------------------------------------ SELFTEST
# Every arm on disk, and - the point of the exercise - the SAME criteria applied to the CLOUD's own
# alpha for the same arm. A gate with no negative control measures nothing (§A.6o-1b). Here the
# negative control is not a synthetic one: it is what actually shipped tonight.
CONTROLS = [
    # clip, slot, from_cloud dir, expected AUTHORED, expected CLOUD, what happened on the pod
    ("c1_p05_single", "p1", "from_cloud", "PASS", "PASS",
     "rendered a person; the cloud's own alpha was already correct"),
    ("c3_p20p21_two", "p1", "from_cloud_male", "PASS", "REFUSE",
     "two characters; the cloud's alpha followed p2 - 0/140 correct"),
    ("c3_p20p21_two", "p2", "from_cloud_female", "PASS", "REFUSE",
     "two characters; the cloud's alpha followed p1 - 0/140 correct"),
    # 🔴 c2 is NOT refused by this module, and that is the honest outcome - see score() below.
    ("c2_p08_single", "p1", "from_cloud", "PASS", "REFUSE",
     "no PERSON generated (a metal cylinder); the cloud's alpha degenerated to 88.66 % of frame"),
]


def score(alpha_path, clip, slot, bundle="", cap=10 ** 9):
    """Apply the SAME coverage + identity criteria to an ALREADY-EXISTING alpha video.

    This is what makes the selftest a controlled comparison rather than a demo: the authored matte
    and the cloud's own matte are judged by one function, on one clip, in one run.
    """
    import check_render as CR
    cov = CR.alpha_coverage(alpha_path, cap)
    if cov is None or not len(cov):
        return dict(error="unreadable")
    c = _cap(alpha_path)
    fr = []
    while len(fr) < cap:
        ok, f = c.read()
        if not ok:
            break
        fr.append((cv2.cvtColor(f, cv2.COLOR_BGR2GRAY) > 127, ""))
    c.release()
    idr = identity_check(fr, clip, slot, bundle)
    m = float(np.mean(cov))
    why = []
    if not (COV_MIN <= m <= COV_MAX):
        why.append(f"coverage {100*m:.2f} % outside {100*COV_MIN:.1f}-{100*COV_MAX:.1f} %")
    if idr and idr[0] < 0.7 * idr[1]:
        why.append(f"follows the driven slot in only {idr[0]}/{idr[1]} frames")
    return dict(coverage_mean=round(m, 6), identity=(f"{idr[0]}/{idr[1]}" if idr else "n/a"),
                refused=bool(why), refusal=why)


def identity_check(alpha_frames, clip, slot, bundle=""):
    """Reuse check_render's rule: is the matte nearest the slot whose sticks drove the render?

    Returns None on a single-slot clip, where there is nothing to confuse.
    """
    import check_render as CR
    d = bundle or os.path.join(HERE, clip, "to_cloud")
    sticks = {os.path.basename(p).split("_")[2]: p
              for p in sorted(glob.glob(os.path.join(d, "pose_sticks_p*_00002.mp4")))}
    sticks.pop("both", None)
    if len(sticks) < 2:
        return None
    cs = {s: CR.centroid_x(p, 25) for s, p in sticks.items()}
    ok = tot = 0
    for i, (a, _u) in enumerate(alpha_frames):
        if not a.any():
            continue
        acx = float(np.where(a)[1].mean())
        ds = {s: abs(acx - v[i]) for s, v in cs.items() if i < len(v) and not np.isnan(v[i])}
        if len(ds) < 2:
            continue
        tot += 1
        ok += (min(ds, key=ds.get) == slot)
    return (ok, tot) if tot else None


def selftest(cap=10 ** 9, out_json=""):
    print("ALPHA-FROM-TIER-1 SELFTEST - every RUN3 arm, authored matte vs the cloud's own, one "
          "criterion set, one run\n")
    rows, bad = [], 0
    for clip, slot, sub, want, want_cloud, note in CONTROLS:
        fc = os.path.join(HERE, clip, sub)
        if not os.path.isdir(fc):
            print(f"{clip}/{slot:3s} MISSING {sub}")
            bad += 1
            continue
        r = author(clip, slot, fc, cap=cap)
        if "error" in r:
            print(f"{clip}/{slot:3s} ERROR {r['error']}")
            bad += 1
            continue
        got = "REFUSE" if r["refused"] else "PASS"
        cal = sorted(glob.glob(os.path.join(fc, "synthetic_alpha_p*.mp4")))
        cr = score(cal[0], clip, slot, cap=cap) if cal else dict(error="no cloud alpha")
        gotc = "n/a" if "error" in cr else ("REFUSE" if cr["refused"] else "PASS")
        bad += (got != want) + (gotc != want_cloud)
        rows.append((clip, slot, got, want, gotc, want_cloud, r, cr, note))
        print(f"      AUTHORED {got} (expected {want}) {'ok' if got == want else '🔴 MISMATCH'}"
              f"   |   CLOUD'S OWN {gotc} (expected {want_cloud}) "
              f"{'ok' if gotc == want_cloud else '🔴 MISMATCH'}")
        if "error" not in cr:
            print(f"      cloud alpha: coverage {100*cr['coverage_mean']:.2f} %, identity "
                  f"{cr['identity']}" + (f" - {'; '.join(cr['refusal'])}" if cr["refusal"] else ""))
        print()
    print(f"{'arm':26s} {'source':>12s} {'cov%':>7s} {'hole%':>7s} {'t1-bnd%':>8s} {'ident':>9s} "
          f"{'AUTHORED':>9s} {'CLOUD':>7s}")
    for clip, slot, got, want, gotc, wc, r, cr, note in rows:
        print(f"{clip+'/'+slot:26s} {r['shape_source']:>12s} {100*r['coverage_mean']:6.2f}% "
              f"{100*r['domain_fill_mean']:6.1f}% {100*r['tier1_boundary_frac_mean']:7.2f}% "
              f"{r.get('identity_gate','n/a'):>9s} {got:>9s} {gotc:>7s}   {note}")
    print()
    if out_json:
        json.dump([dict(clip=c, slot=s, authored=r, cloud=cr, note=nt)
                   for c, s, _g, _w, _gc, _wc, r, cr, nt in rows],
                  io.open(out_json, "w", encoding="utf-8"), indent=1)
        print(f"-> {out_json}")
    if bad:
        print(f"🔴 {bad} control(s) disagree - the alpha authoring is NOT validated.")
    else:
        print("All controls reproduce. Every arm's AUTHORED matte is usable and slot-bound; the "
              "cloud's own matte is refused on three of the four, which is exactly what shipped.")
    return 1 if bad else 0


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--clip", default="")
    ap.add_argument("--slot", default="p1")
    ap.add_argument("--from-cloud", default="")
    ap.add_argument("--bundle", default="", help="override the to_cloud dir Tier-1 is read from")
    ap.add_argument("--out", default="", help="write synthetic_alpha_<slot>.mp4 here")
    ap.add_argument("--json", default="", help="also dump the report")
    ap.add_argument("--cap", type=int, default=10 ** 9)
    ap.add_argument("--selftest", action="store_true")
    a = ap.parse_args()
    if a.selftest:
        return selftest(a.cap, a.json)
    if not (a.clip and a.from_cloud):
        sys.exit("need --clip and --from-cloud (or --selftest)")
    fc = a.from_cloud if os.path.isabs(a.from_cloud) else os.path.join(HERE, a.from_cloud)
    out = (a.out if not a.out or os.path.isabs(a.out) else os.path.join(HERE, a.out))
    r = author(a.clip, a.slot, fc, bundle=a.bundle, out_dir=out, cap=a.cap)
    if a.json:
        json.dump(r, io.open(a.json, "w", encoding="utf-8"), indent=1)
    if "error" in r:
        print(f"  ERROR {r['error']}")
        return 1
    return 1 if r["refused"] else 0


if __name__ == "__main__":
    sys.exit(main())
