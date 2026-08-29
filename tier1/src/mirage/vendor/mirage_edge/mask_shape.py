"""MIRAGE silhouette shape-canonicalisation + temporal mitigation -- VENDORED, DO NOT EDIT.

WHAT THIS IS
------------
The mask-shape half of the MIRAGE re-ID defence (change #2): `_shape_polys()` canonicalises each
person component's outline, `mask_mitigate()` applies the temporal running-max union, runs the
shape op and re-OR-s the result with both the union and the current frame.

PROVENANCE (lifted byte-for-byte, do not reformat, do not "clean up")
--------------------------------------------------------------------
  source file : tree/mirage_edge_deploy/tier1_raspberry_pi5/mirage_tier1.py
  source lines: 789-1520 inclusive (the whole "silhouette mitigation (mask shape channel)"
                section), i.e.
                    _radial_profile   :790-819    helper, used by radiallp + ksame
                    _radial_poly      :822-826    helper, used by radiallp + ksame
                    _shape_polys      :829-1388
                    mask_mitigate     :1391-1520
  repo        : internal source mirror @ c6d82dd59b26616ca9f1e144412978a298368a15
  last commit touching the source file:
                317c937c7c5b9d566d09946a5926c028a06372a2
                "SHIP e2: make it the default preset, and fix the provenance divergence that
                 created" (2026-08-14)

Everything from the `# ----- silhouette mitigation (mask shape channel) -----` banner to the end
of this file is that slice, unmodified -- including comments, spelling and blank lines. No import
adjustment was needed: the lifted code touches only `cv2`, `np` and the module-global `C`, all
three of which exist here. Verify with a byte diff of this file's tail against the source slice;
it must be EMPTY.

THE MEASURED ARM
----------------
  MASK_SHAPE_MODE   = "bbox"   -- the axis-aligned per-component bounding rectangle
  MASK_TEMPORAL_WIN = 2        -- frames of running-max history the CALLER must supply in `hist`

MASK_TEMPORAL_WIN = 2 IS THE VALUE LEDGER §A.6o MEASURED "bbox" AT. The rectangle on its own is
not the measured arm: §A.6o put `bbox` THROUGH mask_mitigate at a 2-frame temporal window and
scored it +7.16 pp of re-ID lift over its own measured null (raw silhouette = +31.88 pp), with the
frozen gait model landing at chance. Quote that number only for shape=bbox AND win=2, and only for
the MIRAGE host it was measured on -- it does not describe this host.

Note that `mask_mitigate()` does NOT read MASK_TEMPORAL_WIN itself; the caller owns the history
deque and passes the last N dilated masks as `hist`. The knob lives in C below so the host adapter
has one authoritative place to read the shipped window length from.

§2 (the superset guarantee) is an EMERGENT PROPERTY OF THIS EXACT BODY: the final
`return simp | sm | cur` is what makes the emitted mask a superset of the temporal union and of the
current frame, so mitigation can only ever ADD grey and never reveal. For "bbox" it holds twice
over, since cv2.boundingRect(c) already contains c. Do not touch the return, the `| sm`, or the
bbox merge loop -- each was measured or audited as written.

DEPENDENCIES: cv2 + numpy only (a hard rule for this vendored module).

HOW TO CONSUME IT
-----------------
Import and drive it from an adapter; never edit anything below the banner. Per-clip / per-frame
runtime state (`C._MASK_DISPLACE_SEED`, `C._MASK_DISPLACE_PHASE`) is set by the caller in MIRAGE
too -- it is unused by mode="bbox" but is kept here so every other mode stays reachable.
"""
import cv2
import numpy as np


class _MaskShapeConfig(object):
    """Mutable stand-in for MIRAGE's `config` module, holding ONLY the knobs the lifted code
    actually reads via `getattr(C, "NAME", default)`.

    Every value below is the SHIPPED default from
    tree/mirage_edge_deploy/tier1_raspberry_pi5/config.py (repo c6d82dd; that file last touched by
    eeb662b "privacy: default silhouette mode was the WORST measured", 2026-08-08), resolved with
    NO MIRAGE_* environment variables set. The line number of each knob in config.py is given so
    the default can be re-checked against its rationale.

    The set is exhaustive: `grep -o 'getattr(C, "[A-Za-z_0-9]*"'` over source lines 790-1520 yields
    36 distinct names and all 36 are defined here. A missing name would silently fall back to the
    getattr default baked into the lifted code, which is NOT always the shipped value (e.g.
    MASK_DISPLACE_AMP_FRAC ships at 0.25 but the inline default is 0.10).

    Mutate it from the host adapter (`C.MASK_SHAPE_MODE = "bbox"`); it is a plain instance, so
    attribute assignment shadows the class default and nothing global is patched.
    """

    # ---- the shape channel itself -------------------------------------------------------
    MASK_SHAPE_MODE = "bbox"          # config.py:532. "none"|"hull"|"ellipse"|"displace"|"bbox"|
                                      # "radiallp"|"ksame"|"bands"|"dirbands"|"mounds"|"close",
                                      # or a "+"-joined composition. Default changed
                                      # displace -> bbox on 2026-08-08 (§A.6o).
    MASK_SIMPLIFY_EPS = 0.01          # config.py:829. approxPolyDP epsilon as a fraction of each
                                      # contour's perimeter. NOT read via getattr -- the caller
                                      # passes it as mask_mitigate()'s `eps_frac` argument -- but
                                      # kept here so the adapter has the shipped value.

    # ---- temporal running-max window ----------------------------------------------------
    # The caller owns the history deque; mask_mitigate() just consumes `hist`.
    MASK_TEMPORAL_S = 0.14            # config.py:133. The window as a DURATION.
    MASK_TEMPORAL_WIN = 2             # config.py:512 = _frames(0.14) at EMIT_FPS=15 (config.py:27)
                                      # = max(1, round(0.14*15)) = 2. THIS IS THE VALUE §A.6o
                                      # MEASURED bbox AT. It is fps-coupled: at EMIT_FPS=10 the
                                      # MIRAGE formula resolves to 1, which is the weakest window.
                                      # Pin it, never re-derive it from this host's fps.

    # ---- mode "bbox" --------------------------------------------------------------------
    MASK_BBOX_MERGE = True            # config.py:558. Merge rects that INTERSECT, to a fixed
                                      # point; non-intersecting rects are left alone so two
                                      # separate people are never fused into one box.
    MASK_BBOX_PAD_FRAC = 0.0          # config.py:562. Outward pad as a FRACTION of each rect's own
                                      # w/h (never absolute px). 0.0 = the exact bounding box.

    # ---- mode "displace" ----------------------------------------------------------------
    MASK_DISPLACE_AMP_FRAC = 0.25     # config.py:571 (inline getattr default is 0.10 -- differs)
    MASK_DISPLACE_HARMONICS = 3       # config.py:637
    MASK_DISPLACE_PHASE_STEP = 0.35   # config.py:638. Phase advance per EMITTED frame; read by
                                      # mode "mounds" to recover a frame index.
    MASK_DISPLACE_RESEED_PHASE = 0.0  # config.py:641. 0.0 = never re-seed = SHIPPED (epoch == 0).
    MASK_DISPLACE_AMP_HEAD = None     # config.py:622. None = uniform amplitude = SHIPPED.
    MASK_DISPLACE_HEAD_FRAC = 0.15    # config.py:626
    MASK_DISPLACE_HEAD_BLEND = 0.10   # config.py:630
    MASK_DISPLACE_UPRIGHT_MIN = 1.2   # config.py:632

    # ---- per-clip / per-frame RUNTIME state, set by the caller ---------------------------
    _MASK_DISPLACE_SEED = 0           # config.py:654. Overwritten per CLIP from the OS CSPRNG --
                                      # never derived from identity or content.
    _MASK_DISPLACE_PHASE = 0.0        # config.py:655. Overwritten per EMITTED frame.

    # ---- mode "ellipse" -----------------------------------------------------------------
    MASK_ELLIPSE_INFLATE = 1.15       # config.py:656

    # ---- mode "radiallp" (also supplies the angular grid for "ksame") -------------------
    MASK_RADIALLP_KEEP = 4            # config.py:672
    MASK_RADIALLP_BINS = 180          # config.py:676

    # ---- mode "close" -------------------------------------------------------------------
    MASK_CLOSE_KERNEL_FRAC = 0.25     # config.py:679

    # ---- mode "mounds" ------------------------------------------------------------------
    MASK_MOUND_N = 4                  # config.py:689
    MASK_MOUND_HEIGHT = 0.13          # config.py:691
    MASK_MOUND_WIDTH_RAD = 0.55       # config.py:694
    MASK_MOUND_DRIFT = 0.020          # config.py:697

    # ---- mode "bands" -------------------------------------------------------------------
    MASK_BAND_N = (4, 7)              # config.py:709-710, inclusive range
    MASK_BAND_AMP = (0.10, 0.40)      # config.py:715-716
    MASK_BAND_AMP_STEP = 0.0025       # config.py:777
    MASK_BAND_ALPHA = 2.5             # config.py:779, Dirichlet concentration for band widths
    MASK_BAND_MIN_W = 0.10            # config.py:783
    MASK_BAND_BLEND_FRAC = 0.25       # config.py:787

    # ---- mode "dirbands" ----------------------------------------------------------------
    MASK_DIRBAND_N = (4, 7)           # config.py:737-738
    MASK_DIRBAND_AMP = (0.10, 0.30)   # config.py:739-740
    MASK_DIRBAND_UPRIGHT_MIN = 0.0    # config.py:763. 0.0 = NO upright guard, deliberately.
    MASK_DIRBAND_THETA = None         # config.py:770. None => band axis drawn per clip.

    # ---- COMPACT: contract the perturbed outline toward its own centroid ----------------
    # Read by mask_mitigate() itself (not by a mode). Both OFF by default, which is SHIPPED --
    # with both off the compact block is skipped entirely and the code path is bit-identical to
    # the version that predates it. Safe by construction either way: the final `| sm | cur` puts
    # any true-person pixel straight back, so a contraction can only remove ADDED area.
    MASK_COMPACT_SCALE = 1.0          # config.py:751. 1.0 = OFF (fixed contraction factor).
    MASK_COMPACT_TARGET = None        # config.py:753. None = OFF (target emitted-area ratio,
                                      # solved per frame by bisection). Wins over _SCALE if set.
    MASK_COMPACT_ITERS = 6            # config.py:760. Bisection steps for _TARGET.

    # ---- mode "ksame" -------------------------------------------------------------------
    MASK_KSAME_SCALE = 1.0            # config.py:806
    # config.py:809-826, verbatim. p50 of 149820 mean-radius-normalised boundary profiles over
    # CASIA-B TRAIN ids 001..074 (180 angular bins). DOMAIN CAVEAT: derived on a full-body walking
    # corpus; re-derive on target-domain masks before shipping "ksame".
    MASK_KSAME_TEMPLATE = (
        0.5824, 0.5817, 0.5824, 0.5929, 0.6108, 0.6269, 0.6405, 0.6531, 0.6656, 0.6777, 0.6913,
        0.7046, 0.7181, 0.7303, 0.7452, 0.7589, 0.7751, 0.7910, 0.8075, 0.8253, 0.8426, 0.8668,
        0.8939, 0.9122, 0.9369, 0.9658, 0.9978, 1.0323, 1.0710, 1.1141, 1.1616, 1.2153, 1.2732,
        1.3367, 1.4016, 1.4636, 1.5166, 1.5642, 1.6124, 1.6802, 1.8187, 2.0979, 2.3009, 2.3927,
        2.4230, 2.4160, 2.3719, 2.2800, 2.1326, 1.9426, 1.8055, 1.7225, 1.6695, 1.6241, 1.5778,
        1.5280, 1.4733, 1.4125, 1.3479, 1.2863, 1.2288, 1.1767, 1.1297, 1.0864, 1.0460, 1.0081,
        0.9744, 0.9457, 0.9130, 0.8822, 0.8574, 0.8325, 0.8103, 0.7888, 0.7682, 0.7503, 0.7316,
        0.7166, 0.7021, 0.6879, 0.6748, 0.6625, 0.6506, 0.6403, 0.6281, 0.6139, 0.5975, 0.5880,
        0.5910, 0.5947, 0.5854, 0.5651, 0.5490, 0.5405, 0.5366, 0.5330, 0.5286, 0.5243, 0.5210,
        0.5179, 0.5159, 0.5144, 0.5144, 0.5154, 0.5170, 0.5200, 0.5246, 0.5302, 0.5378, 0.5479,
        0.5590, 0.5726, 0.5902, 0.6062, 0.6274, 0.6508, 0.6773, 0.7087, 0.7466, 0.7892, 0.8397,
        0.9009, 0.9736, 1.0594, 1.1522, 1.2431, 1.3382, 1.4448, 1.5591, 1.6732, 1.7771, 1.8654,
        1.9344, 1.9767, 2.0043, 2.0275, 2.0365, 2.0382, 2.0078, 1.9294, 1.8256, 1.7043, 1.5594,
        1.4303, 1.3234, 1.2360, 1.1571, 1.0831, 1.0150, 0.9537, 0.8977, 0.8485, 0.8032, 0.7642,
        0.7306, 0.7021, 0.6758, 0.6555, 0.6332, 0.6150, 0.5989, 0.5839, 0.5712, 0.5604, 0.5509,
        0.5429, 0.5367, 0.5318, 0.5286, 0.5268, 0.5259, 0.5260, 0.5273, 0.5290, 0.5309, 0.5333,
        0.5367, 0.5438, 0.5566, 0.5738)


C = _MaskShapeConfig()


# ------------------------- silhouette mitigation (mask shape channel) -------------------------
def _radial_profile(p, bins):
    """Boundary radius profile r(theta) of ONE contour about its own centroid, resampled onto a
    uniform angular grid of `bins` bins by taking the MAX radius per bin.

    -> (centroid[2], r[bins]) or None if the contour is degenerate.

    The max-per-bin resample is deliberately OUTWARD-BIASED: along any ray from the centroid it
    keeps the farthest boundary point, so the profile describes the person's outer envelope and
    the ops built on it can only ever bulge. It is also what makes those ops resolution-
    independent -- the grid is an ANGLE grid, so it carries no pixel constant.
    """
    if p.shape[0] < 3:
        return None
    ctr = p.mean(0)
    v = p - ctr
    r = np.linalg.norm(v, axis=1)
    a = np.arctan2(v[:, 1], v[:, 0])
    idx = np.minimum(((a + np.pi) * (bins / (2.0 * np.pi))).astype(np.int32), bins - 1)
    rb = np.zeros(bins, np.float32)
    np.maximum.at(rb, idx, r)
    hit = rb > 0
    if not hit.any():
        return None
    if not hit.all():
        # circularly interpolate the empty bins (a contour rarely covers all of them)
        ii = np.nonzero(hit)[0].astype(np.float32)
        xs = np.concatenate([ii - bins, ii, ii + bins])
        ys = np.tile(rb[hit], 3)
        rb = np.interp(np.arange(bins, dtype=np.float32), xs, ys).astype(np.float32)
    return ctr, rb


def _radial_poly(ctr, r, bins):
    """Uniform-angular-grid radius profile -> an OpenCV polygon."""
    th = (np.arange(bins, dtype=np.float32) + 0.5) * (2.0 * np.pi / bins) - np.pi
    pts = np.stack([ctr[0] + r * np.cos(th), ctr[1] + r * np.sin(th)], 1)
    return np.round(pts).astype(np.int32).reshape(-1, 1, 2)


def _shape_polys(cnts, mode, eps_frac):
    """ONE shape-canonicalisation op: external contours -> the polygons to fill.

    Split out of mask_mitigate() on 2026-07-31 so that modes can be COMPOSED ("displace+close")
    without duplicating any of them. Every branch below is the code that used to live inline in
    mask_mitigate(), moved VERBATIM — the pre-existing modes are bit-identical (verified against a
    hash snapshot of 60 sequence/window/mode combinations taken before the move).

    §2 is NOT this function's job: whatever it returns, mask_mitigate() OR-s `sm` and `cur` back
    in, so the emitted mask is a superset of the detected one no matter what a mode does.
    """
    if mode == "hull":
        return [cv2.convexHull(c) for c in cnts]
    if mode == "bbox":
        # FULL AXIS-ALIGNED BOUNDING BOX per person component (user request 2026-08-03: "fill the
        # whole bounding box with the mask, full rectangle rather than the person's silhouette").
        #
        # This is the STRONGEST shape canonicalisation in this function: unlike hull/ellipse, which
        # still leak the body's aspect and lean, a rectangle preserves ONLY the component's position
        # and its width/height extent. Every trace of limb articulation and of the width-as-a-
        # function-of-height profile -- which is the bulk of what a silhouette-gait recogniser reads
        # -- is gone by construction. ⚠️ NOT MEASURED against the CASIA-B adversary; hull is the
        # strongest MEASURED mode (20.33 % NM, §A.6g). Do not quote a privacy number for this mode
        # until an arm is run; the arm that would settle it is the §A.6g harness with mode="bbox".
        #
        # §2 IS SAFE BY CONSTRUCTION AND STRICTLY MORE SO THAN ANY OTHER MODE: boundingRect(c) is a
        # superset of c for every contour, so the filled rect ⊇ the component before mask_mitigate's
        # `| sm | cur` is even applied. A rectangle can only ever ADD gray, never reveal.
        #
        # COST, which is the real trade here: a rectangle over an upright person is roughly the
        # inverse of the person's box-fill ratio -- order 2-2.5x the silhouette area -- and that
        # area is the cloud's inpainting bill. build_run.py MEASURES the emitted coverage rather
        # than assuming it, and writes it to the run's AREA.json.
        #
        # MERGE (MASK_BBOX_MERGE, default on): when one person's mask breaks into several blobs
        # (occlusion, a limb separated by the erode margin) per-component rects would emit several
        # overlapping rectangles instead of one. Rects that INTERSECT are merged, iterated to a
        # fixed point. Rects that do not intersect are left alone, so two genuinely separate people
        # are never fused into one box spanning the gap between them.
        rects = [list(cv2.boundingRect(c)) for c in cnts]        # x, y, w, h
        if bool(getattr(C, "MASK_BBOX_MERGE", True)) and len(rects) > 1:
            changed = True
            while changed:
                changed = False
                for i in range(len(rects)):
                    for j in range(i + 1, len(rects)):
                        ax, ay, aw, ah = rects[i]
                        bx, by, bw, bh = rects[j]
                        if ax < bx + bw and bx < ax + aw and ay < by + bh and by < ay + ah:
                            nx, ny = min(ax, bx), min(ay, by)
                            rects[i] = [nx, ny, max(ax + aw, bx + bw) - nx,
                                        max(ay + ah, by + bh) - ny]
                            rects.pop(j)
                            changed = True
                            break
                    if changed:
                        break
        # PAD as a FRACTION of the rect's own size, never absolute px -- the project rule against
        # fitted constants. 0.0 (the default) is the exact bounding box.
        pad = float(getattr(C, "MASK_BBOX_PAD_FRAC", 0.0))
        polys = []
        for x, y, w, h in rects:
            if pad > 0:
                dx, dy = int(round(w * pad)), int(round(h * pad))
                x, y, w, h = x - dx, y - dy, w + 2 * dx, h + 2 * dy
            x2, y2 = x + w - 1, y + h - 1
            polys.append(np.array([[[x, y]], [[x2, y]], [[x2, y2]], [[x, y2]]], np.int32))
        return polys
    if mode == "displace":
        # OUTWARD-ONLY DISPLACEMENT FIELD (2026-07-25). The §2-safe form of the published
        # silhouette-deformation approach (Anonymization of Human Gait in Video Based on
        # Silhouette Deformation and Texture Transfer, IEEE 2022), which reports CNN gait
        # recognition collapsing 100 % -> 1.57 % by perturbing static body shape AND walking
        # rhythm. A GENERAL displacement field moves boundary points both ways and would break
        # our "emitted ⊇ detected" guarantee, so every displacement here is >= 0 along the
        # outward normal: the boundary can only bulge, never cut in.
        #   * low-frequency angular field  -> perturbs STATIC body shape (build, width profile)
        #   * phase advances per frame     -> perturbs WALKING RHYTHM across the sequence
        #   * per-clip seed                -> never derived from identity or content, so two
        #                                     clips of the same person get different fields
        # Far cheaper in area than hull/ellipse (a few px of bulge vs a whole convex blob),
        # which is what makes it viable against the cloud's halo constraint.
        # RELATIVE amplitude: a fraction of each component's own mean radius, NOT absolute px.
        # An absolute value cannot generalise -- 6 px is ~10 % of a 64x64 CASIA-B crop and
        # negligible on a 1264^2 native mask, so a CASIA-B measurement taken with absolute px
        # would not transfer to device footage at all (and would hardcode a fitted constant,
        # which the project rule forbids). Scaling by the component radius makes the perturbation
        # the same PROPORTION of the person at every resolution and distance.
        amp_frac = float(getattr(C, "MASK_DISPLACE_AMP_FRAC", 0.10))
        nh = int(getattr(C, "MASK_DISPLACE_HARMONICS", 3))
        ph = float(getattr(C, "_MASK_DISPLACE_PHASE", 0.0))
        # PER-EPOCH RE-SEEDING (2026-07-31). The phase advance already perturbs walking RHYTHM,
        # but the field's SHAPE (coef/off) is fixed for the whole clip, so a long sequence still
        # carries one consistent deformation an attacker can average over. Re-drawing the field
        # every `MASK_DISPLACE_RESEED_PHASE` radians of phase breaks that: the deformation becomes
        # a piecewise-constant random process instead of a single per-clip transform. It costs
        # ZERO extra area (the amplitude is unchanged; only the field's direction changes), which
        # is why it is worth measuring against an area budget.
        # 0.0 = never re-seed = the SHIPPED behaviour, bit-identical (epoch is then always 0).
        _reseed = float(getattr(C, "MASK_DISPLACE_RESEED_PHASE", 0.0))
        _epoch = int(ph / _reseed) if _reseed > 0 else 0
        rng = np.random.default_rng(int(getattr(C, "_MASK_DISPLACE_SEED", 0))
                                    + 1000003 * _epoch)
        coef = rng.uniform(0.3, 1.0, nh); off = rng.uniform(0, 2 * np.pi, nh)

        # HEAD-vs-BODY AMPLITUDE SPLIT (2026-07-26). A single amplitude has to serve two very
        # different jobs: the torso/limbs carry nearly all of the silhouette's gait signal and want
        # a LARGE bulge, while the head is small, high-contrast and the thing a viewer looks at
        # first, so the same bulge there reads as a deformed skull. This lets the head run a lower
        # amplitude than the body.
        #
        # DERIVED FROM THE MASK ALONE — no pose. That is a hard constraint, not an oversight: the
        # CASIA-B silhouette adversary consumes bare binary masks, so a pose-driven head region
        # could never be evaluated against the only real gait attacker we have (see the block
        # comment above). The head band is therefore the top `MASK_DISPLACE_HEAD_FRAC` of the
        # COMPONENT'S OWN bounding-box height — a proportion of the person, so it transfers between
        # a 64x64 CASIA-B crop and a native 1264^2 mask without hardcoding any pixel constant.
        #
        # UPRIGHT GUARD: "top of the bbox = head" only holds for an upright person. For a component
        # wider than it is tall (someone lying down, a torso-only sliver at the frame edge, two
        # people merged into one blob) the assumption fails, so those fall back to the uniform
        # amplitude rather than silently perturbing whatever happens to be at the top.
        #
        # §2 IS UNAFFECTED: every per-vertex amplitude is still >= 0 and the displacement is still
        # outward-only along the normal, so the emitted mask remains a superset of the detected one.
        amp_head = getattr(C, "MASK_DISPLACE_AMP_HEAD", None)
        amp_head = None if amp_head is None else float(amp_head)
        head_frac = float(getattr(C, "MASK_DISPLACE_HEAD_FRAC", 0.15))
        head_blend = float(getattr(C, "MASK_DISPLACE_HEAD_BLEND", 0.10))
        upright_min = float(getattr(C, "MASK_DISPLACE_UPRIGHT_MIN", 1.2))

        polys = []
        for c in cnts:
            p = c.reshape(-1, 2).astype(np.float32)
            ctr = p.mean(0)
            v = p - ctr
            r = np.linalg.norm(v, axis=1) + 1e-6
            ang = np.arctan2(v[:, 1], v[:, 0])
            # sum of harmonics in [0,1] -> displacement is non-negative by construction
            d = np.zeros_like(ang)
            for k in range(nh):
                d += coef[k] * (0.5 + 0.5 * np.sin((k + 1) * ang + off[k] + ph))
            d = d / max(1e-6, coef.sum())                     # normalise to [0,1]
            rmean = float(r.mean())

            if amp_head is None:
                amp = amp_frac * rmean                # scalar: original uniform behaviour
            else:
                y = p[:, 1]; x = p[:, 0]
                y0 = float(y.min()); h = max(1e-6, float(y.max()) - y0)
                w = max(1e-6, float(x.max()) - float(x.min()))
                if h / w >= upright_min:
                    # y grows DOWNWARD, so y0 is the crown. t ramps 0 (head) -> 1 (body) across
                    # a blend zone; a hard step would leave a visible ledge at the neck.
                    t = np.clip(((y - y0) / h - head_frac) / max(1e-6, head_blend), 0.0, 1.0)
                    frac = amp_head + (amp_frac - amp_head) * t
                else:
                    frac = np.full(p.shape[0], amp_frac, dtype=np.float32)
                amp = frac * rmean                    # per-VERTEX amplitude
            newp = ctr + v / r[:, None] * (r + amp * d)[:, None]
            polys.append(np.round(newp).astype(np.int32).reshape(-1, 1, 2))
        return polys
    if mode == "ellipse":
        polys = []
        for c in cnts:
            fit = None
            if len(c) >= 5:
                # cv2.fitEllipse needs >=5 points but STILL returns NaN on degenerate (near-
                # collinear) contours — a 1-px sliver of a person at the frame edge is enough.
                # Guard on the VALUES, not just the point count: an unguarded int(NaN) killed a
                # 40-minute CASIA-B sweep at subject 34/50 on 2026-07-25.
                try:
                    _f = cv2.fitEllipse(c)
                    if all(np.isfinite(v) for v in (_f[0][0], _f[0][1], _f[1][0], _f[1][1], _f[2])):
                        fit = _f
                except cv2.error:
                    fit = None
            if fit is not None:
                (ex, ey), (ea, eb), eang = fit
                # Inflate so the blob CONTAINS the component it replaces; the union below makes
                # this a correctness backstop rather than a requirement, but a tight ellipse would
                # leave the real contour poking out and re-leak the very shape we are hiding.
                f = float(getattr(C, "MASK_ELLIPSE_INFLATE", 1.15))
                pts = cv2.ellipse2Poly((int(ex), int(ey)),
                                       (max(1, int(ea * f / 2)), max(1, int(eb * f / 2))),
                                       int(eang), 0, 360, 5)
                polys.append(pts.reshape(-1, 1, 2))
            else:
                polys.append(cv2.convexHull(c))
        return polys
    if mode == "radiallp":
        # OUTWARD-ONLY RADIAL LOW-PASS (2026-07-31). What a silhouette-gait recogniser reads is the
        # body-WIDTH profile and limb articulation — i.e. the HIGH angular frequencies of the
        # boundary radius r(theta) about the component centroid. `hull` destroys those by going
        # convex, which is why it scores 20.33 % NM (§A.6g) — but convexity is a blunt instrument
        # and the user has banned it. This mode attacks the same cue WITHOUT going convex:
        #   1. resample r(theta) onto a uniform angular grid, taking the MAX radius per bin. That
        #      alone is a "radial hull": along any ray from the centroid everything up to the
        #      farthest boundary point is filled, so inter-limb gaps close, but the shape may still
        #      be non-convex (a bent arm, a lunging stride).
        #   2. circular low-pass: keep only the DC term + the first MASK_RADIALLP_KEEP harmonics of
        #      that profile. This is what erases the width/articulation signature.
        #   3. take max(r_binned, r_smoothed) so the result only ever bulges OUTWARD.
        # RESOLUTION-INDEPENDENT BY CONSTRUCTION: the grid is angular and the retained harmonic
        # count is a pure integer — there is no pixel constant anywhere, so the op is the same
        # proportion of the person on a 64x64 CASIA-B crop and on a native 1264^2 mask.
        keep = max(0, int(getattr(C, "MASK_RADIALLP_KEEP", 4)))
        bins = max(16, int(getattr(C, "MASK_RADIALLP_BINS", 180)))
        polys = []
        for c in cnts:
            prof = _radial_profile(c.reshape(-1, 2).astype(np.float32), bins)
            if prof is None:
                polys.append(c)
                continue
            ctr, rb = prof
            F = np.fft.rfft(rb)
            F[keep + 1:] = 0                                  # circular low-pass
            rs = np.fft.irfft(F, n=bins).astype(np.float32)
            polys.append(_radial_poly(ctr, np.maximum(rb, rs), bins))   # OUTWARD-ONLY
        return polys
    if mode == "ksame":
        # k-SAME SILHOUETTE COLLAPSE (2026-07-31) — the mask analogue of what the pose anonymiser
        # already does to bone lengths with `_TEMPLATE_RATIOS`. Instead of perturbing each person's
        # own outline (which leaves their build recoverable), every silhouette is pushed OUT to a
        # shared POPULATION TEMPLATE boundary profile: the emitted radius is
        #       r_out(theta) = max( r_person(theta), s * T(theta) )
        # where T is a scale-free canonical profile (mean radius 1) and `s` scales it to this
        # person's own mean radius, so it carries no absolute size and no pixel constant. Where the
        # person is NARROWER than the template their own width profile is replaced by the shared
        # one; where they are wider they stick out (which is what keeps the area cost small and
        # keeps §2 trivially satisfied — the op is outward-only by construction).
        # The template is a POPULATION CONSTANT, exactly like the pose template. See
        # config.MASK_KSAME_TEMPLATE for its provenance and its domain caveat.
        bins = max(16, int(getattr(C, "MASK_RADIALLP_BINS", 180)))
        tmpl = np.asarray(getattr(C, "MASK_KSAME_TEMPLATE", ()), dtype=np.float32)
        if tmpl.size < 8:
            return [cv2.approxPolyDP(c, eps_frac * cv2.arcLength(c, True), True) for c in cnts]
        if tmpl.size != bins:                                 # resample the constant, circularly
            src = (np.arange(tmpl.size, dtype=np.float32) + 0.5) / tmpl.size
            dst = (np.arange(bins, dtype=np.float32) + 0.5) / bins
            tmpl = np.interp(dst, np.concatenate([src - 1, src, src + 1]),
                             np.tile(tmpl, 3)).astype(np.float32)
        tmpl = tmpl / max(1e-6, float(tmpl.mean()))           # scale-free: mean radius == 1
        scale = float(getattr(C, "MASK_KSAME_SCALE", 1.0))
        polys = []
        for c in cnts:
            prof = _radial_profile(c.reshape(-1, 2).astype(np.float32), bins)
            if prof is None:
                polys.append(c)
                continue
            ctr, rb = prof
            polys.append(_radial_poly(ctr, np.maximum(rb, tmpl * (scale * float(rb.mean()))),
                                      bins))
        return polys
    if mode == "bands":
        # RANDOMISED HORIZONTAL BANDS, PER-BAND AMPLITUDE (2026-07-31, user-specified mechanism).
        # The silhouette is cut by imaginary horizontal lines into N sections (N random in
        # MASK_BAND_N), at RANDOMLY PLACED boundaries, and each band displaces with ITS OWN
        # amplitude drawn from MASK_BAND_AMP in MASK_BAND_AMP_STEP steps. Every draw comes from
        # the per-CLIP seed (never identity/content), so each clip presents a DIFFERENT width-
        # profile distortion — aimed at the two audited findings at once:
        #   * GEI (the strongest measured attacker) reads body width as a function of height;
        #     per-band random amplitudes corrupt that profile differently per clip;
        #   * every DETERMINISTIC arm hands a gallery-adapting attacker 9-38 pp back; a band
        #     layout the attacker has never seen cannot be replicated offline (the property the
        #     pose channel's per-clip reseed already demonstrates at +1-3 pp).
        # Geometry: band boundaries are FRACTIONS of the component's own bbox height (no pixel
        # constants); the amplitude profile is blended across band edges over MASK_BAND_BLEND of
        # the height (a hard step would leave a visible ledge = inpaint hazard + fingerprint);
        # the harmonic field/phase machinery is displace's own, so walking-rhythm perturbation
        # is preserved; upright guard falls back to a UNIFORM amplitude = the mean of this
        # clip's drawn band amplitudes (stays random per clip, stays inside MASK_BAND_AMP).
        # Outward-only along the normal, so §2 holds by construction.
        nh = int(getattr(C, "MASK_DISPLACE_HARMONICS", 3))
        ph = float(getattr(C, "_MASK_DISPLACE_PHASE", 0.0))
        _reseed = float(getattr(C, "MASK_DISPLACE_RESEED_PHASE", 0.0))
        _epoch = int(ph / _reseed) if _reseed > 0 else 0
        seed = int(getattr(C, "_MASK_DISPLACE_SEED", 0))
        rng = np.random.default_rng(seed + 1000003 * _epoch)
        coef = rng.uniform(0.3, 1.0, nh); off = rng.uniform(0, 2 * np.pi, nh)
        n_lo, n_hi = (int(x_) for x_ in getattr(C, "MASK_BAND_N", (4, 7)))
        a_lo, a_hi = (float(x_) for x_ in getattr(C, "MASK_BAND_AMP", (0.10, 0.40)))
        a_step = float(getattr(C, "MASK_BAND_AMP_STEP", 0.0025))
        alpha = float(getattr(C, "MASK_BAND_ALPHA", 2.5))
        min_w = float(getattr(C, "MASK_BAND_MIN_W", 0.10))
        blendf = float(getattr(C, "MASK_BAND_BLEND_FRAC", 0.25))
        upright_min = float(getattr(C, "MASK_DISPLACE_UPRIGHT_MIN", 1.2))
        GRID = 512                                       # band-profile sample grid over height
        polys = []
        for ci_, c in enumerate(cnts):
            p = c.reshape(-1, 2).astype(np.float32)
            if p.shape[0] < 8:
                polys.append(c)
                continue
            ctr = p.mean(0)
            v = p - ctr
            r = np.linalg.norm(v, axis=1) + 1e-6
            ang = np.arctan2(v[:, 1], v[:, 0])
            d = np.zeros_like(ang)
            for k in range(nh):
                d += coef[k] * (0.5 + 0.5 * np.sin((k + 1) * ang + off[k] + ph))
            d = d / max(1e-6, coef.sum())
            rmean = float(r.mean())
            # per-CLIP, per-COMPONENT draws — stable across frames by construction.
            # Count uniform{4..7}; widths Dirichlet(alpha=2.5) (random fractions that sum to 1
            # = bands always tile the body), RESAMPLED until every band >= MASK_BAND_MIN_W of
            # the height (a sliver band cannot carry an amplitude difference — it just becomes
            # a ledge); amplitudes on the a_step grid within MASK_BAND_AMP.
            rb = np.random.default_rng(seed + 424243 * (ci_ + 1))
            nb = int(rb.integers(n_lo, n_hi + 1))
            widths = None
            for _try in range(500):
                cand = rb.dirichlet(np.full(nb, alpha))
                if float(cand.min()) >= min_w:
                    widths = cand
                    break
            if widths is None:                            # pathological draw streak: equal split
                widths = np.full(nb, 1.0 / nb)
            steps = int(round((a_hi - a_lo) / max(1e-9, a_step)))
            amps = a_lo + a_step * rb.integers(0, steps + 1, nb)
            cuts = np.cumsum(widths)[:-1]
            y = p[:, 1]; x = p[:, 0]
            y0 = float(y.min()); h = max(1e-6, float(y.max()) - y0)
            w = max(1e-6, float(x.max()) - float(x.min()))
            if h / w >= upright_min:
                tg = (np.arange(GRID, dtype=np.float32) + 0.5) / GRID
                prof = amps[np.searchsorted(cuts, tg)].astype(np.float32)
                # blend each edge over 25 % of the NARROWER neighbour: linear lerp on the grid.
                # Blend half-widths are <= 12.5 % of each band's own width, so zones never
                # overlap and every band keeps a flat plateau at its drawn amplitude.
                for e in range(nb - 1):
                    b = blendf * float(min(widths[e], widths[e + 1]))
                    lo, hi = float(cuts[e]) - b / 2, float(cuts[e]) + b / 2
                    m_ = (tg >= lo) & (tg <= hi)
                    if m_.any():
                        prof[m_] = amps[e] + (amps[e + 1] - amps[e]) * \
                            ((tg[m_] - lo) / max(1e-9, hi - lo))
                frac = np.interp((y - y0) / h, tg, prof).astype(np.float32)
            else:
                # upright guard: uniform fallback at the MEAN of this clip's drawn amplitudes
                # (stays per-clip random, stays inside MASK_BAND_AMP)
                frac = np.full(p.shape[0], float(amps.mean()), np.float32)
            amp = frac * rmean
            newp = ctr + v / r[:, None] * (r + amp * d)[:, None]
            polys.append(np.round(newp).astype(np.int32).reshape(-1, 1, 2))
        return polys
    if mode == "dirbands":
        # RANDOM DIRECTIONAL BANDS (2026-07-31, user proposal 17:17; ledger §A.6k-8).
        # Identical band machinery to "bands" above, with ONE change: the band coordinate is
        # not height, it is the projection onto a PER-CLIP RANDOM AXIS
        #     u = x*sin(theta) + y*cos(theta),  theta ~ U(0, pi)
        # normalised over the component's own extent along that axis.
        #
        # WHY. The silhouette adversary reads the body as 16 HORIZONTAL strips
        # (PART_DISCRIM_OURS.json). Horizontal bands align with those strips, so each strip
        # receives ONE consistent amplitude -- its width is shifted, but cleanly, and a clean
        # offset is exactly what a recogniser normalises away. Bands at an angle CUT ACROSS
        # strips, so every strip gets a MIXTURE of amplitudes and its width estimate is
        # BLURRED rather than displaced. theta is drawn per clip, so a gallery-adapting
        # attacker cannot even assume the banding axis.
        #
        # MEASURED (108 clips, 107 probes, GEI vs a DEFENDED gallery, paired McNemar):
        #   56.07 % -> 30.84 % top-1, -25.23 pp, p < 0.0001, area 1.302x.
        # That is the best arm in the whole sweep that fits the strict 1.358x area ceiling,
        # at an area BELOW the shipped mask's own 1.337x, and at -83.5 pp per unit excess
        # area it is 1.35x more efficient than the next best arm measured.
        #
        # NOTE it carries NO time term in its angular profile. Per §A.6k-5 that is not
        # incidental: GEI is the time-average, so a perturbation that ROTATES averages into a
        # uniform offset the attacker normalises away, while one held FIXED within the clip
        # survives the average. Do NOT add a phase-step term here.
        #
        # Outward-only along the radius, so §2 holds by construction.
        nh = int(getattr(C, "MASK_DISPLACE_HARMONICS", 3))
        seed = int(getattr(C, "_MASK_DISPLACE_SEED", 0))
        rng = np.random.default_rng(seed)
        coef = rng.uniform(0.3, 1.0, nh); off = rng.uniform(0, 2 * np.pi, nh)
        n_lo, n_hi = (int(x_) for x_ in getattr(C, "MASK_DIRBAND_N", getattr(C, "MASK_BAND_N", (4, 7))))
        a_lo, a_hi = (float(x_) for x_ in getattr(C, "MASK_DIRBAND_AMP", (0.10, 0.30)))
        alpha = float(getattr(C, "MASK_BAND_ALPHA", 2.5))
        min_w = float(getattr(C, "MASK_BAND_MIN_W", 0.10))
        # UPRIGHT GUARD -- OFF BY DEFAULT FOR THIS MODE, and that is deliberate.
        # The guard exists for "bands", where the band coordinate is HEIGHT: on a wide/short
        # component (person lying down, a fragment) height-banding is meaningless, so it falls
        # back to a uniform swell. `dirbands` bands along a RANDOM AXIS, so there is no
        # privileged orientation to be wrong about and the guard has no rationale here.
        # It was inherited by copying the "bands" branch, and it cost a great deal:
        # censused over 1341 components of our corpus, h/w < 1.2 fires on 45.6 % of them
        # (median h/w is 1.22 -- the threshold sits right on the mode of the distribution).
        # Nearly half of all components were therefore getting the weak uniform fallback on
        # the shipped path while the lab implementation banded every one, which is what
        # produced the systematic 5-17 pp lab-vs-shipped offset in A.6k-13.
        # Set MASK_DIRBAND_UPRIGHT_MIN > 0 to re-enable it.
        upright_min = float(getattr(C, "MASK_DIRBAND_UPRIGHT_MIN", 0.0))
        theta_env = getattr(C, "MASK_DIRBAND_THETA", None)      # None => random per clip
        # DENSIFY FIRST. mask_mitigate simplifies the contour with approxPolyDP at
        # eps_frac * perimeter BEFORE any shape op runs: measured on a real 1264^2 clip that
        # takes 1220 boundary points down to 453 (CHAIN_APPROX_SIMPLE) and then to TWELVE.
        # You cannot place 4-11 random directional bands, each with its own angular phase, on
        # a 12-gon -- the mechanism is crushed before it executes. That is exactly what the
        # port-validation gate caught: 46.73 % (p=0.0525, not significant) against the lab
        # implementation's 30.84 %, with emitted area 1.183x instead of 1.303x.
        #
        # This mode's value is the structure it ADDS, not detail it preserves, so resampling
        # the simplified polygon back up to a workable number of points recovers the mechanism
        # without touching eps_frac -- which is itself a privacy mitigation and must not be
        # weakened to make this mode work.
        def _densify(poly, target=512):
            q = poly.reshape(-1, 2).astype(np.float32)
            if q.shape[0] >= target:
                return q
            closed = np.vstack([q, q[:1]])
            seg = np.linalg.norm(np.diff(closed, axis=0), axis=1)
            total = float(seg.sum())
            if total <= 1e-6:
                return q
            cum = np.concatenate([[0.0], np.cumsum(seg)])
            t = np.linspace(0.0, total, target, endpoint=False)
            xs = np.interp(t, cum, closed[:, 0])
            ys = np.interp(t, cum, closed[:, 1])
            return np.stack([xs, ys], 1).astype(np.float32)

        polys = []
        for ci_, c in enumerate(cnts):
            p = _densify(c)
            if p.shape[0] < 8:
                polys.append(c)
                continue
            ctr = p.mean(0)
            v = p - ctr
            r = np.linalg.norm(v, axis=1) + 1e-6
            ang = np.arctan2(v[:, 1], v[:, 0])
            rmean = float(r.mean())
            rb = np.random.default_rng(seed + 424243 * (ci_ + 1))
            th = float(rb.uniform(0, np.pi)) if theta_env is None else float(theta_env)
            ct_, st_ = np.cos(th), np.sin(th)
            u = p[:, 0] * st_ + p[:, 1] * ct_                   # projection onto the band axis
            u0 = float(u.min()); span = max(1e-6, float(u.max()) - u0)
            h = max(1e-6, float(p[:, 1].max() - p[:, 1].min()))
            w = max(1e-6, float(p[:, 0].max() - p[:, 0].min()))
            # ANGULAR PROFILE -- PER BAND, NOT GLOBAL. This is the whole mechanism and the
            # first port of it got this wrong: mirroring the "bands" mode above, it computed
            # ONE global multi-harmonic field and merely SCALED it per band, so every band
            # bulged at the same angles. Measured, that port scored 52.34 % (-3.74 pp,
            # p = 0.4807, not significant) against the lab implementation's 30.84 % -- a
            # 21.5 pp miss, i.e. it did essentially nothing.
            #
            # What actually works: each band draws its OWN phase, so band k bulges left while
            # band k+1 bulges right. Combined with the random band AXIS, that is a per-region
            # DIRECTIONAL displacement rather than a globally coherent lobe pattern, and it is
            # what blurs the strip-wise width profile the recogniser reads. Do not "unify"
            # this with the bands mode's shared field.
            if h / w < upright_min:                             # upright guard: uniform fallback
                a_mean = 0.5 * (a_lo + a_hi)
                amp = a_mean * 0.5 * (1.0 + np.sin(nh * ang + float(rb.uniform(0, 2 * np.pi))))
            else:
                nb = int(rb.integers(n_lo, n_hi + 1))
                widths = None
                for _try in range(500):
                    cand = rb.dirichlet(np.full(nb, alpha))
                    if float(cand.min()) >= min_w:
                        widths = cand
                        break
                if widths is None:
                    widths = np.full(nb, 1.0 / nb)
                edges = np.concatenate([[0.0], np.cumsum(widths)])
                amps = rb.uniform(a_lo, a_hi, nb)
                phase = rb.uniform(0, 2 * np.pi, nb)            # PER-BAND phase
                un = (u - u0) / span
                amp = np.zeros(p.shape[0], np.float32)
                for i in range(nb):
                    sel = (un >= edges[i]) & (un <= edges[i + 1])
                    if not sel.any():
                        continue
                    amp[sel] = amps[i] * 0.5 * (1.0 + np.sin(nh * ang[sel] + phase[i]))
                k_ = max(1, int(p.shape[0] * 0.01))             # smooth across band seams
                amp = np.convolve(np.r_[amp[-k_:], amp, amp[:k_]],
                                  np.ones(2 * k_ + 1) / (2 * k_ + 1), "same")[k_:-k_][:p.shape[0]]
            newp = ctr + v / r[:, None] * (r + amp * rmean)[:, None]
            polys.append(np.round(newp).astype(np.int32).reshape(-1, 1, 2))
        return polys
    if mode == "mounds":
        # TRAVELLING RAISED-COSINE MOUNDS ON THE RADIUS PROFILE (2026-07-31). Supersedes the
        # rejected "blobs" discs (user: mounds must be "semi-circles or even less ... like goo
        # wiggling on the body", not balls). A disc half-out of the outline decorates the edge
        # with concave notches (perimeter x1.124) and barely changes body WIDTH — the quantity a
        # silhouette recogniser reads. A raised-cosine swelling of r(theta) changes width
        # smoothly, adds no notches (perimeter x1.031 at the reference params) and is far
        # cheaper in area (x1.077 vs x1.283 measured on a real clip at native 1264^2):
        #     r'(th) = r(th) + height*rmean * sum_k amp_k * 1/2(1+cos(pi*dth_k/width_rad))
        #     for |dth_k| < width_rad,   dth_k = wrapped(th - centre_k(t))
        #     centre_k(t) = phase_k + speed_k * t          # slow drift = the "goo" wiggle
        # phase_k / amp_k / speed_k come from the per-CLIP seed (never identity/content);
        # per-component seed offset so two people get different goo. Outward-only (the bump is
        # >= 0) so §2 holds by construction. NO PIXEL CONSTANTS: height is a fraction of the
        # component's mean radius, width is an ANGLE, drift is rad/frame (⚠️ fps-coupled like
        # MASK_TEMPORAL_WIN — pin via env if EMIT_FPS changes).
        n_m = max(0, int(getattr(C, "MASK_MOUND_N", 4)))
        hgt = float(getattr(C, "MASK_MOUND_HEIGHT", 0.13))
        wid = max(1e-3, float(getattr(C, "MASK_MOUND_WIDTH_RAD", 0.55)))
        drift = float(getattr(C, "MASK_MOUND_DRIFT", 0.020))
        ph = float(getattr(C, "_MASK_DISPLACE_PHASE", 0.0))
        _st = float(getattr(C, "MASK_DISPLACE_PHASE_STEP", 0.35))
        tidx = int(round(ph / _st)) if _st > 0 else 0
        seed = int(getattr(C, "_MASK_DISPLACE_SEED", 0))
        polys = []
        for ci_, c in enumerate(cnts):
            p = c.reshape(-1, 2).astype(np.float32)
            if p.shape[0] < 8 or n_m == 0:
                polys.append(c)
                continue
            ctr = p.mean(0)
            v = p - ctr
            r = np.linalg.norm(v, axis=1) + 1e-6
            ang = np.arctan2(v[:, 1], v[:, 0])
            rmean = float(r.mean())
            rng = np.random.default_rng(seed + 104729 * (ci_ + 1))
            phase = rng.uniform(0.0, 2.0 * np.pi, n_m)
            amp = rng.uniform(0.5, 1.0, n_m)
            # speed: magnitude in [0.5,1]*drift with random sign — avoids accidental statics
            # while keeping `drift` the meaningful scale; drift=0 => static mounds (control).
            speed = drift * rng.uniform(0.5, 1.0, n_m) * rng.choice((-1.0, 1.0), n_m)
            bump = np.zeros_like(ang)
            for k in range(n_m):
                ck = phase[k] + speed[k] * tidx
                dth = np.angle(np.exp(1j * (ang - ck)))
                m_ = np.abs(dth) < wid
                if m_.any():
                    bump[m_] += amp[k] * 0.5 * (1.0 + np.cos(np.pi * dth[m_] / wid))
            newp = ctr + v / r[:, None] * (r + hgt * rmean * bump)[:, None]
            polys.append(np.round(newp).astype(np.int32).reshape(-1, 1, 2))
        return polys
    if mode == "close":
        # MORPHOLOGICAL CLOSE with a component-relative kernel (2026-07-31). Fills the gaps a gait
        # recogniser reads most directly — between the legs, between arm and torso — without
        # touching the outer envelope the way hull does. The kernel is sized as a fraction of the
        # component's OWN equivalent radius sqrt(area/pi), so it is the same proportion of the
        # person at any resolution or camera distance (project rule: no fitted pixel constants).
        # Each component is closed in its own padded bbox buffer, so two people standing close
        # together can never be merged into one blob by this op.
        frac = float(getattr(C, "MASK_CLOSE_KERNEL_FRAC", 0.25))
        polys = []
        for c in cnts:
            x, y, w, h = cv2.boundingRect(c)
            rad = float(np.sqrt(max(1.0, float(cv2.contourArea(c))) / np.pi))
            half = max(1, int(round(frac * rad)))
            k = 2 * half + 1
            pad = k
            sub = np.zeros((h + 2 * pad, w + 2 * pad), np.uint8)
            cv2.drawContours(sub, [c - np.array([[[x - pad, y - pad]]], np.int32)], -1, 1, -1)
            ker = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (k, k))
            sub = cv2.morphologyEx(sub, cv2.MORPH_CLOSE, ker)
            cs2, _ = cv2.findContours(sub, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            if not cs2:
                polys.append(c)
                continue
            for cc in cs2:
                polys.append(cc + np.array([[[x - pad, y - pad]]], np.int32))
        return polys
    return [cv2.approxPolyDP(c, eps_frac * cv2.arcLength(c, True), True) for c in cnts]


def mask_mitigate(hist, cur, eps_frac):
    """Information-REDUCTION on the SHAPE of the emitted silhouette (mask.mkv +
    masked_video.mkv): the per-frame contour dynamics are a soft biometric
    (silhouette/gait shape). Two inclusion-biased ops on the DILATED mask:
      (1) temporal running-MAX over `hist` (the last MASK_TEMPORAL_WIN dilated
          masks) — union, so coverage only grows; smears boundary dynamics;
      (2) contour simplification: cv2.approxPolyDP at eps_frac of each external
          contour's perimeter (also fills interior holes via RETR_EXTERNAL).
    HARD §2 GUARANTEE: the result is OR-ed with BOTH the temporal union `sm` and the
    CURRENT frame's mask, so the emitted mask ⊇ sm ⊇ cur on EVERY frame — mitigation
    can only ADD gray, never reveal, and it is MONOTONE in its input (a larger input
    union can never produce a smaller output).

    ⚠️ WHY `| sm` (added 2026-07-23). The docstring used to claim `| cur` alone made
    coverage "only grow". It did not. `approxPolyDP` at eps = eps_frac·perimeter
    RETRACTS from `sm` at concave corners — and eps SCALES WITH PERIMETER, so when the
    native-res tiled-detection pass enlarged the union (longer contour), the polygon
    cut DEEPER and the emitted mask lost real person pixels the temporal smear had
    covered. Measured as a regression against the pre-tiling baseline (ledger §A.1g).
    `simp` is not a superset of `sm`, so OR-ing `sm` back in restores those bands. The
    area added back is **0.3–0.6 % of the mask per frame** (mean; up to ~5 % on
    high-motion frames — measured; an earlier ~0.09 % estimate was wrong). It is a pure
    ADDITION of temporal-union pixels, so the fix is provably monotone (fixed output ⊇
    old output on 1232/1232 frames) and can only make the silhouette perturbation
    STRONGER, never weaker — §A.6c shows more temporal-union area LOWERS re-ID. Hole-fill
    and running-max are preserved. NOTE: §A.6b's 62.27 % NM was measured on the OLD
    function; re-measure before re-quoting it against the fixed code.
    NOTE: information-reduction mitigation only — NOT a validated silhouette-reID
    defense (unlike the pose anonymizer, this has no lab adversary eval)."""
    sm = np.max(np.stack(hist, 0), 0) if len(hist) > 1 else cur
    cnts, _ = cv2.findContours(sm, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not cnts:
        return sm | cur                                # still ⊇ the temporal union
    simp = np.zeros_like(cur)
    mode = str(getattr(C, "MASK_SHAPE_MODE", "none")).lower() if 'C' in globals() else "none"
    # SHAPE CANONICALISATION (2026-07-25, config.MASK_SHAPE_MODE). The contour simplification
    # above only nibbles at the outline; it leaves limb articulation and the body-width profile
    # intact, which is most of what a silhouette-gait recogniser actually reads (§A.6d: the
    # shipped setting still scores 79.25 % NM = 39.6x chance). The modes replace the PERSON'S OWN
    # SHAPE with a less identifying one at the same position and scale -- see _shape_polys() for
    # each ("hull", "ellipse", "displace", and 2026-07-31's "radiallp" / "close"). All are
    # MASK-ONLY (no pose), which is deliberate: the CASIA-B silhouette adversary consumes bare
    # binary masks, so a pose-driven canonical body could not be evaluated against it at all.
    # §2 is untouched -- the result is still OR-ed with `sm` and `cur` below, so it can only ADD.
    #
    # COMPOSITION (2026-07-31): a "+"-joined mode such as "displace+close" runs the ops in order.
    # Each intermediate stage is rasterised and OR-ed with the temporal union `sm` before its
    # contours are handed to the next op, so every stage is itself a superset of `sm` and the
    # pipeline stays monotone end to end.
    ops = [m for m in mode.split("+") if m]
    if len(ops) > 1:
        work = cnts
        _seed0 = int(getattr(C, "_MASK_DISPLACE_SEED", 0))
        for _i, _op in enumerate(ops):
            # STAGE SALT. Every band-family op derives its per-component RNG from
            # _MASK_DISPLACE_SEED, with no notion of which stage it is. Without a salt,
            # "dirbands+dirbands" draws the SAME axis, layout and amplitudes twice and
            # degenerates into amplitude-doubling of one pattern instead of crossing two;
            # "bands+dirbands" gets correlated layouts. No arm measured so far stacks two
            # band-family stages, so nothing already recorded is affected -- but the first
            # crossed-band config anyone writes would silently underperform.
            C._MASK_DISPLACE_SEED = _seed0 + 7919 * _i
            try:
                polys = _shape_polys(work, _op, eps_frac)
            finally:
                C._MASK_DISPLACE_SEED = _seed0
            if _i == len(ops) - 1:
                break
            _tmp = np.zeros_like(cur)
            cv2.fillPoly(_tmp, polys, 1)
            _tmp |= sm
            work, _ = cv2.findContours(_tmp, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            if not work:
                break
    else:
        polys = _shape_polys(cnts, mode, eps_frac)
    # ---- COMPACT: pull the perturbed outline back toward its own centre ----------------
    # Every shape op above pushes the outline OUTWARD, and the area it adds is the cloud's
    # inpainting bill. This pulls the perturbed polygon back toward its centroid by a factor
    # c <= 1 before it is filled.
    #
    # WHY THIS IS SAFE. The return below OR-s with `sm` and `cur`, so wherever the contraction
    # would cut into the real person the union puts those pixels straight back. S2 therefore
    # still holds by construction, exactly as it does without this step. The contraction can
    # only ever remove ADDED area, never true-person area.
    #
    # WHY IT IS NOT THE SAME AS JUST LOWERING THE AMPLITUDE. Lowering amplitude shrinks the
    # angular VARIATION as well as the size, and A.6k-1 measured that variation - not
    # magnitude - is what buys privacy (raising amplitude made an arm WORSE). A uniform
    # contraction keeps the variation pattern intact and only reduces the overall size, so it
    # should trade area for privacy on better terms. NOT MEASURED YET.
    #
    # Two modes:
    #   MASK_COMPACT_SCALE  - fixed factor, e.g. 0.90.
    #   MASK_COMPACT_TARGET - target emitted-area ratio; c is solved per frame by bisection so
    #                         the frame lands ON the budget instead of averaging there. The
    #                         frontier is convex (A.6k-4/-6, p ~ 1.5-1.8, increasing returns),
    #                         so spending every frame's full allowance beats spending the mean.
    # TARGET wins if both are set. Costs <= _COMPACT_ITERS extra fill+OR per frame.
    _ctgt = getattr(C, "MASK_COMPACT_TARGET", None)
    _csc = float(getattr(C, "MASK_COMPACT_SCALE", 1.0))
    if (_ctgt is not None or _csc < 1.0) and polys:
        def _shrink(ps, c):
            out = []
            for q in ps:
                a = q.reshape(-1, 2).astype(np.float32)
                ctr = a.mean(0)
                out.append(np.round(ctr + (a - ctr) * c).astype(np.int32).reshape(-1, 1, 2))
            return out

        def _emit(ps):
            t = np.zeros_like(cur)
            cv2.fillPoly(t, ps, 1)
            return t | sm | cur

        base_area = float(cur.sum()) or 1.0
        if _ctgt is not None:
            lo, hi = 0.30, 1.0                       # c below 0.30 is pointless: the union dominates
            if float(_emit(polys).sum()) / base_area > float(_ctgt):
                for _ in range(int(getattr(C, "MASK_COMPACT_ITERS", 6))):
                    mid = 0.5 * (lo + hi)
                    if float(_emit(_shrink(polys, mid)).sum()) / base_area > float(_ctgt):
                        hi = mid
                    else:
                        lo = mid
                polys = _shrink(polys, 0.5 * (lo + hi))
        else:
            polys = _shrink(polys, _csc)
    cv2.fillPoly(simp, polys, 1)
    return simp | sm | cur                             # ⊇ sm ⊇ cur: monotone, never retracts
