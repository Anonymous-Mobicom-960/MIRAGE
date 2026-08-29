package com.mirage.npu

import android.content.Context
import android.graphics.Bitmap
import android.graphics.Color
import android.net.Uri
import java.io.File
import java.util.Arrays
import kotlin.math.abs
import kotlin.math.floor
import kotlin.math.hypot
import kotlin.math.max
import kotlin.math.min
import kotlin.math.sqrt

/**
 * BackgroundInpaint (Phase 1) - reconstruct the REAL background hidden behind the person-holes.
 *
 * Input  : input/masked_video.mp4 (real bg with GRAY person holes) + input/mask.mp4 (white=hole),
 *          or the hole is DERIVED from the gray(128) fill when no mask.mp4 is staged.
 * Output : output/background_reconstructed.mp4 (per-frame, holes filled) + .png plate.
 *
 * SELF-DECIDING - a cheap global-motion pre-pass inspects the input and picks the method:
 *
 *   STATIC / JITTER camera  →  "Reveal-and-Fill Static Plate" (the researched optimum):
 *       The people MOVE, so nearly every hidden background pixel is exposed as REAL in *some* frame.
 *       We ALIGN each frame to a middle reference (3-level coarse->fine->native pyramid, sub-pixel,
 *       background-masked - this keeps the plate sharp under the real ~2.5 px/frame handheld jitter),
 *       then per pixel take the temporal TRIMMED-MEAN over the frames where it is real background
 *       (person DILATED out, so codec bleed / edges cannot tint the plate; the trim denoises sensor
 *       noise). The plate is therefore mostly REAL pixels - perfect edges/texture - and a neural
 *       inpainter (LaMa, cropped to the core's bbox for max detail, then grain-matched) fills ONLY the
 *       tiny never-revealed core, ONCE. Per frame we warp the plate back and drop it in with a ring
 *       exposure-match + feathered seam (never per-frame Poisson → no temporal wobble).
 *
 *   DYNAMIC camera (pan/parallax beyond the align range) → union-fill baseline for now; the
 *       motion-compensated method is being researched and will replace this.
 *
 * PRIVACY (§2): the plate only ever samples REAL BACKGROUND (person excluded, DILATED by MASK_DILATE),
 * and the composite fills the FULL contaminated ring (MASK_DILATE+FEATHER) from that clean plate - so
 * no person / gray-fill / codec-bled boundary pixel is ever emitted. All compute on-device.
 */
object BackgroundInpaint {

    fun interface Logger { fun log(line: String) }

    /** logcat tag every Phase-1 line is mirrored under (`adb logcat -s MIRAGE-INPAINT:I`). */
    private const val LOGCAT_TAG = "MIRAGE-INPAINT"

    data class Result(
        val output: File,
        val frames: Int,
        val holePct: Double,           // avg per-frame hole %
        val method: String,
        val corePct: Double = 0.0,     // % of the union hole that was NEVER revealed (neural core)
        val realPct: Double = 100.0,   // % of the union hole reconstructed from REAL pixels
        val motionPx: Double = 0.0,    // measured global camera motion over the clip (working-res px)
        val isStatic: Boolean = true,
        val leakPct: Double = 0.0,     // §2 tripwire: avg % of frame that still looks like un-erased fill
        val prof: String = "",         // sub-stage ms, compact ("decode=12840 align=7010 …") - see [Prof]
        val leakFrames: Int = 0,       // how many frames the §2 auditor actually scored (0 = NOT audited)
    )

    /**
     * §2 audit of a STAGED mask.mp4: are there on-colour FLAT pixels the staged mask failed to cover?
     *
     * Runs the same validated [HoleMask] extractor the derived path uses, purely as an auditor, then
     * asks [HoleMask.auditLeak] how much on-colour flat area lies outside the STAGED mask. `maskInto`
     * must run first - `auditLeak` reuses that frame's working grayscale, which is why the two calls
     * are adjacent and unconditional.
     */
    private fun auditStagedMask(px: IntArray, stagedMask: BooleanArray, t: Int) {
        val a = auditor ?: return
        val b = auditBuf ?: return
        if (auditStride > 1 && t % auditStride != 0) return
        if (a.maskInto(px, fillV, b) < 0) return   // implausible extraction -> don't score this frame
        leakSum += a.auditLeak(px, fillV, stagedMask).toLong()
        leakFrames++
    }

    /**
     * Average per-frame silhouette-leak (see [HoleMask.auditLeak]); warns when above the tripwire.
     *
     * Returns **-1.0 when nothing was audited**. This used to return 0.0, which is why every staged-mask
     * run in the ledger reports "§2 leak audit = 0.000 %": `leakFrames` was only ever incremented on the
     * DERIVED path, so a clip with a mask.mp4 audited zero frames and then reported a perfect score. A
     * null result must never be indistinguishable from a pass on a privacy gate.
     */
    private fun leakReport(np: Int, log: Logger): Double {
        if (leakFrames == 0) {
            log.log("[inpaint] ⚠ §2 leak audit DID NOT RUN (0 frames scored) - this is NOT a pass")
            return -1.0
        }
        val pct = 100.0 * (leakSum.toDouble() / leakFrames) / np
        if (pct > HoleMask.LEAK_WARN_FRAC * 100) {
            log.log("[inpaint] ⚠ §2 LEAK AUDIT: ${"%.3f".format(pct)}% of each frame still looks like solid fill " +
                "outside the mask - inspect background_reconstructed before trusting this run")
        } else {
            log.log("[inpaint] §2 leak audit OK: ${"%.3f".format(pct)}% of frame (same-coloured scene texture only)")
        }
        return pct
    }

    // ---- tunables ----
    private const val GRAY_TOL = 12            // legacy fallback tolerance (used only when auto-detect fails)
    /**
     * The silhouette fill colour for THIS clip, auto-detected in [run] (see [HoleMask.detectFillColor]).
     * 128 is only the legacy fallback: the user's real clip fills with gray **122**, and that offset is
     * an artefact of the export's video-range/colour-matrix handling, so it varies per device.
     * Single-clip scope - one pipeline runs at a time.
     */
    @Volatile private var fillV = 128
    /** Per-clip hole detector (null => no gray-fill clip detected; fall back to the legacy test). */
    @Volatile private var holeMask: HoleMask? = null
    @Volatile private var holeBuf: BooleanArray? = null
    @Volatile private var leakSum = 0L
    @Volatile private var leakFrames = 0
    /** §2 auditor for the STAGED-mask path (the derived path audits inside fillMaskPixels instead).
     *  Non-null only when a mask.mp4 is staged AND a fill colour was detected. */
    @Volatile private var auditor: HoleMask? = null
    @Volatile private var auditBuf: BooleanArray? = null
    @Volatile private var auditStride: Int = 1

    // ---- FIX11: WHERE DID EACH FRAME'S HOLE ACTUALLY COME FROM? ----------------------------------
    // 🔴 The defect this instrumentation exists to make impossible again (FIX11_HOLE.md):
    // `mask.mp4` was staged in H.264 profile "High 4:4:4 Predictive" (x264's lossless mode, what
    // `-crf 0` selects). The S25 cannot decode that profile, so `FrameSource.frameAt` returned null
    // on EVERY frame and `fillMaskPixels` fell silently through to its last-resort whole-frame
    // `|c - fillV| <= GRAY_TOL` colour test - which flags the GRAY PAVING STONES. Dilated by
    // COMPOSITE_DILATE the reported hole went 15.98 % -> 33.66 %, the fill overwrote benches 25 px
    // outside any mask, and NOTHING IN THE RUN LOG SAID SO. The counters below turn that into one
    // loud line. They are read only by [logMaskSource]; no expression that produces a pixel sees them.
    private var mkStaged = 0     // the staged mask.mp4 frame decoded and was used  <- the healthy case
    private var mkRobust = 0     // it did not, and HoleMask's validated silhouette detector stood in
    private var mkLegacy = 0     // neither was available -> the flat colour guess (the 2.1x hole)
    private var mkWarned = false
    private const val MASK_DILATE = 5          // px: exclude person edge + codec bleed from plate samples
    private const val FEATHER = 2              // px: soften the per-frame composite seam
    private const val COMPOSITE_DILATE = MASK_DILATE + FEATHER  // fill the FULL contaminated ring from the plate (no silhouette leak)
    /** Per-frame grain std, in grey levels, estimated from the clip's own real background. Clamped so a
     *  pathological estimate cannot wash the fill out. */
    private const val GRAIN_MIN = 0.4f
    private const val GRAIN_MAX = 6.0f
    private const val CORE_DILATE = 3          // px: LaMa gets a padded context ring; graft back only the true core
    private const val COARSE_DIM = 100         // longest side for the motion pre-pass
    private const val COARSE_SEARCH = 16       // +/- px global-shift search at COARSE_DIM (~±100 working)
    private const val SKIP_ALIGN_PX = 0.8      // <= this global motion => true-static, skip alignment (fast)
    private const val ALIGN_MAX_PX = 90.0      // >  this => DYNAMIC (translation align can't cope) => baseline
    private const val MED_MAX_SAMPLES = 200    // frames aggregated for the temporal trimmed-mean

    // =============================================================================================
    // FIX12 - LOCAL SHARP PLATE  (2026-08-08). The owner: "the masked area gets magnified content …
    // if you look down the benches are also distorted, this is a MUST fixing requirement."
    //
    // WHAT IS ACTUALLY WRONG, measured against ground truth (ledger §C.PHONE-17, frame 106, donor
    // frames warped in by homography, pass-through + blurred-truth controls both passing):
    //   * the fill's CONTENT is nearly right - |diff| 7.1-7.9 grey levels against a 3.42 alignment
    //     floor - but it carries only **0.503-0.622 of the real background's texture energy**. A
    //     smooth patch where bench slats and paving should be is what reads as "magnified".
    //   * ❌ NOT the averaging, and NOT the sample count. On a fixed pixel set the texture is FLAT in
    //     K (+0.004 from K=1 to K=4) while accuracy improves. `robustCenter` and `plateSampleCount`
    //     are both exonerated - which is why the §C.PHONE-16 memory-budget work could not have
    //     helped, and why forcing 140 samples only pushed `minCov` 3→7 and demoted the door to LaMa.
    //   * ✅ it is the ALIGNMENT BASELINE. A homography cannot align across large camera motion:
    //     ring residual vs temporal distance from f106 - ±3-6 frames **1.67-2.00**, ±11 **3.33**,
    //     ±46 **4.67**, ±86 **8.33**. The mosaic stirs all of it into ONE plate (measured mean donor
    //     distance **46.8 frames**), and that mixture equals a blur of σ≈0.5 px - exactly the
    //     measured loss (±0.5 px jitter over 15 samples reproduces texture 0.590 vs the observed
    //     0.622). It lands hardest on NEAR geometry: leftover local shift 0.11 px on the far facade
    //     but **0.56 px at the benches and 1.03 px on the pavement** - the surfaces the owner named.
    //
    // THE FIX, and why it is cheap: composing `canvas→t` with `d→canvas` is as good as estimating
    // `d→t` directly (**−0.04 grey levels** over 9 donors), so NO new alignment work is needed - 
    // `geom` already has everything. Only the CHOICE of samples changes: per hole pixel, walk donors
    // by increasing |d−t| and take the first LOCAL_K that actually see it. Fixed radii cannot work
    // (coverage ±8 → 79 % mean / 35 % worst, ±30 needed for ≥99 %), nearest-first gets both.
    //
    // SIMULATED against truth, with a DEVICE CONTROL that reproduces the shipped output (simulated
    // 0.527 vs measured 0.512 texture): nearest-first K=5 → **texture 0.835, |diff| 2.92** at 100 %
    // coverage, from a mean donor distance of 5.6 frames. Expected: **0.53 → ~0.84**.
    //
    // 🔴 SAFETY. This only ever OVERRIDES `warp[p]` where a local sample set of at least LOCAL_MIN
    // real samples exists. Everywhere else the mosaic plate stands exactly as today, so the door fix
    // and the pushPull floor cannot regress. Set LOCAL_PLATE=false to get byte-identical old output.
    // =============================================================================================
    // ---------------------------------------------------------------------------------------------
    // FILL SHARPNESS - the operator-facing slider (2026-08-17, owner request: "make the background
    // fill less blurry, more real … smallest possible radius fills with longer context window").
    //
    // The three constants below WERE `private const val`. They are the exact knobs FIX12 measured, so
    // rather than invent a new mechanism the slider drives THESE:
    //
    //   * [LOCAL_K]      = how many temporal samples are mixed per hole pixel. This is the blur.
    //                      Every extra sample is a donor at a larger |d − t|, and FIX12 measured that
    //                      mixture as equivalent to a Gaussian of σ ≈ 0.5 px. Fewer samples = the
    //                      SMALLEST possible mixing radius = sharpest, but noisier (less averaging of
    //                      sensor noise) and more sensitive to a single bad donor.
    //   * [LOCAL_RADIUS] = how far in TIME the nearest-first walk may look for those samples - the
    //                      "longer context window". It does NOT blur: nearest-first always spends its
    //                      K on the closest donors that actually see the pixel. A longer window only
    //                      RESCUES pixels that would otherwise fall short of [LOCAL_MIN] and drop
    //                      back to the smooth mosaic/plate value. It costs memory: the donor ring
    //                      holds 2R+1 decoded frames (see [ringRadiusFor], which caps it by the heap).
    //   * [LOCAL_MIN]    = below this many samples the pixel is LEFT ALONE. It tracks K so that a
    //                      sharpness setting of "1 sample" is actually allowed to use 1.
    //
    // 🔴 TWO POSITIONS MATTER, AND THEY ARE NOT THE SAME ONE:
    //   [SHARP_LEGACY]  = 50 reproduces the shipped K=5 / R=8 / min=3 EXACTLY - byte-identical to the
    //                     previous build, so nothing recorded before 2026-08-17 changes, and this is
    //                     the ONLY position FIX12 actually measured (texture 0.835, |diff| 2.92).
    //   [SHARP_DEFAULT] = 100 is where the app now STARTS, on the owner's instruction (2026-08-17,
    //                     "default set the slider at best quality"). K=1: the single NEAREST donor
    //                     that sees the pixel, i.e. a REAL background pixel with no temporal mixing
    //                     at all, over the longest context window (±16 frames) so that strictness
    //                     costs no coverage.
    // ⚠️ NOT MEASURED: everything except 50. K=1 maximises detail and MINIMISES noise averaging, so
    // the honest expectation is sharper but grainier, with more scope for a per-pixel donor switch to
    // show as speckle or temporal flicker. That trade has NOT been rendered or scored on this device.
    // If the fill looks noisy rather than blurry, slide DOWN - 50 is the measured setting.
    // ---------------------------------------------------------------------------------------------
    // 2026-08-17, owner: "make the upper bound even higher for both sliders". K cannot go below 1 - 
    // one sample IS the smallest possible mixing radius - so the range ABOVE 100 buys the other
    // half of the mechanism instead: a LONGER context window, plus the heap share that lets the
    // window actually materialise (at 100 the ±16 request was already being capped to ±9 on the
    // S25, so raising the window alone would have printed a bigger number and changed nothing).
    const val SHARP_MAX = 200
    const val SHARP_LEGACY = 50
    const val SHARP_DEFAULT = 100
    @Volatile private var LOCAL_PLATE = true   // FIX12 master switch; false => previous behaviour
    @Volatile private var LOCAL_RADIUS = 8     // donors searched each side; ring holds 2R+1 frames
    @Volatile private var LOCAL_K = 5          // samples per pixel (K=5 measured 0.835 texture)
    @Volatile private var LOCAL_MIN = 3        // fewer than this => keep the mosaic plate's value
    @Volatile private var sharpness = SHARP_LEGACY
    /** Divisor on `maxMemory` for the donor ring's budget; loosened above sharpness 100. */
    @Volatile private var ringHeapDiv = 4L

    init { setFillSharpness(SHARP_DEFAULT) }

    /**
     * Set the fill-sharpness slider position, 0..100. Single-clip scope, exactly like [fillV] - one
     * pipeline runs at a time - so a plain @Volatile var is the whole mechanism.
     *
     *   0        local plate OFF (pre-FIX12 smooth mosaic / clip-wide plate)
     *   1..100   K = round(9 − 8·s/100) samples, i.e. 9 at s=1 down to 1 at s=100
     *   50       K=5, R=8, min=3  → BIT-IDENTICAL to the shipped build ([SHARP_LEGACY])
     *   >50      the context window also lengthens (R 8 → 16) so the sharper, stricter sample walk
     *            still finds enough donors instead of dropping back to the smooth plate
     *   100      K=1, R=16 - one real nearest pixel, no temporal mixing ([SHARP_DEFAULT])
     */
    @JvmStatic
    fun setFillSharpness(s: Int) {
        val v = s.coerceIn(0, SHARP_MAX)
        sharpness = v
        LOCAL_PLATE = v > 0
        LOCAL_K = if (v <= 0) 5
                  else if (v >= 100) 1
                  else Math.round(9.0 - 8.0 * v / 100.0).toInt().coerceIn(1, 9)
        LOCAL_MIN = min(3, LOCAL_K)
        // Below the default the window stays at the measured 8; above it, it grows - a sharper walk
        // is a STRICTER one (nearer donors only), so it needs more candidates to keep coverage.
        LOCAL_RADIUS = when {
            v <= SHARP_LEGACY -> 8
            v <= 100 -> 8 + Math.round((v - SHARP_LEGACY) * 8.0 / 50.0).toInt()
            else -> 16 + Math.round((v - 100) * 24.0 / 100.0).toInt()      // 16 -> 40 frames each side
        }
        // Above 100 the ring may also claim a larger share of the heap, because on a 1264² clip the
        // ±16 window was ALREADY being cut to ±9 by the quarter-heap cap - without this the extra
        // range would be cosmetic. 4 -> 2 means up to half the heap; [newRing] still catches an OOM.
        ringHeapDiv = if (v <= 100) 4L else max(2L, 4L - (v - 100) / 50L)
    }

    /** The current setting, for the log and the EVALS row - never inferred, always printed. */
    @JvmStatic
    fun describeFillSharpness(): String =
        if (!LOCAL_PLATE) "$sharpness (local sharp plate OFF)"
        else "$sharpness (K=$LOCAL_K, window ±$LOCAL_RADIUS, min=$LOCAL_MIN)" +
            (if (sharpness == SHARP_LEGACY) " = the measured FIX12 setting"
             else " - NOT MEASURED at this position")

    /**
     * The donor ring holds 2R+1 decoded frames plus two bitsets each, so a long context window is a
     * real memory commitment: at 1920x1080 one slot is ~8.6 MB and R=16 is ~290 MB against a 512 MB
     * Dalvik heap. Rather than let the slider OOM the run, the radius is capped here by the same
     * `maxMemory` term [plateSampleCount] already respects, and the applied value is LOGGED - a
     * silently-reduced window would otherwise look exactly like a window that did not help.
     */
    private fun ringRadiusFor(np: Int): Int {
        if (!LOCAL_PLATE) return 0
        val perSlot = ringBytesPerSlot(np)
        // A QUARTER of the heap, not a third: the 208 MB LaMa session is closed only in [run]'s
        // `finally`, so on the static path it is STILL RESIDENT while the composite runs. The ring is
        // the one allocation here that the slider can grow without bound, so it takes the
        // conservative share and [newRing] catches the OOM the estimate can still miss.
        val budget = Runtime.getRuntime().maxMemory() / ringHeapDiv
        val slots = max(3, (budget / max(1L, perSlot)).toInt())
        return min(LOCAL_RADIUS, (slots - 1) / 2).coerceAtLeast(1)
    }

    // ---------------------------------------------------------------------------------------------
    // SEAM MATCH - "pixel matching with the surrounding pixels" (owner request, 2026-08-17).
    //
    // WHAT WAS THERE. [ringGain] matches the fill to the frame with ONE per-channel gain computed
    // over EVERY non-hole pixel in the frame. That is a whole-image exposure match, and it is the
    // wrong statistic for a seam: the fill has to agree with the pavement and the bench slats it
    // actually touches, not with the sky and the facade at the other end of the picture. Any local
    // brightness difference between the hole's neighbourhood and the frame mean lands as a visible
    // step exactly on the hole's outline - the thing that reads as "pasted in".
    //
    // WHAT THIS DOES, in two stages, both driven by the REAL pixels in a band around the hole:
    //   1. the same per-channel gain, but accumulated ONLY over that band, so the exposure match is
    //      local to the region the fill has to blend into;
    //   2. a smooth CORRECTION MEMBRANE: the residual (real − matched fill) sampled on the band is
    //      diffused inward with the same [pushPull] pyramid the uncovered-pixel path already uses,
    //      and added across the hole. This is membrane interpolation, not Poisson: it is a purely
    //      LOW-FREQUENCY additive field, so it cannot move an edge or invent structure - it can only
    //      make the fill's slow brightness/colour drift agree with its surroundings. The file's
    //      header rules out per-frame POISSON because solving gradients per frame wobbles
    //      temporally; a boundary-driven smooth offset has no such term.
    //
    // 🔴 SEAM 0 IS THE OLD BEHAVIOUR EXACTLY - it returns [ringGain] untouched, so nothing recorded
    // before 2026-08-17 changes. ⚠️ NOT MEASURED on a device: no seam metric has been rendered.
    // ---------------------------------------------------------------------------------------------
    const val SEAM_MAX = 200
    const val SEAM_DEFAULT = 100
    @Volatile private var seamMatch = SEAM_DEFAULT
    /** px of REAL background around the hole that the match is fitted to. */
    @Volatile private var seamBandPx = 24
    /** How much of the membrane is added, 0..1. Saturates at slider 100. */
    @Volatile private var seamStrength = 1f
    private val GAIN_ONE = floatArrayOf(1f, 1f, 1f)

    /**
     * Set the seam-match slider, 0..[SEAM_MAX].
     *
     *   0        the previous whole-frame exposure gain, nothing else - byte-identical to 0.16
     *   1..100   the local gain, plus the correction membrane faded in to full strength, fitted to a
     *            24 px band of real background around the hole
     *   >100     strength is already saturated, so the extra range TIGHTENS the band instead - 24 px
     *            down to 8 px at 200. A narrower band hugs the boundary, so the correction is fitted
     *            to the pixels the fill literally touches rather than to a neighbourhood average.
     *            ⚠️ It also leaves fewer samples to fit from; below 50 usable samples
     *            [matchSurroundings] falls back to the whole-frame gain and says nothing changed.
     */
    @JvmStatic
    fun setSeamMatch(s: Int) {
        val v = s.coerceIn(0, SEAM_MAX)
        seamMatch = v
        seamStrength = (min(v, 100) / 100f)
        seamBandPx = if (v <= 100) 24 else max(8, 24 - Math.round((v - 100) * 16.0 / 100.0).toInt())
    }

    @JvmStatic
    fun describeSeamMatch(): String =
        if (seamMatch <= 0) "0 (whole-frame gain only - the previous behaviour)"
        else "$seamMatch (local gain + membrane x${"%.2f".format(seamStrength)} over a ${seamBandPx}px " +
            "surround) - NOT MEASURED"

    /**
     * Match the fill to the pixels immediately AROUND the hole. See the block comment above.
     *
     * @param band/[res]/[unknown] caller-owned np-sized scratch (contents on entry are irrelevant).
     * @return the per-channel gain the CALLER must still apply to `warp`. When this function did the
     *         local match it has already folded the gain into `warp` itself and returns (1,1,1), so
     *         the caller's blend expression is unchanged either way.
     */
    private fun matchSurroundings(
        mvpx: IntArray, warp: IntArray, hole: BooleanArray, ablur: FloatArray,
        w: Int, h: Int, np: Int, band: BooleanArray?, res: IntArray?, unknown: BooleanArray?,
    ): FloatArray {
        if (seamMatch <= 0 || band == null || res == null || unknown == null)
            return ringGain(mvpx, warp, hole, np)
        // 1. the ring of REAL background just outside the (already COMPOSITE_DILATE-grown) hole
        for (p in 0 until np) band[p] = hole[p]
        dilateInPlace(band, w, h, seamBandPx)
        for (p in 0 until np) if (hole[p]) band[p] = false
        // 2. exposure gain from THOSE pixels only. `warp` is valid on every 4th pixel of the frame
        //    (the composite's warp predicate is `ablur > 0 || p % 4 == 0`), so the band is sampled at
        //    that same stride - no extra resampling, and the identical operand set ringGain uses.
        var fr = 0L; var fg = 0L; var fb = 0L; var pr = 0L; var pg = 0L; var pb = 0L; var cnt = 0L
        var p = 0
        while (p < np) {
            if (band[p]) {
                val f = mvpx[p]; val q = warp[p]
                fr += (f ushr 16) and 0xFF; fg += (f ushr 8) and 0xFF; fb += f and 0xFF
                pr += (q ushr 16) and 0xFF; pg += (q ushr 8) and 0xFF; pb += q and 0xFF
                cnt++
            }
            p += 4
        }
        // Too little real background around the hole to fit anything local -> the whole-frame gain is
        // strictly better than a gain fitted to a handful of pixels. Falling back is not a failure.
        if (cnt < 50) return ringGain(mvpx, warp, hole, np)
        satRingN++
        var pinned = false
        fun g(a: Long, b: Long): Float {
            if (b <= 1L) return 1f
            val raw = (a.toDouble() / b.toDouble()).toFloat()
            if (raw < 0.75f || raw > 1.35f) {
                pinned = true
                if (abs(raw - 1f) > abs(satRingWorst - 1.0)) satRingWorst = raw.toDouble()
            }
            return raw.coerceIn(0.75f, 1.35f)
        }
        val gr = g(fr, pr); val gg = g(fg, pg); val gb2 = g(fb, pb)
        if (pinned) satRingPinned++
        // 3. the residual membrane. Seeded ONLY on the band (real pixels, so the residual is real),
        //    diffused inward by the push-pull pyramid, then added across the touched region.
        //    128 is the zero of the encoding; the clamp bounds a correction at +/-127 levels, which
        //    is far beyond anything the gain above can leave behind.
        for (q in 0 until np) {
            if (band[q] && (q and 3) == 0) {
                val f = mvpx[q]; val v = warp[q]
                val dr = ((f ushr 16) and 0xFF) - (((v ushr 16) and 0xFF) * gr).toInt()
                val dg = ((f ushr 8) and 0xFF) - (((v ushr 8) and 0xFF) * gg).toInt()
                val db = (f and 0xFF) - ((v and 0xFF) * gb2).toInt()
                res[q] = Color.rgb((128 + dr).coerceIn(0, 255), (128 + dg).coerceIn(0, 255),
                    (128 + db).coerceIn(0, 255))
                unknown[q] = false
            } else unknown[q] = true
        }
        pushPull(res, unknown, w, h)
        val s = seamStrength
        Par.range(np) { p0, p1 ->
            for (q in p0 until p1) {
                if (ablur[q] <= 0f) continue
                val v = warp[q]; val m = res[q]
                val r = (((v ushr 16) and 0xFF) * gr + s * (((m ushr 16) and 0xFF) - 128)).toInt()
                val gch = (((v ushr 8) and 0xFF) * gg + s * (((m ushr 8) and 0xFF) - 128)).toInt()
                val b = ((v and 0xFF) * gb2 + s * ((m and 0xFF) - 128)).toInt()
                warp[q] = Color.rgb(r.coerceIn(0, 255), gch.coerceIn(0, 255), b.coerceIn(0, 255))
            }
        }
        return GAIN_ONE
    }

    /** Per-composite scratch for [matchSurroundings]; all null (no allocation) when the slider is 0. */
    private fun seamScratch(np: Int): Triple<BooleanArray?, IntArray?, BooleanArray?> =
        if (seamMatch > 0) Triple(BooleanArray(np), IntArray(np), BooleanArray(np))
        else Triple(null, null, null)

    /** px IntArray + the two row-independent bitsets a donor slot carries. */
    private fun ringBytesPerSlot(np: Int): Long = np.toLong() * 4 + 2L * ((np + 63) ushr 6) * 8

    /**
     * Allocate the donor ring, HALVING the window rather than dying if the heap cannot take it.
     *
     * [ringRadiusFor]'s arithmetic is an estimate against `maxMemory`, and it cannot see what else is
     * resident (the LaMa session, the decoder's own buffers, a fragmented heap). An OOM here would
     * kill a run the operator has already waited minutes for, to buy a longer context window that is
     * a refinement, not a requirement - so it degrades instead, and SAYS SO.
     */
    private fun newRing(np: Int, rad0: Int, log: Logger): LocalRing {
        var rad = rad0
        while (true) {
            try {
                return LocalRing(2 * rad + 1, np)
            } catch (e: OutOfMemoryError) {
                if (rad <= 1) throw e
                rad /= 2
                log.log("[inpaint] ⚠ donor ring did not fit - context window reduced to ±$rad " +
                    "(${ringBytesPerSlot(np) * (2 * rad + 1) / 1_000_000} MB). The fill is correspondingly " +
                    "less able to serve sharp samples; this is a memory limit, not a quality choice.")
            }
        }
    }

    // 🔴 2026-08-08 - THIS IS THE KNOB BEHIND THE "SAME SPOT ALWAYS" ARTEFACT, and the owner has
    // granted a much larger budget. Read `plateSampleCount` before changing it: the cap is
    // min(MEM_BUDGET, maxMemory * HEAP_FRACTION), and on the S25 `dalvik.vm.heapsize` is 512 MB, so
    // the maxMemory term is what actually bound us - NOT this constant. Raising this alone does
    // nothing, which is why the log line below now prints every term.
    //
    // WHY IT MATTERS: the mosaic canvas is 1825x1634 = 1.87x the frame area, so every sample costs
    // 1.87x more and the fixed per-pixel overhead scales too. The door fix therefore HALVED the
    // plate's temporal evidence - the code's own comment records ~52 samples for a frame-sized
    // stack, and the device reported 25 for the mosaic. With a fixed 25-of-140 subset the coverage
    // gaps land at FIXED SCENE LOCATIONS, so the same patch is LaMa-invented every time the mask
    // passes over it, in both directions - exactly what the owner reported at ~2.8 s and again on
    // the return pass. Measured: no scene point is EVER permanently occluded (the only never-revealed
    // pixels are 9 columns of canvas edge, covered by zero masks), so the 7.2 % neural core is not
    // geometric necessity - it is a sampling shortfall.
    private const val MEM_BUDGET = 3_500_000_000L
    // Was 3/5, hard-coded inside plateSampleCount. Named and raised so the term is visible and
    // tunable. Kept below 1.0 deliberately: the aggregation also holds np-sized scratch (see
    // FIXED_PER_PX) and the decoder's own buffers, and an OOM here kills the whole run.
    private const val HEAP_FRACTION_NUM = 4L
    private const val HEAP_FRACTION_DEN = 5L
    private const val FIXED_PER_PX = 28L       // co-resident np-sized buffers, for the memory cap
    // alignment pyramid: (downscale longest-side target, +/- refine radius). Coarse->fine; 9999 => native.
    private val ALIGN_PYRAMID = listOf(100 to 16, 300 to 4, 9999 to 3)
    // --- DYNAMIC (moving camera) windowing ---
    private const val R_MAX_BASE = 60.0        // max per-frame deviation from a window ref (px @640 basis; scales by w/640)
    private const val WINDOW_MIN = 16          // frames
    private const val WINDOW_MAX = 96          // frames
    private const val MAX_WINDOWS = 8          // cap K -> bounds retained memory + per-window LaMa calls + alignment
    // Windows use a LIGHTER pyramid (no native level): a ~2px-precise plate is plenty for a moving-camera
    // window, and dropping native slashes per-window align cost (~10x) AND retained aligner memory (~5np->~0.05np).
    private val WINDOW_PYRAMID = listOf(100 to 16, 300 to 4)

    // =============================================================================================
    // SATURATION INSTRUMENTATION (FIX9, 2026-08-08)
    //
    // 🔴 EVERY `coerceIn` ON THIS PATH IS A SILENT FAILURE MODE. A clamp cannot tell a value it
    // merely trimmed from one it PINNED, so a saturated statistic looks exactly like a measured one
    // - in the output and in the log. `GRAIN_MAX` alone has now cost three sessions: §C.PHONE-13c
    // found the estimator reading 16.50 and pinning at 6.0, FIX6 §3.5 found the applied fix STILL
    // pinning at 6.0, and FIX9 found it pinning on 4/12 pairs of `c3` on the STATIC path. None of
    // those pins printed anything. These counters make every pin print one line, so the next
    // occurrence is a row in EVALS.md rather than another three-agent investigation.
    //
    // THREADING: every counter below is incremented from the PHASE'S OWN THREAD - `aggregatePlate`'s
    // per-frame exposure gain, `ringGain`, `addGrain` and `Aligner.shiftOf` all run there, never
    // inside a `Par` band - so plain vars are safe and no atomics are needed. They count only; not
    // one of them is read by any expression that produces a pixel.
    // =============================================================================================
    /**
     * 🔴 FIX9 - DEFAULT OFF, DELIBERATELY. The STATIC composite reads the plate through
     * [bilinearRGB], which CLAMPS its coordinates (`coerceIn(0, w-1)`), so a hole pixel whose source
     * `(x-dx, y-dy)` falls off the frame-sized plate is not skipped - it is filled with the plate's
     * edge pixel, replicated. That is precisely the "moving dark smear at the pan-leading edge"
     * `compositeWindowed` was given an explicit guard against, and `compositeFrames` never was.
     *
     * MEASURED on the real RUN3 inputs (`fix9_ref/F9_EDGE.json`, N = 74 frames per clip):
     * `c2` (STATIC, devC 59.6 px, max |shift| 104.3 px) has off-plate hole pixels on **12/74
     * frames**, and on its worst frame (t=136, dx −103.2) **56.5 % of a 164 312 px hole** is
     * edge-replicated. `c3` on **36/74 frames**, worst 3.14 %.
     *
     * Turning this true routes those pixels to `pushPull` instead, exactly as the other two
     * composites do. It is left OFF because it is PIXEL-AFFECTING on the shipped static path and
     * this session could not render a before/after for the owner to approve (the standing rule).
     * Flipping it changes NOTHING on a frame with no off-plate pixels - the guard only fires where
     * the old code was reading a clamped coordinate - so 62 of `c2`'s 74 frames stay bit-identical.
     * The fire-rate line below prints either way, so the defect is visible without the fix.
     */
    private const val STATIC_EDGE_GUARD = false

    private var satGainN = 0; private var satGainPinned = 0; private var satGainWorst = 1.0
    private var satRingN = 0; private var satRingPinned = 0; private var satRingWorst = 1.0
    private var satAlignN = 0; private var satAlignPinned = 0
    /** Set by [bestShiftSub] when its INTEGER argmin lands on the boundary of its search box, i.e.
     *  the true shift is at least as large as the box and the returned value is a floor, not a
     *  measurement. Read (and cleared) only by [Aligner.shiftOf] - [computeTrajectory] pins by
     *  design on a fast pan and must not be counted. */
    private var lastShiftPinned = false

    private fun resetSaturation() {
        satGainN = 0; satGainPinned = 0; satGainWorst = 1.0
        satRingN = 0; satRingPinned = 0; satRingWorst = 1.0
        satAlignN = 0; satAlignPinned = 0; lastShiftPinned = false
        mkStaged = 0; mkRobust = 0; mkLegacy = 0; mkWarned = false
    }

    /** Logger for the mask-source tripwire - [fillMaskPixels] is on the hot path and takes no Logger. */
    @Volatile private var maskLog: Logger? = null

    /** Fired the FIRST time a staged mask.mp4 frame fails to decode. Names the cause, because this
     *  failure has exactly one plausible one and it cost a delivered video (FIX11_HOLE.md). */
    private fun mkFallbackWarn(t: Int) {
        if (mkWarned) return
        mkWarned = true
        maskLog?.log("[inpaint] ⚠⚠ STAGED mask.mp4 FAILED TO DECODE at frame $t - the hole is now a " +
            "GUESS from the picture, not the Tier-1 mask. The known cause is an encoder profile this " +
            "device cannot open: a mask written with `x264 -crf 0` is profile 'High 4:4:4 Predictive' " +
            "and MediaCodec rejects it. Re-stage the mask as `-profile:v high` (see FIX11_HOLE.md).")
    }

    /** One line per run stating where every frame's hole actually came from. Silence is never a pass. */
    private fun logMaskSource(log: Logger) {
        val n = mkStaged + mkRobust + mkLegacy
        if (n == 0) return
        // n counts mask READS, not frames: the plate pass and the composite pass each read every
        // frame's mask, so n is a small multiple of the clip length. The ratio is the signal.
        if (mkStaged == n) {
            log.log("[inpaint] hole source: the STAGED mask.mp4 on all $n mask reads")
            return
        }
        log.log("[inpaint] ⚠ hole source (of $n mask reads): staged mask.mp4 $mkStaged · " +
            "HoleMask-derived $mkRobust · legacy flat-colour guess $mkLegacy")
        if (mkLegacy > 0) log.log("[inpaint] ⚠ the legacy flat-colour guess flags same-coloured SCENE " +
            "texture (paving stones) as person, and COMPOSITE_DILATE then grows each speckle into a " +
            "15x15 block - measured 15.98% -> 33.66% of frame on c1. Treat this run's fill as suspect.")
    }

    /** One line per clamp that actually pinned; silence means nothing saturated. */
    private fun logSaturation(log: Logger) {
        if (satGainPinned > 0) log.log("[inpaint] ⚠ SATURATED: plate exposure gain hit the " +
            "0.85..1.18 clamp on $satGainPinned/$satGainN sampled frames (worst raw " +
            "${"%.3f".format(satGainWorst)}) - those samples are exposure-normalised only as far as " +
            "the clamp allows, so the plate carries their residual brightness error")
        if (satRingPinned > 0) log.log("[inpaint] ⚠ SATURATED: ringGain hit the 0.75..1.35 clamp on " +
            "$satRingPinned/$satRingN composited frames (worst raw ${"%.3f".format(satRingWorst)}) - " +
            "the fill's exposure does not match that frame's background; expect a visible seam")
        if (satAlignPinned > 0) log.log("[inpaint] ⚠ SATURATED: the alignment pyramid's search box " +
            "was hit on $satAlignPinned/$satAlignN frames - the true shift is LARGER than the " +
            "pyramid can express (FIX1 §4.3: +/-229.5 px at 1264), so those frames are aligned to a " +
            "floor, not to their real shift")
        if (satGainPinned == 0 && satRingPinned == 0 && satAlignPinned == 0)
            log.log("[inpaint] clamp audit: no exposure/ring/alignment clamp saturated on this clip")
    }

    fun run(context: Context, lamaProvider: () -> LamaInpainter?, maxDim: Int, uiLog: Logger, s2Stride: Int = 1): Result {
        auditStride = s2Stride
        // 🔴 FIX11 - MIRROR EVERY [inpaint] LINE TO LOGCAT.
        // Phase 1's log has only ever gone to a TextView in the app, so FIX1/FIX6/FIX9 each recorded
        // "the [inpaint] lines were not captured" and had to reason about the run instead of reading
        // it. Every tripwire this file prints - the pushPull fire rate, the clamp audit, and now the
        // hole source - is unreadable from a scripted run without this. It is write-only: nothing
        // reads it back, so it cannot change a pixel.
        val log = Logger { line -> uiLog.log(line); android.util.Log.i(LOGCAT_TAG, line) }
        // Materialise the 208 MB LaMa session at most once, and only if fillCore actually reaches for it.
        var lamaBuilt = false
        var lamaInst: LamaInpainter? = null
        val getLama: () -> LamaInpainter? = {
            if (!lamaBuilt) { lamaBuilt = true; lamaInst = lamaProvider() }
            lamaInst
        }
        // The METHOD string must not force the load, so it is decided by whether the file is staged.
        val lamaAvailable = NpuFactory.hasLama(context)
        try {
            return runInner(context, getLama, lamaAvailable, maxDim, log)
        } finally {
            lamaInst?.let { runCatching { it.close() } }
        }
    }

    private fun runInner(
        context: Context, getLama: () -> LamaInpainter?, lamaAvailable: Boolean, maxDim: Int, log: Logger,
    ): Result {
        // NOTE: Prof.reset() is deliberately NOT called here - PhaseRunner.phase1Inpaint resets before
        // it builds the LaMa session, so the 208 MB model load lands in the same accounting window.
        // A caller that never resets simply gets an empty report (Prof is inert until started).
        val maskedF = MiragePaths.maskedVideo
        val maskF = MiragePaths.maskVideo
        require(maskedF.exists()) { "missing ${maskedF.name} in input/ (Tier-1 masked video)" }
        MiragePaths.ensureDirs()
        val deriveMask = !maskF.exists()

        val sources = mutableListOf<FrameSource>()   // close every opened source even if the 2nd ctor throws
        try {
            val masked = FrameSource(context, Uri.fromFile(maskedF)).also { sources.add(it) }
            val mask = if (deriveMask) null else FrameSource(context, Uri.fromFile(maskF)).also { sources.add(it) }
            val n = if (mask != null) min(masked.count, mask.count) else masked.count
            require(n >= 1) { "no frames in masked_video" }
            val f0 = masked.frameAt(0, maxDim) ?: throw IllegalStateException("cannot decode masked_video")
            val w = f0.width; val h = f0.height; val np = w * h

            // ---- STEP 0: when no mask.mp4 was exported, derive the holes ROBUSTLY -------------------
            // The old code assumed a gray(128) fill and a flat ±12 box. On the user's clip that flagged
            // 24-32 % of every frame (the gray PAVING STONES) instead of the true 10-17 % silhouette.
            // HoleMask auto-detects the fill colour and adds a flatness cue the pavement cannot fake.
            fillV = 128; holeMask = null; holeBuf = null; leakSum = 0L; leakFrames = 0
            auditor = null; auditBuf = null; resetSaturation(); maskLog = log
            // Detect the fill colour on BOTH paths. It used to be detected only when deriving the mask,
            // which left the staged-mask.mp4 path with no fill colour and therefore no §2 audit at all.
            val v = HoleMask.detectFillColor(masked, n, maxDim, log::log)
            if (v in 0..255) fillV = v
            // Publish it so Phase 1b does not repeat the scan (5 decodes + 5 full-frame flatness
            // passes). Measured: 1b's untimed remainder was 43 % of that phase, and this was the
            // largest identifiable piece of it.
            runCatching { MiragePaths.fillColorFile.writeText(fillV.toString()) }
            // 🔴 FIX11 - BUILD THE ROBUST DETECTOR ON BOTH PATHS, not only when mask.mp4 is absent.
            // It used to exist only under `deriveMask`, so when a STAGED mask.mp4 failed to decode
            // `holeMask` was null and fillMaskPixels skipped straight past the validated silhouette
            // detector to the flat colour guess this file's own comment above says flags 24-32 % of
            // the frame. On a healthy clip this allocation is INERT - fillMaskPixels' staged branch
            // returns before it is ever consulted - so it costs buffers, never pixels.
            if (v in 0..255) { holeMask = HoleMask(w, h); holeBuf = BooleanArray(np) }
            if (deriveMask) {
                if (v in 0..255) {
                    log.log("[inpaint] no mask.mp4 - deriving person holes from the gray($v) fill (flatness-gated)")
                } else {
                    log.log("[inpaint] no mask.mp4 - deriving person holes from the legacy gray(128) test")
                }
            } else if (v in 0..255 && auditStride > 0) {
                // 🔴 §2 AUDIT ON THE STAGED-MASK PATH. Until now `leakFrames` was only ever incremented
                // inside fillMaskPixels' DERIVED branch, so a run with a staged mask.mp4 audited nothing
                // and leakReport's `if (leakFrames == 0) return 0.0` reported a perfect 0.000 % anyway.
                // Every "§2 leak audit = 0.000 %" recorded for a staged-mask arm is therefore a NULL
                // measurement, not a clean one. The auditor below asks the real question - are there
                // on-colour FLAT pixels that the staged mask failed to cover - using the same validated
                // HoleMask logic the derived path uses.
                auditor = HoleMask(w, h)
                auditBuf = BooleanArray(np)
                log.log("[inpaint] §2 auditor armed against the staged mask (fill=gray($fillV))")
            } else if (auditStride <= 0) {
                log.log("[inpaint] §2 audit OFF (verification tool; enable it in the app to re-arm)")
            } else {
                log.log("[inpaint] ⚠ fill colour not detected - §2 audit CANNOT run on this clip")
            }

            // ---- STEP 1: decide STATIC/JITTER vs DYNAMIC (chained global-motion trajectory) ----
            // TRAJECTORY covers only the SEARCH; its 16 frame decodes are timed into DECODE inside
            // computeTrajectory. Keeping the outer wrapper too would double-count them.
            val traj = computeTrajectory(masked, n, w, h, maxDim)
            val out = MiragePaths.backgroundReconstructed

            if (traj.needsWindowing) {
                // ---- DYNAMIC (moving camera) ----
                log.log("[inpaint] camera devC ~${"%.1f".format(traj.devC)}px over clip -> DYNAMIC")
                if (MOSAIC_ENABLED) {
                    // FIX6: homography registration + a pan-sized mosaic canvas. Any failure - 
                    // too few frames converging, a canvas over the area cap, OOM on the sample
                    // stack - falls through to the windowed path below rather than failing the run.
                    val mos = runCatching {
                        buildMosaicPlate(masked, mask, n, w, h, np, maxDim, traj, getLama, log)
                    }.getOrElse {
                        log.log("[inpaint] ⚠ mosaic path failed (${it.javaClass.simpleName}: " +
                            "${it.message?.take(80)}) -> falling back to the windowed path")
                        null
                    }
                    if (mos != null) {
                        val (mHoleSum, mWritten) = compositeMosaic(masked, mask, n, w, h, np, maxDim, mos, out, log)
                        val mAvgHolePct = if (mWritten > 0) 100.0 * mHoleSum / (mWritten.toDouble() * np) else 0.0
                        val mCorePx = mos.core.count { it }.toLong()
                        val mUnionPx = mos.union.count { it }.toLong()
                        val mCorePct = if (mUnionPx > 0) 100.0 * mCorePx / mUnionPx else 0.0
                        VideoIo.savePng(Bitmap.createBitmap(mos.plate, mos.mw, mos.mh, Bitmap.Config.ARGB_8888),
                            File(out.parentFile, "background_reconstructed.png"))
                        val mMethod = "homography mosaic (${mos.mw}x${mos.mh}, ${mos.samples} samples, " +
                            "${if (lamaAvailable) "LaMa" else "push-pull"} core)"
                        logMaskSource(log); logSaturation(log)
                        log.log("[inpaint] done -> ${out.name} ($mWritten frames; " +
                            "${"%.1f".format(100.0 - mCorePct)}% real, ${"%.1f".format(mCorePct)}% core; $mMethod)")
                        log.log("[prof] Phase 1 sub-stage breakdown:\n${Prof.report(mWritten)}")
                        return Result(out, mWritten, mAvgHolePct, mMethod, mCorePct, 100.0 - mCorePct,
                            traj.devC, false, leakReport(np, log), Prof.compact(), leakFrames)
                    }
                }
                // ---- fallback: the previous windowed reveal-and-fill ----
                log.log("[inpaint] DYNAMIC -> windowed reveal-and-fill (translation-aligned)")
                val windows = buildWindowedPlates(masked, mask, n, w, h, np, maxDim, traj, getLama, log)
                val (holeSum, written) = compositeWindowed(masked, mask, n, w, h, np, maxDim, windows, out, log)
                val avgHolePct = if (written > 0) 100.0 * holeSum / (written.toDouble() * np) else 0.0
                var cSum = 0L; var uSum = 0L
                for (win in windows) { cSum += win.core.count { it }.toLong(); uSum += win.union.count { it }.toLong() }
                val corePct = if (uSum > 0) 100.0 * cSum / uSum else 0.0
                windows.getOrNull(windows.size / 2)?.let {
                    VideoIo.savePng(Bitmap.createBitmap(it.plate, w, h, Bitmap.Config.ARGB_8888), File(out.parentFile, "background_reconstructed.png"))
                }
                val method = "windowed reveal-and-fill (${windows.size} windows, ${if (lamaAvailable) "LaMa" else "push-pull"} cores)"
                logMaskSource(log); logSaturation(log)
                log.log("[inpaint] done -> ${out.name} ($written frames; ${windows.size} windows; ${"%.1f".format(100.0 - corePct)}% real, ${"%.1f".format(corePct)}% core; $method)")
                log.log("[prof] Phase 1 sub-stage breakdown:\n${Prof.report(written)}")
                return Result(out, written, avgHolePct, method, corePct, 100.0 - corePct, traj.devC, false, leakReport(np, log), Prof.compact(), leakFrames)
            }

            // ---- STATIC / JITTER: single aligned plate (the verified path) ----
            val doAlign = traj.devC > SKIP_ALIGN_PX
            log.log("[inpaint] camera devC ~${"%.1f".format(traj.devC)}px over clip -> ${if (doAlign) "JITTER (aligned)" else "STATIC"}")
            val refT = n / 2
            val refFrame = masked.frameAt(refT, maxDim)
            val aligner = if (doAlign && refFrame != null) Aligner.build(refFrame, w, h) else null
            val shiftCache = if (aligner != null) HashMap<Int, FloatArray>() else null
            val refMeanLuma = referenceMeanLuma(refFrame, mask, refT, maxDim, w, h, np)

            val plate = IntArray(np); val core = BooleanArray(np); val union = BooleanArray(np)
            val method = "reveal-and-fill (${if (aligner != null) "aligned " else ""}trimmed-mean + ${if (lamaAvailable) "LaMa" else "push-pull"} core)"
            // 🔴 FIX9 - the grain sigma is measured HERE, not inside the composite. Two reasons, both
            // measured (see FIX9_CLAMPS.md):
            //  * it needs the HOMOGRAPHY, and this is the cheapest slot in the phase to hold one:
            //    the 275 MB sample stack does not exist yet and the 208 MB LaMa session is not built
            //    yet, so HAligner's ~50 MB lands in the widest headroom Phase 1 ever has;
            //  * it takes the MEDIAN of several pairs, which the composite's single inline pair
            //    could not.
            val grainSigma = staticGrainSigma(masked, n, w, h, np, maxDim, traj, log)
            buildStaticPlate(masked, mask, n, w, h, np, maxDim, aligner, shiftCache, refMeanLuma, plate, core, union, log)

            val corePx = core.count { it }
            val unionPx = union.count { it }
            val corePct = if (unionPx > 0) 100.0 * corePx / unionPx else 0.0
            val realPct = 100.0 - corePct
            if (corePx > 0) fillCore(plate, core, w, h, np, getLama, log)
            else log.log("[inpaint] every hole pixel was revealed by motion - 100% real, no neural fill needed")

            val (holeSum, written) = compositeFrames(masked, mask, n, w, h, np, maxDim, aligner, shiftCache, plate, refMeanLuma, grainSigma, out, log)
            val avgHolePct = if (written > 0) 100.0 * holeSum / (written.toDouble() * np) else 0.0
            VideoIo.savePng(Bitmap.createBitmap(plate, w, h, Bitmap.Config.ARGB_8888), File(out.parentFile, "background_reconstructed.png"))
            logMaskSource(log); logSaturation(log)
            log.log("[inpaint] done -> ${out.name} ($written frames; ${"%.1f".format(realPct)}% of hole from REAL pixels, ${"%.1f".format(corePct)}% neural core; $method)")
            log.log("[prof] Phase 1 sub-stage breakdown:\n${Prof.report(written)}")
            return Result(out, written, avgHolePct, method, corePct, realPct, traj.devC, true, leakReport(np, log), Prof.compact(), leakFrames)
        } finally {
            sources.forEach { runCatching { it.close() } }
        }
    }

    // =============================================================================================
    // STEP 1 - camera-motion detection (global translation over BACKGROUND-only pixels)
    // =============================================================================================

    private class Traj(
        val devC: Double,            // worst deviation of the chained trajectory from its bbox center (working px)
        val needsWindowing: Boolean, // devC > R_MAX(w) -> a single aligned plate can't cover the motion
        /**
         * The chained trajectory ITSELF: the frame indices that were actually decoded, and their
         * cumulative shift from the first of them, in working px.
         *
         * ADDITIVE (2026-08-08, FIX6). [devC] and [needsWindowing] are computed by exactly the same
         * arithmetic as before - these are the very numbers the existing loop already produced,
         * merely returned instead of discarded. Nothing on the STATIC path reads them. The mosaic
         * path needs them because the SAD pyramid saturates at ±229.5 px (FIX1 §4.3) while a chained
         * consecutive-pair sum is unbounded, and on a 456 px pan the init has to be unbounded.
         */
        val idxs: IntArray = IntArray(0),
        val tx: DoubleArray = DoubleArray(0),
        val ty: DoubleArray = DoubleArray(0),
    )

    /** CHAINED global-motion trajectory: consecutive-pair shifts SUMMED, so the measurable range is
     *  UNBOUNDED (a frame-0-anchored detector saturates + aliases on a smooth pan). devC = worst deviation
     *  from the trajectory's bbox center; > R_MAX(w) means one aligned plate can't cover it -> windowing. */
    private fun computeTrajectory(masked: FrameSource, n: Int, w: Int, h: Int, maxDim: Int): Traj {
        if (n < 3) return Traj(0.0, false)
        val cnt = min(n, 16)
        val idxs = IntArray(cnt) { it * (n - 1) / (cnt - 1) }
        val (dw, dh) = dimsFor(w, h, COARSE_DIM)
        val grays = ArrayList<IntArray>(); val valids = ArrayList<BooleanArray>()
        val used = ArrayList<Int>(cnt)   // which of idxs actually decoded (see Traj.idxs)
        // Timed as DECODE, not TRAJECTORY: these are 16 SCATTERED seeks across the clip, the worst case
        // for MediaMetadataRetriever (each may decode forward from the nearest keyframe), whereas the
        // composite pass reads near-sequentially. Separating them tells us whether the trajectory
        // bucket is seek cost or search cost - the profile could not distinguish them before.
        for (t in idxs) {
            val mv = Prof.time(Prof.DECODE) { masked.frameAt(t, maxDim) } ?: continue
            val (g, v) = downGrayValid(mv, dw, dh); grays.add(g); valids.add(v); used.add(t)
        }
        val m = grays.size
        if (m < 2) return Traj(0.0, false)
        val up = w.toDouble() / dw
        val tx = DoubleArray(m); val ty = DoubleArray(m)
        Prof.time(Prof.TRAJECTORY) {
            for (k in 1 until m) {
                val sh = bestShiftSub(grays[k - 1], valids[k - 1], grays[k], valids[k], dw, dh, 0, 0, COARSE_SEARCH)
                tx[k] = tx[k - 1] + sh[0] * up; ty[k] = ty[k - 1] + sh[1] * up
            }
        }
        var mnx = tx[0]; var mxx = tx[0]; var mny = ty[0]; var mxy = ty[0]
        for (k in 1 until m) { mnx = min(mnx, tx[k]); mxx = max(mxx, tx[k]); mny = min(mny, ty[k]); mxy = max(mxy, ty[k]) }
        val cx = (mnx + mxx) / 2; val cy = (mny + mxy) / 2
        var devC = 0.0; for (k in 0 until m) { val d = hypot(tx[k] - cx, ty[k] - cy); if (d > devC) devC = d }
        return Traj(devC, devC > R_MAX_BASE * w / 640.0, used.toIntArray(), tx, ty)
    }

    /** Chained-trajectory translation at frame [t] (linear between sampled indices), in working px,
     *  relative to the trajectory's own origin. Unbounded - that is the whole point (see [Traj.idxs]). */
    private fun trajAt(traj: Traj, t: Int, out: DoubleArray) {
        val k = traj.idxs.size
        if (k == 0) { out[0] = 0.0; out[1] = 0.0; return }
        if (t <= traj.idxs[0]) { out[0] = traj.tx[0]; out[1] = traj.ty[0]; return }
        if (t >= traj.idxs[k - 1]) { out[0] = traj.tx[k - 1]; out[1] = traj.ty[k - 1]; return }
        var i = 1
        while (i < k && traj.idxs[i] < t) i++
        val a = traj.idxs[i - 1]; val b = traj.idxs[i]
        val f = if (b > a) (t - a).toDouble() / (b - a) else 0.0
        out[0] = traj.tx[i - 1] + f * (traj.tx[i] - traj.tx[i - 1])
        out[1] = traj.ty[i - 1] + f * (traj.ty[i] - traj.ty[i - 1])
    }

    // =============================================================================================
    // Aligner - 3-level coarse->fine->native, sub-pixel global TRANSLATION onto the reference.
    // shiftOf(frame) returns working-res (dx,dy) such that frame(x+dx, y+dy) ~= ref(x,y).
    // =============================================================================================

    private class Aligner(private val w: Int, private val h: Int, private val levels: List<Level>) {
        /**
         * One pyramid level. [gray]/[valid] are the REFERENCE, built once in [build].
         *
         * [mGray]/[mValid]/[mPx] are per-level scratch for the MOVING frame, allocated once and reused
         * on every [shiftOf] call. They used to be allocated fresh inside `downGrayValid`: at the
         * native level that is a 6.39 MB px + 6.39 MB gray + 1.60 MB valid, i.e. **15.7 MB per call
         * across the three levels, ~4.7 GB over a 300-frame clip** - every one of them past ART's
         * large-object threshold, so each cost an LOS map plus a mandatory zero-fill that the very
         * next loop overwrote. Reuse is safe by contract: `shiftOf` is only ever called from the
         * phase's own thread (aggregatePlate and compositeFrames), never from inside a `Par` band.
         */
        class Level(val gray: IntArray, val valid: BooleanArray, val dw: Int, val dh: Int, val search: Int) {
            val mGray = IntArray(dw * dh)
            val mValid = BooleanArray(dw * dh)
            val mPx = IntArray(dw * dh)
        }

        /**
         * @param framePx the caller's ALREADY-EXTRACTED ARGB pixels of [frame], or null.
         *
         * Both hot call sites hold them: `aggregatePlate` does `mv.getPixels(mvpx, …)` immediately
         * before this, and so does `compositeFrames`. The native pyramid level (`9999` → `dimsFor`
         * caps the scale at 1.0, so dw == frame.width) then re-issued the IDENTICAL 1.6 M-pixel
         * `getPixels` into a freshly allocated array. Passing the caller's array in is bit-identical
         * - the values are read from the same Bitmap and neither call site mutates the array in
         * between (`fillMaskPixels` reads it and writes `mkpx`; `dilateInPlace` touches only `hole`).
         */
        fun shiftOf(frame: Bitmap, framePx: IntArray? = null): FloatArray {
            var est: FloatArray? = null; var pdw = 0; var pdh = 0
            var pinned = false          // FIX9: did any level's argmin land on its search boundary?
            for (lv in levels) {
                Prof.time(Prof.ALIGN_PREP) { downGrayValidInto(frame, framePx, lv) }
                val cx = if (est == null) 0 else Math.round(est!![0] * lv.dw / pdw.toDouble()).toInt()
                val cy = if (est == null) 0 else Math.round(est!![1] * lv.dh / pdh.toDouble()).toInt()
                est = Prof.time(Prof.ALIGN_SAD) {
                    bestShiftSub(lv.gray, lv.valid, lv.mGray, lv.mValid, lv.dw, lv.dh, cx, cy, lv.search)
                }
                pinned = pinned || lastShiftPinned
                pdw = lv.dw; pdh = lv.dh
            }
            // Counting only. A boundary argmin means the true shift is at least as large as the box,
            // so the returned value is a FLOOR - FIX1 §4.3's latent ±229.5 px ceiling, which nothing
            // has ever reported. The alignment result itself is unchanged.
            satAlignN++
            if (pinned) satAlignPinned++
            val last = levels.last()
            return floatArrayOf(est!![0] * w.toFloat() / last.dw, est!![1] * h.toFloat() / last.dh)
        }
        /** Shift (dx,dy) in working px such that other.ref(x+dx, y+dy) ~= this.ref(x,y) - lets a window
         *  warp a NEIGHBOR window's plate into its own coords for real-pixel borrowing. Uses the stored
         *  pyramid levels of both aligners (no frame decode). Both must share the same pyramid. */
        fun refShiftTo(other: Aligner): FloatArray {
            var est: FloatArray? = null; var pdw = 0; var pdh = 0
            for (i in levels.indices) {
                val a = levels[i]; val b = other.levels.getOrNull(i) ?: break
                val cx = if (est == null) 0 else Math.round(est!![0] * a.dw / pdw.toDouble()).toInt()
                val cy = if (est == null) 0 else Math.round(est!![1] * a.dh / pdh.toDouble()).toInt()
                est = bestShiftSub(a.gray, a.valid, b.gray, b.valid, a.dw, a.dh, cx, cy, a.search)
                pdw = a.dw; pdh = a.dh
            }
            val last = levels.last()
            return floatArrayOf((est?.get(0) ?: 0f) * w.toFloat() / last.dw, (est?.get(1) ?: 0f) * h.toFloat() / last.dh)
        }

        companion object {
            fun build(ref: Bitmap, w: Int, h: Int, pyramid: List<Pair<Int, Int>> = ALIGN_PYRAMID): Aligner {
                val lv = pyramid.map { (dim, sr) ->
                    val (dw, dh) = dimsFor(w, h, dim); val (g, v) = downGrayValid(ref, dw, dh)
                    Level(g, v, dw, dh, sr)
                }
                return Aligner(w, h, lv)
            }
        }
    }

    /** Downscale target longest-side -> (dw,dh); never UPSCALES (caps scale at 1.0 so 9999 => native). */
    private fun dimsFor(w: Int, h: Int, target: Int): Pair<Int, Int> {
        val s = min(1.0, target.toDouble() / max(w, h)); return max(2, (w * s).toInt()) to max(2, (h * s).toInt())
    }

    /** Downscale to (dw,dh), return grayscale + valid (valid = NOT the gray(128) person region).
     *  ALLOCATING form - kept for the two COLD callers (Aligner.build, 3x per clip; computeTrajectory,
     *  16x per clip). The per-frame path uses [downGrayValidInto], which reuses the level's scratch. */
    private fun downGrayValid(frame: Bitmap, dw: Int, dh: Int): Pair<IntArray, BooleanArray> {
        val small = if (frame.width == dw && frame.height == dh) frame else Bitmap.createScaledBitmap(frame, dw, dh, true)
        val px = IntArray(dw * dh); small.getPixels(px, 0, dw, 0, 0, dw, dh)
        val gray = IntArray(dw * dh); val valid = BooleanArray(dw * dh)
        for (i in px.indices) {
            val c = px[i]
            gray[i] = (Color.red(c) + Color.green(c) + Color.blue(c)) / 3
            // "valid" = NOT the person fill. Must use the SAME auto-detected fill colour as the hole
            // mask: with a hardcoded 128 on a gray(122) clip this both discards well-textured static
            // background (weakening the SAD alignment) and lets real silhouette pixels count as valid,
            // which lets the MOVING person drag the global-motion estimate.
            val v = fillV
            valid[i] = !(abs(Color.red(c) - v) <= GRAY_TOL + 2 &&
                abs(Color.green(c) - v) <= GRAY_TOL + 2 &&
                abs(Color.blue(c) - v) <= GRAY_TOL + 2)
        }
        return gray to valid
    }

    /**
     * [downGrayValid] for the HOT path: writes into the level's reusable scratch and runs ROW-PARALLEL.
     *
     * BIT-IDENTICAL to the allocating form. Three changes, none of which touches a value:
     *  1. **No allocation** - `lv.mGray` / `lv.mValid` / `lv.mPx` replace three fresh arrays. Every
     *     element of gray/valid is written before it is read, so a reused buffer cannot carry state
     *     across frames.
     *  2. **No redundant getPixels** - when the level is native-resolution the caller's [framePx] holds
     *     exactly what `frame.getPixels` would return, so it is used directly.
     *  3. **Row-parallel** - the loop is strictly elementwise (`gray[i]`/`valid[i]` are functions of
     *     `px[i]` and the read-only `fillV`), so disjoint row bands satisfy Par's disjoint-writes
     *     contract and there is no reduction and no nesting. It used to be the single largest piece of
     *     serial work left in Phase 1: 1.6 M iterations per frame, on one core, while six sat idle.
     *
     * The temporary scaled Bitmap is now RECYCLED. The old code leaked one per level per frame (~120 MB
     * of native bitmap over a 300-frame clip) for the GC to reclaim.
     */
    private fun downGrayValidInto(frame: Bitmap, framePx: IntArray?, lv: Aligner.Level) {
        val dw = lv.dw; val dh = lv.dh; val n = dw * dh
        val px: IntArray
        if (frame.width == dw && frame.height == dh) {
            px = if (framePx != null && framePx.size == n) framePx
                 else lv.mPx.also { frame.getPixels(it, 0, dw, 0, 0, dw, dh) }
        } else {
            val small = Bitmap.createScaledBitmap(frame, dw, dh, true)
            small.getPixels(lv.mPx, 0, dw, 0, 0, dw, dh)
            if (small !== frame) small.recycle()
            px = lv.mPx
        }
        val gray = lv.mGray; val valid = lv.mValid; val v = fillV
        Par.rows(dh, dw) { y0, y1 ->
            for (i in y0 * dw until y1 * dw) {
                val c = px[i]
                gray[i] = (Color.red(c) + Color.green(c) + Color.blue(c)) / 3
                valid[i] = !(abs(Color.red(c) - v) <= GRAY_TOL + 2 &&
                    abs(Color.green(c) - v) <= GRAY_TOL + 2 &&
                    abs(Color.blue(c) - v) <= GRAY_TOL + 2)
            }
        }
    }

    /**
     * Mean-SAD with EARLY EXIT, and bit-identical to the unbounded version for argmin purposes.
     *
     * The alignment search is the single largest cost in Phase 1 - 36.3 s of 102.8 s on a 300-frame
     * clip - because the native pyramid level evaluates 49 candidate shifts over ~400 000 strided
     * pixels each, per frame, for ~300 frames. Most of those candidates are obviously bad after a few
     * rows, yet the old code scored every one to completion.
     *
     * THE BOUND IS CONSERVATIVE ON PURPOSE. This returns a MEAN (e/c) and c is not known until the end,
     * so we cannot compare a partial mean against [bound]. What we CAN say is that the final mean is at
     * least `e / cMax`, where cMax is the largest possible sample count. Once `e / cMax > bound` the
     * final mean must exceed the bound too, so this candidate cannot be the minimum and its exact value
     * is irrelevant - return MAX_VALUE and stop.
     *
     * WHY THAT KEEPS THE RESULT IDENTICAL. A candidate is only abandoned when a DIFFERENT candidate has
     * already achieved a strictly lower error, so an abandoned candidate is provably not the argmin.
     * Ties are safe because the test is strict `>`: a candidate that merely equals the bound runs to
     * completion and keeps its place in the first-candidate-wins scan order.
     */
    private fun sadBounded(g0: IntArray, v0: BooleanArray, g1: IntArray, v1: BooleanArray, w: Int, h: Int,
                           dx: Int, dy: Int, bound: Double): Double {
        if (bound >= Double.MAX_VALUE) return sad(g0, v0, g1, v1, w, h, dx, dy)
        val cMax = ((w + 1) / 2).toLong() * ((h + 1) / 2)
        val cutoff = bound * cMax
        var e = 0.0; var c = 0
        var y = 0
        while (y < h) {
            var x = 0
            while (x < w) {
                val xx = x + dx; val yy = y + dy
                if (xx in 0 until w && yy in 0 until h) {
                    val i0 = y * w + x; val i1 = yy * w + xx
                    if (v0[i0] && v1[i1]) { e += abs(g0[i0] - g1[i1]); c++ }
                }
                x += 2
            }
            if (e > cutoff) return Double.MAX_VALUE     // cannot win; exact value never needed
            y += 2
        }
        return if (c > w * h / 16) e / c else Double.MAX_VALUE
    }

    /** Mean-SAD of g1 shifted by (dx,dy) vs g0 over pixels valid in BOTH (strided x2). MAX if too little overlap. */
    private fun sad(g0: IntArray, v0: BooleanArray, g1: IntArray, v1: BooleanArray, w: Int, h: Int, dx: Int, dy: Int): Double {
        var e = 0.0; var c = 0
        var y = 0
        while (y < h) {
            var x = 0
            while (x < w) {
                val xx = x + dx; val yy = y + dy
                if (xx in 0 until w && yy in 0 until h) {
                    val i0 = y * w + x; val i1 = yy * w + xx
                    if (v0[i0] && v1[i1]) { e += abs(g0[i0] - g1[i1]); c++ }
                }
                x += 2
            }
            y += 2
        }
        return if (c > w * h / 16) e / c else Double.MAX_VALUE
    }

    /** Integer best shift in [cx±r]×[cy±r] + parabolic sub-pixel; returns (dx,dy) as floats. */
    private fun bestShiftSub(g0: IntArray, v0: BooleanArray, g1: IntArray, v1: BooleanArray, w: Int, h: Int, cx: Int, cy: Int, r: Int): FloatArray {
        // The candidate grid is evaluated IN PARALLEL, but the argmin is then resolved SERIALLY in the
        // original dy-then-dx scan order. That distinction is the whole point: `sad` is pure, so
        // computing the errors concurrently cannot change them - but `e < best` is a STRICT compare, so
        // the FIRST candidate wins ties, and a parallel argmin that reduced in a different order could
        // pick a shift 1 px away and change every pixel of the plate. Errors in parallel, decision in
        // order => bit-identical.
        val side = 2 * r + 1
        val errs = DoubleArray(side * side)
        // Seed the bound with the CENTRE candidate - the pyramid has already put the true shift near it,
        // so it is usually a tight bound and most of the other 48 abandon early.
        val seed = sad(g0, v0, g1, v1, w, h, cx, cy)
        val centreIdx = r * side + r
        errs[centreIdx] = seed
        Par.range(side * side, 8) { i0, i1 ->
            // Each band tightens its OWN bound as it goes. Bands never share it, so which candidates get
            // abandoned is deterministic per band - and an abandoned candidate is provably not the
            // minimum either way, so the argmin below is unaffected.
            var bound = seed
            for (idx in i0 until i1) {
                if (idx == centreIdx) continue
                val dy = cy - r + idx / side
                val dx = cx - r + idx % side
                val e = sadBounded(g0, v0, g1, v1, w, h, dx, dy, bound)
                errs[idx] = e
                if (e < bound) bound = e
            }
        }
        var bx = cx; var by = cy; var best = Double.MAX_VALUE
        for (idx in 0 until side * side) {
            val e = errs[idx]
            if (e < best) { best = e; by = cy - r + idx / side; bx = cx - r + idx % side }
        }
        // FIX9: a boundary argmin means the true shift is at least as large as the search box, so
        // the value returned is a floor. Recorded, never acted on here - [Aligner.shiftOf] reads it;
        // [computeTrajectory] pins by design on a fast pan and deliberately does not.
        lastShiftPinned = best < Double.MAX_VALUE &&
            (bx == cx - r || bx == cx + r || by == cy - r || by == cy + r)
        if (best == Double.MAX_VALUE) return floatArrayOf(0f, 0f)
        // The four sub-pixel probes are UNBOUNDED full-frame SADs and they used to run one after the
        // other on the calling thread, while the pool that had just evaluated the 48-candidate grid sat
        // idle. At the native level each is a scan of ~400 000 sampled positions, so these four alone
        // are ~1.6 M samples of purely serial work per frame. `sad` is a pure function of its inputs - 
        // it writes nothing - so evaluating the four concurrently cannot change any of the four values,
        // and each lands in its own slot. Bit-identical; `parab` still consumes them in the same order.
        val probe = DoubleArray(4)
        Par.range(4, 1) { i0, i1 ->
            for (i in i0 until i1) probe[i] = when (i) {
                0 -> sad(g0, v0, g1, v1, w, h, bx - 1, by)
                1 -> sad(g0, v0, g1, v1, w, h, bx + 1, by)
                2 -> sad(g0, v0, g1, v1, w, h, bx, by - 1)
                else -> sad(g0, v0, g1, v1, w, h, bx, by + 1)
            }
        }
        return floatArrayOf(bx + parab(probe[0], best, probe[1]), by + parab(probe[2], best, probe[3]))
    }

    private fun parab(a: Double, b: Double, c: Double): Float {
        if (a >= Double.MAX_VALUE || c >= Double.MAX_VALUE) return 0f
        val d = a - 2 * b + c
        if (abs(d) < 1e-6) return 0f
        return (0.5 * (a - c) / d).toFloat().coerceIn(-0.5f, 0.5f)
    }

    private fun bilinearRGB(px: IntArray, w: Int, h: Int, fx: Float, fy: Float): Int {
        val x0 = floor(fx).toInt().coerceIn(0, w - 1); val y0 = floor(fy).toInt().coerceIn(0, h - 1)
        val x1 = (x0 + 1).coerceAtMost(w - 1); val y1 = (y0 + 1).coerceAtMost(h - 1)
        val tx = (fx - x0).coerceIn(0f, 1f); val ty = (fy - y0).coerceIn(0f, 1f)
        val c00 = px[y0 * w + x0]; val c10 = px[y0 * w + x1]; val c01 = px[y1 * w + x0]; val c11 = px[y1 * w + x1]
        fun ch(sh: Int): Int {
            val a = (c00 ushr sh) and 0xFF; val b = (c10 ushr sh) and 0xFF
            val c = (c01 ushr sh) and 0xFF; val d = (c11 ushr sh) and 0xFF
            return ((a * (1 - tx) + b * tx) * (1 - ty) + (c * (1 - tx) + d * tx) * ty).toInt().coerceIn(0, 255)
        }
        return Color.rgb(ch(16), ch(8), ch(0))
    }

    // =============================================================================================
    // STEP 2a - STATIC/JITTER plate: aligned + exposure-normalized per-pixel temporal trimmed-mean
    // =============================================================================================

    private fun buildStaticPlate(
        masked: FrameSource, mask: FrameSource?, n: Int, w: Int, h: Int, np: Int, maxDim: Int,
        aligner: Aligner?, shiftCache: HashMap<Int, FloatArray>?, refMeanLuma: Double,
        plate: IntArray, core: BooleanArray, union: BooleanArray, log: Logger,
    ) {
        val used = plateSampleCount(n, np, w, h)
        log.log("[inpaint] plate budget (static): $lastSampleBudget")
        // 🔴 SPREAD THE SAMPLES ACROSS THE WHOLE CLIP.
        //
        // This was `stride = max(1, n / used)` followed by `.take(used)` - INTEGER division, so on a
        // 70-frame clip with used=41 the stride collapsed to 1 and the plate was built from frames
        // 0..40 ONLY: the first 59 % of the clip. Every piece of background revealed solely in the
        // later frames was therefore absent from the plate, so the second half of the output was
        // reconstructed from a plate that had never seen it - visible as instability there. On a
        // 300-frame clip the same expression gave stride 7 and stopped at frame 280, still clipping
        // the tail. Even spacing over [0, n-1] is the idiom computeTrajectory already uses.
        //
        // This CHANGES the sampled set and therefore the output. It is a quality FIX, not a
        // refactor: more of the clip is represented, so more of the hole is filled from REAL pixels
        // and less is left to the neural core. Watch `hole_from_REAL_%` rise to confirm it.
        val frames = when {
            used >= n -> (0 until n).toList()
            used <= 1 -> listOf(n / 2)
            else -> List(used) { (it.toLong() * (n - 1) / (used - 1L)).toInt() }
        }
        log.log("[inpaint] plate: ${if (aligner != null) "aligned " else ""}trimmed-mean over ${frames.size} " +
            "frames spread over 0..${frames.lastOrNull() ?: 0} of $n @ ${w}x$h")
        aggregatePlate(masked, mask, frames, w, h, np, maxDim, aligner, shiftCache, refMeanLuma, plate, core, union, log)
    }

    /** 64-bit words per image ROW in the validity bitset. Row-aligned on purpose: a word must never
     *  straddle two rows, or two parallel row-bands would read-modify-write the same word and lose
     *  each other's bits. The padding waste is (64 - w % 64) bits per row - 1.3 % at w = 1264. */
    private fun validWordsPerRow(w: Int): Int = (w + 63) ushr 6

    /** Bytes of sample stack per FRAME: 3 bytes of RGB per pixel + 1 BIT of validity per pixel. */
    private fun bytesPerSample(w: Int, h: Int, np: Int): Long =
        np.toLong() * 3 + validWordsPerRow(w).toLong() * h * 8

    /** Max frames to hold in the sample stack, bounded by the real heap, the co-resident fixed buffers, and
     *  (dynamic path) any already-committed window memory [retainedBytes] so K windows can't OOM.
     *
     *  QUALITY: this is the knob that decides how much of the clip the plate actually sees. `valid` used
     *  to be a BooleanArray - one whole BYTE per pixel per frame to store one bit - which made the stack
     *  4 bytes/px/frame and capped the clip at 41 samples on a 1264² frame. As a row-aligned bitset it is
     *  ~3.13 bytes/px/frame, so the SAME memory budget now buys ~52 samples: +27 % temporal evidence for
     *  the trimmed mean, more of the hole recovered from REAL pixels, and a smaller neural core. */
    private fun plateSampleCount(n: Int, np: Int, w: Int, h: Int, retainedBytes: Long = 0L): Int {
        val per = bytesPerSample(w, h, np)
        val maxMem = Runtime.getRuntime().maxMemory()
        val heapBudget = min(MEM_BUDGET, maxMem * HEAP_FRACTION_NUM / HEAP_FRACTION_DEN)
        val avail = (heapBudget - np.toLong() * FIXED_PER_PX - retainedBytes).coerceAtLeast(per)
        val cap = max(1, (avail / per).toInt())
        val out = min(min(n, MED_MAX_SAMPLES), cap)
        // Every term, because the whole point is to find out WHICH ONE BINDS. Diagnosing the
        // "same spot always" artefact required reconstructing this arithmetic off-device from a
        // single reported sample count; printing it turns that into a one-line read. If `maxMem`
        // is the binding term, no MEM_BUDGET will help and the fix has to reduce bytes-per-sample
        // (crop the stack to the mask union) or move it off the Dalvik heap.
        lastSampleBudget = "samples=$out of n=$n  cap=$cap  per=${per / 1_000_000}MB/frame  " +
            "maxMemory=${maxMem / 1_000_000}MB  heapBudget=${heapBudget / 1_000_000}MB  " +
            "fixed=${np.toLong() * FIXED_PER_PX / 1_000_000}MB  retained=${retainedBytes / 1_000_000}MB  " +
            "BINDING=" + when {
                out == n -> "none (all frames used)"
                out == MED_MAX_SAMPLES -> "MED_MAX_SAMPLES"
                heapBudget == MEM_BUDGET -> "MEM_BUDGET"
                else -> "maxMemory*$HEAP_FRACTION_NUM/$HEAP_FRACTION_DEN"
            }
        return out
    }

    /** Last budget computation, logged by the callers so a device run reveals which term binds. */
    @Volatile private var lastSampleBudget: String = ""

    /** Aligned + exposure-normalized per-pixel temporal trimmed-mean of REAL background over [frames],
     *  in the aligner's reference coords. Fills plate/core/union. Reused by the static path AND each
     *  window of the dynamic path. */
    private fun aggregatePlate(
        masked: FrameSource, mask: FrameSource?, frames: List<Int>, w: Int, h: Int, np: Int, maxDim: Int,
        aligner: Aligner?, shiftCache: HashMap<Int, FloatArray>?, refMeanLuma: Double,
        plate: IntArray, core: BooleanArray, union: BooleanArray, log: Logger,
    ) {
        val s = frames.size
        val wpr = validWordsPerRow(w)
        val vPerSample = wpr * h             // 64-bit words of validity per frame
        val samp = ByteArray(s * np * 3)     // RGB samples of the real background
        val valid = LongArray(s * vPerSample)   // 1 bit per (frame,pixel): real-background sample?
        val mvpx = IntArray(np); val mkpx = IntArray(np)
        val hole = BooleanArray(np)   // reused across sampled frames (was a fresh 1.6 MB alloc per frame)
        // SEQUENTIAL DECODE. MediaMetadataRetriever.getFrameAtIndex is far cheaper for a CONSECUTIVE
        // index than a scattered one - it appears to decode forward from a keyframe on every seek.
        // MEASURED on the 70-frame clip: spreading the sample set across the whole clip (the quality
        // fix above) pushed the aggregate's 41 reads from ~50 ms to ~416 ms each and Phase 1's decode
        // bucket from 10.0 s to 25.0 s. Walking EVERY frame in order and skipping the unsampled ones
        // costs more decode CALLS but each is cheap, and the sample SET is unchanged - so this is a
        // pure latency fix with bit-identical output.
        val tMax = frames.lastOrNull() ?: -1
        val idxOf = IntArray(tMax + 1) { -1 }
        frames.forEachIndexed { si, t -> if (t in 0..tMax) idxOf[t] = si }
        // This walk is strictly ascending over 0..tMax, which is exactly what the prefetcher needs.
        // Only `masked` is prefetched: the MASK source is read here for the ~52 SAMPLED indices only,
        // which is not a sequential pattern, and the prefetcher would just retire itself on the first
        // mismatch. (Making the mask read sequential too is a separate change, not this one.)
        masked.startPrefetch(tMax + 1, maxDim)
        for (t in 0..tMax) {
            val mv = Prof.time(Prof.DECODE) { masked.frameAt(t, maxDim) } ?: continue
            val si = idxOf[t]
            if (si < 0) { mv.recycle(); continue }   // decoded only to keep the read head sequential
            Prof.time(Prof.DECODE) { mv.getPixels(mvpx, 0, w, 0, 0, w, h) }
            Prof.time(Prof.MASK) {
                fillMaskPixels(mask, t, maxDim, w, h, mvpx, mkpx)
                maskToBoolInto(mkpx, np, hole)
            }
            dilateInPlace(hole, w, h, MASK_DILATE)   // person edge + codec bleed OUT of the samples
            // mvpx is passed in so the native pyramid level does not re-issue the same full-frame
            // getPixels the line above already did. Nothing between the two writes mvpx.
            val sh = Prof.time(Prof.ALIGN) { aligner?.shiftOf(mv, mvpx) } ?: FLOAT00
            if (aligner != null) shiftCache?.put(t, sh)
            val dx = sh[0]; val dy = sh[1]
            // exposure-normalize. The clamp is instrumented (FIX9): a sample whose true gain is
            // outside 0.85..1.18 enters the trimmed mean with a residual brightness error, and
            // nothing used to say so. Counting only - `g` is computed by the identical expression.
            val gRaw = refMeanLuma / meanLuma(mvpx, hole, np)
            satGainN++
            if (gRaw < 0.85 || gRaw > 1.18) {
                satGainPinned++
                if (abs(gRaw - 1.0) > abs(satGainWorst - 1.0)) satGainWorst = gRaw
            }
            val g = gRaw.coerceIn(0.85, 1.18)
            val base = si * np
            val vbase = si * vPerSample
            Prof.time(Prof.SAMPLE) {
                // Rows are independent: every write is to this pixel's own refp slot in samp/valid, and
                // union[refp]=true is idempotent. Bit-identical to the serial loop.
                Par.rows(h, w) { y0, y1 ->
                    for (y in y0 until y1) for (x in 0 until w) {
                        val refp = y * w + x
                        val fx = x + dx; val fy = y + dy
                        val ix = Math.round(fx); val iy = Math.round(fy)
                        if (ix < 0 || ix >= w || iy < 0 || iy >= h) continue
                        if (hole[iy * w + ix]) { union[refp] = true; continue }
                        val c = if (dx == 0f && dy == 0f) mvpx[refp] else bilinearRGB(mvpx, w, h, fx, fy)
                        // row-aligned word => this band owns it exclusively, no atomics needed
                        val vi = vbase + y * wpr + (x ushr 6)
                        valid[vi] = valid[vi] or (1L shl (x and 63))
                        val o = (base + refp) * 3
                        samp[o] = ((Color.red(c) * g).toInt().coerceIn(0, 255)).toByte()
                        samp[o + 1] = ((Color.green(c) * g).toInt().coerceIn(0, 255)).toByte()
                        samp[o + 2] = ((Color.blue(c) * g).toInt().coerceIn(0, 255)).toByte()
                    }
                }
            }
            if (si % 10 == 0) log.log("[inpaint] sampled $si/$s")
        }

        val minCov = max(3, (0.05 * s).toInt())
        log.log("[inpaint] aggregating ${np / 1000}k px x $s samples on ${Par.WORKERS} worker(s) …")
        Prof.time(Prof.GATHER) {
            Par.rows(h, w) { y0, y1 ->
                // Per-band scratch. rb/gb/bb were shared across the whole loop before, which is only
                // safe single-threaded; the 256-bin histogram must be per-band and all-zero on entry.
                // Writes are disjoint (plate[p], core[p] for rows in this band only).
                val rb = IntArray(s); val gb = IntArray(s); val bb = IntArray(s); val bin = IntArray(256)
                for (y in y0 until y1) for (x in 0 until w) {
                    val p = y * w + x
                    val vOff = y * wpr + (x ushr 6)
                    val vBit = 1L shl (x and 63)
                    var k = 0
                    for (si in 0 until s) if (valid[si * vPerSample + vOff] and vBit != 0L) {
                        val o = (si * np + p) * 3
                        rb[k] = samp[o].toInt() and 0xFF; gb[k] = samp[o + 1].toInt() and 0xFF; bb[k] = samp[o + 2].toInt() and 0xFF; k++
                    }
                    when {
                        k >= minCov -> plate[p] = Color.rgb(robustCenter(rb, k, bin), robustCenter(gb, k, bin), robustCenter(bb, k, bin))
                        k > 0 -> { if (union[p]) core[p] = true; plate[p] = Color.rgb(robustCenter(rb, k, bin), robustCenter(gb, k, bin), robustCenter(bb, k, bin)) }
                        else -> { core[p] = true; plate[p] = Color.BLACK }   // never covered -> neural-fill (never leave raw black)
                    }
                }
            }
        }
    }

    private val FLOAT00 = floatArrayOf(0f, 0f)

    /**
     * Trimmed mean (middle 60%) of k samples - robust like the median but denoises sensor noise → sharper.
     *
     * BIT-IDENTICAL REWRITE (was: Arrays.sort + sum of the middle slice). Sample values are bytes 0..255,
     * so a 256-bin counting histogram reproduces the sorted order exactly for O(k + touched bins) instead
     * of O(k log k) - and without mutating [buf]. This runs ~3.4 M times per clip (3 channels x the ~71 %
     * of 1.6 M pixels that have at least one valid sample).
     *
     * Bit-identity argument: for the k >= 5 branch, the count of value v whose SORTED rank falls inside
     * [lo, hi) is exactly the overlap of [rank, rank+count(v)) with [lo, hi). Summing v * overlap is
     * therefore the same multiset sum as summing sorted[lo until hi], and the Long accumulator plus
     * integer division are unchanged. For k < 5 the walk returns the value at rank k/2, which is what
     * sorted[k/2] was. Integer addition is associative, so no reassociation hazard exists.
     *
     * [bin] must be a caller-owned IntArray(256) that is all-zero on entry; it is left all-zero on exit
     * (only the touched bins are cleared, which is why this stays O(k) and not O(256)). Passing it in
     * rather than sharing one field keeps the function safe for the row-band worker pool.
     */
    private fun robustCenter(buf: IntArray, k: Int, bin: IntArray): Int {
        if (k <= 0) return 0
        var mn = 255; var mx = 0
        for (i in 0 until k) {
            val v = buf[i]
            bin[v]++
            if (v < mn) mn = v
            if (v > mx) mx = v
        }
        val res: Int
        if (k < 5) {
            val target = k / 2
            var rank = 0; var out = mn; var v = mn
            while (v <= mx) {
                val c = bin[v]
                if (c > 0) {
                    if (target < rank + c) { out = v; break }
                    rank += c
                }
                v++
            }
            res = out
        } else {
            val lo = k / 5; val hi = k - k / 5
            var rank = 0; var sum = 0L; var v = mn
            while (v <= mx) {
                val c = bin[v]
                if (c > 0) {
                    val a = max(rank, lo); val b = min(rank + c, hi)
                    if (b > a) sum += (b - a).toLong() * v
                    rank += c
                    if (rank >= hi) break
                }
                v++
            }
            res = (sum / (hi - lo)).toInt()
        }
        for (i in 0 until k) bin[buf[i]] = 0
        return res
    }

    private fun meanLuma(px: IntArray, hole: BooleanArray, np: Int): Double {
        var sum = 0.0; var c = 0; var p = 0
        while (p < np) { if (!hole[p]) { sum += (Color.red(px[p]) + Color.green(px[p]) + Color.blue(px[p])) / 3.0; c++ }; p += 2 }
        return if (c > 0) sum / c else 128.0
    }

    private fun referenceMeanLuma(ref: Bitmap?, mask: FrameSource?, refT: Int, maxDim: Int, w: Int, h: Int, np: Int): Double {
        if (ref == null) return 128.0
        val mvpx = IntArray(np); val mkpx = IntArray(np)
        ref.getPixels(mvpx, 0, w, 0, 0, w, h)
        fillMaskPixels(mask, refT, maxDim, w, h, mvpx, mkpx)
        return meanLuma(mvpx, maskToBool(mkpx, np), np)
    }

    // =============================================================================================
    // STEP 2b - DYNAMIC baseline: union-fill (frame-0 real + neural over the whole union)
    // =============================================================================================

    private fun buildUnionBaseline(
        masked: FrameSource, mask: FrameSource?, n: Int, w: Int, h: Int, np: Int, maxDim: Int,
        plate: IntArray, core: BooleanArray, union: BooleanArray, log: Logger,
    ) {
        val mvpx = IntArray(np); val mkpx = IntArray(np); var frame0: IntArray? = null
        for (t in 0 until n) {
            val mv = masked.frameAt(t, maxDim) ?: continue
            mv.getPixels(mvpx, 0, w, 0, 0, w, h)
            if (t == 0) frame0 = mvpx.copyOf()
            fillMaskPixels(mask, t, maxDim, w, h, mvpx, mkpx)
            for (p in 0 until np) if ((mkpx[p] and 0xFF) >= 128) union[p] = true
            if (t % 20 == 0) log.log("[inpaint] union scan $t/$n")
        }
        System.arraycopy(frame0 ?: IntArray(np), 0, plate, 0, np)
        for (p in 0 until np) if (union[p]) core[p] = true   // the whole union is neural-filled
    }

    // =============================================================================================
    // STEP 2c - DYNAMIC (moving camera): windowed reveal-and-fill
    // =============================================================================================

    private class WindowPlate(
        val plate: IntArray, val core: BooleanArray, val union: BooleanArray,
        val aligner: Aligner?, val start: Int, val end: Int, val overlap: Int,
    ) { val center: Double get() = (start + end - 1) / 2.0 }

    /** Split the clip into overlapping windows short enough that translation-alignment to the window
     *  reference holds (window length adapts to pan speed), build the verified aligned plate per window,
     *  and neural-fill each core. A moving camera reveals background progressively, so per-window cores
     *  are small; adjacent windows cover what a single global reference could not. */
    private fun buildWindowedPlates(
        masked: FrameSource, mask: FrameSource?, n: Int, w: Int, h: Int, np: Int, maxDim: Int,
        traj: Traj, getLama: () -> LamaInpainter?, log: Logger,
    ): List<WindowPlate> {
        val rMax = R_MAX_BASE * w / 640.0
        val vPerFrame = if (n > 1) 2.0 * traj.devC / (n - 1) else 0.0
        val len = if (vPerFrame > 0.01) (2.0 * rMax / vPerFrame).toInt().coerceIn(WINDOW_MIN, WINDOW_MAX) else WINDOW_MAX
        val overlap = max(4, len / 6)
        // cap the window count K <= MAX_WINDOWS by widening the hop (bounds memory + LaMa calls + alignment)
        var hop = max(1, len - overlap)
        if (n > len) hop = max(hop, (n - len + MAX_WINDOWS - 2) / (MAX_WINDOWS - 1))
        log.log("[inpaint] windows: L=$len overlap=$overlap hop=$hop (pan ~${"%.2f".format(vPerFrame)}px/f, K<=$MAX_WINDOWS)")
        val windows = ArrayList<WindowPlate>()
        var start = 0
        while (start < n) {
            val end = min(start + len, n)
            val refT = (start + end) / 2
            val refFrame = masked.frameAt(refT, maxDim)
            val aligner = if (refFrame != null) Aligner.build(refFrame, w, h, WINDOW_PYRAMID) else null
            val refMeanLuma = referenceMeanLuma(refFrame, mask, refT, maxDim, w, h, np)
            // budget-aware: reserve the memory already held by committed windows (~7 bytes/px each)
            // 🔴 SPREAD THE SAMPLES ACROSS THE WHOLE WINDOW - the same integer-division defect
            // f74dd43 fixed in buildStaticPlate was left live here. `st = (end-start)/cap` followed by
            // `.take(cap)` collapses to stride 1 whenever cap < span, so a 71-frame window with cap 55
            // was built from frames [start, start+55) - the FIRST 77 % of the window - and the last
            // 23 % never entered its plate. That is exactly the "later-half artifact" symptom, still
            // present on the DYNAMIC path. MEASURED on c1 (2026-08-08): cap rose 43 -> 55 on
            // 2026-08-04 (the validity bitset), so this truncation ALSO widened by 28 %.
            // Even spacing over [start, end-1] is the idiom buildStaticPlate and computeTrajectory use.
            val cap = plateSampleCount(end - start, np, w, h, windows.size.toLong() * np * 7L)
            val span = end - start
            val frames = when {
                cap >= span -> (start until end).toList()
                cap <= 1 -> listOf((start + end) / 2)
                else -> List(cap) { start + (it.toLong() * (span - 1) / (cap - 1L)).toInt() }
            }
            log.log("[inpaint] window [$start,$end) ref=$refT (${frames.size} frames)")
            val plate = IntArray(np); val core = BooleanArray(np); val union = BooleanArray(np)
            aggregatePlate(masked, mask, frames, w, h, np, maxDim, aligner, null, refMeanLuma, plate, core, union, log)
            windows.add(WindowPlate(plate, core, union, aligner, start, end, overlap))
            if (end >= n) break
            start += hop
        }
        // reuse real pixels revealed in NEIGHBOR windows, then neural-fill only the never-revealed remainder
        borrowAcrossWindows(windows, w, h, np, log)
        for (win in windows) if (win.core.any { it }) fillCore(win.plate, win.core, w, h, np, getLama, log)
        return windows
    }

    /** Cross-window real-pixel borrowing: a moving camera reveals a hidden world point in SOME window even
     *  where it's occluded in another. For each window's never-revealed core pixel, if an adjacent window
     *  (±1, ±2) holds a REAL pixel at that world point (found via the ref->ref shift), copy it in and clear
     *  the core flag - shrinking the neural core toward "never revealed ANYWHERE". Two rounds let reveals
     *  propagate across windows. Real background pixels only -> §2-safe (never a person/gray sample). */
    private fun borrowAcrossWindows(windows: List<WindowPlate>, w: Int, h: Int, np: Int, log: Logger) {
        if (windows.size < 2) return
        val before = windows.sumOf { win -> win.core.count { it } }
        repeat(2) {
            for (ki in windows.indices) {
                val k = windows[ki]; val ka = k.aligner ?: continue
                if (k.core.none { it }) continue
                for (jd in intArrayOf(-1, 1, -2, 2)) {
                    val ji = ki + jd; if (ji < 0 || ji >= windows.size) continue
                    val j = windows[ji]; val ja = j.aligner ?: continue
                    val d = ka.refShiftTo(ja); val dx = d[0]; val dy = d[1]
                    for (y in 0 until h) for (x in 0 until w) {
                        val p = y * w + x
                        if (!k.core[p]) continue
                        val sx = x + dx; val sy = y + dy
                        val ix = Math.round(sx); val iy = Math.round(sy)
                        if (ix < 0 || ix >= w || iy < 0 || iy >= h) continue
                        if (!j.core[iy * w + ix]) {   // neighbor has a REAL pixel at that world point
                            k.plate[p] = bilinearRGB(j.plate, w, h, sx, sy)
                            k.core[p] = false
                        }
                    }
                }
            }
        }
        val after = windows.sumOf { win -> win.core.count { it } }
        val pct = if (before > 0) 100 * (before - after) / before else 0
        log.log("[inpaint] cross-window borrow: core $before -> $after px ($pct% reused from neighbors)")
    }

    /** Trapezoidal window weight in [0,1]: full inside the window, smoothstep-ramped only in the overlap
     *  zones so adjacent windows crossfade - and every frame is covered at weight ~1, so the hole is
     *  always filled from a plate, never the raw frame. */
    private fun windowWeight(win: WindowPlate, t: Int, isFirst: Boolean, isLast: Boolean): Float {
        if (t < win.start || t >= win.end) return 0f
        var wgt = 1.0
        val o = win.overlap
        if (!isFirst && t < win.start + o) wgt = smoothstep((t - win.start + 0.5) / o)
        if (!isLast && t >= win.end - o) wgt = min(wgt, smoothstep((win.end - t - 0.5) / o))
        return wgt.coerceIn(0.0, 1.0).toFloat()
    }

    private fun smoothstep(x: Double): Double { val c = x.coerceIn(0.0, 1.0); return c * c * (3 - 2 * c) }

    /** DYNAMIC composite: for each frame, warp each covering window's plate into the frame, ring-match it,
     *  and weighted-blend by the trapezoidal window weight (a smooth crossfade in overlaps). The hole
     *  (COMPOSITE_DILATE) is always filled from the plate blend; a decode-null re-emits the previous frame. */
    private fun compositeWindowed(
        masked: FrameSource, mask: FrameSource?, n: Int, w: Int, h: Int, np: Int, maxDim: Int,
        windows: List<WindowPlate>, out: File, log: Logger,
    ): Pair<Long, Int> {
        log.log("[inpaint] compositing ${windows.size} windows -> ${out.name}")
        val mvpx = IntArray(np); val mkpx = IntArray(np); val outPx = IntArray(np)
        val alpha = FloatArray(np); val ablur = FloatArray(np); val warp = IntArray(np)
        // Seam-match scratch (null when the slider is at 0, so the old path allocates nothing).
        val (seamBand, seamRes, seamUnk) = seamScratch(np)
        val accR = FloatArray(np); val accG = FloatArray(np); val accB = FloatArray(np); val accW = FloatArray(np)
        val uncov = BooleanArray(np)   // hole pixels no in-bounds window covered -> push-pull, never an edge smear
        val prevMv = IntArray(np); var havePrev = false
        // Shift of the FIRST covering window, this frame and the previous one - the motion
        // compensation estimateGrainSigma needs. Latched per frame from window `wi` = the first with
        // weight > 0, which is the same window at t and t-1 for every pair the estimator uses
        // (grainSigma is computed once, at the first frame with a predecessor).
        var prevDx = 0f; var prevDy = 0f
        var frameDx = 0f; var frameDy = 0f; var haveFrameShift = false
        var grainSigma = 0f
        val lastOut = IntArray(np); var haveLast = false
        var holeSum = 0L; var written = 0
        // pushPull fire-rate tripwire - counting only, no pixel is touched by these four lines.
        var ppFrames = 0; var ppSum = 0.0; var ppWorst = 0.0
        Mp4Encoder(out, masked.srcFps).use { enc ->
            for (t in 0 until n) {
                val mv = masked.frameAt(t, maxDim)
                if (mv == null) { if (haveLast) { enc.writeFrame(lastOut, w, h); written++ }; continue }
                haveFrameShift = false
                mv.getPixels(mvpx, 0, w, 0, 0, w, h)
                fillMaskPixels(mask, t, maxDim, w, h, mvpx, mkpx)
                val hole = maskToBool(mkpx, np); dilateInPlace(hole, w, h, COMPOSITE_DILATE)
                var holePx = 0; for (p in 0 until np) { if (hole[p]) holePx++; alpha[p] = if (hole[p]) 1f else 0f }
                holeSum += holePx
                Arrays.fill(accR, 0f); Arrays.fill(accG, 0f); Arrays.fill(accB, 0f); Arrays.fill(accW, 0f)
                for ((wi, win) in windows.withIndex()) {
                    val wgt = windowWeight(win, t, wi == 0, wi == windows.size - 1); if (wgt <= 0f) continue
                    val sh = win.aligner?.shiftOf(mv, mvpx) ?: FLOAT00
                    val dx = sh[0]; val dy = sh[1]
                    if (!haveFrameShift) { frameDx = dx; frameDy = dy; haveFrameShift = true }
                    // Row-parallel, bit-identical (elementwise, disjoint bands) - this was the last
                    // full-frame serial pixel loop left in the file.
                    // ⚠ UNMEASURED: this DYNAMIC (moving-camera) branch is not taken by the 300 f
                    // CASE1 clip (devC 61.2 px < R_MAX 118.5 px -> the static path runs), so it carries
                    // the same reasoning as compositeFrames but no device number of its own.
                    if (dx == 0f && dy == 0f) System.arraycopy(win.plate, 0, warp, 0, np)
                    else Par.rows(h, w) { y0, y1 ->
                        for (y in y0 until y1) for (x in 0 until w) warp[y * w + x] = bilinearRGB(win.plate, w, h, x - dx, y - dy)
                    }
                    val gain = matchSurroundings(mvpx, warp, hole, ablur, w, h, np, seamBand, seamRes, seamUnk)
                    if (havePrev && grainSigma <= 0f) {
                        grainSigma = estimateGrainSigma(mvpx, prevMv, hole, w, h,
                            Math.round(prevDx - frameDx), Math.round(prevDy - frameDy))
                        log.log("[inpaint] temporal grain sigma = ${"%.2f".format(grainSigma)} " +
                            "grey levels (measured from this clip's own background)")
                        // FIX9: this fallback still uses the INTEGER-compensated single-pair
                        // estimator, and on a moving-camera clip that pins. MEASURED on c1: it
                        // returns 6.00 on 10 of 12 consecutive pairs. Say so rather than shipping a
                        // clamp as if it were a measurement. (The fix is not to patch this path but
                        // to reach the mosaic, which holds an exact homography - see
                        // staticGrainSigma / grainSigmaWarp.)
                        if (grainSigma >= GRAIN_MAX - 1e-3f) log.log("[inpaint] ⚠ SATURATED: that " +
                            "sigma is PINNED at GRAIN_MAX=${"%.1f".format(GRAIN_MAX)} - the " +
                            "windowed fallback's estimator measures residual camera motion, not " +
                            "noise, so the fill will be ~1.9x over-sharp (FIX9_CLAMPS.md)")
                    }
                    // Accumulate ONLY where the plate actually covers this frame pixel. A sample whose source
                    // (x-dx, y-dy) falls OUTSIDE the plate would be edge-clamped by bilinearRGB - painting a
                    // moving dark smear at the pan-leading edge (the "black border marks as the video moves").
                    // Skip those; the pixel is covered by another window, or push-pull-filled below.
                    // Row-parallel and bit-identical: each pixel p is accumulated EXACTLY ONCE per
                    // (window, frame), so no band reassociates another band's float adds - the window
                    // loop that does the summing across windows stays serial and in the same order.
                    Par.rows(h, w) { y0, y1 ->
                        for (y in y0 until y1) for (x in 0 until w) {
                            val p = y * w + x
                            if (!hole[p]) continue
                            val sx = x - dx; val sy = y - dy
                            if (sx < -0.5f || sx > w - 0.5f || sy < -0.5f || sy > h - 0.5f) continue
                            accR[p] += (Color.red(warp[p]) * gain[0]) * wgt
                            accG[p] += (Color.green(warp[p]) * gain[1]) * wgt
                            accB[p] += (Color.blue(warp[p]) * gain[2]) * wgt
                            accW[p] += wgt
                        }
                    }
                }
                boxBlur(alpha, ablur, w, h, FEATHER)
                var anyUncov = false
                for (p in 0 until np) {
                    val a = ablur[p]
                    if (a <= 0f) { outPx[p] = mvpx[p]; uncov[p] = false; continue }
                    val iw = accW[p]
                    if (iw <= 0f) {
                        // Only a genuine HOLE pixel with no window cover needs push-pull; a non-hole feather-ring
                        // pixel (a>0 from blur bleed) is real background - keep it as-is, don't diffuse over it.
                        if (hole[p]) { uncov[p] = true; anyUncov = true } else uncov[p] = false
                        outPx[p] = mvpx[p]; continue
                    }
                    uncov[p] = false
                    val pr = (accR[p] / iw).toInt().coerceIn(0, 255)
                    val pg = (accG[p] / iw).toInt().coerceIn(0, 255)
                    val pb = (accB[p] / iw).toInt().coerceIn(0, 255)
                    // PER-FRAME GRAIN - this branch ESTIMATED grainSigma (and logged it) and then never
                    // used it, so fb4a04d's "the fill was too CLEAN" fix was silently inert on every
                    // moving-camera clip. Same expression and same scaling-by-alpha as compositeFrames,
                    // so the fill fades its grain out through the feather exactly as the plate does.
                    val nz = if (grainSigma > 0f) grainAt(p, t) * grainSigma * a else 0f
                    if (a >= 1f) {
                        outPx[p] = Color.rgb((pr + nz).toInt().coerceIn(0, 255),
                            (pg + nz).toInt().coerceIn(0, 255), (pb + nz).toInt().coerceIn(0, 255))
                        continue
                    }
                    val r = (Color.red(mvpx[p]) * (1 - a) + pr * a + nz).toInt().coerceIn(0, 255)
                    val g = (Color.green(mvpx[p]) * (1 - a) + pg * a + nz).toInt().coerceIn(0, 255)
                    val b = (Color.blue(mvpx[p]) * (1 - a) + pb * a + nz).toInt().coerceIn(0, 255)
                    outPx[p] = Color.rgb(r, g, b)
                }
                // Any hole pixel no in-bounds window covered: fill it structureless from the already-assembled
                // REAL/plate surroundings (never an edge-clamped dark smear; §2-safe - cannot form a person).
                if (anyUncov) {
                    var u = 0
                    for (p in 0 until np) if (uncov[p]) u++
                    ppFrames++
                    val fr = u.toDouble() / max(holePx, 1)
                    ppSum += fr
                    if (fr > ppWorst) ppWorst = fr
                    pushPull(outPx, uncov, w, h)   // pushPull reads `hole` only; uncov is rewritten next frame
                }
                enc.writeFrame(outPx, w, h); written++
                System.arraycopy(outPx, 0, lastOut, 0, np); haveLast = true
                System.arraycopy(mvpx, 0, prevMv, 0, np); havePrev = true
                if (haveFrameShift) { prevDx = frameDx; prevDy = frameDy }
                if (t % 20 == 0 || t == n - 1) log.log("[inpaint] frame $t/$n")
            }
        }
        logPushPull("windowed", ppFrames, written, ppSum, ppWorst, log)
        return holeSum to written
    }

    // =============================================================================================
    // STEP 2d - DYNAMIC (moving camera): HOMOGRAPHY MOSAIC   (FIX6, 2026-08-08)
    //
    // WHY THIS EXISTS. `Aligner` models the camera as a global TRANSLATION. A handheld pan ROTATES.
    // MEASURED on c1_p05_single over N=69 frames (FIX1_INPAINT.md §4.1): translation leaves 21.64
    // grey levels of background residual, affine 21.08, a homography 7.74. The plate is a temporal
    // trimmed mean of ~50 frames each ~21 levels misregistered, so the plate is a smear. Worse, the
    // window plate lives in ONE FRAME-SIZED canvas: at the pan-leading edge `x - dx` runs off it,
    // `accW` stays 0, and the pixel falls through to `pushPull` - a diffusion that by construction
    // cannot contain structure. That wash covered 27-87 % of the hole in EVERY frame, and it is why
    // the owner's door was erased.
    //
    // THE FIX, two halves that only work together:
    //   1. an 8-DOF HOMOGRAPHY per frame, estimated by direct pyramidal Gauss-Newton (Levenberg) on
    //      the SAME grayscale/validity pyramid the app already builds - no feature detector, no
    //      RANSAC, nothing new on the device. MEASURED: mean residual 8.08 / median 6.94 against
    //      SIFT+RANSAC's 7.74 / 7.03, better than SIFT on 59 of 69 frames;
    //   2. a MOSAIC CANVAS sized to the whole camera trajectory, with a recorded integer origin, so
    //      a sample at the pan-leading edge lands on real accumulated pixels instead of out of bounds.
    //
    // MEASURED end to end on c1 (Python reference of THIS algorithm, `kt_hgn.py` + `kt_mosaic.py`,
    // N=70 scored frames): mean |Laplacian| inside the dilated hole 3.09 → 11.20 against a real
    // background of 11.87, i.e. 0.261 → 0.944 of the scene's own detail; the fraction of the hole
    // left to `pushPull` 27-87 % → 0.30 % mean / 4.51 % worst. See _e2e/run3_20260807/FIX6_MOSAIC.md.
    //
    // 🔴 SCOPE. Every line below is reachable ONLY from the `traj.needsWindowing` branch. The STATIC
    // path (`buildStaticPlate` / `compositeFrames`) is not touched, and neither is the previous
    // windowed path, which remains as the fallback when the mosaic cannot be built.
    // ⚠️ NOT VERIFIED ON DEVICE - the phone was off-limits the session this was written.
    // =============================================================================================

    /** Master switch for the mosaic path. DYNAMIC branch only. */
    private const val MOSAIC_ENABLED = true
    /** GN pyramid: (longest-side target, sample stride, max LM iterations, DOF).
     *  The DOF ladder matters: the projective terms are hopelessly ill-conditioned on a 50 px
     *  thumbnail, so the coarsest level solves TRANSLATION only, the next AFFINE, and only the two
     *  finest solve all 8. */
    private val MOSAIC_PYRAMID = arrayOf(
        intArrayOf(50, 1, 20, 2), intArrayOf(100, 1, 16, 6),
        intArrayOf(300, 1, 14, 8), intArrayOf(9999, 3, 8, 8),
    )
    private const val HUBER_T = 8.0            // grey levels; the robust cutoff for the GN residual
    private const val LM_LAMBDA0 = 1e-3
    /** Canvas area cap, x the frame. Beyond this the pan is too wild for one plate -> windowed. */
    private const val MOSAIC_MAX_AREA = 6.0
    /** Consecutive frame pairs used for the grain sigma median (see [grainSigmaWarp]). */
    private const val GRAIN_PAIRS = 5
    private val IDENT9 = doubleArrayOf(1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0)
    private val DOF2 = intArrayOf(2, 5)
    private val DOF6 = intArrayOf(0, 1, 2, 3, 4, 5)
    private val DOF8 = intArrayOf(0, 1, 2, 3, 4, 5, 6, 7)

    // ---- 3x3 helpers (row-major DoubleArray(9)) ----

    private fun mat3Mul(a: DoubleArray, b: DoubleArray): DoubleArray {
        val o = DoubleArray(9)
        for (r in 0..2) for (c in 0..2) {
            var s = 0.0
            for (k in 0..2) s += a[r * 3 + k] * b[k * 3 + c]
            o[r * 3 + c] = s
        }
        return o
    }

    private fun mat3Inv(m: DoubleArray): DoubleArray? {
        val a = m[0]; val b = m[1]; val c = m[2]
        val d = m[3]; val e = m[4]; val f = m[5]
        val g = m[6]; val hh = m[7]; val i = m[8]
        val det = a * (e * i - f * hh) - b * (d * i - f * g) + c * (d * hh - e * g)
        if (!det.isFinite() || abs(det) < 1e-12) return null
        val id = 1.0 / det
        val o = doubleArrayOf(
            (e * i - f * hh) * id, (c * hh - b * i) * id, (b * f - c * e) * id,
            (f * g - d * i) * id, (a * i - c * g) * id, (c * d - a * f) * id,
            (d * hh - e * g) * id, (b * g - a * hh) * id, (a * e - b * d) * id,
        )
        for (v in o) if (!v.isFinite()) return null
        return o
    }

    /**
     * NORMALISED <-> PIXEL homography, for a (w,h) image. The normalisation is
     * `n = ((x - w/2)/S, (y - h/2)/S)` with `S = max(w,h)/2`.
     *
     * WHY THE HOMOGRAPHY IS CARRIED IN NORMALISED COORDS. [dimsFor] scales both axes by the same
     * factor s, so `N_level · D_level == N_native` where `D_level = diag(s, s, 1)`. Therefore
     * `G_norm(level) = N_L D_L G_pix D_L⁻¹ N_L⁻¹ = N G_pix N⁻¹ = G_norm(native)` - the normalised
     * homography is EXACTLY level-invariant, so one 3x3 walks the whole pyramid unchanged and the
     * 8x8 normal equations stay well conditioned (pixel-coordinate homographies are not).
     */
    private fun normToPix(gn: DoubleArray, w: Int, h: Int): DoubleArray {
        val s = max(w, h) / 2.0; val cx = w / 2.0; val cy = h / 2.0
        val nInv = doubleArrayOf(s, 0.0, cx, 0.0, s, cy, 0.0, 0.0, 1.0)
        val nFwd = doubleArrayOf(1 / s, 0.0, -cx / s, 0.0, 1 / s, -cy / s, 0.0, 0.0, 1.0)
        val g = mat3Mul(mat3Mul(nInv, gn), nFwd)
        val d = if (abs(g[8]) > 1e-12) g[8] else 1.0
        for (i in 0..8) g[i] /= d
        return g
    }

    private fun pixToNorm(gp: DoubleArray, w: Int, h: Int): DoubleArray {
        val s = max(w, h) / 2.0; val cx = w / 2.0; val cy = h / 2.0
        val nInv = doubleArrayOf(s, 0.0, cx, 0.0, s, cy, 0.0, 0.0, 1.0)
        val nFwd = doubleArrayOf(1 / s, 0.0, -cx / s, 0.0, 1 / s, -cy / s, 0.0, 0.0, 1.0)
        val g = mat3Mul(mat3Mul(nFwd, gp), nInv)
        val d = if (abs(g[8]) > 1e-12) g[8] else 1.0
        for (i in 0..8) g[i] /= d
        return g
    }

    /** Corner sanity gate: convex, positively oriented, area within [0.25x, 4x] the frame, and no
     *  corner absurdly far out. A homography that fails this is never used - the caller falls back. */
    private fun plausibleH(gn: DoubleArray, w: Int, h: Int): Boolean {
        val g = normToPix(gn, w, h)
        for (v in g) if (!v.isFinite()) return false
        val cxs = doubleArrayOf(0.0, w.toDouble(), w.toDouble(), 0.0)
        val cys = doubleArrayOf(0.0, 0.0, h.toDouble(), h.toDouble())
        val xs = DoubleArray(4); val ys = DoubleArray(4)
        val lim = 6.0 * max(w, h)
        for (k in 0..3) {
            val d = g[6] * cxs[k] + g[7] * cys[k] + g[8]
            if (abs(d) < 1e-9) return false
            xs[k] = (g[0] * cxs[k] + g[1] * cys[k] + g[2]) / d
            ys[k] = (g[3] * cxs[k] + g[4] * cys[k] + g[5]) / d
            if (!xs[k].isFinite() || !ys[k].isFinite()) return false
            if (abs(xs[k]) > lim || abs(ys[k]) > lim) return false
        }
        var area = 0.0
        for (k in 0..3) { val j = (k + 1) and 3; area += xs[k] * ys[j] - xs[j] * ys[k] }
        area *= 0.5
        if (area < 0.25 * w * h || area > 4.0 * w * h) return false
        for (k in 0..3) {
            val b = (k + 1) and 3; val c = (k + 2) and 3
            val cr = (xs[b] - xs[k]) * (ys[c] - ys[k]) - (ys[b] - ys[k]) * (xs[c] - xs[k])
            if (cr <= 0.0) return false
        }
        return true
    }

    private fun bilerpF(a: FloatArray, w: Int, h: Int, fx: Double, fy: Double): Double {
        val x0 = floor(fx).toInt().coerceIn(0, w - 1); val y0 = floor(fy).toInt().coerceIn(0, h - 1)
        val x1 = (x0 + 1).coerceAtMost(w - 1); val y1 = (y0 + 1).coerceAtMost(h - 1)
        val tx = (fx - x0).coerceIn(0.0, 1.0); val ty = (fy - y0).coerceIn(0.0, 1.0)
        val a00 = a[y0 * w + x0].toDouble(); val a10 = a[y0 * w + x1].toDouble()
        val a01 = a[y1 * w + x0].toDouble(); val a11 = a[y1 * w + x1].toDouble()
        return (a00 * (1 - tx) + a10 * tx) * (1 - ty) + (a01 * (1 - tx) + a11 * tx) * ty
    }

    /**
     * HAligner - direct pyramidal Gauss-Newton (Levenberg) estimation of the 8-DOF homography that
     * maps REFERENCE pixel coords onto FRAME pixel coords.
     *
     * The residual is `r(x) = gain · I_frame(W(x; G)) − I_ref(x)` over background-valid pixels, with
     * a Huber weight and a per-evaluation global exposure gain (the same 0.85…1.18 clamp the plate
     * sampler uses). Gradients come from central differences on the frame's own level.
     *
     * 🔴 THE COST HAS A FIXED DENOMINATOR, and that is not a detail. Scoring a warp by the MEAN
     * residual over the pixels that happen to overlap lets a step that SHRINKS the overlap win by
     * throwing away the hard pixels. MEASURED: with a mean cost this estimator scored 10.60 and was
     * WORSE THAN TRANSLATION on 4 of 69 frames; charging every non-overlapping reference pixel a
     * fixed bounded penalty took it to 8.08 with 1 such frame. A direct-alignment cost without an
     * overlap penalty is not a measurement of alignment.
     */
    private class HAligner(private val w: Int, private val h: Int) {

        class Lv(val dw: Int, val dh: Int, val stride: Int, val iters: Int, val dof: Int) {
            val n = dw * dh
            val rG = FloatArray(n); val rV = BooleanArray(n)
            val fG = FloatArray(n); val fV = BooleanArray(n)
            val fGx = FloatArray(n); val fGy = FloatArray(n)
            val px = IntArray(n)
            /** Count of STRIDED reference pixels that are background - the cost's fixed denominator. */
            var nRef = 1
        }

        val levels: List<Lv> = MOSAIC_PYRAMID.map { p ->
            val (dw, dh) = dimsFor(w, h, p[0]); Lv(dw, dh, p[1], p[2], p[3])
        }

        // Per-band accumulators. Bands are indexed by y0/chunk with the SAME chunk formula Par uses,
        // so the reduction below is over a fixed, deterministic set of partial sums (Par.rows may run
        // inline, in which case y0 = 0 and only band 0 is written - still correct).
        private val gainAcc = Array(Par.WORKERS) { DoubleArray(3) }
        private val jAcc = Array(Par.WORKERS) { DoubleArray(46) }

        fun setReference(ref: Bitmap, refPx: IntArray?) {
            for (lv in levels) {
                fill(ref, refPx, lv, true)
                var c = 0
                var y = 0
                while (y < lv.dh) {
                    var x = 0
                    while (x < lv.dw) { if (lv.rV[y * lv.dw + x]) c++; x += lv.stride }
                    y += lv.stride
                }
                lv.nRef = max(1, c)
            }
        }

        private fun fill(frame: Bitmap, framePx: IntArray?, lv: Lv, isRef: Boolean) {
            val dw = lv.dw; val dh = lv.dh; val n = lv.n
            val src: IntArray
            if (frame.width == dw && frame.height == dh) {
                src = if (framePx != null && framePx.size == n) framePx
                      else lv.px.also { frame.getPixels(it, 0, dw, 0, 0, dw, dh) }
            } else {
                val small = Bitmap.createScaledBitmap(frame, dw, dh, true)
                small.getPixels(lv.px, 0, dw, 0, 0, dw, dh)
                if (small !== frame) small.recycle()
                src = lv.px
            }
            val g = if (isRef) lv.rG else lv.fG
            val v = if (isRef) lv.rV else lv.fV
            val fv = fillV; val tol = GRAY_TOL + 2
            Par.rows(dh, dw) { y0, y1 ->
                for (i in y0 * dw until y1 * dw) {
                    val c = src[i]
                    val r = Color.red(c); val gg = Color.green(c); val bb = Color.blue(c)
                    g[i] = ((r + gg + bb) / 3).toFloat()
                    v[i] = !(abs(r - fv) <= tol && abs(gg - fv) <= tol && abs(bb - fv) <= tol)
                }
            }
            if (!isRef) {
                val gx = lv.fGx; val gy = lv.fGy
                Par.rows(dh, dw) { y0, y1 ->
                    for (y in y0 until y1) {
                        val row = y * dw
                        for (x in 0 until dw) {
                            gx[row + x] = if (x >= 1 && x < dw - 1) (g[row + x + 1] - g[row + x - 1]) * 0.5f else 0f
                            gy[row + x] = if (y >= 1 && y < dh - 1) (g[row + dw + x] - g[row - dw + x]) * 0.5f else 0f
                        }
                    }
                }
            }
        }

        private fun bandChunk(dh: Int): Int {
            val bands = min(Par.WORKERS, max(1, dh))
            return (dh + bands - 1) / bands
        }

        /**
         * One robust evaluation of [gn] on [lv]. Returns the cost (MAX_VALUE when the overlap is too
         * small); fills the 8x8 normal matrix [jtj] (row-major, 64) and [jtr] (8) when non-null.
         */
        private fun evalLevel(lv: Lv, gn: DoubleArray, jtj: DoubleArray?, jtr: DoubleArray?): Double {
            val dw = lv.dw; val dh = lv.dh; val st = lv.stride
            val s = max(dw, dh) / 2.0; val cx = dw / 2.0; val cy = dh / 2.0
            val a0 = gn[0]; val a1 = gn[1]; val a2 = gn[2]
            val a3 = gn[3]; val a4 = gn[4]; val a5 = gn[5]
            val a6 = gn[6]; val a7 = gn[7]; val a8 = gn[8]
            val rG = lv.rG; val rV = lv.rV; val fG = lv.fG; val fV = lv.fV
            val chunk = bandChunk(dh)
            // ---- pass A: the global exposure gain over the current overlap ----
            for (b in gainAcc) Arrays.fill(b, 0.0)
            Par.rows(dh, dw) { yy0, yy1 ->
                val acc = gainAcc[min(gainAcc.size - 1, yy0 / chunk)]
                var sr = 0.0; var sf = 0.0; var c = 0.0
                var y = yy0
                if (st > 1 && y % st != 0) y += st - (y % st)
                while (y < yy1) {
                    val ny = (y - cy) / s
                    var x = 0
                    while (x < dw) {
                        val p = y * dw + x
                        if (rV[p]) {
                            val nx = (x - cx) / s
                            val d = a6 * nx + a7 * ny + a8
                            if (abs(d) > 1e-9) {
                                val u = ((a0 * nx + a1 * ny + a2) / d) * s + cx
                                val v = ((a3 * nx + a4 * ny + a5) / d) * s + cy
                                if (u >= 0.0 && u <= dw - 1.0 && v >= 0.0 && v <= dh - 1.0) {
                                    val iu = Math.round(u).toInt().coerceIn(0, dw - 1)
                                    val iv = Math.round(v).toInt().coerceIn(0, dh - 1)
                                    if (fV[iv * dw + iu]) {
                                        sr += rG[p]; sf += bilerpF(fG, dw, dh, u, v); c += 1.0
                                    }
                                }
                            }
                        }
                        x += st
                    }
                    y += st
                }
                acc[0] += sr; acc[1] += sf; acc[2] += c
            }
            var sumR = 0.0; var sumF = 0.0; var cnt0 = 0.0
            for (b in gainAcc) { sumR += b[0]; sumF += b[1]; cnt0 += b[2] }
            if (cnt0 < max(64.0, lv.nRef / 8.0)) return Double.MAX_VALUE
            val gain = if (sumF < 1e-3) 1.0 else (sumR / sumF).coerceIn(0.85, 1.18)
            // ---- pass B: cost + Gauss-Newton normal equations ----
            for (b in jAcc) Arrays.fill(b, 0.0)
            Par.rows(dh, dw) { yy0, yy1 ->
                val acc = jAcc[min(jAcc.size - 1, yy0 / chunk)]
                val j = DoubleArray(8)
                var y = yy0
                if (st > 1 && y % st != 0) y += st - (y % st)
                while (y < yy1) {
                    val ny = (y - cy) / s
                    var x = 0
                    while (x < dw) {
                        val p = y * dw + x
                        if (rV[p]) {
                            val nx = (x - cx) / s
                            val d = a6 * nx + a7 * ny + a8
                            if (abs(d) > 1e-9) {
                                val un = (a0 * nx + a1 * ny + a2) / d
                                val vn = (a3 * nx + a4 * ny + a5) / d
                                val u = un * s + cx; val v = vn * s + cy
                                if (u >= 0.0 && u <= dw - 1.0 && v >= 0.0 && v <= dh - 1.0) {
                                    val iu = Math.round(u).toInt().coerceIn(0, dw - 1)
                                    val iv = Math.round(v).toInt().coerceIn(0, dh - 1)
                                    if (fV[iv * dw + iu]) {
                                        val r = gain * bilerpF(fG, dw, dh, u, v) - rG[p]
                                        val ar = abs(r)
                                        val wgt = if (ar <= HUBER_T) 1.0 else HUBER_T / max(ar, 1e-6)
                                        acc[44] += if (ar <= HUBER_T) 0.5 * r * r else HUBER_T * (ar - 0.5 * HUBER_T)
                                        acc[45] += 1.0
                                        val ix = gain * bilerpF(lv.fGx, dw, dh, u, v)
                                        val iy = gain * bilerpF(lv.fGy, dw, dh, u, v)
                                        val invd = s / d
                                        j[0] = ix * nx * invd; j[1] = ix * ny * invd; j[2] = ix * invd
                                        j[3] = iy * nx * invd; j[4] = iy * ny * invd; j[5] = iy * invd
                                        val q = -(ix * un + iy * vn) * invd
                                        j[6] = q * nx; j[7] = q * ny
                                        var k = 0
                                        for (i2 in 0..7) {
                                            val wi = wgt * j[i2]
                                            for (i3 in i2..7) { acc[k] += wi * j[i3]; k++ }
                                            acc[36 + i2] += wi * r
                                        }
                                    }
                                }
                            }
                        }
                        x += st
                    }
                    y += st
                }
            }
            val tot = DoubleArray(46)
            for (b in jAcc) for (i in 0..45) tot[i] += b[i]
            val cnt = tot[45]
            if (cnt < max(64.0, lv.nRef / 8.0)) return Double.MAX_VALUE
            if (jtj != null && jtr != null) {
                Arrays.fill(jtj, 0.0)
                var k = 0
                for (i2 in 0..7) for (i3 in i2..7) { jtj[i2 * 8 + i3] = tot[k]; jtj[i3 * 8 + i2] = tot[k]; k++ }
                for (i2 in 0..7) jtr[i2] = tot[36 + i2]
            }
            val pen = 2.0 * HUBER_T * HUBER_T   // rho(2.5T): "bad, but bounded"
            return (tot[44] + (lv.nRef - cnt) * pen) / lv.nRef
        }

        /** Damped normal equations over the level's DOF subset; returns an 8-vector (zeros outside
         *  the subset), or null when singular / non-finite. Gaussian elimination, partial pivoting. */
        private fun solveDamped(jtj: DoubleArray, jtr: DoubleArray, idx: IntArray, damp: Double): DoubleArray? {
            val m = idx.size; val w1 = m + 1
            val a = DoubleArray(m * w1)
            for (i in 0 until m) {
                for (j in 0 until m) a[i * w1 + j] = jtj[idx[i] * 8 + idx[j]]
                a[i * w1 + i] += damp
                a[i * w1 + m] = -jtr[idx[i]]
            }
            for (col in 0 until m) {
                var piv = col; var best = abs(a[col * w1 + col])
                for (r in col + 1 until m) { val v = abs(a[r * w1 + col]); if (v > best) { best = v; piv = r } }
                if (!(best > 1e-14)) return null
                if (piv != col) for (k in col..m) {
                    val t = a[col * w1 + k]; a[col * w1 + k] = a[piv * w1 + k]; a[piv * w1 + k] = t
                }
                val d = a[col * w1 + col]
                for (r in col + 1 until m) {
                    val f = a[r * w1 + col] / d
                    if (f == 0.0) continue
                    for (k in col..m) a[r * w1 + k] -= f * a[col * w1 + k]
                }
            }
            val tmp = DoubleArray(m)
            for (i in m - 1 downTo 0) {
                var sm = a[i * w1 + m]
                for (k in i + 1 until m) sm -= a[i * w1 + k] * tmp[k]
                val d = a[i * w1 + i]
                if (!(abs(d) > 1e-14)) return null
                tmp[i] = sm / d
                if (!tmp[i].isFinite()) return null
            }
            val sol = DoubleArray(8)
            for (i in 0 until m) sol[idx[i]] = tmp[i]
            return sol
        }

        /** Levenberg refinement of [gn] (in place) on one level. Returns the final cost. */
        private fun refine(lv: Lv, gn: DoubleArray): Double {
            val jtj = DoubleArray(64); val jtr = DoubleArray(8)
            var cost = evalLevel(lv, gn, jtj, jtr)
            if (cost >= Double.MAX_VALUE) return Double.MAX_VALUE
            val idx = when (lv.dof) { 2 -> DOF2; 6 -> DOF6; else -> DOF8 }
            val cand = DoubleArray(9)
            val jtj2 = DoubleArray(64); val jtr2 = DoubleArray(8)
            var lam = LM_LAMBDA0
            var it = 0
            while (it < lv.iters) {
                var md = 0.0
                for (k in idx) md += jtj[k * 8 + k]
                md = max(md / idx.size, 1e-9)
                var accepted = false
                var maxStep = 0.0
                var tries = 0
                while (tries < 4) {
                    val step = solveDamped(jtj, jtr, idx, lam * md)
                    if (step == null) { lam *= 10.0; tries++; continue }
                    System.arraycopy(gn, 0, cand, 0, 9)
                    for (k in 0..7) cand[k] += step[k]
                    val c2 = evalLevel(lv, cand, jtj2, jtr2)
                    if (c2 < cost) {
                        System.arraycopy(cand, 0, gn, 0, 9)
                        System.arraycopy(jtj2, 0, jtj, 0, 64)
                        System.arraycopy(jtr2, 0, jtr, 0, 8)
                        cost = c2
                        lam = max(lam * 0.3, 1e-6)
                        for (k in 0..7) maxStep = max(maxStep, abs(step[k]))
                        accepted = true
                        break
                    }
                    lam *= 10.0
                    tries++
                }
                if (!accepted || maxStep < 1e-7) break
                it++
            }
            return cost
        }

        /**
         * Estimate the ref->frame NORMALISED homography for [frame].
         *
         * [gnPrev] (the previous frame's pose, ~3 px away) and [gnTraj] (the chained-trajectory
         * translation) are BOTH refined at the coarsest level and the cheaper one wins. That dual
         * init is what stops a single diverged frame poisoning every frame after it: MEASURED on c1,
         * frame 56 still diverges, and frames 57…139 are unaffected.
         */
        fun estimate(frame: Bitmap, framePx: IntArray?, gnPrev: DoubleArray?, gnTraj: DoubleArray): Pair<DoubleArray, Boolean> {
            val l0 = levels[0]
            fill(frame, framePx, l0, false)
            var best: DoubleArray? = null
            var bestCost = Double.MAX_VALUE
            if (gnPrev != null) {
                val g = gnPrev.copyOf()
                val c = refine(l0, g)
                if (c < bestCost && plausibleH(g, w, h)) { best = g; bestCost = c }
            }
            run {
                val g = gnTraj.copyOf()
                val c = refine(l0, g)
                if (c < bestCost && plausibleH(g, w, h)) { best = g; bestCost = c }
            }
            var ok = best != null
            val gn = best ?: gnTraj.copyOf()
            for (i in 1 until levels.size) {
                val lv = levels[i]
                fill(frame, framePx, lv, false)
                val trial = gn.copyOf()
                val c = refine(lv, trial)
                if (c < Double.MAX_VALUE && plausibleH(trial, w, h)) System.arraycopy(trial, 0, gn, 0, 9)
                else ok = false
            }
            if (!plausibleH(gn, w, h)) { System.arraycopy(gnTraj, 0, gn, 0, 9); ok = false }
            return gn to ok
        }
    }

    /**
     * Per-frame sensor-noise sigma with an EXACT SUB-PIXEL homography compensation.
     *
     * 🔴 WHY THIS EXISTS ALONGSIDE [estimateGrainSigma]. That one compensates motion with an INTEGER
     * translation, and on a real pan that is not enough. MEASURED on c1 against the exact per-frame
     * homography: the camera moves 31 px between frames 0 and 1 (direct SAD: dx −31, dy −3), and the
     * integer-compensated estimator returns 6.00 - pinned to `GRAIN_MAX` - at t=1, 2, 3 and 5, while
     * the sub-pixel one returns 2.45, 2.46, 2.05, 2.02 (clip median over 47 pairs: 2.02). Since the
     * composite computes sigma ONCE, at the first frame with a predecessor, a clip whose first pair
     * happens to be fast-moving gets a saturated sigma for its whole length - and feeding sigma 6.0
     * into the composite grain drives the fill 1.94x OVER-sharp (measured: 23.01 against a real
     * background of 11.87). The mosaic path holds an exact homography, so it uses it.
     *
     * [hp] maps CURRENT frame pixel coords to PREVIOUS frame pixel coords. The person is excluded by
     * the fill-colour test rather than the mask, so this needs no extra mask decode; over-excluding a
     * few genuinely grey background pixels cannot move a median taken over ~145 000 samples.
     */
    private fun grainSigmaWarp(cur: IntArray, prev: IntArray, w: Int, h: Int, hp: DoubleArray): Float {
        val bin = IntArray(768); var cnt = 0
        val fv = fillV; val tol = GRAY_TOL + 2
        var y = 0
        while (y < h) {
            var x = 0
            while (x < w) {
                val c = cur[y * w + x]
                val cr = Color.red(c); val cg = Color.green(c); val cb = Color.blue(c)
                if (!(abs(cr - fv) <= tol && abs(cg - fv) <= tol && abs(cb - fv) <= tol)) {
                    val d = hp[6] * x + hp[7] * y + hp[8]
                    if (abs(d) > 1e-9) {
                        val u = (hp[0] * x + hp[1] * y + hp[2]) / d
                        val v = (hp[3] * x + hp[4] * y + hp[5]) / d
                        if (u >= 0.0 && u <= w - 1.0 && v >= 0.0 && v <= h - 1.0) {
                            val q = bilinearRGB(prev, w, h, u.toFloat(), v.toFloat())
                            val qr = Color.red(q); val qg = Color.green(q); val qb = Color.blue(q)
                            if (!(abs(qr - fv) <= tol && abs(qg - fv) <= tol && abs(qb - fv) <= tol)) {
                                bin[min(767, abs(cr - qr) + abs(cg - qg) + abs(cb - qb))]++
                                cnt++
                            }
                        }
                    }
                }
                x += 3
            }
            y += 3
        }
        if (cnt < 500) return 0f
        var acc = 0; var med = 0
        for (v in 0 until 768) { acc += bin[v]; if (acc * 2 >= cnt) { med = v; break } }
        val sigma = (med / 3.0 * 1.0483).toFloat()
        return if (sigma.isFinite()) sigma.coerceIn(GRAIN_MIN, GRAIN_MAX) else 0f
    }

    /**
     * 🔴 FIX9 - THE STATIC PATH'S GRAIN SIGMA. Homography-compensated, and a MEDIAN of pairs.
     *
     * WHAT WAS WRONG. [compositeFrames] computed sigma inline, ONCE, from the FIRST pair it saw,
     * with [estimateGrainSigma]'s INTEGER translation compensation. Three separate defects stack up
     * in that sentence and each of them was measured on the real RUN3 Phase-1 inputs (N = 12
     * consecutive pairs per clip, `_e2e/run3_20260807/fix9_ref/F9_GRAIN.json`):
     *
     *  1. **Integer rounding is not enough** - but neither is sub-pixel TRANSLATION, which is the
     *     obvious cheap fix and which I measured before writing this. On `c3` (devC 39.1 px, a
     *     STATIC-branch clip) the shipped integer estimator pins at `GRAIN_MAX` = 6.00 on **4 of 12
     *     pairs** and sub-pixel translation still pins on **1 of 12**, while the homography pins on
     *     **0 of 12**. Medians: integer 5.07, sub-pixel translation 4.19, homography **3.15** - 
     *     against an INDEPENDENT SIFT+RANSAC homography ground truth of 3.32. A translation cannot
     *     represent a rotating handheld camera at ANY precision, and `devC <= 118.5 px` bounds the
     *     camera's DISPLACEMENT, not its rotation. That is why this reuses [HAligner].
     *  2. **One pair is a lottery.** On `c2` the first pair reads 3.49 where the clip's own median is
     *     2.10 - 1.67x - purely because frame 0->1 happens to be a fast one. The median of
     *     [GRAIN_PAIRS] pairs costs 5 decodes and removes the lottery.
     *  3. **A pinned sigma is invisible.** 6.00 grey levels of injected grain drives the fill 1.94x
     *     OVER-sharp (FIX6 §3.5: |Laplacian| 23.01 against a real background of 11.87) and nothing
     *     in the log said the number was a clamp rather than a measurement. It says so now.
     *
     * STRUCTURE - this is [estimateMosaicHomographies]'s PASS 1, restricted to frames 0..GRAIN_PAIRS.
     * The reference is frame 0, each frame inits from its predecessor's pose (plus the chained
     * trajectory, so one divergence cannot propagate), and the pairwise map is composed exactly as
     * the mosaic composes it: `G(ref->t-1) . H(t->ref)`. No new estimator is introduced.
     *
     * FAILURE IS SAFE. Any frame that does not converge is skipped; if no pair survives this returns
     * 0f and [compositeFrames] adds no grain at all - the pre-2026-08-04 behaviour - and says so.
     */
    private fun staticGrainSigma(
        masked: FrameSource, n: Int, w: Int, h: Int, np: Int, maxDim: Int, traj: Traj, log: Logger,
    ): Float {
        if (n < 2) return 0f
        val ref = masked.frameAt(0, maxDim) ?: return 0f
        val refPx = IntArray(np)
        ref.getPixels(refPx, 0, w, 0, 0, w, h)
        val al = HAligner(w, h)
        al.setReference(ref, refPx)
        val prevPx = IntArray(np); System.arraycopy(refPx, 0, prevPx, 0, np)
        val mvpx = IntArray(np)
        var prevG = IDENT9.copyOf()            // ref -> frame t-1, PIXEL coords
        var prevGn: DoubleArray? = IDENT9.copyOf()
        var havePrev = true
        val sigs = ArrayList<Float>(GRAIN_PAIRS)
        var pinned = 0
        val sh = DoubleArray(2); val sh0 = DoubleArray(2)
        trajAt(traj, 0, sh0)
        var t = 1
        while (t <= GRAIN_PAIRS && t < n) {
            val mv = Prof.time(Prof.DECODE) { masked.frameAt(t, maxDim) } ?: break
            Prof.time(Prof.DECODE) { mv.getPixels(mvpx, 0, w, 0, 0, w, h) }
            trajAt(traj, t, sh)
            val gnTraj = pixToNorm(
                doubleArrayOf(1.0, 0.0, sh[0] - sh0[0], 0.0, 1.0, sh[1] - sh0[1], 0.0, 0.0, 1.0), w, h)
            val res = Prof.time(Prof.ALIGN) { al.estimate(mv, mvpx, prevGn, gnTraj) }
            val gp = normToPix(res.first, w, h)
            val gi = mat3Inv(gp)
            if (gi == null) {
                prevGn = null; havePrev = false
            } else {
                if (havePrev && res.second) {
                    // frame t -> ref (gi), then ref -> frame t-1 (prevG): the SAME composition the
                    // mosaic uses (`mat3Mul(gs[t-1], hs[t])`).
                    val s = grainSigmaWarp(mvpx, prevPx, w, h, mat3Mul(prevG, gi))
                    if (s > 0f) {
                        sigs.add(s)
                        if (s >= GRAIN_MAX - 1e-3f) pinned++
                    }
                }
                prevG = gp; prevGn = res.first
                System.arraycopy(mvpx, 0, prevPx, 0, np)
                havePrev = true
            }
            mv.recycle()
            t++
        }
        if (sigs.isEmpty()) {
            log.log("[inpaint] ⚠ grain sigma UNMEASURABLE (0 of ${t - 1} pairs converged) -> the fill " +
                "gets NO temporal grain on this clip; it will read as a frozen patch inside live video")
            return 0f
        }
        sigs.sort()
        val sigma = sigs[sigs.size / 2]
        log.log("[inpaint] temporal grain sigma = ${"%.2f".format(sigma)} grey levels " +
            "(median of ${sigs.size} homography-compensated pairs; first pair " +
            "${"%.2f".format(sigs.first())}..${"%.2f".format(sigs.last())} range)")
        if (pinned > 0) log.log("[inpaint] ⚠ SATURATED: the grain estimate hit the GRAIN_MAX=" +
            "${"%.1f".format(GRAIN_MAX)} clamp on $pinned/${sigs.size} pairs - a pinned sigma is a " +
            "FLOOR, not a measurement, and it over-sharpens the fill (see FIX9_CLAMPS.md)")
        if (sigma >= GRAIN_MAX - 1e-3f) log.log("[inpaint] ⚠ the MEDIAN pair is pinned too, so the " +
            "sigma actually used is the clamp itself - treat this run's fill texture as a defect")
        return sigma
    }

    /** Homographies + the clip's own grain sigma, produced by the mosaic path's PASS 1. */
    private class MosaicGeom(
        val gs: Array<DoubleArray>,   // ref -> frame t, PIXEL coords
        val hs: Array<DoubleArray>,   // frame t -> ref, PIXEL coords (== gs⁻¹)
        val converged: Int,
        val grainSigma: Float,
        val refMeanLuma: Double,
        val refT: Int,
    )

    /** The mosaic plate itself: one canvas in REF coords with a recorded integer origin, so mosaic
     *  pixel (X,Y) is ref pixel (X + ox, Y + oy). */
    private class MosaicPlate(
        val plate: IntArray, val cov: ByteArray, val core: BooleanArray, val union: BooleanArray,
        val mw: Int, val mh: Int, val ox: Int, val oy: Int, val samples: Int, val geom: MosaicGeom,
    )

    /**
     * PASS 1 - estimate the per-frame homography for the WHOLE clip, and the grain sigma.
     *
     * Walks the clip in ASCENDING order (MediaMetadataRetriever backwards seeks are ruinous), which
     * means every frame's init is its immediate predecessor's pose except frame 0's, which comes from
     * the chained trajectory. The canvas cannot be sized until every homography is known, so this is
     * a pass of its own; the mosaic therefore costs three passes over the clip against the windowed
     * path's K+1. ⚠️ The latency of that is NOT MEASURED.
     */
    private fun estimateMosaicHomographies(
        masked: FrameSource, mask: FrameSource?, n: Int, w: Int, h: Int, np: Int, maxDim: Int,
        traj: Traj, log: Logger,
    ): MosaicGeom? {
        val refT = n / 2
        val refFrame = masked.frameAt(refT, maxDim) ?: return null
        val refPx = IntArray(np)
        refFrame.getPixels(refPx, 0, w, 0, 0, w, h)
        val al = HAligner(w, h)
        al.setReference(refFrame, refPx)
        val refMeanLuma = referenceMeanLuma(refFrame, mask, refT, maxDim, w, h, np)

        val gs = Array(n) { IDENT9.copyOf() }
        val hs = Array(n) { IDENT9.copyOf() }
        val mvpx = IntArray(np); val prevPx = IntArray(np)
        var havePrev = false
        var prevGn: DoubleArray? = null
        var converged = 0
        val sh = DoubleArray(2); val shRef = DoubleArray(2)
        trajAt(traj, refT, shRef)
        val sigs = ArrayList<Float>(GRAIN_PAIRS)
        masked.startPrefetch(n, maxDim)
        for (t in 0 until n) {
            val mv = Prof.time(Prof.DECODE) { masked.frameAt(t, maxDim) }
            if (mv == null) {
                // An undecodable frame must NOT be left with the identity: on a 456 px pan that
                // would make the composite sample the wrong end of the mosaic for that frame.
                // Inherit its predecessor's pose, which is ~3 px away, and note it.
                if (t > 0) {
                    System.arraycopy(gs[t - 1], 0, gs[t], 0, 9)
                    System.arraycopy(hs[t - 1], 0, hs[t], 0, 9)
                }
                havePrev = false     // the next pair is no longer consecutive
                continue
            }
            Prof.time(Prof.DECODE) { mv.getPixels(mvpx, 0, w, 0, 0, w, h) }
            val gn: DoubleArray
            if (t == refT) {
                gn = IDENT9.copyOf(); converged++
            } else {
                trajAt(traj, t, sh)
                val gnTraj = pixToNorm(
                    doubleArrayOf(1.0, 0.0, sh[0] - shRef[0], 0.0, 1.0, sh[1] - shRef[1], 0.0, 0.0, 1.0), w, h)
                val res = Prof.time(Prof.ALIGN) { al.estimate(mv, mvpx, prevGn, gnTraj) }
                gn = res.first
                if (res.second) converged++
            }
            prevGn = gn
            val gp = normToPix(gn, w, h)
            val gi = mat3Inv(gp)
            if (gi == null) {
                // Not invertible - the corner gate should have caught this, so treat it as the same
                // failure mode as a bad decode rather than shipping a half-consistent pair.
                if (t > 0) {
                    System.arraycopy(gs[t - 1], 0, gs[t], 0, 9)
                    System.arraycopy(hs[t - 1], 0, hs[t], 0, 9)
                }
                prevGn = null
                havePrev = false
                mv.recycle()
                continue
            }
            System.arraycopy(gp, 0, gs[t], 0, 9)
            System.arraycopy(gi, 0, hs[t], 0, 9)
            if (havePrev && sigs.size < GRAIN_PAIRS) {
                val s = grainSigmaWarp(mvpx, prevPx, w, h, mat3Mul(gs[t - 1], hs[t]))
                if (s > 0f) sigs.add(s)
            }
            System.arraycopy(mvpx, 0, prevPx, 0, np); havePrev = true
            // This pass keeps nothing from the frame but its pixels, and it runs immediately before a
            // ~230 MB sample-stack allocation, so hand the 6.4 MB bitmap back now rather than leaving
            // 140 of them for the GC to find under memory pressure.
            mv.recycle()
            if (t % 20 == 0 || t == n - 1) log.log("[inpaint] homography $t/$n")
        }
        sigs.sort()
        val sigma = if (sigs.isEmpty()) 0f else sigs[sigs.size / 2]
        log.log("[inpaint] homography: $converged/$n frames converged; " +
            "grain sigma = ${"%.2f".format(sigma)} grey levels " +
            "(median of ${sigs.size} sub-pixel-compensated pairs)")
        if (sigma >= GRAIN_MAX - 1e-3f) log.log("[inpaint] ⚠ SATURATED: the mosaic's grain sigma is " +
            "PINNED at GRAIN_MAX=${"%.1f".format(GRAIN_MAX)} - even the exact homography could not " +
            "separate noise from motion on this clip, so the fill will be over-sharp (FIX9_CLAMPS.md)")
        if (converged < n / 2) {
            log.log("[inpaint] ⚠ fewer than half the frames converged -> abandoning the mosaic")
            return null
        }
        return MosaicGeom(gs, hs, converged, sigma, refMeanLuma, refT)
    }

    /**
     * PASS 2 - size the canvas to the camera trajectory and accumulate the trimmed-mean plate into it.
     *
     * Structurally this is [aggregatePlate] with two substitutions: the destination is the mosaic
     * rather than one frame, and the ref->frame map is a homography rather than (dx, dy). It is a
     * COPY rather than a generalisation of aggregatePlate on purpose - aggregatePlate is on the
     * STATIC path, whose bit-identity is load-bearing (§C.PHONE-9v), and the safest way to leave it
     * bit-identical is not to touch it.
     */
    private fun buildMosaicPlate(
        masked: FrameSource, mask: FrameSource?, n: Int, w: Int, h: Int, np: Int, maxDim: Int,
        traj: Traj, getLama: () -> LamaInpainter?, log: Logger,
    ): MosaicPlate? {
        val geom = estimateMosaicHomographies(masked, mask, n, w, h, np, maxDim, traj, log) ?: return null
        // ---- canvas: the bbox of every frame's quad mapped into REF coords ----
        var minX = Double.MAX_VALUE; var maxX = -Double.MAX_VALUE
        var minY = Double.MAX_VALUE; var maxY = -Double.MAX_VALUE
        val cxs = doubleArrayOf(0.0, w.toDouble(), w.toDouble(), 0.0)
        val cys = doubleArrayOf(0.0, 0.0, h.toDouble(), h.toDouble())
        for (t in 0 until n) {
            val hh = geom.hs[t]
            for (k in 0..3) {
                val d = hh[6] * cxs[k] + hh[7] * cys[k] + hh[8]
                if (abs(d) < 1e-9) continue
                val x = (hh[0] * cxs[k] + hh[1] * cys[k] + hh[2]) / d
                val y = (hh[3] * cxs[k] + hh[4] * cys[k] + hh[5]) / d
                if (!x.isFinite() || !y.isFinite()) continue
                if (x < minX) minX = x
                if (x > maxX) maxX = x
                if (y < minY) minY = y
                if (y > maxY) maxY = y
            }
        }
        if (minX > maxX || minY > maxY) return null
        val ox = floor(minX).toInt(); val oy = floor(minY).toInt()
        val mw = (kotlin.math.ceil(maxX).toInt() - ox).coerceAtLeast(w)
        val mh = (kotlin.math.ceil(maxY).toInt() - oy).coerceAtLeast(h)
        val npM = mw.toLong() * mh
        if (npM > MOSAIC_MAX_AREA * np) {
            log.log("[inpaint] ⚠ mosaic canvas would be ${"%.1f".format(npM.toDouble() / np)}x the frame " +
                "(cap ${MOSAIC_MAX_AREA}x) -> falling back to the windowed path")
            return null
        }
        val npm = npM.toInt()
        val used = plateSampleCount(n, npm, mw, mh)
        log.log("[inpaint] plate budget (mosaic ${mw}x${mh}): $lastSampleBudget")
        val frames = when {
            used >= n -> (0 until n).toList()
            used <= 1 -> listOf(n / 2)
            else -> List(used) { (it.toLong() * (n - 1) / (used - 1L)).toInt() }
        }
        log.log("[inpaint] mosaic canvas ${mw}x$mh = ${"%.2f".format(npM.toDouble() / np)}x frame, " +
            "origin ($ox,$oy); ${frames.size} samples spread over 0..${frames.last()} of $n")

        val s = frames.size
        val wpr = validWordsPerRow(mw)
        val vPer = wpr * mh
        val samp = ByteArray(s * npm * 3)
        val valid = LongArray(s * vPer)
        val union = BooleanArray(npm)
        val mvpx = IntArray(np); val mkpx = IntArray(np); val hole = BooleanArray(np)
        val tMax = frames.last()
        val idxOf = IntArray(tMax + 1) { -1 }
        frames.forEachIndexed { si, t -> if (t in 0..tMax) idxOf[t] = si }
        masked.startPrefetch(tMax + 1, maxDim)
        for (t in 0..tMax) {
            val mv = Prof.time(Prof.DECODE) { masked.frameAt(t, maxDim) } ?: continue
            val si = idxOf[t]
            if (si < 0) { mv.recycle(); continue }
            Prof.time(Prof.DECODE) { mv.getPixels(mvpx, 0, w, 0, 0, w, h) }
            Prof.time(Prof.MASK) {
                fillMaskPixels(mask, t, maxDim, w, h, mvpx, mkpx)
                maskToBoolInto(mkpx, np, hole)
            }
            dilateInPlace(hole, w, h, MASK_DILATE)
            val g = geom.gs[t]
            val gain = (geom.refMeanLuma / meanLuma(mvpx, hole, np)).coerceIn(0.85, 1.18)
            // only the rows/cols this frame's quad can reach
            var qx0 = mw; var qx1 = -1; var qy0 = mh; var qy1 = -1
            val hh = geom.hs[t]
            for (k in 0..3) {
                val d = hh[6] * cxs[k] + hh[7] * cys[k] + hh[8]
                if (abs(d) < 1e-9) { qx0 = 0; qx1 = mw - 1; qy0 = 0; qy1 = mh - 1; break }
                val x = (hh[0] * cxs[k] + hh[1] * cys[k] + hh[2]) / d - ox
                val y = (hh[3] * cxs[k] + hh[4] * cys[k] + hh[5]) / d - oy
                qx0 = min(qx0, floor(x).toInt() - 1); qx1 = max(qx1, kotlin.math.ceil(x).toInt() + 1)
                qy0 = min(qy0, floor(y).toInt() - 1); qy1 = max(qy1, kotlin.math.ceil(y).toInt() + 1)
            }
            qx0 = qx0.coerceIn(0, mw - 1); qx1 = qx1.coerceIn(0, mw - 1)
            qy0 = qy0.coerceIn(0, mh - 1); qy1 = qy1.coerceIn(0, mh - 1)
            val base = si * npm
            val vbase = si * vPer
            Prof.time(Prof.SAMPLE) {
                // Rows are independent: every write lands in this mosaic pixel's own slot, and
                // union[p] = true is idempotent.
                Par.rows(qy1 - qy0 + 1, mw) { b0, b1 ->
                    for (yy in (qy0 + b0)..(qy0 + b1 - 1)) {
                        val ry = (yy + oy).toDouble()
                        val rowOff = yy * mw
                        for (xx in qx0..qx1) {
                            val rx = (xx + ox).toDouble()
                            val d = g[6] * rx + g[7] * ry + g[8]
                            if (abs(d) < 1e-9) continue
                            val fx = (g[0] * rx + g[1] * ry + g[2]) / d
                            val fy = (g[3] * rx + g[4] * ry + g[5]) / d
                            if (!(fx >= -0.5 && fx <= w - 0.5 && fy >= -0.5 && fy <= h - 0.5)) continue
                            val ix = Math.round(fx).toInt().coerceIn(0, w - 1)
                            val iy = Math.round(fy).toInt().coerceIn(0, h - 1)
                            val p = rowOff + xx
                            if (hole[iy * w + ix]) { union[p] = true; continue }
                            val c = bilinearRGB(mvpx, w, h, fx.toFloat(), fy.toFloat())
                            val vi = vbase + yy * wpr + (xx ushr 6)
                            valid[vi] = valid[vi] or (1L shl (xx and 63))
                            val o = (base + p) * 3
                            samp[o] = ((Color.red(c) * gain).toInt().coerceIn(0, 255)).toByte()
                            samp[o + 1] = ((Color.green(c) * gain).toInt().coerceIn(0, 255)).toByte()
                            samp[o + 2] = ((Color.blue(c) * gain).toInt().coerceIn(0, 255)).toByte()
                        }
                    }
                }
            }
            if (si % 5 == 0) log.log("[inpaint] mosaic sampled $si/$s")
        }

        val plate = IntArray(npm); val core = BooleanArray(npm); val cov = ByteArray(npm)
        val minCov = max(3, (0.05 * s).toInt())
        log.log("[inpaint] aggregating ${npm / 1000}k mosaic px x $s samples on ${Par.WORKERS} worker(s) …")
        Prof.time(Prof.GATHER) {
            Par.rows(mh, mw) { y0, y1 ->
                val rb = IntArray(s); val gb = IntArray(s); val bb = IntArray(s); val bin = IntArray(256)
                for (y in y0 until y1) for (x in 0 until mw) {
                    val p = y * mw + x
                    val vOff = y * wpr + (x ushr 6)
                    val vBit = 1L shl (x and 63)
                    var k = 0
                    for (si in 0 until s) if (valid[si * vPer + vOff] and vBit != 0L) {
                        val o = (si * npm + p) * 3
                        rb[k] = samp[o].toInt() and 0xFF
                        gb[k] = samp[o + 1].toInt() and 0xFF
                        bb[k] = samp[o + 2].toInt() and 0xFF
                        k++
                    }
                    cov[p] = min(255, k).toByte()
                    when {
                        k >= minCov -> plate[p] = Color.rgb(robustCenter(rb, k, bin), robustCenter(gb, k, bin), robustCenter(bb, k, bin))
                        k > 0 -> { if (union[p]) core[p] = true; plate[p] = Color.rgb(robustCenter(rb, k, bin), robustCenter(gb, k, bin), robustCenter(bb, k, bin)) }
                        // NOTE the difference from aggregatePlate: a never-covered pixel is only a
                        // NEURAL CORE when some frame's hole actually landed on it. The mosaic has
                        // corners that lie outside every frame's quad; marking those as core would
                        // hand LaMa ~46 % of the canvas to hallucinate for no reader at all.
                        else -> { core[p] = union[p]; plate[p] = Color.BLACK }
                    }
                }
            }
        }
        val corePx = core.count { it }
        log.log("[inpaint] mosaic: core $corePx px (${"%.2f".format(100.0 * corePx / npm)}% of canvas)")
        if (corePx > 0) fillCore(plate, core, mw, mh, npm, getLama, log)
        return MosaicPlate(plate, cov, core, union, mw, mh, ox, oy, s, geom)
    }

    /**
     * PASS 3 - composite the mosaic into every frame through that frame's own homography.
     *
     * The plate sample for frame pixel (x, y) is the mosaic at `H_t · (x, y) − origin`, so the
     * pan-leading edge reads REAL accumulated pixels instead of running off a frame-sized buffer.
     * [pushPull] survives as the last resort it was always meant to be, and its fire rate is LOGGED
     * so a future regression is visible instead of silent.
     */
    /**
     * FIX12 - the donor ring: the last `2*LOCAL_RADIUS+1` decoded masked frames.
     *
     * WHY A RING AND NOT RANDOM ACCESS. `FrameSource.frameAt` only pipelines SEQUENTIAL reads; asking
     * for any other index retires the prefetcher and pays a full `MediaMetadataRetriever` seek, a cost
     * this project has already been bitten by. So the composite loop runs `LOCAL_RADIUS` frames BEHIND
     * a single sequential decode head, which makes the donor window TWO-SIDED for free - donors on
     * both sides matter, because the hole at frame t is revealed by frames before AND after it.
     *
     * Holes are kept as bitsets (two of them) rather than `BooleanArray`s: 17 frames of
     * `BooleanArray(1.6 M)` would be 27 MB where the bitsets are 3.4 MB. `raw` is the undilated mask
     * the §2 audit and the composite need; `dil` is dilated by [MASK_DILATE] exactly as the mosaic
     * sampler does, so a donor can never contribute its own person's edge to someone else's fill.
     */
    private class LocalRing(val cap: Int, np: Int) {
        val idx = IntArray(cap) { Int.MIN_VALUE }
        val px = Array(cap) { IntArray(np) }
        val raw = Array(cap) { LongArray((np + 63) ushr 6) }
        val dil = Array(cap) { LongArray((np + 63) ushr 6) }
        val gain = FloatArray(cap)
        fun slot(t: Int): Int { val m = t % cap; return if (m < 0) m + cap else m }
        fun has(t: Int): Boolean = t >= 0 && idx[slot(t)] == t
        fun isDil(s: Int, p: Int): Boolean = (dil[s][p ushr 6] ushr (p and 63)) and 1L != 0L
        fun isRaw(s: Int, p: Int): Boolean = (raw[s][p ushr 6] ushr (p and 63)) and 1L != 0L
    }

    /**
     * FIX12 - replace the hole's fill with NEAREST-FIRST samples wherever enough of them exist.
     *
     * For each hole pixel: map it to ref coords with `hs[t]`, then walk donors by increasing |d − t|,
     * mapping ref → donor with `gs[d]`, and take the first [LOCAL_K] that see REAL background there.
     * This is the whole fix. It needs no new alignment: composing `canvas→t` with `d→canvas` was
     * measured to cost **−0.04 grey levels** against estimating `d→t` directly, so `geom` already
     * carries everything, and only the CHOICE of which samples to mix changes.
     *
     * Pixels with fewer than [LOCAL_MIN] samples are LEFT ALONE - they keep the mosaic plate's value,
     * which is what makes this strictly non-regressive: the door, the pushPull floor and the coverage
     * guarantee all still come from the mosaic.
     *
     * The final DC match exists because the local and mosaic fills are two different estimators of the
     * same background; without it their boundary could show a brightness step even though each is
     * individually correct. It is a single offset per frame, so it cannot distort structure.
     *
     * @return the number of pixels actually overridden.
     */
    private fun localSharpen(
        ring: LocalRing, t: Int, w: Int, h: Int, np: Int, geom: MosaicGeom,
        hole: BooleanArray, ablur: FloatArray, warp: IntArray, lpx: IntArray, lok: BooleanArray,
    ): Int {
        var oc = 0
        // The EFFECTIVE radius is the ring's, not [LOCAL_RADIUS]: [ringRadiusFor] may have capped the
        // slider's window to fit the heap, and walking past the ring's capacity would only ask
        // `ring.has` about frames it structurally cannot hold.
        val rad = (ring.cap - 1) / 2
        val order = IntArray(max(1, 2 * rad))
        for (r in 1..rad) {                                // nearest first, alternating either side
            if (ring.has(t - r)) order[oc++] = t - r
            if (ring.has(t + r)) order[oc++] = t + r
        }
        if (oc < LOCAL_MIN) return 0
        val slots = IntArray(oc) { ring.slot(order[it]) }
        val gains = FloatArray(oc) { ring.gain[slots[it]] }
        val hm = geom.hs[t]
        Par.rows(h, w) { y0, y1 ->
            val rb = IntArray(LOCAL_K); val gb = IntArray(LOCAL_K); val bb = IntArray(LOCAL_K)
            val bin = IntArray(256)
            for (y in y0 until y1) {
                val row = y * w
                val yd = y.toDouble()
                for (x in 0 until w) {
                    val p = row + x
                    lok[p] = false
                    if (!hole[p] || ablur[p] <= 0f) continue
                    val xd = x.toDouble()
                    val dd = hm[6] * xd + hm[7] * yd + hm[8]
                    if (abs(dd) < 1e-9) continue
                    val rx = (hm[0] * xd + hm[1] * yd + hm[2]) / dd
                    val ry = (hm[3] * xd + hm[4] * yd + hm[5]) / dd
                    var k = 0; var i = 0
                    while (i < oc && k < LOCAL_K) {
                        val g = geom.gs[order[i]]
                        val d2 = g[6] * rx + g[7] * ry + g[8]
                        if (abs(d2) > 1e-9) {
                            val fx = (g[0] * rx + g[1] * ry + g[2]) / d2
                            val fy = (g[3] * rx + g[4] * ry + g[5]) / d2
                            if (fx >= -0.5 && fx <= w - 0.5 && fy >= -0.5 && fy <= h - 0.5) {
                                val ix = Math.round(fx).toInt().coerceIn(0, w - 1)
                                val iy = Math.round(fy).toInt().coerceIn(0, h - 1)
                                val s = slots[i]
                                if (!ring.isDil(s, iy * w + ix)) {
                                    val c = bilinearRGB(ring.px[s], w, h, fx.toFloat(), fy.toFloat())
                                    val gn = gains[i]
                                    rb[k] = (Color.red(c) * gn).toInt().coerceIn(0, 255)
                                    gb[k] = (Color.green(c) * gn).toInt().coerceIn(0, 255)
                                    bb[k] = (Color.blue(c) * gn).toInt().coerceIn(0, 255)
                                    k++
                                }
                            }
                        }
                        i++
                    }
                    if (k >= LOCAL_MIN) {
                        lpx[p] = Color.rgb(robustCenter(rb, k, bin), robustCenter(gb, k, bin),
                            robustCenter(bb, k, bin))
                        lok[p] = true
                    }
                }
            }
        }
        var cnt = 0L; var dr = 0L; var dg = 0L; var db = 0L
        for (p in 0 until np) if (lok[p]) {
            cnt++
            dr += (Color.red(warp[p]) - Color.red(lpx[p])).toLong()
            dg += (Color.green(warp[p]) - Color.green(lpx[p])).toLong()
            db += (Color.blue(warp[p]) - Color.blue(lpx[p])).toLong()
        }
        if (cnt == 0L) return 0
        val or0 = Math.round(dr.toDouble() / cnt).toInt()
        val og0 = Math.round(dg.toDouble() / cnt).toInt()
        val ob0 = Math.round(db.toDouble() / cnt).toInt()
        for (p in 0 until np) if (lok[p]) {
            warp[p] = Color.rgb((Color.red(lpx[p]) + or0).coerceIn(0, 255),
                (Color.green(lpx[p]) + og0).coerceIn(0, 255),
                (Color.blue(lpx[p]) + ob0).coerceIn(0, 255))
        }
        return cnt.toInt()
    }

    private fun compositeMosaic(
        masked: FrameSource, mask: FrameSource?, n: Int, w: Int, h: Int, np: Int, maxDim: Int,
        mos: MosaicPlate, out: File, log: Logger,
    ): Pair<Long, Int> {
        log.log("[inpaint] compositing the ${mos.mw}x${mos.mh} mosaic -> ${out.name}")
        val mkpx = IntArray(np); val outPx = IntArray(np)
        val warp = IntArray(np); val alpha = FloatArray(np); val ablur = FloatArray(np)
        // Seam-match scratch (null when the slider is at 0, so the old path allocates nothing).
        val (seamBand, seamRes, seamUnk) = seamScratch(np)
        val hole = BooleanArray(np); val uncov = BooleanArray(np)
        val lastOut = IntArray(np); var haveLast = false
        val grainSigma = mos.geom.grainSigma
        val mw = mos.mw; val mh = mos.mh; val ox = mos.ox; val oy = mos.oy
        var holeSum = 0L; var written = 0
        var ppFrames = 0; var ppSum = 0.0; var ppWorst = 0.0
        // FIX12 - donors are served from a ring filled by ONE sequential decode head, and the emit
        // loop trails that head by `rad` frames so the donor window is two-sided at no decode cost.
        // rad = 0 (LOCAL_PLATE off) degenerates to a 1-slot ring, i.e. the previous behaviour.
        val ring = newRing(np, ringRadiusFor(np), log)
        val rad = if (LOCAL_PLATE) (ring.cap - 1) / 2 else 0
        if (LOCAL_PLATE) log.log("[inpaint] fill sharpness ${describeFillSharpness()}" +
            (if (rad != LOCAL_RADIUS) " - window CAPPED to ±$rad by the heap" else ""))
        log.log("[inpaint] seam match ${describeSeamMatch()}")
        val lpx = if (LOCAL_PLATE) IntArray(np) else IntArray(0)
        val lok = if (LOCAL_PLATE) BooleanArray(np) else BooleanArray(0)
        var locSum = 0L; var locFrames = 0
        masked.startPrefetch(n, maxDim)
        mask?.startPrefetch(n, maxDim)
        Mp4Encoder(out, masked.srcFps).use { enc ->
            for (dec in 0 until n + rad) {
                if (dec < n) {
                    val bm = Prof.time(Prof.DECODE) { masked.frameAt(dec, maxDim) }
                    if (bm != null) {
                        val s = ring.slot(dec)
                        Prof.time(Prof.DECODE) { bm.getPixels(ring.px[s], 0, w, 0, 0, w, h) }
                        Prof.time(Prof.MASK) {
                            fillMaskPixels(mask, dec, maxDim, w, h, ring.px[s], mkpx)
                            maskToBoolInto(mkpx, np, hole)
                        }
                        val rw = ring.raw[s]; java.util.Arrays.fill(rw, 0L)
                        for (p in 0 until np) if (hole[p]) rw[p ushr 6] = rw[p ushr 6] or (1L shl (p and 63))
                        // the DILATED copy is what a donor is tested against, matching the mosaic
                        // sampler, so a donor can never contribute its own person's edge to a fill
                        dilateInPlace(hole, w, h, MASK_DILATE)
                        val dl = ring.dil[s]; java.util.Arrays.fill(dl, 0L)
                        for (p in 0 until np) if (hole[p]) dl[p ushr 6] = dl[p ushr 6] or (1L shl (p and 63))
                        ring.gain[s] = (mos.geom.refMeanLuma / meanLuma(ring.px[s], hole, np))
                            .coerceIn(0.85, 1.18).toFloat()
                        ring.idx[s] = dec
                    }
                }
                val t = dec - rad
                if (t < 0) continue
                if (!ring.has(t)) {
                    if (haveLast) { Prof.time(Prof.ENCODE) { enc.writeFrame(lastOut, w, h) }; written++ }
                    continue
                }
                val ts = ring.slot(t)
                val mvpx = ring.px[ts]
                Prof.time(Prof.MASK) {
                    // the mask was decoded once, when this frame entered the ring; re-reading it here
                    // would be an out-of-order read that retires the prefetcher
                    for (p in 0 until np) hole[p] = ring.isRaw(ts, p)
                    // §2 audit BEFORE the dilation. The windowed path never did this, which is why a
                    // staged-mask DYNAMIC run reported "audit DID NOT RUN"; the mosaic path audits
                    // exactly as compositeFrames does.
                    auditStagedMask(mvpx, hole, t)
                }
                dilateInPlace(hole, w, h, COMPOSITE_DILATE)
                val hm = mos.geom.hs[t]
                var holePx = 0
                Prof.time(Prof.COMPOSITE) {
                    for (p in 0 until np) { if (hole[p]) holePx++; alpha[p] = if (hole[p]) 1f else 0f }
                    boxBlur(alpha, ablur, w, h, FEATHER)
                    // Warp only what something reads: the blend reads ablur > 0, ringGain samples at
                    // stride 4 - the same predicate compositeFrames uses.
                    Par.rows(h, w) { y0, y1 ->
                        for (y in y0 until y1) {
                            val row = y * w
                            val yd = y.toDouble()
                            for (x in 0 until w) {
                                val p = row + x
                                if (!(ablur[p] > 0f || (p and 3) == 0)) continue
                                val xd = x.toDouble()
                                val d = hm[6] * xd + hm[7] * yd + hm[8]
                                var covered = false
                                if (abs(d) > 1e-9) {
                                    val mxf = (hm[0] * xd + hm[1] * yd + hm[2]) / d - ox
                                    val myf = (hm[3] * xd + hm[4] * yd + hm[5]) / d - oy
                                    if (mxf >= -0.5 && mxf <= mw - 0.5 && myf >= -0.5 && myf <= mh - 0.5) {
                                        val ix = Math.round(mxf).toInt().coerceIn(0, mw - 1)
                                        val iy = Math.round(myf).toInt().coerceIn(0, mh - 1)
                                        if ((mos.cov[iy * mw + ix].toInt() and 0xFF) >= 1) {
                                            warp[p] = bilinearRGB(mos.plate, mw, mh, mxf.toFloat(), myf.toFloat())
                                            covered = true
                                        }
                                    }
                                }
                                if (!covered) warp[p] = mvpx[p]   // keeps ringGain unbiased
                                uncov[p] = hole[p] && !covered
                            }
                        }
                    }
                    // FIX12 - override the mosaic's soft fill with nearest-first local samples
                    // wherever at least LOCAL_MIN of them exist. Pixels it cannot serve keep the
                    // mosaic value, so nothing this touches can be worse than before. A pixel the
                    // mosaic left UNCOVERED but the ring can serve is now real background, so it also
                    // comes off the pushPull list instead of being diffused.
                    if (LOCAL_PLATE) {
                        val nLoc = localSharpen(ring, t, w, h, np, mos.geom, hole, ablur, warp, lpx, lok)
                        if (nLoc > 0) {
                            locSum += nLoc; locFrames++
                            for (p in 0 until np) if (lok[p]) uncov[p] = false
                        }
                    }
                    val gain = matchSurroundings(mvpx, warp, hole, ablur, w, h, np, seamBand, seamRes, seamUnk)
                    Par.range(np) { p0, p1 ->
                        for (p in p0 until p1) {
                            val a = ablur[p]
                            if (a <= 0f) { outPx[p] = mvpx[p]; continue }
                            val pr = (Color.red(warp[p]) * gain[0]).toInt().coerceIn(0, 255)
                            val pg = (Color.green(warp[p]) * gain[1]).toInt().coerceIn(0, 255)
                            val pb = (Color.blue(warp[p]) * gain[2]).toInt().coerceIn(0, 255)
                            val nz = if (grainSigma > 0f) grainAt(p, t) * grainSigma * a else 0f
                            if (a >= 1f) {
                                outPx[p] = Color.rgb((pr + nz).toInt().coerceIn(0, 255),
                                    (pg + nz).toInt().coerceIn(0, 255), (pb + nz).toInt().coerceIn(0, 255))
                                continue
                            }
                            val r = (Color.red(mvpx[p]) * (1 - a) + pr * a + nz).toInt().coerceIn(0, 255)
                            val g = (Color.green(mvpx[p]) * (1 - a) + pg * a + nz).toInt().coerceIn(0, 255)
                            val b = (Color.blue(mvpx[p]) * (1 - a) + pb * a + nz).toInt().coerceIn(0, 255)
                            outPx[p] = Color.rgb(r, g, b)
                        }
                    }
                    var u = 0
                    for (p in 0 until np) {
                        if (ablur[p] <= 0f) uncov[p] = false      // outside the touched region: never diffuse
                        if (uncov[p]) u++
                    }
                    if (u > 0) {
                        ppFrames++
                        val fr = u.toDouble() / max(holePx, 1)
                        ppSum += fr
                        if (fr > ppWorst) ppWorst = fr
                        pushPull(outPx, uncov, w, h)
                    }
                }
                holeSum += holePx
                Prof.time(Prof.ENCODE) { enc.writeFrame(outPx, w, h) }
                written++
                System.arraycopy(outPx, 0, lastOut, 0, np); haveLast = true
                if (t % 20 == 0 || t == n - 1) log.log("[inpaint] frame $t/$n")
            }
        }
        // FIX12 tripwire. The share of the hole served by SHARP local samples is the whole point of
        // this change, so it is logged rather than inferred: if it reads 0 %, the fix did not run and
        // any quality claim about the output is void.
        if (LOCAL_PLATE) {
            val pct = if (holeSum > 0) 100.0 * locSum / holeSum else 0.0
            log.log("[inpaint] FIX12 local sharp plate: fired on $locFrames/$written frames, " +
                "${"%.1f".format(pct)}% of hole px served by nearest-first samples " +
                "(radius ±$rad, K=$LOCAL_K, min=$LOCAL_MIN); the rest kept the mosaic plate")
        } else {
            log.log("[inpaint] FIX12 local sharp plate: DISABLED (LOCAL_PLATE=false)")
        }
        logPushPull("mosaic", ppFrames, written, ppSum, ppWorst, log)
        return holeSum to written
    }

    /**
     * pushPull FIRE-RATE TRIPWIRE.
     *
     * `pushPull` is a structureless multigrid diffusion - it is the LAST RESORT, and on `c1` the
     * shipped windowed path was quietly routing 27-87 % of every hole through it, which is exactly
     * what erased the owner's door. Nothing logged that. This does, so the next time a plate stops
     * covering its holes it shows up in one line of the run log instead of in the owner's eyes.
     *
     * MEASURED reference for the mosaic path on c1 (Python, N=140): fires on 38/140 frames, mean
     * 1.16 % of the hole, worst 4.51 %.
     */
    private fun logPushPull(tag: String, frames: Int, written: Int, sum: Double, worst: Double, log: Logger) {
        if (frames == 0) {
            log.log("[inpaint] pushPull($tag) last-resort fill: NEVER fired (0/$written frames) - the plate covered every hole")
            return
        }
        val mean = 100.0 * sum / frames
        val line = "[inpaint] pushPull($tag) last-resort fill: fired on $frames/$written frames, " +
            "mean ${"%.2f".format(mean)}% of the hole, worst ${"%.2f".format(100.0 * worst)}%"
        if (worst > 0.10) log.log("$line ⚠ over 10% of a hole is a structureless wash - treat this as a defect")
        else log.log(line)
    }

    // =============================================================================================
    // STEP 3 - neural-fill the core once. LaMa is cropped to the core bbox (max detail) and grafted
    // ONLY into the true (undilated) core so the real-median ring around it is kept; then grain-matched.
    // =============================================================================================

    private fun fillCore(plate: IntArray, core: BooleanArray, w: Int, h: Int, np: Int, getLama: () -> LamaInpainter?, log: Logger) {
        val coreD = core.copyOf(); dilateInPlace(coreD, w, h, CORE_DILATE)   // padded context for LaMa
        val corePx = core.count { it }
        log.log("[inpaint] neural core fill: $corePx px (${"%.2f".format(100.0 * corePx / np)}% of frame)")
        // §2b anti-hallucination: a very large person-shaped hole risks a neural inpainter inventing a
        // PERSON; fall to structureless push-pull (smooth gradients that cannot form a person). §2-safe.
        if (corePx > np * 0.35) {
            log.log("[inpaint] core >35% of frame -> structureless push-pull (anti-hallucination)")
            pushPull(plate, coreD, w, h); addGrain(plate, core, w, h, np, log); return
        }
        var done = false
        // Only NOW is the 208 MB session worth building: we know the core is non-empty and
        // under the 35 % anti-hallucination ceiling, i.e. LaMa will genuinely be used.
        val lama = getLama()
        if (lama != null) {
            // crop a padded square around the core bbox so LaMa resolves the core at higher effective res
            var minx = w; var miny = h; var maxx = -1; var maxy = -1
            for (y in 0 until h) for (x in 0 until w) if (coreD[y * w + x]) {
                if (x < minx) minx = x; if (x > maxx) maxx = x; if (y < miny) miny = y; if (y > maxy) maxy = y
            }
            if (maxx >= 0) done = runCatching {
                val pad = 24
                val side = max(maxx - minx + 1, maxy - miny + 1) + 2 * pad
                val cw = min(side, w); val chh = min(side, h)
                var x0 = ((minx + maxx) / 2 - cw / 2).coerceIn(0, w - cw)
                var y0 = ((miny + maxy) / 2 - chh / 2).coerceIn(0, h - chh)
                val cropPx = IntArray(cw * chh); val cropMask = IntArray(cw * chh)
                for (yy in 0 until chh) for (xx in 0 until cw) {
                    val sp = (y0 + yy) * w + (x0 + xx); val dp = yy * cw + xx
                    cropPx[dp] = plate[sp]; cropMask[dp] = if (coreD[sp]) Color.WHITE else Color.BLACK
                }
                // Timed on its own: this is the ONLY neural inference in Phase 1, once per clip (or once
                // per window). Separating it from LAMA_LOAD is the whole point - the documented "~2 s
                // LaMa" figure conflates a 208 MB model load with a single 512^2 forward pass.
                val filled = Prof.time(Prof.LAMA_INFER) {
                    lama.inpaint(Bitmap.createBitmap(cropPx, cw, chh, Bitmap.Config.ARGB_8888),
                        Bitmap.createBitmap(cropMask, cw, chh, Bitmap.Config.ARGB_8888))
                }
                val opx = IntArray(cw * chh); filled.getPixels(opx, 0, cw, 0, 0, cw, chh)
                for (yy in 0 until chh) for (xx in 0 until cw) {
                    val sp = (y0 + yy) * w + (x0 + xx)
                    if (core[sp]) plate[sp] = opx[yy * cw + xx]   // graft LaMa ONLY into the true core; keep the real ring
                }
                true
            }.getOrElse { log.log("[inpaint] LaMa core crop failed (${it.message?.take(60)}); push-pull"); false }
        }
        if (!done) pushPull(plate, core, w, h)   // pushPull reads `hole` only; core is unchanged for the addGrain below
        addGrain(plate, core, w, h, np, log)   // match surrounding sensor grain so the fill isn't too-smooth
    }

    /**
     * Add Gaussian grain (sigma measured from a real-bg ring around the core) so the neural fill
     * matches.
     *
     * 🔴 FIX9 - THIS SIGMA IS A SPATIAL STANDARD DEVIATION OF IMAGE CONTENT, NOT A NOISE ESTIMATE,
     * AND IT SATURATES ON EVERY TEXTURED SCENE. The statistic is the std of the LUMA over an 8 px
     * ring around the core - which measures the ring's own texture (bricks, paving, foliage), not
     * its sensor noise, and is therefore tens of grey levels wherever the background has any detail
     * at all. MEASURED on the RUN3 clips (`fix9_ref/F9_STATIC.json`): the raw ring sigma is 30-50
     * grey levels, so `coerceIn(1.0, 10.0)` PINS AT 10.0 and the clamp - not the scene - is what
     * sets the grain. The clamp is the only reason this is survivable; without it the core would be
     * swamped. It is instrumented rather than changed here because changing it moves pixels on a
     * path whose defect nobody has priced: the core is 0.00-0.02 % of the frame on all three RUN3
     * clips, so the visible cost is bounded and a fix belongs with a device render to look at.
     */
    private fun addGrain(plate: IntArray, core: BooleanArray, w: Int, h: Int, np: Int, log: Logger) {
        val ring = core.copyOf(); dilateInPlace(ring, w, h, 8)
        var sum = 0.0; var sq = 0.0; var c = 0
        for (p in 0 until np) if (ring[p] && !core[p]) {
            val l = (Color.red(plate[p]) + Color.green(plate[p]) + Color.blue(plate[p])) / 3.0; sum += l; sq += l * l; c++
        }
        if (c < 30) return
        val mean = sum / c
        val sRaw = sqrt((sq / c - mean * mean).coerceAtLeast(0.0))
        if (sRaw > 10.0) log.log("[inpaint] ⚠ SATURATED: addGrain's core-ring sigma is " +
            "${"%.1f".format(sRaw)} grey levels and pins at the 10.0 clamp - that statistic is the " +
            "ring's TEXTURE, not its noise, so the core's grain amplitude is set by the clamp")
        val sigma = sRaw.coerceIn(1.0, 10.0)
        val rnd = java.util.Random(20260719L)   // fixed seed -> deterministic
        for (p in 0 until np) if (core[p]) {
            val d = (rnd.nextGaussian() * sigma).toInt()
            plate[p] = Color.rgb((Color.red(plate[p]) + d).coerceIn(0, 255),
                (Color.green(plate[p]) + d).coerceIn(0, 255), (Color.blue(plate[p]) + d).coerceIn(0, 255))
        }
    }

    // =============================================================================================
    // STEP 4 - per-frame composite: real outside, warped plate inside, exposure-matched + feathered.
    // The hole is dilated by COMPOSITE_DILATE so the FULL contaminated ring the plate excluded is filled
    // from the clean plate (never from the raw frame) - no person / gray / codec-bleed leak. Returns
    // (holeSum, framesWritten); a decode-null re-emits the previous frame to keep 1:1 with the source.
    // =============================================================================================

    private fun compositeFrames(
        masked: FrameSource, mask: FrameSource?, n: Int, w: Int, h: Int, np: Int, maxDim: Int,
        aligner: Aligner?, shiftCache: HashMap<Int, FloatArray>?, plate: IntArray,
        refMeanLuma: Double, grainSigma: Float, out: File, log: Logger,
    ): Pair<Long, Int> {
        log.log("[inpaint] compositing plate into each frame -> ${out.name}")
        val mkpx = IntArray(np); val outPx = IntArray(np)
        val warp = IntArray(np); val alpha = FloatArray(np); val ablur = FloatArray(np)
        // Seam-match scratch (null when the slider is at 0, so the old path allocates nothing).
        val (seamBand, seamRes, seamUnk) = seamScratch(np)
        val hole = BooleanArray(np)          // reused: maskToBool used to allocate 1.6 MB per frame
        // FIX9: [grainSigma] now arrives pre-measured from [staticGrainSigma]. The inline estimate
        // that used to live here needed the previous frame's pixels and the previous frame's aligner
        // shift; deleting it also deletes `prevMv` - an np-sized IntArray (6.4 MB at 1264²) plus a
        // full-frame arraycopy on EVERY output frame. `off` costs 1.6 MB and replaces it.
        val off = BooleanArray(np)           // FIX9: hole pixels whose plate read was edge-CLAMPED
        var edgeFrames = 0; var edgeSum = 0.0; var edgeWorst = 0.0
        val lastOut = IntArray(np); var haveLast = false
        var holeSum = 0L; var written = 0
        // -----------------------------------------------------------------------------------------
        // FILL SHARPNESS ON THE STATIC PATH (2026-08-17).
        //
        // FIX12 shipped the nearest-first local plate on the MOSAIC path only, but the blur it fixes
        // is not a property of the mosaic - it is a property of MIXING DONORS THAT ARE FAR APART IN
        // TIME, and the static path does exactly that: `buildStaticPlate` takes a trimmed mean over up
        // to MED_MAX_SAMPLES frames spread across the WHOLE clip, so its mean donor distance is far
        // larger than the mosaic's measured 46.8 frames. On a JITTER clip every one of those donors
        // carries its own sub-pixel alignment residual, and the mean of them is the same σ≈0.5 px
        // Gaussian FIX12 measured. So the same override applies here, through the same function.
        //
        // The geometry is a pure TRANSLATION, which is what `Aligner` measures: frame t's pixel (x,y)
        // is ref pixel (x − dx_t, y − dy_t). Packed as 3x3 matrices that is `hs[t] = T(−dx,−dy)` and
        // `gs[d] = T(+dx,+dy)`, which is exactly the contract `localSharpen` reads - so it needs no
        // second implementation and no new alignment work.
        //
        // 🔴 NON-REGRESSIVE BY CONSTRUCTION, same as FIX12: `localSharpen` only ever overrides
        // `warp[p]` where at least LOCAL_MIN real donor samples exist; every other pixel keeps the
        // plate value this function has always produced. At sharpness 0 the ring collapses to one
        // slot and this whole block is inert - the loop is then the previous single-frame loop with
        // `dec == t`. ⚠️ NOT MEASURED on a device yet: this session could not render.
        // -----------------------------------------------------------------------------------------
        val ring = newRing(np, ringRadiusFor(np), log)
        val rad = if (LOCAL_PLATE) (ring.cap - 1) / 2 else 0
        val geom = if (rad > 0) MosaicGeom(Array(n) { IDENT9.copyOf() }, Array(n) { IDENT9.copyOf() },
            n, grainSigma, refMeanLuma, n / 2) else null
        val lpx = if (geom != null) IntArray(np) else IntArray(0)
        val lok = if (geom != null) BooleanArray(np) else BooleanArray(0)
        val shifts = arrayOfNulls<FloatArray>(n)
        var locSum = 0L; var locFrames = 0
        if (geom != null) log.log("[inpaint] fill sharpness ${describeFillSharpness()}" +
            (if (rad != LOCAL_RADIUS) " - window CAPPED to ±$rad by the heap" else ""))
        else log.log("[inpaint] fill sharpness ${describeFillSharpness()}")
        log.log("[inpaint] seam match ${describeSeamMatch()}")
        // Second strictly-ascending pass over the same clip -> rewind the prefetcher to 0. The mask
        // IS read sequentially here (once per output frame), so it gets one too.
        masked.startPrefetch(n, maxDim)
        mask?.startPrefetch(n, maxDim)
        Mp4Encoder(out, masked.srcFps).use { enc ->
            // ONE sequential decode head; the emit loop trails it by `rad` frames so the donor window
            // is two-sided at no extra decode cost (see [LocalRing]). rad == 0 => dec == t exactly.
            for (dec in 0 until n + rad) {
                if (dec < n) {
                    val bm = Prof.time(Prof.DECODE) { masked.frameAt(dec, maxDim) }
                    if (bm != null) {
                        val s = ring.slot(dec)
                        Prof.time(Prof.DECODE) { bm.getPixels(ring.px[s], 0, w, 0, 0, w, h) }
                        val shd = shiftCache?.get(dec)
                            ?: Prof.time(Prof.ALIGN) { aligner?.shiftOf(bm, ring.px[s]) } ?: FLOAT00
                        shifts[dec] = shd
                        if (geom != null) {
                            geom.hs[dec][2] = -shd[0].toDouble(); geom.hs[dec][5] = -shd[1].toDouble()
                            geom.gs[dec][2] = shd[0].toDouble(); geom.gs[dec][5] = shd[1].toDouble()
                            Prof.time(Prof.MASK) {
                                fillMaskPixels(mask, dec, maxDim, w, h, ring.px[s], mkpx)
                                maskToBoolInto(mkpx, np, hole)
                            }
                            val rw = ring.raw[s]; Arrays.fill(rw, 0L)
                            for (p in 0 until np) if (hole[p]) rw[p ushr 6] = rw[p ushr 6] or (1L shl (p and 63))
                            // the DILATED copy is what a donor is tested against, matching the plate
                            // sampler, so a donor can never contribute its own person's edge to a fill
                            dilateInPlace(hole, w, h, MASK_DILATE)
                            val dl = ring.dil[s]; Arrays.fill(dl, 0L)
                            for (p in 0 until np) if (hole[p]) dl[p ushr 6] = dl[p ushr 6] or (1L shl (p and 63))
                            ring.gain[s] = (refMeanLuma / meanLuma(ring.px[s], hole, np))
                                .coerceIn(0.85, 1.18).toFloat()
                        }
                        ring.idx[s] = dec
                    }
                }
                val t = dec - rad
                if (t < 0) continue
                if (!ring.has(t)) {
                    // undecodable frame -> repeat the previous output so the track keeps its length
                    if (haveLast) {
                        Prof.time(Prof.ENCODE) { enc.writeFrame(lastOut, w, h) }
                        written++
                    }
                    continue
                }
                val ts = ring.slot(t)
                val mvpx = ring.px[ts]
                Prof.time(Prof.MASK) {
                    // With the ring armed the mask was already decoded once, when this frame entered
                    // it; re-reading it here would be an out-of-order read that retires the prefetcher.
                    if (geom != null) for (p in 0 until np) hole[p] = ring.isRaw(ts, p)
                    else {
                        fillMaskPixels(mask, t, maxDim, w, h, mvpx, mkpx)
                        maskToBoolInto(mkpx, np, hole)
                    }
                    // §2 audit BEFORE the dilation - dilating first would hide exactly the leaks we are
                    // looking for. No-op on the derived path, which audits inside fillMaskPixels.
                    auditStagedMask(mvpx, hole, t)
                }
                dilateInPlace(hole, w, h, COMPOSITE_DILATE)   // fill the FULL contaminated ring from the plate
                val sh = shifts[t] ?: FLOAT00
                val dx = sh[0]; val dy = sh[1]
                var holePx = 0
                Prof.time(Prof.COMPOSITE) {
                    // Band-local hole counts, one atomic add each. Integer addition is associative, so
                    // the total is exact regardless of band order or count.
                    val holeAcc = java.util.concurrent.atomic.AtomicInteger(0)
                    Par.range(np) { p0, p1 ->
                        var local = 0
                        for (p in p0 until p1) { if (hole[p]) local++; alpha[p] = if (hole[p]) 1f else 0f }
                        holeAcc.addAndGet(local)
                    }
                    holePx = holeAcc.get()
                    // The feather is computed BEFORE the warp now. It depends only on `alpha` (which
                    // depends only on `hole`), so hoisting it introduces no data hazard - and it makes
                    // `ablur` available as the warp's coverage predicate below.
                    boxBlur(alpha, ablur, w, h, FEATHER)
                    // WARP ONLY THE PIXELS SOMETHING ACTUALLY READS.
                    //
                    // This filled all w*h entries, but `warp` has exactly two readers in this loop and
                    // between them they touch under half of it:
                    //   * the blend below reads warp[p] only after `if (a <= 0f) … continue`, i.e. only
                    //     where ablur[p] > 0 - the dilated hole plus its FEATHER ring, ~30 % of frame;
                    //   * ringGain samples warp at STRIDE 4 (p += 4) over the non-hole pixels.
                    // So the union is {ablur > 0} ∪ {p ≡ 0 mod 4}, and roughly half of every frame's
                    // bilinear resample was being computed and then never looked at.
                    //
                    // BIT-IDENTICAL: every entry that is READ is still produced by the identical
                    // expression, and the predicate here is the same test the reader applies. Entries
                    // outside the union keep the previous frame's values (warp is reused scratch) but
                    // no reader can observe them - which is precisely why the predicate must stay a
                    // superset of both readers, not just of the blend.
                    // FIX9: `off` records where the plate read was a CLAMPED coordinate, i.e. where
                    // bilinearRGB replicated the plate's edge pixel instead of reading real plate.
                    // Counting is unconditional (it is how the defect becomes visible in the log);
                    // only the ROUTING of those pixels is gated by STATIC_EDGE_GUARD.
                    var offPx = 0
                    if (dx == 0f && dy == 0f) System.arraycopy(plate, 0, warp, 0, np)
                    else {
                        val offAcc = java.util.concurrent.atomic.AtomicInteger(0)
                        Par.rows(h, w) { y0, y1 ->
                            var local = 0
                            for (y in y0 until y1) {
                                val row = y * w
                                for (x in 0 until w) {
                                    val p = row + x
                                    if (ablur[p] > 0f || (p and 3) == 0)
                                        warp[p] = bilinearRGB(plate, w, h, x - dx, y - dy)
                                    if (hole[p]) {
                                        val sx = x - dx; val sy = y - dy
                                        val o = sx < 0f || sx > w - 1f || sy < 0f || sy > h - 1f
                                        off[p] = o
                                        if (o) local++
                                    } else off[p] = false
                                }
                            }
                            offAcc.addAndGet(local)
                        }
                        offPx = offAcc.get()
                    }
                    if (offPx > 0) {
                        edgeFrames++
                        val fr = offPx.toDouble() / max(1, holePx)
                        edgeSum += fr
                        if (fr > edgeWorst) edgeWorst = fr
                    }
                    // Override the clip-wide plate with nearest-first REAL samples wherever enough of
                    // them exist. Runs BEFORE ringGain, exactly as on the mosaic path: ringGain
                    // samples only non-hole pixels and localSharpen writes only hole pixels, so the
                    // exposure match sees the same operands it always did. A pixel served locally is
                    // real background, so it also comes off the edge-replication list.
                    if (geom != null) {
                        val nLoc = localSharpen(ring, t, w, h, np, geom, hole, ablur, warp, lpx, lok)
                        if (nLoc > 0) {
                            locSum += nLoc; locFrames++
                            for (p in 0 until np) if (lok[p]) off[p] = false
                        }
                    }
                    val gain = matchSurroundings(mvpx, warp, hole, ablur, w, h, np, seamBand, seamRes, seamUnk)
                    Par.range(np) { p0, p1 ->
                        for (p in p0 until p1) {
                            val a = ablur[p]
                            if (a <= 0f) { outPx[p] = mvpx[p]; continue }
                            val pr = (Color.red(warp[p]) * gain[0]).toInt().coerceIn(0, 255)
                            val pg = (Color.green(warp[p]) * gain[1]).toInt().coerceIn(0, 255)
                            val pb = (Color.blue(warp[p]) * gain[2]).toInt().coerceIn(0, 255)
                            // grain scales with alpha so it fades out through the feather exactly as
                            // the plate does - no noise is added to untouched real pixels
                            val nz = if (grainSigma > 0f) grainAt(p, t) * grainSigma * a else 0f
                            if (a >= 1f) {
                                outPx[p] = Color.rgb((pr + nz).toInt().coerceIn(0, 255),
                                    (pg + nz).toInt().coerceIn(0, 255), (pb + nz).toInt().coerceIn(0, 255))
                                continue
                            }
                            val r = (Color.red(mvpx[p]) * (1 - a) + pr * a + nz).toInt().coerceIn(0, 255)
                            val g = (Color.green(mvpx[p]) * (1 - a) + pg * a + nz).toInt().coerceIn(0, 255)
                            val b = (Color.blue(mvpx[p]) * (1 - a) + pb * a + nz).toInt().coerceIn(0, 255)
                            outPx[p] = Color.rgb(r, g, b)
                        }
                    }
                    // FIX9, gated: replace the edge-replicated smear with a push-pull diffusion, the
                    // same treatment `compositeWindowed` and `compositeMosaic` already give an
                    // uncovered hole pixel. A no-op on any frame with offPx == 0.
                    if (STATIC_EDGE_GUARD && offPx > 0) pushPull(outPx, off, w, h)
                }
                holeSum += holePx
                Prof.time(Prof.ENCODE) { enc.writeFrame(outPx, w, h) }
                written++
                System.arraycopy(outPx, 0, lastOut, 0, np); haveLast = true
                if (t % 20 == 0 || t == n - 1) log.log("[inpaint] frame $t/$n")
            }
        }
        // The share of the hole served by SHARP local samples - the whole point of the slider, so it
        // is LOGGED rather than inferred: if it reads 0 %, the setting did nothing and any quality
        // claim about this output is void.
        if (geom != null) {
            val pct = if (holeSum > 0) 100.0 * locSum / holeSum else 0.0
            log.log("[inpaint] local sharp plate (static): fired on $locFrames/$written frames, " +
                "${"%.1f".format(pct)}% of hole px served by nearest-first samples " +
                "(radius ±$rad, K=$LOCAL_K, min=$LOCAL_MIN); the rest kept the clip-wide plate")
        } else log.log("[inpaint] local sharp plate (static): DISABLED (fill sharpness 0)")
        // FIX9: the edge-replication fire rate. This prints whether or not the guard is enabled - 
        // a defect nothing reports is a defect nobody finds.
        if (edgeFrames > 0) {
            log.log("[inpaint] ⚠ SATURATED: the plate warp read a CLAMPED coordinate on " +
                "$edgeFrames/$written frames - mean ${"%.2f".format(100.0 * edgeSum / edgeFrames)}% " +
                "of the hole, worst ${"%.2f".format(100.0 * edgeWorst)}%. Those pixels are the " +
                "plate's edge column/row REPLICATED, i.e. a structureless smear at the pan-leading " +
                "edge. STATIC_EDGE_GUARD=$STATIC_EDGE_GUARD" +
                (if (STATIC_EDGE_GUARD) " -> diffused instead." else " -> shipped as the smear."))
            if (edgeWorst > 0.10) log.log("[inpaint] ⚠ over 10% of a hole came from a clamped " +
                "coordinate - treat this as a defect, not a tolerance (FIX9_CLAMPS.md)")
        } else log.log("[inpaint] plate warp: no clamped-coordinate reads (the plate covered every hole)")
        return holeSum to written
    }

    /**
     * Per-channel gain so the plate's overall exposure matches THIS frame's real background (kills the
     * seam "pumping" from auto-exposure drift). Global (all non-hole pixels), clamped to a safe range.
     *
     * PARALLEL, AND PROVABLY BIT-IDENTICAL TO THE SERIAL DOUBLE VERSION. This was deliberately left
     * serial, with the reasoning that "splitting the accumulation across bands would reassociate the
     * float adds and stop being bit-identical". That hazard is real in general but does not apply here,
     * and the proof is short:
     *
     *  - every addend is `Color.red/green/blue(...)`, an integer in [0, 255] - exactly representable;
     *  - the loop visits at most np/4 pixels, so at 1264² that is ≤ 399 424 terms and every partial
     *     sum is ≤ 399 424 × 255 = 101 853 120 < 2^53;
     *  - IEEE-754 addition of two exactly-representable integers whose exact sum is below 2^53 is
     *     EXACT. So the serial Double loop was never doing floating-point arithmetic in the first
     *     place - it was doing integer arithmetic in a Double, and integer addition is associative.
     *
     * Therefore any grouping gives the identical total, and Long accumulation gives that same total
     * with no rounding question at all. The Longs convert back to Double exactly (all < 2^53), so
     * `g()` divides identical operands and returns identical Floats.
     *
     * BAND ALIGNMENT. The serial walk visits p ∈ {0, 4, 8, …}. `Par.range` does not hand out 4-aligned
     * bands, so each band rounds its start UP to the next multiple of 4; the union over bands is then
     * exactly the serial index set, with no pixel visited twice and none missed.
     *
     * (The same argument does NOT extend to [meanLuma], which sums `(r+g+b)/3.0` - a value that is not
     * exactly representable in binary. That one stays serial.)
     */
    private fun ringGain(frame: IntArray, plate: IntArray, hole: BooleanArray, np: Int): FloatArray {
        val acc = java.util.concurrent.atomic.AtomicLongArray(7)   // fr fg fb pr pg pb cnt
        Par.range(np) { from, to ->
            var fr = 0L; var fg = 0L; var fb = 0L; var pr = 0L; var pg = 0L; var pb = 0L; var cnt = 0L
            var p = (from + 3) and 3.inv()          // first multiple of 4 at or after `from`
            while (p < to) {
                if (!hole[p]) {
                    val f = frame[p]; val q = plate[p]
                    fr += (f ushr 16) and 0xFF; fg += (f ushr 8) and 0xFF; fb += f and 0xFF
                    pr += (q ushr 16) and 0xFF; pg += (q ushr 8) and 0xFF; pb += q and 0xFF
                    cnt++
                }
                p += 4
            }
            acc.addAndGet(0, fr); acc.addAndGet(1, fg); acc.addAndGet(2, fb)
            acc.addAndGet(3, pr); acc.addAndGet(4, pg); acc.addAndGet(5, pb); acc.addAndGet(6, cnt)
        }
        if (acc.get(6) < 50) return floatArrayOf(1f, 1f, 1f)
        // FIX9 instrumentation. Counting only: `g` still returns the identical clamped Float, and
        // the counters below are never read by any expression that produces a pixel. One frame is
        // counted once even when two channels pin, which is what the log line claims.
        satRingN++
        var pinnedHere = false
        fun g(a: Long, b: Long): Float {
            if (b <= 1L) return 1f
            val raw = (a.toDouble() / b.toDouble()).toFloat()
            if (raw < 0.75f || raw > 1.35f) {
                pinnedHere = true
                if (abs(raw - 1f) > abs(satRingWorst - 1.0)) satRingWorst = raw.toDouble()
            }
            return raw.coerceIn(0.75f, 1.35f)
        }
        val out = floatArrayOf(g(acc.get(0), acc.get(3)), g(acc.get(1), acc.get(4)), g(acc.get(2), acc.get(5)))
        if (pinnedHere) satRingPinned++
        return out
    }

    // =============================================================================================
    // shared helpers
    // =============================================================================================

    private fun fillMaskPixels(mask: FrameSource?, t: Int, maxDim: Int, w: Int, h: Int, mvpx: IntArray, mkpx: IntArray) {
        if (mask != null) {
            val mkRaw = mask.frameAt(t, maxDim)
            if (mkRaw != null) {
                val mk = if (mkRaw.width == w && mkRaw.height == h) mkRaw else Bitmap.createScaledBitmap(mkRaw, w, h, true)
                mk.getPixels(mkpx, 0, w, 0, 0, w, h); mkStaged++; return
            }
            // 🔴 FIX11 - a STAGED mask.mp4 exists and this frame would not decode. Everything below is
            // a GUESS at the silhouette from the picture, so the run is already degraded; the only
            // unforgivable thing is to do it silently, which is what shipped.
            mkFallbackWarn(t)
        }
        // ROBUST path: auto-detected fill colour + flatness + bounded geodesic growth (see HoleMask).
        val hm = holeMask; val hb = holeBuf
        if (hm != null && hb != null && hb.size == mvpx.size) {
            val count = hm.maskPixelsInto(mvpx, fillV, mkpx, hb)
            if (count >= 0) {
                // privacy tripwire: on-colour FLAT pixels left outside the mask (see HoleMask.auditLeak)
                // Skipped when the STAGED auditor is armed - it scores the same quantity on the same
                // frames, and double counting would halve the reported average.
                if (auditor == null) { leakSum += hm.auditLeak(mvpx, fillV, hb).toLong(); leakFrames++ }
                mkRobust++
                return
            }
            // implausible (> 60 % of frame) -> fall through to the legacy test rather than erase the scene
        }
        mkLegacy++
        val v = fillV
        for (p in mvpx.indices) {
            val c = mvpx[p]
            val isHole = abs(Color.red(c) - v) <= GRAY_TOL &&
                abs(Color.green(c) - v) <= GRAY_TOL &&
                abs(Color.blue(c) - v) <= GRAY_TOL
            mkpx[p] = if (isHole) -1 else -0x1000000
        }
    }

    private fun maskToBool(mkpx: IntArray, np: Int): BooleanArray {
        val b = BooleanArray(np); maskToBoolInto(mkpx, np, b); return b
    }

    /** Fill-into variant: the per-frame loops call this ~141x per clip, and the allocating form was
     *  handing the GC a fresh 1.6 MB BooleanArray every single time. */
    private fun maskToBoolInto(mkpx: IntArray, np: Int, out: BooleanArray) {
        for (p in 0 until np) out[p] = (mkpx[p] and 0xFF) >= 128
    }

    /**
     * In-place binary dilation by radius [r] (separable). Only ever ADDS true pixels.
     *
     * BIT-IDENTICAL REWRITE (was: two O(r)-scan passes, the second of them COLUMN-major). Both the old
     * and the new code compute exactly "is any true pixel within Chebyshev radius r, clamped at the
     * border", so the output BooleanArray is unchanged - but this version is O(1) per pixel via a
     * sliding true-count, and both passes now walk ROW-major.
     *
     * The old vertical pass was `for (x) for (y) { ... [y*w+x] }`, i.e. a 1264-element stride per step
     * at native resolution - a cache miss on essentially every access. That pass, not the arithmetic,
     * was the cost: dilateInPlace runs 143x per 100-frame clip (41 sampled frames at r=5 in
     * aggregatePlate + 100 output frames at r=7 in compositeFrames + the two fillCore/addGrain calls).
     */
    private fun dilateInPlace(m: BooleanArray, w: Int, h: Int, r: Int) {
        if (r <= 0) return
        Prof.time(Prof.DILATE) {
            val tmp = scratchBool(m.size)
            // ---- horizontal: sliding count over [x-r, x+r]. Rows are independent; each band reads its
            //      own rows of m and writes its own rows of tmp. ----
            Par.rows(h, w) { y0, y1 ->
                for (y in y0 until y1) {
                    val row = y * w
                    var cnt = 0
                    val prime = min(w - 1, r)
                    for (x in 0..prime) if (m[row + x]) cnt++
                    for (x in 0 until w) {
                        tmp[row + x] = cnt > 0
                        val drop = x - r
                        if (drop >= 0 && m[row + drop]) cnt--
                        val add = x + 1 + r
                        if (add < w && m[row + add]) cnt++
                    }
                }
            }
            // ---- vertical: per-column sliding count over [y-r, y+r], walked ROW-major. Each band primes
            //      its OWN window at its first row, so banding cannot change the result; tmp is read-only
            //      here and each band writes only its own rows of m. ----
            Par.rows(h, w) { y0, y1 ->
                val cnt = IntArray(w)
                for (yy in max(0, y0 - r)..min(h - 1, y0 + r)) {
                    val ro = yy * w
                    for (x in 0 until w) if (tmp[ro + x]) cnt[x]++
                }
                for (y in y0 until y1) {
                    val o = y * w
                    for (x in 0 until w) m[o + x] = cnt[x] > 0
                    val drop = y - r
                    if (drop >= 0) { val ro = drop * w; for (x in 0 until w) if (tmp[ro + x]) cnt[x]-- }
                    val add = y + 1 + r
                    if (add < h) { val ro = add * w; for (x in 0 until w) if (tmp[ro + x]) cnt[x]++ }
                }
            }
        }
    }

    /**
     * Separable box blur of a float plane (used to feather the composite alpha).
     *
     * The vertical pass was COLUMN-major (`for (x) for (y)`) over a row-major FloatArray - a ~5 KB
     * stride per step, so a near-certain cache AND TLB miss on every access. Swapping the nesting is
     * BIT-IDENTICAL: each output element still sums its own 5 taps in the same d = -r..+r order, so no
     * float reassociation occurs. Only the visit order of independent outputs changes.
     */
    private fun boxBlur(src: FloatArray, dst: FloatArray, w: Int, h: Int, r: Int) {
        if (r <= 0) { System.arraycopy(src, 0, dst, 0, src.size); return }
        val tmp = scratchFloat(src.size)
        Par.rows(h, w) { y0, y1 ->
            for (y in y0 until y1) {
                val row = y * w
                for (x in 0 until w) {
                    var acc = 0f; var c = 0; var d = -r
                    while (d <= r) { val xx = x + d; if (xx in 0 until w) { acc += src[row + xx]; c++ }; d++ }
                    tmp[row + x] = acc / c
                }
            }
        }
        Par.rows(h, w) { y0, y1 ->
            for (y in y0 until y1) {
                val row = y * w
                for (x in 0 until w) {
                    var acc = 0f; var c = 0; var d = -r
                    while (d <= r) { val yy = y + d; if (yy in 0 until h) { acc += tmp[yy * w + x]; c++ }; d++ }
                    dst[row + x] = acc / c
                }
            }
        }
    }

    // ---- reusable scratch (these helpers are called 143x / 100x per clip; the old code allocated a
    //      fresh np-sized array on EVERY call, i.e. ~1.6 MB boolean + 6.4 MB float per output frame of
    //      pure GC churn). Single-threaded by contract: the row-band pool parallelises INSIDE these
    //      functions, never calls them concurrently. ----
    /**
     * TEMPORAL GRAIN for the composited fill.
     *
     * MEASURED 2026-08-04 on the 70-frame clip: the temporal 2nd-difference energy INSIDE the filled
     * region is 3.78 grey levels against 4.90 outside - the fill is 23 % SMOOTHER than the scene around
     * it. That is inherent to the method, not a bug: the plate is a trimmed mean over 55 frames, and
     * trimming averages sensor noise away (the KDoc calls that out as a feature, "denoises sensor noise
     * -> sharper"). But a noise-free patch sitting inside live, grainy video reads as a frozen region,
     * which is what remains visible after the plate-sampling fix.
     *
     * `addGrain` already adds grain to the neural core, but it bakes ONE fixed-seed pattern into the
     * plate, so that grain is static too. This adds grain PER FRAME across the whole composited hole,
     * at a sigma estimated from the clip's own frame-to-frame noise, so the fill matches the live
     * texture instead of standing still inside it.
     *
     * A precomputed table + multiplicative hash keeps it to a few ns/pixel; a per-frame seed makes the
     * pattern change every frame (the point), and it is derived from the frame index so a re-run is
     * reproducible.
     */
    private val grainTable = FloatArray(8192).also { t ->
        val r = java.util.Random(0x5EEDL)
        for (i in t.indices) t[i] = r.nextGaussian().toFloat()
    }

    private fun grainAt(p: Int, seed: Int): Float {
        val h = (p * -1640531527) xor (seed * 0x9E3779B1.toInt())
        return grainTable[(h ushr 12) and 8191]
    }

    /**
     * Per-frame noise sigma of the REAL background, from consecutive frames outside the hole.
     *
     * 🔴 FIX9 (2026-08-08): THIS IS NOW THE WINDOWED-FALLBACK PATH ONLY. The STATIC path no longer
     * calls it - [staticGrainSigma] does the job with a homography and a median of pairs, because
     * the integer translation below was MEASURED still pinning at `GRAIN_MAX` on 4/12 pairs of `c3`
     * (a STATIC-branch clip, devC 39.1 px) and on 10/12 pairs of `c1`. Sub-pixel translation was
     * measured too and is NOT enough either (1/12 on c3, 10/12 on c1): `devC <= 118.5 px` bounds the
     * camera's displacement, not its rotation. Left in place, unchanged, for `compositeWindowed` - 
     * which is only reached when the mosaic path fails - and its saturation now prints a line.
     *
     * 🔴 TWO EARLIER FIXES (2026-08-08), because the version before them measured CAMERA MOTION, not noise.
     * It differenced `cur[p]` against `prev[p]` at the SAME index and took the MEAN. On a panning
     * clip that is the frame-to-frame image displacement, which is enormous. MEASURED on c1
     * (140 f @1264², 47 frame pairs): the old formula returns a median of **16.50**, i.e. it pins to
     * the `GRAIN_MAX` = 6.0 clamp on every frame. Taking the median instead of the mean is not
     * enough on its own (**10.13**, still clamped). Only motion compensation gets it right:
     * **2.04**, and an independent homography-registered estimate on the same clip agrees at 1.74.
     *
     *  1. **MOTION-COMPENSATED** - [sdx]/[sdy] shift `prev` into `cur`'s frame first. Both callers
     *     already hold the aligner's per-frame shift, so this costs nothing to supply.
     *  2. **MEDIAN, not mean** - a 768-bin histogram of the summed |ΔRGB| (so 1/3-level precision),
     *     rescaled by the half-normal factor: median|d| = 0.6745·σ_d and σ_d = σ_frame·√2, so
     *     σ_frame = median|d| / (0.6745·√2) = median|d| · 1.0483. The median survives the residual
     *     misregistration and the moving-object pixels the mean does not.
     *
     * ⚠️ PIXEL-AFFECTING on both composite paths. NOT verified on device - see FIX1_INPAINT.md.
     */
    private fun estimateGrainSigma(
        cur: IntArray, prev: IntArray, hole: BooleanArray, w: Int, h: Int, sdx: Int, sdy: Int,
    ): Float {
        val bin = IntArray(768); var cnt = 0
        var y = 0
        while (y < h) {
            val sy = y + sdy
            if (sy in 0 until h) {
                var x = 0
                while (x < w) {
                    val p = y * w + x
                    val sx = x + sdx
                    if (!hole[p] && sx >= 0 && sx < w) {
                        val a = cur[p]; val b = prev[sy * w + sx]
                        val d = abs((a shr 16 and 0xFF) - (b shr 16 and 0xFF)) +
                                abs((a shr 8 and 0xFF) - (b shr 8 and 0xFF)) +
                                abs((a and 0xFF) - (b and 0xFF))
                        bin[min(767, d)]++; cnt++
                    }
                    x += 3
                }
            }
            y += 3
        }
        if (cnt < 500) return 0f
        var acc = 0; var med = 0
        for (v in 0 until 768) { acc += bin[v]; if (acc * 2 >= cnt) { med = v; break } }
        val sigma = (med / 3.0 * 1.0483).toFloat()
        return if (sigma.isFinite()) sigma.coerceIn(GRAIN_MIN, GRAIN_MAX) else 0f
    }

    private var sBool = BooleanArray(0)
    private var sFloat = FloatArray(0)
    /** No clearing needed: dilateInPlace's horizontal pass writes every one of the first n entries
     *  before the vertical pass reads any of them. Clearing here would add ~229 MB of pointless memset
     *  per clip (1.6 MB x 143 calls). */
    private fun scratchBool(n: Int): BooleanArray {
        if (sBool.size < n) sBool = BooleanArray(n)
        return sBool
    }
    private fun scratchFloat(n: Int): FloatArray {
        if (sFloat.size < n) sFloat = FloatArray(n)
        return sFloat
    }

    // ---- push-pull diffusion (fallback core fill when no LaMa; fills [hole] of [rgb] from surroundings) ----
    private fun pushPull(rgb: IntArray, hole: BooleanArray, w: Int, h: Int) {
        val cW = ArrayList<Int>(); val cH = ArrayList<Int>()
        val col = ArrayList<FloatArray>(); val wt = ArrayList<FloatArray>()
        val c0 = FloatArray(w * h * 3); val w0 = FloatArray(w * h)
        for (i in 0 until w * h) if (!hole[i]) {
            c0[3 * i] = Color.red(rgb[i]).toFloat(); c0[3 * i + 1] = Color.green(rgb[i]).toFloat(); c0[3 * i + 2] = Color.blue(rgb[i]).toFloat(); w0[i] = 1f
        }
        cW.add(w); cH.add(h); col.add(c0); wt.add(w0)
        var lw = w; var lh = h
        while (lw > 1 || lh > 1) {
            val nw = (lw + 1) / 2; val nh = (lh + 1) / 2
            val nc = FloatArray(nw * nh * 3); val nwt = FloatArray(nw * nh)
            val pc = col.last(); val pw = wt.last(); val pW = lw; val pH = lh
            for (y in 0 until nh) for (x in 0 until nw) {
                var r = 0f; var g = 0f; var b = 0f; var wsum = 0f
                var dy = 0
                while (dy < 2) { var dx = 0; while (dx < 2) {
                    val px = x * 2 + dx; val py = y * 2 + dy
                    if (px < pW && py < pH) { val pi = py * pW + px; val ww = pw[pi]; r += pc[3 * pi] * ww; g += pc[3 * pi + 1] * ww; b += pc[3 * pi + 2] * ww; wsum += ww }
                    dx++ }; dy++ }
                val ni = y * nw + x
                if (wsum > 0f) { nc[3 * ni] = r / wsum; nc[3 * ni + 1] = g / wsum; nc[3 * ni + 2] = b / wsum; nwt[ni] = min(1f, wsum / 4f) }
            }
            cW.add(nw); cH.add(nh); col.add(nc); wt.add(nwt); lw = nw; lh = nh
        }
        for (l in col.size - 2 downTo 0) {
            val cc = col[l]; val cwt = wt[l]; val cw = cW[l]; val ch = cH[l]
            val fc = col[l + 1]; val fw = cW[l + 1]; val fh = cH[l + 1]
            for (y in 0 until ch) for (x in 0 until cw) {
                val i = y * cw + x
                if (cwt[i] < 1f) {
                    val a = cwt[i]; val fx = x * 0.5f; val fy = y * 0.5f
                    val x0 = fx.toInt().coerceIn(0, fw - 1); val y0 = fy.toInt().coerceIn(0, fh - 1)
                    val x1 = (x0 + 1).coerceAtMost(fw - 1); val y1 = (y0 + 1).coerceAtMost(fh - 1)
                    val tx = fx - x0; val ty = fy - y0
                    for (k in 0 until 3) {
                        val c00 = fc[3 * (y0 * fw + x0) + k]; val c10 = fc[3 * (y0 * fw + x1) + k]
                        val c01 = fc[3 * (y1 * fw + x0) + k]; val c11 = fc[3 * (y1 * fw + x1) + k]
                        val v = (c00 * (1 - tx) + c10 * tx) * (1 - ty) + (c01 * (1 - tx) + c11 * tx) * ty
                        cc[3 * i + k] = a * cc[3 * i + k] + (1 - a) * v
                    }
                }
            }
        }
        val fin = col[0]
        for (i in 0 until w * h) if (hole[i]) {
            rgb[i] = Color.rgb(fin[3 * i].toInt().coerceIn(0, 255), fin[3 * i + 1].toInt().coerceIn(0, 255), fin[3 * i + 2].toInt().coerceIn(0, 255))
        }
    }
}
