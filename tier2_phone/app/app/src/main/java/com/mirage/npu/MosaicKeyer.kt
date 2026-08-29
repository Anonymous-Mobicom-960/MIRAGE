package com.mirage.npu

import android.graphics.Bitmap
import android.graphics.Color
import kotlin.math.abs
import kotlin.math.max
import kotlin.math.min
import kotlin.math.sqrt

/**
 * MosaicKeyer - pull an alpha matte for the synthetic character when the cloud delivers it composited
 * over a **LIGHTMAP background** instead of over black, and no `synthetic_alpha_pK.mp4` was exported.
 *
 * WHY THIS EXISTS (measured): [Compositor.alphaFromPersonLuma] assumes character-on-BLACK and keys on
 * `luma > 8`. The user's `character.mp4` puts the character over the scene's lightmap, so `luma > 8` was
 * true for **99.7 % of the frame** → alpha ≈ 1 everywhere → Phase 2 pasted the ENTIRE character clip
 * (blocky lightmap and all) over the reconstructed background, and the output showed only that clip.
 *
 * THE KEY INSIGHT: a lightmap background is **block-constant by construction** - the scene downscaled
 * to a tiny G×G grid and upscaled back. Verified on the user's clip: at G = 32 the median within-block
 * standard deviation is **1.03** and 86 % of blocks are below 3. The character, by contrast, carries
 * real high-frequency detail. So the background is exactly recoverable and the character is whatever
 * disagrees with it:
 *
 *   1. AUTO-DETECT the grid G - the COARSEST candidate whose blocks are already flat (every finer grid
 *      is trivially flat too, so "coarsest flat" is the only well-posed rule). No hardcoded 32.
 *   2. Per block, take the per-channel **MODE** over the block's INTERIOR (block borders carry codec
 *      ringing that would bias the estimate). A block whose interior standard deviation exceeds
 *      [CONTAM_STD] is CONTAMINATED - the character covers part of it.
 *   3. REPAIR contaminated blocks by diffusing clean neighbours across the tiny G×G grid, so we have a
 *      background estimate even under the character's torso.
 *   4. matte = |pixel − blockBackground| over a ramp, then clean up: close the blocky gaps, drop
 *      specks, fill the interior holes where clothing happened to match its own block, and hand a
 *      BINARY matte to [Compositor.haloAlpha], which owns the halo-safe rim (binarize → erode → AA).
 *      Do not feather here - haloAlpha re-binarizes at >127, so any soft ramp built here is discarded.
 *
 * Measured against Phase 1's independently-derived person hole on the same clip, after cleanup:
 * 12.48 / 16.84 / 17.03 % at frames 0 / 40 / 80 vs. a hole of 12.39 / 16.96 / 17.50 % - agreement
 * within half a point, which is the correctness check that matters (the synthetic character should
 * occupy the space the real person vacated).
 *
 * MEASURED REJECTIONS (do not "improve" these without re-measuring):
 * - **Chroma-weighted distance** (Y + 4·chroma, a plausible-sounding robustness win) is WORSE on this
 *    footage: an olive character over an olive-gray mosaic separates mostly in LUMA, so up-weighting
 *    chroma suppressed the matte by up to 10 points. Plain RGB L2 wins.
 * - **Auto-tuning the ramp** from the clean-block noise floor (μ+4σ, robust MAD, or a p99.9 quantile)
 *    all under-key badly (3-8 % vs a true 12-17 %): "clean" blocks still contain character pixels, so
 *    every dispersion estimate is inflated. Fixed [DEF_LO]/[DEF_HI] measured best; they are exposed as
 *    config knobs instead.
 * - **Mode-agreement** as the contamination test lets a block that is FULLY covered by smooth clothing
 *    pass as clean (it agrees with its own mode). Interior std is the correct test.
 *
 * PRIVACY (§2): reads only the already-anonymized synthetic character clip. No real-person pixels.
 */
object MosaicKeyer {

    /** Candidate mosaic grids, ascending = coarser blocks first. */
    private val GRIDS = intArrayOf(16, 24, 32, 48, 64, 96)
    private const val FLAT_STD = 3.0        // median interior block std below this => block-constant
    private const val ABORT_STD = 4.0       // no grid under this => not a mosaic at all
    private const val MODE_TOL = 6          // per-channel tolerance when counting agreement with the mode
    /**
     * A block keeps its OWN mode as the background estimate as long as this fraction of its interior
     * agrees with that mode; only near-fully-covered blocks get a diffused estimate.
     *
     * This threshold is low ON PURPOSE, and it is the fix for a real artifact: an EDGE block is roughly
     * half character and half background, so "is this block contaminated?" (interior std) says yes and
     * the whole block used to be replaced by a neighbour-diffused average. That average is close to, but
     * not equal to, the block's true value - so the block's BACKGROUND half then read as "different from
     * the estimate" and keyed in, producing rectangular 39 px slabs welded to the character's shoulders
     * and arms (visible in the v0.10 device run). The background half is exactly constant, so it still
     * wins the mode vote; using the block's own mode makes the estimate EXACT there and the slabs vanish.
     */
    private const val MIN_SUPPORT = 0.25f
    private const val EDGE = 4              // px trimmed from each block edge (codec ringing)
    /**
     * Matte ramp on RGB L2 distance from the block background. Measured sweep against Phase 1's
     * independently-derived person hole: lo 15 tracked it best across frames (−0.01 / +0.02 / −0.30
     * points vs −0.03 / −0.09 / −0.36 at lo 18) and slightly reduced boundary raggedness. Exposed as
     * config knobs (`keyer_lo` / `keyer_hi`) because the right floor is a property of the export chain.
     */
    const val DEF_LO = 15f
    const val DEF_HI = 37f
    private const val MIN_AREA_FRAC = 0.003 // drop matte blobs smaller than 0.3 % of the frame
    /**
     * KNOWN LIMIT - read before tuning this. Where the character's clothing happens to match the colour
     * of the mosaic block behind it, the matte develops block-aligned NOTCHES. Interior coincidences are
     * repaired by [fillInteriorHoles]; notches that open onto the silhouette boundary (e.g. the dark
     * olive hem of a kurta over dark pavement blocks, at frame 80 of the user's clip) are NOT repairable
     * here: a measured sweep of this radius (4 → 16) moved boundary raggedness by < 1 % while starting to
     * fatten the figure, so a bigger close is not the answer. No purely image-based keyer can separate
     * "garment that matches its background" from "background" - the real fix is exporting
     * `synthetic_alpha_pK.mp4` from the cloud, which [NCompositor] already prefers when present.
     */
    private const val CLOSE_R = 8
    private const val LUMA_BLACK_FRAC = 0.20 // >= this share near-black => it IS a character-on-black clip

    /**
     * Decide how a character layer should be matted:
     *   > 0  → character over a block-constant lightmap; the value is the detected grid.
     *     0  → classic character-on-black; use [Compositor.alphaFromPersonLuma].
     *    -1  → neither is convincing; caller should warn and fall back.
     */
    fun detectMode(px: IntArray, w: Int, h: Int): Int {
        var dark = 0
        for (c in px) {
            val luma = 0.299 * Color.red(c) + 0.587 * Color.green(c) + 0.114 * Color.blue(c)
            if (luma <= 8) dark++
        }
        if (dark >= LUMA_BLACK_FRAC * px.size) return 0
        return detectGrid(px, w, h)
    }

    /**
     * COARSEST candidate grid whose interior blocks are already flat; -1 if none is.
     * (Any grid finer than the true one is also flat - subdividing a constant block stays constant - 
     * so scanning coarse→fine and stopping at the first flat level recovers the true grid.)
     */
    fun detectGrid(px: IntArray, w: Int, h: Int): Int {
        var bestMed = Double.MAX_VALUE
        for (g in GRIDS) {
            if (g > min(w, h) / 8) continue          // need a usable block interior after trimming EDGE
            val med = medianBlockStd(px, w, h, g)
            if (med < bestMed) bestMed = med
            if (med < FLAT_STD) return g
        }
        return if (bestMed < ABORT_STD) GRIDS.last() else -1
    }

    private fun medianBlockStd(px: IntArray, w: Int, h: Int, g: Int): Double {
        val stds = DoubleArray(g * g)
        for (by in 0 until g) {
            for (bx in 0 until g) {
                stds[by * g + bx] = interiorStd(px, w, h, g, bx, by)
            }
        }
        stds.sort()
        return stds[stds.size / 2]
    }

    /** Mean per-channel std over block (bx,by)'s interior (EDGE px trimmed from each side). */
    private fun interiorStd(px: IntArray, w: Int, h: Int, g: Int, bx: Int, by: Int): Double {
        var y0 = by * h / g; var y1 = (by + 1) * h / g
        var x0 = bx * w / g; var x1 = (bx + 1) * w / g
        if (y1 - y0 > 2 * EDGE) { y0 += EDGE; y1 -= EDGE }
        if (x1 - x0 > 2 * EDGE) { x0 += EDGE; x1 -= EDGE }
        var n = 0
        var sr = 0.0; var sg = 0.0; var sb = 0.0
        var qr = 0.0; var qg = 0.0; var qb = 0.0
        for (y in y0 until y1) { val row = y * w
            for (x in x0 until x1) {
                val c = px[row + x]
                val r = Color.red(c).toDouble(); val gg = Color.green(c).toDouble(); val b = Color.blue(c).toDouble()
                sr += r; sg += gg; sb += b; qr += r * r; qg += gg * gg; qb += b * b; n++
            }
        }
        if (n == 0) return 0.0
        val vr = max(0.0, qr / n - (sr / n) * (sr / n))
        val vg = max(0.0, qg / n - (sg / n) * (sg / n))
        val vb = max(0.0, qb / n - (sb / n) * (sb / n))
        return (sqrt(vr) + sqrt(vg) + sqrt(vb)) / 3.0
    }

    /** Reusable per-clip keyer. Allocate once (buffers are frame-sized); call [matteInto] per frame. */
    class Keyer(private val w: Int, private val h: Int, private val g: Int) {
        private val np = w * h
        private val bg = IntArray(np)              // the repaired block-constant background estimate
        private val scratch = BooleanArray(np)
        private val visited = BooleanArray(np)
        private val queue = IntArray(np)
        private val blockR = IntArray(g * g)
        private val blockG = IntArray(g * g)
        private val blockB = IntArray(g * g)
        private val good = BooleanArray(g * g)
        private val hist = IntArray(256 * 3)

        /** Write a BINARY character matte for this frame into [out]; returns the pixel count. */
        fun matteInto(px: IntArray, out: BooleanArray, lo: Float = DEF_LO, hi: Float = DEF_HI): Int {
            estimateBackground(px)
            for (i in 0 until np) {
                val c = px[i]; val b = bg[i]
                val dr = Color.red(c) - Color.red(b)
                val dg = Color.green(c) - Color.green(b)
                val db = Color.blue(c) - Color.blue(b)
                val d = sqrt((dr * dr + dg * dg + db * db).toFloat())
                out[i] = ((d - lo) / (hi - lo)) > 0.5f
            }
            dilate(out, CLOSE_R); erode(out, CLOSE_R)   // close the blocky gaps
            keepLargeComponents(out)                    // drop specks
            fillInteriorHoles(out)                      // clothing that matched its own block (enclosed)
            // ...and the same coincidence where it opens onto the boundary, which no fill-holes or close
            // can reach. This is where essentially ALL of the remaining see-through lived. See
            // [Compositor.solidify] for the 6-of-8 inside-ness rule and the measurements.
            Compositor.solidify(out, w, h, Compositor.solidifyReach(out, w, h))
            var n = 0
            for (i in 0 until np) if (out[i]) n++
            return n
        }

        /** 0/255 grayscale matte bitmap - the same contract [Compositor.haloAlpha] expects. */
        fun matteBitmap(px: IntArray, out: BooleanArray, lo: Float = DEF_LO, hi: Float = DEF_HI): Bitmap {
            matteInto(px, out, lo, hi)
            val o = IntArray(np)
            for (i in 0 until np) { val v = if (out[i]) 255 else 0; o[i] = Color.rgb(v, v, v) }
            return Bitmap.createBitmap(o, w, h, Bitmap.Config.ARGB_8888)
        }

        /** Per-block per-channel MODE over the interior; contaminated blocks diffused from clean ones. */
        private fun estimateBackground(px: IntArray) {
            for (by in 0 until g) {
                for (bx in 0 until g) {
                    var y0 = by * h / g; var y1 = (by + 1) * h / g
                    var x0 = bx * w / g; var x1 = (bx + 1) * w / g
                    if (y1 - y0 > 2 * EDGE) { y0 += EDGE; y1 -= EDGE }
                    if (x1 - x0 > 2 * EDGE) { x0 += EDGE; x1 -= EDGE }
                    java.util.Arrays.fill(hist, 0)
                    var n = 0
                    for (y in y0 until y1) { val row = y * w
                        for (x in x0 until x1) {
                            val c = px[row + x]
                            hist[Color.red(c)]++
                            hist[256 + Color.green(c)]++
                            hist[512 + Color.blue(c)]++
                            n++
                        }
                    }
                    val bi = by * g + bx
                    if (n == 0) { good[bi] = false; continue }
                    val mr = argmax(hist, 0); val mg = argmax(hist, 256); val mb = argmax(hist, 512)
                    blockR[bi] = mr; blockG[bi] = mg; blockB[bi] = mb
                    // Trust the block's own mode unless the character has swallowed nearly all of it.
                    var agree = 0
                    for (y in y0 until y1) { val row = y * w
                        for (x in x0 until x1) {
                            val c = px[row + x]
                            if (abs(Color.red(c) - mr) <= MODE_TOL && abs(Color.green(c) - mg) <= MODE_TOL &&
                                abs(Color.blue(c) - mb) <= MODE_TOL) agree++
                        }
                    }
                    good[bi] = agree >= MIN_SUPPORT * n
                }
            }
            repairBlocks()
            // upscale NEAREST - the mosaic is block-constant, so this reproduces it exactly
            for (by in 0 until g) {
                val y0 = by * h / g; val y1 = (by + 1) * h / g
                for (bx in 0 until g) {
                    val x0 = bx * w / g; val x1 = (bx + 1) * w / g
                    val bi = by * g + bx
                    val c = Color.rgb(blockR[bi], blockG[bi], blockB[bi])
                    for (y in y0 until y1) { val row = y * w
                        for (x in x0 until x1) bg[row + x] = c
                    }
                }
            }
        }

        private fun argmax(hist: IntArray, off: Int): Int {
            var best = 0
            for (i in 1 until 256) if (hist[off + i] > hist[off + best]) best = i
            return best
        }

        /** Diffuse clean block colours into contaminated ones across the tiny g×g grid. */
        private fun repairBlocks() {
            var remaining = 0
            for (b in good) if (!b) remaining++
            if (remaining == 0 || remaining == g * g) return
            val filled = ArrayList<Int>()
            var guard = 0
            while (remaining > 0 && guard++ < g * 2) {
                filled.clear()
                for (by in 0 until g) for (bx in 0 until g) {
                    val bi = by * g + bx
                    if (good[bi]) continue
                    var sr = 0; var sg = 0; var sb = 0; var n = 0
                    var dy = -1
                    while (dy <= 1) {
                        var dx = -1
                        while (dx <= 1) {
                            val yy = by + dy; val xx = bx + dx
                            if (yy in 0 until g && xx in 0 until g) {
                                val j = yy * g + xx
                                if (good[j]) { sr += blockR[j]; sg += blockG[j]; sb += blockB[j]; n++ }
                            }
                            dx++
                        }
                        dy++
                    }
                    if (n > 0) { blockR[bi] = sr / n; blockG[bi] = sg / n; blockB[bi] = sb / n; filled.add(bi) }
                }
                if (filled.isEmpty()) break
                for (bi in filled) good[bi] = true
                remaining -= filled.size
            }
        }

        private fun keepLargeComponents(m: BooleanArray) {
            java.util.Arrays.fill(visited, false)
            java.util.Arrays.fill(scratch, false)
            val minArea = (MIN_AREA_FRAC * np).toInt().coerceAtLeast(64)
            for (s in 0 until np) {
                if (!m[s] || visited[s]) continue
                var qe = 0
                queue[qe++] = s; visited[s] = true
                var qs = 0
                while (qs < qe) {
                    val i = queue[qs++]
                    val x = i % w; val y = i / w
                    if (x > 0) { val j = i - 1; if (m[j] && !visited[j]) { visited[j] = true; queue[qe++] = j } }
                    if (x < w - 1) { val j = i + 1; if (m[j] && !visited[j]) { visited[j] = true; queue[qe++] = j } }
                    if (y > 0) { val j = i - w; if (m[j] && !visited[j]) { visited[j] = true; queue[qe++] = j } }
                    if (y < h - 1) { val j = i + w; if (m[j] && !visited[j]) { visited[j] = true; queue[qe++] = j } }
                }
                if (qe >= minArea) for (k in 0 until qe) scratch[queue[k]] = true
            }
            System.arraycopy(scratch, 0, m, 0, np)
        }

        private fun fillInteriorHoles(m: BooleanArray) {
            java.util.Arrays.fill(visited, false)
            var qe = 0
            for (x in 0 until w) {
                val a = x; val b = (h - 1) * w + x
                if (!m[a] && !visited[a]) { visited[a] = true; queue[qe++] = a }
                if (!m[b] && !visited[b]) { visited[b] = true; queue[qe++] = b }
            }
            for (y in 0 until h) {
                val a = y * w; val b = y * w + w - 1
                if (!m[a] && !visited[a]) { visited[a] = true; queue[qe++] = a }
                if (!m[b] && !visited[b]) { visited[b] = true; queue[qe++] = b }
            }
            var qs = 0
            while (qs < qe) {
                val i = queue[qs++]
                val x = i % w; val y = i / w
                if (x > 0) { val j = i - 1; if (!m[j] && !visited[j]) { visited[j] = true; queue[qe++] = j } }
                if (x < w - 1) { val j = i + 1; if (!m[j] && !visited[j]) { visited[j] = true; queue[qe++] = j } }
                if (y > 0) { val j = i - w; if (!m[j] && !visited[j]) { visited[j] = true; queue[qe++] = j } }
                if (y < h - 1) { val j = i + w; if (!m[j] && !visited[j]) { visited[j] = true; queue[qe++] = j } }
            }
            for (i in 0 until np) if (!m[i] && !visited[i]) m[i] = true
        }

        private fun dilate(m: BooleanArray, r: Int) {
            if (r <= 0) return
            for (y in 0 until h) { val row = y * w
                for (x in 0 until w) { var on = false; var d = -r
                    while (d <= r) { val xx = x + d; if (xx in 0 until w && m[row + xx]) { on = true; break }; d++ }
                    scratch[row + x] = on } }
            for (x in 0 until w) for (y in 0 until h) { var on = false; var d = -r
                while (d <= r) { val yy = y + d; if (yy in 0 until h && scratch[yy * w + x]) { on = true; break }; d++ }
                m[y * w + x] = on }
        }

        private fun erode(m: BooleanArray, r: Int) {
            if (r <= 0) return
            for (y in 0 until h) { val row = y * w
                for (x in 0 until w) { var on = true; var d = -r
                    while (d <= r) { val xx = x + d; if (xx < 0 || xx >= w || !m[row + xx]) { on = false; break }; d++ }
                    scratch[row + x] = on } }
            for (x in 0 until w) for (y in 0 until h) { var on = true; var d = -r
                while (d <= r) { val yy = y + d; if (yy < 0 || yy >= h || !scratch[yy * w + x]) { on = false; break }; d++ }
                m[y * w + x] = on }
        }

        /** Debug/preview helper: the repaired block-constant background estimate. */
        fun backgroundPreview(): Bitmap = Bitmap.createBitmap(bg, w, h, Bitmap.Config.ARGB_8888)
    }
}
