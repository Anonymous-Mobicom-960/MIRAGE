package com.mirage.npu

import android.graphics.Color
import kotlin.math.abs
import kotlin.math.max
import kotlin.math.min

/**
 * HoleMask - derive the person-hole mask from a Tier-1 "silhouette" video when no mask.mp4 is staged.
 *
 * WHY THIS EXISTS (measured, not theoretical): the old detector was a hardcoded per-channel
 * `|c - 128| <= 12`. On a real clip (outdoor alley, large area of GRAY PAVING STONES) that flagged
 * **24-32 % of every frame** as "person hole" when the true silhouette was only **10-17 %** - it
 * swallowed the pavement and the concrete overpass, so Phase 1 "reconstructed" a third of the real
 * scene and the background fill looked broken. Two independent bugs fed it:
 *   1. the fill colour in the real clip is gray **122**, not 128 - and that offset is a video-range /
 *      colour-matrix artefact of the export chain, so it is device-dependent: never hardcode 122 either; and
 *   2. a flat ±12 colour box cannot separate a solid fill from same-coloured natural texture.
 *
 * THE FIX - two cues the pavement cannot fake:
 *   (a) the fill colour is AUTO-DETECTED per clip, and
 *   (b) the fill is perfectly FLAT. Measured on the real clip: the 5×5 local range inside the
 *       silhouette has median **0** (p90 ≤ 5); on the near-same-colour pavement it is 7-32.
 *
 * So we hysteresis-threshold: a STRONG seed must be flat AND on-colour; a WEAK set is merely
 * on-colour with a loose tolerance; the seed then grows into the weak set by a **distance-bounded
 * geodesic dilation** (≤ [GEO_ITERS] px). That recovers the silhouette's codec-noisy boundary while
 * making it impossible to flood down into the connected pavement - an unbounded connected-component
 * grow would swallow the whole floor, since the person's feet touch it.
 *
 * Measured on the user's 100-frame clip: mask 10-18 % of frame (matches the true silhouette), and the
 * only "on-colour" pixels left unmasked are individual smooth PAVING STONES at the bottom of the
 * frame - i.e. real background, correctly excluded. Silhouette leak ≈ 0.
 *
 * PRIVACY (§2) - FAIL CLOSED. This mask decides what gets erased, so under-masking would leave real
 * silhouette pixels in the output. Three guards: the final [FINAL_DILATE_R] dilation deliberately
 * over-covers the boundary; [BackgroundInpaint] dilates further before it samples or composites; and
 * [auditLeak] re-scans each frame for on-colour flat pixels left OUTSIDE the mask so a regression is
 * caught and reported rather than silently shipped.
 *
 * PERFORMANCE: the whole extraction runs at HALF resolution on large frames (the mask only needs ~2 px
 * precision and is dilated by ≥3 px afterwards), then nearest-upsamples - a 4× saving on a 1264² clip.
 */
class HoleMask(private val w: Int, private val h: Int) {

    private val np = w * h
    private val half = np > 700_000
    private val ww = if (half) (w + 1) / 2 else w
    private val hh = if (half) (h + 1) / 2 else h
    private val nw = ww * hh

    // working-resolution buffers
    private val gray = IntArray(nw)
    private val weak = BooleanArray(nw)
    private val strong = BooleanArray(nw)
    private val visited = BooleanArray(nw)
    private val queue = IntArray(nw)
    private val dist = ByteArray(nw)
    private val scratch = BooleanArray(nw)
    private val work = BooleanArray(nw)
    // full-resolution scratch for the final dilation
    private val scratchFull = BooleanArray(np)

    companion object {
        // --- colour gates (per channel, around the auto-detected fill value V) ---
        const val TOL_STRONG = 8       // seed must be tightly on-colour
        const val TOL_WEAK = 18        // growth may run to here (codec ringing at the silhouette edge)
        const val SPREAD_STRONG = 6    // max-min across R,G,B - the fill is neutral
        const val SPREAD_WEAK = 14
        // --- flatness (the cue the pavement cannot fake) ---
        const val FLAT_R = 2           // 5x5 window
        const val FLAT_T = 4           // local range below this = "solid fill"
        // --- growth / cleanup ---
        const val GEO_ITERS = 14       // max px the seed may grow into the weak set (bounds pavement leak)
        const val MIN_AREA_FRAC = 0.002 // drop seed blobs < 0.2 % of the frame (speckle)
        const val CLOSE_R = 2          // heal the boundary where flatness failed
        const val FINAL_DILATE_R = 3   // cover the gray fill's codec-bleed ring (full-res px)
        const val MAX_COVER = 0.60     // > this => detection is bogus; caller falls back
        const val PROX_R = 6           // audit radius around the person, working-res px (~12 full-res)
        const val LEAK_WARN_FRAC = 0.0005 // audit tripwire: near-person unmasked fill above this

        /**
         * Auto-detect the solid fill colour V by voting over sampled frames: only NEUTRAL and FLAT
         * pixels vote (the two properties of a solid fill), so texture cannot win the ballot.
         * Returns -1 when there is no decisive peak (=> not a gray-fill clip; caller falls back).
         */
        fun detectFillColor(src: FrameSource, n: Int, maxDim: Int, log: (String) -> Unit): Int {
            val votes = LongArray(256)
            var sampled = 0L
            val k = min(5, max(1, n))
            for (s in 0 until k) {
                val t = if (k == 1) 0 else s * (n - 1) / (k - 1)
                val bmp = src.frameAt(t, maxDim) ?: continue
                val fw = bmp.width; val fh = bmp.height
                val px = IntArray(fw * fh)
                bmp.getPixels(px, 0, fw, 0, 0, fw, fh)
                val g = IntArray(fw * fh)
                for (i in px.indices) {
                    val c = px[i]
                    g[i] = (Color.red(c) + Color.green(c) + Color.blue(c)) / 3
                }
                for (y in 0 until fh) for (x in 0 until fw) {
                    val i = y * fw + x
                    val c = px[i]
                    val r = Color.red(c); val gg = Color.green(c); val b = Color.blue(c)
                    if (max(r, max(gg, b)) - min(r, min(gg, b)) > SPREAD_STRONG) continue
                    if (localRange(g, fw, fh, x, y, FLAT_R) >= FLAT_T) continue
                    votes[g[i]]++
                    sampled++
                }
            }
            if (sampled < 1000L) { log("[holemask] no flat neutral pixels - cannot auto-detect a fill colour"); return -1 }
            var v = 0
            for (i in 1 until 256) if (votes[i] > votes[v]) v = i
            var band = 0L
            for (i in max(0, v - 3)..min(255, v + 3)) band += votes[i]
            val share = band.toDouble() / sampled
            if (share < 0.35) {
                log("[holemask] fill-colour vote not decisive (peak $v, share ${"%.2f".format(share)}) - falling back to gray(128)")
                return -1
            }
            log("[holemask] auto-detected silhouette fill colour = gray($v) (band share ${"%.2f".format(share)})")
            return v
        }

        /** max-min of [g] over a (2r+1)² window centred at (x,y), edge-clamped. */
        fun localRange(g: IntArray, w: Int, h: Int, x: Int, y: Int, r: Int): Int {
            var mn = 255; var mx = 0
            var dy = -r
            while (dy <= r) {
                val yy = (y + dy).coerceIn(0, h - 1) * w
                var dx = -r
                while (dx <= r) {
                    val v = g[yy + (x + dx).coerceIn(0, w - 1)]
                    if (v < mn) mn = v
                    if (v > mx) mx = v
                    dx++
                }
                dy++
            }
            return mx - mn
        }

        /** True when [c] is inside the loose on-colour gate around [v] (the "weak" fill test). */
        fun isWeakFill(c: Int, v: Int): Boolean {
            val r = Color.red(c); val g = Color.green(c); val b = Color.blue(c)
            if (max(r, max(g, b)) - min(r, min(g, b)) > SPREAD_WEAK) return false
            return abs(r - v) <= TOL_WEAK && abs(g - v) <= TOL_WEAK && abs(b - v) <= TOL_WEAK
        }
    }

    /**
     * Compute the hole mask for one frame's pixels into [out]. Returns the number of hole pixels, or
     * -1 if the result is implausible (covers > [MAX_COVER] of the frame) so the caller can fall back.
     */
    fun maskInto(px: IntArray, v: Int, out: BooleanArray): Int {
        java.util.Arrays.fill(work, false)
        java.util.Arrays.fill(strong, false)

        // ---- 1) subsample to working res + colour gates + grayscale ----
        for (y in 0 until hh) {
            val sy = if (half) min(h - 1, y * 2) else y
            val srow = sy * w; val drow = y * ww
            for (x in 0 until ww) {
                val sx = if (half) min(w - 1, x * 2) else x
                val c = px[srow + sx]
                val r = Color.red(c); val g = Color.green(c); val b = Color.blue(c)
                val i = drow + x
                gray[i] = (r + g + b) / 3
                weak[i] = max(r, max(g, b)) - min(r, min(g, b)) <= SPREAD_WEAK &&
                    abs(r - v) <= TOL_WEAK && abs(g - v) <= TOL_WEAK && abs(b - v) <= TOL_WEAK
            }
        }

        // ---- 2) STRONG seeds = on-colour AND flat (tested only on weak candidates: strong ⊆ weak) ----
        for (y in 0 until hh) for (x in 0 until ww) {
            val i = y * ww + x
            if (!weak[i]) continue
            val sy = if (half) min(h - 1, y * 2) else y
            val sx = if (half) min(w - 1, x * 2) else x
            val c = px[sy * w + sx]
            val r = Color.red(c); val g = Color.green(c); val b = Color.blue(c)
            if (max(r, max(g, b)) - min(r, min(g, b)) > SPREAD_STRONG) continue
            if (abs(r - v) > TOL_STRONG || abs(g - v) > TOL_STRONG || abs(b - v) > TOL_STRONG) continue
            if (localRange(gray, ww, hh, x, y, FLAT_R) >= FLAT_T) continue
            strong[i] = true
        }

        // ---- 3) keep only seed blobs big enough to be a person (drops flat-gray speckle) ----
        java.util.Arrays.fill(visited, false)
        val minArea = (MIN_AREA_FRAC * nw).toInt().coerceAtLeast(32)
        var seedCount = 0
        for (s in 0 until nw) {
            if (!strong[s] || visited[s]) continue
            var qe = 0
            queue[qe++] = s; visited[s] = true
            var qs = 0
            while (qs < qe) {
                val i = queue[qs++]
                val x = i % ww; val y = i / ww
                var dy = -1
                while (dy <= 1) {                      // 8-connectivity keeps thin limbs together
                    var dx = -1
                    while (dx <= 1) {
                        val xx = x + dx; val yy = y + dy
                        if (xx in 0 until ww && yy in 0 until hh) {
                            val j = yy * ww + xx
                            if (strong[j] && !visited[j]) { visited[j] = true; queue[qe++] = j }
                        }
                        dx++
                    }
                    dy++
                }
            }
            if (qe >= minArea) { for (k in 0 until qe) work[queue[k]] = true; seedCount += qe }
        }
        if (seedCount == 0) { java.util.Arrays.fill(out, false); return 0 }

        // ---- 4) DISTANCE-BOUNDED geodesic dilation of the seeds into the weak set ----
        geodesicGrow()

        // ---- 5) close (heal the boundary) -> fill interior holes ----
        dilate(work, ww, hh, CLOSE_R, scratch); erode(work, ww, hh, CLOSE_R, scratch)
        fillInteriorHoles()

        // ---- 6) upsample to full res, then the final safety dilation there ----
        if (half) {
            for (y in 0 until h) {
                val srow = min(hh - 1, y / 2) * ww; val drow = y * w
                for (x in 0 until w) out[drow + x] = work[srow + min(ww - 1, x / 2)]
            }
        } else {
            System.arraycopy(work, 0, out, 0, np)
        }
        dilate(out, w, h, FINAL_DILATE_R, scratchFull)

        var count = 0
        for (i in 0 until np) if (out[i]) count++
        return if (count > MAX_COVER * np) -1 else count
    }

    /**
     * PRIVACY tripwire: count pixels that still look like solid fill (on-colour AND flat) but are NOT
     * in [mask] - and lie WITHIN [PROX_R] of it.
     *
     * The proximity gate is what makes this a usable alarm rather than noise. The failure mode worth
     * catching is an under-covered BOUNDARY: the fill's codec-bleed ring surviving just outside the
     * mask. Same-coloured scene texture far from the person is not a leak - on the user's clip the
     * ungated count was 0.158 % of the frame, all of it individual smooth PAVING STONES at the bottom
     * of the shot, which would have cried wolf on every single run.
     *
     * Must be called immediately after [maskInto] for the same frame (it reuses that frame's working
     * grayscale and pre-dilation mask).
     */
    fun auditLeak(px: IntArray, v: Int, mask: BooleanArray): Int {
        System.arraycopy(work, 0, scratch, 0, nw)
        dilate(scratch, ww, hh, PROX_R, weak)   // `weak` is spent for this frame -> reuse as scratch
        var leak = 0
        for (y in 0 until hh) {
            val sy = if (half) min(h - 1, y * 2) else y
            for (x in 0 until ww) {
                if (!scratch[y * ww + x]) continue          // too far from the person to be a leak
                val sx = if (half) min(w - 1, x * 2) else x
                if (mask[sy * w + sx]) continue
                val c = px[sy * w + sx]
                val r = Color.red(c); val g = Color.green(c); val b = Color.blue(c)
                if (max(r, max(g, b)) - min(r, min(g, b)) > SPREAD_STRONG) continue
                if (abs(r - v) > TOL_STRONG || abs(g - v) > TOL_STRONG || abs(b - v) > TOL_STRONG) continue
                if (localRange(gray, ww, hh, x, y, FLAT_R) >= FLAT_T) continue
                leak++
            }
        }
        return if (half) leak * 4 else leak     // report in full-res pixel units
    }

    /** Grow [work] outward through [weak] by at most [GEO_ITERS] px (Chebyshev), BFS from every seed. */
    private fun geodesicGrow() {
        var qs = 0; var qe = 0
        for (i in 0 until nw) if (work[i]) { queue[qe++] = i; dist[i] = 0 }
        while (qs < qe) {
            val i = queue[qs++]
            val d = dist[i].toInt()
            if (d >= GEO_ITERS) continue
            val x = i % ww; val y = i / ww
            var dy = -1
            while (dy <= 1) {
                var dx = -1
                while (dx <= 1) {
                    val xx = x + dx; val yy = y + dy
                    if (xx in 0 until ww && yy in 0 until hh) {
                        val j = yy * ww + xx
                        if (!work[j] && weak[j]) { work[j] = true; dist[j] = (d + 1).toByte(); queue[qe++] = j }
                    }
                    dx++
                }
                dy++
            }
        }
    }

    /** Any non-mask pixel not reachable from the border is an interior hole -> part of the person. */
    private fun fillInteriorHoles() {
        java.util.Arrays.fill(visited, false)
        var qe = 0
        fun seed(i: Int) { if (!work[i] && !visited[i]) { visited[i] = true; queue[qe++] = i } }
        for (x in 0 until ww) { seed(x); seed((hh - 1) * ww + x) }
        for (y in 0 until hh) { seed(y * ww); seed(y * ww + ww - 1) }
        var qs = 0
        while (qs < qe) {
            val i = queue[qs++]
            val x = i % ww; val y = i / ww
            if (x > 0) seed(i - 1)
            if (x < ww - 1) seed(i + 1)
            if (y > 0) seed(i - ww)
            if (y < hh - 1) seed(i + ww)
        }
        for (i in 0 until nw) if (!work[i] && !visited[i]) work[i] = true
    }

    private fun dilate(m: BooleanArray, mw: Int, mh: Int, r: Int, tmp: BooleanArray) {
        if (r <= 0) return
        for (y in 0 until mh) { val row = y * mw
            for (x in 0 until mw) { var on = false; var d = -r
                while (d <= r) { val xx = x + d; if (xx in 0 until mw && m[row + xx]) { on = true; break }; d++ }
                tmp[row + x] = on } }
        for (x in 0 until mw) for (y in 0 until mh) { var on = false; var d = -r
            while (d <= r) { val yy = y + d; if (yy in 0 until mh && tmp[yy * mw + x]) { on = true; break }; d++ }
            m[y * mw + x] = on }
    }

    private fun erode(m: BooleanArray, mw: Int, mh: Int, r: Int, tmp: BooleanArray) {
        if (r <= 0) return
        for (y in 0 until mh) { val row = y * mw
            for (x in 0 until mw) { var on = true; var d = -r
                while (d <= r) { val xx = x + d; if (xx < 0 || xx >= mw || !m[row + xx]) { on = false; break }; d++ }
                tmp[row + x] = on } }
        for (x in 0 until mw) for (y in 0 until mh) { var on = true; var d = -r
            while (d <= r) { val yy = y + d; if (yy < 0 || yy >= mh || !tmp[yy * mw + x]) { on = false; break }; d++ }
            m[y * mw + x] = on }
    }

    /** Mask a frame's pixels into a 0/255 ARGB mask-pixel array (the same form mask.mp4 decodes to). */
    fun maskPixelsInto(px: IntArray, v: Int, mkpx: IntArray, out: BooleanArray): Int {
        val c = maskInto(px, v, out)
        if (c < 0) return c
        for (i in 0 until np) mkpx[i] = if (out[i]) -1 else -0x1000000
        return c
    }
}
