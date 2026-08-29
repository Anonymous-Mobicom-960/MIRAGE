package com.mirage.npu

import android.graphics.Bitmap
import android.graphics.Color

/**
 * Compositor - the halo-safe alpha + blend math, a faithful Kotlin port of `_halo_alpha` and the
 * blend in t2b3_final_composite.py. REAL, deterministic image math (testable in principle; not
 * exercised in this scaffold because it needs decoded frames on a device).
 *
 * Privacy/quality invariant preserved (V9_PLAN §2 step 10): the matte is re-binarized then 1px-eroded
 * BEFORE it keys the character, so a resampled alpha can never bleed the synthetic
 * character rim outside the true person mask - the halo/margin discipline holds at 30 fps.
 */
object Compositor {

    /**
     * Build a halo-safe per-pixel alpha in [0,1] from a grayscale matte bitmap:
     *   binarize(>127) -> 3x3 erode (no translucent rim) -> 3x3 box blur (1px anti-alias only).
     * Output length == w*h.
     */
    fun haloAlpha(alphaGray: Bitmap, w: Int, h: Int): FloatArray {
        val scaled = if (alphaGray.width == w && alphaGray.height == h) null
                     else Bitmap.createScaledBitmap(alphaGray, w, h, true)
        val src = scaled ?: alphaGray
        val px = scratchInt(w * h)
        src.getPixels(px, 0, w, 0, 0, w, h)
        scaled?.recycle()

        // binarize on the low byte (works whether the matte is gray or already 0/255)
        val bin = scratchByte(w * h)
        Par.range(w * h) { p0, p1 -> for (i in p0 until p1) bin[i] = if ((px[i] and 0xFF) > 127) 1 else 0 }

        val eroded = erode3x3(bin, w, h)      // 3x3 minimum -> shrink the mask by ~1px
        return boxBlur3x3(eroded, w, h)       // light anti-alias, still ~binary
    }

    /**
     * person*alpha + bg*(1-alpha), IN PLACE over [bgInPlace], in pixel space.
     *
     * BIT-IDENTICAL to [blend] - same per-pixel formula, same order, same rounding - but it allocates
     * NOTHING. [blend] called `scaledPixels` twice and built an output array, i.e. three 6.4 MB
     * allocations per layer per frame at 1264², plus a Bitmap; a 2-character clip paid that eight times
     * a frame. Layers now accumulate directly into one running buffer.
     */
    fun blendInto(person: IntArray, bgInPlace: IntArray, alpha: FloatArray, np: Int) {
        Par.range(np) { p0, p1 ->
            for (i in p0 until p1) {
                val a = alpha[i]
                if (a <= 0f) continue                       // untouched background - skip the math
                val ia = 1f - a
                val pc = person[i]; val bc = bgInPlace[i]
                val r = (Color.red(pc) * a + Color.red(bc) * ia).toInt().coerceIn(0, 255)
                val g = (Color.green(pc) * a + Color.green(bc) * ia).toInt().coerceIn(0, 255)
                val bl = (Color.blue(pc) * a + Color.blue(bc) * ia).toInt().coerceIn(0, 255)
                bgInPlace[i] = Color.rgb(r, g, bl)
            }
        }
    }

    /** person*alpha + bg*(1-alpha), per channel. All three inputs are wxh. */
    fun blend(person: Bitmap, bg: Bitmap, alpha: FloatArray, w: Int, h: Int): Bitmap {
        val p = scaledPixels(person, w, h)
        val b = scaledPixels(bg, w, h)
        val out = IntArray(w * h)
        Par.range(w * h) { p0, p1 ->
            for (i in p0 until p1) {
                val a = alpha[i]
                val ia = 1f - a
                val r = (Color.red(p[i]) * a + Color.red(b[i]) * ia).toInt().coerceIn(0, 255)
                val g = (Color.green(p[i]) * a + Color.green(b[i]) * ia).toInt().coerceIn(0, 255)
                val bl = (Color.blue(p[i]) * a + Color.blue(b[i]) * ia).toInt().coerceIn(0, 255)
                out[i] = Color.rgb(r, g, bl)
            }
        }
        return Bitmap.createBitmap(out, w, h, Bitmap.Config.ARGB_8888)
    }

    /**
     * Fallback matte when the cloud gives no synthetic_alpha.mp4: person luma > 8 (character-on-black),
     * mirroring t2b3's `cv2.cvtColor(...) > 8`. Returns a 0/255 grayscale bitmap.
     */
    fun alphaFromPersonLuma(person: Bitmap, w: Int, h: Int, thresh: Int = 8): Bitmap {
        val p = scaledPixels(person, w, h)
        val fg = BooleanArray(w * h)
        Par.range(w * h) { p0, p1 ->
            for (i in p0 until p1) {
                val c = p[i]
                val luma = 0.299 * Color.red(c) + 0.587 * Color.green(c) + 0.114 * Color.blue(c)
                fg[i] = luma > thresh
            }
        }
        // FILL INTERIOR HOLES: dark parts of the character (e.g. a dark shirt/hair on a black bg) fall
        // below the luma threshold and would show the background THROUGH the character. Any background
        // pixel NOT connected to the frame border is an interior hole -> it is the character, fill it.
        fillHoles(fg, w, h)
        // ...and close the see-through notches that OPEN onto the boundary, which fillHoles cannot reach.
        solidify(fg, w, h, solidifyReach(fg, w, h))
        val out = IntArray(w * h)
        Par.range(w * h) { p0, p1 ->
            for (i in p0 until p1) { val v = if (fg[i]) 255 else 0; out[i] = Color.rgb(v, v, v) }
        }
        return Bitmap.createBitmap(out, w, h, Bitmap.Config.ARGB_8888)
    }

    /**
     * SOLIDIFY - close see-through notches that [fillHoles] cannot, because they open onto the matte's
     * boundary rather than being fully enclosed.
     *
     * Measured on the derived-matte path: every see-through defect in the v0.10.2 device run was of this
     * kind (enclosed holes = 0 on every sampled frame; boundary-connected notches = 9.6 % of the matte on
     * average, peaking at 27 %). A morphological close cannot fix them either - sweeping its radius 4→16
     * moved raggedness < 1 % while starting to fatten the figure.
     *
     * The general test is INSIDE-NESS, and it assumes nothing about how the matte was derived: from each
     * background pixel, look outward along 8 directions and count how many reach foreground within
     * [maxDist]. A pixel walled in on at least [minDirs] sides is inside the body whatever the keyer said.
     *
     * [minDirs] = 6 is not a tuned number: a legitimately open gap - between the legs, or between an arm
     * and the torso - is open toward the ground in exactly 3 of the 8 directions (down, down-left,
     * down-right), scoring 5, so a 6-of-8 rule preserves real gaps by construction while filling anything
     * more enclosed. Measured: notches −67 to −76 %, with the matte landing within ~1 point of the person
     * hole that Phase 1 derives completely independently.
     *
     * Cost is O(w·h) per direction (8 running-distance scans), not a per-pixel ray march.
     */
    fun solidify(fg: BooleanArray, w: Int, h: Int, maxDist: Int, minDirs: Int = 6) {
        if (maxDist <= 0) return
        val np = w * h
        val hits = ByteArray(np)
        val dist = IntArray(np)
        val far = Int.MAX_VALUE / 4
        val dirs = arrayOf(
            intArrayOf(0, -1), intArrayOf(0, 1), intArrayOf(-1, 0), intArrayOf(1, 0),
            intArrayOf(-1, -1), intArrayOf(-1, 1), intArrayOf(1, -1), intArrayOf(1, 1),
        )
        for (d in dirs) {
            val dy = d[0]; val dx = d[1]
            // Visit so that the neighbour the ray came from is already resolved: ascending when the ray
            // points at decreasing indices, descending when it points at increasing ones.
            val yRange = if (dy == -1) 0 until h else h - 1 downTo 0
            for (y in yRange) {
                val xRange = if (dx == -1) 0 until w else w - 1 downTo 0
                for (x in xRange) {
                    val i = y * w + x
                    if (fg[i]) { dist[i] = 0; continue }
                    val py = y + dy; val px = x + dx
                    dist[i] = if (py in 0 until h && px in 0 until w) {
                        val pd = dist[py * w + px]
                        if (pd >= far) far else pd + 1
                    } else far
                }
            }
            for (i in 0 until np) if (!fg[i] && dist[i] <= maxDist) hits[i]++
        }
        for (i in 0 until np) if (!fg[i] && hits[i] >= minDirs) fg[i] = true
    }

    /** Reach for [solidify], derived from the matte's own bounding box so it is resolution-independent
     *  and adapts to how large the character is in frame. Returns 0 when there is no foreground. */
    fun solidifyReach(fg: BooleanArray, w: Int, h: Int): Int {
        var minY = Int.MAX_VALUE; var maxY = -1
        for (y in 0 until h) { val row = y * w
            for (x in 0 until w) if (fg[row + x]) { if (y < minY) minY = y; if (y > maxY) maxY = y; break }
        }
        if (maxY < 0) return 0
        return ((maxY - minY) * 0.25).toInt().coerceAtLeast(8)
    }

    /** Flood the TRUE background inward from the border; any unreached background pixel is an interior
     *  hole and is flipped to foreground. Fixes see-through holes inside a character on black. */
    private fun fillHoles(fg: BooleanArray, w: Int, h: Int) {
        val outside = BooleanArray(w * h)
        val q = ArrayDeque<Int>()
        fun seed(i: Int) { if (!fg[i] && !outside[i]) { outside[i] = true; q.add(i) } }
        for (x in 0 until w) { seed(x); seed((h - 1) * w + x) }
        for (y in 0 until h) { seed(y * w); seed(y * w + w - 1) }
        while (q.isNotEmpty()) {
            val i = q.removeFirst(); val x = i % w; val y = i / w
            if (x > 0) seed(i - 1); if (x < w - 1) seed(i + 1)
            if (y > 0) seed(i - w); if (y < h - 1) seed(i + w)
        }
        for (i in fg.indices) if (!fg[i] && !outside[i]) fg[i] = true
    }

    // ---- reusable scratch. Single-threaded by contract: Par parallelises INSIDE these functions,
    //      it never calls them concurrently. Sized-guarded so a resolution change just reallocates.
    //
    //      Each slot is a DISTINCT buffer on purpose. `haloAlpha` holds `px` in sInt and `bin` in sByte
    //      while it calls erode3x3, which holds its own two buffers while it calls boxBlur3x3 - so
    //      sharing one slot between them would alias live data. The slots are named for their owner.
    //
    //      Between them these removed ~16 MB of allocation PER FRAME from Phase 2's matte path
    //      (erode3x3 took two ByteArrays, boxBlur3x3 an IntArray plus a FloatArray, 1264² each), i.e.
    //      roughly 4.8 GB of pure GC churn over a 300-frame clip. Bit-identical: every one of these
    //      buffers is fully written before it is read, so a reused buffer cannot carry a stale value
    //      into a later frame. ----
    private var sInt = IntArray(0)
    private var sByte = ByteArray(0)
    private var sErodeTmp = ByteArray(0)
    private var sErodeOut = ByteArray(0)
    private var sBlurTmp = IntArray(0)
    private var sBlurOut = FloatArray(0)
    private fun scratchInt(n: Int): IntArray {
        if (sInt.size < n) sInt = IntArray(n); return sInt
    }
    private fun scratchByte(n: Int): ByteArray {
        if (sByte.size < n) sByte = ByteArray(n); return sByte
    }

    // ---- internals ----

    private fun scaledPixels(b: Bitmap, w: Int, h: Int): IntArray {
        val s = if (b.width == w && b.height == h) b else Bitmap.createScaledBitmap(b, w, h, true)
        val px = IntArray(w * h)
        s.getPixels(px, 0, w, 0, 0, w, h)
        return px
    }

    /**
     * 3x3 morphological erosion (minimum) on a 0/1 mask, edge-replicated at the border.
     *
     * SEPARABLE + ROW-PARALLEL REWRITE, bit-identical. A 3x3 square structuring element decomposes as
     * 1x3 ⊕ 3x1, and because the border clamp is applied independently per axis, the clamped 3x3
     * neighbourhood is exactly the product of the two clamped 1-D neighbourhoods - so min-of-min equals
     * the original min-over-9. Cost drops from 9 taps (each with two coerceIn calls) to 6 cheap ones.
     */
    private fun erode3x3(bin: ByteArray, w: Int, h: Int): ByteArray {
        // Reused scratch, not fresh arrays. Both passes write EVERY element of their destination
        // before anything reads it (pass 1 writes tmp[row+x] for all x,y; pass 2 writes out[row+x]
        // for all x,y), so the previous frame's contents are unobservable. See the scratch KDoc.
        if (sErodeTmp.size < w * h) sErodeTmp = ByteArray(w * h)
        val tmp = sErodeTmp
        Par.rows(h, w) { y0, y1 ->
            for (y in y0 until y1) {
                val row = y * w
                for (x in 0 until w) {
                    val l = bin[row + if (x > 0) x - 1 else 0].toInt()
                    val c = bin[row + x].toInt()
                    val r = bin[row + if (x < w - 1) x + 1 else w - 1].toInt()
                    tmp[row + x] = if (l == 0 || c == 0 || r == 0) 0 else 1
                }
            }
        }
        if (sErodeOut.size < w * h) sErodeOut = ByteArray(w * h)
        val out = sErodeOut
        Par.rows(h, w) { y0, y1 ->
            for (y in y0 until y1) {
                val row = y * w
                val up = (if (y > 0) y - 1 else 0) * w
                val dn = (if (y < h - 1) y + 1 else h - 1) * w
                for (x in 0 until w) {
                    out[row + x] = if (tmp[up + x].toInt() == 0 || tmp[row + x].toInt() == 0 || tmp[dn + x].toInt() == 0) 0 else 1
                }
            }
        }
        return out
    }

    /**
     * 3x3 box blur of a 0/1 mask -> float [0,1] (1px anti-alias, matches the small gaussian in t2b3).
     *
     * SEPARABLE + ROW-PARALLEL REWRITE, bit-identical. The intermediate sums are INTEGERS, so splitting
     * the 9-tap sum into a horizontal 3-sum followed by a vertical 3-sum is exact - there is no float
     * reassociation anywhere, because the single division by 9f happens once at the end, as before.
     */
    private fun boxBlur3x3(bin: ByteArray, w: Int, h: Int): FloatArray {
        // Reused scratch - same full-overwrite argument as erode3x3. The returned FloatArray is shared
        // too: its single caller (NCompositor's per-layer loop) fully consumes it in `blendInto` and the
        // union-alpha max before the next haloAlpha call, so no two live mattes ever exist at once.
        if (sBlurTmp.size < w * h) sBlurTmp = IntArray(w * h)
        val tmp = sBlurTmp
        Par.rows(h, w) { y0, y1 ->
            for (y in y0 until y1) {
                val row = y * w
                for (x in 0 until w) {
                    tmp[row + x] = bin[row + if (x > 0) x - 1 else 0] +
                        bin[row + x] + bin[row + if (x < w - 1) x + 1 else w - 1]
                }
            }
        }
        if (sBlurOut.size < w * h) sBlurOut = FloatArray(w * h)
        val out = sBlurOut
        Par.rows(h, w) { y0, y1 ->
            for (y in y0 until y1) {
                val row = y * w
                val up = (if (y > 0) y - 1 else 0) * w
                val dn = (if (y < h - 1) y + 1 else h - 1) * w
                for (x in 0 until w) out[row + x] = (tmp[up + x] + tmp[row + x] + tmp[dn + x]) / 9f
            }
        }
        return out
    }
}
