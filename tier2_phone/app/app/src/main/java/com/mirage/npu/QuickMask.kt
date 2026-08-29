package com.mirage.npu

import ai.onnxruntime.OnnxTensor
import android.graphics.Bitmap
import android.graphics.Color

/**
 * QuickMask - on-device person masking for a RAW clip, so the app can run end-to-end WITHOUT a Tier-1
 * pass. Robust Video Matting (RVM, mobilenetv3) gives a person alpha per frame; we threshold + dilate
 * it and grey-fill the person, producing the `masked_video.mp4` + `mask.mp4` the Tier-2 pipeline
 * expects (real background, grey person hole).
 *
 * SCOPE: this is a CONVENIENCE for demoing on any phone clip. It is NOT the privacy guarantee - the
 * real anonymization is Tier-1 on the edge device (guided seg + FaceGuard + pose-anon + DP). RVM here
 * runs PER-FRAME (zero recurrent state) on the CPU EP: simplest + most robust, at a small cost in
 * temporal smoothness that the grey fill + downstream inpaint absorb.
 *
 * Contract (rvm_mobilenetv3): inputs src[1,3,H,W] RGB[0,1] + r1i..r4i (recurrent, fed empty) +
 * downsample_ratio[1]; output `pha`[1,1,H,W] = alpha. We run at the frame's own (16-snapped) size.
 */
class QuickMask(private val runner: OrtRunner) : AutoCloseable {

    // Recurrent state + ratio are constant across frames in per-frame mode - allocate once.
    private val r = runner.floatTensor(FloatArray(1), longArrayOf(1, 1, 1, 1))
    private val ratio = runner.floatTensor(floatArrayOf(1f), longArrayOf(1))

    /** Public RVM matte (person alpha 0..1) at [f]'s pixel size - used as the guided pipeline's matte
     *  backstop and for the RVM-vs-YOLO mask ablation. [f] should already be at the working size. */
    fun matte(f: Bitmap): FloatArray = alpha(f)

    /** Person alpha (0..1, row-major w*h) for [f] at f's pixel size. */
    private fun alpha(f: Bitmap): FloatArray {
        val w = f.width; val h = f.height; val plane = w * h
        val px = IntArray(plane); f.getPixels(px, 0, w, 0, 0, w, h)
        val data = FloatArray(3 * plane)
        for (i in 0 until plane) {
            val c = px[i]
            data[i] = Color.red(c) / 255f; data[plane + i] = Color.green(c) / 255f; data[2 * plane + i] = Color.blue(c) / 255f
        }
        val src = runner.floatTensor(data, longArrayOf(1, 3, h.toLong(), w.toLong()))
        try {
            runner.run(mapOf("src" to src, "r1i" to r, "r2i" to r, "r3i" to r, "r4i" to r, "downsample_ratio" to ratio)).use { res ->
                val pha = res.get(1) as OnnxTensor   // outputs: fgr, PHA, r1o..r4o
                val buf = pha.floatBuffer ?: error("rvm 'pha' output was not float32")
                val out = FloatArray(buf.remaining()); buf.get(out); return out
            }
        } finally { src.close() }
    }

    /**
     * Mask [frame] into (masked_video frame, mask frame): the person is grey-filled in the first and
     * white in the second; everything else is the real frame / black. Dimensions are snapped to a
     * multiple of 16 (RVM needs /4; the H.264 encoder rounds to /16 anyway - snap once so both outputs
     * and both encoders stay pixel-aligned).
     */
    fun maskAndFill(frame: Bitmap, thresh: Float = 0.5f, dilate: Int = 3, gray: Int = 122): Pair<Bitmap, Bitmap> {
        val fw = (frame.width / 16 * 16).coerceAtLeast(16)
        val fh = (frame.height / 16 * 16).coerceAtLeast(16)
        val f = if (frame.width == fw && frame.height == fh) frame else Bitmap.createScaledBitmap(frame, fw, fh, true)
        val plane = fw * fh
        val a = alpha(f)
        var m = BooleanArray(plane) { a.getOrElse(it) { 0f } > thresh }
        if (dilate > 0) m = dilateMask(m, fw, fh, dilate)
        val fpx = IntArray(plane); f.getPixels(fpx, 0, fw, 0, 0, fw, fh)
        val g = Color.rgb(gray, gray, gray)
        val maskedPx = IntArray(plane); val maskPx = IntArray(plane)
        for (i in 0 until plane) {
            if (m[i]) { maskedPx[i] = g; maskPx[i] = Color.WHITE }
            else { maskedPx[i] = fpx[i]; maskPx[i] = Color.BLACK }
        }
        if (f !== frame) f.recycle()
        return Bitmap.createBitmap(maskedPx, fw, fh, Bitmap.Config.ARGB_8888) to
            Bitmap.createBitmap(maskPx, fw, fh, Bitmap.Config.ARGB_8888)
    }

    /** Separable box dilation by radius [rad] (grow the person region a few px to swallow the matte edge
     *  + codec bleed, mirroring Tier-1's mask dilate). */
    private fun dilateMask(src: BooleanArray, w: Int, h: Int, rad: Int): BooleanArray {
        val tmp = BooleanArray(src.size); val out = BooleanArray(src.size)
        for (y in 0 until h) for (x in 0 until w) {
            var v = false; var k = -rad
            while (k <= rad) { val xx = x + k; if (xx in 0 until w && src[y * w + xx]) { v = true; break }; k++ }
            tmp[y * w + x] = v
        }
        for (y in 0 until h) for (x in 0 until w) {
            var v = false; var k = -rad
            while (k <= rad) { val yy = y + k; if (yy in 0 until h && tmp[yy * w + x]) { v = true; break }; k++ }
            out[y * w + x] = v
        }
        return out
    }

    override fun close() { runCatching { r.close() }; runCatching { ratio.close() }; runner.close() }

    companion object { const val MODEL_FILE = "rvm.onnx" }
}
