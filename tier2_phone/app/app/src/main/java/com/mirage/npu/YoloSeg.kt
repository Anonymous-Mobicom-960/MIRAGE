package com.mirage.npu

import ai.onnxruntime.OnnxTensor
import android.graphics.Bitmap
import android.graphics.Color
import kotlin.math.exp

/**
 * YoloSeg - YOLO11n-seg PERSON segmentation, the REAL Tier-1 seg model (same weights the edge runs).
 *
 * Contract: input `images`[1,3,640,640] RGB[0,1] NCHW; outputs `output0`[1,116,8400] (per anchor:
 * 4 box + 80 class scores + 32 mask coeffs) and `output1`[1,32,160,160] mask prototypes. We UNION
 * every person (class 0) detection's proto mask into one person mask - no NMS needed for a union.
 * Plain resize (not letterbox): the mask maps back to the frame by the same resize, which is fine for
 * a mask (a few px of aspect skew, absorbed by the close+dilate the caller applies).
 *
 * On-device this is the guided pipeline's PRIMARY mask; the caller unions it with the RVM matte
 * (backstop), mirroring the edge's own seg∪matte.
 */
class YoloSeg(private val runner: OrtRunner) : AutoCloseable {
    private val N = 640; private val PN = 160; private val A = 8400

    /** Person mask at [w]x[h] (row-major, true = person), from the union of all person detections. */
    fun personMask(frame: Bitmap, w: Int, h: Int, conf: Float = 0.35f): BooleanArray {
        val img = Bitmap.createScaledBitmap(frame, N, N, true)
        val px = IntArray(N * N); img.getPixels(px, 0, N, 0, 0, N, N); if (img !== frame) img.recycle()
        val plane = N * N; val data = FloatArray(3 * plane)
        for (i in 0 until plane) {
            val c = px[i]; data[i] = Color.red(c) / 255f; data[plane + i] = Color.green(c) / 255f; data[2 * plane + i] = Color.blue(c) / 255f
        }
        val t = runner.floatTensor(data, longArrayOf(1, 3, N.toLong(), N.toLong()))
        val union = BooleanArray(PN * PN)
        try {
            runner.run(mapOf("images" to t)).use { res ->
                val o0 = floats(res.get(0) as OnnxTensor)   // [116*8400], channel-major
                val o1 = floats(res.get(1) as OnnxTensor)   // [32*160*160]
                val confCh = 4 * A                          // person = class 0 -> channel 4
                val sc = PN.toFloat() / N
                val protoPlane = PN * PN
                for (a in 0 until A) {
                    if (o0[confCh + a] < conf) continue
                    val cx = o0[a]; val cy = o0[A + a]; val bw = o0[2 * A + a]; val bh = o0[3 * A + a]
                    val x1 = ((cx - bw / 2) * sc).toInt().coerceIn(0, PN - 1)
                    val y1 = ((cy - bh / 2) * sc).toInt().coerceIn(0, PN - 1)
                    val x2 = ((cx + bw / 2) * sc).toInt().coerceIn(0, PN)
                    val y2 = ((cy + bh / 2) * sc).toInt().coerceIn(0, PN)
                    val coeff = FloatArray(32) { o0[(84 + it) * A + a] }
                    for (yy in y1 until y2) for (xx in x1 until x2) {
                        val p = yy * PN + xx
                        if (union[p]) continue
                        var s = 0f; for (j in 0 until 32) s += coeff[j] * o1[j * protoPlane + p]
                        if (1f / (1f + exp(-s)) > 0.5f) union[p] = true
                    }
                }
            }
        } finally { t.close() }
        val out = BooleanArray(w * h)                        // nearest-neighbour resize 160 -> frame
        for (yy in 0 until h) { val sy = yy * PN / h; val row = sy * PN
            for (xx in 0 until w) out[yy * w + xx] = union[row + xx * PN / w] }
        return out
    }

    private fun floats(t: OnnxTensor): FloatArray {
        val b = t.floatBuffer ?: error("yolo output not float32"); val a = FloatArray(b.remaining()); b.get(a); return a
    }

    override fun close() = runner.close()
    companion object { const val MODEL_FILE = "seg.onnx" }
}
