package com.mirage.npu

import ai.onnxruntime.OnnxTensor
import android.graphics.Bitmap
import android.graphics.Canvas
import android.graphics.Color

/**
 * MoveNet - MoveNet MultiPose Lightning, single-pass multi-person 2D pose (COCO-17) for the ON-PHONE
 * Tier-1 preview. Input NHWC int32 [1,N,N,3] (RGB 0..255, N a multiple of 32); output [1,6,56] = up to
 * 6 instances × (17 keypoints × [y,x,score] + [ymin,xmin,ymax,xmax,box_score]), coords normalised to
 * the padded square. Chosen over the edge's RTMPose-133 (top-down, 124 MB) because it is one forward
 * pass, mobile-native, and 17-COCO is exactly what the skeleton draw needs. The RTMPose-vs-MoveNet
 * gap is measured in the ablation, not hand-waved.
 */
class MoveNet(private val runner: OrtRunner, private val N: Int = 256) : AutoCloseable {

    /** One person: kp[17], each = [x, y, score] in FRAME pixel coords. */
    class Person(val kp: Array<FloatArray>, val boxScore: Float)

    fun detect(frame: Bitmap, minScore: Float = 0.2f): List<Person> {
        val w = frame.width; val h = frame.height; val side = maxOf(w, h)
        val s = N.toFloat() / side
        val sq = Bitmap.createBitmap(N, N, Bitmap.Config.ARGB_8888)
        Canvas(sq).apply {
            drawColor(Color.BLACK)
            val scaled = Bitmap.createScaledBitmap(frame, (w * s).toInt().coerceAtLeast(1), (h * s).toInt().coerceAtLeast(1), true)
            drawBitmap(scaled, 0f, 0f, null); if (scaled !== frame) scaled.recycle()
        }
        val px = IntArray(N * N); sq.getPixels(px, 0, N, 0, 0, N, N); sq.recycle()
        val data = IntArray(N * N * 3)
        for (i in 0 until N * N) { val c = px[i]; data[i * 3] = Color.red(c); data[i * 3 + 1] = Color.green(c); data[i * 3 + 2] = Color.blue(c) }
        val t = runner.intTensor(data, longArrayOf(1, N.toLong(), N.toLong(), 3))
        try {
            runner.run(mapOf("input" to t)).use { res ->
                val out = res.get(0) as OnnxTensor
                val buf = out.floatBuffer ?: error("movenet output not float32")
                val arr = FloatArray(buf.remaining()); buf.get(arr)   // [6*56]
                val people = ArrayList<Person>()
                for (p in 0 until 6) {
                    val base = p * 56
                    val bs = arr[base + 55]
                    if (bs < minScore) continue
                    val kp = Array(17) { k ->
                        // MoveNet returns [y, x, score] normalised to the padded square (frame at top-left).
                        floatArrayOf(arr[base + k * 3 + 1] * side, arr[base + k * 3] * side, arr[base + k * 3 + 2])
                    }
                    people.add(Person(kp, bs))
                }
                return people
            }
        } finally { t.close() }
    }

    override fun close() = runner.close()
    companion object { const val MODEL_FILE = "movenet.onnx" }
}
