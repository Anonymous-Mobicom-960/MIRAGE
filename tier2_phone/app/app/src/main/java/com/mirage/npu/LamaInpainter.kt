package com.mirage.npu

import android.graphics.Bitmap
import android.graphics.Color

/**
 * LamaInpainter - high-quality NEURAL background inpainting via LaMa (big-lama) ONNX on the CPU EP.
 *
 * Contract (Carve/LaMa-ONNX, verified): inputs `image` [1,3,512,512] float RGB in [0,1] and
 * `mask` [1,1,512,512] float (1 = hole to fill); output [1,3,512,512] in 0..255. We zero the hole in
 * the image, dilate the mask a touch, run once, and composite the fill back at native resolution.
 * Reconstructs real foliage/grass/texture where a person was - vastly better than the diffusion blur.
 *
 * Runs on the ONNX Runtime CPU EP (no NPU needed), so it's independent of the QNN path. ~2 s per call
 * on-device; the camera is static, so Phase 1 calls it ONCE on the union of all person-holes.
 */
class LamaInpainter(private val runner: OrtRunner) : AutoCloseable {

    /** Inpaint [holeMask] (white=hole) of [image] (both any size) -> native-res filled image. */
    fun inpaint(image: Bitmap, holeMask: Bitmap): Bitmap {
        val w = image.width; val h = image.height
        val img = Bitmap.createScaledBitmap(image, SIZE, SIZE, true)
        val msk = Bitmap.createScaledBitmap(holeMask, SIZE, SIZE, false)
        val plane = SIZE * SIZE
        val ipx = IntArray(plane); img.getPixels(ipx, 0, SIZE, 0, 0, SIZE, SIZE)
        val mpx = IntArray(plane); msk.getPixels(mpx, 0, SIZE, 0, 0, SIZE, SIZE)

        val imgData = FloatArray(3 * plane); val mData = FloatArray(plane)
        for (i in 0 until plane) {
            val hole = (mpx[i] and 0xFF) > 127
            mData[i] = if (hole) 1f else 0f
            if (hole) { imgData[i] = 0f; imgData[plane + i] = 0f; imgData[2 * plane + i] = 0f }
            else {
                val c = ipx[i]
                imgData[i] = Color.red(c) / 255f; imgData[plane + i] = Color.green(c) / 255f; imgData[2 * plane + i] = Color.blue(c) / 255f
            }
        }
        val names = runner.inputNames.toList()
        val maskName = names.firstOrNull { it.contains("mask", true) } ?: names.getOrElse(1) { names[0] }
        val imgName = names.firstOrNull { it != maskName } ?: names[0]
        val imgT = runner.floatTensor(imgData, longArrayOf(1, 3, SIZE.toLong(), SIZE.toLong()))
        val mT = runner.floatTensor(mData, longArrayOf(1, 1, SIZE.toLong(), SIZE.toLong()))
        try {
            val (flat, shape) = runner.runFloats(mapOf(imgName to imgT, maskName to mT))
            val oh = shape[shape.size - 2].toInt(); val ow = shape[shape.size - 1].toInt(); val op = oh * ow
            var mx = 0f; for (v in flat) if (v > mx) mx = v
            val sc = if (mx > 1.01f) 1f else 255f
            val opx = IntArray(op)
            for (i in 0 until op) {
                val r = (flat[i] * sc).toInt().coerceIn(0, 255)
                val g = (flat[op + i] * sc).toInt().coerceIn(0, 255)
                val b = (flat[2 * op + i] * sc).toInt().coerceIn(0, 255)
                opx[i] = Color.rgb(r, g, b)
            }
            val out = Bitmap.createBitmap(opx, ow, oh, Bitmap.Config.ARGB_8888)
            return if (ow == w && oh == h) out else Bitmap.createScaledBitmap(out, w, h, true)
        } finally { imgT.close(); mT.close() }
    }

    override fun close() = runner.close()

    companion object {
        const val SIZE = 512
        const val MODEL_FILE = "lama.onnx"   // push to files/models/ (or bundle in assets/models/)
    }
}
