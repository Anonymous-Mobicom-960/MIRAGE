package com.mirage.npu

import ai.onnxruntime.OnnxTensor
import android.content.Context
import android.graphics.Bitmap

/**
 * MiGanNpu - OPTIONAL MI-GAN-512 inpaint via ONNX Runtime (QNN/NNAPI/CPU EP ladder).
 *
 * ⚠️ STUB (not wired into any pipeline). MI-GAN
 * inpaint (V9_PLAN §2 step 9 / §9) is the residual background-hole fill in t2b2 and would only run
 * once per clip. It is scaffolded here so a developer can also move 2B2's hole-fill on-device, but
 * nothing calls it - 2B2 still runs in the Termux flow and produces the plate this app consumes.
 *
 * The MI-GAN ONNX I/O contract (image + mask inputs, inpainted-image output, size, normalization)
 * MUST be read off the real MI-GAN-512.onnx and filled into the constants below. Nothing here is
 * tested - it is a device + real-model task. The model loads from the runtime models dir
 * (files/models/), so it too can be pushed later without a reinstall.
 */
class MiGanNpu(private val runner: OrtRunner) : AutoCloseable {

    /**
     * Inpaint [image] where [mask] is white (hole). Returns the filled image.
     * STUB: shapes/normalization/mask polarity are placeholders - verify against the real model.
     */
    fun inpaint(image: Bitmap, mask: Bitmap): Bitmap {
        val img = ensureSize(image)
        val msk = ensureSize(mask)
        val imgT: OnnxTensor = runner.floatTensor(bitmapToNchw(img), longArrayOf(1, 3, SIZE.toLong(), SIZE.toLong()))
        // MI-GAN typically takes a single-channel mask; this scaffold passes a 1x1xHxW plane.
        val maskT: OnnxTensor = runner.floatTensor(maskPlane(msk), longArrayOf(1, 1, SIZE.toLong(), SIZE.toLong()))
        try {
            val inputs = HashMap<String, OnnxTensor>().apply {
                put(runner.inputNames.elementAt(0), imgT)      // <- STUB: confirm input order/names
                put(runner.inputNames.elementAt(1), maskT)
            }
            val (flat, shape) = runner.runFloats(inputs)
            val out = nchwToBitmap(flat, shape)
            return if (out.width == image.width && out.height == image.height) out
            else Bitmap.createScaledBitmap(out, image.width, image.height, true)
        } finally {
            imgT.close(); maskT.close()
        }
    }

    private fun ensureSize(b: Bitmap): Bitmap =
        if (b.width == SIZE && b.height == SIZE) b else Bitmap.createScaledBitmap(b, SIZE, SIZE, true)

    private fun maskPlane(mask: Bitmap): FloatArray {
        val px = IntArray(SIZE * SIZE)
        mask.getPixels(px, 0, SIZE, 0, 0, SIZE, SIZE)
        val out = FloatArray(SIZE * SIZE)
        for (i in px.indices) out[i] = if ((px[i] and 0xFF) > 127) 1f else 0f
        return out
    }

    override fun close() = runner.close()

    companion object {
        const val MODEL_FILE = "migan.onnx"   // push to files/models/ (runtime dir; no reinstall)
        const val SIZE = 512                  // the MI-GAN-512 working square (matches config.py MIGAN_SIZE)

        fun fromModelsDirOrNull(context: Context): MiGanNpu? = runCatching {
            val f = java.io.File(ModelStore.modelsDir(context), MODEL_FILE)
            MiGanNpu(OrtRunner.fromFile(f, wantAccel = true))
        }.getOrNull()
    }
}
