package com.mirage.npu

import android.graphics.Bitmap
import android.graphics.Color

// Bitmap <-> NCHW float helpers (RGB, normalised to [0,1]).
//
// These are generic tensor-packing utilities, not tied to any one model. They previously sat at the
// bottom of a model-specific file, purely because that was the first place to need them; when that
// stage was removed they went with it and MiGanNpu, which also uses them, stopped compiling. They
// live in their own file now so no future deletion can take an unrelated consumer down with it.

internal fun bitmapToNchw(bmp: Bitmap): FloatArray {
    val w = bmp.width; val h = bmp.height
    val px = IntArray(w * h)
    bmp.getPixels(px, 0, w, 0, 0, w, h)
    val out = FloatArray(3 * w * h)
    val plane = w * h
    for (i in 0 until plane) {
        val c = px[i]
        out[i] = Color.red(c) / 255f
        out[plane + i] = Color.green(c) / 255f
        out[2 * plane + i] = Color.blue(c) / 255f
    }
    return out
}

internal fun nchwToBitmap(flat: FloatArray, shape: LongArray): Bitmap {
    val h = shape[shape.size - 2].toInt()
    val w = shape[shape.size - 1].toInt()
    val plane = w * h
    val px = IntArray(plane)
    for (i in 0 until plane) {
        val r = (flat[i].coerceIn(0f, 1f) * 255f).toInt()
        val g = (flat[plane + i].coerceIn(0f, 1f) * 255f).toInt()
        val b = (flat[2 * plane + i].coerceIn(0f, 1f) * 255f).toInt()
        px[i] = Color.rgb(r, g, b)
    }
    return Bitmap.createBitmap(px, w, h, Bitmap.Config.ARGB_8888)
}
