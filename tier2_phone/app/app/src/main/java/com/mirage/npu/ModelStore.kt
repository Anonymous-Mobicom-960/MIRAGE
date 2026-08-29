package com.mirage.npu

import android.content.Context
import java.io.File

/**
 * ModelStore - resolves WHERE model files live, with a strict precedence that makes the installed
 * APK upgradeable forever WITHOUT a reinstall:
 *
 *   1. a file pushed into the app's external models dir
 *        /sdcard/Android/data/com.mirage.npu/files/models/           (adb push, no permission needed)
 *   2. the copy bundled in the APK's assets (models/<name>), if the build shipped one.
 *
 * PRIVACY: models are read from local storage only; nothing is fetched (the app has no INTERNET
 * permission, so it couldn't anyway).
 */
object ModelStore {

    const val MODELS_DIR_NAME = "models"

    /** /sdcard/Android/data/com.mirage.npu/files/models - created eagerly so `adb push` targets exist. */
    fun modelsDir(context: Context): File =
        File(context.getExternalFilesDir(null) ?: context.filesDir, MODELS_DIR_NAME).apply { mkdirs() }

    /**
     * Ensure model [name] (e.g. "seg.onnx") exists as a FILE in files/models - a PUSHED copy always
     * wins (upgrade-forever), otherwise it is copied ONCE from the bundled APK asset (models/[name]).
     * Returns the File, or null if neither a pushed copy nor a bundled asset exists. Loading from a
     * real file (not asset bytes) keeps QNN EPContext path-resolution working too.
     */
    fun ensureModel(context: Context, name: String, log: (String) -> Unit = {}): File? {
        val f = File(modelsDir(context), name)
        if (f.isFile && f.length() > 0L) return f
        val has = runCatching { context.assets.list(MODELS_DIR_NAME)?.contains(name) == true }.getOrDefault(false)
        if (!has) return null
        return try {
            log("[model] copying bundled $name -> files/models/ (first run) …")
            context.assets.open("$MODELS_DIR_NAME/$name").use { i -> f.outputStream().use { o -> i.copyTo(o) } }
            f
        } catch (t: Throwable) { log("[model] bundle copy failed for $name (${t.message?.take(60)})"); null }
    }
}
