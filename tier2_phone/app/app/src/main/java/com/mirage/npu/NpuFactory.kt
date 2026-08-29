package com.mirage.npu

import android.content.Context
import java.io.File

/**
 * NpuFactory - builds the model runners (LaMa, MI-GAN, and the on-phone Tier-1 models) for this
 * device RIGHT NOW:
 *
 *   model  : ModelStore precedence - pushed file in files/models/  >  bundled asset  >  none.
 *   engine : OrtRunner EP ladder - QNN-HTP (only in the -Pmirage.enableOrt=true build)
 *                                   > NNAPI (default build, when [wantAccel])  >  CPU EP.
 *
 * Returns null when no model exists or the model cannot be loaded - the caller then falls back to a
 * model-free path, so the app ALWAYS produces a result. Every failure path logs a human-readable
 * reason instead of throwing.
 *
 * This factory lives in the MAIN source set and is identical in both build flavours; the flavour
 * only changes which ONNX Runtime artifact (and hence which EPs) is inside the APK.
 */
object NpuFactory {

    /**
     * Resolve which file to open for an accelerable model, and whether to ask for the NPU.
     *
     * Prefers a precompiled QNN EPContext at `files/models/<stem>_qnn/model.onnx` (Hexagon HTP), else
     * the plain `.onnx` on the CPU EP. Push a context later with no reinstall.
     *
     * 🔴 TWO TRAPS THIS EXISTS TO AVOID.
     * 1. An EPContext `model.onnx` is a ~478-byte wrapper that references `model.bin` by RELATIVE path,
     *    so it MUST be opened BY PATH. Loading it through `fromFile`/`fromBytes` reads the wrapper as
     *    bytes, the `.bin` never resolves, ORT silently falls back to CPU and NOTHING logs an error.
     * 2. `fromBytes`' rung 1 calls `opts.addQnn(backend_path=…)` - the STATIC Microsoft `-qnn` AAR API.
     *    This build uses Qualcomm's PLUGIN EP instead (`registerExecutionProviderLibrary`), so that
     *    factory is not in the binary and `addQnn` always throws. Rung 1 of `fromBytes` is therefore
     *    dead code in BOTH build flavours, and every `fromBytes(…, wantAccel = true)` silently lands on
     *    NNAPI and then CPU. Everything accelerable must go through `fromModelPath`.
     */
    private fun resolveAccel(context: Context, stem: String, plain: File, wantAccel: Boolean): Pair<File, Boolean> {
        if (wantAccel && ORT_FLAVOR == "qnn") {
            val ctx = File(File(ModelStore.modelsDir(context), stem + "_qnn"), "model.onnx")
            if (ctx.exists()) return ctx to true
        }
        return plain to false
    }


    /** Which ORT artifact this APK was built with ("nnapi-cpu" default, "qnn" optional build). */
    val ORT_FLAVOR: String = BuildConfig.ORT_FLAVOR

    /** Build the LaMa neural inpainter IFF files/models/lama.onnx exists (CPU EP; no NPU needed).
     *  Push it once: adb push lama.onnx /sdcard/Android/data/com.mirage.npu/files/models/ */
    /**
     * Compile the Tier-1 SEG model into a Hexagon context, once, on this device.
     *
     * Seg first and seg alone: the op audit found YOLO11n-seg has **zero** ops outside the QNN table and
     * its input is already static [1,3,640,640], so nothing needs freezing or rewriting. RVM needs three
     * graph edits (bake `downsample_ratio`, rewrite 28 `HardSigmoid`, bake the zero recurrent states) and
     * MoveNet's CenterNet decode tail may not place at all - both are separate jobs with their own gates.
     */
    fun compileSegForNpu(context: Context, log: (String) -> Unit): Boolean {
        if (ORT_FLAVOR != "qnn") { log("[qnn-compile] not the QNN build - nothing to do."); return false }
        val plain = ModelStore.ensureModel(context, YoloSeg.MODEL_FILE, log)
            ?: run { log("[qnn-compile] seg.onnx not on device."); return false }
        val out = File(ModelStore.modelsDir(context), "seg_qnn")
        log("[qnn-compile] compiling ${plain.name} for the Hexagon - this takes a while, once.")
        return OrtRunner.compileQnnContext(context, plain.absolutePath, out, log)
    }

    /** Is lama.onnx staged? Lets a caller decide the METHOD without paying the 208 MB load to find out. */
    fun hasLama(context: Context): Boolean =
        File(ModelStore.modelsDir(context), LamaInpainter.MODEL_FILE).exists()

    fun createLamaOrNull(context: Context, log: (String) -> Unit): LamaInpainter? {
        val f = File(ModelStore.modelsDir(context), LamaInpainter.MODEL_FILE)
        if (!f.exists()) return null
        return try {
            log("[lama] loading ${f.name} (${f.length() / 1024 / 1024} MB, CPU EP) …")
            // Timed separately from inference: the ledger's measured Phase-1 FIXED cost is 22.1 s, and
            // reading + preparing a 208 MB fp32 graph is the prime suspect for most of it - far more
            // than the ~2 s the model then spends actually inpainting once per clip.
            val runner = Prof.time(Prof.LAMA_LOAD) { OrtRunner.fromModelPath(context, f.absolutePath, false, log) }
            try { LamaInpainter(runner) } catch (t: Throwable) { runCatching { runner.close() }; throw t }
        } catch (t: Throwable) { log("[lama] load failed (${t.message?.take(80)}); push-pull."); null }
    }

    /** Build the RVM person-matting QuickMask IFF files/models/rvm.onnx exists (CPU EP; no NPU needed).
     *  Push it once: adb push rvm.onnx /sdcard/Android/data/com.mirage.npu/files/models/rvm.onnx */
    fun createRvmOrNull(context: Context, wantAccel: Boolean, log: (String) -> Unit): QuickMask? {
        val plain = ModelStore.ensureModel(context, QuickMask.MODEL_FILE, log) ?: run { log("[quickmask] rvm.onnx unavailable."); return null }
        val (f, accel) = resolveAccel(context, "rvm", plain, wantAccel)
        return try {
            log("[quickmask] loading ${f.name} (${f.length() / 1024} KB, ${if (accel) "QNN-HTP" else "CPU EP"}) …")
            val runner = OrtRunner.fromModelPath(context, f.absolutePath, accel, log)
            try { QuickMask(runner) } catch (t: Throwable) { runCatching { runner.close() }; throw t }
        } catch (t: Throwable) { log("[quickmask] rvm load failed (${t.message?.take(90)})."); null }
    }

    /** YOLO11n-seg (the real Tier-1 guided-seg model) - pushed file or bundled asset. CPU EP. */
    fun createYoloSegOrNull(context: Context, wantAccel: Boolean, log: (String) -> Unit): YoloSeg? {
        val plain = ModelStore.ensureModel(context, YoloSeg.MODEL_FILE, log) ?: run { log("[tier1] seg.onnx unavailable."); return null }
        val (f, accel) = resolveAccel(context, "seg", plain, wantAccel)
        return try {
            log("[tier1] loading ${f.name} (YOLO11n-seg, ${f.length() / 1024} KB, ${if (accel) "QNN-HTP" else "CPU EP"}) …")
            val runner = OrtRunner.fromModelPath(context, f.absolutePath, accel, log)
            try { YoloSeg(runner) } catch (t: Throwable) { runCatching { runner.close() }; throw t }
        } catch (t: Throwable) { log("[tier1] seg load failed (${t.message?.take(90)})."); null }
    }

    /** MoveNet MultiPose (on-phone pose) - pushed file or bundled asset. CPU EP. */
    fun createMoveNetOrNull(context: Context, wantAccel: Boolean, log: (String) -> Unit): MoveNet? {
        val plain = ModelStore.ensureModel(context, MoveNet.MODEL_FILE, log) ?: run { log("[tier1] movenet.onnx unavailable."); return null }
        val (f, accel) = resolveAccel(context, "movenet", plain, wantAccel)
        return try {
            log("[tier1] loading ${f.name} (MoveNet MultiPose, ${f.length() / 1024} KB, ${if (accel) "QNN-HTP" else "CPU EP"}) …")
            val runner = OrtRunner.fromModelPath(context, f.absolutePath, accel, log)
            try { MoveNet(runner) } catch (t: Throwable) { runCatching { runner.close() }; throw t }
        } catch (t: Throwable) { log("[tier1] movenet load failed (${t.message?.take(90)})."); null }
    }

    /** Build the MI-GAN inpainter IFF files/models/migan.onnx exists (else null -> diffusion only).
     *  Push it later with no reinstall: adb push migan.onnx /sdcard/Android/data/com.mirage.npu/files/models/ */
    fun createMiGanOrNull(context: Context, wantAccel: Boolean, log: (String) -> Unit): MiGanNpu? {
        val f = File(ModelStore.modelsDir(context), MiGanNpu.MODEL_FILE)
        if (!f.exists()) return null
        return try {
            log("[inpaint] loading MI-GAN ${f.name} (${f.length() / 1024} KB) ...")
            val runner = OrtRunner.fromBytes(f.readBytes(), wantAccel, log)
            try { MiGanNpu(runner).also { log("[inpaint] MI-GAN ready") } }
            catch (t: Throwable) { runCatching { runner.close() }; throw t }
        } catch (t: Throwable) {
            log("[inpaint] MI-GAN load failed (${t.message?.take(100)}); push-pull only."); null
        }
    }
}
