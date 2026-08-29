package com.mirage.npu

import android.content.Context
import android.net.Uri
import java.io.File

/**
 * SegNpuGate - compile YOLO11n-seg for the Hexagon, then PROVE it is safe before anything uses it.
 *
 * WHY A GATE AT ALL. Moving a model to the NPU changes its numerics. For a *segmenter* that is not a
 * cosmetic risk: the mask decides what Tier-1 erases, so a mask that shrinks is a PRIVACY regression and
 * a mask that grows is a composite-quality regression. "It looked fine" is not an acceptable standard
 * for that, so this runs both engines over real frames and compares them.
 *
 * THRESHOLDS, and why these numbers. The ledger records RVM-vs-YOLO - two genuinely *different models* - 
 * agreeing at 94.95 % IoU. Two numerics of the *same weights* must be far tighter than two different
 * models, so:
 *  - mean IoU        >= 0.99
 *  - min per-frame   >= 0.97
 *  - mask area delta <= +/-0.5 % of frame   (a shrinking mask leaks, a growing one costs quality - 
 *                                             both directions must be bounded, not just the mean)
 * A frame where BOTH engines find no person is skipped rather than scored as a perfect 1.0, because
 * that would let an all-empty NPU model pass by finding nothing at all.
 *
 * This does NOT switch anything over. It writes the context and reports; `NpuFactory.resolveAccel`
 * picks the context up on the next run only because it now exists on disk. If the gate fails, delete
 * `files/models/seg_qnn/` and the app falls straight back to the CPU EP.
 */
object SegNpuGate {

    const val MEAN_IOU_MIN = 0.99
    const val MIN_IOU_MIN = 0.97
    const val AREA_DELTA_MAX = 0.005   // fraction of frame

    data class Result(
        val compiled: Boolean,
        val frames: Int,
        val meanIou: Double,
        val minIou: Double,
        val maxAreaDelta: Double,
        val passed: Boolean,
        val note: String,
        /** Mean per-frame inference time, CPU EP vs QNN-HTP. The POINT of the migration - proving the
         *  mask is unchanged is necessary but says nothing about whether it was worth doing. */
        val cpuMs: Double = 0.0,
        val npuMs: Double = 0.0,
    )

    /** Frames to score. Enough to be meaningful, few enough to stay a tap-and-wait action. */
    private const val SAMPLES = 24

    fun run(context: Context, log: (String) -> Unit): Result {
        if (NpuFactory.ORT_FLAVOR != "qnn")
            return Result(false, 0, 0.0, 0.0, 0.0, false, "not the QNN build")

        // 1) compile the context (idempotent - skip if one is already there)
        val ctx = File(File(ModelStore.modelsDir(context), "seg_qnn"), "model.onnx")
        val compiled = if (ctx.exists()) {
            log("[seg-gate] seg_qnn context already present - reusing it"); true
        } else NpuFactory.compileSegForNpu(context, log)
        if (!compiled) return Result(false, 0, 0.0, 0.0, 0.0, false, "compile failed - see log")

        // 2) pick frames. Prefer the RAW original: seg is a person detector, and a Tier-1 MASKED clip has
        //    the person replaced by flat grey, so both engines would find nothing and the gate would be
        //    vacuous - the same failure mode as the §2 audit that reported 0.000 % having audited nothing.
        val src = when {
            MiragePaths.originalInput.exists() -> MiragePaths.originalInput
            MiragePaths.maskedVideo.exists() -> MiragePaths.maskedVideo
            else -> return Result(true, 0, 0.0, 0.0, 0.0, false, "no video staged to score against")
        }
        if (src != MiragePaths.originalInput)
            log("[seg-gate] ⚠ scoring against ${src.name} (no raw original staged) - a masked clip may " +
                "contain no detectable person, which would make this gate meaningless")

        var cpu: YoloSeg? = null
        var npu: YoloSeg? = null
        val fs = FrameSource(context, Uri.fromFile(src))
        try {
            cpu = NpuFactory.createYoloSegOrNull(context, false, log) ?: return Result(true, 0, 0.0, 0.0, 0.0, false, "CPU seg failed to load")
            npu = NpuFactory.createYoloSegOrNull(context, true, log) ?: return Result(true, 0, 0.0, 0.0, 0.0, false, "NPU seg failed to load")

            val n = fs.count
            if (n < 1) return Result(true, 0, 0.0, 0.0, 0.0, false, "no frames in ${src.name}")
            val step = maxOf(1, n / SAMPLES)
            var scored = 0
            var iouSum = 0.0
            var iouMin = 1.0
            var areaMax = 0.0
            var bothEmpty = 0
            var cpuNs = 0L; var npuNs = 0L; var timed = 0

            var t = 0
            while (t < n && scored < SAMPLES) {
                val f = fs.frameAt(t, 0)
                if (f == null) { t += step; continue }
                val w = f.width; val h = f.height; val np = w * h
                // warm both engines once before timing anything (first call pays graph finalisation)
                if (timed == 0) { cpu.personMask(f, w, h); npu.personMask(f, w, h) }
                var t0 = System.nanoTime()
                val a = cpu.personMask(f, w, h)
                val t1 = System.nanoTime()
                val b = npu.personMask(f, w, h)
                val t2 = System.nanoTime()
                cpuNs += t1 - t0; npuNs += t2 - t1; timed++
                f.recycle()
                var inter = 0; var union = 0; var ca = 0; var cb = 0
                for (i in 0 until np) {
                    val x = a[i]; val y = b[i]
                    if (x) ca++
                    if (y) cb++
                    if (x || y) { union++; if (x && y) inter++ }
                }
                if (union == 0) { bothEmpty++; t += step; continue }   // never score an empty agreement
                val iou = inter.toDouble() / union
                val dArea = kotlin.math.abs(ca - cb).toDouble() / np
                iouSum += iou
                if (iou < iouMin) iouMin = iou
                if (dArea > areaMax) areaMax = dArea
                scored++
                if (scored % 6 == 0) log("[seg-gate] $scored/$SAMPLES scored (IoU so far ${"%.4f".format(iouSum / scored)})")
                t += step
            }

            if (scored == 0)
                return Result(true, 0, 0.0, 0.0, 0.0, false,
                    "no frame had a person in EITHER engine ($bothEmpty empty) - gate is vacuous, stage a raw clip")

            val cpuMs = if (timed > 0) cpuNs / 1e6 / timed else 0.0
            val npuMs = if (timed > 0) npuNs / 1e6 / timed else 0.0
            log("[seg-gate] inference: CPU EP ${"%.1f".format(cpuMs)} ms/frame vs QNN-HTP " +
                "${"%.1f".format(npuMs)} ms/frame (${"%.2f".format(cpuMs / npuMs.coerceAtLeast(0.001))}x)")
            val mean = iouSum / scored
            val pass = mean >= MEAN_IOU_MIN && iouMin >= MIN_IOU_MIN && areaMax <= AREA_DELTA_MAX
            val note = buildString {
                if (mean < MEAN_IOU_MIN) append("mean IoU below ${MEAN_IOU_MIN}; ")
                if (iouMin < MIN_IOU_MIN) append("a frame fell to ${"%.4f".format(iouMin)}; ")
                if (areaMax > AREA_DELTA_MAX) append("area delta ${"%.3f".format(areaMax * 100)}% exceeds ${AREA_DELTA_MAX * 100}%; ")
                if (isEmpty()) append("within every threshold")
            }
            log("[seg-gate] ${if (pass) "PASS" else "FAIL"} - mean IoU ${"%.4f".format(mean)}, " +
                "min ${"%.4f".format(iouMin)}, max area delta ${"%.3f".format(areaMax * 100)}%, " +
                "$scored frames scored ($bothEmpty skipped as empty)")
            if (!pass) log("[seg-gate] NOT SAFE TO SHIP - delete files/models/seg_qnn/ to fall back to CPU")
            return Result(true, scored, mean, iouMin, areaMax, pass, note, cpuMs, npuMs)
        } finally {
            runCatching { cpu?.close() }; runCatching { npu?.close() }; runCatching { fs.close() }
        }
    }
}
