package com.mirage.npu

import android.os.SystemClock

/**
 * Prof - sub-stage wall-clock accounting for the phases.
 *
 * WHY THIS EXISTS. Until now `PhaseRunner.phase1Inpaint` wrapped the entire `BackgroundInpaint.run()`
 * in a single `elapsedRealtime` pair, so every per-stage cost in the whole project was a projection:
 * the ledger could say "Phase 1 = 22.1 s fixed + 458 ms/frame" but nobody could say how much of that
 * was video decode vs alignment vs the temporal aggregate vs the composite vs LaMa. Every optimisation
 * argument was therefore unfalsifiable. This turns that into a measurement for the cost of a few
 * counters, and the numbers land in the run's own `EVALS.md` next to the phase total.
 *
 * DESIGN NOTES
 * - Buckets are fixed strings ([DECODE] … [ENCODE]) so the report is stable across runs and diffable.
 * - `elapsedRealtime()` includes deep sleep, which is what we want: this measures WALL CLOCK, the thing
 *   the user waits through, not CPU time.
 * - Nesting is NOT summed: a bucket you open inside another bucket double-counts. Keep them disjoint
 *   and let [unaccounted] absorb the remainder - a large unaccounted value is itself a finding.
 * - Thread-safety: [add] is synchronized, so the row-band worker pool can time its own slices. The
 *   overhead is one uncontended lock per stage-exit, not per pixel.
 * - Zero cost when unused: if `reset()` is never called the maps stay empty and `report()` returns "".
 */
object Prof {

    const val DECODE = "decode"          // MediaMetadataRetriever.getFrameAtIndex + prepare()
    const val TRAJECTORY = "trajectory"  // computeTrajectory (camera motion pre-pass)
    const val ALIGN = "align"            // Aligner.shiftOf / refShiftTo (pyramid SAD)
    const val SAMPLE = "sample"          // aggregatePlate's per-frame sample-collection loop
    const val GATHER = "gather"          // aggregatePlate's per-pixel trimmed-mean loop
    const val DILATE = "dilate"          // dilateInPlace
    const val CORE = "core"              // fillCore bookkeeping (excludes the LaMa call itself)
    const val LAMA_LOAD = "lama_load"    // ORT session create + 208 MB model read
    const val LAMA_INFER = "lama_infer"  // LamaInpainter.inpaint
    const val COMPOSITE = "composite"    // compositeFrames per-frame pixel work (excludes decode/encode)
    const val ENCODE = "encode"          // Mp4Encoder.writeFrame (bitmap -> YUV -> codec)
    const val MASK = "mask"              // fillMaskPixels / HoleMask derivation
    const val YUV = "yuv"                // RGB->YUV colour conversion INSIDE encode (CPU work)
    const val CODEC = "codec"            // dequeue/queue/drain around it (hardware backpressure)
    const val ALIGN_PREP = "align_prep"  // downGrayValid INSIDE align: rescale + getPixels + gray/valid
    const val ALIGN_SAD = "align_sad"    // bestShiftSub INSIDE align: the candidate SAD grid itself

    /**
     * Buckets that are NESTED inside another bucket. They are reported for insight but excluded from
     * the sum, or `unaccounted` would go negative.
     *
     * YUV + CODEC together make up ENCODE, and that split is what caught a real regression
     * (§C.PHONE-9r): only YUV is CPU work, CODEC is hardware backpressure, so a flat CODEC is the
     * control that makes an attribution to YUV safe.
     *
     * ALIGN_PREP + ALIGN_SAD together make up ALIGN, for the same reason. `align` is the largest
     * single bucket in the pipeline (34.9 s of 164.5 s) at 116 ms per shiftOf call, and the arithmetic
     * of the three pyramid levels does not account for that on its own - so which half to attack was
     * a guess until this existed. PREP is per-frame extraction work (a rescale, a getPixels and an
     * elementwise loop); SAD is the candidate search. They have completely different fixes.
     */
    private val NESTED = setOf(YUV, CODEC, ALIGN_PREP, ALIGN_SAD)

    /** Declaration order = report order. Anything not listed still prints, after these. */
    private val ORDER = listOf(
        DECODE, MASK, TRAJECTORY, ALIGN, ALIGN_PREP, ALIGN_SAD, SAMPLE, GATHER, DILATE,
        CORE, LAMA_LOAD, LAMA_INFER, COMPOSITE, ENCODE, YUV, CODEC,
    )

    private val ms = LinkedHashMap<String, Long>()
    private val hits = LinkedHashMap<String, Long>()
    private var t0 = 0L
    private var on = false

    /** Start a fresh accounting window. Call once at the top of a phase. */
    @Synchronized
    fun reset() {
        ms.clear(); hits.clear(); t0 = SystemClock.elapsedRealtime(); on = true
    }

    @Synchronized
    fun add(bucket: String, deltaMs: Long) {
        if (!on) return
        ms[bucket] = (ms[bucket] ?: 0L) + deltaMs
        hits[bucket] = (hits[bucket] ?: 0L) + 1L
    }

    /**
     * Time [body] into [bucket]. Inline + reified-free so the lambda does not allocate on the hot path;
     * the only overhead is two clock reads and one synchronized map update.
     */
    inline fun <T> time(bucket: String, body: () -> T): T {
        val s = SystemClock.elapsedRealtime()
        try {
            return body()
        } finally {
            add(bucket, SystemClock.elapsedRealtime() - s)
        }
    }

    @Synchronized
    fun totalMs(): Long = if (on) SystemClock.elapsedRealtime() - t0 else 0L

    /** Per-bucket ms, in report order, for structured consumers (EVALS json/table). */
    @Synchronized
    fun buckets(): LinkedHashMap<String, Long> {
        val out = LinkedHashMap<String, Long>()
        for (k in ORDER) ms[k]?.let { out[k] = it }
        for ((k, v) in ms) if (!out.containsKey(k)) out[k] = v
        return out
    }

    /**
     * One-line-per-bucket report with % of the phase and the call count, plus an explicit
     * `unaccounted` row. Returns "" when profiling never started.
     */
    @Synchronized
    fun report(frames: Int): String {
        if (!on) return ""
        val total = totalMs().coerceAtLeast(1L)
        val sb = StringBuilder()
        var acc = 0L
        for ((k, v) in buckets()) {
            if (k !in NESTED) acc += v
            val perF = if (frames > 0) v.toDouble() / frames else 0.0
            sb.append("  %-11s %7d ms  %5.1f%%  %8.1f ms/f  x%d\n"
                .format(k, v, 100.0 * v / total, perF, hits[k] ?: 0L))
        }
        val un = total - acc
        sb.append("  %-11s %7d ms  %5.1f%%  %8.1f ms/f\n"
            .format("unaccounted", un, 100.0 * un / total, if (frames > 0) un.toDouble() / frames else 0.0))
        sb.append("  %-11s %7d ms".format("TOTAL", total))
        return sb.toString()
    }

    /** Compact single-line form for the EVALS quality map (e.g. "decode=12840 align=7010 …"). */
    @Synchronized
    fun compact(): String = buckets().entries.joinToString(" ") { "${it.key}=${it.value}" }

    @Synchronized
    fun stop() { on = false }
}
