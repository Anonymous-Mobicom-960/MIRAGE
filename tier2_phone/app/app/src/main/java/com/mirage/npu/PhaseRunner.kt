package com.mirage.npu

import android.content.Context
import java.io.File

/**
 * PhaseRunner - the MIRAGE Tier-2 workflow with per-phase EVALUATIONS (latency + quality).
 *
 *   Phase 1   INPAINT   : masked_video (+ mask, or derived from gray) -> background_reconstructed
 *   Phase 1b  LIGHTMAP  : background_reconstructed -> light_map (the cloud's background conditioning)
 *   Phase 2   COMPOSITE : background + N (synthetic_person_pK, alpha_pK) -> composite + final_output
 *   FULL-AUTO           : 1 -> 1b -> 2, each output feeds the next by canonical filename.
 *
 * Every phase is timed on the device and appends a row to output/EVALS.md so you get real latency
 * (total, ms/frame, fps) and quality signals (inpaint never-seen%, composite person count + halo-safe
 * matte). Latency is THIS device's real on-device number (Snapdragon 8 Elite), not a projection.
 */
object PhaseRunner {

    /**
     * One id per APP PROCESS, stamped on every row of EVALS_LOG.jsonl.
     *
     * An eval campaign runs the same phases many times over different clips. Without an id the rows
     * are indistinguishable and a later analysis cannot tell which Phase 2 belongs to which Phase 1.
     * Process-scoped (not per-phase) is deliberate: it groups the phases the user ran together, which
     * is the unit an eval actually reasons about.
     */
    val RUN_ID: String = java.text.SimpleDateFormat("yyyyMMdd-HHmmss", java.util.Locale.US)
        .format(java.util.Date()) + "-" + (100..999).random()

    fun interface Logger { fun log(line: String) }

    data class PhaseEval(
        val phase: String,
        val ms: Long,
        val frames: Int,
        val quality: LinkedHashMap<String, String>,
        val output: File?,
    ) {
        val msPerFrame: Double get() = if (frames > 0) ms.toDouble() / frames else 0.0
        val fps: Double get() = if (ms > 0) frames * 1000.0 / ms else 0.0
    }

    // ---- timing without Date/Random (SystemClock is fine on device) ----
    private fun now(): Long = android.os.SystemClock.elapsedRealtime()

    fun phase1Inpaint(context: Context, wantAccel: Boolean, cfg: RuntimeConfig, log: Logger): PhaseEval {
        runAccel = wantAccel; runMaxDim = cfg.maxDim
        val t0 = now()
        Prof.reset()   // opened here so the LaMa session build below is inside the accounting window
        // LaMa neural background reconstruction (CPU EP) when lama.onnx is present; else push-pull.
        // LAZY: the 208 MB session is built only if fillCore actually needs it. BackgroundInpaint skips
        // fillCore entirely when the camera revealed every hole pixel (ledger §B.1 records a real run at
        // "100 % of hole from REAL pixels, 0 % neural core" - which used to load 208 MB and never use it),
        // and the >35 %-core anti-hallucination branch takes push-pull without touching LaMa either.
        val r = BackgroundInpaint.run(context, { NpuFactory.createLamaOrNull(context, log::log) }, cfg.maxDim, log::log, if (s2AuditEnabled) cfg.s2AuditStride else 0)
        val ms = now() - t0
        val q = linkedMapOf(
            "method" to r.method,
            "camera" to "${if (r.isStatic) "static/jitter" else "dynamic"} (${"%.1f".format(r.motionPx)}px)",
            "avg_hole_%" to "%.2f".format(r.holePct),
            "hole_from_REAL_%" to "%.1f".format(r.realPct),
            "hole_neural_core_%" to "%.1f".format(r.corePct),
            // A negative leakPct means the auditor never ran. Reporting it as "0.000" was how a null
            // measurement got recorded as a clean privacy result on every staged-mask arm.
            "§2_leak_audit_%" to (if (r.leakPct < 0) "NOT AUDITED" else "%.3f".format(r.leakPct)),
            "§2_audited_frames" to "${r.leakFrames}",
            "§2_audit" to (if (s2AuditEnabled) "on (stride ${cfg.s2AuditStride})" else "off"),
            // The fill-sharpness slider is PIXEL-AFFECTING, so it belongs in the row that gets pasted
            // into the ledger - a quality number whose sharpness setting is unknown is unusable.
            "fill_sharpness" to BackgroundInpaint.describeFillSharpness(),
            "seam_match" to BackgroundInpaint.describeSeamMatch(),
            "workers" to "${Par.WORKERS}",
            "substage_ms" to r.prof,
        )
        val e = PhaseEval("1-Inpaint", ms, r.frames, q, r.output)
        log.log("[eval] Phase 1: ${r.frames}f in ${ms}ms (${"%.1f".format(e.msPerFrame)} ms/f, ${"%.1f".format(e.fps)} fps) · ${r.method} · avg hole ${"%.1f".format(r.holePct)}%")
        return e
    }

    fun phaseLightmap(context: Context, cfg: RuntimeConfig, log: Logger): PhaseEval {
        runMaxDim = cfg.maxDim
        Prof.reset()
        val t0 = now()
        val r = LightmapPhase.run(context, cfg.maxDim, log::log, cfg.lightmapSmall, cfg.lightmapBlurRadius, cfg.lightmapBlurSigma)
        val ms = now() - t0
        val q = linkedMapOf(
            "downscale" to "${r.smallDim}x${r.smallDim}",
            "blur" to "gaussian r=${r.blurRadius}",
            "working_fps" to r.fps.toString(),
            "workers" to "${Par.WORKERS}",
            "substage_ms" to Prof.compact(),
        )
        val e = PhaseEval("1b-Lightmap", ms, r.frames, q, r.output)
        log.log("[eval] Lightmap: ${r.frames}f in ${ms}ms (${"%.1f".format(e.msPerFrame)} ms/f) · down ${r.smallDim}² + gaussian r=${r.blurRadius}")
        return e
    }

    fun phase2Composite(context: Context, cfg: RuntimeConfig, log: Logger): PhaseEval {
        runMaxDim = cfg.maxDim
        Prof.reset()
        val t0 = now()
        val r = NCompositor.run(context, cfg.maxDim, log::log, writeCompositeAlpha, !allowKeyerFallback)
        val ms = now() - t0
        val q = linkedMapOf(
            "persons" to r.persons.toString(),
            // Recorded as a COUNT, not only as prose: a run whose EVALS row says anything other than
            // "N/N" composited at least one layer with a keyer and must not be compared with one
            // that did not (owner decision 2026-08-08; ALPHA_MATTE.md §1 for what the difference is).
            "explicit_alpha" to "${r.explicitAlphas}/${r.persons}" +
                (if (r.allExplicit) "" else "  🔴 DOWNGRADE - keyed layers present"),
            "matte_source" to r.mattes.joinToString(", "),
            "matte" to "halo-safe (binarize + 1px erode)",
            "composite_alpha" to (if (writeCompositeAlpha) "written" else "off"),
            "working_fps" to r.fps.toString(),
            "workers" to "${Par.WORKERS}",
            "substage_ms" to Prof.compact(),
        )
        val e = PhaseEval("2-Composite", ms, r.frames, q, r.output)
        log.log("[eval] Phase 2: ${r.frames}f, ${r.persons} person(s) in ${ms}ms (${"%.1f".format(e.msPerFrame)} ms/f, ${"%.1f".format(e.fps)} fps) · " +
            if (r.allExplicit) "matte ✅ ${r.explicitAlphas}/${r.persons} explicit alpha"
            else "matte 🔴 ${r.explicitAlphas}/${r.persons} explicit - THIS RUN IS A DOWNGRADE")
        // Phase 2 is the LAST phase (30 fps in -> 30 fps out, so nothing runs after the composite):
        // publish its output under the canonical final name so the pipeline output is predictable.
        runCatching { r.output.copyTo(MiragePaths.finalOutput, overwrite = true) }
        return e
    }

    // ⚠️ Phase 1b (LIGHTMAP) has no in-app reader - that is expected, not dead code: `light_map.mp4`
    // is the generator's background conditioning (bg_images -> #62 WanVideoAnimateEmbeds) and
    // `build_cloud_bundle.py` hard-fails without it.

    fun fullAuto(context: Context, wantAccel: Boolean, cfg: RuntimeConfig, log: Logger): List<PhaseEval> {
        log.log("==== FULL PIPELINE: inpaint -> lightmap -> composite ====")
        val evals = ArrayList<PhaseEval>()
        // print the eval TABLE to the terminal after EACH phase (growing), and persist EVALS.md each time
        evals.add(phase1Inpaint(context, wantAccel, cfg, log)); writeEvals(evals, log)
        // Lightmap stays in the pipeline: it produces the cloud's background conditioning.
        // Dropping it here would break build_cloud_bundle.py (see the note above).
        evals.add(phaseLightmap(context, cfg, log)); writeEvals(evals, log)
        evals.add(phase2Composite(context, cfg, log)); writeEvals(evals, log)
        return evals
    }

    /** Set by each phase entry point so the run header can record the config a run actually used. */
    @Volatile var runAccel: Boolean = false
    @Volatile var runMaxDim: Int = 0
    /** §2 leak audit is a VERIFICATION tool (it hunts leftover Tier-1 gray fill the mask missed),
     *  not part of producing output, and it measured 130 ms/frame. Owner-toggled; default OFF. */
    @Volatile var s2AuditEnabled: Boolean = false
    /** Emit composite_alpha.mp4 - a reference artifact only; nothing in the app reads it. A second
     *  encoded stream, and encode is hardware-bound, so it costs real time. Owner-toggled; default OFF. */
    @Volatile var writeCompositeAlpha: Boolean = false
    /**
     * Allow Phase 2 to KEY a character layer that arrived without its `synthetic_alpha_pK.mp4`.
     *
     * 🔴 DEFAULT **OFF** - owner decision 2026-08-08: *"keep alpha as primarily turned on ... alpha
     * must be created from tier 1"*. With it off, a missing alpha throws
     * [NCompositor.MissingAlphaException] before a single frame is decoded, instead of quietly
     * downgrading to a keyer that keys ~99.7 % of a lightmap frame. Turning it on is a deliberate,
     * logged, EVALS-recorded downgrade - not a convenience.
     */
    @Volatile var allowKeyerFallback: Boolean = false
    @Volatile private var headerWritten = false

    private fun ts(): String =
        java.text.SimpleDateFormat("yyyy-MM-dd HH:mm:ss", java.util.Locale.US).format(java.util.Date())

    /**
     * The per-run header, written ONCE per app process, the first time a phase reports.
     *
     * Records the things that decide whether two rows are comparable at all - build, engine, worker
     * count, working resolution, and the actual input files with their sizes. A latency number without
     * these is not a measurement, it is a rumour.
     */
    private fun runHeader(): String {
        val sb = StringBuilder()
        sb.appendLine()
        sb.appendLine("---")
        sb.appendLine()
        sb.appendLine("## RUN $RUN_ID · ${ts()}")
        sb.appendLine()
        val ver = runCatching { BuildConfig.VERSION_NAME }.getOrDefault("?")
        sb.appendLine("- build: `$ver` · ORT flavour `${NpuFactory.ORT_FLAVOR}` · accel requested: **${if (runAccel) "ON" else "OFF"}**")
        sb.appendLine("- device: ${android.os.Build.MODEL} (${android.os.Build.SOC_MODEL ?: "?"}) · Android ${android.os.Build.VERSION.RELEASE}")
        sb.appendLine("- CPU workers: ${Par.WORKERS} of ${Runtime.getRuntime().availableProcessors()} cores · max_dim ${runMaxDim}")
        val inputs = runCatching { MiragePaths.inputDir.listFiles()?.sortedBy { it.name } }.getOrNull().orEmpty()
        if (inputs.isNotEmpty()) {
            sb.appendLine("- inputs:")
            for (f in inputs) sb.appendLine("   - `${f.name}` (${f.length() / 1024} KB)")
        }
        sb.appendLine()
        sb.appendLine("| phase | frames | total ms | ms/frame | fps | output | quality |")
        sb.appendLine("|---|---|---|---|---|---|---|")
        return sb.toString()
    }

    fun writeEvals(evals: List<PhaseEval>, log: Logger) {
        val out = File(MiragePaths.outputDir, "EVALS.md")
        // APPEND-ONLY, SECTIONED BY RUN. This file used to be REWRITTEN on every call, so it only ever
        // held the phases of the current invocation - running phases one button at a time (exactly how
        // an eval campaign is driven) meant each write DESTROYED the previous phase's row, and a
        // separate run erased the one before it entirely. Now each app process opens its own section
        // and every phase appends one row under it, so history accumulates across runs and across
        // separately-triggered phases. Nothing is ever overwritten.
        runCatching {
            MiragePaths.ensureDirs()
            if (!out.exists()) {
                out.writeText(
                    "# MIRAGE Tier-2 phone - per-phase evaluation log\n\n" +
                        "Append-only. Every run appends a section; every phase appends a row. Latency is\n" +
                        "REAL on-device wall-clock, not a projection. Machine-readable twin: `EVALS_LOG.jsonl`.\n"
                )
            }
            if (!headerWritten) { out.appendText(runHeader()); headerWritten = true }
            val e = evals.lastOrNull() ?: return@runCatching
            val q = e.quality.entries.joinToString("; ") { "${it.key}=${it.value}" }
            val outName = e.output?.let { "${it.name} (${it.length() / 1024} KB)" } ?: " - "
            out.appendText(
                "| ${e.phase} | ${e.frames} | ${e.ms} | ${"%.1f".format(e.msPerFrame)} | " +
                    "${"%.1f".format(e.fps)} | $outName | $q |\n"
            )
        }.onFailure { log.log("[eval] WARNING: could not append EVALS.md - ${it.message}") }

        // APPEND-ONLY EVAL LOG (2026-07-26). EVALS.md above is OVERWRITTEN on every call, so it only
        // ever holds the phases of the CURRENT invocation -- run the phases one button at a time (which
        // is exactly how an eval campaign is driven) and each write DESTROYS the previous phase's row.
        // That is how a10's and a15's Phase 2 / 2b numbers ended up existing only in a scrollback buffer
        // with no artifact behind them. This appends every phase result, forever, with a run id and a
        // timestamp, so a campaign cannot lose a measurement it already paid for.
        runCatching {
            val jl = File(MiragePaths.outputDir, "EVALS_LOG.jsonl")
            val ts = java.text.SimpleDateFormat("yyyy-MM-dd'T'HH:mm:ss", java.util.Locale.US)
                .format(java.util.Date())
            // APPEND ONLY THE NEWLY-ADDED ROW. fullAuto() calls writeEvals() after EVERY phase with the
            // CUMULATIVE list, so appending the whole list each time re-writes every earlier phase and
            // a 5-phase run produces 15 rows (1+2+3+4+5). Measured on the first real run. The last
            // element is always the phase that just finished, and a manual single-phase run has exactly
            // one element, so taking the last is correct for both callers.
            for (e in listOfNotNull(evals.lastOrNull())) {
                val q = org.json.JSONObject()
                for ((k, v) in e.quality) q.put(k, v)
                val o = org.json.JSONObject()
                    .put("run_id", RUN_ID)
                    .put("ts", ts)
                    .put("phase", e.phase)
                    .put("frames", e.frames)
                    .put("ms", e.ms)
                    .put("ms_per_frame", e.msPerFrame)
                    .put("fps", e.fps)
                    .put("output", e.output?.name ?: "")
                    .put("output_bytes", e.output?.length() ?: 0L)
                    .put("quality", q)
                jl.appendText(o.toString() + "\n")
            }
        }.onFailure { log.log("[eval] WARNING: could not append EVALS_LOG.jsonl - ${it.message}") }
        // also print the summary to the on-screen terminal
        log.log("==================== EVALS ====================")
        for (e in evals) {
            log.log("· ${e.phase}: ${e.frames}f · ${e.ms}ms · ${"%.1f".format(e.msPerFrame)} ms/f · ${"%.1f".format(e.fps)} fps")
            for ((k, v) in e.quality) log.log("    $k = $v")
        }
        log.log("==============================================")
        log.log("[eval] full table -> ${out.absolutePath}")
    }
}
