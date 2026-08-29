package com.mirage.npu

import android.content.Context
import org.json.JSONObject
import java.io.File

/**
 * RuntimeConfig - tunable parameters loaded AT RUNTIME from the app's own external files dir, so the
 * installed APK can be re-tuned later with a single `adb push` and NO reinstall:
 *
 *     adb push config.json /sdcard/Android/data/com.mirage.npu/files/config.json
 *
 * (That directory is the app-specific external dir - the app reads/writes it with NO permission, and
 * `adb push` can write it on any stock device. This file is optional: when absent or malformed, the
 * baked-in defaults below apply and the app says so in the log.)
 *
 * Schema (every key optional; unknown keys ignored):
 * {
 *   "max_dim": 480       // longest side in px the pipeline downscales to (64..3840)
 * }
 *
 * PRIVACY: this only ever READS a local file; nothing is written or transmitted.
 */
data class RuntimeConfig(
    val maxDim: Int,
    /** Where the values came from - for honest logs ("config.json" vs "defaults"). */
    val source: String,
    /**
     * Audit every Nth frame for the §2 silhouette-leak gate. 1 = every frame (default).
     *
     * MEASURED 2026-08-04: the auditor costs ~130 ms/frame - 18 % of Phase 1 on a 70-frame clip and
     * 31 % on a 300-frame one - because it runs a full HoleMask derivation per frame purely to
     * populate the state auditLeak needs. Raising this is a real latency win, but it WEAKENS A PRIVACY
     * GATE, so the default stays at full coverage and the choice is the owner's. `§2_audited_frames`
     * always reports the true N, so a sampled run can never be mistaken for a complete one.
     */
    val s2AuditStride: Int = 1,
    /** Lightmap phase (1b): downscale target (px), Gaussian blur radius + sigma for the illumination map.
     *  Defaults MUST track [LightmapPhase] - a literal here silently overrides the phase's own default. */
    val lightmapSmall: Int = LightmapPhase.DEF_SMALL,
    val lightmapBlurRadius: Int = LightmapPhase.DEF_BLUR_RADIUS,
    val lightmapBlurSigma: Float = LightmapPhase.DEF_BLUR_SIGMA,
    /**
     * Phase-1 BACKGROUND FILL sharpness, 0..100 - the same knob the slider on the Phase 1 card drives
     * (see [BackgroundInpaint.setFillSharpness]). Higher = fewer, nearer temporal samples per hole
     * pixel over a longer search window, i.e. a sharper fill made of more nearly-raw real pixels.
     *
     * The APP'S SLIDER WINS: MainActivity persists its position in SharedPreferences and applies that,
     * using this value only as the initial position the first time the app runs. This key exists so a
     * scripted/headless run can pin the setting with one `adb push` and no reinstall.
     *
     * 50 is byte-identical to the pre-2026-08-17 build; 100 is the shipped default.
     */
    val fillSharpness: Int = BackgroundInpaint.SHARP_DEFAULT,
    /**
     * Phase-1 SEAM MATCH, 0..100 - how hard the fill is matched to the REAL pixels immediately
     * around the hole (see [BackgroundInpaint.setSeamMatch]). 0 is the previous whole-frame exposure
     * gain; 100 fits the gain to the surrounding band and adds a smooth correction membrane.
     * Same precedence as [fillSharpness]: the app's slider wins, this is only the first-run position.
     */
    val seamMatch: Int = BackgroundInpaint.SEAM_DEFAULT,
) {
    companion object {
        const val FILE_NAME = "config.json"

        // Baked-in defaults - safe on any device.
        // NATIVE-resolution by default (do NOT downscale): the streaming pipeline only holds a few
        // frames, so there is no memory reason to shrink. 1920 caps only very large (4K) inputs for
        // latency; anything <= 1920px passes through at full resolution -> no quality loss.
        const val DEF_MAX_DIM = 1920

        /** The runtime config path: /sdcard/Android/data/com.mirage.npu/files/config.json */
        fun configFile(context: Context): File =
            File(context.getExternalFilesDir(null) ?: context.filesDir, FILE_NAME)

        /**
         * Load the config, falling back to defaults for anything absent/invalid. NEVER throws:
         * a corrupt config.json must not take down the pipeline - it logs and uses defaults.
         */
        fun load(context: Context, log: (String) -> Unit = {}): RuntimeConfig {
            val f = configFile(context)
            if (!f.exists()) {
                return RuntimeConfig(DEF_MAX_DIM, "defaults")
            }
            return try {
                val o = JSONObject(f.readText())
                val maxDim = o.optInt("max_dim", DEF_MAX_DIM).coerceIn(64, 3840)
                val s2Stride = o.optInt("s2_audit_stride", 1).coerceIn(1, 30)
                val lmSmall = o.optInt("lightmap_small", LightmapPhase.DEF_SMALL).coerceIn(8, 256)
                val lmBlurR = o.optInt("lightmap_blur_radius", LightmapPhase.DEF_BLUR_RADIUS).coerceIn(0, 12)
                val lmBlurS = o.optDouble("lightmap_blur_sigma", LightmapPhase.DEF_BLUR_SIGMA.toDouble())
                    .toFloat().coerceIn(0f, 20f)
                val sharp = o.optInt("fill_sharpness", BackgroundInpaint.SHARP_DEFAULT).coerceIn(0, BackgroundInpaint.SHARP_MAX)
                val seam = o.optInt("seam_match", BackgroundInpaint.SEAM_DEFAULT).coerceIn(0, BackgroundInpaint.SEAM_MAX)
                RuntimeConfig(maxDim, FILE_NAME, s2Stride, lmSmall, lmBlurR, lmBlurS, sharp, seam)
            } catch (t: Throwable) {
                log("[config] ${f.name} unreadable (${t.message}); using defaults.")
                RuntimeConfig(DEF_MAX_DIM, "defaults (bad json)")
            }
        }
    }
}
