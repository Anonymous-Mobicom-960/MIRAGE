package com.mirage.npu

import android.os.Environment
import java.io.File

/**
 * The MIRAGE Tier-2 shared-folder contract.
 *
 * REAL (not a stub): these paths mirror the Termux flow's `config.py` exactly, so this NPU app is a
 * drop-in for the same "Project MIRAGE" folder. The Termux scripts write:
 *  - input/synthetic_person.mp4     (from the cloud - the AI character)
 *  - input/synthetic_alpha.mp4      (optional character cut-out)
 *  - output/2B2_background_plate.mp4 (the reconstructed REAL background - stays on the phone)
 * ...and t2b3_final_composite.py writes:
 *  - output/final_output.mp4        (the anonymized result, 30 fps)   <-- WE PRODUCE THIS
 *  - output/final_preview.png       (a first-frame eyeball preview)   <-- WE PRODUCE THIS
 *
 * This class resolves that folder under shared external storage. The default location matches
 * PHONE_GUIDE.md: /sdcard/Download/Project MIRAGE . Override [PROJECT_DIR_NAME]/[PARENT_SUBDIR]
 * if the user keeps the folder elsewhere.
 */
object MiragePaths {

    /** Folder name from PHONE_GUIDE.md - must match the Termux flow exactly (spaces included). */
    const val PROJECT_DIR_NAME = "Project MIRAGE"

    /**
     * The folder name used before the 2026-08-29 rename.
     *
     * A phone provisioned before that date still has its clips in the old folder, and those clips
     * belong to the user - a rename in our source tree is no reason to strand them. Consulted ONLY
     * when the current folder does not exist, so a device that has both is never ambiguous.
     */
    const val LEGACY_PROJECT_DIR_NAME = "Project Body Sitara"

    /** Parent under external storage. PHONE_GUIDE puts the project in Download/. */
    const val PARENT_SUBDIR = "Download"

    /** /storage/emulated/0 (a.k.a. /sdcard) - the same tree Termux's ~/storage/shared points at. */
    private val externalRoot: File get() = Environment.getExternalStorageDirectory()

    /**
     * The project folder, preferring the current name and falling back to the pre-rename one.
     *
     * If NEITHER exists this returns the CURRENT-name path, so anything that creates the folder
     * creates it under the new name and a fresh install never resurrects the old one.
     */
    val projectDir: File
        get() {
            val current = File(File(externalRoot, PARENT_SUBDIR), PROJECT_DIR_NAME)
            if (current.isDirectory) return current
            val legacy = File(File(externalRoot, PARENT_SUBDIR), LEGACY_PROJECT_DIR_NAME)
            return if (legacy.isDirectory) legacy else current
        }
    val inputDir: File get() = File(projectDir, "input")
    val outputDir: File get() = File(projectDir, "output")

    // ---- INPUT (read) ----
    val syntheticPerson: File get() = File(inputDir, "synthetic_person.mp4")
    val syntheticAlpha: File get() = File(inputDir, "synthetic_alpha.mp4")   // optional
    val backgroundPlate: File get() = File(outputDir, "2B2_background_plate.mp4")

    // ---- OUTPUT (write) - MUST match the t2b3 contract names ----
    /** Phase 1 publishes the fill colour it detected so Phase 1b need not re-scan the clip for it. */
    val fillColorFile: File get() = File(outputDir, "fill_color.txt")
    val finalOutput: File get() = File(outputDir, "final_output.mp4")
    val finalPreview: File get() = File(outputDir, "final_preview.png")

    // ============== PIPELINE contract (inpaint -> lightmap -> composite) ==============
    // Canonical filenames so each phase auto-feeds the next. Pick buttons in the app copy a chosen
    // file to these names ("rename for the task"); adb push works too.
    // Phase 1 (INPAINT) reads:
    val maskedVideo: File get() = File(inputDir, "masked_video.mp4")   // real bg with GRAY person holes
    val maskVideo: File get() = File(inputDir, "mask.mp4")             // white = person hole
    // The ORIGINAL raw clip the on-phone Tier-1 ran on - the true "before" for the Compare card.
    // Saved by runTier1OnPhone; absent (and Compare falls back to masked_video) when the input was
    // already a Tier-1 masked clip.
    val originalInput: File get() = File(inputDir, "original_input.mp4")
    // ---- TIER-1 (edge) VISUALISATION sidecars - staged by tier2-mobile/tier1_viz.py ----
    // Illustrate the edge stage the phone never sees: the emitted anonymised pose (pose sticks), the
    // identity-collapse (canonicaliser), and a SYNTHETIC canonical face. All drawn over the anonymised
    // frame - no real pixels. The synthetic face is a fixed generic template placed on the silhouette
    // head (no real landmarks - the shipped pipeline zeroes them). Absent files render "not ready".
    val tier1PoseSticks:   File get() = File(inputDir, "tier1_pose_sticks.mp4")
    val tier1Canonical:    File get() = File(inputDir, "tier1_canonical.mp4")
    val tier1SyntheticFace: File get() = File(inputDir, "tier1_synthetic_face.mp4")
    // Phase 1 writes -> Phase 2 reads:
    val backgroundReconstructed: File get() = File(outputDir, "background_reconstructed.mp4")
    // Phase 1b (LIGHTMAP) reads background_reconstructed, writes a low-frequency illumination map.
    // 🔴 `light_map.mp4` has NO in-app reader - that is expected, not dead output. It is the
    // CLOUD's background conditioning:
    //   * _e2e/tools/build_cloud_bundle.py exits non-zero if it is missing, and builds the uploaded
    //     plate masked_video_00002.mp4 from it;
    //   * in the cloud graph it reaches the sampler as bg_images
    //     (VHS_LoadVideo -> Set/Get_light_map -> DrawMaskOnImage -> #62 WanVideoAnimateEmbeds).
    val lightMap: File get() = File(outputDir, "light_map.mp4")
    // ...and the SILHOUETTE-OVER-LIGHTMAP PLATE (2026-07-26): the Tier-1 mask flat-filled on top of the
    // lightmap. This is the same plate the cloud already receives (built off-device by
    // _e2e/tools/build_cloud_bundle.py); producing it here lets the phone hand the cloud a frame whose
    // every pixel is either low-pass lightmap or flat fill - the real scene is in neither.
    val lightMapPlate: File get() = File(outputDir, "light_map_plate.mp4")
    // Phase 2 (COMPOSITE) reads N character layers: synthetic_person_pK.mp4 (+ synthetic_alpha_pK.mp4)
    data class PersonLayer(val index: Int, val person: File, val alpha: File?)
    fun personLayers(): List<PersonLayer> {
        val people = (inputDir.listFiles { f -> f.name.matches(Regex("synthetic_person_p\\d+\\.mp4")) } ?: emptyArray())
            .sortedBy { it.name }
        return people.mapNotNull { p ->
            val k = Regex("p(\\d+)").find(p.name)?.groupValues?.get(1)?.toIntOrNull() ?: return@mapNotNull null
            val a = File(inputDir, "synthetic_alpha_p$k.mp4")
            PersonLayer(k, p, a.takeIf { it.exists() })
        }.sortedBy { it.index }
    }
    // Phase 2 writes (and publishes a copy as finalOutput, above - it is the last phase):
    val compositeVideo: File get() = File(outputDir, "composite.mp4")
    // Phase 2 also writes the EXACT character matte it used, rather than leaving a consumer to
    // re-derive one from |composite − background| across two independent h264 encodes.
    // Nothing in the app reads it. Kept deliberately: the off-device §2 leak audits take the
    // character alpha as an input (ledger §A.1/§B rows "WITH the explicit character alpha"), and
    // re-deriving it by differencing is exactly the error those audits were built to avoid.
    val compositeAlpha: File get() = File(outputDir, "composite_alpha.mp4")

    /** Ensure input/ and output/ exist (mirrors config.require()). */
    fun ensureDirs() {
        inputDir.mkdirs()
        outputDir.mkdirs()
    }

    /** Human-readable summary for the UI / logs. */
    fun describe(): String = buildString {
        appendLine("project : ${projectDir.absolutePath}")
        appendLine("in  person: ${syntheticPerson.name} ${if (syntheticPerson.exists()) "OK" else "MISSING"}")
        appendLine("in  alpha : ${syntheticAlpha.name} ${if (syntheticAlpha.exists()) "OK" else "(optional, absent)"}")
        appendLine("in  plate : ${backgroundPlate.name} ${if (backgroundPlate.exists()) "OK" else "MISSING"}")
        append("out       : ${finalOutput.name}")
    }
}
