package com.mirage.npu

import android.content.Context
import android.graphics.Bitmap
import android.graphics.Color
import android.net.Uri
import java.io.File

/**
 * NCompositor (Phase 2) - composite N character layers over the reconstructed background at working fps.
 *
 * Input  : output/background_reconstructed.mp4  (from Phase 1; or a plate)
 *          input/synthetic_person_pK.mp4  (+ optional input/synthetic_alpha_pK.mp4) for K = 1..N
 * Output : output/composite.mp4  (all characters on the real background)
 *          output/composite_alpha.mp4  (the union matte actually used - a reference artifact)
 *
 * 🔴 THE EXPLICIT ALPHA IS THE PRIMARY PATH, AND A MISSING ONE IS AN ERROR (owner, 2026-08-08:
 * *"keep alpha as primarily turned on ... alpha must be created from tier 1"*).
 * [requireExplicitAlpha] defaults to **true**: every layer must arrive with its own
 * `synthetic_alpha_pK.mp4`, or Phase 2 REFUSES to run and names the file that is missing. The keyers
 * below are no longer a silent safety net - they are an opt-out the operator has to take
 * deliberately (Phase 2 card → "allow keyer fallback"), and every run that takes it says so in the
 * log, in EVALS, and in the returned [Result].
 *
 * Why this is not over-strict: an explicit matte is the best AND the cheapest path - see-through
 * **9.57 % → 0.37 %**, composite **531 → 328 ms/f**, §2 leak audit **0.158 % → 0.000 %**
 * (`tier2-mobile/ALPHA_MATTE.md` §1). A silent downgrade trades all of that away without telling
 * anyone, and it did: on a lightmap plate the luma matte keys ~99.7 % of the frame, i.e. it pastes
 * the whole character clip over the real background and the output looks like the render failed.
 *
 * MATTE SELECTION (per layer, decided once from frame 0 and logged):
 *   1. an explicit `synthetic_alpha_pK.mp4` - the ONLY path when [requireExplicitAlpha] is on;
 *   2. else, if the character sits on a LIGHTMAP background, [MosaicKeyer] keys it out of the mosaic;
 *   3. else, the classic character-on-black luma matte ([Compositor.alphaFromPersonLuma]).
 *
 * Whatever the source, the matte is finished by [Compositor.haloAlpha] (binarize → 1 px erode → 3×3 AA),
 * which is the audited privacy invariant: the synthetic character's rim can never key outside the true
 * person mask.
 *
 * PRIVACY: all on-device; the real background stays on the phone.
 */
object NCompositor {

    fun interface Logger { fun log(line: String) }

    /**
     * @param explicitAlphas how many of [persons] layers used their own `synthetic_alpha_pK.mp4`.
     *   `explicitAlphas < persons` means at least one layer was keyed - a DOWNGRADE, and the caller
     *   must surface it as one.
     */
    data class Result(
        val output: File, val frames: Int, val persons: Int, val fps: Int,
        val mattes: List<String>, val explicitAlphas: Int,
    ) {
        val allExplicit: Boolean get() = explicitAlphas == persons
    }

    /** Thrown instead of silently keying when a layer has no explicit alpha. */
    class MissingAlphaException(val missing: List<String>) : IllegalStateException(
        "🔴 PHASE 2 REFUSED - ${missing.size} character layer(s) have NO EXPLICIT ALPHA: " +
            missing.joinToString(", ") + ". The explicit matte is the PRIMARY path (owner decision " +
            "2026-08-08) and is created from Tier-1; without it Phase 2 would fall back to a keyer " +
            "that keys ~99.7 % of a lightmap frame, i.e. paste the whole character clip over the " +
            "real background. Export the alpha with the character (see ALPHA_MATTE.md §2), or tick " +
            "'allow keyer fallback' on the Phase 2 card to accept the downgrade deliberately."
    )

    /**
     * @param writeAlpha emit `composite_alpha.mp4`, the union matte actually composited with.
     *   Nothing in the app reads it; it
     *   survives purely as a reference artifact for the `mobile_test_cases` bundles. It is a SECOND
     *   encoded stream, and encode is hardware-bound: measured ~17.1 s of Phase 2's 43.0 s on a 300-frame
     *   clip. Off by default; the Phase 2 card has a toggle for when you are rebuilding reference bundles.
     * @param requireExplicitAlpha refuse to composite a layer that has no `synthetic_alpha_pK.mp4`.
     *   **Default true.** Set false only to deliberately accept the keyer downgrade.
     */
    fun run(
        context: Context, maxDim: Int, log: Logger,
        writeAlpha: Boolean = false, requireExplicitAlpha: Boolean = true,
    ): Result {
        val bgF = MiragePaths.backgroundReconstructed
        require(bgF.exists()) { "missing ${bgF.name} - run Phase 1 (Inpaint) first" }
        val layers = MiragePaths.personLayers()
        require(layers.isNotEmpty()) { "no synthetic_person_pK.mp4 in input/ (need at least one character)" }
        MiragePaths.ensureDirs()

        // ---- REQUIREMENT (A): the explicit alpha is PRIMARY, and its absence is LOUD -----------
        // Checked BEFORE anything is opened or decoded, so a missing sidecar costs no device time
        // (§B.57: 429 s of phone time went into a composite that had already lost).
        val missing = layers.filter { it.alpha == null }.map { "synthetic_alpha_p${it.index}.mp4" }
        if (missing.isNotEmpty()) {
            log.log("[composite] 🔴🔴 MISSING EXPLICIT ALPHA for ${missing.size} of ${layers.size} " +
                "layer(s): ${missing.joinToString(", ")}")
            if (requireExplicitAlpha) throw MissingAlphaException(missing)
            log.log("[composite] ⚠⚠ KEYER FALLBACK ALLOWED BY CONFIG - this run is a DOWNGRADE, not " +
                "a normal run. Measured cost of keying instead of an explicit matte: see-through " +
                "0.37 % → 9.57 %, composite 328 → 531 ms/f, §2 leak 0.000 % → 0.158 %. Do not pair " +
                "any quality or privacy number from this run with an explicit-alpha one.")
        }

        // Open every FrameSource INSIDE the try, tracked in a list, so a mid-construction failure (e.g. a
        // corrupt character clip) still closes the ones already opened.
        val sources = mutableListOf<FrameSource>()
        try {
            val bg = FrameSource(context, Uri.fromFile(bgF)).also { sources.add(it) }
            val persons = layers.map { FrameSource(context, Uri.fromFile(it.person)).also { fs -> sources.add(fs) } }
            val alphas = layers.map { l -> l.alpha?.let { a -> FrameSource(context, Uri.fromFile(a)).also { fs -> sources.add(fs) } } }
            var n = bg.count
            for (p in persons) n = minOf(n, p.count)
            require(n >= 1) { "no overlapping frames between background and characters" }
            val f0 = bg.frameAt(0, maxDim) ?: throw IllegalStateException("cannot decode background")
            val w = f0.width; val h = f0.height; val np = w * h

            for ((k, p) in persons.withIndex()) {
                if (p.count != n) log.log("[composite] note: p${layers[k].index} has ${p.count} frames, using the overlapping $n")
            }

            // ---- decide each layer's matte source ONCE, from frame 0 ----
            val px = IntArray(np)
            val keyers = arrayOfNulls<MosaicKeyer.Keyer>(layers.size)
            val how = ArrayList<String>(layers.size)
            var nExplicit = 0
            for (k in layers.indices) {
                if (alphas[k] != null) { how.add("p${layers[k].index}: EXPLICIT ALPHA"); nExplicit++; continue }
                val pf = persons[k].frameAt(0, maxDim)
                if (pf == null) { how.add("p${layers[k].index}: 🔴 FALLBACK luma (undecodable frame 0)"); continue }
                scaledPixels(pf, w, h, px)
                when (val mode = MosaicKeyer.detectMode(px, w, h)) {
                    0 -> how.add("p${layers[k].index}: 🔴 FALLBACK luma (character-on-black)")
                    -1 -> {
                        how.add("p${layers[k].index}: 🔴 FALLBACK luma (no alpha, no mosaic detected)")
                        log.log("[composite] ⚠ p${layers[k].index}: no synthetic_alpha_pK.mp4 and the background is " +
                            "neither black nor a lightmap mosaic - the luma matte will likely key the whole frame. " +
                            "Export synthetic_alpha_p${layers[k].index}.mp4 from the cloud for a correct composite.")
                    }
                    else -> {
                        keyers[k] = MosaicKeyer.Keyer(w, h, mode)
                        how.add("p${layers[k].index}: 🔴 FALLBACK lightmap keyer (grid ${mode}²)")
                    }
                }
            }
            // ONE line that cannot be misread. The tell to look for on device is the count: anything
            // other than "N/N EXPLICIT" means at least one alpha did not land, and every fallback
            // label carries a 🔴 so it is visible without parsing the sentence.
            val verdict = if (nExplicit == layers.size) "✅ ${nExplicit}/${layers.size} EXPLICIT"
                          else "🔴 ${nExplicit}/${layers.size} EXPLICIT - ${layers.size - nExplicit} KEYED (DOWNGRADE)"
            log.log("[composite] $n frames @ ${w}x$h · ${layers.size} character(s) · MATTE $verdict - ${how.joinToString("; ")}")

            val out = MiragePaths.compositeVideo
            val alphaOut = MiragePaths.compositeAlpha
            val fps = bg.srcFps
            var firstPx: IntArray? = null
            val matteBuf = BooleanArray(np)
            val unionAlpha = FloatArray(np)
            val alphaPx = if (writeAlpha) IntArray(np) else null
            // The running composite now lives in PIXEL SPACE for the whole frame: one getPixels for the
            // background, layers blended in place, one encode straight from the array. Previously each
            // layer round-tripped through a Bitmap (blend allocated 3 x 6.4 MB + a Bitmap per layer) and
            // the frame was unwrapped again by the encoder.
            val curPx = IntArray(np)
            val personPx = IntArray(np)

            if (!writeAlpha) log.log("[composite] composite_alpha OFF - one encoder instead of two")
            // decode was 8 524 ms = 40 % of this phase, across THREE streams read strictly in step.
            // Prefetch each one; started here, AFTER the frame-0 matte probe above, so the first
            // prefetched index is still 0. Bit-identical (see FrameSource.startPrefetch).
            bg.startPrefetch(n, maxDim)
            for (p in persons) p.startPrefetch(n, maxDim)
            for (a in alphas) a?.startPrefetch(n, maxDim)
            Mp4Encoder(out, fps).use { enc ->
                val aenc = if (writeAlpha) Mp4Encoder(alphaOut, fps) else null
                try {
                    for (t in 0 until n) {
                        val bgBmp = Prof.time(Prof.DECODE) { bg.frameAt(t, maxDim) } ?: continue
                        Prof.time(Prof.DECODE) { scaledPixels(bgBmp, w, h, curPx) }
                        if (writeAlpha) java.util.Arrays.fill(unionAlpha, 0f)
                        for (k in layers.indices) {
                            val person = Prof.time(Prof.DECODE) { persons[k].frameAt(t, maxDim) } ?: continue
                            val keyer = keyers[k]
                            // ONE getPixels of the character, reused by both the keyer and the blend
                            // (the keyer path used to take its own copy).
                            Prof.time(Prof.DECODE) { scaledPixels(person, w, h, personPx) }
                            val alphaBmp: Bitmap = Prof.time(Prof.DECODE) { alphas[k]?.frameAt(t, maxDim) }
                                ?: if (keyer != null) keyer.matteBitmap(personPx, matteBuf)
                                   else Compositor.alphaFromPersonLuma(person, w, h)
                            val af = Prof.time(Prof.MASK) { Compositor.haloAlpha(alphaBmp, w, h) }
                            Prof.time(Prof.COMPOSITE) { Compositor.blendInto(personPx, curPx, af, np) }
                            if (writeAlpha) Par.range(np) { p0, p1 ->
                                for (i in p0 until p1) if (af[i] > unionAlpha[i]) unionAlpha[i] = af[i]
                            }
                        }
                        if (t == 0) firstPx = curPx.copyOf()
                        Prof.time(Prof.ENCODE) { enc.writeFrame(curPx, w, h) }
                        if (aenc != null && alphaPx != null) {
                            Par.range(np) { p0, p1 ->
                                for (i in p0 until p1) {
                                    val v = (unionAlpha[i] * 255f).toInt().coerceIn(0, 255)
                                    alphaPx[i] = Color.rgb(v, v, v)
                                }
                            }
                            Prof.time(Prof.ENCODE) { aenc.writeFrame(alphaPx, w, h) }
                        }
                        if (t % 20 == 0 || t == n - 1) log.log("[composite] $t/$n frames …")
                    }
                } finally { aenc?.close() }
            }
            val firstComposite = firstPx?.let { Bitmap.createBitmap(it, w, h, Bitmap.Config.ARGB_8888) }
            VideoIo.savePng(firstComposite ?: f0, File(out.parentFile, "composite_preview.png"))
            log.log("[composite] done -> ${out.name} ($n frames @ $fps fps, ${layers.size} chars, " +
                "$nExplicit/${layers.size} explicit alpha)" + if (writeAlpha) " + ${alphaOut.name}" else "")
            return Result(out, n, layers.size, fps, how, nExplicit)
        } finally {
            sources.forEach { runCatching { it.close() } }
        }
    }

    private fun scaledPixels(b: Bitmap, w: Int, h: Int, dst: IntArray) {
        val s = if (b.width == w && b.height == h) b else Bitmap.createScaledBitmap(b, w, h, true)
        s.getPixels(dst, 0, w, 0, 0, w, h)
        if (s !== b) s.recycle()
    }
}
