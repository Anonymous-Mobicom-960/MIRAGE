package com.mirage.npu

import android.content.Context
import android.graphics.Bitmap
import android.graphics.Color
import android.net.Uri
import java.io.File
import kotlin.math.exp
import kotlin.math.max
import kotlin.math.min

/**
 * LightmapPhase (Phase 1b) - derive a low-frequency ILLUMINATION MAP from the reconstructed background.
 *
 * Input  : output/background_reconstructed.mp4  (Phase 1 output - the real bg, people removed)
 * Output : output/light_map.mp4 (per-frame lightmap) + light_map_preview.png
 *
 * Method (mirrors the V9 cloud graph's light_map chain - #694 ImageScale 48x48 -> #693 ImageBlur(2,3)
 * -> #692 ImageScale back up - made resolution-independent):
 *     downscale each frame to SMALL x SMALL (bilinear = strong low-pass) -> mild separable Gaussian blur
 *     -> upscale back to the source WxH (bilinear). The result is a smooth illumination field.
 *
 * 🔴 WHO CONSUMES THIS. Nothing in the app reads it - but this phase is NOT dead and must
 * not be deleted:
 *   * `_e2e/tools/build_cloud_bundle.py` exits non-zero without `light_map.mp4`, and builds the
 *     uploaded plate `masked_video_00002.mp4` from it;
 *   * in the V9 cloud graph the lightmap reaches the sampler as `bg_images`
 *     (VHS_LoadVideo -> Set/Get_light_map -> DrawMaskOnImage -> #62 WanVideoAnimateEmbeds), i.e. it
 *     is the generator's background conditioning;
 *   * `light_map_plate.mp4` (below) is the silhouette-over-lightmap frame the cloud receives INSTEAD
 *     of the real scene - a privacy artifact, scored by `_e2e/tools/lightmap_location_privacy.py`.
 *
 * Pure-Kotlin bitmap math (no model, no NPU): the blur runs on the tiny SMALLxSMALL image so it is
 * effectively free. PRIVACY (§2): on-device only; the real background never leaves the phone.
 */
object LightmapPhase {

    fun interface Logger { fun log(line: String) }
    data class Result(val output: File, val frames: Int, val smallDim: Int, val blurRadius: Int, val fps: Int)

    /**
     * Defaults (user asked for "more downscaling and a stronger blur"):
     *
     * - **32² down** (was 48²) matches the band limit the cloud actually used for the character clip's
     *   lightmap - measured: that clip's background is block-constant on a 32×32 grid.
     * - **radius 3 / σ 1.2** (was radius 2 / σ 3). The old pair was a BUG, not merely weak: a Gaussian
     *   needs radius ≥ 2σ, so radius 2 with σ 3 truncated the kernel into a flat 5-tap box. σ is now
     *   expressed in *blocks* and kept modest on purpose. The consumer is the CLOUD's background
     *   conditioning, which wants the same band limit the cloud graph itself uses; if these ever
     *   need re-tuning, re-tune against the cloud's own light_map chain.
     * - Reconstruction is smoothed harder than the cloud's: matching its NEAREST upscale exactly would
     *   put visible 39.5 px shade steps across the character that crawl as the camera moves.
     */
    const val DEF_SMALL = 32
    const val DEF_BLUR_RADIUS = 3
    const val DEF_BLUR_SIGMA = 1.2f
    /** Intermediate stage for the upscale - smooths away bilinear mach banding for almost nothing. */
    private const val MID_DIM = 128
    private const val MID_BLUR_RADIUS = 2
    private const val MID_BLUR_SIGMA = 1.0f

    fun run(
        context: Context, maxDim: Int, log: Logger,
        smallDim: Int = DEF_SMALL, blurRadius: Int = DEF_BLUR_RADIUS, blurSigma: Float = DEF_BLUR_SIGMA,
    ): Result {
        val bgF = MiragePaths.backgroundReconstructed
        require(bgF.exists()) { "missing ${bgF.name} - run Phase 1 (Inpaint) first" }
        MiragePaths.ensureDirs()

        val src = FrameSource(context, Uri.fromFile(bgF))
        try {
            val n = src.count
            require(n >= 1) { "no frames in ${bgF.name}" }
            val f0 = src.frameAt(0, maxDim) ?: throw IllegalStateException("cannot decode ${bgF.name}")
            val w = f0.width; val h = f0.height
            val s = smallDim.coerceIn(8, min(w, h))
            val out = MiragePaths.lightMap
            val fps = src.srcFps
            log.log("[lightmap] $n frames @ ${w}x$h -> down ${s}x$s + gaussian(r=$blurRadius, σ=${"%.1f".format(blurSigma)}) -> up ${w}x$h")

            // SILHOUETTE-OVER-LIGHTMAP PLATE (2026-07-26, user request: "lightmap behind silhouette").
            // Emitted alongside the lightmap so the phone produces the SAME plate the cloud already
            // receives - see _e2e/tools/build_cloud_bundle.py, which builds it as: take the lightmap
            // frame, then flat-fill wherever the Tier-1 mask is set. The real scene never survives
            // inside the silhouette, and outside it only the low-pass lightmap remains.
            //
            // FILL COLOUR IS DETECTED, NOT HARDCODED. Tier-1 writes GRAY=128, but the video export
            // chain shifts it (measured 122 on a real clip - a video-range/colour-matrix artefact that
            // is device-dependent). HoleMask.detectFillColor() votes for the actual value, so the plate
            // matches whatever this clip really carries. Hardcoding either 128 or 122 would be exactly
            // the fitted-constant mistake the project rule forbids.
            // ── Silhouette source ────────────────────────────────────────────────────────────────
            // Previously the plate was emitted ONLY when mask.mp4 was staged, so the derived-hole path
            // (a masked clip with no separate mask track) silently produced no plate at all - and the
            // cloud bundle builder hard-fails without one. The silhouette is now DERIVED from the gray
            // fill in masked_video.mp4 in that case, exactly as BackgroundInpaint does, so the plate is
            // always produced.
            val maskF = MiragePaths.maskVideo
            val maskedF = MiragePaths.maskedVideo
            val maskSrc = if (maskF.exists()) FrameSource(context, Uri.fromFile(maskF)) else null
            // Detect the fill colour from the MASKED video (that is where the gray silhouette lives).
            // This FrameSource used to be leaked - a MediaMetadataRetriever never released, once per run.
            // Prefer the value Phase 1 already detected on this same clip - re-detecting costs 5
            // decodes plus 5 full-frame flatness scans for an answer we computed minutes ago.
            var detected = runCatching {
                MiragePaths.fillColorFile.takeIf { it.exists() }?.readText()?.trim()?.toIntOrNull() ?: -1
            }.getOrDefault(-1)
            if (detected in 0..255) {
                log.log("[lightmap] fill colour gray($detected) reused from Phase 1 (no re-scan)")
            } else if (maskedF.exists()) {
                val probe = runCatching { FrameSource(context, Uri.fromFile(maskedF)) }.getOrNull()
                if (probe != null) try {
                    detected = runCatching { HoleMask.detectFillColor(probe, n, maxDim, log::log) }.getOrElse { -1 }
                } finally { probe.close() }
            }
            // detectFillColor returns -1 when the vote is not decisive. The old code used
            // `.getOrElse { 128 }`, which only catches THROWN exceptions - a -1 return sailed through
            // into Color.rgb(-1,-1,-1) = 0xFFFFFFFF, i.e. the silhouette was filled WHITE instead of
            // gray. Clamp to the documented Tier-1 default instead.
            val fillV = if (detected in 0..255) detected else 128
            if (detected !in 0..255) log.log("[lightmap] fill colour not decisive - using gray(128)")
            val fillColor = Color.rgb(fillV, fillV, fillV)
            val derive = maskSrc == null && maskedF.exists()
            val derivedSrc = if (derive) runCatching { FrameSource(context, Uri.fromFile(maskedF)) }.getOrNull() else null
            val holeMask = if (derivedSrc != null) HoleMask(w, h) else null
            when {
                maskSrc != null -> log.log("[lightmap] plate: silhouette over lightmap, fill=gray($fillV) [detected from mask.mp4]")
                derivedSrc != null -> log.log("[lightmap] plate: silhouette DERIVED from the gray($fillV) fill (no ${maskF.name})")
                else -> log.log("[lightmap] no ${maskF.name} and no ${maskedF.name} - skipping the silhouette plate")
            }

            var firstLm: Bitmap? = null
            var firstPlate: Bitmap? = null
            val plateOut = MiragePaths.lightMapPlate
            val ms = maskSrc
            val ds = derivedSrc
            val hm = holeMask
            val wantPlate = ms != null || ds != null
            val np = w * h
            val sil = if (wantPlate) BooleanArray(np) else null      // reused every frame
            val srcPx = if (ds != null) IntArray(np) else null
            // The lightmap is unwrapped to pixels ONCE per frame and both encoders are fed from arrays.
            // Before: lightmapOf built a Bitmap, the encoder getPixels'd it, plateOf getPixels'd it AGAIN
            // and allocated a second Bitmap, and the plate encoder getPixels'd that - four 1.6 M-element
            // copies and two 6.4 MB allocations per frame, for a phase that only paints a flat silhouette
            // over a blurred image.
            val lmPx = IntArray(np)
            val platePx = if (wantPlate) IntArray(np) else null
            var firstPlatePx: IntArray? = null
            // Every read below is strictly ascending, on all three sources -> prefetch each, so the
            // decode overlaps the lightmap build and the two encodes instead of blocking in front of
            // them. Bit-identical: same frames, same indices, same order (see FrameSource.startPrefetch).
            src.startPrefetch(n, maxDim)
            ms?.startPrefetch(n, maxDim)
            ds?.startPrefetch(n, maxDim)
            Mp4Encoder(out, fps).use { enc ->
                val penc = if (wantPlate) Mp4Encoder(plateOut, fps) else null
                try {
                    for (t in 0 until n) {
                        val frame = Prof.time(Prof.DECODE) { src.frameAt(t, maxDim) } ?: continue
                        val lm = Prof.time(Prof.COMPOSITE) { lightmapOf(frame, w, h, s, blurRadius, blurSigma) }
                        frame.recycle()
                        Prof.time(Prof.COMPOSITE) { lm.getPixels(lmPx, 0, w, 0, 0, w, h) }
                        if (t == 0) firstLm = lm else lm.recycle()
                        Prof.time(Prof.ENCODE) { enc.writeFrame(lmPx, w, h) }
                        if (penc != null && sil != null && platePx != null) {
                            val ok = Prof.time(Prof.MASK) {
                                when {
                                    ms != null -> silhouetteFromMask(ms, t, maxDim, w, h, sil)
                                    ds != null && hm != null && srcPx != null ->
                                        silhouetteFromFill(ds, t, maxDim, w, h, fillV, hm, srcPx, sil)
                                    else -> false
                                }
                            }
                            if (ok) {
                                Prof.time(Prof.COMPOSITE) {
                                    System.arraycopy(lmPx, 0, platePx, 0, np)
                                    Par.range(np) { p0, p1 ->
                                        for (i in p0 until p1) if (sil[i]) platePx[i] = fillColor
                                    }
                                }
                                if (t == 0) firstPlatePx = platePx.copyOf()
                                Prof.time(Prof.ENCODE) { penc.writeFrame(platePx, w, h) }
                            } else {
                                // no silhouette this frame -> emit the bare lightmap rather than dropping
                                // a frame, so light_map_plate.mp4 stays frame-aligned with light_map.mp4
                                Prof.time(Prof.ENCODE) { penc.writeFrame(lmPx, w, h) }
                            }
                        }
                        if (t % 20 == 0 || t == n - 1) log.log("[lightmap] $t/$n frames …")
                    }
                } finally { penc?.close() }
            }
            firstLm?.let { VideoIo.savePng(it, File(out.parentFile, "light_map_preview.png")) }
            firstPlate = firstPlatePx?.let { Bitmap.createBitmap(it, w, h, Bitmap.Config.ARGB_8888) }
            firstPlate?.let { VideoIo.savePng(it, File(out.parentFile, "light_map_plate_preview.png")) }
            log.log("[prof] Phase 1b sub-stage breakdown:\n${Prof.report(n)}")
            maskSrc?.close()
            derivedSrc?.close()
            log.log("[lightmap] done -> ${out.name} ($n frames @ $fps fps)"
                    + if (maskSrc != null) " + ${plateOut.name}" else "")
            return Result(out, n, s, blurRadius, fps)
        } finally { src.close() }
    }

    /**
     * The silhouette-over-lightmap plate for ONE frame.
     *
     * Every pixel of the output is either lightmap (low-pass, no recoverable scene texture) or a flat
     * fill inside the silhouette. The real frame is never read here - `lm` is already the lightmap - 
     * so there is no path by which real background texture can reach this file. That is the property
     * that makes the plate safe to upload.
     *
     * The mask is binarised at 127 exactly like the cloud builder, and resized NEAREST so a resize can
     * never invent a soft edge that would leak a partial pixel of anything.
     */

    /**
     * Silhouette from a staged mask.mp4. Binarised at 127 exactly like the cloud builder, and resized
     * NEAREST so a resize can never invent a soft edge that would leak a partial pixel of anything.
     */
    private fun silhouetteFromMask(maskSrc: FrameSource, t: Int, maxDim: Int,
                                   w: Int, h: Int, out: BooleanArray): Boolean {
        val mkRaw = maskSrc.frameAt(t, maxDim) ?: return false
        val mk = if (mkRaw.width == w && mkRaw.height == h) mkRaw
                 else Bitmap.createScaledBitmap(mkRaw, w, h, false)   // NEAREST: keep it strictly binary
        val n = w * h
        val mpx = IntArray(n); mk.getPixels(mpx, 0, w, 0, 0, w, h)
        if (mk !== mkRaw) mk.recycle()
        Par.range(n) { p0, p1 ->
            for (i in p0 until p1) {
                val c = mpx[i]
                // luma-free test: the mask is binary, so any channel above mid-grey means "person"
                out[i] = ((c shr 16 and 0xFF) + (c shr 8 and 0xFF) + (c and 0xFF)) / 3 > 127
            }
        }
        return true
    }

    /**
     * Silhouette DERIVED from the gray fill in masked_video.mp4, for clips staged without a separate
     * mask track. Uses the same flatness-gated [HoleMask] extractor Phase 1 uses, so the plate's
     * silhouette matches the one the inpainter reconstructed behind - rather than a naive colour test
     * that would also swallow gray pavement.
     *
     * `maskInto` returns -1 when the result is implausible (covers more than its MAX_COVER of the
     * frame); in that case we report failure and the caller emits the bare lightmap for this frame,
     * which is strictly SAFER than flat-filling a wrong region.
     */
    private fun silhouetteFromFill(src: FrameSource, t: Int, maxDim: Int, w: Int, h: Int,
                                   fillV: Int, hm: HoleMask, px: IntArray, out: BooleanArray): Boolean {
        val f = src.frameAt(t, maxDim) ?: return false
        if (f.width != w || f.height != h) return false
        f.getPixels(px, 0, w, 0, 0, w, h)
        return hm.maskInto(px, fillV, out) >= 0
    }

    /**
     * downscale (bilinear low-pass) -> Gaussian -> TWO-STAGE upscale.
     *
     * The upscale goes s² → 128² → WxH with a small Gaussian in between rather than straight to WxH.
     * A single bilinear jump from 32² to 1264² leaves piecewise-linear mach bands; a full-resolution
     * Gaussian would kill them but costs ~400 M multiply-adds per frame in Kotlin (seconds/frame).
     * Blurring at the 128² waypoint achieves the same smoothness for a rounding error of the cost, and
     * `createScaledBitmap` is hardware-accelerated.
     */
    private fun lightmapOf(frame: Bitmap, w: Int, h: Int, s: Int, radius: Int, sigma: Float): Bitmap {
        val small = Bitmap.createScaledBitmap(frame, s, s, true)          // bilinear low-pass
        val blurred = if (radius > 0) gaussianBlur(small, radius, sigma) else small
        if (small !== blurred) small.recycle()
        val midDim = min(MID_DIM, max(w, h))
        return if (midDim > s) {
            val mid = Bitmap.createScaledBitmap(blurred, midDim, midDim, true)
            blurred.recycle()
            val midBlur = gaussianBlur(mid, MID_BLUR_RADIUS, MID_BLUR_SIGMA)
            mid.recycle()
            val full = Bitmap.createScaledBitmap(midBlur, w, h, true)
            midBlur.recycle()
            full
        } else {
            val full = Bitmap.createScaledBitmap(blurred, w, h, true)
            if (full !== blurred) blurred.recycle()
            full
        }
    }

    /** Separable Gaussian blur on a small ARGB bitmap (s is tiny so this is instant). Edge-clamped. */
    private fun gaussianBlur(bmp: Bitmap, radius: Int, sigma: Float): Bitmap {
        val w = bmp.width; val h = bmp.height
        val k = gaussianKernel(radius, sigma)
        val src = IntArray(w * h); bmp.getPixels(src, 0, w, 0, 0, w, h)
        val tmp = IntArray(w * h)
        for (y in 0 until h) for (x in 0 until w) {
            var r = 0f; var g = 0f; var b = 0f
            for (i in -radius..radius) {
                val xx = (x + i).coerceIn(0, w - 1); val p = src[y * w + xx]; val wt = k[i + radius]
                r += Color.red(p) * wt; g += Color.green(p) * wt; b += Color.blue(p) * wt
            }
            tmp[y * w + x] = Color.rgb(r.toInt().coerceIn(0, 255), g.toInt().coerceIn(0, 255), b.toInt().coerceIn(0, 255))
        }
        val dst = IntArray(w * h)
        for (y in 0 until h) for (x in 0 until w) {
            var r = 0f; var g = 0f; var b = 0f
            for (i in -radius..radius) {
                val yy = (y + i).coerceIn(0, h - 1); val p = tmp[yy * w + x]; val wt = k[i + radius]
                r += Color.red(p) * wt; g += Color.green(p) * wt; b += Color.blue(p) * wt
            }
            dst[y * w + x] = Color.rgb(r.toInt().coerceIn(0, 255), g.toInt().coerceIn(0, 255), b.toInt().coerceIn(0, 255))
        }
        return Bitmap.createBitmap(dst, w, h, Bitmap.Config.ARGB_8888)
    }

    private fun gaussianKernel(radius: Int, sigma: Float): FloatArray {
        val s = if (sigma > 0f) sigma else max(0.5f, radius / 2f)
        val k = FloatArray(2 * radius + 1); var sum = 0f
        for (i in -radius..radius) { val v = exp(-(i * i) / (2f * s * s)); k[i + radius] = v; sum += v }
        for (i in k.indices) k[i] /= sum
        return k
    }
}
