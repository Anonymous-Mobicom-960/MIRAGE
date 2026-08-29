package com.mirage.npu

import android.content.Context
import android.graphics.Bitmap
import android.graphics.Color
import android.media.MediaCodec
import android.media.MediaCodecInfo
import android.media.MediaFormat
import android.media.MediaMetadataRetriever
import android.media.MediaMuxer
import android.net.Uri
import android.util.Log
import java.io.File
import java.io.FileOutputStream

/**
 * VideoIo - decode an mp4 (file OR content Uri) to frames, and encode frames back to an mp4.
 *
 * decodeToBitmaps / decodeUriToBitmaps: MediaMetadataRetriever.getFrameAtIndex (API 28+). Rotation
 * metadata is applied by the platform. Frames are downscaled + capped to keep memory bounded on the
 * demo path (a whole video decoded to ARGB bitmaps would OOM on long clips).
 *
 * encodeBitmaps: a synchronous MediaCodec(H.264) + MediaMuxer encoder that is MORE robust than a
 * naive NV12 dump:
 *  - dimensions are rounded to a multiple of 16 (avoids encoder stride != width surprises),
 *  - the input color format is chosen from what THIS device's encoder actually reports
 *     (SemiPlanar/NV12, Planar/I420, or Flexible via the Image plane API),
 *  - presentation timestamps are exact (offline encode, so PTS is derived from the frame index).
 * It still runs on the device's hardware AVC encoder, which is well-supported on the S25 Ultra.
 */
object VideoIo {
    private const val TAG = "VideoIo"
    private const val MIME = MediaFormat.MIMETYPE_VIDEO_AVC   // H.264
    private const val TIMEOUT_US = 10_000L

    private const val COLOR_FLEXIBLE = MediaCodecInfo.CodecCapabilities.COLOR_FormatYUV420Flexible
    private const val COLOR_SEMIPLANAR = MediaCodecInfo.CodecCapabilities.COLOR_FormatYUV420SemiPlanar
    private const val COLOR_PLANAR = MediaCodecInfo.CodecCapabilities.COLOR_FormatYUV420Planar

    data class Decoded(val frames: List<Bitmap>, val srcFps: Int)

    /** Decode every frame of a FILE to ARGB_8888 bitmaps (used by the t2b3 shared-folder pipeline). */
    fun decodeToBitmaps(path: File): List<Bitmap> {
        val mmr = MediaMetadataRetriever()
        return try {
            mmr.setDataSource(path.absolutePath)
            readFrames(mmr, maxFrames = Int.MAX_VALUE, maxDim = 0)
        } finally {
            runCatching { mmr.release() }
        }
    }

    /**
     * Decode a content Uri (from the SAF picker) to bitmaps. Takes the FIRST [maxFrames]
     * consecutive frames (a short segment) and downscales
     * so the longest side is <= [maxDim]. Returns the frames + an estimated source fps.
     */
    fun decodeUriToBitmaps(
        context: Context,
        uri: Uri,
        maxFrames: Int,
        maxDim: Int,
        log: (String) -> Unit,
    ): Decoded {
        val mmr = MediaMetadataRetriever()
        return try {
            mmr.setDataSource(context, uri)
            val durMs = mmr.extractMetadata(MediaMetadataRetriever.METADATA_KEY_DURATION)?.toLongOrNull() ?: 0L
            val count = mmr.extractMetadata(MediaMetadataRetriever.METADATA_KEY_VIDEO_FRAME_COUNT)?.toIntOrNull() ?: 0
            val srcFps = when {
                count > 0 && durMs > 0 -> ((count * 1000L) / durMs).toInt().coerceIn(5, 60)
                else -> 24
            }
            log("[demo] source: ${count.takeIf { it > 0 } ?: "?"} frames, ${durMs}ms -> ~$srcFps fps")
            val frames = readFrames(mmr, maxFrames, maxDim)
            Decoded(frames, srcFps)
        } finally {
            runCatching { mmr.release() }
        }
    }

    private fun readFrames(mmr: MediaMetadataRetriever, maxFrames: Int, maxDim: Int): List<Bitmap> {
        val count = mmr.extractMetadata(MediaMetadataRetriever.METADATA_KEY_VIDEO_FRAME_COUNT)?.toIntOrNull() ?: 0
        val out = ArrayList<Bitmap>()
        if (count > 0) {
            val n = minOf(count, maxFrames)
            for (i in 0 until n) {
                val raw = runCatching { mmr.getFrameAtIndex(i) }.getOrNull() ?: continue
                out.add(prepare(raw, maxDim))
            }
        } else {
            // Fallback: no frame count -> sample every ~33ms up to maxFrames.
            val durUs = (mmr.extractMetadata(MediaMetadataRetriever.METADATA_KEY_DURATION)?.toLongOrNull() ?: 0L) * 1000
            var t = 0L
            while (t < durUs && out.size < maxFrames) {
                val raw = mmr.getFrameAtTime(t, MediaMetadataRetriever.OPTION_CLOSEST)
                if (raw != null) out.add(prepare(raw, maxDim))
                t += 33_333L
            }
        }
        return out
    }

    /** Force software ARGB_8888 (so getPixels works) and downscale so the longest side <= maxDim. */
    internal fun prepare(raw: Bitmap, maxDim: Int): Bitmap {
        val soft = if (raw.config == Bitmap.Config.ARGB_8888) raw else raw.copy(Bitmap.Config.ARGB_8888, false)
        if (maxDim <= 0) return soft
        val longest = maxOf(soft.width, soft.height)
        if (longest <= maxDim) return soft
        val scale = maxDim.toFloat() / longest
        val w = (soft.width * scale).toInt().coerceAtLeast(2)
        val h = (soft.height * scale).toInt().coerceAtLeast(2)
        val scaled = Bitmap.createScaledBitmap(soft, w, h, true)
        if (scaled !== soft && soft !== raw) soft.recycle()
        return scaled
    }

    fun savePng(bmp: Bitmap, outPath: File) {
        outPath.parentFile?.mkdirs()
        FileOutputStream(outPath).use { bmp.compress(Bitmap.CompressFormat.PNG, 100, it) }
    }

    // ---------------------------------------------------------------------------------------------
    // Encoder
    // ---------------------------------------------------------------------------------------------

    /** Encode a whole in-memory list to an H.264 mp4 at [fps] (used by the t2b3 path). */
    fun encodeBitmaps(outPath: File, frames: List<Bitmap>, fps: Int, bitRate: Int = 8_000_000) {
        require(frames.isNotEmpty()) { "no frames to encode" }
        Mp4Encoder(outPath, fps, bitRate).use { enc -> for (f in frames) enc.writeFrame(f) }
    }

    internal fun chooseColorFormat(caps: MediaCodecInfo.CodecCapabilities): Int {
        val supported = caps.colorFormats.toHashSet()
        return when {
            supported.contains(COLOR_SEMIPLANAR) -> COLOR_SEMIPLANAR
            supported.contains(COLOR_PLANAR) -> COLOR_PLANAR
            supported.contains(COLOR_FLEXIBLE) -> COLOR_FLEXIBLE
            else -> caps.colorFormats.first()
        }
    }

    /** Fill the encoder input for one frame; returns the number of bytes queued (w*h*3/2). */
    internal fun feedFrame(codec: MediaCodec, inIndex: Int, bmp: Bitmap, w: Int, h: Int, colorFormat: Int, px: IntArray, out: ByteArray): Int {
        val src = if (bmp.width == w && bmp.height == h) bmp else Bitmap.createScaledBitmap(bmp, w, h, true)
        src.getPixels(px, 0, w, 0, 0, w, h)
        if (src !== bmp) src.recycle()   // the temporary scaled bitmap, once its pixels are copied out
        return feedFramePixels(codec, inIndex, px, w, h, colorFormat, out)
    }

    /**
     * Same as [feedFrame] but the caller ALREADY has the ARGB pixels.
     *
     * Every phase builds its output frame as an IntArray, wraps it in a Bitmap purely to hand to the
     * encoder, and then the encoder immediately calls `getPixels` to unwrap it again - a pointless
     * 1.6 M-element round trip (plus a 6.4 MB Bitmap allocation) per encoded frame. Phase 1b runs TWO
     * encoders per frame, so it paid that twice. Measured: `encode` was 40 ms/frame in Phase 1.
     */
    internal fun feedFramePixels(codec: MediaCodec, inIndex: Int, px: IntArray, w: Int, h: Int, colorFormat: Int, out: ByteArray): Int {
        if (colorFormat == COLOR_FLEXIBLE) {
            // Strides-aware path via the input Image (safest for arbitrary encoders).
            val image = codec.getInputImage(inIndex)
            if (image != null) {
                fillYuvImage(image, px, w, h)
                return w * h * 3 / 2
            }
            // else fall through to a tight NV12 write below
        }

        val ib = codec.getInputBuffer(inIndex)!!
        ib.clear()
        val ySize = w * h
        // Filled into a plain ByteArray, then ONE bulk `ib.put(out)`.
        //
        // I previously "optimised" this to write straight into the codec's ByteBuffer to save the 2.4 MB
        // bulk copy. MEASURED: that was a REGRESSION. A DirectByteBuffer absolute put is a bounds-checked
        // call per byte, and there are 2.4 M of them per frame - far more expensive than JIT-friendly
        // array stores plus a single native memmove. Phase 1b's encode rose 15 040 -> 18 665 ms after
        // that change. The bulk copy is cheap; the per-byte puts were not.
        val semiPlanar = colorFormat != COLOR_PLANAR   // NV12 unless the encoder wants I420
        // ROW-PARALLEL, and BIT-IDENTICAL. The old NV12 branch carried a running `uv` cursor
        // (`out[uv++] = u`), which is a loop-carried dependency that forced the whole 1.6 M-iteration
        // conversion onto one thread. The cursor is derivable in closed form: the chroma pair for an
        // even (j,i) is the ((j/2)*(w/2) + i/2)-th pair, so its byte offset is
        // ySize + 2*((j/2)*(w/2) + i/2) = ySize + (j/2)*w + i  (w is even - Mp4Encoder rounds to /16).
        // Every output byte is now written at an absolute index, so bands cannot interfere.
        Prof.time(Prof.YUV) {
        Par.rows(h, w) { y0, y1 ->
            for (j in y0 until y1) {
                val rowY = j * w
                val uvBase = ySize + (j / 2) * w
                val uPlane = ySize + (j / 2) * (w / 2)
                val vPlane = ySize + ySize / 4 + (j / 2) * (w / 2)
                for (i in 0 until w) {
                    val c = px[rowY + i]
                    val r = Color.red(c); val g = Color.green(c); val b = Color.blue(c)
                    val y = ((66 * r + 129 * g + 25 * b + 128) shr 8) + 16
                    out[rowY + i] = y.coerceIn(0, 255).toByte()
                    if (j and 1 == 0 && i and 1 == 0) {
                        val u = ((-38 * r - 74 * g + 112 * b + 128) shr 8) + 128
                        val v = ((112 * r - 94 * g - 18 * b + 128) shr 8) + 128
                        if (semiPlanar) {
                            out[uvBase + i] = u.coerceIn(0, 255).toByte()        // NV12: U,V interleaved
                            out[uvBase + i + 1] = v.coerceIn(0, 255).toByte()
                        } else {
                            out[uPlane + (i / 2)] = u.coerceIn(0, 255).toByte()   // I420: U plane
                            out[vPlane + (i / 2)] = v.coerceIn(0, 255).toByte()   // then V plane
                        }
                    }
                }
            }
        }
        }
        ib.put(out, 0, ySize * 3 / 2)
        return ySize * 3 / 2
    }

    internal const val ENC_TIMEOUT_US = 10_000L
    internal const val ENC_MIME = MediaFormat.MIMETYPE_VIDEO_AVC

    /** Fill a COLOR_FormatYUV420Flexible input Image, respecting per-plane row/pixel strides. */
    private fun fillYuvImage(image: android.media.Image, px: IntArray, w: Int, h: Int) {
        val yP = image.planes[0]; val uP = image.planes[1]; val vP = image.planes[2]
        val yBuf = yP.buffer; val uBuf = uP.buffer; val vBuf = vP.buffer
        val yRow = yP.rowStride
        val uRow = uP.rowStride; val uPix = uP.pixelStride
        val vRow = vP.rowStride; val vPix = vP.pixelStride
        // Row-parallel and bit-identical: every write is an ABSOLUTE-index ByteBuffer put, which does
        // not touch the buffer's position, so disjoint row bands cannot race.
        Par.rows(h, w) { y0, y1 ->
            for (j in y0 until y1) {
                val rowY = j * w
                val yOff = j * yRow
                val cj = j / 2
                val uOff = cj * uRow; val vOff = cj * vRow
                for (i in 0 until w) {
                    val c = px[rowY + i]
                    val r = Color.red(c); val g = Color.green(c); val b = Color.blue(c)
                    val y = (((66 * r + 129 * g + 25 * b + 128) shr 8) + 16).coerceIn(0, 255)
                    yBuf.put(yOff + i, y.toByte())
                    if (j and 1 == 0 && i and 1 == 0) {
                        val u = (((-38 * r - 74 * g + 112 * b + 128) shr 8) + 128).coerceIn(0, 255)
                        val v = (((112 * r - 94 * g - 18 * b + 128) shr 8) + 128).coerceIn(0, 255)
                        val ci = i / 2
                        uBuf.put(uOff + ci * uPix, u.toByte())
                        vBuf.put(vOff + ci * vPix, v.toByte())
                    }
                }
            }
        }
    }
}

/**
 * FrameSource - pull a video's frames ON DEMAND (frame i, downscaled) without decoding the whole clip
 * into memory. Backs the streaming whole-video converter so ANY length works in O(1) memory.
 */
class FrameSource(private val context: Context, private val uri: Uri) : AutoCloseable {
    private val mmr = MediaMetadataRetriever()
    val count: Int
    val srcFps: Int

    init {
        try {
            mmr.setDataSource(context, uri)
            val dur = mmr.extractMetadata(MediaMetadataRetriever.METADATA_KEY_DURATION)?.toLongOrNull() ?: 0L
            val cnt = mmr.extractMetadata(MediaMetadataRetriever.METADATA_KEY_VIDEO_FRAME_COUNT)?.toIntOrNull() ?: 0
            count = cnt
            srcFps = if (cnt > 0 && dur > 0) ((cnt * 1000L) / dur).toInt().coerceIn(1, 240) else 24
        } catch (t: Throwable) {
            runCatching { mmr.release() }   // constructor fails -> caller never gets an object to close()
            throw t
        }
    }

    // =============================================================================================
    // OPTIONAL SEQUENTIAL PREFETCH - decode frame t+1 while the phase computes frame t.
    //
    // WHY. `decode` is 19.2 s of the 164.5 s pipeline and every phase loop is strictly
    // `decode t -> compute t -> encode t`, so all of it is dead time on the calling thread. Meanwhile
    // Par.WORKERS is deliberately `cores - 2` (Par.kt:40) precisely to leave room "for the encoder,
    // the decoder and the UI" - those cores are sitting idle during the decode. This also hides
    // decode under `codec`, the hardware encoder backpressure, where the CPU is simply waiting.
    //
    // BIT-IDENTICAL. `frameAt` is a pure function of (file, index, maxDim): it reads no state any
    // phase writes. The consumer receives the same bitmaps, for the same indices, in the same order - 
    // only earlier. This is NOT the inter-frame pipelining Par.kt:18 warns about; that hazard is
    // about running COMPUTE for two frames at once, which would race the non-atomic §2 leak counters
    // and HoleMask's shared per-frame buffers. Prefetching touches neither.
    //
    // THE READ PATTERN IS PRESERVED, which is the whole point: the worker walks indices in ASCENDING
    // order on its own retriever, so every read stays on the ~50 ms consecutive path and never
    // becomes one of the ~416 ms scattered seeks (§C.PHONE-9l).
    //
    // ITS OWN RETRIEVER, not the shared one. MediaMetadataRetriever is not thread-safe, and the
    // fallback below can fire at any moment; giving the worker a second instance means the two can
    // never touch the same object, so no join-before-fallback race exists at all.
    //
    // DEPTH 1 (so at most two frames are in flight). Bitmaps live in native memory on API 26+, not
    // the Java heap, so this does not perturb `plateSampleCount` - which reads `maxMemory()`, a
    // constant - and therefore cannot change how many samples the plate uses, i.e. cannot change
    // the output.
    // =============================================================================================

    private class Slot(val index: Int, val bmp: Bitmap?)

    /**
     * One prefetch generation: its queue, its OWN stop flag, and its thread.
     *
     * The stop flag is per-generation rather than a field on FrameSource, and that is load-bearing.
     * With a shared flag, a `join` that timed out would leave an orphaned worker running - and the
     * next `startPrefetch` would set the shared flag back to false, so the orphan would resume,
     * spinning forever on a queue nobody reads, while `stopPrefetch` had already released the
     * retriever it was still decoding from (a native use-after-free). A generation token cannot be
     * un-set by a later generation, so an orphan always terminates on its own.
     */
    private class Pump(
        val q: java.util.concurrent.ArrayBlockingQueue<Slot>,
        val stop: java.util.concurrent.atomic.AtomicBoolean,
        val thread: Thread,
    )

    private var pump: Pump? = null

    /**
     * (Re)begin prefetching indices `0 until count` at [maxDim].
     *
     * Restartable on purpose: Phase 1 walks the masked clip twice - once in `aggregatePlate` and again
     * in `compositeFrames` - with scattered reads in between (`computeTrajectory`, the reference
     * frame). Each sequential pass calls this again to rewind the prefetcher to 0. A failure to start
     * is silently ignored and every read simply takes today's synchronous path.
     */
    fun startPrefetch(count: Int, maxDim: Int) {
        stopPrefetch()
        if (count <= 1) return
        val m = MediaMetadataRetriever()
        try {
            m.setDataSource(context, uri)
        } catch (t: Throwable) {
            runCatching { m.release() }; return
        }
        val stop = java.util.concurrent.atomic.AtomicBoolean(false)
        val q = java.util.concurrent.ArrayBlockingQueue<Slot>(1)
        val th = Thread({
            try {
                for (t in 0 until count) {
                    if (stop.get()) break
                    val b = runCatching { m.getFrameAtIndex(t) }.getOrNull()?.let { VideoIo.prepare(it, maxDim) }
                    var placed = false
                    while (!placed && !stop.get()) {
                        placed = q.offer(Slot(t, b), 250, java.util.concurrent.TimeUnit.MILLISECONDS)
                    }
                    if (!placed) { b?.recycle(); break }   // consumer gave up on us
                }
            } catch (t: Throwable) {
                Log.w("FrameSource", "prefetch stopped: ${t.message}")
            } finally {
                // Released BY THE WORKER, so a release can never race a decode still in flight on it.
                runCatching { m.release() }
            }
        }, "mirage-decode").apply { isDaemon = true; priority = Thread.NORM_PRIORITY }
        pump = Pump(q, stop, th)
        th.start()
    }

    /** Tear the prefetcher down and recycle anything still queued. Idempotent. */
    private fun stopPrefetch() {
        val p = pump ?: return
        pump = null
        p.stop.set(true)
        p.thread.interrupt()
        runCatching { p.thread.join(3000) }
        // A worker that outlived the join may still enqueue one more Slot after this drain; that
        // bitmap is simply unreferenced and collected. It cannot be handed to a consumer, because
        // `pump` is already null.
        while (true) (p.q.poll() ?: break).bmp?.recycle()
    }

    fun frameAt(i: Int, maxDim: Int): Bitmap? {
        val q = pump?.q
        if (q != null) {
            // Consume in lockstep. Anything that is NOT the requested index means the caller did not
            // read sequentially after all, so retire the prefetcher and fall back - correctness never
            // depends on the guess being right.
            while (true) {
                val s = runCatching { q.poll(10, java.util.concurrent.TimeUnit.SECONDS) }.getOrNull() ?: break
                if (s.index == i) return s.bmp
                s.bmp?.recycle()
                if (s.index > i) break
            }
            stopPrefetch()
        }
        return runCatching { mmr.getFrameAtIndex(i) }.getOrNull()?.let { VideoIo.prepare(it, maxDim) }
    }

    override fun close() { stopPrefetch(); runCatching { mmr.release() } }
}

/**
 * Mp4Encoder - streaming H.264/mp4 encoder: writeFrame() one bitmap at a time, close() to finish.
 * Configures itself from the first frame's size (rounded to a multiple of 16). O(1) memory: nothing is
 * buffered beyond the codec's own queue, so a video of any length can be encoded frame-by-frame.
 */
class Mp4Encoder(
    private val outPath: File,
    private val fps: Int,
    private val bitRate: Int = 12_000_000,
) : AutoCloseable {
    private lateinit var codec: android.media.MediaCodec
    private var muxer: MediaMuxer? = null
    private var trackIndex = -1
    private var muxerStarted = false
    private var started = false
    private var finished = false
    private var w = 0
    private var h = 0
    private var colorFormat = 0
    private var frameIdx = 0
    private var codecCreated = false
    private lateinit var pxBuf: IntArray      // reused per frame (allocated once in start) - no per-frame GC
    private lateinit var yuvBuf: ByteArray
    private val info = android.media.MediaCodec.BufferInfo()

    val frameCount: Int get() = frameIdx

    private fun start(rawW: Int, rawH: Int) {
        w = (rawW / 16) * 16
        h = (rawH / 16) * 16
        require(w >= 16 && h >= 16) { "frame too small to encode (${rawW}x$rawH)" }
        codec = android.media.MediaCodec.createEncoderByType(VideoIo.ENC_MIME); codecCreated = true
        try {
            colorFormat = VideoIo.chooseColorFormat(codec.codecInfo.getCapabilitiesForType(VideoIo.ENC_MIME))
            // High, RESOLUTION-AWARE bitrate so re-encoding does not visibly soften the video. ~0.33
            // bits/px/frame, floored at the constructor value.
            val br = maxOf(bitRate.toLong(), w.toLong() * h * fps / 3).coerceAtMost(60_000_000L).toInt()
            val format = MediaFormat.createVideoFormat(VideoIo.ENC_MIME, w, h).apply {
                setInteger(MediaFormat.KEY_COLOR_FORMAT, colorFormat)
                setInteger(MediaFormat.KEY_BIT_RATE, br)
                setInteger(MediaFormat.KEY_FRAME_RATE, fps)
                setInteger(MediaFormat.KEY_I_FRAME_INTERVAL, 1)
                // 🔴 LABEL THE STREAM WITH THE MATRIX WE ACTUALLY WRITE.
                // feedFramePixels / fillYuvImage convert RGB->YUV with the BT.601 LIMITED-RANGE
                // integer matrix (66/129/25, +16 offset). Without these keys MediaCodec tagged the
                // muxed result `color_space=bt709, color_range=tv` (verified with ffprobe on
                // _e2e/run3_20260807/c1_p05_single/tier2_final/background_reconstructed.mp4), so every
                // decoder applied the 709 matrix to 601 numbers. MEASURED cost on that file, over the
                // 48.9%-of-frame region no mask ever touches: decoding as tagged gives mean|diff|
                // 7.208 against the encoder's own input; forcing 601 on decode gives 6.651 - a free
                // 0.557 grey levels, 7.7% of the whole-frame re-encode error.
                // (This does NOT address the other 92%; see FIX1_INPAINT.md §3.)
                setInteger(MediaFormat.KEY_COLOR_STANDARD, MediaFormat.COLOR_STANDARD_BT601_NTSC)
                setInteger(MediaFormat.KEY_COLOR_RANGE, MediaFormat.COLOR_RANGE_LIMITED)
                setInteger(MediaFormat.KEY_COLOR_TRANSFER, MediaFormat.COLOR_TRANSFER_SDR_VIDEO)
            }
            codec.configure(format, null, null, android.media.MediaCodec.CONFIGURE_FLAG_ENCODE)
            codec.start()
            outPath.parentFile?.mkdirs()
            if (outPath.exists()) outPath.delete()
            muxer = MediaMuxer(outPath.absolutePath, MediaMuxer.OutputFormat.MUXER_OUTPUT_MPEG_4)
            pxBuf = IntArray(w * h); yuvBuf = ByteArray(w * h * 3 / 2)
            started = true
        } catch (t: Throwable) {
            // partial init (configure/start/muxer failed) -> release the hardware encoder + muxer so a
            // repeated failure can't exhaust the limited global AVC-encoder pool.
            runCatching { codec.release() }; runCatching { muxer?.release() }; muxer = null
            throw t
        }
    }

    fun writeFrame(bmp: Bitmap) {
        if (finished) return
        if (!started) start(bmp.width, bmp.height)
        queue { inIndex -> VideoIo.feedFrame(codec, inIndex, bmp, w, h, colorFormat, pxBuf, yuvBuf) }
    }

    /**
     * Encode a frame the caller already holds as ARGB pixels - no Bitmap, no `getPixels`.
     *
     * [srcW]/[srcH] must match this encoder's configured size (they always do in this app: every phase
     * builds its output at the working resolution). Saves a 6.4 MB Bitmap allocation plus a 1.6 M-element
     * copy per encoded frame, on every phase.
     */
    fun writeFrame(px: IntArray, srcW: Int, srcH: Int) {
        if (finished) return
        if (!started) start(srcW, srcH)
        require(srcW == w && srcH == h) { "frame $srcW x $srcH does not match encoder $w x $h" }
        queue { inIndex -> VideoIo.feedFramePixels(codec, inIndex, px, w, h, colorFormat, yuvBuf) }
    }

    private fun queue(fill: (Int) -> Int) {
        var queued = false
        while (!queued) {
            // CODEC covers dequeue+queue+drain. Subtract YUV from ENCODE and what remains is hardware
            // backpressure - the part a GL input Surface could NOT remove.
            val inIndex = Prof.time(Prof.CODEC) { codec.dequeueInputBuffer(VideoIo.ENC_TIMEOUT_US) }
            if (inIndex >= 0) {
                val pts = frameIdx.toLong() * 1_000_000L / fps
                val size = fill(inIndex)
                Prof.time(Prof.CODEC) { codec.queueInputBuffer(inIndex, 0, size, pts, 0) }
                frameIdx++
                queued = true
            }
            Prof.time(Prof.CODEC) { drain(false) }
        }
    }

    private fun drain(endOfStream: Boolean) {
        val mx = muxer ?: return
        while (true) {
            val outIndex = codec.dequeueOutputBuffer(info, VideoIo.ENC_TIMEOUT_US)
            when {
                outIndex == android.media.MediaCodec.INFO_TRY_AGAIN_LATER -> if (!endOfStream) return
                outIndex == android.media.MediaCodec.INFO_OUTPUT_FORMAT_CHANGED -> {
                    trackIndex = mx.addTrack(codec.outputFormat); mx.start(); muxerStarted = true
                }
                outIndex >= 0 -> {
                    val ob = codec.getOutputBuffer(outIndex)!!
                    if (info.flags and android.media.MediaCodec.BUFFER_FLAG_CODEC_CONFIG != 0) info.size = 0
                    if (info.size > 0 && muxerStarted) {
                        ob.position(info.offset); ob.limit(info.offset + info.size)
                        mx.writeSampleData(trackIndex, ob, info)
                    }
                    val eos = info.flags and android.media.MediaCodec.BUFFER_FLAG_END_OF_STREAM != 0
                    codec.releaseOutputBuffer(outIndex, false)
                    if (eos) return
                }
            }
        }
    }

    override fun close() {
        if (finished) return
        finished = true
        if (!started) return
        runCatching {
            while (true) {
                val inIndex = codec.dequeueInputBuffer(VideoIo.ENC_TIMEOUT_US)
                if (inIndex >= 0) {
                    codec.queueInputBuffer(inIndex, 0, 0, frameIdx.toLong() * 1_000_000L / fps,
                        android.media.MediaCodec.BUFFER_FLAG_END_OF_STREAM)
                    break
                }
                drain(false)
            }
            drain(true)
        }
        runCatching { codec.stop() }
        runCatching { codec.release() }
        if (muxerStarted) runCatching { muxer?.stop() }
        runCatching { muxer?.release() }
    }
}
