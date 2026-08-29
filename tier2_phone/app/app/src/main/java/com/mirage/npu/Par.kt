package com.mirage.npu

import java.util.concurrent.Callable
import java.util.concurrent.ExecutionException
import java.util.concurrent.Executors
import java.util.concurrent.ThreadFactory
import java.util.concurrent.atomic.AtomicInteger

/**
 * Par - a tiny fixed worker pool for splitting an index range into independent bands.
 *
 * WHY. The entire Phase-1/1b/2 image engine ran on ONE thread of an 8-core Snapdragon 8 Elite (2 prime
 * + 6 performance). The per-pixel loops are embarrassingly parallel, so this recovers most of the idle
 * silicon. It is deliberately NOT a general-purpose async framework - it is a synchronous
 * "fork, run bands, join" primitive, which is the only shape that keeps results bit-identical.
 *
 * ── WHAT MAKES THIS SAFE (each of these was a real hazard found while auditing the engine) ──
 * 1. INTRA-frame only, never inter-frame. Pipelining whole frames would race the §2 leak counters
 *    (`BackgroundInpaint.leakSum` / `leakFrames` are non-atomic `+=`) and corrupt `HoleMask`'s shared
 *    per-frame buffers, whose `auditLeak` depends on state left by `maskInto` for the SAME frame.
 *    Splitting the pixel range of ONE frame touches neither.
 * 2. Disjoint writes only. Every band writes its own slice of the output array. Callers must not share
 *    scratch across bands - allocate per-band inside the body (see aggregatePlate).
 * 3. No reductions here. An argmin/sum across bands would change accumulation order or tie-breaks.
 *    `Aligner.bestShiftSub` in particular resolves ties by FIRST-CANDIDATE-WINS over a dy-then-dx scan;
 *    a naive parallel argmin can pick a different shift by 1 px and change every output pixel. Any
 *    reduction must be written explicitly with a deterministic combine, not dropped into `range`.
 * 4. Exceptions propagate. A worker that throws re-throws on the calling thread rather than silently
 *    leaving a band unwritten - a half-written plate would be a privacy artifact, not just a bug.
 * 5. Graceful degradation. One core, a tiny range, or a pool failure all fall back to running inline,
 *    matching the app's design philosophy everywhere else (NpuFactory falls back to CPU frame-blend).
 *
 * THERMAL NOTE. The ledger's "no throttling" measurement was taken under a SINGLE-THREADED load, with
 * skin temperature peaking within 1.1 °C of the 38 °C gate. A sustained all-core load will approach
 * that gate sooner, so [WORKERS] deliberately leaves headroom instead of using every core.
 */
object Par {

    /** Leave 2 cores for the encoder, the decoder and the UI - and for thermal headroom. */
    val WORKERS: Int = (Runtime.getRuntime().availableProcessors() - 2).coerceIn(1, 6)

    /** Below this many elements the fork/join overhead outweighs the win; run inline. */
    private const val MIN_PARALLEL = 32_768

    private const val THREAD_PREFIX = "mirage-par-"

    /**
     * DEADLOCK GUARD. This is a FIXED pool, so a task that itself calls [range]/[rows] would submit
     * sub-tasks and then block waiting for them - with every worker already occupied doing the same,
     * nothing can make progress. No current call site nests (the alignment search runs outside every
     * pixel-loop band, deliberately), but that is an invariant a future edit could break silently and
     * catastrophically. Nested calls simply run inline instead.
     */
    private fun onPoolThread(): Boolean = Thread.currentThread().name.startsWith(THREAD_PREFIX)

    private val pool by lazy {
        Executors.newFixedThreadPool(WORKERS, object : ThreadFactory {
            private val n = AtomicInteger(0)
            override fun newThread(r: Runnable): Thread =
                Thread(r, THREAD_PREFIX + n.incrementAndGet()).apply {
                    isDaemon = true                       // never hold the process open
                    priority = Thread.NORM_PRIORITY - 1   // stay below the UI/encoder threads
                }
        })
    }

    /**
     * Split `[0, n)` into [WORKERS] contiguous bands and run [body] on each, blocking until all finish.
     * [body] receives a half-open range and must only write inside it.
     */
    fun range(n: Int, body: (from: Int, to: Int) -> Unit) = range(n, MIN_PARALLEL, body)

    /**
     * As [range], but with an explicit parallelism threshold. Use a small [minParallel] when each
     * ITEM is expensive - the alignment search has only 49 candidates, but each one is a SAD over
     * ~400 000 pixels, so the default element-count threshold would wrongly run it inline.
     */
    fun range(n: Int, minParallel: Int, body: (from: Int, to: Int) -> Unit) {
        if (n <= 0) return
        if (WORKERS <= 1 || n < minParallel || onPoolThread()) { body(0, n); return }
        val bands = WORKERS
        val chunk = (n + bands - 1) / bands
        val tasks = ArrayList<Callable<Unit>>(bands)
        var from = 0
        while (from < n) {
            val lo = from
            val hi = minOf(n, from + chunk)
            tasks.add(Callable { body(lo, hi) })
            from = hi
        }
        if (tasks.size == 1) { body(0, n); return }
        runTasks(tasks) { body(0, n) }
    }

    /**
     * Row-band split of an image: [body] gets a half-open ROW range. Preferred over [range] whenever the
     * work touches neighbours horizontally, because whole rows keep each band's memory contiguous and
     * avoid two bands sharing a cache line on a boolean/byte output.
     */
    fun rows(h: Int, w: Int, body: (y0: Int, y1: Int) -> Unit) {
        if (h <= 0) return
        if (WORKERS <= 1 || h.toLong() * w < MIN_PARALLEL || h < 2 || onPoolThread()) { body(0, h); return }
        val bands = minOf(WORKERS, h)
        val chunk = (h + bands - 1) / bands
        val tasks = ArrayList<Callable<Unit>>(bands)
        var y = 0
        while (y < h) {
            val lo = y
            val hi = minOf(h, y + chunk)
            tasks.add(Callable { body(lo, hi) })
            y = hi
        }
        if (tasks.size == 1) { body(0, h); return }
        runTasks(tasks) { body(0, h) }
    }

    /** Submit, join, and re-throw the first worker failure on the caller's thread. */
    private fun runTasks(tasks: List<Callable<Unit>>, inline: () -> Unit) {
        val futures = try {
            pool.invokeAll(tasks)
        } catch (t: Throwable) {
            // Pool unusable (shutdown, OOM on thread create) - never fail the render over it.
            inline(); return
        }
        var err: Throwable? = null
        for (f in futures) {
            try {
                f.get()
            } catch (e: ExecutionException) {
                if (err == null) err = e.cause ?: e
            } catch (e: InterruptedException) {
                Thread.currentThread().interrupt()
                if (err == null) err = e
            }
        }
        err?.let { throw RuntimeException("parallel band failed: ${it.message}", it) }
    }
}
