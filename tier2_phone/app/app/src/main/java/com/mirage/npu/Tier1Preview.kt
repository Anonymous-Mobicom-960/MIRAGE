package com.mirage.npu

import android.content.Context
import android.graphics.Bitmap
import android.graphics.Canvas
import android.graphics.Color
import android.graphics.Paint
import android.graphics.Typeface
import android.net.Uri
import kotlin.math.cos
import kotlin.math.hypot
import kotlin.math.roundToInt
import kotlin.math.sin

/**
 * Tier1Preview - run the GUIDED Tier-1 pipeline on a RAW clip, entirely ON THE PHONE, and emit the
 * masked video + mask + the three Tier-1 card sidecars (pose sticks, canonicaliser, synthetic face).
 *
 * MASK = the guided pipeline (strict): YOLO11n-seg (primary) ∪ RVM matte (backstop) → guided filter
 * (edge-align to the frame, r=12/ε=0.01) → morph close → d4 dilate → FaceGuard (head-region fill from
 * the pose head keypoints). POSE = MoveNet (COCO-17). The synthetic face is a fixed template on the
 * head keypoints (never the real face).
 *
 * Written in 3 passes so at most TWO H.264 encoders are ever open at once (the device AVC pool is small):
 *   P1 raw  → masked_video + mask  (+ cache the per-frame pose)
 *   P2 masked → pose sticks + synthetic face
 *   P3 panel → canonicaliser
 */
object Tier1Preview {

    private val SKELETON = arrayOf(
        0 to 1, 0 to 2, 1 to 3, 2 to 4, 5 to 7, 7 to 9, 6 to 8, 8 to 10,
        5 to 6, 5 to 11, 6 to 12, 11 to 12, 11 to 13, 13 to 15, 12 to 14, 14 to 16)
    private val COLORS = intArrayOf(0xFF00EB00.toInt(), 0xFFFFC800.toInt(), 0xFF00A0FF.toInt(),
        0xFFC800FF.toInt(), 0xFF00FFFF.toInt(), 0xFFFFFF00.toInt())
    private const val GRAY = 122
    // Emit fps for the masked stream. The pipeline is 30 fps in -> 30 fps out, so the masked stream
    // is emitted at the same rate as the AI-character export (cloud characters are 30 fps).
    private const val EMIT_FPS = 30

    fun generate(context: Context, uri: Uri, seg: YoloSeg, rvm: QuickMask, pose: MoveNet,
                 maxFrames: Int, workDim: Int, log: (String) -> Unit): Int {
        MiragePaths.ensureDirs()
        val poses = ArrayList<List<MoveNet.Person>>()
        var fps = 15; var W = 0; var H = 0

        // ---- P1: guided mask -> masked_video + mask ; cache pose ----
        FrameSource(context, uri).use { fs ->
            // Emit at the edge-style rate (like the real Tier-1's EMIT_FPS), so the masked stream
            // matches the AI-character export fps and the composite plays at real time.
            val emitFps = EMIT_FPS
            val stride = maxOf(1, Math.round(fs.srcFps / emitFps.toFloat()))
            fps = if (fs.srcFps >= emitFps) emitFps else fs.srcFps.coerceIn(1, 30)
            val total = if (fs.count > 0) minOf(fs.count / stride, maxFrames) else maxFrames
            val encM = Mp4Encoder(MiragePaths.maskedVideo, fps)
            val encK = Mp4Encoder(MiragePaths.maskVideo, fps)
            try {
                for (i in 0 until total) {
                    val raw = fs.frameAt(i * stride, workDim) ?: continue
                    val f = snap16(raw); if (f !== raw) raw.recycle()
                    W = f.width; H = f.height
                    val ppl = pose.detect(f); poses.add(ppl)
                    val mask = guidedMask(f, seg, rvm, ppl)          // the guided pipeline
                    val fpx = IntArray(W * H); f.getPixels(fpx, 0, W, 0, 0, W, H)
                    val mk = IntArray(W * H); val g = Color.rgb(GRAY, GRAY, GRAY)
                    for (i2 in 0 until W * H) if (mask[i2]) { fpx[i2] = g; mk[i2] = Color.WHITE } else mk[i2] = Color.BLACK
                    val masked = Bitmap.createBitmap(fpx, W, H, Bitmap.Config.ARGB_8888)
                    val mbmp = Bitmap.createBitmap(mk, W, H, Bitmap.Config.ARGB_8888)
                    encM.writeFrame(masked); encK.writeFrame(mbmp)
                    f.recycle(); masked.recycle(); mbmp.recycle()
                    if ((i + 1) % 10 == 0) log("tier-1 guided (mask+pose) ${i + 1}/$total")
                }
            } finally { encM.close(); encK.close() }
        }
        val n = poses.size
        if (n == 0) return 0

        // ---- P2: pose sticks + synthetic face over the masked frames ----
        FrameSource(context, Uri.fromFile(MiragePaths.maskedVideo)).use { ms ->
            val encP = Mp4Encoder(MiragePaths.tier1PoseSticks, fps)
            val encF = Mp4Encoder(MiragePaths.tier1SyntheticFace, fps)
            try {
                for (i in 0 until n) {
                    val m = ms.frameAt(i, 0) ?: continue
                    val ps = m.copy(Bitmap.Config.ARGB_8888, true)
                    Canvas(ps).let { c -> poses[i].forEachIndexed { pi, p -> drawSkeleton(c, p.kp, COLORS[pi % COLORS.size]) }
                        banner(c, ps.width, "TIER-1 - anonymised pose (on-device)", "MoveNet skeleton - identity-free") }
                    encP.writeFrame(ps)
                    val sf = m.copy(Bitmap.Config.ARGB_8888, true)
                    Canvas(sf).let { c -> poses[i].forEach { p -> drawSyntheticFace(c, p.kp) }
                        banner(c, sf.width, "TIER-1 - synthetic canonical face", "generic template on head keypoints - NOT the real face") }
                    encF.writeFrame(sf)
                    m.recycle(); ps.recycle(); sf.recycle()
                }
            } finally { encP.close(); encF.close() }
        }

        // ---- P3: canonicaliser panel ----
        val encC = Mp4Encoder(MiragePaths.tier1Canonical, fps)
        try {
            for (i in 0 until n) {
                val panel = Bitmap.createBitmap(W, H, Bitmap.Config.ARGB_8888)
                val c = Canvas(panel); c.drawColor(0xFF161620.toInt()); drawGrid(c, W, H)
                val ppl = poses[i]
                ppl.forEachIndexed { pi, p -> drawCanonical(c, p.kp, COLORS[pi % COLORS.size], W, H, pi, ppl.size) }
                banner(c, W, "TIER-1 - canonicaliser (identity-collapsed)", "root-centred + torso-normalised - on-device preview")
                encC.writeFrame(panel); panel.recycle()
            }
        } finally { encC.close() }
        return n
    }

    // ------------------------------------------------------------------ the GUIDED mask pipeline
    private fun guidedMask(f: Bitmap, seg: YoloSeg, rvm: QuickMask, ppl: List<MoveNet.Person>): BooleanArray {
        val W = f.width; val H = f.height; val plane = W * H
        val yolo = seg.personMask(f, W, H)                 // primary seg
        val matte = rvm.matte(f)                           // backstop matte (RVM alpha at f size)
        val I = FloatArray(plane); val p = FloatArray(plane)
        val fpx = IntArray(plane); f.getPixels(fpx, 0, W, 0, 0, W, H)
        for (i in 0 until plane) {
            val c = fpx[i]; I[i] = (Color.red(c) + Color.green(c) + Color.blue(c)) / 765f   // gray guide [0,1]
            p[i] = if (yolo[i] || matte.getOrElse(i) { 0f } > 0.5f) 1f else 0f              // seg ∪ matte
        }
        val q = guidedFilter(I, p, W, H, 12, 0.01f)        // edge-align to the frame
        var m = BooleanArray(plane) { q[it] > 0.5f }
        m = close(m, W, H, 2)                              // morph close
        m = dilate(m, W, H, 4)                             // d4 dilate
        faceGuard(m, W, H, ppl)                            // FaceGuard: guarantee the face region
        return m
    }

    // He et al. guided filter (box means via integral image). guide I, input p in [0,1].
    private fun guidedFilter(I: FloatArray, p: FloatArray, w: Int, h: Int, r: Int, eps: Float): FloatArray {
        val mI = box(I, w, h, r); val mp = box(p, w, h, r)
        val Ip = FloatArray(I.size) { I[it] * p[it] }; val II = FloatArray(I.size) { I[it] * I[it] }
        val mIp = box(Ip, w, h, r); val mII = box(II, w, h, r)
        val a = FloatArray(I.size) { (mIp[it] - mI[it] * mp[it]) / ((mII[it] - mI[it] * mI[it]) + eps) }
        val b = FloatArray(I.size) { mp[it] - a[it] * mI[it] }
        val ma = box(a, w, h, r); val mb = box(b, w, h, r)
        return FloatArray(I.size) { (ma[it] * I[it] + mb[it]).coerceIn(0f, 1f) }
    }

    private fun box(src: FloatArray, w: Int, h: Int, r: Int): FloatArray {
        val ii = DoubleArray((w + 1) * (h + 1))
        for (y in 0 until h) { var rs = 0.0; val o = (y + 1) * (w + 1); val op = y * (w + 1)
            for (x in 0 until w) { rs += src[y * w + x]; ii[o + x + 1] = ii[op + x + 1] + rs } }
        val out = FloatArray(w * h)
        for (y in 0 until h) { val y1 = maxOf(0, y - r); val y2 = minOf(h - 1, y + r)
            for (x in 0 until w) { val x1 = maxOf(0, x - r); val x2 = minOf(w - 1, x + r)
                val area = (y2 - y1 + 1) * (x2 - x1 + 1)
                val s = ii[(y2 + 1) * (w + 1) + (x2 + 1)] - ii[y1 * (w + 1) + (x2 + 1)] - ii[(y2 + 1) * (w + 1) + x1] + ii[y1 * (w + 1) + x1]
                out[y * w + x] = (s / area).toFloat() } }
        return out
    }

    private fun dilate(s: BooleanArray, w: Int, h: Int, r: Int) = morph(s, w, h, r, true)
    private fun erode(s: BooleanArray, w: Int, h: Int, r: Int) = morph(s, w, h, r, false)
    private fun close(s: BooleanArray, w: Int, h: Int, r: Int) = erode(dilate(s, w, h, r), w, h, r)
    private fun morph(s: BooleanArray, w: Int, h: Int, r: Int, dil: Boolean): BooleanArray {
        val tmp = BooleanArray(s.size); val out = BooleanArray(s.size)
        for (y in 0 until h) for (x in 0 until w) { var v = !dil; var k = -r
            while (k <= r) { val xx = x + k; if (xx in 0 until w) { val b = s[y * w + xx]; if (dil && b) { v = true; break }; if (!dil && !b) { v = false; break } }; k++ }
            tmp[y * w + x] = v }
        for (y in 0 until h) for (x in 0 until w) { var v = !dil; var k = -r
            while (k <= r) { val yy = y + k; if (yy in 0 until h) { val b = tmp[yy * w + x]; if (dil && b) { v = true; break }; if (!dil && !b) { v = false; break } }; k++ }
            out[y * w + x] = v }
        return out
    }

    /** FaceGuard: guarantee the face is covered - fill a head ellipse from the pose head keypoints. */
    private fun faceGuard(m: BooleanArray, w: Int, h: Int, ppl: List<MoveNet.Person>) {
        for (p in ppl) {
            val kp = p.kp
            fun ok(i: Int) = kp[i][2] > 0.2f
            if (!ok(0)) continue
            val nose = kp[0]
            val span = when {
                ok(3) && ok(4) -> hypot((kp[3][0] - kp[4][0]).toDouble(), (kp[3][1] - kp[4][1]).toDouble()).toFloat()
                ok(1) && ok(2) -> 2f * hypot((kp[1][0] - kp[2][0]).toDouble(), (kp[1][1] - kp[2][1]).toDouble()).toFloat()
                else -> continue
            }
            if (span < 4) continue
            val cx = nose[0]; val cy = nose[1] - span * 0.15f
            val rx = span * 0.72f; val ry = span * 0.95f
            val x1 = (cx - rx).toInt().coerceIn(0, w - 1); val x2 = (cx + rx).toInt().coerceIn(0, w)
            val y1 = (cy - ry).toInt().coerceIn(0, h - 1); val y2 = (cy + ry).toInt().coerceIn(0, h)
            for (y in y1 until y2) for (x in x1 until x2) {
                val dx = (x - cx) / rx; val dy = (y - cy) / ry
                if (dx * dx + dy * dy <= 1f) m[y * w + x] = true
            }
        }
    }

    // ------------------------------------------------------------------ drawing
    private fun ok(kp: Array<FloatArray>, i: Int) = kp[i][2] > 0.2f
    private fun drawSkeleton(c: Canvas, kp: Array<FloatArray>, color: Int) {
        val line = Paint(Paint.ANTI_ALIAS_FLAG).apply { this.color = color; strokeWidth = 4f; strokeCap = Paint.Cap.ROUND }
        val dot = Paint(Paint.ANTI_ALIAS_FLAG).apply { this.color = color }
        for ((a, b) in SKELETON) if (ok(kp, a) && ok(kp, b)) c.drawLine(kp[a][0], kp[a][1], kp[b][0], kp[b][1], line)
        for (i in 0 until 17) if (ok(kp, i)) c.drawCircle(kp[i][0], kp[i][1], 5f, dot)
    }

    private fun drawCanonical(c: Canvas, kp: Array<FloatArray>, color: Int, W: Int, H: Int, pi: Int, np: Int) {
        val mh = floatArrayOf((kp[11][0] + kp[12][0]) / 2, (kp[11][1] + kp[12][1]) / 2)
        val ms = floatArrayOf((kp[5][0] + kp[6][0]) / 2, (kp[5][1] + kp[6][1]) / 2)
        val torso = hypot((ms[0] - mh[0]).toDouble(), (ms[1] - mh[1]).toDouble()).toFloat().coerceAtLeast(1f)
        val s = H * 0.30f / torso
        val slotX = if (np <= 1) W / 2f else W.toFloat() * (pi + 1) / (np + 1)
        val t = Array(17) { floatArrayOf((kp[it][0] - mh[0]) * s + slotX, (kp[it][1] - mh[1]) * s + H * 0.55f, kp[it][2]) }
        drawSkeleton(c, t, color)
    }

    // synthetic face template (unit inter-ocular; origin at eye-mid) drawn on head keypoints.
    private fun drawSyntheticFace(c: Canvas, kp: Array<FloatArray>) {
        val mid: FloatArray; val ex: FloatArray; val io: Float
        if (ok(kp, 1) && ok(kp, 2)) {
            mid = floatArrayOf((kp[1][0] + kp[2][0]) / 2, (kp[1][1] + kp[2][1]) / 2)
            val dx = kp[2][0] - kp[1][0]; val dy = kp[2][1] - kp[1][1]; io = hypot(dx.toDouble(), dy.toDouble()).toFloat()
            ex = if (io > 1) floatArrayOf(dx / io, dy / io) else floatArrayOf(1f, 0f)
        } else if (ok(kp, 3) && ok(kp, 4)) {
            mid = floatArrayOf((kp[3][0] + kp[4][0]) / 2, (kp[3][1] + kp[4][1]) / 2)
            val dx = kp[4][0] - kp[3][0]; val dy = kp[4][1] - kp[3][1]; val d = hypot(dx.toDouble(), dy.toDouble()).toFloat()
            io = d * 0.5f; ex = if (d > 1) floatArrayOf(dx / d, dy / d) else floatArrayOf(1f, 0f)
        } else return
        if (io < 4) return
        val ey = floatArrayOf(-ex[1], ex[0])
        fun P(lx: Float, ly: Float) = floatArrayOf(mid[0] + io * (lx * ex[0] + ly * ey[0]), mid[1] + io * (lx * ex[1] + ly * ey[1]))
        val paint = Paint(Paint.ANTI_ALIAS_FLAG).apply { color = 0xFF78E678.toInt(); style = Paint.Style.STROKE; strokeWidth = 2f }
        // oval + inner ring (mesh feel)
        polyline(c, ringPts(0.40f, 1.05f, 1.40f) { lx, ly -> P(lx, ly) }, paint)
        polyline(c, ringPts(0.40f, 0.65f, 0.87f) { lx, ly -> P(lx, ly) }, paint)
        // eyes + iris
        polyline(c, ringAt(-0.5f, 0f, 0.22f, 0.12f) { lx, ly -> P(lx, ly) }, paint)
        polyline(c, ringAt(0.5f, 0f, 0.22f, 0.12f) { lx, ly -> P(lx, ly) }, paint)
        val (nx, ny) = P(0f, 0.1f); val (nx2, ny2) = P(0f, 0.55f); c.drawLine(nx, ny, nx2, ny2, paint)
        polyline(c, ringAt(0f, 0.92f, 0.34f, 0.11f) { lx, ly -> P(lx, ly) }, paint)
    }

    private inline fun ringPts(cy: Float, rx: Float, ry: Float, tf: (Float, Float) -> FloatArray): FloatArray {
        val n = 26; val out = FloatArray((n + 1) * 2)
        for (i in 0..n) { val t = 2 * Math.PI * i / n; val pt = tf((rx * cos(t)).toFloat(), (cy + ry * sin(t)).toFloat()); out[i * 2] = pt[0]; out[i * 2 + 1] = pt[1] }
        return out
    }
    private inline fun ringAt(cx: Float, cy: Float, rx: Float, ry: Float, tf: (Float, Float) -> FloatArray): FloatArray {
        val n = 16; val out = FloatArray((n + 1) * 2)
        for (i in 0..n) { val t = 2 * Math.PI * i / n; val pt = tf((cx + rx * cos(t)).toFloat(), (cy + ry * sin(t)).toFloat()); out[i * 2] = pt[0]; out[i * 2 + 1] = pt[1] }
        return out
    }
    private fun polyline(c: Canvas, pts: FloatArray, paint: Paint) {
        var i = 0; while (i + 3 < pts.size) { c.drawLine(pts[i], pts[i + 1], pts[i + 2], pts[i + 3], paint); i += 2 }
    }

    private fun drawGrid(c: Canvas, W: Int, H: Int) {
        val g = Paint().apply { color = 0xFF20202A.toInt(); strokeWidth = 1f }
        var x = 0; while (x < W) { c.drawLine(x.toFloat(), 34f, x.toFloat(), H.toFloat(), g); x += 60 }
        var y = 34; while (y < H) { c.drawLine(0f, y.toFloat(), W.toFloat(), y.toFloat(), g); y += 60 }
    }
    private fun banner(c: Canvas, w: Int, title: String, sub: String) {
        c.drawRect(0f, 0f, w.toFloat(), 34f, Paint().apply { color = 0xFF12121A.toInt() })
        c.drawText(title, 10f, 24f, Paint(Paint.ANTI_ALIAS_FLAG).apply { color = Color.WHITE; textSize = 22f; typeface = Typeface.DEFAULT_BOLD })
    }

    private fun snap16(b: Bitmap): Bitmap {
        val w = (b.width / 16 * 16).coerceAtLeast(16); val h = (b.height / 16 * 16).coerceAtLeast(16)
        return if (w == b.width && h == b.height) b else Bitmap.createScaledBitmap(b, w, h, true)
    }
}
