package com.mirage.npu

import android.app.Dialog
import android.content.Context
import android.content.Intent
import android.graphics.Bitmap
import android.graphics.Canvas
import android.graphics.Color
import android.graphics.Paint
import android.graphics.Path
import android.graphics.RectF
import android.graphics.Typeface
import android.media.MediaMetadataRetriever
import android.net.Uri
import android.os.Bundle
import android.os.Environment
import android.os.Handler
import android.os.Looper
import android.provider.DocumentsContract
import android.provider.Settings
import android.view.Gravity
import android.view.MotionEvent
import android.view.View
import android.view.ViewGroup
import android.widget.ArrayAdapter
import android.widget.Button
import android.widget.CheckBox
import android.widget.FrameLayout
import android.widget.ImageView
import android.widget.LinearLayout
import android.widget.ProgressBar
import android.widget.ScrollView
import android.widget.SeekBar
import android.widget.Spinner
import android.widget.TextView
import android.widget.VideoView
import androidx.activity.result.contract.ActivityResultContracts
import androidx.appcompat.app.AlertDialog
import androidx.appcompat.app.AppCompatActivity
import androidx.core.view.ViewCompat
import androidx.core.view.WindowInsetsCompat
import java.io.File

/**
 * MainActivity - a four-page operator console over the on-device Tier-1 + Tier-2 pipeline.
 *
 *   TIER-1  get a clip in (pick an already-masked clip, or run the guided Tier-1 on a raw one) and
 *           stage the individual input slots.
 *   TIER-2  the pipeline itself as a stepper: ② Phase 1 inpaint · ②ᵇ Phase 1b lightmap ·
 *           ③ Phase 2 composite, each with a state pill (DONE / READY /
 *           BLOCKED·why / QUEUED), an input|output collage and its last measurement. **This is the
 *           page the app opens on.**
 *   RESULT  original-vs-final compare, the EVALS history, and where the files live.
 *   TOOLS   §2 audit toggle, the on-device NPU placement gates, permissions, reset.
 *
 * Header (MIRAGE · readiness chip · NPU toggle · engine chip) and footer (phase label with a live
 * elapsed/ETA ticker, progress bar, copyable terminal) are FIXED chrome, outside the pages.
 *
 * 🔴 AUTOMATION CONTRACT - read before moving any view. Every real run of this app is driven over adb
 * by `_e2e/cloud_a20_20260801/drive_phone.py`, which does `uiautomator dump`, finds a node by its
 * resource-id and taps the centre of its bounds. A view that is not in the accessibility tree does not
 * exist to it, and a GONE page's children are not in that tree. Therefore:
 *   * `npuToggle` lives in the fixed HEADER, so it is reachable from every page without navigating.
 *   * `p1run`, `p1brun`, `p2run` all live on `pageTier2`, and `pageTier2` is selected at
 *     launch. The selected page is deliberately NOT persisted - `am start` must always land there.
 *   * `_e2e/uiContract.json` maps every automation id to its page and is the file a future rework
 *     must update. `drive_phone.select_page()` reads it.
 *   * Run buttons are never disabled by readiness: a disabled button makes an automation tap a silent
 *     no-op and the driver then blocks for its full timeout. The state pill says why it will fail.
 *
 * Reads/writes the canonical shared-folder filenames (MiragePaths), so each phase auto-feeds the next;
 * per-phase evaluations are appended to output/EVALS.md.
 * PRIVACY: no INTERNET permission; all compute on-device.
 */
class MainActivity : AppCompatActivity() {

    private lateinit var logText: TextView
    private lateinit var logScroll: ScrollView
    private lateinit var pathText: TextView
    private lateinit var progressBar: ProgressBar
    private lateinit var npuToggle: CheckBox
    // one input+output COLLAGE per phase (tap left half = input, right half = output)
    private lateinit var t1coll: ImageView      // Tier-1 edge outputs (masked · pose · canonical · face mesh)
    private lateinit var cmpcoll: ImageView     // bottom compare: masked input vs final output
    private lateinit var p1coll: ImageView
    private lateinit var p1bcoll: ImageView
    private lateinit var p2coll: ImageView
    private lateinit var p2hint: TextView
    private val runButtons = ArrayList<Button>()

    // ── PAGES ────────────────────────────────────────────────────────────────────────────────────
    // Index order IS the swipe order and the index selectPage() takes. Tab and page ids are paired
    // positionally; both are permanent and both are listed in _e2e/uiContract.json.
    private val tabIds = intArrayOf(R.id.tabTier1, R.id.tabTier2, R.id.tabResult, R.id.tabTools)
    private val pageIds = intArrayOf(R.id.pageTier1, R.id.pageTier2, R.id.pageResult, R.id.pageTools)
    private val pageNames = arrayOf("Tier-1", "Tier-2", "Result", "Tools")
    private var page = PAGE_TIER2
    // horizontal-fling bookkeeping (see dispatchTouchEvent)
    private var downX = 0f
    private var downY = 0f
    private var downT = 0L
    private var swipePx = 120f

    // ── LONG-JOB TICKER ──────────────────────────────────────────────────────────────────────────
    private val ui = Handler(Looper.getMainLooper())
    private var tick: Runnable? = null
    private var runT0 = 0L
    private var runName = ""
    /** Estimated total seconds for the phase now running, or <=0 when nothing can be estimated yet. */
    private var etaTotalS = 0.0
    private var etaBasis = ""

    // Inputs card: pick each input file one-by-one into its canonical slot.
    private lateinit var inputSpinner: Spinner
    private data class InputSlot(val label: String, val dest: File)
    private var inputSlots: List<InputSlot> = emptyList()

    private var outTreeUri: Uri? = null
    private var pickTarget: File? = null

    private val pickInput = registerForActivityResult(ActivityResultContracts.OpenDocument()) { uri: Uri? ->
        val dst = pickTarget
        if (uri == null || dst == null) { appendLog("pick cancelled."); return@registerForActivityResult }
        if (!hasAllFilesAccess()) { appendLog("need All-Files access first - tap 'Grant files'."); return@registerForActivityResult }
        runCatching {
            dst.parentFile?.mkdirs()
            contentResolver.openInputStream(uri)!!.use { i -> dst.outputStream().use { o -> i.copyTo(o) } }
            // A freshly-PICKED masked clip has no known raw original, so a leftover original_input.mp4
            // would belong to a DIFFERENT clip and Compare would show a mismatched pair.
            //
            // But blanket-deleting also made it impossible to supply the raw original yourself: an
            // eval workflow stages masked_video + mask + characters by push and legitimately HAS the
            // original. Deleting it meant Compare could never show "original vs final" for anything but
            // the on-phone Tier-1 path (fixed 2026-07-26).
            //
            // Resolution: only drop the original when it CANNOT match - i.e. its frame count differs
            // from the masked clip being staged. Equal frame counts is a weak test, not proof of
            // correspondence, so the Compare label says "original" only when it also matches; the
            // fallback label names what is actually shown.
            if (dst == MiragePaths.maskedVideo && MiragePaths.originalInput.exists()) {
                val same = runCatching { frameCountOf(dst) == frameCountOf(MiragePaths.originalInput) }
                    .getOrDefault(false)
                if (!same) {
                    MiragePaths.originalInput.delete()
                    appendLog("dropped a stale original_input.mp4 (frame count differs from the new masked clip)")
                } else appendLog("kept original_input.mp4 (frame count matches the staged masked clip)")
            }
            appendLog("staged ${dst.name}")
            refreshInputSpinner(); refreshPreviews(); refreshStatus()
        }.onFailure { appendLog("stage failed: ${it.message}") }
    }

    private val pickFolder = registerForActivityResult(ActivityResultContracts.OpenDocumentTree()) { uri: Uri? ->
        if (uri == null) return@registerForActivityResult
        runCatching { contentResolver.takePersistableUriPermission(uri, Intent.FLAG_GRANT_WRITE_URI_PERMISSION or Intent.FLAG_GRANT_READ_URI_PERMISSION) }
        outTreeUri = uri
        getSharedPreferences("mirage", Context.MODE_PRIVATE).edit().putString("out_tree", uri.toString()).apply()
        appendLog("output folder set: $uri"); refreshStatus()
    }

    // Raw clip -> the GUIDED Tier-1 pipeline runs on-device -> masked_video + mask + Tier-1 sidecars.
    private val pickForQuickMask = registerForActivityResult(ActivityResultContracts.OpenDocument()) { uri: Uri? ->
        if (uri == null) { appendLog("pick cancelled."); return@registerForActivityResult }
        runTier1OnPhone(uri)
    }

    override fun onCreate(savedInstanceState: Bundle?) {
        super.onCreate(savedInstanceState)
        // QNN/HTP: point the DSP at the APK's extracted skel libs BEFORE any OrtEnvironment is created.
        runCatching { android.system.Os.setenv("ADSP_LIBRARY_PATH", applicationInfo.nativeLibraryDir, true) }
        setContentView(R.layout.activity_main)

        // Tabs first, so nothing below can observe a half-selected page. ALWAYS start on Tier-2:
        // persisting the last tab would make `am start` -> find("p1run") non-deterministic, and the
        // four Run buttons the adb driver taps all live on that page.
        swipePx = 60f * resources.displayMetrics.density
        for (i in tabIds.indices) findViewById<TextView>(tabIds[i]).setOnClickListener { selectPage(i, true) }
        selectPage(PAGE_TIER2)

        logText = findViewById(R.id.logText); logScroll = findViewById(R.id.logScroll)
        pathText = findViewById(R.id.pathText); progressBar = findViewById(R.id.progressBar)
        npuToggle = findViewById(R.id.npuToggle)
        t1coll = findViewById(R.id.t1coll)
        cmpcoll = findViewById(R.id.cmpColl)
        p1coll = findViewById(R.id.p1coll)
        p1bcoll = findViewById(R.id.p1bcoll)
        p2coll = findViewById(R.id.p2coll)
        p2hint = findViewById(R.id.p2hint)
        // Label it for what it actually is. "NNAPI" was misleading in the QNN build (the live NPU path
        // is Qualcomm's plugin EP on the Hexagon HTP, not NNAPI - which is deprecated at API 35, the
        // level this app targets, and has never produced a single latency number in the ledger).
        // SHORT here because the toggle now sits in the fixed header next to the readiness chip; the
        // full sentence ("NPU armed (Hexagon HTP)") is on engineChip directly underneath it.
        npuToggle.text = if (NpuFactory.ORT_FLAVOR == "qnn") "NPU" else "NNAPI"
        // PERSIST IT. The toggle used to reset to OFF on every process start, so a run the operator
        // believed was an "NPU arm" silently measured CPU - the ledger flags exactly that hazard.
        val prefs = getSharedPreferences("mirage", MODE_PRIVATE)
        npuToggle.isChecked = prefs.getBoolean("accel", NpuFactory.ORT_FLAVOR == "qnn")
        npuToggle.setOnCheckedChangeListener { _, on ->
            prefs.edit().putBoolean("accel", on).apply()
            appendLog(if (on) "accel ON - Hexagon HTP where a QNN context exists"
                      else "accel OFF - every model on the CPU EP")
            updateEngineChip()
        }
        updateEngineChip()

        // Tools wired at CREATE time, not inside the toggle's listener (where an earlier edit
        // mistakenly put them, so the readiness panel only populated if you happened to tap the
        // checkbox).
        val s2Toggle = findViewById<CheckBox>(R.id.s2Toggle)
        s2Toggle.isChecked = prefs.getBoolean("s2audit", false)
        PhaseRunner.s2AuditEnabled = s2Toggle.isChecked
        s2Toggle.setOnCheckedChangeListener { _, on ->
            prefs.edit().putBoolean("s2audit", on).apply()
            PhaseRunner.s2AuditEnabled = on
            appendLog(if (on) "§2 leak audit ON - verification tool, costs ~130 ms/frame"
                      else "§2 leak audit OFF - EVALS will report NOT AUDITED, which is honest, not a pass")
        }
        // ---- Phase 1 · BACKGROUND FILL SHARPNESS -------------------------------------------------
        // The knob FIX12 measured, now operator-facing: how many temporal samples get mixed per hole
        // pixel (fewer = a smaller mixing radius = sharper, since every extra sample is a donor
        // further away in time and their mean is a σ≈0.5 px blur) and how long the nearest-first
        // search window is (longer = the strict walk still finds donors instead of falling back to
        // the smooth plate). PERSISTED, and applied at CREATE time - an unpersisted pixel-affecting
        // knob is how a run gets recorded under the wrong config.
        val sharpBar = findViewById<SeekBar>(R.id.p1SharpBar)
        val sharpLabel = findViewById<TextView>(R.id.p1SharpLabel)
        val cfg0 = RuntimeConfig.load(this)
        val startSharp = prefs.getInt("fillsharp", cfg0.fillSharpness).coerceIn(0, BackgroundInpaint.SHARP_MAX)
        BackgroundInpaint.setFillSharpness(startSharp)
        sharpBar.progress = startSharp
        sharpLabel.text = "fill sharpness ${BackgroundInpaint.describeFillSharpness()}"
        sharpBar.setOnSeekBarChangeListener(object : SeekBar.OnSeekBarChangeListener {
            override fun onProgressChanged(sb: SeekBar, v: Int, fromUser: Boolean) {
                BackgroundInpaint.setFillSharpness(v)
                sharpLabel.text = "fill sharpness ${BackgroundInpaint.describeFillSharpness()}"
            }
            override fun onStartTrackingTouch(sb: SeekBar) {}
            override fun onStopTrackingTouch(sb: SeekBar) {
                prefs.edit().putInt("fillsharp", sb.progress).apply()
                appendLog("background fill sharpness -> ${BackgroundInpaint.describeFillSharpness()}" +
                    " · re-run Phase 1 to see it (this changes pixels; nothing already rendered moves)")
            }
        })

        // ---- Phase 1 · SEAM MATCH ----------------------------------------------------------------
        // The second half of "make the fill look real": how hard it is matched to the pixels it
        // actually touches. Same persistence contract as the sharpness slider above.
        val seamBar = findViewById<SeekBar>(R.id.p1SeamBar)
        val seamLabel = findViewById<TextView>(R.id.p1SeamLabel)
        val startSeam = prefs.getInt("seammatch", cfg0.seamMatch).coerceIn(0, BackgroundInpaint.SEAM_MAX)
        BackgroundInpaint.setSeamMatch(startSeam)
        seamBar.progress = startSeam
        seamLabel.text = "seam match ${BackgroundInpaint.describeSeamMatch()}"
        seamBar.setOnSeekBarChangeListener(object : SeekBar.OnSeekBarChangeListener {
            override fun onProgressChanged(sb: SeekBar, v: Int, fromUser: Boolean) {
                BackgroundInpaint.setSeamMatch(v)
                seamLabel.text = "seam match ${BackgroundInpaint.describeSeamMatch()}"
            }
            override fun onStartTrackingTouch(sb: SeekBar) {}
            override fun onStopTrackingTouch(sb: SeekBar) {
                prefs.edit().putInt("seammatch", sb.progress).apply()
                appendLog("background fill seam match -> ${BackgroundInpaint.describeSeamMatch()}" +
                    " · re-run Phase 1 to see it (this changes pixels)")
            }
        })

        val p2Alpha = findViewById<CheckBox>(R.id.p2AlphaToggle)
        p2Alpha.isChecked = prefs.getBoolean("p2alpha", false)
        PhaseRunner.writeCompositeAlpha = p2Alpha.isChecked
        p2Alpha.setOnCheckedChangeListener { _, on ->
            prefs.edit().putBoolean("p2alpha", on).apply()
            PhaseRunner.writeCompositeAlpha = on
            appendLog(if (on) "composite_alpha ON - a second encoder, ~8 s per 300 frames"
                      else "composite_alpha OFF - reference artifact only, nothing in the app reads it")
        }
        // 🔴 The explicit alpha is the PRIMARY matte path (owner, 2026-08-08). This toggle exists so
        // the keyer downgrade can be taken DELIBERATELY and is never taken by accident; default OFF,
        // and it is NOT persisted as ON silently - every run that has it on says so in the log and
        // in the EVALS row.
        val p2Keyer = findViewById<CheckBox>(R.id.p2KeyerToggle)
        p2Keyer.isChecked = prefs.getBoolean("p2keyer", false)
        PhaseRunner.allowKeyerFallback = p2Keyer.isChecked
        if (p2Keyer.isChecked) appendLog(
            "⚠ keyer fallback is ON from a previous session - Phase 2 will DOWNGRADE a layer that " +
                "has no synthetic_alpha_pK.mp4 instead of refusing. Untick it for a normal run.")
        p2Keyer.setOnCheckedChangeListener { _, on ->
            prefs.edit().putBoolean("p2keyer", on).apply()
            PhaseRunner.allowKeyerFallback = on
            appendLog(if (on)
                "⚠⚠ keyer fallback ALLOWED - a missing alpha will be KEYED, not refused. Measured " +
                    "cost: see-through 0.37 % → 9.57 %, composite 328 → 531 ms/f, §2 leak 0.000 % → " +
                    "0.158 %. Do not pair this run's numbers with an explicit-alpha run."
            else "explicit alpha REQUIRED - a layer without synthetic_alpha_pK.mp4 refuses Phase 2")
        }
        findViewById<Button>(R.id.segNpuBtn).setOnClickListener {
            appendLog("---- compiling YOLO11n-seg for the Hexagon, then gating it ----"); setBusy(true)
            showPhase("compiling + gating seg …")
            Thread {
                val r = runCatching { SegNpuGate.run(applicationContext) { l -> post { appendLog(l) } } }
                post {
                    setBusy(false); showPhase(null)
                    r.onSuccess { g ->
                        appendLog(if (g.passed)
                            "seg → NPU ACCEPTED: mean IoU ${"%.4f".format(g.meanIou)}, min ${"%.4f".format(g.minIou)}, " +
                                "area Δ ${"%.3f".format(g.maxAreaDelta * 100)}% over ${g.frames} frames · " +
                                "inference ${"%.1f".format(g.cpuMs)} → ${"%.1f".format(g.npuMs)} ms/frame. Used from the next Tier-1 run."
                        else
                            "seg → NPU REJECTED (${g.note}). Nothing switched over; tap 'Seg → CPU' to remove the context.")
                        refreshReadiness()
                    }.onFailure { appendLog("seg gate ERROR: ${it.message}") }
                }
            }.start()
        }
        findViewById<Button>(R.id.segNpuOffBtn).setOnClickListener {
            val d = java.io.File(ModelStore.modelsDir(applicationContext), "seg_qnn")
            val gone = d.deleteRecursively()
            appendLog(if (gone) "seg_qnn context removed - seg is back on the CPU EP." else "no seg_qnn context to remove.")
            refreshReadiness()
        }
        findViewById<Button>(R.id.recheckBtn).setOnClickListener { refreshReadiness(); appendLog("readiness re-checked.") }
        findViewById<Button>(R.id.evalsBtn).setOnClickListener { showEvals() }
        findViewById<Button>(R.id.logCopyBtn).setOnClickListener {
            val cm = getSystemService(CLIPBOARD_SERVICE) as android.content.ClipboardManager
            cm.setPrimaryClip(android.content.ClipData.newPlainText("mirage log", logText.text))
            appendLog("log copied to clipboard.")
        }
        findViewById<Button>(R.id.logClearBtn).setOnClickListener { logText.text = ""; appendLog("log cleared.") }
        refreshReadiness()

        // system-bar insets so the UI clears the status bar (top) and nav buttons (bottom)
        val root = findViewById<View>(R.id.rootLayout)
        ViewCompat.setOnApplyWindowInsetsListener(root) { v, insets ->
            val b = insets.getInsets(WindowInsetsCompat.Type.systemBars())
            v.setPadding(v.paddingLeft, b.top, v.paddingRight, b.bottom); insets
        }

        getSharedPreferences("mirage", Context.MODE_PRIVATE).getString("out_tree", null)?.let {
            outTreeUri = runCatching { Uri.parse(it) }.getOrNull()
        }

        // Primary "Pick input video" button (top of the scroll): stage the chosen clip as the Tier-1
        // masked input and start a fresh run. Same destination slot as Phase 1's "Set masked".
        findViewById<Button>(R.id.pickInputBtn).setOnClickListener {
            pickTarget = MiragePaths.maskedVideo; appendLog("pick your input clip → staged as masked_video"); launchPick()
        }
        findViewById<Button>(R.id.p1set).setOnClickListener { pickTarget = MiragePaths.maskedVideo; launchPick() }
        // Run the GUIDED Tier-1 pipeline on a RAW clip, on-device -> masked_video + mask + Tier-1 sidecars.
        findViewById<Button>(R.id.quickMaskBtn).setOnClickListener {
            appendLog("Tier-1 on-device: pick a RAW clip (a real person, no grey holes)")
            runCatching { pickForQuickMask.launch(arrayOf("video/*")) }.onFailure { appendLog("picker error: ${it.message}") }
        }
        findViewById<Button>(R.id.p2add).setOnClickListener {
            val n = MiragePaths.personLayers().size + 1
            pickTarget = File(MiragePaths.inputDir, "synthetic_person_p$n.mp4"); launchPick()
        }
        val b1 = findViewById<Button>(R.id.p1run); val b2 = findViewById<Button>(R.id.p2run)
        val bf = findViewById<Button>(R.id.fullAuto)
        val blm = findViewById<Button>(R.id.p1brun)
        runButtons.addAll(listOf(b1, b2, bf, blm))
        blm.setOnClickListener { runPhase("Lightmap") { cfg -> listOf(PhaseRunner.phaseLightmap(applicationContext, cfg, logCb())) } }
        b1.setOnClickListener { runPhase("Phase 1") { cfg -> listOf(PhaseRunner.phase1Inpaint(applicationContext, npuToggle.isChecked, cfg, logCb())) } }
        b2.setOnClickListener { runPhase("Phase 2") { cfg -> listOf(PhaseRunner.phase2Composite(applicationContext, cfg, logCb())) } }
        bf.setOnClickListener { runPhase("Full pipeline") { cfg -> PhaseRunner.fullAuto(applicationContext, npuToggle.isChecked, cfg, logCb()) } }

        findViewById<Button>(R.id.grantButton).setOnClickListener { requestAllFilesAccess() }
        findViewById<Button>(R.id.outFolderButton).setOnClickListener { runCatching { pickFolder.launch(null) } }
        findViewById<Button>(R.id.resetBtn).setOnClickListener { confirmClearOutputs() }

        // Inputs card: choose a target slot, then stage a picked video into it (same SAF flow as the
        // Phase shortcut buttons, but for ANY input - masked_video, mask, character pK, alpha pK).
        inputSpinner = findViewById(R.id.inputSpinner)
        findViewById<Button>(R.id.inPickBtn).setOnClickListener {
            val pos = inputSpinner.selectedItemPosition
            val slot = inputSlots.getOrNull(pos)
            if (slot == null) { appendLog("pick an input slot first"); return@setOnClickListener }
            pickTarget = slot.dest
            appendLog("pick a video to stage as ${slot.dest.name}"); launchPick()
        }
        findViewById<Button>(R.id.inClearBtn).setOnClickListener { clearSelectedInput() }

        refreshInputSpinner(); refreshPreviews(); refreshStatus()
    }

    override fun onResume() { super.onResume(); refreshInputSpinner(); refreshPreviews(); refreshStatus() }

    private fun launchPick() = runCatching { pickInput.launch(arrayOf("video/*")) }
        .onFailure { appendLog("picker error: ${it.message}") }

    private fun logCb() = PhaseRunner.Logger { line -> post { appendLog(line) } }

    /**
     * Un-stage the file in the currently selected Inputs slot.
     *
     * This exists because a LEFTOVER input silently changes the pipeline: [MiragePaths.personLayers]
     * auto-detects every `synthetic_person_pK.mp4` in input/, so a p2 character left behind from an
     * earlier clip gets composited into the next run's scene with no warning. Re-picking a file can
     * overwrite a slot but never remove one, so without this there was no way back to one person.
     *
     * Clearing a CHARACTER also offers to drop its paired alpha: an alpha whose person is gone is never
     * read again (personLayers keys off the person file), so leaving it behind only invites the same
     * confusion later. The pairing is stated in the confirm dialog rather than done silently.
     */
    private fun clearSelectedInput() {
        val slot = inputSlots.getOrNull(inputSpinner.selectedItemPosition)
        if (slot == null) { appendLog("pick an input slot first"); return }
        if (!slot.dest.exists()) { appendLog("${slot.dest.name} is not staged - nothing to clear"); return }
        if (!hasAllFilesAccess()) { appendLog("need All-Files access first - tap 'Grant files'."); return }

        val pairedAlpha = Regex("^synthetic_person_p(\\d+)\\.mp4$").find(slot.dest.name)
            ?.groupValues?.get(1)
            ?.let { File(MiragePaths.inputDir, "synthetic_alpha_p$it.mp4") }
            ?.takeIf { it.exists() }
        val targets = listOfNotNull(slot.dest, pairedAlpha)
        val msg = buildString {
            append("Remove ")
            append(targets.joinToString(" and ") { it.name })
            append(" from input/?")
            if (pairedAlpha != null) append("\n\nThe alpha is included because it is only ever read alongside its character.")
            append("\n\nThis deletes the staged copy only - the original file you picked from is untouched.")
        }
        AlertDialog.Builder(this)
            .setTitle("Clear input")
            .setMessage(msg)
            .setNegativeButton("Cancel", null)
            .setPositiveButton("Clear") { _, _ ->
                var gone = 0
                for (f in targets) {
                    if (runCatching { f.delete() }.getOrDefault(false)) { appendLog("cleared ${f.name}"); gone++ }
                    else appendLog("could not clear ${f.name}")
                }
                if (gone > 0) { refreshInputSpinner(); refreshPreviews(); refreshStatus() }
            }
            .show()
    }

    /** Delete all pipeline OUTPUTS (never inputs) so a fresh run starts clean and every preview resets
     *  to "not ready". Confirmed, because it removes finished results. The staged inputs (masked video,
     *  mask, characters) and the Tier-1 sidecars are all kept. */
    private fun confirmClearOutputs() {
        val outs = listOf(
            MiragePaths.backgroundReconstructed, MiragePaths.lightMap, MiragePaths.lightMapPlate,
            MiragePaths.compositeVideo, MiragePaths.compositeAlpha, MiragePaths.finalOutput,
            MiragePaths.finalPreview
        ).filter { it.exists() }
        if (outs.isEmpty()) { appendLog("no outputs to clear - already fresh."); return }
        AlertDialog.Builder(this)
            .setTitle("Reset run")
            .setMessage("Delete ${outs.size} generated output(s) - background, light map, light-map plate, composite, final?\n\n" +
                "Your INPUTS (masked video, mask, characters) are kept, so you can re-run from scratch.")
            .setNegativeButton("Cancel", null)
            .setPositiveButton("Reset") { _, _ ->
                val gone = outs.count { runCatching { it.delete() }.getOrDefault(false) }
                appendLog("reset: cleared $gone output file(s)"); refreshPreviews(); refreshStatus()
            }
            .show()
    }

    /** Rebuild the Inputs spinner with every stage-able slot in pipeline order: masked_video, mask, then
     *  character pK + its alpha for each existing person, plus one "(new)" character slot to add the next.
     *  A "✓" prefix marks a slot whose file is already staged. Preserves the selection across rebuilds. */
    private fun refreshInputSpinner() {
        val inDir = MiragePaths.inputDir
        fun mark(f: File) = if (f.exists()) "✓" else "·"
        val slots = ArrayList<InputSlot>()
        slots.add(InputSlot("${mark(MiragePaths.maskedVideo)} masked_video", MiragePaths.maskedVideo))
        slots.add(InputSlot("${mark(MiragePaths.maskVideo)} mask", MiragePaths.maskVideo))
        val maxIdx = MiragePaths.personLayers().maxOfOrNull { it.index } ?: 0
        for (k in 1..(maxIdx + 1)) {
            val person = File(inDir, "synthetic_person_p$k.mp4")
            val alpha = File(inDir, "synthetic_alpha_p$k.mp4")
            val tag = if (k == maxIdx + 1) " (new)" else ""
            slots.add(InputSlot("${mark(person)} character p$k$tag", person))
            slots.add(InputSlot("${mark(alpha)} alpha p$k$tag", alpha))
        }
        inputSlots = slots
        val keep = inputSpinner.selectedItemPosition.coerceIn(0, slots.size - 1)
        val adapter = object : ArrayAdapter<String>(this, android.R.layout.simple_spinner_item, slots.map { it.label }) {
            override fun getView(position: Int, convertView: View?, parent: ViewGroup): View =
                (super.getView(position, convertView, parent) as TextView).apply { setTextColor(Color.WHITE); textSize = 13f }
            override fun getDropDownView(position: Int, convertView: View?, parent: ViewGroup): View =
                (super.getDropDownView(position, convertView, parent) as TextView).apply {
                    setTextColor(Color.WHITE); setBackgroundColor(Color.parseColor("#241E33")); textSize = 14f; setPadding(28, 22, 28, 22)
                }
        }
        adapter.setDropDownViewResource(android.R.layout.simple_spinner_dropdown_item)
        inputSpinner.adapter = adapter
        inputSpinner.setSelection(keep)
    }

    /** Run a phase off the UI thread with the progress bar + terminal, then refresh previews + save. */
    private fun runPhase(name: String, block: (RuntimeConfig) -> List<PhaseRunner.PhaseEval>) {
        appendLog("---- $name ----")
        setBusy(true)
        startTicker(name)
        val t0 = android.os.SystemClock.elapsedRealtime()
        val cfg = RuntimeConfig.load(applicationContext) { line -> post { appendLog(line) } }
        Thread {
            val res = runCatching { block(cfg) }
            post {
                setBusy(false)
                stopTicker()
                res.onSuccess { evals ->
                    PhaseRunner.writeEvals(evals, logCb())
                    evals.forEach { e -> e.output?.let { saveToOutputFolder(it) } }
                    evals.forEach { publishPhaseStat(it) }
                    refreshPreviews()
                    val secs = (android.os.SystemClock.elapsedRealtime() - t0) / 1000.0
                    appendLog("$name OK in ${"%.1f".format(secs)} s.")
                }.onFailure { appendLog("$name ERROR: ${it.message}") }
                // Whether it succeeded or threw, the on-disk state changed (or provably did not), so
                // the pills must be recomputed - a stale "READY" after a failure is the exact kind of
                // lie this rework exists to remove.
                refreshStatus()
            }
        }.start()
    }

    // ═══════════════════════ PAGES ═══════════════════════════════════════════════════════════════

    /**
     * Show page [idx] and hide the other three.
     *
     * 🔴 This is the ONLY way a page becomes visible, and visibility is what puts a view into the
     * accessibility tree - i.e. this is the function that decides what `uiautomator dump` can see.
     * `drive_phone.select_page()` reaches it by tapping the matching `tab*` id.
     */
    private fun selectPage(idx: Int, log: Boolean = false) {
        val i = idx.coerceIn(0, pageIds.size - 1)
        page = i
        for (k in pageIds.indices) {
            findViewById<View>(pageIds[k]).visibility = if (k == i) View.VISIBLE else View.GONE
            findViewById<TextView>(tabIds[k]).apply {
                setTextColor(getColor(if (k == i) R.color.tab_on else R.color.tab_off))
                setBackgroundColor(if (k == i) getColor(R.color.surface2) else Color.TRANSPARENT)
            }
        }
        if (log) appendLog("page → ${pageNames[i]}")
    }

    /**
     * Horizontal fling → next/previous page ("make screen shiftable"), for the human only.
     *
     * Hand-rolled rather than a GestureDetector so the event is never consumed: the ScrollViews keep
     * every one of their own gestures, and the adb driver's VERTICAL `input swipe 540 1500 540 500`
     * cannot trip it (dx = 0). The guards - under 600 ms, more than ~60 dp of travel, and at least
     * twice as horizontal as it is vertical - are what separate a deliberate page flick from a
     * diagonal scroll.
     */
    override fun dispatchTouchEvent(ev: MotionEvent): Boolean {
        when (ev.actionMasked) {
            MotionEvent.ACTION_DOWN -> { downX = ev.x; downY = ev.y; downT = ev.eventTime }
            MotionEvent.ACTION_UP -> {
                val dx = ev.x - downX
                val dy = ev.y - downY
                val dt = ev.eventTime - downT
                if (downT != 0L && dt in 1..600 &&
                    kotlin.math.abs(dx) > swipePx && kotlin.math.abs(dx) > 2f * kotlin.math.abs(dy)
                ) selectPage(page + if (dx < 0) 1 else -1)
                downT = 0L
            }
        }
        return super.dispatchTouchEvent(ev)
    }

    // ═══════════════════════ STAGE STATE ═════════════════════════════════════════════════════════

    /**
     * Recompute the four stage pills (plus Tier-1's) from what is actually on disk.
     *
     * The old UI presented every Run button as equally available at all times, so "Phase 2 with no
     * character staged" looked identical to "Phase 2 ready" until the exception landed in the
     * terminal. A pill answers the three questions that matter before you tap: is it DONE, is it
     * READY, or is it BLOCKED - and if blocked, WHICH file is missing.
     *
     * QUEUED vs BLOCKED is a real distinction, not decoration: QUEUED means an earlier stage simply
     * has not run yet and running it fixes this one; BLOCKED means a file only YOU can supply is
     * absent, and no amount of running earlier stages will help.
     */
    private fun refreshStages() {
        val masked = MiragePaths.maskedVideo.exists()
        val mask = MiragePaths.maskVideo.exists()
        val bg = MiragePaths.backgroundReconstructed.exists()
        val lm = MiragePaths.lightMap.exists()
        val comp = MiragePaths.compositeVideo.exists()
        val layers = runCatching { MiragePaths.personLayers().size }.getOrDefault(0)
        val md = ModelStore.modelsDir(applicationContext)
        val t1Models = File(md, "seg.onnx").exists() && File(md, "rvm.onnx").exists() &&
            File(md, "movenet.onnx").exists()

        // ── Tier-1 ──
        when {
            masked && mask -> {
                val side = listOf(MiragePaths.tier1PoseSticks, MiragePaths.tier1Canonical,
                    MiragePaths.tier1SyntheticFace).count { it.exists() }
                pill(R.id.t1state, "DONE", R.color.state_done, "masked + mask staged · $side/3 sidecars")
            }
            masked -> pill(R.id.t1state, "DONE", R.color.state_done, "masked staged; mask will be derived")
            t1Models -> pill(R.id.t1state, "READY", R.color.state_ready, "pick a raw clip and Run Tier-1")
            else -> pill(R.id.t1state, "BLOCKED", R.color.state_blocked,
                "seg/rvm/movenet not on device, and no masked clip staged")
        }

        // ── ② Phase 1 ── mask is OPTIONAL: BackgroundInpaint derives a hole mask when it is absent.
        when {
            bg -> pill(R.id.p1state, "DONE", R.color.state_done, "background_reconstructed exists · re-run overwrites")
            masked -> pill(R.id.p1state, "READY", R.color.state_ready,
                if (mask) "masked + mask staged" else "masked staged; hole mask will be DERIVED (no mask.mp4)")
            else -> pill(R.id.p1state, "BLOCKED", R.color.state_blocked, "no masked_video.mp4 staged")
        }

        // ── ②ᵇ Phase 1b ──
        when {
            lm -> pill(R.id.p1bstate, "DONE", R.color.state_done, "light_map exists · re-run overwrites")
            bg -> pill(R.id.p1bstate, "READY", R.color.state_ready, "background staged")
            else -> pill(R.id.p1bstate, "QUEUED", R.color.state_queued, "needs Phase 1's background")
        }

        // ── ③ Phase 2 ── the one that actually bit: no character => nothing to composite.
        when {
            comp -> pill(R.id.p2state, "DONE", R.color.state_done, "$layers character(s) composited · re-run overwrites")
            layers == 0 -> pill(R.id.p2state, "BLOCKED", R.color.state_blocked,
                "no character staged - add synthetic_person_p1.mp4 (+ its alpha)")
            !bg -> pill(R.id.p2state, "QUEUED", R.color.state_queued, "needs Phase 1's background")
            else -> {
                val noAlpha = runCatching { MiragePaths.personLayers().count { it.alpha == null } }.getOrDefault(0)
                pill(R.id.p2state, "READY", R.color.state_ready,
                    if (noAlpha == 0) "$layers character(s), all with an explicit alpha"
                    else "$layers character(s) · $noAlpha without an alpha → luma/mosaic keyer")
            }
        }
    }

    private fun pill(id: Int, state: String, colorRes: Int, why: String) {
        findViewById<TextView>(id).apply {
            text = if (why.isEmpty()) state else "$state · $why"
            setTextColor(getColor(colorRes))
        }
    }

    // ═══════════════════════ LONG-JOB TICKER ═════════════════════════════════════════════════════

    /**
     * Start the once-a-second elapsed/ETA line beside the progress bar.
     *
     * Phase 1 is ~100 s at 300 frames; an indeterminate bar tells you
     * nothing about it. PhaseRunner emits no per-frame callback, so a DETERMINATE bar would have
     * required editing pipeline code - out of scope. Instead the estimate is derived from this
     * phase's OWN last completed run (seconds + the frame count it processed, persisted by
     * [publishPhaseStat]) rescaled by the frame count of the clip about to be processed.
     *
     * If the phase has never run, or the frame count cannot be read, there is NO estimate and the
     * line shows elapsed only. It never invents a number.
     */
    private fun startTicker(name: String) {
        runName = name
        runT0 = android.os.SystemClock.elapsedRealtime()
        etaTotalS = 0.0
        etaBasis = ""
        showPhase("running $name …")
        // Frame count needs a MediaMetadataRetriever, which must not run on the UI thread.
        Thread {
            val keys = etaKeys(name)
            val src = etaSource(name)
            val n = if (src != null && src.exists()) frameCountOf(src) else -1
            val p = getSharedPreferences("mirage", MODE_PRIVATE)
            var total = 0.0
            var ok = keys.isNotEmpty()
            for (k in keys) {
                val secs = p.getFloat("sec_$k", 0f)
                val frames = p.getInt("frames_$k", 0)
                if (secs <= 0f || frames <= 0) { ok = false; break }
                total += if (n > 0) secs.toDouble() * n / frames else secs.toDouble()
            }
            if (ok && total > 0) post {
                etaTotalS = total
                etaBasis = if (n > 0) "from the last run, rescaled to $n f" else "from the last run"
            }
        }.start()
        val r = object : Runnable {
            override fun run() {
                val el = (android.os.SystemClock.elapsedRealtime() - runT0) / 1000.0
                val sb = StringBuilder("running $runName · ").append(mmss(el)).append(" elapsed")
                if (etaTotalS > 0) {
                    val left = etaTotalS - el
                    if (left > 0) sb.append(" · ~").append(mmss(left)).append(" left (").append(etaBasis).append(")")
                    else sb.append(" · over the ").append(mmss(etaTotalS)).append(" estimate")
                }
                showPhase(sb.toString())
                ui.postDelayed(this, 1000)
            }
        }
        tick = r
        ui.post(r)
    }

    private fun stopTicker() {
        tick?.let { ui.removeCallbacks(it) }
        tick = null
        showPhase(null)
    }

    private fun mmss(s: Double): String {
        val t = s.toInt().coerceAtLeast(0)
        return "%d:%02d".format(t / 60, t % 60)
    }

    /** The persisted stat keys a run of [name] consumes, in order. Empty = no estimate possible. */
    private fun etaKeys(name: String): List<String> = when (name) {
        "Phase 1" -> listOf("1")
        "Lightmap" -> listOf("1b")
        "Phase 2" -> listOf("2")
        "Full pipeline" -> listOf("1", "1b", "2")
        else -> emptyList()
    }

    /** The clip whose frame count sets the scale for [name]'s estimate. */
    private fun etaSource(name: String): File? = when (name) {
        "Phase 1", "Full pipeline" -> MiragePaths.maskedVideo
        "Lightmap", "Phase 2" -> MiragePaths.backgroundReconstructed
        else -> null
    }

    /** `PhaseEval.phase` ("1-Inpaint", "1b-Lightmap", …) → the short key used by the ETA store. */
    private fun statKey(phase: String): String? = when {
        phase.startsWith("1b") -> "1b"
        phase.startsWith("1-") -> "1"
        phase.startsWith("2-") -> "2"
        else -> null
    }

    /**
     * READINESS - one screenful that answers "can I actually run this, and with what?".
     *
     * The app had no way to see whether the 250 MB of models were staged, whether the inputs were
     * present, or which phase was runnable; you found out by pressing a button and reading an
     * exception in the terminal. This inspects the filesystem and says so up front.
     */
    private fun refreshReadiness() {
        val md = ModelStore.modelsDir(applicationContext)
        fun tick(ok: Boolean) = if (ok) "OK" else "--"
        fun mb(f: File) = if (f.exists()) "${f.length() / 1024 / 1024}MB" else "--"
        val lama = File(md, "lama.onnx")
        val seg = File(md, "seg.onnx"); val rvm = File(md, "rvm.onnx"); val mv = File(md, "movenet.onnx")

        val masked = MiragePaths.maskedVideo
        val mask = MiragePaths.maskVideo
        val layers = runCatching { MiragePaths.personLayers().size }.getOrDefault(0)
        val bg = MiragePaths.backgroundReconstructed
        val lm = MiragePaths.lightMap
        val comp = MiragePaths.compositeVideo
        val fin = MiragePaths.finalOutput

        val canP1 = masked.exists()
        val canP1b = bg.exists()
        val canP2 = bg.exists() && layers > 0
        val canTier1 = seg.exists() && rvm.exists() && mv.exists()

        val nl = System.lineSeparator()
        val sb = StringBuilder()
        sb.append("models   lama ").append(mb(lama))
            .append(" · seg ").append(tick(seg.exists()))
            .append(" · rvm ").append(tick(rvm.exists()))
            .append(" · movenet ").append(tick(mv.exists()))
            .append(" · seg_qnn ").append(tick(File(File(md, "seg_qnn"), "model.onnx").exists())).append(nl)
        sb.append("inputs   masked ").append(tick(masked.exists()))
            .append(" · mask ").append(tick(mask.exists()))
            .append(" · characters ").append(layers).append(nl)
        sb.append("outputs  bg ").append(tick(bg.exists()))
            .append(" · lightmap ").append(tick(lm.exists()))
            .append(" · composite ").append(tick(comp.exists()))
            .append(" · final ").append(tick(fin.exists())).append(nl)
        val runnable = buildList {
            if (canTier1) add("Tier-1")
            if (canP1) add("P1")
            if (canP1b) add("P1b")
            if (canP2) add("P2")
        }
        sb.append("runnable ").append(if (runnable.isEmpty()) "nothing - stage a masked clip" else runnable.joinToString(" · "))
        if (!canP2 && bg.exists() && layers == 0) sb.append(nl).append("P2 blocked: no synthetic_person_pK staged")
        if (!lama.exists()) sb.append(nl).append("no lama.onnx - Phase 1 core falls back to push-pull")

        findViewById<TextView>(R.id.readyText).text = sb.toString()

        // The header chip names the NEXT THING TO DO, not a count. "3 runnable" told you nothing
        // about which one to press; on a strictly sequential pipeline the useful answer is the first
        // stage whose output does not exist yet but whose inputs do.
        val chip = findViewById<TextView>(R.id.readyChip)
        val next = when {
            fin.exists() -> null
            canP1 && !bg.exists() -> "Phase 1"
            canP1b && !lm.exists() -> "Phase 1b"
            canP2 && !comp.exists() -> "Phase 2"
            canTier1 -> "Tier-1"
            else -> null
        }
        chip.text = when {
            next != null -> "next · $next"
            fin.exists() -> "final ready"
            runnable.isEmpty() -> "not ready"
            else -> "${runnable.size} runnable"
        }
        chip.setTextColor(getColor(when {
            next != null -> R.color.state_ready
            fin.exists() -> R.color.state_done
            else -> R.color.state_blocked
        }))
    }

    /** Show the tail of the append-only EVALS.md in the terminal - the history, without a file browser. */
    private fun showEvals() {
        val f = File(MiragePaths.outputDir, "EVALS.md")
        if (!f.exists()) { appendLog("no EVALS.md yet - run a phase first."); return }
        val lines = runCatching { f.readLines() }.getOrDefault(emptyList())
        appendLog("---- EVALS.md (${lines.size} lines, ${f.length() / 1024} KB) ----")
        lines.takeLast(40).forEach { appendLog(it) }
        appendLog("---- end (full history is in ${f.name} + EVALS_LOG.jsonl) ----")
    }

    /** The label beside the progress bar - tells the user WHICH phase is running, not just "busy". */
    private fun showPhase(text: String?) {
        val v = findViewById<TextView>(R.id.phaseLabel)
        if (text == null) { v.visibility = View.GONE } else { v.text = text; v.visibility = View.VISIBLE }
    }

    /**
     * Write a one-line result under the phase's own card: wall clock, ms/frame, the engine that ran it,
     * and the quality signals that phase reports. Previously the only place any of this appeared was the
     * scrolling terminal, where it was gone within a few lines.
     */
    private fun publishPhaseStat(e: PhaseRunner.PhaseEval) {
        val id = when {
            e.phase.startsWith("1-") -> R.id.p1stat
            e.phase.startsWith("1b") -> R.id.p1bstat
            e.phase.startsWith("2-") -> R.id.p2stat
            else -> return
        }
        val engine = e.quality["backend"] ?: "CPU x${Par.WORKERS}"
        val sb = StringBuilder("${"%.1f".format(e.ms / 1000.0)} s · ${"%.0f".format(e.msPerFrame)} ms/f · ${e.frames}f · $engine")
        val nl = "\n"
        e.quality["§2_leak_audit_%"]?.let {
            sb.append(nl).append("§2 leak: ").append(it)
                .append(" (").append(e.quality["§2_audited_frames"] ?: "?").append(" frames)")
        }
        e.quality["hole_from_REAL_%"]?.let { sb.append(" · real px: ").append(it).append("%") }
        e.quality["substage_ms"]?.takeIf { it.isNotBlank() }?.let { sb.append(nl).append(it) }
        findViewById<TextView>(id).apply { text = sb.toString(); setTextColor(Color.parseColor("#9BD6FF")) }
        // Persist wall clock + frame count so the NEXT run of this phase can show a remaining-time
        // estimate. Stored per phase, and only when both numbers are real - a zero would produce a
        // confident-looking "~0:00 left".
        statKey(e.phase)?.let { k ->
            if (e.ms > 0 && e.frames > 0) getSharedPreferences("mirage", MODE_PRIVATE).edit()
                .putFloat("sec_$k", (e.ms / 1000.0).toFloat()).putInt("frames_$k", e.frames).apply()
        }
    }

    /** Header chip: what silicon this build can actually reach, and what it is set to right now. */
    private fun updateEngineChip() {
        val qnn = NpuFactory.ORT_FLAVOR == "qnn"
        val on = npuToggle.isChecked
        val npu = if (!qnn) "NPU unavailable (non-QNN build)" else if (on) "NPU armed (Hexagon HTP)" else "NPU off"
        findViewById<TextView>(R.id.engineChip).text =
            "engine: $npu · CPU ${Par.WORKERS} workers · ORT ${NpuFactory.ORT_FLAVOR}"
    }

    /** Run the GUIDED Tier-1 pipeline on a RAW clip, entirely ON-DEVICE, and write masked_video + mask
     *  + the three Tier-1 card sidecars (pose sticks · canonicaliser · synthetic face). MASK = the
     *  guided pipeline strictly (YOLO11n-seg ∪ RVM matte → guided filter → dilate → FaceGuard); POSE =
     *  MoveNet. Capped to 100 frames @ 512px; off the UI thread. See Tier1Preview. */
    private fun runTier1OnPhone(uri: Uri) {
        if (!hasAllFilesAccess()) { appendLog("need All-Files access to write outputs - tap 'Grant files'."); return }
        appendLog("---- Tier-1 (guided pipeline) on a raw clip, on-device ----"); setBusy(true)
        // Tier-1 has no persisted per-frame cost (it is not a PhaseEval), so the ticker shows elapsed
        // only - which is still the difference between "working" and "hung" on a 1-2 minute job.
        startTicker("Tier-1")
        val maxFrames = 100; val workDim = 512
        // Read the toggle on the UI thread; Tier-1 previously ignored it entirely and always ran CPU.
        val accel = npuToggle.isChecked
        Thread {
            val res = runCatching {
                val log: (String) -> Unit = { line -> post { appendLog(line) } }
                val seg = NpuFactory.createYoloSegOrNull(applicationContext, accel, log) ?: throw IllegalStateException("seg.onnx (YOLO11n-seg) not on device")
                val rvm = NpuFactory.createRvmOrNull(applicationContext, accel, log) ?: throw IllegalStateException("rvm.onnx not on device")
                val pose = NpuFactory.createMoveNetOrNull(applicationContext, accel, log) ?: throw IllegalStateException("movenet.onnx not on device")
                // Save the RAW clip as the true "before" for the Compare card (original vs final).
                MiragePaths.ensureDirs()
                runCatching { applicationContext.contentResolver.openInputStream(uri)?.use { i -> MiragePaths.originalInput.outputStream().use { o -> i.copyTo(o) } } }
                    .onFailure { post { appendLog("(couldn't save original clip: ${it.message})") } }
                val t0 = System.currentTimeMillis()
                val n = seg.use { s -> rvm.use { r -> pose.use { p ->
                    Tier1Preview.generate(applicationContext, uri, s, r, p, maxFrames, workDim, log) } } }
                intArrayOf(n, ((System.currentTimeMillis() - t0) / 1000).toInt())
            }
            post {
                setBusy(false); stopTicker()
                res.onSuccess { arr ->
                    appendLog("Tier-1 (guided) OK - masked_video + mask + pose/canonicalizer/synthetic-face written (${arr[0]} frames, ${arr[1]}s). Now Run Phase 1 or FULL PIPELINE.")
                    refreshInputSpinner(); refreshPreviews()
                }.onFailure { appendLog("Tier-1 ERROR: ${it.message}") }
                refreshStatus()
            }
        }.start()
    }

    /** One tile of a phase's collage: a video file + its label; out=true colours the bar green (an output). */
    private data class CTile(val file: File, val label: String, val out: Boolean)

    private fun refreshPreviews() {
        // Tier-1 (edge) card: what the glasses/Pi already produced - the masked video the phone consumes,
        // plus the anonymised-pose / canonicaliser / face-mesh illustration sidecars (tier1_viz.py). All
        // are "outputs" of the upstream edge stage; absent files render "not ready".
        bindCollage(t1coll, listOf(
            CTile(MiragePaths.maskedVideo, "IN · masked", false),
            CTile(MiragePaths.tier1PoseSticks, "pose sticks", true),
            CTile(MiragePaths.tier1Canonical, "canonicalizer", true),
            CTile(MiragePaths.tier1SyntheticFace, "synthetic face", true)))
        // Phase 1: masked -> reconstructed background.
        bindCollage(p1coll, listOf(
            CTile(MiragePaths.maskedVideo, "IN · masked", false),
            CTile(MiragePaths.backgroundReconstructed, "OUT · background", true)))
        // Phase 1b: reconstructed background -> low-frequency lightmap (illumination map).
        bindCollage(p1bcoll, listOf(
            CTile(MiragePaths.backgroundReconstructed, "IN · background", false),
            CTile(MiragePaths.lightMap, "OUT · lightmap", true)))
        // Phase 2: SHOW the character(s) being composited (the background is already visible in Phase 1's
        // output); the background + per-character alpha inputs are WRITTEN in the hint, not shown.
        val chars = MiragePaths.personLayers()
        val p2 = ArrayList<CTile>()
        if (chars.isEmpty()) p2.add(CTile(MiragePaths.syntheticPerson, "IN · character", false))
        else chars.forEach { p2.add(CTile(it.person, "IN · char p${it.index}", false)) }
        p2.add(CTile(MiragePaths.compositeVideo, "OUT · composite", true))
        bindCollage(p2coll, p2)
        p2hint.text = buildP2Hint(chars)
        // Compare card (bottom): the ORIGINAL raw clip vs the finished anonymized result. Falls back to
        // the masked video only when there is no saved original (i.e. the input was already masked).
        // Prefer the RAW original. It is the honest "before" for a before/after claim: the masked clip
        // is already a Tier-1 output, so "masked vs final" understates what the pipeline did.
        // The label always names what is actually on screen, so a fallback can never be mistaken for
        // the raw original.
        val hasOrig = MiragePaths.originalInput.exists() && MiragePaths.originalInput.length() > 0
        bindCollage(cmpcoll, listOf(
            CTile(if (hasOrig) MiragePaths.originalInput else MiragePaths.maskedVideo,
                  if (hasOrig) "IN · original (raw)" else "IN · masked (no raw original staged)", false),
            CTile(MiragePaths.finalOutput, "OUT · final", true)))
    }

    /** Phase 2 uses more inputs than it shows: name the shown characters, and WRITE the rest - the
     *  background (already seen as Phase 1's output) and the per-character alpha mattes. */
    private fun buildP2Hint(chars: List<MiragePaths.PersonLayer>): String {
        val names = if (chars.isEmpty()) "none staged" else chars.joinToString(", ") { "p${it.index}" }
        val bg = if (MiragePaths.backgroundReconstructed.exists()) "background_reconstructed ✓" else "background_reconstructed (run Phase 1)"
        val alphas = chars.filter { it.alpha != null }.joinToString(", ") { "p${it.index}" }
        return "shows characters: $names  →  out: composite\n" +
            "also uses (not shown): $bg" + (if (alphas.isNotEmpty()) " · alpha: $alphas" else " · alpha: none (luma-keyed)")
    }

    /** Build a collage of [tiles] (each a first-frame thumbnail under a label bar) and set it on [iv];
     *  tapping opens all of them playing together fullscreen. (SurfaceView VideoViews don't render inside a
     *  ScrollView, so the card shows a static frame collage.) */
    private fun bindCollage(iv: ImageView, tiles: List<CTile>) {
        iv.setOnClickListener { openCollagePlayer(tiles) }
        Thread {
            val bmps = tiles.map { firstFrame(it.file) }
            val col = makeCollage(tiles, bmps)
            post { iv.setImageBitmap(col) }
        }.start()
    }

    private fun firstFrame(f: File): Bitmap? {
        if (!f.exists()) return null
        return runCatching {
            val r = MediaMetadataRetriever()
            try { r.setDataSource(f.absolutePath); r.getFrameAtTime(0, MediaMetadataRetriever.OPTION_CLOSEST_SYNC) }
            finally { runCatching { r.release() } }   // release even if setDataSource/getFrameAtTime throws
        }.getOrNull()
    }

    /** Compose [tiles] side-by-side, each under a coloured label bar (blue=input, green=output); a missing
     *  frame renders a "not ready" placeholder so the collage keeps a stable shape before a run. */
    private fun makeCollage(tiles: List<CTile>, bmps: List<Bitmap?>): Bitmap {
        val fh = 340; val labelH = 54; val pad = 14; val gap = 14; val r = 20f
        fun scaled(bmp: Bitmap?): Bitmap? = bmp?.let {
            val w = (it.width.toFloat() / it.height.coerceAtLeast(1) * fh).toInt().coerceIn(130, 620)
            Bitmap.createScaledBitmap(it, w, fh, true)
        }
        val sc = bmps.map { scaled(it) }
        val widths = sc.map { it?.width ?: 240 }
        val tileH = labelH + fh
        val totalW = pad + widths.sum() + gap * (tiles.size - 1).coerceAtLeast(0) + pad
        val out = Bitmap.createBitmap(totalW, pad + tileH + pad, Bitmap.Config.ARGB_8888)
        val c = Canvas(out); c.drawColor(Color.parseColor("#0B0B10"))
        val txt = Paint(Paint.ANTI_ALIAS_FLAG).apply { color = Color.WHITE; textSize = 26f; typeface = Typeface.DEFAULT_BOLD }
        val resP = Paint(Paint.ANTI_ALIAS_FLAG).apply { color = Color.parseColor("#B7C0D6"); textSize = 20f }
        val ph = Paint(Paint.ANTI_ALIAS_FLAG).apply { color = Color.parseColor("#5C6474"); textSize = 24f; textAlign = Paint.Align.CENTER }
        val phBg = Paint().apply { color = Color.parseColor("#101018") }
        val border = Paint(Paint.ANTI_ALIAS_FLAG).apply { style = Paint.Style.STROKE; strokeWidth = 2f; color = Color.parseColor("#30364A") }
        var x = pad
        for (i in tiles.indices) {
            val tw = widths[i]; val s = sc[i]
            val l = x.toFloat(); val t = pad.toFloat(); val rt = (x + tw).toFloat(); val b = (pad + tileH).toFloat()
            val rect = RectF(l, t, rt, b)
            c.save(); c.clipPath(Path().apply { addRoundRect(rect, r, r, Path.Direction.CW) })
            val fy = t + labelH
            if (s != null) c.drawBitmap(s, l, fy, null)
            else { c.drawRect(l, fy, rt, b, phBg); c.drawText(" - not ready - ", l + tw / 2f, fy + fh / 2f, ph) }
            // label bar (its top corners come out rounded via the tile clip)
            c.drawRect(l, t, rt, t + labelH, Paint().apply { color = Color.parseColor(if (tiles[i].out) "#1E5631" else "#2A3550") })
            c.drawCircle(l + 18f, t + labelH / 2f, 6f, Paint(Paint.ANTI_ALIAS_FLAG).apply { color = Color.parseColor(if (tiles[i].out) "#7FE6A6" else "#9BE7FF") })
            c.drawText(tiles[i].label, l + 34f, t + 35f, txt)
            val res = bmps[i]?.let { "${it.width}×${it.height}" } ?: ""
            if (res.isNotEmpty() && tw > 300) c.drawText(res, rt - resP.measureText(res) - 14f, t + 34f, resP)
            c.restore()
            c.drawRoundRect(rect, r, r, border)
            x += tw + gap
        }
        return out
    }

    /** Fullscreen COLLAGE player: all existing [tiles] stacked vertically, each under its label bar, looping.
     *  Tap anywhere to close. Stacked (not side-by-side) so square clips stay large on a portrait phone. */
    private fun openCollagePlayer(tiles: List<CTile>) {
        val panes = tiles.filter { it.file.exists() }
        if (panes.isEmpty()) { appendLog("nothing to show yet - run this phase first"); return }
        val dlg = Dialog(this, android.R.style.Theme_Black_NoTitleBar_Fullscreen)
        val root = LinearLayout(this).apply { orientation = LinearLayout.VERTICAL; setBackgroundColor(Color.BLACK) }
        val mp = ViewGroup.LayoutParams.MATCH_PARENT; val wc = ViewGroup.LayoutParams.WRAP_CONTENT
        for (t in panes) {
            val lbl = TextView(this).apply {
                text = t.label; setTextColor(Color.WHITE); textSize = 14f
                setTypeface(typeface, Typeface.BOLD); setBackgroundColor(Color.parseColor(if (t.out) "#1E5631" else "#2A3550"))
                gravity = Gravity.CENTER; setPadding(0, 12, 0, 12)
            }
            root.addView(lbl, LinearLayout.LayoutParams(mp, wc))
            val holder = FrameLayout(this).apply { setBackgroundColor(Color.BLACK); setOnClickListener { dlg.dismiss() } }
            val vv = VideoView(this)
            holder.addView(vv, FrameLayout.LayoutParams(mp, mp).apply { gravity = Gravity.CENTER })
            vv.setOnPreparedListener { it.isLooping = true; vv.start() }
            vv.setOnErrorListener { _, w, e -> appendLog("player error $w/$e"); true }
            vv.setVideoURI(Uri.fromFile(t.file))
            root.addView(holder, LinearLayout.LayoutParams(mp, 0, 1f))
        }
        root.setOnClickListener { dlg.dismiss() }
        dlg.setContentView(root)
        dlg.show()
    }

    /**
     * Frame count of a local file, or -1 if it cannot be determined.
     *
     * Used only to decide whether a staged `original_input.mp4` could possibly correspond to a newly
     * staged masked clip. Equal counts is a WEAK test - two different clips of the same length pass - 
     * so it gates only whether the file is KEPT, never whether a claim is made about it. Uses
     * METADATA_KEY_VIDEO_FRAME_COUNT rather than decoding: this runs on the UI refresh path and a
     * full decode of a 900-frame 1264² clip there would stall the app.
     */
    private fun frameCountOf(f: File): Int = runCatching {
        val r = MediaMetadataRetriever()
        try {
            r.setDataSource(f.absolutePath)
            r.extractMetadata(MediaMetadataRetriever.METADATA_KEY_VIDEO_FRAME_COUNT)?.toIntOrNull() ?: -1
        } finally { runCatching { r.release() } }
    }.getOrDefault(-1)

    private fun setBusy(busy: Boolean) {
        progressBar.visibility = if (busy) View.VISIBLE else View.INVISIBLE
        runButtons.forEach { it.isEnabled = !busy }
    }

    private fun saveToOutputFolder(src: File): String? {
        val tree = outTreeUri ?: return null
        return runCatching {
            val dir = DocumentsContract.buildDocumentUriUsingTree(tree, DocumentsContract.getTreeDocumentId(tree))
            val dst = DocumentsContract.createDocument(contentResolver, dir, "video/mp4", src.name) ?: return null
            contentResolver.openOutputStream(dst)!!.use { o -> src.inputStream().use { it.copyTo(o) } }
            appendLog("saved ${src.name} to your output folder"); ""
        }.getOrNull()
    }

    private fun refreshStatus() {
        // keep the readiness panel AND the per-stage pills in step with staging/clearing/completion
        runCatching { refreshReadiness() }
        runCatching { refreshStages() }
        Thread {
            val s = runCatching {
                val input = if (MiragePaths.maskedVideo.exists()) "input ✓" else "input - "
                val fin = if (MiragePaths.finalOutput.exists()) "final ✓" else "final - "
                "v${BuildConfig.VERSION_NAME} · ort=${NpuFactory.ORT_FLAVOR} · files ${if (hasAllFilesAccess()) "✓" else "✗"}" +
                    " · $input · chars ${MiragePaths.personLayers().size} · $fin · out ${if (outTreeUri != null) "custom" else "app"}"
            }.getOrElse { "status: ${it.message}" }
            post { pathText.text = s }
        }.start()
    }

    private fun hasAllFilesAccess(): Boolean = Environment.isExternalStorageManager()
    private fun requestAllFilesAccess() {
        if (hasAllFilesAccess()) { appendLog("All-Files access already granted."); return }
        runCatching {
            startActivity(Intent(Settings.ACTION_MANAGE_APP_ALL_FILES_ACCESS_PERMISSION).apply { data = Uri.parse("package:$packageName") })
        }.onFailure { startActivity(Intent(Settings.ACTION_MANAGE_ALL_FILES_ACCESS_PERMISSION)) }
    }

    private fun post(action: () -> Unit) = runOnUiThread(action)
    private fun appendLog(line: String) {
        logText.append(line + "\n"); logScroll.post { logScroll.fullScroll(ScrollView.FOCUS_DOWN) }
    }

    override fun onDestroy() { stopTicker(); super.onDestroy() }

    companion object {
        /** Index of the pipeline page. The app ALWAYS opens here - see the class KDoc. */
        const val PAGE_TIER2 = 1
    }
}
