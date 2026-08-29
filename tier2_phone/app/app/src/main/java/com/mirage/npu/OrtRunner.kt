package com.mirage.npu

import ai.onnxruntime.OnnxTensor
import ai.onnxruntime.OrtEnvironment
import ai.onnxruntime.OrtSession
import ai.onnxruntime.TensorInfo
import android.content.Context
import android.util.Log
import java.io.File
import java.nio.FloatBuffer

/**
 * OrtRunner - the single ONNX Runtime wrapper used by BOTH build flavours. It opens a session by
 * walking an execution-provider ladder, so ONE code path serves every device/build:
 *
 *   1. QNN / Hexagon HTP - only succeeds in the -Pmirage.enableOrt=true build (which bundles the
 *                           Qualcomm backend .so's) on a Snapdragon with an HTP. Skipped otherwise.
 *   2. NNAPI             - present in the default onnxruntime-android build; delegates supported
 *                           subgraphs to the device's NPU/GPU/DSP drivers, rest runs on CPU.
 *   3. CPU EP            - always available. Guaranteed baseline.
 *
 * Steps 1-2 are attempted only when the user asks for acceleration (wantAccel); every attempt is
 * exception-guarded, so an EP that is missing from the binary or unsupported on the device just
 * drops through to the next rung - the app NEVER crashes over an EP.
 *
 * The same ai.onnxruntime Java API is shipped by both the plain and the -qnn Maven artifacts, so
 * calls like addQnn()/addNnapi() always compile; at runtime they throw (and are caught here) when
 * that EP's native code is not in the APK.
 */
class OrtRunner private constructor(
    private val env: OrtEnvironment,
    private val session: OrtSession,
    /** Which EP the session actually opened on: "QNN-HTP", "NNAPI" or "CPU". */
    val backend: String,
) : AutoCloseable {

    val inputNames: Set<String> get() = session.inputNames
    val outputNames: Set<String> get() = session.outputNames

    /** Ordered input tensor infos (name -> shape with -1 for dynamic dims). */
    fun inputInfos(): List<Pair<String, LongArray>> =
        session.inputInfo.map { (name, node) ->
            val shape = (node.info as? TensorInfo)?.shape ?: LongArray(0)
            name to shape
        }

    fun floatTensor(data: FloatArray, shape: LongArray): OnnxTensor =
        OnnxTensor.createTensor(env, FloatBuffer.wrap(data), shape)

    fun intTensor(data: IntArray, shape: LongArray): OnnxTensor =
        OnnxTensor.createTensor(env, java.nio.IntBuffer.wrap(data), shape)

    fun run(inputs: Map<String, OnnxTensor>): OrtSession.Result = session.run(inputs)

    /** Run and return the first output flattened to FloatArray + its shape. */
    fun runFloats(inputs: Map<String, OnnxTensor>): Pair<FloatArray, LongArray> {
        run(inputs).use { result ->
            val out = result[0] as OnnxTensor
            val shape = out.info.shape
            val buf = out.floatBuffer ?: error("ORT output was not float32 - check the model dtype")
            val arr = FloatArray(buf.remaining())
            buf.get(arr)
            return arr to shape
        }
    }

    override fun close() {
        runCatching { session.close() }
        // env is the process-wide singleton; do not close it here.
    }

    companion object {
        private const val TAG = "OrtRunner"

        fun fromAsset(context: Context, assetPath: String, wantAccel: Boolean, log: (String) -> Unit = {}): OrtRunner =
            fromBytes(context.assets.open(assetPath).use { it.readBytes() }, wantAccel, log)

        fun fromFile(path: File, wantAccel: Boolean, log: (String) -> Unit = {}): OrtRunner =
            fromBytes(path.readBytes(), wantAccel, log)

        /**
         * Open a session from a FILE PATH (not bytes) - REQUIRED for a precompiled QNN EPContext model:
         * the tiny model.onnx references its model.bin by RELATIVE path, so ORT must load from disk to
         * resolve it. Walks QNN(HTP) -> CPU. Use this for any precompiled EPContext bundle (model.onnx+model.bin).
         */
        /** Load a precompiled QNN (AI-Hub) EPContext model on the Hexagon NPU via QUALCOMM'S PLUGIN EP.
         *  The MS -qnn AAR can't (its bundled QAIRT is older than the context's); the plugin EP + a
         *  matching qnn-runtime can. soc_model 69 = SM8750 / Snapdragon 8 Elite (do NOT pass htp_arch,
         *  ORT rejects "79"). ADSP_LIBRARY_PATH must point at the APK's own skel libs. */
        fun fromModelPath(context: Context, path: String, wantAccel: Boolean, log: (String) -> Unit = {}): OrtRunner {
            // The DSP (FastRPC) resolves the HTP skel via ADSP_LIBRARY_PATH: put the APK's own skel FIRST
            // (so the right QAIRT version wins) but KEEP the system DSP paths (dropping them => the skel's
            // own deps can't load => QNN_DEVICE_ERROR_INVALID_CONFIG on device create).
            // The DSP resolves libQnnHtpV79Skel.so via ADSP_LIBRARY_PATH. On the S25 the plugin does NOT
            // auto-point it at our lib dir (it searches ., /vendor/dsp/... instead), so we set it to
            // nativeLibraryDir ONLY (where extractNativeLibs=true put the extracted skel). No /vendor paths.
            val nld = context.applicationInfo.nativeLibraryDir
            runCatching { android.system.Os.setenv("ADSP_LIBRARY_PATH", nld, true) }
            Log.i(TAG, "ADSP_LIBRARY_PATH=$nld")
            val env = OrtEnvironment.getEnvironment()
            if (wantAccel) {
                runCatching { env.registerExecutionProviderLibrary("QNNExecutionProvider", "libonnxruntime_providers_qnn.so") }
                val qnnDevices = runCatching { env.epDevices.filter { it.epName == "QNNExecutionProvider" } }.getOrDefault(emptyList())
                Log.i(TAG, "epDevices: ${runCatching { env.epDevices.map { it.epName } }.getOrNull()}")
                if (qnnDevices.isNotEmpty()) {
                    // try progressively simpler option sets - one of these should let QnnDevice_create succeed
                    val optionSets = listOf(
                        mapOf("backend_type" to "htp", "htp_performance_mode" to "burst", "soc_model" to "69", "device_id" to "0"),
                        mapOf("backend_type" to "htp", "soc_model" to "69"),
                        mapOf("backend_type" to "htp"),
                    )
                    for (o in optionSets) {
                        val r = runCatching {
                            val opts = OrtSession.SessionOptions(); opts.addExecutionProvider(qnnDevices, o)
                            OrtRunner(env, env.createSession(path, opts), backend = "QNN-HTP")
                        }
                        if (r.isSuccess) {
                            Log.i(TAG, "session on QNN/HTP (plugin EP) opts=${o.keys}"); log("[ort] session on QNN-HTP (NPU, plugin EP)")
                            return r.getOrThrow()
                        }
                        Log.i(TAG, "QNN opts ${o.keys} failed: ${r.exceptionOrNull()?.message?.take(160)}")
                    }
                }
                log("[ort] QNN plugin EP failed (device create); CPU")
            }
            val s = env.createSession(path, OrtSession.SessionOptions())
            log("[ort] session on CPU EP (path)")
            return OrtRunner(env, s, backend = "CPU")
        }

        /**
         * COMPILE a plain .onnx into a precompiled QNN EPContext for THIS device's Hexagon, on device.
         *
         * WHY ON-DEVICE rather than Qualcomm AI Hub. AI Hub compiles against whatever QAIRT version the
         * service used, and the runtime here must be >= that or the context simply will not load - 
         * exactly the skew that once forced a pinned `--qairt_version` and cost
         * the five-part fix documented in TECHNIQUES.md §6. Compiling with the same runtime that will
         * load it makes version skew structurally impossible, and needs no account, token or network.
         *
         * Writes `<outDir>/model.onnx` (a small EPContext wrapper) plus its `model.bin` sibling. The
         * wrapper references the .bin by RELATIVE name, so both must stay together and be loaded BY PATH
         * - [fromModelPath] does that.
         *
         * `session.disable_cpu_ep_fallback=1` is the point of the whole exercise: it makes session
         * creation THROW if even one node cannot be placed on the HTP, turning "is it really all on the
         * NPU?" from a hope into a build-time assertion. A model that quietly runs half on CPU is worse
         * than not migrating - it pays the partition cost and delivers nothing.
         *
         * @return true if a context was produced.
         */
        fun compileQnnContext(context: Context, modelPath: String, outDir: File, log: (String) -> Unit): Boolean {
            val nld = context.applicationInfo.nativeLibraryDir
            runCatching { android.system.Os.setenv("ADSP_LIBRARY_PATH", nld, true) }
            val env = OrtEnvironment.getEnvironment()
            runCatching { env.registerExecutionProviderLibrary("QNNExecutionProvider", "libonnxruntime_providers_qnn.so") }
            val devices = runCatching { env.epDevices.filter { it.epName == "QNNExecutionProvider" } }.getOrDefault(emptyList())
            if (devices.isEmpty()) { log("[qnn-compile] no QNN device - is this the -Pmirage.enableOrt=true build?"); return false }
            outDir.mkdirs()
            val ctxPath = File(outDir, "model.onnx").absolutePath
            val optionSets = listOf(
                mapOf("backend_type" to "htp", "htp_performance_mode" to "burst", "soc_model" to "69",
                      "enable_htp_fp16_precision" to "1", "htp_graph_finalization_optimization_mode" to "3"),
                mapOf("backend_type" to "htp", "soc_model" to "69", "enable_htp_fp16_precision" to "1"),
                mapOf("backend_type" to "htp"),
            )
            for ((i, o) in optionSets.withIndex()) {
                val r = runCatching {
                    val opts = OrtSession.SessionOptions()
                    opts.addExecutionProvider(devices, o)
                    opts.addConfigEntry("ep.context_enable", "1")
                    opts.addConfigEntry("ep.context_file_path", ctxPath)
                    opts.addConfigEntry("ep.context_embed_mode", "0")   // separate .bin sibling file
                    opts.addConfigEntry("session.disable_cpu_ep_fallback", "1")
                    env.createSession(modelPath, opts).close()
                }
                if (r.isSuccess) {
                    val ok = File(ctxPath).exists()
                    log(if (ok) "[qnn-compile] OK -> ${outDir.name}/model.onnx (100% on HTP; CPU fallback was disabled)"
                        else "[qnn-compile] session built but no context file appeared")
                    return ok
                }
                val msg = r.exceptionOrNull()?.message?.take(160)
                log("[qnn-compile] option set ${i + 1}/${optionSets.size} failed: $msg")
                // A disable_cpu_ep_fallback failure is the INFORMATIVE one: it means some node cannot
                // be placed on the HTP, i.e. this model is not fully NPU-eligible as exported.
            }
            log("[qnn-compile] all option sets failed - model not fully placeable on the HTP")
            return false
        }

        /**
         * Open a session for [modelBytes], walking QNN -> NNAPI -> CPU. Throws only if even the CPU
         * EP cannot load the model (i.e. the bytes are not a valid/supported ONNX).
         */
        fun fromBytes(modelBytes: ByteArray, wantAccel: Boolean, log: (String) -> Unit = {}): OrtRunner {
            val env = OrtEnvironment.getEnvironment()

            if (wantAccel) {
                // ---- Rung 1: QNN / Hexagon HTP (only real in the -Pmirage.enableOrt=true build) ----
                runCatching {
                    val opts = OrtSession.SessionOptions()
                    val qnn = HashMap<String, String>().apply {
                        put("backend_path", "libQnnHtp.so")        // Hexagon HTP backend (the NPU)
                        put("htp_performance_mode", "burst")
                        put("htp_graph_finalization_optimization_mode", "3")
                        put("enable_htp_fp16_precision", "1")
                    }
                    opts.addQnn(qnn)
                    val session = env.createSession(modelBytes, opts)
                    OrtRunner(env, session, backend = "QNN-HTP")
                }.onSuccess {
                    Log.i(TAG, "session opened on QNN/HTP"); log("[ort] session on QNN-HTP (NPU)")
                    return it
                }.onFailure {
                    Log.i(TAG, "QNN unavailable (${it.message}); trying NNAPI")
                }

                // ---- Rung 2: NNAPI (in the default onnxruntime-android build) ----
                runCatching {
                    val opts = OrtSession.SessionOptions()
                    opts.addNnapi()
                    val session = env.createSession(modelBytes, opts)
                    OrtRunner(env, session, backend = "NNAPI")
                }.onSuccess {
                    Log.i(TAG, "session opened on NNAPI"); log("[ort] session on NNAPI (partial delegation; rest on CPU)")
                    return it
                }.onFailure {
                    Log.i(TAG, "NNAPI unavailable (${it.message}); falling back to CPU EP")
                    log("[ort] NNAPI not usable here (${it.message?.take(80)}); using CPU EP")
                }
            }

            // ---- Rung 3: CPU EP (always compiled in) ----
            val session = env.createSession(modelBytes, OrtSession.SessionOptions())
            Log.i(TAG, "session opened on CPU EP (${modelBytes.size} bytes)")
            log("[ort] session on CPU EP")
            return OrtRunner(env, session, backend = "CPU")
        }
    }
}
