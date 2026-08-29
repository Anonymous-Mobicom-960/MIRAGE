// App module - MIRAGE Tier-2 NPU app.
//
// Target device: Samsung Galaxy S25 Ultra -> Snapdragon 8 Elite (SM8750), Hexagon HTP NPU, arm64.
//
// ONE DEFINITIVE APK ("install once"): the DEFAULT build always bundles ONNX Runtime
// (com.microsoft.onnxruntime:onnxruntime-android - CPU + NNAPI + XNNPACK execution providers).
// Better models / tuned params are picked up AT RUNTIME from the app's external files dir
// (adb push, no reinstall - see ModelStore.kt / RuntimeConfig.kt / INSTALL.md).
//
// TWO BUILD MODES (both use the same unified source; only the ORT artifact differs):
//   * DEFAULT:  ./gradlew :app:assembleDebug
//      - onnxruntime-android (CPU + NNAPI EPs, no Qualcomm .so). The EP ladder in OrtRunner tries
//         QNN (absent here -> skipped), then NNAPI (if the accel toggle is on), then CPU.
//   * QNN (optional): ./gradlew :app:assembleDebug -Pmirage.enableOrt=true
//      - swaps the dependency to onnxruntime-android-qnn (bundles libQnnHtp.so, ~+100 MB), so the
//         same EP ladder can actually open the Hexagon HTP for an AI-Hub-compiled QNN-context model.
//         The model STILL loads from the same runtime dir/asset - nothing else changes.
plugins {
    id("com.android.application")
    id("org.jetbrains.kotlin.android")
}

// Read -Pmirage.enableOrt=true|false (default false). true = use the QNN-flavoured ORT artifact.
val enableOrt: Boolean = (project.findProperty("mirage.enableOrt") as String?)?.toBoolean() ?: false

android {
    namespace = "com.mirage.npu"
    compileSdk = 35

    defaultConfig {
        applicationId = "com.mirage.npu"
        minSdk = 31          // S25 Ultra ships API 35; 31 is the floor QNN EP + this code assume.
        targetSdk = 35
        // 0.14 = the Tier-A latency pass (align scratch reuse + no redundant getPixels, parallel
        // gray/valid extraction, parallel sub-pixel probes, coverage-restricted warp, parallel
        // ringGain, Phase-2 matte scratch reuse). EXPLICITLY NOT PIXEL-AFFECTING - every change is
        // argued bit-identical in its own KDoc, and the A/B asserts the Phase-1 quality invariants
        // byte-identical against 0.13. The bump exists so an EVALS row can never be ambiguous about
        // WHICH build produced a latency number, which is the whole reason 0.13 was bumped too.
        //
        // 0.15 = 0.14 plus the sequential DECODE PREFETCH (FrameSource.startPrefetch), wired into
        // Phase 1's two passes, Phase 1b's three sources and Phase 2's three. Also not pixel-affecting:
        // the same frames arrive for the same indices in the same order, just earlier. Separate version
        // from 0.14 so the two latency effects are attributable independently.
        //
        // 0.15b = 0.15 with a TEARDOWN-SAFETY fix in the prefetcher (per-generation stop token, and
        // the worker releases its own MediaMetadataRetriever). 0.15's shared stop flag could, if a
        // join ever timed out, be un-set by the next startPrefetch and resurrect an orphaned worker
        // onto a retriever stopPrefetch had already released. Shutdown path only - it cannot touch
        // steady-state latency, and a confirmation run is recorded rather than assumed.
        //
        // 0.17 = the BACKGROUND FILL SHARPNESS slider (Phase 1 card). It drives the FIX12 knobs
        // directly - LOCAL_K (temporal samples mixed per hole pixel; fewer = a smaller mixing radius
        // = sharper) and LOCAL_RADIUS (how far in time the nearest-first walk may look for them) - 
        // and the same nearest-first override now also runs on the STATIC/JITTER path, which mixed
        // donors spread over the WHOLE clip and therefore carried the same σ≈0.5 px alignment blur
        // the mosaic did. PIXEL-AFFECTING, and the default position moved to 100 (K=1) on the
        // owner's instruction, so a 0.17 output must never be compared with a 0.16 one without
        // stating the slider position - which EVALS now records on every Phase-1 row.
        //
        // 0.17 also carries the SEAM MATCH slider: the fill's exposure/colour is fitted to the 24 px
        // band of REAL background around the hole instead of to the whole frame, and a smooth
        // push-pull correction membrane seeded on that band is added across the fill. Default 100.
        // Also pixel-affecting; also recorded on every Phase-1 EVALS row. Both sliders read 0 to get
        // the pre-2026-08-17 behaviour back exactly.
        versionCode = 23
        versionName = if (enableOrt) "0.17-qnn" else "0.17"

        // Snapdragon 8 Elite is arm64 only. Restricting ABIs keeps the ORT native payload small.
        ndk { abiFilters += "arm64-v8a" }
    }

    buildFeatures {
        buildConfig = true
    }

    // QNN/Hexagon NPU: extract native libs to the filesystem. The DSP's FastRPC daemon loads
    // libQnnHtpV79Skel.so as a real FILE (it can't read inside base.apk); AGP's default compressed
    // packaging leaves it trapped in the APK -> QnnDevice_create fails. Legacy packaging unzips them.
    packaging {
        jniLibs {
            useLegacyPackaging = true
        }
    }
    defaultConfig {
        buildConfigField("String", "ORT_FLAVOR", "\"${if (enableOrt) "qnn" else "nnapi-cpu"}\"")
    }

    buildTypes {
        release {
            isMinifyEnabled = false
            proguardFiles(getDefaultProguardFile("proguard-android-optimize.txt"), "proguard-rules.pro")
        }
        debug {
            isMinifyEnabled = false
        }
    }

    compileOptions {
        sourceCompatibility = JavaVersion.VERSION_17
        targetCompatibility = JavaVersion.VERSION_17
    }
    kotlinOptions {
        jvmTarget = "17"
    }

    // Do NOT let AAPT recompress model files shipped in assets (ORT reads them as-is).
    androidResources {
        noCompress += listOf("onnx", "bin")
    }

    packaging {
        jniLibs { useLegacyPackaging = false }
        resources {
            excludes += setOf("META-INF/DEPENDENCIES", "META-INF/LICENSE*", "META-INF/NOTICE*")
        }
    }
}

dependencies {
    implementation("androidx.core:core-ktx:1.13.1")
    implementation("androidx.appcompat:appcompat:1.7.0")
    implementation("com.google.android.material:material:1.12.0")
    implementation("androidx.activity:activity-ktx:1.9.3")

    // ==== ONNX Runtime - ALWAYS present (this is what runs the ONNX models) ====
    // Default: the plain Android package (CPU + NNAPI + XNNPACK EPs; no Qualcomm libraries).
    // -Pmirage.enableOrt=true: the QNN-flavoured package instead (adds the Hexagon HTP backend .so's).
    // 🔴 These are NOT interchangeable. OrtRunner uses registerExecutionProviderLibrary and
    // epDevices, which exist in 1.24.x but not in 1.22.0, so the default (false) branch does
    // NOT compile. -Pmirage.enableOrt=true is required to build, and is also the only flavour
    // that reaches the Hexagon NPU. See tier2_phone/INSTALL.md.
    if (enableOrt) {
        // QNN via QUALCOMM'S PLUGIN EP (not MS's -qnn AAR, which pins QAIRT 2.42 < our AI-Hub context's
        // 2.47 and can never load it). Plugin EP 2.4.0 = QAIRT 2.48.40 (>= 2.47 => loads the context);
        // qnn-runtime ships the V79 HTP skel libs. ORT base 1.24.x is the validated pairing.
        implementation("com.microsoft.onnxruntime:onnxruntime-android:1.24.3")
        implementation("com.qualcomm.qti:onnxruntime-android-qnn:2.4.0")
        implementation("com.qualcomm.qti:qnn-runtime:2.48.0")
    } else {
        implementation("com.microsoft.onnxruntime:onnxruntime-android:1.22.0")       // verified resolves
    }
}
