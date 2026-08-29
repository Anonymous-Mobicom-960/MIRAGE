# SCAFFOLD proguard rules. Minify is OFF by default (see build.gradle.kts), but if you enable it,
# keep the ONNX Runtime classes - the QNN EP resolves native bindings by name/reflection.
-keep class ai.onnxruntime.** { *; }
-keep class com.mirage.npu.** { *; }
-dontwarn ai.onnxruntime.**
