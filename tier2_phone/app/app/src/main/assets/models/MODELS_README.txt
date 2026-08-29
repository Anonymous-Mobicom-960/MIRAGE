assets/models/ - models bundled INTO the APK at build time.

Model files here are gitignored; the APK still builds and runs without them. Any model the
app uses (lama.onnx inpainter; the on-phone Tier-1 models seg.onnx / rvm.onnx / movenet.onnx)
can be bundled here, but none is required at build time.

NOTE - you normally do NOT rebuild to change models. The installed app loads models at
runtime from /sdcard/Android/data/com.mirage.npu/files/models/ (adb push, no reinstall),
which takes precedence over anything bundled here. See INSTALL.md "Upgrade the model later".

An AI-Hub-compiled QNN context model (e.g. migan_qnn_ctx.onnx, for the -Pmirage.enableOrt=true
build) also goes through the SAME runtime dir - no need to bake it into assets.
