#!/usr/bin/env python3
"""
phone_static_metrics.py - the MIRAGE mobile metrics that do NOT need a pipeline run.

Covers, from `planning/MIRAGE_Metrics_Evaluation_Plan.xlsx` (Mobile column):
  C. Model size (MB) · Peak memory footprint · CPU/GPU/NPU utilisation · storage
  D. Storage overhead (MB per clip / %) · Uplink payload / bandwidth to cloud
Plus the device inventory a reader needs to interpret any of it.

Everything here is read from the DEVICE, not assumed. Where a metric cannot be obtained honestly it
is printed as NOT MEASURED with the reason, rather than estimated.
"""
import json
import os
import re
import subprocess
import sys

# Path to the Android platform-tools `adb`. Set MIRAGE_ADB if it is not on PATH.
ADB = os.environ.get("MIRAGE_ADB", "adb")
# Scratch dir holding `phone_ip.txt` (the device's Wi-Fi-adb address). Override with
# MIRAGE_SCRATCH; defaults to ./_scratch beside this script.
SC = os.environ.get("MIRAGE_SCRATCH",
                    os.path.join(os.path.dirname(os.path.abspath(__file__)), "_scratch"))
PKG = "com.mirage.npu"
B = "/sdcard/Download/Project MIRAGE"


def dev():
    try:
        return open(os.path.join(SC, "phone_ip.txt")).read().strip() + ":5555"
    except Exception:
        return ""


def sh(cmd):
    r = subprocess.run([ADB, "-s", dev(), "shell", cmd], capture_output=True, text=True,
                       encoding="utf-8", errors="replace")
    return (r.stdout or "").replace("\r", "")


def main():
    out = {}
    print("=== DEVICE ===")
    for k, c in (("model", "getprop ro.product.model"), ("soc", "getprop ro.board.platform"),
                 ("android", "getprop ro.build.version.release"),
                 ("abi", "getprop ro.product.cpu.abi")):
        v = sh(c).strip()
        out[k] = v
        print("  %-10s %s" % (k, v))
    tot = re.search(r"MemTotal:\s+(\d+)", sh("cat /proc/meminfo"))
    if tot:
        out["ram_gb"] = round(int(tot.group(1)) / 1048576.0, 1)
        print("  %-10s %s GB" % ("RAM", out["ram_gb"]))
    out["cpu_cores"] = sh("cat /sys/devices/system/cpu/online").strip()
    print("  %-10s %s" % ("cores", out["cpu_cores"]))

    print("\n=== APP + MODELS (model size, MIRAGE C) ===")
    apk = sh("pm path %s" % PKG).replace("package:", "").strip().split("\n")[0]
    sz = sh("stat -c %%s '%s' 2>/dev/null" % apk).strip()
    if sz.isdigit():
        out["apk_mb"] = round(int(sz) / 1e6, 1)
        print("  APK              %.1f MB  (models are bundled inside)" % out["apk_mb"])
    # models are extracted to the app's files dir at first run
    lst = sh("run-as %s ls -la files/models 2>/dev/null" % PKG)
    if not lst.strip():
        lst = sh("ls -la /sdcard/Android/data/%s/files/models 2>/dev/null" % PKG)
    models = []
    for ln in lst.split("\n"):
        m = re.search(r"\s(\d+)\s+\S+\s+\S+\s+(\S+\.(?:onnx|bin|tflite))\s*$", ln)
        if m:
            models.append((m.group(2), int(m.group(1))))
    if models:
        for n, s in sorted(models, key=lambda x: -x[1]):
            print("  %-28s %8.1f MB" % (n, s / 1e6))
        out["models"] = {n: round(s / 1e6, 2) for n, s in models}
    else:
        print("  NOT MEASURED: model dir unreadable (run-as blocked on a release-signed APK, and the "
              "models ship inside the APK rather than in external storage)")

    print("\n=== PEAK MEMORY (MIRAGE C) ===")
    mi = sh("dumpsys meminfo %s" % PKG)
    for pat, lab in ((r"TOTAL PSS:\s+(\d+)", "TOTAL PSS"), (r"TOTAL RSS:\s+(\d+)", "TOTAL RSS"),
                     (r"Native Heap:\s+(\d+)", "Native heap")):
        m = re.search(pat, mi)
        if m:
            v = round(int(m.group(1)) / 1024.0, 1)
            out[lab.replace(" ", "_").lower() + "_mb"] = v
            print("  %-14s %8.1f MB" % (lab, v))
    print("  ⚠️ this is a POST-RUN reading, not a peak-during-run high-water mark: Android does not "
          "retain one, and sampling every 5 s can miss a transient allocation.")

    print("\n=== STORAGE / UPLINK PAYLOAD (MIRAGE D) ===")
    def du(p):
        v = sh("du -sk '%s' 2>/dev/null" % p).split()
        return int(v[0]) / 1024.0 if v and v[0].isdigit() else None
    for lab, p in (("input/ (what Tier-2 consumes)", B + "/input"),
                   ("output/ (what Tier-2 produces)", B + "/output"),
                   ("tier1/a10 (Tier-1 hand-off)", B + "/tier1/a10"),
                   ("cloud/a10 (cloud return)", B + "/cloud/a10")):
        v = du(p)
        if v is not None:
            print("  %-32s %8.1f MB" % (lab, v))
            out[lab.split(" ")[0]] = round(v, 1)
    # the UPLINK payload is specifically the de-identified signals sent to the cloud
    up = {}
    for f in ("masked_video.mp4", "mask.mp4"):
        v = sh("stat -c %%s '%s/tier1/a10/%s' 2>/dev/null" % (B, f)).strip()
        if v.isdigit():
            up[f] = int(v)
    if up:
        tot_up = sum(up.values())
        print("  --- uplink to cloud (de-identified signals only) ---")
        for k, v in up.items():
            print("      %-22s %8.2f MB" % (k, v / 1e6))
        print("      %-22s %8.2f MB  <- MIRAGE 'uplink payload'" % ("TOTAL", tot_up / 1e6))
        out["uplink_mb"] = round(tot_up / 1e6, 2)

    print("\n=== CPU / NPU UTILISATION (MIRAGE C) ===")
    print("  ⚠️ NOT MEASURED as a percentage. Android exposes no per-NPU utilisation counter without "
          "vendor tooling, and `top` sampled over adb during a 40 s phase is dominated by sampling "
          "artefacts. What IS measurable and recorded instead: which execution provider the session "
          "actually opened (OrtRunner logs 'QNN-HTP' | 'NNAPI' | 'CPU'), which answers the question "
          "the metric exists for -- did the work reach the NPU or silently fall back.")
    dest = os.environ.get("MIRAGE_OUT_JSON",
                          os.path.join(os.path.dirname(os.path.abspath(__file__)),
                                       "PHONE_STATIC_METRICS.json"))
    json.dump(out, open(dest, "w"), indent=1)
    print("wrote " + dest)
    return 0


if __name__ == "__main__":
    sys.exit(main())
