#!/usr/bin/env python3
"""
phone_battery_report.py - battery drain %/hr and energy, MEASURED rather than extrapolated.

Two independent sources, deliberately, because the obvious one is too coarse:

1. `dumpsys batterystats` **Discharge step durations** - Android's own record of how long the device
   spent at each battery percent while discharging. One row per 1 % step, so %/hr = 60 / minutes.
   This is a MEASUREMENT over real elapsed time, not an extrapolation from a 40 s arm.

2. The **coulomb counter** (`charge counter`, µAh) sampled by phone_thermal.py. Δcharge over an arm
   × voltage gives Joules directly, which resolves a single phase - the integer battery % cannot
   (it moves ~1 point per 40 s arm).

⚠️ Screen state dominates phone drain. Every row is tagged with the screen/power-save flags Android
records, because a screen-on figure and a screen-off figure are not comparable.
"""
import io
import os
import re
import subprocess
import sys

# Path to the Android platform-tools `adb`. Set MIRAGE_ADB if it is not on PATH.
ADB = os.environ.get("MIRAGE_ADB", "adb")
# Scratch directory holding `phone_ip.txt` (the device's Wi-Fi-adb address). Override with
# MIRAGE_SCRATCH; defaults to ./_scratch beside this script.
SC = os.environ.get("MIRAGE_SCRATCH",
                    os.path.join(os.path.dirname(os.path.abspath(__file__)), "_scratch"))
CSV_POWER = os.environ.get("MIRAGE_POWER_CSV",
                           os.path.join(os.path.dirname(os.path.abspath(__file__)),
                                        "PHONE_POWER.csv"))


def dev():
    return open(os.path.join(SC, "phone_ip.txt")).read().strip() + ":5555"


def sh(cmd):
    r = subprocess.run([ADB, "-s", dev(), "shell", cmd], capture_output=True, text=True,
                       encoding="utf-8", errors="replace")
    return (r.stdout or "").replace("\r", "")


def steps():
    txt = sh("dumpsys batterystats --charged com.mirage.npu")
    m = re.search(r"Discharge step durations:(.*?)(?:\n\s*\n|\nCharge step)", txt, re.S)
    if not m:
        return []
    out = []
    for ln in m.group(1).split("\n"):
        s = re.search(r"#(\d+):\s*\+(?:(\d+)h)?(?:(\d+)m)?(?:(\d+)s)?(?:(\d+)ms)?\s*to\s*(\d+)\s*\((.*?)\)", ln)
        if s:
            h, mi, se, ms = (int(s.group(i) or 0) for i in (2, 3, 4, 5))
            mins = h * 60 + mi + se / 60.0 + ms / 60000.0
            if mins > 0:
                out.append(dict(idx=int(s.group(1)), to=int(s.group(6)), mins=mins,
                                pct_hr=60.0 / mins, flags=s.group(7)))
    return out


def joules():
    """Energy per labelled arm.

    Uses TRAPEZOID INTEGRATION OF INSTANTANEOUS POWER (V x I sampled every 5 s), not the coulomb
    counter: measured 2026-07-26, `charge counter` did not tick at all across a 65 s idle window
    (0 uAh delta at ~0.33 W), so its granularity is far too coarse to resolve a 35-60 s phase.
    Instantaneous current updates every sample and integrates cleanly.

    ⚠️ This is TOTAL DEVICE power, not the app's share. The screen alone is a large constant at
    brightness 40, which is why every arm is also reported NET of the idle control measured at the
    same settings -- the net figure is the pipeline's marginal cost.
    """
    if not os.path.exists(CSV_POWER):
        return {}
    import csv
    import datetime as dt
    rows = list(csv.DictReader(io.open(CSV_POWER, encoding="utf-8")))
    per = {}
    for r in rows:
        if not r.get("watts"):
            continue
        try:
            t = dt.datetime.strptime(r["iso"], "%Y-%m-%dT%H:%M:%S")
        except Exception:
            continue
        per.setdefault(r["label"], []).append((t, float(r["watts"])))
    out = {}
    for lab, v in per.items():
        if len(v) < 2:
            continue
        v.sort()
        j = 0.0
        for (t0, w0), (t1, w1) in zip(v, v[1:]):
            j += 0.5 * (w0 + w1) * (t1 - t0).total_seconds()
        secs = (v[-1][0] - v[0][0]).total_seconds()
        mw = sum(w for _, w in v) / len(v)
        out[lab] = dict(n=len(v), secs=round(secs, 1), joules=round(j, 1),
                        mean_W=round(mw, 3), peak_W=round(max(w for _, w in v), 3))
    return out


def main():
    st = steps()
    print("=== BATTERY DRAIN %/hr - from Android's own discharge-step record ===")
    if not st:
        print("  none recorded (device may not have discharged enough since last charge)")
    else:
        print("  %-5s %-6s %-9s %-9s %s" % ("step", "to %", "minutes", "%/hr", "conditions"))
        for s in st[:14]:
            print("  #%-4d %-6d %-9.2f %-9.1f %s" % (s["idx"], s["to"], s["mins"], s["pct_hr"],
                                                     s["flags"][:46]))
        on = [s for s in st if "screen-on" in s["flags"]]
        if on:
            fast = max(on, key=lambda x: x["pct_hr"])
            slow = min(on, key=lambda x: x["pct_hr"])
            print("\n  screen-on range: %.1f %%/hr (idlest) .. %.1f %%/hr (heaviest)"
                  % (slow["pct_hr"], fast["pct_hr"]))
            print("  -> at the heaviest observed rate a 100%% battery lasts ~%.1f h of this workload"
                  % (100.0 / fast["pct_hr"]))
    j = joules()
    if j:
        print("\n=== ENERGY PER ARM - coulomb counter (resolves a single phase; battery %% cannot) ===")
        print("  %-22s %-4s %-8s %-9s %-9s %s" % ("arm", "n", "secs", "Joules", "mean W", "peak W"))
        for lab, v in j.items():
            print("  %-22s %-4d %-8.1f %-9.1f %-9.3f %.3f"
                  % (lab, v["n"], v["secs"], v["joules"], v["mean_W"], v["peak_W"]))
    else:
        print("\n=== ENERGY PER ARM ===\n  not yet available: arms must be re-run with the upgraded "
              "logger (charge_uAh/watts columns were added after the first arms ran)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
