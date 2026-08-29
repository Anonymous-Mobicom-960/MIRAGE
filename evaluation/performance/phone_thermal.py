#!/usr/bin/env python3
"""
phone_thermal.py - read the S25's LIVE thermal + battery state over adb, correctly.

🔴 THE TRAP THIS EXISTS TO AVOID. `dumpsys thermalservice` prints TWO temperature blocks:

    Cached temperatures:            <- STALE. Never refreshed. Held AP=43.9 for over an hour.
    Current temperatures from HAL:  <- LIVE. The same instant read AP=31.7.

A naive `grep mName=AP` matches the CACHED block first and silently returns a number that can be
12 C wrong and never changes no matter what the device is doing. That is exactly what happened on
2026-07-26: a "device baseline at unplug" of AP 43.9 C was recorded, and an inference built on it
("the SoC is elevated by the USB staging") - both wrong. This parser reads ONLY the HAL block and
refuses to fall back to the cached one.

Throttling reference, read from the device itself rather than assumed:
  SKIN hot-throttling thresholds = [38, 40, 42, 45, 47, 60, 90] C - so mitigation begins at SKIN 38.

  python evaluation/performance/phone_thermal.py                  # one reading
  python evaluation/performance/phone_thermal.py --watch 5 --csv out.csv   # sample every 5 s
"""
import argparse
import io
import os
import re
import subprocess
import sys
import time

# Path to the Android platform-tools `adb`. Set MIRAGE_ADB if it is not on PATH.
ADB = os.environ.get("MIRAGE_ADB", "adb")
SERIAL = os.environ.get("MIRAGE_PHONE_SERIAL", "")


def sh(cmd):
    a = [ADB] + (["-s", SERIAL] if SERIAL else []) + ["shell", cmd]
    r = subprocess.run(a, capture_output=True, text=True, encoding="utf-8", errors="replace")
    return (r.stdout or "").replace("\r", "")


def thermal():
    """LIVE temperatures only - the 'Current temperatures from HAL' block."""
    txt = sh("dumpsys thermalservice")
    m = re.search(r"Current temperatures from HAL:(.*?)(?:\nCurrent cooling|\nTemperature static|\Z)",
                  txt, re.S)
    if not m:
        raise SystemExit("could not find the HAL block - refusing to fall back to cached values")
    out = {}
    for name, val in re.findall(r"mValue=([-\d.]+),\s*mType=\d+,\s*mName=(\w+)",
                                m.group(1).replace("mValue=", "mValue=")) or []:
        pass
    for mm in re.finditer(r"Temperature\{mValue=([-\d.]+),\s*mType=(\d+),\s*mName=(\w+)", m.group(1)):
        out[mm.group(3)] = float(mm.group(1))
    return out


def battery():
    """Battery state INCLUDING the fine-grained power counters.

    `/sys/class/power_supply/battery/*` is root-only on this device, but `dumpsys battery` surfaces
    the same two values, and they are what make real energy measurement possible:
      * `current now`    micro-amps, NEGATIVE while discharging -> instantaneous power = V x I
      * `charge counter` micro-amp-hours, a COULOMB COUNTER -> exact charge used over an arm,
                         which converts to Joules and sidesteps the coarse integer battery %.
    Battery % moves ~1 point per 40 s arm and cannot resolve a phase; these can.
    """
    txt = sh("dumpsys battery")
    g = lambda k: (re.search(r"^\s*%s:\s*(-?[\d.]+)\s*$" % k, txt, re.M) or [None, None])[1]
    g2 = lambda k: (re.search(r"^\s*%s:\s*(-?\d+)\s*$" % k, txt, re.M | re.I) or [None, None])[1]
    usb = re.search(r"USB powered:\s*(\w+)", txt)
    cur = g2("current now")
    chg = g2("charge counter")
    v_mv = int(g("voltage") or -1)
    watts = None
    if cur is not None and v_mv > 0:
        watts = abs(int(cur)) / 1e6 * (v_mv / 1000.0)      # A x V
    return {"level": int(g("level") or -1),
            "temperature_C": (float(g("temperature")) / 10.0) if g("temperature") else None,
            "voltage_mV": v_mv,
            "current_uA": int(cur) if cur is not None else None,
            "charge_uAh": int(chg) if chg is not None else None,
            "watts": round(watts, 3) if watts else None,
            "usb_powered": (usb.group(1) == "true") if usb else None}


def gpu_busy():
    """GPU utilisation %, from the Adreno kgsl driver. The Hexagon NPU exposes no equivalent
    counter (only the fastrpc-cdsp device node), so NPU use is evidenced by which execution
    provider ONNX Runtime actually opened, not by a percentage."""
    v = sh("cat /sys/class/kgsl/kgsl-3d0/gpu_busy_percentage 2>/dev/null").strip()
    m = re.match(r"(\d+)", v)
    return int(m.group(1)) if m else None


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--watch", type=float, default=0, help="seconds between samples; 0 = one shot")
    ap.add_argument("--csv", default="")
    ap.add_argument("--label", default="")
    ap.add_argument("--field", default="",
                    help="print ONE value and nothing else (AP|BAT|PA|SKIN|USB|batt). Shell callers "
                         "should use this instead of sed-ing the pretty line: the capture group in "
                         "`s/.*AP \\+\\([0-9.]*\\).*/\\1/p` gets mangled by successive quoting layers, "
                         "which silently yields an empty string and a guessed temperature.")
    a = ap.parse_args()

    if a.field:
        t, b = thermal(), battery()
        v = b["level"] if a.field == "batt" else t.get(a.field)
        if v is None:
            print("", end="")
            return 2
        print(v)
        return 0
    # ---- EXCLUSIVE LOGGER LOCK ----
    # Three separate times today a "killed" logger kept writing and interleaved its rows with a new
    # one, so two arms shared a sample window and BOTH reported the same peak (46.1 C / 36.2 C). An
    # overlapping window makes per-arm attribution impossible, and the damage is invisible in the CSV
    # itself. A second logger must therefore be unable to start at all, rather than relied upon to
    # have been killed. The lock stores the pid so a genuinely dead holder can be reclaimed.
    lock = None
    if a.csv:
        lock = os.path.join(os.path.dirname(os.path.abspath(a.csv)), ".thermal_logger.lock")
        if os.path.exists(lock):
            try:
                held = int(io.open(lock, encoding="utf-8").read().strip() or 0)
            except Exception:
                held = 0
            alive = False
            if held:
                r = subprocess.run(["tasklist", "/FI", "PID eq %d" % held],
                                   capture_output=True, text=True)
                alive = str(held) in (r.stdout or "")
            if alive:
                print("ANOTHER THERMAL LOGGER IS ALREADY RUNNING (pid %d) - refusing to start.\n"
                      "  Two loggers interleave rows into one CSV and make every overlapping arm "
                      "unattributable." % held)
                return 3
            print("  (reclaiming a stale lock from dead pid %s)" % held)
        io.open(lock, "w", encoding="utf-8").write(str(os.getpid()))

    fh = io.open(a.csv, "a", encoding="utf-8") if a.csv else None
    if fh and fh.tell() == 0:
        fh.write("iso,label,AP_C,BAT_C,PA_C,SKIN_C,USB_C,batt_pct,batt_C,mV,usb_powered,"
                 "current_uA,charge_uAh,watts,gpu_busy_pct\n")
    first = True
    while True:
        t, b = thermal(), battery()
        iso = time.strftime("%Y-%m-%dT%H:%M:%S")
        gb = gpu_busy()
        row = (iso, a.label, t.get("AP"), t.get("BAT"), t.get("PA"), t.get("SKIN"), t.get("USB"),
               b["level"], b["temperature_C"], b["voltage_mV"], b["usb_powered"],
               b["current_uA"], b["charge_uAh"], b["watts"], gb)
        if first or not a.watch:
            print("%-19s AP %5s  BAT %5s  PA %5s  SKIN %5s | batt %3s%% %5s C %s mV  usb=%s"
                  % (iso, t.get("AP"), t.get("BAT"), t.get("PA"), t.get("SKIN"),
                     b["level"], b["temperature_C"], b["voltage_mV"], b["usb_powered"]))
            first = False
        else:
            print("%-19s AP %5s SKIN %5s batt %3s%% %6s W gpu %3s%%"
                  % (iso, t.get("AP"), t.get("SKIN"), b["level"], b["watts"], gb))
        if fh:
            fh.write(",".join("" if v is None else str(v) for v in row) + "\n")
            fh.flush()
        if not a.watch:
            break
        try:
            time.sleep(a.watch)
        except KeyboardInterrupt:
            break
    if lock:
        try:
            os.remove(lock)
        except OSError:
            pass
    return 0


if __name__ == "__main__":
    sys.exit(main())
