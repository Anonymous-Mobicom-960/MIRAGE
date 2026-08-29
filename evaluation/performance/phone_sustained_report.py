#!/usr/bin/env python3
"""
phone_sustained_report.py - latency AS A FUNCTION OF TEMPERATURE, from a sustained-load run.

Every phone latency in this project was measured on a device that started cool, gated at a fixed
temperature. A wearable does not work that way: it runs continuously and heats up. §C.PHONE-1 could
only say "arms land within 2 C of the 38.0 C gate without plateauing" and had to mark the sustained
case NOT MEASURED, because extrapolating a throttled latency from an un-throttled one is a guess.

This reads the back-to-back Phase-1 iterations out of the app's own EVALS_LOG (authoritative ms) and
joins each one to the slice of the continuous thermal/power log that covers it, giving the pairing
that matters: how much slower does the SHIPPED (QNN) pipeline get, and at what temperature.

⚠️ The join is by wall-clock timestamp. The app logs an ISO timestamp per phase and the logger
samples every 5 s, so an iteration shorter than ~15 s would get too few samples to characterise.
Phase 1 at 120 f takes ~80-95 s, which is ample.
"""
import argparse
import csv
import datetime as dt
import io
import json
import os
import sys

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
CSV_POWER = os.path.join(REPO, "_e2e", "A100_20260726", "PHONE_POWER.csv")
THROTTLE_SKIN = 38.0


def load_iters(paths, phase, frames):
    """Phase-1 iterations from the app's log, in order, de-duplicated by timestamp."""
    seen, out = set(), []
    for p in paths:
        if not os.path.exists(p):
            continue
        for ln in io.open(p, encoding="utf-8"):
            ln = ln.strip()
            if not ln:
                continue
            try:
                r = json.loads(ln)
            except Exception:
                continue
            if r.get("phase") != phase or r.get("frames") != frames:
                continue
            k = r.get("ts")
            if k in seen:
                continue
            seen.add(k)
            out.append(r)
    out.sort(key=lambda r: r["ts"])
    return out


def load_samples(labels):
    rows = []
    if not os.path.exists(CSV_POWER):
        return rows
    for r in csv.DictReader(io.open(CSV_POWER, encoding="utf-8")):
        if not any(r.get("label", "").startswith(l) for l in labels):
            continue
        try:
            r["_t"] = dt.datetime.strptime(r["iso"], "%Y-%m-%dT%H:%M:%S")
        except Exception:
            continue
        rows.append(r)
    rows.sort(key=lambda r: r["_t"])
    return rows


def fnum(r, k):
    try:
        return float(r[k])
    except Exception:
        return None


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--logs", nargs="+", required=True, help="EVALS_LOG jsonl copies to merge")
    ap.add_argument("--labels", nargs="+", default=["sustained_120f"])
    ap.add_argument("--phase", default="1-Inpaint")
    ap.add_argument("--frames", type=int, default=120)
    a = ap.parse_args()

    iters = load_iters(a.logs, a.phase, a.frames)
    if not iters:
        print("no matching iterations found")
        return 1
    samples = load_samples(a.labels)

    # The app's ts marks phase COMPLETION, so an iteration's window is [ts - duration, ts].
    print("SUSTAINED LOAD - %s, %d f, back-to-back with NO cooldown" % (a.phase, a.frames))
    print("  %-4s %-9s %-10s %-11s %-11s %-9s %-8s %s"
          % ("iter", "app s", "ms/frame", "vs first", "SKIN pk", "AP pk", "batt", "throttling?"))
    base = None
    rows_out = []
    for i, r in enumerate(iters, 1):
        end = dt.datetime.strptime(r["ts"], "%Y-%m-%dT%H:%M:%S")
        start = end - dt.timedelta(milliseconds=r["ms"])
        win = [s for s in samples if start <= s["_t"] <= end]
        sk = [fnum(s, "SKIN_C") for s in win if fnum(s, "SKIN_C") is not None]
        apc = [fnum(s, "AP_C") for s in win if fnum(s, "AP_C") is not None]
        bp = [fnum(s, "batt_pct") for s in win if fnum(s, "batt_pct") is not None]
        secs = r["ms"] / 1000.0
        if base is None:
            base = secs
        d = (secs - base) / base * 100
        thr = ""
        if sk:
            thr = "**YES %.1f**" % max(sk) if max(sk) >= THROTTLE_SKIN else "no (%.1f C to gate)" % (THROTTLE_SKIN - max(sk))
        print("  %-4d %-9.1f %-10.0f %-11s %-11s %-9s %-8s %s"
              % (i, secs, r["ms_per_frame"], "%+.1f%%" % d,
                 ("%.1f" % max(sk)) if sk else "-", ("%.1f" % max(apc)) if apc else "-",
                 ("%.0f%%" % bp[-1]) if bp else "-", thr))
        rows_out.append(dict(i=i, secs=secs, skin=max(sk) if sk else None, n=len(win)))

    thin = [r for r in rows_out if r["n"] < 3]
    if thin:
        print("\n  ⚠️ iterations with <3 thermal samples (window too thinly covered to characterise): %s"
              % ", ".join(str(r["i"]) for r in thin))

    first, last = iters[0]["ms"] / 1000.0, iters[-1]["ms"] / 1000.0
    print("\n  DEGRADATION: %.1f s -> %.1f s = **%+.1f %%** over %d iterations"
          % (first, last, (last - first) / first * 100, len(iters)))
    crossed = [r for r in rows_out if r["skin"] and r["skin"] >= THROTTLE_SKIN]
    if crossed:
        print("  THROTTLE GATE (SKIN %.1f C) first crossed at iteration %d"
              % (THROTTLE_SKIN, crossed[0]["i"]))
    else:
        print("  throttle gate never crossed - degradation, if any, is not from SKIN mitigation")
    # steady state: has it plateaued?
    if len(rows_out) >= 4:
        tail = [r["secs"] for r in rows_out[-3:]]
        sp = (max(tail) - min(tail)) / (sum(tail) / 3) * 100
        print("  last 3 iterations: %s  (spread %.1f %% - %s)"
              % (" / ".join("%.1f" % t for t in tail), sp,
                 "PLATEAUED" if sp < 5 else "still degrading"))
    return 0


if __name__ == "__main__":
    sys.exit(main())
