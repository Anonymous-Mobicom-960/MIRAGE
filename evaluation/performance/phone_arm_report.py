#!/usr/bin/env python3
"""
phone_arm_report.py - per-arm thermal / battery summary, stated NET OF THE IDLE CONTROL.

WHY NET. The device warms on its own: an idle control at the campaign's pinned settings drifted
-0.1 C, but an earlier control on AUTO-brightness drifted +3.2 C AP with no load at all. Reporting a
raw rise would credit the pipeline with heat it did not produce. Every arm is therefore reported
against the control measured at the SAME settings, and the control's own drift is printed so the
reader can judge it.

⚠️ Sampling is not free: each sample is a `dumpsys` over WiFi, which itself wakes the SoC. The
control carries the same overhead, so the NET figure largely cancels it -- the RAW figure does not.

⚠️ SKIN hot-throttling begins at 38.0 C on this device (read from the device, not assumed), so any
arm whose SKIN approaches that is throttling and its wall-clock is not comparable to a cooler arm's.
"""
import csv
import io
import os
import sys

CSV = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..",
                   "_e2e", "A100_20260726", "PHONE_THERMAL.csv")
THROTTLE_SKIN = 38.0
CONTROL = os.environ.get("MIRAGE_CONTROL_LABEL", "final_control")
# the control must be the one measured at the SAME pinned settings as the arms;
# earlier controls ran on auto-brightness and are not comparable.


def load():
    if not os.path.exists(CSV):
        return []
    return list(csv.DictReader(io.open(CSV, encoding="utf-8")))


def stats(rows):
    f = lambda k: [float(r[k]) for r in rows if r.get(k)]
    out = {}
    for k in ("AP_C", "SKIN_C", "BAT_C"):
        v = f(k)
        if v:
            out[k] = dict(start=v[0], end=v[-1], peak=max(v), rise=v[-1] - v[0], span=max(v) - min(v))
    b = [int(r["batt_pct"]) for r in rows if r.get("batt_pct")]
    if b:
        out["batt"] = dict(start=b[0], end=b[-1], drop=b[0] - b[-1])
    out["n"] = len(rows)
    if rows:
        out["from"], out["to"] = rows[0]["iso"][11:], rows[-1]["iso"][11:]
    return out


def main():
    rows = load()
    if not rows:
        print("no thermal csv yet")
        return 1
    labels = []
    for r in rows:
        if r["label"] not in labels:
            labels.append(r["label"])
    ctrl = stats([r for r in rows if r["label"] == CONTROL])
    print("CONTROL (%s): n=%s  AP peak %.1f (span %.1f)  SKIN peak %.1f  -- the no-load reference\n"
          % (CONTROL, ctrl.get("n"), ctrl.get("AP_C", {}).get("peak", float("nan")),
             ctrl.get("AP_C", {}).get("span", float("nan")),
             ctrl.get("SKIN_C", {}).get("peak", float("nan"))) if ctrl else "no control arm yet\n")
    print("%-22s %-5s %-13s %-13s %-11s %s" % ("arm", "n", "AP peak (net)", "SKIN peak", "batt", "throttling?"))
    for lab in labels:
        if lab == CONTROL:
            continue
        s = stats([r for r in rows if r["label"] == lab])
        ap = s.get("AP_C", {}).get("peak")
        sk = s.get("SKIN_C", {}).get("peak")
        cap = ctrl.get("AP_C", {}).get("peak") if ctrl else None
        net = ("%.1f (%+.1f)" % (ap, ap - cap)) if (ap and cap) else ("%.1f" % ap if ap else " - ")
        bt = s.get("batt", {})
        thr = " - "
        if sk:
            thr = "**YES %.1f>=38**" % sk if sk >= THROTTLE_SKIN else "no (%.1f C headroom)" % (THROTTLE_SKIN - sk)
        print("%-22s %-5s %-13s %-13s %-11s %s"
              % (lab, s.get("n"), net, ("%.1f" % sk) if sk else " - ",
                 ("%d->%d%%" % (bt["start"], bt["end"])) if bt else " - ", thr))
    return 0


if __name__ == "__main__":
    sys.exit(main())
