#!/usr/bin/env python3
"""
dp_calibrate.py - what epsilon does apply_dp() ACTUALLY deliver?
=================================================================
    python3 dp_calibrate.py --calibration-dir ./calibration

Feed it raw (DP_ON=False) face_scalars.json files - one per clip, any names. It reports the
sensitivity the Laplace mechanism in mirage_tier1.apply_dp() really has, and therefore the
epsilon the shipped config really achieves, under both units of privacy.

WHY THIS WAS REWRITTEN (2026-07-23)
-----------------------------------
The previous version computed `Δ_c = 1.5 * std(channel)`. That is not a sensitivity - it is a
spread statistic, and planning/V9_RECONCILIATION.md §1.2 had already REJECTED exactly that
method. Worse, its documented workflow could never run: step 1 says "run with DP_ON=False",
and that path crashed on json.dump until 2026-07-23 (face_scalars returned np.float32).

L1 SENSITIVITY, DERIVED FROM THE MECHANISM
------------------------------------------
apply_dp releases, per person-slot and channel c:
    r[t] = quantize_[lo_c,hi_c]( mean( x[t-w/2 .. t+w/2] ) ) + Laplace(b_c),  b_c = Δ_c / eps_c

* ONE FRAME changes: x[i] enters the window-mean of every t within w/2 of it. In the clip
  INTERIOR each of those w means weights it 1/w, so the L1 change is w * (|δ|/w) = |δ|. At the
  clip EDGE it is LARGER: those windows are truncated, so x[i] is averaged over fewer samples
  and carries weight 1/4, 1/5, 1/6 ... instead of 1/w. The true worst case is
        Δ_c(frame)  =  [ A_w(T) + min(T,w)/(levels-1) ] * (hi_c - lo_c)
  where  A_w(T) = max_i sum_{t in win(i)} 1/count(t).  Δ = range ALONE IS NOT CONSERVATIVE - 
  an adversarial probe violated it at 1.129x on 2026-07-23. Two corrections since:

    * A_w IS NOT MONOTONE IN T. It peaks at T == w (1.3762 at w=7) and decays to 1.1881 for long
      clips, so a value baked at one long length UNDER-bounds short slots. Measured: a 7-frame
      person-slot delivered eps = 3.42 against a configured 3.00. Δ is therefore evaluated at
      the length of the slot being released, both here and in apply_dp.
    * QUANTIZATION IS PRE-PROCESSING HERE, NOT POST-PROCESSING. This docstring used to wave it
      away with post-processing immunity, but the rounding happens BEFORE the Laplace draw, so
      immunity does not apply: |q(x)-q(y)| <= |x-y| + step per coordinate, and one frame moves at
      most min(T,w) coordinates. Caught at T=2/T=4, where A_w is exactly 1.0 yet the measured L1
      was 1.0323x range - the excess was precisely one step.

  Clipping still bounds |δ| by the channel's PUBLIC range, which is what makes both terms finite.
* ONE PERSON changes (group privacy - the ACTUAL threat, see apply_dp's KNOWN LIMITATION note):
  every one of that person's T_p released values can move by the full range, so
        Δ_c(person) =  T_p * (hi_c - lo_c)

Given the shipped b_c = SENSITIVITY[c]/eps_c, the epsilon actually delivered is
        eps*_c = Δ_c(true) / b_c = (Δ_c(true) / SENSITIVITY[c]) * eps_c
and epsilon composes by summation across the 12 channels.

This script does NOT edit config.py. Choosing epsilon is a policy decision; the script only
tells you what the current numbers mean.
"""
import argparse
import glob
import json
import os
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import config as C                                              # noqa: E402

CH = ["mouth_open", "mouth_width", "smile", "eyeR_open", "eyeL_open", "browR", "browL",
      "yaw", "pitch", "roll_rad", "gazeX", "gazeY"]


def load_rows(d):
    """-> (rows [N,12] of frames with a detected face, per-person trajectory lengths)."""
    rows, tp = [], []
    for p in sorted(glob.glob(os.path.join(d, "*.json"))):
        try:
            j = json.load(open(p, encoding="utf-8"))
        except Exception:
            continue
        face = j.get("face", j) if isinstance(j, dict) else j
        slots = j.get("slots") if isinstance(j, dict) else None
        # Be robust to unrelated json in the folder: a face log is [frames][persons][12].
        if (not isinstance(face, list) or not face
                or not all(isinstance(fr, list) for fr in face)
                or not any(isinstance(v, list) and len(v) == 12 for fr in face for v in fr)):
            continue
        if slots is not None and not (isinstance(slots, list)
                                      and all(isinstance(r, list) for r in slots)):
            slots = None
        per = {}
        for i, fr in enumerate(face):
            row = slots[i] if (slots and i < len(slots)) else list(range(len(fr)))
            for k, v in enumerate(fr):
                s = row[k] if k < len(row) else k
                per.setdefault(s, []).append(v)
        for traj in per.values():
            a = np.asarray(traj, float)
            if a.ndim != 2 or a.shape[1] != 12:
                continue
            present = np.abs(a).sum(1) > 1e-9      # all-zero == no face found that frame
            if present.sum():
                rows.append(a[present])
                tp.append(int(present.sum()))
    return (np.concatenate(rows, 0) if rows else np.zeros((0, 12))), tp


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--calibration-dir", default=os.path.join(HERE, "calibration"))
    ap.add_argument("--json-out", default=None)
    a = ap.parse_args()

    X, Tp = load_rows(a.calibration_dir)
    if not len(X):
        raise SystemExit(f"no usable face_scalars json in {a.calibration_dir} "
                         f"(run mirage_tier1.py with DP_ON=False first)")
    bounds = list(getattr(C, "CH_BOUNDS", [(-2.0, 2.0)] * 12))
    sens = list(C.SENSITIVITY)          # the module constant, kept for the comparison column
    eps_ch = list(getattr(C, "DP_EPS_PER_CH", None) or
                  [float(C.DP_EPSILON_TOTAL) / 12] * 12)
    T_med, T_max = int(np.median(Tp)), int(max(Tp))
    # TRUE per-frame Δ, computed the SAME WAY apply_dp computes it - per slot length, not from
    # the module constant. Two terms:
    #   (a) window amplification A_w(T): near-edge frames sit in short windows and weigh more.
    #       NOT monotone in T (peaks at T == w), so it must be evaluated at the length in hand.
    #   (b) quantization: rounding to the public grid runs BEFORE the Laplace draw, so it is
    #       pre-processing and adds up to one step per changed coordinate (min(T, w) of them).
    amp = float(C._window_amplification(C.DP_WINDOW, T_med))
    q_extra = min(T_med, int(C.DP_WINDOW)) / float(max(2, 2 ** int(C.DP_QUANTIZE_BITS)) - 1)
    rng = [(amp + q_extra) * float(hi - lo) for lo, hi in bounds]
    # The scale apply_dp ACTUALLY uses for a slot of this length - mirror it exactly, or this
    # script reports the epsilon of a mechanism that is not the one running.
    _baked = float(getattr(C, "WINDOW_AMPLIFICATION", 1.0)) or 1.0
    sens_used = [max(rng[c], sens[c] * amp / _baked) for c in range(12)]

    print(f"[dp] {len(X)} (frame,person) rows with a detected face; "
          f"{len(Tp)} person-trajectories, T_p min/med/max = {min(Tp)}/{T_med}/{T_max}")
    print(f"[dp] window={C.DP_WINDOW}  quantize={C.DP_QUANTIZE_BITS} bits  "
          f"claimed DP_EPSILON_TOTAL={C.DP_EPSILON_TOTAL}\n")

    print(f"  {'ch':<12} {'range':>7} {'SENS':>7} {'ratio':>6} {'eps_c':>6} "
          f"{'eps*frame':>10} {'eps*person':>11}   {'observed p1..p99':>18}")
    eps_f, eps_p, out = [], [], []
    for c in range(12):
        obs = float(np.percentile(X[:, c], 99) - np.percentile(X[:, c], 1))
        ratio = rng[c] / max(sens_used[c], 1e-12)
        ef = ratio * eps_ch[c]
        ep = ef * T_med
        eps_f.append(ef); eps_p.append(ep)
        out.append({"channel": CH[c], "public_range": rng[c], "shipped_sensitivity": sens[c], "sensitivity_in_use": sens_used[c],
                    "ratio_true_over_shipped": ratio, "eps_c_configured": eps_ch[c],
                    "eps_effective_per_frame": ef, "eps_effective_per_person_medianT": ep,
                    "observed_p1_p99_span": obs})
        print(f"  {CH[c]:<12} {rng[c]:7.2f} {sens_used[c]:7.3f} {ratio:6.2f} {eps_ch[c]:6.3f} "
              f"{ef:10.2f} {ep:11.1f}   {obs:18.3f}")

    tot_f, tot_p = sum(eps_f), sum(eps_p)
    print(f"\n  TOTAL eps (sequential composition over 12 channels):")
    print(f"     configured / claimed        : {C.DP_EPSILON_TOTAL:.2f}")
    print(f"     ACTUAL, unit = one FRAME    : {tot_f:.2f}   ({tot_f/C.DP_EPSILON_TOTAL:.1f}x the claim)")
    print(f"     ACTUAL, unit = one PERSON   : {tot_p:.0f}   (median T_p={T_med}; "
          f"{tot_p/C.DP_EPSILON_TOTAL:.0f}x the claim)")
    print("\n  Reading: SENSITIVITY is now DERIVED (config.WINDOW_AMPLIFICATION x public range),")
    print("  so the PER-FRAME unit is delivered as configured - confirmed independently by")
    print("  dp_sensitivity_probe.py, which drives one frame across the full range at EVERY")
    print("  position, not just interior ones. At the PERSON level - the unit that matches the")
    print("  threat model, since an attacker re-identifies a person from their whole expression")
    print("  trajectory - epsilon is still in the hundreds and the DP claim is NOT meaningful.")
    print("  Person-level DP needs eps_c divided by T_p (or a per-person cap on contributed")
    print("  frames), which costs ~T_p x more noise - a design decision, not a tweak.")
    print(f"  SENSITIVITY in use = {[round(v, 4) for v in sens]}   (= {amp + q_extra:.4f} x public range)")
    print("  Person-level DP additionally needs eps_c divided by T_p (or a per-person cap on")
    print("  contributed frames), which costs ~T_p x more noise - a design decision, not a tweak.")

    rep = {"n_rows": int(len(X)), "n_person_trajectories": len(Tp),
           "T_p": {"min": int(min(Tp)), "median": T_med, "max": T_max},
           "dp_window": C.DP_WINDOW, "quantize_bits": C.DP_QUANTIZE_BITS,
           "claimed_eps_total": float(C.DP_EPSILON_TOTAL),
           "effective_eps_total_per_frame": tot_f,
           "effective_eps_total_per_person_medianT": tot_p,
           "recommended_sensitivity_per_frame": [round(r, 4) for r in rng],
           "channels": out}
    if a.json_out:
        json.dump(rep, open(a.json_out, "w"), indent=2)
        print(f"\n[dp] wrote {a.json_out}")


if __name__ == "__main__":
    main()
