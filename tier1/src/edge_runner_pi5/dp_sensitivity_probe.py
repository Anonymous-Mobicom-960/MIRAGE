#!/usr/bin/env python3
"""
dp_sensitivity_probe.py - does apply_dp's Laplace scale actually bound one record's influence?
===============================================================================================
    python3 dp_sensitivity_probe.py

The DP claim rests on Δ_c bounding the L1 change ONE unit of privacy can cause in the released
vector. That is an empirical property of the code, so it should be measured, not argued - this
is the runnable harness for it, committed so the ledger row has a re-runnable source.

ADVERSARIAL BASELINE - the detail that has already fooled one probe. Start every channel at its
LOWER bound and drive a single frame to its UPPER bound, so that frame's excursion is the full
range. A midpoint baseline caps the excursion at range/2 and under-reports the sensitivity by 2x;
that mistake is how a violated bound was recorded as holding on 2026-07-23.

SWEEP EVERY FRAME POSITION, not a sample. The worst case lives at the CLIP EDGE, because a
centred window near the boundary sees a frame more than once (or, with a partial-window mean,
weights it differently). Probing only interior frames misses the sup entirely.
"""
import os
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import config as C                                              # noqa: E402
import mirage_tier1 as ST                                       # noqa: E402
from mirage_tier1 import apply_dp                               # noqa: E402

CH = ["mouth_open", "mouth_width", "smile", "eyeR_open", "eyeL_open", "browR", "browL",
      "yaw", "pitch", "roll_rad", "gazeX", "gazeY"]


class _NoNoise:
    """Real config with the Laplace scale driven to ~0, so the released vector is deterministic.

    Noise is killed via a huge epsilon (b_c = Δ_c/eps_c → 0), NOT by zeroing SENSITIVITY. That
    matters: apply_dp now derives Δ per slot from the window geometry and only uses SENSITIVITY
    as a floor, so a zeroed SENSITIVITY no longer silences the mechanism - it would leave full
    noise in place and the probe would measure its own randomness instead of the sensitivity.
    """
    DP_ON = True
    DP_WINDOW = C.DP_WINDOW
    DP_QUANTIZE_BITS = C.DP_QUANTIZE_BITS
    CH_BOUNDS = C.CH_BOUNDS
    SENSITIVITY = list(C.SENSITIVITY)
    DP_EPS_PER_CH = [1e12] * 12
    DP_EPSILON_TOTAL = 12e12


def released(log):
    return np.array([fr[0] for fr in apply_dp([[list(p) for p in f] for f in log], _NoNoise)])


def probe_length(T, verbose=True):
    """-> list of (channel, worst_L1, x_range, worst_frame, bound_used, ok) for a length-T slot."""
    rows = []
    for c in range(12):
        lo, hi = C.CH_BOUNDS[c]
        rng = hi - lo
        base = [[[(C.CH_BOUNDS[k][0]) for k in range(12)]] for _ in range(T)]   # all at LOWER bound
        b = released(base)[:, c]
        worst, wf = 0.0, -1
        for i in range(T):                       # EVERY position - the sup is at the edge
            x = [[list(p) for p in f] for f in base]
            x[i][0][c] = hi                      # full-range excursion
            d = float(np.abs(released(x)[:, c] - b).sum())
            if d > worst:
                worst, wf = d, i
        # The bound the mechanism ACTUALLY uses for a slot of this length - derived per-slot in
        # apply_dp, not the module constant, which is baked at one length.
        q_extra = min(T, C.DP_WINDOW) / float(2 ** int(C.DP_QUANTIZE_BITS) - 1)
        bound = (ST._window_amplification(C.DP_WINDOW, T) + q_extra) * rng
        rows.append((CH[c], worst, worst / rng, wf, bound, worst <= bound + 1e-9))
    return rows


def main():
    env = os.environ.get("DP_PROBE_T")
    # SWEEP T. Probing a single length is how a violated bound was recorded as holding twice:
    # the amplification is NOT monotone in T and peaks at T == DP_WINDOW, so the default T=60
    # sits in the flat long-clip regime and never sees the worst case. Cover 1..3w plus a long one.
    lengths = [int(env)] if env else sorted(set(list(range(1, 3 * int(C.DP_WINDOW) + 2)) + [60]))
    print(f"window={C.DP_WINDOW}  quantize={C.DP_QUANTIZE_BITS} bits  "
          f"sweeping T over {lengths[0]}..{lengths[-1]} ({len(lengths)} lengths)")
    print(f"  {'T':>4} {'bound/rng':>10} {'worst xrange':>13} {'worst channel':>14}  verdict")
    bad = []
    for T in lengths:
        rows = probe_length(T)
        amp = (ST._window_amplification(C.DP_WINDOW, T)
               + min(T, C.DP_WINDOW) / float(2 ** int(C.DP_QUANTIZE_BITS) - 1))
        wi = max(range(12), key=lambda i: rows[i][2])
        fails = [r for r in rows if not r[5]]
        if fails:
            bad.append((T, fails))
        print(f"  {T:4d} {amp:10.4f} {rows[wi][2]:13.4f} {rows[wi][0]:>14}  "
              f"{'OK' if not fails else f'VIOLATED ({len(fails)}/12 ch)'}")
    if bad:
        print("\nBOUND VIOLATED at T = " + ", ".join(str(t) for t, _ in bad))
        print("=> the configured epsilon is NOT delivered at those slot lengths.")
        raise SystemExit(1)
    print("\nEvery channel at every swept length: worst single-frame L1 <= the per-slot bound.")
    print("The PER-FRAME bound HOLDS for slots from 1 frame to 60.")
    print("NOTE: this is the PER-FRAME unit only. Person-level DP (one person contributes T_p "
          "frames) remains open - see ledger §I.1.")


if __name__ == "__main__":
    main()
