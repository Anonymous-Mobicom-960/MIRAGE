"""
segbench.collage -- assemble the 4 headline masked videos into one 2x2 grid
video with a per-tile metric HUD (seg/total ms, FPS, body-reveal, over-mask,
and the FACE-GATE verdict) and a gate-coloured border (green=pass, red=fail),
so the privacy/quality/speed trade-off is visible frame-by-frame.

x86 latency is labelled INDICATIVE on the HUD -- the honest caveat travels with
the picture.
"""
import os
import json
import cv2
import numpy as np

GREEN = (60, 200, 60)
RED = (40, 40, 220)
WHITE = (255, 255, 255)
YELL = (40, 220, 220)


def _load(run_dir):
    with open(os.path.join(run_dir, "summary.json")) as fh:
        s = json.load(fh)
    with open(os.path.join(run_dir, "metrics.json")) as fh:
        m = json.load(fh)
    cap = cv2.VideoCapture(os.path.join(run_dir, "masked.mp4"))
    return s, m, cap


def _panel_text(tile, lines, org, scale=0.62, thick=2, pad=8):
    x, y = org
    hgt = int(28 * scale / 0.62)
    w = max(cv2.getTextSize(t, cv2.FONT_HERSHEY_SIMPLEX, scale, thick)[0][0] for t, _ in lines) + 2 * pad
    h = hgt * len(lines) + pad
    ov = tile.copy()
    cv2.rectangle(ov, (x - pad, y - hgt), (x - pad + w, y - hgt + h), (0, 0, 0), -1)
    cv2.addWeighted(ov, 0.55, tile, 0.45, 0, tile)
    for i, (t, col) in enumerate(lines):
        cv2.putText(tile, t, (x, y + i * hgt), cv2.FONT_HERSHEY_SIMPLEX, scale, col, thick, cv2.LINE_AA)


def build_collage(run_dirs, out_path, tile=632, titles=None):
    loaded = [_load(d) for d in run_dirs]
    fps_out = loaded[0][0].get("frames") and 10.0 or 10.0
    W = H = tile * 2
    writer = cv2.VideoWriter(out_path, cv2.VideoWriter_fourcc(*"mp4v"), 10.0, (W, H))

    n_frames = int(min(l[2].get(cv2.CAP_PROP_FRAME_COUNT) for l in loaded))
    for fi in range(n_frames):
        canvas = np.zeros((H, W, 3), np.uint8)
        for idx, (s, m, cap) in enumerate(loaded):
            ok, fr = cap.read()
            if not ok:
                fr = np.zeros((tile, tile, 3), np.uint8)
            fr = cv2.resize(fr, (tile, tile))
            gate = m["privacy_gate_pass"]
            bcol = GREEN if gate else RED
            cv2.rectangle(fr, (2, 2), (tile - 3, tile - 3), bcol, 6)
            title = (titles[idx] if titles else s["name"])
            lines = [
                (title, WHITE),
                (f"seg {s['t_seg_ms']['median']:.0f}ms  tot {s['t_total_ms']['median']:.0f}ms  {s['fps_from_total']:.0f} FPS*", YELL),
                (f"reveal {m['body_reveal_rate']*100:.2f}%  overmask {m['overmask_rate']*100:.1f}%", WHITE),
                (f"FACE-GATE {'PASS' if gate else 'FAIL'}  (leak {m['face_leak_frames']}/{s['frames']})", GREEN if gate else RED),
            ]
            _panel_text(fr, lines, (16, 34))
            r, c = idx // 2, idx % 2
            canvas[r*tile:(r+1)*tile, c*tile:(c+1)*tile] = fr
        cv2.putText(canvas, "*x86 latency = INDICATIVE only (ARM re-measured separately)",
                    (16, H - 16), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (200, 200, 200), 2, cv2.LINE_AA)
        writer.write(canvas)
    for _, _, cap in loaded:
        cap.release()
    writer.release()
    return out_path
