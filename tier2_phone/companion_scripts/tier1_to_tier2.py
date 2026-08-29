#!/usr/bin/env python3
"""
tier1_to_tier2.py - package a real Tier-1 run into the phone app's input contract.
==================================================================================
The missing link. Tier-1 emits LOSSLESS FFV1 `.mkv` (`masked_video.mkv`, `mask.mkv`); the phone
app reads H.264 `masked_video.mp4` / `mask.mp4` (`MiragePaths.kt:51-52`). Nothing bridged them,
so **the shipped Tier-1 had never once fed the shipped Tier-2** - every phone test set came from
cloud exports.

    python tier1_to_tier2.py --clip input/reid_dataset/d07_2person_1264.mp4 --tag C

Runs Tier-1 at its DEPLOYED config, transcodes, and then MEASURES whether the transcode was
safe, because that is the real risk here:

  §2 RISK - the mask is a privacy artifact. Tier-1 writes it lossless on purpose: "a lossy codec
  would let DCT ringing re-expand the decoded binary mask past the erode margin, leaking a ring
  of real background into the cloud-bound person region" (mirage_tier1.py). The phone contract is
  H.264, so ringing is now unavoidable - but it must be quantified, and it must ERODE-OR-EQUAL
  rather than expand into background. We measure IoU, and separately the pixels the transcode
  ADDS (grew into background) vs DROPS, because those two have opposite privacy signs:
      dropped px  -> mask shrank  -> a sliver of REAL PERSON may be revealed  (§2 REGRESSION)
      added px    -> mask grew    -> more grey, harmless to privacy, costs inpaint area
  A -crf 0 lossless H.264 fallback is used when the tolerance is exceeded.
"""
import argparse
import json
import os
import shutil
import subprocess
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.dirname(os.path.dirname(HERE))
EDGE = os.path.join(REPO, "tier1", "src", "edge_runner_pi5")


def run_tier1(clip, outdir, max_frames=0):
    os.makedirs(outdir, exist_ok=True)
    drv = os.path.join(outdir, "_t1.py")
    with open(drv, "w", encoding="utf-8") as f:
        f.write(f'''
import os, sys
sys.path.insert(0, r"{EDGE}")
import config as C
C.OUT_DIR = sys.argv[2]
os.makedirs(C.OUT_DIR, exist_ok=True)
import mirage_tier1 as ST
sys.argv = ["mirage_tier1.py", "--source", sys.argv[1], "--max-frames", sys.argv[3]]
ST.main()
''')
    r = subprocess.run([sys.executable, drv, clip, outdir, str(max_frames)],
                       capture_output=True, text=True, encoding="utf-8", errors="replace")
    ok = all(os.path.exists(os.path.join(outdir, n))
             for n in ("masked_video.mkv", "mask.mkv", "pose.json"))
    return ok, (r.stderr or "")[-800:]


def read_frames(path, gray=False):
    import cv2
    cap = cv2.VideoCapture(path)
    out = []
    while True:
        ok, f = cap.read()
        if not ok:
            break
        out.append((f[..., 0] if (gray and f.ndim == 3) else f))
    fps = cap.get(cv2.CAP_PROP_FPS); cap.release()
    return out, float(fps or 0)


def transcode(src, dst, fps, lossless=False):
    cmd = ["ffmpeg", "-y", "-v", "error", "-i", src,
           "-c:v", "libx264", "-pix_fmt", "yuv420p", "-r", f"{fps:g}"]
    # 🔴 NOT `-crf 0`: x264 lossless emits profile High 4:4:4 Predictive, which Android cannot
    # decode -- the app then silently guesses the hole from the gray fill (FIX11_HOLE.md).
    # crf 6 + profile high + all-intra is bit-identical after the consumer's >=128 threshold.
    cmd += (["-crf", "6", "-profile:v", "high", "-g", "1", "-bf", "0", "-preset", "veryslow"]
            if lossless else ["-crf", "16", "-preset", "slow"])
    cmd.append(dst)
    return subprocess.run(cmd, capture_output=True, text=True).returncode == 0


def mask_fidelity(src_mkv, dst_mp4):
    """Compare the lossless source mask with the transcoded one after the app's own >127 rule."""
    a, _ = read_frames(src_mkv, gray=True)
    b, _ = read_frames(dst_mp4, gray=True)
    n = min(len(a), len(b))
    inter = union = dropped = added = tot = 0
    worst = 1.0
    for i in range(n):
        x = a[i] > 127
        y = b[i] > 127
        inter += int((x & y).sum()); union += int((x | y).sum())
        dropped += int((x & ~y).sum()); added += int((~x & y).sum()); tot += int(x.sum())
        u = int((x | y).sum())
        if u:
            worst = min(worst, int((x & y).sum()) / u)
    return {"frames": n, "iou": inter / union if union else 1.0, "worst_frame_iou": worst,
            "dropped_px": dropped, "added_px": added, "source_px": tot,
            "dropped_frac_of_source": dropped / tot if tot else 0.0,
            "added_frac_of_source": added / tot if tot else 0.0}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--clip", required=True)
    ap.add_argument("--tag", default="C")
    ap.add_argument("--max-frames", type=int, default=0)
    ap.add_argument("--work", default=os.path.join(REPO, "input", "sets", "_t1_work"))
    ap.add_argument("--sets", default=os.path.join(REPO, "input", "sets"))
    ap.add_argument("--drop-tol", type=float, default=0.001,
                    help="max fraction of source mask pixels the transcode may DROP before we "
                         "fall back to lossless (dropped px can reveal real person)")
    a = ap.parse_args()

    work = os.path.join(a.work, a.tag)
    print(f"[bridge] Tier-1 (deployed config) on {a.clip}")
    ok, err = run_tier1(a.clip, work, a.max_frames)
    if not ok:
        raise SystemExit(f"Tier-1 failed:\n{err}")
    mk_mkv = os.path.join(work, "mask.mkv"); mv_mkv = os.path.join(work, "masked_video.mkv")
    _, fps = read_frames(mk_mkv, gray=True)
    print(f"[bridge] Tier-1 wrote FFV1 mask.mkv + masked_video.mkv @{fps:g} fps")

    os.makedirs(a.sets, exist_ok=True)
    mk_mp4 = os.path.join(a.sets, f"mask_{a.tag}.mp4")
    mv_mp4 = os.path.join(a.sets, f"masked_video_{a.tag}.mp4")
    if not transcode(mk_mkv, mk_mp4, fps) or not transcode(mv_mkv, mv_mp4, fps):
        raise SystemExit("ffmpeg transcode failed")

    fid = mask_fidelity(mk_mkv, mk_mp4)
    print(f"[bridge] mask transcode (crf16): IoU {fid['iou']:.6f}  worst-frame {fid['worst_frame_iou']:.6f}")
    print(f"           dropped {fid['dropped_px']} px ({fid['dropped_frac_of_source']*100:.4f}% of mask) "
          f"| added {fid['added_px']} px ({fid['added_frac_of_source']*100:.4f}%)")
    used_lossless = False
    if fid["dropped_frac_of_source"] > a.drop_tol:
        print(f"[bridge] !! dropped fraction exceeds tolerance {a.drop_tol} - a shrinking mask can "
              f"reveal real person pixels (§2). Re-encoding LOSSLESS.")
        transcode(mk_mkv, mk_mp4, fps, lossless=True)
        fid = mask_fidelity(mk_mkv, mk_mp4)
        used_lossless = True
        print(f"[bridge] lossless mask: IoU {fid['iou']:.6f}  dropped {fid['dropped_px']} px")

    rep = {"tag": a.tag, "source_clip": os.path.abspath(a.clip), "fps": fps,
           "tier1_out": os.path.abspath(work), "mask_mp4": os.path.abspath(mk_mp4),
           "masked_video_mp4": os.path.abspath(mv_mp4),
           "mask_transcode": fid, "mask_encoded_lossless": used_lossless,
           "pose_meta": {k: v for k, v in
                         json.load(open(os.path.join(work, "pose.json"), encoding="utf-8")).items()
                         if k != "frames"}}
    out = os.path.join(HERE, "reports", f"TIER1_TO_TIER2_{a.tag}.json")
    os.makedirs(os.path.dirname(out), exist_ok=True)
    json.dump(rep, open(out, "w"), indent=2)
    print(f"\n[bridge] set {a.tag} staged in {a.sets}: mask_{a.tag}.mp4, masked_video_{a.tag}.mp4")
    print(f"[bridge] wrote {out}")
    print(f"[bridge] pose.json meta: {rep['pose_meta']}")
    print("\n[bridge] NOTE: a full phone set also needs the CHARACTER (synthetic_person_p*.mp4),")
    print("         which only the cloud tier produces. This bridge supplies the TIER-1 half.")


if __name__ == "__main__":
    main()
