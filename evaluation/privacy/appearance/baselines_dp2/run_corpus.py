#!/usr/bin/env python3
"""Run DeepPrivacy2 (authors' own implementation, unmodified) over the 10 fps corpus.

Runs INSIDE the dp2:pinned container. Loads the anonymizer ONCE and loops, so the ~30 s model
build is paid once rather than 108 times, and every clip is resumable.

SETTINGS AND WHY:
  configs/anonymizers/FB_cse.py  the only full-body config whose checkpoints are still obtainable
                                 (`fdh_styleganL_nocse.ckpt` is HTTP 410 with no mirror anywhere,
                                 which blocks FB_cse_mask and FB_cse_mask_face).
  --mt (multi_modal_truncation)  MANDATORY. The CLI default `truncation_value=0` makes
                                 `get_truncated` return `w_avg.lerp(w, 0)` == `w_avg` exactly, so
                                 every person in every frame becomes the identical mean identity and
                                 --track/--seed go inert (repo issue #14). Multi-modal truncation
                                 instead picks a distinct w_center per track id, which is what the
                                 author states he used for all tracking experiments.
  --track                        one cached latent per tracked person (motpy Kalman bbox tracker),
                                 reset per clip -> per-sequence identity, the analogue of MIRAGE's
                                 per-sequence seeding.
  fps=None, max_res=None         output must be frame-for-frame and pixel-for-pixel aligned with the
                                 input, so the common extractor sees the same timeline for every arm.
  seed=0                         tops.set_seed(0)

Failures are recorded, never silently skipped.
"""
import hashlib
import json
import os
import sys
import time
import traceback
from pathlib import Path

import numpy as np
import tops
from tops.config import instantiate

sys.path.insert(0, "/home/testuser/dp2")
from dp2 import utils                      # noqa: E402
import anonymize as A                      # noqa: E402

IN = Path("/data/in")
OUT = Path("/data/out")
REPORT = Path("/data/out/_RUN_DP2.json")
CFG = "configs/anonymizers/FB_cse.py"
END_S = 12          # 12 s @10 fps = 120 frames = the published per-clip budget
                    # (build_pose.MAX_FRAMES / build_cache.MAX_EMIT). Anonymising beyond it
                    # would be compute nobody scores.


def main():
    tops.set_seed(0)
    cfg = utils.load_config(CFG)
    cfg.detector.score_threshold = 0.3
    anonymizer = instantiate(cfg.anonymizer, load_cache=False)

    # does the checkpoint carry w_centers? decides --mt vs a truncation fallback
    G = anonymizer.generators[type(list(anonymizer.generators)[0])] if False else None
    has_wc = False
    for k, g in anonymizer.generators.items():
        if g is not None and hasattr(g, "style_net") and hasattr(g.style_net, "w_centers"):
            has_wc = True
            print(f"[dp2] {k.__name__}: w_centers present, "
                  f"n={len(g.style_net.w_centers)}", flush=True)
    synth = dict(amp=True, multi_modal_truncation=bool(has_wc),
                 truncation_value=0.0 if has_wc else 0.5,
                 text_prompt=None, text_prompt_strength=0.5)
    print(f"[dp2] synthesis_kwargs = {synth}", flush=True)

    OUT.mkdir(parents=True, exist_ok=True)
    clips = sorted(p for p in IN.iterdir() if p.suffix.lower() == ".mp4")
    print(f"[dp2] {len(clips)} clips to process", flush=True)

    rec = {}
    if REPORT.exists():
        rec = json.load(open(REPORT))
    t0 = time.time()
    for i, src in enumerate(clips, 1):
        dst = OUT / src.name
        if dst.exists() and rec.get(src.name, {}).get("ok"):
            continue
        t1 = time.time()
        try:
            A.anonymize_video(src, dst, anonymizer=anonymizer, visualize=False,
                              max_res=None, start_time=0, fps=None, end_time=END_S,
                              visualize_detection=False, track=True,
                              synthesis_kwargs=dict(synth))
            dt = time.time() - t1
            rec[src.name] = dict(ok=True, seconds=dt, out=str(dst))
            print(f"[dp2] {i:3d}/{len(clips)} {src.name[:38]:38s} {dt/60:6.2f} min "
                  f"[total {(time.time()-t0)/60:.1f} min]", flush=True)
        except Exception as e:
            rec[src.name] = dict(ok=False, error=f"{type(e).__name__}: {e}",
                                 trace=traceback.format_exc()[-2000:])
            print(f"[dp2] {i:3d}/{len(clips)} {src.name[:38]:38s} FAILED {type(e).__name__}: {e}",
                  flush=True)
        json.dump(rec, open(REPORT, "w"), indent=1)

    ok = sum(1 for v in rec.values() if v.get("ok"))
    tot = sum(v.get("seconds", 0) for v in rec.values() if v.get("ok"))
    print(f"\n[dp2] DONE {ok}/{len(clips)} clips in {(time.time()-t0)/60:.1f} min "
          f"(sum per-clip {tot/60:.1f} min)", flush=True)
    for k, v in rec.items():
        if not v.get("ok"):
            print(f"[dp2] FAILED: {k} -> {v.get('error')}", flush=True)


if __name__ == "__main__":
    main()
