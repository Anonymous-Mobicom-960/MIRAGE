#!/usr/bin/env python3
"""Domain-registration helpers shared by the appearance-channel harness.

The three functions below are copied verbatim from the internal harness modules that produced
the published numbers (`build_silhouettes.normalise` and `build_pose.casia_reference` /
`build_pose.to_casia_domain`); only this header and the imports differ. They exist because both
adversary families were trained on CASIA-B, whose inputs are person-registered crops, not
full-frame coordinates:

* `normalise` applies OpenGait's own CASIA-B pretreatment to one binary mask: tight crop,
  height-normalised to 64 px, horizontally centred on the upper-body centroid, 64x64 output.
  Centring on the TOP-half centroid (not the full-mask centroid) is what OpenGait does; it stops
  swinging arms and legs from translating the whole body left/right frame to frame, which the
  recogniser would read as gait.
* `casia_reference` derives the POPULATION mid-hip position and torso length from the CASIA-B
  train split (subjects 1..74). Population values, never fitted to any clip of ours.
* `to_casia_domain` maps a native-pixel (T,17,3) track into the adversary's coordinate domain
  using the track's OWN median torso: translate the median mid-hip onto the CASIA-B population
  mid-hip, scale the median torso onto the population torso. Without this registration the
  GaitGraph-family adversary sees coordinates ~3x outside its training distribution, every
  embedding collapses to nearly the same point, and any "separation" read off the result is
  noise.
"""
import numpy as np

H_OUT, W_OUT = 64, 64          # OpenGait CASIA-B domain


def normalise(mask):
    """OpenGait CASIA-B pretreatment for ONE frame: tight crop, height->64, centre on the
    upper-body horizontal centroid. Returns 64x64 uint8 {0,255} or None if the mask is empty.
    """
    import cv2
    ys, xs = np.nonzero(mask)
    if ys.size == 0:
        return None
    y0, y1 = ys.min(), ys.max() + 1
    x0, x1 = xs.min(), xs.max() + 1
    crop = mask[y0:y1, x0:x1]
    h = crop.shape[0]
    if h < 8:
        return None
    scale = H_OUT / float(h)
    w = max(1, int(round(crop.shape[1] * scale)))
    crop = cv2.resize(crop.astype(np.uint8) * 255, (w, H_OUT), interpolation=cv2.INTER_LINEAR)
    top = crop[:H_OUT // 2]
    cx = int(round(np.nonzero(top)[1].mean())) if top.any() else w // 2
    out = np.zeros((H_OUT, W_OUT), np.uint8)
    half = W_OUT // 2
    src0, src1 = max(0, cx - half), min(w, cx + half)
    dst0 = half - (cx - src0)
    out[:, dst0:dst0 + (src1 - src0)] = crop[:, src0:src1]
    return out


def casia_reference(csv_path):
    """POPULATION mid-hip and torso length from the CASIA-B TRAIN split (ids 1..74)."""
    # schema: col0 = "./<sid>-<cond>-<seq>-<angle>/<frame>.jpg", then 17 x (x,y,conf)
    import csv as _csv
    import re as _re
    hips, torsos = [], []
    with open(csv_path) as f:
        for i, row in enumerate(_csv.reader(f)):
            if i == 0 or len(row) < 1 + 17 * 3:
                continue
            m = _re.search(r"(\d{3})-", row[0])
            if not m or int(m.group(1)) > 74:
                continue
            v = np.asarray(row[1:1 + 17 * 3], np.float32).reshape(17, 3)
            mh = 0.5 * (v[11, :2] + v[12, :2])
            ms = 0.5 * (v[5, :2] + v[6, :2])
            t = float(np.linalg.norm(ms - mh))
            if t > 1:
                hips.append(mh)
                torsos.append(t)
            if len(torsos) > 20000:
                break
    return np.median(np.stack(hips), 0), float(np.median(torsos))


def to_casia_domain(kp, ref_hip, ref_torso):
    """kp (T,17,3) native px -> CASIA-B domain. Uses the track's OWN median torso, so it is a
    per-track registration, never a per-clip fitted constant."""
    body = kp[:, :, :2].astype(np.float64)
    mh = 0.5 * (body[:, 11] + body[:, 12])
    ms = 0.5 * (body[:, 5] + body[:, 6])
    torso = np.median(np.linalg.norm(ms - mh, axis=1))
    if not np.isfinite(torso) or torso < 1e-3:
        return None
    s = ref_torso / torso
    out = kp.copy().astype(np.float32)
    out[:, :, :2] = ((body - np.median(mh, 0)) * s + ref_hip).astype(np.float32)
    return out
