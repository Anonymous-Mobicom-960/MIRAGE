#!/usr/bin/env python3
"""Draw RAW vs e2-anonymised sticks side by side over the emitted mask, and measure the
G.9 distortion signature (per-joint displacement vs the raw pose, normalised by torso)."""
import io, json, os, sys
import cv2, numpy as np
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)),
                                "..", "..", "..", "tier2_cloud", "scripts"))
import tier1_viz as VIZ
W = os.path.dirname(os.path.abspath(__file__))
raw = json.load(io.open(os.path.join(W, "emit_raw", "pose.json"), encoding="utf-8"))["frames"]
ann = json.load(io.open(os.path.join(W, "emit", "pose.json"), encoding="utf-8"))["frames"]
N = min(len(raw), len(ann))
SZ = 1264
# --- G.9 metric: displacement per joint group, normalised by the RAW torso length -------------
J = {"wrist": (9, 10), "elbow": (7, 8), "knee": (13, 14), "ankle": (15, 16), "shoulder": (5, 6), "hip": (11, 12)}
d = {k: [] for k in J}; torso = []
for f in range(N):
    if not raw[f] or not ann[f]: continue
    r = np.asarray(raw[f][0]["kp"], float); a = np.asarray(ann[f][0]["kp"], float)
    rs = np.asarray(raw[f][0]["score"], float); as_ = np.asarray(ann[f][0]["score"], float)
    sh = 0.5*(r[5]+r[6]); hp = 0.5*(r[11]+r[12])
    t = float(np.linalg.norm(sh-hp))
    if t < 1: continue
    torso.append(t)
    for k, js in J.items():
        for j in js:
            if rs[j] > 0.5 and as_[j] > 0.5:
                d[k].append(float(np.linalg.norm(r[j]-a[j])))
T = float(np.mean(torso))
res = {"torso_px_mean": round(T,1), "frames": N,
       "displacement_px_mean": {k: round(float(np.mean(v)),1) if v else None for k,v in d.items()},
       "displacement_pct_of_torso": {k: round(100*float(np.mean(v))/T,1) if v else None for k,v in d.items()},
       "all_joint_mean_px": round(float(np.mean([x for v in d.values() for x in v])),1)}
json.dump(res, open(os.path.join(W, "STICKS_G9.json"), "w"), indent=1)
print(json.dumps(res, indent=1))
# --- visual: RAW | e2, sticks on black --------------------------------------------------------
blank = [np.zeros((SZ, SZ, 3), np.uint8)] * N
VIZ.make_pose_sticks(blank, raw[:N], os.path.join(W, "_sticks_raw.mp4"), 10.0, on_black=True)
VIZ.make_pose_sticks(blank, ann[:N], os.path.join(W, "_sticks_e2.mp4"), 10.0, on_black=True)
cr = cv2.VideoCapture(os.path.join(W, "_sticks_raw.mp4")); ca = cv2.VideoCapture(os.path.join(W, "_sticks_e2.mp4"))
cm = cv2.VideoCapture(os.path.join(W, "out_t1", "mask.mp4"))
picks = [int(N*p) for p in (0.1, 0.35, 0.6, 0.85)]
tiles = []
for i in range(N):
    okr, fr = cr.read(); oka, fa = ca.read(); okm, fm = cm.read()
    if not (okr and oka): break
    if i in picks:
        if okm:
            g = cv2.cvtColor(fm, cv2.COLOR_BGR2GRAY)
            for im in (fr, fa): im[g > 127] = np.maximum(im[g > 127], (40, 40, 40))
        cv2.putText(fr, f"RAW f{i}", (20,50), 0, 1.4, (255,255,255), 3)
        cv2.putText(fa, f"e2 f{i}", (20,50), 0, 1.4, (255,255,255), 3)
        tiles.append(np.hstack([fr, fa]))
cr.release(); ca.release(); cm.release()
grid = np.vstack(tiles)
grid = cv2.resize(grid, (grid.shape[1]//3, grid.shape[0]//3))
cv2.imwrite(os.path.join(W, "STICKS_RAW_vs_E2.png"), grid)
print("wrote STICKS_RAW_vs_E2.png", grid.shape)
