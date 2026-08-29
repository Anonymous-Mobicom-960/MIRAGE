"""bench_tier1_pi.py -- FULL Tier-1 pipeline latency on the Pi 5, stage by stage,
on a real 1264x1264 frame. Reports per-stage median ms + end-to-end + FPS.

Stages (the production Tier-1 winner config): seg (NCNN yolo11n-seg) -> mask
post-proc (guided fast-filter + morph + d4 @1264) -> grey-fill -> FaceGuard
(keypoint ellipse-fill; production reuses RTMPose head kpts, no separate model)
-> RTMPose-t pose -> anti-reID anonymise. Tracker is O(few centroids) ~ 0.
Random/synthetic masks where content doesn't affect latency (mask post-proc,
grey, faceguard are content-independent for a given resolution).
"""
import os, sys, time
import numpy as np
import cv2
HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
sys.path.insert(0, os.path.dirname(os.path.dirname(HERE)))   # evaluation/ -> segbench
from segbench import core
from segbench.pose_anon_v2 import anonymize_v2, population_template

THREADS = 4
cv2.setNumThreads(THREADS)
FRAME = os.path.join(HERE, "frame1264.jpg")
NCNN_DIR = os.path.join(HERE, "models", "yolo11n-seg_ncnn_256")
SEG = 256
OUT = (1264, 1264)
WARMUP, ITERS = 5, 30


def med(fn, warmup=WARMUP, iters=ITERS):
    for _ in range(warmup):
        fn()
    ts = []
    for _ in range(iters):
        t = time.perf_counter(); fn(); ts.append((time.perf_counter() - t) * 1000)
    return float(np.median(ts))


def throttle():
    import subprocess
    try:
        t = subprocess.check_output(["vcgencmd", "get_throttled"]).decode().strip()
        f = int(subprocess.check_output(["vcgencmd", "measure_clock", "arm"]).decode().split("=")[1]) / 1e6
        tp = subprocess.check_output(["vcgencmd", "measure_temp"]).decode().strip()
        return f"{t} arm={f:.0f}MHz {tp}"
    except Exception:
        return "n/a"


def main():
    frame = cv2.imread(FRAME)
    assert frame is not None and frame.shape[:2] == OUT, frame.shape
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    # two representative person instance masks at seg resolution (content-indep for latency)
    inst = []
    for cx, r in [(0.4, 0.28), (0.62, 0.30)]:
        m = np.zeros((SEG, SEG), np.uint8)
        cv2.ellipse(m, (int(cx * SEG), int(0.55 * SEG)), (int(0.12 * SEG), int(r * SEG)), 0, 0, 360, 1, -1)
        inst.append(dict(mask=m, box=[0, 0, 100, 100], score=0.9))
    boxes = [[300, 250, 620, 1050], [640, 230, 980, 1080]]  # person boxes for pose

    res = {}
    res["throttle_before"] = throttle()

    # 1) SEG -- NCNN forward @256
    import ncnn
    net = ncnn.Net()
    net.opt.num_threads = THREADS
    net.opt.use_vulkan_compute = False
    for a in ("use_fp16_packed", "use_fp16_storage", "use_fp16_arithmetic"):
        setattr(net.opt, a, True)
    net.load_param(os.path.join(NCNN_DIR, "model.ncnn.param"))
    net.load_model(os.path.join(NCNN_DIR, "model.ncnn.bin"))
    seg_in = ncnn.Mat(SEG, SEG, 3); seg_in.fill(0.5)

    def seg():
        ex = net.create_extractor(); ex.input("in0", seg_in)
        ex.extract("out0"); ex.extract("out1")
    res["1_seg_ncnn@256"] = med(seg)

    # 2) MASK post-proc (guided + morph + d4) @1264, work_scale 1.0 and 0.5
    res["2_maskpost_ws1.0_d4"] = med(lambda: core.build_union_mask(
        inst, OUT, method="guided", dilate_px=4, close_k=5, guide_gray=gray, work_scale=1.0))
    res["2_maskpost_ws0.5_d8"] = med(lambda: core.build_union_mask(
        inst, OUT, method="guided", dilate_px=8, close_k=5, guide_gray=gray, work_scale=0.5))
    umask = core.build_union_mask(inst, OUT, method="guided", dilate_px=4, close_k=5, guide_gray=gray)

    # 3) grey-fill
    res["3_grey_fill"] = med(lambda: core.apply_grey(frame, umask))

    # 4) FaceGuard -- ellipse-fill for ~2 faces (production reuses RTMPose head kpts)
    faces = [(430, 360, 70, 90), (770, 340, 70, 90)]

    def faceguard():
        m = np.zeros(OUT, np.uint8)
        for cx, cy, w, h in faces:
            cv2.ellipse(m, (cx, cy), (int(w * 1.45), int(h * 1.45)), 0, 0, 360, 1, -1)
        return m.astype(bool)
    res["4_faceguard_fill"] = med(faceguard)

    # 5) RTMPose-t (pose only, on given boxes -- Tier-1 shares boxes with seg)
    try:
        from rtmlib import Body
        body = Body(mode="lightweight", backend="onnxruntime", device="cpu")
        pm = body.pose_model
        pm(frame, np.array(boxes, dtype=np.float32))  # warmup + download
        res["5_rtmpose_t"] = med(lambda: pm(frame, np.array(boxes, dtype=np.float32)), warmup=3, iters=15)
    except Exception as e:
        res["5_rtmpose_t"] = f"ERR {type(e).__name__}: {str(e)[:80]}"

    # 6) anti-reID anonymise (per-clip) -- tiny
    seq = np.random.rand(60, 17, 2).astype(np.float64) * np.array([1264, 1264])
    tmpl = population_template([seq])
    res["6_anon_v2"] = med(lambda: anonymize_v2(seq, tmpl, seed=1, cadence_amp=0.6,
                                                angle_const_deg=20, angle_drift_deg=15,
                                                asymmetry_sigma=0.12), warmup=2, iters=10)
    res["throttle_after"] = throttle()

    # ---- report ----
    print(f"\n# FULL Tier-1 on Pi5 A76, {THREADS} threads, frame 1264x1264, seg@{SEG}")
    print(f"# throttle before: {res['throttle_before']}\n# throttle after:  {res['throttle_after']}\n")
    order = ["1_seg_ncnn@256", "2_maskpost_ws1.0_d4", "3_grey_fill", "4_faceguard_fill",
             "5_rtmpose_t", "6_anon_v2"]
    numeric = [k for k in order if isinstance(res[k], float)]
    print("STAGE                     median_ms")
    for k in order:
        v = res[k]
        print(f"  {k:24s} {v:8.1f}" if isinstance(v, float) else f"  {k:24s} {v}")
    total = sum(res[k] for k in numeric)
    print(f"\n  {'END-TO-END (ws1.0)':24s} {total:8.1f} ms   -> {1000/total:5.1f} FPS")
    # fast-post variant
    if isinstance(res["2_maskpost_ws0.5_d8"], float):
        fast = total - res["2_maskpost_ws1.0_d4"] + res["2_maskpost_ws0.5_d8"]
        print(f"  {'END-TO-END fast-post':24s} {fast:8.1f} ms   -> {1000/fast:5.1f} FPS "
              f"(mask ws0.5/d8: {res['2_maskpost_ws0.5_d8']:.1f}ms)")
    print("\n# note: seg = NCNN forward only (decode/NMS adds a few ms); FaceGuard = keypoint "
          "ellipse-fill (mediapipe has no py3.13 wheel; production reuses RTMPose head kpts).")


if __name__ == "__main__":
    main()
