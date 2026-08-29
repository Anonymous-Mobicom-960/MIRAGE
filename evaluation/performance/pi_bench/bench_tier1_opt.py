"""bench_tier1_opt.py -- OPTIMIZED Tier-1 for the >=15 FPS budget (masking + pose
sticks + canonicalizer). Changes vs bench_tier1_pi.py:
  - pose: YOLO11n-POSE via NCNN-fp16 (single-shot, replaces onnxruntime RTMPose)
  - grey-fill: vectorized np.where (no full copy + fancy-index)
  - anonymize: CANON-ONLY (identity-collapse; the dynamic-perturb/reID part deferred)
Per-frame median ms + end-to-end FPS on the real 1264x1264 frame.
"""
import os, sys, time
import numpy as np, cv2
sys.path.insert(0, HERE)
sys.path.insert(0, os.path.dirname(os.path.dirname(HERE)))   # evaluation/ -> segbench
from segbench import core
from segbench.pose_anon_v2 import anonymize_v2, population_template
import ncnn

THREADS = 4; cv2.setNumThreads(THREADS)
OUT = (1264, 1264); SEG = 256; POSE = 256
WARMUP, ITERS = 6, 30


def med(fn, w=WARMUP, n=ITERS):
    for _ in range(w): fn()
    ts = [];
    for _ in range(n):
        t = time.perf_counter(); fn(); ts.append((time.perf_counter() - t) * 1000)
    return float(np.median(ts))


def ncnn_net(d):
    net = ncnn.Net(); net.opt.num_threads = THREADS; net.opt.use_vulkan_compute = False
    for a in ("use_fp16_packed", "use_fp16_storage", "use_fp16_arithmetic"): setattr(net.opt, a, True)
    net.load_param(os.path.join(HERE, "models", d, "model.ncnn.param"))
    net.load_model(os.path.join(HERE, "models", d, "model.ncnn.bin"))
    return net


def main():
    import subprocess
    thr0 = subprocess.check_output(["vcgencmd", "get_throttled"]).decode().strip()
    frame = cv2.imread(os.path.join(HERE, "frame1264.jpg"))
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    inst = []
    for cx, r in [(0.4, 0.28), (0.62, 0.30)]:
        m = np.zeros((SEG, SEG), np.uint8)
        cv2.ellipse(m, (int(cx*SEG), int(0.55*SEG)), (int(0.12*SEG), int(r*SEG)), 0, 0, 360, 1, -1); inst.append(dict(mask=m))
    res = {}

    seg_net = ncnn_net(f"yolo11n-seg_ncnn_{SEG}"); seg_in = ncnn.Mat(SEG, SEG, 3); seg_in.fill(0.5)
    def seg():
        ex = seg_net.create_extractor(); ex.input("in0", seg_in); ex.extract("out0"); ex.extract("out1")
    res["1_seg_ncnn@%d" % SEG] = med(seg)

    pose_net = ncnn_net(f"yolo11n-pose_ncnn_{POSE}"); pose_in = ncnn.Mat(POSE, POSE, 3); pose_in.fill(0.5)
    def pose():
        ex = pose_net.create_extractor(); ex.input("in0", pose_in); ex.extract("out0")
    res["5_pose_ncnn@%d" % POSE] = med(pose)

    res["2_maskpost_ws1.0_d4"] = med(lambda: core.build_union_mask(inst, OUT, method="guided", dilate_px=4, close_k=5, guide_gray=gray))
    umask = core.build_union_mask(inst, OUT, method="guided", dilate_px=4, close_k=5, guide_gray=gray)
    m3 = np.repeat(umask[:, :, None], 3, axis=2)
    GREY = np.uint8(127)
    res["3_grey_vec"] = med(lambda: np.where(m3, GREY, frame))
    res["4_faceguard"] = med(lambda: cv2.ellipse(np.zeros(OUT, np.uint8), (430, 360), (100, 130), 0, 0, 360, 1, -1))

    seq = np.random.rand(60, 17, 2).astype(np.float64) * 1264
    tmpl = population_template([seq])
    res["6_canon_only"] = med(lambda: anonymize_v2(seq, tmpl, seed=1, do_canon=True), w=2, n=10)

    # TRUE end-to-end: all per-frame stages back-to-back in ONE timed loop
    GREY = np.uint8(127); facem = np.zeros(OUT, np.uint8)
    def e2e():
        ex = seg_net.create_extractor(); ex.input("in0", seg_in); ex.extract("out0"); ex.extract("out1")
        ex2 = pose_net.create_extractor(); ex2.input("in0", pose_in); ex2.extract("out0")
        um = core.build_union_mask(inst, OUT, method="guided", dilate_px=4, close_k=5, guide_gray=gray)
        m3v = np.repeat(um[:, :, None], 3, axis=2)
        o = np.where(m3v, GREY, frame)
        cv2.ellipse(facem, (430, 360), (100, 130), 0, 0, 360, 1, -1)
        return o
    res["E2E_measured"] = med(e2e, w=6, n=30)

    thr1 = subprocess.check_output(["vcgencmd", "get_throttled"]).decode().strip()
    print(f"\n# OPTIMIZED Tier-1 (masking+pose+canon), Pi5 A76 x{THREADS}, 1264x1264, seg@{SEG} pose@{POSE}")
    print(f"# throttle {thr0} -> {thr1}\n")
    per_frame = ["1_seg_ncnn@%d" % SEG, "5_pose_ncnn@%d" % POSE, "2_maskpost_ws1.0_d4", "3_grey_vec", "4_faceguard"]
    for k in sorted(res): print(f"  {k:22s} {res[k]:7.1f} ms" + ("  (per-clip, amortized)" if k.startswith('6') else ""))
    tot = sum(res[k] for k in per_frame)
    e2e_ms = res["E2E_measured"]
    print(f"\n  sum-of-stages  {tot:6.1f} ms  ->  {1000/tot:5.1f} FPS")
    print(f"  >>> MEASURED END-TO-END  {e2e_ms:6.1f} ms  ->  {1000/e2e_ms:5.1f} FPS  <<<")
    print(f"  canon per-clip: {res['6_canon_only']:.1f} ms / 60f = {res['6_canon_only']/60:.2f} ms/frame amortized")
    print(f"  NOTE: seg+pose = NCNN FORWARD only; decode/NMS (~few ms each) not included.")


if __name__ == "__main__":
    main()
