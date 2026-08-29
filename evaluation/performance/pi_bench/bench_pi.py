"""bench_pi.py -- YOLO11n-seg pure-inference latency on the Pi 5: ONNX (onnxruntime
CPU-EP) vs NCNN (fp16), matched threads. Random input (latency is value-independent).
Reports mean/median/p90/min ms over N iters after warmup, and the Pi throttle/freq
state before+after so we can flag power-limited (invalid) runs.

Usage: python bench_pi.py <models_dir> [threads] [iters]
Expects, under <models_dir>:
  yolo11n-seg_{192,256}.onnx
  yolo11n-seg_ncnn_{192,256}/model.ncnn.{param,bin}
"""
import os, sys, time, subprocess
import numpy as np

MDIR = sys.argv[1] if len(sys.argv) > 1 else "."
THREADS = int(sys.argv[2]) if len(sys.argv) > 2 else 4
ITERS = int(sys.argv[3]) if len(sys.argv) > 3 else 50
WARMUP = 8
SIZES = [192, 256]


def throttle():
    try:
        t = subprocess.check_output(["vcgencmd", "get_throttled"]).decode().strip()
        f = subprocess.check_output(["vcgencmd", "measure_clock", "arm"]).decode().strip()
        tp = subprocess.check_output(["vcgencmd", "measure_temp"]).decode().strip()
        mhz = int(f.split("=")[1]) / 1e6
        return f"{t}  arm={mhz:.0f}MHz  {tp}"
    except Exception as e:
        return f"(vcgencmd n/a: {e})"


def stats(ts):
    a = np.array(ts)
    return dict(mean=a.mean(), median=np.median(a), p90=np.percentile(a, 90), min=a.min(), n=len(a))


def bench_onnx(path, sz):
    import onnxruntime as ort
    so = ort.SessionOptions()
    so.intra_op_num_threads = THREADS
    so.inter_op_num_threads = 1
    so.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
    sess = ort.InferenceSession(path, sess_options=so, providers=["CPUExecutionProvider"])
    iname = sess.get_inputs()[0].name
    x = np.random.rand(1, 3, sz, sz).astype(np.float32)
    for _ in range(WARMUP):
        sess.run(None, {iname: x})
    ts = []
    for _ in range(ITERS):
        t = time.perf_counter(); sess.run(None, {iname: x}); ts.append((time.perf_counter() - t) * 1000)
    return stats(ts)


def bench_ncnn(pdir, sz):
    import ncnn
    net = ncnn.Net()
    net.opt.num_threads = THREADS
    net.opt.use_vulkan_compute = False
    net.opt.use_fp16_packed = True
    net.opt.use_fp16_storage = True
    net.opt.use_fp16_arithmetic = True
    net.load_param(os.path.join(pdir, "model.ncnn.param"))
    net.load_model(os.path.join(pdir, "model.ncnn.bin"))
    mat = ncnn.Mat(sz, sz, 3)
    mat.fill(0.5)

    def run():
        ex = net.create_extractor()
        ex.input("in0", mat)
        ex.extract("out0")
        ex.extract("out1")
    for _ in range(WARMUP):
        run()
    ts = []
    for _ in range(ITERS):
        t = time.perf_counter(); run(); ts.append((time.perf_counter() - t) * 1000)
    net.clear()
    return stats(ts)


def main():
    print(f"# Pi5 YOLO11n-seg inference latency | threads={THREADS} iters={ITERS}")
    print(f"# throttle BEFORE: {throttle()}")
    rows = []
    for sz in SIZES:
        variants = [
            ("ONNX-fp32", "onnx", os.path.join(MDIR, f"yolo11n-seg_{sz}.onnx")),
            ("ONNX-int8", "onnx", os.path.join(MDIR, f"yolo11n-seg_{sz}_int8.onnx")),
            ("NCNN-fp16", "ncnn", os.path.join(MDIR, f"yolo11n-seg_ncnn_{sz}")),
        ]
        for label, kind, path in variants:
            exists = os.path.isdir(path) if kind == "ncnn" else os.path.exists(path)
            if not exists:
                continue
            try:
                s = bench_ncnn(path, sz) if kind == "ncnn" else bench_onnx(path, sz)
                rows.append((label, sz, s))
                print(f"  {label} @{sz}: mean {s['mean']:.1f}  median {s['median']:.1f}  p90 {s['p90']:.1f}  min {s['min']:.1f} ms")
            except Exception as e:
                print(f"  {label} @{sz}: ERROR {type(e).__name__}: {e}")
    print(f"# throttle AFTER:  {throttle()}")
    print("\n# SUMMARY (median ms, lower=faster)")
    for sz in SIZES:
        cells = {r: s['median'] for r, z, s in rows if z == sz}
        if cells:
            best = min(cells, key=cells.get)
            print(f"  @{sz}: " + "  ".join(f"{k} {v:.1f}ms" for k, v in cells.items()) + f"   -> fastest: {best}")


if __name__ == "__main__":
    main()
