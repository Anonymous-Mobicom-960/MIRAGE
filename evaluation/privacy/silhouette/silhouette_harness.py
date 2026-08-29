"""
silhouette_harness -- data + arms + recognizer interface for the SILHOUETTE-channel
re-identification evaluation (new metric 8 in MASTER_EVAL_LEDGER).

WHY THIS EXISTS
---------------
Tier-1 emits `mask.mkv` / `masked_video.mkv`. The per-frame body OUTLINE in those files is a
soft biometric (silhouette gait recognition is the classic CASIA-B task), and it crosses the
tier boundary on every clip. `mirage_tier1.mask_mitigate()` reduces that channel (temporal
running-max + contour simplification, then OR-ed with the current frame so coverage can only
grow), but it shipped as INFORMATION REDUCTION with NO adversary evaluation -- unlike the pose
anonymizer, which has the full TM1/TM2/TM3 ladder. This harness is the missing evaluation.

TWO ARMS, one adversary:
  raw        -- silhouettes as the segmenter produced them
  mitigated  -- the SAME silhouettes through the REAL shipped `mask_mitigate()` at the REAL
                deployed parameters (config.MASK_TEMPORAL_WIN=5, MASK_SIMPLIFY_EPS=0.01,
                union-with-current). The function is IMPORTED from the deployed file, never
                re-implemented, so this measures what actually ships.

HONESTY RULE (enforced in code, see require_real_adversary):
a number only counts as a re-ID result when BOTH a real silhouette dataset AND real published
recognizer weights are present. The built-in `PlumbingEmbedder` exists ONLY to prove the
loader -> arms -> protocol -> metrics path is correct on synthetic data; it is not a gait
recognizer and the runner refuses to write a report labelled as a re-ID result from it.
"""
import os
import sys
import glob
import pickle

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
REID = os.path.dirname(HERE)                          # evaluation/privacy/
LAB = os.path.dirname(os.path.dirname(HERE))          # evaluation/
REPO = os.path.dirname(LAB)                            # repository root
EDGE = os.path.join(REPO, "tier1", "src", "edge_runner_pi5")
DATA = os.path.join(HERE, "data")                               # gitignored

sys.path.insert(0, os.path.join(os.path.dirname(HERE), "gait"))   # protocol.py, casia_loader.py

# The DEPLOYED mitigation + its DEPLOYED parameters, imported from the shipped edge code.
# (Same import pattern as the repo-root test_pose_anon_edge.py.)
sys.path.insert(0, EDGE)
import mirage_tier1 as ST                                       # noqa: E402
import config as EDGE_CFG                                       # noqa: E402

MASK_WIN = int(getattr(EDGE_CFG, "MASK_TEMPORAL_WIN", 5))
MASK_EPS = float(getattr(EDGE_CFG, "MASK_SIMPLIFY_EPS", 0.01))

# CASIA-B condition -> protocol status code (protocol.py: nm=0, bg=1, cl=2)
STATUS = {"nm": 0, "bg": 1, "cl": 2}


# --------------------------------------------------------------------------- data
def load_casia_b_silhouettes(root, subjects=None, max_frames=0):
    """CASIA-B silhouettes -> {(subject:int, status:int, seq:int, angle:int): (T,H,W) uint8 0/1}.

    Accepts either layout:
      A) the ORIGINAL release tree  <root>/<sid>/<cond>-<seq>/<angle>/*.png
         (e.g. 001/nm-01/090/001-nm-01-090-042.png), silhouettes at their native crop size;
      B) the OpenGait-preprocessed tree <root>/<sid>/<cond>-<seq>/<angle>/<angle>.pkl
         holding a (T,64,44) uint8 array.

    Layout A is preferred: the deployed mitigation runs at NATIVE resolution BEFORE any
    recognizer normalisation, so applying it to pre-normalised 64x44 crops (layout B) is a
    different operation than the one that ships. The runner records which layout was used.
    """
    out = {}
    if not os.path.isdir(root):
        raise FileNotFoundError(root)
    for sid_dir in sorted(os.listdir(root)):
        if not sid_dir.isdigit():
            continue
        sid = int(sid_dir)
        if subjects is not None and sid not in subjects:
            continue
        for cond_dir in sorted(os.listdir(os.path.join(root, sid_dir))):
            if "-" not in cond_dir:
                continue
            cond, _, seq = cond_dir.partition("-")
            if cond not in STATUS or not seq.isdigit():
                continue
            cpath = os.path.join(root, sid_dir, cond_dir)
            for ang_dir in sorted(os.listdir(cpath)):
                if not ang_dir.isdigit():
                    continue
                apath = os.path.join(cpath, ang_dir)
                pkls = glob.glob(os.path.join(apath, "*.pkl"))
                if pkls:
                    with open(pkls[0], "rb") as f:
                        arr = np.asarray(pickle.load(f))
                else:
                    import cv2
                    pngs = sorted(glob.glob(os.path.join(apath, "*.png")))
                    if not pngs:
                        continue
                    if max_frames:
                        pngs = pngs[:max_frames]
                    frames = [cv2.imread(p, cv2.IMREAD_GRAYSCALE) for p in pngs]
                    frames = [f for f in frames if f is not None]
                    if not frames:
                        continue
                    h = max(f.shape[0] for f in frames); w = max(f.shape[1] for f in frames)
                    arr = np.stack([cv2.copyMakeBorder(f, 0, h - f.shape[0], 0, w - f.shape[1],
                                                       cv2.BORDER_CONSTANT, value=0)
                                    for f in frames])
                if arr.ndim != 3 or arr.shape[0] == 0:
                    continue
                if max_frames:
                    arr = arr[:max_frames]
                # kept AS ON DISK (uint8 0..255, anti-aliased edges and all). The recogniser was
                # trained on exactly these values (/255), so binarising here would silently cost
                # baseline accuracy and confound it with the mitigation. build_arms() derives the
                # binary control and the mitigated arm from this one source.
                out[(sid, STATUS[cond], int(seq), int(ang_dir))] = arr.astype(np.uint8)
    return out


def synth_silhouettes(n_subjects=20, angles=(0, 54, 90, 126, 180), T=40, hw=(128, 88), seed=0):
    """SYNTHETIC identity-bearing silhouettes, for PLUMBING SELF-TEST ONLY.

    Each subject gets fixed body proportions (width/height/stride) plus per-sequence noise, so
    a trivial shape embedder can separate them and the protocol code can be exercised. These
    are NOT real gait data and any number produced from them is a self-test, never a result.
    """
    import cv2
    rng = np.random.RandomState(seed)
    H, W = hw
    persons = {s: dict(torso_w=rng.uniform(0.16, 0.30), height=rng.uniform(0.62, 0.86),
                       stride=rng.uniform(0.10, 0.24), head=rng.uniform(0.05, 0.09))
               for s in range(1, n_subjects + 1)}
    out = {}
    for sid, p in persons.items():
        for status, cond_seqs in ((0, (1, 2, 3, 4, 5, 6)), (1, (1, 2)), (2, (1, 2))):
            for seq in cond_seqs:
                for ang in angles:
                    r = np.random.RandomState((sid * 977 + status * 91 + seq * 13 + ang) & 0x7fffffff)
                    view = 0.55 + 0.45 * abs(np.cos(np.deg2rad(ang)))     # width foreshortening
                    frames = np.zeros((T, H, W), np.uint8)
                    for t in range(T):
                        ph = 2 * np.pi * t / 14.0
                        cx = W // 2
                        bh = int(p["height"] * H)
                        bw = max(3, int(p["torso_w"] * W * view * (1 + 0.03 * r.randn())))
                        top = H - bh
                        cv2.ellipse(frames[t], (cx, top + bh // 2), (bw // 2, bh // 2),
                                    0, 0, 360, 1, -1)
                        hr = max(2, int(p["head"] * H))
                        cv2.circle(frames[t], (cx, max(hr, top - hr // 2)), hr, 1, -1)
                        sw = int(p["stride"] * W * view * np.sin(ph))
                        cv2.line(frames[t], (cx, H - bh // 3), (cx + sw, H - 1), 1,
                                 max(2, bw // 5))
                        cv2.line(frames[t], (cx, H - bh // 3), (cx - sw, H - 1), 1,
                                 max(2, bw // 5))
                    out[(sid, status, seq, ang)] = frames
    return out


# --------------------------------------------------------------------------- arms
def apply_mask_mitigation(seq, win=MASK_WIN, eps=MASK_EPS):
    """Run ONE silhouette sequence (T,H,W) uint8 {0,1} through the REAL deployed
    `mirage_tier1.mask_mitigate()`, reproducing mirage_tier1.main()'s loop EXACTLY:
    the history holds the PRE-mitigation mask (no feedback growth), the window is
    MASK_TEMPORAL_WIN, and the result is OR-ed with the current frame.

    NOTE ON FIDELITY: on device this runs AFTER the EDGE_EXPAND dilate and at native
    resolution. CASIA-B silhouettes arrive already cropped (and, in the OpenGait layout,
    already resized to 64x44), so the contour-simplification epsilon -- a fraction of each
    contour's perimeter -- acts on a different pixel scale. That is a documented caveat of
    using CASIA-B as the stand-in for our own masks, not a change to the shipped function.
    """
    # DISPLACEMENT-FIELD STATE (2026-07-25). mirage_tier1.main() sets a per-CLIP seed and advances
    # the phase every emitted frame; without mirroring that here the lab would measure a field that
    # is CONSTANT over the sequence and identical across sequences - i.e. only the static half of
    # the mitigation, and a version no device ever runs. The seed is per-SEQUENCE (deterministic
    # from the sequence index is NOT used - a fixed lab seed keeps the arm reproducible while still
    # differing from the shipped CSPRNG draw, which is a measurement convenience, not the shipped
    # behaviour; on device it comes from secrets).
    _step = float(getattr(EDGE_CFG, "MASK_DISPLACE_PHASE_STEP", 0.35))
    EDGE_CFG._MASK_DISPLACE_SEED = 12345
    hist, out = [], np.empty_like(seq)
    for t in range(seq.shape[0]):
        cur = np.ascontiguousarray(seq[t].astype(np.uint8))
        hist.append(cur)
        if len(hist) > max(1, int(win)):
            hist.pop(0)
        EDGE_CFG._MASK_DISPLACE_PHASE = _step * t
        out[t] = ST.mask_mitigate(hist, cur, float(eps))
    return out


_MISS = object()


def _cfg_apply(cfg):
    prev = {}
    for k, v in cfg.items():
        prev[k] = getattr(EDGE_CFG, k, _MISS)
        setattr(EDGE_CFG, k, v)
    return prev


def _cfg_restore(prev):
    for k, v in prev.items():
        if v is _MISS:
            if hasattr(EDGE_CFG, k):
                delattr(EDGE_CFG, k)
        else:
            setattr(EDGE_CFG, k, v)


def _mit_worker(args):
    """Worker body for mitigate_batch(). Applies this arm's config overrides, mitigates ONE
    sequence, puts the config back. `mask_mitigate` is a pure function of (frames, config) with a
    pinned seed, so this is bit-identical to the serial path - asserted by --check-parallel."""
    key, seq, win, cfg = args
    prev = _cfg_apply(cfg)
    try:
        return key, apply_mask_mitigation(seq, win=win)
    finally:
        _cfg_restore(prev)


def mitigate_batch(base, win, cfg=None, pool=None):
    """{key: (T,H,W) uint8 {0,1}} -> the same, through the REAL mask_mitigate() at `cfg`.

    Mitigation is single-threaded CPU and, on a full CASIA-B pass, costs 4-9 minutes per ARM
    against ~3 minutes of GPU embedding - so a 25-arm sweep is CPU-bound by a wide margin. Fanning
    the per-sequence work out over processes turns a 5-hour pass into a ~2-hour one. Sequences are
    independent (the displacement seed is pinned per sequence, the phase is a function of the
    frame index), so this changes no number; `run_silhouette_reid.py --check-parallel` asserts
    bit-equality against the serial path before a pass is trusted."""
    cfg = cfg or {}
    if pool is None:
        prev = _cfg_apply(cfg)
        try:
            return {k: apply_mask_mitigation(v, win=win) for k, v in base.items()}
        finally:
            _cfg_restore(prev)
    items = [(k, v, win, cfg) for k, v in base.items()]
    return dict(pool.map(_mit_worker, items, chunksize=4))


def _as255(seq):
    """-> uint8 0..255, preserving anti-aliased edges when the source has them."""
    s = np.asarray(seq)
    return (s.astype(np.uint8) * 255) if s.max() <= 1 else s.astype(np.uint8)


def _asbin(seq):
    """-> uint8 {0,1}. mask_mitigate operates on binary masks, as it does on device."""
    s = np.asarray(seq)
    return ((s > 127) if s.max() > 1 else (s > 0)).astype(np.uint8)


def build_arms(tracklets, arms=("raw", "raw_bin", "mitigated")):
    """{arm_name: {key: (T,H,W) uint8 0..255}} - all three derived from ONE source, so each
    comparison isolates exactly one variable:

      raw        the silhouettes as released (anti-aliased). ONLY this arm is comparable to the
                 PUBLISHED baseline, because it is the domain GaitBase was trained on.
      raw_bin    raw, hard-binarised at 127. This - not `raw` - is the CONTROL for the
                 mitigation, because mask_mitigate() consumes a binary mask. Comparing
                 `mitigated` against `raw` instead would blame binarisation on the mitigation.
      mitigated  raw_bin through the REAL shipped mirage_tier1.mask_mitigate() at deployed params.

    So: raw -> raw_bin measures the cost of binarising; raw_bin -> mitigated is the shipped
    mitigation's actual effect on an attacker."""
    built = {}
    for a in arms:
        if a == "raw":
            built[a] = {k: _as255(v) for k, v in tracklets.items()}
        elif a == "raw_bin":
            built[a] = {k: _asbin(v) * 255 for k, v in tracklets.items()}
        elif a == "mitigated":
            built[a] = {k: apply_mask_mitigation(_asbin(v)) * 255 for k, v in tracklets.items()}
        else:
            raise ValueError(f"unknown arm {a!r}")
    return built


def coverage_accumulate(raw, mitigated, acc=None, contour_budget=20000):
    """§2 + shape-perturbation statistics, accumulated across streamed batches. NOT re-ID numbers.

    The §2 superset check and area/IoU run on EVERY frame (they are cheap numpy and the
    guarantee is absolute). Contour-vertex counting calls findContours twice per frame, so it
    is capped at `contour_budget` frames - a sample is enough for a distribution, and the
    sample size is reported so nothing is silently truncated."""
    import cv2
    a0 = acc or {"frames": 0, "viol": 0, "areas": [], "ious": [],
                 "verts_r": [], "verts_m": [], "contour_frames": 0}
    for k in raw:
        # Vectorised over the whole (T,H,W) tracklet: identical arithmetic to the per-frame loop
        # this replaces, but a 25-arm pass spends minutes per arm in here otherwise.
        r, m = raw[k] > 0, mitigated[k] > 0
        T = r.shape[0]
        a0["viol"] += int(np.count_nonzero(r & ~m))        # raw px NOT covered by mitigated
        a0["frames"] += T
        rf, mf = r.reshape(T, -1), m.reshape(T, -1)
        ar = rf.sum(1).astype(np.int64)
        am = mf.sum(1).astype(np.int64)
        inter = (rf & mf).sum(1).astype(np.int64)
        union = (rf | mf).sum(1).astype(np.int64)
        nz = ar > 0
        a0["areas"].extend((am[nz] / ar[nz]).tolist())
        a0["ious"].extend((inter[nz] / np.maximum(1, union[nz])).tolist())
        for t in range(T):
            if a0["contour_frames"] >= contour_budget:
                break
            a0["contour_frames"] += 1
            for src, key in ((r[t], "verts_r"), (m[t], "verts_m")):
                cs, _ = cv2.findContours(src.astype(np.uint8), cv2.RETR_EXTERNAL,
                                         cv2.CHAIN_APPROX_SIMPLE)
                a0[key].append(sum(len(c) for c in cs))
    return a0


def coverage_finalize(a0):
    f = lambda v: (dict(mean=float(np.mean(v)), p50=float(np.median(v)),
                        p95=float(np.percentile(v, 95))) if len(v) else {})
    return {"frames": a0["frames"], "superset_violations_px": int(a0["viol"]),
            "contour_sample_frames": a0["contour_frames"],
            "area_ratio_mitigated_over_raw": f(a0["areas"]),
            "iou_rawbin_vs_mitigated": f(a0["ious"]),
            "contour_vertices_raw": f(a0["verts_r"]),
            "contour_vertices_mitigated": f(a0["verts_m"]),
            "params": {"MASK_TEMPORAL_WIN": MASK_WIN, "MASK_SIMPLIFY_EPS": MASK_EPS}}


def coverage_report(raw, mitigated):
    """One-shot convenience wrapper (used by measure_local_masks.py)."""
    return coverage_finalize(coverage_accumulate(raw, mitigated, contour_budget=10 ** 9))


# --------------------------------------------------------------------------- adversary
class PlumbingEmbedder:
    """NOT A GAIT RECOGNISER. A deterministic shape descriptor (temporal mean of a
    downsampled silhouette + simple gait statistics) whose ONLY purpose is to prove that
    loader -> arms -> protocol -> metrics runs end to end on synthetic data. Any number it
    produces is a SELF-TEST, never a privacy result -- see require_real_adversary()."""
    is_real_adversary = False
    name = "plumbing-shape-descriptor (SELF-TEST ONLY)"

    def __init__(self, grid=(16, 12)):
        self.grid = grid

    def embed(self, tracklets):
        import cv2
        gh, gw = self.grid
        out = {}
        for k, seq in tracklets.items():
            small = np.stack([cv2.resize(f.astype(np.float32), (gw, gh)) for f in seq])
            mean = small.mean(0).ravel()
            std = small.std(0).ravel()
            widths = np.array([f.sum(axis=1).max() for f in seq], np.float32)
            heights = np.array([f.sum(axis=0).max() for f in seq], np.float32)
            extra = np.array([widths.mean(), widths.std(), heights.mean(), heights.std()],
                             np.float32)
            v = np.concatenate([mean, std, extra])
            out[k] = v / (np.linalg.norm(v) + 1e-9)
        return out


class OpenGaitRecognizer:
    """GaitSet-class silhouette recogniser backed by a PUBLISHED OpenGait checkpoint.

    PROVENANCE (verify before quoting any number):
      repo    https://github.com/ShiqiYu/OpenGait  (model code, Apache-2.0)
      weights https://huggingface.co/opengait/OpenGait  (official model zoo, ungated,
              downloadable without login) -- for CASIA-B the published checkpoint is
              `CASIA-B/Baseline/GaitBase_DA/checkpoints/GaitBase_DA-60000.pt`.
              The zoo's GaitSet checkpoints are trained on CASIA-E / Gait3D-Parsing, NOT
              CASIA-B; using those on CASIA-B silhouettes would be an out-of-domain,
              unvalidated adversary and MUST NOT be substituted.
    Requires OpenGait's model code on PYTHONPATH (vendor it under reid_eval/vendor/OpenGait,
    mirroring how GaitGraph is vendored for the pose adversary).
    """
    is_real_adversary = True

    PUBLISHED = {"NM": 97.6, "BG": 94.0, "CL": 77.4}     # docs/1.model_zoo.md, identical-view excluded

    def __init__(self, ckpt, opengait_root=None, device=None, seq_limit=720):
        import torch
        self.ckpt_path = ckpt
        self.name = f"OpenGait GaitBase_DA CASIA-B ({os.path.basename(ckpt)})"
        self.seq_limit = int(seq_limit)                  # config frames_all_limit: 720
        root = opengait_root or os.path.join(REID, "vendor", "OpenGait")
        if not os.path.isdir(root):
            raise FileNotFoundError(
                f"OpenGait model code not found at {root}. Clone it:\n"
                f"  git clone https://github.com/ShiqiYu/OpenGait {root}")
        if not os.path.isfile(ckpt):
            raise FileNotFoundError(
                f"checkpoint not found: {ckpt}\n"
                f"  huggingface-cli download opengait/OpenGait "
                f"CASIA-B/Baseline/GaitBase_DA/checkpoints/GaitBase_DA-60000.pt "
                f"--local-dir {DATA}")
        # OpenGait's modules do bare `from utils import ...` / `from modeling import ...`, i.e.
        # they expect to be run from INSIDE opengait/. Put both the repo root (for the
        # `opengait.` package path) and opengait/ itself (for those bare imports) on sys.path.
        sys.path.insert(0, root)
        sys.path.insert(0, os.path.join(root, "opengait"))
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.model = self._build(ckpt)

    def _build(self, ckpt):
        """Rebuild GaitBase (`Baseline`) from configs/gaitbase/gaitbase_da_casiab.yaml and load
        the published weights. The submodules come from OpenGait's own code; only BaseModel's
        training/distributed scaffolding is bypassed, because inference needs none of it and
        instantiating it would demand a full experiment config + launcher."""
        import torch
        import torch.nn as nn
        from opengait.modeling.modules import (SetBlockWrapper, HorizontalPoolingPyramid,
                                               PackSequenceWrapper, SeparateFCs, SeparateBNNecks)
        from opengait.modeling.backbones.resnet import ResNet9

        class _GaitBase(nn.Module):
            """Exactly baseline.Baseline.forward()'s inference path (embed = FCs output).
            Every hyper-parameter below is copied from configs/gaitbase/gaitbase_da_casiab.yaml
            (`block` is the STRING the yaml uses - ResNet9 resolves it through block_map)."""
            def __init__(self):
                super().__init__()
                self.Backbone = SetBlockWrapper(ResNet9(
                    block="BasicBlock", channels=[64, 128, 256, 512], layers=[1, 1, 1, 1],
                    strides=[1, 2, 2, 1], maxpool=False))
                self.FCs = SeparateFCs(in_channels=512, out_channels=256, parts_num=16)
                self.BNNecks = SeparateBNNecks(class_num=74, in_channels=256, parts_num=16)
                self.TP = PackSequenceWrapper(torch.max)
                self.HPP = HorizontalPoolingPyramid(bin_num=[16])

            def forward(self, sils):                      # sils: [n, s, h, w]
                x = sils.unsqueeze(1)                     # -> [n, c=1, s, h, w]
                outs = self.Backbone(x)
                outs = self.TP(outs, None, options={"dim": 2})[0]
                feat = self.HPP(outs)                     # [n, c, p]
                return self.FCs(feat)                     # embed_1 == inference embedding

        m = _GaitBase()
        sd = torch.load(ckpt, map_location="cpu", weights_only=False)
        sd = sd.get("model", sd)
        missing, unexpected = m.load_state_dict(sd, strict=False)
        # BNNecks is only used for the training softmax head; its weights load but are unused.
        real_missing = [k for k in missing if not k.startswith("BNNecks.")]
        if real_missing:
            raise RuntimeError(f"checkpoint does not match the model: missing {real_missing[:6]}")
        if unexpected:
            print(f"[silh] note: {len(unexpected)} unused checkpoint tensors (training heads)")
        print(f"[silh] loaded {len(sd)} tensors from {os.path.basename(ckpt)} "
              f"-> GaitBase on {self.device}")
        return m.to(self.device).eval()

    @staticmethod
    def _cut(seq):
        """BaseSilCuttingTransform: 64x64 -> 64x44 (drop `w//64*10` columns each side) and /255.
        Verbatim from opengait/data/transform.py; the CASIA-B configs use it at train AND test."""
        cutting = int(seq.shape[-1] // 64) * 10
        if cutting:
            seq = seq[..., cutting:-cutting]
        return seq.astype(np.float32) / 255.0

    def embed(self, tracklets):
        import torch
        out = {}
        with torch.no_grad():
            for k, seq in tracklets.items():
                s = np.asarray(seq)
                if s.dtype != np.uint8 or s.max() <= 1:
                    s = (s > 0).astype(np.uint8) * 255      # binary arm -> the model's 0..255 domain
                if s.shape[0] > self.seq_limit:             # config frames_all_limit
                    s = s[:self.seq_limit]
                x = torch.from_numpy(self._cut(s))[None].to(self.device)     # [1, s, h, w]
                e = self.model(x)[0]                        # [c=256, p=16]
                out[k] = e.transpose(0, 1).contiguous().cpu().numpy()        # [p, c] for the metric
        return out


def opengait_rank1(embeddings, angles=None):
    """Canonical CASIA-B rank-1 using OPENGAIT'S OWN DISTANCE, so the raw arm is directly
    comparable to the published 97.6 / 94.0 / 77.4.

    Same gallery/probe/view rules as protocol.canonical_casia_b_rank1 (gallery = NM#1-4, probes
    = NM#5-6 / BG#1-2 / CL#1-2, identical view excluded) - but the distance is OpenGait's
    `cuda_dist(metric='euc')`: the MEAN over the 16 horizontal parts of the per-part euclidean
    distance (opengait/evaluation/metric.py L8-27), NOT the euclidean of the flattened vector.
    Embeddings are [parts, channels] as returned by OpenGaitRecognizer.embed().
    """
    angles = list(angles) if angles else list(range(0, 181, 18))
    gal = {k: v for k, v in embeddings.items() if k[1] == 0 and k[2] <= 4}
    probes = [{k: v for k, v in embeddings.items() if k[1] == 0 and k[2] >= 5},
              {k: v for k, v in embeddings.items() if k[1] == 1},
              {k: v for k, v in embeddings.items() if k[1] == 2}]
    n = len(angles)
    correct = np.zeros((3, n, n)); total = np.zeros((3, n, n))
    for gi, ga in enumerate(angles):
        gk = [k for k in gal if k[3] == ga]
        if not gk:
            continue
        G = np.stack([gal[k] for k in gk])                       # [N, p, c]
        gids = np.array([k[0] for k in gk])
        for pn, probe in enumerate(probes):
            for pk, pv in probe.items():
                if pk[3] not in angles:
                    continue
                d = np.linalg.norm(G - pv[None], axis=2).mean(axis=1)   # mean over parts
                if gids[int(np.argmin(d))] == pk[0]:
                    correct[pn, gi, angles.index(pk[3])] += 1
                total[pn, gi, angles.index(pk[3])] += 1
    acc = np.divide(correct, total, out=np.zeros_like(correct), where=total > 0)
    for i in range(3):
        acc[i] -= np.diag(np.diag(acc[i]))                        # exclude identical view
    per = np.sum(acc, 1) / float(n - 1)
    m = np.mean(per, 1)
    return {"NM#5-6": float(m[0]), "BG#1-2": float(m[1]), "CL#1-2": float(m[2]),
            "mean": float(np.mean(m))}


ANGLES_11 = list(range(0, 181, 18))


def require_real_adversary(embedder, allow_selftest=False):
    """Gate every reported re-ID number. The lab's own rule (MASTER_EVAL_LEDGER header):
    a value enters a data cell only if it was actually measured -- and a shape descriptor
    standing in for a gait recogniser is not a measurement of anything an attacker can do."""
    if getattr(embedder, "is_real_adversary", False):
        return True
    if allow_selftest:
        return False
    raise RuntimeError(
        f"refusing to report re-ID numbers from {getattr(embedder, 'name', embedder)!r}: it is "
        "not a validated gait recogniser. Provide published OpenGait CASIA-B weights (see "
        "OpenGaitRecognizer docstring), or run with --selftest to exercise the plumbing only.")
