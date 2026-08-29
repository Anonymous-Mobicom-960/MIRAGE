"""
adversary -- the GaitGraph ResGCN skeleton-gait recognizer as a FROZEN embedding
extractor, reproducing evaluate.py's preprocessing exactly so its published
CASIA-B baseline is recoverable, and exposing embed(tracklet)->128-d vector as the
seam we later attack with the anonymizer.

Preprocessing (faithful to GaitGraph/src/evaluate.py + datasets/augmentation.py):
  tracklet (T,17,3)
   -> SelectSequenceCenter(L): center L frames                     (L=60)
   -> MultiInput(connect_joint, use_multi_branch): (I, C2, L, 17)
   -> ToTensor
   -> [optional flip-TTA] model(.) -> L2-normalized 128-d embedding

Branch config (num_input I, num_channel C2) is AUTO-DETECTED from the checkpoint
state_dict, so we don't guess single- vs multi-branch.
"""
import os
import sys
import numpy as np
import torch

_VENDOR_SRC = os.path.join(os.path.dirname(__file__), "vendor", "GaitGraph", "src")
if _VENDOR_SRC not in sys.path:
    sys.path.insert(0, _VENDOR_SRC)

import models.ResGCNv1 as ResGCNv1          # noqa: E402  (pure torch)
from datasets.graph import Graph            # noqa: E402


# ---- transforms (reimplemented from datasets/augmentation.py, verbatim logic) ----
def select_sequence_center(data, seq_len):
    """data: (T,V,C) -> (seq_len,V,C) centered. Mirrors SelectSequenceCenter."""
    start = int((data.shape[0] / 2) - (seq_len / 2))
    start = max(0, start)
    return data[start:start + seq_len]


def multi_input(data, connect_joint, enabled):
    """data: (T,V,C) -> (I, C*2, T, V). Verbatim from MultiInput.__call__."""
    data = np.transpose(data, (2, 0, 1))  # (C,T,V)
    if not enabled:
        return data[np.newaxis, ...]       # (1,C,T,V)
    C, T, V = data.shape
    out = np.zeros((3, C * 2, T, V), dtype=np.float32)
    out[0, :C] = data
    for i in range(V):
        out[0, C:, :, i] = data[:, :, i] - data[:, :, 1]
    for i in range(T - 2):
        out[1, :C, i, :] = data[:, i + 1, :] - data[:, i, :]
        out[1, C:, i, :] = data[:, i + 2, :] - data[:, i, :]
    for i in range(len(connect_joint)):
        out[2, :C, :, i] = data[:, :, i] - data[:, :, connect_joint[i]]
    bone_length = 0
    for i in range(C - 1):
        bone_length = bone_length + np.power(out[2, i, :, :], 2)
    bone_length = np.sqrt(bone_length) + 1e-4
    for i in range(C - 1):
        out[2, C, :, :] = np.arccos(out[2, i, :, :] / bone_length)
    return out


def _detect_branch_config(state_dict):
    """Infer (num_input, num_channel) from the checkpoint's input branches."""
    branch_idxs = set()
    num_channel = None
    for k, v in state_dict.items():
        if k.startswith("input_branches."):
            branch_idxs.add(int(k.split(".")[1]))
            if k.endswith("input_branches.%s.bn.weight" % k.split(".")[1]):
                num_channel = int(v.shape[0])
    num_input = (max(branch_idxs) + 1) if branch_idxs else 1
    # fall back: first input branch's leading BatchNorm2d(num_channel)
    if num_channel is None:
        for k, v in state_dict.items():
            if k.endswith(".bn.weight") and k.startswith("input_branches.0"):
                num_channel = int(v.shape[0]); break
    return num_input, num_channel


class GaitGraphAdversary:
    def __init__(self, weights_path, network_name="resgcn-n39-r8",
                 sequence_length=60, embedding_size=128, device=None, use_flip=True):
        self.seq_len = sequence_length
        self.use_flip = use_flip
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.graph = Graph("coco")
        self.connect_joint = self.graph.connect_joint

        # trusted official GaitGraph release; it stores an argparse.Namespace ("opt"),
        # which PyTorch 2.6's default weights_only=True refuses.
        ckpt = torch.load(weights_path, map_location="cpu", weights_only=False)
        sd = ckpt["model"] if isinstance(ckpt, dict) and "model" in ckpt else ckpt
        num_input, num_channel = _detect_branch_config(sd)
        self.use_multi_branch = (num_input > 1)
        self.num_input, self.num_channel = num_input, num_channel

        model, self.model_args = ResGCNv1.create(
            network_name,
            A=torch.tensor(self.graph.A, dtype=torch.float32, requires_grad=False),
            num_class=embedding_size, num_input=num_input, num_channel=num_channel,
            parts=self.graph.parts,
        ), None
        missing, unexpected = model.load_state_dict(sd, strict=False)
        self.load_report = (list(missing), list(unexpected))
        model.eval().to(self.device)
        for p in model.parameters():
            p.requires_grad_(False)
        self.model = model
        print(f"[adversary] {network_name}  num_input={num_input} num_channel={num_channel} "
              f"multi_branch={self.use_multi_branch}  device={self.device}")
        if missing or unexpected:
            print(f"[adversary] state_dict load: {len(missing)} missing, "
                  f"{len(unexpected)} unexpected (strict=False)")

    def _preprocess(self, tracklet):
        """(T,17,3) -> (I, C2, L, 17) float32 model input for one tracklet."""
        d = select_sequence_center(np.asarray(tracklet, np.float32), self.seq_len)
        return multi_input(d, self.connect_joint, self.use_multi_branch).astype(np.float32)

    @torch.no_grad()
    def embed_batch(self, tracklets, batch_size=256):
        """tracklets: list of (T,17,3). Returns (N,128) L2-normalized embeddings.
        Reproduces evaluate.py flip-TTA (cat flip along dim=1, split, average)."""
        embs = []
        for i in range(0, len(tracklets), batch_size):
            chunk = tracklets[i:i + batch_size]
            x = torch.from_numpy(np.stack([self._preprocess(t) for t in chunk])).to(self.device)
            if self.use_flip:
                bsz = x.shape[0]
                x = torch.cat([x, torch.flip(x, dims=[1])], dim=0)
                out = self.model(x)
                f1, f2 = torch.split(out, [bsz, bsz], dim=0)
                out = torch.mean(torch.stack([f1, f2]), dim=0)
            else:
                out = self.model(x)
            embs.append(out.cpu().numpy())
        return np.concatenate(embs, axis=0)

    def embed_dict(self, tracklet_dict, batch_size=256):
        """{key:(T,17,3)} -> {key:(128,)} preserving keys."""
        keys = list(tracklet_dict.keys())
        vecs = self.embed_batch([tracklet_dict[k] for k in keys], batch_size)
        return {k: vecs[j] for j, k in enumerate(keys)}
