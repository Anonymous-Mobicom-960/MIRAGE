"""
casia_loader -- stream the GaitGraph CASIA-B COCO-17 pose CSV into per-sequence
tracklets, without torch_geometric or the vendored PoseDataset.

CSV schema (verified against GaitGraph/src/datasets/gait.py and
GaitGraph2/.../casia_b_pose.py):
  row[0]  = "<something>/<seqid>/<frame>"  where seqid = "SUB-STATUS-SEQ-ANGLE"
            e.g. "x/001-nm-01-000/012.jpg"   (STATUS in {nm,bg,cl})
  row[1:] = 51 floats = 17 COCO joints x (x, y, conf), row-major per joint.

Canonical identity-disjoint split (GaitGraph): train = ids 1..74, test = 75..124.

A "tracklet" is one (SUB, STATUS, SEQ, ANGLE) walk => (T, 17, 3) float32 in CSV
row order (frames are stored in file order, mirroring PoseDataset which stacks
data_dict[target].values() unsorted).
"""
import os
import numpy as np

WALKING_STATUS = {"nm": 0, "bg": 1, "cl": 2}

SPLIT_IDS = {
    "train": set(range(1, 60)),        # GaitGraph train
    "valid": set(range(60, 75)),       # GaitGraph valid
    "train_valid": set(range(1, 75)),  # full 74-id train pool
    "test": set(range(75, 125)),       # 50-id disjoint test
    "all": set(range(1, 125)),
}


def load_tracklets(csv_path, split="test", min_frames=61, id_subset=None, verbose=True):
    """Stream the CSV, keep only rows whose subject id is in the split (filtered
    early to bound memory), and return {(sub,status,seq,angle): (T,17,3) float32}.

    min_frames: drop tracklets with fewer frames (PoseDataset drops < seq_len+1,
    i.e. < 61 for the default sequence_length=60). Set 0 to keep everything.
    id_subset: optional explicit set of subject ids (overrides `split`).
    """
    keep_ids = set(id_subset) if id_subset is not None else SPLIT_IDS[split]
    seqs = {}
    n_rows = n_kept = n_bad = 0
    with open(csv_path, "r") as f:
        f.readline()  # header
        for line in f:
            n_rows += 1
            comma = line.find(",")
            if comma < 0:
                continue
            fname = line[:comma]
            parts = fname.split("/")
            if len(parts) < 2:
                continue
            seqid = parts[1]
            sp = seqid.split("-")
            if len(sp) != 4:
                continue
            sub_s, status_s, seq_s, ang_s = sp
            try:
                sub = int(sub_s)
            except ValueError:
                continue
            if sub not in keep_ids:
                continue
            status = WALKING_STATUS.get(status_s)
            if status is None:
                continue
            vals = line[comma + 1:].split(",")
            if len(vals) != 51:
                n_bad += 1
                continue
            try:
                kp = np.asarray(vals, dtype=np.float32).reshape(17, 3)
            except ValueError:
                n_bad += 1
                continue
            key = (sub, status, int(seq_s), int(ang_s))
            seqs.setdefault(key, []).append(kp)
            n_kept += 1

    tracklets = {}
    n_short = 0
    for key, frames in seqs.items():
        if min_frames and len(frames) < min_frames:
            n_short += 1
            continue
        tracklets[key] = np.stack(frames).astype(np.float32)  # (T,17,3)

    if verbose:
        ids = sorted({k[0] for k in tracklets})
        print(f"[casia_loader] split={split}: rows={n_rows} kept_frames={n_kept} "
              f"bad={n_bad} tracklets={len(tracklets)} dropped_short(<{min_frames})={n_short}")
        print(f"[casia_loader]   subjects={len(ids)} range=[{min(ids) if ids else '-'}"
              f"..{max(ids) if ids else '-'}]  "
              f"per-status={{nm,bg,cl}}="
              f"{tuple(sum(1 for k in tracklets if k[1]==s) for s in (0,1,2))}")
    return tracklets


if __name__ == "__main__":
    import sys
    csv = sys.argv[1] if len(sys.argv) > 1 else os.path.join(
        os.path.dirname(__file__), "data", "casia-b_pose_coco.csv")
    tr = load_tracklets(csv, split="test")
    # sanity: frame-count + coord range
    lens = [v.shape[0] for v in tr.values()]
    sample = next(iter(tr.values()))
    print("frame-count min/med/max:", int(np.min(lens)), int(np.median(lens)), int(np.max(lens)))
    print("coord ranges: x[%.1f,%.1f] y[%.1f,%.1f] conf[%.2f,%.2f]" % (
        sample[..., 0].min(), sample[..., 0].max(),
        sample[..., 1].min(), sample[..., 1].max(),
        sample[..., 2].min(), sample[..., 2].max()))
