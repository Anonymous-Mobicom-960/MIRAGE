#!/usr/bin/env python3
"""COMMON POST-ANONYMISATION EXTRACTOR: turns what each defence RELEASES into the two
representations the attack classes consume.

  released artifact  ->  silhouette (T,64,64) uint8   [Class 1, Class 2]
                     ->  pose       (T,17,3) float32  [Class 3, Class 4]

WHAT "RELEASED" MEANS PER ARM, and why the conversion differs:
  raw / dp2  : a photorealistic RGB video. Silhouette = YOLO11s-seg on the OUTPUT video.
               Pose = RTMPose-t on the OUTPUT video.
  mirage     : MIRAGE does not release a photorealistic video at the device boundary. It releases
               a SHAPE MASK plus an ANONYMISED SKELETON. So its silhouette is the emitted
               bounding-box mask and its pose is the emitted anonymised skeleton, produced here
               by the REAL SHIPPED CODE (`mirage_tier1.mask_mitigate`,
               `pose_anon_edge.anonymize_pose_log`) applied at NATIVE RESOLUTION to this same
               10 fps corpus.
  This asymmetry is REAL and must be stated in any table: each arm is attacked on what it
  actually hands out. It is not a protocol inconsistency, it is the protocol.

THE SUBJECT BOX IS TAKEN FROM THE CLEAN ANNOTATION AND IS THE SAME FOR EVERY ARM.
`_annotations/fullbody/*.json` resolved identity on the CLEAN video with face recognition +
ByteTrack; that binding cannot be redone on an anonymised video. Re-detecting per arm would let
detector-localisation differences masquerade as privacy differences. Detector RECALL is still
measured per arm and reported as a failure rate; it is a result, not something hidden.

Frame mapping: the 10 fps corpus came from ffmpeg `fps=10`. The naive map round(n*src_fps/10) is
WRONG BY A CONSTANT +1, measured by decoding both videos and matching pixels (mean |delta| ~0.5
at the true frame vs 15-30 at its neighbours), consistently +1 across every source rate present.
It is a systematic bias, not a +/-1 jitter: uncorrected it displaced the subject box by mean
7.90 px / p95 21.2 px, cost >10 % IoU on 10.1 % of frames, and made the boxes essentially
disjoint on 0.6 %.

Box STALENESS: annotations are not dense on every clip, so `argmin(|keys - s|)` could silently
return a box up to 2 s away. Such a box no longer contains the subject: RTMPose never declines,
so it returns a FABRICATED pose (mean keypoint confidence 0.188 vs 0.522 on fresh boxes). Frames
whose nearest annotation is more than STALE_MAX_S away are excluded from the POSE sequence and
counted separately, so detector recall and annotation sparsity are never conflated into one
"failure rate".

  python extract_arm.py --arm raw --dataset-root <reid_dataset_flat> \
      --video-dir corpus_10fps --also-mirage
  python extract_arm.py --arm dp2 --dataset-root <reid_dataset_flat> \
      --video-dir baselines_dp2/out
"""
import argparse
import json
import os
import subprocess
import sys
import time

import cv2
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.normpath(os.path.join(HERE, "..", "..", ".."))          # repository root
EDGE = os.path.join(ROOT, "tier1", "src", "edge_runner_pi5")
GAIT = os.path.normpath(os.path.join(HERE, "..", "gait"))

sys.path.insert(0, HERE)
sys.path.insert(0, EDGE)
from casia_domain import normalise, casia_reference, to_casia_domain   # noqa: E402

STALE_MAX_S = 0.20  # a subject box older than this is not the subject any more (see docstring)
MAXF = 120        # THE PUBLISHED FRAME BUDGET: 12 s at 10 fps; two clean 60-frame gait windows.
                  # Class 1/2 resample this to L_FRAMES=70; Class 3/4 need >= 60.
WIN = 2           # MASK_TEMPORAL_WIN, the shipped 2-frame temporal window
EPS = 0.01        # MASK_SIMPLIFY_EPS, the shipped simplification epsilon


def ann_key(relpath):
    return relpath.replace(" ", "").replace("/", "__") + ".json"


def src_fps(path):
    p = subprocess.run(["ffprobe", "-v", "error", "-select_streams", "v:0",
                        "-show_entries", "stream=r_frame_rate", "-of", "csv=p=0", path],
                       capture_output=True, text=True)
    a, b = p.stdout.strip().split("/")
    return float(a) / float(b)


def iou(a, b):
    ix0, iy0 = max(a[0], b[0]), max(a[1], b[1])
    ix1, iy1 = min(a[2], b[2]), min(a[3], b[3])
    iw, ih = max(0.0, ix1 - ix0), max(0.0, iy1 - iy0)
    inter = iw * ih
    ua = (a[2] - a[0]) * (a[3] - a[1]) + (b[2] - b[0]) * (b[3] - b[1]) - inter
    return inter / ua if ua > 0 else 0.0


def setup_mirage_mask_config(seed):
    """Reset every knob this study touches to its shipped default, then select the shipped
    bounding-box mode, so arm state can never leak between clips."""
    import config as C
    C.MASK_TEMPORAL_WIN = WIN
    C._MASK_DISPLACE_SEED = seed
    C._MASK_DISPLACE_PHASE = 0.0
    C.MASK_BBOX_MERGE, C.MASK_BBOX_PAD_FRAC = True, 0.0
    C.MASK_ELLIPSE_INFLATE = 1.15
    C.MASK_CLOSE_KERNEL_FRAC = 0.25
    C.MASK_RADIALLP_KEEP, C.MASK_RADIALLP_BINS = 4, 180
    C.MASK_DISPLACE_AMP_FRAC = 0.25
    C.MASK_DISPLACE_PHASE_STEP = 0.35
    C.MASK_DIRBAND_AMP = (0.10, 0.40)
    C.MASK_DIRBAND_N = (4, 7)
    C.MASK_BAND_AMP = (0.10, 0.40)
    C.MASK_BAND_N = (4, 7)
    C.MASK_SHAPE_MODE = "bbox"          # the SHIPPED silhouette mode: axis-aligned bounding box
    return C


def mirage_pose(raw_native, fps=10.0):
    """The REAL shipped gait anonymiser (the default preset) in the emit format."""
    import pose_anon_edge as PA
    log = [[{"kp": np.concatenate([f[:, :2], np.zeros((133 - 17, 2), np.float32)]).tolist(),
             "score": np.concatenate([f[:, 2], np.zeros(133 - 17, np.float32)]).tolist()}]
           for f in raw_native]
    anon = PA.anonymize_pose_log(log, "L4", fps=fps)
    den = np.stack([np.asarray(fr[0]["kp"][:17], np.float32) for fr in anon])
    return np.concatenate([den, raw_native[:, :, 2:3]], 2)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--arm", required=True)
    ap.add_argument("--video-dir", required=True)
    ap.add_argument("--dataset-root", required=True,
                    help="root of the flat evaluation dataset (distributed separately from this "
                         "repository); must contain the source clips and _annotations/fullbody/")
    ap.add_argument("--manifest", default=os.path.join(HERE, "manifests", "corpus_10fps.json"),
                    help="corpus manifest written by make_corpus_manifest.py")
    ap.add_argument("--yolo-weights", default=os.path.join(HERE, "models", "yolo11s-seg.pt"))
    ap.add_argument("--rtm-weights", default=os.path.join(EDGE, "models",
                                                          "rtmpose-t-wholebody.onnx"))
    ap.add_argument("--casia-train-csv", default=os.path.join(GAIT, "data",
                                                              "casia-b_pose_train_valid.csv"),
                    help="CASIA-B train split CSV, used only for the population domain reference")
    ap.add_argument("--also-mirage", action="store_true",
                    help="only valid with --arm raw: additionally emit the MIRAGE arm")
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--maxf", type=int, default=MAXF,
                    help="per-clip frame budget (default 120 = the published budget). Lowering it "
                         "is ONLY valid if every arm in the comparison uses the SAME value: Class "
                         "1/2 resample to L_FRAMES=70, so a 30-frame clip and a 120-frame clip "
                         "reach the attacker with different temporal coverage.")
    ap.add_argument("--clips", default="",
                    help="JSON list of clip filenames to restrict to, for subset comparisons.")
    ap.add_argument("--no-pose", action="store_true",
                    help="SILHOUETTE ONLY: skip RTMPose entirely. RTMPose runs on CPU here and is "
                         "the single biggest cost in extraction, yet Classes 1 and 2 never touch "
                         "pose. The npz is written with zeroed pose and pose_valid=False, so any "
                         "attempt to score pose attacks on it is refused rather than silently "
                         "scoring zeros.")
    ap.add_argument("--suffix", default="",
                    help="write to arms/<arm><suffix>/ instead of arms/<arm>/, to re-extract "
                         "without destroying the arms behind an existing report.")
    a = ap.parse_args()

    FLAT = a.dataset_root
    ANN = os.path.join(FLAT, "_annotations", "fullbody")

    out_dir = os.path.join(HERE, "arms", a.arm + a.suffix)
    os.makedirs(out_dir, exist_ok=True)
    mir_dir = os.path.join(HERE, "arms", "mirage" + a.suffix)
    if a.also_mirage:
        os.makedirs(mir_dir, exist_ok=True)
        import mirage_tier1 as ST

    man = json.load(open(a.manifest))
    if a.clips:
        keep = set(json.load(open(a.clips)))
        man = [m for m in man if m["clip"] in keep]
        print(f"[{a.arm}] restricted to {len(man)} clips", flush=True)
    if a.limit:
        man = man[:a.limit]

    from ultralytics import YOLO
    from rtmlib import RTMPose
    yolo = YOLO(a.yolo_weights)
    pose = None if a.no_pose else RTMPose(onnx_model=a.rtm_weights, model_input_size=(192, 256),
                   backend="onnxruntime", device="cpu")
    ref_hip, ref_torso = casia_reference(a.casia_train_csv)
    print(f"[{a.arm}] CASIA reference: mid-hip {ref_hip.round(2)} torso {ref_torso:.2f}px", flush=True)

    summary = []
    t_all = time.time()
    for ci, m in enumerate(man, 1):
        dst = os.path.join(out_dir, m["clip"] + ".npz")
        mdst = os.path.join(mir_dir, m["clip"] + ".npz") if a.also_mirage else None
        vid = os.path.join(a.video_dir, m["clip"])
        if os.path.exists(dst) and (not a.also_mirage or os.path.exists(mdst)):
            if os.path.exists(dst + ".json"):
                summary.append(json.load(open(dst + ".json")))
            continue
        if not os.path.exists(vid):
            print(f"[{a.arm}] MISSING VIDEO {vid}", flush=True); continue

        d = json.load(open(os.path.join(ANN, ann_key(m["source_relpath"]))))
        byf = {}
        for f in d["frames"]:
            for p in f.get("people", []):
                if p["identity"] == m["identity"]:
                    byf[f["frame"]] = p["bbox_xyxy"]; break
        if not byf:
            print(f"[{a.arm}] NO SUBJECT BOXES {m['clip']}", flush=True); continue
        keys = np.array(sorted(byf))
        f_src = src_fps(os.path.join(FLAT, m["source_file"]))

        if a.also_mirage:
            C = setup_mirage_mask_config(
                int(__import__("hashlib").sha256(m["clip"].encode()).hexdigest()[:8], 16) & 0x7FFFFFFF)

        cap = cv2.VideoCapture(vid)
        sil, mir_sil, kps = [], [], []
        hist = []
        n_no_mask = n_no_pose = n_no_det = 0
        n_stale = 0
        stale_list = []
        n = -1
        while n + 1 < a.maxf:
            ok, fr = cap.read()
            if not ok:
                break
            n += 1
            s = int(round(n * f_src / 10.0)) + 1   # +1: measured, see module docstring
            j = int(np.argmin(np.abs(keys - s)))
            stale_s = abs(int(keys[j]) - s) / max(f_src, 1e-6)
            box = np.asarray(byf[int(keys[j])], np.float32)
            H, W = fr.shape[:2]

            r = yolo.predict(fr, classes=[0], verbose=False, device=0, half=True, imgsz=640)[0]
            mask = np.zeros((H, W), np.uint8)
            if r.masks is not None and len(r.masks) > 0:
                bxs = r.boxes.xyxy.cpu().numpy()
                k = int(np.argmax([iou(box, b) for b in bxs]))
                if iou(box, bxs[k]) > 0.10:
                    poly = np.asarray(r.masks.xy[k], np.int32)
                    if poly.size >= 6:
                        cv2.fillPoly(mask, [poly], 1)
                else:
                    n_no_det += 1
            else:
                n_no_det += 1

            nz = normalise(mask)
            if nz is None:
                n_no_mask += 1
            else:
                sil.append((nz > 127).astype(np.uint8))

            if a.also_mirage:
                C._MASK_DISPLACE_PHASE = float(C.MASK_DISPLACE_PHASE_STEP) * n
                hist.append(mask)
                if len(hist) > WIN:
                    hist.pop(0)
                em = ST.mask_mitigate(hist, mask, EPS)
                mz = normalise(em)
                if mz is not None:
                    mir_sil.append((mz > 127).astype(np.uint8))

            if stale_s > STALE_MAX_S:
                # the box predates this frame by too much to still contain the subject; a pose
                # estimated inside it is fabricated, so it is EXCLUDED rather than silently kept
                n_stale += 1
                stale_list.append(round(stale_s, 3))
                continue
            if a.no_pose:
                kps.append(np.zeros((17, 3), np.float32))
                continue
            k17, s17 = pose(fr, bboxes=box[None])
            if k17 is None or len(k17) == 0:
                n_no_pose += 1
                kps.append(np.zeros((17, 3), np.float32))
            else:
                kk = np.asarray(k17[0][:17], np.float32)
                ss = np.asarray(s17[0][:17], np.float32).reshape(-1, 1)
                kps.append(np.concatenate([kk, ss], 1))
        cap.release()

        if len(sil) < 24:
            # A clip can fall below the floor because the DEFENCE DEFEATED THE PERSON DETECTOR.
            # That is the single most important number about such an arm, so the failure counters
            # are recorded before dropping, flagged, and the rate survives at full N. No npz is
            # written and the clip is still excluded from every attack.
            print(f"[{a.arm}] TOO SHORT {m['clip']} ({len(sil)})", flush=True)
            summary.append(dict(arm=a.arm, clip=m["clip"], identity=m["identity"],
                                condition=m["condition"], source=m["source_file"],
                                n_frames=int(n + 1), n_sil_frames=int(len(sil)),
                                frames_no_person_detected=int(n_no_det),
                                frames_empty_mask=int(n_no_mask),
                                frames_no_pose=int(n_no_pose),
                                frames_stale_box_excluded=int(n_stale),
                                dropped=True, drop_reason="silhouette frames < 24"))
            continue
        sil = np.stack(sil).astype(np.uint8)
        if len(kps) < 2:
            # every (or almost every) frame had a stale subject box, so there is no usable pose
            # for this clip. Keep the silhouette (Classes 1/2 are fine), flag the pose invalid.
            print(f"[{a.arm}] NO USABLE POSE ({n_stale} stale of {n+1}) {m['clip']}", flush=True)
            raw_native = np.zeros((max(len(sil), 1), 17, 3), np.float32)
            casia = raw_native.copy(); pose_valid = False
        elif a.no_pose:
            raw_native = np.zeros((max(len(sil), 1), 17, 3), np.float32)
            casia = raw_native.copy(); pose_valid = False
        else:
            raw_native = np.stack(kps).astype(np.float32)
            casia = to_casia_domain(raw_native, ref_hip, ref_torso)
            pose_valid = casia is not None
            if not pose_valid:
                casia = np.zeros_like(raw_native)
                print(f"[{a.arm}] POSE REGISTRATION FAILED (silhouette kept) {m['clip']}",
                      flush=True)
        rec = dict(arm=a.arm, clip=m["clip"], identity=m["identity"],
                   condition=m["condition"], source=m["source_file"],
                   n_frames=int(n + 1), n_sil_frames=int(sil.shape[0]),
                   frames_no_person_detected=int(n_no_det),
                   frames_empty_mask=int(n_no_mask),
                   frames_no_pose=int(n_no_pose),
                   frames_stale_box_excluded=int(n_stale),
                   max_box_staleness_s=float(max(stale_list)) if stale_list else 0.0,
                   mask_coverage=float(len(sil) / max(1, n + 1)),
                   mean_pose_score=float(raw_native[:, :, 2].mean()), pose_valid=bool(pose_valid))
        np.savez_compressed(dst, sil=sil, pose_casia=casia, pose_native=raw_native,
                            pose_valid=bool(pose_valid),
                            identity=m["identity"], clip=m["clip"], source=m["source_file"],
                            condition=m["condition"], arm=a.arm)
        json.dump(rec, open(dst + ".json", "w"))
        summary.append(rec)

        if a.also_mirage:
            msil = np.stack(mir_sil).astype(np.uint8)
            mpose = mirage_pose(raw_native, fps=10.0)
            mcasia = to_casia_domain(mpose, ref_hip, ref_torso)
            mvalid = mcasia is not None
            if not mvalid:
                mcasia = np.zeros_like(mpose)
                print(f"[mirage] GAIT DEFENCE DECLINED (unobserved joints pruned; "
                      f"silhouette kept) {m['clip']}", flush=True)
            np.savez_compressed(mdst, sil=msil, pose_casia=mcasia, pose_native=mpose,
                                pose_valid=bool(mvalid),
                                identity=m["identity"], clip=m["clip"],
                                source=m["source_file"], condition=m["condition"],
                                arm="mirage")
            mrec = dict(rec); mrec["arm"] = "mirage"; mrec["pose_valid"] = bool(mvalid)
            mrec["mask_coverage"] = float(len(msil) / max(1, n + 1)); mrec["n_sil_frames"] = int(len(msil))
            json.dump(mrec, open(mdst + ".json", "w"))

        el = time.time() - t_all
        print(f"[{a.arm}] {ci:3d}/{len(man)} {m['identity']:5s} {m['clip'][:32]:32s} "
              f"T={sil.shape[0]:3d} nodet={n_no_det:3d} nomask={n_no_mask:3d} nopose={n_no_pose:3d} "
              f"cov={rec['mask_coverage']:.3f} [{el/60:.1f}min]", flush=True)

    os.makedirs(os.path.join(HERE, "reports"), exist_ok=True)
    sp = os.path.join(HERE, "reports", f"EXTRACT_{a.arm}{a.suffix}.json")
    json.dump(summary, open(sp, "w"), indent=1)
    if summary:
        tf = sum(s["n_frames"] for s in summary)
        nd = sum(s["frames_no_person_detected"] for s in summary)
        nm = sum(s["frames_empty_mask"] for s in summary)
        npo = sum(s["frames_no_pose"] for s in summary)
        print(f"\n[{a.arm}] DONE {len(summary)} clips, {tf:,} frames, "
              f"{len({s['identity'] for s in summary})} identities in {(time.time()-t_all)/60:.1f} min")
        print(f"[{a.arm}] EXTRACTOR FAILURE RATES (reported, never silently dropped):")
        print(f"[{a.arm}]   no person detected : {nd:,}/{tf:,} = {100*nd/tf:.2f} %")
        print(f"[{a.arm}]   empty mask         : {nm:,}/{tf:,} = {100*nm/tf:.2f} %")
        print(f"[{a.arm}]   no pose            : {npo:,}/{tf:,} = {100*npo/tf:.2f} %")
        ns = sum(s.get("frames_stale_box_excluded", 0) for s in summary)
        print(f"[{a.arm}]   stale box EXCLUDED : {ns:,}/{tf:,} = {100*ns/tf:.2f} %  "
              f"(annotation sparsity, reported SEPARATELY from detector recall)")
    print(f"[{a.arm}] wrote {sp}")


if __name__ == "__main__":
    main()
