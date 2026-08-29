"""
Tier 2 visual utility eval, following the original MIRAGE paper's exact
methodology (Section V-B): FID, SSIM, PSNR, LPIPS, and a geometry/expression
-preservation distance metric, comparing the synthetic replacement against
the original unblurred video as ground truth.

Adaptation for MIRAGE (full-body, not face-only): the original paper's
"landmark distance" used 468-pt MediaPipe FACE landmarks (since MIRAGE only
replaced faces). MIRAGE replaces the whole body with a synthetic
character, so this script uses RTMPose-T 17-point BODY keypoints instead --
same role (does the replacement preserve real geometry/motion), adapted to
what's actually being replaced here.

Framing note (important, confirmed against the paper's own numbers before
building this): MIRAGE's own reported numbers (SSIM 0.61, PSNR 12.85dB,
landmark distance 15.94 -- all meaningfully worse than their own MFS
baseline) confirm the original paper did NOT expect near-perfect pixel
fidelity even for their own real system, because MIRAGE (like MIRAGE)
replaces with a DIFFERENT synthetic identity, not a same-identity swap.
MFS was their "best achievable quality" ceiling baseline (same generation
approach, unblurred/unrestricted input), not a same-identity requirement.
MIRAGE's equivalent baseline (unaltered WanVideo workflow, richer
input signals) is a separate, later comparison -- this script reports
MIRAGE's own numbers against ground truth first.

Usage:
  python eval_tier2_visual_utility.py <ground_truth.mp4> <generated.mp4> [--skip-keypoint-distance]

--skip-keypoint-distance: the body-keypoint-distance metric currently only
scores the LARGEST-box detected person per frame (no multi-person matching
logic yet) -- correct for single-person clips, but silently discards a real
person's signal on multi-person content. Pass this flag on multi-person
clips rather than get a misleading single-person-only number. SSIM/PSNR/
LPIPS/FID are unaffected either way -- those are whole-frame comparisons
with no per-person bbox logic involved at all, multi-person or not.
"""
import os
import sys
import json

import cv2
import numpy as np
import torch
import lpips as lpips_lib
from skimage.metrics import structural_similarity as ssim_fn
from skimage.metrics import peak_signal_noise_ratio as psnr_fn
from scipy import linalg

REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
sys.path.insert(0, os.path.join(REPO_ROOT, "src"))

from rtmlib import Body

GT_PATH = sys.argv[1] if len(sys.argv) > 1 else None
GEN_PATH = sys.argv[2] if len(sys.argv) > 2 else None
SKIP_KEYPOINT_DISTANCE = "--skip-keypoint-distance" in sys.argv[3:]
if GT_PATH is None or GEN_PATH is None:
    print("Usage: eval_tier2_visual_utility.py <ground_truth.mp4> <generated.mp4> [--skip-keypoint-distance]")
    sys.exit(1)


def load_frames(path):
    cap = cv2.VideoCapture(path)
    frames = []
    while True:
        ret, f = cap.read()
        if not ret:
            break
        frames.append(f)
    cap.release()
    return frames


def compute_fid(real_feats, fake_feats):
    """Standard FID: Frechet distance between two multivariate Gaussians
    fit to Inception-pool3 features of the real and fake frame sets."""
    mu1, sigma1 = real_feats.mean(axis=0), np.cov(real_feats, rowvar=False)
    mu2, sigma2 = fake_feats.mean(axis=0), np.cov(fake_feats, rowvar=False)
    diff = mu1 - mu2
    covmean, _ = linalg.sqrtm(sigma1.dot(sigma2), disp=False)
    if np.iscomplexobj(covmean):
        covmean = covmean.real
    fid = diff.dot(diff) + np.trace(sigma1 + sigma2 - 2 * covmean)
    return float(fid)


def get_inception_features(frames_bgr, device):
    from pytorch_fid.inception import InceptionV3
    block_idx = InceptionV3.BLOCK_INDEX_BY_DIM[2048]
    model = InceptionV3([block_idx]).to(device).eval()
    feats = []
    with torch.no_grad():
        for frame in frames_bgr:
            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            rgb = cv2.resize(rgb, (299, 299))
            t = torch.from_numpy(rgb).permute(2, 0, 1).float().unsqueeze(0) / 255.0
            t = t.to(device)
            out = model(t)[0]
            feats.append(out.squeeze().cpu().numpy().reshape(-1))
    return np.array(feats)


def main():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Device: {device}")

    print(f"Loading ground truth: {GT_PATH}")
    gt_frames = load_frames(GT_PATH)
    print(f"Loading generated: {GEN_PATH}")
    gen_frames = load_frames(GEN_PATH)

    n = min(len(gt_frames), len(gen_frames))
    print(f"Comparing {n} frames (gt has {len(gt_frames)}, gen has {len(gen_frames)})")
    gt_frames = gt_frames[:n]
    gen_frames = gen_frames[:n]

    # --- SSIM / PSNR (pixel-level fidelity) ---
    print("\nComputing SSIM/PSNR...")
    ssim_scores, psnr_scores = [], []
    for gt, gen in zip(gt_frames, gen_frames):
        if gt.shape != gen.shape:
            gen = cv2.resize(gen, (gt.shape[1], gt.shape[0]))
        s = ssim_fn(gt, gen, channel_axis=2, data_range=255)
        p = psnr_fn(gt, gen, data_range=255)
        ssim_scores.append(s)
        psnr_scores.append(p)
    ssim_arr = np.array(ssim_scores)
    # PSNR is +inf wherever the two frames are IDENTICAL. Dropping those silently made the
    # GT-vs-GT sanity check - the first thing anyone should run - crash with None formatting.
    # Keep the count so a perfect match reports as such instead of vanishing.
    psnr_identical = int(sum(1 for x in psnr_scores if not np.isfinite(x)))
    psnr_arr = np.array([x for x in psnr_scores if np.isfinite(x)])

    # --- LPIPS (learned perceptual similarity) ---
    print("Computing LPIPS...")
    lpips_model = lpips_lib.LPIPS(net='alex').to(device)
    lpips_scores = []
    with torch.no_grad():
        for gt, gen in zip(gt_frames, gen_frames):
            if gt.shape != gen.shape:
                gen = cv2.resize(gen, (gt.shape[1], gt.shape[0]))
            gt_t = torch.from_numpy(cv2.cvtColor(gt, cv2.COLOR_BGR2RGB)).permute(2, 0, 1).float() / 127.5 - 1
            gen_t = torch.from_numpy(cv2.cvtColor(gen, cv2.COLOR_BGR2RGB)).permute(2, 0, 1).float() / 127.5 - 1
            d = lpips_model(gt_t.unsqueeze(0).to(device), gen_t.unsqueeze(0).to(device))
            lpips_scores.append(float(d.item()))
    lpips_arr = np.array(lpips_scores)

    # --- FID (distribution-level realism) ---
    # WARNING (confirmed via direct simulation, not assumed): FID's covariance
    # matrix is 2048x2048 (Inception pool3 feature dim). With only ~100
    # frames, that matrix is severely rank-deficient (rank <= n_samples-1)
    # and scipy.linalg.sqrtm on it is numerically unstable -- a controlled
    # test with two NEAR-IDENTICAL simulated 100-sample distributions
    # produced a nonsense FID of ~84,700 (should be ~0). FID needs hundreds
    # to thousands of samples for a stable estimate; standard FID literature
    # commonly uses >=10,000. At n=100 the FID number below is NOT a
    # trustworthy quality signal -- computed anyway for completeness/
    # transparency, but flagged explicitly. Prefer SSIM/PSNR/LPIPS/keypoint-
    # distance (all per-frame metrics, don't need a population covariance
    # estimate) as the reliable numbers at this sample size.
    print("Computing FID (Inception-v3 features) -- UNRELIABLE at n=100, see warning above...")
    gt_feats = get_inception_features(gt_frames, device)
    gen_feats = get_inception_features(gen_frames, device)
    fid_value = compute_fid(gt_feats, gen_feats)
    fid_reliable = n >= 1000

    # --- Body-keypoint distance (geometry/motion preservation, adapted
    # from the original paper's 468-pt FACE landmark distance to MIRAGE's
    # full-body replacement) ---
    # SKIPPED on multi-person clips (--skip-keypoint-distance): current
    # get_kpts() only keeps the largest-box person, silently discarding
    # everyone else -- correct for single-person clips, misleading here.
    if SKIP_KEYPOINT_DISTANCE:
        print("Skipping body-keypoint distance (--skip-keypoint-distance set -- "
              "multi-person clip, current metric only scores the largest-box person).")
        kpt_dist_arr = np.array([])
        n_no_detection = None
    else:
        print("Computing body-keypoint distance (RTMPose-T, gt vs generated)...")
        body = Body(
            det='https://download.openmmlab.com/mmpose/v1/projects/rtmposev1/onnx_sdk/yolox_nano_8xb8-300e_humanart-40f6f0d0.zip',
            det_input_size=(416, 416),
            pose='https://download.openmmlab.com/mmpose/v1/projects/rtmposev1/onnx_sdk/rtmpose-t_simcc-body7_pt-body7_420e-256x192-026a1439_20230504.zip',
            pose_input_size=(192, 256),
            backend='onnxruntime', device='cpu',
        )

        def get_kpts(frame):
            h, w = frame.shape[:2]
            infer_frame = cv2.resize(frame, (416, 416))
            boxes = body.det_model(infer_frame)
            if boxes is None or len(boxes) == 0:
                return None
            sx, sy = w / 416, h / 416
            real_boxes = np.array([[b[0]*sx, b[1]*sy, b[2]*sx, b[3]*sy] for b in boxes])
            kpts, scores = body.pose_model(frame, bboxes=real_boxes)
            # take the largest-box person if multiple detected
            areas = [(b[2]-b[0])*(b[3]-b[1]) for b in real_boxes]
            idx = int(np.argmax(areas))
            return kpts[idx], scores[idx], real_boxes[idx]

        kpt_dists = []
        n_no_detection = 0
        for gt, gen in zip(gt_frames, gen_frames):
            gt_r = get_kpts(gt)
            gen_r = get_kpts(gen)
            if gt_r is None or gen_r is None:
                n_no_detection += 1
                continue
            gt_kpts, gt_scores, gt_box = gt_r
            gen_kpts, gen_scores, gen_box = gen_r
            # normalize by each frame's own box diagonal (scale-invariant --
            # the avatar isn't guaranteed to be pixel-registered to the same
            # box size as the real person), matching the spirit of OKS-style
            # normalization used in the pose Pareto/OKS-AP evals earlier.
            gt_diag = np.sqrt((gt_box[2]-gt_box[0])**2 + (gt_box[3]-gt_box[1])**2)
            gen_diag = np.sqrt((gen_box[2]-gen_box[0])**2 + (gen_box[3]-gen_box[1])**2)
            gt_norm = (gt_kpts - gt_box[:2]) / gt_diag
            gen_norm = (gen_kpts - gen_box[:2]) / gen_diag
            vis = (gt_scores >= 0.3) & (gen_scores >= 0.3)
            if not vis.any():
                n_no_detection += 1
                continue
            d = np.linalg.norm(gt_norm[vis] - gen_norm[vis], axis=1).mean()
            kpt_dists.append(float(d))
        kpt_dist_arr = np.array(kpt_dists)

    # --- Results ---
    results = {
        "ground_truth": GT_PATH,
        "generated": GEN_PATH,
        "n_frames_compared": n,
        "n_frames_no_pose_detection": n_no_detection,
        "ssim": {"mean": float(ssim_arr.mean()), "std": float(ssim_arr.std())},
        "psnr_db": {"mean": float(psnr_arr.mean()) if len(psnr_arr) else float("inf"),
                    "std": float(psnr_arr.std()) if len(psnr_arr) else 0.0,
                    "identical_frames": psnr_identical, "finite_frames": int(len(psnr_arr))},
        "lpips": {"mean": float(lpips_arr.mean()), "std": float(lpips_arr.std())},
        "fid": fid_value,
        "fid_reliable": fid_reliable,
        "fid_caveat": "UNRELIABLE below ~1000 frames -- 2048-dim covariance matrix is rank-deficient/singular at low sample counts, confirmed via direct simulation to produce nonsense values (see script docstring/comments). Do not report this FID number without either (a) many more frames, or (b) an FID variant designed for small samples.",
        "body_keypoint_distance_normalized": {
            "mean": float(kpt_dist_arr.mean()) if len(kpt_dist_arr) else None,
            "std": float(kpt_dist_arr.std()) if len(kpt_dist_arr) else None,
            "n_scored_frames": len(kpt_dist_arr),
        },
        "original_mirage_paper_reference": {
            "note": "face-only, MFS baseline vs MIRAGE (their own system), NOT a same-identity requirement -- see script docstring",
            "fid": {"baseline_mfs": 31.00, "mirage": 63.70},
            "ssim": {"baseline_mfs": 0.76, "mirage": 0.61},
            "psnr_db": {"baseline_mfs": 15.87, "mirage": 12.85},
            "lpips": {"baseline_mfs": 0.14, "mirage": 0.27},
            "landmark_distance_468pt_face": {"baseline_mfs": 8.81, "mirage": 15.94},
        },
    }

    print("\n" + "=" * 60)
    print("TIER 2 VISUAL UTILITY RESULTS")
    print("=" * 60)
    print(f"SSIM:  {results['ssim']['mean']:.4f} ± {results['ssim']['std']:.4f}   (paper MIRAGE: 0.61)")
    _p = results['psnr_db']
    if _p['finite_frames'] == 0:
        print(f"PSNR:  inf dB on ALL {_p['identical_frames']} frames - inputs are IDENTICAL "
              f"(sanity check passed)")
    else:
        print(f"PSNR:  {_p['mean']:.2f} ± {_p['std']:.2f} dB   (paper MIRAGE: 12.85 dB)"
              + (f"   [{_p['identical_frames']} identical frames excluded]"
                 if _p['identical_frames'] else ""))
    print(f"LPIPS: {results['lpips']['mean']:.4f} ± {results['lpips']['std']:.4f}   (paper MIRAGE: 0.27)")
    fid_flag = "" if fid_reliable else "  [UNRELIABLE at n<1000, see JSON's fid_caveat -- do not report]"
    print(f"FID:   {results['fid']:.2f}   (paper MIRAGE: 63.70){fid_flag}")
    kd = results['body_keypoint_distance_normalized']
    if kd['mean'] is None:
        print("Body-keypoint distance: SKIPPED (--skip-keypoint-distance, multi-person clip)")
    else:
        print(f"Body-keypoint distance (normalized): {kd['mean']:.4f} ± {kd['std']:.4f}  "
              f"(scored {kd['n_scored_frames']}/{n} frames; paper's face-landmark-dist MIRAGE: 15.94, different units/metric)")
    print("=" * 60)

    # One FIXED output path meant every run overwrote the last, which is how B.3's
    # sanity-check run stopped existing while the ledger still cited it. Tag the artifact so a
    # sanity run and a real run cannot collide, and keep writing the legacy path as a copy so
    # anything that reads it still works.
    tag = os.environ.get("EVAL_TAG") or os.path.splitext(os.path.basename(GEN_PATH))[0]
    tag = "".join(c if (c.isalnum() or c in "._-") else "_" for c in str(tag))[:60]
    here = os.path.dirname(os.path.abspath(__file__))
    results["inputs"] = {"ground_truth": os.path.abspath(GT_PATH),
                         "generated": os.path.abspath(GEN_PATH), "tag": tag,
                         "n_frames_compared": int(n)}
    out_path = os.path.join(here, "reports", f"VISUAL_UTILITY_{tag}.json")
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    with open(out_path, "w") as f:
        json.dump(results, f, indent=2)
    legacy = os.path.join(here, "tier2_visual_utility_results.json")
    with open(legacy, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nResults written to {out_path}\n  (legacy copy: {legacy})")


if __name__ == "__main__":
    main()
