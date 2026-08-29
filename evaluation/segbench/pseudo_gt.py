"""
segbench.pseudo_gt -- build the platform-independent evaluation reference, ONCE,
and cache it. Two things every config is scored against:

  1. person_union  -- high-quality full-res (1264) person mask from a LARGE model
     (yolo11x-seg, retina masks, high infer size, on the GPU). This is the mask-
     quality / body-reveal reference. Not "truth", but a strong upper-bound
     reference far above any nano candidate. (Mild family bias toward yolo11 is
     acknowledged; the two big foreground people are easy and both big models
     agree closely on them, so it barely moves IoU.)

  2. face_boxes    -- MediaPipe FaceDetection boxes = the set of ACTUALLY-VISIBLE
     faces. MediaPipe naturally fails to fire on tiny/distant/occluded faces, so
     this operationalises the user's scoping: "ignore too small background people
     whose face isn't even visible." The privacy GATE is defined ONLY over these.

GPU (torch .pt) is used here because ONNX-Runtime has no CUDA provider in this
env; this reference step is offline and untimed, so GPU is free to use.
"""
import os
import numpy as np
import cv2


def _pack(m):
    return np.packbits(m.reshape(-1).astype(bool))


def unpack(p, hw):
    n = hw[0] * hw[1]
    p = np.asarray(p, dtype=np.uint8)
    return np.unpackbits(p)[:n].reshape(hw).astype(bool)


def build(video: str, out_dir: str, gt_weights: str = "yolo11x-seg.pt",
          gt_imgsz: int = 1280, gt_conf: float = 0.30, device=0,
          face_conf: float = 0.5, min_person_h: int = 220) -> str:
    os.makedirs(out_dir, exist_ok=True)
    out_npz = os.path.join(out_dir, "gt.npz")

    from ultralytics import YOLO
    seg = YOLO(gt_weights)

    import mediapipe as mp
    fd = mp.solutions.face_detection.FaceDetection(
        model_selection=1, min_detection_confidence=face_conf)

    cap = cv2.VideoCapture(video)
    W = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    H = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    hw = (H, W)

    person_union_all, person_union_fg, person_boxes, face_boxes = [], [], [], []
    idx = 0
    while True:
        ok, frame = cap.read()
        if not ok:
            break
        # --- strong person masks (GPU) ---
        r = seg.predict(frame, imgsz=gt_imgsz, conf=gt_conf, classes=[0],
                        device=device, verbose=False, retina_masks=True)[0]
        union_all = np.zeros(hw, dtype=bool)
        union_fg = np.zeros(hw, dtype=bool)   # only foreground (face-visible-scale) people
        boxes = np.zeros((0, 4), np.float32)
        if r.boxes is not None and len(r.boxes):
            boxes = r.boxes.xyxy.cpu().numpy().astype(np.float32)
        if r.masks is not None:
            md = r.masks.data.cpu().numpy()          # (N, H, W) at orig res
            for i in range(md.shape[0]):
                mi = (md[i] > 0.5)
                union_all |= mi
                # foreground = box tall enough that a face would be resolvable;
                # drops the tiny background gazebo figures the user de-scoped.
                if i < len(boxes) and (boxes[i][3] - boxes[i][1]) >= min_person_h:
                    union_fg |= mi
        person_union_all.append(_pack(union_all))
        person_union_fg.append(_pack(union_fg))
        person_boxes.append(boxes)

        # --- visible faces (MediaPipe) ---
        det = fd.process(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
        fbs = []
        if det.detections:
            for d in det.detections:
                bb = d.location_data.relative_bounding_box
                x1 = max(0, int(bb.xmin * W)); y1 = max(0, int(bb.ymin * H))
                x2 = min(W, int((bb.xmin + bb.width) * W))
                y2 = min(H, int((bb.ymin + bb.height) * H))
                if x2 > x1 and y2 > y1:
                    fbs.append([x1, y1, x2, y2, float(d.score[0])])
        face_boxes.append(np.array(fbs, np.float32) if fbs else np.zeros((0, 5), np.float32))
        idx += 1
    cap.release()

    np.savez_compressed(
        out_npz,
        person_union=np.array(person_union_fg, dtype=object),      # default = foreground
        person_union_all=np.array(person_union_all, dtype=object),
        person_boxes=np.array(person_boxes, dtype=object),
        face_boxes=np.array(face_boxes, dtype=object),
        hw=np.array(hw), frames=idx,
        meta=np.array([gt_weights, str(gt_imgsz), str(gt_conf), str(min_person_h)], dtype=object),
    )
    n_faces = sum(len(f) for f in face_boxes)
    print(f"[pseudo_gt] {idx} frames | face detections total={n_faces} "
          f"| mean persons={np.mean([len(b) for b in person_boxes]):.2f}")
    return out_npz
