#!/usr/bin/env python3
"""
eval_tier1_models.py - ablation for the on-phone Tier-1 building blocks.
=======================================================================
The shipped app runs the GUIDED pipeline (YOLO11n-seg mask + MoveNet pose). This measures the two
model choices against their edge counterparts on the SAME clips/device (laptop CPU EP), so the choice
is grounded, not asserted:

  MASK   RVM (mobilenetv3 matte)   vs   YOLO11n-seg (the guided-seg primary)
         -> person coverage, mutual IoU (agreement), latency ms/frame
  POSE   RTMPose-t-wholebody (edge) vs   MoveNet-MultiPose (on-phone)
         -> body-17 keypoint agreement (mean px dist, % of image diagonal), latency ms/person|frame

    python eval_tier1_models.py [--clips d07_2person_1264,d03_final_10s,d06_input_video] [--frames 8]
"""
import argparse, json, os, time
import cv2, numpy as np, onnxruntime as ort

HERE = os.path.dirname(os.path.abspath(__file__)); REPO = os.path.dirname(HERE)
# The internal development tree still carries this directory under its pre-rename name, and that
# repository cannot simply be renamed: `tree/<x>` maps 1:1 onto the pod's `/workspace/<x>`, so
# moving it would create a second copy on the pod rather than rename the first. Accept either
# name, newest first, so the two repositories do not have to change in the same commit.
def _edge_root(repo, *tail):
    for _name in ("mirage_edge_deploy", "sitara_edge_deploy"):
        _p = os.path.join(repo, "tree", _name, "tier1_raspberry_pi5", *tail)
        if os.path.exists(_p):
            return _p
    return os.path.join(repo, "tree", "mirage_edge_deploy", "tier1_raspberry_pi5", *tail)


M = _edge_root(REPO, "models")
SEG = os.path.join(M, "yolo11n-seg.onnx"); RVM = os.path.join(M, "rvm_mobilenetv3_fp32.onnx")
RTM = os.path.join(M, "rtmpose-t-wholebody.onnx"); MOVE = os.environ.get("MIRAGE_MOVENET_ONNX") or os.path.join(
    REPO, "models", "movenet_multipose.onnx")
SK = [(5,7),(7,9),(6,8),(8,10),(5,6),(5,11),(6,12),(11,12),(11,13),(13,15),(12,14),(14,16)]


def sess(p): return ort.InferenceSession(p, providers=["CPUExecutionProvider"])


def yolo_mask_boxes(s, f, conf=0.35):
    H, W = f.shape[:2]; N = 640
    inp = cv2.resize(cv2.cvtColor(f, cv2.COLOR_BGR2RGB), (N, N)).astype(np.float32)/255.
    t = time.time(); o0, o1 = s.run(None, {s.get_inputs()[0].name: inp.transpose(2,0,1)[None]}); dt = (time.time()-t)*1000
    pred = o0[0].T; protos = o1[0].reshape(32,-1)
    box, cls, coef = pred[:,:4], pred[:,4:84], pred[:,84:116]
    keep = np.where(cls[:,0] > conf)[0]
    M160 = np.zeros((160,160), np.float32); boxes = []
    for i in keep:
        m = 1/(1+np.exp(-(coef[i]@protos))).reshape(160,160)
        cx,cy,w,h = box[i]/N*160; x1,y1,x2,y2 = int(cx-w/2),int(cy-h/2),int(cx+w/2),int(cy+h/2)
        mm = np.zeros_like(m); mm[max(0,y1):y2, max(0,x1):x2] = m[max(0,y1):y2, max(0,x1):x2]
        M160 = np.maximum(M160, (mm>0.5).astype(np.float32))
        bx = box[i]/N*[W,H,W,H]; boxes.append([bx[0]-bx[2]/2, bx[1]-bx[3]/2, bx[0]+bx[2]/2, bx[1]+bx[3]/2])
    mask = cv2.resize(M160, (W,H), interpolation=cv2.INTER_NEAREST) > 0.5
    # crude NMS on person boxes for pose input
    boxes = nms(np.array(boxes), 0.5) if boxes else np.zeros((0,4))
    return mask, boxes, dt


def rvm_mask(s, f, thr=0.5):
    H, W = f.shape[:2]; sc = 512/max(H,W); w,h = (int(W*sc)//4*4, int(H*sc)//4*4)
    src = cv2.cvtColor(cv2.resize(f,(w,h)), cv2.COLOR_BGR2RGB).astype(np.float32)/255.
    z = np.zeros([1,1,1,1], np.float32)
    t = time.time(); pha = s.run(['pha'], {'src':src.transpose(2,0,1)[None],'r1i':z,'r2i':z,'r3i':z,'r4i':z,'downsample_ratio':np.array([1.],np.float32)})[0][0,0]; dt=(time.time()-t)*1000
    return cv2.resize(pha,(W,H)) > thr, dt


def movenet(s, f, thr=0.2):
    H, W = f.shape[:2]; side = max(H,W); N=256
    sq = np.zeros((N,N,3), np.uint8); r = N/side
    sq[:int(H*r),:int(W*r)] = cv2.resize(cv2.cvtColor(f,cv2.COLOR_BGR2RGB),(int(W*r),int(H*r)))
    t=time.time(); out = s.run(None, {'input': sq.astype(np.int32)[None]})[0][0]; dt=(time.time()-t)*1000
    people = []
    for i in range(6):
        if out[i,55] < thr: continue
        kp = np.array([[out[i,k*3+1]*side, out[i,k*3]*side, out[i,k*3+2]] for k in range(17)])
        people.append(kp)
    return people, dt


def rtmpose(pose, f, boxes):
    if len(boxes) == 0: return [], 0.0
    t = time.time(); kpts, scs = pose(f, bboxes=boxes.tolist()); dt = (time.time()-t)*1000
    out = []
    for k, sc in zip(kpts, scs):
        body = np.concatenate([k[:17], sc[:17,None]], axis=1)  # (17, x,y,score)
        out.append(body)
    return out, dt/max(1,len(boxes))


def nms(boxes, iou_thr):
    if len(boxes)==0: return boxes
    a = boxes[(boxes[:,2]-boxes[:,0])*(boxes[:,3]-boxes[:,1]) > 0]
    order = np.argsort(-((a[:,2]-a[:,0])*(a[:,3]-a[:,1]))); keep=[]
    while len(order):
        i=order[0]; keep.append(i); rest=order[1:]
        if len(rest)==0: break
        xx1=np.maximum(a[i,0],a[rest,0]); yy1=np.maximum(a[i,1],a[rest,1])
        xx2=np.minimum(a[i,2],a[rest,2]); yy2=np.minimum(a[i,3],a[rest,3])
        inter=np.maximum(0,xx2-xx1)*np.maximum(0,yy2-yy1)
        ai=(a[i,2]-a[i,0])*(a[i,3]-a[i,1]); ar=(a[rest,2]-a[rest,0])*(a[rest,3]-a[rest,1])
        iou=inter/(ai+ar-inter+1e-6); order=rest[iou<iou_thr]
    return a[keep]


def pose_agreement(rtm_people, mv_people, diag):
    """Match RTM<->MoveNet people by centroid, mean body-17 keypoint dist (% of image diagonal)."""
    if not rtm_people or not mv_people: return None
    def cen(k): v=k[k[:,2]>0.3]; return v[:,:2].mean(0) if len(v) else np.array([1e9,1e9])
    dists=[]
    for rk in rtm_people:
        rc=cen(rk); mj=min(range(len(mv_people)), key=lambda j: np.hypot(*(cen(mv_people[j])-rc)))
        mk=mv_people[mj]
        both=[i for i in range(17) if rk[i,2]>0.3 and mk[i,2]>0.3]
        if both: dists.append(np.mean([np.hypot(*(rk[i,:2]-mk[i,:2])) for i in both]))
    return float(np.mean(dists)/diag*100) if dists else None


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--clips", default="d07_2person_1264,d03_final_10s,d06_input_video")
    ap.add_argument("--frames", type=int, default=8)
    a = ap.parse_args()
    from rtmlib import RTMPose
    sseg, srvm, smove = sess(SEG), sess(RVM), sess(MOVE)
    rtm = RTMPose(RTM, model_input_size=(192,256), backend="onnxruntime", device="cpu")

    agg = {"mask":{"rvm_cov":[], "yolo_cov":[], "iou":[], "rvm_ms":[], "yolo_ms":[]},
           "pose":{"rtm_ms":[], "move_ms":[], "agree_pct":[]}}
    for name in a.clips.split(","):
        p = os.path.join(REPO, "input", "reid_dataset", name + ".mp4")
        if not os.path.exists(p): print("skip missing", name); continue
        cap = cv2.VideoCapture(p); nf = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        idxs = np.linspace(0, max(0,nf-1), a.frames).astype(int)
        for fi in idxs:
            cap.set(cv2.CAP_PROP_POS_FRAMES, int(fi)); ok, f = cap.read()
            if not ok: continue
            H,W = f.shape[:2]; diag = float(np.hypot(H,W))
            ym, ybx, ydt = yolo_mask_boxes(sseg, f); rm, rdt = rvm_mask(srvm, f)
            inter = (ym & rm).sum(); union = (ym | rm).sum()
            agg["mask"]["yolo_cov"].append(ym.mean()*100); agg["mask"]["rvm_cov"].append(rm.mean()*100)
            agg["mask"]["iou"].append(inter/union*100 if union else 100.0)
            agg["mask"]["yolo_ms"].append(ydt); agg["mask"]["rvm_ms"].append(rdt)
            mv, mdt = movenet(smove, f); rp, rpdt = rtmpose(rtm, f, ybx)
            agg["pose"]["move_ms"].append(mdt); agg["pose"]["rtm_ms"].append(rpdt)
            ag = pose_agreement(rp, mv, diag)
            if ag is not None: agg["pose"]["agree_pct"].append(ag)
        cap.release(); print(f"  {name}: done ({len(idxs)} frames)")

    def stat(x): return {"mean": round(float(np.mean(x)),3), "n": len(x)} if x else {"mean": None, "n": 0}
    out = {"clips": a.clips, "frames_per_clip": a.frames,
           "MASK": {"rvm_coverage_pct": stat(agg["mask"]["rvm_cov"]), "yolo_coverage_pct": stat(agg["mask"]["yolo_cov"]),
                    "rvm_vs_yolo_IoU_pct": stat(agg["mask"]["iou"]),
                    "rvm_latency_ms": stat(agg["mask"]["rvm_ms"]), "yolo_latency_ms": stat(agg["mask"]["yolo_ms"])},
           "POSE": {"rtm_latency_ms_per_person": stat(agg["pose"]["rtm_ms"]), "movenet_latency_ms_per_frame": stat(agg["pose"]["move_ms"]),
                    "rtm_vs_movenet_body17_meandist_pct_of_diag": stat(agg["pose"]["agree_pct"])},
           "device": "laptop CPU EP (onnxruntime)"}
    print(json.dumps(out, indent=2))
    op = os.path.join(HERE, "reports", "TIER1_MODEL_ABLATION.json"); os.makedirs(os.path.dirname(op), exist_ok=True)
    json.dump(out, open(op,"w"), indent=2); print("wrote", op)


if __name__ == "__main__":
    main()
