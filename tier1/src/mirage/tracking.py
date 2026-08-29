import os
import json
import time
import uuid
import numpy as np
import cv2

from .encryption import rsa_encrypt_key, aes_gcm_encrypt, encode_crop
from .pose import LK_PARAMS


class PersonState:
    """Tracks one person stream across frames."""

    def __init__(self, rsa_public_key, enc_output_dir: str,
                 embedder, benchmark: bool = False, gender_classifier=None):

        self.prev_nose       = None
        self.movement_tier   = "medium"
        self.face_size_tier  = "medium"
        self.face_mesh_pts   = None
        self.inter_eye_px    = 0.0
        self.stream_id       = str(uuid.uuid4())
        self.aes_key         = os.urandom(16)

        self.best_confidence = 0.0
        self.best_rank_score = 0.0  # confidence * face_quality -- see update_best()
        self.best_face_crop  = None
        self.best_body_crop  = None
        self.best_frame_idx  = -1
        self.best_bbox_face  = None
        self.best_bbox_body  = None
        # (left_eye, right_eye, nose), each in best_face_crop-LOCAL pixel
        # coords (not full-frame coords -- best_face_crop is the only image
        # this state retains). None if the winning frame had no face crop.
        # Used by gender.py's predict_from_keypoints() for proper eye-line
        # alignment instead of the weaker bbox-center-scale fallback.
        self.best_face_eyes_nose = None

        # Per-frame encrypted body-crop archive -- separate from
        # best_body_crop above (which stays best-frame-only, used only as a
        # fallback/reference, never for restoration). This is what actually
        # lets Tier 3 approval restore the real video across every frame the
        # person appeared in, not just one still frame -- see docs on
        # flush_to_disk()'s extended .packet format. Encrypted incrementally
        # as frames arrive (append_frame(), called every full frame from
        # pipeline.py) rather than buffered raw and encrypted at flush time,
        # so peak memory doesn't hold a whole clip's worth of unencrypted
        # crops at once.
        self._frame_records = []  # list of (frame_idx, bbox, nonce, ciphertext)

        self._embedder         = embedder
        self._gender_classifier = gender_classifier
        self._enc_output_dir   = enc_output_dir
        self._benchmark        = benchmark

        if not benchmark:
            wrapped_key = rsa_encrypt_key(self.aes_key, rsa_public_key)
            key_path    = os.path.join(enc_output_dir, f"stream_{self.stream_id}.key")
            with open(key_path, 'wb') as f:
                f.write(wrapped_key)

    def append_frame(self, frame_idx, body_crop, body_bbox):
        """Encrypts and records one frame's body crop for the per-frame
        restoration archive. Call every full frame this person is tracked,
        alongside update_best() -- independent of it, since restoration
        needs every frame, not just the single best one update_best()
        selects for the embedding."""
        if self._benchmark or body_crop is None or body_crop.size == 0:
            return
        crop_bytes = encode_crop(body_crop)
        nonce, ciphertext = aes_gcm_encrypt(self.aes_key, crop_bytes)
        self._frame_records.append((frame_idx, body_bbox, nonce, ciphertext))

    def update_best(self, frame_idx, confidence,
                    face_crop, face_bbox, body_crop, body_bbox,
                    face_quality: float = 1.0, face_eyes_nose=None):
        # Ranked by confidence * face_quality, not confidence alone: raw
        # keypoint confidence comes from BODY pose (shoulders/hips/knees --
        # see pose.compute_frame_confidence), so a frame can score high on a
        # clear, unobstructed body while the face itself is turned away or
        # in profile. face_quality (face_canonical.face_quality_from_yaw)
        # downweights those frames so a confidently-tracked-but-turned-away
        # frame can no longer win "best embedding source" over a modestly-
        # confident frontal one. Callers who don't have a yaw estimate pass
        # the default 1.0, which reduces to the old confidence-only ranking.
        rank_score = confidence * face_quality
        if rank_score > self.best_rank_score:
            self.best_rank_score = rank_score
            self.best_confidence = confidence
            self.best_face_crop  = face_crop.copy() if face_crop is not None else None
            self.best_body_crop  = body_crop.copy() if body_crop is not None else None
            self.best_frame_idx  = frame_idx
            self.best_bbox_face  = face_bbox
            self.best_bbox_body  = body_bbox
            self.best_face_eyes_nose = face_eyes_nose

    def flush_to_disk(self) -> tuple[float, float, str | None, float | None]:
        if self._benchmark:
            return 0.0, 0.0, None, None

        # A stream with no best crops never called append_frame() with a
        # real crop either (both derive from the same per-frame body_crop
        # in pipeline.py's per-person loop), so _frame_records is empty
        # here too -- this guard doesn't need its own separate check.
        if self.best_body_crop is None and self.best_face_crop is None:
            return 0.0, 0.0, None, None

        t_emb0    = time.time()
        embedding = None
        if self._embedder is not None and self.best_face_crop is not None:
            embedding = self._embedder.extract(self.best_face_crop)
        embed_time = time.time() - t_emb0

        # Gender classification rides the same best_face_crop the embedding
        # uses (selected by update_best()'s confidence * face_quality
        # ranking) rather than tracking its own best frame -- one flag per
        # stream, not per frame, matching how the embedding is also a
        # single best-frame value.
        gender_label, gender_conf = None, None
        if self._gender_classifier is not None and self.best_face_crop is not None:
            if self.best_face_eyes_nose is not None:
                left_eye, right_eye, nose = self.best_face_eyes_nose
                gender_result = self._gender_classifier.predict_from_keypoints(
                    self.best_face_crop, left_eye, right_eye, nose
                )
            else:
                gender_result = self._gender_classifier.predict(self.best_face_crop)
            if gender_result is not None:
                gender_label, gender_conf = gender_result

        t_enc0 = time.time()

        face_bytes = encode_crop(self.best_face_crop) if self.best_face_crop is not None else b""
        body_bytes = encode_crop(self.best_body_crop) if self.best_body_crop is not None else b""
        meta_bytes = json.dumps({
            "stream_id"    : self.stream_id,
            "frame_idx"    : self.best_frame_idx,
            "confidence"   : round(self.best_confidence, 4),
            "bbox_body"    : self.best_bbox_body,
            "bbox_face"    : self.best_bbox_face,
            "has_embedding": embedding is not None,
            "embedding_dim": 512 if embedding is not None else 0,
            "gender"       : gender_label,
            "gender_conf"  : round(gender_conf, 4) if gender_conf is not None else None,
        }).encode("utf-8")
        emb_bytes = embedding.astype(np.float32).tobytes() if embedding is not None else b""

        def _enc(data: bytes):
            if not data:
                return b"", b""
            return aes_gcm_encrypt(self.aes_key, data)

        face_nonce, face_ct = _enc(face_bytes)
        body_nonce, body_ct = _enc(body_bytes)
        meta_nonce, meta_ct = _enc(meta_bytes)
        emb_nonce,  emb_ct  = _enc(emb_bytes)

        packet_path = os.path.join(
            self._enc_output_dir, f"stream_{self.stream_id}.packet"
        )

        def write_blob(f, nonce, ct):
            f.write(len(nonce).to_bytes(4, "little"))
            f.write(nonce)
            f.write(len(ct).to_bytes(4, "little"))
            f.write(ct)

        # Format: the original 4 fixed blobs (face, body-BEST-frame-only,
        # meta, embedding) as a header -- unchanged, so anything that only
        # ever read these 4 blobs (e.g. Tier 3 matching, which only touches
        # the embedding blob) keeps working unmodified -- followed by a
        # frame-count u32 and that many per-frame records:
        # {frame_idx: u32, bbox: 4x i32 (or 4x -1 if bbox is None), nonce,
        # ciphertext}. This is the real per-frame restoration archive (see
        # append_frame()) -- a strict superset of the old format, appended
        # after it, never replacing it.
        with open(packet_path, "wb") as f:
            write_blob(f, face_nonce, face_ct)
            write_blob(f, body_nonce, body_ct)
            write_blob(f, meta_nonce, meta_ct)
            write_blob(f, emb_nonce,  emb_ct)

            f.write(len(self._frame_records).to_bytes(4, "little"))
            for frame_idx, bbox, nonce, ct in self._frame_records:
                f.write(int(frame_idx).to_bytes(4, "little", signed=True))
                bbox_vals = bbox if bbox is not None else (-1, -1, -1, -1)
                for v in bbox_vals:
                    f.write(int(v).to_bytes(4, "little", signed=True))
                write_blob(f, nonce, ct)

        enc_time = time.time() - t_enc0
        return enc_time, embed_time, gender_label, gender_conf


def propagate_bboxes(last_bboxes, prev_gray, curr_gray):
    if last_bboxes is None or len(last_bboxes) == 0:
        return last_bboxes
    centers = []
    for bbox in last_bboxes:
        cx = (bbox[0] + bbox[2]) / 2.0
        cy = (bbox[1] + bbox[3]) / 2.0
        centers.append([[cx, cy]])
    old_pts  = np.array(centers, dtype=np.float32)
    new_pts, status, _ = cv2.calcOpticalFlowPyrLK(
        prev_gray, curr_gray, old_pts, None, **LK_PARAMS
    )
    updated = last_bboxes.copy().astype(np.float32)
    for i, (old_c, new_c, st) in enumerate(zip(old_pts, new_pts, status)):
        if st[0] == 1:
            dx = new_c[0][0] - old_c[0][0]
            dy = new_c[0][1] - old_c[0][1]
            updated[i][0] += dx
            updated[i][1] += dy
            updated[i][2] += dx
            updated[i][3] += dy
    return updated
