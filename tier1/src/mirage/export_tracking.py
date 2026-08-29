"""
Stable left-to-right person-slot tracking for the dense per-frame export
mode (see process_video's export_dir/dense_export kwargs), and for
PersonState's encryption/consent-recovery path (see PersonIdentityTracker
below).

Both trackers solve the same underlying problem: rtmlib's det_model()/
pose_model() (and yolo_seg's own detect head) return each frame's people in
whatever order the underlying model happened to produce that frame -- NOT a
stable, identity-persistent order. Naively keying anything long-lived (a
PersonState, an export slot) by raw per-frame array index silently breaks
the moment detection order reshuffles: index 0 can refer to a different
physical person on frame N+1 than it did on frame N, with no error or
signal that this happened. Confirmed as a real bug this way: in a 3-person
clip, PersonState's old raw-index keying let two different encrypted
streams both end up with their best-confidence face crop pulled from the
SAME middle physical person (whichever slot they happened to occupy on
their own best-confidence frame), corrupting which stream's embedding
actually represents which bystander -- exactly the data Tier 3 matching and
restoration depend on being correct.

ExportSlotTracker (small, FIXED slot count, e.g. 0/1/2) and
PersonIdentityTracker (unbounded, IDs created/retired as people appear/
leave) both use the same greedy nearest-centroid matching, just over
different-shaped identity spaces.
"""

import numpy as np


class ExportSlotTracker:
    def __init__(self, n_slots: int, frame_w: int, frame_h: int,
                 grace_frames: int = 30, gate_frac: float = 0.25):
        self.n_slots      = n_slots
        self.gate         = gate_frac * float(np.hypot(frame_w, frame_h))
        self.grace_frames = grace_frames
        self.center       = [None] * n_slots   # (cx, cy) or None
        self.missed       = [0] * n_slots
        self.active       = [False] * n_slots

    def assign(self, detections):
        """
        detections: list of (det_idx, (cx, cy)) for this frame's people.
        det_idx indexes that frame's keypoints/scores arrays.

        Returns {slot_idx: det_idx} for slots matched this frame. Slots
        not in the returned dict are either reserved-but-missed (still
        within grace) or genuinely free.
        """
        pairs = []
        for s in range(self.n_slots):
            if not self.active[s] or self.center[s] is None:
                continue
            for det_idx, c in detections:
                d = float(np.hypot(c[0] - self.center[s][0], c[1] - self.center[s][1]))
                if d <= self.gate:
                    pairs.append((d, s, det_idx, c))
        pairs.sort(key=lambda p: p[0])

        matched_slots = {}
        used_slots, used_det = set(), set()
        for d, s, det_idx, c in pairs:
            if s in used_slots or det_idx in used_det:
                continue
            matched_slots[s] = det_idx
            used_slots.add(s)
            used_det.add(det_idx)
            self.center[s] = c
            self.missed[s] = 0

        for s in range(self.n_slots):
            if self.active[s] and s not in used_slots:
                self.missed[s] += 1
                if self.missed[s] > self.grace_frames:
                    self.active[s] = False
                    self.center[s] = None
                    self.missed[s] = 0

        leftover = sorted(
            ((di, c) for di, c in detections if di not in used_det),
            key=lambda x: x[1][0],
        )
        free_slots = [s for s in range(self.n_slots) if not self.active[s]]
        for (det_idx, c), s in zip(leftover, free_slots):
            self.active[s] = True
            self.center[s] = c
            self.missed[s] = 0
            matched_slots[s] = det_idx

        return matched_slots


class PersonIdentityTracker:
    """
    Same greedy nearest-centroid matching as ExportSlotTracker, generalized
    to an UNBOUNDED, dynamically growing/shrinking set of stable identities
    (uuid4 stream ids) instead of a small fixed slot count -- what
    tracking.py's PersonState population actually needs, since a clip can
    have any number of people appear/depart over its length, not a fixed
    small N known up front.

    Same tolerance-to-brief-misses behavior as ExportSlotTracker
    (grace_frames), so a person who's briefly undetected (motion blur,
    occlusion) doesn't get treated as departed-and-replaced the moment
    their detection drops out for one frame -- their stream_id (and its
    PersonState) is held open through the grace window instead.
    """

    def __init__(self, frame_w: int, frame_h: int,
                 grace_frames: int = 30, gate_frac: float = 0.25):
        self.gate         = gate_frac * float(np.hypot(frame_w, frame_h))
        self.grace_frames = grace_frames
        self._next_id     = 0
        self.center: dict = {}   # identity_id -> (cx, cy)
        self.missed: dict = {}   # identity_id -> consecutive frames unmatched

    def assign(self, detections):
        """
        detections: list of (det_idx, (cx, cy)) for this frame's people.

        Returns (matched, departed):
          matched: {identity_id: det_idx} for every identity active this
                   frame (existing, re-matched by nearest centroid within
                   gate distance, OR newly created for an unmatched
                   detection).
          departed: list of identity_ids that just exceeded grace_frames of
                    consecutive misses this frame -- caller should flush and
                    retire these (mirrors ExportSlotTracker's per-slot
                    grace/retire logic, just returned explicitly here since
                    there's no fixed slot list to re-scan for "went
                    inactive").
        """
        pairs = []
        for identity_id, c0 in self.center.items():
            for det_idx, c in detections:
                d = float(np.hypot(c[0] - c0[0], c[1] - c0[1]))
                if d <= self.gate:
                    pairs.append((d, identity_id, det_idx, c))
        pairs.sort(key=lambda p: p[0])

        matched: dict = {}
        used_ids, used_det = set(), set()
        for d, identity_id, det_idx, c in pairs:
            if identity_id in used_ids or det_idx in used_det:
                continue
            matched[identity_id] = det_idx
            used_ids.add(identity_id)
            used_det.add(det_idx)
            self.center[identity_id] = c
            self.missed[identity_id] = 0

        departed = []
        for identity_id in list(self.center.keys()):
            if identity_id in used_ids:
                continue
            self.missed[identity_id] = self.missed.get(identity_id, 0) + 1
            if self.missed[identity_id] > self.grace_frames:
                departed.append(identity_id)
                del self.center[identity_id]
                del self.missed[identity_id]

        # Leftover detections (not matched to any existing identity, within
        # gate distance) become brand-new identities -- sorted left-to-right
        # only for deterministic/reproducible id assignment order, not for
        # any positional meaning (unlike ExportSlotTracker's fixed slots).
        leftover = sorted(
            ((di, c) for di, c in detections if di not in used_det),
            key=lambda x: x[1][0],
        )
        for det_idx, c in leftover:
            identity_id = self._next_id
            self._next_id += 1
            self.center[identity_id] = c
            self.missed[identity_id] = 0
            matched[identity_id] = det_idx

        return matched, departed

        return matched_slots
