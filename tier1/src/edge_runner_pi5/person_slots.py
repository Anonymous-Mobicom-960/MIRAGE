"""
person_slots -- auto person-count + stable position-based person-slot identity for MIRAGE Tier-1.
=================================================================================================
Pure-stdlib (no numpy / onnx / model deps -- unit-testable off-device, see test_tier1_npersons.py
at the repo root). Nothing here is clip-specific: the person count is derived from THIS clip's own
detector stream, and every threshold is a config knob (frames / fraction-of-width, never absolute
pixels measured on one clip).

Three pieces:

0) slot_groups(log, slot_log)
   The bridge between a per-frame emit log ([frames][persons]) and the whole-clip, PER-PERSON
   processing stages (pose anonymization, face-scalar DP). Groups every (frame, person-index)
   cell by its STABLE slot id, so those stages operate on one person's real trajectory instead
   of on a positional index that shifts whenever someone is missed for a frame.

1) estimate_person_count(counts, sustain_frames)
   How many people are in THIS clip, derived from the detector's own per-frame person counts
   (never hardcoded). Robust statistic: N = the largest n such that the per-frame count stays
   >= n for at least `sustain_frames` CONSECUTIVE frames. A spurious extra detection lasting
   < sustain_frames cannot inflate N; a real person present for even a brief walk-through
   (>= the sustain window) IS counted -- unlike a whole-clip percentile, which misses short
   entrants entirely.

2) SlotTracker
   Per-frame assignment of anonymous person detections to STABLE slot indices by screen
   position -- slot 0 is the LEFTMOST person ("left stays left, right stays right", user spec
   2026-07-22) -- with swap HYSTERESIS so detector x-jitter or a brief near-crossing does not
   flip slot identity frame-to-frame.

   Policy ("x-sorted+hysteresis"), fully deterministic:
     * first frame / count change: slots adopt pure x-sorted order (slot 0 = leftmost);
     * frame-to-frame continuity: greedy globally-nearest matching of detections to each
       slot's last x-center (ties broken by lowest slot index, then lowest detection index);
     * if the matched slots end up OUT of left-to-right index order (someone crossed), the
       renumbering back to x-order is only COMMITTED when the inversion persists for
       >= hysteresis_frames CONSECUTIVE frames with the SAME implied permutation, OR when an
       inverted adjacent pair is separated by more than swap_margin_frac * frame_width
       (an unambiguous real crossing -- no need to wait);
     * until committed, slots keep their previous identity (the transient is suppressed).

   DOCUMENTED LIMITS (position-based by design):
     * a SUSTAINED full crossing DOES swap identities -- after the hysteresis window the
       leftmost person is slot 0 again. That is the spec ("left stays left"), not a bug;
     * detections are anonymous x-points (no appearance / velocity model): two people meeting
       at the same x are inherently ambiguous and identity may hand over at the meeting point;
     * a frame where the detection COUNT changes (entry / exit / 1-frame dropout) resets the
       tracker to pure x-sorted order and clears any suppressed swap -- simple, deterministic.
"""


def slot_groups(log, slot_log=None):
    """Group a per-frame emit log by STABLE slot id.

    log:      [frames][persons] -- any per-frame list-of-people structure (pose_log, face_log).
    slot_log: [frames][persons] of slot ids, parallel to `log` (as emitted by mirage_tier1's
              SlotTracker). None (or a short/ragged entry) => POSITIONAL fallback, i.e. the
              old "person p == index p in every frame" proxy. The fallback keeps every
              pre-slot caller (and pose_anon_edge's own back-compat path) working unchanged.

    Returns {slot_id: [(frame_index, person_index), ...]} with the cell list in frame order.

    WHY this exists: whole-clip stages (pose anonymization, face-scalar DP) must see ONE
    person's real trajectory. With the positional proxy, a single missed detection shifts
    every later index by one and the stage silently splices two different people into one
    "person" -- which for the anonymizer means one collapse template + one cadence warp over
    a spliced sequence, and for DP means smoothing across an identity boundary.
    """
    groups = {}
    for i, frame in enumerate(log):
        row = None
        if slot_log is not None and i < len(slot_log):
            row = slot_log[i]
        for j in range(len(frame)):
            s = int(row[j]) if (row is not None and j < len(row)) else j
            groups.setdefault(s, []).append((i, j))
    return groups


def estimate_person_count(counts, sustain_frames=8):
    """Robust clip-level person count from per-frame detection counts.

    counts: iterable of per-frame person-detection counts (UNTRUNCATED detector output).
    sustain_frames: robustness knob (config.PCOUNT_SUSTAIN_FRAMES) -- a count level n only
      registers if it holds for this many CONSECUTIVE frames. Clips shorter than the window
      use their own length (otherwise a short clip could never register anyone).

    Returns the largest n >= 1 sustained for >= the window, else 0.

    ⚠️ THIS IS MAX SUSTAINED *CONCURRENT* COUNT, NOT "how many people appear in this clip".
    Three people who cross the frame ONE AT A TIME, never overlapping, give 1 — not 3. That is
    the correct answer for the question the mask stages ask ("how many people must I handle at
    once?") and the WRONG answer for the question the name suggests. It matters downstream:
    `person_count` is emitted in pose.json meta and consumed by the cloud graph's
    MIRAGEMaskPersonCount, so a clip with sequential entrants asks the cloud for fewer characters
    than the clip actually contains.
    It is NOT a §2 risk — the mask net runs off the untruncated per-frame detections and never
    consults this number — but a caller wanting "distinct people over the clip" should use the
    PERSISTENT TRACK ids (`assign_tracks`, reported as `distinct_tracks`), which do count
    sequential entrants separately. Flagged 2026-07-23; left as-is because the mask stages depend
    on the concurrent semantics.
    """
    counts = [int(c) for c in counts]
    if not counts:
        return 0
    k = max(1, min(int(sustain_frames), len(counts)))
    for n in range(max(counts), 0, -1):
        run = 0
        for c in counts:
            run = run + 1 if c >= n else 0
            if run >= k:
                return n
    return 0


class SlotTracker:
    """Stable position-based person slots with swap hysteresis (policy: module docstring)."""

    def __init__(self, hysteresis_frames=4, swap_margin_frac=0.04,
                 track_gate_frac=0.15, track_ttl=8):
        self.h = max(1, int(hysteresis_frames))        # frames a new order must persist
        self.margin_frac = float(swap_margin_frac)     # immediate-commit gap, frac of frame width
        self.lane_x = []          # lane_x[s] = last x-center of the person currently in slot s
        self.pending = None       # candidate renumbering permutation awaiting confirmation
        self.pending_n = 0        # consecutive frames the candidate has persisted
        # --- PERSISTENT TRACK IDS (added 2026-07-23 after a measured defect) ------------------
        # `assign()` returns POSITIONAL slots and RENUMBERS on every count change (its documented
        # policy). That is right for "left stays left", but it means the emitted id carries no
        # identity: with 3 people, when the MIDDLE one drops, the third person is emitted at
        # index 1 — the index that was the second person's — and any whole-clip stage grouping by
        # it splices two people together. Measured: emitted rows were only "01"/"012", never a gap
        # like "02", so slot ids were provably positional and grouping by them was identical to
        # the positional fallback it was supposed to replace.
        # Track ids fix that: they SURVIVE count changes, so an absent person leaves a GAP rather
        # than renumbering everyone after them. Slots stay positional for downstream (the cloud
        # graph's p1=RIGHT / p2=LEFT convention); tracks are what per-person stages group by.
        self.gate_frac = float(track_gate_frac)   # max |dx| to keep a track, frac of frame width
        self.ttl = max(1, int(track_ttl))         # frames a track survives while unmatched
        self.tracks = []          # [{"id": int, "x": float, "miss": int}], newest-last
        self._next_id = 0

    def assign_tracks(self, x_centers, frame_width):
        """One frame -> (order, track_ids).

        `order` is exactly what assign() returns (positional slots, unchanged semantics).
        `track_ids[j]` is the PERSISTENT identity of the person in slot j — stable across count
        changes, so a dropped person leaves a gap instead of renumbering everyone after them.

        Matching is greedy-nearest against each live track's last x, gated at
        `track_gate_frac * frame_width` so a track is never stolen by someone on the other side of
        the frame. An unmatched track is kept for `track_ttl` frames (a 1-frame detector dropout
        must not destroy identity) and then retired. An unmatched detection starts a new track.
        """
        xs = [float(x) for x in x_centers]
        order = self.assign(xs, frame_width)          # positional slots, semantics untouched
        gate = self.gate_frac * float(frame_width)

        pairs = sorted((abs(xs[d] - t["x"]), ti, d)
                       for ti, t in enumerate(self.tracks) for d in range(len(xs))
                       if abs(xs[d] - t["x"]) <= gate)
        tid_of_det = {}
        used_t, used_d = set(), set()
        for _dist, ti, d in pairs:
            if ti in used_t or d in used_d:
                continue
            used_t.add(ti); used_d.add(d)
            self.tracks[ti]["x"] = xs[d]; self.tracks[ti]["miss"] = 0
            tid_of_det[d] = self.tracks[ti]["id"]
        for ti, t in enumerate(self.tracks):          # unmatched tracks age out
            if ti not in used_t:
                t["miss"] += 1
        self.tracks = [t for t in self.tracks if t["miss"] <= self.ttl]
        for d in range(len(xs)):                      # unmatched detections start a track
            if d not in tid_of_det:
                self.tracks.append({"id": self._next_id, "x": xs[d], "miss": 0})
                tid_of_det[d] = self._next_id
                self._next_id += 1
        return order, [tid_of_det[d] for d in order]

    def assign(self, x_centers, frame_width):
        """One frame: x_centers (any order) -> `order`, where order[j] = index into x_centers
        of the detection assigned to slot j. len(order) == len(x_centers) always (the caller
        may truncate to its emit budget AFTER slot assignment)."""
        xs = [float(x) for x in x_centers]
        K = len(xs)
        if K != len(self.lane_x):
            # count change (entry/exit/dropout): deterministic reset to pure x-sorted order.
            order = sorted(range(K), key=lambda i: (xs[i], i))
            self.lane_x = [xs[i] for i in order]
            self.pending = None
            self.pending_n = 0
            return order
        if K == 0:
            return []
        # (1) continuity: greedy globally-nearest detection<->slot matching on last x-centers.
        pairs = sorted((abs(xs[d] - self.lane_x[s]), s, d)
                       for s in range(K) for d in range(K))
        det_of = [-1] * K
        used = [False] * K
        matched = 0
        for _dist, s, d in pairs:
            if det_of[s] == -1 and not used[d]:
                det_of[s] = d
                used[d] = True
                matched += 1
                if matched == K:
                    break
        self.lane_x = [xs[det_of[s]] for s in range(K)]
        # (2) positional renumbering with hysteresis: slots should sit in left-to-right index
        # order; an inversion means a crossing (or jitter) -- commit the re-sort only when
        # sustained or unambiguous, else keep the previous identity.
        perm = tuple(sorted(range(K), key=lambda s: (self.lane_x[s], s)))
        if perm == tuple(range(K)):
            self.pending = None
            self.pending_n = 0
            return det_of
        if perm == self.pending:
            self.pending_n += 1
        else:
            self.pending = perm
            self.pending_n = 1
        max_inv_gap = max(self.lane_x[s] - self.lane_x[s + 1]
                          for s in range(K - 1) if self.lane_x[s] > self.lane_x[s + 1])
        if max_inv_gap > self.margin_frac * float(frame_width) or self.pending_n >= self.h:
            self.lane_x = [self.lane_x[s] for s in perm]      # commit: renumber to x-order
            out = [det_of[s] for s in perm]
            self.pending = None
            self.pending_n = 0
            return out
        return det_of                                          # suppressed transient
