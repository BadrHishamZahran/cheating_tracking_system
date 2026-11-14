"""Rule-based 'looking sideways' detection from COCO pose keypoints (no training needed).

Idea: when the head turns left/right, the nose slides toward one ear. We measure how far the
nose is from the middle of the two ears, divided by the ear distance (scale-invariant).
If both ears are not reliable, we fall back to nose offset from the shoulder midpoint.
Looking DOWN at your own paper changes the vertical position only, so it is NOT flagged.

Known limits: eye-only glances (head still) and long hair covering the ears can't be
detected reliably. Tune the thresholds on your own videos with config "debug_overlay": true.
"""
import statistics
from collections import defaultdict, deque


def _ok(k, name, thr):
    v = k.get(name)
    return v is not None and v[2] >= thr


def head_yaw_ratio(k, kpt_thr, ear_thr, shoulder_thr):
    """k: {name: (x, y, conf)}. Returns (signed_ratio, threshold_for_that_source) or None."""
    if not _ok(k, "nose", kpt_thr):
        return None
    nx = k["nose"][0]
    if _ok(k, "l_ear", kpt_thr) and _ok(k, "r_ear", kpt_thr):
        lx, rx = k["l_ear"][0], k["r_ear"][0]
        d = abs(lx - rx)
        if d > 1e-3:
            return (nx - (lx + rx) / 2) / d, ear_thr
    if _ok(k, "l_shoulder", kpt_thr) and _ok(k, "r_shoulder", kpt_thr):
        lx, rx = k["l_shoulder"][0], k["r_shoulder"][0]
        d = abs(lx - rx)
        if d > 1e-3:
            return (nx - (lx + rx) / 2) / d, shoulder_thr
    return None


class HeadTurnDetector:
    def __init__(self, kpt_thr=0.5, ear_ratio_thr=0.35, shoulder_ratio_thr=0.2, smooth_frames=5):
        self.kpt_thr, self.ear_thr, self.sh_thr = kpt_thr, ear_ratio_thr, shoulder_ratio_thr
        self._hist = defaultdict(lambda: deque(maxlen=smooth_frames))
        self.last = {}          # track_id -> last smoothed normalized yaw (1.0 == threshold)

    def update(self, track_id, kpts):
        """Returns a confidence in [0.5, 1.0] if the head is turned past the threshold, else None."""
        r = head_yaw_ratio(kpts, self.kpt_thr, self.ear_thr, self.sh_thr)
        if r is None:
            return None
        ratio, thr = r
        h = self._hist[track_id]
        h.append(ratio / thr)                 # normalize: 1.0 means "exactly at the threshold"
        m = statistics.median(h)              # median smoothing removes single-frame jitter
        self.last[track_id] = m
        if abs(m) < 1.0:
            return None
        return round(min(1.0, 0.5 + 0.5 * (abs(m) - 1.0)), 3)

    def reset(self):
        self._hist.clear()
        self.last.clear()
