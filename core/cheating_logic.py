"""Pure decision logic (no models, no UI) so it can be unit-tested."""
import math
from collections import defaultdict, deque
from dataclasses import dataclass, field


@dataclass
class Student:
    track_id: int
    box: tuple                      # (x1, y1, x2, y2)
    conf: float = 0.0
    wrists: list = field(default_factory=list)   # [(x, y), ...] confident wrists only
    kpts: dict = field(default_factory=dict)     # {name: (x, y, conf)} head/shoulder keypoints


def associate_phone(phone_box, students, max_ratio=0.6):
    """Return the track_id of the student who most likely holds the phone, or None.

    Wrist keypoints are the strongest signal. If a student has no visible wrists,
    fall back to 'phone center is inside / close to the student box'.
    Distances are normalized by the student's box height so it works at any zoom.
    """
    px = (phone_box[0] + phone_box[2]) / 2
    py = (phone_box[1] + phone_box[3]) / 2
    best_id, best_score = None, float("inf")
    for s in students:
        x1, y1, x2, y2 = s.box
        h = max(1.0, y2 - y1)
        if s.wrists:
            d = min(math.hypot(px - wx, py - wy) for wx, wy in s.wrists) / h
            if d > max_ratio:
                continue
            score = d
        else:
            dx = max(x1 - px, 0, px - x2)
            dy = max(y1 - py, 0, py - y2)
            if math.hypot(dx, dy) / h > 0.05:      # not inside the box
                continue
            cx, cy = (x1 + x2) / 2, (y1 + y2) / 2
            score = 0.5 + math.hypot(px - cx, py - cy) / h   # penalty vs wrist match
        if score < best_score:
            best_id, best_score = s.track_id, score
    return best_id


def iou(a, b):
    x1, y1 = max(a[0], b[0]), max(a[1], b[1])
    x2, y2 = min(a[2], b[2]), min(a[3], b[3])
    inter = max(0.0, x2 - x1) * max(0.0, y2 - y1)
    ua = (a[2] - a[0]) * (a[3] - a[1]) + (b[2] - b[0]) * (b[3] - b[1]) - inter
    return inter / ua if ua > 0 else 0.0


def match_person_box(box, students, min_iou=0.3):
    """For classes whose box covers a whole student (e.g. 'cheating' drawn around the person):
    return the tracked student with the highest IoU, or None."""
    best_id, best = None, min_iou
    for s in students:
        v = iou(box, s.box)
        if v >= best:
            best_id, best = s.track_id, v
    return best_id


class CheatingLogic:
    """Flags a track after N consecutive suspicious frames, with a cooldown per track."""

    def __init__(self, consecutive_frames=10, cooldown_seconds=30,
                 hold_red_seconds=3, max_gap_frames=3):
        self.n = consecutive_frames
        self.cooldown = cooldown_seconds
        self.hold = hold_red_seconds
        self.max_gap = max_gap_frames
        self.streak = defaultdict(int)
        self.gap = defaultdict(int)
        self.confs = defaultdict(lambda: deque(maxlen=consecutive_frames))
        self.last_alert = {}
        self.red_until = {}

    def update(self, track_id, behavior, confidence, t):
        """behavior: str or None. t: video time in seconds. Returns an event dict or None."""
        if behavior:
            self.streak[track_id] += 1
            self.gap[track_id] = 0
            self.confs[track_id].append(confidence)
        else:
            self.gap[track_id] += 1
            if self.gap[track_id] > self.max_gap:
                self.streak[track_id] = 0
                self.confs[track_id].clear()

        if self.streak[track_id] < self.n:
            return None

        self.red_until[track_id] = t + self.hold
        if t - self.last_alert.get(track_id, -1e9) < self.cooldown:
            return None
        self.last_alert[track_id] = t
        c = self.confs[track_id]
        return {"track_id": track_id, "behavior": behavior or "Phone detected",
                "confidence": round(sum(c) / len(c), 3) if c else 0.0}

    def is_cheating(self, track_id, t):
        return t < self.red_until.get(track_id, -1e9)
