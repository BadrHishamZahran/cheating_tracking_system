import os, sys, tempfile, json
sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))
import numpy as np
from core.head_pose import HeadTurnDetector, head_yaw_ratio
from core.cheating_logic import Student
from core.pipeline import CheatingPipeline
from db.local_db import LocalDB


def kp(nose_x, ears=(80, 120), sh=(40, 160), conf=0.9):
    d = {"nose": (nose_x, 60, conf), "l_eye": (nose_x - 8, 50, conf), "r_eye": (nose_x + 8, 50, conf)}
    if ears: d.update(l_ear=(ears[0], 55, conf), r_ear=(ears[1], 55, conf))
    if sh: d.update(l_shoulder=(sh[0], 140, conf), r_shoulder=(sh[1], 140, conf))
    return d


def test_facing_front_is_not_flagged():
    h = HeadTurnDetector()
    assert all(h.update(1, kp(100)) is None for _ in range(6))


def test_turned_head_is_flagged_and_looking_down_is_not():
    h = HeadTurnDetector()
    out = [h.update(1, kp(118)) for _ in range(5)]            # nose near the right ear
    assert out[-1] is not None and 0.5 <= out[-1] <= 1.0
    k = kp(100); k["nose"] = (100, 110, 0.9)                   # nose lower only (looking down)
    assert HeadTurnDetector().update(2, k) is None


def test_median_smoothing_ignores_single_spike():
    h = HeadTurnDetector(smooth_frames=5)
    for _ in range(4): h.update(1, kp(100))
    assert h.update(1, kp(125)) is None                        # one noisy frame


def test_fallback_to_shoulders_and_missing_keypoints():
    r = head_yaw_ratio(kp(100, ears=None), 0.5, 0.35, 0.2)
    assert r is not None and r[1] == 0.2
    assert head_yaw_ratio(kp(100, ears=None, sh=None), 0.5, 0.35, 0.2) is None
    assert head_yaw_ratio(kp(100, conf=0.1), 0.5, 0.35, 0.2) is None


class FakeDet:
    def __init__(self, nose_x): self.nose_x = nose_x
    def reset_tracker(self): pass
    def detect(self, frame):
        return [Student(5, (10, 10, 200, 220), 0.9, [], kp(self.nose_x))], []


def make_pipeline(nose_x, tmp):
    from core.config import load_config
    cfg = load_config(os.path.dirname(os.path.dirname(__file__)))
    cfg["head_turn"]["consecutive_frames"] = 8
    cfg["paths"]["screenshots_dir"] = "shots"
    events = []
    p = CheatingPipeline(cfg, FakeDet(nose_x), LocalDB(os.path.join(tmp, "p.db")), tmp, events.append)
    p.start_video()
    return p, events


def run(p, n=30):
    frame = np.zeros((240, 320, 3), np.uint8); last = None
    for i in range(n):
        last, _ = p.process_frame(frame, i, 25, "v.mp4")
    return last


def test_pipeline_looking_sideways_alert_and_red_box():
    tmp = tempfile.mkdtemp(); p, events = make_pipeline(118, tmp)
    out = run(p)
    assert len(events) == 1 and events[0]["behavior"] == "Looking sideways"
    assert os.path.exists(events[0]["screenshot_path"])
    assert tuple(out[10, 100]) == (0, 0, 255)                  # red box top edge (BGR)


def test_pipeline_normal_student_stays_green_and_silent():
    tmp = tempfile.mkdtemp(); p, events = make_pipeline(100, tmp)
    out = run(p)
    assert events == [] and tuple(out[10, 100]) == (0, 255, 0)


class ObjDet:
    """A student holding a 'cell phone' (wrist right on the object)."""
    def reset_tracker(self): pass
    def detect(self, frame):
        s = Student(9, (0, 0, 200, 400), 0.9, [(100, 300)], {})
        return [s], [((95, 295, 110, 310), 0.8, "cell phone")]


def test_generic_behavior_class_becomes_alert_with_label(tmp_path=None):
    tmp = tempfile.mkdtemp()
    from core.config import load_config
    cfg = load_config(os.path.dirname(os.path.dirname(__file__)))
    cfg["head_turn"]["enabled"] = False
    cfg["behavior_labels"] = {"cell phone": "Phone detected"}
    cfg["decision"]["consecutive_frames"] = 5
    cfg["paths"]["screenshots_dir"] = "shots"
    events = []
    p = CheatingPipeline(cfg, ObjDet(), LocalDB(os.path.join(tmp, "g.db")), tmp, events.append)
    p.start_video(); run(p, 12)
    assert len(events) == 1 and events[0]["behavior"] == "Phone detected"
    cfg["behavior_labels"] = {}
    p2 = CheatingPipeline(cfg, ObjDet(), LocalDB(os.path.join(tmp, "g2.db")), tmp, events.append)
    p2.start_video(); run(p2, 12)
    assert events[-1]["behavior"] == "Cell phone detected"          # default label


def test_model_info_overrides_config():
    import json
    from core.config import load_config
    tmp = tempfile.mkdtemp(); os.makedirs(os.path.join(tmp, "models"))
    json.dump({"models": {"pose": "x.pt"}, "thresholds": {"phone_conf": 0.35}}, open(os.path.join(tmp, "config.json"), "w"))
    json.dump({"models": {"behavior": "models/behavior_best.pt"}, "cheating_classes": ["phone"],
               "thresholds": {"behavior_conf": 0.4}}, open(os.path.join(tmp, "models", "model_info.json"), "w"))
    c = load_config(tmp)
    assert c["models"] == {"pose": "x.pt", "behavior": "models/behavior_best.pt"}
    assert c["cheating_classes"] == ["phone"] and c["thresholds"]["behavior_conf"] == 0.4 and c["thresholds"]["phone_conf"] == 0.35


class PersonDet:
    """Model draws a 'cheating' box around the whole student (person-level class)."""
    def reset_tracker(self): pass
    def detect(self, frame):
        a = Student(1, (0, 0, 200, 400), 0.9, [], {})
        b = Student(2, (300, 0, 500, 400), 0.9, [], {})
        return [a, b], [((10, 10, 195, 390), 0.7, "cheating")]


def test_person_level_class_goes_to_best_iou_student_only():
    from core.cheating_logic import match_person_box
    a = Student(1, (0, 0, 200, 400)); b = Student(2, (300, 0, 500, 400))
    assert match_person_box((10, 10, 195, 390), [a, b]) == 1
    assert match_person_box((600, 0, 700, 100), [a, b]) is None
    tmp = tempfile.mkdtemp()
    from core.config import load_config
    cfg = load_config(os.path.dirname(os.path.dirname(__file__)))
    cfg["head_turn"]["enabled"] = False
    cfg["class_kind"] = {"cheating": "person"}
    cfg["decision"]["consecutive_frames"] = 5
    cfg["paths"]["screenshots_dir"] = "shots"
    events = []
    p = CheatingPipeline(cfg, PersonDet(), LocalDB(os.path.join(tmp, "pp.db")), tmp, events.append)
    p.start_video(); run(p, 12)
    assert len(events) == 1 and events[0]["track_id"] == 1 and events[0]["behavior"] == "Cheating detected"


class MixedDet:
    """Dataset-style output: a big 'cheating' box on student 1 AND a small 'suspected cheating' box at student 2's hand."""
    def reset_tracker(self): pass
    def detect(self, frame):
        a = Student(1, (0, 0, 200, 400), 0.9, [(100, 300)], {})
        b = Student(2, (300, 0, 500, 400), 0.9, [(400, 300)], {})
        return [a, b], [((10, 10, 195, 390), 0.7, "cheating"),
                        ((392, 292, 410, 310), 0.6, "suspected cheating")]


def test_auto_mode_mixed_boxes_and_per_behavior_rules():
    from core.config import load_config
    tmp = tempfile.mkdtemp()
    cfg = load_config(os.path.dirname(os.path.dirname(__file__)))
    cfg["head_turn"]["enabled"] = False
    cfg["decision"]["consecutive_frames"] = 5
    cfg["behavior_rules"] = {"Suspected cheating detected": {"consecutive_frames": 12}}
    cfg["paths"]["screenshots_dir"] = "shots"
    events = []
    p = CheatingPipeline(cfg, MixedDet(), LocalDB(os.path.join(tmp, "m.db")), tmp, events.append)
    p.start_video(); run(p, 8)
    assert [(e["track_id"], e["behavior"]) for e in events] == [(1, "Cheating detected")]   # student 2 not yet
    run(p, 20)
    assert (2, "Suspected cheating detected") in [(e["track_id"], e["behavior"]) for e in events]
