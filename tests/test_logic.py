import os, sys, tempfile
sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))
from core.cheating_logic import CheatingLogic, Student, associate_phone
from db.local_db import LocalDB


def test_association_prefers_wrist():
    a = Student(1, (0, 0, 200, 400), wrists=[(100, 300)])
    b = Student(2, (150, 0, 350, 400), wrists=[(300, 300)])
    assert associate_phone((95, 295, 110, 310), [a, b]) == 1
    assert associate_phone((295, 295, 310, 310), [a, b]) == 2


def test_association_far_phone_is_ignored():
    a = Student(1, (0, 0, 200, 400), wrists=[(100, 300)])
    assert associate_phone((900, 900, 910, 910), [a]) is None


def test_association_fallback_inside_box():
    a = Student(1, (0, 0, 200, 400))
    assert associate_phone((90, 90, 110, 110), [a]) == 1


def test_needs_consecutive_frames_and_cooldown():
    L = CheatingLogic(consecutive_frames=5, cooldown_seconds=30, hold_red_seconds=2, max_gap_frames=1)
    evs = [L.update(7, "Phone detected", 0.8, i * 0.1) for i in range(4)]
    assert evs == [None] * 4 and not L.is_cheating(7, 0.4)
    ev = L.update(7, "Phone detected", 0.8, 0.5)
    assert ev and ev["track_id"] == 7 and L.is_cheating(7, 0.6)
    assert L.update(7, "Phone detected", 0.8, 0.6) is None          # cooldown
    assert L.update(7, "Phone detected", 0.8, 31.0) is not None     # after cooldown


def test_gap_resets_streak():
    L = CheatingLogic(consecutive_frames=3, max_gap_frames=1)
    L.update(1, "x", .9, 0); L.update(1, "x", .9, .1)
    L.update(1, None, 0, .2); L.update(1, None, 0, .3)               # 2 misses > max_gap
    assert L.update(1, "x", .9, .4) is None


def test_db_roundtrip():
    db = LocalDB(os.path.join(tempfile.mkdtemp(), "a.db"))
    i = db.insert_alert(dict(timestamp="2026-11-14T10:35:00", behavior="Phone detected",
                             confidence=.8, track_id=1, video_name="v.mp4",
                             frame_number=10, screenshot_path="a.jpg"))
    assert db.list_alerts()[0]["id"] == i and len(db.unsynced()) == 1
    db.mark_synced(i, "http://x"); assert db.unsynced() == []
