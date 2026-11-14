import os, sys, tempfile
sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))
from db.local_db import LocalDB
from db.sync import SyncManager
from db.firebase_client import alert_doc_id


def make_db(n=3):
    db = LocalDB(os.path.join(tempfile.mkdtemp(), "s.db"))
    for i in range(n):
        db.insert_alert(dict(timestamp=f"2026-11-14T10:3{i}:00", behavior="Phone detected",
                             confidence=.8, track_id=i, video_name="v.mp4",
                             frame_number=i, screenshot_path=f"/none/{i}.jpg"))
    return db


class Fake:
    def __init__(self, fail_after=None):
        self.sent, self.fail_after = [], fail_after
    def upload_alert(self, a):
        if self.fail_after is not None and len(self.sent) >= self.fail_after:
            raise ConnectionError("offline")
        self.sent.append(a["id"]); return f"http://x/{a['id']}"


def test_sync_all():
    db, fake, got = make_db(), Fake(), []
    m = SyncManager(db, lambda: fake, on_synced=got.append)
    assert m.sync_once() == (3, 0)
    assert db.unsynced() == [] and got == [1, 2, 3]


def test_offline_keeps_pending_then_retries():
    db = make_db(); fake = Fake(fail_after=1)
    m = SyncManager(db, lambda: fake)
    assert m.sync_once() == (1, 2) and len(db.unsynced()) == 2
    fake.fail_after = None
    assert m.sync_once() == (2, 0) and db.unsynced() == []
    assert fake.sent == [1, 2, 3]                  # nothing sent twice


def test_bad_credentials_disable_cloud_without_crashing():
    db, msgs = make_db(), []
    def bad(): raise FileNotFoundError("no key")
    m = SyncManager(db, bad, on_status=msgs.append)
    assert m.sync_once() == (0, 0) and "disabled" in msgs[-1]
    assert len(db.unsynced()) == 3                  # alerts stay safe locally


def test_doc_id_is_deterministic():
    a = dict(timestamp="t", video_name="v", frame_number=1, track_id=2)
    assert alert_doc_id(a) == alert_doc_id(dict(a)) and alert_doc_id(a) != alert_doc_id({**a, "track_id": 3})
