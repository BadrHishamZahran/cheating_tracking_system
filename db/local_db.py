"""SQLite store (works offline). `synced` is reserved for the Firebase step."""
import sqlite3
from pathlib import Path

SCHEMA = """CREATE TABLE IF NOT EXISTS alerts (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    timestamp TEXT NOT NULL,
    behavior TEXT NOT NULL,
    confidence REAL,
    track_id INTEGER,
    video_name TEXT,
    frame_number INTEGER,
    screenshot_path TEXT,
    screenshot_url TEXT,
    synced INTEGER DEFAULT 0
)"""


class LocalDB:
    def __init__(self, path="alerts/alerts.db"):
        self.path = str(path)
        Path(self.path).parent.mkdir(parents=True, exist_ok=True)
        with self._conn() as c:
            c.execute(SCHEMA)

    def _conn(self):
        # new connection per call: safe across the worker thread and the UI thread
        return sqlite3.connect(self.path, timeout=10)

    def insert_alert(self, e):
        with self._conn() as c:
            cur = c.execute(
                "INSERT INTO alerts (timestamp, behavior, confidence, track_id, video_name,"
                " frame_number, screenshot_path, screenshot_url) VALUES (?,?,?,?,?,?,?,?)",
                (e["timestamp"], e["behavior"], e["confidence"], e["track_id"],
                 e["video_name"], e["frame_number"], e["screenshot_path"],
                 e.get("screenshot_url")))
            return cur.lastrowid

    def list_alerts(self, limit=500):
        with self._conn() as c:
            c.row_factory = sqlite3.Row
            return [dict(r) for r in c.execute(
                "SELECT * FROM alerts ORDER BY id DESC LIMIT ?", (limit,))]

    def unsynced(self):
        with self._conn() as c:
            c.row_factory = sqlite3.Row
            return [dict(r) for r in c.execute("SELECT * FROM alerts WHERE synced=0")]

    def mark_synced(self, alert_id, url=None):
        with self._conn() as c:
            c.execute("UPDATE alerts SET synced=1, screenshot_url=COALESCE(?, screenshot_url)"
                      " WHERE id=?", (url, alert_id))
