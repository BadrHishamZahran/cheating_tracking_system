"""Background sync: SQLite (source of truth, works offline) -> Firebase.

Every alert is saved locally first. This worker uploads rows with synced=0 and retries
every `interval` seconds, so a network drop never loses an alert or blocks detection.
"""
import threading


class SyncManager:
    def __init__(self, db, client_factory=None, interval=30, on_synced=None, on_status=None):
        """client_factory: () -> object with upload_alert(alert) -> url. None = cloud disabled."""
        self.db, self.client_factory = db, client_factory
        self.interval = interval
        self.on_synced, self.on_status = on_synced, on_status
        self._client = None
        self._wake = threading.Event()
        self._stop = threading.Event()
        self._thread = None
        self.disabled_reason = None if client_factory else "Cloud sync disabled (local only)"

    def _status(self, text):
        if self.on_status:
            self.on_status(text)

    def start(self):
        if not self.client_factory:
            self._status(self.disabled_reason)
            return
        self._thread = threading.Thread(target=self._loop, daemon=True)
        self._thread.start()

    def trigger(self):
        self._wake.set()

    def stop(self):
        self._stop.set()
        self._wake.set()

    def _loop(self):
        while not self._stop.is_set():
            self.sync_once()
            self._wake.wait(self.interval)
            self._wake.clear()

    def sync_once(self):
        """One pass over unsynced alerts. Returns (synced_count, pending_count)."""
        if not self.client_factory:
            return 0, 0
        if self._client is None:
            try:
                self._client = self.client_factory()
            except Exception as e:                      # bad credentials / package missing
                self.client_factory = None
                self.disabled_reason = f"Cloud sync disabled: {e}"
                self._status(self.disabled_reason)
                return 0, 0
        done = 0
        pending = self.db.unsynced()
        for a in sorted(pending, key=lambda r: r["id"]):
            try:
                url = self._client.upload_alert(a)
            except Exception as e:                      # offline etc. -> retry next cycle
                self._status(f"Cloud: offline, {len(pending) - done} pending ({type(e).__name__})")
                return done, len(pending) - done
            self.db.mark_synced(a["id"], url)
            done += 1
            if self.on_synced:
                self.on_synced(a["id"])
        self._status("Cloud: all alerts synced" if pending else "Cloud: up to date")
        return done, 0
