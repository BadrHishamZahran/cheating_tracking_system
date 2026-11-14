"""Thin wrapper around firebase-admin (Firestore for records, Storage for screenshots).

Cloud Storage for Firebase needs the Blaze plan (billing account linked) since Feb 2026.
If you stay on the free Spark plan, set "upload_screenshots": false in config.json and
only the alert records go to Firestore (the screenshot stays on the local disk).
"""
import hashlib
from pathlib import Path


def alert_doc_id(a):
    """Deterministic id => re-syncing the same alert overwrites instead of duplicating."""
    key = f"{a['timestamp']}|{a.get('video_name')}|{a.get('frame_number')}|{a.get('track_id')}"
    return hashlib.sha1(key.encode()).hexdigest()[:20]


class FirebaseClient:
    def __init__(self, fb_cfg, base_dir="."):
        import firebase_admin                       # imported lazily: app works without it
        from firebase_admin import credentials, firestore, storage
        cred_path = Path(base_dir) / fb_cfg["service_account"]
        if not cred_path.exists():
            raise FileNotFoundError(f"Service account file not found: {cred_path}")
        self.cfg = fb_cfg
        self.upload_screenshots = bool(fb_cfg.get("upload_screenshots", True))
        opts = {"storageBucket": fb_cfg["storage_bucket"]} if self.upload_screenshots else {}
        if not firebase_admin._apps:
            firebase_admin.initialize_app(credentials.Certificate(str(cred_path)), opts)
        self.db = firestore.client()
        self._firestore = firestore
        self.bucket = storage.bucket() if self.upload_screenshots else None
        self.collection = fb_cfg.get("collection", "alerts")

    def _upload_file(self, local_path, doc_id):
        blob = self.bucket.blob(f"screenshots/{doc_id}{Path(local_path).suffix or '.jpg'}")
        blob.upload_from_filename(local_path, content_type="image/jpeg")
        if self.cfg.get("public_urls", False):
            blob.make_public()
            return blob.public_url
        return f"gs://{self.bucket.name}/{blob.name}"   # private: path only

    def upload_alert(self, a):
        """Uploads one alert. Returns the screenshot URL (or None). Raises on network errors."""
        doc_id = alert_doc_id(a)
        url = a.get("screenshot_url")
        path = a.get("screenshot_path")
        if self.upload_screenshots and path and Path(path).exists():
            url = self._upload_file(path, doc_id)
        doc = {
            "timestamp": a["timestamp"], "behavior": a["behavior"],
            "confidence": a.get("confidence"), "track_id": a.get("track_id"),
            "video_name": a.get("video_name"), "frame_number": a.get("frame_number"),
            "screenshot_path": path, "screenshot_url": url,
            "created_at": self._firestore.SERVER_TIMESTAMP,
        }
        self.db.collection(self.collection).document(doc_id).set(doc, merge=True)
        return url
