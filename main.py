import json
import sys
import tkinter as tk
from pathlib import Path
from tkinter import messagebox

from core.config import behavior_model_path, load_config
from db.local_db import LocalDB
from db.sync import SyncManager
from ui.app import App

BASE = Path(__file__).resolve().parent


def make_pipeline_factory(cfg, db):
    def factory(on_event):
        # heavy imports are deferred so the window opens instantly
        from core.detector import Detector
        from core.pipeline import CheatingPipeline
        for rel in (behavior_model_path(cfg), cfg["models"]["pose"]):
            p = BASE / rel
            if not p.exists():
                raise FileNotFoundError(f"Model file not found: {p}\nPut it in the models/ folder.")
        return CheatingPipeline(cfg, Detector(cfg, BASE), db, base_dir=str(BASE), on_event=on_event)
    return factory


def make_sync(cfg, db):
    fb = cfg.get("firebase", {})
    if not fb.get("enabled"):
        return SyncManager(db, None)

    def client_factory():
        from db.firebase_client import FirebaseClient
        return FirebaseClient(fb, BASE)
    return SyncManager(db, client_factory, interval=fb.get("sync_interval_seconds", 30))


def main():
    cfg = load_config(BASE)
    db = LocalDB(BASE / cfg["paths"]["sqlite_db"])
    root = tk.Tk()
    App(root, cfg, db, make_pipeline_factory(cfg, db), sync=make_sync(cfg, db))
    root.mainloop()


if __name__ == "__main__":
    main()
