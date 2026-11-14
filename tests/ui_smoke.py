"""Headless smoke test of the Tkinter app with a stub pipeline (no models needed).
Run:  xvfb-run -a python tests/ui_smoke.py   (on Windows/Mac just run it normally)"""
import os, sys, tempfile, time
sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))
import cv2, numpy as np, tkinter as tk
from tkinter import filedialog, messagebox
from datetime import datetime
from db.local_db import LocalDB
from ui import app as appmod
from db.sync import SyncManager

tmp = tempfile.mkdtemp()
vid = os.path.join(tmp, "fake.mp4")
w = cv2.VideoWriter(vid, cv2.VideoWriter_fourcc(*"mp4v"), 20, (320, 240))
for i in range(40):
    w.write(np.full((240, 320, 3), (i * 5) % 255, np.uint8))
w.release()

class Stub:
    def __init__(self, db, on_event): self.db, self.on_event = db, on_event
    def start_video(self): pass
    def process_frame(self, frame, n, fps, name):
        out = frame.copy(); cv2.rectangle(out, (20, 20), (120, 120), (0, 0, 255), 3)
        if n == 10:
            p = os.path.join(tmp, "shot.jpg"); cv2.imwrite(p, out)
            ev = dict(timestamp=datetime.now().isoformat(timespec="seconds"), behavior="Phone detected",
                      confidence=0.87, track_id=3, video_name=name, frame_number=n, screenshot_path=p)
            ev["id"] = self.db.insert_alert(ev); self.on_event(ev)
        return out, []

cfg = {"process_every_n_frames": 1, "paths": {"screenshots_dir": "alerts/screenshots"}}
db = LocalDB(os.path.join(tmp, "t.db"))
root = tk.Tk()
errors = []
messagebox.showerror = lambda *a, **k: errors.append(a)
filedialog.askopenfilename = lambda **k: vid
class FakeCloud:
    def upload_alert(self, a): return "http://example/" + str(a["id"])
sync = SyncManager(db, FakeCloud, interval=1)
a = appmod.App(root, cfg, db, lambda on_event: Stub(db, on_event), sync=sync)
a.upload_video()
end = time.time() + 8
while time.time() < end and str(a.btn_upload["state"]) == "disabled":
    root.update(); time.sleep(0.01)
for _ in range(150): root.update(); time.sleep(0.01)   # let the sync thread run

assert not errors, errors
assert str(a.btn_upload["state"]) == "normal", "worker did not finish"
rows = a.tree.get_children()
assert len(rows) == 1, rows
assert a.tree.item(rows[0], 'values')[6] == '✓', a.tree.item(rows[0], 'values')
assert 'synced' in a.cloud.get().lower() or 'up to date' in a.cloud.get().lower(), a.cloud.get()
a.tree.selection_set(rows[0]); a.view_selected(); root.update()
assert any(isinstance(c, tk.Toplevel) for c in root.winfo_children())
print("UI smoke test passed:", a.tree.item(rows[0], "values")[:4], "| status:", a.status.get())
a.on_close()
