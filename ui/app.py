"""Tkinter desktop app.

Threading model (important): the video is processed in a worker thread. The worker never
touches Tkinter. It hands data to the UI through
  - self._latest  : the newest annotated frame (older frames are dropped on purpose)
  - self._ui_q    : status / progress / alert / done / error messages
and the UI thread polls both with root.after().
"""
import os
import queue
import subprocess
import sys
import threading
import time
import tkinter as tk
import webbrowser
from datetime import datetime
from pathlib import Path
from tkinter import filedialog, messagebox, ttk

import cv2
from PIL import Image, ImageTk

BASE_DIR = Path(__file__).resolve().parent.parent
POLL_MS = 30
TOAST_MS = 6000


def fmt_time(iso):
    try:
        return datetime.fromisoformat(iso).strftime("%I:%M %p")
    except Exception:
        return str(iso)


class App:
    def __init__(self, root, cfg, db, pipeline_factory, sync=None):
        """pipeline_factory(on_event) -> CheatingPipeline. Called lazily in the worker thread
        so the window opens instantly and model loading doesn't freeze the UI."""
        self.root, self.cfg, self.db = root, cfg, db
        self.pipeline_factory = pipeline_factory
        self.sync = sync
        self.pipeline = None
        self.worker = None
        self.stop_evt = threading.Event()
        self._ui_q = queue.Queue()
        self._latest = None
        self._lock = threading.Lock()
        self._photo = None

        root.title("Cheating Tracking and Alert System")
        root.geometry("1280x760")
        root.minsize(1000, 620)
        self._set_icon()
        self._build_ui()
        self.load_history()
        if self.sync:
            self.sync.on_synced = lambda i: self._ui_q.put(("synced", i))
            self.sync.on_status = lambda t: self._ui_q.put(("cloud", t))
            self.sync.start()
        root.protocol("WM_DELETE_WINDOW", self.on_close)
        root.after(POLL_MS, self._poll)

    # ---------------------------------------------------------------- UI
    def _set_icon(self):
        ico, png = BASE_DIR / "assets" / "icon.ico", BASE_DIR / "assets" / "icon.png"
        try:
            if sys.platform.startswith("win") and ico.exists():
                self.root.iconbitmap(str(ico))
            elif png.exists():
                self._icon_img = tk.PhotoImage(file=str(png))
                self.root.iconphoto(True, self._icon_img)
        except Exception:
            pass   # icon is optional; never block the app on it

    def _build_ui(self):
        top = ttk.Frame(self.root, padding=8)
        top.pack(fill="x")
        self.btn_upload = ttk.Button(top, text="Upload Video", command=self.upload_video)
        self.btn_upload.pack(side="left")
        self.btn_stop = ttk.Button(top, text="Stop", command=self.stop, state="disabled")
        self.btn_stop.pack(side="left", padx=6)
        self.status = tk.StringVar(value="Ready. Upload an exam video to start.")
        ttk.Label(top, textvariable=self.status).pack(side="left", padx=12)
        self.progress = ttk.Progressbar(top, length=220, mode="determinate")
        self.progress.pack(side="right")

        body = ttk.PanedWindow(self.root, orient="horizontal")
        body.pack(fill="both", expand=True, padx=8, pady=(0, 8))

        left = ttk.Frame(body)
        self.canvas = tk.Label(left, bg="#111", fg="#888", text="No video loaded")
        self.canvas.pack(fill="both", expand=True)
        body.add(left, weight=3)

        right = ttk.Frame(body)
        ttk.Label(right, text="Alert History", font=("Segoe UI", 12, "bold")).pack(anchor="w")
        cols = ("time", "behavior", "conf", "student", "video", "path", "cloud")
        heads = ("Date / Time", "Behavior", "Confidence", "Student ID", "Video", "Screenshot", "Cloud")
        widths = (130, 110, 75, 65, 100, 160, 50)
        self.tree = ttk.Treeview(right, columns=cols, show="headings", selectmode="browse")
        for c, h, w in zip(cols, heads, widths):
            self.tree.heading(c, text=h)
            self.tree.column(c, width=w, anchor="w")
        sb = ttk.Scrollbar(right, orient="vertical", command=self.tree.yview)
        self.tree.configure(yscrollcommand=sb.set)
        self.tree.pack(side="left", fill="both", expand=True, pady=4)
        sb.pack(side="left", fill="y", pady=4)
        self.tree.bind("<Double-1>", lambda e: self.view_selected())
        body.add(right, weight=2)

        bar = ttk.Frame(self.root, padding=(8, 0, 8, 8))
        bar.pack(fill="x")
        ttk.Button(bar, text="View Screenshot", command=self.view_selected).pack(side="left")
        ttk.Button(bar, text="Open Screenshots Folder", command=self.open_folder).pack(side="left", padx=6)
        ttk.Button(bar, text="Refresh", command=self.load_history).pack(side="left")
        ttk.Button(bar, text="Sync Now", command=self.sync_now).pack(side="left", padx=6)
        self.cloud = tk.StringVar(value="Cloud: not configured" if not self.sync else "Cloud: starting...")
        ttk.Label(bar, textvariable=self.cloud).pack(side="right")

    # ---------------------------------------------------------- history
    def _row_values(self, a):
        conf = a.get("confidence")
        return (a.get("timestamp", "").replace("T", " "), a.get("behavior", ""),
                f"{conf:.2f}" if isinstance(conf, (int, float)) else "",
                a.get("track_id", ""), a.get("video_name", "") and os.path.basename(a["video_name"]),
                a.get("screenshot_path", ""), "✓" if a.get("synced") else "…")

    def load_history(self):
        self.tree.delete(*self.tree.get_children())
        for a in self.db.list_alerts():
            self.tree.insert("", "end", iid=str(a["id"]), values=self._row_values(a))

    def add_alert_row(self, a):
        self.tree.insert("", 0, iid=str(a["id"]), values=self._row_values(a))

    # ------------------------------------------------------ screenshots
    def view_selected(self):
        sel = self.tree.selection()
        if not sel:
            messagebox.showinfo("View Screenshot", "Select an alert from the list first.")
            return
        values = self.tree.item(sel[0], "values")
        path = values[5]
        if (not path or not os.path.exists(path)):
            url = next((a.get("screenshot_url") for a in self.db.list_alerts()
                        if str(a["id"]) == sel[0]), None)
            if url and url.startswith("http"):
                webbrowser.open(url)      # local file missing: open the cloud copy
                return
        self.show_screenshot(path, f"{values[1]} | ID {values[3]} | {values[0]}")

    def show_screenshot(self, path, title="Screenshot"):
        if not path or not os.path.exists(path):
            messagebox.showerror("Screenshot not found", f"File does not exist:\n{path}")
            return
        win = tk.Toplevel(self.root)
        win.title(title)
        img = Image.open(path)
        img.thumbnail((1000, 700))
        win._photo = ImageTk.PhotoImage(img)   # keep a reference
        tk.Label(win, image=win._photo).pack()

    def open_folder(self):
        folder = BASE_DIR / self.cfg["paths"]["screenshots_dir"]
        folder.mkdir(parents=True, exist_ok=True)
        try:
            if sys.platform.startswith("win"):
                os.startfile(str(folder))
            elif sys.platform == "darwin":
                subprocess.Popen(["open", str(folder)])
            else:
                subprocess.Popen(["xdg-open", str(folder)])
        except Exception as e:
            messagebox.showerror("Open folder", str(e))

    def sync_now(self):
        if self.sync:
            self.sync.trigger()
        else:
            messagebox.showinfo("Cloud sync", "Firebase is disabled. Enable it in config.json.")

    # ----------------------------------------------------- notification
    def notify(self, ev):
        lines = ["Cheating Alert Detected",
                 f"Behavior: {ev['behavior']}",
                 f"Time: {fmt_time(ev['timestamp'])}",
                 "Screenshot saved successfully."]
        # 1) in-app toast (always works)
        t = tk.Toplevel(self.root)
        t.overrideredirect(True)
        t.attributes("-topmost", True)
        frame = tk.Frame(t, bg="#b71c1c", padx=14, pady=10, bd=2, relief="raised")
        frame.pack()
        tk.Label(frame, text=lines[0], bg="#b71c1c", fg="white",
                 font=("Segoe UI", 11, "bold")).pack(anchor="w")
        for ln in lines[1:]:
            tk.Label(frame, text=ln, bg="#b71c1c", fg="white", font=("Segoe UI", 10)).pack(anchor="w")
        t.update_idletasks()
        x = self.root.winfo_rootx() + self.root.winfo_width() - t.winfo_width() - 20
        y = self.root.winfo_rooty() + self.root.winfo_height() - t.winfo_height() - 50
        t.geometry(f"+{x}+{y}")
        t.after(TOAST_MS, t.destroy)
        frame.bind("<Button-1>", lambda e: t.destroy())
        # 2) system notification (best effort)
        try:
            from plyer import notification
            notification.notify(title=lines[0], message="\n".join(lines[1:]),
                                app_name="Cheating Alert System", timeout=6)
        except Exception:
            pass

    # ------------------------------------------------------------ video
    def upload_video(self):
        if self.worker and self.worker.is_alive():
            messagebox.showinfo("Busy", "A video is already being processed. Stop it first.")
            return
        path = filedialog.askopenfilename(
            title="Select exam video",
            filetypes=[("Video files", "*.mp4 *.avi *.mov *.mkv *.webm"), ("All files", "*.*")])
        if not path:
            return
        self.stop_evt.clear()
        self.btn_upload.config(state="disabled")
        self.btn_stop.config(state="normal")
        self.progress["value"] = 0
        self.worker = threading.Thread(target=self._worker, args=(path,), daemon=True)
        self.worker.start()

    def stop(self):
        self.stop_evt.set()
        self.status.set("Stopping...")

    def _worker(self, path):
        q = self._ui_q
        try:
            if self.pipeline is None:
                q.put(("status", "Loading models (first time only)..."))
                self.pipeline = self.pipeline_factory(lambda ev: q.put(("alert", ev)))
            cap = cv2.VideoCapture(path)
            if not cap.isOpened():
                raise RuntimeError(f"Cannot open video:\n{path}")
            fps = cap.get(cv2.CAP_PROP_FPS) or 25
            total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT)) or 1
            step = max(1, int(self.cfg.get("process_every_n_frames", 1)))
            name = os.path.basename(path)
            self.pipeline.start_video()
            q.put(("status", f"Processing: {name}"))
            i, last, t0 = 0, None, time.time()
            while not self.stop_evt.is_set():
                ok, frame = cap.read()
                if not ok:
                    break
                if i % step == 0:
                    last, _ = self.pipeline.process_frame(frame, i, fps, name)
                show = last if last is not None else frame
                with self._lock:
                    self._latest = cv2.cvtColor(show, cv2.COLOR_BGR2RGB)
                if i % 5 == 0:
                    q.put(("progress", 100.0 * i / total))
                i += 1
                delay = t0 + i / fps - time.time()    # don't play faster than real time
                if delay > 0:
                    time.sleep(delay)
            cap.release()
            q.put(("done", "Stopped." if self.stop_evt.is_set() else f"Finished: {name}"))
        except Exception as e:   # surface any failure in the UI instead of dying silently
            q.put(("error", str(e)))

    # ------------------------------------------------------------- loop
    def _poll(self):
        try:
            while True:
                kind, data = self._ui_q.get_nowait()
                if kind == "status":
                    self.status.set(data)
                elif kind == "progress":
                    self.progress["value"] = data
                elif kind == "alert":
                    self.add_alert_row(data)
                    self.notify(data)
                    if self.sync:
                        self.sync.trigger()
                elif kind == "synced":
                    if self.tree.exists(str(data)):
                        self.tree.set(str(data), "cloud", "✓")
                elif kind == "cloud":
                    self.cloud.set(data)
                elif kind in ("done", "error"):
                    self.status.set(data if kind == "done" else "Error")
                    self.btn_upload.config(state="normal")
                    self.btn_stop.config(state="disabled")
                    if kind == "done":
                        self.progress["value"] = 100 if "Finished" in data else self.progress["value"]
                    else:
                        messagebox.showerror("Error", data)
        except queue.Empty:
            pass
        with self._lock:
            frame, self._latest = self._latest, None
        if frame is not None:
            self._render(frame)
        self.root.after(POLL_MS, self._poll)

    def _render(self, rgb):
        w, h = max(self.canvas.winfo_width(), 100), max(self.canvas.winfo_height(), 100)
        ih, iw = rgb.shape[:2]
        s = min(w / iw, h / ih)
        img = Image.fromarray(rgb).resize((max(1, int(iw * s)), max(1, int(ih * s))))
        self._photo = ImageTk.PhotoImage(img)
        self.canvas.configure(image=self._photo, text="")

    def on_close(self):
        self.stop_evt.set()
        if self.sync:
            self.sync.stop()
        self.root.destroy()
