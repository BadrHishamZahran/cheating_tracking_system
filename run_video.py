"""Quick local test (no Tkinter):  python run_video.py path/to/video.mp4"""
import argparse, json
import cv2
from core.config import load_config
from core.detector import Detector
from core.pipeline import CheatingPipeline
from db.local_db import LocalDB

ap = argparse.ArgumentParser()
ap.add_argument("video")
ap.add_argument("--save", help="optional output video path")
args = ap.parse_args()

cfg = load_config(".")
db = LocalDB(cfg["paths"]["sqlite_db"])
pipe = CheatingPipeline(cfg, Detector(cfg), db,
                        on_event=lambda e: print(f"ALERT id={e['track_id']} {e['behavior']} "
                                                 f"conf={e['confidence']} -> {e['screenshot_path']}"))
cap = cv2.VideoCapture(args.video)
fps = cap.get(cv2.CAP_PROP_FPS) or 25
step = cfg["process_every_n_frames"]
writer = None
pipe.start_video()
i, last = 0, None
while True:
    ok, frame = cap.read()
    if not ok:
        break
    if i % step == 0:
        last, _ = pipe.process_frame(frame, i, fps, args.video)
    shown = last if last is not None else frame
    if args.save:
        if writer is None:
            h, w = shown.shape[:2]
            writer = cv2.VideoWriter(args.save, cv2.VideoWriter_fourcc(*"mp4v"), fps, (w, h))
        writer.write(shown)
    cv2.imshow("Cheating detection (q to quit)", shown)
    if cv2.waitKey(1) & 0xFF == ord("q"):
        break
    i += 1
cap.release()
if writer: writer.release()
cv2.destroyAllWindows()
