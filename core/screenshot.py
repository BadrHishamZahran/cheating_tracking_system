from datetime import datetime
from pathlib import Path
import cv2


def save_screenshot(frame, out_dir, track_id):
    Path(out_dir).mkdir(parents=True, exist_ok=True)
    name = f"alert_{datetime.now():%Y%m%d_%H%M%S}_id{track_id}.jpg"
    path = Path(out_dir) / name
    cv2.imwrite(str(path), frame)
    return str(path)
