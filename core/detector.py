"""Models: one pose model does person detection + tracking + keypoints; one behavior model
(trained on Kaggle) detects cheating-related objects/behaviors (phone, paper, ...)."""
from pathlib import Path
import numpy as np
from ultralytics import YOLO
from core.cheating_logic import Student
from core.config import behavior_model_path

L_WRIST, R_WRIST = 9, 10
HEAD_KPTS = {"nose": 0, "l_eye": 1, "r_eye": 2, "l_ear": 3, "r_ear": 4,
             "l_shoulder": 5, "r_shoulder": 6}


class Detector:
    def __init__(self, cfg, base_dir="."):
        base = Path(base_dir)
        self.cfg = cfg
        self.pose = YOLO(str(base / cfg["models"]["pose"]))
        self.behavior = YOLO(str(base / behavior_model_path(cfg)))
        allowed = cfg.get("cheating_classes") or []
        self.allowed = {c.lower() for c in allowed} or None      # None = every class counts
        self.imgsz = cfg["imgsz"]
        self.kpt_thr = cfg["thresholds"]["pose_kpt_conf"]

    def reset_tracker(self):
        """Call when a new video starts so track IDs restart cleanly."""
        pred = getattr(self.pose, "predictor", None)
        if pred is not None and hasattr(pred, "trackers"):
            pred.trackers = []
        self.pose.predictor = None

    def detect(self, frame):
        students, phones = [], []
        r = self.pose.track(frame, persist=True, tracker=self.cfg["tracker"], classes=[0],
                            conf=self.cfg["thresholds"]["person_conf"], verbose=False)[0]
        if r.boxes is not None and r.boxes.id is not None and r.keypoints is not None:
            ids = r.boxes.id.int().cpu().tolist()
            boxes = r.boxes.xyxy.cpu().numpy()
            confs = r.boxes.conf.cpu().numpy()
            kxy = r.keypoints.xy.cpu().numpy()
            kc = (r.keypoints.conf.cpu().numpy() if r.keypoints.conf is not None
                  else np.ones(kxy.shape[:2]))
            for i, tid in enumerate(ids):
                wrists = [tuple(kxy[i][k]) for k in (L_WRIST, R_WRIST)
                          if kc[i][k] >= self.kpt_thr and kxy[i][k].any()]
                kp = {n: (float(kxy[i][j][0]), float(kxy[i][j][1]), float(kc[i][j]))
                      for n, j in HEAD_KPTS.items()}
                students.append(Student(tid, tuple(boxes[i]), float(confs[i]), wrists, kp))

        thr = self.cfg["thresholds"]
        conf = thr.get("behavior_conf", thr.get("phone_conf", 0.35))
        p = self.behavior.predict(frame, conf=conf, imgsz=self.imgsz, verbose=False,
                                  agnostic_nms=self.cfg.get("agnostic_nms", True))[0]   # one class per region
        names = p.names
        for b, c, k in zip(p.boxes.xyxy.cpu().numpy(), p.boxes.conf.cpu().numpy(),
                           p.boxes.cls.int().cpu().tolist()):
            cls_name = names[k]
            if self.allowed and cls_name.lower() not in self.allowed:
                continue
            phones.append((tuple(b), float(c), cls_name))
        return students, phones
