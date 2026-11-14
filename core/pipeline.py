"""frame -> detect -> (behavior objects + head-turn rules) -> decide -> draw red/green -> screenshot + DB."""
from datetime import datetime
import cv2
from core.cheating_logic import CheatingLogic, associate_phone, match_person_box
from core.head_pose import HeadTurnDetector
from core.screenshot import save_screenshot

HEAD = "Looking sideways"


class CheatingPipeline:
    def __init__(self, cfg, detector, db, base_dir=".", on_event=None):
        self.cfg, self.det, self.db, self.on_event = cfg, detector, db, on_event
        self.max_ratio = cfg["decision"]["phone_wrist_max_dist_ratio"]
        self.head_cfg = cfg.get("head_turn", {"enabled": False})
        self.debug = cfg.get("debug_overlay", False)
        self.kind = cfg.get("class_kind", {})          # optional override: class -> "object" | "person"
        self.min_iou = cfg["decision"].get("person_box_min_iou", 0.3)
        self.shots_dir = f"{base_dir}/{cfg['paths']['screenshots_dir']}"
        self.green = tuple(cfg["colors_bgr"]["normal"])
        self.red = tuple(cfg["colors_bgr"]["cheating"])
        self.obj_color = tuple(cfg["colors_bgr"].get("object", cfg["colors_bgr"].get("phone", (255, 160, 0))))
        self._reset_state()

    def _reset_state(self):
        d, h = self.cfg["decision"], self.head_cfg
        self.logics = {}
        self.head = None
        if h.get("enabled"):
            self.logics[HEAD] = CheatingLogic(h["consecutive_frames"], h["cooldown_seconds"],
                                              d["hold_red_seconds"], h["max_gap_frames"])
            self.head = HeadTurnDetector(self.cfg["thresholds"]["pose_kpt_conf"],
                                         h["ear_ratio_thr"], h["shoulder_ratio_thr"],
                                         h["smooth_frames"])

    def _logic(self, name):
        if name not in self.logics:      # one logic (streak + cooldown) per behavior label
            d = self.cfg["decision"]
            r = self.cfg.get("behavior_rules", {}).get(name, {})
            self.logics[name] = CheatingLogic(r.get("consecutive_frames", d["consecutive_frames"]),
                                              r.get("cooldown_seconds", d["cooldown_seconds"]),
                                              d["hold_red_seconds"], d["max_gap_frames"])
        return self.logics[name]

    def _label(self, cls_name):
        m = self.cfg.get("behavior_labels", {})
        if cls_name in m:
            return m[cls_name]
        return cls_name.replace("_", " ").replace("-", " ").strip().capitalize() + " detected"

    def start_video(self):
        self._reset_state()
        self.det.reset_tracker()

    def process_frame(self, frame, frame_no, fps, video_name):
        """Returns (annotated_frame, [events])."""
        t = frame_no / (fps or 25)
        students, objects = self.det.detect(frame)

        flagged = {}   # track_id -> {behavior label: confidence}
        person_level = set()   # indexes of detections that cover a whole student (not drawn twice)
        for idx, (box, conf, cls_name) in enumerate(objects):
            kind = self.kind.get(cls_name, "auto")      # auto: whole-student box first, else proximity
            sid = None
            if kind in ("person", "auto"):
                sid = match_person_box(box, students, self.min_iou)
                if sid is not None:
                    person_level.add(idx)
            if sid is None and kind in ("object", "auto"):
                sid = associate_phone(box, students, self.max_ratio)
            if sid is not None:
                lab = self._label(cls_name)
                cur = flagged.setdefault(sid, {})
                cur[lab] = max(conf, cur.get(lab, 0.0))

        events = []
        for s in students:
            active = dict(flagged.get(s.track_id, {}))
            if self.head is not None:
                active[HEAD] = self.head.update(s.track_id, s.kpts)
            for name in set(self.logics) | set(active):
                conf = active.get(name)
                ev = self._logic(name).update(s.track_id, name if conf else None, conf or 0.0, t)
                if ev:
                    ev["behavior"] = name
                    events.append(ev)

        out = frame.copy()
        for idx, (box, conf, cls_name) in enumerate(objects):
            if idx in person_level:
                continue                                  # shown through the student's red box
            x1, y1, x2, y2 = map(int, box)
            cv2.rectangle(out, (x1, y1), (x2, y2), self.obj_color, 2)
            cv2.putText(out, f"{cls_name} {conf:.2f}", (x1, max(15, y1 - 5)),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, self.obj_color, 1)
        for s in students:
            why = [n for n, lg in self.logics.items() if lg.is_cheating(s.track_id, t)]
            bad = bool(why)
            color = self.red if bad else self.green
            x1, y1, x2, y2 = map(int, s.box)
            cv2.rectangle(out, (x1, y1), (x2, y2), color, 3 if bad else 2)
            label = f"ID {s.track_id}" + (f" | {' + '.join(w.upper() for w in why)}" if bad else "")
            cv2.putText(out, label, (x1, max(15, y1 - 8)), cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)
            if self.debug and self.head is not None and s.track_id in self.head.last:
                cv2.putText(out, f"yaw {self.head.last[s.track_id]:+.2f} (1.0 = threshold)",
                            (x1, min(out.shape[0] - 5, y2 + 18)),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)

        for ev in events:   # screenshot AFTER drawing so the red box is visible
            ev.update(timestamp=datetime.now().isoformat(timespec="seconds"),
                      video_name=video_name, frame_number=frame_no,
                      screenshot_path=save_screenshot(out, self.shots_dir, ev["track_id"]),
                      screenshot_url=None)
            ev["id"] = self.db.insert_alert(ev)
            if self.on_event:
                self.on_event(ev)
        return out, events
