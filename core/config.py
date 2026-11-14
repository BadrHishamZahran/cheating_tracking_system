"""config.json + (optional) models/model_info.json written by the Kaggle notebook."""
import json
from pathlib import Path


def load_config(base="."):
    base = Path(base)
    cfg = json.loads((base / "config.json").read_text(encoding="utf-8"))
    info = base / "models" / "model_info.json"
    if info.exists():
        for k, v in json.loads(info.read_text(encoding="utf-8")).items():
            if isinstance(v, dict) and isinstance(cfg.get(k), dict):
                cfg[k].update(v)
            else:
                cfg[k] = v
    return cfg


def behavior_model_path(cfg):
    return cfg["models"].get("behavior") or cfg["models"]["phone"]
