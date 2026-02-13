"""Manage config values in runtime and persistent storage."""

from __future__ import annotations

import json
import os
from pathlib import Path
import re

from doorwatch import config


SETTING_KEYS = [
    "CAMERA_INDEX",
    "CAPTURE_WIDTH",
    "CAPTURE_HEIGHT",
    "CAPTURE_FPS",
    "DETECTION_FRAME_COUNT",
    "DETECTION_INTERVAL",
    "MIN_CONTOUR_AREA",
    "MOTION_USE_GPU",
    "MOTION_PREPROCESS_GRAYSCALE",
    "MOTION_FILTER_MEDIAN",
    "MOTION_FILTER_GAUSSIAN",
    "MOTION_REARM_FRAMES",
    "MOTION_REARM_ACTIVE_AREA",
    "MOTION_MAX_ACTIVE_RATIO",
    "PROCESS_WIDTH_IDLE",
    "PROCESS_HEIGHT_IDLE",
    "POPUP_DURATION_SEC",
    "POPUP_WIDTH",
    "POPUP_HEIGHT",
]


def current_settings() -> dict[str, int | float | bool]:
    return {key: getattr(config, key) for key in SETTING_KEYS}


def apply_runtime(settings: dict[str, int | float | bool]) -> None:
    for key, value in sanitize_settings(settings).items():
        if key in SETTING_KEYS:
            setattr(config, key, value)


def save_to_file(settings: dict[str, int | float | bool]) -> None:
    settings = sanitize_settings(settings)
    path = Path(config.__file__).resolve()
    text = path.read_text(encoding="utf-8")
    lines = text.splitlines(keepends=True)
    pattern = re.compile(r"^(\s*)([A-Z_][A-Z0-9_]*)\s*=")
    replaced: set[str] = set()
    out: list[str] = []

    for line in lines:
        match = pattern.match(line)
        if not match:
            out.append(line)
            continue

        key = match.group(2)
        if key not in settings:
            out.append(line)
            continue

        indent = match.group(1)
        out.append(f"{indent}{key} = {_format_value(settings[key])}\n")
        replaced.add(key)

    missing = [key for key in settings.keys() if key not in replaced]
    if missing:
        if out and not out[-1].endswith("\n"):
            out[-1] += "\n"
        out.append("\n# Runtime settings saved\n")
        for key in missing:
            out.append(f"{key} = {_format_value(settings[key])}\n")

    path.write_text("".join(out), encoding="utf-8")


def _format_value(value: int | float | bool) -> str:
    if isinstance(value, bool):
        return "True" if value else "False"
    if isinstance(value, float):
        return f"{value:.4f}".rstrip("0").rstrip(".")
    return str(int(value))


def user_settings_path() -> Path:
    config_home = os.environ.get("XDG_CONFIG_HOME")
    if config_home:
        return Path(config_home) / "doorwatch" / "settings.json"
    return Path.home() / ".config" / "doorwatch" / "settings.json"


def load_user_settings() -> dict[str, int | float | bool]:
    path = user_settings_path()
    if not path.is_file():
        return {}
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
        if not isinstance(data, dict):
            return {}
        return sanitize_settings(data)
    except Exception:
        return {}


def save_user_settings(settings: dict[str, int | float | bool]) -> None:
    path = user_settings_path()
    path.parent.mkdir(parents=True, exist_ok=True)
    data = sanitize_settings(settings)
    path.write_text(json.dumps(data, ensure_ascii=True, indent=2), encoding="utf-8")


def persist_settings(settings: dict[str, int | float | bool]) -> str:
    """
    Persist settings.
    Returns:
      - "config.py": written into source config file
      - "user": written into per-user settings file
    """
    settings = sanitize_settings(settings)
    try:
        save_to_file(settings)
        return "config.py"
    except OSError:
        save_user_settings(settings)
        return "user"


def sanitize_settings(settings: dict[str, int | float | bool]) -> dict[str, int | float | bool]:
    out: dict[str, int | float | bool] = {}
    defaults = current_settings()
    for key in SETTING_KEYS:
        if key not in settings:
            continue
        value = settings[key]
        default_value = defaults[key]
        if isinstance(default_value, bool):
            out[key] = bool(value)
        elif isinstance(default_value, int):
            out[key] = int(value)
        else:
            out[key] = float(value)
    return out
