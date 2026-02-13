"""Linux autostart (.desktop) yardimcilari."""

from __future__ import annotations

import os
from pathlib import Path


def is_enabled() -> bool:
    return _desktop_path().is_file()


def set_enabled(enabled: bool, exec_path: str, icon_path: str, app_name: str) -> None:
    desktop_file = _desktop_path()
    if not enabled:
        if desktop_file.exists():
            desktop_file.unlink()
        return

    desktop_file.parent.mkdir(parents=True, exist_ok=True)
    content = (
        "[Desktop Entry]\n"
        "Type=Application\n"
        f"Name={app_name}\n"
        "Comment=DoorWatch\n"
        f"Exec={exec_path}\n"
        f"Icon={icon_path}\n"
        "Terminal=false\n"
        "Categories=Utility;Security;\n"
        "StartupNotify=false\n"
        "X-GNOME-Autostart-enabled=true\n"
    )
    desktop_file.write_text(content, encoding="utf-8")


def _desktop_path() -> Path:
    config_home = os.environ.get("XDG_CONFIG_HOME")
    if config_home:
        return Path(config_home) / "autostart" / "doorwatch.desktop"
    return Path.home() / ".config" / "autostart" / "doorwatch.desktop"
