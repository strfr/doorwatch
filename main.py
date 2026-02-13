#!/usr/bin/env python3
"""DoorWatch entry point."""

import sys
import os
import ctypes

# Add project root to import path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))


def _set_process_identity() -> None:
    """
    Set a stable app/process name so tray/system monitor does not show `main.py`.
    """
    sys.argv[0] = "doorwatch"

    try:
        import gi
        gi.require_version("GLib", "2.0")
        from gi.repository import GLib
        GLib.set_prgname("doorwatch")
        GLib.set_application_name("DoorWatch")
    except Exception:
        pass

    try:
        libc = ctypes.CDLL("libc.so.6")
        pr_set_name = 15  # Linux PR_SET_NAME
        libc.prctl(pr_set_name, ctypes.c_char_p(b"doorwatch"), 0, 0, 0)
    except Exception:
        pass


def main():
    _set_process_identity()
    from doorwatch.app import DoorWatchApp
    app = DoorWatchApp()
    app.run()


if __name__ == "__main__":
    main()
