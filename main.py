#!/usr/bin/env python3
"""DoorWatch – Kapı gözetleme sistemi giriş noktası."""

import sys
import os

# Proje kökünü path'e ekle
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from doorwatch.app import DoorWatchApp


def main():
    app = DoorWatchApp()
    app.run()


if __name__ == "__main__":
    main()
