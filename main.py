#!/usr/bin/env python3
"""DoorWatch entry point."""

import sys
import os

# Add project root to import path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from doorwatch.app import DoorWatchApp


def main():
    app = DoorWatchApp()
    app.run()


if __name__ == "__main__":
    main()
