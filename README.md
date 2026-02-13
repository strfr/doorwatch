# DoorWatch

Lightweight tray-based motion popup monitor for Linux.

## One-Line Install (Latest Release)

```bash
curl -fsSL https://github.com/strfr/doorwatch/releases/latest/download/install.sh | bash
```

This command:
- downloads installer script from the latest GitHub Release
- clones repository to a temporary folder
- builds a `.deb` package
- installs it with `apt`

Release page:
- https://github.com/strfr/doorwatch/releases/latest

## Local Install (from source checkout)

```bash
./install.sh
```

## Manual Package Build

```bash
./packaging/build_deb.sh
sudo apt install ./dist/doorwatch_*.deb
```

## Run

```bash
doorwatch
```

## Uninstall

```bash
sudo apt remove doorwatch
```

## Features

- Motion-only detection (no person/object detection)
- Tray menu: `Camera Window`, `Settings`, `Silent Mode`, `Exit`
- Popup stays visible while motion continues, then closes after `POPUP_DURATION_SEC` seconds of no motion
- `Settings` lets you change camera, performance parameters, and autostart

## Settings Persistence

- If writable, settings are saved to `doorwatch/config.py`
- In packaged installs (read-only app files), settings are saved to `~/.config/doorwatch/settings.json`
