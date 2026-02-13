# DoorWatch

Lightweight tray-based motion popup monitor for Linux.

## One-Line Install From GitHub

After pushing this repo to GitHub, install with:

```bash
curl -fsSL https://raw.githubusercontent.com/<owner>/<repo>/main/install.sh | bash -s -- https://github.com/<owner>/<repo>.git
```

This command:
- clones the repository to a temporary folder
- builds a `.deb` package
- installs it with `apt`

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
