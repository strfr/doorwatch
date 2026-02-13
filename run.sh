#!/bin/bash
# DoorWatch - Calistirma betigi
SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
VENV_DIR="$SCRIPT_DIR/.venv"

if [ -d "$VENV_DIR" ]; then
    source "$VENV_DIR/bin/activate"
fi

# cupy/nvidia wheel kutuphaneleri icin LD_LIBRARY_PATH ekle
SITE_PACKAGES_DIR="$(find "$VENV_DIR/lib" -maxdepth 3 -type d -name site-packages 2>/dev/null | head -n 1)"
if [ -n "$SITE_PACKAGES_DIR" ] && [ -d "$SITE_PACKAGES_DIR/nvidia" ]; then
    while IFS= read -r CUDA_LIB_DIR; do
        LD_LIBRARY_PATH="$CUDA_LIB_DIR:${LD_LIBRARY_PATH:-}"
    done < <(find "$SITE_PACKAGES_DIR/nvidia" -maxdepth 3 -type d -name lib)
    export LD_LIBRARY_PATH
fi

cd "$SCRIPT_DIR"
python3 main.py "$@"
