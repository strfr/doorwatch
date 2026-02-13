#!/bin/bash
set -euo pipefail

run_local_install() {
    local script_dir="$1"
    local build_script="${script_dir}/packaging/build_deb.sh"

    if ! command -v dpkg-deb >/dev/null 2>&1; then
        echo "dpkg-deb not found. Install it with: sudo apt-get install -y dpkg-dev"
        exit 1
    fi

    chmod +x "${build_script}"
    echo "Building DoorWatch .deb package..."
    local deb_path
    deb_path="$("${build_script}")"
    echo "Package ready: ${deb_path}"

    echo "Installing with apt..."
    sudo apt-get update -qq
    sudo apt-get install -y "${deb_path}"

    echo
    echo "Installation complete."
    echo "Run:    doorwatch"
    echo "Remove: sudo apt remove doorwatch"
}

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]:-$0}")" && pwd)"
if [ -f "${SCRIPT_DIR}/packaging/build_deb.sh" ] && [ -f "${SCRIPT_DIR}/main.py" ]; then
    run_local_install "${SCRIPT_DIR}"
    exit 0
fi

REPO_URL="${1:-${DOORWATCH_REPO_URL:-}}"
REPO_REF="${DOORWATCH_REPO_REF:-main}"
if [ -z "${REPO_URL}" ]; then
    echo "Remote bootstrap mode:"
    echo "Usage: curl -fsSL <raw-install.sh-url> | bash -s -- <repo-url>"
    echo "Example:"
    echo "  curl -fsSL https://raw.githubusercontent.com/<owner>/<repo>/main/install.sh | \\"
    echo "    bash -s -- https://github.com/<owner>/<repo>.git"
    exit 1
fi

if ! command -v git >/dev/null 2>&1; then
    echo "git not found. Installing git..."
    sudo apt-get update -qq
    sudo apt-get install -y git
fi

TMP_DIR="$(mktemp -d)"
trap 'rm -rf "${TMP_DIR}"' EXIT

echo "Cloning repository (${REPO_REF})..."
git clone --depth 1 --branch "${REPO_REF}" "${REPO_URL}" "${TMP_DIR}/repo"
bash "${TMP_DIR}/repo/install.sh"
