#!/bin/bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
PKG_NAME="doorwatch"
ARCH="all"

VERSION="$(python3 - <<'PY'
from doorwatch import __version__
print(__version__)
PY
)"

DIST_DIR="${ROOT_DIR}/dist"
mkdir -p "${DIST_DIR}"

BUILD_ROOT="$(mktemp -d)"
trap 'rm -rf "${BUILD_ROOT}"' EXIT

PKG_DIR="${BUILD_ROOT}/${PKG_NAME}_${VERSION}_${ARCH}"
DEBIAN_DIR="${PKG_DIR}/DEBIAN"
OPT_DIR="${PKG_DIR}/opt/${PKG_NAME}"
BIN_DIR="${PKG_DIR}/usr/bin"
APP_DIR="${PKG_DIR}/usr/share/applications"
PIXMAP_DIR="${PKG_DIR}/usr/share/pixmaps"

install -d "${DEBIAN_DIR}" "${OPT_DIR}" "${BIN_DIR}" "${APP_DIR}" "${PIXMAP_DIR}"

cp -a "${ROOT_DIR}/doorwatch" "${OPT_DIR}/"
install -m 0644 "${ROOT_DIR}/main.py" "${OPT_DIR}/main.py"
install -m 0755 "${ROOT_DIR}/run.sh" "${OPT_DIR}/run.sh"
install -m 0644 "${ROOT_DIR}/README.md" "${OPT_DIR}/README.md"
install -m 0644 "${ROOT_DIR}/requirements.txt" "${OPT_DIR}/requirements.txt"

find "${OPT_DIR}" -type d -name "__pycache__" -prune -exec rm -rf {} +
find "${OPT_DIR}" -type f -name "*.pyc" -delete
rm -f "${OPT_DIR}/doorwatch/nohup.out"

cat > "${BIN_DIR}/doorwatch" <<'EOF'
#!/bin/sh
exec /opt/doorwatch/run.sh "$@"
EOF
chmod 0755 "${BIN_DIR}/doorwatch"

cat > "${APP_DIR}/doorwatch.desktop" <<'EOF'
[Desktop Entry]
Type=Application
Name=DoorWatch
Comment=Lightweight motion popup monitor
Exec=doorwatch
Icon=doorwatch
Terminal=false
Categories=Utility;Security;
StartupNotify=false
EOF

install -m 0644 "${ROOT_DIR}/doorwatch/assets/icon.svg" "${PIXMAP_DIR}/doorwatch.svg"

cat > "${DEBIAN_DIR}/control" <<EOF
Package: ${PKG_NAME}
Version: ${VERSION}
Section: utils
Priority: optional
Architecture: ${ARCH}
Depends: python3, python3-gi, python3-gi-cairo, gir1.2-gtk-3.0, gir1.2-notify-0.7, python3-opencv, python3-numpy, v4l-utils
Recommends: gir1.2-ayatanaappindicator3-0.1 | gir1.2-appindicator3-0.1
Maintainer: DoorWatch <noreply@example.com>
Description: Lightweight tray-based motion popup monitor
 DoorWatch runs in system tray and shows camera popup on motion.
EOF

cat > "${DEBIAN_DIR}/postinst" <<'EOF'
#!/bin/sh
set -e
if command -v update-desktop-database >/dev/null 2>&1; then
    update-desktop-database -q /usr/share/applications || true
fi
exit 0
EOF
chmod 0755 "${DEBIAN_DIR}/postinst"

DEB_PATH="${DIST_DIR}/${PKG_NAME}_${VERSION}_${ARCH}.deb"
dpkg-deb --build --root-owner-group "${PKG_DIR}" "${DEB_PATH}" >/dev/null
echo "${DEB_PATH}"
