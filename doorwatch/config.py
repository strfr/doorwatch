"""DoorWatch configuration constants."""

import os


def _xdg_dir(env_name: str, fallback_suffix: str) -> str:
    value = os.environ.get(env_name)
    if value:
        return value
    return os.path.join(os.path.expanduser("~"), fallback_suffix)

# Camera
CAMERA_DEVICE = "/dev/video1"
CAMERA_INDEX = 1
CAPTURE_WIDTH = 640
CAPTURE_HEIGHT = 480
# Camera/popup/viewer smoothness.
# Tune together with DETECTION_INTERVAL to keep processing load stable.
CAPTURE_FPS = 60

# Motion detection (lightweight)
DETECTION_FRAME_COUNT = 1
# Effective processing rate ~= CAPTURE_FPS / DETECTION_INTERVAL.
# 15 / 3 ~= 5 FPS processing.
DETECTION_INTERVAL = 12
# Main sensitivity knob:
# Lower = more sensitive (more false positives)
# Higher = more selective
MIN_CONTOUR_AREA = 2000
MOTION_USE_GPU = True
# Processing-stage grayscale + filtering (detection path only).
# Popup and camera window always show original color frames.
MOTION_PREPROCESS_GRAYSCALE = True
MOTION_FILTER_MEDIAN = 3
MOTION_FILTER_GAUSSIAN = 5
# Number of no-motion frames required to rearm after a popup
MOTION_REARM_FRAMES = 3
# Minimum "significant motion" area for rearm/trigger logic.
# Motions smaller than this area are treated as noise.
MOTION_REARM_ACTIVE_AREA = 5000
# If most of the frame is active (camera shake/lighting change),
# motion above this ratio is ignored.
MOTION_MAX_ACTIVE_RATIO = 0.75
# Detection resolution while camera window is hidden (reduces CPU/GPU load)
PROCESS_WIDTH_IDLE = 640
PROCESS_HEIGHT_IDLE = 360

# Popup
# Delay before closing popup after motion ends (sec).
POPUP_DURATION_SEC = 3
POPUP_WIDTH = 480
POPUP_HEIGHT = 360

# Runtime output directories (must remain user-writable in packaged installs)
_STATE_HOME = _xdg_dir("XDG_STATE_HOME", ".local/state")
_DATA_HOME = _xdg_dir("XDG_DATA_HOME", ".local/share")
LOG_DIR = os.path.join(_STATE_HOME, "doorwatch")
SNAPSHOT_DIR = os.path.join(_DATA_HOME, "doorwatch", "snapshots")

# Tray
APP_NAME = "DoorWatch"
APP_ICON = os.path.join(os.path.dirname(__file__), "assets", "icon.svg")
