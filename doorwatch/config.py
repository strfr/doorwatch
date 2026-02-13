"""DoorWatch yapilandirma sabitleri."""

import os


def _xdg_dir(env_name: str, fallback_suffix: str) -> str:
    value = os.environ.get(env_name)
    if value:
        return value
    return os.path.join(os.path.expanduser("~"), fallback_suffix)

# Kamera
CAMERA_DEVICE = "/dev/video1"
CAMERA_INDEX = 1
CAPTURE_WIDTH = 640
CAPTURE_HEIGHT = 480
# Kamera/popup/viewer akiciligi.
# Islem yukunu sabit tutmak icin DETECTION_INTERVAL ile birlikte ayarlanir.
CAPTURE_FPS = 60

# Hareket algilama (hafif)
DETECTION_FRAME_COUNT = 1
# Efektif algilama hizi ~= CAPTURE_FPS / DETECTION_INTERVAL.
# 15 / 3 ~= 5 FPS isleme.
DETECTION_INTERVAL = 12
# Hassasiyetin ana dugmesi:
# Dusuk = daha hassas (daha cok false positive)
# Yuksek = daha secici
MIN_CONTOUR_AREA = 2000
MOTION_USE_GPU = True
# Isleme asamasinda (sadece algilama tarafi) siyah-beyaz + filtre.
# Popup ve camera window her zaman renkli/orijinal kareden gosterilir.
MOTION_PREPROCESS_GRAYSCALE = True
MOTION_FILTER_MEDIAN = 3
MOTION_FILTER_GAUSSIAN = 5
# Bir popup'tan sonra yeniden tetikleme icin gereken "hareket yok" kare sayisi
MOTION_REARM_FRAMES = 3
# Rearm ve tetikleme hesabinda "anlamli hareket" alt siniri.
# Bu alandan kucuk hareketler gurultu gibi kabul edilir.
MOTION_REARM_ACTIVE_AREA = 5000
# Tum kare birden hareketli gorunuyorsa (kamera titremesi/isik oynama),
# bu oran ustunde hareket ignore edilir.
MOTION_MAX_ACTIVE_RATIO = 0.75
# Camera window kapaliyken algilama boyutu (CPU/GPU yukunu azaltir)
PROCESS_WIDTH_IDLE = 640
PROCESS_HEIGHT_IDLE = 360

# Popup
# Hareket bittikten sonra popup'in kapanmasi icin beklenecek sure (sn).
POPUP_DURATION_SEC = 3
POPUP_WIDTH = 480
POPUP_HEIGHT = 360

# Kayit (paketli kurulumda da kullanici yazabilir olmali)
_STATE_HOME = _xdg_dir("XDG_STATE_HOME", ".local/state")
_DATA_HOME = _xdg_dir("XDG_DATA_HOME", ".local/share")
LOG_DIR = os.path.join(_STATE_HOME, "doorwatch")
SNAPSHOT_DIR = os.path.join(_DATA_HOME, "doorwatch", "snapshots")

# Tray
APP_NAME = "DoorWatch"
APP_ICON = os.path.join(os.path.dirname(__file__), "assets", "icon.svg")
