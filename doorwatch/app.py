"""DoorWatch ana uygulama: hafif hareket algilama + popup + tray."""

from __future__ import annotations

import gi
gi.require_version("Gtk", "3.0")
from gi.repository import Gtk, GLib

import cv2
import threading
import time
import os
import logging
import glob
from datetime import datetime

from doorwatch import config
from doorwatch import config_store
from doorwatch import autostart
from doorwatch.detector import PersonDetector
from doorwatch.popup import VideoPopup
from doorwatch.viewer import CameraViewer
from doorwatch.tray import TrayIcon
from doorwatch.settings import SettingsWindow

log = logging.getLogger("doorwatch")


class DoorWatchApp:
    """Hareket algilayip popup gosteren tray uygulamasi."""

    def __init__(self):
        self._setup_logging()
        config_store.apply_runtime(config_store.load_user_settings())

        self._muted = False
        self._running = True
        self._popup_active = False
        self._current_popup: VideoPopup | None = None
        self._frame_counter = 0
        self._motion_event_latched = False
        self._no_motion_frames = 0
        self._popup_last_motion_at = 0.0
        self._last_rects: list[tuple[int, int, int, int]] = []
        self._last_detection_text = ""
        self._camera_reopen_requested = False
        self._settings_window: SettingsWindow | None = None

        self._cap = None
        self._detector_lock = threading.Lock()
        self._detector = self._build_detector()

        self._viewer = CameraViewer(
            title="DoorWatch - Camera",
            width=config.CAPTURE_WIDTH,
            height=config.CAPTURE_HEIGHT + 60,
            on_close_cb=self._on_viewer_closed,
            start_visible=False,
        )

        self._tray = TrayIcon(
            app_name=config.APP_NAME,
            icon_path=config.APP_ICON,
            on_mute_toggle=self._on_mute_toggle,
            on_quit=self._on_quit,
            on_show_viewer=self._on_show_viewer,
            on_show_settings=self._on_show_settings,
        )

        log.info("DoorWatch baslatildi.")

    def _build_detector(self) -> PersonDetector:
        return PersonDetector(
            consecutive_frames=config.DETECTION_FRAME_COUNT,
            min_contour_area=config.MIN_CONTOUR_AREA,
            use_gpu=config.MOTION_USE_GPU,
            max_active_ratio=config.MOTION_MAX_ACTIVE_RATIO,
            preprocess_grayscale=config.MOTION_PREPROCESS_GRAYSCALE,
            filter_median=config.MOTION_FILTER_MEDIAN,
            filter_gaussian=config.MOTION_FILTER_GAUSSIAN,
        )

    def _setup_logging(self):
        os.makedirs(config.LOG_DIR, exist_ok=True)
        log_file = os.path.join(config.LOG_DIR, f"doorwatch_{datetime.now():%Y%m%d}.log")
        logging.basicConfig(
            level=logging.INFO,
            format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
            handlers=[
                logging.FileHandler(log_file, encoding="utf-8"),
                logging.StreamHandler(),
            ],
        )

    def _available_camera_indices(self) -> list[int]:
        indices: list[int] = []
        for dev_path in sorted(glob.glob("/dev/video*")):
            base = os.path.basename(dev_path)
            suffix = base.removeprefix("video")
            if suffix.isdigit():
                indices.append(int(suffix))

        if not indices:
            for idx in range(0, 8):
                cap = cv2.VideoCapture(idx)
                if cap.isOpened():
                    indices.append(idx)
                cap.release()

        indices = sorted(set(indices))
        if config.CAMERA_INDEX not in indices:
            indices.insert(0, config.CAMERA_INDEX)
        return indices

    def _on_show_settings(self):
        if self._settings_window and self._settings_window.get_visible():
            self._settings_window.present()
            return

        self._settings_window = SettingsWindow(
            settings=config_store.current_settings(),
            camera_indices=self._available_camera_indices(),
            autostart_enabled=autostart.is_enabled(),
            on_save_cb=self._on_settings_apply,
            on_refresh_cameras_cb=self._available_camera_indices,
        )
        self._settings_window.connect("destroy", self._on_settings_closed)

    def _on_settings_closed(self, *_args):
        self._settings_window = None

    def _on_settings_apply(
        self,
        settings: dict[str, int | float | bool],
        startup_enabled: bool,
    ) -> tuple[bool, str]:
        try:
            old_cam = (
                config.CAMERA_INDEX,
                config.CAPTURE_WIDTH,
                config.CAPTURE_HEIGHT,
                config.CAPTURE_FPS,
            )
            old_detector = (
                config.DETECTION_FRAME_COUNT,
                config.MIN_CONTOUR_AREA,
                config.MOTION_USE_GPU,
                config.MOTION_MAX_ACTIVE_RATIO,
                config.MOTION_PREPROCESS_GRAYSCALE,
                config.MOTION_FILTER_MEDIAN,
                config.MOTION_FILTER_GAUSSIAN,
            )

            config_store.apply_runtime(settings)
            storage = config_store.persist_settings(settings)
            self._set_startup_enabled(startup_enabled)

            with self._detector_lock:
                detector_cfg = (
                    config.DETECTION_FRAME_COUNT,
                    config.MIN_CONTOUR_AREA,
                    config.MOTION_USE_GPU,
                    config.MOTION_MAX_ACTIVE_RATIO,
                    config.MOTION_PREPROCESS_GRAYSCALE,
                    config.MOTION_FILTER_MEDIAN,
                    config.MOTION_FILTER_GAUSSIAN,
                )
                if detector_cfg != old_detector:
                    self._detector = self._build_detector()
                    self._detector.reset()

            new_cam = (
                config.CAMERA_INDEX,
                config.CAPTURE_WIDTH,
                config.CAPTURE_HEIGHT,
                config.CAPTURE_FPS,
            )
            if new_cam != old_cam:
                self._camera_reopen_requested = True
                GLib.idle_add(self._viewer.resize, config.CAPTURE_WIDTH, config.CAPTURE_HEIGHT + 60)

            log.info("Settings uygulandi.")
            if storage == "config.py":
                return True, "Settings kaydedildi (config.py) ve uygulandi."
            return True, "Settings kaydedildi (~/.config/doorwatch/settings.json) ve uygulandi."
        except Exception as exc:
            log.exception("Settings uygulanamadi: %s", exc)
            return False, str(exc)

    def _set_startup_enabled(self, enabled: bool) -> None:
        project_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
        exec_path = os.path.join(project_dir, "run.sh")
        icon_path = os.path.abspath(config.APP_ICON)
        autostart.set_enabled(
            enabled=enabled,
            exec_path=exec_path,
            icon_path=icon_path,
            app_name=config.APP_NAME,
        )

    def _open_camera(self) -> bool:
        indices = [config.CAMERA_INDEX]
        if config.CAMERA_INDEX != 0:
            indices.append(0)

        for index in indices:
            try:
                cap = cv2.VideoCapture(index)
                if not cap.isOpened():
                    cap.release()
                    continue
                self._cap = cap
                self._cap.set(cv2.CAP_PROP_FRAME_WIDTH, config.CAPTURE_WIDTH)
                self._cap.set(cv2.CAP_PROP_FRAME_HEIGHT, config.CAPTURE_HEIGHT)
                self._cap.set(cv2.CAP_PROP_FPS, config.CAPTURE_FPS)
                log.info(
                    "Kamera acildi: index=%d (%dx%d @%dfps)",
                    index,
                    config.CAPTURE_WIDTH,
                    config.CAPTURE_HEIGHT,
                    config.CAPTURE_FPS,
                )
                return True
            except Exception as exc:
                log.warning("Kamera index=%d acilamadi: %s", index, exc)

        log.error("Kamera acilamadi: index=%s (fallback index=0 denendi)", config.CAMERA_INDEX)
        return False

    def _close_camera(self):
        if self._cap and self._cap.isOpened():
            self._cap.release()
            log.info("Kamera kapatildi.")

    def _prepare_detection_frame(
        self,
        frame,
        viewer_visible: bool,
    ) -> tuple[object, float, float, int]:
        """
        Camera window kapaliyken algilamayi dusuk cozunurlukte yap.
        Donus:
          proc_frame: algilama girdisi
          sx, sy: proc -> full frame olcekleri
          min_area_proc: proc frame icin alan esigi
        """
        fh, fw = frame.shape[:2]
        if viewer_visible:
            return frame, 1.0, 1.0, config.MIN_CONTOUR_AREA

        target_w = max(64, min(config.PROCESS_WIDTH_IDLE, fw))
        target_h = max(64, min(config.PROCESS_HEIGHT_IDLE, fh))
        proc_frame = cv2.resize(frame, (target_w, target_h), interpolation=cv2.INTER_AREA)

        sx = fw / target_w
        sy = fh / target_h
        min_area_proc = max(40, int(config.MIN_CONTOUR_AREA / (sx * sy)))
        return proc_frame, sx, sy, min_area_proc

    def _scale_rects(
        self,
        rects: list[tuple[int, int, int, int]],
        sx: float,
        sy: float,
        frame_shape,
    ) -> list[tuple[int, int, int, int]]:
        if sx == 1.0 and sy == 1.0:
            return rects

        fh, fw = frame_shape[:2]
        scaled: list[tuple[int, int, int, int]] = []
        for x, y, w, h in rects:
            rx = int(x * sx)
            ry = int(y * sy)
            rw = int(w * sx)
            rh = int(h * sy)
            if rw <= 0 or rh <= 0:
                continue
            rx = max(0, min(rx, fw - 1))
            ry = max(0, min(ry, fh - 1))
            rw = min(rw, fw - rx)
            rh = min(rh, fh - ry)
            if rw > 0 and rh > 0:
                scaled.append((rx, ry, rw, rh))
        return scaled

    def _has_significant_motion(self, rects: list[tuple[int, int, int, int]]) -> bool:
        if not rects:
            return False
        max_area = max(w * h for _, _, w, h in rects)
        return max_area >= config.MOTION_REARM_ACTIVE_AREA

    def _camera_thread(self):
        while self._running and not self._open_camera():
            time.sleep(2)

        if not self._running:
            return

        while self._running:
            if self._camera_reopen_requested:
                self._camera_reopen_requested = False
                self._close_camera()
                while self._running and not self._open_camera():
                    time.sleep(1)
                if not self._running:
                    break
                continue

            ret, frame = self._cap.read()
            if not ret:
                log.warning("Kare okunamadi, tekrar deneniyor...")
                time.sleep(0.5)
                self._close_camera()
                while self._running and not self._open_camera():
                    time.sleep(2)
                if not self._running:
                    break
                continue

            self._frame_counter += 1

            rects = self._last_rects
            has_motion = False
            detection_text = self._last_detection_text
            viewer_visible = self._viewer.is_visible

            if self._frame_counter % config.DETECTION_INTERVAL == 0:
                proc_frame, sx, sy, min_area_proc = self._prepare_detection_frame(
                    frame,
                    viewer_visible=viewer_visible,
                )
                with self._detector_lock:
                    confirmed, proc_rects, has_motion = self._detector.update(
                        proc_frame,
                        min_contour_area=min_area_proc,
                    )
                rects = self._scale_rects(proc_rects, sx, sy, frame.shape)
                significant_motion = has_motion and self._has_significant_motion(rects)

                # Hareket olayi kilidi: popup bir kez tetiklenir, hareket tamamen
                # bittikten sonra tekrar tetiklenebilir.
                if significant_motion:
                    self._no_motion_frames = 0
                    if self._popup_active:
                        self._popup_last_motion_at = time.monotonic()
                else:
                    self._no_motion_frames += 1
                    if self._no_motion_frames >= config.MOTION_REARM_FRAMES:
                        self._motion_event_latched = False

                if has_motion:
                    detection_text = "Hareket"
                else:
                    detection_text = ""

                self._last_rects = rects
                self._last_detection_text = detection_text

                if confirmed and significant_motion:
                    if not self._motion_event_latched:
                        self._motion_event_latched = True
                        if self._muted:
                            log.info("Hareket algilandi, silent mode acik. Popup gosterilmedi.")
                        else:
                            self._process_motion(frame)

            if viewer_visible:
                status_text = "Silent Mode" if self._muted else "Running"
                GLib.idle_add(
                    self._viewer.update_frame,
                    frame.copy(),
                    rects,
                    status_text,
                    detection_text,
                )

            if self._popup_active and self._current_popup:
                GLib.idle_add(self._update_popup_frame, frame.copy())

            time.sleep(1.0 / config.CAPTURE_FPS)

        self._close_camera()

    def _process_motion(self, frame):
        if self._popup_active:
            return

        with self._detector_lock:
            self._detector.reset()
        log.info("Hareket algilandi, popup aciliyor.")
        GLib.idle_add(self._show_popup, frame.copy())

    def _show_popup(self, frame):
        if self._popup_active:
            return False

        self._popup_active = True
        self._popup_last_motion_at = time.monotonic()
        self._current_popup = VideoPopup(
            title="DoorWatch - Hareket Algilandi",
            width=config.POPUP_WIDTH,
            height=config.POPUP_HEIGHT,
            duration_sec=0,
            on_close_cb=self._on_popup_closed,
        )
        self._current_popup.set_info_text("Hareket algilandi")
        self._current_popup.update_frame(frame)
        return False

    def _update_popup_frame(self, frame):
        if self._current_popup and not self._current_popup.is_closed:
            if config.POPUP_DURATION_SEC > 0:
                no_motion_elapsed = time.monotonic() - self._popup_last_motion_at
                if no_motion_elapsed >= config.POPUP_DURATION_SEC:
                    self._current_popup.close()
                    return False
            still_open = self._current_popup.update_frame(frame)
            if not still_open:
                self._on_popup_closed()
        return False

    def _on_popup_closed(self):
        self._popup_active = False
        self._current_popup = None

    def _on_mute_toggle(self, muted: bool):
        self._muted = muted
        log.info("Silent mode: %s", "ON" if muted else "OFF")

    def _on_quit(self):
        self._running = False

    def _on_show_viewer(self):
        self._viewer.toggle_visibility()

    def _on_viewer_closed(self):
        log.debug("Camera window gizlendi.")

    def run(self):
        cam_thread = threading.Thread(
            target=self._camera_thread,
            daemon=True,
            name="CameraThread",
        )
        cam_thread.start()

        try:
            Gtk.main()
        except KeyboardInterrupt:
            log.info("Ctrl+C ile cikis.")
        finally:
            self._running = False
            cam_thread.join(timeout=3)
            self._close_camera()
            log.info("DoorWatch kapatildi.")
