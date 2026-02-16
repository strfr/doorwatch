"""DoorWatch main app: lightweight motion detection + popup + tray."""

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
import subprocess
import shutil
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
    """Tray app that detects motion and shows a popup."""

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
        self._popup_record_lock = threading.RLock()
        self._popup_motion_started_wall_at: float | None = None
        self._popup_motion_last_seen_wall_at: float | None = None
        self._popup_motion_event_count = 0
        self._last_popup_motion_record_text = "No motion record yet."
        self._last_motion_video_path: str | None = None
        self._motion_record_paths: list[str] = []
        self._popup_record_video_path: str | None = None
        self._popup_record_writer: cv2.VideoWriter | None = None
        self._popup_record_target_fps = 12.0
        self._popup_record_last_write_at = 0.0
        self._popup_record_written_frames = 0
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
            on_show_last_record=self._on_show_last_motion_record,
            on_save_last_record=self._on_save_last_motion_record,
        )
        self._refresh_motion_record_history(prune=True)

        log.info("DoorWatch started.")

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

            self._refresh_motion_record_history(prune=True)
            log.info("Settings applied.")
            if storage == "config.py":
                return True, "Settings saved (config.py) and applied."
            return True, "Settings saved (~/.config/doorwatch/settings.json) and applied."
        except Exception as exc:
            log.exception("Failed to apply settings: %s", exc)
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
                    "Camera opened: index=%d (%dx%d @%dfps)",
                    index,
                    config.CAPTURE_WIDTH,
                    config.CAPTURE_HEIGHT,
                    config.CAPTURE_FPS,
                )
                return True
            except Exception as exc:
                log.warning("Failed to open camera index=%d: %s", index, exc)

        log.error("Unable to open camera: index=%s (fallback index=0 tried)", config.CAMERA_INDEX)
        return False

    def _close_camera(self):
        if self._cap and self._cap.isOpened():
            self._cap.release()
            log.info("Camera closed.")

    def _prepare_detection_frame(
        self,
        frame,
        viewer_visible: bool,
    ) -> tuple[object, float, float, int]:
        """
        Run detection on lower resolution when camera window is hidden.
        Returns:
          proc_frame: detection input frame
          sx, sy: proc -> full frame scale
          min_area_proc: contour area threshold in proc scale
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

    def _begin_popup_motion_record(self, wall_time: float) -> None:
        with self._popup_record_lock:
            self._popup_motion_started_wall_at = wall_time
            self._popup_motion_last_seen_wall_at = wall_time
            self._popup_motion_event_count = 1

    def _touch_popup_motion_record(self, wall_time: float) -> None:
        with self._popup_record_lock:
            if self._popup_motion_started_wall_at is None:
                self._popup_motion_started_wall_at = wall_time
                self._popup_motion_last_seen_wall_at = wall_time
                self._popup_motion_event_count = 1
                return
            self._popup_motion_last_seen_wall_at = wall_time
            self._popup_motion_event_count += 1

    def _clear_popup_motion_record(self) -> None:
        with self._popup_record_lock:
            self._popup_motion_started_wall_at = None
            self._popup_motion_last_seen_wall_at = None
            self._popup_motion_event_count = 0
            self._popup_record_last_write_at = 0.0
            self._popup_record_written_frames = 0

    def _safe_remove_file(self, path: str | None) -> None:
        if not path:
            return
        try:
            if os.path.isfile(path):
                os.remove(path)
        except Exception as exc:
            log.warning("Failed to remove file %s: %s", path, exc)

    def _motion_history_limit(self) -> int:
        try:
            return max(1, int(config.MOTION_RECORD_KEEP_COUNT))
        except Exception:
            return 1

    def _motion_clip_files_newest_first(self) -> list[str]:
        pattern = os.path.join(config.SNAPSHOT_DIR, "last_motion_*.avi")
        paths = [path for path in glob.glob(pattern) if os.path.isfile(path)]
        paths.sort(key=os.path.getmtime, reverse=True)
        return paths

    def _refresh_motion_record_history(self, prune: bool) -> None:
        os.makedirs(config.SNAPSHOT_DIR, exist_ok=True)
        paths = self._motion_clip_files_newest_first()
        if prune:
            limit = self._motion_history_limit()
            for old_path in paths[limit:]:
                self._safe_remove_file(old_path)
            paths = paths[:limit]

        with self._popup_record_lock:
            self._motion_record_paths = paths
            self._last_motion_video_path = paths[0] if paths else None

    def _start_popup_video_recording(self, frame) -> None:
        try:
            os.makedirs(config.SNAPSHOT_DIR, exist_ok=True)
            h, w = frame.shape[:2]
            target_fps = float(min(15, max(5, int(config.CAPTURE_FPS))))
            clip_name = f"last_motion_{datetime.now():%Y%m%d_%H%M%S}.avi"
            clip_path = os.path.join(config.SNAPSHOT_DIR, clip_name)
            fourcc = cv2.VideoWriter_fourcc(*"MJPG")
            writer = cv2.VideoWriter(clip_path, fourcc, target_fps, (w, h))
            if not writer.isOpened():
                writer.release()
                log.error("Failed to start motion clip writer: %s", clip_path)
                return
        except Exception as exc:
            log.exception("Failed to initialize motion clip writer: %s", exc)
            return

        with self._popup_record_lock:
            self._popup_record_writer = writer
            self._popup_record_video_path = clip_path
            self._popup_record_target_fps = target_fps
            self._popup_record_last_write_at = 0.0
            self._popup_record_written_frames = 0

        self._record_popup_frame(frame, time.monotonic())
        log.info("Motion clip recording started: %s", clip_path)

    def _record_popup_frame(self, frame, now_mono: float) -> None:
        with self._popup_record_lock:
            writer = self._popup_record_writer
            if writer is None:
                return

            min_interval = 1.0 / max(1.0, self._popup_record_target_fps)
            if self._popup_record_last_write_at > 0:
                if (now_mono - self._popup_record_last_write_at) < min_interval:
                    return

            try:
                writer.write(frame)
                self._popup_record_last_write_at = now_mono
                self._popup_record_written_frames += 1
            except Exception as exc:
                log.exception("Failed writing motion clip frame: %s", exc)
                try:
                    writer.release()
                except Exception:
                    pass
                failed_path = self._popup_record_video_path
                self._popup_record_writer = None
                self._popup_record_video_path = None
                self._popup_record_last_write_at = 0.0
                self._popup_record_written_frames = 0
                if failed_path:
                    GLib.idle_add(self._safe_remove_file, failed_path)

    def _stop_popup_video_recording(self, keep_as_last: bool) -> None:
        remove_path: str | None = None
        kept_path: str | None = None
        with self._popup_record_lock:
            writer = self._popup_record_writer
            clip_path = self._popup_record_video_path
            frame_count = self._popup_record_written_frames

            self._popup_record_writer = None
            self._popup_record_video_path = None
            self._popup_record_last_write_at = 0.0
            self._popup_record_written_frames = 0

            if writer is not None:
                try:
                    writer.release()
                except Exception as exc:
                    log.warning("Failed to release motion clip writer: %s", exc)

            if keep_as_last and clip_path and frame_count > 0:
                kept_path = clip_path
            else:
                if clip_path:
                    remove_path = clip_path

        if remove_path:
            self._safe_remove_file(remove_path)
        if kept_path:
            log.info("Motion clip saved: %s (%d frames)", kept_path, frame_count)
            self._refresh_motion_record_history(prune=True)
        else:
            self._refresh_motion_record_history(prune=False)

    def _popup_motion_summary_text(self) -> str:
        with self._popup_record_lock:
            if self._popup_motion_started_wall_at is None:
                return "Motion detected"

            started = datetime.fromtimestamp(self._popup_motion_started_wall_at).strftime("%H:%M:%S")
            if self._popup_motion_last_seen_wall_at is None:
                last_seen = started
            else:
                last_seen = datetime.fromtimestamp(self._popup_motion_last_seen_wall_at).strftime("%H:%M:%S")
            return (
                f"Motion detected | Start: {started} | "
                f"Last: {last_seen} | Events: {self._popup_motion_event_count}"
            )

    def _last_motion_dialog_text(self) -> str:
        with self._popup_record_lock:
            if self._popup_active and self._popup_motion_started_wall_at is not None:
                return (
                    f"Current popup record:\n{self._popup_motion_summary_text()}\n"
                    "Video is still recording. Close popup to finalize."
                )

            text = self._last_popup_motion_record_text
            if self._last_motion_video_path and os.path.isfile(self._last_motion_video_path):
                return f"{text}\nVideo file: {self._last_motion_video_path}"
            return text

    def _copy_last_motion_to_videos(self) -> tuple[bool, str]:
        with self._popup_record_lock:
            popup_active = self._popup_active
            src_path = self._last_motion_video_path

        if popup_active:
            return False, "Close current popup first to finalize the motion clip."
        if not src_path or not os.path.isfile(src_path):
            return False, "No last motion record to save."

        try:
            videos_dir = config.VIDEOS_DIR
            os.makedirs(videos_dir, exist_ok=True)

            ext = os.path.splitext(src_path)[1] or ".avi"
            stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            base_name = f"doorwatch_saved_motion_{stamp}"
            dst_path = os.path.join(videos_dir, f"{base_name}{ext}")
            suffix = 1
            while os.path.exists(dst_path):
                dst_path = os.path.join(videos_dir, f"{base_name}_{suffix}{ext}")
                suffix += 1

            shutil.copy2(src_path, dst_path)
            return True, dst_path
        except Exception as exc:
            log.exception("Failed to save last motion record: %s", exc)
            return False, str(exc)

    def _camera_thread(self):
        while self._running and not self._open_camera():
            time.sleep(2)

        if not self._running:
            return

        while self._running:
            try:
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
                    log.warning("Frame read failed, retrying...")
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

                    # Motion event latch: trigger popup once, rearm only after
                    # motion has been gone long enough.
                    if significant_motion:
                        self._no_motion_frames = 0
                        if self._popup_active:
                            self._popup_last_motion_at = time.monotonic()
                            self._touch_popup_motion_record(time.time())
                    else:
                        self._no_motion_frames += 1
                        if self._no_motion_frames >= config.MOTION_REARM_FRAMES:
                            self._motion_event_latched = False

                    if has_motion:
                        detection_text = "Motion"
                    else:
                        detection_text = ""

                    self._last_rects = rects
                    self._last_detection_text = detection_text

                    if confirmed and significant_motion:
                        if not self._motion_event_latched:
                            self._motion_event_latched = True
                            if self._muted:
                                log.info("Motion detected, silent mode enabled. Popup not shown.")
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
                    self._record_popup_frame(frame, time.monotonic())
                    GLib.idle_add(self._update_popup_frame, frame.copy())

                time.sleep(1.0 / config.CAPTURE_FPS)
            except Exception as exc:
                log.exception("Unhandled error in camera loop: %s", exc)
                time.sleep(0.2)

        self._close_camera()

    def _process_motion(self, frame):
        if self._popup_active:
            return

        with self._detector_lock:
            self._detector.reset()
        log.info("Motion detected, opening popup.")
        GLib.idle_add(self._show_popup, frame.copy())

    def _show_popup(self, frame):
        if self._popup_active:
            return False

        self._popup_active = True
        # Clear previous closed record when a new popup session starts.
        with self._popup_record_lock:
            self._last_popup_motion_record_text = "No motion record yet."
        self._stop_popup_video_recording(keep_as_last=False)
        self._popup_last_motion_at = time.monotonic()
        self._begin_popup_motion_record(time.time())
        self._start_popup_video_recording(frame)
        self._current_popup = VideoPopup(
            title="DoorWatch - Motion Detected",
            width=config.POPUP_WIDTH,
            height=config.POPUP_HEIGHT,
            duration_sec=0,
            on_close_cb=self._on_popup_closed,
        )
        self._current_popup.set_info_text(self._popup_motion_summary_text())
        self._current_popup.update_frame(frame)
        return False

    def _update_popup_frame(self, frame):
        if self._current_popup and not self._current_popup.is_closed:
            if config.POPUP_DURATION_SEC > 0:
                no_motion_elapsed = time.monotonic() - self._popup_last_motion_at
                if no_motion_elapsed >= config.POPUP_DURATION_SEC:
                    self._current_popup.close()
                    return False
            self._current_popup.set_info_text(self._popup_motion_summary_text())
            still_open = self._current_popup.update_frame(frame)
            if not still_open:
                self._on_popup_closed()
        return False

    def _on_popup_closed(self):
        if self._popup_motion_started_wall_at is not None:
            record_text = self._popup_motion_summary_text()
            with self._popup_record_lock:
                self._last_popup_motion_record_text = record_text
            log.info("Popup motion record: %s", record_text)
        self._stop_popup_video_recording(keep_as_last=True)
        self._popup_active = False
        self._current_popup = None
        self._clear_popup_motion_record()

    def _on_mute_toggle(self, muted: bool):
        self._muted = muted
        log.info("Silent mode: %s", "ON" if muted else "OFF")

    def _on_quit(self):
        self._running = False

    def _on_show_viewer(self):
        self._viewer.toggle_visibility()

    def _on_show_last_motion_record(self):
        try:
            with self._popup_record_lock:
                popup_active = self._popup_active
                clip_path = self._last_motion_video_path

            if popup_active:
                dialog = Gtk.MessageDialog(
                    transient_for=self._viewer if self._viewer.is_visible else None,
                    flags=Gtk.DialogFlags.MODAL,
                    message_type=Gtk.MessageType.INFO,
                    buttons=Gtk.ButtonsType.OK,
                    text="Last Motion Record",
                )
                dialog.format_secondary_text(self._last_motion_dialog_text())
                dialog.run()
                dialog.destroy()
                return

            if clip_path and os.path.isfile(clip_path):
                subprocess.Popen(
                    ["xdg-open", clip_path],
                    stdout=subprocess.DEVNULL,
                    stderr=subprocess.DEVNULL,
                    start_new_session=True,
                )
                log.info("Opening last motion clip: %s", clip_path)
                return

            dialog = Gtk.MessageDialog(
                transient_for=self._viewer if self._viewer.is_visible else None,
                flags=Gtk.DialogFlags.MODAL,
                message_type=Gtk.MessageType.INFO,
                buttons=Gtk.ButtonsType.OK,
                text="Last Motion Record",
            )
            dialog.format_secondary_text(self._last_motion_dialog_text())
            dialog.run()
            dialog.destroy()
        except Exception as exc:
            log.exception("Failed to open last motion record: %s", exc)

    def _on_save_last_motion_record(self):
        ok, result = self._copy_last_motion_to_videos()
        try:
            dialog = Gtk.MessageDialog(
                transient_for=self._viewer if self._viewer.is_visible else None,
                flags=Gtk.DialogFlags.MODAL,
                message_type=Gtk.MessageType.INFO if ok else Gtk.MessageType.ERROR,
                buttons=Gtk.ButtonsType.OK,
                text="Save Last Record",
            )
            if ok:
                dialog.format_secondary_text(f"Saved permanently to:\n{result}")
            else:
                dialog.format_secondary_text(result)
            dialog.run()
            dialog.destroy()
        except Exception as exc:
            log.exception("Failed to show save-last-record dialog: %s", exc)

        if ok:
            try:
                subprocess.Popen(
                    ["xdg-open", os.path.dirname(result)],
                    stdout=subprocess.DEVNULL,
                    stderr=subprocess.DEVNULL,
                    start_new_session=True,
                )
            except Exception as exc:
                log.warning("Failed to open videos directory: %s", exc)

    def _on_viewer_closed(self):
        log.debug("Camera window hidden.")

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
            log.info("Exit via Ctrl+C.")
        finally:
            self._running = False
            cam_thread.join(timeout=3)
            self._close_camera()
            log.info("DoorWatch stopped.")
