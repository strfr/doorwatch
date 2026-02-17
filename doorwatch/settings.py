"""Tray settings window."""

from __future__ import annotations

import gi
gi.require_version("Gtk", "3.0")
from gi.repository import Gtk


class SettingsWindow(Gtk.Window):
    """GUI window for editing runtime config values."""

    def __init__(
        self,
        settings: dict[str, int | float | bool | str],
        camera_indices: list[int],
        autostart_enabled: bool,
        on_save_cb,
        on_refresh_cameras_cb,
    ):
        super().__init__(title="DoorWatch - Settings")
        self.set_default_size(520, 860)
        self.set_position(Gtk.WindowPosition.CENTER)

        self._settings = dict(settings)
        self._on_save_cb = on_save_cb
        self._on_refresh_cameras_cb = on_refresh_cameras_cb

        self._int_widgets: dict[str, Gtk.SpinButton] = {}
        self._float_widgets: dict[str, Gtk.SpinButton] = {}
        self._bool_widgets: dict[str, Gtk.CheckButton] = {}

        outer = Gtk.Box(orientation=Gtk.Orientation.VERTICAL, spacing=8)
        outer.set_margin_start(10)
        outer.set_margin_end(10)
        outer.set_margin_top(10)
        outer.set_margin_bottom(10)
        self.add(outer)

        scrolled = Gtk.ScrolledWindow()
        scrolled.set_policy(Gtk.PolicyType.NEVER, Gtk.PolicyType.AUTOMATIC)
        scrolled.set_vexpand(True)
        outer.pack_start(scrolled, True, True, 0)

        grid = Gtk.Grid(column_spacing=10, row_spacing=8)
        scrolled.add(grid)

        row = 0
        row = self._add_camera_row(grid, row, camera_indices)
        row = self._add_bool_row(grid, row, "Start on Login", "AUTOSTART_ENABLED", autostart_enabled)

        row = self._add_int_row(grid, row, "Capture Width", "CAPTURE_WIDTH", 160, 3840, 16)
        row = self._add_int_row(grid, row, "Capture Height", "CAPTURE_HEIGHT", 120, 2160, 16)
        row = self._add_int_row(grid, row, "Capture FPS", "CAPTURE_FPS", 1, 120, 1)
        row = self._add_int_row(grid, row, "Detection Interval", "DETECTION_INTERVAL", 1, 120, 1)
        row = self._add_int_row(grid, row, "Detection Frame Count", "DETECTION_FRAME_COUNT", 1, 8, 1)
        row = self._add_int_row(grid, row, "Min Contour Area", "MIN_CONTOUR_AREA", 100, 500000, 100)
        row = self._add_int_row(grid, row, "Rearm Frames", "MOTION_REARM_FRAMES", 1, 120, 1)
        row = self._add_int_row(grid, row, "Rearm Active Area", "MOTION_REARM_ACTIVE_AREA", 100, 500000, 100)
        row = self._add_float_row(grid, row, "Max Active Ratio", "MOTION_MAX_ACTIVE_RATIO", 0.05, 0.98, 0.01, 2)
        row = self._add_int_row(grid, row, "Idle Process Width", "PROCESS_WIDTH_IDLE", 64, 3840, 16)
        row = self._add_int_row(grid, row, "Idle Process Height", "PROCESS_HEIGHT_IDLE", 64, 2160, 16)
        row = self._add_int_row(grid, row, "Popup Hold (sec)", "POPUP_DURATION_SEC", 0, 120, 1)
        row = self._add_int_row(grid, row, "Popup Width", "POPUP_WIDTH", 240, 3840, 16)
        row = self._add_int_row(grid, row, "Popup Height", "POPUP_HEIGHT", 180, 2160, 16)
        row = self._add_int_row(grid, row, "Record History Count", "MOTION_RECORD_KEEP_COUNT", 1, 100, 1)
        row = self._add_bool_row(grid, row, "Use GPU", "MOTION_USE_GPU")
        row = self._add_choice_row(
            grid,
            row,
            "Subtractor Type",
            "MOTION_SUBTRACTOR_TYPE",
            options=["KNN", "MOG2"],
            default="KNN",
        )
        row = self._add_bool_row(grid, row, "Preprocess Grayscale", "MOTION_PREPROCESS_GRAYSCALE")
        row = self._add_int_row(grid, row, "Median Filter Kernel", "MOTION_FILTER_MEDIAN", 0, 15, 1)
        row = self._add_int_row(grid, row, "Gaussian Filter Kernel", "MOTION_FILTER_GAUSSIAN", 0, 15, 1)
        row = self._add_int_row(grid, row, "Shadow Threshold", "MOTION_SHADOW_THRESHOLD", 128, 255, 1)
        row = self._add_float_row(grid, row, "Lighting Luma Delta", "MOTION_LIGHTING_LUMA_DELTA", 1.0, 50.0, 0.5, 1)
        row = self._add_float_row(
            grid,
            row,
            "Lighting Active Ratio",
            "MOTION_LIGHTING_ACTIVE_RATIO",
            0.01,
            0.98,
            0.01,
            2,
        )
        row = self._add_float_row(
            grid,
            row,
            "Lighting Max Blob Ratio",
            "MOTION_LIGHTING_MAX_BLOB_RATIO",
            0.01,
            0.98,
            0.01,
            2,
        )

        self._status_label = Gtk.Label(label="")
        self._status_label.set_xalign(0.0)
        outer.pack_start(self._status_label, False, False, 0)

        buttons = Gtk.Box(orientation=Gtk.Orientation.HORIZONTAL, spacing=8)
        outer.pack_start(buttons, False, False, 0)

        apply_btn = Gtk.Button(label="Apply")
        apply_btn.connect("clicked", self._on_apply)
        buttons.pack_start(apply_btn, False, False, 0)

        close_btn = Gtk.Button(label="Close")
        close_btn.connect("clicked", self._on_close)
        buttons.pack_start(close_btn, False, False, 0)

        self.show_all()

    def _add_camera_row(self, grid: Gtk.Grid, row: int, camera_indices: list[int]) -> int:
        lbl = Gtk.Label(label="Camera")
        lbl.set_xalign(0.0)
        grid.attach(lbl, 0, row, 1, 1)

        box = Gtk.Box(orientation=Gtk.Orientation.HORIZONTAL, spacing=6)
        self._camera_combo = Gtk.ComboBoxText()
        box.pack_start(self._camera_combo, True, True, 0)

        refresh_btn = Gtk.Button(label="Refresh")
        refresh_btn.connect("clicked", self._on_refresh_cameras)
        box.pack_start(refresh_btn, False, False, 0)

        grid.attach(box, 1, row, 1, 1)
        self._populate_camera_combo(camera_indices)
        return row + 1

    def _add_int_row(self, grid: Gtk.Grid, row: int, label: str, key: str, low: int, high: int, step: int) -> int:
        lbl = Gtk.Label(label=label)
        lbl.set_xalign(0.0)
        grid.attach(lbl, 0, row, 1, 1)

        spin = Gtk.SpinButton.new_with_range(float(low), float(high), float(step))
        spin.set_value(float(self._settings.get(key, low)))
        spin.set_numeric(True)
        grid.attach(spin, 1, row, 1, 1)
        self._int_widgets[key] = spin
        return row + 1

    def _add_float_row(
        self,
        grid: Gtk.Grid,
        row: int,
        label: str,
        key: str,
        low: float,
        high: float,
        step: float,
        digits: int,
    ) -> int:
        lbl = Gtk.Label(label=label)
        lbl.set_xalign(0.0)
        grid.attach(lbl, 0, row, 1, 1)

        spin = Gtk.SpinButton.new_with_range(low, high, step)
        spin.set_digits(digits)
        spin.set_value(float(self._settings.get(key, low)))
        spin.set_numeric(True)
        grid.attach(spin, 1, row, 1, 1)
        self._float_widgets[key] = spin
        return row + 1

    def _add_bool_row(self, grid: Gtk.Grid, row: int, label: str, key: str, default: bool = False) -> int:
        lbl = Gtk.Label(label=label)
        lbl.set_xalign(0.0)
        grid.attach(lbl, 0, row, 1, 1)

        check = Gtk.CheckButton()
        check.set_active(bool(self._settings.get(key, default)))
        grid.attach(check, 1, row, 1, 1)
        self._bool_widgets[key] = check
        return row + 1

    def _add_choice_row(
        self,
        grid: Gtk.Grid,
        row: int,
        label: str,
        key: str,
        options: list[str],
        default: str,
    ) -> int:
        lbl = Gtk.Label(label=label)
        lbl.set_xalign(0.0)
        grid.attach(lbl, 0, row, 1, 1)

        combo = Gtk.ComboBoxText()
        current = str(self._settings.get(key, default)).strip().upper()
        active_idx = 0
        normalized = [str(opt).strip().upper() for opt in options]
        for i, opt in enumerate(normalized):
            combo.append_text(opt)
            if opt == current:
                active_idx = i
        combo.set_active(active_idx)
        grid.attach(combo, 1, row, 1, 1)
        setattr(self, f"_{key}_combo", combo)
        return row + 1

    def _populate_camera_combo(self, camera_indices: list[int]) -> None:
        self._camera_combo.remove_all()
        current = int(self._settings.get("CAMERA_INDEX", 0))

        indices = list(dict.fromkeys(sorted(camera_indices)))
        if current not in indices:
            indices.insert(0, current)

        active_idx = 0
        for i, cam_idx in enumerate(indices):
            self._camera_combo.append_text(str(cam_idx))
            if cam_idx == current:
                active_idx = i
        self._camera_combo.set_active(active_idx)

    def _on_refresh_cameras(self, *_args):
        try:
            camera_indices = self._on_refresh_cameras_cb()
            self._populate_camera_combo(camera_indices)
            self._status_label.set_text("Camera list refreshed.")
        except Exception as exc:
            self._status_label.set_text(f"Failed to refresh camera list: {exc}")

    def _collect_settings(self) -> dict[str, int | float | bool | str]:
        out: dict[str, int | float | bool | str] = {}
        camera_text = self._camera_combo.get_active_text()
        out["CAMERA_INDEX"] = int(camera_text) if camera_text is not None else 0

        for key, widget in self._int_widgets.items():
            value = int(widget.get_value())
            if key in {"MOTION_FILTER_MEDIAN", "MOTION_FILTER_GAUSSIAN"}:
                value = _normalize_kernel(value)
            out[key] = value

        for key, widget in self._float_widgets.items():
            out[key] = float(widget.get_value())

        for key, widget in self._bool_widgets.items():
            if key == "AUTOSTART_ENABLED":
                continue
            out[key] = bool(widget.get_active())

        subtractor_combo = getattr(self, "_MOTION_SUBTRACTOR_TYPE_combo", None)
        subtractor_value = subtractor_combo.get_active_text() if subtractor_combo else None
        out["MOTION_SUBTRACTOR_TYPE"] = (
            str(subtractor_value).strip().upper() if subtractor_value is not None else "KNN"
        )

        return out

    def _on_apply(self, *_args):
        settings = self._collect_settings()
        startup_enabled = bool(self._bool_widgets["AUTOSTART_ENABLED"].get_active())

        ok, message = self._on_save_cb(settings, startup_enabled)
        if ok:
            self._settings.update(settings)
            self._status_label.set_text(message)
        else:
            self._status_label.set_text(f"Error: {message}")

    def _on_close(self, *_args):
        self.destroy()


def _normalize_kernel(value: int) -> int:
    if value <= 1:
        return 0
    if value % 2 == 0:
        value += 1
    return max(3, value)
