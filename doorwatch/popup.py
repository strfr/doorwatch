"""GTK popup penceresi - kamera goruntusunu kisa sure gosterir."""

from __future__ import annotations

import gi
gi.require_version("Gtk", "3.0")
gi.require_version("GdkPixbuf", "2.0")
from gi.repository import Gtk, GdkPixbuf, Gdk

import cv2
import numpy as np
import time
import logging

log = logging.getLogger("doorwatch.popup")


class VideoPopup(Gtk.Window):
    """Hareket algilaninca acilan, sure dolunca kapanan popup."""

    def __init__(
        self,
        title: str = "DoorWatch - Hareket Algilandi",
        width: int = 480,
        height: int = 360,
        duration_sec: int = 3,
        on_close_cb=None,
    ):
        super().__init__(title=title)
        self.set_default_size(width, height)
        self.set_keep_above(True)
        self.set_urgency_hint(True)
        self.set_resizable(False)
        self.set_type_hint(Gdk.WindowTypeHint.DIALOG)
        self._window_width = int(width)
        self._window_height = int(height)
        self._edge_margin = 24

        # duration_sec <= 0 ise otomatik kapanma devre disi (hareket bazli kontrol).
        self._duration = max(0, int(duration_sec))
        self._start_time = time.monotonic()
        self._on_close_cb = on_close_cb
        self._closed = False
        self._info_text = "Hareket algilandi!"

        vbox = Gtk.Box(orientation=Gtk.Orientation.VERTICAL, spacing=4)
        self.add(vbox)

        self._image = Gtk.Image()
        vbox.pack_start(self._image, True, True, 0)

        self._label = Gtk.Label(label=self._info_text)
        self._label.set_margin_top(4)
        self._label.set_margin_bottom(4)
        vbox.pack_start(self._label, False, False, 0)

        btn_box = Gtk.Box(orientation=Gtk.Orientation.HORIZONTAL, spacing=8)
        btn_box.set_halign(Gtk.Align.CENTER)
        btn_box.set_margin_bottom(6)

        btn_close = Gtk.Button(label="Kapat")
        btn_close.connect("clicked", self._on_dismiss)
        btn_box.pack_start(btn_close, False, False, 0)

        vbox.pack_start(btn_box, False, False, 0)

        self.connect("destroy", self._on_destroy)
        self.connect("realize", self._on_realize)
        self.show_all()

    def _on_realize(self, *_args):
        self._move_to_second_monitor_bottom_right()

    def _move_to_second_monitor_bottom_right(self):
        display = Gdk.Display.get_default()
        if display is None:
            return

        monitor_count = display.get_n_monitors()
        if monitor_count <= 0:
            return

        if monitor_count > 1:
            monitor = display.get_monitor(1)
        else:
            monitor = display.get_primary_monitor() or display.get_monitor(0)

        if monitor is None:
            return

        geom = monitor.get_geometry()
        win_w, win_h = self.get_size()
        if win_w <= 1 or win_h <= 1:
            win_w = self._window_width
            win_h = self._window_height

        x = geom.x + geom.width - win_w - self._edge_margin
        y = geom.y + geom.height - win_h - self._edge_margin
        x = max(geom.x, x)
        y = max(geom.y, y)
        self.move(x, y)

    def update_frame(self, frame: np.ndarray) -> bool:
        if self._closed:
            return False

        if self._duration > 0:
            elapsed = time.monotonic() - self._start_time
            if elapsed >= self._duration:
                self._close()
                return False

            remaining = max(0, self._duration - int(elapsed))
            self._label.set_text(f"{self._info_text}  ({remaining}s)")
        else:
            self._label.set_text(self._info_text)

        try:
            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            h, w, ch = rgb.shape
            pixbuf = GdkPixbuf.Pixbuf.new_from_data(
                rgb.tobytes(), GdkPixbuf.Colorspace.RGB, False, 8, w, h, w * ch
            )

            alloc = self._image.get_allocation()
            tw = max(alloc.width, 320)
            th = max(alloc.height, 240)
            # Orani koruyarak olcekle; goruntu esneyip bozulmasin.
            scale = min(tw / w, th / h)
            nw = max(1, int(w * scale))
            nh = max(1, int(h * scale))
            scaled = pixbuf.scale_simple(nw, nh, GdkPixbuf.InterpType.BILINEAR)
            self._image.set_from_pixbuf(scaled)
        except Exception as exc:
            log.error("Popup kare guncellenemedi: %s", exc)

        return True

    def set_info_text(self, text: str) -> None:
        self._info_text = text

    def _on_dismiss(self, *_args):
        self._close()

    def _on_destroy(self, *_args):
        self._closed = True

    def _close(self):
        if not self._closed:
            self._closed = True
            if self._on_close_cb:
                self._on_close_cb()
            self.destroy()

    def close(self):
        self._close()

    @property
    def is_closed(self) -> bool:
        return self._closed
