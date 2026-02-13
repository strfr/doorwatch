"""
Live camera viewer window.
Shows detection state and overlays.
"""

import gi
gi.require_version("Gtk", "3.0")
gi.require_version("GdkPixbuf", "2.0")
from gi.repository import Gtk, GdkPixbuf, GLib, Gdk

import cv2
import numpy as np
import logging

log = logging.getLogger("doorwatch.viewer")


class CameraViewer(Gtk.Window):
    """Live camera window that shows detection info."""

    def __init__(self, title: str = "DoorWatch - Camera",
                 width: int = 640, height: int = 520,
                 on_close_cb=None,
                 start_visible: bool = False):
        super().__init__(title=title)
        self.set_default_size(width, height)
        self.set_position(Gtk.WindowPosition.CENTER)
        self.set_icon_name("camera-web")

        self._on_close_cb = on_close_cb
        self._visible = bool(start_visible)

        # Root container
        vbox = Gtk.Box(orientation=Gtk.Orientation.VERTICAL, spacing=2)
        self.add(vbox)

        # Video area
        self._image = Gtk.Image()
        self._image.set_size_request(width, height - 80)
        vbox.pack_start(self._image, True, True, 0)

        # Status bar
        hbox = Gtk.Box(orientation=Gtk.Orientation.HORIZONTAL, spacing=8)
        hbox.set_margin_start(8)
        hbox.set_margin_end(8)
        hbox.set_margin_top(4)
        hbox.set_margin_bottom(4)

        self._status_label = Gtk.Label(label="Starting...")
        self._status_label.set_xalign(0)
        hbox.pack_start(self._status_label, True, True, 0)

        self._detection_label = Gtk.Label(label="")
        self._detection_label.set_xalign(1)
        hbox.pack_start(self._detection_label, False, False, 0)

        vbox.pack_start(hbox, False, False, 0)

        # Hide instead of destroy when user closes window
        self.connect("delete-event", self._on_delete)
        if self._visible:
            self.show_all()
        else:
            self.hide()

    def _on_delete(self, widget, event):
        """Handle window close request by hiding it."""
        self.hide()
        self._visible = False
        if self._on_close_cb:
            self._on_close_cb()
        return True  # True = prevent destroy

    def toggle_visibility(self):
        """Toggle viewer visibility."""
        if self._visible:
            self.hide()
            self._visible = False
        else:
            self.show_all()
            self.present()
            self._visible = True
        return self._visible

    def show_window(self):
        if not self._visible:
            self.show_all()
            self.present()
            self._visible = True

    @property
    def is_visible(self) -> bool:
        return self._visible

    def update_frame(self, frame: np.ndarray, rects=None,
                     status_text: str = None,
                     detection_text: str = None) -> bool:
        """
        Update live frame and draw bounding boxes.
        Should be called with GLib.idle_add.
        """
        if not self._visible:
            return True

        try:
            display = frame.copy()

            # Draw detection boxes
            if rects:
                for (x, y, w, h) in rects:
                    cv2.rectangle(display, (x, y), (x + w, y + h),
                                  (0, 255, 0), 2)
                    cv2.putText(display, "Motion", (x, y - 10),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.6,
                                (0, 255, 0), 2)

            # BGR -> RGB -> Pixbuf
            rgb = cv2.cvtColor(display, cv2.COLOR_BGR2RGB)
            h, w, ch = rgb.shape
            pixbuf = GdkPixbuf.Pixbuf.new_from_data(
                rgb.tobytes(), GdkPixbuf.Colorspace.RGB, False, 8,
                w, h, w * ch)

            # Scale to image widget
            alloc = self._image.get_allocation()
            tw = max(alloc.width, 320)
            th = max(alloc.height, 240)
            # Keep aspect ratio
            scale = min(tw / w, th / h)
            nw, nh = int(w * scale), int(h * scale)
            scaled = pixbuf.scale_simple(nw, nh,
                                         GdkPixbuf.InterpType.BILINEAR)
            self._image.set_from_pixbuf(scaled)

            # Status labels
            if status_text is not None:
                self._status_label.set_text(status_text)
            if detection_text is not None:
                self._detection_label.set_text(detection_text)

        except Exception as exc:
            log.error("Failed to update viewer frame: %s", exc)

        return False  # one-shot callback for GLib.idle_add
