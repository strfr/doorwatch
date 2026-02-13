"""
Canlı kamera görüntüleme penceresi – her zaman açık,
algılama durumunu da gösteren GTK penceresi.
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
    """Canlı kamera penceresi – sürekli açık, algılama bilgisi gösterir."""

    def __init__(self, title: str = "DoorWatch – Kamera",
                 width: int = 640, height: int = 520,
                 on_close_cb=None,
                 start_visible: bool = False):
        super().__init__(title=title)
        self.set_default_size(width, height)
        self.set_position(Gtk.WindowPosition.CENTER)
        self.set_icon_name("camera-web")

        self._on_close_cb = on_close_cb
        self._visible = bool(start_visible)

        # Ana kutu
        vbox = Gtk.Box(orientation=Gtk.Orientation.VERTICAL, spacing=2)
        self.add(vbox)

        # Görüntü alanı
        self._image = Gtk.Image()
        self._image.set_size_request(width, height - 80)
        vbox.pack_start(self._image, True, True, 0)

        # Durum çubuğu
        hbox = Gtk.Box(orientation=Gtk.Orientation.HORIZONTAL, spacing=8)
        hbox.set_margin_start(8)
        hbox.set_margin_end(8)
        hbox.set_margin_top(4)
        hbox.set_margin_bottom(4)

        self._status_label = Gtk.Label(label="⏳ Başlatılıyor...")
        self._status_label.set_xalign(0)
        hbox.pack_start(self._status_label, True, True, 0)

        self._detection_label = Gtk.Label(label="")
        self._detection_label.set_xalign(1)
        hbox.pack_start(self._detection_label, False, False, 0)

        vbox.pack_start(hbox, False, False, 0)

        # Pencere kapatıldığında gizle, yok etme
        self.connect("delete-event", self._on_delete)
        if self._visible:
            self.show_all()
        else:
            self.hide()

    def _on_delete(self, widget, event):
        """Pencere kapatma isteği – yok etmek yerine gizle."""
        self.hide()
        self._visible = False
        if self._on_close_cb:
            self._on_close_cb()
        return True  # True = yok etme

    def toggle_visibility(self):
        """Pencereyi göster/gizle."""
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
        Canlı kareyi güncelle. Bounding box'ları çizer.
        GLib.idle_add ile çağrılmalı.
        """
        if not self._visible:
            return True

        try:
            display = frame.copy()

            # Bounding box çiz
            if rects:
                for (x, y, w, h) in rects:
                    cv2.rectangle(display, (x, y), (x + w, y + h),
                                  (0, 255, 0), 2)
                    cv2.putText(display, "Hareket", (x, y - 10),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.6,
                                (0, 255, 0), 2)

            # BGR → RGB → Pixbuf
            rgb = cv2.cvtColor(display, cv2.COLOR_BGR2RGB)
            h, w, ch = rgb.shape
            pixbuf = GdkPixbuf.Pixbuf.new_from_data(
                rgb.tobytes(), GdkPixbuf.Colorspace.RGB, False, 8,
                w, h, w * ch)

            # Pencere boyutuna ölçekle
            alloc = self._image.get_allocation()
            tw = max(alloc.width, 320)
            th = max(alloc.height, 240)
            # En-boy oranını koru
            scale = min(tw / w, th / h)
            nw, nh = int(w * scale), int(h * scale)
            scaled = pixbuf.scale_simple(nw, nh,
                                         GdkPixbuf.InterpType.BILINEAR)
            self._image.set_from_pixbuf(scaled)

            # Durum etiketleri
            if status_text is not None:
                self._status_label.set_text(status_text)
            if detection_text is not None:
                self._detection_label.set_text(detection_text)

        except Exception as exc:
            log.error("Viewer kare güncellenemedi: %s", exc)

        return False  # GLib.idle_add tek seferlik
