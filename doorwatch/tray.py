"""Sistem tepsisi ikonu."""

import gi
gi.require_version("Gtk", "3.0")
try:
    gi.require_version("AyatanaAppIndicator3", "0.1")
    from gi.repository import AyatanaAppIndicator3 as AppIndicator
    INDICATOR_AVAILABLE = True
except (ValueError, ImportError):
    try:
        gi.require_version("AppIndicator3", "0.1")
        from gi.repository import AppIndicator3 as AppIndicator
        INDICATOR_AVAILABLE = True
    except (ValueError, ImportError):
        INDICATOR_AVAILABLE = False

from gi.repository import Gtk
import logging

log = logging.getLogger("doorwatch.tray")


class TrayIcon:
    """Tray menusu: Camera Window, Settings, Silent Mode, Exit."""

    def __init__(self, app_name: str, icon_path: str,
                 on_mute_toggle=None, on_quit=None, on_show_viewer=None,
                 on_show_settings=None):
        self._muted = False
        self._on_mute_toggle = on_mute_toggle
        self._on_quit = on_quit
        self._on_show_viewer = on_show_viewer
        self._on_show_settings = on_show_settings

        if INDICATOR_AVAILABLE:
            self._build_indicator(app_name, icon_path)
        else:
            self._build_status_icon(app_name, icon_path)

    def _build_indicator(self, app_name, icon_path):
        import os

        if os.path.isfile(icon_path):
            self._indicator = AppIndicator.Indicator.new(
                app_name, icon_path, AppIndicator.IndicatorCategory.APPLICATION_STATUS
            )
        else:
            self._indicator = AppIndicator.Indicator.new(
                app_name, "camera-web", AppIndicator.IndicatorCategory.APPLICATION_STATUS
            )
        self._indicator.set_status(AppIndicator.IndicatorStatus.ACTIVE)

        menu = Gtk.Menu()

        viewer_item = Gtk.MenuItem(label="Camera Window")
        viewer_item.connect("activate", self._on_viewer_clicked)
        menu.append(viewer_item)

        settings_item = Gtk.MenuItem(label="Settings")
        settings_item.connect("activate", self._on_settings_clicked)
        menu.append(settings_item)

        sep0 = Gtk.SeparatorMenuItem()
        menu.append(sep0)

        self._mute_item = Gtk.CheckMenuItem(label="Silent Mode")
        self._mute_item.set_active(False)
        self._mute_item.connect("toggled", self._on_mute_clicked)
        menu.append(self._mute_item)

        sep1 = Gtk.SeparatorMenuItem()
        menu.append(sep1)

        exit_item = Gtk.MenuItem(label="Exit")
        exit_item.connect("activate", self._on_quit_clicked)
        menu.append(exit_item)

        menu.show_all()
        self._indicator.set_menu(menu)
        self._use_indicator = True
        log.info("Tray: AppIndicator kullaniliyor.")

    def _build_status_icon(self, app_name, icon_path):
        import os

        if os.path.isfile(icon_path):
            self._status_icon = Gtk.StatusIcon.new_from_file(icon_path)
        else:
            self._status_icon = Gtk.StatusIcon.new_from_icon_name("camera-web")

        self._status_icon.set_title(app_name)
        self._status_icon.set_tooltip_text(app_name)
        self._status_icon.connect("activate", self._on_viewer_clicked)
        self._status_icon.connect("popup-menu", self._on_popup_menu)
        self._use_indicator = False
        log.info("Tray: StatusIcon fallback kullaniliyor.")

    def _on_popup_menu(self, icon, button, event_time):
        menu = Gtk.Menu()

        viewer_item = Gtk.MenuItem(label="Camera Window")
        viewer_item.connect("activate", self._on_viewer_clicked)
        menu.append(viewer_item)

        settings_item = Gtk.MenuItem(label="Settings")
        settings_item.connect("activate", self._on_settings_clicked)
        menu.append(settings_item)

        sep0 = Gtk.SeparatorMenuItem()
        menu.append(sep0)

        mute_item = Gtk.CheckMenuItem(label="Silent Mode")
        mute_item.set_active(self._muted)
        mute_item.connect("toggled", self._on_mute_clicked)
        menu.append(mute_item)

        sep1 = Gtk.SeparatorMenuItem()
        menu.append(sep1)

        exit_item = Gtk.MenuItem(label="Exit")
        exit_item.connect("activate", self._on_quit_clicked)
        menu.append(exit_item)

        menu.show_all()
        menu.popup(None, None, None, None, button, event_time)

    def _on_viewer_clicked(self, *_args):
        if self._on_show_viewer:
            self._on_show_viewer()

    def _on_mute_clicked(self, widget):
        if isinstance(widget, Gtk.CheckMenuItem):
            self._muted = widget.get_active()
        else:
            self._muted = not self._muted
        log.info("Silent mode: %s", "ON" if self._muted else "OFF")
        if self._on_mute_toggle:
            self._on_mute_toggle(self._muted)

    def _on_settings_clicked(self, *_args):
        if self._on_show_settings:
            self._on_show_settings()

    def _on_quit_clicked(self, *_args):
        log.info("Cikis istendi.")
        if self._on_quit:
            self._on_quit()
        Gtk.main_quit()

    @property
    def is_muted(self) -> bool:
        return self._muted

    def update_tooltip(self, text: str):
        if not self._use_indicator and hasattr(self, "_status_icon"):
            self._status_icon.set_tooltip_text(text)
