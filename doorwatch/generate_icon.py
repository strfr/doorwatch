#!/usr/bin/env python3
"""Generates a simple tray icon for DoorWatch."""

import os

ICON_SVG = '''<?xml version="1.0" encoding="UTF-8"?>
<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 64 64" width="64" height="64">
  <!-- Door -->
  <rect x="16" y="8" width="32" height="48" rx="3" ry="3"
        fill="#5B4A3F" stroke="#3E3028" stroke-width="2"/>
  <!-- Peephole -->
  <circle cx="32" cy="26" r="6" fill="#1A1A2E" stroke="#B0B0B0" stroke-width="1.5"/>
  <!-- Lens reflection -->
  <circle cx="30" cy="24" r="2" fill="#4FC3F7" opacity="0.7"/>
  <!-- Door handle -->
  <circle cx="42" cy="34" r="2.5" fill="#FFD54F" stroke="#F9A825" stroke-width="1"/>
  <!-- Base -->
  <rect x="12" y="54" width="40" height="4" rx="2" fill="#6D4C41"/>
  <!-- Detection wave -->
  <path d="M 20 18 Q 12 26 20 34" fill="none" stroke="#4FC3F7"
        stroke-width="1.5" stroke-dasharray="3,2" opacity="0.6"/>
  <path d="M 44 18 Q 52 26 44 34" fill="none" stroke="#4FC3F7"
        stroke-width="1.5" stroke-dasharray="3,2" opacity="0.6"/>
</svg>'''


def generate_icon():
    assets_dir = os.path.join(os.path.dirname(__file__), "assets")
    os.makedirs(assets_dir, exist_ok=True)

    svg_path = os.path.join(assets_dir, "icon.svg")
    png_path = os.path.join(assets_dir, "icon.png")

    # Save SVG
    with open(svg_path, "w") as f:
        f.write(ICON_SVG)
    print(f"SVG saved: {svg_path}")

    # Try to render PNG
    try:
        import gi
        gi.require_version("Rsvg", "2.0")
        gi.require_version("GdkPixbuf", "2.0")
        from gi.repository import Rsvg, GdkPixbuf
        import cairo

        handle = Rsvg.Handle.new_from_file(svg_path)
        surface = cairo.ImageSurface(cairo.FORMAT_ARGB32, 64, 64)
        ctx = cairo.Context(surface)
        viewport = Rsvg.Rectangle()
        viewport.x, viewport.y, viewport.width, viewport.height = 0, 0, 64, 64
        handle.render_document(ctx, viewport)
        surface.write_to_png(png_path)
        print(f"PNG saved: {png_path}")
    except Exception as exc:
        print(f"PNG render failed ({exc}), using SVG.")
        # Fallback: try PIL/cairosvg
        try:
            from PIL import Image
            import cairosvg
            cairosvg.svg2png(url=svg_path, write_to=png_path,
                             output_width=64, output_height=64)
            print(f"PNG saved (cairosvg): {png_path}")
        except Exception:
            # Create a direct SVG symlink as a final fallback
            if not os.path.exists(png_path):
                os.symlink("icon.svg", png_path)
                print("SVG -> PNG symlink created.")


if __name__ == "__main__":
    generate_icon()
