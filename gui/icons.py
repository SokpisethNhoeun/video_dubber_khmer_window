from __future__ import annotations

from PyQt6.QtGui import QIcon

from gui.theme import get_palette, get_saved_theme


def icon(name: str, *, color: str | None = None) -> QIcon:
    """Return a themed qtawesome icon (e.g. icon('mdi.movie-outline'))."""
    try:
        import qtawesome as qta
    except ImportError:
        # Keep source checkouts usable before optional GUI dependencies are installed.
        return QIcon()

    resolved_color = color or get_palette(get_saved_theme())["accent"]
    return qta.icon(name, color=resolved_color)
