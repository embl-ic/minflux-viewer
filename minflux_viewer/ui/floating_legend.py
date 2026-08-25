"""Floating legend for colour-coded series (the stacked-iteration notation).

pyqtgraph's own `LegendItem` cannot serve here for two reasons: it draws an
"invisible eye" icon instead of a colour whenever the item it samples is hidden
— which every series is on the GPU path, where the points live on the OpenGL
canvas rather than in the scatter item — and it is a scene item, so it cannot
appear over the 3-D view at all. This widget draws the colours it is given, on
any renderer, and docks or floats like the colorbar.
"""

from __future__ import annotations

from collections.abc import Callable, Sequence

from PyQt6.QtCore import QEvent, QPoint, QRect, Qt
from PyQt6.QtGui import QColor, QMouseEvent, QPainter, QPen
from PyQt6.QtWidgets import QMenu, QWidget

#: Room for the swatch, its gap, and the panel's own padding.
_SWATCH = 12
_PADDING = 7
_GAP = 6
_MIN_WIDTH = 54
_MIN_HEIGHT = 28
_BORDER = 6


class FloatingLegend(QWidget):
    """A camera-independent legend that can be dragged and edge-resized."""

    def __init__(
        self,
        parent: QWidget,
        *,
        on_visibility_changed: Callable[[bool], None],
        on_state_changed: Callable[[], None],
        plot_area: Callable[[], QRect | None] | None = None,
        background_color: Callable[[], QColor | None] | None = None,
    ) -> None:
        super().__init__(parent)
        self._on_visibility_changed = on_visibility_changed
        self._on_state_changed = on_state_changed
        self._plot_area = plot_area
        self._background_color = background_color
        self._entries: list[tuple[str, tuple[int, int, int, int]]] = []
        self._title = ""
        self._manual_geometry = False
        self._float_geometry: QRect | None = None
        self._drag_start = QPoint()
        self._start_geometry = QRect()
        self._resize_edges: set[str] = set()
        self._dragging = False

        self.setMouseTracking(True)
        self.setMinimumSize(_MIN_WIDTH, _MIN_HEIGHT)
        parent.installEventFilter(self)
        self.hide()

    # ------------------------------------------------------------------
    # State
    # ------------------------------------------------------------------

    @property
    def docked(self) -> bool:
        """True while the legend is pinned to the plot's top-right corner."""
        return not self._manual_geometry

    def set_entries(
        self,
        entries: Sequence[tuple[str, Sequence[int]]],
        *,
        title: str = "",
    ) -> None:
        self._entries = [
            (str(label), tuple(int(channel) for channel in tuple(color)[:4]) if len(tuple(color)) >= 4
             else (*(int(channel) for channel in tuple(color)[:3]), 255))
            for label, color in entries
        ]
        self._title = str(title)
        if self.docked:
            self._place_default()
        self.update()

    def set_legend_visible(self, visible: bool) -> None:
        if visible and self._entries:
            self.show()
            if self.docked:
                self._place_default()
            self.raise_()
        else:
            self.hide()

    def serialized_geometry(self) -> list[int] | None:
        if not self._manual_geometry:
            return None
        geometry = self.geometry()
        return [geometry.x(), geometry.y(), geometry.width(), geometry.height()]

    def restore_geometry(self, geometry: Sequence[object] | None) -> None:
        if geometry is None:
            self._manual_geometry = False
            self._place_default()
            return
        try:
            x, y, width, height = (int(value) for value in geometry)
        except (TypeError, ValueError):
            self._manual_geometry = False
            self._place_default()
            return
        self._manual_geometry = True
        self.setGeometry(self._clamped_rect(QRect(x, y, width, height)))
        self._float_geometry = QRect(self.geometry())

    def set_docked(self, docked: bool, *, notify: bool = True) -> None:
        """Pin the legend to the plot corner, or lift it into a free panel."""
        docked = bool(docked)
        if docked == self.docked:
            return
        if docked:
            self._float_geometry = QRect(self.geometry())
            self._manual_geometry = False
            self._place_default()
        else:
            target = self._float_geometry or QRect(self.geometry())
            self._manual_geometry = True
            self.setGeometry(self._clamped_rect(target))
        self.update()
        if notify:
            self._on_state_changed()

    # ------------------------------------------------------------------
    # Geometry
    # ------------------------------------------------------------------

    def eventFilter(self, watched, event) -> bool:
        if watched is self.parentWidget() and event.type() == QEvent.Type.Resize:
            if self._manual_geometry:
                self.setGeometry(self._clamped_rect(self.geometry()))
            else:
                self._place_default()
        return super().eventFilter(watched, event)

    def sizeForEntries(self) -> tuple[int, int]:
        """The size this legend needs for its current entries."""
        metrics = self.fontMetrics()
        rows = self._entries or [("", (0, 0, 0, 0))]
        text_width = max(metrics.horizontalAdvance(label) for label, _ in rows)
        if self._title:
            text_width = max(text_width, metrics.horizontalAdvance(self._title))
        width = _PADDING * 2 + _SWATCH + _GAP + text_width
        height = _PADDING * 2 + metrics.height() * (
            len(rows) + (1 if self._title else 0)
        )
        return max(_MIN_WIDTH, width), max(_MIN_HEIGHT, height)

    def _place_default(self) -> None:
        """Top-right inside the plot area — where a legend is expected."""
        parent = self.parentWidget()
        if parent is None:
            return
        width, height = self.sizeForEntries()
        area = self._area_rect()
        right = area.right() if area is not None else parent.width()
        top = area.top() if area is not None else 0
        x = max(0, min(right - width - 10, max(0, parent.width() - width)))
        y = max(0, min(top + 10, max(0, parent.height() - height)))
        self.setGeometry(x, y, width, height)

    def _area_rect(self) -> QRect | None:
        if self._plot_area is None:
            return None
        try:
            area = self._plot_area()
        except Exception:
            return None
        return area if area is not None and area.isValid() else None

    def _clamped_rect(self, rect: QRect) -> QRect:
        parent = self.parentWidget()
        if parent is None:
            return rect
        width = max(_MIN_WIDTH, min(rect.width(), max(_MIN_WIDTH, parent.width())))
        height = max(_MIN_HEIGHT, min(rect.height(), max(_MIN_HEIGHT, parent.height())))
        x = max(0, min(rect.x(), max(0, parent.width() - width)))
        y = max(0, min(rect.y(), max(0, parent.height() - height)))
        return QRect(x, y, width, height)

    # ------------------------------------------------------------------
    # Interaction — only a floating legend can be moved
    # ------------------------------------------------------------------

    def _edges_at(self, pos: QPoint) -> set[str]:
        edges: set[str] = set()
        if pos.x() <= _BORDER:
            edges.add("left")
        elif pos.x() >= self.width() - _BORDER:
            edges.add("right")
        if pos.y() <= _BORDER:
            edges.add("top")
        elif pos.y() >= self.height() - _BORDER:
            edges.add("bottom")
        return edges

    def mousePressEvent(self, event: QMouseEvent) -> None:
        if event.button() != Qt.MouseButton.LeftButton or self.docked:
            super().mousePressEvent(event)
            return
        self._drag_start = event.globalPosition().toPoint()
        self._start_geometry = self.geometry()
        self._resize_edges = self._edges_at(event.position().toPoint())
        self._dragging = True
        event.accept()

    def mouseMoveEvent(self, event: QMouseEvent) -> None:
        if self.docked:
            self.unsetCursor()
            super().mouseMoveEvent(event)
            return
        edges = self._edges_at(event.position().toPoint())
        if not self._dragging:
            if edges & {"left", "right"} and edges & {"top", "bottom"}:
                cursor = Qt.CursorShape.SizeFDiagCursor
            elif edges & {"left", "right"}:
                cursor = Qt.CursorShape.SizeHorCursor
            elif edges & {"top", "bottom"}:
                cursor = Qt.CursorShape.SizeVerCursor
            else:
                cursor = Qt.CursorShape.SizeAllCursor
            self.setCursor(cursor)
            super().mouseMoveEvent(event)
            return
        delta = event.globalPosition().toPoint() - self._drag_start
        rect = QRect(self._start_geometry)
        if not self._resize_edges:
            rect.translate(delta)
        else:
            if "left" in self._resize_edges:
                rect.setLeft(rect.left() + delta.x())
            if "right" in self._resize_edges:
                rect.setRight(rect.right() + delta.x())
            if "top" in self._resize_edges:
                rect.setTop(rect.top() + delta.y())
            if "bottom" in self._resize_edges:
                rect.setBottom(rect.bottom() + delta.y())
        self.setGeometry(self._clamped_rect(rect.normalized()))
        event.accept()

    def mouseReleaseEvent(self, event: QMouseEvent) -> None:
        if self._dragging and event.button() == Qt.MouseButton.LeftButton:
            self._dragging = False
            self._float_geometry = QRect(self.geometry())
            self._on_state_changed()
            event.accept()
            return
        super().mouseReleaseEvent(event)

    def contextMenuEvent(self, event) -> None:
        menu = QMenu(self)
        menu.addAction("Hide legend", lambda: self._on_visibility_changed(False))
        menu.addAction(
            "Dock" if self._manual_geometry else "Undock",
            lambda: self.set_docked(self._manual_geometry),
        )
        menu.exec(event.globalPos())

    # ------------------------------------------------------------------
    # Painting
    # ------------------------------------------------------------------

    def _panel_color(self) -> QColor:
        if self._background_color is not None:
            try:
                color = self._background_color()
            except Exception:
                color = None
            if color is not None:
                backing = QColor(color)
                backing.setAlpha(205)
                return backing
        return QColor(255, 255, 255, 205)

    def paintEvent(self, _event) -> None:
        painter = QPainter(self)
        painter.setRenderHint(QPainter.RenderHint.Antialiasing)
        backing = self._panel_color()
        luminance = (
            0.2126 * backing.red() + 0.7152 * backing.green() + 0.0722 * backing.blue()
        )
        ink = QColor(25, 25, 25) if luminance >= 145 else QColor(235, 235, 235)

        panel = self.rect().adjusted(1, 1, -2, -2)
        painter.setBrush(backing)
        if self.docked:
            # A docked legend sits over the data: it needs the backing so the
            # labels stay readable, but no outline — nothing here is grabbable.
            painter.setPen(Qt.PenStyle.NoPen)
        else:
            painter.setPen(QPen(QColor(70, 70, 70, 190), 1.0))
        painter.drawRoundedRect(panel, 4.0, 4.0)

        metrics = painter.fontMetrics()
        y = _PADDING
        if self._title:
            painter.setPen(ink)
            painter.drawText(
                QRect(_PADDING, y, self.width() - 2 * _PADDING, metrics.height()),
                Qt.AlignmentFlag.AlignLeft | Qt.AlignmentFlag.AlignVCenter,
                self._title,
            )
            y += metrics.height()
        for label, color in self._entries:
            if y + metrics.height() > self.height() - _PADDING + metrics.height():
                break
            swatch = QRect(
                _PADDING,
                y + (metrics.height() - _SWATCH) // 2,
                _SWATCH,
                _SWATCH,
            )
            painter.setPen(QPen(QColor(45, 45, 45, 160), 1.0))
            painter.setBrush(QColor(*color))
            painter.drawRect(swatch)
            painter.setPen(ink)
            painter.drawText(
                QRect(
                    _PADDING + _SWATCH + _GAP,
                    y,
                    max(0, self.width() - _PADDING * 2 - _SWATCH - _GAP),
                    metrics.height(),
                ),
                Qt.AlignmentFlag.AlignLeft | Qt.AlignmentFlag.AlignVCenter,
                metrics.elidedText(
                    label,
                    Qt.TextElideMode.ElideRight,
                    max(0, self.width() - _PADDING * 2 - _SWATCH - _GAP),
                ),
            )
            y += metrics.height()
