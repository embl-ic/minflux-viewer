"""Floating, movable colorbar overlay for 2-D and OpenGL plot widgets."""

from __future__ import annotations

from collections.abc import Callable, Sequence

import numpy as np
from PyQt6.QtCore import QEvent, QPoint, QRect, Qt
from PyQt6.QtGui import QColor, QLinearGradient, QMouseEvent, QPainter, QPen
from PyQt6.QtWidgets import QMenu, QWidget

from .attribute_help import apply_attribute_menu_tooltips


class FloatingColorBar(QWidget):
    """A camera-independent colorbar that can be dragged and edge-resized."""

    _BORDER = 6
    _MIN_WIDTH = 58
    _MIN_HEIGHT = 58
    _DEFAULT_VERTICAL_WIDTH = 84
    _DEFAULT_HORIZONTAL_HEIGHT = 64

    def __init__(
        self,
        parent: QWidget,
        *,
        on_visibility_changed: Callable[[bool], None],
        on_customize: Callable[[], None],
        on_state_changed: Callable[[], None],
        attribute_names: Callable[[], Sequence[str]],
        current_attribute: Callable[[], str],
        on_attribute_changed: Callable[[str], None],
        plot_area: Callable[[], QRect | None] | None = None,
        background_color: Callable[[], QColor | None] | None = None,
    ) -> None:
        super().__init__(parent)
        self._on_visibility_changed = on_visibility_changed
        self._on_customize = on_customize
        self._on_state_changed = on_state_changed
        self._attribute_names = attribute_names
        self._current_attribute = current_attribute
        self._on_attribute_changed = on_attribute_changed
        self._plot_area = plot_area
        self._background_color = background_color
        self._orientation = "vertical"
        self._show_values = True
        self._lut = np.column_stack(
            [np.arange(256), np.arange(256), np.arange(256), np.full(256, 255)]
        ).astype(np.uint8)
        self._lo = 0.0
        self._hi = 1.0
        self._label = "C"
        self._manual_geometry = False
        self._float_geometry: QRect | None = None
        self._drag_start = QPoint()
        self._start_geometry = QRect()
        self._resize_edges: set[str] = set()
        self._dragging = False
        self._owner_reserved_orientation: str | None = None
        margins = parent.contentsMargins()
        self._parent_margins = (
            margins.left(),
            margins.top(),
            margins.right(),
            margins.bottom(),
        )

        self.setMouseTracking(True)
        self.setMinimumSize(self._MIN_WIDTH, self._MIN_HEIGHT)
        self.resize(128, 270)
        parent.installEventFilter(self)
        self._place_default()

    @property
    def orientation(self) -> str:
        return self._orientation

    @property
    def show_values(self) -> bool:
        return self._show_values

    @property
    def uses_default_placement(self) -> bool:
        return not self._manual_geometry

    @property
    def docked(self) -> bool:
        """True while the bar is pinned to the plot border (not floating)."""
        return not self._manual_geometry

    def set_docked(self, docked: bool, *, notify: bool = True) -> None:
        """Pin the bar to the plot border, or lift it into a floating panel.

        A docked bar reserves its own gutter, cannot be moved or resized, and
        is painted flush with the plot (no panel outline).  Undocking keeps it
        exactly where it is; the next dock/undock round trip restores that
        floating rectangle.
        """
        docked = bool(docked)
        if docked == self.docked:
            return
        if docked:
            self._float_geometry = QRect(self.geometry())
            self._manual_geometry = False
            if not self.isHidden():
                self._reserve_owner_space()
            self._sync_parent_margins()
            self._place_default()
        else:
            target = self._float_geometry or QRect(self.geometry())
            self._release_owner_space()
            self._manual_geometry = True
            self._sync_parent_margins()
            self.setGeometry(self._clamped_rect(target))
        self.update()
        if notify:
            self._on_state_changed()

    def set_bar_visible(self, visible: bool) -> None:
        """Show/hide the bar and reserve plot space for its default placement."""
        if visible:
            was_hidden = self.isHidden()
            self.show()
            if was_hidden and not self._manual_geometry:
                self._reserve_owner_space()
            self._sync_parent_margins()
            if not self._manual_geometry:
                self._place_default()
            self.raise_()
        else:
            was_visible = not self.isHidden()
            self.hide()
            self._sync_parent_margins()
            if was_visible:
                self._release_owner_space()

    def serialized_geometry(self) -> list[int] | None:
        if not self._manual_geometry:
            return None
        geometry = self.geometry()
        return [geometry.x(), geometry.y(), geometry.width(), geometry.height()]

    def restore_geometry(self, geometry: Sequence[object] | None) -> None:
        if geometry is None:
            was_manual = self._manual_geometry
            self._manual_geometry = False
            if was_manual and not self.isHidden():
                self._reserve_owner_space()
            self._sync_parent_margins()
            self._place_default()
            return
        try:
            x, y, width, height = (int(value) for value in geometry)
        except (TypeError, ValueError):
            was_manual = self._manual_geometry
            self._manual_geometry = False
            if was_manual and not self.isHidden():
                self._reserve_owner_space()
            self._sync_parent_margins()
            self._place_default()
            return
        if not self._manual_geometry:
            self._release_owner_space()
        self._manual_geometry = True
        self._sync_parent_margins()
        self.setGeometry(self._clamped_rect(QRect(x, y, width, height)))
        self._float_geometry = QRect(self.geometry())

    def set_orientation(self, orientation: str, *, notify: bool = True) -> None:
        orientation = "horizontal" if orientation == "horizontal" else "vertical"
        if orientation == self._orientation and not self._manual_geometry:
            return
        self._release_owner_space()
        self._manual_geometry = False
        # The remembered floating rect belongs to the old orientation, so the
        # next undock should lift the bar in place instead of restoring it.
        self._float_geometry = None
        self._orientation = orientation
        if not self.isHidden():
            self._reserve_owner_space()
        self._sync_parent_margins()
        self._place_default()
        self.update()
        if notify:
            self._on_state_changed()

    def set_show_values(self, show: bool, *, notify: bool = True) -> None:
        show = bool(show)
        if show == self._show_values:
            return
        self._show_values = show
        self.update()
        if notify:
            self._on_state_changed()

    def set_color_data(
        self,
        lut: np.ndarray,
        lo: float,
        hi: float,
        label: str,
    ) -> None:
        table = np.asarray(lut, dtype=np.uint8)
        if table.ndim == 2 and table.shape[0] >= 2 and table.shape[1] in (3, 4):
            if table.shape[1] == 3:
                table = np.column_stack([table, np.full(table.shape[0], 255)])
            self._lut = table
        self._lo = float(lo)
        self._hi = float(hi)
        self._label = str(label)
        self.update()

    def eventFilter(self, watched, event) -> bool:
        if watched is self.parentWidget() and event.type() == QEvent.Type.Resize:
            if self._manual_geometry:
                self.setGeometry(self._clamped_rect(self.geometry()))
            else:
                self._place_default()
        return super().eventFilter(watched, event)

    def _sync_parent_margins(self) -> None:
        parent = self.parentWidget()
        if parent is None:
            return
        left, top, right, bottom = self._parent_margins
        if not self.isHidden() and not self._manual_geometry:
            if self._orientation == "vertical":
                right += self._DEFAULT_VERTICAL_WIDTH
            else:
                top += self._DEFAULT_HORIZONTAL_HEIGHT
        parent.setContentsMargins(left, top, right, bottom)

    def _reserve_owner_space(self) -> None:
        if self._owner_reserved_orientation is not None:
            return
        owner = self.window()
        if owner is None:
            return
        if self._orientation == "vertical":
            owner.resize(owner.width() + self._DEFAULT_VERTICAL_WIDTH, owner.height())
        else:
            owner.resize(owner.width(), owner.height() + self._DEFAULT_HORIZONTAL_HEIGHT)
        self._owner_reserved_orientation = self._orientation

    def _release_owner_space(self) -> None:
        orientation = self._owner_reserved_orientation
        if orientation is None:
            return
        owner = self.window()
        if owner is not None:
            if orientation == "vertical":
                owner.resize(
                    max(owner.minimumWidth(), owner.width() - self._DEFAULT_VERTICAL_WIDTH),
                    owner.height(),
                )
            else:
                owner.resize(
                    owner.width(),
                    max(owner.minimumHeight(), owner.height() - self._DEFAULT_HORIZONTAL_HEIGHT),
                )
        self._owner_reserved_orientation = None

    def _place_default(self) -> None:
        parent = self.parentWidget()
        if parent is None:
            return
        if self._orientation == "vertical":
            width = min(
                self._DEFAULT_VERTICAL_WIDTH,
                max(self._MIN_WIDTH, parent.width()),
            )
            self.setGeometry(parent.width() - width, 0, width, parent.height())
        else:
            height = min(
                self._DEFAULT_HORIZONTAL_HEIGHT,
                max(self._MIN_HEIGHT, parent.height()),
            )
            self.setGeometry(0, 0, parent.width(), height)

    def _clamped_rect(self, rect: QRect) -> QRect:
        parent = self.parentWidget()
        if parent is None:
            return rect
        width = max(self._MIN_WIDTH, min(rect.width(), max(self._MIN_WIDTH, parent.width())))
        height = max(self._MIN_HEIGHT, min(rect.height(), max(self._MIN_HEIGHT, parent.height())))
        x = max(0, min(rect.x(), max(0, parent.width() - width)))
        y = max(0, min(rect.y(), max(0, parent.height() - height)))
        return QRect(x, y, width, height)

    def _edges_at(self, pos: QPoint) -> set[str]:
        edges: set[str] = set()
        if pos.x() <= self._BORDER:
            edges.add("left")
        elif pos.x() >= self.width() - self._BORDER:
            edges.add("right")
        if pos.y() <= self._BORDER:
            edges.add("top")
        elif pos.y() >= self.height() - self._BORDER:
            edges.add("bottom")
        return edges

    def _update_cursor(self, edges: set[str]) -> None:
        if edges in ({"left", "top"}, {"right", "bottom"}):
            cursor = Qt.CursorShape.SizeFDiagCursor
        elif edges in ({"right", "top"}, {"left", "bottom"}):
            cursor = Qt.CursorShape.SizeBDiagCursor
        elif edges & {"left", "right"}:
            cursor = Qt.CursorShape.SizeHorCursor
        elif edges & {"top", "bottom"}:
            cursor = Qt.CursorShape.SizeVerCursor
        else:
            cursor = Qt.CursorShape.SizeAllCursor
        self.setCursor(cursor)

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
            self._update_cursor(edges)
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
            if rect.width() < self._MIN_WIDTH:
                if "left" in self._resize_edges:
                    rect.setLeft(rect.right() - self._MIN_WIDTH + 1)
                else:
                    rect.setRight(rect.left() + self._MIN_WIDTH - 1)
            if rect.height() < self._MIN_HEIGHT:
                if "top" in self._resize_edges:
                    rect.setTop(rect.bottom() - self._MIN_HEIGHT + 1)
                else:
                    rect.setBottom(rect.top() + self._MIN_HEIGHT - 1)
        self.setGeometry(self._clamped_rect(rect.normalized()))
        event.accept()

    def mouseReleaseEvent(self, event: QMouseEvent) -> None:
        if self._dragging and event.button() == Qt.MouseButton.LeftButton:
            self._dragging = False
            self._release_owner_space()
            self._manual_geometry = True
            self._sync_parent_margins()
            self._float_geometry = QRect(self.geometry())
            self._on_state_changed()
            event.accept()
            return
        super().mouseReleaseEvent(event)

    def mouseDoubleClickEvent(self, event: QMouseEvent) -> None:
        if event.button() == Qt.MouseButton.LeftButton:
            self._on_customize()
            event.accept()
            return
        super().mouseDoubleClickEvent(event)

    def contextMenuEvent(self, event) -> None:
        menu = QMenu(self)
        menu.addAction("Hide colorbar", lambda: self._on_visibility_changed(False))

        attribute_menu = menu.addMenu("Attribute:")
        current_attribute = self._current_attribute()
        names = list(self._attribute_names())
        for name in names:
            action = attribute_menu.addAction(name)
            action.setCheckable(True)
            action.setChecked(name == current_attribute)
            action.triggered.connect(
                lambda _checked=False, value=name: self._on_attribute_changed(value)
            )
        apply_attribute_menu_tooltips(attribute_menu, names)

        values_action = menu.addAction("Show values")
        values_action.setCheckable(True)
        values_action.setChecked(self._show_values)
        values_action.triggered.connect(self.set_show_values)

        placement = menu.addMenu("Placement")
        for label, orientation in (
            ("Vertical", "vertical"),
            ("Horizontal", "horizontal"),
        ):
            action = placement.addAction(label)
            action.setCheckable(True)
            action.setChecked(self._orientation == orientation)
            action.triggered.connect(
                lambda _checked=False, value=orientation: self.set_orientation(value)
            )
        menu.addAction(
            "Dock" if self._manual_geometry else "Undock",
            lambda: self.set_docked(self._manual_geometry),
        )
        menu.addSeparator()
        menu.addAction("Customize", self._on_customize)
        menu.exec(event.globalPos())

    @classmethod
    def _regular_tick_spec(
        cls,
        lo: float,
        hi: float,
        *,
        target: int = 8,
        max_ticks: int = 10,
    ) -> tuple[list[float], float]:
        """Return major ruler ticks and their human-friendly decimal step."""
        if not (np.isfinite(lo) and np.isfinite(hi)) or hi <= lo:
            return [], 0.0
        span = float(hi - lo)
        base = int(np.floor(np.log10(span)))
        candidates = sorted(
            {
                factor * 10.0**exponent
                for exponent in range(base - 5, base + 3)
                for factor in (1.0, 2.0, 2.5, 5.0, 10.0)
            }
        )

        raw_step = span / max(2, int(target) - 1)

        def tick_count(step: float) -> int:
            tolerance = step * 1e-9
            start = np.ceil((lo - tolerance) / step) * step
            stop = np.floor((hi + tolerance) / step) * step
            return max(0, int(np.floor((stop - start) / step + 0.5)) + 1)

        usable = [
            step
            for step in candidates
            if 3 <= tick_count(step) <= max_ticks
        ]
        pool = usable or candidates
        step = min(
            pool,
            key=lambda candidate: (
                abs(np.log(candidate / raw_step)),
                abs(tick_count(candidate) - target),
            ),
        )

        tolerance = step * 1e-9
        start = np.ceil((lo - tolerance) / step) * step
        stop = np.floor((hi + tolerance) / step) * step
        count = max(0, int(np.floor((stop - start) / step + 0.5)) + 1)
        values = start + np.arange(count, dtype=float) * step
        values[np.isclose(values, lo, rtol=0.0, atol=tolerance)] = lo
        values[np.isclose(values, hi, rtol=0.0, atol=tolerance)] = hi
        return [float(value) for value in values], float(step)

    @classmethod
    def _regular_tick_values(
        cls,
        lo: float,
        hi: float,
        *,
        target: int = 8,
        max_ticks: int = 10,
    ) -> list[float]:
        """Return evenly spaced ruler ticks on a human-friendly decimal grid."""
        values, _step = cls._regular_tick_spec(
            lo,
            hi,
            target=target,
            max_ticks=max_ticks,
        )
        return values

    @staticmethod
    def _minor_tick_values(
        lo: float,
        hi: float,
        major_ticks: Sequence[float],
        major_step: float,
    ) -> list[float]:
        """Return four shorter ruler subdivisions between major grid lines."""
        if (
            not (np.isfinite(lo) and np.isfinite(hi) and np.isfinite(major_step))
            or hi <= lo
            or major_step <= 0
        ):
            return []
        minor_step = major_step / 5.0
        tolerance = minor_step * 1e-8
        start = np.ceil((lo - tolerance) / minor_step) * minor_step
        stop = np.floor((hi + tolerance) / minor_step) * minor_step
        count = max(0, int(np.floor((stop - start) / minor_step + 0.5)) + 1)
        if count > 1_000:
            return []
        values = start + np.arange(count, dtype=float) * minor_step
        return [
            float(value)
            for value in values
            if not np.isclose(value, lo, rtol=0.0, atol=tolerance)
            and not np.isclose(value, hi, rtol=0.0, atol=tolerance)
            and not any(
                np.isclose(value, major, rtol=0.0, atol=tolerance)
                for major in major_ticks
            )
        ]

    @staticmethod
    def _scale_exponent(values: Sequence[float]) -> int:
        finite = np.abs(np.asarray(values, dtype=float))
        finite = finite[np.isfinite(finite) & (finite > 0)]
        if finite.size == 0:
            return 0
        largest = float(np.max(finite))
        if largest >= 1_000.0 or largest < 0.01:
            return int(np.floor(np.log10(largest) / 3.0) * 3)
        return 0

    @staticmethod
    def _superscript(exponent: int) -> str:
        table = str.maketrans("-0123456789", "⁻⁰¹²³⁴⁵⁶⁷⁸⁹")
        return str(exponent).translate(table)

    @staticmethod
    def _format_tick(value: float, exponent: int) -> str:
        if not np.isfinite(value):
            return "—"
        scaled = value / 10.0**exponent
        if abs(scaled) < 5e-13:
            scaled = 0.0
        return f"{scaled:.3g}"

    @staticmethod
    def _format_endpoint(value: float, exponent: int, major_step: float) -> str:
        """Round an endpoint to the precision established by the major ruler."""
        if not np.isfinite(value):
            return "—"
        scale = 10.0**exponent
        scaled = value / scale
        scaled_step = abs(major_step / scale)
        if not np.isfinite(scaled_step) or scaled_step <= 0:
            return FloatingColorBar._format_tick(value, exponent)
        decimals = max(0, min(6, int(np.ceil(-np.log10(scaled_step)))))
        if abs(scaled) < 0.5 * 10.0 ** (-decimals):
            scaled = 0.0
        return f"{scaled:.{decimals}f}"

    def _attribute_label(self, exponent: int) -> str:
        label = self._label[3:] if self._label.startswith("C: ") else self._label
        if exponent:
            return f"{label} (×10{self._superscript(exponent)})"
        return label


    # ------------------------------------------------------------------
    # Painting
    # ------------------------------------------------------------------

    @staticmethod
    def _with_alpha(color: QColor, alpha: int) -> QColor:
        faded = QColor(color)
        faded.setAlpha(int(alpha))
        return faded

    def _panel_color(self) -> QColor:
        """Backing colour: the floating panel, or the plot's own background."""
        if not self.docked:
            return QColor(250, 250, 250, 225)
        if self._background_color is not None:
            try:
                color = self._background_color()
            except Exception:
                color = None
            if color is not None:
                return QColor(color)
        return QColor(255, 255, 255)

    def _ink_colors(self) -> tuple[QColor, QColor]:
        """(ink, halo) - text/tick colour and its contrasting backing."""
        background = self._panel_color()
        luminance = (
            0.2126 * background.red()
            + 0.7152 * background.green()
            + 0.0722 * background.blue()
        )
        if luminance >= 145:
            return QColor(25, 25, 25), QColor(255, 255, 255)
        return QColor(235, 235, 235), QColor(15, 15, 15)

    def _docked_plot_span(self) -> tuple[int, int] | None:
        """The plot data area's extent along the bar axis, in local pixels.

        A docked bar reproduces the in-plot colorbar it replaced: the gradient
        starts and ends exactly where the plot's own axes do.
        """
        if not self.docked or self._plot_area is None:
            return None
        try:
            area = self._plot_area()
        except Exception:
            return None
        if area is None or not area.isValid():
            return None
        area = area.translated(-self.x(), -self.y())
        if self._orientation == "vertical":
            lo, hi, limit = area.top(), area.bottom(), self.height()
        else:
            lo, hi, limit = area.left(), area.right(), self.width()
        lo = max(0, min(int(lo), limit))
        hi = max(0, min(int(hi), limit))
        if hi - lo < 10:
            return None
        return lo, hi

    def _draw_tick(
        self,
        painter: QPainter,
        bar: QRect,
        position: int,
        length: int,
        ink: QColor,
        halo: QColor,
        docked: bool,
    ) -> None:
        """One ruler tick: outside the gradient when docked, inside when not.

        Docked reproduces the plot axis it replaced (an unbroken gradient with
        the ticks out in the label margin); a floating panel has no margin to
        spare, so its ticks bite into the gradient over a contrast halo.
        """
        vertical = self._orientation == "vertical"
        if docked:
            near, far = 1, length
        else:
            near, far = -1, -length
            painter.setPen(QPen(self._with_alpha(halo, 200), 2.0))
            if vertical:
                painter.drawLine(
                    bar.right() + far, position, bar.right() + near, position
                )
            else:
                painter.drawLine(
                    position, bar.bottom() + far, position, bar.bottom() + near
                )
        painter.setPen(QPen(ink, 1.0))
        if vertical:
            painter.drawLine(
                bar.right() + far, position, bar.right() + near, position
            )
        else:
            painter.drawLine(
                position, bar.bottom() + far, position, bar.bottom() + near
            )

    def _value_label_rect(
        self, left: int, centre: int, width: int, height: int
    ) -> QRect:
        """A vertical value label, kept inside the widget at both endpoints."""
        top = max(0, min(centre - height // 2, max(0, self.height() - height)))
        return QRect(left, top, width, height)

    def paintEvent(self, _event) -> None:
        painter = QPainter(self)
        painter.setRenderHint(QPainter.RenderHint.Antialiasing)
        docked = self.docked
        panel_color = self._panel_color()
        ink, halo = self._ink_colors()
        if docked:
            # Flush with the plot: the reserved gutter is painted in the plot's
            # own background and carries no panel outline, so the bar reads as
            # part of the plot exactly like the in-plot colorbar it replaced.
            painter.fillRect(self.rect(), panel_color)
        else:
            panel = self.rect().adjusted(1, 1, -2, -2)
            painter.setPen(QPen(QColor(70, 70, 70, 190), 1.0))
            painter.setBrush(panel_color)
            painter.drawRoundedRect(panel, 5.0, 5.0)

        metrics = painter.fontMetrics()
        draw_ruler = (
            self._show_values
            and np.isfinite(self._lo)
            and np.isfinite(self._hi)
            and self._hi > self._lo
        )
        if draw_ruler:
            ticks, major_step = self._regular_tick_spec(self._lo, self._hi)
            minor_ticks = self._minor_tick_values(
                self._lo,
                self._hi,
                ticks,
                major_step,
            )
        else:
            ticks, minor_ticks, major_step = [], [], 0.0
        exponent = self._scale_exponent([self._lo, self._hi, *ticks])
        attribute_label = self._attribute_label(exponent)

        if self._orientation == "vertical":
            label_height = metrics.height() + 8
            endpoint_padding = metrics.height() // 2
            if docked:
                # Rotated attribute name in an outer column, where the plot's
                # own right-axis label used to sit.
                label_width = metrics.height() + 4
                bar_width = max(14, min(20, self.width() // 4))
                span = self._docked_plot_span()
                if span is None:
                    top = 4 + endpoint_padding
                    bottom = self.height() - 4 - endpoint_padding
                else:
                    top, bottom = span
                bar = QRect(1, top, bar_width, max(10, bottom - top))
            else:
                label_width = 0
                bar_width = max(16, min(24, self.width() // 3))
                bar = QRect(
                    7,
                    7 + endpoint_padding,
                    bar_width,
                    max(
                        10,
                        self.height()
                        - label_height
                        - 12
                        - 2 * endpoint_padding,
                    ),
                )
            gradient = QLinearGradient(
                float(bar.left()),
                float(bar.bottom()),
                float(bar.left()),
                float(bar.top()),
            )
        else:
            attribute_width = min(
                max(38, metrics.horizontalAdvance(attribute_label) + 12),
                max(38, self.width() // 3),
            )
            bar_height = max(16, min(22, self.height() // 3))
            if docked:
                span = self._docked_plot_span()
                left = attribute_width + 6
                right = self.width() - 8
                if span is not None:
                    left = max(left, span[0])
                    right = max(left + 10, span[1])
                bar = QRect(left, 6, max(10, right - left), bar_height)
                attribute_rect = QRect(2, 2, max(0, left - 6), self.height() - 4)
            else:
                bar = QRect(
                    attribute_width + 6,
                    7,
                    max(10, self.width() - attribute_width - 14),
                    bar_height,
                )
                attribute_rect = QRect(4, 2, attribute_width, self.height() - 4)
            gradient = QLinearGradient(
                float(bar.left()),
                float(bar.top()),
                float(bar.right()),
                float(bar.top()),
            )
        count = self._lut.shape[0]
        for index, rgba in enumerate(self._lut):
            gradient.setColorAt(
                index / max(1, count - 1),
                QColor(*(int(channel) for channel in rgba)),
            )
        painter.setPen(QPen(self._with_alpha(ink, 220), 1.0))
        painter.setBrush(gradient)
        painter.drawRect(bar)

        if self._orientation == "vertical":
            if draw_ruler:
                label_left = bar.right() + (13 if docked else 6)
                label_rect_width = max(
                    0, self.width() - label_left - 4 - label_width
                )
                for value in minor_ticks:
                    fraction = (value - self._lo) / (self._hi - self._lo)
                    y = int(round(bar.bottom() - fraction * bar.height()))
                    self._draw_tick(painter, bar, y, 4, ink, halo, docked)
                for value in ticks:
                    fraction = (value - self._lo) / (self._hi - self._lo)
                    y = int(round(bar.bottom() - fraction * bar.height()))
                    self._draw_tick(painter, bar, y, 7, ink, halo, docked)

                endpoint_values = [self._lo, self._hi]
                label_positions: list[int] = []
                for value in endpoint_values:
                    fraction = (value - self._lo) / (self._hi - self._lo)
                    y = int(round(bar.bottom() - fraction * bar.height()))
                    self._draw_tick(painter, bar, y, 9, ink, halo, docked)
                    painter.setPen(ink)
                    painter.drawText(
                        self._value_label_rect(
                            label_left, y, label_rect_width, metrics.height()
                        ),
                        Qt.AlignmentFlag.AlignLeft
                        | Qt.AlignmentFlag.AlignVCenter,
                        self._format_endpoint(value, exponent, major_step),
                    )
                    label_positions.append(y)

                for value in ticks:
                    if any(
                        np.isclose(
                            value,
                            endpoint,
                            rtol=0.0,
                            atol=max(major_step, 1.0) * 1e-9,
                        )
                        for endpoint in endpoint_values
                    ):
                        continue
                    fraction = (value - self._lo) / (self._hi - self._lo)
                    y = int(round(bar.bottom() - fraction * bar.height()))
                    if any(
                        abs(y - existing) < metrics.height() + 2
                        for existing in label_positions
                    ):
                        continue
                    painter.drawText(
                        self._value_label_rect(
                            label_left, y, label_rect_width, metrics.height()
                        ),
                        Qt.AlignmentFlag.AlignLeft
                        | Qt.AlignmentFlag.AlignVCenter,
                        self._format_tick(value, exponent),
                    )
                    label_positions.append(y)
            painter.setPen(ink)
            if docked:
                text_span = max(10, bar.height())
                painter.save()
                painter.translate(
                    self.width() - 2,
                    (bar.top() + bar.bottom()) / 2.0,
                )
                painter.rotate(-90.0)
                painter.drawText(
                    QRect(-text_span // 2, -label_width, text_span, label_width),
                    Qt.AlignmentFlag.AlignCenter,
                    metrics.elidedText(
                        attribute_label,
                        Qt.TextElideMode.ElideRight,
                        text_span,
                    ),
                )
                painter.restore()
            else:
                painter.drawText(
                    QRect(
                        4,
                        self.height() - label_height,
                        self.width() - 8,
                        label_height - 2,
                    ),
                    Qt.AlignmentFlag.AlignCenter,
                    metrics.elidedText(
                        attribute_label,
                        Qt.TextElideMode.ElideRight,
                        self.width() - 10,
                    ),
                )
        else:
            painter.setPen(ink)
            painter.drawText(
                attribute_rect,
                Qt.AlignmentFlag.AlignCenter,
                metrics.elidedText(
                    attribute_label,
                    Qt.TextElideMode.ElideRight,
                    max(0, attribute_rect.width() - 6),
                ),
            )
            if draw_ruler:
                text_y = bar.bottom() + (12 if docked else 4)
                for value in minor_ticks:
                    fraction = (value - self._lo) / (self._hi - self._lo)
                    x = int(round(bar.left() + fraction * bar.width()))
                    self._draw_tick(painter, bar, x, 4, ink, halo, docked)
                for value in ticks:
                    fraction = (value - self._lo) / (self._hi - self._lo)
                    x = int(round(bar.left() + fraction * bar.width()))
                    self._draw_tick(painter, bar, x, 7, ink, halo, docked)

                label_rects: list[QRect] = []
                endpoint_values = [self._lo, self._hi]
                for value in endpoint_values:
                    fraction = (value - self._lo) / (self._hi - self._lo)
                    x = int(round(bar.left() + fraction * bar.width()))
                    self._draw_tick(painter, bar, x, 9, ink, halo, docked)
                    painter.setPen(ink)
                    text = self._format_endpoint(value, exponent, major_step)
                    text_width = metrics.horizontalAdvance(text) + 6
                    text_x = max(
                        bar.left(),
                        min(
                            x - text_width // 2,
                            bar.right() - text_width + 1,
                        ),
                    )
                    text_rect = QRect(
                        text_x,
                        text_y,
                        text_width,
                        metrics.height(),
                    )
                    painter.drawText(
                        text_rect,
                        Qt.AlignmentFlag.AlignHCenter
                        | Qt.AlignmentFlag.AlignTop,
                        text,
                    )
                    label_rects.append(text_rect)

                for value in ticks:
                    if any(
                        np.isclose(
                            value,
                            endpoint,
                            rtol=0.0,
                            atol=max(major_step, 1.0) * 1e-9,
                        )
                        for endpoint in endpoint_values
                    ):
                        continue
                    fraction = (value - self._lo) / (self._hi - self._lo)
                    x = int(round(bar.left() + fraction * bar.width()))
                    text = self._format_tick(value, exponent)
                    text_width = metrics.horizontalAdvance(text) + 6
                    text_x = max(
                        bar.left(),
                        min(
                            x - text_width // 2,
                            bar.right() - text_width + 1,
                        ),
                    )
                    text_rect = QRect(
                        text_x,
                        text_y,
                        text_width,
                        metrics.height(),
                    )
                    if any(
                        text_rect.adjusted(-3, 0, 3, 0).intersects(existing)
                        for existing in label_rects
                    ):
                        continue
                    painter.drawText(
                        text_rect,
                        Qt.AlignmentFlag.AlignHCenter
                        | Qt.AlignmentFlag.AlignTop,
                        text,
                    )
                    label_rects.append(text_rect)
