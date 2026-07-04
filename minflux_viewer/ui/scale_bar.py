"""
Scale-bar overlay for coordinate-based 2-D views (render / scatter).

A :class:`ScaleBarItem` is a **draggable** pyqtgraph item (not burned into the
view): a physical-size bar (``width`` × ``height`` nm) drawn in data coordinates
plus a fixed-pixel-font value label and an optional background box. It can be
dragged with the left button and right-clicked for a *Property* / *Delete* menu
(it emits :attr:`sigEditRequested` / :attr:`sigDeleteRequested`). The initial
placement (a viewport corner, or a rectangle-ROI centre) is computed by the pure
:func:`scale_bar_anchor` helper (unit-tested); after that the item is free.
"""

from __future__ import annotations

import pyqtgraph as pg
from PyQt6.QtCore import QPointF, QRectF, Qt, pyqtSignal
from PyQt6.QtGui import QFont, QFontMetricsF

_MARGIN_PX = 18.0
_PAD_PX = 6.0
_GAP_PX = 4.0


def format_nm(value: float) -> str:
    """A scale-bar label for a length in nm with an auto unit (nm / µm / mm)."""
    v = abs(float(value))
    if v >= 1.0e6:
        return f"{value / 1.0e6:g} mm"
    if v >= 1.0e3:
        return f"{value / 1.0e3:g} µm"
    return f"{value:g} nm"


def scale_bar_anchor(
    view_range, view_pixel_size, *, bar_w, bar_h, location, x_inv, y_inv,
    roi_center=None, margin_px: float = _MARGIN_PX,
) -> tuple[float, float]:
    """Top-left (data) corner of the ``bar_w`` × ``bar_h`` nm bar.

    For a corner *location* the bar is inset ``margin_px`` from the matching
    **screen** edges (axis inversion respected); for a *roi_center* it is centred
    there. ``view_range`` is ``((xmin, xmax), (ymin, ymax))`` and
    ``view_pixel_size`` is ``(px, py)`` data-units-per-pixel.
    """
    (xmin, xmax), (ymin, ymax) = view_range
    if roi_center is not None:
        cx, cy = roi_center
        return cx - bar_w / 2.0, cy - bar_h / 2.0
    px, py = view_pixel_size
    mx, my = margin_px * abs(px), margin_px * abs(py)
    right = "Right" in location
    upper = "Upper" in location
    bx = (xmin + mx) if (right == bool(x_inv)) else (xmax - mx - bar_w)
    by = (ymin + my) if (upper == bool(y_inv)) else (ymax - my - bar_h)
    return bx, by


def initial_scale_bar_center(view_box, params, *, roi_center=None) -> tuple[float, float]:
    """Initial data-centre for a scale bar from the dialog *params* (Location)."""
    bar_w = params["width_nm"] if params["horizontal"] else params["height_nm"]
    bar_h = params["height_nm"] if params["horizontal"] else params["width_nm"]
    if roi_center is not None:
        return float(roi_center[0]), float(roi_center[1])
    try:
        vr = view_box.viewRange()
        px, py = view_box.viewPixelSize()
        x_inv, y_inv = view_box.xInverted(), view_box.yInverted()
    except Exception:
        return 0.0, 0.0
    bx, by = scale_bar_anchor(vr, (px, py), bar_w=bar_w, bar_h=bar_h,
                              location=params.get("location", "Lower Right"),
                              x_inv=x_inv, y_inv=y_inv)
    return bx + bar_w / 2.0, by + bar_h / 2.0


class ScaleBarItem(pg.GraphicsObject):
    """A draggable scale bar (bar + label + optional background) on a ViewBox."""

    sigEditRequested = pyqtSignal(object)
    sigDeleteRequested = pyqtSignal(object)

    def __init__(self, view_box, *, width_nm, height_nm, font_size, color,
                 bg_color=None, horizontal=True, center=(0.0, 0.0)) -> None:
        super().__init__()
        self._vb = view_box
        self._color = color
        self._bg_color = bg_color
        self._horizontal = bool(horizontal)
        self._width_nm = float(width_nm)
        self._height_nm = float(height_nm)
        self._font = QFont()
        self._font.setPointSize(int(font_size))
        self._text = format_nm(self._width_nm)

        self.setZValue(1000)
        self.setAcceptedMouseButtons(Qt.MouseButton.LeftButton | Qt.MouseButton.RightButton)
        self.setAcceptHoverEvents(True)
        self.setCursor(Qt.CursorShape.SizeAllCursor)
        self.setFlag(self.GraphicsItemFlag.ItemSendsGeometryChanges, True)

        self._label = pg.TextItem(self._text, color=color, anchor=(0.5, 1.0))
        self._apply_font()
        self._label.setParentItem(self)

        view_box.addItem(self, ignoreBounds=True)
        self.setPos(QPointF(float(center[0]), float(center[1])))
        try:
            self._vb.sigRangeChanged.connect(self._on_range_changed)
        except Exception:
            pass
        self._relayout()

    # -- parameters -----------------------------------------------------
    def params(self) -> dict:
        return {
            "width_nm": self._width_nm,
            "height_nm": self._height_nm,
            "font_size": self._font.pointSize(),
            "color": self._color,
            "bg_color": self._bg_color,
            "horizontal": self._horizontal,
        }

    def set_params(self, params: dict) -> None:
        self._width_nm = float(params["width_nm"])
        self._height_nm = float(params["height_nm"])
        self._font.setPointSize(int(params["font_size"]))
        self._color = params["color"]
        self._bg_color = params["bg_color"]
        self._horizontal = bool(params["horizontal"])
        self._text = format_nm(self._width_nm)
        self._label.setText(self._text)
        self._label.setColor(self._color)
        self._apply_font()
        self.prepareGeometryChange()
        self._relayout()
        self.update()

    def _apply_font(self) -> None:
        try:
            self._label.setFont(self._font)
        except Exception:
            try:
                self._label.textItem.setFont(self._font)
            except Exception:
                pass

    # -- geometry -------------------------------------------------------
    def _bar_size(self) -> tuple[float, float]:
        if self._horizontal:
            return self._width_nm, self._height_nm
        return self._height_nm, self._width_nm

    def _pixel_size(self) -> tuple[float, float]:
        try:
            px, py = self._vb.viewPixelSize()
            if px and py:
                return abs(px), abs(py)
        except Exception:
            pass
        return 1.0, 1.0

    def _axis_inverted(self) -> tuple[bool, bool]:
        """(y_inverted, x_inverted) of the host ViewBox (both default False)."""
        try:
            return bool(self._vb.yInverted()), bool(self._vb.xInverted())
        except Exception:
            return False, False

    def _content_rect(self) -> QRectF:
        """Local bounding rect enclosing bar + label (+ padding)."""
        bw, bh = self._bar_size()
        px, py = self._pixel_size()
        y_inv, x_inv = self._axis_inverted()
        fm = QFontMetricsF(self._font)
        lw = fm.horizontalAdvance(self._text) * px
        lh = fm.height() * py
        pad_x, pad_y = _PAD_PX * px, _PAD_PX * py
        gap_x, gap_y = _GAP_PX * px, _GAP_PX * py
        if self._horizontal:
            up = -1.0 if y_inv else 1.0            # data direction that is up on screen
            x0 = min(-bw / 2.0, -lw / 2.0) - pad_x
            x1 = max(bw / 2.0, lw / 2.0) + pad_x
            near = -up * (bh / 2.0)                 # bar edge opposite the label
            far = up * (bh / 2.0 + gap_y + lh)      # far edge of the label
            y0 = min(near, far) - pad_y
            y1 = max(near, far) + pad_y
        else:
            right = -1.0 if x_inv else 1.0          # data direction that is right on screen
            near = -right * (bw / 2.0)
            far = right * (bw / 2.0 + gap_x + lw)
            x0 = min(near, far) - pad_x
            x1 = max(near, far) + pad_x
            y0 = min(-bh / 2.0, -lh / 2.0) - pad_y
            y1 = max(bh / 2.0, lh / 2.0) + pad_y
        return QRectF(x0, y0, x1 - x0, y1 - y0)

    def _relayout(self) -> None:
        # Keep the label just outside the bar ON SCREEN (anchor = the text edge
        # nearest the bar), so it never hides behind a thick bar when zoomed in.
        # A TextItem always draws upright, so with a bottom anchor the text sits
        # above its position on screen — we just place that position at the bar's
        # screen-top edge, flipping the data-offset sign when the axis is inverted
        # (in a y-inverted view "up on screen" is -y in data).
        bw, bh = self._bar_size()
        px, py = self._pixel_size()
        y_inv, x_inv = self._axis_inverted()
        if self._horizontal:
            up = -1.0 if y_inv else 1.0
            self._label.setAnchor((0.5, 1.0))
            self._label.setPos(0.0, up * (bh / 2.0 + _GAP_PX * py))
        else:
            right = -1.0 if x_inv else 1.0
            self._label.setAnchor((0.0, 0.5))
            self._label.setPos(right * (bw / 2.0 + _GAP_PX * px), 0.0)

    def _on_range_changed(self, *_args) -> None:
        self.prepareGeometryChange()
        self._relayout()
        self.update()

    def boundingRect(self) -> QRectF:
        return self._content_rect()

    def paint(self, p, *_args) -> None:
        if self._bg_color is not None:
            p.setPen(pg.mkPen(None))
            p.setBrush(pg.mkBrush(self._bg_color))
            p.drawRect(self._content_rect())
        bw, bh = self._bar_size()
        p.setPen(pg.mkPen(None))
        p.setBrush(pg.mkBrush(self._color))
        p.drawRect(QRectF(-bw / 2.0, -bh / 2.0, bw, bh))

    # -- interaction ----------------------------------------------------
    def mouseDragEvent(self, ev) -> None:
        if ev.button() != Qt.MouseButton.LeftButton:
            ev.ignore()
            return
        ev.accept()
        try:
            if ev.isStart():
                self._drag_start = self.pos()
                self._drag_anchor = self._vb.mapSceneToView(ev.buttonDownScenePos())
            cur = self._vb.mapSceneToView(ev.scenePos())
            self.setPos(self._drag_start + (cur - self._drag_anchor))
        except Exception:
            pass

    def mouseClickEvent(self, ev) -> None:
        if ev.button() != Qt.MouseButton.RightButton:
            ev.ignore()
            return
        ev.accept()
        from PyQt6.QtWidgets import QMenu

        menu = QMenu()
        prop = menu.addAction("Property")
        dele = menu.addAction("Delete")
        try:
            pos = ev.screenPos().toPoint()
        except Exception:
            pos = ev.screenPos()
        chosen = menu.exec(pos)
        if chosen is prop:
            self.sigEditRequested.emit(self)
        elif chosen is dele:
            self.sigDeleteRequested.emit(self)

    def remove(self) -> None:
        try:
            self._vb.sigRangeChanged.disconnect(self._on_range_changed)
        except Exception:
            pass
        try:
            self._vb.removeItem(self)
        except Exception:
            pass
