"""CPU rendering primitives for large 2-D attribute plots.

The normal :class:`pyqtgraph.ScatterPlotItem` is feature rich, but its Python
per-point style preparation becomes the bottleneck for large MINFLUX raw-row
selections.  This module keeps the ordinary pyqtgraph ``ViewBox`` and replaces
only the marker item:

* :class:`BulkScatterItem` sends contiguous point arrays to ``QPainter``;
* :func:`aggregate_screen_points` reduces dense, overplotted data to one
  display-sized count/mean grid without dropping the input before aggregation;
* :func:`spatial_representative_indices` provides deterministic spatial LOD for
  the legacy/GPU memory guard and for very long connecting curves.

The array helpers are Qt-free and deliberately unit-testable.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pyqtgraph as pg
from PyQt6.QtCore import QRectF, Qt
from PyQt6.QtGui import QColor, QPainter, QPen, QPixmap


@dataclass(frozen=True)
class ScreenAggregation:
    """One screen-space reduction of a complete visible point selection."""

    counts: np.ndarray
    value_sum: np.ndarray | None
    value_count: np.ndarray | None
    input_count: int
    drawable_count: int
    visible_count: int

    @property
    def occupied_count(self) -> int:
        return int(np.count_nonzero(self.counts))

    def mean_values(self) -> np.ndarray | None:
        if self.value_sum is None or self.value_count is None:
            return None
        out = np.full(self.counts.shape, np.nan, dtype=np.float64)
        np.divide(
            self.value_sum,
            self.value_count,
            out=out,
            where=self.value_count > 0,
        )
        return out


def _numeric_flat(values) -> np.ndarray:
    array = np.asarray(values).ravel()
    if not np.issubdtype(array.dtype, np.number):
        array = array.astype(float)
    return array


def joint_finite_mask(x, y, values=None) -> np.ndarray:
    """Rows that can paint a marker (and, when supplied, have a C value)."""

    x_values = _numeric_flat(x)
    y_values = _numeric_flat(y)
    n = min(x_values.size, y_values.size)
    mask = np.isfinite(x_values[:n]) & np.isfinite(y_values[:n])
    if values is not None:
        c_values = _numeric_flat(values)
        n = min(n, c_values.size)
        mask = mask[:n] & np.isfinite(c_values[:n])
    return mask


def joint_extent(x, y) -> tuple[float, float, float, float] | None:
    """Finite X/Y-pair extent; coordinates from non-drawable rows are ignored."""

    x_values = _numeric_flat(x)
    y_values = _numeric_flat(y)
    mask = joint_finite_mask(x_values, y_values)
    if not mask.any():
        return None
    x_values = x_values[: mask.size][mask]
    y_values = y_values[: mask.size][mask]
    return (
        float(np.min(x_values)),
        float(np.max(x_values)),
        float(np.min(y_values)),
        float(np.max(y_values)),
    )


def _safe_bounds(bounds) -> tuple[float, float, float, float]:
    x0, x1, y0, y1 = (float(value) for value in bounds)
    if not np.all(np.isfinite((x0, x1, y0, y1))):
        raise ValueError("screen aggregation bounds must be finite")
    if x1 <= x0:
        pad = max(abs(x0) * 1e-9, 0.5)
        x0, x1 = x0 - pad, x1 + pad
    if y1 <= y0:
        pad = max(abs(y0) * 1e-9, 0.5)
        y0, y1 = y0 - pad, y1 + pad
    return x0, x1, y0, y1


def aggregate_screen_points(
    x,
    y,
    *,
    bounds,
    width: int,
    height: int,
    values=None,
    chunk_size: int = 1_000_000,
) -> ScreenAggregation:
    """Aggregate every visible drawable row into a display-sized grid.

    Processing is chunked so a 20-million-row selection never needs a second
    20-million-element integer bin array.  When *values* is supplied, the grid
    also carries a finite-value sum/count pair; callers can display the named
    reduction (currently mean C) without conflating it with density.
    """

    x_values = _numeric_flat(x)
    y_values = _numeric_flat(y)
    c_values = None if values is None else _numeric_flat(values)
    n = min(x_values.size, y_values.size)
    if c_values is not None:
        n = min(n, c_values.size)
    width = max(1, int(width))
    height = max(1, int(height))
    chunk_size = max(1, int(chunk_size))
    x0, x1, y0, y1 = _safe_bounds(bounds)
    bins = width * height
    counts = np.zeros(bins, dtype=np.uint64)
    value_sum = np.zeros(bins, dtype=np.float64) if c_values is not None else None
    value_count = np.zeros(bins, dtype=np.uint64) if c_values is not None else None
    drawable_count = 0
    visible_count = 0

    x_scale = width / (x1 - x0)
    y_scale = height / (y1 - y0)
    for start in range(0, n, chunk_size):
        stop = min(n, start + chunk_size)
        x_chunk = x_values[start:stop]
        y_chunk = y_values[start:stop]
        finite_xy = np.isfinite(x_chunk) & np.isfinite(y_chunk)
        drawable_count += int(finite_xy.sum())
        visible = finite_xy & (
            (x_chunk >= x0) & (x_chunk <= x1)
            & (y_chunk >= y0) & (y_chunk <= y1)
        )
        if c_values is not None:
            c_chunk = c_values[start:stop]
            visible &= np.isfinite(c_chunk)
        if not visible.any():
            continue
        visible_count += int(visible.sum())
        ix = np.floor((x_chunk[visible] - x0) * x_scale).astype(np.int64)
        iy = np.floor((y_chunk[visible] - y0) * y_scale).astype(np.int64)
        np.clip(ix, 0, width - 1, out=ix)
        np.clip(iy, 0, height - 1, out=iy)
        flat = iy * width + ix
        counts += np.bincount(flat, minlength=bins).astype(np.uint64, copy=False)
        if c_values is not None:
            weights = c_chunk[visible]
            value_sum += np.bincount(
                flat, weights=weights, minlength=bins
            )
            value_count += np.bincount(flat, minlength=bins).astype(
                np.uint64, copy=False
            )

    shape = (height, width)
    return ScreenAggregation(
        counts=counts.reshape(shape),
        value_sum=None if value_sum is None else value_sum.reshape(shape),
        value_count=None if value_count is None else value_count.reshape(shape),
        input_count=n,
        drawable_count=drawable_count,
        visible_count=visible_count,
    )


def spatial_representative_indices(
    x,
    y,
    budget: int,
    *,
    candidate_indices=None,
) -> np.ndarray:
    """Return at most *budget* spatially representative rows.

    One deterministic first representative is retained from every occupied
    2-D cell, then the original row order is restored.  Unlike ``rows[::k]``,
    a small isolated feature gets its own cell and cannot disappear merely
    because its storage index falls between two stride positions.
    """

    x_values = _numeric_flat(x)
    y_values = _numeric_flat(y)
    n = min(x_values.size, y_values.size)
    if candidate_indices is None:
        candidates = np.flatnonzero(joint_finite_mask(x_values[:n], y_values[:n]))
    else:
        candidates = np.asarray(candidate_indices, dtype=np.int64).ravel()
        candidates = candidates[(candidates >= 0) & (candidates < n)]
        if candidates.size:
            finite = np.isfinite(x_values[candidates]) & np.isfinite(y_values[candidates])
            candidates = candidates[finite]
    budget = max(0, int(budget))
    if budget == 0 or candidates.size == 0:
        return np.empty(0, dtype=np.int64)
    if candidates.size <= budget:
        return candidates

    xv = x_values[candidates]
    yv = y_values[candidates]
    x0, x1 = float(np.min(xv)), float(np.max(xv))
    y0, y1 = float(np.min(yv)), float(np.max(yv))
    x_span = x1 - x0
    y_span = y1 - y0
    if x_span <= 0 and y_span <= 0:
        return candidates[:1]
    if y_span <= 0:
        nx, ny = budget, 1
    elif x_span <= 0:
        nx, ny = 1, budget
    else:
        aspect = x_span / y_span
        nx = max(1, min(budget, int(np.sqrt(budget * aspect))))
        ny = max(1, budget // nx)
        while nx * ny > budget:
            ny -= 1

    if nx == 1:
        ix = np.zeros(candidates.size, dtype=np.int64)
    else:
        ix = np.floor((xv - x0) * (nx / max(x_span, np.finfo(float).eps))).astype(
            np.int64
        )
        np.clip(ix, 0, nx - 1, out=ix)
    if ny == 1:
        iy = np.zeros(candidates.size, dtype=np.int64)
    else:
        iy = np.floor((yv - y0) * (ny / max(y_span, np.finfo(float).eps))).astype(
            np.int64
        )
        np.clip(iy, 0, ny - 1, out=iy)
    cell = iy * nx + ix
    _cells, first = np.unique(cell, return_index=True)
    chosen = candidates[first]
    if chosen.size < budget:
        # A diagonal/curve may occupy only sqrt(budget) cells in a square grid.
        # Spatial coverage has already been guaranteed above; use the remaining
        # capacity to retain deterministic detail along the candidate sequence.
        # This is not the old fixed-k stride (and it runs after non-finite and
        # visible-row selection), so a validity toggle cannot change which
        # spatial cells are represented.
        remaining = budget - chosen.size
        positions = np.linspace(
            0, candidates.size - 1, num=remaining + 2, dtype=np.int64
        )[1:-1]
        chosen = np.unique(np.concatenate((chosen, candidates[positions])))
    return np.sort(chosen[:budget])


class BulkScatterItem(pg.GraphicsObject):
    """Pixel-sized scatter markers painted through bulk Qt primitive calls."""

    def __init__(
        self,
        x,
        y,
        *,
        color=(30, 90, 180, 255),
        size: float = 3.0,
        symbol: str = "o",
        color_bins=None,
        lut=None,
    ) -> None:
        super().__init__()
        x_values = np.asarray(x, dtype=float).ravel()
        y_values = np.asarray(y, dtype=float).ravel()
        n = min(x_values.size, y_values.size)
        finite = np.isfinite(x_values[:n]) & np.isfinite(y_values[:n])
        bins = None
        if color_bins is not None:
            raw_bins = np.asarray(color_bins).ravel()
            n = min(n, raw_bins.size)
            finite = finite[:n]
            bins = raw_bins[:n]
        x_values = x_values[:n][finite]
        y_values = y_values[:n][finite]
        self._size = max(1.0, float(size))
        self._symbol = str(symbol or "o")
        self._point_array = pg.Qt.internals.PrimitiveArray(pg.QtCore.QPointF, 2)
        self._fragment_array = pg.Qt.internals.PrimitiveArray(
            pg.QtGui.QPainter.PixmapFragment, 10
        )
        self._groups: list[tuple[tuple[int, int, int, int], np.ndarray, np.ndarray]] = []
        if bins is None:
            self._groups.append((self._rgba(color), x_values, y_values))
        else:
            bins = np.asarray(bins[finite], dtype=np.int64)
            palette = np.asarray(lut, dtype=np.uint8)
            if palette.ndim != 2 or palette.shape[1] != 4:
                raise ValueError("BulkScatterItem LUT must be an N x 4 RGBA array")
            valid = (bins >= 0) & (bins < palette.shape[0])
            x_values, y_values, bins = x_values[valid], y_values[valid], bins[valid]
            if bins.size:
                order = np.argsort(bins, kind="stable")
                x_values, y_values, bins = x_values[order], y_values[order], bins[order]
                starts = np.flatnonzero(np.r_[True, bins[1:] != bins[:-1]])
                stops = np.r_[starts[1:], bins.size]
                for start, stop in zip(starts, stops):
                    rgba = self._rgba(palette[bins[start]])
                    if rgba[3] > 0:
                        self._groups.append((rgba, x_values[start:stop], y_values[start:stop]))
        all_x = [group[1] for group in self._groups if group[1].size]
        all_y = [group[2] for group in self._groups if group[2].size]
        if all_x:
            x_min = min(float(np.min(value)) for value in all_x)
            x_max = max(float(np.max(value)) for value in all_x)
            y_min = min(float(np.min(value)) for value in all_y)
            y_max = max(float(np.max(value)) for value in all_y)
            if x_max <= x_min:
                x_min, x_max = x_min - 0.5, x_max + 0.5
            if y_max <= y_min:
                y_min, y_max = y_min - 0.5, y_max + 0.5
            self._bounds = QRectF(x_min, y_min, x_max - x_min, y_max - y_min)
        else:
            self._bounds = QRectF()
        self._pixmaps: dict[tuple[int, int, int, int], QPixmap] = {}

    @staticmethod
    def _rgba(color) -> tuple[int, int, int, int]:
        values = tuple(max(0, min(255, int(value))) for value in color)
        return values if len(values) == 4 else (*values[:3], 255)

    @property
    def point_count(self) -> int:
        return sum(group[1].size for group in self._groups)

    def boundingRect(self) -> QRectF:
        return self._bounds

    def dataBounds(self, axis: int, frac: float = 1.0, orthoRange=None):
        if self._bounds.isNull():
            return None, None
        if axis == 0:
            return self._bounds.left(), self._bounds.right()
        return self._bounds.top(), self._bounds.bottom()

    def _visible(self, x_values, y_values):
        rect = self.viewRect()
        if rect is None or rect.isNull():
            return x_values, y_values
        mask = (
            (x_values >= rect.left()) & (x_values <= rect.right())
            & (y_values >= rect.top()) & (y_values <= rect.bottom())
        )
        return x_values[mask], y_values[mask]

    def _device_points(self, painter: QPainter, x_values, y_values) -> np.ndarray:
        points = np.vstack((x_values, y_values))
        mapped = pg.functions.transformCoordinates(painter.transform(), points)
        return np.clip(mapped.T, -(2**30), 2**30)

    def _draw_points(self, painter: QPainter, points: np.ndarray, rgba) -> None:
        if points.size == 0:
            return
        self._point_array.resize(points.shape[0])
        self._point_array.ndarray()[:] = points
        pen = QPen(QColor(*rgba))
        pen.setWidthF(self._size)
        pen.setCapStyle(
            Qt.PenCapStyle.SquareCap
            if self._symbol == "s" else Qt.PenCapStyle.RoundCap
        )
        painter.setPen(pen)
        painter.drawPoints(*self._point_array.drawargs())

    def _symbol_pixmap(self, rgba) -> QPixmap:
        cached = self._pixmaps.get(rgba)
        if cached is not None:
            return cached
        from pyqtgraph.graphicsItems.ScatterPlotItem import drawSymbol

        side = max(3, int(np.ceil(self._size)) + 4)
        pixmap = QPixmap(side, side)
        pixmap.fill(Qt.GlobalColor.transparent)
        painter = QPainter(pixmap)
        painter.setRenderHint(QPainter.RenderHint.Antialiasing, True)
        painter.translate(side / 2.0, side / 2.0)
        drawSymbol(
            painter,
            self._symbol,
            self._size,
            pg.mkPen(None),
            pg.mkBrush(*rgba),
        )
        painter.end()
        self._pixmaps[rgba] = pixmap
        return pixmap

    def _draw_fragments(self, painter: QPainter, points: np.ndarray, rgba) -> None:
        if points.size == 0:
            return
        pixmap = self._symbol_pixmap(rgba)
        self._fragment_array.resize(points.shape[0])
        fragments = self._fragment_array.ndarray()
        fragments[:, 0:2] = points
        fragments[:, 2:6] = [0.0, 0.0, pixmap.width(), pixmap.height()]
        fragments[:, 6:10] = [1.0, 1.0, 0.0, 1.0]
        painter.drawPixmapFragments(
            *self._fragment_array.drawargs(), pixmap
        )

    def paint(self, painter: QPainter, _option, _widget=None) -> None:
        transform = painter.transform()
        painter.setRenderHint(QPainter.RenderHint.Antialiasing, False)
        prepared: list[tuple[tuple[int, int, int, int], np.ndarray]] = []
        for rgba, x_values, y_values in self._groups:
            x_visible, y_visible = self._visible(x_values, y_values)
            if x_visible.size:
                prepared.append((rgba, self._device_points(painter, x_visible, y_visible)))
        painter.resetTransform()
        for rgba, points in prepared:
            if self._symbol in ("o", "s"):
                self._draw_points(painter, points, rgba)
            else:
                self._draw_fragments(painter, points, rgba)
        painter.setTransform(transform)
