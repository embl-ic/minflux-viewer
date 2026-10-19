"""Monitor-resolution input layer independent of scientific convolution grids."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from PyQt6.QtCore import QObject, QRectF, QTimer

from minflux_viewer.colormaps import colormap_lut
from minflux_viewer.core.inspection_raster import histogram_preview
from minflux_viewer.core.spatial_grid import SpatialGrid
from minflux_viewer.ui.common.background_tasks import (
    BackgroundTask,
    detach_task_signals,
    retire_background_tasks,
    shared_thread_pool,
)
from minflux_viewer.ui.widgets.layer_brightness import percentile_levels


@dataclass(frozen=True)
class InspectionAppearance:
    """Copied channel appearance; LUT contains gamma/inversion/background already."""

    lut: np.ndarray
    contrast_pct: tuple[float, float] = (0., 99.7)
    white_background: bool = False
    source: str = "render defaults"


def render_appearance(owner, state, idx):
    """Read an existing basic/advanced render, without opening or changing it."""
    key = state.key_at(idx)
    if owner is not None:
        for registry in (getattr(owner, "_render_windows", {}),
                         getattr(owner, "_advanced_render_windows", {})):
            for window in registry.values():
                appearance = window.inspection_appearance(key)
                if appearance is not None:
                    return appearance
    ds = state.datasets[idx]
    name = ds.state.get("render_channel_lut") or state.prefs.get("plot", {}).get("render_cmap", "hot")
    return InspectionAppearance(colormap_lut(name, alpha=False))


class InspectionLayer(QObject):
    """One running + one latest pending histogram, retired without GUI waits."""

    def __init__(self, owner, plot, image_item, appearance, on_ready, on_error):
        super().__init__(owner)
        self.plot, self.item = plot, image_item
        self.appearance = appearance
        self.on_ready, self.on_error = on_ready, on_error
        self.pixels = np.empty((0, 0))
        self.levels = (0., 1.)
        self._xy = np.empty((0, 2))
        self._grid = None
        self._source_epoch = 0
        self._bounds = None
        self._generation = 0
        self._closed = False
        self._task = None
        self._pending = False
        self._timer = QTimer(self)
        self._timer.setSingleShot(True)
        self._timer.setInterval(100)
        self._timer.timeout.connect(self._dispatch)
        plot.getViewBox().sigRangeChanged.connect(self.request)
        plot.getViewBox().sigResized.connect(self.request)
        self.set_appearance(appearance)

    def set_appearance(self, appearance):
        self.appearance = appearance
        self.item.setLookupTable(appearance.lut)
        self.plot.setBackground("w" if appearance.white_background else "k")
        self._set_default_levels()

    def _set_default_levels(self):
        self.levels = percentile_levels(self.pixels, *self.appearance.contrast_pct,
                                        positive_only=True) or (0., 1.)

    def set_points(self, xy):
        self._xy = xy
        self._grid = None
        self._source_epoch += 1
        self._bounds = None
        self.pixels = np.empty((0, 0))
        self.item.clear()
        if len(xy):
            lo, hi = xy.min(axis=0), xy.max(axis=0)
            self._bounds = (lo, hi)
            span = np.maximum(hi - lo, 1.)
            rect = QRectF(float(lo[0]), float(lo[1]), float(span[0]), float(span[1]))
            self.plot.getViewBox().setRange(rect, padding=.02)
        self.request()

    def request(self, *_args):
        if self._closed:
            return
        self._generation += 1
        self._pending = True
        if self._task is not None:
            self._task.cancel()
        self._timer.start()

    def _dispatch(self):
        if self._closed or self._task is not None:
            return
        self._pending = False
        if not len(self._xy):
            return
        generation, xy = self._generation, self._xy
        source_epoch, grid, source_bounds = self._source_epoch, self._grid, self._bounds
        view = self.plot.getViewBox()
        (x0, x1), (y0, y1) = view.viewRange()
        bounds = (x0, x1, y0, y1)
        ratio = self.plot.devicePixelRatioF()
        shape = tuple(max(32, min(1024, int(n * ratio))) for n in (view.width(), view.height()))

        def work(report):
            report("Rendering inspection histogram")
            index = grid
            lo, hi = source_bounds
            if index is None:
                # Bounded cells also prevent pathological empty-space zoom-outs
                # from turning the spatial query into an enormous nested loop.
                cell_size = max(float(np.max(hi-lo)) / 64., 1.)
                index = SpatialGrid(xy[:, 0], xy[:, 1], cell_size)
            report("Selecting visible inspection points")
            clipped = (max(x0, lo[0]), min(x1, hi[0]), max(y0, lo[1]), min(y1, hi[1]))
            rows = index.query(*clipped) if clipped[0] <= clipped[1] and clipped[2] <= clipped[3] else np.empty(0, dtype=int)
            pixels = histogram_preview(xy[rows], bounds, shape)
            report("Inspection histogram ready")
            return pixels, index

        task = BackgroundTask(work, description="Convolution Data inspection raster", category="render",
                              estimated_bytes=64 * shape[0] * shape[1] + (160 if grid is None else 48) * len(xy))
        self._task = task

        def done(result):
            pixels, index = result
            if not self._closed and source_epoch == self._source_epoch:
                self._grid = index
            if self._closed or generation != self._generation or task.is_cancelled:
                return
            self.pixels = pixels
            self._set_default_levels()
            self.item.setImage(pixels.T, levels=self.levels)
            self.item.setRect(QRectF(x0, y0, x1-x0, y1-y0))
            self.on_ready()

        def failed(message):
            if not self._closed and generation == self._generation:
                self.on_error(message)

        def finished():
            detach_task_signals(task)
            if self._closed:
                return
            self._task = None
            if self._pending and not self._timer.isActive():
                self._dispatch()

        task.signals.done.connect(done)
        task.signals.failed.connect(failed)
        task.signals.finished.connect(finished)
        shared_thread_pool("render-tiles").start(task)

    def close(self):
        self._closed = True
        self._grid = None
        self._timer.stop()
        if self._task is not None:
            retire_background_tasks([self._task])
            self._task = None
