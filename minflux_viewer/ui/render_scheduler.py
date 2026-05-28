"""
minflux_viewer.ui.render_scheduler
=====================================
Async tile-render pipeline for the pyramid render window.

RenderScheduler manages a QThreadPool of TileWorker tasks.  A monotonically
increasing generation counter ensures that stale results (produced after the
viewport has moved again) are silently discarded rather than displayed.

Usage
-----
    scheduler = RenderScheduler(on_tile_ready_callback)
    scheduler.request(tile_keys, channel_locs, spatial_grid, sigma_nm, lod)
    # callback fires on the main thread for each completed tile

Cancellation
------------
Call ``scheduler.cancel()`` to abandon all in-flight work (e.g. on dataset
change).  The generation counter is incremented so any workers that finish
after cancel() are silently dropped.
"""

from __future__ import annotations

import os
from typing import Callable

import numpy as np
from PyQt6.QtCore import QObject, QRunnable, QThreadPool, pyqtSignal, pyqtSlot

from .render_config import (
    PHYSICAL_TILE_NM,
    PER_LOC_SWITCH_COUNT,
    PREFETCH_PRIORITY,
    WORKER_THREAD_COUNT,
    actual_pixel_size_nm,
    render_tile_px,
)
from .tile_cache import TileKey


# ---------------------------------------------------------------------------
# Worker signals helper (QRunnable can't emit signals directly)
# ---------------------------------------------------------------------------

class _WorkerSignals(QObject):
    tile_ready = pyqtSignal(object, object)  # (TileKey, np.ndarray)


# ---------------------------------------------------------------------------
# TileWorker
# ---------------------------------------------------------------------------

class TileWorker(QRunnable):
    """Renders one physical tile in a background thread.

    xnm, ynm are the projected "column 0" and "column 1" coordinates for the
    current orientation (already re-ordered by _oriented_locs).  znm is the
    depth coordinate (oriented[:, 2]).  depth_range is the active depth gate
    as (lo_nm, hi_nm) or None.  The spatial_grid was built from xnm/ynm.
    """

    def __init__(
        self,
        key: TileKey,
        generation: int,
        scheduler: "RenderScheduler",
        xnm: np.ndarray,
        ynm: np.ndarray,
        znm: np.ndarray,
        spatial_grid,
        sigma_nm: tuple[float, float],
        depth_range: tuple[float, float] | None,
        signals: _WorkerSignals,
        tile_grid_x0: float,
        tile_grid_y0: float,
    ) -> None:
        super().__init__()
        self.setAutoDelete(True)
        self._key = key
        self._generation = generation
        self._scheduler = scheduler
        self._xnm = xnm
        self._ynm = ynm
        self._znm = znm
        self._grid = spatial_grid
        self._sigma_nm = sigma_nm
        self._depth_range = depth_range
        self._signals = signals
        self._tile_grid_x0 = tile_grid_x0
        self._tile_grid_y0 = tile_grid_y0

    def run(self) -> None:
        if self._scheduler.generation != self._generation:
            return

        key = self._key
        tile_x0 = self._tile_grid_x0 + key.tile_col * PHYSICAL_TILE_NM
        tile_x1 = tile_x0 + PHYSICAL_TILE_NM
        tile_y0 = self._tile_grid_y0 + key.tile_row * PHYSICAL_TILE_NM
        tile_y1 = tile_y0 + PHYSICAL_TILE_NM

        indices = self._grid.query(tile_x0, tile_x1, tile_y0, tile_y1)
        if len(indices) == 0:
            tile_px = render_tile_px(key.lod)
            result = np.zeros((tile_px, tile_px), dtype=np.float32)
            if self._scheduler.generation == self._generation:
                self._signals.tile_ready.emit(key, result)
            return

        x = self._xnm[indices]
        y = self._ynm[indices]

        # Exact clip to tile bounds
        in_tile = (x >= tile_x0) & (x <= tile_x1) & (y >= tile_y0) & (y <= tile_y1)

        # Depth filter — znm is always the depth axis for oriented locs
        if self._depth_range is not None:
            z = self._znm[indices]
            d_lo, d_hi = self._depth_range
            in_tile &= (z >= d_lo) & (z <= d_hi)

        x = x[in_tile]
        y = y[in_tile]
        count = len(x)

        tile_px = render_tile_px(key.lod)
        px_nm = actual_pixel_size_nm(key.lod)

        if count == 0:
            result = np.zeros((tile_px, tile_px), dtype=np.float32)
        elif count < PER_LOC_SWITCH_COUNT or key.lod >= 3:
            result = _render_per_localization(x, y, tile_x0, tile_y0, px_nm, tile_px, self._sigma_nm)
        else:
            result = _render_histogram(x, y, tile_x0, tile_x1, tile_y0, tile_y1, tile_px, self._sigma_nm, px_nm)

        if self._scheduler.generation == self._generation:
            self._signals.tile_ready.emit(key, result)


# ---------------------------------------------------------------------------
# Render functions (called from worker thread)
# ---------------------------------------------------------------------------

def _render_histogram(
    x: np.ndarray,
    y: np.ndarray,
    x0: float,
    x1: float,
    y0: float,
    y1: float,
    tile_px: int,
    sigma_nm: tuple[float, float],
    px_nm: float,
) -> np.ndarray:
    from scipy.ndimage import gaussian_filter

    hist, _, _ = np.histogram2d(
        y, x,
        bins=[tile_px, tile_px],
        range=[[y0, y1], [x0, x1]],
    )
    sigma_px = (
        max(sigma_nm[0] / px_nm, 0.5),
        max(sigma_nm[1] / px_nm, 0.5),
    )
    if max(sigma_px) < 30.0:
        hist = gaussian_filter(hist.astype(np.float32), sigma=sigma_px, mode="constant")
    return hist.astype(np.float32, copy=False)


def _render_per_localization(
    x: np.ndarray,
    y: np.ndarray,
    tile_x0: float,
    tile_y0: float,
    px_nm: float,
    tile_px: int,
    sigma_nm: tuple[float, float],
) -> np.ndarray:
    """Render each localization as an individual Gaussian blob.

    Uses a truncated (±3σ) Gaussian kernel around each point so the
    cost is O(count × kernel_px²) rather than O(count × tile_px²).
    """
    result = np.zeros((tile_px, tile_px), dtype=np.float32)
    if len(x) == 0:
        return result

    sigma_y_px = max(sigma_nm[0] / px_nm, 0.5)
    sigma_x_px = max(sigma_nm[1] / px_nm, 0.5)
    r_y = int(np.ceil(3.0 * sigma_y_px))
    r_x = int(np.ceil(3.0 * sigma_x_px))

    # Pixel coordinates of each localization centre
    px_x = (x - tile_x0) / px_nm  # column (x axis)
    px_y = (y - tile_y0) / px_nm  # row (y axis)

    for cx, cy in zip(px_x, px_y):
        c0 = int(round(cx)) - r_x
        c1 = int(round(cx)) + r_x + 1
        r0 = int(round(cy)) - r_y
        r1 = int(round(cy)) + r_y + 1

        # Clip to tile
        dc0 = max(0, c0)
        dc1 = min(tile_px, c1)
        dr0 = max(0, r0)
        dr1 = min(tile_px, r1)
        if dc1 <= dc0 or dr1 <= dr0:
            continue

        kc = np.arange(dc0, dc1, dtype=np.float32) - cx
        kr = np.arange(dr0, dr1, dtype=np.float32) - cy
        gauss = np.exp(-0.5 * (kr[:, None] ** 2 / sigma_y_px ** 2 + kc[None, :] ** 2 / sigma_x_px ** 2))
        result[dr0:dr1, dc0:dc1] += gauss

    return result


# ---------------------------------------------------------------------------
# RenderScheduler
# ---------------------------------------------------------------------------

class RenderScheduler(QObject):
    """Manages async tile rendering with generation-counter cancellation.

    Signals
    -------
    tile_ready(TileKey, np.ndarray)
        Emitted on the main thread when a tile finishes rendering.
    """

    tile_ready = pyqtSignal(object, object)  # (TileKey, np.ndarray)

    def __init__(self, parent: QObject | None = None) -> None:
        super().__init__(parent)
        self._generation: int = 0
        self._pool = QThreadPool.globalInstance()
        self._pool.setMaxThreadCount(max(1, min(WORKER_THREAD_COUNT, os.cpu_count() or 4)))
        self._in_flight: set[TileKey] = set()

    # ------------------------------------------------------------------
    # Public interface
    # ------------------------------------------------------------------

    @property
    def generation(self) -> int:
        return self._generation

    def request(
        self,
        tile_keys: list[TileKey],
        xnm: np.ndarray,
        ynm: np.ndarray,
        znm: np.ndarray,
        spatial_grid,
        sigma_nm: tuple[float, float],
        depth_range: tuple[float, float] | None,
        tile_grid_x0: float = 0.0,
        tile_grid_y0: float = 0.0,
        *,
        priority: int = 0,
    ) -> None:
        """Submit render jobs for the given tile keys.

        Already-in-flight tiles are not duplicated.  Callers should call
        new_generation() before request() for a new viewport so stale workers
        are automatically discarded.
        """
        signals = _WorkerSignals()
        signals.tile_ready.connect(self._on_worker_done)

        for key in tile_keys:
            if key in self._in_flight:
                continue
            self._in_flight.add(key)
            worker = TileWorker(
                key=key,
                generation=self._generation,
                scheduler=self,
                xnm=xnm,
                ynm=ynm,
                znm=znm,
                spatial_grid=spatial_grid,
                sigma_nm=sigma_nm,
                depth_range=depth_range,
                signals=signals,
                tile_grid_x0=tile_grid_x0,
                tile_grid_y0=tile_grid_y0,
            )
            self._pool.start(worker, priority)

    def cancel(self) -> None:
        """Abandon all in-flight work by incrementing the generation."""
        self._generation += 1
        self._in_flight.clear()

    def new_generation(self) -> int:
        """Increment and return the new generation counter."""
        self._generation += 1
        self._in_flight.clear()
        return self._generation

    # ------------------------------------------------------------------
    # Internal
    # ------------------------------------------------------------------

    @pyqtSlot(object, object)
    def _on_worker_done(self, key: TileKey, array: np.ndarray) -> None:
        self._in_flight.discard(key)
        # Generation was already checked inside TileWorker.run(); if we're
        # here the result is current — forward it to the render window.
        self.tile_ready.emit(key, array)
