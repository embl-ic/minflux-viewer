"""Experimental precision-aware 2-D render view.

The window inherits the established render UI and interactions. Only the
localization-to-scalar-image stage is replaced.
"""

from __future__ import annotations

import math

import numpy as np
from PyQt6.QtCore import Qt, QTimer
from PyQt6.QtWidgets import (
    QComboBox,
    QDoubleSpinBox,
    QHBoxLayout,
    QLabel,
    QMessageBox,
    QWidget,
)

from .precision_render import (
    RENDER_METHOD_BASIC,
    RENDER_METHOD_BICUBIC,
    RENDER_METHOD_BILINEAR,
    RENDER_METHOD_FIXED_GAUSSIAN,
    RENDER_METHOD_HISTOGRAM,
    RENDER_METHOD_PRECISION_GAUSSIAN,
    PrecisionChannelData,
    PrecisionRenderScheduler,
    PrecisionTileRequest,
    PrecisionTileResult,
    ViewportScalarCache,
    resolve_precision_xyz_nm,
    transform_precision_marginals,
)
from .render_window import RenderWindow

# Localization-precision Gaussian is the default (top); the rest follow
# rough → smooth. (The old redundant "Automatic" alias was removed.)
_RENDER_METHOD_OPTIONS = (
    ("Localization-precision Gaussian", RENDER_METHOD_PRECISION_GAUSSIAN),
    ("Histogram", RENDER_METHOD_HISTOGRAM),
    ("Bilinear histogram", RENDER_METHOD_BILINEAR),
    ("Bicubic histogram", RENDER_METHOD_BICUBIC),
    ("Basic (smoothed histogram)", RENDER_METHOD_BASIC),
    ("Fixed Gaussian", RENDER_METHOD_FIXED_GAUSSIAN),
)
_RENDER_METHOD_LABELS = dict(
    (method, label) for label, method in _RENDER_METHOD_OPTIONS
)
# Per-item hover help shown in the dropdown as you scroll through the methods.
_RENDER_METHOD_TIPS = {
    RENDER_METHOD_PRECISION_GAUSSIAN:
        "Default. Each localization is a unit-mass, pixel-integrated\n"
        "ANISOTROPIC Gaussian sized by its OWN precision (per-loc →\n"
        "per-trace StdDev → calibration → 5 nm). Most faithful; slowest.",
    RENDER_METHOD_HISTOGRAM:
        "One count per pixel (the pixel each localization falls in).\n"
        "Fastest and rawest; grainy, no sub-pixel information.",
    RENDER_METHOD_BILINEAR:
        "Each count is split over the 4 nearest pixel centres by\n"
        "bilinear weights — keeps sub-pixel position, no blur (PSF-free).",
    RENDER_METHOD_BICUBIC:
        "Each count is split over a 4×4 neighbourhood with a Catmull-Rom\n"
        "cubic kernel — smoother sub-pixel than bilinear (~2× the cost;\n"
        "negative lobes clamped to zero).",
    RENDER_METHOD_BASIC:
        "The previous production look: histogram + a ½-pixel anti-alias\n"
        "blur that scales with zoom. Fast, smooth.",
    RENDER_METHOD_FIXED_GAUSSIAN:
        "One isotropic Gaussian of a FIXED sigma (the Sigma box) for\n"
        "every localization, via a fast histogram+blur (stays quick at\n"
        "any zoom / sigma).",
}


class PrecisionRenderWindow(RenderWindow):
    """SMAP-inspired advanced 2-D renderer with selectable reconstruction."""

    SUPPORTS_VOLUME_3D = True
    SIGMA_MENU_TEXT = "Render Methods…"
    RENDER_MODE = "advanced"
    _TILE_PX = 256
    _MIN_TILE_PIXEL_NM = 0.125
    _MAX_TILE_PIXEL_NM = 1024.0
    _CACHE_BYTES = 192 * 1024 * 1024
    _INTERACTION_DEBOUNCE_MS = 45
    _PROGRESSIVE_COALESCE_MS = 20

    def __init__(self, *args, **kwargs) -> None:
        # _build_channel_grid is dynamically dispatched during RenderWindow init.
        self._advanced_render_method = RENDER_METHOD_PRECISION_GAUSSIAN
        self._fixed_sigma_nm = 5.0
        self._precision_channels: dict[int, PrecisionChannelData] = {}
        self._precision_cache = ViewportScalarCache(
            max_bytes=self._CACHE_BYTES, max_items=2048
        )
        self._precision_scheduler: PrecisionRenderScheduler | None = None
        self._active_tile_generation = -1
        self._active_tile_keys: dict[int, list[tuple[int, int, tuple]]] = {}
        self._active_tile_geometry: tuple[float, float, float, float] | None = None
        self._active_tile_pixel_nm = 1.0
        self._active_tile_grid: tuple[int, int, int, int] | None = None
        # Coarse preview base: the previous frame, reused (resampled) to fill
        # not-yet-rendered tiles so a view change paints instantly and tiles
        # sharpen in progressively — the production renderer's responsiveness.
        self._preview_scalar: np.ndarray | None = None
        self._preview_geometry: tuple[float, float, float, float] | None = None
        self._last_frame_orientation: str | None = None
        super().__init__(*args, **kwargs)
        self._build_advanced_controls()
        self._redraw_timer.setInterval(self._INTERACTION_DEBOUNCE_MS)
        self._precision_scheduler = PrecisionRenderScheduler(parent=self)
        self._precision_scheduler.result_ready.connect(self._on_precision_tile_result)
        # Coalesce progressive re-composites so a burst of finished tiles paints
        # at most once per interval instead of once per tile.
        self._progressive_timer = QTimer(self)
        self._progressive_timer.setSingleShot(True)
        self._progressive_timer.setInterval(self._PROGRESSIVE_COALESCE_MS)
        self._progressive_timer.timeout.connect(
            lambda: self._compose_precision_tiles("partial")
        )
        self._update_overlay_title()
        self._schedule_render()

    def _update_overlay_title(self) -> None:
        super()._update_overlay_title()
        title = self.windowTitle()
        if not title.endswith(" [advanced]"):
            self.setWindowTitle(f"{title} [advanced]")

    def _build_advanced_controls(self) -> None:
        row = QWidget(self)
        layout = QHBoxLayout(row)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(6)

        layout.addWidget(QLabel("Renderer:"))
        self._render_method_combo = QComboBox()
        for index, (label, method) in enumerate(_RENDER_METHOD_OPTIONS):
            self._render_method_combo.addItem(label, method)
            tip = _RENDER_METHOD_TIPS.get(method)
            if tip:  # per-item hover help in the open dropdown
                self._render_method_combo.setItemData(
                    index, tip, Qt.ItemDataRole.ToolTipRole
                )
        self._render_method_combo.setToolTip(
            "Choose how localizations are reconstructed into scalar pixels.\n"
            "Hover an item in the list for what it does."
        )
        self._render_method_combo.currentIndexChanged.connect(
            self._on_render_method_changed
        )
        layout.addWidget(self._render_method_combo)

        self._fixed_sigma_label = QLabel("Sigma:")
        layout.addWidget(self._fixed_sigma_label)
        self._fixed_sigma_spin = QDoubleSpinBox()
        self._fixed_sigma_spin.setDecimals(2)
        self._fixed_sigma_spin.setRange(0.01, 10000.0)
        self._fixed_sigma_spin.setSingleStep(0.5)
        self._fixed_sigma_spin.setValue(self._fixed_sigma_nm)
        self._fixed_sigma_spin.setSuffix(" nm")
        self._fixed_sigma_spin.setToolTip(
            "Standard deviation of the isotropic Gaussian assigned to every localization."
        )
        self._fixed_sigma_spin.valueChanged.connect(self._on_fixed_sigma_changed)
        layout.addWidget(self._fixed_sigma_spin)
        layout.addStretch(1)
        self.layout().insertWidget(1, row)
        self._update_advanced_controls()

    def _update_advanced_controls(self) -> None:
        enabled = self._advanced_render_method == RENDER_METHOD_FIXED_GAUSSIAN
        self._fixed_sigma_label.setEnabled(enabled)
        self._fixed_sigma_spin.setEnabled(enabled)

    def _invalidate_advanced_render(self) -> None:
        self._precision_cache.clear()
        self._active_tile_keys = {}
        # The render model/params changed, so the last frame is no longer a
        # valid preview base for the new tiles.
        self._preview_scalar = None
        self._preview_geometry = None
        self._scheduler.cancel()
        if self._precision_scheduler is not None:
            self._precision_scheduler.cancel()
        self._schedule_render()

    def _on_render_method_changed(self, index: int) -> None:
        method = self._render_method_combo.itemData(index)
        if not method or method == self._advanced_render_method:
            return
        self._advanced_render_method = str(method)
        self._update_advanced_controls()
        self._invalidate_advanced_render()

    def _on_fixed_sigma_changed(self, value: float) -> None:
        value = max(float(value), 0.01)
        if np.isclose(value, self._fixed_sigma_nm):
            return
        self._fixed_sigma_nm = value
        if self._advanced_render_method == RENDER_METHOD_FIXED_GAUSSIAN:
            self._invalidate_advanced_render()

    def _render_method_label(self) -> str:
        return _RENDER_METHOD_LABELS.get(
            self._advanced_render_method, self._advanced_render_method
        )

    def _build_channel_grid(self, ch: dict) -> None:
        super()._build_channel_grid(ch)
        ds_idx = ch["dataset_idx"]
        ds = self._state.datasets[ds_idx]
        try:
            raw_locs = np.asarray(ds.loc_nm, dtype=np.float64)
        except Exception:
            raw_locs = np.empty((0, 3), dtype=np.float64)
        if raw_locs.ndim != 2 or raw_locs.shape[1] < 2:
            raw_locs = np.empty((0, 3), dtype=np.float64)
        elif raw_locs.shape[1] == 2:
            raw_locs = np.column_stack(
                [raw_locs, np.zeros(raw_locs.shape[0], dtype=np.float64)]
            )

        n_rows = raw_locs.shape[0]
        keep = np.asarray(ds.filter_mask, dtype=bool)
        if keep.shape != (n_rows,):
            keep = np.ones(n_rows, dtype=bool)
        if n_rows:
            keep &= np.all(np.isfinite(raw_locs[:, :3]), axis=1)

        sigma_xyz, source = resolve_precision_xyz_nm(ds, n_rows)
        sigma_xyz = sigma_xyz[keep]
        transform = ds.state.get("overlay_transform") or ds.state.get("render_transform_2d")
        matrix = self._transform_matrix4(transform) if transform else None
        sigma_xyz = transform_precision_marginals(sigma_xyz, matrix)

        if self._orientation == "XZ":
            order = (0, 2, 1)
        elif self._orientation == "YZ":
            order = (1, 2, 0)
        else:
            order = (0, 1, 2)
        sigma_oriented = sigma_xyz[:, order] if sigma_xyz.size else np.empty((0, 3))
        x, y, depth = self._channel_locs_xyz.get(
            ds_idx,
            (
                np.empty(0, dtype=np.float64),
                np.empty(0, dtype=np.float64),
                np.empty(0, dtype=np.float64),
            ),
        )
        if len(x) != len(sigma_oriented):
            # Keep the inherited coordinate pipeline authoritative if malformed
            # precision metadata cannot be aligned one-to-one.
            sigma_oriented = np.full((len(x), 3), 5.0, dtype=np.float64)
            source = "5 nm fallback (precision rows did not align)"

        self._precision_channels[ds_idx] = PrecisionChannelData(
            dataset_idx=ds_idx,
            x_nm=x,
            y_nm=y,
            depth_nm=depth,
            sigma_x_nm=np.ascontiguousarray(sigma_oriented[:, 0], dtype=np.float64),
            sigma_y_nm=np.ascontiguousarray(sigma_oriented[:, 1], dtype=np.float64),
            sigma_depth_nm=np.ascontiguousarray(sigma_oriented[:, 2], dtype=np.float64),
            grid=self._channel_grids.get(ds_idx),
            source=source,
        )
        self._precision_cache.remove_dataset(ds_idx)
        self._preview_scalar = None
        self._preview_geometry = None

    def _depth_key(self) -> tuple[float, float] | None:
        if not self._has_depth or self._all_depth_check.isChecked():
            return None
        return tuple(round(float(value), 4) for value in self._depth_range)

    def _tile_pixel_size_nm(
        self, bounds: tuple[float, float, float, float]
    ) -> float:
        viewport = self._image_view.ui.graphicsView.viewport()
        width = max(int(viewport.width()), 96)
        height = max(int(viewport.height()), 96)
        x0, x1, y0, y1 = bounds
        target = max((x1 - x0) / width, (y1 - y0) / height)
        if not np.isfinite(target) or target <= 0.0:
            return 1.0
        pixel_nm = 2.0 ** math.floor(math.log2(target))
        return float(
            np.clip(
                pixel_nm, self._MIN_TILE_PIXEL_NM, self._MAX_TILE_PIXEL_NM
            )
        )

    def _tile_key(
        self,
        channel: PrecisionChannelData,
        pixel_nm: float,
        tile_row: int,
        tile_col: int,
    ) -> tuple:
        ch = next(
            (item for item in self._channels if item["dataset_idx"] == channel.dataset_idx),
            {},
        )
        transform_key = self._channel_loc_transform_key(ch) if ch else None
        return (
            "precision-tile",
            channel.dataset_idx,
            self._mask_versions.get(channel.dataset_idx, 0),
            self._orientation,
            transform_key,
            self._depth_key(),
            self._advanced_render_method,
            round(self._fixed_sigma_nm, 6),
            round(float(pixel_nm), 9),
            int(tile_row),
            int(tile_col),
            id(channel.sigma_x_nm),
            id(channel.sigma_y_nm),
        )

    def _tile_plan(
        self, bounds: tuple[float, float, float, float]
    ) -> tuple[
        float,
        tuple[float, float, float, float],
        tuple[int, int, int, int],
        dict[int, list[tuple[int, int, tuple]]],
        list[PrecisionTileRequest],
    ]:
        pixel_nm = self._tile_pixel_size_nm(bounds)
        tile_span = self._TILE_PX * pixel_nm
        x0, x1, y0, y1 = bounds
        col0 = int(math.floor(x0 / tile_span))
        col1 = int(math.floor(np.nextafter(x1, -np.inf) / tile_span))
        row0 = int(math.floor(y0 / tile_span))
        row1 = int(math.floor(np.nextafter(y1, -np.inf) / tile_span))
        geometry = (
            col0 * tile_span,
            (col1 + 1) * tile_span,
            row0 * tile_span,
            (row1 + 1) * tile_span,
        )
        depth_range = None if self._depth_key() is None else self._depth_range
        keys_by_dataset: dict[int, list[tuple[int, int, tuple]]] = {}
        missing: list[PrecisionTileRequest] = []
        for ch in self._channels:
            if ch["kind"] != "localizations":
                continue
            channel = self._precision_channels.get(ch["dataset_idx"])
            if channel is None or channel.grid is None:
                continue
            entries: list[tuple[int, int, tuple]] = []
            for tile_row in range(row0, row1 + 1):
                for tile_col in range(col0, col1 + 1):
                    key = self._tile_key(channel, pixel_nm, tile_row, tile_col)
                    entries.append((tile_row, tile_col, key))
                    if self._precision_cache.get(key) is None:
                        tile_bounds = (
                            tile_col * tile_span,
                            (tile_col + 1) * tile_span,
                            tile_row * tile_span,
                            (tile_row + 1) * tile_span,
                        )
                        missing.append(
                            PrecisionTileRequest(
                                key=key,
                                channel=channel,
                                bounds=tile_bounds,
                                shape=(self._TILE_PX, self._TILE_PX),
                                depth_range=depth_range,
                                render_method=self._advanced_render_method,
                                fixed_sigma_nm=self._fixed_sigma_nm,
                            )
                        )
            keys_by_dataset[channel.dataset_idx] = entries
        return (
            pixel_nm,
            geometry,
            (row0, row1, col0, col1),
            keys_by_dataset,
            missing,
        )

    def _render(self) -> None:
        self._apply_y_axis_direction()
        if self._render_mode == "image":
            super()._render_image_mode()
            return
        if self._precision_scheduler is None:
            return

        (x0, x1), (y0, y1) = self._view_box.viewRange()
        bounds = (float(x0), float(x1), float(y0), float(y1))
        if x1 <= x0 or y1 <= y0:
            return

        # One consistent precision-tile path at every zoom level — the tile
        # pixel size adapts to the view (coarse when zoomed out, fine when zoomed
        # in), so there is no separate production-style overview to switch styles.
        self._scheduler.cancel()
        (
            pixel_nm,
            geometry,
            tile_grid,
            keys_by_dataset,
            missing,
        ) = self._tile_plan(bounds)
        self._active_tile_keys = keys_by_dataset
        self._active_tile_geometry = geometry
        self._active_tile_pixel_nm = pixel_nm
        self._active_tile_grid = tile_grid

        if not missing:
            self._precision_scheduler.cancel()
            self._active_tile_generation = self._precision_scheduler.generation
            self._compose_precision_tiles("cached")
            return

        # Snapshot the current frame as the coarse preview base, request the
        # missing precision tiles, then paint an immediate composite (preview
        # where tiles are still loading, sharp where already cached). Tiles then
        # fill in progressively via _on_precision_tile_result.
        self._capture_preview_base()
        self._progressive_timer.stop()
        self._active_tile_generation = self._precision_scheduler.request(missing)
        self._compose_precision_tiles("preview")

    def _all_active_tiles_cached(self) -> bool:
        return all(
            self._precision_cache.get(key) is not None
            for entries in self._active_tile_keys.values()
            for _row, _col, key in entries
        )

    def _capture_preview_base(self) -> None:
        """Remember the current frame as the coarse base for the next render.

        Only used when it matches the current orientation (an XY/XZ/YZ flip maps
        different world axes, so a cross-orientation frame would smear).
        """
        if (
            self._last_scalar_tile is not None
            and self._last_tile_geometry is not None
            and self._last_frame_orientation == self._orientation
        ):
            self._preview_scalar = self._last_scalar_tile
            self._preview_geometry = self._last_tile_geometry
        else:
            self._preview_scalar = None
            self._preview_geometry = None

    def _resample_preview(
        self, geometry: tuple[float, float, float, float], shape: tuple[int, int]
    ) -> np.ndarray | None:
        """Nearest-neighbour resample the preview base into *geometry*/*shape*.

        Returns ``(C, H, W)`` float32 with 0 outside the base's coverage, or
        ``None`` when there is no usable preview base.
        """
        prev = self._preview_scalar
        pgeo = self._preview_geometry
        if prev is None or pgeo is None or prev.ndim != 3:
            return None
        height, width = shape
        gx0, gx1, gy0, gy1 = geometry
        px0, px1, py0, py1 = pgeo
        if px1 <= px0 or py1 <= py0 or gx1 <= gx0 or gy1 <= gy0:
            return None
        _c, h0, w0 = prev.shape
        wx = gx0 + (np.arange(width) + 0.5) * (gx1 - gx0) / width
        wy = gy0 + (np.arange(height) + 0.5) * (gy1 - gy0) / height
        col = np.floor((wx - px0) / (px1 - px0) * w0).astype(np.int64)
        row = np.floor((wy - py0) / (py1 - py0) * h0).astype(np.int64)
        col_ok = (col >= 0) & (col < w0)
        row_ok = (row >= 0) & (row < h0)
        out = prev[:, np.clip(row, 0, h0 - 1)[:, None], np.clip(col, 0, w0 - 1)[None, :]]
        mask = row_ok[:, None] & col_ok[None, :]
        return np.where(mask[None, :, :], out, 0.0).astype(np.float32)

    def _compose_precision_tiles(self, stage: str) -> None:
        if self._active_tile_geometry is None or self._active_tile_grid is None:
            return
        row0, row1, col0, col1 = self._active_tile_grid
        height = (row1 - row0 + 1) * self._TILE_PX
        width = (col1 - col0 + 1) * self._TILE_PX
        x0, x1, y0, y1 = self._active_tile_geometry
        # Coarse base (the previous frame, stretched) fills tiles not yet ready.
        # On the final stages every block is overwritten, so skip the resample.
        base = (
            self._resample_preview(self._active_tile_geometry, (height, width))
            if stage in ("preview", "partial")
            else None
        )
        channels = []
        ready = total = 0
        for ci, ch in enumerate(self._channels):
            if ch["kind"] == "image":
                tile = np.asarray(
                    self._image_tile(
                        self._state.datasets[ch["dataset_idx"]],
                        x0, x1, y0, y1, height, width,
                    ),
                    dtype=np.float32,
                )
                channels.append(tile)
                continue
            if base is not None and ci < base.shape[0]:
                tile = np.array(base[ci], dtype=np.float32, copy=True)
            else:
                tile = np.zeros((height, width), dtype=np.float32)
            for tile_row, tile_col, key in self._active_tile_keys.get(
                ch["dataset_idx"], []
            ):
                total += 1
                value = self._precision_cache.get(key)
                if value is None:
                    continue  # keep the coarse placeholder for this block
                ready += 1
                dst_row = (tile_row - row0) * self._TILE_PX
                dst_col = (tile_col - col0) * self._TILE_PX
                tile[
                    dst_row : dst_row + self._TILE_PX,
                    dst_col : dst_col + self._TILE_PX,
                ] = value
            channels.append(tile)
        if not channels:
            return

        scalar = np.stack(channels, axis=0).astype(np.float32, copy=False)
        self._last_scalar_tile = scalar
        self._last_tile_geometry = self._active_tile_geometry
        self._last_px_nm = self._active_tile_pixel_nm
        self._last_lod = -3
        self._last_frame_orientation = self._orientation
        final = stage in ("ready", "cached")
        # Only settle auto brightness/contrast on the final image so levels do
        # not flicker as tiles fill in over the coarse preview.
        if final and self._auto_bc and scalar.shape[0] == 1:
            levels = self._compute_auto_levels(scalar[0])
            if levels is not None:
                self._manual_levels = levels
                self._channels[0]["levels"] = None
                if self._bc_dialog is not None and self._bc_dialog.isVisible():
                    self._bc_dialog.set_levels(*levels)

        self._set_composited_image(scalar)
        self._redraw_roi_highlight()

        cache_mb = self._precision_cache.nbytes / (1024.0 * 1024.0)
        if final:
            self._info_label.setText(
                f"{self._dataset_dim_label}  |  {self._render_method_label()} "
                f"({stage})  |  px={self._active_tile_pixel_nm:.3g} nm  |  "
                f"{total} tile(s)  |  cache {cache_mb:.1f} MB"
            )
        else:
            self._info_label.setText(
                f"{self._dataset_dim_label}  |  {self._render_method_label()}  |  "
                f"px={self._active_tile_pixel_nm:.3g} nm  |  {ready}/{total} tile(s)…"
            )
        if final and self._bc_dialog is not None and self._bc_dialog.isVisible():
            self._bc_dialog.set_data(scalar[self._active_channel_index()])

    def _set_composited_image(self, scalar: np.ndarray) -> None:
        if self._last_tile_geometry is None:
            return
        x0, x1, y0, y1 = self._last_tile_geometry
        height, width = scalar.shape[1:]
        rgba = self._compose_rgba(scalar)
        self._image_view.setImage(
            rgba,
            autoRange=False,
            autoLevels=False,
            pos=[x0, y0],
            scale=[(x1 - x0) / width, (y1 - y0) / height],
        )

    def _compose_from_cache(self, *_args) -> None:
        if self._last_scalar_tile is None:
            self._schedule_render()
            return
        self._set_composited_image(self._last_scalar_tile)
        self._redraw_roi_highlight()

    def _on_precision_tile_result(self, result: PrecisionTileResult) -> None:
        if self._precision_scheduler is None:
            return
        if (
            result.generation != self._precision_scheduler.generation
            or result.generation != self._active_tile_generation
        ):
            return
        self._precision_cache.put(result.key, result.array)
        if self._all_active_tiles_cached():
            self._progressive_timer.stop()
            self._compose_precision_tiles("ready")
            return
        # Coalesce a burst of finished tiles into one progressive re-composite.
        if not self._progressive_timer.isActive():
            self._progressive_timer.start()

    def _show_sigma_dialog(self) -> None:
        sources = sorted({channel.source for channel in self._precision_channels.values()})
        source_text = "\n".join(f"- {source}" for source in sources) or "- no localization channel"
        QMessageBox.information(
            self,
            "Advanced Rendering Methods",
            "• Localization-precision Gaussian (default) — a unit-mass,\n"
            "  pixel-integrated ANISOTROPIC Gaussian per localization,\n"
            "  sized by its own precision. The most faithful, and slowest.\n"
            "• Histogram — one count per pixel (raw, grainy).\n"
            "• Bilinear histogram — one count split over the 4 nearest\n"
            "  pixel centres (keeps sub-pixel position).\n"
            "• Bicubic histogram — one count split over a 4×4 neighbourhood\n"
            "  with a Catmull-Rom cubic kernel (smoother sub-pixel).\n"
            "• Basic (smoothed histogram) — histogram + a ½-pixel blur;\n"
            "  the previous production look, fast.\n"
            "• Fixed Gaussian — one isotropic sigma for every localization\n"
            "  (rendered via a fast histogram+blur, so it stays quick).\n\n"
            "Precision source(s):\n"
            f"{source_text}\n\n"
            "Source priority: per-localization precision, per-trace StdDev,\n"
            "dataset calibration, then the reported 5 nm fallback.",
        )

    def _current_3d_region(self):
        """The current 2-D view mapped to a native 3-D box (xlo,xhi,ylo,yhi,
        zlo,zhi nm) so the volume focuses on what's on screen. The off-plane axis
        uses the depth slider when active, else the full data extent. Returns
        None (→ whole-data volume) for an overlay-transformed dataset (its
        display coords aren't native)."""
        if self._idx is None or not (0 <= self._idx < len(self._state.datasets)):
            return None
        ds = self._state.datasets[self._idx]
        if ds.state.get("overlay_transform") or ds.state.get("render_transform_2d"):
            return None
        try:
            (vx0, vx1), (vy0, vy1) = self._view_box.viewRange()
            loc = np.asarray(ds.loc_nm, dtype=np.float64)
        except Exception:
            return None
        if loc.ndim != 2 or loc.shape[1] < 3:
            return None
        finite = np.all(np.isfinite(loc[:, :3]), axis=1)
        if not np.any(finite):
            return None
        lo = np.nanmin(loc[finite, :3], axis=0)
        hi = np.nanmax(loc[finite, :3], axis=0)
        ix0, ix1 = sorted((float(vx0), float(vx1)))
        iy0, iy1 = sorted((float(vy0), float(vy1)))

        def off(axis):
            if self._has_depth and not self._all_depth_check.isChecked():
                d0, d1 = self._depth_range
                return min(d0, d1), max(d0, d1)
            return float(lo[axis]), float(hi[axis])

        if self._orientation == "XZ":       # view x→X, view y→Z, off→Y
            oy0, oy1 = off(1)
            return (ix0, ix1, oy0, oy1, iy0, iy1)
        if self._orientation == "YZ":       # view x→Y, view y→Z, off→X
            ox0, ox1 = off(0)
            return (ox0, ox1, ix0, ix1, iy0, iy1)
        oz0, oz1 = off(2)                    # XY: view x→X, view y→Y, off→Z
        return (ix0, ix1, iy0, iy1, oz0, oz1)

    def _current_2d_contrast_pct(self) -> tuple[float, float]:
        """The active channel's 2-D brightness/contrast expressed as (black,
        white) percentiles of its nonzero scalar — so the 3-D volume can apply
        the SAME percentile stretch and show comparable voxel visibility despite
        the pixel-vs-voxel scale difference."""
        default = (0.0, 99.7)
        try:
            pixels = self._bc_pixels()
            if pixels is None:
                return default
            ci = self._active_channel_index()
            levels = (
                self._channels[ci].get("levels")
                if 0 <= ci < len(self._channels) else None
            )
            if levels is None:
                levels = (
                    self._manual_levels
                    if (len(self._channels) == 1 and self._manual_levels)
                    else self._compute_auto_levels(pixels)
                )
            if not levels:
                return default
            lo, hi = float(levels[0]), float(levels[1])
            vals = np.asarray(pixels, dtype=np.float64).ravel()
            nz = vals[vals > 0.0]
            if nz.size == 0:
                return default
            black = float(np.mean(nz < lo) * 100.0)
            white = float(np.mean(nz < hi) * 100.0)
            if white <= black:
                white = min(black + 1.0, 100.0)
            return (black, white)
        except Exception:
            return default

    def _show_3d_volume_window(self) -> None:
        if self._idx is None or not (0 <= self._idx < len(self._state.datasets)):
            return
        self._state.set_active(self._idx)
        region = self._current_3d_region()
        contrast = self._current_2d_contrast_pct()
        if self._volume_window is None:
            from .volume_window import VolumeRenderWindow
            self._volume_window = VolumeRenderWindow(
                self._state, self._idx,
                sigma_nm_xyz=self._sigma_nm_xyz,
                render_method=self._advanced_render_method,
                region_bounds=region,
                contrast_pct=contrast,
                parent=self,
            )
            self._volume_window.destroyed.connect(
                lambda *_: setattr(self, "_volume_window", None)
            )
        else:
            self._volume_window._sigma_nm_xyz = self._sigma_nm_xyz
            self._volume_window._render_method = self._advanced_render_method
            self._volume_window._region_bounds = region
            self._volume_window.set_contrast_pct(*contrast)
            self._volume_window.refresh_from_dataset()
        self._volume_window.show()
        self._volume_window.raise_()
        self._volume_window.activateWindow()

    def closeEvent(self, event) -> None:
        self._redraw_timer.stop()
        self._progressive_timer.stop()
        self._scheduler.cancel()
        if self._precision_scheduler is not None:
            self._precision_scheduler.cancel()
        super().closeEvent(event)
