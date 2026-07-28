"""
ROI duplicate / crop options dialog (Phase 1: rectangle ROIs).

Shown the first time a dataset is duplicated/cropped in a session (or whenever
"stop asking" is off). Collects a :class:`~minflux_viewer.core.roi_crop.CropOptions`;
the actual crop is executed by the caller via ``core.roi_crop``.
"""

from __future__ import annotations

import numpy as np
from PyQt6.QtCore import Qt
from PyQt6.QtWidgets import (
    QCheckBox,
    QDialog,
    QDialogButtonBox,
    QHBoxLayout,
    QLabel,
    QLineEdit,
    QSizePolicy,
    QVBoxLayout,
)

from ..core.roi_crop import CropOptions, parse_channel_spec


def auto_z_bounds(counts: np.ndarray, edges: np.ndarray) -> tuple[float, float]:
    """Data-driven Z bounds: the outermost histogram bins whose count stays at
    or above a threshold, trimming the sparse outlier tails. The threshold is
    ``max(1, ~5% of the peak)`` — i.e. the two ends are where the distribution
    has clearly fallen off. Falls back to the full extent when nothing (or
    everything) qualifies."""
    counts = np.asarray(counts, dtype=float)
    edges = np.asarray(edges, dtype=float)
    if counts.size == 0 or edges.size < 2 or counts.max() <= 0:
        return float(edges[0]), float(edges[-1])
    threshold = max(1.0, 0.05 * float(counts.max()))
    keep = np.flatnonzero(counts >= threshold)
    if keep.size == 0:
        return float(edges[0]), float(edges[-1])
    return float(edges[keep[0]]), float(edges[keep[-1] + 1])


class CropDialog(QDialog):
    """Collect ROI duplicate/crop options."""

    def __init__(
        self,
        dataset_name: str,
        *,
        has_roi: bool,
        channels: list[str] | None = None,
        z_values: np.ndarray | None = None,
        initial: CropOptions | None = None,
        parent=None,
    ) -> None:
        super().__init__(parent)
        self.setWindowTitle("Duplicate / crop")
        self.setMinimumWidth(460)
        self._channels = list(channels or [])
        init = initial or CropOptions()
        # Z-histogram hover read-out state (set in _build_z_section).
        self._z_plot = None
        self._z_counts = None
        self._z_edges = None
        self._z_cursor = None
        self._z_hover_label = QLabel(" ")

        root = QVBoxLayout(self)
        root.addWidget(QLabel(f"Dataset: <b>{dataset_name}</b>"))

        # With an active region ROI the duplicate always crops; this chooses the
        # ROI's exact outline vs its (axis-aligned) bounding box.
        self._exact_shape = QCheckBox("duplicate only the ROI region")
        self._exact_shape.setChecked(bool(init.exact_shape))
        self._exact_shape.setEnabled(has_roi)
        self._exact_shape.setToolTip(
            "On: keep data inside the exact ROI outline (oval/polygon/rotated rect).\n"
            "Off: keep data inside the ROI's bounding box (as if it were a rectangle ROI)."
        )
        root.addWidget(self._exact_shape)

        self._clip = QCheckBox("clip data to ROI (allow partial traces)")
        self._clip.setChecked(bool(init.clip))
        self._clip.setToolTip(
            "On: keep only localizations inside the ROI (traces may be cut).\n"
            "Off: keep a whole trace when its centroid is inside (trace-complete)."
        )
        root.addWidget(self._clip)

        # Channel selector — only meaningful for a multi-channel overlay.
        self._channel_edit = QLineEdit()
        if len(self._channels) > 1:
            default = init.channels or list(range(1, len(self._channels) + 1))
            self._channel_edit.setText(",".join(str(i) for i in default))
            self._channel_edit.setToolTip(
                "Channels to include (1-based), e.g. 1-3 or 1,3. Rows:\n"
                + "\n".join(f"  {i+1}: {n}" for i, n in enumerate(self._channels))
            )
            row = QHBoxLayout()
            row.addWidget(QLabel("Channels:"))
            row.addWidget(self._channel_edit, 1)
            root.addLayout(row)

        # Z section — only for 3-D data.
        self._all_z = QCheckBox("All Z")
        self._slider = None
        self._region = None         # LinearRegionItem on the histogram
        self._syncing = False       # guard against slider<->region feedback
        self._z_label = QLabel("")
        self._has_z = z_values is not None and np.asarray(z_values).size > 0
        if self._has_z:
            self._build_z_section(root, np.asarray(z_values, dtype=float), init)

        self._spatial = QCheckBox("ROI as spatial filter (data outside preserved in the duplicate)")
        self._spatial.setChecked(bool(init.spatial_filter))
        self._spatial.setToolTip(
            "On (Model A): duplicate the full dataset; the ROI gates it as a "
            "reversible filter.\nOff (Model B): create a real subset of only the "
            "in-ROI localizations."
        )
        root.addWidget(self._spatial)

        self._stop_asking = QCheckBox("use the same setup for ROI options and stop asking")
        self._stop_asking.setChecked(bool(init.stop_asking))
        root.addWidget(self._stop_asking)

        buttons = QDialogButtonBox(
            QDialogButtonBox.StandardButton.Ok | QDialogButtonBox.StandardButton.Cancel
        )
        buttons.accepted.connect(self.accept)
        buttons.rejected.connect(self.reject)
        root.addWidget(buttons)

        self._all_z.toggled.connect(self._sync_enabled)
        self._sync_enabled()
        # Open tall enough to see the Z histogram; the plot then grows on resize.
        if self._has_z:
            self.resize(520, 620)

    # ------------------------------------------------------------------
    def _build_z_section(self, root, z: np.ndarray, init: CropOptions) -> None:
        z = z[np.isfinite(z)]
        if z.size == 0:
            self._has_z = False
            return
        zmin, zmax = float(z.min()), float(z.max())
        if zmax <= zmin:
            zmax = zmin + 1.0
        counts, edges = np.histogram(z, bins=min(64, max(8, z.size // 20)))
        self._z_counts = np.asarray(counts, dtype=float)
        self._z_edges = np.asarray(edges, dtype=float)
        # Default the two Z bounds to the data-driven edges (sparse tails
        # trimmed) so they're ready when the user unchecks "All Z".
        lo, hi = init.z_range if init.z_range is not None else auto_z_bounds(counts, edges)

        root.addWidget(QLabel(
            "Z distribution — drag the shaded range or the slider; drag the dialog "
            "taller to expand the count axis, hover a bar to read its count:"))
        try:
            import pyqtgraph as pg
            plot = pg.PlotWidget()
            plot.setMinimumHeight(120)
            # Grow vertically with the dialog so the count axis expands on drag.
            plot.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Expanding)
            plot.setMouseEnabled(x=False, y=False)
            plot.setLabel("left", "count")
            plot.setLabel("bottom", "Z (nm)")
            centers = 0.5 * (edges[:-1] + edges[1:])
            width = (edges[1] - edges[0]) if edges.size > 1 else 1.0
            plot.addItem(pg.BarGraphItem(x=centers, height=counts, width=width, brush="#5a9bd4"))
            # Histogram x-axis IS the Z value, so the selection region lines up
            # exactly with the distribution. The region's two edges are the
            # draggable Z bounds; it stays synced with the slider below.
            plot.setXRange(zmin, zmax, padding=0.02)
            self._region = pg.LinearRegionItem(values=[lo, hi], bounds=[zmin, zmax])
            self._region.setZValue(10)
            plot.addItem(self._region)
            self._region.sigRegionChanged.connect(self._on_region_changed)
            # Hover read-out: a dashed vertical marker + a label with the count
            # of the bar under the cursor, so the user can read exact counts and
            # decide the Z bounds.
            self._z_plot = plot
            self._z_cursor = pg.InfiniteLine(
                angle=90, movable=False,
                pen=pg.mkPen("#666", width=1, style=Qt.PenStyle.DashLine))
            self._z_cursor.setZValue(5)
            self._z_cursor.hide()
            plot.addItem(self._z_cursor, ignoreBounds=True)
            plot.scene().sigMouseMoved.connect(self._on_z_hover)
            root.addWidget(plot, 1)   # stretch → takes the dialog's extra height
            self._z_hover_label.setStyleSheet("color: #555; font-size: 11px;")
            root.addWidget(self._z_hover_label)
        except Exception:
            self._region = None

        from .render_window import DepthRangeSlider
        self._slider = DepthRangeSlider()
        self._slider.set_limits(zmin, zmax)
        self._slider.set_range(lo, hi)
        self._slider.rangeChanged.connect(self._on_slider_changed)

        self._all_z.setChecked(bool(init.z_all))
        z_row = QHBoxLayout()
        z_row.addWidget(self._all_z)
        z_row.addWidget(self._slider, 1)
        z_row.addWidget(self._z_label)
        root.addLayout(z_row)
        self._update_z_label()

    def _on_z_hover(self, pos) -> None:
        """Show the count of the histogram bar under the mouse (with a marker)."""
        plot = self._z_plot
        if plot is None or self._z_counts is None or self._z_edges is None:
            return
        vb = plot.getPlotItem().vb
        if not vb.sceneBoundingRect().contains(pos):
            self._z_cursor.hide()
            self._z_hover_label.setText(" ")
            return
        x = float(vb.mapSceneToView(pos).x())
        idx = int(np.searchsorted(self._z_edges, x) - 1)
        if idx < 0 or idx >= self._z_counts.size:
            self._z_cursor.hide()
            self._z_hover_label.setText(" ")
            return
        self._z_cursor.setPos(x)
        self._z_cursor.show()
        e0, e1 = float(self._z_edges[idx]), float(self._z_edges[idx + 1])
        self._z_hover_label.setText(
            f"Z ≈ {x:.1f} nm   ·   bin [{e0:.1f}, {e1:.1f}] nm   ·   "
            f"count = {int(self._z_counts[idx])}"
        )

    def _on_region_changed(self, *_args) -> None:
        if self._syncing or self._region is None or self._slider is None:
            return
        lo, hi = self._region.getRegion()
        self._syncing = True
        self._slider.set_range(lo, hi)
        self._syncing = False
        self._update_z_label()

    def _on_slider_changed(self, lo: float, hi: float) -> None:
        if self._syncing:
            return
        self._syncing = True
        if self._region is not None:
            self._region.setRegion((lo, hi))
        self._syncing = False
        self._update_z_label()

    def _update_z_label(self, *_args) -> None:
        if self._slider is not None:
            lo, hi = self._slider.range()
            self._z_label.setText(f"[{lo:.1f}, {hi:.1f}] nm")

    def _sync_enabled(self) -> None:
        # A region ROI always crops, so the crop options are always available;
        # only the Z slider follows the All-Z toggle.
        if self._has_z:
            active = not self._all_z.isChecked()
            if self._slider is not None:
                self._slider.setEnabled(active)
            if self._region is not None:
                self._region.setMovable(active)

    # ------------------------------------------------------------------
    def options(self) -> CropOptions:
        z_all = (not self._has_z) or self._all_z.isChecked()
        z_range = None
        if self._has_z and not z_all and self._slider is not None:
            z_range = tuple(self._slider.range())
        channels: list[int] = []
        if len(self._channels) > 1:
            channels = parse_channel_spec(self._channel_edit.text(), len(self._channels))
        return CropOptions(
            only_roi=True,                      # region ROI present ⇒ always crops
            exact_shape=self._exact_shape.isChecked(),
            clip=self._clip.isChecked(),
            channels=channels,
            z_all=z_all,
            z_range=z_range,
            spatial_filter=self._spatial.isChecked(),
            stop_asking=self._stop_asking.isChecked(),
        )
