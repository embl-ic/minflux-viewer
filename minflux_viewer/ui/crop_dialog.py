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
    QVBoxLayout,
)

from ..core.roi_crop import CropOptions, parse_channel_spec


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

        root = QVBoxLayout(self)
        root.addWidget(QLabel(f"Dataset: <b>{dataset_name}</b>"))

        # With an active region ROI the duplicate always crops; this chooses the
        # ROI's exact outline vs its (axis-aligned) bounding box.
        self._exact_shape = QCheckBox("duplicate only the ROI region (exact shape; else bounding box)")
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

    # ------------------------------------------------------------------
    def _build_z_section(self, root, z: np.ndarray, init: CropOptions) -> None:
        z = z[np.isfinite(z)]
        if z.size == 0:
            self._has_z = False
            return
        zmin, zmax = float(z.min()), float(z.max())
        if zmax <= zmin:
            zmax = zmin + 1.0
        lo, hi = (init.z_range or (zmin, zmax))

        root.addWidget(QLabel("Z distribution (drag the shaded range, or the slider):"))
        try:
            import pyqtgraph as pg
            plot = pg.PlotWidget()
            plot.setMaximumHeight(110)
            plot.setMouseEnabled(x=False, y=False)
            plot.hideAxis("left")
            plot.setLabel("bottom", "Z (nm)")
            counts, edges = np.histogram(z, bins=min(64, max(8, z.size // 20)))
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
            root.addWidget(plot)
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
