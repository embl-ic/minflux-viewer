"""
minflux_viewer.ui.width_profile_window
=======================================
Result window for **filament width measurement** (FWHM of the perpendicular
intensity profile), launched from the Curvilinear Structure Segmentation tool.

Shows the average image intensity vs perpendicular distance to the traced centre
lines, with the half-maximum level and the two FWHM crossings marked, plus the
combined FWHM and the per-segment FWHM distribution. Modeless / non-owned.
"""

from __future__ import annotations

import numpy as np
import pyqtgraph as pg
from PyQt6.QtCore import Qt
from PyQt6.QtWidgets import QDialog, QHBoxLayout, QLabel, QPushButton, QVBoxLayout

from ..analysis.curvilinear_segmentation import fwhm_bounds
from .text_select import make_labels_selectable

_PROFILE_COLOR = (0, 229, 255)
_HALF_COLOR = (255, 235, 0)


class WidthProfileWindow(QDialog):
    """Perpendicular intensity profile + FWHM for traced filament centre lines."""

    def __init__(self, result: dict, *, title: str = "", owner=None) -> None:
        super().__init__(None)
        self._owner = owner
        self.setWindowTitle("Filament Width (FWHM)")
        self.resize(640, 480)

        offsets = np.asarray(result.get("offsets", []), dtype=float)
        profile = np.asarray(result.get("profile", []), dtype=float)
        combined_fwhm = float(result.get("fwhm", float("nan")))
        per_fwhm = np.asarray(result.get("per_path_fwhm", []), dtype=float)
        n_seg = int(np.asarray(result.get("vertex_counts", [])).size)

        root = QVBoxLayout(self)
        head = QLabel(f"<b>{title}</b>" if title else "<b>Filament width</b>")
        head.setWordWrap(True)
        root.addWidget(head)

        plot = pg.PlotWidget()
        plot.setLabel("bottom", "Perpendicular distance (nm)")
        plot.setLabel("left", "Mean intensity")
        plot.showGrid(x=True, y=True, alpha=0.15)
        if offsets.size and profile.size:
            plot.addItem(pg.PlotDataItem(
                offsets, profile, pen=pg.mkPen(_PROFILE_COLOR, width=2),
                fillLevel=0.0, brush=(0, 229, 255, 40)))
            left, right, half = fwhm_bounds(offsets, profile)
            if np.isfinite(half):
                plot.addItem(pg.InfiniteLine(
                    pos=half, angle=0, pen=pg.mkPen(_HALF_COLOR, width=1,
                                                    style=Qt.PenStyle.DashLine)))
            for b in (left, right):
                if np.isfinite(b):
                    plot.addItem(pg.InfiniteLine(
                        pos=b, angle=90, pen=pg.mkPen(_HALF_COLOR, width=1,
                                                      style=Qt.PenStyle.DashLine)))
        root.addWidget(plot, 1)

        valid = per_fwhm[np.isfinite(per_fwhm)]
        lines = [f"Combined FWHM = <b>{combined_fwhm:.1f} nm</b>"
                 if np.isfinite(combined_fwhm) else "Combined FWHM = n/a",
                 f"Segments measured: {n_seg}"]
        if valid.size:
            lines.append(f"Per-segment FWHM: mean {valid.mean():.1f} ± "
                         f"{valid.std():.1f} nm  (median {np.median(valid):.1f}, "
                         f"n={valid.size})")
        stats = QLabel("<br>".join(lines))
        stats.setTextFormat(Qt.TextFormat.RichText)
        root.addWidget(stats)

        btns = QHBoxLayout()
        btns.addStretch(1)
        close_btn = QPushButton("Close")
        close_btn.clicked.connect(self.close)
        btns.addWidget(close_btn)
        root.addLayout(btns)

        make_labels_selectable(self)
