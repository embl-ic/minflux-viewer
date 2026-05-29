"""
minflux_viewer.ui.trace_viewer
================================
Trace Viewer — scatter plot of localisations coloured by trace ID (tid).

Each unique trace ID receives a distinct cyclic colour so individual
traces are visually separated.  Controls:

* **Orientation** dropdown — XY (default), XZ, YZ
* **Point size** spin box
* **Colour by** — trace ID (default) or any available dataset attribute
* **Z slider** — shown for 3-D datasets (same semantics as render window)

The window pins to one dataset at load time (Fiji-style).
"""

from __future__ import annotations

import numpy as np
import pyqtgraph as pg
from PyQt6.QtCore import Qt
from PyQt6.QtGui import QIcon
from PyQt6.QtWidgets import (
    QComboBox,
    QDoubleSpinBox,
    QHBoxLayout,
    QLabel,
    QSizePolicy,
    QVBoxLayout,
    QWidget,
)

from .. import resource_path
from ..core.app_state import AppState


# ---------------------------------------------------------------------------
# Cyclic colour palette for trace IDs (12 high-contrast colours)
# ---------------------------------------------------------------------------

_PALETTE = np.array([
    [31,  119, 180, 200],
    [255, 127,  14, 200],
    [44,  160,  44, 200],
    [214,  39,  40, 200],
    [148, 103, 189, 200],
    [140,  86,  75, 200],
    [227, 119, 194, 200],
    [127, 127, 127, 200],
    [188, 189,  34, 200],
    [ 23, 190, 207, 200],
    [ 82, 109, 130, 200],
    [255, 187,  90, 200],
], dtype=np.ubyte)


def _tid_colours(tid: np.ndarray) -> np.ndarray:
    """Map trace IDs to RGBA colours cycling through _PALETTE."""
    uid, inverse = np.unique(tid, return_inverse=True)
    colours = _PALETTE[inverse % len(_PALETTE)]
    return colours


class TraceViewer(QWidget):
    """Interactive scatter plot coloured by trace ID."""

    TAG = "trace_viewer"

    def __init__(
        self,
        state: AppState,
        dataset_idx: int | None = None,
        parent: QWidget | None = None,
    ) -> None:
        super().__init__(parent)
        self._state = state
        self._idx = dataset_idx if dataset_idx is not None else state.active_idx
        self._orientation = "XY"

        self.setWindowTitle("Trace Viewer")
        self.setWindowIcon(QIcon(str(resource_path("icons", "minflux_viewer_logo.png"))))
        self.setWindowFlags(Qt.WindowType.Window)
        self.resize(700, 720)
        self.setAttribute(Qt.WidgetAttribute.WA_DeleteOnClose)

        self._build_ui()
        self._refresh()

        state.filter_changed.connect(self._on_filter_changed)

    # ------------------------------------------------------------------
    # UI construction
    # ------------------------------------------------------------------

    def _build_ui(self) -> None:
        root = QVBoxLayout(self)
        root.setContentsMargins(6, 6, 6, 6)
        root.setSpacing(4)

        # Controls row
        ctrl = QHBoxLayout()

        ctrl.addWidget(QLabel("Orientation:"))
        self._orientation_combo = QComboBox()
        self._orientation_combo.addItems(["XY", "XZ", "YZ"])
        self._orientation_combo.currentTextChanged.connect(self._on_orientation)
        ctrl.addWidget(self._orientation_combo)

        ctrl.addSpacing(12)
        ctrl.addWidget(QLabel("Point size:"))
        self._size_spin = QDoubleSpinBox()
        self._size_spin.setRange(0.5, 20.0)
        self._size_spin.setSingleStep(0.5)
        self._size_spin.setValue(3.0)
        self._size_spin.setDecimals(1)
        self._size_spin.valueChanged.connect(self._on_size_changed)
        ctrl.addWidget(self._size_spin)

        ctrl.addStretch()
        root.addLayout(ctrl)

        # pyqtgraph scatter plot
        pg.setConfigOptions(antialias=True, imageAxisOrder="row-major")
        self._plot = pg.PlotWidget()
        self._plot.setBackground("k")
        self._plot.setAspectLocked(True)
        self._plot.showGrid(x=True, y=True, alpha=0.2)
        self._scatter = pg.ScatterPlotItem(
            size=3.0,
            pxMode=True,
            pen=None,
        )
        self._plot.addItem(self._scatter)
        self._plot.setSizePolicy(
            QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Expanding
        )
        root.addWidget(self._plot, stretch=1)

        # Info label
        self._info = QLabel("")
        self._info.setStyleSheet("color: gray; font-size: 11px;")
        root.addWidget(self._info)

    # ------------------------------------------------------------------
    # Data refresh
    # ------------------------------------------------------------------

    def _refresh(self) -> None:
        ds = self._state.datasets[self._idx] if self._idx is not None else None
        if ds is None:
            self._scatter.setData([])
            self._info.setText("No dataset.")
            return

        self.setWindowTitle(f"Traces — {ds.name}")

        try:
            loc = np.asarray(ds.loc_nm, dtype=float)
        except Exception:
            self._scatter.setData([])
            self._info.setText("Cannot read localisations.")
            return

        if loc.ndim != 2 or loc.shape[0] == 0:
            self._scatter.setData([])
            self._info.setText("No localisations.")
            return

        mask = np.asarray(ds.filter_mask, dtype=bool)
        if mask.shape[0] == loc.shape[0]:
            loc = loc[mask]

        if loc.shape[0] == 0:
            self._scatter.setData([])
            self._info.setText("No localisations pass the current filter.")
            return

        # Pad to (N,3) if 2-D data
        if loc.shape[1] == 2:
            loc = np.column_stack([loc, np.zeros(loc.shape[0])])

        # Orientation
        o = self._orientation
        if o == "XY":
            x, y = loc[:, 0], loc[:, 1]
            xlabel, ylabel = "x (nm)", "y (nm)"
        elif o == "XZ":
            x, y = loc[:, 0], loc[:, 2]
            xlabel, ylabel = "x (nm)", "z (nm)"
        else:
            x, y = loc[:, 1], loc[:, 2]
            xlabel, ylabel = "y (nm)", "z (nm)"

        self._plot.setLabel("bottom", xlabel)
        self._plot.setLabel("left", ylabel)

        # Colours by trace ID
        tid_raw = ds.attr.get("tid")
        if tid_raw is not None:
            tid = np.asarray(tid_raw, dtype=np.int64).ravel()
            if mask.shape[0] == len(tid):
                tid = tid[mask]
            colours = _tid_colours(tid)
            n_traces = int(np.unique(tid[tid != 0]).size)
        else:
            grey = np.array([[180, 180, 180, 180]], dtype=np.ubyte)
            colours = np.repeat(grey, len(x), axis=0)
            n_traces = 0

        size = self._size_spin.value()
        self._scatter.setData(
            x=x, y=y,
            brush=[pg.mkBrush(*c) for c in colours],
            size=size,
        )

        tid_info = f"{n_traces} traces  |  " if n_traces else ""
        self._info.setText(
            f"{tid_info}{len(x):,} localisations  |  {o}"
        )

    # ------------------------------------------------------------------
    # Handlers
    # ------------------------------------------------------------------

    def _on_orientation(self, text: str) -> None:
        self._orientation = text
        self._refresh()

    def _on_size_changed(self, val: float) -> None:
        self._scatter.setSize(val)

    def _on_filter_changed(self, idx: int) -> None:
        if idx == self._idx:
            self._refresh()
