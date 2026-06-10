"""
minflux_viewer.ui.attribute_window
=====================================
Attribute plot window — port of ``plot_attribute.m``.

Plots two numeric attributes against each other (X vs Y), with optional
colouring by a third attribute.  The most common usage is plotting one
attribute over time (X = ``tim``) or index (X = ``idx``).
"""

from __future__ import annotations

import numpy as np
import pyqtgraph as pg
from PyQt6.QtCore import Qt
from PyQt6.QtWidgets import (
    QCheckBox,
    QComboBox,
    QHBoxLayout,
    QLabel,
    QSizePolicy,
    QVBoxLayout,
    QWidget,
)

from ..core.app_state import AppState
from ..core.attributes import plot_attribute_names
from ..core.iteration import iteration_labels, ordinal, parse_iteration_label
from ..core.loader import attr_matches_last_valid, mfx_filter_mask, mfx_get
from ..core.roi_selection import rectangle_mask
from .plot_format import plot_widget

# Distinct per-iteration colours for the "all iterations" overlay.
_ITER_COLORS = [
    (31, 119, 180), (255, 127, 14), (44, 160, 44), (214, 39, 40),
    (148, 103, 189), (140, 86, 75), (227, 119, 194), (127, 127, 127),
    (188, 189, 34), (23, 190, 207),
]


def _iter_color(k: int) -> tuple[int, int, int]:
    return _ITER_COLORS[k % len(_ITER_COLORS)]


class AttributeWindow(QWidget):
    """Plot any two numeric attributes against each other."""

    TAG = "attribute_window"

    def __init__(self, state: AppState, parent: QWidget | None = None, *, dataset_idx: int | None = None) -> None:
        super().__init__(parent)
        self._state = state
        self._dataset_idx = dataset_idx if dataset_idx is not None else state.active_idx
        self._view_state_key = "attribute_plot_state"

        self.setWindowTitle("Attribute Plot")
        self.setWindowFlags(Qt.WindowType.Window)
        self.resize(700, 400)
        self.setAttribute(Qt.WidgetAttribute.WA_DeleteOnClose)
        self._roi_overlay = None

        self._build_ui()
        self._refresh()

        state.filter_changed.connect(self._on_filter_changed)

    @property
    def dataset_idx(self) -> int | None:
        return self._dataset_idx

    def _dataset(self):
        if self._dataset_idx is None:
            return None
        if not (0 <= self._dataset_idx < len(self._state.datasets)):
            return None
        return self._state.datasets[self._dataset_idx]

    # ------------------------------------------------------------------
    # UI
    # ------------------------------------------------------------------

    def _build_ui(self) -> None:
        root = QVBoxLayout(self)
        root.setContentsMargins(6, 6, 6, 6)
        root.setSpacing(4)

        # ── Control row ───────────────────────────────────────────
        bar = QHBoxLayout()
        bar.setSpacing(8)

        bar.addWidget(QLabel("X:"))
        self._x_combo = QComboBox()
        self._x_combo.setMinimumWidth(100)
        self._x_combo.currentTextChanged.connect(self._draw)
        bar.addWidget(self._x_combo)

        bar.addWidget(QLabel("Y:"))
        self._y_combo = QComboBox()
        self._y_combo.setMinimumWidth(100)
        self._y_combo.currentTextChanged.connect(self._draw)
        bar.addWidget(self._y_combo)

        self._iter_label = QLabel("Iter:")
        bar.addWidget(self._iter_label)
        self._iter_combo = QComboBox()
        self._iter_combo.setMinimumWidth(96)
        self._iter_combo.currentTextChanged.connect(self._draw)
        bar.addWidget(self._iter_combo)

        self._valid_chk = QCheckBox("Valid only")
        self._valid_chk.setChecked(True)
        self._valid_chk.setToolTip("Show only vld=True localizations. Uncheck to include invalid ones.")
        self._valid_chk.stateChanged.connect(self._draw)
        bar.addWidget(self._valid_chk)

        self._lines_chk = QCheckBox("Lines")
        self._lines_chk.setChecked(False)
        self._lines_chk.stateChanged.connect(self._draw)
        bar.addWidget(self._lines_chk)

        self._filter_chk = QCheckBox("Filtered only")
        self._filter_chk.setChecked(True)
        self._filter_chk.stateChanged.connect(self._draw)
        bar.addWidget(self._filter_chk)

        bar.addStretch()
        root.addLayout(bar)

        # ── Plot ─────────────────────────────────────────────────
        pg.setConfigOptions(antialias=True)
        self._plot = plot_widget(background="w")
        self._plot.setSizePolicy(
            QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Expanding
        )
        self._plot.showGrid(x=True, y=True, alpha=0.2)
        self._plot.setContextMenuPolicy(Qt.ContextMenuPolicy.NoContextMenu)
        self._plot.getPlotItem().setMenuEnabled(False)
        try:
            self._plot.getPlotItem().vb.setMenuEnabled(False)
        except Exception:
            pass

        # Series items are (curve, scatter) pairs, created on demand so the
        # "all iterations" overlay can show one coloured series per iteration.
        self._series_items: list[tuple] = []
        self._legend = None
        from .roi_overlay import RoiOverlayController
        self._roi_overlay = RoiOverlayController(
            self._state.rois,
            self,
            self._plot,
            self._plot.getPlotItem(),
            coordinate_space="plot",
        )

        root.addWidget(self._plot)

        # ── Info ─────────────────────────────────────────────────
        self._info = QLabel("")
        self._info.setStyleSheet("color: gray; font-size: 11px;")
        root.addWidget(self._info)

    # ------------------------------------------------------------------
    # Data
    # ------------------------------------------------------------------

    def _refresh(self) -> None:
        ds = self._dataset()
        if ds is None:
            self.setWindowTitle("Attribute Plot")
            self._clear_series()
            return

        self.setWindowTitle(f"Attribute Plot  —  {ds.name}")

        numeric = plot_attribute_names(ds, self._state.prefs)
        saved = ds.state.get(self._view_state_key, {})

        for combo in (self._x_combo, self._y_combo):
            old = combo.currentText()
            combo.blockSignals(True)
            combo.clear()
            combo.addItems(numeric)
            if old in numeric:
                combo.setCurrentText(old)
            combo.blockSignals(False)

        # Iteration dropdown: 1st … last (Nth) · all [flatten] · all [stacked]
        iter_opts = self._iter_labels(ds)
        self._iter_combo.blockSignals(True)
        self._iter_combo.clear()
        self._iter_combo.addItems(iter_opts)
        default_label = self._default_iter_label(ds)
        saved_label = str(saved.get("iter", "") or "")
        self._iter_combo.setCurrentText(saved_label if saved_label in iter_opts else default_label)
        self._iter_combo.blockSignals(False)
        has_iters = bool(iter_opts)
        self._iter_combo.setVisible(has_iters)
        self._iter_label.setVisible(has_iters)

        x_default = saved.get("x", "idx")
        y_default = saved.get("y", "efo")
        self._lines_chk.blockSignals(True)
        self._filter_chk.blockSignals(True)
        self._valid_chk.blockSignals(True)
        self._lines_chk.setChecked(bool(saved.get("lines", False)))
        self._filter_chk.setChecked(bool(saved.get("filtered_only", True)))
        self._valid_chk.setChecked(bool(saved.get("valid_only", True)))
        self._lines_chk.blockSignals(False)
        self._filter_chk.blockSignals(False)
        self._valid_chk.blockSignals(False)
        if x_default in numeric:
            self._x_combo.setCurrentText(x_default)
        elif "idx" in numeric:
            self._x_combo.setCurrentText("idx")
        elif self._x_combo.currentText() == "" and "tim" in numeric:
            self._x_combo.setCurrentText("tim")
        if y_default in numeric:
            self._y_combo.setCurrentText(y_default)
        elif "efo" in numeric:
            self._y_combo.setCurrentText("efo")

        self._draw()

    # ------------------------------------------------------------------
    # Iteration / validity helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _num_itr(ds) -> int:
        return max(1, int(ds.metadata.get("raw_num_itr", ds.prop.num_itr or 1)))

    def _iter_labels(self, ds) -> list[str]:
        return iteration_labels(self._num_itr(ds))

    def _default_iter_label(self, ds) -> str:
        labels = self._iter_labels(ds)
        return f"last ({ordinal(self._num_itr(ds))})" if labels else ""

    def _selection(self) -> tuple:
        """Return (itr_selector, render_mode) for the current label."""
        return parse_iteration_label(self._iter_combo.currentText())

    def _clear_series(self) -> None:
        for curve, scatter in self._series_items:
            self._plot.removeItem(curve)
            self._plot.removeItem(scatter)
        self._series_items = []
        if self._legend is not None:
            try:
                self._legend.scene().removeItem(self._legend)
            except Exception:
                pass
            self._legend = None
        # PlotItem caches its legend; clear it or a later addLegend() returns
        # the detached (invisible) old one.
        try:
            self._plot.getPlotItem().legend = None
        except Exception:
            pass

    def _add_series(self, x, y, color, *, use_lines: bool, name: str | None = None):
        # self._plot.plot(name=...) registers a legend sample reliably.
        curve = self._plot.plot(
            x if use_lines else [], y if use_lines else [],
            pen=pg.mkPen(color, width=1) if use_lines else None,
            name=name if use_lines else None,
        )
        scatter = pg.ScatterPlotItem(size=3, pen=None, brush=pg.mkBrush(*color))
        scatter.setData([] if use_lines else x, [] if use_lines else y)
        self._plot.addItem(scatter)
        self._series_items.append((curve, scatter))
        if name is not None and not use_lines and self._legend is not None:
            self._legend.addItem(scatter, name)

    def _axis_values(self, ds, name, sel, vld_only, default):
        """Values for one axis. ``idx`` is always the synthetic flattened index;
        other attributes use ds.attr in the default view, raw store otherwise."""
        if name == "idx":
            v = mfx_get(ds, "idx",
                        itr="last" if default else sel,
                        vld_only=True if default else vld_only)
        elif default:
            v = ds.attr.get(name)
        else:
            v = mfx_get(ds, name, itr=sel, vld_only=vld_only)
        return None if v is None else np.asarray(v).ravel().astype(float)

    def _series_data(self, ds, x_name, y_name, sel, vld_only, filtered_only):
        """Return downsampled (x, y, n_total, missing) for one iteration selector.

        Default selection (last + valid) uses the materialized ``ds.attr`` store
        so derived attributes (den, dst, ...) plot as before. Other selections
        use the raw store, where only raw attributes + ``idx`` are available.
        The ds.attr path is valid only when ds.attr is the last-valid
        materialization (``iter_load="last"``).
        """
        default = (sel == "last" and vld_only and attr_matches_last_valid(ds))
        missing: list[str] = []

        x = self._axis_values(ds, x_name, sel, vld_only, default)
        y = self._axis_values(ds, y_name, sel, vld_only, default)
        if x is None or y is None:
            return np.empty(0), np.empty(0), 0, [a for a, v in ((x_name, x), (y_name, y)) if v is None]

        # Keep idx aligned to the other axis if their lengths disagree
        # (only happens in non-default load modes).
        if x.size != y.size:
            if x_name == "idx" and y.size:
                x = np.arange(1, y.size + 1, dtype=float)
            elif y_name == "idx" and x.size:
                y = np.arange(1, x.size + 1, dtype=float)
        n = min(x.size, y.size)
        x, y = x[:n], y[:n]

        if filtered_only and n:
            if default:
                fmask = np.asarray(ds.filter_mask, dtype=bool).ravel()
                if fmask.shape[0] == n:
                    x, y = x[fmask], y[fmask]
                    n = x.size
            else:
                res = mfx_filter_mask(ds, itr=sel, vld_only=vld_only)
                if res is not None:
                    fmask, missing = res
                    if fmask.shape[0] == n:
                        x, y = x[fmask], y[fmask]
                        n = x.size

        if n == 0:
            return np.empty(0), np.empty(0), 0, missing
        if n > 50_000:
            step = n // 50_000
            return x[::step], y[::step], n, missing
        return x, y, n, missing

    def _draw(self) -> None:
        ds = self._dataset()
        if ds is None or self._x_combo.count() == 0:
            return

        x_name = self._x_combo.currentText()
        y_name = self._y_combo.currentText()
        if not x_name or not y_name:
            return

        itr_sel, render = self._selection()
        vld_only = self._valid_chk.isChecked()
        filtered_only = self._filter_chk.isChecked()
        use_lines = self._lines_chk.isChecked()

        ds.state[self._view_state_key] = {
            "x": x_name,
            "y": y_name,
            "lines": use_lines,
            "filtered_only": filtered_only,
            "iter": self._iter_combo.currentText() or "",
            "valid_only": vld_only,
        }

        self._clear_series()

        if render == "stacked":
            n_itr = self._num_itr(ds)
            self._legend = self._plot.addLegend(offset=(-10, 10))
            total = 0
            missing: list[str] = []
            for k in range(n_itr):
                x, y, n, miss = self._series_data(ds, x_name, y_name, k, vld_only, filtered_only)
                missing = miss or missing
                total += n
                if n:
                    self._add_series(x, y, _iter_color(k), use_lines=use_lines, name=ordinal(k + 1))
            note = f"{total:,} points across {n_itr} iterations  |  all [stacked]"
        else:
            # "single" (last / k-th) and "flatten" are one blue series.
            sel = "all" if render == "flatten" else itr_sel
            x, y, n, missing = self._series_data(ds, x_name, y_name, sel, vld_only, filtered_only)
            if n:
                self._add_series(x, y, (70, 130, 180), use_lines=use_lines)
            sel_label = self._iter_combo.currentText() or "last"
            note = f"{n:,} points  |  {sel_label}"

        self._plot.setLabel("bottom", x_name)
        self._plot.setLabel("left", y_name)
        if missing:
            axis_missing = [a for a in missing if a in (x_name, y_name)]
            filter_missing = [a for a in missing if a not in (x_name, y_name)]
            if axis_missing:
                note += f"  |  {', '.join(axis_missing)} has no per-iteration values"
            if filter_missing:
                note += f"  |  filter on {', '.join(filter_missing)} not applied"
        if not vld_only:
            note += "  |  incl. invalid"
        self._info.setText(note)

    def compute_roi_selection(self, record):
        if record.type != "rectangle":
            return None
        ds = self._dataset()
        if ds is None:
            return None
        # ROI selection maps back onto the materialized (last-iteration, valid)
        # store. It cannot be mapped when the view shows a different iteration
        # or includes invalid localizations.
        itr_sel, render = self._selection()
        if (itr_sel != "last" or render != "single"
                or not self._valid_chk.isChecked()
                or not attr_matches_last_valid(ds)):
            return None
        x_name = self._x_combo.currentText()
        y_name = self._y_combo.currentText()
        if not x_name or not y_name:
            return None
        x = np.asarray(ds.attr.get(x_name, np.empty(0))).ravel().astype(float)
        y = np.asarray(ds.attr.get(y_name, np.empty(0))).ravel().astype(float)
        n = min(x.size, y.size, ds.prop.num_loc)
        if n == 0:
            return None
        base = np.ones(n, dtype=bool)
        if self._filter_chk.isChecked():
            ftr = np.asarray(ds.filter_mask, dtype=bool).ravel()
            if ftr.size == n:
                base = ftr.copy()
        mask = rectangle_mask(x[:n], y[:n], record, base_mask=base)
        if mask.size != ds.prop.num_loc:
            full = np.zeros(ds.prop.num_loc, dtype=bool)
            full[:mask.size] = mask
            mask = full
        context = {
            "source_view": "attribute",
            "dataset_idx": self._dataset_idx,
            "x_attr": x_name,
            "y_attr": y_name,
            "filtered_only": self._filter_chk.isChecked(),
        }
        return ds, mask, context

    # ------------------------------------------------------------------
    # Slots
    # ------------------------------------------------------------------

    def _on_filter_changed(self, idx: int) -> None:
        if idx == self._dataset_idx:
            self._draw()

    def focusInEvent(self, event) -> None:
        if self._dataset_idx is not None and 0 <= self._dataset_idx < len(self._state.datasets):
            self._state.set_active(self._dataset_idx)
        if self._roi_overlay is not None:
            self._roi_overlay.activate()
        super().focusInEvent(event)

    def changeEvent(self, event) -> None:
        from PyQt6.QtCore import QEvent
        if event.type() == QEvent.Type.ActivationChange and self.isActiveWindow():
            if self._dataset_idx is not None and 0 <= self._dataset_idx < len(self._state.datasets):
                self._state.set_active(self._dataset_idx)
            if self._roi_overlay is not None:
                self._roi_overlay.activate()
        super().changeEvent(event)
