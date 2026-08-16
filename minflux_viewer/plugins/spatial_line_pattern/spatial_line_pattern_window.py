"""Modeless UI for directed repeating-pattern analysis along a line ROI."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pyqtgraph as pg
from PyQt6.QtCore import QPointF, QRectF, Qt, QTimer
from PyQt6.QtGui import QPainterPath, QPolygonF
from PyQt6.QtWidgets import (
    QApplication,
    QCheckBox,
    QComboBox,
    QDialog,
    QDoubleSpinBox,
    QFileDialog,
    QFormLayout,
    QGraphicsPathItem,
    QHBoxLayout,
    QLabel,
    QMessageBox,
    QPushButton,
    QScrollArea,
    QSpinBox,
    QSplitter,
    QStyle,
    QTabWidget,
    QVBoxLayout,
    QWidget,
)

from ...colormaps import colormap_lut
from ...analysis.plot_profile import band_polygon
from ...analysis.spatial_line_pattern import (
    DEFAULT_BACKGROUND_SCALE_NM,
    DEFAULT_HALF_WIDTH_NM,
    DEFAULT_INTERPOLATION_STEP_NM,
    DEFAULT_MAX_PERIOD_NM,
    DEFAULT_MIN_PERIOD_NM,
    DEFAULT_PEAK_ORDER,
    DEFAULT_PEAK_PROMINENCE,
    DEFAULT_PROFILE_BIN_NM,
    DEFAULT_PROFILE_SMOOTHING_NM,
    DEFAULT_TRANSVERSE_BIN_NM,
    SpatialLinePatternResult,
    analyze_spatial_line_pattern,
)
from ...core.roi_crop import plane_localizations, plane_localizations_version

_LINE_TYPES = {"line", "polyline", "freehand_line"}
_POLL_MS = 150
_TOTAL_COLOR = "#f2f2f2"
_POSITIVE_COLOR = "#00b8d9"
_NEGATIVE_COLOR = "#e754b7"
_CENTROID_COLOR = "#ff9f43"
_AUTOCORR_COLOR = "#8dd35f"


def _finite_text(value: float, suffix: str = " nm") -> str:
    return f"{value:.2f}{suffix}" if np.isfinite(value) else "not resolved"


def _normalized(values: np.ndarray) -> np.ndarray:
    values = np.asarray(values, dtype=float)
    maximum = float(np.max(values)) if values.size else 0.0
    return values / maximum if maximum > 0.0 else np.zeros(values.shape)


class SpatialLinePatternWindow(QDialog):
    """Straighten and analyze one active dataset around one directed line ROI."""

    def __init__(self, state, dataset_idx: int, view, owner=None) -> None:
        super().__init__(None)
        self._state = state
        self._dataset = state.datasets[dataset_idx]
        self._view = view
        self._owner = owner
        self._result: SpatialLinePatternResult | None = None
        self._last_signature = None
        self._data_token = None
        self._localizations = np.zeros((0, 2), dtype=float)
        self._curve_item: QGraphicsPathItem | None = None
        self._band_item: QGraphicsPathItem | None = None

        self.setWindowTitle("Spatial Pattern Analysis along Line Profile")
        self.resize(1180, 760)
        self._build_ui()

        self._timer = QTimer(self)
        self._timer.setInterval(_POLL_MS)
        self._timer.timeout.connect(self._tick)
        self._timer.start()
        self._tick()

    # ------------------------------------------------------------------ UI
    def _build_ui(self) -> None:
        root = QVBoxLayout(self)
        root.setContentsMargins(8, 8, 8, 8)
        root.setSpacing(6)

        self._summary = QLabel("")
        self._summary.setWordWrap(True)
        self._summary.setTextInteractionFlags(
            Qt.TextInteractionFlag.TextSelectableByMouse
        )
        root.addWidget(self._summary)

        splitter = QSplitter(Qt.Orientation.Horizontal)
        root.addWidget(splitter, 1)

        controls_host = QWidget()
        controls_layout = QVBoxLayout(controls_host)
        controls_layout.setContentsMargins(4, 4, 8, 4)
        form = QFormLayout()
        form.setFieldGrowthPolicy(QFormLayout.FieldGrowthPolicy.AllNonFixedFieldsGrow)

        self._model = QComboBox()
        self._model.addItem("Cubic spline", "cubic")
        self._model.addItem("Polyline", "polyline")
        form.addRow("Centerline:", self._model)

        self._spline_smoothing = self._double_spin(
            0.0,
            10_000.0,
            3.0,
            decimals=2,
            step=1.0,
            suffix=" nm",
            tooltip=(
                "Approximate RMS smoothing allowance at ROI vertices. Zero makes "
                "the spline interpolate every vertex."
            ),
        )
        form.addRow("Spline smoothing:", self._spline_smoothing)

        self._interpolation_step = self._double_spin(
            0.05,
            10_000.0,
            DEFAULT_INTERPOLATION_STEP_NM,
            decimals=2,
            step=0.5,
            suffix=" nm",
            tooltip="Sampling step of the fitted centerline used for projection.",
        )
        form.addRow("Interpolation step:", self._interpolation_step)

        self._half_width = self._double_spin(
            0.1,
            1_000_000.0,
            DEFAULT_HALF_WIDTH_NM,
            decimals=1,
            step=5.0,
            suffix=" nm",
            tooltip="One-sided distance from the centerline included in the analysis.",
        )
        form.addRow("One-sided width:", self._half_width)

        self._profile_bin = self._double_spin(
            0.05,
            100_000.0,
            DEFAULT_PROFILE_BIN_NM,
            decimals=2,
            step=1.0,
            suffix=" nm",
            tooltip="Longitudinal summary distance for profiles and periodicity.",
        )
        form.addRow("Profile bin:", self._profile_bin)

        self._transverse_bin = self._double_spin(
            0.05,
            100_000.0,
            DEFAULT_TRANSVERSE_BIN_NM,
            decimals=2,
            step=0.5,
            suffix=" nm",
            tooltip="Signed perpendicular bin size of the straightened map.",
        )
        form.addRow("Transverse bin:", self._transverse_bin)

        self._profile_smoothing = self._double_spin(
            0.0,
            100_000.0,
            DEFAULT_PROFILE_SMOOTHING_NM,
            decimals=2,
            step=1.0,
            suffix=" nm",
            tooltip="Gaussian smoothing scale applied before peak and period analysis.",
        )
        form.addRow("Profile smoothing:", self._profile_smoothing)

        self._background_scale = self._double_spin(
            0.0,
            1_000_000.0,
            DEFAULT_BACKGROUND_SCALE_NM,
            decimals=1,
            step=10.0,
            suffix=" nm",
            tooltip="Gaussian scale removed as slowly varying longitudinal background.",
        )
        form.addRow("Background scale:", self._background_scale)

        self._min_period = self._double_spin(
            0.1,
            1_000_000.0,
            DEFAULT_MIN_PERIOD_NM,
            decimals=1,
            step=5.0,
            suffix=" nm",
        )
        form.addRow("Minimum period:", self._min_period)

        self._max_period = self._double_spin(
            0.1,
            10_000_000.0,
            DEFAULT_MAX_PERIOD_NM,
            decimals=1,
            step=10.0,
            suffix=" nm",
        )
        form.addRow("Maximum period:", self._max_period)

        self._prominence = self._double_spin(
            0.0,
            1.0,
            DEFAULT_PEAK_PROMINENCE,
            decimals=2,
            step=0.05,
            tooltip="Peak prominence as a fraction of the detrended profile range.",
        )
        form.addRow("Peak prominence:", self._prominence)

        self._peak_order = QSpinBox()
        self._peak_order.setRange(1, 20)
        self._peak_order.setValue(DEFAULT_PEAK_ORDER)
        self._peak_order.setToolTip(
            "Maximum neighbor order for peak-to-peak spacing differences."
        )
        form.addRow("Peak-spacing order:", self._peak_order)

        self._flip_side = QCheckBox("Flip signed sides")
        self._flip_side.setToolTip(
            "Reverse the sign of the local perpendicular coordinate without "
            "changing the line direction."
        )
        form.addRow("Side convention:", self._flip_side)
        controls_layout.addLayout(form)

        refresh = QPushButton("Recompute")
        refresh.setIcon(
            self.style().standardIcon(QStyle.StandardPixmap.SP_BrowserReload)
        )
        refresh.clicked.connect(self._invalidate)
        controls_layout.addWidget(refresh)
        controls_layout.addStretch(1)

        scroll = QScrollArea()
        scroll.setWidgetResizable(True)
        scroll.setFrameShape(QScrollArea.Shape.NoFrame)
        scroll.setWidget(controls_host)
        scroll.setMinimumWidth(270)
        scroll.setMaximumWidth(340)
        splitter.addWidget(scroll)

        self._tabs = QTabWidget()
        self._build_straightened_tab()
        self._build_profiles_tab()
        self._build_periodicity_tab()
        splitter.addWidget(self._tabs)
        splitter.setStretchFactor(1, 1)

        export_row = QHBoxLayout()
        export_row.addStretch(1)
        save_csv = QPushButton("Save CSV...")
        save_csv.clicked.connect(self._save_csv)
        export_row.addWidget(save_csv)
        save_npz = QPushButton("Save NPZ...")
        save_npz.clicked.connect(self._save_npz)
        export_row.addWidget(save_npz)
        close = QPushButton("Close")
        close.clicked.connect(self.close)
        export_row.addWidget(close)
        root.addLayout(export_row)

        for control in (
            self._model,
            self._spline_smoothing,
            self._interpolation_step,
            self._half_width,
            self._profile_bin,
            self._transverse_bin,
            self._profile_smoothing,
            self._background_scale,
            self._min_period,
            self._max_period,
            self._prominence,
            self._peak_order,
            self._flip_side,
        ):
            signal = getattr(control, "valueChanged", None)
            if signal is None:
                signal = getattr(control, "currentIndexChanged", None)
            if signal is None:
                signal = getattr(control, "toggled", None)
            signal.connect(self._invalidate)

    def _double_spin(
        self,
        minimum: float,
        maximum: float,
        value: float,
        *,
        decimals: int,
        step: float,
        suffix: str = "",
        tooltip: str = "",
    ) -> QDoubleSpinBox:
        spin = QDoubleSpinBox()
        spin.setRange(minimum, maximum)
        spin.setDecimals(decimals)
        spin.setSingleStep(step)
        spin.setValue(value)
        spin.setSuffix(suffix)
        if tooltip:
            spin.setToolTip(tooltip)
        return spin

    def _plot_widget(self, bottom: str, left: str) -> pg.PlotWidget:
        plot = pg.PlotWidget(background="#202124")
        plot.showGrid(x=True, y=True, alpha=0.2)
        plot.setLabel("bottom", bottom, units="nm")
        plot.setLabel("left", left)
        return plot

    def _build_straightened_tab(self) -> None:
        tab = QWidget()
        layout = QVBoxLayout(tab)
        self._map_plot = self._plot_widget("directed distance", "signed offset")
        self._map_image = pg.ImageItem(axisOrder="row-major")
        try:
            lut = colormap_lut("viridis", alpha=False)
            self._map_image.setLookupTable(lut)
        except Exception:
            pass
        self._map_plot.addItem(self._map_image)
        self._map_zero = pg.InfiniteLine(
            pos=0.0,
            angle=0,
            pen=pg.mkPen("#f0f0f0", width=1, style=Qt.PenStyle.DashLine),
        )
        self._map_plot.addItem(self._map_zero)
        layout.addWidget(self._map_plot)
        self._tabs.addTab(tab, "Straightened Map")

    def _build_profiles_tab(self) -> None:
        tab = QWidget()
        layout = QVBoxLayout(tab)
        self._profile_plot = self._plot_widget("directed distance", "localizations / bin")
        self._profile_plot.addLegend(offset=(8, 8))
        self._total_curve = self._profile_plot.plot(
            pen=pg.mkPen(_TOTAL_COLOR, width=2.0),
            name="total (smoothed)",
        )
        self._positive_curve = self._profile_plot.plot(
            pen=pg.mkPen(_POSITIVE_COLOR, width=1.5),
            name="+ side",
        )
        self._negative_curve = self._profile_plot.plot(
            pen=pg.mkPen(_NEGATIVE_COLOR, width=1.5),
            name="- side",
        )
        self._peak_points = pg.ScatterPlotItem(
            size=8,
            brush=pg.mkBrush("#ffdd55"),
            pen=pg.mkPen("#202124"),
        )
        self._profile_plot.addItem(self._peak_points)
        layout.addWidget(self._profile_plot, 2)

        self._centroid_plot = self._plot_widget(
            "directed distance",
            "transverse centroid (nm)",
        )
        self._centroid_curve = self._centroid_plot.plot(
            pen=pg.mkPen(_CENTROID_COLOR, width=2.0)
        )
        self._centroid_plot.addItem(
            pg.InfiniteLine(
                pos=0.0,
                angle=0,
                pen=pg.mkPen("#888888", style=Qt.PenStyle.DashLine),
            )
        )
        layout.addWidget(self._centroid_plot, 1)
        self._tabs.addTab(tab, "Profiles")

    def _build_periodicity_tab(self) -> None:
        tab = QWidget()
        layout = QVBoxLayout(tab)
        self._spectrum_plot = self._plot_widget("period", "normalized power")
        self._spectrum_plot.addLegend(offset=(8, 8))
        self._density_power_curve = self._spectrum_plot.plot(
            pen=pg.mkPen(_POSITIVE_COLOR, width=2.0),
            name="longitudinal density",
        )
        self._transverse_power_curve = self._spectrum_plot.plot(
            pen=pg.mkPen(_CENTROID_COLOR, width=2.0),
            name="transverse centroid",
        )
        self._density_period_line = pg.InfiniteLine(
            angle=90,
            movable=False,
            pen=pg.mkPen(_POSITIVE_COLOR, width=1, style=Qt.PenStyle.DashLine),
        )
        self._transverse_period_line = pg.InfiniteLine(
            angle=90,
            movable=False,
            pen=pg.mkPen(_CENTROID_COLOR, width=1, style=Qt.PenStyle.DashLine),
        )
        self._spectrum_plot.addItem(self._density_period_line)
        self._spectrum_plot.addItem(self._transverse_period_line)
        layout.addWidget(self._spectrum_plot, 1)

        self._autocorr_plot = self._plot_widget("lag", "normalized autocorrelation")
        self._autocorr_curve = self._autocorr_plot.plot(
            pen=pg.mkPen(_AUTOCORR_COLOR, width=2.0)
        )
        self._autocorr_period_line = pg.InfiniteLine(
            angle=90,
            movable=False,
            pen=pg.mkPen(_AUTOCORR_COLOR, width=1, style=Qt.PenStyle.DashLine),
        )
        self._autocorr_plot.addItem(self._autocorr_period_line)
        layout.addWidget(self._autocorr_plot, 1)
        self._tabs.addTab(tab, "Periodicity")

    # -------------------------------------------------------------- analysis
    def _dataset_index(self) -> int | None:
        for index, dataset in enumerate(self._state.datasets):
            if dataset is self._dataset:
                return index
        return None

    def _alive(self) -> bool:
        try:
            return self._view.coordinate_view_box() is not None
        except RuntimeError:
            return False

    def _active_line(self):
        overlay = getattr(self._view, "_roi_overlay", None)
        if overlay is None:
            return None, None
        record = overlay.active_open_line_record()
        if record is None or record.type not in _LINE_TYPES:
            return None, None
        idx = self._dataset_index()
        context = getattr(record, "context", {}) or {}
        target_idx = context.get("dataset_idx")
        if target_idx is not None and idx is not None and int(target_idx) != idx:
            return None, None
        plane = self._view.roi_view_plane()
        source_plane = context.get("view_plane")
        source_points = record.geometry.get("points", [])
        is_3d = bool(source_points) and all(len(point) >= 3 for point in source_points)
        if source_plane and source_plane != plane and not is_3d:
            return None, None
        points = overlay.active_open_line_points()
        points = np.asarray(points if points is not None else [], dtype=float)
        if points.ndim != 2 or points.shape[0] < 2:
            return None, None
        return record, points[:, :2]

    def _localization_token(self, idx: int, plane: str):
        channel = [{"dataset_idx": idx, "visible": True, "kind": "localizations"}]
        token = plane_localizations_version(self._state, channel, plane)
        transform = (
            self._dataset.state.get("overlay_transform")
            or self._dataset.state.get("render_transform_2d")
        )
        transform_token = (
            tuple(np.asarray(transform, dtype=float).ravel())
            if transform is not None
            else None
        )
        return token, transform_token

    def _parameters(self) -> dict:
        return {
            "centerline_model": self._model.currentData(),
            "interpolation_step_nm": self._interpolation_step.value(),
            "spline_smoothing_nm": self._spline_smoothing.value(),
            "half_width_nm": self._half_width.value(),
            "profile_bin_nm": self._profile_bin.value(),
            "transverse_bin_nm": self._transverse_bin.value(),
            "profile_smoothing_nm": self._profile_smoothing.value(),
            "background_scale_nm": self._background_scale.value(),
            "min_period_nm": self._min_period.value(),
            "max_period_nm": self._max_period.value(),
            "peak_prominence": self._prominence.value(),
            "peak_order": self._peak_order.value(),
            "flip_side": self._flip_side.isChecked(),
        }

    def _invalidate(self, *_args) -> None:
        self._last_signature = None
        self._tick()

    def _tick(self) -> None:
        if not self._alive():
            self._timer.stop()
            self._clear("The source coordinate view was closed.")
            return
        idx = self._dataset_index()
        if idx is None:
            self._timer.stop()
            self._clear("The analyzed dataset was closed.")
            return
        record, points = self._active_line()
        if record is None:
            self._clear(
                "Select one line, polyline, or freehand-line ROI for this dataset."
            )
            self._hide_source_overlay()
            self._last_signature = None
            return
        plane = self._view.roi_view_plane()
        data_token = self._localization_token(idx, plane)
        if data_token != self._data_token:
            channel = [
                {"dataset_idx": idx, "visible": True, "kind": "localizations"}
            ]
            self._localizations = plane_localizations(
                self._state,
                channel,
                plane,
            )
            self._data_token = data_token

        parameters = self._parameters()
        parameter_token = tuple(parameters.items())
        signature = (
            record.id,
            plane,
            record.stroke_color,
            np.round(points, 3).tobytes(),
            data_token,
            parameter_token,
        )
        if signature == self._last_signature:
            return
        self._last_signature = signature

        QApplication.setOverrideCursor(Qt.CursorShape.WaitCursor)
        try:
            result = analyze_spatial_line_pattern(
                self._localizations,
                points,
                **parameters,
            )
        except Exception as exc:
            self._result = None
            self._clear(f"Analysis failed: {exc}")
            self._state.log(
                f"Spatial line-pattern analysis failed: {exc}",
                level="ERROR",
                dataset_idx=idx,
            )
            return
        finally:
            QApplication.restoreOverrideCursor()

        self._result = result
        self._draw_result(result)
        self._update_source_overlay(result, record.stroke_color)

    # --------------------------------------------------------------- drawing
    def _clear(self, message: str) -> None:
        self._summary.setText(message)
        self._map_image.clear()
        self._total_curve.setData([], [])
        self._positive_curve.setData([], [])
        self._negative_curve.setData([], [])
        self._peak_points.setData([], [])
        self._centroid_curve.setData([], [])
        self._density_power_curve.setData([], [])
        self._transverse_power_curve.setData([], [])
        self._autocorr_curve.setData([], [])
        for line in (
            self._density_period_line,
            self._transverse_period_line,
            self._autocorr_period_line,
        ):
            line.setVisible(False)

    def _draw_result(self, result: SpatialLinePatternResult) -> None:
        s = result.s_centers_nm
        self._map_image.setImage(
            result.straightened_counts.T,
            autoLevels=True,
        )
        self._map_image.setRect(
            QRectF(
                float(result.s_edges_nm[0]),
                float(result.u_edges_nm[0]),
                float(np.ptp(result.s_edges_nm)),
                float(np.ptp(result.u_edges_nm)),
            )
        )
        self._map_plot.autoRange()

        self._total_curve.setData(s, result.smoothed_profile)
        self._positive_curve.setData(s, result.positive_profile)
        self._negative_curve.setData(s, result.negative_profile)
        if result.peak_indices.size:
            self._peak_points.setData(
                result.peak_positions_nm,
                result.smoothed_profile[result.peak_indices],
            )
        else:
            self._peak_points.setData([], [])
        self._centroid_curve.setData(
            s,
            result.smoothed_transverse_centroid_nm,
        )

        periods = result.spectrum_periods_nm
        self._density_power_curve.setData(
            periods,
            _normalized(result.density_spectrum_power),
        )
        self._transverse_power_curve.setData(
            periods,
            _normalized(result.transverse_spectrum_power),
        )
        self._set_period_line(
            self._density_period_line,
            result.density_fft_period_nm,
        )
        self._set_period_line(
            self._transverse_period_line,
            result.transverse_fft_period_nm,
        )

        lag_keep = (
            result.autocorrelation_lags_nm
            <= max(self._min_period.value(), self._max_period.value())
        )
        self._autocorr_curve.setData(
            result.autocorrelation_lags_nm[lag_keep],
            result.density_autocorrelation[lag_keep],
        )
        self._set_period_line(
            self._autocorr_period_line,
            result.density_autocorr_period_nm,
        )

        first_spacing = (
            float(np.median(result.peak_spacing_by_order_nm[0]))
            if result.peak_spacing_by_order_nm
            and result.peak_spacing_by_order_nm[0].size
            else float("nan")
        )
        agreement = ""
        if np.isfinite(result.density_fft_period_nm) and np.isfinite(
            result.density_autocorr_period_nm
        ):
            relative = abs(
                result.density_fft_period_nm - result.density_autocorr_period_nm
            ) / max(result.density_fft_period_nm, 1.0e-9)
            agreement = " · FFT/autocorrelation agree" if relative <= 0.20 else ""
        self._summary.setText(
            f"{self._dataset.name} · {self._view.roi_view_plane()} · "
            f"length {result.centerline.arc_nm[-1]:.1f} nm · "
            f"{result.n_used:,}/{result.n_input:,} localizations · "
            f"{result.peak_positions_nm.size} peaks · "
            f"median adjacent spacing {_finite_text(first_spacing)} · "
            f"density FFT {_finite_text(result.density_fft_period_nm)} "
            f"(peak/background {result.density_fft_snr:.2g}) · "
            f"autocorrelation {_finite_text(result.density_autocorr_period_nm)} · "
            f"transverse FFT {_finite_text(result.transverse_fft_period_nm)}"
            f"{agreement}"
        )

    @staticmethod
    def _set_period_line(line: pg.InfiniteLine, value: float) -> None:
        line.setVisible(bool(np.isfinite(value)))
        if np.isfinite(value):
            line.setPos(float(value))

    def _graphics_path(self, points: np.ndarray, *, closed: bool) -> QPainterPath:
        path = QPainterPath()
        if points.shape[0] == 0:
            return path
        path.moveTo(QPointF(float(points[0, 0]), float(points[0, 1])))
        for point in points[1:]:
            path.lineTo(QPointF(float(point[0]), float(point[1])))
        if closed:
            path.closeSubpath()
        return path

    def _update_source_overlay(
        self,
        result: SpatialLinePatternResult,
        color,
    ) -> None:
        try:
            view_box = self._view.coordinate_view_box()
        except Exception:
            return
        if view_box is None:
            return
        if self._curve_item is None:
            self._curve_item = QGraphicsPathItem()
            self._curve_item.setZValue(7)
            view_box.addItem(self._curve_item, ignoreBounds=True)
        if self._band_item is None:
            self._band_item = QGraphicsPathItem()
            self._band_item.setZValue(5)
            view_box.addItem(self._band_item, ignoreBounds=True)

        line_color = pg.mkColor(color or "#00b8d9")
        line_color.setAlpha(230)
        self._curve_item.setPen(
            pg.mkPen(
                line_color,
                width=2.0,
                style=Qt.PenStyle.DashLine,
            )
        )
        self._curve_item.setBrush(pg.mkBrush(None))
        self._curve_item.setPath(
            self._graphics_path(result.centerline.points_nm, closed=False)
        )
        self._curve_item.setVisible(True)

        band = band_polygon(
            result.centerline.points_nm,
            2.0 * self._half_width.value(),
            max(self._interpolation_step.value(), 0.5),
        )
        fill = pg.mkColor(line_color)
        fill.setAlpha(35)
        edge = pg.mkColor(line_color)
        edge.setAlpha(120)
        self._band_item.setPen(pg.mkPen(edge, width=1.0))
        self._band_item.setBrush(pg.mkBrush(fill))
        polygon = QPolygonF(
            [QPointF(float(x), float(y)) for x, y in band]
        )
        band_path = QPainterPath()
        band_path.addPolygon(polygon)
        band_path.closeSubpath()
        self._band_item.setPath(band_path)
        self._band_item.setVisible(band.shape[0] >= 3)

    def _hide_source_overlay(self) -> None:
        for item in (self._curve_item, self._band_item):
            if item is not None:
                item.setVisible(False)

    def _remove_source_overlay(self) -> None:
        try:
            view_box = self._view.coordinate_view_box()
        except Exception:
            view_box = None
        if view_box is not None:
            for item in (self._curve_item, self._band_item):
                if item is not None:
                    try:
                        view_box.removeItem(item)
                    except Exception:
                        pass
        self._curve_item = None
        self._band_item = None

    # ---------------------------------------------------------------- export
    def _default_stem(self) -> str:
        stem = Path(str(self._dataset.name or "dataset")).stem
        safe = "".join(char if char.isalnum() or char in "-_" else "_" for char in stem)
        return safe or "dataset"

    def _save_csv(self) -> None:
        result = self._result
        if result is None:
            return
        path, _selected = QFileDialog.getSaveFileName(
            self,
            "Save spatial line-pattern profiles",
            f"{self._default_stem()}_line_pattern.csv",
            "CSV (*.csv)",
        )
        if not path:
            return
        if not str(path).lower().endswith(".csv"):
            path += ".csv"
        table = np.column_stack(
            [
                result.s_centers_nm,
                result.total_profile,
                result.positive_profile,
                result.negative_profile,
                result.asymmetry,
                result.transverse_centroid_nm,
                result.smoothed_profile,
                result.detrended_profile,
            ]
        )
        try:
            np.savetxt(
                path,
                table,
                delimiter=",",
                header=(
                    "distance_nm,total_count,positive_side_count,"
                    "negative_side_count,side_asymmetry,"
                    "transverse_centroid_nm,smoothed_count,detrended_count"
                ),
                comments="",
                fmt="%.8g",
            )
        except Exception as exc:
            self._report_export_error(path, exc)
            return
        self._state.log(
            f"Saved spatial line-pattern profiles: {path}",
            dataset_idx=self._dataset_index(),
        )

    def _save_npz(self) -> None:
        result = self._result
        if result is None:
            return
        path, _selected = QFileDialog.getSaveFileName(
            self,
            "Save complete spatial line-pattern analysis",
            f"{self._default_stem()}_line_pattern.npz",
            "NumPy archive (*.npz)",
        )
        if not path:
            return
        if not str(path).lower().endswith(".npz"):
            path += ".npz"
        summary = {
            "dataset": self._dataset.name,
            "plane": self._view.roi_view_plane(),
            "parameters": self._parameters(),
            "density_fft_period_nm": result.density_fft_period_nm,
            "density_fft_snr": result.density_fft_snr,
            "density_autocorr_period_nm": result.density_autocorr_period_nm,
            "transverse_fft_period_nm": result.transverse_fft_period_nm,
            "transverse_fft_snr": result.transverse_fft_snr,
            "n_input": result.n_input,
            "n_used": result.n_used,
        }
        spacing_arrays = {
            f"peak_spacing_order_{order}_nm": values
            for order, values in enumerate(
                result.peak_spacing_by_order_nm,
                start=1,
            )
        }
        try:
            np.savez_compressed(
                path,
                metadata_json=np.asarray(json.dumps(summary)),
                source_roi_points_nm=result.centerline.source_points_nm,
                centerline_points_nm=result.centerline.points_nm,
                centerline_arc_nm=result.centerline.arc_nm,
                centerline_tangent=result.centerline.tangent,
                centerline_normal=result.centerline.normal,
                filtered_input_row_indices=result.point_indices,
                point_s_nm=result.point_s_nm,
                point_u_nm=result.point_u_nm,
                s_edges_nm=result.s_edges_nm,
                u_edges_nm=result.u_edges_nm,
                straightened_counts=result.straightened_counts,
                total_profile=result.total_profile,
                positive_profile=result.positive_profile,
                negative_profile=result.negative_profile,
                asymmetry=result.asymmetry,
                transverse_centroid_nm=result.transverse_centroid_nm,
                smoothed_profile=result.smoothed_profile,
                detrended_profile=result.detrended_profile,
                smoothed_transverse_centroid_nm=(
                    result.smoothed_transverse_centroid_nm
                ),
                detrended_transverse_centroid_nm=(
                    result.detrended_transverse_centroid_nm
                ),
                peak_indices=result.peak_indices,
                peak_positions_nm=result.peak_positions_nm,
                peak_prominences=result.peak_prominences,
                spectrum_periods_nm=result.spectrum_periods_nm,
                density_spectrum_power=result.density_spectrum_power,
                transverse_spectrum_power=result.transverse_spectrum_power,
                autocorrelation_lags_nm=result.autocorrelation_lags_nm,
                density_autocorrelation=result.density_autocorrelation,
                **spacing_arrays,
            )
        except Exception as exc:
            self._report_export_error(path, exc)
            return
        self._state.log(
            f"Saved complete spatial line-pattern analysis: {path}",
            dataset_idx=self._dataset_index(),
        )

    def _report_export_error(self, path: str, exc: Exception) -> None:
        message = f"Could not save spatial line-pattern analysis to {path}: {exc}"
        self._state.log(
            message,
            level="ERROR",
            dataset_idx=self._dataset_index(),
        )
        QMessageBox.critical(self, "Spatial Pattern Analysis", message)

    def closeEvent(self, event) -> None:
        self._timer.stop()
        self._remove_source_overlay()
        super().closeEvent(event)
