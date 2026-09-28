"""Modeless trajectory-analysis workbench built on the pure tracking contract."""

from __future__ import annotations

import json
import math
from collections.abc import Mapping
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pyqtgraph as pg
from PyQt6.QtCore import QAbstractTableModel, QModelIndex, QSortFilterProxyModel, Qt, QTimer
from PyQt6.QtGui import QFontDatabase
from PyQt6.QtWidgets import (
    QAbstractItemView,
    QCheckBox,
    QComboBox,
    QDoubleSpinBox,
    QFileDialog,
    QFormLayout,
    QGroupBox,
    QHBoxLayout,
    QHeaderView,
    QLabel,
    QMessageBox,
    QPlainTextEdit,
    QPushButton,
    QSpinBox,
    QTableView,
    QTabWidget,
    QVBoxLayout,
    QWidget,
)

from ..analysis.tracking_attributes import install_tracking_attributes
from ..analysis.tracking_export import export_tracking_result_zip
from ..analysis.tracking_stats import (
    PreparedTrajectories,
    TrackingResult,
    prepare_dataset_trajectories,
    run_tracking_method,
    snapshot_dataset_trajectories,
)
from ..core.tracking_time import tracking_precision_from_prefs
from .plot_format import plot_widget

MSD_METHOD_ID = "minflux_viewer.tracking.msd"


@dataclass(frozen=True)
class _AnalysisPayload:
    dataset_identity: int
    prepared: PreparedTrajectories
    result: TrackingResult
    kinematics: TrackingResult | None = None


def _display_value(value: Any) -> str:
    if isinstance(value, np.generic):
        value = value.item()
    if value is None:
        return ""
    if isinstance(value, bool):
        return "true" if value else "false"
    if isinstance(value, (int, np.integer)):
        return str(int(value))
    if isinstance(value, (float, np.floating)):
        number = float(value)
        if math.isnan(number):
            return "nan"
        if math.isinf(number):
            return "inf" if number > 0 else "-inf"
        return format(number, ".8g")
    return str(value)


class _ArrayTableModel(QAbstractTableModel):
    """Read-only, non-materialising view over one TrackingResult table."""

    def __init__(self, parent=None) -> None:
        super().__init__(parent)
        self._columns: tuple[str, ...] = ()
        self._arrays: tuple[np.ndarray, ...] = ()
        self._units: Mapping[str, str] = {}

    def set_table(
        self,
        table: Mapping[str, np.ndarray],
        units: Mapping[str, str] | None = None,
    ) -> None:
        self.beginResetModel()
        self._columns = tuple(table)
        self._arrays = tuple(np.asarray(table[key]).reshape(-1) for key in self._columns)
        self._units = dict(units or {})
        self.endResetModel()

    def rowCount(self, parent=QModelIndex()) -> int:  # noqa: N802 - Qt API
        if parent.isValid() or not self._arrays:
            return 0
        return int(self._arrays[0].size)

    def columnCount(self, parent=QModelIndex()) -> int:  # noqa: N802 - Qt API
        return 0 if parent.isValid() else len(self._columns)

    def data(self, index: QModelIndex, role=Qt.ItemDataRole.DisplayRole):
        if not index.isValid():
            return None
        value = self._arrays[index.column()][index.row()]
        if isinstance(value, np.generic):
            value = value.item()
        if role == Qt.ItemDataRole.DisplayRole:
            return _display_value(value)
        if role == Qt.ItemDataRole.UserRole:
            return value
        if role == Qt.ItemDataRole.TextAlignmentRole:
            horizontal = (
                Qt.AlignmentFlag.AlignRight
                if isinstance(value, (int, float)) and not isinstance(value, bool)
                else Qt.AlignmentFlag.AlignLeft
            )
            return horizontal | Qt.AlignmentFlag.AlignVCenter
        return None

    def headerData(  # noqa: N802 - Qt API
        self,
        section: int,
        orientation: Qt.Orientation,
        role=Qt.ItemDataRole.DisplayRole,
    ):
        if role != Qt.ItemDataRole.DisplayRole:
            return None
        if orientation == Qt.Orientation.Vertical:
            return section + 1
        name = self._columns[section]
        unit = self._units.get(name, "")
        return f"{name} ({unit})" if unit and unit != "1" else name


class _NumericSortProxy(QSortFilterProxyModel):
    def lessThan(self, left: QModelIndex, right: QModelIndex) -> bool:  # noqa: N802
        a = left.data(Qt.ItemDataRole.UserRole)
        b = right.data(Qt.ItemDataRole.UserRole)
        try:
            af, bf = float(a), float(b)
            if np.isfinite(af) and np.isfinite(bf):
                return af < bf
        except (TypeError, ValueError):
            pass
        return str(a).casefold() < str(b).casefold()


class MsdAnalysisWindow(QWidget):
    """Parameters, curves, per-segment fits and audit trail for MSD analysis."""

    def __init__(self, state, dataset_idx: int, owner=None) -> None:
        super().__init__(None)
        self._state = state
        self._owner = owner
        self._idx = int(dataset_idx)
        self._dataset_identity = -1
        dataset = self._dataset()
        self._dataset_identity = id(dataset) if dataset is not None else -1
        self._generation = 0
        self._tasks: dict[int, object] = {}
        self._closing = False
        self._installing_attributes = False
        self._payload: _AnalysisPayload | None = None

        name = str(getattr(dataset, "name", "dataset"))
        self.setWindowTitle(f"MSD Analysis — {name}")
        self.setWindowFlags(Qt.WindowType.Window)
        self.setAttribute(Qt.WidgetAttribute.WA_DeleteOnClose)
        self.resize(1040, 760)
        self._build_ui()

        state.filter_changed.connect(self._on_dataset_changed)
        state.calibration_changed.connect(self._on_dataset_changed)
        state.attributes_changed.connect(self._on_dataset_changed)
        state.dataset_removed.connect(self._on_dataset_removed)
        QTimer.singleShot(0, self.run_analysis)

    def _dataset(self):
        if 0 <= self._idx < len(self._state.datasets):
            dataset = self._state.datasets[self._idx]
            if self._dataset_identity in {-1, id(dataset)}:
                return dataset
        return None

    def _build_ui(self) -> None:
        root = QVBoxLayout(self)
        root.setContentsMargins(8, 8, 8, 8)
        root.setSpacing(6)

        controls = QGroupBox("Analysis input and fit")
        form = QFormLayout(controls)
        form.setHorizontalSpacing(12)
        form.setVerticalSpacing(5)

        self._scope = QComboBox()
        self._scope.addItem("Current filter", True)
        self._scope.addItem("All materialized localizations", False)
        self._scope.setToolTip(
            "A removed middle row becomes a segment boundary; MSD never bridges it.")
        form.addRow("Rows", self._scope)

        self._time_mode = QComboBox()
        self._time_mode.addItem("Timestamp (physical seconds)", "timestamp")
        self._time_mode.addItem("Relative index (non-physical)", "index")
        dataset = self._dataset()
        try:
            snapshot = snapshot_dataset_trajectories(dataset, filtered=True)
            has_time = snapshot.timestamps_s is not None
        except Exception:
            has_time = False
        if not has_time:
            self._time_mode.setCurrentIndex(1)
            self._time_mode.model().item(0).setEnabled(False)
        self._time_mode.currentIndexChanged.connect(self._update_unit_labels)
        form.addRow("Time source", self._time_mode)

        self._dimensions = QComboBox()
        self._dimensions.addItem("XY (2 dimensions)", 2)
        if dataset is not None and int(getattr(dataset.prop, "num_dim", 2)) >= 3:
            self._dimensions.addItem("XYZ (3 dimensions)", 3)
        form.addRow("Coordinates", self._dimensions)

        self._lag_bin = QDoubleSpinBox()
        self._lag_bin.setRange(0.0, 1.0e9)
        self._lag_bin.setDecimals(9)
        self._lag_bin.setSingleStep(0.001)
        self._lag_bin.setSpecialValueText("Automatic (modal interval)")
        self._lag_bin.setToolTip(
            "Width used to group measured pair lags. Zero uses the modal interval; "
            "positions are never interpolated.")
        self._lag_bin_label = QLabel("Lag-bin width (s)")
        form.addRow(self._lag_bin_label, self._lag_bin)

        self._max_lag_fraction = QDoubleSpinBox()
        self._max_lag_fraction.setRange(1.0, 100.0)
        self._max_lag_fraction.setDecimals(1)
        self._max_lag_fraction.setValue(25.0)
        self._max_lag_fraction.setSuffix(" % of segment")
        form.addRow("Maximum lag", self._max_lag_fraction)

        integer_row = QHBoxLayout()
        self._min_pairs = QSpinBox()
        self._min_pairs.setRange(1, 1_000_000)
        self._min_pairs.setValue(3)
        self._fit_lags = QSpinBox()
        self._fit_lags.setRange(2, 1000)
        self._fit_lags.setValue(4)
        self._min_segment_points = QSpinBox()
        self._min_segment_points.setRange(2, 1_000_000)
        self._min_segment_points.setValue(5)
        for label, widget in (
            ("pairs/bin", self._min_pairs),
            ("fit lags", self._fit_lags),
            ("points/segment", self._min_segment_points),
        ):
            integer_row.addWidget(QLabel(label))
            integer_row.addWidget(widget)
        integer_row.addStretch()
        form.addRow("Minimums", integer_row)

        run_row = QHBoxLayout()
        self._run_button = QPushButton("Run / refresh")
        self._run_button.clicked.connect(self.run_analysis)
        run_row.addWidget(self._run_button)
        self._export_button = QPushButton("Export result ZIP…")
        self._export_button.setEnabled(False)
        self._export_button.clicked.connect(self._export_result)
        run_row.addWidget(self._export_button)
        self._attach_attributes = QCheckBox("Add selected attributes to dataset")
        self._attach_attributes.setChecked(bool(
            (self._state.prefs.get("tracking", {}) or {}).get(
                "materialize_analysis_attributes", False)))
        self._attach_attributes.setToolTip(
            "Add the attributes selected in Preferences › Tracking after a "
            "successful fit. Excluded rows remain NaN and provenance is retained.")
        run_row.addWidget(self._attach_attributes)
        self._log_axes = QCheckBox("Log axes")
        self._log_axes.toggled.connect(self._set_log_axes)
        run_row.addWidget(self._log_axes)
        run_row.addStretch()
        form.addRow("", run_row)
        root.addWidget(controls)

        self._tabs = QTabWidget()
        root.addWidget(self._tabs, 1)
        self._build_curve_tab()
        self._build_segment_tab()
        self._build_fit_tab()
        self._build_audit_tab()

        self._status = QLabel("Waiting to run…")
        self._status.setWordWrap(True)
        self._status.setStyleSheet("color: #777;")
        root.addWidget(self._status)

    def _build_curve_tab(self) -> None:
        page = QWidget()
        layout = QVBoxLayout(page)
        layout.setContentsMargins(4, 4, 4, 4)
        self._summary = QLabel("No result yet.")
        self._summary.setTextInteractionFlags(Qt.TextInteractionFlag.TextSelectableByMouse)
        self._summary.setWordWrap(True)
        layout.addWidget(self._summary)
        self._overview_plot = plot_widget(background="w")
        self._overview_plot.showGrid(x=True, y=True, alpha=0.18)
        layout.addWidget(self._overview_plot, 1)
        self._tabs.addTab(page, "Pooled & ensemble")

    def _build_segment_tab(self) -> None:
        page = QWidget()
        layout = QVBoxLayout(page)
        top = QHBoxLayout()
        top.addWidget(QLabel("Segment"))
        self._segment_combo = QComboBox()
        self._segment_combo.currentIndexChanged.connect(self._draw_segment)
        top.addWidget(self._segment_combo)
        self._segment_fit = QLabel("")
        self._segment_fit.setTextInteractionFlags(
            Qt.TextInteractionFlag.TextSelectableByMouse)
        top.addWidget(self._segment_fit, 1)
        layout.addLayout(top)
        self._segment_plot = plot_widget(background="w")
        self._segment_plot.showGrid(x=True, y=True, alpha=0.18)
        layout.addWidget(self._segment_plot, 1)
        self._tabs.addTab(page, "Per segment")

    def _build_fit_tab(self) -> None:
        page = QWidget()
        layout = QVBoxLayout(page)
        self._fit_model = _ArrayTableModel(self)
        self._fit_proxy = _NumericSortProxy(self)
        self._fit_proxy.setSourceModel(self._fit_model)
        table = QTableView()
        table.setModel(self._fit_proxy)
        table.setSortingEnabled(True)
        table.setAlternatingRowColors(True)
        table.setEditTriggers(QAbstractItemView.EditTrigger.NoEditTriggers)
        table.setSelectionBehavior(QAbstractItemView.SelectionBehavior.SelectRows)
        table.verticalHeader().setVisible(False)
        table.horizontalHeader().setSectionResizeMode(QHeaderView.ResizeMode.ResizeToContents)
        table.setFont(QFontDatabase.systemFont(QFontDatabase.SystemFont.FixedFont))
        layout.addWidget(table)
        self._fit_table = table
        self._tabs.addTab(page, "Segment fits")

    def _build_audit_tab(self) -> None:
        page = QWidget()
        layout = QVBoxLayout(page)
        self._audit = QPlainTextEdit()
        self._audit.setReadOnly(True)
        self._audit.setFont(QFontDatabase.systemFont(QFontDatabase.SystemFont.FixedFont))
        layout.addWidget(self._audit)
        self._tabs.addTab(page, "Diagnostics & provenance")

    def _parameters(self) -> dict[str, Any]:
        lag_bin = float(self._lag_bin.value())
        return {
            "dimensions": int(self._dimensions.currentData()),
            "lag_bin": None if lag_bin <= 0.0 else lag_bin,
            "max_lag_fraction": float(self._max_lag_fraction.value()) / 100.0,
            "min_pairs": int(self._min_pairs.value()),
            "fit_lags": int(self._fit_lags.value()),
            "min_segment_points": int(self._min_segment_points.value()),
        }

    def run_analysis(self) -> None:
        if self._closing:
            return
        dataset = self._dataset()
        if dataset is None:
            self._status.setText("The source dataset is no longer available.")
            self._run_button.setEnabled(False)
            return
        from .background_tasks import BackgroundTask, request_task_cancel

        self._generation += 1
        generation = self._generation
        for task in tuple(self._tasks.values()):
            request_task_cancel(task)
        try:
            snapshot = snapshot_dataset_trajectories(
                dataset, filtered=bool(self._scope.currentData()))
            precision = tracking_precision_from_prefs(self._state.prefs)
            mode = str(self._time_mode.currentData())
            parameters = self._parameters()
            tracking_prefs = self._state.prefs.get("tracking", {}) or {}
            attribute_names = tuple(tracking_prefs.get(
                "analysis_attributes",
                ("msd_d", "msd_alpha", "msd_sigma_apparent"),
            ))
            attach_attributes = bool(self._attach_attributes.isChecked())
        except Exception as exc:  # noqa: BLE001 - present input failure in the window
            self._on_failed(generation, str(exc))
            return

        def work(report):
            report("Preparing calibrated, gap-safe trajectories")
            prepared = prepare_dataset_trajectories(
                snapshot,
                time_mode=mode,
                timestamp_precision_s=precision,
            )
            report("Computing measured-lag MSD curves")
            result = run_tracking_method(MSD_METHOD_ID, prepared, **parameters)
            kinematics = None
            if attach_attributes and any(
                name in {"step_angle", "track_straightness"}
                for name in attribute_names
            ):
                report("Computing gap-safe trajectory kinematics")
                kinematics = run_tracking_method(
                    "minflux_viewer.tracking.kinematics", prepared)
            report("Finalizing fit diagnostics")
            return _AnalysisPayload(
                snapshot.dataset_identity, prepared, result, kinematics)

        task = BackgroundTask(
            work,
            description=f"MSD analysis: {snapshot.dataset_name}",
            category="analysis",
        )
        task.signals.stage.connect(
            lambda text, g=generation: self._on_stage(g, text))
        task.signals.done.connect(
            lambda payload, g=generation: self._on_done(g, payload))
        task.signals.failed.connect(
            lambda message, g=generation: self._on_failed(g, message))
        task.signals.cancelled.connect(
            lambda g=generation: self._on_cancelled(g))
        task.signals.finished.connect(
            lambda g=generation, t=task: self._forget_task(g, t))
        self._tasks[generation] = task
        self._run_button.setEnabled(False)
        self._status.setText("Running MSD analysis…")
        self._start_task(task)

    def _start_task(self, task) -> None:
        from .background_tasks import shared_thread_pool

        shared_thread_pool("tracking-analysis", max_threads=1).start(task)

    def _on_stage(self, generation: int, text: str) -> None:
        if generation == self._generation and not self._closing:
            self._status.setText(f"Running MSD analysis… {text}")

    def _forget_task(self, generation: int, task) -> None:
        if self._tasks.get(generation) is task:
            self._tasks.pop(generation, None)

    def _on_done(self, generation: int, payload: _AnalysisPayload) -> None:
        if generation != self._generation or self._closing:
            return
        dataset = self._dataset()
        if dataset is None or id(dataset) != payload.dataset_identity:
            self._on_failed(generation, "the source dataset changed during analysis")
            return
        self._payload = payload
        self._run_button.setEnabled(True)
        self._export_button.setEnabled(True)
        self._populate_result(payload)
        if self._attach_attributes.isChecked():
            names = tuple((self._state.prefs.get("tracking", {}) or {}).get(
                "analysis_attributes",
                ("msd_d", "msd_alpha", "msd_sigma_apparent"),
            ))
            try:
                installed = install_tracking_attributes(
                    dataset,
                    payload.prepared,
                    msd=payload.result,
                    kinematics=payload.kinematics,
                    names=names,
                )
            except Exception as exc:  # noqa: BLE001 - preserve the valid result
                self._state.log(
                    f"MSD result is valid, but attributes were not added: {exc}",
                    "WARN",
                )
            else:
                if installed:
                    self._installing_attributes = True
                    try:
                        self._state.notify_attributes_changed(self._idx)
                    finally:
                        self._installing_attributes = False
                    self._status.setText(
                        f"{self._status.text()} Added {', '.join(installed)} to the dataset.")

    def _on_failed(self, generation: int, message: str) -> None:
        if generation != self._generation or self._closing:
            return
        self._run_button.setEnabled(self._dataset() is not None)
        self._status.setText(f"MSD analysis failed: {message}")
        self._state.log(f"MSD analysis failed: {message}", "ERROR")

    def _on_cancelled(self, generation: int) -> None:
        if generation != self._generation or self._closing:
            return
        self._run_button.setEnabled(self._dataset() is not None)
        self._status.setText("MSD analysis cancelled.")

    def _populate_result(self, payload: _AnalysisPayload) -> None:
        result = payload.result
        fit = result.table("pooled_fit")
        unit = result.units["pooled_fit"]["diffusion"]
        diffusion = float(fit["diffusion"][0]) if fit["diffusion"].size else np.nan
        alpha = float(fit["alpha"][0]) if fit["alpha"].size else np.nan
        sigma = (
            float(fit["apparent_sigma_nm"][0])
            if fit["apparent_sigma_nm"].size else np.nan
        )
        self._summary.setText(
            f"{payload.prepared.n_traces:,} trace(s), "
            f"{payload.prepared.n_segments:,} segment(s), "
            f"{payload.prepared.n_points:,} localization(s)  ·  "
            f"D = {_display_value(diffusion)} {unit}  ·  "
            f"α = {_display_value(alpha)}  ·  "
            f"apparent σ = {_display_value(sigma)} nm"
        )
        self._draw_overview()

        fits = result.table("segment_fit")
        self._fit_model.set_table(fits, result.units.get("segment_fit", {}))
        self._fit_table.resizeColumnsToContents()
        self._segment_combo.blockSignals(True)
        self._segment_combo.clear()
        for segment_id, trace_id in zip(fits["segment_id"], fits["trace_id"]):
            self._segment_combo.addItem(
                f"{int(segment_id)}  ·  trace {trace_id}", int(segment_id))
        self._segment_combo.blockSignals(False)
        self._draw_segment()

        audit = {
            "method_id": result.method_id,
            "method_version": result.method_version,
            "diagnostics": asdict(result.diagnostics),
            "preparation": asdict(payload.prepared.diagnostics),
            "provenance": dict(result.provenance),
            "citations": list(result.citations),
            "units": result.units,
        }
        self._audit.setPlainText(
            json.dumps(audit, indent=2, ensure_ascii=False, default=str))
        excluded = sum(int(value) for value in result.diagnostics.excluded.values())
        self._status.setText(
            f"Finished: {result.diagnostics.status}; "
            f"{excluded:,} excluded segment case(s). "
            "The intercept reports apparent σ; motion blur and tracking-loop "
            "dynamic error are not silently corrected."
        )

    def _draw_overview(self) -> None:
        self._overview_plot.clear()
        payload = self._payload
        if payload is None:
            return
        result = payload.result
        legend = self._overview_plot.addLegend()
        for level, label, color, symbol in (
            ("pooled_time_averaged_msd", "Pooled time-averaged", "#1565c0", "o"),
            ("ensemble_msd", "Ensemble from onset", "#2e7d32", "t"),
        ):
            table = result.table(level)
            if table["lag"].size:
                self._overview_plot.plot(
                    table["lag"],
                    table["msd_nm2"],
                    pen=pg.mkPen(color, width=2),
                    symbol=symbol,
                    symbolSize=6,
                    symbolBrush=color,
                    name=label,
                )
        legend.setVisible(True)
        time_unit = result.units["pooled_time_averaged_msd"]["lag"]
        self._overview_plot.setLabel("bottom", f"Measured lag ({time_unit})")
        self._overview_plot.setLabel("left", "MSD (nm²)")
        self._set_log_axes(self._log_axes.isChecked())
        self._overview_plot.autoRange()

    def _draw_segment(self) -> None:
        self._segment_plot.clear()
        payload = self._payload
        segment_id = self._segment_combo.currentData()
        if payload is None or segment_id is None:
            self._segment_fit.setText("No fitted segment.")
            return
        result = payload.result
        curves = result.table("segment_msd")
        selected = np.asarray(curves["segment_id"]) == int(segment_id)
        if np.any(selected):
            self._segment_plot.plot(
                np.asarray(curves["lag"])[selected],
                np.asarray(curves["msd_nm2"])[selected],
                pen=pg.mkPen("#7b1fa2", width=2),
                symbol="o",
                symbolSize=6,
                symbolBrush="#7b1fa2",
            )
        fits = result.table("segment_fit")
        row = np.flatnonzero(np.asarray(fits["segment_id"]) == int(segment_id))
        if row.size:
            i = int(row[0])
            unit = result.units["segment_fit"]["diffusion"]
            self._segment_fit.setText(
                f"D {_display_value(fits['diffusion'][i])} {unit}  ·  "
                f"α {_display_value(fits['alpha'][i])}  ·  "
                f"R² {_display_value(fits['r_squared'][i])}  ·  "
                f"{fits['reason'][i]}"
            )
        time_unit = result.units["segment_msd"]["lag"]
        self._segment_plot.setLabel("bottom", f"Measured lag ({time_unit})")
        self._segment_plot.setLabel("left", "MSD (nm²)")
        self._segment_plot.setLogMode(
            x=self._log_axes.isChecked(), y=self._log_axes.isChecked())
        self._segment_plot.autoRange()

    def _set_log_axes(self, enabled: bool) -> None:
        self._overview_plot.setLogMode(x=bool(enabled), y=bool(enabled))
        self._segment_plot.setLogMode(x=bool(enabled), y=bool(enabled))

    def _update_unit_labels(self) -> None:
        unit = "s" if self._time_mode.currentData() == "timestamp" else "index"
        self._lag_bin_label.setText(f"Lag-bin width ({unit})")

    def _export_result(self) -> None:
        if self._payload is None:
            return
        dataset = self._dataset()
        base = str(getattr(dataset, "name", "tracking")) if dataset is not None else "tracking"
        suggested = f"{base}_msd_result.zip"
        path, _filter = QFileDialog.getSaveFileName(
            self,
            "Export tracking-analysis result",
            str(Path.home() / suggested),
            "Tracking result ZIP (*.zip)",
        )
        if not path:
            return
        try:
            saved = export_tracking_result_zip(self._payload.result, path)
        except Exception as exc:  # noqa: BLE001 - user-facing export boundary
            QMessageBox.critical(self, "Could not export result", str(exc))
            return
        self._state.log(f"Exported MSD analysis result to {saved}")
        self._status.setText(f"Exported result to {saved}")

    def _invalidate(self, reason: str) -> None:
        from .background_tasks import request_task_cancel

        self._generation += 1
        for task in tuple(self._tasks.values()):
            request_task_cancel(task)
        self._payload = None
        self._export_button.setEnabled(False)
        self._run_button.setEnabled(self._dataset() is not None)
        self._summary.setText("No current result.")
        self._segment_combo.clear()
        self._fit_model.set_table({})
        self._overview_plot.clear()
        self._segment_plot.clear()
        self._audit.clear()
        self._status.setText(reason)

    def _on_dataset_changed(self, idx: int) -> None:
        if not self._closing and not self._installing_attributes and idx == self._idx:
            self._invalidate("The dataset changed. Run again to refresh the analysis.")

    def refresh_preferences(self) -> None:
        if not self._closing:
            self._invalidate(
                "Tracking preferences changed. Run again to refresh the analysis.")

    def _on_dataset_removed(self, _removed_idx: int) -> None:
        if self._closing:
            return
        for idx, dataset in enumerate(self._state.datasets):
            if id(dataset) == self._dataset_identity:
                self._idx = idx
                return
        self.close()

    def focusInEvent(self, event) -> None:  # noqa: N802 - Qt API
        if self._dataset() is not None:
            self._state.set_active(self._idx)
        super().focusInEvent(event)

    def closeEvent(self, event) -> None:  # noqa: N802 - Qt API
        self._closing = True
        from .background_tasks import retire_background_tasks
        from .qt_lifecycle import close_plot_widgets

        retire_background_tasks(self._tasks.values())
        self._tasks.clear()
        close_plot_widgets(self._overview_plot, self._segment_plot)
        super().closeEvent(event)


__all__ = ["MSD_METHOD_ID", "MsdAnalysisWindow"]
