"""Editor for separating an acquisition into time-window channels."""

from __future__ import annotations

import numpy as np
import pyqtgraph as pg
from PyQt6.QtCore import Qt
from PyQt6.QtWidgets import (
    QAbstractItemView,
    QComboBox,
    QDialog,
    QDialogButtonBox,
    QDoubleSpinBox,
    QHBoxLayout,
    QHeaderView,
    QLabel,
    QLineEdit,
    QMessageBox,
    QPushButton,
    QSpinBox,
    QTableWidget,
    QTableWidgetItem,
    QVBoxLayout,
)

from ..colormaps import channel_colormap_names, representative_rgb
from ..colors import is_solid_color, solid_color_rgb
from ..core.time_channels import TimeWindow, time_channel_selections


def _lut_preview_rgb(lut: str) -> tuple[int, int, int]:
    if is_solid_color(lut):
        return solid_color_rgb(lut)
    try:
        return tuple(
            int(round(channel * 255.0)) for channel in representative_rgb(lut)
        )
    except (KeyError, ValueError):
        return 80, 120, 210


class TimeChannelDialog(QDialog):
    """Edit named, non-overlapping time windows over a localization histogram."""

    def __init__(
        self,
        tim_s,
        *,
        source_name: str,
        base_mask=None,
        color_cycle: list[str] | None = None,
        parent=None,
    ) -> None:
        super().__init__(parent)
        self._tim_s = np.asarray(tim_s, dtype=float).ravel()
        if base_mask is None:
            self._base_mask = np.ones(self._tim_s.size, dtype=bool)
        else:
            self._base_mask = np.asarray(base_mask, dtype=bool).ravel()
        if self._base_mask.size != self._tim_s.size:
            raise ValueError("The active filter mask does not align with tim.")
        self._source_name = str(source_name)
        self._color_cycle = list(color_cycle or ["Red", "Green", "Blue", "Cyan"])
        self._rows: list[dict] = []
        self._synchronizing = False
        self._accepted_windows: list[TimeWindow] | None = None

        finite = np.isfinite(self._tim_s)
        visible = finite & self._base_mask
        range_values = self._tim_s[visible] if np.any(visible) else self._tim_s[finite]
        if range_values.size == 0:
            raise ValueError("The active dataset has no finite tim values.")
        self._time_min_s = float(np.min(range_values))
        self._time_max_s = float(np.max(range_values))
        if self._time_max_s <= self._time_min_s:
            raise ValueError("The active dataset does not span a usable time range.")

        self.setWindowTitle("Separate Channels from Time Windows")
        self.resize(900, 660)
        self._build_ui()
        self._set_even_windows(2)

    def _build_ui(self) -> None:
        root = QVBoxLayout(self)
        root.setContentsMargins(8, 8, 8, 8)
        root.setSpacing(8)

        duration_min = (self._time_max_s - self._time_min_s) / 60.0
        heading = QLabel(
            f"<b>{self._source_name}</b> &nbsp; "
            f"{int(self._base_mask.sum()):,} active localizations &nbsp; "
            f"{duration_min:.2f} min"
        )
        root.addWidget(heading)

        self._plot = pg.PlotWidget()
        self._plot.setLabel("bottom", "acquisition time", units="min")
        self._plot.setLabel("left", "localizations / bin")
        self._plot.showGrid(x=True, y=True, alpha=0.15)
        self._plot.setMinimumHeight(250)
        root.addWidget(self._plot, 1)
        self._draw_histogram()

        controls = QHBoxLayout()
        controls.addWidget(QLabel("Windows:"))
        self._window_count = QSpinBox()
        self._window_count.setRange(2, 32)
        self._window_count.setValue(2)
        controls.addWidget(self._window_count)
        even_button = QPushButton("Split evenly")
        even_button.clicked.connect(
            lambda: self._set_even_windows(self._window_count.value())
        )
        controls.addWidget(even_button)
        controls.addSpacing(12)
        add_button = QPushButton("Add window")
        add_button.clicked.connect(self._add_by_splitting)
        controls.addWidget(add_button)
        remove_button = QPushButton("Remove selected")
        remove_button.clicked.connect(self._remove_selected)
        controls.addWidget(remove_button)
        controls.addStretch(1)
        root.addLayout(controls)

        self._table = QTableWidget(0, 5, self)
        self._table.setHorizontalHeaderLabels(
            ["Channel name", "Start (min)", "End (min)", "LUT / color", "Locs"]
        )
        self._table.verticalHeader().setVisible(False)
        self._table.setSelectionBehavior(QAbstractItemView.SelectionBehavior.SelectRows)
        self._table.setSelectionMode(QAbstractItemView.SelectionMode.ExtendedSelection)
        header = self._table.horizontalHeader()
        header.setSectionResizeMode(QHeaderView.ResizeMode.ResizeToContents)
        header.setSectionResizeMode(0, QHeaderView.ResizeMode.Stretch)
        self._table.setMinimumHeight(170)
        root.addWidget(self._table)

        self._status = QLabel()
        root.addWidget(self._status)

        buttons = QDialogButtonBox(
            QDialogButtonBox.StandardButton.Ok | QDialogButtonBox.StandardButton.Cancel
        )
        buttons.button(QDialogButtonBox.StandardButton.Ok).setText("Create channels")
        buttons.accepted.connect(self.accept)
        buttons.rejected.connect(self.reject)
        root.addWidget(buttons)

    def _draw_histogram(self) -> None:
        values_min = self._tim_s[self._base_mask & np.isfinite(self._tim_s)] / 60.0
        if values_min.size == 0:
            return
        lo = float(values_min.min())
        hi = float(values_min.max())
        bins = int(np.clip(np.sqrt(values_min.size), 20, 160))
        if hi <= lo:
            edges = np.array([lo - 0.5, hi + 0.5])
        else:
            edges = np.linspace(lo, hi, bins + 1)
        counts, edges = np.histogram(values_min, bins=edges)
        curve = pg.PlotCurveItem(
            edges,
            counts,
            stepMode="center",
            fillLevel=0,
            brush=(100, 130, 165, 90),
            pen=pg.mkPen((80, 105, 135), width=1),
        )
        self._plot.addItem(curve)

    def _set_even_windows(self, count: int) -> None:
        self._clear_rows()
        edges = np.linspace(self._time_min_s, self._time_max_s, int(count) + 1)
        for index in range(int(count)):
            self._append_row(
                name=f"{self._source_name} [time {index + 1}]",
                start_s=float(edges[index]),
                end_s=float(edges[index + 1]),
                lut=self._color_cycle[index % len(self._color_cycle)],
            )
        self._window_count.setValue(int(count))
        self._refresh_counts()

    def _clear_rows(self) -> None:
        for row in self._rows:
            self._plot.removeItem(row["region"])
        self._rows.clear()
        self._table.setRowCount(0)

    @staticmethod
    def _spin(value_min: float) -> QDoubleSpinBox:
        spin = QDoubleSpinBox()
        spin.setDecimals(4)
        spin.setRange(-1.0e9, 1.0e9)
        spin.setSingleStep(0.1)
        spin.setValue(value_min)
        spin.setAlignment(Qt.AlignmentFlag.AlignRight)
        return spin

    def _append_row(self, *, name: str, start_s: float, end_s: float, lut: str) -> None:
        table_row = self._table.rowCount()
        self._table.insertRow(table_row)

        name_edit = QLineEdit(name)
        start_spin = self._spin(start_s / 60.0)
        end_spin = self._spin(end_s / 60.0)
        lut_combo = QComboBox()
        lut_combo.addItems(channel_colormap_names())
        if lut_combo.findText(lut) < 0:
            lut_combo.addItem(lut)
        lut_combo.setCurrentText(lut)
        count_item = QTableWidgetItem("0")
        count_item.setFlags(count_item.flags() & ~Qt.ItemFlag.ItemIsEditable)
        count_item.setTextAlignment(
            Qt.AlignmentFlag.AlignRight | Qt.AlignmentFlag.AlignVCenter
        )

        self._table.setCellWidget(table_row, 0, name_edit)
        self._table.setCellWidget(table_row, 1, start_spin)
        self._table.setCellWidget(table_row, 2, end_spin)
        self._table.setCellWidget(table_row, 3, lut_combo)
        self._table.setItem(table_row, 4, count_item)

        rgb = _lut_preview_rgb(lut)
        region = pg.LinearRegionItem(
            values=(start_s / 60.0, end_s / 60.0),
            orientation=pg.LinearRegionItem.Vertical,
            movable=True,
            brush=(*rgb, 35),
            pen=pg.mkPen(rgb, width=2),
            hoverPen=pg.mkPen(rgb, width=3),
        )
        region.setZValue(10 + table_row)
        self._plot.addItem(region)

        row = {
            "name": name_edit,
            "start": start_spin,
            "end": end_spin,
            "lut": lut_combo,
            "count": count_item,
            "region": region,
        }
        self._rows.append(row)
        name_edit.textChanged.connect(self._refresh_counts)
        start_spin.valueChanged.connect(lambda _value, item=row: self._spin_changed(item))
        end_spin.valueChanged.connect(lambda _value, item=row: self._spin_changed(item))
        lut_combo.currentTextChanged.connect(
            lambda _value, item=row: self._lut_changed(item)
        )
        region.sigRegionChanged.connect(
            lambda _region, item=row: self._region_changed(item)
        )

    def _spin_changed(self, row: dict) -> None:
        if self._synchronizing:
            return
        self._synchronizing = True
        row["region"].setRegion((row["start"].value(), row["end"].value()))
        self._synchronizing = False
        self._refresh_counts()

    def _region_changed(self, row: dict) -> None:
        if self._synchronizing:
            return
        lo, hi = sorted(float(value) for value in row["region"].getRegion())
        self._synchronizing = True
        row["start"].setValue(lo)
        row["end"].setValue(hi)
        self._synchronizing = False
        self._refresh_counts()

    def _lut_changed(self, row: dict) -> None:
        rgb = _lut_preview_rgb(row["lut"].currentText())
        row["region"].setBrush((*rgb, 35))
        row["region"].setPen(pg.mkPen(rgb, width=2))
        row["region"].setHoverPen(pg.mkPen(rgb, width=3))

    def refresh_colors(self) -> None:
        choices = channel_colormap_names()
        for row in self._rows:
            combo = row["lut"]
            current = combo.currentText()
            blocked = combo.blockSignals(True)
            try:
                combo.clear()
                combo.addItems(choices)
                if current in choices:
                    combo.setCurrentText(current)
            finally:
                combo.blockSignals(blocked)
            self._lut_changed(row)

    def _add_by_splitting(self) -> None:
        if not self._rows:
            self._set_even_windows(2)
            return
        selected = self._table.currentRow()
        index = selected if 0 <= selected < len(self._rows) else len(self._rows) - 1
        row = self._rows[index]
        start_s = row["start"].value() * 60.0
        end_s = row["end"].value() * 60.0
        midpoint = (start_s + end_s) / 2.0
        row["end"].setValue(midpoint / 60.0)
        new_index = len(self._rows)
        self._append_row(
            name=f"{self._source_name} [time {new_index + 1}]",
            start_s=midpoint,
            end_s=end_s,
            lut=self._color_cycle[new_index % len(self._color_cycle)],
        )
        self._window_count.setValue(len(self._rows))
        self._refresh_counts()

    def _remove_selected(self) -> None:
        selected = sorted({index.row() for index in self._table.selectionModel().selectedRows()})
        if not selected and self._table.currentRow() >= 0:
            selected = [self._table.currentRow()]
        if len(self._rows) - len(selected) < 2:
            QMessageBox.information(
                self,
                "Time windows",
                "At least two windows are required to create separate channels.",
            )
            return
        for index in reversed(selected):
            self._plot.removeItem(self._rows[index]["region"])
            self._rows.pop(index)
            self._table.removeRow(index)
        self._window_count.setValue(len(self._rows))
        self._refresh_counts()

    def _current_windows(self) -> list[TimeWindow]:
        return [
            TimeWindow(
                name=row["name"].text().strip(),
                start_s=float(row["start"].value() * 60.0),
                end_s=float(row["end"].value() * 60.0),
                lut=row["lut"].currentText(),
            )
            for row in self._rows
        ]

    def _refresh_counts(self) -> None:
        if not self._rows:
            return
        try:
            selections = time_channel_selections(
                self._tim_s,
                self._current_windows(),
                base_mask=self._base_mask,
            )
        except ValueError as exc:
            for row in self._rows:
                row["count"].setText("-")
            self._status.setText(str(exc))
            self._status.setStyleSheet("color: #b04a28;")
            return

        count_by_name = {
            selection.window.name: int(selection.mask.sum())
            for selection in selections
        }
        union = np.zeros(self._tim_s.size, dtype=bool)
        for selection in selections:
            union |= selection.mask
        for row in self._rows:
            row["count"].setText(f"{count_by_name.get(row['name'].text().strip(), 0):,}")
        outside = int((self._base_mask & np.isfinite(self._tim_s) & ~union).sum())
        assigned = int(union.sum())
        self._status.setText(
            f"{assigned:,} assigned to {len(selections)} channels; "
            f"{outside:,} outside the windows"
        )
        self._status.setStyleSheet("")

    def windows(self) -> list[TimeWindow]:
        """Chronologically ordered values, in canonical seconds."""
        if self._accepted_windows is not None:
            return list(self._accepted_windows)
        selections = time_channel_selections(
            self._tim_s,
            self._current_windows(),
            base_mask=self._base_mask,
        )
        return [selection.window for selection in selections]

    def accept(self) -> None:
        try:
            selections = time_channel_selections(
                self._tim_s,
                self._current_windows(),
                base_mask=self._base_mask,
            )
        except ValueError as exc:
            QMessageBox.warning(self, "Time windows", str(exc))
            return
        empty = [item.window.name for item in selections if not np.any(item.mask)]
        if empty:
            QMessageBox.warning(
                self,
                "Time windows",
                "These windows contain no active localizations:\n"
                + "\n".join(empty),
            )
            return
        self._accepted_windows = [item.window for item in selections]
        super().accept()
