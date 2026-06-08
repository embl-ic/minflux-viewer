from __future__ import annotations

from PyQt6.QtCore import Qt
from PyQt6.QtWidgets import (
    QAbstractItemView,
    QCheckBox,
    QComboBox,
    QDialog,
    QDialogButtonBox,
    QHBoxLayout,
    QHeaderView,
    QLabel,
    QTableWidget,
    QTableWidgetItem,
    QVBoxLayout,
)

from ..core.app_state import AppState
from ..core.overlay import CHANNEL_LUTS


class ChannelCombineDialog(QDialog):
    """Pick arbitrary loaded datasets and create a display overlay."""

    def __init__(self, state: AppState, *, previous: dict | None = None, parent=None) -> None:
        super().__init__(parent)
        self._state = state
        self._checks: list[QCheckBox] = []
        self._orders: list[QComboBox] = []
        self._luts: list[QComboBox] = []
        self._previous = previous or {}

        self.setWindowTitle("Combine datasets")
        self.resize(860, 420)
        root = QVBoxLayout(self)
        root.setContentsMargins(8, 8, 8, 8)
        root.setSpacing(8)

        self._table = QTableWidget(len(state.datasets), 5, self)
        self._table.setHorizontalHeaderLabels(["Include", "Dataset", "Dims / locs / loaded", "Order", "LUT / color"])
        self._table.verticalHeader().setVisible(False)
        self._table.setSelectionBehavior(QAbstractItemView.SelectionBehavior.SelectRows)
        hh = self._table.horizontalHeader()
        hh.setSectionResizeMode(QHeaderView.ResizeMode.ResizeToContents)
        hh.setSectionResizeMode(1, QHeaderView.ResizeMode.Stretch)
        root.addWidget(self._table, stretch=1)

        order_values = [str(i + 1) for i in range(max(1, len(state.datasets)))]
        selected = set(self._previous.get("selected", []))
        orders = self._previous.get("orders", {})
        luts = self._previous.get("luts", {})
        for idx, ds in enumerate(state.datasets):
            chk = QCheckBox()
            chk.setChecked(idx in selected if selected else idx < 2)
            self._table.setCellWidget(idx, 0, chk)
            self._checks.append(chk)

            name = QTableWidgetItem(ds.name)
            name.setFlags(name.flags() & ~Qt.ItemFlag.ItemIsEditable)
            self._table.setItem(idx, 1, name)
            summary = QTableWidgetItem(
                f"{ds.prop.num_dim}D / {ds.prop.num_loc:,} locs / {ds.file.datetime or '-'}"
            )
            summary.setFlags(summary.flags() & ~Qt.ItemFlag.ItemIsEditable)
            self._table.setItem(idx, 2, summary)

            order = QComboBox()
            order.addItems(order_values)
            order.setCurrentText(str(orders.get(idx, idx + 1)))
            self._table.setCellWidget(idx, 3, order)
            self._orders.append(order)

            lut = QComboBox()
            lut.addItems(CHANNEL_LUTS)
            lut.setCurrentText(str(luts.get(idx, CHANNEL_LUTS[idx % len(CHANNEL_LUTS)])))
            self._table.setCellWidget(idx, 4, lut)
            self._luts.append(lut)

        bottom = QHBoxLayout()
        bottom.addWidget(QLabel("align with:"))
        self.align_combo = QComboBox()
        self.align_combo.addItems(["mbm info if available", "stage origin", "data centroid"])
        align = self._previous.get("align_with", "mbm info if available")
        if self.align_combo.findText(align) >= 0:
            self.align_combo.setCurrentText(align)
        bottom.addWidget(self.align_combo)
        bottom.addStretch(1)
        self.keep_source_check = QCheckBox("keep source dataset")
        self.keep_source_check.setChecked(bool(self._previous.get("keep_source", True)))
        bottom.addWidget(self.keep_source_check)
        root.addLayout(bottom)

        buttons = QDialogButtonBox(QDialogButtonBox.StandardButton.Ok | QDialogButtonBox.StandardButton.Cancel)
        buttons.accepted.connect(self.accept)
        buttons.rejected.connect(self.reject)
        root.addWidget(buttons)

    def session_state(self) -> dict:
        return {
            "selected": [i for i, chk in enumerate(self._checks) if chk.isChecked()],
            "orders": {i: int(combo.currentText()) for i, combo in enumerate(self._orders)},
            "luts": {i: combo.currentText() for i, combo in enumerate(self._luts)},
            "align_with": self.align_combo.currentText(),
            "keep_source": self.keep_source_check.isChecked(),
        }

    def selected_rows(self) -> list[dict]:
        rows = []
        for idx, chk in enumerate(self._checks):
            if not chk.isChecked():
                continue
            rows.append({
                "dataset_idx": idx,
                "order": int(self._orders[idx].currentText()),
                "lut": self._luts[idx].currentText(),
            })
        return sorted(rows, key=lambda row: (row["order"], row["dataset_idx"]))
