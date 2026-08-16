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

from ..colormaps import channel_colormap_names
from ..core.app_state import AppState


class ChannelCombineDialog(QDialog):
    """Pick arbitrary loaded datasets and create a display overlay.

    ``dataset_indices`` restricts the table to those datasets (the Dataset
    Manager's multi-selection *Combine as multi-channel overlay*), all of them
    pre-checked; ``None`` lists every loaded dataset.
    """

    def __init__(
        self,
        state: AppState,
        *,
        previous: dict | None = None,
        dataset_indices=None,
        parent=None,
    ) -> None:
        super().__init__(parent)
        self._state = state
        self._checks: list[QCheckBox] = []
        self._orders: list[QComboBox] = []
        self._luts: list[QComboBox] = []
        self._previous = previous or {}

        # Dataset index shown on each table row (identity, not position).
        restricted = dataset_indices is not None
        if restricted:
            self._rows = [i for i in dataset_indices if 0 <= i < len(state.datasets)]
        else:
            self._rows = list(range(len(state.datasets)))

        self.setWindowTitle("Combine datasets")
        self.resize(860, 420)
        root = QVBoxLayout(self)
        root.setContentsMargins(8, 8, 8, 8)
        root.setSpacing(8)

        self._table = QTableWidget(len(self._rows), 5, self)
        self._table.setHorizontalHeaderLabels(["Include", "Dataset", "Dims / locs / loaded", "Order", "LUT / color"])
        self._table.verticalHeader().setVisible(False)
        self._table.setSelectionBehavior(QAbstractItemView.SelectionBehavior.SelectRows)
        hh = self._table.horizontalHeader()
        hh.setSectionResizeMode(QHeaderView.ResizeMode.ResizeToContents)
        hh.setSectionResizeMode(1, QHeaderView.ResizeMode.Stretch)
        root.addWidget(self._table, stretch=1)

        order_values = [str(i + 1) for i in range(max(1, len(self._rows)))]
        selected = set(self._previous.get("selected", []))
        orders = self._previous.get("orders", {})
        luts = self._previous.get("luts", {})
        channel_luts = channel_colormap_names()
        for row, idx in enumerate(self._rows):
            ds = state.datasets[idx]
            chk = QCheckBox()
            # A restricted list *is* the user's selection — check all of it.
            chk.setChecked(True if restricted else (idx in selected if selected else idx < 2))
            self._table.setCellWidget(row, 0, chk)
            self._checks.append(chk)

            name = QTableWidgetItem(ds.name)
            name.setFlags(name.flags() & ~Qt.ItemFlag.ItemIsEditable)
            self._table.setItem(row, 1, name)
            summary = QTableWidgetItem(
                f"{ds.prop.num_dim}D / {ds.prop.num_loc:,} locs / {ds.file.datetime or '-'}"
            )
            summary.setFlags(summary.flags() & ~Qt.ItemFlag.ItemIsEditable)
            self._table.setItem(row, 2, summary)

            order = QComboBox()
            order.addItems(order_values)
            # Remembered orders/LUTs are keyed by dataset index over the *full*
            # list, so they only apply when the full list is shown.
            order.setCurrentText(str(row + 1) if restricted else str(orders.get(idx, idx + 1)))
            self._table.setCellWidget(row, 3, order)
            self._orders.append(order)

            lut = QComboBox()
            lut.addItems(channel_luts)
            default_lut = channel_luts[row % len(channel_luts)]
            lut.setCurrentText(
                default_lut if restricted
                else str(luts.get(idx, channel_luts[idx % len(channel_luts)]))
            )
            self._table.setCellWidget(row, 4, lut)
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
            "selected": [self._rows[r] for r, chk in enumerate(self._checks) if chk.isChecked()],
            "orders": {self._rows[r]: int(c.currentText()) for r, c in enumerate(self._orders)},
            "luts": {self._rows[r]: c.currentText() for r, c in enumerate(self._luts)},
            "align_with": self.align_combo.currentText(),
            "keep_source": self.keep_source_check.isChecked(),
        }

    def selected_rows(self) -> list[dict]:
        rows = []
        for row, chk in enumerate(self._checks):
            if not chk.isChecked():
                continue
            rows.append({
                "dataset_idx": self._rows[row],
                "order": int(self._orders[row].currentText()),
                "lut": self._luts[row].currentText(),
            })
        return sorted(rows, key=lambda row: (row["order"], row["dataset_idx"]))
