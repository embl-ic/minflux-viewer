'Array preview dialog'

from __future__ import annotations

import numpy as np
from PyQt6.QtWidgets import (
    QDialog,
    QDialogButtonBox,
    QTableWidget,
    QTableWidgetItem,
    QVBoxLayout,
)


class ArrayPreviewDialog(QDialog):
    def __init__(self, title: str, array, preview_rows: int | None, parent=None):
        super().__init__(parent)
        self.setWindowTitle(title)
        self.resize(900, 500)

        layout = QVBoxLayout(self)
        table = QTableWidget(self)
        layout.addWidget(table)

        arr = np.asarray(array)
        total_rows = arr.shape[0] if arr.ndim >= 1 else 0
        n = total_rows if preview_rows is None else min(preview_rows, total_rows)
        if arr.ndim == 1:
            arr2 = arr[:n].reshape(-1, 1)
        elif arr.ndim > 2:
            arr2 = arr[:n].reshape(n, -1)
        else:
            arr2 = arr[:n]

        table.setRowCount(n)
        table.setColumnCount(arr2.shape[1] if n else 1)
        table.setHorizontalHeaderLabels([str(i) for i in range(table.columnCount())])
        for row in range(n):
            for col in range(arr2.shape[1]):
                table.setItem(row, col, QTableWidgetItem(str(arr2[row, col])))
        table.resizeColumnsToContents()

        buttons = QDialogButtonBox(QDialogButtonBox.StandardButton.Close)
        buttons.rejected.connect(self.reject)
        buttons.button(QDialogButtonBox.StandardButton.Close).clicked.connect(self.close)
        layout.addWidget(buttons)
