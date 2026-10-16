'Alignment save dialog'

from __future__ import annotations

from PyQt6.QtWidgets import (
    QCheckBox,
    QDialog,
    QDialogButtonBox,
    QFileDialog,
    QGridLayout,
    QLabel,
    QLineEdit,
    QPushButton,
    QVBoxLayout,
)


class AlignmentSaveDialog(QDialog):
    def __init__(self, initial_paths: dict[str, str], initial_options: dict | None = None, parent=None):
        super().__init__(parent)
        self.setWindowTitle("Save channel alignment outputs")
        self.resize(820, 240)

        # Remembered checkbox states from the last run (persisted in settings).
        opts = initial_options or {}
        self.save_beads = QCheckBox("Matched bead summary CSV")
        self.save_beads.setChecked(bool(opts.get("save_beads", True)))
        self.save_transform = QCheckBox("Alignment transform CSV")
        self.save_transform.setChecked(bool(opts.get("save_transform", True)))
        self.show_plot = QCheckBox("Show beads and alignment result")
        self.show_plot.setChecked(bool(opts.get("show_plot", False)))

        self.path_beads = QLineEdit(initial_paths["beads"])
        self.path_transform = QLineEdit(initial_paths["transform"])

        layout = QVBoxLayout(self)
        hint = QLabel(
            "Use <b>Show beads drift</b> to manually check bead drift and select "
            "beads for alignment — otherwise all common beads are used to compute "
            "the alignment transformation.")
        hint.setWordWrap(True)
        hint.setStyleSheet("color: #555;")
        layout.addWidget(hint)
        grid = QGridLayout()
        layout.addLayout(grid)
        self._add_path_row(grid, 0, self.save_beads, self.path_beads)
        self._add_path_row(grid, 1, self.save_transform, self.path_transform)
        layout.addWidget(self.show_plot)

        buttons = QDialogButtonBox(QDialogButtonBox.StandardButton.Ok | QDialogButtonBox.StandardButton.Cancel)
        buttons.accepted.connect(self.accept)
        buttons.rejected.connect(self.reject)
        layout.addWidget(buttons)

    def _add_path_row(self, grid, row: int, checkbox, lineedit):
        button = QPushButton("Browse…")
        button.clicked.connect(lambda: self._browse(lineedit))
        grid.addWidget(checkbox, row, 0)
        grid.addWidget(lineedit, row, 1)
        grid.addWidget(button, row, 2)

    def _browse(self, lineedit):
        current = lineedit.text().strip()
        path, _ = QFileDialog.getSaveFileName(
            self,
            "Choose output CSV",
            current or "",
            "CSV (*.csv);;All files (*.*)",
        )
        if path:
            lineedit.setText(path)

    def result_payload(self):
        return {
            "save_beads": self.save_beads.isChecked(),
            "save_transform": self.save_transform.isChecked(),
            "show_plot": self.show_plot.isChecked(),
            "path_beads": self.path_beads.text().strip(),
            "path_transform": self.path_transform.text().strip(),
        }
