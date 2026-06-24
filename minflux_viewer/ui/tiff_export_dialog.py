"""Parameter dialog + background worker for "Export to TIFF…".

The dialog collects the output path, XY pixel size and (for 3-D data) Z voxel
depth; the actual histogram binning and TIFF writing run on a
:class:`TiffExportWorker` QThread so the viewer stays responsive. Progress and
completion are reported to the application Log — there is no progress bar and no
completion pop-up (by design).
"""

from __future__ import annotations

from pathlib import Path

from PyQt6.QtCore import QThread, pyqtSignal
from PyQt6.QtWidgets import (
    QDialog,
    QDialogButtonBox,
    QDoubleSpinBox,
    QFileDialog,
    QFormLayout,
    QHBoxLayout,
    QLabel,
    QLineEdit,
    QPushButton,
    QVBoxLayout,
    QWidget,
)

from ..core.tiff_export import TiffExportChannel, export_render_to_tiff


def _coord_spin(value: float) -> QDoubleSpinBox:
    spin = QDoubleSpinBox()
    spin.setDecimals(1)
    spin.setRange(-1.0e9, 1.0e9)
    spin.setSingleStep(10.0)
    spin.setValue(float(value))
    spin.setSuffix(" nm")
    return spin


class TiffExportDialog(QDialog):
    """Modal dialog for the TIFF export parameters.

    Collects the output path, XY pixel size, Z voxel depth, the XY/Z export
    range and (for 3-D data) the editable RIMF z-scaling factor. The XY range is
    pre-filled from the active rectangle ROI when present, otherwise the data
    extent. Editing RIMF rescales the Z-range fields so they keep tracking the
    same physical structure (the new RIMF is applied to the dataset on export).
    """

    def __init__(
        self,
        *,
        default_path: str,
        default_pixel_nm: float,
        default_voxel_nm: float,
        is_3d: bool,
        x_span: tuple[float, float],
        y_span: tuple[float, float],
        z_span: tuple[float, float],
        rimf: float | None = None,
        xy_from_roi: bool = False,
        parent: QWidget | None = None,
    ) -> None:
        super().__init__(parent)
        self.setWindowTitle("Export to TIFF")
        self.setModal(True)
        self._is_3d = bool(is_3d)
        self._rimf_current = float(rimf) if rimf else None

        root = QVBoxLayout(self)
        form = QFormLayout()
        root.addLayout(form)

        self._path_edit = QLineEdit(default_path)
        browse = QPushButton("Browse…")
        browse.clicked.connect(self._browse)
        path_row = QHBoxLayout()
        path_row.addWidget(self._path_edit, 1)
        path_row.addWidget(browse)
        path_box = QWidget()
        path_box.setLayout(path_row)
        form.addRow("Output file:", path_box)

        self._pixel_spin = QDoubleSpinBox()
        self._pixel_spin.setDecimals(2)
        self._pixel_spin.setRange(0.10, 10000.0)
        self._pixel_spin.setSingleStep(1.0)
        self._pixel_spin.setValue(float(default_pixel_nm))
        self._pixel_spin.setSuffix(" nm")
        form.addRow("Pixel size (XY):", self._pixel_spin)

        self._voxel_spin = QDoubleSpinBox()
        self._voxel_spin.setDecimals(2)
        self._voxel_spin.setRange(0.10, 10000.0)
        self._voxel_spin.setSingleStep(1.0)
        self._voxel_spin.setValue(float(default_voxel_nm))
        self._voxel_spin.setSuffix(" nm")
        self._voxel_spin.setEnabled(self._is_3d)
        if not self._is_3d:
            self._voxel_spin.setToolTip("2-D dataset: a single Z slice is exported.")
        form.addRow("Voxel depth (Z):", self._voxel_spin)

        self._rimf_spin: QDoubleSpinBox | None = None
        if self._is_3d and self._rimf_current:
            self._rimf_spin = QDoubleSpinBox()
            self._rimf_spin.setDecimals(4)
            self._rimf_spin.setRange(0.1000, 2.0000)
            self._rimf_spin.setSingleStep(0.01)
            self._rimf_spin.setValue(self._rimf_current)
            self._rimf_spin.setToolTip(
                "Refractive Index Mismatch Factor (z-scaling). Changing it rescales "
                "the Z range and is applied to the dataset on export."
            )
            self._rimf_spin.valueChanged.connect(self._on_rimf_changed)
            form.addRow("RIMF (z-scaling):", self._rimf_spin)

        self._x_lo, self._x_hi = self._range_row(form, "X range:", x_span)
        self._y_lo, self._y_hi = self._range_row(form, "Y range:", y_span)
        self._z_lo, self._z_hi = self._range_row(form, "Z range:", z_span, enabled=self._is_3d)

        note = QLabel(
            "XY range from active rectangle ROI."
            if xy_from_roi
            else "XY range from data extent."
        )
        note.setStyleSheet("color: #666;")
        root.addWidget(note)

        buttons = QDialogButtonBox(
            QDialogButtonBox.StandardButton.Ok | QDialogButtonBox.StandardButton.Cancel
        )
        buttons.button(QDialogButtonBox.StandardButton.Ok).setText("Export")
        buttons.accepted.connect(self.accept)
        buttons.rejected.connect(self.reject)
        root.addWidget(buttons)

    def _range_row(self, form, label, span, *, enabled: bool = True):
        lo = _coord_spin(span[0])
        hi = _coord_spin(span[1])
        lo.setEnabled(enabled)
        hi.setEnabled(enabled)
        row = QHBoxLayout()
        row.addWidget(lo, 1)
        row.addWidget(QLabel("to"))
        row.addWidget(hi, 1)
        box = QWidget()
        box.setLayout(row)
        form.addRow(label, box)
        return lo, hi

    def _on_rimf_changed(self, value: float) -> None:
        old = self._rimf_current or 0.0
        if old <= 0.0 or value <= 0.0:
            self._rimf_current = float(value)
            return
        scale = float(value) / old
        self._z_lo.setValue(self._z_lo.value() * scale)
        self._z_hi.setValue(self._z_hi.value() * scale)
        self._rimf_current = float(value)

    def _browse(self) -> None:
        current = self._path_edit.text().strip()
        path, _ = QFileDialog.getSaveFileName(
            self,
            "Export render to TIFF",
            current,
            "OME-TIFF (*.ome.tif *.ome.tiff *.tif *.tiff);;All files (*)",
        )
        if path:
            self._path_edit.setText(path)

    def params(self) -> dict:
        return {
            "path": self._path_edit.text().strip(),
            "pixel_size_nm": float(self._pixel_spin.value()),
            "voxel_depth_nm": float(self._voxel_spin.value()),
            "rimf": float(self._rimf_spin.value()) if self._rimf_spin is not None else None,
            "x_range": (float(self._x_lo.value()), float(self._x_hi.value())),
            "y_range": (float(self._y_lo.value()), float(self._y_hi.value())),
            "z_range": (
                (float(self._z_lo.value()), float(self._z_hi.value()))
                if self._is_3d
                else None
            ),
        }


class TiffExportWorker(QThread):
    """Runs :func:`export_render_to_tiff` off the UI thread.

    Emits :attr:`progress` (throttled to ~10 % steps), :attr:`completed` with the
    :class:`~minflux_viewer.core.tiff_export.TiffExportResult`, or :attr:`failed`
    with an error string.
    """

    progress = pyqtSignal(str)
    completed = pyqtSignal(object)
    failed = pyqtSignal(str)

    def __init__(
        self,
        channels: list[TiffExportChannel],
        path: str,
        *,
        pixel_size_nm: float,
        voxel_depth_nm: float,
        is_3d: bool,
        x_range: tuple[float, float] | None = None,
        y_range: tuple[float, float] | None = None,
        z_range: tuple[float, float] | None = None,
        parent=None,
    ) -> None:
        super().__init__(parent)
        self._channels = channels
        self._path = path
        self._pixel_size_nm = pixel_size_nm
        self._voxel_depth_nm = voxel_depth_nm
        self._is_3d = is_3d
        self._x_range = x_range
        self._y_range = y_range
        self._z_range = z_range

    def run(self) -> None:
        last_pct = [-1]

        def report(done: int, total: int, message: str) -> None:
            pct = int(done * 100 / max(total, 1))
            if pct >= last_pct[0] + 10 or done >= total:
                last_pct[0] = pct
                self.progress.emit(
                    f"Export to TIFF: {pct}% ({done}/{total} slices) — {message}"
                )

        try:
            result = export_render_to_tiff(
                self._channels,
                self._path,
                pixel_size_nm=self._pixel_size_nm,
                voxel_depth_nm=self._voxel_depth_nm,
                is_3d=self._is_3d,
                x_range=self._x_range,
                y_range=self._y_range,
                z_range=self._z_range,
                progress=report,
            )
            self.completed.emit(result)
        except Exception as exc:  # noqa: BLE001 - surfaced to the Log, never crashes the app
            self.failed.emit(str(exc))
