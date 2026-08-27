"""
minflux_viewer.ui.data_window
==============================
Per-dataset floating info window.

Each :class:`MinfluxDataset` that is loaded gets one :class:`DataWindow`.
Clicking or focusing the window activates that dataset in
:class:`~minflux_viewer.core.app_state.AppState` — exactly the Fiji
"click the image window to work on it" behaviour.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
from PyQt6.QtCore import QEvent, Qt
from PyQt6.QtGui import QFont
from PyQt6.QtWidgets import (
    QDialog,
    QFrame,
    QGridLayout,
    QLabel,
    QMessageBox,
    QPushButton,
    QSizePolicy,
    QWidget,
)

from ..core.app_state import AppState
from ..core.dataset import MinfluxDataset
from .z_scaling_widgets import ZScalingFactorDialog, format_z_scaling_factor


class DataWindow(QWidget):
    """
    Compact info card for one dataset.

    ::

        ┌──────────────────────────────────────────────────────────────┐
        │  filename.mat                                                │
        ├──────────────────────────────────────────────────────────────┤
        │  Folder        …/experiment/day1/                            │
        │  Acquisition   2025-Apr-16,09:31:02 ~ 10:25:11 (span 54 min) │
        │  File created  2025-Apr-16, 11:02:44                         │
        │  Locs          45,231                                        │
        │  Traces        1,204                                         │
        │  Dims          3D  |  5 iterations                           │
        │                                                              │
        │  [ Set as active ]                                           │
        └──────────────────────────────────────────────────────────────┘
    """

    def __init__(
        self,
        dataset: MinfluxDataset,
        dataset_idx: int,
        state: AppState,
        parent: QWidget | None = None,
    ) -> None:
        super().__init__(parent)
        self._idx   = dataset_idx
        self._state = state

        self.setWindowTitle("Dataset Information")
        self.setMinimumWidth(420)
        self.setMaximumWidth(560)
        self.setSizePolicy(QSizePolicy.Policy.Fixed, QSizePolicy.Policy.Fixed)
        self.setWindowFlags(Qt.WindowType.Window)

        self._build_ui(dataset)

        # Track active-dataset changes so we can update the title / button
        state.active_changed.connect(self._refresh)
        # Refresh info values (e.g. Z scaling factor) when this dataset's calibration changes
        state.calibration_changed.connect(self._on_calibration_changed)
        state.overlay_transform_changed.connect(self._on_overlay_transform_changed)

    # ------------------------------------------------------------------
    # UI
    # ------------------------------------------------------------------

    def _build_ui(self, ds: MinfluxDataset) -> None:
        layout = QGridLayout(self)
        layout.setContentsMargins(14, 12, 14, 14)
        layout.setVerticalSpacing(4)
        layout.setHorizontalSpacing(12)
        layout.setColumnStretch(1, 1)

        row = 0

        # ── Header ──────────────────────────────────────────────────
        header = QLabel(_display_name(ds))
        font   = QFont()
        font.setBold(True)
        font.setPointSize(11)
        header.setFont(font)
        header.setWordWrap(True)
        header.setMaximumWidth(520)
        layout.addWidget(header, row, 0, 1, 2)
        row += 1

        # ── Separator ───────────────────────────────────────────────
        sep = QFrame()
        sep.setFrameShape(QFrame.Shape.HLine)
        sep.setFrameShadow(QFrame.Shadow.Sunken)
        layout.addWidget(sep, row, 0, 1, 2)
        row += 1

        # ── Info rows ───────────────────────────────────────────────
        self._value_labels: dict[str, QLabel] = {}
        self._z_scaling_widgets: list[QLabel] = []
        self._transform_widgets: list[QLabel] = []
        info_rows = _info_rows(ds)
        for label, value in info_rows:
            key_lbl = QLabel(label)
            key_lbl.setStyleSheet("color: gray; font-size: 11px;")
            val_lbl = QLabel(value)
            val_lbl.setWordWrap(True)
            val_lbl.setMaximumWidth(430)
            val_lbl.setStyleSheet("font-size: 11px;")
            layout.addWidget(key_lbl, row, 0)
            layout.addWidget(val_lbl, row, 1)
            self._value_labels[label] = val_lbl
            if label == "Dims" and int(ds.prop.num_dim) >= 3:
                tip = "Double-click to set this dataset's Z scaling factor manually."
                for widget in (key_lbl, val_lbl):
                    widget.setCursor(Qt.CursorShape.PointingHandCursor)
                    widget.setToolTip(tip)
                    widget.installEventFilter(self)
                    self._z_scaling_widgets.append(widget)
            if label == "Transformed":
                for widget in (key_lbl, val_lbl):
                    widget.installEventFilter(self)
                    self._transform_widgets.append(widget)
            row += 1

        self._update_transform_row_interactivity(ds)

        # ── Spacer ──────────────────────────────────────────────────
        layout.setRowMinimumHeight(row, 6)
        row += 1

        # ── Activate button ─────────────────────────────────────────
        self._btn = QPushButton("Set as active")
        self._btn.clicked.connect(self._activate)
        layout.addWidget(self._btn, row, 0, 1, 2)

        from .text_select import make_labels_selectable
        make_labels_selectable(self)   # copyable name / path / values

        self._refresh(self._state.active_idx)

    # ------------------------------------------------------------------
    # Interaction
    # ------------------------------------------------------------------

    def _activate(self) -> None:
        self._state.set_active(self._idx)

    def eventFilter(self, watched, event) -> bool:
        """Handle the editable Dataset Information rows."""
        if (
            watched in self._z_scaling_widgets
            and event.type() == QEvent.Type.MouseButtonDblClick
        ):
            self._edit_z_scaling_factor()
            event.accept()
            return True
        if (
            watched in self._transform_widgets
            and event.type() == QEvent.Type.MouseButtonDblClick
            and 0 <= self._idx < len(self._state.datasets)
            and _is_transformed(self._state.datasets[self._idx])
        ):
            self._edit_transform()
            event.accept()
            return True
        return super().eventFilter(watched, event)

    def _edit_z_scaling_factor(self) -> None:
        """Prompt for and apply a manual Z scaling factor to this 3-D dataset."""
        if not (0 <= self._idx < len(self._state.datasets)):
            return
        ds = self._state.datasets[self._idx]
        if int(ds.prop.num_dim) < 3:
            return

        self._state.set_active(self._idx)
        current = float(getattr(ds.cali, "z_scaling_factor", 1.0) or 1.0)
        # Not QInputDialog.getDouble: its spin box uses one step for both
        # the arrows and the wheel, and this factor wants a fine arrow step
        # with a coarse wheel sweep.
        value, accepted = ZScalingFactorDialog.ask(self, current)
        if not accepted or np.isclose(value, current, rtol=0.0, atol=1e-12):
            return

        ds.set_z_scaling_factor(value, source="manual (Dataset Information)")
        self._state.log(
            f"Z scaling factor for '{ds.name}': "
            f"{format_z_scaling_factor(value)} "
            "(manual, Dataset Information).",
            dataset_idx=self._idx,
        )
        self._state.notify_calibration_changed(self._idx)

    def _edit_transform(self) -> None:
        """Inspect/edit this dataset's canonical overlay transform."""
        if not (0 <= self._idx < len(self._state.datasets)):
            return
        ds = self._state.datasets[self._idx]
        transform = _dataset_transform(ds)
        if transform is None:
            QMessageBox.information(
                self,
                "Dataset Transform",
                "This dataset is marked as transformed, but no editable transform matrix is available.",
            )
            return

        from ..core.overlay import is_multichannel_overlay
        from .transform_dialog import TransformDialog

        self._state.set_active(self._idx)
        plot_prefs = self._state.prefs.get("plot", {})
        xy_origin_top_left = str(
            plot_prefs.get("render_xy_origin", "top_left")
        ).lower() == "top_left"
        dialog = TransformDialog(
            transform,
            dataset_name=ds.name,
            xy_origin_top_left=xy_origin_top_left,
            manual_align_enabled=is_multichannel_overlay(self._state, self._idx),
            parent=self,
        )
        result = dialog.exec()
        if dialog.manual_alignment_requested:
            self._state.request_overlay_manual_alignment(self._idx)
            return
        if result != QDialog.DialogCode.Accepted:
            return

        record = dialog.updated_record()
        ds.state["overlay_transform"] = record
        ds.state["render_transform_2d"] = record
        matrix = np.asarray(record["matrix_4x4"], dtype=float)
        translation = matrix[:3, 3]
        self._state.log(
            f"Transform matrix for '{ds.name}' edited in Dataset Information "
            f"(translation X {translation[0]:+.4g}, Y {translation[1]:+.4g}, "
            f"Z {translation[2]:+.4g} nm).",
            dataset_idx=self._idx,
        )
        self._state.notify_overlay_transform_changed(self._idx)

    # ------------------------------------------------------------------
    # Fiji-style active-dataset-follows-focus (requirement #2)
    # ------------------------------------------------------------------

    def focusInEvent(self, event) -> None:
        if 0 <= self._idx < len(self._state.datasets):
            self._state.set_active(self._idx)
        super().focusInEvent(event)

    def changeEvent(self, event) -> None:
        from PyQt6.QtCore import QEvent
        if event.type() == QEvent.Type.ActivationChange and self.isActiveWindow():
            if 0 <= self._idx < len(self._state.datasets):
                self._state.set_active(self._idx)
        super().changeEvent(event)

    def _on_calibration_changed(self, idx: int) -> None:
        """Refresh info values (e.g. the Z scaling factor row) for this dataset."""
        if idx != self._idx or not (0 <= self._idx < len(self._state.datasets)):
            return
        self._refresh_info_values()

    def _on_overlay_transform_changed(self, idx: int) -> None:
        """Refresh the transformed value after matrix or interactive alignment edits."""
        if idx != self._idx or not (0 <= self._idx < len(self._state.datasets)):
            return
        self._refresh_info_values()

    def _refresh_info_values(self) -> None:
        ds = self._state.datasets[self._idx]
        for label, value in _info_rows(ds):
            lbl = self._value_labels.get(label)
            if lbl is not None:
                lbl.setText(value)
        self._update_transform_row_interactivity(ds)

    def _update_transform_row_interactivity(self, ds: MinfluxDataset) -> None:
        transformed = _is_transformed(ds)
        tip = (
            "Double-click to inspect or edit this dataset's transform matrix."
            if transformed
            else "No coordinate-changing transform is currently applied."
        )
        cursor = (
            Qt.CursorShape.PointingHandCursor
            if transformed
            else Qt.CursorShape.ArrowCursor
        )
        for widget in self._transform_widgets:
            widget.setCursor(cursor)
            widget.setToolTip(tip)

    def _refresh(self, active_idx: int | None) -> None:
        """Update the window title and button state."""
        if active_idx == self._idx:
            self.setWindowTitle("[ACTIVE]  Dataset Information")
            self._btn.setEnabled(False)
            self._btn.setText("Active dataset")
        else:
            self.setWindowTitle("Dataset Information")
            self._btn.setEnabled(True)
            self._btn.setText("Set as active")


# ---------------------------------------------------------------------------
# Helper
# ---------------------------------------------------------------------------

def _display_name(ds: MinfluxDataset) -> str:
    """Return the user-facing dataset name shown below the dialog title."""
    if ds.metadata.get("msr_source_path") and ds.metadata.get("msr_dataset_key"):
        msr_name = Path(str(ds.metadata["msr_source_path"])).name
        return f"{msr_name} | {ds.metadata['msr_dataset_key']}"
    return ds.file.name


def _info_rows(ds: MinfluxDataset) -> list[tuple[str, str]]:
    locs_label, locs_value = _locs_row(ds)
    return [
        ("Folder", _folder_text(ds)),
        ("Acquisition", _acquisition_text(ds)),
        ("File created", _created_text(ds)),
        (locs_label, locs_value),
        ("Iterations", _iterations_text(ds)),
        ("Traces", f"{int(ds.prop.num_traces):,}"),
        ("Dims", _dims_text(ds)),
        ("Version", _version_text(ds)),
        ("Transformed", "yes" if _is_transformed(ds) else "no"),
    ]


def _folder_text(ds: MinfluxDataset) -> str:
    source = ds.metadata.get("msr_source_path") or ds.file.recent_path
    if source:
        try:
            return str(Path(str(source)).resolve().parent)
        except Exception:
            pass
    return str(Path(ds.file.folder).resolve()) if ds.file.folder else "—"


def _acquisition_text(ds: MinfluxDataset) -> str:
    """When the instrument recorded this dataset — start, end and duration.

    Distinct from ``File created``: the file may be written, converted or copied
    long after (or, for a multi-run ``.msr``, hours after the first run).
    """
    from ..core.acquisition_time import dataset_acquisition_text

    return dataset_acquisition_text(ds)


def _created_text(ds: MinfluxDataset) -> str:
    note = ds.metadata.get("created_note")
    if note:
        return str(note)

    source = ds.metadata.get("msr_source_path") or ds.file.recent_path
    path = Path(str(source)) if source else Path(ds.file.path)
    try:
        if path.exists():
            from datetime import datetime

            return datetime.fromtimestamp(path.stat().st_ctime).strftime("%Y-%b-%d, %H:%M:%S")
    except Exception:
        pass
    return ds.file.datetime or "—"


def _locs_row(ds: MinfluxDataset) -> tuple[str, str]:
    valid = int(ds.metadata.get("valid_num_loc", ds.prop.num_loc))
    total = int(ds.metadata.get("raw_num_loc", ds.metadata.get("overall_num_loc", valid)))
    loaded = int(ds.prop.num_loc)
    if bool(ds.metadata.get("includes_invalid")):
        # only_valid_locs=False: loaded rows may be a subset of the raw file
        # (e.g. last-iteration rows only), so report the loaded count.
        return "Locs (with invalid)", f"{loaded:,} ({valid:,} valid)"
    if valid >= total:
        return "Locs (with invalid)", f"{total:,}"
    return "Locs (valid)", f"{valid:,} / {total:,}"


def _iterations_text(ds: MinfluxDataset) -> str:
    total = max(1, int(ds.metadata.get("raw_num_itr", ds.prop.num_itr or 1)))
    mode = str(ds.metadata.get("iteration_load_mode", "")).lower()
    if not mode:
        itr = ds.attr.get("itr")
        if itr is not None:
            arr = np.asarray(itr).ravel()
            finite = arr[np.isfinite(arr)] if np.issubdtype(arr.dtype, np.number) else arr
            mode = "all" if np.unique(finite).size > 1 else "last"
        else:
            mode = "last"
    prefix = "all" if mode == "all" else "last"
    parts = [f"{prefix} of {total} iterations"]

    cfr_iter = ds.metadata.get("cfr_iteration")
    efc_iter = ds.metadata.get("efc_iteration") or ds.metadata.get("efo_iteration")
    if cfr_iter is not None:
        parts.append(f"CFR {_ordinal(cfr_iter)}")
    if efc_iter is not None:
        parts.append(f"EFC {_ordinal(efc_iter)}")
    return "  |  ".join(parts)


def _z_scaling_factor_source_label(source: str) -> tuple[str, bool]:
    """Map a Z scaling factor provenance source to (display label, is_calculated)."""
    s = (source or "").lower()
    if s.startswith("estimated (trace anisotropy") or s.startswith("auto (estimate"):
        return "estimated from trace anisotropy", True
    if s.startswith("estimate out of range") or s.startswith("auto (out of range"):
        return "estimate out of range; reset to 1", False
    if s.startswith("manual"):
        return "manual", False
    if s.startswith("fixed"):
        return "fixed preference", False
    return "", False


def _dims_text(ds: MinfluxDataset) -> str:
    dims = f"{int(ds.prop.num_dim)}D"
    if int(ds.prop.num_dim) < 3:
        return dims
    z_scaling_factor = float(getattr(ds.cali, "z_scaling_factor", 1.0) or 1.0)
    source = str((ds.metadata.get("z_scaling_factor_provenance") or {}).get("source", "") or "")
    label, _calculated = _z_scaling_factor_source_label(source)
    # Up to four decimals for every source, trailing zeros trimmed but never
    # below two: a manually set 0.6667 must not be shown as 0.67, while a plain
    # 1.0 still reads as "1.00".
    value_txt = format_z_scaling_factor(z_scaling_factor)
    suffix = f" ({label})" if label else ""
    return f"{dims}  |  Z scaling factor = {value_txt}{suffix}"


def _version_text(ds: MinfluxDataset) -> str:
    from ..core.dataset_kind import is_minflux
    if not is_minflux(ds):
        return "Non-MINFLUX data"
    # Data version is the structural format (m2410/m2205/legacy), detected from
    # the mfx structure — independent of how the file was read.
    version = str(ds.metadata.get("source_version", "")).lower()
    if version in {"m2410", "m2205", "legacy"}:
        base = version
    elif version == "simulation":
        base = "simulation"
    elif version in {"csv", "spreadsheet", "imported", "plain_array", "json"}:
        base = "MINFLUX (imported)"
    else:
        base = "unidentified"
    # The obf/mfxdta container (.msr embedded zarr) is the transport, orthogonal
    # to the data version — show it alongside.
    if str(ds.metadata.get("source_format", "")).lower() == "obf / mfxdta":
        return f"{base} (obf / mfxdta)"
    return base


def _is_transformed(ds: MinfluxDataset) -> bool:
    for transform in (
        ds.state.get("overlay_transform"),
        ds.state.get("render_transform_2d"),
        ds.state.get("channel_transform"),
        ds.metadata.get("overlay_transform"),
        ds.metadata.get("render_transform_2d"),
        ds.metadata.get("channel_transform"),
    ):
        if _transform_changes_coordinates(transform):
            return True
    return ds.metadata.get("transformed") is True


def _dataset_transform(ds: MinfluxDataset):
    """Resolve the editable transform, preferring current live view state."""
    from ..core.overlay import transform_to_matrix4

    for transform in (
        ds.state.get("overlay_transform"),
        ds.state.get("render_transform_2d"),
        ds.state.get("channel_transform"),
        ds.metadata.get("overlay_transform"),
        ds.metadata.get("render_transform_2d"),
        ds.metadata.get("channel_transform"),
    ):
        if transform_to_matrix4(transform) is not None:
            return transform
    return None


def _transform_changes_coordinates(transform) -> bool:
    if not transform:
        return False
    from ..core.overlay import transform_to_matrix4

    matrix = transform_to_matrix4(transform)
    if matrix is None:
        return True
    return not np.allclose(matrix, np.eye(4), rtol=0.0, atol=1e-9)


def _ordinal(value) -> str:
    try:
        n = int(value)
    except Exception:
        return str(value)
    if 10 <= n % 100 <= 20:
        suffix = "th"
    else:
        suffix = {1: "st", 2: "nd", 3: "rd"}.get(n % 10, "th")
    return f"{n}{suffix}"
