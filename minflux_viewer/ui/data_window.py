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
    QFrame,
    QGridLayout,
    QLabel,
    QPushButton,
    QSizePolicy,
    QWidget,
)

from ..core.app_state import AppState
from ..core.dataset import MinfluxDataset


class DataWindow(QWidget):
    """
    Compact info card for one dataset.

    ::

        ┌─────────────────────────────────┐
        │  filename.mat                   │
        ├─────────────────────────────────┤
        │  Folder   …/experiment/day1/    │
        │  Created  2025-Apr-16, 09:31:02 │
        │  Locs     45,231                │
        │  Traces   1,204                 │
        │  Dims     3D  |  5 iterations   │
        │                                 │
        │  [ Set as active ]              │
        └─────────────────────────────────┘
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
        # Refresh info values (e.g. RIMF) when this dataset's calibration changes
        state.calibration_changed.connect(self._on_calibration_changed)

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
            row += 1

        # ── Spacer ──────────────────────────────────────────────────
        layout.setRowMinimumHeight(row, 6)
        row += 1

        # ── Activate button ─────────────────────────────────────────
        self._btn = QPushButton("Set as active")
        self._btn.clicked.connect(self._activate)
        layout.addWidget(self._btn, row, 0, 1, 2)

        self._refresh(self._state.active_idx)

    # ------------------------------------------------------------------
    # Interaction
    # ------------------------------------------------------------------

    def _activate(self) -> None:
        self._state.set_active(self._idx)

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
        """Refresh info values (e.g. the RIMF row) for this dataset."""
        if idx != self._idx or not (0 <= self._idx < len(self._state.datasets)):
            return
        ds = self._state.datasets[self._idx]
        for label, value in _info_rows(ds):
            lbl = self._value_labels.get(label)
            if lbl is not None:
                lbl.setText(value)

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
        ("Created", _created_text(ds)),
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


def _rimf_source_label(source: str) -> tuple[str, bool]:
    """Map a RIMF provenance source to (display label, is_calculated)."""
    s = (source or "").lower()
    if s.startswith("auto (estimate"):
        return "calculated, anisotropy plugin", True
    if s.startswith("auto (out of range"):
        return "calculated out of range, reset to 1", False
    if s.startswith("manual"):
        return "manual, anisotropy plugin", False
    if s.startswith("fixed"):
        return "fixed, Preference/Data", False
    return "", False


def _dims_text(ds: MinfluxDataset) -> str:
    dims = f"{int(ds.prop.num_dim)}D"
    if int(ds.prop.num_dim) < 3:
        return dims
    rimf = float(getattr(ds.cali, "RIMF", 1.0) or 1.0)
    source = str((ds.metadata.get("rimf_provenance") or {}).get("source", "") or "")
    label, calculated = _rimf_source_label(source)
    # A calculated in-range value keeps 4 decimals; manual / fixed / any 1.0
    # use 2 decimals (e.g. "1.00").
    value_txt = f"{rimf:.4f}" if (calculated and rimf != 1.0) else f"{rimf:.2f}"
    suffix = f" ({label})" if label else ""
    return f"{dims}  |  RIMF = {value_txt}{suffix}"


def _version_text(ds: MinfluxDataset) -> str:
    from ..core.dataset_kind import is_minflux
    if not is_minflux(ds):
        return "Non-MINFLUX data"
    version = str(ds.metadata.get("source_version", "")).lower()
    if version in {"m2410", "m2205", "legacy"}:
        return version
    if version == "obf / mfxdta":
        return "obf / mfxdta"
    if version in {"csv", "spreadsheet", "imported", "plain_array", "json"}:
        return "MINFLUX (imported)"
    return "unidentified"


def _is_transformed(ds: MinfluxDataset) -> bool:
    if ds.metadata.get("transformed") is False:
        return False
    if ds.metadata.get("transformed") is True:
        return True

    for transform in (
        ds.metadata.get("render_transform_2d"),
        ds.metadata.get("channel_transform"),
        ds.state.get("render_transform_2d"),
        ds.state.get("channel_transform"),
    ):
        if _transform_changes_coordinates(transform):
            return True
    return False


def _transform_changes_coordinates(transform) -> bool:
    if not transform:
        return False
    try:
        matrix = np.asarray(transform.get("matrix_3x3"), dtype=float)
    except Exception:
        return True
    if matrix.shape != (3, 3):
        return True
    return not np.allclose(matrix, np.eye(3), rtol=0.0, atol=1e-9)


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
