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
        │  Loaded   2025-Apr-16, 09:31:02 │
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

        self.setWindowTitle(dataset.file.name)
        self.setMinimumWidth(310)
        self.setSizePolicy(QSizePolicy.Policy.Fixed, QSizePolicy.Policy.Fixed)
        # Tool window: stays above the main window, no taskbar entry
        self.setWindowFlags(Qt.WindowType.Window | Qt.WindowType.Tool)

        self._build_ui(dataset)

        # Track active-dataset changes so we can update the title / button
        state.active_changed.connect(self._refresh)

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
        header = QLabel(ds.file.name)
        font   = QFont()
        font.setBold(True)
        font.setPointSize(11)
        header.setFont(font)
        layout.addWidget(header, row, 0, 1, 2)
        row += 1

        # ── Separator ───────────────────────────────────────────────
        sep = QFrame()
        sep.setFrameShape(QFrame.Shape.HLine)
        sep.setFrameShadow(QFrame.Shadow.Sunken)
        layout.addWidget(sep, row, 0, 1, 2)
        row += 1

        # ── Info rows ───────────────────────────────────────────────
        info_rows = [
            ("Folder",   _shorten_path(ds.file.folder)),
            ("Loaded",   ds.file.datetime or "—"),
            ("Locs",     f"{ds.prop.num_loc:,}"),
            ("Traces",   f"{ds.prop.num_traces:,}"),
            ("Dims",     f"{ds.prop.num_dim}D  |  {ds.prop.num_itr} iterations"),
        ]
        for label, value in info_rows:
            key_lbl = QLabel(label)
            key_lbl.setStyleSheet("color: gray; font-size: 11px;")
            val_lbl = QLabel(value)
            val_lbl.setWordWrap(True)
            val_lbl.setStyleSheet("font-size: 11px;")
            layout.addWidget(key_lbl, row, 0)
            layout.addWidget(val_lbl, row, 1)
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

    def changeEvent(self, event: QEvent) -> None:
        """Activate this dataset whenever the window gains focus."""
        if (
            event.type() == QEvent.Type.ActivationChange
            and self.isActiveWindow()
        ):
            self._activate()
        super().changeEvent(event)

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

    def _refresh(self, active_idx: int | None) -> None:
        """Update the window title and button state."""
        ds = self._state.datasets[self._idx]
        if active_idx == self._idx:
            self.setWindowTitle(f"[ACTIVE]  {ds.file.name}")
            self._btn.setEnabled(False)
            self._btn.setText("Active dataset")
        else:
            self.setWindowTitle(ds.file.name)
            self._btn.setEnabled(True)
            self._btn.setText("Set as active")


# ---------------------------------------------------------------------------
# Helper
# ---------------------------------------------------------------------------

def _shorten_path(folder: str, max_parts: int = 3) -> str:
    """Display only the last *max_parts* components of a path."""
    from pathlib import Path
    parts = Path(folder).parts
    if len(parts) > max_parts:
        return "…/" + "/".join(parts[-max_parts:])
    return folder
