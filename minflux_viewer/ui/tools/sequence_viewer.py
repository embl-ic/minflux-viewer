"""Read-only viewer for MINFLUX acquisition sequences.

Shows a sequence opened from a ``.json`` file, or every sequence a loaded
dataset recorded (one per measurement thread). Three tabs: the sequence-wide
settings, an iteration table with the derived values a methods section needs
(pattern diameter L, CFR check, thresholds, which iterations give the final
localization), and the raw document as a described JSON tree.

The model is ``core.acquisition_sequence``; this window only presents it.
Modeless and unparented (shown through ``show_modeless``); it never edits or
writes a sequence.
"""

from __future__ import annotations

import json

from PyQt6.QtCore import Qt
from PyQt6.QtGui import QBrush, QColor, QFont
from PyQt6.QtWidgets import (
    QApplication,
    QComboBox,
    QDialogButtonBox,
    QFormLayout,
    QHBoxLayout,
    QHeaderView,
    QLabel,
    QScrollArea,
    QTableWidget,
    QTableWidgetItem,
    QTabWidget,
    QVBoxLayout,
    QWidget,
)

from minflux_viewer.colors import iteration_color
from minflux_viewer.core.acquisition_sequence import (
    ITERATION_COLUMNS,
    SequenceSource,
    describe_key,
    pattern_description,
    sequence_overview,
)
from minflux_viewer.core.tiff_source import MetadataDocument
from minflux_viewer.ui.tools.metadata_viewer import MetadataDocumentView
from minflux_viewer.ui.widgets.text_select import make_labels_selectable

#: Header tooltip per iteration column: the sequence key it shows.
_COLUMN_KEYS = {
    "mode": ("Itr", "Mode", "id"), "dim": ("Itr", "Mode", "dim"),
    "pattern": ("Itr", "Mode", "pattern"), "geo": ("Itr", "patGeoFactor"),
    "l_nm": ("Itr", "patGeoFactor"), "pht": ("Itr", "phtLimit"),
    "dwell": ("Itr", "patDwellTime"), "repeat": ("Itr", "patRepeat"),
    "power": ("Itr", "pwrFactor"), "ccr": ("Itr", "ccrLimit"),
    "bgc": ("Itr", "bgcThreshold"), "max_off": ("Itr", "maxOffTime"),
    "excitation": ("Itr", "_excitation"), "detection": ("Itr", "_detection"),
}
_COLUMN_NOTES = {
    "role": "What the pattern localizes, from its name; 'unknown' when the name is not documented.",
    "final": ("Iterations at the smallest pattern of their axis: the final lateral and axial "
              "localization (only patterns whose role is documented are counted)."),
    "itr": "Iteration number (1-based). Rows from the loop start onward are repeated after a "
           "final localization.",
}


class SequenceViewerWindow(QWidget):
    """One window per sequence file or per dataset."""

    def __init__(self, sources: list[SequenceSource], *, title: str,
                 prefs: dict | None = None, source_key: str = "") -> None:
        super().__init__(None)
        if not sources:
            raise ValueError("No sequence to show.")
        self._sources = list(sources)
        self._prefs = prefs or {}
        #: Identifies what is shown, so opening the same source raises this window.
        self.source_key = source_key
        self.setWindowTitle(f"Acquisition sequence - {title}")
        self.resize(1180, 640)

        layout = QVBoxLayout(self)
        bar = QHBoxLayout()
        self._selector_label = QLabel("Sequence:")
        self._selector = QComboBox()
        for source in self._sources:
            self._selector.addItem(source.label)
        self._selector.currentIndexChanged.connect(self._show)
        bar.addWidget(self._selector_label)
        bar.addWidget(self._selector, 1)
        visible = len(self._sources) > 1
        self._selector_label.setVisible(visible)
        self._selector.setVisible(visible)
        layout.addLayout(bar)
        self._origin = QLabel()
        self._origin.setWordWrap(True)
        layout.addWidget(self._origin)

        self.tabs = QTabWidget()
        self._overview_area = QScrollArea()
        self._overview_area.setWidgetResizable(True)
        self.tabs.addTab(self._overview_area, "Overview")
        self.table = QTableWidget()
        self.table.setEditTriggers(QTableWidget.EditTrigger.NoEditTriggers)
        self.table.setAlternatingRowColors(True)
        self.tabs.addTab(self.table, "Iterations")
        self.document_view = MetadataDocumentView(parent=self, describe=describe_key)
        self.tabs.addTab(self.document_view, "JSON")
        layout.addWidget(self.tabs, 1)

        buttons = QDialogButtonBox(QDialogButtonBox.StandardButton.Close)
        buttons.rejected.connect(self.close)
        for text, slot in (("Copy table", self.copy_table), ("Copy JSON", self.copy_json)):
            button = buttons.addButton(text, QDialogButtonBox.ButtonRole.ActionRole)
            if button is not None:
                button.clicked.connect(slot)
        layout.addWidget(buttons)
        self._show(0)

    # ----------------------------------------------------------------- views
    def current_source(self) -> SequenceSource:
        index = self._selector.currentIndex()
        return self._sources[index if 0 <= index < len(self._sources) else 0]

    def _show(self, _index: int = 0) -> None:
        source = self.current_source()
        overview = sequence_overview(source.sequence)
        self._origin.setText(f"<b>{overview.name}</b> &nbsp; from {source.origin}"
                             if source.origin else f"<b>{overview.name}</b>")
        self._fill_overview(source, overview)
        self._fill_table(overview)
        self.document_view.set_documents([MetadataDocument(
            overview.name, "json", json.dumps(source.sequence, indent=2))])

    def _fill_overview(self, source: SequenceSource, overview) -> None:
        page = QWidget()
        form = QFormLayout(page)
        for label, value in source.details:
            form.addRow(f"{label}:", QLabel(value))
        for label, value, key in overview.globals:
            text = QLabel(value)
            text.setWordWrap(True)
            description = describe_key((key,)) if key else ""
            if description:
                text.setToolTip(description)
            form.addRow(f"{label}:", text)
        make_labels_selectable(page)
        self._overview_area.setWidget(page)

    def _fill_table(self, overview) -> None:
        rows = overview.iterations
        self.table.clear()
        self.table.setColumnCount(len(ITERATION_COLUMNS))
        self.table.setRowCount(len(rows))
        self.table.setHorizontalHeaderLabels([header for _key, header in ITERATION_COLUMNS])
        for column, (key, _header) in enumerate(ITERATION_COLUMNS):
            tip = _COLUMN_NOTES.get(key) or describe_key(_COLUMN_KEYS.get(key, ()))
            header = self.table.horizontalHeaderItem(column)
            if header is not None and tip:
                header.setToolTip(tip)
        bold = QFont(self.table.font())
        bold.setBold(True)
        n = len(rows)
        for row_index, row in enumerate(rows):
            for column, (key, _header) in enumerate(ITERATION_COLUMNS):
                item = QTableWidgetItem(row.cells.get(key, ""))
                if row.final:
                    item.setFont(bold)
                if key == "pattern":
                    item.setToolTip(pattern_description(row.cells.get("pattern", "")))
                if key == "itr":
                    r, g, b, _a = iteration_color(self._prefs, row.index, n)
                    item.setBackground(QBrush(QColor(r, g, b)))
                    light = (0.299 * r + 0.587 * g + 0.114 * b) > 140
                    item.setForeground(QBrush(QColor(0, 0, 0) if light else QColor(255, 255, 255)))
                    if row.in_loop:
                        item.setText(f"{row.cells['itr']} ↻")
                        item.setToolTip("Repeated after a final localization (headstart loop).")
                self.table.setItem(row_index, column, item)
        self.table.resizeColumnsToContents()
        self.table.setVerticalScrollMode(QTableWidget.ScrollMode.ScrollPerPixel)
        horizontal, vertical = self.table.horizontalHeader(), self.table.verticalHeader()
        if horizontal is not None:
            horizontal.setSectionResizeMode(QHeaderView.ResizeMode.Interactive)
            horizontal.setDefaultAlignment(Qt.AlignmentFlag.AlignLeft)
        if vertical is not None:
            vertical.setVisible(False)

    # ------------------------------------------------------------ clipboard
    def table_text(self) -> str:
        """The iteration table as tab-separated text, header first."""
        overview = sequence_overview(self.current_source().sequence)
        lines = ["\t".join(header for _key, header in ITERATION_COLUMNS)]
        for row in overview.iterations:
            lines.append("\t".join(row.cells.get(key, "") for key, _h in ITERATION_COLUMNS))
        return "\n".join(lines)

    def copy_table(self) -> None:
        _copy_text(self.table_text())

    def copy_json(self) -> None:
        _copy_text(json.dumps(self.current_source().sequence, indent=2))


def _copy_text(text: str) -> None:
    clipboard = QApplication.clipboard()
    if clipboard is not None:
        clipboard.setText(text)


def open_sequence_window(owner, sources: list[SequenceSource], *, title: str,
                         source_key: str, prefs: dict | None = None) -> SequenceViewerWindow:
    """Raise the window already showing *source_key*, else open a new one.

    The window is modeless and unparented; *owner* (the main window) retains
    it through ``show_modeless`` and closes it at shutdown.
    """
    from minflux_viewer.ui.common.modeless import show_modeless

    for window in list(getattr(owner, "_modeless_windows", []) or []):
        try:
            if isinstance(window, SequenceViewerWindow) and window.source_key == source_key:
                window.raise_()
                window.activateWindow()
                return window
        except RuntimeError:                       # C++ peer already deleted
            continue
    window = SequenceViewerWindow(sources, title=title, prefs=prefs, source_key=source_key)
    show_modeless(window, owner)
    return window
