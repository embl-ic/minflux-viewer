"""
minflux_viewer.plugins.generate_method_text
=============================================
Generate a publication-ready *Methods* paragraph that describes — in
scientific prose — every processing step applied to the active dataset
from the moment it was loaded up to the current state.

Source of truth: the :class:`ProcessingJournal` attached to
:class:`AppState`. Every loader, filter, duplicator, exporter and
analysis routine appends an entry; this plugin compiles those entries
into a paragraph organised by stage (loading → filtering → analysis →
export), then appends a credit line and the MIT licence notice.

The output is shown in a non-modal dialog with **Copy to clipboard**
and **Save to file…** buttons. No automatic citation lookup is
performed — that's the user's responsibility — but the plugin does
include explicit references for the analysis routines that come with
papers attached (StdDev, FRC stub, CRLB stub).
"""

from __future__ import annotations

from datetime import datetime
from pathlib import Path

from PyQt6.QtCore import Qt
from PyQt6.QtGui import QGuiApplication
from PyQt6.QtWidgets import (
    QDialog,
    QFileDialog,
    QHBoxLayout,
    QPlainTextEdit,
    QPushButton,
    QSizePolicy,
    QVBoxLayout,
)

from .. import PluginEntry, register
from ... import __author__, __email__, __institution__, __license__, __version__


# ---------------------------------------------------------------------------
# Plugin entry
# ---------------------------------------------------------------------------

def _launch(state, parent=None) -> None:
    text = generate_methods_text(state)
    dlg = MethodTextDialog(text, parent=parent)
    dlg.show()
    if parent is not None:
        parent._plugin_method_text_dialog = dlg


register(PluginEntry(
    name="Generate Method Text",
    tooltip="Compile a Methods paragraph from the processing-history journal.",
    launch=_launch,
))


# ---------------------------------------------------------------------------
# Text generation
# ---------------------------------------------------------------------------

def generate_methods_text(state) -> str:
    """
    Build a Methods-paragraph-style description of the session.

    Returns a single string with line breaks (no markup).
    """
    journal  = getattr(state, "journal", None)
    entries  = list(journal) if journal is not None else []

    lines: list[str] = []

    # Title block
    lines.append("METHODS — DATA PROCESSING")
    lines.append("=" * 64)
    lines.append("")

    # Section 1: Acquisition / loading
    load_entries = [e for e in entries if e.category == "load"]
    if load_entries:
        lines.append("Data acquisition and loading.")
        for e in load_entries:
            d = e.details
            n_loc = d.get("num_loc", "?")
            n_tr  = d.get("num_traces", "?")
            n_dim = d.get("num_dim", "?")
            src   = d.get("source", "(unknown source)")
            n_loc_str = f"{n_loc:,}" if isinstance(n_loc, int) else str(n_loc)
            n_tr_str  = f"{n_tr:,}"  if isinstance(n_tr, int)  else str(n_tr)
            lines.append(
                f"  • Dataset '{e.summary.replace(chr(0x27), '')[len('Loaded dataset '):]}': "
                f"loaded {n_loc_str} localizations grouped into {n_tr_str} trace(s) "
                f"({n_dim}-D coordinates) from {Path(src).name}."
            )
        lines.append("")
    else:
        lines.append("No dataset has been loaded in this session.")
        lines.append("")

    # Section 2: Filtering
    filter_entries = [e for e in entries if e.category == "filter"]
    if filter_entries:
        lines.append("Filtering.")
        for e in filter_entries:
            lines.append(f"  • {e.summary}")
        lines.append("")

    # Section 3: Transformations (duplicate, etc.)
    transform_entries = [e for e in entries if e.category == "transform"]
    if transform_entries:
        lines.append("Transformations.")
        for e in transform_entries:
            lines.append(f"  • {e.summary}")
        lines.append("")

    # Section 4: Analysis
    analysis_entries = [e for e in entries if e.category == "analysis"]
    if analysis_entries:
        lines.append("Analysis.")
        for e in analysis_entries:
            lines.append(f"  • {e.summary}")
        lines.append("")
        # Standard reference list for the analyses we ship
        used = " ".join(e.summary for e in analysis_entries).lower()
        refs: list[str] = []
        if "stddev" in used:
            refs.append(
                "Schmidt et al., Nat. Methods 19:1246 (2022) — per-trace "
                "standard-deviation precision metric."
            )
        if "frc" in used:
            refs.append(
                "Banterle et al., J. Struct. Biol. 183:363 (2013); "
                "Nieuwenhuizen et al., Nat. Methods 10:557 (2013) — FRC."
            )
        if "crlb" in used:
            refs.append(
                "Mortensen et al., Nat. Methods 7:377 (2010) — CRLB."
            )
        if refs:
            lines.append("References:")
            for r in refs:
                lines.append(f"  - {r}")
            lines.append("")

    # Section 5: Export
    export_entries = [e for e in entries if e.category == "export"]
    if export_entries:
        lines.append("Export.")
        for e in export_entries:
            lines.append(f"  • {e.summary}")
        lines.append("")

    # Section 6: Other / plugin
    other_entries = [
        e for e in entries
        if e.category not in (
            "load", "filter", "transform", "analysis", "export",
        )
    ]
    if other_entries:
        lines.append("Other operations.")
        for e in other_entries:
            lines.append(f"  • [{e.category}] {e.summary}")
        lines.append("")

    if len(entries) == 0:
        lines.append(
            "(The processing journal is empty. Load a dataset and apply at "
            "least one filter, transformation or analysis to populate it.)"
        )
        lines.append("")

    # Tooling block
    lines.append("Tooling.")
    lines.append(
        f"  • Generated by MINFLUX Data Viewer v{__version__} "
        f"({datetime.now().strftime('%Y-%m-%d %H:%M:%S')})."
    )
    lines.append(
        "  • Python/Qt port (PyQt6, pyqtgraph, NumPy, SciPy) of the original "
        "MATLAB MINFLUX viewer."
    )
    lines.append("")

    # Credit + licence
    lines.append("=" * 64)
    lines.append(f"Author: {__author__} <{__email__}>")
    lines.append(f"        {__institution__}")
    lines.append(
        f"License: {__license__} — see the project repository for the full "
        "licence text. The MIT License permits free use, modification, "
        "distribution, and commercial use, provided the licence and "
        "copyright notice are included with all copies or substantial "
        "portions of the software."
    )
    return "\n".join(lines)


# ---------------------------------------------------------------------------
# Dialog
# ---------------------------------------------------------------------------

class MethodTextDialog(QDialog):
    def __init__(self, text: str, parent=None) -> None:
        super().__init__(parent)
        self.setAttribute(Qt.WidgetAttribute.WA_DeleteOnClose)
        self.setWindowTitle("Generate Method Text")
        self.resize(720, 540)

        root = QVBoxLayout(self)
        self._editor = QPlainTextEdit()
        self._editor.setPlainText(text)
        self._editor.setReadOnly(False)   # allow user edits before saving
        self._editor.setSizePolicy(
            QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Expanding,
        )
        # Monospace for nicer alignment of the section bars
        font = self._editor.font()
        font.setFamily("Consolas")
        font.setStyleHint(font.StyleHint.Monospace)
        self._editor.setFont(font)
        root.addWidget(self._editor, stretch=1)

        btn_row = QHBoxLayout()
        copy_btn = QPushButton("Copy to clipboard")
        copy_btn.clicked.connect(self._copy)
        btn_row.addWidget(copy_btn)

        save_btn = QPushButton("Save to file…")
        save_btn.clicked.connect(self._save)
        btn_row.addWidget(save_btn)

        btn_row.addStretch()

        close_btn = QPushButton("Close")
        close_btn.clicked.connect(self.close)
        btn_row.addWidget(close_btn)

        root.addLayout(btn_row)

    def _copy(self) -> None:
        QGuiApplication.clipboard().setText(self._editor.toPlainText())

    def _save(self) -> None:
        path, _ = QFileDialog.getSaveFileName(
            self, "Save methods text", "methods.txt",
            "Text files (*.txt);;Markdown (*.md);;All files (*)",
        )
        if not path:
            return
        Path(path).write_text(self._editor.toPlainText(), encoding="utf-8")
