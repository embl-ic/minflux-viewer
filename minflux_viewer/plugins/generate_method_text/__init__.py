"""
minflux_viewer.plugins.generate_method_text
=============================================
Generate a publication-style *Methods* paragraph from the session's **Log
events**.

Flow: the user ticks the log events that belong to the dataset / method they want
to describe and clicks **Generate**. Events are listed oldest→newest (same order
as the Log dialog). Events for the active dataset are pre-checked; if the active
dataset is part of an overlay, every channel's events in that overlay are
pre-checked (so the whole multi-channel acquisition is described regardless of
which channel is active). The selected events are compiled by
:func:`minflux_viewer.analysis.method_text.generate_method_text` /
``generate_method_html`` into prose grouped by stage, shown in an editable
rich-text window whose citations are clickable DOI hyperlinks, with Copy / Save.
"""

from __future__ import annotations

from pathlib import Path

from PyQt6.QtCore import Qt, QMimeData, QUrl
from PyQt6.QtGui import QDesktopServices, QGuiApplication
from PyQt6.QtWidgets import (
    QDialog,
    QFileDialog,
    QHBoxLayout,
    QLabel,
    QListWidget,
    QListWidgetItem,
    QMessageBox,
    QPushButton,
    QSizePolicy,
    QTextEdit,
    QVBoxLayout,
)

from .. import PluginEntry, register


def _launch(state, parent=None) -> None:
    if not getattr(state, "log_history", None):
        QMessageBox.information(
            parent, "Generate Method Text",
            "No log events yet. Load a dataset and perform some operations, then try again.")
        return
    dlg = MethodEventSelectionDialog(state, parent=parent)
    if dlg.exec() != QDialog.DialogCode.Accepted:
        return
    events = dlg.selected_events()
    if not events:
        QMessageBox.information(parent, "Generate Method Text", "No events were selected.")
        return
    from ...analysis.method_text import generate_method_text, generate_method_html
    from ... import __version__
    text = generate_method_text(state, events, version=__version__)
    html = generate_method_html(state, events, version=__version__)
    out = MethodTextDialog(text, html, parent=parent)
    out.show()
    if parent is not None:
        parent._plugin_method_text_dialog = out   # keep alive


register(PluginEntry(
    name="Generate Method Text",
    tooltip="Compile a Methods paragraph from selected Log events.",
    launch=_launch,
))


# ---------------------------------------------------------------------------
# Event-selection dialog
# ---------------------------------------------------------------------------

class MethodEventSelectionDialog(QDialog):
    """Pick the log events relevant to the method to describe."""

    def __init__(self, state, parent=None) -> None:
        super().__init__(parent)
        self._state = state
        self.setWindowTitle("Generate Method Text — select events")
        self.resize(760, 480)

        root = QVBoxLayout(self)
        note = QLabel(
            "Tick the Log events that belong to the dataset/method you want to describe, "
            "then click Generate. Events are listed oldest first (like the Log). Events for "
            "the active dataset — or, for an overlay, every channel in it — are pre-checked.")
        note.setWordWrap(True)
        root.addWidget(note)

        self._list = QListWidget()
        root.addWidget(self._list, 1)

        auto_idx = self._auto_check_indices(state)
        for ev in state.log_history:     # oldest first (same order as the Log dialog)
            t = ev["time"].strftime("%H:%M:%S") if ev.get("time") else ""
            ds = ev.get("dataset_name") or "—"
            item = QListWidgetItem(f"[{t}]  ({ds})  {ev.get('message', '')}")
            item.setFlags(item.flags() | Qt.ItemFlag.ItemIsUserCheckable)
            checked = ev.get("dataset_idx") in auto_idx
            item.setCheckState(Qt.CheckState.Checked if checked else Qt.CheckState.Unchecked)
            item.setData(Qt.ItemDataRole.UserRole, ev)
            self._list.addItem(item)
        self._list.scrollToBottom()      # newest events visible, like the Log

        btns = QHBoxLayout()
        for text, checked in (("Select all", True), ("Select none", False)):
            b = QPushButton(text)
            b.clicked.connect(lambda _=False, c=checked: self._set_all(c))
            btns.addWidget(b)
        btns.addStretch(1)
        gen = QPushButton("Generate")
        gen.setDefault(True)
        gen.clicked.connect(self.accept)
        btns.addWidget(gen)
        cancel = QPushButton("Cancel")
        cancel.clicked.connect(self.reject)
        btns.addWidget(cancel)
        root.addLayout(btns)

    @staticmethod
    def _auto_check_indices(state) -> set:
        """Dataset indices whose events are pre-checked: the active dataset, or —
        if it belongs to an overlay — every channel in that overlay."""
        active = getattr(state, "active_idx", None)
        if active is None:
            return set()
        try:
            from ...core.overlay import overlay_members
            return {idx for idx, _ds in overlay_members(state, active)}
        except Exception:
            return {active}

    def _set_all(self, checked: bool) -> None:
        st = Qt.CheckState.Checked if checked else Qt.CheckState.Unchecked
        for i in range(self._list.count()):
            self._list.item(i).setCheckState(st)

    def selected_events(self) -> list:
        """Checked events in display order (oldest first / chronological)."""
        events = []
        for i in range(self._list.count()):
            item = self._list.item(i)
            if item.checkState() == Qt.CheckState.Checked:
                events.append(item.data(Qt.ItemDataRole.UserRole))
        return events


# ---------------------------------------------------------------------------
# Editable output window (rich text with clickable DOI hyperlinks)
# ---------------------------------------------------------------------------

class _LinkTextEdit(QTextEdit):
    """Editable rich-text edit that opens the anchor under the cursor on click."""

    def mouseReleaseEvent(self, ev) -> None:  # noqa: N802 (Qt signature)
        if (ev.button() == Qt.MouseButton.LeftButton
                and not self.textCursor().hasSelection()):
            anchor = self.anchorAt(ev.pos())
            if anchor:
                QDesktopServices.openUrl(QUrl(anchor))
                return
        super().mouseReleaseEvent(ev)


class MethodTextDialog(QDialog):
    def __init__(self, text: str, html: str, parent=None) -> None:
        super().__init__(parent)
        self.setAttribute(Qt.WidgetAttribute.WA_DeleteOnClose)
        self.setWindowTitle("Generate Method Text")
        self.resize(760, 560)

        root = QVBoxLayout(self)
        self._editor = _LinkTextEdit()
        self._editor.setHtml(html)
        self._editor.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Expanding)
        root.addWidget(self._editor, 1)

        row = QHBoxLayout()
        copy = QPushButton("Copy to clipboard")
        copy.clicked.connect(self._copy)
        row.addWidget(copy)
        save = QPushButton("Save to file…")
        save.clicked.connect(self._save)
        row.addWidget(save)
        row.addStretch(1)
        close = QPushButton("Close")
        close.clicked.connect(self.close)
        row.addWidget(close)
        root.addLayout(row)

    def _copy(self) -> None:
        """Copy both rich (links preserved for Word/Docs) and plain (URLs as text)."""
        md = QMimeData()
        md.setHtml(self._editor.toHtml())
        md.setText(self._editor.toPlainText())
        QGuiApplication.clipboard().setMimeData(md)

    def _save(self) -> None:
        path, selected = QFileDialog.getSaveFileName(
            self, "Save methods text", "methods.txt",
            "Text files (*.txt);;HTML (*.html);;All files (*)")
        if not path:
            return
        if path.lower().endswith(".html") or "html" in (selected or "").lower():
            Path(path).write_text(self._editor.toHtml(), encoding="utf-8")
        else:
            Path(path).write_text(self._editor.toPlainText(), encoding="utf-8")
