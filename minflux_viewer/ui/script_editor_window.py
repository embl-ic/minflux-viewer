"""Fiji-like Python script editor for local MINFLUX Viewer automation."""

from __future__ import annotations

import contextlib
import io
import traceback
from pathlib import Path

import numpy as np
from PyQt6.QtCore import QEvent, Qt, QStringListModel, QTimer
from PyQt6.QtGui import QFont, QTextCursor, QTextOption
from PyQt6.QtWidgets import (
    QCompleter,
    QDialog,
    QDialogButtonBox,
    QFileDialog,
    QHBoxLayout,
    QLabel,
    QMessageBox,
    QPlainTextEdit,
    QPushButton,
    QSplitter,
    QVBoxLayout,
    QWidget,
)

from ..scripting import install_runtime_module
from .console_window import ConsoleWindow


_DEFAULT_SCRIPT = """import mfv
import numpy as np

ds = mfv.get_active_dataset()
print("Active dataset:", ds.name if ds is not None else "none")

if ds is not None:
    loc = mfv.get_loc(ds, unit="nm", filtered=True)
    view = mfv.viewer.scatter(loc[:, 0], loc[:, 1], title=ds.name)
    view.show()
"""


_API_HELP = """MINFLUX Viewer scripting MVP

Python scripts run inside the current viewer session. The runtime module
`mfv` is bound to the active application state.

Common pattern:

import mfv
import numpy as np

ds = mfv.get_active_dataset()
loc = mfv.get_loc(ds, unit="nm", filtered=True)
view = mfv.viewer.scatter(loc[:, 0], loc[:, 1], title=ds.name)
view.show()

Dataset helpers:

mfv.get_active_dataset()
mfv.get_datasets()
mfv.get_dataset(index_or_name=None)
mfv.get_attr("efo", dataset=None, source="mfx", filtered=True)
mfv.get_loc(dataset=None, unit="nm", filtered=True)
mfv.log("message", level="INFO")
mfv.show_console()

Viewer helpers:

mfv.viewer.render(dataset=None)
mfv.viewer.scatter(x=None, y=None, z=None, dataset=None, color_by=None, title=None)
mfv.viewer.histogram(values=None, dataset=None, attr=None, bins=None)
mfv.viewer.attribute_plot(dataset=None, x="idx", y="efo")

Notes:

- Scripts are trusted local Python, not sandboxed.
- Long-running scripts run synchronously in this MVP and may pause the UI.
- Raw localization data are not modified by these helpers.
"""


_COMPLETION_MAP = {
    "mfv": [
        "get_active_dataset()",
        "get_datasets()",
        "get_dataset()",
        "get_attr()",
        "get_loc()",
        "log()",
        "show_console()",
        "viewer",
        "ScriptError",
    ],
    "mfv.viewer": [
        "render()",
        "scatter()",
        "histogram()",
        "attribute_plot()",
    ],
    "np": [
        "array()",
        "asarray()",
        "column_stack()",
        "concatenate()",
        "isfinite()",
        "nanmean()",
        "nanmedian()",
        "nanstd()",
        "mean()",
        "median()",
        "std()",
        "min()",
        "max()",
        "histogram()",
        "where()",
        "unique()",
        "zeros()",
        "ones()",
        "linspace()",
        "arange()",
    ],
    "numpy": [
        "array()",
        "asarray()",
        "column_stack()",
        "concatenate()",
        "isfinite()",
        "nanmean()",
        "nanmedian()",
        "nanstd()",
        "mean()",
        "median()",
        "std()",
        "min()",
        "max()",
        "histogram()",
        "where()",
        "unique()",
        "zeros()",
        "ones()",
        "linspace()",
        "arange()",
    ],
}


class _ScriptOutputStream(io.TextIOBase):
    def __init__(self, callback, *, is_err: bool = False) -> None:
        self._callback = callback
        self._is_err = is_err

    def write(self, text: str) -> int:
        if text:
            self._callback(text, is_err=self._is_err)
        return len(text)

    def flush(self) -> None:
        return None


class ScriptEditorWindow(QWidget):
    """Non-blocking Python editor with a persistent execution namespace."""

    TAG = "script_editor_window"

    def __init__(self, state, parent=None) -> None:
        super().__init__(parent)
        self._state = state
        self._parent_window = parent
        self._path: Path | None = None
        self._namespace: dict[str, object] = {}
        self._force_close = False
        self._completion_prefix = ""

        self.setWindowTitle("Script Editor")
        self.setWindowFlags(Qt.WindowType.Window)
        self.resize(920, 720)
        self.setAttribute(Qt.WidgetAttribute.WA_DeleteOnClose, False)

        self._build_ui()
        self._install_completer()
        self._reset_namespace()

    def _build_ui(self) -> None:
        root = QVBoxLayout(self)
        root.setContentsMargins(6, 6, 6, 6)
        root.setSpacing(6)

        toolbar = QHBoxLayout()
        for label, slot in (
            ("Run", self._run_all),
            ("Run Selection", self._run_selection),
            ("Open...", self._open_script),
            ("Save", self._save_script),
            ("Save As...", self._save_script_as),
            ("Clear Output", self._clear_output),
            ("API Help", self._show_api_help),
        ):
            button = QPushButton(label)
            button.clicked.connect(slot)
            toolbar.addWidget(button)
        toolbar.addStretch(1)
        root.addLayout(toolbar)

        splitter = QSplitter(Qt.Orientation.Vertical, self)
        self.editor = QPlainTextEdit(self)
        self.editor.setPlainText(_DEFAULT_SCRIPT)
        self.editor.setLineWrapMode(QPlainTextEdit.LineWrapMode.NoWrap)
        self.editor.document().setDefaultTextOption(QTextOption(Qt.AlignmentFlag.AlignLeft))
        self.editor.setFont(self._mono_font(10))
        splitter.addWidget(self.editor)

        output_wrap = QWidget(self)
        output_layout = QVBoxLayout(output_wrap)
        output_layout.setContentsMargins(0, 0, 0, 0)
        output_layout.addWidget(QLabel("Output"))
        self.output = QPlainTextEdit(self)
        self.output.setReadOnly(True)
        self.output.setMaximumBlockCount(5000)
        self.output.setFont(self._mono_font(9))
        self.output.setStyleSheet(
            "QPlainTextEdit { background: #1f1f1f; color: #dddddd; border: 1px solid #999; }"
        )
        output_layout.addWidget(self.output)
        splitter.addWidget(output_wrap)
        splitter.setStretchFactor(0, 3)
        splitter.setStretchFactor(1, 1)
        root.addWidget(splitter, 1)

    def _mono_font(self, size: int) -> QFont:
        font = QFont("Consolas")
        font.setStyleHint(QFont.StyleHint.Monospace)
        font.setPointSize(size)
        return font

    def _install_completer(self) -> None:
        self._completion_model = QStringListModel(self)
        self._completer = QCompleter(self._completion_model, self)
        self._completer.setWidget(self.editor)
        self._completer.setCaseSensitivity(Qt.CaseSensitivity.CaseInsensitive)
        self._completer.setCompletionMode(QCompleter.CompletionMode.PopupCompletion)
        self._completer.activated[str].connect(self._insert_completion)
        self.editor.installEventFilter(self)

    def eventFilter(self, obj, event) -> bool:  # noqa: N802 - Qt API
        if obj is self.editor and event.type() == QEvent.Type.KeyPress:
            if self._handle_editor_key(event):
                return True
        return super().eventFilter(obj, event)

    def _handle_editor_key(self, event) -> bool:
        popup = self._completer.popup()
        if popup.isVisible() and event.key() in (
            Qt.Key.Key_Enter,
            Qt.Key.Key_Return,
            Qt.Key.Key_Escape,
            Qt.Key.Key_Tab,
            Qt.Key.Key_Backtab,
        ):
            return False

        ctrl_space = (
            event.key() == Qt.Key.Key_Space
            and event.modifiers() & Qt.KeyboardModifier.ControlModifier
        )
        if ctrl_space:
            self._show_completion()
            return True

        if event.text() == ".":
            QTimer.singleShot(0, self._show_completion)
            return False

        if event.text() or event.key() in (Qt.Key.Key_Backspace, Qt.Key.Key_Delete):
            QTimer.singleShot(0, self._maybe_update_completion)
        return False

    def _completion_context(self) -> tuple[str, str] | None:
        cursor = self.editor.textCursor()
        text = self.editor.toPlainText()[:cursor.position()]
        tail = []
        for ch in reversed(text):
            if ch.isalnum() or ch in {"_", "."}:
                tail.append(ch)
            else:
                break
        token = "".join(reversed(tail))
        if "." not in token:
            return None
        obj, prefix = token.rsplit(".", 1)
        obj = obj.strip()
        if obj not in _COMPLETION_MAP:
            return None
        return obj, prefix

    def _maybe_update_completion(self) -> None:
        if self._completer.popup().isVisible():
            self._show_completion()

    def _show_completion(self) -> None:
        context = self._completion_context()
        if context is None:
            self._completer.popup().hide()
            return
        obj, prefix = context
        candidates = [
            item for item in _COMPLETION_MAP[obj]
            if item.lower().startswith(prefix.lower())
        ]
        if not candidates:
            self._completer.popup().hide()
            return
        self._completion_prefix = prefix
        self._completion_model.setStringList(candidates)
        self._completer.setCompletionPrefix(prefix)
        popup = self._completer.popup()
        popup.setCurrentIndex(self._completer.completionModel().index(0, 0))
        rect = self.editor.cursorRect()
        rect.setWidth(max(260, popup.sizeHintForColumn(0) + popup.verticalScrollBar().sizeHint().width()))
        self._completer.complete(rect)

    def _insert_completion(self, completion: str) -> None:
        cursor = self.editor.textCursor()
        for _ in self._completion_prefix:
            cursor.deletePreviousChar()
        cursor.insertText(completion)
        self.editor.setTextCursor(cursor)

    def _reset_namespace(self) -> None:
        mfv_module = install_runtime_module(self._state.mfv)
        self._namespace = {
            "__name__": "__mfv_script__",
            "__file__": str(self._path) if self._path else "<script-editor>",
            "mfv": mfv_module,
            "np": np,
            "numpy": np,
        }

    def _append_output(self, text: str, *, is_err: bool = False) -> None:
        self.output.moveCursor(QTextCursor.MoveOperation.End)
        self.output.insertPlainText(text)
        self.output.verticalScrollBar().setValue(self.output.verticalScrollBar().maximum())
        ConsoleWindow.write_app_message(text, is_err=is_err, show_on_error=is_err)

    def _clear_output(self) -> None:
        self.output.clear()

    def _run_all(self) -> None:
        self._execute(self.editor.toPlainText())

    def _run_selection(self) -> None:
        cursor = self.editor.textCursor()
        code = cursor.selectedText().replace("\u2029", "\n")
        if not code.strip():
            code = self.editor.toPlainText()
        self._execute(code)

    def _execute(self, code: str) -> None:
        if not code.strip():
            return
        install_runtime_module(self._state.mfv)
        self._namespace["__file__"] = str(self._path) if self._path else "<script-editor>"
        self._append_output("\n>>> Running script\n")
        stdout = _ScriptOutputStream(self._append_output, is_err=False)
        stderr = _ScriptOutputStream(self._append_output, is_err=True)
        try:
            with contextlib.redirect_stdout(stdout), contextlib.redirect_stderr(stderr):
                exec(compile(code, str(self._namespace["__file__"]), "exec"), self._namespace)
        except BaseException:
            self._append_output(traceback.format_exc(), is_err=True)
        else:
            self._append_output(">>> Done\n")

    def _open_script(self) -> None:
        path, _ = QFileDialog.getOpenFileName(
            self,
            "Open Python script",
            str(self._path.parent if self._path else Path.home()),
            "Python scripts (*.py);;All files (*)",
        )
        if not path:
            return
        try:
            text = Path(path).read_text(encoding="utf-8")
        except Exception as exc:
            QMessageBox.critical(self, "Open script", f"Could not open script:\n{exc}")
            return
        self._path = Path(path)
        self.editor.setPlainText(text)
        self._update_title()
        self._reset_namespace()

    def _save_script(self) -> None:
        if self._path is None:
            self._save_script_as()
            return
        self._write_script(self._path)

    def _save_script_as(self) -> None:
        path, _ = QFileDialog.getSaveFileName(
            self,
            "Save Python script",
            str(self._path if self._path else Path.home() / "minflux_script.py"),
            "Python scripts (*.py);;All files (*)",
        )
        if not path:
            return
        self._path = Path(path)
        self._write_script(self._path)
        self._update_title()

    def _write_script(self, path: Path) -> None:
        try:
            path.write_text(self.editor.toPlainText(), encoding="utf-8")
        except Exception as exc:
            QMessageBox.critical(self, "Save script", f"Could not save script:\n{exc}")
            return
        self._append_output(f"Saved script: {path}\n")

    def _update_title(self) -> None:
        suffix = f" - {self._path.name}" if self._path else ""
        self.setWindowTitle(f"Script Editor{suffix}")

    def _show_api_help(self) -> None:
        dialog = QDialog(self)
        dialog.setWindowTitle("Script Editor API Help")
        dialog.resize(760, 560)
        layout = QVBoxLayout(dialog)
        text = QPlainTextEdit(dialog)
        text.setReadOnly(True)
        text.setFont(self._mono_font(10))
        text.setPlainText(_API_HELP)
        layout.addWidget(text)
        buttons = QDialogButtonBox(QDialogButtonBox.StandardButton.Close, dialog)
        buttons.rejected.connect(dialog.reject)
        layout.addWidget(buttons)
        dialog.exec()

    def closeEvent(self, event) -> None:  # noqa: N802 - Qt API
        if self._force_close:
            event.accept()
            return
        self.hide()
        event.ignore()

    def force_close(self) -> None:
        """Allow the main application shutdown path to really close the editor."""
        self._force_close = True
        self.close()
