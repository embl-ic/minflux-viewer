"""Modeless Fiji-style macro recorder window."""

from __future__ import annotations

from pathlib import Path

from PyQt6.QtCore import Qt, QTimer
from PyQt6.QtGui import QCloseEvent, QFontDatabase
from PyQt6.QtWidgets import (
    QCheckBox,
    QFileDialog,
    QHBoxLayout,
    QLabel,
    QMessageBox,
    QPlainTextEdit,
    QPushButton,
    QVBoxLayout,
    QWidget,
)


class MacroRecorderWindow(QWidget):
    """Live view of the application recorder, with export controls."""

    TAG = "macro_recorder_window"

    def __init__(self, state, owner=None) -> None:
        # Modeless plugin windows are deliberately non-owned top-level windows.
        super().__init__(None)
        self._state = state
        self._owner = owner
        self._recorder = state.recorder
        self._last_script: str | None = None

        self.setObjectName(self.TAG)
        self.setWindowTitle("Macro Recorder")
        self.setWindowFlags(Qt.WindowType.Window)
        self.resize(900, 650)
        self._build_ui()

        # Opening the window starts (or resumes) recording, preserving any
        # steps deliberately captured through mfv.record before it was opened.
        if not self._recorder.enabled:
            self._recorder.start(clear=False)

        self._timer = QTimer(self)
        self._timer.setInterval(250)
        self._timer.timeout.connect(self._refresh)
        self._timer.start()
        self._refresh(force=True)

    def _build_ui(self) -> None:
        root = QVBoxLayout(self)
        root.setContentsMargins(8, 8, 8, 8)
        root.setSpacing(7)

        controls = QHBoxLayout()
        self.record_button = QPushButton("Record", self)
        self.pause_button = QPushButton("Pause", self)
        self.clear_button = QPushButton("Clear", self)
        self.silent_checkbox = QCheckBox("Silent mode", self)
        self.silent_checkbox.setToolTip(
            "Omit display-only steps; data-producing GUI outcomes are kept."
        )
        self.create_script_button = QPushButton("Create Script", self)
        self.save_button = QPushButton("Save...", self)

        controls.addWidget(self.record_button)
        controls.addWidget(self.pause_button)
        controls.addWidget(self.clear_button)
        controls.addSpacing(12)
        controls.addWidget(self.silent_checkbox)
        controls.addStretch(1)
        controls.addWidget(self.create_script_button)
        controls.addWidget(self.save_button)
        root.addLayout(controls)

        self.status_label = QLabel(self)
        root.addWidget(self.status_label)

        self.code_view = QPlainTextEdit(self)
        self.code_view.setReadOnly(True)
        self.code_view.setLineWrapMode(QPlainTextEdit.LineWrapMode.NoWrap)
        self.code_view.setFont(
            QFontDatabase.systemFont(QFontDatabase.SystemFont.FixedFont)
        )
        self.code_view.setToolTip(
            "Unknown commands remain as TODO comments until their parameters "
            "and mfv call are instrumented."
        )
        root.addWidget(self.code_view, 1)

        self.record_button.clicked.connect(self._record)
        self.pause_button.clicked.connect(self._pause)
        self.clear_button.clicked.connect(self._clear)
        self.silent_checkbox.toggled.connect(lambda _checked: self._refresh(force=True))
        self.create_script_button.clicked.connect(self._create_script)
        self.save_button.clicked.connect(self._save)

    def _script(self) -> str:
        return self._recorder.script(silent=self.silent_checkbox.isChecked())

    def _refresh(self, *, force: bool = False) -> None:
        script = self._script()
        if force or script != self._last_script:
            scrollbar = self.code_view.verticalScrollBar()
            was_at_end = scrollbar.value() >= scrollbar.maximum()
            self.code_view.setPlainText(script)
            if was_at_end:
                scrollbar.setValue(scrollbar.maximum())
            self._last_script = script

        count = len(self._recorder.steps())
        mode = "Recording" if self._recorder.enabled else "Paused"
        noun = "step" if count == 1 else "steps"
        self.status_label.setText(f"{mode} · {count} {noun}")
        self.record_button.setEnabled(not self._recorder.enabled)
        self.pause_button.setEnabled(self._recorder.enabled)

    def _record(self) -> None:
        self._recorder.start(clear=False)
        self._refresh(force=True)

    def _pause(self) -> None:
        self._recorder.stop()
        self._refresh(force=True)

    def _clear(self) -> None:
        self._recorder.clear()
        self._refresh(force=True)

    def _create_script(self) -> None:
        show_editor = getattr(self._owner, "show_script_editor", None)
        try:
            editor = show_editor() if callable(show_editor) else None
            if editor is None:
                editor = self._state.mfv.view.script_editor()
            setter = getattr(editor, "set_script", None)
            if not callable(setter):
                raise RuntimeError("The Script Editor does not accept generated scripts.")
            setter(self._script(), source_name="Recorded Macro")
            editor.show()
            editor.raise_()
            editor.activateWindow()
        except Exception as exc:
            QMessageBox.warning(
                self,
                "Create Script",
                f"Could not open the recorded macro in the Script Editor:\n{exc}",
            )

    def _save(self) -> None:
        path, _selected_filter = QFileDialog.getSaveFileName(
            self,
            "Save recorded macro",
            str(Path.home() / "minflux_macro.py"),
            "Python scripts (*.py);;All files (*)",
        )
        if not path:
            return
        target = Path(path)
        if not target.suffix:
            target = target.with_suffix(".py")
        try:
            target.write_text(self._script(), encoding="utf-8")
        except Exception as exc:
            QMessageBox.warning(
                self, "Save recorded macro", f"Could not save the macro:\n{exc}"
            )
            return
        self._state.log(f"Saved recorded macro: {target}", "INFO")

    def closeEvent(self, event: QCloseEvent) -> None:  # noqa: N802 - Qt API
        self._timer.stop()
        self._recorder.stop()
        event.accept()


__all__ = ["MacroRecorderWindow"]
