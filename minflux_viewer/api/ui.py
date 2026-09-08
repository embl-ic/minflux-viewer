"""
``mfv.ui`` -- talking to the user.

Logging, the status bar, a generated parameter dialog, file pickers and message
boxes. A plugin that needs input should use :meth:`Ui.ask` rather than building
its own dialog: it keeps the plugin free of Qt, and it means the plugin cannot
violate the project's window-ownership rules by accident.
"""

from __future__ import annotations

import os
from collections.abc import Mapping
from typing import Any

from ._base import ApiError, Namespace

#: Levels accepted by :meth:`Ui.log`, matching ``AppState.log``.
LOG_LEVELS = ("DEBUG", "INFO", "WARN", "ERROR")


class Ui(Namespace):
    """Log, report progress, and ask the user for things."""

    name = "ui"

    # -- reporting -----------------------------------------------------------

    def log(self, message: str, level: str = "INFO", *, dataset: Any = None) -> None:
        """
        Write a line to the viewer Log.

        Passing *dataset* tags the entry with it, which is what lets the
        method-text generator attribute the event to the right dataset later.
        """
        idx = None if dataset is None else self._dataset_index(dataset)
        self._state.log(str(message), str(level).upper(), dataset_idx=idx)

    def status(self, text: str, fraction: float | None = None) -> None:
        """
        Set the main-window status line, optionally with a percentage.

        *fraction* is 0..1. For work running through :mod:`mfv.run`, prefer the
        task progress callback -- it drives this and the task monitor together.
        """
        self._state.status_progress(str(text), fraction)

    # -- asking --------------------------------------------------------------

    def ask(
        self,
        fields: Mapping[str, Any],
        *,
        title: str = "Parameters",
        descriptions: Mapping[str, str] | None = None,
    ) -> dict | None:
        """
        Show a generated parameter dialog and return the edited values.

        *fields* maps a name to a default, and the default's type picks the
        widget: ``bool`` a checkbox, ``int``/``float`` a spin box, ``str`` a
        line edit, and a ``list``/``tuple`` a dropdown whose first entry is
        preselected. *descriptions* supplies per-field hover help.

        Returns a dict of the same keys, or **None** if the user cancelled --
        check for ``None`` before proceeding.
        """
        from PyQt6.QtWidgets import (
            QCheckBox,
            QComboBox,
            QDialog,
            QDialogButtonBox,
            QDoubleSpinBox,
            QFormLayout,
            QLineEdit,
            QSpinBox,
            QVBoxLayout,
        )

        if not fields:
            raise ApiError("ask() needs at least one field.")
        help_text = dict(descriptions or {})

        dialog = QDialog(self._parent_widget())
        dialog.setWindowTitle(str(title))
        layout = QVBoxLayout(dialog)
        form = QFormLayout()
        layout.addLayout(form)

        widgets: dict[str, Any] = {}
        for key, default in fields.items():
            if isinstance(default, bool):
                widget = QCheckBox()
                widget.setChecked(default)
            elif isinstance(default, int):
                widget = QSpinBox()
                widget.setRange(-2_147_483_648, 2_147_483_647)
                widget.setValue(int(default))
            elif isinstance(default, float):
                widget = QDoubleSpinBox()
                widget.setDecimals(4)
                widget.setRange(-1e12, 1e12)
                widget.setValue(float(default))
            elif isinstance(default, (list, tuple)):
                widget = QComboBox()
                widget.addItems([str(v) for v in default])
            else:
                widget = QLineEdit(str(default))
            if key in help_text:
                widget.setToolTip(str(help_text[key]))
            widgets[key] = widget
            form.addRow(str(key), widget)

        buttons = QDialogButtonBox(
            QDialogButtonBox.StandardButton.Ok | QDialogButtonBox.StandardButton.Cancel
        )
        buttons.accepted.connect(dialog.accept)
        buttons.rejected.connect(dialog.reject)
        layout.addWidget(buttons)

        if dialog.exec() != QDialog.DialogCode.Accepted:
            return None

        out: dict[str, Any] = {}
        for key, widget in widgets.items():
            if isinstance(widget, QCheckBox):
                out[key] = widget.isChecked()
            elif isinstance(widget, (QSpinBox, QDoubleSpinBox)):
                out[key] = widget.value()
            elif isinstance(widget, QComboBox):
                out[key] = widget.currentText()
            else:
                out[key] = widget.text()
        return out

    def confirm(self, message: str, *, title: str = "Confirm") -> bool:
        """Yes/no question. Returns whether the user accepted."""
        from PyQt6.QtWidgets import QMessageBox

        answer = QMessageBox.question(self._parent_widget(), str(title), str(message))
        return answer == QMessageBox.StandardButton.Yes

    def info(self, message: str, *, title: str = "MINFLUX Viewer") -> None:
        """Show an informational message box."""
        from PyQt6.QtWidgets import QMessageBox

        QMessageBox.information(self._parent_widget(), str(title), str(message))

    def warn(self, message: str, *, title: str = "Warning") -> None:
        """Show a warning message box, and log the same text at WARN."""
        from PyQt6.QtWidgets import QMessageBox

        self.log(message, "WARN")
        QMessageBox.warning(self._parent_widget(), str(title), str(message))

    def error(self, message: str, *, title: str = "Error") -> None:
        """
        Show an error message box, and log the same text at ERROR.

        Say what went wrong and how to fix it -- a plugin's error reaches a user
        who did not write the plugin.
        """
        from PyQt6.QtWidgets import QMessageBox

        self.log(message, "ERROR")
        QMessageBox.critical(self._parent_widget(), str(title), str(message))

    # -- files ---------------------------------------------------------------

    def choose_file(
        self,
        *,
        caption: str = "Choose a file",
        filter: str = "All files (*)",
        directory: str | None = None,
        save: bool = False,
    ) -> str | None:
        """Open a file picker. Returns the chosen path, or ``None`` if cancelled."""
        from PyQt6.QtWidgets import QFileDialog

        parent = self._parent_widget()
        start = directory or self._default_folder()
        if save:
            path, _ = QFileDialog.getSaveFileName(parent, caption, start, filter)
        else:
            path, _ = QFileDialog.getOpenFileName(parent, caption, start, filter)
        return path or None

    def choose_files(
        self,
        *,
        caption: str = "Choose files",
        filter: str = "All files (*)",
        directory: str | None = None,
    ) -> list[str]:
        """Open a multi-select file picker. Returns ``[]`` if cancelled."""
        from PyQt6.QtWidgets import QFileDialog

        paths, _ = QFileDialog.getOpenFileNames(
            self._parent_widget(), caption, directory or self._default_folder(), filter,
        )
        return list(paths or [])

    def choose_dir(
        self,
        *,
        caption: str = "Choose a folder",
        directory: str | None = None,
    ) -> str | None:
        """Open a directory picker. Returns the chosen path, or ``None``."""
        from PyQt6.QtWidgets import QFileDialog

        path = QFileDialog.getExistingDirectory(
            self._parent_widget(), caption, directory or self._default_folder(),
        )
        return path or None

    def drop_files(self, paths: Any) -> None:
        """Replay files/folders being dropped on the main viewer window.

        This is the interactive routing primitive emitted by the Macro
        Recorder for a physical OS drag-and-drop. It deliberately follows the
        same broad route as the gesture: an MSR opens the MSR Reader, a filter
        preset opens the Filter dialog, and an unsupported or unreadable path
        is reported in the Log. Consequently this call is ``GUI_ONLY`` in a
        recording and is omitted from Silent mode; a successful import's
        resolved, batch-capable load is a separate semantic recorder step.

        The drop attempt exists even when routing later fails. This is useful
        both for diagnosing a recording and for faithfully replaying an
        interactive session without synthesizing mouse events.
        """
        if isinstance(paths, (str, bytes, os.PathLike)):
            raw_paths = [paths]
        else:
            try:
                raw_paths = list(paths)
            except TypeError:
                raise ApiError(
                    "drop_files() needs a path or an iterable of paths."
                ) from None
        normalized: list[str] = []
        for index, path in enumerate(raw_paths):
            try:
                text = os.fsdecode(os.fspath(path))
            except TypeError:
                raise ApiError(
                    f"drop_files() path {index + 1} is not path-like: "
                    f"{type(path).__name__}."
                ) from None
            if text.strip():
                normalized.append(text)
        if not normalized:
            raise ApiError("drop_files() needs at least one non-empty local path.")
        self._main_window().route_paths(normalized)

    # -- internal ------------------------------------------------------------

    def _parent_widget(self):
        """
        The main window when one is bound, else ``None``.

        A modal dialog is one of the few places a Qt parent is correct here:
        it centres the dialog on the application and keeps it above it. Modeless
        windows must stay unparented -- see the project's window rules.
        """
        try:
            return self._facade.require_main_window()
        except Exception:
            return None

    def _default_folder(self) -> str:
        return str(self._state.prefs.get("file", {}).get("default_folder", "") or "")
