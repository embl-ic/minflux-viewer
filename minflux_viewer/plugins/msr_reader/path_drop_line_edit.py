'Path drop line edit'

from __future__ import annotations

from pathlib import Path

from PyQt6.QtCore import pyqtSignal
from PyQt6.QtWidgets import (
    QLineEdit,
)


class PathDropLineEdit(QLineEdit):
    """Path field that accepts a dropped file or folder.

    ``accept`` decides which dropped paths are taken: ``"dir"`` for a folder
    only, ``"msr_or_dir"`` for an ``.msr`` file or a folder.  The first
    acceptable path of the drop wins, and :attr:`pathDropped` carries it so the
    owner can react (switch input mode, start a parse, persist settings).
    """

    pathDropped = pyqtSignal(str)

    def __init__(self, text: str = "", *, accept: str = "msr_or_dir", parent=None):
        super().__init__(text, parent)
        self._accept = accept
        self.setAcceptDrops(True)

    def _acceptable(self, path: str) -> bool:
        if not path:
            return False
        candidate = Path(path)
        if candidate.is_dir():
            return True
        if self._accept == "dir":
            return False
        return candidate.suffix.lower() == ".msr" and candidate.is_file()

    def _first_acceptable(self, event) -> str:
        data = event.mimeData()
        if data is None or not data.hasUrls():
            return ""
        for url in data.urls():
            local = url.toLocalFile()
            if self._acceptable(local):
                return local
        return ""

    @staticmethod
    def _has_urls(event) -> bool:
        data = event.mimeData()
        return bool(data is not None and data.hasUrls())

    def _handle_drag(self, event, fallback) -> None:
        """A file/folder drag is ours to judge; anything else is plain text.

        Delegating a rejected *file* drag to QLineEdit would let it through as
        text and paste the ``file:///…`` URL into the path, which looks like a
        successful drop of an unsupported file.
        """
        if self._has_urls(event):
            if self._first_acceptable(event):
                event.acceptProposedAction()
            else:
                event.ignore()
            return
        fallback(event)

    def dragEnterEvent(self, event):
        self._handle_drag(event, super().dragEnterEvent)

    def dragMoveEvent(self, event):
        self._handle_drag(event, super().dragMoveEvent)

    def dropEvent(self, event):
        if not self._has_urls(event):
            super().dropEvent(event)
            return
        path = self._first_acceptable(event)
        if not path:
            event.ignore()
            return
        event.acceptProposedAction()
        self.setText(path)
        self.pathDropped.emit(path)
