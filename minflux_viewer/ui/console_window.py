"""
minflux_viewer.ui.console_window
=================================
Stdout / stderr console window.

This window captures everything written to ``sys.stdout`` and
``sys.stderr`` after redirection is installed, so the user can see runtime
errors and ``print()`` output inside the viewer without also echoing it to
the terminal that launched the app.

Distinct from the Log window:
* **Log**      — structured application events (dataset loaded, filter
                 applied, etc.) emitted via ``AppState.log()``.
* **Console**  — raw stdout/stderr, including Python tracebacks and
                 library messages.
"""

from __future__ import annotations

import io
import sys
from datetime import datetime

from PyQt6.QtCore import QThread, Qt, pyqtSignal
from PyQt6.QtGui import QColor, QFont, QTextCharFormat, QTextCursor
from PyQt6.QtWidgets import (
    QHBoxLayout,
    QApplication,
    QPlainTextEdit,
    QPushButton,
    QSizePolicy,
    QVBoxLayout,
    QWidget,
)


# ---------------------------------------------------------------------------
# Stream capture — sends writes to the viewer Console only
# ---------------------------------------------------------------------------

class _ConsoleCaptureStream(io.TextIOBase):
    """
    Thin wrapper around a file-like stream that redirects writes to the
    viewer Console.

    The original stream is retained only for metadata-like operations such as
    ``encoding`` and ``fileno``. Writes are not forwarded to the terminal.
    """

    def __init__(self, real_stream, emit_fn):
        self._real = real_stream
        self._emit = emit_fn

    # io.TextIOBase delegations
    def write(self, data: str) -> int:
        try:
            self._emit(data)
        except Exception:
            # Never let a UI-side failure break stdout
            pass
        return len(data)

    def flush(self) -> None:
        try:
            self._real.flush()
        except Exception:
            pass

    def isatty(self) -> bool:
        try:
            return self._real.isatty()
        except Exception:
            return False

    def fileno(self):
        return self._real.fileno()

    @property
    def encoding(self):
        return getattr(self._real, "encoding", "utf-8")


# ---------------------------------------------------------------------------
# ConsoleWindow
# ---------------------------------------------------------------------------

class ConsoleWindow(QWidget):
    """
    Floating stdout / stderr console.

    Call ``install_redirection()`` once (at app startup) to start capturing
    output. Output before the window is opened is still captured and shown
    when the window appears.
    """

    TAG = "console_window"

    # Signals are emitted from the stream-redirect callback which may run
    # on any thread that prints; Qt's queued-connection semantics make this
    # safe for widget updates.
    _stdout_written = pyqtSignal(str)
    _stderr_written = pyqtSignal(str)

    _instance: "ConsoleWindow | None" = None
    _buffered_stdout: list[str] = []
    _buffered_stderr: list[str] = []

    def __init__(self, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self._autoscroll = True

        self.setWindowTitle("Console (stdout / stderr)")
        self.setWindowFlags(Qt.WindowType.Window | Qt.WindowType.Tool)
        self.resize(780, 320)
        self.setAttribute(Qt.WidgetAttribute.WA_DeleteOnClose, False)

        self._build_ui()

        # Subscribe to redirected streams
        self._stdout_written.connect(lambda s: self._append(s, is_err=False))
        self._stderr_written.connect(lambda s: self._append(s, is_err=True))

        # Flush any output captured before the window was opened
        if ConsoleWindow._buffered_stdout:
            self._append("".join(ConsoleWindow._buffered_stdout), is_err=False)
            ConsoleWindow._buffered_stdout.clear()
        if ConsoleWindow._buffered_stderr:
            self._append("".join(ConsoleWindow._buffered_stderr), is_err=True)
            ConsoleWindow._buffered_stderr.clear()

        ConsoleWindow._instance = self

    # ------------------------------------------------------------------
    # UI
    # ------------------------------------------------------------------

    def _build_ui(self) -> None:
        root = QVBoxLayout(self)
        root.setContentsMargins(4, 4, 4, 4)
        root.setSpacing(4)

        self._text = QPlainTextEdit()
        self._text.setReadOnly(True)
        self._text.setMaximumBlockCount(5000)
        self._text.setSizePolicy(
            QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Expanding
        )
        font = QFont("Consolas, Courier New, monospace")
        font.setPointSize(10)
        font.setStyleHint(QFont.StyleHint.Monospace)
        self._text.setFont(font)
        self._text.setStyleSheet(
            "QPlainTextEdit {"
            "  background: #1e1e1e;"
            "  color: #d4d4d4;"
            "  border: none;"
            "  padding: 4px;"
            "}"
        )
        root.addWidget(self._text)

        bar = QHBoxLayout()
        bar.setSpacing(6)
        copy_btn = QPushButton("Copy all")
        copy_btn.setFixedHeight(24)
        copy_btn.clicked.connect(self._copy_all)
        bar.addWidget(copy_btn)

        clear_btn = QPushButton("Clear")
        clear_btn.setFixedHeight(24)
        clear_btn.clicked.connect(self._text.clear)
        bar.addWidget(clear_btn)

        bar.addStretch()

        root.addLayout(bar)

    # ------------------------------------------------------------------
    # Append with colouring
    # ------------------------------------------------------------------

    def _append(self, text: str, *, is_err: bool) -> None:
        if not text:
            return
        colour = "#ff7b72" if is_err else "#d4d4d4"
        cursor = self._text.textCursor()
        cursor.movePosition(QTextCursor.MoveOperation.End)
        fmt = QTextCharFormat()
        fmt.setForeground(QColor(colour))
        cursor.setCharFormat(fmt)
        cursor.insertText(text)
        if self._autoscroll:
            self._text.verticalScrollBar().setValue(
                self._text.verticalScrollBar().maximum()
            )

    def _copy_all(self) -> None:
        from PyQt6.QtWidgets import QApplication
        QApplication.clipboard().setText(self._text.toPlainText())

    @classmethod
    def _ensure_visible_for_stderr(cls) -> "ConsoleWindow | None":
        app = QApplication.instance()
        if app is None:
            return None
        if QThread.currentThread() is not app.thread():
            return None
        if cls._instance is None:
            cls._instance = ConsoleWindow()
        if not cls._instance.isVisible():
            cls._instance.show()
        cls._instance.raise_()
        cls._instance.activateWindow()
        return cls._instance

    @classmethod
    def write_app_message(
        cls,
        text: str,
        *,
        is_err: bool = False,
        show_on_error: bool = False,
    ) -> None:
        """
        Append an application-generated message to the in-app Console only.

        This is intentionally not a ``print()`` replacement: it bypasses the
        tee stream so messages can be shown in the viewer Console without also
        writing to the terminal that launched the app.
        """
        if not text:
            return
        if not text.endswith("\n"):
            text += "\n"
        instance = cls._instance
        if instance is None and is_err and show_on_error:
            instance = cls._ensure_visible_for_stderr()
        if instance is not None:
            signal = instance._stderr_written if is_err else instance._stdout_written
            signal.emit(text)
        elif is_err:
            cls._buffered_stderr.append(text)
        else:
            cls._buffered_stdout.append(text)


# ---------------------------------------------------------------------------
# One-shot redirection installer — call once at app startup
# ---------------------------------------------------------------------------

_installed = False
_orig_stdout = None
_orig_stderr = None


def install_redirection() -> None:
    """
    Replace ``sys.stdout`` and ``sys.stderr`` with viewer-only capture streams.

    Call once at application startup. Safe to call before any ConsoleWindow
    is created. Stdout is buffered until the Console is opened; stderr opens
    the Console automatically once a QApplication exists.
    """
    global _installed, _orig_stdout, _orig_stderr
    if _installed:
        return
    _installed = True

    _orig_stdout = sys.stdout
    _orig_stderr = sys.stderr

    def _push_stdout(data: str) -> None:
        ConsoleWindow.write_app_message(data, is_err=False)

    def _push_stderr(data: str) -> None:
        ConsoleWindow.write_app_message(data, is_err=True, show_on_error=True)

    sys.stdout = _ConsoleCaptureStream(_orig_stdout, _push_stdout)
    sys.stderr = _ConsoleCaptureStream(_orig_stderr, _push_stderr)


def restore_redirection() -> None:
    """Restore the original terminal streams during application shutdown."""
    global _installed, _orig_stdout, _orig_stderr
    if not _installed:
        return
    if _orig_stdout is not None:
        sys.stdout = _orig_stdout
    if _orig_stderr is not None:
        sys.stderr = _orig_stderr
    _installed = False
