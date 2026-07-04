"""
minflux_viewer.ui.log_window
=============================
Application log / console window.

A floating (or dockable) window that displays timestamped messages about
everything that happens in the viewer: files loaded, filters applied,
errors, unsupported drops, dataset changes, etc.

All parts of the application post messages via ``AppState.log()``, which
emits the ``log_message`` signal.  The log window subscribes to that signal
and appends each entry.

Features
--------
* Timestamped lines with severity colouring (INFO / WARN / ERROR)
* Copy-to-clipboard button
* Clear button
* Auto-scroll that can be paused by scrolling up
* Persists across dataset changes — it is opened once and stays open
"""

from __future__ import annotations

from datetime import datetime

from PyQt6.QtCore import QEventLoop, Qt
from PyQt6.QtGui import QColor, QFont, QFontDatabase, QTextCharFormat, QTextCursor
from PyQt6.QtWidgets import (
    QApplication,
    QHBoxLayout,
    QPlainTextEdit,
    QPushButton,
    QSizePolicy,
    QVBoxLayout,
    QWidget,
)


# ---------------------------------------------------------------------------
# Severity levels
# ---------------------------------------------------------------------------

class Level:
    INFO  = "INFO"
    WARN  = "WARN"
    ERROR = "ERROR"
    DEBUG = "DEBUG"


# Colours (compatible with both light and dark Qt palettes)
_LEVEL_COLOUR: dict[str, str] = {
    Level.INFO:  "#000000",
    Level.WARN:  "#000000",
    Level.ERROR: "#000000",
    Level.DEBUG: "#000000",
}


# ---------------------------------------------------------------------------
# LogWindow
# ---------------------------------------------------------------------------

class LogWindow(QWidget):
    """
    Floating log / console window.

    Usage — post a message from anywhere that has access to AppState::

        state.log("Dataset loaded: my_data.mat")
        state.log("Filter applied: 12,345 / 45,231 locs pass", level="INFO")
        state.log("Unsupported file type: foo.xyz", level="WARN")
        state.log("Parse failed: ...", level="ERROR")
    """

    TAG = "log_window"

    def __init__(self, state, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self._state      = state
        self._autoscroll = True      # pause autoscroll when user scrolls up
        self._progress_active = False  # a refreshing progress line is open

        self.setWindowTitle("Log (events)")
        self.setWindowFlags(Qt.WindowType.Window)
        self.resize(700, 280)
        self.setAttribute(Qt.WidgetAttribute.WA_DeleteOnClose, False)  # don't delete on close

        self._build_ui()

        # Post a startup banner, then replay events already logged while the window
        # was closed (they accumulate in state.log_history). The Log window is no
        # longer auto-shown on the first message, so opening it later must still
        # show the full history rather than only messages from this point on.
        self._append("MINFLUX Data Viewer started", Level.INFO)
        try:
            for entry in state.log_history:
                self._append(str(entry.get("message", "")), entry.get("level", Level.INFO))
        except Exception:
            pass

        # Connect to AppState log signals for live updates
        state.log_message.connect(self._append)
        state.progress_log.connect(self._set_progress_line)

    # ------------------------------------------------------------------
    # UI
    # ------------------------------------------------------------------

    def _build_ui(self) -> None:
        root = QVBoxLayout(self)
        root.setContentsMargins(4, 4, 4, 4)
        root.setSpacing(4)

        # ── Text area ─────────────────────────────────────────────
        self._text = QPlainTextEdit()
        self._text.setReadOnly(True)
        self._text.setMaximumBlockCount(5000)   # keep last 5000 lines
        self._text.setSizePolicy(
            QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Expanding
        )

        font = QFontDatabase.systemFont(QFontDatabase.SystemFont.FixedFont)
        font.setPointSize(10)
        font.setStyleHint(QFont.StyleHint.Monospace)
        self._text.setFont(font)
        self._text.setStyleSheet(
            "QPlainTextEdit {"
            "  background: #ffffff;"
            "  color: #000000;"
            "  border: 1px solid #b8b8b8;"
            "  padding: 4px;"
            "}"
        )

        # Detect manual scroll to pause autoscroll
        sb = self._text.verticalScrollBar()
        sb.valueChanged.connect(self._on_scroll)

        root.addWidget(self._text)

        # ── Button row ────────────────────────────────────────────
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

        self._scroll_btn = QPushButton("▼ Auto-scroll: ON")
        self._scroll_btn.setFixedHeight(24)
        self._scroll_btn.setCheckable(True)
        self._scroll_btn.setChecked(True)
        self._scroll_btn.clicked.connect(self._toggle_autoscroll)
        bar.addWidget(self._scroll_btn)

        root.addLayout(bar)

    # ------------------------------------------------------------------
    # Append
    # ------------------------------------------------------------------

    def _append(self, message: str, level: str = Level.INFO) -> None:
        """Append a timestamped, coloured log line."""
        # If a refreshing progress line is open, freeze it on its own line first
        # so this message doesn't get concatenated onto the bar.
        if self._progress_active:
            c = self._text.textCursor()
            c.movePosition(QTextCursor.MoveOperation.End)
            c.insertText("\n")
            self._progress_active = False

        ts    = datetime.now().strftime("%H:%M:%S")
        line  = f"[{ts}] [{level:5s}]  {message}"
        colour = _LEVEL_COLOUR.get(level, "#000000")

        cursor = self._text.textCursor()
        cursor.movePosition(QTextCursor.MoveOperation.End)

        fmt = QTextCharFormat()
        fmt.setForeground(QColor(colour))
        cursor.setCharFormat(fmt)
        cursor.insertText(line + "\n")

        if self._autoscroll:
            self._text.verticalScrollBar().setValue(
                self._text.verticalScrollBar().maximum()
            )

    def _set_progress_line(self, text: str, final: bool = False) -> None:
        """Update one refreshing progress line in place (Fiji-style).

        The line replaces itself on each call; ``final=True`` commits it (e.g.
        the DONE bar). ``processEvents`` is scoped to this widget so the bar
        repaints during a synchronous computation without disturbing the rest
        of the UI (the status bar, in particular, is untouched).
        """
        cursor = self._text.textCursor()
        cursor.movePosition(QTextCursor.MoveOperation.End)
        if self._progress_active:                     # replace the current line
            cursor.movePosition(
                QTextCursor.MoveOperation.StartOfBlock, QTextCursor.MoveMode.KeepAnchor
            )
            cursor.removeSelectedText()

        fmt = QTextCharFormat()
        fmt.setForeground(QColor(_LEVEL_COLOUR.get(Level.INFO, "#000000")))
        cursor.setCharFormat(fmt)
        cursor.insertText(text)

        if final:
            cursor.insertText("\n")
            self._progress_active = False
        else:
            self._progress_active = True

        if self._autoscroll:
            self._text.verticalScrollBar().setValue(
                self._text.verticalScrollBar().maximum()
            )
        QApplication.processEvents(QEventLoop.ProcessEventsFlag.ExcludeUserInputEvents)

    # ------------------------------------------------------------------
    # Scroll / copy / clear
    # ------------------------------------------------------------------

    def _on_scroll(self, value: int) -> None:
        sb  = self._text.verticalScrollBar()
        at_bottom = value >= sb.maximum() - 4
        if at_bottom and not self._autoscroll:
            self._set_autoscroll(True)
        elif not at_bottom and self._autoscroll:
            self._set_autoscroll(False)

    def _toggle_autoscroll(self) -> None:
        self._set_autoscroll(self._scroll_btn.isChecked())

    def _set_autoscroll(self, enabled: bool) -> None:
        self._autoscroll = enabled
        self._scroll_btn.setChecked(enabled)
        self._scroll_btn.setText(
            "▼ Auto-scroll: ON" if enabled else "▼ Auto-scroll: OFF"
        )

    def _copy_all(self) -> None:
        from PyQt6.QtWidgets import QApplication
        QApplication.clipboard().setText(self._text.toPlainText())
