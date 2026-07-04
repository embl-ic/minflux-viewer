"""
minflux_viewer.ui.command_finder
================================
Fiji-style **command finder** for the main-window search bar.

* :func:`collect_commands` walks a ``QMenuBar`` and returns every leaf command as a
  :class:`CommandEntry` (display text, menu path, shortcut, the backing ``QAction``).
* :func:`filter_commands` ranks entries against a query (case-insensitive,
  all-tokens-must-match substring, name matches ranked above menu-path matches).
* :class:`CommandFinderDialog` is the full list dialog (filter field + table,
  run on double-click / Enter).
* :class:`SearchLineEdit` is the toolbar field, emitting ``focusedIn`` (rebuild the
  index) and ``doubleClicked`` (open the dialog).

Pure-ish: the collection/filter helpers are unit-testable with a bare ``QMenuBar``
(no main window).
"""

from __future__ import annotations

from dataclasses import dataclass

from PyQt6.QtCore import QEvent, Qt, pyqtSignal
from PyQt6.QtGui import QColor
from PyQt6.QtWidgets import (
    QAbstractItemView,
    QDialog,
    QHeaderView,
    QLabel,
    QLineEdit,
    QListView,
    QTableWidget,
    QTableWidgetItem,
    QVBoxLayout,
)


@dataclass
class CommandEntry:
    text: str            # cleaned command label (no '&' mnemonic / trailing '…')
    path: str            # menu path, e.g. "Analyze › Segmentation › NPC"
    shortcut: str        # portable shortcut text, or ""
    action: object       # the backing QAction
    enabled: bool
    source: str = ""     # implementing python file/library (plugins), or ""
    keywords: str = ""   # extra search terms not in name/path (tags/synonyms)


def source_of(fn) -> str:
    """Best-effort source file for a callable, as ``minflux_viewer/…`` (or basename).

    Used to tag plugin actions with the file that implements them; falls back to the
    module name. Never raises."""
    import inspect
    import os
    try:
        path = inspect.getsourcefile(fn) or inspect.getfile(fn)
    except Exception:
        path = None
    if path:
        norm = str(path).replace("\\", "/")
        i = norm.rfind("/minflux_viewer/")
        return norm[i + 1:] if i >= 0 else os.path.basename(norm)
    return str(getattr(fn, "__module__", "") or "")


def _clean(text: str) -> str:
    t = (text or "").replace("&", "").strip()
    for suffix in ("...", "…"):
        if t.endswith(suffix):
            t = t[: -len(suffix)].strip()
    return t


def _excluded(obj) -> bool:
    """A menu / action flagged ``setProperty("command_finder_exclude", True)`` — and
    everything under it — is left out of the index (e.g. the *Open Recent* submenu,
    whose entries are file paths, not commands)."""
    try:
        return bool(obj.property("command_finder_exclude"))
    except Exception:
        return False


def collect_commands(menubar) -> list[CommandEntry]:
    """Every leaf command under *menubar*, with its full menu path.

    Skips separators, submenu-opening actions, and anything under a menu/action
    flagged with the ``command_finder_exclude`` property (file lists, etc.)."""
    entries: list[CommandEntry] = []
    seen: set[int] = set()

    def walk(menu, path: list[str]) -> None:
        for act in menu.actions():
            if act.isSeparator() or _excluded(act):
                continue
            sub = act.menu()
            label = _clean(act.text())
            if sub is not None:
                if _excluded(sub):
                    continue
                walk(sub, path + ([label] if label else []))
            elif label:
                if id(act) in seen:              # same QAction reachable via 2 menus
                    continue
                seen.add(id(act))
                sc = act.shortcut()
                entries.append(CommandEntry(
                    text=label,
                    path=" › ".join(path),
                    shortcut=("" if sc is None or sc.isEmpty() else sc.toString()),
                    action=act,
                    enabled=bool(act.isEnabled()),
                    source=str(act.property("command_source") or ""),
                    keywords=str(act.property("command_keywords") or ""),
                ))

    for act in menubar.actions():
        if _excluded(act):
            continue
        sub = act.menu()
        if sub is not None and not _excluded(sub):
            walk(sub, [_clean(act.text())])
    return entries


def filter_commands(entries, query: str) -> list[CommandEntry]:
    """Rank *entries* for *query* (all tokens must match text+path substrings)."""
    q = (query or "").strip().lower()
    if not q:
        return sorted(entries, key=lambda e: (e.path.lower(), e.text.lower()))
    tokens = q.split()
    scored = []
    for e in entries:
        name = e.text.lower()
        # Keywords/tags join the haystack (but not the name-rank), so a leaf like
        # "2D" under Convolution is findable by "convolution"/"matched filter".
        hay = f"{e.text} {e.path} {e.source} {e.keywords}".lower()
        if not all(tok in hay for tok in tokens):
            continue
        if name == q:
            score = 0
        elif name.startswith(q):
            score = 1
        elif q in name:
            score = 2
        elif all(tok in name for tok in tokens):
            score = 3
        else:
            score = 4                            # matched only via the menu path
        scored.append((score, name, e))
    scored.sort(key=lambda x: (x[0], x[1]))
    return [e for _s, _n, e in scored]


class WideCompleterPopup(QListView):
    """Completer dropdown that stays fully on-screen when widened.

    A ``QCompleter`` sizes its popup to the search field's (small) width and anchors
    it at the field's bottom-left. We widen it (``setMinimumWidth`` to fit the
    longest command, which ``setGeometry`` then honours) so full text is readable —
    a top-level popup, so it may extend **past the main window**. This subclass then
    nudges it left on show so a wide popup near the right edge isn't clipped by the
    screen instead of running off it.
    """

    def showEvent(self, event) -> None:  # noqa: N802 - Qt API
        super().showEvent(event)
        try:
            from PyQt6.QtGui import QGuiApplication
            g = self.frameGeometry()
            screen = self.screen() or QGuiApplication.primaryScreen()
            if screen is None:
                return
            avail = screen.availableGeometry()
            x = g.x()
            if x + g.width() > avail.right():
                x = avail.right() - g.width()
            x = max(x, avail.left())
            if x != g.x():
                self.move(x, g.y())
        except Exception:
            pass


class SearchLineEdit(QLineEdit):
    """Toolbar search field: emits ``focusedIn`` / ``doubleClicked``."""

    focusedIn = pyqtSignal()
    doubleClicked = pyqtSignal()

    def focusInEvent(self, event) -> None:       # noqa: N802 - Qt API
        super().focusInEvent(event)
        self.focusedIn.emit()

    def mouseDoubleClickEvent(self, event) -> None:  # noqa: N802 - Qt API
        self.doubleClicked.emit()
        super().mouseDoubleClickEvent(event)


class CommandFinderDialog(QDialog):
    """Full command finder: filter field + table, run on double-click / Enter."""

    def __init__(self, provider, owner=None, initial: str = "") -> None:
        super().__init__(None)
        self._provider = provider          # callable -> fresh list[CommandEntry]
        self._owner = owner
        self._entries: list[CommandEntry] = []
        self._rows: list[CommandEntry] = []
        self.setWindowTitle("Command Finder")
        self.setWindowFlags(Qt.WindowType.Window)
        self.resize(620, 480)

        root = QVBoxLayout(self)
        self._filter = QLineEdit()
        self._filter.setPlaceholderText("Type to search commands…")
        self._filter.setClearButtonEnabled(True)
        root.addWidget(self._filter)

        self._table = QTableWidget(0, 4)
        self._table.setHorizontalHeaderLabels(["Command", "Menu", "Source", "Shortcut"])
        hh = self._table.horizontalHeader()
        # User-adjustable columns (Interactive); auto-sized once on first build so
        # no text is truncated (see _autosize_columns).
        for c in range(4):
            hh.setSectionResizeMode(c, QHeaderView.ResizeMode.Interactive)
        hh.setStretchLastSection(True)
        self._sized = False
        self._table.verticalHeader().setVisible(False)
        self._table.setEditTriggers(QAbstractItemView.EditTrigger.NoEditTriggers)
        self._table.setSelectionBehavior(QAbstractItemView.SelectionBehavior.SelectRows)
        self._table.setSelectionMode(QAbstractItemView.SelectionMode.SingleSelection)
        root.addWidget(self._table, 1)

        self._info = QLabel("")
        self._info.setStyleSheet("color:#888;")
        root.addWidget(self._info)

        self._filter.textChanged.connect(self._repopulate)
        self._filter.returnPressed.connect(self._run_current)
        self._filter.installEventFilter(self)          # Down-arrow → table
        self._table.doubleClicked.connect(lambda idx: self._run_row(idx.row()))
        self._table.activated.connect(lambda idx: self._run_row(idx.row()))

        self.refresh()
        self._filter.setText(initial)
        self._filter.setFocus()

    # -- data ----------------------------------------------------------------
    def refresh(self) -> None:
        self._entries = list(self._provider() or [])
        self._repopulate(self._filter.text())

    def set_query(self, text: str) -> None:
        self._filter.setText(text or "")

    def _repopulate(self, query: str) -> None:
        self._rows = filter_commands(self._entries, query)
        self._table.setRowCount(len(self._rows))
        for r, e in enumerate(self._rows):
            self._cell(r, 0, e.text, e.enabled)
            self._cell(r, 1, e.path, e.enabled)
            self._cell(r, 2, e.source, e.enabled)
            self._cell(r, 3, e.shortcut, e.enabled)
        if self._rows:
            self._table.selectRow(0)
        if not self._sized:                            # fit columns once, then leave
            self._autosize_columns()                   # them user-adjustable
            self._sized = True
        n_off = sum(1 for e in self._rows if not e.enabled)
        note = f"   ({n_off} unavailable)" if n_off else ""
        self._info.setText(f"{len(self._rows)} command(s){note} — double-click or Enter to run")

    def _autosize_columns(self) -> None:
        """Size each column to fit the widest cell/header across **all** commands
        (so nothing is truncated regardless of the current filter), then widen the
        dialog to fit — capped to the screen. Columns stay user-resizable."""
        fm = self._table.fontMetrics()
        headers = ["Command", "Menu", "Source", "Shortcut"]
        values = [
            [e.text for e in self._entries],
            [e.path for e in self._entries],
            [e.source for e in self._entries],
            [e.shortcut for e in self._entries],
        ]
        total = 0
        for c in range(4):
            w = fm.horizontalAdvance(headers[c])
            for v in values[c]:
                w = max(w, fm.horizontalAdvance(str(v)))
            w = min(w + 28, 700)                        # padding, capped per column
            self._table.setColumnWidth(c, w)
            total += w
        try:
            from PyQt6.QtGui import QCursor, QGuiApplication
            screen = QGuiApplication.screenAt(QCursor.pos()) or QGuiApplication.primaryScreen()
            cap = screen.availableGeometry().width() - 80 if screen else 1400
        except Exception:
            cap = 1400
        want = min(total + 60, cap)                     # + scrollbar/margins
        if want > self.width():
            self.resize(want, self.height())

    def _cell(self, r: int, c: int, text: str, enabled: bool) -> None:
        item = QTableWidgetItem(text)
        if not enabled:
            item.setForeground(QColor("#999"))
        self._table.setItem(r, c, item)

    # -- run -----------------------------------------------------------------
    def _run_current(self) -> None:
        r = self._table.currentRow()
        self._run_row(r if r >= 0 else 0)

    def _run_row(self, r: int) -> None:
        if not (0 <= r < len(self._rows)):
            return
        entry = self._rows[r]
        if not entry.enabled:
            self._info.setText(f"'{entry.text}' is currently unavailable (needs a dataset?)")
            return
        self.close()
        entry.action.trigger()

    def eventFilter(self, obj, event):  # noqa: N802 - Qt API
        if obj is self._filter and event.type() == QEvent.Type.KeyPress:
            if event.key() in (Qt.Key.Key_Down, Qt.Key.Key_Up):
                self._table.setFocus()
                return False
        return super().eventFilter(obj, event)
