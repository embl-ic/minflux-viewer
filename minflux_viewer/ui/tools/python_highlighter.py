"""Python syntax highlighting for the Script Editor.

The usual token classes: keywords, ``True``/``False``/``None``, builtins,
``self``/``cls``, numbers, decorators, the names introduced by ``def`` and
``class``, strings (any prefix, and triple-quoted ones spanning lines) and
comments. Strings and comments are found by one left-to-right scan that
overrides the token colours, so a ``#`` inside a string, or a quote inside a
comment, is coloured as what it is.

Two schemes follow the editor's background, the same lightness rule the main
window uses for dark mode: an IntelliJ-like light scheme and a VS Code
Dark+-like dark scheme.
"""

from __future__ import annotations

import builtins
import keyword
import re

from PyQt6.QtGui import QColor, QFont, QSyntaxHighlighter, QTextCharFormat, QTextDocument

__all__ = ["PythonHighlighter", "LIGHT_SCHEME", "DARK_SCHEME"]

LIGHT_SCHEME: dict[str, str] = {
    "keyword": "#0033b3", "constant": "#0033b3", "builtin": "#000080",
    "self": "#94558d", "number": "#1750eb", "decorator": "#9e880d",
    "definition": "#00627a", "string": "#067d17", "comment": "#8c8c8c",
}
DARK_SCHEME: dict[str, str] = {
    "keyword": "#569cd6", "constant": "#569cd6", "builtin": "#4ec9b0",
    "self": "#9cdcfe", "number": "#b5cea8", "decorator": "#dcdcaa",
    "definition": "#dcdcaa", "string": "#ce9178", "comment": "#6a9955",
}

_CONSTANTS = ("True", "False", "None")
_KEYWORDS = tuple(word for word in keyword.kwlist if word not in _CONSTANTS)
_BUILTINS = tuple(sorted(
    name for name in dir(builtins)
    if not name.startswith("_") and name not in keyword.kwlist))

# Applied first; strings and comments are laid over them afterwards.
_TOKEN_RULES: tuple[tuple[re.Pattern[str], str, int], ...] = (
    (re.compile(r"\b(?:" + "|".join(_KEYWORDS) + r")\b"), "keyword", 0),
    (re.compile(r"\b(?:" + "|".join(_CONSTANTS) + r")\b"), "constant", 0),
    (re.compile(r"(?<![\w.])(?:" + "|".join(_BUILTINS) + r")\b"), "builtin", 0),
    (re.compile(r"\b(?:self|cls)\b"), "self", 0),
    (re.compile(r"\b0[xX][0-9a-fA-F_]+\b|\b0[oO][0-7_]+\b|\b0[bB][01_]+\b"
                r"|(?<![\w.])(?:\d[\d_]*\.?[\d_]*|\.\d[\d_]*)(?:[eE][+-]?\d+)?[jJ]?\b"),
     "number", 0),
    (re.compile(r"^\s*(@[A-Za-z_][\w.]*)"), "decorator", 1),
    (re.compile(r"\b(?:def|class)\s+([A-Za-z_]\w*)"), "definition", 1),
)
# The next comment or string opening; a prefix must not end an identifier.
_OPENING = re.compile(r"#|(?<!\w)(?i:rb|br|fr|rf|[rbuf])?('''|\"\"\"|'|\")")
#: Block states: inside a ''' or a \"\"\" string carried over from the line above.
_STATE_OF = {"'''": 1, '"""': 2}
_DELIM_OF = {1: "'''", 2: '"""'}


def _closing(text: str, delim: str, start: int) -> int:
    """Index of the unescaped *delim* at or after *start*, or -1."""
    i = start
    while i < len(text):
        if text[i] == "\\":
            i += 2
            continue
        if text.startswith(delim, i):
            return i
        i += 1
    return -1


class PythonHighlighter(QSyntaxHighlighter):
    """Colour Python source in a :class:`QTextDocument`."""

    def __init__(self, document: QTextDocument, *, dark: bool = False) -> None:
        super().__init__(document)
        self._formats: dict[str, QTextCharFormat] = {}
        self._dark = bool(dark)
        self._build_formats()

    @property
    def dark(self) -> bool:
        return self._dark

    def set_dark(self, dark: bool) -> None:
        """Switch scheme and recolour the whole document (no-op if unchanged)."""
        if bool(dark) == self._dark:
            return
        self._dark = bool(dark)
        self._build_formats()
        self.rehighlight()

    def color(self, kind: str) -> QColor:
        """The colour currently used for token class *kind*."""
        return QColor(self._formats[kind].foreground().color())

    def _build_formats(self) -> None:
        scheme = DARK_SCHEME if self._dark else LIGHT_SCHEME
        self._formats = {}
        for kind, colour in scheme.items():
            fmt = QTextCharFormat()
            fmt.setForeground(QColor(colour))
            if kind == "comment":
                fmt.setFontItalic(True)
            if kind in ("keyword", "constant"):
                fmt.setFontWeight(QFont.Weight.DemiBold)
            self._formats[kind] = fmt

    def highlightBlock(self, text: str | None) -> None:  # noqa: N802 - Qt API
        text = text or ""
        # Qt positions count UTF-16 code units; Python counts code points.
        units = None
        if not text.isascii() and any(ord(ch) > 0xFFFF for ch in text):
            units = [0]
            for ch in text:
                units.append(units[-1] + (2 if ord(ch) > 0xFFFF else 1))

        def paint(start: int, end: int, kind: str) -> None:
            if end <= start:
                return
            if units is not None:
                start, end = units[start], units[end]
            self.setFormat(start, end - start, self._formats[kind])

        for pattern, kind, group in _TOKEN_RULES:
            for match in pattern.finditer(text):
                paint(match.start(group), match.end(group), kind)

        position = 0
        state = self.previousBlockState()
        self.setCurrentBlockState(0)
        if state in _DELIM_OF:
            delim = _DELIM_OF[state]
            end = _closing(text, delim, 0)
            if end < 0:
                paint(0, len(text), "string")
                self.setCurrentBlockState(state)
                return
            paint(0, end + 3, "string")
            position = end + 3

        while True:
            opening = _OPENING.search(text, position)
            if opening is None:
                return
            if opening.group(0) == "#":
                paint(opening.start(), len(text), "comment")
                return
            delim = opening.group(1)
            end = _closing(text, delim, opening.end())
            if end < 0:
                # An unterminated string runs to the end of the line; a
                # triple-quoted one continues on the next.
                paint(opening.start(), len(text), "string")
                if len(delim) == 3:
                    self.setCurrentBlockState(_STATE_OF[delim])
                return
            paint(opening.start(), end + len(delim), "string")
            position = end + len(delim)
