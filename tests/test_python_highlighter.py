"""Python syntax highlighting in the Script Editor."""

from __future__ import annotations

import pytest

pytest.importorskip("PyQt6")

from PyQt6.QtGui import QColor, QTextDocument

from minflux_viewer.ui.tools.python_highlighter import (
    DARK_SCHEME,
    LIGHT_SCHEME,
    PythonHighlighter,
)


@pytest.fixture(scope="module")
def _app(qapp):
    return qapp


def _highlight(code: str, *, dark: bool = False):
    from PyQt6.QtWidgets import QPlainTextDocumentLayout

    document = QTextDocument()
    # As in the editor. Without a document layout Qt emits no contentsChange,
    # so edits would never reach the highlighter.
    document.setDocumentLayout(QPlainTextDocumentLayout(document))
    document.setPlainText(code)
    highlighter = PythonHighlighter(document, dark=dark)
    highlighter.rehighlight()
    return document, highlighter


def _kind_at(document, highlighter, line: int, column: int) -> str | None:
    """The token class coloured at (line, column), by its colour; None = plain."""
    block = document.findBlockByNumber(line)
    for rng in block.layout().formats():
        if rng.start <= column < rng.start + rng.length:
            colour = rng.format.foreground().color().name()
            scheme = DARK_SCHEME if highlighter.dark else LIGHT_SCHEME
            kinds = [k for k, v in scheme.items() if v == colour]
            # keyword and constant share a colour by design; either reads right.
            return "keyword" if set(kinds) == {"keyword", "constant"} else kinds[0]
    return None


def _col(line: str, needle: str) -> int:
    return line.index(needle)


def test_the_standard_token_classes_are_coloured(_app):
    lines = [
        "@decorator.name",
        "def analyse(self, data=None):",
        "    total = len(data) + 0x1F + 3.5e-2  # a comment",
        "    return f'{total} items'",
        "class Model:",
        "    pass",
    ]
    doc, hl = _highlight("\n".join(lines))

    assert _kind_at(doc, hl, 0, 0) == "decorator"
    assert _kind_at(doc, hl, 1, 0) == "keyword"                          # def
    assert _kind_at(doc, hl, 1, _col(lines[1], "analyse")) == "definition"
    assert _kind_at(doc, hl, 1, _col(lines[1], "self")) == "self"
    assert _kind_at(doc, hl, 1, _col(lines[1], "None")) == "keyword"      # constant
    assert _kind_at(doc, hl, 2, _col(lines[2], "len")) == "builtin"
    assert _kind_at(doc, hl, 2, _col(lines[2], "0x1F")) == "number"
    assert _kind_at(doc, hl, 2, _col(lines[2], "3.5e-2")) == "number"
    assert _kind_at(doc, hl, 2, _col(lines[2], "# a")) == "comment"
    assert _kind_at(doc, hl, 2, _col(lines[2], "total")) is None          # plain name
    assert _kind_at(doc, hl, 3, _col(lines[3], "f'")) == "string"
    assert _kind_at(doc, hl, 4, _col(lines[4], "Model")) == "definition"


def test_a_hash_in_a_string_and_a_quote_in_a_comment(_app):
    line = "url = 'http://x#frag'  # it's a comment with print"
    doc, hl = _highlight(line)

    assert _kind_at(doc, hl, 0, _col(line, "#frag")) == "string"
    assert _kind_at(doc, hl, 0, _col(line, "it's")) == "comment"
    assert _kind_at(doc, hl, 0, _col(line, "print")) == "comment", (
        "a builtin name inside a comment is comment-coloured")


def test_keywords_inside_strings_are_string_coloured(_app):
    line = 'msg = "if this is not None: return"'
    doc, hl = _highlight(line)
    for word in ("if", "not", "None", "return"):
        assert _kind_at(doc, hl, 0, _col(line, word)) == "string", word


def test_escaped_quotes_do_not_end_a_string(_app):
    line = r"s = 'it\'s # still' + x"
    doc, hl = _highlight(line)
    assert _kind_at(doc, hl, 0, _col(line, "# still")) == "string"
    assert _kind_at(doc, hl, 0, _col(line, "+ x")) is None


def test_triple_quoted_strings_span_lines(_app):
    lines = ['doc = """first line', "def not_code(): # inside", 'end""" ; x = 1', "y = 2"]
    doc, hl = _highlight("\n".join(lines))

    assert _kind_at(doc, hl, 1, 0) == "string"
    assert _kind_at(doc, hl, 1, _col(lines[1], "# inside")) == "string"
    assert _kind_at(doc, hl, 2, 0) == "string"
    assert _kind_at(doc, hl, 2, _col(lines[2], "1")) == "number"
    assert _kind_at(doc, hl, 3, _col(lines[3], "2")) == "number"
    assert doc.findBlockByNumber(3).userState() in (0, -1)


def test_editing_closes_a_triple_quoted_string_downstream(_app):
    """Removing the opening quotes must recolour the lines below."""
    doc, hl = _highlight('a = """\nimport os\n"""')
    assert _kind_at(doc, hl, 1, 0) == "string"

    from PyQt6.QtGui import QTextCursor

    # Select the block's text explicitly: LineUnderCursor needs a laid-out view.
    cursor = QTextCursor(doc.findBlockByNumber(0))
    cursor.movePosition(QTextCursor.MoveOperation.EndOfBlock, QTextCursor.MoveMode.KeepAnchor)
    cursor.insertText("a = 1")
    assert doc.findBlockByNumber(0).text() == "a = 1"
    assert _kind_at(doc, hl, 1, 0) == "keyword"            # import


def test_prefixed_strings_and_identifiers_ending_in_a_prefix_letter(_app):
    line = "path = rb'raw' ; buf = b'x' ; nf'' "
    doc, hl = _highlight(line)
    assert _kind_at(doc, hl, 0, _col(line, "rb'")) == "string"
    assert _kind_at(doc, hl, 0, _col(line, "b'x")) == "string"
    assert _kind_at(doc, hl, 0, _col(line, "buf")) is None


def test_attribute_access_is_not_a_builtin(_app):
    line = "np.sum(values) + sum(values)"
    doc, hl = _highlight(line)
    assert _kind_at(doc, hl, 0, line.index("sum")) is None
    assert _kind_at(doc, hl, 0, line.rindex("sum")) == "builtin"


def test_positions_after_a_non_bmp_character_stay_aligned(_app):
    """Qt counts UTF-16 units: an emoji is two, a Python index counts one."""
    line = "x = '\U0001F600' # note"
    doc, hl = _highlight(line)
    utf16_hash = len(line[:line.index("#")].encode("utf-16-le")) // 2
    assert _kind_at(doc, hl, 0, utf16_hash) == "comment"
    assert _kind_at(doc, hl, 0, utf16_hash - 1) is None        # the space before '#'
    assert _kind_at(doc, hl, 0, utf16_hash - 2) == "string"    # the closing quote


def _contrast(a: str, b: str) -> float:
    def lum(hex_colour: str) -> float:
        colour = QColor(hex_colour)
        channels = []
        for value in (colour.redF(), colour.greenF(), colour.blueF()):
            channels.append(value / 12.92 if value <= 0.03928 else ((value + 0.055) / 1.055) ** 2.4)
        return 0.2126 * channels[0] + 0.7152 * channels[1] + 0.0722 * channels[2]

    hi, lo = sorted((lum(a), lum(b)), reverse=True)
    return (hi + 0.05) / (lo + 0.05)


@pytest.mark.parametrize(("scheme", "background"), [
    (LIGHT_SCHEME, "#ffffff"), (DARK_SCHEME, "#1e1e1e"), (DARK_SCHEME, "#2d2d2d")])
def test_every_colour_is_readable_on_its_background(scheme, background):
    for kind, colour in scheme.items():
        assert _contrast(colour, background) >= 3.0, (kind, colour, background)


def test_switching_scheme_recolours_the_document(_app):
    doc, hl = _highlight("import os")
    assert hl.color("keyword").name() == LIGHT_SCHEME["keyword"]
    hl.set_dark(True)
    assert hl.color("keyword").name() == DARK_SCHEME["keyword"]
    block = doc.findBlockByNumber(0)
    assert block.layout().formats()[0].format.foreground().color().name() == DARK_SCHEME["keyword"]


def test_the_script_editor_highlights_its_document(_app):
    from minflux_viewer import scripting
    from minflux_viewer.core.app_state import AppState
    from minflux_viewer.ui.tools.script_editor_window import ScriptEditorWindow

    state = AppState()
    state.save_prefs = lambda: None
    state.mfv = scripting.create_facade(state)
    editor = ScriptEditorWindow(state)
    try:
        highlighter = editor._highlighter
        assert highlighter is not None
        assert highlighter.document() is editor.editor.document()
        editor.set_script("import mfv\n")
        highlighter.rehighlight()
        formats = editor.editor.document().findBlockByNumber(0).layout().formats()
        assert any(r.start == 0 and r.length == len("import") for r in formats)
    finally:
        editor.force_close()
