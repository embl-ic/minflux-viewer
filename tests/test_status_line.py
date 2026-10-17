"""ui/widgets/status_line.py: a status read-out that never widens its window."""

from __future__ import annotations

from PyQt6.QtWidgets import QLabel, QVBoxLayout, QWidget

from minflux_viewer.ui.widgets.status_line import StatusLine

_LONG = ("3D  |  1 ch  |  LOD 1  |  px=50.0 nm  |  projected over the visible "
         "region  |  crosshair X=-136.3  Y=-202.0  Z=133.4 nm")


def test_a_long_text_asks_for_no_width(qapp):
    line = StatusLine(_LONG)
    assert line.minimumSizeHint().width() == 0
    assert line.text() == _LONG                  # code appending to it sees it all
    assert line.toolTip() == _LONG


def test_the_window_keeps_its_width_and_the_text_is_shortened(qapp):
    window = QWidget()
    layout = QVBoxLayout(window)
    line = StatusLine("")
    layout.addWidget(line)
    window.resize(300, 60)
    window.show()
    qapp.processEvents()
    try:
        line.setText(_LONG)
        qapp.processEvents()
        assert window.width() == 300
        assert window.minimumSizeHint().width() < 300
        shown = line.displayed_text()
        assert shown.endswith("…") and _LONG.startswith(shown[:-1])
    finally:
        window.close()


def test_the_height_does_not_depend_on_the_text(qapp):
    """An empty QLabel asks for a different height than a filled one; a
    status line must not, or the panes above it move when the first status
    arrives (by 2 px, enough to misalign the floating Ortho-View panes)."""
    line = StatusLine("")
    line.setStyleSheet("color: gray; font-size: 11px;")
    line.ensurePolished()
    empty = (line.sizeHint().height(), line.minimumSizeHint().height())
    line.setText(_LONG)
    assert (line.sizeHint().height(), line.minimumSizeHint().height()) == empty
    filled = QLabel(_LONG)
    filled.setStyleSheet("color: gray; font-size: 11px;")
    filled.ensurePolished()
    assert empty[0] == filled.sizeHint().height()


def test_a_text_that_fits_is_shown_whole(qapp):
    window = QWidget()
    layout = QVBoxLayout(window)
    line = StatusLine("No dataset.")
    layout.addWidget(line)
    window.resize(400, 60)
    window.show()
    qapp.processEvents()
    try:
        assert line.displayed_text() == "No dataset."
        line.setText(None)
        assert line.text() == "" and line.toolTip() == ""
    finally:
        window.close()
