from __future__ import annotations

import sys

import numpy as np
import pytest


@pytest.fixture
def _app():
    pytest.importorskip("PyQt6")
    from PyQt6.QtWidgets import QApplication

    return QApplication.instance() or QApplication(sys.argv)


def test_dialog_defaults_to_two_even_minute_windows(_app):
    from minflux_viewer.ui.time_channel_dialog import TimeChannelDialog

    dialog = TimeChannelDialog(
        np.linspace(0.0, 600.0, 601),
        source_name="exchange-paint.mat",
        color_cycle=["Red", "Green"],
    )
    try:
        windows = dialog.windows()
        assert len(windows) == 2
        assert windows[0].name == "exchange-paint.mat [time 1]"
        assert windows[0].start_s == pytest.approx(0.0)
        assert windows[0].end_s == pytest.approx(300.0)
        assert windows[1].start_s == pytest.approx(300.0)
        assert windows[1].end_s == pytest.approx(600.0)
        assert dialog._rows[0]["count"].text() == "300"
        assert dialog._rows[1]["count"].text() == "301"
    finally:
        dialog.close()
        _app.processEvents()


def test_drag_region_and_add_window_update_table(_app):
    from minflux_viewer.ui.time_channel_dialog import TimeChannelDialog

    dialog = TimeChannelDialog(
        np.arange(0.0, 601.0),
        source_name="source.mat",
    )
    try:
        first = dialog._rows[0]
        first["region"].setRegion((1.0, 4.0))
        _app.processEvents()
        assert first["start"].value() == pytest.approx(1.0)
        assert first["end"].value() == pytest.approx(4.0)

        dialog._table.selectRow(1)
        dialog._add_by_splitting()
        assert len(dialog._rows) == 3
        assert dialog._window_count.value() == 3
        assert dialog.windows()[-1].end_s == pytest.approx(600.0)
    finally:
        dialog.close()
        _app.processEvents()
