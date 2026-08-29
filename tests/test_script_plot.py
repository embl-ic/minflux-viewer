from __future__ import annotations

import os
import subprocess
import sys
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")


@pytest.fixture(scope="module")
def qt_app():
    from PyQt6.QtWidgets import QApplication

    return QApplication.instance() or QApplication([])


@pytest.fixture
def facade(qt_app):
    from PyQt6.QtWidgets import QWidget

    from minflux_viewer.scripting import create_facade

    state = SimpleNamespace(datasets=[], active_dataset=None)
    api = create_facade(state)
    owner = QWidget()
    api.bind_main_window(owner)
    yield api, owner
    from minflux_viewer.ui.modeless import close_modeless

    close_modeless(owner)
    api.close_windows()
    owner.close()
    qt_app.processEvents()


def test_line_plot_is_chainable_multi_series_and_modeless(facade, qt_app):
    from PyQt6.QtCore import Qt

    api, owner = facade
    handle = (
        api.plot.line([0, 1, 2], [2, 3, 5], label="first", title="Growth")
        .series(
            [0, 1, 2],
            [1, 4, 9],
            label="second",
            color="#d43f3a",
            style="--",
            width=2,
            symbol="o",
        )
        .labels(x="time (s)", y="signal", title="Growth curves")
        .legend()
        .grid(alpha=0.25)
        .range(x=(0, 2), y=(0, 10))
        .show()
    )
    qt_app.processEvents()

    window = handle._window
    assert window.parentWidget() is None
    assert window in owner._modeless_windows
    assert window.testAttribute(Qt.WidgetAttribute.WA_DeleteOnClose)
    assert len(window.plot_item.listDataItems()) == 2
    assert window.plot_item.getAxis("bottom").labelText == "time (s)"
    assert window.plot_item.getAxis("left").labelText == "signal"
    assert window.legend_item.isVisible()

    assert handle.log(x=True, y=False) is handle
    assert window.plot_item.ctrl.logXCheck.isChecked()
    assert not window.plot_item.ctrl.logYCheck.isChecked()
    assert handle.clear() is handle
    assert window.plot_item.listDataItems() == []


def test_scatter_histogram_and_image(facade):
    api, _owner = facade

    scatter = api.plot.scatter(
        [0, 1, 2],
        [2, 1, 3],
        color_by=[0.1, np.nan, 0.9],
        symbol="s",
        size=8,
    )
    scatter_item = scatter._window.plot_item.listDataItems()[0]
    assert scatter_item.opts["pen"] is None
    assert scatter_item.opts["symbol"] == "s"

    histogram = api.plot.hist([0, 0, 1, 2, 2, 2], bins=3, label="counts")
    hist_item = histogram._window.plot_item.listDataItems()[0]
    assert int(np.sum(hist_item.yData)) == 6

    image = api.plot.image(
        np.arange(12).reshape(3, 4),
        extent=(10.0, 20.0, 50.0, 80.0),
        colormap="parula",
        title="Map",
    )
    window = image._window
    assert window.image_view is not None
    item = window.image_view.getImageItem()
    rect = item.mapRectToParent(item.boundingRect())
    assert (rect.x(), rect.y(), rect.width(), rect.height()) == (10.0, 20.0, 40.0, 60.0)
    assert not window.image_view.ui.histogram.isHidden()


def test_save_png_and_input_errors(facade, tmp_path):
    from minflux_viewer.api._base import ApiError

    api, _owner = facade
    handle = api.plot.line([1, 2, 3], title="PNG")
    target = tmp_path / "plot.png"
    assert handle.save_png(str(target), width=420) == str(target)
    assert target.read_bytes().startswith(b"\x89PNG\r\n\x1a\n")

    with pytest.raises(ApiError, match="same length"):
        api.plot.line([1], [1, 2])
    with pytest.raises(ApiError, match="bins or bin_width"):
        api.plot.hist([1, 2], bins=2, bin_width=1)
    with pytest.raises(ApiError, match="two-dimensional"):
        api.plot.image([1, 2, 3])
    with pytest.raises(ApiError, match="Unknown application colormap"):
        api.plot.image(np.ones((2, 2)), colormap="not-a-map")
    with pytest.raises(ApiError, match="between 0 and 1"):
        handle.grid(alpha=2)
    with pytest.raises(ApiError, match="finite and increasing"):
        handle.range(x=(2, 1))

    handle.close()
    with pytest.raises(ApiError, match="closed"):
        handle.show()


def test_plot_window_child_process_teardown_is_clean():
    code = r"""
import os
os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
from types import SimpleNamespace
import numpy as np
from PyQt6.QtWidgets import QApplication, QWidget
from minflux_viewer.scripting import create_facade
from minflux_viewer.ui.modeless import close_modeless

app = QApplication([])
owner = QWidget()
facade = create_facade(SimpleNamespace(datasets=[], active_dataset=None))
facade.bind_main_window(owner)
facade.plot.line(np.arange(100), np.sin(np.arange(100))).labels(x="x", y="y").show()
facade.plot.image(np.arange(100).reshape(10, 10), colormap="hot").show()
app.processEvents()
close_modeless(owner)
facade.close_windows()
owner.close()
app.processEvents()
"""
    env = os.environ.copy()
    env["QT_QPA_PLATFORM"] = "offscreen"
    completed = subprocess.run(
        [sys.executable, "-c", code],
        cwd=Path(__file__).resolve().parents[1],
        env=env,
        capture_output=True,
        text=True,
        timeout=30,
    )
    assert completed.returncode == 0, completed.stdout + completed.stderr
    assert "has been deleted" not in completed.stderr
    assert "Fatal Python error" not in completed.stderr
