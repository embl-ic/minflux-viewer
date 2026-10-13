"""Export figure: suggested names, the repeat counter, overwrite confirmation and
*Open saved folder when completed*."""

from __future__ import annotations

import os
import sys
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

pytest.importorskip("PyQt6")
pytest.importorskip("pyqtgraph")

# ⚠ Everything Qt-side is imported inside the tests, not at collection: importing
# figure_export (and the Qt widgets) while the WHOLE suite is being collected
# reproducibly tipped the known native abort (the project notes, "Lifecycle regressions")
# at test_attribute_separation_dialog — 2/2 with module-level imports, 0/2
# without this file. The tests are unaffected; only collection is.


class _Lazy:
    """Attribute access imports the module on first use."""

    def __init__(self, name):
        self._name = name

    def __getattr__(self, attr):
        import importlib
        return getattr(importlib.import_module(self._name), attr)


fe = _Lazy("minflux_viewer.ui.figure_export")
_qtw = _Lazy("PyQt6.QtWidgets")


@pytest.fixture
def _qt_app():
    if not os.environ.get("DISPLAY") and os.name != "nt" and sys.platform != "darwin":
        pytest.skip("No display available for Qt tests")
    return _qtw.QApplication.instance() or _qtw.QApplication(sys.argv)


# ---------------------------------------------------------------------------
# Names
# ---------------------------------------------------------------------------

def test_dataset_label_drops_the_source_extension_and_joins_the_parts():
    assert (fe.dataset_label("3_Bonly_MINFLUX_3D.msr | 260624-173440_minflux")
            == "3_Bonly_MINFLUX_3D_260624-173440_minflux")
    assert fe.dataset_label("run.zarr.zip | ch 1") == "run_ch_1"
    assert fe.dataset_label("a:b/c") == "a_b_c"            # characters no file system takes
    assert fe.dataset_label("") == ""


@pytest.mark.parametrize("kind, view, want", [
    ("loc_scatter", "XY", "run_ds_loc_scatter_XY"),
    ("render_view", "Ortho-View", "run_ds_render_view_Ortho-View"),
    ("attribute_histogram", "efo", "run_ds_attribute_histogram_efo"),
    ("attribute_plot", "3D", "run_ds_attribute_plot_3D"),
    ("render_view", "", "run_ds_render_view"),
])
def test_suggested_stem_names_the_dataset_the_plot_and_the_view(kind, view, want):
    fig = SimpleNamespace(dataset="run.msr | ds", name="Window title", kind=kind, view=view)
    assert fe.suggested_stem(fig) == want


def test_suggested_stem_falls_back_to_the_window_title():
    fig = SimpleNamespace(dataset="", name="Scatter — sim", kind="loc_scatter", view="XY")
    assert fe.suggested_stem(fig) == "Scatter_—_sim_loc_scatter_XY"


def test_repeat_counter_counts_like_save_as(tmp_path):
    assert fe.next_free_path(tmp_path, "fig", ".pdf") == tmp_path / "fig.pdf"
    (tmp_path / "fig.pdf").touch()
    assert fe.next_free_path(tmp_path, "fig", ".pdf") == tmp_path / "fig(_01).pdf"
    (tmp_path / "fig(_01).pdf").touch()
    assert fe.next_free_path(tmp_path, "fig", ".pdf") == tmp_path / "fig(_02).pdf"
    # continuing from a counted name counts on, rather than nesting counters
    assert fe.next_free_path(tmp_path, "fig(_01)", ".pdf") == tmp_path / "fig(_02).pdf"
    assert fe.next_free_path(tmp_path, "fig", ".png") == tmp_path / "fig.png"   # per extension
    assert fe.base_stem("fig(_12)") == "fig" and fe.base_stem("fig(1)") == "fig(1)"


def test_extension_is_replaced_only_when_it_is_one():
    assert fe.with_extension("d/x.png", ".pdf") == Path("d/x.pdf")
    assert fe.with_extension("d/run.v2", ".pdf") == Path("d/run.v2.pdf")
    assert fe.with_extension("d/x", ".tif") == Path("d/x.tif")


# ---------------------------------------------------------------------------
# The dialog
# ---------------------------------------------------------------------------

def _figure(folder, view="XY"):
    w = _qtw.QWidget()
    w.resize(40, 30)
    return fe.figure_of(w, name="t", dataset="run.msr | ds", kind="loc_scatter",
                        view=view, folder_hint=str(folder)), w


def _dialog(figure, memory=None):
    return fe.FigureExportDialog(None, figure, SimpleNamespace(prefs={}), memory=memory)


def _export(dlg):
    """Press Save and let run() finish without showing the modal dialog."""
    dlg._on_save()
    dlg.dialog.exec = lambda: int(_qtw.QDialog.DialogCode.Accepted)
    return dlg.run()


def test_suggestion_and_repeated_export_from_one_view(_qt_app, tmp_path):
    fig, _w = _figure(tmp_path)
    memory: dict = {}
    first = _dialog(fig, memory)
    assert Path(first._path.text()) == tmp_path / "run_ds_loc_scatter_XY.png"
    assert _export(first)["path"] == str(tmp_path / "run_ds_loc_scatter_XY.png")
    second = _dialog(fig, memory)
    assert Path(second._path.text()) == tmp_path / "run_ds_loc_scatter_XY(_01).png"
    _export(second)
    assert Path(_dialog(fig, memory)._path.text()) == tmp_path / "run_ds_loc_scatter_XY(_02).png"


def test_a_typed_name_is_continued_and_a_new_view_mode_renamed(_qt_app, tmp_path):
    fig, _w = _figure(tmp_path)
    memory: dict = {}
    dlg = _dialog(fig, memory)
    dlg._path.setText(str(tmp_path / "panel_a.png"))
    _export(dlg)
    assert Path(_dialog(fig, memory)._path.text()) == tmp_path / "panel_a(_01).png"
    # an untouched suggestion follows the view mode instead
    fig2, _w2 = _figure(tmp_path)
    memory2: dict = {}
    _export(_dialog(fig2, memory2))
    fig2.view = "XZ"
    assert Path(_dialog(fig2, memory2)._path.text()) == tmp_path / "run_ds_loc_scatter_XZ.png"


def test_changing_the_format_keeps_the_suggestion_free(_qt_app, tmp_path):
    (tmp_path / "run_ds_loc_scatter_XY.pdf").touch()
    fig, _w = _figure(tmp_path)
    dlg = _dialog(fig)
    assert Path(dlg._path.text()).name == "run_ds_loc_scatter_XY.png"
    dlg._vector.setChecked(True)
    dlg._format.setCurrentIndex(dlg._format.findData("pdf"))
    assert Path(dlg._path.text()).name == "run_ds_loc_scatter_XY(_01).pdf"


def test_an_existing_file_asks_before_it_is_replaced(_qt_app, tmp_path, monkeypatch):
    fig, _w = _figure(tmp_path)
    target = tmp_path / "taken.png"
    target.write_bytes(b"old")
    asked = []

    def answer(reply):
        def _question(*args, **kwargs):
            asked.append(args[2])
            return reply
        return _question

    dlg = _dialog(fig)
    dlg._path.setText(str(target))
    monkeypatch.setattr(_qtw.QMessageBox, "question", staticmethod(answer(_qtw.QMessageBox.StandardButton.No)))
    dlg._on_save()
    assert dlg._final_path is None and "taken.png already exists" in asked[0]
    assert target.read_bytes() == b"old"                    # the dialog stays open

    monkeypatch.setattr(_qtw.QMessageBox, "question", staticmethod(answer(_qtw.QMessageBox.StandardButton.Yes)))
    dlg.dialog.exec = lambda: int(_qtw.QDialog.DialogCode.Accepted)
    dlg._on_save()
    assert dlg.run() is not None and target.read_bytes() != b"old"


def test_a_name_confirmed_in_browse_is_not_asked_again(_qt_app, tmp_path, monkeypatch):
    fig, _w = _figure(tmp_path)
    target = tmp_path / "taken.png"
    target.touch()
    dlg = _dialog(fig)
    monkeypatch.setattr(dlg._file_dialog, "getSaveFileName",
                        staticmethod(lambda *a, **k: (str(target), "")))
    monkeypatch.setattr(_qtw.QMessageBox, "question",
                        staticmethod(lambda *a, **k: pytest.fail("asked twice")))
    dlg._browse()
    dlg._on_save()
    assert dlg._final_path == target


def test_open_saved_folder_reveals_the_file_and_is_remembered(_qt_app, tmp_path):
    fig, _w = _figure(tmp_path)
    state = SimpleNamespace(prefs={})
    revealed = []
    dlg = fe.FigureExportDialog(None, fig, state)
    dlg._reveal = revealed.append
    assert not dlg._open_folder.isChecked()
    dlg._open_folder.setChecked(True)
    result = _export(dlg)
    assert revealed == [Path(result["path"])]
    assert state.prefs["figure_export"]["open_folder"] is True
    assert fe.FigureExportDialog(None, fig, state)._open_folder.isChecked()


# ---------------------------------------------------------------------------
# The views say what they are
# ---------------------------------------------------------------------------

def _state():
    from minflux_viewer.core.app_state import AppState
    from minflux_viewer.core.dataset import build_localization_dataset

    state = AppState()
    state.add_dataset(build_localization_dataset(
        name="figure-test", x_nm=np.array([0.0, 10.0, 20.0]),
        y_nm=np.array([0.0, 20.0, 10.0]), z_nm=np.array([0.0, 5.0, 10.0])))
    return state


def test_each_view_names_its_plot_type_and_view_mode(_qt_app):
    from minflux_viewer.ui.attribute_window import AttributeWindow
    from minflux_viewer.ui.histogram_window import HistogramWindow
    from minflux_viewer.ui.render_window import RenderWindow
    from minflux_viewer.ui.scatter_window import ScatterWindow

    state = _state()
    windows = [RenderWindow(state, dataset_idx=0), ScatterWindow(state, dataset_idx=0),
               HistogramWindow(state, dataset_idx=0), AttributeWindow(state, dataset_idx=0)]
    try:
        stems = [fe.suggested_stem(w.figure_for_export()) for w in windows]
        hist_attr = windows[2]._attr_combo.currentText()
        assert stems == ["figure-test_render_view_XY", "figure-test_loc_scatter_XY",
                         f"figure-test_attribute_histogram_{hist_attr}",
                         "figure-test_attribute_plot_XY"]
    finally:
        for w in windows:
            w.close()
