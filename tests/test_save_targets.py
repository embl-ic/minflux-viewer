"""Where saving goes: the Save As menu, File > Save (always Zarr v2), and the
Save / export dialog's format list.

The rule these pin down is that the **format registry decides what the UI
offers and in what order** -- ``formats.offered_save_formats()`` -- so the Save
As menu, the Save/Export dropdown and the Preferences checkboxes cannot drift
apart, and a writer can be kept while its menu entry is withdrawn.
"""

from __future__ import annotations

import pathlib
import sys
import types
from pathlib import Path

import pytest


@pytest.fixture
def _app():
    pytest.importorskip("PyQt6")
    from PyQt6.QtWidgets import QApplication
    return QApplication.instance() or QApplication(sys.argv)


def _real_window(_app):
    from minflux_viewer.core.app_state import AppState
    from minflux_viewer.ui.main_window import MainWindow
    state = AppState()
    state.prefs.setdefault("data", {}).update({"show_data_info": False,
                                               "show_render": False})
    return MainWindow(state)


def _fake_dataset(tmp_path, *, metadata=None):
    """Enough of a dataset for the path-proposing helpers."""
    return types.SimpleNamespace(
        name="run1",
        metadata=dict(metadata or {}),
        file=types.SimpleNamespace(folder=str(tmp_path), name="run1.mat"),
    )


# --------------------------------------------------------------- Save As menu
def test_save_as_lists_the_offered_formats_in_registry_order(_app):
    win = _real_window(_app)
    try:
        texts = [a.text() for a in win.menuSaveAs.actions() if not a.isSeparator()]
        assert texts == [
            "Zarr (.zarr v2) format",
            "MINFLUX data formats (.mat; .npy; .json)",
            "Custom table (.csv)...",
            "MINFLUX .msr file (experimental)",
        ]
    finally:
        win.close()
        _app.processEvents()


def test_withdrawn_entries_keep_their_actions_but_leave_every_menu(_app):
    """The sealed Zarr package, Picasso HDF5 and the OME-TIFF forwarder are
    still constructed and still callable -- only their menu entries are gone."""
    win = _real_window(_app)
    try:
        for name in ("actionSaveAsZarrZip", "actionSaveAsOmeTiff", "actionSaveAsHdf5"):
            assert hasattr(win, name), name
        placed = set()
        for menu_action in win.menuBar().actions():
            menu = menu_action.menu()
            if menu is None:
                continue
            for act in menu.actions():
                placed.add(act)
                if act.menu() is not None:
                    placed.update(act.menu().actions())
        for name in ("actionSaveAsZarrZip", "actionSaveAsOmeTiff", "actionSaveAsHdf5"):
            assert getattr(win, name) not in placed, name
    finally:
        win.close()
        _app.processEvents()


# --------------------------------------------------------------- File > Save
def test_file_save_is_always_zarr_v2(_app, tmp_path, monkeypatch):
    from minflux_viewer.ui import zarr_save_dialog

    win = _real_window(_app)
    try:
        ds = _fake_dataset(tmp_path)
        win._state._datasets.append(ds)
        win._state._active_idx = 0

        asked = {}
        monkeypatch.setattr(
            zarr_save_dialog, "ask_zarr_save_path",
            lambda parent, suggested, **kw: asked.setdefault("suggested", Path(suggested))
            and None or Path(tmp_path / "out.zarr"))
        calls = []
        monkeypatch.setattr(type(win), "_save_as_format",
                            lambda self, fmt, title, *, path=None: calls.append((fmt, path)))

        win._save_data()
        assert calls == [("zarr", str(tmp_path / "out.zarr"))]
        # ... and it proposed a path beside the dataset's own file.
        assert asked["suggested"] == tmp_path / "run1.zarr"
    finally:
        win.close()
        _app.processEvents()


def test_file_save_proposes_the_store_the_dataset_came_from(_app, tmp_path):
    """Ctrl+S on work opened from a .zarr saves back over it, not to a copy."""
    win = _real_window(_app)
    try:
        store = tmp_path / "acquisition.zarr"
        ds = _fake_dataset(tmp_path,
                           metadata={"minflux_viewer_zarr_path": str(store)})
        assert win._zarr_save_suggestion(ds) == store
        # A project store counts too; anything else falls back beside the file.
        ds2 = _fake_dataset(tmp_path,
                            metadata={"minflux_viewer_project_path": str(store)})
        assert win._zarr_save_suggestion(ds2) == store
        assert win._zarr_save_suggestion(_fake_dataset(tmp_path)) == tmp_path / "run1.zarr"
    finally:
        win.close()
        _app.processEvents()


def test_cancelling_the_save_dialog_writes_nothing(_app, tmp_path, monkeypatch):
    from minflux_viewer.ui import zarr_save_dialog

    win = _real_window(_app)
    try:
        win._state._datasets.append(_fake_dataset(tmp_path))
        win._state._active_idx = 0
        monkeypatch.setattr(zarr_save_dialog, "ask_zarr_save_path",
                            lambda *a, **k: None)
        calls = []
        monkeypatch.setattr(type(win), "_save_as_format",
                            lambda self, *a, **k: calls.append(a))
        win._save_data()
        assert calls == []
    finally:
        win.close()
        _app.processEvents()


def test_quick_save_dialog_normalizes_and_refuses_a_bad_path(_app, tmp_path):
    from minflux_viewer.ui.zarr_save_dialog import ZarrQuickSaveDialog

    dlg = ZarrQuickSaveDialog(tmp_path / "run1.zarr", dataset_name="run1")
    assert dlg.path() == tmp_path / "run1.zarr"
    # The extension is the format's, whatever was typed -- and normalize_path
    # must not turn an already-correct name into run1.zarr.zarr.
    dlg._path.setText(str(tmp_path / "other"))
    assert dlg.path() == tmp_path / "other.zarr"
    dlg._path.setText("   ")
    assert dlg.path() is None
    dlg.close()


# ------------------------------------------------------- Save / export dialog
def test_export_dropdown_is_the_registry_order_and_csv_is_the_custom_table(_app):
    from minflux_viewer.ui.save_dialog import SaveProcessedDataDialog

    dlg = SaveProcessedDataDialog("ds1", prefs={"data": {}, "file": {}})
    keys = [dlg._format.itemData(i) for i in range(dlg._format.count())]
    labels = [dlg._format.itemText(i) for i in range(dlg._format.count())]
    assert keys == ["zarr", "npy", "mat", "json", "csv", "msr"]
    assert labels == [
        "MINFLUX Viewer Zarr v2 (.zarr)",
        "NumPy (.npy)",
        "MATLAB (.mat)",
        "JSON (.json)",
        "Custom table (.csv)",
        "MINFLUX (.msr)",
    ]
    assert "zarr_zip" not in keys
    dlg.close()


def test_csv_means_the_custom_picker_and_canonical_is_under_more_options(_app):
    """One .csv entry, two writers behind it.

    The custom table is the current view with columns the user picks; the
    canonical table is the whole mfx node (every iteration and validity state)
    and is the only CSV that reloads without the column-mapping dialog, so it
    stays reachable rather than being dropped.
    """
    from minflux_viewer.ui.save_dialog import SaveProcessedDataDialog

    dlg = SaveProcessedDataDialog("ds1", prefs={"data": {}, "file": {}})
    dlg._format.setCurrentIndex(dlg._format.findData("csv"))
    assert dlg.is_custom_csv()
    assert dlg.options()["csv_mode"] == "custom"
    # A custom table names its own columns, so the snapshot options that would
    # otherwise describe them are inapplicable.
    assert not dlg._inc_attrs.isEnabled() and not dlg._filter_mode.isEnabled()

    dlg._csv_canonical.setChecked(True)
    assert not dlg.is_custom_csv()
    assert dlg.options()["csv_mode"] == "canonical"
    assert dlg._inc_attrs.isEnabled() and dlg._filter_mode.isEnabled()

    # Every other format is unaffected by the switch.
    dlg._format.setCurrentIndex(dlg._format.findData("npy"))
    assert dlg.options()["csv_mode"] is None
    assert not dlg._csv_canonical.isVisible()
    dlg.close()


def test_zarr_still_owns_the_options_it_stores_internally(_app):
    """The csv sync must not undo the Zarr overrides it runs after."""
    from minflux_viewer.ui.save_dialog import SaveProcessedDataDialog

    dlg = SaveProcessedDataDialog("ds1", prefs={"data": {}, "file": {}})
    dlg._format.setCurrentIndex(dlg._format.findData("zarr"))
    assert dlg._inc_recipe.isChecked() and not dlg._inc_recipe.isEnabled()
    assert dlg._inc_derived.isChecked() and not dlg._inc_derived.isEnabled()
    dlg.close()


def test_leaving_zarr_restores_the_options_it_forced(_app):
    """Zarr stores derived arrays and processing metadata internally, so it
    forces both on. Switching to another format must give the user's own values
    back -- the forced ticks used to stay, quietly overriding the preference."""
    from minflux_viewer.ui.save_dialog import SaveProcessedDataDialog

    prefs = {"data": {"export_include_derived": False,
                      "export_include_recipe": True}, "file": {}}
    dlg = SaveProcessedDataDialog("ds1", prefs=prefs)
    assert dlg._format.currentData() == "zarr"
    assert dlg._inc_derived.isChecked() and not dlg._inc_derived.isEnabled()

    dlg._format.setCurrentIndex(dlg._format.findData("mat"))
    assert not dlg._inc_derived.isChecked() and dlg._inc_derived.isEnabled()
    assert dlg._inc_recipe.isChecked()

    # A value the user sets themselves survives a round trip through Zarr.
    dlg._inc_derived.setChecked(True)
    dlg._format.setCurrentIndex(dlg._format.findData("zarr"))
    dlg._format.setCurrentIndex(dlg._format.findData("npy"))
    assert dlg._inc_derived.isChecked()
    dlg.close()


def test_the_dialog_and_preferences_say_the_same_words(_app):
    """Both show these three options; they must not read differently."""
    from minflux_viewer.core.app_state import AppState
    from minflux_viewer.ui import preferences_dialog as P
    from minflux_viewer.ui import save_dialog as S

    dlg = P.PreferencesDialog(AppState())
    try:
        assert dlg._export_inc_attrs.text() == S.LBL_ATTRS
        assert dlg._export_inc_derived.text() == S.LBL_DERIVED
        assert dlg._export_inc_recipe.text() == S.LBL_RECIPE
        assert [dlg._export_filter_mode.itemText(i) for i in range(2)] == [
            S.LBL_FILTER_FLAG, S.LBL_FILTER_APPLY]
        # The sidecar name is derived, never typed: it was renamed once already.
        from minflux_viewer.core.save import METADATA_SUFFIX
        assert METADATA_SUFFIX in S.LBL_RECIPE
    finally:
        dlg.close()


def test_the_dataset_manager_is_the_only_way_to_the_save_export_dialog():
    """File > Save used to open it too; it now saves Zarr directly.

    Pinned because the dialog carries options (content, attribute/derived
    inclusion, sidecar, filter handling) that only make sense when a *format*
    is being chosen, and there is exactly one place left that chooses one.
    """
    import minflux_viewer

    root = pathlib.Path(minflux_viewer.__file__).parent   # not the CWD
    callers = set()
    for path in root.rglob("*.py"):
        for line in path.read_text(encoding="utf-8").splitlines():
            if "save_dataset" in line and "def save_dataset" not in line:
                callers.add(path.relative_to(root).as_posix())
    assert callers == {"ui/dataset_manager.py"}, callers


def test_preferences_offers_only_formats_that_have_a_menu_entry(_app):
    """Ticking a withdrawn format could not put it back anywhere."""
    from minflux_viewer.ui.preferences_dialog import _EXPORT_FORMATS

    keys = [key for _label, key in _EXPORT_FORMATS]
    assert keys == ["zarr", "npy", "mat", "json", "csv"]     # .msr has its own row
    assert "zarr_zip" not in keys
