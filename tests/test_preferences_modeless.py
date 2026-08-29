"""The Preferences dialog is modeless, so it must not clobber concurrent writes.

Modality used to be load-bearing: ``_accept`` assigned its opening snapshot
straight onto ``AppState.prefs``, so anything written while it was open was
silently reverted. Being modal meant nothing *could* write in between. These
tests pin the merge that replaced that guarantee.
"""

from __future__ import annotations

import copy

import pytest

pytest.importorskip("PyQt6")

from PyQt6.QtWidgets import QApplication

from minflux_viewer.core.app_state import AppState, merge_pref_changes


@pytest.fixture(scope="module")
def _app():
    return QApplication.instance() or QApplication([])


@pytest.fixture
def _state():
    state = AppState()
    state.save_prefs = lambda: None          # never touch the real QSettings
    return state


def _dialog(state):
    from minflux_viewer.ui.preferences_dialog import PreferencesDialog

    return PreferencesDialog(state)


def test_ok_keeps_every_write_made_while_the_dialog_was_open(_app, _state):
    """The five subtrees other components write must all survive OK."""
    dialog = _dialog(_state)
    try:
        _state.prefs.setdefault("file", {})["recent_files"] = ["D:/new.mat"]
        _state.prefs.setdefault("plot", {})["custom_colormaps"] = {"mine": ["#000"]}
        _state.prefs.setdefault("colors", {}).setdefault("solid", {})["NEW"] = [1, 2, 3, 4]
        _state.prefs.setdefault("aggregation", {})["photon_threshold"] = 9999
        _state.prefs.setdefault("conv_segmentation", {})["pixel_nm"] = 42

        dialog._accept()

        assert _state.prefs["file"]["recent_files"] == ["D:/new.mat"]
        assert _state.prefs["plot"]["custom_colormaps"] == {"mine": ["#000"]}
        assert _state.prefs["colors"]["solid"]["NEW"] == [1, 2, 3, 4]
        assert _state.prefs["aggregation"]["photon_threshold"] == 9999
        assert _state.prefs["conv_segmentation"]["pixel_nm"] == 42
    finally:
        dialog.deleteLater()


def test_ok_still_applies_what_the_user_actually_edited(_app, _state):
    """Merging must not turn OK into a no-op."""
    dialog = _dialog(_state)
    try:
        dialog._files_in_history.setValue(17)
        dialog._accept()
        assert _state.prefs["file"]["num_file_history"] == 17
    finally:
        dialog.deleteLater()


def test_ok_does_not_replace_the_prefs_object(_app, _state):
    """Other components hold sub-dict references; rebinding prefs orphans them."""
    dialog = _dialog(_state)
    try:
        before = id(_state.prefs)
        plot_ref = _state.prefs["plot"]
        dialog._accept()
        assert id(_state.prefs) == before
        assert _state.prefs["plot"] is plot_ref
    finally:
        dialog.deleteLater()


def test_an_external_colour_edit_is_adopted_not_overwritten(_app, _state):
    """The COLOR dialog is modeless too, and edits the same registry."""
    dialog = _dialog(_state)
    try:
        colors = copy.deepcopy(_state.prefs.get("colors", {}))
        colors.setdefault("viewer", {})["roi_edge"] = [1, 2, 3, 4]
        _state.prefs["colors"] = colors
        _state.colors_changed.emit({"paths": ["viewer.roi_edge"]})

        # The swatch follows the external edit...
        assert list(dialog._roi_color.rgba()) == [1, 2, 3, 4]
        dialog._accept()
        # ...and OK does not write the stale value back over it.
        assert _state.prefs["colors"]["viewer"]["roi_edge"] == [1, 2, 3, 4]
    finally:
        dialog.deleteLater()


def test_cancel_writes_nothing(_app, _state):
    dialog = _dialog(_state)
    try:
        before = copy.deepcopy(_state.prefs)
        dialog._files_in_history.setValue(17)
        dialog.reject()
        assert _state.prefs == before
    finally:
        dialog.deleteLater()


def test_merge_reports_deletions_and_reorders():
    """Reset-to-defaults removes keys; solid colour order is itself a setting."""
    baseline = {"a": {"x": 1}, "drop": {"y": 2},
                "colors": {"solid": {"A": [1, 1, 1, 1], "B": [2, 2, 2, 2]}}}
    live = copy.deepcopy(baseline)
    edited = copy.deepcopy(baseline)
    del edited["drop"]
    edited["colors"]["solid"] = {"B": [2, 2, 2, 2], "A": [1, 1, 1, 1]}

    merge_pref_changes(live, baseline, edited)

    assert "drop" not in live
    # dicts compare equal regardless of order, so this needs the order-aware path
    assert list(live["colors"]["solid"]) == ["B", "A"]


def test_merge_does_not_materialise_empty_containers():
    baseline = {"plot": {"deep": {"k": 1}}}
    live = {}
    edited = copy.deepcopy(baseline)
    merge_pref_changes(live, baseline, edited)
    assert live == {}, "an unchanged subtree must not appear as an empty dict"
