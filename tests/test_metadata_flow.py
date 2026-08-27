"""The four ways a processing recipe reaches a dataset.

1. beside a data file being opened -- asked about, never silent;
2. handed over on its own -- asked which dataset it acts on;
3. dropped together with its data -- paired, and NOT asked about;
4. held for a dataset not yet loaded -- one at a time, one load long.

These drive ``MainWindow`` through a lightweight stand-in for the pieces that
matter: constructing a real one per test drags in the render chain and every
child window, which is too heavy for a flow test.
"""

from __future__ import annotations

import json

import numpy as np
import pytest

from minflux_viewer.ui.main_window import MainWindow


@pytest.fixture(scope="module")
def _app():
    pytest.importorskip("PyQt6")
    from PyQt6.QtWidgets import QApplication

    return QApplication.instance() or QApplication([])


def _recipe(**rest):
    meta = {"minflux_viewer_metadata": 1, "content": "raw",
            "calibration": {"z_scaling_factor": 0.67}}
    meta.update(rest)
    return meta


def _dataset(name="A"):
    from minflux_viewer.core.dataset import build_localization_dataset

    rng = np.random.default_rng(0)
    return build_localization_dataset(
        name=name, x_nm=rng.random(30) * 100, y_nm=rng.random(30) * 100,
        z_nm=rng.random(30) * 10)


class _Window:
    """The slice of MainWindow these flows touch, with the real methods bound."""

    apply_metadata_to_dataset = MainWindow.apply_metadata_to_dataset
    _remember_pending_metadata = MainWindow._remember_pending_metadata
    _take_pending_metadata = MainWindow._take_pending_metadata
    _sidecar_for_load = MainWindow._sidecar_for_load
    route_paths = MainWindow.route_paths

    def __init__(self, state):
        self._state = state
        self._pending_metadata = ()
        self._batch_sidecars = {}
        self._batch_sidecars_used = set()
        self.routed = []
        self.restored_rois = []
        self.shown_filters = []
        self.asked = []

    # --- the collaborators the real window supplies -----------------------
    def _route_path(self, path):
        self.routed.append(str(path))

    def _restore_saved_rois(self, ds, idx):
        self.restored_rois.append(idx)

    def _show_saved_filters(self, ds, idx):
        self.shown_filters.append(idx)

    def _refresh_overlay_windows(self):
        pass


def _window():
    from minflux_viewer.core.app_state import AppState

    return _Window(AppState())


# ---------------------------------------------------- 1. beside the data

def test_a_sidecar_beside_the_data_is_never_applied_without_asking(
        _app, tmp_path, monkeypatch):
    data = tmp_path / "run.mat"
    data.write_bytes(b"")
    (tmp_path / "run_viewer_metadata.json").write_text(json.dumps(_recipe()),
                                                encoding="utf-8")
    window = _window()

    import minflux_viewer.ui.metadata_apply_dialog as dialogs

    monkeypatch.setattr(dialogs, "ask_load_sidecar",
                        lambda *a, **k: dialogs.SKIP)
    assert window._sidecar_for_load(str(data)) is False

    monkeypatch.setattr(dialogs, "ask_load_sidecar",
                        lambda *a, **k: dialogs.LOAD)
    assert window._sidecar_for_load(str(data)) is True


def test_no_sidecar_means_no_question(_app, tmp_path):
    data = tmp_path / "run.mat"
    data.write_bytes(b"")
    assert _window()._sidecar_for_load(str(data)) is False


# --------------------------------------------------- 3. dropped together

def test_dropping_the_data_with_its_recipe_asks_nothing(
        _app, tmp_path, monkeypatch):
    """Dropping both files IS the answer, whichever order the OS lists them."""
    data = tmp_path / "run.mat"
    data.write_bytes(b"")
    side = tmp_path / "run_viewer_metadata.json"
    side.write_text(json.dumps(_recipe()), encoding="utf-8")

    import minflux_viewer.ui.metadata_apply_dialog as dialogs

    def _refuse(*a, **k):
        raise AssertionError("the user must not be asked for a paired drop")

    monkeypatch.setattr(dialogs, "ask_load_sidecar", _refuse)

    window = _window()
    window._batch_sidecars = {str(side.resolve()).casefold(): str(side)}
    assert window._sidecar_for_load(str(data)) is True
    # ...and the recipe is then marked consumed, so routing it again is skipped.
    assert str(side.resolve()).casefold() in window._batch_sidecars_used


def test_the_batch_routes_data_first_and_skips_a_consumed_recipe(
        _app, tmp_path, monkeypatch):
    data = tmp_path / "run.mat"
    data.write_bytes(b"")
    side = tmp_path / "run_viewer_metadata.json"
    side.write_text(json.dumps(_recipe()), encoding="utf-8")

    window = _window()

    def _route(path):
        window.routed.append(str(path))
        if str(path) == str(data):            # what _load_mat would do
            window._sidecar_for_load(str(path))

    window._route_path = _route
    # Recipe listed FIRST, the order that used to decide the outcome.
    window.route_paths([str(side), str(data)])

    assert window.routed == [str(data)], (
        "data routes first and the paired recipe is not routed again")
    assert window._batch_sidecars == {} and window._batch_sidecars_used == set()


def test_a_recipe_dropped_alone_is_still_routed(_app, tmp_path):
    side = tmp_path / "run_viewer_metadata.json"
    side.write_text(json.dumps(_recipe()), encoding="utf-8")
    window = _window()
    window.route_paths([str(side)])
    assert window.routed == [str(side)]


# ------------------------------------------------------------ 4. pending

def test_only_one_recipe_is_held_and_a_second_replaces_it(_app, tmp_path):
    window = _window()
    first, second = tmp_path / "a_viewer_metadata.json", tmp_path / "b_viewer_metadata.json"
    window._remember_pending_metadata(first, _recipe(data_file="a.mat"))
    window._remember_pending_metadata(second, _recipe(data_file="b.mat"))
    assert window._pending_metadata[0].name == "b_viewer_metadata.json"


def test_the_held_recipe_applies_to_a_matching_dataset(_app, tmp_path):
    window = _window()
    ds = _dataset()
    ds.file.name = "run.mat"
    window._remember_pending_metadata(tmp_path / "run_viewer_metadata.json",
                                      _recipe(data_file="run.mat"))
    taken = window._take_pending_metadata(ds)
    assert taken and taken[2] == "data_file"
    assert window._pending_metadata == ()          # consumed either way


def test_the_held_recipe_is_discarded_by_a_dataset_that_does_not_match(
        _app, tmp_path):
    """Its lifetime is exactly one dataset load -- the expected next action
    after loading a recipe is loading its data."""
    window = _window()
    ds = _dataset()
    ds.file.name = "unrelated.mat"
    window._remember_pending_metadata(tmp_path / "run_viewer_metadata.json",
                                      _recipe(data_file="run.mat"))
    assert window._take_pending_metadata(ds) == ()
    assert window._pending_metadata == ()

    levels = [entry.get("level") for entry in window._state.log_history]
    assert "WARN" in levels, "the discard is reported, not silent"


def test_holding_a_recipe_opens_nothing(_app, tmp_path):
    """3.1: no Filter dialog, no ROI Manager -- there is no dataset yet."""
    window = _window()
    window._remember_pending_metadata(
        tmp_path / "run_viewer_metadata.json",
        _recipe(filters=[{"attribute": "idx", "mode": "per loc",
                          "lo": 0.0, "hi": 1.0}],
                rois=[{"id": "r1", "type": "rectangle"}]))
    assert window.restored_rois == [] and window.shown_filters == []
    assert window._state.datasets == []


# ------------------------------------------- applying reaches the ROI Manager

def test_applying_a_recipe_puts_its_rois_in_front_of_the_user(_app):
    """The verified bug: apply_metadata_recipe only writes ds.metadata, and it
    is _post_load_finalize that fills the ROI Manager -- which does not run for
    an already-loaded dataset. So it reported "1 ROI(s)" over an empty Manager."""
    window = _window()
    window._state.add_dataset(_dataset())
    ok = window.apply_metadata_to_dataset(
        0, _recipe(rois=[{"id": "r1", "type": "rectangle"}]), "run_viewer_metadata.json")
    assert ok
    assert window.restored_rois == [0]
    assert window.shown_filters == [0]


def test_an_empty_recipe_is_reported_rather_than_claimed_as_applied(_app):
    window = _window()
    window._state.add_dataset(_dataset())
    assert window.apply_metadata_to_dataset(0, {"minflux_viewer_metadata": 1},
                                            "empty_viewer_metadata.json") is True
    assert window.restored_rois == []
    messages = [entry.get("message", "") for entry in window._state.log_history]
    assert any("nothing to apply" in text for text in messages)


def test_applying_to_a_row_that_no_longer_exists_is_refused(_app):
    assert _window().apply_metadata_to_dataset(3, _recipe(), "x.json") is False


# ------------------------------------------------------- the dialog itself

def _apply_dialog(meta, datasets):
    from minflux_viewer.ui.metadata_apply_dialog import MetadataApplyDialog

    return MetadataApplyDialog("run_viewer_metadata.json", meta, datasets)


def test_the_dropdown_preselects_the_match_and_still_offers_every_dataset(_app):
    """The match picks the DEFAULT only — a recipe is portable by design, so
    any loaded dataset must remain choosable."""
    a, b, c = _dataset("a"), _dataset("b"), _dataset("c")
    b.metadata["msr_dataset_did"] = "DID-1"
    dialog = _apply_dialog(_recipe(msr_dataset_did="DID-1"), [a, b, c])
    try:
        assert dialog._combo.count() == 3
        assert dialog.dataset_index() == 1
        dialog._combo.setCurrentIndex(2)
        assert dialog.dataset_index() == 2
    finally:
        dialog.close()


def test_with_no_match_the_headline_says_so_and_the_first_is_offered(_app):
    from PyQt6.QtWidgets import QLabel

    dialog = _apply_dialog(_recipe(msr_dataset_did="DID-1"), [_dataset("a")])
    try:
        # Still applicable -- a recipe is portable, so an unmatched one is
        # offered rather than refused; the headline is what says it is a guess.
        assert dialog._apply_btn.isEnabled()
        assert dialog.dataset_index() == 0
        texts = " ".join(child.text() for child in dialog.findChildren(QLabel))
        assert "No matching dataset" in texts
    finally:
        dialog.close()


def test_with_no_dataset_loaded_the_apply_row_is_disabled(_app):
    dialog = _apply_dialog(_recipe(), [])
    try:
        assert not dialog._apply_btn.isEnabled()
        assert not dialog._combo.isEnabled()
        assert dialog.dataset_index() is None
    finally:
        dialog.close()


def test_a_snapshot_recipe_is_warned_about_before_it_flattens_z(_app):
    from PyQt6.QtWidgets import QLabel

    from minflux_viewer.ui.metadata_apply_dialog import SNAPSHOT_WARNING

    dialog = _apply_dialog(_recipe(content="snapshot"), [_dataset("a")])
    try:
        texts = " ".join(child.text() for child in dialog.findChildren(QLabel))
        assert SNAPSHOT_WARNING in texts
    finally:
        dialog.close()
    plain = _apply_dialog(_recipe(), [_dataset("a")])
    try:
        texts = " ".join(child.text() for child in plain.findChildren(QLabel))
        assert SNAPSHOT_WARNING not in texts
    finally:
        plain.close()


def test_cancel_is_distinct_from_keep(_app):
    from minflux_viewer.ui.metadata_apply_dialog import CANCEL

    dialog = _apply_dialog(_recipe(), [])
    try:
        assert dialog.choice() == CANCEL     # closing without choosing
    finally:
        dialog.close()
