"""Plugins › HlyB/D pooled pair analysis — the accumulator window."""

import os
import sys

import numpy as np
import pytest

pytest.importorskip("PyQt6")
os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

from PyQt6.QtWidgets import QApplication  # noqa: E402

from minflux_viewer.core.cell_collection import CellCollection  # noqa: E402
from minflux_viewer.core.dataset import build_localization_dataset  # noqa: E402
from minflux_viewer.core.roi import RoiRecord, RoiStore  # noqa: E402
from minflux_viewer.ui.hlyb_collection_dialog import (  # noqa: E402
    HlyBCollectionWindow,
)


@pytest.fixture(scope="module")
def app():
    return QApplication.instance() or QApplication(sys.argv[:1])


def _rod(centre_nm, seed, *, n_sites=160, locs_per_trace=14):
    rng = np.random.default_rng(seed)
    axial = rng.uniform(-1000.0, 1000.0, n_sites)
    phi = rng.uniform(0.0, 2.0 * np.pi, n_sites)
    sites = np.column_stack([axial, 300.0 * np.cos(phi), 300.0 * np.sin(phi)])
    sites = sites + np.asarray(centre_nm, dtype=float)
    loc, tid = [], []
    for index, site in enumerate(sites):
        n = int(rng.integers(locs_per_trace, locs_per_trace + 5))
        loc.append(site + rng.normal(0.0, 3.0, size=(n, 3)))
        tid.append(np.full(n, index, dtype=float))
    return np.concatenate(loc), np.concatenate(tid)


def _dataset(name, centres, seed0=0):
    parts = [_rod(c, seed0 + i) for i, c in enumerate(centres)]
    loc = np.vstack([p[0] for p in parts])
    tid = np.concatenate([p[1] + 100_000 * i for i, p in enumerate(parts)])
    return build_localization_dataset(
        name=name, x_nm=loc[:, 0], y_nm=loc[:, 1], z_nm=loc[:, 2], tid=tid,
        tim=np.arange(loc.shape[0], dtype=float) * 1e-3,
        source_version="simulation")


def _polygon(name, cx, cy, half, dataset_idx):
    record = RoiRecord.create(
        "polygon",
        {"points": [[cx - half, cy - half], [cx + half, cy - half],
                    [cx + half, cy + half], [cx - half, cy + half]],
         "closed": True},
        name=name, coordinate_space="plot")
    record.context = {"dataset_idx": dataset_idx}
    return record


class _State:
    def __init__(self):
        self.datasets = []
        self.active_idx = None
        self.rois = RoiStore()
        self.prefs = {}
        self.messages = []

    def log(self, message, *args, **kwargs):
        self.messages.append(message)


def _state_with_two_datasets():
    state = _State()
    state.datasets = [_dataset("ds A", [(0.0, 0.0, 0.0), (6000.0, 0.0, 0.0)]),
                      _dataset("ds B", [(0.0, 0.0, 0.0)], seed0=50)]
    state.active_idx = 0
    return state


def _collect_dataset(window, state, index, rois):
    state.active_idx = index
    for record in rois:
        state.rois.add(record)
    window._collect()


def test_window_starts_empty_and_disables_the_run(app):
    state = _state_with_two_datasets()
    win = HlyBCollectionWindow(state, owner=None)
    try:
        assert len(win._collection) == 0
        assert win._table.rowCount() == 0
        assert not win._run_btn.isEnabled()
        assert not win._save_btn.isEnabled()
    finally:
        win.close()


def test_collecting_appends_one_row_per_region_roi(app):
    state = _state_with_two_datasets()
    win = HlyBCollectionWindow(state, owner=None)
    try:
        _collect_dataset(win, state, 0, [
            _polygon("cell 1", 0.0, 0.0, 1600.0, 0),
            _polygon("cell 2", 6000.0, 0.0, 1600.0, 0)])
        assert len(win._collection) == 2
        assert win._table.rowCount() == 2
        assert win._table.item(0, 0).text() == "ds A"
        assert win._table.item(0, 1).text() == "cell 1"
        assert win._run_btn.isEnabled()
        assert any("collected 2 cell(s)" in m for m in state.messages)
    finally:
        win.close()


def test_pooling_concatenates_across_datasets(app):
    """The requested workflow: collect, switch dataset, collect again."""
    state = _state_with_two_datasets()
    win = HlyBCollectionWindow(state, owner=None)
    try:
        _collect_dataset(win, state, 0, [
            _polygon("cell 1", 0.0, 0.0, 1600.0, 0),
            _polygon("cell 2", 6000.0, 0.0, 1600.0, 0)])
        _collect_dataset(win, state, 1, [
            _polygon("cell 1", 0.0, 0.0, 1600.0, 1)])
        assert len(win._collection) == 3
        assert win._collection.datasets == ["ds A", "ds B"]
        assert win._table.rowCount() == 3
        assert "3</b> cell(s)" in win._summary.text()
        assert "2</b> dataset(s)" in win._summary.text()
    finally:
        win.close()


def test_recollecting_the_same_dataset_does_not_double_count(app, monkeypatch):
    monkeypatch.setattr(
        "PyQt6.QtWidgets.QMessageBox.information",
        staticmethod(lambda *a, **k: None))
    state = _state_with_two_datasets()
    win = HlyBCollectionWindow(state, owner=None)
    try:
        rois = [_polygon("cell 1", 0.0, 0.0, 1600.0, 0)]
        _collect_dataset(win, state, 0, rois)
        assert len(win._collection) == 1
        win._collect()                      # same dataset, same ROI, again
        assert len(win._collection) == 1
    finally:
        win.close()


def test_a_new_roi_on_an_already_collected_dataset_is_still_added(app):
    state = _state_with_two_datasets()
    win = HlyBCollectionWindow(state, owner=None)
    try:
        _collect_dataset(win, state, 0, [_polygon("cell 1", 0.0, 0.0, 1600.0, 0)])
        state.rois.add(_polygon("cell 2", 6000.0, 0.0, 1600.0, 0))
        win._collect()
        assert [c.roi for c in win._collection] == ["cell 1", "cell 2"]
    finally:
        win.close()


def test_remove_and_clear(app, monkeypatch):
    from PyQt6.QtWidgets import QMessageBox
    monkeypatch.setattr(
        "PyQt6.QtWidgets.QMessageBox.question",
        staticmethod(lambda *a, **k: QMessageBox.StandardButton.Yes))
    state = _state_with_two_datasets()
    win = HlyBCollectionWindow(state, owner=None)
    try:
        _collect_dataset(win, state, 0, [
            _polygon("cell 1", 0.0, 0.0, 1600.0, 0),
            _polygon("cell 2", 6000.0, 0.0, 1600.0, 0)])
        win._table.selectRow(0)
        win._remove_selected()
        assert [c.roi for c in win._collection] == ["cell 2"]
        win._clear()
        assert len(win._collection) == 0
        assert not win._run_btn.isEnabled()
    finally:
        win.close()


def test_the_pool_survives_closing_the_window(app):
    """A pooling session spans several datasets; an accidental close must not
    discard it."""
    state = _state_with_two_datasets()
    win = HlyBCollectionWindow(state, owner=None)
    _collect_dataset(win, state, 0, [_polygon("cell 1", 0.0, 0.0, 1600.0, 0)])
    win.close()

    again = HlyBCollectionWindow(state, owner=None)
    try:
        assert len(again._collection) == 1
        assert again._table.rowCount() == 1
    finally:
        again.close()


def test_collecting_without_rois_explains_itself(app, monkeypatch):
    seen = {}
    monkeypatch.setattr(
        "PyQt6.QtWidgets.QMessageBox.information",
        staticmethod(lambda parent, title, text, *a, **k:
                     seen.setdefault("text", text)))
    state = _state_with_two_datasets()
    win = HlyBCollectionWindow(state, owner=None)
    try:
        win._collect()
        assert len(win._collection) == 0
        assert "No region ROI" in seen.get("text", "")
        assert "Shape Model" in seen.get("text", "")
    finally:
        win.close()


def test_collecting_without_an_active_dataset_explains_itself(app, monkeypatch):
    seen = {}
    monkeypatch.setattr(
        "PyQt6.QtWidgets.QMessageBox.information",
        staticmethod(lambda parent, title, text, *a, **k:
                     seen.setdefault("text", text)))
    state = _State()
    win = HlyBCollectionWindow(state, owner=None)
    try:
        win._collect()
        assert "Open a dataset" in seen.get("text", "")
    finally:
        win.close()


def test_save_and_load_round_trip_through_the_window(app, monkeypatch, tmp_path):
    state = _state_with_two_datasets()
    win = HlyBCollectionWindow(state, owner=None)
    path = tmp_path / "pool.h5"
    try:
        _collect_dataset(win, state, 0, [
            _polygon("cell 1", 0.0, 0.0, 1600.0, 0),
            _polygon("cell 2", 6000.0, 0.0, 1600.0, 0)])
        monkeypatch.setattr(
            "PyQt6.QtWidgets.QFileDialog.getSaveFileName",
            staticmethod(lambda *a, **k: (str(path), "")))
        win._save()
        assert path.exists()
    finally:
        win.close()

    fresh = _State()
    other = HlyBCollectionWindow(fresh, owner=None)
    try:
        monkeypatch.setattr(
            "PyQt6.QtWidgets.QFileDialog.getOpenFileName",
            staticmethod(lambda *a, **k: (str(path), "")))
        other._load()
        assert len(other._collection) == 2
        assert [c.roi for c in other._collection] == ["cell 1", "cell 2"]
    finally:
        other.close()


def test_the_pooled_plugin_entry_is_registered():
    from minflux_viewer import plugins

    plugins.ensure_loaded()
    names = [entry.name for entry in plugins._REGISTRY]
    assert "HlyB/D subunit pair analysis" in names          # unchanged
    assert "HlyB/D pooled pair analysis (multi-dataset)" in names
    pooled = next(e for e in plugins._REGISTRY if "pooled" in e.name)
    assert "multiple datasets" in pooled.keywords
    assert "pool" in pooled.keywords


def test_pooled_log_line_and_payload_describe_the_pool():
    from minflux_viewer.analysis.hlyb_staged import (
        Staged3DConfig, analyze_hlyb_staged_pooled)
    from minflux_viewer.plugins.hlyb_pair_analysis.runner import (
        pooled_log_line, pooled_payload)

    cfg = Staged3DConfig(z_scaling_factor=1.0, null_replicates=19,
                         run_sensitivity=False, run_stratum_profile=False,
                         bootstrap_replicates=0)
    cells = []
    for index, name in enumerate(("ds A", "ds B")):
        loc, tid = _rod((20_000.0 * index, 0.0, 0.0), 200 + index)
        cells.append({"loc_m": loc * 1e-9, "tid": tid, "tim": None,
                      "label": f"{name} · cell 1", "dataset": name,
                      "roi": "cell 1"})
    result = analyze_hlyb_staged_pooled(cells, cfg)

    line = pooled_log_line(cfg, result)
    # Shares the prefix the method-text generator matches on.
    assert line.startswith("HlyB/D subunit pair analysis on ")
    assert "pooled" in line and "2 dataset(s)" in line

    payload = pooled_payload(cfg, result)
    assert payload["pooled"] is True
    assert payload["schema"] == "hlyb_staged_short_range_3d/v1"
    assert payload["input"]["datasets"] == ["ds A", "ds B"]
    assert len(payload["input"]["cells"]) == 2
    assert payload["parameters"]["component_mode"] == "given"


def test_method_text_generates_from_a_pooled_run():
    from minflux_viewer.analysis.hlyb_staged import (
        Staged3DConfig, analyze_hlyb_staged_pooled)
    from minflux_viewer.analysis.method_text import generate_method_text
    from minflux_viewer.plugins.hlyb_pair_analysis.runner import (
        pooled_log_line, pooled_payload)

    cfg = Staged3DConfig(z_scaling_factor=1.0, null_replicates=19,
                         run_sensitivity=False, run_stratum_profile=False,
                         bootstrap_replicates=0)
    loc, tid = _rod((0.0, 0.0, 0.0), 300)
    result = analyze_hlyb_staged_pooled(
        [{"loc_m": loc * 1e-9, "tid": tid, "tim": None, "label": "c",
          "dataset": "ds A", "roi": "cell 1"}], cfg)
    events = [{"message": pooled_log_line(cfg, result), "level": "INFO",
               "method_data": pooled_payload(cfg, result)}]
    text = generate_method_text(None, events)
    assert isinstance(text, str) and len(text) > 200
