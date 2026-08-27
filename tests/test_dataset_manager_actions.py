"""Dataset Manager — the single-row context menu and file-drop-on-a-row.

Right-clicking one row offers per-dataset actions (reset / save / close /
duplicate, then its MBM beads and image series, then confocal mapping), with the
source-dependent entries greyed out rather than hidden.

Dropping a file **on a row** applies it to that dataset — a different verb from
dropping on the main window, which opens a file as a new dataset.
"""

from __future__ import annotations

import json

import numpy as np
import pytest

pytest.importorskip("PyQt6")

from PyQt6.QtCore import QMimeData, QPoint, QPointF, QUrl
from PyQt6.QtGui import QDropEvent
from PyQt6.QtWidgets import QApplication, QMenu

from minflux_viewer.core.app_state import AppState
from minflux_viewer.core.dataset import build_localization_dataset
from minflux_viewer.core.dataset_reset import reset_dataset
from minflux_viewer.ui.dataset_manager import DatasetManager
from minflux_viewer.ui.main_window import MainWindow


@pytest.fixture(scope="module")
def _app():
    return QApplication.instance() or QApplication([])


def _dataset(name: str = "ds0"):
    rng = np.random.default_rng(0)
    p = rng.normal(size=(64, 3)) * 100.0
    return build_localization_dataset(
        name=name, x_nm=p[:, 0], y_nm=p[:, 1], z_nm=p[:, 2],
        source_version="simulation")


@pytest.fixture
def manager(_app):
    win = MainWindow(AppState())
    win._state.prefs.setdefault("data", {}).update(show_render=False, show_data_info=False)
    for i in range(2):
        win._state.add_dataset(_dataset(f"ds{i}"))
    mgr = DatasetManager(win._state, win)
    try:
        yield mgr, win
    finally:
        mgr.close()
        win.close()


# --- reset (pure core) ----------------------------------------------------

def test_reset_clears_filters_and_roi_masks_and_restores_z_scaling_factor(_app):
    ds = _dataset()
    ds.set_z_scaling_factor(0.72, source="auto (estimate anisotropy)")     # the as-loaded value
    ds.set_z_scaling_factor(0.90, source="manual (anisotropy plugin)")     # a later user edit

    ds.state["filter_specs"] = [{"attribute": "efo", "mode": "per loc",
                                 "lo": 0.0, "hi": 1.0}]
    ds.filter_mask = np.zeros(ds.prop.num_loc, dtype=bool)
    ds.derived["roi_abc"] = np.ones(ds.prop.num_loc, dtype=bool)
    ds.state["roi_masks"] = {"abc": {"key": "roi_abc"}}
    ds.state["active_roi_draft_id"] = "abc"
    ds.state["render_channel_lut"] = "jet"

    changes = reset_dataset(ds)

    assert ds.state["filter_specs"] == []
    assert ds.filter_mask.all()
    assert "roi_masks" not in ds.state and "active_roi_draft_id" not in ds.state
    assert ds.derived.get("roi_abc") is None
    assert ds.cali.z_scaling_factor == pytest.approx(0.72)
    assert "render_channel_lut" not in ds.state
    assert len(changes) == 4


def test_reset_keeps_overlay_membership_but_reverts_the_edited_transform(_app):
    """A per-dataset reset must not dissolve the group it is a channel of."""
    ds = _dataset()
    ds.state["overlay_id"] = "overlay:1:abc"
    ds.state["render_group_id"] = "overlay:1:abc"
    ds.state["overlay_order"] = 2
    imported = np.eye(4)
    ds.metadata["overlay_transform"] = imported
    ds.state["overlay_transform"] = np.full((4, 4), 3.0)       # a manual alignment

    reset_dataset(ds)

    assert ds.state["overlay_id"] == "overlay:1:abc"
    assert ds.state["overlay_order"] == 2
    assert ds.state["overlay_transform"] is imported


def test_reset_of_an_untouched_dataset_reports_nothing(_app):
    assert reset_dataset(_dataset()) == []


def test_reset_action_runs_through_the_manager(manager):
    mgr, win = manager
    ds = win._state.datasets[1]
    ds.state["filter_specs"] = [{"attribute": "efo", "mode": "per loc",
                                 "lo": 0.0, "hi": 1.0}]

    mgr._reset_row(1)

    assert ds.state["filter_specs"] == []


# --- single-row menu ------------------------------------------------------

def _menu_entries(mgr, monkeypatch, row):
    labels: list[str] = []

    def _capture(menu, *_a, **_k):
        labels.extend("---" if a.isSeparator() else a.text() for a in menu.actions())
        return None

    monkeypatch.setattr(QMenu, "exec", _capture)
    pos = mgr._table.visualRect(mgr._table.model().index(row, 1)).center()
    mgr._show_context_menu(pos)
    return labels


def test_single_row_menu_has_the_four_sections(manager, monkeypatch):
    mgr, _win = manager
    mgr._table.selectRow(0)

    assert _menu_entries(mgr, monkeypatch, 0) == [
        "Open file location",
        "---", "Reset", "Save as…", "Close", "Duplicate",
        "---", "View mbm info…", "View image series",
        "---", "Map confocal signal…",
    ]


def test_source_dependent_entries_are_disabled_not_hidden(manager, monkeypatch):
    """A simulated dataset has no MBM beads and no source .msr."""
    mgr, _win = manager
    disabled: list[str] = []

    def _capture(menu, *_a, **_k):
        disabled.extend(a.text() for a in menu.actions()
                        if not a.isSeparator() and not a.isEnabled())
        return None

    monkeypatch.setattr(QMenu, "exec", _capture)
    pos = mgr._table.visualRect(mgr._table.model().index(0, 1)).center()
    mgr._show_context_menu(pos)

    # A simulated dataset has no file behind it either.
    assert disabled == [
        "Open file location",
        "View mbm info…", "View image series", "Map confocal signal…",
    ]


def test_open_file_location_is_offered_only_with_a_file_on_disk(
    manager, monkeypatch, tmp_path
):
    """The first entry reveals the dataset's file, and greys out without one."""
    from minflux_viewer.core.dataset import dataset_source_file

    mgr, win = manager
    dataset = win._state.datasets[0]
    assert dataset_source_file(dataset) is None      # simulated: nothing on disk

    source = tmp_path / "run_1.mat"
    source.write_bytes(b"not really a .mat, but it exists")
    dataset.file.folder = str(tmp_path)
    dataset.file.name = source.name
    assert dataset_source_file(dataset) == source

    revealed: list[str] = []
    monkeypatch.setattr(
        type(win), "open_file_location", lambda _self, path: revealed.append(path)
    )
    captured: list[QMenu] = []

    def _choose(menu, *_a, **_k):
        captured.append(menu)
        return next(a for a in menu.actions() if a.text() == "Open file location")

    monkeypatch.setattr(QMenu, "exec", _choose)
    pos = mgr._table.visualRect(mgr._table.model().index(0, 1)).center()
    mgr._show_context_menu(pos)

    entry = captured[0].actions()[0]
    assert entry.text() == "Open file location"      # the top item
    assert entry.isEnabled() and entry.toolTip() == str(source)
    assert revealed == [str(source)]

    # A moved-away source is treated as absent.
    source.unlink()
    assert dataset_source_file(dataset) is None


def test_mbm_entry_enables_once_the_dataset_carries_beads(manager, monkeypatch):
    mgr, win = manager
    points = np.zeros(3, dtype=[("gri", "i4"), ("xyz", "f8", 3), ("tim", "f8")])
    win._state.datasets[0].metadata["mbm_points"] = points

    enabled: list[str] = []

    def _capture(menu, *_a, **_k):
        enabled.extend(a.text() for a in menu.actions()
                       if not a.isSeparator() and a.isEnabled())
        return None

    monkeypatch.setattr(QMenu, "exec", _capture)
    pos = mgr._table.visualRect(mgr._table.model().index(0, 1)).center()
    mgr._show_context_menu(pos)

    assert "View mbm info…" in enabled


def test_close_and_duplicate_from_the_single_row_menu(manager):
    mgr, win = manager

    mgr._duplicate_rows([0])
    assert [d.name for d in win._state.datasets] == ["ds0", "ds1", "DUP_ds0"]

    mgr._close_rows([1])
    assert [d.name for d in win._state.datasets] == ["ds0", "DUP_ds0"]


# --- view MBM / image series ----------------------------------------------

def test_view_mbm_reports_instead_of_opening_an_empty_plot(manager, monkeypatch):
    shown: list[str] = []
    monkeypatch.setattr(
        "minflux_viewer.ui.main_window.QMessageBox.information",
        staticmethod(lambda *a, **k: shown.append(a[2])))

    mgr, win = manager
    win.view_dataset_mbm(0)

    assert shown and "no beam-monitoring" in shown[0]


def test_view_mbm_opens_a_drift_window_for_a_dataset_with_beads(manager):
    mgr, win = manager
    # Two beads, three samples each, with the gri↔R-ID map the extractor needs.
    points = np.zeros(6, dtype=[("gri", "i4"), ("xyz", "f8", 3), ("tim", "f8")])
    points["gri"] = [1, 1, 1, 2, 2, 2]
    points["tim"] = [0.0, 1.0, 2.0] * 2
    points["xyz"][:, 0] = np.arange(6) * 1e-9
    ds = win._state.datasets[0]
    ds.metadata["mbm_points"] = points
    ds.metadata["mbm_points_by_gri"] = {"1": {"gri": 1, "name": "R1"},
                                        "2": {"gri": 2, "name": "R2"}}
    ds.metadata["mbm_used"] = ["R1", "R2"]

    before = len(getattr(win, "_modeless_windows", []))
    win.view_dataset_mbm(0)

    assert len(win._modeless_windows) == before + 1


def test_view_image_series_reports_when_there_is_no_source_msr(manager, monkeypatch):
    shown: list[str] = []
    monkeypatch.setattr(
        "minflux_viewer.ui.main_window.QMessageBox.information",
        staticmethod(lambda *a, **k: shown.append(a[2])))

    mgr, win = manager
    win.view_dataset_image_series(0)

    assert shown and "no available source .msr" in shown[0]


# --- drop on a row --------------------------------------------------------

def _drop(mgr, row, paths):
    """Synthesise a QDropEvent over *row* and push it through the event filter."""
    mime = QMimeData()
    mime.setUrls([QUrl.fromLocalFile(str(p)) for p in paths])
    centre = mgr._table.visualRect(mgr._table.model().index(row, 1)).center()
    from PyQt6.QtCore import Qt as _Qt
    event = QDropEvent(
        QPointF(centre), _Qt.DropAction.CopyAction, mime,
        _Qt.MouseButton.LeftButton, _Qt.KeyboardModifier.NoModifier,
    )
    from PyQt6.QtCore import QEvent
    assert event.type() == QEvent.Type.Drop
    mgr.eventFilter(mgr._table.viewport(), event)
    return event


def test_dropping_a_filter_preset_on_a_row_adds_it_to_that_dataset(manager, tmp_path):
    mgr, win = manager
    preset = tmp_path / "f.json"
    preset.write_text(json.dumps([{
        "attribute": "efo", "value_as": "per loc", "min": 1.0, "max": 2.0,
        "apply": True, "iteration": "last",
    }]), encoding="utf-8")

    win._state.set_active(0)
    _drop(mgr, 1, [preset])

    # Targeting the row means that dataset becomes the filter dialog's subject.
    assert win._state.active_idx == 1
    assert 1 in win._filter_dlgs
    assert win._filter_dlgs[1]._table.rowCount() == 1


def _sidecar(tmp_path, name="d_metadata.json"):
    from minflux_viewer.core.save import METADATA_JSON_MARKER

    side = tmp_path / name
    side.write_text(json.dumps({
        METADATA_JSON_MARKER: 1,
        "content": "raw",
        "calibration": {"z_scaling_factor": 0.67},
        "filters": [{"attribute": "efo", "mode": "per loc",
                     "lo": 0.0, "hi": 1e9, "itr": "last"}],
    }), encoding="utf-8")
    return side


def _answer(monkeypatch, button):
    """Answer the confirmation QMessageBox without a user."""
    from PyQt6.QtWidgets import QMessageBox

    monkeypatch.setattr(QMessageBox, "question",
                        staticmethod(lambda *a, **k: button))


def test_dropping_a_metadata_sidecar_applies_its_recipe(
        manager, tmp_path, monkeypatch):
    """A recipe changes Z scaling, filters and ROIs, so the drop confirms first
    -- the row already names the target, so the only question is whether."""
    from PyQt6.QtWidgets import QMessageBox

    mgr, win = manager
    _answer(monkeypatch, QMessageBox.StandardButton.Apply)

    _drop(mgr, 1, [_sidecar(tmp_path)])

    ds = win._state.datasets[1]
    assert ds.cali.z_scaling_factor == pytest.approx(0.67)
    assert len(ds.state["filter_specs"]) == 1


def test_cancelling_the_confirmation_applies_nothing(
        manager, tmp_path, monkeypatch):
    from PyQt6.QtWidgets import QMessageBox

    mgr, win = manager
    _answer(monkeypatch, QMessageBox.StandardButton.Cancel)

    _drop(mgr, 1, [_sidecar(tmp_path)])

    ds = win._state.datasets[1]
    assert ds.cali.z_scaling_factor == pytest.approx(1.0)
    assert not ds.state.get("filter_specs")


def test_dropping_a_roi_set_targets_the_dropped_on_dataset(manager, tmp_path):
    mgr, win = manager
    rois = tmp_path / "r.json"
    rois.write_text(json.dumps({"version": 1, "rois": [{
        "id": "a1", "type": "rectangle", "name": "rect-1",
        "geometry": {"bounds": [0, 0, 10, 10]},
    }]}), encoding="utf-8")

    win._state.set_active(0)
    _drop(mgr, 1, [rois])

    assert win._state.active_idx == 1
    assert len(win._state.rois.records) == 1


def test_a_data_file_on_a_row_is_opened_as_a_new_dataset(manager, tmp_path):
    """A row is not a meaningful target for a data file, but refusing one
    dropped on the window that *lists* datasets reads as a bug -- so it opens,
    and the Log says the row was not used."""
    mgr, win = manager
    data = tmp_path / "x.mat"
    data.write_bytes(b"not really a mat")

    routed = []
    win._route_path = lambda path: routed.append(str(path))

    assert win.drop_file_on_dataset(0, str(data)) is True
    assert routed == [str(data)]
    messages = [entry.get("message", "") for entry in win._state.log_history]
    assert any("not used" in text for text in messages)


def test_a_drop_between_rows_is_ignored(manager, tmp_path):
    mgr, win = manager
    preset = tmp_path / "f.json"
    preset.write_text("[]", encoding="utf-8")

    handled = []
    win.drop_file_on_dataset = lambda *a: handled.append(a)

    mime = QMimeData()
    mime.setUrls([QUrl.fromLocalFile(str(preset))])
    from PyQt6.QtCore import Qt as _Qt
    event = QDropEvent(
        QPointF(QPoint(5, 10_000)), _Qt.DropAction.CopyAction, mime,
        _Qt.MouseButton.LeftButton, _Qt.KeyboardModifier.NoModifier,
    )
    mgr.eventFilter(mgr._table.viewport(), event)

    assert handled == []


def test_an_extension_the_application_cannot_open_at_all_is_refused(
        manager, tmp_path):
    """Only a kind that is neither an annotation nor openable data is refused."""
    mgr, win = manager
    other = tmp_path / "x.docx"
    other.write_bytes(b"")

    assert win.drop_file_on_dataset(0, str(other)) is False


# --- TIFF as a confocal channel -------------------------------------------

def _calibrated_tiff(path, *, nm_per_px=20.0, n=64):
    tifffile = pytest.importorskip("tifffile")
    img = np.arange(n * n, dtype=np.uint16).reshape(n, n)
    tifffile.imwrite(
        str(path), img, ome=True,
        metadata={"PhysicalSizeX": nm_per_px / 1000.0, "PhysicalSizeXUnit": "µm",
                  "PhysicalSizeY": nm_per_px / 1000.0, "PhysicalSizeYUnit": "µm"},
    )
    return path


def test_tiff_candidate_is_centred_on_the_datasets_own_extent(_app, tmp_path):
    """A TIFF has a pixel size but no stage origin, so the drop itself supplies
    the placement: the image is centred on the dataset it was dropped on."""
    from minflux_viewer.core.confocal_mapping import candidates_from_tiff

    tif = _calibrated_tiff(tmp_path / "c.tif", nm_per_px=20.0, n=64)
    rng = np.random.default_rng(0)
    p = rng.normal(size=(500, 3)) * 200.0 + 50_000.0            # nm, around 50 µm
    ds = build_localization_dataset(
        name="d", x_nm=p[:, 0], y_nm=p[:, 1], z_nm=p[:, 2],
        source_version="simulation")

    (cand,) = candidates_from_tiff(tif, ds)

    assert cand.raw_index == -1 and cand.matches == ()          # not an OBF stack
    assert cand.shape == (64, 64) and cand.axes == "YX"
    assert cand.x_step_m == pytest.approx(20e-9)
    (x0, x1), (y0, y1) = cand.bounds_xy_m
    assert x1 - x0 == pytest.approx(64 * 20e-9)                 # 1280 nm field
    # Centred on the data, not parked at the stage origin.
    assert 0.5 * (x0 + x1) == pytest.approx(np.mean([p[:, 0].min(), p[:, 0].max()]) * 1e-9)
    assert 0.5 * (y0 + y1) == pytest.approx(np.mean([p[:, 1].min(), p[:, 1].max()]) * 1e-9)


def test_tiff_signal_maps_onto_the_localizations(_app, tmp_path):
    from minflux_viewer.core.confocal_mapping import (
        attach_confocal_signal,
        candidates_from_tiff,
        load_confocal_candidate_array,
    )

    tif = _calibrated_tiff(tmp_path / "c.tif", nm_per_px=20.0, n=64)
    rng = np.random.default_rng(0)
    p = rng.normal(size=(500, 3)) * 200.0 + 50_000.0
    ds = build_localization_dataset(
        name="d", x_nm=p[:, 0], y_nm=p[:, 1], z_nm=p[:, 2],
        source_version="simulation")

    (cand,) = candidates_from_tiff(tif, ds)
    image = load_confocal_candidate_array(tif, cand)             # dispatches to TIFF
    assert image.shape == (64, 64)

    result = attach_confocal_signal(ds, tif, cand, "Confocal", image=image)

    assert result.attribute_name == "Confocal"
    # Most localizations land inside the image; the Gaussian's tails fall outside
    # the 1280 nm field and are NaN rather than clamped.
    assert 0.9 < result.finite_count / result.total_count < 1.0
    assert ds.attr["Confocal"].shape == (500,)


def test_an_uncalibrated_tiff_is_refused_rather_than_guessed(_app, tmp_path):
    tifffile = pytest.importorskip("tifffile")
    from minflux_viewer.core.confocal_mapping import candidates_from_tiff

    tif = tmp_path / "plain.tif"
    tifffile.imwrite(str(tif), np.zeros((8, 8), dtype=np.uint16))
    ds = _dataset()

    with pytest.raises(ValueError, match="pixel-size calibration"):
        candidates_from_tiff(tif, ds)
