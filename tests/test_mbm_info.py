"""MBM (beam-monitoring bead) info for a loaded dataset.

Two things are covered:

* **the reconstruction bug** — a dataset carries its bead ``points`` array but
  not the ``points_by_gri`` name map, and ``extract_bead_drift`` then produced
  *no* traces at all, so *View mbm info…* always reported "no bead trace could
  be reconstructed". The gri ids are in the array itself and are now the last
  fallback;
* **the combined window** — the drift traces and the beads-vs-data-region view
  were both locked inside the MSR reader; they are now two tabs of one window
  driven by the dataset's own arrays.
"""

from __future__ import annotations

import numpy as np
import pytest

pytest.importorskip("PyQt6")

from PyQt6.QtWidgets import QApplication

from minflux_viewer.core.app_state import AppState
from minflux_viewer.core.dataset import AttributeComponent, build_localization_dataset
from minflux_viewer.plugins.msr_reader.beads_drift import (
    dataset_bead_drift,
    extract_bead_drift,
    single_channel_bead_summary,
)
from minflux_viewer.ui.main_window import MainWindow


@pytest.fixture(scope="module")
def _app():
    return QApplication.instance() or QApplication([])


def _points(n_beads: int = 3, n_t: int = 40, first_gri: int = 100):
    """A ``grd/mbm/points``-shaped array: per-sample gri / xyz (m) / tim (s)."""
    rng = np.random.default_rng(0)
    rows = []
    for b in range(n_beads):
        base = rng.normal(size=3) * 5e-6
        for t in range(n_t):
            drift = np.array([t, -t, 2 * t]) * 0.02e-9
            rows.append((first_gri + b, base + drift, float(t) * 10.0))
    return np.array(rows, dtype=[("gri", "u4"), ("xyz", "f8", 3), ("tim", "f8")])


def _dataset_with_beads(points=None, *, name="msr_ch1", **meta):
    rng = np.random.default_rng(1)
    p = rng.normal(size=(200, 3)) * 300.0 + 50_000.0
    ds = build_localization_dataset(
        name=name, x_nm=p[:, 0], y_nm=p[:, 1], z_nm=p[:, 2],
        source_version="simulation")
    if points is not None:
        ds.mbm = AttributeComponent({"points": points})
        ds.metadata["mbm_points"] = points
    ds.metadata.update(meta)
    return ds


# --- the reconstruction fallback ------------------------------------------

def test_beads_are_reconstructed_without_a_name_map():
    """The reported bug: an MSR-imported dataset carries `mbm_points` but no
    `mbm_points_by_gri`, and every bead was silently dropped."""
    beads = extract_bead_drift(_points(n_beads=3), {}, [])

    assert [b["gri"] for b in beads] == [100, 101, 102]
    assert [b["rid"] for b in beads] == ["100", "101", "102"]     # named by id
    assert beads[0]["xyz_nm"].shape == (40, 3)
    assert beads[0]["tim_s"][0] == 0.0                            # zeroed to start


def test_a_name_map_still_wins_over_the_id_fallback():
    pbg = {"1": {"gri": 100, "name": "R113"}, "2": {"gri": 101, "name": "R114"}}

    beads = extract_bead_drift(_points(n_beads=3), pbg, [])

    assert [b["rid"] for b in beads] == ["R113", "R114"]          # only the named two


def test_an_explicit_used_list_still_narrows_the_selection():
    pbg = {"1": {"gri": 100, "name": "R113"}, "2": {"gri": 101, "name": "R114"}}

    beads = extract_bead_drift(_points(n_beads=3), pbg, ["R114"])

    assert [b["rid"] for b in beads] == ["R114"]


def test_a_points_array_without_the_required_columns_yields_nothing():
    bad = np.zeros(3, dtype=[("gri", "u4"), ("tim", "f8")])       # no xyz

    assert extract_bead_drift(bad, {}, []) == []


def test_dataset_bead_drift_reads_the_dataset_s_own_arrays():
    ds = _dataset_with_beads(_points(n_beads=2))

    assert len(dataset_bead_drift(ds)) == 2
    assert dataset_bead_drift(_dataset_with_beads(None)) == []


def test_single_channel_summary_reports_peak_to_peak_drift():
    beads = extract_bead_drift(_points(n_beads=2, n_t=40), {}, [])

    payload = single_channel_bead_summary("ch", beads)

    assert payload["bead_ids"].tolist() == [100, 101]
    assert payload["pos_nm"].shape == (2, 3)
    # x drifts 0.02 nm/sample over 40 samples → 39 * 0.02 nm peak-to-peak.
    assert payload["drift_nm"][0][0] == pytest.approx(39 * 0.02, rel=1e-6)
    assert single_channel_bead_summary("ch", []) is None


# --- the combined window ---------------------------------------------------

@pytest.fixture
def window(_app):
    win = MainWindow(AppState())
    win._state.prefs.setdefault("data", {}).update(show_render=False, show_data_info=False)
    try:
        yield win
    finally:
        win.close()


def test_view_mbm_info_opens_both_views_in_one_window(window):
    window._state.add_dataset(_dataset_with_beads(_points(n_beads=3)))

    window.view_dataset_mbm(0)

    win = window._modeless_windows[-1]
    assert win.windowTitle() == "MBM info — msr_ch1"
    # "&&" is Qt's escape for a literal "&" in a tab label.
    assert [win._tabs.tabText(i) for i in range(win._tabs.count())] == [
        "Drift", "Beads && data region"]
    assert len(win._beads) == 3
    win.close()


def test_the_embedded_drift_view_is_read_only(window):
    """Embedded in the info window there is no alignment for a bead selection to
    feed, so the per-bead checkboxes and Apply/Cancel are hidden."""
    window._state.add_dataset(_dataset_with_beads(_points(n_beads=2)))

    window.view_dataset_mbm(0)
    drift = window._modeless_windows[-1]._drift

    assert drift._info_mode is True
    assert not drift._selection_label.isVisible()
    # The checkboxes still exist (the linked-selection bookkeeping is uniform)
    # but none of them was put into a layout.
    boxes = [cb for group in drift._checkboxes.values() for cb in group]
    assert boxes and all(cb.parent() is None for cb in boxes)
    window._modeless_windows[-1].close()


def test_the_region_view_gets_the_datasets_own_data_bounds(window):
    from minflux_viewer.ui.mbm_info_window import dataset_loc_bounds_nm

    ds = _dataset_with_beads(_points(n_beads=2))
    window._state.add_dataset(ds)
    expected = dataset_loc_bounds_nm(ds)

    window.view_dataset_mbm(0)
    region = window._modeless_windows[-1]._region

    assert np.allclose(region.data_bounds_nm[0], expected[0])
    assert np.allclose(region.data_bounds_nm[1], expected[1])
    assert region._single_channel is not None      # no-alignment mode
    window._modeless_windows[-1].close()


def test_a_dataset_without_beads_reports_the_reason(window, monkeypatch):
    shown: list[str] = []
    monkeypatch.setattr(
        "minflux_viewer.ui.main_window.QMessageBox.information",
        staticmethod(lambda *a, **k: shown.append(a[2])))
    window._state.add_dataset(_dataset_with_beads(None))

    window.view_dataset_mbm(0)

    assert shown and "no beam-monitoring" in shown[0]
    # show_modeless creates the registry lazily, so "no window" is also "no list".
    assert not [w for w in getattr(window, "_modeless_windows", [])
                if type(w).__name__ == "MbmInfoWindow"]


def test_a_malformed_points_array_is_reported_distinctly(window, monkeypatch):
    """Different cause, different message — "carries none" must not be reported
    for an array that is present but unusable."""
    shown: list[str] = []
    monkeypatch.setattr(
        "minflux_viewer.ui.main_window.QMessageBox.information",
        staticmethod(lambda *a, **k: shown.append(a[2])))
    bad = np.zeros(3, dtype=[("gri", "u4"), ("tim", "f8")])
    window._state.add_dataset(_dataset_with_beads(bad))

    window.view_dataset_mbm(0)

    assert shown and "no usable" in shown[0]
