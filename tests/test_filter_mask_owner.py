"""The filter mask has one owner: ``state["filter_mask"]``.

``attr["ftr"]`` is a read-only, virtual view of it -- listed where it always
was, read from the mask on every access, and never stored a second time -- and
``filter_mask`` hands out a read-only view, so the only way to change the mask
is to assign it (which is what notifies the views).
"""

from __future__ import annotations

import copy

import numpy as np
import pytest

from minflux_viewer.core.dataset import (
    AttrStore,
    DataProp,
    FileInfo,
    MinfluxDataset,
    _Virtual,
    build_localization_dataset,
)


def _dataset(n: int = 6, ftr=None) -> MinfluxDataset:
    attrs = {
        "loc_x": np.linspace(0, 1e-6, n),
        "loc_y": np.linspace(0, 1e-6, n),
        "loc_z": np.zeros(n),
        "ftr": np.ones(n, dtype=bool) if ftr is None else np.asarray(ftr, dtype=bool),
        "tid": np.repeat([0, 1], n // 2),
    }
    return MinfluxDataset(
        file=FileInfo(name="t.mat", folder="/tmp"),
        prop=DataProp(num_loc=n, num_itr=1, num_dim=2, num_traces=2),
        attr=AttrStore(attrs),
    )


def test_ftr_is_stored_once_and_read_from_the_mask():
    ds = _dataset()
    raw = object.__getattribute__(ds.attr, "_d")
    assert isinstance(raw["ftr"], _Virtual)
    assert ds.attr.keys() == ["loc_x", "loc_y", "loc_z", "ftr", "tid"]  # same place

    mask = np.array([True, False, True, True, False, True])
    ds.filter_mask = mask
    np.testing.assert_array_equal(ds.attr["ftr"], mask)
    np.testing.assert_array_equal(ds.attr.ftr, mask)
    np.testing.assert_array_equal(ds.attr.get("ftr"), mask)
    np.testing.assert_array_equal(dict(ds.attr.items())["ftr"], mask)
    assert isinstance(raw["ftr"], _Virtual)  # still nothing stored there


@pytest.mark.parametrize("write", [
    lambda a: a.__setitem__("ftr", np.zeros(6, dtype=bool)),
    lambda a: setattr(a, "ftr", np.zeros(6, dtype=bool)),
    lambda a: a.__delitem__("ftr"),
    lambda a: a.pop("ftr"),
])
def test_writing_ftr_is_refused_and_names_the_mask(write):
    ds = _dataset()
    with pytest.raises(TypeError, match="filter_mask"):
        write(ds.attr)


def test_the_mask_is_read_only_and_the_setter_copies():
    ds = _dataset()
    with pytest.raises(ValueError):
        ds.filter_mask[0] = False            # the default all-pass mask
    with pytest.raises(ValueError):
        ds.attr["ftr"][0] = False

    mask = np.ones(6, dtype=bool)
    ds.filter_mask = mask
    mask[0] = False                          # the caller's array is not the mask
    assert ds.filter_mask.all()
    with pytest.raises(ValueError):
        ds.filter_mask[1] = False
    with pytest.raises(ValueError):
        ds.filter_mask &= np.zeros(6, dtype=bool)


def test_everything_passes_without_writing_state():
    ds = _dataset()
    assert ds.filter_mask.all() and ds.filter_mask.shape == (6,)
    assert "filter_mask" not in ds.state


def test_a_loaded_filter_is_adopted_and_an_all_pass_one_is_not():
    saved = [True, True, False, True, False, True]
    ds = _dataset(ftr=saved)
    np.testing.assert_array_equal(ds.state["filter_mask"], saved)
    np.testing.assert_array_equal(ds.attr["ftr"], saved)

    assert "filter_mask" not in _dataset().state


@pytest.mark.parametrize("mask", [True, [], [True] * 5, [True] * 7,
                                  np.ones((6, 1)), np.ones((1, 6))])
def test_invalid_mask_assignment_does_not_replace_the_previous_selection(mask):
    ds = _dataset()
    ds.filter_mask = [True, False] * 3
    before = ds.filter_mask.copy()
    with pytest.raises(ValueError, match="filter_mask must have shape"):
        ds.filter_mask = mask
    np.testing.assert_array_equal(ds.filter_mask, before)


def test_malformed_all_pass_import_is_not_silently_ignored():
    with pytest.raises(ValueError, match="filter_mask must have shape"):
        _dataset(ftr=[True] * 5)


@pytest.mark.parametrize("mask", [[True, False], True, np.ones((6, 1))])
def test_api_refuses_a_corrupted_state_mask(mask):
    from minflux_viewer.api._base import ApiError
    from minflux_viewer.api.data import Data

    ds = _dataset()
    ds.state["filter_mask"] = mask
    with pytest.raises(ApiError, match="Invalid filter mask"):
        Data._apply_filter(ds, np.arange(6))
    # Non-materialized values are intentionally not filtered.
    np.testing.assert_array_equal(Data._apply_filter(ds, np.arange(12)), np.arange(12))


def test_a_copied_dataset_has_its_own_mask():
    ds = _dataset()
    ds.filter_mask = np.array([True, False] * 3)
    clone = copy.deepcopy(ds)
    clone.filter_mask = np.zeros(6, dtype=bool)
    np.testing.assert_array_equal(ds.attr["ftr"], [True, False] * 3)
    np.testing.assert_array_equal(clone.attr["ftr"], np.zeros(6, dtype=bool))


def test_a_store_copied_alone_reads_the_dataset_it_is_given_to():
    source = _dataset()
    source.filter_mask = np.zeros(6, dtype=bool)
    store = copy.deepcopy(source.attr)
    with pytest.raises(RuntimeError):
        store["ftr"]                         # bound to nothing yet

    other = MinfluxDataset(
        file=FileInfo(name="u.mat", folder="/tmp"),
        prop=DataProp(num_loc=6, num_itr=1, num_dim=2, num_traces=2),
        attr=store,
    )
    assert other.attr["ftr"].all()           # the new dataset's own (default) mask
    assert not source.attr["ftr"].any()


def test_mfv_reads_ftr_through_the_mask(qapp):
    from minflux_viewer.core.app_state import AppState

    state = AppState()
    ds = build_localization_dataset(name="m", x_nm=np.arange(6.0), y_nm=np.arange(6.0))
    state.add_dataset(ds)
    ds.filter_mask = np.array([True, False] * 3)
    np.testing.assert_array_equal(state.mfv.data.attr("ftr", filtered=False),
                                  [True, False] * 3)


def test_a_roi_selection_leaves_the_dataset_mask_alone(qapp):
    """Selection narrowed the dataset's own mask in place (rows with NaN coordinates)."""
    from minflux_viewer.core.app_state import AppState
    from minflux_viewer.core.roi import RoiRecord
    from minflux_viewer.ui.plots.scatter_window import ScatterWindow

    state = AppState()
    x = np.array([0.0, 10.0, np.nan, 30.0, 40.0, 50.0])
    ds = build_localization_dataset(name="nan", x_nm=x, y_nm=np.arange(6.0) * 10)
    state.add_dataset(ds)
    ds.filter_mask = np.array([True, True, True, True, False, True])
    before = np.array(ds.state["filter_mask"])

    window = ScatterWindow(state, dataset_idx=0)
    try:
        record = RoiRecord.create("rectangle", {"bounds": [-5.0, -5.0, 60.0, 60.0]})
        window.compute_roi_selection(record)
        np.testing.assert_array_equal(ds.state["filter_mask"], before)
    finally:
        window.close()
        window.deleteLater()
        qapp.processEvents()
