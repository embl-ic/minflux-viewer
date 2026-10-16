"""Localization coordinates are written in one place (slice 4.4).

Once a dataset is built its coordinate columns -- ``loc_x/y/z`` and the
uncorrected ``lnc_x/y/z``, in ``attr`` and in the all-iteration ``mfx_raw`` --
are read-only, as the filter mask became in 3.7: an in-place write raises, and
so does assigning or deleting a column. ``AttrStore.replace_coordinates`` is
the one way to change them, and it counts a revision a coordinate cache can
key on. No code writes coordinates in place today; this turns a future mistake
into an error instead of a silently changed dataset.
"""

from __future__ import annotations

import copy

import numpy as np
import pytest

from minflux_viewer.core.dataset import (
    COORDINATE_COLUMNS,
    AttrStore,
    build_localization_dataset,
)


def _dataset(n=50):
    rng = np.random.default_rng(0)
    return build_localization_dataset(
        name="guarded", x_nm=rng.uniform(0, 1000, n), y_nm=rng.uniform(0, 1000, n),
        z_nm=rng.normal(0, 50, n), source_version="simulation")


def _raw_store(n=50):
    rng = np.random.default_rng(1)
    return AttrStore({"loc_x": rng.normal(size=3 * n), "loc_y": rng.normal(size=3 * n),
                      "loc_z": rng.normal(size=3 * n), "lnc_x": rng.normal(size=3 * n),
                      "itr": np.tile(np.arange(3), n), "vld": np.ones(3 * n, bool)})


def test_the_guarded_columns_are_the_coordinates():
    assert set(COORDINATE_COLUMNS) == {"loc_x", "loc_y", "loc_z", "lnc_x", "lnc_y", "lnc_z"}


def test_a_built_dataset_refuses_an_in_place_coordinate_write():
    ds = _dataset()
    for name in ("loc_x", "loc_y", "loc_z"):
        column = ds.attr[name]
        assert not column.flags.writeable, name
        with pytest.raises(ValueError, match="read-only"):
            column[0] = 1.0
        with pytest.raises(ValueError, match="read-only"):
            column *= 2.0
    # The calibrated view is the dataset's own copy, free to change.
    view = ds.loc_nm
    view[0, 0] = -1.0
    assert ds.loc_nm[0, 0] != -1.0


def test_assigning_or_deleting_a_coordinate_column_names_the_write_path():
    ds = _dataset()
    for write in (lambda: ds.attr.__setitem__("loc_x", np.zeros(50)),
                  lambda: setattr(ds.attr, "loc_y", np.zeros(50)),
                  lambda: ds.attr.__delitem__("loc_z"),
                  lambda: ds.attr.__setitem__("lnc_x", np.zeros(50))):
        with pytest.raises(TypeError, match="replace_coordinates"):
            write()
    ds.attr["efo_doubled"] = np.zeros(50)        # other columns are unaffected
    assert "efo_doubled" in ds.attr


def test_replace_coordinates_is_the_one_way_to_change_them():
    ds = _dataset()
    revision = ds.attr.coordinate_revision
    new_x = np.linspace(0.0, 1e-6, 50)
    ds.attr.replace_coordinates(loc_x=new_x)
    assert ds.attr.coordinate_revision == revision + 1
    np.testing.assert_array_equal(ds.attr["loc_x"], new_x)
    assert not ds.attr["loc_x"].flags.writeable
    new_x[0] = 99.0                              # the caller's array is not kept
    assert ds.attr["loc_x"][0] == 0.0
    np.testing.assert_allclose(ds.loc_nm[:, 0], np.linspace(0.0, 1000.0, 50))

    with pytest.raises(ValueError, match="not a coordinate"):
        ds.attr.replace_coordinates(efo=np.zeros(50))
    with pytest.raises(ValueError, match="rows"):
        ds.attr.replace_coordinates(loc_y=np.zeros(49))
    assert ds.attr.coordinate_revision == revision + 1, "a refused write counts nothing"


def test_the_all_iteration_store_is_guarded_whenever_it_is_attached():
    ds = _dataset()
    raw = _raw_store()
    raw["loc_x"][0] = 5.0                        # a store being built is writable
    ds.components.mfx_raw = raw                  # as every loader attaches it
    assert not ds.mfx_raw["loc_x"].flags.writeable
    assert not ds.mfx_raw["lnc_x"].flags.writeable
    with pytest.raises(ValueError, match="read-only"):
        ds.mfx_raw["loc_z"][0] = 0.0
    with pytest.raises(TypeError, match="replace_coordinates"):
        ds.mfx_raw["loc_y"] = np.zeros(150)
    assert ds.mfx_raw["itr"].flags.writeable, "only coordinates are guarded"
    ds.mfx_raw["loc_id"] = np.arange(150)        # the lazy loc_id column still works


def test_a_copied_dataset_keeps_the_guard_on_its_own_columns():
    ds = _dataset()
    ds.components.mfx_raw = _raw_store()
    ds.attr.replace_coordinates(loc_z=np.zeros(50))
    clone = copy.deepcopy(ds)
    assert clone.attr["loc_x"] is not ds.attr["loc_x"]
    assert not clone.attr["loc_x"].flags.writeable
    assert not clone.mfx_raw["loc_y"].flags.writeable
    with pytest.raises(TypeError, match="replace_coordinates"):
        clone.attr["loc_x"] = np.zeros(50)
    assert clone.attr.coordinate_revision == ds.attr.coordinate_revision


def test_a_loaded_file_is_guarded(tmp_path):
    from minflux_viewer.core.loader import load_npy
    from minflux_viewer.core.save import save_processed

    ds = _dataset()
    path = tmp_path / "guarded.npy"
    save_processed(ds, data_path=path, fmt="npy")
    loaded = load_npy(path, apply_sidecar=False)
    for store in (loaded.attr, loaded.mfx_raw):
        for name in ("loc_x", "loc_y", "loc_z"):
            if name in store:
                assert not store[name].flags.writeable, name
