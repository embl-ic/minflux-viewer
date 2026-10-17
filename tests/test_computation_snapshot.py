"""Post-load workers capture recipes and stable published column versions."""

import numpy as np

from minflux_viewer.core.computation_snapshot import computation_snapshot, input_signature
from minflux_viewer.core.dataset import build_localization_dataset
from minflux_viewer.core.loader_raw_access import evaluate_saved_filters


def test_snapshot_keeps_original_recipe_columns_and_calibration():
    ds = build_localization_dataset(name="test", x_nm=np.arange(4.), y_nm=np.arange(4.),
                                     attrs={"efo": np.arange(4.)})
    ds.state["filter_specs"] = [dict(attribute="efo", mode="per loc", lo=1, hi=2)]
    signature = input_signature(ds)
    snapshot = computation_snapshot(ds)
    old_coordinates = snapshot.attr["loc_x"]
    assert np.shares_memory(old_coordinates, ds.attr["loc_x"])
    ds.state["filter_specs"][0]["lo"] = 3
    ds.attr["efo"][1] = 99  # the captured mutable filter dependency is private
    ds.attr.replace_coordinates(loc_x=np.arange(4.) + 1)
    ds.set_z_scaling_factor(.5)
    assert input_signature(ds) != signature
    assert snapshot.cali.z_scaling_factor == 1
    assert not snapshot.attr["loc_x"].flags.writeable
    np.testing.assert_array_equal(evaluate_saved_filters(snapshot), [False, True, True, False])
    np.testing.assert_array_equal(old_coordinates, np.arange(4.) * 1e-9)


def test_lazy_raw_cache_does_not_invalidate_signature():
    ds = build_localization_dataset(name="test", x_nm=np.arange(4.), y_nm=np.arange(4.))
    signature = input_signature(ds)
    ds.mfx_raw["loc_id"] = np.arange(4)
    assert input_signature(ds) == signature


def test_version_checks_identity_revisions_and_unannounced_replacements():
    from minflux_viewer.core.app_state import AppState
    from minflux_viewer.core.computation_snapshot import DatasetInputVersion

    ds = build_localization_dataset(name="version", x_nm=np.arange(4.), y_nm=np.arange(4.))
    state = AppState()
    state.add_dataset(ds)
    version = DatasetInputVersion.capture(state, ds)
    assert version.is_current(state)
    ds.attr.replace_coordinates(loc_x=np.ones(4))
    assert not version.is_current(state)
    version = DatasetInputVersion.capture(state, ds)
    state.notify_attributes_changed(0)
    assert not version.is_current(state)
    version = DatasetInputVersion.capture(state, ds)
    state.remove_dataset(0)
    state.add_dataset(ds)
    assert not version.is_current(state)


def test_trajectory_snapshot_isolates_writable_times_and_trace_ids():
    from minflux_viewer.analysis.tracking_stats import snapshot_dataset_trajectories

    ds = build_localization_dataset(name="tracking", x_nm=np.arange(4.), y_nm=np.arange(4.),
                                     tid=np.arange(4), tim=np.arange(4.))
    snapshot = snapshot_dataset_trajectories(ds)
    ds.attr["tid"][:] = 99
    ds.attr["tim"][:] = 55
    np.testing.assert_array_equal(snapshot.trace_ids, np.arange(4))
    np.testing.assert_array_equal(snapshot.timestamps_s, np.arange(4.))
