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
