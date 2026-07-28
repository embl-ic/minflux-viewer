from __future__ import annotations

import numpy as np
import pytest

from minflux_viewer.core.dataset import (
    AttrStore,
    DataProp,
    DatasetComponents,
    FileInfo,
    MinfluxDataset,
)
from minflux_viewer.core.time_channels import (
    TimeWindow,
    clone_time_channel_dataset,
    time_channel_selections,
    validate_time_windows,
)


def _dataset() -> MinfluxDataset:
    tim = np.arange(7, dtype=float)
    attrs = AttrStore({
        "tim": tim,
        "tid": np.arange(tim.size),
        "loc_x": tim * 1e-9,
        "loc_y": tim * 1e-9,
        "loc_z": np.zeros(tim.size),
    })
    components = DatasetComponents(mfx_raw=AttrStore({
        "tim": tim.copy(),
        "tid": np.arange(tim.size),
        "itr": np.zeros(tim.size, dtype=int),
    }))
    ds = MinfluxDataset(
        file=FileInfo("source.mat", "/tmp"),
        prop=DataProp(num_loc=tim.size, num_itr=1, num_dim=2),
        attr=attrs,
        components=components,
    )
    ds.filter_mask = np.array([True, True, False, True, True, True, True])
    ds.state["filter_specs"] = [{
        "attribute": "efo",
        "mode": "per loc",
        "lo": 1000.0,
        "hi": 100000.0,
        "lo_inc": True,
        "hi_inc": True,
    }]
    ds.state["overlay_id"] = "old-overlay"
    ds.metadata["overlay_id"] = "old-overlay"
    return ds


def test_validate_time_windows_sorts_and_rejects_overlap():
    windows = [
        TimeWindow("late", 5.0, 7.0, "Green"),
        TimeWindow("early", 0.0, 3.0, "Red"),
    ]
    assert [item.name for item in validate_time_windows(windows)] == ["early", "late"]

    with pytest.raises(ValueError, match="overlap"):
        validate_time_windows([
            TimeWindow("one", 0.0, 4.0, "Red"),
            TimeWindow("two", 3.0, 6.0, "Green"),
        ])


def test_adjacent_time_windows_assign_boundary_once_and_keep_current_filter():
    source = _dataset()
    selections = time_channel_selections(
        source.attr["tim"],
        [
            TimeWindow("one", 0.0, 3.0, "Red"),
            TimeWindow("two", 3.0, 6.0, "Green"),
        ],
        base_mask=source.filter_mask,
    )

    assert selections[0].upper_inclusive is False
    assert selections[1].upper_inclusive is True
    np.testing.assert_array_equal(
        selections[0].mask,
        [True, True, False, False, False, False, False],
    )
    np.testing.assert_array_equal(
        selections[1].mask,
        [False, False, False, True, True, True, True],
    )


def test_gap_keeps_upper_endpoint_and_clone_omits_private_raw_store():
    source = _dataset()
    selections = time_channel_selections(
        source.attr["tim"],
        [
            TimeWindow("one", 0.0, 2.0, "Red"),
            TimeWindow("two", 4.0, 6.0, "Green"),
        ],
        base_mask=source.filter_mask,
    )
    assert selections[0].upper_inclusive is True

    duplicate = clone_time_channel_dataset(source, selections[1], name="round 2")
    assert duplicate.name == "round 2"
    assert duplicate.prop.num_loc == 3
    assert duplicate.state.get("overlay_id") is None
    assert duplicate.metadata.get("overlay_id") is None
    assert duplicate.metadata["time_channels_source_dataset"] == source.name
    assert duplicate.metadata["time_channels_source_num_loc"] == 7
    assert duplicate.metadata["time_channels_selected_num_loc"] == 3
    assert duplicate.metadata["time_channels_source_num_itr"] == 1
    assert duplicate.metadata["raw_num_itr"] == 1
    assert duplicate.metadata["iteration_load_mode"] == "last"
    assert duplicate.state["filter_specs"][-1] == {
        "attribute": "tim",
        "mode": "per loc",
        "lo": 4.0,
        "hi": 6.0,
        "lo_inc": True,
        "hi_inc": True,
    }
    np.testing.assert_array_equal(
        duplicate.filter_mask,
        [True, True, True],
    )
    np.testing.assert_array_equal(duplicate.attr["tim"], [4.0, 5.0, 6.0])
    assert len(duplicate.mfx_raw) == 0
    duplicate.attr["tim"][0] = 99.0
    assert source.attr["tim"][4] == 4.0
