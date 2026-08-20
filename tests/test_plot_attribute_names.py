"""Canonical attribute-list ordering and synthetic index coverage."""

from __future__ import annotations

import numpy as np

from minflux_viewer.core.attributes import plot_attribute_names
from minflux_viewer.core.dataset import AttrStore, DataProp, FileInfo, MinfluxDataset
from minflux_viewer.core.loader import attr_values_1d, mfx_get


def _dataset_without_idx() -> MinfluxDataset:
    attrs = AttrStore({
        "loc_x": np.array([0.0, 1.0, 2.0]),
        "loc_y": np.array([0.0, 1.0, 2.0]),
        "tid": np.array([1, 1, 2]),
    })
    return MinfluxDataset(
        FileInfo("no-idx", ""),
        prop=DataProp(
            num_loc=3,
            num_itr=1,
            num_dim=2,
            num_traces=2,
            attr_names=attrs.keys(),
        ),
        attr=attrs,
    )


def test_idx_is_always_first_even_without_a_stored_column():
    dataset = _dataset_without_idx()
    prefs = {"attributes": {"enabled": ["loc", "tid"], "computed": []}}

    names = plot_attribute_names(dataset, prefs)

    assert names[0] == "idx"
    assert names.count("idx") == 1
    assert "xnm" in names


def test_specialized_attribute_list_can_exclude_idx():
    dataset = _dataset_without_idx()

    names = plot_attribute_names(dataset, {"attributes": {}}, exclude=("idx",))

    assert "idx" not in names


def test_idx_values_are_synthesized_for_generic_datasets():
    dataset = _dataset_without_idx()
    expected = np.array([1, 2, 3], dtype=np.uint32)

    np.testing.assert_array_equal(attr_values_1d(dataset, "idx"), expected)
    np.testing.assert_array_equal(mfx_get(dataset, "idx"), expected)
