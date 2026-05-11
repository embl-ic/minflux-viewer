"""
Basic unit tests for the data model and loader utilities.
Run with:  poetry run pytest
"""

import numpy as np
import pytest

from minflux_viewer.core.dataset import AttrStore, MinfluxDataset, FileInfo, DataProp


# ---------------------------------------------------------------------------
# AttrStore
# ---------------------------------------------------------------------------

class TestAttrStore:
    def test_set_get_attribute_style(self):
        s = AttrStore()
        s.loc_x = np.array([1.0, 2.0, 3.0])
        np.testing.assert_array_equal(s.loc_x, [1.0, 2.0, 3.0])

    def test_set_get_dict_style(self):
        s = AttrStore()
        s["efo"] = np.array([100.0])
        assert "efo" in s
        np.testing.assert_array_equal(s["efo"], [100.0])

    def test_get_missing_raises_attribute_error(self):
        s = AttrStore()
        with pytest.raises(AttributeError):
            _ = s.nonexistent

    def test_get_with_default(self):
        s = AttrStore()
        result = s.get("missing", None)
        assert result is None

    def test_keys(self):
        s = AttrStore({"a": np.array([1]), "b": np.array([2])})
        assert set(s.keys()) == {"a", "b"}

    def test_len(self):
        s = AttrStore({"x": np.zeros(3)})
        assert len(s) == 1


# ---------------------------------------------------------------------------
# MinfluxDataset
# ---------------------------------------------------------------------------

def _make_dataset(n: int = 10, three_d: bool = False) -> MinfluxDataset:
    """Build a minimal synthetic dataset for testing."""
    file  = FileInfo(name="test.mat", folder="/tmp", datetime="2025-Jan-01")
    prop  = DataProp(
        num_loc=n,
        num_itr=3,
        num_dim=3 if three_d else 2,
        num_traces=2,
        trace_idx=np.array([[0, n // 2 - 1], [n // 2, n - 1]]),
        num_loc_per_trace=np.array([n // 2, n // 2]),
    )
    attrs = AttrStore({
        "loc_x": np.linspace(0, 1e-6, n),
        "loc_y": np.linspace(0, 1e-6, n),
        "loc_z": np.zeros(n) if not three_d else np.linspace(0, 5e-7, n),
        "tid":   np.repeat([0, 1], n // 2),
        "ftr":   np.ones(n, dtype=bool),
    })
    return MinfluxDataset(file=file, prop=prop, attr=attrs)


class TestMinfluxDataset:
    def test_name(self):
        ds = _make_dataset()
        assert ds.name == "test.mat"

    def test_loc_nm_shape(self):
        ds = _make_dataset(n=20)
        locs = ds.loc_nm
        assert locs.shape == (20, 3)

    def test_loc_nm_units(self):
        """loc values are in metres; loc_nm should convert to nanometres."""
        ds = _make_dataset(n=4)
        ds.attr.loc_x = np.array([1e-9, 2e-9, 3e-9, 4e-9])   # 1–4 nm in metres
        nm = ds.loc_nm[:, 0]
        np.testing.assert_allclose(nm, [1.0, 2.0, 3.0, 4.0], rtol=1e-10)

    def test_filter_mask_default_all_true(self):
        ds = _make_dataset(n=8)
        del ds.attr["ftr"]   # remove ftr to test default
        assert ds.filter_mask.all()
        assert ds.filter_mask.shape == (8,)

    def test_filter_mask_setter(self):
        ds = _make_dataset(n=6)
        mask = np.array([True, False, True, False, True, False])
        ds.filter_mask = mask
        np.testing.assert_array_equal(ds.filter_mask, mask)

    def test_summary_contains_name(self):
        ds = _make_dataset()
        assert "test.mat" in ds.summary()

    def test_repr(self):
        ds = _make_dataset(n=10)
        assert "test.mat" in repr(ds)


# ---------------------------------------------------------------------------
# Loader helpers (no actual .mat file needed)
# ---------------------------------------------------------------------------

class TestLoaderHelpers:
    def test_compute_properties_2d(self):
        from minflux_viewer.core.loader import _compute_properties, _add_derived_attributes

        n = 6
        attrs = AttrStore({
            "loc_x": np.ones(n) * 1e-6,
            "loc_y": np.ones(n) * 1e-6,
            "loc_z": np.zeros(n),
            "tid":   np.array([0, 0, 0, 1, 1, 1]),
            "tim":   np.arange(n, dtype=float),
        })
        prop = _compute_properties(attrs, num_loc=n, num_itr=2)
        assert prop.num_dim == 2
        assert prop.num_traces == 2
        assert prop.num_loc_per_trace.tolist() == [3, 3]

    def test_derived_attributes_added(self):
        from minflux_viewer.core.loader import _compute_properties, _add_derived_attributes

        n = 4
        attrs = AttrStore({
            "loc_x": np.array([0.0, 1e-9, 2e-9, 3e-9]),
            "loc_y": np.zeros(n),
            "loc_z": np.zeros(n),
            "tid":   np.array([0, 0, 1, 1]),
            "tim":   np.array([0.0, 0.1, 0.2, 0.3]),
        })
        prop  = _compute_properties(attrs, num_loc=n, num_itr=1)
        attrs = _add_derived_attributes(attrs, prop)

        assert "dt" in attrs
        assert "dst" in attrs
        assert "ftr" in attrs
        assert attrs["ftr"].all()   # default all-pass

    def test_attr_store_independence(self):
        """Modifying one AttrStore must not affect another."""
        a = AttrStore({"x": np.zeros(3)})
        b = AttrStore({"x": np.ones(3)})
        a["x"][0] = 99.0
        assert b["x"][0] == 1.0
