"""Tests for the MINFLUX-dataset definition / capabilities + canonical builder."""

import numpy as np

from minflux_viewer.core.dataset import build_localization_dataset
from minflux_viewer.core.dataset_kind import (
    capabilities,
    has_quality,
    has_traces,
    is_minflux,
    require,
)


def _ds(n=50, *, tid=None, z=None, tim=None, attrs=None):
    rng = np.random.default_rng(0)
    return build_localization_dataset(
        name="t.csv",
        x_nm=rng.uniform(0, 1000, n),
        y_nm=rng.uniform(0, 1000, n),
        z_nm=z,
        tid=tid,
        tim=tim,
        attrs=attrs,
    )


def test_builder_stores_canonical_metres_not_nm():
    ds = _ds(40)
    assert ds.attr.get("loc_x") is not None
    assert ds.attr.get("loc_x_nm") is None          # the duplicate convention is gone
    # loc_x is metres; the nm view (xnm / loc_nm) recovers nanometres
    assert np.allclose(ds.xnm, np.asarray(ds.attr.get("loc_x")) * 1e9)
    assert np.allclose(ds.loc_nm[:, 0], ds.xnm)
    assert "idx" in ds.attr                          # derived 1-based index always present


def test_z_zero_filled_when_2d():
    ds = _ds(30, z=None)
    z = np.asarray(ds.attr.get("loc_z"))
    assert z.shape[0] == 30 and np.all(z == 0)
    assert ds.prop.num_dim == 2


def test_real_tid_is_minflux():
    ds = _ds(60, tid=np.arange(60) % 10)            # real trace ids
    assert has_traces(ds) and is_minflux(ds)
    assert ds.metadata["has_real_tid"] is True
    assert require(ds, "loc", "traces") == []


def test_synthetic_tid_is_non_minflux():
    ds = _ds(60, tid=None)                            # no tid → synthetic 1..N
    assert not has_traces(ds) and not is_minflux(ds)
    assert ds.metadata["is_minflux"] is False
    assert require(ds, "loc") == []                  # loc still available
    assert require(ds, "traces") == ["traces"]       # but not trace-based analyses


def test_quality_capability():
    ds_no = _ds(20, tid=np.zeros(20))
    assert not has_quality(ds_no)
    assert require(ds_no, "quality") == ["quality"]
    ds_q = _ds(20, tid=np.zeros(20), attrs={"efo": np.full(20, 1e5), "fbg": np.full(20, 1e4)})
    assert has_quality(ds_q)
    assert require(ds_q, "quality") == []


def test_capabilities_map():
    ds = _ds(40, tid=np.arange(40) % 8, z=np.full(40, 25.0),
             tim=np.arange(40, dtype=float), attrs={"cfr": np.full(40, 0.5)})
    caps = capabilities(ds)
    assert caps == {"loc": True, "traces": True, "time": True, "quality": True, "3d": True}
