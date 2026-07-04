"""Flatten a multi-channel overlay into one dataset (core/channel_flatten.py)."""

from __future__ import annotations

import numpy as np


def _channel(seed, *, n=20, attrs=None, tid=None):
    from minflux_viewer.core.dataset import build_localization_dataset
    rng = np.random.default_rng(seed)
    return build_localization_dataset(
        name=f"ch{seed}",
        x_nm=rng.normal(0, 50, n), y_nm=rng.normal(0, 50, n), z_nm=rng.normal(0, 20, n),
        tid=tid, attrs=attrs or {})


def _shift(ds, dx):
    from minflux_viewer.core.overlay import display_transform_record
    mat = np.eye(4)
    mat[0, 3] = dx
    ds.state["overlay_transform"] = display_transform_record(
        overlay_id="g", overlay_index=2, order=2, lut="Green",
        source_dataset_idx=1, alignment_mode="none", matrix_4x4=mat)


def test_flatten_bakes_transform_remaps_tids_and_combines_attrs():
    from minflux_viewer.core.channel_flatten import flatten_overlay
    from minflux_viewer.core.roi_crop import display_coords

    a = _channel(0, n=20, attrs={"efo": np.arange(20.0)},
                 tid=np.repeat(np.arange(10), 2))          # 10 traces, ids 0..9
    b = _channel(1, n=15, attrs={"efo": np.arange(15.0), "cfr": np.arange(15.0)},
                 tid=np.repeat(np.arange(5) + 3, 3))       # 5 traces, ids 3..7 (collide!)
    _shift(b, 100.0)                                        # overlay transform: +100 nm X

    res = flatten_overlay([a, b])
    r = res.report

    # coordinates concatenated with each channel's transform baked in
    assert res.coords_nm.shape == (35, 3)
    np.testing.assert_allclose(res.coords_nm[:20], display_coords(a), atol=1e-6)
    np.testing.assert_allclose(res.coords_nm[20:], display_coords(b), atol=1e-6)  # shifted +100

    # trace ids remapped so the colliding ids stay distinct
    assert r.n_traces == 15
    assert np.unique(res.tid).size == 15
    assert set(res.tid[:20].tolist()) == set(range(10))    # a → 0..9
    assert set(res.tid[20:].tolist()) == set(range(10, 15))  # b → 10..14 (offset)

    # efo present in both channels → kept; cfr only in b → dropped; tim → dropped (time)
    assert "efo" in res.attrs and res.attrs["efo"].shape == (35,)
    assert "efo" in r.kept_attrs
    dropped = dict(r.dropped)
    assert "cfr" in dropped and "missing" in dropped["cfr"]
    assert "tim" in dropped                                 # time attr always dropped
    assert r.n_channels == 2 and r.n_total == 35 and r.n_kept == 35


def test_flatten_honours_filter_mask():
    from minflux_viewer.core.channel_flatten import flatten_overlay

    a = _channel(2, n=20, tid=np.arange(20))
    b = _channel(3, n=20, tid=np.arange(20))
    # keep only the first 8 of channel a
    a.filter_mask = np.array([True] * 8 + [False] * 12)

    res = flatten_overlay([a, b])
    assert res.coords_nm.shape[0] == 8 + 20
    assert res.report.n_total == 40 and res.report.n_kept == 28
    assert any("filters" in note for note in res.report.notes)


def test_flatten_reports_differing_iteration_counts():
    from minflux_viewer.core.channel_flatten import flatten_overlay

    a = _channel(4, n=10, tid=np.arange(10))
    b = _channel(5, n=10, tid=np.arange(10))
    a.prop.num_itr = 5                                     # differing iteration counts
    b.prop.num_itr = 10
    res = flatten_overlay([a, b])
    assert any("iteration" in note for note in res.report.notes)
    assert res.coords_nm.shape[0] == 20                    # still flattens fine
