"""Trace Viewer per-trace computations (MATLAB trace viewer port)."""

import numpy as np

from minflux_viewer.core.dataset import build_localization_dataset
from minflux_viewer.plugins.trace_viewer.trace_data import (
    bbox_diagonal_nm,
    build_trace_table,
    moving_rms,
    remove_trace_tail,
    trace_ids,
    trace_time_series,
    _movmean,
)


def test_movmean_shrinking_edges():
    x = np.array([1.0, 2.0, 3.0, 4.0])
    # window 1 = identity
    assert np.allclose(_movmean(x, 1), x)
    # window 3, centred, edges shrink
    out = _movmean(x, 3)
    assert np.isclose(out[0], 1.5)            # mean of [1,2]
    assert np.isclose(out[1], 2.0)            # mean of [1,2,3]
    assert np.isclose(out[2], 3.0)            # mean of [2,3,4]


def test_moving_rms():
    x = np.array([3.0, 4.0])
    assert np.isclose(moving_rms(x, 1)[0], 3.0)
    assert np.isclose(moving_rms(x, 1)[1], 4.0)


def test_bbox_diagonal():
    xyz = np.array([[0, 0, 0], [3, 4, 0]], dtype=float)
    assert np.isclose(bbox_diagonal_nm(xyz), 5.0)
    assert bbox_diagonal_nm(np.empty((0, 3))) == 0.0


def test_trace_ids_drops_zero():
    assert trace_ids(np.array([0, 1, 1, 2, 0, 2])).tolist() == [1, 2]


def _synthetic_ds():
    # two traces: tid 7 (4 locs) and tid 3 (2 locs); plus efo/dcr/eco attrs
    rng = np.random.default_rng(0)
    n = 6
    tid = np.array([7, 7, 7, 7, 3, 3])
    tim = np.array([0.0, 0.01, 0.02, 0.03, 0.0, 0.01]) + np.arange(n) * 1e-6
    x = np.array([0.0, 10.0, 20.0, 30.0, 0.0, 5.0])     # nm
    y = np.array([0.0, 0.0, 0.0, 0.0, 0.0, 0.0])
    z = np.zeros(n)
    attrs = {"efo": np.full(n, 1e5), "dcr": np.full(n, 0.3), "eco": rng.uniform(100, 200, n)}
    return build_localization_dataset(name="syn", x_nm=x, y_nm=y, z_nm=z,
                                      attrs=attrs, tid=tid, tim=tim)


def test_build_trace_table_sorted_and_columns():
    ds = _synthetic_ds()
    rows, headers = build_trace_table(ds)
    assert headers[:3] == ["trace ID", "N loc", "efo"]
    assert headers[-1] == "size (nm)"
    # sorted by N loc descending → trace 7 (4 locs) first
    assert rows[0]["trace ID"] == 7 and rows[0]["N loc"] == 4
    assert rows[1]["trace ID"] == 3 and rows[1]["N loc"] == 2
    # trace 7 spans 0..30 nm in x → size 30
    assert np.isclose(rows[0]["size (nm)"], 30.0, atol=1e-6)


def test_trace_time_series_drops_first_point():
    ds = _synthetic_ds()
    tid = np.asarray(ds.attr["tid"]).ravel()
    s = trace_time_series(ds, tid, 7)
    assert s["n"] == 4
    assert s["ts"].size == 3                  # 4 locs → 3 time-series points
    assert s["xyz"].shape == (4, 3)
    # centred: mean ~ 0
    assert np.allclose(s["xyz"].mean(axis=0), 0.0, atol=1e-6)


def test_remove_trace_tail_returns_same_length():
    ds = _synthetic_ds()
    tid = np.asarray(ds.attr["tid"]).ravel()
    out = remove_trace_tail(ds)
    assert out.shape == tid.shape             # same length, some may be zeroed
    assert set(np.unique(out)).issubset({0, 3, 7})
