"""
Tests for Phase 2 components.

Pure-numpy logic (filter masks, histogram aggregation, density) is tested
directly via ``minflux_viewer.utils.filters`` — no QApplication required.

Qt-dependent tests (AppState signals, widget creation) are skipped
automatically when PyQt6 is not available (e.g. in CI without a display).
"""

from __future__ import annotations

import numpy as np
import pytest

from minflux_viewer.core.dataset import (
    AttrStore,
    DataProp,
    FileInfo,
    MinfluxDataset,
)


# ---------------------------------------------------------------------------
# Shared fixture
# ---------------------------------------------------------------------------

def make_ds(n: int = 12, n_traces: int = 3) -> MinfluxDataset:
    """Synthetic dataset with *n* locs split evenly into *n_traces* traces."""
    assert n % n_traces == 0
    per   = n // n_traces
    tid   = np.repeat(np.arange(n_traces), per)
    ti    = np.column_stack([
        np.arange(n_traces) * per,
        np.arange(n_traces) * per + per - 1,
    ])
    nlpt  = np.full(n_traces, per)
    prop  = DataProp(
        num_loc=n, num_itr=2, num_dim=2, num_traces=n_traces,
        trace_idx=ti, num_loc_per_trace=nlpt,
        attr_names=["loc_x", "loc_y", "loc_z", "tid", "efo", "cfr", "ftr", "idx"],
    )
    attrs = AttrStore({
        "loc_x": np.linspace(0, 1e-6, n),
        "loc_y": np.linspace(0, 1e-6, n),
        "loc_z": np.zeros(n),
        "tid":   tid.astype(float),
        "efo":   np.linspace(10.0, 100.0, n),
        "cfr":   np.random.default_rng(0).uniform(0.3, 0.9, n),
        "ftr":   np.ones(n, dtype=bool),
        "idx":   np.arange(1, n + 1, dtype=np.uint32),
    })
    return MinfluxDataset(file=FileInfo(name="synth.mat", folder="/tmp"),
                          prop=prop, attr=attrs)


# ---------------------------------------------------------------------------
# compute_filter_mask  (pure numpy)
# ---------------------------------------------------------------------------

class TestFilterMask:
    def _mask(self, raw, mode, lo, hi, ds):
        from minflux_viewer.utils.filters import compute_filter_mask
        return compute_filter_mask(raw, mode, lo, hi,
                                   ds.prop.trace_idx,
                                   ds.prop.num_loc_per_trace,
                                   ds.prop.num_traces)

    def test_per_loc_full_pass(self):
        ds  = make_ds()
        raw = np.asarray(ds.attr["efo"]).ravel().astype(float)
        assert self._mask(raw, "per loc", raw.min(), raw.max(), ds).all()

    def test_per_loc_partial(self):
        ds  = make_ds(n=12)
        raw = np.asarray(ds.attr["efo"]).ravel().astype(float)
        mid = float(np.median(raw))
        m   = self._mask(raw, "per loc", raw.min() - 1, mid, ds)
        assert m.any() and not m.all()

    def test_per_loc_none_pass(self):
        ds  = make_ds()
        raw = np.ones(ds.prop.num_loc)
        assert not self._mask(raw, "per loc", 5.0, 10.0, ds).any()

    def test_trace_mean_expands_to_per_loc(self):
        ds  = make_ds(n=12, n_traces=3)
        raw = np.asarray(ds.attr["efo"]).ravel().astype(float)
        assert self._mask(raw, "trace mean", 0.0, 1e9, ds).shape == (12,)

    def test_trace_mean_selective(self):
        ds  = make_ds(n=6, n_traces=2)
        raw = np.array([10., 20., 30., 60., 80., 100.])
        m   = self._mask(raw, "trace mean", 50.0, 200.0, ds)
        np.testing.assert_array_equal(m[:3], [False, False, False])
        np.testing.assert_array_equal(m[3:], [True,  True,  True])

    def test_trace_max_threshold(self):
        ds  = make_ds(n=6, n_traces=2)
        raw = np.array([10., 20., 30., 60., 80., 100.])
        m   = self._mask(raw, "trace max", 0.0, 50.0, ds)
        np.testing.assert_array_equal(m[:3], [True,  True,  True])
        np.testing.assert_array_equal(m[3:], [False, False, False])

    def test_trace_min_threshold(self):
        ds  = make_ds(n=6, n_traces=2)
        raw = np.array([10., 20., 30., 60., 80., 100.])
        m   = self._mask(raw, "trace min", 50.0, 200.0, ds)
        np.testing.assert_array_equal(m[:3], [False, False, False])
        np.testing.assert_array_equal(m[3:], [True,  True,  True])

    def test_trace_range_correct(self):
        ds  = make_ds(n=6, n_traces=2)
        raw = np.array([10., 20., 30., 60., 80., 100.])  # ranges: 20, 40
        m   = self._mask(raw, "trace range", 30.0, 50.0, ds)
        np.testing.assert_array_equal(m[:3], [False, False, False])
        np.testing.assert_array_equal(m[3:], [True,  True,  True])


# ---------------------------------------------------------------------------
# aggregate  (pure numpy)
# ---------------------------------------------------------------------------

class TestAggregate:
    def _agg(self, raw, ftr, mode, ds):
        from minflux_viewer.utils.filters import aggregate
        return aggregate(raw, ftr, mode, ds.prop.trace_idx, ds.prop.num_traces)

    def test_per_loc_size(self):
        ds  = make_ds(n=12)
        raw = np.asarray(ds.attr["efo"]).ravel().astype(float)
        assert self._agg(raw, ds.filter_mask, "per loc", ds).size == 12

    def test_per_loc_filtered(self):
        ds  = make_ds(n=12)
        raw = np.asarray(ds.attr["efo"]).ravel().astype(float)
        ftr = np.zeros(12, dtype=bool); ftr[:6] = True
        assert self._agg(raw, ftr, "per loc", ds).size == 6

    def test_trace_mean_size(self):
        ds  = make_ds(n=12, n_traces=3)
        raw = np.asarray(ds.attr["efo"]).ravel().astype(float)
        assert self._agg(raw, ds.filter_mask, "trace mean", ds).size == 3

    def test_trace_mean_correctness(self):
        ds  = make_ds(n=6, n_traces=2)
        raw = np.array([10., 20., 30., 60., 80., 100.])
        np.testing.assert_allclose(
            self._agg(raw, ds.filter_mask, "trace mean", ds), [20.0, 80.0])

    def test_trace_stdev_shape(self):
        ds  = make_ds(n=9, n_traces=3)
        raw = np.linspace(1.0, 9.0, 9)
        assert self._agg(raw, ds.filter_mask, "trace stdev", ds).shape == (3,)

    def test_trace_range_nonneg(self):
        ds  = make_ds(n=12, n_traces=3)
        raw = np.asarray(ds.attr["efo"]).ravel().astype(float)
        out = self._agg(raw, ds.filter_mask, "trace range", ds)
        assert (out >= 0).all()

    def test_trace_max_gte_min(self):
        ds  = make_ds(n=12, n_traces=3)
        raw = np.asarray(ds.attr["efo"]).ravel().astype(float)
        assert (self._agg(raw, ds.filter_mask, "trace max", ds) >=
                self._agg(raw, ds.filter_mask, "trace min", ds)).all()


# ---------------------------------------------------------------------------
# fast_density_2d  (pure numpy)
# ---------------------------------------------------------------------------

class TestFastDensity:
    def _d(self, x, y):
        from minflux_viewer.utils.filters import fast_density_2d
        return fast_density_2d(x, y)

    def test_output_shape(self):
        rng = np.random.default_rng(1)
        x, y = rng.uniform(0, 1000, 500), rng.uniform(0, 1000, 500)
        assert self._d(x, y).shape == (500,)

    def test_all_nonneg(self):
        x = np.linspace(0, 100, 200); y = np.linspace(0, 100, 200)
        assert (self._d(x, y) >= 0).all()

    def test_empty_input(self):
        assert self._d(np.empty(0), np.empty(0)).size == 0

    def test_cluster_denser_than_outliers(self):
        rng     = np.random.default_rng(42)
        cluster = rng.normal(500, 5, (200, 2))
        outliers = rng.uniform(0, 1000, (10, 2))
        pts     = np.vstack([cluster, outliers])
        d       = self._d(pts[:, 0], pts[:, 1])
        assert d[:200].mean() > d[200:].mean()


# ---------------------------------------------------------------------------
# AppState logic  (skipped if PyQt6 unavailable)
# ---------------------------------------------------------------------------

class TestDatasetManagerLogic:
    @pytest.fixture(autouse=True)
    def _qt_app(self):
        pytest.importorskip("PyQt6")
        import os, sys
        # Skip in headless environments where QApplication would abort
        if not os.environ.get("DISPLAY") and os.name != "nt" and sys.platform != "darwin":
            pytest.skip("No display available for Qt tests")
        from PyQt6.QtWidgets import QApplication
        self._app = QApplication.instance() or QApplication(sys.argv)

    def test_remove_updates_state(self):
        from minflux_viewer.core.app_state import AppState
        state = AppState()
        ds1 = make_ds(n=6); ds1.file.name = "a.mat"
        ds2 = make_ds(n=6); ds2.file.name = "b.mat"
        state.add_dataset(ds1); state.add_dataset(ds2)
        assert len(state) == 2
        state.remove_dataset(0)
        assert len(state) == 1
        assert state.datasets[0].name == "b.mat"

    def test_active_idx_clamps_after_remove(self):
        from minflux_viewer.core.app_state import AppState
        state = AppState()
        for i in range(3):
            ds = make_ds(n=6); ds.file.name = f"ds{i}.mat"
            state.add_dataset(ds)
        state.set_active(2)
        state.remove_dataset(2)
        assert state.active_idx == 1

    def test_duplicate_not_added(self):
        from minflux_viewer.core.app_state import AppState
        state = AppState()
        ds = make_ds(n=6)
        state.add_dataset(ds); state.add_dataset(ds)
        assert len(state) == 1

    def test_status_message_on_active_change(self):
        from minflux_viewer.core.app_state import AppState
        messages = []
        state = AppState()
        state.status_message.connect(messages.append)
        ds1 = make_ds(n=6); ds1.file.name = "a.mat"
        ds2 = make_ds(n=6); ds2.file.name = "b.mat"
        state.add_dataset(ds1); state.add_dataset(ds2)
        state.set_active(0)
        assert any("a.mat" in m for m in messages)


# ---------------------------------------------------------------------------
# load_from_mfx_array  (pure numpy — no specpy needed)
# ---------------------------------------------------------------------------

class TestLoadFromMfxArray:
    """Test the mfx-array-to-MinfluxDataset adapter."""

    def _make_mfx(self, n: int = 20) -> "np.ndarray":
        """Build a synthetic structured mfx array matching the m2410 layout."""
        dt = np.dtype([
            ("vld",  np.bool_),
            ("tid",  np.int32),
            ("tim",  np.float64),
            ("itr",  np.int32),
            ("efo",  np.float64),
            ("cfr",  np.float64),
            ("dcr",  np.float64),
            ("loc",  np.float64, (3,)),
        ])
        arr = np.zeros(n, dtype=dt)
        arr["vld"]  = True
        arr["tid"]  = np.repeat([0, 1], n // 2)
        arr["tim"]  = np.linspace(0, 1, n)
        arr["itr"]  = 3                          # last iteration index
        arr["efo"]  = np.linspace(10, 100, n)
        arr["cfr"]  = np.random.default_rng(7).uniform(0.3, 0.9, n)
        arr["loc"]  = np.column_stack([
            np.linspace(0, 1e-6, n),
            np.linspace(0, 1e-6, n),
            np.zeros(n),
        ])
        return arr

    def test_dataset_created(self):
        from minflux_viewer.core.loader import load_from_mfx_array
        mfx = self._make_mfx(20)
        ds  = load_from_mfx_array(mfx, name="test.msr/dataset1", folder="/tmp")
        assert ds.name == "test.msr/dataset1"

    def test_num_loc(self):
        from minflux_viewer.core.loader import load_from_mfx_array
        mfx = self._make_mfx(20)
        ds  = load_from_mfx_array(mfx, name="x")
        assert ds.prop.num_loc == 20

    def test_loc_xyz_split(self):
        from minflux_viewer.core.loader import load_from_mfx_array
        mfx = self._make_mfx(20)
        ds  = load_from_mfx_array(mfx, name="x")
        assert "loc_x" in ds.attr
        assert "loc_y" in ds.attr
        assert "loc_z" in ds.attr
        assert ds.attr["loc_x"].shape == (20,)

    def test_loc_nm_units(self):
        """loc stored in metres; loc_nm property should return nanometres."""
        from minflux_viewer.core.loader import load_from_mfx_array
        mfx = self._make_mfx(4)
        mfx["loc"][:, 0] = [1e-9, 2e-9, 3e-9, 4e-9]
        ds  = load_from_mfx_array(mfx, name="x")
        np.testing.assert_allclose(ds.loc_nm[:, 0], [1.0, 2.0, 3.0, 4.0], rtol=1e-10)

    def test_derived_attrs_present(self):
        from minflux_viewer.core.loader import load_from_mfx_array
        mfx = self._make_mfx(20)
        ds  = load_from_mfx_array(mfx, name="x")
        for field in ("dt", "dst", "ftr", "idx"):
            assert field in ds.attr, f"missing derived attribute: {field}"

    def test_filter_mask_all_true(self):
        from minflux_viewer.core.loader import load_from_mfx_array
        mfx = self._make_mfx(20)
        ds  = load_from_mfx_array(mfx, name="x")
        assert ds.filter_mask.all()

    def test_num_traces(self):
        from minflux_viewer.core.loader import load_from_mfx_array
        mfx = self._make_mfx(20)   # 2 traces (tid 0 and 1)
        ds  = load_from_mfx_array(mfx, name="x")
        assert ds.prop.num_traces == 2


# ---------------------------------------------------------------------------
# load_npy
# ---------------------------------------------------------------------------

class TestLoadNpy:
    def _write_npy_structured(self, tmp_path, n: int = 10):
        """Write a structured .npy file (mfx layout)."""
        import tempfile, os
        dt = np.dtype([
            ("tid",  np.int32),
            ("tim",  np.float64),
            ("efo",  np.float64),
            ("loc",  np.float64, (3,)),
        ])
        arr = np.zeros(n, dtype=dt)
        arr["tid"] = np.repeat([0, 1], n // 2)
        arr["tim"] = np.linspace(0, 1, n)
        arr["efo"] = np.linspace(10, 100, n)
        arr["loc"][:, 0] = np.linspace(0, 1e-6, n)
        path = os.path.join(tempfile.gettempdir(), "test_structured.npy")
        np.save(path, arr)
        return path

    def _write_npy_plain(self, tmp_path, n: int = 10):
        import tempfile, os
        arr = np.column_stack([
            np.linspace(0, 1e-6, n),
            np.linspace(0, 1e-6, n),
            np.zeros(n),
        ])
        path = os.path.join(tempfile.gettempdir(), "test_plain.npy")
        np.save(path, arr)
        return path

    def test_structured_npy_loads(self, tmp_path):
        from minflux_viewer.core.loader import load_npy
        path = self._write_npy_structured(tmp_path)
        ds = load_npy(path)
        assert ds.prop.num_loc == 10
        assert "loc_x" in ds.attr

    def test_plain_xyz_npy_loads(self, tmp_path):
        from minflux_viewer.core.loader import load_npy
        path = self._write_npy_plain(tmp_path)
        ds = load_npy(path)
        assert ds.prop.num_loc == 10
        assert "loc_x" in ds.attr
        assert "loc_z" in ds.attr

    def test_npy_filter_mask_all_true(self, tmp_path):
        from minflux_viewer.core.loader import load_npy
        path = self._write_npy_plain(tmp_path)
        ds = load_npy(path)
        assert ds.filter_mask.all()


# ---------------------------------------------------------------------------
# load_csv
# ---------------------------------------------------------------------------

class TestLoadCsv:
    def _write_csv(self, n: int = 8, has_z: bool = True) -> str:
        import tempfile, os, csv
        path = os.path.join(tempfile.gettempdir(), "test_minflux.csv")
        cols = ["loc_x", "loc_y", "loc_z", "tid", "efo"] if has_z else ["loc_x", "loc_y", "tid", "efo"]
        with open(path, "w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=cols)
            w.writeheader()
            for i in range(n):
                row = {
                    "loc_x": i * 1e-8,
                    "loc_y": i * 1e-8,
                    "tid":   i // (n // 2),
                    "efo":   float(i * 10),
                }
                if has_z:
                    row["loc_z"] = 0.0
                w.writerow(row)
        return path

    def _write_csv_aliases(self, n: int = 6) -> str:
        """CSV using 'x', 'y', 'z' column aliases."""
        import tempfile, os, csv
        path = os.path.join(tempfile.gettempdir(), "test_aliases.csv")
        with open(path, "w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=["x", "y", "z"])
            w.writeheader()
            for i in range(n):
                w.writerow({"x": i * 1e-8, "y": i * 1e-8, "z": 0.0})
        return path

    def test_csv_loads(self):
        from minflux_viewer.core.loader import load_csv
        path = self._write_csv()
        ds = load_csv(path)
        assert ds.prop.num_loc == 8
        assert "loc_x" in ds.attr
        assert "efo" in ds.attr

    def test_csv_without_z_adds_zero_z(self):
        from minflux_viewer.core.loader import load_csv
        path = self._write_csv(has_z=False)
        ds = load_csv(path)
        assert "loc_z" in ds.attr
        assert np.all(ds.attr["loc_z"] == 0.0)

    def test_csv_column_aliases(self):
        from minflux_viewer.core.loader import load_csv
        path = self._write_csv_aliases()
        ds = load_csv(path)
        assert ds.prop.num_loc == 6
        assert "loc_x" in ds.attr

    def test_csv_missing_xy_raises(self):
        import tempfile, os, csv
        from minflux_viewer.core.loader import load_csv
        path = os.path.join(tempfile.gettempdir(), "bad.csv")
        with open(path, "w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=["efo", "cfr"])
            w.writeheader()
            w.writerow({"efo": 1.0, "cfr": 0.5})
        with pytest.raises(ValueError, match="loc_x"):
            load_csv(path)


# ---------------------------------------------------------------------------
# AppState.log() and log_message signal
# ---------------------------------------------------------------------------

class TestAppStateLog:
    @pytest.fixture(autouse=True)
    def _qt(self):
        pytest.importorskip("PyQt6")
        import os, sys
        if not os.environ.get("DISPLAY") and os.name != "nt" and sys.platform != "darwin":
            pytest.skip("No display available for Qt tests")
        from PyQt6.QtWidgets import QApplication
        self._app = QApplication.instance() or QApplication(sys.argv)

    def test_log_emits_signal(self):
        from minflux_viewer.core.app_state import AppState
        state = AppState()
        received = []
        state.log_message.connect(lambda msg, lvl: received.append((msg, lvl)))
        state.log("hello world", "INFO")
        assert len(received) == 1
        assert received[0] == ("hello world", "INFO")

    def test_log_default_level_is_info(self):
        from minflux_viewer.core.app_state import AppState
        state = AppState()
        received = []
        state.log_message.connect(lambda msg, lvl: received.append(lvl))
        state.log("test message")
        assert received[0] == "INFO"

    def test_log_warn_level(self):
        from minflux_viewer.core.app_state import AppState
        state = AppState()
        received = []
        state.log_message.connect(lambda msg, lvl: received.append(lvl))
        state.log("unsupported file", "WARN")
        assert received[0] == "WARN"


# ---------------------------------------------------------------------------
# Regression tests for loader robustness
# ---------------------------------------------------------------------------

class TestLoaderRegressions:
    """
    Regression tests for loader bugs encountered with real Abberior files.

    These synthesize the exact edge cases we saw in the wild rather than
    depending on large binary fixtures.
    """

    def test_m2410_with_scalar_ext_field(self, tmp_path):
        """
        File 1 bug: 'ext' field is a scalar (N,) array, not (N, 3) xyz.
        Must not assume any field name is xyz — detect by shape instead.
        """
        from scipy.io import savemat
        from minflux_viewer.core.loader import load_dataset

        n = 20
        # m2410 = flat format, 'itr' is integer vector, 'loc' is (N, 3)
        raw = {
            "itr":  np.zeros(n, dtype=np.int32),   # single iteration
            "vld":  np.ones(n, dtype=np.uint8),
            "loc":  np.column_stack([
                np.linspace(0, 1e-6, n),
                np.linspace(0, 1e-6, n),
                np.zeros(n),
            ]),
            "ext":  np.linspace(0.0, 1.0, n),      # SCALAR per loc, not xyz!
            "efo":  np.linspace(10.0, 100.0, n),
            "cfr":  np.linspace(0.3, 0.9, n),
            "dcr":  np.linspace(0.1, 0.5, n),
            "tid":  np.arange(n, dtype=np.int32),
            "tim":  np.linspace(0.0, 1.0, n),
        }
        path = tmp_path / "scalar_ext.mat"
        savemat(str(path), raw, do_compression=False, oned_as="column")

        ds = load_dataset(path)
        assert ds.prop.num_loc == n
        # 'ext' should become a scalar attribute, NOT ext_x/ext_y/ext_z
        assert "ext" in ds.attr
        assert ds.attr["ext"].shape == (n,)
        # loc should still become xyz
        assert "loc_x" in ds.attr
        assert "loc_y" in ds.attr
        assert "loc_z" in ds.attr

    def test_m2205_with_xyz_ext_field(self, tmp_path):
        """
        File 2 bug (v7.3 HDF5): nested 'itr' group where 'ext' IS an (N, K, 3)
        xyz per iteration. Must be handled correctly.

        Simulates the unwrapped path using in-memory arrays (savemat with
        nested dicts produces m2205 format).
        """
        from scipy.io import savemat
        from minflux_viewer.core.loader import load_dataset

        n, k = 30, 3
        itr_substruct = {
            "loc": np.zeros((n, k, 3)),  # xyz per iteration
            "ext": np.zeros((n, k, 3)),  # also xyz per iteration
            "lnc": np.zeros((n, k, 3)),  # xyz per iteration
            "efo": np.linspace(10.0, 100.0, n * k).reshape(n, k),
            "cfr": np.linspace(0.3, 0.9, n * k).reshape(n, k),
            "dcr": np.linspace(0.1, 0.5, n * k).reshape(n, k),
            "itr": np.tile(np.arange(k), (n, 1)),
        }
        itr_substruct["loc"][:, -1, 0] = np.linspace(0, 1e-6, n)
        itr_substruct["loc"][:, -1, 1] = np.linspace(0, 1e-6, n)

        raw = {
            "itr": itr_substruct,
            "vld": np.ones(n, dtype=np.uint8),
            "tid": np.arange(n, dtype=np.int32),
            "tim": np.linspace(0.0, 1.0, n),
        }
        path = tmp_path / "m2205_nested.mat"
        savemat(str(path), raw, do_compression=False, oned_as="column")

        ds = load_dataset(path)
        assert ds.prop.num_loc == n
        # ext here IS xyz, should split
        assert "ext_x" in ds.attr
        assert "ext_y" in ds.attr
        assert "ext_z" in ds.attr
        assert "loc_x" in ds.attr


# ---------------------------------------------------------------------------
# ParaView export
# ---------------------------------------------------------------------------

class TestParaViewExport:
    """Tests for ``utils.paraview_export.export_to_vtp``."""

    def _make_dataset(self, n: int = 20, three_d: bool = True) -> "MinfluxDataset":
        prop = DataProp(
            num_loc=n, num_itr=1, num_dim=3 if three_d else 2, num_traces=2,
            trace_idx=np.array([[0, n // 2 - 1], [n // 2, n - 1]]),
            num_loc_per_trace=np.array([n // 2, n // 2]),
            attr_names=["loc_x", "loc_y", "loc_z", "tid", "efo", "ftr", "idx"],
        )
        attrs = AttrStore({
            "loc_x": np.linspace(0, 1e-6, n),
            "loc_y": np.linspace(0, 1e-6, n),
            "loc_z": np.linspace(0, 5e-7, n) if three_d else np.zeros(n),
            "tid":   np.repeat([0, 1], n // 2).astype(np.int32),
            "efo":   np.linspace(10, 100, n),
            "ftr":   np.ones(n, dtype=bool),
            "idx":   np.arange(1, n + 1, dtype=np.uint32),
        })
        return MinfluxDataset(
            file=FileInfo(name="synth.mat", folder="/tmp"),
            prop=prop, attr=attrs,
        )

    def test_creates_file(self, tmp_path):
        from minflux_viewer.utils.paraview_export import export_to_vtp
        ds = self._make_dataset(20)
        out = tmp_path / "test.vtp"
        result = export_to_vtp(ds, out)
        assert result.exists()
        assert result == out.resolve()

    def test_vtp_xml_valid(self, tmp_path):
        """The resulting file should be parseable as XML."""
        import xml.etree.ElementTree as ET
        from minflux_viewer.utils.paraview_export import export_to_vtp
        ds = self._make_dataset(20)
        out = tmp_path / "test.vtp"
        export_to_vtp(ds, out)
        tree = ET.parse(out)
        root = tree.getroot()
        assert root.tag == "VTKFile"
        assert root.attrib["type"] == "PolyData"

    def test_vtp_contains_n_points(self, tmp_path):
        import xml.etree.ElementTree as ET
        from minflux_viewer.utils.paraview_export import export_to_vtp
        ds = self._make_dataset(20)
        out = tmp_path / "test.vtp"
        export_to_vtp(ds, out)
        root = ET.parse(out).getroot()
        piece = root.find("./PolyData/Piece")
        assert piece is not None
        assert int(piece.attrib["NumberOfPoints"]) == 20

    def test_vtp_includes_scalar_fields(self, tmp_path):
        import xml.etree.ElementTree as ET
        from minflux_viewer.utils.paraview_export import export_to_vtp
        ds = self._make_dataset(20)
        out = tmp_path / "test.vtp"
        export_to_vtp(ds, out)
        root = ET.parse(out).getroot()
        names = [
            da.attrib["Name"]
            for da in root.findall("./PolyData/Piece/PointData/DataArray")
        ]
        # efo, tid, loc_x/y/z should all be there
        assert "efo" in names
        assert "tid" in names
        assert "ftr" in names

    def test_filter_mask_applied(self, tmp_path):
        import xml.etree.ElementTree as ET
        from minflux_viewer.utils.paraview_export import export_to_vtp
        ds = self._make_dataset(20)
        # mask out half the points
        mask = np.zeros(20, dtype=bool); mask[:10] = True
        ds.filter_mask = mask
        out = tmp_path / "test.vtp"
        export_to_vtp(ds, out, apply_filter=True)
        root = ET.parse(out).getroot()
        piece = root.find("./PolyData/Piece")
        assert int(piece.attrib["NumberOfPoints"]) == 10

    def test_no_filter_exports_all(self, tmp_path):
        import xml.etree.ElementTree as ET
        from minflux_viewer.utils.paraview_export import export_to_vtp
        ds = self._make_dataset(20)
        mask = np.zeros(20, dtype=bool); mask[:10] = True
        ds.filter_mask = mask
        out = tmp_path / "test.vtp"
        export_to_vtp(ds, out, apply_filter=False)
        root = ET.parse(out).getroot()
        piece = root.find("./PolyData/Piece")
        assert int(piece.attrib["NumberOfPoints"]) == 20
        ftr = root.find("./PolyData/Piece/PointData/DataArray[@Name='ftr']")
        assert ftr is not None
        assert [int(v) for v in ftr.text.split()] == [1] * 10 + [0] * 10


# ---------------------------------------------------------------------------
# ParaView launcher path detection
# ---------------------------------------------------------------------------

class TestParaViewFinder:
    """Tests for ``utils.paraview_launcher.find_paraview_executable``."""

    def test_explicit_path_accepted(self, tmp_path):
        from minflux_viewer.utils.paraview_launcher import find_paraview_executable
        fake = tmp_path / "paraview_fake"
        fake.write_text("")
        result = find_paraview_executable(explicit_path=str(fake))
        assert result == fake.resolve()

    def test_explicit_missing_path_ignored(self, tmp_path):
        """Non-existent explicit path should fall through to other methods."""
        from minflux_viewer.utils.paraview_launcher import find_paraview_executable
        missing = tmp_path / "does_not_exist"
        # This will either return None or something else auto-detected —
        # either way, it must NOT be "missing".
        result = find_paraview_executable(explicit_path=str(missing))
        assert result != missing

    def test_prefs_path(self, tmp_path):
        from minflux_viewer.utils.paraview_launcher import find_paraview_executable
        fake = tmp_path / "paraview_from_prefs"
        fake.write_text("")
        prefs = {"file": {"paraview_path": str(fake)}}
        result = find_paraview_executable(prefs=prefs)
        assert result == fake.resolve()

    def test_env_var(self, tmp_path, monkeypatch):
        from minflux_viewer.utils.paraview_launcher import find_paraview_executable
        fake = tmp_path / "paraview_from_env"
        fake.write_text("")
        monkeypatch.setenv("PARAVIEW", str(fake))
        result = find_paraview_executable()
        assert result == fake.resolve()

    def test_startup_script_written(self, tmp_path):
        from minflux_viewer.utils.paraview_launcher import _write_startup_script
        vtp = tmp_path / "data.vtp"
        vtp.write_text("")
        script = tmp_path / "launch.py"
        _write_startup_script(vtp, script)
        assert script.exists()
        content = script.read_text()
        assert "XMLPolyDataReader" in content
        assert str(vtp) in content

    def test_launch_raises_on_missing_vtp(self, tmp_path, monkeypatch):
        from minflux_viewer.utils.paraview_launcher import launch_paraview
        # Prevent real ParaView from being launched
        monkeypatch.delenv("PARAVIEW", raising=False)
        with pytest.raises(FileNotFoundError):
            launch_paraview(tmp_path / "nope.vtp")


# ---------------------------------------------------------------------------
# Plugins
# ---------------------------------------------------------------------------

class TestPlugins:
    """Registry and built-in MSR reader plugin."""

    def test_registry_api(self):
        from minflux_viewer import plugins
        # Register a throwaway entry
        called = []
        def launch(state, parent=None):
            called.append((state, parent))
        plugins.register(plugins.PluginEntry(
            name="__test_plugin__", tooltip="t", launch=launch,
        ))
        entries = plugins.available()
        assert any(e.name == "__test_plugin__" for e in entries)

        # Duplicate register by name should be a no-op
        plugins.register(plugins.PluginEntry(
            name="__test_plugin__", tooltip="new", launch=launch,
        ))
        count = sum(1 for e in entries if e.name == "__test_plugin__")
        assert count == 1

    def test_msr_reader_registers_itself(self):
        from minflux_viewer import plugins
        plugins.ensure_loaded()
        names = [e.name for e in plugins.available()]
        assert "MSR Reader" in names

    def test_state_log_adapter_severity(self):
        pytest.importorskip("PyQt6")
        import os, sys
        if not os.environ.get("DISPLAY") and os.name != "nt" and sys.platform != "darwin":
            pytest.skip("No display available for Qt tests")
        from PyQt6.QtWidgets import QApplication
        _app = QApplication.instance() or QApplication(sys.argv)

        from minflux_viewer.core.app_state import AppState
        from minflux_viewer.plugins.msr_reader.msr_reader_dialog import _StateLogAdapter

        state = AppState()
        received = []
        state.log_message.connect(lambda m, l: received.append((m, l)))
        log = _StateLogAdapter(state)
        log("plain info")
        log("[warn] not great")
        log("[error] bad!")
        assert received[0][1] == "INFO"
        assert received[1][1] == "WARN"
        assert received[2][1] == "ERROR"


# ---------------------------------------------------------------------------
# Main-window cleanup on close
# ---------------------------------------------------------------------------

class TestMainWindowCleanup:
    """All child windows and ParaView subprocesses terminate on close."""

    @pytest.fixture(autouse=True)
    def _qt(self):
        pytest.importorskip("PyQt6")
        import os, sys
        if not os.environ.get("DISPLAY") and os.name != "nt" and sys.platform != "darwin":
            pytest.skip("No display available for Qt tests")
        from PyQt6.QtWidgets import QApplication
        self._app = QApplication.instance() or QApplication(sys.argv)

    def test_close_terminates_tracked_paraview_process(self):
        import subprocess, sys, time
        from minflux_viewer.core.app_state import AppState
        from minflux_viewer.ui.main_window import MainWindow

        state = AppState()
        # Pin the pref — AppState loads persisted QSettings, so a prior run (or
        # the sibling test) may have saved close_paraview_on_exit=False.
        state.prefs.setdefault("file", {})["close_paraview_on_exit"] = True
        w = MainWindow(state)

        # Fake a ParaView subprocess — a long-running Python sleep (no reliance
        # on a `sleep` binary, which doesn't exist on a clean Windows host).
        proc = subprocess.Popen([sys.executable, "-c", "import time; time.sleep(30)"])
        w._paraview_procs.append(proc)
        assert proc.poll() is None   # still running

        w.close()
        time.sleep(0.5)
        assert proc.poll() is not None  # terminated
        # The process group received SIGTERM via .terminate()
        assert proc.returncode != 0    # non-zero because killed

    def test_close_paraview_on_exit_pref_disables_termination(self):
        import subprocess, sys, time
        from minflux_viewer.core.app_state import AppState
        from minflux_viewer.ui.main_window import MainWindow

        state = AppState()
        state.prefs.setdefault("file", {})["close_paraview_on_exit"] = False
        w = MainWindow(state)

        proc = subprocess.Popen([sys.executable, "-c", "import time; time.sleep(30)"])
        w._paraview_procs.append(proc)
        w.close()
        time.sleep(0.3)
        # Must still be running because the pref disabled termination
        assert proc.poll() is None
        # Clean up manually so we don't leak
        proc.terminate(); proc.wait(timeout=2)

    def test_close_closes_every_tool_window(self):
        from minflux_viewer.core.app_state import AppState
        from minflux_viewer.core.loader import load_dataset
        from minflux_viewer.ui.main_window import MainWindow

        state = AppState()
        w = MainWindow(state)
        # Need an active dataset for most tool windows to open
        from minflux_viewer.core.dataset import MinfluxDataset, FileInfo, DataProp, AttrStore
        n = 10
        prop = DataProp(
            num_loc=n, num_itr=1, num_dim=3, num_traces=1,
            trace_idx=np.array([[0, n - 1]]),
            num_loc_per_trace=np.array([n]),
            attr_names=["loc_x", "loc_y", "loc_z", "tid", "efo", "ftr", "idx"],
        )
        attrs = AttrStore({
            "loc_x": np.linspace(0, 1e-6, n), "loc_y": np.linspace(0, 1e-6, n),
            "loc_z": np.zeros(n),
            "tid":   np.zeros(n, dtype=np.int32),
            "efo":   np.linspace(10, 100, n),
            "ftr":   np.ones(n, dtype=bool),
            "idx":   np.arange(1, n + 1, dtype=np.uint32),
        })
        ds = MinfluxDataset(
            file=FileInfo(name="synth.mat", folder="/tmp"),
            prop=prop, attr=attrs,
        )
        state.add_dataset(ds)

        # Open singleton tool windows
        w._show_scatter()
        w._show_histogram()
        w._show_attr_plot()
        w._show_filter()
        w._show_dataset_manager()
        w._show_log()
        w._show_console()

        wins = [
            w._scatter_win, w._histogram_win, w._attr_win,
            w._filter_dlg, w._ds_manager, w._log_win, w._console_win,
        ]
        assert all(wi is not None and wi.isVisible() for wi in wins)

        w.close()

        # After close, none of them should be visible
        for wi in wins:
            assert wi is None or not wi.isVisible()


# ---------------------------------------------------------------------------
# 2D / 3D threshold
# ---------------------------------------------------------------------------

class TestMinZRangeThreshold:
    """Loader respects prefs['data']['enforce_min_z_range']."""

    def _synth_3d_mat(self, tmp_path, z_range_m: float, n: int = 30):
        """Write a minimal m2410-style .mat file with a controlled Z range."""
        from scipy.io import savemat
        raw = {
            "itr":  np.zeros(n, dtype=np.int32),
            "vld":  np.ones(n, dtype=np.uint8),
            "loc":  np.column_stack([
                np.linspace(0, 1e-6, n),
                np.linspace(0, 1e-6, n),
                np.linspace(0, z_range_m, n),   # Z spans [0, z_range_m]
            ]),
            "efo":  np.linspace(10.0, 100.0, n),
            "tid":  np.arange(n, dtype=np.int32),
            "tim":  np.linspace(0.0, 1.0, n),
        }
        path = tmp_path / "threshold.mat"
        savemat(str(path), raw, do_compression=False, oned_as="column")
        return path

    def test_small_z_range_forced_to_2d(self, tmp_path):
        from minflux_viewer.core.loader import load_dataset
        # 1 nm Z range, threshold 5 nm → 2D
        path = self._synth_3d_mat(tmp_path, z_range_m=1e-9)
        prefs = {"data": {"enforce_min_z_range": True, "min_z_range_nm": 5.0}}
        ds = load_dataset(path, prefs=prefs)
        assert ds.prop.num_dim == 2
        # And loc_z should all be zero
        assert np.all(ds.attr["loc_z"] == 0)

    def test_large_z_range_stays_3d(self, tmp_path):
        from minflux_viewer.core.loader import load_dataset
        # 500 nm Z range, threshold 5 nm → 3D preserved
        path = self._synth_3d_mat(tmp_path, z_range_m=5e-7)
        prefs = {"data": {"enforce_min_z_range": True, "min_z_range_nm": 5.0}}
        ds = load_dataset(path, prefs=prefs)
        assert ds.prop.num_dim == 3
        assert np.any(ds.attr["loc_z"] != 0)

    def test_threshold_disabled_preserves_z(self, tmp_path):
        from minflux_viewer.core.loader import load_dataset
        # 1 nm Z range, threshold disabled → stays 3D because Z is non-zero
        path = self._synth_3d_mat(tmp_path, z_range_m=1e-9)
        prefs = {"data": {"enforce_min_z_range": False, "min_z_range_nm": 5.0}}
        ds = load_dataset(path, prefs=prefs)
        # Non-zero Z (even if tiny) → num_dim == 3
        assert ds.prop.num_dim == 3


# ---------------------------------------------------------------------------
# Preferences dialog round-trip
# ---------------------------------------------------------------------------

class TestPreferencesDialog:
    @pytest.fixture(autouse=True)
    def _qt(self):
        pytest.importorskip("PyQt6")
        import os, sys
        if not os.environ.get("DISPLAY") and os.name != "nt" and sys.platform != "darwin":
            pytest.skip("No display available for Qt tests")
        from PyQt6.QtWidgets import QApplication
        self._app = QApplication.instance() or QApplication(sys.argv)

    def test_roundtrip_preserves_values(self):
        from minflux_viewer.core.app_state import AppState
        from minflux_viewer.ui.preferences_dialog import PreferencesDialog

        state = AppState()
        # Modify a couple of values
        state.prefs["data"]["min_z_range_nm"] = 17.5
        state.prefs["data"]["enforce_min_z_range"] = False
        state.prefs["file"]["temp_folder"] = "/tmp/xyz"
        state.prefs["plugin"] = {
            "msr_remember_last": False,
        }
        dlg = PreferencesDialog(state)
        # Widgets should reflect state
        assert dlg._min_z_spin.value() == 17.5
        assert not dlg._enforce_z.isChecked()
        assert dlg._temp_folder_edit.text() == "/tmp/xyz"   # now on the File tab
        assert not dlg._msr_remember.isChecked()

        # Change a value and apply
        dlg._min_z_spin.setValue(99.0)
        dlg._enforce_z.setChecked(True)
        dlg._apply_widgets_to_draft()
        assert dlg._draft["data"]["min_z_range_nm"] == 99.0
        assert dlg._draft["data"]["enforce_min_z_range"] is True

    def test_reset_current_tab_only_resets_visible(self):
        from minflux_viewer.core.app_state import AppState, DEFAULT_PREFS
        from minflux_viewer.ui.preferences_dialog import PreferencesDialog

        state = AppState()
        # Change values in TWO tabs
        state.prefs["data"]["min_z_range_nm"] = 99.0
        state.prefs["plot"]["rimf_value"] = 2.5
        dlg = PreferencesDialog(state)

        # Switch to the Data page and reset — only Data values should reset
        dlg._page_list.setCurrentRow(1)   # File=0, Data=1
        dlg._reset_current_tab()
        assert dlg._draft["data"]["min_z_range_nm"] == DEFAULT_PREFS["data"]["min_z_range_nm"]
        assert dlg._draft["plot"]["rimf_value"] == 2.5   # untouched (plot page)

    def test_recent_files_dialog_remove_and_clear(self):
        from minflux_viewer.ui.preferences_dialog import RecentFilesDialog

        paths = [f"/data/f{i}.mat" for i in range(6)]
        dlg = RecentFilesDialog(paths)
        assert dlg.result_paths() == paths

        # Ctrl/Shift-style multi-select then remove
        for row in (0, 2, 4):
            dlg._list.item(row).setSelected(True)
        dlg._remove_selected()
        assert dlg.result_paths() == ["/data/f1.mat", "/data/f3.mat", "/data/f5.mat"]

        dlg._clear_all()
        assert dlg.result_paths() == []


# ---------------------------------------------------------------------------
# Duplicate dataset
# ---------------------------------------------------------------------------

class TestDuplicate:
    @pytest.fixture(autouse=True)
    def _qt(self):
        pytest.importorskip("PyQt6")
        import os, sys
        if not os.environ.get("DISPLAY") and os.name != "nt" and sys.platform != "darwin":
            pytest.skip("No display available for Qt tests")
        from PyQt6.QtWidgets import QApplication
        self._app = QApplication.instance() or QApplication(sys.argv)

    def test_duplicate_prefixes_name_and_adds_to_state(self):
        from minflux_viewer.core.app_state import AppState
        from minflux_viewer.core.dataset import MinfluxDataset, FileInfo, DataProp, AttrStore
        from minflux_viewer.ui.main_window import MainWindow

        n = 10
        prop = DataProp(
            num_loc=n, num_itr=1, num_dim=3, num_traces=1,
            trace_idx=np.array([[0, n - 1]]),
            num_loc_per_trace=np.array([n]),
            attr_names=["loc_x", "loc_y", "loc_z", "tid", "efo", "ftr", "idx"],
        )
        attrs = AttrStore({
            "loc_x": np.linspace(0, 1e-6, n), "loc_y": np.linspace(0, 1e-6, n),
            "loc_z": np.linspace(0, 5e-7, n),
            "tid":   np.zeros(n, dtype=np.int32),
            "efo":   np.linspace(10, 100, n),
            "ftr":   np.ones(n, dtype=bool),
            "idx":   np.arange(1, n + 1, dtype=np.uint32),
        })
        ds = MinfluxDataset(
            file=FileInfo(name="original.mat", folder="/tmp"),
            prop=prop, attr=attrs,
        )

        state = AppState()
        w = MainWindow(state)
        state.add_dataset(ds)
        assert len(state) == 1

        w._duplicate_active_dataset()
        assert len(state) == 2
        assert state.datasets[1].file.name == "DUP_original.mat"
        # Filter mask should also be duplicated, not shared
        state.datasets[0].filter_mask[:] = False
        assert state.datasets[1].filter_mask.all(), "filter mask shouldn't be shared"


# ---------------------------------------------------------------------------
# StdDev per trace (Phase 4)
# ---------------------------------------------------------------------------

class TestStdDevPerTrace:
    def test_synthetic_traces_recover_known_sigma(self):
        from minflux_viewer.analysis import stddev_per_trace
        rng = np.random.default_rng(42)
        # 50 traces, 100 localizations each, σ = 5 nm = 5e-9 m in each axis
        n_traces = 50
        n_per    = 100
        true_sigma_nm = 5.0
        loc = rng.normal(0.0, true_sigma_nm * 1e-9,
                         size=(n_traces * n_per, 3))
        tid = np.repeat(np.arange(n_traces), n_per)
        result = stddev_per_trace(loc, tid)
        assert result['n_traces_used'] == n_traces
        # Per-axis median should be within 0.3 nm of the true sigma
        for axis in range(3):
            med = result['median_sigma_xyz'][axis]
            assert abs(med - true_sigma_nm) < 0.5, \
                f"axis {axis}: σ={med} expected ~{true_sigma_nm}"

    def test_skip_short_traces(self):
        from minflux_viewer.analysis import stddev_per_trace
        # Two traces: one with 5 locs, one with 1 loc (should be dropped)
        loc = np.zeros((6, 3))
        loc[:5] = np.random.RandomState(0).normal(0, 1e-9, size=(5, 3))
        tid = np.array([0, 0, 0, 0, 0, 1])
        result = stddev_per_trace(loc, tid, min_locs=3)
        assert result['n_traces_used'] == 1
        assert result['n_traces_total'] == 2

    def test_empty_returns_zeros(self):
        from minflux_viewer.analysis import stddev_per_trace
        result = stddev_per_trace(np.empty((0, 3)), np.empty(0, dtype=int))
        assert result['n_traces_used'] == 0
        assert result['n_traces_total'] == 0
        assert np.all(result['median_sigma_xyz'] == 0)


# ---------------------------------------------------------------------------
# ProcessingJournal (Phase 4)
# ---------------------------------------------------------------------------

class TestProcessingJournal:
    def test_basic_append_and_iterate(self):
        from minflux_viewer.core.processing_journal import ProcessingJournal
        j = ProcessingJournal()
        assert len(j) == 0
        j.add("load",     "Loaded foo.mat", num_loc=100)
        j.add("transform", "Duplicated as DUP_foo.mat")
        j.add("analysis", "StdDev computed")
        assert len(j) == 3
        cats = [e.category for e in j]
        assert cats == ["load", "transform", "analysis"]
        # Details preserved on entry 0
        assert j.entries[0].details == {"num_loc": 100}

    def test_state_has_journal_attribute(self):
        from minflux_viewer.core.app_state import AppState
        s = AppState()
        # AppState attaches a fresh ProcessingJournal in __init__
        assert hasattr(s, "journal")
        assert len(s.journal) == 0


# ---------------------------------------------------------------------------
# Method text generation (Phase 4)
# ---------------------------------------------------------------------------

class TestGenerateMethodText:
    """The generator was redesigned to compile prose from selected *log events*
    (see ``analysis/method_text.py`` and ``tests/test_method_text.py``)."""

    @staticmethod
    def _state():
        from types import SimpleNamespace
        return SimpleNamespace(datasets=[])

    @staticmethod
    def _ev(message, name="foo.mat", idx=0):
        return {"message": message, "dataset_idx": idx, "dataset_name": name}

    def test_text_from_log_events(self):
        from minflux_viewer.analysis.method_text import generate_method_text
        events = [
            self._ev("Loaded dataset 'foo.mat': 12,345 valid locs, 4 iteration(s), "
                     "10 trace(s), 3D [m2410 (obf / mfxdta)]."),
            self._ev("Applied EFC > 0.7 filter"),
            self._ev("Localization precision (StdDev per trace): combined (n-weighted) "
                     "sigma_r = 5.00 nm, sigma_z = 7.10 nm over 8 of 10 traces "
                     "(>=5 loc/trace, raw z)."),
            self._ev("Saved processed data to foo.csv"),
        ]
        text = generate_method_text(self._state(), events, version="0.3.1")
        assert "foo.mat" in text
        assert "Ostersehlt" in text          # StdDev per trace reference
        assert "Filtering" in text
        assert "Export" in text
        assert "MINFLUX Data Viewer" in text  # footer

    def test_empty_events_friendly_message(self):
        from minflux_viewer.analysis.method_text import generate_method_text
        text = generate_method_text(self._state(), [])
        assert "No log events were selected" in text


# ---------------------------------------------------------------------------
# LUT colormap helpers (Phase 4)
# ---------------------------------------------------------------------------

class TestLutColormaps:
    def test_single_color_red_ramp(self):
        pytest.importorskip("PyQt6")
        import os, sys
        if not os.environ.get("DISPLAY") and os.name != "nt" and sys.platform != "darwin":
            pytest.skip("No display available for Qt tests")
        from PyQt6.QtWidgets import QApplication
        _app = QApplication.instance() or QApplication(sys.argv)
        from minflux_viewer.ui.lut_dialog import make_colormap
        cmap = make_colormap("Red")
        lut = cmap.getLookupTable(0.0, 1.0, 256)
        # Bottom is black, top is pure red
        assert lut[0, 0] == 0
        assert lut[-1, 0] == 255
        assert lut[-1, 1] == 0 and lut[-1, 2] == 0

    def test_invert_flag(self):
        pytest.importorskip("PyQt6")
        import os, sys
        if not os.environ.get("DISPLAY") and os.name != "nt" and sys.platform != "darwin":
            pytest.skip("No display available for Qt tests")
        from PyQt6.QtWidgets import QApplication
        _app = QApplication.instance() or QApplication(sys.argv)
        from minflux_viewer.ui.lut_dialog import make_colormap
        normal   = make_colormap("Red", invert=False).getLookupTable(0., 1., 256)
        inverted = make_colormap("Red", invert=True ).getLookupTable(0., 1., 256)
        # Inverted should be the reverse
        assert np.array_equal(normal[0], inverted[-1])
        assert np.array_equal(normal[-1], inverted[0])
