"""Round-trip fidelity: processing state must survive save -> reload, per format.

These pin the defects found by the 2026-08 save/export audit (F1-F4, F7, F8).
Every one of them shipped while the existing 61 save/export tests passed, because
no test round-tripped a dataset that had actually been *processed*.
"""

import numpy as np
import pytest

from minflux_viewer.core import loader as L
from minflux_viewer.core import overlay as OV
from minflux_viewer.core import roi_crop as RC
from minflux_viewer.core.save import dataset_to_mfx_array, save_processed

# formats that can be written and read back within the app
ROUNDTRIP = [
    ("mat", L.load_dataset), ("npy", L.load_npy), ("npz", L.load_npz),
    ("json", L.load_json), ("csv", L.load_csv), ("zarr", L.load_zarr),
]


def _raw_mfx(n_loc=50, n_itr=4, seed=0):
    rng = np.random.default_rng(seed)
    n = n_loc * n_itr
    dt = np.dtype([
        ("vld", np.bool_), ("tid", np.int32), ("tim", np.float64), ("itr", np.int32),
        ("efo", np.float64), ("cfr", np.float64), ("eco", np.int32),
        ("dcr", np.float64, (2,)), ("loc", np.float64, (3,)),
    ])
    a = np.zeros(n, dt)
    a["vld"] = True
    a["tid"] = np.repeat(np.arange(n_loc), n_itr) // 3
    a["tim"] = np.repeat(np.linspace(0, 10, n_loc), n_itr)
    a["itr"] = np.tile(np.arange(n_itr), n_loc)
    a["efo"] = rng.uniform(10, 100, n)
    a["cfr"] = rng.uniform(0.3, 0.9, n)
    a["eco"] = rng.integers(50, 500, n)
    a["dcr"] = rng.uniform(0, 1, (n, 2))
    a["loc"] = np.column_stack([rng.uniform(0, 2e-6, n), rng.uniform(0, 2e-6, n),
                                rng.uniform(0, 5e-7, n)])
    return a


def _processed(tmp_path, dx=100.0, dy=-50.0):
    """A dataset carrying RIMF, a channel transform and an active filter."""
    ds = L.load_from_mfx_array(_raw_mfx(), name="p.mat", folder=str(tmp_path))
    ds.set_rimf(0.67, source="manual (test)")
    matrix = np.eye(4)
    matrix[0, 3] = dx
    matrix[1, 3] = dy
    record = OV.display_transform_record(
        overlay_id="g", overlay_index=1, order=1, lut="solid:Green",
        source_dataset_idx=0, alignment_mode="mbm info", matrix_4x4=matrix)
    ds.state["overlay_transform"] = record
    ds.state["render_transform_2d"] = record
    ds.state["filter_specs"] = [{
        "attribute": "efo", "mode": "per loc", "itr": "last",
        "lo": 20.0, "hi": 90.0, "lo_inc": True, "hi_inc": False}]
    L.apply_saved_filters(ds)
    return ds


# --- F1: the transform must come back ---------------------------------------
@pytest.mark.parametrize("fmt,loader", ROUNDTRIP)
def test_raw_roundtrip_preserves_channel_alignment(tmp_path, fmt, loader):
    """A saved overlay transform must survive reload.

    It is written as a ``display_transform_record`` dict; restoring it as a bare
    ndarray makes ``apply_display_transform_nm`` silently ignore it, which moved
    the channel by exactly the alignment (measured: 100 nm) with no error at all.
    """
    ds = _processed(tmp_path)
    before = RC.display_xy_filtered(ds)[0]

    written = save_processed(ds, data_path=tmp_path / f"r.{fmt}", fmt=fmt,
                             content="raw", include={"recipe": True})
    back = loader(written[0])

    restored = back.state.get("overlay_transform")
    assert isinstance(restored, dict), f"{fmt}: transform must restore as a record"
    assert OV.transform_to_matrix4(restored) is not None
    after = RC.display_xy_filtered(back)[0]
    assert after[0] == pytest.approx(before[0], abs=1e-6), f"{fmt}: x displaced"
    assert after[1] == pytest.approx(before[1], abs=1e-6), f"{fmt}: y displaced"


def test_unreadable_transform_is_reported_not_swallowed(tmp_path):
    """An uninterpretable transform must be surfaced, never dropped silently."""
    ds = L.load_from_mfx_array(_raw_mfx(), name="p.mat", folder=str(tmp_path))
    applied = L.apply_metadata_recipe(ds, {"transform": {"nonsense": True}})
    assert any("NOT restored" in phrase for phrase in applied)
    assert "overlay_transform" not in ds.state


# --- F2: raw content must reassemble, not become one loc per raw row --------
@pytest.mark.parametrize("fmt,loader", ROUNDTRIP)
def test_raw_roundtrip_preserves_structure_and_calibration(tmp_path, fmt, loader):
    ds = _processed(tmp_path)
    written = save_processed(ds, data_path=tmp_path / f"s.{fmt}", fmt=fmt,
                             content="raw", include={"recipe": True})
    back = loader(written[0])

    assert back.prop.num_loc == ds.prop.num_loc, f"{fmt}: localization count changed"
    assert len(back.mfx_raw["itr"]) == len(ds.mfx_raw["itr"]), f"{fmt}: raw rows lost"
    assert back.metadata.get("raw_num_itr") == ds.metadata.get("raw_num_itr")
    assert float(back.cali.RIMF) == pytest.approx(0.67), f"{fmt}: RIMF lost"
    assert len(back.state.get("filter_specs") or []) == 1, f"{fmt}: filter lost"
    assert int(np.asarray(back.filter_mask).sum()) == int(np.asarray(ds.filter_mask).sum())


# --- F3: every offered format must be reopenable ----------------------------
@pytest.mark.parametrize("fmt,loader", ROUNDTRIP)
@pytest.mark.parametrize("content", ["raw", "snapshot"])
def test_every_written_format_can_be_reopened(tmp_path, fmt, loader, content):
    """Writing a file the application cannot read back is a silent dead end."""
    ds = _processed(tmp_path)
    written = save_processed(ds, data_path=tmp_path / f"{content}.{fmt}", fmt=fmt,
                             content=content, include={"recipe": True})
    back = loader(written[0])
    assert back.prop.num_loc == ds.prop.num_loc


def test_npz_is_routed_to_its_own_loader():
    """`.npz` is a zip, and used to sniff as .xlsx -> the spreadsheet importer."""
    from minflux_viewer.core.format_sniff import EXT_TO_FMT
    from minflux_viewer.ui.main_window import _FMT_LOADERS, _SUPPORTED_EXTS

    assert ".npz" in _SUPPORTED_EXTS
    assert EXT_TO_FMT[".npz"] == "npz"
    assert _FMT_LOADERS["npz"] == "_load_npz"


def test_npz_content_sniff_distinguishes_it_from_xlsx(tmp_path):
    from minflux_viewer.core.format_sniff import sniff_format

    path = tmp_path / "bundle.npz"
    np.savez(path, loc_x=np.arange(4.0), loc_y=np.arange(4.0))
    assert sniff_format(path) == "npz"


# --- F8: the flagged filter state must come back ----------------------------
@pytest.mark.parametrize("fmt,loader", ROUNDTRIP)
def test_snapshot_flag_mode_restores_the_filter(tmp_path, fmt, loader):
    """``filter_mode="flag"`` keeps every row and records an ``ftr`` column.

    ``_add_derived_attributes`` resets ``attrs["ftr"]`` to all-True, so a reader
    that picks it up afterwards silently reports "nothing filtered".
    """
    ds = _processed(tmp_path)
    kept = int(np.asarray(ds.filter_mask).sum())
    assert 0 < kept < ds.prop.num_loc, "fixture must actually filter something"

    written = save_processed(ds, data_path=tmp_path / f"f.{fmt}", fmt=fmt,
                             content="snapshot", filter_mode="flag",
                             include={"recipe": True})
    back = loader(written[0])
    assert int(np.asarray(back.filter_mask).sum()) == kept, f"{fmt}: ftr lost"


# --- F7: internal cache columns must not be exported ------------------------
def test_internal_loc_id_cache_is_never_exported(tmp_path):
    """``_raw_loc_id`` caches ``loc_id`` into mfx_raw the first time rows are grouped.

    Exporting it made the file's columns depend on what the user clicked first.
    """
    ds = _processed(tmp_path)                 # applying the filter populates it
    assert "loc_id" in ds.mfx_raw.keys(), "fixture should have triggered the cache"
    assert "loc_id" not in (dataset_to_mfx_array(ds).dtype.names or ())


def test_raw_csv_records_structure_and_container_separately(tmp_path):
    """``source_version`` is the data structure; ``source_format`` is the container."""
    ds = _processed(tmp_path)
    written = save_processed(ds, data_path=tmp_path / "v.csv", fmt="csv",
                             content="raw", include={"recipe": False})
    back = L.load_csv(written[0])
    assert back.metadata.get("source_version") == "m2410"
    assert back.metadata.get("source_format") == "csv"
