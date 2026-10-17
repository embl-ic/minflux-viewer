"""The public CPU I/O contract needs no running viewer or implicit processing."""

import subprocess
import sys
from concurrent.futures import ThreadPoolExecutor

import numpy as np
import pytest

from minflux_viewer.api.io import load, save
from minflux_viewer.core.dataset import build_localization_dataset


def dataset():
    return build_localization_dataset(name="synthetic", x_nm=np.arange(6.),
                                      y_nm=np.arange(6.), tid=np.arange(6, dtype=np.int64),
                                      attrs={"efo": np.arange(6.) + 1000})


@pytest.mark.parametrize("fmt", ["npy", "mat", "json", "csv", "zarr", "zarr_zip"])
def test_single_dataset_roundtrip_and_no_overwrite(tmp_path, fmt):
    ds = dataset()
    paths = save(ds, tmp_path / "result", fmt=fmt)
    restored = load(paths[0], apply_sidecar=True)
    np.testing.assert_array_equal(restored.loc_nm, ds.loc_nm)
    np.testing.assert_array_equal(restored.attr["tid"], ds.attr["tid"])
    with pytest.raises(FileExistsError):
        save(ds, paths[0], fmt=fmt)


def test_sidecar_is_opt_in_and_no_auto_computation(tmp_path):
    ds = dataset()
    ds.set_z_scaling_factor(.75, source="test")
    paths = save(ds, tmp_path / "one.npy")
    assert "z_scaling_factor" not in load(paths[0]).derived
    restored = load(paths[0], apply_sidecar=True)
    assert restored.cali.z_scaling_factor == .75
    assert "den" not in restored.derived
    assert "sigma_per_trace_nm" not in restored.derived


@pytest.mark.parametrize("fmt", ["npy", "mat", "json", "csv"])
def test_snapshot_recipe_provenance_is_restored_for_every_ordinary_reader(tmp_path, fmt):
    ds = dataset()
    ds.metadata["msr_dataset_did"] = "synthetic-acquisition"
    path = save(ds, tmp_path / "snapshot", fmt=fmt, content="snapshot")[0]
    assert load(path, apply_sidecar=True).metadata["msr_dataset_did"] == "synthetic-acquisition"
    assert "msr_dataset_did" not in load(path).metadata


def test_explicit_malformed_recipe_is_an_error_not_silent_absence(tmp_path):
    path = save(dataset(), tmp_path / "source")[0]
    path.with_name("source_viewer_metadata.json").write_text("{}")
    assert load(path).prop.num_loc == 6
    with pytest.raises(ValueError, match="not a metadata sidecar"):
        load(path, apply_sidecar=True)


@pytest.mark.parametrize("raw_store", [True, False])
def test_disabled_saved_filter_is_not_applied_on_either_selection_path(raw_store):
    from minflux_viewer.core.dataset import AttrStore
    from minflux_viewer.core.loader_raw_access import apply_saved_filters

    ds = dataset()
    if not raw_store:
        ds.components.mfx_raw = AttrStore()
    ds.state["filter_specs"] = [{"attribute": "efo", "lo": 1002, "hi": 1005,
                                "mode": "per loc", "enabled": False}]
    apply_saved_filters(ds)
    assert ds.filter_mask.all()
    ds.state["filter_specs"][0]["enabled"] = True
    assert apply_saved_filters(ds)
    np.testing.assert_array_equal(ds.filter_mask, np.arange(6) >= 2)


def test_deferred_recipe_filter_cannot_silently_become_unfiltered_cpu_output(tmp_path):
    ds = dataset()
    ds.state["filter_specs"] = [{"attribute": "den", "lo": 5, "hi": 10, "mode": "per loc"}]
    path = save(ds, tmp_path / "source")[0]
    assert load(path).prop.num_loc == 6
    with pytest.raises(ValueError, match="not materialized"):
        load(path, apply_sidecar=True)


@pytest.mark.parametrize("factor", [0, -1, float("nan"), float("inf")])
def test_invalid_recipe_calibration_is_not_ignored(tmp_path, factor):
    import json

    path = save(dataset(), tmp_path / "source")[0]
    sidecar = path.with_name("source_viewer_metadata.json")
    recipe = json.loads(sidecar.read_text())
    recipe["calibration"]["z_scaling_factor"] = factor
    sidecar.write_text(json.dumps(recipe))
    with pytest.raises(ValueError, match="positive and finite"):
        load(path, apply_sidecar=True)


def test_multi_member_project_and_msr_are_not_silently_truncated(tmp_path):
    from minflux_viewer.core.minflux_zarr import write_minflux_zarr_project

    project = write_minflux_zarr_project([dataset(), dataset()], tmp_path / "project.zarr")
    with pytest.raises(ValueError):
        load(project)
    msr = tmp_path / "example.msr"
    msr.write_bytes(b"not an acquisition")
    with pytest.raises(ValueError, match="selection"):
        load(msr)


def test_colliding_concurrent_saves_have_exactly_one_owner(tmp_path):
    def write(_):
        try:
            return save(dataset(), tmp_path / "one.npy")[0]
        except FileExistsError:
            return None
    with ThreadPoolExecutor(2) as pool:
        results = list(pool.map(write, range(2)))
    assert sum(result is not None for result in results) == 1
    assert load(tmp_path / "one.npy").prop.num_loc == 6


@pytest.mark.parametrize("fmt,content,mode", [("unknown", "raw", "flag"),
                                             ("npy", "typo", "flag"),
                                             ("npy", "raw", "typo"),
                                             ("zarr", "snapshot", "apply")])
def test_invalid_save_contract_writes_nothing(tmp_path, fmt, content, mode):
    with pytest.raises(ValueError):
        save(dataset(), tmp_path / "result", fmt=fmt, content=content, filter_mode=mode)
    assert not list(tmp_path.iterdir())


def test_cpu_api_import_and_roundtrip_do_not_import_qt(tmp_path):
    code = '''
import sys
from pathlib import Path
import numpy as np
from minflux_viewer.api.io import load, save, batch
from minflux_viewer.core.dataset import build_localization_dataset
ds = build_localization_dataset(name="headless", x_nm=np.arange(4.), y_nm=np.arange(4.))
path = save(ds, Path(sys.argv[1]) / "data.npy")[0]
assert load(path).prop.num_loc == 4
assert not any(name.startswith(("PyQt", "PySide")) for name in sys.modules)
'''
    result = subprocess.run([sys.executable, "-c", code, str(tmp_path)], capture_output=True,
                            text=True, timeout=30)
    assert result.returncode == 0, result.stdout + result.stderr
