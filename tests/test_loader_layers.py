"""Loader capabilities have one owner, with stable legacy import names."""

import importlib
import subprocess
import sys

import pytest


@pytest.mark.parametrize("module,symbol", [
    ("format_readers", "load_dataset"), ("format_readers", "load_npy"),
    ("format_readers", "load_npz"), ("format_readers", "load_csv"),
    ("format_readers", "load_json"), ("format_readers", "load_zarr"),
    ("format_readers", "load_tiff"), ("format_readers", "load_from_mfx_array"),
    ("normalization", "_parse_m2410"), ("normalization", "_parse_m2205"),
    ("normalization", "_normalize_mfx_attrs"), ("normalization", "_compute_properties"),
    ("raw_access", "mfx_get"), ("raw_access", "mfx_row_mask"),
    ("raw_access", "attr_values_1d"), ("raw_access", "iteration_attr_values_1d"),
    ("raw_access", "_LOC_ID_PUBLISH_LOCK"),
    ("metadata_recipe", "apply_metadata_recipe"),
    ("metadata_recipe", "apply_metadata_sidecar"),
    ("metadata_recipe", "read_metadata_file"),
])
def test_legacy_loader_exports_the_single_implementation_owner(module, symbol):
    old = importlib.import_module("minflux_viewer.core.loader")
    new = importlib.import_module("minflux_viewer.core.loader_" + module)
    assert getattr(old, symbol) is getattr(new, symbol)


def test_loader_capabilities_import_without_gui_dependencies():
    code = """
import sys
import minflux_viewer.core.loader
import minflux_viewer.core.loader_format_readers
import minflux_viewer.core.loader_normalization
import minflux_viewer.core.loader_raw_access
import minflux_viewer.core.loader_metadata_recipe
assert not any(n.startswith(('PyQt6', 'pyqtgraph', 'minflux_viewer.ui')) for n in sys.modules)
"""
    result = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True)
    assert result.returncode == 0, result.stdout + result.stderr


def test_raw_access_does_not_initialize_format_readers_or_scipy_io():
    code = """
import sys
import minflux_viewer.core.loader_raw_access
assert 'minflux_viewer.core.loader_format_readers' not in sys.modules
assert 'scipy.io' not in sys.modules
"""
    result = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True)
    assert result.returncode == 0, result.stdout + result.stderr


@pytest.mark.parametrize("symbol", [
    "AttrStore", "Calibration", "Channel", "DataProp", "FileInfo", "MinfluxDataset",
    "build_image_dataset", "build_localization_dataset",
])
def test_legacy_loader_preserves_its_reexported_dataset_constructors(symbol):
    old = importlib.import_module("minflux_viewer.core.loader")
    owner = importlib.import_module("minflux_viewer.core.dataset")
    assert getattr(old, symbol) is getattr(owner, symbol)
