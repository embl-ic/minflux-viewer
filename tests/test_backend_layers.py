"""Core imports stay independent of the optional MSR backend."""

import importlib
import subprocess
import sys

import pytest

from minflux_viewer.core import formats


def test_core_image_and_save_imports_do_not_initialize_msr_backend():
    code = """
import sys
import minflux_viewer.core.confocal_mapping
import minflux_viewer.core.mbm_file
import minflux_viewer.core.minflux_zarr
import minflux_viewer.core.obf_image_source
import minflux_viewer.core.overlay
import minflux_viewer.core.save
assert not any(name == 'minflux_viewer.msr' or name.startswith('minflux_viewer.msr.')
               for name in sys.modules), sorted(name for name in sys.modules if '.msr' in name)
"""
    result = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True)
    assert result.returncode == 0, result.stdout + result.stderr


@pytest.mark.parametrize("old,new", [
    ("msr.alignment", "core.bead_alignment"),
    ("msr.acquisition_roi", "core.acquisition_metadata"),
    ("msr.zarr2", "utils.zarr2"),
    ("msr.obf_compat", "utils.obf_compat"),
])
def test_compatibility_imports_preserve_module_identity(old, new):
    assert importlib.import_module(f"minflux_viewer.{old}") is importlib.import_module(f"minflux_viewer.{new}")


def test_msr_writer_is_resolved_through_its_registered_capability(monkeypatch):
    sentinel = object()
    requested = []

    def resolve(name):
        requested.append(name)
        return sentinel

    monkeypatch.setattr(formats, "_load_predicate", resolve)
    assert formats.writer_for("msr") is sentinel
    assert requested == ["minflux_viewer.msr.writer:write_datasets_msr"]
    with pytest.raises(ValueError, match="no registered"):
        formats.writer_for("npy")
