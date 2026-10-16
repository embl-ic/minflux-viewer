"""Scientific calculations import without Qt; UI compatibility stays lazy."""

import importlib
import subprocess
import sys

import numpy as np
import pytest

from minflux_viewer.core.dataset import build_localization_dataset
from minflux_viewer.core.post_load import PostLoadAlgorithms, compute_post_load_results


def test_analysis_package_does_not_import_qt_or_pyqtgraph():
    code = """
import sys
import minflux_viewer.analysis
import minflux_viewer.analysis.local_density
import minflux_viewer.analysis.localization_precision
import minflux_viewer.analysis.trace_analysis
assert not any(n.startswith(('PyQt6', 'pyqtgraph', 'minflux_viewer.ui')) for n in sys.modules), [n for n in sys.modules if n.startswith(('PyQt6', 'pyqtgraph', 'minflux_viewer.ui'))]
"""
    result = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True)
    assert result.returncode == 0, result.stdout + result.stderr


@pytest.mark.parametrize("old,new,symbol", [
    ("analysis.localization_precision", "ui.precision_analysis", "CRLBResultDialog"),
    ("analysis.local_density", "ui.density_analysis", "LocalDensityImageWindow"),
    ("analysis.trace_analysis", "ui.trace_analysis_dialogs", "_AnisotropyDialog"),
    ("analysis", "ui.precision_analysis", "run_frc"),
])
def test_old_symbols_resolve_to_the_canonical_ui_implementation(old, new, symbol):
    legacy = importlib.import_module("minflux_viewer." + old)
    canonical = importlib.import_module("minflux_viewer." + new)
    assert getattr(legacy, symbol) is getattr(canonical, symbol)


def test_post_load_uses_injected_algorithms_without_publishing_results():
    ds = build_localization_dataset(
        name="post-load", x_nm=np.arange(20), y_nm=np.arange(20) * 2,
        z_nm=np.arange(20) * 3, tid=np.repeat(np.arange(4), 5))
    calls = []

    def anisotropy(dataset):
        assert dataset is ds
        calls.append("anisotropy")
        return {"z_scaling_factor": 0.7}

    def precision(points, tid):
        calls.append("precision")
        assert points.shape == (20, 3)
        return {"median_sigma_xyz": np.ones(3)}

    def density(points, prefs, *, dimensions):
        calls.append("density")
        assert dimensions == 3
        np.testing.assert_allclose(points[:, 2], ds.attr["loc_z"] * 1e9 * 0.7)
        return np.ones(20), "injected", "test"

    factor = ds.cali.z_scaling_factor
    result = compute_post_load_results(
        ds, {"data": {"estimate_z_scaling_factor": True, "compute_loc_prec": True,
                       "compute_local_density": True}}, lambda stage: None,
        algorithms=PostLoadAlgorithms(anisotropy, precision, density))
    assert calls == ["anisotropy", "precision", "density"]
    assert result["z_update"]["value"] == 0.7
    np.testing.assert_array_equal(result["density"], np.ones(20))
    assert "den" not in ds.attr
    assert ds.cali.z_scaling_factor == factor
