"""UI-level integration test for the Drift Correction plugin window."""

from __future__ import annotations

import sys

import numpy as np
import pytest


@pytest.fixture(scope="module")
def _qt_app():
    pytest.importorskip("PyQt6")
    from PyQt6.QtWidgets import QApplication
    return QApplication.instance() or QApplication(sys.argv)


def _make_drifting_dataset(seed=3):
    """A drifting 2D MINFLUX-like dataset built via the canonical builder."""
    from minflux_viewer.core.dataset import build_localization_dataset

    rng = np.random.default_rng(seed)
    n_struct = 300
    fx = rng.uniform(-800, 800, n_struct)
    fy = rng.uniform(-800, 800, n_struct)
    reps = 50
    t = np.linspace(0.0, 3000.0, n_struct * reps)
    feat = np.tile(np.arange(n_struct), reps)
    feat = feat[rng.permutation(t.size)]
    t = np.sort(t)
    dxt = 15.0 * (t / 3000.0)
    dyt = -10.0 * (t / 3000.0)
    x = fx[feat] + dxt + rng.normal(0, 1.0, t.size)
    y = fy[feat] + dyt + rng.normal(0, 1.0, t.size)
    ds = build_localization_dataset(
        name="drifting", x_nm=x, y_nm=y, tid=feat.astype(np.int64), tim=t)
    return ds


def test_plugin_registered(_qt_app):
    from minflux_viewer import plugins
    plugins.ensure_loaded()
    assert "Drift Correction" in [e.name for e in plugins.available()]


def test_estimate_and_apply_creates_corrected_dataset(_qt_app):
    from minflux_viewer.core.app_state import AppState
    from minflux_viewer.plugins.drift_correction.drift_correction_window import (
        DriftCorrectionWindow,
    )

    state = AppState()
    state.add_dataset(_make_drifting_dataset())

    win = DriftCorrectionWindow(state)
    win._dims_combo.setCurrentIndex(1)          # force 2D
    win._res_spin.setValue(4.0)
    win._auto_T.setChecked(False)
    win._T_spin.setValue(400.0)

    win._on_estimate()
    assert win._result is not None
    assert win._result.dims == 2
    assert win._result.n_windows >= 2
    # Estimated drift should trend (nonzero spread).
    assert np.std(win._result.dxt) > 1.0

    n_before = len(state.datasets)
    win._on_apply()
    assert len(state.datasets) == n_before + 1
    new = state.datasets[-1]
    assert new.metadata.get("drift_corrected") is True
    assert new.metadata["drift_correction"]["dims"] == 2
    # Corrected coordinates are final -> RIMF pinned to 1.0.
    assert float(getattr(new.cali, "RIMF", 1.0)) == 1.0
    # Same number of localizations carried over.
    assert new.prop.num_loc == state.datasets[0].prop.num_loc

    win.close()


def test_non_minflux_dataset_is_rejected(_qt_app, monkeypatch):
    from minflux_viewer.core.dataset import build_localization_dataset
    from minflux_viewer.core.app_state import AppState
    from minflux_viewer.plugins.drift_correction import drift_correction_window as mod

    # No real tid -> not MINFLUX.
    rng = np.random.default_rng(0)
    ds = build_localization_dataset(
        name="plain", x_nm=rng.normal(size=500), y_nm=rng.normal(size=500))
    state = AppState()
    state.add_dataset(ds)

    win = mod.DriftCorrectionWindow(state)
    # _validate should report a problem (non-MINFLUX) rather than proceeding.
    assert win._validate(state.active_dataset) is not None
    win.close()
