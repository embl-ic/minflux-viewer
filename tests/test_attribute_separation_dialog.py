"""
End-to-end smoke test for the attribute-agnostic channel separation dialog
(the redesigned "Separate Channel by DCR").

Uses pytest-qt's ``qtbot`` fixture — manually creating the QApplication in the
test body crashes pyqtgraph widget creation on this Windows setup, whereas the
``qtbot``-managed app is stable. The deferred initial fit (``QTimer`` → sklearn)
is never spun here (sklearn's threadpool × a live Qt loop hard-crashes under
pytest-qt on Windows); the fit *math* is covered by ``test_distribution_fit`` /
``test_attribute_channels``, and the full fit path is verified via a standalone
integration run.
"""

from __future__ import annotations

import numpy as np
import pytest

pytest.importorskip("PyQt6")
pytest.importorskip("pyqtgraph")
pytest.importorskip("pytestqt")

from minflux_viewer.core import loader


def _make_bimodal_dcr(n_traces=200, locs_per=6, seed=0):
    """Flat m2410 mfx array: half the traces sit at DCR≈0.3, half at ≈0.7."""
    rng = np.random.default_rng(seed)
    tids = np.repeat(np.arange(n_traces), locs_per)
    n = tids.size
    dt = np.dtype([
        ("vld", "?"), ("itr", "<i4"), ("tid", "<i4"),
        ("loc", "<f8", (3,)), ("dcr", "<f8"), ("eco", "<f8"), ("efo", "<f8"),
    ])
    mfx = np.zeros(n, dtype=dt)
    mfx["tid"] = tids
    mfx["itr"] = 0
    mfx["vld"] = True
    mfx["loc"][:, 0] = rng.uniform(0, 1e-6, n)
    mfx["loc"][:, 1] = rng.uniform(0, 1e-6, n)
    trace_dcr = np.where(np.arange(n_traces) % 2 == 0, 0.30, 0.70)
    mfx["dcr"] = np.clip(np.repeat(trace_dcr, locs_per) + rng.normal(0, 0.03, n), 0, 1)
    mfx["eco"] = rng.poisson(50, n).astype(float) + 1.0
    mfx["efo"] = rng.uniform(1e4, 2e5, n)
    return mfx


def _make_multi_iter_dcr(n_loc=300, n_itr=3, seed=0):
    """Flat m2410 mfx with several iterations whose DCR distribution shifts per
    iteration — so 'all [stacked]' differs visibly from 'all [flatten]'."""
    rng = np.random.default_rng(seed)
    n = n_loc * n_itr
    itr = np.tile(np.arange(n_itr), n_loc)
    tid = np.repeat(np.arange(n_loc), n_itr)
    dt = np.dtype([
        ("vld", "?"), ("itr", "<i4"), ("tid", "<i4"),
        ("loc", "<f8", (3,)), ("dcr", "<f8"), ("eco", "<f8"),
    ])
    mfx = np.zeros(n, dtype=dt)
    mfx["itr"] = itr
    mfx["tid"] = tid
    mfx["vld"] = True
    mfx["loc"][:, 0] = np.repeat(rng.uniform(0, 1e-6, n_loc), n_itr)
    base = np.repeat(np.where(np.arange(n_loc) % 2 == 0, 0.3, 0.7), n_itr)
    mfx["dcr"] = np.clip(base + 0.05 * itr + rng.normal(0, 0.02, n), 0, 1)
    mfx["eco"] = rng.poisson(50, n).astype(float) + 1.0
    return mfx


class _FakeOwner:
    def __init__(self):
        self.applied = None

    def apply_channel_separation(self, idx, labels, channels, *, attribute="", method_label=""):
        self.applied = (idx, np.asarray(labels).copy(), list(channels), attribute)
        return True


def _dialog(qtbot):
    from minflux_viewer.core.app_state import AppState
    from minflux_viewer.ui.attribute_separation_dialog import AttributeSeparationDialog

    ds = loader.load_from_mfx_array(_make_bimodal_dcr(), "dcr_test")
    state = AppState()
    state.add_dataset(ds)
    owner = _FakeOwner()
    dlg = AttributeSeparationDialog(state, 0, attribute="dcr",
                                    allow_photon_weight=True, owner=owner)
    qtbot.addWidget(dlg)                        # qtbot owns lifetime / cleanup
    return dlg, ds, owner


def test_dialog_seeds_two_channels_and_assigns(qtbot):
    dlg, ds, owner = _dialog(qtbot)
    # opens with two evenly-placed channels (fit is deferred to the event loop)
    assert len(dlg._rows) == 2
    labels = dlg._current_labels()
    assert labels is not None
    assert labels.size == ds.prop.num_loc
    # the two DCR populations split across the two channels
    assert set(np.unique(labels[labels >= 0]).tolist()) == {0, 1}


def test_dialog_apply_calls_owner_with_channels(qtbot):
    dlg, ds, owner = _dialog(qtbot)
    dlg._apply()
    assert owner.applied is not None
    idx, labels, channels, attribute = owner.applied
    assert idx == 0
    assert attribute == "dcr"
    assert len(channels) == 2
    assert labels.size == ds.prop.num_loc


def test_dialog_channel_ops_and_photon_weight(qtbot):
    dlg, ds, owner = _dialog(qtbot)
    dlg._nch_combo.setCurrentText("3")
    dlg._place_evenly()
    assert len(dlg._rows) == 3
    dlg._add_channel()
    assert len(dlg._rows) == 4
    # photon-weighted DCR pooling path (falls back cleanly if unavailable)
    dlg._photon_chk.setChecked(True)
    labels = dlg._current_labels()
    assert labels is None or labels.size == ds.prop.num_loc
    # majority-vote decision path
    dlg._decision_combo.setCurrentText("trace majority vote")
    dlg._refresh_counts()


def test_dialog_attribute_picker_switches(qtbot):
    """The generic 'by attribute' mode shows an attribute picker and rebuilds
    the distribution + channels when the attribute changes."""
    from minflux_viewer.core.app_state import AppState
    from minflux_viewer.ui.attribute_separation_dialog import AttributeSeparationDialog

    ds = loader.load_from_mfx_array(_make_bimodal_dcr(), "dcr_test")
    state = AppState()
    state.add_dataset(ds)
    dlg = AttributeSeparationDialog(state, 0, attribute="dcr", pick_attribute=True,
                                    allow_photon_weight=True, owner=_FakeOwner())
    qtbot.addWidget(dlg)

    assert dlg._attr_combo is not None
    items = [dlg._attr_combo.itemText(i) for i in range(dlg._attr_combo.count())]
    assert "dcr" in items and "efo" in items
    # switch to efo → basis + channels rebuild for the new attribute
    dlg._attr_combo.setCurrentText("efo")
    assert dlg._attribute == "efo"
    assert len(dlg._rows) == 2
    labels = dlg._current_labels()
    assert labels is not None and labels.size == ds.prop.num_loc


def test_stacked_iteration_differs_from_flatten(qtbot):
    """'all [stacked]' draws one coloured series per iteration + a legend, while
    'all [flatten]' is a single pooled histogram — the fit basis (pooled values)
    is identical, only the display differs."""
    from minflux_viewer.core.app_state import AppState
    from minflux_viewer.core.iteration import FLATTEN_LABEL, STACKED_LABEL
    from minflux_viewer.ui.attribute_separation_dialog import AttributeSeparationDialog

    ds = loader.load_from_mfx_array(_make_multi_iter_dcr(n_itr=3), "multi")
    state = AppState()
    state.add_dataset(ds)
    dlg = AttributeSeparationDialog(state, 0, attribute="dcr", owner=_FakeOwner())
    qtbot.addWidget(dlg)

    def snapshot(label):
        dlg._iter_combo.setCurrentText(label)          # → _on_basis_changed → redraw
        pi = dlg._plot.getPlotItem()
        curves = [it for it in pi.items if it.__class__.__name__ == "PlotDataItem"]
        bars = [it for it in pi.items if it.__class__.__name__ == "BarGraphItem"]
        return dlg._values.size, len(curves), len(bars), (pi.legend is not None)

    n_flat, curves_flat, bars_flat, legend_flat = snapshot(FLATTEN_LABEL)
    n_stack, curves_stack, bars_stack, legend_stack = snapshot(STACKED_LABEL)

    assert n_flat == n_stack                            # same pooled fit basis
    assert bars_flat == 1 and not legend_flat           # flatten: single pooled bar, no legend
    assert bars_stack == 0 and legend_stack             # stacked: no pooled bar, has a legend
    assert curves_stack == 3                            # one series per iteration


def test_dialog_fit_updates_channels(qtbot):
    """The fit path (sklearn under a live Qt loop) is stable under qtbot."""
    dlg, ds, owner = _dialog(qtbot)
    dlg._fit_combo.setCurrentIndex(dlg._fit_combo.findData("gaussian"))
    dlg._comp_combo.setCurrentText("2")
    dlg._run_fit()
    assert len(dlg._rows) == 2
    # a fitted 2-component split still assigns both DCR populations
    labels = dlg._current_labels()
    assert set(np.unique(labels[labels >= 0]).tolist()) == {0, 1}
