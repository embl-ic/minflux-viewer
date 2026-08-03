"""
"Bold the useful iterations" feature: mark iteration-selector entries that hold
real values for the current attribute.

* ``loader.effective_iterations_for_attr`` — which iterations carry real data.
* ``iteration.iteration_bold_flags`` — map that to which dropdown labels to bold.
* Histogram window — applies a bold ``FontRole`` to those items.
"""

from __future__ import annotations

import numpy as np
import pytest

from minflux_viewer.core import loader
from minflux_viewer.core.iteration import iteration_bold_flags, iteration_labels


def _make_ds(n_loc=200, n_itr=5, cfr_iter=2, seed=0):
    """Flat m2410 mfx: cfr non-zero only at iteration *cfr_iter*; dcr/efo/eco at
    every iteration."""
    rng = np.random.default_rng(seed)
    n = n_loc * n_itr
    itr = np.tile(np.arange(n_itr), n_loc)
    tid = np.repeat(np.arange(n_loc), n_itr)
    dt = np.dtype([
        ("vld", "?"), ("itr", "<i4"), ("tid", "<i4"), ("loc", "<f8", (3,)),
        ("cfr", "<f8"), ("dcr", "<f8"), ("efo", "<f8"), ("eco", "<f8"),
    ])
    mfx = np.zeros(n, dtype=dt)
    mfx["itr"] = itr
    mfx["tid"] = tid
    mfx["vld"] = True
    mfx["loc"][:, 0] = np.repeat(rng.uniform(0, 1e-6, n_loc), n_itr)
    rows = itr == cfr_iter
    mfx["cfr"][rows] = rng.uniform(0.3, 0.8, int(rows.sum()))     # measured at one iter
    mfx["dcr"] = rng.uniform(0.2, 0.8, n)                          # every iteration
    mfx["efo"] = rng.uniform(1e4, 2e5, n)
    mfx["eco"] = rng.poisson(50, n).astype(float) + 1.0
    return loader.load_from_mfx_array(mfx, "iters")


# --------------------------------------------------------------------------- pure
def test_iteration_bold_flags_maps_effective_to_labels():
    labels = iteration_labels(10)
    eff = np.zeros(10, bool)
    eff[[4, 6]] = True                                            # cfr-like (5th, 7th)
    flags = iteration_bold_flags(labels, eff, 10)
    bold = {labels[i] for i, f in enumerate(flags) if f}
    assert bold == {"5th", "7th"}
    # pooled modes and 'last' (iter 9 not effective) are not bold
    assert not flags[labels.index("all [flatten]")]
    assert not flags[labels.index("all [stacked]")]
    assert not flags[labels.index("last (10th)")]


def test_iteration_bold_flags_last_bold_when_final_iter_effective():
    labels = iteration_labels(10)
    eff = np.ones(10, bool)                                        # dcr-like (all)
    flags = iteration_bold_flags(labels, eff, 10)
    assert flags[labels.index("last (10th)")]
    assert flags[labels.index("1st")]
    # still never the pooled modes
    assert not flags[labels.index("all [flatten]")]


# --------------------------------------------------------------------------- loader
def test_effective_iterations_cfr_single_dcr_all():
    ds = _make_ds(n_itr=5, cfr_iter=2)
    cfr = loader.effective_iterations_for_attr(ds, "cfr")
    assert np.array_equal(np.flatnonzero(cfr), [2])               # only the measured iter
    dcr = loader.effective_iterations_for_attr(ds, "dcr")
    assert dcr.all()                                              # every iteration
    assert loader.effective_iterations_for_attr(ds, "efo").all()


def test_effective_iterations_unknown_attr_all_true():
    ds = _make_ds(n_itr=5)
    # an attribute not in the raw store cannot be resolved → assume all effective
    out = loader.effective_iterations_for_attr(ds, "not_a_real_attr")
    assert out.shape == (5,)
    assert out.all()


# --------------------------------------------------------------------------- UI
def test_histogram_bolds_effective_iterations(qtbot):
    from PyQt6.QtCore import Qt
    from minflux_viewer.core.app_state import AppState
    from minflux_viewer.ui.histogram_window import HistogramWindow

    ds = _make_ds(n_itr=5, cfr_iter=2)
    state = AppState()
    state.add_dataset(ds)
    win = HistogramWindow(state, dataset_idx=0)
    qtbot.addWidget(win)

    def bold_labels():
        combo = win._iter_combo
        out = set()
        for i in range(combo.count()):
            font = combo.itemData(i, Qt.ItemDataRole.FontRole)
            if font is not None and font.bold():
                out.add(combo.itemText(i))
        return out

    win._attr_combo.setCurrentText("cfr")
    assert bold_labels() == {"3rd"}                              # only the measured iter
    win._attr_combo.setCurrentText("dcr")
    # dcr recorded every iteration → all individual iters + 'last' bold, pooled not
    dcr_bold = bold_labels()
    assert "last (5th)" in dcr_bold and "1st" in dcr_bold and "3rd" in dcr_bold
    assert "all [flatten]" not in dcr_bold and "all [stacked]" not in dcr_bold
