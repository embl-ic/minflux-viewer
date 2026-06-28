"""
Regression tests for the effective CFR/EFC iteration shown in the dataset
info dialog (``_iterations_text`` → ``metadata["cfr_iteration"]`` /
``["efc_iteration"]``).

The selector must work for **both** source layouts:

* m2205: 2-D ``(num_loc, num_itr)`` per-iteration columns.
* m2410: flat 1-D event stream (one row per localization-iteration, indexed
  by a 1-D ``itr`` column). CFR/EFC are only measured at a subset of loop
  iterations and are zero/NaN elsewhere — previously the selector returned
  ``None`` for this layout, so the dialog showed nothing.
"""

from __future__ import annotations

import numpy as np

from minflux_viewer.core import loader


def test_m2410_flat_effective_iteration():
    """Flat per-row CFR/EFC: pick the dominant-signal iteration (1-based)."""
    n_loc, n_itr = 100, 10
    itr = np.tile(np.arange(n_itr), n_loc)          # 0..9 repeated
    vld = np.ones(itr.size, dtype=bool)

    # CFR measured (non-zero) only at iteration index 6 (→ "7th"); 0 elsewhere.
    cfr = np.zeros(itr.size)
    cfr[itr == 6] = 0.4
    # EFC measured at iteration index 6, NaN elsewhere (m2410 convention).
    efc = np.full(itr.size, np.nan)
    efc[itr == 6] = 5.0e4

    raw = {"itr": itr, "vld": vld, "cfr": cfr, "efc": efc}
    assert loader._effective_iteration_column(raw, "cfr") == 7
    assert loader._effective_iteration_column(raw, "efc") == 7

    meta = loader._effective_iteration_metadata(raw)
    assert meta == {"cfr_iteration": 7, "efc_iteration": 7}


def test_m2410_flat_dominant_iteration_when_multiple():
    """When CFR is carried by more than one iteration, the iteration with the
    most total signal wins (mirrors the two-sequence sample files)."""
    # A small sub-sequence carries CFR at itr 4; the dominant one at itr 6.
    small = np.full(50, 4)
    big = np.full(2000, 6)
    itr = np.concatenate([small, big])
    cfr = np.concatenate([np.full(50, 0.5), np.full(2000, 0.4)])
    raw = {"itr": itr, "vld": np.ones(itr.size, bool), "cfr": cfr}
    assert loader._effective_iteration_column(raw, "cfr") == 7  # itr 6 → 7th


def test_m2410_flat_respects_validity():
    itr = np.tile(np.arange(4), 10)
    cfr = np.zeros(itr.size)
    cfr[itr == 2] = 1.0          # valid CFR iteration
    cfr[itr == 3] = 5.0          # bigger, but all-invalid → must be ignored
    vld = np.ones(itr.size, bool)
    vld[itr == 3] = False
    raw = {"itr": itr, "vld": vld, "cfr": cfr}
    assert loader._effective_iteration_column(raw, "cfr") == 3  # itr 2 → 3rd


def test_m2410_flat_no_signal_returns_none():
    itr = np.tile(np.arange(5), 10)
    raw = {"itr": itr, "vld": np.ones(itr.size, bool), "cfr": np.zeros(itr.size)}
    assert loader._effective_iteration_column(raw, "cfr") is None


def test_m2205_2d_effective_iteration_unchanged():
    """The original m2205 2-D path keeps working (argmax over columns)."""
    n_loc, n_itr = 30, 5
    itr2d = np.tile(np.arange(n_itr), (n_loc, 1))    # (num_loc, num_itr)
    cfr = np.zeros((n_loc, n_itr))
    cfr[:, 3] = 0.4                                   # column 3 → "4th"
    raw = {"itr": itr2d, "vld": np.ones(n_loc, bool), "cfr": cfr}
    assert loader._effective_iteration_column(raw, "cfr") == 4
