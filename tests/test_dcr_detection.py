"""
Two-channel DCR detection (`loader._dcr_is_two_channel`).

Real MINFLUX two-channel DCR columns are complementary detector ratios, but
each is quantised independently (~12-bit, step 1/4096), so the two columns sum
to 1.0 only within ~2.5e-4 — verified on the 260727-155608 sample (col0+col1 in
[0.99976, 1.00024]; 46% exactly 1.0 at float tol, 100% within 1e-3). The
detector must tolerate that while still rejecting legacy per-iteration DCR.
"""

from __future__ import annotations

import numpy as np

from minflux_viewer.core.loader import _dcr_is_two_channel

_Q = 1.0 / 4096.0


def _quantized_two_channel(n=1000, seed=0):
    """Complementary ratios, each independently rounded to n/4096 — so the sum
    drifts to (4096 ± 1)/4096, exactly like the real instrument data."""
    rng = np.random.default_rng(seed)
    c0 = np.round(rng.uniform(0.25, 0.95, n) * 4096) / 4096
    c1 = (np.round((1.0 - c0) * 4096) + rng.choice([-1, 0, 1], n)) / 4096
    c1 = np.clip(c1, 0.0, 1.0)
    return np.column_stack([c0, c1])


def test_detects_quantized_two_channel():
    arr = _quantized_two_channel()
    # sums are near 1.0 but not exact (this is the whole point)
    s = arr[:, 0] + arr[:, 1]
    assert np.any(np.abs(s - 1.0) > 1e-6)          # not float-exact
    assert np.all(np.abs(s - 1.0) <= _Q + 1e-9)    # but within one quantum
    assert _dcr_is_two_channel(arr) is True


def test_rejects_independent_per_iteration_ratios():
    rng = np.random.default_rng(1)
    ind = np.column_stack([rng.uniform(0, 1, 1000), rng.uniform(0, 1, 1000)])
    assert _dcr_is_two_channel(ind) is False


def test_skips_all_zero_invalid_rows():
    arr = _quantized_two_channel()
    arr[:60] = 0.0                                  # invalid placeholder rows first
    assert _dcr_is_two_channel(arr) is True         # skipped, still detected


def test_old_1e6_tolerance_would_have_failed():
    """Guard: the previous float-epsilon tolerance misclassified this data."""
    arr = _quantized_two_channel()
    s = arr[:, 0] + arr[:, 1]
    assert not np.all(np.abs(s - 1.0) < 1e-6)       # would have returned False
    assert _dcr_is_two_channel(arr, tol=1e-3) is True
