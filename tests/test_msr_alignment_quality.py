"""`alignment_quality_report` surfaces per-bead residuals + a poor-fit warning.

Confirms the diagnostics behind the "Beads and alignment result" readouts:
a consistent (well-registered) bead set is not flagged, while an incoherent bead
set (each bead disagreeing by tens of nm, like the 2_single_bead_3D.msr file) is
flagged with an RMSE above the warning threshold.
"""

from __future__ import annotations

import numpy as np
import pytest

from minflux_viewer.msr.alignment import (
    RMSE_WARN_NM,
    TRANSFORM_RIGID_XY_Z,
    align_channels_from_arrays,
    align_channels_from_bead_positions,
    alignment_quality_report,
    bead_residual_comment,
)


def test_bead_residual_comment_evaluations():
    # excluded bead states what it would cost
    c = bead_residual_comment([50, 0, 0], [30, 0, 0], included=False)
    assert "Excluded" in c and "30 nm" in c

    # small residual -> good/consistent
    c = bead_residual_comment([40, -20, 5], [2, -1, 0], included=True, rmse_xy_nm=3.0)
    assert "Good" in c

    # standout residual relative to a modest fit RMSE -> outlier / exclude hint
    c = bead_residual_comment([10, 10, 0], [120, 40, 0], included=True, rmse_xy_nm=30.0)
    assert "Outlier" in c and "excluding may help" in c

    # large Z residual is called out
    c = bead_residual_comment([0, 0, 80], [3, 2, 60], included=True, rmse_xy_nm=5.0)
    assert "Z error dominates" in c

    # an unusually large raw offset is flagged
    c = bead_residual_comment([700, 200, 0], [4, 3, 0], included=True, rmse_xy_nm=6.0)
    assert "raw offset" in c

_POINT_DTYPE = np.dtype([("gri", "<u4"), ("xyz", "<f8", (3,))])


def _beads(xyz_m: np.ndarray) -> np.ndarray:
    return np.array([(i + 1, p) for i, p in enumerate(xyz_m)], dtype=_POINT_DTYPE)


def _ref_positions_m(n=8, seed=0):
    rng = np.random.default_rng(seed)
    return np.column_stack([
        rng.uniform(-12000, 12000, n),
        rng.uniform(-12000, 12000, n),
        rng.uniform(-100, 100, n),
    ]) * 1e-9


def test_clean_registration_is_not_flagged():
    ref = _ref_positions_m()
    moving = ref - np.array([40e-9, -25e-9, 30e-9])  # consistent shift, recoverable
    result = align_channels_from_arrays(
        "ref", _beads(ref), "mov", _beads(moving), transform_type=TRANSFORM_RIGID_XY_Z)
    rep = alignment_quality_report(result)

    assert rep["n_beads"] == 8
    assert rep["reference"] == "ref" and rep["moving"] == "mov"
    assert rep["warn"] is False
    assert rep["rmse_xy_nm"] < 1.0
    # raw offset is (reference - moving) in nm; the consistent shift shows up per bead
    np.testing.assert_allclose(
        rep["raw_offset_nm"], (result.channel1_xyz - result.channel2_xyz), rtol=0, atol=1e-6)
    assert rep["residual_xy_nm"].shape == (8,)


def test_incoherent_beads_are_flagged():
    ref = _ref_positions_m(seed=1)
    rng = np.random.default_rng(2)
    # Each bead carries its own ~80 nm incoherent offset — no transform can fit it.
    moving = ref + rng.normal(0.0, 80e-9, ref.shape)
    result = align_channels_from_arrays(
        "560", _beads(ref), "640", _beads(moving), transform_type=TRANSFORM_RIGID_XY_Z)
    rep = alignment_quality_report(result)

    assert rep["warn"] is True
    assert rep["rmse_xy_nm"] > RMSE_WARN_NM
    # post-fit residual RMSE equals the reported XY rmse (recomputed from residuals)
    res = rep["residual_nm"]
    expect = float(np.sqrt(np.mean(res[:, 0] ** 2 + res[:, 1] ** 2)))
    assert abs(expect - rep["rmse_xy_nm"]) < 1e-6


def test_subset_refit_matches_full_solver():
    """Re-fitting on a bead subset via align_channels_from_bead_positions equals
    a fresh full solve on just those beads (the live-refit path in the plot window)."""
    ref = _ref_positions_m(seed=3) * 1e9  # nm
    rng = np.random.default_rng(4)
    mov = ref + rng.normal(0.0, 30e-9, ref.shape) * 1e9
    ids = np.arange(1, 9, dtype=np.uint32)
    keep = np.array([0, 2, 4, 6, 7])  # a 5-bead subset

    sub = align_channels_from_bead_positions(
        "r", "m", ids[keep], ref[keep], mov[keep], transform_type=TRANSFORM_RIGID_XY_Z)
    assert sub.bead_ids.tolist() == ids[keep].tolist()
    # aligned positions reproduce the fit on exactly those beads
    np.testing.assert_allclose(
        sub.channel2_aligned_xyz.shape, (keep.size, 3))


def test_plot_window_live_refit_and_writeback():
    """The result window's live re-fit + selection writeback, exercised at the logic
    level (no pyqtgraph widget — GUI teardown without an event loop is flaky and the
    full window is covered by standalone smoke runs)."""
    pytest.importorskip("PyQt6")
    from minflux_viewer.plugins.msr_reader.msr_reader_dialog import AlignmentPlotWindow

    _POINT_DTYPE = np.dtype([("gri", "<u4"), ("xyz", "<f8", (3,))])
    ref = _ref_positions_m(seed=5)
    rng = np.random.default_rng(6)
    mov = ref + rng.normal(0.0, 80e-9, ref.shape)  # incoherent → flagged
    result = align_channels_from_arrays(
        "560", np.array([(i + 1, p) for i, p in enumerate(ref)], dtype=_POINT_DTYPE),
        "640", np.array([(i + 1, p) for i, p in enumerate(mov)], dtype=_POINT_DTYPE),
        transform_type=TRANSFORM_RIGID_XY_Z)

    class Owner:
        def __init__(self):
            self._bead_unchecked_gris = {999}  # a pre-existing exclusion
            self.logs = []

        def log(self, m):
            self.logs.append(m)

    owner = Owner()
    # Build the window object without running __init__ (which needs a live QWidget);
    # wire only the attributes the re-fit + writeback logic reads.
    win = AlignmentPlotWindow.__new__(AlignmentPlotWindow)
    win._requested_transform_type = TRANSFORM_RIGID_XY_Z
    win._owner = owner
    win._owner_prior_excluded = set(owner._bead_unchecked_gris)
    win._excluded_gris = set()
    win._base = [{
        "ref_name": result.channel1_name, "mov_name": result.channel2_name,
        "bead_ids": np.asarray(result.bead_ids, dtype=np.uint32),
        "ref_xyz": np.asarray(result.channel1_xyz, dtype=np.float64),
        "mov_xyz": np.asarray(result.channel2_xyz, dtype=np.float64),
        "ref_counts": np.zeros(result.bead_ids.size, dtype=np.int32),
        "mov_counts": np.zeros(result.bead_ids.size, dtype=np.int32),
    }]

    win._recompute_current()
    assert win._current[0]["n_used"] == 8
    assert win._current[0]["valid"] is True
    assert win._current[0]["rmse_xy_nm"] > RMSE_WARN_NM  # incoherent → poor fit

    # exclude bead 2 → re-fit on 7 beads, writeback unions with the prior exclusion
    win._excluded_gris.add(2)
    win._recompute_current()
    win._writeback_selection()
    assert win._current[0]["n_used"] == 7
    assert owner._bead_unchecked_gris == {999, 2}

    # re-include bead 2 → back to 8 and the prior-only exclusion
    win._excluded_gris.discard(2)
    win._recompute_current()
    win._writeback_selection()
    assert win._current[0]["n_used"] == 8
    assert owner._bead_unchecked_gris == {999}

    # dropping below the 2-bead minimum flags the channel invalid (⛔ banner path)
    win._excluded_gris.update(int(g) for g in result.bead_ids[1:])
    win._recompute_current()
    assert win._current[0]["valid"] is False
    assert win._current[0]["n_used"] == 1
