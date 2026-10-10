"""Dataset-level fiducial (MBM bead) drift correction — analysis/bead_drift.py."""

from __future__ import annotations

import copy
from pathlib import Path

import numpy as np
import pytest

from minflux_viewer.analysis import bead_drift as bd

PAIR = Path(r"D:\Workspace\Users\Cigdem\2026\20260625")


def _points(n_beads=4, bad=None, seed=0):
    rng = np.random.default_rng(seed)
    t = np.arange(0.0, 4000.0, 4.0)
    rows = []
    for k in range(n_beads):
        pos = rng.uniform(-5e-6, 5e-6, 3)
        xyz = pos + np.column_stack([20e-9 * t / 4000, 0 * t, 0 * t])
        if k == bad:
            xyz[:, 1] += 30e-9 * np.sin(2 * np.pi * t / 1000)
        for a, b in zip(t, xyz):
            rows.append((10 + k, b, a, 1.0))
    return np.array(rows, dtype=[("gri", "<u4"), ("xyz", "<f8", (3,)),
                                 ("tim", "<f4"), ("str", "<f4")])


def test_deviation_compares_motion_not_position():
    """Beads micrometres apart but moving together deviate ~0, whatever their
    positions; the one that moves on its own stands out."""
    infos = bd.bead_quality(_points(bad=2), [10, 11, 12, 13])
    dev = {b.gri: b.deviation_nm for b in infos}
    assert dev[12] > 15.0
    assert max(dev[g] for g in (10, 11, 13)) < 1.0
    assert bd.suggest_beads(infos) == [10, 11, 13]


def test_suggestion_needs_three_beads():
    infos = bd.bead_quality(_points(n_beads=2, bad=1), [10, 11])
    assert bd.suggest_beads(infos) == [10, 11]


def test_median_consensus_ignores_one_bad_bead():
    pts = _points(bad=2)
    t = np.linspace(0, 3990, 50)
    mean = bd.consensus_drift(pts, [10, 11, 12, 13], t, consensus="mean")
    med = bd.consensus_drift(pts, [10, 11, 12, 13], t, consensus="median")
    good = bd.consensus_drift(pts, [10, 11, 13], t, consensus="mean")
    # displacements (the absolute level is arbitrary)
    disp = lambda d: (d - d[0]) * 1e9                       # noqa: E731
    assert np.max(np.abs(disp(med) - disp(good))[:, 1]) < 0.5
    assert np.max(np.abs(disp(mean) - disp(good))[:, 1]) > 3.0


def test_recorded_selection_prefers_the_used_list():
    by_gri = {"10": {"gri": 10, "name": "R1", "used": 1}, "11": {"gri": 11, "name": "R2", "used": 1},
              "12": {"gri": 12, "name": "R3", "used": 1}}
    meta = {"native_zarr_mbm_points_attrs": {"points_by_gri": by_gri},
            "native_zarr_mbm_attrs": {"used": ["R1", "R3"]}}
    gris, source = bd._recorded_selection(meta, [10, 11, 12])
    assert gris == [10, 12] and "used" in source
    meta.pop("native_zarr_mbm_attrs")
    gris, source = bd._recorded_selection(meta, [10, 11, 12])
    assert gris == [10, 11, 12] and "flags" in source
    gris, source = bd._recorded_selection({}, [10, 11])
    assert gris == [10, 11] and "no selection" in source


def test_reproduction_removes_only_a_constant_offset():
    rng = np.random.default_rng(1)
    loc = rng.normal(0, 1e-6, (100, 3))
    rms, off = bd.reproduction_rms_nm(loc + np.array([2e-9, -1e-9, 0.0]), loc)
    assert np.allclose(rms, 0.0, atol=1e-9) and np.allclose(off, (2.0, -1.0, 0.0))


@pytest.mark.skipif(not (PAIR / "3_BD_MRED_75pM_MINFLUX_3D.msr").exists(),
                    reason="Imspector-corrected reference pair not available")
def test_real_pair_operator_selection_reproduces_imspector():
    """The operator's four beads, applied to the uncorrected file through the
    dataset path, reproduce Imspector's corrected coordinates."""
    from minflux_viewer.core.app_state import DEFAULT_PREFS
    from minflux_viewer.msr.quick_open import load_viewer_msr

    prefs = copy.deepcopy(DEFAULT_PREFS)
    ds = load_viewer_msr(PAIR / "3_BD_MRED_75pM_MINFLUX_3D.msr", prefs=prefs)[-1]
    dsc = load_viewer_msr(PAIR / "drift_corrected" / "3_corrected_BD_MRED_75pM_MINFLUX_3D.msr",
                          prefs=prefs)[-1]
    inputs, why = bd.bead_inputs_for_dataset(ds)
    corrected, _ = bd.bead_inputs_for_dataset(dsc)
    assert inputs is not None, why
    assert len(inputs.recorded_gris) == 6 and len(corrected.recorded_gris) == 4
    own = bd.correct_with_beads(inputs, inputs.recorded_gris)
    assert max(own.reproduction_nm) < 0.6               # the file's own correction
    op = bd.correct_with_beads(inputs, corrected.recorded_gris)
    rms, _ = bd.reproduction_rms_nm(op.positions_m, corrected.loc_m)
    assert max(rms) < 0.5                                # Imspector's post-hoc one
