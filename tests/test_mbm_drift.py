"""Post-hoc MBM drift correction (``msr/mbm_drift.py``).

The synthetic tests pin the algorithm's invariants; the reference tests at the
bottom check it against two real Imspector-corrected ``.msr`` pairs and are
skipped when that sample data is not on the machine.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from minflux_viewer.msr.mbm_drift import (
    DEFAULT_SMOOTH_SAMPLES,
    DriftCorrectionError,
    apply_correction,
    bead_trace,
    corrected_loc,
    drift_at,
    drift_curve,
    usable_bead_gris,
    used_bead_gris,
)

POINTS_DTYPE = np.dtype([("gri", "<u4"), ("xyz", "<f8", (3,)),
                         ("tim", "<f4"), ("str", "<f4")])
MFX_DTYPE = np.dtype([("tim", "<f8"), ("loc", "<f8", (3,)), ("lnc", "<f8", (3,))])


def _points(traces: dict[int, tuple[np.ndarray, np.ndarray]], signal=1.0):
    """Build an mbm points array from ``{gri: (times, xyz_m)}``."""
    rows = []
    for gri, (tim, xyz) in traces.items():
        for t, p in zip(tim, xyz):
            rows.append((gri, p, t, signal))
    out = np.zeros(len(rows), dtype=POINTS_DTYPE)
    for i, (gri, p, t, s) in enumerate(rows):
        out[i] = (gri, p, t, s)
    return out


def _linear_drift(times, per_s=(1e-9, 2e-9, -0.5e-9), origin=(1e-5, 2e-5, 0.0)):
    times = np.asarray(times, dtype=float)
    return np.asarray(origin) + times[:, None] * np.asarray(per_s)


def _mfx(times, lnc):
    out = np.zeros(len(times), dtype=MFX_DTYPE)
    out["tim"] = times
    out["lnc"] = lnc
    out["loc"] = lnc                      # pretend nothing was corrected yet
    return out


# ---------------------------------------------------------------- invariants
def test_drift_is_the_unweighted_mean_of_the_selected_beads():
    """Not a weighted or a median combination — a plain arithmetic mean, and
    beads outside the selection must not contribute at all."""
    t = np.arange(0, 100, 4.0)
    good = _linear_drift(t)
    traces = {
        1: (t, good),
        2: (t, good + 10e-9),                 # +10 nm on every axis
        99: (t, good + 1000e-9),              # a wild bead, NOT selected
    }
    pts = _points(traces)
    at = drift_at(pts, [1, 2], [50.0], smooth_samples=0)[0]
    expected = (np.interp(50.0, t, good[:, 0]) + np.interp(50.0, t, good[:, 0]) + 10e-9) / 2
    assert at[0] == pytest.approx(expected, abs=1e-15)
    # The unselected bead is genuinely excluded. Compare in nm, not with
    # np.allclose: these are metre-scale stage coordinates, so its default
    # rtol=1e-5 would call a 5 nm difference equal.
    only_good = drift_at(pts, [1], [50.0], smooth_samples=0)[0]
    assert (at[0] - only_good[0]) * 1e9 == pytest.approx(5.0, abs=1e-6)
    all_three = drift_at(pts, [1, 2, 99], [50.0], smooth_samples=0)[0]
    assert (all_three[0] - only_good[0]) * 1e9 == pytest.approx((10 + 1000) / 3, abs=1e-6)


def test_a_perfectly_linear_drift_is_removed_exactly():
    """The end-to-end contract: a known drift injected into lnc comes back out."""
    bead_t = np.arange(0, 200, 4.0)
    loc_t = np.linspace(1.0, 195.0, 500)
    truth = np.tile([1e-6, 2e-6, 3e-7], (len(loc_t), 1))       # a still sample
    drift = _linear_drift(loc_t)
    lnc = truth + drift                                        # what we would measure
    pts = _points({1: (bead_t, _linear_drift(bead_t)), 2: (bead_t, _linear_drift(bead_t))})
    mfx = _mfx(loc_t, lnc)
    out = corrected_loc(mfx, pts, [1, 2], smooth_samples=0)
    # a rigid offset is a free convention; the SHAPE must be flat
    assert np.ptp(out, axis=0) == pytest.approx([0, 0, 0], abs=1e-13)


def test_correction_reads_lnc_so_it_is_idempotent():
    """Applying twice with the same beads must be a no-op the second time —
    this is what makes 'lnc' rather than 'loc' the required input."""
    bead_t = np.arange(0, 200, 4.0)
    loc_t = np.linspace(1.0, 195.0, 300)
    pts = _points({1: (bead_t, _linear_drift(bead_t))})
    mfx = _mfx(loc_t, np.tile([1e-6, 2e-6, 0.0], (len(loc_t), 1)) + _linear_drift(loc_t))
    first = apply_correction(mfx, pts, [1])
    snapshot = mfx["loc"].copy()
    second = apply_correction(mfx, pts, [1])
    assert np.array_equal(snapshot, mfx["loc"])
    assert second["median_shift_nm"] == pytest.approx(0.0, abs=1e-9)
    assert first["n_rows"] == len(loc_t)


def test_apply_correction_mutates_in_place_so_shared_arrays_follow():
    """The reader's mfx_map and the parsed dataset entry hold the SAME array;
    the correction must reach both, which is why it writes in place."""
    bead_t = np.arange(0, 100, 4.0)
    loc_t = np.linspace(1.0, 95.0, 50)
    pts = _points({1: (bead_t, _linear_drift(bead_t))})
    mfx = _mfx(loc_t, _linear_drift(loc_t))
    alias = mfx                                   # what mfx_map holds
    before = mfx["loc"].copy()
    apply_correction(mfx, pts, [1])
    assert not np.array_equal(before, alias["loc"])
    assert np.array_equal(mfx["loc"], alias["loc"])


def test_non_finite_rows_stay_non_finite():
    """Failed probes carry NaN coordinates and must not acquire a position."""
    bead_t = np.arange(0, 100, 4.0)
    loc_t = np.linspace(1.0, 95.0, 40)
    lnc = _linear_drift(loc_t)
    lnc[5:9] = np.nan
    pts = _points({1: (bead_t, _linear_drift(bead_t))})
    out = corrected_loc(_mfx(loc_t, lnc), pts, [1])
    assert np.isnan(out[5:9]).all()
    assert np.isfinite(np.delete(out, np.s_[5:9], axis=0)).all()


def test_dead_bead_samples_are_dropped():
    """A zero PMT signal marks a sample where the bead was not seen; its stale
    position must not enter the average."""
    t = np.arange(0, 40, 4.0)
    xyz = _linear_drift(t)
    pts = _points({1: (t, xyz)})
    pts["str"][3] = 0.0                       # kill one sample
    kept_t, kept_xyz = bead_trace(pts, 1)
    assert len(kept_t) == len(t) - 1
    assert np.array_equal(bead_trace(pts, 1, require_signal=False)[0].shape, t.shape)


def test_a_selection_with_no_usable_bead_is_an_error_not_a_silent_zero():
    t = np.arange(0, 40, 4.0)
    pts = _points({1: (t, _linear_drift(t))})
    pts["str"][:] = 0.0                        # every sample dead
    assert usable_bead_gris(pts, [1]) == []
    with pytest.raises(DriftCorrectionError, match="usable samples"):
        drift_at(pts, [1], [10.0])


def test_a_dataset_without_lnc_is_refused_with_the_reason():
    """'loc' alone cannot be re-corrected — the applied correction is not
    invertible — so this must fail loudly rather than correct the wrong field."""
    dtype = np.dtype([("tim", "<f8"), ("loc", "<f8", (3,))])
    mfx = np.zeros(4, dtype=dtype)
    t = np.arange(0, 40, 4.0)
    pts = _points({1: (t, _linear_drift(t))})
    with pytest.raises(DriftCorrectionError, match="lnc"):
        corrected_loc(mfx, pts, [1])


def test_centroid_reference_does_not_move_the_cloud():
    bead_t = np.arange(0, 200, 4.0)
    loc_t = np.linspace(1.0, 195.0, 200)
    pts = _points({1: (bead_t, _linear_drift(bead_t))})
    mfx = _mfx(loc_t, _linear_drift(loc_t))
    out = corrected_loc(mfx, pts, [1], reference="centroid")
    assert out.mean(0) == pytest.approx(mfx["loc"].mean(0), abs=1e-15)


def test_drift_curve_is_zeroed_and_spans_the_bead_times():
    t = np.arange(10.0, 210.0, 4.0)
    pts = _points({1: (t, _linear_drift(t))})
    # smoothing off: its edge handling (mode="nearest") flattens the ramp ends,
    # which would hide whether the curve spans the right range.
    times, drift_nm = drift_curve(pts, [1], n_samples=50, smooth_samples=0)
    assert times[0] == pytest.approx(10.0)
    assert times[-1] == pytest.approx(t[-1])
    assert drift_nm[0] == pytest.approx([0, 0, 0], abs=1e-9)
    assert drift_nm[-1][0] == pytest.approx((t[-1] - 10.0) * 1.0, abs=0.05)   # 1 nm/s


# ------------------------------------------------------- Imspector reference
_ROOT = Path(r"D:\Workspace\Users\Cigdem\2026")
_PAIRS = [
    ("20260625",
     _ROOT / "20260625" / "3_BD_MRED_75pM_MINFLUX_3D.msr",
     _ROOT / "20260625" / "drift_corrected" / "3_corrected_BD_MRED_75pM_MINFLUX_3D.msr",
     [185, 247, 433, 495, 557, 619], [185, 247, 495, 557]),
    ("20260626",
     _ROOT / "20260626" / "3_BD+A_MRED_75pM_MINFLUX_3D.msr",
     _ROOT / "20260626" / "drift_corrected" / "3_corrected_BD+A_MRED_75pM_MINFLUX_3D.msr",
     [123, 247, 309, 371, 495, 557, 619, 681, 743, 805], [309, 371, 495, 681, 743, 805]),
]


#: Keep 1 row in N from the reference files. They hold 3.3 M and 20.1 M rows
#: (~0.3 and ~2.1 GB each, and a pair is loaded per test); at full size the
#: memory these tests hold destabilised the whole suite. A strided sample spans
#: the entire acquisition, so it exercises the drift curve just as well.
_REFERENCE_STRIDE = 20


def _load_msr(path, stride: int = _REFERENCE_STRIDE):
    from minflux_viewer.msr import zarr2
    from minflux_viewer.msr.mfxdta import extract_zarr_store, read_obf_mfxdta_stacks

    _idx, _desc, blob = read_obf_mfxdta_stacks(str(path))[0]
    store = extract_zarr_store(blob)
    arch = zarr2.open(store, mode="r")
    full = arch["mfx"][:]
    mfx = np.array(full[::stride])            # copy, so the full array is freed
    del full
    return mfx, arch["grd/mbm/points"][:], store


@pytest.mark.parametrize("tag,unc,cor,online,manual", _PAIRS,
                         ids=[p[0] for p in _PAIRS])
def test_reproduces_imspector_correction(tag, unc, cor, online, manual):
    """Reproduce a real Imspector drift correction from the uncorrected file.

    Only the *shape* is asserted: the absolute offset is a convention Imspector
    does not store (see the module docstring), and a constant offset is a rigid
    translation that changes no measured distance.
    """
    if not unc.is_file() or not cor.is_file():
        pytest.skip(f"reference sample data not present: {unc}")
    mfx_u, points, store_u = _load_msr(unc)
    mfx_c, _, store_c = _load_msr(cor)

    # The bead sets are recorded in the stores, and differ exactly as expected:
    # the uncorrected file lists the online set, the corrected one the manual pick.
    assert used_bead_gris(store_u) == online
    assert used_bead_gris(store_c) == manual

    predicted = corrected_loc(mfx_u, points, manual)
    truth = np.asarray(mfx_c["loc"], dtype=float)
    finite = np.isfinite(truth).all(1) & np.isfinite(predicted).all(1)
    assert finite.sum() > 5_000

    residual = predicted[finite] - truth[finite]
    shape = residual - residual.mean(0)                    # drop the free offset
    rms_nm = float(np.sqrt((np.linalg.norm(shape, axis=1) ** 2).mean()) * 1e9)
    assert rms_nm < 0.5, f"{tag}: drift shape off by {rms_nm:.3f} nm rms"
    # and the offset really is small — a few nm, not a mis-registration
    assert float(np.linalg.norm(residual.mean(0)) * 1e9) < 5.0
    # NaN rows are preserved exactly
    assert np.array_equal(np.isnan(predicted).all(1), np.isnan(truth).all(1))


@pytest.mark.parametrize("tag,unc,cor,online,manual", _PAIRS,
                         ids=[p[0] for p in _PAIRS])
def test_reproduces_the_online_correction_from_the_online_bead_set(
        tag, unc, cor, online, manual):
    """The same algorithm, run with the beads the instrument used, reproduces
    the correction already baked into the uncorrected file's ``loc`` — which is
    what establishes that online and post-hoc are one mechanism."""
    if not unc.is_file():
        pytest.skip("reference sample data not present")
    mfx_u, points, _store = _load_msr(unc)
    predicted = corrected_loc(mfx_u, points, online)
    truth = np.asarray(mfx_u["loc"], dtype=float)
    finite = np.isfinite(truth).all(1) & np.isfinite(predicted).all(1)
    residual = predicted[finite] - truth[finite]
    shape = residual - residual.mean(0)
    rms_nm = float(np.sqrt((np.linalg.norm(shape, axis=1) ** 2).mean()) * 1e9)
    assert rms_nm < 0.6, f"{tag}: online drift shape off by {rms_nm:.3f} nm rms"


def test_default_smoothing_is_the_fitted_value():
    """Guard the fitted constant: 1.2 bead samples was optimal on both reference
    datasets (unsmoothed leaves 1.13 nm, this leaves 0.16 nm per axis)."""
    assert DEFAULT_SMOOTH_SAMPLES == 1.2


# ------------------------------------------------------------------------ UI
def _ui_fixture():
    """A tiny dataset + beads good enough to drive the dialogs."""
    bead_t = np.arange(0.0, 200.0, 4.0)
    loc_t = np.linspace(1.0, 195.0, 120)
    points = _points({10: (bead_t, _linear_drift(bead_t)),
                      20: (bead_t, _linear_drift(bead_t) + 5e-9)})
    mfx = _mfx(loc_t, _linear_drift(loc_t))
    return points, mfx


def test_beads_drift_dialog_offers_the_button_and_passes_the_selection(qtbot):
    """The button hands the owner exactly what is checked — the dialog itself
    holds no localization data, so the owner performs the correction."""
    pytest.importorskip("PyQt6")
    from minflux_viewer.plugins.msr_reader.beads_drift import extract_bead_drift
    from minflux_viewer.plugins.msr_reader.beads_drift_dialog import BeadsDriftDialog

    points, _mfx_unused = _ui_fixture()
    beads = extract_bead_drift(points, {"10": {"gri": 10, "name": "R1"},
                                        "20": {"gri": 20, "name": "R2"}}, ["R1", "R2"])
    seen = []
    dialog = BeadsDriftDialog([{"name": "ds", "beads": beads}],
                              drift_correction=seen.append)
    qtbot.addWidget(dialog)

    assert dialog._drift_btn.text() == "Drift correction with selected beads"
    assert dialog._drift_btn.isEnabled()
    dialog._run_drift_correction()
    assert seen == [sorted(dialog.selected_gris())]

    # Unchecking everything disables it — there is no drift without a bead.
    for gri in list(dialog._gri_checked):
        dialog._on_toggle(gri, False)
    assert not dialog._drift_btn.isEnabled()
    dialog._run_drift_correction()
    assert len(seen) == 1                       # refused, not called with nothing


def test_beads_drift_dialog_without_the_callback_disables_the_button(qtbot):
    """The MBM-info window builds this dialog too, with no data to rewrite."""
    pytest.importorskip("PyQt6")
    from minflux_viewer.plugins.msr_reader.beads_drift import extract_bead_drift
    from minflux_viewer.plugins.msr_reader.beads_drift_dialog import BeadsDriftDialog

    points, _ = _ui_fixture()
    beads = extract_bead_drift(points, {"10": {"gri": 10, "name": "R1"}}, ["R1"])
    dialog = BeadsDriftDialog([{"name": "ds", "beads": beads}])
    qtbot.addWidget(dialog)
    assert not dialog._drift_btn.isEnabled()
    assert "only available from the MSR reader" in dialog._drift_btn.toolTip()


def test_info_mode_has_no_drift_button_and_still_updates_its_selection_line(qtbot):
    """info_mode builds no button; the selection-line update must not assume one."""
    pytest.importorskip("PyQt6")
    from minflux_viewer.plugins.msr_reader.beads_drift import extract_bead_drift
    from minflux_viewer.plugins.msr_reader.beads_drift_dialog import BeadsDriftDialog

    points, _ = _ui_fixture()
    beads = extract_bead_drift(points, {"10": {"gri": 10, "name": "R1"}}, ["R1"])
    dialog = BeadsDriftDialog([{"name": "ds", "beads": beads}], info_mode=True)
    qtbot.addWidget(dialog)
    assert not hasattr(dialog, "_drift_btn")
    dialog._update_selection_line()              # must not raise


def test_preview_dialog_applies_in_place_and_leaves_lnc_alone(qtbot):
    """Apply rewrites the caller's own array, which is how the correction
    reaches the reader's mfx_map and 'Open in MINFLUX viewer'."""
    pytest.importorskip("PyQt6")
    from minflux_viewer.plugins.msr_reader.drift_correction_dialog import (
        DriftCorrectionDialog,
    )

    points, mfx = _ui_fixture()
    before_loc = mfx["loc"].copy()
    before_lnc = mfx["lnc"].copy()
    dialog = DriftCorrectionDialog([{"name": "ds", "mfx": mfx, "points": points}], [10, 20])
    qtbot.addWidget(dialog)
    assert dialog._tabs.count() == 1
    assert len(dialog._plots) == 1

    dialog._apply()
    assert not np.array_equal(before_loc, mfx["loc"])          # loc rewritten
    assert np.array_equal(before_lnc, mfx["lnc"])              # lnc preserved
    assert dialog.applied[0]["n_rows"] == len(mfx)
    assert dialog.applied[0]["name"] == "ds"


def test_preview_dialog_cancel_changes_nothing(qtbot):
    pytest.importorskip("PyQt6")
    from minflux_viewer.plugins.msr_reader.drift_correction_dialog import (
        DriftCorrectionDialog,
    )

    points, mfx = _ui_fixture()
    before = mfx["loc"].copy()
    dialog = DriftCorrectionDialog([{"name": "ds", "mfx": mfx, "points": points}], [10])
    qtbot.addWidget(dialog)
    dialog.reject()
    assert np.array_equal(before, mfx["loc"])
    assert dialog.applied == []


@pytest.mark.parametrize("finish", ["accept", "reject", "close"])
def test_preview_dialog_retires_its_plots_however_it_is_dismissed(qtbot, finish):
    """⚠ accept()/reject() raise no close event, so hooking only closeEvent left
    the ViewBox registrations leaking for the life of the process in the normal
    modal flow — which is what makes pyqtgraph abort later, in unrelated windows.
    The disposal therefore hangs off done(), the funnel all three routes share."""
    pytest.importorskip("PyQt6")
    import pyqtgraph as pg

    from minflux_viewer.plugins.msr_reader.drift_correction_dialog import (
        DriftCorrectionDialog,
    )

    points, mfx = _ui_fixture()
    dialog = DriftCorrectionDialog([{"name": "ds", "mfx": mfx, "points": points}], [10])
    qtbot.addWidget(dialog)
    boxes = [p.getViewBox() for p in dialog.findChildren(pg.PlotWidget)]
    assert boxes and not any(getattr(b, "_mfv_view_box_closed", False) for b in boxes)

    getattr(dialog, finish)()
    assert all(getattr(b, "_mfv_view_box_closed", False) for b in boxes), (
        f"plots still registered after {finish}()")


def test_preview_dialog_explains_an_uncorrectable_dataset_instead_of_failing(qtbot):
    """A dataset with no lnc, or with none of the selected beads, must say so
    on its own tab and leave Apply unavailable — not raise."""
    pytest.importorskip("PyQt6")
    from minflux_viewer.plugins.msr_reader.drift_correction_dialog import (
        DriftCorrectionDialog,
    )

    points, _ = _ui_fixture()
    no_lnc = np.zeros(5, dtype=np.dtype([("tim", "<f8"), ("loc", "<f8", (3,))]))
    dialog = DriftCorrectionDialog([{"name": "no-lnc", "mfx": no_lnc, "points": points}],
                                   [10])
    qtbot.addWidget(dialog)
    assert dialog._tabs.count() == 1
    dialog._apply()                                # must not raise
    # It reports the reason rather than silently claiming a correction, so the
    # reader can log it against the dataset name.
    assert len(dialog.applied) == 1
    assert "lnc" in dialog.applied[0]["error"]
    assert "n_rows" not in dialog.applied[0]

    # A selection naming only beads this dataset does not have.
    _points_b, mfx = _ui_fixture()
    dialog2 = DriftCorrectionDialog([{"name": "ds", "mfx": mfx, "points": _points_b}],
                                    [999])
    qtbot.addWidget(dialog2)
    before = mfx["loc"].copy()
    dialog2._apply()
    assert np.array_equal(before, mfx["loc"])
