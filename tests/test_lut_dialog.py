"""
LUT dialog: draggable min/max level lines (like the histogram filter bounds) and
the gamma tilt line. Plus the make_colormap gamma warp (pure).
"""

from __future__ import annotations

from types import SimpleNamespace

import numpy as np
import pytest

pytest.importorskip("PyQt6")
pytest.importorskip("pytestqt")

from minflux_viewer.ui.lut_dialog import make_colormap


# --------------------------------------------------------------------------- pure
def test_make_colormap_gamma_warps_midtones():
    def mid(cm):
        return int(cm.getLookupTable(0.0, 1.0, 256)[128][0])

    base = make_colormap("gray")
    assert mid(make_colormap("gray", gamma=0.5)) > mid(base)   # <1 brightens
    assert mid(make_colormap("gray", gamma=2.0)) < mid(base)   # >1 darkens
    # endpoints preserved
    for g in (0.5, 1.0, 2.0):
        lut = make_colormap("gray", gamma=g).getLookupTable(0.0, 1.0, 256)
        assert lut[0][0] < 5 and lut[-1][0] > 250


# --------------------------------------------------------------------------- UI
def _dialog(qtbot):
    from minflux_viewer.ui.lut_dialog import LutDialog
    rec = {"levels": [], "cmaps": [], "gammas": []}
    dlg = LutDialog(
        on_levels_changed=lambda lo, hi: rec["levels"].append((lo, hi)),
        on_cmap_changed=lambda n, i: rec["cmaps"].append((n, i)),
        on_gamma_changed=lambda g: rec["gammas"].append(g))
    qtbot.addWidget(dlg)
    dlg.load_image(pixels=np.linspace(0, 100, 1000), data_lo=0.0, data_hi=100.0,
                   lo=20.0, hi=80.0, cmap_name="gray", invert=False, gamma=1.0)
    return dlg, rec


def test_dragging_level_lines_sets_levels(qtbot):
    dlg, rec = _dialog(qtbot)

    dlg._lo_line.setValue(30.0)                     # drag the min line
    assert dlg._lo == pytest.approx(30.0)
    assert rec["levels"][-1] == pytest.approx((30.0, 80.0))
    assert dlg._min_spin.value() == pytest.approx(30.0)     # spinbox synced

    dlg._hi_line.setValue(70.0)                     # drag the max line
    assert dlg._hi == pytest.approx(70.0)
    assert rec["levels"][-1] == pytest.approx((30.0, 70.0))

    # lines can't cross: dragging max below min clamps to min
    dlg._hi_line.setValue(10.0)
    assert dlg._hi == pytest.approx(dlg._lo)


def test_gamma_tilt_line_and_spinbox(qtbot):
    dlg, rec = _dialog(qtbot)
    assert dlg._tf_curve.getData()[0] is not None and len(dlg._tf_curve.getData()[0]) > 0

    # drag the curve up (y = 0.8·ymax) → gamma < 1 (brighter mid-tones)
    ymax = dlg._hist_ymax
    dlg._on_curve_dragged(0.5 * (dlg._lo + dlg._hi), 0.8 * ymax)
    assert rec["gammas"] and rec["gammas"][-1] < 1.0
    assert dlg._gamma < 1.0

    # the spinbox sets it precisely
    dlg._gamma_spin.setValue(2.0)
    assert rec["gammas"][-1] == pytest.approx(2.0)
    assert dlg._gamma == pytest.approx(2.0)
    # γ=1 reset
    dlg._set_gamma(1.0)
    assert dlg._gamma == pytest.approx(1.0)


def test_whole_line_drag_fits_gamma_through_cursor(qtbot):
    """Grabbing the curve anywhere (not just the mid dot) and dragging re-fits
    gamma so the curve passes through the cursor point."""
    dlg, rec = _dialog(qtbot)
    ymax = dlg._hist_ymax
    lo, hi = dlg._lo, dlg._hi

    # grab at t=0.25 and pull up to 0.8·ymax
    x = lo + 0.25 * (hi - lo)
    y = 0.8 * ymax
    dlg._on_curve_dragged(x, y)

    assert rec["gammas"][-1] < 1.0                      # pulling up brightens
    # the redrawn transfer curve passes through (x, y)
    xs, ys = dlg._tf_curve.getData()
    y_at_x = float(np.interp(x, xs, ys))
    assert y_at_x == pytest.approx(y, rel=0.02)

    # grabbing lower down pushes gamma the other way (>1, darker)
    dlg._on_curve_dragged(lo + 0.5 * (hi - lo), 0.2 * ymax)
    assert dlg._gamma > 1.0


def test_live_sync_preserves_reset_baseline(qtbot):
    """A live sync from the owning window (capture_baseline=False) updates the
    display but must NOT move the Reset target."""
    dlg, rec = _dialog(qtbot)                              # opened with lo=20, hi=80, gray
    base = dict(dlg._initial_state)

    dlg.load_image(pixels=np.linspace(0, 100, 1000), data_lo=0.0, data_hi=100.0,
                   lo=10.0, hi=90.0, cmap_name="jet", invert=False, gamma=2.0,
                   capture_baseline=False)
    # displayed state followed the external change …
    assert (dlg._lo, dlg._hi) == pytest.approx((10.0, 90.0))
    assert dlg._gamma == pytest.approx(2.0)
    assert dlg._cmap_combo.currentText() == "jet"
    # … but the Reset baseline is untouched
    assert dlg._initial_state == base

    # a normal (re)open recaptures it
    dlg.load_image(pixels=np.linspace(0, 100, 1000), data_lo=0.0, data_hi=100.0,
                   lo=30.0, hi=70.0, cmap_name="hot", invert=False, gamma=0.5)
    assert dlg._initial_state["lo"] == 30.0 and dlg._initial_state["cmap"] == "hot"


def test_viewbox_left_drag_routes_to_gamma(qtbot):
    """A left-drag anywhere on the plot (not the dot/lines) reaches the gamma
    fitter through the custom viewbox — the whole plot is grabbable."""
    from PyQt6.QtCore import Qt, QPointF

    dlg, rec = _dialog(qtbot)
    ymax = dlg._hist_ymax
    x = 0.5 * (dlg._lo + dlg._hi)
    y = 0.85 * ymax
    dlg._gamma_vb.mapSceneToView = lambda p: QPointF(x, y)   # known drop point

    class _Ev:
        def button(self):
            return Qt.MouseButton.LeftButton
        def accept(self):
            pass
        def scenePos(self):
            return QPointF(0.0, 0.0)

    dlg._gamma_vb.mouseDragEvent(_Ev())
    assert rec["gammas"] and dlg._gamma < 1.0                # pulled up → brighter


def test_auto_uses_same_repeated_imagej_cycle_as_brightness_contrast(qtbot):
    from minflux_viewer.ui.lut_dialog import LutDialog
    from minflux_viewer.ui.render_window import RenderWindow

    rng = np.random.default_rng(7)
    pixels = np.zeros(20_000, dtype=float)
    pixels[:5_000] = rng.lognormal(mean=0.0, sigma=0.8, size=5_000)
    recorded = []
    dlg = LutDialog(
        on_levels_changed=lambda lo, hi: recorded.append((lo, hi)),
        on_cmap_changed=lambda _name, _invert: None,
    )
    qtbot.addWidget(dlg)
    dlg.load_image(
        pixels=pixels,
        data_lo=float(pixels.min()),
        data_hi=float(pixels.max()),
        lo=float(pixels.min()),
        hi=float(pixels.max()),
        cmap_name="hot",
        invert=False,
    )

    reference = SimpleNamespace(_bc_auto_threshold=0)
    expected_first = RenderWindow._compute_auto_levels(
        reference, pixels, advance_auto_threshold=True
    )
    dlg._auto_btn.click()
    assert (dlg._lo, dlg._hi) == pytest.approx(expected_first)
    assert dlg._auto_threshold == reference._bc_auto_threshold == 5000
    assert dlg._auto_btn.isChecked()

    expected_second = RenderWindow._compute_auto_levels(
        reference, pixels, advance_auto_threshold=True
    )
    dlg._auto_btn.click()
    assert (dlg._lo, dlg._hi) == pytest.approx(expected_second)
    assert dlg._auto_threshold == reference._bc_auto_threshold == 2500
    assert dlg._auto_btn.isChecked()
    assert len(recorded) == 2

    dlg._min_spin.setValue(dlg._lo + 0.1 * (dlg._hi - dlg._lo))
    assert dlg._auto_threshold == 0
    assert not dlg._auto_btn.isChecked()


def test_owner_managed_auto_updates_lut_controls_without_double_emitting(qtbot):
    from minflux_viewer.ui.lut_dialog import LutDialog

    auto_calls = []
    level_calls = []
    dlg = LutDialog(
        on_levels_changed=lambda lo, hi: level_calls.append((lo, hi)),
        on_cmap_changed=lambda _name, _invert: None,
        on_auto=lambda: auto_calls.append(True) or (12.5, 87.5),
    )
    qtbot.addWidget(dlg)
    dlg.load_image(
        pixels=np.linspace(0.0, 100.0, 1000),
        data_lo=0.0,
        data_hi=100.0,
        lo=0.0,
        hi=100.0,
        cmap_name="gray",
        invert=False,
    )

    dlg._auto_btn.click()

    assert auto_calls == [True]
    assert level_calls == []
    assert (dlg._lo, dlg._hi) == (12.5, 87.5)
    assert dlg._auto_btn.isChecked()
