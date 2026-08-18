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

from PyQt6.QtWidgets import QApplication

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


def test_custom_colormap_saved_from_lut_menu_is_selected(qtbot):
    from minflux_viewer.colormaps import configure_custom_colormaps
    from minflux_viewer.ui.lut_dialog import LutDialog

    configure_custom_colormaps({})

    class _State:
        def __init__(self):
            self.prefs = {"plot": {"custom_colormaps": {}}}
            self.saved = 0

        def save_prefs(self):
            self.saved += 1

    class _EditorResult:
        replacing_name = None

        @staticmethod
        def result_name():
            return "My LUT"

        @staticmethod
        def result_stops():
            return [
                [0.0, [0, 10, 30, 255]],
                [1.0, [240, 180, 20, 255]],
            ]

        @staticmethod
        def windowTitle():
            return "Create custom colormap"

    state = _State()
    applied = []
    dlg = LutDialog(
        on_levels_changed=lambda _lo, _hi: None,
        on_cmap_changed=lambda name, invert: applied.append((name, invert)),
        state=state,
    )
    qtbot.addWidget(dlg)
    dlg.load_image(
        pixels=np.linspace(0.0, 1.0, 100),
        data_lo=0.0,
        data_hi=1.0,
        lo=0.0,
        hi=1.0,
        cmap_name="hot",
        invert=False,
    )
    try:
        dlg._save_custom_colormap_dialog(_EditorResult())

        assert state.saved == 1
        assert "My LUT" in state.prefs["plot"]["custom_colormaps"]
        assert dlg._cmap_combo.currentText() == "My LUT"
        assert applied[-1] == ("My LUT", False)
    finally:
        configure_custom_colormaps({})


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


# ------------------------------------------------- render LUT: alpha / invert
def _render_window(qtbot):
    from minflux_viewer.core.app_state import AppState, default_prefs
    from minflux_viewer.core.dataset import build_localization_dataset
    from minflux_viewer.ui.render_window import RenderWindow

    state = AppState()
    state.prefs = default_prefs()
    state.save_prefs = lambda: None
    state.add_dataset(build_localization_dataset(
        name="lut-test",
        x_nm=np.array([0.0, 10.0, 20.0]),
        y_nm=np.array([0.0, 20.0, 10.0]),
        z_nm=np.array([0.0, 5.0, 10.0]),
    ))
    win = RenderWindow(state, dataset_idx=0)
    qtbot.addWidget(win)
    return state, win


def test_render_colormap_alpha_dims_the_channel(qtbot):
    """Alpha was dropped entirely; it now scales intensity, as solid colors do."""
    from minflux_viewer import colormaps as cm

    state, win = _render_window(qtbot)
    prefs: dict = {}
    cm.store_custom_colormap(prefs, "lut_alpha_full", [[0.0, [0, 0, 0, 255]], [1.0, [255, 0, 0, 255]]])
    cm.store_custom_colormap(prefs, "lut_alpha_faint", [[0.0, [0, 0, 0, 64]], [1.0, [255, 0, 0, 64]]])

    ramp = np.linspace(0.0, 1.0, 256, dtype=np.float32)
    full = win._map_norm_to_rgb(ramp, "lut_alpha_full")
    faint = win._map_norm_to_rgb(ramp, "lut_alpha_faint")

    assert not np.allclose(full, faint)
    assert faint[-1, 0] == pytest.approx(full[-1, 0] * 64 / 255, rel=0.02)


def test_render_invert_flips_the_ramp_and_the_background(qtbot):
    """Invert LUT was a no-op on the localization render."""
    state, win = _render_window(qtbot)
    ramp = np.linspace(0.0, 1.0, 256, dtype=np.float32)

    normal = win._map_norm_to_rgb(ramp, "hot", invert=False)
    inverted = win._map_norm_to_rgb(ramp, "hot", invert=True)
    assert not np.allclose(normal, inverted)
    assert np.allclose(inverted[0], normal[-1], atol=0.02)

    # Inverting also flips the page, so 'no signal' still matches the background.
    assert win._white_bg is False
    win._on_lut_invert_changed(True)
    assert win._white_bg is True
    assert win._channels[0]["lut_invert"] is True
    win._on_lut_invert_changed(False)
    assert win._white_bg is False
    assert win._channels[0]["lut_invert"] is False


def _close_stray_lut_dialogs():
    """Hide LUT dialogs left visible by an earlier test in the same session."""
    from minflux_viewer.ui.lut_dialog import LutDialog

    for widget in QApplication.topLevelWidgets():
        if isinstance(widget, LutDialog):
            try:
                widget.hide()
            except RuntimeError:
                continue
    QApplication.processEvents()

def test_only_one_lut_dialog_is_visible_app_wide(qtbot):
    from minflux_viewer.ui.main_window import MainWindow

    _close_stray_lut_dialogs()

    state, render = _render_window(qtbot)
    window = MainWindow(state)
    qtbot.addWidget(window)
    window.show()
    window._show_render(0)
    window._show_scatter(0)
    rwin = window._render_windows[0]
    swin = window._scatter_windows[0]
    rwin.show()
    swin.show()
    for _ in range(8):
        QApplication.processEvents()

    def visible():
        return [
            name for name, view in (("render", rwin), ("scatter", swin))
            if getattr(view, "_lut_dialog", None) is not None
            and view._lut_dialog.isVisible()
        ]

    for view, expected in ((rwin, "render"), (swin, "scatter"), (rwin, "render")):
        window._open_lut_on_view(view, 0)
        QApplication.processEvents()
        assert visible() == [expected]

    # The render window no longer closes other views' dialogs on mere focus.
    assert not hasattr(rwin, "_adopt_visible_lut_dialog")

    # And it is literally one object, not one-visible-of-several.
    from minflux_viewer.ui.lut_dialog import LutDialog
    instances = [w for w in QApplication.topLevelWidgets() if isinstance(w, LutDialog)]
    assert len(instances) == 1
    assert instances[0] is state._shared_lut_dialog

    window.close()
    QApplication.processEvents()


def test_shared_lut_dialog_is_closed_with_the_application(qtbot):
    """It is parentless, so it would otherwise outlive the main window."""
    from minflux_viewer.core.app_state import AppState, default_prefs
    from minflux_viewer.core.dataset import build_localization_dataset
    from minflux_viewer.ui.lut_dialog import LutDialog
    from minflux_viewer.ui.main_window import MainWindow

    _close_stray_lut_dialogs()

    state = AppState()
    state.prefs = default_prefs()
    state.save_prefs = lambda: None
    state.add_dataset(build_localization_dataset(
        name="lut-shutdown",
        x_nm=np.array([0.0, 10.0, 20.0]),
        y_nm=np.array([0.0, 20.0, 10.0]),
        z_nm=np.array([0.0, 5.0, 10.0]),
    ))
    window = MainWindow(state)
    qtbot.addWidget(window)
    window.show()
    window._show_render(0)
    rwin = window._render_windows[0]
    rwin.show()
    for _ in range(8):
        QApplication.processEvents()

    window._open_lut_on_view(rwin, 0)
    for _ in range(4):
        QApplication.processEvents()
    assert state._shared_lut_dialog is not None
    assert state._shared_lut_dialog.isVisible()

    window.close()
    for _ in range(10):
        QApplication.processEvents()

    assert state._shared_lut_dialog is None
    survivors = [
        w for w in QApplication.topLevelWidgets()
        if isinstance(w, LutDialog) and w.isVisible()
    ]
    assert survivors == []

    rwin.close()
    QApplication.processEvents()


def test_inverting_actually_repaints_the_background(qtbot):
    """Invert set white_bg but the image still read black: the white composite
    is itself an inversion, so applying the LUT flag again cancelled it out."""
    state, win = _render_window(qtbot)
    ramp = np.linspace(0.0, 1.0, 256, dtype=np.float32)

    # The white path must ignore the flag rather than double-invert.
    plain = win._channel_rgb_white(ramp, "hot", invert=False)
    flagged = win._channel_rgb_white(ramp, "hot", invert=True)
    assert np.allclose(plain, flagged)

    def zero_pixel():
        channel = win._channels[0]
        if win._white_bg:
            return np.asarray(win._channel_rgb_white(ramp, channel["lut"]))[0]
        return np.asarray(
            win._map_norm_to_rgb(ramp, channel["lut"],
                                 invert=channel.get("lut_invert", False))
        )[0]

    assert zero_pixel().mean() < 0.2                 # dark on the black page
    win._on_lut_invert_changed(True)
    assert win._white_bg is True
    assert zero_pixel().mean() > 0.8                 # now light on the white page
    win._on_lut_invert_changed(False)
    assert zero_pixel().mean() < 0.2


def test_lut_dialog_follows_focus_between_views(qtbot):
    from minflux_viewer.ui.main_window import MainWindow

    _close_stray_lut_dialogs()

    from minflux_viewer.core.app_state import AppState, default_prefs
    from minflux_viewer.core.dataset import build_localization_dataset

    state = AppState()
    state.prefs = default_prefs()
    state.save_prefs = lambda: None
    state.add_dataset(build_localization_dataset(
        name="lut-focus",
        x_nm=np.array([0.0, 10.0, 20.0]),
        y_nm=np.array([0.0, 20.0, 10.0]),
        z_nm=np.array([0.0, 5.0, 10.0]),
    ))
    window = MainWindow(state)
    qtbot.addWidget(window)
    window.show()
    window._show_render(0)
    window._show_scatter(0)
    rwin = window._render_windows[0]
    swin = window._scatter_windows[0]
    rwin.show()
    swin.show()
    for _ in range(8):
        QApplication.processEvents()

    def owner():
        return [
            name for name, view in (("render", rwin), ("scatter", swin))
            if getattr(view, "_lut_dialog", None) is not None
            and view._lut_dialog.isVisible()
        ]

    window._open_lut_on_view(rwin, 0)
    for _ in range(6):
        QApplication.processEvents()
    assert owner() == ["render"]
    geometry = rwin._lut_dialog.geometry()

    window._retarget_lut_dialog(swin)
    for _ in range(6):
        QApplication.processEvents()
    assert owner() == ["scatter"]
    assert swin._lut_dialog.geometry() == geometry, "should not jump"

    window._retarget_lut_dialog(rwin)
    for _ in range(6):
        QApplication.processEvents()
    assert owner() == ["render"]

    # Focusing a view must never summon a dialog that was not open.
    rwin._lut_dialog.close()
    QApplication.processEvents()
    window._retarget_lut_dialog(swin)
    for _ in range(6):
        QApplication.processEvents()
    assert owner() == []
