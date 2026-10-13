"""Manual overlay alignment: the preview must show exactly what Apply commits.

The reported bug: align channel 2 onto channel 1 in the render view, click
Apply, and channel 2 "disappears" -- the scatter then shows it still far from
channel 1. The stored transform was always the one typed; the PREVIEW lied.
It snapshotted the last exact frame once, but drew and warped it at whatever
geometry the latest render left behind, and any render while the panel was
open (the panel's own insertion resizing the view, a wheel zoom, a pan, Reset
View) changed that geometry. A by-eye alignment then committed a shift scaled
by the zoom ratio, landing the channel micrometres away.

A second failure on the same path: after a zoom, the exact render cropped the
raster to the new view, so a channel being brought in from outside the view
had no source pixels and vanished from the preview altogether.
"""

from __future__ import annotations

from types import SimpleNamespace

import numpy as np
import pytest

pytest.importorskip("PyQt6")
pytest.importorskip("pyqtgraph")

from minflux_viewer.core.overlay import sync_transform_decomposition  # noqa: E402

#: Red blobs, in nm; green is the same pattern shifted by GREEN_OFFSET.
BLOBS = ((2000.0, 2000.0), (6000.0, 6000.0))
GREEN_OFFSET = (1500.0, 0.0)


@pytest.fixture
def _qt_app(qapp):
    return qapp


def _blob_dataset(name: str, offset: tuple[float, float], seed: int):
    from minflux_viewer.core.dataset import build_localization_dataset

    rng = np.random.default_rng(seed)
    parts = [rng.normal((cx + offset[0], cy + offset[1]), 25.0, (1500, 2)) for cx, cy in BLOBS]
    xy = np.vstack(parts)
    ds = build_localization_dataset(
        name=name, x_nm=xy[:, 0], y_nm=xy[:, 1], z_nm=np.zeros(len(xy)),
    )
    ds.state.update(overlay_id="grp", render_group_id="grp")
    return ds


def _overlay_window(qtbot):
    from minflux_viewer.core.app_state import AppState
    from minflux_viewer.ui.precision_render_window import PrecisionRenderWindow

    state = AppState()
    state.save_prefs = lambda: None
    state.prefs.setdefault("data", {}).update(
        show_render=False, show_data_info=False,
        compute_local_density=False, compute_loc_prec=False,
    )
    red = _blob_dataset("red", (0.0, 0.0), 1)
    green = _blob_dataset("green", GREEN_OFFSET, 2)
    red.state.update(overlay_order=1, overlay_lut="Red", render_channel_lut="Red")
    green.state.update(overlay_order=2, overlay_lut="Green", render_channel_lut="Green")
    state.add_dataset(red)
    state.add_dataset(green)
    window = PrecisionRenderWindow(state, dataset_idx=0)
    qtbot.addWidget(window)
    window.resize(700, 700)
    window.show()
    qtbot.waitUntil(lambda: window._last_scalar_tile is not None, timeout=10000)
    qtbot.waitUntil(lambda: "(ready)" in window._info_label.text()
                    or "(cached)" in window._info_label.text(), timeout=10000)
    assert len(window._channels) == 2
    return state, window


def _centroid(raster: np.ndarray, geometry, *, near, radius=600.0):
    """Mass centroid (world nm) of *raster* within *radius* of *near*."""
    x0, x1, y0, y1 = geometry
    h, w = raster.shape
    xs = x0 + (np.arange(w) + 0.5) * (x1 - x0) / w
    ys = y0 + (np.arange(h) + 0.5) * (y1 - y0) / h
    gx, gy = np.meshgrid(xs, ys)
    mask = np.hypot(gx - near[0], gy - near[1]) <= radius
    weight = np.where(mask, np.asarray(raster, dtype=np.float64), 0.0)
    total = float(weight.sum())
    if total <= 0.0:
        return None, 0.0
    return (float((weight * gx).sum() / total), float((weight * gy).sum() / total)), total


def _displayed(window):
    """The RGBA actually on screen and the world rectangle it covers."""
    item = window._image_view.getImageItem()
    image = np.asarray(item.image, dtype=np.float64)
    if image.max() > 1.5:
        image = image / 255.0
    rect = item.mapRectToView(item.boundingRect())
    y0, y1 = sorted((rect.top(), rect.bottom()))
    return image, (rect.left(), rect.right(), y0, y1)


def _settle_preview(window, qtbot):
    """Let the alignment preview finish re-rendering, then draw it now."""
    qtbot.wait(400)
    qtbot.waitUntil(lambda: not getattr(window, "_alignment_tile_plans", {}), timeout=10000)
    window._request_overlay_alignment_preview(invalidate=True, immediate=True)


def _await_exact_render(window, qtbot, previous):
    """Wait for the exact render that replaces *previous* (the Apply frame).

    ⚠ The info label still reads "(ready)" from before the alignment began --
    no exact render runs while the panel is open -- so it alone proves nothing.
    """
    qtbot.waitUntil(
        lambda: window._last_scalar_tile is not previous
        and ("(ready)" in window._info_label.text() or "(cached)" in window._info_label.text()),
        timeout=10000,
    )


def _assert_preview_shows_green_on_red(window, target):
    image, geometry = _displayed(window)
    red_c, red_mass = _centroid(image[..., 0], geometry, near=target)
    green_c, green_mass = _centroid(image[..., 1], geometry, near=target)
    assert red_mass > 0.0, "red is not where the data puts it"
    assert green_mass > 0.0, "the preview does not show green where the move puts it"
    assert np.hypot(red_c[0] - target[0], red_c[1] - target[1]) < 60.0
    assert np.hypot(green_c[0] - target[0], green_c[1] - target[1]) < 60.0


def test_preview_stays_true_after_a_zoom_and_apply_lands_where_it_showed(_qt_app, qtbot):
    """Zoom onto red while aligning; green's matching blob is OUTSIDE the view.

    Moving green by exactly the true offset must SHOW it on red -- the old
    preview drew the alignment-start snapshot squeezed into the zoomed view
    (so green was shown somewhere else, if at all) and a user aligning by eye
    therefore typed the wrong move.
    """
    state, window = _overlay_window(qtbot)
    target = BLOBS[1]                                    # red blob, where green must go
    try:
        window._manual_align_channel(1)
        qtbot.wait(200)
        window._view_box.setRange(
            xRange=(target[0] - 900, target[0] + 900),
            yRange=(target[1] - 900, target[1] + 900), padding=0,
        )
        qtbot.wait(400)
        window._update_overlay_alignment_transform(1, -GREEN_OFFSET[0], -GREEN_OFFSET[1], 0.0)
        _settle_preview(window, qtbot)
        _assert_preview_shows_green_on_red(window, target)

        window._overlay_alignment_apply()
        applied = window._last_scalar_tile
        # No flash of the pre-Apply position: what is drawn until the exact
        # render lands already has green on red.
        flash_c, flash_mass = _centroid(applied[1], window._last_tile_geometry, near=target)
        assert flash_mass > 0.0
        assert np.hypot(flash_c[0] - target[0], flash_c[1] - target[1]) < 60.0

        _await_exact_render(window, qtbot, applied)
        exact_c, exact_mass = _centroid(
            window._last_scalar_tile[1], window._last_tile_geometry, near=target)
        assert exact_mass > 0.0, "green vanished from the render after Apply"
        assert np.hypot(exact_c[0] - target[0], exact_c[1] - target[1]) < 40.0

        matrix = np.asarray(state.datasets[1].state["overlay_transform"]["matrix_4x4"])
        assert matrix[0, 3] == pytest.approx(-GREEN_OFFSET[0], abs=1e-6)
        assert matrix[1, 3] == pytest.approx(-GREEN_OFFSET[1], abs=1e-6)
    finally:
        if window._overlay_alignment_panel is not None:
            window._overlay_alignment_cancel()


def test_reset_view_during_alignment_keeps_the_preview_true(_qt_app, qtbot):
    """The reporter's own gesture: zoomed in, start aligning, Reset View."""
    state, window = _overlay_window(qtbot)
    target = BLOBS[1]
    try:
        window._view_box.setRange(
            xRange=(target[0] - 900, target[0] + 900),
            yRange=(target[1] - 900, target[1] + 900), padding=0,
        )
        qtbot.waitUntil(lambda: "(ready)" in window._info_label.text()
                        or "(cached)" in window._info_label.text(), timeout=10000)
        window._manual_align_channel(1)
        qtbot.wait(150)
        window._reset_view()
        qtbot.wait(300)
        window._update_overlay_alignment_transform(1, -GREEN_OFFSET[0], -GREEN_OFFSET[1], 0.0)
        _settle_preview(window, qtbot)
        _assert_preview_shows_green_on_red(window, target)
        window._overlay_alignment_apply()
        _await_exact_render(window, qtbot, window._last_scalar_tile)
        exact_c, exact_mass = _centroid(
            window._last_scalar_tile[1], window._last_tile_geometry, near=target)
        assert exact_mass > 0.0
        assert np.hypot(exact_c[0] - target[0], exact_c[1] - target[1]) < 40.0
    finally:
        if window._overlay_alignment_panel is not None:
            window._overlay_alignment_cancel()


def test_a_rotated_preview_lands_where_the_committed_matrix_puts_it(_qt_app, qtbot):
    """Move + rotate, with a zoom in between; the preview frozen at Apply must
    place every green blob where the stored matrix sends it."""
    state, window = _overlay_window(qtbot)
    try:
        window._manual_align_channel(1)
        qtbot.wait(200)
        # Chosen so both green blobs stay inside the reset view.
        window._update_overlay_alignment_transform(1, -GREEN_OFFSET[0], 0.0, 0.0)
        window._update_overlay_alignment_transform(1, 0.0, 0.0, -5.0)
        window._view_box.setRange(xRange=(5000, 7000), yRange=(5000, 7000), padding=0)
        qtbot.wait(300)
        window._reset_view()
        _settle_preview(window, qtbot)
        window._overlay_alignment_apply()
        applied = window._last_scalar_tile
        preview = applied[1].copy()
        preview_geometry = window._last_tile_geometry

        matrix = np.asarray(state.datasets[1].state["overlay_transform"]["matrix_4x4"])
        for bx, by in BLOBS:
            source = np.array([bx + GREEN_OFFSET[0], by + GREEN_OFFSET[1], 0.0, 1.0])
            where = (matrix @ source)[:2]
            x0, x1, y0, y1 = preview_geometry
            assert x0 + 150 < where[0] < x1 - 150 and y0 + 150 < where[1] < y1 - 150
            centre, mass = _centroid(preview, preview_geometry, near=where, radius=200.0)
            assert mass > 0.0
            assert np.hypot(centre[0] - where[0], centre[1] - where[1]) < 80.0

        _await_exact_render(window, qtbot, applied)
        for bx, by in BLOBS:
            source = np.array([bx + GREEN_OFFSET[0], by + GREEN_OFFSET[1], 0.0, 1.0])
            where = (matrix @ source)[:2]
            centre, mass = _centroid(
                window._last_scalar_tile[1], window._last_tile_geometry, near=where, radius=200.0)
            assert mass > 0.0
            assert np.hypot(centre[0] - where[0], centre[1] - where[1]) < 40.0
    finally:
        if window._overlay_alignment_panel is not None:
            window._overlay_alignment_cancel()


def test_alignment_started_before_the_first_frame_still_previews(_qt_app, qtbot):
    """Dataset Information › Manual align can open a fresh view and start at
    once. No exact render runs while aligning, so with no first frame to
    snapshot the preview must render its sources itself, not stay blank."""
    from minflux_viewer.core.app_state import AppState
    from minflux_viewer.ui.precision_render_window import PrecisionRenderWindow

    state = AppState()
    state.save_prefs = lambda: None
    red = _blob_dataset("red", (0.0, 0.0), 1)
    green = _blob_dataset("green", GREEN_OFFSET, 2)
    red.state.update(overlay_order=1, render_channel_lut="Red")
    green.state.update(overlay_order=2, render_channel_lut="Green")
    state.add_dataset(red)
    state.add_dataset(green)
    window = PrecisionRenderWindow(state, dataset_idx=0)
    qtbot.addWidget(window)
    window.resize(700, 700)
    assert window._last_scalar_tile is None
    window._manual_align_channel(1)
    window.show()
    try:
        qtbot.waitUntil(lambda: len(window._overlay_alignment_sources) == 2, timeout=10000)
        window._update_overlay_alignment_transform(1, -GREEN_OFFSET[0], -GREEN_OFFSET[1], 0.0)
        _settle_preview(window, qtbot)
        _assert_preview_shows_green_on_red(window, BLOBS[1])
        window._overlay_alignment_apply()
        _await_exact_render(window, qtbot, window._last_scalar_tile)
        centre, mass = _centroid(window._last_scalar_tile[1], window._last_tile_geometry, near=BLOBS[1])
        assert mass > 0.0
        assert np.hypot(centre[0] - BLOBS[1][0], centre[1] - BLOBS[1][1]) < 40.0
    finally:
        if window._overlay_alignment_panel is not None:
            window._overlay_alignment_cancel()


def test_apply_with_source_tiles_in_flight_never_shows_a_blank_frame(_qt_app, qtbot):
    """Apply while the zoomed preview's source tiles are still rendering.

    Those late results used to land after the panel closed, be composed
    against the pre-alignment tile plan (whose cache entries Apply evicts) and
    produce an all-zero frame that the next render then used as its
    placeholder: the whole view went black for a moment after Apply.
    """
    import time

    from minflux_viewer.core.app_state import AppState
    from minflux_viewer.core.dataset import build_localization_dataset
    from minflux_viewer.ui.precision_render_window import PrecisionRenderWindow

    def busy(name, offset, seed):
        rng = np.random.default_rng(seed)
        parts = [rng.normal((cx + offset[0], cy + offset[1]), 25.0, (1500, 2)) for cx, cy in BLOBS]
        parts.append(rng.uniform(0, 8000, (6000, 2)))      # so many tiles have work
        xy = np.vstack(parts)
        ds = build_localization_dataset(name=name, x_nm=xy[:, 0], y_nm=xy[:, 1],
                                        z_nm=np.zeros(len(xy)))
        ds.state.update(overlay_id="grp", render_group_id="grp")
        return ds

    state = AppState()
    state.save_prefs = lambda: None
    red, green = busy("red", (0.0, 0.0), 1), busy("green", GREEN_OFFSET, 2)
    red.state.update(overlay_order=1, render_channel_lut="Red")
    green.state.update(overlay_order=2, render_channel_lut="Green")
    state.add_dataset(red)
    state.add_dataset(green)
    window = PrecisionRenderWindow(state, dataset_idx=0)
    qtbot.addWidget(window)
    window.resize(700, 700)
    window.show()
    qtbot.waitUntil(lambda: window._last_scalar_tile is not None
                    and "(ready)" in window._info_label.text(), timeout=10000)
    target = BLOBS[1]
    staged = 0
    for attempt in range(3):
        window._manual_align_channel(1)
        qtbot.wait(50)
        span = 900 + 37 * attempt                        # uncached tiles each time
        window._view_box.setRange(xRange=(target[0] - span, target[0] + span),
                                  yRange=(target[1] - span, target[1] + span), padding=0)
        window._update_overlay_alignment_transform(1, -GREEN_OFFSET[0] / 10, 0.0, 0.0)
        window._refresh_alignment_sources()
        if not window._alignment_tile_plans:
            # The tiles were already done, so this attempt cannot stage the
            # race (rare, under load). Try the next span; only fail if none can.
            window._overlay_alignment_cancel()
            continue
        staged += 1
        window._overlay_alignment_apply()
        last = window._last_scalar_tile
        sums = [float(last.sum())]
        end = time.monotonic() + 3.0
        while time.monotonic() < end:
            _qt_app.processEvents()
            current = window._last_scalar_tile
            if current is not last:
                last = current
                sums.append(float(current.sum()))
                if "(ready)" in window._info_label.text() or "(cached)" in window._info_label.text():
                    break
            time.sleep(0.001)
        assert len(sums) >= 2, "the exact render never landed"
        assert min(sums) > 0.0, f"attempt {attempt}: a blank frame was shown after Apply ({sums})"
    assert staged, "no attempt had source tiles in flight; the race was never staged"


def test_reset_button_after_a_long_move_brings_the_channel_back(_qt_app, qtbot):
    """Move far (the source is re-rendered over where it is brought in from),
    then the panel's Reset: the channel must reappear at its own position."""
    state, window = _overlay_window(qtbot)
    own = (BLOBS[1][0] + GREEN_OFFSET[0], BLOBS[1][1] + GREEN_OFFSET[1])
    try:
        window._manual_align_channel(1)
        qtbot.wait(200)
        window._update_overlay_alignment_transform(1, 6000.0, 0.0, 0.0)
        _settle_preview(window, qtbot)
        window._overlay_alignment_reset()
        _settle_preview(window, qtbot)
        image, geometry = _displayed(window)
        centre, mass = _centroid(image[..., 1], geometry, near=own)
        assert mass > 0.0, "after Reset the preview no longer shows the channel"
        assert np.hypot(centre[0] - own[0], centre[1] - own[1]) < 60.0
    finally:
        if window._overlay_alignment_panel is not None:
            window._overlay_alignment_cancel()


def test_apply_does_not_keep_a_channel_hidden_in_the_panel(_qt_app, qtbot):
    """The panel's checkboxes are a temporary aid; Apply commits the move."""
    state, window = _overlay_window(qtbot)
    window._manual_align_channel(1)
    qtbot.wait(100)
    window._overlay_alignment_visibility(1, False)          # look at red only
    window._update_overlay_alignment_transform(1, -GREEN_OFFSET[0], 0.0, 0.0)
    window._overlay_alignment_apply()
    assert window._channels[1]["visible"] is True


def test_the_projection_cannot_change_under_a_pending_move(_qt_app, qtbot):
    """A move made in XZ is X/Z; switching to XY used to commit it as X/Y."""
    state, window = _overlay_window(qtbot)
    try:
        window._set_orientation("XZ")
        qtbot.wait(300)
        window._manual_align_channel(1)
        qtbot.wait(150)
        window._update_overlay_alignment_transform(1, -GREEN_OFFSET[0], 250.0, 0.0)
        window._set_orientation("XY")                       # refused
        assert window._orientation == "XZ"
        window._reset_view()                                # re-fits, keeps XZ
        assert window._orientation == "XZ"
        window._overlay_alignment_apply()
        matrix = np.asarray(state.datasets[1].state["overlay_transform"]["matrix_4x4"])
        assert matrix[0, 3] == pytest.approx(-GREEN_OFFSET[0], abs=1e-6)
        assert matrix[1, 3] == pytest.approx(0.0, abs=1e-6)  # not Y...
        assert matrix[2, 3] == pytest.approx(250.0, abs=1e-6)  # ...but Z
        window._set_orientation("XY")                       # free again
        assert window._orientation == "XY"
    finally:
        if window._overlay_alignment_panel is not None:
            window._overlay_alignment_cancel()


def test_side_panes_never_apply_the_primary_panes_warp():
    """A side projection composed with a pending move must equal one without:
    the move is a warp in the PRIMARY pane's rectangle, meaningless in XZ/YZ."""
    from minflux_viewer.ui.render_window import RenderWindow

    tile = np.zeros((40, 60), dtype=np.float32)
    tile[10:14, 20:24] = 5.0
    channel = {"lut": "Green", "visible": True, "levels": (0.0, 5.0),
               "transform": {"dx_nm": 300.0, "dy_nm": 0.0, "angle": 4.0,
                             "anchor_x_nm": 10.0, "anchor_y_nm": 10.0}}
    seen = []

    class Owner(SimpleNamespace):
        def _compose_rgba(self, scalar):
            seen.append([dict(ch.get("transform") or {}) for ch in self._channels])
            return np.zeros(scalar.shape[1:] + (4,), dtype=np.float32)

    owner = Owner(_channels=[], _manual_levels=None, _last_scalar_tile=None)
    RenderWindow._compose_rgba_for(owner, tile[None], [channel], auto=False)
    assert seen == [[{}]]
    assert channel["transform"]["dx_nm"] == 300.0           # the live channel untouched


def test_alignment_warp_maps_between_grids_of_different_extent():
    from minflux_viewer.ui.render_window import RenderWindow

    # One bright pixel centred at world (25, 35) in a 0..100 x 0..100 grid.
    source = np.zeros((10, 10), dtype=np.float32)
    source[3, 2] = 1.0
    src_geom = (0.0, 100.0, 0.0, 100.0)
    out_geom = (0.0, 200.0, 0.0, 100.0)                  # twice as wide
    out = RenderWindow._alignment_warp(
        source, src_geom, out_geom, (20, 40),
        {"dx_nm": 100.0, "dy_nm": 20.0, "angle": 0.0},
        order=0, prefilter=False,
    )
    row, col = np.unravel_index(int(np.argmax(out)), out.shape)
    x = out_geom[0] + (col + 0.5) * 5.0
    y = out_geom[2] + (row + 0.5) * 5.0
    assert (x, y) == pytest.approx((125.0, 55.0), abs=2.5)

    # A rotation is about the anchor, as manual_alignment_matrix4 commits it.
    out = RenderWindow._alignment_warp(
        source, src_geom, src_geom, (10, 10),
        {"dx_nm": 0.0, "dy_nm": 0.0, "angle": 90.0, "anchor_x_nm": 50.0, "anchor_y_nm": 50.0},
        order=0, prefilter=False,
    )
    row, col = np.unravel_index(int(np.argmax(out)), out.shape)
    # (25, 35) rotated +90 deg about (50, 50) -> (65, 25).
    assert ((col + 0.5) * 10.0, (row + 0.5) * 10.0) == pytest.approx((65.0, 25.0), abs=5.0)

    # Same grid, no move: untouched.
    assert RenderWindow._alignment_warp(source, src_geom, src_geom, (10, 10), {},
                                        order=1, prefilter=False) is not None


def test_apply_refreshes_the_decomposed_record_fields():
    """Method text reads translation_nm / rotation_2x2; they must follow the matrix."""
    matrix = np.eye(4)
    theta = np.deg2rad(5.0)
    matrix[:2, :2] = [[np.cos(theta), -np.sin(theta)], [np.sin(theta), np.cos(theta)]]
    matrix[:3, 3] = (-150.0, 800.0, 4.0)
    record = {
        "matrix_4x4": matrix.tolist(),
        "translation_nm": [1.0, 2.0],
        "z_translation_nm": 0.0,
        "rotation_2x2": [[1.0, 0.0], [0.0, 1.0]],
        "method": "mbm xy rigid plus z translation",
    }
    sync_transform_decomposition(record)
    assert record["translation_nm"] == pytest.approx([-150.0, 800.0])
    assert record["z_translation_nm"] == pytest.approx(4.0)
    assert np.allclose(record["rotation_2x2"], matrix[:2, :2])
    # Fields a record does not carry are not invented.
    bare = {"matrix_4x4": matrix.tolist()}
    assert set(sync_transform_decomposition(bare)) == {"matrix_4x4"}


def test_a_lost_mouse_release_does_not_keep_dragging_the_channel(_qt_app, qtbot):
    from PyQt6.QtCore import QEvent, QPointF, Qt
    from PyQt6.QtGui import QMouseEvent
    from PyQt6.QtWidgets import QWidget

    from minflux_viewer.ui.overlay_alignment import OverlayAlignmentPanel

    view = QWidget()
    qtbot.addWidget(view)
    drags = []

    class Owner(QWidget):
        def _overlay_alignment_drag_view(self):
            return view

        def _overlay_alignment_view_delta(self, start, end):
            return float(end.x() - start.x()), float(end.y() - start.y())

        def _overlay_alignment_drag(self, index, dx, dy):
            drags.append((dx, dy))

        def _overlay_alignment_control_config(self):
            return {}

        def _overlay_alignment_status(self, _index):
            return ""

        def __getattr__(self, name):
            if name.startswith("_overlay_alignment"):
                return lambda *a, **k: None
            raise AttributeError(name)

    owner = Owner()
    qtbot.addWidget(owner)
    panel = OverlayAlignmentPanel(owner, [{"name": "a"}, {"name": "b"}], 1)

    def mouse(kind, x, buttons):
        button = Qt.MouseButton.LeftButton if kind != QEvent.Type.MouseMove else Qt.MouseButton.NoButton
        return QMouseEvent(kind, QPointF(x, 0.0), QPointF(x, 0.0), button, buttons,
                           Qt.KeyboardModifier.NoModifier)

    left = Qt.MouseButton.LeftButton
    panel.eventFilter(view, mouse(QEvent.Type.MouseButtonPress, 0.0, left))
    panel.eventFilter(view, mouse(QEvent.Type.MouseMove, 10.0, left))
    assert drags == [(10.0, 0.0)]
    # The release never arrived; the next hover has no button held.
    panel.eventFilter(view, mouse(QEvent.Type.MouseMove, 50.0, Qt.MouseButton.NoButton))
    panel.eventFilter(view, mouse(QEvent.Type.MouseMove, 90.0, Qt.MouseButton.NoButton))
    assert drags == [(10.0, 0.0)]
    panel.detach()


def test_raising_the_render_window_does_not_cancel_an_alignment():
    """Ctrl+R on an open render used to call _refresh_from_dataset, which
    cancels an in-progress manual alignment."""
    from minflux_viewer.ui.main_window import MainWindow

    calls = []
    window = SimpleNamespace(
        _overlay_alignment_panel=object(),
        _refresh_from_dataset=lambda: calls.append("refresh"),
        show=lambda: calls.append("show"),
        raise_=lambda: None,
        activateWindow=lambda: None,
    )
    owner = SimpleNamespace(
        _state=SimpleNamespace(active_dataset=object(), active_idx=0),
        _render_windows={0: window},
        _render_window_for_dataset=lambda _idx: None,
        _install_window_shortcuts=lambda _win: None,
        _notify_view_state_changed=lambda: None,
    )
    assert MainWindow._show_render(owner, 0) is window
    assert calls == ["show"]

    window._overlay_alignment_panel = None
    MainWindow._show_render(owner, 0)
    assert "refresh" in calls
