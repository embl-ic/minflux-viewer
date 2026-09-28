"""The Tracking View: comets over a structure, and the things that make it play.

These drive the real window rather than its pieces, because the claims worth
pinning are about what ends up on screen: that the head is at the leading edge,
that a tail segment never joins two molecules, that the structure channel is
recognised, and that a frame stays cheap.
"""

from __future__ import annotations

import os
import sys
import time

import numpy as np
import pytest


@pytest.fixture
def _qt_app():
    pytest.importorskip("PyQt6")
    pytest.importorskip("pyqtgraph")
    if not os.environ.get("DISPLAY") and os.name != "nt" and sys.platform != "darwin":
        pytest.skip("No display available for Qt tests")
    from PyQt6.QtWidgets import QApplication
    return QApplication.instance() or QApplication(sys.argv)


def _pump(app, n: int = 6):
    for _ in range(n):
        app.processEvents()


def _wait_for_tracks(app, win, timeout_s: float = 5.0):
    deadline = time.perf_counter() + timeout_s
    while win._building_tracks and time.perf_counter() < deadline:
        app.processEvents()
        time.sleep(0.005)
    app.processEvents()
    assert not win._building_tracks, "trajectory indexing did not finish"


def _tracking_dataset(name="cargo", n_tracks=24, per=60, step=12.0, seed=0):
    """Traces that walk, timed like the reference file: 0.9 ms and ~12 nm a step."""
    from minflux_viewer.core.dataset import build_localization_dataset

    rng = np.random.default_rng(seed)
    parts, tid, tim = [], [], []
    for track in range(n_tracks):
        walk = np.cumsum(rng.normal(0, step, (per, 3)), axis=0)
        parts.append(walk + rng.uniform(-3000, 3000, 3))
        tid.append(np.full(per, track + 1, dtype=np.int64))
        # Each trace starts somewhere else in a long run, as a real one does.
        tim.append(rng.uniform(0, 300.0) + np.arange(per) * 9.0e-4)
    coords = np.vstack(parts)
    return build_localization_dataset(
        name=name, x_nm=coords[:, 0], y_nm=coords[:, 1], z_nm=coords[:, 2],
        tid=np.concatenate(tid), tim=np.concatenate(tim))


def _structure_dataset(name="scaffold", n_sites=300, per=6, seed=1):
    """Blinking fluorophores on fixed sites: traces that do not go anywhere."""
    from minflux_viewer.core.dataset import build_localization_dataset

    rng = np.random.default_rng(seed)
    centres = np.repeat(rng.uniform(-3000, 3000, (n_sites, 3)), per, axis=0)
    coords = centres + rng.normal(0, 4.0, centres.shape)
    tid = np.repeat(np.arange(1, n_sites + 1, dtype=np.int64), per)
    tim = rng.uniform(0, 300.0, coords.shape[0])
    return build_localization_dataset(
        name=name, x_nm=coords[:, 0], y_nm=coords[:, 1], z_nm=coords[:, 2],
        tid=tid, tim=tim)


def _state(*datasets, overlay=False):
    from minflux_viewer.core.app_state import AppState

    state = AppState()
    state.prefs.setdefault("data", {}).update(
        {"show_data_info": False, "show_render": False})
    for order, ds in enumerate(datasets, start=1):
        if overlay:
            ds.state.update({"overlay_id": "test", "render_group_id": "test",
                             "overlay_index": 1, "overlay_order": order})
        state.add_dataset(ds)
    return state


def _window(app, state, idx=0):
    from minflux_viewer.ui.tracking_window import TrackingWindow

    win = TrackingWindow(state, dataset_idx=idx)
    win.resize(760, 680)
    win.show()
    _wait_for_tracks(app, win)
    _pump(app, 4)
    return win


# ------------------------------------------------------------------ the basics


def test_worker_snapshot_coordinates_match_the_existing_display_path() -> None:
    from minflux_viewer.core.loader import attr_values_1d
    from minflux_viewer.core.roi_crop import display_coords
    from minflux_viewer.ui.tracking_window import (
        _snapshot_coordinates,
        _TrackChannelSnapshot,
    )

    ds = _tracking_dataset(n_tracks=2, per=5)
    ds.set_z_scaling_factor(0.73, source="test")
    matrix = np.eye(4)
    matrix[:3, 3] = (125.0, -80.0, 42.0)
    ds.state["overlay_transform"] = {"matrix_4x4": matrix.tolist()}
    snapshot = _TrackChannelSnapshot(
        dataset_idx=0,
        dataset_identity=id(ds),
        x_m=attr_values_1d(ds, "loc_x"),
        y_m=attr_values_1d(ds, "loc_y"),
        z_m=attr_values_1d(ds, "loc_z"),
        z_scaling_factor=ds.cali.z_scaling_factor,
        display_transform=ds.state["overlay_transform"],
        timestamps_s=None,
        trace_ids=None,
        keep=None,
    )

    np.testing.assert_allclose(
        _snapshot_coordinates(snapshot), display_coords(ds), atol=1e-9)


def test_it_opens_on_trace_relative_time_and_a_comet_length_tail(_qt_app):
    """Absolute time would be an empty field: the traces are spread over minutes."""
    win = _window(_qt_app, _state(_tracking_dataset()))
    try:
        axis = win._time_row.axis()
        assert axis.mode == "trace"
        # The axis spans the longest trace, not the acquisition.
        assert axis.hi < 1.0
        window = win._time_row.range()
        assert window is not None
        assert 0.0 < window[1] - window[0] < axis.span
    finally:
        win.close()
        _pump(_qt_app)


def test_tracking_view_defaults_come_from_tracking_preferences(_qt_app):
    state = _state(_tracking_dataset())
    state.prefs["tracking"].update({
        "default_axis_mode": "index",
        "default_projection": "XZ",
        "tail_fraction": 0.12,
        "tail_grow": True,
        "tail_color_mode": "Time",
        "tail_bands": 9,
        "tail_width": 3.5,
        "tail_head_opacity": 210,
        "tail_tip_opacity": 25,
        "head_symbol": "d",
        "head_size": 11,
        "backdrop_mode": "Scatter",
        "backdrop_level": "Strong",
        "playback_rate_hz": 24.0,
        "playback_loop": False,
        "role_displacement_nm": 1.0e6,
        "role_min_median_locs": 7,
    })
    win = _window(_qt_app, state)
    try:
        row_state = win._time_row.state()
        assert win._time_row.mode() == "index"
        assert win._axis == "XZ"
        assert win._tail_bands == 9
        assert win._tail_color_mode == "Time"
        assert win._tail_width == pytest.approx(3.5)
        assert win._tail_alpha_head == 210 and win._tail_alpha_tip == 25
        assert win._head_symbol == "d" and win._head_size == 11
        assert win._backdrop_mode == "Scatter" and win._backdrop_level == "Strong"
        assert row_state["grow"] is True
        assert row_state["loop"] is False
        assert row_state["rate_hz"] == pytest.approx(24.0)
        assert win._channels[0]["role"] == "Structure"
        assert "tracking threshold" in win._channels[0]["reason"]
        # XZ's vertical coordinate is Z, so it retains the mathematical direction.
        assert not win._plot.getViewBox().yInverted()
    finally:
        win.close()
        _pump(_qt_app)


def test_tracking_xy_direction_matches_the_scatter_origin_preference(_qt_app):
    state = _state(_tracking_dataset())
    win = _window(_qt_app, state)
    try:
        assert win._axis == "XY"
        assert win._plot.getViewBox().yInverted()

        win._set_axis("YZ")
        assert not win._plot.getViewBox().yInverted()  # vertical coordinate is Z
        win._set_axis("XY")
        assert win._plot.getViewBox().yInverted()

        state.prefs["plot"]["scatter_xy_origin"] = "bottom_left"
        win._apply_y_axis_direction()
        assert not win._plot.getViewBox().yInverted()
    finally:
        win.close()
        _pump(_qt_app)


def test_the_head_is_the_leading_edge_of_the_drawn_tail(_qt_app):
    """A comet head anywhere but the newest point is not a comet."""
    win = _window(_qt_app, _state(_tracking_dataset()))
    try:
        tracks = win._tracks[0]
        window = win._time_row.range()
        sel = tracks.window(*window)
        assert sel.n_traces > 0
        for head in sel.head_indices:
            code = tracks.trace_codes[head]
            same = sel.indices[tracks.trace_codes[sel.indices] == code]
            assert tracks.t[head] == pytest.approx(tracks.t[same].max())
    finally:
        win.close()
        _pump(_qt_app)


def test_moving_the_time_window_moves_the_comets(_qt_app):
    win = _window(_qt_app, _state(_tracking_dataset()))
    try:
        win._draw()
        first = win._head_item.getData()
        assert len(first[0]) > 0
        for _ in range(4):
            win._time_row._step_forward()
        win._draw()
        later = win._head_item.getData()
        assert len(later[0]) > 0
        assert len(first[0]) != len(later[0]) or not np.allclose(
            np.sort(first[0]), np.sort(later[0]))
    finally:
        win.close()
        _pump(_qt_app)


def test_a_tail_segment_never_joins_two_molecules(_qt_app):
    """Drawn as one polyline, so the pen has to lift between traces."""
    win = _window(_qt_app, _state(_tracking_dataset()))
    try:
        from minflux_viewer.core.tracks import band_edges

        win._draw()
        drawn = sum(1 for item in win._tail_items
                    if np.asarray(item.getData()[0]).size)
        assert drawn, "nothing was drawn"

        tracks = win._tracks[0]
        lo_t, hi_t = win._time_row.range()
        checked = 0
        for lo, hi, _age in band_edges(hi_t, hi_t - lo_t, 0.0, win._tail_bands):
            sel = tracks.window(lo, hi)
            if sel.size == 0:
                continue
            checked += 1
            codes = tracks.trace_codes[sel.indices]
            joins = np.flatnonzero(sel.connect[:-1])
            assert not np.any(codes[joins] != codes[joins + 1])
        assert checked, "no band held any points to check"
    finally:
        win.close()
        _pump(_qt_app)


# ------------------------------------------------------- structure and tracking


def test_a_scaffold_channel_and_a_cargo_channel_are_told_apart(_qt_app):
    """The two-colour case, and the reason each verdict was reached."""
    state = _state(_structure_dataset(), _tracking_dataset(), overlay=True)
    win = _window(_qt_app, state, idx=1)
    try:
        roles = {ch["name"]: ch["role"] for ch in win._channels}
        assert roles == {"scaffold": "Structure", "cargo": "Tracking"}
        reasons = {ch["name"]: ch["reason"] for ch in win._channels}
        assert "standing still" in reasons["scaffold"]
        assert "move" in reasons["cargo"]
        # Only the cargo gets comets; the scaffold is the backdrop.
        assert [ch["name"] for ch in win._comet_channels()] == ["cargo"]
        assert [ch["name"] for ch in win._visible_channels("Structure")] == ["scaffold"]
    finally:
        win.close()
        _pump(_qt_app)


def test_the_role_can_be_corrected_by_hand(_qt_app):
    """A default that cannot be overridden is a guess in disguise."""
    state = _state(_structure_dataset(), _tracking_dataset(), overlay=True)
    win = _window(_qt_app, state, idx=1)
    try:
        win._set_channel_role(0, "Tracking")
        assert sorted(ch["name"] for ch in win._comet_channels()) == ["cargo", "scaffold"]
        assert state.datasets[0].state["tracking_role"] == "Tracking"
        # Re-indexing after a filter/preference change must not discard it.
        win._request_track_build(
            reset_window=False, fit_view=False, rebuild_rows=True)
        _wait_for_tracks(_qt_app, win)
        assert win._channels[0]["role"] == "Tracking"
    finally:
        win.close()
        _pump(_qt_app)


def test_a_single_tracking_run_uses_its_own_points_as_the_backdrop(_qt_app):
    """With no scaffold there is still context: everywhere the tracks have been."""
    win = _window(_qt_app, _state(_tracking_dataset()))
    try:
        assert win._visible_channels("Structure") == []
        assert win._backdrop_points().shape[0] == win._tracks[0].n_points
    finally:
        win.close()
        _pump(_qt_app)


def test_the_channel_colour_survives_the_encoded_overlay_lut(_qt_app):
    """⚠ ``solid_color_rgb`` returns grey for ``solid:custom:#rrggbbaa`` without
    raising, which is how the tails came out grey instead of the channel colour."""
    from minflux_viewer.core.overlay import channel_rgb

    ds = _tracking_dataset()
    ds.state["overlay_lut"] = "solid:custom:#00ff00ff"
    win = _window(_qt_app, _state(ds))
    try:
        assert win._channels[0]["color"] == (0, 255, 0)
        assert channel_rgb("solid:custom:#ff0000ff") == (255, 0, 0)
    finally:
        win.close()
        _pump(_qt_app)


def test_time_coloured_tail_keeps_channel_identity_on_the_head(_qt_app):
    ds = _tracking_dataset(n_tracks=4, per=30)
    ds.state["overlay_lut"] = "solid:custom:#ff0000ff"
    state = _state(ds)
    state.prefs["tracking"]["tail_color_mode"] = "Time"
    win = _window(_qt_app, state)
    try:
        win._draw()
        pens = [item.opts.get("pen") for item in win._tail_items
                if np.asarray(item.getData()[0]).size]
        colors = {pen.color().name() for pen in pens if pen is not None}
        assert len(colors) > 1
        assert win._channels[0]["color"] == (255, 0, 0)
        head_brush = win._head_item.data[0]["brush"]
        assert head_brush.color().red() == 255
    finally:
        win.close()
        _pump(_qt_app)


def test_selecting_one_track_isolates_it_and_exposes_hover_statistics(_qt_app):
    win = _window(_qt_app, _state(_tracking_dataset(n_tracks=6, per=40)))
    try:
        win._draw()
        assert win._candidate_trace_ids.size
        dataset_idx = int(win._candidate_dataset_indices[0])
        trace_id = win._candidate_trace_ids[0]
        win._selected_track = (dataset_idx, trace_id)
        win._draw()

        assert np.all(win._candidate_dataset_indices == dataset_idx)
        assert np.all(win._candidate_trace_ids == trace_id)
        summary = win._track_summary(dataset_idx, trace_id)
        assert f"tid {trace_id}" in summary
        assert "median" in summary and "net" in summary
    finally:
        win.close()
        _pump(_qt_app)


def test_trace_viewer_and_tracking_view_share_selected_trace_and_playhead(_qt_app):
    from minflux_viewer.plugins.trace_viewer.trace_viewer_window import TraceViewerWindow

    state = _state(_tracking_dataset(n_tracks=5, per=30))
    tracking = _window(_qt_app, state)
    trace = TraceViewerWindow(state, 0)
    trace.show()
    try:
        assert trace._scatter_plot.getViewBox().yInverted()
        trace._orient_combo.setCurrentText("XZ")
        assert not trace._scatter_plot.getViewBox().yInverted()
        trace._orient_combo.setCurrentText("XY")
        state.tracking_playhead_changed.emit(
            state.datasets[0], 3, 0.012, object())
        _pump(_qt_app)
        assert tracking._selected_track == (0, 3)
        assert trace._selected_tid == 3
        times = np.asarray(trace._series["t"])
        assert times[trace._time_slider.value()] == pytest.approx(0.012, abs=5e-4)
    finally:
        trace.close()
        tracking.close()
        _pump(_qt_app)


def test_3d_mode_draws_the_same_live_trajectories_lazily(_qt_app):
    pytest.importorskip("OpenGL")
    win = _window(_qt_app, _state(_tracking_dataset(n_tracks=4, per=35)))
    try:
        win._set_axis("3D")
        axis = win._time_row.axis()
        win._time_row.set_window(axis.lo, axis.lo + axis.span * 0.5)
        win._draw()
        _pump(_qt_app)
        assert win._3d_view is not None
        assert win._view_stack.currentWidget() is win._3d_view
        assert np.asarray(win._3d_head.pos).shape[0] > 0
        assert any(np.asarray(item.pos).shape[0] > 0 for item in win._3d_tail_items)
    finally:
        win.close()
        _pump(_qt_app)


def test_movie_export_is_a_deterministic_tiff_stack_with_sidecar(
    _qt_app, tmp_path,
):
    import tifffile

    win = _window(_qt_app, _state(_tracking_dataset(n_tracks=3, per=20)))
    target = tmp_path / "tracking.tif"
    try:
        win._begin_movie_export(target, frame_count=3, fps=12.0)
        deadline = time.perf_counter() + 5.0
        while win._movie_export is not None and time.perf_counter() < deadline:
            _qt_app.processEvents()
            time.sleep(0.005)
        assert win._movie_export is None
        frames = tifffile.imread(target)
        assert frames.shape[0] == 3 and frames.shape[-1] == 3
        assert target.with_suffix(".tif.json").is_file()
        assert not target.with_name("tracking.tif.partial").exists()
    finally:
        win.close()
        _pump(_qt_app)


def test_absolute_time_axis_covers_every_overlay_channel(_qt_app):
    first = _tracking_dataset(name="early", seed=10)
    second = _tracking_dataset(name="late", seed=11)
    first.attr["tim"] = np.asarray(first.attr["tim"]) - 1000.0
    second.attr["tim"] = np.asarray(second.attr["tim"]) + 1000.0
    win = _window(_qt_app, _state(first, second, overlay=True), idx=1)
    try:
        win._time_row._sync_mode_combo("absolute")
        win._on_time_mode_changed("absolute")
        axis = win._time_row.axis()
        expected_lo = min(track.t_min for track in win._tracks.values())
        expected_hi = max(track.t_max for track in win._tracks.values())
        assert axis.lo == pytest.approx(expected_lo)
        assert axis.hi == pytest.approx(expected_hi)
    finally:
        win.close()
        _pump(_qt_app)


# --------------------------------------------------------------- staying playable


def test_the_backdrop_is_not_rebuilt_for_every_frame(_qt_app):
    """The structure does not move, and rebuilding it cost 20 ms a frame.

    Measured before the cache: 49.7 ms per playback step on the reference file,
    against 5.1 ms after it.
    """
    win = _window(_qt_app, _state(_tracking_dataset()))
    try:
        calls = {"n": 0}
        original = win._draw_backdrop_now

        def counted():
            calls["n"] += 1
            original()

        win._draw_backdrop_now = counted
        win._draw()
        first = calls["n"]
        for _ in range(6):
            win._time_row._step_forward()
            win._draw()
        assert calls["n"] == first, "the backdrop was rebuilt while only time moved"

        # ...but a change it does depend on must rebuild it.
        win._set_backdrop("Scatter")
        assert calls["n"] > first
    finally:
        win.close()
        _pump(_qt_app)


def test_a_playback_step_stays_in_frame_budget(_qt_app):
    """Generous bound: the reference file measures ~5 ms, this asserts under 60."""
    win = _window(_qt_app, _state(_tracking_dataset(n_tracks=60, per=120)))
    try:
        win._draw()
        costs = []
        for _ in range(12):
            start = time.perf_counter()
            win._time_row._step_forward()
            win._draw()
            costs.append((time.perf_counter() - start) * 1e3)
        assert np.median(costs) < 60.0, f"median {np.median(costs):.1f} ms/frame"
    finally:
        win.close()
        _pump(_qt_app)


# -------------------------------------------------------- background indexing


def test_track_indexing_runs_off_the_gui_thread(_qt_app, monkeypatch):
    import threading

    from minflux_viewer.ui import tracking_window as module

    gui_thread = threading.get_ident()
    worker_threads = []
    original = module.build_track_set

    def observed(*args, **kwargs):
        worker_threads.append(threading.get_ident())
        return original(*args, **kwargs)

    monkeypatch.setattr(module, "build_track_set", observed)
    win = _window(_qt_app, _state(_tracking_dataset()))
    try:
        assert worker_threads
        assert all(identifier != gui_thread for identifier in worker_threads)
    finally:
        win.close()
        _pump(_qt_app)


def test_new_generation_cancels_and_rejects_an_older_result(_qt_app, monkeypatch):
    from minflux_viewer.ui.tracking_window import TrackingWindow

    queued = []
    monkeypatch.setattr(
        TrackingWindow, "_start_track_task", lambda _self, task: queued.append(task))
    state = _state(_tracking_dataset(n_tracks=4, per=20))
    win = TrackingWindow(state, dataset_idx=0)
    win.show()
    try:
        assert len(queued) == 1
        old_generation = win._track_generation
        # Materialize what the first generation would eventually return even
        # though its task is about to be cancelled.
        stale_result = queued[0]._work(lambda _stage: None)

        keep = np.zeros(state.datasets[0].prop.num_loc, dtype=bool)
        keep[:20] = True
        state.datasets[0].filter_mask = keep
        win._request_track_build(
            reset_window=False, fit_view=False, rebuild_rows=True)
        assert len(queued) == 2
        assert queued[0].is_cancelled

        # Run the newest task deterministically and let its queued signal land.
        queued[1].run()
        _pump(_qt_app, 4)
        assert not win._building_tracks
        current = win._tracks[0]
        assert current.n_points == 20

        # A non-cooperative/native worker can finish after cancellation. The
        # generation check, not timing luck, must make that result harmless.
        win._on_track_build_done(
            old_generation,
            stale_result,
            reset_window=True,
            fit_view=True,
            rebuild_rows=True,
        )
        assert win._tracks[0] is current
        assert win._tracks[0].n_points == 20
    finally:
        win.close()
        _pump(_qt_app)


def test_result_from_a_replaced_dataset_object_is_rejected(_qt_app, monkeypatch):
    from minflux_viewer.ui.tracking_window import TrackingWindow

    queued = []
    monkeypatch.setattr(
        TrackingWindow, "_start_track_task", lambda _self, task: queued.append(task))
    state = _state(_tracking_dataset(n_tracks=2, per=10))
    win = TrackingWindow(state, dataset_idx=0)
    win.show()
    try:
        result = queued[0]._work(lambda _stage: None)
        state.datasets[0] = _tracking_dataset(
            name="replacement", n_tracks=2, per=10, seed=8)

        win._on_track_build_done(
            win._track_generation,
            result,
            reset_window=True,
            fit_view=True,
            rebuild_rows=True,
        )

        assert not win._tracks
        assert not win._time_row.is_available()
        assert "source dataset changed" in win._info_label.text()
    finally:
        win.close()
        _pump(_qt_app)


def test_closing_cancels_and_detaches_a_pending_track_build(_qt_app, monkeypatch):
    from minflux_viewer.ui.tracking_window import TrackingWindow

    queued = []
    monkeypatch.setattr(
        TrackingWindow, "_start_track_task", lambda _self, task: queued.append(task))
    win = TrackingWindow(_state(_tracking_dataset(n_tracks=2, per=10)), dataset_idx=0)
    win.show()
    assert len(queued) == 1 and not queued[0].is_cancelled

    win.close()
    _pump(_qt_app)

    assert queued[0].is_cancelled
    assert win._track_tasks == {}


def test_a_dataset_with_no_time_says_so_rather_than_drawing_nothing(_qt_app):
    from minflux_viewer.core.dataset import build_localization_dataset

    rng = np.random.default_rng(3)
    ds = build_localization_dataset(
        name="no-time", x_nm=rng.uniform(0, 1000, 400),
        y_nm=rng.uniform(0, 1000, 400), z_nm=np.zeros(400))
    win = _window(_qt_app, _state(ds))
    try:
        # A synthesised tid still gives an index axis, which is a usable one.
        assert win._time_row.is_available()
        assert win._time_row.mode() in ("index", "trace", "absolute")
    finally:
        win.close()
        _pump(_qt_app)


# ------------------------------------------------------------ the simulated data


def test_the_tracking_simulations_produce_moving_timed_molecules() -> None:
    from minflux_viewer.core.simulate import (
        sim_kind,
        simulate_npc_tracking,
        simulate_tracking_shells,
    )
    from minflux_viewer.core.tracks import build_track_set, looks_like_tracking

    assert sim_kind("tracking_shells") == "tracking"
    assert sim_kind("npc_tracking_2ch") == "track_overlay"

    coords, tid, attrs = simulate_tracking_shells({"n_tracks": 20}, seed=3)
    assert "tim" in attrs and attrs["tim"].size == coords.shape[0]
    tracks = build_track_set(coords, tid, attrs["tim"])
    assert looks_like_tracking(tracks)[0]

    scaffold, cargo = simulate_npc_tracking({"n_pores": 6, "n_tracks": 12}, seed=3)
    assert scaffold["role"] == "structure" and "tim" not in scaffold["attrs"]
    assert cargo["role"] == "tracking" and "tim" in cargo["attrs"]
    still = build_track_set(scaffold["coords"], scaffold["tid"],
                            np.arange(scaffold["coords"].shape[0], dtype=float))
    moving = build_track_set(cargo["coords"], cargo["tid"], cargo["attrs"]["tim"])
    assert not looks_like_tracking(still)[0]
    assert looks_like_tracking(moving)[0]


def test_the_simulated_cargo_really_goes_through_the_pore() -> None:
    """The point of the two-colour design: translocation along the pore axis.

    The cargo must travel much further along the axis than it wanders across it,
    or the simulation is not modelling the event the experiment looks for.
    """
    from minflux_viewer.core.simulate import simulate_npc_tracking

    _scaffold, cargo = simulate_npc_tracking(
        {"n_pores": 4, "n_tracks": 20, "field_curvature": 0.0,
         "local_tilt_deg": 0.0, "abort_fraction": 0.0, "travel_nm": 200,
         "channel_radius_nm": 20}, seed=5)
    coords, tid = cargo["coords"], cargo["tid"]
    axial, lateral = [], []
    for track in np.unique(tid):
        pts = coords[tid == track]
        axial.append(float(np.ptp(pts[:, 2])))
        lateral.append(float(max(np.ptp(pts[:, 0]), np.ptp(pts[:, 1]))))
    assert np.median(axial) > 3 * np.median(lateral)
