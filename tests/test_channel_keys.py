"""A view holds its channels by dataset key (slice 4.2).

An overlay view's channel list recorded each member's position, so closing an
earlier dataset left every later view pointing at the wrong datasets until the
main window told it to reload -- which rebuilt its channels, grids, colour and
track caches and re-rendered it from scratch. Channels now hold the member's
key: an earlier close changes nothing in a view, and a closed member leaves the
view on its own.
"""

from __future__ import annotations

from types import SimpleNamespace

import numpy as np
import pytest


def _dataset(name: str, offset: float, *, group: str | None = None, order: int = 1):
    from minflux_viewer.core.dataset import build_localization_dataset

    rng = np.random.default_rng(sum(map(ord, name)))
    n_tracks, per = 30, 10
    walks = np.cumsum(rng.normal(0.0, 12.0, (n_tracks, per, 3)), axis=1)
    xyz = (walks + rng.uniform(-3000.0, 3000.0, (n_tracks, 1, 3))).reshape(-1, 3)
    xyz += offset
    tid = np.repeat(np.arange(1, n_tracks + 1), per)
    tim = (rng.uniform(0.0, 30.0, (n_tracks, 1)) + np.arange(per) * 9.0e-4).ravel()
    ds = build_localization_dataset(
        name=name, x_nm=xyz[:, 0], y_nm=xyz[:, 1], z_nm=xyz[:, 2], tid=tid, tim=tim,
        source_version="simulation")
    if group:
        ds.state.update(overlay_id=group, render_group_id=group, overlay_order=order,
                        overlay_index=order)
    return ds


def _flush():
    from PyQt6.QtCore import QCoreApplication, QEvent

    for _ in range(3):
        QCoreApplication.processEvents()
        QCoreApplication.sendPostedEvents(None, QEvent.Type.DeferredDelete)


def _keys(view) -> list:
    return [ch["dataset_key"] for ch in view._channels]


def _counting(view, *names: str) -> dict:
    """Count calls of *view*'s methods *names* (the ones that reload a view)."""
    calls = dict.fromkeys(names, 0)
    for name in names:
        original = getattr(view, name)

        def wrapper(*args, _name=name, _original=original, **kwargs):
            calls[_name] += 1
            return _original(*args, **kwargs)

        setattr(view, name, wrapper)
    return calls


@pytest.fixture
def views(qapp, qtbot):
    """A standalone, then C, D and E as one overlay drawn by C's render,
    scatter and tracking views."""
    from minflux_viewer.core.app_state import AppState
    from minflux_viewer.ui.main_window import MainWindow

    state = AppState()
    state.prefs.setdefault("data", {}).update(
        compute_local_density=False, compute_loc_prec=False,
        estimate_z_scaling_factor=False, show_dataset_manager=False,
        show_data_info=False, show_render=False)
    state.prefs.setdefault("plot", {})["use_fixed_z_scaling_factor"] = False
    win = MainWindow(state)
    ds = {"A": _dataset("A", 0.0)}
    for order, name in enumerate("CDE", start=1):
        ds[name] = _dataset(name, 9000.0, group="grp", order=order)
    for name in "ACDE":
        state.add_dataset(ds[name])
    anchor = state.index_of_dataset(ds["C"])
    render = win._show_render(anchor)
    scatter = win._show_scatter(anchor)
    tracking = win._show_tracking_view(anchor)
    for view in (render, scatter, tracking):
        view.resize(400, 400)
    _flush()
    qtbot.waitUntil(lambda: not tracking._building_tracks and bool(tracking._tracks),
                    timeout=20_000)
    members = [state.key_of(ds[name]) for name in "CDE"]
    for view in (render, scatter, tracking):
        assert _keys(view) == members
    try:
        yield SimpleNamespace(win=win, state=state, ds=ds, render=render,
                              scatter=scatter, tracking=tracking, members=members)
    finally:
        for view in (tracking, scatter, render):
            try:
                view.close()
            except RuntimeError:
                pass
        _flush()
        win.close()
        _flush()


def test_an_earlier_close_leaves_every_overlay_view_as_it_was(views, qtbot):
    render, scatter, tracking = views.render, views.scatter, views.tracking
    render_calls = _counting(
        render, "_refresh_from_dataset", "_build_channels", "_schedule_render")
    scatter_calls = _counting(scatter, "_refresh", "_build_channels", "_redraw_current")
    tracking_calls = _counting(tracking, "_refresh_from_dataset", "_build_channels")
    grids = dict(render._channel_grids)
    view_range = render._image_view.getView().viewRange()
    tracks = dict(tracking._tracks)
    generation = tracking._track_generation

    views.win.close_datasets([views.state.index_of_dataset(views.ds["A"])])
    _flush()

    assert render_calls == dict.fromkeys(render_calls, 0)
    assert scatter_calls == dict.fromkeys(scatter_calls, 0)
    assert tracking_calls == dict.fromkeys(tracking_calls, 0)
    for view in (render, scatter, tracking):
        assert _keys(view) == views.members
    assert all(render._channel_grids[key] is grid for key, grid in grids.items())
    assert render._image_view.getView().viewRange() == view_range
    assert tracking._track_generation == generation
    assert all(tracking._tracks[key] is value for key, value in tracks.items())
    # Every member still finds the overlay's view, wherever it now sits.
    d_now = views.state.index_of_dataset(views.ds["D"])
    assert views.win._render_window_for_dataset(d_now) is render


def test_a_closed_member_leaves_every_overlay_view_by_itself(views, qtbot):
    render, scatter, tracking = views.render, views.scatter, views.tracking
    key_c, key_d, key_e = views.members
    # D is the member each view has selected (a click on its channel row),
    # while another dataset is active.
    for view in (render, scatter, tracking):
        view._dataset_key = key_d
    assert views.state.active_key not in (None, key_d)

    # Removed through the state, as every caller other than close_datasets
    # does: nothing but the views' own handlers updates them.
    views.state.remove_dataset(views.state.index_of(key_d))
    _flush()
    qtbot.waitUntil(lambda: not tracking._building_tracks, timeout=20_000)

    for view in (render, scatter, tracking):
        assert _keys(view) == [key_c, key_e]
        assert view._dataset_key == key_c, "the view lost its overlay with D"
    assert render._idx == views.state.index_of(key_c)
    assert not render.isHidden()
    assert key_d not in render._channel_grids
    assert {key.dataset_key for key in render._phys_tile_cache._cache} <= {key_c, key_e}
    assert key_d not in tracking._tracks
    assert set(tracking._tracks) == {key_c, key_e}


def test_the_volume_view_finds_its_channels_by_key(views):
    from minflux_viewer.ui.volume_window import VolumeRenderWindow

    render = views.render
    render._channels[1]["lut"] = "Magenta"
    views.win.close_datasets([views.state.index_of_dataset(views.ds["A"])])

    spec_state = render._volume_display_state()
    stand_in = SimpleNamespace(_display_state=spec_state)
    stand_in._display_channel_specs = (
        lambda: VolumeRenderWindow._display_channel_specs(stand_in))
    by_key = VolumeRenderWindow._display_channel_map(stand_in)
    assert list(by_key) == views.members
    assert by_key[views.members[1]]["lut"] == "Magenta"


# ------------------------------------------------------------ no Qt needed
def _state_with(*names):
    from minflux_viewer.core.app_state import AppState
    from minflux_viewer.core.dataset import build_localization_dataset

    state = AppState()
    for i, name in enumerate(names):
        rng = np.random.default_rng(i)
        state.datasets.append(build_localization_dataset(
            name=name, x_nm=rng.normal(size=50), y_nm=rng.normal(size=50),
            z_nm=rng.normal(size=50)))
    return state


def test_a_window_is_asked_which_datasets_its_channels_draw():
    from minflux_viewer.ui.dataset_binding import window_channel_keys, window_draws_dataset

    state = _state_with("a", "b", "c")
    key_a, key_b, key_c = (state.key_at(i) for i in range(3))
    view = SimpleNamespace(_channels=[
        {"dataset_key": key_b, "visible": True},
        {"dataset_key": key_c, "visible": False},
        {"kind": "image", "visible": True},           # not a dataset
    ])

    assert window_channel_keys(view) == [key_b, key_c]
    assert window_channel_keys(view, visible_only=True) == [key_b]
    assert window_channel_keys(SimpleNamespace()) == []
    assert window_draws_dataset(view, state, 1)
    assert window_draws_dataset(view, state, key_c)
    assert not window_draws_dataset(view, state, key_c, visible_only=True)
    assert not window_draws_dataset(view, state, 0)
    assert not window_draws_dataset(view, state, 7)     # no dataset there

    state.datasets.pop(0)                               # b and c move down
    assert window_draws_dataset(view, state, 0)
    assert not window_draws_dataset(view, state, key_a)


def test_plot_profile_localizations_follow_the_channel_keys():
    from minflux_viewer.core.roi_crop import plane_localizations, plane_localizations_version

    state = _state_with("a", "b")
    channels = [{"dataset_key": state.key_at(1), "visible": True}]
    token = plane_localizations_version(state, channels, "XY")
    assert plane_localizations(state, channels, "XY").shape == (50, 2)

    state.datasets.pop(0)                               # b moves down
    assert plane_localizations(state, channels, "XY").shape == (50, 2)
    assert plane_localizations_version(state, channels, "XY") == token

    state.datasets.pop(0)                               # b closes
    assert plane_localizations(state, channels, "XY").shape == (0, 2)
    assert plane_localizations_version(state, channels, "XY") == ("XY",)
