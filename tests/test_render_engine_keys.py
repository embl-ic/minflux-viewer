"""The render engine keys its caches by dataset, not by position (slice 4.1).

A render window's tiles, spatial grids and mask versions used to be keyed by
the dataset's position in the open list. Closing an earlier dataset moved every
later one down, so their windows re-rendered every tile; a dataset that took a
closed one's place in an overlay could be served the closed one's tiles; and a
closed dataset's tiles stayed in the cache until pushed out.
"""

from __future__ import annotations

import numpy as np
import pytest


def _dataset(name: str, seed: int, *, overlay: str | None = None):
    from minflux_viewer.core.dataset import build_localization_dataset

    rng = np.random.default_rng(seed)
    n = 2000
    ds = build_localization_dataset(
        name=name,
        # 25 x 20 um in a 400 px window is ~60 nm/px: the tiled path, with
        # few tiles (each arrival recomposites the view, which is slow).
        x_nm=rng.uniform(0.0, 25_000.0, n),
        y_nm=rng.uniform(0.0, 20_000.0, n),
        z_nm=rng.normal(0.0, 100.0, n),
    )
    if overlay is not None:
        ds.state["overlay_id"] = overlay
    return ds


def _state(*datasets):
    from minflux_viewer.core.app_state import AppState

    state = AppState()
    state.prefs.setdefault("data", {}).update(
        {"show_data_info": False, "show_render": False})
    for ds in datasets:
        state.add_dataset(ds)
    return state


def _settled(win) -> bool:
    return (not win._redraw_timer.isActive()
            and not win._scheduler._in_flight
            and len(win._phys_tile_cache) > 0)


def _window(qtbot, state, idx, cls=None):
    from minflux_viewer.ui.render_window import RenderWindow

    win = (cls or RenderWindow)(state, dataset_idx=idx)
    qtbot.addWidget(win)
    win.resize(400, 400)
    win.show()
    qtbot.waitUntil(lambda: _settled(win), timeout=20_000)
    qtbot.wait(100)                      # the deferred first-show re-fit
    qtbot.waitUntil(lambda: _settled(win), timeout=20_000)
    assert win._last_lod is not None, "the view must use the tiled path"
    return win


def _record_requests(win) -> list:
    requested: list = []
    original = win._scheduler.request

    def request(keys, *args, **kwargs):
        requested.extend(keys)
        return original(keys, *args, **kwargs)

    win._scheduler.request = request
    return requested


def test_tiles_survive_an_earlier_dataset_closing(qtbot):
    state = _state(_dataset("A", 1), _dataset("B", 2))
    win = _window(qtbot, state, 1)
    before = set(win._phys_tile_cache._cache)
    requested = _record_requests(win)

    state.remove_dataset(0)
    win._refresh_from_dataset()          # what the main window does for it today
    qtbot.waitUntil(lambda: _settled(win), timeout=20_000)

    assert requested == [], "an earlier close re-rendered tiles it already had"
    assert before <= set(win._phys_tile_cache._cache)
    assert {key.dataset_key for key in before} == {state.key_at(0)}


def test_a_closing_dataset_evicts_its_own_tiles(qtbot):
    state = _state(_dataset("A", 1, overlay="g"), _dataset("B", 2, overlay="g"))
    win = _window(qtbot, state, 0)
    key_a, key_b = state.key_at(0), state.key_at(1)
    assert {key.dataset_key for key in win._phys_tile_cache._cache} == {key_a, key_b}

    state.remove_dataset(1)

    assert {key.dataset_key for key in win._phys_tile_cache._cache} == {key_a}
    assert key_b not in win._channel_grids
    assert key_b not in win._channel_locs_xyz
    assert key_b not in win._mask_versions


def test_a_dataset_taking_a_closed_ones_place_gets_none_of_its_tiles(qtbot):
    state = _state(_dataset("A", 1, overlay="g"), _dataset("B", 2, overlay="g"))
    win = _window(qtbot, state, 0)

    state.remove_dataset(1)
    state.add_dataset(_dataset("C", 3, overlay="g"))       # B's old position
    requested = _record_requests(win)
    win._refresh_from_dataset()
    qtbot.waitUntil(lambda: _settled(win), timeout=20_000)

    key_c = state.key_at(1)
    c_tiles = {key for key in win._phys_tile_cache._cache if key.dataset_key == key_c}
    assert c_tiles, "the new channel was never drawn"
    assert c_tiles <= set(requested), "the new channel was served cached tiles"


def test_a_closing_dataset_leaves_the_advanced_view_too(qtbot):
    from minflux_viewer.ui.precision_render_window import PrecisionRenderWindow

    state = _state(_dataset("A", 1, overlay="g"), _dataset("B", 2, overlay="g"))
    win = PrecisionRenderWindow(state, dataset_idx=0)
    qtbot.addWidget(win)
    win.resize(400, 400)
    win.show()
    qtbot.waitUntil(lambda: len(win._precision_cache._items) > 0, timeout=20_000)
    key_b = state.key_at(1)
    assert key_b in win._precision_channels

    state.remove_dataset(1)

    assert key_b not in win._precision_channels
    assert key_b not in win._precision_channel_versions
    assert key_b not in win._channel_grids
    for cache in (win._precision_cache, win._ortho_precision_cache):
        assert not [key for key in cache._items if key[1] == key_b]


def test_a_tile_finishing_after_its_dataset_closed_is_not_kept(qtbot):
    from minflux_viewer.ui.tile_cache import TileKey

    state = _state(_dataset("A", 1, overlay="g"), _dataset("B", 2, overlay="g"))
    win = _window(qtbot, state, 0)
    key_b = state.key_at(1)
    late = TileKey(dataset_key=key_b, mask_version=0, orientation="XY", lod=1,
                   tile_row=0, tile_col=0, transform_key=(), depth_range=None)

    state.remove_dataset(1)
    win._on_tile_ready(late, np.ones((4, 4), dtype=np.float32))

    assert late not in win._phys_tile_cache._cache


def test_an_advanced_tile_finishing_after_its_dataset_closed_is_not_kept(qtbot):
    from minflux_viewer.ui.precision_render import PrecisionTileResult
    from minflux_viewer.ui.precision_render_window import PrecisionRenderWindow

    state = _state(_dataset("A", 1, overlay="g"), _dataset("B", 2, overlay="g"))
    win = PrecisionRenderWindow(state, dataset_idx=0)
    qtbot.addWidget(win)
    win.resize(400, 400)
    win.show()
    qtbot.waitUntil(lambda: len(win._precision_cache._items) > 0, timeout=20_000)
    key_b = state.key_at(1)
    generation = win._precision_scheduler.generation
    win._active_tile_generation = generation
    late = ("precision-tile", key_b, 0, "XY")

    state.remove_dataset(1)
    win._on_precision_tile_result(PrecisionTileResult(
        key=late, generation=generation, array=np.ones((4, 4), dtype=np.float32),
        count=16))

    assert win._precision_cache.get(late) is None


@pytest.mark.parametrize("field", ["dataset_key"])
def test_a_tile_key_names_its_dataset_by_key(field):
    from dataclasses import fields

    from minflux_viewer.ui.tile_cache import TileKey

    assert field in {f.name for f in fields(TileKey)}
    assert "dataset_id" not in {f.name for f in fields(TileKey)}
