"""Only reachable tiled levels remain; finer zoom still uses direct rendering."""
from types import SimpleNamespace

import numpy as np
import pytest

from minflux_viewer.core.dataset_key import DatasetKey
from minflux_viewer.ui import render_config as config
from minflux_viewer.ui.tile_cache import PhysicalTileCache, TileKey


def key(lod):
    return TileKey(DatasetKey(1), 0, "XY", lod, 0, 0, (), None)


def test_only_reachable_tile_resolutions_are_configured():
    assert config.LOD_BREAKPOINTS_NM == [100.]
    assert config.RENDER_TILE_PX == {0: 50, 1: 100}


@pytest.mark.parametrize("pixel_nm,lod", [(1000., 0), (100., 0), (99.9, 1), (20., 1)])
def test_reachable_thresholds_and_tile_pixels_are_unchanged(pixel_nm, lod):
    assert config.lod_for_pixel_size(pixel_nm) == lod
    assert config.actual_pixel_size_nm(lod) == (100. if lod == 0 else 50.)


@pytest.mark.parametrize("lod", [-1, 2, 3, 4, 999])
def test_removed_tile_levels_are_rejected_before_caching_or_allocating(lod):
    with pytest.raises(ValueError, match="level"):
        config.render_tile_px(lod)
    with pytest.raises(ValueError, match="level"):
        key(lod)


def test_two_level_progressive_placeholder_preserves_pixels_and_identity():
    cache = PhysicalTileCache()
    cache.put(key(0), np.full((50, 50), 3., np.float32))
    placeholder = cache.get_placeholder(key(1), 100)
    assert placeholder.shape == (100, 100)
    assert placeholder.dtype == np.float32
    np.testing.assert_array_equal(placeholder, 3.)
    assert cache.get_placeholder(key(0), 50) is None
    assert len(cache) == 1


@pytest.mark.parametrize("pixel_nm", [.05, 1., 5., 19.99, 20., 100.])
def test_zoom_below_threshold_uses_direct_path_not_removed_tile_levels(monkeypatch, pixel_nm):
    from minflux_viewer.ui import render_window

    class TiledPathReached(Exception):
        pass

    calls = []

    def select(pixel):
        calls.append(("tiled", config.lod_for_pixel_size(pixel)))
        raise TiledPathReached

    monkeypatch.setattr(render_window, "lod_for_pixel_size", select)
    span = pixel_nm * render_window._RENDER_SIZE
    view = SimpleNamespace(
        _tile_compose_timer=SimpleNamespace(stop=lambda: None),
        _apply_y_axis_direction=lambda: None,
        _render_alignment_view_change=lambda: False,
        _render_mode="loc",
        _view_box=SimpleNamespace(viewRange=lambda: ((0., span), (0., span))),
        _render_direct=lambda *args: calls.append(("direct", args[-1])),
    )
    if pixel_nm < config.DIRECT_RENDER_THRESHOLD_NM:
        render_window.RenderWindow._render(view)
        assert calls == [("direct", pixel_nm)]
    else:
        with pytest.raises(TiledPathReached):
            render_window.RenderWindow._render(view)
        assert calls == [("tiled", 0 if pixel_nm >= 100. else 1)]


@pytest.mark.parametrize("mode,lod", [("loc", -1), ("loc", -3), ("image", 0)])
def test_queued_tile_composite_cannot_take_over_a_non_tiled_view(mode, lod):
    from minflux_viewer.ui.render_window import RenderWindow

    class NonTiled(SimpleNamespace):
        @property
        def _last_px_nm(self):
            raise AssertionError("tile code reached a non-tiled view")

    view = NonTiled(_last_tile_geometry=(0., 100., 0., 100.),
                    _last_lod=lod, _render_mode=mode)
    RenderWindow._recomposite_from_tiles(view)


@pytest.mark.parametrize("mode,lod", [("loc", -1), ("loc", -3), ("image", 0)])
def test_late_tile_result_may_be_cached_but_cannot_repaint_a_non_tiled_view(mode, lod):
    from minflux_viewer.ui.render_window import RenderWindow

    tile_key = key(1)
    tile = np.ones((100, 100), np.float32)
    seen = []
    view = SimpleNamespace(
        _channel_grids={tile_key.dataset_key: object()},
        _phys_tile_cache=PhysicalTileCache(), _pending_tile_keys={tile_key},
        _overlay_alignment_panel=None, _render_mode=mode, _last_lod=lod,
        _active_plane=lambda: "XY", _mask_versions={tile_key.dataset_key: 0},
        _last_tile_geometry=(0., 100., 0., 100.),
        _tile_compose_timer=SimpleNamespace(stop=lambda: seen.append("stop")),
        _recomposite_from_tiles=lambda: seen.append("paint"),
    )
    RenderWindow._on_tile_ready(view, tile_key, tile)
    assert seen == []
    assert view._phys_tile_cache.get(tile_key) is tile
    assert not view._pending_tile_keys
