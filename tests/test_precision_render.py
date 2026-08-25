from __future__ import annotations

import numpy as np

from minflux_viewer.analysis.voronoi_density import build_projected_voronoi_field
from minflux_viewer.core.dataset import build_localization_dataset
from minflux_viewer.core.simulate import simulate_localizations
from minflux_viewer.core.spatial_grid import SpatialGrid
from minflux_viewer.ui.precision_render import (
    RENDER_METHOD_BASIC,
    RENDER_METHOD_BICUBIC,
    RENDER_METHOD_BILINEAR,
    RENDER_METHOD_FIXED_GAUSSIAN,
    RENDER_METHOD_HISTOGRAM,
    RENDER_METHOD_LABELS,
    RENDER_METHOD_PREFERENCE_ORDER,
    RENDER_METHOD_PRECISION_GAUSSIAN,
    RENDER_METHOD_VORONOI,
    PrecisionChannelData,
    ViewportScalarCache,
    VoronoiFieldCache,
    render_advanced_tile,
    render_bicubic_histogram,
    render_bilinear_histogram,
    render_gaussian_filtered,
    render_histogram,
    render_precision_gaussians,
    render_precision_tile,
    resolve_precision_xyz_nm,
    transform_precision_marginals,
)


def test_histogram_preserves_one_count_per_in_bounds_localization():
    image = render_histogram(
        np.array([0.1, 1.9, 2.0, np.nan]),
        np.array([0.1, 1.9, 1.0, 0.0]),
        (0.0, 2.0, 0.0, 2.0),
        (2, 2),
    )

    assert image.sum() == 2.0
    assert image[0, 0] == 1.0
    assert image[1, 1] == 1.0


def test_bilinear_histogram_preserves_mass_for_interior_points():
    image = render_bilinear_histogram(
        np.array([0.60, 1.30]),
        np.array([0.65, 1.35]),
        (0.0, 2.0, 0.0, 2.0),
        (4, 4),
    )

    assert np.isclose(image.sum(), 2.0)
    assert np.count_nonzero(image) > 2


def test_render_method_catalog_matches_preference_order():
    assert [RENDER_METHOD_LABELS[method] for method in RENDER_METHOD_PREFERENCE_ORDER] == [
        "Histogram",
        "Bilinear Histogram",
        "Smoothed histogram",
        "Fixed Gaussian",
        "Localization-precision Gaussian",
        "Voronoi density",
    ]


def _image_moments(image: np.ndarray, bounds):
    x0, x1, y0, y1 = bounds
    height, width = image.shape
    x = x0 + (np.arange(width) + 0.5) * (x1 - x0) / width
    y = y0 + (np.arange(height) + 0.5) * (y1 - y0) / height
    total = float(image.sum())
    mx = float((image.sum(axis=0) * x).sum() / total)
    my = float((image.sum(axis=1) * y).sum() / total)
    vx = float((image.sum(axis=0) * (x - mx) ** 2).sum() / total)
    vy = float((image.sum(axis=1) * (y - my) ** 2).sum() / total)
    return total, mx, my, vx, vy


def test_pixel_integrated_gaussian_conserves_mass_across_sampling():
    bounds = (-50.0, 50.0, -50.0, 50.0)
    args = (
        np.array([3.2]),
        np.array([-4.7]),
        np.array([3.0]),
        np.array([7.0]),
        bounds,
    )
    fine = render_precision_gaussians(*args, (400, 400))
    coarse = render_precision_gaussians(*args, (40, 40))

    assert np.isclose(fine.sum(), coarse.sum(), rtol=2e-5)
    assert np.isclose(fine.sum(), 1.0, rtol=2e-4)


def test_pixel_integrated_gaussian_preserves_anisotropic_width():
    bounds = (-50.0, 50.0, -50.0, 50.0)
    image = render_precision_gaussians(
        np.array([0.0]),
        np.array([0.0]),
        np.array([2.0]),
        np.array([8.0]),
        bounds,
        (400, 400),
    )
    total, mx, my, vx, vy = _image_moments(image, bounds)

    assert np.isclose(total, 1.0, rtol=2e-4)
    assert abs(mx) < 0.05
    assert abs(my) < 0.05
    assert np.isclose(np.sqrt(vx), 2.0, rtol=0.02)
    assert np.isclose(np.sqrt(vy), 8.0, rtol=0.02)


def test_halo_aware_precision_tiles_match_one_continuous_raster():
    x = np.array([0.0])
    y = np.array([0.0])
    sx = np.array([3.0])
    sy = np.array([5.0])
    channel = PrecisionChannelData(
        dataset_idx=0,
        x_nm=x,
        y_nm=y,
        depth_nm=np.zeros(1),
        sigma_x_nm=sx,
        sigma_y_nm=sy,
        sigma_depth_nm=sy,
        grid=SpatialGrid(x, y),
        source="test",
    )

    left, left_count = render_precision_tile(
        channel, (-20.0, 0.0, -20.0, 20.0), (200, 100)
    )
    right, right_count = render_precision_tile(
        channel, (0.0, 20.0, -20.0, 20.0), (200, 100)
    )
    whole = render_precision_gaussians(
        x, y, sx, sy, (-20.0, 20.0, -20.0, 20.0), (200, 200)
    )

    assert left_count == right_count == 1
    assert np.allclose(np.hstack([left, right]), whole, atol=2e-7)


def test_advanced_tile_supports_each_explicit_method():
    x = np.array([-2.0, 0.0, 2.0])
    y = np.array([-1.0, 0.0, 1.0])
    sigma = np.array([1.0, 2.0, 3.0])
    channel = PrecisionChannelData(
        dataset_idx=0,
        x_nm=x,
        y_nm=y,
        depth_nm=np.zeros(3),
        sigma_x_nm=sigma,
        sigma_y_nm=sigma,
        sigma_depth_nm=sigma,
        grid=SpatialGrid(x, y),
        source="test",
    )
    methods = (
        RENDER_METHOD_HISTOGRAM,
        RENDER_METHOD_BILINEAR,
        RENDER_METHOD_BICUBIC,
        RENDER_METHOD_BASIC,
        RENDER_METHOD_FIXED_GAUSSIAN,
        RENDER_METHOD_PRECISION_GAUSSIAN,
    )

    for method in methods:
        image, count = render_advanced_tile(
            channel,
            (-10.0, 10.0, -10.0, 10.0),
            (64, 64),
            render_method=method,
            fixed_sigma_nm=2.5,
        )
        assert count == 3
        assert image.shape == (64, 64)
        assert np.all(np.isfinite(image))
        assert image.max() > 0.0


def test_advanced_tile_samples_precomputed_voronoi_field():
    axis = np.arange(5, dtype=np.float64) * 10.0
    x, y = np.meshgrid(axis, axis)
    x, y = x.ravel(), y.ravel()
    sigma = np.ones_like(x)
    channel = PrecisionChannelData(
        dataset_idx=0,
        x_nm=x,
        y_nm=y,
        depth_nm=np.zeros_like(x),
        sigma_x_nm=sigma,
        sigma_y_nm=sigma,
        sigma_depth_nm=sigma,
        grid=SpatialGrid(x, y),
        source="test",
    )
    field = build_projected_voronoi_field(x, y)

    image, count = render_advanced_tile(
        channel,
        (0.0, 40.0, 0.0, 40.0),
        (64, 64),
        render_method=RENDER_METHOD_VORONOI,
        voronoi_field=field,
    )

    assert count == 25
    assert image.shape == (64, 64)
    assert np.all(np.isfinite(image))
    assert image.max() > 0.0


def test_bicubic_histogram_is_nonnegative_and_smoother_than_bilinear():
    # A single interior localization: bicubic must stay non-negative (its kernel
    # has negative lobes that are clamped) and spread over more pixels than the
    # 4-pixel bilinear splat.
    x = np.array([0.30])
    y = np.array([-0.20])
    bounds = (-2.0, 2.0, -2.0, 2.0)
    shape = (8, 8)
    bic = render_bicubic_histogram(x, y, bounds, shape)
    bil = render_bilinear_histogram(x, y, bounds, shape)
    assert bic.min() >= 0.0
    # Clamping the kernel's negative lobes inflates an isolated point's mass a
    # little (neighbouring lobes cancel over a dense field → ~0.3%).
    assert 0.95 <= bic.sum() <= 1.15
    assert np.count_nonzero(bic) > np.count_nonzero(bil)


def test_gaussian_filtered_conserves_mass_and_is_tile_seamless():
    # Fixed-sigma histogram+blur must (a) roughly conserve one count per interior
    # localization and (b) join seamlessly across a tile split, like the per-loc
    # renderer, because the halo padding feeds the blur near the tile edge.
    x = np.array([0.0])
    y = np.array([0.0])
    whole = render_gaussian_filtered(x, y, (-20.0, 20.0, -20.0, 20.0), (200, 200), 3.0, 5.0)
    assert np.isclose(whole.sum(), 1.0, rtol=1e-3)

    left = render_gaussian_filtered(x, y, (-20.0, 0.0, -20.0, 20.0), (200, 100), 3.0, 5.0)
    right = render_gaussian_filtered(x, y, (0.0, 20.0, -20.0, 20.0), (200, 100), 3.0, 5.0)
    assert np.allclose(np.hstack([left, right]), whole, atol=1e-6)


def test_scalar_lru_enforces_byte_limit_and_dataset_invalidation():
    cache = ViewportScalarCache(max_bytes=80, max_items=10)
    first_key = ("precision-tile", 4, "first")
    second_key = ("precision-tile", 4, "second")
    cache.put(first_key, np.ones(12, dtype=np.float32))
    cache.put(second_key, np.ones(12, dtype=np.float32))

    assert cache.nbytes <= 80
    assert cache.get(first_key) is None
    assert cache.get(second_key) is not None

    cache.remove_dataset(4)
    assert len(cache) == 0
    assert cache.nbytes == 0


def test_voronoi_field_lru_enforces_item_limit_and_dataset_invalidation():
    x, y = np.meshgrid(np.arange(5, dtype=float), np.arange(5, dtype=float))
    field = build_projected_voronoi_field(x.ravel(), y.ravel())
    cache = VoronoiFieldCache(max_items=1)
    first_key = ("voronoi-field", 4, "first")
    second_key = ("voronoi-field", 4, "second")
    cache.put(first_key, field)
    cache.put(second_key, field)

    assert len(cache) == 1
    assert cache.get(first_key) is None
    assert cache.get(second_key) is field
    assert cache.nbytes > 0

    cache.remove_dataset(4)
    assert len(cache) == 0


def test_precision_resolution_prefers_per_loc_then_per_trace():
    ds = build_localization_dataset(
        name="precision-test",
        x_nm=np.arange(4.0),
        y_nm=np.arange(4.0),
        z_nm=np.arange(4.0),
        tid=np.array([1, 1, 2, 2]),
        attrs={"sx_nm": np.array([1.0, 2.0, np.nan, np.nan])},
    )
    ds.derived["sigma_per_trace_nm"] = np.array(
        [[4.0, 5.0, 6.0], [7.0, 8.0, 9.0]]
    )
    ds.derived["sigma_trace_ids"] = np.array([1, 2])

    sigma, source = resolve_precision_xyz_nm(ds, 4)

    assert np.allclose(sigma[:2, 0], [1.0, 2.0])
    assert np.allclose(sigma[:2, 1:], [[5.0, 6.0], [5.0, 6.0]])
    assert np.allclose(sigma[2:], [[7.0, 8.0, 9.0], [7.0, 8.0, 9.0]])
    assert "per-localization sx_nm" in source
    assert "per-trace StdDev" in source


def test_rigid_rotation_projects_anisotropic_precision():
    sigma = np.array([[2.0, 5.0, 8.0]])
    matrix = np.eye(4)
    matrix[:2, :2] = np.array([[0.0, -1.0], [1.0, 0.0]])

    transformed = transform_precision_marginals(sigma, matrix)

    assert np.allclose(transformed, [[5.0, 2.0, 8.0]])


def test_requested_anisotropic_npc_stress_dataset_renders():
    coords, tid, attrs = simulate_localizations(
        "npc",
        n_points=300 * 16,
        locs_per_trace=2.0,
        precision_nm=2.0,
        params={
            "size_nm": 5000.0,
            "n_pores": 300,
            "diameter_nm": 113.0,
            "ring_sep_nm": 50.0,
            "field_curvature": 0.05,
        },
        dim=3,
        seed=20260723,
    )

    # Spread the curved NPC field through the requested 3000 nm test volume.
    rng = np.random.default_rng(20260723)
    z_offset = rng.uniform(-1500.0, 1500.0, int(tid.max()) + 1)
    coords[:, 2] += z_offset[tid]
    radial = np.hypot(coords[:, 0], coords[:, 1]) / 2500.0
    attrs["sx_nm"] = 1.5 + 3.0 * np.clip(radial, 0.0, 1.0)
    attrs["sy_nm"] = 2.0 + 4.0 * np.clip(np.abs(coords[:, 0]) / 2500.0, 0.0, 1.0)
    attrs["sz_nm"] = 4.0 + 8.0 * np.clip(np.abs(coords[:, 2]) / 1500.0, 0.0, 1.0)

    ds = build_localization_dataset(
        name="300-npc-precision-render-test",
        x_nm=coords[:, 0],
        y_nm=coords[:, 1],
        z_nm=coords[:, 2],
        tid=tid,
        attrs=attrs,
    )
    sigma, source = resolve_precision_xyz_nm(ds, len(coords))
    image = render_precision_gaussians(
        coords[:, 0],
        coords[:, 1],
        sigma[:, 0],
        sigma[:, 1],
        (-2500.0, 2500.0, -2500.0, 2500.0),
        (128, 128),
    )

    assert coords.shape[0] > 9000
    assert np.ptp(coords[:, 2]) > 2900.0
    assert np.ptp(sigma[:, 0]) > 2.0
    assert "per-localization" in source
    assert image.shape == (128, 128)
    assert np.all(np.isfinite(image))
    assert image.max() > 0.0


def test_precision_render_window_smoke(qtbot):
    from minflux_viewer.core.app_state import AppState
    from minflux_viewer.ui.precision_render_window import PrecisionRenderWindow

    rng = np.random.default_rng(7)
    state = AppState()
    sx = rng.uniform(1.0, 4.0, 500)
    sy = rng.uniform(2.0, 6.0, 500)
    sz = rng.uniform(4.0, 10.0, 500)
    dataset = build_localization_dataset(
        name="precision-window-smoke",
        x_nm=rng.normal(0.0, 50.0, 500),
        y_nm=rng.normal(0.0, 50.0, 500),
        z_nm=rng.normal(0.0, 100.0, 500),
        tid=np.repeat(np.arange(100), 5),
        attrs={"sx_nm": sx, "sy_nm": sy, "sz_nm": sz},
    )
    dataset.set_z_scaling_factor(0.7, source="test")
    state.add_dataset(dataset)
    window = PrecisionRenderWindow(state, dataset_idx=0)
    qtbot.addWidget(window)
    window.resize(500, 500)
    window.show()

    # The unified render window defaults to the production smoothed histogram,
    # while every method still uses the responsive precision tile path.
    qtbot.waitUntil(
        lambda: window._last_scalar_tile is not None,
        timeout=5000,
    )
    assert not window.windowTitle().endswith("[advanced]")
    assert not hasattr(window, "_render_method_combo")
    assert window._advanced_render_method == RENDER_METHOD_BASIC
    assert window._last_scalar_tile.shape[0] == 1
    assert np.all(np.isfinite(window._last_scalar_tile))
    assert not hasattr(window, "_overview_scalar")
    qtbot.waitUntil(
        lambda: "Smoothed histogram (" in window._info_label.text(),
        timeout=5000,
    )
    assert window._last_lod == -3

    x0, x1, y0, y1 = window._bounds_xy
    cx = (x0 + x1) / 2.0
    cy = (y0 + y1) / 2.0
    window._view_box.setRange(
        xRange=(cx - (x1 - x0) * 0.1, cx + (x1 - x0) * 0.1),
        yRange=(cy - (y1 - y0) * 0.1, cy + (y1 - y0) * 0.1),
        padding=0,
    )
    qtbot.waitUntil(
        lambda: "Smoothed histogram (" in window._info_label.text(),
        timeout=5000,
    )
    assert window._active_tile_keys
    assert window._precision_cache.nbytes <= window._CACHE_BYTES

    window._set_render_method(RENDER_METHOD_HISTOGRAM)
    qtbot.waitUntil(
        lambda: "Histogram (" in window._info_label.text(),
        timeout=5000,
    )
    window._set_render_method(RENDER_METHOD_FIXED_GAUSSIAN)
    assert window._fixed_sigma_for_orientation() == (5.0, 5.0)

    old_generation = window._precision_scheduler.generation
    window._set_orientation("XZ")
    x0, x1, y0, y1 = window._bounds_xy
    cx = (x0 + x1) / 2.0
    cy = (y0 + y1) / 2.0
    window._view_box.setRange(
        xRange=(cx - (x1 - x0) * 0.1, cx + (x1 - x0) * 0.1),
        yRange=(cy - (y1 - y0) * 0.1, cy + (y1 - y0) * 0.1),
        padding=0,
    )
    qtbot.waitUntil(
        lambda: (
            window._precision_scheduler.generation > old_generation
            and "Fixed Gaussian (" in window._info_label.text()
        ),
        timeout=5000,
    )
    channel = window._precision_channels[0]
    assert np.allclose(channel.sigma_x_nm, sx)
    assert np.allclose(channel.sigma_y_nm, sz * 0.7)


def test_precision_render_window_voronoi_supports_xy_xz_yz(qtbot):
    from minflux_viewer.core.app_state import AppState
    from minflux_viewer.ui.precision_render_window import PrecisionRenderWindow

    rng = np.random.default_rng(18)
    n = 300
    state = AppState()
    dataset = build_localization_dataset(
        name="projected-voronoi-window-test",
        x_nm=rng.normal(0.0, 80.0, n),
        y_nm=rng.normal(0.0, 60.0, n),
        z_nm=rng.normal(0.0, 110.0, n),
        tid=np.arange(n),
    )
    state.add_dataset(dataset)
    window = PrecisionRenderWindow(state, dataset_idx=0)
    qtbot.addWidget(window)
    window.resize(480, 480)
    window.show()
    window._set_render_method(RENDER_METHOD_VORONOI)

    for orientation in ("XY", "XZ", "YZ"):
        window._set_orientation(orientation)
        qtbot.waitUntil(
            lambda orientation=orientation: (
                window._last_frame_orientation == orientation
                and "Voronoi density (" in window._info_label.text()
                and window._last_scalar_tile is not None
                and bool(np.any(window._last_scalar_tile > 0.0))
            ),
            timeout=10_000,
        )
        assert window._last_scalar_tile.shape[0] == 1
        assert np.all(np.isfinite(window._last_scalar_tile))

    assert len(window._voronoi_cache) == 3
    generation = window._voronoi_scheduler.generation
    window._set_orientation("XY")
    qtbot.waitUntil(
        lambda: (
            window._last_frame_orientation == "XY"
            and "Voronoi density (" in window._info_label.text()
        ),
        timeout=5_000,
    )
    assert window._voronoi_scheduler.generation == generation


def test_voronoi_3d_action_opens_volume_with_true_3d_method(qtbot, monkeypatch):
    from PyQt6.QtWidgets import QWidget

    from minflux_viewer.core.app_state import AppState
    from minflux_viewer.ui import volume_window as volume_module
    from minflux_viewer.ui.precision_render_window import PrecisionRenderWindow

    rng = np.random.default_rng(19)
    state = AppState()
    state.add_dataset(
        build_localization_dataset(
            name="3d-voronoi-volume-link-test",
            x_nm=rng.normal(0.0, 40.0, 80),
            y_nm=rng.normal(0.0, 35.0, 80),
            z_nm=rng.normal(0.0, 25.0, 80),
            tid=np.arange(80),
        )
    )

    class FakeVolumeWindow(QWidget):
        def __init__(self, _state, _dataset_idx, *, render_method=None, parent=None, **_kwargs):
            super().__init__(parent)
            self.render_method = render_method

    monkeypatch.setattr(volume_module, "VolumeRenderWindow", FakeVolumeWindow)
    window = PrecisionRenderWindow(state, dataset_idx=0)
    qtbot.addWidget(window)
    window._advanced_render_method = RENDER_METHOD_VORONOI
    window._show_3d_volume_window()
    qtbot.addWidget(window._volume_window)
    assert window._volume_window.render_method == RENDER_METHOD_VORONOI
