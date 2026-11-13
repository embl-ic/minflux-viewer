"""The 3-D volume follows the advanced view's render method in both directions.

Reported: after Fixed Gaussian with a large sigma, switching the 2-D view back
to the localization-precision Gaussian or to a histogram left the 3-D view
showing the large-sigma blur. A multi-channel overlay volume blurred every
channel by the Fixed Gaussian sigma whatever the method was, so only Fixed
Gaussian ever looked right there.
"""

from __future__ import annotations

import numpy as np
import pytest

from minflux_viewer.core.app_state import AppState
from minflux_viewer.core.dataset import build_localization_dataset
from minflux_viewer.ui.rendering.precision_render import (
    RENDER_METHOD_FIXED_GAUSSIAN,
    RENDER_METHOD_HISTOGRAM,
    RENDER_METHOD_PRECISION_GAUSSIAN,
)


def _state(overlay: bool) -> AppState:
    rng = np.random.default_rng(31)
    state = AppState()
    for index, x_centre in enumerate((0.0, 120.0) if overlay else (0.0,)):
        n = 400
        ds = build_localization_dataset(
            name=f"volume-method-{index}",
            x_nm=rng.normal(x_centre, 60.0, n),
            y_nm=rng.normal(0.0, 50.0, n),
            z_nm=rng.normal(0.0, 40.0, n),
            tid=np.arange(n),
        )
        if overlay:
            ds.state["overlay_id"] = "volume-method-pair"
            ds.state["overlay_order"] = index + 1
        state.add_dataset(ds)
    return state


def _rebuild_after(volume, change) -> np.ndarray:
    """Make *change* in the 2-D view, check it scheduled a volume rebuild, and
    run that rebuild now (what the volume's single-shot timer would do)."""
    volume._rebuild_timer.stop()
    change()
    assert volume._rebuild_timer.isActive(), "the method change did not reach the volume"
    volume._rebuild_timer.stop()
    volume.refresh_from_dataset()
    return volume._payload.scalar.copy()


@pytest.mark.parametrize("overlay", [False, True], ids=["single", "overlay"])
def test_the_volume_drops_the_fixed_sigma_when_the_method_changes_back(qtbot, overlay):
    from minflux_viewer.ui.rendering.precision_render_window import PrecisionRenderWindow

    window = PrecisionRenderWindow(_state(overlay), dataset_idx=0)
    qtbot.addWidget(window)          # destroyed at teardown: it holds an OpenGL view
    window.resize(420, 420)
    window.show()
    window._set_render_method(RENDER_METHOD_PRECISION_GAUSSIAN)
    window._show_3d_volume_window()
    volume = window._volume_window
    assert volume is not None and volume._view is not None
    # A coarse grid keeps the large blur cheap; the behaviour does not depend on it.
    volume._xy_voxel_spin.setValue(10.0)
    volume._z_voxel_spin.setValue(10.0)
    volume._rebuild_timer.stop()
    volume.refresh_from_dataset()
    assert len(volume._volume_channel_ids) == (2 if overlay else 0)
    precision = volume._payload.scalar.copy()

    _rebuild_after(volume, lambda: window._set_render_method(RENDER_METHOD_FIXED_GAUSSIAN))
    window._apply_fixed_sigma_values(60.0, 60.0)        # rebuilds the volume at once
    large = volume._payload.scalar.copy()
    assert large.max() < 0.5 * precision.max(), "the large sigma never reached the volume"

    back = _rebuild_after(
        volume, lambda: window._set_render_method(RENDER_METHOD_PRECISION_GAUSSIAN))
    np.testing.assert_allclose(back, precision, rtol=1e-5, atol=1e-6)

    counts = _rebuild_after(
        volume, lambda: window._set_render_method(RENDER_METHOD_HISTOGRAM))
    assert np.array_equal(counts, np.round(counts)), "the histogram volume is blurred"
    assert int(counts.sum()) == volume._payload.n_locs
