from __future__ import annotations

import sys

import numpy as np
import pytest


@pytest.fixture
def _app():
    pytest.importorskip("PyQt6")
    from PyQt6.QtWidgets import QApplication

    return QApplication.instance() or QApplication(sys.argv)


def test_main_window_creates_filtered_time_channel_overlay(_app, tmp_path):
    from minflux_viewer.core.app_state import AppState
    from minflux_viewer.core.dataset import build_localization_dataset
    from minflux_viewer.core.time_channels import TimeWindow
    from minflux_viewer.ui.main_window import MainWindow

    state = AppState()
    state.prefs.setdefault("data", {}).update({
        "show_data_info": False,
        "show_render": False,
        "show_attr_plot": False,
        "show_histogram": False,
        "show_scatter": False,
        "compute_rimf": False,
        "compute_loc_prec": False,
        "compute_local_density": False,
    })
    window = MainWindow(state)
    try:
        tim = np.arange(7, dtype=float)
        source = build_localization_dataset(
            name="source.mat",
            folder=str(tmp_path),
            x_nm=tim,
            y_nm=tim,
            tim=tim,
            tid=np.arange(tim.size),
            prefs=state.prefs,
        )
        source.filter_mask = np.array(
            [True, True, False, True, True, True, True], dtype=bool
        )
        state.add_dataset(source)

        shown: list[int] = []
        window._show_render = lambda idx=None: shown.append(idx)
        window._notify_view_state_changed = lambda: None
        result = window.apply_time_channel_separation(
            0,
            [
                TimeWindow("round 1", 0.0, 3.0, "Red"),
                TimeWindow("round 2", 3.0, 6.0, "Green"),
            ],
        )

        assert result is True
        assert len(state.datasets) == 3
        assert source.state.get("overlay_id") is None
        first, second = state.datasets[1:]
        assert first.prop.num_loc == 2
        assert second.prop.num_loc == 4
        np.testing.assert_array_equal(
            first.filter_mask,
            [True, True],
        )
        np.testing.assert_array_equal(
            second.filter_mask,
            [True, True, True, True],
        )
        np.testing.assert_array_equal(first.attr["tim"], [0.0, 1.0])
        np.testing.assert_array_equal(second.attr["tim"], [3.0, 4.0, 5.0, 6.0])
        assert len(first.mfx_raw) == 0
        assert len(second.mfx_raw) == 0
        assert first.state["overlay_id"] == second.state["overlay_id"]
        assert first.state["overlay_order"] == 1
        assert second.state["overlay_order"] == 2
        assert first.state["overlay_lut"] == "Red"
        assert second.state["overlay_lut"] == "Green"
        assert first.state["filter_specs"][-1]["hi_inc"] is False
        assert second.state["filter_specs"][-1]["hi_inc"] is True
        assert state.active_idx == 1
        assert shown == [1]

        menu_labels = [
            action.text() for action in window.menuProcessChannel.actions()
        ]
        assert "Separate Channels from Time Windows" in menu_labels
    finally:
        window.close()
        _app.processEvents()
