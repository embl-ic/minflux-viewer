"""Preferences coverage for the tracking time-axis page."""

from __future__ import annotations

import pytest

pytest.importorskip("PyQt6")

from PyQt6.QtWidgets import QApplication

from minflux_viewer.core.app_state import DEFAULT_PREFS, AppState
from minflux_viewer.ui.preferences_dialog import PreferencesDialog


@pytest.fixture(scope="module")
def _app():
    return QApplication.instance() or QApplication([])


def test_preference_categories_have_requested_order(_app) -> None:
    dialog = PreferencesDialog(AppState())
    try:
        labels = [dialog._page_list.item(row).text() for row in range(dialog._page_list.count())]
        assert labels == [
            "File",
            "Data",
            "Attributes",
            "Tracking",
            "Appearance",
            "MBM Handling",
            "Shortcuts",
            "Plugin",
        ]
    finally:
        dialog.close()


def test_tracking_precision_round_trip_and_reset(_app) -> None:
    state = AppState()
    state.prefs["tracking"] = {
        "timestamp_precision_value": 10.0,
        "timestamp_precision_unit": "ns",
    }
    dialog = PreferencesDialog(state)
    try:
        assert dialog._tracking_precision_value.value() == 10.0
        assert dialog._tracking_precision_unit.currentData() == "ns"

        dialog._tracking_precision_value.setValue(0.1)
        dialog._tracking_precision_unit.setCurrentIndex(
            dialog._tracking_precision_unit.findData("ms")
        )
        dialog._apply_widgets_to_draft()
        assert dialog._draft["tracking"] == {
            **DEFAULT_PREFS["tracking"],
            "timestamp_precision_value": 0.1,
            "timestamp_precision_unit": "ms",
        }

        dialog._page_list.setCurrentRow(3)
        dialog._reset_current_tab()
        assert dialog._draft["tracking"] == DEFAULT_PREFS["tracking"]
    finally:
        dialog.close()


def test_tracking_analysis_and_view_defaults_round_trip(_app) -> None:
    state = AppState()
    dialog = PreferencesDialog(state)
    try:
        dialog._tracking_materialize_attributes.setChecked(True)
        dialog._tracking_attribute_checks["step_angle"].setChecked(False)
        dialog._tracking_tail_fraction.setValue(12.5)
        dialog._tracking_tail_color_mode.setCurrentText("Time")
        dialog._tracking_default_projection.setCurrentText("3D")
        dialog._tracking_tail_bands.setValue(9)
        dialog._tracking_head_size.setValue(11)
        dialog._tracking_playback_rate.setValue(24.0)
        dialog._tracking_role_displacement.setValue(42.0)
        dialog._apply_widgets_to_draft()
        tracking = dialog._draft["tracking"]

        assert tracking["materialize_analysis_attributes"] is True
        assert "step_angle" not in tracking["analysis_attributes"]
        assert tracking["tail_fraction"] == pytest.approx(0.125)
        assert tracking["tail_color_mode"] == "Time"
        assert tracking["default_projection"] == "3D"
        assert tracking["tail_bands"] == 9
        assert tracking["head_size"] == 11
        assert tracking["playback_rate_hz"] == pytest.approx(24.0)
        assert tracking["role_displacement_nm"] == pytest.approx(42.0)
    finally:
        dialog.close()
