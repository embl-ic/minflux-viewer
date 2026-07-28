"""Spatial repeating-pattern analysis along a directed line-profile ROI."""

from __future__ import annotations

from .. import PluginEntry, register


def _launch(state, parent=None) -> None:
    from PyQt6.QtWidgets import QMessageBox

    idx = state.active_idx
    if idx is None or state.active_dataset is None:
        QMessageBox.information(
            parent,
            "Spatial Pattern Analysis",
            "Load and activate a localization dataset first.",
        )
        return
    if parent is None:
        QMessageBox.information(
            None,
            "Spatial Pattern Analysis",
            "Open the plugin from the main MINFLUX Viewer window.",
        )
        return

    view = None
    try:
        view = parent._active_coordinate_view()
    except Exception:
        pass
    if view is None:
        try:
            view = parent._render_window_for_dataset(idx)
        except Exception:
            pass
    if view is None:
        try:
            view = parent._show_render(idx)
        except Exception:
            pass
    try:
        is_coordinate_view = (
            view is not None
            and view.coordinate_view_box() is not None
            and view.roi_view_plane() in {"XY", "XZ", "YZ"}
        )
    except Exception:
        is_coordinate_view = False
    if not is_coordinate_view:
        QMessageBox.information(
            parent,
            "Spatial Pattern Analysis",
            "Open a render or scatter view in an XY, XZ, or YZ orientation first.",
        )
        return

    overlay = getattr(view, "_roi_overlay", None)
    record = overlay.active_open_line_record() if overlay is not None else None
    if record is None:
        QMessageBox.information(
            parent,
            "Spatial Pattern Analysis",
            "Draw a line, polyline, or freehand line, or select exactly one "
            "line-style ROI in the ROI Manager.",
        )
        return

    from ...ui.modeless import show_modeless
    from .spatial_line_pattern_window import SpatialLinePatternWindow

    win = SpatialLinePatternWindow(state, idx, view, owner=parent)
    show_modeless(win, parent)
    state.log(
        f"Spatial line-pattern analysis opened for '{state.active_dataset.name}'.",
        dataset_idx=idx,
    )


register(
    PluginEntry(
        name="Spatial Pattern Analysis along Line Profile",
        tooltip=(
            "Straighten a directed line ROI and analyze longitudinal repetition, "
            "signed side profiles, transverse displacement, FFT and autocorrelation."
        ),
        launch=_launch,
    )
)
