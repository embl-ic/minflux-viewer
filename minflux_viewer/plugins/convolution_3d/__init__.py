"""Region-focused 3D (XYZ) convolution plugin, computed on the CPU."""

from __future__ import annotations

from minflux_viewer.core.plugin_registry import PluginEntry, register


def _launch(state, parent=None):
    from minflux_viewer.ui.analysis.convolution_3d_window import Convolution3DWindow
    from minflux_viewer.ui.common.modeless import show_modeless

    idx = state.active_idx
    if idx is None or state.datasets[idx].prop.num_dim != 3:
        state.log("3D Convolution requires an active 3D localization dataset.", "WARNING")
        return
    show_modeless(Convolution3DWindow(state, idx, owner=parent), parent)


register(PluginEntry(name="3D Convolution…", launch=_launch,
                     tooltip="Region-focused XYZ matched filtering of the active 3D dataset. "
                             "Separate from the 2D convolution tool (Analyze › Segmentation).",
                     keywords=("3D", "convolution", "matched filter", "segmentation", "NPC", "XYZ")))
