"""Optional region-focused CUDA 3D convolution plugin."""

from __future__ import annotations

from minflux_viewer.core.plugin_registry import PluginEntry, register


def _launch(state, parent=None):
    from minflux_viewer.ui.analysis.convolution_3d_window import Convolution3DWindow
    from minflux_viewer.ui.common.modeless import show_modeless

    idx = state.active_idx
    if idx is None or state.datasets[idx].prop.num_dim != 3:
        state.log("GPU 3D Convolution requires an active 3D localization dataset.", "WARNING")
        return
    show_modeless(Convolution3DWindow(state, idx, owner=parent), parent)


register(PluginEntry(name="GPU 3D Convolution…", launch=_launch,
                     tooltip="Region-focused XYZ matched filtering using an isolated CUDA Python environment. "
                             "Separate from the CPU 2D convolution tool; no automatic CPU fallback.",
                     keywords=("GPU", "CUDA", "3D", "convolution", "segmentation", "NPC")))
