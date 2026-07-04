"""Data Simulator plugin — generate synthetic MINFLUX-like sample datasets."""

from __future__ import annotations

from .. import PluginEntry, register


def _launch(state, parent=None) -> None:
    from ...ui import modeless
    from ...ui.data_simulator_window import DataSimulatorWindow

    win = DataSimulatorWindow(state, owner=parent)
    modeless.show_modeless(win, parent)


register(PluginEntry(
    name="Data Simulator",
    tooltip="Generate synthetic MINFLUX-like sample datasets (NPC / microtubule / "
            "sphere / … ) and manage the File › Open Sample Data presets.",
    launch=_launch,
))
