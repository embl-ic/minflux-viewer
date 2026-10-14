"""The Combine dialog remembers its choices per dataset, not per position.

Stabilization Phase 3, slice 3.6. The dialog offers the previous run's
selection, channel order and LUTs again. They were remembered by position, so
after an earlier dataset closed they were offered for whichever dataset had
moved into that position. They are now held by dataset key (in memory only).
"""

from __future__ import annotations

import numpy as np

from minflux_viewer.core.app_state import AppState
from minflux_viewer.core.dataset import build_localization_dataset


def _dataset(name):
    x = np.linspace(0.0, 100.0, 12)
    return build_localization_dataset(name=name, x_nm=x, y_nm=x[::-1])


def test_remembered_choices_stay_with_their_datasets(qapp):
    from minflux_viewer.ui.channel_combine_dialog import ChannelCombineDialog

    state = AppState()
    for name in ("A", "B", "C"):
        state.add_dataset(_dataset(name))
    first = ChannelCombineDialog(state)
    try:
        # Combine B and C, C first, B in Magenta.
        for row, checked in enumerate((False, True, True)):
            first._checks[row].setChecked(checked)
        first._orders[1].setCurrentText("2")
        first._orders[2].setCurrentText("1")
        first._luts[1].setCurrentText("Magenta")
        remembered = first.session_state()
    finally:
        first.close()

    state.remove_dataset(0)                       # B and C move to 0 and 1

    again = ChannelCombineDialog(state, previous=remembered)
    try:
        assert [c.isChecked() for c in again._checks] == [True, True]
        assert [o.currentText() for o in again._orders] == ["2", "1"]
        assert again._luts[0].currentText() == "Magenta"
        assert [r["dataset_idx"] for r in again.selected_rows()] == [1, 0]
    finally:
        again.close()
