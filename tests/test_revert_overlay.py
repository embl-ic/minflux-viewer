"""Process › Channel › Revert Overlay to Original Dataset.

The inverse of a channel separation: the channel datasets close and the source
comes back -- re-activated when it is still open, rebuilt from the channels
otherwise. Datasets are found by key, never by comparing them, so a dataset
listed between the channels must not be mistaken for one of them.
"""

from __future__ import annotations

from types import SimpleNamespace

import numpy as np

from minflux_viewer.core.app_state import AppState
from minflux_viewer.core.dataset import build_localization_dataset
from minflux_viewer.ui.main_window import MainWindow


def _ds(name, x):
    return build_localization_dataset(name=name, x_nm=x, y_nm=x[::-1])


def _separated(*, keep_source=True):
    st = AppState()
    x = np.linspace(0.0, 100.0, 12)
    source = _ds("run", x)
    source.metadata["produced_overlays"] = ["g"]
    a, b = _ds("run (ch 1)", x[:6]), _ds("run (ch 2)", x[6:])
    for order, channel in enumerate((a, b)):
        channel.metadata["separated_from"] = "run"
        channel.state["overlay_id"] = "g"
        channel.state["overlay_order"] = order
    for ds in ((a, source, b) if keep_source else (a, b)):
        st.add_dataset(ds)
    st.set_active(0)
    shown = []
    owner = SimpleNamespace(_state=st, _show_render=shown.append,
                            _notify_view_state_changed=lambda: None,
                            _no_data_warning=lambda: None)
    return st, owner, shown, source


def test_revert_reactivates_the_source_that_is_still_open():
    st, owner, shown, source = _separated()

    MainWindow._revert_overlay_to_original(owner)

    assert len(st.datasets) == 1 and st.datasets[0] is source
    assert st.active_dataset is source and shown == [0]
    assert source.metadata["produced_overlays"] == []


def test_revert_rebuilds_the_source_when_it_was_closed():
    st, owner, shown, _source = _separated(keep_source=False)

    MainWindow._revert_overlay_to_original(owner)

    (rebuilt,) = st.datasets
    assert rebuilt.name == "run" and rebuilt.prop.num_loc == 12
    assert rebuilt.metadata["reverted_from_overlay"] == "g"
    assert st.active_dataset is rebuilt and shown == [0]
