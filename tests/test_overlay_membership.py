"""Overlay identity must describe a multi-dataset channel group."""

from types import SimpleNamespace

from minflux_viewer.core.overlay import (
    clear_overlay_assignment,
    is_multichannel_overlay,
)


def _dataset(group_id=None, order=1):
    state = {"overlay_order": order} if group_id else {}
    if group_id:
        state.update(overlay_id=group_id, render_group_id=group_id, overlay_index=3)
    return SimpleNamespace(state=state, metadata={})


def test_singleton_overlay_assignment_is_not_multichannel():
    state = SimpleNamespace(datasets=[_dataset("msr:one")])

    assert not is_multichannel_overlay(state, 0)


def test_two_datasets_with_same_group_are_multichannel_overlay():
    state = SimpleNamespace(datasets=[_dataset("msr:two", 1), _dataset("msr:two", 2)])

    assert is_multichannel_overlay(state, 0)
    assert is_multichannel_overlay(state, 1)


def test_clear_overlay_assignment_makes_single_import_standalone():
    ds = _dataset("msr:one")
    ds.state.update(
        overlay_lut="Red",
        render_channel_lut="Red",
        overlay_transform={"matrix_3x3": [[1, 0, 0], [0, 1, 0], [0, 0, 1]]},
    )
    ds.metadata.update(
        overlay_id="msr:one",
        overlay_channels=["only"],
        overlay_reference="only",
        transformed=True,
    )

    clear_overlay_assignment(ds)

    assert ds.state == {}
    assert ds.metadata == {}


def test_dataset_manager_status_does_not_label_singleton_as_overlay():
    from minflux_viewer.ui.main_window import MainWindow

    state = SimpleNamespace(datasets=[_dataset("msr:one")])
    window = SimpleNamespace(
        _state=state,
        _render_windows={},
        _scatter_windows={},
        _histogram_windows={},
        _attr_windows={},
        _filter_dlgs={},
    )

    assert MainWindow.dataset_view_status(window, 0) == "None"


def test_dataset_manager_status_keeps_multichannel_label():
    from minflux_viewer.ui.main_window import MainWindow

    state = SimpleNamespace(datasets=[_dataset("msr:two", 1), _dataset("msr:two", 2)])
    window = SimpleNamespace(
        _state=state,
        _render_windows={},
        _scatter_windows={},
        _histogram_windows={},
        _attr_windows={},
        _filter_dlgs={},
    )

    assert MainWindow.dataset_view_status(window, 0) == "Overlay 3"
