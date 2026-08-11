"""Process › Channel… › Combine — what happens to the open views.

Combining datasets *in place* (``keep source dataset`` unchecked) turns the
selected datasets into channels of one overlay.  Render and scatter are
overlay-aware — a single window keyed by the anchor draws every channel — so
each member's own window becomes a stale duplicate, still titled and drawn as a
standalone dataset.  Nothing folded them, so one was always left behind looking
like the "Own" view the Dataset Manager no longer lists.

No dataset is ever closed by combining: it changes how they are viewed, not
what exists.

The folding itself is registry bookkeeping and is unit-tested without Qt
windows; one end-to-end test drives the real command.
"""

from __future__ import annotations

import numpy as np
import pytest

pytest.importorskip("PyQt6")

from PyQt6.QtWidgets import QApplication, QDialog

from minflux_viewer.core.app_state import AppState
from minflux_viewer.core.dataset import build_localization_dataset
from minflux_viewer.ui.main_window import MainWindow


@pytest.fixture(scope="module")
def _app():
    return QApplication.instance() or QApplication([])


# --- folding the members' coordinate views --------------------------------

class _Win:
    """Stand-in for an open viewer window: all the folding needs is close()."""

    def __init__(self):
        self.closed = False

    def close(self):
        self.closed = True


def _folder(render=(), scatter=()):
    """A MainWindow with only the two window registries populated."""
    win = MainWindow.__new__(MainWindow)
    win._render_windows = {i: _Win() for i in render}
    win._scatter_windows = {i: _Win() for i in scatter}
    return win


def test_members_own_views_are_closed_and_the_anchor_keeps_its_own():
    win = _folder(render=(0, 1, 2), scatter=(0, 1, 2))
    kept_render, kept_scatter = win._render_windows[0], win._scatter_windows[0]
    dropped = [win._render_windows[1], win._render_windows[2],
               win._scatter_windows[1], win._scatter_windows[2]]

    assert win._collapse_member_coordinate_views([0, 1, 2], 0) is True

    assert sorted(win._render_windows) == [0]
    assert sorted(win._scatter_windows) == [0]
    assert not kept_render.closed and not kept_scatter.closed
    assert all(w.closed for w in dropped)


def test_a_dataset_outside_the_overlay_is_untouched():
    win = _folder(render=(0, 1, 2))
    outsider = win._render_windows[0]

    win._collapse_member_coordinate_views([1, 2], 1)

    assert sorted(win._render_windows) == [0, 1]
    assert not outsider.closed


def test_a_scatter_is_wanted_when_only_a_member_had_one():
    """The member's scatter is closed, so the anchor must be given one —
    otherwise folding silently leaves the user with no scatter at all."""
    win = _folder(render=(1,), scatter=(1,))

    assert win._collapse_member_coordinate_views([0, 1], 0) is True
    assert win._scatter_windows == {}          # caller reopens it on the anchor


def test_no_scatter_is_wanted_when_none_was_open():
    win = _folder(render=(0, 1))

    assert win._collapse_member_coordinate_views([0, 1], 0) is False


def test_folding_is_a_no_op_when_the_members_have_no_views():
    """The *keep source dataset* path builds the overlay from fresh copies,
    which have no windows — the originals' views must survive untouched."""
    win = _folder(render=(0, 1), scatter=(0,))
    originals = [win._render_windows[0], win._render_windows[1],
                 win._scatter_windows[0]]

    assert win._collapse_member_coordinate_views([2, 3], 2) is False

    assert sorted(win._render_windows) == [0, 1]
    assert sorted(win._scatter_windows) == [0]
    assert not any(w.closed for w in originals)


# --- end to end -----------------------------------------------------------

def _dataset(name: str, offset: float):
    rng = np.random.default_rng(0)
    xyz = rng.normal(size=(200, 3)) * 100.0 + offset
    return build_localization_dataset(
        name=name, x_nm=xyz[:, 0], y_nm=xyz[:, 1], z_nm=xyz[:, 2],
        source_version="simulation")


def test_combining_in_place_leaves_one_overlay_view_not_several(_app, monkeypatch):
    """The reported bug: after combining, one dataset was still shown in its
    original standalone render view."""
    import minflux_viewer.ui.channel_combine_dialog as mod

    class _Dialog:
        def __init__(self, *a, **k):
            pass

        def exec(self):
            return QDialog.DialogCode.Accepted

        def session_state(self):
            return {}

        def selected_rows(self):
            return [{"dataset_idx": i, "order": i + 1, "lut": lut}
                    for i, lut in enumerate(("Red", "Green", "Blue"))]

        keep_source_check = type("_C", (), {"isChecked": staticmethod(lambda: False)})()
        align_combo = type("_C", (), {"currentText": staticmethod(
            lambda: "none (keep original coordinates)")})()

    monkeypatch.setattr(mod, "ChannelCombineDialog", _Dialog)

    win = MainWindow(AppState())
    try:
        for i in range(3):
            win._state.add_dataset(_dataset(f"ch{chr(65 + i)}", i * 500.0))
        for i in range(3):
            win._show_render(i)
            win._show_scatter(i)
        win._show_histogram(1)
        assert sorted(win._render_windows) == [0, 1, 2]

        win._show_channel_combine()

        # One overlay view showing every channel — not three, two of which still
        # draw a single dataset as if it were standalone.
        assert sorted(win._render_windows) == [0]
        assert [c.get("dataset_idx") for c in win._render_windows[0]._channels] == [0, 1, 2]
        assert "Overlay" in win._render_windows[0].windowTitle()
        assert sorted(win._scatter_windows) == [0]

        # A per-channel histogram is still meaningful, so it stays open.
        assert sorted(win._histogram_windows) == [1]

        # No dataset was closed; all three now read as the overlay.
        assert [d.name for d in win._state.datasets] == ["chA", "chB", "chC"]
        assert [win.dataset_view_status(i) for i in range(3)] == ["Overlay 1"] * 3
    finally:
        win.close()
