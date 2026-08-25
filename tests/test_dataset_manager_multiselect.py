"""Dataset Manager — multi-selection batch actions and the Close-button walk.

Two behaviours are covered:

* rows are multi-selectable, and a right-click inside the selection acts on the
  whole selection (close / duplicate / combine), never on the active dataset —
  selecting must not activate;
* the *Close* button re-highlights a neighbour after each close, so it can be
  pressed repeatedly: closing the top row walks down, closing any other row
  moves the highlight to the entry above it (so the bottom row walks up).
"""

from __future__ import annotations

import numpy as np
import pytest

pytest.importorskip("PyQt6")

from PyQt6.QtWidgets import QAbstractItemView, QApplication, QDialog

from minflux_viewer.core.app_state import AppState
from minflux_viewer.core.dataset import build_localization_dataset
from minflux_viewer.ui.dataset_manager import DatasetManager
from minflux_viewer.ui.main_window import MainWindow


@pytest.fixture(scope="module")
def _app():
    return QApplication.instance() or QApplication([])


def _dataset(name: str, offset: float = 0.0):
    rng = np.random.default_rng(0)
    xyz = rng.normal(size=(64, 3)) * 100.0 + offset
    return build_localization_dataset(
        name=name, x_nm=xyz[:, 0], y_nm=xyz[:, 1], z_nm=xyz[:, 2],
        source_version="simulation")


@pytest.fixture
def manager(_app):
    """A manager over four datasets, owned by a real (headless) main window."""
    win = MainWindow(AppState())
    win._state.prefs.setdefault("data", {})["show_render"] = False
    win._state.prefs["data"]["show_data_info"] = False
    for i in range(4):
        win._state.add_dataset(_dataset(f"ds{i}", i * 500.0))
    mgr = DatasetManager(win._state, win)
    try:
        yield mgr, win
    finally:
        mgr.close()
        win.close()


def _select(mgr, rows):
    """Ctrl-click *rows* (``selectRow`` alone would clear the previous one)."""
    from PyQt6.QtCore import QItemSelection, QItemSelectionModel

    mgr._table.setCurrentCell(rows[0], 1)      # before selecting: it re-selects
    model = mgr._table.selectionModel()
    model.clearSelection()
    for row in rows:
        left = mgr._table.model().index(row, 0)
        right = mgr._table.model().index(row, mgr._table.columnCount() - 1)
        model.select(QItemSelection(left, right),
                     QItemSelectionModel.SelectionFlag.Select)


# --- selection ------------------------------------------------------------

def test_rows_are_multi_selectable_and_selecting_does_not_activate(manager):
    mgr, win = manager
    assert mgr._table.selectionMode() is QAbstractItemView.SelectionMode.ExtendedSelection

    win._state.set_active(0)
    _select(mgr, [1, 3])

    assert mgr._selected_rows() == [1, 3]
    assert win._state.active_idx == 0          # selection is not activation


# --- Close button walk ----------------------------------------------------

def test_closing_the_top_row_keeps_the_highlight_at_the_top(manager):
    mgr, win = manager
    _select(mgr, [0])

    mgr._close_selected()

    assert [d.name for d in win._state.datasets] == ["ds1", "ds2", "ds3"]
    assert mgr._selected_rows() == [0]         # was ds2, now the top → walks down


def test_closing_the_bottom_row_walks_up(manager):
    mgr, win = manager
    _select(mgr, [3])

    mgr._close_selected()

    assert [d.name for d in win._state.datasets] == ["ds0", "ds1", "ds2"]
    assert mgr._selected_rows() == [2]         # the new bottom row (was 2nd last)


def test_closing_a_middle_row_highlights_the_one_above(manager):
    mgr, win = manager
    _select(mgr, [2])

    mgr._close_selected()

    assert [d.name for d in win._state.datasets] == ["ds0", "ds1", "ds3"]
    assert mgr._selected_rows() == [1]


def test_repeated_close_presses_empty_the_list_from_the_bottom(manager):
    """The point of the auto-highlight: press Close four times, no reselecting."""
    mgr, win = manager
    _select(mgr, [3])

    for _ in range(4):
        mgr._close_selected()

    assert win._state.datasets == []
    assert mgr._table.rowCount() == 0
    mgr._close_selected()                       # nothing left — must not raise


def test_last_close_leaves_no_selection_and_no_active_dataset(manager):
    mgr, win = manager
    _select(mgr, [0, 1, 2, 3])

    mgr._close_rows(mgr._selected_rows())

    assert win._state.datasets == []
    assert win._state.active_idx is None
    assert mgr._selected_rows() == []


# --- context-menu routing -------------------------------------------------

def _menu_entries(mgr, monkeypatch, row):
    """Right-click *row* and return the menu labels, without running anything."""
    from PyQt6.QtWidgets import QMenu

    labels: list[str] = []

    def _capture(menu, *_a, **_k):
        labels.extend(action.text() for action in menu.actions())
        return None                       # nothing chosen

    monkeypatch.setattr(QMenu, "exec", _capture)
    pos = mgr._table.visualRect(mgr._table.model().index(row, 1)).center()
    mgr._show_context_menu(pos)
    return labels


def test_right_click_inside_a_multi_selection_offers_the_batch_actions(manager, monkeypatch):
    mgr, _win = manager
    _select(mgr, [1, 2])

    assert _menu_entries(mgr, monkeypatch, 1) == [
        "Close all", "Duplicate all", "Combine as multi-channel overlay"]


#: The batch menu's own entries — their absence is what makes a menu "single-row".
#: The single-row menu's full contents are asserted in test_dataset_manager_actions.
_BATCH_ENTRIES = {"Close all", "Duplicate all", "Combine as multi-channel overlay"}


def test_right_click_outside_the_selection_keeps_the_single_row_menu(manager, monkeypatch):
    """Clicking a row that is not part of the multi-selection acts on that row."""
    mgr, _win = manager
    _select(mgr, [1, 2])

    entries = _menu_entries(mgr, monkeypatch, 3)
    assert not _BATCH_ENTRIES & set(entries)
    assert entries[:6] == [
        "Open file location", "", "Reset", "Save as…", "Close", "Duplicate",
    ]


def test_a_single_selection_keeps_the_single_row_menu(manager, monkeypatch):
    mgr, _win = manager
    _select(mgr, [2])

    entries = _menu_entries(mgr, monkeypatch, 2)
    assert not _BATCH_ENTRIES & set(entries)
    assert entries[:6] == [
        "Open file location", "", "Reset", "Save as…", "Close", "Duplicate",
    ]


# --- batch actions --------------------------------------------------------

def test_close_all_removes_every_selected_dataset(manager):
    mgr, win = manager
    _select(mgr, [0, 2])

    mgr._close_rows(mgr._selected_rows())

    assert [d.name for d in win._state.datasets] == ["ds1", "ds3"]
    assert mgr._table.rowCount() == 2


def test_duplicate_all_makes_one_copy_per_selected_dataset(manager):
    mgr, win = manager
    _select(mgr, [0, 2])

    mgr._duplicate_rows(mgr._selected_rows())

    names = [d.name for d in win._state.datasets]
    assert names[:4] == ["ds0", "ds1", "ds2", "ds3"]     # originals untouched
    assert len(names) == 6
    assert [d.metadata.get("duplicated_from_dataset") for d in win._state.datasets[4:]] \
        == ["ds0", "ds2"]


def test_combine_puts_only_the_selected_datasets_in_the_dialog(manager, monkeypatch):
    """The Combine dialog opened from a multi-selection lists exactly those
    datasets, pre-checked — not the whole loaded list."""
    from minflux_viewer.ui.channel_combine_dialog import ChannelCombineDialog

    seen: dict = {}

    class _Dialog(ChannelCombineDialog):
        def exec(self):
            seen["rows"] = list(self._rows)
            seen["checked"] = [c.isChecked() for c in self._checks]
            seen["selected"] = self.selected_rows()
            return QDialog.DialogCode.Rejected

    monkeypatch.setattr(
        "minflux_viewer.ui.channel_combine_dialog.ChannelCombineDialog", _Dialog)

    mgr, win = manager
    _select(mgr, [1, 3])
    mgr._combine_rows(mgr._selected_rows())

    assert seen["rows"] == [1, 3]
    assert seen["checked"] == [True, True]
    # selected_rows() reports the real dataset indices, not table positions.
    assert [r["dataset_idx"] for r in seen["selected"]] == [1, 3]


def test_combine_needs_at_least_two_selected(manager, monkeypatch):
    opened = []
    monkeypatch.setattr(
        "minflux_viewer.ui.channel_combine_dialog.ChannelCombineDialog",
        lambda *a, **k: opened.append(True))
    monkeypatch.setattr(
        "minflux_viewer.ui.main_window.QMessageBox.information",
        staticmethod(lambda *a, **k: None))

    mgr, win = manager
    win.combine_datasets_as_overlay([2])

    assert opened == []


# --- unrestricted combine still lists everything --------------------------

def test_full_combine_dialog_is_unchanged(_app):
    """``dataset_indices=None`` keeps the old behaviour: every dataset listed,
    the first two pre-checked."""
    from minflux_viewer.ui.channel_combine_dialog import ChannelCombineDialog

    state = AppState()
    for i in range(3):
        state.add_dataset(_dataset(f"ds{i}"))
    dlg = ChannelCombineDialog(state)
    try:
        assert dlg._rows == [0, 1, 2]
        assert [c.isChecked() for c in dlg._checks] == [True, True, False]
        assert [r["dataset_idx"] for r in dlg.selected_rows()] == [0, 1]
    finally:
        dlg.close()
