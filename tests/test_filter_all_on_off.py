"""The Filter dialog's two bulk buttons act on the ROWS, not behind their back.

They used to be *Apply all* and *Reset all filters*, and reset was the bug:
it cleared ``filter_specs`` and the mask **directly** and never touched the
table, so every ``On`` box stayed ticked over ungated data -- and because
``_apply_all`` reads those boxes, the next edit to any row silently brought
every filter back. They are now *All filters On* / *All filters Off*, which
tick and untick the rows and then apply once, so the table and the linked views
can never disagree.
"""

from __future__ import annotations

import numpy as np
import pytest


@pytest.fixture(scope="module")
def _app():
    pytest.importorskip("PyQt6")
    from PyQt6.QtWidgets import QApplication

    return QApplication.instance() or QApplication([])


def _dialog_with_two_rows():
    """A dialog over a small dataset with two enabled, genuinely gating rows."""
    from minflux_viewer.core.app_state import AppState
    from minflux_viewer.core.dataset import build_localization_dataset
    from minflux_viewer.ui.filter_dialog import FilterDialog

    rng = np.random.default_rng(0)
    ds = build_localization_dataset(
        name="A", x_nm=rng.random(200) * 1000, y_nm=rng.random(200) * 1000,
        z_nm=rng.random(200) * 100, tid=np.repeat(np.arange(40), 5))
    state = AppState()
    state.add_dataset(ds)
    dialog = FilterDialog(state, dataset_idx=0)
    dialog._add_row(attr="idx", mode="per loc", lo=1.0, hi=100.0,
                    enabled=True, auto_range=False)
    dialog._add_row(attr="tid", mode="per loc", lo=0.0, hi=19.0,
                    enabled=True, auto_range=False)
    dialog._apply_all()
    return dialog, ds, state


def _ticks(dialog):
    from PyQt6.QtWidgets import QCheckBox

    from minflux_viewer.ui.filter_dialog import _COL_ENABLED

    out = []
    for row in range(dialog._table.rowCount()):
        box = dialog._table.cellWidget(row, _COL_ENABLED)
        out.append(box.isChecked() if isinstance(box, QCheckBox) else None)
    return out


def test_all_filters_off_unticks_the_rows_as_well_as_ungating_the_data(_app):
    """The reported defect: the data was ungated but the boxes stayed ticked."""
    dialog, ds, _state = _dialog_with_two_rows()
    try:
        assert _ticks(dialog) == [True, True]
        gated = int(np.asarray(ds.filter_mask, dtype=bool).sum())
        assert gated < ds.prop.num_loc, "the rows must actually filter something"

        dialog._set_all_rows_enabled(False)

        assert _ticks(dialog) == [False, False]                 # the dialog
        assert int(np.asarray(ds.filter_mask, dtype=bool).sum()) == ds.prop.num_loc
        assert ds.state.get("filter_specs") == []               # ...and the data
    finally:
        dialog.close()


def test_off_keeps_the_rows_so_they_do_not_come_back_on_the_next_edit(_app):
    """The second half of the old bug: an unrelated edit re-applied everything.

    ``_apply_all`` builds the mask from the ticked rows, so leaving them ticked
    meant any later apply resurrected filters the user had just cleared.
    """
    dialog, ds, _state = _dialog_with_two_rows()
    try:
        dialog._set_all_rows_enabled(False)
        assert dialog._table.rowCount() == 2, "rows are kept, only unticked"

        dialog._apply_all()          # stands in for any later row edit

        assert int(np.asarray(ds.filter_mask, dtype=bool).sum()) == ds.prop.num_loc
        assert ds.state.get("filter_specs") == []
    finally:
        dialog.close()


def test_all_filters_on_reticks_every_row_and_regates(_app):
    dialog, ds, _state = _dialog_with_two_rows()
    try:
        gated = int(np.asarray(ds.filter_mask, dtype=bool).sum())
        dialog._set_all_rows_enabled(False)
        dialog._set_all_rows_enabled(True)

        assert _ticks(dialog) == [True, True]
        assert int(np.asarray(ds.filter_mask, dtype=bool).sum()) == gated
        assert len(ds.state.get("filter_specs") or []) == 2
    finally:
        dialog.close()


def test_the_linked_views_are_notified_once_not_once_per_row(_app):
    """Every row's checkbox is wired to ``_apply_all``; setting eight of them
    naively would re-filter and redraw every linked view eight times."""
    dialog, _ds, state = _dialog_with_two_rows()
    try:
        for value in (5.0, 6.0, 7.0):
            dialog._add_row(attr="idx", mode="per loc", lo=value, hi=100.0,
                            enabled=True, auto_range=False)
        seen = []
        state.filter_changed.connect(lambda idx: seen.append(idx))

        dialog._set_all_rows_enabled(False)

        assert len(seen) == 1, f"one apply per click, got {len(seen)}"
    finally:
        dialog.close()


def test_the_buttons_say_what_they_do(_app):
    """The labels are the contract: they name the row state, not an action on
    the data, because that is what they set."""
    from PyQt6.QtWidgets import QPushButton

    dialog, _ds, _state = _dialog_with_two_rows()
    try:
        labels = {button.text() for button in dialog.findChildren(QPushButton)}
        assert {"All filters On", "All filters Off"} <= labels
        assert "Reset all filters" not in labels and "Apply all" not in labels
    finally:
        dialog.close()
