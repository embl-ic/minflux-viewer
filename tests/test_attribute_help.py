"""Every attribute picker explains its entries, from one shared source."""

from __future__ import annotations

import numpy as np
import pytest

pytest.importorskip("PyQt6")
pytest.importorskip("pyqtgraph")

from PyQt6.QtCore import QEvent, QPoint, Qt
from PyQt6.QtGui import QContextMenuEvent, QHelpEvent
from PyQt6.QtWidgets import QApplication, QComboBox, QMenu, QToolTip, QWidget

from minflux_viewer.core.app_state import AppState
from minflux_viewer.core.attributes import aggregation_description, attribute_description
from minflux_viewer.core.dataset import build_localization_dataset
from minflux_viewer.ui.widgets.attribute_help import (
    apply_attribute_menu_tooltips,
    apply_attribute_tooltips,
)


@pytest.fixture
def _qt_app(qapp):
    return qapp


def _state() -> AppState:
    rng = np.random.default_rng(0)
    n = 40
    state = AppState()
    state.add_dataset(
        build_localization_dataset(
            name="help",
            x_nm=rng.normal(0.0, 100.0, n),
            y_nm=rng.normal(0.0, 100.0, n),
            z_nm=np.zeros(n),
            tid=np.repeat(np.arange(1, n // 4 + 1), 4).astype(float),
            attrs={
                "efo": rng.normal(5e4, 9e3, n),
                "cfr": rng.uniform(0.0, 1.0, n),
                "dcr": rng.uniform(0.0, 1.0, n),
            },
        )
    )
    return state


def _tips(combo: QComboBox) -> list[str]:
    return [
        combo.itemData(i, Qt.ItemDataRole.ToolTipRole) or ""
        for i in range(combo.count())
    ]


def _hovered(combo: QComboBox, target: QWidget | None = None) -> str:
    """The tooltip *combo* shows on hover: the help event Qt's hover timer
    sends, delivered to *target* (a child such as an editable combo's line
    edit) or to the combo itself."""
    widget = target or combo
    pos = QPoint(4, 4)
    QApplication.sendEvent(widget, QHelpEvent(QEvent.Type.ToolTip, pos, widget.mapToGlobal(pos)))
    QToolTip.hideText()
    return combo.toolTip()


def test_helper_describes_every_entry_and_follows_the_selection(_qt_app):
    combo = QComboBox()
    combo.addItems(["efo", "cfr"])
    apply_attribute_tooltips(combo)

    assert _tips(combo) == [attribute_description("efo"), attribute_description("cfr")]
    assert combo.toolTip() == attribute_description("efo")
    combo.setCurrentText("cfr")
    assert combo.toolTip() == attribute_description("cfr")

    # Repopulating must not stack a second connection on the same combo.
    apply_attribute_tooltips(combo)
    apply_attribute_tooltips(combo)
    combo.setCurrentText("efo")
    assert combo.toolTip() == attribute_description("efo")


def test_hover_describes_a_selection_made_with_signals_blocked(_qt_app):
    """The Filter dialog's Auto rows all said 'Index of localization data
    points': the help followed ``currentTextChanged``, and every row is set to
    its attribute with signals blocked, so it kept describing the first entry."""
    combo = QComboBox()
    combo.addItems(["idx", "efo", "cfr"])
    apply_attribute_tooltips(combo)
    combo.blockSignals(True)
    combo.setCurrentText("cfr")
    combo.blockSignals(False)

    assert _hovered(combo) == attribute_description("cfr")


def test_a_note_follows_the_description_and_reaches_an_editable_box(_qt_app):
    """Preferences' rule box had two writers of one tooltip; which one showed
    depended on whether loading the rule changed the text (efo: the hint only,
    cfr: the description only)."""
    from minflux_viewer.ui.settings.prefs_process import AutoRuleRow, ProcessPage

    page = ProcessPage()
    try:
        for attr in ("efo", "cfr"):
            row = AutoRuleRow(page, {"attribute": attr, "method": "percentile"})
            shown = _hovered(row.attribute, row.attribute.lineEdit())
            assert shown.startswith(attribute_description(attr) + "\n\n")
            assert shown.endswith("Any attribute name may be typed.")
            # The dropdown's entries carry the description alone.
            assert _tips(row.attribute)[0] == attribute_description("efo")
            row.deleteLater()
    finally:
        page.close()


def test_histogram_boxes_describe_a_selection_set_by_a_filter_edit(_qt_app):
    """``start_filter_edit`` (a filter row's eye button) sets Attribute and As
    with signals blocked, and every redraw used to blank the As tooltip."""
    from minflux_viewer.ui.plots.histogram_window import HistogramWindow

    histogram = HistogramWindow(_state(), dataset_idx=0)
    try:
        assert _hovered(histogram._agg_combo) == aggregation_description("per loc")
        histogram._attr_combo.setCurrentText("efo")
        histogram._agg_combo.setCurrentText("trace mean")
        histogram._draw()
        assert _hovered(histogram._agg_combo) == aggregation_description("trace mean")

        histogram.start_filter_edit(attr="cfr", mode="per loc", lo=0.0, hi=0.5)
        assert _hovered(histogram._attr_combo) == attribute_description("cfr")
        assert _hovered(histogram._agg_combo) == aggregation_description("per loc")
        histogram._clear_filter_edit(restore=False)

        # A track-level attribute locks the box, and the tooltip says why.
        histogram._attr_combo.setCurrentText("siz")
        assert not histogram._agg_combo.isEnabled()
        assert "track-level attribute" in _hovered(histogram._agg_combo)
    finally:
        histogram.close()


def test_scatter_color_by_describes_its_selection(_qt_app):
    """Repopulated with signals blocked, it described the first entry (idx)
    whatever it was set to."""
    from minflux_viewer.ui.plots.scatter_window import ScatterWindow

    scatter = ScatterWindow(_state(), dataset_idx=0)
    try:
        combo = scatter._cbar_combo
        assert combo.currentText() != combo.itemText(0)
        assert _hovered(combo) == attribute_description(combo.currentText())
    finally:
        scatter.close()


def test_a_filter_row_follows_the_attribute_chosen_in_the_histogram(_qt_app):
    """Eye button, change the attribute in the histogram, Update Filter: the row
    takes it with signals blocked and kept describing the old one."""
    from minflux_viewer.ui.dialogs import filter_dialog as fd
    from minflux_viewer.ui.dialogs.filter_dialog import FilterDialog
    from minflux_viewer.ui.plots.histogram_window import HistogramWindow

    dialog = FilterDialog(_state(), dataset_idx=0)
    histograms: list[HistogramWindow] = []
    try:
        dialog._add_row(attr="cfr")
        combo = dialog._table.cellWidget(0, fd._COL_ATTR)
        assert _hovered(combo) == attribute_description("cfr")

        dialog._display_filter(0)
        histograms = [w for w in QApplication.topLevelWidgets()
                      if isinstance(w, HistogramWindow) and w._filter_edit]
        assert len(histograms) == 1
        histograms[0]._attr_combo.setCurrentText("efo")
        histograms[0]._update_filter_edit()

        assert combo.currentText() == "efo"
        assert _hovered(combo) == attribute_description("efo")
    finally:
        for window in histograms:
            window.close()
        dialog.close()


def test_menu_helper_leaves_non_attribute_entries_alone(_qt_app):
    menu = QMenu()
    menu.addAction("efo")
    menu.addAction("Colormap")            # not an attribute
    apply_attribute_menu_tooltips(menu, ["efo"])

    assert menu.toolTipsVisible()
    assert menu.actions()[0].toolTip() == attribute_description("efo")
    # Untouched, so Qt keeps its default (the action's own text). Without the
    # guard it would read "Unknown parameter…", which is worse than nothing.
    assert menu.actions()[1].toolTip() == "Colormap"
    assert "Unknown parameter" not in menu.actions()[1].toolTip()


def test_every_attribute_picker_carries_the_help(_qt_app):
    """Attribute Plot, Histogram, Scatter, Filter and the menus that list names."""
    from minflux_viewer.ui.dialogs.filter_dialog import FilterDialog
    from minflux_viewer.ui.plots.attribute_window import AttributeWindow
    from minflux_viewer.ui.plots.histogram_window import HistogramWindow
    from minflux_viewer.ui.plots.scatter_window import ScatterWindow

    state = _state()
    windows = []
    try:
        plot = AttributeWindow(state, dataset_idx=0)
        windows.append(plot)
        plot._add_dimension("C", "cfr")
        for combo in (plot._x_combo, plot._y_combo, plot._c_combo):
            assert all(_tips(combo)) and combo.toolTip()

        histogram = HistogramWindow(state, dataset_idx=0)
        windows.append(histogram)
        assert all(_tips(histogram._attr_combo))

        scatter = ScatterWindow(state, dataset_idx=0)
        windows.append(scatter)
        assert all(_tips(scatter._cbar_combo))

        filters = FilterDialog(state, dataset_idx=0)
        windows.append(filters)
        filters._add_row()
        combo = filters._table.cellWidget(0, 1)
        assert isinstance(combo, QComboBox)
        assert all(_tips(combo))

        # ...and the menus that list the same names.
        captured: list[QMenu] = []
        original_exec = QMenu.exec
        QMenu.exec = lambda self, *args: captured.append(self)
        try:
            scatter._show_context_menu(QPoint(5, 5))
            color_menu = next(
                action.menu() for action in captured[0].actions()
                if action.text() == "Color by"
            )
            assert color_menu.toolTipsVisible()
            assert color_menu.actions()[0].toolTip()

            captured.clear()
            plot._show_context_menu(QPoint(5, 5))
            add_z = next(
                action.menu() for action in captured[0].actions()
                if action.text().startswith("add new attribute as Z")
            )
            assert add_z.toolTipsVisible() and add_z.actions()[0].toolTip()

            captured.clear()
            bar = scatter._colorbar
            bar.contextMenuEvent(
                QContextMenuEvent(
                    QContextMenuEvent.Reason.Mouse,
                    QPoint(4, 4),
                    bar.mapToGlobal(QPoint(4, 4)),
                )
            )
            attribute_menu = next(
                action.menu() for action in captured[0].actions()
                if action.text() == "Attribute:"
            )
            assert attribute_menu.toolTipsVisible()
            assert attribute_menu.actions()[0].toolTip()
        finally:
            QMenu.exec = original_exec
    finally:
        for window in windows:
            window.close()
