"""Global COLOR registry, dialog, and live-notification regressions."""

from __future__ import annotations

import copy
from types import SimpleNamespace

from PyQt6.QtCore import Qt
from PyQt6.QtGui import QColor
from PyQt6.QtWidgets import (
    QApplication,
    QColorDialog,
    QGridLayout,
    QLabel,
    QPushButton,
    QSpinBox,
    QStyle,
    QStyleOptionFrame,
)

from minflux_viewer.colormaps import channel_colormap_names
from minflux_viewer.colors import (
    DEFAULT_COLOR_PREFS,
    changed_color_paths,
    configure_colors,
    normalize_color_preferences,
    solid_color_names,
)
from minflux_viewer.core.app_state import AppState, _merge, default_prefs
from minflux_viewer.ui.global_color_dialog import GlobalColorDialog
from minflux_viewer.ui.main_window import MainWindow
from minflux_viewer.ui.preferences_dialog import PreferencesDialog


def test_every_configurable_color_is_rgba():
    colors = normalize_color_preferences({})

    def walk(value):
        if isinstance(value, dict):
            for child in value.values():
                walk(child)
        elif isinstance(value, list) and value and isinstance(value[0], list):
            for child in value:
                walk(child)
        else:
            assert len(value) == 4
            assert all(0 <= channel <= 255 for channel in value)

    walk(colors)


def test_changed_paths_are_leaf_specific():
    before = normalize_color_preferences({})
    after = copy.deepcopy(before)
    after["viewer"]["attribute_data"] = [1, 2, 3, 4]

    assert changed_color_paths(before, after) == {"viewer.attribute_data"}


def test_color_dialog_is_modeless_unowned_and_apply_keeps_it_open(qtbot):
    state = AppState()
    state.prefs = default_prefs()
    state.save_prefs = lambda: None
    emitted = []
    state.colors_changed.connect(emitted.append)
    dialog = GlobalColorDialog(state)
    qtbot.addWidget(dialog)
    dialog.show()

    assert dialog.parentWidget() is None
    assert not dialog.isModal()
    assert dialog._picker.testOption(QColorDialog.ColorDialogOption.ShowAlphaChannel)

    dialog._set_value(("viewer", "attribute_data"), (1, 2, 3, 4))
    dialog._apply()
    assert dialog.isVisible()
    assert state.prefs["colors"]["viewer"]["attribute_data"] == [1, 2, 3, 4]
    assert emitted[-1]["paths"] == {"viewer.attribute_data"}


def test_toolbar_reuses_one_color_dialog(qtbot):
    state = AppState()
    state.prefs = default_prefs()
    state.save_prefs = lambda: None
    window = MainWindow(state)
    qtbot.addWidget(window)

    window._show_color_picker()
    first = window._color_dialog
    window._show_color_picker()

    assert window._color_dialog is first
    assert first.parentWidget() is None
    assert not first.isModal()


def test_preferences_and_color_dialog_share_the_registry(qtbot):
    state = AppState()
    state.prefs = default_prefs()
    state.save_prefs = lambda: None
    dialog = PreferencesDialog(state)
    qtbot.addWidget(dialog)

    assert not hasattr(dialog, "_plot_cmap_combo")
    dialog._attribute_data_color.set_rgba((9, 8, 7, 6))
    dialog._apply_widgets_to_draft()
    assert dialog._draft["colors"]["viewer"]["attribute_data"] == [9, 8, 7, 6]
    assert "attr_cmap" not in dialog._draft["plot"]


def test_open_color_dialog_tracks_external_preference_changes(qtbot):
    state = AppState()
    state.prefs = default_prefs()
    state.save_prefs = lambda: None
    dialog = GlobalColorDialog(state)
    qtbot.addWidget(dialog)
    dialog._set_value(("viewer", "attribute_data"), (1, 2, 3, 4))

    external = copy.deepcopy(state.prefs["colors"])
    external["viewer"]["attribute_data"] = [9, 8, 7, 6]
    state.apply_color_preferences(external)

    assert dialog._value(("viewer", "attribute_data")) == [9, 8, 7, 6]
    assert dialog._buttons[("viewer", "attribute_data")].rgba() == (9, 8, 7, 6)


def test_section_reset_does_not_reset_other_sections(qtbot):
    state = AppState()
    state.prefs = default_prefs()
    dialog = GlobalColorDialog(state)
    qtbot.addWidget(dialog)
    dialog._set_value(("solid", "Red"), (1, 2, 3, 4))
    dialog._set_value(("viewer", "roi_edge"), (5, 6, 7, 8))

    dialog._reset_solid()

    assert dialog._value(("solid", "Red")) == DEFAULT_COLOR_PREFS["solid"]["Red"]
    assert dialog._value(("viewer", "roi_edge")) == [5, 6, 7, 8]


def test_palette_update_is_explicit_and_fields_are_compact(qtbot):
    state = AppState()
    state.prefs = default_prefs()
    dialog = GlobalColorDialog(state)
    qtbot.addWidget(dialog)
    dialog.show()
    qtbot.wait(1)
    original = dialog._value(("solid", "Red"))
    dialog._select_path(("solid", "Red"))
    dialog._picker.setCurrentColor(QColor(1, 2, 3, 4))

    assert dialog._value(("solid", "Red")) == original
    assert dialog._current_preview.rgba() == (1, 2, 3, 4)
    assert dialog._update_button.text() == "Update Selected Feature Color"
    dialog._update_button.click()
    assert dialog._value(("solid", "Red")) == [1, 2, 3, 4]

    labels = dialog.findChildren(QLabel)
    alpha = next(label for label in labels if label.text() == "Alpha")
    html = next(label for label in labels if label.text() == "HEX:")
    grid = dialog._picker_form
    assert isinstance(grid, QGridLayout)
    assert grid.getItemPosition(grid.indexOf(html))[:2] == (3, 0)
    assert grid.getItemPosition(grid.indexOf(alpha))[:2] == (3, 3)
    assert dialog._apply_button.geometry().x() < dialog._ok_button.geometry().x()
    assert dialog._palette_group.height() >= dialog._palette_group.minimumHeight()
    assert dialog._tabs.count() == 4
    assert not any(label.text() == "COLOR" for label in labels)

    solid_positions = [
        dialog._solid_layout.getItemPosition(index)[:2]
        for index in range(dialog._solid_layout.count())
    ]
    columns, _spacing = dialog._solid_columns()
    assert len(solid_positions) == len(dialog._draft["solid"])
    assert {row for row, _column in solid_positions} == {
        1 + index // columns for index in range(len(solid_positions))
    }
    assert max(column for _row, column in solid_positions) == (
        min(columns, len(solid_positions)) - 1
    )
    assert {edit.width() for edit in dialog._solid_name_edits.values()} == {
        dialog._solid_entry_width
    }
    solid_edits = list(dialog._solid_name_edits.values())
    first_right = solid_edits[0].mapTo(dialog, solid_edits[0].rect().topRight()).x()
    second_left = solid_edits[1].mapTo(dialog, solid_edits[1].rect().topLeft()).x()
    assert second_left - first_right - 1 >= 8

    # A flat group (the default one is row-nested; see the ROI Manager tests).
    dialog._function_combo.setCurrentText("Iteration series")
    QApplication.processEvents()
    function_positions = [
        dialog._function_components.getItemPosition(index)[:2]
        for index in range(dialog._function_components.count())
    ]
    assert max(column for _row, column in function_positions) <= 4
    assert max(row for row, _column in function_positions) >= 1

    first_pair = dialog._function_components.itemAt(0).widget()
    pair_layout = first_pair.layout()
    name_label = pair_layout.itemAt(0).widget()
    swatch = pair_layout.itemAt(1).widget()
    assert swatch.geometry().left() - name_label.geometry().right() <= 5
    assert first_pair.width() >= first_pair.sizeHint().width()

    assert [dialog._tabs.tabText(i) for i in range(dialog._tabs.count())][2:] == [
        "Components", "Plugins",
    ]
    # The minimum is content-derived now (it was a fixed 190), but never below
    # the old floor -- see test_misc_section_clears_its_last_row_of_swatches.

    # Preview placement is asserted by test_preview_sits_beside_the_gradient.
    # Button placement is asserted by test_update_button_sits_under_the_value_fields.

    # Height is the content height capped to the screen — see
    # test_dialog_height_is_capped_to_the_screen.
    opened_height = dialog.height()
    assert dialog.minimumHeight() == opened_height
    dialog.resize(dialog.width(), 600)
    QApplication.processEvents()
    assert dialog.height() >= opened_height
    assert dialog._update_button.isVisible()
    assert dialog._cancel_button.isVisible()


def _shown_dialog(qtbot):
    state = AppState()
    state.prefs = default_prefs()
    dialog = GlobalColorDialog(state)
    qtbot.addWidget(dialog)
    dialog.show()
    qtbot.wait(1)
    QApplication.processEvents()
    return dialog


def test_dialog_opens_narrow_and_without_scrollbars(qtbot):
    dialog = _shown_dialog(qtbot)

    # Width is fixed and set by the palette; see
    # test_dialog_width_is_fixed_and_set_by_the_palette.
    assert dialog.minimumWidth() == dialog.maximumWidth()
    # One section per tab keeps the dialog short.
    assert [dialog._tabs.tabText(i) for i in range(dialog._tabs.count())] == [
        "Solid Color List", "Viewer / Plots", "Components", "Plugins",
    ]


def test_dialog_height_is_capped_to_the_screen(qtbot):
    """Still capped to the screen, though tabs make that rarely bite."""
    dialog = _shown_dialog(qtbot)
    available = QApplication.primaryScreen().availableGeometry().height()

    assert dialog.height() <= available
    assert dialog.height() == max(600, min(760, available - 60))
    assert dialog.minimumHeight() == dialog.height()


def test_solid_entries_pack_left_and_fill_the_available_row(qtbot):
    dialog = _shown_dialog(qtbot)

    # 105 % of the former 50 px is the floor, not the answer.
    assert dialog._solid_entry_width >= int(50 * 1.05 + 0.5)

    # 'Magenta' was clipped: its editor must now be able to show the whole name.
    magenta = dialog._solid_name_edits["Magenta"]
    option = QStyleOptionFrame()
    magenta.initStyleOption(option)
    content = magenta.style().subElementRect(
        QStyle.SubElement.SE_LineEditContents, option, magenta
    )
    assert content.width() >= magenta.fontMetrics().horizontalAdvance("Magenta")

    # Entries start at the left edge rather than being centred in the section.
    host_left = dialog._solid_host.mapTo(
        dialog, dialog._solid_host.rect().topLeft()
    ).x()
    first = next(iter(dialog._solid_name_edits.values()))
    first_left = first.mapTo(dialog, first.rect().topLeft()).x()
    assert 0 <= first_left - host_left <= 6

    # Six on the top row at the default width, so Orange and White wrap down.
    columns, _spacing = dialog._solid_columns()
    assert columns == 5
    rows: dict[int, list[int]] = {}
    for index in range(dialog._solid_layout.count()):
        row, column = dialog._solid_layout.getItemPosition(index)[:2]
        rows.setdefault(row, []).append(column)
    assert all(column < columns for used in rows.values() for column in used)
    populated = [r for r in sorted(rows) if rows[r]]
    for row in populated[:-1]:
        assert len(rows[row]) == columns

    names = list(dialog._draft["solid"])
    top_row = names[:columns]
    assert "Magenta" in top_row
    assert "Yellow" not in top_row and "Orange" not in top_row


def test_section_reset_shares_the_first_content_row(qtbot):
    dialog = _shown_dialog(qtbot)
    resets = [
        button
        for button in dialog.findChildren(QPushButton)
        if button.text() == "Reset"
    ]
    assert len(resets) == 4

    solid_group = dialog._solid_host.parentWidget()
    solid_reset = next(b for b in resets if b.parentWidget() is solid_group)
    # Same row as the entries, not a row of its own above them...
    assert abs(solid_reset.geometry().top() - dialog._solid_host.geometry().top()) <= 8
    # ...and still in the section's top-right corner.
    assert 0 <= solid_group.width() - solid_reset.geometry().right() <= 14


def test_picker_value_fields_do_not_clip_three_digit_values(qtbot):
    dialog = _shown_dialog(qtbot)

    for name in (
        "_hue",
        "_saturation",
        "_value_spin",
        "_red",
        "_green",
        "_blue",
        "_alpha",
    ):
        spin = getattr(dialog, name)
        spin.setValue(spin.maximum())
        assert spin.width() >= spin.sizeHint().width(), name


def test_misc_color_items_sit_below_their_dropdown(qtbot):
    """Items under the combo, not beside it — that reclaims the left column."""
    dialog = _shown_dialog(qtbot)

    for tab, (combo, components) in enumerate((
        (dialog._function_combo, dialog._function_components),
        (dialog._plugin_combo, dialog._plugin_components),
    ), start=2):
        dialog._tabs.setCurrentIndex(tab)
        QApplication.processEvents()
        first_pair = components.itemAt(0).widget()
        combo_bottom = combo.mapTo(dialog, combo.rect().bottomLeft()).y()
        combo_left = combo.mapTo(dialog, combo.rect().topLeft()).x()
        pair_top = first_pair.mapTo(dialog, first_pair.rect().topLeft()).y()
        pair_left = first_pair.mapTo(dialog, first_pair.rect().topLeft()).x()
        assert pair_top >= combo_bottom
        assert abs(pair_left - combo_left) <= 6


def test_preview_sits_beside_the_gradient(qtbot):
    dialog = _shown_dialog(qtbot)
    panel = dialog._picker_panel

    gradient = panel._picker_child_rects()["gradient"]
    assert gradient is not None, "gradient widget not found inside QColorDialog"
    preview = dialog._current_preview.geometry()
    # To the right of the picker (so past the alpha bar), not below it...
    assert preview.left() >= dialog._picker.geometry().right()
    # ...stretched so its foot lines up with the gradient and alpha bar...
    assert abs(preview.bottom() - gradient.bottom()) <= 2
    # ...under a caption level with the 'Basic colors' heading.
    rects = panel._picker_child_rects()
    label = panel._preview_label.geometry()
    assert panel._preview_label.text() == "Preview"
    assert abs(label.top() - rects["basic_label"].top()) <= 2
    assert abs(preview.top() - rects["basic_grid"].top()) <= 2


def test_value_fields_keep_their_spinner_buttons(qtbot):
    dialog = _shown_dialog(qtbot)

    for name in ("_hue", "_saturation", "_value_spin", "_red", "_green",
                 "_blue", "_alpha"):
        spin = getattr(dialog, name)
        assert spin.buttonSymbols() != QSpinBox.ButtonSymbols.NoButtons, name
        spin.setValue(spin.maximum())
        assert spin.width() >= spin.sizeHint().width(), name




def test_update_button_matches_the_add_to_custom_colors_button(qtbot):
    dialog = _shown_dialog(qtbot)
    panel = dialog._picker_panel

    add = panel._picker_child_rects()["add_button"]
    button = dialog._update_button.geometry()
    assert add is not None
    # Same width and left edge, so the two stack as one column...
    assert button.width() == add.width()
    assert button.left() == add.left()
    assert button.top() >= add.bottom()
    # ...and its foot lands on the last field row, closing the block evenly.
    # Same baseline as the last field row.  The button also has to clear the
    # Add button above it, and those two constraints sit ~10 px apart, so this
    # is 'level with', not pixel-identical.
    hex_bottom = dialog._html.mapTo(panel, dialog._html.rect().bottomLeft()).y()
    assert abs(button.bottom() - hex_bottom) <= 12
    assert dialog._html.width() == dialog._hue.width()


def test_wrap_columns_closes_gaps_before_it_wraps():
    """Pure rule behind every wrapping grid: spacing gives way before a row does."""
    from minflux_viewer.ui.global_color_dialog import wrap_columns

    kwargs = dict(preferred=5, max_spacing=32, min_spacing=8)
    wide = wrap_columns(600, 55, **kwargs)
    assert wide == (5, 32)                      # roomy: full row, widest gap

    snug = wrap_columns(330, 55, **kwargs)
    assert snug[0] == 5 and 8 <= snug[1] < 32   # same row, gaps closed

    tight = wrap_columns(300, 55, **kwargs)
    assert tight[0] < 5                         # only now does an item move down

    assert wrap_columns(40, 55, **kwargs)[0] == 1
    assert wrap_columns(0, 55, **kwargs) == (5, 32)   # before the first layout


def test_all_section_resets_are_on_screen(qtbot):
    dialog = _shown_dialog(qtbot)
    resets = [
        button
        for button in dialog.findChildren(QPushButton)
        if button.text() == "Reset"
    ]

    # One per tab: Solid, Viewer, Components, Plugins.
    assert len(resets) == 4
    # Check each page's own Reset while that page is the visible one — a hidden
    # page keeps whatever geometry it last had.
    for index in range(dialog._tabs.count()):
        dialog._tabs.setCurrentIndex(index)
        QApplication.processEvents()
        page = dialog._tabs.widget(index)
        button = next(
            b for b in page.findChildren(QPushButton) if b.text() == "Reset"
        )
        right = button.mapTo(dialog, button.rect().topRight()).x()
        assert right <= dialog.width(), dialog._tabs.tabText(index)

def test_registered_color_groups_are_offered_in_the_dialog(qtbot):
    dialog = _shown_dialog(qtbot)

    offered_functions = {
        dialog._function_combo.itemText(i)
        for i in range(dialog._function_combo.count())
    }
    offered_plugins = {
        dialog._plugin_combo.itemText(i)
        for i in range(dialog._plugin_combo.count())
    }
    assert offered_functions == set(DEFAULT_COLOR_PREFS["functions"])
    assert offered_plugins == set(DEFAULT_COLOR_PREFS["plugins"])
    assert {"Iteration series", "Localization precision"} <= offered_functions
    assert {
        "Spatial Line Pattern", "Drift Correction", "Trace Viewer"
    } <= offered_plugins


def test_runtime_component_colors_follow_the_registry():
    """Modules that cannot carry ``prefs`` still see the user's edits."""
    from minflux_viewer.colors import runtime_component_colors

    prefs = default_prefs()
    configure_colors(prefs)
    assert runtime_component_colors("plugins", "Drift Correction")["X drift"] == (
        255, 90, 90, 255
    )

    prefs["colors"]["plugins"]["Drift Correction"]["X drift"] = [1, 2, 3, 4]
    configure_colors(prefs)
    assert runtime_component_colors("plugins", "Drift Correction")["X drift"] == (
        1, 2, 3, 4
    )

    # An unknown group falls back to the declared defaults, never empty.
    configure_colors({})
    assert runtime_component_colors("functions", "Localization precision")
    assert runtime_component_colors("plugins", "nope") == {}


def test_roi_stroke_is_written_in_the_format_pyqtgraph_reads():
    """Qt's #AARRGGBB read as #RRGGBBAA turned every ROI fully transparent."""
    import pyqtgraph as pg

    from minflux_viewer.colors import pg_safe_hex, rgba_qt_hex

    roi = DEFAULT_COLOR_PREFS["viewer"]["roi_edge"]

    # The old writer: pyqtgraph takes the last byte as alpha, so an opaque
    # color arrived fully transparent — invisible edges and invisible labels.
    assert pg.mkColor(rgba_qt_hex(roi)).alpha() == 0

    # The fix round-trips through pyqtgraph unchanged.
    stored = pg_safe_hex(roi)
    assert tuple(pg.mkColor(stored).getRgb()) == tuple(roi)

    # A record still holding the legacy string is repaired on read.
    assert pg_safe_hex(rgba_qt_hex(roi)) == stored

    # Opaque colors and plain 6-digit values are unaffected.
    assert tuple(pg.mkColor(pg_safe_hex([255, 255, 0, 255])).getRgb()) == (
        255, 255, 0, 255
    )
    assert tuple(pg.mkColor(pg_safe_hex("#00e5ff")).getRgb()) == (0, 229, 255, 255)


def test_changing_the_roi_color_recolors_existing_rois(qtbot):
    """'Cannot change back': the rewrite compared with the wrong parser."""
    from minflux_viewer.colors import normalize_rgba, pg_safe_hex, rgba_qt_hex
    from minflux_viewer.core.roi import RoiRecord

    state = AppState()
    state.prefs = default_prefs()
    window = MainWindow(state)
    qtbot.addWidget(window)

    original = list(DEFAULT_COLOR_PREFS["viewer"]["roi_edge"])
    # A ROI stored by the buggy build, i.e. carrying the legacy Qt-format string.
    record = RoiRecord.create(
        "rectangle",
        {"bounds": [0.0, 0.0, 10.0, 10.0]},
        stroke_color=rgba_qt_hex(original),
    )
    state.rois.add(record)

    colors = copy.deepcopy(state.prefs["colors"])
    colors["viewer"]["roi_edge"] = [0, 128, 255, 255]
    state.apply_color_preferences(colors)

    stored = state.rois.records[0].stroke_color
    assert normalize_rgba(stored) == (0, 128, 255, 255)

    # ...and back again.
    colors = copy.deepcopy(state.prefs["colors"])
    colors["viewer"]["roi_edge"] = original
    state.apply_color_preferences(colors)
    assert state.rois.records[0].stroke_color == pg_safe_hex(original)


def test_roi_row_exposes_face_edge_corner_and_highlight(qtbot):
    dialog = _shown_dialog(qtbot)

    for key in ("roi_face", "roi_edge", "roi_corner", "roi_highlight"):
        assert ("viewer", key) in dialog._buttons
    labels = {label.text() for label in dialog.findChildren(QLabel)}
    assert {"face", "edge", "corner", "highlight data in ROI"} <= labels


def test_roi_manager_group_is_first_and_renders_labelled_rows(qtbot):
    dialog = _shown_dialog(qtbot)

    assert dialog._function_combo.itemText(0) == "ROI Manager"
    assert dialog._function_combo.currentText() == "ROI Manager"

    headings = []
    for index in range(dialog._function_components.count()):
        widget = dialog._function_components.itemAt(index).widget()
        if widget.layout() is None:          # a row heading, not a name+swatch
            headings.append(widget.text())
    assert headings == ["ROI entries:", "ROI selected:"]

    for row in ("ROI entries", "ROI selected"):
        for item in ("face", "edge", "corner", "label"):
            assert ("functions", "ROI Manager", row, item) in dialog._buttons


def test_legacy_single_roi_color_seeds_the_split_keys():
    """An existing preference must not silently revert to the default."""
    saved = {"viewer": {"roi": [10, 20, 30, 200]}}

    result = normalize_color_preferences(saved)

    assert "roi" not in result["viewer"]
    assert result["viewer"]["roi_edge"] == [10, 20, 30, 255]
    assert result["viewer"]["roi_face"] == [10, 20, 30, 128]
    assert result["viewer"]["roi_highlight"] == [10, 20, 30, 255]


def test_roi_palette_follows_state_and_keeps_custom_stroke():
    """Draft / entry / selected each take their own colors."""
    from minflux_viewer.colors import pg_safe_hex, viewer_color
    from minflux_viewer.core.roi import RoiRecord
    from minflux_viewer.ui.roi_overlay import RoiOverlayController

    state = AppState()
    state.prefs = default_prefs()
    # The palette is pure lookup, so drive it without building a render view.
    controller = RoiOverlayController.__new__(RoiOverlayController)
    controller.owner = SimpleNamespace(_state=state)
    controller.draft = None

    system = RoiRecord.create(
        "rectangle", {"bounds": [0.0, 0.0, 10.0, 10.0]},
        stroke_color=pg_safe_hex(viewer_color(state.prefs, "roi_edge")),
    )
    entry = controller._roi_palette(system, manager_highlight=False)
    chosen = controller._roi_palette(system, manager_highlight=True)
    defaults = DEFAULT_COLOR_PREFS["functions"]["ROI Manager"]
    assert entry["face"] == tuple(defaults["ROI entries"]["face"])
    assert chosen["face"] == tuple(defaults["ROI selected"]["face"])
    assert entry["corner"] != entry["edge"]      # corner is cyan, edge yellow

    # A ROI with its own color keeps it rather than following the palette.
    custom = RoiRecord.create(
        "rectangle", {"bounds": [0.0, 0.0, 10.0, 10.0]},
        stroke_color="#ff00ff",
    )
    assert controller._roi_palette(custom, manager_highlight=False)["edge"] == (
        255, 0, 255, 255
    )


def test_dialog_width_is_fixed_and_set_by_the_palette(qtbot):
    dialog = _shown_dialog(qtbot)

    assert dialog.minimumWidth() == dialog.maximumWidth() == dialog.width()
    # The palette is the widest section, so it decides the width.
    margins = dialog.layout().contentsMargins()
    assert dialog.width() == (
        dialog._palette_group.sizeHint().width() + margins.left() + margins.right()
    )
    # Dragging wider must not take.
    dialog.resize(dialog.width() + 200, dialog.height())
    QApplication.processEvents()
    assert dialog.width() == dialog.maximumWidth()


def test_viewer_rows_sit_below_a_blank_row_but_reset_does_not(qtbot):
    dialog = _shown_dialog(qtbot)
    dialog._tabs.setCurrentIndex(1)
    QApplication.processEvents()

    grid = dialog._viewer_grid
    assert grid.rowMinimumHeight(0) > 0
    # Nothing occupies the blank row.
    rows = {
        grid.getItemPosition(i)[0] for i in range(grid.count())
    }
    assert 0 not in rows

    first_label = grid.itemAtPosition(1, 0).widget()
    first_top = first_label.mapTo(dialog, first_label.rect().topLeft()).y()
    reset_top = dialog._viewer_reset.mapTo(
        dialog, dialog._viewer_reset.rect().topLeft()
    ).y()
    # Reset stays up top; only the grid moved down.
    assert reset_top < first_top


def test_a_long_section_scrolls_instead_of_squeezing(qtbot):
    """Adding many solid colors must not shrink the rows or grow the dialog."""
    from PyQt6.QtWidgets import QScrollArea

    dialog = _shown_dialog(qtbot)
    page = dialog._tabs.widget(0)
    assert isinstance(page, QScrollArea)
    assert page.verticalScrollBar().maximum() == 0

    tabs_height = dialog._tabs.height()
    for _ in range(30):
        dialog._add_solid()
    for _ in range(8):          # the scroll range settles over a few passes
        QApplication.processEvents()

    assert page.verticalScrollBar().maximum() > 0, "should scroll"
    assert dialog._tabs.height() == tabs_height, "dialog must not grow"
    # Entries keep their natural size rather than being squeezed to fit.
    for edit in dialog._solid_name_edits.values():
        assert edit.height() >= edit.sizeHint().height()
        assert edit.width() == dialog._solid_entry_width


def test_every_tab_page_fits_the_capped_tab_area(qtbot):
    """The tab area is capped at the tallest page plus a spare row."""
    dialog = _shown_dialog(qtbot)

    available = dialog._tabs.height() - dialog._tabs.tabBar().sizeHint().height()
    for index in range(dialog._tabs.count()):
        dialog._tabs.setCurrentIndex(index)
        QApplication.processEvents()
        page = dialog._tabs.widget(index)
        assert page.sizeHint().height() <= available, dialog._tabs.tabText(index)
    # Capped, not soaking up every spare pixel under the palette.
    tallest = max(
        dialog._tabs.widget(i).sizeHint().height()
        for i in range(dialog._tabs.count())
    )
    assert available - tallest <= 60


def test_solid_entries_start_below_a_blank_row(qtbot):
    dialog = _shown_dialog(qtbot)

    rows = {
        dialog._solid_layout.getItemPosition(i)[0]
        for i in range(dialog._solid_layout.count())
    }
    assert min(rows) == 1, "entries should leave row 0 empty"


def test_value_fields_are_inset_from_the_gradient(qtbot):
    """One swatch width right of the gradient, clear of the left column."""
    dialog = _shown_dialog(qtbot)
    panel = dialog._picker_panel
    rects = panel._picker_child_rects()

    swatch = rects["basic_grid"].width() // 8
    assert panel._fields.geometry().left() == rects["gradient"].left() + swatch


def test_palette_group_is_sized_to_its_content(qtbot):
    """It was sized from QColorDialog's un-laid-out 640x480 default.

    That left the group hundreds of pixels too tall: the panel floated under a
    blank band and the bottom of the palette fell off the dialog.
    """
    dialog = _shown_dialog(qtbot)
    group = dialog._palette_group
    panel = dialog._picker_panel

    # No dead space above the panel, and none hiding below it either.
    assert panel.geometry().top() <= 30
    assert group.height() - panel.height() <= 60
    # The whole group, and the button row under it, stay inside the dialog.
    assert group.geometry().bottom() < dialog.height()
    assert dialog._apply_button.geometry().bottom() <= dialog.height()


def test_hex_field_is_centered_and_upper_case(qtbot):
    dialog = _shown_dialog(qtbot)

    assert bool(dialog._html.alignment() & Qt.AlignmentFlag.AlignHCenter)

    dialog._picker.setCurrentColor(QColor(255, 128, 0))
    QApplication.processEvents()
    assert dialog._html.text() == "#FF8000"

    # Lower case still parses, and is normalized once the edit completes.
    dialog._html.setText("#0a0b0c")
    dialog._html.editingFinished.emit()
    QApplication.processEvents()
    assert dialog._html.text() == "#0A0B0C"
    assert dialog._picker.currentColor() == QColor(10, 11, 12)


def test_pick_screen_color_button_is_available(qtbot):
    """Qt's own screen picker is kept; only its value form is replaced."""
    dialog = _shown_dialog(qtbot)

    picker = dialog._picker
    button = next(
        b for b in picker.findChildren(QPushButton)
        if "screen" in b.text().replace("&", "").casefold()
    )
    assert button.isVisible()
    # Sits under the Basic colors grid, not off in the replaced value form.
    grid = dialog._picker_panel._picker_child_rects()["basic_grid"]
    assert button.geometry().top() >= grid.bottom()
    assert abs(button.geometry().left() - grid.left()) <= 2


def test_value_fields_line_up_with_the_custom_color_swatches(qtbot):
    dialog = _shown_dialog(qtbot)
    panel = dialog._picker_panel

    custom_grid = panel._picker_child_rects()["custom_grid"]
    assert custom_grid is not None
    hue_top = dialog._hue.mapTo(panel, dialog._hue.rect().topLeft()).y()
    assert abs(hue_top - custom_grid.top()) <= 4


def test_solid_name_editor_consumes_application_shortcuts(qtbot):
    state = AppState()
    state.prefs = default_prefs()
    window = MainWindow(state)
    qtbot.addWidget(window)
    preferences_triggered = []
    window._ui.actionPreferences.triggered.disconnect()
    window._ui.actionPreferences.triggered.connect(
        lambda: preferences_triggered.append(True)
    )
    window.show()
    dialog = GlobalColorDialog(state)
    qtbot.addWidget(dialog)
    dialog.show()
    dialog.activateWindow()
    qtbot.wait(1)
    QApplication.processEvents()
    editor = dialog._solid_name_edits["Orange"]
    editor.clear()
    qtbot.mouseClick(editor, Qt.MouseButton.LeftButton)
    QApplication.processEvents()
    assert QApplication.focusWidget() is editor
    qtbot.keyClick(
        editor, Qt.Key.Key_P, modifier=Qt.KeyboardModifier.ShiftModifier
    )

    assert editor.text().casefold() == "p"
    assert preferences_triggered == []
    editor.setText("Pink")
    dialog._rename_solid("Orange", editor.text())
    assert "Pink" in dialog._draft["solid"]
    assert "Orange" not in dialog._draft["solid"]


def test_solid_add_rename_delete_updates_runtime_lists(qtbot):
    state = AppState()
    state.prefs = default_prefs()
    state.save_prefs = lambda: None
    emitted = []
    state.colors_changed.connect(emitted.append)
    dialog = GlobalColorDialog(state)
    qtbot.addWidget(dialog)

    dialog._picker.setCurrentColor(QColor(12, 34, 56, 78))
    dialog._add_solid()
    assert list(dialog._draft["solid"])[-1] == "Color 1"
    assert dialog._draft["solid"]["Color 1"] == [12, 34, 56, 78]
    dialog._rename_solid("Color 1", "Ocean")
    dialog._rename_solid("Red", "Signal")
    dialog._delete_solid("Orange")
    dialog._apply()

    assert solid_color_names()[-1] == "Ocean"
    assert "Signal" in channel_colormap_names()
    assert "Red" not in channel_colormap_names()
    assert "Orange" not in channel_colormap_names()
    assert emitted[-1]["solid_renames"] == {"Red": "Signal"}
    configure_colors(default_prefs())


def test_saved_solid_registry_does_not_restore_deleted_defaults():
    colors = normalize_color_preferences({"solid": {"Only": [1, 2, 3, 4]}})
    assert colors["solid"] == {"Only": [1, 2, 3, 4]}
    merged = _merge({"colors": {"solid": colors["solid"]}}, default_prefs())
    assert merged["colors"]["solid"] == {"Only": [1, 2, 3, 4]}


def test_rename_and_delete_preserve_existing_dataset_lut(qtbot):
    state = AppState()
    state.prefs = default_prefs()
    state.save_prefs = lambda: None
    dataset = SimpleNamespace(
        state={"render_channel_lut": "Red", "overlay_lut": "solid:Red"}
    )
    state.datasets.append(dataset)
    window = MainWindow(state)
    qtbot.addWidget(window)
    dialog = GlobalColorDialog(state)
    qtbot.addWidget(dialog)

    dialog._rename_solid("Red", "Signal")
    dialog._apply()
    assert dataset.state["render_channel_lut"] == "Signal"
    assert dataset.state["overlay_lut"] == "solid:Signal"

    dialog._delete_solid("Signal")
    dialog._apply()
    assert dataset.state["render_channel_lut"] == "solid:custom:#ff0000ff"
    assert dataset.state["overlay_lut"] == "solid:custom:#ff0000ff"
    configure_colors(default_prefs())


# --------------------------------------------------- the HEX field (COLOR dialog)

def test_hex_parser_accepts_the_forms_a_pasted_code_arrives_in():
    """``QColor('FF8000')`` is invalid for want of a '#', which is how a hex
    code copied from a web page or a figure legend normally arrives — so the
    field silently reverted whatever was pasted into it."""
    from minflux_viewer.colors import parse_hex_rgba

    assert parse_hex_rgba("#FF8000") == (255, 128, 0, 255)
    assert parse_hex_rgba("FF8000") == (255, 128, 0, 255)      # no '#'
    assert parse_hex_rgba("  ff8000 ") == (255, 128, 0, 255)   # padded, lower case
    assert parse_hex_rgba("0xFF8000") == (255, 128, 0, 255)
    assert parse_hex_rgba("f80") == (255, 136, 0, 255)         # shorthand
    # Eight digits are RRGGBBAA (the CSS convention an external source uses),
    # deliberately NOT Qt's #AARRGGBB, which would change the colour.
    assert parse_hex_rgba("#FF800080") == (255, 128, 0, 128)
    assert parse_hex_rgba("FF8000FF") == (255, 128, 0, 255)
    assert parse_hex_rgba("#f80", default_alpha=64) == (255, 136, 0, 64)

    for text in ("", "#GG0000", "12345", "not a colour", None, 7):
        assert parse_hex_rgba(text) is None


def test_typing_a_hex_code_recolours_the_preview_at_once(qtbot):
    from PyQt6.QtGui import QColor

    from minflux_viewer.core.app_state import AppState
    from minflux_viewer.ui.global_color_dialog import GlobalColorDialog

    dialog = GlobalColorDialog(AppState())
    qtbot.addWidget(dialog)
    dialog._picker.setCurrentColor(QColor(10, 20, 30))

    for text, expected in (("FF8000", (255, 128, 0)),
                           ("#00ff00", (0, 255, 0)),
                           ("0x0000FF", (0, 0, 255))):
        dialog._html.setText(text)
        dialog._html_field_edited(text)                 # what typing/pasting emits
        colour = dialog._picker.currentColor()
        assert (colour.red(), colour.green(), colour.blue()) == expected
        assert (dialog._red.value(), dialog._green.value(),
                dialog._blue.value()) == expected

    # Incomplete or invalid input is left alone rather than fought mid-typing.
    dialog._html.setText("#0")
    dialog._html_field_edited("#0")
    colour = dialog._picker.currentColor()
    assert (colour.red(), colour.green(), colour.blue()) == (0, 0, 255)
