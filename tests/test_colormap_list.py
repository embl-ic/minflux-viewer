"""The shipped colormap set, its user-settable order, and the naming rule."""

from __future__ import annotations

import numpy as np
import pytest

from minflux_viewer.colormaps import (
    BUILTIN_COLORMAP_NAMES,
    LEGACY_COLORMAP_NAMES,
    SOLID_GROUP_TOKEN,
    channel_colormap_names,
    colormap_lut,
    configure_colormap_order,
    make_colormap,
    named_colormap_names,
    ordered_colormap_names,
    solid_color_names,
    validate_custom_colormap_name,
)

PARULA_TABLE = r"C:/Users/zhuang/Desktop/Parula_256_RGB_code.txt"


@pytest.fixture(autouse=True)
def _clean_order():
    """Every test starts from the shipped order and restores it afterwards."""
    configure_colormap_order([], True)
    yield
    configure_colormap_order([], True)


# ------------------------------------------------------------------ the set

def test_the_offered_set_is_what_the_application_ships():
    assert BUILTIN_COLORMAP_NAMES == (
        "hot", "jet", "HiLo", "glasbey", "viridis", "inferno", "parula", "gray")
    # Resolvable so an old saved selection still opens, but not offered.
    assert LEGACY_COLORMAP_NAMES == ("turbo", "magma", "plasma", "cividis")
    assert "parula" not in LEGACY_COLORMAP_NAMES         # promoted, not duplicated
    for name in (*BUILTIN_COLORMAP_NAMES, *LEGACY_COLORMAP_NAMES):
        assert make_colormap(name) is not None


def test_parula_reproduces_the_matlab_table_it_was_built_from():
    """Sampled control points, so fidelity to the source table is the contract."""
    try:
        reference = np.loadtxt(PARULA_TABLE)
    except OSError:
        pytest.skip("the exported parula table is not on this machine")
    reference = np.rint(np.clip(reference, 0.0, 1.0) * 255).astype(int)
    lut = np.asarray(colormap_lut("parula", n=len(reference), alpha=False), dtype=int)
    assert lut.shape == reference.shape
    # Within 2/255 on every channel: imperceptible, and a tenth of the source
    # lines a full 256-entry literal would take.
    assert int(np.abs(lut - reference).max()) <= 2
    assert tuple(lut[0]) == tuple(reference[0])          # endpoints are exact
    assert tuple(lut[-1]) == tuple(reference[-1])


# ------------------------------------------------------- the naming rule

def test_an_offered_colormap_name_is_reserved_but_a_hidden_one_is_not():
    """Adopting a preset that is not in the list is the point of the check."""
    for name in (*BUILTIN_COLORMAP_NAMES, "Red", "Gray"):
        with pytest.raises(ValueError, match="reserved"):
            validate_custom_colormap_name(name)
    # A hidden compatibility map may be adopted; the custom one then wins,
    # because both name resolution and colormap building check custom first.
    for name in LEGACY_COLORMAP_NAMES:
        assert validate_custom_colormap_name(name) == name
    assert validate_custom_colormap_name("  My  density map ") == "My density map"


def test_the_preset_menu_offers_its_name_for_the_name_field():
    pytest.importorskip("PyQt6")
    from minflux_viewer.ui.custom_colormap_dialog import CustomColormapDialog

    display = CustomColormapDialog._preset_display_name
    assert display("preset-gradient:thermal") == "thermal"
    assert display("local/viridis") == "viridis"
    assert display("CET-L01") == "CET-L01"
    assert display("") == ""


# --------------------------------------------------------------- the order

def _maps():
    return ["hot", "jet", "viridis", "My map"]


def _solids():
    return ["Red", "Green", "Gray"]


def test_no_saved_order_keeps_the_shipped_sequence():
    assert channel_colormap_names() == [
        *solid_color_names(), *BUILTIN_COLORMAP_NAMES]


def test_a_saved_order_reaches_every_selector():
    configure_colormap_order(["jet", SOLID_GROUP_TOKEN, "hot"], True)
    names = channel_colormap_names()
    assert names[0] == "jet"
    assert names[1:1 + len(solid_color_names())] == list(solid_color_names())
    assert names[1 + len(solid_color_names())] == "hot"
    # Anything not named in the order still appears, after the placed entries.
    assert set(BUILTIN_COLORMAP_NAMES) <= set(names)

    # named_colormap_names is colormaps only, in the same relative order.
    only_maps = named_colormap_names(include_custom=False)
    assert only_maps[:2] == ["jet", "hot"]
    assert "Red" not in only_maps


def test_folding_moves_the_solids_as_one_block_in_the_colour_dialogs_order():
    folded = ordered_colormap_names(
        _maps(), _solids(), ["jet", SOLID_GROUP_TOKEN, "hot"], fold_solids=True)
    assert folded[:5] == ["jet", "Red", "Green", "Gray", "hot"]
    # An individual solid in a folded order is ignored: folded, the block is
    # the only thing placed and the COLOR dialog owns what is inside it.
    assert ordered_colormap_names(
        _maps(), _solids(), ["Gray", "jet", SOLID_GROUP_TOKEN],
        fold_solids=True)[0] == "jet"


def test_unfolded_a_single_solid_can_be_placed_anywhere():
    """The stated use: a frequently used 'Gray' at the top of the list."""
    unfolded = ordered_colormap_names(
        _maps(), _solids(), ["Gray", "jet"], fold_solids=False)
    assert unfolded[0] == "Gray"
    assert unfolded[1] == "jet"
    # The solids not placed default to the end, never vanish.
    assert set(_solids()) <= set(unfolded)
    assert SOLID_GROUP_TOKEN not in unfolded


def test_the_order_never_hides_a_new_map_or_keeps_a_deleted_one():
    order = ["gone", "jet", "hot"]
    result = ordered_colormap_names(_maps(), _solids(), order, fold_solids=True)
    assert "gone" not in result                        # deleted custom map
    assert result[:2] == ["jet", "hot"]
    assert "My map" in result and "viridis" in result   # added since, appended


# ------------------------------------------------------- the reorder dialog

def test_sorting_orders_the_maps_and_leaves_the_solid_rows_in_place():
    pytest.importorskip("PyQt6")
    from minflux_viewer.ui.colormap_order_dialog import sort_entries

    rows = ["viridis", SOLID_GROUP_TOKEN, "hot", "jet"]
    by_name = sort_entries(rows, "name")
    assert by_name[1] == SOLID_GROUP_TOKEN             # the group holds its slot
    assert [by_name[0], by_name[2], by_name[3]] == ["hot", "jet", "viridis"]
    assert sort_entries(rows, "name", descending=True)[0] == "viridis"

    by_added = sort_entries(rows, "added")
    assert [by_added[0], by_added[2], by_added[3]] == ["hot", "jet", "viridis"]

    # An individual solid row is a category too, and also holds its position.
    mixed = sort_entries(["viridis", "Gray", "hot"], "name")
    assert mixed == ["hot", "Gray", "viridis"]

    with pytest.raises(ValueError):
        sort_entries(rows, "nonsense")


def test_the_dialog_folds_and_unfolds_without_losing_anything(qtbot):
    pytest.importorskip("PyQt6")
    from minflux_viewer.ui.colormap_order_dialog import ColormapOrderDialog

    dialog = ColormapOrderDialog(entries=channel_colormap_names(), fold_solids=True)
    qtbot.addWidget(dialog)
    assert dialog.entries()[0] == SOLID_GROUP_TOKEN
    assert dialog.fold_solids() is True

    dialog._fold.setChecked(False)
    unfolded = dialog.entries()
    assert SOLID_GROUP_TOKEN not in unfolded
    assert set(solid_color_names()) <= set(unfolded)
    assert set(BUILTIN_COLORMAP_NAMES) <= set(unfolded)

    dialog._fold.setChecked(True)
    assert dialog.entries().count(SOLID_GROUP_TOKEN) == 1
    assert set(BUILTIN_COLORMAP_NAMES) <= set(dialog.entries())


def test_the_first_click_sorts_ascending_and_the_second_reverses(qtbot):
    pytest.importorskip("PyQt6")
    from minflux_viewer.ui.colormap_order_dialog import ColormapOrderDialog

    dialog = ColormapOrderDialog(entries=channel_colormap_names(), fold_solids=True)
    qtbot.addWidget(dialog)

    dialog._sort("name")
    ascending = [n for n in dialog.entries() if n != SOLID_GROUP_TOKEN]
    assert ascending == sorted(ascending, key=str.casefold)

    dialog._sort("name")
    descending = [n for n in dialog.entries() if n != SOLID_GROUP_TOKEN]
    assert descending == list(reversed(ascending))

    # 'time added' starts ascending too, even after the other button was used.
    dialog._sort("added")
    by_added = [n for n in dialog.entries() if n != SOLID_GROUP_TOKEN]
    assert by_added == list(BUILTIN_COLORMAP_NAMES)


def test_reordering_from_the_lut_dialog_persists_and_reaches_the_list(qtbot,
                                                                      monkeypatch):
    """Custom ▸ Reorder colormap list… is the one place the order is set."""
    pytest.importorskip("PyQt6")
    from PyQt6.QtWidgets import QDialog

    from minflux_viewer.core.app_state import AppState
    from minflux_viewer.ui import colormap_order_dialog as order_module
    from minflux_viewer.ui.lut_dialog import LutDialog

    state = AppState()
    dialog = LutDialog(lambda *a: None, lambda *a: None, state=state)
    qtbot.addWidget(dialog)
    labels = [action.text() for action in dialog._custom_cmap_button.menu().actions()
              if action.text()]
    assert any("Reorder colormap list" in label for label in labels)

    chosen = ["jet", "hot", SOLID_GROUP_TOKEN, "parula", "viridis",
              "HiLo", "glasbey", "inferno", "gray"]

    class _Accepted(order_module.ColormapOrderDialog):
        def exec(self):
            self._set_rows(chosen)
            return QDialog.DialogCode.Accepted

    monkeypatch.setattr(order_module, "ColormapOrderDialog", _Accepted)
    try:
        dialog._reorder_colormaps()
        _assert_reorder_applied(state, dialog, chosen)
    finally:
        # The preferences file is shared for the whole session, so put it back.
        state.prefs["plot"]["colormap_order"] = []
        state.prefs["plot"]["colormap_fold_solids"] = True
        state.save_prefs()


def _assert_reorder_applied(state, dialog, chosen):
    assert state.prefs["plot"]["colormap_order"] == chosen
    assert state.prefs["plot"]["colormap_fold_solids"] is True
    names = channel_colormap_names()
    assert names[:2] == ["jet", "hot"]
    assert names[2:2 + len(solid_color_names())] == list(solid_color_names())
    # The dialog's own combo is repopulated, not left showing the old order.
    assert [dialog._cmap_combo.itemText(i) for i in range(2)] == ["jet", "hot"]


def test_cancelling_the_reorder_changes_nothing(qtbot, monkeypatch):
    pytest.importorskip("PyQt6")
    from PyQt6.QtWidgets import QDialog

    from minflux_viewer.core.app_state import AppState
    from minflux_viewer.ui import colormap_order_dialog as order_module
    from minflux_viewer.ui.lut_dialog import LutDialog

    state = AppState()
    dialog = LutDialog(lambda *a: None, lambda *a: None, state=state)
    qtbot.addWidget(dialog)
    before = channel_colormap_names()

    class _Rejected(order_module.ColormapOrderDialog):
        def exec(self):
            self._set_rows(["gray", SOLID_GROUP_TOKEN])
            return QDialog.DialogCode.Rejected

    monkeypatch.setattr(order_module, "ColormapOrderDialog", _Rejected)
    saved = list(state.prefs["plot"].get("colormap_order", []))
    dialog._reorder_colormaps()
    assert channel_colormap_names() == before
    assert state.prefs["plot"].get("colormap_order", []) == saved


# ----------------------------------------------- the hierarchical dropdown

def test_folding_puts_the_solids_under_a_node_in_the_lut_dropdown(qtbot):
    """The fold setting is about the dropdown, not the reorder list.

    The render and scatter menus already group the solids under a *Solid color*
    submenu; a ``QComboBox`` is flat by default, so the LUT dialog's list is
    backed by a tree model to match.
    """
    pytest.importorskip("PyQt6")
    from minflux_viewer.colormaps import colormap_tree_entries
    from minflux_viewer.ui.colormap_combo import SOLID_GROUP_LABEL, ColormapComboBox

    combo = ColormapComboBox()
    qtbot.addWidget(combo)
    combo.set_entries(colormap_tree_entries())

    model = combo._model
    top = [model.item(row).text() for row in range(model.rowCount())]
    assert SOLID_GROUP_LABEL in top
    for name in BUILTIN_COLORMAP_NAMES:
        assert name in top
    for name in solid_color_names():
        assert name not in top                    # nested, not at the top level

    group = model.item(top.index(SOLID_GROUP_LABEL))
    children = [group.child(row).text() for row in range(group.rowCount())]
    assert children == list(solid_color_names())
    # Every name is still reachable, and the group itself is not a colormap.
    assert set(combo.colormap_names()) == set(solid_color_names()) | set(
        BUILTIN_COLORMAP_NAMES)
    assert combo.contains("Red") and not combo.contains(SOLID_GROUP_LABEL)


def test_unfolded_the_dropdown_is_flat_again(qtbot):
    pytest.importorskip("PyQt6")
    from minflux_viewer.colormaps import colormap_tree_entries
    from minflux_viewer.ui.colormap_combo import SOLID_GROUP_LABEL, ColormapComboBox

    configure_colormap_order(["Gray", "jet"], False)
    combo = ColormapComboBox()
    qtbot.addWidget(combo)
    combo.set_entries(colormap_tree_entries())
    top = [combo._model.item(row).text() for row in range(combo._model.rowCount())]
    assert SOLID_GROUP_LABEL not in top
    assert top[0] == "Gray"                       # placed individually
    assert set(solid_color_names()) <= set(top)


def test_selecting_a_nested_solid_works_where_findText_does_not(qtbot):
    """``QComboBox.findText`` does not descend, which broke the silent select."""
    pytest.importorskip("PyQt6")
    from PyQt6.QtCore import Qt

    from minflux_viewer.colormaps import colormap_tree_entries
    from minflux_viewer.ui.colormap_combo import ColormapComboBox

    combo = ColormapComboBox()
    qtbot.addWidget(combo)
    combo.set_entries(colormap_tree_entries())

    assert combo.findText("Red", Qt.MatchFlag.MatchFixedString) < 0   # the trap
    assert combo.set_current_colormap("Red") is True
    assert combo.current_colormap() == "Red"
    # The branch pointer is restored, or the next popup would show only solids.
    assert not combo.rootModelIndex().isValid()

    assert combo.set_current_colormap("viridis") is True
    assert combo.current_colormap() == "viridis"
    assert combo.set_current_colormap("not a colormap") is False
    assert combo.current_colormap() == "viridis"                     # unchanged


def _click_row(combo, name, qtbot):
    """Pick *name* the way a user does — a real click in the open popup.

    ⚠ Drive the popup, never the handler. ``QComboBoxPrivateContainer`` filters
    the mouse release out of the popup view and runs the combo's own
    item-selected path, so ``QTreeView.clicked`` **never fires** for a popup
    click. A test that called the click handler directly passed while the
    dropdown was inert in the application — which is exactly what happened.
    """
    from PyQt6.QtCore import Qt
    from PyQt6.QtTest import QTest
    from PyQt6.QtWidgets import QApplication

    combo.showPopup()
    QApplication.processEvents()
    view = combo.view()
    index = combo._index_for(name)
    assert index is not None, name
    view.scrollTo(index)
    QApplication.processEvents()
    rect = view.visualRect(index)
    assert rect.isValid() and not rect.isEmpty(), f"{name} is not laid out"
    QTest.mouseClick(view.viewport(), Qt.MouseButton.LeftButton,
                     Qt.KeyboardModifier.NoModifier, rect.center())
    QApplication.processEvents()


def test_picking_from_the_open_popup_announces_the_colormap(qtbot):
    """The regression: the dropdown changed nothing in the view it was open on."""
    pytest.importorskip("PyQt6")
    from minflux_viewer.colormaps import colormap_tree_entries
    from minflux_viewer.ui.colormap_combo import ColormapComboBox

    combo = ColormapComboBox()
    qtbot.addWidget(combo)
    combo.set_entries(colormap_tree_entries())
    combo.resize(200, 24)
    combo.show()
    qtbot.waitExposed(combo)
    chosen: list[str] = []
    combo.colormap_changed.connect(chosen.append)
    combo.set_current_colormap("hot", silent=True)
    chosen.clear()

    _click_row(combo, "viridis", qtbot)              # a top-level colormap
    assert combo.current_colormap() == "viridis"
    assert chosen == ["viridis"]

    nested = solid_color_names()[1]
    _click_row(combo, nested, qtbot)                 # one inside the group
    assert combo.current_colormap() == nested
    assert chosen == ["viridis", nested]
    # The branch pointer is restored, or the next popup shows only the solids.
    assert not combo.rootModelIndex().isValid()


def test_a_programmatic_selection_is_not_announced_as_a_user_pick(qtbot):
    pytest.importorskip("PyQt6")
    from minflux_viewer.colormaps import colormap_tree_entries
    from minflux_viewer.ui.colormap_combo import ColormapComboBox

    combo = ColormapComboBox()
    qtbot.addWidget(combo)
    combo.set_entries(colormap_tree_entries())
    chosen: list[str] = []
    combo.colormap_changed.connect(chosen.append)

    combo.set_current_colormap("jet", silent=True)
    combo.set_entries(colormap_tree_entries())       # a rebuild restores it
    assert combo.current_colormap() == "jet"
    assert chosen == []


def test_clicking_a_group_row_never_becomes_the_selection(qtbot):
    pytest.importorskip("PyQt6")
    from minflux_viewer.colormaps import colormap_tree_entries
    from minflux_viewer.ui.colormap_combo import SOLID_GROUP_LABEL, ColormapComboBox

    combo = ColormapComboBox()
    qtbot.addWidget(combo)
    combo.set_entries(colormap_tree_entries())
    combo.resize(200, 24)
    combo.show()
    qtbot.waitExposed(combo)
    chosen: list[str] = []
    combo.colormap_changed.connect(chosen.append)
    combo.set_current_colormap("hot", silent=True)
    chosen.clear()

    from PyQt6.QtCore import Qt
    from PyQt6.QtTest import QTest
    from PyQt6.QtWidgets import QApplication

    combo.showPopup()
    QApplication.processEvents()
    view = combo.view()
    top = [combo._model.item(r).text() for r in range(combo._model.rowCount())]
    group = combo._model.item(top.index(SOLID_GROUP_LABEL))
    QTest.mouseClick(view.viewport(), Qt.MouseButton.LeftButton,
                     Qt.KeyboardModifier.NoModifier,
                     view.visualRect(group.index()).center())
    QApplication.processEvents()
    combo.hidePopup()

    assert combo.current_colormap() == "hot"       # a heading is not a colormap
    assert chosen == []


def test_the_lut_dropdown_keeps_its_selection_across_a_rebuild(qtbot):
    pytest.importorskip("PyQt6")
    from minflux_viewer.core.app_state import AppState
    from minflux_viewer.ui.colormap_combo import SOLID_GROUP_LABEL
    from minflux_viewer.ui.lut_dialog import LutDialog

    dialog = LutDialog(lambda *a: None, lambda *a: None, state=AppState())
    qtbot.addWidget(dialog)
    combo = dialog._cmap_combo
    top = [combo._model.item(r).text() for r in range(combo._model.rowCount())]
    assert SOLID_GROUP_LABEL in top

    dialog._set_combo_silent("Magenta")           # a nested entry
    assert combo.current_colormap() == "Magenta"
    dialog._refresh_colormap_combo()
    assert combo.current_colormap() == "Magenta"

    # A hidden compatibility map stays reachable while it is the selection.
    dialog._set_combo_silent("turbo")
    assert combo.current_colormap() == "turbo"


def test_the_grayscale_map_and_the_grey_solid_are_not_confused():
    """``gray`` (a colormap) and ``Gray`` (a solid) differ only by case.

    Keying the order case-insensitively made the colormap ``gray`` read as the
    solid and vanish from every list — the distinction ``canonical_colormap_name``
    explicitly relies on.
    """
    maps = ["hot", "gray"]
    solids = ["Red", "Gray"]

    folded = ordered_colormap_names(maps, solids, ["hot", SOLID_GROUP_TOKEN, "gray"],
                                    fold_solids=True)
    assert folded == ["hot", "Red", "Gray", "gray"]

    unfolded = ordered_colormap_names(maps, solids, ["Gray", "gray", "hot"],
                                      fold_solids=False)
    assert unfolded == ["Gray", "gray", "hot", "Red"]
    assert unfolded.count("gray") == 1 and unfolded.count("Gray") == 1

    # And neither disappears when the order mentions neither of them.
    both = ordered_colormap_names(maps, solids, ["hot"], fold_solids=False)
    assert "gray" in both and "Gray" in both


def test_unfolding_from_the_dropdown_keeps_every_entry(qtbot):
    """The end-to-end version of the case collision above."""
    pytest.importorskip("PyQt6")
    from minflux_viewer.colormaps import colormap_tree_entries
    from minflux_viewer.ui.colormap_combo import ColormapComboBox

    configure_colormap_order([], True)
    combo = ColormapComboBox()
    qtbot.addWidget(combo)
    combo.set_entries(colormap_tree_entries())
    folded = set(combo.colormap_names())

    configure_colormap_order(
        [*solid_color_names(), *BUILTIN_COLORMAP_NAMES], False)
    combo.set_entries(colormap_tree_entries())
    assert set(combo.colormap_names()) == folded
    assert "gray" in combo.colormap_names() and "Gray" in combo.colormap_names()


def test_building_the_list_never_leaves_a_heading_as_the_selection(qtbot):
    """Qt selects row 0 on insert, and folded that row is the group heading."""
    pytest.importorskip("PyQt6")
    from minflux_viewer.colormaps import colormap_tree_entries
    from minflux_viewer.ui.colormap_combo import SOLID_GROUP_LABEL, ColormapComboBox

    configure_colormap_order([], True)               # folded: the group leads
    combo = ColormapComboBox()
    qtbot.addWidget(combo)
    announced: list[str] = []
    combo.colormap_changed.connect(announced.append)

    combo.set_entries(colormap_tree_entries())
    assert combo.current_colormap() != SOLID_GROUP_LABEL
    assert combo.contains(combo.current_colormap())
    assert announced == []                           # a rebuild is not a pick
    assert not combo.contains(SOLID_GROUP_LABEL)


def test_the_lut_dropdown_actually_recolours_the_view_it_is_open_on(qtbot):
    """The regression, end to end: picking in the dropdown changed nothing.

    ``colormap_changed`` was raised only from the view's ``clicked`` signal,
    which a combo popup never delivers — so the LUT dialog's callback never ran
    and the render kept its original colormap while B/C and gamma still worked.
    """
    pytest.importorskip("PyQt6")
    import numpy as np

    from minflux_viewer.core.app_state import AppState
    from minflux_viewer.core.dataset import build_localization_dataset
    from minflux_viewer.ui.render_window import RenderWindow

    rng = np.random.default_rng(0)
    state = AppState()
    state.add_dataset(build_localization_dataset(
        name="A", x_nm=rng.random(400) * 1000, y_nm=rng.random(400) * 1000,
        z_nm=rng.random(400) * 100))
    window = RenderWindow(state, dataset_idx=0)
    qtbot.addWidget(window)
    window.open_lut_dialog()
    dialog = window._lut_dialog
    assert dialog is not None
    qtbot.addWidget(dialog)

    for target in ("viridis", solid_color_names()[4], "inferno"):
        _click_row(dialog._cmap_combo, target, qtbot)
        assert dialog._cmap_combo.current_colormap() == target
        assert window._active_cmap == target
        if window._channels:
            assert window._channels[0]["lut"] == target
