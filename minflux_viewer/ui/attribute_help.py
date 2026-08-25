"""Hover help for attribute pickers.

Every attribute dropdown and attribute menu in the application explains the
same names, so the wording lives in one place: ``core/attributes.py::
attribute_description`` is the single source, and these two helpers are the
only way it reaches a widget. Call them **after** (re)populating the picker.
"""

from __future__ import annotations

from PyQt6.QtCore import Qt
from PyQt6.QtWidgets import QComboBox, QMenu

from ..core.attributes import attribute_description

#: Marks a combo whose current-text tooltip is already wired, so repopulating
#: it does not stack another connection.
_WIRED = "_attribute_help_wired"


def apply_attribute_tooltips(combo: QComboBox, *, follow_current: bool = True) -> None:
    """Describe every entry of an attribute dropdown, and the box itself.

    ``follow_current`` also keeps the closed combo's own tooltip on the
    selected attribute, which is what makes the help reachable without opening
    the list. Safe to call repeatedly.
    """
    for index in range(combo.count()):
        combo.setItemData(
            index,
            attribute_description(combo.itemText(index)),
            Qt.ItemDataRole.ToolTipRole,
        )
    if not follow_current:
        return
    combo.setToolTip(attribute_description(combo.currentText()))
    if not getattr(combo, _WIRED, False):
        combo.currentTextChanged.connect(
            lambda text, box=combo: box.setToolTip(attribute_description(text))
        )
        setattr(combo, _WIRED, True)


def apply_attribute_menu_tooltips(menu: QMenu, names=None) -> None:
    """Describe the attribute entries of a menu that lists attribute names.

    A QMenu shows no tooltips at all unless asked, so this switches them on as
    well. ``names`` restricts the help to entries that really are attributes —
    without it every entry is described, and `attribute_description` answers
    "Unknown parameter…" for anything else, which would be worse than silence.
    """
    menu.setToolTipsVisible(True)
    known = None if names is None else set(names)
    for action in menu.actions():
        if action.isSeparator() or action.menu() is not None:
            continue
        if known is not None and action.text() not in known:
            continue
        action.setToolTip(attribute_description(action.text()))
