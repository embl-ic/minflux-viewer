"""Hover help for attribute pickers.

Every attribute dropdown and attribute menu in the application explains the
same names, so the wording lives in one place: ``core/attributes.py::
attribute_description`` is the single source, and these helpers are the
only way it reaches a widget. Call them **after** (re)populating the picker.

A closed combo's own tooltip describes its *current* entry, and it is worked
out when the tooltip is about to show (`describe_current_entry`). Following
``currentTextChanged`` alone is not enough: callers change the selection
under ``blockSignals`` to keep their own handlers quiet, which silences that
listener too, and the tooltip then went on describing the entry that was
current before — for every new Filter row the list's first entry, ``idx``.
So a caller never writes such a combo's tooltip itself; whatever the text
depends on goes into ``describe``.
"""

from __future__ import annotations

from collections.abc import Callable

from PyQt6.QtCore import QEvent, QObject, Qt
from PyQt6.QtWidgets import QComboBox, QMenu

from minflux_viewer.core.attributes import attribute_description

#: The combo's `_CurrentEntryTip`, so a repeated call reuses it rather than
#: stacking a second filter and connection.
_TIP = "_attribute_help_tip"


class _CurrentEntryTip(QObject):
    """Sets its parent combo's tooltip to ``describe(current text)``."""

    def __init__(self, combo: QComboBox, describe: Callable[[str], str]) -> None:
        super().__init__(combo)
        self.describe = describe
        combo.installEventFilter(self)
        # Keeps ``toolTip()`` current for readers that do not hover; the event
        # filter is what makes the shown text right after a blocked change.
        combo.currentTextChanged.connect(self._on_current_text_changed)

    def refresh(self) -> None:
        combo = self.parent()
        if isinstance(combo, QComboBox):
            combo.setToolTip(self.describe(combo.currentText()))

    def _on_current_text_changed(self, _text: str) -> None:
        self.refresh()

    def eventFilter(self, watched, event) -> bool:  # noqa: N802 - Qt API
        if event.type() == QEvent.Type.ToolTip:
            self.refresh()
        return False


def describe_current_entry(combo: QComboBox, describe: Callable[[str], str]) -> None:
    """Keep *combo*'s own tooltip on ``describe(current text)``.

    The text is worked out again each time the tooltip is about to show, so it
    is right however the selection changed, signals blocked or not. Safe to
    call repeatedly; a later call replaces *describe*.
    """
    tip = getattr(combo, _TIP, None)
    if tip is None:
        tip = _CurrentEntryTip(combo, describe)
        setattr(combo, _TIP, tip)
    else:
        tip.describe = describe
    tip.refresh()


def apply_attribute_tooltips(
    combo: QComboBox, *, follow_current: bool = True, note: str = ""
) -> None:
    """Describe every entry of an attribute dropdown, and the box itself.

    ``follow_current`` also keeps the closed combo's own tooltip on the
    selected attribute, which is what makes the help reachable without opening
    the list; ``note`` is appended to that tooltip (not to the entries) for a
    box that needs a word about itself. Safe to call repeatedly.
    """
    for index in range(combo.count()):
        combo.setItemData(
            index,
            attribute_description(combo.itemText(index)),
            Qt.ItemDataRole.ToolTipRole,
        )
    if not follow_current:
        return
    if not note:
        describe_current_entry(combo, attribute_description)
        return

    def describe(text: str) -> str:
        return "\n\n".join(part for part in (attribute_description(text), note) if part)

    describe_current_entry(combo, describe)


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
