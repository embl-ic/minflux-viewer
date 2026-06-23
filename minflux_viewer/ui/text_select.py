"""
minflux_viewer.ui.text_select
==============================
Small helper to make dialog text **copyable**.

By default a ``QLabel`` is not selectable, so users can't copy a dataset name,
file path or a reported number out of an info/result dialog. ``make_labels_
selectable`` flips on mouse + keyboard text selection for every label under a
container.

Apply it **only** to read-only info/result dialogs — not to interactive panels
that rely on label clicks propagating to the parent (e.g. the render overlay
channel rows), because a selectable label consumes those clicks.
"""

from __future__ import annotations

from PyQt6.QtCore import Qt
from PyQt6.QtWidgets import QLabel

_SELECTABLE = (Qt.TextInteractionFlag.TextSelectableByMouse
               | Qt.TextInteractionFlag.TextSelectableByKeyboard)


def make_labels_selectable(container) -> None:
    """Enable text selection on every ``QLabel`` under *container* (and itself)."""
    labels = list(container.findChildren(QLabel))
    if isinstance(container, QLabel):
        labels.append(container)
    for label in labels:
        label.setTextInteractionFlags(label.textInteractionFlags() | _SELECTABLE)
