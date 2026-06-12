"""
minflux_viewer.ui.modeless
==========================
Helper for showing **modeless** (non-blocking) windows.

Project convention: analysis result windows and plugin windows are
*modeless* and *non-owned* — they float like the viewer windows, never
sit permanently in front of them, and never freeze the rest of the app
while open. Use :func:`show_modeless` for any such window unless a task
explicitly calls for a modal dialog (e.g. one that must return a decision
before the caller can continue — alignment choices, parameter pickers,
file dialogs, confirmations).

The owner (usually the main window) keeps a reference so the window is not
garbage-collected before the user closes it; ``WA_DeleteOnClose`` then
frees it on close. Build the window with **no QWidget parent** so the OS
does not pin it above the owner in Z-order.
"""

from __future__ import annotations

from PyQt6.QtCore import Qt
from PyQt6.QtWidgets import QWidget

# Fallback registry when no owner is supplied — keyed by object identity so
# entries can be discarded safely even after the C++ object is destroyed.
_ORPHAN_WINDOWS: set[QWidget] = set()


def _alive(window: QWidget) -> bool:
    try:
        window.objectName()        # touches the C++ object; raises if deleted
        return True
    except RuntimeError:
        return False


def show_modeless(window: QWidget, owner: QWidget | None = None) -> QWidget:
    """Show *window* as a modeless, non-owned top-level window.

    Sets non-modal state and ``WA_DeleteOnClose``, retains a reference on
    *owner* (pruning closed entries) so the window survives the call, then
    shows and raises it. Returns *window* for convenience.
    """
    window.setModal(False)
    try:
        window.setWindowModality(Qt.WindowModality.NonModal)
    except Exception:
        pass
    window.setAttribute(Qt.WidgetAttribute.WA_DeleteOnClose, True)

    if owner is not None:
        bag = [w for w in getattr(owner, "_modeless_windows", []) if _alive(w)]
        bag.append(window)
        owner._modeless_windows = bag
    else:
        _ORPHAN_WINDOWS.add(window)
        window.destroyed.connect(lambda *_a, w=window: _ORPHAN_WINDOWS.discard(w))

    window.show()
    window.raise_()
    window.activateWindow()
    return window


def close_modeless(owner: QWidget | None) -> None:
    """Close every modeless window retained for *owner*.

    Non-owned modeless windows do not close automatically when the main
    window closes (they have no QWidget parent), so the owner must close
    them explicitly on shutdown. Safe to call repeatedly.
    """
    for window in list(getattr(owner, "_modeless_windows", []) or []):
        try:
            window.close()
        except Exception:
            pass
    if owner is not None and hasattr(owner, "_modeless_windows"):
        owner._modeless_windows = []
