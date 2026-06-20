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


def ensure_on_screen(window: QWidget, owner: QWidget | None = None, *, margin: int = 24) -> None:
    """Cap *window* to the available screen area and center it fully on-screen.

    Unparented top-level windows (the MSR reader, Dataset Manager, modeless
    result/plugin windows) get no automatic placement relative to a parent, so a
    tall one can open with its title bar above the screen's top edge — worse on
    laptop / scaled screens or when the taskbar shrinks the available height.
    Call this **after** ``show()`` so the window frame geometry is known. The
    target screen is the owner's screen, else the screen under the cursor, else
    the primary screen. No-op on any failure.
    """
    try:
        from PyQt6.QtGui import QCursor, QGuiApplication

        screen = None
        if owner is not None:
            try:
                top = owner.window()
                screen = top.screen()
            except Exception:
                screen = None
        if screen is None:
            screen = QGuiApplication.screenAt(QCursor.pos()) or QGuiApplication.primaryScreen()
        if screen is None:
            return

        avail = screen.availableGeometry()
        frame = window.frameGeometry()
        # Shrink so the *frame* (incl. title bar) fits within the available area.
        overflow_w = frame.width() - (avail.width() - 2 * margin)
        overflow_h = frame.height() - (avail.height() - 2 * margin)
        if overflow_w > 0 or overflow_h > 0:
            window.resize(max(320, window.width() - max(0, overflow_w)),
                          max(200, window.height() - max(0, overflow_h)))
            frame = window.frameGeometry()
        # Center the frame, then clamp its top-left inside the available area.
        x = avail.left() + (avail.width() - frame.width()) // 2
        y = avail.top() + (avail.height() - frame.height()) // 2
        x = min(max(x, avail.left()), max(avail.left(), avail.right() - frame.width()))
        y = min(max(y, avail.top()), max(avail.top(), avail.bottom() - frame.height()))
        # move() positions the client area; compensate for the frame margins.
        window.move(x + (window.x() - frame.x()), y + (window.y() - frame.y()))
    except Exception:
        pass


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
    ensure_on_screen(window, owner)
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
