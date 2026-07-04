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
from PyQt6.QtWidgets import QDialog, QWidget

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


def beside_position(anchor, avail, size, *, margin: int = 12,
                    prefer: tuple[str, ...] = ("right", "below", "left", "above")):
    """Pure geometry: top-left ``(x, y)`` to place a ``size=(w, h)`` window beside
    ``anchor`` within ``avail``.

    ``anchor`` and ``avail`` are exclusive ``(left, top, right, bottom)`` rects.
    Returns the first side in *prefer* where the window fits **fully** inside
    ``avail``; if none fit, the first preferred side **clamped** on-screen (so an
    anchor in the extreme top-left goes right/below, one in the bottom-right goes
    left/above, always staying on the monitor).
    """
    al, at, ar, ab = anchor
    vl, vt, vr, vb = avail
    w, h = size
    cands = {
        "right": (ar + margin, at),
        "left": (al - margin - w, at),
        "below": (al, ab + margin),
        "above": (al, at - margin - h),
    }

    def fits(xy) -> bool:
        x, y = xy
        return x >= vl and y >= vt and x + w <= vr and y + h <= vb

    for side in prefer:
        if side in cands and fits(cands[side]):
            return cands[side]
    x, y = cands.get(prefer[0], (ar + margin, at))
    x = min(max(x, vl), max(vl, vr - w))
    y = min(max(y, vt), max(vt, vb - h))
    return (x, y)


def place_beside(window: QWidget, anchor: QWidget | None, *, margin: int = 12,
                 prefer: tuple[str, ...] = ("right", "below", "left", "above")) -> None:
    """Move *window* next to *anchor* on *anchor*'s screen (best effort).

    Thin Qt wrapper over :func:`beside_position`. Call after both windows have a
    realistic size (``anchor`` shown; ``window`` ``resize``d or ``adjustSize()``-d).
    Falls back to :func:`ensure_on_screen` if *anchor* is missing; never throws.
    """
    try:
        from PyQt6.QtGui import QCursor, QGuiApplication
        if anchor is None:
            ensure_on_screen(window)
            return
        try:
            screen = anchor.window().screen()
        except Exception:
            screen = None
        if screen is None:
            screen = QGuiApplication.screenAt(QCursor.pos()) or QGuiApplication.primaryScreen()
        if screen is None:
            return
        av = screen.availableGeometry()
        a = anchor.frameGeometry()
        f = window.frameGeometry()
        anchor_rect = (a.left(), a.top(), a.left() + a.width(), a.top() + a.height())
        avail_rect = (av.left(), av.top(), av.left() + av.width(), av.top() + av.height())
        x, y = beside_position(anchor_rect, avail_rect, (f.width(), f.height()),
                               margin=margin, prefer=prefer)
        # move() positions the client area; compensate for frame margins.
        window.move(x + (window.x() - f.x()), y + (window.y() - f.y()))
    except Exception:
        pass


def show_modeless(window: QWidget, owner: QWidget | None = None) -> QWidget:
    """Show *window* as a modeless, non-owned top-level window.

    Sets non-modal state and ``WA_DeleteOnClose``, retains a reference on
    *owner* (pruning closed entries) so the window survives the call, then
    shows and raises it. Returns *window* for convenience.
    """
    # ``setModal`` is QDialog-only; ``setWindowModality`` is the QWidget-level
    # API and covers the non-modal intent for any top-level widget.
    if isinstance(window, QDialog):
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
