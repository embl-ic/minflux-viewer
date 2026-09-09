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


def corner_position(avail, size, *, align="center", margin: int = 24):
    """Pure geometry: top-left ``(x, y)`` to place a ``size=(w, h)`` window at an
    anchor of the exclusive ``(left, top, right, bottom)`` rect *avail*.

    ``align`` is ``"center"`` (default), ``"top_right"``, ``"top_left"``,
    ``"bottom_right"`` or ``"bottom_left"`` — **or** a ``(fx, fy)`` fraction pair
    (CSS ``background-position`` semantics): ``fx=0`` flush-left … ``fx=1``
    flush-right, ``fy=0`` flush-top … ``fy=1`` flush-bottom. So ``(0.75, 0.25)``
    sits the window 75 % of the way toward the right and 25 % down (= 75 % up
    from the bottom). The result is always clamped so the window sits fully
    inside *avail* (assuming it is not larger than *avail*).
    """
    vl, vt, vr, vb = avail
    w, h = size
    if isinstance(align, (tuple, list)):
        fx, fy = float(align[0]), float(align[1])
        x = vl + int(round(fx * max(0, (vr - vl - w))))
        y = vt + int(round(fy * max(0, (vb - vt - h))))
    elif align == "center":
        x = vl + (vr - vl - w) // 2
        y = vt + (vb - vt - h) // 2
    else:
        right = "right" in align
        bottom = "bottom" in align
        x = (vr - margin - w) if right else (vl + margin)
        y = (vb - margin - h) if bottom else (vt + margin)
    x = min(max(x, vl), max(vl, vr - w))
    y = min(max(y, vt), max(vt, vb - h))
    return (x, y)


def ensure_on_screen(window: QWidget, owner: QWidget | None = None, *,
                     margin: int = 24,
                     align: "str | tuple[float, float]" = "center") -> None:
    """Cap *window* to the available screen area and place it fully on-screen.

    Unparented top-level windows (the MSR reader, Dataset Manager, modeless
    result/plugin windows) get no automatic placement relative to a parent, so a
    tall one can open with its title bar above the screen's top edge — worse on
    laptop / scaled screens or when the taskbar shrinks the available height.
    Call this **after** ``show()`` so the window frame geometry is known. The
    target screen is the owner's screen, else the screen under the cursor, else
    the primary screen. *align* (see :func:`corner_position`) chooses where on
    that screen — ``"center"`` by default, e.g. ``"top_right"`` for the main
    window. No-op on any failure.
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
        avail_rect = (avail.left(), avail.top(), avail.right() + 1, avail.bottom() + 1)
        x, y = corner_position(
            avail_rect, (frame.width(), frame.height()), align=align, margin=margin
        )
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


def _overlap_area(a, b) -> int:
    """Area shared by two exclusive ``(left, top, right, bottom)`` rects."""
    w = min(a[2], b[2]) - max(a[0], b[0])
    h = min(a[3], b[3]) - max(a[1], b[1])
    return w * h if w > 0 and h > 0 else 0


def open_position(avail, size, occupied, *, margin: int = 12, step: int = 28,
                  cascade: int = 6,
                  prefer: tuple[str, ...] = ("right", "below", "left", "above")):
    """Pure geometry: where to open a new ``size=(w, h)`` window so it does not
    cover the ones already open.

    ``avail`` and each rect in ``occupied`` are exclusive
    ``(left, top, right, bottom)``. Candidates are tried in order -- beside the
    **most recently opened** occupied window on each preferred side, then beside
    each earlier one, then a cascade stepping down-right from the most recent --
    and the first that overlaps nothing wins. When every candidate overlaps
    something (a crowded or small screen) the least-overlapping one is returned,
    so the result degrades to "as visible as possible" rather than to a
    coin toss. Every candidate is clamped inside *avail* first, so the answer is
    always fully on-screen.

    Returns ``None`` when there is nothing to avoid -- the caller should then
    leave placement to Qt rather than invent one.
    """
    occupied = [tuple(int(v) for v in rect) for rect in occupied]
    if not occupied:
        return None
    vl, vt, vr, vb = avail
    w, h = size

    candidates = []
    for left, top, right, bottom in reversed(occupied):     # most recent first
        beside = {
            "right": (right + margin, top),
            "left": (left - margin - w, top),
            "below": (left, bottom + margin),
            "above": (left, top - margin - h),
        }
        candidates.extend(beside[side] for side in prefer if side in beside)
    last_left, last_top = occupied[-1][0], occupied[-1][1]
    candidates.extend((last_left + k * step, last_top + k * step)
                      for k in range(1, cascade + 1))

    best, best_cost = None, None
    for x, y in candidates:
        # ⚠ Judge the candidate where it was ASKED for, not where it would be
        # clamped to. Clamping a "to the right" that runs off the monitor slides
        # the window back over the very thing it was meant to clear -- so a
        # clamp-then-measure order reports a clean placement and delivers an
        # overlapping one.
        if (x >= vl and y >= vt and x + w <= vr and y + h <= vb
                and not any(_overlap_area((x, y, x + w, y + h), other)
                            for other in occupied)):
            return (x, y)
        # It did not fit as asked; keep its on-screen form as a fallback, for
        # the case where nothing fits and "least covered" is the best on offer.
        cx = min(max(x, vl), max(vl, vr - w))
        cy = min(max(y, vt), max(vt, vb - h))
        rect = (cx, cy, cx + w, cy + h)
        cost = sum(_overlap_area(rect, other) for other in occupied)
        if best_cost is None or cost < best_cost:
            best, best_cost = (cx, cy), cost
    return best


#: How far from dead centre a window may sit and still count as one nobody has
#: placed. Qt centres a top-level it was given no position for, and frame
#: margins / DPI rounding move that by a few pixels, not by fifty.
UNPLACED_TOLERANCE = 48


def looks_unplaced(rect, avail, *, tolerance: int = UNPLACED_TOLERANCE) -> bool:
    """Whether *rect* is still where Qt puts a window nobody has positioned.

    ⚠ This is what keeps tiling from yanking a window the user arranged: it is
    fair to move a neighbour that only just appeared at the default spot, and
    not fair to move one somebody dragged where they wanted it.
    """
    vl, vt, vr, vb = avail
    w, h = rect[2] - rect[0], rect[3] - rect[1]
    return (abs(rect[0] - (vl + ((vr - vl) - w) // 2)) <= tolerance
            and abs(rect[1] - (vt + ((vb - vt) - h) // 2)) <= tolerance)


def tile_pair(avail, size_a, size_b, *, margin: int = 12, anchor=None):
    """Pure geometry: top-lefts that put two windows beside or below each other.

    Side by side is tried first -- two views of the same data read better that
    way -- then stacked, which is what a **portrait** monitor has room for: two
    880x920 windows do not fit across a 1440-wide screen but do fit down a
    2560-tall one. The pair is centred on the axis it is tiled along, so it
    reads as a deliberate layout rather than a shove against one edge, and keeps
    *anchor*'s coordinate (clamped) on the other axis so it stays where the eye
    already is.

    Returns ``None`` when neither axis fits the pair; the caller then keeps its
    single-window placement rather than shuffling two windows for no gain.
    """
    vl, vt, vr, vb = avail
    (wa, ha), (wb, hb) = size_a, size_b

    def clamped(value, low, high):
        return min(max(value, low), max(low, high))

    if wa + margin + wb <= vr - vl:
        x = vl + ((vr - vl) - (wa + margin + wb)) // 2
        top = vt if anchor is None else anchor[1]
        return ((x, clamped(top, vt, vb - ha)),
                (x + wa + margin, clamped(top, vt, vb - hb)))
    if ha + margin + hb <= vb - vt:
        y = vt + ((vb - vt) - (ha + margin + hb)) // 2
        left = vl if anchor is None else anchor[0]
        return ((clamped(left, vl, vr - wa), y),
                (clamped(left, vl, vr - wb), y + ha + margin))
    return None


def place_clear_of(window: QWidget, occupied, *, margin: int = 12,
                   step: int = 28,
                   prefer: tuple[str, ...] = ("right", "below", "left", "above")
                   ) -> bool:
    """Move *window* so it does not cover the widgets in *occupied* (best effort).

    Thin Qt wrapper over :func:`open_position`. Dead C++ wrappers and hidden
    windows are skipped, so the caller may hand over whole registries. Returns
    whether the window was moved; never throws.
    """
    try:
        from PyQt6.QtGui import QCursor, QGuiApplication

        rects, anchors, screen = [], [], None
        for other in occupied:
            try:
                if other is window or other is None or not other.isVisible():
                    continue
                geometry = other.frameGeometry()
                if screen is None:
                    screen = other.window().screen()
            except RuntimeError:                    # C++ object already deleted
                continue
            except Exception:
                continue
            rect = (geometry.left(), geometry.top(),
                    geometry.left() + geometry.width(),
                    geometry.top() + geometry.height())
            rects.append(rect)
            anchors.append((other, rect))
        if not rects:
            return False
        if screen is None:
            screen = (QGuiApplication.screenAt(QCursor.pos())
                      or QGuiApplication.primaryScreen())
        if screen is None:
            return False

        av = screen.availableGeometry()
        avail = (av.left(), av.top(), av.left() + av.width(), av.top() + av.height())
        frame = window.frameGeometry()
        size = (frame.width(), frame.height())
        placed = open_position(avail, size, rects,
                               margin=margin, step=step, prefer=prefer)
        if placed is None:
            return False

        def _move(widget, xy):
            f = widget.frameGeometry()
            widget.move(xy[0] + (widget.x() - f.x()), xy[1] + (widget.y() - f.y()))

        # ⚠ Two 880x920 windows do fit on a 1920x1040 screen -- but not if the
        # first one is sitting in the middle, which is exactly where Qt centres
        # a window it was given no position for. Placing only the newcomer then
        # leaves it half over its neighbour with nowhere better to go. So when
        # there is exactly ONE window in the way and the pair would fit tiled,
        # both are moved. One window, deliberately: re-arranging a busy desktop
        # because a new plot opened would be worse than the overlap.
        clear = not any(_overlap_area((placed[0], placed[1],
                                       placed[0] + size[0], placed[1] + size[1]),
                                      other) for other in rects)
        if (not clear and len(rects) == 1 and anchors
                and looks_unplaced(anchors[0][1], avail)):
            other, rect = anchors[0]
            pair = tile_pair(avail, (rect[2] - rect[0], rect[3] - rect[1]),
                             size, margin=margin, anchor=(rect[0], rect[1]))
            if pair is not None:
                _move(other, pair[0])
                _move(window, pair[1])
                return True

        _move(window, placed)
        return True
    except Exception:
        return False


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
