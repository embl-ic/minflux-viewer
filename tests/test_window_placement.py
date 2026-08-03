"""Pure geometry for placing a window beside another (ui/modeless.beside_position)."""

from __future__ import annotations

from minflux_viewer.ui.modeless import beside_position, corner_position

# 1920×1080 available screen (exclusive right/bottom).
AVAIL = (0, 0, 1920, 1080)


def _overlaps(a, size, xy):
    ax0, ay0, ax1, ay1 = a
    x, y = xy
    w, h = size
    return not (x >= ax1 or x + w <= ax0 or y >= ay1 or y + h <= ay0)


def test_prefers_right_when_room():
    anchor = (0, 0, 800, 600)                     # top-left
    x, y = beside_position(anchor, AVAIL, (500, 400), margin=12)
    assert (x, y) == (812, 0)                      # to the right, aligned top
    assert not _overlaps(anchor, (500, 400), (x, y))
    assert x + 500 <= 1920 and y + 400 <= 1080     # fully on-screen


def test_falls_to_below_when_no_room_right():
    anchor = (1200, 0, 1900, 600)                  # hugging the right edge
    x, y = beside_position(anchor, AVAIL, (500, 400), margin=12)
    assert (x, y) == (1200, 612)                   # right doesn't fit → below
    assert not _overlaps(anchor, (500, 400), (x, y))


def test_bottom_right_corner_clamps_on_screen():
    anchor = (1400, 700, 1900, 1060)               # bottom-right
    size = (600, 500)
    x, y = beside_position(anchor, AVAIL, size, margin=12)
    # nothing fits fully → clamped, but always on the monitor
    assert 0 <= x and x + size[0] <= 1920
    assert 0 <= y and y + size[1] <= 1080


def test_prefer_order_is_honoured():
    anchor = (400, 400, 800, 700)
    # force "below" first
    x, y = beside_position(anchor, AVAIL, (300, 200), margin=10,
                           prefer=("below", "right", "above", "left"))
    assert (x, y) == (400, 710)


def test_top_left_anchor_never_goes_offscreen_left_or_up():
    anchor = (0, 0, 300, 200)
    x, y = beside_position(anchor, AVAIL, (300, 200),
                           prefer=("left", "above", "right", "below"))
    # left/above are off-screen → must pick right/below (on-screen)
    assert x >= 0 and y >= 0
    assert not _overlaps(anchor, (300, 200), (x, y))


def test_corner_top_right_anchors_to_upper_right_with_margin():
    x, y = corner_position(AVAIL, (900, 150), align="top_right", margin=24)
    assert (x, y) == (1920 - 24 - 900, 24)          # right-aligned, near the top
    assert x + 900 <= 1920 and y >= 0


def test_corner_center_is_centered():
    x, y = corner_position(AVAIL, (600, 400), align="center")
    assert (x, y) == ((1920 - 600) // 2, (1080 - 400) // 2)


def test_corner_never_goes_offscreen_when_window_too_wide():
    # A window wider than the screen still clamps its top-left on-screen.
    x, y = corner_position(AVAIL, (2000, 150), align="top_right", margin=24)
    assert x == 0 and y == 24


def test_corner_fraction_places_window_toward_center():
    # (fx, fy) = fraction of the free space (CSS background-position semantics).
    # The main window uses (0.75, 0.25): 75 % toward the right, 25 % down.
    w, h = 1000, 700
    x, y = corner_position(AVAIL, (w, h), align=(0.75, 0.25))
    assert x == round(0.75 * (1920 - w))            # 690
    assert y == round(0.25 * (1080 - h))            # 95
    # strictly inside the top-right corner (moved left + down toward centre)
    tr_x, tr_y = corner_position(AVAIL, (w, h), align="top_right", margin=24)
    assert x < tr_x and y > tr_y
    assert 0 <= x and x + w <= 1920 and 0 <= y and y + h <= 1080


def test_corner_fraction_extremes_flush_and_onscreen():
    assert corner_position(AVAIL, (400, 300), align=(1.0, 1.0)) == (1520, 780)   # flush BR
    assert corner_position(AVAIL, (400, 300), align=(0.0, 0.0)) == (0, 0)         # flush TL


def test_corner_bottom_left():
    x, y = corner_position(AVAIL, (500, 300), align="bottom_left", margin=24)
    assert x == 24 and y == 1080 - 24 - 300
