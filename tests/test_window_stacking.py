"""A dataset's windows must not open on top of each other, and after a `.msr`
import they must be stacked the way the reader lists the file's contents.

The report: opening a `.msr` that carries image series showed the render view
and then, immediately, an image viewer covering it exactly -- both are 880x920
and neither positions itself, so Qt opened them at the same spot.
"""

from __future__ import annotations

import sys

import pytest

from minflux_viewer.ui.modeless import open_position

SCREEN = (0, 0, 1920, 1040)


def _overlaps(a, b) -> bool:
    return (min(a[2], b[2]) - max(a[0], b[0]) > 0
            and min(a[3], b[3]) - max(a[1], b[1]) > 0)


def _rect(xy, size):
    return (xy[0], xy[1], xy[0] + size[0], xy[1] + size[1])


# ------------------------------------------------------------------ geometry
def test_nothing_to_avoid_means_no_opinion():
    """Qt's own placement is left alone rather than replaced by a guess."""
    assert open_position(SCREEN, (880, 920), []) is None


def test_the_second_880x920_window_lands_beside_the_first_not_on_it():
    """The reported case, in numbers: two identically sized windows."""
    render = (0, 0, 880, 920)
    at = open_position(SCREEN, (880, 920), [render])
    assert at is not None
    assert not _overlaps(_rect(at, (880, 920)), render)
    # 'right' is the first preference and two of them do fit side by side
    assert at == (render[2] + 12, render[1])


def test_a_candidate_that_would_hang_off_the_screen_is_not_slid_back_over_it():
    """⚠ The bug this class of code invites: clamping a "to the right" that
    does not fit slides the window back onto what it was clearing. Two 880-wide
    windows do not both fit right of x=500 on a 1920 screen, so the answer is
    the least-covered spot -- and it must not be the one that covers most."""
    render = (500, 60, 500 + 880, 60 + 920)
    at = open_position(SCREEN, (880, 920), [render])
    assert at is not None
    covered = (min(at[0] + 880, render[2]) - max(at[0], render[0])) * 920
    assert 0 < covered < 880 * 920, "it should be partly clear, not stacked"
    assert at == (1040, 60), "flush to the right edge, the most visible option"


def test_a_third_window_clears_both():
    """Every occupied rect is a candidate source, not just the newest one."""
    size = (600, 400)
    first = (0, 0, 600, 500)
    second = (612, 0, 1212, 500)
    at = open_position(SCREEN, size, [first, second])
    rect = _rect(at, size)
    assert not _overlaps(rect, first) and not _overlaps(rect, second)
    assert at == (second[2] + 12, second[1])


def test_it_sits_beside_the_most_recent_window():
    """Opening order is what the user is looking at, so it anchors there."""
    old = (0, 0, 300, 300)
    recent = (1000, 500, 1300, 800)
    at = open_position(SCREEN, (300, 300), [old, recent])
    assert at == (recent[2] + 12, recent[1])


def test_every_answer_is_fully_on_screen():
    """Including when the only free space is off the edge of the monitor."""
    against = (1500, 700, 1900, 1000)
    for size in [(880, 920), (400, 300), (1900, 1000)]:
        at = open_position(SCREEN, size, [against])
        rect = _rect(at, size)
        assert rect[0] >= SCREEN[0] and rect[1] >= SCREEN[1]
        assert rect[2] <= SCREEN[2] and rect[3] <= SCREEN[3]


def test_a_full_screen_falls_back_to_the_least_covered_spot():
    """⚠ It must still answer. Refusing to place would leave the window exactly
    where the collision was, which is the bug."""
    filling = (0, 0, 1920, 1040)
    at = open_position(SCREEN, (880, 920), [filling])
    assert at is not None
    rect = _rect(at, (880, 920))
    assert rect[0] >= 0 and rect[2] <= 1920
    # the cascade reaches a corner, so it is not simply the same place again
    assert at != (filling[0], filling[1])


def test_the_cascade_only_runs_when_no_side_is_free():
    """A step-and-repeat offset is the fallback, not the first idea: a window
    tucked beside another is easier to read than one shingled over it."""
    single = (700, 300, 1000, 600)
    assert open_position(SCREEN, (300, 300), [single]) == (1012, 300)


def test_two_render_sized_windows_tile_across_a_1920_screen():
    """They fit -- 880 + 12 + 880 = 1772 -- but only if neither is centred."""
    from minflux_viewer.ui.modeless import tile_pair

    pair = tile_pair(SCREEN, (880, 920), (880, 920), anchor=(0, 60))
    assert pair is not None
    (ax, ay), (bx, by) = pair
    assert not _overlaps((ax, ay, ax + 880, ay + 920), (bx, by, bx + 880, by + 920))
    assert ax >= 0 and bx + 880 <= 1920
    assert ay == by == 60, "both keep the height the eye is already at"
    assert ax + 880 + 12 == bx


def test_a_portrait_monitor_tiles_the_pair_top_and_bottom():
    """⚠ This machine has one: 1440x2560. Two 880-wide windows do not fit
    across it and two 920-tall ones do fit down it, so a horizontal-only rule
    left the second window half over the first with room going unused."""
    from minflux_viewer.ui.modeless import tile_pair

    portrait = (-1440, -572, 0, 1940)                 # a real second monitor
    pair = tile_pair(portrait, (880, 951), (880, 951), anchor=(-1159, 193))
    assert pair is not None
    (ax, ay), (bx, by) = pair
    assert not _overlaps((ax, ay, ax + 880, ay + 951), (bx, by, bx + 880, by + 951))
    assert ax == bx, "stacked, so they share a left edge"
    assert ay + 951 + 12 == by
    assert ay >= -572 and by + 951 <= 1940


def test_a_pair_that_fits_neither_way_is_not_tiled():
    """Shuffling two windows for no gain is worse than leaving one placed."""
    from minflux_viewer.ui.modeless import tile_pair

    assert tile_pair(SCREEN, (1200, 800), (1200, 800)) is None


def test_tiling_clamps_a_window_taller_than_the_screen():
    from minflux_viewer.ui.modeless import tile_pair

    (_ax, ay), (_bx, by) = tile_pair(SCREEN, (400, 1400), (400, 300),
                                     anchor=(0, 900))
    assert ay == 0, "a window taller than the screen starts at the top"
    assert by == 740, "a short one is pushed up only as far as it must be"


# -------------------------------------------------------------------- Qt side
@pytest.fixture
def _app():
    pytest.importorskip("PyQt6")
    from PyQt6.QtWidgets import QApplication
    return QApplication.instance() or QApplication(sys.argv)


def _window(app, x, y, w=300, h=200, title=""):
    from PyQt6.QtWidgets import QWidget

    win = QWidget()
    win.setWindowTitle(title)
    win.resize(w, h)
    win.move(x, y)
    win.show()
    app.processEvents()
    return win


def test_place_clear_of_moves_the_window_off_its_neighbour(_app):
    from minflux_viewer.ui.modeless import place_clear_of

    first = _window(_app, 200, 200)
    second = _window(_app, 200, 200)
    try:
        assert place_clear_of(second, [first]) is True
        _app.processEvents()
        a, b = first.frameGeometry(), second.frameGeometry()
        assert not a.intersects(b), (a, b)
    finally:
        first.close()
        second.close()


class _StubScreen:
    """⚠ The screen a test runs on is not the screen the user has. Under pytest
    Qt reported ~808x804 here, so a test written against the real monitor passed
    alone and failed in the suite. These stubs pin the geometry instead."""

    def __init__(self, rect):
        self._rect = rect

    def availableGeometry(self):
        return self._rect


class _StubWindow:
    """Widget-shaped stand-in -- ``place_clear_of`` only needs geometry, a
    screen and ``move``. Frame margins are zero, so a move lands exactly."""

    def __init__(self, x, y, w, h, screen, visible=True):
        from PyQt6.QtCore import QRect

        self._rect = QRect(x, y, w, h)
        self._screen, self._visible = screen, visible

    def isVisible(self):
        return self._visible

    def frameGeometry(self):
        from PyQt6.QtCore import QRect

        return QRect(self._rect)

    def window(self):
        return self

    def screen(self):
        return self._screen

    def x(self):
        return self._rect.x()

    def y(self):
        return self._rect.y()

    def move(self, x, y):
        self._rect.moveTo(x, y)

    def rect(self):
        r = self._rect
        return (r.x(), r.y(), r.x() + r.width(), r.y() + r.height())


def _stub_pair(_app, *, w=880, h=920, screen_w=1920, screen_h=1040):
    from PyQt6.QtCore import QRect

    screen = _StubScreen(QRect(0, 0, screen_w, screen_h))
    centred = ((screen_w - w) // 2, 60)          # where Qt puts an unplaced window
    return (_StubWindow(*centred, w, h, screen),
            _StubWindow(*centred, w, h, screen))


def test_the_neighbour_moves_too_when_that_is_what_makes_both_visible(_app):
    """⚠ The reported case. Qt centres a window it was given no position for, so
    placing only the newcomer leaves it half over the one already there with
    nowhere better to go: 880 + 12 + 880 fits across 1920, but not starting from
    the middle. With exactly one window in the way, both are tiled."""
    from minflux_viewer.ui.modeless import place_clear_of

    first, second = _stub_pair(_app)
    before = first.rect()
    assert place_clear_of(second, [first]) is True
    assert not _overlaps(first.rect(), second.rect())
    assert first.rect() != before, "the neighbour moved as well"
    for rect in (first.rect(), second.rect()):
        assert rect[0] >= 0 and rect[2] <= 1920


def test_a_crowded_desktop_is_not_rearranged(_app):
    """Tiling is for one window in the way, not for reflowing a session."""
    from PyQt6.QtCore import QRect

    from minflux_viewer.ui.modeless import place_clear_of

    screen = _StubScreen(QRect(0, 0, 1920, 1040))
    a = _StubWindow(520, 60, 880, 920, screen)
    b = _StubWindow(560, 80, 880, 920, screen)
    win = _StubWindow(520, 60, 880, 920, screen)
    before = (a.rect(), b.rect())

    place_clear_of(win, [a, b])
    assert (a.rect(), b.rect()) == before


def test_a_neighbour_the_user_has_placed_is_never_moved(_app):
    """⚠ Tiling is only fair on a window that just appeared where Qt put it.
    Dragging a view somewhere and having the next one shove it back would be a
    worse bug than the overlap this is fixing."""
    from PyQt6.QtCore import QRect

    from minflux_viewer.ui.modeless import place_clear_of

    screen = _StubScreen(QRect(0, 0, 1920, 1040))
    parked = _StubWindow(30, 40, 880, 920, screen)      # nowhere near centred
    win = _StubWindow(520, 60, 880, 920, screen)
    before = parked.rect()

    place_clear_of(win, [parked])
    assert parked.rect() == before


def test_looks_unplaced_knows_the_default_spot():
    from minflux_viewer.ui.modeless import looks_unplaced

    avail = (0, 0, 1920, 1040)
    centred = (520, 60, 520 + 880, 60 + 920)
    assert looks_unplaced(centred, avail)
    # a few pixels of frame/DPI slop still counts
    assert looks_unplaced((530, 70, 530 + 880, 70 + 920), avail)
    assert not looks_unplaced((30, 40, 30 + 880, 40 + 920), avail)


def test_a_pair_too_wide_to_tile_leaves_the_neighbour_alone(_app):
    """Nothing is gained by moving both, so nothing is moved but the newcomer."""
    from PyQt6.QtCore import QRect

    from minflux_viewer.ui.modeless import place_clear_of

    screen = _StubScreen(QRect(0, 0, 1200, 1040))
    first = _StubWindow(150, 60, 900, 920, screen)
    win = _StubWindow(150, 60, 900, 920, screen)
    before = first.rect()

    place_clear_of(win, [first])
    assert first.rect() == before


def test_place_clear_of_survives_a_dead_or_hidden_neighbour(_app):
    """Registries are handed over whole, so they carry closed windows."""
    from PyQt6.QtWidgets import QWidget

    from minflux_viewer.ui.modeless import place_clear_of

    hidden = QWidget()
    hidden.resize(300, 200)                      # never shown
    live = _window(_app, 100, 100)
    win = _window(_app, 100, 100)
    try:
        # only `live` counts, and the call must not raise on the others
        assert place_clear_of(win, [hidden, None, live]) is True
        # nothing visible to avoid -> no opinion, no move
        before = win.pos()
        assert place_clear_of(win, [hidden, None]) is False
        assert win.pos() == before
    finally:
        for w in (hidden, live, win):
            w.close()


# ------------------------------------------------------------------ stacking
class _FakeWindow:
    """Records raise/activate order without needing a real window manager."""

    def __init__(self, name, log, visible=True):
        self.name, self._log, self._visible = name, log, visible
        self.activated = False

    def isVisible(self):
        return self._visible

    def raise_(self):
        self._log.append(self.name)

    def activateWindow(self):
        self.activated = True


class _DeadWindow(_FakeWindow):
    def isVisible(self):
        raise RuntimeError("wrapped C/C++ object has been deleted")


def _viewer(order_log, *, dead=False):
    """A MainWindow with only the attributes the stacking code touches."""
    from minflux_viewer.ui.main_window import MainWindow

    win = MainWindow.__new__(MainWindow)
    make = _DeadWindow if dead else _FakeWindow
    win._tiff_windows = {"f#obf": make("image", order_log)}
    win._render_windows = {0: _FakeWindow("render", order_log)}
    win._scatter_windows = {0: _FakeWindow("scatter", order_log)}
    win._histogram_windows = {}
    win._attr_windows = {}
    win._data_windows = {0: _FakeWindow("data-info", order_log)}
    win._mbm_windows = lambda: [_FakeWindow("mbm", order_log)]
    return win


def test_the_stack_follows_the_readers_own_order(_app):
    """mfx, then mbm, then image -- so on screen: data in front, image behind."""
    order: list[str] = []
    win = _viewer(order)
    win.restack_import_windows([0])

    assert order[0] == "image", "the image viewer must end up at the back"
    assert order[order.index("mbm")] == "mbm"
    assert order.index("image") < order.index("mbm") < order.index("render")
    # raise_() puts a window on TOP, so the last one raised is the front one,
    # and the render view is the view of the localizations themselves
    assert order[-1] == "render"


def test_only_the_front_window_takes_focus(_app):
    order: list[str] = []
    win = _viewer(order)
    win.restack_import_windows([0])
    activated = [w.name for reg in (win._tiff_windows, win._render_windows,
                                    win._scatter_windows, win._data_windows)
                 for w in reg.values() if w.activated]
    assert activated == ["render"]


def test_a_hidden_or_deleted_window_does_not_break_the_stack(_app):
    order: list[str] = []
    win = _viewer(order, dead=True)
    win.restack_import_windows([0])              # must not raise
    assert "image" not in order
    assert order[-1] == "render"


def test_stacking_an_import_that_produced_nothing_is_a_no_op(_app):
    from minflux_viewer.ui.main_window import MainWindow

    win = MainWindow.__new__(MainWindow)
    for name in ("_tiff_windows", "_render_windows", "_scatter_windows",
                 "_histogram_windows", "_attr_windows", "_data_windows"):
        setattr(win, name, {})
    win._mbm_windows = lambda: []
    win.restack_import_windows([])               # must not raise


def test_the_data_tier_puts_the_render_view_last(_app):
    """It is the view of the localizations; the plots derive from it."""
    order: list[str] = []
    win = _viewer(order)
    names = [w.name for w in win._dataset_windows([0])]
    assert names[-1] == "render"
    assert set(names) == {"data-info", "scatter", "render"}


def test_the_reader_restacks_after_it_closes():
    """⚠ Not before: closing a window hands activation to whatever the window
    manager picks next, which is the thing the restack undoes."""
    import inspect

    from minflux_viewer.plugins.msr_reader import msr_reader_dialog as mrd

    source = inspect.getsource(mrd.MsrReaderDialog._on_open_in_viewer)
    assert "restack_import_windows" in source
    assert mrd.RESTACK_DELAY_MS > 0, "a zero delay races the close"
    # scheduled, not called inline -- the singleShot wraps the call
    call = source.index("p.restack_import_windows(i)")
    assert "QTimer.singleShot" in source[max(0, call - 300):call]
