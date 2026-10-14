"""What happens to everything that refers to a dataset when another one closes.

Datasets are addressed by their position in ``AppState.datasets``, and closing
one shifts every later one down. The main window re-keys its registries and
windows by hand, ROI scopes and the Restore ROI memory are remapped, and the
post-load steps re-resolve their dataset by identity. Stabilization Phase 3
replaces the position with a stable key; these tests pin today's behaviour so
that change can be checked against it.

Where today's behaviour is wrong the test states what it should be and is
marked ``xfail(strict=True)`` with the slice that fixes it, so the suite stays
green now and fails loudly the moment a fix lands without removing the mark:

* 3.4 -- closing one dataset retires every dataset's running post-load task,
  and a delayed post-load start is scheduled with an index.

(3.3 fixed the third: a Log event was matched to a dataset by the index it was
written with. Its tests below now pin the fixed behaviour.)

Writing these found a third, worse defect, fixed at once rather than marked:
a window unregistered itself by the index it was opened at, so closing an
earlier dataset dropped the next dataset's windows from their registries --
and with them their last Python reference, so a garbage collection destroyed
them on screen and aborted the process (tests/test_window_registry_identity.py).
The registry tests below now pin the fixed behaviour.
"""

from __future__ import annotations

import threading

import numpy as np
import pytest

pytest.importorskip("PyQt6")

from PyQt6 import sip
from PyQt6.QtCore import QCoreApplication, QEvent

from minflux_viewer.analysis import method_text
from minflux_viewer.core.app_state import AppState
from minflux_viewer.core.dataset import build_localization_dataset
from minflux_viewer.core.overlay import is_multichannel_overlay, overlay_members
from minflux_viewer.core.roi import RoiRecord
from minflux_viewer.core.roi_scope import (
    ORPHANED_DATASET,
    SHARED_DATASETS_KEY,
    roi_owner_dataset,
    roi_shared_datasets,
)
from minflux_viewer.ui.main_window import MainWindow

#: Window kinds and the main-window registry each lives in.
REGISTRIES = {
    "data": "_data_windows",
    "render": "_render_windows",
    "scatter": "_scatter_windows",
    "histogram": "_histogram_windows",
    "attribute": "_attr_windows",
    "filter": "_filter_dlgs",
}


def _dataset(name: str, offset: float, *, group: str | None = None, order: int = 1):
    rng = np.random.default_rng(sum(map(ord, name)))
    xyz = rng.normal(size=(300, 3)) * 80.0 + offset
    tid = np.repeat(np.arange(60), 5)
    ds = build_localization_dataset(
        name=name, x_nm=xyz[:, 0], y_nm=xyz[:, 1], z_nm=xyz[:, 2], tid=tid,
        source_version="simulation")
    if group:
        ds.state.update(overlay_id=group, render_group_id=group, overlay_order=order,
                        overlay_index=order)
    return ds


def _flush():
    """Run queued events and the deferred deletes of closed windows."""
    for _ in range(3):
        QCoreApplication.processEvents()
        QCoreApplication.sendPostedEvents(None, QEvent.Type.DeferredDelete)


def _index_of(win: MainWindow, ds) -> int:
    for i, other in enumerate(win._state.datasets):
        if other is ds:
            return i
    raise AssertionError(f"{ds.name} is not open")


def _window_index(window):
    for name in ("_idx", "_dataset_idx"):
        value = getattr(window, name, None)
        if isinstance(value, int):
            return value
    return None


def _main(prefs_update: dict | None = None) -> MainWindow:
    state = AppState()
    data = state.prefs.setdefault("data", {})
    # Post-load computations off unless a test asks for them: they run on a
    # timer and a worker, and these tests are about what refers to what.
    data.update(compute_local_density=False, compute_loc_prec=False,
                estimate_z_scaling_factor=False, show_dataset_manager=False)
    state.prefs.setdefault("plot", {})["use_fixed_z_scaling_factor"] = False
    for section, values in (prefs_update or {}).items():
        state.prefs.setdefault(section, {}).update(values)
    return MainWindow(state)


def _teardown(win: MainWindow) -> None:
    """Close every per-dataset window still open, then the main window.

    Found as top-level widgets rather than through the registries, so a window
    a registry has lost track of cannot stay open into the next test."""
    from PyQt6.QtWidgets import QApplication

    from minflux_viewer.ui.attribute_window import AttributeWindow
    from minflux_viewer.ui.data_window import DataWindow
    from minflux_viewer.ui.filter_dialog import FilterDialog
    from minflux_viewer.ui.histogram_window import HistogramWindow
    from minflux_viewer.ui.render_window import RenderWindow
    from minflux_viewer.ui.scatter_window import ScatterWindow

    kinds = (AttributeWindow, DataWindow, FilterDialog, HistogramWindow,
             RenderWindow, ScatterWindow)
    for widget in QApplication.topLevelWidgets():
        try:
            if isinstance(widget, kinds) and not sip.isdeleted(widget):
                widget.close()
        except RuntimeError:
            pass
    _flush()
    win.close()
    _flush()


@pytest.fixture
def session(qapp):
    """Four open datasets, every per-dataset window open for each."""
    win = _main()
    names = ["A", "B", "C", "D"]
    datasets = {}
    for i, name in enumerate(names):
        datasets[name] = _dataset(name, 1000.0 * i)
        win._state.add_dataset(datasets[name])
    opened = {}
    for name in names:
        i = _index_of(win, datasets[name])
        win._state.set_active(i)
        opened[name] = {
            "data": win._data_windows[i],
            "render": win._show_render(i),
            "scatter": win._show_scatter(i),
            "histogram": win._show_histogram(i),
            "attribute": win._show_attr_plot(i),
            "filter": win._show_filter(),
        }
    _flush()
    try:
        yield win, datasets, opened
    finally:
        _teardown(win)


def _close_earlier_current_and_later(win, datasets, check):
    """Close A (earlier than the active C), then C (the active one), then D
    (later than the active B), calling *check* with the closed set each time."""
    state = win._state
    state.set_active(_index_of(win, datasets["C"]))
    win.close_datasets([_index_of(win, datasets["A"])])
    _flush()
    assert state.active_dataset is datasets["C"]
    check({"A"})

    win.close_datasets([state.active_idx])
    _flush()
    assert state.active_dataset is datasets["D"]
    check({"A", "C"})

    state.set_active(_index_of(win, datasets["B"]))
    win.close_datasets([_index_of(win, datasets["D"])])
    _flush()
    assert state.active_dataset is datasets["B"]
    check({"A", "C", "D"})


# --------------------------------------------------------------- windows
def test_every_window_keeps_showing_its_dataset_through_closes(session):
    win, datasets, opened = session

    def check(closed):
        for name, ds in datasets.items():
            for kind, window in opened[name].items():
                if name in closed:
                    assert sip.isdeleted(window), (name, kind)
                else:
                    assert not sip.isdeleted(window), (name, kind)
                    assert _window_index(window) == _index_of(win, ds), (name, kind)

    _close_earlier_current_and_later(win, datasets, check)


def test_the_registries_hold_each_open_window_under_its_dataset(session):
    win, datasets, opened = session

    def check(closed):
        for name, ds in datasets.items():
            for kind, attr in REGISTRIES.items():
                registry = getattr(win, attr)
                window = opened[name][kind]
                if name in closed:
                    assert window not in registry.values(), (name, kind)
                else:
                    assert registry.get(_index_of(win, ds)) is window, (name, kind)

    _close_earlier_current_and_later(win, datasets, check)


def test_closing_a_dataset_closes_its_windows_after_an_earlier_close(session):
    win, datasets, opened = session
    win.close_datasets([_index_of(win, datasets["A"])])
    _flush()
    win.close_datasets([_index_of(win, datasets["B"])])
    _flush()
    left_open = [kind for kind, w in opened["B"].items() if not sip.isdeleted(w)]
    assert not left_open


def test_closing_a_reindexed_window_unregisters_only_itself(session):
    win, datasets, opened = session
    win.close_datasets([_index_of(win, datasets["A"])])
    _flush()

    opened["B"]["scatter"].close()                     # created at 1, now at 0
    _flush()

    # C's scatter (now at 1) must still be the one registered for C ...
    assert win._scatter_windows.get(_index_of(win, datasets["C"])) is opened["C"]["scatter"]
    # ... and B must get a fresh window rather than a deleted one.
    fresh = win._show_scatter(_index_of(win, datasets["B"]))
    assert not sip.isdeleted(fresh) and fresh is not opened["B"]["scatter"]


# --------------------------------------------------------------- overlays
@pytest.fixture
def overlay_session(qapp):
    """A and B standalone, C and D one overlay drawn by C's render view."""
    win = _main()
    ds = {"A": _dataset("A", 0.0), "B": _dataset("B", 1000.0),
          "C": _dataset("C", 2000.0, group="grp", order=1),
          "D": _dataset("D", 2000.0, group="grp", order=2)}
    for name in "ABCD":
        win._state.add_dataset(ds[name])
    _flush()
    views = dict(win._render_windows)
    view = views[_index_of(win, ds["C"])]
    try:
        yield win, ds, view
    finally:
        _teardown(win)


def _close_the_two_standalone(win, ds):
    win.close_datasets([_index_of(win, ds["A"])])
    win.close_datasets([_index_of(win, ds["B"])])
    _flush()


def test_an_overlay_keeps_its_members_and_its_channels_when_others_close(overlay_session):
    win, ds, view = overlay_session
    _close_the_two_standalone(win, ds)

    anchor = _index_of(win, ds["C"])
    assert is_multichannel_overlay(win._state, anchor)
    assert [d for _, d in overlay_members(win._state, anchor)] == [ds["C"], ds["D"]]
    assert not sip.isdeleted(view)
    channels = [c.get("dataset_idx") for c in view._channels]
    assert [win._state.datasets[i] for i in channels] == [ds["C"], ds["D"]]


def test_an_overlay_view_stays_registered_when_others_close(overlay_session):
    win, ds, view = overlay_session
    _close_the_two_standalone(win, ds)
    assert win._render_windows.get(_index_of(win, ds["C"])) is view


# --------------------------------------------------------------- ROIs
def _roi(owner: int, shared=()):
    context = {"source_view": "render", "dataset_idx": owner}
    if shared:
        context[SHARED_DATASETS_KEY] = list(shared)
    return RoiRecord.create("rectangle", {"bounds": [0.0, 0.0, 10.0, 10.0]},
                            context=context)


def test_roi_scopes_and_restore_memory_follow_their_datasets(session):
    win, datasets, _opened = session
    state = win._state
    idx = {name: _index_of(win, d) for name, d in datasets.items()}
    on_a, on_b = _roi(idx["A"]), _roi(idx["B"], shared=[idx["A"], idx["D"]])
    on_c = _roi(idx["C"])
    for record in (on_a, on_b, on_c):
        state.rois.add(record)
    win._roi_last_active = {idx["B"]: _roi(idx["B"]), idx["C"]: _roi(idx["C"])}

    win.close_datasets([idx["A"]])
    _flush()

    records = {r.id: r for r in state.rois.records}
    assert roi_owner_dataset(records[on_a.id]) == ORPHANED_DATASET
    assert state.datasets[roi_owner_dataset(records[on_b.id])] is datasets["B"]
    assert [state.datasets[i] for i in roi_shared_datasets(records[on_b.id])] == [datasets["D"]]
    assert state.datasets[roi_owner_dataset(records[on_c.id])] is datasets["C"]

    memory = win._roi_last_active
    assert sorted(memory) == [_index_of(win, datasets["B"]), _index_of(win, datasets["C"])]
    for key, record in memory.items():
        assert roi_owner_dataset(record) == key


# --------------------------------------------------------------- the Log
def _load_event(state, name):
    marker = f"Loaded dataset '{name}':"
    events = [e for e in state.log_history if e["message"].startswith(marker)]
    assert len(events) == 1, name
    return events[0]


def test_a_log_event_names_its_dataset_while_nothing_has_closed(session):
    win, datasets, _opened = session
    for name, ds in datasets.items():
        assert method_text._ds_for(win._state, _load_event(win._state, name)) is ds


def test_a_log_event_keeps_its_dataset_after_an_earlier_one_closes(session):
    win, datasets, _opened = session
    event = _load_event(win._state, "B")
    win.close_datasets([_index_of(win, datasets["A"])])
    _flush()
    assert method_text._ds_for(win._state, event) is datasets["B"]


def test_a_log_event_of_a_closed_dataset_names_no_open_dataset(session):
    win, datasets, _opened = session
    event = _load_event(win._state, "A")
    win.close_datasets([_index_of(win, datasets["A"])])
    _flush()
    assert method_text._ds_for(win._state, event) is None


# --------------------------------------------------------------- post-load
_DENSITY_ON = {"data": {"compute_local_density": True}}


def test_a_closed_dataset_never_receives_its_post_load_result(qapp, qtbot, monkeypatch):
    import minflux_viewer.core.post_load as post_load

    real = post_load.compute_post_load_results
    started, release = threading.Event(), threading.Event()

    def held(ds, prefs, report):
        if ds.name == "A":
            started.set()
            release.wait(10)
        return real(ds, prefs, report)

    monkeypatch.setattr(post_load, "compute_post_load_results", held)
    win = _main(_DENSITY_ON)
    try:
        a = _dataset("A", 0.0)
        win._state.add_dataset(a)
        qtbot.waitUntil(started.is_set, timeout=5000)
        win.close_datasets([0])
        release.set()
        qtbot.wait(500)
        _flush()
        assert "den" not in a.attr
    finally:
        release.set()
        _teardown(win)


@pytest.mark.xfail(strict=True, reason="slice 3.4: closing any dataset retires every "
                   "dataset's running post-load task")
def test_closing_one_dataset_leaves_another_datasets_post_load_running(qapp, qtbot,
                                                                        monkeypatch):
    import minflux_viewer.core.post_load as post_load

    real = post_load.compute_post_load_results
    started, release = threading.Event(), threading.Event()

    def held(ds, prefs, report):
        if ds.name == "B":
            started.set()
            release.wait(10)
        return real(ds, prefs, report)

    monkeypatch.setattr(post_load, "compute_post_load_results", held)
    win = _main(_DENSITY_ON)
    try:
        a, b = _dataset("A", 0.0), _dataset("B", 1000.0)
        win._state.add_dataset(a)
        qtbot.waitUntil(lambda: "den" in a.attr, timeout=5000)
        win._state.add_dataset(b)
        qtbot.waitUntil(started.is_set, timeout=5000)
        win.close_datasets([_index_of(win, a)])
        release.set()
        qtbot.waitUntil(lambda: "den" in b.attr, timeout=3000)
    finally:
        release.set()
        _teardown(win)


@pytest.mark.xfail(strict=True, reason="slice 3.4: the delayed post-load start is "
                   "scheduled with the dataset's index")
def test_a_scheduled_post_load_reaches_its_dataset_after_an_earlier_close(qapp, qtbot):
    win = _main(_DENSITY_ON)
    try:
        a, b, c = _dataset("A", 0.0), _dataset("B", 1000.0), _dataset("C", 2000.0)
        win._state.add_dataset(a)
        qtbot.waitUntil(lambda: "den" in a.attr, timeout=5000)
        win._state.add_dataset(b)          # starts at index 1, after ~550 ms
        win._state.add_dataset(c)          # starts at index 2, after ~650 ms
        win.close_datasets([_index_of(win, a)])   # before either timer fires
        qtbot.waitUntil(lambda: "den" in b.attr and "den" in c.attr, timeout=3000)
    finally:
        _teardown(win)
