"""A ROI belongs to one dataset and one family of axes.

Before this, ``RoiOverlayController.refresh`` drew every record of the store in
every view: a spatial ROI drawn on a render also appeared on an Attribute
Histogram (whose x axis is an attribute value and whose y axis is a count), and
opening a render for a second dataset immediately covered it in the first
dataset's ROIs.

Sharing across datasets stays possible, but only when the user asks for it.
"""

from __future__ import annotations

from types import SimpleNamespace

import pytest

from minflux_viewer.core import roi_scope
from minflux_viewer.core.roi import RoiRecord

pytest.importorskip("PyQt6")
pytest.importorskip("pyqtgraph")


def _rect(name="r", *, source_view=None, dataset_idx=None, shared=None):
    record = RoiRecord.create("rectangle", {"bounds": [0.0, 0.0, 10.0, 10.0]},
                              name=name)
    context = {}
    if source_view is not None:
        context["source_view"] = source_view
    if dataset_idx is not None:
        context["dataset_idx"] = dataset_idx
    if shared is not None:
        context["shared_datasets"] = list(shared)
    record.context = context
    return record


# --------------------------------------------------------------- the pure rule

def test_render_and_scatter_are_one_family_and_the_value_plots_are_not():
    """Restore ROI deliberately copies a draft render<->scatter, so they share."""
    assert roi_scope.view_family("render") == roi_scope.view_family("scatter")
    assert roi_scope.view_family("histogram") != roi_scope.view_family("render")
    assert roi_scope.view_family("attribute") != roi_scope.view_family("histogram")
    assert roi_scope.view_family("nonsense") is None


def test_a_spatial_roi_is_not_drawn_on_a_value_plot():
    record = _rect(source_view="render", dataset_idx=0)
    assert roi_scope.roi_visible_in(record, family="coordinate", dataset_indices={0})
    assert not roi_scope.roi_visible_in(record, family="histogram", dataset_indices={0})
    assert not roi_scope.roi_visible_in(record, family="attribute", dataset_indices={0})


def test_a_roi_does_not_follow_focus_onto_another_dataset():
    record = _rect(source_view="render", dataset_idx=0)
    assert roi_scope.roi_visible_in(record, family="coordinate", dataset_indices={0})
    assert not roi_scope.roi_visible_in(record, family="coordinate", dataset_indices={1})


def test_an_overlay_view_shows_every_channels_roi_not_only_the_anchors():
    """An overlay render is keyed by its anchor but draws several datasets."""
    channel = _rect(source_view="render", dataset_idx=2)
    assert roi_scope.roi_visible_in(
        channel, family="coordinate", dataset_indices={0, 1, 2})


def test_a_roi_with_no_recorded_scope_is_still_shown():
    """Saved before this existed: hiding a user's ROI is worse than over-showing."""
    record = _rect()
    assert roi_scope.roi_dataset_indices(record) is None
    assert roi_scope.roi_family(record) is None
    for family in ("coordinate", "histogram", "attribute"):
        assert roi_scope.roi_visible_in(record, family=family, dataset_indices={7})


def test_sharing_is_explicit_additive_and_reversible():
    record = _rect(source_view="render", dataset_idx=0)
    assert roi_scope.roi_shared_datasets(record) == ()

    assert roi_scope.share_with_dataset(record, 1) is True
    assert roi_scope.share_with_dataset(record, 1) is False        # idempotent
    assert roi_scope.share_with_dataset(record, 0) is False        # already the owner
    assert roi_scope.roi_shared_datasets(record) == (1,)
    assert roi_scope.roi_visible_in(record, family="coordinate", dataset_indices={1})
    # Sharing crosses datasets, never families.
    assert not roi_scope.roi_visible_in(record, family="histogram", dataset_indices={1})

    assert roi_scope.unshare_dataset(record, 1) is True
    assert roi_scope.roi_shared_datasets(record) == ()
    assert not roi_scope.roi_visible_in(record, family="coordinate", dataset_indices={1})
    # The owning dataset is never removed: a ROI shown nowhere looks deleted.
    assert roi_scope.roi_visible_in(record, family="coordinate", dataset_indices={0})


def test_recomputing_a_selection_never_retargets_or_unshares_the_roi():
    """``store_roi_mask`` replaces the whole context; the scope must survive."""
    record = _rect(source_view="render", dataset_idx=0)
    roi_scope.share_with_dataset(record, 1)
    merged = roi_scope.scope_context(
        record.context,
        {"source_view": "render", "dataset_idx": 1, "orientation": "XY"})
    assert merged["dataset_idx"] == 0                  # owner, not the shared one
    assert merged["shared_datasets"] == [1]
    assert merged["orientation"] == "XY"               # the new details survive


# ------------------------------------------------------- the live controller

def _controller(source_view, dataset_indices, store, qtbot=None):
    import pyqtgraph as pg
    from PyQt6.QtWidgets import QWidget

    from minflux_viewer.ui.roi_overlay import RoiOverlayController

    class _Owner(QWidget):
        def __init__(self):
            super().__init__()
            self._state = SimpleNamespace(
                prefs={"plot": {"roi_color": "Yellow"}}, datasets=[])

        def roi_dataset_indices(self):
            return set(dataset_indices)

        def roi_view_plane(self):
            return "XY"

        def roi_depth_center(self):
            return 0.0

        def normalize_roi_record(self, record):
            return record

    plot = pg.PlotWidget()
    owner = _Owner()
    if qtbot is not None:                      # let pytest-qt own the teardown
        qtbot.addWidget(owner)
        qtbot.addWidget(plot)
    return RoiOverlayController(store, owner, plot, plot.getPlotItem(),
                                source_view=source_view)


@pytest.fixture(scope="module")
def _app(qapp):
    return qapp


def test_views_draw_only_the_rois_in_their_own_scope(_app, qtbot):
    from minflux_viewer.core.roi import RoiStore

    store = RoiStore()
    render_a = _controller("render", {0}, store, qtbot)
    scatter_a = _controller("scatter", {0}, store, qtbot)
    render_b = _controller("render", {1}, store, qtbot)
    histogram_a = _controller("histogram", {0}, store, qtbot)
    store.set_show_all(True)

    spatial = _rect("spatial", source_view="render", dataset_idx=0)
    band = _rect("band", source_view="histogram", dataset_idx=0)
    store.add(spatial)
    store.add(band)

    assert spatial.id in render_a.items          # its own view
    assert spatial.id in scatter_a.items         # same coordinate family
    assert spatial.id not in render_b.items      # the other dataset
    assert spatial.id not in histogram_a.items   # the reported defect
    assert band.id in histogram_a.items
    assert band.id not in render_a.items



def test_sharing_from_the_manager_reveals_it_on_the_other_dataset(_app, qtbot):
    from minflux_viewer.core.roi import RoiStore

    store = RoiStore()
    render_a = _controller("render", {0}, store, qtbot)
    render_b = _controller("render", {1}, store, qtbot)
    store.set_show_all(True)
    record = _rect("spatial", source_view="render", dataset_idx=0)
    store.add(record)
    assert record.id not in render_b.items

    roi_scope.share_with_dataset(record, 1)
    store.changed.emit()                         # what the Manager command does
    assert record.id in render_b.items
    assert record.id in render_a.items            # still on its own dataset

    roi_scope.unshare_dataset(record)
    store.changed.emit()
    assert record.id not in render_b.items
    assert record.id in render_a.items



def test_a_drawn_roi_is_scoped_at_creation_even_without_a_selection_pass(_app, qtbot):
    """A line / point / angle never runs ``compute_roi_selection``."""
    from minflux_viewer.core.roi import RoiStore

    controller = _controller("render", {3}, RoiStore(), qtbot)
    kwargs = controller._record_kwargs()
    record = RoiRecord.create("line", {"points": [[0, 0], [1, 1]]}, **kwargs)

    assert roi_scope.roi_family(record) == "coordinate"
    assert roi_scope.roi_owner_dataset(record) == 3


def test_out_of_scope_rois_are_not_hit_tested(_app, qtbot):
    """The click target must match what is drawn, or an invisible ROI is hit."""
    from minflux_viewer.core.roi import RoiStore

    store = RoiStore()
    histogram = _controller("histogram", {0}, store, qtbot)
    store.set_show_all(True)
    store.add(_rect("spatial", source_view="render", dataset_idx=0))

    assert histogram._record_at((5.0, 5.0)) is None


# ------------------------------------------------- dataset indices shift

def test_closing_a_dataset_shifts_roi_scopes_instead_of_re_attributing_them():
    """``remove_dataset`` pops from a list, so every later index shifts down."""
    keep_low = _rect(source_view="render", dataset_idx=0)
    closed = _rect(source_view="render", dataset_idx=1)
    later = _rect(source_view="render", dataset_idx=2, shared=[0, 3])
    much_later = _rect(source_view="render", dataset_idx=5)
    records = [keep_low, closed, later, much_later]

    assert roi_scope.remap_dataset_indices(records, 1) == 3

    assert roi_scope.roi_owner_dataset(keep_low) == 0          # before it: unchanged
    assert roi_scope.roi_owner_dataset(later) == 1             # after it: shifted down
    assert roi_scope.roi_owner_dataset(much_later) == 4
    assert roi_scope.roi_shared_datasets(later) == (0, 2)      # shared list shifts too

    # The closed dataset's own ROIs are kept and re-attachable, not deleted and
    # not silently un-scoped (which would make them reappear in every view).
    assert roi_scope.roi_owner_dataset(closed) == roi_scope.ORPHANED_DATASET
    assert not roi_scope.roi_visible_in(
        closed, family="coordinate", dataset_indices={0, 1, 2})
    assert roi_scope.share_with_dataset(closed, 0) is True
    assert roi_scope.roi_visible_in(closed, family="coordinate", dataset_indices={0})


def test_a_shared_entry_on_the_closed_dataset_is_simply_dropped():
    record = _rect(source_view="render", dataset_idx=0, shared=[1])
    assert roi_scope.remap_dataset_indices([record], 1) == 1
    assert roi_scope.roi_owner_dataset(record) == 0
    assert roi_scope.roi_shared_datasets(record) == ()
    assert "shared_datasets" not in record.context


def test_remapping_leaves_an_unscoped_roi_alone():
    record = _rect()
    assert roi_scope.remap_dataset_indices([record], 0) == 0
    assert record.context == {}


def test_a_selection_result_keeps_what_was_recorded_when_the_roi_was_drawn():
    """store_roi_mask replaces the context with the selection's. The facts
    stamped at drawing time -- the plane, the depth a flat ROI sits at, the Z
    scaling factor -- are not part of a selection and must survive it: the
    side panes draw a flat ROI edge-on at ``depth_value``, and Scale Z
    pre-fills from ``z_scaling_factor``, which cannot be recovered later."""
    drawn = {"source_view": "render", "dataset_idx": 0, "view_plane": "XZ",
             "depth_axis": "Y", "depth_value": 12.5, "z_scaling_factor": 0.67}
    selection = {"source_view": "render", "dataset_idx": 0, "orientation": "XZ",
                 "depth_axis": "Y", "depth_range": None,
                 "depth_inclusive": [True, True]}

    merged = roi_scope.scope_context(drawn, selection)

    assert merged["view_plane"] == "XZ"
    assert merged["depth_value"] == 12.5
    assert merged["z_scaling_factor"] == 0.67
    # ...while the selection's own details still come from the selection.
    assert merged["orientation"] == "XZ" and merged["depth_range"] is None


def test_a_fresh_selection_detail_wins_over_a_stale_one():
    old = {"dataset_idx": 0, "depth_range": [10.0, 20.0], "view_plane": "XY"}
    new = {"dataset_idx": 0, "depth_range": None}
    merged = roi_scope.scope_context(old, new)
    assert merged["depth_range"] is None and merged["view_plane"] == "XY"


def test_a_per_dataset_roi_cache_follows_its_datasets_when_one_closes():
    """*Restore ROI* remembers each dataset's last draft, keyed by index.

    Without re-keying, closing dataset 0 left dataset 2's draft under key 2,
    where *Restore ROI* on whatever then sat at index 2 brought it back.
    """
    first = _rect("first", source_view="render", dataset_idx=0)
    closed = _rect("closed", source_view="render", dataset_idx=1)
    later = _rect("later", source_view="scatter", dataset_idx=2)
    cache = {0: first, 1: closed, 2: later}

    remapped = roi_scope.remap_dataset_keyed(cache, 1)

    assert remapped == {0: first, 1: later}
    assert roi_scope.roi_owner_dataset(later) == 1   # its scope moves with it
    assert roi_scope.roi_owner_dataset(first) == 0
    assert cache == {0: first, 1: closed, 2: later}  # input left untouched


def test_closing_a_dataset_re_keys_the_restore_roi_memory(qapp):
    """The main window's removal handler applies it (a stand-in, not a real
    MainWindow: constructing one per test drags in every child window)."""
    from minflux_viewer.core.app_state import AppState
    from minflux_viewer.core.dataset_events import DatasetRemoved
    from minflux_viewer.core.dataset_key import DatasetKey
    from minflux_viewer.ui.dataset_binding import DatasetWindowMap
    from minflux_viewer.ui.main_window import MainWindow

    later = _rect("later", source_view="render", dataset_idx=2)
    calls = []
    window = SimpleNamespace(
        _state=SimpleNamespace(rois=SimpleNamespace(
            records=[], changed=SimpleNamespace(emit=lambda: None))),
        _roi_last_active={1: _rect("closed", dataset_idx=1), 2: later},
        _notify_view_state_changed=lambda: calls.append("notified"),
    )
    registry_state = AppState()
    for name in ("_data_windows", "_render_windows", "_scatter_windows",
                 "_tracking_windows", "_msd_windows", "_histogram_windows",
                 "_attr_windows", "_filter_dlgs"):
        setattr(window, name, DatasetWindowMap(registry_state))
    window._retire_dataset_tasks = lambda _event: None
    window._refresh_shifted_windows = (
        MainWindow._refresh_shifted_windows.__get__(window))

    MainWindow._on_dataset_removed(window, DatasetRemoved(
        key=DatasetKey(2), dataset=None, old_index=1, new_active_key=None))

    assert window._roi_last_active == {1: later}
    assert roi_scope.roi_owner_dataset(later) == 1
    assert calls == ["notified"]
