"""The in-ROI data highlight is one rule for every view of a dataset.

A ROI's stored artefact is one boolean per localization row, so "data in the
active ROI" means the same rows in render, scatter, the Attribute Plot and the
Histogram — whichever view drew it. These tests pin that, the row-alignment gate
that keeps a mask from being painted on rows it does not describe, and the
histogram's second bar series carrying true in-ROI counts.
"""

from __future__ import annotations

import os
import sys

import numpy as np
import pytest

pytest.importorskip("PyQt6")
pytest.importorskip("pyqtgraph")

from PyQt6.QtWidgets import QApplication

from minflux_viewer.core.app_state import AppState
from minflux_viewer.core.dataset import build_localization_dataset
from minflux_viewer.core.roi import RoiRecord
from minflux_viewer.core.roi_selection import store_roi_mask
from minflux_viewer.ui.attribute_window import AttributeWindow
from minflux_viewer.ui.histogram_window import HistogramWindow
from minflux_viewer.ui.roi_highlight import decimate, highlight_masks, union_mask


@pytest.fixture(scope="module")
def _app():
    if not os.environ.get("DISPLAY") and os.name != "nt" and sys.platform != "darwin":
        pytest.skip("No display available for Qt tests")
    return QApplication.instance() or QApplication([])


def _state(efo=None, n: int = 12) -> AppState:
    state = AppState()
    state.add_dataset(build_localization_dataset(
        name="highlight",
        x_nm=np.arange(n, dtype=float) * 10.0,
        y_nm=np.full(n, 5.0),
        z_nm=np.linspace(0.0, 100.0, n),
        tid=np.repeat([1, 2, 3], n // 3),
        tim=np.linspace(0.0, 1.0, n),
        attrs={
            "efo": np.linspace(1_000.0, 12_000.0, n) if efo is None else efo,
            "cfr": np.linspace(0.1, 0.9, n),
        },
    ))
    return state


def _add_mask_roi(state, ds, rows, *, source_view: str, name: str = "r") -> RoiRecord:
    """A stored ROI carrying *rows* as its selection, as if drawn in *source_view*."""
    record = RoiRecord.create("rectangle", {"bounds": [0.0, 0.0, 1.0, 1.0]}, name=name)
    mask = np.zeros(int(ds.prop.num_loc), dtype=bool)
    mask[rows] = True
    store_roi_mask(ds, record, mask,
                   context={"source_view": source_view, "dataset_idx": 0})
    # RoiStore.add replaces the selection with the new ROI, so keep what was
    # selected and re-select the union — the Manager state the highlight reads.
    previously_selected = list(state.rois.selected_ids)
    state.rois.add(record)
    state.rois.select(previously_selected + [record.id])
    return record


# ---------------------------------------------------------------------------
# The shared rule
# ---------------------------------------------------------------------------
def test_the_highlight_does_not_care_which_view_drew_the_roi():
    state = _state()
    ds = state.datasets[0]
    _add_mask_roi(state, ds, [0, 1, 2], source_view="attribute", name="from-attribute")
    _add_mask_roi(state, ds, [9, 10], source_view="render", name="from-render")

    pairs = highlight_masks(state, ds)

    assert len(pairs) == 2
    combined = union_mask(pairs, int(ds.prop.num_loc))
    assert np.flatnonzero(combined).tolist() == [0, 1, 2, 9, 10]


def test_apply_filter_is_the_callers_choice():
    state = _state()
    ds = state.datasets[0]
    mask = np.ones(int(ds.prop.num_loc), dtype=bool)
    mask[0] = False
    ds.filter_mask = mask
    _add_mask_roi(state, ds, [0, 1, 2], source_view="render")

    filtered = highlight_masks(state, ds, apply_filter=True)[0][1]
    unfiltered = highlight_masks(state, ds, apply_filter=False)[0][1]

    assert np.flatnonzero(filtered).tolist() == [1, 2]
    assert np.flatnonzero(unfiltered).tolist() == [0, 1, 2]


def test_decimate_caps_painted_points_without_touching_the_mask():
    mask = np.zeros(12, dtype=bool)
    mask[:10] = True
    indices = decimate(mask, 12, 3)
    assert indices.size <= 3
    assert set(indices).issubset(set(np.flatnonzero(mask)))


# ---------------------------------------------------------------------------
# Attribute Plot
# ---------------------------------------------------------------------------
def test_attribute_plot_paints_the_rows_of_a_roi_drawn_in_render(_app):
    state = _state()
    ds = state.datasets[0]
    window = AttributeWindow(state, dataset_idx=0)
    window.resize(420, 320)
    window.show()
    _app.processEvents()
    try:
        _add_mask_roi(state, ds, [0, 1, 2, 3], source_view="render")
        window._redraw_roi_highlight()
        x, _y = window._roi_highlight_item.getData()
        assert x is not None and len(x) == 4
    finally:
        window.close()
        _app.processEvents()


def test_attribute_plot_clears_the_highlight_when_rows_stop_aligning(_app, monkeypatch):
    state = _state()
    ds = state.datasets[0]
    window = AttributeWindow(state, dataset_idx=0)
    window.resize(420, 320)
    window.show()
    _app.processEvents()
    try:
        record = _add_mask_roi(state, ds, [0, 1, 2, 3], source_view="render")
        window._redraw_roi_highlight()
        assert len(window._roi_highlight_item.getData()[0]) == 4

        # Browsing another iteration re-numbers the rows, so the mask no longer
        # describes what is plotted: nothing may be painted, and no mask may be
        # produced either.
        monkeypatch.setattr(window, "_selection", lambda: ("all", "flatten"))
        assert window._roi_rows_aligned(ds) is False
        assert window.compute_roi_selection(record) is None
        window._redraw_roi_highlight()
        x, _y = window._roi_highlight_item.getData()
        assert x is None or len(x) == 0
    finally:
        window.close()
        _app.processEvents()


def test_attribute_plot_highlight_follows_the_filtered_only_box(_app):
    state = _state()
    ds = state.datasets[0]
    mask = np.ones(int(ds.prop.num_loc), dtype=bool)
    mask[0] = False
    ds.filter_mask = mask
    window = AttributeWindow(state, dataset_idx=0)
    window.resize(420, 320)
    window.show()
    _app.processEvents()
    try:
        _add_mask_roi(state, ds, [0, 1, 2], source_view="render")
        window._filter_chk.setChecked(True)
        window._redraw_roi_highlight()
        assert len(window._roi_highlight_item.getData()[0]) == 2
        # Unfiltered rows are on screen, so their highlight must be too.
        window._filter_chk.setChecked(False)
        window._redraw_roi_highlight()
        assert len(window._roi_highlight_item.getData()[0]) == 3
    finally:
        window.close()
        _app.processEvents()


def test_attribute_plot_highlight_is_ignored_by_the_auto_range(_app):
    """It must never feed the view bounds (the A button / data-bounds trap)."""
    state = _state()
    window = AttributeWindow(state, dataset_idx=0)
    window.resize(420, 320)
    window.show()
    _app.processEvents()
    try:
        view_box = window._plot.getPlotItem().getViewBox()
        # addedItems is exactly the set of bounds-contributing items (auto-range
        # iterates it), and addItem(ignoreBounds=True) deliberately skips it — so
        # being absent from it is what "ignored by the auto-range" means.
        assert window._roi_highlight_item not in view_box.addedItems

        # A highlight point far outside the data must not drag the view to it.
        window._roi_highlight_item.setData(x=[1e6], y=[1e6])
        view_box.autoRange()
        _app.processEvents()
        (x0, x1), (y0, y1) = view_box.viewRange()
        assert x1 < 1e5 and y1 < 1e5
    finally:
        window.close()
        _app.processEvents()


@pytest.mark.parametrize("roi_type, geometry", [
    ("rectangle", {"bounds": [500.0, 0.05, 3_000.0, 0.23]}),
    ("oval", {"bounds": [500.0, 0.05, 3_200.0, 0.25]}),
    ("polygon", {"points": [[500.0, 0.05], [3_500.0, 0.05],
                            [3_500.0, 0.28], [500.0, 0.28]]}),
    ("freehand", {"points": [[500.0, 0.05], [3_500.0, 0.05],
                             [3_500.0, 0.28], [500.0, 0.28]]}),
])
def test_every_region_shape_selects_rows_on_attribute_axes(_app, roi_type, geometry):
    """An outline encloses rows on attribute axes exactly as it does on spatial ones."""
    state = _state()
    window = AttributeWindow(state, dataset_idx=0)
    window.resize(420, 320)
    window.show()
    _app.processEvents()
    try:
        # Pin the axes so the shape's coordinates are known: efo 1000..12000,
        # cfr 0.1..0.9 over 12 rows, so each shape above covers rows 0-2.
        for dimension in ("X", "Y", "Z"):
            window._dimension_attrs[dimension] = "efo" if dimension == "X" else "cfr"
        record = RoiRecord.create(roi_type, dict(geometry), name=roi_type)

        result = window.compute_roi_selection(record)

        assert result is not None, f"{roi_type} produced no selection"
        ds, mask, context = result
        assert np.flatnonzero(mask).tolist() == [0, 1, 2]
        assert context["source_view"] == "attribute"

        # …and the highlight paints exactly those rows.
        store_roi_mask(ds, record, mask, context=context)
        previously_selected = list(state.rois.selected_ids)
        state.rois.add(record)
        state.rois.select(previously_selected + [record.id])
        window._redraw_roi_highlight()
        assert len(window._roi_highlight_item.getData()[0]) == 3
    finally:
        window.close()
        _app.processEvents()


def test_a_line_roi_encloses_nothing_on_attribute_axes_either(_app):
    state = _state()
    window = AttributeWindow(state, dataset_idx=0)
    window.show()
    _app.processEvents()
    try:
        window._dimension_attrs["X"] = "efo"
        window._dimension_attrs["Y"] = "cfr"
        line = RoiRecord.create(
            "line", {"points": [[500.0, 0.05], [3_500.0, 0.28]]}, name="cut")
        assert window.compute_roi_selection(line) is None
    finally:
        window.close()
        _app.processEvents()


# ---------------------------------------------------------------------------
# Histogram
# ---------------------------------------------------------------------------
def test_histogram_draws_a_second_histogram_of_true_in_roi_counts(_app):
    # Six localizations share one value, so a bin holds more rows than are
    # selected: the highlight bar has to show 2, not the whole bin.
    state = _state(efo=np.r_[np.full(6, 1_000.0), np.full(6, 2_000.0)])
    ds = state.datasets[0]
    window = HistogramWindow(state, dataset_idx=0)
    window.resize(520, 360)
    window.show()
    _app.processEvents()
    try:
        window._attr_combo.setCurrentText("efo")
        window._draw()
        _add_mask_roi(state, ds, [0, 1], source_view="render")
        window._redraw_roi_highlight()

        main = np.asarray(window._hist_item.opts["height"], dtype=float)
        marked = np.asarray(window._roi_highlight_item.opts["height"], dtype=float)

        assert marked.size == main.size                  # same bins
        assert marked.sum() == 2                         # the two selected rows
        assert (marked <= main).all()
        # The bin carrying them holds more rows than are highlighted, i.e. the
        # whole bin is not lit up.
        assert main[int(np.argmax(marked))] > marked.max()
        assert window._roi_highlight_item.opts["width"] < window._hist_item.opts["width"]
    finally:
        window.close()
        _app.processEvents()


def test_histogram_clears_the_highlight_where_a_mask_cannot_be_mapped(_app, monkeypatch):
    state = _state()
    ds = state.datasets[0]
    window = HistogramWindow(state, dataset_idx=0)
    window.resize(520, 360)
    window.show()
    _app.processEvents()
    try:
        window._attr_combo.setCurrentText("efo")
        window._draw()
        _add_mask_roi(state, ds, [0, 1], source_view="render")
        window._redraw_roi_highlight()
        assert np.asarray(window._roi_highlight_item.opts["height"], dtype=float).sum() == 2

        monkeypatch.setattr(window, "_is_raw_mode", lambda: True)
        window._redraw_roi_highlight()
        assert len(np.asarray(window._roi_highlight_item.opts["height"], dtype=float)) == 0
    finally:
        window.close()
        _app.processEvents()


# ---------------------------------------------------------------------------
# Render: the depth slider is a view setting, not part of the mask
# ---------------------------------------------------------------------------
def test_render_roi_mask_spans_the_whole_depth_axis(_app):
    from minflux_viewer.ui.render_window import RenderWindow

    state = _state()
    ds = state.datasets[0]
    window = RenderWindow(state, 0)
    window.resize(420, 420)
    window.show()
    _app.processEvents()
    try:
        if not window._has_depth:
            pytest.skip("dataset did not load as 3-D, so there is no depth slider")
        window._all_depth_check.setChecked(False)
        window._depth_range = (0.0, 10.0)          # a thin slice of z 0..100
        record = RoiRecord.create(
            "rectangle", {"bounds": [-50.0, -50.0, 500.0, 500.0]}, name="all-xy")

        result = window.compute_roi_selection(record)

        assert result is not None
        _ds, mask, context = result
        # Every localization is inside the rectangle in XY, so every one is in
        # the ROI — the slider narrowed the view, not the selection.
        assert int(np.count_nonzero(mask)) == int(ds.prop.num_loc)
        assert context["depth_range"] == [0.0, 10.0]   # kept as provenance
    finally:
        window.close()
        _app.processEvents()
