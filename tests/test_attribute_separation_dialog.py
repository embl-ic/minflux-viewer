"""
End-to-end smoke test for the channel separation dialog ("Separate Channel by
DCR" / "Convert to Multi-Channel Overlay (by attribute)").

Uses pytest-qt's ``qtbot`` fixture — manually creating the QApplication in the
test body crashes pyqtgraph widget creation on this Windows setup, whereas the
``qtbot``-managed app is stable. The fit *math* is covered by
``test_distribution_fit`` / ``test_attribute_channels``; peak detection by
``test_peak_channels``.
"""

from __future__ import annotations

import numpy as np
import pytest

pytest.importorskip("PyQt6")
pytest.importorskip("pyqtgraph")
pytest.importorskip("pytestqt")

from minflux_viewer.core import loader


def _make_bimodal_dcr(n_traces=200, locs_per=6, seed=0):
    """Flat m2410 mfx array: half the traces sit at DCR≈0.3, half at ≈0.7."""
    rng = np.random.default_rng(seed)
    tids = np.repeat(np.arange(n_traces), locs_per)
    n = tids.size
    dt = np.dtype([
        ("vld", "?"), ("itr", "<i4"), ("tid", "<i4"),
        ("loc", "<f8", (3,)), ("dcr", "<f8"), ("eco", "<f8"), ("efo", "<f8"),
    ])
    mfx = np.zeros(n, dtype=dt)
    mfx["tid"] = tids
    mfx["itr"] = 0
    mfx["vld"] = True
    mfx["loc"][:, 0] = rng.uniform(0, 1e-6, n)
    mfx["loc"][:, 1] = rng.uniform(0, 1e-6, n)
    trace_dcr = np.where(np.arange(n_traces) % 2 == 0, 0.30, 0.70)
    mfx["dcr"] = np.clip(np.repeat(trace_dcr, locs_per) + rng.normal(0, 0.03, n), 0, 1)
    mfx["eco"] = rng.poisson(50, n).astype(float) + 1.0
    mfx["efo"] = rng.uniform(1e4, 2e5, n)
    return mfx


def _make_unimodal_dcr(n_traces=120, locs_per=5, seed=3):
    mfx = _make_bimodal_dcr(n_traces, locs_per, seed)
    rng = np.random.default_rng(seed)
    mfx["dcr"] = np.clip(0.5 + rng.normal(0, 0.03, mfx.size), 0, 1)
    return mfx


def _make_multi_iter_dcr(n_loc=300, n_itr=3, seed=0):
    """Flat m2410 mfx with several iterations whose DCR distribution shifts per
    iteration — so 'all [stacked]' differs visibly from 'all [flatten]'."""
    rng = np.random.default_rng(seed)
    n = n_loc * n_itr
    itr = np.tile(np.arange(n_itr), n_loc)
    tid = np.repeat(np.arange(n_loc), n_itr)
    dt = np.dtype([
        ("vld", "?"), ("itr", "<i4"), ("tid", "<i4"),
        ("loc", "<f8", (3,)), ("dcr", "<f8"), ("eco", "<f8"),
    ])
    mfx = np.zeros(n, dtype=dt)
    mfx["itr"] = itr
    mfx["tid"] = tid
    mfx["vld"] = True
    mfx["loc"][:, 0] = np.repeat(rng.uniform(0, 1e-6, n_loc), n_itr)
    base = np.repeat(np.where(np.arange(n_loc) % 2 == 0, 0.3, 0.7), n_itr)
    mfx["dcr"] = np.clip(base + 0.05 * itr + rng.normal(0, 0.02, n), 0, 1)
    mfx["eco"] = rng.poisson(50, n).astype(float) + 1.0
    return mfx


class _FakeOwner:
    def __init__(self):
        self.applied = None

    def apply_channel_separation(self, idx, labels, channels, *, attribute="",
                                 method_label="", masks=None, include_unassigned=True):
        self.applied = {
            "idx": idx,
            "labels": labels,
            "channels": list(channels),
            "attribute": attribute,
            "masks": None if masks is None else [np.asarray(m, dtype=bool).copy() for m in masks],
            "include_unassigned": include_unassigned,
        }
        return True


def _dialog(qtbot, mfx=None, **kwargs):
    from minflux_viewer.core.app_state import AppState
    from minflux_viewer.ui.attribute_separation_dialog import AttributeSeparationDialog

    ds = loader.load_from_mfx_array(_make_bimodal_dcr() if mfx is None else mfx, "dcr_test")
    state = AppState()
    state.add_dataset(ds)
    owner = _FakeOwner()
    dlg = AttributeSeparationDialog(state, 0, attribute="dcr", owner=owner, **kwargs)
    qtbot.addWidget(dlg)                        # qtbot owns lifetime / cleanup
    return dlg, ds, owner


# ---------------------------------------------------------------- base flow
def test_dialog_seeds_two_channels_and_assigns(qtbot):
    dlg, ds, _owner = _dialog(qtbot)
    assert len(dlg._rows) == 2
    masks, _overlap, unassigned = dlg._resolve_assignment()
    assert masks is not None and len(masks) == 2
    assert all(m.size == ds.prop.num_loc for m in masks)
    assert all(m.any() for m in masks)          # both DCR populations are claimed
    assert unassigned.size == ds.prop.num_loc


def test_apply_hands_over_one_mask_per_channel(qtbot):
    dlg, ds, owner = _dialog(qtbot)
    dlg._apply()
    assert owner.applied is not None
    assert owner.applied["idx"] == 0
    assert owner.applied["attribute"] == "dcr"
    assert owner.applied["labels"] is None      # masks are the general form
    masks = owner.applied["masks"]
    assert masks is not None and len(masks) == len(owner.applied["channels"]) == 2
    assert all(m.size == ds.prop.num_loc for m in masks)
    assert owner.applied["include_unassigned"] is True


def test_unassigned_policy_reaches_the_owner(qtbot):
    from minflux_viewer.ui.attribute_separation_dialog import UNASSIGNED_DISCARD

    dlg, _ds, owner = _dialog(qtbot)
    dlg._unassigned_combo.setCurrentText(UNASSIGNED_DISCARD)
    dlg._apply()
    assert owner.applied["include_unassigned"] is False


def test_channel_count_spinner_drives_place_evenly(qtbot):
    dlg, _ds, _owner = _dialog(qtbot)
    dlg._nch_spin.setValue(3)
    dlg._place_evenly()
    assert len(dlg._rows) == 3
    dlg._nch_spin.setValue(1)
    dlg._place_evenly()
    assert len(dlg._rows) == 1


# ------------------------------------------------------------- detect peak
def test_detect_peak_splits_the_two_dcr_populations(qtbot):
    dlg, _ds, _owner = _dialog(qtbot)
    dlg._nch_spin.setValue(2)
    dlg._detect_peaks()
    assert len(dlg._rows) == 2
    cut = dlg._rows[0]["end"].value()
    assert 0.35 < cut < 0.65                    # the valley between 0.30 and 0.70


def test_detect_peak_places_only_the_peaks_that_exist(qtbot):
    """One hump cannot be cut into three channels, and it says so."""
    dlg, _ds, _owner = _dialog(qtbot, mfx=_make_unimodal_dcr())
    dlg._nch_spin.setValue(3)
    dlg._detect_peaks()
    assert len(dlg._rows) == 1
    assert any("peak" in str(entry.get("message", "")).lower()
               for entry in dlg._state.log_history)


# ------------------------------------------------------------ overlap policy
def _overlap_two_channels(dlg):
    """Make the two window channels overlap over the middle of the range."""
    dlg._nch_spin.setValue(2)
    dlg._place_evenly()
    first, second = dlg._rows[0], dlg._rows[1]
    lo = float(dlg._values.min())
    hi = float(dlg._values.max())
    mid = 0.5 * (lo + hi)
    span = hi - lo
    first["start"].setValue(lo)
    first["end"].setValue(mid + 0.2 * span)
    second["start"].setValue(mid - 0.2 * span)
    second["end"].setValue(hi)


def test_overlap_keep_in_both_channels(qtbot):
    from minflux_viewer.ui.attribute_separation_dialog import OVERLAP_KEEP_BOTH

    dlg, _ds, _owner = _dialog(qtbot)
    _overlap_two_channels(dlg)
    dlg._overlap_combo.setCurrentText(OVERLAP_KEEP_BOTH)
    masks, overlap, _unassigned = dlg._resolve_assignment()
    assert overlap > 0
    both = masks[0] & masks[1]
    assert int(both.sum()) == overlap          # contested rows are in both channels


def test_overlap_discard_leaves_contested_rows_unassigned(qtbot):
    from minflux_viewer.ui.attribute_separation_dialog import OVERLAP_DISCARD

    dlg, _ds, _owner = _dialog(qtbot)
    _overlap_two_channels(dlg)
    dlg._overlap_combo.setCurrentText(OVERLAP_DISCARD)
    masks, overlap, unassigned = dlg._resolve_assignment()
    assert overlap > 0
    assert not (masks[0] & masks[1]).any()
    assert int(unassigned.sum()) >= overlap


def test_overlap_by_weight_assigns_each_contested_row_once(qtbot):
    from minflux_viewer.ui.attribute_separation_dialog import OVERLAP_BY_WEIGHT

    dlg, _ds, _owner = _dialog(qtbot)
    _overlap_two_channels(dlg)
    dlg._overlap_combo.setCurrentText(OVERLAP_BY_WEIGHT)
    masks, overlap, _unassigned = dlg._resolve_assignment()
    assert overlap > 0
    stack = np.vstack(masks)
    assert stack.sum(axis=0).max() <= 1        # exclusive again
    assert int(stack.any(axis=0).sum()) > 0


# ------------------------------------------------------- ROI / filter sources
def test_empty_sources_read_as_a_dash(qtbot):
    dlg, _ds, _owner = _dialog(qtbot)
    for combo in (dlg._roi_combo, dlg._filter_combo):
        assert combo.count() == 1
        assert combo.itemText(0) == "-"
        assert not combo.isEnabled()


def test_channel_from_roi(qtbot):
    from minflux_viewer.core.roi import RoiRecord

    dlg, ds, _owner = _dialog(qtbot)
    # loc spans 0..1000 nm; this rectangle covers the lower half in X.
    record = RoiRecord.create("rectangle", {"bounds": [-10.0, -10.0, 510.0, 1100.0]},
                              name="left half", context={"dataset_idx": 0})
    dlg._state.rois.add(record)
    dlg._populate_sources()
    assert dlg._roi_combo.isEnabled() and dlg._roi_combo.count() == 2

    before = len(dlg._rows)
    dlg._on_roi_source_picked(1)
    assert len(dlg._rows) == before + 1
    row = dlg._rows[-1]
    assert row["kind"] == "roi"
    assert row["mask"].size == ds.prop.num_loc and row["mask"].any()
    assert row["region"] is None                         # a selection has no window
    assert dlg._table.item(len(dlg._rows) - 1, 1).text() == "—"
    assert dlg._roi_combo.currentIndex() == 0            # back to the placeholder


def test_channel_from_filter_row(qtbot):
    dlg, ds, _owner = _dialog(qtbot)
    ds.state["filter_specs"] = [
        {"attribute": "dcr", "mode": "per loc", "lo": 0.0, "hi": 0.5,
         "lo_inc": True, "hi_inc": True},
    ]
    dlg._populate_sources()
    assert dlg._filter_combo.isEnabled() and dlg._filter_combo.count() == 2
    assert "dcr" in dlg._filter_combo.itemText(1)

    dlg._on_filter_source_picked(1)
    row = dlg._rows[-1]
    assert row["kind"] == "filter"
    expected = np.asarray(ds.attr["dcr"], dtype=float) <= 0.5
    assert row["mask"].tolist() == expected.tolist()


def test_mask_channels_survive_a_refit(qtbot):
    """Re-placing the windows must not drop a ROI/filter channel."""
    dlg, ds, _owner = _dialog(qtbot)
    ds.state["filter_specs"] = [{"attribute": "dcr", "mode": "per loc",
                                 "lo": 0.0, "hi": 0.5}]
    dlg._populate_sources()
    dlg._on_filter_source_picked(1)
    assert [row["kind"] for row in dlg._rows] == ["window", "window", "filter"]
    dlg._nch_spin.setValue(3)
    dlg._place_evenly()
    assert [row["kind"] for row in dlg._rows] == ["window"] * 3 + ["filter"]


# ------------------------------------------------------------------ details
def test_bin_step_adapts_to_the_attribute_being_plotted(qtbot):
    dlg, _ds, _owner = _dialog(qtbot, pick_attribute=True)
    dcr_step = dlg._bin_spin.singleStep()
    dlg._attr_combo.setCurrentText("efo")
    efo_step = dlg._bin_spin.singleStep()
    # dcr spans ~0.5, efo ~2e5: the step has to follow the data, not stay at 0.01.
    assert dcr_step < 0.05
    assert efo_step > 100 * dcr_step


def test_channel_colours_are_transparent_solids_in_rgbmcy_order(qtbot):
    from minflux_viewer.ui.attribute_separation_dialog import (
        CHANNEL_ALPHA, encoded_channel_lut)

    dlg, _ds, _owner = _dialog(qtbot)
    assert dlg._lut_cycle()[:6] == ["Red", "Green", "Blue", "Magenta", "Cyan", "Yellow"]
    encoded = encoded_channel_lut("Red")
    assert encoded.startswith("solid:custom:#")
    assert encoded.endswith(f"{CHANNEL_ALPHA:02x}")      # transparency is carried
    # …and that is what Apply hands over.
    dlg._apply()
    assert all(ch.lut.startswith("solid:custom:#") for ch in _owner_channels(dlg))


def _owner_channels(dlg):
    return dlg._owner.applied["channels"]


def test_region_right_click_deletes_the_channel(qtbot, monkeypatch):
    from PyQt6.QtWidgets import QMenu

    dlg, _ds, _owner = _dialog(qtbot)
    captured: list[QMenu] = []
    monkeypatch.setattr(QMenu, "exec", lambda menu, *_a: captured.append(menu))
    dlg._show_region_menu(dlg._rows[0])
    assert captured, "right-click produced no menu"
    texts = [action.text() for action in captured[0].actions()]
    assert any("Delete channel" in text for text in texts)

    before = len(dlg._rows)
    next(action for action in captured[0].actions()
         if "Delete channel" in action.text()).trigger()
    assert len(dlg._rows) == before - 1


def test_preview_draws_the_separation(qtbot):
    import pyqtgraph as pg

    dlg, _ds, _owner = _dialog(qtbot)
    dlg._refresh_preview()
    items = [it for it in dlg._preview.getPlotItem().items
             if isinstance(it, pg.ScatterPlotItem)]
    assert items, "the preview drew nothing"
    assert sum(len(it.getData()[0]) for it in items) > 0


def test_dialog_attribute_picker_switches(qtbot):
    """The generic 'by attribute' mode shows an attribute picker and rebuilds
    the distribution + channels when the attribute changes."""
    dlg, ds, _owner = _dialog(qtbot, pick_attribute=True)
    assert dlg._attr_combo is not None
    items = [dlg._attr_combo.itemText(i) for i in range(dlg._attr_combo.count())]
    assert "dcr" in items and "efo" in items
    dlg._attr_combo.setCurrentText("efo")
    assert dlg._attribute == "efo"
    assert len(dlg._rows) == 2
    masks, _overlap, _unassigned = dlg._resolve_assignment()
    assert masks is not None and masks[0].size == ds.prop.num_loc


def test_stacked_iteration_differs_from_flatten(qtbot):
    """'all [stacked]' draws one colored series per iteration + a legend, while
    'all [flatten]' is a single pooled histogram — the fit basis (pooled values)
    is identical, only the display differs."""
    from minflux_viewer.core.iteration import FLATTEN_LABEL, STACKED_LABEL

    dlg, _ds, _owner = _dialog(qtbot, mfx=_make_multi_iter_dcr(n_itr=3))

    def snapshot(label):
        dlg._iter_combo.setCurrentText(label)          # → _on_basis_changed → redraw
        pi = dlg._plot.getPlotItem()
        curves = [it for it in pi.items if it.__class__.__name__ == "PlotDataItem"]
        bars = [it for it in pi.items if it.__class__.__name__ == "BarGraphItem"]
        return dlg._values.size, len(curves), len(bars), (pi.legend is not None)

    n_flat, _curves_flat, bars_flat, legend_flat = snapshot(FLATTEN_LABEL)
    n_stack, curves_stack, bars_stack, legend_stack = snapshot(STACKED_LABEL)

    assert n_flat == n_stack                            # same pooled fit basis
    assert bars_flat == 1 and not legend_flat           # flatten: single pooled bar
    assert bars_stack == 0 and legend_stack             # stacked: legend, no pooled bar
    assert curves_stack == 3                            # one series per iteration


def test_fit_updates_channels(qtbot):
    """The fit path (sklearn under a live Qt loop) is stable under qtbot."""
    dlg, _ds, _owner = _dialog(qtbot)
    dlg._fit_combo.setCurrentIndex(dlg._fit_combo.findData("gaussian"))
    dlg._nch_spin.setValue(2)
    dlg._run_fit()
    assert len(dlg._rows) == 2
    masks, _overlap, _unassigned = dlg._resolve_assignment()
    assert all(m.any() for m in masks)
