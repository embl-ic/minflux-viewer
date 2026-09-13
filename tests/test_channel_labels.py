"""Channel labels from a selection — the input ``apply_channel_separation`` takes.

These are the groundwork for *channel from ROI* and *channel from filter*: the
separation path only ever sees an exclusive label array, so the label builders
are where the semantics (precedence, whole traces, skipped selections) live.
"""

from __future__ import annotations

import numpy as np

from minflux_viewer.core.channel_labels import (
    expand_to_traces,
    labels_from_filter,
    labels_from_filter_specs,
    labels_from_masks,
    labels_from_rois,
    mask_from_filter_specs,
    roi_mask_for_record,
)
from minflux_viewer.core.dataset import build_localization_dataset
from minflux_viewer.core.filter_io import filter_specs_from_rows, rows_from_filter_specs
from minflux_viewer.core.roi import RoiRecord


def _dataset(n: int = 12):
    """``n`` localizations on a line, 10 nm apart, in three equal traces."""
    return build_localization_dataset(
        name="labels",
        x_nm=np.arange(n, dtype=float) * 10.0,
        y_nm=np.full(n, 5.0),
        z_nm=np.zeros(n),
        tid=np.repeat([1, 2, 3], n // 3),
        tim=np.linspace(0.0, 1.0, n),
        attrs={"efo": np.linspace(1_000.0, 12_000.0, n)},
    )


def _rect(x: float, y: float, w: float, h: float, name: str = "r") -> RoiRecord:
    return RoiRecord.create("rectangle", {"bounds": [x, y, w, h]}, name=name)


def test_labels_are_exclusive_and_the_earlier_channel_keeps_contested_rows():
    n = 12
    first = np.zeros(n, dtype=bool)
    first[0:6] = True
    second = np.zeros(n, dtype=bool)
    second[3:9] = True

    plan = labels_from_masks([first, second], n, names=["a", "b"])

    assert plan.labels.tolist() == [0, 0, 0, 0, 0, 0, 1, 1, 1, -1, -1, -1]
    assert plan.counts == [6, 3]
    assert plan.overlaps == 3          # rows 3-5 were claimed twice, reported
    assert plan.unassigned == 3
    assert plan.names == ["a", "b"]


def test_trace_rules_decide_whether_a_straddling_trace_is_kept_whole():
    ds = _dataset()                     # 3 traces x 4 localizations
    trace_idx = ds.prop.trace_idx
    one_of_four = np.zeros(12, dtype=bool)
    one_of_four[1] = True
    three_of_four = np.zeros(12, dtype=bool)
    three_of_four[0:3] = True

    assert expand_to_traces(one_of_four, trace_idx, "per loc").tolist() == one_of_four.tolist()
    assert expand_to_traces(one_of_four, trace_idx, "any")[0:4].all()
    assert not expand_to_traces(one_of_four, trace_idx, "majority").any()
    assert not expand_to_traces(one_of_four, trace_idx, "all").any()
    assert expand_to_traces(three_of_four, trace_idx, "majority")[0:4].all()
    assert not expand_to_traces(three_of_four, trace_idx, "all").any()
    # Only the straddled trace grows; the others are untouched.
    assert not expand_to_traces(one_of_four, trace_idx, "any")[4:].any()


def test_labels_from_filter_splits_in_and_out_of_the_live_filter():
    ds = _dataset()
    mask = np.zeros(12, dtype=bool)
    mask[:5] = True
    ds.filter_mask = mask

    plan = labels_from_filter(ds)

    assert plan.counts == [5, 7]
    assert plan.unassigned == 0         # every localization lands in a channel
    assert plan.labels[:5].tolist() == [0] * 5
    assert plan.labels[5:].tolist() == [1] * 7


def test_filter_specs_are_evaluated_without_touching_the_live_filter():
    ds = _dataset()
    before = ds.filter_mask.copy()
    specs = [{"attribute": "efo", "mode": "per loc", "lo": 1_000.0, "hi": 5_000.0}]

    mask, skipped = mask_from_filter_specs(ds, specs)

    expected = np.asarray(ds.attr["efo"], dtype=float) <= 5_000.0
    assert mask.tolist() == expected.tolist()
    assert skipped == []
    assert ds.filter_mask.tolist() == before.tolist()


def test_an_unusable_spec_is_skipped_with_a_reason_not_silently_ignored():
    ds = _dataset()
    mask, skipped = mask_from_filter_specs(
        ds, [{"attribute": "not_an_attribute", "mode": "per loc", "lo": 0.0, "hi": 1.0}])

    assert mask.all()                   # an unevaluable spec filters nothing
    assert len(skipped) == 1 and "not_an_attribute" in skipped[0]


def test_one_channel_per_filter_preset():
    ds = _dataset()
    low = [{"attribute": "efo", "mode": "per loc", "lo": 0.0, "hi": 4_000.0}]
    high = [{"attribute": "efo", "mode": "per loc", "lo": 8_000.0, "hi": 99_000.0}]

    plan = labels_from_filter_specs(ds, [low, high], names=["low", "high"])

    assert plan.n_channels == 2
    assert plan.names == ["low", "high"]
    assert plan.counts[0] > 0 and plan.counts[1] > 0
    assert plan.overlaps == 0
    assert plan.unassigned == 12 - sum(plan.counts)


def test_roi_masks_are_recomputed_when_no_stored_mask_exists():
    ds = _dataset()
    # A ROI straight from a loaded set has selection_dirty set and no mask.
    record = _rect(-5.0, 0.0, 40.0, 10.0, name="left")
    assert record.selection_dirty

    mask = roi_mask_for_record(ds, record)

    assert mask is not None
    assert mask.tolist() == [True] * 4 + [False] * 8


def test_one_channel_per_roi_and_a_line_roi_is_skipped_with_a_reason():
    ds = _dataset()
    line = RoiRecord.create("line", {"points": [[0.0, 0.0], [100.0, 0.0]]}, name="cut")

    plan = labels_from_rois(
        ds, [_rect(-5.0, 0.0, 40.0, 10.0, name="left"),
             _rect(75.0, 0.0, 50.0, 10.0, name="right"),
             line])

    assert plan.n_channels == 2                      # the line carries no area
    assert plan.names == ["left", "right"]
    assert plan.labels[:4].tolist() == [0] * 4
    assert plan.labels[8:].tolist() == [1] * 4
    assert plan.labels[4:8].tolist() == [-1] * 4     # between the two ROIs
    assert any("cut" in reason and "no area" in reason for reason in plan.skipped)


def test_an_roi_channel_can_keep_whole_traces():
    ds = _dataset()
    # Covers localizations 3-4, i.e. one row of trace 0 and one of trace 1.
    record = _rect(25.0, 0.0, 20.0, 10.0, name="straddle")

    per_loc = labels_from_rois(ds, [record], trace_rule="per loc")
    whole = labels_from_rois(ds, [record], trace_rule="any")

    assert per_loc.counts == [2]
    assert whole.counts == [8]                       # both traces, complete


def test_preset_rows_and_filter_specs_round_trip():
    rows = [{
        "apply": True, "attribute": "cfr", "value_as": "trace mean",
        "min": 0.1, "max": 0.8, "min_inclusive": True, "max_inclusive": False,
        "iteration": "effective",
    }]

    specs = filter_specs_from_rows(rows)

    assert specs == [{
        "attribute": "cfr", "mode": "trace mean", "lo": 0.1, "hi": 0.8,
        "lo_inc": True, "hi_inc": False, "itr": "effective",
    }]
    assert rows_from_filter_specs(specs) == rows


def test_unticked_preset_rows_are_left_out_like_the_filter_dialog_does():
    rows = [
        {"apply": False, "attribute": "efo", "min": 0.0, "max": 1.0},
        {"apply": True, "attribute": "cfr", "min": 0.0, "max": 1.0},
    ]

    specs = filter_specs_from_rows(rows)
    assert [spec["attribute"] for spec in specs] == ["cfr"]
    assert len(filter_specs_from_rows(rows, only_enabled=False)) == 2


def test_a_spec_without_an_iteration_key_keeps_it_absent():
    """resolve_spec_iteration owns that default; guessing here would override it."""
    specs = filter_specs_from_rows([{"apply": True, "attribute": "efo",
                                     "min": 0.0, "max": 1.0}])
    assert "itr" not in specs[0]
