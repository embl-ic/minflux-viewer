"""The time axis a view gates on: rounding, trace onsets and interval statistics."""

from __future__ import annotations

import numpy as np
import pytest

from minflux_viewer.core.tracking_time import (
    analyze_time_intervals,
    build_time_axis,
    derived_step,
    relative_trace_index,
    rounded_relative_time_ticks,
    time_axis_values,
    timestamp_precision_seconds,
    tracking_precision_from_prefs,
)


def test_round_then_zero_each_trace_onset() -> None:
    # Sub-microsecond readout detail is removed before each trace's onset is
    # subtracted.  Rows are deliberately interleaved by trace.
    tim = np.array([10.0000004, 20.0000002, 10.0001006, 20.0000997])
    tid = np.array([4, 8, 4, 8])

    ticks = rounded_relative_time_ticks(tim, tid, 1.0e-6)

    assert ticks.tolist() == [0, 0, 101, 100]


def test_relative_index_is_zero_based_per_trace_in_source_order() -> None:
    tid = np.array([20, 10, 20, 20, 10])
    assert relative_trace_index(tid).tolist() == [0, 0, 1, 2, 1]


def test_global_mode_reports_the_regular_gap_it_found() -> None:
    tid = np.array([1, 1, 1, 1, 2, 2, 2])
    ticks = np.array([0, 100, 200, 400, 0, 100, 200])

    result = analyze_time_intervals(ticks, tid, 1.0e-6)

    assert result.baseline_interval_ticks == 100
    assert result.baseline_interval_s == pytest.approx(100.0e-6)
    assert result.mode_count == 4
    assert result.positive_interval_count == 5
    assert result.missing_frame_count == 1
    assert result.irregular_interval_count == 0
    assert result.has_gaps
    assert not result.low_confidence


def test_axis_modes_are_different_axes_not_the_same_one_relabelled() -> None:
    # Two traces recorded half a second apart: absolute time keeps that offset,
    # trace-relative time removes it, and the index axis reads neither.
    tim = np.array([0.0, 1.0e-4, 0.5, 0.5001])
    tid = np.array([1, 1, 2, 2])

    absolute, unit, diag = time_axis_values(tim, tid, mode="absolute")
    assert unit == "s"
    assert absolute.tolist() == pytest.approx(tim.tolist())
    assert diag is not None and diag.baseline_interval_ticks == 100

    trace, _unit, _diag = time_axis_values(tim, tid, mode="trace")
    assert trace.tolist() == pytest.approx([0.0, 1.0e-4, 0.0, 1.0e-4])

    index, unit, diag = time_axis_values(tim, tid, mode="index")
    assert unit == "index"
    assert index.tolist() == [0, 1, 0, 1]
    assert diag is None


def test_index_mode_ignores_time_entirely() -> None:
    tid = np.array([3, 9, 3, 9])
    values, _unit, _diag = time_axis_values(None, tid, mode="index")
    assert values.tolist() == [0, 0, 1, 1]


def test_a_missing_attribute_raises_rather_than_silently_switching_axis() -> None:
    tid = np.array([1, 1, 2])
    with pytest.raises(ValueError, match="tim"):
        time_axis_values(None, tid, mode="absolute")
    with pytest.raises(ValueError, match="tim"):
        time_axis_values(None, tid, mode="trace")
    with pytest.raises(ValueError, match="tid"):
        time_axis_values(np.array([0.0, 1.0, 2.0]), None, mode="index")


def test_axis_bounds_follow_the_kept_rows() -> None:
    tim = np.array([0.0, 1.0, 2.0, 3.0])
    tid = np.ones(4, dtype=int)
    keep = np.array([False, True, True, False])

    axis = build_time_axis(tim, tid, mode="absolute", keep=keep)
    assert (axis.lo, axis.hi) == (1.0, 2.0)
    # ``values`` stays row-aligned so the gate can still be applied to any rows.
    assert axis.values.size == 4


def test_step_is_the_measured_interval_unless_it_is_unusably_fine() -> None:
    # A usable interval is kept verbatim...
    assert derived_step(1.0, 0.01) == pytest.approx(0.01)
    # ...and one a millionth of the span is raised to a workable floor, so the
    # wheel and playback stay usable on a long acquisition.
    assert derived_step(1.0, 1.0e-6) == pytest.approx(1.0 / 200.0)
    assert derived_step(1.0, None) == pytest.approx(1.0 / 200.0)


def test_precision_units_and_default_are_one_microsecond() -> None:
    assert timestamp_precision_seconds(1.0, "us") == pytest.approx(1.0e-6)
    assert timestamp_precision_seconds(0.1, "ms") == pytest.approx(100.0e-6)
    assert timestamp_precision_seconds(100.0, "ns") == pytest.approx(0.1e-6)
    assert tracking_precision_from_prefs({}) == pytest.approx(1.0e-6)
