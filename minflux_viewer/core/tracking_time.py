"""The time axis shared by tracking views and future tracking analyses.

The playable UI is a dedicated Tracking View rather than a gate added to Render
or Loc Scatter.  This module remains independent of that presentation: it turns
a dataset's ``tim``/``tid`` into a row-aligned :class:`TimeAxis`, diagnoses how
regular the acquisition timing is, and never modifies the dataset.

This module is the Qt-free half: it turns a dataset's ``tim``/``tid`` into a
:class:`TimeAxis` (row-aligned values, bounds, a natural step) and diagnoses how
regular the acquisition timing is.  Three axis modes:

``absolute``
    ``tim`` as recorded, in seconds.  Traces keep their true offsets, so two
    traces seen at the same moment are gated together.
``trace``
    Each trace starts at zero: timestamps are rounded to the configured
    precision first, then that trace's earliest rounded value is subtracted.
    Comparable across traces, at the cost of absolute timing between them.
``index``
    Each row's zero-based occurrence number within its own trace.  ``idx`` is
    deliberately *not* used for this: it is the application's global, 1-based row
    index, not a within-trace counter.

Rounding happens before the onset is subtracted so sub-picosecond float detail
cannot make two traces start at different zeros.  The modal within-trace
interval is measured from the data, never assumed from the preference.

**There is deliberately no interpolation of missing frames.**  Render and
Scatter draw *rows of the dataset*; a synthetic localization is not one, so a
row gate has nowhere to put it -- and inventing points would fabricate
measurement in views whose counts and ROI selections are read as data.  What
this module does instead is *report* irregularity, through
:class:`TimeIntervalDiagnostics`.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

TIME_UNIT_SECONDS: dict[str, float] = {
    "s": 1.0,
    "sec": 1.0,
    "second": 1.0,
    "seconds": 1.0,
    "ms": 1.0e-3,
    "us": 1.0e-6,
    "µs": 1.0e-6,
    "μs": 1.0e-6,
    "ns": 1.0e-9,
}

#: Axis modes, in the order the UI offers them.
TIME_MODES: tuple[str, ...] = ("absolute", "trace", "index")

TIME_MODE_LABELS: dict[str, str] = {
    "absolute": "absolute time",
    "trace": "time, each trace from 0",
    "index": "index within trace",
}

#: Short forms for the row's own combo, which competes for width with the plot.
#: The long labels stay for the read-out and the diagnostics line, where there is
#: room to say what the axis means.
TIME_MODE_SHORT: dict[str, str] = {
    "absolute": "absolute",
    "trace": "per trace",
    "index": "index",
}

TIME_MODE_TIPS: dict[str, str] = {
    "absolute": (
        "The recorded 'tim' in seconds. Traces keep their true offsets, so two "
        "traces seen at the same moment are gated together."
    ),
    "trace": (
        "Each trace starts at zero, so traces can be compared step for step. "
        "Absolute timing between traces is discarded, which makes this unsuitable "
        "for a synchronized ensemble-event question."
    ),
    "index": (
        "Each localization's zero-based position within its own trace. No "
        "timestamp is read, so an uneven acquisition interval cannot distort it."
    ),
}

#: A derived step is never finer than the span divided by this.
_MIN_STEPS_PER_SPAN = 200


@dataclass(frozen=True)
class TimeIntervalDiagnostics:
    """How regular the within-trace sampling interval is, in rounding ticks."""

    precision_s: float
    baseline_interval_ticks: int | None
    positive_interval_count: int
    mode_count: int
    duplicate_interval_count: int
    irregular_interval_count: int
    missing_frame_count: int
    off_grid_point_count: int

    @property
    def baseline_interval_s(self) -> float | None:
        if self.baseline_interval_ticks is None:
            return None
        return float(self.baseline_interval_ticks) * self.precision_s

    @property
    def mode_fraction(self) -> float:
        if self.positive_interval_count <= 0:
            return 0.0
        return self.mode_count / self.positive_interval_count

    @property
    def low_confidence(self) -> bool:
        """The modal interval describes fewer than half the measured steps.

        Reported rather than hidden: photon-driven or adaptive acquisition timing
        has no single interval, and a step derived from a weak mode is a display
        convenience, not a property of the instrument.
        """
        return self.positive_interval_count > 0 and self.mode_fraction < 0.5

    @property
    def has_gaps(self) -> bool:
        """The modal grid contains frames no localization was measured in."""
        return self.missing_frame_count > 0


@dataclass(frozen=True)
class TimeAxis:
    """A dataset's time axis: row-aligned values plus what a slider needs.

    ``values`` is aligned to ``ds.loc_nm`` / ``ds.attr`` rows, so it combines
    directly with ``filter_mask`` and with a per-channel row selection.
    """

    mode: str
    values: np.ndarray
    lo: float
    hi: float
    unit: str
    step: float
    interval: float | None
    diagnostics: TimeIntervalDiagnostics | None
    n_traces: int

    @property
    def span(self) -> float:
        return float(self.hi) - float(self.lo)

    @property
    def is_index(self) -> bool:
        return self.mode == "index"

    def format_value(self, value: float) -> str:
        if self.is_index:
            return f"{int(round(float(value)))}"
        return format_time_seconds(float(value))

    def format_range(self, lo: float, hi: float) -> str:
        if self.is_index:
            return f"{int(round(float(lo)))} – {int(round(float(hi)))}"
        return (
            f"{format_time_seconds(float(lo))} – "
            f"{format_time_seconds(float(hi))}"
        )

    def describe(self) -> str:
        """One line naming the mode, the extent and the measured regularity."""
        head = (
            f"{TIME_MODE_LABELS.get(self.mode, self.mode)} · "
            f"{self.format_range(self.lo, self.hi)} · {self.n_traces:,} trace(s)"
        )
        diag = self.diagnostics
        if diag is None or diag.baseline_interval_ticks is None:
            return head
        parts = [
            head,
            f"modal Δt {format_time_seconds(diag.baseline_interval_s)} "
            f"({diag.mode_count:,}/{diag.positive_interval_count:,}, "
            f"{100.0 * diag.mode_fraction:.0f}%)",
        ]
        if diag.low_confidence:
            parts.append("low-confidence interval")
        if diag.missing_frame_count:
            parts.append(f"{diag.missing_frame_count:,} unmeasured grid frames")
        if diag.irregular_interval_count:
            parts.append(f"{diag.irregular_interval_count:,} irregular intervals")
        return " · ".join(parts)


def timestamp_precision_seconds(value: float, unit: str) -> float:
    """Convert a positive timestamp rounding precision to seconds."""
    try:
        scale = TIME_UNIT_SECONDS[str(unit).strip().lower()]
    except KeyError as exc:
        raise ValueError(f"Unsupported time unit: {unit!r}") from exc
    precision = float(value) * scale
    if not np.isfinite(precision) or precision <= 0.0:
        raise ValueError("Timestamp precision must be a finite positive value.")
    return precision


def tracking_precision_from_prefs(prefs: dict) -> float:
    """Read the persisted timestamp rounding precision, defaulting to 1 microsecond."""
    tracking = prefs.get("tracking", {}) if isinstance(prefs, dict) else {}
    return timestamp_precision_seconds(
        tracking.get("timestamp_precision_value", 1.0),
        tracking.get("timestamp_precision_unit", "us"),
    )


def _validated_trace_codes(trace_ids: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    values = np.asarray(trace_ids).reshape(-1)
    if values.dtype.kind in "fc" and not np.all(np.isfinite(values)):
        raise ValueError("Trace IDs must be finite.")
    try:
        unique, codes = np.unique(values, return_inverse=True)
    except TypeError as exc:
        raise ValueError("Trace IDs must have one comparable scalar type.") from exc
    return unique, codes.astype(np.int64, copy=False)


def relative_trace_index(trace_ids: np.ndarray) -> np.ndarray:
    """Return each row's zero-based occurrence index within its trace.

    Rows do not need to be contiguous by trace.  Their source order is retained.
    """
    values = np.asarray(trace_ids).reshape(-1)
    if values.size == 0:
        return np.empty(0, dtype=np.int64)
    _unique, codes = _validated_trace_codes(values)
    order = np.argsort(codes, kind="stable")
    sorted_codes = codes[order]
    starts = np.flatnonzero(np.r_[True, sorted_codes[1:] != sorted_codes[:-1]])
    counts = np.diff(np.r_[starts, values.size])
    ranks = np.arange(values.size, dtype=np.int64) - np.repeat(starts, counts)
    result = np.empty(values.size, dtype=np.int64)
    result[order] = ranks
    return result


def rounded_relative_time_ticks(
    timestamps_s: np.ndarray,
    trace_ids: np.ndarray,
    precision_s: float,
) -> np.ndarray:
    """Round timestamps, then zero the rounded onset of every trace.

    The trace onset is the earliest rounded timestamp, not merely the first row,
    so interleaved or reordered localization rows remain well defined.
    """
    tim = np.asarray(timestamps_s, dtype=float).reshape(-1)
    tids = np.asarray(trace_ids).reshape(-1)
    if tim.size != tids.size:
        raise ValueError("timestamps and trace IDs must have the same length.")
    if tim.size == 0:
        return np.empty(0, dtype=np.int64)
    precision_s = float(precision_s)
    if not np.isfinite(precision_s) or precision_s <= 0.0:
        raise ValueError("Timestamp precision must be a finite positive value.")
    if not np.all(np.isfinite(tim)):
        raise ValueError("Timestamps must be finite.")
    _unique, codes = _validated_trace_codes(tids)

    scaled = tim / precision_s
    limit = float(np.iinfo(np.int64).max - 1)
    if not np.all(np.isfinite(scaled)) or np.max(np.abs(scaled)) > limit:
        raise ValueError("Timestamps are too large for the selected precision.")
    ticks = np.rint(scaled).astype(np.int64)
    onsets = np.full(int(np.max(codes)) + 1, np.iinfo(np.int64).max, dtype=np.int64)
    np.minimum.at(onsets, codes, ticks)
    return ticks - onsets[codes]


def _ordered_intervals(
    relative_ticks: np.ndarray,
    trace_ids: np.ndarray,
) -> tuple[np.ndarray, int]:
    ticks = np.asarray(relative_ticks, dtype=np.int64).reshape(-1)
    tids = np.asarray(trace_ids).reshape(-1)
    if ticks.size != tids.size:
        raise ValueError("relative ticks and trace IDs must have the same length.")
    if ticks.size <= 1:
        return np.empty(0, dtype=np.int64), 0
    _unique, codes = _validated_trace_codes(tids)
    order = np.lexsort((ticks, codes))
    ordered_ticks = ticks[order]
    ordered_codes = codes[order]
    same_trace = ordered_codes[1:] == ordered_codes[:-1]
    differences = np.diff(ordered_ticks)[same_trace]
    return differences[differences > 0], int(np.count_nonzero(differences == 0))


def analyze_time_intervals(
    relative_ticks: np.ndarray,
    trace_ids: np.ndarray,
    precision_s: float,
) -> TimeIntervalDiagnostics:
    """Find the modal positive interval and diagnose gaps on its shared grid."""
    ticks = np.asarray(relative_ticks, dtype=np.int64).reshape(-1)
    tids = np.asarray(trace_ids).reshape(-1)
    positive, duplicate_count = _ordered_intervals(ticks, tids)
    if positive.size == 0:
        return TimeIntervalDiagnostics(
            precision_s=float(precision_s),
            baseline_interval_ticks=None,
            positive_interval_count=0,
            mode_count=0,
            duplicate_interval_count=duplicate_count,
            irregular_interval_count=0,
            missing_frame_count=0,
            off_grid_point_count=0,
        )

    values, counts = np.unique(positive, return_counts=True)
    winning_count = int(np.max(counts))
    # ``values`` is ascending, so ties intentionally choose the smallest mode.
    baseline = int(values[np.flatnonzero(counts == winning_count)[0]])
    nearest_multiple = np.maximum(1, np.rint(positive / baseline).astype(np.int64))
    residual = np.abs(positive - nearest_multiple * baseline)
    tolerance = max(1, int(round(0.05 * baseline)))
    regular = residual <= tolerance
    missing = int(np.sum(np.maximum(nearest_multiple[regular] - 1, 0)))
    frames = np.rint(ticks / baseline).astype(np.int64)
    off_grid = int(np.count_nonzero(np.abs(ticks - frames * baseline) > tolerance))
    return TimeIntervalDiagnostics(
        precision_s=float(precision_s),
        baseline_interval_ticks=baseline,
        positive_interval_count=int(positive.size),
        mode_count=winning_count,
        duplicate_interval_count=duplicate_count,
        irregular_interval_count=int(np.count_nonzero(~regular)),
        missing_frame_count=missing,
        off_grid_point_count=off_grid,
    )


def derived_step(span: float, interval: float | None) -> float:
    """The slider's wheel/playback step: the modal interval where it is usable.

    The measured interval is the scientifically meaningful step, but on a long
    acquisition it can be a millionth of the span, which would make the wheel
    and playback useless.  It is therefore raised to
    ``span / _MIN_STEPS_PER_SPAN`` when it is finer than that -- a display
    decision, never a claim about the instrument.  The measured value stays
    visible in :meth:`TimeAxis.describe`.
    """
    span = float(span)
    floor = span / _MIN_STEPS_PER_SPAN if span > 0 else 0.0
    if interval is None or not np.isfinite(interval) or interval <= 0.0:
        return max(floor, 1e-12)
    return max(float(interval), floor)


def time_axis_values(
    timestamps_s: np.ndarray | None,
    trace_ids: np.ndarray | None,
    *,
    mode: str,
    precision_s: float = 1.0e-6,
    with_diagnostics: bool = True,
) -> tuple[np.ndarray, str, TimeIntervalDiagnostics | None]:
    """Per-row axis values for *mode*, their unit, and interval diagnostics.

    ``trace`` and ``index`` need ``trace_ids``; ``absolute`` and ``trace`` need
    ``timestamps_s``.  A missing input raises rather than falling back to a
    different axis: a slider silently gating on something other than what its
    label says is worse than one that is unavailable and says why.

    ⚠ ``with_diagnostics=False`` is the cheap path, and on a large acquisition
    the difference is the whole cost.  Measured at 20,000,000 rows:
    ``analyze_time_intervals`` is **4.50 s** (a lexsort over every row), against
    1.12 s for the ``trace`` values, 1.18 s for the ``index`` values and 0.02 s
    for the ``absolute`` ones.  A caller that only needs to know which rows fall
    in a window -- which is every render path -- must not pay it.
    """
    normalized = str(mode).strip().lower()
    if normalized not in TIME_MODES:
        raise ValueError(f"mode must be one of {TIME_MODES!r}.")

    if normalized == "index":
        if trace_ids is None:
            raise ValueError("Index mode requires an aligned 'tid' attribute.")
        return relative_trace_index(trace_ids).astype(float), "index", None

    if timestamps_s is None:
        raise ValueError(f"{normalized} mode requires an aligned 'tim' attribute.")
    tim = np.asarray(timestamps_s, dtype=float).reshape(-1)
    if trace_ids is None:
        if normalized == "trace":
            raise ValueError("Trace-relative time requires an aligned 'tid' attribute.")
        return tim, "s", None

    if normalized == "absolute" and not with_diagnostics:
        return tim, "s", None            # the values are 'tim' itself: no work
    ticks = rounded_relative_time_ticks(tim, trace_ids, precision_s)
    diagnostics = (
        analyze_time_intervals(ticks, trace_ids, precision_s)
        if with_diagnostics else None
    )
    if normalized == "trace":
        return ticks.astype(float) * float(precision_s), "s", diagnostics
    return tim, "s", diagnostics


def build_time_axis(
    timestamps_s: np.ndarray | None,
    trace_ids: np.ndarray | None,
    *,
    mode: str,
    precision_s: float = 1.0e-6,
    keep: np.ndarray | None = None,
    with_diagnostics: bool = True,
    values: np.ndarray | None = None,
    diagnostics: TimeIntervalDiagnostics | None = None,
    n_traces: int | None = None,
) -> TimeAxis:
    """Assemble a :class:`TimeAxis`, bounding it to the rows *keep* selects.

    ``values`` stays full length and row-aligned; only the reported ``lo``/``hi``
    honour *keep*, so the slider spans what is actually drawable while the gate
    it produces can still be applied to any row selection.

    Pass ``values`` to reuse a row array the caller has already computed and
    cached, and ``diagnostics`` likewise.  ⚠ The diagnostics do **not** depend on
    the mode: they are computed from the trace-relative ticks, which are the same
    whichever axis is displayed -- so one measurement serves ``absolute`` and
    ``trace`` both, and a caller that caches them per (dataset, precision) is
    caching the right thing.  Without them the step falls back to ``span / 200``,
    which is a usable slider immediately; the measured interval replaces it as
    soon as someone asks for it.
    """
    reuse = values is not None and (diagnostics is not None or not with_diagnostics)
    if reuse:
        unit = "index" if str(mode).strip().lower() == "index" else "s"
        values = np.asarray(values, dtype=float)
    else:
        values, unit, measured = time_axis_values(
            timestamps_s, trace_ids, mode=mode, precision_s=precision_s,
            with_diagnostics=with_diagnostics,
        )
        diagnostics = diagnostics if diagnostics is not None else measured
    finite = np.isfinite(values)
    if keep is not None:
        keep_mask = np.asarray(keep, dtype=bool).reshape(-1)
        if keep_mask.size == values.size:
            finite = finite & keep_mask
    usable = values[finite]
    if usable.size:
        lo, hi = float(np.min(usable)), float(np.max(usable))
    else:
        lo, hi = 0.0, 0.0
    if hi <= lo:
        hi = lo + (1.0 if unit == "index" else max(abs(lo) * 1e-9, 1e-9))
    interval = None if diagnostics is None else diagnostics.baseline_interval_s
    if unit == "index":
        interval = 1.0
    # The trace count is reported in ``describe()`` and nowhere else, and
    # counting it is a sort over every row -- so it is taken from the caller when
    # it has one to hand, and otherwise counted only on the path that has already
    # paid for a full pass. A cheap axis rebuild must not add one.
    if n_traces is None:
        n_traces = 0
        if trace_ids is not None and diagnostics is not None:
            try:
                n_traces = int(np.unique(np.asarray(trace_ids).reshape(-1)).size)
            except ValueError:
                n_traces = 0
    n_traces = int(n_traces)
    return TimeAxis(
        mode=str(mode).strip().lower(),
        values=values,
        lo=lo,
        hi=hi,
        unit=unit,
        step=derived_step(hi - lo, interval),
        interval=interval,
        diagnostics=diagnostics,
        n_traces=n_traces,
    )


def format_time_seconds(seconds: float | None) -> str:
    """Compact human-readable time for slider labels and diagnostics."""
    if seconds is None or not np.isfinite(seconds):
        return "n/a"
    value = abs(float(seconds))
    if value == 0.0:
        return "0 s"
    if value < 1.0e-6:
        return f"{seconds * 1.0e9:.3g} ns"
    if value < 1.0e-3:
        return f"{seconds * 1.0e6:.3g} µs"
    if value < 1.0:
        return f"{seconds * 1.0e3:.3g} ms"
    return f"{seconds:.4g} s"


def dataset_time_inputs(ds) -> tuple:
    """``(tim, tid)`` for *ds*, row-aligned to ``ds.attr`` / ``ds.loc_nm``.

    Either may be ``None`` when the dataset does not carry it -- a generic
    imported table often has neither.  ``attr_values_1d`` is the accessor
    because it is the one that resolves a non-last-valid materialization; a
    direct ``ds.attr[...]`` read can be 2-D and would not align.
    """
    from .loader import attr_values_1d

    def read(name):
        try:
            values = attr_values_1d(ds, name)
        except Exception:
            return None
        if values is None:
            return None
        array = np.asarray(values).reshape(-1)
        return array if array.size else None

    return read("tim"), read("tid")


def dataset_time_axis(
    ds,
    *,
    mode: str,
    precision_s: float = 1.0e-6,
    keep: np.ndarray | None = None,
    with_diagnostics: bool = True,
    values: np.ndarray | None = None,
    diagnostics: TimeIntervalDiagnostics | None = None,
) -> TimeAxis:
    """The :class:`TimeAxis` of *ds* under *mode*, raising when unavailable."""
    tim, tid = dataset_time_inputs(ds)
    counted = getattr(getattr(ds, "prop", None), "num_traces", None)
    return build_time_axis(
        tim, tid, mode=mode, precision_s=precision_s, keep=keep,
        with_diagnostics=with_diagnostics, values=values, diagnostics=diagnostics,
        n_traces=int(counted) if isinstance(counted, (int, np.integer)) else None,
    )


def dataset_time_diagnostics(ds, *, precision_s: float = 1.0e-6):
    """The interval statistics of *ds*, or ``None`` when it has no time axis.

    Independent of the display mode, so a caller caches one per (dataset,
    precision).  This is the expensive half of the axis -- 4.50 s of 5.92 s at
    20,000,000 rows -- so it is asked for separately and only when needed.
    """
    tim, tid = dataset_time_inputs(ds)
    if tim is None or tid is None:
        return None
    ticks = rounded_relative_time_ticks(tim, tid, precision_s)
    return analyze_time_intervals(ticks, tid, precision_s)


def available_time_modes(ds) -> tuple[str, ...]:
    """The modes *ds* can actually offer, in :data:`TIME_MODES` order."""
    tim, tid = dataset_time_inputs(ds)
    modes = []
    if tim is not None:
        modes.append("absolute")
        if tid is not None:
            modes.append("trace")
    if tid is not None:
        modes.append("index")
    return tuple(mode for mode in TIME_MODES if mode in modes)
