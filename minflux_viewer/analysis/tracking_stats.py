"""Pure foundations for trajectory processing and tracking analysis.

The module deliberately separates four concerns:

``prepare_trajectories``
    validates row-aligned coordinates/trace IDs/time, rounds timestamps without
    overwriting them, preserves source-row provenance, and makes gaps explicit;
``kinematic_statistics``
    derives jumps, speed, turning angle and segment summaries;
``mean_squared_displacement``
    bins *measured* time differences (no synthetic interpolation), and reports
    time-averaged and ensemble curves plus diagnostic fits;
``TrackingMethodSpec``
    is the small registry contract through which later built-ins and experiment-
    specific plugins can consume the same input and return the same result form.

Coordinates are nanometres. Timestamp input and diffusion output use seconds
when ``time_source='timestamp'``; index mode is explicitly reported as
``nm²/index`` and is never presented as a physical diffusion coefficient.
"""

from __future__ import annotations

from collections.abc import Callable, Mapping
from dataclasses import dataclass, field
from typing import Any

import numpy as np

from ..core.tracking_time import analyze_time_intervals, relative_trace_index

MICHALET_2010_DOI = "https://doi.org/10.1103/PhysRevE.82.041914"
BERGLUND_2010_DOI = "https://doi.org/10.1103/PhysRevE.82.011917"


@dataclass(frozen=True)
class PreparationDiagnostics:
    """Auditable account of which source rows and boundaries were used."""

    n_input_rows: int
    n_used_rows: int
    n_traces: int
    n_segments: int
    excluded_nonfinite: int = 0
    excluded_by_filter: int = 0
    filter_boundaries: int = 0
    nonpositive_time_boundaries: int = 0
    long_gap_boundaries: int = 0
    warnings: tuple[str, ...] = ()


@dataclass(frozen=True)
class PreparedTrajectories:
    """Validated, trace/time-sorted rows shared by every tracking method.

    ``row_indices`` maps every row back to its materialized dataset row.
    ``trace_time`` is zeroed at the onset of the original trace, not at the
    filter boundary. ``segment_codes`` splits a trace wherever a source row was
    removed, time did not advance, or a configured long gap occurred.
    """

    coords_nm: np.ndarray
    trace_ids: np.ndarray
    trace_codes: np.ndarray
    segment_codes: np.ndarray
    row_indices: np.ndarray
    absolute_time: np.ndarray
    trace_time: np.ndarray
    time_unit: str
    time_source: str
    timestamp_precision_s: float | None
    baseline_interval: float | None
    gap_threshold: float | None
    diagnostics: PreparationDiagnostics
    provenance: Mapping[str, Any] = field(default_factory=dict)

    @property
    def n_points(self) -> int:
        return int(self.row_indices.size)

    @property
    def n_traces(self) -> int:
        return int(np.unique(self.trace_codes).size)

    @property
    def n_segments(self) -> int:
        return int(self.segment_codes[-1] + 1) if self.segment_codes.size else 0

    @property
    def segment_starts(self) -> np.ndarray:
        if not self.segment_codes.size:
            return np.empty(0, dtype=np.int64)
        return np.flatnonzero(
            np.r_[True, self.segment_codes[1:] != self.segment_codes[:-1]]
        ).astype(np.int64)

    @property
    def segment_counts(self) -> np.ndarray:
        starts = self.segment_starts
        return np.diff(np.r_[starts, self.n_points]).astype(np.int64)


@dataclass(frozen=True)
class DatasetTrajectorySnapshot:
    """Non-Qt, read-only dataset inputs safe to hand to an analysis worker."""

    dataset_identity: int
    dataset_name: str
    dataset_did: Any
    x_m: np.ndarray
    y_m: np.ndarray
    z_m: np.ndarray | None
    z_scaling_factor: float
    trace_ids: np.ndarray
    timestamps_s: np.ndarray | None
    keep: np.ndarray | None
    source_scope: str


@dataclass(frozen=True)
class TrackingDiagnostics:
    """Method-level status that travels beside every numeric result."""

    status: str
    n_input_rows: int
    n_used_rows: int
    n_traces: int
    n_segments: int
    excluded: Mapping[str, int] = field(default_factory=dict)
    warnings: tuple[str, ...] = ()


@dataclass(frozen=True)
class TrackingResult:
    """Common result envelope for built-in and plugin tracking methods.

    Tables are grouped by analysis level (for example ``localization``,
    ``segment`` and ``ensemble``). Every column in a table has equal length.
    This deliberately simple mapping can feed the results window, CSV export,
    the scripting API, or a plugin without depending on Qt or pandas.
    """

    method_id: str
    method_version: str
    tables: Mapping[str, Mapping[str, np.ndarray]]
    units: Mapping[str, Mapping[str, str]]
    diagnostics: TrackingDiagnostics
    provenance: Mapping[str, Any]
    citations: tuple[str, ...] = ()

    def table(self, level: str) -> Mapping[str, np.ndarray]:
        try:
            return self.tables[level]
        except KeyError as exc:
            raise KeyError(f"Tracking result has no {level!r} table.") from exc


@dataclass(frozen=True)
class TrackingMethodSpec:
    """One extensible analysis entry point suitable for a built-in or plugin."""

    method_id: str
    label: str
    version: str
    description: str
    runner: Callable[..., TrackingResult]
    citations: tuple[str, ...] = ()


_TRACKING_METHODS: dict[str, TrackingMethodSpec] = {}


def register_tracking_method(spec: TrackingMethodSpec, *, replace: bool = False) -> None:
    """Register a method, refusing accidental identifier collisions."""
    key = str(spec.method_id).strip()
    if not key:
        raise ValueError("A tracking method needs a non-empty method_id.")
    if key in _TRACKING_METHODS and not replace:
        raise ValueError(f"Tracking method {key!r} is already registered.")
    _TRACKING_METHODS[key] = spec


def tracking_methods() -> tuple[TrackingMethodSpec, ...]:
    """Registered methods in stable identifier order."""
    return tuple(_TRACKING_METHODS[key] for key in sorted(_TRACKING_METHODS))


def get_tracking_method(method_id: str) -> TrackingMethodSpec:
    try:
        return _TRACKING_METHODS[str(method_id)]
    except KeyError as exc:
        raise KeyError(f"Unknown tracking method {method_id!r}.") from exc


def run_tracking_method(
    method_id: str, data: PreparedTrajectories, **parameters,
) -> TrackingResult:
    """Run a registered method against an already prepared immutable snapshot."""
    return get_tracking_method(method_id).runner(data, **parameters)


def _validated_trace_codes(values: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    try:
        unique, codes = np.unique(values, return_inverse=True)
    except TypeError as exc:
        raise ValueError("Trace IDs must have one comparable scalar type.") from exc
    return unique, codes.astype(np.int64, copy=False)


def _finite_trace_ids(values: np.ndarray) -> np.ndarray:
    if values.dtype.kind in "fc":
        return np.isfinite(values)
    if values.dtype.kind == "O":
        return np.fromiter(
            (item is not None for item in values), dtype=bool, count=values.size)
    return np.ones(values.size, dtype=bool)


def _timestamp_ticks(timestamps_s: np.ndarray, precision_s: float) -> np.ndarray:
    precision = float(precision_s)
    if not np.isfinite(precision) or precision <= 0.0:
        raise ValueError("timestamp_precision_s must be finite and positive.")
    scaled = timestamps_s / precision
    limit = float(np.iinfo(np.int64).max - 1)
    if np.any(np.abs(scaled[np.isfinite(scaled)]) > limit):
        raise ValueError("Timestamps are too large for the selected precision.")
    ticks = np.zeros(timestamps_s.size, dtype=np.int64)
    finite = np.isfinite(scaled)
    ticks[finite] = np.rint(scaled[finite]).astype(np.int64)
    return ticks


def prepare_trajectories(
    coords_nm: np.ndarray,
    trace_ids: np.ndarray,
    timestamps_s: np.ndarray | None = None,
    *,
    keep: np.ndarray | None = None,
    timestamp_precision_s: float = 1.0e-6,
    gap_factor: float | None = 5.0,
    max_gap_s: float | None = None,
    source_scope: str = "unfiltered",
    provenance: Mapping[str, Any] | None = None,
) -> PreparedTrajectories:
    """Create the canonical analysis input without interpolating any position.

    Timestamp mode rounds into integer precision ticks before subtracting each
    trace onset. Index mode is selected by passing ``timestamps_s=None``.
    Filtering is applied only after onset and baseline-interval calculation, so
    filtering cannot silently redefine time zero. Removed middle rows become
    segment boundaries and are never bridged by a step or MSD pair.

    The automatic long-gap split uses ``gap_factor * modal_interval`` only when
    the modal interval describes at least half of positive intervals. For an
    adaptive/irregular acquisition, no such split is guessed; callers can pass
    ``max_gap_s`` explicitly.
    """
    coords = np.asarray(coords_nm, dtype=np.float64)
    tids = np.asarray(trace_ids).reshape(-1)
    if coords.ndim != 2 or coords.shape[1] not in (2, 3):
        raise ValueError("coords_nm must be an N x 2 or N x 3 array.")
    if coords.shape[0] != tids.size:
        raise ValueError("Coordinates and trace IDs must be row-aligned.")
    if coords.shape[1] == 2:
        coords = np.column_stack([coords, np.zeros(coords.shape[0], dtype=float)])
    n_input = tids.size
    if keep is None:
        keep_mask = np.ones(n_input, dtype=bool)
    else:
        keep_mask = np.asarray(keep, dtype=bool).reshape(-1)
        if keep_mask.size != n_input:
            raise ValueError("keep must be row-aligned with the trajectory input.")

    finite_tid = _finite_trace_ids(tids)
    finite_coords = np.all(np.isfinite(coords[:, :3]), axis=1)
    if timestamps_s is None:
        time_source = "index"
        time_unit = "index"
        # Occurrence index is computed before coordinate/filter exclusions.
        if not np.all(finite_tid):
            source_values = np.zeros(n_input, dtype=np.int64)
            source_values[finite_tid] = relative_trace_index(tids[finite_tid])
        else:
            source_values = relative_trace_index(tids)
        ticks = np.asarray(source_values, dtype=np.int64)
        precision: float | None = None
        finite_time = np.ones(n_input, dtype=bool)
    else:
        time_source = "timestamp"
        time_unit = "s"
        timestamps = np.asarray(timestamps_s, dtype=np.float64).reshape(-1)
        if timestamps.size != n_input:
            raise ValueError("Coordinates, trace IDs and timestamps must be row-aligned.")
        finite_time = np.isfinite(timestamps)
        precision = float(timestamp_precision_s)
        ticks = _timestamp_ticks(timestamps, precision)

    usable = finite_tid & finite_coords & finite_time
    rows = np.flatnonzero(usable).astype(np.int64)
    if rows.size == 0:
        raise ValueError("No finite trajectory rows remain after validation.")
    _unique, full_codes = _validated_trace_codes(tids[rows])
    order = np.lexsort((rows, ticks[rows], full_codes))
    rows = rows[order]
    full_codes = full_codes[order]
    full_ticks = ticks[rows]

    trace_starts = np.flatnonzero(
        np.r_[True, full_codes[1:] != full_codes[:-1]]).astype(np.int64)
    trace_counts = np.diff(np.r_[trace_starts, rows.size]).astype(np.int64)
    source_rank = (
        np.arange(rows.size, dtype=np.int64)
        - np.repeat(trace_starts, trace_counts)
    )
    relative_ticks = full_ticks - np.repeat(full_ticks[trace_starts], trace_counts)

    interval_diagnostics = None
    if time_source == "timestamp":
        interval_diagnostics = analyze_time_intervals(
            relative_ticks, tids[rows], float(precision))
        baseline = interval_diagnostics.baseline_interval_s
    else:
        baseline = 1.0

    selected = keep_mask[rows]
    excluded_filter = int(np.count_nonzero(~selected))
    rows = rows[selected]
    full_codes = full_codes[selected]
    full_ticks = full_ticks[selected]
    relative_ticks = relative_ticks[selected]
    source_rank = source_rank[selected]
    if rows.size == 0:
        raise ValueError("The active filter removes every finite trajectory row.")

    # Recode after selection so trace codes are dense even when a whole trace is
    # filtered out. Sorting remains trace/time order.
    _selected_unique, trace_codes = _validated_trace_codes(tids[rows])
    same_trace = trace_codes[1:] == trace_codes[:-1]
    missing_source = same_trace & (source_rank[1:] != source_rank[:-1] + 1)
    dt_ticks = np.diff(full_ticks)
    nonpositive = same_trace & (dt_ticks <= 0)

    if max_gap_s is not None:
        gap_threshold = float(max_gap_s)
        if not np.isfinite(gap_threshold) or gap_threshold <= 0.0:
            raise ValueError("max_gap_s must be finite and positive.")
    elif (
        time_source == "timestamp"
        and gap_factor is not None
        and baseline is not None
        and interval_diagnostics is not None
        and not interval_diagnostics.low_confidence
    ):
        factor = float(gap_factor)
        if not np.isfinite(factor) or factor <= 1.0:
            raise ValueError("gap_factor must be greater than one, or None.")
        gap_threshold = factor * float(baseline)
    else:
        gap_threshold = None

    if time_source == "timestamp" and gap_threshold is not None:
        long_gap = same_trace & (dt_ticks.astype(float) * float(precision) > gap_threshold)
    elif time_source == "index" and gap_factor is not None:
        long_gap = same_trace & (dt_ticks > float(gap_factor))
        gap_threshold = float(gap_factor)
    else:
        long_gap = np.zeros(max(rows.size - 1, 0), dtype=bool)

    break_before = np.r_[
        True,
        (~same_trace) | missing_source | nonpositive | long_gap,
    ]
    segment_codes = np.cumsum(break_before, dtype=np.int64) - 1

    warnings: list[str] = []
    if interval_diagnostics is not None and interval_diagnostics.low_confidence:
        warnings.append(
            "No automatic long-gap threshold was inferred because the modal "
            "timestamp interval describes fewer than half of measured intervals."
        )
    if time_source == "index":
        warnings.append(
            "Index mode has no physical time unit; diffusion coefficients are per index."
        )
    diagnostics = PreparationDiagnostics(
        n_input_rows=n_input,
        n_used_rows=int(rows.size),
        n_traces=int(np.unique(trace_codes).size),
        n_segments=int(segment_codes[-1] + 1),
        excluded_nonfinite=int(n_input - np.count_nonzero(usable)),
        excluded_by_filter=excluded_filter,
        filter_boundaries=int(np.count_nonzero(missing_source)),
        nonpositive_time_boundaries=int(np.count_nonzero(nonpositive)),
        long_gap_boundaries=int(np.count_nonzero(long_gap)),
        warnings=tuple(warnings),
    )

    scale = 1.0 if time_source == "index" else float(precision)
    return PreparedTrajectories(
        coords_nm=np.ascontiguousarray(coords[rows, :3]),
        trace_ids=np.ascontiguousarray(tids[rows]),
        trace_codes=np.ascontiguousarray(trace_codes),
        segment_codes=np.ascontiguousarray(segment_codes),
        row_indices=np.ascontiguousarray(rows),
        absolute_time=np.ascontiguousarray(full_ticks.astype(float) * scale),
        trace_time=np.ascontiguousarray(relative_ticks.astype(float) * scale),
        time_unit=time_unit,
        time_source=time_source,
        timestamp_precision_s=precision,
        baseline_interval=float(baseline) if baseline is not None else None,
        gap_threshold=gap_threshold,
        diagnostics=diagnostics,
        provenance={
            "source_scope": str(source_scope),
            "coordinate_unit": "nm",
            "time_source": time_source,
            "timestamp_precision_s": precision,
            "gap_factor": gap_factor,
            "max_gap_s": max_gap_s,
            **dict(provenance or {}),
        },
    )


def trajectories_from_dataset(
    dataset,
    *,
    filtered: bool = True,
    time_mode: str = "timestamp",
    timestamp_precision_s: float = 1.0e-6,
    gap_factor: float | None = 5.0,
    max_gap_s: float | None = None,
) -> PreparedTrajectories:
    """Snapshot a viewer dataset into the analysis model.

    Coordinates use calibrated ``dataset.loc_nm`` but intentionally exclude the
    overlay display transform: diffusion is measured in the dataset's physical
    coordinate frame. ``filtered=False`` means all materialized valid rows, not
    the all-iteration ``mfx_raw`` store.
    """
    snapshot = snapshot_dataset_trajectories(dataset, filtered=filtered)
    return prepare_dataset_trajectories(
        snapshot,
        time_mode=time_mode,
        timestamp_precision_s=timestamp_precision_s,
        gap_factor=gap_factor,
        max_gap_s=max_gap_s,
    )


def _readonly_vector(values, *, name: str) -> np.ndarray:
    array = np.asarray(values).reshape(-1).view()
    if not array.size:
        raise ValueError(f"Tracking analysis requires a non-empty {name!r} attribute.")
    array.flags.writeable = False
    return array


def snapshot_dataset_trajectories(
    dataset,
    *,
    filtered: bool = True,
) -> DatasetTrajectorySnapshot:
    """Capture immutable trajectory inputs without constructing coordinates.

    This intentionally does only cheap column lookup on the caller's thread.
    Coordinate calibration and the large N×3 allocation happen later in
    :func:`prepare_dataset_trajectories`, which a UI can run in a worker.
    """
    from ..core.loader import attr_values_1d
    from ..core.tracking_time import dataset_time_inputs

    x = attr_values_1d(dataset, "loc_x")
    y = attr_values_1d(dataset, "loc_y")
    z = attr_values_1d(dataset, "loc_z")
    if x is None or y is None:
        raise ValueError("Tracking analysis requires aligned localization coordinates.")
    x_view = _readonly_vector(x, name="loc_x")
    y_view = _readonly_vector(y, name="loc_y")
    z_view = None if z is None else _readonly_vector(z, name="loc_z")
    tim, tid = dataset_time_inputs(dataset)
    if tid is None:
        raise ValueError("Tracking analysis requires an aligned 'tid' attribute.")
    tid_view = _readonly_vector(tid, name="tid")
    tim_view = None if tim is None else _readonly_vector(tim, name="tim")
    sizes = {x_view.size, y_view.size, tid_view.size}
    if z_view is not None:
        sizes.add(z_view.size)
    if tim_view is not None:
        sizes.add(tim_view.size)
    if len(sizes) != 1:
        raise ValueError("Coordinates, trace IDs and timestamps must be row-aligned.")

    keep = None
    if filtered:
        keep = np.asarray(dataset.filter_mask, dtype=bool).reshape(-1).copy()
        if keep.size != x_view.size:
            raise ValueError("Dataset filter mask is not aligned to coordinates.")
        keep.flags.writeable = False
    metadata = getattr(dataset, "metadata", {})
    return DatasetTrajectorySnapshot(
        dataset_identity=id(dataset),
        dataset_name=str(getattr(dataset, "name", "")),
        dataset_did=metadata.get("did") if isinstance(metadata, dict) else None,
        x_m=x_view,
        y_m=y_view,
        z_m=z_view,
        z_scaling_factor=float(getattr(dataset.cali, "z_scaling_factor", 1.0)),
        trace_ids=tid_view,
        timestamps_s=tim_view,
        keep=keep,
        source_scope="filtered" if filtered else "materialized-unfiltered",
    )


def prepare_dataset_trajectories(
    snapshot: DatasetTrajectorySnapshot,
    *,
    time_mode: str = "timestamp",
    timestamp_precision_s: float = 1.0e-6,
    gap_factor: float | None = 5.0,
    max_gap_s: float | None = None,
) -> PreparedTrajectories:
    """Prepare a captured dataset snapshot; pure NumPy and worker-safe."""
    x = np.asarray(snapshot.x_m, dtype=np.float64).reshape(-1)
    y = np.asarray(snapshot.y_m, dtype=np.float64).reshape(-1)
    z = (
        np.zeros(x.size, dtype=np.float64)
        if snapshot.z_m is None
        else np.asarray(snapshot.z_m, dtype=np.float64).reshape(-1)
    )
    if x.size != y.size or x.size != z.size:
        raise ValueError("Coordinates must remain row-aligned while preparing trajectories.")
    coords = np.empty((x.size, 3), dtype=np.float64)
    coords[:, 0] = x * 1.0e9
    coords[:, 1] = y * 1.0e9
    coords[:, 2] = z * 1.0e9 * float(snapshot.z_scaling_factor)
    normalized = str(time_mode).strip().lower()
    if normalized not in {"timestamp", "index"}:
        raise ValueError("time_mode must be 'timestamp' or 'index'.")
    if normalized == "timestamp" and snapshot.timestamps_s is None:
        raise ValueError("Timestamp analysis requires an aligned 'tim' attribute.")
    return prepare_trajectories(
        coords,
        snapshot.trace_ids,
        snapshot.timestamps_s if normalized == "timestamp" else None,
        keep=snapshot.keep,
        timestamp_precision_s=timestamp_precision_s,
        gap_factor=gap_factor,
        max_gap_s=max_gap_s,
        source_scope=snapshot.source_scope,
        provenance={
            "dataset_name": snapshot.dataset_name,
            "dataset_did": snapshot.dataset_did,
            "coordinate_space": "calibrated dataset nm (no overlay transform)",
            "z_scaling_factor": float(snapshot.z_scaling_factor),
        },
    )


def _base_diagnostics(
    data: PreparedTrajectories,
    *,
    status: str = "ok",
    excluded: Mapping[str, int] | None = None,
    warnings: tuple[str, ...] = (),
) -> TrackingDiagnostics:
    return TrackingDiagnostics(
        status=status,
        n_input_rows=data.diagnostics.n_input_rows,
        n_used_rows=data.n_points,
        n_traces=data.n_traces,
        n_segments=data.n_segments,
        excluded=dict(excluded or {}),
        warnings=tuple(data.diagnostics.warnings) + tuple(warnings),
    )


def kinematic_statistics(data: PreparedTrajectories) -> TrackingResult:
    """Step distance, speed, turning angle, and per-segment path summaries."""
    n = data.n_points
    dt = np.full(n, np.nan, dtype=float)
    jump = np.full(n, np.nan, dtype=float)
    speed = np.full(n, np.nan, dtype=float)
    angle = np.full(n, np.nan, dtype=float)

    segment_rows: dict[str, list] = {
        "segment_id": [], "trace_id": [], "n_points": [], "duration": [],
        "path_length_nm": [], "net_displacement_nm": [], "straightness": [],
        "mean_speed": [],
    }
    starts, counts = data.segment_starts, data.segment_counts
    for segment_id, (start, count) in enumerate(zip(starts, counts)):
        end = int(start + count)
        points = data.coords_nm[start:end]
        times = data.absolute_time[start:end]
        steps = np.diff(points, axis=0)
        elapsed = np.diff(times)
        distances = np.linalg.norm(steps, axis=1)
        if count > 1:
            dt[start + 1:end] = elapsed
            jump[start + 1:end] = distances
            valid_dt = elapsed > 0
            speed_values = np.full(elapsed.size, np.nan, dtype=float)
            speed_values[valid_dt] = distances[valid_dt] / elapsed[valid_dt]
            speed[start + 1:end] = speed_values
        if count > 2:
            left, right = steps[:-1], steps[1:]
            denom = np.linalg.norm(left, axis=1) * np.linalg.norm(right, axis=1)
            valid = denom > 0
            values = np.full(count - 2, np.nan, dtype=float)
            values[valid] = np.arccos(np.clip(
                np.sum(left[valid] * right[valid], axis=1) / denom[valid], -1.0, 1.0))
            angle[start + 2:end] = values

        duration = float(times[-1] - times[0]) if count > 1 else 0.0
        path = float(np.sum(distances))
        net = float(np.linalg.norm(points[-1] - points[0])) if count > 1 else 0.0
        segment_rows["segment_id"].append(segment_id)
        segment_rows["trace_id"].append(data.trace_ids[start])
        segment_rows["n_points"].append(int(count))
        segment_rows["duration"].append(duration)
        segment_rows["path_length_nm"].append(path)
        segment_rows["net_displacement_nm"].append(net)
        segment_rows["straightness"].append(net / path if path > 0 else np.nan)
        segment_rows["mean_speed"].append(path / duration if duration > 0 else np.nan)

    segment_table = {key: np.asarray(values) for key, values in segment_rows.items()}
    time_suffix = "s" if data.time_unit == "s" else "index"
    speed_unit = "nm/s" if data.time_unit == "s" else "nm/index"
    return TrackingResult(
        method_id="minflux_viewer.tracking.kinematics",
        method_version="1.0",
        tables={
            "localization": {
                "row_index": data.row_indices.copy(),
                "trace_id": data.trace_ids.copy(),
                "segment_id": data.segment_codes.copy(),
                "time": data.trace_time.copy(),
                "dt": dt,
                "jump_distance_nm": jump,
                "speed": speed,
                "turning_angle_rad": angle,
            },
            "segment": segment_table,
        },
        units={
            "localization": {
                "time": time_suffix, "dt": time_suffix,
                "jump_distance_nm": "nm", "speed": speed_unit,
                "turning_angle_rad": "rad",
            },
            "segment": {
                "duration": time_suffix, "path_length_nm": "nm",
                "net_displacement_nm": "nm", "mean_speed": speed_unit,
            },
        },
        diagnostics=_base_diagnostics(data),
        provenance={**data.provenance, "segmentation_preserved": True},
    )


def _binned_curve(
    lag: np.ndarray,
    squared_displacement: np.ndarray,
    bin_width: float,
    *,
    min_pairs: int,
) -> dict[str, np.ndarray]:
    if lag.size == 0:
        empty_f = np.empty(0, dtype=float)
        return {
            "lag": empty_f, "nominal_lag": empty_f.copy(),
            "msd_nm2": empty_f.copy(), "sem_nm2": empty_f.copy(),
            "n_pairs": np.empty(0, dtype=np.int64),
        }
    codes = np.maximum(1, np.rint(lag / float(bin_width)).astype(np.int64))
    unique, inverse = np.unique(codes, return_inverse=True)
    counts = np.bincount(inverse).astype(np.int64, copy=False)
    lag_sum = np.bincount(inverse, weights=lag)
    value_sum = np.bincount(inverse, weights=squared_displacement)
    value_square_sum = np.bincount(inverse, weights=squared_displacement ** 2)
    accepted = counts >= int(min_pairs)
    count = counts[accepted]
    mean = value_sum[accepted] / count
    sem = np.full(count.size, np.nan, dtype=float)
    repeated = count > 1
    if np.any(repeated):
        variance = (
            value_square_sum[accepted][repeated]
            - value_sum[accepted][repeated] ** 2 / count[repeated]
        ) / (count[repeated] - 1)
        sem[repeated] = np.sqrt(np.maximum(variance, 0.0) / count[repeated])
    return {
        "lag": lag_sum[accepted] / count,
        "nominal_lag": unique[accepted].astype(float) * float(bin_width),
        "msd_nm2": mean,
        "sem_nm2": sem,
        "n_pairs": count,
    }


def _fit_msd_curve(
    curve: Mapping[str, np.ndarray],
    *,
    dimensions: int,
    fit_lags: int,
) -> dict[str, float | int | str]:
    lag = np.asarray(curve["lag"], dtype=float)
    msd = np.asarray(curve["msd_nm2"], dtype=float)
    count = np.asarray(curve["n_pairs"], dtype=float)
    valid = np.isfinite(lag) & np.isfinite(msd) & (lag > 0) & (msd > 0)
    lag, msd, count = lag[valid], msd[valid], count[valid]
    take = min(int(fit_lags), lag.size)
    if take < 2:
        return {
            "n_lags": take, "diffusion": np.nan, "alpha": np.nan,
            "intercept_nm2": np.nan, "apparent_sigma_nm": np.nan,
            "r_squared": np.nan, "reason": "fewer than two usable MSD lags",
        }
    lag, msd, count = lag[:take], msd[:take], count[:take]
    weights = np.sqrt(np.maximum(count, 1.0))
    design = np.column_stack([lag, np.ones(take)])
    beta, *_ = np.linalg.lstsq(design * weights[:, None], msd * weights, rcond=None)
    slope, intercept = float(beta[0]), float(beta[1])
    fitted = design @ beta
    residual = float(np.sum((weights * (msd - fitted)) ** 2))
    total = float(np.sum(
        (weights * (msd - np.average(msd, weights=weights ** 2))) ** 2))
    r_squared = 1.0 - residual / total if total > 0 else np.nan
    alpha = float(np.polyfit(np.log(lag), np.log(msd), 1)[0])
    diffusion = slope / (2.0 * int(dimensions))
    apparent_sigma = (
        float(np.sqrt(intercept / (2.0 * int(dimensions))))
        if intercept >= 0 else np.nan
    )
    reason = "ok" if diffusion >= 0 else "negative fitted diffusion slope"
    return {
        "n_lags": take,
        "diffusion": diffusion,
        "alpha": alpha,
        "intercept_nm2": intercept,
        "apparent_sigma_nm": apparent_sigma,
        "r_squared": r_squared,
        "reason": reason,
    }


def mean_squared_displacement(
    data: PreparedTrajectories,
    *,
    dimensions: int = 2,
    lag_bin: float | None = None,
    max_lag_points: int | None = None,
    max_lag_fraction: float = 0.25,
    min_pairs: int = 3,
    fit_lags: int = 4,
    min_segment_points: int = 5,
) -> TrackingResult:
    """Irregular-time MSD curves and diagnostic short-lag fits.

    Pair displacements are formed only within explicit segments. Their actual
    measured time differences are binned; coordinates are never resampled or
    interpolated. The linear fit uses ``MSD = 2*d*D*t + intercept`` and the
    log-log slope reports descriptive ``alpha``. The intercept-derived sigma is
    labelled *apparent*: exposure-time motion blur and MINFLUX feedback-loop lag
    require acquisition metadata and are not silently guessed here.
    """
    dimensions = int(dimensions)
    if dimensions not in (1, 2, 3):
        raise ValueError("dimensions must be 1, 2 or 3.")
    if min_pairs < 1 or fit_lags < 2 or min_segment_points < 2:
        raise ValueError("min_pairs >= 1, fit_lags >= 2 and min_segment_points >= 2.")
    if not (0.0 < float(max_lag_fraction) <= 1.0):
        raise ValueError("max_lag_fraction must be in (0, 1].")
    if lag_bin is None:
        lag_bin = data.baseline_interval
    if lag_bin is None or not np.isfinite(lag_bin) or lag_bin <= 0:
        positive = []
        for start, count in zip(data.segment_starts, data.segment_counts):
            values = np.diff(data.absolute_time[start:start + count])
            positive.extend(values[values > 0])
        lag_bin = float(np.median(positive)) if positive else np.nan
    if not np.isfinite(lag_bin) or lag_bin <= 0:
        raise ValueError("A finite positive lag_bin could not be inferred.")

    segment_table: dict[str, list] = {
        "segment_id": [], "trace_id": [], "lag": [], "nominal_lag": [],
        "msd_nm2": [], "sem_nm2": [], "n_pairs": [],
    }
    fit_table: dict[str, list] = {
        "segment_id": [], "trace_id": [], "n_points": [], "n_lags": [],
        "diffusion": [], "alpha": [], "intercept_nm2": [],
        "apparent_sigma_nm": [], "r_squared": [], "reason": [],
    }
    pooled_lag: list[np.ndarray] = []
    pooled_dsq: list[np.ndarray] = []
    ensemble_lag: list[np.ndarray] = []
    ensemble_dsq: list[np.ndarray] = []
    too_short = 0
    no_curve = 0

    for segment_id, (start, count) in enumerate(
        zip(data.segment_starts, data.segment_counts)
    ):
        end = int(start + count)
        trace_id = data.trace_ids[start]
        if count < min_segment_points:
            too_short += 1
            continue
        points = data.coords_nm[start:end, :dimensions]
        times = data.absolute_time[start:end]
        if max_lag_points is None:
            maximum = max(4, int(np.floor((count - 1) * float(max_lag_fraction))))
        else:
            maximum = int(max_lag_points)
        maximum = min(max(maximum, 1), int(count - 1))
        lag_parts, dsq_parts = [], []
        for offset in range(1, maximum + 1):
            lag_values = times[offset:] - times[:-offset]
            displacements = points[offset:] - points[:-offset]
            valid = lag_values > 0
            if np.any(valid):
                lag_parts.append(lag_values[valid])
                dsq_parts.append(np.sum(displacements[valid] ** 2, axis=1))
        if not lag_parts:
            no_curve += 1
            continue
        pair_lag = np.concatenate(lag_parts)
        pair_dsq = np.concatenate(dsq_parts)
        curve = _binned_curve(pair_lag, pair_dsq, float(lag_bin), min_pairs=min_pairs)
        if curve["lag"].size == 0:
            no_curve += 1
            continue
        pooled_lag.append(pair_lag)
        pooled_dsq.append(pair_dsq)
        for row in range(curve["lag"].size):
            segment_table["segment_id"].append(segment_id)
            segment_table["trace_id"].append(trace_id)
            for key in ("lag", "nominal_lag", "msd_nm2", "sem_nm2", "n_pairs"):
                segment_table[key].append(curve[key][row])

        fit = _fit_msd_curve(curve, dimensions=dimensions, fit_lags=fit_lags)
        fit_table["segment_id"].append(segment_id)
        fit_table["trace_id"].append(trace_id)
        fit_table["n_points"].append(int(count))
        for key in (
            "n_lags", "diffusion", "alpha", "intercept_nm2",
            "apparent_sigma_nm", "r_squared", "reason",
        ):
            fit_table[key].append(fit[key])

        elapsed = times[1:] - times[0]
        valid = elapsed > 0
        if np.any(valid):
            ensemble_lag.append(elapsed[valid])
            ensemble_dsq.append(np.sum((points[1:][valid] - points[0]) ** 2, axis=1))

    pooled_curve = _binned_curve(
        np.concatenate(pooled_lag) if pooled_lag else np.empty(0),
        np.concatenate(pooled_dsq) if pooled_dsq else np.empty(0),
        float(lag_bin), min_pairs=min_pairs)
    ensemble_curve = _binned_curve(
        np.concatenate(ensemble_lag) if ensemble_lag else np.empty(0),
        np.concatenate(ensemble_dsq) if ensemble_dsq else np.empty(0),
        float(lag_bin), min_pairs=min_pairs)
    pooled_fit = _fit_msd_curve(
        pooled_curve, dimensions=dimensions, fit_lags=fit_lags)

    segment_arrays = {key: np.asarray(values) for key, values in segment_table.items()}
    fit_arrays = {key: np.asarray(values) for key, values in fit_table.items()}
    pooled_table = {key: value.copy() for key, value in pooled_curve.items()}
    ensemble_table = {key: value.copy() for key, value in ensemble_curve.items()}
    pooled_fit_table = {
        key: np.asarray([value]) for key, value in pooled_fit.items()
    }
    diffusion_unit = "nm²/s" if data.time_unit == "s" else "nm²/index"
    time_unit = data.time_unit
    warnings = (
        "The MSD intercept yields apparent localization sigma only; motion blur "
        "and MINFLUX feedback-loop dynamic error are not corrected without "
        "acquisition metadata.",
        "Alpha is a descriptive log-log slope over the reported short-lag fit range, "
        "not a motion-state classification.",
    )
    status = "ok" if pooled_curve["lag"].size else "insufficient-data"
    return TrackingResult(
        method_id="minflux_viewer.tracking.msd",
        method_version="1.0",
        tables={
            "segment_msd": segment_arrays,
            "pooled_time_averaged_msd": pooled_table,
            "ensemble_msd": ensemble_table,
            "segment_fit": fit_arrays,
            "pooled_fit": pooled_fit_table,
        },
        units={
            "segment_msd": {"lag": time_unit, "nominal_lag": time_unit,
                            "msd_nm2": "nm²", "sem_nm2": "nm²"},
            "pooled_time_averaged_msd": {
                "lag": time_unit, "nominal_lag": time_unit,
                "msd_nm2": "nm²", "sem_nm2": "nm²"},
            "ensemble_msd": {"lag": time_unit, "nominal_lag": time_unit,
                             "msd_nm2": "nm²", "sem_nm2": "nm²"},
            "segment_fit": {"diffusion": diffusion_unit, "alpha": "1",
                            "intercept_nm2": "nm²", "apparent_sigma_nm": "nm"},
            "pooled_fit": {"diffusion": diffusion_unit, "alpha": "1",
                           "intercept_nm2": "nm²", "apparent_sigma_nm": "nm"},
        },
        diagnostics=_base_diagnostics(
            data,
            status=status,
            excluded={"segments_too_short": too_short, "segments_without_curve": no_curve},
            warnings=warnings,
        ),
        provenance={
            **data.provenance,
            "dimensions": dimensions,
            "lag_bin": float(lag_bin),
            "max_lag_points": max_lag_points,
            "max_lag_fraction": float(max_lag_fraction),
            "min_pairs": int(min_pairs),
            "fit_lags": int(fit_lags),
            "min_segment_points": int(min_segment_points),
            "interpolation": "none",
            "fit_model": "MSD = 2*d*D*lag + intercept",
        },
        citations=(MICHALET_2010_DOI, BERGLUND_2010_DOI),
    )


register_tracking_method(TrackingMethodSpec(
    method_id="minflux_viewer.tracking.kinematics",
    label="Trajectory kinematics",
    version="1.0",
    description="Jump distance, speed, turning angle and segment path summaries.",
    runner=kinematic_statistics,
))
register_tracking_method(TrackingMethodSpec(
    method_id="minflux_viewer.tracking.msd",
    label="Irregular-time MSD",
    version="1.0",
    description="Measured-lag time-averaged and ensemble MSD with diagnostic fits.",
    runner=mean_squared_displacement,
    citations=(MICHALET_2010_DOI, BERGLUND_2010_DOI),
))


__all__ = [
    "DatasetTrajectorySnapshot",
    "PreparedTrajectories",
    "PreparationDiagnostics",
    "TrackingDiagnostics",
    "TrackingMethodSpec",
    "TrackingResult",
    "get_tracking_method",
    "kinematic_statistics",
    "mean_squared_displacement",
    "prepare_dataset_trajectories",
    "prepare_trajectories",
    "register_tracking_method",
    "run_tracking_method",
    "snapshot_dataset_trajectories",
    "tracking_methods",
    "trajectories_from_dataset",
]
