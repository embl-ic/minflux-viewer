"""Trajectories as a playable comet: the data structure behind the Tracking View.

A MINFLUX tracking acquisition is a set of short, fast traces scattered through a
long run -- on the reference file, 230 traces of a few tens of milliseconds each
inside 24 minutes.  Played on absolute time you would watch an empty field for
minutes at a stretch and almost never see two traces at once, so the default is
to **zero every trace to its own start** and play them together.  That is also
what the MATLAB NPC-trafficking workflow this view is modelled on does.

The one thing that has to be fast is *"which localizations are inside the tail
window right now"*, answered many times a second while the slider moves.  This
module answers it with no Python loop over traces:

1. rows are sorted **once** by ``(trace, t)``, so every trace occupies a
   contiguous, time-ordered block;
2. a monotone ``key = trace_code * scale + t`` is built over that order, which
   lets a **single** :func:`numpy.searchsorted` call resolve the window's start
   and end *in every trace at once*;
3. the selected rows are gathered with a vectorised ragged-range expansion.

So a frame costs ``O(T log N)`` to locate plus ``O(k)`` to gather, where *T* is
the trace count and *k* the number of points actually drawn -- never ``O(N)``.

Nothing here touches Qt, and nothing here modifies the dataset: a comet is a
*view* of rows that already exist.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

#: Tail length presets offered by the UI, in seconds of trace-relative time.
#: ``None`` means "everything since the trace began", the cumulative trail.
TAIL_PRESETS_S: tuple[float | None, ...] = (0.005, 0.02, 0.05, 0.2, 1.0, None)

#: How many age bands a comet tail is drawn in. Each band is one polyline at its
#: own opacity, which is how the fade is achieved without per-vertex colour --
#: pyqtgraph's curve item takes one pen for the whole line.
DEFAULT_TAIL_BANDS = 6


def ragged_indices(lo: np.ndarray, hi: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Expand per-trace ``[lo, hi)`` spans into one index array, without a loop.

    Returns ``(indices, counts)``.  This is the standard vectorised ragged-range
    construction: build the running offset of each span, repeat it across that
    span's length, and add a global arange.  It is what keeps a frame off a
    Python loop over traces.
    """
    lo = np.asarray(lo, dtype=np.int64).reshape(-1)
    hi = np.asarray(hi, dtype=np.int64).reshape(-1)
    counts = np.maximum(hi - lo, 0)
    total = int(counts.sum())
    if total == 0:
        return np.empty(0, dtype=np.int64), counts
    offsets = np.zeros(counts.size, dtype=np.int64)
    np.cumsum(counts[:-1], out=offsets[1:])
    return np.repeat(lo - offsets, counts) + np.arange(total, dtype=np.int64), counts


@dataclass(frozen=True)
class TrackWindow:
    """The rows a comet draws for one time window.

    ``indices`` addresses :attr:`TrackSet.coords`, already grouped by trace and
    ordered in time, so the points can be drawn as one polyline broken at the
    trace boundaries.  ``connect`` is that break pattern in pyqtgraph's own
    form: 1 to draw a segment to the next point, 0 to lift the pen.
    """

    indices: np.ndarray
    connect: np.ndarray
    counts: np.ndarray
    head_indices: np.ndarray
    head_traces: np.ndarray

    @property
    def size(self) -> int:
        return int(self.indices.size)

    @property
    def n_traces(self) -> int:
        return int(self.head_indices.size)


@dataclass(frozen=True)
class TrackSet:
    """Trace-ordered coordinates plus the index that makes windowing cheap.

    ``coords`` is ``(N, 3)`` display nm and ``t`` the per-row time in seconds on
    the chosen axis, both sorted by ``(trace, t)``.  ``trace_codes`` is the dense
    0-based trace number of each row, and ``order`` maps a row back to its
    position in the dataset, so a selection made here can be reported against the
    original rows.
    """

    coords: np.ndarray
    t: np.ndarray
    trace_codes: np.ndarray
    segment_codes: np.ndarray
    order: np.ndarray
    trace_ids: np.ndarray
    starts: np.ndarray
    counts: np.ndarray
    t_start: np.ndarray
    t_end: np.ndarray
    key: np.ndarray
    key_scale: float
    relative: bool

    @property
    def n_traces(self) -> int:
        return int(self.starts.size)

    @property
    def n_points(self) -> int:
        return int(self.t.size)

    @property
    def t_min(self) -> float:
        return float(self.t.min()) if self.t.size else 0.0

    @property
    def t_max(self) -> float:
        return float(self.t.max()) if self.t.size else 0.0

    @property
    def duration(self) -> float:
        """The longest single trace, which is how long a relative play lasts."""
        if self.t_end.size == 0:
            return 0.0
        return float(np.max(self.t_end - self.t_start))

    def window(self, t0: float, t1: float) -> TrackWindow:
        """The rows with ``t0 <= t <= t1``, grouped by trace and time-ordered.

        One ``searchsorted`` pair over the monotone key resolves the span in
        every trace at once; nothing here loops over traces.
        """
        if self.n_points == 0 or t1 < t0:
            empty_i = np.empty(0, dtype=np.int64)
            return TrackWindow(empty_i, np.empty(0, dtype=np.uint8),
                               np.zeros(self.n_traces, dtype=np.int64),
                               empty_i, empty_i)
        base = np.arange(self.n_traces, dtype=np.float64) * self.key_scale
        lo = np.searchsorted(self.key, base + float(t0), side="left")
        hi = np.searchsorted(self.key, base + float(t1), side="right")
        # ⚠ Clamp into each trace's own block. A query time beyond a trace's last
        # sample would otherwise let searchsorted run past it into the next
        # trace's rows, silently drawing a segment between two different tracks.
        block_end = self.starts + self.counts
        lo = np.clip(lo, self.starts, block_end)
        hi = np.clip(hi, self.starts, block_end)

        indices, counts = ragged_indices(lo, hi)
        connect = np.ones(indices.size, dtype=np.uint8)
        ends = np.cumsum(counts) - 1
        drawn = counts > 0
        if indices.size:
            connect[ends[drawn]] = 0          # lift the pen between traces
        if indices.size > 1:
            # A filter can remove rows from the middle of a trace.  Those rows
            # are not an invitation to draw a straight line over an unobserved
            # part of the trajectory: retain the discontinuities established
            # when the TrackSet was built.
            separated = (
                (indices[1:] != indices[:-1] + 1)
                | (self.segment_codes[indices[1:]]
                   != self.segment_codes[indices[:-1]])
            )
            connect[:-1][separated] = 0
        head_indices = indices[ends[drawn]] if indices.size else indices
        return TrackWindow(
            indices=indices,
            connect=connect,
            counts=counts,
            head_indices=head_indices,
            head_traces=np.flatnonzero(drawn).astype(np.int64),
        )


def build_track_set(
    coords_nm: np.ndarray,
    trace_ids: np.ndarray,
    times_s: np.ndarray,
    *,
    relative: bool = True,
    keep: np.ndarray | None = None,
) -> TrackSet:
    """Index *coords_nm* as trajectories on a playable time axis.

    With ``relative`` (the default) each trace starts at zero, so they play
    together; otherwise the recorded times are used and the traces keep their
    true offsets.  ``keep`` drops rows before indexing -- pass the filter mask
    here rather than gating later, so the per-trace spans are built from the rows
    that will actually be drawn.
    """
    coords = np.asarray(coords_nm, dtype=np.float64)
    tids = np.asarray(trace_ids).reshape(-1)
    times = np.asarray(times_s, dtype=np.float64).reshape(-1)
    if coords.ndim != 2 or coords.shape[1] < 2:
        raise ValueError("coordinates must be an N x 2 or N x 3 array.")
    if not (coords.shape[0] == tids.size == times.size):
        raise ValueError("coordinates, trace IDs and times must be row-aligned.")
    if coords.shape[1] == 2:
        coords = np.column_stack([coords, np.zeros(coords.shape[0])])

    rows = np.arange(tids.size, dtype=np.int64)
    usable = np.isfinite(times) & np.all(np.isfinite(coords[:, :3]), axis=1)
    # Rank every finite row inside its trace *before* the filter is applied.
    # A jump in these ranks after filtering is a real display discontinuity.
    finite_rows = rows[usable]
    source_rank = np.full(tids.size, -1, dtype=np.int64)
    if finite_rows.size:
        _all_unique, all_codes = np.unique(tids[finite_rows], return_inverse=True)
        all_time_order = np.lexsort((times[finite_rows], all_codes))
        ordered_rows = finite_rows[all_time_order]
        ordered_codes = all_codes[all_time_order]
        all_starts = np.flatnonzero(
            np.r_[True, ordered_codes[1:] != ordered_codes[:-1]])
        all_counts = np.diff(np.r_[all_starts, ordered_rows.size])
        source_rank[ordered_rows] = (
            np.arange(ordered_rows.size, dtype=np.int64)
            - np.repeat(all_starts, all_counts)
        )
    if keep is not None:
        keep_mask = np.asarray(keep, dtype=bool).reshape(-1)
        if keep_mask.size == usable.size:
            usable &= keep_mask
    rows = rows[usable]
    if rows.size == 0:
        raise ValueError("No finite, timed localizations to build tracks from.")

    unique, codes = np.unique(tids[rows], return_inverse=True)
    codes = codes.astype(np.int64, copy=False)
    t = times[rows]
    order_local = np.lexsort((t, codes))
    rows = rows[order_local]
    codes = codes[order_local]
    t = t[order_local]
    coords = coords[rows, :3]
    ordered_source_rank = source_rank[rows]

    starts = np.flatnonzero(np.r_[True, codes[1:] != codes[:-1]]).astype(np.int64)
    counts = np.diff(np.r_[starts, codes.size]).astype(np.int64)
    if relative:
        # Every trace begins at zero. The onset is the earliest *sorted* time,
        # which is the block's first row by construction.
        t = t - np.repeat(t[starts], counts)
    t_start = t[starts]
    t_end = t[starts + counts - 1]

    break_before = np.r_[
        True,
        (codes[1:] != codes[:-1])
        | (ordered_source_rank[1:] != ordered_source_rank[:-1] + 1),
    ]
    segment_codes = np.cumsum(break_before, dtype=np.int64) - 1

    # A strictly increasing key across the whole array, so one searchsorted pair
    # answers the window in every trace. The scale has to exceed any in-trace
    # time, or two traces' keys would interleave and the spans would run together.
    span = float(t.max() - t.min()) if t.size else 0.0
    key_scale = max(span, 1.0) * 4.0 + 1.0
    key = codes.astype(np.float64) * key_scale + t

    return TrackSet(
        coords=np.ascontiguousarray(coords),
        t=np.ascontiguousarray(t),
        trace_codes=codes,
        segment_codes=segment_codes,
        order=rows,
        trace_ids=unique,
        starts=starts,
        counts=counts,
        t_start=t_start,
        t_end=t_end,
        key=key,
        key_scale=key_scale,
        relative=bool(relative),
    )


def band_edges(t: float, tail_s: float | None, t_floor: float, bands: int) -> list:
    """``[(t0, t1, age)]`` for the comet's age bands, oldest first.

    ``age`` runs 0 at the head to 1 at the tip, and is what the caller turns into
    an opacity.  A ``tail_s`` of ``None`` is the cumulative trail: one band from
    the trace's start, because a fade over an unbounded length says nothing.
    """
    bands = max(int(bands), 1)
    if tail_s is None or tail_s <= 0.0:
        return [(t_floor, t, 0.0)]
    start = max(t - float(tail_s), t_floor)
    if t <= start:
        return [(start, t, 0.0)]
    edges = np.linspace(start, t, bands + 1)
    out = []
    for i in range(bands):
        # Oldest band first so the head is painted last and stays on top.
        age = 1.0 - (i + 0.5) / bands
        out.append((float(edges[i]), float(edges[i + 1]), float(age)))
    return out


def trace_displacements(tracks: TrackSet) -> np.ndarray:
    """Straight-line start-to-end distance of every trace, in nm.

    The discriminator between a *tracking* channel and a *structure* channel: a
    molecule walking over a scaffold covers tens to hundreds of nm, while a
    blinking fluorophore on the scaffold stays inside its own localization
    precision.  Measured on the reference tracking file: median 104 nm.
    """
    if tracks.n_traces == 0:
        return np.empty(0, dtype=float)
    first = tracks.coords[tracks.starts]
    last = tracks.coords[tracks.starts + tracks.counts - 1]
    return np.linalg.norm(last - first, axis=1)


#: A channel whose median trace travels at least this far is treated as tracking.
#: Well above any localization precision and well below a real excursion, so the
#: two populations are not near it: the reference file reads 104 nm and a
#: structure channel reads single-digit nm.
TRACKING_DISPLACEMENT_NM = 25.0

#: ...and it needs traces long enough to be a trajectory rather than a blink.
TRACKING_MIN_MEDIAN_LOCS = 5


def looks_like_tracking(
    tracks: TrackSet,
    *,
    displacement_nm: float = TRACKING_DISPLACEMENT_NM,
    min_median_locs: int = TRACKING_MIN_MEDIAN_LOCS,
) -> tuple[bool, str]:
    """Whether *tracks* reads as a moving species, and the reason either way.

    Returned as a reason, not just a verdict, because this only ever chooses a
    *default*: the Tracking View lets the role be set per channel, and a wrong
    guess that says what it measured is correctable, while a silent one is not.
    """
    if tracks.n_traces == 0:
        return False, "no traces"
    displacement = float(np.median(trace_displacements(tracks)))
    median_locs = float(np.median(tracks.counts))
    displacement_threshold = max(float(displacement_nm), 0.0)
    minimum_locs = max(int(min_median_locs), 2)
    if median_locs < minimum_locs:
        return False, (
            f"traces are {median_locs:.0f} localizations long on average, too "
            f"short for the {minimum_locs}-localization tracking threshold"
        )
    if displacement < displacement_threshold:
        return False, (
            f"traces move {displacement:.1f} nm end to end, below the "
            f"{displacement_threshold:g} nm tracking threshold and consistent "
            "with standing still"
        )
    return True, f"traces move {displacement:.0f} nm end to end over {median_locs:.0f} localizations"


def dataset_tracking_role(dataset) -> tuple[str | None, str, bool]:
    """Return ``(role, reason, explicit)`` recorded by the Tracking View.

    A manual role is persistent dataset state.  The automatic role and its
    diagnostic are cached separately so analyses can explain why they refuse a
    static-emitter method without pretending the heuristic was a user choice.
    Datasets never opened in the Tracking View simply return ``None``.
    """
    state = getattr(dataset, "state", {})
    if not isinstance(state, dict):
        return None, "", False
    manual = state.get("tracking_role")
    if manual in {"Tracking", "Structure"}:
        return str(manual), "user-selected Tracking View role", True
    automatic = state.get("tracking_role_auto")
    if automatic in {"Tracking", "Structure"}:
        return (
            str(automatic),
            str(state.get("tracking_role_auto_reason") or "automatic role classification"),
            False,
        )
    return None, "", False
