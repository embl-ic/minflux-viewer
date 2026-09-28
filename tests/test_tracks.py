"""The comet's data structure: windowing, trace separation and role detection."""

from __future__ import annotations

import numpy as np
import pytest

from minflux_viewer.core.tracks import (
    TRACKING_DISPLACEMENT_NM,
    band_edges,
    build_track_set,
    looks_like_tracking,
    ragged_indices,
    trace_displacements,
)


def _two_tracks():
    """Two traces that start half a second apart and walk in opposite directions."""
    per = 10
    t = np.arange(per) * 1.0e-3
    coords = np.zeros((2 * per, 3))
    coords[:per, 0] = np.arange(per) * 10.0          # trace 1 walks +x
    coords[per:, 1] = np.arange(per) * -10.0         # trace 2 walks -y
    tid = np.repeat([7, 3], per)                     # deliberately not sorted
    times = np.concatenate([t, t + 0.5])
    return coords, tid, times


def test_ragged_indices_expands_spans_without_a_loop() -> None:
    idx, counts = ragged_indices([2, 10, 5], [5, 10, 7])
    assert counts.tolist() == [3, 0, 2]
    assert idx.tolist() == [2, 3, 4, 5, 6]


def test_ragged_indices_handles_an_all_empty_selection() -> None:
    idx, counts = ragged_indices([4, 9], [4, 9])
    assert idx.size == 0
    assert counts.tolist() == [0, 0]


def test_each_trace_starts_at_zero_and_keeps_its_own_order() -> None:
    coords, tid, times = _two_tracks()
    tracks = build_track_set(coords, tid, times)

    assert tracks.n_traces == 2
    assert tracks.t_start.tolist() == pytest.approx([0.0, 0.0])
    # Rows are grouped by trace and ordered in time inside each group.
    for start, count in zip(tracks.starts, tracks.counts):
        block = tracks.t[start:start + count]
        assert np.all(np.diff(block) >= 0)
        assert len(set(tracks.trace_codes[start:start + count].tolist())) == 1


def test_absolute_time_keeps_the_offset_between_traces() -> None:
    coords, tid, times = _two_tracks()
    tracks = build_track_set(coords, tid, times, relative=False)
    # One trace begins half a second after the other, and that survives.
    assert sorted(tracks.t_start.tolist()) == pytest.approx([0.0, 0.5])


def test_a_window_never_joins_two_traces() -> None:
    """The failure this guards against draws a line between two molecules.

    ``searchsorted`` over the shared key would happily run past the end of one
    trace's block into the next one's rows; the clamp is what stops it, and a
    segment spanning two traces is a line that was never measured.
    """
    coords, tid, times = _two_tracks()
    tracks = build_track_set(coords, tid, times)
    window = tracks.window(0.0, 1.0)            # far past the end of both

    assert window.size == coords.shape[0]
    codes = tracks.trace_codes[window.indices]
    joins = np.flatnonzero(window.connect[:-1])
    assert not np.any(codes[joins] != codes[joins + 1])


def test_a_window_holds_exactly_the_rows_in_it_and_heads_are_the_latest() -> None:
    coords, tid, times = _two_tracks()
    tracks = build_track_set(coords, tid, times)
    window = tracks.window(0.002, 0.005)

    picked = tracks.t[window.indices]
    assert picked.min() >= 0.002 - 1e-12
    assert picked.max() <= 0.005 + 1e-12
    assert window.n_traces == 2
    # Each head is that trace's last point inside the window.
    for head, code in zip(window.head_indices, tracks.trace_codes[window.head_indices]):
        same = window.indices[tracks.trace_codes[window.indices] == code]
        assert tracks.t[head] == pytest.approx(tracks.t[same].max())


def test_an_empty_window_is_empty_rather_than_an_error() -> None:
    coords, tid, times = _two_tracks()
    tracks = build_track_set(coords, tid, times)
    window = tracks.window(5.0, 6.0)
    assert window.size == 0 and window.n_traces == 0


def test_band_edges_run_oldest_first_and_cover_the_tail() -> None:
    bands = band_edges(1.0, 0.4, 0.0, 4)
    assert len(bands) == 4
    assert bands[0][0] == pytest.approx(0.6)          # the tip
    assert bands[-1][1] == pytest.approx(1.0)         # the head
    assert bands[0][2] > bands[-1][2]                 # oldest is the most faded
    # Contiguous, so no part of the tail is missed between two bands.
    for earlier, later in zip(bands, bands[1:]):
        assert earlier[1] == pytest.approx(later[0])


def test_a_cumulative_tail_is_one_band_from_the_trace_start() -> None:
    """A fade spread over an unbounded length says nothing, so it is not drawn."""
    bands = band_edges(1.0, None, 0.0, 6)
    assert bands == [(0.0, 1.0, 0.0)]


def test_the_tail_is_clamped_at_the_axis_floor() -> None:
    bands = band_edges(0.1, 1.0, 0.0, 3)
    assert bands[0][0] == pytest.approx(0.0)


def test_a_moving_species_and_a_standing_one_are_told_apart() -> None:
    rng = np.random.default_rng(0)
    per, n = 40, 12

    walk = np.cumsum(rng.normal(0, 12.0, (n * per, 3)), axis=0)
    tid = np.repeat(np.arange(n), per)
    t = np.tile(np.arange(per) * 1e-3, n)
    moving = build_track_set(walk, tid, t)
    verdict, reason = looks_like_tracking(moving)
    assert verdict and "move" in reason
    assert float(np.median(trace_displacements(moving))) > TRACKING_DISPLACEMENT_NM

    # A scaffold: every trace is a blinking fluorophore inside its own precision.
    centres = np.repeat(rng.uniform(-2000, 2000, (n, 3)), per, axis=0)
    still = build_track_set(centres + rng.normal(0, 4.0, centres.shape), tid, t)
    verdict, reason = looks_like_tracking(still)
    assert not verdict and "standing still" in reason


def test_a_short_blink_is_not_called_a_trajectory() -> None:
    """Two or three localizations far apart are noise, not a track."""
    rng = np.random.default_rng(1)
    coords = rng.uniform(-500, 500, (30, 3))
    tid = np.repeat(np.arange(15), 2)
    t = np.tile([0.0, 1e-3], 15)
    verdict, reason = looks_like_tracking(build_track_set(coords, tid, t))
    assert not verdict and "too short" in reason


def test_rows_with_no_time_or_no_position_are_dropped_before_indexing() -> None:
    coords = np.zeros((6, 3))
    coords[2, 0] = np.nan
    tid = np.array([1, 1, 1, 2, 2, 2])
    t = np.array([0.0, 1.0, 2.0, 0.0, np.nan, 2.0])
    tracks = build_track_set(coords, tid, t)
    assert tracks.n_points == 4
    assert tracks.n_traces == 2


def test_keep_narrows_the_tracks_to_the_filtered_rows() -> None:
    coords, tid, times = _two_tracks()
    keep = np.zeros(coords.shape[0], dtype=bool)
    keep[:10] = True
    tracks = build_track_set(coords, tid, times, keep=keep)
    assert tracks.n_traces == 1
    assert tracks.n_points == 10


def test_keep_does_not_draw_across_rows_removed_inside_a_trace() -> None:
    coords = np.zeros((6, 3))
    coords[:, 0] = [0, 1, 2, 1000, 1001, 1002]
    tid = np.ones(6, dtype=int)
    times = np.arange(6, dtype=float)
    keep = np.array([True, True, False, True, True, True])
    tracks = build_track_set(coords, tid, times, relative=False, keep=keep)
    window = tracks.window(0.0, 5.0)

    selected_rows = tracks.order[window.indices]
    before_gap = int(np.flatnonzero(selected_rows == 1)[0])
    assert selected_rows[before_gap + 1] == 3
    assert window.connect[before_gap] == 0


def test_windowing_is_cheap_enough_to_play() -> None:
    """The contract that makes the view playable: no ``O(N)`` pass per frame.

    Measured on the reference file (230 traces, 45,105 points) at 0.20 ms per
    frame. This asserts the scaling rather than a wall-clock figure: ten times
    the points in the same number of traces must not cost ten times as much.
    """
    import time

    def build(per: int):
        n_traces = 200
        rng = np.random.default_rng(0)
        tid = np.repeat(np.arange(n_traces), per)
        t = np.tile(np.arange(per) * 1e-3, n_traces)
        xyz = np.cumsum(rng.normal(0, 8, (n_traces * per, 3)), axis=0)
        return build_track_set(xyz, tid, t)

    def cost(tracks, reps=40):
        start = time.perf_counter()
        for k in range(reps):
            tracks.window(0.0, 0.02 * (k + 1) / reps)
        return (time.perf_counter() - start) / reps

    small, large = build(50), build(500)
    assert large.n_points == 10 * small.n_points
    # Generous: the point is that it is nothing like linear in the point count.
    assert cost(large) < cost(small) * 4.0
