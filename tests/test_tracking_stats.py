"""Pure trajectory-analysis foundations and their scientific guardrails."""

from __future__ import annotations

import numpy as np
import pytest

from minflux_viewer.analysis.tracking_stats import (
    TrackingMethodSpec,
    TrackingResult,
    get_tracking_method,
    kinematic_statistics,
    mean_squared_displacement,
    prepare_trajectories,
    register_tracking_method,
    run_tracking_method,
    tracking_methods,
    trajectories_from_dataset,
)


def test_rounding_and_trace_onset_happen_before_filtering() -> None:
    coords = np.column_stack([np.arange(5.0), np.zeros((5, 2))])
    tid = np.ones(5, dtype=int)
    tim = 10.0 + np.arange(5) * 1.0000002e-3
    keep = np.array([False, True, False, True, True])

    data = prepare_trajectories(
        coords, tid, tim, keep=keep, timestamp_precision_s=1e-6)

    assert data.row_indices.tolist() == [1, 3, 4]
    # The first retained point is not silently redefined as the trace onset.
    assert data.trace_time[0] == pytest.approx(1e-3)
    # Removing source row 2 creates a segment boundary before row 3.
    assert data.segment_codes.tolist() == [0, 1, 1]
    assert data.diagnostics.filter_boundaries == 1


def test_nonpositive_time_and_long_gaps_split_segments() -> None:
    coords = np.column_stack([np.arange(6.0), np.zeros((6, 2))])
    tid = np.ones(6, dtype=int)
    tim = np.array([0.0, 0.001, 0.001, 0.002, 0.003, 0.020])
    data = prepare_trajectories(
        coords, tid, tim, timestamp_precision_s=1e-6, max_gap_s=0.005)

    assert data.n_segments == 3
    assert data.diagnostics.nonpositive_time_boundaries == 1
    assert data.diagnostics.long_gap_boundaries == 1


def test_index_mode_is_explicitly_nonphysical() -> None:
    coords = np.zeros((6, 3))
    tid = np.repeat([1, 2], 3)
    data = prepare_trajectories(coords, tid)
    result = mean_squared_displacement(
        data, min_segment_points=2, min_pairs=1, max_lag_points=2)

    assert data.time_source == "index" and data.time_unit == "index"
    assert result.units["pooled_fit"]["diffusion"] == "nm²/index"
    assert any("no physical time unit" in item for item in result.diagnostics.warnings)


def test_kinematics_never_bridges_a_filtered_gap() -> None:
    coords = np.zeros((6, 3))
    coords[:, 0] = [0, 1, 2, 1000, 1001, 1002]
    tid = np.ones(6, dtype=int)
    tim = np.arange(6, dtype=float)
    keep = np.array([True, True, False, True, True, True])
    data = prepare_trajectories(coords, tid, tim, keep=keep, gap_factor=None)
    result = kinematic_statistics(data)
    local = result.table("localization")

    # Source row 3 starts a new segment: the unmeasured 999 nm bridge is absent.
    row3 = int(np.flatnonzero(local["row_index"] == 3)[0])
    assert np.isnan(local["jump_distance_nm"][row3])
    np.testing.assert_allclose(
        local["jump_distance_nm"][np.isfinite(local["jump_distance_nm"])], 1.0)
    assert result.table("segment")["straightness"].tolist() == pytest.approx([1.0, 1.0])


def test_irregular_time_msd_uses_measured_lags_without_interpolation() -> None:
    tim = np.array([0.0, 0.0010, 0.0021, 0.0031, 0.0042, 0.0052])
    coords = np.zeros((tim.size, 3))
    coords[:, 0] = np.arange(tim.size)
    data = prepare_trajectories(
        coords, np.ones(tim.size, int), tim,
        timestamp_precision_s=1e-4, gap_factor=None)
    result = mean_squared_displacement(
        data, lag_bin=1e-3, min_pairs=1, max_lag_points=3,
        min_segment_points=3)
    curve = result.table("pooled_time_averaged_msd")

    assert curve["lag"].size >= 3
    assert not np.allclose(curve["lag"], curve["nominal_lag"])
    assert result.provenance["interpolation"] == "none"


def test_linear_motion_has_quadratic_msd_and_alpha_near_two() -> None:
    n = 30
    tim = np.arange(n, dtype=float) * 0.01
    coords = np.zeros((n, 3))
    coords[:, 0] = np.arange(n) * 2.0
    data = prepare_trajectories(coords, np.ones(n, int), tim, gap_factor=None)
    result = mean_squared_displacement(
        data, min_pairs=1, max_lag_points=6, fit_lags=5)
    curve = result.table("pooled_time_averaged_msd")
    fit = result.table("pooled_fit")

    np.testing.assert_allclose(curve["msd_nm2"][:3], [4.0, 16.0, 36.0])
    assert fit["alpha"][0] == pytest.approx(2.0, abs=0.03)


def test_brownian_fit_recovers_a_known_diffusion_scale() -> None:
    rng = np.random.default_rng(4)
    diffusion_nm2_s = 4.0e5
    dt = 1.0e-3
    traces, per = 120, 80
    increments = rng.normal(
        0.0, np.sqrt(2.0 * diffusion_nm2_s * dt), (traces, per - 1, 2))
    xy = np.concatenate(
        [np.zeros((traces, 1, 2)), np.cumsum(increments, axis=1)], axis=1)
    coords = np.zeros((traces * per, 3))
    coords[:, :2] = xy.reshape(-1, 2)
    tid = np.repeat(np.arange(traces), per)
    tim = np.tile(np.arange(per) * dt, traces)
    data = prepare_trajectories(coords, tid, tim, gap_factor=None)
    result = mean_squared_displacement(
        data, dimensions=2, min_pairs=3, max_lag_points=5, fit_lags=4)
    estimate = float(result.table("pooled_fit")["diffusion"][0])

    assert estimate == pytest.approx(diffusion_nm2_s, rel=0.18)
    assert result.diagnostics.status == "ok"


def test_dataset_adapter_distinguishes_filtered_and_unfiltered_scope() -> None:
    from minflux_viewer.core.dataset import build_localization_dataset

    ds = build_localization_dataset(
        name="track", x_nm=np.arange(5.0), y_nm=np.zeros(5), z_nm=np.zeros(5),
        tid=np.ones(5, int), tim=np.arange(5) * 1e-3)
    ds.filter_mask = np.array([True, True, False, True, True])

    filtered = trajectories_from_dataset(ds, filtered=True)
    unfiltered = trajectories_from_dataset(ds, filtered=False)
    assert filtered.n_points == 4 and filtered.provenance["source_scope"] == "filtered"
    assert unfiltered.n_points == 5
    assert filtered.provenance["coordinate_space"].startswith("calibrated dataset nm")


def test_tracking_method_registry_is_extensible_and_collision_safe() -> None:
    builtins = {item.method_id for item in tracking_methods()}
    assert {"minflux_viewer.tracking.kinematics", "minflux_viewer.tracking.msd"} <= builtins
    assert get_tracking_method("minflux_viewer.tracking.msd").citations

    data = prepare_trajectories(
        np.zeros((3, 3)), np.ones(3, int), np.arange(3, dtype=float))
    result = run_tracking_method("minflux_viewer.tracking.kinematics", data)
    assert isinstance(result, TrackingResult)

    duplicate = TrackingMethodSpec(
        method_id="minflux_viewer.tracking.kinematics", label="duplicate",
        version="1", description="", runner=kinematic_statistics)
    with pytest.raises(ValueError, match="already registered"):
        register_tracking_method(duplicate)
