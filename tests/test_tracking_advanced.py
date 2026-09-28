"""Advanced tracking methods keep their assumptions explicit."""

from __future__ import annotations

import numpy as np
import pytest

from minflux_viewer.analysis import (
    TrackingResult,
    TrackingStateAdapterSpec,
    confinement_index,
    ergodicity_breaking,
    fit_jump_distance_mixture,
    get_tracking_method,
    jump_distance_distribution,
    prepare_trajectories,
    register_tracking_state_adapter,
    run_tracking_state_adapter,
)


def _brownian_data(
    *,
    diffusion: float = 3.0e5,
    traces: int = 80,
    points: int = 70,
    dt: float = 1.0e-3,
    seed: int = 5,
):
    rng = np.random.default_rng(seed)
    steps = rng.normal(
        0.0, np.sqrt(2.0 * diffusion * dt), (traces, points - 1, 2))
    xy = np.concatenate(
        [np.zeros((traces, 1, 2)), np.cumsum(steps, axis=1)], axis=1)
    coords = np.zeros((traces * points, 3))
    coords[:, :2] = xy.reshape(-1, 2)
    tid = np.repeat(np.arange(traces), points)
    tim = np.tile(np.arange(points, dtype=float) * dt, traces)
    return prepare_trajectories(coords, tid, tim, gap_factor=None)


def test_jump_distribution_preserves_measured_intervals_and_has_no_model() -> None:
    data = _brownian_data(traces=3, points=8)
    result = jump_distance_distribution(data, bins=6)

    assert result.table("jump")["dt"].size == 21
    assert result.table("histogram")["count"].sum() == 21
    assert result.provenance["model"] == "none"
    assert result.citations


def test_one_component_jump_fit_recovers_brownian_diffusion() -> None:
    expected = 4.0e5
    data = _brownian_data(diffusion=expected)
    result = fit_jump_distance_mixture(data, components=1)

    estimate = float(result.table("component")["diffusion"][0])
    assert estimate == pytest.approx(expected, rel=0.12)
    assert result.provenance["model_selection"] == "not automatic"
    assert result.table("model")["converged"][0]


def test_insufficient_jump_fit_keeps_every_result_table_row_aligned() -> None:
    data = _brownian_data(traces=1, points=5)
    result = fit_jump_distance_mixture(data, components=1, min_jumps=6)

    jump = result.table("jump")
    assert result.diagnostics.status == "insufficient-data"
    assert {np.asarray(column).size for column in jump.values()} == {4}
    assert np.all(jump["most_likely_component"] == -1)


def test_two_component_jump_fit_resolves_separated_populations() -> None:
    slow = _brownian_data(diffusion=5.0e4, traces=45, seed=8)
    fast = _brownian_data(diffusion=8.0e5, traces=45, seed=9)
    coords = np.vstack([slow.coords_nm, fast.coords_nm])
    tid = np.concatenate([slow.trace_ids, fast.trace_ids + 1000])
    tim = np.concatenate([slow.absolute_time, fast.absolute_time])
    data = prepare_trajectories(coords, tid, tim, gap_factor=None)

    result = fit_jump_distance_mixture(data, components=2)
    estimates = result.table("component")["diffusion"]
    assert estimates[0] == pytest.approx(5.0e4, rel=0.35)
    assert estimates[1] == pytest.approx(8.0e5, rel=0.25)
    assert np.sum(result.table("component")["weight"]) == pytest.approx(1.0)


def test_confinement_index_needs_an_explicit_diffusion_and_is_gap_safe() -> None:
    points = np.zeros((12, 3))
    points[:, 0] = np.sin(np.linspace(0, 2 * np.pi, 12))
    data = prepare_trajectories(
        points, np.ones(12, int), np.arange(12) * 0.01, gap_factor=None)

    result = confinement_index(data, diffusion=5.0e4, window_points=5)
    assert result.table("window")["confinement_index"].size == 8
    assert np.nanmax(result.table("localization")["confinement_index"]) > 0
    assert result.provenance["diffusion_source"] == "explicit caller input"
    with pytest.raises(ValueError, match="2-D"):
        confinement_index(data, diffusion=5.0e4, dimensions=3)


def test_ergodicity_result_is_the_formal_tamsd_scatter_statistic() -> None:
    data = _brownian_data(traces=30, points=60)
    result = ergodicity_breaking(
        data, min_segments=10, min_pairs_per_segment=2)

    ensemble = result.table("ensemble")
    segment = result.table("segment_tamsd")
    assert ensemble["ergodicity_breaking"].size > 0
    assert np.all(ensemble["ergodicity_breaking"] >= -1e-12)
    for lag in np.unique(segment["nominal_lag"]):
        selected = segment["nominal_lag"] == lag
        assert np.mean(segment["normalized_amplitude"][selected]) == pytest.approx(1.0)
    assert any("finite-time" in text for text in result.diagnostics.warnings)


def test_advanced_methods_are_registered_without_choosing_a_state_model() -> None:
    for method_id in (
        "minflux_viewer.tracking.jump_distribution",
        "minflux_viewer.tracking.jump_mixture",
        "minflux_viewer.tracking.confinement_index",
        "minflux_viewer.tracking.ergodicity_breaking",
    ):
        assert get_tracking_method(method_id).citations


def test_optional_state_backend_uses_a_versioned_adapter_contract() -> None:
    data = _brownian_data(traces=2, points=8)

    def runner(prepared, **_parameters):
        raw = jump_distance_distribution(prepared, bins=3)
        return TrackingResult(
            method_id="test.state", method_version="2.1", tables=raw.tables,
            units=raw.units, diagnostics=raw.diagnostics,
            provenance={"backend": "test", "model_version": "fixture-4"})

    adapter = TrackingStateAdapterSpec(
        adapter_id="tests.fixture.state", label="Fixture", version="2.1",
        description="test", runner=runner)
    register_tracking_state_adapter(adapter, replace=True)
    result = run_tracking_state_adapter(adapter.adapter_id, data)

    assert result.method_version == "2.1"
    assert result.provenance["model_version"] == "fixture-4"
