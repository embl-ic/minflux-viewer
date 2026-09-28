"""Validated advanced trajectory statistics built on ``tracking_stats``.

The functions in this module are deliberately separate analyses.  An empirical
jump distribution does not silently select a mixture model, the Simson
confinement index requires an explicit free-diffusion coefficient, and the
finite-time ergodicity-breaking statistic is not promoted to a motion-state
label.
"""

from __future__ import annotations

import numpy as np
from scipy.optimize import minimize_scalar
from scipy.special import logsumexp

from .tracking_stats import (
    PreparedTrajectories,
    TrackingMethodSpec,
    TrackingResult,
    _base_diagnostics,
    register_tracking_method,
)

SCHUTZ_1997_DOI = "https://doi.org/10.1016/S0006-3495(97)78139-6"
SIMSON_1995_DOI = "https://doi.org/10.1016/S0006-3495(95)79972-6"
BUROV_2011_DOI = "https://doi.org/10.1039/C0CP01879A"


def _validate_dimensions(dimensions: int) -> int:
    value = int(dimensions)
    if value not in (1, 2, 3):
        raise ValueError("dimensions must be 1, 2 or 3.")
    return value


def _adjacent_steps(
    data: PreparedTrajectories,
    dimensions: int,
) -> dict[str, np.ndarray]:
    """Return gap-safe adjacent displacements from explicit segments."""
    parts: dict[str, list[np.ndarray]] = {
        "row_from": [],
        "row_to": [],
        "trace_id": [],
        "segment_id": [],
        "dt": [],
        "squared_distance_nm2": [],
        "jump_distance_nm": [],
    }
    for segment_id, (start, count) in enumerate(
        zip(data.segment_starts, data.segment_counts)
    ):
        end = int(start + count)
        if count < 2:
            continue
        dt = np.diff(data.absolute_time[start:end])
        delta = np.diff(data.coords_nm[start:end, :dimensions], axis=0)
        squared = np.sum(delta * delta, axis=1)
        valid = np.isfinite(dt) & (dt > 0) & np.isfinite(squared)
        if not np.any(valid):
            continue
        left = np.arange(start, end - 1, dtype=np.int64)[valid]
        right = left + 1
        n_valid = int(np.count_nonzero(valid))
        parts["row_from"].append(data.row_indices[left])
        parts["row_to"].append(data.row_indices[right])
        parts["trace_id"].append(np.repeat(data.trace_ids[start], n_valid))
        parts["segment_id"].append(np.full(n_valid, segment_id, dtype=np.int64))
        parts["dt"].append(dt[valid])
        parts["squared_distance_nm2"].append(squared[valid])
        parts["jump_distance_nm"].append(np.sqrt(squared[valid]))
    arrays: dict[str, np.ndarray] = {}
    for key, values in parts.items():
        dtype = np.int64 if key in {"row_from", "row_to", "segment_id"} else float
        arrays[key] = np.concatenate(values) if values else np.empty(0, dtype=dtype)
    return arrays


def jump_distance_distribution(
    data: PreparedTrajectories,
    *,
    dimensions: int = 2,
    bins: int | str = "auto",
) -> TrackingResult:
    """Return measured adjacent jumps and an empirical radial histogram.

    No diffusion components are inferred here.  Keeping the observations and
    model fitting separate makes it possible to inspect a distribution before
    imposing a Brownian mixture on it.
    """
    dimensions = _validate_dimensions(dimensions)
    jumps = _adjacent_steps(data, dimensions)
    distances = jumps["jump_distance_nm"]
    if isinstance(bins, str):
        if bins != "auto":
            raise ValueError("bins must be a positive integer or 'auto'.")
        bin_spec: int | str = "auto"
    else:
        bin_spec = int(bins)
        if bin_spec < 1:
            raise ValueError("bins must be a positive integer or 'auto'.")
    if distances.size:
        counts, edges = np.histogram(distances, bins=bin_spec)
        widths = np.diff(edges)
        density = counts / (distances.size * widths)
    else:
        counts = np.empty(0, dtype=np.int64)
        edges = np.empty(0, dtype=float)
        density = np.empty(0, dtype=float)
    histogram = {
        "bin_left_nm": edges[:-1].copy(),
        "bin_right_nm": edges[1:].copy(),
        "bin_center_nm": ((edges[:-1] + edges[1:]) * 0.5),
        "count": counts.astype(np.int64, copy=False),
        "density_per_nm": density,
    }
    time_unit = data.time_unit
    return TrackingResult(
        method_id="minflux_viewer.tracking.jump_distribution",
        method_version="1.0",
        tables={"jump": jumps, "histogram": histogram},
        units={
            "jump": {
                "dt": time_unit,
                "squared_distance_nm2": "nm²",
                "jump_distance_nm": "nm",
            },
            "histogram": {
                "bin_left_nm": "nm",
                "bin_right_nm": "nm",
                "bin_center_nm": "nm",
                "density_per_nm": "nm⁻¹",
            },
        },
        diagnostics=_base_diagnostics(
            data,
            status="ok" if distances.size else "insufficient-data",
            excluded={"segments_without_steps": int(np.count_nonzero(
                data.segment_counts < 2))},
        ),
        provenance={
            **data.provenance,
            "dimensions": dimensions,
            "lag": "adjacent measured localizations within explicit segments",
            "histogram_bins": bins,
            "model": "none",
        },
        citations=(SCHUTZ_1997_DOI,),
    )


def _mixture_log_terms(
    squared: np.ndarray,
    dt: np.ndarray,
    diffusion: np.ndarray,
    weights: np.ndarray,
    *,
    dimensions: int,
    localization_sigma_nm: float,
) -> np.ndarray:
    variance = (
        2.0 * dt[:, None] * diffusion[None, :]
        + 2.0 * float(localization_sigma_nm) ** 2
    )
    return (
        np.log(np.maximum(weights, np.finfo(float).tiny))[None, :]
        - 0.5 * dimensions * np.log(2.0 * np.pi * variance)
        - 0.5 * squared[:, None] / variance
    )


def fit_jump_distance_mixture(
    data: PreparedTrajectories,
    *,
    dimensions: int = 2,
    components: int = 1,
    localization_sigma_nm: float = 0.0,
    min_jumps: int = 20,
    max_iterations: int = 200,
    tolerance: float = 1.0e-7,
) -> TrackingResult:
    """Fit a fixed-size isotropic Brownian mixture to adjacent displacements.

    Each measured interval enters its own variance,
    ``2 * D_k * dt + 2 * sigma²`` per coordinate, so irregular timestamp data
    are not resampled.  ``components`` is a caller choice: the function reports
    AIC/BIC but does not automatically select or biologically name a model.
    """
    dimensions = _validate_dimensions(dimensions)
    components = int(components)
    min_jumps = int(min_jumps)
    if components < 1 or components > 6:
        raise ValueError("components must be between 1 and 6.")
    if min_jumps < max(2, components * 3):
        raise ValueError("min_jumps is too small for the requested mixture.")
    sigma = float(localization_sigma_nm)
    if not np.isfinite(sigma) or sigma < 0:
        raise ValueError("localization_sigma_nm must be finite and non-negative.")
    if max_iterations < 1 or not np.isfinite(tolerance) or tolerance <= 0:
        raise ValueError("max_iterations and tolerance must be positive.")

    jumps = _adjacent_steps(data, dimensions)
    dt = np.asarray(jumps["dt"], dtype=float)
    squared = np.asarray(jumps["squared_distance_nm2"], dtype=float)
    diffusion_unit = "nm²/s" if data.time_unit == "s" else "nm²/index"
    if squared.size < min_jumps:
        empty = np.empty(0, dtype=float)
        return TrackingResult(
            method_id="minflux_viewer.tracking.jump_mixture",
            method_version="1.0",
            tables={
                "jump": {
                    **jumps,
                    "most_likely_component": np.full(
                        squared.size, -1, dtype=np.int64),
                    "component_probability": np.full(
                        squared.size, np.nan, dtype=float),
                },
                "component": {"component": np.empty(0, dtype=int),
                              "weight": empty.copy(), "diffusion": empty.copy()},
                "model": {"components": np.asarray([components]),
                          "n_jumps": np.asarray([squared.size]),
                          "log_likelihood": np.asarray([np.nan]),
                          "aic": np.asarray([np.nan]), "bic": np.asarray([np.nan]),
                          "converged": np.asarray([False])},
            },
            units={"jump": {"dt": data.time_unit, "jump_distance_nm": "nm"},
                   "component": {"diffusion": diffusion_unit}, "model": {}},
            diagnostics=_base_diagnostics(
                data,
                status="insufficient-data",
                excluded={"available_jumps": int(squared.size)},
                warnings=(f"At least {min_jumps} jumps are required for this fit.",),
            ),
            provenance={**data.provenance, "components": components},
            citations=(SCHUTZ_1997_DOI,),
        )

    apparent = squared / (2.0 * dimensions * dt)
    positive = apparent[np.isfinite(apparent) & (apparent > 0)]
    if not positive.size:
        positive = np.asarray([np.finfo(float).eps])
    quantiles = np.linspace(0.15, 0.85, components)
    diffusion = np.maximum(np.quantile(positive, quantiles), np.finfo(float).eps)
    weights = np.full(components, 1.0 / components)
    previous = -np.inf
    converged = False

    for iteration in range(1, int(max_iterations) + 1):
        terms = _mixture_log_terms(
            squared, dt, diffusion, weights,
            dimensions=dimensions, localization_sigma_nm=sigma)
        normalizer = logsumexp(terms, axis=1)
        responsibilities = np.exp(terms - normalizer[:, None])
        totals = np.sum(responsibilities, axis=0)
        weights = np.maximum(totals / squared.size, np.finfo(float).eps)
        weights /= np.sum(weights)
        for component in range(components):
            responsibility = responsibilities[:, component]
            if sigma == 0.0:
                numerator = float(np.sum(responsibility * squared))
                denominator = float(2.0 * dimensions * np.sum(responsibility * dt))
                diffusion[component] = max(
                    numerator / denominator, np.finfo(float).eps)
                continue

            def objective(log_diffusion: float) -> float:
                value = float(np.exp(log_diffusion))
                variance = 2.0 * value * dt + 2.0 * sigma ** 2
                return float(np.sum(responsibility * (
                    0.5 * dimensions * np.log(variance)
                    + 0.5 * squared / variance)))

            centre = float(np.log(max(diffusion[component], np.finfo(float).eps)))
            optimum = minimize_scalar(
                objective, bounds=(centre - 12.0, centre + 12.0), method="bounded")
            diffusion[component] = float(np.exp(optimum.x))

        terms = _mixture_log_terms(
            squared, dt, diffusion, weights,
            dimensions=dimensions, localization_sigma_nm=sigma)
        likelihood = float(np.sum(logsumexp(terms, axis=1)))
        if np.isfinite(previous) and abs(likelihood - previous) <= (
            float(tolerance) * (1.0 + abs(previous))
        ):
            converged = True
            break
        previous = likelihood

    order = np.argsort(diffusion)
    diffusion = diffusion[order]
    weights = weights[order]
    terms = _mixture_log_terms(
        squared, dt, diffusion, weights,
        dimensions=dimensions, localization_sigma_nm=sigma)
    normalizer = logsumexp(terms, axis=1)
    responsibilities = np.exp(terms - normalizer[:, None])
    likelihood = float(np.sum(normalizer))
    assignment = np.argmax(responsibilities, axis=1)
    probability = responsibilities[np.arange(squared.size), assignment]
    parameter_count = 2 * components - 1
    aic = 2.0 * parameter_count - 2.0 * likelihood
    bic = np.log(squared.size) * parameter_count - 2.0 * likelihood
    jump_table = {
        **jumps,
        "most_likely_component": assignment.astype(np.int64),
        "component_probability": probability,
    }
    component_table = {
        "component": np.arange(components, dtype=np.int64),
        "weight": weights,
        "diffusion": diffusion,
    }
    warnings = (
        "Components are statistical Brownian-mixture components, not biological "
        "state labels; directed, confined, or correlated motion violates the model.",
        "AIC and BIC describe this fixed candidate fit. Compare explicitly requested "
        "candidate component counts; no model is selected automatically.",
    )
    return TrackingResult(
        method_id="minflux_viewer.tracking.jump_mixture",
        method_version="1.0",
        tables={
            "jump": jump_table,
            "component": component_table,
            "model": {
                "components": np.asarray([components]),
                "n_jumps": np.asarray([squared.size]),
                "log_likelihood": np.asarray([likelihood]),
                "aic": np.asarray([aic]),
                "bic": np.asarray([bic]),
                "iterations": np.asarray([iteration]),
                "converged": np.asarray([converged]),
            },
        },
        units={
            "jump": {
                "dt": data.time_unit,
                "squared_distance_nm2": "nm²",
                "jump_distance_nm": "nm",
                "component_probability": "1",
            },
            "component": {"weight": "1", "diffusion": diffusion_unit},
            "model": {"log_likelihood": "1", "aic": "1", "bic": "1"},
        },
        diagnostics=_base_diagnostics(
            data,
            warnings=warnings + (() if converged else (
                "The mixture reached the iteration limit before convergence.",)),
        ),
        provenance={
            **data.provenance,
            "dimensions": dimensions,
            "components": components,
            "localization_sigma_nm": sigma,
            "lag": "adjacent measured localizations within explicit segments",
            "model": "isotropic zero-drift Brownian Gaussian displacement mixture",
            "model_selection": "not automatic",
        },
        citations=(SCHUTZ_1997_DOI,),
    )


def confinement_index(
    data: PreparedTrajectories,
    *,
    diffusion: float,
    dimensions: int = 2,
    window_points: int = 10,
) -> TrackingResult:
    """Simson fixed-window confinement probability index.

    The original probability approximation is two-dimensional and requires the
    free Brownian diffusion coefficient.  It is therefore explicit input here;
    estimating it from the same confined window would make the test circular.
    """
    dimensions = _validate_dimensions(dimensions)
    if dimensions != 2:
        raise ValueError("The Simson confinement probability is defined for 2-D motion.")
    diffusion = float(diffusion)
    window_points = int(window_points)
    if not np.isfinite(diffusion) or diffusion <= 0:
        raise ValueError("diffusion must be finite and positive.")
    if window_points < 4:
        raise ValueError("window_points must be at least four.")

    accumulation = np.zeros(data.n_points, dtype=float)
    support = np.zeros(data.n_points, dtype=np.int64)
    rows: dict[str, list] = {
        "segment_id": [], "trace_id": [], "start_row": [], "end_row": [],
        "duration": [], "radius_nm": [], "log10_probability": [],
        "confinement_index": [],
    }
    too_short = 0
    invalid = 0
    for segment_id, (start, count) in enumerate(
        zip(data.segment_starts, data.segment_counts)
    ):
        end = int(start + count)
        if count < window_points:
            too_short += 1
            continue
        for left in range(start, end - window_points + 1):
            right = left + window_points
            points = data.coords_nm[left:right, :2]
            duration = float(data.absolute_time[right - 1] - data.absolute_time[left])
            radius = float(np.max(np.linalg.norm(points - points[0], axis=1)))
            if duration <= 0 or not np.isfinite(radius):
                invalid += 1
                continue
            if radius == 0.0:
                log_probability = -np.inf
                index = np.inf
            else:
                log_probability = 0.2048 - 2.5117 * diffusion * duration / radius ** 2
                index = max(0.0, -log_probability - 1.0) if log_probability <= -1 else 0.0
            accumulation[left:right] += index
            support[left:right] += 1
            rows["segment_id"].append(segment_id)
            rows["trace_id"].append(data.trace_ids[start])
            rows["start_row"].append(data.row_indices[left])
            rows["end_row"].append(data.row_indices[right - 1])
            rows["duration"].append(duration)
            rows["radius_nm"].append(radius)
            rows["log10_probability"].append(log_probability)
            rows["confinement_index"].append(index)

    per_point = np.full(data.n_points, np.nan, dtype=float)
    included = support > 0
    per_point[included] = accumulation[included] / support[included]
    window_table = {key: np.asarray(values) for key, values in rows.items()}
    localization_table = {
        "row_index": data.row_indices.copy(),
        "trace_id": data.trace_ids.copy(),
        "segment_id": data.segment_codes.copy(),
        "time": data.trace_time.copy(),
        "confinement_index": per_point,
        "supporting_windows": support,
    }
    warnings = (
        "The index tests residence against 2-D free Brownian motion with the "
        "supplied diffusion coefficient; it is sensitive to that coefficient and "
        "window length and is not itself a motion-state label.",
    )
    return TrackingResult(
        method_id="minflux_viewer.tracking.confinement_index",
        method_version="1.0",
        tables={"localization": localization_table, "window": window_table},
        units={
            "localization": {"time": data.time_unit, "confinement_index": "1"},
            "window": {
                "duration": data.time_unit,
                "radius_nm": "nm",
                "log10_probability": "1",
                "confinement_index": "1",
            },
        },
        diagnostics=_base_diagnostics(
            data,
            status="ok" if rows["segment_id"] else "insufficient-data",
            excluded={"segments_too_short": too_short, "invalid_windows": invalid},
            warnings=warnings,
        ),
        provenance={
            **data.provenance,
            "dimensions": dimensions,
            "diffusion": diffusion,
            "window_points": window_points,
            "probability_model": "log10(psi) = 0.2048 - 2.5117*D*t/R²",
            "diffusion_source": "explicit caller input",
        },
        citations=(SIMSON_1995_DOI,),
    )


def _segment_tamsd(
    times: np.ndarray,
    points: np.ndarray,
    *,
    lag_bin: float,
    max_lag_fraction: float,
    min_pairs: int,
) -> dict[int, tuple[float, float, int]]:
    maximum = max(1, int(np.floor((times.size - 1) * max_lag_fraction)))
    lag_parts: list[np.ndarray] = []
    value_parts: list[np.ndarray] = []
    for offset in range(1, maximum + 1):
        lag = times[offset:] - times[:-offset]
        delta = points[offset:] - points[:-offset]
        valid = lag > 0
        if np.any(valid):
            lag_parts.append(lag[valid])
            value_parts.append(np.sum(delta[valid] ** 2, axis=1))
    if not lag_parts:
        return {}
    lag = np.concatenate(lag_parts)
    values = np.concatenate(value_parts)
    codes = np.maximum(1, np.rint(lag / lag_bin).astype(np.int64))
    result: dict[int, tuple[float, float, int]] = {}
    for code in np.unique(codes):
        selected = codes == code
        count = int(np.count_nonzero(selected))
        if count >= min_pairs:
            result[int(code)] = (
                float(np.mean(lag[selected])), float(np.mean(values[selected])), count)
    return result


def ergodicity_breaking(
    data: PreparedTrajectories,
    *,
    dimensions: int = 2,
    lag_bin: float | None = None,
    max_lag_fraction: float = 0.1,
    min_pairs_per_segment: int = 3,
    min_segments: int = 5,
    min_segment_points: int = 8,
) -> TrackingResult:
    """Finite-time EB statistic from the distribution of per-segment TAMSDs.

    For each common lag, ``EB = <TAMSD²>/<TAMSD>² - 1`` and the normalized
    amplitude ``xi = TAMSD/<TAMSD>`` are reported.  This is the formal scatter
    statistic; finite/unequal observation times and biological heterogeneity
    still prevent it from proving a stochastic process non-ergodic.
    """
    dimensions = _validate_dimensions(dimensions)
    if lag_bin is None:
        lag_bin = data.baseline_interval
    if lag_bin is None or not np.isfinite(lag_bin) or lag_bin <= 0:
        raise ValueError("A finite positive lag_bin is required.")
    if not (0 < float(max_lag_fraction) <= 0.5):
        raise ValueError("max_lag_fraction must be in (0, 0.5].")
    if min_pairs_per_segment < 1 or min_segments < 2 or min_segment_points < 3:
        raise ValueError("Minimum pair, segment, and point counts are too small.")

    records: list[tuple[int, object, float, int, float, float, int]] = []
    too_short = 0
    for segment_id, (start, count) in enumerate(
        zip(data.segment_starts, data.segment_counts)
    ):
        end = int(start + count)
        if count < min_segment_points:
            too_short += 1
            continue
        times = data.absolute_time[start:end]
        curve = _segment_tamsd(
            times, data.coords_nm[start:end, :dimensions],
            lag_bin=float(lag_bin), max_lag_fraction=float(max_lag_fraction),
            min_pairs=int(min_pairs_per_segment))
        duration = float(times[-1] - times[0])
        for code, (lag, value, pairs) in curve.items():
            records.append((
                segment_id, data.trace_ids[start], duration, code, lag, value, pairs))

    means: dict[int, float] = {}
    ensemble_rows: dict[str, list] = {
        "lag": [], "nominal_lag": [], "mean_tamsd_nm2": [],
        "sem_tamsd_nm2": [], "ergodicity_breaking": [], "n_segments": [],
    }
    codes = sorted({record[3] for record in records})
    for code in codes:
        selected = [record for record in records if record[3] == code]
        if len(selected) < min_segments:
            continue
        lags = np.asarray([record[4] for record in selected], dtype=float)
        values = np.asarray([record[5] for record in selected], dtype=float)
        mean = float(np.mean(values))
        means[code] = mean
        eb = float(np.mean(values ** 2) / mean ** 2 - 1.0) if mean > 0 else np.nan
        sem = float(np.std(values, ddof=1) / np.sqrt(values.size))
        ensemble_rows["lag"].append(float(np.mean(lags)))
        ensemble_rows["nominal_lag"].append(code * float(lag_bin))
        ensemble_rows["mean_tamsd_nm2"].append(mean)
        ensemble_rows["sem_tamsd_nm2"].append(sem)
        ensemble_rows["ergodicity_breaking"].append(eb)
        ensemble_rows["n_segments"].append(len(selected))

    segment_rows: dict[str, list] = {
        "segment_id": [], "trace_id": [], "observation_duration": [],
        "lag": [], "nominal_lag": [], "tamsd_nm2": [],
        "normalized_amplitude": [], "n_pairs": [],
    }
    for segment_id, trace_id, duration, code, lag, value, pairs in records:
        if code not in means:
            continue
        segment_rows["segment_id"].append(segment_id)
        segment_rows["trace_id"].append(trace_id)
        segment_rows["observation_duration"].append(duration)
        segment_rows["lag"].append(lag)
        segment_rows["nominal_lag"].append(code * float(lag_bin))
        segment_rows["tamsd_nm2"].append(value)
        segment_rows["normalized_amplitude"].append(value / means[code])
        segment_rows["n_pairs"].append(pairs)

    segment_table = {key: np.asarray(values) for key, values in segment_rows.items()}
    ensemble_table = {key: np.asarray(values) for key, values in ensemble_rows.items()}
    warnings = (
        "This is a finite-time EB statistic. Unequal observation durations, "
        "localization error, drift, and heterogeneous particle populations can "
        "broaden it without proving weak ergodicity breaking.",
        "Interpret the normalized-amplitude distribution together with its "
        "observation_duration column and the ensemble/time-averaged MSD curves.",
    )
    return TrackingResult(
        method_id="minflux_viewer.tracking.ergodicity_breaking",
        method_version="1.0",
        tables={"segment_tamsd": segment_table, "ensemble": ensemble_table},
        units={
            "segment_tamsd": {
                "observation_duration": data.time_unit,
                "lag": data.time_unit,
                "nominal_lag": data.time_unit,
                "tamsd_nm2": "nm²",
                "normalized_amplitude": "1",
            },
            "ensemble": {
                "lag": data.time_unit,
                "nominal_lag": data.time_unit,
                "mean_tamsd_nm2": "nm²",
                "sem_tamsd_nm2": "nm²",
                "ergodicity_breaking": "1",
            },
        },
        diagnostics=_base_diagnostics(
            data,
            status="ok" if ensemble_rows["lag"] else "insufficient-data",
            excluded={"segments_too_short": too_short},
            warnings=warnings,
        ),
        provenance={
            **data.provenance,
            "dimensions": dimensions,
            "lag_bin": float(lag_bin),
            "max_lag_fraction": float(max_lag_fraction),
            "min_pairs_per_segment": int(min_pairs_per_segment),
            "min_segments": int(min_segments),
            "min_segment_points": int(min_segment_points),
            "definition": "<TAMSD²>/<TAMSD>² - 1 at common lag",
        },
        citations=(BUROV_2011_DOI,),
    )


register_tracking_method(TrackingMethodSpec(
    method_id="minflux_viewer.tracking.jump_distribution",
    label="Empirical jump-distance distribution",
    version="1.0",
    description="Gap-safe adjacent measured jumps without an imposed state model.",
    runner=jump_distance_distribution,
    citations=(SCHUTZ_1997_DOI,),
))
register_tracking_method(TrackingMethodSpec(
    method_id="minflux_viewer.tracking.jump_mixture",
    label="Brownian jump-distance mixture",
    version="1.0",
    description="Fixed-size isotropic Brownian mixture for irregular adjacent lags.",
    runner=fit_jump_distance_mixture,
    citations=(SCHUTZ_1997_DOI,),
))
register_tracking_method(TrackingMethodSpec(
    method_id="minflux_viewer.tracking.confinement_index",
    label="Simson confinement index",
    version="1.0",
    description="Fixed-window 2-D Brownian residence-probability index.",
    runner=confinement_index,
    citations=(SIMSON_1995_DOI,),
))
register_tracking_method(TrackingMethodSpec(
    method_id="minflux_viewer.tracking.ergodicity_breaking",
    label="Finite-time ergodicity-breaking statistic",
    version="1.0",
    description="TAMSD amplitude distribution and formal finite-time EB statistic.",
    runner=ergodicity_breaking,
    citations=(BUROV_2011_DOI,),
))


__all__ = [
    "confinement_index",
    "ergodicity_breaking",
    "fit_jump_distance_mixture",
    "jump_distance_distribution",
]
