"""Directed spatial-pattern analysis around a 2-D line or curved centerline.

The centerline is parameterized by arc length ``s`` from the ROI's first
vertex. Localizations are projected onto the nearest centerline segment and
expressed in a local frame:

``s``
    Directed distance along the centerline.
``u``
    Signed perpendicular distance. Positive ``u`` is the +90 degree normal
    from the directed tangent; callers may flip the sign for specimen-facing
    left/right or above/below conventions.

The resulting straightened ``(s, u)`` count map retains side information that
an ordinary summed line profile discards. Period estimates are provided for
both longitudinal density and the signed transverse centroid.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .plot_profile import path_cumlen

DEFAULT_INTERPOLATION_STEP_NM = 2.0
DEFAULT_HALF_WIDTH_NM = 50.0
DEFAULT_PROFILE_BIN_NM = 5.0
DEFAULT_TRANSVERSE_BIN_NM = 2.0
DEFAULT_PROFILE_SMOOTHING_NM = 5.0
DEFAULT_BACKGROUND_SCALE_NM = 200.0
DEFAULT_MIN_PERIOD_NM = 10.0
DEFAULT_MAX_PERIOD_NM = 500.0
DEFAULT_PEAK_PROMINENCE = 0.15
DEFAULT_PEAK_ORDER = 5

_MAX_CENTERLINE_SAMPLES = 1_000_000
_MAX_STRAIGHTENED_CELLS = 20_000_000


@dataclass(frozen=True)
class Centerline2D:
    """Uniformly sampled directed centerline and its local frame."""

    source_points_nm: np.ndarray
    points_nm: np.ndarray
    arc_nm: np.ndarray
    tangent: np.ndarray
    normal: np.ndarray
    model: str
    smoothing_nm: float


@dataclass(frozen=True)
class SpatialLinePatternResult:
    """Straightened coordinates, profiles, and exploratory period estimates."""

    centerline: Centerline2D
    point_indices: np.ndarray
    point_s_nm: np.ndarray
    point_u_nm: np.ndarray
    s_edges_nm: np.ndarray
    u_edges_nm: np.ndarray
    s_centers_nm: np.ndarray
    u_centers_nm: np.ndarray
    straightened_counts: np.ndarray
    total_profile: np.ndarray
    positive_profile: np.ndarray
    negative_profile: np.ndarray
    asymmetry: np.ndarray
    transverse_centroid_nm: np.ndarray
    smoothed_profile: np.ndarray
    detrended_profile: np.ndarray
    smoothed_transverse_centroid_nm: np.ndarray
    detrended_transverse_centroid_nm: np.ndarray
    peak_indices: np.ndarray
    peak_positions_nm: np.ndarray
    peak_prominences: np.ndarray
    peak_spacing_by_order_nm: tuple[np.ndarray, ...]
    spectrum_periods_nm: np.ndarray
    density_spectrum_power: np.ndarray
    transverse_spectrum_power: np.ndarray
    autocorrelation_lags_nm: np.ndarray
    density_autocorrelation: np.ndarray
    density_fft_period_nm: float
    density_fft_snr: float
    density_autocorr_period_nm: float
    transverse_fft_period_nm: float
    transverse_fft_snr: float
    n_input: int
    n_used: int


def _clean_path(path_nm) -> np.ndarray:
    points = np.asarray(path_nm, dtype=float)
    if points.ndim != 2 or points.shape[1] < 2:
        raise ValueError("The line ROI must provide at least two 2-D vertices.")
    points = points[:, :2]
    points = points[np.all(np.isfinite(points), axis=1)]
    if points.shape[0] < 2:
        raise ValueError("The line ROI must provide at least two finite vertices.")
    length = np.linalg.norm(np.diff(points, axis=0), axis=1)
    points = points[np.concatenate([[True], length > 1.0e-9])]
    if points.shape[0] < 2:
        raise ValueError("The line ROI has zero length.")
    return points


def _uniform_samples(points: np.ndarray, step_nm: float) -> tuple[np.ndarray, np.ndarray]:
    arc_source = path_cumlen(points)
    total = float(arc_source[-1])
    if total <= 0.0:
        raise ValueError("The line ROI has zero length.")
    step = max(float(step_nm), 1.0e-6)
    sample_count = int(np.ceil(total / step)) + 1
    if sample_count > _MAX_CENTERLINE_SAMPLES:
        raise ValueError(
            f"Centerline sampling would need {sample_count:,} points "
            f"(cap {_MAX_CENTERLINE_SAMPLES:,}); increase the interpolation step."
        )
    arc = np.arange(0.0, total, step, dtype=float)
    if arc.size == 0 or not np.isclose(arc[-1], total):
        arc = np.concatenate([arc, [total]])
    sampled = np.column_stack(
        [np.interp(arc, arc_source, points[:, axis]) for axis in range(2)]
    )
    return sampled, arc


def _check_centerline_sample_count(total_nm: float, step_nm: float) -> None:
    sample_count = int(np.ceil(float(total_nm) / max(float(step_nm), 1.0e-6))) + 1
    if sample_count > _MAX_CENTERLINE_SAMPLES:
        raise ValueError(
            f"Centerline sampling would need {sample_count:,} points "
            f"(cap {_MAX_CENTERLINE_SAMPLES:,}); increase the interpolation step."
        )


def _local_frame(points: np.ndarray, *, flip_side: bool) -> tuple[np.ndarray, np.ndarray]:
    delta = np.empty_like(points)
    delta[0] = points[1] - points[0]
    delta[-1] = points[-1] - points[-2]
    if points.shape[0] > 2:
        delta[1:-1] = points[2:] - points[:-2]
    norm = np.linalg.norm(delta, axis=1)
    bad = norm <= 1.0e-12
    if np.any(bad):
        delta[bad] = np.array([1.0, 0.0])
        norm[bad] = 1.0
    tangent = delta / norm[:, None]
    normal = np.column_stack([-tangent[:, 1], tangent[:, 0]])
    if flip_side:
        normal = -normal
    return tangent, normal


def fit_centerline(
    path_nm,
    *,
    model: str = "cubic",
    interpolation_step_nm: float = DEFAULT_INTERPOLATION_STEP_NM,
    smoothing_nm: float = 0.0,
    flip_side: bool = False,
) -> Centerline2D:
    """Fit and uniformly sample a directed polyline or smoothing cubic spline.

    The spline smoothing value is an approximate RMS vertex deviation in nm.
    Endpoints are strongly anchored so the ROI direction and analyzed interval
    remain defined by the user's start and end vertices.
    """

    source = _clean_path(path_nm)
    step = max(float(interpolation_step_nm), 1.0e-6)
    _check_centerline_sample_count(path_cumlen(source)[-1], step)
    mode = str(model).strip().lower()
    if mode not in {"polyline", "cubic"}:
        raise ValueError(f"Unknown centerline model: {model!r}")

    fitted = source
    smoothing = max(float(smoothing_nm), 0.0)
    if mode == "cubic" and source.shape[0] >= 3:
        from scipy.interpolate import splev, splprep

        source_arc = path_cumlen(source)
        degree = min(3, source.shape[0] - 1)
        weights = np.ones(source.shape[0], dtype=float)
        weights[[0, -1]] = 1.0e6
        smooth_bound = source.shape[0] * smoothing * smoothing
        tck, _u = splprep(
            [source[:, 0], source[:, 1]],
            u=source_arc,
            w=weights,
            k=degree,
            s=smooth_bound,
            per=False,
        )
        dense_step = max(min(step / 4.0, 1.0), 0.05)
        dense_count = min(
            _MAX_CENTERLINE_SAMPLES,
            max(
                source.shape[0] * 16,
                int(np.ceil(float(source_arc[-1]) / dense_step)) + 1,
            ),
        )
        dense_arc = np.linspace(0.0, float(source_arc[-1]), dense_count)
        fitted = np.column_stack(splev(dense_arc, tck))
        fitted[0] = source[0]
        fitted[-1] = source[-1]
        fitted = _clean_path(fitted)

    points, arc = _uniform_samples(fitted, step)
    tangent, normal = _local_frame(points, flip_side=flip_side)
    return Centerline2D(
        source_points_nm=source,
        points_nm=points,
        arc_nm=arc,
        tangent=tangent,
        normal=normal,
        model=mode,
        smoothing_nm=smoothing,
    )


def project_to_centerline(
    localizations_xy_nm,
    centerline: Centerline2D,
    *,
    half_width_nm: float,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Project finite localizations onto the nearest centerline segment.

    Returns original row indices plus directed ``s`` and signed ``u`` for rows
    inside ``abs(u) <= half_width_nm``.
    """

    points = np.asarray(localizations_xy_nm, dtype=float)
    if points.ndim != 2 or points.shape[1] < 2:
        raise ValueError("Localizations must be an (N, 2) coordinate array.")
    points = points[:, :2]
    half = float(half_width_nm)
    if not np.isfinite(half) or half <= 0.0:
        raise ValueError("The one-sided analysis width must be positive.")

    finite = np.all(np.isfinite(points), axis=1)
    original = np.flatnonzero(finite)
    candidates = points[finite]
    if candidates.shape[0] == 0:
        empty = np.zeros(0, dtype=float)
        return np.zeros(0, dtype=np.int64), empty, empty

    curve = centerline.points_nm
    lo = curve.min(axis=0) - half
    hi = curve.max(axis=0) + half
    broad = np.all((candidates >= lo) & (candidates <= hi), axis=1)
    original = original[broad]
    candidates = candidates[broad]
    if candidates.shape[0] == 0:
        empty = np.zeros(0, dtype=float)
        return np.zeros(0, dtype=np.int64), empty, empty

    from scipy.spatial import cKDTree

    nearest = np.asarray(cKDTree(curve).query(candidates, k=1)[1], dtype=np.int64)
    best_d2 = np.full(candidates.shape[0], np.inf)
    best_s = np.zeros(candidates.shape[0], dtype=float)
    best_u = np.zeros(candidates.shape[0], dtype=float)
    last_segment = curve.shape[0] - 2

    for offset in (-1, 0):
        segment = np.clip(nearest + offset, 0, last_segment)
        start = curve[segment]
        vector = curve[segment + 1] - start
        length2 = np.einsum("ij,ij->i", vector, vector)
        valid = length2 > 1.0e-18
        t = np.zeros(candidates.shape[0], dtype=float)
        t[valid] = np.clip(
            np.einsum("ij,ij->i", candidates[valid] - start[valid], vector[valid])
            / length2[valid],
            0.0,
            1.0,
        )
        foot = start + t[:, None] * vector
        residual = candidates - foot
        d2 = np.einsum("ij,ij->i", residual, residual)
        length = np.sqrt(np.maximum(length2, 1.0e-18))
        normal = np.column_stack([-vector[:, 1], vector[:, 0]]) / length[:, None]
        align = np.einsum("ij,ij->i", normal, centerline.normal[segment])
        normal[align < 0.0] *= -1.0
        signed = np.einsum("ij,ij->i", residual, normal)
        better = valid & (d2 < best_d2)
        best_d2[better] = d2[better]
        best_s[better] = (
            centerline.arc_nm[segment[better]]
            + t[better] * length[better]
        )
        best_u[better] = signed[better]

    keep = np.sqrt(best_d2) <= half
    return original[keep].astype(np.int64), best_s[keep], best_u[keep]


def _uniform_edges(start: float, stop: float, requested_step: float) -> np.ndarray:
    span = float(stop) - float(start)
    if span <= 0.0:
        return np.array([float(start), float(stop)], dtype=float)
    count = max(1, int(np.ceil(span / max(float(requested_step), 1.0e-6))))
    return np.linspace(float(start), float(stop), count + 1)


def _smooth_and_detrend(
    values: np.ndarray,
    bin_nm: float,
    smoothing_nm: float,
    background_scale_nm: float,
) -> tuple[np.ndarray, np.ndarray]:
    from scipy.ndimage import gaussian_filter1d

    values = np.asarray(values, dtype=float)
    smooth_sigma = max(float(smoothing_nm) / max(bin_nm, 1.0e-9), 0.0)
    smoothed = (
        gaussian_filter1d(values, smooth_sigma, mode="reflect")
        if smooth_sigma > 1.0e-6
        else values.copy()
    )
    background_sigma = max(
        float(background_scale_nm) / max(bin_nm, 1.0e-9),
        0.0,
    )
    if background_sigma > 1.0e-6:
        background = gaussian_filter1d(smoothed, background_sigma, mode="reflect")
    else:
        background = np.full(smoothed.shape, np.mean(smoothed))
    return smoothed, smoothed - background


def _fill_missing(values: np.ndarray) -> np.ndarray:
    values = np.asarray(values, dtype=float)
    finite = np.isfinite(values)
    if np.count_nonzero(finite) == 0:
        return np.zeros(values.shape, dtype=float)
    if np.count_nonzero(finite) == 1:
        return np.full(values.shape, values[finite][0], dtype=float)
    index = np.arange(values.size, dtype=float)
    return np.interp(index, index[finite], values[finite])


def _period_spectrum(
    values: np.ndarray,
    bin_nm: float,
    min_period_nm: float,
    max_period_nm: float,
) -> tuple[np.ndarray, np.ndarray, float, float]:
    signal = np.asarray(values, dtype=float)
    if signal.size < 4 or not np.any(np.abs(signal) > 0.0):
        empty = np.zeros(0, dtype=float)
        return empty, empty, float("nan"), float("nan")
    windowed = signal * np.hanning(signal.size)
    spectrum = np.fft.rfft(windowed)
    frequency = np.fft.rfftfreq(signal.size, d=max(float(bin_nm), 1.0e-9))
    power = np.abs(spectrum) ** 2
    valid = frequency > 0.0
    period = np.zeros_like(frequency)
    period[valid] = 1.0 / frequency[valid]
    lo, hi = sorted((float(min_period_nm), float(max_period_nm)))
    valid &= (period >= lo) & (period <= hi)
    periods = period[valid]
    powers = power[valid]
    if periods.size == 0:
        return periods, powers, float("nan"), float("nan")
    order = np.argsort(periods)
    periods = periods[order]
    powers = powers[order]
    peak = int(np.argmax(powers))
    positive = powers[powers > 0.0]
    baseline = float(np.median(positive)) if positive.size else 0.0
    snr = float(powers[peak] / baseline) if baseline > 0.0 else float("inf")
    return periods, powers, float(periods[peak]), snr


def _autocorrelation(
    values: np.ndarray,
    bin_nm: float,
    min_period_nm: float,
    max_period_nm: float,
) -> tuple[np.ndarray, np.ndarray, float]:
    from scipy.signal import fftconvolve, find_peaks

    signal = np.asarray(values, dtype=float)
    if signal.size < 4 or not np.any(np.abs(signal) > 0.0):
        empty = np.zeros(0, dtype=float)
        return empty, empty, float("nan")
    corr = fftconvolve(signal, signal[::-1], mode="full")[signal.size - 1 :]
    corr /= np.arange(signal.size, 0, -1, dtype=float)
    if corr[0] > 0.0:
        corr /= corr[0]
    lags = np.arange(signal.size, dtype=float) * float(bin_nm)
    lo, hi = sorted((float(min_period_nm), float(max_period_nm)))
    search = (lags >= lo) & (lags <= hi)
    indices = np.flatnonzero(search)
    if indices.size == 0:
        return lags, corr, float("nan")
    local_peaks, _ = find_peaks(corr[indices])
    if local_peaks.size == 0:
        period = float("nan")
    else:
        selected = indices[local_peaks]
        period = float(lags[selected[int(np.argmax(corr[selected]))]])
    return lags, corr, period


def analyze_spatial_line_pattern(
    localizations_xy_nm,
    path_nm,
    *,
    centerline_model: str = "cubic",
    interpolation_step_nm: float = DEFAULT_INTERPOLATION_STEP_NM,
    spline_smoothing_nm: float = 0.0,
    half_width_nm: float = DEFAULT_HALF_WIDTH_NM,
    profile_bin_nm: float = DEFAULT_PROFILE_BIN_NM,
    transverse_bin_nm: float = DEFAULT_TRANSVERSE_BIN_NM,
    profile_smoothing_nm: float = DEFAULT_PROFILE_SMOOTHING_NM,
    background_scale_nm: float = DEFAULT_BACKGROUND_SCALE_NM,
    min_period_nm: float = DEFAULT_MIN_PERIOD_NM,
    max_period_nm: float = DEFAULT_MAX_PERIOD_NM,
    peak_prominence: float = DEFAULT_PEAK_PROMINENCE,
    peak_order: int = DEFAULT_PEAK_ORDER,
    flip_side: bool = False,
) -> SpatialLinePatternResult:
    """Straighten localizations around a directed line and analyze repetition."""

    localizations = np.asarray(localizations_xy_nm, dtype=float)
    if localizations.ndim != 2 or localizations.shape[1] < 2:
        raise ValueError("Localizations must be an (N, 2) coordinate array.")
    centerline = fit_centerline(
        path_nm,
        model=centerline_model,
        interpolation_step_nm=interpolation_step_nm,
        smoothing_nm=spline_smoothing_nm,
        flip_side=flip_side,
    )
    point_indices, point_s, point_u = project_to_centerline(
        localizations,
        centerline,
        half_width_nm=half_width_nm,
    )

    total_length = float(centerline.arc_nm[-1])
    profile_step = float(profile_bin_nm)
    transverse_step = float(transverse_bin_nm)
    if not np.isfinite(profile_step) or profile_step <= 0.0:
        raise ValueError("The profile bin size must be positive.")
    if not np.isfinite(transverse_step) or transverse_step <= 0.0:
        raise ValueError("The transverse bin size must be positive.")
    n_s = max(1, int(np.ceil(total_length / profile_step)))
    n_u = max(
        1,
        int(np.ceil((2.0 * float(half_width_nm)) / transverse_step)),
    )
    if n_s * n_u > _MAX_STRAIGHTENED_CELLS:
        raise ValueError(
            f"Straightened map would need {n_s:,} x {n_u:,} cells "
            f"(cap {_MAX_STRAIGHTENED_CELLS:,}); increase the profile or "
            "transverse bin size."
        )
    s_edges = _uniform_edges(0.0, total_length, profile_bin_nm)
    u_edges = _uniform_edges(
        -float(half_width_nm),
        float(half_width_nm),
        transverse_bin_nm,
    )
    s_centers = 0.5 * (s_edges[:-1] + s_edges[1:])
    u_centers = 0.5 * (u_edges[:-1] + u_edges[1:])
    bin_nm = float(s_edges[1] - s_edges[0])

    if point_s.size:
        straightened, _, _ = np.histogram2d(
            point_s,
            point_u,
            bins=[s_edges, u_edges],
        )
    else:
        straightened = np.zeros(
            (s_edges.size - 1, u_edges.size - 1),
            dtype=float,
        )
    total = straightened.sum(axis=1)
    positive, _ = np.histogram(point_s[point_u >= 0.0], bins=s_edges)
    negative, _ = np.histogram(point_s[point_u < 0.0], bins=s_edges)
    positive = positive.astype(float)
    negative = negative.astype(float)
    asymmetry = np.divide(
        positive - negative,
        total,
        out=np.zeros(total.shape, dtype=float),
        where=total > 0.0,
    )

    s_bin = np.clip(
        np.searchsorted(s_edges, point_s, side="right") - 1,
        0,
        s_centers.size - 1,
    )
    counts = np.bincount(s_bin, minlength=s_centers.size).astype(float)
    u_sum = np.bincount(
        s_bin,
        weights=point_u,
        minlength=s_centers.size,
    )
    centroid = np.divide(
        u_sum,
        counts,
        out=np.full(counts.shape, np.nan, dtype=float),
        where=counts > 0.0,
    )

    smoothed, detrended = _smooth_and_detrend(
        total,
        bin_nm,
        profile_smoothing_nm,
        background_scale_nm,
    )
    filled_centroid = _fill_missing(centroid)
    centroid_smoothed, centroid_detrended = _smooth_and_detrend(
        filled_centroid,
        bin_nm,
        profile_smoothing_nm,
        background_scale_nm,
    )

    from scipy.signal import find_peaks

    prominence_fraction = max(float(peak_prominence), 0.0)
    prominence_scale = float(np.ptp(detrended))
    minimum_peak_distance = max(
        1,
        int(np.floor(float(min_period_nm) / max(bin_nm, 1.0e-9))),
    )
    if prominence_scale > 0.0:
        peak_indices, properties = find_peaks(
            detrended,
            prominence=prominence_fraction * prominence_scale,
            distance=minimum_peak_distance,
        )
        peak_prominences = np.asarray(properties.get("prominences", []), dtype=float)
    else:
        peak_indices = np.zeros(0, dtype=np.int64)
        peak_prominences = np.zeros(0, dtype=float)
    peak_indices = np.asarray(peak_indices, dtype=np.int64)
    peak_positions = s_centers[peak_indices]
    spacing = tuple(
        peak_positions[order:] - peak_positions[:-order]
        for order in range(1, min(max(int(peak_order), 0), peak_positions.size - 1) + 1)
    )

    periods, density_power, density_period, density_snr = _period_spectrum(
        detrended,
        bin_nm,
        min_period_nm,
        max_period_nm,
    )
    transverse_periods, transverse_power, transverse_period, transverse_snr = (
        _period_spectrum(
            centroid_detrended,
            bin_nm,
            min_period_nm,
            max_period_nm,
        )
    )
    if periods.size and transverse_periods.size and not np.array_equal(
        periods, transverse_periods
    ):
        transverse_power = np.interp(
            periods,
            transverse_periods,
            transverse_power,
            left=0.0,
            right=0.0,
        )
    elif periods.size == 0 and transverse_periods.size:
        periods = transverse_periods
        density_power = np.zeros(periods.shape, dtype=float)

    lags, autocorrelation, autocorr_period = _autocorrelation(
        detrended,
        bin_nm,
        min_period_nm,
        max_period_nm,
    )
    return SpatialLinePatternResult(
        centerline=centerline,
        point_indices=point_indices,
        point_s_nm=point_s,
        point_u_nm=point_u,
        s_edges_nm=s_edges,
        u_edges_nm=u_edges,
        s_centers_nm=s_centers,
        u_centers_nm=u_centers,
        straightened_counts=straightened,
        total_profile=total,
        positive_profile=positive,
        negative_profile=negative,
        asymmetry=asymmetry,
        transverse_centroid_nm=centroid,
        smoothed_profile=smoothed,
        detrended_profile=detrended,
        smoothed_transverse_centroid_nm=centroid_smoothed,
        detrended_transverse_centroid_nm=centroid_detrended,
        peak_indices=peak_indices,
        peak_positions_nm=peak_positions,
        peak_prominences=peak_prominences,
        peak_spacing_by_order_nm=spacing,
        spectrum_periods_nm=periods,
        density_spectrum_power=density_power,
        transverse_spectrum_power=transverse_power,
        autocorrelation_lags_nm=lags,
        density_autocorrelation=autocorrelation,
        density_fft_period_nm=density_period,
        density_fft_snr=density_snr,
        density_autocorr_period_nm=autocorr_period,
        transverse_fft_period_nm=transverse_period,
        transverse_fft_snr=transverse_snr,
        n_input=int(localizations.shape[0]),
        n_used=int(point_s.size),
    )
