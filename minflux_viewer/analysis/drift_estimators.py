"""Fiducial-free drift estimators: redundant cross-correlation and entropy minimization.

Two published alternatives to the time-window autocorrelation in
:mod:`minflux_viewer.analysis.drift_correction`, both returning the same
trajectory shape so they can be swapped and compared:

**RCC** — redundant cross-correlation. Wang, Schnitzbauer, Hu, Li, Cheng, Huang &
Huang, *Localization events-based sample drift correction for localization
microscopy with redundant cross-correlation algorithm*, Opt. Express 22, 15982
(2014). https://doi.org/10.1364/OE.22.015982

**DME** — drift at minimum entropy. Cnossen, Cui, Zhang & Smith, *Drift
correction in localization microscopy using entropy minimization*, Opt. Express
29, 27961 (2021). https://doi.org/10.1364/OE.426620

Why these rather than sequential cross-correlation: correlating each segment only
against its neighbour accumulates error along the run, and correlating everything
against the first segment throws away most of the available information. RCC
correlates **all** segment pairs and solves the overdetermined system, so no
error accumulates and the redundancy averages the noise down. DME drops the
histogram entirely and works on the points, which is what makes it usable when
each time bin holds too few localizations to form a correlatable image — the
regime a single MINFLUX cell sits in.

Both work in genuine 3-D here. That is affordable because a rod-shaped cell at a
20 nm bin is only about 100 x 40 x 40 voxels; it is *not* affordable on a
full-field fine-pixel grid, which is why the existing estimator's 3-D path is
avoided in favour of this one.

Pure NumPy/SciPy; no Qt.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

__all__ = [
    "DriftTrajectory",
    "estimate_drift_rcc",
    "estimate_drift_dme",
    "apply_drift",
]


@dataclass(frozen=True)
class DriftTrajectory:
    """A drift estimate: ``offsets_nm[k]`` is the displacement at ``t_s[k]``.

    Mean-subtracted per axis — drift is only defined up to a constant, and
    leaving the offset free would let a comparison between two estimators be
    dominated by an irrelevant translation.
    """

    t_s: np.ndarray
    offsets_nm: np.ndarray
    method: str = ""
    info: dict = field(default_factory=dict)

    def at(self, t) -> np.ndarray:
        """Interpolate the drift onto arbitrary times, as ``(N, 3)`` nm."""
        t = np.asarray(t, dtype=float).ravel()
        return np.column_stack([
            np.interp(t, self.t_s, self.offsets_nm[:, axis])
            for axis in range(self.offsets_nm.shape[1])])

    def excursion_nm(self) -> np.ndarray:
        return np.ptp(self.offsets_nm, axis=0)


def apply_drift(xyz_nm: np.ndarray, t_s: np.ndarray,
                trajectory: DriftTrajectory) -> np.ndarray:
    """Subtract an estimated drift from localizations."""
    return np.asarray(xyz_nm, dtype=float) - trajectory.at(t_s)


def _time_bins(t: np.ndarray, n_bins: int) -> np.ndarray:
    """Assign each localization to one of *n_bins* equal-occupancy time bins.

    Equal occupancy rather than equal duration: MINFLUX acquisition rates vary
    a lot over a run, and equal-duration bins would leave some almost empty.
    """
    t = np.asarray(t, dtype=float)
    order = np.argsort(t, kind="stable")
    edges = np.linspace(0, t.size, int(n_bins) + 1).astype(int)
    labels = np.empty(t.size, dtype=np.int64)
    for index in range(int(n_bins)):
        labels[order[edges[index]:edges[index + 1]]] = index
    return labels


def _bin_times(t: np.ndarray, labels: np.ndarray, n_bins: int) -> np.ndarray:
    return np.array([float(np.mean(t[labels == index])) if np.any(labels == index)
                     else np.nan for index in range(n_bins)])


# --------------------------------------------------------------------------- #
# Redundant cross-correlation
# --------------------------------------------------------------------------- #
def _render(xyz: np.ndarray, origin: np.ndarray, shape: tuple, bin_nm: float):
    index = np.floor((xyz - origin) / bin_nm).astype(np.int64)
    inside = np.all((index >= 0) & (index < np.asarray(shape)), axis=1)
    grid = np.zeros(shape, dtype=float)
    if inside.any():
        np.add.at(grid, tuple(index[inside].T), 1.0)
    return grid


def _subpixel_peak(correlation: np.ndarray) -> np.ndarray:
    """Peak of a cross-correlation, refined to sub-bin by a local centroid."""
    flat = int(np.argmax(correlation))
    peak = np.array(np.unravel_index(flat, correlation.shape), dtype=float)
    offset = np.zeros(correlation.ndim)
    for axis in range(correlation.ndim):
        lo = int(peak[axis]) - 1
        hi = int(peak[axis]) + 2
        if lo < 0 or hi > correlation.shape[axis]:
            continue
        sl = [slice(int(p), int(p) + 1) for p in peak]
        sl[axis] = slice(lo, hi)
        window = correlation[tuple(sl)].ravel().astype(float)
        window = window - window.min()
        total = window.sum()
        if total > 0:
            offset[axis] = float((window * np.array([-1.0, 0.0, 1.0])).sum() / total)
    return peak + offset


def estimate_drift_rcc(xyz_nm: np.ndarray, t_s: np.ndarray, *,
                       n_segments: int = 20, bin_nm: float = 20.0,
                       outlier_nm: float | None = None,
                       min_localizations: int = 50) -> DriftTrajectory:
    """Redundant cross-correlation drift estimate (Wang et al. 2014).

    All ``N(N-1)/2`` segment pairs are correlated and the resulting
    overdetermined system ``r = A D`` is solved by least squares, then residual
    outliers are dropped and it is re-solved. ``outlier_nm`` defaults to one
    bin, mirroring the paper's 0.2 x pixel threshold at its recommended binning.
    """
    from scipy import fft as sp_fft

    xyz = np.asarray(xyz_nm, dtype=float)
    t = np.asarray(t_s, dtype=float).ravel()
    if xyz.ndim != 2 or xyz.shape[1] != 3 or xyz.shape[0] != t.size:
        raise ValueError("xyz_nm must be (N, 3) and match t_s")
    n_segments = max(int(n_segments), 2)
    labels = _time_bins(t, n_segments)
    counts = np.bincount(labels, minlength=n_segments)
    if np.any(counts < int(min_localizations)):
        raise ValueError(
            f"segment(s) with fewer than {min_localizations} localizations "
            f"(smallest {counts.min()}); use fewer segments")

    margin = 4.0 * bin_nm
    origin = xyz.min(axis=0) - margin
    extent = xyz.max(axis=0) + margin - origin
    shape = tuple(int(np.ceil(v / bin_nm)) + 1 for v in extent)
    if int(np.prod(shape)) > 40_000_000:
        raise ValueError(f"correlation grid {shape} is too large; raise bin_nm")

    spectra = []
    for index in range(n_segments):
        grid = _render(xyz[labels == index], origin, shape, bin_nm)
        grid -= grid.mean()                      # drop the DC term
        spectra.append(sp_fft.rfftn(grid))
    centre = np.array(shape, dtype=float) // 2

    rows, shifts = [], []
    for i in range(n_segments):
        for j in range(i + 1, n_segments):
            product = np.conj(spectra[i]) * spectra[j]
            correlation = sp_fft.irfftn(product, s=shape)
            correlation = np.fft.fftshift(correlation)
            peak = _subpixel_peak(correlation)
            shifts.append((peak - centre) * bin_nm)
            row = np.zeros(n_segments)
            row[i], row[j] = -1.0, 1.0
            rows.append(row)
    design = np.asarray(rows)[:, 1:]             # segment 0 is the reference
    measured = np.asarray(shifts)

    threshold = float(outlier_nm) if outlier_nm is not None else bin_nm
    keep = np.ones(design.shape[0], dtype=bool)
    solution = np.zeros((n_segments - 1, 3))
    for _round in range(3):
        solution, *_ = np.linalg.lstsq(design[keep], measured[keep], rcond=None)
        residual = np.linalg.norm(design @ solution - measured, axis=1)
        fresh = residual <= threshold
        # Never drop so many that the system stops being solvable.
        if fresh.sum() < design.shape[1] + 2 or np.array_equal(fresh, keep):
            break
        keep = fresh

    offsets = np.vstack([np.zeros(3), solution])
    offsets -= offsets.mean(axis=0)
    return DriftTrajectory(
        _bin_times(t, labels, n_segments), offsets, "rcc",
        {"n_segments": n_segments, "bin_nm": bin_nm,
         "n_pairs": int(design.shape[0]), "n_pairs_used": int(keep.sum()),
         "grid_shape": shape})


# --------------------------------------------------------------------------- #
# Drift at minimum entropy
# --------------------------------------------------------------------------- #
def _entropy_and_gradient(xyz, labels, offsets, pairs, inv_var, n_bins):
    """Entropy bound and its gradient with respect to the per-bin drift.

    The bound is ``-sum_i log sum_j exp(-|r_ij|^2 / (2 (s_i^2 + s_j^2)))`` over
    neighbour pairs of the *drift-corrected* positions: removing real drift
    superimposes repeated views of the same structure, which concentrates the
    cloud and lowers the entropy.
    """
    left, right = pairs
    corrected = xyz - offsets[labels]
    delta = corrected[left] - corrected[right]
    weight = np.exp(-0.5 * np.sum(delta * delta, axis=1) * inv_var)

    # Per-localization neighbour sums; +eps keeps an isolated point finite.
    sums = np.zeros(xyz.shape[0])
    np.add.at(sums, left, weight)
    np.add.at(sums, right, weight)
    sums += 1e-12
    energy = -float(np.sum(np.log(sums)))

    # dE/dD for E = -sum_i log S_i. With r = (x_i - D_bi) - (x_j - D_bj),
    # dw/dD_bi = +w r / s^2, so dE/dD_bi = -(1/S) w r / s^2 — the pair pulls its
    # two time bins toward each other, which is what concentrates the cloud.
    scale = ((1.0 / sums[left] + 1.0 / sums[right]) * weight * inv_var)[:, None]
    force = scale * delta
    gradient = np.zeros((n_bins, 3))
    np.add.at(gradient, labels[left], -force)
    np.add.at(gradient, labels[right], force)
    return energy, gradient


def estimate_drift_dme(xyz_nm: np.ndarray, t_s: np.ndarray, *,
                       sigma_nm: np.ndarray | float = 5.0,
                       n_bins: int = 20, neighbour_radius_nm: float | None = None,
                       max_pairs: int = 4_000_000,
                       iterations: int = 300, step_nm: float = 2.0,
                       initial: DriftTrajectory | None = None,
                       rcc_segments: int = 10,
                       rcc_bin_nm: float = 20.0) -> DriftTrajectory:
    """Drift at minimum entropy (Cnossen et al. 2021).

    Works on the localizations directly rather than on a rendered image, which
    is what makes it usable when a time bin holds too few points to correlate.
    Gradient descent on a non-convex landscape, so it is started from an RCC
    estimate unless *initial* is given — as the reference implementation does.
    """
    from scipy.spatial import cKDTree

    xyz = np.asarray(xyz_nm, dtype=float)
    t = np.asarray(t_s, dtype=float).ravel()
    if xyz.ndim != 2 or xyz.shape[1] != 3 or xyz.shape[0] != t.size:
        raise ValueError("xyz_nm must be (N, 3) and match t_s")
    n_bins = max(int(n_bins), 2)
    labels = _time_bins(t, n_bins)
    times = _bin_times(t, labels, n_bins)

    sigma = np.broadcast_to(np.asarray(sigma_nm, dtype=float).ravel()
                            if np.ndim(sigma_nm) else
                            np.full(xyz.shape[0], float(sigma_nm)),
                            (xyz.shape[0],)).astype(float)
    radius = (float(neighbour_radius_nm) if neighbour_radius_nm
              else 4.0 * float(np.median(sigma)))

    if initial is None:
        try:
            initial = estimate_drift_rcc(xyz, t, n_segments=min(rcc_segments, n_bins),
                                         bin_nm=rcc_bin_nm)
        except Exception:                                         # noqa: BLE001
            initial = None
    offsets = (initial.at(times) if initial is not None
               else np.zeros((n_bins, 3)))
    offsets -= offsets.mean(axis=0)

    # Neighbours are taken once, on the initialised positions: they define which
    # localizations could plausibly be views of the same structure, and letting
    # that set change every step would make the objective discontinuous.
    tree = cKDTree(xyz - offsets[labels])
    pair_array = tree.query_pairs(radius, output_type="ndarray")
    if pair_array.shape[0] == 0:
        raise ValueError(
            f"no localization pair within {radius:.1f} nm — nothing to align; "
            f"raise neighbour_radius_nm")
    if pair_array.shape[0] > int(max_pairs):
        keep = np.random.default_rng(0).choice(
            pair_array.shape[0], int(max_pairs), replace=False)
        pair_array = pair_array[keep]
    left, right = pair_array[:, 0], pair_array[:, 1]
    # Pairs inside one time bin carry no information about relative drift.
    cross = labels[left] != labels[right]
    left, right = left[cross], right[cross]
    if left.size == 0:
        raise ValueError("every neighbour pair falls in one time bin; "
                         "use fewer bins")
    inv_var = 1.0 / (sigma[left] ** 2 + sigma[right] ** 2)

    energy, gradient = _entropy_and_gradient(
        xyz, labels, offsets, (left, right), inv_var, n_bins)
    history = [energy]
    step = float(step_nm)
    for _ in range(int(iterations)):
        norm = np.max(np.linalg.norm(gradient, axis=1))
        if not np.isfinite(norm) or norm <= 0:
            break
        trial = offsets - step * gradient / norm
        trial -= trial.mean(axis=0)
        new_energy, new_gradient = _entropy_and_gradient(
            xyz, labels, trial, (left, right), inv_var, n_bins)
        if new_energy < energy:
            offsets, energy, gradient = trial, new_energy, new_gradient
            step *= 1.1
        else:
            step *= 0.5
            if step < 1e-3:
                break
        history.append(energy)

    offsets -= offsets.mean(axis=0)
    return DriftTrajectory(
        times, offsets, "dme",
        {"n_bins": n_bins, "n_pairs": int(left.size), "radius_nm": radius,
         "iterations": len(history) - 1, "energy": energy,
         "energy_start": history[0],
         "initialised_with": None if initial is None else initial.method})
