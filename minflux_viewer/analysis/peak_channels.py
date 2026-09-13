"""Channel boundaries from the peaks of a distribution.

The complement of a mixture fit: instead of assuming a component shape, smooth
the distribution, keep the ``n`` most prominent peaks and cut at the lowest point
between neighbours. Benchmarked against a fit on synthetic channels — it matches
when the peaks are distinct, and it is the honest answer when they are not,
because it reports how many peaks it actually found instead of always returning
``n`` components.

Two measured lessons are baked in:

* **A prominence floor is mandatory.** Counting every local maximum of a
  data-driven smoothing found 15-21 "peaks" on real DCR trace means, each an
  isolated point in a tail. Peaks below :data:`PROMINENCE_FLOOR` of the tallest
  are ignored.
* **The cut is the valley, not the midpoint.** The lowest density between two
  peaks is where the fewest localizations sit, so a small misplacement moves the
  fewest rows. (It is not the Bayes boundary; a fit gives that.)

Pure NumPy/SciPy, Qt-free.
"""

from __future__ import annotations

import numpy as np
from scipy.ndimage import gaussian_filter1d
from scipy.signal import find_peaks

__all__ = ["PROMINENCE_FLOOR", "peak_channel_boundaries", "smoothed_density"]

#: Ignore peaks shorter than this fraction of the tallest one.
PROMINENCE_FLOOR = 0.02
#: Grid resolution of the smoothed density.
GRID_BINS = 2048


def _bandwidth(values: np.ndarray) -> float:
    """Silverman's rule of thumb, the usual data-driven smoothing width."""
    sd = float(np.std(values))
    q75, q25 = np.percentile(values, [75, 25])
    iqr = float(q75 - q25)
    spread = min(sd, iqr / 1.349) if iqr > 0 else sd
    if not np.isfinite(spread) or spread <= 0:
        spread = abs(float(np.mean(values))) or 1.0
    return 0.9 * spread * float(values.size) ** -0.2


def smoothed_density(values, *, bandwidth: float | None = None,
                     bins: int = GRID_BINS) -> tuple[np.ndarray, np.ndarray]:
    """``(centres, density)`` of *values*, Gaussian-smoothed on a regular grid."""
    x = np.asarray(values, dtype=float).ravel()
    x = x[np.isfinite(x)]
    if x.size < 2:
        return np.empty(0), np.empty(0)
    lo, hi = float(x.min()), float(x.max())
    if hi <= lo:
        hi = lo + 1.0
    h = float(bandwidth) if bandwidth else _bandwidth(x)
    pad = 0.05 * (hi - lo) + 3.0 * max(h, 0.0)
    edges = np.linspace(lo - pad, hi + pad, int(bins) + 1)
    counts, _ = np.histogram(x, bins=edges)
    step = edges[1] - edges[0]
    sigma = max(h / step, 0.5) if step > 0 else 0.5
    density = gaussian_filter1d(counts.astype(float), sigma, mode="constant")
    centres = 0.5 * (edges[:-1] + edges[1:])
    return centres, density


def peak_channel_boundaries(
    values,
    n_channels: int,
    *,
    prominence_floor: float = PROMINENCE_FLOOR,
    bandwidth: float | None = None,
) -> tuple[np.ndarray, int]:
    """Cut points separating the most prominent peaks of *values*.

    Returns ``(boundaries, n_peaks)``: at most ``n_channels`` peaks are kept and
    the boundaries are the density minima between consecutive kept peaks, so
    ``len(boundaries) == n_peaks - 1``. ``n_peaks`` is what the data supports,
    which can be **fewer** than requested — two channels that overlap into one
    hump have one peak, and saying so is more useful than inventing a cut.
    """
    n_channels = max(1, int(n_channels))
    centres, density = smoothed_density(values, bandwidth=bandwidth)
    if centres.size == 0 or density.max() <= 0:
        return np.empty(0), 0
    peaks, properties = find_peaks(
        density, prominence=float(prominence_floor) * float(density.max()))
    if peaks.size == 0:
        return np.empty(0), 0
    order = np.argsort(properties["prominences"])[::-1][:n_channels]
    kept = np.sort(peaks[order])
    bounds = [
        float(centres[a + int(np.argmin(density[a:b + 1]))])
        for a, b in zip(kept[:-1], kept[1:])
    ]
    return np.asarray(bounds, dtype=float), int(kept.size)
