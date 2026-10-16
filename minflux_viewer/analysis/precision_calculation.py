"""
minflux_viewer.analysis.localization_precision
================================================
Localization precision estimators.

Three independent measures are exposed in the *Analysis → Localization
Precision* menu, all fully implemented:

* **StdDev per trace** — the *measured* precision: sample standard deviation of
  the localization coordinates within each trace (one molecule observed multiple
  times). Ostersehlt et al., *Nat. Methods* 19:1072, 2022 (see also Schmidt
  et al., *Nat. Methods* 19:1246, 2022). Groups localizations by trace ID
  (``tid``), computes σx / σy / σz per trace, then summarises across all traces
  with at least :data:`MIN_LOCS_PER_TRACE` repeats (nm; raw z, no Z scaling factor). Also
  written to ``dataset.attr`` for follow-up work.

* **CRLB** — the *theoretical* MINFLUX Cramér-Rao bound for the targeted-donut
  measurement model (Balzarotti et al., *Science* 355:606, 2017; Marin & Ries,
  arXiv:2410.12427, 2024) — **not** the camera/Gaussian bound (Mortensen 2010).
  Per-localization σ from the photons of all exposures (``eco`` + ``ecc``), the
  background measured by the centre exposure (``efc``/``efo``) or the
  instrument's estimate (``fbg``), and the pattern size ``L`` read from the
  acquisition sequence (assumed, and said so, when the file has none).

* **FRC** — Fourier Ring Correlation (Banterle et al., *J. Struct. Biol.*
  183:363, 2013; Nieuwenhuizen et al., *Nat. Methods* 10:557, 2013): the radial
  Fourier ring correlation of two independent half-images, resolution read off at
  the 1/7 crossing.

The CRLB dialog compares measured spread with per-trace RMS model bounds on
matched rows. This is descriptive, not a physical variance decomposition. A
below-bound spread remains unexplained; tracking motion is not precision.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from functools import lru_cache

import numpy as np

from ..core.execution import cpu_threads
from ..core.mfx_sequence import DEFAULT_FINAL_GEO_FACTOR, PATTERN_SCALE_NM
from ..utils.import_compat import resolve_moved_symbol
from .precision_provenance import _bound_kind as _bound_kind
from .precision_provenance import crlb_log_line as crlb_log_line

# ---------------------------------------------------------------------------
# Constants & data class
# ---------------------------------------------------------------------------

MIN_LOCS_PER_TRACE: int = 5   # Schmidt et al. use traces with ≥ 5 localizations


# ---------------------------------------------------------------------------
# StdDev per trace — full implementation
# ---------------------------------------------------------------------------

def stddev_per_trace(
    loc_xyz: np.ndarray,
    tid: np.ndarray,
    *,
    min_locs: int = MIN_LOCS_PER_TRACE,
) -> dict:
    """
    Compute the per-trace localization precision (Ostersehlt et al. 2022).

    For every trace (one molecule observed n times) the sample standard
    deviation of each raw coordinate is

        σ_x, σ_y, σ_z  (nm, ddof = 1),

    the combined lateral spread is ``σ_r = sqrt((σ_x² + σ_y²) / 2)``, and the
    n-weighted standard error of the mean position, averaged over all traces,
    gives the reported precision

        σ_•,combined = mean_traces( σ_• / sqrt(n) ).

    Parameters
    ----------
    loc_xyz : (N, 3) float ndarray
        Localizations in *metres* (raw MINFLUX storage convention) — the **raw**
        coordinates (no Z scaling factor applied to z). Output is reported in nm.
    tid : (N,) int ndarray
        Trace ID per localization.
    min_locs : int
        Traces with fewer than this many localizations are dropped (default 5).

    Returns
    -------
    dict with keys:

        ``per_trace_sigma_xyz``  (M, 3) ndarray of σ_x/σ_y/σ_z in nm, M traces
        ``per_trace_sigma_r``     (M,) ndarray of σ_r in nm
        ``per_trace_n``           (M,) ndarray of N localizations per trace
        ``trace_ids``             (M,) ndarray of the corresponding tids
        ``median_sigma_xyz``      (3,) ndarray, median of column-wise σ in nm
        ``mean_sigma_xyz``        (3,) ndarray, mean of column-wise σ in nm
        ``median_sigma_r``        float, median σ_r in nm
        ``mean_sigma_r``          float, mean σ_r in nm
        ``combined_sigma_xyz``    (3,) ndarray, mean_traces(σ/sqrt(n)) in nm
        ``combined_sigma_r``      float, mean_traces(σ_r/sqrt(n)) in nm
        ``n_traces_used``         int
        ``n_traces_total``        int
    """
    def _empty(n_total: int) -> dict:
        z3 = np.zeros(3, dtype=float)
        return dict(
            per_trace_sigma_xyz=np.empty((0, 3)),
            per_trace_sigma_r=np.empty(0),
            per_trace_n=np.empty(0, dtype=int),
            trace_ids=np.empty(0, dtype=int),
            median_sigma_xyz=z3, mean_sigma_xyz=z3,
            median_sigma_r=float("nan"), mean_sigma_r=float("nan"),
            combined_sigma_xyz=z3, combined_sigma_r=float("nan"),
            n_traces_used=0, n_traces_total=n_total,
        )

    if loc_xyz.size == 0:
        return _empty(0)

    # nm conversion
    loc_nm = np.asarray(loc_xyz, dtype=float) * 1.0e9

    tid = np.asarray(tid).ravel()
    uniq = np.unique(tid)
    n_total = uniq.size

    # Group by trace via a single stable sort + contiguous blocks, instead of an
    # O(N × n_traces) ``inverse == i`` scan per trace (the per-trace loop was a
    # major contributor to dataset open time on files with many traces).
    order = np.argsort(tid, kind="stable")
    tid_sorted = tid[order]
    loc_sorted = loc_nm[order]
    starts = np.flatnonzero(np.r_[True, tid_sorted[1:] != tid_sorted[:-1]])
    ends = np.r_[starts[1:], tid_sorted.size]

    sigmas = []
    counts = []
    used_ids = []
    for s, e in zip(starts, ends):
        n = int(e - s)
        if n < min_locs:
            continue
        # ddof=1 → sample standard deviation
        sig = loc_sorted[s:e].std(axis=0, ddof=1)
        sigmas.append(sig)
        counts.append(n)
        used_ids.append(tid_sorted[s])

    if not sigmas:
        return _empty(n_total)

    per_trace = np.array(sigmas)                     # (M, 3) σ_x/σ_y/σ_z
    n_per = np.array(counts, dtype=int)              # (M,)
    # σ_r = sqrt((σ_x² + σ_y²) / 2) — combined lateral spread per trace.
    sigma_r = np.sqrt((per_trace[:, 0] ** 2 + per_trace[:, 1] ** 2) / 2.0)
    # n-weighted standard error of the mean position, averaged over traces.
    sqrt_n = np.sqrt(n_per.astype(float))
    combined_xyz = (per_trace / sqrt_n[:, None]).mean(axis=0)
    combined_r = float((sigma_r / sqrt_n).mean())
    return dict(
        per_trace_sigma_xyz=per_trace,
        per_trace_sigma_r=sigma_r,
        per_trace_n=n_per,
        trace_ids=np.array(used_ids),
        median_sigma_xyz=np.median(per_trace, axis=0),
        mean_sigma_xyz=per_trace.mean(axis=0),
        median_sigma_r=float(np.median(sigma_r)),
        mean_sigma_r=float(sigma_r.mean()),
        combined_sigma_xyz=combined_xyz,
        combined_sigma_r=combined_r,
        n_traces_used=per_trace.shape[0],
        n_traces_total=n_total,
    )


# ---------------------------------------------------------------------------
# Fourier Ring Correlation (FRC) — image resolution
# ---------------------------------------------------------------------------
#
# Two statistically independent reconstructions of the same structure are
# rendered from a random 50/50 split of the localizations, their 2-D Fourier
# transforms correlated over rings of constant spatial frequency q, and the
# resolution read off at the frequency q_c where the (smoothed) correlation
# first drops below the fixed 1/7 threshold; resolution = 1/q_c.
#
# This mirrors the MINFLUX implementation of Stefan Hell's group: Gwosch et al. (Nat.
# Methods 17:217, 2020, Fig. 2d) and Ostersehlt et al. (Nat. Methods 19:1072,
# 2022): per-localization random split, ~1 nm histogram render, the FRC curve
# averaged over several splits, ring-binned FRC = Σ F1·conj(F2) / √(Σ|F1|²·Σ|F2|²)
# with Savitzky-Golay smoothing, the high-frequency tail cut at 0.8·q_max, and a
# fixed 1/7 criterion. (For data with many localizations per emitter the
# per-localization split can be optimistic, as repeated localizations correlate.)

FRC_THRESHOLD: float = 1.0 / 7.0     # fixed 1/7 resolution criterion
FRC_DEFAULT_REPEATS: int = 5         # average the curve over this many splits
FRC_DEFAULT_IMAGE_PX: int = 1024     # minimum grid size when pixel is auto
FRC_TARGET_PIXEL_NM: float = 2.0     # auto pixel target (grid grows to reach it)
FRC_MAX_IMAGE_PX: int = 4096         # hard cap on grid size (FFT cost)


def _rfft2(img: np.ndarray) -> np.ndarray:
    """Real 2-D FFT — half the spectrum, ~2× faster than a full complex FFT.

    Uses multithreaded ``scipy.fft`` (on the threads the analysis budget has
    free) when available, else NumPy.
    """
    try:
        from scipy.fft import rfft2
    except ImportError:
        return np.fft.rfft2(img)
    # Once selected, a backend failure must not silently select another FFT.
    with cpu_threads() as workers:
        return rfft2(img, workers=workers)


@lru_cache(maxsize=4)
def _rfft_ring_index(n: int) -> tuple[np.ndarray, int]:
    """Flattened integer ring index for an ``rfft2`` spectrum of an n×n image.

    Depends only on the grid size, so it is built once and cached across calls
    (and reused for every repeat) instead of being recomputed each time.
    """
    fy = np.round(np.fft.fftfreq(n) * n).astype(np.int32)[:, None]   # rows: −n/2..n/2
    fx = np.arange(n // 2 + 1, dtype=np.int32)[None, :]              # rfft cols: 0..n/2
    ring = np.round(np.sqrt(fy.astype(np.float32) ** 2
                            + fx.astype(np.float32) ** 2)).astype(np.intp)
    return ring.ravel(), n // 2


@dataclass
class FRCResult:
    freq_inv_nm: np.ndarray   # radial spatial frequency per ring (1/nm)
    frc_mean: np.ndarray      # mean FRC curve over the repeats
    frc_std: np.ndarray       # std of FRC across repeats (shaded band)
    threshold: float          # 1/7
    resolution_nm: float      # 1 / crossing frequency (nm)
    crossing_freq: float      # frequency where FRC < threshold (1/nm)
    pixel_size_nm: float      # super-resolution render pixel
    image_px: int             # render size (pixels per side)
    n_locs: int               # input localizations
    n_repeats: int
    status: str               # "" or a human-readable note
    mode: str = "localization"  # "localization" | "combined"
    n_points: int = 0           # points fed to FRC (locs, or traces if combined)


def _radial_frc(img1: np.ndarray, img2: np.ndarray,
                ring: np.ndarray, rmax: int) -> np.ndarray:
    """Ring-averaged Fourier correlation of two equally sized images.

        FRC(r) = Σ_ring Re{F1·conj(F2)} / sqrt(Σ_ring|F1|² · Σ_ring|F2|²)

    ``ring`` is the precomputed flat ring index for the ``rfft2`` spectrum
    (see :func:`_rfft_ring_index`). Uses the real half-spectrum — the per-ring
    sums halve in both numerator and denominator so the ratio is unchanged.
    """
    f1 = _rfft2(img1)
    f2 = _rfft2(img2)
    cross = (f1.real * f2.real + f1.imag * f2.imag).ravel()     # Re{F1·conj(F2)}
    p1 = (f1.real * f1.real + f1.imag * f1.imag).ravel()        # |F1|²
    p2 = (f2.real * f2.real + f2.imag * f2.imag).ravel()        # |F2|²
    num = np.bincount(ring, weights=cross, minlength=rmax + 1)[: rmax + 1]
    a1 = np.bincount(ring, weights=p1, minlength=rmax + 1)[: rmax + 1]
    a2 = np.bincount(ring, weights=p2, minlength=rmax + 1)[: rmax + 1]
    denom = np.sqrt(a1 * a2)
    return np.where(denom > 0, num / denom, 0.0)


def _smooth_frc(y: np.ndarray) -> np.ndarray:
    """Savitzky-Golay smoothing of the radial FRC curve (as in the reference
    implementation of Ostersehlt et al. 2022)."""
    if y.size < 7:
        return y
    try:
        from scipy.signal import savgol_filter
    except ImportError:
        return np.convolve(y, np.ones(3) / 3, mode="same")
    return savgol_filter(y, 7, 1)


def _frc_crossing(freq: np.ndarray, frc: np.ndarray, threshold: float) -> float:
    """First spatial frequency where ``frc`` falls below ``threshold``.

    Linearly interpolates the crossing. The first two bins (DC + lowest
    frequency, where FRC ≈ 1) are skipped. Returns NaN if it never crosses.
    """
    for i in range(2, frc.size):
        if frc[i] < threshold <= frc[i - 1]:
            y0, y1 = frc[i - 1], frc[i]
            t = (threshold - y0) / (y1 - y0) if y1 != y0 else 0.0
            return float(freq[i - 1] + t * (freq[i] - freq[i - 1]))
    return float("nan")


def frc_resolution(
    x_nm: np.ndarray,
    y_nm: np.ndarray,
    tid: np.ndarray | None = None,
    *,
    mode: str = "localization",
    image_px: int = FRC_DEFAULT_IMAGE_PX,
    pixel_size_nm: float | None = None,
    repeats: int = FRC_DEFAULT_REPEATS,
    threshold: float = FRC_THRESHOLD,
    seed: int = 0,
    progress=None,
) -> FRCResult:
    """Estimate image resolution by Fourier Ring Correlation.

    Follows the MINFLUX method of Stefan Hell's group (Gwosch et al. 2020; Ostersehlt et al.
    2022): a per-localization random 50/50 split into two images, the radial FRC
    curve averaged over ``repeats`` splits, Savitzky-Golay smoothed, tail cut at
    0.8·q_max, resolution = 1/q_c at the 1/7 crossing.

    Parameters
    ----------
    x_nm, y_nm : (N,) float
        Localization coordinates in **nanometres** (2-D; z is ignored).
    tid : (N,) int, optional
        Trace IDs — required for ``mode="combined"``.
    mode : "localization" | "combined"
        ``"localization"`` runs FRC on all localizations. ``"combined"`` first
        collapses each trace to its mean position (one point per molecule) and
        runs FRC on those — the honest MINFLUX resolution, since per-localization
        FRC is optimistic when emitters are localized many times (Ostersehlt
        et al. report both). Falls back to localization mode without ``tid``.
    image_px : int
        Minimum square render size in pixels per side (auto pixel path).
    pixel_size_nm : float, optional
        Super-resolution pixel size. Defaults to a ~1 nm target, the grid
        growing (capped) so it always covers the data extent.
    repeats : int
        Number of independent random half-splits whose FRC curves are averaged.
    threshold : float
        Resolution criterion (default 1/7).
    """
    x = np.asarray(x_nm, dtype=float).ravel()
    y = np.asarray(y_nm, dtype=float).ravel()
    finite = np.isfinite(x) & np.isfinite(y)
    x, y = x[finite], y[finite]
    tid_arr = None if tid is None else np.asarray(tid).ravel()[finite]
    n_locs = int(x.size)

    # Combined mode: collapse each trace to its mean (x, y) position.
    mode_used = "localization"
    if mode == "combined" and tid_arr is not None and n_locs > 0:
        uniq, inv = np.unique(tid_arr, return_inverse=True)
        cnt = np.bincount(inv).astype(float)
        x = np.bincount(inv, weights=x) / cnt
        y = np.bincount(inv, weights=y) / cnt
        mode_used = "combined"
    n_points = int(x.size)

    def _fail(status: str, pixel: float = 0.0) -> FRCResult:
        return FRCResult(
            freq_inv_nm=np.empty(0), frc_mean=np.empty(0), frc_std=np.empty(0),
            threshold=threshold, resolution_nm=float("nan"), crossing_freq=float("nan"),
            pixel_size_nm=pixel, image_px=image_px, n_locs=n_locs,
            n_repeats=repeats, status=status, mode=mode_used, n_points=n_points,
        )

    pts = "combined positions" if mode_used == "combined" else "localizations"
    if n_points < 20:
        return _fail(f"Too few {pts} for FRC (need ≥ 20).")

    x0, x1 = float(x.min()), float(x.max())
    y0, y1 = float(y.min()), float(y.max())
    extent = max(x1 - x0, y1 - y0)
    if extent <= 0:
        return _fail("Localizations span zero area — cannot render an image.")

    # Size the square grid so it always *covers* the data extent. When the pixel
    # is user-specified, derive the grid size from it (capped). Otherwise target
    # a fine ~1 nm pixel (the reference value), growing the grid from ``image_px``
    # up to the cap so large fields of view still resolve without a recompute.
    if pixel_size_nm and pixel_size_nm > 0:
        pixel = float(pixel_size_nm)
        n_px = int(min(FRC_MAX_IMAGE_PX, max(64, np.ceil(extent / pixel))))
    else:
        n_px = int(min(FRC_MAX_IMAGE_PX,
                       max(int(image_px), np.ceil(extent / FRC_TARGET_PIXEL_NM))))
        pixel = extent / n_px
    # Centre the (square) field of view on the data.
    cx_mid, cy_mid = 0.5 * (x0 + x1), 0.5 * (y0 + y1)
    half = 0.5 * n_px * pixel
    edges_x = np.linspace(cx_mid - half, cx_mid + half, n_px + 1)
    edges_y = np.linspace(cy_mid - half, cy_mid + half, n_px + 1)

    def _render(mask: np.ndarray) -> np.ndarray:
        img, _, _ = np.histogram2d(y[mask], x[mask], bins=[edges_y, edges_x])
        return img                                  # no apodization (as in ref)

    # The ring index depends only on the grid size — build it once, not per repeat.
    ring, rmax = _rfft_ring_index(n_px)
    rng = np.random.default_rng(seed)
    curves: list[np.ndarray] = []
    total = max(1, repeats)
    for i in range(total):
        # Random 50/50 split of the points (Gwosch/Ostersehlt: rand < 0.5).
        ix = rng.random(n_points) < 0.5
        if ix.sum() >= 2 and (~ix).sum() >= 2:
            curves.append(_radial_frc(_render(ix), _render(~ix), ring, rmax))
        if progress is not None:
            progress(i + 1, total)

    if not curves:
        return _fail("Could not form two non-empty halves for FRC.", pixel)

    frc_stack = np.vstack(curves)
    frc_mean = frc_stack.mean(axis=0)
    frc_std = frc_stack.std(axis=0)
    freq = np.arange(frc_mean.size) / (n_px * pixel)   # 1/nm per ring

    # Cut the noisy high-frequency tail (q < 0.8·q_max), as in the reference.
    keep = freq <= 0.8 * freq[-1]
    freq, frc_mean, frc_std = freq[keep], frc_mean[keep], frc_std[keep]

    qc = _frc_crossing(freq, _smooth_frc(frc_mean), threshold)
    status = ""
    if not np.isfinite(qc):
        # Never crossed: resolution is finer than the sampling supports.
        qc = float(freq[-1])
        resolution = 1.0 / qc if qc > 0 else 2.0 * pixel
        status = ("FRC did not fall below 1/7 within the sampled frequencies — "
                  "resolution is at or below the pixel limit; re-run with a "
                  "smaller pixel size.")
    else:
        resolution = 1.0 / qc if qc > 0 else float("nan")

    return FRCResult(
        freq_inv_nm=freq, frc_mean=frc_mean, frc_std=frc_std,
        threshold=threshold, resolution_nm=float(resolution), crossing_freq=float(qc),
        pixel_size_nm=pixel, image_px=n_px, n_locs=n_locs,
        n_repeats=len(curves), status=status, mode=mode_used, n_points=n_points,
    )


# ---------------------------------------------------------------------------
# Cramér-Rao Lower Bound (CRLB) — theoretical MINFLUX precision
# ---------------------------------------------------------------------------
#
# The MINFLUX CRB is *not* the camera/Gaussian-PSF bound (Mortensen 2010): it
# follows from the targeted-illumination (donut) measurement model of Balzarotti
# et al. (Science 355:606, 2017). We use the closed forms of Marin & Ries
# (arXiv:2410.12427), evaluated with the emitter at the pattern centre:
#
#   lateral  σ_xy,ideal = L / √(8N) · (1 − L² / (8σ_q²))⁻¹      (Eq. 45)
#   axial    σ_z,ideal  = L_z / (4√N)                             (Eq. 29)
#
# where σ_q is the width of their donut model I ∝ e·r²/(2σ_q²)·exp(−r²/(2σ_q²))
# (Eq. 42) and N is the total number of photons over ALL exposures of the
# iteration, centre included. The default σ_q is SimuFLUX's analytic 2-D donut
# (FWHM 310 nm → σ_q = FWHM/√(8 ln 2) ≈ 131.6 nm); it enters only through the
# envelope term, ≈1 % at L = 40 nm.
#
# Other patterns follow from the same Fisher information: with K equal-dwell
# exposures at radius L/2, σ_axis = L·G·F/(4√N), G² = K·(L/2)²/Σ p_axis² — √2
# for any planar orbit (the lateral form above), 1 for a z-line pair, √3 on every
# axis for an octahedron (``PatternGeometry.geometry_factors``).
#
# Verified, same inputs: SimuFLUX's numerical CRB (``calculateCRBpattern``) for
# 3/4/6-point orbits with and without a centre exposure, L 40–150 nm, ρ 0–3,
# agrees with these closed forms to ≤ 0.16 % (180 scenarios; the residual is its
# 1-nm finite difference and the 1e-4 it adds to each probability).
#
# Background. Their Eq. 44 (lateral) and Eq. 28 (axial) multiply the ideal bound
# by √[(1 + 1/SBR)(1 + w/SBR)], with SBR the total signal over the total
# background summed over the exposures and w the orbit's share of the total
# dwell (3/4 for Eq. 44's triangle + centre, 2/3 for Eq. 28's two points +
# centre). Re-deriving the multinomial Fisher information for the same quadratic
# minimum shows the factor depends on the pattern only through that share, so
# it holds for every Abberior pattern (hexagon, square, zline2) once w is taken
# from the recorded centre dwell. Written with ρ = background ÷ signal rate at an
# orbit position and c = centre dwell ÷ orbit dwell (0 without a centre):
#
#   σ = σ_ideal · √[(1 + ρ)(1 + ρ(1 + c))],   SBR = 1 / (ρ(1 + c)),   w = 1/(1 + c).
#
# ρ comes from the data, per localization, as a background RATE b set against
# the photon iteration's orbit rate: ρ = b / (efo − b). b is, in order:
#   * the photon iteration's own centre exposure, b = efc. With the emitter at
#     the pattern centre the centre exposure sees no fluorescence signal, so
#     everything it records is background — including the light that leaks
#     through an imperfect donut zero, which Marin & Ries identify as a
#     background that limits precision. This is the preferred estimate.
#   * the latest EARLIER centre exposure, scaled by the laser power: a 3-D
#     imaging sequence probes the centre at an intermediate lateral iteration
#     (ccrLimit set) and not in its final pair, at a lower power (pwrFactor 4
#     against 6). Background — autofluorescence and the zero leakage alike —
#     scales with the excitation power, so b = efc_k · P_final / P_k. The
#     instrument's own estimate obeys the same scaling (3_Bonly: fbg rises
#     3.62e4 → 5.42e4 Hz, ×1.50, between those iterations — exactly the power
#     ratio — and fbg ≈ efc where both are recorded).
#   * the instrument's background estimate, b = fbg — only with background
#     sensing on (``bgcSense``); with it off fbg is 0 or a placeholder.
#   * none → no background information: the bound reported is the ideal one
#     and is labelled as such, never as "background-limited".
#
# The pattern size comes from the acquisition sequence (L = patGeoFactor ×
# 360 nm, the conversion SimuFLUX uses for Abberior sequences); the centre dwell
# from ``ctrDwellFactor`` — or, without a sequence, from the counts themselves:
# c = (ecc/eco) ÷ (efc/efo).

CRLB_DEFAULT_L_NM: float = DEFAULT_FINAL_GEO_FACTOR * PATTERN_SCALE_NM  # 39.6 nm, assumed
#: σ_q of SimuFLUX's analytic 2-D donut (``PsfDonut2D``: FWHM 310 nm, which its
#: authors matched to their vectorial PSF) written as Marin & Ries' σ_q. The
#: paper's ~250 nm is an illustrative value for its 1-D quadratic; σ_q enters only
#: the envelope term (1 − L²/8σ_q²)⁻¹, ≈1 % at L = 40 nm and ≈4 % at 76 nm.
CRLB_DEFAULT_SIGMA_Q_NM: float = float(310.0 / np.sqrt(8.0 * np.log(2.0)))  # ≈ 131.6 nm
_SQRT2 = float(np.sqrt(2.0))
CRLB_DEFAULT_LZ_NM: float = DEFAULT_FINAL_GEO_FACTOR * PATTERN_SCALE_NM  # 39.6 nm, assumed
_CENTRE_DWELL_RANGE = (0.0, 20.0)       # sanity clip for a dwell ratio read off the counts
#: An earlier centre exposure is borrowed only when it was recorded for at least
#: this share of the photon iteration's localizations.
_REFERENCE_MIN_COVERAGE = 0.5

#: Per-localization background sources (``CRLBResult.per_loc_bg_source``).
BG_NONE, BG_CENTRE, BG_FBG, BG_CENTRE_EARLIER = 0, 1, 2, 3
# Plain tokens (no parentheses, commas or semicolons): they are written into the
# Log line that the method-text generator parses.
_BG_SOURCE_TEXT = {BG_CENTRE: "centre exposure", BG_FBG: "fbg estimate",
                   BG_CENTRE_EARLIER: "earlier centre exposure"}


@dataclass
class CRLBResult:
    per_loc_sigma_ideal: np.ndarray   # lateral σ_xy without background, nm
    per_loc_sigma_bg: np.ndarray      # lateral σ_xy with the measured background, nm (NaN: unknown)
    per_loc_n_total: np.ndarray       # N_xy = eco + ecc at the final lateral iteration
    per_loc_sbr: np.ndarray           # SBR as Marin & Ries define it (NaN: unknown)
    median_sigma_ideal: float
    mean_sigma_ideal: float
    median_sigma_bg: float            # over localizations WITH background information
    mean_sigma_bg: float
    median_n: float
    median_sbr: float                 # NaN if no background data
    has_background: bool
    L_nm: float
    sigma_q_nm: float
    n_locs: int
    status: str
    # Axial (z) channel — populated only when ``has_z`` (3-D, L_z given).
    has_z: bool = False
    L_z_nm: float = 0.0
    per_loc_sigma_z_ideal: np.ndarray = field(default_factory=lambda: np.empty(0))
    per_loc_sigma_z_bg: np.ndarray = field(default_factory=lambda: np.empty(0))
    median_sigma_z_ideal: float = float("nan")
    mean_sigma_z_ideal: float = float("nan")
    median_sigma_z_bg: float = float("nan")
    mean_sigma_z_bg: float = float("nan")
    # N_z = eco + ecc at the final axial iteration (equals median_n unless a
    # distinct per-dimension photon count was supplied).
    median_n_z: float = float("nan")
    per_dimension_photons: bool = False   # True when N_xy and N_z differ by design
    # Background provenance (lateral, and axial when has_z).
    per_loc_bg_source: np.ndarray = field(default_factory=lambda: np.empty(0, np.int8))
    background_source: str = "none"      # "centre exposure", "fbg estimate", or both joined by " + "
    background_coverage: float = 0.0      # fraction of localizations with background info
    centre_dwell_ratio: float = 0.0       # median c used for the centre exposure (0 = none)
    has_background_z: bool = False
    background_source_z: str = "none"
    background_coverage_z: float = 0.0
    median_sbr_z: float = float("nan")
    centre_dwell_ratio_z: float = 0.0
    # Where the pattern sizes came from ("acquisition sequence" / "assumed ...").
    L_source: str = "user input"
    L_z_source: str = "user input"
    pattern: str = ""
    pattern_z: str = ""
    counts_centre: bool = False         # N includes the centre exposure (ecc)
    # Where an earlier centre exposure was borrowed from, e.g. "iteration 7 scaled
    # x1.50 by the laser power" ("" when not used), per channel.
    centre_reference: str = ""
    centre_reference_z: str = ""
    # σ = L·G·F/(4√N) per axis, and whether the 2-D donut envelope term applies.
    geometry_factor: float = _SQRT2
    geometry_factor_z: float = 1.0
    donut_envelope: bool = True
    # Things about the inputs the reader should know (e.g. an fbg not used).
    background_notes: tuple = ()
    background_pooled: bool = False     # rates pooled per trace before ρ
    axis_status: str = "ideal_only"
    axis_status_z: str = "ideal_only"
    count_convention: str = "detected"
    issues: tuple = ()
    invalid_background_fraction: float = 0.0
    invalid_background_fraction_z: float = 0.0
    input_rows: int = 0
    source_indices: np.ndarray = field(default_factory=lambda: np.empty(0, dtype=np.int64))
    excluded_photon_rows: int = 0
    provenance: dict = field(default_factory=dict)

    @property
    def primary_sigma(self) -> float:
        """Conditional headline; inconsistent inputs never fall back to ideal."""
        if self.axis_status in {"invalid_input", "unsupported_geometry", "background_inconsistent"}:
            return float("nan")
        return self.median_sigma_bg if self.has_background else self.median_sigma_ideal

    @property
    def primary_sigma_z(self) -> float:
        if self.axis_status_z in {"invalid_input", "unsupported_geometry", "background_inconsistent"}:
            return float("nan")
        return self.median_sigma_z_bg if self.has_background_z else self.median_sigma_z_ideal


def _aligned(arr, finite: np.ndarray) -> np.ndarray | None:
    """Restrict an aligned input; a malformed vector is never treated as absent."""
    if arr is None:
        return None
    a = np.asarray(arr, dtype=float)
    if a.shape != finite.shape:
        raise ValueError("CRLB input arrays must be one-dimensional and row-aligned with eco.")
    return a[finite]


def _group_median(values: np.ndarray, groups: np.ndarray) -> np.ndarray:
    """Each element replaced by the median of the finite values in its group.

    Elements of a group with no finite value stay NaN. Vectorised: one sort by
    (group, value), then each group's middle element(s).
    """
    out = np.full(values.shape, np.nan)
    fin = np.isfinite(values)
    if not fin.any():
        return out
    keys, inverse = np.unique(groups, return_inverse=True)
    g, v = inverse[fin], values[fin]
    order = np.lexsort((v, g))
    g_sorted, v_sorted = g[order], v[order]
    starts = np.r_[0, np.flatnonzero(np.diff(g_sorted)) + 1]
    counts = np.diff(np.r_[starts, g_sorted.size])
    lo = starts + (counts - 1) // 2
    hi = starts + counts // 2
    med = np.full(keys.size, np.nan)
    med[g_sorted[starts]] = 0.5 * (v_sorted[lo] + v_sorted[hi])
    out[:] = med[inverse]
    return out


def _channel_background(eco, ecc, efo, fbg, efc, centre_dwell_ratio, efc_ref=None,
                        groups=None, fbg_zero_measured=False):
    """Per-localization (ρ, c, source) for one channel; see the module comment.

    ``eco``/``ecc``/``efo``/``fbg``/``efc``/``efc_ref`` are already restricted to
    the valid rows (all but ``eco`` may be ``None``). ``efc_ref`` is an earlier
    iteration's centre rate, already scaled to this iteration's laser power.
    ``centre_dwell_ratio`` is the recorded centre dwell ÷ orbit dwell, or ``None``
    to read it off the counts.

    ``groups`` (trace ids): the background rate b and the orbit rate are pooled
    per trace (median) before ρ = b / (efo − b). One localization's rates rest on
    ~100 photons, so its own ρ is noisy — and F is convex in ρ, so the noise
    inflates σ, while every localization whose noisy b reaches its efo is lost,
    and those are the worst-SBR ones. Pooling assumes stable rates within a
    trace; it does not establish that the background really is constant.
    """
    n = eco.size
    rho = np.full(n, np.nan)
    c = np.zeros(n)
    src = np.zeros(n, dtype=np.int8)
    inconsistent = np.zeros(n, dtype=bool)
    if efo is None:
        return rho, c, src, float("nan"), inconsistent
    pooled = groups is not None and np.shape(groups) == (n,)

    # Centre dwell ratio read off the counts: (ecc/eco) ÷ (efc/efo).
    c_data = float("nan")
    if efc is not None and ecc is not None:
        with np.errstate(divide="ignore", invalid="ignore"):
            est = (ecc * efo) / (eco * efc)
        est = est[np.isfinite(est) & (est > 0)]
        if est.size:
            c_data = float(np.clip(np.median(est), *_CENTRE_DWELL_RANGE))
    c_centre = (float(centre_dwell_ratio) if centre_dwell_ratio is not None
                else (c_data if np.isfinite(c_data) else 1.0))

    def _own_centre_share(ok):
        # This iteration's centre exposure, when it has one, still costs dwell
        # (and collects background) even where its rate is unusable.
        if centre_dwell_ratio is not None:
            return float(centre_dwell_ratio)
        centre_seen = ecc is not None and (ecc[ok] > 0)
        return np.where(centre_seen, c_centre, 0.0)

    # Select the first recorded source, then judge consistency. A contradictory
    # centre rate must not silently turn into a lower fbg or an ideal bound.
    b = np.full(n, np.nan)
    for rate, kind in ((efc, BG_CENTRE), (efc_ref, BG_CENTRE_EARLIER), (fbg, BG_FBG)):
        if rate is None:
            continue
        nonnegative = rate >= 0 if kind == BG_FBG and fbg_zero_measured else rate > 0
        ok = (src == BG_NONE) & np.isfinite(rate) & nonnegative
        b[ok] = rate[ok]
        c[ok] = c_centre if kind == BG_CENTRE else _own_centre_share(ok)
        src[ok] = kind

    orbit = efo
    if pooled:
        known = src != BG_NONE
        b = np.where(known, _group_median(np.where(known, b, np.nan), groups), np.nan)
        orbit = _group_median(np.where(np.isfinite(efo), efo, np.nan), groups)
    ok = (src != BG_NONE) & np.isfinite(orbit) & np.isfinite(b) & (orbit > b)
    inconsistent = ((src != BG_NONE) & np.isfinite(orbit) & np.isfinite(b)
                    & (b >= orbit))
    rho[ok] = b[ok] / (orbit[ok] - b[ok])
    src[~ok] = BG_NONE
    c[~ok] = 0.0
    for kind in (BG_CENTRE, BG_CENTRE_EARLIER, BG_FBG):
        used = src == kind
        if used.any():
            return rho, c, src, float(np.median(c[used])), inconsistent
    return rho, c, src, 0.0, inconsistent


def _source_text(src: np.ndarray, centre_reference: str = "") -> str:
    kinds = [k for k in (BG_CENTRE, BG_CENTRE_EARLIER, BG_FBG) if np.any(src == k)]
    if not kinds:
        return "none"

    def _text(k):
        if k == BG_CENTRE_EARLIER and centre_reference:
            return f"centre exposure of {centre_reference}"
        return _BG_SOURCE_TEXT[k]
    return " + ".join(_text(k) for k in kinds)


def crlb_precision(
    eco: np.ndarray,
    *,
    L_nm: float = CRLB_DEFAULT_L_NM,
    sigma_q_nm: float = CRLB_DEFAULT_SIGMA_Q_NM,
    efo: np.ndarray | None = None,
    fbg: np.ndarray | None = None,
    ecc: np.ndarray | None = None,
    efc: np.ndarray | None = None,
    centre_dwell_ratio: float | None = None,
    L_z_nm: float | None = None,
    eco_z: np.ndarray | None = None,
    efo_z: np.ndarray | None = None,
    fbg_z: np.ndarray | None = None,
    ecc_z: np.ndarray | None = None,
    efc_z: np.ndarray | None = None,
    centre_dwell_ratio_z: float | None = None,
    L_source: str = "user input",
    L_z_source: str = "user input",
    pattern: str = "",
    pattern_z: str = "",
    efc_ref: np.ndarray | None = None,
    efc_ref_z: np.ndarray | None = None,
    centre_reference: str = "",
    centre_reference_z: str = "",
    geometry_factor_xy: float = _SQRT2,
    geometry_factor_z: float = 1.0,
    donut_envelope: bool = True,
    background_notes: tuple = (),
    groups: np.ndarray | None = None,
    fbg_zero_measured: bool = False,
    fbg_zero_measured_z: bool = False,
    count_convention: str = "detected",
    unsupported_xy: str = "",
    unsupported_z: str = "",
) -> CRLBResult:
    """Background-aware MINFLUX Cramér-Rao precision bound, per localization.

    Lateral (Marin & Ries, Eq. 45 × the background factor of Eq. 44, generalised
    to the recorded centre dwell)::

        σ_xy = L/√(8N) · (1 − L²/(8σ_q²))⁻¹ · √[(1+ρ)(1+ρ(1+c))]

    Axial, when ``L_z_nm`` is given (Eq. 29 × the factor of Eq. 28)::

        σ_z = L_z/(4√N) · √[(1+ρ)(1+ρ(1+c))]

    Parameters
    ----------
    eco, ecc : (N,) float
        Photons at the offset (pattern) positions and at the centre exposure;
        N = eco + ecc. ``ecc`` may be omitted (no centre exposure).
    L_nm : float
        Lateral pattern diameter L of the final iteration, nm.
    sigma_q_nm : float
        Donut width σ_q of the Marin & Ries PSF model (default 131.6 nm). Requires
        L < 2√2·σ_q.
    efo, efc, fbg : (N,) float, optional
        Rate at the offset positions, rate at the centre, and the instrument's
        background estimate. The background-to-signal ratio ρ is taken from the
        centre exposure when recorded, else from ``fbg``; without either the
        background-limited bound is NaN and only the ideal one is reported.
    centre_dwell_ratio : float, optional
        Centre dwell ÷ whole-orbit dwell (``ctrDwellFactor`` when the centre is
        probed, 0 when it is not). ``None`` reads it off the counts.
    L_z_nm, eco_z, ecc_z, efo_z, efc_z, fbg_z, centre_dwell_ratio_z
        The same for the final axial iteration. Without ``eco_z`` the axial
        channel reuses the lateral inputs.
    efc_ref, efc_ref_z : (N,) float, optional
        An earlier iteration's centre rate, already scaled to this iteration's
        laser power — used where the iteration has no centre exposure of its own,
        ahead of ``fbg``. ``centre_reference`` names where it came from.
    geometry_factor_xy, geometry_factor_z, donut_envelope
        The pattern's G (σ = L·G·F/(4√N)) and whether the 2-D donut envelope
        term applies; the defaults are a planar orbit and a z-line pair.
    groups : (N,) array, optional
        Trace ids: the background and orbit rates are then pooled per trace
        (median) before ρ is formed — see :func:`_channel_background`.
    fbg_zero_measured, fbg_zero_measured_z : bool
        Include exact fbg zeros as observations (also in trace pooling), only
        when background sensing is known to be on. This never promotes a zero
        centre exposure to a measured zero background.
    count_convention : {"detected", "unknown"}
        Direct callers supply detected counts/total rates by contract. Dataset
        inputs remain conditional until instrument conventions are established;
        no count-restoration conversion is performed.
    """
    n_full = np.asarray(eco, dtype=float)
    if n_full.ndim != 1:
        raise ValueError("eco must be a one-dimensional photon-count vector.")
    if count_convention not in {"detected", "unknown"}:
        raise ValueError("count_convention must be 'detected' or 'unknown'.")
    for value in (ecc, efo, fbg, efc, efc_ref, eco_z, ecc_z, efo_z, fbg_z, efc_z,
                  efc_ref_z, groups):
        if value is not None and np.shape(value) != n_full.shape:
            raise ValueError("CRLB input arrays must be one-dimensional and row-aligned with eco.")
    finite = np.isfinite(n_full) & (n_full > 0)
    eco_v = n_full[finite]
    n_locs = int(eco_v.size)

    ecc_v = _aligned(ecc, finite)
    n = eco_v + (np.where(np.isfinite(ecc_v) & (ecc_v > 0), ecc_v, 0.0)
                 if ecc_v is not None else 0.0)

    L = float(L_nm)
    sq = float(sigma_q_nm)
    Lz = None if L_z_nm is None or L_z_nm <= 0 else float(L_z_nm)
    # An unknown count convention is carried by ``count_convention`` itself; the
    # dialog and the method text state it from that field, so it is not repeated
    # here in other words.
    issues = list(background_notes)
    if ecc_v is not None and np.any(np.isfinite(ecc_v) & (ecc_v < 0)):
        issues.append("Negative ecc values indicate correction/bookkeeping rather than detected "
                      "centre photons; they are not added to N. The count convention is uncertain.")
        count_convention = "unknown"

    def _empty(status: str) -> CRLBResult:
        z = np.empty(0)
        return CRLBResult(
            per_loc_sigma_ideal=z, per_loc_sigma_bg=z, per_loc_n_total=z,
            per_loc_sbr=z, median_sigma_ideal=float("nan"),
            mean_sigma_ideal=float("nan"), median_sigma_bg=float("nan"),
            mean_sigma_bg=float("nan"), median_n=float("nan"),
            median_sbr=float("nan"), has_background=False, L_nm=L,
            sigma_q_nm=sq, n_locs=n_locs, status=status,
            has_z=Lz is not None, L_z_nm=Lz or 0.0,
            L_source=L_source, L_z_source=L_z_source, pattern=pattern,
            pattern_z=pattern_z,
            axis_status="invalid_input", axis_status_z="invalid_input",
            count_convention=count_convention, issues=tuple(issues + [status]),
            input_rows=n_full.size, source_indices=np.flatnonzero(finite),
            excluded_photon_rows=int((~finite).sum()),
        )

    if n_locs == 0:
        return _empty("No localizations with a positive photon count (eco).")
    if not (np.isfinite(L) and L > 0):
        return _empty("Pattern diameter L must be positive.")
    if not (np.isfinite(sq) and sq > 0):
        return _empty("Donut width σ_q must be positive.")

    # PSF-gradient correction (1 − L²/(8σ_q²))⁻¹ of the 2-D donut — requires
    # L < 2√2·σ_q.
    grad_denom = 1.0 - (L * L) / (8.0 * sq * sq) if donut_envelope else 1.0
    if grad_denom <= 0:
        return _empty(
            f"σ_q ({sq:.0f} nm) is too small relative to L ({L:.0f} nm): the "
            "donut model requires L < 2√2·σ_q ≈ "
            f"{2.0 * np.sqrt(2.0) * sq:.0f} nm. Increase σ_q or decrease L.")
    if Lz is not None and not np.isfinite(Lz):
        return _empty("Axial pattern diameter must be finite.")
    for ratio in (centre_dwell_ratio, centre_dwell_ratio_z):
        if ratio is not None and (not np.isfinite(ratio) or ratio < 0):
            return _empty("Centre dwell ratio must be finite and nonnegative.")
    G = float(geometry_factor_xy)
    Gz = float(geometry_factor_z)
    if not (np.isfinite(G) and G > 0):
        unsupported_xy = unsupported_xy or "No supported lateral geometry factor."
    if Lz is not None and not (np.isfinite(Gz) and Gz > 0):
        unsupported_z = unsupported_z or "No supported axial geometry factor."
    # An unsupported pattern's geometry factor is unknown, so even the
    # zero-background benchmark is undefined: never the planar-orbit default.
    if unsupported_xy:
        G = float("nan")
    if unsupported_z:
        Gz = float("nan")

    sigma_ideal = L * G / (4.0 * np.sqrt(n) * grad_denom)

    groups_v = None
    if groups is not None:
        g = np.asarray(groups).ravel()
        groups_v = g[finite] if g.shape == finite.shape else None
    rho, c, src, c_used, inconsistent = _channel_background(
        eco_v, ecc_v, _aligned(efo, finite), _aligned(fbg, finite),
        _aligned(efc, finite), centre_dwell_ratio, _aligned(efc_ref, finite), groups_v,
        fbg_zero_measured)
    known = np.isfinite(rho) & np.isfinite(sigma_ideal)
    factor = np.sqrt((1.0 + rho) * (1.0 + rho * (1.0 + c)))
    sigma_bg = np.where(known, sigma_ideal * factor, np.nan)
    with np.errstate(divide="ignore", invalid="ignore"):
        sbr = np.where(known, 1.0 / (rho * (1.0 + c)), np.nan)
    has_bg = bool(known.any())

    # Axial z bound (Eq. 29 × Eq. 28's factor), 3-D only.
    z_fields: dict = {"per_dimension_photons": eco_z is not None}
    if Lz is not None:
        if eco_z is not None:
            ez = _aligned(eco_z, finite)
            ecc_zv = _aligned(ecc_z, finite)
            efo_zv, fbg_zv, efc_zv = (_aligned(efo_z, finite), _aligned(fbg_z, finite),
                                      _aligned(efc_z, finite))
            efc_ref_zv = _aligned(efc_ref_z, finite)
            c_ratio_z = centre_dwell_ratio_z
            ref_z = centre_reference_z
        else:   # reuse the lateral inputs (single-iteration data)
            ez, ecc_zv = eco_v, ecc_v
            efo_zv, fbg_zv, efc_zv = (_aligned(efo, finite), _aligned(fbg, finite),
                                      _aligned(efc, finite))
            efc_ref_zv = _aligned(efc_ref, finite)
            c_ratio_z = centre_dwell_ratio
            ref_z = centre_reference
        n_z = ez + (np.where(np.isfinite(ecc_zv) & (ecc_zv > 0), ecc_zv, 0.0)
                    if ecc_zv is not None else 0.0)
        with np.errstate(invalid="ignore", divide="ignore"):
            sigma_z_ideal = np.where(n_z > 0, Lz * Gz / (4.0 * np.sqrt(n_z)), np.nan)
        rho_z, c_z, src_z, c_used_z, inconsistent_z = _channel_background(
            ez, ecc_zv, efo_zv, fbg_zv, efc_zv, c_ratio_z, efc_ref_zv, groups_v,
            fbg_zero_measured_z if eco_z is not None else fbg_zero_measured)
        known_z = np.isfinite(rho_z) & np.isfinite(sigma_z_ideal)
        sigma_z_bg = np.where(
            known_z, sigma_z_ideal * np.sqrt((1.0 + rho_z) * (1.0 + rho_z * (1.0 + c_z))),
            np.nan)
        with np.errstate(divide="ignore", invalid="ignore"):
            sbr_z = np.where(known_z, 1.0 / (rho_z * (1.0 + c_z)), np.nan)
        with np.errstate(invalid="ignore"):
            z_fields.update(
                has_z=True, L_z_nm=Lz,
                per_loc_sigma_z_ideal=sigma_z_ideal,
                per_loc_sigma_z_bg=sigma_z_bg,
                median_sigma_z_ideal=(float(np.nanmedian(sigma_z_ideal))
                                      if np.isfinite(sigma_z_ideal).any() else float("nan")),
                mean_sigma_z_ideal=(float(np.nanmean(sigma_z_ideal))
                                    if np.isfinite(sigma_z_ideal).any() else float("nan")),
                median_sigma_z_bg=(float(np.nanmedian(sigma_z_bg)) if known_z.any()
                                   else float("nan")),
                mean_sigma_z_bg=(float(np.nanmean(sigma_z_bg)) if known_z.any()
                                 else float("nan")),
                median_n_z=float(np.nanmedian(n_z)) if np.isfinite(n_z).any() else float("nan"),
                has_background_z=bool(known_z.any()),
                background_source_z=_source_text(src_z, ref_z),
                centre_reference_z=ref_z if np.any(src_z == BG_CENTRE_EARLIER) else "",
                background_coverage_z=float(known_z.mean()) if n_z.size else 0.0,
                median_sbr_z=(float(np.nanmedian(sbr_z)) if known_z.any() else float("nan")),
                centre_dwell_ratio_z=c_used_z,
                invalid_background_fraction_z=float(inconsistent_z.mean()),
                axis_status_z=("unsupported_geometry" if unsupported_z else
                               "background_estimated" if known_z.any() else
                               "background_inconsistent" if inconsistent_z.any() else
                               "invalid_input" if not np.isfinite(sigma_z_ideal).any() else
                               "ideal_only"),
            )
    elif eco_z is not None:
        ez = _aligned(eco_z, finite)
        if ez is not None and ez.size:
            z_fields["median_n_z"] = float(np.nanmedian(ez))

    return CRLBResult(
        per_loc_sigma_ideal=sigma_ideal,
        per_loc_sigma_bg=sigma_bg,
        per_loc_n_total=n,
        per_loc_sbr=sbr,
        median_sigma_ideal=float(np.median(sigma_ideal)),
        mean_sigma_ideal=float(sigma_ideal.mean()),
        median_sigma_bg=float(np.nanmedian(sigma_bg)) if has_bg else float("nan"),
        mean_sigma_bg=float(np.nanmean(sigma_bg)) if has_bg else float("nan"),
        median_n=float(np.median(n)),
        median_sbr=float(np.nanmedian(sbr)) if has_bg else float("nan"),
        has_background=has_bg,
        L_nm=L,
        sigma_q_nm=sq,
        n_locs=n_locs,
        status="",
        per_loc_bg_source=src,
        background_source=_source_text(src, centre_reference),
        centre_reference=centre_reference if np.any(src == BG_CENTRE_EARLIER) else "",
        geometry_factor=G,
        geometry_factor_z=Gz,
        donut_envelope=bool(donut_envelope),
        background_notes=tuple(background_notes),
        background_pooled=groups_v is not None,
        background_coverage=float(known.mean()),
        centre_dwell_ratio=c_used,
        L_source=L_source,
        L_z_source=L_z_source,
        pattern=pattern,
        pattern_z=pattern_z,
        counts_centre=bool(ecc_v is not None and np.any(np.isfinite(ecc_v) & (ecc_v > 0))),
        axis_status=("unsupported_geometry" if unsupported_xy else
                     "background_estimated" if has_bg else
                     "background_inconsistent" if inconsistent.any() else "ideal_only"),
        count_convention=count_convention,
        issues=tuple(dict.fromkeys(issues + [v for v in (unsupported_xy, unsupported_z) if v])),
        invalid_background_fraction=float(inconsistent.mean()),
        input_rows=n_full.size, source_indices=np.flatnonzero(finite),
        excluded_photon_rows=int((~finite).sum()),
        **z_fields,
    )


# ---------------------------------------------------------------------------
# Precision budget — measured spread vs the photon-limited bound
# ---------------------------------------------------------------------------

def excess_precision(std_nm: float, crb_nm: float) -> float:
    """Legacy arithmetic helper, not an identified noise component.

    A quadrature difference is meaningful only for matched variances and a
    justified additive, independent, attained-bound model. It is not used by
    CRLB reporting. A below-bound value is unresolved (NaN), not zero error.
    """
    s = float(std_nm)
    c = float(crb_nm)
    if not (np.isfinite(s) and np.isfinite(c)) or s <= 0.0 or c < 0 or s < c:
        return float("nan")
    return float(np.sqrt(max(s * s - c * c, 0.0)))


# ---------------------------------------------------------------------------
# UI launchers — these are what the menu items call
# ---------------------------------------------------------------------------

def _precision_role(ds, prefs=None) -> tuple[str | None, str]:
    """Conservative static-analysis gate, including datasets never viewed as tracks.

    The displacement classifier is a heuristic, not an acquisition-mode proof.
    A tracking-like sequence name also withholds static inference (closed paths
    can end near their start). An explicit user role takes precedence.
    """
    from minflux_viewer.core.loader_raw_access import attr_values_1d

    from ..core.mfx_sequence import sequence_for_dataset
    from ..core.tracks import build_track_set, dataset_tracking_role, looks_like_tracking
    role, reason, explicit = dataset_tracking_role(ds)
    if explicit or role == "Tracking":
        return role, reason
    seq = sequence_for_dataset(ds) or {}
    name = str(seq.get("id", seq.get("name", "")))
    if "track" in name.lower() or "seqtrk" in name.lower():
        return "Tracking", f"tracking-like sequence name '{name}' (hint, not confirmed acquisition mode)"
    values = [attr_values_1d(ds, key) for key in ("loc_x", "loc_y", "tid", "tim")]
    if any(v is None for v in values):
        return None, "insufficient coordinates, trace IDs or times to exclude tracking"
    x, y, tid, tim = values
    z = attr_values_1d(ds, "loc_z")
    coords = np.column_stack([x, y, np.zeros_like(x) if z is None else z]) * 1e9
    opts = (prefs or {}).get("tracking", {})
    try:
        tracks = build_track_set(coords, tid, tim)
        moving, reason = looks_like_tracking(
            tracks, displacement_nm=opts.get("role_displacement_nm", 25.0),
            min_median_locs=opts.get("role_min_median_locs", 5))
    except ValueError as exc:
        return None, str(exc)
    return ("Tracking" if moving else "Structure"), "fresh-load heuristic: " + reason


def _measured_trace_precision(ds) -> dict | None:
    """Median per-trace measured σ (StdDev per trace, raw z) for the CRLB dialog's
    descriptive use only; tracking or an unknown role returns ``None``."""
    role, _reason = _precision_role(ds)
    if role != "Structure":
        return None
    try:
        from minflux_viewer.core.loader_raw_access import attr_values_1d
        lx = np.asarray(attr_values_1d(ds, "loc_x"), dtype=float)
        ly = np.asarray(attr_values_1d(ds, "loc_y"), dtype=float)
        _z = attr_values_1d(ds, "loc_z")
        lz = np.zeros_like(lx) if _z is None else np.asarray(_z, dtype=float)
        _tid = attr_values_1d(ds, "tid")
    except (AttributeError, IndexError, KeyError, TypeError, ValueError):
        return None
    if _tid is None or lx.size == 0:
        return None
    tid = np.asarray(_tid)
    ftr = np.asarray(ds.filter_mask, dtype=bool)
    if ftr.shape[0] == lx.shape[0]:
        lx, ly, lz, tid = lx[ftr], ly[ftr], lz[ftr], tid[ftr]
    sd = stddev_per_trace(np.column_stack([lx, ly, lz]), tid)
    if sd["n_traces_used"] == 0:
        return None
    return {
        "sigma_r": float(sd["median_sigma_r"]),
        "sigma_z": float(sd["median_sigma_xyz"][2]),
        "n_traces": int(sd["n_traces_used"]),
    }


_CRLB_FIELDS = ("eco", "ecc", "efo", "efc", "fbg")


def crlb_inputs_for_dataset(ds, *, prefs=None) -> dict | None:
    """Everything the CRB needs, read from *ds* (Qt-free).

    Photon counts and rates are gathered per channel from the iteration that
    localized it — the final lateral iteration for σ_xy, the final axial one for
    σ_z — and laid onto the last-valid localizations. When ``ds.attr`` *is* that
    materialization the interactive filter applies. The pattern geometry comes
    from the acquisition sequence when the file records it, otherwise it is
    assumed and reported as such. ``None`` when the dataset has no photon counts.
    """
    from minflux_viewer.core.loader_raw_access import (
        attr_matches_last_valid,
        iteration_attr_values_1d,
        mfx_get,
    )

    from ..core.mfx_sequence import (
        background_sensing,
        centre_reference_iteration,
        crlb_geometry_for_dataset,
        iteration_power,
        sequence_for_dataset,
    )

    if ds.attr.get("eco") is None and ds.mfx_raw.get("eco") is None:
        return None
    use_attr = attr_matches_last_valid(ds) and ds.attr.get("eco") is not None
    ftr = np.asarray(ds.filter_mask, dtype=bool) if use_attr else None
    has_raw = len(ds.mfx_raw) > 0

    def _materialized(name: str):
        if use_attr:
            v = ds.attr.get(name)
            if v is None:
                return None
            v = np.asarray(v, dtype=float).ravel()
            return v[ftr] if ftr.shape[0] == v.shape[0] else v
        v = mfx_get(ds, name, itr="last", vld_only=True)
        return None if v is None else np.asarray(v, dtype=float).ravel()

    def _at(name: str, iteration):
        if iteration is not None and has_raw:
            v = iteration_attr_values_1d(ds, name, iteration, itr="last", vld_only=True)
            if v is not None:
                v = np.asarray(v, dtype=float).ravel()
                if ftr is not None and ftr.shape[0] == v.shape[0]:
                    v = v[ftr]
                return v
            return None   # never substitute an effective/materialized OTHER iteration
        return _materialized(name)

    def _strict(name: str, iteration: int):
        # This iteration's values only — no fallback to the materialized view.
        try:
            v = iteration_attr_values_1d(ds, name, iteration, itr="last", vld_only=True)
        except (AttributeError, IndexError, KeyError, TypeError, ValueError):
            return None
        if v is None:
            return None
        v = np.asarray(v, dtype=float).ravel()
        if ftr is not None and ftr.shape[0] == v.shape[0]:
            v = v[ftr]
        return v

    sequence = sequence_for_dataset(ds)
    sensing = background_sensing(sequence)

    def _channel(iteration) -> dict:
        ch = {k: _at(k, iteration) for k in _CRLB_FIELDS}
        ch["efc_ref"], ch["centre_reference"], ch["notes"] = None, "", []
        ch["fbg_zero_measured"] = sensing is True
        ch["iteration"] = iteration
        if sensing is False and ch.get("fbg") is not None:
            ch["fbg"] = None
            ch["notes"].append(
                "fbg not used: background sensing (bgcSense) was off in this sequence, "
                "so the file's fbg is a placeholder, not a measurement")
        own = ch.get("efc")
        own_ok = own is not None and bool(np.any(np.isfinite(own) & (own > 0)))
        k = centre_reference_iteration(sequence, iteration)
        if not own_ok and k is not None and iteration is not None and k != iteration:
            ref = _strict("efc", k)
            p_now, p_ref = iteration_power(sequence, iteration), iteration_power(sequence, k)
            # The reference must actually cover the localizations: an m2205
            # file keeps hardly any values from its intermediate iterations, and
            # a few stray rows must not stand in for the whole dataset.
            if (ref is not None and p_now and p_ref
                    and float(np.mean(np.isfinite(ref) & (ref > 0))) >= _REFERENCE_MIN_COVERAGE):
                scale = p_now / p_ref
                ch["efc_ref"] = ref * scale
                # The power-scaling assumption this rests on is stated by the
                # dialog and the method text wherever ``centre_reference`` is
                # used, so it is not added to the notes a second time.
                ch["centre_reference"] = (
                    f"iteration {k + 1} scaled x{scale:.2f} by the laser power")
                if sensing is True:
                    ref_counts = _strict("ecc", k)
                    ref_bg = _strict("fbg", k)
                    with np.errstate(divide="ignore", invalid="ignore"):
                        dwell = None if ref_counts is None else ref_counts / ref
                    dt = (dwell[np.isfinite(dwell) & (dwell > 0)]
                          if dwell is not None else np.empty(0))
                    low = np.zeros(ref.shape, dtype=bool)
                    if dt.size and ref_bg is not None:
                        expected = ref_bg * float(np.median(dt))
                        low = ((ref_counts > 0) & (ref_counts < 0.5 * expected)
                               & (expected >= 10))
                    zeros = np.isfinite(ref) & (ref == 0)
                    if zeros.any() or low.any():
                        ch["notes"].append(
                            f"bgcSense-on centre exposure at iteration {k + 1}: "
                            f"{int(zeros.sum())} exact zeros and {int(low.sum())} low-tail rows "
                            "(0 < ecc < half fbg-predicted counts, expected >= 10). "
                            "Centre count/rate convention and zero meaning are unresolved; "
                            "zeros are not used as centre background estimates.")
        return ch

    pit, geo_xy, geo_z = crlb_geometry_for_dataset(ds)
    num_dim = int(getattr(ds.prop, "num_dim", 2) or 2)
    tid = _materialized("tid")
    xy = _channel(pit.lateral_iter)
    z = None
    if num_dim >= 3 and geo_z is not None:
        # One iteration localizing every axis (an octahedron) serves both.
        z = xy if pit.axial_iter == pit.lateral_iter else _channel(pit.axial_iter)
    else:
        geo_z = None
    role, reason = _precision_role(ds, prefs)
    result = {"xy": xy, "z": z, "geometry_xy": geo_xy, "geometry_z": geo_z,
              "photon_iterations": pit, "num_dim": num_dim, "tid": tid,
              "count_convention": "unknown", "source_version": ds.metadata.get("source_version"),
              "comparison_role": role, "comparison_reason": reason}
    if role == "Structure" and tid is not None:
        x, y, zz = (_materialized(k) for k in ("loc_x", "loc_y", "loc_z"))
        if x is not None and y is not None:
            result["comparison_loc_xyz"] = np.column_stack(
                [x, y, np.zeros_like(x) if zz is None else zz])
            result["comparison_tid"] = tid
    return result


def _recorded_centre_ratio(geometry) -> float | None:
    """The sequence's centre dwell ratio when it says the centre was probed.

    ``None`` otherwise, so the counts decide (a recorded centre exposure is
    used whatever the sequence claims; its dwell is then read off the counts).
    """
    if geometry is not None and geometry.from_sequence and geometry.centre_probed:
        return geometry.centre_dwell_ratio
    return None


def crlb_from_inputs(inputs: dict, *, L_nm: float | None = None,
                     sigma_q_nm: float = CRLB_DEFAULT_SIGMA_Q_NM,
                     L_z_nm: float | None = None) -> CRLBResult:
    """:func:`crlb_precision` on :func:`crlb_inputs_for_dataset`'s output.

    ``L_nm`` / ``L_z_nm`` override the recorded (or assumed) pattern sizes; an
    override that differs from the recorded value is reported as user input.
    """
    gxy, gz = inputs["geometry_xy"], inputs["geometry_z"]
    xy, z = inputs["xy"], inputs["z"]

    def _size(override, geometry):
        if override is None or abs(float(override) - geometry.L_nm) < 1e-6:
            return geometry.L_nm, geometry.source
        return float(override), "user input"

    L, L_src = _size(L_nm, gxy)
    g_xy = gxy.geometry_factors[0]
    notes = list(xy.get("notes") or ())
    kwargs = dict(L_nm=L, sigma_q_nm=sigma_q_nm, efo=xy["efo"], fbg=xy["fbg"],
                  ecc=xy["ecc"], efc=xy["efc"],
                  centre_dwell_ratio=_recorded_centre_ratio(gxy),
                  L_source=L_src, pattern=gxy.pattern,
                  efc_ref=xy.get("efc_ref"), centre_reference=xy.get("centre_reference", ""),
                  geometry_factor_xy=g_xy, unsupported_xy=gxy.unsupported_reason,
                  donut_envelope=gxy.donut_envelope,
                  fbg_zero_measured=xy.get("fbg_zero_measured", False),
                  count_convention=inputs.get("count_convention", "detected"))
    if z is not None and gz is not None:
        Lz, Lz_src = _size(L_z_nm, gz)
        g_z = gz.geometry_factors[1]
        kwargs.update(L_z_nm=Lz, L_z_source=Lz_src, pattern_z=gz.pattern,
                      geometry_factor_z=g_z,
                      unsupported_z=gz.unsupported_reason)
        # Plain array inputs deliberately assume a z-line; they do not record a pattern.
        if not gz.pattern and gz.supported:
            kwargs["geometry_factor_z"] = 1.0
        if z is not xy:     # a distinct axial iteration — its own photons/background
            kwargs.update(eco_z=(z["eco"] if z.get("eco") is not None else
                                 np.full(np.shape(xy["eco"]), np.nan)),
                          efo_z=z["efo"], fbg_z=z["fbg"],
                          ecc_z=z["ecc"], efc_z=z["efc"],
                          centre_dwell_ratio_z=_recorded_centre_ratio(gz),
                          efc_ref_z=z.get("efc_ref"),
                          fbg_zero_measured_z=z.get("fbg_zero_measured", False),
                          centre_reference_z=z.get("centre_reference", ""))
            notes += [n for n in (z.get("notes") or ()) if n not in notes]
    kwargs["background_notes"] = tuple(notes)
    kwargs["groups"] = inputs.get("tid")
    result = crlb_precision(xy["eco"], **kwargs)
    result.provenance = {
        "source_version": inputs.get("source_version"),
        "comparison_reason": inputs.get("comparison_reason", ""),
        "xy_iteration": xy.get("iteration"),
        "z_iteration": z.get("iteration") if z else None,
        "orbit_interpretation": "detected counts and total rates; no background restoration",
        "centre_interpretation": "conditional; bgcSense-on zero/low-tail convention unresolved",
    }
    return result


def crlb_matched_comparison(result: CRLBResult, inputs: dict) -> dict | None:
    """Descriptive matched trace medians, never a variance decomposition.

    For each axis select rows with finite coordinates AND a finite bound, then
    keep traces with at least five such rows. Compare each trace's sample SD
    with sqrt(mean(bound**2)) on exactly those rows, and take equal-weight
    trace medians. Filtering, invalid photons and missing background therefore
    affect both sides identically. XY is per-coordinate RMS, not radial SD.
    """
    loc = inputs.get("comparison_loc_xyz")
    tid = inputs.get("comparison_tid")
    if loc is None or tid is None or result.status:
        return None
    loc, tid = np.asarray(loc, float), np.asarray(tid)
    if loc.shape != (result.input_rows, 3) or tid.shape != (result.input_rows,):
        raise ValueError("CRLB comparison coordinates and trace IDs must match the input rows.")
    loc, tid = loc[result.source_indices], tid[result.source_indices]
    out = {}
    channels = [("xy", result.axis_status, result.has_background,
                 result.per_loc_sigma_bg, result.per_loc_sigma_ideal, (0, 1))]
    if result.has_z:
        channels.append(("z", result.axis_status_z, result.has_background_z,
                         result.per_loc_sigma_z_bg, result.per_loc_sigma_z_ideal, (2,)))
    for axis, status, has_bg, bg, ideal, columns in channels:
        if status in {"invalid_input", "unsupported_geometry", "background_inconsistent"}:
            continue
        sig = bg if has_bg else ideal
        valid = np.all(np.isfinite(loc[:, columns]), axis=1) & np.isfinite(sig)
        if not valid.any():
            continue
        coords = np.zeros((int(valid.sum()), 3))
        coords[:, columns] = loc[valid][:, columns]
        ids, codes = np.unique(tid[valid], return_inverse=True)
        count = np.bincount(codes)
        keep = count >= MIN_LOCS_PER_TRACE
        if not keep.any():
            continue
        bounds = np.sqrt(np.bincount(codes, weights=np.square(sig[valid])) / count)
        spread = stddev_per_trace(coords, tid[valid])
        measured = spread["per_trace_sigma_r"] if axis == "xy" else spread["per_trace_sigma_xyz"][:, 2]
        m, b = float(np.median(measured)), float(np.median(bounds[keep]))
        n_locs = int(count[keep].sum())
        out[axis] = {"measured_nm": m, "bound_nm": b, "ratio": m / b if b > 0 else None,
                     "n_traces": int(keep.sum()), "n_locs": n_locs,
                     "coverage": n_locs / result.input_rows if result.input_rows else 0.0,
                     "bound_kind": "background estimated" if has_bg else "ideal only"}
    return out or None


# ---------------------------------------------------------------------------
# Result dialog
# ---------------------------------------------------------------------------


# ---------------------------------------------------------------------------
# FRC result dialog
# ---------------------------------------------------------------------------


# ---------------------------------------------------------------------------
# CRLB result dialog
# ---------------------------------------------------------------------------


def crlb_inputs_from_arrays(
    eco, *, efo=None, fbg=None, ecc=None, efc=None, num_dim: int = 2,
    L_nm: float = CRLB_DEFAULT_L_NM, L_z_nm: float = CRLB_DEFAULT_LZ_NM,
    eco_z=None, efo_z=None, fbg_z=None, ecc_z=None, efc_z=None, tid=None,
) -> dict:
    """Inputs in :func:`crlb_inputs_for_dataset`'s shape, from plain arrays.

    For scripting and tests: no sequence is involved, so the pattern sizes are
    reported as assumed and the centre dwell (if any) is read off the counts.
    ``tid`` (trace ids) pools the background per trace, as for a dataset.
    """
    from ..core.mfx_sequence import PatternGeometry

    def _arr(v):
        return None if v is None else np.asarray(v, dtype=float)

    xy = {"eco": _arr(eco), "ecc": _arr(ecc), "efo": _arr(efo), "efc": _arr(efc),
          "fbg": _arr(fbg)}
    z = None
    gz = None
    if int(num_dim) >= 3:
        # Without axial arrays the axial channel reuses the lateral ones —
        # the very same dict, which crlb_from_inputs recognises.
        z = xy if eco_z is None else {
            "eco": _arr(eco_z), "ecc": _arr(ecc_z), "efo": _arr(efo_z),
            "efc": _arr(efc_z), "fbg": _arr(fbg_z)}
        gz = PatternGeometry(L_nm=float(L_z_nm))
    return {"xy": xy, "z": z, "geometry_xy": PatternGeometry(L_nm=float(L_nm)),
            "geometry_z": gz, "photon_iterations": None, "num_dim": int(num_dim),
            "tid": None if tid is None else np.asarray(tid).ravel()}


# ---------------------------------------------------------------------------
# Internal helper
# ---------------------------------------------------------------------------


_MOVED_SYMBOLS = {'run_crlb': 'minflux_viewer.ui.precision_analysis:run_crlb', 'run_frc': 'minflux_viewer.ui.precision_analysis:run_frc', 'run_stddev_per_trace': 'minflux_viewer.ui.precision_analysis:run_stddev_per_trace', '_precision_color': 'minflux_viewer.ui.precision_analysis:_precision_color', 'StdDevResultDialog': 'minflux_viewer.ui.precision_analysis:StdDevResultDialog', '_frc_curve_plot': 'minflux_viewer.ui.precision_analysis:_frc_curve_plot', 'FRCResultDialog': 'minflux_viewer.ui.precision_analysis:FRCResultDialog', '_res': 'minflux_viewer.ui.precision_analysis:_res', '_references_html': 'minflux_viewer.ui.precision_analysis:_references_html', '_clear_layout': 'minflux_viewer.ui.precision_analysis:_clear_layout', '_pattern_text': 'minflux_viewer.ui.precision_analysis:_pattern_text', 'CRLBResultDialog': 'minflux_viewer.ui.precision_analysis:CRLBResultDialog', 'InfoDialog': 'minflux_viewer.ui.precision_analysis:InfoDialog', '_show_info_dialog': 'minflux_viewer.ui.precision_analysis:_show_info_dialog', '_try_journal': 'minflux_viewer.ui.precision_analysis:_try_journal'}

def __getattr__(name):
    return resolve_moved_symbol(__name__, name, _MOVED_SYMBOLS)
