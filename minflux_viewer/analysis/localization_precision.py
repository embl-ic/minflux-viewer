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
  with at least :data:`MIN_LOCS_PER_TRACE` repeats (nm; raw z, no RIMF). Also
  written to ``dataset.attr`` for follow-up work.

* **CRLB** — the *theoretical* MINFLUX Cramér-Rao bound for the targeted-donut
  measurement model (Balzarotti et al., *Science* 355:606, 2017; Marin & Ries,
  arXiv:2410.12427, 2024) — **not** the camera/Gaussian bound (Mortensen 2010).
  Per-localization σ from the photon count (``eco``), the measured SBR
  (``efo``/``fbg``) and the user's pattern diameter ``L`` / donut steepness ``σ_q``.

* **FRC** — Fourier Ring Correlation (Banterle et al., *J. Struct. Biol.*
  183:363, 2013; Nieuwenhuizen et al., *Nat. Methods* 10:557, 2013): the radial
  Fourier ring correlation of two independent half-images, resolution read off at
  the 1/7 crossing.

The CRLB dialog also reports a **precision budget** relating the two: by the
variance decomposition of Marin & Ries (*Nat. Commun.* 17:246, 2026, SimuFLUX),
``STD(measured)² = σ_fl² + σ_CRB²``, so the excess ``σ_fl = √(STD² − σ_CRB²)``
is the photon-count-independent error (flickering / drift / vibration /
misalignment) beyond the photon-limited bound (see :func:`excess_precision`).
"""

from __future__ import annotations

from dataclasses import dataclass, field
from functools import lru_cache

import numpy as np
from PyQt6.QtCore import Qt
from PyQt6.QtWidgets import (
    QComboBox,
    QDialog,
    QDialogButtonBox,
    QDoubleSpinBox,
    QGridLayout,
    QGroupBox,
    QHBoxLayout,
    QLabel,
    QPushButton,
    QSizePolicy,
    QVBoxLayout,
    QWidget,
)

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
    Compute the per-trace localization precision (Schmidt et al. 2022).

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
        coordinates (no RIMF applied to z). Output is reported in nm.
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
# This mirrors the Ries-group MINFLUX implementation of Gwosch et al. (Nat.
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

    Uses multithreaded ``scipy.fft`` when available, else NumPy.
    """
    try:
        from scipy.fft import rfft2
        return rfft2(img, workers=-1)
    except Exception:                       # pragma: no cover - scipy present
        return np.fft.rfft2(img)


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
    """Savitzky-Golay smoothing of the radial FRC curve (Ries-group default)."""
    if y.size < 7:
        return y
    try:
        from scipy.signal import savgol_filter
        return savgol_filter(y, 7, 1)
    except Exception:                       # pragma: no cover - scipy always present
        return np.convolve(y, np.ones(3) / 3, mode="same")


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

    Follows the Ries-group MINFLUX method (Gwosch et al. 2020; Ostersehlt et al.
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
# et al. (Science 355:606, 2017). We use the background-aware closed form of
# Marin & Ries (arXiv:2410.12427, 2024) for the standard 2-D donut (K = 4 TCPs)
# with the emitter at the pattern centre (their Eq. 44 / 45):
#
#   σ_CRB = L / (2·√(2N)) · (1 − L²·ln2/σ_q²)⁻¹ · √[(1 + 1/SBR)(1 + 3/(4·SBR))]
#   σ_ideal (no background, SBR→∞) = L / (2·√(2N)) · (1 − L²·ln2/σ_q²)⁻¹
#
# Parameters and where each comes from:
#   N    photons per localization                       — taken from data (eco)
#   SBR  signal-to-background ratio = (efo − fbg)/fbg    — computed from data
#   L    targeted-pattern diameter of the last iteration — user input (default)
#   σ_q  excitation-PSF steepness / donut FWHM (≳250 nm) — user input (default)
#
# σ_q sets the PSF-gradient correction (1 − L²·ln2/σ_q²)⁻¹; it requires L < σ_q.
# The bound is the per-coordinate lateral precision at the centre (best case →
# a genuine lower bound). Background enters through the measured SBR — replacing
# the earlier ad-hoc effective-photon (1 − fbg/efo) correction.
#
# Axial (z) channel — 3-D only. The z position is probed with a 1-D quadratic
# minimum (K = 3 points at −L_z/2, 0, +L_z/2; Marin & Ries Eq. 28). Factoring L_z
# out and substituting b = L_z²/(12·σ_qz²·SBR) makes σ_qz cancel, leaving
#
#   σ_z = L_z/(4·√N) · √[(1 + 1/SBR)(1 + 2/(3·SBR))]   ;  σ_z,ideal = L_z/(4·√N).
#
# It reuses the same N and the same measured (lateral) SBR — an approximation,
# since the axial exposures may carry a different SBR — and an editable axial
# pattern size L_z. No σ_qz is needed (it cancels in the SBR form).

CRLB_DEFAULT_L_NM: float = 50.0        # last-iteration pattern diameter L (editable)
CRLB_DEFAULT_SIGMA_Q_NM: float = 250.0  # donut steepness σ_q, diffraction-bounded
CRLB_DEFAULT_LZ_NM: float = 50.0       # axial pattern size L_z (editable, 3-D)
_LN2 = float(np.log(2.0))


@dataclass
class CRLBResult:
    per_loc_sigma_ideal: np.ndarray   # lateral σ_xy at SBR→∞ (no background), nm
    per_loc_sigma_bg: np.ndarray      # lateral σ_xy with the measured per-loc SBR, nm
    per_loc_n_total: np.ndarray       # N_xy = eco at the final lateral iteration
    per_loc_sbr: np.ndarray           # (efo − fbg)/fbg per localization
    median_sigma_ideal: float
    mean_sigma_ideal: float
    median_sigma_bg: float
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
    # N_z = eco at the final axial iteration (equals median_n unless a distinct
    # per-dimension photon count was supplied).
    median_n_z: float = float("nan")
    per_dimension_photons: bool = False   # True when N_xy and N_z differ by design


def crlb_precision(
    eco: np.ndarray,
    *,
    L_nm: float = CRLB_DEFAULT_L_NM,
    sigma_q_nm: float = CRLB_DEFAULT_SIGMA_Q_NM,
    efo: np.ndarray | None = None,
    fbg: np.ndarray | None = None,
    L_z_nm: float | None = None,
    eco_z: np.ndarray | None = None,
    efo_z: np.ndarray | None = None,
    fbg_z: np.ndarray | None = None,
) -> CRLBResult:
    """Background-aware MINFLUX Cramér-Rao precision bound, per localization.

    Lateral (xy) — 2-D donut (K = 4), emitter at the centre (Marin & Ries 2024,
    Eq. 44 / 45):

        σ_xy = L/(2√(2N)) · (1 − L²·ln2/σ_q²)⁻¹ · √[(1+1/SBR)(1+3/(4·SBR))].

    Axial (z), when ``L_z_nm`` is given (3-D) — 1-D quadratic minimum, K = 3
    (Eq. 28 → SBR form):

        σ_z = L_z/(4√N) · √[(1+1/SBR)(1+2/(3·SBR))].

    Parameters
    ----------
    eco : (N,) float
        Photon count per localization (effective counts) → N.
    L_nm : float
        Targeted-pattern diameter L of the (last) iteration, in nm.
    sigma_q_nm : float
        Excitation-PSF steepness / donut FWHM σ_q (diffraction-bounded), in nm.
        Must exceed L·√ln2 for the quadratic approximation to hold.
    efo, fbg : (N,) float, optional
        Photon rate and background rate. When both are present the per-loc
        SBR = (efo − fbg)/fbg degrades the bound; otherwise only σ_ideal is given.
    L_z_nm : float, optional
        Axial pattern size L_z. When given (> 0) the axial z bound is also
        computed (reusing N and the lateral SBR; σ_qz cancels in the SBR form).
    """
    n_full = np.asarray(eco, dtype=float).ravel()
    finite = np.isfinite(n_full) & (n_full > 0)
    n = n_full[finite]
    n_locs = int(n.size)
    # Axial photon count N_z (final axial iteration). Defaults to the lateral N
    # → backward-compatible single-eco behaviour. Per-dimension when a distinct
    # eco_z is supplied (final lateral vs final axial iteration photons differ).
    per_dim = eco_z is not None
    if per_dim:
        n_z_full = np.asarray(eco_z, dtype=float).ravel()
        n_z = n_z_full[finite] if n_z_full.shape == finite.shape else n
    else:
        n_z = n
    L = float(L_nm)
    sq = float(sigma_q_nm)
    Lz = None if L_z_nm is None or L_z_nm <= 0 else float(L_z_nm)

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
        )

    if n_locs == 0:
        return _empty("No localizations with a positive photon count (eco).")
    if not (L > 0):
        return _empty("Pattern diameter L must be positive.")
    if not (sq > 0):
        return _empty("PSF steepness σ_q must be positive.")

    # PSF-gradient correction (1 − L²·ln2/σ_q²)⁻¹ — requires L < σ_q/√ln2.
    grad_denom = 1.0 - (L * L * _LN2) / (sq * sq)
    status = ""
    if grad_denom <= 0:
        return _empty(
            f"σ_q ({sq:.0f} nm) is too small relative to L ({L:.0f} nm): the "
            "quadratic-PSF approximation requires L < σ_q·√ln2 ≈ "
            f"{sq * np.sqrt(_LN2):.0f} nm. Increase σ_q or decrease L.")

    # σ_ideal: no-background limit (Eq. 45), per localization via N.
    sigma_ideal = L / (2.0 * np.sqrt(2.0 * n) * grad_denom)

    # SBR from the measured rates (per localization). Lateral uses efo/fbg;
    # axial uses efo_z/fbg_z when given (final axial iteration rates), else efo/fbg.
    def _sbr(efo_arr, fbg_arr):
        has = False
        sbr_p = np.full(n_locs, np.nan)
        okm = np.zeros(n_locs, dtype=bool)
        vals = np.empty(0)
        if efo_arr is not None and fbg_arr is not None:
            ef = np.asarray(efo_arr, dtype=float).ravel()
            bg = np.asarray(fbg_arr, dtype=float).ravel()
            if ef.shape == finite.shape and bg.shape == finite.shape:
                ef = ef[finite]
                bg = bg[finite]
                okm = np.isfinite(ef) & np.isfinite(bg) & (bg > 0) & (ef > bg)
                if okm.any():
                    has = True
                    vals = np.clip((ef[okm] - bg[okm]) / bg[okm], 1e-3, None)
                    sbr_p[okm] = vals
        return has, okm, vals, sbr_p

    has_bg, ok, sbr, sbr_per = _sbr(efo, fbg)
    if per_dim and (efo_z is not None or fbg_z is not None):
        has_bg_z, ok_z, sbr_z, _ = _sbr(
            efo_z if efo_z is not None else efo, fbg_z if fbg_z is not None else fbg)
    else:
        has_bg_z, ok_z, sbr_z = has_bg, ok, sbr

    # Lateral xy bound — background degradation (Eq. 44).
    sigma_bg = sigma_ideal.copy()
    if has_bg:
        factor_xy = np.sqrt((1.0 + 1.0 / sbr) * (1.0 + 3.0 / (4.0 * sbr)))
        sigma_bg[ok] = sigma_ideal[ok] * factor_xy

    # Axial z bound (1-D quadratic minimum, Eq. 28 → SBR form), 3-D only.
    if Lz is not None:
        with np.errstate(invalid="ignore", divide="ignore"):
            sigma_z_ideal = Lz / (4.0 * np.sqrt(n_z))     # NaN where N_z invalid
        sigma_z_bg = sigma_z_ideal.copy()
        if has_bg_z:
            factor_z = np.sqrt((1.0 + 1.0 / sbr_z) * (1.0 + 2.0 / (3.0 * sbr_z)))
            sigma_z_bg[ok_z] = sigma_z_ideal[ok_z] * factor_z
        with np.errstate(invalid="ignore"):
            z_fields = dict(
                has_z=True, L_z_nm=Lz,
                per_loc_sigma_z_ideal=sigma_z_ideal,
                per_loc_sigma_z_bg=sigma_z_bg,
                median_sigma_z_ideal=float(np.nanmedian(sigma_z_ideal)),
                mean_sigma_z_ideal=float(np.nanmean(sigma_z_ideal)),
                median_sigma_z_bg=float(np.nanmedian(sigma_z_bg)),
                mean_sigma_z_bg=float(np.nanmean(sigma_z_bg)),
                median_n_z=float(np.nanmedian(n_z)) if n_z.size else float("nan"),
                per_dimension_photons=bool(per_dim),
            )
    else:
        z_fields = dict(
            has_z=False, L_z_nm=0.0,
            median_n_z=float(np.nanmedian(n_z)) if (per_dim and n_z.size) else float("nan"),
            per_dimension_photons=bool(per_dim),
        )

    sbr_finite = sbr_per[np.isfinite(sbr_per)]
    return CRLBResult(
        per_loc_sigma_ideal=sigma_ideal,
        per_loc_sigma_bg=sigma_bg,
        per_loc_n_total=n,
        per_loc_sbr=sbr_per,
        median_sigma_ideal=float(np.median(sigma_ideal)),
        mean_sigma_ideal=float(sigma_ideal.mean()),
        median_sigma_bg=float(np.median(sigma_bg)),
        mean_sigma_bg=float(sigma_bg.mean()),
        median_n=float(np.median(n)),
        median_sbr=float(np.median(sbr_finite)) if sbr_finite.size else float("nan"),
        has_background=has_bg,
        L_nm=L,
        sigma_q_nm=sq,
        n_locs=n_locs,
        status=status,
        **z_fields,
    )


# ---------------------------------------------------------------------------
# Precision budget — measured spread vs the photon-limited bound
# ---------------------------------------------------------------------------

def excess_precision(std_nm: float, crb_nm: float) -> float:
    """Excess (non-photon-limited) localization error σ_fl beyond the CRB.

    From the variance decomposition of Marin & Ries (*Nat. Commun.* 17:246,
    2026, SimuFLUX), the measured per-localization spread combines the
    photon-limited Cramér-Rao bound with a photon-count-**independent** error
    from fluorophore flickering, drift, vibration and misalignment:

        STD(r, N)² = σ_fl(r)² + σ_CRB(N)²   ⇒   σ_fl = √( STD² − σ_CRB² ).

    Returns σ_fl in nm — ``0`` when the measured spread is at or below the bound
    (i.e. the localization is essentially photon-limited), or NaN if either input
    is not finite / not positive.
    """
    s = float(std_nm)
    c = float(crb_nm)
    if not (np.isfinite(s) and np.isfinite(c)) or s <= 0.0:
        return float("nan")
    return float(np.sqrt(max(s * s - c * c, 0.0)))


# ---------------------------------------------------------------------------
# UI launchers — these are what the menu items call
# ---------------------------------------------------------------------------

def _measured_trace_precision(ds) -> dict | None:
    """Median per-trace measured σ (StdDev per trace, raw z) for the CRLB dialog's
    precision budget; ``None`` when there are no usable traces / no trace IDs."""
    try:
        from ..core.loader import attr_values_1d
        lx = np.asarray(attr_values_1d(ds, "loc_x"), dtype=float)
        ly = np.asarray(attr_values_1d(ds, "loc_y"), dtype=float)
        _z = attr_values_1d(ds, "loc_z")
        lz = np.zeros_like(lx) if _z is None else np.asarray(_z, dtype=float)
        _tid = attr_values_1d(ds, "tid")
    except Exception:
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


def run_crlb(parent, state) -> None:
    """Compute the MINFLUX CRB precision for the active dataset, show a dialog."""
    from ..core.loader import attr_matches_last_valid, mfx_get

    ds = state.active_dataset
    if ds is None:
        return

    if ds.attr.get("eco") is None and ds.mfx_raw.get("eco") is None:
        _show_info_dialog(
            parent, "CRLB — Cramér-Rao bound",
            "Dataset has no photon-count attribute (<b>eco</b>), which the "
            "MINFLUX Cramér-Rao bound requires. Load a MINFLUX dataset that "
            "carries effective counts.")
        return

    # N (eco) and the rates (efo, fbg) must come from the SAME measurement —
    # the last (smallest-L) iteration, valid only, consistent with L. When the
    # materialized ``ds.attr`` already IS that selection we use it (so the
    # interactive filter applies); otherwise route through the raw store's
    # last-valid selection (as the anisotropy estimator does).
    use_attr = attr_matches_last_valid(ds) and ds.attr.get("eco") is not None
    ftr = np.asarray(ds.filter_mask, dtype=bool) if use_attr else None

    if use_attr:
        def _col(name: str):
            v = ds.attr.get(name)
            if v is None:
                return None
            v = np.asarray(v, dtype=float).ravel()
            return v[ftr] if ftr.shape[0] == v.shape[0] else v
    else:
        def _col(name: str):
            v = mfx_get(ds, name, itr="last", vld_only=True)
            return None if v is None else np.asarray(v, dtype=float).ravel()

    # ``_col`` gives the materialized (final = axial iteration) per-loc values.
    eco = _col("eco")          # N_z: photons at the final axial (z) iteration
    efo = _col("efo")
    fbg = _col("fbg")
    num_dim = int(getattr(ds.prop, "num_dim", 2) or 2)

    # Per-dimension photon budget: the lateral (xy) CRLB must use the photons of
    # the final LATERAL iteration, not the materialized final (axial) one. Detect
    # the lateral/axial iterations and gather the lateral eco aligned to the same
    # last-valid rows; fall back to the single materialized eco otherwise.
    eco_xy = None
    try:
        from ..core.loader import iteration_attr_values_1d
        from ..core.mfx_sequence import photon_iterations_for_dataset
        pit = photon_iterations_for_dataset(ds)
        if (pit.lateral_iter is not None and pit.axial_iter is not None
                and pit.lateral_iter != pit.axial_iter):
            v = iteration_attr_values_1d(ds, "eco", pit.lateral_iter,
                                         itr="last", vld_only=True)
            if v is not None:
                v = np.asarray(v, dtype=float).ravel()
                if use_attr and ftr is not None and ftr.shape[0] == v.shape[0]:
                    v = v[ftr]
                eco_xy = v
    except Exception:
        eco_xy = None

    if eco_xy is not None and eco is not None and eco_xy.shape == eco.shape:
        # N_xy = final lateral iteration photons; N_z = final axial (materialized).
        result = crlb_precision(
            eco_xy, L_nm=CRLB_DEFAULT_L_NM, sigma_q_nm=CRLB_DEFAULT_SIGMA_Q_NM,
            efo=efo, fbg=fbg,
            L_z_nm=CRLB_DEFAULT_LZ_NM if num_dim >= 3 else None,
            eco_z=eco)
        eco = eco_xy      # display / budget use the lateral photon count
    else:
        result = crlb_precision(
            eco, L_nm=CRLB_DEFAULT_L_NM, sigma_q_nm=CRLB_DEFAULT_SIGMA_Q_NM,
            efo=efo, fbg=fbg,
            L_z_nm=CRLB_DEFAULT_LZ_NM if num_dim >= 3 else None)
    # Measured per-trace spread → the CRLB dialog's precision budget (measured vs
    # the photon-limited bound; excess σ_fl per Marin & Ries 2026).
    measured = _measured_trace_precision(ds)

    if np.isfinite(result.median_sigma_bg):
        z_txt = (f", σ_z = {result.median_sigma_z_bg:.2f} nm (L_z = "
                 f"{result.L_z_nm:.0f} nm)" if result.has_z else "")
        budget_txt = ""
        if measured is not None:
            fl_r = excess_precision(measured["sigma_r"], result.median_sigma_bg)
            budget_txt = (f"; measured σ_r = {measured['sigma_r']:.2f} nm "
                          f"(StdDev/trace) → excess σ_fl = {fl_r:.2f} nm")
        if result.per_dimension_photons and np.isfinite(result.median_n_z):
            n_txt = (f"median N = {result.median_n:.0f} photons (final lateral "
                     f"iteration; axial N_z = {result.median_n_z:.0f})")
        else:
            n_txt = f"median N = {result.median_n:.0f} photons"
        _try_journal(
            state,
            f"Localization precision (CRLB, Marin-Ries): median σ_xy = "
            f"{result.median_sigma_bg:.2f} nm (background-limited), "
            f"{result.median_sigma_ideal:.2f} nm (ideal){z_txt}, "
            f"L = {result.L_nm:.0f} nm, σ_q = {result.sigma_q_nm:.0f} nm, "
            f"{n_txt}{budget_txt}."
        )

    from ..ui.modeless import show_modeless
    show_modeless(
        CRLBResultDialog(result, eco, efo, fbg, num_dim=num_dim, measured=measured),
        parent,
    )


def run_frc(parent, state) -> None:
    """Compute the FRC resolution of the active dataset and show a dialog."""
    ds = state.active_dataset
    if ds is None:
        return
    from ..core.dataset_kind import missing_reason, require
    miss = require(ds, "loc")
    if miss:
        _show_info_dialog(parent, "FRC — Fourier Ring Correlation",
                          f"This analysis requires {missing_reason(miss)}.")
        return

    from ..core.loader import attr_values_1d
    loc_x = attr_values_1d(ds, "loc_x")
    loc_y = attr_values_1d(ds, "loc_y")

    x = np.asarray(loc_x, dtype=float) * 1.0e9    # m → nm
    y = np.asarray(loc_y, dtype=float) * 1.0e9
    tid = ds.attr.get("tid")
    tid = None if tid is None else np.asarray(tid)
    ftr = np.asarray(ds.filter_mask, dtype=bool)
    if ftr.shape[0] == x.shape[0]:
        x, y = x[ftr], y[ftr]
        if tid is not None:
            tid = tid[ftr]

    with state.task(f"Computing FRC of dataset {ds.name}…") as t:
        result = frc_resolution(
            x, y, tid, progress=lambda done, total: t.update(done / total)
        )

    if np.isfinite(result.resolution_nm):
        _try_journal(
            state,
            f"Localization precision (FRC): resolution = "
            f"{result.resolution_nm:.2f} nm (1/7 threshold, {result.mode}, "
            f"{result.n_points:,} points, pixel {result.pixel_size_nm:.2f} nm)."
        )

    # Modeless result window (project convention for analysis output).
    from ..ui.modeless import show_modeless
    show_modeless(FRCResultDialog(result, x, y, tid), parent)


def run_stddev_per_trace(parent, state) -> None:
    """Compute σ per trace for the active dataset and show a results dialog."""
    ds = state.active_dataset
    if ds is None:
        return
    from ..core.dataset_kind import missing_reason, require
    miss = require(ds, "loc", "traces")
    if miss:
        _show_info_dialog(
            parent, "Localization Precision — StdDev per trace",
            f"This analysis requires {missing_reason(miss)}.")
        return

    from ..core.loader import attr_values_1d
    loc_x = np.asarray(attr_values_1d(ds, "loc_x"))
    loc_y = np.asarray(attr_values_1d(ds, "loc_y"))
    # Raw z — no RIMF applied. The precision is a property of the measurement,
    # so it follows Ostersehlt et al. and uses the uncorrected coordinate.
    _z = attr_values_1d(ds, "loc_z")
    loc_z = np.zeros_like(loc_x) if _z is None else np.asarray(_z)
    _tid = attr_values_1d(ds, "tid")
    tid   = np.arange(loc_x.size) if _tid is None else np.asarray(_tid)
    ftr   = np.asarray(ds.filter_mask, dtype=bool)

    loc_xyz = np.column_stack([loc_x[ftr], loc_y[ftr], loc_z[ftr]])
    tid_f   = tid[ftr]

    result = stddev_per_trace(loc_xyz, tid_f)

    # Stash the per-trace σ on the dataset for re-use
    try:
        ds.attr["sigma_per_trace_nm"] = result["per_trace_sigma_xyz"]
        ds.attr["sigma_trace_ids"]    = result["trace_ids"]
        ds.derived["sigma_per_trace_nm"] = result["per_trace_sigma_xyz"]
        ds.derived["sigma_trace_ids"]    = result["trace_ids"]
    except Exception:
        pass

    _try_journal(
        state,
        f"Localization precision (StdDev per trace): "
        f"combined (n-weighted) sigma_r = {result['combined_sigma_r']:.2f} nm, "
        f"sigma_z = {result['combined_sigma_xyz'][2]:.2f} nm "
        f"over {result['n_traces_used']:,} of {result['n_traces_total']:,} traces "
        f"(>={MIN_LOCS_PER_TRACE} loc/trace, raw z)."
    )

    # Modeless result window (project convention for analysis output).
    from ..ui.modeless import show_modeless
    show_modeless(StdDevResultDialog(result), parent)


# ---------------------------------------------------------------------------
# Result dialog
# ---------------------------------------------------------------------------

_R_COLOR = "#22d3d3"   # σ_r (combined lateral) — cyan
_Z_COLOR = "#ff5a5a"   # σ_z (axial) — red


class StdDevResultDialog(QDialog):
    """Localization-precision report (Schmidt et al.), styled like anisotropy.

    Reports the n-weighted combined precision σ_r (lateral) and σ_z (axial),
    with per-trace σ_r / σ_z distributions and a per-axis X/Y/R(xy)/Z table.
    """

    def __init__(self, result: dict, parent=None) -> None:
        super().__init__(parent)
        self.setWindowTitle("Localization Precision — StdDev per trace")
        self.setModal(False)
        self.resize(880, 640)

        root = QVBoxLayout(self)
        root.setContentsMargins(10, 10, 10, 10)
        root.setSpacing(8)

        n_used  = result["n_traces_used"]
        n_total = result["n_traces_total"]
        med = result["median_sigma_xyz"]
        mean = result["mean_sigma_xyz"]

        if n_used == 0:
            root.addWidget(QLabel(
                f"<b>No traces had at least {MIN_LOCS_PER_TRACE} localizations.</b><br><br>"
                f"Total traces in selection: {n_total:,}<br><br>"
                "Try widening the filter, or increase the number of repeats per trace."
            ))
            bb = QDialogButtonBox(QDialogButtonBox.StandardButton.Close)
            bb.rejected.connect(self.reject)
            bb.button(QDialogButtonBox.StandardButton.Close).clicked.connect(self.close)
            root.addWidget(bb)
            return

        comb   = result["combined_sigma_xyz"]
        med_r  = result["median_sigma_r"]
        mean_r = result["mean_sigma_r"]
        comb_r = result["combined_sigma_r"]

        info = QLabel(
            f"<b>{n_used:,}</b> of {n_total:,} traces used "
            f"(≥ {MIN_LOCS_PER_TRACE} loc/trace, raw coordinates)"
        )
        info.setAlignment(Qt.AlignmentFlag.AlignCenter)
        root.addWidget(info)

        headline = QLabel(
            "<span style='font-size:18px'><b>localization precision: "
            f"σ<sub>r</sub> = {comb_r:.2f} nm, "
            f"σ<sub>z</sub> = {comb[2]:.2f} nm</b></span>"
            "<span style='color:gray'> &nbsp;(n-weighted)</span>"
        )
        headline.setAlignment(Qt.AlignmentFlag.AlignCenter)
        headline.setToolTip(
            "σ_r = sqrt((σ_x² + σ_y²)/2);  reported value = mean over traces of σ/sqrt(n)"
        )
        root.addWidget(headline)

        # Per-axis table — X / Y / R(xy) / Z, with the combined row reported.
        grp = QGroupBox("Localization precision across traces (nm) — combined (n-weighted) is reported")
        grid = QGridLayout(grp)
        grid.setHorizontalSpacing(20)
        grid.setVerticalSpacing(4)
        for col, h in enumerate(("", "X", "Y", "R(xy)", "Z")):
            lbl = QLabel(f"<b>{h}</b>")
            lbl.setAlignment(
                Qt.AlignmentFlag.AlignLeft if col == 0 else Qt.AlignmentFlag.AlignCenter
            )
            grid.addWidget(lbl, 0, col)
        rows = (
            ("Median",      [med[0],  med[1],  med_r,  med[2]],  False),
            ("Mean",        [mean[0], mean[1], mean_r, mean[2]], False),
            ("Combined ★",  [comb[0], comb[1], comb_r, comb[2]], True),
        )
        for r, (row_lbl, vals, bold) in enumerate(rows, start=1):
            name = QLabel(f"<b>{row_lbl}:</b>" if bold else f"{row_lbl}:")
            name.setAlignment(Qt.AlignmentFlag.AlignLeft)
            grid.addWidget(name, r, 0)
            for col, v in enumerate(vals):
                t = f"{v:.2f} nm"
                cell = QLabel(f"<b>{t}</b>" if bold else t)
                cell.setAlignment(Qt.AlignmentFlag.AlignCenter)
                grid.addWidget(cell, r, col + 1)
        for col in range(1, 5):
            grid.setColumnStretch(col, 1)
        root.addWidget(grp)

        # Per-trace distributions — σ_r (lateral) and σ_z (axial), median marked.
        try:
            from .trace_analysis import _value_hist_plot
            hist_group = QGroupBox("Per-trace σ distributions: R(xy) and Z")
            row = QHBoxLayout(hist_group)
            row.setContentsMargins(8, 8, 8, 8)
            row.addWidget(_value_hist_plot(
                "σ<sub>r</sub> per trace (lateral)",
                result["per_trace_sigma_r"], _R_COLOR,
                marker=float(med_r), marker_label=f"median = {med_r:.2f} nm",
                x_label="σ<sub>r</sub> (nm)", y_label="traces",
            ))
            row.addWidget(_value_hist_plot(
                "σ<sub>z</sub> per trace (axial)",
                result["per_trace_sigma_xyz"][:, 2], _Z_COLOR,
                marker=float(med[2]), marker_label=f"median = {med[2]:.2f} nm",
                x_label="σ<sub>z</sub> (nm)", y_label="traces",
            ))
            root.addWidget(hist_group, stretch=1)
        except Exception as exc:
            warn = QLabel(f"Histogram report could not be created: {exc}")
            warn.setStyleSheet("color: #a66;")
            root.addWidget(warn)

        note = QLabel(
            "σ<sub>x</sub>, σ<sub>y</sub>, σ<sub>z</sub> — sample standard deviation "
            "(ddof = 1) of each <b>raw</b> coordinate over the localizations of one "
            "trace (nm)<br>"
            "σ<sub>r</sub> = √( (σ<sub>x</sub>² + σ<sub>y</sub>²) ∕ 2 ) — combined "
            "lateral spread of a trace<br>"
            f"n — localizations in a trace; only traces with n ≥ {MIN_LOCS_PER_TRACE} "
            "are used<br>"
            "reported precision = n-weighted standard error of the mean position, "
            "averaged over the M traces:<br>"
            "&nbsp;&nbsp;σ<sub>r</sub> = ⟨ σ<sub>r</sub> ∕ √n ⟩ , &nbsp; "
            "σ<sub>z</sub> = ⟨ σ<sub>z</sub> ∕ √n ⟩ &nbsp; ( ⟨·⟩ = mean over traces )<br>"
            "z uses the raw coordinate (no RIMF applied)."
            + _references_html(
                "<a href='https://www.nature.com/articles/s41592-022-01577-1'>"
                "Ostersehlt <i>et al.</i>, <i>Nat. Methods</i> 19:1072 (2022)</a>",
            )
        )
        note.setWordWrap(True)
        note.setOpenExternalLinks(True)
        note.setTextInteractionFlags(Qt.TextInteractionFlag.TextBrowserInteraction)
        note.setStyleSheet("color: gray; font-size: 11px;")
        root.addWidget(note)

        bb = QDialogButtonBox(QDialogButtonBox.StandardButton.Close)
        bb.rejected.connect(self.reject)
        bb.button(QDialogButtonBox.StandardButton.Close).clicked.connect(self.close)
        root.addWidget(bb)


# ---------------------------------------------------------------------------
# FRC result dialog
# ---------------------------------------------------------------------------

_FRC_CURVE_COLOR = "#4a90ff"   # FRC curve — blue
_FRC_RES_COLOR = "#35d07f"     # resolution crossing — green


def _frc_curve_plot(result: "FRCResult"):
    """Anisotropy-style panel of the FRC curve vs spatial frequency."""
    import pyqtgraph as pg

    pw = pg.PlotWidget(background="#222")
    pw.setMinimumHeight(320)
    pw.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Expanding)
    pw.showGrid(x=True, y=True, alpha=0.22)
    pw.setTitle("Fourier Ring Correlation")
    pw.setLabel("bottom", "spatial frequency q (nm<sup>-1</sup>)")
    pw.setLabel("left", "FRC")
    if result.freq_inv_nm.size < 2:
        return pw

    # Skip the DC bin (FRC ≈ 1 by construction).
    freq = result.freq_inv_nm[1:]
    frc = result.frc_mean[1:]

    # ± std band across the random splits.
    if result.frc_std.size == result.frc_mean.size and result.n_repeats > 1:
        std = result.frc_std[1:]
        hi = pw.plot(freq, frc + std, pen=None)
        lo = pw.plot(freq, frc - std, pen=None)
        band = pg.FillBetweenItem(hi, lo, brush=pg.mkBrush(74, 144, 255, 45))
        pw.addItem(band)

    pw.plot(freq, frc, pen=pg.mkPen(_FRC_CURVE_COLOR, width=2.0))

    # 1/7 threshold (horizontal).
    thr = pg.InfiniteLine(
        pos=result.threshold, angle=0,
        pen=pg.mkPen("#bbbbbb", width=1.3, style=Qt.PenStyle.DashLine),
        label=f"1/7 = {result.threshold:.3f}",
        labelOpts={"position": 0.06, "color": "#bbbbbb", "fill": (20, 20, 20, 200)},
    )
    pw.addItem(thr)

    # Resolution crossing (vertical).
    if np.isfinite(result.crossing_freq) and np.isfinite(result.resolution_nm):
        cl = pg.InfiniteLine(
            pos=result.crossing_freq, angle=90,
            pen=pg.mkPen(_FRC_RES_COLOR, width=1.7, style=Qt.PenStyle.DashLine),
            label=f"resolution = {result.resolution_nm:.2f} nm",
            labelOpts={"position": 0.9, "color": _FRC_RES_COLOR, "fill": (20, 20, 20, 220)},
        )
        pw.addItem(cl)

    pw.setYRange(-0.15, 1.05, padding=0)
    pw.setXRange(0.0, float(result.freq_inv_nm[-1]), padding=0)
    return pw


class FRCResultDialog(QDialog):
    """Fourier Ring Correlation resolution report, anisotropy-plugin style.

    Holds the source 2-D coordinates so the user can recompute with a
    different super-resolution pixel size from the dialog.
    """

    def __init__(self, result: "FRCResult", x_nm: np.ndarray, y_nm: np.ndarray,
                 tid: np.ndarray | None = None, parent=None) -> None:
        super().__init__(parent)
        self._x, self._y, self._tid = x_nm, y_nm, tid
        self.setWindowTitle("Localization Precision — FRC resolution")
        self.setModal(False)
        self.resize(820, 700)

        root = QVBoxLayout(self)
        root.setContentsMargins(10, 10, 10, 10)
        root.setSpacing(8)
        self._root = root
        self._build(result)

    # -- (re)build the body for a freshly computed result -----------------
    def _build(self, result: "FRCResult") -> None:
        # Clear any previously built widgets (on recompute), sub-layouts included.
        _clear_layout(self._root)
        root = self._root
        res = result.resolution_nm

        if result.mode == "combined":
            pts_txt = (f"<b>{result.n_points:,}</b> combined positions "
                       f"(from {result.n_locs:,} localizations)")
        else:
            pts_txt = f"<b>{result.n_locs:,}</b> localizations"
        info = QLabel(
            f"{pts_txt} &nbsp;|&nbsp; {result.n_repeats}× averaged "
            f"&nbsp;|&nbsp; {result.image_px}² px @ {result.pixel_size_nm:.2f} nm/px"
        )
        info.setAlignment(Qt.AlignmentFlag.AlignCenter)
        root.addWidget(info)

        headline = QLabel(
            f"<span style='font-size:18px'><b>FRC resolution = {_res(res)}</b></span>"
            "<span style='color:gray'> &nbsp;(1/7 threshold)</span>"
        )
        headline.setAlignment(Qt.AlignmentFlag.AlignCenter)
        root.addWidget(headline)

        if result.status:
            warn = QLabel(result.status)
            warn.setWordWrap(True)
            warn.setAlignment(Qt.AlignmentFlag.AlignCenter)
            warn.setStyleSheet("color: #d2a24a; font-size: 11px;")
            root.addWidget(warn)

        try:
            grp = QGroupBox("FRC curve")
            box = QVBoxLayout(grp)
            box.setContentsMargins(8, 8, 8, 8)
            box.addWidget(_frc_curve_plot(result))
            root.addWidget(grp, stretch=1)
        except Exception as exc:
            warn = QLabel(f"FRC plot could not be created: {exc}")
            warn.setStyleSheet("color: #a66;")
            root.addWidget(warn)

        note = QLabel(
            "The localizations are split into two independent images by a random "
            "50/50 per-localization assignment, each rendered as a ~1 nm-pixel "
            "histogram.<br>"
            "FRC(q) = Σ<sub>ring</sub> Re{ F<sub>1</sub>(q)·F<sub>2</sub>*(q) } ∕ "
            "√( Σ<sub>ring</sub>|F<sub>1</sub>(q)|² · Σ<sub>ring</sub>|F<sub>2</sub>(q)|² )<br>"
            "where F<sub>1</sub>, F<sub>2</sub> are the 2-D Fourier transforms of the "
            "half-images, summed over each ring of constant spatial frequency q. The "
            "curve is averaged over the random splits, Savitzky-Golay smoothed, and the "
            "high-frequency tail cut at 0.8·q<sub>max</sub>.<br>"
            "resolution = 1 ∕ q<sub>c</sub> , with q<sub>c</sub> the frequency where "
            "FRC first drops below 1/7.<br>"
            "<i>points</i>: <b>per localization</b> uses every localization (optimistic "
            "when emitters are localized many times); <b>combined</b> first collapses "
            "each trace to its mean position — the honest MINFLUX resolution."
            + _references_html(
                "<a href='https://www.nature.com/articles/s41592-019-0688-0'>Gwosch "
                "<i>et al.</i>, <i>Nat. Methods</i> 17:217 (2020)</a>",
                "<a href='https://www.nature.com/articles/s41592-022-01577-1'>Ostersehlt "
                "<i>et al.</i>, <i>Nat. Methods</i> 19:1072 (2022)</a>",
            )
        )
        note.setWordWrap(True)
        note.setOpenExternalLinks(True)
        note.setTextInteractionFlags(Qt.TextInteractionFlag.TextBrowserInteraction)
        note.setStyleSheet("color: gray; font-size: 11px;")
        root.addWidget(note)

        # -- controls: points mode + editable pixel size + recompute ------
        btn_row = QHBoxLayout()
        self._status_lbl = QLabel("")
        self._status_lbl.setStyleSheet("color: #6c6;")
        btn_row.addWidget(self._status_lbl)
        btn_row.addStretch()
        btn_row.addWidget(QLabel("points:"))
        self._mode_combo = QComboBox()
        self._mode_combo.addItem("per localization", "localization")
        self._mode_combo.addItem("combined (per trace)", "combined")
        has_tid = self._tid is not None
        self._mode_combo.model().item(1).setEnabled(has_tid)
        idx = self._mode_combo.findData(result.mode if has_tid else "localization")
        self._mode_combo.setCurrentIndex(max(0, idx))
        self._mode_combo.setToolTip(
            "Run FRC on every localization, or on one mean position per trace "
            "(combined) — the honest MINFLUX resolution. Needs trace IDs.")
        btn_row.addWidget(self._mode_combo)
        btn_row.addSpacing(8)
        btn_row.addWidget(QLabel("pixel (nm):"))
        self._px_spin = QDoubleSpinBox()
        self._px_spin.setRange(0.1, 1000.0)
        self._px_spin.setDecimals(2)
        self._px_spin.setSingleStep(0.5)
        self._px_spin.setValue(result.pixel_size_nm if result.pixel_size_nm > 0 else 1.0)
        self._px_spin.setToolTip("Super-resolution render pixel size for the FRC images.")
        btn_row.addWidget(self._px_spin)
        recompute = QPushButton("Recompute")
        recompute.setToolTip("Re-run FRC with the chosen points mode and pixel size.")
        recompute.clicked.connect(self._recompute)
        btn_row.addWidget(recompute)
        close_btn = QPushButton("Close")
        close_btn.clicked.connect(self.reject)
        btn_row.addWidget(close_btn)
        root.addLayout(btn_row)

    def _recompute(self) -> None:
        pixel = float(self._px_spin.value())
        mode = self._mode_combo.currentData()
        try:
            result = frc_resolution(self._x, self._y, self._tid,
                                    mode=mode, pixel_size_nm=pixel)
        except Exception as exc:
            self._status_lbl.setText(f"failed: {exc}")
            return
        self._build(result)
        self._status_lbl.setText(f"recomputed ({result.mode}) @ {pixel:.2f} nm/px")


def _res(val: float) -> str:
    if val is None or not np.isfinite(val):
        return "—"
    return f"{val:.2f} nm"


def _references_html(*refs: str) -> str:
    """Standard reference block for result/UI dialogs.

    Project convention: a dialog's reference section is preceded by one blank
    row and lists each reference on its own line (vertically stacked). Pass the
    fully-formatted reference strings (HTML, including any ``<a>`` links); the
    body text this is appended to must NOT end with a trailing ``<br>``.
    """
    label = "Reference" if len(refs) == 1 else "References"
    items = "<br>".join(refs)
    return f"<br><br><b>{label}</b><br>{items}"


def _clear_layout(layout) -> None:
    """Recursively remove and delete every widget/sub-layout from *layout*.

    A plain ``takeAt``/``setParent(None)`` loop misses widgets nested inside
    child layouts (e.g. a controls ``QHBoxLayout``), so they accumulate on each
    rebuild. This walks sub-layouts too.
    """
    while layout.count():
        item = layout.takeAt(0)
        w = item.widget()
        if w is not None:
            w.setParent(None)
            w.deleteLater()
            continue
        child = item.layout()
        if child is not None:
            _clear_layout(child)
            child.deleteLater()


# ---------------------------------------------------------------------------
# CRLB result dialog
# ---------------------------------------------------------------------------

_CRLB_COLOR = "#a78bfa"   # lateral σ_xy distribution — violet
_CRLBZ_COLOR = "#ff5a5a"  # axial σ_z distribution — red
_N_COLOR = "#f5a623"      # photon count distribution — amber


class CRLBResultDialog(QDialog):
    """MINFLUX Cramér-Rao bound report, anisotropy-plugin style.

    Holds the source photon counts / rates so the user can recompute with a
    different pattern diameter L from the dialog.
    """

    def __init__(self, result: "CRLBResult", eco: np.ndarray,
                 efo: np.ndarray | None = None, fbg: np.ndarray | None = None,
                 num_dim: int = 2, measured: dict | None = None, parent=None) -> None:
        super().__init__(parent)
        self._eco, self._efo, self._fbg = eco, efo, fbg
        self._num_dim = int(num_dim)
        # Measured per-trace σ (StdDev/trace) for the precision budget, or None.
        self._measured = measured
        self.setWindowTitle("Localization Precision — CRLB (MINFLUX)")
        self.setModal(False)
        self.resize(1120 if self._num_dim >= 3 else 880, 680)

        root = QVBoxLayout(self)
        root.setContentsMargins(10, 10, 10, 10)
        root.setSpacing(8)
        self._root = root
        self._build(result)

    def _build(self, result: "CRLBResult") -> None:
        _clear_layout(self._root)
        root = self._root

        if result.n_locs == 0 or not np.isfinite(result.median_sigma_ideal):
            warn = QLabel(f"<b>{result.status or 'CRLB could not be computed.'}</b>")
            warn.setWordWrap(True)
            root.addWidget(warn)
            self._add_controls(root, result)
            return

        sbr_txt = (f" &nbsp;|&nbsp; median SBR = {result.median_sbr:.1f}"
                   if result.has_background and np.isfinite(result.median_sbr) else "")
        if result.per_dimension_photons and np.isfinite(result.median_n_z):
            n_txt = (f"median N<sub>xy</sub> = <b>{result.median_n:.0f}</b>, "
                     f"N<sub>z</sub> = <b>{result.median_n_z:.0f}</b> photons")
        else:
            n_txt = f"median N = <b>{result.median_n:.0f}</b> photons"
        info = QLabel(
            f"<b>{result.n_locs:,}</b> localizations &nbsp;|&nbsp; "
            f"{n_txt} &nbsp;|&nbsp; "
            f"L = <b>{result.L_nm:.0f}</b> nm &nbsp;|&nbsp; "
            f"σ<sub>q</sub> = <b>{result.sigma_q_nm:.0f}</b> nm{sbr_txt}"
        )
        info.setAlignment(Qt.AlignmentFlag.AlignCenter)
        root.addWidget(info)

        primary = result.median_sigma_bg if result.has_background else result.median_sigma_ideal
        kind = "background-limited" if result.has_background else "ideal (no background data)"
        if result.has_z:
            z_primary = result.median_sigma_z_bg if result.has_background else result.median_sigma_z_ideal
            head_value = (f"σ<sub>xy</sub> = {_res(primary)}, "
                          f"σ<sub>z</sub> = {_res(z_primary)}")
        else:
            head_value = _res(primary)
        headline = QLabel(
            f"<span style='font-size:18px'><b>CRLB precision (median σ): "
            f"{head_value}</b></span>"
            f"<span style='color:gray'> &nbsp;({kind})</span>"
        )
        headline.setAlignment(Qt.AlignmentFlag.AlignCenter)
        root.addWidget(headline)

        if self._num_dim >= 3 and result.has_z:
            caveat = QLabel(
                "3-D dataset: lateral (xy) from the 2-D donut; axial (z) from the "
                "1-D quadratic-minimum estimator (Eq. 28), reusing N and the measured "
                "(lateral) SBR with an axial pattern size L_z."
            )
            caveat.setWordWrap(True)
            caveat.setAlignment(Qt.AlignmentFlag.AlignCenter)
            caveat.setStyleSheet("color: #d2a24a; font-size: 11px;")
            root.addWidget(caveat)

        # σ_CRB table — ideal vs background-limited (per channel for 3-D).
        grp = QGroupBox("Cramér-Rao bound σ_CRB across localizations  (nm)")
        grid = QGridLayout(grp)
        grid.setHorizontalSpacing(20)
        grid.setVerticalSpacing(4)
        for col, h in enumerate(("", "ideal (no bg)", "background-limited")):
            lbl = QLabel(f"<b>{h}</b>")
            lbl.setAlignment(Qt.AlignmentFlag.AlignLeft if col == 0
                             else Qt.AlignmentFlag.AlignCenter)
            grid.addWidget(lbl, 0, col)
        if result.has_z:
            rows = (
                ("xy Median ★", result.median_sigma_ideal, result.median_sigma_bg, True),
                ("xy Mean", result.mean_sigma_ideal, result.mean_sigma_bg, False),
                ("z Median ★", result.median_sigma_z_ideal, result.median_sigma_z_bg, True),
                ("z Mean", result.mean_sigma_z_ideal, result.mean_sigma_z_bg, False),
            )
        else:
            rows = (
                ("Median ★", result.median_sigma_ideal, result.median_sigma_bg, True),
                ("Mean", result.mean_sigma_ideal, result.mean_sigma_bg, False),
            )
        for r, (row_lbl, v_ideal, v_bg, bold) in enumerate(rows, start=1):
            name = QLabel(f"<b>{row_lbl}:</b>" if bold else f"{row_lbl}:")
            grid.addWidget(name, r, 0)
            for col, v in enumerate((v_ideal, v_bg), start=1):
                t = _res(v)
                cell = QLabel(f"<b>{t}</b>" if bold else t)
                cell.setAlignment(Qt.AlignmentFlag.AlignCenter)
                grid.addWidget(cell, r, col)
        for col in (1, 2):
            grid.setColumnStretch(col, 1)
        root.addWidget(grp)

        # Precision budget — measured spread vs the photon-limited bound.
        self._add_precision_budget(root, result)

        # Distributions — σ_xy [· σ_z for 3-D] · photon count N, in one row.
        try:
            from .trace_analysis import _value_hist_plot
            hist_group = QGroupBox("Per-localization distributions")
            row = QHBoxLayout(hist_group)
            row.setContentsMargins(8, 8, 8, 8)
            sigma = result.per_loc_sigma_bg if result.has_background else result.per_loc_sigma_ideal
            row.addWidget(_value_hist_plot(
                "σ<sub>xy</sub> per localization (lateral)", sigma, _CRLB_COLOR,
                marker=float(primary), marker_label=f"median = {primary:.2f} nm",
                x_label="σ<sub>xy</sub> (nm)", y_label="localizations",
            ))
            if result.has_z:
                sigma_z = (result.per_loc_sigma_z_bg if result.has_background
                           else result.per_loc_sigma_z_ideal)
                z_primary = (result.median_sigma_z_bg if result.has_background
                             else result.median_sigma_z_ideal)
                row.addWidget(_value_hist_plot(
                    "σ<sub>z</sub> per localization (axial)", sigma_z, _CRLBZ_COLOR,
                    marker=float(z_primary), marker_label=f"median = {z_primary:.2f} nm",
                    x_label="σ<sub>z</sub> (nm)", y_label="localizations",
                ))
            row.addWidget(_value_hist_plot(
                "photon count N per localization", result.per_loc_n_total, _N_COLOR,
                marker=float(result.median_n), marker_label=f"median = {result.median_n:.0f}",
                x_label="N (photons)", y_label="localizations",
            ))
            root.addWidget(hist_group, stretch=1)
        except Exception as exc:
            warn = QLabel(f"Histogram report could not be created: {exc}")
            warn.setStyleSheet("color: #a66;")
            root.addWidget(warn)

        z_formula = (
            "&nbsp;&nbsp;σ<sub>z</sub> = L<sub>z</sub> ∕ ( 4·√N ) · "
            "√[ ( 1 + 1∕SBR )( 1 + 2∕(3·SBR) ) ] &nbsp;&nbsp;(axial, 1-D quadratic "
            "minimum, K = 3; ideal = L<sub>z</sub> ∕ (4·√N))<br>"
        ) if result.has_z else ""
        z_param = (
            "&nbsp;&nbsp;L<sub>z</sub> — axial pattern size &nbsp;(user input; σ<sub>qz</sub> "
            "cancels in the SBR form, so the axial PSF width is not needed)<br>"
        ) if result.has_z else ""

        # References + an optional precision-budget note (its decomposition source).
        refs = [
            "<a href='https://arxiv.org/abs/2410.12427'>Marin &amp; Ries, "
            "<i>arXiv</i>:2410.12427 (2024)</a> (Eq. 44/45 lateral, Eq. 28 axial)",
            "<a href='https://doi.org/10.1126/science.aak9913'>Balzarotti "
            "<i>et al.</i>, <i>Science</i> 355:606 (2017)</a>",
        ]
        budget_note = ""
        if self._measured:
            budget_note = (
                "<br><b>Precision budget</b>: the measured per-trace spread and this "
                "bound combine as STD² = σ<sub>fl</sub>² + σ<sub>CRB</sub>², so the "
                "excess σ<sub>fl</sub> = √(STD² − σ<sub>CRB</sub>²) is the "
                "photon-count-independent error (flickering / drift / vibration / "
                "misalignment) beyond the photon limit."
            )
            refs.append(
                "<a href='https://www.nature.com/articles/s41467-025-66952-w'>Marin "
                "&amp; Ries, <i>Nat. Commun.</i> 17:246 (2026)</a> (SimuFLUX; precision "
                "decomposition)"
            )
        note = QLabel(
            "<b>Formulas</b> (emitter at the pattern centre):<br>"
            "&nbsp;&nbsp;σ<sub>xy</sub> = L ∕ ( 2·√(2N) ) · "
            "( 1 − L²·ln2 ∕ σ<sub>q</sub>² )<sup>−1</sup> · "
            "√[ ( 1 + 1∕SBR )( 1 + 3∕(4·SBR) ) ] &nbsp;&nbsp;(lateral, 2-D donut, "
            "K = 4; ideal drops the SBR factor)<br>"
            + z_formula +
            "<b>Parameters</b><br>"
            "&nbsp;&nbsp;N — photons per localization = <i>eco</i> &nbsp;(taken from data)<br>"
            "&nbsp;&nbsp;SBR — signal-to-background ratio = ( <i>efo</i> − <i>fbg</i> ) ∕ "
            "<i>fbg</i> &nbsp;(computed per localization from data)<br>"
            "&nbsp;&nbsp;L — lateral targeted-pattern diameter of the last iteration "
            "&nbsp;(user input; not stored in the file)<br>"
            "&nbsp;&nbsp;σ<sub>q</sub> — excitation-PSF steepness / donut FWHM, "
            "diffraction-bounded ≳ 250 nm &nbsp;(user input)<br>"
            + z_param +
            "Per-coordinate precision at the centre — a best-case lower bound; background "
            "enters through the measured SBR. The axial channel reuses N and the lateral "
            "SBR (an approximation, since the z exposures may carry a different SBR)."
            + budget_note
            + _references_html(*refs)
        )
        note.setWordWrap(True)
        note.setOpenExternalLinks(True)
        note.setTextInteractionFlags(Qt.TextInteractionFlag.TextBrowserInteraction)
        note.setStyleSheet("color: gray; font-size: 11px;")
        root.addWidget(note)

        self._add_controls(root, result)

    def _add_precision_budget(self, root, result: "CRLBResult") -> None:
        """Measured spread (StdDev/trace) vs the photon-limited bound → the excess
        error σ_fl = √(STD² − σ_CRB²) (Marin & Ries 2026 decomposition). Shown only
        when per-trace measured precision is available; recomputes with L/σ_q."""
        m = self._measured
        if not m:
            return
        crb_xy = result.median_sigma_bg if result.has_background else result.median_sigma_ideal
        if not np.isfinite(crb_xy):
            return
        rows = [("lateral (xy)", m.get("sigma_r"), crb_xy,
                 excess_precision(m.get("sigma_r"), crb_xy))]
        if result.has_z:
            crb_z = (result.median_sigma_z_bg if result.has_background
                     else result.median_sigma_z_ideal)
            if np.isfinite(crb_z):
                rows.append(("axial (z)", m.get("sigma_z"), crb_z,
                             excess_precision(m.get("sigma_z"), crb_z)))

        grp = QGroupBox(
            f"Precision budget — measured vs photon limit  "
            f"({m.get('n_traces', 0):,} traces, ≥ {MIN_LOCS_PER_TRACE} loc, raw z)")
        grid = QGridLayout(grp)
        grid.setHorizontalSpacing(20)
        grid.setVerticalSpacing(4)
        for col, h in enumerate(("", "measured σ (StdDev/trace)",
                                 "σ<sub>CRB</sub> (photon limit)", "excess σ<sub>fl</sub>")):
            lbl = QLabel(f"<b>{h}</b>")
            lbl.setAlignment(Qt.AlignmentFlag.AlignLeft if col == 0
                             else Qt.AlignmentFlag.AlignCenter)
            grid.addWidget(lbl, 0, col)
        for r, (name, meas, crb, fl) in enumerate(rows, start=1):
            grid.addWidget(QLabel(f"<b>{name}:</b>"), r, 0)
            for col, v in enumerate((meas, crb, fl), start=1):
                cell = QLabel(_res(v))
                cell.setAlignment(Qt.AlignmentFlag.AlignCenter)
                grid.addWidget(cell, r, col)
        for col in (1, 2, 3):
            grid.setColumnStretch(col, 1)
        root.addWidget(grp)

        # One-line interpretation of the lateral budget.
        meas_r, fl_r = rows[0][1], rows[0][3]
        if np.isfinite(fl_r) and meas_r and meas_r > 0:
            if fl_r <= 0.15 * crb_xy:
                verdict = "measured precision is essentially photon-limited (at the bound)"
            else:
                frac = 100.0 * fl_r / float(meas_r)
                verdict = (f"an excess of σ<sub>fl</sub> = {fl_r:.1f} nm "
                           f"(~{frac:.0f}% of the measured spread) is <b>not</b> "
                           "photon-limited — attributable to flickering, drift, "
                           "vibration or misalignment (not reducible by more photons)")
            tip = QLabel(f"Lateral: {verdict}.")
            tip.setWordWrap(True)
            tip.setAlignment(Qt.AlignmentFlag.AlignCenter)
            tip.setStyleSheet("color: #8ab4d8; font-size: 11px;")
            root.addWidget(tip)

    def _add_controls(self, root, result: "CRLBResult") -> None:
        """Editable L + σ_q + Recompute row (also shown on the error path)."""
        btn_row = QHBoxLayout()
        self._status_lbl = QLabel("")
        self._status_lbl.setStyleSheet("color: #6c6;")
        btn_row.addWidget(self._status_lbl)
        btn_row.addStretch()
        btn_row.addWidget(QLabel("pattern L (nm):"))
        self._l_spin = QDoubleSpinBox()
        self._l_spin.setRange(1.0, 1000.0)
        self._l_spin.setDecimals(1)
        self._l_spin.setSingleStep(5.0)
        self._l_spin.setValue(result.L_nm)
        self._l_spin.setToolTip("Targeted-coordinate pattern diameter L of the "
                                "last MINFLUX iteration (sequence parameter).")
        btn_row.addWidget(self._l_spin)
        btn_row.addSpacing(8)
        btn_row.addWidget(QLabel("σ_q (nm):"))
        self._sq_spin = QDoubleSpinBox()
        self._sq_spin.setRange(1.0, 2000.0)
        self._sq_spin.setDecimals(0)
        self._sq_spin.setSingleStep(10.0)
        self._sq_spin.setValue(result.sigma_q_nm)
        self._sq_spin.setToolTip("Excitation-PSF steepness / donut FWHM σ_q "
                                 "(diffraction-bounded, ~250 nm). Requires L < σ_q.")
        btn_row.addWidget(self._sq_spin)
        # Axial pattern size L_z — only for 3-D (when an axial bound is computed).
        self._lz_spin = None
        if result.has_z or self._num_dim >= 3:
            btn_row.addSpacing(8)
            btn_row.addWidget(QLabel("L_z (nm):"))
            self._lz_spin = QDoubleSpinBox()
            self._lz_spin.setRange(1.0, 2000.0)
            self._lz_spin.setDecimals(1)
            self._lz_spin.setSingleStep(5.0)
            self._lz_spin.setValue(result.L_z_nm if result.L_z_nm > 0 else CRLB_DEFAULT_LZ_NM)
            self._lz_spin.setToolTip("Axial targeted-pattern size L_z of the last "
                                     "iteration (3-D z channel).")
            btn_row.addWidget(self._lz_spin)
        recompute = QPushButton("Recompute")
        recompute.clicked.connect(self._recompute)
        btn_row.addWidget(recompute)
        close_btn = QPushButton("Close")
        close_btn.clicked.connect(self.reject)
        btn_row.addWidget(close_btn)
        root.addLayout(btn_row)

    def _recompute(self) -> None:
        L = float(self._l_spin.value())
        sq = float(self._sq_spin.value())
        lz = float(self._lz_spin.value()) if self._lz_spin is not None else None
        try:
            result = crlb_precision(
                self._eco, L_nm=L, sigma_q_nm=sq, efo=self._efo, fbg=self._fbg,
                L_z_nm=lz)
        except Exception as exc:
            self._status_lbl.setText(f"failed: {exc}")
            return
        self._build(result)
        msg = f"recomputed @ L = {L:.0f} nm, σ_q = {sq:.0f} nm"
        if lz is not None:
            msg += f", L_z = {lz:.0f} nm"
        self._status_lbl.setText(msg)


# ---------------------------------------------------------------------------
# Internal helper
# ---------------------------------------------------------------------------

class InfoDialog(QDialog):
    """Small non-blocking rich-text information dialog with clickable links."""

    def __init__(self, title: str, html: str, parent=None) -> None:
        super().__init__(parent)
        self.setWindowTitle(title)
        self.setModal(False)
        self.resize(620, 360)

        root = QVBoxLayout(self)
        label = QLabel(html)
        label.setWordWrap(True)
        label.setOpenExternalLinks(True)
        label.setTextInteractionFlags(Qt.TextInteractionFlag.TextBrowserInteraction)
        label.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Expanding)
        root.addWidget(label)

        bb = QDialogButtonBox(QDialogButtonBox.StandardButton.Close)
        bb.rejected.connect(self.reject)
        bb.button(QDialogButtonBox.StandardButton.Close).clicked.connect(self.close)
        root.addWidget(bb)


def _show_info_dialog(parent, title: str, html: str) -> None:
    dlg = InfoDialog(title, html, parent=parent)
    dlg.show()
    dlg.raise_()
    dlg.activateWindow()
    if parent is not None:
        parent._loc_precision_info_dialog = dlg

def _try_journal(state, text: str) -> None:
    """Append a journal entry if the state has a processing journal."""
    journal = getattr(state, "journal", None)
    if journal is not None:
        try:
            journal.add("analysis", text)
        except Exception:
            pass
    try:
        state.log(text, "INFO")
    except Exception:
        pass
