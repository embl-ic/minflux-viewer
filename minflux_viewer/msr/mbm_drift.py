"""
Post-hoc MBM drift correction — the Imspector algorithm, reverse-engineered.

Every ``.msr`` localization row carries **two** coordinate fields:

``lnc``
    the *uncorrected* localization — the position as measured, before any
    fiducial-based stabilisation. It is never rewritten; a drift-corrected
    ``.msr`` exported by Imspector carries it bit-identical to the original.
``loc``
    the *drift-corrected* localization. This is the field every viewer plots,
    and the only field a drift correction changes.

So an "uncorrected" file is already corrected: its ``loc`` holds the **online**
correction the instrument applied during acquisition, using every fiducial bead
it was tracking. A post-hoc correction re-derives the same quantity from a
**manually reduced** bead set — dropping fiducials that drifted, bleached or
were mis-tracked — and rewrites ``loc`` from the preserved ``lnc``::

    loc(t) = lnc(t) - ( D(t) - D(t_ref) )

    D(t) = unweighted mean over the SELECTED beads of each bead's
           gaussian-smoothed, linearly-interpolated xyz(t)

Established empirically (2026-09-01) against two Imspector-corrected
uncorrected/corrected ``.msr`` pairs, reproducing their ``loc`` to **0.14-0.16 nm
rms** on corrections spanning 141-157 nm. Each property was measured, not
assumed:

* **A pure function of time** — a global rigid translation. The spread of the
  applied shift within a 0.1 s window is 1.5 pm; there is no rotation, scaling
  or position dependence.
* **An unweighted arithmetic mean** over the selected beads. An unconstrained
  least-squares fit for per-bead weights, given no prior about the selection,
  returns ~1/N on each selected bead and ~0 on every other, summing to 1.0000.
  Weighting by the ``str`` PMT signal is ~20x worse.
* **Gaussian smoothing of sigma ~ 1.2 bead samples** (~5 s at the usual 4 s bead
  cadence), optimal in both reference datasets: unsmoothed leaves 1.13 nm,
  sigma=1.2 leaves 0.16 nm. Linear interpolation beats sample-and-hold or nearest.
* **The reference is the start of the acquisition.**

⚠ **The absolute offset is a convention we cannot fully recover, and it does not
matter.** Reproducing Imspector's ``loc`` leaves a *constant* residual of
1-2 nm — no candidate origin available in the file (first localization time,
earliest bead sample, ``t=0``, each bead's own first sample) matches it exactly,
so Imspector anchors to something it does not store. That residual is a **rigid
translation of the entire dataset**: it changes no distance, no relative
geometry, no structure. The part that does matter, the *shape* of ``D(t)``,
reproduces to **0.28 nm rms in 3-D** (per-axis 0.13/0.09/0.23 nm). So expect our
output to differ from an Imspector export by a global few-nm shift and by
essentially nothing else.

Which beads a given file used is recorded in the store: ``mbm/.zattrs["used"]``
holds the online set in an uncorrected file and the manual selection in a
corrected one (see :func:`used_bead_gris`).

This module is pure NumPy/SciPy and Qt-free.
"""

from __future__ import annotations

import numpy as np

#: Gaussian smoothing width applied to each bead trace, in **bead samples**.
#: 1.2 samples is about 5 s at the usual 4 s MBM cadence; fitted against Imspector.
DEFAULT_SMOOTH_SAMPLES = 1.2

#: Required localization fields. ``lnc`` is the uncorrected position the
#: correction is computed from; without it a file cannot be re-corrected,
#: because ``loc`` alone cannot be un-done.
UNCORRECTED_FIELD = "lnc"
CORRECTED_FIELD = "loc"


class DriftCorrectionError(RuntimeError):
    """Raised when a drift correction cannot be computed from the given inputs."""


def _smooth(values: np.ndarray, sigma: float) -> np.ndarray:
    """Gaussian-smooth a bead trace along time, edges extended (``nearest``)."""
    if sigma is None or sigma <= 0 or len(values) < 2:
        return values
    from scipy.ndimage import gaussian_filter1d

    return gaussian_filter1d(values, float(sigma), axis=0, mode="nearest")


def bead_trace(points, gri: int, *, require_signal: bool = True):
    """One bead's ``(times_s, xyz_m)``, chronological, with dead samples dropped.

    A ``str`` (PMT signal) of zero marks a sample where the bead was not
    actually seen — those rows carry a stale or zero position and must not enter
    the average. ``require_signal=False`` keeps them, for callers inspecting the
    raw record.
    """
    if points is None or getattr(points, "dtype", None) is None:
        return np.empty(0), np.empty((0, 3))
    names = points.dtype.names or ()
    if not {"gri", "xyz", "tim"} <= set(names):
        return np.empty(0), np.empty((0, 3))
    mask = np.asarray(points["gri"]).ravel() == int(gri)
    if require_signal and "str" in names:
        mask &= np.asarray(points["str"], dtype=float).ravel() > 0
    if not mask.any():
        return np.empty(0), np.empty((0, 3))
    tim = np.asarray(points["tim"], dtype=float).ravel()[mask]
    xyz = np.asarray(points["xyz"], dtype=float)[mask]
    order = np.argsort(tim, kind="stable")
    return tim[order], xyz[order]


def usable_bead_gris(points, gris, *, min_samples: int = 2) -> list[int]:
    """The subset of *gris* that actually carries a usable trace, sorted."""
    out = []
    for gri in gris:
        tim, _ = bead_trace(points, gri)
        if len(tim) >= min_samples:
            out.append(int(gri))
    return sorted(out)


def drift_at(points, gris, times, *, smooth_samples: float = DEFAULT_SMOOTH_SAMPLES,
             min_samples: int = 2) -> np.ndarray:
    """``D(t)`` in **metres** at ``times``: the mean smoothed bead position.

    Not yet referenced to anything — subtract ``D(t_ref)`` to obtain a
    displacement. Beads with fewer than *min_samples* live samples are skipped;
    if that leaves none, :class:`DriftCorrectionError` is raised rather than
    returning a silently meaningless zero.

    The mean is accumulated **in place**, one axis at a time, rather than
    building each bead's full ``(n, 3)`` interpolation and averaging the stack:
    ``times`` is one entry per localization, so on a 20 M-row acquisition with
    six beads the stacked form costs ~2.9 GB of temporaries against ~0.6 GB here.
    """
    times = np.asarray(times, dtype=float).ravel()
    total = np.zeros((times.size, 3), dtype=float)
    used = 0
    for gri in gris:
        tim, xyz = bead_trace(points, gri)
        if len(tim) < min_samples:
            continue
        smoothed = _smooth(xyz, smooth_samples)
        for k in range(3):
            total[:, k] += np.interp(times, tim, smoothed[:, k])
        used += 1
    if not used:
        raise DriftCorrectionError(
            "None of the selected beads has enough usable samples to define a "
            "drift trace (each needs at least %d samples with a non-zero PMT "
            "signal)." % min_samples)
    total /= used
    return total


def drift_curve(points, gris, *, smooth_samples: float = DEFAULT_SMOOTH_SAMPLES,
                n_samples: int = 2000, t_range=None) -> tuple[np.ndarray, np.ndarray]:
    """A ``(times_s, drift_nm)`` curve for plotting, zeroed at its first point.

    Sampled on a regular grid across the beads' own time span (or *t_range*), so
    the plot shows the correction the user is about to apply rather than the
    localization sampling pattern.
    """
    if t_range is None:
        lo, hi = np.inf, -np.inf
        for gri in gris:
            tim, _ = bead_trace(points, gri)
            if len(tim):
                lo, hi = min(lo, tim[0]), max(hi, tim[-1])
        if not np.isfinite(lo) or hi <= lo:
            raise DriftCorrectionError("The selected beads span no time range.")
    else:
        lo, hi = float(t_range[0]), float(t_range[1])
    times = np.linspace(lo, hi, int(max(2, n_samples)))
    drift = drift_at(points, gris, times, smooth_samples=smooth_samples)
    return times, (drift - drift[0]) * 1.0e9


def per_bead_curves(points, gris, times, *,
                    smooth_samples: float = DEFAULT_SMOOTH_SAMPLES) -> dict[int, np.ndarray]:
    """Each selected bead's own smoothed trace in nm on *times*, zeroed at the
    first sample — what the mean is taken over, for plotting alongside it."""
    out: dict[int, np.ndarray] = {}
    times = np.asarray(times, dtype=float).ravel()
    for gri in gris:
        tim, xyz = bead_trace(points, gri)
        if len(tim) < 2:
            continue
        smoothed = _smooth(xyz, smooth_samples)
        q = np.column_stack([np.interp(times, tim, smoothed[:, k]) for k in range(3)])
        out[int(gri)] = (q - q[0]) * 1.0e9
    return out


def corrected_loc(mfx, points, gris, *,
                  smooth_samples: float = DEFAULT_SMOOTH_SAMPLES,
                  reference: str = "start") -> np.ndarray:
    """The drift-corrected ``loc`` array (metres) for one dataset's ``mfx``.

    Computed from ``lnc``, so it is independent of whatever correction ``loc``
    already carries — applying this twice with the same bead set is idempotent.
    Rows whose ``lnc`` is non-finite (failed probes) stay non-finite.

    *reference* fixes the global offset, which is a pure rigid translation of the
    whole dataset and changes no measured distance:

    ``"start"``
        origin at the first localization time — Imspector's convention, to
        within the 1-2 nm noted in the module docstring.
    ``"centroid"``
        offset chosen so the corrected cloud keeps the **current** ``loc``
        centroid, so applying the correction does not visibly translate the
        view. Useful when comparing before/after in the viewer.
    ``"none"``
        no reference subtracted — leaves the absolute bead position in the
        coordinates. Almost never what you want.
    """
    names = getattr(getattr(mfx, "dtype", None), "names", None) or ()
    if UNCORRECTED_FIELD not in names:
        raise DriftCorrectionError(
            "This dataset has no '%s' field, so the uncorrected positions it "
            "needs are not available. Drift correction reads 'lnc' and writes "
            "'loc'; a file carrying only 'loc' cannot be re-corrected, because "
            "the correction already applied to it cannot be undone."
            % UNCORRECTED_FIELD)
    if "tim" not in names:
        raise DriftCorrectionError(
            "This dataset has no 'tim' field to place the beads against.")

    lnc = np.asarray(mfx[UNCORRECTED_FIELD], dtype=float)
    times = np.asarray(mfx["tim"], dtype=float).ravel()
    drift = drift_at(points, gris, times, smooth_samples=smooth_samples)

    if reference == "start":
        finite = np.isfinite(times)
        t_ref = float(times[finite].min()) if finite.any() else float(times[0])
        ref = drift_at(points, gris, [t_ref], smooth_samples=smooth_samples)[0]
    elif reference == "centroid":
        # Match the current loc centroid, over the rows both frames define.
        old = np.asarray(mfx[CORRECTED_FIELD], dtype=float)
        both = np.isfinite(lnc).all(1) & np.isfinite(old).all(1) & np.isfinite(drift).all(1)
        if not both.any():
            raise DriftCorrectionError(
                "No row has both a finite 'lnc' and a finite 'loc', so the "
                "current centroid cannot be matched.")
        # corrected = lnc - drift + ref, so ref = mean(old) - mean(lnc - drift).
        ref = (old[both] - (lnc[both] - drift[both])).mean(0)
    elif reference == "none":
        ref = np.zeros(3)
    else:
        raise ValueError("unknown reference %r" % (reference,))
    return lnc - (drift - ref)


def apply_correction(mfx, points, gris, *,
                     smooth_samples: float = DEFAULT_SMOOTH_SAMPLES,
                     reference: str = "start") -> dict:
    """Rewrite ``mfx['loc']`` **in place** and report what changed.

    In place because the parsed-result dataset entries and the reader's
    ``mfx_map`` hold the *same* array objects — mutating the field updates both,
    so a subsequent "Open in MINFLUX viewer" loads the corrected coordinates
    without any further plumbing.

    Returns ``{"n_rows", "median_shift_nm", "max_shift_nm", "gris"}`` where the
    shifts describe how far this correction moved ``loc`` from where it was —
    i.e. the difference from the *previous* (online) correction, not the total
    drift removed.
    """
    new_loc = corrected_loc(mfx, points, gris,
                            smooth_samples=smooth_samples, reference=reference)
    old_loc = np.asarray(mfx[CORRECTED_FIELD], dtype=float)
    moved = np.linalg.norm(new_loc - old_loc, axis=1)
    finite = np.isfinite(moved)
    mfx[CORRECTED_FIELD] = new_loc
    return {
        "n_rows": int(finite.sum()),
        "median_shift_nm": float(np.median(moved[finite]) * 1e9) if finite.any() else 0.0,
        "max_shift_nm": float(moved[finite].max() * 1e9) if finite.any() else 0.0,
        "gris": sorted(int(g) for g in gris),
    }


def used_bead_gris(store) -> list[int]:
    """The bead gri-IDs a store records as *used*, via ``mbm/.zattrs["used"]``.

    That attribute is the only metadata an Imspector drift correction rewrites:
    it lists the online bead set in an uncorrected file and the manual selection
    in a corrected one. Returns ``[]`` when the store carries neither the used
    list nor the R-ID map.
    """
    from .io import read_zarr_attrs

    try:
        used = (read_zarr_attrs(store, "mbm") or {}).get("used") or []
        by_gri = (read_zarr_attrs(store, "grd/mbm/points") or {}).get("points_by_gri") or {}
    except Exception:
        return []
    name_to_gri = {}
    for key, info in by_gri.items():
        if isinstance(info, dict) and info.get("name"):
            try:
                name_to_gri[str(info["name"])] = int(info.get("gri", key))
            except (TypeError, ValueError):
                continue
    return sorted({name_to_gri[str(r)] for r in used if str(r) in name_to_gri})
