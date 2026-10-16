"""
minflux_viewer.analysis.bead_drift
==================================
Fiducial (MBM bead) drift correction of a **loaded dataset**.

The computation is Imspector's own, reconstructed in :mod:`minflux_viewer.msr.mbm_drift`
and verified against Imspector-corrected files: every ``.msr`` localization keeps
its uncorrected position ``lnc``, and the drift-corrected ``loc`` is

    loc(t) = lnc(t) − ( D(t) − D(t_start) ),

with ``D`` the mean of the selected beads' Gaussian-smoothed (σ ≈ 1.2 samples),
linearly interpolated tracks. The instrument applies it online with every bead it
tracks; a post-hoc correction in Imspector re-runs it with a hand-picked subset.
With the *same* subset the result is the online correction again, bit for bit —
which is why four of six "drift-corrected" files in the 2026-06-25/26 data equal
their uncorrected originals: their recorded bead set (``mbm/.zattrs["used"]``) was
unchanged.

What this module adds beyond Imspector:

* **per-bead quality** — each bead's drift-only deviation from the median of all
  beads (:class:`BeadInfo.deviation_nm`), so a misbehaving bead is visible instead
  of found by trial (the median, because a mean reference is pulled toward the
  bad bead and hides it);
* **a suggested selection** (:func:`suggest_beads`) that drops the outliers;
* **a median consensus** as an alternative to the mean, robust to one bad bead
  among three or more;
* **a reproduction check** (:func:`reproduction_rms_nm`): with the recorded bead
  set the result must reproduce the file's own ``loc``, and how closely it does
  is reported rather than assumed.

The beads see only what moves them — the stage and coverslip. Drift of the
structure relative to the beads is invisible to this method; the fiducial-free
time-window correction (:mod:`minflux_viewer.analysis.drift_correction`) can be
run afterwards on the bead-corrected coordinates to estimate that residual.

Pure NumPy/SciPy, Qt-free.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from ..msr import mbm_drift

#: A bead whose deviation from the bead consensus exceeds this many times the
#: median deviation (and :data:`SUGGEST_MIN_DEVIATION_NM`) is suggested for
#: removal. 2.5 flags the beads that are plainly unlike the rest (3.4x and 4.0x
#: in the 2026-06-25/26 data) and nothing else; it is an objective outlier rule,
#: not an attempt to reproduce an operator's judgement, which in that data also
#: weighed things a single number does not see.
SUGGEST_FACTOR = 2.5
SUGGEST_MIN_DEVIATION_NM = 2.0
_QUALITY_GRID = 2000           # samples of the common time grid for bead comparison

CONSENSUS_MODES = ("mean", "median")


@dataclass
class BeadInfo:
    gri: int
    name: str                       # R-ID ("R101") when the file names it
    n_samples: int
    span_s: float
    ptp_nm: tuple                   # peak-to-peak of the smoothed track, per axis
    deviation_nm: float             # rms of (this bead − median of all beads), offset removed
    recorded: bool                  # in the file's recorded selection


@dataclass
class BeadInputs:
    points: np.ndarray              # the MBM point record (gri, xyz [m], tim [s], str)
    lnc_m: np.ndarray               # (n, 3) uncorrected positions of the last-valid rows
    loc_m: np.ndarray               # (n, 3) the file's corrected positions, same rows
    t: np.ndarray                   # (n,) localization times
    beads: list[BeadInfo]
    recorded_gris: list[int]
    recorded_source: str            # where the recorded selection came from
    z_scaling_factor: float = 1.0
    names: dict = field(default_factory=dict)

    @property
    def usable_gris(self) -> list[int]:
        return [b.gri for b in self.beads]


# ---------------------------------------------------------------------------
# Inputs
# ---------------------------------------------------------------------------

def _points_of(ds):
    meta = getattr(ds, "metadata", {}) or {}
    points = meta.get("mbm_points")
    if points is None:
        mbm = getattr(ds, "mbm", None)
        try:
            points = mbm.get("points") if mbm is not None else None
        except (AttributeError, TypeError):
            points = None
    if points is None or getattr(points, "dtype", None) is None:
        return None
    if not {"gri", "xyz", "tim"} <= set(points.dtype.names or ()):
        return None
    return points


def _points_by_gri(meta) -> dict:
    by = meta.get("mbm_points_by_gri")
    if not by:
        by = (meta.get("native_zarr_mbm_points_attrs") or {}).get("points_by_gri")
    return by if isinstance(by, dict) else {}


def _recorded_selection(meta, usable: list[int]) -> tuple[list[int], str]:
    """The bead set the file records as used, and where it was found."""
    by_gri = _points_by_gri(meta)
    name_to_gri = {}
    for key, info in by_gri.items():
        if isinstance(info, dict) and info.get("name"):
            try:
                name_to_gri[str(info["name"])] = int(info.get("gri", key))
            except (TypeError, ValueError):
                continue
    for source, used in (("recorded selection (mbm 'used')",
                          (meta.get("native_zarr_mbm_attrs") or {}).get("used")),
                         ("recorded selection", meta.get("mbm_used"))):
        if used:
            gris = sorted({name_to_gri[str(r)] for r in used if str(r) in name_to_gri}
                          | {int(r) for r in used if isinstance(r, (int, np.integer))})
            gris = [g for g in gris if g in usable]
            if gris:
                return gris, source
    flagged = sorted(int(info.get("gri", k)) for k, info in by_gri.items()
                     if isinstance(info, dict) and info.get("used"))
    flagged = [g for g in flagged if g in usable]
    if flagged:
        return flagged, "per-bead 'used' flags"
    return list(usable), "all usable beads (no selection recorded)"


def bead_inputs_for_dataset(ds) -> tuple[BeadInputs | None, str]:
    """``(inputs, "")``, or ``(None, reason)`` when the dataset cannot be
    re-corrected with its beads."""
    from minflux_viewer.core.loader_raw_access import mfx_get

    points = _points_of(ds)
    if points is None:
        return None, "the dataset carries no MBM bead record"
    cols = {}
    for name in ("lnc_x", "lnc_y", "lnc_z", "loc_x", "loc_y", "loc_z", "tim"):
        v = mfx_get(ds, name, itr="last", vld_only=True)
        cols[name] = None if v is None else np.asarray(v, dtype=float).ravel()
    if any(cols[k] is None for k in ("lnc_x", "lnc_y")):
        return None, ("the dataset has no uncorrected positions (lnc), so the "
                      "correction already in its coordinates cannot be redone")
    if cols["tim"] is None:
        return None, "the dataset has no localization times (tim)"
    n = cols["lnc_x"].size
    zeros = np.zeros(n)
    lnc = np.column_stack([cols["lnc_x"], cols["lnc_y"],
                           cols["lnc_z"] if cols["lnc_z"] is not None else zeros])
    loc = np.column_stack([cols["loc_x"], cols["loc_y"],
                           cols["loc_z"] if cols["loc_z"] is not None else zeros])

    meta = getattr(ds, "metadata", {}) or {}
    all_gris = sorted(int(g) for g in np.unique(np.asarray(points["gri"]).ravel()))
    usable = mbm_drift.usable_bead_gris(points, all_gris)
    if not usable:
        return None, "no bead has a usable track"
    recorded, source = _recorded_selection(meta, usable)
    names = {}
    for key, info in _points_by_gri(meta).items():
        if isinstance(info, dict) and info.get("name"):
            try:
                names[int(info.get("gri", key))] = str(info["name"])
            except (TypeError, ValueError):
                pass
    zsf = float(getattr(getattr(ds, "cali", None), "z_scaling_factor", 1.0) or 1.0)
    inputs = BeadInputs(points=points, lnc_m=lnc, loc_m=loc, t=cols["tim"],
                        beads=[], recorded_gris=recorded, recorded_source=source,
                        z_scaling_factor=zsf, names=names)
    inputs.beads = bead_quality(points, usable, recorded=recorded, names=names)
    return inputs, ""


# ---------------------------------------------------------------------------
# Quality and selection
# ---------------------------------------------------------------------------

def _common_grid(points, gris) -> np.ndarray:
    lo, hi = np.inf, -np.inf
    for g in gris:
        tim, _ = mbm_drift.bead_trace(points, g)
        if len(tim):
            lo, hi = min(lo, tim[0]), max(hi, tim[-1])
    if not np.isfinite(lo) or hi <= lo:
        return np.empty(0)
    return np.linspace(lo, hi, _QUALITY_GRID)


def _tracks_on(points, gris, grid, smooth_samples, *, centred: bool = True) -> np.ndarray:
    """(k, len(grid), 3) smoothed bead tracks in nm.

    ``centred`` subtracts each bead's own mean position: beads sit micrometres
    apart, so a median over *absolute* positions would simply pick the bead in
    the middle by position. Only the motion is comparable.
    """
    out = []
    for g in gris:
        tim, xyz = mbm_drift.bead_trace(points, g)
        sm = mbm_drift._smooth(xyz, smooth_samples)
        tr = np.column_stack([np.interp(grid, tim, sm[:, k]) for k in range(3)]) * 1e9
        out.append(tr - tr.mean(axis=0) if centred else tr)
    return np.asarray(out)


def bead_quality(points, gris, *, recorded=(), names=None,
                 smooth_samples: float = mbm_drift.DEFAULT_SMOOTH_SAMPLES) -> list[BeadInfo]:
    """One :class:`BeadInfo` per bead in *gris* (usable ones)."""
    names = names or {}
    grid = _common_grid(points, gris)
    tracks = _tracks_on(points, gris, grid, smooth_samples) if grid.size else None
    consensus = np.median(tracks, axis=0) if tracks is not None else None
    infos = []
    for i, g in enumerate(gris):
        tim, _ = mbm_drift.bead_trace(points, g)
        dev = float("nan")
        ptp = (float("nan"),) * 3
        if tracks is not None:
            own = tracks[i]
            ptp = tuple(float(v) for v in np.ptp(own, axis=0))
            if len(gris) > 1:
                diff = own - consensus
                diff -= diff.mean(axis=0)          # a constant offset is the bead's position
                dev = float(np.sqrt(np.mean(np.sum(diff ** 2, axis=1))))
        infos.append(BeadInfo(
            gri=int(g), name=names.get(int(g), f"gri {g}"), n_samples=int(len(tim)),
            span_s=float(tim[-1] - tim[0]) if len(tim) > 1 else 0.0,
            ptp_nm=ptp, deviation_nm=dev, recorded=int(g) in set(recorded)))
    return infos


def suggest_beads(beads: list[BeadInfo]) -> list[int]:
    """The usable beads minus outliers: a deviation above
    ``SUGGEST_FACTOR ×`` the median deviation and above
    ``SUGGEST_MIN_DEVIATION_NM``. Needs three beads to judge; otherwise all."""
    usable = [b for b in beads if np.isfinite(b.deviation_nm)]
    if len(usable) < 3:
        return [b.gri for b in beads]
    med = float(np.median([b.deviation_nm for b in usable]))
    limit = max(SUGGEST_FACTOR * med, SUGGEST_MIN_DEVIATION_NM)
    keep = [b.gri for b in beads if not (np.isfinite(b.deviation_nm) and b.deviation_nm > limit)]
    return keep or [b.gri for b in beads]


# ---------------------------------------------------------------------------
# Correction
# ---------------------------------------------------------------------------

def consensus_drift(points, gris, times, *, consensus: str = "mean",
                    smooth_samples: float = mbm_drift.DEFAULT_SMOOTH_SAMPLES) -> np.ndarray:
    """``D(t)`` in metres at *times*.

    ``"mean"`` is Imspector's computation exactly (:func:`mbm_drift.drift_at`).
    ``"median"`` takes the per-axis median of the beads' *motion* (each track
    centred on its own mean) on their common time grid and interpolates it —
    robust to one deviating bead among three or more. Its absolute level is
    arbitrary; only ``D(t) − D(t_ref)`` is used.
    """
    times = np.asarray(times, dtype=float).ravel()
    if consensus == "mean":
        return mbm_drift.drift_at(points, gris, times, smooth_samples=smooth_samples)
    if consensus != "median":
        raise ValueError(f"unknown consensus {consensus!r}")
    usable = mbm_drift.usable_bead_gris(points, gris)
    if not usable:
        raise mbm_drift.DriftCorrectionError("None of the selected beads has a usable track.")
    grid = np.unique(np.concatenate([mbm_drift.bead_trace(points, g)[0] for g in usable]))
    tracks = _tracks_on(points, usable, grid, smooth_samples) * 1e-9
    med = np.median(tracks, axis=0)
    return np.column_stack([np.interp(times, grid, med[:, k]) for k in range(3)])


@dataclass
class BeadCorrection:
    positions_m: np.ndarray        # (n, 3) corrected positions (raw z, metres)
    gris: list[int]
    consensus: str
    smooth_samples: float
    curve_t: np.ndarray            # dense time grid (s)
    curve_nm: np.ndarray           # (len, 3) drift removed, zeroed at the start
    shift_from_file_nm: np.ndarray  # (n,) |new − file's loc|
    reproduction_nm: tuple         # rms per axis vs the file's loc, constant offset removed
    offset_nm: tuple               # that constant offset


def correct_with_beads(inputs: BeadInputs, gris, *, consensus: str = "mean",
                       smooth_samples: float = mbm_drift.DEFAULT_SMOOTH_SAMPLES) -> BeadCorrection:
    """``lnc − (D(t) − D(t_start))`` for the selected beads."""
    gris = sorted(int(g) for g in gris)
    if not gris:
        raise mbm_drift.DriftCorrectionError("Select at least one bead.")
    t = inputs.t
    finite = np.isfinite(t)
    t_ref = float(t[finite].min()) if finite.any() else 0.0
    drift = consensus_drift(inputs.points, gris, t, consensus=consensus,
                            smooth_samples=smooth_samples)
    ref = consensus_drift(inputs.points, gris, [t_ref], consensus=consensus,
                          smooth_samples=smooth_samples)[0]
    positions = inputs.lnc_m - (drift - ref)

    span = t[finite]
    curve_t = np.linspace(float(span.min()), float(span.max()), 1500) if span.size else np.empty(0)
    curve = ((consensus_drift(inputs.points, gris, curve_t, consensus=consensus,
                              smooth_samples=smooth_samples) - ref) * 1e9
             if curve_t.size else np.empty((0, 3)))
    delta = positions - inputs.loc_m
    moved = np.linalg.norm(delta, axis=1) * 1e9
    rms, offset = reproduction_rms_nm(positions, inputs.loc_m)
    return BeadCorrection(positions_m=positions, gris=gris, consensus=consensus,
                          smooth_samples=float(smooth_samples), curve_t=curve_t,
                          curve_nm=curve, shift_from_file_nm=moved,
                          reproduction_nm=rms, offset_nm=offset)


def reproduction_rms_nm(positions_m: np.ndarray, loc_m: np.ndarray) -> tuple[tuple, tuple]:
    """Per-axis rms of ``positions − loc`` (nm) once the constant offset is
    removed, and that offset. The offset is the reference convention (a rigid
    shift of the whole dataset); the rms is the disagreement that matters."""
    d = (np.asarray(positions_m, float) - np.asarray(loc_m, float)) * 1e9
    ok = np.all(np.isfinite(d), axis=1)
    if not ok.any():
        nan = (float("nan"),) * 3
        return nan, nan
    d = d[ok]
    off = np.median(d, axis=0)
    rms = np.sqrt(np.mean((d - off) ** 2, axis=0))
    return tuple(float(v) for v in rms), tuple(float(v) for v in off)


def display_coordinates_nm(positions_m: np.ndarray, z_scaling_factor: float) -> np.ndarray:
    """Positions as the viewer shows them: nm, with the Z scaling factor on z."""
    out = np.asarray(positions_m, float) * 1e9
    out[:, 2] *= float(z_scaling_factor)
    return out
