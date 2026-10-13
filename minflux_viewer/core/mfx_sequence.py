"""
minflux_viewer.core.mfx_sequence
=================================
Detect which MINFLUX **loop iterations carry the localization photons**.

A MINFLUX acquisition runs a *sequence* of iterations that progressively tighten
the targeted beam pattern. The **final localization** is fixed at the smallest
pattern scale — and in a 3-D sequence that final fix is split across two
iterations: a lateral (xy) step and an axial (z) step. The photons that actually
determine the localization precision are those collected at these final-scale
iterations, **not** the whole per-iteration ``eco`` stream and **not** just the
single last iteration.

This module derives, from the acquisition **sequence definition** embedded in an
Abberior ``.msr`` (``mfx/.zattrs → measurement.threads[].sequences[0]`` in an
m2410 file, ``mfx/.zattrs → seq`` in an m2205 one), the set of photon-bearing
iterations:

* ``photon_iters``  — every iteration at its axis's final (minimum) beam scale; the sum
  of their ``eco`` is the localization photon count Imspector uses for its
  *Aggregation* photon threshold (verified ≈99 % against reference files).
* ``lateral_iter``  — the final lateral (xy) iteration → photon budget for the
  lateral CRLB precision.
* ``axial_iter``    — the final axial (z) iteration (3-D only) → photon budget for
  the axial CRLB precision.

When the sequence definition is unavailable (e.g. a ``.mat`` export that dropped
it), :func:`fallback_photon_iters` uses a dimensionality-based rule (3-D → the
last two iteration indices, 2-D → the last one), which reproduces the same result
for the standard Abberior 3-D sequence.

Iteration indices are **0-based** and refer to the ``itr`` values in the data.
Pure Python (no Qt / NumPy) so it is unit-testable in isolation.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from math import isfinite
from typing import Optional

# Mode.pattern values that localize laterally (xy) vs axially (z). An octahedron
# (Mode.id ``mx3d``: ±L/2 on each axis) localizes all three axes in ONE iteration,
# so it is both the final lateral and the final axial iteration.
_LATERAL_PATTERNS = {"square", "hexagon", "triangle"}
_AXIAL_PATTERNS = {"zline", "zline2", "zline3"}
_VOLUME_PATTERNS = {"octahedron"}


@dataclass
class IterationInfo:
    """One loop iteration of a MINFLUX sequence."""
    index: int                       # 0-based iteration index (matches ``itr``)
    geo_factor: float                # patGeoFactor (∝ pattern scale L; smaller = tighter)
    pattern: str = ""                # Mode.pattern (e.g. "square", "zline2")
    mode_id: str = ""                # Mode.id (e.g. "mxl3", "mxa3")
    dim: int = 0                     # Mode.dim
    is_lateral: bool = True          # localizes xy
    is_axial: bool = False           # localizes z


@dataclass
class PhotonIterations:
    """Detected photon-bearing iterations (0-based indices)."""
    photon_iters: list               # final-scale iterations (eco summed → localization photons)
    lateral_iter: Optional[int]      # final lateral (xy) iteration
    axial_iter: Optional[int]        # final axial (z) iteration, or None (2-D)
    source: str = "sequence"         # "sequence" | "fallback"
    n_itr: int = 0
    detail: dict = field(default_factory=dict)


def _as_float(value) -> float:
    """Reduce a patGeoFactor (scalar or list) to a comparable scalar scale."""
    if isinstance(value, (list, tuple)):
        nums = [float(v) for v in value if isinstance(v, (int, float))]
        return max(nums) if nums else float("nan")
    try:
        return float(value)
    except (TypeError, ValueError):
        return float("nan")


def _classify(pattern: str, mode_id: str) -> tuple[bool, bool]:
    """(is_lateral, is_axial) from the pattern / mode id."""
    p = (pattern or "").lower()
    m = (mode_id or "").lower()
    if p in _VOLUME_PATTERNS or m == "mx3d":
        return True, True
    if p in _AXIAL_PATTERNS or m.startswith("mxa") or m == "mxax":
        return False, True
    if p in _LATERAL_PATTERNS or m.startswith("mxl"):
        return True, False
    # "pt3d" / unknown initial modes: treat as lateral (they set xy first).
    return True, False


def parse_sequence(sequence: dict) -> list[IterationInfo]:
    """Parse the ``Itr`` list of one sequence dict into :class:`IterationInfo`.

    ``sequence`` is ``measurement.threads[i].sequences[j]`` from an ``.msr``
    ``mfx/.zattrs``. Returns an empty list if it carries no usable ``Itr`` list.
    """
    itrs = (sequence or {}).get("Itr") or []
    out: list[IterationInfo] = []
    for i, it in enumerate(itrs):
        mode = it.get("Mode") or {} if isinstance(it, dict) else {}
        pattern = str(mode.get("pattern", ""))
        mode_id = str(mode.get("id", ""))
        dim = int(mode.get("dim", 0) or 0)
        geo = _as_float(it.get("patGeoFactor")) if isinstance(it, dict) else float("nan")
        lat, ax = _classify(pattern, mode_id)
        out.append(IterationInfo(index=i, geo_factor=geo, pattern=pattern,
                                 mode_id=mode_id, dim=dim, is_lateral=lat, is_axial=ax))
    return out


def detect_from_sequence(iters: list[IterationInfo]) -> Optional[PhotonIterations]:
    """Photon-bearing iterations from parsed :class:`IterationInfo`.

    Lateral and axial scales are minimized independently: a sequence can end
    with different XY and Z extents. ``photon_iters`` retains every iteration
    at either axis's own minimum for photon pooling; each CRLB axis uses the
    last iteration at its own minimum. ``None`` if scales are unusable.
    """
    if not iters:
        return None
    usable = [it for it in iters if isfinite(it.geo_factor) and it.geo_factor > 0]
    if not usable:
        return None
    tol = 1e-6

    def final_for_axis(flag):
        candidates = [it for it in usable if getattr(it, flag)]
        if not candidates:
            return [], None
        minimum = min(it.geo_factor for it in candidates)
        return [it for it in candidates
                if abs(it.geo_factor - minimum) <= tol * max(1.0, abs(minimum))], minimum

    lateral, lateral_scale = final_for_axis("is_lateral")
    axial, axial_scale = final_for_axis("is_axial")
    final = sorted({it.index: it for it in lateral + axial}.values(), key=lambda it: it.index)
    if not final:
        return None
    photon_iters = sorted(it.index for it in final)
    lateral_iter = max(it.index for it in lateral) if lateral else None
    axial_iter = max(it.index for it in axial) if axial else None
    return PhotonIterations(
        photon_iters=photon_iters, lateral_iter=lateral_iter, axial_iter=axial_iter,
        source="sequence", n_itr=len(iters),
        detail={"min_geo_factor": min(it.geo_factor for it in usable),
                "min_lateral_geo_factor": lateral_scale,
                "min_axial_geo_factor": axial_scale,
                "modes": {it.index: (it.mode_id, it.pattern) for it in final}})


def fallback_photon_iters(n_itr: int, dims: int) -> PhotonIterations:
    """Dimensionality-based fallback when the sequence definition is absent.

    3-D → the last two iteration indices (final lateral + final axial), matching
    the standard Abberior 3-D sequence; 2-D → the last iteration only.
    """
    n = max(int(n_itr), 1)
    last = n - 1
    if dims >= 3 and n >= 2:
        return PhotonIterations(
            photon_iters=[last - 1, last], lateral_iter=last - 1, axial_iter=last,
            source="fallback", n_itr=n, detail={"rule": "3D: last two iterations"})
    return PhotonIterations(
        photon_iters=[last], lateral_iter=last, axial_iter=None,
        source="fallback", n_itr=n, detail={"rule": "2D: last iteration"})


def detect_photon_iterations(
    *,
    sequence: Optional[dict] = None,
    n_itr: int = 0,
    dims: int = 2,
) -> PhotonIterations:
    """Best available detection: sequence definition if usable, else fallback."""
    if sequence:
        info = detect_from_sequence(parse_sequence(sequence))
        if info is not None and info.photon_iters:
            return info
    return fallback_photon_iters(n_itr, dims)


def photon_iterations_for_dataset(ds) -> PhotonIterations:
    """Photon-bearing iterations for a loaded dataset.

    Uses the dataset's acquisition sequence (:func:`sequence_for_dataset`) when
    available (robust, any sequence), else the dimensionality-based fallback from
    ``ds.prop.num_dim`` + ``ds.metadata["raw_num_itr"]``.
    """
    meta = getattr(ds, "metadata", {}) or {}
    n_itr = int(meta.get("raw_num_itr", getattr(getattr(ds, "prop", None), "num_itr", 1)) or 1)
    dims = int(getattr(getattr(ds, "prop", None), "num_dim", 2) or 2)
    return detect_photon_iterations(sequence=sequence_for_dataset(ds), n_itr=n_itr, dims=dims)


def sequence_for_dataset(ds) -> Optional[dict]:
    """The acquisition sequence of *ds*, or ``None``.

    ``metadata["mfx_sequence"]`` is stamped at import; a dataset stamped before the
    m2205 location (``mfx/.zattrs → seq``) was known still carries the native
    ``mfx`` attributes, so the sequence is recovered from those.
    """
    meta = getattr(ds, "metadata", {}) or {}
    seq = meta.get("mfx_sequence")
    if isinstance(seq, dict) and seq.get("Itr"):
        return seq
    return extract_sequence_from_zattrs(meta.get("native_zarr_mfx_attrs"))


# ---------------------------------------------------------------------------
# Targeted-pattern geometry of one iteration (for the Cramér-Rao bound)
# ---------------------------------------------------------------------------

#: Abberior stores the pattern size as a dimensionless ``patGeoFactor``; the
#: physical pattern diameter is ``patGeoFactor × 360 nm``. This is the conversion
#: SimuFLUX uses to simulate Abberior sequences (Marin & Ries, *Nat. Commun.*
#: 17:246, 2026; ``getAbberiorPattern``).
PATTERN_SCALE_NM: float = 360.0

#: Final ``patGeoFactor`` of the standard Abberior imaging sequences — every 2-D
#: and 3-D imaging sequence in the sample data ends at 0.11 (L = 39.6 nm). It is
#: the *assumption* used when a file carries no sequence (a ``.mat``/``.npy``
#: export), and it is reported as such.
DEFAULT_FINAL_GEO_FACTOR: float = 0.11

# Exposure positions of each pattern, excluding the optional centre exposure.
_PATTERN_POINTS = {"triangle": 3, "square": 4, "hexagon": 6,
                   "zline": 4, "zline2": 2, "octahedron": 6}

# Geometry factor G per axis in σ = L·G·F / (4√N) (quadratic minimum, emitter at
# the centre, K equal-dwell exposures at radius L/2): G² = K·(L/2)² / Σ p_axis²
# over the exposure positions. A planar orbit gives √2 laterally (σ = L/√(8N),
# Marin & Ries Eq. 45 without the envelope term), a z-line pair 1 axially (their
# Eq. 26/29), and an octahedron √3 on every axis — its photons are split over
# three axes. NaN: the pattern carries no information on that axis.
_SQRT2, _SQRT3 = 2.0 ** 0.5, 3.0 ** 0.5
_GEOMETRY_FACTORS = {"triangle": (_SQRT2, float("nan")), "square": (_SQRT2, float("nan")),
                     "hexagon": (_SQRT2, float("nan")), "zline": (float("nan"), 1.0),
                     "zline2": (float("nan"), 1.0), "octahedron": (_SQRT3, _SQRT3)}
_PLANAR_DONUT_PATTERNS = {"triangle", "square", "hexagon"}


@dataclass(frozen=True)
class PatternGeometry:
    """The targeted pattern of one iteration, as far as the CRB needs it.

    ``centre_probed`` / ``centre_dwell_ratio`` are ``None`` when unknown (no
    sequence): the caller then decides from the data whether a centre exposure
    was recorded (``ecc``/``efc``) and estimates its dwell from the counts.
    """
    L_nm: float                               # pattern diameter (lateral) / axial extent
    pattern: str = ""                         # Mode.pattern, e.g. "hexagon", "zline2"
    n_points: int = 0                         # exposure positions besides the centre
    centre_probed: Optional[bool] = None      # ccrLimit != -1 (SimuFLUX's rule)
    centre_dwell_ratio: Optional[float] = None  # centre dwell ÷ whole-orbit dwell
    iteration: Optional[int] = None           # 0-based itr the geometry belongs to
    from_sequence: bool = False               # False → assumed defaults
    power: Optional[float] = None             # relative laser power (pwrFactor)
    multi_radius: bool = False               # unequal patGeoFactor entries

    @property
    def unsupported_reason(self) -> str:
        """Why the centred closed form cannot describe this recorded pattern."""
        if self.multi_radius:
            return "multi-radius pattern requires an exposure-resolved Fisher model"
        if self.pattern.lower() not in _GEOMETRY_FACTORS:
            if self.pattern or self.from_sequence:
                return "unknown acquisition pattern: " + (self.pattern or "not named")
        if not isfinite(self.L_nm) or self.L_nm <= 0:
            return "pattern extent must be finite and positive"
        return ""

    @property
    def supported(self) -> bool:
        """Whether the current model supports this geometry (not a calibration claim)."""
        return not self.unsupported_reason

    @property
    def source(self) -> str:
        return "from the acquisition sequence" if self.from_sequence else "assumed: not recorded in the file"

    @property
    def geometry_factors(self) -> tuple[float, float]:
        """``(lateral, axial)`` geometry factor G; see ``_GEOMETRY_FACTORS``.

        Only a genuinely unrecorded pattern keeps the assumed planar / axial
        defaults. An unsupported recorded pattern returns NaNs, not a plausible
        bound computed with a different geometry.
        """
        if not self.supported:
            return float("nan"), float("nan")
        return _GEOMETRY_FACTORS.get(self.pattern.lower(), (_SQRT2, 1.0))

    @property
    def donut_envelope(self) -> bool:
        """Whether the 2-D donut's envelope term (1 − L²/8σ_q²)⁻¹ applies.

        It belongs to the lateral donut of Marin & Ries Eq. 42 — a planar orbit,
        or an unrecorded pattern read as one. A volume pattern (octahedron) uses
        a 3-D focal field the paper does not model, so it keeps the quadratic
        minimum alone.
        """
        p = self.pattern.lower()
        return self.supported and (p in _PLANAR_DONUT_PATTERNS or not p)


def _geo_to_nm(value) -> float:
    """Pattern size in nm from a ``patGeoFactor`` scalar or list.

    A list contributes its smallest extent for reporting only. Unequal radii
    are marked unsupported by :func:`pattern_geometry`; reducing them to one
    extent is not sufficient to use all their photons in a two-point bound.
    """
    if isinstance(value, (list, tuple)):
        nums = [float(v) for v in value if isinstance(v, (int, float))]
        return min(nums) * PATTERN_SCALE_NM if nums else float("nan")
    try:
        return float(value) * PATTERN_SCALE_NM
    except (TypeError, ValueError):
        return float("nan")


def pattern_geometry(sequence: Optional[dict], iteration: Optional[int]) -> Optional[PatternGeometry]:
    """:class:`PatternGeometry` of 0-based *iteration* of *sequence*, or ``None``.

    The centre exposure lasts ``ctrDwellFactor`` times the whole orbit (SimuFLUX:
    centre dwell = point dwell × points × ``ctrDwellFactor``), so its dwell ratio
    is that factor; it is probed only when the iteration's ``ccrLimit`` is not −1.
    """
    itrs = (sequence or {}).get("Itr") or []
    if iteration is None or not (0 <= int(iteration) < len(itrs)):
        return None
    it = itrs[int(iteration)]
    if not isinstance(it, dict):
        return None
    L = _geo_to_nm(it.get("patGeoFactor"))
    if not (L == L and L > 0):           # NaN or non-positive
        return None
    mode = it.get("Mode") or {}
    pattern = str(mode.get("pattern", "") or "")
    try:
        ccr = float(it.get("ccrLimit", -1))
    except (TypeError, ValueError):
        ccr = -1.0
    probed = ccr != -1.0
    try:
        ctr = float((sequence or {}).get("ctrDwellFactor", 1.0))
    except (TypeError, ValueError):
        ctr = 1.0
    return PatternGeometry(
        L_nm=L, pattern=pattern, n_points=_PATTERN_POINTS.get(pattern.lower(), 0),
        centre_probed=probed, centre_dwell_ratio=(ctr if probed else 0.0),
        iteration=int(iteration), from_sequence=True,
        power=iteration_power(sequence, iteration),
        multi_radius=_has_multiple_radii(it.get("patGeoFactor")))


def _has_multiple_radii(value) -> bool:
    """Whether a recorded scale vector cannot be represented by one radius."""
    if not isinstance(value, (list, tuple)) or len(value) < 2:
        return False
    try:
        numbers = [float(v) for v in value]
    except (TypeError, ValueError):
        return True
    return (any(not isfinite(v) or v <= 0 for v in numbers)
            or any(abs(v - numbers[0]) > 1e-12 * max(1.0, abs(numbers[0]))
                   for v in numbers[1:]))


def iteration_power(sequence: Optional[dict], iteration: Optional[int]) -> Optional[float]:
    """Relative excitation power of 0-based *iteration*, or ``None``.

    ``pwrFactor`` is the relative power both file versions record (and the one
    SimuFLUX simulates with); ``_excitation.power``, the absolute setting of an
    m2410 sequence, is proportional to it and is the fallback.
    """
    itrs = (sequence or {}).get("Itr") or []
    if iteration is None or not (0 <= int(iteration) < len(itrs)):
        return None
    it = itrs[int(iteration)]
    if not isinstance(it, dict):
        return None
    for value in (it.get("pwrFactor"), (it.get("_excitation") or {}).get("power")):
        try:
            v = float(value)
        except (TypeError, ValueError):
            continue
        if v > 0:
            return v
    return None


def centre_reference_iteration(sequence: Optional[dict], iteration: Optional[int]) -> Optional[int]:
    """The last iteration at or before *iteration* that probes the centre.

    MINFLUX records the centre exposure (``efc``/``ecc``) only at iterations whose
    ``ccrLimit`` is set; a 3-D imaging sequence probes it at an intermediate
    lateral iteration and not in its final pair. The latest such iteration has
    the best-centred emitter, so its centre rate carries the least residual
    fluorescence. ``None`` when the sequence probes the centre nowhere before.
    """
    itrs = (sequence or {}).get("Itr") or []
    if iteration is None:
        return None
    for k in range(min(int(iteration), len(itrs) - 1), -1, -1):
        it = itrs[k]
        if not isinstance(it, dict):
            continue
        try:
            if float(it.get("ccrLimit", -1)) != -1.0:
                return k
        except (TypeError, ValueError):
            continue
    return None


def background_sensing(sequence: Optional[dict]) -> Optional[bool]:
    """The sequence's ``bgcSense`` switch, or ``None`` when not recorded.

    ``fbg`` is a measurement only with background sensing on. With it off the
    instrument writes 0 or a placeholder (6.5535e7 Hz — 65535 × 1000 — in the
    m2205 sample files), never a background rate.
    """
    if not isinstance(sequence, dict) or "bgcSense" not in sequence:
        return None
    return bool(sequence.get("bgcSense"))


def assumed_pattern_geometry(iteration: Optional[int] = None) -> PatternGeometry:
    """The geometry assumed when the file records no sequence (reported as such)."""
    return PatternGeometry(L_nm=DEFAULT_FINAL_GEO_FACTOR * PATTERN_SCALE_NM,
                           iteration=iteration, from_sequence=False)


def crlb_geometry_for_dataset(ds) -> tuple[PhotonIterations, PatternGeometry, Optional[PatternGeometry]]:
    """Photon iterations plus the lateral and (3-D) axial pattern geometry.

    The axial geometry is ``None`` for a 2-D dataset. Each geometry falls back to
    :func:`assumed_pattern_geometry` when the sequence does not describe it.
    """
    pit = photon_iterations_for_dataset(ds)
    sequence = sequence_for_dataset(ds)
    lateral = pattern_geometry(sequence, pit.lateral_iter) or assumed_pattern_geometry(pit.lateral_iter)
    if sequence is not None and pit.lateral_iter is None:
        # A recorded axial-only sequence is not missing metadata: it supplies
        # no lateral measurement to which the planar default could apply.
        lateral = PatternGeometry(L_nm=DEFAULT_FINAL_GEO_FACTOR * PATTERN_SCALE_NM,
                                  pattern="no lateral measurement", from_sequence=True)
    axial = None
    if pit.axial_iter is not None:
        axial = pattern_geometry(sequence, pit.axial_iter) or assumed_pattern_geometry(pit.axial_iter)
    return pit, lateral, axial


def extract_sequence_from_zattrs(mfx_zattrs: dict) -> Optional[dict]:
    """Pull the first usable sequence dict out of an ``mfx/.zattrs`` mapping.

    An m2410 file keeps it under ``measurement.threads[].sequences[]``; an m2205
    file directly under ``seq``. Returns the first one that carries an ``Itr``
    list, ``None`` when absent.
    """
    try:
        threads = ((mfx_zattrs or {}).get("measurement") or {}).get("threads") or []
        for th in threads:
            for seq in (th.get("sequences") or []):
                if isinstance(seq, dict) and seq.get("Itr"):
                    return seq
        seq = (mfx_zattrs or {}).get("seq")
        if isinstance(seq, dict) and seq.get("Itr"):
            return seq
    except (AttributeError, TypeError):
        pass
    return None
