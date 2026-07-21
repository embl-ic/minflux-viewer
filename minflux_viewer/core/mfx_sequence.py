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
Abberior ``.msr`` (``mfx/.zattrs → measurement.threads[].sequences[0]``), the set
of photon-bearing iterations:

* ``photon_iters``  — every iteration at the final (minimum) beam scale; the sum
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
from typing import Optional

# Mode.pattern values that localize laterally (xy) vs axially (z).
_LATERAL_PATTERNS = {"square", "hexagon", "triangle"}
_AXIAL_PATTERNS = {"zline", "zline2", "zline3"}


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

    The final localization is at the minimum ``geo_factor`` (tightest pattern).
    ``photon_iters`` = all iterations sharing that minimum scale; the final
    lateral / axial iterations are the highest-index lateral / axial iterations
    within that final-scale set. ``None`` if scales are unusable.
    """
    if not iters:
        return None
    scales = [it.geo_factor for it in iters if it.geo_factor == it.geo_factor]  # drop NaN
    if not scales:
        return None
    tol = 1e-6
    min_scale = min(scales)
    final = [it for it in iters
             if it.geo_factor == it.geo_factor and abs(it.geo_factor - min_scale) <= tol * max(1.0, abs(min_scale))]
    if not final:
        return None
    photon_iters = sorted(it.index for it in final)
    lateral = [it.index for it in final if it.is_lateral]
    axial = [it.index for it in final if it.is_axial]
    lateral_iter = max(lateral) if lateral else (max(photon_iters) if photon_iters else None)
    axial_iter = max(axial) if axial else None
    return PhotonIterations(
        photon_iters=photon_iters, lateral_iter=lateral_iter, axial_iter=axial_iter,
        source="sequence", n_itr=len(iters),
        detail={"min_geo_factor": min_scale,
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

    Uses the acquisition sequence stored at load in ``ds.metadata["mfx_sequence"]``
    when available (robust, any sequence), else the dimensionality-based fallback
    from ``ds.prop.num_dim`` + ``ds.metadata["raw_num_itr"]``.
    """
    meta = getattr(ds, "metadata", {}) or {}
    n_itr = int(meta.get("raw_num_itr", getattr(getattr(ds, "prop", None), "num_itr", 1)) or 1)
    dims = int(getattr(getattr(ds, "prop", None), "num_dim", 2) or 2)
    sequence = meta.get("mfx_sequence")
    return detect_photon_iterations(sequence=sequence, n_itr=n_itr, dims=dims)


def extract_sequence_from_zattrs(mfx_zattrs: dict) -> Optional[dict]:
    """Pull the first usable sequence dict out of an ``mfx/.zattrs`` mapping.

    Walks ``measurement.threads[].sequences[]`` and returns the first sequence
    that carries an ``Itr`` list. ``None`` when absent.
    """
    try:
        threads = ((mfx_zattrs or {}).get("measurement") or {}).get("threads") or []
        for th in threads:
            for seq in (th.get("sequences") or []):
                if isinstance(seq, dict) and seq.get("Itr"):
                    return seq
    except (AttributeError, TypeError):
        pass
    return None
