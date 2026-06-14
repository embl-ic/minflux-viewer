"""
minflux_viewer.core.dataset_kind
=================================
Defines what makes a dataset a **MINFLUX** dataset and exposes per-dataset
*capabilities* so functions/plugins can gate on exactly what they need.

The viewer loads both MINFLUX and non-MINFLUX (e.g. generic spreadsheet)
datasets. The minimum to be MINFLUX is **localizations + a real trace id**
(``tid``). Non-MINFLUX datasets keep the basic functions (render, scatter,
histogram, filter) but the MINFLUX-specific plugins are disabled with a clear
message rather than crashing.

Essential attributes (always present after a successful load):
    xnm, ynm, znm   localization coordinates in nm (znm zero-filled if 2-D)
    tid             trace id (synthetic 1..N for non-MINFLUX data)
    idx             1-based localization index (derived)
Important-but-optional: tim (tracking/dynamics), and the MINFLUX quality
attributes efo / cfr / dcr / fbg (quality filtering).
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np

if TYPE_CHECKING:
    from .dataset import MinfluxDataset

#: Source-version tags written by the native MINFLUX loaders.
_MINFLUX_SOURCES = {"m2410", "m2205", "legacy", "json", "plain_array"}

#: The MINFLUX quality attributes used by quality filtering.
QUALITY_ATTRS = ("efo", "cfr", "dcr", "fbg")

#: Capability names (see :func:`capabilities`).
CAPABILITIES = ("loc", "traces", "time", "quality", "3d")


def _coord_count(ds: "MinfluxDataset") -> int:
    a = ds.attr
    for k in ("loc_x", "loc_x_nm", "xnm"):
        v = a.get(k)
        if v is not None:
            return int(np.asarray(v).size)
    return int(getattr(ds.prop, "num_loc", 0) or 0)


def has_loc(ds: "MinfluxDataset") -> bool:
    """Localizations (x, y) are available."""
    return _coord_count(ds) > 0


def has_traces(ds: "MinfluxDataset") -> bool:
    """A *real* trace id is available (not a synthetic 1..N index)."""
    md = ds.metadata
    if "has_real_tid" in md:
        return bool(md["has_real_tid"])
    return ds.attr.get("tid") is not None       # native loaders always carry tid


def has_time(ds: "MinfluxDataset") -> bool:
    """A non-trivial time/`tim` attribute is available."""
    tim = ds.attr.get("tim")
    if tim is None:
        return False
    try:
        return bool(np.any(np.asarray(tim, dtype=float) != 0))
    except (TypeError, ValueError):
        return False


def has_quality(ds: "MinfluxDataset") -> bool:
    """At least one MINFLUX quality attribute (efo/cfr/dcr/fbg) is available."""
    return any(ds.attr.get(k) is not None for k in QUALITY_ATTRS)


def is_3d(ds: "MinfluxDataset") -> bool:
    return int(getattr(ds.prop, "num_dim", 2) or 2) >= 3


def is_minflux(ds: "MinfluxDataset") -> bool:
    """A dataset is MINFLUX when it has localizations **and** a real trace id.

    Honours an explicit ``metadata['is_minflux']`` flag (set by importers);
    otherwise native MINFLUX source tags qualify, falling back to the
    loc + real-tid test.
    """
    md = ds.metadata
    if "is_minflux" in md:
        return bool(md["is_minflux"])
    if str(md.get("source_version", "")).lower() in _MINFLUX_SOURCES:
        return True
    return has_loc(ds) and has_traces(ds)


_CAP_FUNCS = {
    "loc": has_loc,
    "traces": has_traces,
    "time": has_time,
    "quality": has_quality,
    "3d": is_3d,
}


def capabilities(ds: "MinfluxDataset") -> dict[str, bool]:
    """Boolean map of every capability for *ds*."""
    return {name: fn(ds) for name, fn in _CAP_FUNCS.items()}


def require(ds: "MinfluxDataset", *caps: str) -> list[str]:
    """Return the names of the requested capabilities that *ds* lacks.

    Empty list ⇒ all satisfied. Use to gate a plugin::

        missing = require(ds, "loc", "traces")
        if missing:
            warn(f"needs: {', '.join(missing)}"); return
    """
    out: list[str] = []
    for cap in caps:
        fn = _CAP_FUNCS.get(cap)
        if fn is not None and not fn(ds):
            out.append(cap)
    return out


#: Human-readable explanation for each capability (for gate messages).
CAPABILITY_REASON = {
    "loc": "localization coordinates (x, y)",
    "traces": "a trace id (tid) — this is a non-MINFLUX dataset",
    "time": "a time attribute (tim)",
    "quality": "MINFLUX quality attributes (efo / cfr / dcr / fbg)",
    "3d": "3-D data (z)",
}


def missing_reason(caps: list[str]) -> str:
    """Join capability names into a readable 'requires …' clause."""
    return "; ".join(CAPABILITY_REASON.get(c, c) for c in caps)
