"""
minflux_viewer.core.particle_extract
=====================================
Extract :class:`~minflux_viewer.core.particle_set.ParticleData` particles from a
dataset + a set of (rectangle) ROIs — the shared logic behind both the Particle
Average plugin's *Collect from active dataset* and its *Load data + ROI pair*
options.

Each rectangle ROI's localizations are cropped (filter mask honoured), **re-zeroed**
(XY to the box centre, Z to the median), and captured together with their trace
id, sub-unit id, the raw metres localization (provenance), and — optionally —
every available per-localization attribute. Pure / Qt-light (only pulls Qt via
``core.roi`` when parsing ROI JSON).
"""

from __future__ import annotations

from pathlib import Path

import numpy as np

from .loader import attr_values_1d
from .overlay import transform_to_matrix4
from .particle_set import ParticleData, _RESERVED_COLUMNS
from .roi_crop import display_coords
from .roi_selection import rectangle_bounds, rectangle_mask

MIN_LOCS_PER_PARTICLE = 5

#: per-loc keys that are stored explicitly or are not meaningful as attributes.
_SKIP_ATTRS = _RESERVED_COLUMNS | {
    "loc", "loc_nm", "vld", "itr", "ftr", "filter_mask", "fbg_raw",
}


def _tid_unit_map(ds) -> dict:
    """trace_id → sub-unit id, from an NPC-detection (Wanlu) ``raw6`` if present."""
    stored = (getattr(ds, "derived", {}) or {}).get("npc_wanlu")
    if not stored:
        return {}
    raw6 = np.asarray(stored.get("raw6", []), dtype=float)
    if raw6.ndim != 2 or raw6.shape[0] == 0 or raw6.shape[1] < 6:
        return {}
    return {float(t): float(u) for t, u in zip(raw6[:, 5], raw6[:, 4])}


def _raw_loc_m(ds, n: int) -> np.ndarray | None:
    """(n, 3) raw localization in **metres** aligned to the materialized rows."""
    loc = ds.attr.get("loc") if hasattr(getattr(ds, "attr", None), "get") else None
    if loc is not None:
        arr = np.asarray(loc, dtype=float)
        if arr.ndim == 2 and arr.shape[0] == n and arr.shape[1] >= 2:
            if arr.shape[1] >= 3:
                return arr[:, :3].copy()
            return np.column_stack([arr[:, :2], np.zeros(n)])
    cols = []
    for c in ("loc_x", "loc_y", "loc_z"):
        v = attr_values_1d(ds, c)
        cols.append(np.asarray(v, dtype=float).ravel() if v is not None else None)
    if cols[0] is not None and cols[1] is not None and cols[0].shape[0] == n:
        z = cols[2] if (cols[2] is not None and cols[2].shape[0] == n) else np.zeros(n)
        return np.column_stack([cols[0], cols[1], z])
    return None


def _capturable_attr_names(ds) -> list[str]:
    """Per-localization attribute names available on *ds* (raw + derived)."""
    names: list[str] = []
    seen: set[str] = set()
    for src in (getattr(ds, "attr", {}) or {}, getattr(ds, "derived", {}) or {}):
        try:
            keys = list(src.keys())
        except Exception:
            keys = []
        for k in keys:
            if k in seen or k in _SKIP_ATTRS:
                continue
            seen.add(k)
            names.append(k)
    return names


def _attr_column(ds, name: str, n: int) -> np.ndarray | None:
    """A length-*n* float64 per-loc column for *name*, or None if unusable."""
    v = attr_values_1d(ds, name)
    if v is None:
        v = (getattr(ds, "derived", {}) or {}).get(name)
    if v is None:
        return None
    arr = np.asarray(v)
    if arr.ndim != 1 or arr.shape[0] != n or not np.issubdtype(arr.dtype, np.number):
        return None
    return arr.astype(np.float64)


def extract_particles(
    ds,
    records,
    *,
    capture_attrs: bool = True,
    min_locs: int = MIN_LOCS_PER_PARTICLE,
) -> list[ParticleData]:
    """Crop + re-zero each rectangle ROI in *records* into a :class:`ParticleData`.

    Non-rectangle ROIs and ROIs with fewer than *min_locs* localizations are
    skipped. ``capture_attrs`` controls whether every per-loc attribute is
    captured (the raw metres localization is always captured when available).
    """
    coords = display_coords(ds)
    n = int(coords.shape[0]) if coords.ndim == 2 else 0
    if n == 0:
        return []

    fmask = np.asarray(getattr(ds, "filter_mask", np.ones(n, bool)), dtype=bool)
    if fmask.shape[0] != n:
        fmask = np.ones(n, dtype=bool)

    tid_all = attr_values_1d(ds, "tid")
    tid_all = np.asarray(tid_all).ravel() if tid_all is not None else None
    if tid_all is not None and tid_all.shape[0] != n:
        tid_all = None

    raw_loc = _raw_loc_m(ds, n)
    tid_to_unit = _tid_unit_map(ds)

    attr_cols: dict[str, np.ndarray] = {}
    if capture_attrs:
        for name in _capturable_attr_names(ds):
            col = _attr_column(ds, name, n)
            if col is not None:
                attr_cols[name] = col

    try:
        rimf = float(getattr(ds.cali, "RIMF", 1.0))
    except Exception:
        rimf = 1.0
    state = getattr(ds, "state", {}) or {}
    # Overlay datasets store the transform as a display_transform_record *dict*;
    # extract the bare 4×4 (np.asarray(dict) would raise) for the recipe.
    transform_mat4 = transform_to_matrix4(
        state.get("overlay_transform") or state.get("render_transform_2d"))
    src_ver = (getattr(ds, "metadata", {}) or {}).get("source_version", "")
    src_name = getattr(ds, "name", "")

    out: list[ParticleData] = []
    for rec in records:
        if getattr(rec, "type", None) != "rectangle":
            continue
        in_box = rectangle_mask(coords[:, 0], coords[:, 1], rec)
        mask = in_box & fmask
        if int(mask.sum()) < min_locs:
            continue
        pts = coords[mask].astype(float)
        b = rectangle_bounds(rec)
        cx, cy = 0.5 * (b[0] + b[1]), 0.5 * (b[2] + b[3])
        z_med = float(np.median(pts[:, 2]))
        rez = pts.copy()
        rez[:, 0] -= cx
        rez[:, 1] -= cy
        rez[:, 2] -= z_med

        tid = unit = None
        if tid_all is not None:
            tid = tid_all[mask]
            if tid_to_unit:
                raw_unit = np.array([tid_to_unit.get(float(t), np.nan) for t in tid])
            else:
                raw_unit = tid.astype(float)            # fallback: each trace = a sub-unit
            valid = np.isfinite(raw_unit)
            unit = np.zeros(tid.shape[0], dtype=int)
            if valid.any():
                _u, inv = np.unique(raw_unit[valid], return_inverse=True)
                unit[valid] = inv + 1

        meta = {
            "rezero_offset": [cx, cy, z_med],
            "rimf": rimf,
            "z_scale": 1.0,
            "source_version": str(src_ver),
        }
        if transform_mat4 is not None:
            meta["transform_4x4"] = transform_mat4

        out.append(ParticleData(
            points=rez,
            tid=tid,
            unit_id=unit,
            loc=(raw_loc[mask] if raw_loc is not None else None),
            attrs={k: v[mask] for k, v in attr_cols.items()},
            source=src_name,
            roi_name=getattr(rec, "name", ""),
            meta=meta,
        ))
    return out


def channel_points_in_roi(ds, record, rezero_offset) -> np.ndarray:
    """(M, 3) display-nm points of *ds* inside *record*'s rectangle, **re-zeroed by
    the reference particle's offset** (multi-channel averaging).

    Overlay channels are already aligned in display coordinates, so the same ROI
    box selects the co-located region in each channel; re-zeroing every channel by
    the *reference* offset (box centre + reference Z median) preserves the
    inter-channel geometry so the reference's per-particle transform can be replayed."""
    coords = display_coords(ds)
    n = int(coords.shape[0]) if coords.ndim == 2 else 0
    if n == 0:
        return np.empty((0, 3))
    fmask = np.asarray(getattr(ds, "filter_mask", np.ones(n, bool)), dtype=bool)
    if fmask.shape[0] != n:
        fmask = np.ones(n, dtype=bool)
    mask = rectangle_mask(coords[:, 0], coords[:, 1], record) & fmask
    pts = coords[mask].astype(float)
    off = np.asarray(rezero_offset, dtype=float).ravel()
    if pts.shape[0] and off.size >= 3:
        pts = pts - off[:3]
    return pts


def records_from_roi_json(path: str | Path) -> list:
    """Parse a native ROI-set ``.json`` into RoiRecords (no ROI store mutation)."""
    import json

    from .roi import RoiRecord

    data = json.loads(Path(path).read_text(encoding="utf-8"))
    items = data.get("rois", []) if isinstance(data, dict) else []
    return [RoiRecord(**item) for item in items]


def load_any_dataset(path: str | Path, prefs: dict | None = None):
    """Load a dataset file (.mat/.npy/.csv/.tsv/.txt/.json) into a MinfluxDataset.

    Used by the *data + ROI pair* loading option to crop particles from a file on
    disk without opening it in the viewer. RIMF/derived auto-computations done in
    the main window's post-load chain are NOT applied here — the dataset's default
    RIMF is recorded in the particle metadata.
    """
    from .loader import load_csv, load_dataset, load_json, load_npy

    ext = Path(path).suffix.lower()
    if ext == ".npy":
        return load_npy(path, prefs=prefs)
    if ext in (".csv", ".tsv", ".txt"):
        return load_csv(path, prefs=prefs)
    if ext == ".json":
        return load_json(path, prefs=prefs)
    return load_dataset(path, prefs=prefs)        # .mat and default
