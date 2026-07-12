"""
Bead (MBM) drift extraction for the MSR reader "Show Beads Drift" feature.

Each ``.msr`` dataset carries beam-monitoring (MBM) reference beads used for
drift correction. Per dataset we combine three zarr nodes:

- ``mbm`` attrs ``["used"]`` — the R-IDs of the beads actually used (e.g. ``R113``).
- ``grd/mbm/points`` attrs ``["points_by_gri"]`` — ``{gri: {"name": R-ID, …}}``,
  the digit gri-ID ↔ R-ID map.
- ``grd/mbm/points`` array — per-measurement ``gri`` / ``xyz`` (metres) / ``tim``
  (seconds).

:func:`extract_bead_drift` returns, for each *used* bead, its drift trace in
**nanometres** (re-zeroed to the per-bead median) with time zeroed to its start.
"""

from __future__ import annotations

import numpy as np


def _name_to_gri(points_by_gri: dict) -> dict[str, int]:
    out: dict[str, int] = {}
    for key, info in (points_by_gri or {}).items():
        if not isinstance(info, dict):
            continue
        rid = info.get("name")
        try:
            gri = int(info.get("gri", key))
        except (TypeError, ValueError):
            continue
        if rid:
            out.setdefault(str(rid), gri)
    return out


def extract_bead_drift(points, points_by_gri, used_rids, *, min_points: int = 1) -> list[dict]:
    """Per used bead, return ``{gri, rid, xyz_nm, tim_s, n}``.

    ``xyz_nm`` is (n×3) in nm, re-zeroed to the bead's per-axis **median**;
    ``tim_s`` is sorted ascending and zeroed to its start. When ``used_rids`` is
    empty, all beads in ``points_by_gri`` are used as a fallback.
    """
    if points is None or getattr(points, "dtype", None) is None:
        return []
    names = points.dtype.names or ()
    if not {"gri", "xyz", "tim"} <= set(names):
        return []
    name_to_gri = _name_to_gri(points_by_gri)
    # Fallback: no explicit "used" list → show every known bead.
    rids = list(used_rids) if used_rids else list(name_to_gri.keys())

    gri_arr = np.asarray(points["gri"]).ravel()
    xyz = np.asarray(points["xyz"], dtype=float)
    tim = np.asarray(points["tim"], dtype=float).ravel()
    if xyz.ndim != 2 or xyz.shape[1] < 3:
        return []

    beads: list[dict] = []
    for rid in rids:
        gri = name_to_gri.get(str(rid))
        if gri is None:
            continue
        mask = gri_arr == gri
        n = int(mask.sum())
        if n < min_points:
            continue
        xyz_nm = xyz[mask] * 1.0e9
        xyz_nm = xyz_nm - np.median(xyz_nm, axis=0)         # re-zero to median
        tim_s = tim[mask]
        order = np.argsort(tim_s)                            # chronological
        tim_s = tim_s[order]
        tim_s = tim_s - tim_s[0]                             # zero to start
        xyz_nm = xyz_nm[order]
        beads.append({"gri": int(gri), "rid": str(rid),
                      "xyz_nm": xyz_nm, "tim_s": tim_s, "n": n})
    beads.sort(key=lambda b: b["gri"])
    return beads


def gather_msr_bead_drift(datasets, mbm_map, meta_map=None) -> list[dict]:
    """Collect ``{"name", "beads"}`` per dataset that has bead data.

    *datasets* is the MSR parse result's ``datasets`` list (each with
    ``display_name`` + ``zroot``); *mbm_map* maps name → ``grd/mbm/points`` array.
    *meta_map* (name → translated legacy MBM metadata) is passed by the caller so
    the reader dialog stays independent of the shared global ``state``; it falls
    back to that global when not given (back-compat).
    """
    from ...msr.io import read_zarr_attrs

    if meta_map is None:
        from ...msr import state as _state
        meta_map = getattr(_state, "mbm_meta_map", {}) or {}
    meta_map = meta_map or {}
    out: list[dict] = []
    for d in datasets or []:
        name = d.get("display_name") or d.get("did") or "dataset"
        store = d.get("zroot")
        points = (mbm_map or {}).get(name)
        if store is None or points is None or not getattr(points, "size", 0):
            continue
        # Prefer the translated legacy MBM metadata; else read the modern in-store
        # zarr attrs.
        meta = meta_map.get(name) or {}
        pbg = meta.get("points_by_gri") or {}
        used = meta.get("used") or []
        if not pbg or not used:
            try:
                if not pbg:
                    pbg = (read_zarr_attrs(store, "grd/mbm/points") or {}).get("points_by_gri", {})
                if not used:
                    used = (read_zarr_attrs(store, "mbm") or {}).get("used", [])
            except Exception:
                pass
        beads = extract_bead_drift(points, pbg, used)
        if beads:
            out.append({"name": name, "beads": beads})
    return out
