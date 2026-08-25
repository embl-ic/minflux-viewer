"""
Bead (MBM) drift extraction for the MSR reader "Show Beads Drift" feature.

Each ``.msr`` dataset carries beam-monitoring (MBM) reference beads used for
drift correction. Per dataset we combine three zarr nodes:

- ``mbm`` attrs ``["used"]`` — the R-IDs of the beads actually used (e.g. ``R113``).
- ``grd/mbm/points`` attrs ``["points_by_gri"]`` — ``{gri: {"name": R-ID, …}}``,
  the digit gri-ID ↔ R-ID map.
- ``grd/mbm/points`` array — per-measurement ``gri`` / ``xyz`` (metres) / ``tim``
  (seconds) / ``str`` (PMT signal).

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
    """Per used bead, return drift arrays plus the raw ``str`` PMT signal.

    ``xyz_nm`` is (n×3) in nm, re-zeroed to the bead's per-axis **median**
    (``pos_nm`` = that median, the bead's absolute position for a scatter);
    ``tim_s`` is sorted ascending and zeroed to its start. ``pmt_signal`` is
    ``points["str"]`` in the same chronological order, or ``None`` for older
    point arrays that do not carry that field.

    Bead identity degrades gracefully through three sources, because only the
    first is guaranteed to travel with a dataset:

    1. ``used_rids`` — the R-IDs the acquisition marked as used;
    2. every R-ID in ``points_by_gri`` when no "used" list is available;
    3. **the distinct ``gri`` values in the points array itself** when neither
       is available — the array always carries them, and the ``points_by_gri``
       map only adds the human-readable R-ID name on top.  Such beads are named
       by their gri.

    Without (3) a dataset carrying a perfectly good points array but no
    accompanying metadata yielded *no* bead traces at all.
    """
    if points is None or getattr(points, "dtype", None) is None:
        return []
    names = points.dtype.names or ()
    if not {"gri", "xyz", "tim"} <= set(names):
        return []
    name_to_gri = _name_to_gri(points_by_gri)

    gri_arr = np.asarray(points["gri"]).ravel()
    xyz = np.asarray(points["xyz"], dtype=float)
    tim = np.asarray(points["tim"], dtype=float).ravel()
    pmt = np.asarray(points["str"], dtype=float).ravel() if "str" in names else None
    if xyz.ndim != 2 or xyz.shape[1] < 3:
        return []

    rids = list(used_rids) if used_rids else list(name_to_gri.keys())
    if not rids:
        # No name map at all — fall back to the ids in the data.
        for gri in sorted({int(v) for v in gri_arr}):
            name_to_gri.setdefault(str(gri), gri)
        rids = list(name_to_gri.keys())

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
        pos_nm = np.median(xyz_nm, axis=0)                   # absolute bead position
        xyz_nm = xyz_nm - pos_nm                             # re-zero to median
        tim_s = tim[mask]
        order = np.argsort(tim_s)                            # chronological
        tim_s = tim_s[order]
        tim_s = tim_s - tim_s[0]                             # zero to start
        xyz_nm = xyz_nm[order]
        pmt_signal = pmt[mask][order] if pmt is not None else None
        beads.append({"gri": int(gri), "rid": str(rid),
                      "xyz_nm": xyz_nm, "pos_nm": pos_nm, "tim_s": tim_s,
                      "pmt_signal": pmt_signal, "n": n})
    beads.sort(key=lambda b: b["gri"])
    return beads


def single_channel_bead_summary(name: str, beads: list[dict]) -> dict | None:
    """The ``single_channel`` payload ``AlignmentPlotWindow`` draws for one channel.

    ``pos_nm`` is each bead's absolute position and ``drift_nm`` its total
    peak-to-peak excursion per axis over the acquisition — the "how far did this
    fiducial wander" number, which is what the window's table reports when there
    is no alignment to residual against.
    """
    if not beads:
        return None
    return {
        "name": str(name),
        "bead_ids": np.array([b["gri"] for b in beads], dtype=np.uint32),
        "rids": [str(b.get("rid", b["gri"])) for b in beads],
        "pos_nm": np.array([b["pos_nm"] for b in beads], dtype=float),
        "drift_nm": np.array([
            (np.ptp(np.asarray(b["xyz_nm"]), axis=0)
             if np.asarray(b["xyz_nm"]).shape[0] else np.zeros(3))
            for b in beads
        ], dtype=float),
    }


def dataset_bead_drift(ds) -> list[dict]:
    """Bead traces for a **loaded** dataset, from the arrays it carries itself.

    ``ds.mbm`` / ``metadata["mbm_points"]`` travel with a dataset imported from
    an ``.msr`` (or round-tripped through one), so this needs no reader state and
    no re-parse of the source file.  ``mbm_points_by_gri`` / ``mbm_used`` refine
    the naming when present; :func:`extract_bead_drift` falls back to the ids in
    the array when they are not.
    """
    from ...core.overlay import mbm_points_array

    points = mbm_points_array(ds)
    if points is None or not getattr(points, "size", 0):
        return []
    meta = getattr(ds, "metadata", {}) or {}
    return extract_bead_drift(
        points,
        meta.get("mbm_points_by_gri") or {},
        meta.get("mbm_used") or [],
    )


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
