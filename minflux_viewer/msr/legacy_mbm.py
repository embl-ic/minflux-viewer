"""
minflux_viewer.msr.legacy_mbm
=============================
Translate the **legacy** bead (MBM) layout into the model ``points`` structure.

Modern ``.msr`` files store beads as a single flat structured array at
``grd/mbm/points`` (fields ``gri``/``xyz``/``tim``/``str``) plus a
``points_by_gri`` attr mapping the numeric ``gri`` to its R-ID name, and a
``used`` bead list on the ``mbm`` node.

Some early files instead nest one **structured array per bead** under
``grd/mbm/<R-ID>`` (fields ``tim`` (N), ``pos`` (N×3, metres), ``str`` (N),
``tid`` (N)); each R-ID node's attrs carry ``gri`` (int) and ``name`` (the R-ID),
and the ``grd/mbm`` group attrs carry the authoritative ``used`` R-ID list.

:func:`build_legacy_mbm` reconstructs the model ``points`` array + ``points_by_gri``
+ ``used`` from that layout **without modifying the parsed tree** — the synthesized
metadata is stored alongside (see ``state.mbm_meta_map``) so "Show beads drift" and
"Align channel" work just as for modern files.
"""

from __future__ import annotations

import numpy as np

#: Model bead-points dtype (matches the modern ``grd/mbm/points`` fields used by
#: the drift/alignment code: ``gri`` tag, ``xyz`` in metres, ``tim`` in seconds).
POINTS_DTYPE = np.dtype([
    ("gri", np.int64), ("xyz", np.float64, (3,)),
    ("tim", np.float64), ("str", np.float64),
])


def _node_attrs(node) -> dict:
    try:
        return dict(node.attrs.asdict())
    except Exception:
        try:
            return dict(node.attrs)
        except Exception:
            return {}


def _bead_rid_keys(mbm_group) -> list[str]:
    """The per-bead R-ID child arrays of the ``grd/mbm`` group."""
    try:
        return list(mbm_group.array_keys())
    except Exception:
        return []


def build_legacy_mbm(arch, log=print):
    """Reconstruct ``(points, points_by_gri, used)`` from a legacy ``grd/mbm``
    layout, or ``None`` if *arch* doesn't have one.

    *arch* is an opened zarr group (the dataset's in-memory store).
    """
    try:
        if "grd/mbm" not in arch:
            return None
        mbm = arch["grd/mbm"]
    except Exception:
        return None

    rids = _bead_rid_keys(mbm)
    if not rids:
        return None

    parts: list[np.ndarray] = []
    points_by_gri: dict[str, dict] = {}
    for rid in rids:
        try:
            node = arch[f"grd/mbm/{rid}"]
        except Exception:
            continue
        na = _node_attrs(node)
        gri = na.get("gri")
        if gri is None:
            continue
        try:
            gri = int(gri)
        except (TypeError, ValueError):
            continue
        name = str(na.get("name", rid)) or str(rid)
        points_by_gri[str(gri)] = {"name": name, "gri": gri}

        try:
            a = node[:]
        except Exception:
            a = None
        if a is None or getattr(a, "dtype", None) is None or a.dtype.names is None:
            continue
        names = set(a.dtype.names)
        if not ({"pos", "tim"} <= names):
            continue
        n = int(a.shape[0])
        if n == 0:
            continue
        pos = np.asarray(a["pos"], dtype=np.float64)
        # tolerate either N×3 (seen) or 3×N (as some exports store it)
        if pos.ndim == 2 and pos.shape[0] == 3 and pos.shape[1] != 3:
            pos = pos.T
        pos = pos.reshape(n, 3) if pos.size == n * 3 else None
        if pos is None:
            continue
        rec = np.zeros(n, dtype=POINTS_DTYPE)
        rec["gri"] = gri
        rec["xyz"] = pos
        rec["tim"] = np.asarray(a["tim"], dtype=np.float64).ravel()
        if "str" in names:
            rec["str"] = np.asarray(a["str"], dtype=np.float64).ravel()
        parts.append(rec)

    if not points_by_gri:
        return None
    points = np.concatenate(parts) if parts else np.zeros(0, dtype=POINTS_DTYPE)

    used: list[str] = []
    u = _node_attrs(mbm).get("used")
    if isinstance(u, (list, tuple)):
        used = [str(x).strip() for x in u if str(x).strip()]

    try:
        log(f"    [mbm] legacy layout: {len(points_by_gri)} bead(s), "
            f"{points.size} measurement(s), {len(used)} used")
    except Exception:
        pass
    return points, points_by_gri, used
