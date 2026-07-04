"""
minflux_viewer.core.particle_set
=================================
On-disk **particle set** format for the Particle Average plugin — a collection of
re-zeroed localization point clouds (one per detected object, e.g. an NPC),
pooled from one or many datasets, saved/loaded independently of the dataset and
ROI files.

Format: a single self-describing **HDF5** file (``.h5``). Localizations from all
particles are pooled into equal-length 1-D column datasets under
``/localizations`` (mirroring the app's ``AttrStore`` dict-of-arrays model), keyed
by a ``particle_id`` column; per-particle metadata lives under ``/particles``.

Two coordinate frames are stored (design rule 5 — raw / view kept separate):

* ``xnm / ynm / znm`` — the **authoritative** coordinates the averaging consumes:
  re-zeroed *display* nm (RIMF + overlay/alignment transform + unit change + ROI
  re-zero all baked in). The loader uses these directly and never re-applies
  anything.
* ``loc_x / loc_y / loc_z`` — the **raw** canonical localization (float64, metres,
  Z *not* RIMF-baked), untouched; **provenance only**, never auto-corrected
  (avoids the baked-plus-recipe double-correction footgun).

Per-localization attributes (``efo``, ``cfr``, ``dcr``, ``eco``, …) are an open
set of extra columns; particles from different datasets are unioned (missing
columns filled with NaN). Pure / Qt-free / unit-testable.
"""

from __future__ import annotations

import datetime as _dt
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

PARTICLE_SET_MARKER = "minflux_particle_set"
PARTICLE_SET_VERSION = 1

#: column names that are stored explicitly, not as generic per-loc attributes.
_RESERVED_COLUMNS = {
    "particle_id", "xnm", "ynm", "znm",
    "loc_x", "loc_y", "loc_z", "tid", "unit_id",
}


@dataclass
class ParticleData:
    """One particle — an independent, re-zeroed localization point cloud.

    ``points`` is the authoritative ``(M, 3)`` re-zeroed display-nm cloud the
    averaging operates on; ``loc`` is the optional raw ``(M, 3)`` metres cloud
    (provenance). ``attrs`` is an open dict of per-loc arrays (length ``M``).
    """

    points: np.ndarray                      # (M, 3) re-zeroed display nm
    tid: np.ndarray | None = None           # (M,) trace ids
    unit_id: np.ndarray | None = None       # (M,) sub-unit ids (0 = unassigned)
    loc: np.ndarray | None = None           # (M, 3) raw metres, or None
    attrs: dict[str, np.ndarray] = field(default_factory=dict)
    source: str = ""                        # source dataset name
    roi_name: str = ""                      # source ROI name
    meta: dict = field(default_factory=dict)  # rezero_offset, rimf, z_scale, …

    @property
    def n_locs(self) -> int:
        pts = np.asarray(self.points)
        return int(pts.shape[0]) if pts.ndim == 2 else 0


def _as_points3(points) -> np.ndarray:
    pts = np.asarray(points, dtype=float)
    if pts.ndim != 2 or pts.shape[0] == 0:
        return np.empty((0, 3), dtype=float)
    if pts.shape[1] == 2:
        pts = np.column_stack([pts, np.zeros(pts.shape[0])])
    return pts[:, :3]


def save_particle_set(
    path: str | Path,
    particles,
    *,
    content: str = "all",
    include_raw_loc: bool = True,
    coord_dtype=np.float32,
    raw_dtype=np.float64,
) -> Path:
    """Write *particles* (an iterable of :class:`ParticleData`) to an HDF5 file.

    ``content``: ``"all"`` writes raw loc (if available + ``include_raw_loc``)
    plus every per-loc attribute; ``"loc_tid"`` writes only the coordinates +
    ``tid`` + ``unit_id`` (smallest file). ``xnm/ynm/znm`` are always written.
    """
    import h5py

    if content not in ("all", "loc_tid"):
        raise ValueError(f"content must be 'all' or 'loc_tid', got {content!r}")
    particles = [p for p in particles]
    path = Path(path)
    K = len(particles)
    sizes = np.array([p.n_locs for p in particles], dtype=np.int64)
    N = int(sizes.sum())
    pid = np.repeat(np.arange(K, dtype=np.int32), sizes)

    def _stack(getter, width, fill=np.nan, dtype=float):
        """Pool a per-particle (M, width) field into one (N, width) array."""
        out = np.full((N, width), fill, dtype=dtype)
        off = 0
        for p in particles:
            m = p.n_locs
            v = getter(p)
            if v is not None:
                v = np.asarray(v, dtype=dtype)
                if v.ndim == 1:
                    v = v.reshape(-1, 1)
                out[off:off + m, : v.shape[1]] = v[:m, :width]
            off += m
        return out

    xyz = _stack(lambda p: _as_points3(p.points), 3, dtype=coord_dtype)

    want_raw = content == "all" and include_raw_loc and any(p.loc is not None for p in particles)
    raw = _stack(lambda p: p.loc, 3, dtype=raw_dtype) if want_raw else None

    has_tid = any(p.tid is not None for p in particles)
    tid = _stack(lambda p: p.tid, 1, dtype=np.float64).ravel() if has_tid else None
    has_unit = any(p.unit_id is not None for p in particles)
    unit = _stack(lambda p: p.unit_id, 1, fill=0, dtype=np.int32).ravel() if has_unit else None

    attr_keys: list[str] = []
    if content == "all":
        attr_keys = sorted({k for p in particles for k in (p.attrs or {})} - _RESERVED_COLUMNS)

    str_dt = h5py.string_dtype(encoding="utf-8")
    comp = dict(compression="gzip", compression_opts=4, shuffle=True)

    def _meta_col(key, default=np.nan):
        return np.array([float(p.meta.get(key, default)) for p in particles], dtype=np.float64) \
            if K else np.empty(0)

    with h5py.File(path, "w") as f:
        f.attrs["format"] = PARTICLE_SET_MARKER
        f.attrs["version"] = PARTICLE_SET_VERSION
        f.attrs["units"] = "nm"
        f.attrs["coordinate_frame"] = "rezeroed_display"
        f.attrs["content"] = content
        f.attrs["n_particles"] = K
        f.attrs["created"] = _dt.datetime.now().isoformat(timespec="seconds")

        loc_g = f.create_group("localizations")
        loc_g.create_dataset("particle_id", data=pid, **comp)
        loc_g.create_dataset("xnm", data=xyz[:, 0], **comp)
        loc_g.create_dataset("ynm", data=xyz[:, 1], **comp)
        loc_g.create_dataset("znm", data=xyz[:, 2], **comp)
        if raw is not None:
            loc_g.create_dataset("loc_x", data=raw[:, 0], **comp)
            loc_g.create_dataset("loc_y", data=raw[:, 1], **comp)
            loc_g.create_dataset("loc_z", data=raw[:, 2], **comp)
        if tid is not None:
            loc_g.create_dataset("tid", data=tid, **comp)
        if unit is not None:
            loc_g.create_dataset("unit_id", data=unit, **comp)
        for k in attr_keys:
            col = _stack(lambda p, _k=k: (p.attrs or {}).get(_k), 1, dtype=np.float64).ravel()
            loc_g.create_dataset(k, data=col, **comp)

        part_g = f.create_group("particles")
        part_g.create_dataset("particle_id", data=np.arange(K, dtype=np.int32))
        part_g.create_dataset("n_locs", data=sizes.astype(np.int64))
        part_g.create_dataset("source", data=np.array([p.source for p in particles], dtype=object), dtype=str_dt)
        part_g.create_dataset("roi_name", data=np.array([p.roi_name for p in particles], dtype=object), dtype=str_dt)
        part_g.create_dataset("source_version",
                              data=np.array([str(p.meta.get("source_version", "")) for p in particles], dtype=object),
                              dtype=str_dt)
        part_g.create_dataset("rimf", data=_meta_col("rimf", 1.0))
        part_g.create_dataset("z_scale", data=_meta_col("z_scale", 1.0))
        rezero = np.full((K, 3), np.nan, dtype=np.float64)
        tform = np.tile(np.eye(4, dtype=np.float64), (max(K, 1), 1, 1))[:K]
        for i, p in enumerate(particles):
            ro = p.meta.get("rezero_offset")
            if ro is not None:
                rezero[i, :len(ro)] = np.asarray(ro, dtype=float)[:3]
            tf = p.meta.get("transform_4x4")
            if tf is not None:
                tform[i] = np.asarray(tf, dtype=float).reshape(4, 4)
        part_g.create_dataset("rezero_offset", data=rezero)
        part_g.create_dataset("transform_4x4", data=tform)
    return path


def load_particle_set(path: str | Path) -> list[ParticleData]:
    """Read an HDF5 particle set written by :func:`save_particle_set`."""
    import h5py

    path = Path(path)
    with h5py.File(path, "r") as f:
        if f.attrs.get("format") != PARTICLE_SET_MARKER:
            raise ValueError(f"'{path.name}' is not a {PARTICLE_SET_MARKER} file.")
        K = int(f.attrs.get("n_particles", 0))
        loc = f["localizations"]
        pid = np.asarray(loc["particle_id"][:])
        xyz = np.column_stack([loc["xnm"][:], loc["ynm"][:], loc["znm"][:]]).astype(float)
        has_raw = all(c in loc for c in ("loc_x", "loc_y", "loc_z"))
        raw = np.column_stack([loc["loc_x"][:], loc["loc_y"][:], loc["loc_z"][:]]).astype(float) \
            if has_raw else None
        tid = np.asarray(loc["tid"][:]) if "tid" in loc else None
        unit = np.asarray(loc["unit_id"][:]) if "unit_id" in loc else None
        attr_keys = [k for k in loc.keys() if k not in _RESERVED_COLUMNS]
        attr_all = {k: np.asarray(loc[k][:]) for k in attr_keys}

        part = f["particles"]
        sources = _str_list(part["source"][:]) if "source" in part else [""] * K
        roi_names = _str_list(part["roi_name"][:]) if "roi_name" in part else [""] * K
        src_ver = _str_list(part["source_version"][:]) if "source_version" in part else [""] * K
        rimf = np.asarray(part["rimf"][:]) if "rimf" in part else np.ones(K)
        z_scale = np.asarray(part["z_scale"][:]) if "z_scale" in part else np.ones(K)
        rezero = np.asarray(part["rezero_offset"][:]) if "rezero_offset" in part else None
        tform = np.asarray(part["transform_4x4"][:]) if "transform_4x4" in part else None

    out: list[ParticleData] = []
    for i in range(K):
        m = pid == i
        p_tid = None
        if tid is not None:
            seg = tid[m]
            p_tid = None if seg.size and np.all(np.isnan(seg)) else seg
        meta = {
            "rimf": float(rimf[i]) if i < len(rimf) else 1.0,
            "z_scale": float(z_scale[i]) if i < len(z_scale) else 1.0,
            "source_version": src_ver[i] if i < len(src_ver) else "",
        }
        if rezero is not None and i < len(rezero):
            meta["rezero_offset"] = rezero[i].tolist()
        if tform is not None and i < len(tform):
            meta["transform_4x4"] = tform[i]
        out.append(ParticleData(
            points=xyz[m],
            tid=p_tid,
            unit_id=(unit[m] if unit is not None else None),
            loc=(raw[m] if raw is not None else None),
            attrs={k: v[m] for k, v in attr_all.items()},
            source=sources[i] if i < len(sources) else "",
            roi_name=roi_names[i] if i < len(roi_names) else "",
            meta=meta,
        ))
    return out


def _str_list(arr) -> list[str]:
    return [s.decode("utf-8") if isinstance(s, bytes) else str(s) for s in arr]


def is_particle_set_file(path: str | Path) -> bool:
    """True if *path* is an HDF5 file tagged as a minflux particle set."""
    try:
        import h5py
        with h5py.File(Path(path), "r") as f:
            return f.attrs.get("format") == PARTICLE_SET_MARKER
    except Exception:
        return False
