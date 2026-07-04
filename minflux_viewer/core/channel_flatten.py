"""
minflux_viewer.core.channel_flatten
====================================
Flatten a multi-dataset render **overlay** into a single, non-overlay dataset.

Every channel's localizations are concatenated **in place** — each channel's
overlay transform is baked into display nm via :func:`core.roi_crop.display_coords`,
so the channels stay aligned exactly as shown. Trace ids are **remapped** so the
same ``tid`` from different channels stays a distinct trace, and per-localization
attributes are combined only when they can be combined safely; anything that would
**conflict or go stale** is dropped with a logged reason:

* time / per-channel-timing attributes (``tim``, ``dt``, ``spd``, …) overlap across
  channels → dropped;
* grouping-dependent trace attributes (``siz``, ``dur``, ``len``, …) and
  density/precision (``den``, ``sigma_per_trace_nm``) are meaningless once channels
  merge and are **recomputed** on the combined cloud by the post-load chain →
  dropped;
* an attribute missing from (or invalid in) any channel → dropped.

The result feeds :func:`core.dataset.build_localization_dataset`, so the flattened
dataset renders / filters / runs convolution + particle analysis like any other.
Iteration-count differences between channels are a non-issue: each channel
contributes its **last-valid materialized** per-localization values.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from .loader import attr_values_1d
from .roi_crop import display_coords

#: attributes that conflict or go stale when channels concatenate → dropped
#: (with the logged reason). Density/precision are recomputed post-load.
DROP_REASONS: dict[str, str] = {
    "tim": "per-channel timestamps overlap and would conflict",
    "tim_trace": "trace timestamp is per-channel",
    "dt": "time delta depends on per-channel timing",
    "spd": "speed depends on per-channel timing",
    "den": "local density is recomputed on the combined cloud",
    "siz": "trace size depends on per-channel trace grouping",
    "dur": "trace duration depends on per-channel timing",
    "len": "trace length depends on per-channel trace grouping",
    "idx": "synthetic per-channel row index",
    "sigma_per_trace_nm": "per-trace precision recomputed on the combined cloud",
    "loc_precision_xy": "per-channel precision recomputed on the combined cloud",
}

#: structural attrs handled explicitly (coords baked, tid remapped) → not generic.
_STRUCTURAL = {"loc", "loc_x", "loc_y", "loc_z", "loc_nm",
               "tid", "vld", "itr", "ftr", "filter_mask"}


@dataclass
class FlattenReport:
    n_channels: int = 0
    n_total: int = 0                 # localizations before per-channel filtering
    n_kept: int = 0                  # after each channel's filter mask
    n_traces: int = 0
    kept_attrs: list[str] = field(default_factory=list)
    dropped: list[tuple[str, str]] = field(default_factory=list)   # (attr, reason)
    notes: list[str] = field(default_factory=list)


@dataclass
class FlattenResult:
    coords_nm: np.ndarray            # (N, 3) display nm (overlay transform baked)
    tid: np.ndarray                  # (N,) globally-unique remapped trace ids
    attrs: dict[str, np.ndarray]     # combined per-loc attributes
    report: FlattenReport


def _channel_keep(ds) -> tuple[np.ndarray, np.ndarray]:
    """(coords (n,3) display nm, keep mask) — filter mask + finite XYZ."""
    coords = display_coords(ds)
    n = int(coords.shape[0]) if coords.ndim == 2 else 0
    if n == 0:
        return np.empty((0, 3)), np.zeros(0, bool)
    mask = np.asarray(getattr(ds, "filter_mask", np.ones(n, bool)), dtype=bool)
    if mask.shape[0] != n:
        mask = np.ones(n, bool)
    finite = np.all(np.isfinite(coords[:, :3]), axis=1)
    return coords, (mask & finite)


def _channel_attr(ds, name: str, keep: np.ndarray) -> np.ndarray | None:
    """Length-``keep.sum()`` numeric per-loc column for *name*, or None if it is
    unavailable / non-scalar / misaligned in this channel."""
    v = attr_values_1d(ds, name)
    if v is None:
        v = (getattr(ds, "derived", {}) or {}).get(name)
    if v is None:
        return None
    arr = np.asarray(v)
    if arr.ndim != 1 or arr.shape[0] != keep.shape[0] or not np.issubdtype(arr.dtype, np.number):
        return None
    return arr[keep]


def _remap_tids(datasets, keeps) -> tuple[np.ndarray, int]:
    """Concatenated, globally-unique trace ids: each channel's tids are dense-ranked
    (0..k−1, preserving its trace grouping) then offset past the previous channel."""
    parts: list[np.ndarray] = []
    offset = 0
    for ds, keep in zip(datasets, keeps):
        n = int(keep.sum())
        t = attr_values_1d(ds, "tid")
        if t is None:
            local = np.arange(n)
        else:
            t = np.asarray(t).ravel()
            local = (np.arange(n) if t.shape[0] != keep.shape[0]
                     else np.unique(t[keep], return_inverse=True)[1])
        parts.append(local.astype(np.int64) + offset)
        offset += (int(local.max()) + 1) if local.size else 0
    tid = np.concatenate(parts) if parts else np.empty(0, np.int64)
    return tid, offset


def flatten_overlay(datasets) -> FlattenResult:
    """Flatten *datasets* (an overlay's channels, in order) into one point cloud."""
    report = FlattenReport(n_channels=len(datasets))

    coords_list: list[np.ndarray] = []
    keeps: list[np.ndarray] = []
    itr_counts: set[int] = set()
    for ds in datasets:
        coords, keep = _channel_keep(ds)
        coords_list.append(coords)
        keeps.append(keep)
        report.n_total += int(coords.shape[0])
        try:
            itr_counts.add(int(ds.prop.num_itr))
        except Exception:
            pass
    if len(itr_counts) > 1:
        report.notes.append(
            f"channels have differing iteration counts ({sorted(itr_counts)}); "
            "each contributes its last-valid localization")

    kept = [c[k] for c, k in zip(coords_list, keeps)]
    coords_all = np.vstack(kept) if kept and any(a.size for a in kept) else np.empty((0, 3))
    report.n_kept = int(coords_all.shape[0])
    if report.n_kept < report.n_total:
        report.notes.append(
            f"applied per-channel filters: {report.n_kept:,} of {report.n_total:,} "
            "localizations kept")

    tid_all, _ = _remap_tids(datasets, keeps)
    report.n_traces = int(np.unique(tid_all).size) if tid_all.size else 0
    report.notes.append(f"trace ids remapped to keep {report.n_traces:,} traces distinct")

    # candidate generic attributes = union across channels, minus structural.
    candidate: set[str] = set()
    for ds in datasets:
        for src in (getattr(ds, "attr", {}) or {}, getattr(ds, "derived", {}) or {}):
            try:
                candidate.update(src.keys())
            except Exception:
                pass
    candidate -= _STRUCTURAL

    attrs: dict[str, np.ndarray] = {}
    for name in sorted(candidate):
        if name in DROP_REASONS:
            report.dropped.append((name, DROP_REASONS[name]))
            continue
        cols = [_channel_attr(ds, name, keep) for ds, keep in zip(datasets, keeps)]
        missing = sum(1 for c in cols if c is None)
        if missing:
            report.dropped.append(
                (name, f"missing/invalid in {missing} of {len(cols)} channel(s)"))
            continue
        try:
            attrs[name] = np.concatenate(cols)
            report.kept_attrs.append(name)
        except Exception as exc:                                  # pragma: no cover
            report.dropped.append((name, f"could not concatenate ({exc})"))

    return FlattenResult(coords_nm=coords_all, tid=tid_all, attrs=attrs, report=report)
