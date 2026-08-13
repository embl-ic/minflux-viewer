"""
minflux_viewer.analysis.attribute_channels
===========================================
Attribute-agnostic **channel model** for separating one dataset into a
multi-channel overlay by the distribution of a chosen MINFLUX attribute. This is
the Phase-2 core behind the redesigned separation tool (DCR is its first
instance); it is deliberately independent of any specific attribute so the same
model drives *by DCR*, *by time window*, *by any attribute* in the future generic
"Convert Dataset to Multi-Channel Overlay" tool.

A **channel** is an independent carrier: a named ``[lo, hi]`` value window with a
LUT/colour. Channels are produced three ways —

* :func:`place_evenly` — equal-width windows across the value range,
* :func:`channels_from_fit` — one window per fitted mixture component, split at
  the Bayes boundaries (:mod:`minflux_viewer.analysis.distribution_fit`),
* hand-edited in the UI (drag the region / edit start-end),

— and a localization (or whole trace) is assigned to the channel whose window
contains its attribute value. This mirrors the time-window model but on an
arbitrary attribute axis.

Also here: :func:`eco_weighted_group_mean` / :func:`pooled_dcr_per_loc`, the
photon-weighted DCR pooling (``Σ dcr·eco / Σ eco`` over the final-scale
iterations) that makes the DCR peaks separate more cleanly — pyMINFLUX's
"Pool DCR values (ECO-weighted)", but weighted over *our* detected final-scale
iterations rather than a single last iteration.

Pure NumPy where possible; the ``ds``-level helpers read the raw store through
:mod:`minflux_viewer.core.loader`.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from ..core.overlay import PURE_COLOR_NAMES
from .distribution_fit import MixtureResult

_DEFAULT_LUTS = list(PURE_COLOR_NAMES)


@dataclass
class Channel:
    """One channel carrier: a named ``[lo, hi]`` attribute window + LUT colour."""
    name: str
    lo: float
    hi: float
    lut: str

    def contains(self, values) -> np.ndarray:
        v = np.asarray(values, dtype=float)
        return (v >= self.lo) & (v <= self.hi)


# ---------------------------------------------------------------------------
# Channel construction
# ---------------------------------------------------------------------------
def _lut_for(index: int, luts) -> str:
    seq = list(luts) if luts else _DEFAULT_LUTS
    return seq[index % len(seq)]


def _channel_name(base_name: str, attribute: str, index: int) -> str:
    tag = attribute if attribute else "ch"
    return f"{base_name} [{tag} {index + 1}]"


def place_evenly(
    lo: float,
    hi: float,
    n: int,
    *,
    base_name: str = "channel",
    attribute: str = "",
    luts=None,
) -> list[Channel]:
    """``n`` equal-width, adjacent channel windows spanning ``[lo, hi]``."""
    n = int(n)
    if n < 1:
        raise ValueError("need at least one channel")
    edges = np.linspace(float(lo), float(hi), n + 1)
    return [
        Channel(name=_channel_name(base_name, attribute, i),
                lo=float(edges[i]), hi=float(edges[i + 1]), lut=_lut_for(i, luts))
        for i in range(n)
    ]


def channels_from_boundaries(
    boundaries,
    lo: float,
    hi: float,
    *,
    base_name: str = "channel",
    attribute: str = "",
    luts=None,
) -> list[Channel]:
    """Adjacent channels split at *boundaries* (internal cut points), spanning
    ``[lo, hi]``. ``k`` boundaries → ``k + 1`` channels."""
    cuts = np.sort(np.asarray(boundaries, dtype=float).ravel())
    edges = np.concatenate([[float(lo)], cuts, [float(hi)]])
    edges = np.clip(edges, float(lo), float(hi))
    edges = np.maximum.accumulate(edges)                 # keep monotone
    return [
        Channel(name=_channel_name(base_name, attribute, i),
                lo=float(edges[i]), hi=float(edges[i + 1]), lut=_lut_for(i, luts))
        for i in range(edges.size - 1)
    ]


def channels_from_fit(
    result: MixtureResult,
    *,
    data_range: tuple[float, float] | None = None,
    base_name: str = "channel",
    attribute: str = "",
    luts=None,
) -> list[Channel]:
    """One channel per fitted mixture component, split at the fit's Bayes
    boundaries. Outer edges span the full *data_range* (so no localization falls
    outside the outermost channels), falling back to the fit's own domain."""
    lo, hi = data_range if data_range is not None else result.domain
    lo = min(float(lo), result.domain[0])
    hi = max(float(hi), result.domain[1])
    return channels_from_boundaries(
        result.boundaries(), lo, hi,
        base_name=base_name, attribute=attribute, luts=luts,
    )


# ---------------------------------------------------------------------------
# Assignment
# ---------------------------------------------------------------------------
def assign_values(values, channels: list[Channel]) -> np.ndarray:
    """Per-value channel label: index of the first channel whose window contains
    the value, else -1 (outside every window / NaN)."""
    v = np.asarray(values, dtype=float).ravel()
    if not channels:
        return np.full(v.size, -1, dtype=int)
    los = np.array([c.lo for c in channels], dtype=float)[None, :]
    his = np.array([c.hi for c in channels], dtype=float)[None, :]
    contains = (v[:, None] >= los) & (v[:, None] <= his)     # (N, K)
    any_c = contains.any(axis=1)
    labels = np.where(any_c, contains.argmax(axis=1), -1).astype(int)
    labels[~np.isfinite(v)] = -1
    return labels


def assign_traces(
    values,
    tid,
    channels: list[Channel],
    *,
    mode: str = "mean",
    min_confidence: float = 0.5,
) -> np.ndarray:
    """Per-localization channel labels where **a whole trace shares one label**.

    * ``"mean"`` / ``"median"`` — collapse the trace's attribute to one value
      (NaN-ignoring), assign that value's containing channel.
    * ``"majority"`` / ``"vote"`` — assign each localization, then the trace takes
      the majority channel; ``min_confidence`` is the required vote-agreement
      fraction (below it, or on a tie, the trace is unassigned).

    Returns an int array aligned to *values* (channel index, or -1). Falls back to
    per-localization :func:`assign_values` if *tid* does not align.
    """
    v = np.asarray(values, dtype=float).ravel()
    tid = np.asarray(tid).ravel()
    if tid.shape[0] != v.shape[0] or v.size == 0 or not channels:
        return assign_values(v, channels)
    uniq, inv = np.unique(tid, return_inverse=True)
    order = np.argsort(inv, kind="stable")
    s_inv = inv[order]
    s_vals = v[order]
    bnd = np.flatnonzero(np.diff(s_inv)) + 1
    starts = np.concatenate([[0], bnd])
    ends = np.concatenate([bnd, [s_inv.size]])

    m = str(mode).lower()
    if "major" in m or "vote" in m:
        per_loc = assign_values(v, channels)             # -1 for out-of-window / NaN
        s_lab = per_loc[order]
        n_ch = len(channels)
        trace_label = np.full(uniq.size, -1, dtype=int)
        for k, (a, b) in enumerate(zip(starts, ends)):
            grp = s_lab[a:b]
            grp = grp[grp >= 0]
            if grp.size == 0:
                continue
            counts = np.bincount(grp, minlength=n_ch)
            top = int(counts.argmax())
            best = int(counts[top])
            # ambiguous tie → unassigned
            if np.count_nonzero(counts == best) > 1:
                continue
            if best / grp.size >= float(min_confidence):
                trace_label[k] = top
        return trace_label[inv]

    fn = np.nanmedian if ("median" in m) else np.nanmean
    trace_val = np.full(uniq.size, np.nan)
    with np.errstate(all="ignore"):
        for k, (a, b) in enumerate(zip(starts, ends)):
            grp = s_vals[a:b]
            grp = grp[np.isfinite(grp)]
            if grp.size:
                trace_val[k] = fn(grp)
    trace_label = assign_values(trace_val, channels)
    return trace_label[inv]


# ---------------------------------------------------------------------------
# Photon (eco) weighted DCR pooling
# ---------------------------------------------------------------------------
def eco_weighted_group_mean(values, weights, group_id, n_groups: int) -> np.ndarray:
    """Per-group weighted mean ``Σ vᵢwᵢ / Σ wᵢ`` (NaN for empty/zero-weight groups).

    Vectorised via :func:`numpy.bincount`. The generic kernel behind the
    photon-weighted DCR pooling and any future eco-weighted attribute.
    """
    v = np.asarray(values, dtype=float).ravel()
    w = np.asarray(weights, dtype=float).ravel()
    g = np.asarray(group_id).ravel().astype(np.int64)
    ok = np.isfinite(v) & np.isfinite(w) & (w > 0.0) & (g >= 0)
    num = np.bincount(g[ok], weights=(v[ok] * w[ok]), minlength=int(n_groups))
    den = np.bincount(g[ok], weights=w[ok], minlength=int(n_groups))
    out = np.full(int(n_groups), np.nan)
    nz = den > 0.0
    out[nz] = num[nz] / den[nz]
    return out


_REVERT_CANON_ATTRS = ["efo", "cfr", "dcr", "eco", "ecc", "efc", "fbg", "tim", "sta"]


def reconstruct_from_channels(datasets) -> dict:
    """Concatenate separation-overlay channel members back into one localization
    set — the fallback for "Revert Overlay to Original Dataset" when the source
    dataset is no longer open.

    Separation uses identity overlay transforms and a single shared source, so
    the channel subsets concatenate directly: display-nm coordinates, original
    ``tid`` (no remap), and every canonical per-loc attribute present in **all**
    members. Derived attributes (``den``, ``siz``, …) are omitted — they are
    recomputed by the dataset builder. Returns ``{}`` when nothing usable.
    """
    from ..core.loader import attr_values_1d

    datasets = list(datasets)
    xs, ys, zs, tids = [], [], [], []
    keep = [a for a in _REVERT_CANON_ATTRS
            if all(attr_values_1d(d, a) is not None for d in datasets)]
    cols: dict[str, list] = {a: [] for a in keep}
    for ds in datasets:
        x = attr_values_1d(ds, "xnm")
        y = attr_values_1d(ds, "ynm")
        if x is None or y is None:
            continue
        x = np.asarray(x, dtype=float).ravel()
        y = np.asarray(y, dtype=float).ravel()
        n = x.size
        z = attr_values_1d(ds, "znm")
        t = attr_values_1d(ds, "tid")
        xs.append(x)
        ys.append(y)
        zs.append(np.zeros(n) if z is None else np.asarray(z, dtype=float).ravel())
        tids.append(np.arange(n) if t is None else np.asarray(t).ravel())
        for a in keep:
            cols[a].append(np.asarray(attr_values_1d(ds, a), dtype=float).ravel())
    if not xs:
        return {}
    attrs = {a: np.concatenate(v) for a, v in cols.items() if len(v) == len(xs)}
    return {
        "x_nm": np.concatenate(xs), "y_nm": np.concatenate(ys),
        "z_nm": np.concatenate(zs), "tid": np.concatenate(tids), "attrs": attrs,
    }


def pooled_dcr_per_loc(ds) -> np.ndarray | None:
    """Photon-weighted DCR per localization, aligned to ``ds.attr`` rows.

    For each localization, ``dcr`` is pooled over its **final-scale (photon-
    bearing) iterations** weighted by ``eco``: ``Σ dcr·eco / Σ eco``
    (pyMINFLUX's ECO-weighted pooling, over
    :func:`~minflux_viewer.core.mfx_sequence.photon_iterations_for_dataset`).
    Returns ``None`` when the raw store, ``eco``, or the last-valid alignment is
    unavailable (caller then uses the plain materialized DCR).
    """
    from ..core.loader import _mfx_raw_len, _raw_loc_id, mfx_get, mfx_row_mask
    from ..core.mfx_sequence import photon_iterations_for_dataset

    raw = getattr(ds, "mfx_raw", None)
    if raw is None or _mfx_raw_len(raw) == 0:
        return None
    n_rows = _mfx_raw_len(raw)
    loc_id = _raw_loc_id(raw)
    if loc_id is None:
        return None
    itr_all = np.asarray(raw.get("itr", np.zeros(n_rows, int))).ravel()
    vld_all = np.asarray(raw.get("vld", np.ones(n_rows, bool)), dtype=bool).ravel()
    dcr_all = mfx_get(ds, "dcr", itr="all", vld_only=False)
    eco_all = mfx_get(ds, "eco", itr="all", vld_only=False)
    if dcr_all is None or eco_all is None:
        return None
    dcr_all = np.asarray(dcr_all, dtype=float).ravel()
    eco_all = np.asarray(eco_all, dtype=float).ravel()
    if dcr_all.shape[0] != n_rows or eco_all.shape[0] != n_rows:
        return None

    photon_iters = photon_iterations_for_dataset(ds).photon_iters
    sel = vld_all & np.isin(itr_all, np.asarray(photon_iters, dtype=int))
    if not sel.any():
        return None
    n_groups = int(loc_id.max()) + 1
    per_loc = eco_weighted_group_mean(dcr_all[sel], eco_all[sel], loc_id[sel], n_groups)

    # Align per-loc values to ds.attr rows (their last-valid loc order).
    last_mask = mfx_row_mask(raw, itr="last", vld_only=True)
    if last_mask is None:
        return None
    order = loc_id[last_mask]
    if order.shape[0] != int(getattr(ds.prop, "num_loc", -1)):
        return None
    return per_loc[order]
