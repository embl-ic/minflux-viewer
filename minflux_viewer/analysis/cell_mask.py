"""Shared numerical geometry used by live analysis workflows."""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np


def _otsu_threshold(img: np.ndarray) -> float:
    """Otsu threshold of a 2-D image (MATLAB ``imbinarize`` default)."""
    vals = np.asarray(img, dtype=np.float64).ravel()
    vals = vals[np.isfinite(vals)]
    if vals.size == 0 or vals.max() <= vals.min():
        return float(vals.min()) if vals.size else 0.0
    hist, edges = np.histogram(vals, bins=256)
    centers = 0.5 * (edges[:-1] + edges[1:])
    total = float(hist.sum())
    wb = np.cumsum(hist).astype(np.float64)
    wf = total - wb
    csum = np.cumsum(hist * centers)
    mb = csum / np.maximum(wb, 1.0)
    mf = (csum[-1] - csum) / np.maximum(wf, 1.0)
    between = wb * wf * (mb - mf) ** 2
    return float(centers[int(np.argmax(between))])

def _disk_struct(radius: int) -> np.ndarray:
    r = max(int(radius), 1)
    y, x = np.ogrid[-r:r + 1, -r:r + 1]
    return (x * x + y * y) <= r * r

@dataclass
class CellMaskResult:
    """Per-E.coli delineation plus the inward-shrunk interior.

    ``border_loc`` is the per-localization exclusion flag used by the 2-D
    pipeline; everything else is diagnostic/reusable — ``edge_distance_nm``
    (exact distance from each localization to its cell boundary) and
    ``cell_id`` make the delineation useful beyond border removal.
    """

    border_loc: np.ndarray          # bool per loc; True = excluded margin
    cell_id: np.ndarray             # int per loc; 0 = outside every cell
    edge_distance_nm: np.ndarray    # float per loc; 0 outside every cell
    mask: np.ndarray                # bool image, the cell mask
    interior: np.ndarray            # bool image, the retained interior
    labels: np.ndarray              # int image, per-cell labels (0 = background)
    x_edges: np.ndarray
    y_edges: np.ndarray
    pixel_size_nm: float
    cell_areas_nm2: np.ndarray
    cell_half_width_nm: np.ndarray  # max distance-to-edge per cell
    stats: dict

def compute_cell_mask(
    loc_nm_xy: np.ndarray,
    *,
    border_size_nm: float = 200.0,
    border_mode: str = "absolute",
    border_fraction: float = 0.35,
    pixel_size_nm: float = 20.0,
    smooth_nm: float = 60.0,
    close_nm: float = 120.0,
    open_nm: float = 0.0,
    min_cell_area_nm2: float = 50_000.0,
) -> CellMaskResult:
    """Delineate each E.coli from the localization density and shrink inward.

    Replaces the original ``get_border_loc`` port. The old pipeline binarized a
    lightly smoothed density and then applied a 25 px (500 nm) **median filter**
    to despeckle; on punctate MINFLUX data that filter deletes exactly the thin
    parts of a cell it is meant to clean, so the mask ended up far smaller than
    the cell and a subsequent 200 nm erosion annihilated it. Here instead:

    1. render an XY density histogram and smooth it with a radius given in
       **nanometres** (``smooth_nm``), so the bridging scale is physical rather
       than tied to the pixel size;
    2. Otsu-binarize;
    3. **morphologically close** by ``close_nm`` — dilation followed by erosion
       bridges the gaps between puncta *without* shrinking the object, which a
       median filter cannot do — then fill enclosed holes;
    4. optionally open by ``open_nm`` to drop spurs, and drop connected
       components below ``min_cell_area_nm2`` (despeckling by area, not by
       erosion);
    5. shrink inward using the exact Euclidean **distance transform** rather
       than a discretized disk erosion, which also yields each localization's
       distance to its cell boundary.

    ``border_mode="absolute"`` keeps localizations at least ``border_size_nm``
    from the boundary. ``border_mode="relative"`` keeps those at least
    ``border_fraction`` × that cell's own half-width from the boundary, which
    adapts to cells of different width and bounds the membrane tilt that the
    shrink is meant to exclude (a point kept at relative depth *f* of a
    cylinder of radius *R* sits within ``arcsin(1 - f)`` of face-on).
    """
    from scipy.ndimage import (
        binary_closing,
        binary_fill_holes,
        binary_opening,
        distance_transform_edt,
        gaussian_filter,
        label,
    )

    xy = np.asarray(loc_nm_xy, dtype=np.float64)
    n_loc = int(xy.shape[0]) if xy.ndim == 2 else 0
    ps = float(pixel_size_nm)
    if ps <= 0:
        raise ValueError("pixel_size_nm must be positive")
    mode = str(border_mode).lower()
    if mode not in ("absolute", "relative"):
        raise ValueError("border_mode must be 'absolute' or 'relative'")

    def _empty(reason: str) -> CellMaskResult:
        return CellMaskResult(
            border_loc=np.zeros(n_loc, dtype=bool),
            cell_id=np.zeros(n_loc, dtype=np.int64),
            edge_distance_nm=np.zeros(n_loc, dtype=float),
            mask=np.zeros((0, 0), dtype=bool),
            interior=np.zeros((0, 0), dtype=bool),
            labels=np.zeros((0, 0), dtype=np.int64),
            x_edges=np.empty(0), y_edges=np.empty(0), pixel_size_nm=ps,
            cell_areas_nm2=np.empty(0), cell_half_width_nm=np.empty(0),
            stats={"n_cells": 0, "reason": reason, "retained_fraction": 1.0,
                   "in_mask_fraction": 1.0},
        )

    if xy.ndim != 2 or xy.shape[1] < 2 or n_loc == 0:
        return _empty("no localizations")

    x, y = xy[:, 0], xy[:, 1]
    pad = max(10.0 * ps, 2.0 * float(close_nm), 2.0 * float(smooth_nm))
    x_edges = np.arange(x.min() - pad, x.max() + pad + ps, ps)
    y_edges = np.arange(y.min() - pad, y.max() + pad + ps, ps)
    nx, ny = x_edges.size - 1, y_edges.size - 1
    if nx < 1 or ny < 1:
        return _empty("field smaller than one pixel")

    counts = np.histogram2d(x, y, bins=[x_edges, y_edges])[0]
    sigma_px = max(float(smooth_nm) / ps, 0.0)
    img = gaussian_filter(counts, sigma_px) if sigma_px > 0 else counts
    mask = img > _otsu_threshold(img)

    close_px = int(round(float(close_nm) / ps))
    if close_px >= 1:
        mask = binary_closing(mask, structure=_disk_struct(close_px))
    mask = binary_fill_holes(mask)
    open_px = int(round(float(open_nm) / ps))
    if open_px >= 1:
        mask = binary_opening(mask, structure=_disk_struct(open_px))

    labels, n_lab = label(mask)
    px_area = ps * ps
    if n_lab:
        comp_px = np.bincount(labels.ravel(), minlength=n_lab + 1)
        too_small = np.flatnonzero(comp_px * px_area < float(min_cell_area_nm2))
        too_small = too_small[too_small > 0]
        if too_small.size:
            drop = np.isin(labels, too_small)
            mask = mask & ~drop
            labels, n_lab = label(mask)

    xb = np.clip(np.searchsorted(x_edges, x, side="right") - 1, 0, nx - 1)
    yb = np.clip(np.searchsorted(y_edges, y, side="right") - 1, 0, ny - 1)

    if not mask.any():
        res = _empty("no cell mask found")
        res.stats["in_mask_fraction"] = 0.0
        return res

    # Exact Euclidean distance to the cell boundary, in nm.
    edt_nm = distance_transform_edt(mask) * ps
    cell_ids = np.arange(1, n_lab + 1)
    cell_px = np.bincount(labels.ravel(), minlength=n_lab + 1)[1:]
    cell_area = cell_px * px_area
    half_width = np.zeros(n_lab, dtype=float)
    for k in cell_ids:
        sel = labels == k
        half_width[k - 1] = float(edt_nm[sel].max()) if sel.any() else 0.0

    if mode == "absolute":
        threshold_img = np.full(mask.shape, float(border_size_nm), dtype=float)
    else:
        f = float(border_fraction)
        threshold_img = np.zeros(mask.shape, dtype=float)
        for k in cell_ids:
            threshold_img[labels == k] = f * half_width[k - 1]
    interior = mask & (edt_nm >= threshold_img)

    border_loc = ~interior[xb, yb]
    in_mask = mask[xb, yb]
    stats = {
        "n_cells": int(n_lab),
        "mask_area_nm2": float(mask.sum() * px_area),
        "interior_area_nm2": float(interior.sum() * px_area),
        "in_mask_fraction": float(in_mask.mean()),
        "retained_fraction": float((~border_loc).mean()),
        "max_half_width_nm": float(half_width.max()) if n_lab else 0.0,
        "median_half_width_nm": float(np.median(half_width)) if n_lab else 0.0,
        "border_mode": mode,
        "border_size_nm": float(border_size_nm),
        "border_fraction": float(border_fraction),
        "smooth_nm": float(smooth_nm),
        "close_nm": float(close_nm),
        "pixel_size_nm": ps,
    }
    if mode == "relative":
        f = min(max(float(border_fraction), 0.0), 1.0)
        stats["implied_max_tilt_deg"] = float(np.degrees(np.arcsin(min(1.0 - f, 1.0))))
    return CellMaskResult(
        border_loc=border_loc,
        cell_id=np.where(in_mask, labels[xb, yb], 0).astype(np.int64),
        edge_distance_nm=np.where(in_mask, edt_nm[xb, yb], 0.0),
        mask=mask, interior=interior, labels=labels,
        x_edges=x_edges, y_edges=y_edges, pixel_size_nm=ps,
        cell_areas_nm2=cell_area, cell_half_width_nm=half_width, stats=stats,
    )
