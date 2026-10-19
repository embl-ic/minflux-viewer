"""Bounded inspection rasters. Never inputs to scientific computations."""

from __future__ import annotations

import numpy as np
from scipy.ndimage import gaussian_filter


def histogram_preview(xy, bounds, shape):
    """Smoothed XY counts on an explicit monitor grid, with exact world bounds."""
    nx, ny = (int(v) for v in shape)
    if not (1 <= nx <= 1536 and 1 <= ny <= 1536):
        raise ValueError("Inspection raster dimensions must be in [1, 1536]")
    x0, x1, y0, y1 = (float(v) for v in bounds)
    if not np.isfinite(bounds).all() or x1 <= x0 or y1 <= y0:
        raise ValueError("Inspection raster requires finite increasing bounds")
    points = np.asarray(xy)
    image, _, _ = np.histogram2d(points[:, 0], points[:, 1], bins=(nx, ny),
                                range=((x0, x1), (y0, y1)))
    return gaussian_filter(image, .5, mode="constant")


def peak_preview(image, max_dim=1024):
    """Max-pool display only, retaining bright features; never used for NMS."""
    image = np.asarray(image)
    if image.size == 0:
        return image
    steps = [max(1, int(np.ceil(n / max_dim))) for n in image.shape]
    result = image
    for axis, step in enumerate(steps):
        if step > 1:
            result = np.maximum.reduceat(result, np.arange(0, result.shape[axis], step), axis=axis)
    return result
