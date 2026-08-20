"""Shared geometry helpers for OpenGL axis ticks and three-plane grids."""

from __future__ import annotations

import numpy as np


def nice_step(span: float, *, target: int = 8) -> float:
    raw = float(span) / max(int(target), 1)
    if not np.isfinite(raw) or raw <= 0:
        return 1.0
    magnitude = 10.0 ** np.floor(np.log10(raw))
    normalized = raw / magnitude
    step = (
        1 if normalized < 1.5
        else 2 if normalized < 3
        else 5 if normalized < 7
        else 10
    ) * magnitude
    return float(max(step, 1e-12))


def tick_values(lo: float, hi: float, *, max_ticks: int = 5) -> list[float]:
    if not (np.isfinite(lo) and np.isfinite(hi)) or hi <= lo:
        return []
    step = nice_step(hi - lo, target=max_ticks)
    start = np.ceil(lo / step) * step
    values: list[float] = []
    value = start
    while value <= hi + step * 1e-9 and len(values) < max_ticks + 2:
        values.append(float(value))
        value += step
    return values[:max_ticks]


def _grid_values(lo: float, hi: float, *, target: int) -> np.ndarray:
    if not (np.isfinite(lo) and np.isfinite(hi)) or hi <= lo:
        return np.asarray([lo, hi], dtype=np.float64)
    step = nice_step(hi - lo, target=target)
    first = np.ceil(lo / step) * step
    interior = np.arange(first, hi + step * 1e-9, step, dtype=np.float64)
    return np.unique(np.concatenate(([lo], interior, [hi])))


def three_plane_grid_positions(
    mins: np.ndarray,
    maxs: np.ndarray,
    *,
    target: int = 8,
) -> np.ndarray:
    """Return independent line segments for XY, XZ, and YZ back planes."""
    mins = np.asarray(mins, dtype=np.float64).ravel()[:3]
    maxs = np.asarray(maxs, dtype=np.float64).ravel()[:3]
    if mins.size != 3 or maxs.size != 3 or np.any(~np.isfinite([*mins, *maxs])):
        return np.empty((0, 3), dtype=np.float32)
    x0, y0, z0 = mins
    x1, y1, z1 = maxs
    xs = _grid_values(x0, x1, target=target)
    ys = _grid_values(y0, y1, target=target)
    zs = _grid_values(z0, z1, target=target)
    segments: list[list[float]] = []

    # XY floor at z=min.
    for x in xs:
        segments.extend(([x, y0, z0], [x, y1, z0]))
    for y in ys:
        segments.extend(([x0, y, z0], [x1, y, z0]))
    # XZ back plane at y=min.
    for x in xs:
        segments.extend(([x, y0, z0], [x, y0, z1]))
    for z in zs:
        segments.extend(([x0, y0, z], [x1, y0, z]))
    # YZ side plane at x=min.
    for y in ys:
        segments.extend(([x0, y, z0], [x0, y, z1]))
    for z in zs:
        segments.extend(([x0, y0, z], [x0, y1, z]))
    return np.asarray(segments, dtype=np.float32)
