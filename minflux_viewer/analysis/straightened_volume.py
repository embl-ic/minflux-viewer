"""
Centerline-based straightening for 3-D localization point clouds.

This is the geometry core behind the experimental
``Analyze > Segmentation > Straightened Volume along Skeleton`` tool.  It
converts a 3-D skeleton into a smooth local frame, maps localizations into
``(arc length, u, v)`` coordinates around that skeleton, and bins them into a
straightened count volume.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

import numpy as np

FrameMode = Literal["minimum_twist", "global_reference"]
ProjectionMode = Literal["sum", "max"]


@dataclass(frozen=True)
class StraightenedVolumeResult:
    centerline_nm: np.ndarray
    arc_nm: np.ndarray
    tangent: np.ndarray
    normal: np.ndarray
    binormal: np.ndarray
    volume: np.ndarray
    s_edges_nm: np.ndarray
    u_edges_nm: np.ndarray
    v_edges_nm: np.ndarray
    projection_su: np.ndarray
    projection_sv: np.ndarray
    projection_uv: np.ndarray
    n_input: int
    n_used: int


def clean_skeleton_points(points_nm: np.ndarray) -> np.ndarray:
    """Return finite 3-D skeleton vertices with consecutive duplicates removed."""
    pts = np.asarray(points_nm, dtype=float)
    if pts.ndim != 2 or pts.shape[1] < 3:
        raise ValueError("Skeleton requires at least two 3-D points.")
    pts = pts[:, :3]
    pts = pts[np.all(np.isfinite(pts), axis=1)]
    if pts.shape[0] < 2:
        raise ValueError("Skeleton requires at least two finite 3-D points.")
    seg = np.linalg.norm(np.diff(pts, axis=0), axis=1)
    keep = np.concatenate([[True], seg > 1e-9])
    pts = pts[keep]
    if pts.shape[0] < 2:
        raise ValueError("Skeleton length is zero.")
    return pts


def resample_skeleton(points_nm: np.ndarray, step_nm: float) -> tuple[np.ndarray, np.ndarray]:
    """Linearly resample a 3-D skeleton at approximately uniform arc-length steps."""
    pts = clean_skeleton_points(points_nm)
    step = max(float(step_nm), 1e-6)
    seg_len = np.linalg.norm(np.diff(pts, axis=0), axis=1)
    arc_src = np.concatenate([[0.0], np.cumsum(seg_len)])
    total = float(arc_src[-1])
    if total <= 0:
        raise ValueError("Skeleton length is zero.")
    arc = np.arange(0.0, total, step, dtype=float)
    if arc.size == 0 or not np.isclose(arc[-1], total):
        arc = np.concatenate([arc, [total]])
    out = np.column_stack([np.interp(arc, arc_src, pts[:, ax]) for ax in range(3)])
    return out, arc


def _normalize(v: np.ndarray, *, fallback: np.ndarray | None = None) -> np.ndarray:
    v = np.asarray(v, dtype=float)
    n = float(np.linalg.norm(v))
    if n > 1e-12:
        return v / n
    if fallback is not None:
        return _normalize(fallback)
    return np.array([1.0, 0.0, 0.0], dtype=float)


def _fallback_normal(tangent: np.ndarray) -> np.ndarray:
    axes = np.eye(3, dtype=float)
    axis = axes[int(np.argmin(np.abs(axes @ tangent)))]
    n = axis - np.dot(axis, tangent) * tangent
    return _normalize(n)


def _project_reference(reference: np.ndarray, tangent: np.ndarray,
                       previous: np.ndarray | None = None) -> np.ndarray:
    ref = _normalize(reference, fallback=np.array([0.0, 0.0, 1.0]))
    n = ref - np.dot(ref, tangent) * tangent
    if np.linalg.norm(n) <= 1e-12:
        n = _fallback_normal(tangent)
    else:
        n = _normalize(n)
    if previous is not None and np.dot(n, previous) < 0:
        n = -n
    return n


def _rotate_between(vec: np.ndarray, src: np.ndarray, dst: np.ndarray) -> np.ndarray:
    """Apply the shortest rotation that maps unit vector *src* to *dst*."""
    src = _normalize(src)
    dst = _normalize(dst)
    cross = np.cross(src, dst)
    s = float(np.linalg.norm(cross))
    c = float(np.clip(np.dot(src, dst), -1.0, 1.0))
    if s <= 1e-12:
        if c > 0:
            return np.asarray(vec, dtype=float)
        # 180-degree turn: choose any stable perpendicular axis.
        axis = _fallback_normal(src)
        return 2.0 * np.dot(axis, vec) * axis - vec
    vx = np.array([
        [0.0, -cross[2], cross[1]],
        [cross[2], 0.0, -cross[0]],
        [-cross[1], cross[0], 0.0],
    ])
    rot = np.eye(3) + vx + (vx @ vx) * ((1.0 - c) / (s * s))
    return rot @ vec


def build_centerline_frames(
    centerline_nm: np.ndarray,
    *,
    reference_vector: tuple[float, float, float] | np.ndarray = (0.0, 0.0, 1.0),
    mode: FrameMode = "minimum_twist",
    angle_offset_deg: float = 0.0,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Return tangent/normal/binormal frames for a resampled centerline.

    ``minimum_twist`` uses parallel transport after initializing the first
    normal from *reference_vector*. ``global_reference`` projects the same
    reference direction into every cross-section plane and keeps signs
    continuous, useful when the sample has a meaningful external orientation.
    """
    c = np.asarray(centerline_nm, dtype=float)
    if c.ndim != 2 or c.shape[0] < 2 or c.shape[1] < 3:
        raise ValueError("Centerline requires at least two 3-D samples.")
    delta = np.empty_like(c[:, :3])
    delta[0] = c[1, :3] - c[0, :3]
    delta[-1] = c[-1, :3] - c[-2, :3]
    if c.shape[0] > 2:
        delta[1:-1] = c[2:, :3] - c[:-2, :3]
    tangent = np.vstack([_normalize(d, fallback=np.array([1.0, 0.0, 0.0])) for d in delta])

    ref = np.asarray(reference_vector, dtype=float)
    normal = np.empty_like(tangent)
    normal[0] = _project_reference(ref, tangent[0])
    if mode == "global_reference":
        for i in range(1, tangent.shape[0]):
            normal[i] = _project_reference(ref, tangent[i], normal[i - 1])
    elif mode == "minimum_twist":
        for i in range(1, tangent.shape[0]):
            n = _rotate_between(normal[i - 1], tangent[i - 1], tangent[i])
            n = n - np.dot(n, tangent[i]) * tangent[i]
            normal[i] = _normalize(n, fallback=_project_reference(ref, tangent[i], normal[i - 1]))
            if np.dot(normal[i], normal[i - 1]) < 0:
                normal[i] = -normal[i]
    else:
        raise ValueError(f"Unknown frame mode: {mode!r}")
    binormal = np.vstack([_normalize(np.cross(t, n)) for t, n in zip(tangent, normal)])
    # Re-orthogonalize normal so all three arrays are mutually consistent.
    normal = np.vstack([_normalize(np.cross(b, t)) for t, b in zip(tangent, binormal)])

    angle = np.deg2rad(float(angle_offset_deg))
    if abs(angle) > 1e-12:
        ca, sa = float(np.cos(angle)), float(np.sin(angle))
        n2 = ca * normal + sa * binormal
        b2 = -sa * normal + ca * binormal
        normal, binormal = n2, b2
    return tangent, normal, binormal


def _centred_edges(centres: np.ndarray, fallback_step: float) -> np.ndarray:
    centres = np.asarray(centres, dtype=float).ravel()
    if centres.size == 1:
        half = max(float(fallback_step), 1.0) / 2.0
        return np.array([centres[0] - half, centres[0] + half], dtype=float)
    edges = np.empty(centres.size + 1, dtype=float)
    edges[1:-1] = 0.5 * (centres[:-1] + centres[1:])
    edges[0] = centres[0] - 0.5 * (centres[1] - centres[0])
    edges[-1] = centres[-1] + 0.5 * (centres[-1] - centres[-2])
    return edges


def _symmetric_edges(half_width_nm: float, pixel_nm: float) -> np.ndarray:
    half = max(float(half_width_nm), 1e-6)
    step = max(float(pixel_nm), 1e-6)
    n = max(1, int(np.ceil((2.0 * half) / step)))
    return np.linspace(-half, half, n + 1, dtype=float)


def straightened_count_volume(
    xyz_nm: np.ndarray,
    skeleton_points_nm: np.ndarray,
    *,
    step_nm: float = 20.0,
    half_width_nm: float = 250.0,
    pixel_nm: float = 10.0,
    frame_mode: FrameMode = "minimum_twist",
    reference_vector: tuple[float, float, float] | np.ndarray = (0.0, 0.0, 1.0),
    angle_offset_deg: float = 0.0,
    projection_mode: ProjectionMode = "sum",
) -> StraightenedVolumeResult:
    """Map localizations into a straightened skeleton coordinate volume.

    The output volume axes are ``(s, u, v)``: arc length along the skeleton,
    local normal coordinate, and local binormal coordinate.  Localizations are
    assigned to their nearest resampled centerline sample.
    """
    xyz = np.asarray(xyz_nm, dtype=float)
    if xyz.ndim != 2 or xyz.shape[1] < 3:
        raise ValueError("Dataset must provide 3-D localization coordinates.")
    xyz = xyz[:, :3]
    finite = np.all(np.isfinite(xyz), axis=1)
    xyz = xyz[finite]
    centerline, arc = resample_skeleton(skeleton_points_nm, step_nm)
    tangent, normal, binormal = build_centerline_frames(
        centerline,
        reference_vector=reference_vector,
        mode=frame_mode,
        angle_offset_deg=angle_offset_deg,
    )
    if xyz.shape[0] == 0:
        used = np.zeros(0, dtype=bool)
        s_val = u_val = v_val = np.zeros(0, dtype=float)
    else:
        from scipy.spatial import cKDTree

        dist, nearest = cKDTree(centerline).query(xyz, k=1)
        offsets = xyz - centerline[nearest]
        u_val = np.einsum("ij,ij->i", offsets, normal[nearest])
        v_val = np.einsum("ij,ij->i", offsets, binormal[nearest])
        s_val = arc[nearest]
        half = max(float(half_width_nm), 1e-6)
        used = (np.abs(u_val) <= half) & (np.abs(v_val) <= half) & np.isfinite(dist)

    s_edges = _centred_edges(arc, step_nm)
    u_edges = _symmetric_edges(half_width_nm, pixel_nm)
    v_edges = _symmetric_edges(half_width_nm, pixel_nm)
    if np.any(used):
        vol, _ = np.histogramdd(
            np.column_stack([s_val[used], u_val[used], v_val[used]]),
            bins=[s_edges, u_edges, v_edges],
        )
        vol = vol.astype(np.float32, copy=False)
    else:
        vol = np.zeros((arc.size, u_edges.size - 1, v_edges.size - 1), dtype=np.float32)

    if projection_mode == "max":
        proj_su = vol.max(axis=2)
        proj_sv = vol.max(axis=1)
        proj_uv = vol.max(axis=0)
    elif projection_mode == "sum":
        proj_su = vol.sum(axis=2)
        proj_sv = vol.sum(axis=1)
        proj_uv = vol.sum(axis=0)
    else:
        raise ValueError(f"Unknown projection mode: {projection_mode!r}")

    return StraightenedVolumeResult(
        centerline_nm=centerline,
        arc_nm=arc,
        tangent=tangent,
        normal=normal,
        binormal=binormal,
        volume=vol,
        s_edges_nm=s_edges,
        u_edges_nm=u_edges,
        v_edges_nm=v_edges,
        projection_su=proj_su.astype(np.float32, copy=False),
        projection_sv=proj_sv.astype(np.float32, copy=False),
        projection_uv=proj_uv.astype(np.float32, copy=False),
        n_input=int(finite.sum()),
        n_used=int(np.count_nonzero(used)),
    )
