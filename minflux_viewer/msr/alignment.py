from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable

import numpy as np

NM_PER_M = 1e9


@dataclass
class BeadSummary:
    bead_id: int
    count_used: int
    median_xyz: np.ndarray


@dataclass
class ChannelAlignmentResult:
    channel1_name: str
    channel2_name: str
    bead_ids: np.ndarray
    channel1_xyz: np.ndarray
    channel2_xyz: np.ndarray
    channel2_aligned_xyz: np.ndarray
    channel1_counts: np.ndarray
    channel2_counts: np.ndarray
    rotation: np.ndarray
    translation: np.ndarray
    z_translation: float
    matrix_3x3: np.ndarray
    matrix_4x4: np.ndarray
    rmse: float


def _require_fields(arr: np.ndarray, fields: Iterable[str], label: str) -> None:
    names = set(getattr(getattr(arr, "dtype", None), "names", ()) or ())
    missing = [f for f in fields if f not in names]
    if missing:
        raise ValueError(f"{label} is missing required field(s): {', '.join(missing)}")


def extract_bead_summaries(points: np.ndarray, max_points_per_bead: int = 10) -> dict[int, BeadSummary]:
    _require_fields(points, ("gri", "xyz"), "mbm points")
    bead_xyz: dict[int, list[np.ndarray]] = {}
    for bead_id_raw, xyz_raw in zip(points["gri"], points["xyz"], strict=False):
        bead_id = int(bead_id_raw)
        samples = bead_xyz.setdefault(bead_id, [])
        if len(samples) >= max_points_per_bead:
            continue
        xyz = np.asarray(xyz_raw, dtype=np.float64).reshape(-1)
        if xyz.size < 3:
            raise ValueError("mbm.xyz must contain at least x, y, z coordinates")
        samples.append(xyz[:3])
    summaries: dict[int, BeadSummary] = {}
    for bead_id, samples in bead_xyz.items():
        stack = np.vstack(samples)
        summaries[bead_id] = BeadSummary(
            bead_id=bead_id,
            count_used=stack.shape[0],
            median_xyz=np.median(stack, axis=0),
        )
    return summaries


def compute_rigid_transform_2d(
    source_xy: np.ndarray, target_xy: np.ndarray
) -> tuple[np.ndarray, np.ndarray, np.ndarray, float]:
    src = np.asarray(source_xy, dtype=np.float64)
    dst = np.asarray(target_xy, dtype=np.float64)
    if src.shape != dst.shape or src.ndim != 2 or src.shape[1] != 2:
        raise ValueError("source and target coordinates must both have shape (N, 2)")
    if src.shape[0] < 1:
        raise ValueError("at least one matched bead is required")
    src_centroid = src.mean(axis=0)
    dst_centroid = dst.mean(axis=0)
    src_centered = src - src_centroid
    dst_centered = dst - dst_centroid
    h = src_centered.T @ dst_centered
    u, _, vt = np.linalg.svd(h)
    rotation = vt.T @ u.T
    if np.linalg.det(rotation) < 0:
        vt[-1, :] *= -1.0
        rotation = vt.T @ u.T
    translation = dst_centroid - rotation @ src_centroid
    transformed = (rotation @ src.T).T + translation
    rmse = float(np.sqrt(np.mean(np.sum((transformed - dst) ** 2, axis=1))))
    matrix = np.eye(3, dtype=np.float64)
    matrix[:2, :2] = rotation
    matrix[:2, 2] = translation
    return rotation, translation, matrix, rmse


def apply_rigid_transform_2d(points: np.ndarray, matrix_3x3: np.ndarray) -> np.ndarray:
    pts = np.asarray(points, dtype=np.float64)
    if pts.ndim != 2 or pts.shape[1] < 2:
        raise ValueError("points must have shape (N, 2+) to apply a 2D rigid transform")
    out = pts.copy()
    xy_h = np.concatenate([out[:, :2], np.ones((out.shape[0], 1), dtype=np.float64)], axis=1)
    out[:, :2] = (matrix_3x3 @ xy_h.T).T[:, :2]
    return out


def apply_homogeneous_transform(points: np.ndarray, matrix_4x4: np.ndarray) -> np.ndarray:
    pts = np.asarray(points, dtype=np.float64)
    if pts.ndim != 2 or pts.shape[1] < 2:
        raise ValueError("points must have shape (N, 2+) to apply a display transform")
    out = pts.copy()
    if out.shape[1] < 3:
        out = np.column_stack([out[:, :2], np.zeros(out.shape[0], dtype=np.float64)])
    xyz_h = np.concatenate([out[:, :3], np.ones((out.shape[0], 1), dtype=np.float64)], axis=1)
    out[:, :3] = (matrix_4x4 @ xyz_h.T).T[:, :3]
    return out


def align_channels_from_arrays(
    channel1_name: str,
    channel1_points: np.ndarray,
    channel2_name: str,
    channel2_points: np.ndarray,
    max_points_per_bead: int = 10,
) -> ChannelAlignmentResult:
    ch1 = extract_bead_summaries(channel1_points, max_points_per_bead=max_points_per_bead)
    ch2 = extract_bead_summaries(channel2_points, max_points_per_bead=max_points_per_bead)
    bead_ids = np.array(sorted(set(ch1) & set(ch2)), dtype=np.uint32)
    if bead_ids.size == 0:
        raise ValueError("no overlapping bead IDs were found between the two channels")
    channel1_xyz = np.vstack([ch1[int(b)].median_xyz for b in bead_ids]) * NM_PER_M
    channel2_xyz = np.vstack([ch2[int(b)].median_xyz for b in bead_ids]) * NM_PER_M
    channel1_counts = np.array([ch1[int(b)].count_used for b in bead_ids], dtype=np.int32)
    channel2_counts = np.array([ch2[int(b)].count_used for b in bead_ids], dtype=np.int32)
    rotation, translation, matrix_3x3, rmse = compute_rigid_transform_2d(
        channel2_xyz[:, :2], channel1_xyz[:, :2]
    )
    xy_aligned = apply_rigid_transform_2d(channel2_xyz, matrix_3x3)
    z_translation = 0.0
    if (
        np.any(np.isfinite(channel1_xyz[:, 2]))
        and np.any(np.isfinite(channel2_xyz[:, 2]))
        and (np.nanmax(np.abs(channel1_xyz[:, 2])) > 0 or np.nanmax(np.abs(channel2_xyz[:, 2])) > 0)
    ):
        z_translation = float(np.nanmedian(channel1_xyz[:, 2] - channel2_xyz[:, 2]))
    matrix_4x4 = np.eye(4, dtype=np.float64)
    matrix_4x4[:2, :2] = rotation
    matrix_4x4[:2, 3] = translation
    matrix_4x4[2, 3] = z_translation
    channel2_aligned_xyz = xy_aligned
    channel2_aligned_xyz[:, 2] = channel2_aligned_xyz[:, 2] + z_translation
    return ChannelAlignmentResult(
        channel1_name=channel1_name,
        channel2_name=channel2_name,
        bead_ids=bead_ids,
        channel1_xyz=channel1_xyz,
        channel2_xyz=channel2_xyz,
        channel2_aligned_xyz=channel2_aligned_xyz,
        channel1_counts=channel1_counts,
        channel2_counts=channel2_counts,
        rotation=rotation,
        translation=translation,
        z_translation=z_translation,
        matrix_3x3=matrix_3x3,
        matrix_4x4=matrix_4x4,
        rmse=rmse,
    )
