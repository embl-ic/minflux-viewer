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
    transform_type: str = ""


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


#: MBM-handling transform modes (Preferences > MBM Handling > transform_type).
TRANSFORM_RIGID_XY_Z = "rigid XY + translational Z"
TRANSFORM_TRANS_XYZ = "translational XYZ"
TRANSFORM_RIGID_XYZ = "rigid XYZ"

#: Absolute minimum matched beads required to attempt any channel alignment.
#: (2 → translational XYZ only; 3 → also rigid XY + translational Z;
#: 4 → full rigid XYZ — see ``effective_transform_type``.)
MIN_BEADS_FOR_ALIGNMENT = 2


def effective_transform_type(requested: str, n_beads: int) -> str:
    """Constrain the *requested* MBM transform mode to one solvable with
    *n_beads* matched beads.

    Bead-count rules (identical for 2-D and 3-D, overriding the preference):
    2 beads → translational XYZ only; 3 beads → translational XYZ or
    rigid XY + translational Z (full rigid XYZ is downgraded to the latter,
    since 3 points cannot fix a unique 3-D rotation robustly); 4+ beads →
    the requested mode is honoured unchanged.
    """
    mode = (requested or "").strip().lower()
    if n_beads >= 4:
        return requested
    if n_beads == 3:
        return TRANSFORM_RIGID_XY_Z if mode == TRANSFORM_RIGID_XYZ.lower() else requested
    return TRANSFORM_TRANS_XYZ


def _z_meaningful(*arrays: np.ndarray) -> bool:
    return any(
        np.any(np.isfinite(a[:, 2])) and np.nanmax(np.abs(a[:, 2])) > 0
        for a in arrays
    )


def _kabsch(src: np.ndarray, dst: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Full 3-D rigid (rotation + translation) least-squares fit src → dst."""
    src_c, dst_c = src.mean(axis=0), dst.mean(axis=0)
    h = (src - src_c).T @ (dst - dst_c)
    u, _, vt = np.linalg.svd(h)
    rotation = vt.T @ u.T
    if np.linalg.det(rotation) < 0:
        vt = vt.copy()
        vt[-1, :] *= -1.0
        rotation = vt.T @ u.T
    return rotation, dst_c - rotation @ src_c


def _rmse3(aligned: np.ndarray, target: np.ndarray) -> float:
    return float(np.sqrt(np.mean(np.sum((aligned - target) ** 2, axis=1))))


def solve_channel_transform(src_xyz: np.ndarray, dst_xyz: np.ndarray, transform_type: str):
    """Solve a channel2→channel1 transform for the chosen MBM-handling mode.

    Returns ``(rotation2x2, translation2, z_translation, matrix_3x3, matrix_4x4,
    aligned_xyz, rmse)``. ``rigid XY + translational Z`` (default) reproduces the
    original 2-D rigid + Z-shift; ``translational XYZ`` is a pure 3-D shift;
    ``rigid XYZ`` is a full 3-D rigid (rotation in Z too).
    """
    mode = (transform_type or "").strip().lower()
    src = np.asarray(src_xyz, dtype=np.float64)
    dst = np.asarray(dst_xyz, dtype=np.float64)

    if mode == TRANSFORM_TRANS_XYZ.lower():
        t3 = np.median(dst - src, axis=0)
        rot3 = np.eye(3, dtype=np.float64)
        aligned = src + t3
        rmse = _rmse3(aligned, dst)
    elif mode == TRANSFORM_RIGID_XYZ.lower():
        rot3, t3 = _kabsch(src, dst)
        aligned = (rot3 @ src.T).T + t3
        rmse = _rmse3(aligned, dst)
    else:  # rigid XY + translational Z (default)
        rotation, translation, _m3, rmse = compute_rigid_transform_2d(src[:, :2], dst[:, :2])
        rot3 = np.eye(3, dtype=np.float64)
        rot3[:2, :2] = rotation
        t3 = np.array([translation[0], translation[1], 0.0], dtype=np.float64)
        if _z_meaningful(src, dst):
            t3[2] = float(np.nanmedian(dst[:, 2] - src[:, 2]))
        aligned = np.column_stack([
            (rotation @ src[:, :2].T).T + translation,
            src[:, 2] + t3[2],
        ])

    matrix_4x4 = np.eye(4, dtype=np.float64)
    matrix_4x4[:3, :3] = rot3
    matrix_4x4[:3, 3] = t3
    matrix_3x3 = np.eye(3, dtype=np.float64)
    matrix_3x3[:2, :2] = rot3[:2, :2]
    matrix_3x3[:2, 2] = t3[:2]
    return rot3[:2, :2], t3[:2], float(t3[2]), matrix_3x3, matrix_4x4, aligned, float(rmse)


def align_channels_from_arrays(
    channel1_name: str,
    channel1_points: np.ndarray,
    channel2_name: str,
    channel2_points: np.ndarray,
    max_points_per_bead: int = 10,
    transform_type: str = TRANSFORM_RIGID_XY_Z,
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
    # Downgrade the requested mode to what this many matched beads can support
    # (overriding the preference for low bead counts — see effective_transform_type).
    transform_type = effective_transform_type(transform_type, int(bead_ids.size))
    (rotation, translation, z_translation, matrix_3x3, matrix_4x4,
     channel2_aligned_xyz, rmse) = solve_channel_transform(
        channel2_xyz, channel1_xyz, transform_type)
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
        transform_type=transform_type,
    )
