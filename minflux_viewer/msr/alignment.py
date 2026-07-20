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


#: Bead-fit XY RMSE (nm) above which the fiducial registration is flagged as poor.
#: MINFLUX localization precision is ~1-5 nm and a good two-colour bead registration
#: is typically well under this; a larger residual means the beads disagree among
#: themselves and a single transform cannot register the channels accurately.
RMSE_WARN_NM = 20.0


def alignment_quality_report(result: ChannelAlignmentResult) -> dict:
    """Per-bead residual diagnostics for a channel-alignment result.

    Returns a plain dict (Qt-free, so it is unit-testable and reusable by any UI):
    the raw per-bead offset (reference − moving), the post-fit residual
    (aligned − reference), XY/Z RMSE recomputed consistently from the residuals
    (``result.rmse`` is XY-only for the default mode but 3-D for the others), and a
    ``warn`` flag set when the XY RMSE exceeds :data:`RMSE_WARN_NM`.
    """
    ref = np.asarray(result.channel1_xyz, dtype=np.float64)
    mov = np.asarray(result.channel2_xyz, dtype=np.float64)
    aligned = np.asarray(result.channel2_aligned_xyz, dtype=np.float64)
    bead_ids = np.asarray(result.bead_ids)
    raw_offset = ref - mov          # what the user inspects in "Show beads drift"
    residual = aligned - ref        # remaining error after the fitted transform
    if residual.shape[0]:
        resid_xy = np.hypot(residual[:, 0], residual[:, 1])
        rmse_xy = float(np.sqrt(np.mean(residual[:, 0] ** 2 + residual[:, 1] ** 2)))
        rmse_z = float(np.sqrt(np.mean(residual[:, 2] ** 2)))
    else:
        resid_xy = np.zeros(0, dtype=np.float64)
        rmse_xy = rmse_z = 0.0
    return {
        "reference": result.channel1_name,
        "moving": result.channel2_name,
        "transform_type": getattr(result, "transform_type", "") or "",
        "n_beads": int(bead_ids.size),
        "rmse_xy_nm": rmse_xy,
        "rmse_z_nm": rmse_z,
        "warn": rmse_xy > RMSE_WARN_NM,
        "bead_ids": bead_ids,
        "raw_offset_nm": raw_offset,
        "residual_nm": residual,
        "residual_xy_nm": resid_xy,
    }


def bead_residual_comment(
    raw_offset_nm,
    residual_nm,
    *,
    included: bool = True,
    rmse_xy_nm: float | None = None,
    warn_nm: float = RMSE_WARN_NM,
) -> str:
    """A short plain-language evaluation of one bead, to help the user decide whether
    to keep or exclude it. Reads the raw per-bead offset (reference − moving) and the
    post-fit residual (aligned − reference); ``rmse_xy_nm`` is the current fit's XY
    RMSE, used to judge whether a bead is a *standout* relative to the whole set (in
    an incoherent field every bead has a large residual, so "outlier" must be relative,
    not absolute).
    """
    dx, dy, dz = (float(v) for v in np.asarray(raw_offset_nm, dtype=np.float64).reshape(-1)[:3])
    rx, ry, rz = (float(v) for v in np.asarray(residual_nm, dtype=np.float64).reshape(-1)[:3])
    rxy = float(np.hypot(rx, ry))
    raw_xy = float(np.hypot(dx, dy))
    if not included:
        return f"Excluded — would sit {rxy:.0f} nm from the current fit."

    has_rmse = rmse_xy_nm is not None and np.isfinite(rmse_xy_nm) and rmse_xy_nm > 0
    standout = max(2.0 * warn_nm, 1.5 * rmse_xy_nm) if has_rmse else 2.0 * warn_nm
    if rxy <= warn_nm:
        parts = [f"Good — consistent with the fit ({rxy:.0f} nm)."]
    elif rxy >= standout:
        parts = [f"Outlier ({rxy:.0f} nm) — disagrees with the fit; excluding may help."]
    elif has_rmse and rxy <= rmse_xy_nm:
        parts = [f"Near the fit's average ({rxy:.0f} nm ≈ RMSE {rmse_xy_nm:.0f} nm)."]
    else:
        tail = f"; fit is poor overall (RMSE {rmse_xy_nm:.0f} nm)" if has_rmse else ""
        parts = [f"Elevated residual ({rxy:.0f} nm){tail}."]

    if abs(rz) > warn_nm and abs(rz) >= rxy:
        parts.append(f"Z error dominates ({rz:+.0f} nm).")
    if raw_xy > 500.0:
        parts.append(f"Unusually large raw offset ({raw_xy:.0f} nm) — check bead.")
    return " ".join(parts)


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
    return align_channels_from_bead_positions(
        channel1_name, channel2_name, bead_ids, channel1_xyz, channel2_xyz,
        channel1_counts=channel1_counts, channel2_counts=channel2_counts,
        transform_type=transform_type)


def align_channels_from_bead_positions(
    channel1_name: str,
    channel2_name: str,
    bead_ids: np.ndarray,
    channel1_xyz: np.ndarray,
    channel2_xyz: np.ndarray,
    channel1_counts: np.ndarray | None = None,
    channel2_counts: np.ndarray | None = None,
    transform_type: str = TRANSFORM_RIGID_XY_Z,
) -> ChannelAlignmentResult:
    """Solve a channel alignment from already-extracted per-bead median positions
    (nm). Splits the fit step out of :func:`align_channels_from_arrays` so a UI can
    cheaply re-fit on a bead *subset* without re-parsing the raw MBM point arrays.
    """
    bead_ids = np.asarray(bead_ids, dtype=np.uint32)
    channel1_xyz = np.asarray(channel1_xyz, dtype=np.float64)
    channel2_xyz = np.asarray(channel2_xyz, dtype=np.float64)
    if bead_ids.size == 0:
        raise ValueError("no beads were provided for alignment")
    if channel1_counts is None:
        channel1_counts = np.zeros(bead_ids.size, dtype=np.int32)
    if channel2_counts is None:
        channel2_counts = np.zeros(bead_ids.size, dtype=np.int32)
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
        channel1_counts=np.asarray(channel1_counts, dtype=np.int32),
        channel2_counts=np.asarray(channel2_counts, dtype=np.int32),
        rotation=rotation,
        translation=translation,
        z_translation=z_translation,
        matrix_3x3=matrix_3x3,
        matrix_4x4=matrix_4x4,
        rmse=rmse,
        transform_type=transform_type,
    )
