from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable

import numpy as np


CHANNEL_LUTS = [
    "Red", "Green", "Blue", "Cyan", "Magenta", "Yellow", "Gray",
    "hot", "gray", "viridis", "plasma", "inferno", "magma", "turbo",
]

PURE_COLOR_RGB = {
    "Red": (220, 40, 40),
    "Green": (20, 170, 70),
    "Blue": (50, 90, 230),
    "Cyan": (0, 170, 190),
    "Magenta": (190, 50, 190),
    "Yellow": (210, 170, 20),
    "Gray": (120, 120, 120),
}

#: Default per-dataset overlay channel colours (1st..6th), cycled for overlays.
DEFAULT_OVERLAY_COLORS = ["Red", "Green", "Blue", "Cyan", "Magenta", "Yellow"]


def overlay_color_cycle(prefs: dict | None) -> list[str]:
    """The configured overlay channel-colour cycle (Preferences > Appearance >
    Overlay), or the built-in default. Always returns a non-empty list."""
    colors = None
    try:
        colors = (prefs or {}).get("plot", {}).get("overlay_colors")
    except Exception:
        colors = None
    if isinstance(colors, (list, tuple)):
        out = [str(c) for c in colors if str(c).strip()]
        if out:
            return out
    return list(DEFAULT_OVERLAY_COLORS)


@dataclass
class OverlayMemberSpec:
    dataset_idx: int
    order: int
    lut: str


def dataset_group_id(ds) -> str | None:
    state = getattr(ds, "state", {})
    return state.get("overlay_id") or state.get("render_group_id")


def dataset_overlay_index(ds) -> int | None:
    try:
        return int(getattr(ds, "state", {}).get("overlay_index"))
    except Exception:
        return None


def overlay_members(state, anchor_idx: int) -> list[tuple[int, object]]:
    if not (0 <= anchor_idx < len(state.datasets)):
        return []
    group_id = dataset_group_id(state.datasets[anchor_idx])
    if not group_id:
        return [(anchor_idx, state.datasets[anchor_idx])]
    out: list[tuple[int, object]] = []
    for idx, ds in enumerate(state.datasets):
        if dataset_group_id(ds) == group_id:
            out.append((idx, ds))
    return sorted(out, key=lambda pair: int(pair[1].state.get("overlay_order", pair[0] + 1)))


def matrix4_to_xy3(matrix_4x4: np.ndarray) -> np.ndarray:
    matrix = np.asarray(matrix_4x4, dtype=np.float64)
    out = np.eye(3, dtype=np.float64)
    out[:2, :2] = matrix[:2, :2]
    out[:2, 2] = matrix[:2, 3]
    return out


def identity_matrix4() -> np.ndarray:
    return np.eye(4, dtype=np.float64)


def display_transform_record(
    *,
    overlay_id: str,
    overlay_index: int,
    order: int,
    lut: str,
    source_dataset_idx: int,
    alignment_mode: str,
    matrix_4x4,
    provenance: dict | None = None,
) -> dict:
    matrix = np.asarray(matrix_4x4, dtype=np.float64)
    matrix_3 = matrix4_to_xy3(matrix)
    return {
        "overlay_id": overlay_id,
        "overlay_index": int(overlay_index),
        "order": int(order),
        "lut": lut,
        "source_dataset_idx": int(source_dataset_idx),
        "alignment_mode": alignment_mode,
        "matrix_4x4": matrix.tolist(),
        "matrix_3x3": matrix_3.tolist(),
        "provenance": dict(provenance or {}),
    }


def apply_display_transform_nm(points_nm: np.ndarray, transform: dict | None) -> np.ndarray:
    pts = np.asarray(points_nm, dtype=np.float64)
    if pts.ndim != 2 or pts.shape[0] == 0:
        return pts.copy()
    if transform is None:
        return pts.copy()
    matrix_4 = transform.get("matrix_4x4") if isinstance(transform, dict) else None
    if matrix_4 is not None:
        mat = np.asarray(matrix_4, dtype=np.float64)
        if mat.shape == (4, 4):
            out = pts.copy()
            if out.shape[1] < 3:
                out = np.column_stack([out[:, :2], np.zeros(out.shape[0], dtype=np.float64)])
            xyz_h = np.column_stack([out[:, :3], np.ones(out.shape[0], dtype=np.float64)])
            out[:, :3] = (mat @ xyz_h.T).T[:, :3]
            return out
    matrix_3 = transform.get("matrix_3x3") if isinstance(transform, dict) else None
    if matrix_3 is not None:
        mat = np.asarray(matrix_3, dtype=np.float64)
        if mat.shape == (3, 3) and pts.shape[1] >= 2:
            out = pts.copy()
            xy_h = np.column_stack([out[:, :2], np.ones(out.shape[0], dtype=np.float64)])
            out[:, :2] = (mat @ xy_h.T).T[:, :2]
            return out
    return pts.copy()


def transform_to_matrix4(transform) -> np.ndarray | None:
    """The 4×4 display matrix of a *transform*, or ``None`` if not resolvable.

    Accepts a ``display_transform_record`` **dict** (``matrix_4x4`` / ``matrix_3x3``),
    a raw 4×4 array, or a 3×3 XY affine (embedded into 4×4). Overlay datasets store
    the transform as a dict, so callers that want the bare matrix must go through
    this rather than ``np.asarray(transform)`` (which raises on a dict)."""
    def _embed3(a: np.ndarray) -> np.ndarray:
        out = np.eye(4, dtype=np.float64)
        out[:2, :2] = a[:2, :2]
        out[:2, 3] = a[:2, 2]
        return out

    if transform is None:
        return None
    if isinstance(transform, dict):
        m4 = transform.get("matrix_4x4")
        if m4 is not None:
            arr = np.asarray(m4, dtype=np.float64)
            if arr.shape == (4, 4):
                return arr
        m3 = transform.get("matrix_3x3")
        if m3 is not None:
            arr = np.asarray(m3, dtype=np.float64)
            if arr.shape == (3, 3):
                return _embed3(arr)
        return None
    try:
        arr = np.asarray(transform, dtype=np.float64)
    except (TypeError, ValueError):
        return None
    if arr.shape == (4, 4):
        return arr
    if arr.shape == (3, 3):
        return _embed3(arr)
    return None


def transform_key(transform: dict | None) -> tuple:
    if not isinstance(transform, dict):
        return ()
    matrix = transform.get("matrix_4x4") or transform.get("matrix_3x3")
    if matrix is None:
        return ()
    arr = np.asarray(matrix, dtype=np.float64)
    if arr.shape not in ((3, 3), (4, 4)):
        return ()
    return tuple(np.round(arr.ravel(), 6).tolist())


def localization_centroid_nm(ds) -> np.ndarray:
    loc = np.asarray(ds.loc_nm, dtype=np.float64)
    if loc.size == 0:
        return np.zeros(3, dtype=np.float64)
    if loc.shape[1] < 3:
        loc = np.column_stack([loc[:, :2], np.zeros(loc.shape[0], dtype=np.float64)])
    finite = np.all(np.isfinite(loc[:, :3]), axis=1)
    if not np.any(finite):
        return np.zeros(3, dtype=np.float64)
    return np.nanmedian(loc[finite, :3], axis=0)


def mbm_points_array(ds):
    mbm = getattr(ds, "mbm", None)
    if mbm is not None:
        points = mbm.get_attr("points", None)
        if points is not None:
            return np.asarray(points)
    points = getattr(ds, "metadata", {}).get("mbm_points")
    if points is not None:
        return np.asarray(points)
    return None


def build_overlay_transforms(
    *,
    state,
    members: Iterable[OverlayMemberSpec],
    overlay_id: str,
    overlay_index: int,
    alignment_mode: str,
) -> dict[int, dict]:
    ordered = sorted(list(members), key=lambda spec: spec.order)
    if not ordered:
        return {}
    ref_idx = ordered[0].dataset_idx
    ref_ds = state.datasets[ref_idx]
    records: dict[int, dict] = {}
    for spec in ordered:
        ds = state.datasets[spec.dataset_idx]
        matrix = identity_matrix4()
        provenance: dict = {"reference_dataset_idx": ref_idx}
        if spec.dataset_idx != ref_idx and alignment_mode == "data centroid":
            matrix[:3, 3] = localization_centroid_nm(ref_ds) - localization_centroid_nm(ds)
            provenance["method"] = "median localization centroid"
        elif spec.dataset_idx != ref_idx and alignment_mode == "mbm info if available":
            ref_points = mbm_points_array(ref_ds)
            moving_points = mbm_points_array(ds)
            if ref_points is not None and moving_points is not None:
                from ..msr.alignment import align_channels_from_arrays

                result = align_channels_from_arrays(ref_ds.name, ref_points, ds.name, moving_points)
                matrix = np.asarray(result.matrix_4x4, dtype=np.float64)
                provenance.update({
                    "method": "mbm xy rigid plus z translation",
                    "matched_bead_count": int(result.bead_ids.size),
                    "rmse_xy_nm": float(result.rmse),
                    "z_translation_nm": float(result.z_translation),
                })
            else:
                matrix[:3, 3] = localization_centroid_nm(ref_ds) - localization_centroid_nm(ds)
                provenance["method"] = "mbm unavailable; median localization centroid fallback"
        else:
            provenance["method"] = "identity"
        records[spec.dataset_idx] = display_transform_record(
            overlay_id=overlay_id,
            overlay_index=overlay_index,
            order=spec.order,
            lut=spec.lut,
            source_dataset_idx=spec.dataset_idx,
            alignment_mode=alignment_mode,
            matrix_4x4=matrix,
            provenance=provenance,
        )
    return records
