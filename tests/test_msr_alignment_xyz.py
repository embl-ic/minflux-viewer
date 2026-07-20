"""MSR 'mbm info' channel alignment must align BOTH XY and Z.

The moving channel sees the same MBM beads through a known misalignment (XY
rotation + XY translation + Z shift); the bead-based transform must invert it in
all three axes, and the same 4x4 applied through the overlay display path must move
the localizations to the reference frame in Z as well as XY.
"""

from __future__ import annotations

import numpy as np

from minflux_viewer.core.overlay import apply_display_transform_nm
from minflux_viewer.msr.alignment import (
    TRANSFORM_RIGID_XY_Z,
    TRANSFORM_RIGID_XYZ,
    TRANSFORM_TRANS_XYZ,
    align_channels_from_arrays,
)

_POINT_DTYPE = np.dtype([("gri", "<u4"), ("xyz", "<f8", (3,))])


def _beads(xyz_m: np.ndarray, seed: int = 0) -> np.ndarray:
    """Structured MBM points (metres): 5 slightly-jittered samples per bead."""
    rng = np.random.default_rng(seed)
    rows = [(i + 1, p + rng.normal(0, 0.3e-9, 3)) for i, p in enumerate(xyz_m) for _ in range(5)]
    return np.array(rows, dtype=_POINT_DTYPE)


def _reference_and_moving():
    rng = np.random.default_rng(0)
    ref_m = np.column_stack([rng.uniform(-1000, 1000, 6), rng.uniform(-1000, 1000, 6),
                             rng.uniform(-100, 100, 6)]) * 1e-9
    theta = np.radians(12.0)
    c, s = np.cos(theta), np.sin(theta)
    Rz = np.array([[c, -s, 0], [s, c, 0], [0, 0, 1.0]])
    shift = np.array([40e-9, -25e-9, 30e-9])           # 40, -25 nm XY;  +30 nm Z
    moving_m = (Rz @ ref_m.T).T + shift
    return ref_m, moving_m


def test_mbm_alignment_aligns_xy_and_z():
    ref_m, moving_m = _reference_and_moving()
    res = align_channels_from_arrays("ref", _beads(ref_m, 1), "mov", _beads(moving_m, 2),
                                     transform_type=TRANSFORM_RIGID_XY_Z)
    err = res.channel2_aligned_xyz - res.channel1_xyz     # aligned beads vs reference (nm)
    assert np.sqrt(np.mean(err[:, :2] ** 2)) < 1.0        # XY aligned (sub-nm vs jitter)
    assert np.sqrt(np.mean(err[:, 2] ** 2)) < 1.0         # Z aligned
    assert np.isfinite(res.z_translation)
    assert abs(abs(res.z_translation) - 30.0) < 1.5       # ~ the 30 nm Z shift (sign undoes it)


def test_mbm_alignment_z_is_in_the_4x4_and_applied_by_the_overlay_path():
    """The Z component lives in matrix_4x4[2, 3] and is applied to localizations by
    the same overlay display transform the render/scatter windows use."""
    ref_m, moving_m = _reference_and_moving()
    res = align_channels_from_arrays("ref", _beads(ref_m, 1), "mov", _beads(moving_m, 2),
                                     transform_type=TRANSFORM_RIGID_XY_Z)
    assert abs(res.matrix_4x4[2, 3]) > 25.0               # Z translation is in the 4x4

    aligned = apply_display_transform_nm(moving_m * 1e9, {"matrix_4x4": res.matrix_4x4.tolist()})
    ref_nm = ref_m * 1e9
    assert np.sqrt(np.mean((aligned[:, 2] - ref_nm[:, 2]) ** 2)) < 1.0   # Z aligned in display
    assert np.sqrt(np.mean((aligned[:, :2] - ref_nm[:, :2]) ** 2)) < 1.0  # XY aligned in display


def test_rigid_xyz_mode_aligns_all_three_axes():
    ref_m, moving_m = _reference_and_moving()
    res = align_channels_from_arrays("ref", _beads(ref_m, 1), "mov", _beads(moving_m, 2),
                                     transform_type=TRANSFORM_RIGID_XYZ)
    err = res.channel2_aligned_xyz - res.channel1_xyz
    assert np.sqrt(np.mean(err ** 2)) < 1.0              # full 3-D rigid aligns everything


def test_translational_mode_aligns_z_but_not_the_rotation():
    """Pure-translation mode still aligns the Z shift, but cannot undo the XY
    rotation — documents the mode's scope."""
    ref_m, moving_m = _reference_and_moving()
    res = align_channels_from_arrays("ref", _beads(ref_m, 1), "mov", _beads(moving_m, 2),
                                     transform_type=TRANSFORM_TRANS_XYZ)
    err = res.channel2_aligned_xyz - res.channel1_xyz
    assert np.sqrt(np.mean(err[:, 2] ** 2)) < 1.0         # Z shift undone
    assert np.sqrt(np.mean(err[:, :2] ** 2)) > 10.0       # 12 deg rotation not undone
