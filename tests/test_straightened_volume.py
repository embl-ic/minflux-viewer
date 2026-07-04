import numpy as np

from minflux_viewer.analysis.straightened_volume import (
    build_centerline_frames,
    resample_skeleton,
    straightened_count_volume,
)


def test_resample_skeleton_preserves_endpoints():
    pts = np.array([[0.0, 0.0, 0.0], [35.0, 0.0, 0.0]])
    line, arc = resample_skeleton(pts, 10.0)

    assert np.allclose(line[0], pts[0])
    assert np.allclose(line[-1], pts[-1])
    assert np.allclose(arc, [0.0, 10.0, 20.0, 30.0, 35.0])


def test_centerline_frames_are_orthonormal():
    centre = np.array([
        [0.0, 0.0, 0.0],
        [10.0, 0.0, 0.0],
        [20.0, 10.0, 0.0],
        [25.0, 20.0, 5.0],
    ])
    tangent, normal, binormal = build_centerline_frames(centre)

    for vecs in (tangent, normal, binormal):
        assert np.allclose(np.linalg.norm(vecs, axis=1), 1.0)
    assert np.allclose(np.einsum("ij,ij->i", tangent, normal), 0.0, atol=1e-12)
    assert np.allclose(np.einsum("ij,ij->i", tangent, binormal), 0.0, atol=1e-12)
    assert np.allclose(np.einsum("ij,ij->i", normal, binormal), 0.0, atol=1e-12)


def test_straightened_count_volume_bins_points_near_skeleton():
    skeleton = np.array([[0.0, 0.0, 0.0], [100.0, 0.0, 0.0]])
    xyz = np.array([
        [0.0, 0.0, 0.0],
        [10.0, 5.0, 0.0],
        [20.0, 0.0, 5.0],
        [50.0, 1000.0, 0.0],
    ])

    result = straightened_count_volume(
        xyz,
        skeleton,
        step_nm=10.0,
        half_width_nm=20.0,
        pixel_nm=10.0,
        reference_vector=(0.0, 0.0, 1.0),
    )

    assert result.n_input == 4
    assert result.n_used == 3
    assert result.volume.shape == (11, 4, 4)
    assert float(result.volume.sum()) == 3.0
    assert float(result.projection_su.sum()) == 3.0
    assert float(result.projection_sv.sum()) == 3.0
    assert float(result.projection_uv.sum()) == 3.0
