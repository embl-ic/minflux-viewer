"""NPC ring-convolution segmentation (port of segment_NPC_with_units 2-D step)."""

import numpy as np

from minflux_viewer.analysis.npc_segmentation import (
    ring_kernel,
    ring_support_scores,
    segment_npc_2d,
)


def _ring(cx, cy, diameter, n=120, sigma=4.0, seed=0):
    rng = np.random.default_rng(seed)
    theta = rng.uniform(0, 2 * np.pi, n)
    r = diameter / 2.0 + rng.normal(0, sigma, n)
    return np.column_stack([cx + r * np.cos(theta), cy + r * np.sin(theta)])


def test_ring_kernel_peaks_at_radius():
    k = ring_kernel(70.0, 20.0, 4.0)
    assert np.isclose(k.sum(), 1.0)
    # kernel centre (radius 0) should be lower than the ring (radius 35)
    c = k.shape[0] // 2
    assert k[c, c] < k.max()


def test_detects_two_npcs():
    d = 100.0
    rings = np.vstack([_ring(0, 0, d, seed=1), _ring(400, 250, d, seed=2)])
    res = segment_npc_2d(rings, diameter_nm=d, rim_nm=25.0, pixel_size_nm=5.0)
    centers = res["centers"]
    assert centers.shape[0] == 2
    # each true centre is matched within a few nm
    for tc in ([0, 0], [400, 250]):
        assert np.min(np.hypot(centers[:, 0] - tc[0], centers[:, 1] - tc[1])) < 20.0


def test_support_high_on_ring_low_on_blob():
    d = 100.0
    ring = _ring(0, 0, d, n=200, sigma=3.0, seed=3)
    s_ring = ring_support_scores(ring, np.array([[0.0, 0.0]]), d, 25.0)[0]
    # a dense central blob has no annulus support
    rng = np.random.default_rng(4)
    blob = rng.normal(0, 4, (200, 2))
    s_blob = ring_support_scores(blob, np.array([[0.0, 0.0]]), d, 25.0)[0]
    assert s_ring > 0.4
    assert not np.isfinite(s_blob) or s_blob < s_ring


def test_empty_and_sparse_safe():
    assert segment_npc_2d(np.empty((0, 2)), diameter_nm=100, rim_nm=20)["centers"].shape == (0, 2)
    assert segment_npc_2d(np.zeros((2, 2)), diameter_nm=100, rim_nm=20)["centers"].shape == (0, 2)
