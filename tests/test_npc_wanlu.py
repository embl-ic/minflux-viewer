"""MINFLUX sub-unit (trace-cluster) detection (analysis/npc_wanlu.py).

Pure NumPy/SciPy — no Qt. The former Wanlu two-ring detection/averaging pipeline
was removed (superseded by the unbiased ``npc_geomfit`` model fitter); this module
now provides only the sub-unit clustering that the geometry fit reuses to seed
its fits.
"""

import numpy as np

from minflux_viewer.analysis import npc_wanlu as nw


def _npc_traces(cx, cy, cz, *, radius=37.5, inter=25.0, rot=0.0, seed=0,
                locs_per_unit=20):
    """One NPC as 16 corner traces (8 corners × 2 Z rings) → (loc (M,3), tid)."""
    r = np.random.default_rng(seed)
    locs, tids, tid = [], [], 0
    for zc in (-inter / 2, inter / 2):
        for c in range(8):
            tid += 1
            ang = rot + 2 * np.pi * c / 8
            ux, uy, uz = cx + radius * np.cos(ang), cy + radius * np.sin(ang), cz + zc
            n = locs_per_unit
            locs.append(np.column_stack([r.normal(ux, 3, n), r.normal(uy, 3, n), r.normal(uz, 2, n)]))
            tids.append(np.full(n, tid))
    return np.vstack(locs), np.concatenate(tids)


def test_estimate_trace_size_positive():
    loc, tid = _npc_traces(0, 0, 0, seed=1)
    s = nw.estimate_trace_size(loc, tid)
    assert 1.0 < s < 30.0


def test_single_linkage_groups_clusters():
    pts = np.vstack([np.zeros((5, 3)), np.full((5, 3), 100.0)])
    labels = nw._single_linkage_labels(pts, eps=5.0)
    assert len(np.unique(labels)) == 2


def test_locate_unit_clusters_finds_corners():
    loc, tid = _npc_traces(0, 0, 0, radius=40, seed=2, locs_per_unit=25)
    centers, loc_unit = nw.locate_unit_clusters(loc, tid, d_unit=20.0)
    assert centers.shape[0] >= 8           # at least the 8 XY corners resolved
    assert loc_unit.max() == centers.shape[0]


def test_empty_inputs_safe():
    centers, loc_unit = nw.locate_unit_clusters(np.empty((0, 3)), np.empty(0), d_unit=20.0)
    assert centers.shape[0] == 0 and loc_unit.shape[0] == 0
    assert nw.estimate_trace_size(np.empty((0, 3)), np.empty(0)) == 10.0
