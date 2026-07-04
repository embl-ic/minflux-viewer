"""Wanlu NPC detection + two-ring averaging (analysis/npc_wanlu.py).

Pure NumPy/SciPy — no Qt. Synthetic 8-fold two-ring NPCs validate sub-unit
clustering, Z-peak detection, the constrained two-ring fit, 8-fold phase
alignment, and the end-to-end detect → average pipeline.
"""

import numpy as np
import pytest

from minflux_viewer.analysis import npc_wanlu as nw


# --- synthetic builders --------------------------------------------------------
def _npc_traces(cx, cy, cz, *, radius=37.5, inter=25.0, rot=0.0, seed=0,
                locs_per_unit=20, tid0=0):
    """One NPC as 16 corner traces (8 corners × 2 Z rings). Returns
    (loc (M,3), tid (M,), next_tid)."""
    r = np.random.default_rng(seed)
    locs, tids = [], []
    tid = tid0
    for zc in (-inter / 2, inter / 2):
        for c in range(8):
            tid += 1
            ang = rot + 2 * np.pi * c / 8
            ux, uy, uz = cx + radius * np.cos(ang), cy + radius * np.sin(ang), cz + zc
            n = locs_per_unit
            locs.append(np.column_stack([r.normal(ux, 3, n), r.normal(uy, 3, n), r.normal(uz, 2, n)]))
            tids.append(np.full(n, tid))
    return np.vstack(locs), np.concatenate(tids), tid


def _raw6_npc(cx, cy, cz, npc_id, *, rot=0.0, seed=0, tid0=0):
    loc, tid, nxt = _npc_traces(cx, cy, cz, rot=rot, seed=seed, tid0=tid0)
    # unit_id == trace id here (one trace per corner)
    uid = tid - tid0
    raw = np.column_stack([loc, np.full(loc.shape[0], npc_id, float), uid.astype(float), tid.astype(float)])
    return raw, nxt


# --- sub-unit clustering -------------------------------------------------------
def test_estimate_trace_size_positive():
    loc, tid, _ = _npc_traces(0, 0, 0, seed=1)
    s = nw.estimate_trace_size(loc, tid)
    assert 1.0 < s < 30.0


def test_single_linkage_groups_clusters():
    pts = np.vstack([np.zeros((5, 3)), np.full((5, 3), 100.0)])
    labels = nw._single_linkage_labels(pts, eps=5.0)
    assert len(np.unique(labels)) == 2


def test_locate_unit_clusters_finds_corners():
    loc, tid, _ = _npc_traces(0, 0, 0, radius=40, seed=2, locs_per_unit=25)
    centers, loc_unit = nw.locate_unit_clusters(loc, tid, d_unit=20.0)
    assert centers.shape[0] >= 8           # at least the 8 XY corners resolved
    assert loc_unit.max() == centers.shape[0]


# --- Z peaks -------------------------------------------------------------------
def test_detect_z_peaks_bimodal():
    rng = np.random.default_rng(3)
    z = np.concatenate([rng.normal(-12.5, 2, 2000), rng.normal(12.5, 2, 2000)])
    peaks = nw.detect_z_peaks(z)
    assert peaks.size == 2
    assert abs(peaks[0] - (-12.5)) < 3 and abs(peaks[1] - 12.5) < 3


def test_resolve_z_peaks_fallback():
    z = np.random.default_rng(4).normal(0, 1, 500)     # single mode
    p1, p2, fb = nw._resolve_z_peaks(z, (20, 40), 25.0)
    assert fb is True
    assert np.isclose(p2 - p1, 25.0)


# --- two-ring fit + alignment --------------------------------------------------
def _two_ring_points(radius=37.5, inter=25.0, n=400, seed=0):
    r = np.random.default_rng(seed)
    th = r.uniform(0, 2 * np.pi, n)
    lo = np.column_stack([radius * np.cos(th[:n // 2]), radius * np.sin(th[:n // 2]),
                          np.full(n // 2, -inter / 2)]) + r.normal(0, 2, (n // 2, 3))
    up = np.column_stack([radius * np.cos(th[n // 2:]), radius * np.sin(th[n // 2:]),
                          np.full(n - n // 2, inter / 2)]) + r.normal(0, 2, (n - n // 2, 3))
    return lo, up


def test_fit_two_ring_recovers_geometry():
    lo, up = _two_ring_points(radius=37.5, inter=25.0, seed=5)
    opt = dict(max_xy=20, max_z=8, tilt_max=15, r_bounds=(30, 45), d_bounds=(18, 35),
               rim=20, k=3, min_fit_pts=3, max_iter=4000, max_fev=12000)
    fit = nw.fit_two_ring(lo, up, np.zeros(3), opt)
    assert fit["ok"]
    assert abs(fit["radius"] - 37.5) < 6
    assert abs(fit["inter"] - 25.0) < 8


def test_align_one_npc_phase_aligns_8fold():
    rng = np.random.default_rng(6)
    # 8 corners at a known rotation; fit normal = +Z, center = 0
    rot = 0.7
    th = rot + 2 * np.pi * np.arange(8) / 8
    units = np.column_stack([37.5 * np.cos(th), 37.5 * np.sin(th), np.zeros(8)])
    fit = {"normal": np.array([0.0, 0, 1]), "center": np.zeros(3)}
    _locA, unitA, _ph = nw.align_one_npc(units, units, fit, symmetry=8)
    # after phase alignment a corner sits near angle 0 (mod 45°)
    ang = np.mod(np.degrees(np.arctan2(unitA[:, 1], unitA[:, 0])), 45.0)
    assert min(ang.min(), 45 - ang.max()) < 5.0


# --- end-to-end ----------------------------------------------------------------
def test_average_npc_wanlu_sharp_8fold_two_ring():
    rng = np.random.default_rng(7)
    rows, tid0 = [], 0
    for i in range(8):
        cx, cy, cz = rng.uniform(-2000, 2000), rng.uniform(-2000, 2000), rng.uniform(-40, 40)
        raw, tid0 = _raw6_npc(cx, cy, cz, i + 1, rot=rng.uniform(0, 2 * np.pi),
                              seed=i, tid0=tid0)
        rows.append(raw)
    raw6 = np.vstack(rows)
    res = nw.average_npc_wanlu(raw6, diameter_bounds=(65, 85), inter_ring_bounds=(20, 32),
                               expected_inter_ring=25)
    assert res["n_accepted"] >= 6
    A = res["average_loc"]
    rr = np.hypot(A[:, 0], A[:, 1])
    assert abs(rr.mean() - 37.5) < 6 and rr.std() < 8        # tight ring
    z = A[:, 2]
    assert z[z < 0].mean() < -5 and z[z > 0].mean() > 5      # two Z rings


def test_detect_npc_wanlu_finds_centers():
    rng = np.random.default_rng(8)
    locs, tids, tid0, truth = [], [], 0, []
    for i in range(6):
        cx, cy = rng.uniform(-1600, 1600, 2)
        truth.append((cx, cy))
        loc, tid, tid0 = _npc_traces(cx, cy, rng.uniform(-30, 30), rot=rng.uniform(0, 6),
                                     seed=i, tid0=tid0, locs_per_unit=25)
        locs.append(loc)
        tids.append(tid)
    loc = np.vstack(locs)
    tid = np.concatenate(tids)
    res = nw.detect_npc_wanlu(loc, tid, pixel_size=4, diameter=75, rim=20, min_units=3)
    assert res["raw6"].shape[1] == 6
    assert res["n_npc"] >= 3                                   # detects most NPCs
    truth_arr = np.array(truth)
    for cx, cy in res["centers"]:                             # precision: each hit is real
        assert np.min(np.hypot(truth_arr[:, 0] - cx, truth_arr[:, 1] - cy)) < 20


def test_z_scale_flattens_stored_z():
    rng = np.random.default_rng(9)
    locs, tids, tid0 = [], [], 0
    for i in range(4):
        cx, cy = rng.uniform(-1200, 1200, 2)
        loc, tid, tid0 = _npc_traces(cx, cy, rng.uniform(-30, 30), rot=rng.uniform(0, 6),
                                     seed=i, tid0=tid0, locs_per_unit=25)
        locs.append(loc)
        tids.append(tid)
    loc = np.vstack(locs)
    tid = np.concatenate(tids)
    full = nw.detect_npc_wanlu(loc, tid, pixel_size=4, diameter=75, rim=20, min_units=3)
    half = nw.detect_npc_wanlu(loc, tid, pixel_size=4, diameter=75, rim=20, min_units=3,
                               z_scale=0.5)
    assert full["n_npc"] >= 1 and half["n_npc"] >= 1
    # XY detection unaffected; stored Z spread roughly halved
    assert np.isclose(np.ptp(half["raw6"][:, 2]), 0.5 * np.ptp(full["raw6"][:, 2]), rtol=0.15)


def test_average_reports_progress_per_npc():
    rows, tid0 = [], 0
    rng = np.random.default_rng(11)
    for i in range(5):
        raw, tid0 = _raw6_npc(rng.uniform(-1500, 1500), rng.uniform(-1500, 1500),
                              rng.uniform(-30, 30), i + 1, rot=rng.uniform(0, 6),
                              seed=i, tid0=tid0)
        rows.append(raw)
    calls = []
    nw.average_npc_wanlu(np.vstack(rows), diameter_bounds=(65, 85),
                         inter_ring_bounds=(20, 32), expected_inter_ring=25,
                         progress=lambda d, t: calls.append((d, t)))
    assert len(calls) == 5                       # one per NPC
    assert calls[-1] == (5, 5)                   # monotone to total


# --- goodness-of-fit metrics + gate -------------------------------------------
def test_fit_metrics_on_good_fit():
    lo, up = _two_ring_points(radius=37.5, inter=25.0, seed=2)
    opt = dict(max_xy=20, max_z=8, tilt_max=15, r_bounds=(30, 45), d_bounds=(18, 35),
               rim=20, k=3, min_fit_pts=3, max_iter=4000, max_fev=12000)
    fit = nw.fit_two_ring(lo, up, np.zeros(3), opt)
    m = nw._fit_metrics(fit, lo, up)
    assert np.isfinite([m["radius"], m["rim"], m["inter"], m["tilt"],
                        m["gof"], m["rmse"], m["nrmse"]]).all()
    assert 0.5 < m["gof"] <= 1.0                 # a real two-ring explains most spread
    assert m["rmse"] > 0
    assert np.isclose(m["nrmse"], m["rmse"] / m["radius"])


def test_average_table_and_gof_gate():
    rng = np.random.default_rng(7)
    rows, tid0 = [], 0
    for i in range(8):
        cx, cy, cz = rng.uniform(-2000, 2000), rng.uniform(-2000, 2000), rng.uniform(-40, 40)
        raw, tid0 = _raw6_npc(cx, cy, cz, i + 1, rot=rng.uniform(0, 2 * np.pi),
                              seed=i, tid0=tid0)
        rows.append(raw)
    raw6 = np.vstack(rows)
    kw = dict(diameter_bounds=(65, 85), inter_ring_bounds=(20, 32), expected_inter_ring=25)

    permissive = nw.average_npc_wanlu(raw6, min_gof=-10.0, **kw)
    table = permissive["table"]
    assert len(table) == permissive["n_npc"]
    for r in table:                              # every reported column present
        for k in ("npc_id", "n_locs", "radius", "rim", "inter", "tilt",
                  "gof", "rmse", "nrmse", "accepted"):
            assert k in r
    assert all(r["accepted"] for r in table)     # nothing gated when min_gof=-10
    assert permissive["n_accepted"] == sum(r["accepted"] for r in table)

    strict = nw.average_npc_wanlu(raw6, min_gof=0.999, **kw)   # near-perfect required
    assert strict["n_accepted"] <= permissive["n_accepted"]
    assert strict["n_accepted"] == sum(r["accepted"] for r in strict["table"])


def test_average_aligned_parallel_for_subset_repool():
    """Result carries per-NPC aligned points (parallel to table/fits) so a user
    sub-selection can re-pool any subset without re-fitting."""
    rng = np.random.default_rng(11)
    rows, tid0 = [], 0
    for i in range(6):
        cx, cy, cz = rng.uniform(-2000, 2000), rng.uniform(-2000, 2000), rng.uniform(-40, 40)
        raw, tid0 = _raw6_npc(cx, cy, cz, i + 1, rot=rng.uniform(0, 2 * np.pi), seed=i, tid0=tid0)
        rows.append(raw)
    res = nw.average_npc_wanlu(np.vstack(rows), min_gof=-10.0, diameter_bounds=(65, 85),
                               inter_ring_bounds=(20, 32), expected_inter_ring=25)
    aligned = res["aligned"]
    assert len(aligned) == len(res["table"]) == len(res["fits"])
    ok = [a for a in aligned if a is not None]
    assert ok and all(np.asarray(a).shape[1] == 3 for a in ok)
    # re-pooling any subset is a plain vstack of the cached aligned arrays
    subset = np.vstack([np.asarray(aligned[i])[:, :3] for i in (0, 2) if aligned[i] is not None])
    assert subset.shape[0] > 0
    # stacking every accepted NPC's aligned points reproduces the pooled average count
    acc = [np.asarray(aligned[j])[:, :3] for j, r in enumerate(res["table"])
           if r["accepted"] and aligned[j] is not None]
    assert np.vstack(acc).shape[0] == res["average_loc"].shape[0]


def test_empty_inputs_safe():
    assert nw.detect_npc_wanlu(np.empty((0, 3)), np.empty(0))["n_npc"] == 0
    assert nw.average_npc_wanlu(np.empty((0, 6)))["n_accepted"] == 0


def test_ring_points_two_circles_at_fitted_radius():
    fit = {"ok": True, "u": np.array([1.0, 0.0, 0.0]), "v": np.array([0.0, 1.0, 0.0]),
           "radius": 37.5, "lower": np.array([0.0, 0.0, -15.0]),
           "upper": np.array([0.0, 0.0, 15.0])}
    rings = nw.ring_points(fit, n=64)
    assert len(rings) == 2
    for centre, ring in zip((fit["lower"], fit["upper"]), rings):
        assert ring.shape == (64, 3)
        assert np.allclose(np.linalg.norm(ring - centre, axis=1), 37.5, atol=1e-6)
    assert nw.ring_points({"ok": False}) == []
    assert nw.ring_points(None) == []
