"""Canonical NPC geometry-fit averager (analysis/npc_geomfit.py).

Pure NumPy/SciPy — a direct maximum-likelihood fit of an 8-corner two-ring model
to a particle's raw 3-D points (no sub-unit detection / ring pre-assignment).
"""

from __future__ import annotations

import numpy as np

from minflux_viewer.analysis import npc_geomfit as gf


def _make_npc(*, diameter=100.0, inter=40.0, rim=6.0, sym=8, per=40,
              tilt_deg=0.0, az=0.0, phase=0.0, center=(0.0, 0.0, 0.0), seed=0):
    rng = np.random.default_rng(seed)
    corners = gf.npc_corner_centers(diameter / 2.0, inter, phase, sym)
    pts = np.vstack([rng.normal(c, rim, (per, 3)) for c in corners])
    if tilt_deg:
        tr = np.deg2rad(tilt_deg)
        n = np.array([np.sin(tr) * np.cos(az), np.sin(tr) * np.sin(az), np.cos(tr)])
        pts = (gf.rotation_a_to_b([0, 0, 1], n) @ pts.T).T
    return pts + np.asarray(center, float)


def test_corner_centers_geometry():
    c = gf.npc_corner_centers(50.0, 40.0, 0.0, 8)
    assert c.shape == (16, 3)
    assert np.allclose(np.hypot(c[:, 0], c[:, 1]), 50.0)
    assert np.allclose(np.abs(c[:, 2]), 20.0)                 # two planes at ±inter/2
    single = gf.npc_corner_centers(50.0, 0.0, 0.0, 8)
    assert single.shape == (8, 3) and np.allclose(single[:, 2], 0.0)


def test_fit_recovers_diameter_inter_rim():
    fit = gf.fit_npc_model(_make_npc(diameter=100, inter=40, rim=6, seed=1))
    assert fit["ok"]
    assert abs(fit["diameter"] - 100) < 10
    assert abs(fit["inter"] - 40) < 10
    assert abs(fit["rim"] - 6) < 4
    assert fit["gof"] > 0.6


def test_fit_recovers_tilt():
    fit = gf.fit_npc_model(_make_npc(diameter=110, inter=50, rim=5, tilt_deg=25, az=0.7, seed=2))
    assert fit["ok"]
    assert abs(fit["tilt_deg"] - 25) < 8
    assert abs(fit["diameter"] - 110) < 10


def test_single_ring_when_inter_zero():
    fit = gf.fit_npc_model(_make_npc(diameter=90, inter=0.0, rim=6, seed=3),
                           inter_bounds=(0.0, 80.0))
    assert fit["ok"]
    assert fit["inter"] < 15                                  # collapses toward one ring
    assert abs(fit["diameter"] - 90) < 10


def test_align_to_canonical_flattens_and_scales():
    P = _make_npc(diameter=100, inter=40, rim=4, tilt_deg=20, az=1.0, phase=0.3, seed=4)
    fit = gf.fit_npc_model(P)
    A = gf.align_to_canonical(P, fit)
    assert abs(np.median(np.abs(A[:, 2])) - 20) < 8           # two rings near ±20
    assert abs(np.median(np.hypot(A[:, 0], A[:, 1])) - 50) < 8  # radius ~50


def test_average_geomfit_table_and_aligned_parallel():
    rng = np.random.default_rng(5)
    parts = [_make_npc(diameter=100 + rng.normal(0, 3), inter=40, rim=5,
                       tilt_deg=rng.uniform(0, 20), az=rng.uniform(0, 6.28),
                       phase=rng.uniform(0, 0.78), seed=i) for i in range(8)]
    res = gf.average_npc_geomfit(parts, min_gof=-10.0)
    assert res["n_particles"] == 8
    assert len(res["table"]) == len(res["fits"]) == len(res["aligned"]) == 8
    for r in res["table"]:
        for k in ("particle", "n_locs", "diameter", "inter", "rim", "tilt",
                  "phase", "gof", "rmse", "accepted"):
            assert k in r
    diams = [r["diameter"] for r in res["table"] if np.isfinite(r["diameter"])]
    assert diams and abs(np.median(diams) - 100) < 10         # recovers the ~100 nm diameter
    assert res["average_loc"].shape[0] > 0
    # subset re-pool is a plain vstack of the cached aligned arrays
    sub = np.vstack([np.asarray(res["aligned"][i])[:, :3] for i in (0, 2, 4)
                     if res["aligned"][i] is not None])
    assert sub.shape[0] > 0


def test_subunit_init_detects_corners_and_recovers():
    # each corner = several repeated blinks (traces), each a tight cluster of
    # correlated locs at ~localization precision — the realistic MINFLUX case.
    rng = np.random.default_rng(1)
    corners = gf.npc_corner_centers(105 / 2.0, 45.0, 0.0, 8)
    pts, tid, k = [], [], 0
    for c in corners:
        for _ in range(4):
            pts.append(rng.normal(c, 1.5, (12, 3)))       # a blink at the corner
            tid.append(np.full(12, k)); k += 1
    P, t = np.vstack(pts), np.concatenate(tid)

    units = gf._subunit_centers(P, t)
    assert units is not None and units.shape[0] >= 6      # LoG sub-units detected
    fit = gf.fit_npc_model(P, init_points=P, init_tid=t, subunit_init=True)
    assert fit["ok"] and abs(fit["diameter"] - 105) < 10 and abs(fit["inter"] - 45) < 12
    # fallback: no tid → no sub-units, moment init still fits
    assert gf._subunit_centers(P, None) is None
    assert gf.fit_npc_model(P, subunit_init=True, init_tid=None)["ok"]


def test_too_few_points_safe():
    assert gf.fit_npc_model(np.zeros((5, 3)))["ok"] is False
    assert gf.average_npc_geomfit([])["n_accepted"] == 0
