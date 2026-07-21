"""Tests for analysis.aggregation and core.mfx_sequence (pure, no Qt)."""

from __future__ import annotations

import os
from collections import defaultdict

import numpy as np
import pytest

from minflux_viewer.analysis.aggregation import (
    aggregate_localizations,
    event_ids,
    localization_photons,
)
from minflux_viewer.core import mfx_sequence as seq


# ---------------------------------------------------------------------------
# Sequence detection
# ---------------------------------------------------------------------------

def _abberior_3d_sequence():
    """A minimal stand-in for the real Abberior 3-D sequence (10 iterations)."""
    geo = [0.8, 0.8, 0.8, 0.8, 0.42, 0.42, 0.21, 0.21, 0.11, 0.11]
    patt = ["hexagon", "zline", "square", "zline2", "square", "zline2",
            "square", "zline2", "square", "zline2"]
    mode = ["pt3d", "mxax", "mxl3", "mxa3", "mxl3", "mxa3", "mxl3", "mxa3", "mxl3", "mxa3"]
    return {"Itr": [{"patGeoFactor": g, "Mode": {"pattern": p, "id": m, "dim": 3}}
                    for g, p, m in zip(geo, patt, mode)]}


def test_detect_from_sequence_3d_final_scale():
    info = seq.detect_photon_iterations(sequence=_abberior_3d_sequence(), n_itr=10, dims=3)
    assert info.source == "sequence"
    assert info.photon_iters == [8, 9]      # the two smallest-patGeoFactor iterations
    assert info.lateral_iter == 8           # final "square"/mxl3
    assert info.axial_iter == 9             # final "zline2"/mxa3


def test_fallback_3d_and_2d():
    f3 = seq.fallback_photon_iters(10, dims=3)
    assert f3.photon_iters == [8, 9] and f3.lateral_iter == 8 and f3.axial_iter == 9
    f2 = seq.fallback_photon_iters(6, dims=2)
    assert f2.photon_iters == [5] and f2.lateral_iter == 5 and f2.axial_iter is None


def test_detect_falls_back_without_sequence():
    info = seq.detect_photon_iterations(sequence=None, n_itr=10, dims=3)
    assert info.source == "fallback" and info.photon_iters == [8, 9]


# ---------------------------------------------------------------------------
# Aggregation core — synthetic
# ---------------------------------------------------------------------------

def _make_raw(events, *, n_itr=10, photon_iters=(8, 9)):
    """Build a raw-iteration dict from a list of events.

    Each event is ``(tid, tim, [(x,y,z)] position (nm→m below), eco_per_photon_iter)``.
    We emit two rows per event (itr 8 and itr 9), each carrying half the photon
    count, ending in a fnl row (itr 9). All rows valid.
    """
    itr, fnl, vld, eco, tid, tim = [], [], [], [], [], []
    loc = []
    for (t, tm, pos, photons) in events:
        # two photon-bearing rows (itr 8, itr 9); split photons evenly
        for k, it in enumerate((8, 9)):
            itr.append(it)
            fnl.append(1 if it == 9 else 0)
            vld.append(1)
            eco.append(photons / 2.0)
            tid.append(t)
            tim.append(tm)
            loc.append(pos)
    return {
        "itr": np.array(itr), "fnl": np.array(fnl), "vld": np.array(vld),
        "eco": np.array(eco, float), "tid": np.array(tid), "tim": np.array(tim, float),
        "loc": np.array(loc, float),
    }


def test_photon_sum_over_photon_iters():
    raw = _make_raw([(1, 0.0, (0, 0, 0), 1000.0)])
    ev_ph, eid = localization_photons(raw["itr"], raw["fnl"], raw["eco"], [8, 9])
    assert np.isclose(ev_ph[0], 1000.0)              # 500 + 500 over itr 8 & 9
    # Only itr 9 -> half the photons.
    ev_ph9, _ = localization_photons(raw["itr"], raw["fnl"], raw["eco"], [9])
    assert np.isclose(ev_ph9[0], 500.0)


def test_photon_threshold_binning_and_flags():
    # One trace, 4 events of 1000 photons each, threshold 2500.
    events = [(1, float(i), (i * 10.0, 0, 0), 1000.0) for i in range(4)]
    raw = _make_raw(events)
    res = aggregate_localizations(raw, photon_threshold=2500, photon_iters=[8, 9],
                                  carry_attrs=False)
    # e0+e1+e2 = 3000 >= 2500 -> chunk1; e3 = 1000 trailing -> chunk2.
    assert len(res["tid"]) == 2
    assert np.isclose(res["eco"][0], 3000.0)
    assert np.isclose(res["eco"][1], 1000.0)
    assert list(res["bot"]) == [1, 0]
    assert list(res["eot"]) == [0, 1]
    assert list(res["n"]) == [3, 1]
    # position = mean of member positions (x = mean(0,10,20)=10 for chunk1).
    assert np.isclose(res["loc"][0, 0], 10.0)
    assert np.isclose(res["loc"][1, 0], 30.0)


def test_invalid_and_nonfinal_rows_excluded():
    raw = _make_raw([(1, 0.0, (0, 0, 0), 1000.0), (1, 1.0, (5, 0, 0), 1000.0)])
    # Mark the second event invalid (both its rows).
    raw["vld"][2:] = 0
    res = aggregate_localizations(raw, photon_threshold=100000, photon_iters=[8, 9],
                                  carry_attrs=False)
    assert len(res["tid"]) == 1           # only the valid event survives
    assert res["n"][0] == 1


def test_tolerates_1d_dcr_odd_length():
    """A 1-D per-row `dcr` (primary ratio) must not be treated as a 2-vector
    (regression: odd row counts raised 'cannot reshape array ... into shape (2)')."""
    events = [(1, float(i), (i * 5.0, 0, 0), 1000.0) for i in range(3)]
    raw = _make_raw(events)                      # 6 rows (odd-safe: use 2/event)
    raw["dcr"] = np.linspace(0.1, 0.9, raw["itr"].size)   # 1-D per row
    # Must not raise, and must not emit a bogus 2-channel dcr.
    res = aggregate_localizations(raw, photon_threshold=2500, photon_iters=[8, 9],
                                  carry_attrs=True)
    assert res["tid"].size >= 1
    assert "dcr" not in res or np.asarray(res["dcr"]).ndim <= 1


def test_dcr_two_channel_aggregated():
    """A genuine (rows, 2) dcr is carried as a per-point 2-channel mean."""
    events = [(1, 0.0, (0, 0, 0), 1000.0), (1, 1.0, (0, 0, 0), 1000.0)]
    raw = _make_raw(events)
    n = raw["itr"].size
    raw["dcr"] = np.column_stack([np.full(n, 0.3), np.full(n, 0.7)])
    res = aggregate_localizations(raw, photon_threshold=100000, photon_iters=[8, 9],
                                  carry_attrs=True)
    assert res["dcr"].shape == (1, 2)
    assert np.allclose(res["dcr"][0], [0.3, 0.7])


def test_two_traces_stay_separate():
    events = [(1, 0.0, (0, 0, 0), 6000.0), (2, 0.5, (100, 0, 0), 6000.0)]
    raw = _make_raw(events)
    res = aggregate_localizations(raw, photon_threshold=3000, photon_iters=[8, 9],
                                  carry_attrs=False)
    # Each trace's single 6000-photon event exceeds threshold -> one point each.
    assert sorted(res["tid"].tolist()) == [1, 2]


# ---------------------------------------------------------------------------
# Regression against the reference Imspector files (skipped if absent)
# ---------------------------------------------------------------------------

_REF_DIR = r"D:\Workspace\Users\Cigdem\2026\test_data_for_agg"
_RAW_MAT = os.path.join(_REF_DIR, "45pM.mat")
_AGG_MAT = os.path.join(_REF_DIR, "45pM_agg3000.mat")


@pytest.mark.skipif(not (os.path.exists(_RAW_MAT) and os.path.exists(_AGG_MAT)),
                    reason="reference Imspector aggregation files not present")
def test_reproduces_imspector_agg3000():
    sio = pytest.importorskip("scipy.io")

    def col(m, k):
        v = m[k]
        return v.ravel() if v.shape[0] == 1 else v

    m = sio.loadmat(_RAW_MAT)
    raw = {"itr": col(m, "itr"), "fnl": col(m, "fnl"), "vld": col(m, "vld"),
           "eco": col(m, "eco").astype(float), "tid": col(m, "tid"),
           "tim": col(m, "tim").astype(float), "loc": m["loc"]}
    n_itr = int(np.asarray(raw["itr"]).max()) + 1
    photon_iters = seq.fallback_photon_iters(n_itr, dims=3).photon_iters  # [8, 9]
    res = aggregate_localizations(raw, photon_threshold=3000,
                                  photon_iters=photon_iters, carry_attrs=False)

    agg = sio.loadmat(_AGG_MAT)
    a_tid = col(agg, "tid"); a_loc = agg["loc"]; a_eco = col(agg, "eco").astype(float)
    a_tim = col(agg, "tim"); a_bot = col(agg, "bot"); a_eot = col(agg, "eot")

    # Point count within 1 % of Imspector.
    assert abs(len(res["tid"]) - len(a_tid)) <= 0.01 * len(a_tid)

    def by(tids, tims):
        b = defaultdict(list)
        for i in range(len(tids)):
            b[tids[i]].append(i)
        for t in b:
            b[t] = sorted(b[t], key=lambda i: tims[i])
        return b

    A, R = by(a_tid, a_tim), by(res["tid"], res["tim"])
    dpos, ex, nt, bo, eo = [], 0, 0, 0, 0
    for t in R:
        aa, rr = A.get(t, []), R[t]
        for k in range(min(len(aa), len(rr))):
            ai, ri = aa[k], rr[k]
            nt += 1
            dpos.append(np.linalg.norm((res["loc"][ri] - a_loc[ai]) * 1e9))
            ex += abs(res["eco"][ri] - a_eco[ai]) < 0.5
            bo += res["bot"][ri] == a_bot[ai]
            eo += res["eot"][ri] == a_eot[ai]
    dpos = np.array(dpos)
    assert np.median(dpos) < 2.0                 # sub-2 nm median position
    assert ex / nt > 0.80                        # >80 % exact photon match
    assert bo / nt > 0.98 and eo / nt > 0.98     # begin/end flags
