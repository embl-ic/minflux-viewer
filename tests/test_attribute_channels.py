"""
Attribute-agnostic channel model: window construction, assignment, and the
photon (eco) weighted DCR pooling kernel.
"""

from __future__ import annotations

import numpy as np

from minflux_viewer.analysis.attribute_channels import (
    Channel,
    assign_traces,
    assign_values,
    channels_from_boundaries,
    channels_from_fit,
    eco_weighted_group_mean,
    place_evenly,
    reconstruct_from_channels,
)
from minflux_viewer.analysis.distribution_fit import fit_mixture


# --------------------------------------------------------------------------- construction
def test_place_evenly_partitions_range():
    ch = place_evenly(0.0, 90.0, 3, base_name="ds", attribute="tim",
                      luts=["Red", "Green", "Blue"])
    assert [c.lo for c in ch] == [0.0, 30.0, 60.0]
    assert [c.hi for c in ch] == [30.0, 60.0, 90.0]
    assert [c.lut for c in ch] == ["Red", "Green", "Blue"]
    assert ch[0].name == "ds [tim 1]"


def test_channels_from_boundaries_adjacent_and_monotone():
    ch = channels_from_boundaries([30.0, 60.0], 0.0, 90.0)
    assert len(ch) == 3
    assert ch[0].lo == 0.0 and ch[0].hi == 30.0
    assert ch[2].lo == 60.0 and ch[2].hi == 90.0


def test_channels_from_fit_splits_at_boundary():
    rng = np.random.default_rng(0)
    x = np.concatenate([rng.normal(0.3, 0.05, 4000), rng.normal(0.7, 0.06, 3000)])
    res = fit_mixture(x, "gaussian", 2)
    ch = channels_from_fit(res, data_range=(x.min(), x.max()), base_name="d", attribute="dcr")
    assert len(ch) == 2
    b = res.boundaries()[0]
    assert abs(ch[0].hi - b) < 1e-9 and abs(ch[1].lo - b) < 1e-9
    # outer edges cover the full data range
    assert ch[0].lo <= x.min() and ch[1].hi >= x.max()


# --------------------------------------------------------------------------- assignment
def test_assign_values_windows_and_out_of_range():
    ch = place_evenly(0.0, 90.0, 3)
    v = np.array([5.0, 45.0, 85.0, 200.0, np.nan])
    lab = assign_values(v, ch)
    assert lab[0] == 0 and lab[1] == 1 and lab[2] == 2
    assert lab[3] == -1 and lab[4] == -1               # out of range, NaN


def test_assign_traces_mean_keeps_trace_together():
    ch = place_evenly(0.0, 1.0, 2)                     # [0,0.5], [0.5,1]
    vals = np.array([0.30, 0.35, 0.28, 0.80,   0.70, 0.72, 0.90])
    tid = np.array([10, 10, 10, 10,            20, 20, 20])
    lab = assign_traces(vals, tid, ch, mode="mean")
    assert set(lab[:4]) == {0}                          # trace 10 mean ~0.43 → ch0
    assert set(lab[4:]) == {1}                          # trace 20 → ch1
    assert lab[3] == 0                                  # noisy 0.80 follows its trace


def test_assign_traces_majority_vote_and_confidence():
    ch = place_evenly(0.0, 1.0, 2)
    # trace 10: 3 in ch0 + 1 in ch1 (frac 0.75); trace 20: 2/2 tie.
    vals = np.array([0.2, 0.3, 0.4, 0.8,   0.2, 0.3, 0.8, 0.9])
    tid = np.array([10, 10, 10, 10,        20, 20, 20, 20])
    lab_lo = assign_traces(vals, tid, ch, mode="majority", min_confidence=0.5)
    assert set(lab_lo[:4]) == {0}
    assert set(lab_lo[4:]) == {-1}                      # tie → unassigned
    lab_hi = assign_traces(vals, tid, ch, mode="majority", min_confidence=0.8)
    assert set(lab_hi[:4]) == {-1}                      # 0.75 < 0.80


# --------------------------------------------------------------------------- eco weighting
def test_eco_weighted_group_mean():
    # group 0: values 0,1 with weights 1,3 → 0.75 ; group 1: value 1 weight 5 → 1.0
    values = np.array([0.0, 1.0, 1.0])
    weights = np.array([1.0, 3.0, 5.0])
    gid = np.array([0, 0, 1])
    out = eco_weighted_group_mean(values, weights, gid, 3)
    assert np.isclose(out[0], 0.75)
    assert np.isclose(out[1], 1.0)
    assert np.isnan(out[2])                             # empty group → NaN


def test_eco_weighted_group_mean_ignores_zero_and_nan_weights():
    values = np.array([0.0, 1.0, 5.0, np.nan])
    weights = np.array([0.0, 2.0, np.nan, 4.0])
    gid = np.array([0, 0, 0, 0])
    out = eco_weighted_group_mean(values, weights, gid, 1)
    assert np.isclose(out[0], 1.0)                      # only value=1 weight=2 counts


# --------------------------------------------------------------------------- revert
def test_reconstruct_from_channels_concatenates():
    """Revert fallback: channel subsets recombine into one localization set,
    preserving coordinates, tids and shared attributes."""
    from minflux_viewer.core.dataset import build_localization_dataset

    d1 = build_localization_dataset(
        name="c1", x_nm=np.array([1.0, 2.0]), y_nm=np.array([1.0, 2.0]),
        tid=np.array([1, 1]),
        attrs={"efo": np.array([10.0, 20.0]), "dcr": np.array([0.3, 0.3])})
    d2 = build_localization_dataset(
        name="c2", x_nm=np.array([3.0, 4.0, 5.0]), y_nm=np.array([3.0, 4.0, 5.0]),
        tid=np.array([2, 2, 3]),
        attrs={"efo": np.array([30.0, 40.0, 50.0]), "dcr": np.array([0.7, 0.7, 0.7])})

    rec = reconstruct_from_channels([d1, d2])
    assert rec["x_nm"].size == 5
    assert np.allclose(np.sort(rec["x_nm"]), [1, 2, 3, 4, 5])
    assert np.allclose(np.sort(rec["tid"]), [1, 1, 2, 2, 3])       # tids preserved, no remap
    assert set(rec["attrs"]).issuperset({"efo", "dcr"})
    assert np.allclose(np.sort(rec["attrs"]["efo"]), [10, 20, 30, 40, 50])
