"""Photon counts and rates are used as what they are.

``eco``/``ecc`` are photon counts and ``efo``/``efc``/``fbg`` rates in Hz. A
localization's photon count is ``eco`` summed over its final-scale iterations.
The pattern diameter follows Abberior's definition
``L = patGeoFactor x 360 nm x wavelength / 642 nm``. The vendor documents the
counts as background-corrected while recorded files behave as total counts, so
the user-facing text asserts neither.
"""

from __future__ import annotations

import numpy as np
import pytest

from minflux_viewer.core import loader as L
from minflux_viewer.core.attributes import RAW_ATTRIBUTE_DESCRIPTIONS
from minflux_viewer.core.mfx_sequence import (
    PATTERN_SCALE_NM,
    localization_photon_counts,
    pattern_geometry,
    wavelength_scale,
)
from minflux_viewer.core.save import build_snapshot_table, write_picasso_hdf5
from minflux_viewer.core.simulate import SIM_DWELL_S, simulate_attributes
from minflux_viewer.msr.descriptions import FIELD_DESCRIPTIONS

COUNT_FIELDS = ("eco", "ecc", "efo", "efc", "fbg")


# --- descriptions ---------------------------------------------------------------
@pytest.mark.parametrize("name", COUNT_FIELDS)
def test_descriptions_do_not_assert_a_background_convention(name):
    for table in (RAW_ATTRIBUTE_DESCRIPTIONS, FIELD_DESCRIPTIONS):
        text = table[name]
        assert "background-corrected" not in text
        assert "EOD pattern" not in text


@pytest.mark.parametrize("name", COUNT_FIELDS)
def test_both_description_tables_agree(name):
    assert RAW_ATTRIBUTE_DESCRIPTIONS[name] == FIELD_DESCRIPTIONS[name]


def test_rates_are_described_as_rates_and_counts_as_counts():
    for name in ("efo", "efc", "fbg"):
        assert "(Hz)" in RAW_ATTRIBUTE_DESCRIPTIONS[name], name
    for name in ("eco", "ecc"):
        assert "photons detected" in RAW_ATTRIBUTE_DESCRIPTIONS[name], name


# --- pattern diameter -----------------------------------------------------------
def _one_iteration(**extra):
    return {"Itr": [{"patGeoFactor": 0.11, "ccrLimit": -1,
                     "Mode": {"pattern": "hexagon"}, **extra}]}


def test_the_reference_wavelength_keeps_the_standard_diameter():
    # Imspector records the reference as 6.420000000000001e-07.
    for wavelength in (None, 6.42e-7, 6.420000000000001e-07):
        extra = {} if wavelength is None else {"wavelength": wavelength}
        assert pattern_geometry(_one_iteration(**extra), 0).L_nm == 0.11 * PATTERN_SCALE_NM


def test_a_recorded_wavelength_scales_the_diameter():
    geo = pattern_geometry(_one_iteration(wavelength=6.8e-7), 0)
    assert geo.L_nm == pytest.approx(0.11 * 360.0 * 680.0 / 642.0)


@pytest.mark.parametrize("value", [None, True, "642e-9", 0.0, -6.42e-7, float("nan")])
def test_an_unusable_wavelength_is_the_reference(value):
    assert wavelength_scale(value) == 1.0


# --- localization photon count and the Picasso export --------------------------
SEQUENCE_3D = {"bgcSense": True, "ctrDwellFactor": 1, "Itr": [
    {"patGeoFactor": 0.8, "ccrLimit": -1, "Mode": {"pattern": "hexagon"}},
    {"patGeoFactor": 0.21, "ccrLimit": 0.8, "Mode": {"pattern": "hexagon"}},
    {"patGeoFactor": 0.11, "ccrLimit": -1, "Mode": {"pattern": "hexagon"}},
    {"patGeoFactor": 0.11, "ccrLimit": -1, "Mode": {"pattern": "zline2"}},
]}


def _ladder_dataset(n_loc=12, *, with_eco=True, seed=1):
    """3-D localizations, each a ladder of iterations 0..3 (final = 3)."""
    fields = [("vld", np.bool_), ("fnl", np.bool_), ("tid", np.int32),
              ("tim", np.float64), ("itr", np.int32), ("efo", np.float64),
              ("fbg", np.float64), ("loc", np.float64, (3,))]
    if with_eco:
        fields.append(("eco", np.int32))
    rng = np.random.default_rng(seed)
    a = np.zeros(n_loc * 4, np.dtype(fields))
    a["vld"] = True
    a["itr"] = np.tile(np.arange(4), n_loc)
    a["fnl"] = a["itr"] == 3
    a["tid"] = np.repeat(np.arange(n_loc) // 3, 4)
    a["tim"] = np.repeat(np.linspace(0.0, 1.0, n_loc), 4) + a["itr"] * 1e-3
    a["efo"] = 5.0e4
    a["fbg"] = 7.0e3
    loc = np.column_stack([rng.uniform(0, 1e-6, n_loc), rng.uniform(0, 1e-6, n_loc),
                           rng.uniform(-2e-7, 2e-7, n_loc)])
    a["loc"] = np.repeat(loc, 4, axis=0)
    if with_eco:
        k = np.repeat(np.arange(n_loc), 4)
        a["eco"] = 100 * (a["itr"] + 1) + k        # distinct per iteration and localization
    ds = L.load_from_mfx_array(a, name="ladder.mat", folder="/tmp")
    ds.metadata["mfx_sequence"] = SEQUENCE_3D
    return ds, a


def _expected_photons(a):
    rows = np.isin(a["itr"], (2, 3))
    loc = np.repeat(np.arange(a.size // 4), 4)
    return np.bincount(loc[rows], weights=a["eco"][rows].astype(float))


def test_localization_photons_sum_the_final_lateral_and_axial_counts():
    ds, a = _ladder_dataset()
    counts = localization_photon_counts(ds)
    np.testing.assert_array_equal(counts, _expected_photons(a))


def test_localization_photons_need_eco():
    ds, _a = _ladder_dataset(with_eco=False)
    assert localization_photon_counts(ds) is None


def test_extra_snapshot_columns_are_filtered_with_the_rest():
    ds, a = _ladder_dataset()
    photons = localization_photon_counts(ds)
    xn = np.asarray(L.mfx_get(ds, "xnm", itr="last", vld_only=True), float)
    lo = float(np.median(xn))
    ds.state["filter_specs"] = [{"attribute": "xnm", "mode": "per loc",
                                 "lo": lo, "hi": float(xn.max()),
                                 "lo_inc": True, "hi_inc": True}]
    keep = xn >= lo
    cols, dropped = build_snapshot_table(ds, filter_mode="apply",
                                         extra_columns={"photons_x": photons,
                                                        "wrong_length": photons[:-1]})
    np.testing.assert_array_equal(cols["photons_x"], photons[keep])
    assert "wrong_length" not in cols
    assert dropped == int((~keep).sum())


def _picasso(ds, tmp_path):
    import h5py

    h5_path, yaml_path = write_picasso_hdf5(ds, tmp_path / "picasso.hdf5")
    with h5py.File(h5_path, "r") as h5:
        locs = h5["locs"][:]
    return locs, yaml_path.read_text(encoding="utf-8")


def test_picasso_photons_are_the_localization_photon_count(tmp_path):
    ds, a = _ladder_dataset()
    locs, yaml_text = _picasso(ds, tmp_path)
    np.testing.assert_allclose(locs["photons"], _expected_photons(a))
    assert 'MINFLUX Viewer photons: "eco summed over the final-scale iterations"' in yaml_text


def test_picasso_never_writes_rates_as_photon_counts(tmp_path):
    # Without eco, efo (Hz) must not become photons, nor fbg (Hz) the background.
    ds, _a = _ladder_dataset(with_eco=False)
    locs, yaml_text = _picasso(ds, tmp_path)
    assert np.all(locs["photons"] == 1.0)
    assert np.all(locs["bg"] == 0.0)
    assert 'MINFLUX Viewer photons: "not available (1 per localization)"' in yaml_text
    assert 'MINFLUX Viewer bg: "not available (0)"' in yaml_text


# --- simulated attributes -------------------------------------------------------
def test_simulated_counts_and_rates_are_consistent():
    a = simulate_attributes(20000, np.random.default_rng(5))
    np.testing.assert_allclose(a["efo"], a["eco"] / SIM_DWELL_S)
    assert np.all(a["eco"] >= 1)
    # The centre sees a fraction cfr of the outer rate over the same dwell.
    assert a["ecc"].sum() < a["eco"].sum()
    ratio = a["ecc"].sum() / a["eco"].sum()
    assert ratio == pytest.approx(np.average(a["cfr"], weights=a["eco"]), rel=0.05)
    # fbg is a background rate of a few kHz, not a handful of Hz.
    assert 1.0e3 < float(np.median(a["fbg"])) < 2.0e4
