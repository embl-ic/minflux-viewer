"""MINFLUX Cramér-Rao bound: the closed forms, the background model and the inputs.

The closed forms are Marin & Ries, arXiv:2410.12427 (Eq. 28/29 axial, Eq. 44/45
lateral). Their background factor is written for one pattern each (triangle +
centre, two points + centre, equal dwell per exposure); ours is generalised to
the recorded centre dwell, so it is checked here both against those equations
and against a brute-force multinomial Fisher information for the patterns an
Abberior instrument actually uses (hexagon, square, zline2, with or without a
centre exposure).
"""

from __future__ import annotations

import copy
from pathlib import Path

import numpy as np
import pytest

from minflux_viewer.analysis import method_text as mt
from minflux_viewer.analysis.localization_precision import (
    BG_CENTRE,
    BG_CENTRE_EARLIER,
    BG_FBG,
    CRLB_DEFAULT_L_NM,
    CRLB_DEFAULT_SIGMA_Q_NM,
    crlb_from_inputs,
    crlb_inputs_for_dataset,
    crlb_inputs_from_arrays,
    crlb_log_line,
    crlb_precision,
)
from minflux_viewer.core import loader
from minflux_viewer.core.mfx_sequence import (
    PATTERN_SCALE_NM,
    PatternGeometry,
    assumed_pattern_geometry,
    background_sensing,
    centre_reference_iteration,
    detect_from_sequence,
    extract_sequence_from_zattrs,
    iteration_power,
    parse_sequence,
    pattern_geometry,
)

SAMPLE = Path(r"D:\Workspace\Microscopes\MINFLUX\sample data")
QUADRATIC = 1.0e9      # σ_q so large the donut-gradient correction is exactly 1


# ---------------------------------------------------------------------------
# Brute-force Fisher information for a quadratic minimum with background
# ---------------------------------------------------------------------------

def _fisher_sigma(positions, dwell, *, beta_over_g, n_photons, axis=0):
    """σ = 1/√I from the multinomial Fisher information, emitter at the origin.

    Exposure k lasts ``dwell[k]`` and has intensity g·|x − p_k|² + β, so its
    expected share of the photons is p_k ∝ dwell[k]·(|x − p_k|² + β/g).
    Derivatives by central differences.
    """
    positions = np.asarray(positions, dtype=float)
    dwell = np.asarray(dwell, dtype=float)

    def probs(shift):
        x = np.zeros(positions.shape[1])
        x[axis] = shift
        mu = dwell * (np.sum((x - positions) ** 2, axis=1) + beta_over_g)
        return mu / mu.sum()

    h = 1e-4
    p0 = probs(0.0)
    dp = (probs(h) - probs(-h)) / (2 * h)
    info = n_photons * np.sum(dp * dp / p0)
    return 1.0 / np.sqrt(info)


def _orbit(n_points, L):
    ang = 2 * np.pi * np.arange(n_points) / n_points
    return np.column_stack([0.5 * L * np.cos(ang), 0.5 * L * np.sin(ang)])


def _rates_for(L, rho, c, n_points=6, n=400.0):
    """eco/ecc/efo/efc for an emitter at the centre: signal a at each orbit
    position, background β = ρ·a everywhere, centre dwell = c × orbit dwell."""
    a = 1.0e5
    beta = rho * a
    t_orbit = 1.0e-3                          # whole orbit
    efo, efc = a + beta, beta
    total = efo * t_orbit + efc * c * t_orbit
    scale = n / total                         # so that eco + ecc = n
    return {"eco": np.array([efo * t_orbit * scale]),
            "ecc": np.array([efc * c * t_orbit * scale]),
            "efo": np.array([efo]), "efc": np.array([efc])}


# ---------------------------------------------------------------------------
# Closed forms
# ---------------------------------------------------------------------------

def test_ideal_lateral_is_eq45():
    eco = np.array([100.0, 400.0, 900.0])
    L, sq = 50.0, 250.0
    r = crlb_precision(eco, L_nm=L, sigma_q_nm=sq)
    # Eq. 45 exactly as printed: (σ_q/√N)·(√8·σ_q/L)·L²/(8σ_q² − L²)
    eq45 = (sq / np.sqrt(eco)) * (np.sqrt(8) * sq / L) * L ** 2 / (8 * sq ** 2 - L ** 2)
    assert np.allclose(r.per_loc_sigma_ideal, eq45)
    assert r.median_n == pytest.approx(400.0)


def test_ideal_axial_is_eq29():
    eco = np.array([100.0, 400.0])
    r = crlb_precision(eco, L_nm=50.0, L_z_nm=60.0)
    assert r.has_z
    assert np.allclose(r.per_loc_sigma_z_ideal, 60.0 / (4.0 * np.sqrt(eco)))


def test_triangle_plus_centre_reproduces_eq44():
    """Equal dwell on 3 orbit points + centre → c = 1/3 and the Eq. 44 factor,
    with SBR the total signal over the total background (their definition)."""
    L, sq, rho = 50.0, 250.0, 0.4
    d = _rates_for(L, rho, c=1.0 / 3.0, n_points=3)
    r = crlb_precision(d["eco"], ecc=d["ecc"], efo=d["efo"], efc=d["efc"],
                       L_nm=L, sigma_q_nm=sq)
    sbr = 3.0 / (4.0 * rho)                   # 3 signal exposures / 4 background ones
    n = d["eco"] + d["ecc"]
    eq44 = ((sq / np.sqrt(n)) * (np.sqrt(8) * sq / L) * L ** 2 / (8 * sq ** 2 - L ** 2)
            * np.sqrt((1 + 1 / sbr) * (1 + 3 / (4 * sbr))))
    assert r.centre_dwell_ratio == pytest.approx(1.0 / 3.0)
    assert r.median_sbr == pytest.approx(sbr)
    assert np.allclose(r.per_loc_sigma_bg, eq44)


def test_two_points_plus_centre_reproduces_eq28():
    rho, Lz = 0.7, 40.0
    d = _rates_for(Lz, rho, c=0.5)
    r = crlb_precision(d["eco"], ecc=d["ecc"], efo=d["efo"], efc=d["efc"], L_z_nm=Lz)
    sbr = 2.0 / (3.0 * rho)
    n = d["eco"] + d["ecc"]
    eq28 = Lz / (4 * np.sqrt(n)) * np.sqrt((1 + 1 / sbr) * (1 + 2 / (3 * sbr)))
    assert np.allclose(r.per_loc_sigma_z_bg, eq28)


@pytest.mark.parametrize("n_points", [3, 4, 6])
@pytest.mark.parametrize("c", [0.0, 0.16, 1.0])
@pytest.mark.parametrize("rho", [0.05, 0.5, 2.3])
def test_lateral_factor_matches_brute_force_fisher(n_points, c, rho):
    """The generalised factor is the exact Fisher bound for every orbit size
    and centre dwell — not only for Eq. 44's triangle + centre."""
    L, n = 40.0, 300.0
    positions = _orbit(n_points, L)
    dwell = np.full(n_points, 1.0 / n_points)
    if c > 0:
        positions = np.vstack([positions, [0.0, 0.0]])
        dwell = np.append(dwell, c)
    g_r2 = (0.5 * L) ** 2                    # signal at an orbit point, per unit g
    expected = _fisher_sigma(positions, dwell, beta_over_g=rho * g_r2, n_photons=n)

    d = _rates_for(L, rho, c=c, n_points=n_points, n=n)
    r = crlb_precision(d["eco"], ecc=d["ecc"] if c > 0 else None, efo=d["efo"],
                       efc=d["efc"], fbg=None, L_nm=L, sigma_q_nm=QUADRATIC,
                       centre_dwell_ratio=c)
    if c == 0:
        # no centre exposure: the background comes from fbg instead
        r = crlb_precision(d["eco"], efo=d["efo"], fbg=d["efc"], L_nm=L,
                           sigma_q_nm=QUADRATIC, centre_dwell_ratio=0.0)
    assert r.per_loc_sigma_bg[0] == pytest.approx(expected, rel=1e-5)


@pytest.mark.parametrize("c", [0.0, 0.5, 1.0])
def test_axial_factor_matches_brute_force_fisher(c):
    Lz, n, rho = 40.0, 150.0, 0.8
    positions = np.array([[-0.5 * Lz], [0.5 * Lz]])
    dwell = np.array([0.5, 0.5])
    if c > 0:
        positions = np.vstack([positions, [0.0]])
        dwell = np.append(dwell, c)
    expected = _fisher_sigma(positions, dwell, beta_over_g=rho * (0.5 * Lz) ** 2,
                             n_photons=n)
    d = _rates_for(Lz, rho, c=c, n=n)
    kw = (dict(ecc=d["ecc"], efc=d["efc"]) if c > 0 else dict(fbg=d["efc"]))
    r = crlb_precision(d["eco"], efo=d["efo"], L_z_nm=Lz, centre_dwell_ratio=c, **kw)
    assert r.per_loc_sigma_z_bg[0] == pytest.approx(expected, rel=1e-5)


# ---------------------------------------------------------------------------
# The background model
# ---------------------------------------------------------------------------

def test_centre_exposure_is_preferred_over_fbg():
    d = _rates_for(40.0, 1.0, c=1.0)
    fbg = np.array([0.1e5])                  # a much smaller instrument estimate
    r = crlb_precision(d["eco"], ecc=d["ecc"], efo=d["efo"], efc=d["efc"], fbg=fbg)
    assert r.per_loc_bg_source[0] == BG_CENTRE
    assert r.background_source == "centre exposure"
    only_fbg = crlb_precision(d["eco"], ecc=d["ecc"], efo=d["efo"], fbg=fbg)
    assert only_fbg.per_loc_bg_source[0] == BG_FBG
    assert only_fbg.primary_sigma < r.primary_sigma     # leakage counted too


def test_photons_are_counted_over_all_exposures():
    d = _rates_for(40.0, 0.5, c=1.0, n=500.0)
    r = crlb_precision(d["eco"], ecc=d["ecc"], efo=d["efo"], efc=d["efc"])
    assert r.median_n == pytest.approx(500.0)
    assert r.counts_centre


def test_centre_dwell_is_read_off_the_counts():
    d = _rates_for(40.0, 0.5, c=0.16)
    r = crlb_precision(d["eco"], ecc=d["ecc"], efo=d["efo"], efc=d["efc"])
    assert r.centre_dwell_ratio == pytest.approx(0.16)


def test_no_background_information_is_not_called_background_limited():
    eco = np.full(5, 200.0)
    r = crlb_precision(eco, efo=np.full(5, 1e5), fbg=np.zeros(5))   # fbg = 0: unknown
    assert not r.has_background
    assert r.background_source == "none"
    assert np.all(np.isnan(r.per_loc_sigma_bg))
    assert r.primary_sigma == pytest.approx(r.median_sigma_ideal)
    assert "no background information" in crlb_log_line(r)


def test_partial_background_coverage_is_reported():
    eco = np.full(4, 200.0)
    fbg = np.array([1e4, 1e4, 0.0, np.nan])
    r = crlb_precision(eco, efo=np.full(4, 1e5), fbg=fbg)
    assert r.has_background
    assert r.background_coverage == pytest.approx(0.5)
    assert "for 50% of localizations" in crlb_log_line(r)


def test_high_sbr_approaches_ideal():
    eco = np.full(4, 150.0)
    r = crlb_precision(eco, efo=np.full(4, 1e7), fbg=np.full(4, 1.0))
    assert np.allclose(r.per_loc_sigma_bg, r.per_loc_sigma_ideal, rtol=1e-5)


def test_sigma_q_too_small_is_invalid():
    r = crlb_precision(np.full(4, 200.0), L_nm=800.0, sigma_q_nm=250.0)  # L > 2√2·σ_q
    assert not np.isfinite(r.median_sigma_ideal)
    assert "2√2" in r.status


def test_no_valid_counts():
    r = crlb_precision(np.array([0.0, -1.0, np.nan]))
    assert r.n_locs == 0
    assert r.status


def test_axial_channel_has_its_own_photons_and_background():
    eco_xy = np.full(4, 200.0)
    eco_z = np.full(4, 100.0)
    r = crlb_precision(eco_xy, efo=np.full(4, 1e5), fbg=np.full(4, 5e4),
                       L_z_nm=40.0, eco_z=eco_z, efo_z=np.full(4, 1e5),
                       fbg_z=np.zeros(4))
    assert r.per_dimension_photons
    assert r.median_n_z == pytest.approx(100.0)
    assert r.has_background and not r.has_background_z
    assert np.allclose(r.per_loc_sigma_z_ideal, 40.0 / (4 * np.sqrt(100.0)))


# ---------------------------------------------------------------------------
# Pattern geometry from the acquisition sequence
# ---------------------------------------------------------------------------

def _sequence(final_lateral="hexagon", ccr=2, ctr=1, axial=False):
    itr = [{"patGeoFactor": 0.8, "ccrLimit": -1, "Mode": {"pattern": "hexagon", "id": "prbt"}},
           {"patGeoFactor": 0.11, "ccrLimit": ccr,
            "Mode": {"pattern": final_lateral, "id": "mxl3" if axial else "mflx"}}]
    if axial:
        itr.append({"patGeoFactor": 0.11, "ccrLimit": -1,
                    "Mode": {"pattern": "zline2", "id": "mxa3"}})
    return {"ctrDwellFactor": ctr, "Itr": itr}


def test_pattern_geometry_reads_l_and_centre():
    g = pattern_geometry(_sequence(ccr=2, ctr=1), 1)
    assert g.L_nm == pytest.approx(0.11 * PATTERN_SCALE_NM)
    assert g.pattern == "hexagon" and g.n_points == 6
    assert g.centre_probed and g.centre_dwell_ratio == 1.0
    assert g.from_sequence and "sequence" in g.source
    g = pattern_geometry(_sequence(ccr=-1), 1)
    assert not g.centre_probed and g.centre_dwell_ratio == 0.0
    assert pattern_geometry(_sequence(), 7) is None
    assert pattern_geometry(None, 0) is None


def test_zline_reports_smallest_extent_but_does_not_use_inner_pair_bound():
    seq = {"Itr": [{"patGeoFactor": [0.8, 4], "ccrLimit": -1, "Mode": {"pattern": "zline"}}]}
    assert pattern_geometry(seq, 0).L_nm == pytest.approx(0.8 * PATTERN_SCALE_NM)
    assert not pattern_geometry(seq, 0).supported


def test_assumed_geometry_says_so():
    g = assumed_pattern_geometry(4)
    assert g.L_nm == pytest.approx(CRLB_DEFAULT_L_NM)
    assert not g.from_sequence and "assumed" in g.source


# ---------------------------------------------------------------------------
# Inputs from a dataset: per-iteration gathering + the recorded sequence
# ---------------------------------------------------------------------------

def _dataset(n_loc=80, n_itr=2, seed=0, with_centre=True):
    """Flat m2410 data where every field differs between the two iterations, so
    reading the wrong iteration is visible."""
    rng = np.random.default_rng(seed)
    n = n_loc * n_itr
    dt = np.dtype([("vld", "?"), ("itr", "<i4"), ("tid", "<i4"), ("tim", "<f8"),
                   ("loc", "<f8", (3,)), ("eco", "<f8"), ("ecc", "<f8"),
                   ("efo", "<f8"), ("efc", "<f8"), ("fbg", "<f8")])
    mfx = np.zeros(n, dtype=dt)
    itr = np.tile(np.arange(n_itr), n_loc)
    mfx["itr"], mfx["vld"] = itr, True
    mfx["tid"] = np.repeat(np.arange(n_loc) // 4, n_itr)
    mfx["tim"] = np.repeat(np.linspace(0, 10, n_loc), n_itr)
    mfx["loc"][:, :2] = rng.normal(0, 3e-9, (n, 2)) + 1e-6
    final = itr == n_itr - 1
    mfx["eco"] = np.where(final, 300.0, 50.0)
    mfx["efo"] = np.where(final, 2.0e5, 5.0e4)
    mfx["efc"] = np.where(final & with_centre, 1.0e5, np.nan)
    mfx["ecc"] = np.where(final & with_centre, 150.0, 0.0)
    mfx["fbg"] = 0.0
    return loader.load_from_mfx_array(mfx, "synthetic")


def test_inputs_come_from_the_final_iteration_and_the_sequence():
    ds = _dataset()
    ds.metadata["mfx_sequence"] = _sequence(ccr=2, ctr=1)
    inputs = crlb_inputs_for_dataset(ds)
    assert np.allclose(inputs["xy"]["eco"], 300.0)       # final iteration, not 50
    assert inputs["geometry_xy"].from_sequence
    r = crlb_from_inputs(inputs)
    assert r.L_nm == pytest.approx(39.6) and "sequence" in r.L_source
    assert r.background_source == "centre exposure"
    assert r.median_n == pytest.approx(450.0)             # eco + ecc
    rho = 1.0e5 / (2.0e5 - 1.0e5)
    sq = CRLB_DEFAULT_SIGMA_Q_NM
    ideal = 39.6 / np.sqrt(8 * 450.0) / (1 - 39.6 ** 2 / (8 * sq ** 2))
    assert r.primary_sigma == pytest.approx(ideal * np.sqrt((1 + rho) * (1 + 2 * rho)))


def test_without_a_sequence_l_is_assumed_and_reported():
    ds = _dataset(with_centre=False)
    r = crlb_from_inputs(crlb_inputs_for_dataset(ds))
    assert r.L_nm == pytest.approx(CRLB_DEFAULT_L_NM)
    assert "assumed" in r.L_source
    assert not r.has_background                            # fbg 0, no centre


def test_a_user_l_is_reported_as_user_input():
    ds = _dataset()
    ds.metadata["mfx_sequence"] = _sequence()
    inputs = crlb_inputs_for_dataset(ds)
    assert crlb_from_inputs(inputs, L_nm=75.0).L_source == "user input"
    assert "sequence" in crlb_from_inputs(inputs, L_nm=39.6).L_source


def test_inputs_from_arrays_reuse_the_lateral_channel_for_z():
    inputs = crlb_inputs_from_arrays(np.full(3, 200.0), num_dim=3)
    r = crlb_from_inputs(inputs)
    assert r.has_z and not r.per_dimension_photons


# ---------------------------------------------------------------------------
# Log line ↔ method text
# ---------------------------------------------------------------------------

class _State:
    datasets = []


def _event(msg):
    return {"message": msg, "level": "INFO", "dataset_idx": None, "dataset_name": "ds1"}


@pytest.mark.parametrize("with_bg", [True, False])
def test_log_line_round_trips_through_method_text(with_bg):
    d = _rates_for(39.6, 1.2, c=1.0)
    kw = dict(ecc=d["ecc"], efo=d["efo"], efc=d["efc"]) if with_bg else {}
    r = crlb_precision(d["eco"], L_nm=39.6, L_source="from the acquisition sequence",
                       L_z_nm=39.6, **kw)
    line = crlb_log_line(r)
    txt = mt.generate_method_text(_State(), [_event(line)])
    assert f"{r.primary_sigma:.2f} nm" in txt
    assert "from the acquisition sequence" in txt
    if with_bg:
        assert "centre exposure" in txt
    else:
        assert "background-free" in txt.lower()
        assert "background-limited" not in txt


def test_a_spread_below_the_bound_is_not_called_an_excess():
    d = _rates_for(39.6, 2.0, c=1.0)
    r = crlb_precision(d["eco"], ecc=d["ecc"], efo=d["efo"], efc=d["efc"], L_nm=39.6)
    below = {"xy": {"measured_nm": 0.5 * r.primary_sigma, "bound_nm": r.primary_sigma,
                    "ratio": 0.5, "n_traces": 10, "n_locs": 50, "coverage": 1.0}}
    txt = mt.generate_method_text(_State(), [_event(crlb_log_line(r, below))])
    assert "unexplained" in txt
    assert "was identified" not in txt


def test_old_log_lines_still_generate_text():
    old = ("Localization precision (CRLB, Marin-Ries): median σ_xy = 5.20 nm "
           "(background-limited), 3.10 nm (ideal), σ_z = 12.34 nm (L_z = 100 nm), "
           "L = 50 nm, σ_q = 100 nm, median N = 200 photons.")
    txt = mt.generate_method_text(_State(), [_event(old)])
    assert "5.20 nm" in txt and "12.34 nm" in txt


# ---------------------------------------------------------------------------
# Real data (skipped when the sample folder is absent)
# ---------------------------------------------------------------------------

def _need(name):
    path = SAMPLE / name
    if not path.exists():
        pytest.skip(f"sample data not available: {path}")
    return path


def _prefs():
    from minflux_viewer.core.app_state import DEFAULT_PREFS
    return copy.deepcopy(DEFAULT_PREFS)


def test_real_2d_msr_takes_l_and_background_from_the_file():
    from minflux_viewer.msr.quick_open import load_viewer_msr
    ds = load_viewer_msr(_need("2_3C_measurement.msr"), prefs=_prefs())[0]
    r = crlb_from_inputs(crlb_inputs_for_dataset(ds))
    assert r.L_nm == pytest.approx(39.6) and "sequence" in r.L_source
    assert r.pattern == "hexagon"
    assert r.background_source == "centre exposure"       # fbg is 0 in this file
    assert r.counts_centre and r.centre_dwell_ratio == pytest.approx(1.0)
    # heavy zero leakage (cfr ≈ 0.7): the bound is several times the ideal one
    assert r.primary_sigma > 3 * r.median_sigma_ideal


def test_real_mat_without_sequence_flags_its_contradictory_background_input():
    from minflux_viewer.core.particle_extract import load_any_dataset
    ds = load_any_dataset(_need("rotar_project_confocal_overlay/230921-102431_minflux_mfx.mat"),
                          prefs=_prefs())
    r = crlb_from_inputs(crlb_inputs_for_dataset(ds))
    assert not r.has_background and r.background_source == "none"
    assert "assumed" in r.L_source
    # This reduced export retains the large fbg placeholder but not bgcSense.
    # Without that metadata it must be flagged, not silently become ideal-only.
    assert r.axis_status == "background_inconsistent"
    assert np.isnan(r.primary_sigma)
    assert "background >= orbit" in crlb_log_line(r)


def test_real_mat_with_both_sources_prefers_the_centre():
    from minflux_viewer.core.particle_extract import load_any_dataset
    ds = load_any_dataset(_need("matlab_data_m2410.mat"), prefs=_prefs())
    inputs = crlb_inputs_for_dataset(ds)
    r = crlb_from_inputs(inputs)
    # the centre wins wherever its rate is usable; the rest fall back to fbg
    assert r.background_source.startswith("centre exposure")
    assert np.mean(r.per_loc_bg_source == BG_CENTRE) > 0.9
    only_fbg = dict(inputs, xy=dict(inputs["xy"], efc=None))
    rf = crlb_from_inputs(only_fbg)
    assert rf.background_source == "fbg estimate"
    assert rf.primary_sigma < r.primary_sigma


# ---------------------------------------------------------------------------
# Where the inputs come from: the centre-probe iteration, the power, bgcSense
# ---------------------------------------------------------------------------

def _seq_3d(bgc=False):
    """The standard Abberior 3-D imaging sequence of the sample files."""
    spec = [(0.8, "hexagon", "pt3d", -1, 1), ([0.8, 4], "zline", "mxax", -1, 1),
            (0.8, "square", "mxl3", -1, 1), (0.8, "zline2", "mxa3", -1, 1),
            (0.42, "square", "mxl3", 0.9, 2), (0.42, "zline2", "mxa3", -1, 2),
            (0.21, "square", "mxl3", 0.8, 4), (0.21, "zline2", "mxa3", -1, 4),
            (0.11, "square", "mxl3", -1, 6), (0.11, "zline2", "mxa3", -1, 6)]
    return {"bgcSense": bgc, "ctrDwellFactor": 1,
            "Itr": [{"patGeoFactor": g, "Mode": {"pattern": pt, "id": i}, "ccrLimit": c,
                     "pwrFactor": w, "_excitation": {"power": 12 * w}}
                    for g, pt, i, c, w in spec]}


def test_a_3d_sequence_probes_the_centre_before_its_final_pair():
    seq = _seq_3d()
    pit = detect_from_sequence(parse_sequence(seq))
    assert (pit.lateral_iter, pit.axial_iter) == (8, 9)
    assert not pattern_geometry(seq, 8).centre_probed
    assert centre_reference_iteration(seq, 8) == 6 and centre_reference_iteration(seq, 9) == 6
    assert centre_reference_iteration(seq, 3) is None              # nothing probed before
    assert iteration_power(seq, 8) / iteration_power(seq, 6) == pytest.approx(1.5)
    assert background_sensing(seq) is False and background_sensing(_seq_3d(True)) is True
    assert background_sensing({"Itr": []}) is None


def test_m2205_keeps_its_sequence_under_seq():
    seq = _seq_3d()
    assert extract_sequence_from_zattrs({"seq": seq}) is seq
    assert extract_sequence_from_zattrs({"measurement": {"threads": [{"sequences": [seq]}]}}) is seq
    assert extract_sequence_from_zattrs({"seq": {"Itr": []}}) is None


def test_an_octahedron_is_lateral_and_axial_at_once():
    seq = {"ctrDwellFactor": 0.16, "Itr": [
        {"patGeoFactor": 0.8, "Mode": {"pattern": "hexagon", "id": "pt3d"}, "ccrLimit": -1},
        {"patGeoFactor": 0.42, "Mode": {"pattern": "octahedron", "id": "mx3d"}, "ccrLimit": 0.5},
        {"patGeoFactor": 0.28, "Mode": {"pattern": "octahedron", "id": "mx3d"}, "ccrLimit": -1}]}
    pit = detect_from_sequence(parse_sequence(seq))
    assert pit.lateral_iter == pit.axial_iter == 2 and pit.photon_iters == [2]
    geo = pattern_geometry(seq, 2)
    assert geo.geometry_factors == pytest.approx((np.sqrt(3), np.sqrt(3)))
    assert not geo.donut_envelope
    assert PatternGeometry(L_nm=40.0, pattern="square").geometry_factors[0] == pytest.approx(np.sqrt(2))
    assert PatternGeometry(L_nm=40.0, pattern="square").donut_envelope


def test_an_earlier_centre_exposure_comes_after_the_own_one_and_before_fbg():
    eco = np.full(3, 200.0)
    efo = np.full(3, 2.0e5)
    own = np.array([1.0e5, np.nan, np.nan])
    ref = np.array([5.0e4, 5.0e4, np.nan])
    fbg = np.array([1.0e4, 1.0e4, 1.0e4])
    r = crlb_precision(eco, efo=efo, efc=own, efc_ref=ref, fbg=fbg, centre_dwell_ratio=0.0,
                       centre_reference="iteration 7 scaled x1.50 by the laser power")
    assert r.per_loc_bg_source.tolist() == [BG_CENTRE, BG_CENTRE_EARLIER, BG_FBG]
    assert "centre exposure of iteration 7 scaled x1.50" in r.background_source
    assert r.centre_reference.startswith("iteration 7")
    sigma = r.per_loc_sigma_ideal * np.sqrt((1 + np.array([1.0, 1 / 3, 1 / 19])) ** 2)
    assert r.per_loc_sigma_bg == pytest.approx(sigma)


def test_background_pooled_per_trace_keeps_the_noisy_localizations():
    """One localization whose noisy background reaches its own orbit rate would be
    dropped; pooled with its trace it keeps a background."""
    eco = np.full(6, 100.0)
    efo = np.array([1.00e5, 1.10e5, 0.55e5, 2.0e5, 2.1e5, 1.9e5])
    efc = np.array([0.60e5, 0.62e5, 0.58e5, 0.5e5, 0.5e5, 0.5e5])   # loc 2: efc > efo
    tid = np.array([1, 1, 1, 2, 2, 2])
    alone = crlb_precision(eco, efo=efo, efc=efc, centre_dwell_ratio=1.0)
    assert alone.background_coverage == pytest.approx(5 / 6)
    pooled = crlb_precision(eco, efo=efo, efc=efc, centre_dwell_ratio=1.0, groups=tid)
    assert pooled.background_coverage == 1.0 and pooled.background_pooled
    rho1 = 0.60e5 / (1.00e5 - 0.60e5)          # trace 1: medians of efc and efo
    assert pooled.per_loc_sbr[:3] == pytest.approx(1 / (rho1 * 2))
    assert "pooled per trace" in crlb_log_line(pooled)


# ---------------------------------------------------------------------------
# First principles with the paper's full donut, and the octahedron
# ---------------------------------------------------------------------------

def _donut_fisher_sigma(pos, dwell, rho, sq, n_photons, axis=0):
    """σ from the multinomial Fisher information of Marin & Ries' donut (Eq. 42)
    with a uniform background ρ·s (s: the signal at an orbit point), complex step."""
    pos = np.asarray(pos, float)
    dwell = np.asarray(dwell, float)
    r_orbit = np.max(np.hypot(pos[:, 0], pos[:, 1]))
    s = np.e * r_orbit**2 / (2 * sq**2) * np.exp(-r_orbit**2 / (2 * sq**2))

    def probs(shift):
        x = np.zeros(2, dtype=complex)
        x[axis] = shift
        r2 = np.sum((x - pos) ** 2, axis=1)
        mu = dwell * (np.e * r2 / (2 * sq**2) * np.exp(-r2 / (2 * sq**2)) + rho * s)
        return mu / mu.sum()

    h = 1e-20
    p0 = probs(0.0).real
    dp = probs(1j * h).imag / h
    seen = p0 > 0                     # an exposure that collects nothing informs nothing
    return 1.0 / np.sqrt(n_photons * np.sum(dp[seen] ** 2 / p0[seen]))


@pytest.mark.parametrize("n_points", [3, 4, 6])
@pytest.mark.parametrize("c", [0.0, 0.16, 1.0])
@pytest.mark.parametrize("rho", [0.0, 0.3, 2.0])
def test_lateral_bound_matches_the_full_donut_fisher_information(n_points, c, rho):
    L, sq, n = 75.6, 131.6, 400.0
    pos = _orbit(n_points, L)
    dwell = np.ones(n_points)
    if c > 0:
        pos = np.vstack([pos, [0.0, 0.0]])
        dwell = np.r_[dwell, c * n_points]
    want = _donut_fisher_sigma(pos, dwell, rho, sq, n)
    d = _rates_for(L, rho, c, n_points=n_points, n=n)
    r = crlb_precision(d["eco"], ecc=d["ecc"] if c > 0 else None, efo=d["efo"],
                       efc=d["efc"] if c > 0 else None,
                       fbg=None if c > 0 else d["efc"], L_nm=L, sigma_q_nm=sq,
                       centre_dwell_ratio=c)
    got = r.primary_sigma if rho > 0 else r.median_sigma_ideal
    assert got == pytest.approx(want, rel=1e-9)


@pytest.mark.parametrize("rho", [0.0, 0.5])
def test_octahedron_shares_its_photons_over_three_axes(rho):
    """√3·L/(4√N)·F on every axis: the Fisher information of six exposures at
    ±L/2 on x, y and z (quadratic minimum, emitter at the centre)."""
    L, n = 100.8, 300.0
    R = L / 2
    pos = np.array([[R, 0, 0], [-R, 0, 0], [0, R, 0], [0, -R, 0], [0, 0, R], [0, 0, -R]])
    want = _fisher_sigma(pos, np.ones(6), beta_over_g=rho * R * R, n_photons=n)
    want_z = _fisher_sigma(pos, np.ones(6), beta_over_g=rho * R * R, n_photons=n, axis=2)
    r = crlb_precision(np.array([n]), efo=np.array([1.0 + rho]), fbg=np.array([rho or np.nan]),
                       L_nm=L, L_z_nm=L, geometry_factor_xy=np.sqrt(3),
                       geometry_factor_z=np.sqrt(3), donut_envelope=False)
    got = r.primary_sigma if rho > 0 else r.median_sigma_ideal
    got_z = r.primary_sigma_z if rho > 0 else r.median_sigma_z_ideal
    assert got == pytest.approx(want, rel=1e-6) and got_z == pytest.approx(want_z, rel=1e-6)


# Marin & Ries, arXiv:2410.12427v2, Fig. 2 — the two curves digitised from the
# figure's own vector data (3-point quadratic minimum, b = 0.01, σ_q = 250 nm,
# N_s = 100): (L nm, σ nm). They sit 0.012 nm below Eqs. 31/33 throughout, the
# digitiser's label-centre offset.
_FIG2_IMPERFECT_ZERO = [(24.8, 6.802), (49.8, 4.327), (101.9, 4.057), (150.9, 4.791)]
_FIG2_OUT_OF_FOCUS = [(24.2, 1.856), (48.7, 2.134), (101.9, 3.089), (150.9, 4.154)]


@pytest.mark.parametrize("out_of_focus, points", [(False, _FIG2_IMPERFECT_ZERO),
                                                  (True, _FIG2_OUT_OF_FOCUS)])
def test_reproduces_the_papers_figure_2(out_of_focus, points):
    b, sq, ns = 0.01, 250.0, 100.0
    for L, plotted in points:
        rho = 8 * b * sq**2 / L**2                 # background ÷ signal at ±L/2
        n = ns * (1 + 1.5 * rho) if out_of_focus else ns
        share = 2 * (1 + rho) / (2 * (1 + rho) + rho)      # orbit's photons: 2 points + centre
        r = crlb_precision(np.array([n * share]), ecc=np.array([n * (1 - share)]),
                           efo=np.array([1 + rho]), efc=np.array([rho]),
                           centre_dwell_ratio=0.5, L_nm=L, L_z_nm=L, sigma_q_nm=sq)
        assert r.primary_sigma_z == pytest.approx(plotted, abs=0.03)


# ---------------------------------------------------------------------------
# Real data: the centre exposure lives at another iteration
# ---------------------------------------------------------------------------

def test_real_3d_msr_borrows_the_centre_exposure_of_iteration_7():
    from minflux_viewer.msr.quick_open import load_viewer_msr
    ds = load_viewer_msr(_need("3_Bonly_MRED_75pM_MINFLUX_3D.msr"), prefs=_prefs())[0]
    inputs = crlb_inputs_for_dataset(ds)
    r = crlb_from_inputs(inputs)
    assert "centre exposure of iteration 7 scaled x1.50" in r.background_source
    assert r.background_coverage > 0.95 and r.background_pooled
    assert "centre exposure of iteration 7" in r.background_source_z
    # The power-scaled centre rate and the instrument's own fbg (sensing on)
    # agree — the check that the transfer is right.
    a, b = inputs["xy"]["efc_ref"], inputs["xy"]["fbg"]
    ok = np.isfinite(a) & np.isfinite(b) & (a > 0) & (b > 0)
    assert 0.8 < np.median(a[ok] / b[ok]) < 1.25


def test_real_m2205_octahedron_reads_its_sequence():
    from minflux_viewer.msr.quick_open import load_viewer_msr
    ds = load_viewer_msr(_need("1_sample_A_1-100_seq_3D_ori_exc_5.msr"), prefs=_prefs())[0]
    inputs = crlb_inputs_for_dataset(ds)
    pit = inputs["photon_iterations"]
    assert pit.lateral_iter == pit.axial_iter == 4
    r = crlb_from_inputs(inputs)
    assert r.L_nm == pytest.approx(0.28 * PATTERN_SCALE_NM) and "sequence" in r.L_source
    assert r.pattern == "octahedron" and r.geometry_factor == pytest.approx(np.sqrt(3))
    # The centre was probed only at iteration 3, where m2205 kept almost no
    # values: it is not borrowed, and fbg (sensing on) is used instead.
    assert r.background_source == "fbg estimate"


def test_real_m2205_without_usable_background_says_why():
    from minflux_viewer.msr.quick_open import load_viewer_msr
    ds = load_viewer_msr(_need("rotar_project_confocal_overlay/1_C5C3_PEG_surface_1-25k_overview_"
                               "FLUX_seq_Tracking_2D_Rot_faster_L75nm_final_final_high_salt_"
                               "plus_fuel_1.msr"), prefs=_prefs())[0]
    r = crlb_from_inputs(crlb_inputs_for_dataset(ds))
    assert r.L_nm == pytest.approx(75.6)            # the file name says L75nm
    assert not r.has_background
    assert any("bgcSense" in note for note in r.background_notes)

