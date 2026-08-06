import numpy as np
import pytest

from minflux_viewer.analysis.hlyb_pairwise import (
    HLYB_CLASS_DISTANCES_NM,
    PairFitConfig,
    analyze_hlyb_pairwise,
    calibrate_repeat_kernel,
    complex_profile,
    envelope_null,
    fit_pair_model,
    maxwell_pdf,
    offset_gaussian_pdf,
    pair_distance_profile,
    trace_centroids,
)


# -- building blocks -------------------------------------------------------

def test_offset_gaussian_reduces_to_maxwell_at_zero_separation():
    r = np.linspace(0.01, 20, 400)
    assert np.allclose(offset_gaussian_pdf(r, 0.0, 2.0), maxwell_pdf(r, 2.0))


def test_offset_gaussian_is_normalised_and_peaks_near_the_true_distance():
    r = np.linspace(0.001, 60, 6000)
    for d, s in ((10.0, 2.0), (19.0, 3.0), (5.0, 1.0)):
        pdf = offset_gaussian_pdf(r, d, s)
        assert np.trapezoid(pdf, r) == pytest.approx(1.0, abs=1e-3)
        # 3-D blur biases the mode slightly outward, never inward
        assert d - 0.1 <= r[np.argmax(pdf)] <= d + 2.0 * s


def test_offset_gaussian_matches_a_direct_simulation():
    """``sigma`` is the per-axis spread of the SEPARATION vector, so a single
    displaced point with per-axis sd ``s`` reproduces it directly."""
    rng = np.random.default_rng(4)
    d, s = 11.0, 2.5
    a = np.zeros((200000, 3))
    b = np.array([d, 0.0, 0.0]) + rng.normal(scale=s, size=(200000, 3))
    sample = np.linalg.norm(b - a, axis=1)
    edges = np.arange(0, 40.5, 0.5)
    emp = np.histogram(sample, bins=edges, density=True)[0]
    centres = 0.5 * (edges[:-1] + edges[1:])
    assert np.max(np.abs(emp - offset_gaussian_pdf(centres, d, s))) < 0.01


def test_complex_profile_is_normalised_and_shifts_with_the_label_offset():
    centres = np.arange(0.25, 60, 0.5)
    base = complex_profile(centres, label_offset_nm=0.0, sigma_nm=2.0)
    assert base.sum() * 0.5 == pytest.approx(1.0, abs=1e-3)
    shifted = complex_profile(centres, label_offset_nm=4.0, sigma_nm=2.0)
    mean_base = float((centres * base).sum() / base.sum())
    mean_shift = float((centres * shifted).sum() / shifted.sum())
    assert mean_shift - mean_base == pytest.approx(4.0, abs=0.3)


# -- structure model registry ---------------------------------------------

def test_every_structure_shape_is_normalised_and_moves_with_its_parameters():
    from minflux_viewer.analysis.hlyb_pairwise import STRUCTURE_MODELS, structure_profile

    centres = np.arange(0.25, 60, 0.5)
    cases = {
        "dimer_gaussian": ([10.0, 1.0], [20.0, 1.0]),
        "dimer_uniform": ([8.0, 12.0], [18.0, 22.0]),
        "dimer_lognormal": ([10.0, 0.15], [20.0, 0.15]),
        "trimer_six_site": ([0.0], [4.0]),
    }
    for key, (low, high) in cases.items():
        assert key in STRUCTURE_MODELS
        a = structure_profile(centres, key, low, sigma_nm=2.0, bin_nm=0.5)
        b = structure_profile(centres, key, high, sigma_nm=2.0, bin_nm=0.5)
        assert a.sum() * 0.5 == pytest.approx(1.0, abs=1e-3), key
        mean_a = float((centres * a).sum() / a.sum())
        mean_b = float((centres * b).sum() / b.sum())
        assert mean_b > mean_a + 3.0, key


def test_uniform_band_is_differentiable_so_the_optimiser_can_move_it():
    """Regression: a hard top-hat has zero gradient with respect to its edges
    almost everywhere, so the band stayed pinned at its starting values and the
    elastic hypothesis was never actually fitted."""
    from minflux_viewer.analysis.hlyb_pairwise import STRUCTURE_MODELS

    grid = np.arange(0.25, 40, 0.25)
    pdf = STRUCTURE_MODELS["dimer_uniform"].pdf
    base = pdf(grid, [8.0, 16.0])
    nudged = pdf(grid, [8.0 + 1e-6, 16.0])
    assert not np.array_equal(base, nudged), "band edge has no gradient"
    # the softening is confined to the edges; the interior is still flat
    interior = (grid > 10.0) & (grid < 14.0)
    assert np.ptp(base[interior]) < 5e-3


def test_distance_summary_describes_a_broad_population_honestly():
    """A Gaussian centred at 7 nm with a 9 nm spread is not "a 7 nm distance";
    the reported percentiles must show the population is broad."""
    from minflux_viewer.analysis.hlyb_pairwise import (
        distance_grid, structure_distance_summary)

    grid = distance_grid(0.25, 60.0)
    narrow = structure_distance_summary("dimer_gaussian", [12.0, 0.5], grid)
    broad = structure_distance_summary("dimer_gaussian", [7.0, 9.0], grid)
    assert narrow["median_nm"] == pytest.approx(12.0, abs=0.4)
    assert narrow["spread_nm"] < 1.0
    assert broad["spread_nm"] > 4.0
    assert broad["p84_nm"] - broad["p16_nm"] > 8.0


# -- trace centroids -------------------------------------------------------

def test_trace_centroids_reports_standard_error_and_applies_z_scale():
    rng = np.random.default_rng(1)
    n_tr, per = 40, 25
    tid = np.repeat(np.arange(n_tr), per)
    truth = rng.uniform(0, 1000, size=(n_tr, 3))
    pts = np.repeat(truth, per, axis=0) + rng.normal(scale=5.0, size=(n_tr * per, 3))
    out = trace_centroids(pts * 1e-9, tid, z_scale=0.5, min_loc_per_trace=10)
    assert out["centroids_nm"].shape == (n_tr, 3)
    # z is scaled, x/y are not.  The centroid error is 5/sqrt(25) = 1 nm, so a
    # robust deviation is asserted rather than a per-trace bound that a single
    # 3-sigma draw would trip.
    assert np.median(np.abs(out["centroids_nm"][:, 2] - truth[:, 2] * 0.5)) < 1.5
    assert np.median(np.abs(out["centroids_nm"][:, 0] - truth[:, 0])) < 1.5
    # standard error is sd/sqrt(n), i.e. much smaller than the raw scatter
    assert np.nanmedian(out["sem_nm"]) < 2.0


def test_trace_centroids_drops_short_traces():
    tid = np.array([0, 0, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1])
    pts = np.zeros((tid.size, 3))
    out = trace_centroids(pts, tid, min_loc_per_trace=10)
    assert out["n_traces_total"] == 2
    assert out["centroids_nm"].shape[0] == 1


# -- profile and null ------------------------------------------------------

def test_pair_distance_profile_matches_brute_force():
    rng = np.random.default_rng(2)
    pts = rng.uniform(0, 100, size=(200, 3))
    counts, edges = pair_distance_profile(pts, 40.0, 1.0)
    from scipy.spatial.distance import pdist
    brute = np.histogram(pdist(pts), bins=edges)[0]
    assert np.array_equal(counts, brute)


def test_pair_distance_profile_has_no_exclusion_zone():
    """The whole point of this method: nothing is merged, so arbitrarily close
    pairs survive -- unlike the template pipeline, whose DBSCAN merge empties
    every bin below Dunit/2."""
    rng = np.random.default_rng(3)
    pts = np.repeat(rng.uniform(0, 500, size=(60, 3)), 2, axis=0)
    pts[1::2] += rng.normal(scale=0.4, size=(60, 3))
    counts, edges = pair_distance_profile(pts, 20.0, 0.5)
    assert counts[:4].sum() > 0


def test_envelope_null_preserves_coarse_density_but_destroys_fine_structure():
    rng = np.random.default_rng(5)
    # tight pairs at 10 nm inside a few widely separated blobs
    blobs = rng.uniform(0, 4000, size=(20, 3))
    pts = []
    for b in blobs:
        for _ in range(25):
            base = b + rng.normal(scale=60.0, size=3)
            pts.append(base)
            pts.append(base + np.array([10.0, 0.0, 0.0]))
    pts = np.array(pts)
    counts, _ = pair_distance_profile(pts, 40.0, 1.0)
    null = envelope_null(pts, r_max_nm=40.0, bin_nm=1.0, cell_nm=50.0,
                         replicates=4, rng_seed=0)
    # the real data has a spike at 10 nm; the surrogate must not
    assert counts[10] > 5 * max(null["mean"][10], 1.0)
    # but the two agree in overall scale at large r, where nothing was destroyed
    assert null["mean"][30:].sum() > 0


# -- repeat kernel ---------------------------------------------------------

def test_repeat_kernel_selects_on_time_only_and_recovers_the_true_width():
    rng = np.random.default_rng(6)
    n = 400
    sigma = 2.0
    # consecutive traces alternate: a re-acquisition of the same molecule
    # (short gap, small displacement) then a jump to a new molecule
    cent, t0, t1 = [], [], []
    t = 0.0
    for i in range(n):
        pos = rng.uniform(0, 5000, size=3)
        cent.append(pos); t0.append(t); t1.append(t + 1.0); t += 1.05
        cent.append(pos + rng.normal(scale=sigma, size=3))
        t0.append(t); t1.append(t + 1.0); t += 30.0
    out = calibrate_repeat_kernel(np.array(cent), np.array(t0), np.array(t1),
                                  gap_s=0.2, max_nm=40.0, min_pairs=10)
    assert out["source"].startswith("empirical")
    assert out["n_pairs"] > 200
    # only the second trace of each pair is displaced, so the separation has
    # per-axis sd sigma and its median is 1.538 * sigma
    assert out["median_nm"] == pytest.approx(1.538 * sigma, rel=0.2)


def test_repeat_kernel_falls_back_without_time():
    cent = np.random.default_rng(0).uniform(0, 100, size=(50, 3))
    out = calibrate_repeat_kernel(cent, None, None, fallback_sigma_nm=2.5)
    assert out["source"] == "assumed"
    assert out["sigma_nm"] == pytest.approx(2.5)
    assert out["shape"].sum() > 0


# -- fit -------------------------------------------------------------------

def _synthetic_profile(cfg, n_rep, n_str, structure, params, sigma, seed=0):
    """A profile with a known structural shape planted in it."""
    from minflux_viewer.analysis.hlyb_pairwise import structure_profile

    edges = np.arange(0.0, cfg.r_max_nm + cfg.bin_nm, cfg.bin_nm)
    centres = 0.5 * (edges[:-1] + edges[1:])
    rep_shape = maxwell_pdf(centres, 2.0)
    rep_shape = rep_shape / (rep_shape.sum() * cfg.bin_nm)
    bkg = np.linspace(0.0, 40.0, centres.size)
    truth = (n_rep * rep_shape * cfg.bin_nm
             + n_str * structure_profile(centres, structure, params,
                                         sigma_nm=sigma, bin_nm=cfg.bin_nm) * cfg.bin_nm
             + bkg)
    counts = np.random.default_rng(seed).poisson(truth)
    return counts, edges, rep_shape, bkg


def test_fit_recovers_a_planted_dimer_distance():
    """The distance is an OUTPUT here, free over a wide range, not an input."""
    cfg = PairFitConfig(r_max_nm=60.0, bin_nm=0.5)
    counts, edges, rep, bkg = _synthetic_profile(
        cfg, 4000.0, 6000.0, "dimer_gaussian", [13.5, 1.0], 3.0)
    fit = fit_pair_model(counts, edges, rep, bkg, cfg,
                         structure="dimer_gaussian", sigma_floor_nm=2.0)
    assert fit["success"]
    assert fit["n_repeat_pairs"] == pytest.approx(4000.0, rel=0.3)
    assert fit["n_structure_pairs"] == pytest.approx(6000.0, rel=0.3)
    assert fit["distance_summary"]["median_nm"] == pytest.approx(13.5, abs=1.2)
    assert fit["distance_summary"]["spread_nm"] < 3.0


def test_fit_recovers_a_planted_flat_band():
    """An elastic dimer -- a flat range of distances -- must be recoverable as
    such, not forced onto a single value."""
    cfg = PairFitConfig(r_max_nm=60.0, bin_nm=0.5)
    counts, edges, rep, bkg = _synthetic_profile(
        cfg, 3000.0, 9000.0, "dimer_uniform", [9.0, 21.0], 2.0, seed=4)
    fit = fit_pair_model(counts, edges, rep, bkg, cfg,
                         structure="dimer_uniform", sigma_floor_nm=1.8)
    lo, hi = sorted(fit["structure_params"].values())
    assert lo == pytest.approx(9.0, abs=2.5)
    assert hi == pytest.approx(21.0, abs=2.5)


def test_dimer_distance_stays_inside_its_permitted_range():
    cfg = PairFitConfig(r_max_nm=60.0, bin_nm=0.5,
                        dimer_distance_bounds_nm=(6.0, 20.0))
    counts, edges, rep, bkg = _synthetic_profile(
        cfg, 3000.0, 5000.0, "dimer_gaussian", [11.0, 1.0], 3.0)
    fit = fit_pair_model(counts, edges, rep, bkg, cfg,
                         structure="dimer_gaussian", sigma_floor_nm=2.0)
    assert 6.0 <= fit["structure_params"]["distance_nm"] <= 20.0


def test_trimer_label_offset_cannot_go_negative():
    """Two label displacements can only lengthen an expected distance."""
    cfg = PairFitConfig(r_max_nm=60.0, bin_nm=0.5)
    counts, edges, rep, bkg = _synthetic_profile(
        cfg, 3000.0, 5000.0, "trimer_six_site", [0.0], 3.0)
    fit = fit_pair_model(counts, edges, rep, bkg, cfg,
                         structure="trimer_six_site", sigma_floor_nm=2.0)
    assert fit["structure_params"]["label_offset_nm"] >= 0.0


def test_fitted_blur_cannot_beat_the_centroid_precision():
    cfg = PairFitConfig(r_max_nm=60.0, bin_nm=0.5)
    counts, edges, rep, bkg = _synthetic_profile(
        cfg, 3000.0, 5000.0, "dimer_gaussian", [12.0, 1.0], 3.0)
    fit = fit_pair_model(counts, edges, rep, bkg, cfg,
                         structure="dimer_gaussian", sigma_floor_nm=2.8)
    assert fit["sigma_nm"] >= 2.8 - 1e-6


@pytest.mark.parametrize("planted,expected_median", [
    ("dimer_gaussian", 14.0), ("dimer_uniform", 14.0), ("trimer_six_site", 13.5)])
def test_comparison_recovers_the_planted_distance_whichever_shape_wins(
        planted, expected_median):
    """The shapes are partly degenerate — a flat band can mimic a narrow
    Gaussian by being narrow — so the winning *label* is not the guarantee.
    What must hold is that the planted architecture stays competitive and that
    the recovered distance summary is right regardless of which shape wins,
    which is why the summary and not the shape parameters is what gets
    reported."""
    from minflux_viewer.analysis.hlyb_pairwise import compare_hypotheses

    params = {"dimer_gaussian": [14.0, 0.8], "dimer_uniform": [8.0, 20.0],
              "trimer_six_site": [1.0]}[planted]
    # blur planted at 2.0 nm; tell the fit the same, so the comparison is about
    # shape and not about who guesses the blur
    label = float(np.sqrt(max(2.0 ** 2 - 1.8 ** 2, 1e-3)))
    cfg = PairFitConfig(r_max_nm=60.0, bin_nm=0.5, label_spread_nm=label)
    counts, edges, rep, bkg = _synthetic_profile(
        cfg, 3000.0, 12000.0, planted, params, 2.0, seed=7)
    fits = compare_hypotheses(counts, edges, rep, bkg, cfg, sigma_floor_nm=1.8)
    ranked = sorted(fits, key=lambda k: fits[k]["delta_aic"])
    assert planted in ranked[:2], f"{planted} was rejected: {ranked}"
    assert fits["no_structure"]["delta_aic"] > 10
    winner = fits[ranked[0]]
    assert winner["distance_summary"]["median_nm"] == pytest.approx(
        expected_median, abs=2.0)


def test_a_fixed_geometry_wins_on_its_own_ground_truth():
    """The sharpest hypothesis is the easiest to handicap accidentally: a
    slightly misplaced or over-smoothed model peak costs it enormous likelihood
    against a flexible one. With the blur derived from the measurement and the
    model distances placed exactly, the trimer must simply win here."""
    from minflux_viewer.analysis.hlyb_pairwise import compare_hypotheses

    label = float(np.sqrt(max(2.0 ** 2 - 1.8 ** 2, 1e-3)))
    cfg = PairFitConfig(r_max_nm=60.0, bin_nm=0.5, label_spread_nm=label)
    counts, edges, rep, bkg = _synthetic_profile(
        cfg, 3000.0, 12000.0, "trimer_six_site", [1.0], 2.0, seed=7)
    fits = compare_hypotheses(counts, edges, rep, bkg, cfg, sigma_floor_nm=1.8)
    assert min(fits, key=lambda k: fits[k]["aic"]) == "trimer_six_site"


def test_rigid_distances_are_not_quantised_by_the_distance_grid():
    """Regression: snapping a model distance to the nearest grid point moved it
    by up to half a step. Against a sharp measurement that is a sixth of the
    blur, and it cost the fixed geometry thousands of AIC units for a reason
    unrelated to the structure."""
    from minflux_viewer.analysis.hlyb_pairwise import STRUCTURE_MODELS

    grid = np.arange(5.0, 30.0, 0.25)
    for target in (8.936, 10.138, 17.302):
        pdf = STRUCTURE_MODELS["dimer_gaussian"].pdf(grid, [target, 0.0])
        assert pdf.sum() == pytest.approx(1.0, abs=1e-9)
        # mass is split between neighbours so the mean is exact, not snapped
        assert float((pdf * grid).sum()) == pytest.approx(target, abs=1e-9)
        assert int(np.count_nonzero(pdf)) <= 2


def test_hypotheses_share_one_blur():
    """sigma describes the measurement, not the hypothesis. If each shape could
    choose its own, a broad one would win by claiming blur a narrow one is
    denied."""
    cfg = PairFitConfig(r_max_nm=60.0, bin_nm=0.5)
    counts, edges, rep, bkg = _synthetic_profile(
        cfg, 3000.0, 9000.0, "dimer_gaussian", [12.0, 1.0], 2.5)
    from minflux_viewer.analysis.hlyb_pairwise import compare_hypotheses
    fits = compare_hypotheses(counts, edges, rep, bkg, cfg, sigma_floor_nm=2.0)
    sigmas = {round(f["sigma_nm"], 9) for f in fits.values()}
    assert len(sigmas) == 1


def test_profile_likelihood_reports_an_interval_and_flags_a_flat_scan():
    from minflux_viewer.analysis.hlyb_pairwise import profile_likelihood_distance

    cfg = PairFitConfig(r_max_nm=60.0, bin_nm=0.5,
                        dimer_distance_bounds_nm=(6.0, 22.0))
    counts, edges, rep, bkg = _synthetic_profile(
        cfg, 2000.0, 12000.0, "dimer_gaussian", [14.0, 0.8], 2.0, seed=3)
    scan = profile_likelihood_distance(counts, edges, rep, bkg, cfg,
                                       structure="dimer_gaussian",
                                       sigma_floor_nm=1.8, n_points=25)
    assert scan["available"] and scan["constrained"]
    assert scan["best_nm"] == pytest.approx(14.0, abs=1.5)
    lo, hi = scan["ci95_nm"]
    assert lo <= scan["best_nm"] <= hi


# -- end to end ------------------------------------------------------------

def _simulate_complexes(n_complex, seed=0, label_eff=0.7, traces_per_site=2,
                        locs_per_trace=25, sigma=np.array([2.0, 2.0, 3.0])):
    from minflux_viewer.analysis.hlyb_clustering import HlyBConfig, hlyb_template_model
    T = hlyb_template_model(HlyBConfig())["template_coords_nm"]
    rng = np.random.default_rng(seed)
    pts, tids, tim, t, tt = [], [], [], 0, 0.0
    for _ in range(n_complex):
        q = rng.normal(size=4); q /= np.linalg.norm(q)
        w, x, y, z = q
        R = np.array([[1-2*(y*y+z*z), 2*(x*y-z*w), 2*(x*z+y*w)],
                      [2*(x*y+z*w), 1-2*(x*x+z*z), 2*(y*z-x*w)],
                      [2*(x*z-y*w), 2*(y*z+x*w), 1-2*(x*x+y*y)]])
        origin = rng.uniform(0, 3000, size=3)
        sites = (T @ R.T) + origin
        for s in range(6):
            if rng.random() > label_eff:
                continue
            # Repeats of ONE site follow each other immediately; moving to a
            # different site takes much longer.  Emitting every trace 0.05 s
            # apart would let the time-gap calibration collect inter-site pairs
            # as though they were re-acquisitions, and the kernel would then
            # describe the structure it is supposed to be separated from.
            for _ in range(traces_per_site):
                block = sites[s] + rng.normal(scale=sigma, size=(locs_per_trace, 3))
                pts.append(block)
                tids.append(np.full(locs_per_trace, t)); t += 1
                tim.append(tt + np.arange(locs_per_trace) * 1e-3)
                tt += 0.05
            tt += 30.0
    return (np.concatenate(pts) * 1e-9, np.concatenate(tids), np.concatenate(tim))


def test_end_to_end_on_intact_trimers_keeps_the_trimer_competitive():
    """The shape family must not be biased against the trimer either: on real
    trimers it has to stay in contention, even though a flexible band can
    partly mimic five distances spread over 9-19 nm."""
    loc, tid, tim = _simulate_complexes(400, seed=11)
    # the simulation places sites exactly, with no antibody, so the labelling
    # allowance must be set to zero -- assuming the default 2.3 nm over-smooths
    # every candidate and lets a flexible band beat the true sharp geometry
    cfg = PairFitConfig(min_loc_per_trace=10, z_scaling_factor=1.0,
                        null_replicates=3, label_spread_nm=0.05)
    res = analyze_hlyb_pairwise(loc, tid, tim, cfg)
    assert res["is_pairwise"] is True
    assert res["n_traces_used"] > 1000
    ranked = sorted(res["fits"], key=lambda k: res["fits"][k]["delta_aic"])
    assert "trimer_six_site" in ranked[:2], f"trimer rejected on its own truth: {ranked}"
    assert res["fits"]["no_structure"]["delta_aic"] > 50
    # the excess must extend to about the complex diameter, not beyond
    assert 12.0 < res["excess_outer_nm"] < 30.0
    # and both comparison passes must be reported
    assert set(res["fits_relaxed_kernel"]) == set(res["fits"])


def _simulate_dimers(n_dimers, distance_nm, spread_nm, seed=0,
                     locs_per_trace=25, sigma=np.array([2.0, 2.0, 3.0])):
    """Isolated dimers only -- the trimer lost in sample preparation."""
    rng = np.random.default_rng(seed)
    pts, tids, tim, t, clock = [], [], [], 0, 0.0
    for _ in range(n_dimers):
        origin = rng.uniform(0, 3000, size=3)
        direction = rng.normal(size=3)
        direction /= np.linalg.norm(direction)
        separation = max(rng.normal(distance_nm, spread_nm), 1.0)
        for site in (origin, origin + direction * separation):
            for _ in range(2):
                pts.append(site + rng.normal(scale=sigma, size=(locs_per_trace, 3)))
                tids.append(np.full(locs_per_trace, t)); t += 1
                tim.append(clock + np.arange(locs_per_trace) * 1e-3)
                clock += 0.05
    return (np.concatenate(pts) * 1e-9, np.concatenate(tids), np.concatenate(tim))


def test_end_to_end_on_dimers_recovers_the_dimer_distance_not_the_trimer():
    """The motivating case: if the trimer does not survive preparation, the
    analysis must report the dimer distance rather than forcing the published
    six-site geometry onto the data."""
    loc, tid, tim = _simulate_dimers(700, distance_nm=15.0, spread_nm=1.0, seed=21)
    res = analyze_hlyb_pairwise(loc, tid, tim,
                                PairFitConfig(min_loc_per_trace=10,
                                              z_scaling_factor=1.0,
                                              null_replicates=3))
    assert res["best_hypothesis"].startswith("dimer")
    assert res["fits"]["trimer_six_site"]["delta_aic"] > 20
    summary = res["best_fit"]["distance_summary"]
    assert summary["median_nm"] == pytest.approx(15.0, abs=2.5)


def test_end_to_end_on_elastic_dimers_reports_a_broad_population():
    """A dimer with a flexible linkage gives a broad band; the reported spread
    must exceed the measurement blur rather than collapsing to a sharp value."""
    loc, tid, tim = _simulate_dimers(700, distance_nm=15.0, spread_nm=5.0, seed=22)
    res = analyze_hlyb_pairwise(loc, tid, tim,
                                PairFitConfig(min_loc_per_trace=10,
                                              z_scaling_factor=1.0,
                                              null_replicates=3))
    best = res["best_fit"]
    assert best["distance_summary"]["spread_nm"] > best["sigma_nm"]


def test_end_to_end_finds_no_structure_in_structureless_data():
    rng = np.random.default_rng(12)
    n_tr, per = 1200, 25
    tid = np.repeat(np.arange(n_tr), per)
    centres = rng.uniform(0, 3000, size=(n_tr, 3))
    pts = np.repeat(centres, per, axis=0) + rng.normal(scale=2.0, size=(n_tr * per, 3))
    tim = np.repeat(np.arange(n_tr) * 5.0, per)
    res = analyze_hlyb_pairwise(pts * 1e-9, tid, tim,
                                PairFitConfig(min_loc_per_trace=10,
                                              z_scaling_factor=1.0,
                                              null_replicates=3))
    assert res["excess_outer_nm"] == 0.0


def test_analysis_runs_without_a_time_column():
    loc, tid, _ = _simulate_complexes(200, seed=13)
    res = analyze_hlyb_pairwise(loc, tid, None,
                                PairFitConfig(min_loc_per_trace=10,
                                              z_scaling_factor=1.0,
                                              null_replicates=2))
    assert res["repeat_kernel"]["source"] == "assumed"
    assert res["best_fit"]
