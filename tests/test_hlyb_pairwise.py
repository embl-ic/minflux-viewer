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

def _synthetic_profile(cfg, n_rep, n_cx, delta, sigma, seed=0):
    edges = np.arange(0.0, cfg.r_max_nm + cfg.bin_nm, cfg.bin_nm)
    centres = 0.5 * (edges[:-1] + edges[1:])
    rep_shape = maxwell_pdf(centres, 2.0)
    rep_shape = rep_shape / (rep_shape.sum() * cfg.bin_nm)
    bkg = np.linspace(0.0, 40.0, centres.size)
    truth = (n_rep * rep_shape * cfg.bin_nm
             + n_cx * complex_profile(centres, label_offset_nm=delta,
                                      sigma_nm=sigma, bin_nm=cfg.bin_nm) * cfg.bin_nm
             + bkg)
    counts = np.random.default_rng(seed).poisson(truth)
    return counts, edges, rep_shape, bkg


def test_fit_recovers_planted_amplitudes_and_label_offset():
    cfg = PairFitConfig(r_max_nm=60.0, bin_nm=0.5)
    counts, edges, rep, bkg = _synthetic_profile(cfg, 4000.0, 6000.0, 3.0, 3.0)
    fit = fit_pair_model(counts, edges, rep, bkg, cfg, sigma_floor_nm=2.0)
    assert fit["success"]
    assert fit["n_repeat_pairs"] == pytest.approx(4000.0, rel=0.25)
    assert fit["n_complex_pairs"] == pytest.approx(6000.0, rel=0.25)
    assert fit["label_offset_nm"] == pytest.approx(3.0, abs=1.0)
    assert fit["sigma_nm"] == pytest.approx(3.0, abs=1.0)


def test_label_offset_cannot_go_negative():
    """Two label displacements can only lengthen an expected distance, so a
    negative offset is unphysical; leaving it free let a one-distance model
    slide its peak away and impersonate a broad background."""
    cfg = PairFitConfig(r_max_nm=60.0, bin_nm=0.5)
    counts, edges, rep, bkg = _synthetic_profile(cfg, 3000.0, 5000.0, 0.0, 3.0)
    fit = fit_pair_model(counts, edges, rep, bkg, cfg, sigma_floor_nm=2.0)
    assert fit["label_offset_nm"] >= 0.0


def test_fitted_blur_cannot_beat_the_centroid_precision():
    cfg = PairFitConfig(r_max_nm=60.0, bin_nm=0.5)
    counts, edges, rep, bkg = _synthetic_profile(cfg, 3000.0, 5000.0, 1.0, 3.0)
    fit = fit_pair_model(counts, edges, rep, bkg, cfg, sigma_floor_nm=2.8)
    assert fit["sigma_nm"] >= 2.8 - 1e-6


def test_hypothesis_comparison_prefers_the_planted_geometry():
    cfg = PairFitConfig(r_max_nm=60.0, bin_nm=0.5)
    counts, edges, rep, bkg = _synthetic_profile(cfg, 3000.0, 9000.0, 1.0, 2.5)
    from minflux_viewer.analysis.hlyb_pairwise import compare_hypotheses
    fits = compare_hypotheses(counts, edges, rep, bkg, cfg, sigma_floor_nm=2.0)
    assert min(fits, key=lambda k: fits[k]["aic"]) == "six_site"
    assert fits["dimer_only"]["delta_aic"] > 10
    assert fits["no_structure"]["delta_aic"] > 10


def test_hypotheses_share_one_blur():
    """sigma describes the measurement, not the hypothesis. If each model could
    choose its own, a single-distance model would inflate into a featureless
    bump and win for the wrong reason."""
    cfg = PairFitConfig(r_max_nm=60.0, bin_nm=0.5)
    counts, edges, rep, bkg = _synthetic_profile(cfg, 3000.0, 9000.0, 1.0, 2.5)
    from minflux_viewer.analysis.hlyb_pairwise import compare_hypotheses
    fits = compare_hypotheses(counts, edges, rep, bkg, cfg, sigma_floor_nm=2.0)
    sigmas = {round(f["sigma_nm"], 9) for f in fits.values()}
    assert len(sigmas) == 1


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
            for _ in range(traces_per_site):
                block = sites[s] + rng.normal(scale=sigma, size=(locs_per_trace, 3))
                pts.append(block)
                tids.append(np.full(locs_per_trace, t)); t += 1
                tim.append(tt + np.arange(locs_per_trace) * 1e-3)
                tt += 0.05
    return (np.concatenate(pts) * 1e-9, np.concatenate(tids), np.concatenate(tim))


def test_end_to_end_recovers_the_six_site_geometry():
    loc, tid, tim = _simulate_complexes(400, seed=11)
    cfg = PairFitConfig(min_loc_per_trace=10, z_scaling_factor=1.0, null_replicates=3)
    res = analyze_hlyb_pairwise(loc, tid, tim, cfg)
    assert res["is_pairwise"] is True
    assert res["n_traces_used"] > 1000
    assert res["best_hypothesis"] == "six_site"
    assert res["fits"]["no_structure"]["delta_aic"] > 50
    # the excess must extend to about the complex diameter, not beyond
    assert 12.0 < res["excess_outer_nm"] < 30.0
    # and both comparison passes must be reported
    assert set(res["fits_relaxed_kernel"]) == set(res["fits"])


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
