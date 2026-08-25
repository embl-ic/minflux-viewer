"""Pooling ROI-delimited cells across datasets for the HlyB/D pair analysis."""

import numpy as np
import pytest

from minflux_viewer.analysis.hlyb_staged import (
    Staged3DConfig,
    analyze_hlyb_staged_3d,
    analyze_hlyb_staged_pooled,
    components_from_cells,
    infer_label_sites,
    segment_spatial_components,
    trace_centroids,
)
from minflux_viewer.core.cell_collection import (
    CellCollection,
    CellSample,
    extract_cells,
    is_cell_collection_file,
    load_cell_collection,
    region_roi_records,
    save_cell_collection,
)
from minflux_viewer.core.roi import RoiRecord, RoiStore


# --------------------------------------------------------------------------- #
# Synthetic rod cells
# --------------------------------------------------------------------------- #
def _rod_cell(centre_nm, *, n_sites=160, length_nm=2000.0, radius_nm=300.0,
              pair_nm=14.0, pair_fraction=0.35, locs_per_trace=14,
              precision_nm=3.0, seed=0):
    """A rod-surface cell with a planted short-range population.

    Returns raw metres, so it can be fed to the analysis exactly like a dataset.
    """
    rng = np.random.default_rng(seed)
    axial = rng.uniform(-0.5 * length_nm, 0.5 * length_nm, n_sites)
    phi = rng.uniform(0.0, 2.0 * np.pi, n_sites)
    sites = np.column_stack([
        axial, radius_nm * np.cos(phi), radius_nm * np.sin(phi)])
    n_pairs = int(pair_fraction * n_sites)
    if n_pairs:
        partner = sites[:n_pairs].copy()
        step = rng.normal(size=(n_pairs, 3))
        step /= np.linalg.norm(step, axis=1, keepdims=True)
        sites = np.vstack([sites, partner + pair_nm * step])
    sites = sites + np.asarray(centre_nm, dtype=float)

    loc, tid, tim = [], [], []
    for index, site in enumerate(sites):
        n = int(rng.integers(locs_per_trace, locs_per_trace + 5))
        loc.append(site + rng.normal(0.0, precision_nm, size=(n, 3)))
        tid.append(np.full(n, index, dtype=float))
        tim.append(np.full(n, float(index)) + rng.random(n))
    return (np.concatenate(loc) * 1e-9, np.concatenate(tid),
            np.concatenate(tim))


def _cell(centre, seed, *, dataset="ds", roi="cell", **kwargs):
    loc, tid, tim = _rod_cell(centre, seed=seed, **kwargs)
    return CellSample(loc_m=loc, tid=tid, tim=tim, dataset=dataset, roi=roi)


def _cfg(**changes):
    base = dict(z_scaling_factor=1.0, null_replicates=19,
                sensitivity_replicates=9, run_sensitivity=False,
                run_stratum_profile=False, min_sites_per_component=20,
                bootstrap_replicates=0)
    base.update(changes)
    return Staged3DConfig(**base)


# --------------------------------------------------------------------------- #
# Collection container
# --------------------------------------------------------------------------- #
def test_collection_accumulates_and_summarises():
    pool = CellCollection()
    assert len(pool) == 0
    pool.add(_cell((0.0, 0.0, 0.0), 1, dataset="A", roi="cell 1"))
    pool.extend([_cell((9000.0, 0.0, 0.0), 2, dataset="A", roi="cell 2"),
                 _cell((0.0, 9000.0, 0.0), 3, dataset="B", roi="cell 1")])
    assert len(pool) == 3
    assert pool.datasets == ["A", "B"]
    info = pool.summary()
    assert info["n_cells"] == 3 and info["n_datasets"] == 2
    assert info["n_locs"] == sum(c.n_locs for c in pool)
    assert info["n_traces"] == sum(c.n_traces for c in pool)


def test_collection_reports_an_already_pooled_cell():
    """Collecting is a repeated manual step; re-collecting must be detectable."""
    pool = CellCollection([_cell((0.0, 0.0, 0.0), 1, dataset="A", roi="cell 1")])
    assert pool.has("A", "cell 1")
    assert not pool.has("A", "cell 2")
    assert not pool.has("B", "cell 1")


def test_collection_remove_and_clear():
    pool = CellCollection([_cell((0.0, 0.0, 0.0), i, roi=f"c{i}") for i in range(4)])
    assert pool.remove([1, 3]) == 2
    assert [c.roi for c in pool] == ["c0", "c2"]
    pool.clear()
    assert len(pool) == 0


def test_cells_keep_their_original_coordinates():
    """Unlike particle averaging, pooling must not re-zero: cells are compared
    only within themselves, and moving them would corrupt the surface null."""
    cell = _cell((12345.0, -6789.0, 400.0), 7)
    centre = cell.loc_m.mean(axis=0) * 1e9
    assert abs(centre[0] - 12345.0) < 200.0
    assert abs(centre[1] - (-6789.0)) < 200.0


# --------------------------------------------------------------------------- #
# Persistence
# --------------------------------------------------------------------------- #
def test_collection_round_trips_through_a_file(tmp_path):
    pool = CellCollection([
        _cell((0.0, 0.0, 0.0), 1, dataset="A", roi="cell 1"),
        _cell((9000.0, 0.0, 0.0), 2, dataset="B", roi="cell 7"),
    ])
    path = tmp_path / "pool.h5"
    save_cell_collection(path, pool)
    assert is_cell_collection_file(path)

    back = load_cell_collection(path)
    assert len(back) == 2
    for before, after in zip(pool, back):
        assert after.dataset == before.dataset
        assert after.roi == before.roi
        assert np.allclose(after.loc_m, before.loc_m)
        assert np.array_equal(after.tid, before.tid)
        assert np.allclose(after.tim, before.tim)


def test_a_collection_without_time_round_trips(tmp_path):
    loc, tid, _tim = _rod_cell((0.0, 0.0, 0.0), seed=4)
    pool = CellCollection([CellSample(loc, tid, None, "A", "cell 1")])
    path = tmp_path / "no_time.h5"
    save_cell_collection(path, pool)
    back = load_cell_collection(path)
    assert back.cells[0].tim is None


def test_saving_an_empty_collection_is_refused(tmp_path):
    with pytest.raises(ValueError, match="empty"):
        save_cell_collection(tmp_path / "empty.h5", CellCollection())


def test_a_foreign_file_is_not_mistaken_for_a_collection(tmp_path):
    path = tmp_path / "other.h5"
    path.write_bytes(b"not hdf5 at all")
    assert not is_cell_collection_file(path)
    with pytest.raises(OSError):        # h5py refuses it before we ever look
        load_cell_collection(path)

    import h5py
    stranger = tmp_path / "stranger.h5"
    with h5py.File(stranger, "w") as handle:
        handle.attrs["format"] = "something_else"
    assert not is_cell_collection_file(stranger)
    with pytest.raises(ValueError, match="not a MINFLUX cell collection"):
        load_cell_collection(stranger)


# --------------------------------------------------------------------------- #
# ROI selection
# --------------------------------------------------------------------------- #
def _polygon(name, x0, y0, x1, y1, dataset_idx=0):
    record = RoiRecord.create(
        "polygon",
        {"points": [[x0, y0], [x1, y0], [x1, y1], [x0, y1]], "closed": True},
        name=name, coordinate_space="plot")
    record.context = {"dataset_idx": dataset_idx}
    return record


def test_only_region_rois_of_this_dataset_are_offered():
    store = RoiStore()
    keep = _polygon("cell 1", 0, 0, 100, 100)
    other_ds = _polygon("cell 2", 0, 0, 100, 100, dataset_idx=1)
    line = RoiRecord.create("line", {"points": [[0, 0], [50, 50]]},
                            name="a line", coordinate_space="plot")
    line.context = {"dataset_idx": 0}
    point = RoiRecord.create("point", {"point": [1.0, 2.0, 3.0]},
                             name="a point", coordinate_space="plot")
    point.context = {"dataset_idx": 0}
    for record in (keep, other_ds, line, point):
        store.add(record)
    assert [r.name for r in region_roi_records(store, 0)] == ["cell 1"]


# --------------------------------------------------------------------------- #
# Extraction from a dataset
# --------------------------------------------------------------------------- #
def _dataset_with_two_cells():
    from minflux_viewer.core.dataset import build_localization_dataset

    left, tid_l, tim_l = _rod_cell((1000.0, 1000.0, 0.0), seed=11)
    right, tid_r, tim_r = _rod_cell((6000.0, 1000.0, 0.0), seed=12)
    loc = np.vstack([left, right]) * 1e9
    tid = np.concatenate([tid_l, tid_r + 10000])
    ds = build_localization_dataset(
        name="two cells", x_nm=loc[:, 0], y_nm=loc[:, 1], z_nm=loc[:, 2],
        tid=tid, tim=np.concatenate([tim_l, tim_r]),
        source_version="simulation")
    return ds


def test_extract_cells_cuts_one_cell_per_region_roi():
    ds = _dataset_with_two_cells()
    records = [_polygon("cell 1", -500, -500, 3000, 2500),
               _polygon("cell 2", 4500, -500, 8000, 2500)]
    cells, skipped = extract_cells(ds, records)
    assert skipped == []
    assert [c.roi for c in cells] == ["cell 1", "cell 2"]
    assert all(c.dataset == "two cells" for c in cells)
    assert all(c.n_locs > 100 for c in cells)
    # Each ROI takes its own cell only.
    for cell, expect_x in zip(cells, (1000.0, 6000.0)):
        assert abs(cell.loc_m[:, 0].mean() * 1e9 - expect_x) < 400.0
    # Together they account for essentially the whole dataset.
    assert sum(c.n_locs for c in cells) == ds.prop.num_loc


def test_an_roi_holding_too_few_localizations_is_reported_not_dropped():
    ds = _dataset_with_two_cells()
    records = [_polygon("cell 1", -500, -500, 3000, 2500),
               _polygon("empty corner", 20000, 20000, 21000, 21000)]
    cells, skipped = extract_cells(ds, records)
    assert [c.roi for c in cells] == ["cell 1"]
    assert len(skipped) == 1
    assert "empty corner" in skipped[0]
    assert "below the" in skipped[0]


def test_extracted_cells_carry_raw_metres_not_display_nm():
    ds = _dataset_with_two_cells()
    cells, _ = extract_cells(ds, [_polygon("cell 1", -500, -500, 3000, 2500)])
    assert np.max(np.abs(cells[0].loc_m)) < 1e-3          # metres, not nm


# --------------------------------------------------------------------------- #
# Pooled analysis
# --------------------------------------------------------------------------- #
def test_components_from_cells_makes_one_component_per_cell():
    centres = np.zeros((60, 3))
    index = np.repeat([0, 1, 2], 20)
    out = components_from_cells(centres, index, min_sites=10)
    assert len(out["components"]) == 3
    assert set(np.unique(out["labels"]).tolist()) == {0, 1, 2}
    assert out["n_excluded_sites"] == 0


def test_a_cell_below_the_minimum_is_excluded_not_merged():
    centres = np.zeros((25, 3))
    index = np.concatenate([np.zeros(20), np.full(5, 1)]).astype(int)
    out = components_from_cells(centres, index, min_sites=20)
    assert len(out["components"]) == 1
    assert out["n_excluded_sites"] == 5
    assert out["n_all_components"] == 2


def test_pooling_adds_pair_counts_without_creating_cross_cell_pairs():
    """The core guarantee: two pooled cells contribute exactly the pairs they
    contribute alone, and nothing more."""
    cfg = _cfg()
    a = _cell((0.0, 0.0, 0.0), 21, roi="a")
    b = _cell((500_000.0, 0.0, 0.0), 22, roi="b")   # far away, and irrelevant
    only_a = analyze_hlyb_staged_pooled([a.as_cell()], cfg)
    only_b = analyze_hlyb_staged_pooled([b.as_cell()], cfg)
    both = analyze_hlyb_staged_pooled([a.as_cell(), b.as_cell()], cfg)

    assert both["n_cells"] == 2 and both["n_components"] == 2
    assert np.allclose(both["observed"], only_a["observed"] + only_b["observed"])


def test_pooled_cells_may_share_coordinates_without_interacting():
    """Two acquisitions can put different cells at the same coordinates; pooling
    must not fuse them, which is why sites are inferred per cell."""
    cfg = _cfg()
    a = _cell((0.0, 0.0, 0.0), 31, dataset="A", roi="cell 1")
    b = _cell((0.0, 0.0, 0.0), 32, dataset="B", roi="cell 1")   # same place!
    pooled = analyze_hlyb_staged_pooled([a.as_cell(), b.as_cell()], cfg)
    assert pooled["n_components"] == 2
    separate = [analyze_hlyb_staged_pooled([c.as_cell()], cfg) for c in (a, b)]
    assert np.allclose(pooled["observed"],
                       separate[0]["observed"] + separate[1]["observed"])


def test_pooled_matches_the_single_dataset_path_on_the_same_components():
    """Given the same cells, the pooled entry point must reproduce the
    single-dataset statistics exactly — it only changes where components
    come from."""
    cfg = _cfg()
    parts = [_rod_cell((0.0, 0.0, 0.0), seed=41),
             _rod_cell((12_000.0, 0.0, 0.0), seed=42)]
    loc = np.vstack([p[0] for p in parts])
    tid = np.concatenate([parts[0][1], parts[1][1] + 10_000])
    tim = np.concatenate([p[2] for p in parts])
    single = analyze_hlyb_staged_3d(loc, tid, tim, cfg)

    traces = trace_centroids(loc, tid, tim, z_scale=cfg.z_scaling_factor,
                             min_loc_per_trace=cfg.min_loc_per_trace)
    sites = infer_label_sites(
        traces["centroids_nm"], traces["sem_nm"], traces["n_locs"],
        traces["t_start"], traces["t_end"], merge_nm=cfg.site_merge_nm,
        sigma_factor=cfg.site_sigma_factor,
        precision_floor_nm=cfg.site_precision_floor_nm)
    comps = segment_spatial_components(
        sites["centers_nm"], link_nm=cfg.cell_link_nm,
        min_sites=cfg.min_sites_per_component)
    # However many components link mode happens to form, hand exactly those to
    # the pooled path — the claim under test is equivalence, not a component count.
    n_components = len(comps["components"])
    assert n_components >= 2

    trace_label = comps["labels"][sites["trace_to_site"]]
    ids = np.asarray(traces["trace_ids"])
    cells = []
    for cid in range(n_components):
        wanted = set(ids[trace_label == cid].tolist())
        sel = np.array([t in wanted for t in tid])
        cells.append({"loc_m": loc[sel], "tid": tid[sel], "tim": tim[sel],
                      "label": f"cell {cid}", "dataset": "sim", "roi": f"r{cid}"})
    pooled = analyze_hlyb_staged_pooled(cells, cfg)

    assert pooled["n_components"] == single["n_components"]
    assert np.array_equal(pooled["observed"], single["observed"])
    assert pooled["summary"]["band_ratio"] == pytest.approx(
        single["summary"]["band_ratio"], rel=1e-9)
    assert pooled["summary"]["positive_excess_centroid_nm"] == pytest.approx(
        single["summary"]["positive_excess_centroid_nm"], rel=1e-9)


def test_pooled_result_matches_the_single_dataset_result_contract():
    """The result window and method-text generator consume both, so the key set
    must not drift apart."""
    cfg = _cfg()
    loc, tid, tim = _rod_cell((0.0, 0.0, 0.0), seed=51, n_sites=140)
    single = analyze_hlyb_staged_3d(loc, tid, tim, cfg)
    pooled = analyze_hlyb_staged_pooled(
        [{"loc_m": loc, "tid": tid, "tim": tim, "label": "c", "dataset": "d",
          "roi": "r"}], cfg)
    assert not (set(single) - set(pooled))
    for key in ("centers_nm", "observed", "null_mean", "null_lo", "null_hi",
                "excess_counts", "summary", "config"):
        assert key in pooled


def test_pooling_recovers_a_planted_distance_across_datasets():
    cfg = _cfg(run_sensitivity=False)
    cells = [
        _cell((0.0, 0.0, 0.0), 60 + i, dataset=f"ds{i // 2}", roi=f"cell {i}",
              pair_nm=14.0, n_sites=110)
        for i in range(4)
    ]
    result = analyze_hlyb_staged_pooled([c.as_cell() for c in cells], cfg)
    assert result["n_cells"] == 4
    assert result["n_datasets"] == 2
    assert result["n_components"] == 4
    assert result["summary"]["band_ratio"] > 1.0
    assert 10.0 < result["summary"]["positive_excess_centroid_nm"] < 18.0


def test_pooled_run_reports_per_cell_provenance():
    cfg = _cfg()
    cells = [_cell((0.0, 0.0, 0.0), 71, dataset="A", roi="cell 1"),
             _cell((20_000.0, 0.0, 0.0), 72, dataset="B", roi="cell 4")]
    result = analyze_hlyb_staged_pooled([c.as_cell() for c in cells], cfg)
    rows = result["per_cell"]
    assert [r["dataset"] for r in rows] == ["A", "B"]
    assert [r["roi"] for r in rows] == ["cell 1", "cell 4"]
    assert all(r["n_localizations"] > 0 and r["analysed"] for r in rows)
    assert result["datasets"] == ["A", "B"]
    assert any("within a cell" in text for text in result["limitations"])
    assert any("hand-drawn ROI" in text for text in result["limitations"])


def test_pooled_analysis_refuses_an_empty_pool():
    with pytest.raises(ValueError, match="No cells were collected"):
        analyze_hlyb_staged_pooled([], _cfg())


def test_pooled_analysis_explains_a_pool_with_too_few_sites():
    cfg = _cfg(min_sites_per_component=5000)
    cell = _cell((0.0, 0.0, 0.0), 81)
    with pytest.raises(ValueError, match="minimum"):
        analyze_hlyb_staged_pooled([cell.as_cell()], cfg)


def test_pooled_analysis_rejects_a_malformed_cell():
    cfg = _cfg()
    with pytest.raises(ValueError, match=r"loc_m must have shape"):
        analyze_hlyb_staged_pooled(
            [{"loc_m": np.zeros((10, 2)), "tid": np.arange(10),
              "tim": None, "label": "bad"}], cfg)
