"""Shared cell delineation remains available to live rod analysis."""
import numpy as np
import pytest

from minflux_viewer.analysis.cell_mask import compute_cell_mask


def _punctate_rod(rng, x0, y0, length=2000.0, width=700.0, n_puncta=260, per=40):
    """A rod whose signal is punctate, like a labelled membrane protein: the
    density has large gaps between clusters, which is what broke the original
    mask (Otsu + median filter fragmented the cell and then deleted it)."""
    cx = rng.uniform(x0, x0 + length, n_puncta)
    cy = rng.uniform(y0, y0 + width, n_puncta)
    pts = np.repeat(np.column_stack([cx, cy]), per, axis=0)
    return pts + rng.normal(0, 4.0, size=pts.shape)

def test_cell_mask_delineates_punctate_cells_that_the_median_filter_destroyed():
    rng = np.random.default_rng(7)
    xy = np.vstack([_punctate_rod(rng, 0, 0), _punctate_rod(rng, 0, 1600)])
    res = compute_cell_mask(xy, border_size_nm=150.0)
    # two separate rods, and the mask must cover the great majority of the data
    assert res.stats["n_cells"] == 2
    assert res.stats["in_mask_fraction"] > 0.9
    # a rod 700 nm wide has a ~350 nm half-width
    assert 250.0 < res.stats["max_half_width_nm"] < 450.0
    # shrinking by 150 nm must leave a usable core, not annihilate the cell
    assert 0.2 < res.stats["retained_fraction"] < 0.9
    assert res.cell_id[~res.border_loc].min() >= 1

def test_relative_border_mode_adapts_to_cell_width():
    rng = np.random.default_rng(8)

    def rod(x0, y0, length, width, density=0.012):
        n = int(density * length * width)
        return np.column_stack([rng.uniform(x0, x0 + length, n),
                                rng.uniform(y0, y0 + width, n)])

    # a wide cell and a narrow one at the same surface density: a fixed nm
    # shrink treats them very differently, a relative shrink does not
    xy = np.vstack([rod(0, 0, 2600, 1400), rod(0, 2600, 2600, 500)])
    absolute = compute_cell_mask(xy, border_size_nm=200.0)
    relative = compute_cell_mask(xy, border_mode="relative", border_fraction=0.4)

    def kept_per_cell(res):
        return [float((~res.border_loc[res.cell_id == cid]).mean()) for cid in (1, 2)]

    assert sorted(np.round(absolute.cell_half_width_nm)) == [260.0, 700.0]
    abs_kept = kept_per_cell(absolute)
    rel_kept = kept_per_cell(relative)
    assert abs(abs_kept[0] - abs_kept[1]) > 0.3          # absolute is width-dependent
    assert abs(rel_kept[0] - rel_kept[1]) < 0.1          # relative is not
    assert relative.stats["implied_max_tilt_deg"] == pytest.approx(36.87, abs=0.1)

def test_cell_mask_empty_input_retains_per_localization_shapes():
    result = compute_cell_mask(np.empty((0, 2)))
    assert result.border_loc.shape == (0,)
    assert result.edge_distance_nm.shape == (0,)
    assert result.stats["n_cells"] == 0
