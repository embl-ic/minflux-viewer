"""Particle-set I/O (core/particle_set.py) + extraction (core/particle_extract.py)."""

from __future__ import annotations

import numpy as np
import pytest

from minflux_viewer.core.particle_set import (
    ParticleData,
    is_particle_set_file,
    load_particle_set,
    save_particle_set,
)


def _particle(seed, m=40, attrs=("efo", "cfr"), with_loc=True, source="ds", name="npc-1"):
    rng = np.random.default_rng(seed)
    pts = rng.normal(0, 50, size=(m, 3))
    return ParticleData(
        points=pts,
        tid=rng.integers(1, 9, m).astype(float),
        unit_id=rng.integers(1, 9, m).astype(np.int32),
        loc=(rng.normal(0, 50, size=(m, 3)) * 1e-9 if with_loc else None),
        attrs={a: rng.normal(0, 1, m) for a in attrs},
        source=source,
        roi_name=name,
        meta={"rezero_offset": [1.0, 2.0, 3.0], "rimf": 0.67, "z_scale": 1.0,
              "source_version": "m2410"},
    )


def test_roundtrip_all(tmp_path):
    parts = [_particle(0, name="npc-1"), _particle(1, m=25, name="npc-2", source="ds2")]
    path = tmp_path / "p.h5"
    save_particle_set(path, parts, content="all")
    assert is_particle_set_file(path)

    out = load_particle_set(path)
    assert len(out) == 2
    for a, b in zip(parts, out):
        assert a.n_locs == b.n_locs
        np.testing.assert_allclose(a.points, b.points, atol=1e-2)   # float32 coords
        np.testing.assert_allclose(np.asarray(a.tid, float), b.tid)
        np.testing.assert_array_equal(a.unit_id, b.unit_id)
        np.testing.assert_allclose(a.loc, b.loc)                    # float64 raw
        for k in a.attrs:
            np.testing.assert_allclose(a.attrs[k], b.attrs[k])
        assert b.source == a.source and b.roi_name == a.roi_name
        assert b.meta["rimf"] == pytest.approx(0.67)
        assert b.meta["source_version"] == "m2410"


def test_loc_tid_drops_attrs_and_raw(tmp_path):
    parts = [_particle(2)]
    path = tmp_path / "p.h5"
    save_particle_set(path, parts, content="loc_tid")
    out = load_particle_set(path)
    assert out[0].attrs == {}
    assert out[0].loc is None
    np.testing.assert_allclose(np.asarray(parts[0].tid, float), out[0].tid)
    np.testing.assert_array_equal(parts[0].unit_id, out[0].unit_id)


def test_heterogeneous_attrs_union_with_nan(tmp_path):
    a = _particle(3, attrs=("efo",), name="a")
    b = _particle(4, attrs=("cfr",), name="b")
    path = tmp_path / "p.h5"
    save_particle_set(path, [a, b], content="all")
    out = load_particle_set(path)
    # both columns exist for both particles; missing one is all-NaN
    assert set(out[0].attrs) == {"efo", "cfr"} == set(out[1].attrs)
    assert np.isfinite(out[0].attrs["efo"]).all() and np.isnan(out[0].attrs["cfr"]).all()
    assert np.isfinite(out[1].attrs["cfr"]).all() and np.isnan(out[1].attrs["efo"]).all()


def test_include_raw_loc_false(tmp_path):
    path = tmp_path / "p.h5"
    save_particle_set(path, [_particle(5)], content="all", include_raw_loc=False)
    out = load_particle_set(path)
    assert out[0].loc is None
    assert "efo" in out[0].attrs        # attrs still kept


# --------------------------------------------------------------------------- #
# extraction from a dataset + rectangle ROIs
# --------------------------------------------------------------------------- #
def _dataset_with_cluster():
    from minflux_viewer.core.dataset import AttrStore, DataProp, FileInfo, MinfluxDataset

    rng = np.random.default_rng(0)
    # 40 inside the box [400..500, 400..500] nm, 40 outside
    inside = rng.uniform(400, 500, size=(40, 2))
    outside = rng.uniform(0, 200, size=(40, 2))
    xy = np.vstack([inside, outside])
    n = xy.shape[0]
    z = rng.normal(300, 10, n)                          # display nm
    attrs = AttrStore({
        "loc_x": xy[:, 0] * 1e-9,                       # metres
        "loc_y": xy[:, 1] * 1e-9,
        "loc_z": z * 1e-9,
        "tid": np.repeat(np.arange(n // 2), 2).astype(float),
        "efo": rng.normal(8e4, 1e3, n),
        "ftr": np.ones(n, bool),
    })
    prop = DataProp(num_loc=n, num_itr=1, num_dim=3, num_traces=n // 2,
                    trace_idx=np.zeros((n // 2, 2), int),
                    num_loc_per_trace=np.full(n // 2, 2),
                    attr_names=["loc_x", "loc_y", "loc_z", "tid", "efo", "ftr"])
    return MinfluxDataset(file=FileInfo(name="synth.mat", folder="/tmp"), prop=prop, attr=attrs)


def _rect(x, y, w, h, name="npc-1"):
    from minflux_viewer.core.roi import RoiRecord
    return RoiRecord.create("rectangle", {"bounds": [x, y, w, h], "angle": 0.0}, name=name)


def test_extract_particles_rezeroes_and_captures_attrs():
    from minflux_viewer.core.particle_extract import extract_particles

    ds = _dataset_with_cluster()
    parts = extract_particles(ds, [_rect(400, 400, 100, 100)])
    assert len(parts) == 1
    p = parts[0]
    assert p.n_locs == 40                              # only the inside cluster
    # re-zeroed to the box centre (450,450): |x|,|y| <= 50
    assert np.abs(p.points[:, 0]).max() <= 50 + 1e-6
    assert np.abs(p.points[:, 1]).max() <= 50 + 1e-6
    assert abs(np.median(p.points[:, 2])) < 1e-6       # Z re-zeroed to its median
    assert p.tid is not None and "efo" in p.attrs
    assert p.loc is not None and p.loc.shape == (40, 3)   # raw metres captured
    assert p.roi_name == "npc-1"
    assert p.meta["rezero_offset"][0] == pytest.approx(450.0, abs=1.0)


def test_transform_to_matrix4_handles_dict_and_array():
    from minflux_viewer.core.overlay import display_transform_record, transform_to_matrix4

    mat = np.eye(4)
    mat[1, 3] = 5.0
    rec = display_transform_record(
        overlay_id="g", overlay_index=1, order=1, lut="Red",
        source_dataset_idx=0, alignment_mode="none", matrix_4x4=mat)
    out = transform_to_matrix4(rec)                     # a transform-record dict
    assert out.shape == (4, 4)
    np.testing.assert_allclose(out, mat)
    assert transform_to_matrix4(None) is None
    np.testing.assert_allclose(transform_to_matrix4(mat), mat)   # bare 4×4 array
    out3 = transform_to_matrix4({"matrix_3x3": [[1, 0, 7], [0, 1, 0], [0, 0, 1]]})
    assert out3.shape == (4, 4) and out3[0, 3] == 7.0            # 3×3 embedded


def test_extract_particles_on_overlay_transform_dict():
    """Collecting from an overlay channel (transform stored as a dict) must not
    crash on `float(dict)` — it records the bare 4×4 in the recipe."""
    from minflux_viewer.core.overlay import display_transform_record
    from minflux_viewer.core.particle_extract import extract_particles

    ds = _dataset_with_cluster()
    mat = np.eye(4)
    mat[0, 3] = 10.0                                    # translate X by +10 nm
    ds.state["overlay_transform"] = display_transform_record(
        overlay_id="g", overlay_index=1, order=1, lut="Red",
        source_dataset_idx=0, alignment_mode="none", matrix_4x4=mat)

    parts = extract_particles(ds, [_rect(410, 400, 100, 100)])   # box follows the shift
    assert len(parts) == 1
    assert parts[0].n_locs == 40
    np.testing.assert_allclose(np.asarray(parts[0].meta["transform_4x4"]), mat)


def test_display_xy_filtered_applies_overlay_transform():
    """Segmentation detects in display coords so ROIs land on the transformed
    overlay channel — display_xy/xyz_filtered must apply the overlay transform."""
    from minflux_viewer.core.overlay import display_transform_record
    from minflux_viewer.core.roi_crop import display_xy_filtered, display_xyz_filtered

    ds = _dataset_with_cluster()
    raw = display_xy_filtered(ds)                       # identity (no transform yet)
    assert raw.shape == (80, 2)

    mat = np.eye(4)
    mat[0, 3] = 10.0                                    # +10 nm X
    mat[1, 3] = -5.0                                    # −5 nm Y
    ds.state["overlay_transform"] = display_transform_record(
        overlay_id="g", overlay_index=1, order=1, lut="Red",
        source_dataset_idx=0, alignment_mode="none", matrix_4x4=mat)

    shifted = display_xy_filtered(ds)
    np.testing.assert_allclose(shifted[:, 0], raw[:, 0] + 10.0, atol=1e-6)
    np.testing.assert_allclose(shifted[:, 1], raw[:, 1] - 5.0, atol=1e-6)
    xyz = display_xyz_filtered(ds)                      # 3-D variant transforms too
    assert xyz.shape == (80, 3)
    np.testing.assert_allclose(xyz[:, 0], raw[:, 0] + 10.0, atol=1e-6)


def test_channel_points_in_roi_rezeroes_by_reference_offset():
    """The other-channel extractor selects the co-located ROI region and re-zeroes
    by the *reference* offset (so inter-channel geometry is preserved)."""
    from minflux_viewer.core.particle_extract import channel_points_in_roi

    ds = _dataset_with_cluster()                 # inside cluster at [400..500] nm, z≈300
    rec = _rect(400, 400, 100, 100)
    offset = [450.0, 450.0, 300.0]               # reference box centre + z median
    pts = channel_points_in_roi(ds, rec, offset)
    assert pts.shape == (40, 3)                   # only the inside cluster
    assert np.abs(pts[:, 0]).max() <= 50 + 1e-6   # re-zeroed to the box centre
    assert np.abs(pts[:, 1]).max() <= 50 + 1e-6
    assert abs(np.median(pts[:, 2])) < 20         # z re-zeroed by the reference median


def test_extract_skips_too_few_and_non_rectangle():
    from minflux_viewer.core.particle_extract import extract_particles
    from minflux_viewer.core.roi import RoiRecord

    ds = _dataset_with_cluster()
    tiny = _rect(0, 0, 5, 5)                            # captures < MIN_LOCS
    line = RoiRecord.create("line", {"points": [[0, 0], [10, 10]]}, name="l")
    assert extract_particles(ds, [tiny, line]) == []


def test_extract_then_roundtrip(tmp_path):
    from minflux_viewer.core.particle_extract import extract_particles

    ds = _dataset_with_cluster()
    parts = extract_particles(ds, [_rect(400, 400, 100, 100)])
    path = tmp_path / "p.h5"
    save_particle_set(path, parts, content="all")
    out = load_particle_set(path)
    assert len(out) == 1
    np.testing.assert_allclose(parts[0].points, out[0].points, atol=1e-2)
    np.testing.assert_allclose(parts[0].loc, out[0].loc)
    np.testing.assert_allclose(parts[0].attrs["efo"], out[0].attrs["efo"])


# --------------------------------------------------------------------------- #
# multi-file load: several particle-set files pooled in one load
# --------------------------------------------------------------------------- #
def test_pool_multiple_particle_set_files(tmp_path):
    """Loading a collection of .h5 files pools every contained particle."""
    files = []
    for i in range(3):
        parts = [_particle(i, name=f"f{i}-a"), _particle(10 + i, m=20, name=f"f{i}-b")]
        p = tmp_path / f"set{i}.h5"
        save_particle_set(p, parts, content="all")
        files.append(str(p))

    pool = []
    for f in files:
        pool.extend(load_particle_set(f))
    assert len(pool) == 6                               # 2 particles × 3 files


def test_load_dialog_spec_carries_multiple_paths(tmp_path):
    pytest.importorskip("PyQt6")
    import sys

    from PyQt6.QtWidgets import QApplication
    from minflux_viewer.ui.particle_average_dialog import ParticleLoadDialog

    QApplication.instance() or QApplication(sys.argv)
    dlg = ParticleLoadDialog()
    try:
        assert dlg._build_spec() is None               # nothing selected yet
        picks = [str(tmp_path / "a.h5"), str(tmp_path / "b.h5")]
        dlg._set_paths = list(picks)
        spec = dlg._build_spec()
        assert spec == {"mode": "particle_set", "paths": picks}
    finally:
        dlg.close()
