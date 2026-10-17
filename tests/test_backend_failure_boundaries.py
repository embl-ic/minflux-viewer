"""Unexpected backend failures must not become valid-looking scientific data."""
from types import SimpleNamespace

import numpy as np
import pytest

from minflux_viewer.core import post_load
from minflux_viewer.core.min_enclosing import convex_hull_vertices
from minflux_viewer.core.roi_depth import weighted_depths
from minflux_viewer.core.tiff_source import TiffImageSource
from minflux_viewer.msr import legacy_mbm
from minflux_viewer.msr.session_alignment import AlignmentSession


def raising(error):
    def fail(*args, **kwargs):
        raise error("injected backend failure")
    return fail


@pytest.mark.parametrize("stage", ["anisotropy", "precision", "density", "last_density"])
@pytest.mark.parametrize("error", [MemoryError, RuntimeError])
def test_unexpected_post_load_failure_is_not_a_successful_default(stage, error, monkeypatch):
    from minflux_viewer.core import loader_raw_access

    ds = SimpleNamespace(name="synthetic", cali=SimpleNamespace(z_scaling_factor=0.67),
                         prop=SimpleNamespace(num_dim=3), derived={}, metadata={}, attr={},
                         components=SimpleNamespace(derived_last={}))
    monkeypatch.setattr(loader_raw_access, "attr_values_1d", lambda *args: np.arange(3.0))
    monkeypatch.setattr(post_load, "_materialized_points_nm", lambda *args: np.zeros((3, 3)))
    monkeypatch.setattr(post_load, "_last_valid_points_nm", lambda *args: np.zeros((3, 3)))
    if stage == "last_density":
        ds.attr["den"] = np.zeros(3)
        ds.components.derived_last["marker"] = 1
    prefs = {"data": {"estimate_z_scaling_factor": stage == "anisotropy",
                       "compute_loc_prec": stage == "precision",
                       "compute_local_density": stage in {"density", "last_density"}}}
    algorithms = post_load.PostLoadAlgorithms(raising(error), raising(error), raising(error))
    with pytest.raises(error, match="injected"):
        post_load.compute_post_load_results(ds, prefs, lambda value: None, algorithms=algorithms)
    assert ds.cali.z_scaling_factor == 0.67


@pytest.mark.parametrize("error", [MemoryError, RuntimeError])
def test_kdtree_failure_does_not_start_a_brute_force_fallback(error, monkeypatch):
    monkeypatch.setattr("scipy.spatial.cKDTree", raising(error))
    with pytest.raises(error, match="injected"):
        weighted_depths([(0, 0)] * 9, [0, 1, 2], [0, 1, 2], [4, 5, 6])


@pytest.mark.parametrize("error", [MemoryError, RuntimeError])
def test_hull_failure_does_not_return_an_uncomputed_copy(error, monkeypatch):
    monkeypatch.setattr("scipy.spatial.ConvexHull", raising(error))
    with pytest.raises(error, match="injected"):
        convex_hull_vertices([[0, 0], [1, 0], [0, 1]])


def test_degenerate_hull_still_has_the_documented_fallback():
    points = np.array([[0, 0], [1, 0], [2, 0]])
    np.testing.assert_array_equal(convex_hull_vertices(points), points)


@pytest.mark.parametrize("error", [MemoryError, RuntimeError, OSError])
def test_tiff_page_failure_does_not_trigger_a_whole_volume_read(error):
    source = SimpleNamespace(metadata=SimpleNamespace(axes="ZYX", shape=(2, 3, 3)),
                             _series=SimpleNamespace(pages=[SimpleNamespace(asarray=raising(error))]))
    with pytest.raises(error, match="injected"):
        TiffImageSource._read_plane_page(source, {"Z": 0})


def test_tiff_missing_page_still_allows_the_supported_layout_fallback():
    source = SimpleNamespace(metadata=SimpleNamespace(axes="ZYX", shape=(2, 3, 3)),
                             _series=SimpleNamespace(pages=[]))
    assert TiffImageSource._read_plane_page(source, {"Z": 0}) is None


@pytest.mark.parametrize("error", [MemoryError, RuntimeError, OSError])
def test_legacy_mbm_metadata_failure_is_not_missing_metadata(error):
    node = SimpleNamespace(attrs=SimpleNamespace(asdict=raising(error)))
    with pytest.raises(error, match="injected"):
        legacy_mbm._node_attrs(node)


@pytest.mark.parametrize("error", [MemoryError, RuntimeError, OSError, AttributeError])
def test_legacy_mbm_enumeration_failure_is_not_no_beads(error):
    arch = {"grd/mbm": SimpleNamespace(array_keys=raising(error))}
    with pytest.raises(error, match="injected"):
        legacy_mbm.build_legacy_mbm(arch, log=lambda value: None)


@pytest.mark.parametrize("error", [MemoryError, RuntimeError, OSError, ValueError, TypeError, IndexError])
def test_legacy_mbm_array_failure_does_not_drop_the_bead(error):
    class Node:
        attrs = {"gri": 1, "name": "R1"}

        def __getitem__(self, key):
            raise error("injected backend failure")

    arch = {"grd/mbm": SimpleNamespace(array_keys=lambda: ["R1"], attrs={"used": ["R1"]}),
            "grd/mbm/R1": Node()}
    with pytest.raises(error, match="injected"):
        legacy_mbm.build_legacy_mbm(arch, log=lambda value: None)


@pytest.mark.parametrize("error", [MemoryError, RuntimeError, ValueError])
def test_failed_bead_restriction_never_restores_all_beads(error, monkeypatch):
    points = np.array([(1,), (2,)], dtype=[("gri", "i4")])
    owner = AlignmentSession({}, SimpleNamespace(mbm_map={"channel": points}),
                             log=lambda message: None, excluded={1})
    monkeypatch.setattr(np, "isin", raising(error))
    with pytest.raises(error, match="injected"):
        owner._aligned_mbm_points("channel")


def test_legacy_attrs_mapping_and_absent_node_are_supported():
    assert legacy_mbm._node_attrs(SimpleNamespace(attrs={"gri": 1})) == {"gri": 1}
    assert legacy_mbm._node_attrs(SimpleNamespace()) == {}
    assert legacy_mbm.build_legacy_mbm({}, log=lambda value: None) is None


def test_successful_bead_restriction_preserves_only_selected_ids():
    points = np.array([(1,), (2,)], dtype=[("gri", "i4")])
    owner = AlignmentSession({}, SimpleNamespace(mbm_map={"channel": points}),
                             log=lambda message: None, excluded={1})
    np.testing.assert_array_equal(owner._aligned_mbm_points("channel")["gri"], [2])


def test_expected_numeric_estimate_failure_remains_a_logged_optional_result():
    ds = SimpleNamespace(name="synthetic", cali=SimpleNamespace(z_scaling_factor=0.67),
                         prop=SimpleNamespace(num_dim=3), derived={}, metadata={}, attr={},
                         components=SimpleNamespace(derived_last={}))
    prefs = {"data": {"estimate_z_scaling_factor": True, "compute_loc_prec": False,
                       "compute_local_density": False}}
    algorithms = post_load.PostLoadAlgorithms(raising(ValueError), raising(ValueError), raising(ValueError))
    result = post_load.compute_post_load_results(ds, prefs, lambda value: None, algorithms=algorithms)
    assert result["z_update"]["value"] == 1.0
    assert any(level == "WARN" and "failed" in text for text, level in result["logs"])
    assert ds.cali.z_scaling_factor == 0.67  # worker returns a proposal; never writes calibration


def test_kdtree_unavailable_keeps_the_supported_capability_fallback(monkeypatch):
    import builtins

    args = ([(0, 0)] * 9, [0, 1, 2], [0, 1, 2], [4, 5, 6])
    expected = weighted_depths(*args)
    original = builtins.__import__

    def without_tree(name, globals=None, locals=None, fromlist=(), level=0):
        if name == "scipy.spatial" and "cKDTree" in fromlist:
            raise ImportError("tree unavailable")
        return original(name, globals, locals, fromlist, level)

    monkeypatch.setattr(builtins, "__import__", without_tree)
    np.testing.assert_allclose(weighted_depths(*args), expected)


def test_explicit_missing_tiff_page_allows_layout_fallback():
    source = SimpleNamespace(metadata=SimpleNamespace(axes="ZYX", shape=(2, 3, 3)),
                             _series=SimpleNamespace(pages=[None]))
    assert TiffImageSource._read_plane_page(source, {"Z": 0}) is None


def test_legacy_summary_callback_failure_is_logged_without_dropping_decoded_data(caplog):
    class Node:
        attrs = {"gri": 1, "name": "R1"}

        def __getitem__(self, key):
            return np.zeros(2, dtype=[("pos", "f8", (3,)), ("tim", "f8")])

    arch = {"grd/mbm": SimpleNamespace(array_keys=lambda: ["R1"], attrs={"used": ["R1"]}),
            "grd/mbm/R1": Node()}
    points, names, used = legacy_mbm.build_legacy_mbm(arch, log=raising(RuntimeError))
    assert len(points) == 2 and used == ["R1"] and "1" in names
    assert any(record.exc_info and "summary callback" in record.message for record in caplog.records)
