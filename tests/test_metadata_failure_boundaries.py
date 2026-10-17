"""Malformed input is recoverable; unexpected failures are not missing data."""
from __future__ import annotations

import ast
import inspect
from functools import partial
from types import SimpleNamespace

import numpy as np
import pytest


def fail(error):
    def raise_error(*args, **kwargs):
        raise error("injected backend failure")
    return raise_error


@pytest.mark.parametrize("error", [MemoryError, RuntimeError])
def test_format_resolution_does_not_swallow_detector_failure(error, monkeypatch, tmp_path):
    from minflux_viewer.core import formats

    monkeypatch.setattr(formats, "_load_predicate", lambda name: fail(error))
    with pytest.raises(error, match="injected"):
        formats.resolve_open(tmp_path / "test.json")


@pytest.mark.parametrize("error", [MemoryError, RuntimeError])
@pytest.mark.parametrize("kind", ["filter", "metadata", "collection", "particles", "beads", "image"])
def test_file_predicates_do_not_treat_unexpected_failures_as_another_format(error, kind, monkeypatch, tmp_path):
    if kind == "filter":
        from minflux_viewer.core import filter_io as module
        monkeypatch.setattr(module, "load_filter_json", fail(error))
        call = partial(module.is_filter_json_file, tmp_path / "test.json")
    elif kind == "metadata":
        from minflux_viewer.core import save as module
        path = tmp_path / "test.json"
        path.write_text("{}")
        monkeypatch.setattr(module.json, "load", fail(error))
        call = partial(module.is_metadata_json_file, path)
    elif kind in {"collection", "particles"}:
        from minflux_viewer.core import cell_collection, particle_set
        module = cell_collection if kind == "collection" else particle_set
        import h5py
        monkeypatch.setattr(h5py, "File", fail(error))
        function = module.is_cell_collection_file if kind == "collection" else module.is_particle_set_file
        call = partial(function, tmp_path / "test.h5")
    elif kind == "beads":
        from minflux_viewer.core import mbm_file as module
        monkeypatch.setitem(module._FIELD_READERS, ".npy", fail(error))
        call = partial(module.is_mbm_points_file, tmp_path / "test.npy")
    else:
        from minflux_viewer.core import obf_image_source as module
        monkeypatch.setattr(module, "_scan_stacks", fail(error))
        call = partial(module.msr_is_image_only, tmp_path / "test.msr")
    with pytest.raises(error, match="injected"):
        call()


@pytest.mark.parametrize("error", [MemoryError, RuntimeError])
def test_raw_time_failure_is_not_a_different_acquisition_span(error):
    from minflux_viewer.core.acquisition_time import _dataset_tim

    class Raw(dict):
        get = fail(error)

    dataset = SimpleNamespace(mfx_raw=Raw(), attr={"tim": np.arange(3.0)})
    with pytest.raises(error, match="injected"):
        _dataset_tim(dataset)


@pytest.mark.parametrize("error", [MemoryError, RuntimeError])
def test_filter_reset_keeps_specifications_when_mask_allocation_fails(error, monkeypatch):
    from minflux_viewer.core import dataset_reset

    specs = [{"attribute": "efo", "min": 1}]
    dataset = SimpleNamespace(state={"filter_specs": specs}, prop=SimpleNamespace(num_loc=3),
                              filter_mask=np.array([True, False, True]))
    monkeypatch.setattr(dataset_reset.np, "ones", fail(error))
    with pytest.raises(error, match="injected"):
        dataset_reset.clear_filters(dataset)
    assert dataset.state["filter_specs"] is specs
    np.testing.assert_array_equal(dataset.filter_mask, [True, False, True])


@pytest.mark.parametrize("error", [MemoryError, RuntimeError])
def test_failed_coordinate_read_is_not_an_empty_roi(error, monkeypatch):
    from minflux_viewer.core import roi_crop

    monkeypatch.setattr(roi_crop, "display_coordinate_view", fail(error))
    with pytest.raises(error, match="injected"):
        roi_crop.display_coords(object())


@pytest.mark.parametrize("error", [MemoryError, RuntimeError])
@pytest.mark.parametrize("kind", ["convex", "concave", "volume"])
def test_geometry_failures_do_not_select_another_algorithm(error, kind, monkeypatch):
    from minflux_viewer.core import roi_convert, roi_fit, roi_volume

    points = np.array([[0, 0], [2, 0], [1, 2], [0, 2], [1, 1]], dtype=float)
    if kind == "concave":
        monkeypatch.setattr("scipy.spatial.Delaunay", fail(error))
        call = partial(roi_fit.concave_hull, points)
    else:
        monkeypatch.setattr("scipy.spatial.ConvexHull", fail(error))
        call = partial(roi_convert._convex_hull, points) if kind == "convex" else (
            partial(roi_volume._convex_level, points, 0.0))
    with pytest.raises(error, match="injected"):
        call()


@pytest.mark.parametrize("error", [MemoryError, RuntimeError])
def test_tiff_spacing_failure_is_not_missing_calibration(error):
    from minflux_viewer.core.tiff_source import _parse_imagej_metadata

    class Number:
        __float__ = fail(error)

    with pytest.raises(error, match="injected"):
        _parse_imagej_metadata({"spacing": Number(), "unit": "um"})


@pytest.mark.parametrize("error", [MemoryError, RuntimeError])
def test_spline_failure_is_not_linear_interpolation(error, monkeypatch):
    from minflux_viewer.analysis.curvilinear_segmentation import fit_spline_path

    monkeypatch.setattr("scipy.interpolate.splprep", fail(error))
    with pytest.raises(error, match="injected"):
        fit_spline_path([[0, 0], [20, 5], [40, 10], [60, 5]])


@pytest.mark.parametrize("error", [MemoryError, RuntimeError])
def test_npc_seed_failure_does_not_change_the_fit_initialization(error, monkeypatch):
    from minflux_viewer.analysis import npc_geomfit, npc_wanlu

    monkeypatch.setattr(npc_wanlu, "estimate_trace_size", fail(error))
    with pytest.raises(error, match="injected"):
        npc_geomfit._subunit_centers(np.zeros((3, 3)), np.arange(3))


def test_curve_fit_memory_failure_is_not_a_successful_size_estimate(monkeypatch):
    from minflux_viewer.analysis import trace_analysis

    monkeypatch.setattr(trace_analysis, "curve_fit", fail(MemoryError))
    values = np.linspace(-4, 4, 1000)
    with pytest.raises(MemoryError, match="injected"):
        trace_analysis.estimate_size_nd_details(values, np.repeat(np.arange(100), 10))


@pytest.mark.parametrize("error", [MemoryError, RuntimeError])
def test_rcc_initialization_failure_does_not_start_dme_from_zero(error, monkeypatch):
    from minflux_viewer.analysis import drift_estimators

    monkeypatch.setattr(drift_estimators, "estimate_drift_rcc", fail(error))
    with pytest.raises(error, match="injected"):
        drift_estimators.estimate_drift_dme(np.zeros((20, 3)), np.arange(20.0), n_bins=2, iterations=0)


@pytest.mark.parametrize("error", [MemoryError, RuntimeError])
def test_saved_view_failure_is_not_absent_state(error, monkeypatch):
    from minflux_viewer.msr import io, viewer_state

    monkeypatch.setattr(io, "read_zarr_attrs", fail(error))
    with pytest.raises(error, match="injected"):
        viewer_state.read_viewer_state({})


@pytest.mark.parametrize("error", [MemoryError, RuntimeError])
def test_failed_centroid_is_not_the_origin(error, monkeypatch):
    from minflux_viewer.core import loader_normalization
    from minflux_viewer.msr.session_alignment import AlignmentSession

    monkeypatch.setattr(loader_normalization, "_normalize_mfx_attrs", fail(error))
    session = AlignmentSession({}, SimpleNamespace(mfx_map={"channel": object()}), log=lambda text: None)
    with pytest.raises(error, match="injected"):
        session._mfx_centroid_nm("channel")


def test_bad_bead_chunk_cannot_publish_half_a_stack(monkeypatch):
    from minflux_viewer.msr import mfxdta, msr_parser

    class Node:
        attrs = {}

        def __getitem__(self, key):
            return np.zeros(3, dtype=[("tid", "i4")])

    class BadBeads:
        __getitem__ = fail(ValueError)

    arch = {"mfx": Node(), "grd/mbm/points": BadBeads()}
    monkeypatch.setattr(mfxdta, "extract_zarr_store", lambda blob: {})
    monkeypatch.setattr(msr_parser.zarr2, "open", lambda *args, **kwargs: arch)
    published = []
    monkeypatch.setattr(msr_parser, "set_mfx_for", lambda *args: published.append("mfx"))
    monkeypatch.setattr(msr_parser, "set_mbm_for", lambda *args: published.append("mbm"))
    result = msr_parser.GeneralMSRParser()._mfxdta_dataset_entry(
        0, "channel", b"", lambda store: [], {}, {}, lambda text: None)
    assert result is None
    assert not published


@pytest.mark.parametrize("error", [MemoryError, RuntimeError])
def test_failed_roi_recomputation_is_not_an_absent_roi(error, monkeypatch):
    from minflux_viewer.core import channel_labels, roi_crop

    monkeypatch.setattr(roi_crop, "compute_crop_mask", fail(error))
    dataset = SimpleNamespace(state={}, derived={}, prop=SimpleNamespace(num_loc=3))
    record = SimpleNamespace(type="rectangle", id="test", selection_dirty=True)
    with pytest.raises(error, match="injected"):
        channel_labels.roi_mask_for_record(dataset, record)


@pytest.mark.parametrize("error", [MemoryError, RuntimeError])
def test_failed_filter_mask_read_is_not_an_unfiltered_dataset(error):
    from minflux_viewer.core.dataset_reset import _mask_is_all

    class Dataset:
        filter_mask = property(fail(error))

    with pytest.raises(error, match="injected"):
        _mask_is_all(Dataset())


@pytest.mark.parametrize("error", [MemoryError, RuntimeError])
def test_lossy_header_decoder_failure_is_not_unrecognized_text(error):
    from minflux_viewer.core.format_sniff import _looks_text

    class Header(bytes):
        def decode(self, encoding="utf-8", errors="strict"):
            if errors == "strict":
                raise UnicodeDecodeError("utf-8", self, 0, 1, "injected invalid UTF-8")
            raise error("injected backend failure")

    with pytest.raises(error, match="injected"):
        _looks_text(Header(b"\xffa"))


def test_plugin_method_failure_is_reported_without_inventing_builtin_attribution(monkeypatch):
    from minflux_viewer.analysis import method_text

    monkeypatch.setattr(method_text, "plugin_method_entry", fail(RuntimeError))
    stages, citations = method_text._collect(None, [{"message": "Z scaling factor: 0.7"}])
    assert "Plugin method details unavailable" in stages["other"][0]
    assert "Z scaling factor: 0.7" in stages["other"][0]
    assert not stages["analysis"]
    assert not citations


@pytest.mark.parametrize("finished", [False, True])
def test_failed_stop_callback_restores_status_without_overwriting_completion(finished):
    from minflux_viewer.core.task_registry import TaskRegistry

    registry = TaskRegistry()
    def cancel():
        if finished:
            handle.finish()
        raise RuntimeError("cancel failed")
    handle = registry.register("test", "analysis", cancel=cancel)
    handle.start()
    assert not registry.request_stop(handle.id)
    assert registry.get(handle.id).status == ("done" if finished else "running")


def test_malformed_update_version_is_an_error_not_up_to_date(monkeypatch):
    from minflux_viewer.core import updater

    monkeypatch.setattr(updater, "fetch_latest_release", lambda **kwargs: SimpleNamespace(version="garbage"))
    assert updater.check_for_update("0.4.4").status == "error"


@pytest.mark.parametrize("error", [MemoryError, RuntimeError])
def test_preference_failure_does_not_replace_saved_settings(error, monkeypatch):
    from minflux_viewer.core import app_state

    monkeypatch.setattr(app_state, "QSettings", lambda *args: SimpleNamespace(value=lambda *args: "{}"))
    monkeypatch.setattr(app_state, "_migrate_prefs", fail(error))
    monkeypatch.setattr(app_state, "default_prefs", lambda: {})
    # The session fixture replaces this method to protect real user settings.
    # Compile only the real method against the patched globals, never a QObject.
    tree = ast.parse(inspect.getsource(app_state))
    cls = next(node for node in tree.body if isinstance(node, ast.ClassDef) and node.name == "AppState")
    method = next(node for node in cls.body if isinstance(node, ast.FunctionDef) and node.name == "_load_prefs")
    namespace = vars(app_state).copy()
    exec(compile(ast.Module(body=[method], type_ignores=[]), app_state.__file__, "exec"), namespace)
    with pytest.raises(error, match="injected"):
        namespace["_load_prefs"](object())


@pytest.mark.parametrize("error", [MemoryError, RuntimeError])
@pytest.mark.parametrize("command", ["file", "roi_channel"])
def test_gui_command_reports_backend_failure_without_continuing(error, command, monkeypatch, tmp_path):
    from PyQt6.QtWidgets import QMessageBox
    from minflux_viewer.core import channel_labels, formats
    from minflux_viewer.ui.analysis.attribute_separation_dialog import AttributeSeparationDialog
    from minflux_viewer.ui.controllers.file_open_controller import _route_file

    messages, dialogs, added = [], [], []
    monkeypatch.setattr(QMessageBox, "critical", lambda *args: dialogs.append(args))
    owner = SimpleNamespace(_state=SimpleNamespace(log=lambda text, *args: messages.append(text)))
    if command == "file":
        monkeypatch.setattr(formats, "resolve_open", fail(error))
        _route_file(owner, str(tmp_path / "test.json"))
    else:
        monkeypatch.setattr(channel_labels, "roi_mask_for_record", fail(error))
        owner._roi_candidates = lambda: [SimpleNamespace(name="test")]
        owner._roi_combo = SimpleNamespace(setCurrentIndex=lambda index: None)
        owner._dataset = lambda: object()
        owner._add_mask_channel = lambda *args, **kwargs: added.append(True)
        AttributeSeparationDialog._on_roi_source_picked(owner, 1)
    assert dialogs and "injected backend failure" in dialogs[0][-1]
    assert messages and "injected backend failure" in messages[-1]
    assert not added
