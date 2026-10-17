"""Expected missing inputs may fall back; execution failures must stay visible."""

import builtins
from pathlib import Path

import numpy as np
import pytest

from minflux_viewer.analysis import precision_calculation as precision
from minflux_viewer.core import acquisition_metadata, self_update
from minflux_viewer.msr.utils import to_numpy


def _raises(error):
    def fail(*args, **kwargs):
        raise error("injected failure")
    return fail


@pytest.mark.parametrize("error", [MemoryError, RuntimeError, ValueError, ImportError])
@pytest.mark.parametrize("name,module,operation", [
    ("_rfft2", "scipy.fft", "rfft2"),
    ("_smooth_frc", "scipy.signal", "savgol_filter"),
])
def test_frc_execution_failure_does_not_select_a_different_algorithm(monkeypatch, error, name, module, operation):
    import importlib

    monkeypatch.setattr(importlib.import_module(module), operation, _raises(error))
    data = np.arange(16, dtype=float).reshape(4, 4) if name == "_rfft2" else np.arange(16.)
    with pytest.raises(error, match="injected failure"):
        getattr(precision, name)(data)


@pytest.mark.parametrize("name,module,expected", [
    ("_rfft2", "scipy.fft", np.fft.rfft2),
    ("_smooth_frc", "scipy.signal", lambda y: np.convolve(y, np.ones(3) / 3, mode="same")),
])
def test_frc_falls_back_only_when_optional_import_is_missing(monkeypatch, name, module, expected):
    original = builtins.__import__

    def missing(name, *args, **kwargs):
        if name == module:
            raise ImportError("optional dependency unavailable")
        return original(name, *args, **kwargs)

    monkeypatch.setattr(builtins, "__import__", missing)
    data = np.arange(16.).reshape(4, 4) if name == "_rfft2" else np.arange(16.)
    np.testing.assert_array_equal(getattr(precision, name)(data), expected(data))


@pytest.mark.parametrize("error", [OSError, MemoryError, RuntimeError, AssertionError, ValueError])
def test_lazy_array_read_failure_is_not_converted_into_an_object_array(error):
    class BrokenArray:
        __getitem__ = _raises(error)

    with pytest.raises(error, match="injected failure"):
        to_numpy(BrokenArray())


def test_array_conversion_keeps_supported_inputs_and_non_sliceable_protocol():
    class ArrayOnly:
        def __getitem__(self, key):
            raise TypeError("no slice protocol")

        def __array__(self, dtype=None, copy=None):
            return np.array([3, 4], dtype=dtype)

    array = np.arange(3)
    assert to_numpy(None) is None
    assert to_numpy(array) is array
    np.testing.assert_array_equal(to_numpy([1, 2]), [1, 2])
    np.testing.assert_array_equal(to_numpy(ArrayOnly()), [3, 4])


@pytest.mark.parametrize("error", [MemoryError, AssertionError])
def test_write_probe_does_not_hide_unexpected_failures(tmp_path, monkeypatch, error):
    monkeypatch.setattr(Path, "write_bytes", _raises(error))
    with pytest.raises(error, match="injected failure"):
        self_update.is_writable(tmp_path)


def test_write_probe_reports_expected_permission_failure(tmp_path, monkeypatch):
    monkeypatch.setattr(Path, "write_bytes", _raises(PermissionError))
    assert self_update.is_writable(tmp_path) is False


@pytest.mark.parametrize("operation", ["read_acquisition_rois", "extract_did_label_map"])
@pytest.mark.parametrize("error", [MemoryError, AssertionError])
def test_metadata_scan_does_not_hide_unexpected_mapping_failures(tmp_path, monkeypatch, operation, error):
    import mmap

    path = tmp_path / "input.msr"
    path.write_bytes(b"not empty")
    monkeypatch.setattr(mmap, "mmap", _raises(error))
    with pytest.raises(error, match="injected failure"):
        getattr(acquisition_metadata, operation)(path)


@pytest.mark.parametrize("present", [False, True])
def test_metadata_scan_still_accepts_missing_and_empty_files(tmp_path, present):
    path = tmp_path / "input.msr"
    if present:
        path.touch()
    assert acquisition_metadata.read_acquisition_rois(path) == []
    assert acquisition_metadata.extract_did_label_map(path) == {}


@pytest.mark.parametrize("error", [MemoryError, RuntimeError, AssertionError])
@pytest.mark.parametrize("operation", [
    "_raw_localization_count", "_raw_iteration_count", "_effective_iteration_column",
    "_iteration_count", "_attrs_include_invalid", "_attrs_valid_count",
])
def test_normalization_does_not_turn_execution_failure_into_empty_or_valid_data(error, operation):
    from minflux_viewer.core import loader_normalization as norm
    from minflux_viewer.core import loader_raw_access as raw

    class BrokenArray:
        __array__ = _raises(error)

    array = BrokenArray()
    calls = {
        "_raw_localization_count": lambda: norm._raw_localization_count({"itr": array}),
        "_raw_iteration_count": lambda: norm._raw_iteration_count({"itr": array}),
        "_effective_iteration_column": lambda: norm._effective_iteration_column({"itr": array, "cfr": array}, "cfr"),
        "_iteration_count": lambda: norm._iteration_count(array, 1),
        "_attrs_include_invalid": lambda: raw._attrs_include_invalid({"vld": array}),
        "_attrs_valid_count": lambda: raw._attrs_valid_count({"vld": array}, 10),
    }
    with pytest.raises(error, match="injected failure"):
        calls[operation]()


@pytest.mark.parametrize("error", [MemoryError, RuntimeError])
@pytest.mark.parametrize("module,operation", [("openpyxl", "excel"), ("numcodecs", "zarr")])
def test_optional_backend_import_does_not_relabel_unexpected_failures(monkeypatch, error, module, operation):
    from minflux_viewer.core import export_size, spreadsheet_loader

    original = builtins.__import__

    def fail_import(name, *args, **kwargs):
        if name == module:
            raise error("injected failure")
        return original(name, *args, **kwargs)

    monkeypatch.setattr(builtins, "__import__", fail_import)
    with pytest.raises(error, match="injected failure"):
        if operation == "excel":
            spreadsheet_loader._read_excel(Path("unused.xlsx"))
        else:
            export_size._estimate_zarr_data(np.arange(5))


@pytest.mark.parametrize("error", [MemoryError, RuntimeError])
def test_msr_marker_scan_does_not_hide_execution_failure(tmp_path, monkeypatch, error):
    import mmap

    from minflux_viewer.msr.quick_open import is_viewer_msr

    path = tmp_path / "input.msr"
    path.write_bytes(b"x" * 1024)
    monkeypatch.setattr(mmap, "mmap", _raises(error))
    with pytest.raises(error, match="injected failure"):
        is_viewer_msr(path)


@pytest.mark.parametrize("error", [MemoryError, RuntimeError, AssertionError])
def test_report_matrix_does_not_hide_unexpected_conversion_failure(error):
    from minflux_viewer.analysis.method_rules import _fmt_matrix

    class Scalar:
        __float__ = _raises(error)

    with pytest.raises(error, match="injected failure"):
        _fmt_matrix([[Scalar()]])


@pytest.mark.parametrize("slot", [
    "_data_windows", "_render_windows", "_tiff_windows", "_scatter_windows",
    "_filter_dlgs", "_log_win", "_plugin_msr_reader_dialogs",
])
@pytest.mark.parametrize("error", [RuntimeError, AssertionError, MemoryError])
def test_child_shutdown_tolerates_deleted_qobjects_but_not_arbitrary_failures(monkeypatch, slot, error):
    from types import SimpleNamespace

    from minflux_viewer.ui.common import modeless
    from minflux_viewer.ui.controllers import window_controller
    from minflux_viewer.ui.dialogs import lut_dialog

    closed = []
    monkeypatch.setattr(lut_dialog, "close_shared_lut_dialog", lambda state: None)
    monkeypatch.setattr(modeless, "close_modeless", lambda owner: None)
    owner = SimpleNamespace(_state=SimpleNamespace(mfv=SimpleNamespace(close_windows=lambda: None)))
    for name in ("_data_windows", "_render_windows", "_tiff_windows", "_scatter_windows",
                 "_tracking_windows", "_msd_windows", "_histogram_windows", "_attr_windows", "_filter_dlgs"):
        setattr(owner, name, {})
    broken = SimpleNamespace(close=_raises(error))
    if slot == "_plugin_msr_reader_dialogs":
        setattr(owner, slot, [broken])
    elif slot == "_log_win":
        setattr(owner, slot, broken)
    else:
        getattr(owner, slot)[0] = broken
    # A stale earlier window must not stop the rest from being closed.
    owner._script_editor_win = SimpleNamespace(close=lambda: closed.append(True))
    if error is RuntimeError:
        window_controller._close_all_child_windows(owner)
        assert closed == [True]
    else:
        with pytest.raises(error, match="injected failure"):
            window_controller._close_all_child_windows(owner)


@pytest.mark.parametrize("error", [RuntimeError, MemoryError, AssertionError])
def test_modeless_shutdown_only_ignores_deleted_qobjects(error):
    from types import SimpleNamespace

    from minflux_viewer.ui.common.modeless import close_modeless

    closed = []
    owner = SimpleNamespace(_modeless_windows=[SimpleNamespace(close=_raises(error)),
                                              SimpleNamespace(close=lambda: closed.append(True))])
    if error is RuntimeError:
        close_modeless(owner)
        assert closed == [True]
        assert owner._modeless_windows == []
    else:
        with pytest.raises(error, match="injected failure"):
            close_modeless(owner)
