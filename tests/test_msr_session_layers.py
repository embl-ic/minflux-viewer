"""MSR session calculations and exports have no widget or plugin-loader dependency."""

from __future__ import annotations

import importlib
import subprocess
import sys
from types import SimpleNamespace

import numpy as np
import pytest


def test_session_backends_import_without_qt_or_ui():
    code = """
import sys
import minflux_viewer.msr.session
import minflux_viewer.msr.session_export
import minflux_viewer.msr.session_alignment
import minflux_viewer.msr.viewer_import
assert not any(n.startswith(('PyQt6', 'pyqtgraph', 'minflux_viewer.ui', 'minflux_viewer.plugins')) for n in sys.modules)
"""
    result = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True)
    assert result.returncode == 0, result.stdout + result.stderr


@pytest.mark.parametrize("module,symbol", [
    ("json_view_dialog", "JsonViewDialog"),
    ("array_preview_dialog", "ArrayPreviewDialog"),
    ("plot_window", "PlotWindow"),
    ("alignment_save_dialog", "AlignmentSaveDialog"),
    ("field_dialog", "FieldDialog"),
    ("viewer_alignment_dialog", "ViewerAlignmentDialog"),
    ("alignment_plot_window", "AlignmentPlotWindow"),
    ("batch_export_signals", "_BatchExportSignals"),
    ("batch_export_task", "_BatchExportTask"),
    ("parsed_export_task", "_ParsedExportTask"),
    ("viewer_open_signals", "_ViewerOpenSignals"),
    ("export_estimate_signals", "_ExportEstimateSignals"),
    ("export_estimate_task", "_ExportEstimateTask"),
    ("path_drop_line_edit", "PathDropLineEdit"),
    ("parse_worker", "_ParseWorker"),
])
def test_legacy_dialog_exports_the_canonical_class(module, symbol):
    prefix = "minflux_viewer.plugins.msr_reader."
    old = importlib.import_module(prefix + "msr_reader_dialog")
    new = importlib.import_module(prefix + module)
    assert getattr(old, symbol) is getattr(new, symbol)


def test_fallback_alignment_uses_one_session_and_leaves_input_arrays_unchanged():
    from minflux_viewer.msr.session_alignment import AlignmentSession

    dtype = [("loc", "f8", (3,)), ("itr", "i4"), ("vld", "?"),
             ("fnl", "?"), ("tid", "i4")]
    reference = np.zeros(8, dtype=dtype)
    reference["loc"][:, 0] = np.arange(8) * 1e-9
    reference["loc"][:, 1] = np.arange(8) * 2e-9
    reference["vld"] = reference["fnl"] = True
    reference["tid"] = np.arange(8)
    moving = reference.copy()
    moving["loc"] += np.array([10, 20, 0]) * 1e-9
    before = moving.copy()
    state = SimpleNamespace(mfx_map={"reference": reference, "moving": moving},
                            mbm_map={}, mbm_meta_map={})
    parsed = {"datasets": [{"display_name": name} for name in state.mfx_map]}
    messages = []
    session = AlignmentSession(parsed, state, log=messages.append)
    assert session.parsed is parsed
    assert session._msr_state() is state
    transforms = session._viewer_fallback_alignment_transforms(
        ["reference", "moving"], "data centroid")
    np.testing.assert_allclose(np.asarray(transforms["moving"]["matrix_4x4"])[:3, 3],
                               [-10, -20, 0], atol=1e-10)
    np.testing.assert_array_equal(transforms["reference"]["matrix_4x4"], np.eye(4))
    np.testing.assert_array_equal(moving, before)
    assert len(messages) == 2
