"""UI glue for the .msr-as-a-format work: the Save dialog offers .msr and forces
raw content for it; the Dataset Manager right-click delegates to save_dataset."""

from types import SimpleNamespace

import pytest

pytest.importorskip("PyQt6")

from PyQt6.QtWidgets import QApplication

from minflux_viewer.core.app_state import AppState
from minflux_viewer.ui.dataset_manager import DatasetManager
from minflux_viewer.ui.save_dialog import _ALL_FORMATS, SaveProcessedDataDialog


@pytest.fixture(scope="module")
def _app():
    yield QApplication.instance() or QApplication([])


def test_msr_offered_in_save_dialog(_app):
    assert "msr" in _ALL_FORMATS
    prefs = {"data": {"export_formats": ["mat", "msr"]}, "file": {}}
    dlg = SaveProcessedDataDialog("ds1", prefs=prefs)
    assert dlg._format.findData("msr") >= 0


def test_msr_forces_raw_content(_app):
    prefs = {"data": {"export_formats": ["mat", "msr"],
                      "export_content": ["raw", "snapshot"]}, "file": {}}
    dlg = SaveProcessedDataDialog("ds1", prefs=prefs)
    # pick snapshot, then switch the format to .msr → content must come out raw
    si = dlg._content_combo.findData("snapshot")
    if si >= 0:
        dlg._content_combo.setCurrentIndex(si)
    dlg._format.setCurrentIndex(dlg._format.findData("msr"))
    opts = dlg.options()
    assert opts["fmt"] == "msr" and opts["content"] == "raw"
    assert not dlg._content_combo.isEnabled()          # locked to raw while .msr
    # switching back to a normal format re-enables the content choice
    dlg._format.setCurrentIndex(dlg._format.findData("mat"))
    assert dlg._content_combo.isEnabled()


def test_dataset_manager_save_as_delegates(_app):
    saved = []
    state = AppState()
    state.datasets.append(SimpleNamespace(
        file=SimpleNamespace(name="a.mat", datetime="", folder="/tmp"),
        prop=SimpleNamespace(num_dim=2, num_loc=10, num_traces=2, num_itr=1),
        metadata={}, name="a"))

    class _Owner:
        def save_dataset(self, ds):
            saved.append(ds)

        def dataset_view_status(self, i):
            return "None"

    dm = DatasetManager(state, _Owner())
    dm._save_dataset(0)
    assert len(saved) == 1 and saved[0] is state.datasets[0]
    dm._save_dataset(99)                               # out of range → no-op
    assert len(saved) == 1
