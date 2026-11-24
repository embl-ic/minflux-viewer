"""The read-only acquisition-sequence viewer and its entry points."""

from __future__ import annotations

import json

import pytest

pytest.importorskip("PyQt6")

from minflux_viewer.core.acquisition_sequence import SequenceSource, sequences_from_mfx_attrs
from tests.test_acquisition_sequence import imaging_2d, imaging_3d, native_record


class _Owner:
    """What the open/view helpers need from the main window."""

    def __init__(self):
        self._modeless_windows = []
        self.logs: list[tuple[str, str]] = []
        self._state = self
        self._status_label = self
        self.prefs = {}

    def log(self, message, level="INFO", **_kwargs):
        self.logs.append((str(level), message))

    def setText(self, _text):
        pass


def _window(qtbot, sources):
    from minflux_viewer.ui.tools.sequence_viewer import SequenceViewerWindow

    window = SequenceViewerWindow(sources, title="t")
    qtbot.addWidget(window)
    return window


def test_the_iteration_table_shows_the_derived_values(qtbot):
    from minflux_viewer.core.acquisition_sequence import ITERATION_COLUMNS

    window = _window(qtbot, [SequenceSource("Imaging_3D", imaging_3d(), "file")])
    assert window.parentWidget() is None                      # modeless, unparented
    assert window.table.rowCount() == 10
    headers = [window.table.horizontalHeaderItem(c).text() for c in range(window.table.columnCount())]
    assert headers == [header for _key, header in ITERATION_COLUMNS]
    l_column = headers.index("L (nm)")
    final_column = headers.index("Final for")
    assert window.table.item(9, l_column).text() == "39.6"
    assert window.table.item(8, final_column).text() == "lateral"
    assert window.table.item(9, final_column).text() == "axial"
    assert window.table.item(9, 0).font().bold()
    assert not window.table.item(0, 0).font().bold()
    assert not window._selector.isVisibleTo(window)            # one sequence: no selector


def test_threads_are_selectable_and_the_copy_follows_the_selection(qtbot):
    sources = sequences_from_mfx_attrs(native_record(), origin="dataset")
    window = _window(qtbot, sources)
    assert window._selector.isVisibleTo(window)
    assert window._selector.count() == 2
    window._selector.setCurrentIndex(1)
    assert window.current_source().sequence["id"] == "seqMBM2d_v8"
    text = window.table_text().splitlines()
    assert text[0].startswith("Iteration\tMode\t")
    assert len(text) == 1 + len(imaging_2d()["Itr"])


def test_the_json_tree_describes_documented_keys(qtbot):
    window = _window(qtbot, [SequenceSource("Imaging_2D", imaging_2d(), "file")])
    tree = window.document_view.tree
    root = tree.invisibleRootItem().child(0)
    by_name = {root.child(i).text(0): root.child(i) for i in range(root.childCount())}
    assert "loop resumes" in by_name["headstart"].toolTip(0)


def test_opening_the_same_source_raises_the_existing_window(qtbot):
    from minflux_viewer.ui.tools.sequence_viewer import open_sequence_window

    owner = _Owner()
    sources = [SequenceSource("Imaging_2D", imaging_2d(), "file")]
    first = open_sequence_window(owner, sources, title="a", source_key="file:a")
    qtbot.addWidget(first)
    again = open_sequence_window(owner, sources, title="a", source_key="file:a")
    other = open_sequence_window(owner, sources, title="b", source_key="file:b")
    qtbot.addWidget(other)
    assert again is first and other is not first


def test_a_sequence_file_opens_its_viewer_and_a_damaged_one_says_why(qtbot, tmp_path, monkeypatch):
    from minflux_viewer.ui.controllers import file_open_controller
    from minflux_viewer.ui.tools.sequence_viewer import SequenceViewerWindow

    good = tmp_path / "Imaging_2D_Nov2024.json"
    good.write_text(json.dumps(imaging_2d()), encoding="utf-8")
    owner = _Owner()
    file_open_controller._open_sequence_file(owner, str(good))
    windows = [w for w in owner._modeless_windows if isinstance(w, SequenceViewerWindow)]
    assert len(windows) == 1
    qtbot.addWidget(windows[0])
    assert "Imaging_2D" in owner.logs[-1][1]

    damaged = imaging_2d()
    del damaged["Itr"][0]["Mode"]
    bad = tmp_path / "bad.json"
    bad.write_text(json.dumps(damaged), encoding="utf-8")
    shown = []
    monkeypatch.setattr(file_open_controller.QMessageBox, "warning",
                        lambda *args: shown.append(args[-1]))
    file_open_controller._open_sequence_file(owner, str(bad))
    assert shown and "iteration 1 has no 'Mode'" in shown[0]
    assert owner.logs[-1][0] == "ERROR"


def test_the_dataset_manager_offers_the_sequence_only_when_recorded():
    from minflux_viewer.ui.tools.dataset_manager import DatasetManager

    class _Ds:
        def __init__(self, metadata):
            self.metadata = metadata
            self.name = "ds"

    assert DatasetManager._has_sequence(_Ds({"native_zarr_mfx_attrs": native_record()}))
    assert DatasetManager._has_sequence(_Ds({"mfx_sequence": imaging_2d()}))
    assert not DatasetManager._has_sequence(_Ds({}))
