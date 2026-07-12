"""MSR reader plugin UI behaviour:
- multiple simultaneous reader dialogs (registry keeps a list),
- each dialog keeps its own parsed data (no shared-global clobbering),
- the button is renamed + gated to Folder (batch) mode (single files auto-parse).
"""

from __future__ import annotations

from types import SimpleNamespace

import numpy as np
import pytest

pytest.importorskip("PyQt6")

from PyQt6.QtWidgets import QApplication

from minflux_viewer.plugins.msr_reader.msr_reader_dialog import (
    MsrReaderDialog,
    register_msr_dialog,
)


@pytest.fixture(scope="module")
def _app():
    return QApplication.instance() or QApplication([])


def test_register_msr_dialog_supports_multiple(_app):
    """Opening the plugin again keeps BOTH dialogs (no single-instance limit)."""
    owner = SimpleNamespace()
    d1 = MsrReaderDialog(state=None, parent=owner)
    d2 = MsrReaderDialog(state=None, parent=owner)
    register_msr_dialog(owner, d1)
    register_msr_dialog(owner, d2)
    assert owner._plugin_msr_reader_dialogs == [d1, d2]     # both retained
    assert owner._plugin_msr_reader_dialog is d2            # back-compat alias = latest
    d1.close()
    d2.close()


def test_button_renamed_and_gated_to_folder_mode(_app):
    """The button is 'Select a file to view content' and only works in Folder
    (batch) mode — a single .msr auto-parses, so it is disabled in file mode."""
    dlg = MsrReaderDialog(state=None)
    assert dlg._parse_button.text() == "Select a file to view content"
    dlg.mode_file.setChecked(True)
    assert dlg._parse_button.isEnabled() is False          # single file → disabled
    dlg.mode_folder.setChecked(True)
    assert dlg._parse_button.isEnabled() is True           # folder → pick a file
    dlg.close()


def test_per_instance_parsed_maps_are_isolated(_app):
    """Two readers each keep their OWN parsed arrays — a second parse must not
    wipe the first (the old shared module-global behaviour)."""
    a = MsrReaderDialog(state=None)
    b = MsrReaderDialog(state=None)
    a._store_parse_result({"mode": "modern", "mfx_map": {"red": np.zeros(3)},
                           "mbm_map": {}, "mbm_meta_map": {}})
    b._store_parse_result({"mode": "modern", "mfx_map": {"green": np.zeros(5)},
                           "mbm_map": {}, "mbm_meta_map": {}})
    assert set(a._msr_state().mfx_map) == {"red"}           # unaffected by b's parse
    assert set(b._msr_state().mfx_map) == {"green"}
    assert a._msr_state().mfx.shape == (3,)                 # first-array alias
    a.close()
    b.close()


def test_auto_parse_only_fires_for_a_single_existing_file(_app, tmp_path, monkeypatch):
    """`_auto_parse_single_input` parses only when the input is a real single
    file in file mode — not in folder mode, not for a missing path."""
    dlg = MsrReaderDialog(state=None)
    calls: list[str] = []
    monkeypatch.setattr(dlg, "on_parse", lambda: calls.append("parsed"))

    f = tmp_path / "sample.msr"
    f.write_bytes(b"\x00")

    dlg.mode_folder.setChecked(True)                        # folder mode → no auto-parse
    dlg.input_path_edit.setText(str(f))
    dlg._auto_parse_single_input()
    assert calls == []

    dlg.mode_file.setChecked(True)
    dlg.input_path_edit.setText(str(tmp_path / "missing.msr"))
    dlg._auto_parse_single_input()
    assert calls == []                                      # missing file → no parse

    dlg.input_path_edit.setText(str(f))
    dlg._auto_parse_single_input()
    assert calls == ["parsed"]                              # single existing file → parse
    dlg.close()
