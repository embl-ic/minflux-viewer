"""Pytest-wide setup.

**Isolate preferences from the developer's real config.** The application stores
its preferences via ``QSettings("EMBL-IC", "MinfluxViewer")`` — on Windows that is
the registry (``HKCU\\Software\\EMBL-IC\\MinfluxViewer``), a *single shared location*
with no per-run isolation. ``MainWindow.closeEvent`` calls ``AppState.save_prefs()``,
and several tests deliberately set headless flags (e.g. ``show_render=False``,
``show_data_info=False``) before closing a MainWindow. Without isolation, simply
*running the test suite* overwrites the user's real "open render view on load"
preference.

``QSettings.setDefaultFormat`` does **not** affect the 2-arg ``QSettings(org, app)``
constructor (it stays ``NativeFormat``), and ``setPath`` is ignored for the Windows
registry — so redirecting QSettings itself does not work. Instead we patch the two
``AppState`` persistence methods (the only pref read/write path) to use a throwaway
temp file for the whole test session.

The **MSR reader plugin** keeps its own settings in a JSON file at
``~/.minflux_viewer_msr_settings.json``, likewise a single shared location.
Merely constructing ``MsrReaderDialog`` writes it (several handlers call
``_save_settings``), so any test that opens the dialog would otherwise overwrite
the user's real output folder, format selection and last-used input path. It is
redirected the same way.
"""

from __future__ import annotations

import copy
import json

import pytest


@pytest.fixture(scope="session", autouse=True)
def _isolate_preferences(tmp_path_factory):
    try:
        from minflux_viewer.core import app_state as _mod
    except Exception:                       # PyQt6 missing → pure tests still run
        yield
        return

    prefs_file = tmp_path_factory.mktemp("prefs") / "prefs.json"

    def _load(self):
        if prefs_file.exists():
            try:
                return _mod._migrate_prefs(
                    _mod._merge(json.loads(prefs_file.read_text()), _mod.DEFAULT_PREFS))
            except Exception:
                pass
        # deepcopy, like the real AppState._load_prefs: a shallow dict() hands
        # back the live DEFAULT_PREFS sub-dicts, so one test editing
        # prefs["plot"][...] would rewrite the defaults for every later test.
        return _mod._migrate_prefs(copy.deepcopy(_mod.DEFAULT_PREFS))

    def _save(self):
        try:
            prefs_file.write_text(json.dumps(self.prefs))
        except Exception:
            pass

    # Patch the class for the ENTIRE process lifetime and deliberately do NOT
    # restore the originals: a MainWindow's closeEvent can fire during interpreter
    # teardown (exactly when the flaky pyqtgraph teardown crash occurs), and if the
    # real save_prefs had been restored by then it would write the last test's
    # headless prefs (show_render=False, …) to the user's real registry. Leaving the
    # patch in place is harmless — the process exits at session end regardless.
    _mod.AppState._load_prefs = _load
    _mod.AppState.save_prefs = _save
    yield


@pytest.fixture(scope="session", autouse=True)
def _isolate_msr_reader_settings(tmp_path_factory):
    """Redirect the MSR reader plugin's settings file away from the real one."""
    try:
        from minflux_viewer.plugins.msr_reader import msr_reader_dialog as _dlg
    except Exception:                       # PyQt6 missing → pure tests still run
        yield
        return

    settings_file = tmp_path_factory.mktemp("msr") / "msr_settings.json"
    # Patched for the whole process, like the preferences above: a dialog's
    # close/destroy can save during interpreter teardown.
    _dlg.MsrReaderDialog._settings_path = lambda self: settings_file
    yield
