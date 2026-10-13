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

import json
import os
import sys

import pytest

# --------------------------------------------------------------------- Qt tests
#
# There is ONE QApplication: pytest-qt's session-wide ``qapp`` fixture (and
# ``qtbot``, which depends on it). Test modules request it, or alias it under a
# local name, rather than each creating the application its own way. Every test
# that does is marked ``qt`` automatically, so the pure tests can run as a
# separate job, out of reach of the native Qt teardown crash (CLAUDE.md, "Qt
# lifecycle"). A module whose tests start Qt in a child process marks itself
# with ``pytestmark = pytest.mark.qt``.

QT_MARK = "qt"
_QT_FIXTURES = frozenset({"qapp", "qtbot"})

#: Modules where some tests build widgets without requesting ``qapp``, relying
#: on an earlier test having created the application. Each is treated as a Qt
#: module wholesale -- marked, and given the application -- until its tests ask
#: for it themselves; then its entry goes. Found by running every module's
#: ``-m "not qt"`` tests in a process of its own (only these aborted).
_QT_MODULES: set[str] = set()   # empty: test_precision_histograms.py now asks for qapp


def pytest_configure(config):
    config.addinivalue_line(
        "markers",
        "qt: needs a QApplication. Applied automatically to every test that uses "
        "the qapp or qtbot fixture; select with -m qt / -m 'not qt'.")


def _no_display() -> bool:
    """Linux with no display server and no Qt platform chosen: creating a
    QApplication would abort the whole run, so the Qt tests are skipped.
    ``QT_QPA_PLATFORM=offscreen`` runs them instead."""
    return (sys.platform.startswith("linux")
            and not os.environ.get("DISPLAY")
            and not os.environ.get("WAYLAND_DISPLAY")
            and not os.environ.get("QT_QPA_PLATFORM"))


def _in_listed_qt_module(item) -> bool:
    return os.path.basename(str(getattr(item, "fspath", ""))) in _QT_MODULES


@pytest.hookimpl(tryfirst=True)   # before -m deselects by marker
def pytest_collection_modifyitems(config, items):
    skip = pytest.mark.skip(
        reason="No display for Qt tests (set QT_QPA_PLATFORM=offscreen to run them)")
    no_display = _no_display()
    for item in items:
        if (_QT_FIXTURES & set(getattr(item, "fixturenames", ()))
                or _in_listed_qt_module(item)):
            item.add_marker(QT_MARK)
        if no_display and item.get_closest_marker(QT_MARK) is not None:
            item.add_marker(skip)


@pytest.fixture(autouse=True)
def _listed_qt_modules_get_the_application(request):
    """The application for a module in ``_QT_MODULES`` (see there)."""
    if _in_listed_qt_module(request.node):
        request.getfixturevalue("qapp")


@pytest.fixture(autouse=True)
def _only_qt_tests_start_qt(request):
    """A test that brings the QApplication into existence must be a ``qt`` test.

    Otherwise ``-m 'not qt'`` would quietly start Qt, and the pure-test job
    would be exposed to exactly the crash it exists to avoid. Request ``qapp``
    (or ``qtbot``) instead of constructing the application.
    """
    from PyQt6.QtCore import QCoreApplication

    existed = QCoreApplication.instance() is not None
    yield
    if (not existed and QCoreApplication.instance() is not None
            and request.node.get_closest_marker(QT_MARK) is None):
        pytest.fail(
            f"{request.node.nodeid} created a QApplication without being a qt "
            "test: request the qapp (or qtbot) fixture instead.", pytrace=False)


@pytest.fixture(scope="session", autouse=True)
def _isolate_preferences(tmp_path_factory):
    try:
        from minflux_viewer.core import app_state as _mod
    except Exception:                       # PyQt6 missing → pure tests still run
        yield
        return

    prefs_root = tmp_path_factory.mktemp("prefs")
    prefs_file = prefs_root / "prefs.json"
    plugin_root = prefs_root / "plugins"
    pylibs_root = prefs_root / "pylibs"
    plugin_root.mkdir()
    pylibs_root.mkdir()

    def _with_isolated_extension_dirs(prefs):
        """Keep plugin discovery and external imports out of real user dirs."""
        plugin = prefs.setdefault("plugin", {})
        plugin["paths"] = [str(plugin_root)]
        plugin["scan_user_dir"] = False
        plugin["confirmed_dirs"] = [str(plugin_root)]
        plugin["user_lib_paths"] = [str(pylibs_root)]
        return prefs

    def _load(self):
        if prefs_file.exists():
            try:
                return _with_isolated_extension_dirs(
                    _mod._migrate_prefs(
                        _mod._merge(
                            json.loads(prefs_file.read_text()), _mod.DEFAULT_PREFS
                        )
                    )
                )
            except Exception:
                pass
        # Match the real fresh-install path, including pre-recording every
        # one-shot migration so old migrations cannot rewrite current defaults.
        return _with_isolated_extension_dirs(_mod.default_prefs())

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
