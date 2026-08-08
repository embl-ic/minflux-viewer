"""Opening a file handed to us by the OS.

On macOS a file dropped on the app icon arrives as a QFileOpenEvent on the
running instance, never in argv; on Windows/Linux it arrives in argv. Both
routes converge on ``MainWindow.open_path_from_desktop``.
"""

from __future__ import annotations

import ast
from pathlib import Path

import pytest

from minflux_viewer.ui.main_window import startup_paths_from_argv

SPEC = Path(__file__).resolve().parents[1] / "minflux_viewer.spec"


# --------------------------------------------------------------- command line
def test_argv_accepts_every_supported_extension(tmp_path):
    """The old filter stopped at .mat/.npy/.csv/.msr and silently dropped the
    rest, so `viewer data.json` opened nothing."""
    names = ["a.mat", "b.npy", "c.csv", "d.msr", "e.json", "f.tif", "g.tiff",
             "h.xlsx", "i.tsv", "j.roi"]
    for name in names:
        (tmp_path / name).write_bytes(b"")
    args = [str(tmp_path / name) for name in names]

    assert startup_paths_from_argv(args) == args


def test_argv_accepts_directories_and_zarr_stores(tmp_path):
    """A folder is scanned like a drop; a .zarr store IS the dataset."""
    folder = tmp_path / "acquisition"
    folder.mkdir()
    store = tmp_path / "export.zarr"
    store.mkdir()

    assert startup_paths_from_argv([str(folder), str(store)]) == [
        str(folder), str(store)]


def test_argv_drops_the_macos_process_serial_number_argument():
    """macOS passes -psn_0_… to a bundled .app; probing it as a file is wrong."""
    assert startup_paths_from_argv(["-psn_0_774321"]) == []
    assert startup_paths_from_argv(["-platform", "offscreen"]) == []


def test_argv_ignores_unsupported_and_missing_paths(tmp_path):
    (tmp_path / "notes.docx").write_bytes(b"")
    assert startup_paths_from_argv([str(tmp_path / "notes.docx"),
                                    str(tmp_path / "absent.mat")]) == [
        str(tmp_path / "absent.mat")]          # missing but supported → reported later


def test_argv_preserves_order(tmp_path):
    first, second = tmp_path / "1.mat", tmp_path / "2.msr"
    for p in (first, second):
        p.write_bytes(b"")
    assert startup_paths_from_argv([str(second), str(first)]) == [
        str(second), str(first)]


def test_startup_paths_are_deduplicated_across_argv_and_apple_event(tmp_path):
    from minflux_viewer.__main__ import _deduplicate_startup_paths

    path = tmp_path / "sample.msr"
    path.write_bytes(b"")

    assert _deduplicate_startup_paths([str(path), str(path)]) == [str(path)]


# ------------------------------------------------------- macOS odoc delivery
class _FakeFileOpenEvent:
    """Stand-in for QFileOpenEvent, which PyQt6 refuses to instantiate."""

    def __init__(self, file="", url=None):
        self._file, self._url = file, url

    def type(self):
        from PyQt6.QtCore import QEvent
        return QEvent.Type.FileOpen

    def file(self):
        return self._file

    def url(self):
        return self._url


def test_file_open_event_is_forwarded_to_the_handler(qapp_class):
    """QFileOpenEvent must reach the handler so the requested file is opened."""
    app = qapp_class
    seen: list[str] = []
    app.set_open_handler(seen.append)

    assert app.event(_FakeFileOpenEvent(file="/tmp/sample.msr")) is True
    assert seen == ["/tmp/sample.msr"]


def test_path_is_read_from_the_url_when_file_is_empty():
    """A dropped item can arrive as a file:// URL instead of a plain path."""
    from PyQt6.QtCore import QUrl

    from minflux_viewer.ui.file_open_app import FileOpenApplication

    event = _FakeFileOpenEvent(file="", url=QUrl.fromLocalFile("/tmp/from_url.msr"))
    assert FileOpenApplication._path_from_event(event).endswith("from_url.msr")


def test_paths_arriving_before_the_window_exists_are_queued(qapp_class):
    """A launch-with-document delivers the event during QApplication
    construction, long before MainWindow exists."""
    app = qapp_class
    app.open_path("/tmp/early_one.msr")
    app.open_path("/tmp/early_two.mat")
    assert app.pending_paths == ["/tmp/early_one.msr", "/tmp/early_two.mat"]

    seen: list[str] = []
    app.set_open_handler(seen.append)

    assert seen == ["/tmp/early_one.msr", "/tmp/early_two.mat"]  # in order
    assert app.pending_paths == []


def test_shutdown_drops_late_open_requests(qapp_class):
    """A late Apple Event must not reopen a reader during Qt teardown."""
    app = qapp_class
    seen: list[str] = []
    app.set_open_handler(seen.append)
    app.stop_opening()

    app.open_path("/tmp/late.msr")

    assert seen == []
    assert app.pending_paths == []


def test_a_failing_handler_does_not_escape_into_the_event_loop(qapp_class):
    """An exception out of QApplication.event() crosses the C++ boundary."""
    app = qapp_class
    app.set_open_handler(lambda _p: (_ for _ in ()).throw(RuntimeError("boom")))
    app.open_path("/tmp/x.msr")            # must not raise


def test_a_non_file_url_yields_no_path():
    """A remote URL has no local path; it must not be probed as a file."""
    from PyQt6.QtCore import QUrl

    from minflux_viewer.ui.file_open_app import FileOpenApplication

    event = _FakeFileOpenEvent(file="", url=QUrl("https://example.org/x.msr"))
    assert FileOpenApplication._path_from_event(event) == ""


# ------------------------------------------------------------- bundle plist
def _info_plist() -> dict:
    """The info_plist mapping from the spec's macOS BUNDLE call.

    Evaluated from the AST rather than by importing the spec (which needs
    PyInstaller). Values that are not literals — ``VERSION`` interpolations —
    are irrelevant here and become None instead of failing the parse.
    """
    def value(node):
        try:
            return ast.literal_eval(node)
        except (ValueError, TypeError, SyntaxError):
            return None

    tree = ast.parse(SPEC.read_text(encoding="utf-8"))
    for node in ast.walk(tree):
        if isinstance(node, ast.Call) and getattr(node.func, "id", "") == "BUNDLE":
            for kw in node.keywords:
                if kw.arg == "info_plist" and isinstance(kw.value, ast.Dict):
                    return {value(k): value(v)
                            for k, v in zip(kw.value.keys, kw.value.values)}
    raise AssertionError("no BUNDLE(info_plist=...) in minflux_viewer.spec")


def test_bundle_declares_the_document_types_it_opens():
    """Document declarations make Launch Services send Open-Document events."""
    plist = _info_plist()
    types = plist.get("CFBundleDocumentTypes")
    assert types, "CFBundleDocumentTypes missing"

    declared = {ext.lower()
                for entry in types
                for ext in entry.get("CFBundleTypeExtensions", [])}
    for ext in ("msr", "mat", "npy", "json", "csv", "tif", "zarr"):
        assert ext in declared, ext


def test_zarr_is_declared_as_a_package_and_others_are_not():
    """A .zarr store is a directory; without LSTypeIsPackage Finder opens it as
    a folder instead of handing it to the app."""
    entries = {e["CFBundleTypeName"]: e for e in _info_plist()["CFBundleDocumentTypes"]}
    zarr = next(e for e in entries.values()
                if "zarr" in [x.lower() for x in e["CFBundleTypeExtensions"]])
    assert zarr["LSTypeIsPackage"] is True
    others = [e for e in entries.values() if e is not zarr]
    assert all(e.get("LSTypeIsPackage") is False for e in others)


def test_bundle_allows_intentional_independent_instances():
    assert not _info_plist().get("LSMultipleInstancesProhibited", False)


def test_we_do_not_claim_ownership_of_the_msr_association():
    """Imspector owns .msr where it is installed; we register as an
    alternative handler, not the default."""
    for entry in _info_plist()["CFBundleDocumentTypes"]:
        assert entry.get("LSHandlerRank") == "Alternate"


@pytest.fixture
def qapp_class(qapp):
    """The app under test.

    ``pytest-qt`` has already created a plain QApplication for the session, and
    Qt allows only one, so exercise ``FileOpenApplication``'s own logic on an
    instance created without running ``QApplication.__init__`` again.
    """
    from minflux_viewer.ui.file_open_app import FileOpenApplication

    app = FileOpenApplication.__new__(FileOpenApplication)
    app._pending_paths = []
    app._open_handler = None
    app._accept_open_events = True
    return app


# --------------------------------------------------------------- integration
def test_both_routes_converge_on_route_path_and_log_which_one(qtbot, tmp_path):
    """The Log line is how you tell, on macOS, WHICH instance handled a drop.

    Both processes share one Dock icon and analysis windows are non-owned
    top-levels, so the window that appears is not evidence of the process that
    opened it. The pid + source in the log is.
    """
    import os

    from minflux_viewer.core.app_state import AppState
    from minflux_viewer.ui.main_window import MainWindow

    state = AppState()
    state.prefs.setdefault("data", {}).update({"show_data_info": False,
                                               "show_render": False})
    window = MainWindow(state)
    qtbot.addWidget(window)

    routed: list[str] = []
    window._route_path = routed.append

    sample = tmp_path / "acq.msr"
    sample.write_bytes(b"")
    window.open_path_from_desktop(str(sample), source="macOS Open-Document event")
    window.open_path_from_desktop(str(sample), source="command line")

    assert routed == [str(sample), str(sample)]

    messages = [entry.get("message", "") for entry in state.log_history]
    opens = [m for m in messages if m.startswith("Open request")]
    assert len(opens) == 2
    assert "macOS Open-Document event" in opens[0]
    assert "command line" in opens[1]
    assert all(f"pid {os.getpid()}" in m for m in opens)
    assert all("acq.msr" in m for m in opens)


def test_the_application_handler_routes_through_the_window(qtbot, tmp_path):
    """Wiring check: app.set_open_handler(...) -> window opens the file."""
    from minflux_viewer.core.app_state import AppState
    from minflux_viewer.ui.file_open_app import FileOpenApplication
    from minflux_viewer.ui.main_window import MainWindow

    state = AppState()
    state.prefs.setdefault("data", {}).update({"show_data_info": False,
                                               "show_render": False})
    window = MainWindow(state)
    qtbot.addWidget(window)
    routed: list[str] = []
    window._route_path = routed.append

    app = FileOpenApplication.__new__(FileOpenApplication)
    app._pending_paths = []
    app._open_handler = None
    app._accept_open_events = True
    # Arrives before the window is ready, as on a cold launch-with-document.
    app.open_path(str(tmp_path / "early.msr"))
    app.set_open_handler(
        lambda path: window.open_path_from_desktop(
            path, source="macOS Open-Document event"))

    assert routed == [str(tmp_path / "early.msr")]
