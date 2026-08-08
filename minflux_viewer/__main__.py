"""
Entry point.

Run with::

    python -m minflux_viewer            # from source tree
    minflux-viewer                      # after  poetry install
    minflux-viewer /path/to/data.mat    # open file on launch
    minflux-viewer --new-instance data  # do not relay startup documents
"""

from __future__ import annotations

import sys
from pathlib import Path


def _deduplicate_startup_paths(paths) -> list[str]:
    """Keep one request when macOS/PyInstaller report the same document twice."""
    seen: set[str] = set()
    unique: list[str] = []
    for raw in paths:
        path = str(raw)
        try:
            key = str(Path(path).expanduser().resolve(strict=False))
        except (OSError, RuntimeError, ValueError):
            key = path
        if key in seen:
            continue
        seen.add(key)
        unique.append(path)
    return unique


def main() -> int:
    # Install stdout/stderr redirection ASAP so even early prints get captured
    from .ui.console_window import install_redirection
    install_redirection()

    from PyQt6.QtCore import Qt
    from PyQt6.QtGui import QIcon
    from PyQt6.QtWidgets import QApplication
    from . import __version__, resource_path
    from .ui.file_open_app import FileOpenApplication

    # pyqtgraph caches compiled GL shader programs globally, but GL program
    # objects are per-context. Without context sharing, a second GLViewWidget
    # (e.g. the scatter 3-D view opened after the volume/3-D view) reuses a
    # program handle that is invalid in its own context, and OpenGL raises
    # GL_INVALID_VALUE in glUseProgram. Sharing contexts makes the cached
    # programs valid everywhere. Must be set BEFORE the QApplication is created.
    QApplication.setAttribute(Qt.ApplicationAttribute.AA_ShareOpenGLContexts, True)
    # Keep the menu bar attached to the main window on platforms that support a
    # native/global menu bar (macOS, and some Linux desktop environments).
    QApplication.setAttribute(Qt.ApplicationAttribute.AA_DontUseNativeMenuBar, True)

    # FileOpenApplication also captures the macOS Open-Document (odoc) event,
    # which is the ONLY way a running instance is told about a file dropped on
    # the app icon — macOS never puts it in argv. It must exist before the
    # window, because the event can arrive during QApplication construction; it
    # queues those until the handler is installed below.
    app = FileOpenApplication(sys.argv)
    app.setApplicationName("MINFLUX Data Viewer")
    app.setApplicationVersion(__version__)
    app.setOrganizationName("EMBL-IC")
    app.setWindowIcon(QIcon(str(resource_path("icons", "minflux_viewer_logo.png"))))

    from .core.app_state import AppState
    from .ui.main_window import MainWindow, startup_paths_from_argv
    relay = None
    relayed_startup_paths: list[str] = []
    relayed_startup_requests: list[dict] = []
    queue_relay_path = relayed_startup_paths.append
    queue_relay_request = relayed_startup_requests.append
    if sys.platform == "darwin":
        from .ui.document_open_relay import (
            DocumentOpenRelay,
            should_handoff_documents,
        )

    # Everything this process was asked to open. On macOS the request can be an
    # Apple Event rather than argv, and it is delivered as soon as the
    # QApplication exists — so drain the event queue before deciding whether a
    # document-bearing launch should offer its paths to a running viewer.
    app.processEvents()
    startup_paths = _deduplicate_startup_paths(
        startup_paths_from_argv(sys.argv[1:]) + app.take_pending_paths()
    )

    if sys.platform == "darwin":
        # This is a document-only fallback, not a single-instance policy. A
        # normal second launch has no paths and always continues to its own UI.
        # A process unexpectedly launched for an odoc request exits only after
        # a live viewer explicitly acknowledges accepting the paths.
        relay = DocumentOpenRelay()
        if (
            should_handoff_documents(startup_paths)
            and relay.hand_off_documents(startup_paths)
        ):
            return 0
        relay.path_received.connect(queue_relay_path)
        relay.request_received.connect(queue_relay_request)
        relay.start()

    state  = AppState()
    window = MainWindow(state)
    window.show()

    def _open(path: str, source: str) -> None:
        window.open_path_from_desktop(path, source=source)

    def _log_relay_request(request: dict) -> None:
        pid = request.get("pid")
        executable = request.get("executable") or "unknown executable"
        state.log(
            f"macOS document relay accepted request from pid {pid}: "
            f"'{executable}'",
            "INFO",
        )

    # Install the handler before opening startup paths. A second Apple Event
    # can arrive during a slow MSR startup and must be dispatched through the
    # same live-window route instead of being left in a queue until teardown.
    app.set_open_handler(lambda p: _open(p, "macOS Open-Document event"))
    if relay is not None:
        try:
            relay.path_received.disconnect(queue_relay_path)
        except (TypeError, RuntimeError):
            pass
        try:
            relay.request_received.disconnect(queue_relay_request)
        except (TypeError, RuntimeError):
            pass
        relay.request_received.connect(_log_relay_request)
        relay.path_received.connect(
            lambda p: _open(p, "macOS document relay")
        )
    for request in relayed_startup_requests:
        _log_relay_request(request)
    startup_paths = _deduplicate_startup_paths(
        startup_paths + relayed_startup_paths
    )
    for path in startup_paths:
        _open(path, "command line")
    app.aboutToQuit.connect(app.stop_opening)
    if relay is not None:
        # Reject (rather than falsely acknowledge) requests once teardown has
        # begun. Keep the server name claimed until app.exec() returns.
        app.aboutToQuit.connect(relay.begin_shutdown)

    exit_code = app.exec()

    if relay is not None:
        relay.stop()

    # Give Qt one explicit cleanup turn before Python starts tearing down
    # module globals and local QWidget wrappers. This is especially helpful on
    # macOS, where late QObject/OpenGL destruction can otherwise surface as a
    # native "quit unexpectedly" dialog even after a normal user quit.
    try:
        window.deleteLater()
        app.processEvents()
    except RuntimeError:
        pass
    try:
        from .ui.console_window import restore_redirection
        restore_redirection()
    except Exception:
        pass
    return int(exit_code)


if __name__ == "__main__":
    sys.exit(main())
