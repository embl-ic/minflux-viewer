"""
Entry point.

Run with::

    python -m minflux_viewer            # from source tree
    minflux-viewer                      # after  poetry install
    minflux-viewer /path/to/data.mat    # open file on launch
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
    from .ui.single_instance import SingleInstanceGuard

    # Everything this process was asked to open. On macOS the request can be an
    # Apple Event rather than argv, and it is delivered as soon as the
    # QApplication exists — so drain the event queue once before deciding
    # whether we are a duplicate, or a hand-off would lose the file.
    app.processEvents()
    startup_paths = _deduplicate_startup_paths(
        startup_paths_from_argv(sys.argv[1:]) + app.take_pending_paths()
    )

    # If a viewer is already running, give it the files and stop. Launch
    # Services is supposed to prevent a second copy (LSMultipleInstancesProhibited)
    # but does not reliably do so for an unregistered/relocated bundle, so the
    # guard does not depend on it. Must happen before any UI is built.
    guard = SingleInstanceGuard()
    if guard.hand_off_to_primary(startup_paths):
        return 0
    guard.listen()

    state  = AppState()
    window = MainWindow(state)
    window.show()

    def _open(path: str, source: str) -> None:
        window.open_path_from_desktop(path, source=source)

    # Install the handler before opening startup paths.  A second Apple Event
    # can arrive during a slow MSR startup and must be dispatched through the
    # same live-window route instead of being left in a queue until teardown.
    app.set_open_handler(lambda p: _open(p, "macOS Open-Document event"))
    for path in startup_paths:
        _open(path, "command line")
    # Later requests: macOS Apple Event to this process, or a duplicate launch
    # that handed its files over rather than opening a window of its own.
    guard.path_received.connect(lambda p: _open(p, "second launch, handed over"))
    guard.raise_requested.connect(window.raise_from_second_launch)
    # Stop accepting hand-offs while shutting down, so a newcomer becomes the
    # primary instead of handing files to a dying process.
    app.aboutToQuit.connect(app.stop_opening)
    app.aboutToQuit.connect(guard.stop)

    exit_code = app.exec()

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
