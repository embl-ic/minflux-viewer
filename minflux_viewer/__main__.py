"""
Entry point.

Run with::

    python -m minflux_viewer            # from source tree
    minflux-viewer                      # after  poetry install
    minflux-viewer /path/to/data.mat    # open file on launch
"""

from __future__ import annotations

import sys


def main() -> int:
    # Install stdout/stderr redirection ASAP so even early prints get captured
    from .ui.console_window import install_redirection

    install_redirection()

    from PyQt6.QtCore import Qt
    from PyQt6.QtGui import QIcon
    from PyQt6.QtWidgets import QApplication

    from . import __version__, resource_path

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

    app = QApplication(sys.argv)
    app.setApplicationName("MINFLUX Data Viewer")
    app.setApplicationVersion(__version__)
    app.setOrganizationName("EMBL-IC")
    app.setWindowIcon(QIcon(str(resource_path("icons", "minflux_viewer_logo.png"))))

    from .core.app_state import AppState
    from .ui.gpu_capabilities import probe_gpu_capabilities
    from .ui.main_window import MainWindow, startup_paths_from_argv

    state = AppState()
    # Probe once after QApplication exists and before any GL-backed window is
    # offered. Attribute views consult this result instead of discovering a
    # missing driver only after the user has requested a multi-million-row draw.
    state.gpu_capabilities = probe_gpu_capabilities()
    window = MainWindow(state)
    window.show()

    # Data files named on the command line. route_paths (not _route_file) so a
    # directory is scanned exactly as dropping it is, AND so naming a data file
    # together with its _metadata.json applies the recipe without a prompt --
    # the command line is a batch, like a multi-file drop.
    window.route_paths(startup_paths_from_argv(sys.argv[1:]))

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
