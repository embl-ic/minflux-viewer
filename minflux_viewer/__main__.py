"""
Entry point.

Run with::

    python -m minflux_viewer            # from source tree
    minflux-viewer                      # after  poetry install
    minflux-viewer /path/to/data.mat    # open file on launch
"""

from __future__ import annotations

import sys


def main() -> None:
    # Install stdout/stderr redirection ASAP so even early prints get captured
    from .ui.console_window import install_redirection
    install_redirection()

    from PyQt6.QtGui import QIcon
    from PyQt6.QtWidgets import QApplication
    from . import __version__, resource_path

    app = QApplication(sys.argv)
    app.setApplicationName("MINFLUX Data Viewer")
    app.setApplicationVersion(__version__)
    app.setOrganizationName("EMBL-IC")
    app.setWindowIcon(QIcon(str(resource_path("icons", "minflux_viewer_logo.png"))))

    from .core.app_state import AppState
    from .ui.main_window import MainWindow

    state  = AppState()
    window = MainWindow(state)
    window.show()

    # Allow passing supported data files as CLI arguments
    for arg in sys.argv[1:]:
        if arg.lower().endswith((".mat", ".npy", ".csv", ".msr")):
            window._route_file(arg)

    sys.exit(app.exec())


if __name__ == "__main__":
    main()
