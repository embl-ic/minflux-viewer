"""
minflux_viewer.ui.common.file_manager
==============================
Reveal a file in the operating system's file manager.

Shared by *Open file location* (Dataset Manager, Open Recent) and by the figure
export's *Open saved folder when completed*.
"""

from __future__ import annotations

import subprocess
import sys
from pathlib import Path


def reveal_in_file_manager(path) -> None:
    """Show *path* selected in the file manager; falls back to opening its folder.

    Cross-platform: Explorer on Windows, Finder on macOS, the desktop's default
    file manager elsewhere (folder only — there is no portable "select").
    """
    from PyQt6.QtCore import QUrl
    from PyQt6.QtGui import QDesktopServices

    p = Path(path)
    try:
        if sys.platform.startswith("win"):
            # Keep the switch separate from the path. If they are one argument,
            # subprocess quotes the whole string when the path contains spaces
            # ("/select,C:\\some path\\file"), and Explorer then treats it as a
            # location instead of as /select.
            subprocess.Popen(["explorer.exe", "/select,", str(p)])
            return
        if sys.platform == "darwin":
            subprocess.Popen(["open", "-R", str(p)])
            return
    except Exception:
        pass
    QDesktopServices.openUrl(QUrl.fromLocalFile(str(p.parent)))
