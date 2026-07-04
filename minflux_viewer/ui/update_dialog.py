"""
Update-check result presentation (Tier-A in-app updater).

The actual version check lives in :mod:`minflux_viewer.core.updater`; this
renders the outcome and, on the **frozen Windows** build, offers a one-click
*Download & Install* that downloads the release asset and hands off to the
:mod:`minflux_viewer.core.self_update` helper (swap-after-exit + relaunch). In
*silent* mode (the on-startup check) nothing is shown unless an update exists.
"""

from __future__ import annotations

import tempfile
from pathlib import Path

from PyQt6.QtCore import Qt, QThread, QUrl, pyqtSignal
from PyQt6.QtGui import QDesktopServices
from PyQt6.QtWidgets import (
    QApplication,
    QCheckBox,
    QDialog,
    QDialogButtonBox,
    QLabel,
    QMessageBox,
    QProgressBar,
    QTextBrowser,
    QVBoxLayout,
)

from ..core import self_update
from ..core.updater import (
    RELEASES_PAGE_URL,
    UpdateCheckResult,
    download_asset,
    select_asset_for_platform,
)


def show_update_result(result: UpdateCheckResult, parent=None, *, silent: bool = False,
                       state=None) -> None:
    """Present an :class:`UpdateCheckResult`.

    ``silent`` (startup check) suppresses the up-to-date and error cases so the
    user is only interrupted when a newer release exists. ``state`` (AppState)
    lets the "check on startup" preference be toggled from the dialog.
    """
    if result.update_available and result.latest is not None:
        _UpdateAvailableDialog(result, parent, state=state).exec()
    elif silent:
        return
    elif result.status == "up_to_date":
        QMessageBox.information(
            parent, "Check for Updates",
            f"You are running the latest version (v{result.current_version}).",
        )
    else:
        QMessageBox.warning(
            parent, "Check for Updates",
            "Could not check for updates.\n\n"
            f"{result.error or 'Unknown error.'}\n\n"
            f"You can check manually at:\n{RELEASES_PAGE_URL}",
        )


class _DownloadWorker(QThread):
    """Streams a release asset to *dest* off the UI thread."""

    progress = pyqtSignal(int, int)          # (bytes_done, bytes_total)
    finished_ok = pyqtSignal(str)            # dest path
    failed = pyqtSignal(str)

    def __init__(self, asset, dest, parent=None) -> None:
        super().__init__(parent)
        self._asset = asset
        self._dest = dest

    def run(self) -> None:
        try:
            download_asset(self._asset, self._dest,
                           progress=lambda d, t: self.progress.emit(int(d), int(t)))
            self.finished_ok.emit(str(self._dest))
        except Exception as exc:                                  # noqa: BLE001
            self.failed.emit(str(exc))


class _UpdateAvailableDialog(QDialog):
    """Shows the new version + notes; offers one-click install (frozen Windows)."""

    def __init__(self, result: UpdateCheckResult, parent=None, *, state=None) -> None:
        super().__init__(parent)
        self._result = result
        self._state = state
        rel = result.latest
        self._asset = select_asset_for_platform(rel.assets)
        self._worker: _DownloadWorker | None = None
        self._downloaded_path: str | None = None       # set once a verified zip is fetched

        self.setWindowTitle("Update available")
        self.setMinimumSize(540, 460)

        root = QVBoxLayout(self)
        head = QLabel(
            "<h3>A new version is available</h3>"
            f"<p>Installed: <b>v{result.current_version}</b> &nbsp;→&nbsp; "
            f"Latest: <b>{rel.tag or ('v' + rel.version)}</b>"
            + (f" &nbsp;·&nbsp; {rel.published_at[:10]}" if rel.published_at else "")
            + "</p>"
        )
        head.setTextFormat(Qt.TextFormat.RichText)
        head.setWordWrap(True)
        root.addWidget(head)

        notes = QTextBrowser()
        notes.setOpenExternalLinks(True)
        if rel.notes.strip():
            try:
                notes.setMarkdown(rel.notes)
            except Exception:
                notes.setPlainText(rel.notes)
        else:
            notes.setPlainText("(No release notes provided.)")
        root.addWidget(notes, 1)

        self._status = QLabel("")
        self._status.setWordWrap(True)
        self._status.hide()
        root.addWidget(self._status)

        self._progress = QProgressBar()
        self._progress.hide()
        root.addWidget(self._progress)

        # Opt-in toggle for the on-startup check (same preference as Preferences ›
        # File). Off by default: the app only reaches out to GitHub on startup if
        # the user explicitly enables it here (or in Preferences).
        self._auto_check = QCheckBox("Automatically check for updates on startup")
        self._auto_check.setToolTip(
            "When on, MINFLUX Viewer checks the GitHub releases page for a newer "
            "version at startup (a single request; no data is sent). Off by default.")
        self._auto_check.setChecked(self._auto_check_pref())
        self._auto_check.toggled.connect(self._on_auto_check_toggled)
        root.addWidget(self._auto_check)

        self._buttons = QDialogButtonBox()
        # One-click install is only possible for the frozen Windows build with a
        # matching downloadable asset; otherwise fall back to the browser links.
        self._install_btn = None
        if self._asset is not None and self_update.can_self_update():
            self._install_btn = self._buttons.addButton(
                "Download && Install", QDialogButtonBox.ButtonRole.AcceptRole)
            self._install_btn.clicked.connect(self._start_install)
        elif self._asset is not None:
            dl = self._buttons.addButton(
                f"Download {self._asset.name}", QDialogButtonBox.ButtonRole.ActionRole)
            dl.clicked.connect(lambda: QDesktopServices.openUrl(QUrl(self._asset.url)))

        self._page_btn = self._buttons.addButton(
            "Open download page", QDialogButtonBox.ButtonRole.ActionRole)
        self._page_btn.clicked.connect(
            lambda: QDesktopServices.openUrl(QUrl(rel.html_url or RELEASES_PAGE_URL)))
        self._close_btn = self._buttons.addButton(
            "Close", QDialogButtonBox.ButtonRole.RejectRole)
        self._close_btn.clicked.connect(self.reject)
        root.addWidget(self._buttons)

    # -- on-startup check preference -----------------------------------------
    def _auto_check_pref(self) -> bool:
        if self._state is None:
            return False
        return bool(self._state.prefs.get("file", {}).get("check_updates_on_startup", False))

    def _on_auto_check_toggled(self, checked: bool) -> None:
        if self._state is None:
            return
        self._state.prefs.setdefault("file", {})["check_updates_on_startup"] = bool(checked)
        try:
            self._state.save_prefs()
        except Exception:
            pass

    # -- one-click install ---------------------------------------------------
    def _start_install(self) -> None:
        if self._asset is None:
            return
        # Already fetched this session (user picked "Later") → skip re-download.
        if self._downloaded_path and Path(self._downloaded_path).is_file():
            self._prompt_restart(self._downloaded_path)
            return
        dest = Path(tempfile.gettempdir()) / self._asset.name
        if self._install_btn is not None:
            self._install_btn.setEnabled(False)
        self._page_btn.setEnabled(False)
        self._status.setText(f"Downloading {self._asset.name}…")
        self._status.show()
        self._progress.setRange(0, 0)          # busy until first progress tick
        self._progress.show()

        self._worker = _DownloadWorker(self._asset, dest, self)
        self._worker.progress.connect(self._on_progress)
        self._worker.finished_ok.connect(self._on_downloaded)
        self._worker.failed.connect(self._on_failed)
        self._worker.start()

    def _on_progress(self, done: int, total: int) -> None:
        if total > 0:
            self._progress.setRange(0, total)
            self._progress.setValue(done)
            mb = 1024 * 1024
            self._status.setText(
                f"Downloading {self._asset.name}…  {done / mb:.0f} / {total / mb:.0f} MB")
        else:
            self._progress.setRange(0, 0)

    def _on_downloaded(self, path: str) -> None:
        self._downloaded_path = path
        self._progress.hide()
        self._prompt_restart(path)

    def _prompt_restart(self, path: str) -> None:
        """Fiji-style: confirm before restarting to apply the downloaded update."""
        rel = self._result.latest
        version = rel.tag or ("v" + rel.version)
        box = QMessageBox(self)
        box.setWindowTitle("Update ready")
        box.setIcon(QMessageBox.Icon.Question)
        box.setText(f"{version} has been downloaded.")
        box.setInformativeText(
            "MINFLUX Viewer needs to close and reopen to apply the update "
            "(the files are replaced in place). Restart now?")
        restart_btn = box.addButton("Restart now", QMessageBox.ButtonRole.AcceptRole)
        box.addButton("Later", QMessageBox.ButtonRole.RejectRole)
        box.setDefaultButton(restart_btn)
        box.exec()
        if box.clickedButton() is restart_btn:
            self._apply_and_quit(path)
            return
        # Deferred: keep the downloaded file; offer to apply without re-downloading.
        self._status.setText(
            f"{version} downloaded — click “Install && Restart” to apply when ready.")
        self._status.show()
        if self._install_btn is not None:
            self._install_btn.setText("Install && Restart")
            self._install_btn.setEnabled(True)
        self._page_btn.setEnabled(True)

    def _apply_and_quit(self, path: str) -> None:
        self._status.setText("Installing… the application will close and reopen.")
        self._status.show()
        self._progress.setRange(0, 0)
        self._progress.show()
        QApplication.processEvents()
        try:
            self_update.apply_update(path)
        except Exception as exc:                                  # noqa: BLE001
            self._on_failed(f"Could not start the installer: {exc}")
            return
        # Hand-off launched; quit so the helper can replace the locked files.
        self.accept()
        app = QApplication.instance()
        if app is not None:
            app.quit()

    def _on_failed(self, message: str) -> None:
        self._progress.hide()
        self._status.hide()
        if self._install_btn is not None:
            self._install_btn.setEnabled(True)
        self._page_btn.setEnabled(True)
        QMessageBox.warning(
            self, "Update",
            f"The update could not be downloaded or installed:\n\n{message}\n\n"
            f"You can download it manually from:\n{RELEASES_PAGE_URL}")
