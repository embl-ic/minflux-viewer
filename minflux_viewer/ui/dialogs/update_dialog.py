"""
Update-check result presentation (Tier-A in-app updater).

The actual version check lives in :mod:`minflux_viewer.core.updater`; this
renders the outcome and, on frozen Windows/Linux builds with configured trust,
offers *Download & Install* that authenticates a signed release and hands off to the
:mod:`minflux_viewer.core.self_update` helper (swap-after-exit + relaunch). In
*silent* mode (the on-startup check) nothing is shown unless an update exists.
"""

from __future__ import annotations

from html import escape

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

from minflux_viewer.core import self_update, signed_updates
from minflux_viewer.core.updater import (
    RELEASES_PAGE_URL,
    UpdateCheckResult,
    select_asset_for_platform,
)
from minflux_viewer.ui.common.background_tasks import (
    BackgroundTask,
    BackgroundTaskCancelled,
    retain_qthread,
    retire_background_tasks,
    retire_qthreads,
    shared_thread_pool,
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


class _DownloadCancelled(Exception):
    """Cooperative-cancellation checkpoint raised from the progress callback."""


class _ReleaseNotesBrowser(QTextBrowser):
    """Release metadata is untrusted: never resolve embedded local resources."""

    def loadResource(self, resource_type, url):
        return None  # no remote/local images or styles; formatting still works


class _DownloadWorker(QThread):
    """Authenticates and streams a release into owned storage off the UI thread.

    The thread is deliberately **parentless** and retained by the process-level
    registry (see ``ui/common/background_tasks.py``): a ``QThread`` owned by the dialog
    is destroyed with it, and Qt aborts the process when that happens while the
    download is still running. Closing the dialog requests interruption instead;
    :func:`~minflux_viewer.core.updater.download_url` calls ``progress`` once per
    64 KiB block, so that callback is the cancellation checkpoint.
    """

    progress = pyqtSignal(object, object)    # Python ints; archives may exceed 2 GiB
    finished_ok = pyqtSignal(object)         # owned authenticated download
    failed = pyqtSignal(str)

    def __init__(self, tag: str, current_version: str) -> None:
        super().__init__()  # always parentless, retained until inherited finished
        self._tag = tag
        self._current_version = current_version
        self._last_progress = 0

    def _checkpoint(self) -> None:
        if self.isInterruptionRequested():
            raise _DownloadCancelled

    def _report(self, done: int, total: int) -> None:
        self._checkpoint()
        if done == 0 or done == total or done - self._last_progress >= 1 << 20:
            self._last_progress = done
            self.progress.emit(int(done), int(total))

    def run(self) -> None:
        try:
            download = signed_updates.download_release(
                self._tag, self._current_version, progress=self._report,
                checkpoint=self._checkpoint)
            delivered = False
            try:
                self._checkpoint()
                self.finished_ok.emit(download)
                delivered = True
            finally:
                if not delivered:
                    download.cleanup()
        except _DownloadCancelled:
            return  # the authenticated downloader cleans partial data itself
        except Exception as exc:                                  # noqa: BLE001
            self.failed.emit(str(exc))


class _UpdateAvailableDialog(QDialog):
    """Shows release metadata; automatic installation requires pinned trust."""

    def __init__(self, result: UpdateCheckResult, parent=None, *, state=None) -> None:
        super().__init__(parent)
        self._result = result
        self._state = state
        rel = result.latest
        self._asset = select_asset_for_platform(rel.assets)
        try:
            manual_asset_url = (signed_updates.release_asset_url(rel.tag, self._asset.name)
                                if self._asset is not None else None)
        except ValueError:
            manual_asset_url = None  # invalid API hints never become external URLs
        self._worker: _DownloadWorker | None = None
        self._downloaded: signed_updates.DownloadedUpdate | None = None
        self._prepare_task: BackgroundTask | None = None
        self._closing = False

        self.setWindowTitle("Update available")
        self.setMinimumSize(540, 460)

        root = QVBoxLayout(self)
        head = QLabel(
            "<h3>A new version is available</h3>"
            f"<p>Installed: <b>v{escape(result.current_version)}</b> &nbsp;→&nbsp; "
            f"Latest: <b>{escape(rel.tag or ('v' + rel.version))}</b>"
            + (f" &nbsp;·&nbsp; {escape(rel.published_at[:10])}" if rel.published_at else "")
            + "</p>"
        )
        head.setTextFormat(Qt.TextFormat.RichText)
        head.setWordWrap(True)
        root.addWidget(head)

        notes = _ReleaseNotesBrowser()
        notes.setOpenLinks(False)
        notes.setOpenExternalLinks(False)
        notes.anchorClicked.connect(self._open_notes_link)
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
        # API asset hints are for manual links only. Installation selects its
        # exact platform/architecture from the authenticated manifest instead.
        self._install_btn = None
        if self_update.can_self_update() and signed_updates.trust_configured():
            self._install_btn = self._buttons.addButton(
                "Download && Install", QDialogButtonBox.ButtonRole.AcceptRole)
            self._install_btn.clicked.connect(self._start_install)
        elif manual_asset_url is not None:
            dl = self._buttons.addButton(
                f"Download {self._asset.name}", QDialogButtonBox.ButtonRole.ActionRole)
            dl.clicked.connect(lambda: QDesktopServices.openUrl(QUrl(manual_asset_url)))

        self._page_btn = self._buttons.addButton(
            "Open download page", QDialogButtonBox.ButtonRole.ActionRole)
        self._page_btn.clicked.connect(
            lambda: QDesktopServices.openUrl(QUrl(RELEASES_PAGE_URL)))
        self._close_btn = self._buttons.addButton(
            "Close", QDialogButtonBox.ButtonRole.RejectRole)
        self._close_btn.clicked.connect(self.reject)
        root.addWidget(self._buttons)
        if self_update.can_self_update() and not signed_updates.trust_configured():
            self._status.setText(
                "Automatic installation is unavailable: no trusted release signing key "
                "is configured in this build. Manual downloads remain available.")
            self._status.show()

    def _open_notes_link(self, url: QUrl) -> None:
        if url.scheme().lower() == "https" and url.host():
            QDesktopServices.openUrl(url)
        else:
            self._status.setText("Only HTTPS links can be opened from release notes.")
            self._status.show()

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

    def closeEvent(self, event) -> None:
        """Detach a running download instead of destroying its thread.

        Never waits: the request is cooperative and takes effect at the next
        64 KiB block, while the process-level registry keeps the thread alive
        until its inherited ``finished`` signal fires.
        """
        self._retire_work()
        super().closeEvent(event)

    def done(self, result: int) -> None:
        # Esc and the Close button reject the QDialog without necessarily
        # delivering closeEvent. They must retire work too.
        self._retire_work()
        super().done(result)

    def _retire_work(self) -> None:
        self._closing = True
        if self._worker is not None:
            retire_qthreads(
                [self._worker],
                signal_names=("progress", "finished_ok", "failed"),
            )
            self._worker = None
        if self._prepare_task is not None:
            retire_background_tasks([self._prepare_task])
            self._prepare_task = None
        if self._downloaded is not None:
            self._downloaded.cleanup()
            self._downloaded = None

    # -- one-click install ---------------------------------------------------
    def _start_install(self) -> None:
        if (self._closing or self._worker is not None or self._prepare_task is not None
                or not self_update.can_self_update() or not signed_updates.trust_configured()):
            return
        release = self._result.latest
        if release is None:
            return
        # Already fetched this session (user picked "Later") → skip re-download.
        if self._downloaded is not None:
            self._prompt_restart()
            return
        if self._install_btn is not None:
            self._install_btn.setEnabled(False)
        self._page_btn.setEnabled(False)
        self._status.setText("Authenticating release manifest and downloading update…")
        self._status.show()
        self._progress.setRange(0, 0)          # busy until first progress tick
        self._progress.show()

        # Parentless: a QThread destroyed as a child of this dialog while the
        # download is running aborts the process.
        self._worker = _DownloadWorker(release.tag, self._result.current_version)
        retain_qthread(self._worker)
        self._worker.progress.connect(self._on_progress)
        self._worker.finished_ok.connect(self._on_downloaded)
        self._worker.failed.connect(self._on_failed)
        self._worker.start()

    def _on_progress(self, done: int, total: int) -> None:
        if self._closing:
            return
        if total > 0:
            self._progress.setRange(0, 1000)
            self._progress.setValue(min(1000, done * 1000 // total))
            mb = 1024 * 1024
            self._status.setText(
                f"Downloading authenticated update…  {done / mb:.0f} / {total / mb:.0f} MB")
        else:
            self._progress.setRange(0, 0)

    def _on_downloaded(self, download: signed_updates.DownloadedUpdate) -> None:
        if self._closing:
            download.cleanup()
            return
        self._worker = None
        self._downloaded = download
        self._progress.hide()
        self._prompt_restart()

    def _prompt_restart(self) -> None:
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
        if self._closing:
            return
        if box.clickedButton() is restart_btn:
            self._apply_and_quit()
            return
        # Deferred: keep the downloaded file; offer to apply without re-downloading.
        self._status.setText(
            f"{version} downloaded — click “Install && Restart” to apply when ready.")
        self._status.show()
        if self._install_btn is not None:
            self._install_btn.setText("Install && Restart")
            self._install_btn.setEnabled(True)
        self._page_btn.setEnabled(True)

    def _apply_and_quit(self) -> None:
        if self._closing or self._downloaded is None or self._prepare_task is not None:
            return
        download = self._downloaded
        self._downloaded = None  # the worker now owns it until its finally callback
        self._status.setText("Reverifying and staging update…")
        self._status.show()
        self._progress.setRange(0, 0)
        self._progress.show()
        if self._install_btn is not None:
            self._install_btn.setEnabled(False)
        self._page_btn.setEnabled(False)

        def work(report):
            report("Reverifying and staging update")

            def checkpoint():
                if task.is_cancelled:
                    raise BackgroundTaskCancelled

            return signed_updates.prepare_download(
                download, checkpoint=checkpoint)

        task = BackgroundTask(work, description="Prepare authenticated update",
                              discard_result=self_update.discard_prepared_update,
                              finally_callback=download.cleanup)
        self._prepare_task = task
        task.signals.done.connect(lambda prepared: self._on_prepared(task, prepared, download))
        task.signals.failed.connect(lambda message: self._on_prepare_failed(task, message))
        task.signals.cancelled.connect(lambda: self._on_prepare_failed(task, "Update preparation cancelled."))
        shared_thread_pool("file-io").start(task)

    def _on_prepare_failed(self, task, message: str) -> None:
        if not self._closing and self._prepare_task is task:
            self._prepare_task = None
            self._on_failed(message)

    def _on_prepared(self, task, prepared: self_update.PreparedUpdate,
                     download: signed_updates.DownloadedUpdate) -> None:
        if self._closing or self._prepare_task is not task:
            self_update.discard_prepared_update(prepared)
            return
        self._prepare_task = None
        try:
            # Small, bounded authentication only: catches expiry or changed
            # trust while the completed worker waited for GUI delivery.
            signed_updates.verify_manifest(
                download.manifest_bytes, download.signature,
                tag=download.tag, current_version=download.current_version)
            self_update.launch_prepared_update(prepared)
        except Exception as exc:                                  # noqa: BLE001
            self_update.discard_prepared_update(prepared)
            self._on_failed(f"Could not start the installer: {exc}")
            return
        # Hand-off launched; quit so the helper can replace the locked files.
        self.accept()
        app = QApplication.instance()
        if app is not None:
            app.quit()

    def _on_failed(self, message: str) -> None:
        if self._closing:
            return
        self._worker = None
        self._progress.hide()
        self._status.hide()
        if self._install_btn is not None:
            self._install_btn.setEnabled(True)
        self._page_btn.setEnabled(True)
        QMessageBox.warning(
            self, "Update",
            f"The update could not be downloaded or installed:\n\n{message}\n\n"
            f"You can download it manually from:\n{RELEASES_PAGE_URL}")
