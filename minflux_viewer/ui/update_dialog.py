"""
Update-check result presentation (Tier-A in-app updater).

The actual version check lives in :mod:`minflux_viewer.core.updater`; this only
renders the outcome. In *silent* mode (the on-startup check) nothing is shown
unless an update is actually available.
"""

from __future__ import annotations

from PyQt6.QtCore import Qt, QUrl
from PyQt6.QtGui import QDesktopServices
from PyQt6.QtWidgets import (
    QDialog,
    QDialogButtonBox,
    QLabel,
    QMessageBox,
    QTextBrowser,
    QVBoxLayout,
)

from ..core.updater import (
    RELEASES_PAGE_URL,
    UpdateCheckResult,
    select_asset_for_platform,
)


def show_update_result(result: UpdateCheckResult, parent=None, *, silent: bool = False) -> None:
    """Present an :class:`UpdateCheckResult`.

    ``silent`` (startup check) suppresses the up-to-date and error cases so the
    user is only interrupted when a newer release exists.
    """
    if result.update_available and result.latest is not None:
        _UpdateAvailableDialog(result, parent).exec()
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


class _UpdateAvailableDialog(QDialog):
    """Shows the new version, release notes, and download links."""

    def __init__(self, result: UpdateCheckResult, parent=None) -> None:
        super().__init__(parent)
        rel = result.latest
        self.setWindowTitle("Update available")
        self.setMinimumSize(540, 440)

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

        buttons = QDialogButtonBox()
        page_btn = buttons.addButton("Open download page", QDialogButtonBox.ButtonRole.ActionRole)
        page_btn.clicked.connect(
            lambda: QDesktopServices.openUrl(QUrl(rel.html_url or RELEASES_PAGE_URL))
        )
        asset = select_asset_for_platform(rel.assets)
        if asset is not None:
            asset_btn = buttons.addButton(
                f"Download {asset.name}", QDialogButtonBox.ButtonRole.ActionRole
            )
            asset_btn.clicked.connect(lambda: QDesktopServices.openUrl(QUrl(asset.url)))
        close_btn = buttons.addButton("Close", QDialogButtonBox.ButtonRole.RejectRole)
        close_btn.clicked.connect(self.reject)
        root.addWidget(buttons)
