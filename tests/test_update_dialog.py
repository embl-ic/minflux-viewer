"""Authenticated update UI boundaries; no network, restart or real install."""

from __future__ import annotations

from types import SimpleNamespace

import pytest

from minflux_viewer.core.updater import ReleaseAsset, ReleaseInfo, UpdateCheckResult
from minflux_viewer.ui.dialogs import update_dialog as ui


@pytest.fixture
def result():
    return UpdateCheckResult("update_available", "0.4.4", ReleaseInfo(
        "9.9.9", "v9.9.9", "Test", "Test notes", "https://github.com/embl-ic/minflux-viewer/releases",
        assets=(ReleaseAsset("app-win.zip", "https://example.invalid/app.zip"),)))


@pytest.fixture
def trusted(monkeypatch):
    # Public test pin only. No production private/public trust material.
    monkeypatch.setattr(ui.signed_updates, "TRUSTED_RELEASE_KEYS", ("00" * 32,))
    monkeypatch.setattr(ui.self_update, "can_self_update", lambda: True)


def test_unconfigured_trust_has_manual_links_but_no_installer(qtbot, result, monkeypatch):
    monkeypatch.setattr(ui.signed_updates, "TRUSTED_RELEASE_KEYS", ())
    monkeypatch.setattr(ui.self_update, "can_self_update", lambda: True)
    dialog = ui._UpdateAvailableDialog(result)
    qtbot.addWidget(dialog)
    assert dialog._install_btn is None
    assert "no trusted" in dialog._status.text()
    dialog._start_install()
    assert dialog._worker is None
    dialog.close()


def test_signed_installer_does_not_depend_on_api_asset_heuristics(qtbot, trusted):
    result = UpdateCheckResult("update_available", "0.4.4", ReleaseInfo(
        "9.9.9", "v9.9.9", "Test", "", "", assets=()))
    dialog = ui._UpdateAvailableDialog(result)
    qtbot.addWidget(dialog)
    assert dialog._asset is None
    assert dialog._install_btn is not None
    dialog.close()


def test_download_worker_uses_authentication_and_keeps_python_sized_progress(qtbot, trusted, monkeypatch):
    calls = []
    download = SimpleNamespace(cleanup=lambda: calls.append("cleanup"))
    monkeypatch.setattr(ui.signed_updates, "download_release",
                        lambda tag, current, **kwargs: calls.append((tag, current)) or download)
    worker = ui._DownloadWorker("v9.9.9", "0.4.4")
    assert worker.parent() is None
    seen = []
    progress = []
    worker.finished_ok.connect(seen.append)
    worker.progress.connect(lambda done, total: progress.append((done, total)))
    worker.run()
    worker._report(1 << 34, 1 << 35)
    assert calls == [("v9.9.9", "0.4.4")]
    assert seen == [download]
    assert progress == [(1 << 34, 1 << 35)]


def test_large_archive_progress_does_not_overflow_qt(qtbot, result, trusted):
    dialog = ui._UpdateAvailableDialog(result)
    qtbot.addWidget(dialog)
    dialog._on_progress(1 << 34, 1 << 35)
    assert dialog._progress.value() == 500
    dialog.close()


def test_preparation_failure_never_launches_and_cleans_download(qtbot, result, trusted, monkeypatch):
    calls = []
    monkeypatch.setattr(ui.QMessageBox, "warning", lambda *args: calls.append("warning"))
    monkeypatch.setattr(ui.self_update, "launch_prepared_update", lambda *args: pytest.fail("launched"))

    def fail(download, **kwargs):
        raise ValueError("expired or changed archive")

    monkeypatch.setattr(ui.signed_updates, "prepare_download", fail)
    dialog = ui._UpdateAvailableDialog(result)
    qtbot.addWidget(dialog)
    dialog._downloaded = SimpleNamespace(cleanup=lambda: calls.append("cleanup"))
    dialog._apply_and_quit()
    qtbot.waitUntil(lambda: "warning" in calls)
    qtbot.waitUntil(lambda: "cleanup" in calls)
    assert dialog._prepare_task is None
    assert dialog._downloaded is None
    assert dialog._install_btn.isEnabled()
    dialog.close()


def test_preparation_is_async_single_flight_then_authenticates_before_handoff(qtbot, result, trusted, monkeypatch):
    import threading

    calls = []
    gui_thread = threading.get_ident()
    prepared = object()

    def prepare(download, **kwargs):
        assert threading.get_ident() != gui_thread
        calls.append("prepare")
        return prepared

    def verify(*args, **kwargs):
        assert threading.get_ident() == gui_thread
        calls.append("verify-at-handoff")

    monkeypatch.setattr(ui.signed_updates, "prepare_download", prepare)
    monkeypatch.setattr(ui.signed_updates, "verify_manifest", verify)
    monkeypatch.setattr(ui.self_update, "launch_prepared_update", lambda value: calls.append(("launch", value)))
    monkeypatch.setattr(ui.QApplication.instance(), "quit", lambda: calls.append("quit"))
    dialog = ui._UpdateAvailableDialog(result)
    qtbot.addWidget(dialog)
    dialog._downloaded = SimpleNamespace(cleanup=lambda: calls.append("cleanup"),
                                         manifest_bytes=b"test", signature=b"test",
                                         tag="v9.9.9", current_version="0.4.4")
    dialog._apply_and_quit()
    dialog._apply_and_quit()  # a second click must not submit another task
    qtbot.waitUntil(lambda: "quit" in calls)
    assert calls.count("prepare") == 1
    assert calls.index("verify-at-handoff") < calls.index(("launch", prepared)) < calls.index("quit")
    qtbot.waitUntil(lambda: "cleanup" in calls)
    assert dialog._closing


def test_stale_prepared_result_is_discarded_not_installed(qtbot, result, trusted, monkeypatch):
    discarded = []
    monkeypatch.setattr(ui.self_update, "discard_prepared_update", discarded.append)
    monkeypatch.setattr(ui.self_update, "launch_prepared_update", lambda *args: pytest.fail("stale launch"))
    dialog = ui._UpdateAvailableDialog(result)
    qtbot.addWidget(dialog)
    dialog._prepare_task = object()
    result_object = object()
    dialog._on_prepared(object(), result_object, None)
    assert discarded == [result_object]
    dialog._prepare_task = None
    dialog.close()


def test_expiry_while_waiting_for_gui_handoff_prevents_launch(qtbot, result, trusted, monkeypatch):
    calls = []

    def expired(*args, **kwargs):
        raise ValueError("expired while queued")

    monkeypatch.setattr(ui.signed_updates, "verify_manifest", expired)
    monkeypatch.setattr(ui.self_update, "launch_prepared_update", lambda *args: pytest.fail("expired launch"))
    monkeypatch.setattr(ui.self_update, "discard_prepared_update", lambda value: calls.append("discard"))
    monkeypatch.setattr(ui.QMessageBox, "warning", lambda *args: calls.append("warning"))
    dialog = ui._UpdateAvailableDialog(result)
    qtbot.addWidget(dialog)
    task = object()
    dialog._prepare_task = task
    download = SimpleNamespace(manifest_bytes=b"test", signature=b"test", tag="v9.9.9", current_version="0.4.4")
    dialog._on_prepared(task, object(), download)
    assert calls == ["discard", "warning"]
    dialog.close()


def test_manual_buttons_ignore_api_supplied_urls(qtbot, monkeypatch):
    from PyQt6.QtWidgets import QDialogButtonBox

    monkeypatch.setattr(ui.self_update, "can_self_update", lambda: False)
    opened = []
    monkeypatch.setattr(ui.QDesktopServices, "openUrl", lambda url: opened.append(url.toString()))
    result = UpdateCheckResult("update_available", "0.4.4", ReleaseInfo(
        "9.9.9", "v9.9.9", "Test", "", "file:///not-a-download-page",
        assets=(ReleaseAsset("app-win.zip", "file:///not-a-download"),)))
    # Keep the same explicit platform hint on every CI platform.
    monkeypatch.setattr(ui, "select_asset_for_platform", lambda assets: assets[0])
    dialog = ui._UpdateAvailableDialog(result)
    qtbot.addWidget(dialog)
    download_button = next(button for button in dialog._buttons.buttons()
                           if dialog._buttons.buttonRole(button) == QDialogButtonBox.ButtonRole.ActionRole
                           and button.text().startswith("Download "))
    download_button.click()
    dialog._page_btn.click()
    assert opened == [
        "https://github.com/embl-ic/minflux-viewer/releases/download/v9.9.9/app-win.zip",
        ui.RELEASES_PAGE_URL,
    ]
    dialog.close()


@pytest.mark.parametrize("url", ["file:///local", "javascript:alert(1)", "http://example.org", "relative"])
def test_release_notes_cannot_open_local_or_non_https_links(qtbot, result, monkeypatch, url):
    from PyQt6.QtCore import QUrl

    opened = []
    monkeypatch.setattr(ui.QDesktopServices, "openUrl", lambda url: opened.append(url))
    dialog = ui._UpdateAvailableDialog(result)
    qtbot.addWidget(dialog)
    dialog._open_notes_link(QUrl(url))
    assert not opened
    assert "HTTPS" in dialog._status.text()
    dialog.close()


def test_release_notes_allow_https_but_no_embedded_resources(qtbot, result, monkeypatch):
    from PyQt6.QtCore import QUrl
    from PyQt6.QtGui import QTextDocument

    opened = []
    monkeypatch.setattr(ui.QDesktopServices, "openUrl", lambda url: opened.append(url.toString()))
    dialog = ui._UpdateAvailableDialog(result)
    qtbot.addWidget(dialog)
    dialog._open_notes_link(QUrl("https://example.org/paper"))
    assert opened == ["https://example.org/paper"]
    notes = dialog.findChild(ui._ReleaseNotesBrowser)
    assert not notes.openLinks() and not notes.openExternalLinks()
    assert notes.loadResource(QTextDocument.ResourceType.ImageResource, QUrl("file:///local.png")) is None
    assert notes.loadResource(QTextDocument.ResourceType.ImageResource, QUrl("https://example.org/image.png")) is None
    dialog.close()


def test_release_headline_escapes_untrusted_html(qtbot):
    from PyQt6.QtWidgets import QLabel

    result = UpdateCheckResult("update_available", "0.4.4", ReleaseInfo(
        "9.9.9", 'v9.9.9<img src="file:///local">', "Test", "", ""))
    dialog = ui._UpdateAvailableDialog(result)
    qtbot.addWidget(dialog)
    headline = next(label.text() for label in dialog.findChildren(QLabel)
                    if "A new version is available" in label.text())
    assert "&lt;img" in headline and '<img src="file:' not in headline
    dialog.close()
