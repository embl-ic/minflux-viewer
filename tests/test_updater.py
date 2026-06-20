"""Tier-A update check (core.updater) — version compare + GitHub API parsing."""

from __future__ import annotations

import io
import json
import urllib.error

import pytest

from minflux_viewer.core import updater as U


# ---------------------------------------------------------------------------
# version comparison
# ---------------------------------------------------------------------------
@pytest.mark.parametrize("latest,current,expected", [
    ("0.3.1", "0.3.0", True),
    ("v0.3.1", "0.3.0", True),     # tolerates leading 'v'
    ("0.3.0", "0.3.0", False),
    ("0.2.9", "0.3.0", False),
    ("1.0.0", "0.9.9", True),
    ("0.3.0", "0.3.0a1", True),    # release newer than prerelease
])
def test_is_newer(latest, current, expected):
    assert U.is_newer(latest, current) is expected


def test_normalize_strips_v():
    assert U._normalize("v0.3.1") == "0.3.1"
    assert U._normalize("  V1.2.3 ") == "1.2.3"


# ---------------------------------------------------------------------------
# fetch / check (network mocked)
# ---------------------------------------------------------------------------
class _FakeResp(io.BytesIO):
    def __enter__(self):
        return self

    def __exit__(self, *a):
        self.close()
        return False


def _patch_urlopen(monkeypatch, payload):
    def fake(req, timeout=0):
        return _FakeResp(json.dumps(payload).encode("utf-8"))
    monkeypatch.setattr(U.urllib.request, "urlopen", fake)


_RELEASE = {
    "tag_name": "v0.3.1",
    "name": "v0.3.1",
    "body": "## What's new\n- thing",
    "html_url": "https://github.com/EMBL-ICLM/minflux-viewer/releases/tag/v0.3.1",
    "published_at": "2026-06-16T00:00:00Z",
    "assets": [
        {"name": "minflux_viewer-0.3.1-windows.zip",
         "browser_download_url": "https://example/win.zip", "size": 123},
        {"name": "minflux_viewer-0.3.1-linux.AppImage",
         "browser_download_url": "https://example/linux.AppImage", "size": 456},
    ],
}


def test_check_update_available(monkeypatch):
    _patch_urlopen(monkeypatch, _RELEASE)
    res = U.check_for_update("0.3.0")
    assert res.status == "update_available" and res.update_available
    assert res.latest.tag == "v0.3.1"
    assert res.latest.version == "0.3.1"
    assert len(res.latest.assets) == 2


def test_check_up_to_date(monkeypatch):
    _patch_urlopen(monkeypatch, _RELEASE)
    res = U.check_for_update("0.3.1")
    assert res.status == "up_to_date" and not res.update_available


def test_check_404_is_up_to_date(monkeypatch):
    def fake(req, timeout=0):
        raise urllib.error.HTTPError(req.full_url, 404, "Not Found", {}, None)
    monkeypatch.setattr(U.urllib.request, "urlopen", fake)
    res = U.check_for_update("0.3.0")
    assert res.status == "up_to_date"


def test_check_network_error_is_error(monkeypatch):
    def fake(req, timeout=0):
        raise urllib.error.URLError("offline")
    monkeypatch.setattr(U.urllib.request, "urlopen", fake)
    res = U.check_for_update("0.3.0")
    assert res.status == "error" and res.latest is None and res.error


# ---------------------------------------------------------------------------
# platform asset selection
# ---------------------------------------------------------------------------
def test_select_asset_windows():
    items = (
        U.ReleaseAsset("app-windows.zip", "u1"),
        U.ReleaseAsset("app-linux.AppImage", "u2"),
        U.ReleaseAsset("app-macos.dmg", "u3"),
    )
    win = U.select_asset_for_platform(items, system="Windows", machine="AMD64")
    assert win.name == "app-windows.zip"
    lin = U.select_asset_for_platform(items, system="Linux", machine="x86_64")
    assert lin.name == "app-linux.AppImage"
    mac = U.select_asset_for_platform(items, system="Darwin", machine="arm64")
    assert mac.name == "app-macos.dmg"


def test_select_asset_none_when_no_match():
    items = (U.ReleaseAsset("readme.txt", "u"),)
    assert U.select_asset_for_platform(items, system="Windows", machine="AMD64") is None
