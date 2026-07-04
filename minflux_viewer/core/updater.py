"""
minflux_viewer.core.updater
============================
Tier-A in-app update check.

Queries the GitHub Releases API for the latest release and compares it to the
running version. **No self-update is performed** — this only reports whether a
newer release exists and where to download it. The check is Qt-free and
network-isolated so it can be unit-tested and run from a worker thread.
"""

from __future__ import annotations

import json
import platform
import urllib.error
import urllib.request
from dataclasses import dataclass

#: GitHub repository hosting the releases.
GITHUB_REPO = "EMBL-ICLM/minflux-viewer"
RELEASES_API_URL = f"https://api.github.com/repos/{GITHUB_REPO}/releases/latest"
RELEASES_PAGE_URL = f"https://github.com/{GITHUB_REPO}/releases/latest"

_TIMEOUT = 8.0


@dataclass(frozen=True)
class ReleaseAsset:
    name: str
    url: str
    size: int = 0


@dataclass(frozen=True)
class ReleaseInfo:
    version: str            # normalized, no leading 'v' (e.g. "0.3.1")
    tag: str                # raw tag (e.g. "v0.3.1")
    name: str
    notes: str              # release body (markdown)
    html_url: str
    published_at: str = ""
    assets: tuple[ReleaseAsset, ...] = ()


@dataclass(frozen=True)
class UpdateCheckResult:
    status: str             # "update_available" | "up_to_date" | "error"
    current_version: str
    latest: ReleaseInfo | None = None
    error: str | None = None

    @property
    def update_available(self) -> bool:
        return self.status == "update_available"


def _normalize(tag: str) -> str:
    """Strip a leading ``v``/``V`` and surrounding whitespace from a tag."""
    return (tag or "").strip().lstrip("vV")


def is_newer(latest: str, current: str) -> bool:
    """True if release version *latest* is strictly newer than *current*."""
    try:
        from packaging.version import parse as vparse
        return vparse(_normalize(latest)) > vparse(_normalize(current))
    except Exception:
        # Fallback: compare dot-separated integer components.
        def _t(s: str) -> tuple:
            return tuple(int(p) for p in _normalize(s).split(".") if p.isdigit())
        return _t(latest) > _t(current)


def fetch_latest_release(*, timeout: float = _TIMEOUT) -> ReleaseInfo:
    """Fetch the latest (non-prerelease, non-draft) release.

    Raises ``urllib.error.HTTPError`` / ``URLError`` on failure — callers
    typically use :func:`check_for_update`, which turns those into a result.
    """
    req = urllib.request.Request(
        RELEASES_API_URL,
        headers={
            "Accept": "application/vnd.github+json",
            "User-Agent": "minflux-viewer-updater",
        },
    )
    with urllib.request.urlopen(req, timeout=timeout) as resp:  # noqa: S310 (https only)
        data = json.loads(resp.read().decode("utf-8"))

    assets = tuple(
        ReleaseAsset(
            name=a.get("name", ""),
            url=a.get("browser_download_url", ""),
            size=int(a.get("size", 0) or 0),
        )
        for a in (data.get("assets") or [])
    )
    tag = data.get("tag_name", "") or ""
    return ReleaseInfo(
        version=_normalize(tag),
        tag=tag,
        name=data.get("name") or tag,
        notes=data.get("body") or "",
        html_url=data.get("html_url") or RELEASES_PAGE_URL,
        published_at=data.get("published_at") or "",
        assets=assets,
    )


def check_for_update(current_version: str, *, timeout: float = _TIMEOUT) -> UpdateCheckResult:
    """Compare the running version against the latest GitHub release.

    Never raises — network/HTTP problems are reported as a ``"error"`` result,
    and a missing release (HTTP 404, e.g. before the first release) is treated
    as up-to-date.
    """
    try:
        latest = fetch_latest_release(timeout=timeout)
    except urllib.error.HTTPError as exc:
        if exc.code == 404:
            return UpdateCheckResult("up_to_date", current_version, None)
        return UpdateCheckResult(
            "error", current_version, None, f"GitHub returned HTTP {exc.code}."
        )
    except urllib.error.URLError as exc:
        return UpdateCheckResult(
            "error", current_version, None, f"Network error: {exc.reason}"
        )
    except Exception as exc:  # malformed JSON, etc.
        return UpdateCheckResult("error", current_version, None, str(exc))

    if is_newer(latest.version, current_version):
        return UpdateCheckResult("update_available", current_version, latest)
    return UpdateCheckResult("up_to_date", current_version, latest)


def download_url(url: str, dest, *, progress=None, expected_size: int = 0,
                 timeout: float = 60.0, chunk: int = 1 << 16):
    """Stream *url* to *dest*, calling ``progress(done, total)`` as it goes.

    Follows redirects (GitHub asset URLs redirect to a signed CDN URL). When
    *expected_size* is given it is verified against the written byte count and a
    mismatch raises ``ValueError`` — so a truncated download is never applied.
    Returns the destination :class:`~pathlib.Path`.
    """
    from pathlib import Path

    dest = Path(dest)
    req = urllib.request.Request(
        url,
        headers={
            "Accept": "application/octet-stream",
            "User-Agent": "minflux-viewer-updater",
        },
    )
    dest.parent.mkdir(parents=True, exist_ok=True)
    done = 0
    with urllib.request.urlopen(req, timeout=timeout) as resp:  # noqa: S310
        total = int(resp.headers.get("Content-Length") or expected_size or 0)
        if progress:
            progress(0, total)
        with open(dest, "wb") as fh:
            while True:
                block = resp.read(chunk)
                if not block:
                    break
                fh.write(block)
                done += len(block)
                if progress:
                    progress(done, total)
    if expected_size and dest.stat().st_size != int(expected_size):
        raise ValueError(
            f"Downloaded size {dest.stat().st_size} != expected {expected_size} "
            "(incomplete download)."
        )
    return dest


def download_asset(asset: ReleaseAsset, dest, *, progress=None, timeout: float = 60.0):
    """Download a :class:`ReleaseAsset` to *dest*, verifying its known size."""
    return download_url(asset.url, dest, progress=progress,
                        expected_size=int(getattr(asset, "size", 0) or 0),
                        timeout=timeout)


def select_asset_for_platform(
    assets, *, system: str | None = None, machine: str | None = None
) -> ReleaseAsset | None:
    """Best-guess the download asset for the current OS/arch from filenames.

    Returns ``None`` when nothing matches (the caller falls back to the release
    page). Heuristic only — never blocks the version check.
    """
    system = (system or platform.system()).lower()       # 'windows'|'darwin'|'linux'
    machine = (machine or platform.machine()).lower()
    is_arm = machine in ("arm64", "aarch64")

    def score(name: str) -> int:
        n = name.lower()
        s = 0
        if system.startswith("win") and ("win" in n or n.endswith((".exe", ".msi", ".zip"))):
            s += 10 if ("win" in n) else 4
        if system == "darwin" and ("mac" in n or "osx" in n or "darwin" in n or n.endswith(".dmg")):
            s += 10
            s += 3 if (is_arm == any(t in n for t in ("arm", "m1", "apple", "silicon"))) else 0
        if system == "linux" and ("linux" in n or n.endswith((".appimage", ".tar.gz", ".tar.xz"))):
            s += 10
        return s

    best = max(assets, key=lambda a: score(a.name), default=None)
    return best if (best is not None and score(best.name) > 0) else None
