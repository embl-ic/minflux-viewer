"""Pinned-key authentication for executable updates (not a full TUF client).

Release discovery and GitHub's API checksums are not an authenticity boundary.
The installer uses only assets named and hashed by an Ed25519-signed manifest.
Trust is shipped in the application, never learned from a release or preference.
An empty key set disables automatic installation, not manual release discovery.
"""

from __future__ import annotations

import hashlib
import json
import platform
import re
import tempfile
import urllib.parse
import urllib.request
from collections.abc import Callable
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from pathlib import Path

from cryptography.exceptions import InvalidSignature
from cryptography.hazmat.primitives.asymmetric.ed25519 import Ed25519PublicKey
from packaging.version import Version

from minflux_viewer.core import self_update
from minflux_viewer.core.updater import GITHUB_REPO, download_url

# Raw, 32-byte Ed25519 public keys as hexadecimal. Configure out-of-band before
# shipping a trusted build. Private signing keys must never enter this project
# or the GitHub release/CI environment. Multiple pins permit deliberate rotation.
TRUSTED_RELEASE_KEYS: tuple[str, ...] = ()
MANIFEST_NAME = "release-manifest.json"
SIGNATURE_NAME = "release-manifest.sig"
MAX_MANIFEST_BYTES = 64 * 1024
MAX_ASSETS = 32
_MAX_CLOCK_SKEW = timedelta(minutes=5)


@dataclass(frozen=True)
class SignedAsset:
    name: str
    system: str
    arch: str
    size: int
    sha256: str


@dataclass(frozen=True)
class ReleaseManifest:
    tag: str
    issued_at: datetime
    expires_at: datetime
    assets: tuple[SignedAsset, ...]


@dataclass(frozen=True)
class DownloadedUpdate:
    """Owns a private temporary download and the evidence to reverify it."""

    directory: tempfile.TemporaryDirectory
    manifest_bytes: bytes
    signature: bytes
    tag: str
    current_version: str
    asset_name: str

    @property
    def archive_path(self) -> Path:
        return Path(self.directory.name) / self.asset_name

    def cleanup(self) -> None:
        self.directory.cleanup()


def _utc_now() -> datetime:
    return datetime.now(timezone.utc)


def _trusted_keys() -> tuple[Ed25519PublicKey, ...]:
    if not TRUSTED_RELEASE_KEYS or len(TRUSTED_RELEASE_KEYS) > 8:
        raise ValueError("No trusted release signing keys are configured.")
    keys = []
    for pin in TRUSTED_RELEASE_KEYS:
        if not isinstance(pin, str) or not re.fullmatch(r"[0-9a-fA-F]{64}", pin):
            raise ValueError("Invalid trusted release signing key configuration.")
        keys.append(Ed25519PublicKey.from_public_bytes(bytes.fromhex(pin)))
    return tuple(keys)


def trust_configured() -> bool:
    try:
        _trusted_keys()
    except ValueError:
        return False
    return True


def _tag_version(tag: str) -> Version:
    if not isinstance(tag, str) or not re.fullmatch(r"[vV]?[0-9][A-Za-z0-9.+-]{0,79}", tag):
        raise ValueError("Invalid release tag.")
    return Version(tag.removeprefix("v").removeprefix("V"))


def _unique_object(pairs: list[tuple[str, object]]) -> dict:
    value = {}
    for key, item in pairs:
        if key in value:
            raise ValueError(f"Duplicate manifest key: {key}")
        value[key] = item
    return value


def _timestamp(value: object) -> datetime:
    if not isinstance(value, str):
        raise ValueError("Manifest timestamps must be UTC strings.")
    parsed = datetime.fromisoformat(value.replace("Z", "+00:00"))
    if parsed.tzinfo is None or parsed.utcoffset() != timedelta(0):
        raise ValueError("Manifest timestamps must be UTC.")
    return parsed


def validate_asset_name(name: object) -> str:
    """Portable basenames only; no paths, reserved device names or URL syntax."""
    if not isinstance(name, str) or not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9._-]{0,179}", name):
        raise ValueError("Invalid manifest asset name.")
    stem = name.split(".", 1)[0].upper()
    if name.endswith(".") or stem in {"CON", "PRN", "AUX", "NUL"} or re.fullmatch(r"(?:COM|LPT)[0-9]", stem):
        raise ValueError("Reserved manifest asset name.")
    return name


def validate_asset_target(system: object, arch: object) -> tuple[str, str]:
    if not isinstance(system, str) or system not in {"windows", "linux", "darwin"}:
        raise ValueError("Unsupported manifest platform.")
    allowed_arches = {"x86_64", "arm64"} | ({"universal2"} if system == "darwin" else set())
    if not isinstance(arch, str) or arch not in allowed_arches:
        raise ValueError("Unsupported manifest architecture.")
    return system, arch


def verify_manifest(payload: bytes, signature: bytes, *, tag: str,
                    current_version: str, now: datetime | None = None) -> ReleaseManifest:
    """Authenticate raw bytes before parsing; bind repository, tag, time, version.

    Expiry limits replay, and installed-version comparison prevents downgrade.
    This is not TUF: there is no online timestamp role, persistent highest-seen
    release counter, or automatic root rotation. A correct system clock and
    deliberately distributed public-key pins are required.
    """
    keys = _trusted_keys()
    if not 0 < len(payload) <= MAX_MANIFEST_BYTES:
        raise ValueError("Invalid manifest size.")
    if len(signature) != 64:
        raise ValueError("Invalid release signature size.")
    for key in keys:
        try:
            key.verify(signature, payload)
        except InvalidSignature:
            continue
        break
    else:
        raise ValueError("Release signature does not match a trusted key.")

    data = json.loads(payload, object_pairs_hook=_unique_object)
    if not isinstance(data, dict) or set(data) != {
        "schema", "repository", "tag", "issued_at", "expires_at", "assets"
    }:
        raise ValueError("Invalid release manifest schema.")
    if type(data["schema"]) is not int or data["schema"] != 1:
        raise ValueError("Unsupported release manifest schema.")
    if data["repository"] != GITHUB_REPO or data["tag"] != tag:
        raise ValueError("Manifest repository or release tag does not match.")
    if _tag_version(tag) <= _tag_version(current_version):
        raise ValueError("Manifest is not newer than the installed version.")
    issued = _timestamp(data["issued_at"])
    expires = _timestamp(data["expires_at"])
    clock = now if now is not None else _utc_now()
    if clock.tzinfo is None:
        raise ValueError("Verification requires an aware UTC clock.")
    if expires <= clock:
        raise ValueError("Release manifest has expired.")
    if issued > clock + _MAX_CLOCK_SKEW or expires <= issued:
        raise ValueError("Invalid release manifest validity interval.")
    records = data["assets"]
    if not isinstance(records, list) or not 0 < len(records) <= MAX_ASSETS:
        raise ValueError("Invalid manifest asset list.")
    assets = []
    names: set[str] = set()
    targets: set[tuple[str, str]] = set()
    for item in records:
        if not isinstance(item, dict) or set(item) != {"name", "system", "arch", "size", "sha256"}:
            raise ValueError("Invalid manifest asset record.")
        name = validate_asset_name(item["name"])
        system, arch = validate_asset_target(item["system"], item["arch"])
        size, digest = item["size"], item["sha256"]
        if type(size) is not int or size <= 0:
            raise ValueError("Invalid signed asset size.")
        if not isinstance(digest, str) or not re.fullmatch(r"[0-9a-f]{64}", digest):
            raise ValueError("Invalid signed asset checksum.")
        target = (system, arch)
        if name.casefold() in names or target in targets:
            raise ValueError("Duplicate manifest asset or platform.")
        names.add(name.casefold())
        targets.add(target)
        assets.append(SignedAsset(name, system, arch, size, digest))
    return ReleaseManifest(tag, issued, expires, tuple(assets))


def select_signed_asset(manifest: ReleaseManifest, *, system: str | None = None,
                        machine: str | None = None) -> SignedAsset:
    system = (system or platform.system()).lower()
    machine = (machine or platform.machine()).lower()
    machine = {"amd64": "x86_64", "x64": "x86_64", "aarch64": "arm64"}.get(machine, machine)
    if machine not in {"x86_64", "arm64"}:
        raise ValueError("No signed update supports this platform architecture.")
    for arch in (machine, "universal2" if system == "darwin" else machine):
        for asset in manifest.assets:
            if asset.system == system and asset.arch == arch:
                return asset
    raise ValueError("No signed update asset matches this platform and architecture.")


def release_asset_url(tag: str, name: str) -> str:
    _tag_version(tag)
    validate_asset_name(name)
    return (f"https://github.com/{GITHUB_REPO}/releases/download/"
            f"{urllib.parse.quote(tag, safe='')}/{urllib.parse.quote(name, safe='')}")


def _fetch_small(tag: str, name: str, limit: int, checkpoint: Callable[[], None],
                 timeout: float) -> bytes:
    checkpoint()
    request = urllib.request.Request(release_asset_url(tag, name), headers={
        "User-Agent": "minflux-viewer-updater", "Accept": "application/octet-stream"})
    with urllib.request.urlopen(request, timeout=timeout) as response:  # noqa: S310
        payload = response.read(limit + 1)
    checkpoint()
    if len(payload) > limit:
        raise ValueError("Release metadata exceeds its permitted size.")
    return payload


def download_release(tag: str, current_version: str, *, progress=None,
                     checkpoint: Callable[[], None] = lambda: None,
                     timeout: float = 30.0) -> DownloadedUpdate:
    """Fetch authenticated metadata and the exactly matching signed archive.

    API asset names, URLs, sizes and digests never enter this install path.
    Ownership transfers to the returned object; every failure removes partial
    data. Cancellation is checked during metadata and every archive block.
    """
    _trusted_keys()
    if _tag_version(tag) <= _tag_version(current_version):
        raise ValueError("Release is not newer than the installed version.")
    payload = _fetch_small(tag, MANIFEST_NAME, MAX_MANIFEST_BYTES, checkpoint, timeout)
    signature = _fetch_small(tag, SIGNATURE_NAME, 64, checkpoint, timeout)
    manifest = verify_manifest(payload, signature, tag=tag, current_version=current_version)
    asset = select_signed_asset(manifest)
    archive_suffixes = (".zip",) if asset.system == "windows" else (".zip", ".tar.gz", ".tar.xz", ".tgz")
    if not asset.name.lower().endswith(archive_suffixes):
        raise ValueError("Signed asset is not a supported update archive.")
    directory = tempfile.TemporaryDirectory(prefix="mfv_download_")
    completed = False
    try:
        def report(done: int, total: int) -> None:
            checkpoint()
            if progress is not None:
                progress(done, total)

        download_url(release_asset_url(tag, asset.name), Path(directory.name) / asset.name,
                     expected_size=asset.size, expected_sha256=asset.sha256,
                     progress=report, timeout=timeout)
        checkpoint()
        result = DownloadedUpdate(directory, payload, signature, tag, current_version, asset.name)
        completed = True
        return result
    finally:
        if not completed:
            directory.cleanup()


def prepare_download(download: DownloadedUpdate, *,
                     checkpoint: Callable[[], None] = lambda: None) -> self_update.PreparedUpdate:
    """Reauthenticate before extraction, including updates deferred with 'Later'.

    Hashing and extraction belong on an I/O worker. No subprocess is launched
    here. The caller must either hand off the prepared helper or discard it.
    """
    checkpoint()
    manifest = verify_manifest(download.manifest_bytes, download.signature,
                               tag=download.tag, current_version=download.current_version)
    asset = select_signed_asset(manifest)
    if asset.name != download.asset_name:
        raise ValueError("Downloaded asset does not match the signed platform.")
    digest = hashlib.sha256()
    done = 0
    with download.archive_path.open("rb") as stream:
        while block := stream.read(1 << 16):
            checkpoint()
            done += len(block)
            if done > asset.size:
                raise ValueError("Downloaded archive exceeds signed size.")
            digest.update(block)
    if done != asset.size or digest.hexdigest() != asset.sha256:
        raise ValueError("Downloaded archive no longer matches its signed checksum.")
    checkpoint()
    prepared = self_update.prepare_update(download.archive_path)
    completed = False
    try:
        checkpoint()
        # A large archive can take long enough to cross its expiry while
        # staging. Never return an expired prepared update for handoff.
        verify_manifest(download.manifest_bytes, download.signature,
                        tag=download.tag, current_version=download.current_version)
        completed = True
        return prepared
    finally:
        if not completed:
            self_update.discard_prepared_update(prepared)
