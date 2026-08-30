"""Download and unpack pip into ``resources/pip`` for frozen builds.

pip must remain ordinary files on disk: freezing it into PyInstaller's module
archive breaks ``pip._vendor.distlib`` finder discovery. Run this helper before
building, then include the resulting directory in the spec's data files.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import shutil
import tempfile
import urllib.request
import zipfile
from collections.abc import Callable
from pathlib import Path
from typing import BinaryIO

REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_TARGET = REPO_ROOT / "resources" / "pip"
DEFAULT_PIP_VERSION = "26.0.1"


class VendorPipError(RuntimeError):
    """The requested pip wheel could not be downloaded or unpacked safely."""


def _read_url(
    url: str,
    opener: Callable[..., BinaryIO] = urllib.request.urlopen,
) -> bytes:
    with opener(url, timeout=60) as response:
        return response.read()


def _wheel_record(version: str, metadata: dict) -> tuple[str, str]:
    expected = f"pip-{version}-py3-none-any.whl"
    for file_info in metadata.get("urls", []):
        if file_info.get("packagetype") != "bdist_wheel":
            continue
        if file_info.get("filename") != expected:
            continue
        url = str(file_info.get("url", ""))
        digest = str((file_info.get("digests") or {}).get("sha256", ""))
        if url and digest:
            return url, digest
    raise VendorPipError(f"PyPI did not publish the expected universal wheel {expected}")


def _validate_member(name: str) -> None:
    path = Path(name.replace("\\", "/"))
    if path.is_absolute() or ".." in path.parts:
        raise VendorPipError(f"Unsafe path in pip wheel: {name}")


def _replace_directory(staged: Path, target: Path) -> None:
    target = target.resolve()
    if target == Path(target.anchor):
        raise VendorPipError(f"Refusing to replace unexpected target directory: {target}")
    if target.exists() and not target.is_dir():
        raise VendorPipError(f"pip target exists but is not a directory: {target}")
    backup = target.with_name(f".{target.name}.previous")
    if backup.is_symlink() or (backup.exists() and not backup.is_dir()):
        raise VendorPipError(f"pip backup path exists but is not a directory: {backup}")
    if backup.exists():
        shutil.rmtree(backup)
    moved_old = False
    try:
        if target.exists():
            target.rename(backup)
            moved_old = True
        staged.rename(target)
    except Exception:
        if moved_old and not target.exists() and backup.exists():
            backup.rename(target)
        raise
    if backup.exists():
        shutil.rmtree(backup)


def vendor_pip(
    version: str = DEFAULT_PIP_VERSION,
    target: str | Path = DEFAULT_TARGET,
    *,
    opener: Callable[..., BinaryIO] = urllib.request.urlopen,
) -> Path:
    """Download the official universal pip wheel, verify it, and unpack it."""
    version = str(version).strip()
    if not version:
        raise VendorPipError("A pip version is required")
    unresolved_destination = Path(target).expanduser().absolute()
    if unresolved_destination.is_symlink():
        raise VendorPipError(f"Refusing to replace a symlinked pip target: {unresolved_destination}")
    destination = unresolved_destination.resolve()
    metadata_url = f"https://pypi.org/pypi/pip/{version}/json"
    try:
        metadata = json.loads(_read_url(metadata_url, opener).decode("utf-8"))
        wheel_url, expected_digest = _wheel_record(version, metadata)
        wheel_bytes = _read_url(wheel_url, opener)
    except (OSError, ValueError, KeyError) as exc:
        raise VendorPipError(f"Could not download pip {version}: {exc}") from exc
    actual_digest = hashlib.sha256(wheel_bytes).hexdigest()
    if actual_digest != expected_digest:
        raise VendorPipError(
            f"pip wheel checksum mismatch: expected {expected_digest}, got {actual_digest}"
        )

    destination.parent.mkdir(parents=True, exist_ok=True)
    staging_root = Path(tempfile.mkdtemp(prefix="vendor-pip-", dir=destination.parent))
    staged = staging_root / "pip"
    staged.mkdir()
    wheel_file = staging_root / f"pip-{version}.whl"
    wheel_file.write_bytes(wheel_bytes)
    try:
        with zipfile.ZipFile(wheel_file) as archive:
            for member in archive.infolist():
                _validate_member(member.filename)
            archive.extractall(staged)
        if not (staged / "pip" / "__init__.py").is_file():
            raise VendorPipError("Downloaded wheel does not contain pip/__init__.py")
        if not (staged / "pip" / "_vendor" / "distlib").is_dir():
            raise VendorPipError("Downloaded wheel does not contain pip's vendored distlib")
        wheel_file.unlink(missing_ok=True)
        _replace_directory(staged, destination)
    finally:
        shutil.rmtree(staging_root, ignore_errors=True)
    return destination


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--version", default=DEFAULT_PIP_VERSION, help="pip version to vendor")
    parser.add_argument(
        "--target",
        type=Path,
        default=DEFAULT_TARGET,
        help="destination directory (default: resources/pip)",
    )
    args = parser.parse_args()
    destination = vendor_pip(args.version, args.target)
    print(f"Vendored pip {args.version} into {destination}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
