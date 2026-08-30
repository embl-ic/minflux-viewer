"""Safe external Python-library paths and managed ``pip --target`` installs.

The frozen application owns its scientific and Qt stack. User packages are
therefore added *after* the bundled modules, and packages that could replace
that stack are never offered through the managed installer.
"""

from __future__ import annotations

import contextlib
import importlib
import importlib.metadata
import importlib.util
import logging
import os
import re
import sys
import sysconfig
import threading
from collections.abc import Callable, Iterator, Sequence
from dataclasses import dataclass
from email.parser import Parser
from pathlib import Path
from typing import Any

logger = logging.getLogger(__name__)

BLOCKED_PACKAGES = frozenset(
    {
        "numpy",
        "scipy",
        "pyqt6",
        "pyside2",
        "pyside6",
        "pyqtgraph",
        "h5py",
        "zarr",
        "tifffile",
    }
)

_NAME_RE = re.compile(r"^\s*([A-Za-z0-9][A-Za-z0-9._-]*)")
_PIP_MODULE_LOCK = threading.RLock()


class UserLibraryError(RuntimeError):
    """A user-facing external-library or managed-package error."""


@dataclass(frozen=True)
class PathInspection:
    """Compatibility and bundled-package findings for one external root."""

    path: Path
    ok: bool
    message: str
    incompatible: tuple[str, ...] = ()
    blocked_packages: tuple[str, ...] = ()


@dataclass(frozen=True)
class PackageInstallResult:
    """The requested distribution installed into the managed library root."""

    name: str
    version: str
    target: Path
    requested: str


def _canonical_name(name: str) -> str:
    return re.sub(r"[-_.]+", "-", str(name)).lower()


def managed_pylibs_dir() -> Path:
    """Return the per-user directory populated by the managed installer."""
    if sys.platform.startswith("win"):
        base = Path(os.environ.get("APPDATA", Path.home() / "AppData" / "Roaming"))
        return base / "MINFLUX Viewer" / "pylibs"
    if sys.platform == "darwin":
        return (
            Path.home()
            / "Library"
            / "Application Support"
            / "MINFLUX Viewer"
            / "pylibs"
        )
    return Path.home() / ".local" / "share" / "minflux-viewer" / "pylibs"


def vendored_pip_dir() -> Path:
    """Return the real-filesystem pip data directory shipped with the app."""
    from minflux_viewer import resource_path

    return Path(resource_path("pip"))


def _metadata_name(dist_info: Path) -> str:
    metadata = dist_info / "METADATA"
    try:
        message = Parser().parsestr(metadata.read_text(encoding="utf-8", errors="replace"))
        if message.get("Name"):
            return str(message["Name"])
    except OSError:
        pass
    stem = dist_info.name.removesuffix(".dist-info")
    return stem.rsplit("-", 1)[0]


def blocked_packages_in(path: str | Path) -> tuple[str, ...]:
    """Return blocklisted distributions/modules present below *path*."""
    root = Path(path).expanduser()
    if not root.is_dir():
        return ()

    found: set[str] = set()
    try:
        entries = list(root.iterdir())
    except OSError:
        return ()

    for entry in entries:
        name = entry.name
        if entry.is_dir() and name.endswith(".dist-info"):
            canonical = _canonical_name(_metadata_name(entry))
        else:
            canonical = _canonical_name(name.split(".", 1)[0])
        if canonical in BLOCKED_PACKAGES:
            found.add(canonical)
    return tuple(sorted(found))


def _fallback_tag_compatible(tag: str) -> bool:
    """Conservative compatibility check used when ``packaging`` is unavailable."""
    pieces = tag.strip().lower().split("-")
    if len(pieces) != 3:
        return False
    interpreters, abis, platforms = (part.split(".") for part in pieces)
    py_major = sys.version_info.major
    py_minor = sys.version_info.minor
    cp_tag = f"cp{py_major}{py_minor}"
    interpreter_ok = any(
        value in {cp_tag, f"py{py_major}", f"py{py_major}{py_minor}"}
        for value in interpreters
    )
    abi_ok = any(value in {"none", "abi3", cp_tag} for value in abis)
    if "any" in platforms:
        platform_ok = True
    else:
        native = sysconfig.get_platform().replace("-", "_").replace(".", "_").lower()
        machine = native.rsplit("_", 1)[-1]
        platform_ok = any(
            value == native
            or (
                sys.platform.startswith("linux")
                and value.endswith(f"_{machine}")
                and value.startswith(("linux_", "manylinux", "musllinux"))
            )
            or (
                sys.platform == "darwin"
                and value.startswith("macosx_")
                and value.endswith(f"_{machine}")
            )
            for value in platforms
        )
    return interpreter_ok and abi_ok and platform_ok


def _tag_compatible(tag: str) -> bool:
    try:
        from packaging.tags import parse_tag, sys_tags

        return bool(parse_tag(tag) & set(sys_tags()))
    except (ImportError, ValueError):
        return _fallback_tag_compatible(tag)


def inspect_path(path: str | Path) -> PathInspection:
    """Inspect one external site-packages root without importing from it."""
    root = Path(path).expanduser()
    if not root.exists():
        return PathInspection(root, False, "Path does not exist")
    if not root.is_dir():
        return PathInspection(root, False, "Path is not a directory")

    incompatible: list[str] = []
    try:
        wheel_files = sorted(root.glob("*.dist-info/WHEEL"))
    except OSError as exc:
        return PathInspection(root, False, f"Cannot read path: {exc}")

    for wheel_file in wheel_files:
        try:
            message = Parser().parsestr(
                wheel_file.read_text(encoding="utf-8", errors="replace")
            )
        except OSError as exc:
            return PathInspection(root, False, f"Cannot read {wheel_file.name}: {exc}")
        tags = [str(value).strip() for value in message.get_all("Tag", []) if value]
        if tags and not any(_tag_compatible(tag) for tag in tags):
            distribution = _metadata_name(wheel_file.parent)
            incompatible.append(f"{distribution} ({', '.join(tags)})")

    blocked = blocked_packages_in(root)
    if incompatible:
        running = (
            f"CPython {sys.version_info.major}.{sys.version_info.minor} "
            f"on {sysconfig.get_platform()}"
        )
        summary = "; ".join(incompatible[:3])
        if len(incompatible) > 3:
            summary += f"; and {len(incompatible) - 3} more"
        return PathInspection(
            root,
            False,
            f"ABI mismatch for {running}: {summary}",
            tuple(incompatible),
            blocked,
        )

    message = "ABI OK"
    if not wheel_files:
        message += " (no wheel metadata found)"
    if blocked:
        message += f"; contains bundled packages: {', '.join(blocked)}"
    return PathInspection(root, True, message, (), blocked)


def _path_key(path: str | Path) -> str:
    return os.path.normcase(os.path.abspath(os.fspath(path)))


def install_paths(prefs: dict) -> list[str]:
    """Validate and append configured external roots to :data:`sys.path`.

    The path is deliberately appended, never prepended. Frozen-build testing
    showed that prepending a user's site-packages replaced the bundled NumPy
    with an ABI-uncontrolled copy; appending preserves the application stack.
    """
    settings = (prefs or {}).get("plugin", {}) or {}
    if not settings.get("user_lib_enabled", True):
        return []

    existing = {_path_key(value) for value in sys.path if value}
    added: list[str] = []
    for configured in settings.get("user_lib_paths", []) or []:
        text = str(configured).strip()
        if not text:
            continue
        inspection = inspect_path(text)
        if not inspection.ok:
            logger.warning("External Python path refused: %s — %s", text, inspection.message)
            continue
        if inspection.blocked_packages:
            logger.warning(
                "External Python path %s contains bundled packages (%s); it is appended so the bundled copies keep priority",
                inspection.path,
                ", ".join(inspection.blocked_packages),
            )
        try:
            resolved = str(inspection.path.resolve())
        except OSError:
            resolved = str(inspection.path.absolute())
        key = _path_key(resolved)
        if key in existing:
            continue
        sys.path.append(resolved)
        existing.add(key)
        added.append(resolved)
        logger.info("Appended external Python-library path: %s", resolved)
    if added:
        importlib.invalidate_caches()
    return added


def _requested_name(spec: str) -> str:
    match = _NAME_RE.match(str(spec))
    if match is None:
        raise UserLibraryError(
            "Enter a package name, optionally followed by a version constraint"
        )
    return _canonical_name(match.group(1))


def _ensure_allowed_spec(spec: str) -> str:
    name = _requested_name(spec)
    if name in BLOCKED_PACKAGES:
        raise UserLibraryError(
            f"{name} is part of MINFLUX Viewer's bundled runtime and cannot be installed here"
        )
    return name


@contextlib.contextmanager
def _vendored_pip_modules(pip_root: Path) -> Iterator[None]:
    """Temporarily load the unpacked pip package under its real module name."""
    pip_init = pip_root / "pip" / "__init__.py"
    distlib = pip_root / "pip" / "_vendor" / "distlib"
    if not pip_init.is_file() or not distlib.is_dir():
        raise UserLibraryError(
            "The unpacked pip runtime is missing. Run tools/vendor_pip.py and ship resources/pip as data."
        )

    with _PIP_MODULE_LOCK:
        saved = {
            name: module
            for name, module in tuple(sys.modules.items())
            if name == "pip" or name.startswith("pip.")
        }
        for name in saved:
            sys.modules.pop(name, None)
        spec = importlib.util.spec_from_file_location(
            "pip", pip_init, submodule_search_locations=[str(pip_init.parent)]
        )
        if spec is None or spec.loader is None:
            raise UserLibraryError(f"Cannot load unpacked pip from {pip_init}")
        module = importlib.util.module_from_spec(spec)
        sys.modules["pip"] = module
        try:
            spec.loader.exec_module(module)
            yield
        finally:
            for name in tuple(sys.modules):
                if name == "pip" or name.startswith("pip."):
                    sys.modules.pop(name, None)
            sys.modules.update(saved)


def _run_pip(args: Sequence[str], pip_root: str | Path | None = None) -> int:
    root = Path(pip_root) if pip_root is not None else vendored_pip_dir()
    with _vendored_pip_modules(root):
        from pip._internal.cli.main import main as pip_main

        return int(pip_main(list(args)))


def _distribution_at(target: Path, name: str) -> importlib.metadata.Distribution | None:
    canonical = _canonical_name(name)
    for distribution in importlib.metadata.distributions(path=[str(target)]):
        if _canonical_name(distribution.metadata.get("Name", "")) == canonical:
            return distribution
    return None


def _record_install(prefs: dict, result: PackageInstallResult) -> None:
    plugin = prefs.setdefault("plugin", {})
    installed = plugin.setdefault("installed_packages", {})
    installed[result.name] = result.version
    paths = plugin.setdefault("user_lib_paths", [])
    target = str(result.target)
    if _path_key(target) not in {_path_key(value) for value in paths if value}:
        paths.append(target)


def install_package(
    spec: str,
    prefs: dict | None = None,
    *,
    target: str | Path | None = None,
    report: Callable[[str], None] | None = None,
) -> PackageInstallResult:
    """Install one wheel-only distribution into the managed library root.

    ``report`` is the cooperative cancellation checkpoint supplied by
    :class:`~minflux_viewer.ui.background_tasks.BackgroundTask`. pip itself is
    synchronous; cancellation takes effect after pip returns, before its result
    reaches the UI.
    """
    requested = str(spec).strip()
    name = _ensure_allowed_spec(requested)
    destination = Path(target) if target is not None else managed_pylibs_dir()
    destination = destination.expanduser().resolve()
    destination.mkdir(parents=True, exist_ok=True)
    notify = report or (lambda _stage: None)
    notify(f"Installing {requested}…")

    # --no-deps prevents a normal package dependency from placing a second
    # copy of the bundled NumPy/SciPy/Qt stack in this directory. Missing
    # pure-Python dependencies can be installed explicitly and remain visible.
    arguments = [
        "install",
        requested,
        "--target",
        str(destination),
        "--only-binary=:all:",
        "--no-deps",
        "--no-input",
        "--disable-pip-version-check",
        "--upgrade",
    ]
    return_code = _run_pip(arguments)
    if return_code != 0:
        raise UserLibraryError(f"pip could not install {requested} (exit code {return_code})")

    distribution = _distribution_at(destination, name)
    if distribution is None:
        raise UserLibraryError(
            f"pip reported success, but {name} was not found in {destination}"
        )
    version = str(distribution.version)
    result = PackageInstallResult(name, version, destination, requested)
    if prefs is not None:
        _record_install(prefs, result)
    notify(f"Installed {name} {version}")
    return result


def _safe_distribution_files(
    distribution: importlib.metadata.Distribution, target: Path
) -> list[Path]:
    files = distribution.files
    if files is None:
        raise UserLibraryError("The package has no RECORD manifest and cannot be removed safely")
    root = target.resolve()
    safe: list[Path] = []
    for relative in files:
        candidate = Path(distribution.locate_file(relative)).resolve()
        try:
            candidate.relative_to(root)
        except ValueError as exc:
            raise UserLibraryError(
                f"Refusing to remove a package file outside the managed directory: {candidate}"
            ) from exc
        safe.append(candidate)
    return safe


def remove_package(
    name: str,
    prefs: dict | None = None,
    *,
    target: str | Path | None = None,
    report: Callable[[str], None] | None = None,
) -> bool:
    """Remove one RECORD-described distribution from the managed directory."""
    canonical = _canonical_name(name)
    destination = (
        Path(target) if target is not None else managed_pylibs_dir()
    ).expanduser().resolve()
    notify = report or (lambda _stage: None)
    notify(f"Removing {canonical}…")
    distribution = _distribution_at(destination, canonical)
    if distribution is None:
        removed = False
    else:
        files = _safe_distribution_files(distribution, destination)
        parents: set[Path] = set()
        for file_path in files:
            if file_path.is_file() or file_path.is_symlink():
                file_path.unlink()
            parents.update(
                parent
                for parent in file_path.parents
                if parent != destination and parent.is_relative_to(destination)
            )
        for directory in sorted(parents, key=lambda value: len(value.parts), reverse=True):
            try:
                directory.rmdir()
            except (FileNotFoundError, OSError):
                pass
        removed = True

    if prefs is not None:
        installed = prefs.setdefault("plugin", {}).setdefault("installed_packages", {})
        installed.pop(canonical, None)
    notify(f"Removed {canonical}" if removed else f"{canonical} was not installed")
    importlib.invalidate_caches()
    return removed


def record_install(prefs: dict, result: PackageInstallResult) -> None:
    """Record a worker-produced install result on the GUI thread."""
    _record_install(prefs, result)


def record_removal(prefs: dict, name: str) -> None:
    """Remove one managed-package manifest entry on the GUI thread."""
    installed = prefs.setdefault("plugin", {}).setdefault("installed_packages", {})
    installed.pop(_canonical_name(name), None)


def discard_install(result: Any) -> None:
    """Best-effort cleanup when a cancelled UI no longer accepts an install."""
    if isinstance(result, PackageInstallResult):
        try:
            remove_package(result.name, target=result.target)
        except Exception:  # noqa: BLE001 - cancellation cleanup must not escape a worker
            logger.exception("Could not clean up cancelled package install %s", result.name)


__all__ = [
    "BLOCKED_PACKAGES",
    "PackageInstallResult",
    "PathInspection",
    "UserLibraryError",
    "blocked_packages_in",
    "discard_install",
    "inspect_path",
    "install_package",
    "install_paths",
    "managed_pylibs_dir",
    "record_install",
    "record_removal",
    "remove_package",
    "vendored_pip_dir",
]
