"""
minflux_viewer.plugins.manifest
===============================
Parsing and validation of ``plugin.toml`` — the declaration a Tier 2 user
plugin ships beside its code.

TOML rather than Fiji's ``plugins.config`` because :mod:`tomllib` is standard
library from 3.11, the format has real types, and it carries the fields a
support relationship needs: which API version the plugin was written against,
and what it expects to be installed.

Qt-free and side-effect-free on purpose: reading a manifest must never import
the plugin, so the Plugins menu can be built at startup without paying for a
plugin that imports scikit-image.

A malformed manifest raises :class:`ManifestError`, which the loader turns into
a visible, explained menu entry. It must never escape as a bare exception and
stop the rest of the menu building.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import tomllib

#: The file that makes a directory a Tier 2 plugin.
MANIFEST_NAME = "plugin.toml"

#: Reverse-DNS-ish id: dot-separated lowercase segments. Used as the module
#: namespace (``mfv_plugins.<id>``), so it must be import-safe.
_ID_RE = re.compile(r"^[a-z0-9]+(?:[._-][a-z0-9]+)*$")

#: Comparison operators accepted in a requirement such as ``">=1.0,<2.0"``.
_OPS = ("!=", ">=", "<=", "==", ">", "<")


class ManifestError(ValueError):
    """A ``plugin.toml`` is missing, unreadable, or does not declare enough."""


@dataclass(frozen=True)
class PluginManifest:
    """One validated ``plugin.toml``."""

    #: Stable unique id, also the module namespace.
    id: str
    #: Human-readable plugin name.
    name: str
    version: str = "0.0.0"
    author: str = ""
    description: str = ""

    #: Menu placement. ``menu_path`` is a ``>``-separated trail such as
    #: ``"Plugins > HlyB"``; ``entry`` is the leaf label.
    menu_path: str = ""
    entry: str = ""
    keywords: tuple[str, ...] = ()

    #: ``module`` is a module name relative to the plugin directory (no ``.py``);
    #: ``callable_name`` is the function in it that takes ``ctx``.
    module: str = "main"
    callable_name: str = "run"

    #: Requirements, as written. Empty means "no constraint".
    requires_mfv_api: str = ""
    requires_app: str = ""
    requires_python: tuple[str, ...] = ()

    #: Where the manifest was read from.
    directory: Path = field(default_factory=Path)

    @property
    def label(self) -> str:
        """The leaf menu label, falling back to the plugin name."""
        return self.entry or self.name

    @property
    def entry_path(self) -> Path:
        """The file the plugin body lives in."""
        return self.directory / f"{self.module}.py"


def read_manifest(directory: str | Path) -> PluginManifest:
    """
    Read and validate ``<directory>/plugin.toml``.

    Raises :class:`ManifestError` with a message written for the plugin author.
    """
    directory = Path(directory)
    path = directory / MANIFEST_NAME
    if not path.is_file():
        raise ManifestError(f"No {MANIFEST_NAME} in {directory}")
    try:
        raw = tomllib.loads(path.read_text(encoding="utf-8"))
    except tomllib.TOMLDecodeError as exc:
        raise ManifestError(f"{path} is not valid TOML: {exc}") from None
    except OSError as exc:
        raise ManifestError(f"{path} could not be read: {exc}") from None
    return parse_manifest(raw, directory)


def parse_manifest(raw: dict[str, Any], directory: str | Path) -> PluginManifest:
    """Validate an already-decoded manifest mapping."""
    directory = Path(directory)
    plugin = _section(raw, "plugin")
    menu = _section(raw, "menu", required=False)
    run = _section(raw, "run", required=False)
    requires = _section(raw, "requires", required=False)

    plugin_id = str(plugin.get("id", "") or "").strip()
    if not plugin_id:
        raise ManifestError("[plugin] needs an 'id' (reverse-DNS, e.g. \"embl.my_tool\").")
    if not _ID_RE.match(plugin_id):
        raise ManifestError(
            f"Plugin id {plugin_id!r} must be lowercase letters, digits and "
            "separators (. _ -) — it is used as a Python module name."
        )

    name = str(plugin.get("name", "") or "").strip()
    if not name:
        raise ManifestError(f"[plugin] {plugin_id}: needs a 'name' for the menu.")

    module = str(run.get("module", "main") or "main").strip()
    if not module.isidentifier():
        raise ManifestError(
            f"[run] module {module!r} must be a plain module name such as \"main\" "
            "(no path, no .py)."
        )
    callable_name = str(run.get("callable", "run") or "run").strip()
    if not callable_name.isidentifier():
        raise ManifestError(f"[run] callable {callable_name!r} is not a valid name.")

    keywords = menu.get("keywords", ()) or ()
    if isinstance(keywords, str):
        keywords = (keywords,)

    python_reqs = requires.get("python", ()) or ()
    if isinstance(python_reqs, str):
        python_reqs = (python_reqs,)

    for key in ("mfv_api", "app"):
        spec = str(requires.get(key, "") or "")
        if spec:
            _parse_requirement(spec, key)          # validate eagerly

    return PluginManifest(
        id=plugin_id,
        name=name,
        version=str(plugin.get("version", "0.0.0") or "0.0.0"),
        author=str(plugin.get("author", "") or ""),
        description=str(plugin.get("description", "") or ""),
        menu_path=str(menu.get("path", "") or ""),
        entry=str(menu.get("entry", "") or ""),
        keywords=tuple(str(k) for k in keywords),
        module=module,
        callable_name=callable_name,
        requires_mfv_api=str(requires.get("mfv_api", "") or ""),
        requires_app=str(requires.get("app", "") or ""),
        requires_python=tuple(str(p) for p in python_reqs),
        directory=directory,
    )


# ---------------------------------------------------------------------------
# version requirements
# ---------------------------------------------------------------------------

def version_satisfies(version: str, requirement: str) -> bool:
    """
    Whether *version* satisfies a comma-separated *requirement*.

    Supports ``>= > <= < == !=`` over dotted numeric versions, which is all a
    plugin manifest needs. Deliberately does not depend on ``packaging``: it is
    not a declared dependency of this application and so is not guaranteed to
    be in the frozen bundle, and a missing import here would break the whole
    Plugins menu rather than one plugin.

    An empty requirement is satisfied by anything.
    """
    if not requirement.strip():
        return True
    parts = _parse_requirement(requirement, "requirement")
    actual = _version_key(version)
    for op, wanted in parts:
        target = _version_key(wanted)
        if op == ">=" and not actual >= target:
            return False
        if op == ">" and not actual > target:
            return False
        if op == "<=" and not actual <= target:
            return False
        if op == "<" and not actual < target:
            return False
        if op == "==" and actual != target:
            return False
        if op == "!=" and actual == target:
            return False
    return True


def _parse_requirement(requirement: str, field_name: str) -> list[tuple[str, str]]:
    out: list[tuple[str, str]] = []
    for piece in requirement.split(","):
        piece = piece.strip()
        if not piece:
            continue
        for op in _OPS:
            if piece.startswith(op):
                value = piece[len(op):].strip()
                if not value:
                    raise ManifestError(
                        f"[requires] {field_name}: {piece!r} has no version after {op!r}."
                    )
                out.append((op, value))
                break
        else:
            raise ManifestError(
                f"[requires] {field_name}: {piece!r} must start with one of "
                f"{', '.join(_OPS)} — for example \">=1.0,<2.0\"."
            )
    return out


def _version_key(version: str) -> tuple[int, ...]:
    """
    Dotted version to a comparable tuple.

    Non-numeric trailers (``1.2.0rc1``) are truncated at the first non-digit;
    plugin manifests pin release versions, and guessing at pre-release ordering
    would be worse than ignoring it.
    """
    numbers: list[int] = []
    for piece in str(version).split("."):
        digits = ""
        for char in piece:
            if not char.isdigit():
                break
            digits += char
        numbers.append(int(digits) if digits else 0)
    while len(numbers) < 3:
        numbers.append(0)
    return tuple(numbers)


def _section(raw: dict[str, Any], key: str, *, required: bool = True) -> dict[str, Any]:
    value = raw.get(key, {})
    if value in (None, {}):
        if required:
            raise ManifestError(f"{MANIFEST_NAME} needs a [{key}] section.")
        return {}
    if not isinstance(value, dict):
        raise ManifestError(f"[{key}] must be a table, not {type(value).__name__}.")
    return value
