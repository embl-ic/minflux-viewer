"""
minflux_viewer.plugins.loader
=============================
Discovery of user plugins from folders **outside** this package.

This is what lets a customer's analysis live in their own directory instead of
being absorbed into the core, and what keeps one customer's tools out of every
other customer's menu.

Three roots (:func:`plugin_roots`), two tiers:

* **Tier 1** — a single ``.py`` file dropped in a root is one menu entry. No
  manifest. The label comes from the filename, Fiji-style (``My_Tool.py`` ->
  *My Tool*). This is the tier that gets used, because project code starts as
  one file.
* **Tier 2** — a directory with a ``plugin.toml`` (see :mod:`.manifest`).

Four rules this module exists to enforce
----------------------------------------
1. **Namespaced module names.** A plugin is imported as ``mfv_plugins.<id>``
   with ``submodule_search_locations`` set to its own directory, so two
   customers can both ship a ``utils.py``.
2. **Lazy import.** Scanning reads filenames and manifests only. The plugin
   body is imported when its menu entry is first invoked, so a plugin that
   imports scikit-image costs nothing at startup.
3. **Contained failure.** Anything that goes wrong with one plugin becomes a
   visible, explained menu entry; it never stops the others, and never stops
   the built-in plugins.
4. **Trust is confirmed before code runs, not before it is listed.** Nothing is
   imported at scan time, so the moment that matters is the first *launch* from
   a root the user has not yet accepted.

Frozen-build note
-----------------
Roots resolve from ``sys.executable`` when frozen, **not** ``__file__``:
``sys._MEIPASS`` is the bundle's ``_internal/``, which is both the wrong place
to look for a user's plugins and replaced on every update.
"""

from __future__ import annotations

import importlib.util
import os
import sys
from collections.abc import Callable
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from .manifest import MANIFEST_NAME, ManifestError, PluginManifest, read_manifest, version_satisfies

#: Package name every user plugin is imported under.
NAMESPACE = "mfv_plugins"

#: Filenames skipped in a Tier 1 scan.
_TIER1_SKIP_PREFIXES = ("_", ".")


@dataclass
class DiscoveredPlugin:
    """One plugin found on disk, before anything of it has been imported."""

    #: Unique id; the module namespace. Tier 1 derives it from the filename.
    id: str
    #: Leaf label shown in the menu.
    label: str
    #: Directory the plugin lives in (Tier 1: the root it was found in).
    directory: Path
    #: The file holding the plugin body.
    entry_path: Path
    tier: int = 1
    manifest: PluginManifest | None = None
    root: Path = field(default_factory=Path)
    #: Submenus beneath the application's top-level Plugins menu.
    menu_path: tuple[str, ...] = ()
    #: Non-empty when the plugin cannot run; the reason is shown to the user.
    error: str = ""

    @property
    def tooltip(self) -> str:
        if self.error:
            return f"This plugin cannot run: {self.error}"
        bits = []
        if self.manifest is not None:
            if self.manifest.description:
                bits.append(self.manifest.description)
            if self.manifest.version:
                bits.append(f"version {self.manifest.version}")
            if self.manifest.author:
                bits.append(f"by {self.manifest.author}")
        bits.append(str(self.entry_path))
        return " — ".join(bits)

    @property
    def keywords(self) -> tuple[str, ...]:
        base = ("plugin", "user plugin")
        if self.manifest is not None:
            return tuple(self.manifest.keywords) + base
        return base


# ---------------------------------------------------------------------------
# roots
# ---------------------------------------------------------------------------

def app_plugin_dir() -> Path:
    """
    ``<app dir>/plugins`` — beside the executable, or beside the repo when
    running from source. The portable root: ships a plugin with the
    application on a share or a USB stick, the Fiji habit.
    """
    if getattr(sys, "frozen", False):
        return Path(sys.executable).resolve().parent / "plugins"
    # Running from source: the repository root, not the package directory, so
    # a checkout's plugins folder is a sibling of minflux_viewer/.
    return Path(__file__).resolve().parents[2] / "plugins"


def user_plugin_dir() -> Path:
    """
    The per-user root, and the default place a plugin is installed.

    Per-machine and per-account, never touched by the installer — this is what
    keeps one user's project plugins out of another user's menu.
    """
    if sys.platform.startswith("win"):
        base = Path(os.environ.get("APPDATA", Path.home() / "AppData" / "Roaming"))
        return base / "MINFLUX Viewer" / "plugins"
    if sys.platform == "darwin":
        # Never inside the .app bundle: writing there breaks its signature.
        return Path.home() / "Library" / "Application Support" / "MINFLUX Viewer" / "plugins"
    return Path.home() / ".local" / "share" / "minflux-viewer" / "plugins"


def plugin_roots(prefs: dict | None = None) -> list[Path]:
    """
    Every folder scanned for plugins, in order, de-duplicated.

    The per-user folder can be switched off with
    ``prefs["plugin"]["scan_user_dir"]``; extra folders come from
    ``prefs["plugin"]["paths"]``.
    """
    settings = (prefs or {}).get("plugin", {}) or {}
    roots: list[Path] = [app_plugin_dir()]
    if settings.get("scan_user_dir", True):
        roots.append(user_plugin_dir())
    for extra in settings.get("paths", []) or []:
        text = str(extra).strip()
        if text:
            roots.append(Path(text).expanduser())

    seen: set[str] = set()
    unique: list[Path] = []
    for root in roots:
        try:
            key = str(root.resolve())
        except OSError:
            key = str(root)
        if key not in seen:
            seen.add(key)
            unique.append(root)
    return unique


# ---------------------------------------------------------------------------
# scanning
# ---------------------------------------------------------------------------

def scan_root(root: str | Path) -> list[DiscoveredPlugin]:
    """
    Everything plugin-shaped in one folder. Never imports, never raises.

    A directory holding a ``plugin.toml`` is Tier 2; a top-level ``.py`` file is
    Tier 1. A Tier 2 directory whose manifest is broken still produces an entry,
    carrying the reason — a plugin that silently fails to appear is a worse bug
    report than one that appears and explains itself.
    """
    root = Path(root)
    found: list[DiscoveredPlugin] = []
    try:
        if not root.is_dir():
            return found
        children = sorted(root.iterdir())
    except OSError:
        return found

    for child in children:
        try:
            if child.is_dir():
                if (child / MANIFEST_NAME).is_file():
                    found.append(_tier2(child, root))
                continue
            if child.suffix.lower() == ".py" and not child.name.startswith(_TIER1_SKIP_PREFIXES):
                found.append(_tier1(child, root))
        except OSError as exc:
            found.append(DiscoveredPlugin(
                id=_safe_id(child.stem), label=child.stem, directory=root,
                entry_path=child, root=root, error=f"unreadable: {exc}",
            ))
    return found


def _tier1(path: Path, root: Path) -> DiscoveredPlugin:
    """A single ``.py`` file. Label from the filename, Fiji-style."""
    label = path.stem.replace("_", " ").strip() or path.stem
    return DiscoveredPlugin(
        id=_safe_id(path.stem),
        label=label,
        directory=root,
        entry_path=path,
        tier=1,
        root=root,
    )


def _tier2(directory: Path, root: Path) -> DiscoveredPlugin:
    """A directory with a manifest."""
    try:
        manifest = read_manifest(directory)
    except ManifestError as exc:
        return DiscoveredPlugin(
            id=_safe_id(directory.name), label=directory.name, directory=directory,
            entry_path=directory / MANIFEST_NAME, tier=2, root=root, error=str(exc),
        )

    trail = [p.strip() for p in manifest.menu_path.split(">") if p.strip()]
    # "Plugins" is where the menu already is; a manifest naming it is not asking
    # for a submenu called Plugins inside Plugins.
    trail = [p for p in trail if p.lower() != "plugins"]
    error = ""
    if not manifest.entry_path.is_file():
        error = (
            f"[run] module \"{manifest.module}\" — expected "
            f"{manifest.entry_path.name} in {directory}"
        )
    else:
        error = _requirement_error(manifest)

    return DiscoveredPlugin(
        id=manifest.id, label=manifest.label, directory=directory,
        entry_path=manifest.entry_path, tier=2, manifest=manifest,
        root=root, menu_path=tuple(trail), error=error,
    )


def _requirement_error(manifest: PluginManifest) -> str:
    """Refuse a plugin written against an API this build does not provide."""
    from ..api import __api_version__

    if manifest.requires_mfv_api:
        try:
            ok = version_satisfies(__api_version__, manifest.requires_mfv_api)
        except ManifestError as exc:
            return str(exc)
        if not ok:
            return (
                f"needs mfv API {manifest.requires_mfv_api}, this build provides "
                f"{__api_version__}"
            )
    if manifest.requires_app:
        from .. import __version__

        try:
            ok = version_satisfies(__version__, manifest.requires_app)
        except ManifestError as exc:
            return str(exc)
        if not ok:
            return (
                f"needs MINFLUX Viewer {manifest.requires_app}, this is {__version__}"
            )
    missing = [name for name in manifest.requires_python if not _importable(name)]
    if missing:
        return (
            "needs Python package(s) not installed: " + ", ".join(missing) +
            " — see Preferences ▸ Plugin ▸ Python packages"
        )
    return ""


def _importable(requirement: str) -> bool:
    """Whether a ``requires.python`` entry resolves to an importable module."""
    name = requirement
    for op in ("!=", ">=", "<=", "==", ">", "<", "~=", "["):
        if op in name:
            name = name.split(op)[0]
    name = name.strip().replace("-", "_")
    if not name:
        return True
    try:
        return importlib.util.find_spec(name) is not None
    except (ImportError, ValueError, AttributeError):
        return False


def _safe_id(text: str) -> str:
    cleaned = "".join(c if (c.isalnum() or c in "._-") else "_" for c in str(text).lower())
    return cleaned.strip("._-") or "plugin"


# ---------------------------------------------------------------------------
# loading
# ---------------------------------------------------------------------------

def module_name_for(plugin_id: str) -> str:
    """
    The module name a plugin is imported under.

    The id is flattened to a **single** segment (``embl.hlyb_dimer`` ->
    ``mfv_plugins.embl_hlyb_dimer``): a dotted module name would imply
    intermediate packages (``mfv_plugins.embl``) that do not exist, and the
    import machinery would have to invent them.
    """
    flat = str(plugin_id).replace(".", "_").replace("-", "_")
    return f"{NAMESPACE}.{flat}"


def load_entry_callable(plugin: DiscoveredPlugin) -> Callable[[Any], Any]:
    """
    Import the plugin body and return its entry callable.

    The body is imported as a **package** rooted at the plugin's own directory
    (:func:`module_name_for`), so a Tier 2 plugin reaches its own helpers with a
    relative import::

        from .utils import something          # in main.py

    That is what lets two customers both ship a ``utils.py``. An *absolute*
    ``import utils`` cannot be isolated -- it is a single global name in
    ``sys.modules``, so the second plugin to load would silently get the
    first one's module. Relative imports are the only form that can be, and
    the manifest documentation says so.

    Raises with a message naming the plugin.
    """
    module_name = module_name_for(plugin.id)
    _ensure_namespace_package()

    cached = sys.modules.get(module_name)
    module = cached if cached is not None else _import_module(plugin, module_name)

    if plugin.tier == 2 and plugin.manifest is not None:
        wanted = plugin.manifest.callable_name
    else:
        wanted = "run"
    func = getattr(module, wanted, None)
    if func is None:
        # Tier 1 convenience: a bare script with no run() is still runnable —
        # importing it *is* running it, which is how a Fiji script behaves.
        if plugin.tier == 1:
            return lambda _ctx: None
        raise AttributeError(
            f"{plugin.entry_path.name} defines no {wanted}() — a plugin body is "
            f"`def {wanted}(ctx): ...`"
        )
    if not callable(func):
        raise TypeError(f"{plugin.entry_path.name}: {wanted} is not callable.")
    return func


def _import_module(plugin: DiscoveredPlugin, module_name: str):
    search = [str(plugin.entry_path.parent)]
    spec = importlib.util.spec_from_file_location(
        module_name, plugin.entry_path, submodule_search_locations=search,
    )
    if spec is None or spec.loader is None:
        raise ImportError(f"Could not load {plugin.entry_path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[module_name] = module
    try:
        spec.loader.exec_module(module)
    except BaseException:
        sys.modules.pop(module_name, None)
        raise
    return module


def _ensure_namespace_package() -> None:
    """Create the ``mfv_plugins`` parent package if it does not exist yet."""
    if NAMESPACE in sys.modules:
        return
    import types

    package = types.ModuleType(NAMESPACE)
    package.__doc__ = "User plugins loaded from folders outside the application."
    package.__path__ = []          # a namespace package with no fixed location
    sys.modules[NAMESPACE] = package


def unload_plugin_modules() -> int:
    """
    Drop every imported user-plugin module. Returns how many were removed.

    Used by a rescan so an edited plugin is picked up without restarting.
    """
    names = [n for n in sys.modules if n == NAMESPACE or n.startswith(f"{NAMESPACE}.")]
    for name in names:
        sys.modules.pop(name, None)
    return len(names)


# ---------------------------------------------------------------------------
# trust
# ---------------------------------------------------------------------------

def root_is_confirmed(prefs: dict | None, root: str | Path) -> bool:
    """Whether the user has accepted running code from *root*."""
    confirmed = ((prefs or {}).get("plugin", {}) or {}).get("confirmed_dirs", []) or []
    target = _key(root)
    return any(_key(entry) == target for entry in confirmed)


def confirm_root(prefs: dict, root: str | Path) -> None:
    """Record that the user accepted *root*. Mutates *prefs* in place."""
    settings = prefs.setdefault("plugin", {})
    confirmed = list(settings.get("confirmed_dirs", []) or [])
    target = _key(root)
    if not any(_key(entry) == target for entry in confirmed):
        confirmed.append(str(root))
        settings["confirmed_dirs"] = confirmed


def _key(path: str | Path) -> str:
    try:
        return str(Path(path).expanduser().resolve()).lower()
    except OSError:
        return str(path).lower()
