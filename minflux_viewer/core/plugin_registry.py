'Plugin registry'

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass


@dataclass(frozen=True)
class PluginEntry:
    """Describes one plugin."""
    name: str
    tooltip: str
    launch: Callable[..., None]   # launch(state, parent=None)
    # Nested menu placement, e.g. ``("HlyB",)`` for *Plugins ▸ HlyB ▸ …*.
    menu_path: tuple[str, ...] = ()
    # Non-empty when the plugin was found but cannot run. The entry is still
    # listed — a plugin that silently fails to appear is a worse bug report
    # than one that appears and explains itself — and its action is disabled.
    error: str = ""
    # True for an entry created by ``discover()`` from a user folder, as
    # opposed to a built-in that registered itself at import time. ``rediscover``
    # drops only these.
    discovered: bool = False
    # Extra search tags for the Command Finder — synonyms and domain terms that
    # are NOT already in ``name``. Optional; the plugin is findable by its name
    # either way. Built-in menu commands get the equivalent from
    # ``ui/common/command_meta.py``, which is keyed by QAction attribute name and so
    # cannot describe a plugin action (it has none).
    keywords: tuple[str, ...] = ()
    # Implementing file shown by the Command Finder. Discovered plugins set
    # this to their real entry file rather than the registry wrapper above it.
    source: str = ""
    # Stable identity for a discovered plugin: the manifest id, or the file
    # stem for Tier 1. Empty for a built-in, whose name IS its identity.
    # ``discover`` de-duplicates on this rather than on the display name, so
    # two vendors may publish the same leaf label under different submenus.
    plugin_id: str = ""
    # The plugin's declared ``[method]`` block, or None. Carried on the entry
    # so the method-text generator and the macro recorder can reach it from a
    # recorded run without re-reading plugin.toml or importing the plugin.
    method: object | None = None


_REGISTRY: list[PluginEntry] = []


def register(entry: PluginEntry) -> None:
    """Add *entry* to the plugin registry (idempotent on name)."""
    for existing in _REGISTRY:
        if existing.name == entry.name:
            return
    _REGISTRY.append(entry)


def available() -> list[PluginEntry]:
    """Return all registered plugins in insertion order."""
    return list(_REGISTRY)
