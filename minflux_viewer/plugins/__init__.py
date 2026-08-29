"""
minflux_viewer.plugins
=======================
Plugin infrastructure for the MINFLUX Data Viewer.

A *plugin* is anything that extends the viewer with optional functionality
that is only loaded when the user invokes it from the **Plugins** menu.
Each plugin exposes:

* A short human-readable ``NAME`` (used as the menu item label).
* A ``TOOLTIP`` (shown on hover).
* A ``launch(state, parent)`` callable that opens its UI. The callable
  receives the active :class:`AppState` and an optional parent widget
  (usually the main window).

Registration
------------
Plugins are registered by calling :func:`register` at import time. The
main window imports :mod:`minflux_viewer.plugins` during startup, which
in turn triggers the registration of every built-in plugin.

Third-party plugins can be added without touching the core by importing
the package and calling ``plugins.register(...)`` from their own
``__init__.py``.

Usage from the main window
--------------------------
.. code-block:: python

    from .. import plugins
    plugins.ensure_loaded()
    for entry in plugins.available():
        action = QAction(entry.name, self, triggered=lambda _=False, e=entry: e.launch(self._state, self))
        plugins_menu.addAction(action)
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, List


@dataclass(frozen=True)
class PluginEntry:
    """Describes one plugin."""
    name: str
    tooltip: str
    launch: Callable[..., None]   # launch(state, parent=None)
    # Extra search tags for the Command Finder — synonyms and domain terms that
    # are NOT already in ``name``. Optional; the plugin is findable by its name
    # either way. Built-in menu commands get the equivalent from
    # ``ui/command_meta.py``, which is keyed by QAction attribute name and so
    # cannot describe a plugin action (it has none).
    keywords: tuple[str, ...] = ()


_REGISTRY: List[PluginEntry] = []
_LOADED = False


def register(entry: PluginEntry) -> None:
    """Add *entry* to the plugin registry (idempotent on name)."""
    for existing in _REGISTRY:
        if existing.name == entry.name:
            return
    _REGISTRY.append(entry)


def available() -> List[PluginEntry]:
    """Return all registered plugins in insertion order."""
    return list(_REGISTRY)


def discover(prefs: dict | None = None) -> List[PluginEntry]:
    """
    Find and register user plugins from folders **outside** this package.

    Owned by extension-layer **track B**; Phase 0 lands it as a no-op so the
    main window can call it from ``_populate_plugins_menu`` without track B
    having started. Returns the entries it registered, so the caller can report
    how many were found.

    The design it will implement (``docs/extension-layer/PLAN.md`` 3.2-3.4):

    * three roots -- ``<app dir>/plugins/``, the per-user application-data
      folder, and any extra paths in ``prefs["plugin"]["paths"]``. When frozen,
      resolve the app directory from ``sys.executable``, **not** ``__file__``:
      ``_MEIPASS`` is the bundle's ``_internal/``, which is both the wrong place
      to look and replaced on every update.
    * two tiers -- a single ``.py`` file is one menu entry; a directory with a
      ``plugin.toml`` is a full plugin with a manifest.
    * load under a namespaced module name (``mfv_plugins.<id>``) so two users'
      plugins can both contain a ``utils.py``.
    * import the plugin body lazily, when its menu entry is first invoked, so a
      plugin that imports a heavy library does not slow every launch.
    * contain every failure: a broken plugin becomes a *disabled* entry whose
      tooltip is the error, and never stops the rest of the menu building.
    """
    return []


def ensure_loaded() -> None:
    """
    Import every built-in plugin package so each one gets to register.

    Called once from the main window before the Plugins menu is populated.
    Safe to call repeatedly.
    """
    global _LOADED
    if _LOADED:
        return
    _LOADED = True

    # Built-in plugins — registration (and thus Plugins-menu) order follows the
    # import order here.
    from . import msr_reader      # noqa: F401
    from . import paraview        # noqa: F401
    from . import data_simulator  # noqa: F401  (immediately under ParaView)
    from . import drift_correction  # noqa: F401
    from . import trace_viewer    # noqa: F401
    from . import spatial_line_pattern  # noqa: F401
    from . import hlyb_pair_analysis  # noqa: F401  (project-specific)
    from . import generate_method_text  # noqa: F401
    from . import script_editor   # noqa: F401  (moved to the bottom of the list)
