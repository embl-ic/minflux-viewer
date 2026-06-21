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

    # Built-in plugins — add more imports here as new plugins are authored.
    from . import msr_reader      # noqa: F401
    from . import paraview        # noqa: F401
    from . import script_editor   # noqa: F401
    from . import trace_viewer    # noqa: F401
    from . import generate_method_text  # noqa: F401
