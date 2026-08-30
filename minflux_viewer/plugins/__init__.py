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
    # ``ui/command_meta.py``, which is keyed by QAction attribute name and so
    # cannot describe a plugin action (it has none).
    keywords: tuple[str, ...] = ()
    # Implementing file shown by the Command Finder. Discovered plugins set
    # this to their real entry file rather than the registry wrapper above it.
    source: str = ""


_REGISTRY: list[PluginEntry] = []
_LOADED = False


def register(entry: PluginEntry) -> None:
    """Add *entry* to the plugin registry (idempotent on name)."""
    for existing in _REGISTRY:
        if existing.name == entry.name:
            return
    _REGISTRY.append(entry)


def available() -> list[PluginEntry]:
    """Return all registered plugins in insertion order."""
    return list(_REGISTRY)


def discover(prefs: dict | None = None) -> list[PluginEntry]:
    """
    Find and register user plugins from folders **outside** this package.

    Scans the three roots (:func:`loader.plugin_roots`), builds one
    :class:`PluginEntry` per plugin found, and registers those not already
    registered. Returns the entries it added, so the caller can report how many
    were found.

    Nothing is imported here: the plugin body is loaded when its menu entry is
    first invoked. A plugin that cannot run is still registered, carrying the
    reason -- silently missing is a worse bug report than present-and-explained.

    Registration is by display name, so calling this repeatedly (a rescan) does
    not duplicate entries. To pick up an *edited* plugin, call
    :func:`rediscover` instead.
    """
    from . import loader

    added: list[PluginEntry] = []
    known = {entry.name for entry in _REGISTRY}
    for root in loader.plugin_roots(prefs):
        for found in loader.scan_root(root):
            entry = _entry_for(found, prefs)
            if entry.name in known:
                continue
            register(entry)
            known.add(entry.name)
            added.append(entry)
    return added


def rediscover(prefs: dict | None = None) -> list[PluginEntry]:
    """
    Drop every discovered user plugin and scan again.

    Used by *Preferences ▸ Plugin ▸ Rescan now* so an edited plugin is picked
    up without restarting. Built-in plugins are untouched: only entries this
    module created are removed, and the imported plugin modules are dropped
    from ``sys.modules`` so the next launch re-imports from disk.
    """
    from . import loader

    # Mutate in place rather than rebinding: anything holding a reference to
    # the registry list (including ``ensure_loaded``'s idea of what is already
    # registered) must see the same object.
    _REGISTRY[:] = [e for e in _REGISTRY if not e.discovered]
    loader.unload_plugin_modules()
    return discover(prefs)


def _entry_for(found, prefs: dict | None) -> PluginEntry:
    """Build the registry entry for one discovered plugin."""

    def launch(state, parent=None, *, _found=found, _prefs=prefs) -> None:
        _launch_discovered(_found, state, parent, _prefs)

    return PluginEntry(
        name=found.label,
        tooltip=found.tooltip,
        launch=launch,
        menu_path=found.menu_path,
        keywords=found.keywords,
        error=found.error,
        discovered=True,
        source=str(found.entry_path),
    )


def _launch_discovered(found, state, parent, prefs: dict | None) -> None:
    """
    Run one discovered plugin, containing every failure.

    Three gates, in order: the plugin must be runnable at all; the user must
    have accepted the folder it came from; and the import must succeed. Each
    failure is reported where the user is looking -- a message box when there is
    a window, and always the Log -- rather than as a traceback into the void.
    """
    from . import loader

    log = getattr(state, "log", None)

    def report(message: str, level: str = "ERROR") -> None:
        if callable(log):
            log(message, level)
        _show_message(parent, message, level)

    if found.error:
        report(f"Plugin '{found.label}' cannot run: {found.error}")
        return

    settings = prefs if prefs is not None else getattr(state, "prefs", None)
    if settings is not None and not loader.root_is_confirmed(settings, found.root):
        if not _confirm_root(parent, found.root):
            if callable(log):
                log(f"Plugins in {found.root} were not enabled.", "INFO")
            return
        loader.confirm_root(settings, found.root)
        save = getattr(state, "save_prefs", None)
        if callable(save):
            try:
                save()
            except Exception:
                pass

    try:
        func = loader.load_entry_callable(found)
    except BaseException as exc:                # noqa: BLE001 - third-party code
        report(f"Plugin '{found.label}' failed to load: {exc!r}")
        return

    ctx = getattr(state, "mfv", None)
    try:
        func(ctx)
    except BaseException as exc:                # noqa: BLE001 - third-party code
        report(f"Plugin '{found.label}' failed: {exc!r}")


def _confirm_root(parent, root) -> bool:
    """
    Ask once before running code from a folder the user has not accepted.

    Deliberately asked at **first launch**, not at scan: nothing is imported
    while the menu is built, so listing a plugin is harmless and a dialog at
    startup would be noise. This is the moment the trust actually matters.

    With no window (headless, tests) there is nobody to ask, so the folder is
    accepted -- a script-driven session has already chosen its preferences.
    """
    try:
        from PyQt6.QtWidgets import QApplication, QMessageBox
    except Exception:
        return True
    if QApplication.instance() is None:
        return True
    answer = QMessageBox.question(
        parent, "Enable plugins from this folder?",
        f"Run plugins from:\n\n{root}\n\n"
        "Plugins are ordinary Python with full access to your files and this "
        "application — the same as a Fiji plugin or a Script Editor script. "
        "Enable this folder only if you trust what is in it.",
    )
    return answer == QMessageBox.StandardButton.Yes


def _show_message(parent, message: str, level: str) -> None:
    """
    Report a plugin failure in a box, **without blocking**, beside its window.

    Two deliberate choices:

    * Not ``QMessageBox.critical(...)``. The static helpers run their own event
      loop, so a failure reported from a scripted or automated run — or from
      anywhere nobody is there to click OK — stops the application dead.
      Nothing here needs an answer, so nothing here waits for one.
      (``_confirm_root`` *is* modal, correctly: it does need an answer.)
    * **No parent, no box.** Every real call site passes the main window, which
      owns and cleans up the box. A parentless one would be an unowned
      top-level widget nothing closes — the shape this project's window rules
      exist to avoid — and the message is in the Log either way.
    """
    if parent is None:
        return
    try:
        from PyQt6.QtCore import Qt
        from PyQt6.QtWidgets import QApplication, QMessageBox
    except Exception:
        return
    if QApplication.instance() is None:
        return

    box = QMessageBox(parent)
    box.setWindowTitle("Plugin")
    box.setText(message)
    box.setIcon(
        QMessageBox.Icon.Critical if level == "ERROR" else QMessageBox.Icon.Warning
    )
    box.setAttribute(Qt.WidgetAttribute.WA_DeleteOnClose, True)
    box.show()


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
    from . import (
        data_simulator,  # noqa: F401  (immediately under ParaView)
        drift_correction,  # noqa: F401
        generate_method_text,  # noqa: F401
        msr_reader,  # noqa: F401
        paraview,  # noqa: F401
        script_editor,  # noqa: F401  (moved to the bottom of the list)
        spatial_line_pattern,  # noqa: F401
        trace_viewer,  # noqa: F401
    )
