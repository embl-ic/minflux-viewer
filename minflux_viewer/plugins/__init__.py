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

import contextlib

from collections.abc import Callable
from dataclasses import dataclass, replace


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
    # Stable identity for a discovered plugin: the manifest id, or the file
    # stem for Tier 1. Empty for a built-in, whose name IS its identity.
    # ``discover`` de-duplicates on this rather than on the display name, so
    # two vendors may publish the same leaf label under different submenus.
    plugin_id: str = ""


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

    **Identity is the plugin id, not the display name.** Rescanning is
    idempotent because the same id is skipped, and two vendors may legitimately
    publish the same leaf label under different submenus
    (*Plugins ▸ Lab A ▸ Analyse* and *Plugins ▸ Lab B ▸ Analyse*). Since
    :func:`register` de-duplicates on ``name`` -- it must, for built-ins, whose
    name is all the identity they have -- a colliding label is **disambiguated**
    rather than dropped: the second entry is renamed ``"Analyse (lab.b)"``. A
    plugin that silently fails to appear is the failure this whole layer exists
    to avoid. ``menu_path`` is untouched, so the submenu still groups it.

    To pick up an *edited* plugin, call :func:`rediscover` instead.
    """
    from . import loader

    added: list[PluginEntry] = []
    known_ids = {e.plugin_id for e in _REGISTRY if e.plugin_id}
    used_names = {e.name for e in _REGISTRY}
    for root in loader.plugin_roots(prefs):
        for found in loader.scan_root(root):
            if found.id in known_ids:
                continue
            entry = _entry_for(found, prefs)
            if entry.name in used_names:
                entry = replace(entry, name=f"{entry.name} ({found.id})")
            register(entry)
            known_ids.add(found.id)
            used_names.add(entry.name)
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
        plugin_id=found.id,
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
    # A plugin is ONE logical step, not the API calls inside it: its own
    # ctx.journal.record() is what belongs in a recorded macro. Suppress the
    # per-call recording for the duration (design §3.3, and the plan's
    # "does it record the plugin, or the plugin's actions?").
    calls = getattr(ctx, "calls", None)
    suppression = calls.suppress() if calls is not None else contextlib.nullcontext()
    try:
        with suppression:
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
        macro_recorder,  # noqa: F401
        msr_reader,  # noqa: F401
        paraview,  # noqa: F401
        script_editor,  # noqa: F401  (moved to the bottom of the list)
        spatial_line_pattern,  # noqa: F401
        trace_viewer,  # noqa: F401
    )
