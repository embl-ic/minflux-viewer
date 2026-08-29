"""
Preferences > Plugin > *Plugin folders* group.

Owned by extension-layer **track B** (plugin discovery). It lives in its own
file, rather than inside ``preferences_dialog.py``, purely so that track B and
track D can each add a Preferences group without both editing the same file
while they run in parallel. See ``docs/extension-layer/PLAN.md`` section 4.3.

Seam contract -- ``preferences_dialog.py`` calls exactly three things and must
not need changing again:

* ``build_plugin_paths_group(parent)`` at page-build time,
* ``group.load(prefs)`` when the dialog populates itself,
* ``group.save(prefs)`` when the user accepts.

``load``/``save`` take the dialog's **draft** preferences dict and mutate it in
place. In-place mutation is required: the modeless Preferences dialog writes
back only the leaves that differ between its baseline and the draft
(``core/app_state.py::merge_pref_changes``), so replacing the dict would be
seen as no change at all. The keys are ``plugin.paths``,
``plugin.scan_user_dir`` and ``plugin.confirmed_dirs``, declared in
``DEFAULT_PREFS``.

Note that ``plugin.paths`` is a **list**, and lists are values rather than
containers to the preference merge -- it is replaced whole, which is what a
reorderable path list wants.
"""

from __future__ import annotations

from PyQt6.QtWidgets import QGroupBox, QLabel, QVBoxLayout, QWidget


class PluginPathsGroup(QGroupBox):
    """Extra plugin folders, and whether the per-user folder is scanned."""

    def __init__(self, parent: QWidget | None = None) -> None:
        super().__init__("Plugin folders", parent)
        layout = QVBoxLayout(self)
        layout.setContentsMargins(12, 8, 12, 10)
        layout.setSpacing(6)
        self._placeholder = QLabel(
            "Plugin folder discovery is not enabled in this build."
        )
        self._placeholder.setWordWrap(True)
        layout.addWidget(self._placeholder)

    # -- seam ---------------------------------------------------------------

    def load(self, prefs: dict) -> None:
        """Populate the widgets from *prefs* (the dialog's draft)."""
        return None

    def save(self, prefs: dict) -> None:
        """Write the widget values back into *prefs*, in place."""
        return None


def build_plugin_paths_group(parent: QWidget | None = None) -> PluginPathsGroup:
    """Construct the group. The only entry point ``preferences_dialog`` uses."""
    return PluginPathsGroup(parent)
