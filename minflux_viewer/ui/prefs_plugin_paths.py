"""
Preferences > Plugin > *Plugin folders* group.

Lives in its own file, rather than inside ``preferences_dialog.py``, so the
plugin-discovery and external-library work could each add a Preferences group
without both editing the same file. See ``docs/extension-layer/PLAN.md`` §4.3.

Seam contract — ``preferences_dialog.py`` calls exactly three things and must
not need changing again:

* ``build_plugin_paths_group(parent)`` at page-build time,
* ``group.load(prefs)`` when the dialog populates itself,
* ``group.save(prefs)`` when the user accepts.

``load``/``save`` take the dialog's **draft** preferences dict and mutate it in
place. In-place mutation is required: the modeless Preferences dialog writes
back only the leaves that differ between its baseline and the draft
(``core/app_state.py::merge_pref_changes``), so replacing the dict would be
seen as no change at all. ``plugin.paths`` is a list, and lists are values
rather than containers to that merge — it is replaced whole, which is what an
orderable path list wants.
"""

from __future__ import annotations

from pathlib import Path

from PyQt6.QtCore import Qt
from PyQt6.QtWidgets import (
    QCheckBox,
    QFileDialog,
    QGroupBox,
    QHBoxLayout,
    QLabel,
    QListWidget,
    QListWidgetItem,
    QPushButton,
    QVBoxLayout,
    QWidget,
)


class PluginPathsGroup(QGroupBox):
    """Extra plugin folders, and whether the per-user folder is scanned."""

    def __init__(self, parent: QWidget | None = None) -> None:
        super().__init__("Plugin folders", parent)
        layout = QVBoxLayout(self)
        layout.setContentsMargins(12, 8, 12, 10)
        layout.setSpacing(6)

        self._standard = QLabel()
        self._standard.setWordWrap(True)
        self._standard.setTextInteractionFlags(
            Qt.TextInteractionFlag.TextSelectableByMouse
        )
        layout.addWidget(self._standard)

        self._scan_user = QCheckBox("scan the per-user plugin folder")
        self._scan_user.setToolTip(
            "The default place a plugin is installed. Per-machine and "
            "per-account, so another user of this computer does not see your "
            "plugins, and an application update never touches it."
        )
        layout.addWidget(self._scan_user)

        layout.addWidget(QLabel("Additional folders:"))
        self._list = QListWidget()
        self._list.setMinimumHeight(72)
        self._list.setToolTip(
            "Extra folders scanned for plugins — a development checkout, or a "
            "folder shared by a group."
        )
        layout.addWidget(self._list)

        buttons = QHBoxLayout()
        add = QPushButton("Add…")
        add.clicked.connect(self._add_folder)
        buttons.addWidget(add)
        remove = QPushButton("Remove")
        remove.clicked.connect(self._remove_selected)
        buttons.addWidget(remove)
        buttons.addStretch()
        rescan = QPushButton("Rescan now")
        rescan.setToolTip(
            "Re-read every plugin folder and reload edited plugins. "
            "Takes effect after Preferences is accepted."
        )
        rescan.clicked.connect(self._rescan)
        buttons.addWidget(rescan)
        layout.addLayout(buttons)

        self._status = QLabel()
        self._status.setWordWrap(True)
        layout.addWidget(self._status)

    # -- seam ---------------------------------------------------------------

    def load(self, prefs: dict) -> None:
        """Populate the widgets from *prefs* (the dialog's draft)."""
        from ..plugins import loader

        settings = (prefs or {}).get("plugin", {}) or {}
        self._scan_user.setChecked(bool(settings.get("scan_user_dir", True)))

        self._list.clear()
        for path in settings.get("paths", []) or []:
            self._add_row(str(path))

        self._standard.setText(
            "Always scanned:\n"
            f"    {loader.app_plugin_dir()}    (beside the application)\n"
            f"    {loader.user_plugin_dir()}    (per-user)"
        )
        self._refresh_status(prefs)

    def save(self, prefs: dict) -> None:
        """Write the widget values back into *prefs*, in place."""
        settings = prefs.setdefault("plugin", {})
        settings["scan_user_dir"] = bool(self._scan_user.isChecked())
        settings["paths"] = [
            self._list.item(row).text() for row in range(self._list.count())
        ]

    # -- widgets ------------------------------------------------------------

    def _add_row(self, path: str) -> None:
        item = QListWidgetItem(path)
        if not Path(path).expanduser().is_dir():
            item.setToolTip("This folder does not exist; it is skipped when scanning.")
        self._list.addItem(item)

    def _add_folder(self) -> None:
        path = QFileDialog.getExistingDirectory(self, "Choose a plugin folder")
        if not path:
            return
        existing = {self._list.item(i).text() for i in range(self._list.count())}
        if path not in existing:
            self._add_row(path)
            self._refresh_status(None)

    def _remove_selected(self) -> None:
        for item in self._list.selectedItems():
            self._list.takeItem(self._list.row(item))
        self._refresh_status(None)

    def _rescan(self) -> None:
        """
        Count what the current folder list holds.

        Deliberately only *counts* — it does not register anything. The draft
        preferences have not been accepted yet, so registering from them would
        apply a setting the user may still cancel.
        """
        from ..plugins import loader

        prefs = {"plugin": {}}
        self.save(prefs)
        total = 0
        broken = 0
        for root in loader.plugin_roots(prefs):
            for found in loader.scan_root(root):
                total += 1
                broken += bool(found.error)
        self._status.setText(self._summary(total, broken) + "  Accept Preferences to apply.")

    def _refresh_status(self, prefs: dict | None) -> None:
        from ..plugins import loader

        draft = {"plugin": {}}
        self.save(draft)
        total = 0
        broken = 0
        for root in loader.plugin_roots(draft):
            for found in loader.scan_root(root):
                total += 1
                broken += bool(found.error)
        self._status.setText(self._summary(total, broken))

    @staticmethod
    def _summary(total: int, broken: int) -> str:
        if total == 0:
            return "No plugins found in these folders."
        text = f"{total} plugin(s) found."
        if broken:
            text += f" {broken} cannot run; the Plugins menu says why."
        return text


def build_plugin_paths_group(parent: QWidget | None = None) -> PluginPathsGroup:
    """Construct the group. The only entry point ``preferences_dialog`` uses."""
    return PluginPathsGroup(parent)
