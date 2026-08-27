"""A colormap dropdown that can group the solid colours under a child node.

The render view and the scatter plot show their colormaps as a ``QMenu`` with a
*Solid color* submenu; the LUT dialog shows the same list in a ``QComboBox``,
which is flat by default. A combo *can* be hierarchical — give it a
``QTreeView`` and a tree model — but three things break when you do, and this
widget exists to handle them so no caller has to:

* ``QComboBox.findText`` does **not** descend into children (it returns -1 for
  every solid), so the dialog's "select this map without emitting" helper
  silently stopped working;
* ``QComboBox`` tracks the branch it is showing in ``rootModelIndex``, and its
  built-in item-selected path leaves it pointing at the child's parent, so the
  next popup shows only that branch;
* clicking a group row would otherwise select the group and close the popup,
  rather than expanding it.

Selection therefore goes through :meth:`set_current_colormap` /
:meth:`current_colormap`, and the widget emits ``colormap_changed`` rather than
relying on ``currentTextChanged``.
"""

from __future__ import annotations

from PyQt6.QtCore import Qt, pyqtSignal
from PyQt6.QtGui import QStandardItem, QStandardItemModel
from PyQt6.QtWidgets import QComboBox, QTreeView

from ..colormaps import SOLID_GROUP_TOKEN, solid_color_names

#: Text of the folded group row (the same wording the render/scatter submenus use).
SOLID_GROUP_LABEL = "Solid color"


class ColormapComboBox(QComboBox):
    """Dropdown of colormaps, optionally with the solids under a group node."""

    #: A colormap was chosen by the user (never emitted by a programmatic set).
    colormap_changed = pyqtSignal(str)

    def __init__(self, parent=None) -> None:
        super().__init__(parent)
        self._tree = QTreeView(self)
        self._tree.setHeaderHidden(True)
        self._tree.setItemsExpandable(True)
        self._tree.setRootIsDecorated(True)
        self._tree.setUniformRowHeights(True)
        self._tree.setAllColumnsShowFocus(True)
        self.setView(self._tree)
        self._model = QStandardItemModel(self)
        self.setModel(self._model)
        self._entries: list[str] = []
        self._silent = False
        # ⚠ The user's pick is taken from ``currentTextChanged``, NOT from the
        # view's ``clicked``: ``QComboBoxPrivateContainer`` filters the mouse
        # release out of the popup view and calls the combo's own item-selected
        # path, so ``QTreeView.clicked`` never fires for a popup click (verified:
        # currentTextChanged / currentIndexChanged / activated all fire, clicked
        # does not). Wiring only ``clicked`` left the dropdown inert.
        self.currentTextChanged.connect(self._on_current_text_changed)
        # ``clicked`` is still connected for the one case Qt does deliver it --
        # a row that cannot be selected, i.e. the group heading -- so it toggles
        # the branch instead of doing nothing.
        self._tree.clicked.connect(self._on_row_clicked)

    # ------------------------------------------------------------- contents
    def set_entries(self, entries) -> None:
        """Rebuild from an ordered list; ``SOLID_GROUP_TOKEN`` becomes a node.

        The current selection is preserved when it still exists, so a rebuild
        (a new custom map, a reordered list) never silently changes the LUT.
        """
        current = self.current_colormap()
        self._entries = [str(name) for name in entries]
        # ⚠ Silent for the whole rebuild. Inserting the first row makes Qt
        # select it, and when the solid group leads the list that row is the
        # *heading* -- which is not a colormap. Unguarded, the combo announced
        # "Solid color" as a chosen colormap and briefly reported it as the
        # current one.
        previous_silent = self._silent
        self._silent = True
        self.blockSignals(True)
        try:
            self._build_rows()
        finally:
            self.blockSignals(False)
            self._silent = previous_silent
        # Keep the selection when it still exists, else fall back to the first
        # row that is actually a colormap, so the current text is never a
        # heading.
        if not (current and self.set_current_colormap(current, silent=True)):
            for name in self.colormap_names():
                if self.set_current_colormap(name, silent=True):
                    break

    def _build_rows(self) -> None:
        self._model.clear()
        root = self._model.invisibleRootItem()
        for name in self._entries:
            if name == SOLID_GROUP_TOKEN:
                group = QStandardItem(SOLID_GROUP_LABEL)
                # A group is a heading, not a colormap: it must not become the
                # current text, and it must not look editable.
                group.setSelectable(False)
                group.setEditable(False)
                group.setData(None, Qt.ItemDataRole.UserRole)
                for solid in solid_color_names():
                    child = QStandardItem(solid)
                    child.setEditable(False)
                    child.setData(solid, Qt.ItemDataRole.UserRole)
                    group.appendRow(child)
                root.appendRow(group)
                continue
            item = QStandardItem(name)
            item.setEditable(False)
            item.setData(name, Qt.ItemDataRole.UserRole)
            root.appendRow(item)
        self._tree.expandAll()

    def colormap_names(self) -> list[str]:
        """Every selectable name, groups expanded — what a flat combo would show."""
        names: list[str] = []
        for name in self._entries:
            if name == SOLID_GROUP_TOKEN:
                names.extend(solid_color_names())
                continue
            names.append(name)
        return names

    def contains(self, name: str) -> bool:
        target = str(name).casefold()
        return any(item.casefold() == target for item in self.colormap_names())

    # ------------------------------------------------------------ selection
    def current_colormap(self) -> str:
        return self.currentText()

    def set_current_colormap(self, name: str, *, silent: bool = False) -> bool:
        """Select *name*, descending into the group when needed.

        ``QComboBox.setCurrentText`` and ``findText`` only see the top level,
        which is why this exists. Returns ``False`` when the name is not in the
        list, leaving the selection alone.
        """
        index = self._index_for(str(name))
        if index is None:
            return False
        previous = self._silent
        self._silent = True if silent else previous
        try:
            root = self._model.invisibleRootItem().index()
            parent = index.parent()
            if parent.isValid():
                # Point the combo at the branch, take the row, then point it
                # back — otherwise the next popup shows only that branch.
                self.setRootModelIndex(parent)
                super().setCurrentIndex(index.row())
                self.setRootModelIndex(root)
            else:
                self.setRootModelIndex(root)
                super().setCurrentIndex(index.row())
        finally:
            self._silent = previous
        return True

    def _index_for(self, name: str):
        target = name.casefold()
        root = self._model.invisibleRootItem()
        for row in range(root.rowCount()):
            item = root.child(row)
            if item is None:
                continue
            if item.isSelectable() and item.text().casefold() == target:
                return item.index()
            for child_row in range(item.rowCount()):
                child = item.child(child_row)
                if child is not None and child.text().casefold() == target:
                    return child.index()
        return None

    def _on_row_clicked(self, index) -> None:
        """Toggle a group heading. Leaf rows are Qt's business, not ours."""
        item = self._model.itemFromIndex(index)
        if item is None or item.isSelectable():
            return
        self._tree.setExpanded(index, not self._tree.isExpanded(index))

    def _on_current_text_changed(self, text: str) -> None:
        """Republish Qt's selection as ``colormap_changed``.

        Silent while a programmatic ``set_current_colormap`` is running, so
        rebuilding the list or restoring a saved selection never looks like the
        user choosing a colormap.
        """
        name = str(text or "")
        # ``contains`` as well as the silent guard: a group heading is not a
        # colormap, and nothing downstream can resolve one.
        if self._silent or not name or not self.contains(name):
            return
        self.colormap_changed.emit(name)
