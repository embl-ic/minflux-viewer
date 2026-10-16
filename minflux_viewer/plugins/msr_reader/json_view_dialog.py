'Json view dialog'

from __future__ import annotations

import json

from PyQt6.QtCore import Qt
from PyQt6.QtWidgets import (
    QApplication,
    QCheckBox,
    QDialog,
    QDialogButtonBox,
    QHBoxLayout,
    QLineEdit,
    QMenu,
    QPlainTextEdit,
    QPushButton,
    QTabWidget,
    QTreeWidget,
    QTreeWidgetItem,
    QVBoxLayout,
)

from minflux_viewer.plugins.msr_reader.metadata_helpers import (
    _json_compact,
    _json_leaf_text,
    _json_type_name,
)


class JsonViewDialog(QDialog):
    """Browsable JSON / metadata viewer.

    Tree tab: key | value | type columns with a filter box, compact-view
    toggle (unwraps Imspector's XML packing), expand/collapse, and copy actions.
    Raw tab: the plain ``json.dumps`` text as before.
    """

    def __init__(self, title: str, data, parent=None):
        super().__init__(parent)
        self.setWindowTitle(title)
        self.resize(900, 680)
        self._data = data

        layout = QVBoxLayout(self)

        bar = QHBoxLayout()
        self._filter_edit = QLineEdit(self)
        self._filter_edit.setPlaceholderText("Filter keys and values…")
        self._filter_edit.setClearButtonEnabled(True)
        self._filter_edit.textChanged.connect(self._apply_filter)
        bar.addWidget(self._filter_edit, 1)
        self._compact_chk = QCheckBox("Compact", self)
        self._compact_chk.setChecked(True)
        self._compact_chk.setToolTip(
            "Unwrap single-element lists and XML content wrappers for readability."
        )
        self._compact_chk.toggled.connect(self._rebuild)
        bar.addWidget(self._compact_chk)
        expand_btn = QPushButton("Expand all", self)
        expand_btn.clicked.connect(lambda: self.tree.expandAll())
        bar.addWidget(expand_btn)
        collapse_btn = QPushButton("Collapse all", self)
        collapse_btn.clicked.connect(lambda: self.tree.collapseAll())
        bar.addWidget(collapse_btn)
        layout.addLayout(bar)

        self._tabs = QTabWidget(self)
        self.tree = QTreeWidget(self)
        self.tree.setColumnCount(3)
        self.tree.setHeaderLabels(["Key", "Value", "Type"])
        self.tree.setAlternatingRowColors(True)
        self.tree.setContextMenuPolicy(Qt.ContextMenuPolicy.CustomContextMenu)
        self.tree.customContextMenuRequested.connect(self._show_tree_menu)
        self._tabs.addTab(self.tree, "Tree")
        self._raw_edit = QPlainTextEdit(self)
        self._raw_edit.setReadOnly(True)
        self._raw_edit.setPlainText(self._dumps(data))
        self._tabs.addTab(self._raw_edit, "Raw JSON")
        layout.addWidget(self._tabs, 1)

        buttons = QDialogButtonBox(QDialogButtonBox.StandardButton.Close)
        copy_btn = buttons.addButton("Copy JSON", QDialogButtonBox.ButtonRole.ActionRole)
        copy_btn.clicked.connect(lambda: QApplication.clipboard().setText(self._dumps(self._data)))
        buttons.rejected.connect(self.reject)
        buttons.accepted.connect(self.accept)
        buttons.button(QDialogButtonBox.StandardButton.Close).clicked.connect(self.close)
        layout.addWidget(buttons)

        self._rebuild()

    @staticmethod
    def _dumps(data) -> str:
        return json.dumps(data, indent=2, ensure_ascii=False, default=str)

    def _display_data(self):
        return _json_compact(self._data) if self._compact_chk.isChecked() else self._data

    def _rebuild(self) -> None:
        self.tree.clear()
        data = self._display_data()
        root = self.tree.invisibleRootItem()
        if isinstance(data, (dict, list, tuple)):
            self._populate(root, data, path="")
        else:
            self._add_leaf(root, "(value)", data, path="")
        self.tree.expandToDepth(2)
        for col in range(3):
            self.tree.resizeColumnToContents(col)
        self.tree.setColumnWidth(0, min(self.tree.columnWidth(0), 380))
        self._apply_filter(self._filter_edit.text())

    def _populate(self, parent, node, path: str) -> None:
        if isinstance(node, dict):
            pairs = node.items()
        else:
            pairs = ((f"[{i}]", v) for i, v in enumerate(node))
        for key, value in pairs:
            child_path = f"{path}/{key}" if path else str(key)
            if isinstance(value, (dict, list, tuple)):
                n = len(value)
                summary = f"{{{n} key{'s' if n != 1 else ''}}}" if isinstance(value, dict) else f"[{n} item{'s' if n != 1 else ''}]"
                item = QTreeWidgetItem([str(key), summary, _json_type_name(value)])
                item.setForeground(1, Qt.GlobalColor.gray)
                item.setForeground(2, Qt.GlobalColor.gray)
                item.setData(0, Qt.ItemDataRole.UserRole, child_path)
                item.setData(0, Qt.ItemDataRole.UserRole + 1, value)
                parent.addChild(item)
                self._populate(item, value, child_path)
            else:
                self._add_leaf(parent, str(key), value, child_path)

    def _add_leaf(self, parent, key: str, value, path: str) -> None:
        item = QTreeWidgetItem([key, _json_leaf_text(value), _json_type_name(value)])
        item.setForeground(2, Qt.GlobalColor.gray)
        item.setToolTip(1, _json_leaf_text(value))
        item.setData(0, Qt.ItemDataRole.UserRole, path)
        item.setData(0, Qt.ItemDataRole.UserRole + 1, value)
        parent.addChild(item)

    # -- filtering ------------------------------------------------------

    def _apply_filter(self, text: str) -> None:
        needle = (text or "").strip().lower()

        def visit(item) -> bool:
            self_match = bool(needle) and (
                needle in item.text(0).lower() or needle in item.text(1).lower()
            )
            child_match = False
            for i in range(item.childCount()):
                if visit(item.child(i)):
                    child_match = True
            visible = not needle or self_match or child_match
            item.setHidden(not visible)
            if needle and child_match:
                item.setExpanded(True)
            return self_match or child_match

        root = self.tree.invisibleRootItem()
        for i in range(root.childCount()):
            visit(root.child(i))

    # -- context menu ---------------------------------------------------

    def _show_tree_menu(self, pos) -> None:
        item = self.tree.itemAt(pos)
        if item is None:
            return
        menu = QMenu(self)
        copy_key = menu.addAction("Copy key")
        copy_value = menu.addAction("Copy value")
        copy_path = menu.addAction("Copy path")
        action = menu.exec(self.tree.viewport().mapToGlobal(pos))
        if action is copy_key:
            QApplication.clipboard().setText(item.text(0))
        elif action is copy_value:
            value = item.data(0, Qt.ItemDataRole.UserRole + 1)
            text = self._dumps(value) if isinstance(value, (dict, list, tuple)) else _json_leaf_text(value)
            QApplication.clipboard().setText(text)
        elif action is copy_path:
            QApplication.clipboard().setText(str(item.data(0, Qt.ItemDataRole.UserRole) or ""))
