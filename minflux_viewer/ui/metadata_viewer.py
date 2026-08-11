"""Shared, structured metadata document viewer for XML and JSON sources."""

from __future__ import annotations

import html
import json
from xml.etree import ElementTree as ET

from PyQt6.QtGui import QFontDatabase
from PyQt6.QtWidgets import (
    QApplication,
    QComboBox,
    QDialog,
    QDialogButtonBox,
    QHBoxLayout,
    QHeaderView,
    QLabel,
    QLineEdit,
    QPushButton,
    QTabWidget,
    QTextBrowser,
    QTreeWidget,
    QTreeWidgetItem,
    QVBoxLayout,
    QWidget,
)

from ..core.tiff_source import MetadataDocument


def _local_name(tag: str) -> str:
    return str(tag).rsplit("}", 1)[-1]


def _pretty_xml(content: str) -> str:
    try:
        root = ET.fromstring(content)
        ET.indent(root, space="  ")
        return ET.tostring(root, encoding="unicode", xml_declaration=True)
    except Exception:
        return content


def _pretty_json(content: str) -> str:
    try:
        return json.dumps(json.loads(content), indent=2, ensure_ascii=False)
    except Exception:
        return content


class MetadataDocumentView(QWidget):
    """Name/value table plus a readable raw-document preview."""

    def __init__(self, documents=(), parent=None) -> None:
        super().__init__(parent)
        self._documents: tuple[MetadataDocument, ...] = ()

        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        bar = QHBoxLayout()
        self._document_label = QLabel("Document:")
        self._document_combo = QComboBox()
        self._document_combo.currentIndexChanged.connect(self._show_document)
        bar.addWidget(self._document_label)
        bar.addWidget(self._document_combo, 1)
        self._filter = QLineEdit()
        self._filter.setPlaceholderText("Filter names and values…")
        self._filter.setClearButtonEnabled(True)
        self._filter.textChanged.connect(self._apply_filter)
        bar.addWidget(self._filter, 1)
        expand = QPushButton("Expand all")
        expand.clicked.connect(lambda: self.tree.expandAll())
        collapse = QPushButton("Collapse all")
        collapse.clicked.connect(lambda: self.tree.collapseAll())
        bar.addWidget(expand)
        bar.addWidget(collapse)
        layout.addLayout(bar)

        self.tabs = QTabWidget()
        self.tree = QTreeWidget()
        self.tree.setAlternatingRowColors(True)
        self._configure_columns("xml")
        self.tabs.addTab(self.tree, "Table")
        self.raw = QTextBrowser()
        self.raw.setOpenExternalLinks(False)
        self.raw.setFont(QFontDatabase.systemFont(QFontDatabase.SystemFont.FixedFont))
        self.tabs.addTab(self.raw, "Raw")
        layout.addWidget(self.tabs, 1)
        self.set_documents(documents)

    def set_documents(self, documents) -> None:
        self._documents = tuple(documents or ())
        self._document_combo.blockSignals(True)
        self._document_combo.clear()
        for document in self._documents:
            self._document_combo.addItem(document.name)
        self._document_combo.blockSignals(False)
        visible = len(self._documents) > 1
        self._document_label.setVisible(visible)
        self._document_combo.setVisible(visible)
        self._show_document(0)

    def current_document(self) -> MetadataDocument | None:
        index = self._document_combo.currentIndex()
        return self._documents[index] if 0 <= index < len(self._documents) else None

    def _show_document(self, index: int) -> None:
        self.tree.clear()
        document = self.current_document()
        if document is None:
            self.raw.setHtml("<p><i>No metadata document is available.</i></p>")
            return
        fmt = document.format.strip().lower()
        self._configure_columns(fmt)
        if fmt == "xml":
            pretty = _pretty_xml(document.content)
            self.tabs.setTabText(0, "XML table")
            self.tabs.setTabText(1, "Raw XML")
            try:
                root = ET.fromstring(document.content)
                self._add_xml_item(self.tree.invisibleRootItem(), root)
            except Exception as exc:
                self.tree.addTopLevelItem(QTreeWidgetItem(["Invalid XML", str(exc)]))
        elif fmt == "json":
            pretty = _pretty_json(document.content)
            self.tabs.setTabText(0, "Table")
            self.tabs.setTabText(1, "Raw JSON")
            try:
                self._add_json_items(self.tree.invisibleRootItem(), json.loads(document.content))
            except Exception as exc:
                self.tree.addTopLevelItem(QTreeWidgetItem(["Invalid JSON", str(exc), ""]))
        else:
            pretty = document.content
            self.tabs.setTabText(0, "Lines")
            self.tabs.setTabText(1, "Raw text")
            for number, line in enumerate(pretty.splitlines(), 1):
                self.tree.addTopLevelItem(QTreeWidgetItem([str(number), line]))
        self.raw.setHtml(
            "<html><head><style>body{background:#fff;color:#202124;}"
            "pre{white-space:pre-wrap;word-wrap:break-word;font-family:monospace;"
            "font-size:12px;line-height:1.35;}</style></head><body><pre>"
            + html.escape(pretty) + "</pre></body></html>"
        )
        self.tree.expandToDepth(1)
        self._apply_filter(self._filter.text())

    def _configure_columns(self, fmt: str) -> None:
        if fmt == "json":
            self.tree.setColumnCount(3)
            self.tree.setHeaderLabels(["Name", "Value", "Type"])
        else:
            self.tree.setColumnCount(2)
            self.tree.setHeaderLabels(["Name", "Value"])
        self.tree.header().setSectionResizeMode(
            0, QHeaderView.ResizeMode.ResizeToContents
        )
        for column in range(1, self.tree.columnCount()):
            self.tree.header().setSectionResizeMode(column, QHeaderView.ResizeMode.Stretch)

    def _add_xml_item(self, parent, element) -> None:
        text = (element.text or "").strip()
        item = QTreeWidgetItem([_local_name(element.tag), text])
        parent.addChild(item)
        # XML attributes are independent name/value facts. Keeping them as
        # child rows avoids an ambiguous comma-joined cell (attribute values,
        # such as OME Creator, can themselves legitimately contain commas).
        for name, value in element.attrib.items():
            item.addChild(QTreeWidgetItem([_local_name(name), str(value)]))
        for child in element:
            self._add_xml_item(item, child)

    def _add_json_items(self, parent, value, name="root") -> None:
        if isinstance(value, dict):
            item = QTreeWidgetItem([str(name), "", f"{len(value)} keys"])
            parent.addChild(item)
            for key, child in value.items():
                self._add_json_items(item, child, str(key))
        elif isinstance(value, list):
            item = QTreeWidgetItem([str(name), "", f"{len(value)} items"])
            parent.addChild(item)
            for index, child in enumerate(value):
                self._add_json_items(item, child, f"[{index}]")
        else:
            parent.addChild(QTreeWidgetItem([str(name), str(value), type(value).__name__]))

    def _apply_filter(self, query: str) -> None:
        needle = query.strip().casefold()

        def visit(item: QTreeWidgetItem) -> bool:
            child_match = False
            for index in range(item.childCount()):
                child_match = visit(item.child(index)) or child_match
            own_match = not needle or any(
                needle in item.text(i).casefold()
                for i in range(self.tree.columnCount())
            )
            visible = own_match or child_match
            item.setHidden(not visible)
            if needle and child_match:
                item.setExpanded(True)
            return visible

        root = self.tree.invisibleRootItem()
        for index in range(root.childCount()):
            visit(root.child(index))


class MetadataViewDialog(QDialog):
    def __init__(self, title: str, documents, parent=None) -> None:
        super().__init__(parent)
        self.setWindowTitle(title)
        self.resize(980, 720)
        layout = QVBoxLayout(self)
        self.view = MetadataDocumentView(documents, self)
        layout.addWidget(self.view, 1)
        buttons = QDialogButtonBox(QDialogButtonBox.StandardButton.Close)
        copy = buttons.addButton("Copy raw", QDialogButtonBox.ButtonRole.ActionRole)
        copy.clicked.connect(self._copy_raw)
        buttons.button(QDialogButtonBox.StandardButton.Close).clicked.connect(self.close)
        layout.addWidget(buttons)

    def _copy_raw(self) -> None:
        document = self.view.current_document()
        if document is not None:
            QApplication.clipboard().setText(document.content)
