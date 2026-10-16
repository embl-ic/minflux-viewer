'Metadata helpers'

from __future__ import annotations

import numpy as np
from PyQt6.QtWidgets import (
    QTreeWidgetItem,
)


def _is_numeric_dtype(dt) -> bool:
    try:
        return np.issubdtype(dt, np.integer) or np.issubdtype(dt, np.floating) or np.issubdtype(dt, np.bool_)
    except Exception:
        return False


def _json_compact(node):
    """Readability transform for Imspector's XML-derived metadata dicts.

    Imspector OME metadata packs every XML element as a single-item list of
    dicts whose attributes are named keys and whose child content sits under
    an empty-string key. This unwraps single-element lists and hoists the
    ``""`` content next to the attributes, so ``{"Name": [{"": "x"}]}``
    reads as ``Name: x``. Generic JSON without those patterns passes through
    unchanged.
    """
    if isinstance(node, dict):
        if set(node.keys()) == {""}:
            return _json_compact(node[""])
        merged = {k: _json_compact(v) for k, v in node.items() if k != ""}
        if "" in node:
            content = _json_compact(node[""])
            if isinstance(content, dict):
                for key, val in content.items():
                    merged[key if key not in merged else f"{key} (content)"] = val
            else:
                merged["(text)" if "(text)" not in merged else "(text content)"] = content
        return merged
    if isinstance(node, (list, tuple)):
        items = [_json_compact(v) for v in node]
        return items[0] if len(items) == 1 else items
    return node


def _json_type_name(value) -> str:
    if value is None:
        return "null"
    if isinstance(value, bool):
        return "bool"
    if isinstance(value, int):
        return "int"
    if isinstance(value, float):
        return "float"
    if isinstance(value, str):
        return "str"
    if isinstance(value, dict):
        return "object"
    if isinstance(value, (list, tuple)):
        return "list"
    return type(value).__name__


def _json_leaf_text(value) -> str:
    if value is None:
        return "null"
    if isinstance(value, bool):
        return "true" if value else "false"
    return str(value)


def _populate_metadata_items(parent_item, node, *, _budget: list[int] | None = None) -> None:
    """Hierarchical key/value items for (compacted) JSON metadata in the
    main reader tree.

    Main-tree columns are reused as: leaf -> "key: value" | type;
    container -> key | object/list | item count. ``_budget`` caps the total
    number of created items so a pathological metadata blob cannot freeze
    the tree.
    """
    if _budget is None:
        _budget = [5000]
    pairs = node.items() if isinstance(node, dict) else ((f"[{i}]", v) for i, v in enumerate(node))
    for key, value in pairs:
        if _budget[0] <= 0:
            parent_item.addChild(QTreeWidgetItem(["… (truncated)", "", "", ""]))
            return
        _budget[0] -= 1
        if isinstance(value, (dict, list, tuple)):
            n = len(value)
            count = f"{n} key{'s' if n != 1 else ''}" if isinstance(value, dict) else f"{n} item{'s' if n != 1 else ''}"
            item = QTreeWidgetItem([str(key), _json_type_name(value), count, ""])
            parent_item.addChild(item)
            _populate_metadata_items(item, value, _budget=_budget)
        else:
            item = QTreeWidgetItem([f"{key}: {_json_leaf_text(value)}", _json_type_name(value), "", ""])
            item.setToolTip(0, f"{key} = {_json_leaf_text(value)}")
            parent_item.addChild(item)
