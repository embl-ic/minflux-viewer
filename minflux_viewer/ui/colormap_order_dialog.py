"""Reorder the colormap list — LUT dialog ▸ Custom ▸ Reorder colormap list…

The list every colormap selector shows is one order, shared application-wide
(``colormaps.channel_colormap_names``). This dialog is where it is set: drag a
row, or sort the whole list by name or by when it was added.

**Solid colours are handled apart from the colormaps**, because they are a
different kind of thing and there are ten of them. Folded (the default) they
collapse into one *Solid color* group row: the group's position is draggable
like any other row, but what is inside it — and in which order — stays the
COLOR dialog's Solid Color List, which is the one place solids are managed.
Unfolded, each becomes an ordinary row that can be dragged anywhere, which is
what makes a frequently used one such as ``Gray`` reachable at the top.

The sort buttons deliberately do **not** touch the solids: sorting a category
alphabetically alongside the maps would scatter them through the list, undoing
the grouping that makes the list readable.
"""

from __future__ import annotations

from PyQt6.QtCore import Qt
from PyQt6.QtGui import QColor, QIcon, QLinearGradient, QPainter, QPixmap
from PyQt6.QtWidgets import (
    QAbstractItemView,
    QCheckBox,
    QDialog,
    QDialogButtonBox,
    QHBoxLayout,
    QLabel,
    QListWidget,
    QListWidgetItem,
    QPushButton,
    QVBoxLayout,
)

from ..colormaps import (
    SOLID_GROUP_TOKEN,
    colormap_lut,
    custom_colormap_names,
    is_solid_color,
    solid_color_names,
)

#: Label of the folded solid-colour group row.
SOLID_GROUP_LABEL = "Solid color"
#: Size of the swatch drawn beside each row.
_SWATCH = (72, 14)


def sort_entries(entries, key: str, *, descending: bool = False) -> list[str]:
    """Sort the colormap rows of *entries*, leaving solid rows where they are.

    Pure, so the rule is testable without a window. *key* is ``"name"`` (case-
    insensitive alphabetical) or ``"added"`` (the order the application declares
    them in, which for custom maps is the order they were created).

    Solid rows — the folded group token or an individual solid colour — hold
    their positions: they are a category, and sorting them in among the maps
    would undo the grouping that makes the list readable.
    """
    entries = [str(name) for name in entries]
    movable = [name for name in entries
               if name != SOLID_GROUP_TOKEN and not is_solid_color(name)]
    if key == "name":
        movable.sort(key=str.casefold)
    elif key == "added":
        added = _added_order()
        movable.sort(key=lambda name: added.get(name.casefold(), len(added)))
    else:
        raise ValueError(f"Unknown colormap sort key: {key!r}")
    if descending:
        movable.reverse()
    stream = iter(movable)
    return [name if (name == SOLID_GROUP_TOKEN or is_solid_color(name))
            else next(stream) for name in entries]


def _added_order() -> dict[str, int]:
    """Name -> the position it is declared/created in (built-ins, then customs)."""
    from ..colormaps import BUILTIN_COLORMAP_NAMES

    names = [*BUILTIN_COLORMAP_NAMES, *custom_colormap_names()]
    return {name.casefold(): index for index, name in enumerate(names)}


def _swatch(name: str) -> QIcon:
    """A gradient strip for a colormap, or a flat block for a solid colour."""
    width, height = _SWATCH
    pixmap = QPixmap(width, height)
    pixmap.fill(QColor(0, 0, 0, 0))
    painter = QPainter(pixmap)
    try:
        if name == SOLID_GROUP_LABEL:
            gradient = QLinearGradient(0, 0, width, 0)
            for index, solid in enumerate(solid_color_names() or ["Gray"]):
                try:
                    lut = colormap_lut(solid, n=2, alpha=False)
                except Exception:                               # noqa: BLE001
                    continue
                red, green, blue = (int(channel) for channel in lut[-1][:3])
                count = max(1, len(solid_color_names()) - 1)
                gradient.setColorAt(min(1.0, index / count),
                                    QColor(red, green, blue))
            painter.fillRect(0, 0, width, height, gradient)
        else:
            lut = colormap_lut(name, n=width, alpha=False)
            for x in range(width):
                red, green, blue = (int(channel) for channel in lut[x][:3])
                painter.fillRect(x, 0, 1, height, QColor(red, green, blue))
        painter.setPen(QColor(120, 120, 120))
        painter.drawRect(0, 0, width - 1, height - 1)
    except Exception:                                           # noqa: BLE001
        pass
    finally:
        painter.end()
    return QIcon(pixmap)


class ColormapOrderDialog(QDialog):
    """Drag-to-reorder editor for the shared colormap list."""

    def __init__(self, parent=None, *, entries, fold_solids: bool = True) -> None:
        super().__init__(parent)
        self.setWindowTitle("Reorder colormap list")
        self.resize(360, 520)
        self._sort_descending: dict[str, bool] = {"name": False, "added": False}

        root = QVBoxLayout(self)
        note = QLabel(
            "Drag a row to move it. The order applies to every colormap "
            "selector in the application."
        )
        note.setWordWrap(True)
        root.addWidget(note)

        self._list = QListWidget()
        self._list.setDragDropMode(QAbstractItemView.DragDropMode.InternalMove)
        self._list.setSelectionMode(QAbstractItemView.SelectionMode.SingleSelection)
        self._list.setDefaultDropAction(Qt.DropAction.MoveAction)
        root.addWidget(self._list, 1)

        buttons_row = QHBoxLayout()
        buttons_row.addWidget(QLabel("Order by"))
        self._by_name = QPushButton("alphabetic")
        self._by_name.setToolTip(
            "Sort the colormaps by name; click again to reverse. "
            "Solid colours keep their positions.")
        self._by_name.clicked.connect(lambda: self._sort("name"))
        buttons_row.addWidget(self._by_name)
        self._by_added = QPushButton("time added")
        self._by_added.setToolTip(
            "Sort the colormaps by when they were added — the built-in ones "
            "first, then custom maps in creation order; click again to reverse.")
        self._by_added.clicked.connect(lambda: self._sort("added"))
        buttons_row.addWidget(self._by_added)
        self._fold = QCheckBox("fold solid colors")
        self._fold.setChecked(bool(fold_solids))
        self._fold.setToolTip(
            "Collapse the solid colours into one 'Solid color' group. The "
            "group can be dragged as a whole; what is inside it follows the "
            "COLOR dialog's Solid Color List. Unfold to place an individual "
            "solid — a frequently used 'Gray', say — anywhere in the list.")
        self._fold.toggled.connect(self._on_fold_toggled)
        buttons_row.addWidget(self._fold)
        buttons_row.addStretch()
        root.addLayout(buttons_row)

        box = QDialogButtonBox(
            QDialogButtonBox.StandardButton.Ok
            | QDialogButtonBox.StandardButton.Cancel
            | QDialogButtonBox.StandardButton.Reset
        )
        box.accepted.connect(self.accept)
        box.rejected.connect(self.reject)
        box.button(QDialogButtonBox.StandardButton.Reset).setToolTip(
            "Back to the order this application ships with.")
        box.button(QDialogButtonBox.StandardButton.Reset).clicked.connect(self._reset)
        root.addWidget(box)

        self._set_rows(self._collapse(list(entries), self._fold.isChecked()))

    # ------------------------------------------------------------------ rows
    @staticmethod
    def _collapse(entries: list[str], fold: bool) -> list[str]:
        """Replace the individual solids with the group token, or expand it."""
        solids = list(solid_color_names())
        if fold:
            out: list[str] = []
            placed = False
            for name in entries:
                if name == SOLID_GROUP_TOKEN or is_solid_color(name):
                    if not placed:
                        out.append(SOLID_GROUP_TOKEN)
                        placed = True
                    continue
                out.append(name)
            if not placed:
                out.append(SOLID_GROUP_TOKEN)
            return out
        out = []
        for name in entries:
            if name == SOLID_GROUP_TOKEN:
                out.extend(solids)
                continue
            out.append(name)
        for solid in solids:                       # any not yet placed
            if solid not in out:
                out.append(solid)
        return out

    def _set_rows(self, entries: list[str]) -> None:
        self._list.clear()
        for name in entries:
            label = SOLID_GROUP_LABEL if name == SOLID_GROUP_TOKEN else name
            item = QListWidgetItem(_swatch(label), label)
            item.setData(Qt.ItemDataRole.UserRole, name)
            if name == SOLID_GROUP_TOKEN:
                item.setToolTip(
                    f"{len(solid_color_names())} solid colours, in the COLOR "
                    "dialog's order. Drag to move them together; untick "
                    "'fold solid colors' to place one individually.")
            self._list.addItem(item)

    def entries(self) -> list[str]:
        """The rows as saved: the group token when folded, else every solid."""
        return [self._list.item(row).data(Qt.ItemDataRole.UserRole)
                for row in range(self._list.count())]

    def fold_solids(self) -> bool:
        return bool(self._fold.isChecked())

    # --------------------------------------------------------------- actions
    def _sort(self, key: str) -> None:
        """Sort by *key*; a second click on the same button reverses it.

        The FIRST click ascends — the toggle is advanced after sorting, not
        before, or the button's first use would surprise by descending.
        """
        descending = self._sort_descending[key]
        self._set_rows(sort_entries(self.entries(), key, descending=descending))
        self._sort_descending[key] = not descending
        other = "added" if key == "name" else "name"
        self._sort_descending[other] = False       # the other button restarts
        button = self._by_name if key == "name" else self._by_added
        base = "alphabetic" if key == "name" else "time added"
        button.setText(f"{base} {'↓' if descending else '↑'}")
        (self._by_added if key == "name" else self._by_name).setText(
            "time added" if key == "name" else "alphabetic")

    def _on_fold_toggled(self, folded: bool) -> None:
        self._set_rows(self._collapse(self.entries(), bool(folded)))

    def _reset(self) -> None:
        from ..colormaps import BUILTIN_COLORMAP_NAMES

        self._fold.setChecked(True)
        self._set_rows(self._collapse(
            [*BUILTIN_COLORMAP_NAMES, *custom_colormap_names()], True))
