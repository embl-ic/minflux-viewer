"""Editor for persistent application-owned custom colormaps."""

from __future__ import annotations

import numpy as np
import pyqtgraph as pg
from PyQt6.QtCore import QEvent, Qt
from PyQt6.QtWidgets import (
    QDialog,
    QDialogButtonBox,
    QFormLayout,
    QLabel,
    QLineEdit,
    QMessageBox,
    QVBoxLayout,
    QWidget,
)

from ..colormaps import validate_custom_colormap_name

_MAX_CUSTOM_STOPS = 64
_IMPORTED_MAP_MIN_STOPS = 3
_IMPORTED_MAP_MAX_STOPS = 5
_IMPORTED_MAP_MAX_RGB_ERROR = 40.0


def _editable_colormap_control_points(
    cmap: pg.ColorMap,
) -> tuple[np.ndarray, np.ndarray]:
    """Approximate a dense lookup map with three to five editable stops."""
    positions = np.linspace(0.0, 1.0, 257)
    colors = np.asarray(cmap.map(positions, mode=pg.ColorMap.BYTE), dtype=float)
    selected = [0, len(positions) - 1]

    while True:
        selected.sort()
        reconstructed = np.column_stack(
            [
                np.interp(
                    positions,
                    positions[selected],
                    colors[selected, channel],
                )
                for channel in range(3)
            ]
        )
        errors = np.linalg.norm(colors[:, :3] - reconstructed, axis=1)
        errors[selected] = -1.0
        next_index = int(np.argmax(errors))
        enough_stops = len(selected) >= _IMPORTED_MAP_MIN_STOPS
        accurate_enough = errors[next_index] <= _IMPORTED_MAP_MAX_RGB_ERROR
        at_limit = len(selected) >= _IMPORTED_MAP_MAX_STOPS
        if enough_stops and (accurate_enough or at_limit):
            break
        selected.append(next_index)

    selected.sort()
    indices = np.asarray(selected, dtype=int)
    return positions[indices], np.rint(colors[indices]).astype(np.uint8)


class CustomColormapDialog(QDialog):
    """Create or edit a named gradient using PyQtGraph's gradient editor."""

    def __init__(
        self,
        parent: QWidget | None = None,
        *,
        name: str = "",
        stops: list[list[object]] | None = None,
    ) -> None:
        super().__init__(parent)
        self._replacing = str(name) if name else None
        self._result_name = ""
        self._result_stops: list[list[object]] = []
        self.setWindowTitle(
            "Edit custom colormap" if self._replacing else "Create custom colormap"
        )
        self.resize(520, 210)

        root = QVBoxLayout(self)
        form = QFormLayout()
        self._name_edit = QLineEdit(str(name))
        self._name_edit.setPlaceholderText("e.g. My density map")
        form.addRow("Name", self._name_edit)
        root.addLayout(form)

        help_label = QLabel(
            "Right-click on the gradient for preset maps. Click a stop to set "
            "color, drag to change position, right-click to remove. Double-click "
            "in empty stop area to add a new color stop."
        )
        help_label.setWordWrap(True)
        root.addWidget(help_label)

        self._gradient = pg.GradientWidget(orientation="bottom")
        self._gradient.setMaxDim(70)
        if stops:
            ticks = [
                (float(position), tuple(int(channel) for channel in rgba))
                for position, rgba in stops
            ]
            self._gradient.restoreState(
                {"mode": "rgb", "ticks": ticks, "ticksVisible": True}
            )
        else:
            self._gradient.restoreState(
                {
                    "mode": "rgb",
                    "ticks": [
                        (0.0, (0, 0, 0, 255)),
                        (0.5, (40, 120, 220, 255)),
                        (1.0, (255, 255, 255, 255)),
                    ],
                    "ticksVisible": True,
                }
            )
        root.addWidget(self._gradient)

        # PyQtGraph's non-legacy colormap entries (the ``local`` and
        # ``cet (local)`` menus) replace the ticks and then hide them, which
        # also disables adding new ticks.  In a colormap *editor* they must
        # remain visible and editable.  Replace only that menu callback; the
        # menu itself and its lazy-built collections remain PyQtGraph-owned.
        gradient_menu = self._gradient.item.menu
        gradient_menu.sigColorMapTriggered.disconnect(
            self._gradient.item.colorMapMenuClicked
        )
        gradient_menu.sigColorMapTriggered.connect(
            self._on_gradient_colormap_selected
        )

        # PyQtGraph normally adds a tick on a single click in the stop lane.
        # Disable that path and handle a viewport double-click explicitly so
        # creation is unambiguous and works after every preset selection.
        self._gradient.item.allowAdd = False
        self._gradient.viewport().installEventFilter(self)

        # Keep PyQtGraph's tick menu and behaviour, shortening only its label.
        self._raise_tick_context_menu_original = (
            self._gradient.item.raiseTickContextMenu
        )
        self._gradient.item.raiseTickContextMenu = self._raise_tick_context_menu

        buttons = QDialogButtonBox(
            QDialogButtonBox.StandardButton.Save
            | QDialogButtonBox.StandardButton.Cancel
        )
        buttons.accepted.connect(self._accept_if_valid)
        buttons.rejected.connect(self.reject)
        root.addWidget(buttons)

    def _on_gradient_colormap_selected(self, cmap: pg.ColorMap) -> None:
        """Load a PyQtGraph menu colormap as a manageable editable gradient."""
        name = str(getattr(cmap, "name", ""))
        if name.startswith("preset-gradient:"):
            self._gradient.item.loadPreset(name.split(":", 1)[1])
        else:
            # Local/CET maps normally contain 256 lookup samples. Select the
            # most informative one to three interior points so the result is as
            # practical to edit as the legacy preset gradients.
            positions, colors = _editable_colormap_control_points(cmap)
            self._gradient.item.setColorMap(pg.ColorMap(positions, colors))
        self._ensure_endpoint_stops()
        self._gradient.item.showTicks(True)

    def eventFilter(self, watched, event) -> bool:
        if (
            watched is self._gradient.viewport()
            and event.type() == QEvent.Type.MouseButtonDblClick
            and event.button() == Qt.MouseButton.LeftButton
            and self._add_color_stop_at(event.position())
        ):
            event.accept()
            return True
        return super().eventFilter(watched, event)

    def _add_color_stop_at(self, viewport_position) -> bool:
        """Add one stop when *viewport_position* is in empty stop-lane space."""
        item = self._gradient.item
        scene_position = self._gradient.mapToScene(viewport_position.toPoint())
        item_position = item.mapFromScene(scene_position)
        if not (
            0.0 <= float(item_position.x()) <= item.length
            and 0.0 <= float(item_position.y()) <= item.tickSize
        ):
            return False
        if any(
            tick.contains(tick.mapFromScene(scene_position)) for tick in item.ticks
        ):
            return False
        if len(item.ticks) >= _MAX_CUSTOM_STOPS:
            QMessageBox.warning(
                self,
                self.windowTitle(),
                f"A custom colormap can contain at most {_MAX_CUSTOM_STOPS} color stops.",
            )
            return True
        position = min(1.0, max(0.0, float(item_position.x()) / item.length))
        item.addTick(position)
        item.showTicks(True)
        return True

    def _raise_tick_context_menu(self, tick, event) -> None:
        self._raise_tick_context_menu_original(tick, event)
        self._gradient.item.tickMenu.removeAct.setText("Remove")

    def _ensure_endpoint_stops(self) -> None:
        """Keep explicit minimum and maximum stops in the editable state."""
        ticks = self._gradient.item.listTicks()
        if not ticks:
            self._gradient.item.addTick(0.0, (0, 0, 0, 255))
            self._gradient.item.addTick(1.0, (255, 255, 255, 255))
            return
        if float(ticks[0][1]) > 0.0:
            self._gradient.item.addTick(0.0, self._gradient.item.getColor(0.0))
        if float(ticks[-1][1]) < 1.0:
            self._gradient.item.addTick(1.0, self._gradient.item.getColor(1.0))

    def _accept_if_valid(self) -> None:
        try:
            name = validate_custom_colormap_name(
                self._name_edit.text(), replacing=self._replacing
            )
            state = self._gradient.saveState()
            ticks = sorted(state.get("ticks", []), key=lambda item: float(item[0]))
            if not 2 <= len(ticks) <= _MAX_CUSTOM_STOPS:
                raise ValueError(
                    f"Use between 2 and {_MAX_CUSTOM_STOPS} color stops."
                )
            stops = [
                [float(position), [int(channel) for channel in rgba]]
                for position, rgba in ticks
            ]
        except (TypeError, ValueError) as exc:
            QMessageBox.warning(self, self.windowTitle(), str(exc))
            return
        self._result_name = name
        self._result_stops = stops
        self.accept()

    def result_name(self) -> str:
        return self._result_name

    def result_stops(self) -> list[list[object]]:
        return [
            [float(position), list(rgba)]
            for position, rgba in self._result_stops
        ]

    @property
    def replacing_name(self) -> str | None:
        return self._replacing
