"""Editor for persistent application-owned custom colormaps."""

from __future__ import annotations

import pyqtgraph as pg
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
        self.resize(520, 190)

        root = QVBoxLayout(self)
        form = QFormLayout()
        self._name_edit = QLineEdit(str(name))
        self._name_edit.setPlaceholderText("e.g. My density map")
        form.addRow("Name", self._name_edit)
        root.addLayout(form)

        help_label = QLabel(
            "Right-click the gradient to add a color stop. Drag stops to move "
            "them; click a stop to change its color. Right-click a stop to remove it."
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

        buttons = QDialogButtonBox(
            QDialogButtonBox.StandardButton.Save
            | QDialogButtonBox.StandardButton.Cancel
        )
        buttons.accepted.connect(self._accept_if_valid)
        buttons.rejected.connect(self.reject)
        root.addWidget(buttons)

    def _accept_if_valid(self) -> None:
        try:
            name = validate_custom_colormap_name(
                self._name_edit.text(), replacing=self._replacing
            )
            state = self._gradient.saveState()
            ticks = sorted(state.get("ticks", []), key=lambda item: float(item[0]))
            if not 2 <= len(ticks) <= 64:
                raise ValueError("Use between 2 and 64 color stops.")
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
