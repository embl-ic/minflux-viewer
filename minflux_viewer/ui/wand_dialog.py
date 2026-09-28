"""Magic Wand Tool options: the two tolerances the wand grows with.

Fiji opens a tool's options by double-clicking its toolbar icon, and this is the
same gesture. There is deliberately **no connectivity option**: 4- versus
8-connected is a question about neighbouring *pixels*, and this wand walks
Cartesian coordinates, where "neighbour" is just a distance.

Both controls are sliders because they are explored rather than typed -- the
useful value depends on the structure under the cursor, so the answer is found
by moving them and watching the selection change.
"""

from __future__ import annotations

from PyQt6.QtCore import Qt, QTimer, pyqtSignal
from PyQt6.QtWidgets import (
    QDialog,
    QGridLayout,
    QGroupBox,
    QHBoxLayout,
    QLabel,
    QPushButton,
    QSlider,
    QVBoxLayout,
)

from ..analysis.wand_select import (
    DEFAULT_DISTANCE_NM,
    DEFAULT_VALUE_TOLERANCE_PCT,
)

#: Slider bounds. Distance is in nm over the range a MINFLUX structure occupies;
#: the value band is a percentage of the attribute's own finite range, so one
#: slider serves every attribute whatever its units.
MIN_DISTANCE_NM = 1
MAX_DISTANCE_NM = 1000
MIN_VALUE_PCT = 0
MAX_VALUE_PCT = 100

#: A slider drag emits a change per step, and a wand re-run costs a KD-tree
#: query; coalescing means the selection follows the drag without recomputing
#: at every intermediate value.
_APPLY_DELAY_MS = 150


class MagicWandDialog(QDialog):
    """Two tolerances, applied live to the last wand click.

    Emits :attr:`parameters_changed` after the sliders settle. Modeless and
    parentless per the project convention, so it does not sit in front of the
    view whose selection it is changing.
    """

    parameters_changed = pyqtSignal()

    def __init__(self, prefs: dict, owner=None) -> None:
        super().__init__(None)
        self.setWindowTitle("Magic Wand Tool")
        self._prefs = prefs if isinstance(prefs, dict) else {}
        self._owner = owner

        saved = (self._prefs.get("wand") or {})
        distance = float(saved.get("distance_nm", DEFAULT_DISTANCE_NM))
        percent = float(saved.get("value_percent", DEFAULT_VALUE_TOLERANCE_PCT))

        self._apply_timer = QTimer(self)
        self._apply_timer.setSingleShot(True)
        self._apply_timer.setInterval(_APPLY_DELAY_MS)
        self._apply_timer.timeout.connect(self.parameters_changed)

        layout = QVBoxLayout(self)
        group = QGroupBox("Tolerances", self)
        grid = QGridLayout(group)

        self._distance = QSlider(Qt.Orientation.Horizontal, self)
        self._distance.setRange(MIN_DISTANCE_NM, MAX_DISTANCE_NM)
        self._distance.setValue(int(round(max(MIN_DISTANCE_NM,
                                              min(MAX_DISTANCE_NM, distance)))))
        self._distance.setToolTip(
            "How far apart two localizations may be and still count as connected.\n"
            "This is a Cartesian distance between coordinates, not a pixel step,\n"
            "and it also sets the resolution of the traced outline.")
        self._distance_read = QLabel(self)

        self._value = QSlider(Qt.Orientation.Horizontal, self)
        self._value.setRange(MIN_VALUE_PCT, MAX_VALUE_PCT)
        self._value.setValue(int(round(max(MIN_VALUE_PCT, min(MAX_VALUE_PCT, percent)))))
        self._value.setToolTip(
            "How far a localization's value may differ from the CLICKED point's\n"
            "value and still be included, as a percentage of the attribute's\n"
            "range. Measured against the clicked point, not against each\n"
            "neighbour, so the selection cannot drift away along a chain.")
        self._value_read = QLabel(self)

        grid.addWidget(QLabel("Spatial distance:", self), 0, 0)
        grid.addWidget(self._distance, 0, 1)
        grid.addWidget(self._distance_read, 0, 2)
        grid.addWidget(QLabel("Value tolerance:", self), 1, 0)
        grid.addWidget(self._value, 1, 1)
        grid.addWidget(self._value_read, 1, 2)
        grid.setColumnStretch(1, 1)
        layout.addWidget(group)

        self._note = QLabel(
            "Click a localization to grow a region from it. The scatter plot grows "
            "on its Color by value; the render view grows on local density.", self)
        self._note.setWordWrap(True)
        layout.addWidget(self._note)

        buttons = QHBoxLayout()
        buttons.addStretch(1)
        reset = QPushButton("Reset", self)
        reset.clicked.connect(self._reset)
        close = QPushButton("Close", self)
        close.clicked.connect(self.close)
        buttons.addWidget(reset)
        buttons.addWidget(close)
        layout.addLayout(buttons)

        self._distance.valueChanged.connect(self._on_changed)
        self._value.valueChanged.connect(self._on_changed)
        self._refresh_readouts()
        self.setMinimumWidth(430)

    # ------------------------------------------------------------------ values
    def parameters(self) -> dict:
        return {
            "distance_nm": float(self._distance.value()),
            "value_percent": float(self._value.value()),
        }

    def _refresh_readouts(self) -> None:
        self._distance_read.setText(f"{self._distance.value()} nm")
        self._value_read.setText(f"{self._value.value()} %")

    def _on_changed(self, _value: int = 0) -> None:
        self._refresh_readouts()
        # Write through immediately so a click made before the timer fires still
        # uses what the slider shows; only the re-run is coalesced.
        self._prefs.setdefault("wand", {}).update(self.parameters())
        self._apply_timer.start()

    def _reset(self) -> None:
        self._distance.setValue(int(DEFAULT_DISTANCE_NM))
        self._value.setValue(int(DEFAULT_VALUE_TOLERANCE_PCT))

    def closeEvent(self, event):       # noqa: N802 - Qt API
        self._apply_timer.stop()
        super().closeEvent(event)
