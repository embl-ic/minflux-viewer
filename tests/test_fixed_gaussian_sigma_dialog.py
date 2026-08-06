"""Dataset-derived fixed-Gaussian sigma limits and slider dialog behavior."""

import numpy as np
from PyQt6.QtCore import QPoint, QPointF, Qt
from PyQt6.QtGui import QWheelEvent
from PyQt6.QtWidgets import QApplication, QDialogButtonBox

from minflux_viewer.ui.precision_render_window import PrecisionRenderWindow
from minflux_viewer.ui.render_window import (
    SigmaDialog,
    fixed_gaussian_sigma_limits_nm,
)


def _app() -> QApplication:
    return QApplication.instance() or QApplication([])


def test_sigma_limits_follow_lateral_and_axial_dataset_spans():
    locs = np.asarray([
        [0.0, -20.0, 10.0],
        [100.05, 60.05, 40.05],
    ])

    xy_max, z_max = fixed_gaussian_sigma_limits_nm(locs)

    assert xy_max == 20.0
    assert z_max == 15.0


def test_sigma_limits_use_single_minimum_step_for_flat_z():
    locs = np.asarray([
        [0.0, 0.0, np.nan],
        [100.0, 80.0, np.nan],
    ])

    assert fixed_gaussian_sigma_limits_nm(locs) == (20.0, 0.1)


def test_sigma_dialog_modeless_controls_sync_and_apply_without_closing():
    app = _app()
    applied = []
    accepted = []
    dialog = SigmaDialog(
        (5.0, 5.0),
        maxima_xy_z=(250.0, 150.0),
        on_apply=lambda xy, z: applied.append((xy, z)),
    )
    dialog.accepted.connect(lambda: accepted.append(True))
    dialog.show()
    app.processEvents()

    assert not dialog.isModal()
    assert dialog.windowModality() == Qt.WindowModality.NonModal
    assert dialog._sliders[0].minimum() == 0
    assert dialog._sliders[0].maximum() == 1000
    assert dialog._sliders[1].maximum() == 1000
    assert dialog._sliders[0].tickPosition() != dialog._sliders[0].TickPosition.NoTicks
    assert dialog._spins[0].maximum() == 250.0
    assert dialog._spins[1].maximum() == 150.0
    assert dialog._spins[0].singleStep() == 0.1

    dialog._sliders[0].setValue(0)
    assert dialog._sliders[0].value() == 1
    assert dialog._spins[0].value() == 0.1

    dialog._sliders[0].setValue(123)
    dialog._sliders[1].setValue(77)
    assert dialog.values_xy_z() == (12.3, 7.7)

    wheel = QWheelEvent(
        QPointF(),
        QPointF(),
        QPoint(),
        QPoint(0, 120),
        Qt.MouseButton.NoButton,
        Qt.KeyboardModifier.NoModifier,
        Qt.ScrollPhase.ScrollUpdate,
        False,
    )
    dialog._sliders[0].wheelEvent(wheel)
    assert dialog.values_xy_z() == (13.3, 7.7)

    dialog._spins[0].setValue(120.0)
    assert dialog._sliders[0].value() == 1000
    assert dialog.values_xy_z() == (120.0, 7.7)

    dialog._apply_button.click()
    app.processEvents()
    assert applied == [(120.0, 7.7)]
    assert dialog.isVisible()

    dialog._spins[0].setValue(12.5)
    dialog._button_box.button(QDialogButtonBox.StandardButton.Ok).click()
    app.processEvents()
    assert applied[-1] == (12.5, 7.7)
    assert accepted == [True]


def test_precision_window_apply_updates_all_sigma_state_and_invalidates():
    class State:
        _fixed_sigma_xy_nm = 5.0
        _fixed_sigma_z_nm = 5.0
        _fixed_sigma_nm = 5.0
        _sigma_nm_xyz = (5.0, 5.0, 5.0)
        _volume_window = None
        invalidations = 0

        def _invalidate_advanced_render(self):
            self.invalidations += 1

    state = State()

    PrecisionRenderWindow._apply_fixed_sigma_values(state, 2.3, 4.7)

    assert state._fixed_sigma_xy_nm == 2.3
    assert state._fixed_sigma_z_nm == 4.7
    assert state._fixed_sigma_nm == 2.3
    assert state._sigma_nm_xyz == (2.3, 2.3, 4.7)
    assert state.invalidations == 1
