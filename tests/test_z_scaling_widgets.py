"""The Z scaling factor is edited and shown at four decimals.

Setting 0.6667 and reading it back as 0.67 in Dataset Information is a silent
rounding of a value the user typed exactly; and one step size cannot serve both
a deliberate arrow click and a coarse wheel sweep.
"""

from __future__ import annotations

import pytest

from minflux_viewer.ui.z_scaling_widgets import (
    Z_SCALING_ARROW_STEP,
    Z_SCALING_DECIMALS,
    Z_SCALING_WHEEL_STEP,
    ZScalingFactorSpinBox,
    format_z_scaling_factor,
)


def test_display_keeps_four_decimals_but_never_shows_fewer_than_two():
    assert format_z_scaling_factor(0.6667) == "0.6667"      # was rounded to 0.67
    assert format_z_scaling_factor(0.6375) == "0.6375"      # the Gaussian-fit estimate
    assert format_z_scaling_factor(0.67) == "0.67"
    assert format_z_scaling_factor(1.0) == "1.00"
    assert format_z_scaling_factor(0.5) == "0.50"
    assert format_z_scaling_factor(2) == "2.00"
    # Beyond four decimals is rounded, deliberately: that is the editor's range.
    assert format_z_scaling_factor(0.66666666) == "0.6667"


def test_dataset_information_dims_row_shows_the_value_that_was_set():
    np = pytest.importorskip("numpy")
    from minflux_viewer.core.dataset import build_localization_dataset
    from minflux_viewer.ui.data_window import _dims_text

    rng = np.random.default_rng(0)
    n = 200
    ds = build_localization_dataset(
        name="run", x_nm=rng.random(n) * 100, y_nm=rng.random(n) * 100,
        z_nm=rng.random(n) * 100)
    assert int(ds.prop.num_dim) == 3

    ds.set_z_scaling_factor(0.6667, source="manual (Dataset Information)")
    assert "Z scaling factor = 0.6667" in _dims_text(ds)

    # A calculated value is unchanged by this, and 1.0 still reads as "1.00".
    ds.set_z_scaling_factor(0.6375, source="estimated (trace anisotropy)")
    assert "Z scaling factor = 0.6375" in _dims_text(ds)
    ds.set_z_scaling_factor(1.0, source="manual (Dataset Information)")
    assert "Z scaling factor = 1.00" in _dims_text(ds)


def test_arrow_steps_finely_and_the_wheel_steps_coarsely(qtbot):
    pytest.importorskip("PyQt6")
    from PyQt6.QtCore import QPoint, QPointF, Qt
    from PyQt6.QtGui import QWheelEvent

    spin = ZScalingFactorSpinBox()
    qtbot.addWidget(spin)
    assert spin.decimals() == Z_SCALING_DECIMALS
    assert spin.singleStep() == pytest.approx(Z_SCALING_ARROW_STEP)

    spin.setValue(0.6667)
    spin.stepBy(1)                                   # what an arrow click does
    assert spin.value() == pytest.approx(0.6667 + Z_SCALING_ARROW_STEP)

    spin.setValue(0.6667)
    event = QWheelEvent(
        QPointF(5.0, 5.0), QPointF(5.0, 5.0), QPoint(0, 0), QPoint(0, 120),
        Qt.MouseButton.NoButton, Qt.KeyboardModifier.NoModifier,
        Qt.ScrollPhase.NoScrollPhase, False)
    spin.wheelEvent(event)
    assert spin.value() == pytest.approx(0.6667 + Z_SCALING_WHEEL_STEP)
    # The wheel step is borrowed, not kept: the next arrow click is fine again.
    assert spin.singleStep() == pytest.approx(Z_SCALING_ARROW_STEP)
