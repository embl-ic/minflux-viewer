"""Focused rendering checks for the Brightness/Contrast histogram preview."""

import numpy as np
from PyQt6.QtGui import QColor, QImage
from PyQt6.QtWidgets import QApplication

from minflux_viewer.ui.brightness_contrast_dialog import HistogramPreview


def _app() -> QApplication:
    return QApplication.instance() or QApplication([])


def test_histogram_draws_vertical_minimum_and_maximum_level_markers():
    app = _app()
    preview = HistogramPreview()
    preview.resize(202, 82)
    preview.set_state(
        np.zeros(128, dtype=float),
        data_min=0.0,
        data_max=100.0,
        lo=25.0,
        hi=75.0,
    )

    image = QImage(preview.size(), QImage.Format.Format_RGB32)
    image.fill(QColor("white"))
    preview.render(image)
    app.processEvents()

    rect = preview.rect().adjusted(1, 1, -2, -2)
    x_lo = rect.left() + round(0.25 * rect.width())
    x_hi = rect.left() + round(0.75 * rect.width())
    y_mid = rect.center().y()

    assert image.pixelColor(x_lo, y_mid) == QColor("black")
    assert image.pixelColor(x_hi, y_mid) == QColor("black")
    assert image.pixelColor(x_lo + 5, y_mid) == QColor("white")

