"""The localization-precision result dialogs actually draw their histograms.

Regression for a silent break: ``_precision_color`` moved to the global colour
registry and began returning a ``QColor``, while the shared plot helpers built
their translucent fill by **string concatenation** (``color + "66"``). ``QColor +
str`` raises ``TypeError``, which both dialogs caught with a broad ``except`` and
replaced with a "Histogram report could not be created" label — so the histograms
simply disappeared, with no traceback and nothing in the log.

These tests assert the drawn result rather than the colour plumbing, so they stay
honest whatever form the registry returns next.
"""

from __future__ import annotations

import numpy as np
import pytest

pytest.importorskip("PyQt6")
pytest.importorskip("pyqtgraph")

from minflux_viewer.analysis.localization_precision import (  # noqa: E402
    CRLBResultDialog,
    StdDevResultDialog,
    crlb_precision,
    stddev_per_trace,
)
from minflux_viewer.analysis.trace_analysis import (  # noqa: E402
    _fill_brush,
    _plot_color,
    _value_hist_plot,
)

_FALLBACK = "could not be created"


def _synthetic_traces(n_traces: int = 200, per_trace: int = 12, seed: int = 0):
    rng = np.random.default_rng(seed)
    tid = np.repeat(np.arange(n_traces), per_trace)
    centres = np.repeat(rng.normal(0, 500.0, (n_traces, 3)), per_trace, axis=0)
    loc = centres + rng.normal(0, 5.0, (n_traces * per_trace, 3))
    return loc, tid


def _plots(dialog):
    import pyqtgraph as pg

    return dialog.findChildren(pg.PlotWidget)


def _fallback_labels(dialog):
    from PyQt6.QtWidgets import QLabel

    return [w.text() for w in dialog.findChildren(QLabel) if _FALLBACK in (w.text() or "")]


def test_stddev_dialog_draws_its_two_histograms(qtbot):
    loc, tid = _synthetic_traces()
    dialog = StdDevResultDialog(stddev_per_trace(loc, tid))
    qtbot.addWidget(dialog)
    assert _fallback_labels(dialog) == []
    # σ_r (lateral) and σ_z (axial), one panel each.
    assert len(_plots(dialog)) >= 2


def test_crlb_dialog_draws_its_three_histograms(qtbot):
    rng = np.random.default_rng(1)
    n = 2400
    eco = rng.poisson(300, n).astype(float)
    efo = rng.normal(2e4, 2e3, n)
    fbg = np.full(n, 500.0)
    result = crlb_precision(eco, efo=efo, fbg=fbg, L_nm=75.0,
                            sigma_q_nm=180.0, L_z_nm=75.0)
    dialog = CRLBResultDialog(result, eco, efo, fbg, num_dim=3)
    qtbot.addWidget(dialog)
    assert _fallback_labels(dialog) == []
    # σ_xy, σ_z (3-D) and the photon count N.
    assert len(_plots(dialog)) >= 3


@pytest.mark.parametrize("colour_form", ["hex", "qcolor", "rgb", "rgba"])
def test_value_hist_plot_accepts_every_colour_form_the_app_produces(colour_form):
    """The registry hands out QColor/tuples and the literals hand out hex; the
    helper must not care, because the alpha is its business, not the caller's."""
    from PyQt6.QtGui import QColor
    import pyqtgraph as pg

    colour = {"hex": "#35d07f", "qcolor": QColor(53, 208, 127),
              "rgb": (53, 208, 127), "rgba": (53, 208, 127, 255)}[colour_form]
    values = np.random.default_rng(2).normal(10.0, 2.0, 500)
    widget = _value_hist_plot("t", values, colour, marker=10.0, marker_label="m")
    curves = [i for i in widget.plotItem.items if isinstance(i, pg.PlotDataItem)]
    assert curves, f"no curve drawn for a {colour_form} colour"
    assert len(curves[0].getData()[0]) == 40          # default n_bins


def test_plot_color_and_fill_brush_normalise_consistently():
    from PyQt6.QtGui import QColor

    for spec in ("#ff0000", QColor(255, 0, 0), (255, 0, 0)):
        assert _plot_color(spec).getRgb() == (255, 0, 0, 255)
    assert _plot_color((255, 0, 0, 128)).alpha() == 128
    # The fill alpha is applied on top, and does not leak back to the caller's
    # colour object.
    source = QColor(53, 208, 127)
    assert _fill_brush(source, 0x66).color().alpha() == 0x66
    assert source.alpha() == 255


def test_an_empty_distribution_returns_a_plot_rather_than_raising():
    """A dialog must still lay out when a distribution is all-NaN."""
    widget = _value_hist_plot("t", np.full(10, np.nan), "#35d07f")
    assert widget is not None
