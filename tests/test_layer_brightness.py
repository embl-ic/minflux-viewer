"""Per-layer brightness/contrast controller (ui/layer_brightness.py).

Exercises the level/auto/reset/target logic without opening the Qt dialog (the
BrightnessContrastDialog is imported lazily only in ``open()``).
"""

import numpy as np

from minflux_viewer.ui.layer_brightness import LayerBrightness, percentile_levels


def _make_controller():
    applied = {"resp": [], "data": []}
    lb = LayerBrightness()
    lb.add_layer("Response",
                 get_pixels=lambda: np.linspace(0, 1, 100),
                 apply=lambda lo, hi: applied["resp"].append((lo, hi)),
                 auto=lambda: (0.1, 0.9),
                 default=lambda: (0.0, 1.0))
    lb.add_layer("Data",
                 get_pixels=lambda: np.arange(50.0),
                 apply=lambda lo, hi: applied["data"].append((lo, hi)),
                 auto=lambda: percentile_levels(np.arange(50.0), 0, 99),
                 default=lambda: (0.0, 49.0))
    return lb, applied


def test_percentile_levels_basic():
    lo, hi = percentile_levels(np.arange(101.0), 0, 100)
    assert lo == 0.0 and hi == 100.0
    assert percentile_levels(np.array([np.nan, np.inf])) is None


def test_default_until_set():
    lb, _ = _make_controller()
    assert lb.effective_levels("Response") == (0.0, 1.0)
    assert lb.effective_levels("Data") == (0.0, 49.0)
    assert lb.effective_levels("missing") is None


def test_first_layer_is_default_target():
    lb, _ = _make_controller()
    assert lb.target == "Response"
    assert lb.names() == ["Response", "Data"]


def test_on_levels_sets_and_applies_target():
    lb, applied = _make_controller()
    lb._on_levels(0.2, 0.7)
    assert lb.effective_levels("Response") == (0.2, 0.7)
    assert applied["resp"] == [(0.2, 0.7)]
    assert applied["data"] == []


def test_target_switch_routes_levels():
    lb, applied = _make_controller()
    lb.set_target("Data")
    lb._on_levels(5.0, 40.0)
    assert lb.effective_levels("Data") == (5.0, 40.0)
    assert applied["data"] == [(5.0, 40.0)]
    assert applied["resp"] == []
    # the Response layer still reports its default
    assert lb.effective_levels("Response") == (0.0, 1.0)


def test_auto_uses_auto_fn():
    lb, applied = _make_controller()
    lb._on_auto()
    assert lb.effective_levels("Response") == (0.1, 0.9)
    assert applied["resp"][-1] == (0.1, 0.9)


def test_reset_clears_to_default():
    lb, applied = _make_controller()
    lb._on_levels(0.3, 0.6)
    lb._on_reset()
    assert lb.effective_levels("Response") == (0.0, 1.0)   # back to default
    assert applied["resp"][-1] == (0.0, 1.0)
