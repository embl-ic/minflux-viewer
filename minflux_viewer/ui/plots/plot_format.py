"""Shared plotting number formatting helpers."""

from __future__ import annotations

import math

import pyqtgraph as pg

_SUPERSCRIPT = str.maketrans("-0123456789", "⁻⁰¹²³⁴⁵⁶⁷⁸⁹")


def superscript_int(value: int) -> str:
    return str(int(value)).translate(_SUPERSCRIPT)


def format_plot_number(value: float, *, decimals: int = 1, exponent: int | None = None) -> str:
    """Human-friendly tick labels with one decimal and scientific fallback."""
    try:
        value = float(value)
    except Exception:
        return str(value)
    if not math.isfinite(value):
        return ""
    if value == 0:
        return "0"

    abs_value = abs(value)
    if exponent is None and 1e-3 <= abs_value < 10_000:
        if abs(value - round(value)) < 1e-9 and abs_value >= 10:
            return f"{int(round(value))}"
        return f"{value:.{decimals}f}"

    exponent = int(math.floor(math.log10(abs_value))) if exponent is None else int(exponent)
    mantissa = value / (10 ** exponent)
    return f"{mantissa:.{decimals}f}×10{superscript_int(exponent)}"


class ScientificAxisItem(pg.AxisItem):
    """PyQtGraph axis that avoids SI-prefix labels and uses compact sci ticks."""

    def __init__(self, orientation: str, *, decimals: int = 1, **kwargs) -> None:
        super().__init__(orientation, **kwargs)
        self._decimals = int(decimals)
        self.enableAutoSIPrefix(False)

    def tickStrings(self, values, scale, spacing):  # noqa: N802 - Qt/pyqtgraph API
        scaled = [float(value) * scale for value in values]
        finite = [abs(value) for value in scaled if math.isfinite(value) and value != 0]
        display_decimals = self._decimals
        exponent = None
        if finite:
            max_abs = max(finite)
            min_abs = min(finite)
            if (max_abs >= 10_000 or max_abs < 1e-3) and (min_abs / max_abs) >= 0.01:
                exponent = int(math.floor(math.log10(max_abs)))
        display_spacing = abs(float(spacing) * scale)
        if display_spacing > 0 and math.isfinite(display_spacing):
            effective_spacing = display_spacing / (10 ** exponent) if exponent is not None else display_spacing
            if effective_spacing > 0:
                # Keep at least one significant digit below the grid spacing,
                # so zoomed-in tick labels do not collapse to identical values.
                display_decimals = max(
                    self._decimals,
                    int(math.ceil(-math.log10(effective_spacing))) + 1,
                )
                display_decimals = min(display_decimals, 8)
        return [
            format_plot_number(value, decimals=display_decimals, exponent=exponent)
            for value in scaled
        ]


def plot_widget(*, background="w", **kwargs) -> pg.PlotWidget:
    """Create a PlotWidget with consistent numeric axes."""
    axis_items = kwargs.pop("axisItems", None) or {
        "bottom": ScientificAxisItem("bottom"),
        "left": ScientificAxisItem("left"),
    }
    return pg.PlotWidget(background=background, axisItems=axis_items, **kwargs)


def xy_origin_top_left(prefs: dict | None, preference_key: str) -> bool:
    """Return the configured XY screen convention, defaulting to image-style Y.

    Render and Scatter intentionally keep separate preferences, so secondary
    coordinate views name which family they follow instead of inventing a
    third origin setting. Unknown/stale values retain the established
    top-left default; only the explicit ``bottom_left`` value opts out.
    """
    plot_prefs = (prefs or {}).get("plot", {}) or {}
    return str(plot_prefs.get(preference_key, "top_left")).lower() != "bottom_left"


def apply_spatial_y_direction(
    plot,
    *,
    vertical_coordinate: str,
    prefs: dict | None = None,
    preference_key: str = "scatter_xy_origin",
) -> bool:
    """Apply the shared screen direction when a plot's ordinate is coordinate Y.

    The top-left convention means larger Y values appear lower on screen. A Z
    ordinate and non-coordinate values (counts, speed, intensity, …) keep the
    mathematical bottom-left direction. ``plot`` may be a PlotWidget or
    PlotItem. The returned flag is useful to regression tests.
    """
    inverted = (
        str(vertical_coordinate).strip().upper() == "Y"
        and xy_origin_top_left(prefs, preference_key)
    )
    if hasattr(plot, "getPlotItem"):
        view_box = plot.getPlotItem().getViewBox()
    elif hasattr(plot, "getViewBox"):
        view_box = plot.getViewBox()
    else:
        raise TypeError("spatial plot must provide getPlotItem() or getViewBox()")
    view_box.invertY(inverted)
    return bool(inverted)
