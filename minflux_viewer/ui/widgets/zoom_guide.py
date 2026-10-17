"""The rubber-band a constrained zoom drag draws, in one place.

Three plots offer the same drag-to-zoom tool — the Attribute Plot, the Attribute
Histogram, and the HlyB/D result window's distance histogram — and each had its
own copy of the guide geometry. They drifted: the histograms drew the guide at
the **cursor**, while the Attribute Plot pinned it to the middle of the view, so
a vertical zoom drawn near an edge appeared far from the mouse.

The shape per mode:

* ``horizontal`` — an **H** lying on its side: a bar spanning the dragged x
  range with a cap at each end;
* ``vertical`` — the same rotated: a bar spanning the dragged y range;
* ``unconstrained`` — a plain rectangle.

**The guide rides at the cursor on its free axis** (the horizontal bar at the
mouse's y, the vertical bar at its x). The free axis is not being zoomed, so its
position carries no information — which makes it free to sit where the user is
looking, so the bar can be lined up against the data it is about to zoom into.
Pinned to the middle of the view it could be most of a plot away from the mouse.

Pure geometry, so the rule is testable without a window.
"""

from __future__ import annotations

import numpy as np

__all__ = ["ZOOM_MODES", "GUIDE_CAP_FRACTION", "zoom_guide_points"]

#: The zoom modes, in the order the menus offer them.
ZOOM_MODES: tuple[str, ...] = ("horizontal", "vertical", "unconstrained")

#: Half-length of a guide's end cap, as a fraction of the view's *other* axis.
GUIDE_CAP_FRACTION: float = 0.08


def zoom_guide_points(
    mode: str,
    start: tuple[float, float],
    current: tuple[float, float],
    view_range: tuple[tuple[float, float], tuple[float, float]],
) -> tuple[list[float], list[float]]:
    """``(xs, ys)`` for the guide of *mode*, ready for ``PlotDataItem.setData``.

    *start* and *current* are the drag's endpoints in view coordinates, and
    *view_range* is ``((x0, x1), (y0, y1))`` — used only to size the end caps
    relative to the visible span. ``NaN`` separates the disjoint strokes, which
    is how pyqtgraph draws several segments from one item.
    """
    x0, y0 = float(start[0]), float(start[1])
    x1, y1 = float(current[0]), float(current[1])
    (vx0, vx1), (vy0, vy1) = view_range

    if mode == "horizontal":
        cap = (float(vy1) - float(vy0)) * GUIDE_CAP_FRACTION
        # The bar rides at the cursor's y, not at the middle of the view.
        return (
            [x0, x1, np.nan, x0, x0, np.nan, x1, x1],
            [y1, y1, np.nan, y1 - cap, y1 + cap, np.nan, y1 - cap, y1 + cap],
        )
    if mode == "vertical":
        cap = (float(vx1) - float(vx0)) * GUIDE_CAP_FRACTION
        # ...and the vertical bar at the cursor's x, for the same reason.
        return (
            [x1, x1, np.nan, x1 - cap, x1 + cap, np.nan, x1 - cap, x1 + cap],
            [y0, y1, np.nan, y0, y0, np.nan, y1, y1],
        )
    return ([x0, x1, x1, x0, x0], [y0, y0, y1, y1, y0])
