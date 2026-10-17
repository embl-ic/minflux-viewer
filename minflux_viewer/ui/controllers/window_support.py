"""Shared window helpers with one implementation owner."""

from __future__ import annotations

from PyQt6.QtCore import QObject, QRunnable, pyqtSignal
from PyQt6.QtGui import QColor, QIcon, QImage, QPalette, QPixmap
from PyQt6.QtWidgets import QApplication

from minflux_viewer.core import formats as _formats
from minflux_viewer.core.app_state import AppState
from minflux_viewer.ui.common.background_tasks import BackgroundTask

_SUPPORTED_EXTS: tuple[str, ...] = _formats.supported_extensions()


_FMT_LOADERS: dict[str, str] = {
    "mat": "_load_mat", "npy": "_load_npy", "npz": "_load_npz",
    "spreadsheet": "_load_spreadsheet",
    "msr": "_open_msr_dialog", "tiff": "_load_tiff", "json": "_load_json",
    "zarr": "_load_zarr",
}


_ZarrIoTask = BackgroundTask


_ROI_TOOL_DEFS: tuple[tuple[str, str, str], ...] = (
    ("Rectangle", "rectangle", "toolRect"),
    ("Oval", "oval", "toolOval"),
    ("Polygon", "polygon", "toolPolygon"),
    ("Freehand", "freehand", "toolFreehand"),
    ("Line", "line", "toolLine"),
    ("Point", "point", "toolPoint"),
)


_LINE_FAMILY: tuple[tuple[str, str, str], ...] = (
    ("Straight Line", "line", "line.png"),
    ("Polyline", "polyline", "polyline.png"),
    ("Freehand Line", "freehand_line", "freeline.png"),
)


_RECT_FAMILY: tuple[tuple[str, str, str, float], ...] = (
    ("Rectangle", "rectangle", "rec.png", 0.0),
    ("Rotated Rectangle", "rotated_rectangle", "rec.png", 45.0),
    ("Cuboid (3D)", "cuboid", "cuboid.png", 0.0),
)


_OVAL_FAMILY: tuple[tuple[str, str, str, float], ...] = (
    ("Oval", "oval", "ellipse.png", 0.0),
    ("Ellipse", "ellipse", "ellipse.png", 45.0),
    ("Sphere (3D)", "sphere", "sphere.png", 0.0),
    # A cylinder is an oval cross-section extruded along the axis normal to the
    # drawing plane, so it belongs to the oval family: drawn as an oval in one
    # plane, a rectangle in the other two.
    ("Cylinder (3D)", "cylinder", "cylinder.png", 0.0),
)


_POLY_FAMILY: tuple[tuple[str, str, str, float], ...] = (
    ("Polygon", "polygon", "polygon.png", 0.0),
    ("Polyhedron (3D)", "polyhedron", "polyhedron.png", 0.0),
)


_POINT_FAMILY: tuple[tuple[str, str, str, float], ...] = (
    ("Point", "point", "point.png", 0.0),
    ("Multi-point", "multi_point", "multipoint.png", 0.0),
)


_LASSO_FAMILY: tuple[tuple[str, str, str, float], ...] = (
    ("Magnetic Lasso", "magnetic_lasso", "lasso.png", 0.0),
    ("Magic Wand", "magic_wand", "wand.png", 0.0),
)


_ROI_TOOL_EXTRA_LABELS = {"angle": "Angle"}


_ROI_TOOL_VERBS = {"point": "placing", "multi_point": "placing"}


def roi_tool_labels() -> dict:
    """Tool key -> the label the menus show for it.

    Gathered from the same toolbar tables the menus are built from, so the status
    line cannot announce a tool under a name the user never sees.
    """
    labels: dict = {}
    for label, tool, _attr in _ROI_TOOL_DEFS:
        labels[tool] = label
    for table in (_LINE_FAMILY, _RECT_FAMILY, _OVAL_FAMILY, _POLY_FAMILY,
                  _POINT_FAMILY, _LASSO_FAMILY):
        for entry in table:
            labels[entry[1]] = entry[0]
    labels.update(_ROI_TOOL_EXTRA_LABELS)
    return labels


def roi_tool_status(tool: str, *, has_view: bool = True) -> str:
    """The status line for an armed ROI tool, or ``""`` when none is armed.

    *has_view* is False when no coordinate view is open to draw in; the line then
    says what to do about it, because the status bar is where the user is looking
    and a bare "drawing Rectangle ROI..." would be a promise nothing can keep.
    """
    key = str(tool or "")
    if not key:
        return ""
    label = roi_tool_labels().get(key)
    if label is None:
        return ""
    if not has_view:
        return f"{label} ROI: open a render, scatter, histogram or attribute plot to draw in…"
    if key == "magic_wand":
        return f"{label}: click a localization to grow a ROI…"
    return f"{_ROI_TOOL_VERBS.get(key, 'drawing')} {label} ROI…"


def _adaptive_toolbar_pixmap(path: str) -> QPixmap | None:
    """Load a toolbar icon without its baked-in white matte.

    The small ROI/LUT PNGs are opaque black-on-white images. Treating the white
    pixels as an alpha mask removes the visible white square on dark palettes.
    Monochrome artwork is recolored with the current button-text color so the
    same source remains legible in both light and dark modes; the colored
    LUT-picker icon keeps its original colors.
    """
    image = QImage(path)
    if image.isNull():
        return None
    image = image.convertToFormat(QImage.Format.Format_ARGB32)

    monochrome = True
    for y in range(image.height()):
        for x in range(image.width()):
            color = image.pixelColor(x, y)
            if max(color.red(), color.green(), color.blue()) - min(
                color.red(), color.green(), color.blue()
            ) > 3:
                monochrome = False
                break
        if not monochrome:
            break

    palette = QApplication.palette()
    foreground = palette.color(
        QPalette.ColorGroup.Active, QPalette.ColorRole.ButtonText
    )
    if not foreground.isValid():
        foreground = palette.color(
            QPalette.ColorGroup.Active, QPalette.ColorRole.WindowText
        )
    # Keep dark-mode strokes slightly softer than a fully opaque pure white.
    foreground_alpha = 230

    for y in range(image.height()):
        for x in range(image.width()):
            color = image.pixelColor(x, y)
            red, green, blue = color.red(), color.green(), color.blue()
            if monochrome:
                luminance = round(0.2126 * red + 0.7152 * green + 0.0722 * blue)
                coverage = max(0, min(255, 255 - luminance))
                alpha = round(color.alpha() * coverage * foreground_alpha / (255 * 255))
                image.setPixelColor(
                    x,
                    y,
                    QColor(foreground.red(), foreground.green(), foreground.blue(), alpha),
                )
            else:
                # For colored artwork, only remove the white matte while
                # preserving the original wheel colors and antialiasing.
                coverage = max(0, min(255, 255 - min(red, green, blue)))
                alpha = round(color.alpha() * coverage / 255)
                image.setPixelColor(x, y, QColor(red, green, blue, alpha))

    return QPixmap.fromImage(image)


def _adaptive_toolbar_icon(path: str) -> QIcon:
    pixmap = _adaptive_toolbar_pixmap(path)
    return QIcon(pixmap) if pixmap is not None else QIcon(path)


def _raise_or_create(existing, cls, state: AppState):
    """Return *existing* (raised) if still alive, else create a new instance."""
    if existing is not None:
        try:
            if not existing.isHidden():
                existing.raise_()
                existing.activateWindow()
                return existing
        except RuntimeError:
            existing = None
    win = cls(state)
    win.show()
    return win


class _UpdateCheckSignals(QObject):
    done = pyqtSignal(object)   # core.updater.UpdateCheckResult


class _UpdateCheckTask(QRunnable):
    """Run the GitHub update check off the UI thread (Tier-A in-app updater)."""

    def __init__(self, current_version: str) -> None:
        super().__init__()
        self.signals = _UpdateCheckSignals()
        self._current = current_version
        self._cancelled = False

    def cancel(self) -> None:
        self._cancelled = True

    def run(self) -> None:  # noqa: N802 - Qt API
        from minflux_viewer.core.task_registry import track
        from minflux_viewer.core.updater import UpdateCheckResult, check_for_update
        with track("Check for updates", "update", cancel=self.cancel):
            try:
                result = check_for_update(self._current)
            except Exception as exc:  # never let a worker exception escape
                result = UpdateCheckResult("error", self._current, None, str(exc))
        if not self._cancelled:
            self.signals.done.emit(result)
