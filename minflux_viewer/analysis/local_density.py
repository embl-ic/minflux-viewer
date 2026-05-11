"""
minflux_viewer.analysis.local_density
=====================================
Local density estimators for localization coordinates.

The primary implementation mirrors the MATLAB ``rangesearch`` approach:
build a KD-tree and count neighbours within a physical radius. Coordinates
are expected in nanometres.
"""

from __future__ import annotations

import numpy as np
import pyqtgraph as pg
from scipy.ndimage import gaussian_filter
from scipy.spatial import cKDTree
from PyQt6.QtCore import Qt
from PyQt6.QtWidgets import (
    QComboBox,
    QDialog,
    QDialogButtonBox,
    QDoubleSpinBox,
    QFormLayout,
    QLabel,
    QMessageBox,
    QSizePolicy,
    QVBoxLayout,
    QWidget,
)

from ..ui.lut_dialog import make_colormap


def local_density_kdtree(points_nm: np.ndarray, radius_nm: float) -> np.ndarray:
    """
    Count neighbours within ``radius_nm`` for each localization.

    The point itself is excluded, matching MATLAB ``length(x)-1`` after
    ``rangesearch([x,y,z], [x,y,z], range)``.
    """
    pts = _finite_points(points_nm, force_3d=False)
    if pts.size == 0:
        return np.empty(0, dtype=float)
    radius_nm = float(radius_nm)
    if radius_nm <= 0:
        raise ValueError("radius_nm must be positive")
    tree = cKDTree(pts)
    counts = tree.query_ball_point(pts, r=radius_nm, return_length=True)
    return np.asarray(counts, dtype=float) - 1.0


def local_density_histogram_2d(
    points_nm: np.ndarray,
    *,
    voxel_size_nm: float | None = None,
    smooth_sigma: float = 1.0,
) -> np.ndarray:
    """
    Approximate local density from a 2D binned count map.

    This follows the histogram branch of the MATLAB prototype, limited to XY
    for now. The value returned for each point is the count/smoothed-count of
    the XY bin containing that point, not a normalized density per area.
    """
    pts = _finite_points(points_nm, force_3d=False)
    if pts.size == 0:
        return np.empty(0, dtype=float)
    xy = pts[:, :2]
    if voxel_size_nm is None:
        span = np.ptp(xy, axis=0)
        voxel_size_nm = float(np.nanmax(span) / 100.0) if np.any(span > 0) else 1.0
    voxel_size_nm = max(float(voxel_size_nm), np.finfo(float).eps)

    x_edges = _edges_for(xy[:, 0], voxel_size_nm)
    y_edges = _edges_for(xy[:, 1], voxel_size_nm)
    counts, _, _ = np.histogram2d(xy[:, 0], xy[:, 1], bins=[x_edges, y_edges])
    values = gaussian_filter(counts, smooth_sigma) if smooth_sigma > 0 else counts

    x_bin = np.clip(np.searchsorted(x_edges, xy[:, 0], side="right") - 1, 0, values.shape[0] - 1)
    y_bin = np.clip(np.searchsorted(y_edges, xy[:, 1], side="right") - 1, 0, values.shape[1] - 1)
    return values[x_bin, y_bin].astype(float)


def compute_local_density_for_dataset(ds, prefs: dict) -> tuple[np.ndarray, str, str]:
    """
    Compute local density for the filtered localizations of ``ds``.

    Returns
    -------
    density_full, method_label, detail
        ``density_full`` has length ``ds.prop.num_loc`` and contains NaN for
        currently filtered-out localizations.
    """
    data_prefs = prefs.get("data", {}) if prefs else {}
    method = str(data_prefs.get("local_density_method", "kdtree")).lower()
    radius_nm = float(data_prefs.get("local_density_radius", 100.0))
    voxel_nm = float(data_prefs.get("local_density_voxel_size", radius_nm))
    smooth_sigma = float(data_prefs.get("local_density_smooth_sigma", 1.0))
    dimensions = int(data_prefs.get("local_density_dimensions", 3 if ds.prop.num_dim == 3 else 2))

    return compute_local_density(
        ds,
        dimensions=dimensions,
        radius_nm=radius_nm,
        method=method,
        voxel_size_nm=voxel_nm,
        smooth_sigma=smooth_sigma,
    )[:3]


def compute_local_density(
    ds,
    *,
    dimensions: int,
    radius_nm: float,
    method: str,
    voxel_size_nm: float | None = None,
    smooth_sigma: float = 1.0,
) -> tuple[np.ndarray, str, str, np.ndarray, tuple[np.ndarray, np.ndarray]]:
    """Compute local density and a 2D density image for display."""
    mask = np.asarray(ds.filter_mask, dtype=bool)
    points = np.asarray(ds.loc_nm, dtype=float)
    valid = mask & np.isfinite(points).all(axis=1)
    density_full = np.full(points.shape[0], np.nan, dtype=float)
    active_points = points[valid, : max(2, min(int(dimensions), 3))]
    method_key = str(method).lower()
    voxel_nm = float(voxel_size_nm if voxel_size_nm is not None else radius_nm)

    if method_key in {"histogram", "histogram_2d", "2d histogram"}:
        if int(dimensions) != 2:
            raise NotImplementedError("Histogram local density is currently implemented for 2D only.")
        values = local_density_histogram_2d(
            active_points,
            voxel_size_nm=voxel_nm,
            smooth_sigma=smooth_sigma,
        )
        method_label = "2D histogram"
        detail = f"voxel={voxel_nm:g} nm, sigma={smooth_sigma:g}"
    else:
        values = local_density_kdtree(active_points, radius_nm)
        method_label = f"{int(dimensions)}D KD-tree range search"
        detail = f"radius={radius_nm:g} nm"

    density_full[valid] = values
    image, edges = density_image_xy(points[valid], values, pixel_size_nm=voxel_nm)
    return density_full, method_label, detail, image, edges


def density_image_xy(
    points_nm: np.ndarray,
    density_values: np.ndarray,
    *,
    pixel_size_nm: float,
) -> tuple[np.ndarray, tuple[np.ndarray, np.ndarray]]:
    """Bin per-localization density values into an XY image by pixel mean."""
    pts = _finite_points(points_nm, force_3d=False)
    values = np.asarray(density_values, dtype=float).ravel()
    keep = np.isfinite(values)
    pts = pts[keep]
    values = values[keep]
    if pts.size == 0 or values.size == 0:
        return np.empty((0, 0), dtype=float), (np.array([]), np.array([]))

    pixel_size_nm = max(float(pixel_size_nm), np.finfo(float).eps)
    x_edges = _edges_for(pts[:, 0], pixel_size_nm)
    y_edges = _edges_for(pts[:, 1], pixel_size_nm)
    sums, _, _ = np.histogram2d(pts[:, 0], pts[:, 1], bins=[x_edges, y_edges], weights=values)
    counts, _, _ = np.histogram2d(pts[:, 0], pts[:, 1], bins=[x_edges, y_edges])
    with np.errstate(divide="ignore", invalid="ignore"):
        image = np.where(counts > 0, sums / counts, 0.0)
    return image.astype(float), (x_edges, y_edges)


def run_local_density(parent, state) -> None:
    """Ask for settings, compute local density, and show the density image."""
    ds = state.active_dataset
    if ds is None:
        return

    dialog = LocalDensityOptionsDialog(ds, state.prefs, parent=parent)
    if dialog.exec() != QDialog.DialogCode.Accepted:
        return
    opts = dialog.options()
    data_prefs = state.prefs.setdefault("data", {})
    data_prefs["local_density_dimensions"] = int(opts["dimensions"])
    data_prefs["local_density_radius"] = float(opts["radius_nm"])
    data_prefs["local_density_method"] = str(opts["method"])
    data_prefs["local_density_voxel_size"] = float(opts["voxel_size_nm"])
    data_prefs["local_density_smooth_sigma"] = float(opts["smooth_sigma"])
    try:
        state.save_prefs()
    except Exception:
        pass

    try:
        density, method, detail, image, edges = compute_local_density(ds, **opts)
    except Exception as exc:
        state.log(f"Local density failed: {exc}", "ERROR")
        QMessageBox.critical(parent, "Local Density", str(exc))
        return

    ds.attr["local_density"] = density
    ds.derived["local_density"] = density
    if "local_density" not in ds.prop.attr_names:
        ds.prop.attr_names.append("local_density")

    finite = density[np.isfinite(density)]
    msg = (
        f"Computed local density for '{ds.name}' using {method} ({detail}). "
        f"Valid values: {finite.size:,}; median={np.nanmedian(finite):.3g}, "
        f"mean={np.nanmean(finite):.3g}."
        if finite.size
        else f"Computed local density for '{ds.name}', but no finite values were produced."
    )
    state.log(msg)
    try:
        state.journal.add("analysis", msg)
    except Exception:
        pass
    win = LocalDensityImageWindow(
        image,
        edges,
        title=f"Local Density — {ds.name}",
        info=msg,
        parent=None,
    )
    if parent is not None:
        windows = getattr(parent, "_local_density_windows", None)
        if windows is None:
            windows = []
            setattr(parent, "_local_density_windows", windows)
        windows.append(win)
        win.destroyed.connect(lambda *_args, w=win: windows.remove(w) if w in windows else None)
    win.show()


class LocalDensityOptionsDialog(QDialog):
    """Small ordered options dialog for Analyze > Local Density."""

    def __init__(self, ds, prefs: dict, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self.setWindowTitle("Local Density")
        self.setModal(True)
        self.resize(360, 180)
        data_prefs = prefs.get("data", {}) if prefs else {}

        root = QVBoxLayout(self)
        form = QFormLayout()

        self._dimensions = QComboBox()
        self._dimensions.addItem("2D (XY)", 2)
        self._dimensions.addItem("3D (XYZ)", 3)
        default_dim = 3 if getattr(ds.prop, "num_dim", 2) == 3 else 2
        idx = self._dimensions.findData(int(data_prefs.get("local_density_dimensions", default_dim)))
        self._dimensions.setCurrentIndex(max(idx, 0))
        self._dimensions.currentIndexChanged.connect(self._update_method_state)
        form.addRow("Compute density in", self._dimensions)

        self._radius = QDoubleSpinBox()
        self._radius.setRange(0.1, 1_000_000.0)
        self._radius.setDecimals(2)
        self._radius.setValue(float(data_prefs.get("local_density_radius", 100.0)))
        self._radius.setSuffix(" nm")
        form.addRow("Local density radius", self._radius)

        self._method = QComboBox()
        self._method.addItem("KD-tree range search", "kdtree")
        self._method.addItem("Histogram", "histogram_2d")
        idx = self._method.findData(data_prefs.get("local_density_method", "kdtree"))
        self._method.setCurrentIndex(max(idx, 0))
        form.addRow("Method", self._method)

        root.addLayout(form)
        self._hint = QLabel("")
        self._hint.setStyleSheet("color: gray;")
        self._hint.setWordWrap(True)
        root.addWidget(self._hint)

        buttons = QDialogButtonBox(QDialogButtonBox.StandardButton.Ok | QDialogButtonBox.StandardButton.Cancel)
        buttons.accepted.connect(self.accept)
        buttons.rejected.connect(self.reject)
        root.addWidget(buttons)
        self._update_method_state()

    def options(self) -> dict:
        return {
            "dimensions": int(self._dimensions.currentData()),
            "radius_nm": float(self._radius.value()),
            "method": str(self._method.currentData()),
            "voxel_size_nm": float(self._radius.value()),
            "smooth_sigma": 1.0,
        }

    def _update_method_state(self) -> None:
        is_3d = int(self._dimensions.currentData()) == 3
        hist_idx = self._method.findData("histogram_2d")
        if hist_idx >= 0:
            model_item = self._method.model().item(hist_idx)
            if model_item is not None:
                model_item.setEnabled(not is_3d)
        if is_3d and self._method.currentData() == "histogram_2d":
            self._method.setCurrentIndex(self._method.findData("kdtree"))
        self._hint.setText(
            "Histogram density is currently available for 2D only. "
            "3D density uses KD-tree range search and is displayed as an XY projection."
            if is_3d
            else "The result image shows XY pixels whose values are local density readouts."
        )


class LocalDensityImageWindow(QWidget):
    """Display a local-density image with a hot/red colormap."""

    TAG = "local_density_image"

    def __init__(
        self,
        image: np.ndarray,
        edges: tuple[np.ndarray, np.ndarray],
        *,
        title: str,
        info: str,
        parent: QWidget | None = None,
    ) -> None:
        super().__init__(parent)
        self.setWindowTitle(title)
        self.setWindowFlags(Qt.WindowType.Window)
        self.resize(760, 620)
        self.setAttribute(Qt.WidgetAttribute.WA_DeleteOnClose)

        root = QVBoxLayout(self)
        self._view = pg.ImageView()
        self._view.ui.roiBtn.hide()
        self._view.ui.menuBtn.hide()
        self._view.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Expanding)
        root.addWidget(self._view, stretch=1)

        label = QLabel(info)
        label.setWordWrap(True)
        label.setStyleSheet("color: gray;")
        root.addWidget(label)

        img = np.asarray(image, dtype=float)
        if img.size == 0:
            img = np.zeros((1, 1), dtype=float)
        self._view.setImage(img.T, autoLevels=True)
        self._view.setColorMap(make_colormap("hot"))
        x_edges, y_edges = edges
        if len(x_edges) > 1 and len(y_edges) > 1:
            item = self._view.getImageItem()
            item.setRect(float(x_edges[0]), float(y_edges[0]), float(x_edges[-1] - x_edges[0]), float(y_edges[-1] - y_edges[0]))
            self._view.getView().setAspectLocked(True)


def _finite_points(points_nm: np.ndarray, *, force_3d: bool = True) -> np.ndarray:
    pts = np.asarray(points_nm, dtype=float)
    if pts.ndim != 2:
        pts = np.reshape(pts, (-1, 1))
    if force_3d and pts.shape[1] == 2:
        pts = np.column_stack([pts, np.zeros(pts.shape[0])])
    return pts[np.isfinite(pts).all(axis=1)]


def _edges_for(values: np.ndarray, step: float) -> np.ndarray:
    lo = float(np.nanmin(values))
    hi = float(np.nanmax(values))
    if hi <= lo:
        hi = lo + step
    return np.arange(lo, hi + step, step, dtype=float)
