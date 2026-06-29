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
from scipy.ndimage import convolve, gaussian_filter
from scipy.signal import fftconvolve
from scipy.spatial import cKDTree

from ..ui.lut_dialog import make_colormap

#: Leaf size for the KD-tree. Larger than scipy's default (16): for radius
#: counting on dense MINFLUX data, fewer/larger leaves cut tree-traversal
#: overhead (the dominant cost) at negligible extra build time.
_KDTREE_LEAFSIZE = 32

#: Cap on histogram-density voxel-grid cells, so a too-small voxel size can't
#: silently allocate gigabytes (raised as a clear error instead).
_MAX_HISTOGRAM_CELLS = 20_000_000

# Direct voxel-kernel convolution is fast for small disks/balls. Larger kernels
# switch to FFT convolution, but this cap prevents accidental very-large padded
# arrays from exhausting memory.
_DIRECT_CONVOLUTION_OPS = 50_000_000
_MAX_FFT_PADDED_CELLS = 50_000_000

# Loading-time local-density computation should prefer exact KD-tree counts,
# but not at the cost of making file open feel stuck for very dense/large
# radius queries. These caps are intentionally conservative and only apply to
# the automatic on-load computation; the Analyze > Local Density dialog still
# lets the user explicitly request KD-tree.
_AUTO_KDTREE_MAX_EXPECTED_NEIGHBOURS = 50_000.0
_AUTO_KDTREE_MAX_TOTAL_WORK = 5_000_000_000.0
_AUTO_KDTREE_MAX_POINTS = 10_000_000


def local_density_kdtree(points_nm: np.ndarray, radius_nm: float) -> np.ndarray:
    """
    Count neighbours within ``radius_nm`` for each localization.

    The point itself is excluded, matching MATLAB ``length(x)-1`` after
    ``rangesearch([x,y,z], [x,y,z], range)``.

    The range count runs in parallel across all CPU cores (``workers=-1``) and
    only returns neighbour *counts* (``return_length=True``), never the lists —
    together ≈8× faster than the serial version while giving the identical
    result.
    """
    pts = _finite_points(points_nm, force_3d=False)
    if pts.size == 0:
        return np.empty(0, dtype=float)
    radius_nm = float(radius_nm)
    if radius_nm <= 0:
        raise ValueError("radius_nm must be positive")
    tree = cKDTree(pts, leafsize=_KDTREE_LEAFSIZE)
    counts = tree.query_ball_point(
        pts, r=radius_nm, return_length=True, workers=-1,
    )
    return np.asarray(counts, dtype=float) - 1.0


def _estimate_radius_query_work(
    points_nm: np.ndarray,
    *,
    dimensions: int,
    radius_nm: float,
) -> tuple[float, float]:
    """Estimate ``(neighbours_per_point, total_radius_work)`` for KD queries."""
    pts = _finite_points(points_nm, force_3d=False)[:, : max(2, min(int(dimensions), 3))]
    n = int(pts.shape[0])
    if n <= 1:
        return 0.0, 0.0

    radius_nm = float(radius_nm)
    if radius_nm <= 0.0:
        return 0.0, 0.0

    spans = np.ptp(pts, axis=0)
    spans = np.maximum(spans, np.finfo(float).eps)
    domain = float(np.prod(spans, dtype=np.float64))
    if not np.isfinite(domain) or domain <= 0.0:
        return float(n - 1), float(n) * float(n - 1)

    dims = pts.shape[1]
    if dims == 2:
        query_measure = float(np.pi * radius_nm * radius_nm)
    else:
        query_measure = float((4.0 / 3.0) * np.pi * radius_nm**3)
    expected = min(float(n - 1), float(n) * query_measure / domain)
    return expected, float(n) * expected


def _auto_density_method(
    points_nm: np.ndarray,
    *,
    dimensions: int,
    radius_nm: float,
) -> tuple[str, str | None]:
    """Choose the on-load local-density method and explain any fallback."""
    pts = _finite_points(points_nm, force_3d=False)
    n = int(pts.shape[0])
    expected, total_work = _estimate_radius_query_work(
        pts, dimensions=dimensions, radius_nm=radius_nm,
    )
    if n > _AUTO_KDTREE_MAX_POINTS:
        return (
            "voxel_radius",
            f"auto fallback from KD-tree: {n:,} points exceeds "
            f"{_AUTO_KDTREE_MAX_POINTS:,} point loading-time cap",
        )
    if (
        expected > _AUTO_KDTREE_MAX_EXPECTED_NEIGHBOURS
        or total_work > _AUTO_KDTREE_MAX_TOTAL_WORK
    ):
        return (
            "voxel_radius",
            f"auto fallback from KD-tree: estimated {expected:,.0f} neighbours/loc "
            f"and {total_work:,.0f} total radius-query work",
        )
    return "kdtree", None


def local_density_histogram_nd(
    points_nm: np.ndarray,
    *,
    dimensions: int = 2,
    voxel_size_nm: float | None = None,
    smooth_sigma: float = 1.0,
) -> np.ndarray:
    """
    Approximate local density from a 2D/3D binned count map.

    Each point gets the (optionally Gaussian-smoothed) count of the voxel it
    falls in — a coarse density proxy, not a per-volume normalization. Runtime
    depends on the voxel grid, not the point count, so this is far faster than
    the KD-tree range search and scales to very large/dense datasets.
    """
    dims = max(2, min(int(dimensions), 3))
    pts = _finite_points(points_nm, force_3d=False)
    if pts.size == 0:
        return np.empty(0, dtype=float)
    coords = pts[:, :dims]
    if coords.shape[1] < dims:   # 2D points requested as 3D → pad the Z column
        coords = np.column_stack(
            [coords, np.zeros((coords.shape[0], dims - coords.shape[1]))]
        )
    if voxel_size_nm is None:
        span = np.ptp(coords, axis=0)
        voxel_size_nm = float(np.nanmax(span) / 100.0) if np.any(span > 0) else 1.0
    voxel_size_nm = max(float(voxel_size_nm), np.finfo(float).eps)

    edges = [_edges_for(coords[:, d], voxel_size_nm) for d in range(dims)]
    n_cells = int(np.prod([max(len(e) - 1, 1) for e in edges]))
    if n_cells > _MAX_HISTOGRAM_CELLS:
        raise ValueError(
            f"Histogram density needs {n_cells:,} voxels at {voxel_size_nm:g} nm "
            f"(cap {_MAX_HISTOGRAM_CELLS:,}). Use a larger voxel size or the "
            f"KD-tree method."
        )
    counts, _ = np.histogramdd(coords, bins=edges)
    values = gaussian_filter(counts, smooth_sigma) if smooth_sigma > 0 else counts
    bin_idx = tuple(
        np.clip(
            np.searchsorted(edges[d], coords[:, d], side="right") - 1,
            0, values.shape[d] - 1,
        )
        for d in range(dims)
    )
    return values[bin_idx].astype(float)


def local_density_histogram_2d(
    points_nm: np.ndarray,
    *,
    voxel_size_nm: float | None = None,
    smooth_sigma: float = 1.0,
) -> np.ndarray:
    """2D binned-count density (thin wrapper over :func:`local_density_histogram_nd`)."""
    return local_density_histogram_nd(
        points_nm, dimensions=2, voxel_size_nm=voxel_size_nm, smooth_sigma=smooth_sigma,
    )


def local_density_voxel_radius_count(
    points_nm: np.ndarray,
    *,
    dimensions: int = 2,
    radius_nm: float,
    voxel_size_nm: float | None = None,
) -> np.ndarray:
    """
    Approximate radius-neighbour count using voxel disk/ball convolution.

    Points are binned into a regular 2D/3D grid. A binary disk/ball kernel whose
    physical radius is ``radius_nm`` is convolved with that grid, then each point
    receives the convolved value at its voxel. One self-count is subtracted to
    match :func:`local_density_kdtree` semantics.
    """
    dims = max(2, min(int(dimensions), 3))
    pts = _finite_points(points_nm, force_3d=False)
    if pts.size == 0:
        return np.empty(0, dtype=float)
    radius_nm = float(radius_nm)
    if radius_nm <= 0:
        raise ValueError("radius_nm must be positive")
    coords = pts[:, :dims]
    if coords.shape[1] < dims:
        coords = np.column_stack(
            [coords, np.zeros((coords.shape[0], dims - coords.shape[1]))]
        )
    if voxel_size_nm is None:
        voxel_size_nm = radius_nm
    voxel_size_nm = max(float(voxel_size_nm), np.finfo(float).eps)

    edges = [_edges_for(coords[:, d], voxel_size_nm) for d in range(dims)]
    n_cells = int(np.prod([max(len(e) - 1, 1) for e in edges]))
    if n_cells > _MAX_HISTOGRAM_CELLS:
        raise ValueError(
            f"Voxel radius count needs {n_cells:,} voxels at {voxel_size_nm:g} nm "
            f"(cap {_MAX_HISTOGRAM_CELLS:,}). Use a larger voxel size or the "
            f"KD-tree method."
        )

    counts, _ = np.histogramdd(coords, bins=edges)
    kernel = _radius_kernel(dims, radius_nm=radius_nm, voxel_size_nm=voxel_size_nm)
    values = _convolve_radius_counts(counts, kernel)
    bin_idx = tuple(
        np.clip(
            np.searchsorted(edges[d], coords[:, d], side="right") - 1,
            0, values.shape[d] - 1,
        )
        for d in range(dims)
    )
    result = values[bin_idx].astype(float) - 1.0
    return np.maximum(result, 0.0)


def _density_values(
    points_nm: np.ndarray,
    select: np.ndarray,
    *,
    dimensions: int,
    radius_nm: float,
    method: str,
    voxel_size_nm: float | None = None,
    smooth_sigma: float = 1.0,
) -> tuple[np.ndarray, str, str, np.ndarray]:
    """Method dispatch shared by the dataset and points-level entry points.

    Returns ``(density_full, method_label, detail, valid)`` where
    ``density_full`` is aligned to ``points_nm`` rows (NaN outside ``valid``).
    """
    points = np.asarray(points_nm, dtype=float)
    valid = np.asarray(select, dtype=bool) & np.isfinite(points).all(axis=1)
    density_full = np.full(points.shape[0], np.nan, dtype=float)
    active_points = points[valid, : max(2, min(int(dimensions), 3))]
    method_key = str(method).lower()
    voxel_nm = float(voxel_size_nm if voxel_size_nm is not None else radius_nm)

    auto_reason: str | None = None
    if method_key in {"auto", "auto_load", "auto_kdtree"}:
        method_key, auto_reason = _auto_density_method(
            active_points, dimensions=int(dimensions), radius_nm=radius_nm,
        )

    if method_key in {"voxel_histogram", "histogram", "histogram_3d", "histogram_nd"}:
        values = local_density_histogram_nd(
            active_points,
            dimensions=int(dimensions),
            voxel_size_nm=voxel_nm,
            smooth_sigma=smooth_sigma,
        )
        method_label = f"{int(dimensions)}D voxel histogram"
        detail = f"voxel={voxel_nm:g} nm, sigma={smooth_sigma:g}"
    elif method_key in {
        "voxel_radius",
        "voxel_radius_count",
        "voxel_ball",
        "voxel_disk",
        "radius_histogram",
    }:
        values = local_density_voxel_radius_count(
            active_points,
            dimensions=int(dimensions),
            radius_nm=radius_nm,
            voxel_size_nm=voxel_nm,
        )
        shape_name = "disk" if int(dimensions) == 2 else "ball"
        method_label = f"{int(dimensions)}D voxel {shape_name} radius count"
        detail = f"radius={radius_nm:g} nm, voxel={voxel_nm:g} nm"
        if auto_reason:
            detail = f"{detail}; {auto_reason}"
    else:
        try:
            values = local_density_kdtree(active_points, radius_nm)
            method_label = f"{int(dimensions)}D KD-tree range search"
            detail = f"radius={radius_nm:g} nm"
        except MemoryError as exc:
            if str(method).lower() not in {"auto", "auto_load", "auto_kdtree"}:
                raise
            values = local_density_voxel_radius_count(
                active_points,
                dimensions=int(dimensions),
                radius_nm=radius_nm,
                voxel_size_nm=voxel_nm,
            )
            shape_name = "disk" if int(dimensions) == 2 else "ball"
            method_label = f"{int(dimensions)}D voxel {shape_name} radius count"
            detail = (
                f"radius={radius_nm:g} nm, voxel={voxel_nm:g} nm; "
                f"auto fallback from KD-tree after MemoryError: {exc}"
            )

    density_full[valid] = values
    return density_full, method_label, detail, valid


def compute_local_density_for_points(
    points_nm: np.ndarray, prefs: dict, *, dimensions: int,
) -> tuple[np.ndarray, str, str]:
    """Loading-time local density for an arbitrary (N, 3) nm point array.

    Uses exact KD-tree by default and automatically falls back to voxel
    disk/ball radius counts when the estimated radius-query work is too large.
    """
    data_prefs = prefs.get("data", {}) if prefs else {}
    points = np.asarray(points_nm, dtype=float)
    radius_nm = float(data_prefs.get("local_density_radius", 100.0))
    density_full, method_label, detail, _ = _density_values(
        points,
        np.ones(points.shape[0], dtype=bool),
        dimensions=int(dimensions),
        radius_nm=radius_nm,
        method="auto_load",
        voxel_size_nm=radius_nm,
        smooth_sigma=1.0,
    )
    return density_full, method_label, detail


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
    radius_nm = float(data_prefs.get("local_density_radius", 100.0))
    dimensions = 3 if ds.prop.num_dim == 3 else 2

    return compute_local_density(
        ds,
        dimensions=dimensions,
        radius_nm=radius_nm,
        method="auto_load",
        voxel_size_nm=radius_nm,
        smooth_sigma=1.0,
        use_filter=False,
    )[:3]


def compute_local_density(
    ds,
    *,
    dimensions: int,
    radius_nm: float,
    method: str,
    voxel_size_nm: float | None = None,
    smooth_sigma: float = 1.0,
    use_filter: bool = True,
) -> tuple[np.ndarray, str, str, np.ndarray, tuple[np.ndarray, np.ndarray]]:
    """Compute local density and a 2D density image for display."""
    points = np.asarray(ds.loc_nm, dtype=float)
    mask = (
        np.asarray(ds.filter_mask, dtype=bool)
        if use_filter
        else np.ones(points.shape[0], dtype=bool)
    )
    voxel_nm = float(voxel_size_nm if voxel_size_nm is not None else radius_nm)
    density_full, method_label, detail, valid = _density_values(
        points,
        mask,
        dimensions=dimensions,
        radius_nm=radius_nm,
        method=method,
        voxel_size_nm=voxel_nm,
        smooth_sigma=smooth_sigma,
    )
    image, edges = density_image_xy(points[valid], density_full[valid], pixel_size_nm=voxel_nm)
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

    ds.attr["den"] = density
    ds.attr["local_density"] = density
    ds.derived["den"] = density
    ds.derived["local_density"] = density
    for name in ("den", "local_density"):
        if name not in ds.prop.attr_names:
            ds.prop.attr_names.append(name)

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
    # Modeless + non-owned, but registered on the owner so it is closed with the
    # main window on shutdown (close_modeless in _close_all_child_windows).
    from ..ui.modeless import show_modeless
    show_modeless(win, parent)


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
        self._method.addItem("Voxel histogram", "voxel_histogram")
        self._method.addItem("Voxel disk/ball radius count", "voxel_radius")
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
        self._hint.setText(
            "KD-tree counts neighbours within the radius (exact, parallelised). "
            "Histogram bins counts into voxels (faster, coarser). "
            "Voxel disk/ball counts neighbouring voxels inside the radius "
            "(fast approximation). "
            "The result image shows an XY projection of the density."
            if is_3d
            else "KD-tree is exact; histogram is a voxel proxy; voxel disk/ball is a fast radius-count approximation."
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
    return np.arange(lo, hi + step * (1.0 + 1e-9), step, dtype=float)


def _radius_kernel(
    dimensions: int,
    *,
    radius_nm: float,
    voxel_size_nm: float,
) -> np.ndarray:
    half_width = int(np.ceil(float(radius_nm) / float(voxel_size_nm)))
    offsets = np.arange(-half_width, half_width + 1, dtype=float) * float(voxel_size_nm)
    grids = np.meshgrid(*([offsets] * int(dimensions)), indexing="ij")
    dist2 = np.zeros_like(grids[0], dtype=float)
    for grid in grids:
        dist2 += grid * grid
    kernel = dist2 <= (float(radius_nm) + 1e-9) ** 2
    if not np.any(kernel):
        center = tuple(s // 2 for s in kernel.shape)
        kernel[center] = True
    return kernel.astype(float)


def _convolve_radius_counts(counts: np.ndarray, kernel: np.ndarray) -> np.ndarray:
    counts = np.asarray(counts, dtype=float)
    kernel = np.asarray(kernel, dtype=float)
    direct_ops = int(counts.size) * int(kernel.size)
    if direct_ops <= _DIRECT_CONVOLUTION_OPS:
        return convolve(counts, kernel, mode="constant", cval=0.0)

    padded_shape = tuple(c + k - 1 for c, k in zip(counts.shape, kernel.shape))
    padded_cells = int(np.prod(padded_shape))
    if padded_cells > _MAX_FFT_PADDED_CELLS:
        raise ValueError(
            f"Voxel radius count would need FFT grid {padded_cells:,} cells "
            f"(cap {_MAX_FFT_PADDED_CELLS:,}). Use a larger voxel size, smaller "
            f"radius, or the KD-tree method."
        )
    result = fftconvolve(counts, kernel, mode="same")
    result[np.abs(result) < 1e-9] = 0.0
    return result
