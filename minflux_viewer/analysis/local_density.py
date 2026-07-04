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
from PyQt6.QtGui import QIcon
from PyQt6.QtWidgets import (
    QComboBox,
    QDialog,
    QDialogButtonBox,
    QDoubleSpinBox,
    QFormLayout,
    QHBoxLayout,
    QLabel,
    QMenu,
    QMessageBox,
    QSizePolicy,
    QStackedWidget,
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
    Count localizations within ``radius_nm`` of each localization, **including
    the point itself**, so an isolated localization reports 1 (not 0) — this
    distinguishes "a single point in empty space" from genuinely void space.
    (This differs from MATLAB ``length(...)-1`` after ``rangesearch``, which
    excludes the centre point.)

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
    return np.asarray(counts, dtype=float)


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
    receives the convolved value at its voxel. The centre point is **included**
    (the kernel covers its own voxel), matching :func:`local_density_kdtree`
    semantics so an isolated localization reports 1.
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
    result = values[bin_idx].astype(float)
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
) -> tuple[np.ndarray, str, str, np.ndarray, tuple[np.ndarray, np.ndarray], np.ndarray, np.ndarray]:
    """Compute local density, a 2D density image, and the valid points+values.

    Returns ``(density_full, method, detail, image, edges, points_valid, density_valid)``
    where ``points_valid`` (nm, 2-D or 3-D) + ``density_valid`` feed the result
    window's optional 3-D scatter."""
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
    points_valid = points[valid]
    density_valid = density_full[valid]
    # Display the density map at a pixel FINER than the computation radius, so a
    # large radius (e.g. 100 nm) doesn't produce a blocky "cubic"-looking heatmap.
    # (The KD-tree neighbourhood itself is a sphere/ball of the given radius.)
    image_pixel_nm = _image_pixel_nm(points_valid, radius_nm)
    image, edges = density_image_xy(points_valid, density_valid, pixel_size_nm=image_pixel_nm)
    return density_full, method_label, detail, image, edges, points_valid, density_valid


def _image_pixel_nm(points_nm: np.ndarray, radius_nm: float,
                    *, max_cells: int = 4_000_000, min_pixel: float = 1.0) -> float:
    """A display pixel ≈ radius/5 (smooth heatmap), enlarged only if the image
    would exceed *max_cells* — decoupled from the computation radius."""
    pixel = max(float(radius_nm) / 5.0, float(min_pixel))
    pts = _finite_points(points_nm, force_3d=False)
    if pts.shape[0] < 2:
        return pixel
    span_x, span_y = float(np.ptp(pts[:, 0])), float(np.ptp(pts[:, 1]))
    if span_x <= 0 or span_y <= 0:
        return pixel
    while (span_x / pixel) * (span_y / pixel) > max_cells:
        pixel *= 1.5
    return pixel


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

    # The dialog seeds its fields from the Preferences (a sensible starting
    # point), but the plugin is for ad-hoc exploration at different radii — it
    # must NOT write its choices back. The `data.local_density_*` preferences are
    # user-owned and drive the on-load `den` derived attribute; they change only
    # via the Preferences dialog. (Previously each run overwrote + saved them.)
    dialog = LocalDensityOptionsDialog(ds, state.prefs, parent=parent)
    if dialog.exec() != QDialog.DialogCode.Accepted:
        return
    opts = dialog.options()

    try:
        density, method, detail, image, edges, pts_valid, dens_valid = compute_local_density(ds, **opts)
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
    shape = "ball" if int(opts["dimensions"]) == 3 else "disk"
    msg = (
        f"Local density of '{ds.name}': neighbours within a {opts['radius_nm']:g} nm "
        f"{shape} ({method}, {detail}). "
        f"median={np.nanmedian(finite):.3g}, mean={np.nanmean(finite):.3g} "
        f"over {finite.size:,} localizations."
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
        points_nm=pts_valid,
        density_values=dens_valid,
        dimensions=int(opts["dimensions"]),
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
        self._method.addItem("KD-tree ball — exact, spherical (recommended)", "kdtree")
        self._method.addItem("Voxel ball radius count — fast, spherical", "voxel_radius")
        self._method.addItem("Voxel histogram — fastest, cubic voxel", "voxel_histogram")
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
        shape = "ball (sphere)" if int(self._dimensions.currentData()) == 3 else "disk (circle)"
        self._hint.setText(
            f"Counts each localization's neighbours inside a {shape} of the given "
            "radius. KD-tree is exact + parallelised (recommended); voxel ball is a "
            "fast spherical approximation; voxel histogram is fastest but uses a "
            "cubic voxel (coarser). The result window shows the XY density heatmap "
            "(plus a 3-D scatter coloured by density for 3-D data)."
        )


class LocalDensityImageWindow(QWidget):
    """Local-density result viewer: a 2-D XY heatmap and (for 3-D data) a 3-D
    scatter of the localizations coloured by density, in the app's viewer style."""

    TAG = "local_density_image"
    _CMAPS = ["hot", "inferno", "magma", "plasma", "viridis", "turbo", "jet", "gray"]
    _MAX_3D_POINTS = 400_000

    def __init__(
        self,
        image: np.ndarray,
        edges: tuple[np.ndarray, np.ndarray],
        *,
        title: str,
        info: str,
        points_nm: np.ndarray | None = None,
        density_values: np.ndarray | None = None,
        dimensions: int = 2,
        parent: QWidget | None = None,
    ) -> None:
        super().__init__(parent)
        self.setWindowTitle(title)
        self.setWindowFlags(Qt.WindowType.Window)
        try:
            from .. import resource_path
            self.setWindowIcon(QIcon(str(resource_path("icons", "minflux_viewer_logo.png"))))
        except Exception:
            pass
        self.resize(820, 720)
        self.setAttribute(Qt.WidgetAttribute.WA_DeleteOnClose)

        self._cmap = "hot"
        self._img = np.asarray(image, dtype=float)
        if self._img.size == 0:
            self._img = np.zeros((1, 1), dtype=float)
        self._edges = edges
        self._pts3d, self._dens3d, self._anchor = self._prepare_3d(points_nm, density_values, dimensions)
        self._gl_view = None
        self._gl_scatter = None

        root = QVBoxLayout(self)
        # View selector — a 3-D scatter is offered only when the data is 3-D.
        top = QHBoxLayout()
        self._view_label = QLabel("View:")
        top.addWidget(self._view_label)
        self._view_combo = QComboBox()
        self._view_combo.addItem("2D heatmap", "2d")
        if self._pts3d is not None:
            self._view_combo.addItem("3D scatter (coloured by density)", "3d")
        self._view_combo.currentIndexChanged.connect(self._on_view_changed)
        top.addWidget(self._view_combo)
        top.addSpacing(12)
        top.addWidget(QLabel("Colormap:"))
        self._cmap_combo = QComboBox()
        for name in self._CMAPS:
            self._cmap_combo.addItem(name)
        self._cmap_combo.setCurrentText(self._cmap)
        self._cmap_combo.currentTextChanged.connect(self._set_cmap)
        top.addWidget(self._cmap_combo)
        top.addStretch(1)
        root.addLayout(top)
        # hide the view selector (leave the colormap) when there is no 3-D data
        self._view_label.setVisible(self._pts3d is not None)
        self._view_combo.setVisible(self._pts3d is not None)

        self._stack = QStackedWidget()
        self._stack.addWidget(self._build_2d_widget())        # index 0
        root.addWidget(self._stack, stretch=1)

        label = QLabel(info)
        label.setWordWrap(True)
        label.setStyleSheet("color: gray;")
        root.addWidget(label)

        self._apply_image(fit=True)

    # -- 3-D data prep -------------------------------------------------------
    def _prepare_3d(self, points_nm, density_values, dimensions):
        """(pts (M,3), dens (M,), anchor) for the 3-D scatter, or (None, None, None)
        when there is no usable 3-D data (2-D request or flat Z)."""
        if points_nm is None or density_values is None or int(dimensions) != 3:
            return None, None, None
        pts = np.asarray(points_nm, dtype=float)
        if pts.ndim != 2 or pts.shape[0] == 0 or pts.shape[1] < 3:
            return None, None, None
        dens = np.asarray(density_values, dtype=float).ravel()
        keep = np.all(np.isfinite(pts[:, :3]), axis=1) & np.isfinite(dens)
        pts, dens = pts[keep, :3], dens[keep]
        if pts.shape[0] == 0 or float(np.ptp(pts[:, 2])) <= 0:
            return None, None, None                            # flat Z ⇒ 2-D only
        if pts.shape[0] > self._MAX_3D_POINTS:                 # decimate for GL
            step = int(np.ceil(pts.shape[0] / self._MAX_3D_POINTS))
            pts, dens = pts[::step], dens[::step]
        return pts, dens, pts.mean(axis=0)

    # -- 2-D heatmap ---------------------------------------------------------
    def _build_2d_widget(self) -> QWidget:
        self._view = pg.ImageView(view=pg.PlotItem())
        self._view.ui.roiBtn.hide()
        self._view.ui.menuBtn.hide()
        self._view.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Expanding)
        self._plot = self._view.getView()
        self._plot.setAspectLocked(True)
        self._plot.setLabel("bottom", "X", units="nm")
        self._plot.setLabel("left", "Y", units="nm")
        try:
            self._plot.getViewBox().setMenuEnabled(False)     # use our context menu
        except Exception:
            pass
        gv = self._view.ui.graphicsView
        gv.setContextMenuPolicy(Qt.ContextMenuPolicy.CustomContextMenu)
        gv.customContextMenuRequested.connect(self._show_context_menu)
        return self._view

    def _apply_image(self, *, fit: bool) -> None:
        # autoRange=False so we can place the image at nm coordinates (setRect)
        # and THEN fit the view to it — otherwise the view stays ranged to the
        # pixel grid and the image sits off-screen (the old "need View All" bug).
        self._view.setImage(self._img.T, autoRange=False, autoLevels=True)
        self._view.setColorMap(make_colormap(self._cmap))
        x_edges, y_edges = self._edges
        if len(x_edges) > 1 and len(y_edges) > 1:
            item = self._view.getImageItem()
            item.setRect(float(x_edges[0]), float(y_edges[0]),
                         float(x_edges[-1] - x_edges[0]), float(y_edges[-1] - y_edges[0]))
        if fit:
            self._plot.getViewBox().autoRange()

    # -- 3-D scatter ---------------------------------------------------------
    def _scatter_colors(self) -> np.ndarray:
        """RGBA (float32) per point from the density values (1–99 % robust range)."""
        d = self._dens3d
        finite = np.isfinite(d)
        if finite.any():
            lo, hi = np.percentile(d[finite], [1.0, 99.0])
        else:
            lo, hi = 0.0, 1.0
        if hi <= lo:
            hi = lo + 1.0
        norm = np.clip((d - lo) / (hi - lo), 0.0, 1.0)
        norm[~finite] = 0.0
        cols = make_colormap(self._cmap).map(norm.astype(float), mode="float")
        return np.asarray(cols, dtype=np.float32)

    def _build_3d_widget(self) -> QWidget:
        try:
            import pyqtgraph.opengl as gl
        except Exception as exc:                               # OpenGL unavailable
            fallback = QLabel(f"3-D view unavailable: {exc}")
            fallback.setAlignment(Qt.AlignmentFlag.AlignCenter)
            fallback.setStyleSheet("color: gray;")
            return fallback
        pts = (self._pts3d - self._anchor).astype(np.float32)
        view = gl.GLViewWidget()
        view.setBackgroundColor("k")
        self._gl_scatter = gl.GLScatterPlotItem(
            pos=pts, color=self._scatter_colors(), size=3.0, pxMode=True)
        view.addItem(self._gl_scatter)
        span = float(np.ptp(pts)) or 100.0
        view.setCameraPosition(distance=span * 1.4)
        grid = gl.GLGridItem()
        grid.setSize(span * 1.5, span * 1.5)
        grid.setSpacing(max(span / 10.0, 1.0), max(span / 10.0, 1.0))
        view.addItem(grid)
        try:
            view.addItem(gl.GLAxisItem(size=pg.Vector(span / 2, span / 2, span / 2)))
        except Exception:
            pass
        self._gl_view = view
        return view

    # -- view switching / colormap ------------------------------------------
    def _on_view_changed(self, *_a) -> None:
        if self._view_combo.currentData() == "3d":
            if self._stack.count() < 2:                        # lazy-build on first use
                self._stack.addWidget(self._build_3d_widget())
            self._stack.setCurrentIndex(1)
        else:
            self._stack.setCurrentIndex(0)

    def _show_context_menu(self, pos) -> None:
        menu = QMenu(self)
        cmap_menu = menu.addMenu("Colormap")
        for name in self._CMAPS:
            act = cmap_menu.addAction(name)
            act.setCheckable(True)
            act.setChecked(self._cmap == name)
            act.triggered.connect(lambda _=False, n=name: self._set_cmap(n))
        menu.addAction("Reset view", lambda: self._plot.getViewBox().autoRange())
        menu.exec(self._view.ui.graphicsView.mapToGlobal(pos))

    def _set_cmap(self, name: str) -> None:
        self._cmap = name
        if self._cmap_combo.currentText() != name:             # keep the combo in sync
            self._cmap_combo.blockSignals(True)
            self._cmap_combo.setCurrentText(name)
            self._cmap_combo.blockSignals(False)
        self._view.setColorMap(make_colormap(name))
        if self._gl_scatter is not None:                       # recolour the 3-D scatter
            try:
                self._gl_scatter.setData(color=self._scatter_colors())
            except Exception:
                pass


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
