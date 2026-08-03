"""Projected two-dimensional Voronoi local-density fields."""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
from scipy.spatial import Delaunay, QhullError, cKDTree

MAX_VORONOI_UNIQUE_POINTS = 500_000


class ProjectedVoronoiError(ValueError):
    """Raised when a projected point selection cannot form a useful field."""


@dataclass
class ProjectedVoronoiField:
    """Inverse-cell-area density and its nearest-seed spatial index."""

    points_nm: np.ndarray
    density_per_nm2: np.ndarray
    source_count: int
    finite_cell_count: int
    _tree: cKDTree = field(repr=False)

    @property
    def unique_count(self) -> int:
        return int(self.points_nm.shape[0])

    @property
    def nbytes(self) -> int:
        return int(self.points_nm.nbytes + self.density_per_nm2.nbytes)

    def sample(
        self,
        bounds: tuple[float, float, float, float],
        shape: tuple[int, int],
    ) -> np.ndarray:
        """Sample density at the pixel centers of ``bounds``/``shape``."""
        height, width = max(int(shape[0]), 1), max(int(shape[1]), 1)
        x0, x1, y0, y1 = (float(value) for value in bounds)
        if x1 <= x0 or y1 <= y0:
            return np.zeros((height, width), dtype=np.float32)

        x = x0 + (np.arange(width, dtype=np.float64) + 0.5) * (x1 - x0) / width
        y = y0 + (np.arange(height, dtype=np.float64) + 0.5) * (y1 - y0) / height
        grid_x, grid_y = np.meshgrid(x, y)
        query_points = np.column_stack((grid_x.ravel(), grid_y.ravel()))
        _, nearest = self._tree.query(query_points, k=1, workers=1)
        image = self.density_per_nm2[nearest].reshape(height, width)
        np.nan_to_num(image, copy=False, nan=0.0, posinf=0.0, neginf=0.0)
        np.maximum(image, 0.0, out=image)
        return image.astype(np.float32)


def _finite_voronoi_cell_areas(
    points: np.ndarray, triangulation: Delaunay
) -> np.ndarray:
    """Compute bounded Voronoi-cell areas from one Delaunay triangulation."""
    simplices = np.asarray(triangulation.simplices, dtype=np.intp)
    p0 = points[simplices[:, 0]]
    edge_b = points[simplices[:, 1]] - p0
    edge_c = points[simplices[:, 2]] - p0
    b2 = np.einsum("ij,ij->i", edge_b, edge_b)
    c2 = np.einsum("ij,ij->i", edge_c, edge_c)
    denominator = 2.0 * (
        edge_b[:, 0] * edge_c[:, 1] - edge_b[:, 1] * edge_c[:, 0]
    )
    with np.errstate(divide="ignore", invalid="ignore"):
        offsets = np.column_stack(
            (
                (b2 * edge_c[:, 1] - edge_b[:, 1] * c2) / denominator,
                (edge_b[:, 0] * c2 - b2 * edge_c[:, 0]) / denominator,
            )
        )
    circumcenters = p0 + offsets

    triangle_ids = np.repeat(np.arange(simplices.shape[0], dtype=np.intp), 3)
    opposite = np.tile(np.arange(3, dtype=np.intp), simplices.shape[0])
    neighbors = np.asarray(triangulation.neighbors, dtype=np.intp).ravel()
    # Each internal Delaunay edge is one finite Voronoi edge. Keep it once.
    keep = neighbors > triangle_ids
    triangle_ids = triangle_ids[keep]
    opposite = opposite[keep]
    neighbors = neighbors[keep]

    site_a = simplices[triangle_ids, (opposite + 1) % 3]
    site_b = simplices[triangle_ids, (opposite + 2) % 3]
    center_a = circumcenters[triangle_ids]
    center_b = circumcenters[neighbors]

    def contributions(site_ids: np.ndarray) -> np.ndarray:
        site = points[site_ids]
        first = center_a - site
        second = center_b - site
        return 0.5 * np.abs(
            first[:, 0] * second[:, 1] - first[:, 1] * second[:, 0]
        )

    areas = np.bincount(
        site_a,
        weights=contributions(site_a),
        minlength=points.shape[0],
    ).astype(np.float64, copy=False)
    areas += np.bincount(
        site_b,
        weights=contributions(site_b),
        minlength=points.shape[0],
    )
    # Convex-hull cells are unbounded. Their partial finite contributions must
    # not be presented as valid cell areas.
    areas[np.unique(triangulation.convex_hull)] = np.nan
    return areas


def build_projected_voronoi_field(
    x_nm: np.ndarray,
    y_nm: np.ndarray,
    *,
    max_unique_points: int = MAX_VORONOI_UNIQUE_POINTS,
) -> ProjectedVoronoiField:
    """Build a bandwidth-free projected density estimator from XY positions.

    Exact coincident positions share one Voronoi cell and contribute their
    multiplicity to its density. Coordinates remain in their original floating-
    point nanometer representation; no spatial quantization is applied.
    """
    x = np.asarray(x_nm, dtype=np.float64).ravel()
    y = np.asarray(y_nm, dtype=np.float64).ravel()
    if x.size != y.size:
        raise ProjectedVoronoiError("Projected X and Y arrays must have equal length.")

    finite = np.isfinite(x) & np.isfinite(y)
    points = np.column_stack((x[finite], y[finite]))
    source_count = int(points.shape[0])
    if source_count < 4:
        raise ProjectedVoronoiError(
            "Voronoi density requires at least four finite projected localizations."
        )

    unique_points, multiplicity = np.unique(points, axis=0, return_counts=True)
    unique_count = int(unique_points.shape[0])
    if unique_count < 4:
        raise ProjectedVoronoiError(
            "Voronoi density requires at least four distinct projected positions."
        )
    max_unique_points = max(int(max_unique_points), 4)
    if unique_count > max_unique_points:
        raise ProjectedVoronoiError(
            f"Voronoi density supports at most {max_unique_points:,} distinct projected "
            f"positions; this selection has {unique_count:,}. Narrow the depth range or "
            "apply a filter."
        )

    centered = unique_points - unique_points[0]
    if np.linalg.matrix_rank(centered) < 2:
        raise ProjectedVoronoiError(
            "Voronoi density requires projected positions that are not all collinear."
        )

    try:
        triangulation = Delaunay(unique_points)
    except QhullError as exc:
        detail = str(exc).splitlines()[0] if str(exc) else "Qhull rejected the positions"
        raise ProjectedVoronoiError(
            f"Could not construct the projected Voronoi diagram: {detail}."
        ) from exc

    areas = _finite_voronoi_cell_areas(unique_points, triangulation)
    density = np.divide(
        multiplicity.astype(np.float64),
        areas,
        out=np.zeros(unique_count, dtype=np.float64),
        where=np.isfinite(areas) & (areas > 0.0),
    )
    finite_cell_count = int(np.count_nonzero(density > 0.0))

    if finite_cell_count == 0:
        raise ProjectedVoronoiError(
            "Voronoi density needs at least one finite interior cell; enlarge the "
            "selection or include more surrounding localizations."
        )

    del triangulation
    tree = cKDTree(unique_points, compact_nodes=True, balanced_tree=True)

    return ProjectedVoronoiField(
        points_nm=np.ascontiguousarray(unique_points, dtype=np.float64),
        density_per_nm2=np.ascontiguousarray(density, dtype=np.float64),
        source_count=source_count,
        finite_cell_count=finite_cell_count,
        _tree=tree,
    )
