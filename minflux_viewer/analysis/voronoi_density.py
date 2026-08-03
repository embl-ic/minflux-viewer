"""Projected two-dimensional Voronoi local-density fields."""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
from scipy.spatial import Delaunay, QhullError, cKDTree

MAX_VORONOI_UNIQUE_POINTS = 500_000
MAX_VORONOI_3D_UNIQUE_POINTS = 100_000


class ProjectedVoronoiError(ValueError):
    """Raised when a projected point selection cannot form a useful field."""


class Voronoi3DError(ValueError):
    """Raised when a 3-D point selection cannot form a useful field."""


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


@dataclass
class Voronoi3DField:
    """Inverse-cell-volume density and its nearest-seed spatial index."""

    points_nm: np.ndarray
    density_per_nm3: np.ndarray
    source_count: int
    finite_cell_count: int
    _tree: cKDTree = field(repr=False)

    @property
    def unique_count(self) -> int:
        return int(self.points_nm.shape[0])

    @property
    def nbytes(self) -> int:
        return int(self.points_nm.nbytes + self.density_per_nm3.nbytes)

    def sample(
        self,
        bounds: tuple[float, float, float, float, float, float],
        shape: tuple[int, int, int],
        *,
        chunk_size: int = 262_144,
    ) -> np.ndarray:
        """Sample density at voxel centres without materialising a full query grid."""
        nx, ny, nz = (max(int(value), 1) for value in shape)
        x0, x1, y0, y1, z0, z1 = (float(value) for value in bounds)
        if x1 <= x0 or y1 <= y0 or z1 <= z0:
            return np.zeros((nx, ny, nz), dtype=np.float32)

        x = x0 + (np.arange(nx, dtype=np.float64) + 0.5) * (x1 - x0) / nx
        y = y0 + (np.arange(ny, dtype=np.float64) + 0.5) * (y1 - y0) / ny
        z = z0 + (np.arange(nz, dtype=np.float64) + 0.5) * (z1 - z0) / nz
        result = np.empty((nx, ny, nz), dtype=np.float32)
        chunk_size = max(int(chunk_size), 1)

        # Work plane-by-plane and in bounded chunks.  A complete 3-D query
        # array would itself consume hundreds of MB at the volume window's
        # normal multi-million-voxel limit.
        for iz, z_value in enumerate(z):
            for start in range(0, nx * ny, chunk_size):
                stop = min(start + chunk_size, nx * ny)
                flat = np.arange(start, stop, dtype=np.int64)
                rows = flat // ny
                cols = flat - rows * ny
                query = np.column_stack((x[rows], y[cols], np.full(stop - start, z_value)))
                _, nearest = self._tree.query(query, k=1, workers=1)
                values = self.density_per_nm3[nearest]
                values = np.nan_to_num(values, nan=0.0, posinf=0.0, neginf=0.0)
                result.reshape(-1, nz)[start:stop, iz] = np.maximum(values, 0.0)
        return result


def _pair_keys(pairs: np.ndarray, point_count: int) -> np.ndarray:
    """Encode sorted integer pairs into collision-free int64 keys."""
    return (
        np.asarray(pairs[:, 0], dtype=np.int64) * int(point_count)
        + np.asarray(pairs[:, 1], dtype=np.int64)
    )


def _finite_voronoi_cell_volumes(
    points: np.ndarray, triangulation: Delaunay
) -> np.ndarray:
    """Compute bounded Voronoi-cell volumes from one 3-D Delaunay complex.

    Each Voronoi facet is dual to a Delaunay edge.  Its polygon is bounded by
    circumcentres of tetrahedra sharing the edge; the cell volume contribution
    is ``facet_area * edge_length / 6``.  Internal Delaunay faces provide the
    polygon boundary segments, so no per-cell Python polyhedron construction is
    needed.
    """
    simplices = np.asarray(triangulation.simplices, dtype=np.intp).copy()
    hull_faces = np.asarray(triangulation.convex_hull, dtype=np.intp).copy()
    point_count = int(points.shape[0])
    tetra_count = int(simplices.shape[0])
    if tetra_count == 0:
        return np.full(point_count, np.nan, dtype=np.float64)

    p0 = points[simplices[:, 0]]
    a = 2.0 * (points[simplices[:, 1]] - p0)
    b = 2.0 * (points[simplices[:, 2]] - p0)
    c = 2.0 * (points[simplices[:, 3]] - p0)
    matrix = np.stack((a, b, c), axis=1)
    rhs = np.stack(
        (
            np.einsum("ij,ij->i", points[simplices[:, 1]], points[simplices[:, 1]])
            - np.einsum("ij,ij->i", p0, p0),
            np.einsum("ij,ij->i", points[simplices[:, 2]], points[simplices[:, 2]])
            - np.einsum("ij,ij->i", p0, p0),
            np.einsum("ij,ij->i", points[simplices[:, 3]], points[simplices[:, 3]])
            - np.einsum("ij,ij->i", p0, p0),
        ),
        axis=1,
    )
    try:
        circumcentres = np.linalg.solve(matrix, rhs[..., None])[..., 0]
    except np.linalg.LinAlgError as exc:
        raise Voronoi3DError(
            "The Delaunay tetrahedra contain a singular or degenerate cell."
        ) from exc

    # Every tetrahedron contributes four sorted triangular faces.  A face with
    # two owners is internal and yields one edge of a dual Voronoi facet.
    faces = np.concatenate(
        (
            simplices[:, [1, 2, 3]],
            simplices[:, [0, 2, 3]],
            simplices[:, [0, 1, 3]],
            simplices[:, [0, 1, 2]],
        ),
        axis=0,
    )
    faces.sort(axis=1)
    face_tets = np.tile(np.arange(tetra_count, dtype=np.intp), 4)
    face_keys = (
        faces[:, 0].astype(np.int64) * point_count * point_count
        + faces[:, 1].astype(np.int64) * point_count
        + faces[:, 2].astype(np.int64)
    )
    order = np.argsort(face_keys, kind="mergesort")
    sorted_keys = face_keys[order]
    starts = np.flatnonzero(
        np.r_[True, sorted_keys[1:] != sorted_keys[:-1]]
    )
    counts = np.diff(np.r_[starts, sorted_keys.size])
    if np.any(counts > 2):
        raise Voronoi3DError("The Delaunay complex contains a face with more than two tetrahedra.")
    internal = counts == 2
    first = starts[internal]
    second = first + 1
    internal_faces = faces[order[first]]
    owner_a = face_tets[order[first]]
    owner_b = face_tets[order[second]]

    # A Delaunay edge on the convex hull has an unbounded Voronoi facet.  Such
    # cells are marked invalid below, and their dual segments are excluded from
    # the finite-volume accumulation.
    hull_edges = np.concatenate(
        (
            hull_faces[:, [0, 1]],
            hull_faces[:, [0, 2]],
            hull_faces[:, [1, 2]],
        ),
        axis=0,
    )
    hull_edges.sort(axis=1)
    hull_keys = np.unique(_pair_keys(hull_edges, point_count))

    # A Delaunay edge can occur in different pair columns depending on the
    # rank of the third vertex in its incident face.  First collect the
    # endpoint sums across all three columns so every dual facet gets one
    # common interior centroid; calculating each column independently would
    # incorrectly treat a partial facet as a complete polygon.
    partial_keys: list[np.ndarray] = []
    partial_sums: list[np.ndarray] = []
    partial_counts: list[np.ndarray] = []
    for left, right in ((0, 1), (0, 2), (1, 2)):
        edges = np.sort(internal_faces[:, [left, right]], axis=1)
        edge_keys = _pair_keys(edges, point_count)
        keep = ~np.isin(edge_keys, hull_keys)
        if not np.any(keep):
            continue
        edges = edges[keep]
        edge_keys = edge_keys[keep]
        centers_a = circumcentres[owner_a[keep]]
        centers_b = circumcentres[owner_b[keep]]

        edge_order = np.argsort(edge_keys, kind="mergesort")
        sorted_edge_keys = edge_keys[edge_order]
        edge_starts = np.flatnonzero(
            np.r_[True, sorted_edge_keys[1:] != sorted_edge_keys[:-1]]
        )
        edge_counts = np.diff(np.r_[edge_starts, sorted_edge_keys.size])
        sorted_a = centers_a[edge_order]
        sorted_b = centers_b[edge_order]
        partial_keys.append(sorted_edge_keys[edge_starts])
        partial_sums.append(
            np.add.reduceat(sorted_a, edge_starts, axis=0)
            + np.add.reduceat(sorted_b, edge_starts, axis=0)
        )
        partial_counts.append(edge_counts)

    if not partial_keys:
        return np.full(point_count, np.nan, dtype=np.float64)

    all_keys = np.concatenate(partial_keys)
    all_sums = np.concatenate(partial_sums, axis=0)
    all_counts = np.concatenate(partial_counts)
    all_order = np.argsort(all_keys, kind="mergesort")
    sorted_all_keys = all_keys[all_order]
    all_starts = np.flatnonzero(
        np.r_[True, sorted_all_keys[1:] != sorted_all_keys[:-1]]
    )
    unique_keys = sorted_all_keys[all_starts]
    centroid = (
        np.add.reduceat(all_sums[all_order], all_starts, axis=0)
        / (2.0 * np.add.reduceat(all_counts[all_order], all_starts)[:, None])
    )

    volumes = np.zeros(point_count, dtype=np.float64)
    for left, right in ((0, 1), (0, 2), (1, 2)):
        edges = np.sort(internal_faces[:, [left, right]], axis=1)
        edge_keys = _pair_keys(edges, point_count)
        keep = ~np.isin(edge_keys, hull_keys)
        if not np.any(keep):
            continue
        edges = edges[keep]
        edge_keys = edge_keys[keep]
        centers_a = circumcentres[owner_a[keep]]
        centers_b = circumcentres[owner_b[keep]]
        edge_order = np.argsort(edge_keys, kind="mergesort")
        sorted_edge_keys = edge_keys[edge_order]
        edge_starts = np.flatnonzero(
            np.r_[True, sorted_edge_keys[1:] != sorted_edge_keys[:-1]]
        )
        edge_counts = np.diff(np.r_[edge_starts, sorted_edge_keys.size])
        unique_edges = edges[edge_order[edge_starts]]
        sorted_a = centers_a[edge_order]
        sorted_b = centers_b[edge_order]
        global_centroid_index = np.searchsorted(unique_keys, sorted_edge_keys)
        centre_per_segment = centroid[global_centroid_index]
        group_index = np.repeat(np.arange(edge_starts.size, dtype=np.intp), edge_counts)
        edge_vectors = points[unique_edges[:, 1]] - points[unique_edges[:, 0]]
        edge_lengths = np.linalg.norm(edge_vectors, axis=1)
        with np.errstate(divide="ignore", invalid="ignore"):
            edge_unit = edge_vectors / edge_lengths[:, None]
            cross = np.cross(sorted_a - centre_per_segment, sorted_b - centre_per_segment)
            segment_areas = 0.5 * np.abs(
                np.einsum("ij,ij->i", cross, edge_unit[group_index])
            )
        facet_areas = np.add.reduceat(segment_areas, edge_starts)
        contributions = facet_areas * edge_lengths / 6.0
        good = np.isfinite(contributions) & (contributions > 0.0)
        if np.any(good):
            np.add.at(volumes, unique_edges[good, 0], contributions[good])
            np.add.at(volumes, unique_edges[good, 1], contributions[good])

    # Every convex-hull site has an unbounded cell.  It is not a valid local
    # volume, even though it may have some finite dual facets to interior sites.
    if hull_faces.size:
        volumes[np.unique(hull_faces)] = np.nan
    return volumes


def build_voronoi_3d_field(
    points_nm: np.ndarray,
    *,
    max_unique_points: int = MAX_VORONOI_3D_UNIQUE_POINTS,
) -> Voronoi3DField:
    """Build a bandwidth-free 3-D density field from XYZ positions.

    Exact coincident positions share one Voronoi cell and contribute their
    multiplicity to its density. Convex-hull cells are unbounded and therefore
    receive zero density, matching the projected 2-D implementation.
    """
    points = np.asarray(points_nm, dtype=np.float64)
    if points.ndim != 2 or points.shape[1] != 3:
        raise Voronoi3DError("3-D Voronoi positions must have shape (N, 3).")
    finite = np.all(np.isfinite(points), axis=1)
    finite_points = points[finite]
    source_count = int(finite_points.shape[0])
    if source_count < 5:
        raise Voronoi3DError(
            "3-D Voronoi density requires at least five finite localizations."
        )

    unique_points, multiplicity = np.unique(finite_points, axis=0, return_counts=True)
    unique_count = int(unique_points.shape[0])
    if unique_count < 5:
        raise Voronoi3DError(
            "3-D Voronoi density requires at least five distinct positions."
        )
    max_unique_points = max(int(max_unique_points), 5)
    if unique_count > max_unique_points:
        raise Voronoi3DError(
            f"3-D Voronoi density supports at most {max_unique_points:,} distinct "
            f"positions; this selection has {unique_count:,}. Narrow the region or "
            "apply a filter."
        )

    centered = unique_points - unique_points[0]
    if np.linalg.matrix_rank(centered) < 3:
        raise Voronoi3DError(
            "3-D Voronoi density requires positions that span all three dimensions."
        )

    try:
        triangulation = Delaunay(unique_points)
    except QhullError as exc:
        detail = str(exc).splitlines()[0] if str(exc) else "Qhull rejected the positions"
        raise Voronoi3DError(
            f"Could not construct the 3-D Voronoi diagram: {detail}."
        ) from exc

    volumes = _finite_voronoi_cell_volumes(unique_points, triangulation)
    density = np.divide(
        multiplicity.astype(np.float64),
        volumes,
        out=np.zeros(unique_count, dtype=np.float64),
        where=np.isfinite(volumes) & (volumes > 0.0),
    )
    finite_cell_count = int(np.count_nonzero(density > 0.0))
    if finite_cell_count == 0:
        raise Voronoi3DError(
            "3-D Voronoi density needs at least one finite interior cell; enlarge "
            "the selection or include more surrounding localizations."
        )

    del triangulation
    tree = cKDTree(unique_points, compact_nodes=True, balanced_tree=True)
    return Voronoi3DField(
        points_nm=np.ascontiguousarray(unique_points, dtype=np.float64),
        density_per_nm3=np.ascontiguousarray(density, dtype=np.float64),
        source_count=source_count,
        finite_cell_count=finite_cell_count,
        _tree=tree,
    )
