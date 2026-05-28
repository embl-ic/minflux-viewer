"""
minflux_viewer.core.spatial_grid
=================================
Spatial index for fast bounding-box queries over localization arrays.

Build once per (dataset, mask_version); query O(k) where k is the number
of localizations returned, rather than O(N) scan over all locs.
"""

from __future__ import annotations

import numpy as np

from ..ui.render_config import SPATIAL_GRID_CELL_NM


class SpatialGrid:
    """2-D grid index mapping (col, row) cell → numpy index array.

    Parameters
    ----------
    xnm, ynm:
        1-D arrays of x and y coordinates in nanometres.  Must be the
        already-filtered (masked) arrays — the grid stores indices into
        these arrays, not into the original full dataset.
    cell_size_nm:
        Physical size of each grid cell in nm.
    """

    def __init__(
        self,
        xnm: np.ndarray,
        ynm: np.ndarray,
        cell_size_nm: float = SPATIAL_GRID_CELL_NM,
    ) -> None:
        self.cell_size = float(cell_size_nm)
        n = len(xnm)
        if n == 0:
            self._x0 = 0.0
            self._y0 = 0.0
            self._grid: dict[tuple[int, int], np.ndarray] = {}
            return

        self._x0 = float(xnm.min())
        self._y0 = float(ynm.min())

        cols = ((xnm - self._x0) / self.cell_size).astype(np.int32)
        rows = ((ynm - self._y0) / self.cell_size).astype(np.int32)

        # Group indices by cell using argsort on a combined key.
        combined = cols.astype(np.int64) * (int(rows.max()) + 2) + rows.astype(np.int64)
        order = np.argsort(combined, kind="stable")
        sorted_combined = combined[order]

        self._grid = {}
        unique_keys, starts = np.unique(sorted_combined, return_index=True)
        ends = np.empty_like(starts)
        ends[:-1] = starts[1:]
        ends[-1] = n
        sorted_cols = cols[order]
        sorted_rows = rows[order]
        for i, key in enumerate(unique_keys):
            idx_slice = order[starts[i] : ends[i]]
            c = int(sorted_cols[starts[i]])
            r = int(sorted_rows[starts[i]])
            self._grid[(c, r)] = idx_slice

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def query(
        self,
        x0: float,
        x1: float,
        y0: float,
        y1: float,
    ) -> np.ndarray:
        """Return an index array of all locs whose coordinates fall inside
        the axis-aligned bounding box [x0, x1] × [y0, y1].

        The returned array contains indices into the *xnm/ynm* arrays that
        were passed to ``__init__``.  It may contain locs slightly outside
        the box when the cell is partially overlapping — callers that need
        exact clipping should apply a secondary mask themselves.
        """
        if not self._grid:
            return np.array([], dtype=np.int64)

        col0 = int((x0 - self._x0) / self.cell_size) - 1
        col1 = int((x1 - self._x0) / self.cell_size) + 1
        row0 = int((y0 - self._y0) / self.cell_size) - 1
        row1 = int((y1 - self._y0) / self.cell_size) + 1

        parts: list[np.ndarray] = []
        for c in range(col0, col1 + 1):
            for r in range(row0, row1 + 1):
                cell = self._grid.get((c, r))
                if cell is not None:
                    parts.append(cell)

        if not parts:
            return np.array([], dtype=np.int64)
        return np.concatenate(parts)

    def __len__(self) -> int:
        return sum(len(v) for v in self._grid.values())
