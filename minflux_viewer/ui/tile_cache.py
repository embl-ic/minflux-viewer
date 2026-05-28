"""
minflux_viewer.ui.tile_cache
==============================
Fixed-tile LRU cache for the pyramid render pipeline.

TileKey identifies a rendered tile by dataset, mask version, orientation,
LOD level, and physical grid position.  Because mask_version is part of
the key, stale tiles are never served — they simply fail the LRU lookup
and are eventually evicted by capacity pressure.
"""

from __future__ import annotations

from collections import OrderedDict
from dataclasses import dataclass

import numpy as np

from .render_config import TILE_CACHE_CAPACITY


@dataclass(frozen=True)
class TileKey:
    """Immutable identifier for one rendered physical tile."""

    dataset_id: int
    """Index of the dataset in AppState.datasets."""

    mask_version: int
    """Monotonically increasing counter; incremented whenever filter_mask
    or vld changes so stale tiles are never served."""

    orientation: str
    """'XY' | 'XZ' | 'YZ'"""

    lod: int
    """Level-of-detail index (0 = coarsest, 4 = finest)."""

    tile_row: int
    """Y-axis tile index in the physical grid."""

    tile_col: int
    """X-axis tile index in the physical grid."""

    transform_key: tuple
    """Hashable summary of the dataset render transform (empty tuple if none)."""

    depth_range: tuple[float, float] | None
    """Active depth gate as (lo_nm, hi_nm) rounded to 1 nm, or None for all-depth."""


class PhysicalTileCache:
    """LRU cache mapping TileKey → rendered float32 array.

    Parameters
    ----------
    capacity:
        Maximum number of tiles to retain before evicting least-recently
        used entries.
    """

    def __init__(self, capacity: int = TILE_CACHE_CAPACITY) -> None:
        self._capacity = capacity
        self._cache: OrderedDict[TileKey, np.ndarray] = OrderedDict()

    # ------------------------------------------------------------------
    # Core operations
    # ------------------------------------------------------------------

    def get(self, key: TileKey) -> np.ndarray | None:
        """Return the cached tile or None if not present."""
        entry = self._cache.get(key)
        if entry is None:
            return None
        self._cache.move_to_end(key)
        return entry

    def put(self, key: TileKey, tile: np.ndarray) -> None:
        """Insert or update a tile.  Evicts LRU entry if over capacity."""
        self._cache[key] = tile
        self._cache.move_to_end(key)
        while len(self._cache) > self._capacity:
            self._cache.popitem(last=False)

    def invalidate_dataset(self, dataset_id: int) -> None:
        """Remove all tiles for a given dataset (e.g., after dataset reload)."""
        stale = [k for k in self._cache if k.dataset_id == dataset_id]
        for k in stale:
            del self._cache[k]

    def clear(self) -> None:
        self._cache.clear()

    def __len__(self) -> int:
        return len(self._cache)

    # ------------------------------------------------------------------
    # Progressive placeholder
    # ------------------------------------------------------------------

    def get_placeholder(self, key: TileKey, target_px: int) -> np.ndarray | None:
        """Try to find a coarser-LOD tile covering the same physical region
        and upscale it as a placeholder for the missing *key* tile.

        Returns a float32 array of shape (target_px, target_px) or None.
        """
        from .render_config import PHYSICAL_TILE_NM, render_tile_px
        from scipy.ndimage import zoom as ndimage_zoom

        for coarser_lod in range(key.lod - 1, -1, -1):
            coarser_key = TileKey(
                dataset_id=key.dataset_id,
                mask_version=key.mask_version,
                orientation=key.orientation,
                lod=coarser_lod,
                tile_row=key.tile_row,
                tile_col=key.tile_col,
                transform_key=key.transform_key,
            )
            coarser = self._cache.get(coarser_key)
            if coarser is not None:
                coarser_px = render_tile_px(coarser_lod)
                if coarser_px == target_px:
                    return coarser.copy()
                factor = target_px / max(coarser_px, 1)
                try:
                    upscaled = ndimage_zoom(coarser, factor, order=1)
                    h, w = upscaled.shape
                    result = np.zeros((target_px, target_px), dtype=np.float32)
                    result[: min(h, target_px), : min(w, target_px)] = upscaled[
                        : target_px, : target_px
                    ]
                    return result
                except Exception:
                    return None
        return None
