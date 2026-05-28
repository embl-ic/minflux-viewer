"""
minflux_viewer.ui.render_config
================================
Tunable constants for the pyramid / lazy-load render pipeline.

All physical units are nanometres.
"""

from __future__ import annotations

# ---------------------------------------------------------------------------
# Physical tile grid
# ---------------------------------------------------------------------------

PHYSICAL_TILE_NM: float = 5_000.0
"""Side length of one physical tile in nm.  Increase for very sparse data."""

SPATIAL_GRID_CELL_NM: float = 2_000.0
"""Cell size for the SpatialGrid spatial index in nm."""

# ---------------------------------------------------------------------------
# LOD pyramid
# ---------------------------------------------------------------------------

LOD_BREAKPOINTS_NM: list[float] = [100.0, 20.0, 5.0, 1.0]
"""nm/pixel thresholds that separate LOD levels.
   LOD 0 fires when pixel_size >= LOD_BREAKPOINTS_NM[0], etc.
   LOD len(LOD_BREAKPOINTS_NM) is the finest level."""

RENDER_TILE_PX: dict[int, int] = {
    0: 50,
    1: 100,
    2: 256,
    3: 512,
    4: 1024,
}
"""Render resolution (pixels per tile side) at each LOD level."""

PER_LOC_SWITCH_COUNT: int = 500
"""Switch from histogram to per-localization Gaussian when the tile contains
   fewer than this many localizations (at any LOD)."""

# ---------------------------------------------------------------------------
# Cache
# ---------------------------------------------------------------------------

TILE_CACHE_CAPACITY: int = 512
"""Maximum number of rendered tiles kept in the LRU cache.
   At LOD 2 (256×256 float32) this is roughly 128 MB."""

# ---------------------------------------------------------------------------
# Async scheduler
# ---------------------------------------------------------------------------

WORKER_THREAD_COUNT: int = 4
"""Maximum render worker threads.  Leave headroom for the UI thread."""

PREFETCH_RING_SIZE: int = 1
"""Number of tile rings outside the viewport to prefetch when idle."""

PREFETCH_PRIORITY: int = -1
"""QThreadPool priority for prefetch workers (< 0 = below normal)."""

DEBOUNCE_MS: int = 25
"""Milliseconds between viewport change and render trigger."""

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

_N_LODS = len(LOD_BREAKPOINTS_NM) + 1


def lod_for_pixel_size(pixel_size_nm: float) -> int:
    """Return the LOD index for a given viewport pixel size in nm."""
    for lod, threshold in enumerate(LOD_BREAKPOINTS_NM):
        if pixel_size_nm >= threshold:
            return lod
    return _N_LODS - 1


def render_tile_px(lod: int) -> int:
    """Return the render resolution (px per side) for a given LOD."""
    return RENDER_TILE_PX.get(lod, RENDER_TILE_PX[max(RENDER_TILE_PX)])


def actual_pixel_size_nm(lod: int) -> float:
    """Return the nm/pixel resolution a tile is rendered at for *lod*."""
    return PHYSICAL_TILE_NM / render_tile_px(lod)
