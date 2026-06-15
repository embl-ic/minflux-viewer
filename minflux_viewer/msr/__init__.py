"""
minflux_viewer.msr
==================
Backend for reading and parsing Abberior Imspector ``.msr`` files.

Vendored from the standalone ``minflux_msr`` package and adapted for use
as a subpackage of the MINFLUX Data Viewer.

Public API
----------
``parse_msr_general(msr_file, tmp_dir, log)``
    Parse an .msr file; fills ``state.mfx_map`` with structured arrays.

``state``
    Module holding the parsed arrays (``mfx_map``, ``mbm_map``).

``export_arrays(out_dir, base_name, formats, mfx, mbm, log)``
    Export arrays to .mat / .npy / .npz / .json.

Platform note
-------------
Reading is fully cross-platform via the pure-Python ``msr-reader``; no Abberior
``specpy`` SDK or Imspector install is required. MINFLUX localizations (early
single-channel and modern multi-channel files alike) are recovered from the
embedded ``MFXDTA`` stacks, so ``parse_msr_general`` works on every platform.
"""

from .io import parse_msr_general, pick_one_msr, collect_zarr_fields
from .export import export_arrays
from .utils import slug
from . import state

__all__ = [
    "parse_msr_general",
    "pick_one_msr",
    "collect_zarr_fields",
    "export_arrays",
    "slug",
    "state",
]
