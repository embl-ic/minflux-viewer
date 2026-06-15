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
Two readers back this package. ``specpy`` (Abberior's **Windows-only** SDK) is
preferred when installed — it is the vendor reference and exposes the richest
modern-file metadata. When specpy is unavailable (e.g. Linux/macOS), parsing
falls back to the pure-Python, cross-platform ``msr-reader``: MINFLUX
localizations are recovered from the embedded ``MFXDTA`` stacks (both early and
modern files carry them), so ``parse_msr_general`` works on every platform.
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
