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
The underlying ``specpy`` library is a **Windows-only** binary supplied by
Abberior Instruments.  On other platforms ``parse_msr_general`` will raise
an ``ImportError``; the UI layer handles this gracefully.
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
