"""
minflux_viewer.utils.paraview_export
=====================================
Export a MINFLUX dataset as a ParaView-readable file.

We write a **VTK PolyData XML (.vtp)** file, which is ParaView's native
XML format for unstructured point clouds. The format is plain XML with
optional base64-encoded arrays — we use the ``ascii`` encoding for the
arrays so the file is human-readable and requires no VTK library.

Each localisation becomes one 3-D point. All numeric per-localisation
attributes (``efo``, ``cfr``, ``dcr``, ``tid``, ``tim``, …) are attached
as point scalar fields so ParaView can colour points by any of them.

Only the currently active filter mask is exported by default — what the
user sees in the viewer is what lands in ParaView.  The same mask is also
written as a point scalar field named ``ftr`` so ParaView can display or
filter by it explicitly when exporting all points.
"""

from __future__ import annotations

import base64
from pathlib import Path

import numpy as np

from ..core.dataset import MinfluxDataset


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def export_to_vtp(
    dataset: MinfluxDataset,
    out_path: str | Path,
    apply_filter: bool = True,
) -> Path:
    """
    Write *dataset* to ``out_path`` as a VTK PolyData XML file.

    Parameters
    ----------
    dataset:
        Dataset to export.
    out_path:
        Target ``.vtp`` path.  Parent directory is created if missing.
    apply_filter:
        When True (default), only localisations passing the current
        ``filter_mask`` are exported.

    Returns
    -------
    Path
        The absolute path written (convenience for the caller).
    """
    out_path = Path(out_path).expanduser().resolve()
    out_path.parent.mkdir(parents=True, exist_ok=True)

    # Select points
    mask = dataset.filter_mask if apply_filter else np.ones(
        dataset.prop.num_loc, dtype=bool
    )
    loc_nm = dataset.loc_nm[mask]      # (N, 3) nanometres

    # ParaView expects metres in scientific-data convention; MINFLUX raw
    # units are already metres, but our ``loc_nm`` is in nm.  Convert
    # back to metres so that ParaView's ruler / histograms show SI units.
    points_m = loc_nm * 1e-9
    n_points = points_m.shape[0]

    # Gather per-point scalar attributes (1-D numeric arrays only)
    scalar_attrs: dict[str, np.ndarray] = {}
    scalar_attrs["ftr"] = np.asarray(dataset.filter_mask, dtype=np.uint8)[mask]
    for name in dataset.attr.keys():
        if name in ("ftr", "idx"):
            continue
        arr = np.asarray(dataset.attr[name])
        if arr.ndim != 1 or arr.shape[0] != dataset.prop.num_loc:
            continue
        if not np.issubdtype(arr.dtype, np.number):
            continue
        scalar_attrs[name] = arr[mask]

    _write_vtp_ascii(out_path, points_m, scalar_attrs)
    return out_path


# ---------------------------------------------------------------------------
# .vtp writer
# ---------------------------------------------------------------------------

def _write_vtp_ascii(
    path: Path,
    points: np.ndarray,
    scalars: dict[str, np.ndarray],
) -> None:
    """
    Write a VTK PolyData XML file using ASCII arrays (no base64).

    For very large datasets ASCII is verbose but the file opens reliably
    in ParaView on every platform.  For datasets with > 1,000,000 points
    we switch to raw-binary appended data for speed and size.
    """
    n = int(points.shape[0])

    # Choose format: ascii for small datasets, binary appended for large.
    if n > 1_000_000:
        _write_vtp_binary(path, points, scalars)
        return

    lines: list[str] = []
    lines.append('<?xml version="1.0"?>')
    lines.append(
        '<VTKFile type="PolyData" version="1.0" '
        'byte_order="LittleEndian" header_type="UInt64">'
    )
    lines.append(f'  <PolyData>')
    lines.append(
        f'    <Piece NumberOfPoints="{n}" NumberOfVerts="{n}" '
        f'NumberOfLines="0" NumberOfStrips="0" NumberOfPolys="0">'
    )

    # ── Per-point scalar data ─────────────────────────────────────────
    if scalars:
        # Expose the first scalar as the default active scalar so
        # ParaView shows a coloured point cloud on load
        active_name = next(iter(scalars.keys()))
        lines.append(f'      <PointData Scalars="{active_name}">')
        for name, values in scalars.items():
            dtype_str = _vtk_dtype_name(values.dtype)
            lines.append(
                f'        <DataArray type="{dtype_str}" Name="{name}" '
                f'format="ascii">'
            )
            lines.append("          " + _ascii_values(values))
            lines.append('        </DataArray>')
        lines.append('      </PointData>')

    # ── Points ────────────────────────────────────────────────────────
    lines.append('      <Points>')
    lines.append(
        '        <DataArray type="Float64" NumberOfComponents="3" '
        'Name="Points" format="ascii">'
    )
    lines.append("          " + _ascii_values(points.ravel()))
    lines.append('        </DataArray>')
    lines.append('      </Points>')

    # ── Verts: each point is its own vertex ───────────────────────────
    connectivity = np.arange(n, dtype=np.int64)
    offsets      = np.arange(1, n + 1, dtype=np.int64)
    lines.append('      <Verts>')
    lines.append(
        '        <DataArray type="Int64" Name="connectivity" format="ascii">'
    )
    lines.append("          " + _ascii_values(connectivity))
    lines.append('        </DataArray>')
    lines.append(
        '        <DataArray type="Int64" Name="offsets" format="ascii">'
    )
    lines.append("          " + _ascii_values(offsets))
    lines.append('        </DataArray>')
    lines.append('      </Verts>')

    lines.append('    </Piece>')
    lines.append('  </PolyData>')
    lines.append('</VTKFile>')

    path.write_text("\n".join(lines), encoding="utf-8")


def _write_vtp_binary(
    path: Path,
    points: np.ndarray,
    scalars: dict[str, np.ndarray],
) -> None:
    """
    Binary-appended .vtp for large datasets.

    Uses VTK's "appended" data format with base64 encoding, which keeps
    the file self-contained (no sidecar) while being much more compact
    than ASCII.
    """
    n = int(points.shape[0])
    connectivity = np.arange(n, dtype=np.int64)
    offsets      = np.arange(1, n + 1, dtype=np.int64)

    # Collect all arrays in order; record each one's offset in the
    # appended data block.
    blocks: list[tuple[str, np.ndarray]] = []
    for name, values in scalars.items():
        blocks.append((f"scalar:{name}", _contig(values)))
    blocks.append(("points",       _contig(points.astype(np.float64).ravel())))
    blocks.append(("connectivity", _contig(connectivity)))
    blocks.append(("offsets",      _contig(offsets)))

    # Build encoded byte string with header_type=UInt64 length prefixes
    parts: list[bytes] = []
    block_offsets: dict[str, int] = {}
    cursor = 0
    for tag, arr in blocks:
        payload = arr.tobytes()
        header  = np.uint64(len(payload)).tobytes()
        block_offsets[tag] = cursor
        cursor += len(header) + len(payload)
        parts.append(header + payload)

    appended_bytes  = b"".join(parts)
    appended_base64 = base64.b64encode(appended_bytes).decode("ascii")

    # Write XML header referencing the offsets
    lines: list[str] = []
    lines.append('<?xml version="1.0"?>')
    lines.append(
        '<VTKFile type="PolyData" version="1.0" '
        'byte_order="LittleEndian" header_type="UInt64">'
    )
    lines.append('  <PolyData>')
    lines.append(
        f'    <Piece NumberOfPoints="{n}" NumberOfVerts="{n}" '
        f'NumberOfLines="0" NumberOfStrips="0" NumberOfPolys="0">'
    )

    if scalars:
        active_name = next(iter(scalars.keys()))
        lines.append(f'      <PointData Scalars="{active_name}">')
        for name, values in scalars.items():
            dtype_str = _vtk_dtype_name(values.dtype)
            off       = block_offsets[f"scalar:{name}"]
            lines.append(
                f'        <DataArray type="{dtype_str}" Name="{name}" '
                f'format="appended" offset="{off}"/>'
            )
        lines.append('      </PointData>')

    lines.append('      <Points>')
    lines.append(
        f'        <DataArray type="Float64" NumberOfComponents="3" '
        f'Name="Points" format="appended" offset="{block_offsets["points"]}"/>'
    )
    lines.append('      </Points>')

    lines.append('      <Verts>')
    lines.append(
        f'        <DataArray type="Int64" Name="connectivity" '
        f'format="appended" offset="{block_offsets["connectivity"]}"/>'
    )
    lines.append(
        f'        <DataArray type="Int64" Name="offsets" '
        f'format="appended" offset="{block_offsets["offsets"]}"/>'
    )
    lines.append('      </Verts>')

    lines.append('    </Piece>')
    lines.append('  </PolyData>')
    lines.append('  <AppendedData encoding="base64">')
    lines.append('    _' + appended_base64)
    lines.append('  </AppendedData>')
    lines.append('</VTKFile>')

    path.write_text("\n".join(lines), encoding="utf-8")


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _vtk_dtype_name(dtype: np.dtype) -> str:
    """Map a numpy dtype to a VTK XML type name."""
    kind = np.dtype(dtype).kind
    size = np.dtype(dtype).itemsize
    if kind == "f":
        return {4: "Float32", 8: "Float64"}.get(size, "Float64")
    if kind in ("i", "b"):
        return {1: "Int8", 2: "Int16", 4: "Int32", 8: "Int64"}.get(size, "Int64")
    if kind == "u":
        return {1: "UInt8", 2: "UInt16", 4: "UInt32", 8: "UInt64"}.get(size, "UInt64")
    return "Float64"


def _ascii_values(arr: np.ndarray) -> str:
    """Space-separated ASCII representation of *arr*."""
    flat = arr.ravel()
    if np.issubdtype(flat.dtype, np.floating):
        return " ".join(f"{v:g}" for v in flat)
    return " ".join(str(int(v)) for v in flat)


def _contig(arr: np.ndarray) -> np.ndarray:
    """Return a C-contiguous copy of *arr* (or the array itself)."""
    if arr.flags.c_contiguous:
        return arr
    return np.ascontiguousarray(arr)
