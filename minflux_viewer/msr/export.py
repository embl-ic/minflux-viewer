"""MSR export adapters.

The MSR parser returns source arrays, including legacy m2205 nested-iteration
arrays.  Serialization must not operate on those arrays directly: doing so
turns the nested ``mfx.itr`` structure into names such as ``itr_itr`` and
``itr_loc``.  This module is deliberately only an adapter around the canonical
loader and File > Save writers.
"""

from __future__ import annotations

from pathlib import Path
from typing import List, Optional

import numpy as np


def ensure_dir(path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True)


def make_export_base(export_root: str, did: str, idx: int) -> str:
    """Backwards-compatible helper for older command-line flows."""
    out_root = Path(export_root)
    ensure_dir(out_root)
    return str(out_root / f"data#{idx}_(did#{did})")


def _structured_field_names(array: np.ndarray) -> set[str]:
    return set(getattr(getattr(array, "dtype", None), "names", None) or ())


def _nested_field_names(array: np.ndarray, name: str) -> set[str]:
    fields = getattr(getattr(array, "dtype", None), "fields", None) or {}
    field = fields.get(name)
    if field is None:
        return set()
    dtype = field[0]
    # A legacy m2205 field is commonly a structured subarray, whose ``names``
    # live on ``dtype.base`` rather than on the shaped dtype itself.
    return set(
        getattr(dtype, "names", None)
        or getattr(getattr(dtype, "base", None), "names", None)
        or ()
    )


def _validate_mfx_source(mfx: np.ndarray) -> None:
    """Reject selections that cannot be normalized without inventing data."""
    arr = np.asarray(mfx)
    names = _structured_field_names(arr)
    if not names:
        raise ValueError("MSR export requires a one-dimensional structured mfx array.")
    if arr.ndim != 1:
        raise ValueError(f"MSR export requires a one-dimensional mfx array; got shape {arr.shape}.")

    # m2410 has flat itr/loc; m2205 stores the same fields below mfx.itr.
    itr_children = _nested_field_names(arr, "itr")
    if "itr" not in names:
        raise ValueError("Selected mfx fields omit required 'itr'; cannot produce canonical m2410 data.")
    if "loc" not in names and "loc" not in itr_children:
        raise ValueError("Selected mfx fields omit required 'loc'; cannot produce canonical m2410 data.")
    if "vld" not in names:
        raise ValueError("Selected mfx fields omit required 'vld'; validity cannot be reconstructed safely.")


def _validate_mbm_source(mbm: np.ndarray) -> None:
    arr = np.asarray(mbm)
    if arr.dtype.names is None or arr.ndim != 1:
        raise ValueError("MSR export requires MBM data to be a one-dimensional structured array.")


def _canonical_dataset(
    mfx: np.ndarray,
    *,
    name: str,
    folder: str,
    mbm: np.ndarray | None,
    mbm_meta: dict | None,
):
    """Normalize source mfx once, then expose it to ``save_processed``."""
    _validate_mfx_source(mfx)
    from ..core.loader import load_from_mfx_array

    # Export is raw and must not depend on the user's last/all-iteration or
    # valid-only display preference.  mfx_raw is always built from all source
    # rows; these preferences only control temporary materialization.
    ds = load_from_mfx_array(
        np.asarray(mfx),
        name=name,
        folder=folder,
        prefs={"data": {"iter_load": "all", "only_valid_locs": False}},
    )
    if mbm is not None and np.asarray(mbm).size:
        _validate_mbm_source(mbm)
        from ..core.dataset import AttributeComponent

        points = np.asarray(mbm)
        ds.mbm = AttributeComponent({"points": points})
        ds.metadata["mbm_points"] = points
        meta = dict(mbm_meta or {})
        if meta.get("points_by_gri"):
            ds.metadata["mbm_points_by_gri"] = meta["points_by_gri"]
        if meta.get("used"):
            ds.metadata["mbm_used"] = meta["used"]
    return ds


def _planned_paths(
    out_dir: Path,
    base_name: str,
    formats: list[str],
    *,
    has_mfx: bool,
    has_mbm: bool,
) -> list[Path]:
    from ..core.save import _EXT

    paths: list[Path] = []
    for fmt in formats:
        suffix = _EXT[fmt]
        if fmt == "msr":
            if has_mfx:
                paths.append(out_dir / f"{base_name}{suffix}")
            continue
        if has_mfx:
            paths.append(out_dir / f"{base_name}_mfx{suffix}")
        if has_mbm:
            paths.append(out_dir / f"{base_name}_mbm{suffix}")
    return paths


def export_arrays(
    out_dir: str,
    base_name: str,
    formats: List[str],
    mfx: Optional[np.ndarray],
    mbm: Optional[np.ndarray],
    log=print,
    json_chunk_rows: int = 100_000,
    *,
    mbm_meta: dict | None = None,
    overwrite: bool = False,
) -> List[str]:
    """Export parsed MSR components through the canonical File > Save writers.

    The mfx output is always normalized to a flat m2410 structured array before
    any serializer runs.  Non-MSR formats use ``<base>_mfx.<ext>`` and, when
    present, ``<base>_mbm.<ext>`` companions.  ``.msr`` is the one combined
    format and preserves MBM metadata through the viewer's MSR writer.

    ``json_chunk_rows`` remains accepted for compatibility with the old fast
    writer; JSON now intentionally follows the File > Save implementation.
    """
    del json_chunk_rows  # compatibility argument; the canonical writer owns JSON policy

    from ..core.save import DATA_FORMATS, save_processed, write_raw_array

    normalized_formats: list[str] = []
    for value in formats:
        fmt = str(value).lower().lstrip(".")
        if fmt not in DATA_FORMATS:
            raise ValueError(f"Unsupported MSR export format: {value!r}.")
        if fmt not in normalized_formats:
            normalized_formats.append(fmt)
    if not normalized_formats:
        raise ValueError("Select at least one MSR export format.")

    mfx_arr = None if mfx is None else np.asarray(mfx)
    mbm_arr = None if mbm is None else np.asarray(mbm)
    has_mfx = mfx_arr is not None and mfx_arr.size > 0
    has_mbm = mbm_arr is not None and mbm_arr.size > 0
    if not has_mfx and not has_mbm:
        raise ValueError("MSR export has no selected mfx or MBM data.")
    if "msr" in normalized_formats and not has_mfx:
        raise ValueError("The .msr export requires mfx localizations; MBM-only output is ambiguous.")

    out_root = Path(out_dir)
    ensure_dir(out_root)
    planned = _planned_paths(
        out_root,
        str(base_name),
        normalized_formats,
        has_mfx=has_mfx,
        has_mbm=has_mbm,
    )
    if len({str(path.resolve()) for path in planned}) != len(planned):
        raise ValueError("MSR export produced duplicate output paths; choose a unique dataset name.")
    if not overwrite:
        existing = [path for path in planned if path.exists()]
        if existing:
            joined = ", ".join(str(path) for path in existing)
            raise FileExistsError(
                f"Refusing to overwrite existing MSR export file(s): {joined}. "
                "Choose another output folder/name or pass overwrite=True."
            )

    ds = None
    if has_mfx:
        ds = _canonical_dataset(
            mfx_arr,
            name=f"{base_name}.msr",
            folder=str(out_root),
            mbm=mbm_arr if has_mbm else None,
            mbm_meta=mbm_meta,
        )

    written: list[str] = []
    for fmt in normalized_formats:
        if fmt == "msr":
            paths = save_processed(
                ds,
                data_path=out_root / str(base_name),
                fmt="msr",
                content="raw",
                include={"attrs": True, "derived": False, "recipe": False},
            )
            written.extend(str(path) for path in paths)
            log(f"[msr] wrote {paths[0]}")
            continue

        if has_mfx:
            paths = save_processed(
                ds,
                data_path=out_root / f"{base_name}_mfx",
                fmt=fmt,
                content="raw",
                include={"attrs": True, "derived": False, "recipe": False},
            )
            written.extend(str(path) for path in paths)
            log(f"[{fmt}] wrote {paths[0]}")

        if has_mbm:
            mbm_path = write_raw_array(
                out_root / f"{base_name}_mbm",
                fmt,
                mbm_arr,
                root="mbm",
            )
            written.append(str(mbm_path))
            log(f"[{fmt}] wrote {mbm_path}")

    return written
