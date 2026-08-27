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


def canonical_dataset(
    mfx: np.ndarray,
    *,
    name: str,
    folder: str,
    mbm: np.ndarray | None,
    mbm_meta: dict | None,
    source_zarr=None,
):
    """Normalize source mfx once, then expose it to ``save_processed``."""
    _validate_mfx_source(mfx)
    from ..core.loader import load_from_mfx_array

    # Export is raw and must not depend on the user's valid-only display
    # preference.  mfx_raw is always built from all source rows, and that is
    # what the writers read; the preference only controls the temporary
    # materialization, so it is pinned here rather than inherited.
    ds = load_from_mfx_array(
        np.asarray(mfx),
        name=name,
        folder=folder,
        prefs={"data": {"only_valid_locs": False}},
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
    if source_zarr is not None:
        from ..core.minflux_zarr import capture_native_zarr_metadata

        capture_native_zarr_metadata(ds, source_zarr)
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
        if fmt in {"msr", "zarr"}:
            if has_mfx:
                paths.append(out_dir / f"{base_name}{suffix}")
            continue
        if has_mfx:
            paths.append(out_dir / f"{base_name}_mfx{suffix}")
        if has_mbm:
            paths.append(out_dir / f"{base_name}_mbm{suffix}")
    return paths


#: Relative cost of writing one dataset in each format, used **only** to keep
#: the export progress reading honest. Counting one tick per dataset made the
#: bar sit at a single value for nearly the whole run: measured on a synthetic
#: 26-column canonical array (200 k / 1 M rows, ratios stable across both),
#: taking the NumPy write as 1 -- npy 1, msr 2.5, zarr 4-10, mat 20-24,
#: csv 23-29, json 51-67. Every writer scales close to linearly with row count,
#: so a weight times the row count predicts the share of the run each write is.
FORMAT_WORK_WEIGHT: dict[str, float] = {
    "npy": 1.0,
    "msr": 2.5,
    "zarr": 6.0,
    "zarr_zip": 7.0,
    "hdf5": 3.0,
    "mat": 22.0,
    "csv": 25.0,
    "json": 55.0,
}
#: Weight for a format not measured above -- between mat and npy, so an
#: unlisted writer neither dominates nor vanishes from the reading.
DEFAULT_FORMAT_WORK_WEIGHT: float = 10.0
#: Row count one weight unit is expressed per, so the numbers above are
#: "cost of writing this many rows in npy" and stay comparable to an image.
WORK_ROWS_UNIT: float = 1.0e6
#: Preparing one dataset (``canonical_dataset``: normalization + building the
#: dataset object) before any format is written -- twice the cost of the NumPy
#: write it precedes, and measured linear in row count like the writers
#: (10 k / 200 k / 1 M -> 0.024 / 0.081 / 0.387 s, once imports are warm).
#: Charging it separately is what makes the FIRST tick arrive on time.
DATASET_PREPARE_WEIGHT: float = 2.0
#: Flat floor per dataset, so one with almost no rows still moves the bar.
DATASET_WORK_OVERHEAD: float = 0.1
#: One exported image series, in the same units (an OME-TIFF page write is
#: comparable to a small array write).
IMAGE_WORK_WEIGHT: float = 1.0


def format_work_weight(fmt) -> float:
    """Relative cost of writing one million rows in *fmt* (npy = 1)."""
    key = str(fmt).lower().lstrip(".")
    return FORMAT_WORK_WEIGHT.get(key, DEFAULT_FORMAT_WORK_WEIGHT)


def dataset_prepare_weight(n_rows) -> float:
    """Cost of the shared canonical dataset every format write is built on."""
    scale = max(int(n_rows or 0), 0) / WORK_ROWS_UNIT
    return DATASET_PREPARE_WEIGHT * scale + DATASET_WORK_OVERHEAD


def dataset_work_weight(n_rows, formats) -> float:
    """Predicted share of the export run that writing *n_rows* in *formats* is."""
    scale = max(int(n_rows or 0), 0) / WORK_ROWS_UNIT
    written = sum(format_work_weight(fmt) for fmt in formats) * scale
    return written + dataset_prepare_weight(n_rows)


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
    source_zarr=None,
    overwrite: bool = False,
    on_format=None,
    on_prepared=None,
) -> List[str]:
    """Export parsed MSR components through the canonical File > Save writers.

    The mfx output is always normalized to a flat m2410 structured array before
    any serializer runs.  Non-MSR formats use ``<base>_mfx.<ext>`` and, when
    present, ``<base>_mbm.<ext>`` companions.  ``.msr`` is the one combined
    format and preserves MBM metadata through the viewer's MSR writer.

    ``json_chunk_rows`` remains accepted for compatibility with the old fast
    writer; JSON now intentionally follows the File > Save implementation.

    ``on_format(fmt)`` -- when given -- is called after each format finishes, so
    a caller can advance a progress reading per written file rather than once
    per dataset (a CSV or JSON write is tens of times a NumPy one, see
    :data:`FORMAT_WORK_WEIGHT`). ``on_prepared()`` fires once the shared
    canonical dataset exists, before the first format is written -- that step is
    a flat ~0.37 s and would otherwise make the first tick arrive late.
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
    if ({"msr", "zarr"} & set(normalized_formats)) and not has_mfx:
        raise ValueError(
            "The .msr and MINFLUX Viewer .zarr exports require mfx localizations; "
            "MBM-only output is ambiguous."
        )

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
        ds = canonical_dataset(
            mfx_arr,
            name=f"{base_name}.msr",
            folder=str(out_root),
            mbm=mbm_arr if has_mbm else None,
            mbm_meta=mbm_meta,
            source_zarr=source_zarr,
        )
    if on_prepared is not None:
        on_prepared()

    written: list[str] = []

    def _done(fmt: str) -> None:
        if on_format is not None:
            on_format(fmt)

    for fmt in normalized_formats:
        if fmt == "zarr":
            paths = save_processed(
                ds,
                data_path=out_root / str(base_name),
                fmt="zarr",
                content="raw",
                include={"attrs": True, "derived": True, "recipe": True},
            )
            written.extend(str(path) for path in paths)
            log(f"[zarr] wrote {paths[0]} (self-contained MINFLUX Viewer store)")
            _done(fmt)
            continue
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
            _done(fmt)
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

        _done(fmt)

    return written
