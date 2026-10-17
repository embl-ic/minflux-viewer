"""Detached single-dataset I/O; no windows, registration or implicit processing."""

from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING

import numpy as np

from . import formats
from .file_transactions import destination_locks, file_transaction

if TYPE_CHECKING:
    from .dataset import MinfluxDataset

IO_FORMATS = frozenset({"npy", "mat", "json", "csv", "zarr", "zarr_zip"})


def load_dataset(path: str | Path, *, apply_sidecar: bool = False) -> MinfluxDataset:
    """Load one dataset, never silently choose one member of an acquisition.

    Optional sibling recipes are opt-in; embedded processing is intrinsic to
    application Zarr. No automatic density/precision/anisotropy is performed.
    Generic spreadsheets needing interactive mapping and MSR are not supported.
    """
    from . import loader_format_readers as readers

    source = Path(path).resolve(strict=True)
    if source.name.lower().endswith((".zarr", ".zarr.zip")):
        from .minflux_zarr import load_minflux_zarr

        # This reader rejects project markers instead of loading all members
        # merely to discover that this is not a single-dataset request.
        dataset = load_minflux_zarr(source)
        _require_filter_inputs(dataset)
        return dataset
    loaders = {".npy": readers.load_npy, ".npz": readers.load_npz,
               ".mat": readers.load_dataset, ".json": readers.load_json,
               ".csv": readers.load_csv, ".tsv": readers.load_csv, ".txt": readers.load_csv}
    reader = loaders.get(source.suffix.lower())
    if reader is None or not source.is_file():
        raise ValueError("Single-dataset I/O supports NPY, NPZ, MAT, JSON, delimited tables and dataset Zarr; "
                         "MSR and multi-member projects require explicit acquisition selection.")
    dataset = reader(source, apply_sidecar=False)
    if apply_sidecar:
        import json

        from .loader_metadata_recipe import apply_metadata_recipe
        from .save import is_metadata_json_payload, metadata_sidecar_path

        sidecar = metadata_sidecar_path(source)
        if sidecar.exists():
            # An explicit request is not speculative format detection: a bad
            # or unreadable recipe must not be silently treated as absent.
            recipe = json.loads(sidecar.read_text(encoding="utf-8"))
            if not is_metadata_json_payload(recipe):
                raise ValueError(f"'{sidecar.name}' is not a metadata sidecar")
            calibration = recipe.get("calibration") or {}
            if "z_scaling_factor" in calibration:
                factor = float(calibration["z_scaling_factor"])
                if not np.isfinite(factor) or factor <= 0:
                    raise ValueError("Recipe Z scaling factor must be positive and finite")
            applied = apply_metadata_recipe(dataset, recipe)
            warnings = [message for message in applied if message.startswith("[!]")]
            if warnings:
                raise ValueError("Recipe could not be fully restored: " + "; ".join(warnings))
    _require_filter_inputs(dataset)
    return dataset


def _require_filter_inputs(dataset: MinfluxDataset) -> None:
    """No GUI post-load chain exists here to complete a partial selection later."""
    from .loader_raw_access import attr_values_for_selection, resolve_spec_iteration

    for spec in dataset.state.get("filter_specs") or ():
        if not isinstance(spec, dict):
            raise ValueError("Saved filter specifications must be dictionaries")
        if not spec.get("enabled", True):
            continue
        name = str(spec.get("attribute", ""))
        values = attr_values_for_selection(dataset, name, itr=resolve_spec_iteration(dataset, spec))
        if values is None or np.asarray(values).shape != (dataset.prop.num_loc,):
            raise ValueError(f"Saved filter '{name}' is not materialized. CPU I/O performs no "
                             "automatic computations; load without the sidecar or prepare the "
                             "required attributes explicitly before applying this recipe.")


def validate_save_options(fmt: str, content: str, filter_mode: str) -> None:
    """Reject an invalid request before callbacks, file reads or allocations."""
    if fmt not in IO_FORMATS:
        raise ValueError(f"Unsupported single-dataset output format: {fmt!r}")
    if content not in {"raw", "snapshot"} or filter_mode not in {"flag", "apply"}:
        raise ValueError("content must be raw/snapshot and filter_mode flag/apply")
    if content == "snapshot" and formats.spec_for(fmt).raw_only:
        raise ValueError("Application Zarr stores raw data plus processing, not baked snapshots")


def save_dataset(dataset: MinfluxDataset, path: str | Path, *, fmt: str = "npy",
                 content: str = "raw", filter_mode: str = "flag") -> list[Path]:
    """Write a detached dataset and its recipe, refusing existing outputs.

    Caller owns the dataset for this synchronous operation. GUI callers must
    capture a snapshot first; a worker must not pass a live viewer dataset.
    No external images or overlay siblings are implicitly collected.
    """
    from .save import metadata_sidecar_path, save_processed

    validate_save_options(fmt, content, filter_mode)
    target = formats.normalize_path(fmt, path).absolute()
    include = {"attrs": True, "derived": True, "recipe": True}
    if fmt in {"zarr", "zarr_zip"}:
        with destination_locks([target]):
            if target.exists() or target.is_symlink():
                raise FileExistsError(target)
            return save_processed(dataset, data_path=target, zarr_overwrite="error", fmt=fmt,
                                  content=content, filter_mode=filter_mode, include=include)
    sidecar = metadata_sidecar_path(target)
    with file_transaction([target, sidecar], overwrite=False) as staged:
        save_processed(dataset, data_path=staged[target], fmt=fmt, content=content,
                       filter_mode=filter_mode, include=include)
    return [target, sidecar]
