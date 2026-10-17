'Loader metadata recipe'

from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from minflux_viewer.core.dataset import (
    MinfluxDataset,
)
from minflux_viewer.core.loader_raw_access import apply_saved_filters


def _restore_filter_mask(ds: MinfluxDataset, saved_ftr) -> bool:
    """Apply a saved ``ftr`` column as ``ds.filter_mask``.

    A processed snapshot saved with ``filter_mode="flag"`` keeps every row and
    records the filter as an ``ftr`` boolean column. It must be captured from the
    source columns *before* ``_add_derived_attributes`` runs, because that resets
    ``attrs["ftr"]`` to all-True — which silently turned a saved filter back into
    "nothing filtered" on reload.
    """
    if saved_ftr is None:
        return False
    mask = np.asarray(saved_ftr)
    if mask.shape != (ds.prop.num_loc,):
        raise ValueError("Saved filter mask must align with materialized localizations")
    ds.filter_mask = mask.astype(bool, copy=False)
    return True


def _transform_record_from_recipe(tf) -> dict | None:
    """Coerce a recipe ``transform`` payload into a display-transform record.

    Accepts what :func:`core.save.build_metadata` writes (a
    ``display_transform_record`` dict) and, for sidecars written before that was
    settled, a bare 4x4 or 3x3 matrix. Returns ``None`` when the payload cannot be
    interpreted, so the caller can report it rather than dropping the alignment.
    """
    from minflux_viewer.core.overlay import matrix4_to_xy3, transform_to_matrix4

    if isinstance(tf, dict):
        if transform_to_matrix4(tf) is None:
            return None
        record = dict(tf)
        if "matrix_4x4" not in record:
            matrix = transform_to_matrix4(tf)
            record["matrix_4x4"] = np.asarray(matrix, dtype=float).tolist()
        return record

    matrix = transform_to_matrix4(tf)          # bare 4x4 / 3x3 from an old sidecar
    if matrix is None:
        return None
    matrix = np.asarray(matrix, dtype=float)
    return {
        "matrix_4x4": matrix.tolist(),
        "matrix_3x3": np.asarray(matrix4_to_xy3(matrix), dtype=float).tolist(),
        "alignment_mode": "restored (metadata sidecar)",
        "provenance": {"source": "metadata recipe", "original_type": type(tf).__name__},
    }


def apply_metadata_recipe(ds: MinfluxDataset, meta: dict) -> list[str]:
    """Apply an already-parsed metadata-sidecar payload to *ds*.

    Restores Z scaling factor (pinned via ``ds.derived['z_scaling_factor']`` so post-load auto-estimation
    is suppressed), the overlay transform, and the filter specs (applying what is
    evaluable now).  Returns user-facing phrases for what was applied, so a
    caller can report an empty recipe rather than claiming success.
    """
    applied: list[str] = []

    # Identity, restored silently (it is provenance, not processing, so it is
    # not reported as something "applied"). Carrying it onto the dataset is what
    # lets a *later* sidecar find this dataset by DID: the data file itself
    # cannot hold one. Never overwritten -- a dataset that already knows its own
    # DID outranks a recipe that may have come from a different copy.
    did = str(meta.get("msr_dataset_did") or "")
    if did and not ds.metadata.get("msr_dataset_did"):
        ds.metadata["msr_dataset_did"] = did

    # Overlay membership, recorded silently: it says this file is one of a set,
    # which is provenance until something acts on it. Regrouping is a change to
    # what the user sees, so the UI asks first -- applying it here would rebuild
    # an overlay behind their back on every load.
    from minflux_viewer.core.overlay_save import DATASET_OVERLAY_KEY, overlay_from_metadata

    block = overlay_from_metadata(meta)
    if block:
        ds.metadata[DATASET_OVERLAY_KEY] = block

    # Acquisition provenance: restored before anything else because it is a
    # plain recorded fact — nothing re-derives it, and a data file alone
    # (.csv/.mat/.npy) cannot carry it.
    acq = meta.get("acquisition") or {}
    if acq.get("date"):
        from minflux_viewer.core.acquisition_time import stamp_acquisition

        stamp_acquisition(ds.metadata, acq.get("date"), acq.get("span_s"))
        if ds.metadata.get("acquisition_date"):
            applied.append("acquisition date")

    calibration = meta.get("calibration") or {}
    z_scaling_factor = calibration.get("z_scaling_factor")
    if z_scaling_factor is not None:
        try:
            ds.set_z_scaling_factor(float(z_scaling_factor), source="processed (metadata sidecar)")
            ds.derived["z_scaling_factor"] = np.asarray([float(z_scaling_factor)], dtype=float)
            applied.append(f"Z scaling factor {float(z_scaling_factor):.4g}")
        except (TypeError, ValueError, OverflowError):
            pass
    elif calibration.get("rimf") is not None:
        # A sidecar written before the factor was renamed. The old key is NOT
        # honoured: "RIMF" asserted a refractive-index model this value never
        # came from, so silently adopting it would carry that claim forward.
        # Reported instead of dropped, because the number is real and the user
        # can re-enter it -- an unrestored z scaling silently changes 3-D shape.
        applied.append(
            f"[!] Z scaling factor NOT restored - sidecar predates the rename "
            f"and stores the legacy 'rimf' key (value {calibration['rimf']!r}); "
            f"set it in Dataset Information if it still applies")

    tf = meta.get("transform")
    if tf is not None:
        # The canonical in-app transform is a ``display_transform_record`` *dict*.
        # It must be restored as one: ``apply_display_transform_nm`` reads
        # ``matrix_4x4``/``matrix_3x3`` off a dict and silently returns the points
        # unchanged for anything else, so storing a bare ndarray here loses the
        # channel alignment without any error — a 100 nm displacement on a
        # two-colour dataset. (It also poisons the ``a or b`` fallbacks in
        # render/scatter/roi_crop, which raise on an ndarray.)
        from minflux_viewer.core.overlay import transform_to_matrix4

        record = _transform_record_from_recipe(tf)
        if record is None:
            # Surfaced through the returned list (the caller logs it) rather than
            # swallowed: an unrestored transform silently moves a channel.
            applied.append(
                f"[!] overlay transform NOT restored - unreadable "
                f"{type(tf).__name__} in recipe")
        else:
            ds.state["overlay_transform"] = record
            ds.state["render_transform_2d"] = record
            matrix = transform_to_matrix4(record)
            shift = ""
            if matrix is not None:
                dx, dy, dz = (float(matrix[0, 3]), float(matrix[1, 3]), float(matrix[2, 3]))
                if any(abs(v) > 1e-9 for v in (dx, dy, dz)):
                    shift = f" (translation {dx:+.3g}, {dy:+.3g}, {dz:+.3g} nm)"
            applied.append(f"overlay transform{shift}")

    # Filters and ROIs CONCATENATE onto whatever the dataset already carries,
    # rather than replacing it: a recipe may be deliberately applied on top of a
    # dataset the user has already been working on, and silently discarding
    # their filters would be the worse failure. A repeated apply therefore
    # duplicates rows -- which is plainly visible in the Filter dialog, where it
    # can be undone, unlike work that has vanished.
    filters = meta.get("filters") or []
    if filters:
        existing = list(ds.state.get("filter_specs") or [])
        ds.state["filter_specs"] = existing + list(filters)
        apply_saved_filters(ds)
        applied.append(f"{len(filters)} filter(s)"
                       + (f" (added to {len(existing)} already set)" if existing else ""))

    # ROIs land on the same metadata key the Zarr v2 loader uses, so a caller
    # that repopulates the ROI Manager works identically whether the dataset
    # came from a self-contained store or from raw data plus this sidecar.
    # Records already present *by id* are skipped, which is what makes applying
    # the same sidecar twice idempotent for ROIs (the ids are stable) while
    # still concatenating genuinely different sets.
    rois = [dict(record) for record in (meta.get("rois") or [])
            if isinstance(record, dict)]
    if rois:
        kept = [dict(record)
                for record in (ds.metadata.get("minflux_viewer_roi_records") or [])
                if isinstance(record, dict)]
        seen = {str(record.get("id")) for record in kept if record.get("id")}
        added = [record for record in rois
                 if not record.get("id") or str(record["id"]) not in seen]
        ds.metadata["minflux_viewer_roi_records"] = kept + added
        if added:
            applied.append(f"{len(added)} ROI(s)")
    return applied


def read_metadata_file(path: str | Path) -> dict | None:
    """Parse *path* itself as a metadata sidecar, or ``None`` if it is not one.

    ⚠ Distinct from :func:`read_metadata_sidecar`, which derives a *sibling*
    ``<stem>_viewer_metadata.json`` from a **data** file. Handing this one's
    job to that one looks for ``d_viewer_metadata_viewer_metadata.json`` and
    silently finds nothing.
    """
    from minflux_viewer.core.save import is_metadata_json_payload

    try:
        meta = json.loads(Path(path).read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return None
    return meta if is_metadata_json_payload(meta) else None


def read_metadata_sidecar(data_path: str | Path) -> tuple[Path, dict] | None:
    """The ``<stem>_viewer_metadata.json`` beside *data_path*, and its payload.

    Split out of :func:`apply_metadata_sidecar` so the UI can *find* a recipe
    and describe it to the user **before** deciding whether to apply it: a
    processing recipe changes Z scaling, filters and ROIs, so it must never be
    restored silently. Returns ``None`` when there is no sidecar or it is not
    one (an unreadable file is "not a sidecar", never an error, because this
    runs speculatively for every data file opened).
    """
    from minflux_viewer.core.save import metadata_sidecar_path

    side = metadata_sidecar_path(data_path)
    if not side.is_file():
        return None
    meta = read_metadata_file(side)
    return None if meta is None else (side, meta)


def apply_metadata_sidecar(ds: MinfluxDataset, data_path: str | Path) -> bool:
    """Restore the processing recipe from a ``<stem>_viewer_metadata.json`` file.

    If a metadata sidecar sits next to *data_path*, its recipe is applied via
    :func:`apply_metadata_recipe`.  Returns True if a sidecar was consumed.
    """
    found = read_metadata_sidecar(data_path)
    if found is None:
        return False
    apply_metadata_recipe(ds, found[1])
    return True


def apply_metadata_sidecar_file(ds: MinfluxDataset, sidecar_path: str | Path) -> list[str]:
    """Apply a metadata sidecar addressed **by its own path** to *ds*.

    :func:`apply_metadata_sidecar` finds the sidecar beside a data file; this is
    the entry point for one handed over directly (dropping it on a Dataset
    Manager row).  Raises ``ValueError`` if the file is not a sidecar, so the
    caller can report it instead of silently doing nothing.
    """
    from minflux_viewer.core.save import is_metadata_json_payload

    meta = json.loads(Path(sidecar_path).read_text(encoding="utf-8"))
    if not is_metadata_json_payload(meta):
        raise ValueError(f"'{Path(sidecar_path).name}' is not a metadata sidecar.")
    return apply_metadata_recipe(ds, meta)
