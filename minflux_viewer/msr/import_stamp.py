"""The metadata a dataset gets when it is imported from a ``.msr``.

Extracted so the two import routes cannot drift: the MSR reader dialog (which
also offers channel alignment, confocal mapping and field selection) and the
direct open of a ``.msr`` this application wrote, which needs none of those
because the choices they make are already recorded in the file.

Qt-free, so it can run on either path.
"""

from __future__ import annotations

from pathlib import Path

__all__ = ["stamp_msr_dataset"]


def stamp_msr_dataset(ds, *, entry: dict, msr_path, key: str,
                      mbm_points=None, mbm_points_by_gri=None, mbm_used=None,
                      log=None) -> None:
    """Record where *ds* came from, and the small native payloads that travel.

    ``entry`` is one item of ``parse_general``'s ``datasets`` list. Everything
    here is provenance or a small array; the bulky source store is deliberately
    not retained (it is hundreds of MB).
    """
    msr_path = Path(msr_path)
    ds.metadata["msr_source_path"] = str(msr_path)
    ds.metadata["msr_dataset_key"] = key
    ds.metadata["msr_dataset_name"] = key
    ds.metadata["msr_dataset_did"] = str(entry.get("did") or "")

    source_format = entry.get("source_format")
    if source_format:
        ds.metadata["source_format"] = source_format
        ds.metadata["is_minflux"] = True
        ds.metadata["has_real_tid"] = True
        if entry.get("mfxdta_version") is not None:
            ds.metadata["mfxdta_version"] = entry.get("mfxdta_version")

    zroot = entry.get("zroot")
    if zroot is not None:
        try:
            from ..core.acquisition_time import (
                acquisition_date_from_zattrs, stamp_dataset_acquisition,
            )
            from ..core.mfx_sequence import extract_sequence_from_zattrs
            from ..core.minflux_zarr import capture_native_zarr_metadata
            from .io import read_zarr_attrs

            # The small native attrs + search grid the self-contained Zarr
            # writer needs. Never the full in-memory source store.
            capture_native_zarr_metadata(ds, zroot)
            mfx_attrs = read_zarr_attrs(zroot, "mfx")
            sequence = extract_sequence_from_zattrs(mfx_attrs)
            if sequence:
                ds.metadata["mfx_sequence"] = sequence
            # The instrument's own timestamp (m2410 'acquisition_date' or the
            # m2205 'tms' epoch), NOT the MFXDTA container timestamp, which is
            # when the file was saved.
            stamp_dataset_acquisition(ds, acquisition_date_from_zattrs(mfx_attrs))
        except Exception as exc:                            # noqa: BLE001
            if log is not None:
                log(f"[warn] '{key}': native metadata could not be read: {exc}")

    if mbm_points is not None:
        from ..core.dataset import AttributeComponent

        ds.mbm = AttributeComponent({"points": mbm_points})
        ds.metadata["mbm_points"] = mbm_points
        # The bead NAMING/selection too, not just the points: without it a
        # loaded dataset falls back to bare gri ids, losing the R-IDs and which
        # beads were actually used for drift correction.
        if mbm_points_by_gri:
            ds.metadata["mbm_points_by_gri"] = mbm_points_by_gri
        if mbm_used:
            ds.metadata["mbm_used"] = mbm_used
