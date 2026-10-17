"""Build and stamp a viewer dataset from one parsed MSR job, without Qt."""
from __future__ import annotations

import json
from datetime import datetime

from .session_export import subset_struct_fields


def build_viewer_dataset(job, *, msr_path, prefs, logs, progress,
                          overlay, n_datasets_total):
    """Build one dataset from its parsed ``mfx`` array, fully stamped.

        Runs on a worker thread, so it must not touch a widget or this
        dialog's state: everything it needs is in *job*, and messages go to
        *logs* (delivered to the Log on the GUI thread). Returns ``None`` when
        the dataset has nothing importable; raises on a genuine failure.
        """
    from minflux_viewer.core.loader_format_readers import load_from_mfx_array
    from minflux_viewer.msr.import_stamp import stamp_msr_dataset
    key = job["key"]
    mfx = job["mfx"]
    if mfx is None:
        raise ValueError("mfx array not in memory — skipped.")
    field_sel = job["field_sel"]
    if field_sel is not None:
        names = set(getattr(getattr(mfx, "dtype", None), "names", []) or [])
        if "loc" in names and "loc" not in field_sel:
            field_sel = set(field_sel)
            field_sel.add("loc")
            logs.append(f"[viewer] added required 'loc' field for '{key}' so it can render.")
        mfx_to_load = subset_struct_fields(mfx, field_sel)
        if mfx_to_load is None:
            raise ValueError("selected fields contain no importable mfx data — skipped.")
    else:
        mfx_to_load = mfx
    display_name = job["display_name"]
    dataset = load_from_mfx_array(
        mfx=mfx_to_load,
        name=display_name,
        folder=str(msr_path.parent),
        datetime_str=datetime.now().strftime("%Y-%b-%d, %H:%M:%S"),
        recent_path=str(msr_path),
        prefs=prefs,
        progress=lambda f: progress(0.95 * f),
    )
    if overlay is not None:
        dataset.state["overlay_id"] = overlay["id"]
        dataset.state["render_group_id"] = overlay["id"]
        dataset.state["overlay_index"] = overlay["index"]
        dataset.state["overlay_order"] = overlay["order"]
        dataset.state["overlay_lut"] = overlay["lut"]
        dataset.state["render_channel_lut"] = overlay["lut"]
    # Provenance + the small native payloads that travel with a
    # dataset. Shared with the direct open of a .msr we wrote
    # (msr/quick_open.py), so the two import routes cannot drift.
    stamp_msr_dataset(
        dataset, entry=job["entry"], msr_path=msr_path, key=key,
        mbm_points=job["mbm_points"],
        mbm_points_by_gri=job["mbm_points_by_gri"], mbm_used=job["mbm_used"],
        log=logs.append)
    if overlay is not None:
        dataset.metadata["overlay_id"] = overlay["id"]
        dataset.metadata["overlay_index"] = overlay["index"]
    transform = job["transform"]
    if transform is not None:
        dataset.state["overlay_transform"] = transform
        dataset.state["render_transform_2d"] = transform
        dataset.metadata["overlay_transform"] = transform
        dataset.metadata["render_transform_2d"] = transform
        dataset.metadata["transformed"] = bool(transform.get("moving_channel") != transform.get("reference_channel"))
    # Processing state embedded in the file itself, by our own
    # .msr writer. Applied LAST of the in-file sources so a saved
    # overlay id, LUT or transform wins over the defaults the
    # import invents, and before add_dataset so a restored Z
    # scaling factor is in place before the post-load chain runs
    # its auto-estimate.
    try:
        from minflux_viewer.msr.viewer_state import apply_viewer_state, read_viewer_state
        _vs = read_viewer_state(job["entry"].get("zroot"))
        if _vs:
            _done = apply_viewer_state(dataset, _vs)
            if _done:
                logs.append(f"[viewer] restored saved state for "
                                f"'{display_name}': {', '.join(_done)}")
    except (OSError, AttributeError, IndexError, KeyError, TypeError, ValueError) as _exc:
        logs.append(f"[warn] embedded viewer state in "
                        f"'{msr_path.name}' could not be applied: {_exc}")
    # A recipe sidecar saved beside this .msr (File > Save As >
    # .msr writes one) was previously written and never read:
    # `apply_metadata_sidecar` is wired into the load_* functions,
    # and .msr opens through this plugin instead, so Z scaling factor, filters
    # and the channel transform silently evaporated on reopen.
    # Applied before add_dataset so the Z scaling factor pin is in place before
    # the post-load chain runs its auto-estimate.
    try:
        from minflux_viewer.core.loader_metadata_recipe import apply_metadata_recipe
        from minflux_viewer.core.save import is_metadata_json_payload, metadata_sidecar_path
        _side = metadata_sidecar_path(msr_path)
        if _side.is_file():
            _recipe = json.loads(_side.read_text(encoding="utf-8"))
            # One sidecar sits beside a file that may hold several
            # channels, so only apply it to the dataset it names —
            # or when this import produced a single dataset.
            _named = str(_recipe.get("name") or "")
            if is_metadata_json_payload(_recipe) and (
                not _named or _named == display_name
                or n_datasets_total == 1
            ):
                _applied = apply_metadata_recipe(dataset, _recipe)
                if _applied:
                    logs.append(f"[viewer] applied '{_side.name}' to "
                                    f"'{display_name}': {', '.join(_applied)}")
    except (OSError, AttributeError, IndexError, KeyError, TypeError, ValueError) as _exc:
        logs.append(f"[warn] recipe sidecar beside '{msr_path.name}' "
                        f"could not be applied: {_exc}")
    progress(1.0)
    return dataset
