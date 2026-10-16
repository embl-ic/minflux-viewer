'Session export'

from __future__ import annotations

import os
import tempfile
from pathlib import Path

import numpy as np

_FILENAME_FORBIDDEN = set('<>:"/\\|?*') | {chr(c) for c in range(32)}


_RESERVED_STEMS = {"CON", "PRN", "AUX", "NUL"} | {
    f"{p}{i}" for p in ("COM", "LPT") for i in range(1, 10)}


def safe_export_stem(value: str) -> str:
    """Sanitize a dataset/series label into a file stem."""
    safe = "".join(ch if ch.isalnum() or ch in "-_." else "_" for ch in str(value))
    return safe.strip("._") or "export"


def image_export_stem(name: str) -> str:
    """File stem for an exported image series — its ``.msr`` name, verbatim.

    Imspector names image stacks ``Ch1 {1}`` / ``MF(<run>)/density/loc``, and an
    image is only findable again if the export keeps that name, so only
    genuinely illegal characters are replaced (a run of them collapses to one
    ``_``).  Unlike :func:`safe_export_stem` this preserves spaces, braces and
    parentheses.
    """
    out: list[str] = []
    for ch in str(name):
        if ch in _FILENAME_FORBIDDEN:
            if out and out[-1] == "_":
                continue
            out.append("_")
        else:
            out.append(ch)
    # Windows silently strips trailing dots/spaces, which would break a later
    # exact-name lookup, so they go here instead.
    stem = "".join(out).strip().strip(".").strip() or "image"
    if stem.upper() in _RESERVED_STEMS:
        stem = f"{stem}_"
    return stem


def unique_export_stem(key: str, used: set[str], log=None) -> str:
    """A file stem for *key* that no earlier dataset already took.

    Two datasets can want the same stem either because the source labels collide
    after sanitization, or because the file genuinely lists the same label twice
    with different data.  Both used to raise, which aborted the export part-way
    and dropped every dataset after the clash — including, in a real file, its
    largest one.  Disambiguating with a numeric suffix keeps all of the data.
    """
    base = safe_export_stem(key)
    if base not in used:
        used.add(base)
        return base
    index = 2
    while f"{base}_{index}" in used:
        index += 1
    stem = f"{base}_{index}"
    used.add(stem)
    if log is not None:
        log(f"[export] '{key}' would reuse the file name '{base}'; "
            f"writing it as '{stem}' instead.")
    return stem


def prepare_export_output_dir(value: str | os.PathLike[str]) -> Path:
    """Validate and prepare the MSR export destination entered by the user.

    The dialog's output-folder field is intentionally absolute: a relative
    value such as ``=====`` otherwise becomes a silently-created directory
    below whatever directory launched the application.  Missing directories
    are created when their parent is usable, then a temporary file proves that
    the destination is actually writable before export starts.
    """
    raw = os.path.expandvars(os.path.expanduser(str(value).strip()))
    if not raw:
        raise ValueError("Please set an output folder.")
    candidate = Path(raw)
    if not candidate.is_absolute():
        raise ValueError(
            f"Output folder must be an absolute path; got {str(value)!r}."
        )
    try:
        output = candidate.resolve(strict=False)
    except OSError as exc:
        raise ValueError(f"Output folder is not available: {candidate}\n{exc}") from exc
    if output.exists() and not output.is_dir():
        raise ValueError(f"Output path is not a folder: {output}")
    try:
        output.mkdir(parents=True, exist_ok=True)
    except OSError as exc:
        raise ValueError(f"Output folder is not available: {candidate}\n{exc}") from exc
    if not output.is_dir():
        raise ValueError(f"Output path is not a folder: {output}")
    try:
        with tempfile.NamedTemporaryFile(
            prefix=".minflux-viewer-write-test-", dir=output, delete=True
        ):
            pass
    except OSError as exc:
        raise ValueError(f"Output folder is not writable: {output}\n{exc}") from exc
    return output


def subset_struct_fields(arr, names_sel):
    """Structured *arr* restricted to *names_sel* (``None`` = keep every field)."""
    if arr is None:
        return None
    names = getattr(getattr(arr, "dtype", None), "names", None)
    if names and names_sel is not None:
        keep = [n for n in names if n in names_sel]
        if not keep:
            return None
        # A failed restriction must never become an unrestricted export.
        return arr[keep]
    return arr


def image_series_of(parsed: dict, log=None) -> list[dict]:
    """Viewable OBF image series of a parsed ``.msr`` (empty when it has none)."""
    msr_path = (parsed or {}).get("msr")
    if not msr_path:
        return []
    try:
        from minflux_viewer.core.obf_image_source import list_obf_image_series
        return list_obf_image_series(msr_path)
    except Exception as exc:                                   # noqa: BLE001
        if log is not None:
            log(f"[image] could not list image series: {exc}")
        return []


def select_image_series(series: list[dict], *, names=None, indices=None) -> list[dict]:
    """The series to export: by *names* when given, else by raw stack *indices*.

    Batch export matches on **name** because raw stack indices address one
    file's stack table and mean nothing in another.  ``indices=None`` (with no
    *names*) means **every** series — the "include all images" selection, which
    neither indices nor names can express across files whose stack tables and
    channel labels differ.
    """
    if not series:
        return []
    if names is not None:
        return [s for s in series if s.get("name") in names]
    if indices is None:
        return list(series)
    wanted = {int(i) for i in indices}
    return [s for s in series if int(s.get("raw_index", -1)) in wanted]


def export_image_series(out_dir: str, msr_path: str, series: list[dict],
                        log=None, used: set[str] | None = None) -> None:
    """Write each selected OBF image series as an OME-TIFF beside the datasets.

    Images go out through the shared writer in ``core/tiff_export.py`` — the
    same path the render view's "Export to TIFF…" uses — so the pixel
    calibration round-trips into the viewer's TIFF reader.  Image data is copied
    through unchanged; the chosen data formats (.mat/.npy/…) apply to the
    localization arrays, not to images.

    *used* is the set of file stems already written for this output folder. The
    caller owns it so a series-at-a-time loop (which is what gives the batch a
    per-image progress tick) still detects duplicate ``.msr`` labels.
    """
    if not msr_path or not series:
        return
    from minflux_viewer.core.obf_image_source import ObfImageSource
    from minflux_viewer.core.tiff_export import export_image_series_to_tiff

    used = set() if used is None else used
    for entry in series:
        raw_index = int(entry.get("raw_index", -1))
        stem = image_export_stem(str(entry.get("name") or f"image_{raw_index}"))
        # Two OBF stacks can carry the same label; keep both addressable.
        if stem in used:
            stem = f"{stem}_{raw_index}"
        used.add(stem)
        target = Path(out_dir) / f"{stem}.tif"
        source = None
        try:
            source = ObfImageSource(msr_path, raw_stack_index=raw_index)
            result = export_image_series_to_tiff(source, target)
            if log is not None:
                log(f"[tif] wrote {target} "
                    f"({result.axes} {'x'.join(str(v) for v in result.shape)}, "
                    f"{result.dtype})")
        except Exception as exc:                               # noqa: BLE001
            if log is not None:
                log(f"[error] image series {raw_index} "
                    f"('{entry.get('name')}'): {exc}")
            raise
        finally:
            if source is not None:
                source.close()


EXPORT_CONFLICT_MODES = ("error", "overwrite", "skip")


def _export_entries(parsed, msr_state):
    """The (display name, entry) pairs one export will write, in order."""
    if (parsed or {}).get("mode") == "modern":
        return [((ds.get("display_name") or ds.get("did") or "dataset"), ds)
                for ds in (parsed or {}).get("datasets") or []]
    keys = (getattr(msr_state, "mfx_map", {}).keys()
            | getattr(msr_state, "mbm_map", {}).keys())
    return [(key, None) for key in sorted(keys)] or [("legacy", None)]


def _entry_row_count(entry, msr_state, key: str) -> int:
    """Rows the mfx array of one export entry holds, for the work budget.

    Resolved the same way the export itself resolves the array (the per-entry
    capture first, the name-keyed map as the fallback), so the budget and the
    writes cannot disagree about which dataset is the large one. Zero when
    nothing is there -- an unresolvable entry simply costs the flat overhead.
    """
    array = (entry or {}).get("_mfx")
    if array is None:
        array = getattr(msr_state, "mfx_map", {}).get(key)
    try:
        return int(np.asarray(array).size) if array is not None else 0
    except (TypeError, ValueError):                                       # noqa: BLE001
        return 0


def existing_export_targets(parsed, msr_state, out_dir: str, formats: list[str], *,
                            field_selection=None) -> list[Path]:
    """Planned outputs of this export that already exist on disk.

    Used to ask before writing. The writers refuse to clobber regardless, so
    this only decides whether the question is worth asking -- it never decides
    whether the data is safe.
    """
    from minflux_viewer.msr.export import _planned_paths

    out = Path(out_dir)
    formats_lower = {str(fmt).lower().lstrip(".") for fmt in formats}
    other = [fmt for fmt in formats if str(fmt).lower().lstrip(".") != "zarr"]
    selection = field_selection or {}
    found: list[Path] = []

    if "zarr" in formats_lower:
        stem = safe_export_stem(Path(str((parsed or {}).get("msr") or "acquisition")).stem)
        target = out / f"{stem}.zarr"
        if target.exists():
            found.append(target)

    if other:
        used: set[str] = set()
        for key, entry in _export_entries(parsed, msr_state):
            if not selection.get(key, {"checked": True}).get("checked", True):
                continue
            has_mfx = ((entry or {}).get("_mfx") is not None
                       or getattr(msr_state, "mfx_map", {}).get(key) is not None)
            has_mbm = ((entry or {}).get("_mbm") is not None
                       or getattr(msr_state, "mbm_map", {}).get(key) is not None)
            stem = unique_export_stem(key, used, lambda _m: None)
            found.extend(path for path in _planned_paths(
                out, stem, other, has_mfx=has_mfx, has_mbm=has_mbm)
                if path.exists())
    return found


def export_parsed_result(parsed, msr_state, out_dir: str, formats: list[str], *,
                         mfx_sel_global=None, mbm_sel_global=None,
                         image_names=None, image_indices=None,
                         field_selection=None, log=None, progress=None,
                         on_conflict: str = "error") -> None:
    """Write every selected dataset and image series of one parsed ``.msr``.

    Free of Qt and of dialog state so the batch runner can call it from a
    worker thread; the dialog's own export is a thin wrapper over it.

    ``progress(done, total, label)`` — when given — is called after every
    written file. ``done``/``total`` are **work units**, not counts: writing one
    dataset as CSV costs ~25× writing it as NumPy and ~50 M rows of JSON is most
    of a run, so counting one unit per dataset left the bar at a single value
    for nearly the whole export (see ``msr/export.py::FORMAT_WORK_WEIGHT``).
    ``label`` names what has just been written, for the progress line.
    """
    os.makedirs(out_dir, exist_ok=True)
    _log = log if log is not None else (lambda _m: None)

    from minflux_viewer.msr.export import (
        IMAGE_WORK_WEIGHT,
        canonical_dataset,
        dataset_prepare_weight,
        dataset_work_weight,
        export_arrays,
        format_work_weight,
    )
    from minflux_viewer.msr.export import WORK_ROWS_UNIT as _WORK_ROWS_UNIT

    # Pair each dataset with its own arrays.  A .msr can carry two entries under
    # one display name (same did, different data — seen in the wild with a large
    # localization store beside a small companion), and ``mfx_map`` is keyed by
    # name, so the map alone silently keeps only one of them.  The per-entry
    # ``_mfx``/``_mbm`` captured at parse time are authoritative; the map is the
    # fallback for the legacy path.
    if str(on_conflict) not in EXPORT_CONFLICT_MODES:
        raise ValueError(f"Unsupported export conflict mode: {on_conflict!r}")
    overwrite_existing = on_conflict == "overwrite"
    skip_existing = on_conflict == "skip"
    entries = _export_entries(parsed, msr_state)

    images = select_image_series(image_series_of(parsed, _log),
                                 names=image_names, indices=image_indices)

    used_names: set[str] = set()
    zarr_requested = "zarr" in {str(fmt).lower().lstrip(".") for fmt in formats}
    other_formats = [fmt for fmt in formats if str(fmt).lower().lstrip(".") != "zarr"]

    # Budget the run in work units before writing anything, so the reading is a
    # share of the real cost instead of a share of the dataset count.
    rows_by_key = {key: _entry_row_count(entry, msr_state, key)
                   for key, entry in entries}
    zarr_weight = {
        key: (format_work_weight("zarr") * max(rows, 0) / _WORK_ROWS_UNIT
              if zarr_requested else 0.0)
        for key, rows in rows_by_key.items()
    }
    total = float(sum(dataset_work_weight(rows, other_formats)
                      for rows in rows_by_key.values()))
    total += float(sum(zarr_weight.values()))
    total += IMAGE_WORK_WEIGHT * len(images)
    total = max(total, 1e-9)
    done = 0.0

    def _advance(weight: float, label: str = "") -> None:
        nonlocal done
        done = min(done + float(weight), total)
        if progress is not None:
            progress(done, total, label)

    def _tick_dataset(key: str) -> None:
        """The whole remaining budget of one dataset (skipped or finished)."""
        _advance(dataset_work_weight(rows_by_key.get(key, 0), other_formats), key)

    zarr_members = []
    selection = field_selection or {}
    for key, entry in entries:
        mfx_arr = (entry or {}).get("_mfx")
        mbm_arr = (entry or {}).get("_mbm")
        if mfx_arr is None:
            mfx_arr = getattr(msr_state, "mfx_map", {}).get(key)
        if mbm_arr is None:
            mbm_arr = getattr(msr_state, "mbm_map", {}).get(key)
        if mfx_sel_global is not None or mbm_sel_global is not None:
            mfx_sel, mbm_sel = mfx_sel_global, mbm_sel_global
        else:
            sel = selection.get(key, {"checked": True, "mfx": None, "mbm": None})
            if not sel.get("checked", True):
                _log(f"[export] skip dataset '{key}' (unchecked).")
                _tick_dataset(key)
                continue
            mfx_sel = sel.get("mfx")
            mbm_sel = sel.get("mbm")
        mfx_sub = subset_struct_fields(mfx_arr, mfx_sel)
        mbm_sub = subset_struct_fields(mbm_arr, mbm_sel)
        if mfx_sub is None and mbm_sub is None:
            _log(f"[export] skip dataset '{key}' (no selected data).")
            _tick_dataset(key)
            continue

        stem = unique_export_stem(key, used_names, _log)
        kept = (list(getattr(getattr(mfx_sub, "dtype", None), "names", []) or [])
                if mfx_sel is not None else "all")
        _log(f"[export] '{key}': mfx fields = "
             f"{', '.join(kept) if isinstance(kept, list) else kept}")
        export_kwargs = {
            "log": _log,
            "mbm_meta": getattr(msr_state, "mbm_meta_map", {}).get(key),
            "overwrite": overwrite_existing,
        }
        source_zarr = (entry or {}).get("zroot")
        if zarr_requested:
            if mfx_sub is None:
                raise ValueError(
                    f"Dataset '{key}' has no selected MFX data for the Zarr project."
                )
            member = canonical_dataset(
                mfx_sub,
                name=key,
                folder=str(out_dir),
                mbm=mbm_sub,
                mbm_meta=export_kwargs["mbm_meta"],
                source_zarr=source_zarr,
            )
            member.metadata["msr_source_path"] = str(parsed.get("msr") or "")
            member.metadata["msr_dataset_key"] = key
            member.metadata["msr_dataset_name"] = key
            member.metadata["msr_dataset_did"] = str((entry or {}).get("did") or "")
            zarr_members.append(member)
        if other_formats:
            if skip_existing:
                from minflux_viewer.msr.export import _planned_paths

                planned = _planned_paths(
                    Path(out_dir), stem, other_formats,
                    has_mfx=mfx_sub is not None, has_mbm=mbm_sub is not None)
                if planned and all(path.exists() for path in planned):
                    _log(f"[export] skip '{key}': outputs already exist.")
                    _tick_dataset(key)
                    continue
            rows = rows_by_key.get(key, 0)
            export_arrays(
                out_dir, stem, other_formats, mfx_sub, mbm_sub,
                on_prepared=lambda _key=key, _rows=rows: _advance(
                    dataset_prepare_weight(_rows), _key),
                on_format=lambda fmt, _key=key, _rows=rows: _advance(
                    format_work_weight(fmt) * max(_rows, 0) / _WORK_ROWS_UNIT,
                    f"{_key} · {fmt}"),
                **export_kwargs)
        else:
            _tick_dataset(key)

    if zarr_requested and zarr_members:
        from minflux_viewer.core.minflux_zarr import write_minflux_zarr, write_minflux_zarr_project
        from minflux_viewer.core.overlay import overlay_color_cycle

        if len(zarr_members) > 1:
            group_id = f"msr-export:{Path(str(parsed.get('msr') or '')).resolve()}"
            colors = overlay_color_cycle(None)
            reference = zarr_members[0].name
            for position, member in enumerate(zarr_members):
                transform = {
                    "method": "unregistered identity (MSR export)",
                    "reference_channel": reference,
                    "moving_channel": member.name,
                    "matrix_4x4": [
                        [1.0, 0.0, 0.0, 0.0],
                        [0.0, 1.0, 0.0, 0.0],
                        [0.0, 0.0, 1.0, 0.0],
                        [0.0, 0.0, 0.0, 1.0],
                    ],
                    "matrix_3x3": [
                        [1.0, 0.0, 0.0],
                        [0.0, 1.0, 0.0],
                        [0.0, 0.0, 1.0],
                    ],
                    "translation_nm": [0.0, 0.0],
                    "z_translation_nm": 0.0,
                    "matched_bead_count": 0,
                }
                member.state.update({
                    "overlay_id": group_id,
                    "render_group_id": group_id,
                    "overlay_index": 1,
                    "overlay_order": position + 1,
                    "overlay_lut": colors[position % len(colors)],
                    "render_channel_lut": colors[position % len(colors)],
                    "overlay_transform": transform,
                    "render_transform_2d": transform,
                })
                member.metadata["overlay_id"] = group_id
                member.metadata["overlay_transform"] = transform
                member.metadata["render_transform_2d"] = transform
        source_path = Path(str(parsed.get("msr") or "MINFLUX_acquisition.msr"))
        target = Path(out_dir) / f"{safe_export_stem(source_path.stem)}.zarr"
        image_specs = [{**entry, "msr_path": str(source_path)} for entry in images]
        if skip_existing and target.exists():
            _log(f"[zarr] skip {target}: already exists.")
        elif len(zarr_members) > 1:
            write_minflux_zarr_project(
                zarr_members, target, overwrite=overwrite_existing,
                image_specs=image_specs, name=source_path.stem,
            )
        else:
            write_minflux_zarr(
                zarr_members[0], target, overwrite=overwrite_existing,
                image_specs=image_specs,
            )
        _log(
            f"[zarr] wrote {target} ({len(zarr_members)} dataset(s), "
            f"{len(images)} embedded image(s))"
        )
        # One store carried every member and every embedded image, so their
        # whole remaining budget is spent here.
        _advance(sum(zarr_weight.values()) + IMAGE_WORK_WEIGHT * len(images),
                 target.name)
    elif images:
        # ASCII only: the dialog's log() falls back to print(), which raises on
        # a cp1252 console.
        _log(f"[export] {len(images)} image series -> OME-TIFF")
        used_image_stems: set[str] = set()
        for entry in images:
            export_image_series(out_dir, parsed.get("msr"), [entry], _log,
                                used=used_image_stems)
            _advance(IMAGE_WORK_WEIGHT, str((entry or {}).get("name") or "image"))

    # Anything the branches above did not reach (a Zarr export whose members all
    # dropped out, for instance) still has to leave the reading at 100 %.
    if progress is not None and done < total - 1e-9:
        _advance(total - done)
