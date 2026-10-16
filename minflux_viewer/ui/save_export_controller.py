"""Save export controller delegated by MainWindow."""

from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING

from PyQt6.QtWidgets import (
    QDialog,
    QDoubleSpinBox,
    QFileDialog,
    QLabel,
    QLineEdit,
    QMessageBox,
    QPushButton,
    QVBoxLayout,
    QWidget,
)

from ..core import formats as _formats
from .background_tasks import BackgroundTask

if TYPE_CHECKING:
    from .main_window import MainWindow


def _save_data(self: MainWindow, *args) -> None:
    """File › Save (Ctrl+S) — save the active dataset as MINFLUX Viewer Zarr v2.

        The format is fixed, so the dialog asks only where. Zarr is the
        application's own self-contained store -- raw canonical data plus
        processing state, ROIs, overlay channels and linked images, no sidecar --
        which is why it needs none of the content/attribute/sidecar choices the
        Save / export dialog carries. Those live in *Save As* and in the Dataset
        Manager's *Save / export data*.
        """
    ds = self._active_dataset_for_save()
    if ds is None:
        return
    from .zarr_save_dialog import ask_zarr_save_path
    chosen = ask_zarr_save_path(
        self, self._zarr_save_suggestion(ds),
        dataset_name=str(getattr(ds, "name", "") or ""),
    )
    if chosen is None:
        return
    self._save_as_format("zarr", "Zarr v2", path=str(chosen))


def _zarr_save_suggestion(self: MainWindow, ds) -> Path:
    """Where File › Save proposes to write *ds*.

        A dataset opened from an application store saves back over it, so
        Ctrl+S on work in progress is a genuine "save", not "save a copy
        somewhere else". Anything else is proposed beside its source file.
        """
    store = str(ds.metadata.get("minflux_viewer_zarr_path")
                or ds.metadata.get("minflux_viewer_project_path") or "")
    if store and Path(store).suffix.lower() == ".zarr":
        return Path(store)
    return self._default_save_path(ds, ".zarr")


def _active_dataset_for_save(self):
    ds = self._state.active_dataset
    if ds is None:
        QMessageBox.information(self, "Save", "No active dataset to save.")
        return None
    return ds


def _start_file_save(self: MainWindow, work, *, name: str, on_done) -> None:
    """Run a prepared writer without blocking Qt's event loop."""
    task = BackgroundTask(
        work, description=f"Saving {name}", category="save"
    )
    task.signals.stage.connect(
        lambda text: self._state.status_progress(text)
    )
    task.signals.done.connect(on_done)
    task.signals.failed.connect(
        lambda message, _n=name: self._on_file_save_failed(_n, message)
    )
    self._begin_file_io(task)


def _on_file_save_failed(self: MainWindow, name: str, message: str) -> None:
    self._state.log(f"Failed to save '{name}': {message}", "ERROR")
    if not self._is_shutting_down:
        QMessageBox.critical(
            self, "Save failed", f"Could not save {name}:\n{message}"
        )
    self._status_label.setText("Save failed.")


def _default_save_path(self: MainWindow, ds, suffix: str) -> Path:
    default_dir = Path(self._state.prefs["file"].get("default_folder", str(Path.home())))
    folder = Path(str(getattr(getattr(ds, "file", None), "folder", "") or default_dir))
    if not folder.exists():
        folder = default_dir
    stem = Path(str(getattr(ds, "name", "") or "dataset")).stem or "dataset"
    return folder / f"{stem}{suffix}"


def _save_as_minflux_formats(self: MainWindow) -> None:
    ds = self._active_dataset_for_save()
    if ds is None:
        return
    suggested = str(self._default_save_path(ds, ".mat"))
    filters = (
        "MATLAB (*.mat);;NumPy (*.npy);;JSON (*.json);;"
            "MINFLUX data formats (*.mat *.npy *.json)"
    )
    path, selected = QFileDialog.getSaveFileName(
        self, "Save MINFLUX data", suggested, filters
    )
    if not path:
        return
    suffix = Path(path).suffix.lower()
    fmt = {".mat": "mat", ".npy": "npy", ".json": "json"}.get(suffix)
    if fmt is None:
        if "NumPy" in selected:
            fmt = "npy"
        elif "JSON" in selected:
            fmt = "json"
        else:
            fmt = "mat"
    self._save_as_format(fmt, "MINFLUX data", path=path)


def _zarr_overwrite_mode(self: MainWindow, path: str | Path) -> str | None:
    """Ask how an existing application Zarr store should be updated."""
    target = Path(path)
    if target.suffix.lower() != ".zarr":
        target = target.with_suffix(".zarr")
    if not target.exists():
        return "replace"
    message = QMessageBox(self)
    message.setIcon(QMessageBox.Icon.Question)
    message.setWindowTitle("Update existing Zarr dataset?")
    message.setText(f"{target}\nalready exists.")
    message.setInformativeText(
        "Update processing only verifies that every canonical raw dataset "
            "is identical, then replaces only MINFLUX Viewer state, derived data, "
            "ROIs, overlay settings, and the project manifest. Embedded images and "
            "raw MFX/MBM/search data are left untouched.\n\n"
            "Replace complete store rewrites everything in the .zarr directory."
    )
    update_button = message.addButton(
        "Update processing only", QMessageBox.ButtonRole.AcceptRole
    )
    replace_button = message.addButton(
        "Replace complete store", QMessageBox.ButtonRole.DestructiveRole
    )
    cancel_button = message.addButton(QMessageBox.StandardButton.Cancel)
    message.setDefaultButton(update_button)
    message.setEscapeButton(cancel_button)
    message.exec()
    clicked = message.clickedButton()
    if clicked is update_button:
        return "viewer"
    if clicked is replace_button:
        return "replace"
    return None


def _save_as_format(self: MainWindow, fmt: str, title: str, *, path: str | None = None) -> None:
    ds = self._active_dataset_for_save()
    if ds is None:
        return
    ext = _formats.extension_for(fmt)
    if path is None:
        suggested = self._default_save_path(ds, ext)
        if fmt == "zarr":
            # A .zarr store is a DIRECTORY; the ordinary save dialog would
            # enter an existing one instead of selecting it. A .zarr.zip is
            # an ordinary file and needs no such treatment.
            from .zarr_save_dialog import choose_zarr_save_path
            selected_path = choose_zarr_save_path(
                self, f"Save {title}", suggested
            )
            if selected_path is None:
                return
            path = str(selected_path)
        else:
            name_filter = _formats.label_for(fmt).replace("(.", "(*.")
            path, _ = QFileDialog.getSaveFileName(
                self, f"Save {title}", str(suggested), name_filter
            )
            if not path:
                return
    from ..core.save import save_processed
    # ROIs travel with EVERY format: the Zarr store keeps them inside, the
    # others in the metadata sidecar. Only the Zarr context (overlay members,
    # linked images) is format-specific.
    zarr_context = (self._zarr_save_context(ds)
                    if fmt in {"zarr", "zarr_zip"}
                    else {"roi_records": self.save_roi_records(ds)})
    # Only the directory store can be updated in place; a sealed package is
    # always rewritten, and QFileDialog already confirmed the overwrite.
    zarr_overwrite = self._zarr_overwrite_mode(path) if fmt == "zarr" else "replace"
    if zarr_overwrite is None:
        return
    # An overlay saved to .msr writes every channel into the one file, and
    # the writer reuses the source measurement's container when it can --
    # which is what makes the result open in Imspector.
    if fmt == "msr":
        zarr_context = dict(zarr_context)
        zarr_context.setdefault("related_datasets",
                                self._overlay_members_for(ds))
    kwargs = dict(
        data_path=path, fmt=fmt, content="raw",
        include={"attrs": True, "derived": False, "recipe": True},
        filter_mode="flag", zarr_overwrite=zarr_overwrite,
        log=self._state.log, **zarr_context,
    )
    action = "Updated processing in " if zarr_overwrite == "viewer" else "Saved "
    name = Path(path).name
    def work(report, _ds=ds, _kwargs=kwargs, _name=name):
        report(f"Writing {_name}")
        written = save_processed(_ds, **_kwargs)
        report(f"Wrote {_name}")
        return written
    self._start_file_save(
        work,
        name=name,
        on_done=lambda written, _a=action, _n=name, _ds=ds: self._on_files_saved(
            written, _a, _n, _ds
        ),
    )


def _on_zarr_saved(self: MainWindow, written, action: str, name: str) -> None:
    self._on_files_saved(written, action, name, None)


def _on_files_saved(self: MainWindow, written, action: str, name: str, ds) -> None:
    if self._is_shutting_down:
        return
    idx = self._post_load_index(ds) if ds is not None else self._state.active_idx
    self._state.log(
        action + ", ".join(str(path) for path in written),
        dataset_idx=idx,
    )
    self._status_label.setText(f"{action.strip()} {name}.")


def save_roi_records(self: MainWindow, ds, member_indices=None, index_to_id=None) -> list[dict]:
    """Every ROI displayed on *ds* right now, as portable dicts.

        The ROI Manager holds many ROIs at once, so a saved dataset carries the
        whole set rather than only the active draft. ``dataset_id`` is stamped
        for a Zarr project (whose members each get their own id); a single-file
        save passes neither argument and gets the active dataset's ROIs.
        """
    from dataclasses import asdict

    from ..core.roi_selection import ROI_MASKS_STATE_KEY
    if member_indices is None:
        idx = self._state.index_of_dataset(ds)
        if idx is None:
            return []
        member_indices = {idx}
        index_to_id = {idx: "d000000"}
    index_to_id = index_to_id or {}
    mask_owner = {
        roi_id: idx
        for idx in member_indices
        for roi_id in dict(
            self._state.datasets[idx].state.get(ROI_MASKS_STATE_KEY) or {})
    }
    candidates = list(self._state.rois.records)
    adapter = self._state.rois.active_adapter
    if adapter is not None:
        # The ROI being drawn is not in the store yet, but it is on screen.
        try:
            draft = adapter.current_record()
        except Exception:                                   # noqa: BLE001
            draft = None
        if draft is not None and all(r.id != draft.id for r in candidates):
            candidates.append(draft)
    from ..core.roi_scope import portable_scope
    records = []
    for record in candidates:
        # Written to a file, so positions, not the in-memory keys.
        context = portable_scope(getattr(record, "context", {}) or {}, self._state)
        owner = context.get("dataset_idx")
        if owner not in member_indices:
            owner = mask_owner.get(record.id)
        if owner not in member_indices:
            continue
        payload = asdict(record)
        if index_to_id:
            payload["dataset_id"] = index_to_id[int(owner)]
        payload["context"] = {
            key: value for key, value in context.items() if key != "dataset_idx"
        }
        records.append(payload)
    return records


def _overlay_members_for(self: MainWindow, ds) -> list:
    """Every channel of *ds*'s overlay, in order, or just *ds*.

        A ``.msr`` holds all channels in one file, so saving one channel of an
        overlay has to write the group -- otherwise re-opening it would silently
        give back a single channel.
        """
    from ..core.overlay import overlay_members
    idx = self._state.index_of_dataset(ds)
    if idx is None:
        return [ds]
    pairs = overlay_members(self._state, idx)
    return [member for _i, member in pairs] if len(pairs) > 1 else [ds]


def _zarr_save_context(self: MainWindow, ds) -> dict:
    """Overlay members, portable ROI geometry and linked MSR images."""
    from ..core.overlay import overlay_members
    anchor_idx = self._state.index_of_dataset(ds)
    if anchor_idx is None:
        return {}
    pairs = overlay_members(self._state, anchor_idx)
    if len(pairs) < 2:
        pairs = [(anchor_idx, ds)]
    member_indices = {idx for idx, _member in pairs}
    index_to_id = {idx: f"d{position:06d}" for position, (idx, _ds) in enumerate(pairs)}
    roi_records = self.save_roi_records(
        ds, member_indices=member_indices, index_to_id=index_to_id)
    image_specs = []
    seen_images: set[tuple[str, int]] = set()
    sources: dict[str, set[str]] = {}
    for _idx, member in pairs:
        source = str(member.metadata.get("msr_source_path") or "")
        did = str(member.metadata.get("msr_dataset_did") or "")
        if source and did and Path(source).is_file():
            sources.setdefault(source, set()).add(did)
    if sources:
        # Every image series of the source .msr travels with the store, not
        # only the DID-linked ones. At save time the dataset can already
        # show them all (View image series falls back to reopening the
        # .msr), so leaving them out made the store depend on that external
        # file: the confocal channels and overviews carry no source_did and
        # were silently dropped. Ones with no matching dataset are written
        # under images/unassigned by _export_embedded_images.
        from ..core.obf_image_source import list_obf_image_series
        for source, _dids in sources.items():
            for entry in list_obf_image_series(source):
                key = (source, int(entry["raw_index"]))
                if key in seen_images:
                    continue
                seen_images.add(key)
                image_specs.append({**entry, "msr_path": source})
    members = [member for _idx, member in pairs]
    return {
        "related_datasets": members,
        "roi_records": roi_records,
        "image_specs": image_specs,
        "project_name": Path(next(iter(sources), ds.name)).stem,
    }


def _save_as_spreadsheet(self: MainWindow) -> None:
    ds = self._active_dataset_for_save()
    if ds is None:
        return
    self.export_custom_csv(ds)


def export_custom_csv(self: MainWindow, ds, suggested: str | Path | None = None) -> None:
    """The custom column table: pick the attributes, name the columns.

        Shared by *File › Save As › Custom table* and the Save / export
        dialog's ``.csv`` entry, so the two cannot drift into different CSVs.
        The table is the current view (filtered, last valid iteration) with
        ``xnm/ynm/znm`` as displayed; the all-iteration canonical table is a
        different writer, reached from that dialog's More options.
        """
    from ..core.save import spreadsheet_export_columns, write_spreadsheet_csv
    from .export_dialogs import CsvExportDialog
    try:
        columns = spreadsheet_export_columns(ds)
    except Exception as exc:
        QMessageBox.critical(
            self, "Spreadsheet export failed", f"Could not prepare spreadsheet:\n{exc}"
        )
        return
    dialog = CsvExportDialog(
        list(columns),
        Path(suggested) if suggested else self._default_save_path(ds, ".csv"),
        self,
    )
    if dialog.exec() != QDialog.DialogCode.Accepted:
        return
    output_path = dialog.path_edit.text().strip()
    selected_columns = dialog.selected_columns()
    separator = dialog.separator_edit.text()
    name = Path(output_path).name or "spreadsheet"
    def work(report):
        report(f"Writing {name}")
        path = write_spreadsheet_csv(
            ds,
            output_path,
            column_headers=selected_columns,
            separator=separator,
        )
        report(f"Wrote {name}")
        return [path]
    self._start_file_save(
        work,
        name=name,
        on_done=lambda written, _n=name, _ds=ds: self._on_files_saved(
            written, "Saved spreadsheet: ", _n, _ds
        ),
    )


def save_metadata_only(self: MainWindow, ds) -> None:
    """Write just the processing recipe, to a path the user chooses.

        Shared by *File › Save As › Viewer metadata only* and the Save / export
        dialog's entry for it. The path is asked for rather than derived: a
        dataset with no file behind it has no folder to be beside, and writing
        into the working directory would be a file the user never chose.
        """
    if ds is None:
        return
    from ..core.save import dataset_data_filename, write_metadata_sidecar
    from .metadata_save_dialog import ask_metadata_save_path
    data_filename = dataset_data_filename(ds)
    default_dir = self._state.prefs["file"].get("default_folder", str(Path.home()))
    target = ask_metadata_save_path(self, ds, default_dir,
                                    data_filename=data_filename)
    if target is None:
        return
    try:
        written = write_metadata_sidecar(
            ds, target, data_filename=data_filename,
            roi_records=self.save_roi_records(ds))
    except Exception as exc:                                # noqa: BLE001
        self._state.log(f"Failed to write viewer metadata: {exc}", "ERROR")
        QMessageBox.critical(self, "Save viewer metadata",
                             f"Could not write the metadata:\n{exc}")
        return
    note = (f" (accompanies {data_filename})" if data_filename
            else " (no data file on disk; it names none)")
    self._state.log(f"Saved viewer metadata: {written}{note}",
                    dataset_idx=self._post_load_index(ds))
    self._status_label.setText(f"Saved {written.name}.")


def _save_as_picasso_hdf5(self: MainWindow) -> None:
    ds = self._active_dataset_for_save()
    if ds is None:
        return
    from PyQt6.QtWidgets import QDialogButtonBox, QFormLayout
    dialog = QDialog(self)
    dialog.setWindowTitle("Save Picasso HDF5")
    layout = QVBoxLayout(dialog)
    form = QFormLayout()
    path_edit = QLineEdit(str(self._default_save_path(ds, ".hdf5")))
    browse = QPushButton("Browse...")
    row = QWidget()
    from PyQt6.QtWidgets import QHBoxLayout
    row_layout = QHBoxLayout(row)
    row_layout.setContentsMargins(0, 0, 0, 0)
    row_layout.addWidget(path_edit, 1)
    row_layout.addWidget(browse)
    form.addRow("File:", row)
    pixel_spin = QDoubleSpinBox()
    pixel_spin.setDecimals(3)
    pixel_spin.setRange(0.001, 1_000_000.0)
    pixel_spin.setValue(1.0)
    pixel_spin.setSuffix(" nm")
    pixel_spin.setToolTip(
        "Virtual Picasso camera pixel size. With 1 nm, x/y camera-pixel "
            "coordinates numerically match MINFLUX nanometres after origin shift."
    )
    form.addRow("Picasso pixel size:", pixel_spin)
    layout.addLayout(form)
    note = QLabel(
        "Writes a Picasso-compatible localization HDF5 at /locs plus a "
            "same-name .yaml metadata file. x/y are exported in camera pixels; "
            "z remains in nm."
    )
    note.setWordWrap(True)
    layout.addWidget(note)
    buttons = QDialogButtonBox(
        QDialogButtonBox.StandardButton.Save | QDialogButtonBox.StandardButton.Cancel
    )
    buttons.accepted.connect(dialog.accept)
    buttons.rejected.connect(dialog.reject)
    layout.addWidget(buttons)
    def choose_path() -> None:
        path, _ = QFileDialog.getSaveFileName(
            dialog, "Save Picasso HDF5", path_edit.text(), "Picasso HDF5 (*.hdf5)"
        )
        if path:
            path_edit.setText(path)
    browse.clicked.connect(choose_path)
    if dialog.exec() != QDialog.DialogCode.Accepted:
        return
    path = path_edit.text().strip()
    if not path:
        QMessageBox.warning(self, "Save Picasso HDF5", "Choose an output file path.")
        return
    from ..core.save import write_picasso_hdf5
    pixel_size_nm = pixel_spin.value()
    name = Path(path).name
    def work(report):
        report(f"Writing {name}")
        written = write_picasso_hdf5(
            ds, path, pixel_size_nm=pixel_size_nm
        )
        report(f"Wrote {name}")
        return written
    self._start_file_save(
        work,
        name=name,
        on_done=lambda written, _n=name, _ds=ds: self._on_files_saved(
            written, "Saved Picasso HDF5: ", _n, _ds
        ),
    )


def _save_as_ome_tiff(self: MainWindow) -> None:
    ds = self._active_dataset_for_save()
    if ds is None:
        return
    idx = self._state.active_idx
    if idx is None:
        return
    win = self._render_window_for_dataset(idx)
    if win is None:
        win = self._show_render(idx)
    if win is None:
        return
    try:
        win.show()
        win.raise_()
        win.activateWindow()
        win._export_to_tiff()
    except Exception as exc:
        QMessageBox.critical(self, "OME-TIFF export failed", str(exc))


def save_dataset(self: MainWindow, ds) -> None:
    """Open the Save / Export dialog for *ds* (the active dataset from File ›
        Save Processed Data, or a right-clicked one from the Dataset Manager)."""
    from . import save_dialog as save_dialog_module
    from .save_dialog import SaveProcessedDataDialog
    # A dataset is "file-backed" when it has a physical data file that, when
    # re-opened, reproduces it (.mat/.npy/.json). For those we only write the
    # metadata sidecar; otherwise (.msr extract, duplicate) we save the data.
    src = Path(getattr(ds.file, "folder", "") or "") / (getattr(ds.file, "name", "") or "")
    file_backed = src.is_file() and src.suffix.lower() in (".mat", ".npy", ".json")
    default_dir = self._state.prefs["file"].get("default_folder", str(Path.home()))
    # A one-dataset-per-file format cannot hold an overlay, so the dialog
    # offers the whole group: one file per channel in a folder of its own.
    members = self._overlay_members_for(ds)
    if len(members) > 1:
        from ..core.overlay_save import default_group_name
        default_dir = str(Path(default_dir) / default_group_name(members))
    dlg = SaveProcessedDataDialog(
        getattr(ds, "name", ""),
        members=[getattr(member, "name", "") for member in members],
        file_backed=file_backed,
        source_path=src if file_backed else None,
        default_dir=default_dir,
        prefs=self._state.prefs,
        parent=self,
    )
    if dlg.exec() != QDialog.DialogCode.Accepted:
        return
    opts = dlg.options()
    if opts.get("fmt") == save_dialog_module.METADATA_ONLY:
        # Not a data format: it writes the recipe alone, and asks where.
        self.save_metadata_only(ds)
        return
    if opts.get("group_folder") is not None:
        self._save_overlay_group(members, opts)
        return
    if opts.get("csv_mode") == "custom":
        # A custom table is written by a different function that takes no
        # content/include/filter arguments, so it leaves this path entirely.
        self.export_custom_csv(ds, opts.get("data_path"))
        return
    from ..core.save import save_processed
    # Overlay members and linked images travel with BOTH Zarr forms; only the
    # directory store can be updated in place. ROIs travel with every format
    # -- inside the store, or in the sidecar.
    zarr_context = (self._zarr_save_context(ds)
                    if opts.get("fmt") in {"zarr", "zarr_zip"}
                    else {"roi_records": self.save_roi_records(ds)})
    if opts.get("fmt") == "msr":
        zarr_context.setdefault("related_datasets", self._overlay_members_for(ds))
    zarr_overwrite = (
        self._zarr_overwrite_mode(opts["data_path"])
        if opts.get("fmt") == "zarr"
        else "replace"
    )
    if zarr_overwrite is None:
        return
    action = (
        "Updated processing in" if zarr_overwrite == "viewer"
        else "Saved processed data"
    )
    name = Path(opts["data_path"] or src.name or ds.name).name
    def work(report):
        report(f"Writing {name}")
        written = save_processed(
            ds,
            data_path=opts["data_path"],
            fmt=opts["fmt"],
            metadata_dir=src.parent if file_backed else None,
            content=opts.get("content", "raw"),
            include=opts.get("include"),
            filter_mode=opts.get("filter_mode", "flag"),
            zarr_overwrite=zarr_overwrite,
            log=self._state.log,
            **zarr_context,
        )
        report(f"Wrote {name}")
        return written
    self._start_file_save(
        work,
        name=name,
        on_done=lambda written, _a=action, _n=name, _ds=ds: self._on_processed_saved(
            written, _a, _n, _ds
        ),
    )


def _schedule_overlay_regroup(self: MainWindow) -> None:
    """Ask about regrouping once the batch has settled.

        Loading is asynchronous, so the channels of a dropped folder arrive one
        at a time; checking on each arrival would ask about the first two before
        the third had landed. The timer coalesces the batch into one question.
        """
    timer = getattr(self, "_regroup_timer", None)
    if timer is None:
        from PyQt6.QtCore import QTimer
        timer = QTimer(self)
        timer.setSingleShot(True)
        timer.timeout.connect(self._offer_overlay_regroup)
        self._regroup_timer = timer
    timer.start(400)


def _offer_overlay_regroup(self: MainWindow) -> None:
    """Offer to rebuild an overlay whose channels are all loaded."""
    if self._is_shutting_down:
        return
    from ..core.overlay_save import group_saved_datasets
    for group_id, found in group_saved_datasets(self._state.datasets).items():
        if len(found) < 2:
            continue
        if group_id in getattr(self, "_regroup_declined", set()):
            continue
        names = [str(getattr(ds, "name", "") or "") for _i, ds, _b in found]
        block = found[0][2]
        total = int(block.get("member_count") or len(found))
        listing = "\n".join(f"    • {name}" for name in names)
        missing = total - len(found)
        tail = (f"\n\n{missing} further channel(s) of this set are not open."
                if missing > 0 else "")
        answer = QMessageBox.question(
            self, "Overlay found",
            f"These {len(found)} datasets were saved together as one "
                f"overlay:\n\n{listing}{tail}\n\nCombine them as a "
                "multi-channel overlay again?",
            QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No,
            QMessageBox.StandardButton.Yes,
        )
        if answer != QMessageBox.StandardButton.Yes:
            # Remembered for the session: asking again on the next load of
            # the same set would be nagging about a settled decision.
            self._regroup_declined = getattr(self, "_regroup_declined", set())
            self._regroup_declined.add(group_id)
            self._state.log(
                f"Kept {len(found)} datasets separate; they were saved as "
                    "one overlay.")
            continue
        self._regroup_saved_overlay(found)


def _regroup_saved_overlay(self: MainWindow, found) -> None:
    """Put the loaded members back into one overlay, in their saved order."""
    import uuid

    from ..core.overlay import overlay_color_cycle
    from ..core.overlay_save import DATASET_OVERLAY_KEY
    # A fresh id per session: the saved one identified the set inside the
    # folder it was written to, not across sessions or against whatever else
    # is already loaded.
    group = f"saved:{uuid.uuid4()}"
    ordered = sorted(found, key=lambda item: int(item[2].get("member_index") or 0))
    luts = overlay_color_cycle(self._state.prefs)
    for position, (_index, ds, block) in enumerate(ordered):
        member = None
        members = block.get("members") or []
        at = int(block.get("member_index") or position)
        if 0 <= at < len(members):
            member = members[at]
        ds.state["overlay_id"] = group
        ds.state["render_group_id"] = group
        ds.state["overlay_index"] = position
        ds.state["overlay_order"] = position
        lut = (member or {}).get("lut") or luts[position % len(luts)]
        ds.state.setdefault("overlay_lut", lut)
        ds.state.setdefault("render_channel_lut", lut)
        # The block has done its job; leaving it would offer the same
        # regrouping again after any later load.
        ds.metadata.pop(DATASET_OVERLAY_KEY, None)
    names = ", ".join(str(getattr(ds, "name", "")) for _i, ds, _b in ordered)
    self._state.log(f"Combined {len(ordered)} saved channels as an overlay: {names}")
    self._refresh_overlay_windows()
    anchor = ordered[0][0]
    self._show_render(anchor)


def _save_overlay_group(self: MainWindow, members, opts) -> None:
    """Write every channel of an overlay into one folder, off the UI thread.

        Each channel becomes a data file plus its own metadata sidecar, and each
        sidecar names the siblings, so dropping the folder back on the window
        can recognise the set.
        """
    from ..core.overlay_save import save_overlay_group
    folder = Path(opts["group_folder"])
    fmt = opts["fmt"]
    if fmt == "csv" and opts.get("csv_mode") == "custom":
        # The custom table asks its own per-column questions, which cannot
        # be answered once for several channels.
        QMessageBox.information(
            self, "Save / export data",
            "A custom .csv table chooses its columns per dataset, so it "
                "cannot write a whole overlay at once.\n\nTick 'Write the "
                "canonical table instead' under More options, or save each "
                "channel on its own.")
        return
    rois = {index: self.save_roi_records(member)
            for index, member in enumerate(members)}
    names, content = list(opts["group_names"]), opts.get("content", "raw")
    include, filter_mode = opts.get("include"), opts.get("filter_mode", "flag")
    def work(report):
        report(f"Writing {len(members)} channel(s) to {folder.name}")
        written = save_overlay_group(
            members, folder, fmt, names=names, content=content,
            include=include, filter_mode=filter_mode,
            roi_records_for=rois.get)
        report(f"Wrote {folder.name}")
        return written
    self._start_file_save(
        work, name=folder.name,
        on_done=lambda written, _f=folder, _n=len(members): self._on_group_saved(
            written, _f, _n))


def _on_group_saved(self: MainWindow, written, folder, channels: int) -> None:
    if self._is_shutting_down:
        return
    self._state.log(
        f"Saved {channels} overlay channel(s) to {folder}: "
        + ", ".join(sorted(p.name for p in written)))
    self._status_label.setText(
        f"Saved {channels} channels to {folder.name}.")


def _on_processed_saved(self: MainWindow, written, action: str, name: str, ds) -> None:
    if self._is_shutting_down:
        return
    idx = self._post_load_index(ds)
    self._state.log(
        f"{action}: {', '.join(p.name for p in written)}",
        "INFO",
        dataset_idx=idx,
    )
    self._status_label.setText(f"{action}: {name}.")
    QMessageBox.information(
        self, "Save processed data",
        f"{action}:\n" + "\n".join(str(p) for p in written),
    )


def _maybe_offer_regroup(self: MainWindow, idx: int) -> None:
    from ..core.overlay_save import dataset_overlay_block
    try:
        ds = self._state.datasets[idx]
    except (IndexError, TypeError):
        return
    if dataset_overlay_block(ds):
        self._schedule_overlay_regroup()
