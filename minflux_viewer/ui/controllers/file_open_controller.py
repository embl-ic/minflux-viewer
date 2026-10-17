"""File open controller delegated by MainWindow."""

from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING

from PyQt6.QtWidgets import QDialog, QFileDialog, QMessageBox

from minflux_viewer.core import formats as _formats
from minflux_viewer.ui.common.background_tasks import BackgroundTask
from minflux_viewer.ui.common.qt_lifecycle import gui_action
from minflux_viewer.ui.controllers.window_support import _FMT_LOADERS, _SUPPORTED_EXTS, _ZarrIoTask

if TYPE_CHECKING:
    from minflux_viewer.ui.main_window import MainWindow


def _open_dialog(self: MainWindow) -> None:
    default = self._state.prefs["file"].get("default_folder", str(Path.home()))
    # Open… is for recognized MINFLUX data formats. Spreadsheets and TIFFs
    # have their own File-menu entries; drag-drop still accepts everything.
    paths, _ = QFileDialog.getOpenFileNames(
        self,
        "Open MINFLUX data",
        default,
        "MINFLUX data (*.mat *.npy *.msr *.json *.zarr);;"
            "MATLAB (*.mat);;"
            "NumPy (*.npy);;"
            "Imspector .msr (*.msr);;"
            "MINFLUX JSON (*.json);;"
            "Zarr (*.zarr);;"
            "All files (*)",
    )
    for p in paths:
        self._route_file(p)


def _open_msr_dialog(self: MainWindow, msr_path: str) -> None:
    """Hand an ``.msr`` file to the MSR Reader plugin (drag-drop entry point).

        **Every** ``.msr`` opens in the MSR reader — including image-only files
        with no MINFLUX data. Their OBF image series are viewable from the reader
        (right-click a Series node › *Preview* / *Open as image*, or select them in
        *Datasets / Fields included…* and *Open in MINFLUX viewer*)."""
    if not msr_path:
        return
    # ...except one we wrote ourselves. The reader exists to ask which
    # datasets to import, how to align the channels and whether to map a
    # confocal image; our own file already answers all three, and the saved
    # state would override the answers anyway. The reader stays reachable
    # from the Plugins menu for these files.
    from minflux_viewer.msr.quick_open import is_viewer_msr
    if is_viewer_msr(msr_path):
        self._open_viewer_msr(msr_path)
        return
    from minflux_viewer.ui.dialogs.msr_import_dialog import open_msr
    open_msr(msr_path, self._state, parent=self)


def _open_viewer_msr(self: MainWindow, msr_path: str) -> None:
    """Open a ``.msr`` this application wrote, restoring its session."""
    from minflux_viewer.msr.quick_open import load_viewer_msr
    name = Path(msr_path).name
    self._state.log(f"Opening '{name}': MINFLUX Viewer .msr, "
                        "restoring its saved processing state.")
    prefs = self._state.prefs
    def work(report):
        report(f"Reading {name}")
        # The worker must not touch Qt, so what the restore did is collected
        # here and logged on the GUI thread; list.append is all this is.
        messages: list[str] = []
        datasets = load_viewer_msr(msr_path, prefs=prefs, log=messages.append)
        report(f"Read {name}")
        return datasets, messages
    task = BackgroundTask(work, description=f"Opening {name}", category="load")
    task.signals.stage.connect(lambda text: self._state.status_progress(text))
    task.signals.done.connect(
        lambda result, _n=name, _p=msr_path: self._on_viewer_msr_loaded(
            result[0], _n, _p, messages=result[1]))
    task.signals.failed.connect(
        lambda message, _n=name: self._on_msr_open_failed(_n, message))
    self._begin_file_io(task)


def _on_viewer_msr_loaded(self: MainWindow, datasets, name: str, path: str,
                          *, messages=None) -> None:
    if self._is_shutting_down:
        return
    for message in messages or ():
        self._state.log(message)
    if not datasets:
        self._state.log(f"'{name}' holds no MINFLUX datasets.", "WARN")
        return
    # An overlay saved to .msr keeps every channel in the one file, so the
    # group is rebuilt here rather than left as unrelated datasets. The ids
    # are re-minted per open: the saved one identified the group inside the
    # file, not across sessions.
    group = f"msr:{Path(path).resolve()}"
    multi = len(datasets) > 1
    for order, ds in enumerate(datasets):
        if multi and ds.state.get("overlay_id"):
            ds.state["overlay_id"] = group
            ds.state["render_group_id"] = group
            ds.state.setdefault("overlay_index", order)
            ds.state.setdefault("overlay_order", order)
        self._state.add_dataset(ds)
    # Open Recent is recorded by add_dataset from dataset.file.recent_path,
    # which load_viewer_msr sets to the source .msr -- there is deliberately
    # no second call here.
    self._status_label.setText(
        f"Opened {name}: {len(datasets)} dataset(s), saved state restored.")


def _on_msr_open_failed(self: MainWindow, name: str, message: str) -> None:
    self._state.log(f"Failed to open '{name}': {message}", "ERROR")
    if not self._is_shutting_down:
        QMessageBox.critical(self, "Open failed",
                             f"Could not open {name}:\n{message}")


def sample_presets(self: MainWindow) -> list:
    from minflux_viewer.core.sample_presets import load_presets
    return load_presets(self._state.prefs)


def set_sample_presets(self: MainWindow, presets: list) -> None:
    """Persist the Sample-Data presets and rebuild the File submenu."""
    from minflux_viewer.core.sample_presets import save_presets
    save_presets(self._state.prefs, presets)
    self._state.save_prefs()
    self._rebuild_sample_menu()


def _rebuild_sample_menu(self: MainWindow) -> None:
    """(Re)build File › Open Sample Data from the persisted presets."""
    menu = getattr(self, "menuOpenSample", None)
    if menu is None:
        return
    menu.clear()
    for preset in self.sample_presets():
        act = menu.addAction(preset["name"])
        act.triggered.connect(lambda _=False, p=dict(preset): self.generate_simulated(p))
    menu.addSeparator()
    edit = menu.addAction("Edit presets (Data Simulator)…")
    edit.triggered.connect(self._open_data_simulator)


def _open_data_simulator(self: MainWindow) -> None:
    from minflux_viewer.ui.common.modeless import show_modeless
    from minflux_viewer.ui.tools.data_simulator_window import DataSimulatorWindow
    show_modeless(DataSimulatorWindow(self._state, owner=self), self)


def _finalize_sim_dataset(self, name, coords, tid, attrs):
    """Build one simulated dataset from arrays (version = 'simulation', Z scaling factor pinned)."""
    import uuid

    from minflux_viewer.core.dataset import build_localization_dataset
    attrs = dict(attrs or {})
    tim = attrs.pop("tim", None)
    ds = build_localization_dataset(
        name=name, x_nm=coords[:, 0], y_nm=coords[:, 1], z_nm=coords[:, 2],
        tid=tid, tim=tim, attrs=attrs, source_version="simulation",
        prefs=self._state.prefs)
    ds.file.folder = f"<simulated>/{uuid.uuid4().hex}"
    ds.metadata["simulated"] = True
    # Coordinates are the true simulated nm — pin Z scaling factor to 1.0 and suppress the
    # post-load auto-anisotropy estimate.
    try:
        ds.set_z_scaling_factor(1.0, source="simulated (true z)")
        ds.derived["z_scaling_factor"] = 1.0
    except Exception:
        pass
    # Attach the acquisition's shared fiducial beads (one set per Generate
    # call, so every channel of an overlay carries the same beads) — written
    # to grd/mbm/points when saved to .msr.
    spec = getattr(self, "_sim_bead_spec", None)
    if spec is not None:
        pts, pbg, used = spec
        ds.metadata["mbm_points"] = pts
        ds.metadata["mbm_points_by_gri"] = pbg
        ds.metadata["mbm_used"] = used
    return ds, int(coords.shape[0]), (int(tid.max()) if tid.size else 0)


def _build_simulated_dataset(self: MainWindow, name: str, kwargs: dict):
    from minflux_viewer.core.simulate import simulate_localizations
    coords, tid, attrs = simulate_localizations(**kwargs)
    return self._finalize_sim_dataset(name, coords, tid, attrs)


def generate_simulated(self: MainWindow, config: dict, *, include_beads: bool = False,
                       bead_count: int = 4) -> None:
    """Generate the dataset(s) for a Data-Simulator config / Sample-Data preset.
        ``channels > 1`` builds a multi-channel overlay; the NPC overlay / DCR
        multi-sims build a co-registered overlay / a single DCR-mixed dataset.

        When ``include_beads`` is set, one shared set of synthetic fiducial beads
        is generated for this call and attached to every produced dataset (so
        saving to .msr writes ``grd/mbm/points``)."""
    from minflux_viewer.core.sample_presets import normalize_preset
    from minflux_viewer.core.simulate import sim_kind
    p = normalize_preset(config)
    self._sim_bead_spec = None
    if include_beads:
        try:
            from minflux_viewer.core.simulate import simulate_beads
            field = float((p.get("params") or {}).get("size_nm", 5000.0))
            self._sim_bead_spec = simulate_beads(
                bead_count, field_nm=field, dim=int(p["dim"]), seed=p["seed"])
        except Exception as exc:
            self._state.log(f"Bead simulation failed: {exc}", "WARNING")
            self._sim_bead_spec = None
    try:
        kind = sim_kind(p["structure"])
        if kind == "overlay":
            self._generate_sim_overlay(p)
        elif kind == "dcr":
            self._generate_sim_dcr(p)
        elif kind == "ecoli":
            self._generate_sim_ecoli(p)
        elif kind == "tracking":
            self._generate_sim_tracking(p)
        elif kind == "track_overlay":
            self._generate_sim_track_overlay(p)
        else:
            self._generate_sim_single(p)
    finally:
        self._sim_bead_spec = None


def _generate_sim_overlay(self: MainWindow, p: dict) -> None:
    """NPC 3-channel overlay: co-registered channels on one shared scaffold."""
    import uuid

    from minflux_viewer.core.overlay import overlay_color_cycle
    from minflux_viewer.core.simulate import simulate_npc_overlay
    label = p["name"]
    self._status_label.setText(f"Simulating {label}…")
    try:
        chans = simulate_npc_overlay(
            p["params"], locs_per_trace=p["locs_per_trace"],
            precision_nm=p["precision_nm"], seed=p["seed"])
        overlay_id = f"sim:{uuid.uuid4().hex}"
        overlay_index = int(getattr(self, "_next_overlay_index", 1))
        self._next_overlay_index = overlay_index + 1
        cycle = overlay_color_cycle(self._state.prefs)
        prev_suspend = getattr(self._state, "suspend_auto_render", False)
        self._state.suspend_auto_render = True
        made: list[int] = []
        try:
            for order, ch in enumerate(chans, start=1):
                ds, _n, _t = self._finalize_sim_dataset(
                    f"Sample: {label} · {ch['name']}", ch["coords"], ch["tid"], ch["attrs"])
                lut = ch.get("lut") or cycle[(order - 1) % len(cycle)]
                ds.state.update({
                    "overlay_id": overlay_id, "render_group_id": overlay_id,
                    "overlay_index": overlay_index, "overlay_order": order,
                    "overlay_lut": lut, "render_channel_lut": lut})
                ds.metadata["overlay_id"] = overlay_id
                made.append(self._state.add_dataset(ds))
        finally:
            self._state.suspend_auto_render = prev_suspend
        self._state.log(
            f"Generated sample data '{label}': {len(chans)} co-registered "
                f"channel(s) on a shared NPC scaffold ({', '.join(c['name'] for c in chans)}).")
        if made:
            self._show_render(made[0])
    except Exception as exc:
        self._state.log(f"Sample data generation failed: {exc}", "ERROR")
        QMessageBox.critical(self, "Data Simulator", str(exc))
        self._status_label.setText("Sample data failed.")


def _generate_sim_dcr(self: MainWindow, p: dict) -> None:
    """NPC 2-channel by DCR: one dataset, two reporters via a bimodal ``dcr``."""
    from minflux_viewer.core.simulate import simulate_npc_dcr
    label = p["name"]
    self._status_label.setText(f"Simulating {label}…")
    try:
        coords, tid, attrs = simulate_npc_dcr(
            p["params"], locs_per_trace=p["locs_per_trace"],
            precision_nm=p["precision_nm"], seed=p["seed"])
        ds, nloc, ntr = self._finalize_sim_dataset(f"Sample: {label}", coords, tid, attrs)
        self._state.log(
            f"Generated sample data '{label}': {nloc:,} localization(s), {ntr:,} trace(s); "
                "two reporters mixed with a bimodal 'dcr' (separate via Process › Channel › "
                "Separate Channel by DCR).")
        self._state.add_dataset(ds)
    except Exception as exc:
        self._state.log(f"Sample data generation failed: {exc}", "ERROR")
        QMessageBox.critical(self, "Data Simulator", str(exc))
        self._status_label.setText("Sample data failed.")


def _generate_sim_tracking(self: MainWindow, p: dict) -> None:
    """A pure tracking dataset: molecules moving, with real timestamps."""
    from minflux_viewer.core.simulate import simulate_tracking_shells
    label = p["name"]
    self._status_label.setText(f"Simulating {label}\u2026")
    try:
        coords, tid, attrs = simulate_tracking_shells(
            p["params"], precision_nm=p["precision_nm"], seed=p["seed"])
        ds, nloc, ntr = self._finalize_sim_dataset(
            f"Sample: {label}", coords, tid, attrs)
        span = float(attrs["tim"].max() - attrs["tim"].min())
        self._state.log(
            f"Generated sample data '{label}': {nloc:,} localization(s), "
                f"{ntr:,} track(s) over {span:,.0f} s. Open View \u203a Tracking "
                "View to play them.")
        self._state.add_dataset(ds)
    except Exception as exc:
        self._state.log(f"Sample data generation failed: {exc}", "ERROR")
        QMessageBox.critical(self, "Data Simulator", str(exc))
        self._status_label.setText("Sample data failed.")


def _generate_sim_track_overlay(self: MainWindow, p: dict) -> None:
    """A structure channel and a tracking channel, as one overlay.

        The two-colour trafficking design: the scaffold is the reference frame
        and the cargo moves through it, so the two must arrive grouped or the
        Tracking View has nothing to draw its comets over.
        """
    import uuid

    from minflux_viewer.core.simulate import simulate_npc_tracking
    label = p["name"]
    self._status_label.setText(f"Simulating {label}\u2026")
    try:
        chans = simulate_npc_tracking(
            p["params"], locs_per_trace=p["locs_per_trace"],
            precision_nm=p["precision_nm"], seed=p["seed"])
        overlay_id = f"sim:{uuid.uuid4().hex}"
        overlay_index = int(getattr(self, "_next_overlay_index", 1))
        self._next_overlay_index = overlay_index + 1
        prev_suspend = getattr(self._state, "suspend_auto_render", False)
        self._state.suspend_auto_render = True
        made: list[int] = []
        try:
            for order, ch in enumerate(chans, start=1):
                ds, _n, _t = self._finalize_sim_dataset(
                    f"Sample: {label} \u00b7 {ch['name']}",
                    ch["coords"], ch["tid"], ch["attrs"])
                lut = ch.get("lut") or "Gray"
                ds.state.update({
                    "overlay_id": overlay_id, "render_group_id": overlay_id,
                    "overlay_index": overlay_index, "overlay_order": order,
                    "overlay_lut": lut, "render_channel_lut": lut})
                ds.metadata["overlay_id"] = overlay_id
                # Recorded so the Tracking View's role guess can be checked
                # against what was simulated, rather than only inferred.
                ds.metadata["simulated_role"] = ch.get("role", "")
                made.append(self._state.add_dataset(ds))
        finally:
            self._state.suspend_auto_render = prev_suspend
        counts = " + ".join(f"{c['coords'].shape[0]:,}" for c in chans)
        self._state.log(
            f"Generated sample data '{label}': {len(chans)} channel(s) "
                f"({counts} localizations) as one overlay \u2014 a static scaffold "
                "and cargo tracked through it. Open View \u203a Tracking View.")
        if made:
            self._state.set_active(made[0])
    except Exception as exc:
        self._state.log(f"Sample data generation failed: {exc}", "ERROR")
        QMessageBox.critical(self, "Data Simulator", str(exc))
        self._status_label.setText("Sample data failed.")


def _generate_sim_ecoli(self: MainWindow, p: dict) -> None:
    """One rod cell of labelled HlyB dimers — a known-distance control."""
    from minflux_viewer.core.simulate import simulate_ecoli_hlyb
    label = p["name"]
    self._status_label.setText(f"Simulating {label}…")
    try:
        params = p["params"]
        coords, tid, attrs = simulate_ecoli_hlyb(
            params, locs_per_trace=p["locs_per_trace"],
            precision_nm=p["precision_nm"], seed=p["seed"])
        ds, nloc, ntr = self._finalize_sim_dataset(
            f"Sample: {label}", coords, tid, attrs)
        self._state.log(
            f"Generated sample data '{label}': {nloc:,} localization(s), "
                f"{ntr:,} trace(s); HlyB dimers planted at "
                f"{float(params['dimer_distance_nm']):g} +/- "
                f"{float(params['dimer_distance_sd_nm']):g} nm on a "
                f"{float(params['cell_length_nm']):g} x "
                f"{2 * float(params['cell_radius_nm']):g} nm rod "
                f"({float(params['dimer_fraction']) * 100:.0f}% of subunits "
                f"paired, {float(params['detection_probability']) * 100:.0f}% "
                f"detected). Every trace is one subunit seen once, so there are "
                f"no repeat visits and no drift — run Plugins > HlyB/D subunit "
                f"pair analysis on it to check the workflow recovers the "
                f"planted distance.")
        self._state.add_dataset(ds)
    except Exception as exc:
        self._state.log(f"Sample data generation failed: {exc}", "ERROR")
        QMessageBox.critical(self, "Data Simulator", str(exc))
        self._status_label.setText("Sample data failed.")


def _generate_sim_single(self: MainWindow, p: dict) -> None:
    """Generate a single-structure sample (``channels > 1`` = independent overlay)."""
    import uuid

    from minflux_viewer.core.overlay import overlay_color_cycle
    from minflux_viewer.core.sample_presets import simulate_kwargs
    label = p["name"]
    channels = p["channels"]
    kwargs = simulate_kwargs(p)
    self._status_label.setText(f"Simulating {label}…")
    try:
        if channels <= 1:
            ds, nloc, ntr = self._build_simulated_dataset(f"Sample: {label}", kwargs)
            self._state.log(
                f"Generated sample data '{label}': {nloc:,} localization(s), "
                    f"{ntr:,} trace(s) ({p['dim']}-D, precision {p['precision_nm']:g} nm).")
            self._state.add_dataset(ds)
            return
        # multi-channel overlay: build each channel, link, open one grouped view
        overlay_id = f"sim:{uuid.uuid4().hex}"
        overlay_index = int(getattr(self, "_next_overlay_index", 1))
        self._next_overlay_index = overlay_index + 1
        cycle = overlay_color_cycle(self._state.prefs)
        prev_suspend = getattr(self._state, "suspend_auto_render", False)
        self._state.suspend_auto_render = True
        made: list[int] = []
        try:
            for ch in range(channels):
                kw = dict(kwargs)
                if kw.get("seed") is not None:
                    kw["seed"] = int(kw["seed"]) + ch      # distinct realizations
                ds, nloc, _ntr = self._build_simulated_dataset(
                    f"Sample: {label} (ch{ch + 1})", kw)
                lut = cycle[ch % len(cycle)]
                ds.state.update({
                    "overlay_id": overlay_id, "render_group_id": overlay_id,
                    "overlay_index": overlay_index, "overlay_order": ch + 1,
                    "overlay_lut": lut, "render_channel_lut": lut})
                ds.metadata["overlay_id"] = overlay_id
                made.append(self._state.add_dataset(ds))
        finally:
            self._state.suspend_auto_render = prev_suspend
        self._state.log(
            f"Generated sample data '{label}': {channels} channel(s) overlaid.")
        if made:
            self._show_render(made[0])
    except Exception as exc:
        self._state.log(f"Sample data generation failed: {exc}", "ERROR")
        QMessageBox.critical(self, "Data Simulator", str(exc))
        self._status_label.setText("Sample data failed.")


@gui_action("Open file")
def _route_file(self: MainWindow, path: str) -> None:
    """
        Route a single file path to the correct loader based on extension.
        Unsupported types are reported to the log.
        """
    p   = Path(path)
    ext = p.suffix.lower()
    # Opening a file is not always "load a dataset": the registry says what
    # this one should DO. A .msr opens the MSR reader, a ROI set goes to the
    # ROI Manager, a filter preset to the Filter dialog. The .json fork is
    # decided by positive content markers, in a declared order -- the old
    # code tried to load data first and only looked for a filter preset if
    # that raised, so a malformed data file was indistinguishable from one.
    spec = _formats.resolve_open(path)
    if spec is not None:
        action = spec.action
        if action is _formats.OpenAction.ROI_MANAGER:
            self._load_roi_json(path)
            return
        if action is _formats.OpenAction.FILTER_DIALOG:
            self._load_filter_json(path)
            return
        if action is _formats.OpenAction.METADATA_RECIPE:
            # A recipe needs a dataset to act on, so this asks which one --
            # every loaded dataset is offered and the match only picks the
            # default, because applying a saved processing to another copy
            # (or another dataset) is a legitimate thing to want.
            self._open_metadata_sidecar(path)
            return
        if action is _formats.OpenAction.MSR_READER:
            self._open_msr_dialog(path)
            return
        if action is _formats.OpenAction.IMAGE_VIEWER:
            self._load_tiff(path)
            return
        if action is _formats.OpenAction.MBM_INFO:
            self._load_mbm_companion(path)
            return
    # Dataset and spreadsheet paths still go through content sniffing, which
    # resolves an extension that disagrees with the bytes (a .npz that is
    # really an .xlsx, and so on) and reports the mismatch.
    from minflux_viewer.core.format_sniff import resolve_format
    fmt, note = resolve_format(path)
    if note:
        self._state.log(f"'{p.name}': {note}.", "WARN")
    loader = _FMT_LOADERS.get(fmt) if fmt else None
    if loader is None:
        msg = (
            f"Unsupported file type: '{p.name}'  "
                f"(extension '{ext or '(none)'}' unrecognised and content could "
                f"not be identified).  Supported: .mat, .npy, .csv, .tsv, .txt, "
                f".xlsx, .xlsm, .msr, .tif, .tiff, .json, .zarr"
        )
        self._state.log(msg, "WARN")
        self._status_label.setText(f"Skipped: {p.name} (unsupported type)")
        return
    getattr(self, loader)(path)


def _route_path(self: MainWindow, path: str) -> None:
    """
        Route a path that may be a file OR a directory.
        Directories are scanned for all supported files (non-recursive).
        """
    p = Path(path)
    # A Zarr dataset is a directory with a data suffix, not a folder to
    # scan.  Route it as one dataset before the generic folder branch.
    if p.is_dir() and p.suffix.lower() == ".zarr":
        self._route_file(path)
        return
    if p.is_dir():
        found = sorted(
            f for f in p.iterdir()
            if f.is_file() and f.suffix.lower() in _SUPPORTED_EXTS
        )
        if found:
            self._state.log(
                f"Folder dropped: found {len(found)} supported file(s) in '{p.name}'",
                "INFO",
            )
            # Through the batch router, so a folder holding a data file
            # and its recipe pairs them exactly as dropping both does.
            self.route_paths([str(f) for f in found])
        else:
            msg = (
                f"Folder '{p.name}' contains no supported files "
                    f"(.mat, .npy, .csv, .tsv, .xlsx, .xlsm, .msr, .tif, .tiff, .json, .zarr)."
            )
            self._state.log(msg, "WARN")
            self._status_label.setText(f"No supported files in folder: {p.name}")
    else:
        self._route_file(path)


def _start_dataset_load(self: MainWindow, path: str, loader, *, kind: str) -> None:
    """Load one ordinary dataset without running its parser on Qt's thread."""
    import copy
    name = Path(path).name
    prefs = copy.deepcopy(self._state.prefs)
    apply_sidecar = self._sidecar_for_load(path)
    def work(report):
        report(f"Reading {name}")
        dataset = loader(path, prefs=prefs, apply_sidecar=apply_sidecar)
        report(f"Read {name}")
        return dataset
    task = BackgroundTask(
        work, description=f"Loading {name}", category="load"
    )
    task.signals.done.connect(
        lambda dataset, _n=name, _k=kind: self._on_dataset_loaded(
            dataset, _n, _k
        )
    )
    task.signals.failed.connect(
        lambda message, _n=name: self._on_file_load_failed(_n, message)
    )
    self._begin_file_io(task)


def _on_dataset_loaded(self: MainWindow, dataset, name: str, kind: str) -> None:
    if self._is_shutting_down:
        return
    try:
        self._state.add_dataset(dataset)
    except Exception as exc:  # noqa: BLE001 - Qt callback boundary
        self._on_file_load_failed(name, str(exc))
        return
    self._state.log(
        f"Opened {kind} '{name}' ({dataset.prop.num_loc:,} localizations).",
        "INFO",
    )
    self._status_label.setText(f"Opened {name}.")


def _on_file_load_failed(self: MainWindow, name: str, message: str) -> None:
    self._state.log(f"Failed to load '{name}': {message}", "ERROR")
    if not self._is_shutting_down:
        QMessageBox.critical(self, "Load error", message)
    self._status_label.setText("Load failed.")


def _load_mat(self: MainWindow, path: str) -> None:
    self._status_label.setText(f"Loading {Path(path).name}…")
    self._state.log(f"Opening .mat: {path}", "INFO")
    from minflux_viewer.core.loader_format_readers import load_dataset
    self._start_dataset_load(path, load_dataset, kind="MATLAB data")


def _load_npy(self: MainWindow, path: str) -> None:
    self._status_label.setText(f"Loading {Path(path).name}…")
    self._state.log(f"Opening .npy: {path}", "INFO")
    from minflux_viewer.core.loader_format_readers import load_npy
    self._start_dataset_load(path, load_npy, kind="NumPy data")


def _load_npz(self: MainWindow, path: str) -> None:
    self._status_label.setText(f"Loading {Path(path).name}…")
    self._state.log(f"Opening .npz: {path}", "INFO")
    from minflux_viewer.core.loader_format_readers import load_npz
    self._start_dataset_load(path, load_npz, kind="NumPy archive")


def _load_zarr(self: MainWindow, path: str) -> None:
    """Open a ``.zarr`` store off the UI thread.

        Reading one is CPU-bound (decompress, rebuild the structured array,
        normalize), which on a large acquisition is ~10 s. Doing that inline
        froze the whole window; the dataset is still ADDED on the UI thread,
        because everything downstream of ``add_dataset`` touches widgets.
        """
    name = Path(path).name
    self._state.log(f"Opening .zarr: {path}", "INFO")
    prefs = self._state.prefs
    def work(report):
        from minflux_viewer.core.minflux_zarr import load_minflux_zarr_project
        report(f"Reading {name}")
        project = load_minflux_zarr_project(path, prefs=prefs)
        report(f"Read {name}: {len(project.datasets)} dataset(s)")
        return project
    task = _ZarrIoTask(work, description=f"Opening {name}")
    task.signals.stage.connect(
        lambda text: self._state.status_progress(text))
    task.signals.done.connect(
        lambda project: self._on_zarr_loaded(project, name))
    task.signals.failed.connect(
        lambda message: self._on_zarr_io_failed(name, message, "Load error"))
    self._begin_zarr_io(task)


def _begin_file_io(self: MainWindow, task, *, pool_name: str = "file-io") -> None:
    """Start heavy non-Qt work and retain it until its terminal signal."""
    from minflux_viewer.ui.common.background_tasks import shared_thread_pool
    self._status_label.setText(f"{task.description}…")
    self._state.status_progress(task.description)
    tasks = getattr(self, "_file_io_tasks", None)
    if tasks is None:
        tasks = self._file_io_tasks = []
    tasks.append(task)
    task.signals.finished.connect(lambda _t=task: self._finish_file_io(_t))
    shared_thread_pool(pool_name).start(task)


def _begin_zarr_io(self: MainWindow, task) -> None:
    """Compatibility wrapper for the former Zarr-only task runner."""
    if getattr(self, "_file_io_tasks", None) is None:
        self._file_io_tasks = []
    # Some extensions/tests use this list to pump the event loop until a
    # store finishes. Keep it as an alias, not a second ownership registry.
    self._zarr_io_tasks = self._file_io_tasks
    self._begin_file_io(task, pool_name="zarr-io")


def _finish_file_io(self: MainWindow, task) -> None:
    try:
        getattr(self, "_file_io_tasks", []).remove(task)
    except ValueError:
        pass


def _finish_zarr_io(self: MainWindow, task) -> None:
    """Compatibility wrapper for the former Zarr-only task runner."""
    self._finish_file_io(task)


def _on_zarr_io_failed(self: MainWindow, name: str, message: str, title: str) -> None:
    self._state.log(f"Failed: '{name}': {message}", "ERROR")
    if not self._is_shutting_down:
        QMessageBox.critical(self, title, message)
    self._status_label.setText(f"{title}.")


def _on_zarr_loaded(self: MainWindow, project, name: str) -> None:
    """Install a loaded project — on the UI thread, where widgets live."""
    if self._is_shutting_down:
        return
    try:
        grouped = bool(project.manifest.get("is_overlay")) and len(project.datasets) > 1
        previous = self._state.suspend_auto_render
        if grouped:
            self._state.suspend_auto_render = True
        indices = []
        try:
            for dataset in project.datasets:
                indices.append(self._state.add_dataset(dataset))
        finally:
            self._state.suspend_auto_render = previous
        self._restore_zarr_rois(project.roi_records, project.manifest, indices)
        if grouped and indices and self._state.prefs.get("data", {}).get("show_render", True):
            self._show_render(indices[0])
    except Exception as exc:                                # noqa: BLE001
        self._on_zarr_io_failed(name, str(exc), "Load error")
        return
    self._state.log(
        f"Opened '{name}': {len(project.datasets)} dataset(s).")
    self._status_label.setText(f"Opened {name}.")


def _restore_zarr_rois(self: MainWindow, records, manifest: dict, dataset_indices) -> None:
    """Restore portable project ROI records into the session-level store."""
    if not records:
        return
    from minflux_viewer.core.roi import RoiRecord
    members = list(manifest.get("datasets") or [])
    id_to_idx = {
        str(spec.get("id")): int(idx)
        for spec, idx in zip(members, dataset_indices)
    }
    existing = {record.id for record in self._state.rois.records}
    added = 0
    for payload in records:
        if not isinstance(payload, dict):
            continue
        values = dict(payload)
        dataset_id = str(values.pop("dataset_id", "") or "")
        context = dict(values.get("context") or {})
        if dataset_id in id_to_idx:
            context["dataset_idx"] = id_to_idx[dataset_id]
        values["context"] = context
        try:
            record = RoiRecord(**values)
        except (TypeError, ValueError):
            continue
        if record.id not in existing:
            self._state.rois.add(record)
            existing.add(record.id)
            added += 1
    if added:
        # Same contract as the sidecar path: the ROIs are visible in the
        # Manager, concatenated with whatever it already held.
        self._show_roi_manager()
        self._state.log(f"Restored {added} ROI(s) into the ROI Manager.")


def _load_mbm_companion(self: MainWindow, path: str) -> None:
    """Open a standalone ``_mbm`` bead table in the MBM info window.

        ``.mat`` / ``.npy`` / ``.json`` / ``.csv`` cannot hold localizations and
        their bead reference together, so the MSR reader writes the beads as a
        companion file. It is not a dataset -- loading it as one produced an
        error or an empty dataset -- so it goes straight to the same window
        *View mbm info...* opens for a loaded dataset.
        """
    name = Path(path).name
    self._status_label.setText(f"Loading {name}…")
    self._state.log(f"Opening MBM companion: {path}", "INFO")
    def work(report):
        from minflux_viewer.core.mbm_file import read_mbm_points
        from minflux_viewer.plugins.msr_reader.beads_drift import extract_bead_drift
        report(f"Reading {name}")
        points = read_mbm_points(path)
        beads = extract_bead_drift(points, {}, [])
        return points, beads
    task = BackgroundTask(
        work, description=f"Loading {name}", category="load"
    )
    task.signals.done.connect(
        lambda result, _n=name: self._on_mbm_companion_loaded(_n, result)
    )
    task.signals.failed.connect(
        lambda message, _n=name: self._on_file_load_failed(_n, message)
    )
    self._begin_file_io(task)


def _on_mbm_companion_loaded(self: MainWindow, name: str, result) -> None:
    if self._is_shutting_down:
        return
    points, beads = result
    if not beads:
        msg = (f"'{name}' is an MBM bead table but carries no usable bead "
                   "traces (no gri / xyz / tim rows).")
        self._state.log(msg, "WARN")
        QMessageBox.information(self, "MBM companion", msg)
        self._status_label.setText(f"Skipped: {name} (no bead traces)")
        return
    from minflux_viewer.ui.common.modeless import show_modeless
    from minflux_viewer.ui.tools.mbm_info_window import MbmInfoWindow
    # No localizations accompany a companion file, so there is no data
    # region box to draw the beads against.
    win = MbmInfoWindow(
        name, beads, data_bounds_nm=None, prefs=self._state.prefs)
    show_modeless(win, self)
    self._state.log(
        f"Opened MBM companion '{name}': {len(beads)} bead(s), "
            f"{int(points.size):,} recorded position(s).")
    self._status_label.setText(f"MBM info: {name}")


def _load_csv(self: MainWindow, path: str) -> None:
    # CSV/TSV/TXT all go through the smart spreadsheet importer.
    self._load_spreadsheet(path)


def _load_spreadsheet(self: MainWindow, path: str) -> None:
    # A table this application wrote is recognised by its headers and loaded
    # directly; anything else opens the column-mapping dialog (pre-filled
    # from a value-based best guess of x/y/z/tid/tim + units) so the user
    # confirms the mapping — never a silent *generic* import.
    self._status_label.setText(f"Loading {Path(path).name}…")
    self._state.log(f"Opening spreadsheet: {path}", "INFO")
    from minflux_viewer.ui.dialogs.spreadsheet_import_dialog import minflux_direct_load_kind
    kind = minflux_direct_load_kind(path)
    if kind is not None:
        from minflux_viewer.core.loader_format_readers import load_csv
        self._state.log(
            f"'{Path(path).name}': canonical MINFLUX {kind} table — "
                "loading directly, no column mapping needed.",
            "INFO",
        )
        self._start_dataset_load(path, load_csv, kind="spreadsheet")
        return
    import copy

    from minflux_viewer.core.spreadsheet_loader import read_table_preview
    prefs = copy.deepcopy(self._state.prefs)
    apply_sidecar = self._sidecar_for_load(path)
    name = Path(path).name
    def work(report):
        report(f"Reading spreadsheet preview for {name}")
        return read_table_preview(path)
    task = BackgroundTask(
        work, description=f"Inspecting {name}", category="load"
    )
    task.signals.done.connect(
        lambda table, _p=prefs, _s=apply_sidecar: self._on_spreadsheet_preview(
            table, _p, _s
        )
    )
    task.signals.failed.connect(
        lambda message, _n=name: self._on_file_load_failed(_n, message)
    )
    self._begin_file_io(task)


def _on_spreadsheet_preview(self: MainWindow, table, prefs: dict, apply_sidecar: bool) -> None:
    if self._is_shutting_down:
        return
    from minflux_viewer.ui.dialogs.spreadsheet_import_dialog import (
        SpreadsheetMappingDialog,
        build_spreadsheet_dataset,
    )
    dialog = SpreadsheetMappingDialog(table, prefs=prefs, parent=self)
    if dialog.exec() != QDialog.DialogCode.Accepted:
        self._status_label.setText("Spreadsheet import cancelled.")
        return
    spec = dialog.dataset_build_spec()
    name = Path(table.path).name
    def work(report):
        report(f"Building dataset from {name}")
        return build_spreadsheet_dataset(
            table, spec, prefs=prefs, apply_sidecar=apply_sidecar
        )
    task = BackgroundTask(
        work, description=f"Importing {name}", category="load"
    )
    task.signals.done.connect(
        lambda dataset, _n=name: self._on_dataset_loaded(
            dataset, _n, "spreadsheet"
        )
    )
    task.signals.failed.connect(
        lambda message, _n=name: self._on_file_load_failed(_n, message)
    )
    self._begin_file_io(task)


def _open_image_viewer(self: MainWindow, source, key: str, *, initial_series_index: int | None = None) -> None:
    """Show *source* (a TIFF or OBF image source) in a TIFF viewer window,
        de-duplicated by *key* (normally one key per file). The window owns /
        closes the source; a multi-series source can switch series in-window.
        When a file already has an image window, *initial_series_index* selects
        the requested series in that existing window and the replacement source
        is closed immediately.
        """
    existing = self._tiff_windows.get(key)
    if existing is not None:
        try:
            if initial_series_index is not None:
                existing.set_series_index(int(initial_series_index))
            existing.show()
            existing.raise_()
            existing.activateWindow()
            if getattr(existing, "_source", None) is not source:
                source.close()
            return
        except RuntimeError:
            self._tiff_windows.pop(key, None)
    from minflux_viewer.ui.tools.tiff_viewer_window import TiffViewerWindow
    win = TiffViewerWindow(source, state=self._state)
    win.destroyed.connect(lambda _=None, k=key: self._tiff_windows.pop(k, None))
    self._tiff_windows[key] = win
    win.show()
    self._place_new_window(win)
    win.raise_()
    win.activateWindow()
    meta = source.metadata
    self._state.log(
        f"Opened image viewer: "
            f"{getattr(source, 'display_name', None) or Path(source.path).name} "
            f"[{meta.image_name}]  |  "
            f"axes={meta.axes}  |  shape={meta.shape}  |  dtype={meta.dtype}",
        "INFO",
    )


def _load_tiff(self: MainWindow, path: str) -> None:
    self._status_label.setText(f"Opening TIFF {Path(path).name}…")
    self._state.log(f"Opening TIFF: {path}", "INFO")
    resolved = str(Path(path).resolve())
    def work(report):
        from minflux_viewer.core.tiff_source import TiffImageSource
        report(f"Reading TIFF metadata for {Path(path).name}")
        return TiffImageSource(path)
    task = BackgroundTask(
        work,
        description=f"Opening TIFF {Path(path).name}",
        category="load",
        discard_result=lambda source: source.close(),
    )
    task.signals.done.connect(
        lambda source, _p=path, _r=resolved: self._on_tiff_source_loaded(
            source, _p, _r
        )
    )
    task.signals.failed.connect(
        lambda message, _n=Path(path).name: self._on_file_load_failed(
            _n, message
        )
    )
    self._begin_file_io(task)


def _on_tiff_source_loaded(self: MainWindow, source, path: str, resolved: str) -> None:
    if self._is_shutting_down:
        source.close()
        return
    try:
        # One window per file; multi-series files switch via the in-window
        # Series dropdown (no separate chooser dialog).
        self._open_image_viewer(source, f"{resolved}#img")
    except Exception as exc:  # noqa: BLE001 - Qt callback boundary
        source.close()
        self._on_file_load_failed(Path(path).name, str(exc))
        return
    try:
        self._state._record_recent(resolved)
        self._mark_recent_menu_stale()
    except Exception:
        pass
    self._status_label.setText(f"TIFF opened: {Path(path).name}")


def _load_json(self: MainWindow, path: str) -> None:
    self._status_label.setText(f"Loading {Path(path).name}…")
    self._state.log(f"Opening JSON: {path}", "INFO")
    # A native ROI-set JSON (a dict with a "rois" list) loads ROIs, not data.
    try:
        from minflux_viewer.core.roi import is_roi_json_file
        if is_roi_json_file(path):
            self._load_roi_json(path)
            return
    except Exception:
        pass
    source = Path(path)
    if source.stat().st_size >= 1 << 30:
        from minflux_viewer.core.export_size import format_file_size
        choice = QMessageBox.warning(
            self,
            "Very large JSON dataset",
            f"{source.name} is {format_file_size(source.stat().st_size)}.\n\n"
                "JSON must decode every textual field before the numeric arrays "
                "can be reconstructed. Loading can take minutes and the viewer "
                "may be temporarily unresponsive. Zarr, MAT or NumPy is normally "
                "a better working format.\n\nOpen it anyway?",
            QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.Cancel,
            QMessageBox.StandardButton.Cancel,
        )
        if choice != QMessageBox.StandardButton.Yes:
            self._status_label.setText("JSON load cancelled.")
            return
    from minflux_viewer.core.loader_format_readers import load_json
    self._start_dataset_load(path, load_json, kind="JSON data")


def _load_roi_json(self: MainWindow, path: str) -> None:
    """Load a native ROI-set file (.json / .roi / .zip) into the ROI Manager,
        attaching the ROIs to the active dataset and revealing them."""
    try:
        records = self._state.rois.load(path)
    except Exception as exc:
        self._state.log(f"Failed to load ROI file '{Path(path).name}': {exc}", "ERROR")
        QMessageBox.critical(self, "Open ROI set", str(exc))
        self._status_label.setText("Load failed.")
        return
    idx = self._state.active_idx
    from minflux_viewer.core.roi_scope import with_owner
    for r in records:
        ctx = dict(r.context) if isinstance(r.context, dict) else {}
        if isinstance(idx, int):
            # The records are already in the store, held by key: retarget
            # the key, not a position the store has already replaced.
            ctx = with_owner(ctx, idx, self._state)
        # An ImageJ .roi/.zip carries no family, and neither does a native
        # set saved before scoping existed; both are spatial. Only fill it
        # in when absent, so a saved histogram band keeps its own family.
        ctx.setdefault("source_view", "render")
        r.context = ctx
        r.selection_dirty = True
    self._state.rois.set_show_all(True)
    if records:
        self._state.rois.select([r.id for r in records])
    self._show_roi_manager()
    self._state.log(f"Loaded {len(records)} ROI(s) from {Path(path).name}.")
    self._status_label.setText("Ready.")


def apply_metadata_to_dataset(self: MainWindow, idx: int, meta: dict, source: str) -> bool:
    """Apply a parsed recipe to dataset *idx* and bring the UI in line.

        ⚠ ``apply_metadata_recipe`` only writes ROI records into
        ``ds.metadata``; it is ``_post_load_finalize`` that puts them in the ROI
        Manager, and that runs at *dataset-load* time. Applying a recipe to an
        already-loaded dataset therefore reported "3 ROI(s)" and left the
        Manager empty. Every apply path goes through here so the ROIs and the
        filter dialog actually appear.
        """
    from minflux_viewer.core.loader_metadata_recipe import apply_metadata_recipe
    if not (0 <= idx < len(self._state.datasets)):
        return False
    ds = self._state.datasets[idx]
    try:
        applied = apply_metadata_recipe(ds, meta)
    except Exception as exc:
        self._state.log(f"Processing metadata '{source}' on '{ds.name}': {exc}",
                        "ERROR")
        QMessageBox.critical(self, "Apply processing metadata", str(exc))
        return False
    if not applied:
        self._state.log(
            f"Processing metadata '{source}' on '{ds.name}': nothing to "
                "apply (empty recipe).", "WARN")
        return True
    self._state.log(
        f"Applied '{source}' to '{ds.name}': " + ", ".join(applied) + ".",
        dataset_idx=idx)
    self._state.notify_filter_changed(idx)
    self._state.notify_calibration_changed(idx)
    self._restore_saved_rois(ds, idx)
    self._show_saved_filters(ds, idx)
    self._refresh_overlay_windows()
    return True


def _remember_pending_metadata(self: MainWindow, path, meta: dict) -> None:
    """Hold one recipe for a dataset not yet loaded (points 3.1 / 4).

        Deliberately inert: nothing is opened, no Filter dialog, no ROI Manager
        — there is no dataset for them to describe yet.
        """
    previous = self._pending_metadata
    self._pending_metadata = (Path(path), dict(meta))
    if previous:
        self._state.log(
            f"Processing metadata '{Path(path).name}' held; replaced the "
                f"pending '{previous[0].name}' (only one is kept).", "WARN")
    else:
        self._state.log(
            f"Processing metadata '{Path(path).name}' held — it will be "
                "applied to the next matching dataset loaded, and discarded "
                "otherwise.")


def _take_pending_metadata(self: MainWindow, ds) -> tuple:
    """Consume the pending recipe for a just-added *ds*, or drop it.

        The lifetime is one dataset load, by design: the expected next action
        after loading a recipe is loading its data. Anything else and the
        recipe is discarded (reported to the Log, never a dialog).
        """
    pending = self._pending_metadata
    if not pending:
        return ()
    from minflux_viewer.core.metadata_match import match_kind
    path, meta = pending
    self._pending_metadata = ()
    kind = match_kind(meta, ds)
    if kind is None:
        self._state.log(
            f"Pending processing metadata '{path.name}' discarded — "
                f"'{getattr(ds, 'name', '?')}' does not match it.", "WARN")
        return ()
    return path, meta, kind


def _sidecar_for_load(self: MainWindow, path) -> bool:
    """Whether the recipe beside *path* may be applied — asks if needed.

        Point 1: a sidecar is never restored silently. The question is asked
        **before** the data is read, so the answer is simply passed to the
        loader and nothing has to be undone afterwards. A sidecar named in the
        same drop is not asked about — dropping both files IS the answer.
        """
    from minflux_viewer.core.loader_metadata_recipe import read_metadata_sidecar
    found = read_metadata_sidecar(path)
    if found is None:
        return False
    side, meta = found
    key = str(side.resolve()).casefold()
    if key in self._batch_sidecars:
        self._batch_sidecars_used.add(key)
        self._state.log(
            f"'{side.name}' was dropped together with its data — applying "
                "it without asking.")
        return True
    from minflux_viewer.ui.dialogs.metadata_apply_dialog import LOAD, ask_load_sidecar
    choice = ask_load_sidecar(self, Path(path).name, side.name, meta)
    if choice != LOAD:
        self._state.log(
            f"Processing metadata '{side.name}' skipped by the user; "
                f"'{Path(path).name}' opened unprocessed.")
        return False
    return True


def _open_metadata_sidecar(self: MainWindow, path: str) -> None:
    """A recipe dropped on the main window: ask which dataset it acts on."""
    from minflux_viewer.core.loader_metadata_recipe import read_metadata_file
    from minflux_viewer.ui.dialogs.metadata_apply_dialog import APPLY, KEEP, ask_apply_metadata
    name = Path(path).name
    meta = read_metadata_file(path)
    if meta is None:
        self._state.log(
            f"'{name}' is not a readable processing metadata file.", "ERROR")
        self._status_label.setText(f"Skipped: {name}")
        return
    choice, index = ask_apply_metadata(self, name, meta, self._state.datasets)
    if choice == APPLY and index is not None:
        self.apply_metadata_to_dataset(index, meta, name)
        self._status_label.setText(f"Applied {name}.")
        return
    if choice == KEEP:
        self._remember_pending_metadata(path, meta)
        self._status_label.setText(f"Holding {name}.")
        return
    self._state.log(f"Processing metadata '{name}' cancelled.")
    self._status_label.setText(f"Cancelled: {name}")


def drop_file_on_dataset(self: MainWindow, idx: int, path: str) -> bool:
    """Apply a dropped file to the dataset at *idx*.  Returns True if handled.

        Dropping **onto a dataset** means "use this file on *that* dataset",
        which is a different verb from dropping on the main window ("open this").
        Four kinds are understood, all of them things that modify or annotate an
        existing dataset rather than create one:

        * a **filter preset** JSON → its rows are appended to that dataset's filter
        * a **ROI set** (native JSON, ImageJ ``.roi`` or a RoiSet ``.zip``) → the
          ROIs load into the ROI Manager retargeted to that dataset
        * a **metadata sidecar** JSON → its processing recipe (Z scaling factor / transform /
          filters) is applied to that dataset
        * a **TIFF** → mapped as a confocal signal, like *Map confocal signal…*

        Anything else is refused here and reported, rather than silently falling
        through to "open as a new dataset" — which would ignore the row the user
        aimed at.
        """
    if not (0 <= idx < len(self._state.datasets)):
        return False
    p = Path(path)
    ext = p.suffix.lower()
    ds = self._state.datasets[idx]
    if ext in (".tif", ".tiff"):
        return self._drop_confocal_tiff(idx, p)
    # ⚠ Compound suffix first: a ``.zarr.zip`` is a sealed dataset store,
    # not a RoiSet archive, and both end in ``.zip``.
    if p.name.lower().endswith(".zarr.zip"):
        ext = ".zarr.zip"
    elif ext in (".roi", ".zip"):
        return self._drop_roi_file(idx, p)
    if ext == ".json":
        # Resolve by content — the three JSON kinds share an extension.
        from minflux_viewer.core.filter_io import is_filter_json_file
        from minflux_viewer.core.roi import is_roi_json_file
        from minflux_viewer.core.save import is_metadata_json_file
        try:
            if is_roi_json_file(str(p)):
                return self._drop_roi_file(idx, p)
            if is_filter_json_file(str(p)):
                return self._drop_filter_file(idx, p)
            if is_metadata_json_file(str(p)):
                return self._drop_metadata_sidecar(idx, p)
        except Exception as exc:
            self._state.log(f"Drop on '{ds.name}': could not read '{p.name}': {exc}",
                            "ERROR")
            return False
        # Not one of the three annotation kinds — it may still be a data
        # file or an MBM companion, so let the open path decide rather than
        # refusing a .json the application can read.
    # Not an apply-to-a-dataset kind. If the application can open it at
    # all, open it -- the row is simply not a meaningful target for a data
    # file, which the Log says rather than leaving the drop looking broken.
    if ext in _SUPPORTED_EXTS or (p.is_dir() and ext == ".zarr"):
        self._state.log(
            f"Drop on '{ds.name}': '{p.name}' is a data file — opening it "
                "as a new dataset (the row it was dropped on is not used).")
        self._route_path(str(p))
        return True
    self._state.log(
        f"Drop on '{ds.name}': '{p.name}' cannot be applied to a dataset "
            "(drop it on the main window to open it).", "WARN")
    return False


def _drop_filter_file(self: MainWindow, idx: int, path: Path) -> bool:
    """Append a dropped filter preset to the row's dataset."""
    # The filter dialog is dataset-owned and keyed by the ACTIVE index, so
    # the dropped-on dataset has to become active for its rows to land there.
    self._state.set_active(idx)
    self._load_filter_json(str(path))
    self._state.log(
        f"Filters from '{path.name}' added to "
            f"'{self._state.datasets[idx].name}'.", dataset_idx=idx)
    return True


def _drop_roi_file(self: MainWindow, idx: int, path: Path) -> bool:
    """Load a dropped ROI set into the ROI Manager, targeting the row's dataset."""
    self._state.set_active(idx)       # _load_roi_json retargets to the active one
    self._load_roi_json(str(path))
    return True


def _drop_metadata_sidecar(self: MainWindow, idx: int, path: Path) -> bool:
    """Apply a dropped processing recipe to the row's dataset, once confirmed.

        The row already names the target, so this asks only *whether* to apply
        — never silently, because a recipe changes Z scaling, filters and ROIs.
        """
    from minflux_viewer.core.loader_metadata_recipe import read_metadata_file
    from minflux_viewer.core.metadata_match import is_snapshot_recipe, match_kind, recipe_summary
    from minflux_viewer.ui.dialogs.metadata_apply_dialog import SNAPSHOT_WARNING
    ds = self._state.datasets[idx]
    meta = read_metadata_file(path)
    if meta is None:
        self._state.log(
            f"'{path.name}' is not a readable processing metadata file.",
            "ERROR")
        return False
    text = (f"Apply the processing in '{path.name}' to '{ds.name}'?\n\n"
                f"{recipe_summary(meta)}")
    if match_kind(meta, ds) is None:
        text += ("\n\nThis recipe does not identify this dataset — it was "
                     "saved for other data. Filters and ROIs are added to what "
                     "the dataset already has.")
    if is_snapshot_recipe(meta):
        text += f"\n\n⚠ {SNAPSHOT_WARNING}"
    answer = QMessageBox.question(
        self, "Apply processing metadata", text,
        QMessageBox.StandardButton.Apply | QMessageBox.StandardButton.Cancel,
        QMessageBox.StandardButton.Apply,
    )
    if answer != QMessageBox.StandardButton.Apply:
        self._state.log(f"Processing metadata '{path.name}' cancelled.")
        return True
    return self.apply_metadata_to_dataset(idx, meta, path.name)


def _drop_confocal_tiff(self: MainWindow, idx: int, path: Path) -> bool:
    """Map a dropped TIFF onto the row's dataset as a confocal signal attribute."""
    from minflux_viewer.core.confocal_mapping import candidates_from_tiff
    from minflux_viewer.ui.analysis.confocal_mapping_dialog import (
        ConfocalMappingOptionsDialog,
        apply_confocal_mapping_options,
    )
    ds = self._state.datasets[idx]
    try:
        candidates = candidates_from_tiff(path, ds)
    except Exception as exc:
        self._state.log(f"Confocal TIFF '{path.name}' on '{ds.name}': {exc}", "ERROR")
        QMessageBox.critical(self, "Map confocal signal", str(exc))
        return False
    if not candidates:
        self._state.log(
            f"Confocal TIFF '{path.name}': no plane overlaps '{ds.name}'.", "WARN")
        QMessageBox.information(
            self, "Map confocal signal",
            f"'{path.name}' has no calibrated image plane overlapping "
                f"'{ds.name}'.\n\nA TIFF must carry a pixel size (OME / ImageJ / "
                "TIFF resolution tags) and cover the dataset's coordinates.",
        )
        return False
    dialog = ConfocalMappingOptionsDialog(
        candidates, self,
        reserved_names=set(ds.attr.keys()) | set(ds.mfx_raw.keys()),
    )
    if dialog.exec() != QDialog.DialogCode.Accepted:
        return True                      # handled: the user cancelled
    try:
        results = apply_confocal_mapping_options(ds, path, dialog.options(), self)
    except Exception as exc:
        self._state.log(f"Confocal signal mapping failed for '{ds.name}': {exc}",
                        "ERROR")
        QMessageBox.warning(self, "Map confocal signal", str(exc))
        return False
    if results:
        self._state.notify_attributes_changed(idx)
        for result in results:
            self._state.log(
                f"Mapped confocal signal '{result.attribute_name}' from "
                    f"'{path.name}' onto '{ds.name}': {result.finite_count:,}/"
                    f"{result.total_count:,} localizations in bounds.",
                dataset_idx=idx,
            )
    return True
