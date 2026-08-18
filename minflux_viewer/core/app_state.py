"""
minflux_viewer.core.app_state
==============================
Central application state — the Python/Qt equivalent of the MATLAB ``app``
object.

Holds
-----
* The list of loaded datasets (``datasets``)
* The active dataset index   (``active_idx``)
* Application preferences    (``prefs``)

Emits Qt signals when state changes so every UI component can react without
being directly coupled to any other component.
"""

from __future__ import annotations

import copy
import json
from contextlib import contextmanager
from pathlib import Path
from typing import TYPE_CHECKING

from PyQt6.QtCore import QObject, QSettings, pyqtSignal

from ..colors import (
    DEFAULT_COLOR_PREFS,
    changed_color_paths,
    configure_colors,
    normalize_color_preferences,
    normalize_rgba,
)
from .dataset import MinfluxDataset

if TYPE_CHECKING:
    pass


def format_progress_bar(fraction: float, *, width: int = 10, done: bool = False) -> str:
    """Render an ASCII progress bar like ``==========  42 % ==========``."""
    bar = "=" * width
    if done:
        return f"{bar}  DONE  {bar}"
    pct = int(round(max(0.0, min(1.0, fraction)) * 100.0))
    return f"{bar}  {pct:3d} % {bar}"


class _TaskProgress:
    """Handle yielded by :meth:`AppState.task` — call ``update(fraction)``."""

    def __init__(self, state: "AppState", header: str = "") -> None:
        self._state = state
        self._header = header
        self._last_pct = -1

    def update(self, fraction: float) -> None:
        pct = int(round(max(0.0, min(1.0, fraction)) * 100.0))
        if pct != self._last_pct:                  # throttle: only on % change
            self._last_pct = pct
            self._state.log_progress(format_progress_bar(fraction))
            if self._header:                       # mirror into the status bar
                self._state.status_progress(self._header, fraction)


# ---------------------------------------------------------------------------
# Defaults
# ---------------------------------------------------------------------------

#: How many recent files are remembered on disk (the menu shows only
#: ``num_file_history`` of these). ~0.1-0.27 MB of paths — negligible.
MAX_RECENT_REMEMBERED: int = 1000

DEFAULT_PREFS: dict = {
    "file": {
        "num_file_history": 5,
        "keep_last_folder": True,
        "default_folder": str(Path.home()),
        "recent_files": [],
        "confirm_overwrite": True,
        "close_paraview_on_exit": True,
        "check_updates_on_startup": False,   # opt-in: no startup network call by default
        "paraview_path": "",
        "temp_folder": "",           # app-wide temp dir; empty = use system temp
    },
    "data": {
        "iter_load": "last",            # "last" | "all"
        "load_efc_cfr": True,
        "load_all_dcr": False,
        # Fixed 0.67 is the normal MINFLUX z correction.  Anisotropy-based
        # estimation is an explicit opt-in in Preferences.
        "compute_rimf": False,
        "compute_loc_prec": True,
        "loc_precision_method": "stddev",  # "stddev" | "crlb" | "frc"
        "compute_local_density": True,
        "local_density_radius": 100,
        # 2D/3D threshold: datasets whose Z range is below this many nm
        # are treated as 2D (Z values forced to zero).
        "enforce_min_z_range": True,
        "min_z_range_nm": 5.0,
        # auto-open windows on load:
        "show_data_info": True,
        "show_dataset_manager": False,
        "show_attr_plot": False,
        "show_scatter": False,
        "show_histogram": False,
        "show_render": True,
        # When saving/exporting a data file (defaults that shrink the Save dialog):
        "export_formats": ["mat", "npy", "npz", "json", "csv", "zarr", "msr"],  # offered in the dialog
        "export_content": ["raw", "snapshot"],   # raw canonical / processed snapshot
        "export_include_attrs": True,            # original properties & attributes
        "export_include_derived": False,         # freeze derived attributes (snapshot)
        "export_include_recipe": True,           # write the metadata sidecar
        "export_filter_mode": "flag",            # "apply" (drop rows) | "flag" (ftr col)
    },
    "plot": {
        "rimf_value": 0.67,
        "use_fixed_rimf": False,
        "render_pixel_size": 2,
        "render_cmap": "hot",
        "render_xy_origin": "top_left",     # "top_left" | "bottom_left"
        "render_method": "basic",            # smoothed histogram
        "scatter_color_by": "tid",
        "scatter_cmap": "jet",
        "scatter_xy_origin": "top_left",    # "top_left" | "bottom_left"
        # Application-owned named gradients created from the LUT dialog.
        # Values are JSON-compatible ``{"stops": [[position, RGBA], ...]}``.
        "custom_colormaps": {},
        "roi_highlight_in_roi": True,   # highlight in-ROI data on the drawing view
        "roi_sync_highlight": True,     # highlight in-ROI data on other views too
        "roi_edge_size": 1,
        "roi_edit_widget_size": 8,
        "filter_bounds_size": 1,
        # Histogram Plot: which trace read-outs appear in the "As" dropdown...
        "histogram_values": ["trace mean", "trace median"],
        # ...and which pooled modes appear in its "Iter" dropdown (see
        # core.iteration.POOL_KEYS). All four by default.
        "histogram_iterations": ["flatten", "stacked", "sum", "average"],
        # Last-used manual overlay-alignment controls. Translation is physical nm.
        "render_alignment_translation_nm": 1.0,
        "render_alignment_rotation_deg": 0.1,
        "scatter_alignment_translation_nm": 1.0,
        "scatter_alignment_rotation_deg": 0.1,
        "confocal_alignment_translation_px": 0.5,
        "confocal_alignment_rotation_deg": 0.1,
    },
    "colors": copy.deepcopy(DEFAULT_COLOR_PREFS),
    "plugin": {
        "msr_export_folder": "",
        "msr_last_open_folder": "",
        "msr_remember_last": True,
    },
    "shortcuts": {
        "focus_main_window": "Shift+V",
        "close_window": "W",
        "show_info": "Ctrl+I",
        "duplicate": "Shift+D",
        "filter": "Shift+F",
        "next_window": "Tab",
        "previous_window": "Shift+Tab",
        "next_dataset": "Ctrl+Tab",
        "previous_dataset": "Ctrl+Shift+Tab",
        "open": "Ctrl+O",
        "open_msr": "",
        "save": "Ctrl+S",
        "render": "Ctrl+R",
        "brightness_contrast": "Shift+C",
        "attribute_plot": "Ctrl+1",
        "attribute_histogram": "Ctrl+2",
        "scatter_plot": "Ctrl+3",
        "log": "Ctrl+L",
        "console": "Ctrl+Shift+L",
        "preferences": "Shift+P",
        "dataset_manager": "Ctrl+D",
    },
    "attributes": {
        "enabled": [
            "vld", "itr", "tid", "loc", "efo", "cfr", "dcr", "tim", "sta",
        ],
        "computed": [
            "idx", "siz", "dst", "dur", "len", "spd", "dt", "tim_trace", "den",
        ],
    },
    "mbm_handling": {
        "only_used_for_drift_correction": False,
        "minimum_localizations_per_bead": 10,
        "average_method": "median",
        "average_occurrence_count": 10,
        "transform_type": "rigid XY + translational Z",
        "align_to_channel": "first",
    },
    "measurements": {
        "area": True,
        "mean": True,
        "standard_deviation": False,
        "modal": False,
        "min_max": False,
        "centroid": False,
        "center_of_mass": False,
        "perimeter": False,
        "bounding_rectangle": False,
        "fit_ellipse": False,
        "shape_descriptors": False,
        "feret": False,
        "integrated_density": False,
        "median": False,
        "skewness": False,
        "kurtosis": False,
        "area_fraction": False,
        "stack_position": False,
        "limit_to_threshold": False,
        "display_label": False,
        "invert_y": False,
        "scientific_notation": False,
        "add_to_overlay": False,
        "nan_empty_cells": False,
        "redirect_to": "None",
        "decimal_places": 3,
    },
}


def _merge(saved: dict, defaults: dict, _path: tuple[str, ...] = ()) -> dict:
    """Recursively fill missing keys from *defaults* into *saved*.

    *defaults* is deep-copied: a plain ``dict()`` is shallow, so every key the
    user has not saved would hand back the **live** ``DEFAULT_PREFS`` sub-dict /
    list, and editing preferences would then mutate the module-level defaults
    for the rest of the process.
    """
    result = copy.deepcopy(defaults)
    for k, v in saved.items():
        path = (*_path, str(k))
        if path == ("colors", "solid") and isinstance(v, dict):
            # The ordered solid mapping is a user-managed collection. Missing
            # keys mean deleted colors, not old settings needing new defaults.
            result[k] = copy.deepcopy(v)
        elif k in result and isinstance(result[k], dict) and isinstance(v, dict):
            result[k] = _merge(v, result[k], path)
        else:
            result[k] = v
    return result


#: Every one-shot ``prefs["_migrations"]`` key, in application order.
#:
#: A **fresh** preference set starts with all of them recorded as applied (see
#: :func:`default_prefs`). A migration exists to rewrite *saved* preferences that
#: predate a layout change; ``DEFAULT_PREFS`` already **is** the current layout, so
#: running one against it can only contradict it. That is not hypothetical: the
#: fresh-install path is ``_migrate_prefs(deepcopy(DEFAULT_PREFS))``, and a fresh
#: dict carries no ``_migrations`` key, so every block below used to fire at once
#: and overwrite the declared defaults — ``v036`` forced ``use_fixed_rimf`` back to
#: ``True`` no matter what ``DEFAULT_PREFS`` said, and ``v021`` sets ``compute_rimf``
#: to ``True`` against a ``False`` default (saved only by ``v036`` running after it).
#: Changing a default in ``DEFAULT_PREFS`` must be enough on its own.
_MIGRATION_KEYS: tuple[str, ...] = (
    "v021_compute_show_defaults",
    "v035_update_check_optin",
    "v036_fixed_rimf_default",
    "v037_metric_overlay_alignment_steps",
    "v041_global_rgba_colours",
)


def default_prefs() -> dict:
    """A fresh preference set: ``DEFAULT_PREFS`` with every migration pre-recorded.

    Used when nothing is saved yet, so a new install gets exactly the declared
    defaults. See :data:`_MIGRATION_KEYS` for why the migrations are skipped.
    """
    prefs = copy.deepcopy(DEFAULT_PREFS)
    prefs["_migrations"] = {key: True for key in _MIGRATION_KEYS}
    return _migrate_prefs(prefs)


def _migrate_prefs(prefs: dict) -> dict:
    """Move older built-in defaults to the current shortcut layout.

    Only the guarded ``_migrations`` blocks below rewrite values; everything before
    them is ``setdefault``-style and is a no-op against a current preference set.
    """
    shortcuts = prefs.setdefault("shortcuts", {})
    shortcuts.setdefault("focus_main_window", "Shift+V")
    if shortcuts.get("open_msr") == "Ctrl+Shift+O":
        shortcuts["open_msr"] = ""
    shortcuts.setdefault("brightness_contrast", "Shift+C")
    shortcuts.setdefault("preferences", "Shift+P")
    shortcuts.setdefault("dataset_manager", "Ctrl+D")
    if shortcuts.get("show_info") == "I":
        shortcuts["show_info"] = "Ctrl+I"
    if shortcuts.get("attribute_plot") == "Ctrl+3":
        shortcuts["attribute_plot"] = "Ctrl+1"
    if shortcuts.get("scatter_plot") == "Ctrl+1":
        shortcuts["scatter_plot"] = "Ctrl+3"
    attrs = prefs.setdefault("attributes", {})
    attrs.setdefault("computed", ["idx", "siz", "dst", "dur", "len", "spd", "dt", "tim_trace", "den"])
    enabled = list(attrs.get("enabled", []))
    if "sta" not in enabled:
        enabled.append("sta")
    attrs["enabled"] = enabled
    computed = ["siz" if name == "nLoc" else name for name in attrs.get("computed", [])]
    for name in ("idx", "siz", "dst", "dur", "len", "spd", "dt", "tim_trace", "den"):
        if name not in computed:
            computed.append(name)
    attrs["computed"] = computed

    migrations = prefs.setdefault("_migrations", {})
    if not migrations.get("v021_compute_show_defaults"):
        data = prefs.setdefault("data", {})
        data["compute_rimf"] = True
        data["compute_loc_prec"] = True
        data["compute_local_density"] = True
        data["show_data_info"] = True
        data["show_attr_plot"] = False
        data["show_render"] = True
        migrations["v021_compute_show_defaults"] = True
    # The on-startup update check is now opt-in. The previous default was ON and
    # got persisted implicitly (not an explicit choice), so reset it once so no
    # install reaches out to GitHub on startup unless the user re-enables it.
    if not migrations.get("v035_update_check_optin"):
        prefs.setdefault("file", {})["check_updates_on_startup"] = False
        migrations["v035_update_check_optin"] = True
    # Project-wide RIMF policy: ordinary real MINFLUX loads use the established
    # 0.67 z factor.  Estimation remains available as an explicit opt-in.
    if not migrations.get("v036_fixed_rimf_default"):
        prefs.setdefault("data", {})["compute_rimf"] = False
        plot = prefs.setdefault("plot", {})
        plot["use_fixed_rimf"] = True
        plot["rimf_value"] = 0.67
        migrations["v036_fixed_rimf_default"] = True
    if not migrations.get("v037_metric_overlay_alignment_steps"):
        plot = prefs.setdefault("plot", {})
        plot.pop("render_alignment_translation_px", None)
        plot["render_alignment_translation_nm"] = 1.0
        for key in ("render_alignment_rotation_deg", "scatter_alignment_rotation_deg"):
            if float(plot.get(key, 0.5)) == 0.5:
                plot[key] = 0.1
        migrations["v037_metric_overlay_alignment_steps"] = True
    if not migrations.get("v041_global_rgba_colours"):
        # Consolidate the old name + separate-alpha fields into the one RGBA
        # registry.  The merge step has already supplied the new defaults, while
        # saved legacy keys remain available here for a lossless one-time move.
        plot = prefs.setdefault("plot", {})
        colors = normalize_color_preferences(prefs.get("colors", {}))

        def legacy_rgba(name_key: str, fallback, alpha: int = 255):
            rgba = normalize_rgba(plot.get(name_key, fallback), fallback)
            return [rgba[0], rgba[1], rgba[2], max(0, min(255, int(alpha)))]

        if "roi_color" in plot or "roi_transparency" in plot:
            transparency = max(0, min(100, int(plot.get("roi_transparency", 50))))
            colors["viewer"]["roi"] = legacy_rgba(
                "roi_color", "Yellow", round(255 * (100 - transparency) / 100)
            )
        if "filter_range_color" in plot or "filter_range_alpha" in plot:
            opacity = max(0, min(100, int(plot.get("filter_range_alpha", 45))))
            colors["viewer"]["filter_range"] = legacy_rgba(
                "filter_range_color", "Green", round(255 * opacity / 100)
            )
        if "filter_bounds_color" in plot:
            colors["viewer"]["filter_bounds"] = legacy_rgba(
                "filter_bounds_color", "Green"
            )
            colors["viewer"]["filter_text"] = list(colors["viewer"]["filter_bounds"])
        old_overlay = plot.get("overlay_colors")
        if isinstance(old_overlay, (list, tuple)):
            for index, value in enumerate(old_overlay[:6]):
                colors["viewer"]["overlay"][index] = list(normalize_rgba(value))

        prefs["colors"] = normalize_color_preferences(colors)
        for key in (
            "attr_cmap", "roi_color", "roi_transparency", "filter_range_color",
            "filter_range_alpha", "filter_bounds_color", "overlay_colors",
        ):
            plot.pop(key, None)
        migrations["v041_global_rgba_colours"] = True
    return prefs


# ---------------------------------------------------------------------------
# AppState
# ---------------------------------------------------------------------------

class AppState(QObject):
    """
    Single source of truth for all application state.

    Signals
    -------
    dataset_added(int)
        Emitted after a dataset is appended; carries its new index.
    dataset_removed(int)
        Emitted after a dataset is removed; carries the index it *had*.
    active_changed(int)
        Emitted when the active dataset changes; carries the new index.
    filter_changed(int)
        Emitted when ``dataset[idx].filter_mask`` is modified.
    attributes_changed(int)
        Emitted when user-visible columns are added to an existing dataset.
    roi_selection_changed(int)
        Emitted when a dataset's cached ROI selection mask is modified.
    status_message(str)
        Short text for the main-window status bar.
    """

    dataset_added   = pyqtSignal(int)
    dataset_removed = pyqtSignal(int)
    active_changed  = pyqtSignal(int)
    filter_changed  = pyqtSignal(int)
    attributes_changed = pyqtSignal(int)
    calibration_changed = pyqtSignal(int)   # RIMF / z-scaling changed for a dataset
    roi_selection_changed = pyqtSignal(int)
    colors_changed = pyqtSignal(object)  # {paths, previous, current}
    status_message  = pyqtSignal(str)
    log_message     = pyqtSignal(str, str)  # (message, level)
    progress_log    = pyqtSignal(str, bool)  # (bar text, final) — refreshing line

    # ------------------------------------------------------------------
    def __init__(self, parent: QObject | None = None) -> None:
        super().__init__(parent)
        self._datasets: list[MinfluxDataset] = []
        self._active_idx: int | None = None
        self.prefs: dict = self._load_prefs()
        configure_colors(self.prefs)
        from ..colormaps import configure_custom_colormaps
        configure_custom_colormaps(
            self.prefs.get("plot", {}).get("custom_colormaps", {})
        )

        # Processing-history journal — used by the Generate Method Text plugin.
        from .processing_journal import ProcessingJournal
        self.journal = ProcessingJournal()
        # Retained log history (used by Generate Method Text to let the user pick
        # the events relevant to a dataset). Each entry is tagged with the active
        # dataset at emit time, so multi-dataset sessions stay attributable.
        self._log_history: list[dict] = []
        self._log_history_max = 5000
        from .roi import RoiStore
        self.rois = RoiStore(self)
        from ..scripting import create_facade
        self.mfv = create_facade(self)
        # Batch importers can flip this while adding multiple datasets, then
        # open one grouped render view after the batch is complete.
        self.suspend_auto_render: bool = False

    # ------------------------------------------------------------------
    # Read-only accessors
    # ------------------------------------------------------------------

    @property
    def datasets(self) -> list[MinfluxDataset]:
        return self._datasets

    @property
    def active_idx(self) -> int | None:
        return self._active_idx

    @property
    def active_dataset(self) -> MinfluxDataset | None:
        if self._active_idx is None:
            return None
        return self._datasets[self._active_idx]

    def __len__(self) -> int:
        return len(self._datasets)

    def __getitem__(self, idx: int) -> MinfluxDataset:
        return self._datasets[idx]

    # ------------------------------------------------------------------
    # Dataset management
    # ------------------------------------------------------------------

    def add_dataset(self, dataset: MinfluxDataset) -> int:
        """
        Append *dataset*, make it active, and return its index.

        If a dataset with the same file path is already loaded, switches
        to it instead of loading a duplicate.
        """
        for i, ds in enumerate(self._datasets):
            if ds.file.path == dataset.file.path:
                self.set_active(i)
                self.status_message.emit(f"Already loaded: {dataset.name}")
                self.log(f"Already loaded: {dataset.name}", "WARN")
                return i

        self._datasets.append(dataset)
        idx = len(self._datasets) - 1
        # IMPORTANT: set_active BEFORE dataset_added so handlers that query
        # active_dataset (e.g. MainWindow._show_render) see a valid active idx.
        self._active_idx = idx
        self.dataset_added.emit(idx)
        self.active_changed.emit(idx)
        self.status_message.emit(f"Active: {dataset.summary()}")
        recent_path = dataset.file.recent_path or dataset.file.path
        self._record_recent(recent_path)
        self.log(
            f"Loaded: {dataset.name}  |  {dataset.prop.num_loc:,} loc  |  "
            f"{dataset.prop.num_traces:,} traces  |  {dataset.prop.num_dim}D"
        )
        # Note MINFLUX kind + missing quality attributes (efo/cfr/dcr/fbg).
        try:
            from .dataset_kind import QUALITY_ATTRS, is_minflux
            if not is_minflux(dataset):
                self.log(
                    f"'{dataset.name}' is a non-MINFLUX dataset (no trace id) — "
                    "MINFLUX-specific analyses are disabled.", "WARN")
            else:
                missing = [a for a in QUALITY_ATTRS if dataset.attr.get(a) is None]
                if missing:
                    self.log(
                        f"'{dataset.name}': MINFLUX quality attributes missing "
                        f"({', '.join(missing)}) — quality filtering on these is "
                        "unavailable.", "WARN")
        except Exception:
            pass
        # Record to processing journal for the methods-text generator
        try:
            self.journal.add(
                "load",
                f"Loaded dataset '{dataset.name}'",
                num_loc=int(dataset.prop.num_loc),
                num_traces=int(dataset.prop.num_traces),
                num_dim=int(dataset.prop.num_dim),
                source=str(dataset.file.path),
            )
        except Exception:
            pass
        # Parseable, dataset-tagged load summary for the method-text generator.
        try:
            md = dataset.metadata
            container = md.get("source_format")
            ver = md.get("source_version", "?")
            ver_str = f"{ver} ({container})" if container else str(ver)
            n_itr = md.get("raw_num_itr", 1)
            valid = md.get("valid_num_loc", dataset.prop.num_loc)
            self.log(
                f"Loaded dataset '{dataset.name}': {valid:,} valid locs, {n_itr} iteration(s), "
                f"{dataset.prop.num_traces} trace(s), {dataset.prop.num_dim}D [{ver_str}].",
                dataset_idx=idx,
            )
        except Exception:
            pass
        return idx

    def remove_dataset(self, idx: int) -> None:
        """Remove the dataset at *idx* and update the active index."""
        if not (0 <= idx < len(self._datasets)):
            return

        self._datasets.pop(idx)
        self.dataset_removed.emit(idx)

        if not self._datasets:
            self._active_idx = None
            self.status_message.emit("No data loaded.")
            self.log("All datasets removed.", "INFO")
            return

        # Clamp active index if it pointed at or past the removed entry
        if self._active_idx is not None:
            new_active = min(self._active_idx, len(self._datasets) - 1)
            if new_active != self._active_idx:
                self._active_idx = new_active
                self.active_changed.emit(new_active)

    def set_active(self, idx: int) -> None:
        """Make the dataset at *idx* the active one."""
        if not (0 <= idx < len(self._datasets)):
            return
        if self._active_idx == idx:
            return
        self._active_idx = idx
        self.active_changed.emit(idx)
        self.status_message.emit(f"Active: {self._datasets[idx].summary()}")

    def notify_filter_changed(self, idx: int | None = None) -> None:
        """
        Notify all views that the filter mask has changed.
        Call this after modifying ``dataset.filter_mask`` directly.
        """
        if idx is None:
            idx = self._active_idx
        if idx is not None:
            self.filter_changed.emit(idx)

    def notify_attributes_changed(self, idx: int | None = None) -> None:
        """Notify attribute/filter windows that dataset columns changed."""
        if idx is None:
            idx = self._active_idx
        if idx is not None:
            self.attributes_changed.emit(idx)

    def notify_roi_selection_changed(self, idx: int | None = None) -> None:
        """Notify views that cached ROI selection masks changed."""
        if idx is None:
            idx = self._active_idx
        if idx is not None:
            self.roi_selection_changed.emit(idx)

    def notify_calibration_changed(self, idx: int | None = None) -> None:
        """Notify views that a dataset's calibration (RIMF / z-scaling) changed
        so geometry-dependent displays re-pull the RIMF-corrected coordinates."""
        if idx is None:
            idx = self._active_idx
        if idx is not None:
            self.calibration_changed.emit(idx)

    def log(
        self,
        message: str,
        level: str = "INFO",
        *,
        dataset_idx: int | None = None,
        method_data: dict | None = None,
    ) -> None:
        """
        Post a message to the log window.

        Log messages go to the structured event Log window only.
        Raw stdout/stderr (including ``print()`` and tracebacks) is shown
        in the separate Console window.

        Parameters
        ----------
        message:
            Human-readable description of the event.
        level:
            One of "INFO", "WARN", "ERROR", "DEBUG".
        method_data:
            Optional structured, run-specific provenance retained with the log
            event for Generate Method Text. It is not shown in the Log window.
        """
        from datetime import datetime
        idx = self._active_idx if dataset_idx is None else dataset_idx
        ds = self._datasets[idx] if (idx is not None and 0 <= idx < len(self._datasets)) else None
        event = {
            "time": datetime.now(),
            "level": str(level),
            "message": str(message),
            "dataset_idx": idx,
            "dataset_name": ds.name if ds is not None else None,
        }
        if method_data is not None:
            event["method_data"] = copy.deepcopy(method_data)
        self._log_history.append(event)
        if len(self._log_history) > self._log_history_max:
            del self._log_history[: len(self._log_history) - self._log_history_max]
        self.log_message.emit(message, level)

    @property
    def log_history(self) -> list[dict]:
        """All retained log events (oldest first), each tagged with the active
        dataset at emit time."""
        return list(self._log_history)

    def log_progress(self, text: str, *, final: bool = False) -> None:
        """Update the single refreshing progress line in the Log window.

        ``final=True`` freezes the line (e.g. the DONE bar) so the next progress
        starts fresh. Prefer :meth:`task` for the common header + bar pattern.
        """
        self.progress_log.emit(text, final)

    def status_progress(self, header: str, fraction: float | None = None) -> None:
        """Push a concise progress line to the **main-window status bar** so a
        long computation can be traced there (not just in the Log window).

        ``fraction`` (0..1) appends a percentage; omit it for an indeterminate
        "<header>…" line. Safe to call from a signal slot on the GUI thread."""
        head = str(header).rstrip("… ").rstrip(".")
        if fraction is None:
            self.status_message.emit(f"{head}…")
        else:
            pct = int(round(max(0.0, min(1.0, fraction)) * 100.0))
            self.status_message.emit(f"{head}…  {pct}%")

    @contextmanager
    def task(self, message: str):
        """Log *message* as a header, then a refreshing ASCII progress bar.

        ::

            with state.task(f"Computing FRC of dataset {ds.name}…") as t:
                for i in range(n):
                    ...
                    t.update((i + 1) / n)        # → "==========  N % =========="
            # the bar is finalized to DONE (or FAILED on exception) on exit
        """
        header = str(message).rstrip("… ").rstrip(".")
        self.log(message)
        self.log_progress(format_progress_bar(0.0))
        self.status_progress(header)
        handle = _TaskProgress(self, header)
        try:
            yield handle
        except BaseException:
            self.log_progress("=" * 10 + "  FAILED  " + "=" * 10, final=True)
            self.status_message.emit(f"{header}: failed.")
            raise
        else:
            self.log_progress(format_progress_bar(1.0, done=True), final=True)
            self.status_message.emit(f"{header}: done.")

    # ------------------------------------------------------------------
    # Preferences
    # ------------------------------------------------------------------

    def _load_prefs(self) -> dict:
        qs = QSettings("EMBL-IC", "MinfluxViewer")
        raw = qs.value("prefs", None)
        if raw:
            try:
                return _migrate_prefs(_merge(json.loads(raw), DEFAULT_PREFS))
            except Exception:
                pass
        return default_prefs()

    def save_prefs(self) -> None:
        configure_colors(self.prefs)
        from ..colormaps import configure_custom_colormaps
        configure_custom_colormaps(
            self.prefs.get("plot", {}).get("custom_colormaps", {})
        )
        qs = QSettings("EMBL-IC", "MinfluxViewer")
        qs.setValue("prefs", json.dumps(self.prefs))

    def notify_color_preferences_changed(
        self, previous, *, solid_renames: dict[str, str] | None = None
    ) -> set[str]:
        """Normalize the current registry and emit one targeted-change payload."""
        old = normalize_color_preferences(previous)
        current = normalize_color_preferences(self.prefs.get("colors", {}))
        self.prefs["colors"] = current
        configure_colors(self.prefs)
        paths = changed_color_paths(old, current)
        if paths:
            self.colors_changed.emit({
                "paths": paths,
                "previous": old,
                "current": copy.deepcopy(current),
                "solid_renames": dict(solid_renames or {}),
            })
        return paths

    def apply_color_preferences(
        self, colors, *, solid_renames: dict[str, str] | None = None
    ) -> set[str]:
        """Persist a complete color draft and notify only affected consumers."""
        previous = normalize_color_preferences(self.prefs.get("colors", {}))
        self.prefs["colors"] = normalize_color_preferences(colors)
        self.save_prefs()
        return self.notify_color_preferences_changed(
            previous, solid_renames=solid_renames
        )

    def _record_recent(self, path: str) -> None:
        try:
            path_obj = Path(path)
        except (TypeError, ValueError):
            return
        if not path_obj.is_file():
            return
        recent = self.prefs["file"].setdefault("recent_files", [])
        if path in recent:
            recent.remove(path)
        recent.insert(0, path)
        # Remember up to MAX_RECENT_REMEMBERED silently; the menu only shows
        # ``num_file_history`` of them, so raising that preference later instantly
        # repopulates from this store. Only successful loads reach add_dataset (the
        # sole caller), so filter/ROI files never land here.
        self.prefs["file"]["recent_files"] = recent[:MAX_RECENT_REMEMBERED]
        if self.prefs["file"].get("keep_last_folder", True):
            self.prefs["file"]["default_folder"] = str(path_obj.parent)
        self.save_prefs()
