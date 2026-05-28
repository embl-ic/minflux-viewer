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

import json
from pathlib import Path
from typing import TYPE_CHECKING

from PyQt6.QtCore import QObject, QSettings, pyqtSignal

from .dataset import MinfluxDataset

if TYPE_CHECKING:
    pass


# ---------------------------------------------------------------------------
# Defaults
# ---------------------------------------------------------------------------

DEFAULT_PREFS: dict = {
    "file": {
        "num_file_history": 5,
        "keep_last_folder": True,
        "default_folder": str(Path.home()),
        "recent_files": [],
        "close_paraview_on_exit": True,
        "paraview_path": "",
    },
    "data": {
        "iter_load": "last",            # "last" | "all"
        "load_efc_cfr": True,
        "load_all_dcr": False,
        "compute_rimf": False,
        "compute_loc_prec": False,
        "compute_local_density": False,
        "local_density_radius": 100,
        "local_density_dimensions": 2,
        "local_density_method": "kdtree",
        "local_density_voxel_size": 100,
        "local_density_smooth_sigma": 1.0,
        # 2D/3D threshold: datasets whose Z range is below this many nm
        # are treated as 2D (Z values forced to zero).
        "enforce_min_z_range": True,
        "min_z_range_nm": 5.0,
        # auto-open windows on load:
        "show_data_info": True,
        "show_attr_plot": True,
        "show_scatter": False,
        "show_histogram": False,
        "show_render": False,
    },
    "plot": {
        "rimf_value": 0.67,
        "render_pixel_size": 2,
        "render_cmap": "hot",
        "scatter_cmap": "single color",
        "roi_color": "Yellow",
        "histogram_values": ["trace mean"],
    },
    "plugin": {
        "msr_temp_folder": "",       # empty = use system temp
        "msr_export_folder": "",
        "msr_last_open_folder": "",
        "msr_remember_last": True,
    },
    "shortcuts": {
        "close_window": "W",
        "show_info": "Ctrl+I",
        "duplicate": "Shift+D",
        "filter": "Shift+F",
        "next_window": "Tab",
        "previous_window": "Shift+Tab",
        "next_dataset": "Ctrl+Tab",
        "previous_dataset": "Ctrl+Shift+Tab",
        "open": "Ctrl+O",
        "open_msr": "Ctrl+Shift+O",
        "open_spreadsheet": "",
        "open_tiff": "",
        "save": "Ctrl+S",
        "render": "Ctrl+R",
        "attribute_plot": "Ctrl+1",
        "attribute_histogram": "Ctrl+2",
        "scatter_plot": "Ctrl+3",
        "log": "Ctrl+L",
        "console": "Ctrl+Shift+L",
    },
    "attributes": {
        "enabled": [
            "itr", "tid", "tim", "vld", "loc", "efo", "cfr", "dcr",
        ],
    },
    "mbm_handling": {
        "only_used_for_drift_correction": False,
        "minimum_localizations_per_bead": 10,
        "average_method": "median",
        "average_occurrence_count": 10,
        "transform_type": "rigid",
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


def _merge(saved: dict, defaults: dict) -> dict:
    """Recursively fill missing keys from *defaults* into *saved*."""
    result = dict(defaults)
    for k, v in saved.items():
        if k in result and isinstance(result[k], dict) and isinstance(v, dict):
            result[k] = _merge(v, result[k])
        else:
            result[k] = v
    return result


def _migrate_prefs(prefs: dict) -> dict:
    """Move older built-in defaults to the current shortcut layout."""
    shortcuts = prefs.setdefault("shortcuts", {})
    if shortcuts.get("show_info") == "I":
        shortcuts["show_info"] = "Ctrl+I"
    if shortcuts.get("attribute_plot") == "Ctrl+3":
        shortcuts["attribute_plot"] = "Ctrl+1"
    if shortcuts.get("scatter_plot") == "Ctrl+1":
        shortcuts["scatter_plot"] = "Ctrl+3"
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
    status_message(str)
        Short text for the main-window status bar.
    """

    dataset_added   = pyqtSignal(int)
    dataset_removed = pyqtSignal(int)
    active_changed  = pyqtSignal(int)
    filter_changed  = pyqtSignal(int)
    status_message  = pyqtSignal(str)
    log_message     = pyqtSignal(str, str)  # (message, level)

    # ------------------------------------------------------------------
    def __init__(self, parent: QObject | None = None) -> None:
        super().__init__(parent)
        self._datasets: list[MinfluxDataset] = []
        self._active_idx: int | None = None
        self.prefs: dict = self._load_prefs()

        # Processing-history journal — used by the Generate Method Text plugin.
        from .processing_journal import ProcessingJournal
        self.journal = ProcessingJournal()
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

    def log(self, message: str, level: str = "INFO") -> None:
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
        """
        self.log_message.emit(message, level)

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
        return _migrate_prefs(dict(DEFAULT_PREFS))

    def save_prefs(self) -> None:
        qs = QSettings("EMBL-IC", "MinfluxViewer")
        qs.setValue("prefs", json.dumps(self.prefs))

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
        limit = self.prefs["file"].get("num_file_history", 10)
        self.prefs["file"]["recent_files"] = recent[:limit]
        if self.prefs["file"].get("keep_last_folder", True):
            self.prefs["file"]["default_folder"] = str(path_obj.parent)
        self.save_prefs()
