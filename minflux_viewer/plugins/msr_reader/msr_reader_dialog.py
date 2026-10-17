from __future__ import annotations

import copy
import csv
import json
import os
import tempfile
import uuid
from pathlib import Path
from typing import Any

import numpy as np
from PyQt6.QtCore import Qt, QTimer
from PyQt6.QtWidgets import (
    QApplication,
    QCheckBox,
    QComboBox,
    QDialog,
    QFileDialog,
    QGroupBox,
    QHBoxLayout,
    QHeaderView,
    QLabel,
    QLineEdit,
    QMenu,
    QMessageBox,
    QProgressBar,
    QPushButton,
    QRadioButton,
    QSpinBox,
    QTreeWidget,
    QTreeWidgetItem,
    QVBoxLayout,
    QWidget,
)

from minflux_viewer.msr.session import (
    _StateLogAdapter as _StateLogAdapter,
)
from minflux_viewer.msr.session import (
    state_namespace_for as state_namespace_for,
)
from minflux_viewer.msr.session_export import (
    _FILENAME_FORBIDDEN as _FILENAME_FORBIDDEN,
)
from minflux_viewer.msr.session_export import (
    _RESERVED_STEMS as _RESERVED_STEMS,
)
from minflux_viewer.msr.session_export import (
    EXPORT_CONFLICT_MODES as EXPORT_CONFLICT_MODES,
)
from minflux_viewer.msr.session_export import (
    _entry_row_count as _entry_row_count,
)
from minflux_viewer.msr.session_export import (
    _export_entries as _export_entries,
)
from minflux_viewer.msr.session_export import (
    existing_export_targets as existing_export_targets,
)
from minflux_viewer.msr.session_export import (
    export_image_series as export_image_series,
)
from minflux_viewer.msr.session_export import (
    export_parsed_result as export_parsed_result,
)
from minflux_viewer.msr.session_export import (
    image_export_stem as image_export_stem,
)
from minflux_viewer.msr.session_export import (
    image_series_of as image_series_of,
)
from minflux_viewer.msr.session_export import (
    prepare_export_output_dir as prepare_export_output_dir,
)
from minflux_viewer.msr.session_export import (
    safe_export_stem as safe_export_stem,
)
from minflux_viewer.msr.session_export import (
    select_image_series as select_image_series,
)
from minflux_viewer.msr.session_export import (
    subset_struct_fields as subset_struct_fields,
)
from minflux_viewer.msr.session_export import (
    unique_export_stem as unique_export_stem,
)
from minflux_viewer.plugins.msr_reader.alignment_plot_window import (
    AlignmentPlotWindow as AlignmentPlotWindow,
)
from minflux_viewer.plugins.msr_reader.alignment_save_dialog import (
    AlignmentSaveDialog as AlignmentSaveDialog,
)
from minflux_viewer.plugins.msr_reader.array_preview_dialog import (
    ArrayPreviewDialog as ArrayPreviewDialog,
)
from minflux_viewer.plugins.msr_reader.batch_export_signals import (
    _BatchExportSignals as _BatchExportSignals,
)
from minflux_viewer.plugins.msr_reader.batch_export_task import (
    _BatchExportTask as _BatchExportTask,
)
from minflux_viewer.plugins.msr_reader.export_estimate_signals import (
    _ExportEstimateSignals as _ExportEstimateSignals,
)
from minflux_viewer.plugins.msr_reader.export_estimate_task import (
    _ExportEstimateTask as _ExportEstimateTask,
)
from minflux_viewer.plugins.msr_reader.field_dialog import (
    FieldDialog as FieldDialog,
)
from minflux_viewer.plugins.msr_reader.json_view_dialog import (
    JsonViewDialog as JsonViewDialog,
)
from minflux_viewer.plugins.msr_reader.metadata_helpers import (
    _is_numeric_dtype as _is_numeric_dtype,
)
from minflux_viewer.plugins.msr_reader.metadata_helpers import (
    _json_compact as _json_compact,
)
from minflux_viewer.plugins.msr_reader.metadata_helpers import (
    _json_leaf_text as _json_leaf_text,
)
from minflux_viewer.plugins.msr_reader.metadata_helpers import (
    _json_type_name as _json_type_name,
)
from minflux_viewer.plugins.msr_reader.metadata_helpers import (
    _populate_metadata_items as _populate_metadata_items,
)
from minflux_viewer.plugins.msr_reader.parse_worker import (
    _ParseWorker as _ParseWorker,
)
from minflux_viewer.plugins.msr_reader.parsed_export_task import (
    _ParsedExportTask as _ParsedExportTask,
)
from minflux_viewer.plugins.msr_reader.path_drop_line_edit import (
    PathDropLineEdit as PathDropLineEdit,
)
from minflux_viewer.plugins.msr_reader.plot_window import (
    PlotWindow as PlotWindow,
)
from minflux_viewer.plugins.msr_reader.reader_shared import (
    _ALIGN_CANCELLED as _ALIGN_CANCELLED,
)
from minflux_viewer.plugins.msr_reader.reader_shared import (
    _BATCH_STATUS_HEADER as _BATCH_STATUS_HEADER,
)
from minflux_viewer.plugins.msr_reader.reader_shared import (
    _IMAGE_NODE_TYPES as _IMAGE_NODE_TYPES,
)
from minflux_viewer.plugins.msr_reader.reader_shared import (
    _OPEN_INDIVIDUAL as _OPEN_INDIVIDUAL,
)
from minflux_viewer.plugins.msr_reader.reader_shared import (
    _READER_TITLE as _READER_TITLE,
)
from minflux_viewer.plugins.msr_reader.reader_shared import (
    RESTACK_DELAY_MS as RESTACK_DELAY_MS,
)
from minflux_viewer.plugins.msr_reader.viewer_alignment_dialog import (
    ViewerAlignmentDialog as ViewerAlignmentDialog,
)
from minflux_viewer.plugins.msr_reader.viewer_open_signals import (
    _ViewerOpenSignals as _ViewerOpenSignals,
)

# Sentinel: user cancelled the alignment dialog -> abort the whole
# "Open in MINFLUX viewer" action (distinct from None = "not applicable").
# The user chose to open each selected dataset as an individual (standalone)
# dataset — no overlay group, no channel alignment.
#: Window title of the reader; the batch run appends its percentage to it.
#: Header used for the main-window status line and the Log progress bar.
#: Tree-node types that stand for one OBF image series (``legacy_series`` is the
#: image-only-file tree, ``image_series`` the modern per-file/per-dataset groups).
#: Characters no mainstream filesystem accepts in a name (Windows is the strict
#: one; ``/`` and NUL cover POSIX). Everything else — spaces, braces, brackets,
#: parentheses — is kept so an exported image keeps the name the ``.msr`` uses.
#: Windows refuses these stems whatever the extension.
#: What to do when an export target already exists.
from minflux_viewer.ui.common.background_tasks import BackgroundTaskCancelled

#: How long after the reader closes its windows are restacked. Zero would
#: race the window manager's own post-close activation, which is the thing the
#: restack exists to undo; a few frames is imperceptible and reliably after it.


class MsrReaderDialog(QWidget):
    """
    MINFLUX .msr parser & exporter.  Near-verbatim PyQt6 port of
    D:/Git/MINFLUX_msr_reader/ui_qt/app.py with viewer integration.
    """

    def __init__(
        self,
        state=None,
        msr_path: str | None = None,
        parent: QWidget | None = None,
    ):
        # Deliberately a NON-owned top-level window: passing the main window as
        # the QWidget parent (with the Qt.Window flag) makes the OS keep this
        # dialog permanently above its owner in Z-order, so it sits in front of
        # every viewer window. The viewer plot windows avoid this by having no
        # parent; mirror that here and keep an explicit owner reference for the
        # few main-window lookups below (``self._owner``).
        super().__init__(None)
        self._owner = parent
        self.setWindowTitle(_READER_TITLE)
        self.setWindowFlags(Qt.WindowType.Window)
        # Batch export runs on a worker; these track it for progress + shutdown.
        self._batch_task = None
        self._batch_context: dict | None = None
        self._last_batch_pct = -1
        self._estimate_generation = 0
        self._estimate_task: _ExportEstimateTask | None = None
        # The background build behind "Open in MINFLUX viewer" (one at a time).
        self._viewer_open_task = None
        self._viewer_open_context: dict | None = None
        self._export_estimates: dict = {}
        self.resize(980, 860)
        self.setAttribute(Qt.WidgetAttribute.WA_DeleteOnClose)
        self._positioned = False

        self._state = state
        self._worker: _ParseWorker | None = None

        self.parsed: dict[str, Any] = {}
        # Per-instance parsed arrays (name -> array). Each reader keeps its OWN
        # copy so several dialogs (or a multi-file drop) don't clobber each other
        # through the shared module-global ``msr.state`` map. Read via
        # ``self._msr_state()``, which every data method uses in place of the
        # global ``state`` module.
        self._mfx_map: dict[str, Any] = {}
        self._mbm_map: dict[str, Any] = {}
        self._mbm_meta_map: dict[str, dict] = {}
        self.field_selection = {}
        # Image series to export / open as images: ``None`` = all of whatever
        # each file has, a set of raw OBF stack indices = exactly those,
        # empty set = none.
        self.image_field_selection: set | None = set()
        # gri-IDs the user excluded via "Show Beads Drift". Alignment always
        # starts from metadata-marked used beads; these exclusions further
        # narrow that set after the user applies a manual selection.
        self._bead_unchecked_gris = None
        # Remembered "Align channel" save-dialog checkbox states (persisted).
        self._alignment_save_options: dict = {}
        self.nodeinfo: dict[int, dict[str, Any]] = {}
        self.fullpath_by_item: dict[int, str] = {}
        self.datasetnode_info: dict[int, dict[str, Any]] = {}
        self._plot_windows: list[PlotWindow] = []
        self._active_plot_window: PlotWindow | None = None
        self._child_dialogs: list[QDialog] = []

        settings = self._load_settings(tempfile.gettempdir())
        self._build_ui(settings)

        if msr_path:
            # Opened for a specific file (drag-drop / File open) — it is a single
            # file, so force single-file mode and auto-parse it (no button click).
            if os.path.isfile(msr_path):
                self.mode_file.setChecked(True)
            self._on_mode_changed()
            self.input_path_edit.setText(msr_path)
            self._last_input_dir = str(Path(msr_path).parent)
            self._save_settings()
            self._start_parse_worker(msr_path, self._temp_dir())

    # ------------------------------------------------------------------
    # Settings persistence
    # ------------------------------------------------------------------

    def _settings_path(self) -> Path:
        return Path.home() / ".minflux_viewer_msr_settings.json"

    def _temp_dir(self) -> str:
        """App-wide temp folder (Preferences > File > Temporary folder), else the
        system temp dir. This is the single place a temp folder is defined."""
        prefs = getattr(self._state, "prefs", None)
        folder = ""
        if isinstance(prefs, dict):
            folder = str((prefs.get("file") or {}).get("temp_folder", "") or "").strip()
        return folder or tempfile.gettempdir()

    def _load_settings(self, system_tmp: str) -> dict[str, Any]:
        path = self._settings_path()
        defaults = {
            "out_dir": system_tmp,
            "last_input_dir": str(Path.home()),
            "preview_rows": 100,
            "preview_all_rows": False,
            "mode_folder": False,
            "plot_new_window": True,
            "recursive": False,
            "reproduce_tree": False,
            "formats": {"mat": True, "npy": False, "npz": False, "json": False, "csv": False, "zarr": False},
            "field_selection": {},
        }
        try:
            if path.exists():
                data = json.loads(path.read_text(encoding="utf-8"))
                if isinstance(data, dict):
                    merged = dict(defaults)
                    merged.update(data)
                    if not isinstance(merged.get("formats"), dict):
                        merged["formats"] = defaults["formats"]
                    if not isinstance(merged.get("field_selection"), dict):
                        merged["field_selection"] = {}
                    return merged
        except Exception:
            pass
        return defaults

    def _save_settings(self):
        def clean_selection(selection):
            out = {}
            for key, value in (selection or {}).items():
                out[key] = {
                    "checked": bool(value.get("checked", True)),
                    "mfx": None if value.get("mfx") is None else sorted(value.get("mfx") or []),
                    "mbm": None if value.get("mbm") is None else sorted(value.get("mbm") or []),
                }
            return out

        input_path = self.input_path_edit.text().strip()
        last_input_dir = input_path
        if input_path and Path(input_path).is_file():
            last_input_dir = str(Path(input_path).parent)
        elif not input_path:
            last_input_dir = getattr(self, "_last_input_dir", str(Path.home()))
        data = {
            "out_dir": self.out_dir_edit.text().strip() or self._temp_dir(),
            "last_input_dir": last_input_dir,
            "preview_rows": int(self.preview_rows_spin.value()),
            "preview_all_rows": bool(self.preview_all_rows_check.isChecked()),
            "mode_folder": bool(self.mode_folder.isChecked()),
            "plot_new_window": bool(self.plot_new_window_check.isChecked()),
            "recursive": bool(self.recursive_check.isChecked()),
            "reproduce_tree": bool(self.reproduce_tree_check.isChecked()),
            "batch_conflict": str(self.conflict_combo.currentData() or "skip"),
            "formats": {
                "mat": bool(self.fmt_mat.isChecked()),
                "npy": bool(self.fmt_npy.isChecked()),
                "json": bool(self.fmt_json.isChecked()),
                "csv": bool(self.fmt_csv.isChecked()),
                "zarr": bool(self.fmt_zarr.isChecked()),
            },
            "field_selection": clean_selection(self.field_selection),
            # None (= all images) must survive a round trip; an empty list means
            # "none selected", which is a different instruction.
            "image_field_selection": (
                None if self.image_field_selection is None
                else sorted(int(i) for i in self.image_field_selection)),
            "alignment_save": dict(self._alignment_save_options or {}),
        }
        try:
            self._settings_path().write_text(json.dumps(data, indent=2), encoding="utf-8")
            self._last_input_dir = last_input_dir
        except Exception as exc:
            self.log(f"[warn] could not save settings: {exc}")

    def showEvent(self, event):
        super().showEvent(event)
        # Non-owned top-level window. Cap it to the active screen (it is large —
        # 980×860 — and would otherwise open with its title bar above the top on
        # shorter / scaled displays), then drop it into the top corner OPPOSITE
        # the main window's side, so it clears the main window and the Log window
        # (which stack together on one side) instead of centering on top of them.
        # The main window is a wide strip, so both can't sit fully side by side on
        # a 1080p monitor; the opposite corner leaves them fully clear on a wide
        # monitor and only a thin top strip overlapping on a narrow one.
        if not self._positioned:
            self._positioned = True
            from minflux_viewer.ui.common.modeless import ensure_on_screen

            align = "top_left"
            try:
                from PyQt6.QtGui import QGuiApplication

                owner_win = self._owner.window() if self._owner is not None else None
                screen = owner_win.screen() if owner_win is not None else None
                screen = screen or QGuiApplication.primaryScreen()
                if owner_win is not None and screen is not None:
                    avail = screen.availableGeometry()
                    owner_cx = owner_win.frameGeometry().center().x()
                    align = "top_left" if owner_cx >= avail.center().x() else "top_right"
            except Exception:
                align = "top_left"
            ensure_on_screen(self, self._owner, align=align)

    def closeEvent(self, event):
        from minflux_viewer.ui.common.background_tasks import (
            retire_background_tasks,
            retire_qthreads,
        )

        self._save_settings()
        if self._worker is not None:
            retire_qthreads(
                [self._worker], signal_names=("progress", "succeeded", "failed")
            )
            self._worker = None
        # The dialog is WA_DeleteOnClose, so a running batch task must stop
        # signalling into it before Qt tears the widgets down.
        if self._batch_task is not None:
            retire_background_tasks(
                [self._batch_task],
                signal_names=("progress", "message", "finished"),
            )
            self._batch_task = None
            self._batch_context = None
        if self._estimate_task is not None:
            retire_background_tasks(
                [self._estimate_task], signal_names=("done",)
            )
            self._estimate_task = None
        # Closing the reader mid-open abandons the open: nothing has been added
        # to the viewer yet (datasets are added only when the build delivers).
        if self._viewer_open_task is not None:
            retire_background_tasks(
                [self._viewer_open_task],
                signal_names=("stage", "done", "failed", "cancelled", "finished"),
            )
            self._viewer_open_task = None
            if self._viewer_open_context is not None and self._state is not None:
                self._state.log_progress("Opening datasets: cancelled.", final=True)
                self._state.status_message.emit("Opening datasets: cancelled.")
            self._viewer_open_context = None
        for win in list(self._plot_windows):
            try:
                win.close()
            except Exception:
                pass
        self._plot_windows.clear()
        self._active_plot_window = None
        for dlg in list(self._child_dialogs):
            try:
                dlg.close()
            except Exception:
                pass
        self._child_dialogs.clear()
        super().closeEvent(event)

    # ------------------------------------------------------------------
    # UI construction
    # ------------------------------------------------------------------

    def _build_ui(self, settings):
        root = QVBoxLayout(self)

        input_type_row = QHBoxLayout()
        input_type_row.addWidget(QLabel("Input type:"))
        self.mode_file = QRadioButton("Single .msr file")
        self.mode_folder = QRadioButton("Folder (batch)")
        self.mode_folder.setChecked(bool(settings.get("mode_folder", False)))
        self.mode_file.setChecked(not self.mode_folder.isChecked())
        self.mode_file.toggled.connect(lambda _checked: self._on_mode_changed())
        input_type_row.addWidget(self.mode_file)
        input_type_row.addWidget(self.mode_folder)
        input_type_row.addStretch(1)
        root.addLayout(input_type_row)

        self.input_path_edit = PathDropLineEdit(accept="msr_or_dir")
        # Enter in the path field parses a single file (same as Browse / drop).
        self.input_path_edit.returnPressed.connect(self._auto_parse_single_input)
        self.input_path_edit.pathDropped.connect(self._on_input_path_dropped)
        self._last_input_dir = settings.get("last_input_dir") or str(Path.home())
        self.recursive_check = QCheckBox("recursive")
        self.recursive_check.setChecked(bool(settings.get("recursive", False)))
        self.recursive_check.setToolTip(
            "Search the folder and every sub-folder for .msr files.\n"
            "Off: only .msr files directly inside the folder are exported.")
        self.recursive_check.toggled.connect(lambda _checked: self._save_settings())
        root.addLayout(self._browse_row(
            "MSR file or folder:", self.input_path_edit, self.browse_input,
            extra=self.recursive_check))

        action_row = QHBoxLayout()
        # A single .msr file auto-parses (on drop / Browse / Enter). This button is
        # only meaningful in Folder (batch) mode, where it lets the user pick which
        # file in the folder to parse & preview.
        self._parse_button = QPushButton("Select a file to view content")
        self._parse_button.clicked.connect(self.on_select_view_file)
        action_row.addWidget(self._parse_button)
        action_row.addWidget(QLabel("Preview"))
        self.preview_all_rows_check = QCheckBox("all rows")
        self.preview_all_rows_check.setChecked(bool(settings.get("preview_all_rows", False)))
        action_row.addWidget(self.preview_all_rows_check)
        self.preview_rows_spin = QSpinBox()
        self.preview_rows_spin.setRange(1, 1_000_000)
        self.preview_rows_spin.setValue(int(settings.get("preview_rows", 100) or 100))
        action_row.addWidget(self.preview_rows_spin)
        self.preview_all_rows_check.toggled.connect(self._on_preview_all_rows_toggled)
        self.preview_rows_spin.valueChanged.connect(lambda _value: self._save_settings())
        self._on_preview_all_rows_toggled(self.preview_all_rows_check.isChecked())
        self.plot_new_window_check = QCheckBox("plot in new window")
        self.plot_new_window_check.setChecked(bool(settings.get("plot_new_window", True)))
        self.plot_new_window_check.toggled.connect(lambda _checked: self._save_settings())
        action_row.addWidget(self.plot_new_window_check)
        action_row.addStretch(1)
        root.addLayout(action_row)

        tree_group = QGroupBox("Parsed content")
        tree_layout = QVBoxLayout(tree_group)
        self.tree = QTreeWidget()
        self.tree.setColumnCount(4)
        self.tree.setHeaderLabels(["Path / Info", "Kind", "Shape", "DType"])
        self.tree.setSelectionMode(QTreeWidget.SelectionMode.ExtendedSelection)
        self.tree.header().setSectionResizeMode(0, QHeaderView.ResizeMode.ResizeToContents)
        self.tree.header().setSectionResizeMode(1, QHeaderView.ResizeMode.ResizeToContents)
        self.tree.header().setSectionResizeMode(2, QHeaderView.ResizeMode.ResizeToContents)
        self.tree.header().setSectionResizeMode(3, QHeaderView.ResizeMode.ResizeToContents)
        self.tree.setContextMenuPolicy(Qt.ContextMenuPolicy.CustomContextMenu)
        self.tree.customContextMenuRequested.connect(self._show_context_menu)
        tree_layout.addWidget(self.tree)
        root.addWidget(tree_group, 1)

        self.out_dir_edit = PathDropLineEdit(settings["out_dir"], accept="dir")
        self.out_dir_edit.pathDropped.connect(lambda _path: self._save_settings())
        self.reproduce_tree_check = QCheckBox("reproduce input folder structure")
        self.reproduce_tree_check.setChecked(bool(settings.get("reproduce_tree", False)))
        self.reproduce_tree_check.setToolTip(
            "Mirror each input file's path below the output folder.\n"
            "root/1st_dir/2nd_dir/test.msr  →  output/1st_dir/2nd_dir/test.mat\n"
            "Off: every file exports into its own <name> sub-folder directly "
            "under the output folder.")
        self.reproduce_tree_check.toggled.connect(lambda _checked: self._save_settings())
        self.conflict_label = QLabel("when file exists:")
        self.conflict_combo = QComboBox()
        # Skip first, and the default: a batch that re-runs over a folder should
        # fill in what is missing rather than rewrite hours of finished output.
        self.conflict_combo.addItem("skip", "skip")
        self.conflict_combo.addItem("overwrite", "overwrite")
        saved_conflict = str(settings.get("batch_conflict", "skip"))
        self.conflict_combo.setCurrentIndex(
            max(0, self.conflict_combo.findData(saved_conflict)))
        self.conflict_combo.setToolTip(
            "What a batch export does when an output file already exists.\n\n"
            "skip - keep the existing file and write only what is missing, "
            "so a re-run completes an interrupted batch.\n"
            "overwrite - replace it.\n\n"
            "A single-file export asks instead, and can also be cancelled.")
        self.conflict_combo.currentIndexChanged.connect(
            lambda _index: self._save_settings())
        root.addLayout(self._browse_row(
            "Output folder:", self.out_dir_edit, self.browse_out,
            extra=(self.reproduce_tree_check, self.conflict_label,
                   self.conflict_combo)))

        export_row = QHBoxLayout()
        field_button = QPushButton("Datasets / Fields included…")
        field_button.clicked.connect(self.on_fields_dialog)
        export_row.addWidget(field_button)
        export_row.addWidget(QLabel("Formats:"))
        self.fmt_mat = QCheckBox("MATLAB (.mat)")
        self.fmt_npy = QCheckBox("NumPy (.npy)")
        self.fmt_json = QCheckBox("JSON (.json)")
        self.fmt_csv = QCheckBox("Canonical table (.csv)")
        self.fmt_zarr = QCheckBox("Zarr (.zarr)")
        self.fmt_mat.setToolTip(
            "Compressed MATLAB data. Size depends on the values; the estimate "
            "samples MATLAB's real compressor."
        )
        self.fmt_npy.setToolTip(
            "Uncompressed binary structured array. Its size is predictable and "
            "loading is normally much faster than text."
        )
        self.fmt_json.setToolTip(
            "Plain-text record objects. Large MINFLUX acquisitions can expand to "
            "several GiB; formatting and parsing every number remains slow even "
            "though the writer/reader stream the file."
        )
        self.fmt_csv.setToolTip(
            "Plain-text decimal table. More compact than JSON but commonly much "
            "larger and slower than binary formats for all-iteration MINFLUX data."
        )
        self.fmt_zarr.setToolTip(
            "Write one self-contained MINFLUX Viewer Zarr v2 project for this "
            "MSR file. Every selected dataset becomes an overlay member with "
            "canonical raw MFX columns (including both "
            "DCR columns), MBM points and naming/used metadata, the search grid, "
            "native acquisition attributes and processing state. Selected images "
            "are embedded; DID-linked images are stored below their dataset. Re-open it by "
            "dragging the .zarr directory onto the viewer.\n\n"
            "The former per-dataset and _mfxdta.zarr companion exports are retired."
        )
        formats = settings.get("formats") or {}
        self.fmt_mat.setChecked(bool(formats.get("mat", True)))
        self.fmt_npy.setChecked(bool(formats.get("npy", False)))
        self.fmt_json.setChecked(bool(formats.get("json", False)))
        self.fmt_csv.setChecked(bool(formats.get("csv", False)))
        self.fmt_zarr.setChecked(bool(formats.get("zarr", False)))
        export_row.addWidget(self.fmt_mat)
        export_row.addWidget(self.fmt_npy)
        export_row.addWidget(self.fmt_json)
        export_row.addWidget(self.fmt_csv)
        export_row.addWidget(self.fmt_zarr)
        export_row.addStretch(1)
        root.addLayout(export_row)
        self._export_size_label = QLabel(
            "Estimated output: parse a file to calculate from its actual rows and fields."
        )
        self._export_size_label.setWordWrap(True)
        self._export_size_label.setTextInteractionFlags(
            Qt.TextInteractionFlag.TextSelectableByMouse
        )
        self._export_size_label.setToolTip(
            "NumPy is calculated from the exact dtype. CSV/JSON are sampled from "
            "the actual decimal serialization; MAT/Zarr use sampled compression."
        )
        root.addWidget(self._export_size_label)
        for checkbox in (
            self.fmt_mat, self.fmt_npy, self.fmt_json, self.fmt_csv, self.fmt_zarr,
        ):
            checkbox.toggled.connect(self._update_export_estimate_label)

        self.logbox = None
        self.field_selection = settings.get("field_selection") or {}
        stored_images = settings.get("image_field_selection", [])
        self.image_field_selection = (
            None if stored_images is None else {int(i) for i in stored_images})
        self._alignment_save_options = settings.get("alignment_save") or {}

        # Batch progress: hidden until a run starts, so the dialog looks the
        # same as before for single-file work.
        self._progress_row = QWidget()
        progress_layout = QHBoxLayout(self._progress_row)
        progress_layout.setContentsMargins(0, 0, 0, 0)
        self._progress_bar = QProgressBar()
        self._progress_bar.setRange(0, 1000)          # 0.1 % resolution
        self._progress_bar.setTextVisible(True)
        self._progress_label = QLabel("")
        self._cancel_batch_btn = QPushButton("Stop")
        self._cancel_batch_btn.setToolTip(
            "Stop after the file currently being exported finishes.")
        self._cancel_batch_btn.clicked.connect(self._cancel_batch_export)
        progress_layout.addWidget(self._progress_bar, 1)
        progress_layout.addWidget(self._progress_label)
        progress_layout.addWidget(self._cancel_batch_btn)
        self._progress_row.setVisible(False)
        root.addWidget(self._progress_row)

        bottom = QHBoxLayout()
        self._beads_drift_btn = QPushButton("Show beads drift")
        self._beads_drift_btn.setToolTip(
            "Inspect MBM/bead drift per dataset and manually select beads for "
            "alignment.")
        self._beads_drift_btn.clicked.connect(self._on_show_beads_drift)
        align_button = QPushButton("Align channel")
        align_button.clicked.connect(self.on_align_channel)
        self._open_viewer_btn = QPushButton("Open in MINFLUX viewer")
        self._open_viewer_btn.clicked.connect(self._on_open_in_viewer)
        self._open_viewer_btn.setEnabled(self._state is not None)
        export_button = QPushButton("OK (export)")
        export_button.clicked.connect(self.on_ok)
        self._export_button = export_button
        cancel_button = QPushButton("Cancel")
        cancel_button.clicked.connect(self.close)
        bottom.addWidget(self._beads_drift_btn)
        bottom.addWidget(align_button)
        bottom.addWidget(self._open_viewer_btn)
        bottom.addStretch(1)
        bottom.addWidget(cancel_button)
        bottom.addWidget(export_button)
        root.addLayout(bottom)

        # Set the initial "Select a file to view content" button state for the
        # persisted input mode (enabled only in Folder (batch) mode).
        self._on_mode_changed()

    def _browse_row(self, label: str, lineedit: QLineEdit, callback, *, extra=None):
        row = QHBoxLayout()
        row.addWidget(QLabel(label))
        # The field is the only stretching item, so trailing widgets take their
        # width out of the field rather than widening the dialog.
        lineedit.setMinimumWidth(120)
        row.addWidget(lineedit, 1)
        button = QPushButton("Browse…")
        button.clicked.connect(callback)
        row.addWidget(button)
        for widget in ((extra,) if isinstance(extra, QWidget) else tuple(extra or ())):
            row.addWidget(widget)
        return row

    # ------------------------------------------------------------------
    # Logging
    # ------------------------------------------------------------------

    def log(self, msg: str):
        if self._state is not None:
            level = "ERROR" if "[error]" in msg.lower() or "[ERROR]" in msg else "INFO"
            self._state.log(msg, level)
            return
        if self.logbox is not None:
            self.logbox.appendPlainText(msg)
        print(msg)

    # ------------------------------------------------------------------
    # Browse helpers
    # ------------------------------------------------------------------

    def browse_input(self):
        start_dir = self._last_input_dir if os.path.exists(str(self._last_input_dir)) else str(Path.home())
        single = self.mode_file.isChecked()
        if single:
            path, _ = QFileDialog.getOpenFileName(self, "Choose an .msr file", start_dir, "Imspector .msr (*.msr);;All files (*.*)")
        else:
            path = QFileDialog.getExistingDirectory(self, "Choose a folder of .msr files", start_dir)
        if path:
            self.input_path_edit.setText(path)
            self._last_input_dir = str(Path(path).parent if Path(path).is_file() else Path(path))
            self._save_settings()
            # A single .msr file parses immediately (no button click).
            if single and os.path.isfile(path):
                self._parse_specific_file(path)

    def browse_out(self):
        path = QFileDialog.getExistingDirectory(self, "Choose output folder", self.out_dir_edit.text().strip() or self._temp_dir())
        if path:
            self.out_dir_edit.setText(path)
            self._save_settings()

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    def _item_id(self, item) -> int:
        return id(item)

    def _ordinal(self, value: int) -> str:
        if 10 <= (value % 100) <= 20:
            suffix = "th"
        else:
            suffix = {1: "st", 2: "nd", 3: "rd"}.get(value % 10, "th")
        return f"{value}{suffix}"

    def _set_item_tooltip(self, item: QTreeWidgetItem, text: str):
        for column in range(self.tree.columnCount()):
            item.setToolTip(column, text)

    def _on_preview_all_rows_toggled(self, checked: bool) -> None:
        self.preview_rows_spin.setEnabled(not checked)
        self.preview_rows_spin.setVisible(not checked)
        if hasattr(self, "fmt_mat"):
            self._save_settings()

    def _preview_rows_limit(self) -> int | None:
        return None if self.preview_all_rows_check.isChecked() else int(self.preview_rows_spin.value())

    def _preview_limited_array(self, data):
        rows = self._preview_rows_limit()
        arr = np.asarray(data)
        if rows is None or arr.ndim == 0:
            return arr
        return arr[: min(rows, arr.shape[0])]

    def _plot_label_for_path(self, item, path: str) -> str:
        ds_item = self._dataset_item_of(item)
        if ds_item is None:
            return path
        return f"{ds_item.text(0)}. {path}"

    def _safe_export_stem(self, item, path: str) -> str:
        label = self._plot_label_for_path(item, path)
        safe = "".join(ch if ch.isalnum() or ch in "-_." else "_" for ch in label)
        return safe.strip("._") or "export"

    def _show_child_dialog(self, dlg: QDialog) -> None:
        self._child_dialogs.append(dlg)
        dlg.destroyed.connect(lambda _obj=None, d=dlg: self._child_dialogs.remove(d) if d in self._child_dialogs else None)
        dlg.show()
        dlg.raise_()

    # ------------------------------------------------------------------
    # Parse (async)
    # ------------------------------------------------------------------

    def _reset_parse_state(self) -> None:
        """Clear the tree + parsed data before a new parse."""
        self.tree.clear()
        self.parsed = {}
        self._mfx_map = {}
        self._mbm_map = {}
        self._mbm_meta_map = {}
        self.nodeinfo.clear()
        self.fullpath_by_item.clear()
        self.datasetnode_info.clear()
        self._export_estimates = {}
        self._estimate_generation += 1
        if self._estimate_task is not None:
            self._estimate_task.cancel()
            self._estimate_task = None
        if hasattr(self, "_export_size_label"):
            self._update_export_estimate_label()

    def _parse_specific_file(self, msr: str) -> None:
        """Reset + parse one specific ``.msr`` file (async)."""
        if not msr or not os.path.isfile(msr):
            self.log("[error] not a valid .msr file.")
            return
        self._reset_parse_state()
        self.log(f"[parse] using: {msr}")
        self._start_parse_worker(msr, self._temp_dir())

    def on_parse(self):
        """Parse the current input — a single file, or the first ``.msr`` in a
        folder. Used by the auto-parse triggers (drop / Browse / Enter)."""
        ipath = self.input_path_edit.text().strip()
        if not ipath or not os.path.exists(ipath):
            self.log("[error] set an input .msr file (or folder) first.")
            return
        msr = ipath if os.path.isfile(ipath) else next((str(p) for p in Path(ipath).glob("*.msr")), None)
        if not msr:
            self.log("no msr file found!")
            return
        self._save_settings()
        self._parse_specific_file(msr)

    def _auto_parse_single_input(self) -> None:
        """Parse immediately when the input is a single ``.msr`` file (drop /
        Browse / Enter) — no button press needed."""
        if not self.mode_file.isChecked():
            return
        ipath = self.input_path_edit.text().strip()
        if ipath and os.path.isfile(ipath):
            self.on_parse()

    def on_select_view_file(self) -> None:
        """Button handler ("Select a file to view content"). In Folder (batch)
        mode, choose which ``.msr`` in the folder to parse & preview (the batch
        folder input is left unchanged). In single-file mode the file already
        auto-parses, so this just (re)parses the current file."""
        if not self.mode_folder.isChecked():
            self.on_parse()
            return
        folder = self.input_path_edit.text().strip()
        if folder and os.path.isdir(folder):
            start = folder
        elif os.path.exists(str(self._last_input_dir)):
            start = str(self._last_input_dir)
        else:
            start = str(Path.home())
        path, _ = QFileDialog.getOpenFileName(
            self, "Select a .msr file to view content", start,
            "Imspector .msr (*.msr);;All files (*.*)")
        if path:
            self._parse_specific_file(path)

    def _on_input_path_dropped(self, path: str) -> None:
        """A dropped path also selects the input mode it implies."""
        candidate = Path(path)
        if candidate.is_dir():
            self.mode_folder.setChecked(True)
            self._last_input_dir = str(candidate)
            self._save_settings()
            return
        self.mode_file.setChecked(True)
        self._last_input_dir = str(candidate.parent)
        self._save_settings()
        self._parse_specific_file(str(candidate))

    def _on_mode_changed(self) -> None:
        """Keep the button meaningful per input type: a single file auto-parses,
        so "Select a file to view content" is only enabled in Folder (batch)
        mode (where it picks which file in the folder to preview)."""
        folder = self.mode_folder.isChecked()
        self._parse_button.setEnabled(folder)
        # Both batch options are meaningless for a single file.
        if hasattr(self, "recursive_check"):
            self.recursive_check.setEnabled(folder)
        if hasattr(self, "reproduce_tree_check"):
            self.reproduce_tree_check.setEnabled(folder)
        # A single file asks instead, so the fixed choice is batch-only.
        for name in ("conflict_label", "conflict_combo"):
            if hasattr(self, name):
                getattr(self, name).setEnabled(folder)
        self._parse_button.setToolTip(
            "Pick which .msr file in the folder to parse and preview."
            if folder else
            "A single .msr file parses automatically on drop / Browse / Enter. "
            "Switch to Folder (batch) mode to pick a file to preview.")

    def _start_parse_worker(self, msr: str, tmp: str):
        from minflux_viewer.ui.common.background_tasks import retain_qthread, retire_qthreads

        self._parse_button.setEnabled(False)
        if self._worker is not None:
            retire_qthreads(
                [self._worker], signal_names=("progress", "succeeded", "failed")
            )
        self._worker = _ParseWorker(msr, tmp)
        retain_qthread(self._worker)
        self._worker.progress.connect(self.log)
        self._worker.succeeded.connect(self._on_parse_done)
        self._worker.failed.connect(self._on_parse_failed)
        self._worker.start()

    def _store_parse_result(self, result: dict) -> None:
        """Adopt a parse result as this dialog's data. The per-parse ``mfx_map`` /
        ``mbm_map`` / ``mbm_meta_map`` (built by the parser from that file alone)
        become the dialog's own copy, so it is independent of the shared global
        ``msr.state`` (which only ever reflects the most-recently-parsed file)."""
        self.parsed = result or {}
        self._mfx_map = dict(self.parsed.get("mfx_map") or {})
        self._mbm_map = dict(self.parsed.get("mbm_map") or {})
        self._mbm_meta_map = dict(self.parsed.get("mbm_meta_map") or {})

    def _msr_state(self):
        """A per-instance stand-in for the module-global ``msr.state`` — exposes
        ``mfx_map`` / ``mbm_map`` / ``mbm_meta_map`` plus ``mfx`` / ``mbm`` (the
        first array) over THIS dialog's parsed data. Data methods use this instead
        of ``from ...msr import state`` so multiple reader dialogs stay independent."""
        from types import SimpleNamespace
        return SimpleNamespace(
            mfx_map=self._mfx_map,
            mbm_map=self._mbm_map,
            mbm_meta_map=self._mbm_meta_map,
            mfx=next(iter(self._mfx_map.values()), None),
            mbm=next(iter(self._mbm_map.values()), None),
        )

    # The dialog-facing names delegate to the module-level implementations that
    # the batch worker also uses, so there is one export path, not two.
    def _unique_export_stem(self, key: str, used: set[str]) -> str:
        return unique_export_stem(key, used, self.log)

    def _subset_struct_fields(self, arr, names_sel):
        return subset_struct_fields(arr, names_sel)

    def _export_image_series(self, out_dir: str, series: list[dict]) -> None:
        export_image_series(out_dir, self.parsed.get("msr"), series, self.log)

    def _on_parse_done(self, result: dict):
        self._worker = None
        self._store_parse_result(result)
        self._parse_button.setEnabled(True)
        self._build_tree_from_result(result)

        if self._mfx_map:
            self.log(f"[parsed] mfx datasets: {len(self._mfx_map)}")
            for key, arr in self._mfx_map.items():
                self.log(f"          - {key}: shape={arr.shape}, dtype={arr.dtype}")
        else:
            self.log("[parsed] mfx is empty")
        if self._mbm_map:
            self.log(f"[parsed] mbm datasets: {len(self._mbm_map)}")
            for key, arr in self._mbm_map.items():
                self.log(f"          - {key}: shape={arr.shape}, dtype={arr.dtype}")
        else:
            self.log("[parsed] mbm is empty")
        self._start_export_estimate()

    def _on_parse_failed(self, error: str):
        self._worker = None
        self.log(f"[ERROR] {error}")
        self._parse_button.setEnabled(True)

    def _build_tree_from_result(self, result: dict):
        from ...msr.descriptions import describe_path

        msr = result.get("msr", "unknown.msr")
        root_item = QTreeWidgetItem([os.path.basename(msr), "", "", ""])
        self._set_item_tooltip(root_item, "MSR file used to store measurement data generated by iMSPECTOR software.")
        self.tree.addTopLevelItem(root_item)
        self.nodeinfo[self._item_id(root_item)] = {"type": "msr_root", "description": root_item.toolTip(0)}

        if result.get("mode") == "modern":
            for ds_index, ds in enumerate(result.get("datasets", []), start=1):
                ds_item = QTreeWidgetItem([str(ds.get("display_name") or ds.get("did")), "dataset", "", str(ds.get("did", ""))])
                self._set_item_tooltip(ds_item, f"{self._ordinal(ds_index)} dataset from MINFLUX measurement.")
                root_item.addChild(ds_item)
                self.datasetnode_info[self._item_id(ds_item)] = {"did": ds.get("did"), "zroot": ds.get("zroot")}
                self.fullpath_by_item[self._item_id(ds_item)] = ""
                self.nodeinfo[self._item_id(ds_item)] = {"type": "dataset", "description": ds_item.toolTip(0), "zroot": ds.get("zroot"), "path": ""}

                node_map = {"": ds_item}
                for fld in ds.get("fields") or []:
                    path = fld["path"]
                    parts = path.split("/")
                    cur = ""
                    parent = ds_item
                    for i, part in enumerate(parts):
                        cur = part if i == 0 else f"{cur}/{part}"
                        if cur in node_map:
                            parent = node_map[cur]
                            continue
                        is_last = i == len(parts) - 1
                        kind = fld["kind"] if is_last else "group"
                        shape = "" if kind != "array" else str(fld.get("shape"))
                        dtype = "" if kind != "array" else fld.get("dtype", "")
                        item = QTreeWidgetItem([part, kind, shape, dtype])
                        self._set_item_tooltip(item, fld.get("description") if is_last else describe_path(cur, is_array=False))
                        parent.addChild(item)
                        node_map[cur] = item
                        self.fullpath_by_item[self._item_id(item)] = cur
                        if not is_last:
                            self.nodeinfo[self._item_id(item)] = {"type": "modern_group", "zroot": ds["zroot"], "path": cur, "description": item.toolTip(0)}
                        if is_last and kind == "array":
                            self.nodeinfo[self._item_id(item)] = {
                                "type": "modern_array",
                                "zroot": ds["zroot"],
                                "path": path,
                                "description": item.toolTip(0),
                                "dtype_raw": fld.get("dtype_raw") or fld.get("dtype"),
                                "show_raw_dtype": path not in {"grd/mbm/points", "grd/search_0/points", "mfx"},
                            }
                            for sub in fld.get("dtype_fields") or []:
                                self._add_dtype_field_nodes(item, path, [sub])
                        parent = item
        else:
            meta = result.get("metadata")
            series_root = QTreeWidgetItem(["Series", "", "", ""])
            root_item.addChild(series_root)
            for s in result.get("legacy_series_tree") or []:
                idx = int(s.get("index", 0))
                name = s.get("display_name", f"Series {idx + 1}")
                shape_str = s.get("shape_str", "")
                item = QTreeWidgetItem([f"- Series {idx + 1}: {name}", "", shape_str, s.get("dtype", "")])
                series_root.addChild(item)
                # ndim >= 2 (a shape like "A x B") ⇒ a viewable image series.
                if "x" in str(shape_str):
                    self.nodeinfo[self._item_id(item)] = {
                        "type": "legacy_series", "index": idx, "name": name}
            meta_item = QTreeWidgetItem(["<metadata>", "", "", ""])
            self._set_item_tooltip(
                meta_item,
                "OME metadata of this legacy MSR file. Right-click > Show metadata "
                "opens it as an XML table and formatted XML document.",
            )
            root_item.addChild(meta_item)
            if isinstance(meta, dict):
                self.nodeinfo[self._item_id(meta_item)] = {
                    "type": "legacy_metadata",
                    "data": meta,
                    "xml": result.get("metadata_xml"),
                }
                _populate_metadata_items(meta_item, _json_compact(meta))
            else:
                meta_item.addChild(QTreeWidgetItem(["(metadata unavailable)", "", "", ""]))

        root_item.setExpanded(True)
        for i in range(root_item.childCount()):
            ds_item = root_item.child(i)
            ds_item.setExpanded(True)
            for j in range(ds_item.childCount()):
                child = ds_item.child(j)
                child.setExpanded(child.text(0) == "grd")
                for k in range(child.childCount()):
                    child.child(k).setExpanded(False)
        # After the expansion pass, so the image groups keep their collapsed
        # default instead of being expanded along with the dataset nodes.
        if result.get("mode") == "modern":
            self._add_image_series_nodes(root_item, result)
        for column in range(self.tree.columnCount()):
            self.tree.resizeColumnToContents(column)

    def _add_image_series_nodes(self, root_item, result: dict) -> None:
        """Show the file's OBF image series in the parsed-contents tree.

        A density/trace render names the MINFLUX dataset it was computed from
        (``source_did``), so it is nested under that dataset's node, parallel to
        ``mfx``/``grd``.  Confocal channels belong to no dataset and go into one
        ``Image series`` group beside the datasets, last.  Both groups start
        **collapsed** — they are usually many and rarely the reason the tree was
        opened.
        """
        series = self._gather_image_series_for_dialog()
        if not series:
            return

        # A did can appear twice (a dataset and its aggregated companion share
        # one); the first node carrying it owns the linked images.
        ds_item_by_did: dict[str, QTreeWidgetItem] = {}
        for i in range(root_item.childCount()):
            item = root_item.child(i)
            did = str((self.datasetnode_info.get(self._item_id(item)) or {}).get("did") or "")
            if did:
                ds_item_by_did.setdefault(did, item)

        loose: list[dict] = []
        historical: list[dict] = []
        nested: dict[int, list[dict]] = {}
        for entry in series:
            owner = ds_item_by_did.get(str(entry.get("source_did") or ""))
            if owner is None:
                # Project-style .msr files can retain trace rasters from old
                # runs after their localization payload has been replaced.
                # Keep these useful images, but never misassociate them with
                # whichever dataset happens to remain in the file.
                if entry.get("source_did"):
                    historical.append(entry)
                else:
                    loose.append(entry)
            else:
                nested.setdefault(id(owner), []).append(entry)

        for i in range(root_item.childCount()):
            item = root_item.child(i)
            group = nested.get(id(item))
            if group:
                self._add_image_series_group(
                    item, group,
                    "Image series rendered from this MINFLUX dataset "
                    "(density / trace). Right-click > Open as image.")
        if loose:
            self._add_image_series_group(
                root_item, loose,
                "Image series stored in this .msr file (confocal / overview "
                "channels). Right-click > Open as image.")
        if historical:
            self._add_image_series_group(
                root_item,
                historical,
                "Historical MINFLUX-derived image series whose source DID is no "
                "longer present as a localization payload in this project file. "
                "The raster remains viewable and exportable, but is not attached "
                "to the unrelated current dataset.",
                label="Historical MINFLUX images",
            )

    def _add_image_series_group(
        self, parent, series: list[dict], tooltip: str, *, label: str = "Image series"
    ) -> None:
        group = QTreeWidgetItem([f"{label} ({len(series)})", "", "", ""])
        self._set_item_tooltip(group, tooltip)
        parent.addChild(group)
        self.nodeinfo[self._item_id(group)] = {
            "type": "image_series_group", "description": group.toolTip(0)}
        for entry in series:
            raw = int(entry.get("raw_index", -1))
            name = str(entry.get("name") or f"Series {raw + 1}")
            item = QTreeWidgetItem(
                [name, "image", str(entry.get("shape_str", "")), str(entry.get("dtype", ""))])
            self._set_item_tooltip(
                item,
                f"OBF image stack {raw}. Source DID: "
                f"{entry.get('source_did') or 'none'}. Right-click > Show metadata "
                "or Open as image.",
            )
            group.addChild(item)
            self.nodeinfo[self._item_id(item)] = {
                "type": "image_series", "index": raw, "name": name,
                "description": item.toolTip(0)}
        group.setExpanded(False)

    def _add_dtype_field_nodes(self, parent_item, path_prefix: str, fields: list):
        from ...msr.descriptions import describe_path

        for sub in fields or []:
            text = sub.get("name", "")
            kind = "struct" if sub.get("children") else sub.get("kind", "field")
            shape = sub.get("logical_shape") or str(sub.get("shape") or ())
            dtype = sub.get("dtype", "")
            fullpath = f"{path_prefix}/{text}" if path_prefix else text
            if sub.get("children") and fullpath == "mfx/itr":
                dtype = "Structured per-iteration MINFLUX table"
            item = QTreeWidgetItem([text, kind, shape, dtype])
            self._set_item_tooltip(item, sub.get("description") or describe_path(fullpath, is_array=False))
            parent_item.addChild(item)
            self.fullpath_by_item[self._item_id(item)] = fullpath
            self.nodeinfo[self._item_id(item)] = {"type": "modern_field", "description": item.toolTip(0)}
            if sub.get("children"):
                self._add_dtype_field_nodes(item, fullpath, sub["children"])

    # ------------------------------------------------------------------
    # Tree helpers
    # ------------------------------------------------------------------

    def _current_item(self):
        items = self.tree.selectedItems()
        return items[0] if items else None

    def _selected_items(self) -> list[QTreeWidgetItem]:
        return self.tree.selectedItems() or ([self._current_item()] if self._current_item() is not None else [])

    def _selected_array_items(self) -> list[QTreeWidgetItem]:
        return [item for item in self._selected_items() if item.text(1) in {"field", "array"}]

    def _dataset_item_of(self, item):
        while item is not None:
            if item.text(1) == "dataset":
                return item
            item = item.parent()
        return None

    def _zroot_for_dataset_item(self, item):
        info = self.datasetnode_info.get(self._item_id(item))
        return (info or {}).get("zroot"), (info or {}).get("did")

    def _load_array_for_path(self, zroot: str, path: str):
        from ...msr import zarr2
        arch = zarr2.open(zroot, mode="r")
        if path in arch:
            return np.asarray(arch[path])

        parts = [part for part in path.split("/") if part]
        for split_at in range(len(parts) - 1, 0, -1):
            parent = "/".join(parts[:split_at])
            if parent not in arch:
                continue

            data = np.asarray(arch[parent])
            for field in parts[split_at:]:
                names = getattr(data.dtype, "names", None)
                if not names or field not in names:
                    raise KeyError(f"field not found: {parent}/{'/'.join(parts[split_at:])}")
                data = data[field]
            return np.asarray(data)
        raise KeyError(f"path not found: {path}")

    def _resolve_context_array(self, item=None):
        item = item or self._current_item()
        if item is None:
            return None, None, "Choose a field or array node."
        ds_item = self._dataset_item_of(item)
        if ds_item is None:
            return None, None, "Choose a field or array node."
        zroot, _ = self._zroot_for_dataset_item(ds_item)
        if not zroot:
            return None, None, "Dataset has no zarr root."
        path = self.fullpath_by_item.get(self._item_id(item))
        if not path:
            return None, None, "Choose a field or array node."
        try:
            arr = self._load_array_for_path(zroot, path)
        except Exception as exc:
            return None, None, f"Cannot load {path}:\n{exc}"
        return path, arr, None

    # ------------------------------------------------------------------
    # Plot window handling
    # ------------------------------------------------------------------

    def _on_plot_window_focused(self, window: PlotWindow):
        if window in self._plot_windows:
            self._active_plot_window = window

    def _on_plot_window_closed(self, window: PlotWindow):
        if window in self._plot_windows:
            self._plot_windows.remove(window)
        if self._active_plot_window is window:
            self._active_plot_window = self._plot_windows[-1] if self._plot_windows else None

    def _visible_plot_windows(self) -> list[PlotWindow]:
        self._plot_windows = [win for win in self._plot_windows if win is not None and win.isVisible()]
        return self._plot_windows

    def _target_plot_window(self) -> PlotWindow | None:
        windows = self._visible_plot_windows()
        if not windows:
            return None
        active = QApplication.activeWindow()
        if isinstance(active, PlotWindow) and active in windows:
            return active
        if self._active_plot_window in windows:
            return self._active_plot_window
        return windows[-1]

    def _default_plot_mode(self, path: str, data) -> str:
        name = path.rsplit("/", 1)[-1].lower()
        arr = np.asarray(data)
        if name in {"xyz", "loc", "lnc", "ext"} or (arr.ndim == 2 and arr.shape[1] in (2, 3)):
            return "scatter"
        if np.issubdtype(arr.dtype, np.bool_):
            return "histogram"
        if name in {"tim", "time", "timestamp", "tid"}:
            return "line"
        if name in {"efo", "cfr", "dcr"}:
            return "histogram"
        if np.issubdtype(arr.dtype, np.integer) and np.unique(arr[: min(arr.size, 10000)]).size <= 16:
            return "histogram"
        return "line" if arr.ndim == 1 else "scatter"

    def _show_plot_window(self, data, title: str, label: str, mode: str | None = None) -> PlotWindow:
        win = PlotWindow(data, title=title, parent=self, label=label, mode=mode)
        self._plot_windows.append(win)
        self._active_plot_window = win
        win.show()
        win.raise_()
        win.activateWindow()
        return win

    def _plot_data(self, data, title: str, label: str, mode: str | None = None) -> None:
        if self.plot_new_window_check.isChecked():
            self._show_plot_window(data, title, label, mode)
            return

        target = self._target_plot_window()
        if target is None:
            self._show_plot_window(data, title, label, mode)
            return

        can_overlay, reason = target.can_overlay(data)
        if not can_overlay:
            self.log(
                f"[warn] plot range for '{label}' does not fit the active plot window "
                f"({reason}); opening a new plot window."
            )
            self._show_plot_window(data, title, label, mode)
            return

        target.add_data(data, label, redraw=True)
        target.raise_()
        target.activateWindow()

    # ------------------------------------------------------------------
    # Context menu
    # ------------------------------------------------------------------

    def _show_context_menu(self, pos):
        item = self.tree.itemAt(pos)
        if item is None:
            return
        if not item.isSelected():
            self.tree.clearSelection()
            item.setSelected(True)
        self.tree.setCurrentItem(item)
        menu = QMenu(self)
        selected_arrays = self._selected_array_items()
        can_export = bool(selected_arrays)
        image_series = [self.nodeinfo.get(self._item_id(it), {}) for it in self._selected_items()]
        image_series = [info for info in image_series if info.get("type") in _IMAGE_NODE_TYPES]
        open_image = None
        if image_series:
            open_image = menu.addAction(
                "Open as image" if len(image_series) == 1
                else f"Open {len(image_series)} series as images")
            menu.addSeparator()
        preview = menu.addAction("Preview")
        plot = menu.addAction("Plot…")
        export_csv = menu.addAction("Export to CSV…")
        export_npy = menu.addAction("Export to NumPy (.npy)…")
        export_json = menu.addAction("Export to JSON…")
        show_meta = menu.addAction("Show metadata")
        plot.setEnabled(can_export)
        export_csv.setEnabled(can_export)
        export_npy.setEnabled(can_export)
        export_json.setEnabled(can_export)
        show_meta.setEnabled(any(self._can_show_metadata(selected) for selected in self._selected_items()))

        action = menu.exec(self.tree.viewport().mapToGlobal(pos))
        if open_image is not None and action == open_image:
            self._ctx_open_as_image([info["index"] for info in image_series])
        elif action == preview:
            self._ctx_preview()
        elif action == plot:
            self._ctx_plot()
        elif action == export_csv:
            self._ctx_export("csv")
        elif action == export_npy:
            self._ctx_export("npy")
        elif action == export_json:
            self._ctx_export("json")
        elif action == show_meta:
            self._ctx_show_metadata()

    def _node_metadata_target(self, item):
        info = self.nodeinfo.get(self._item_id(item), {})
        if info.get("type") in {"modern_group", "modern_array", "dataset"}:
            return info.get("zroot"), info.get("path") or ""
        ds_item = self._dataset_item_of(item)
        if ds_item is not None:
            zroot, _ = self._zroot_for_dataset_item(ds_item)
            return zroot, self.fullpath_by_item.get(self._item_id(item), "") or ""
        return None, None

    def _legacy_metadata_of(self, item):
        """The parsed legacy OME metadata dict if *item* is (inside) the
        ``<metadata>`` node of a legacy file, else None."""
        while item is not None:
            info = self.nodeinfo.get(self._item_id(item), {})
            if info.get("type") == "legacy_metadata":
                return info.get("data")
            item = item.parent()
        return None

    def _legacy_metadata_xml_of(self, item):
        while item is not None:
            info = self.nodeinfo.get(self._item_id(item), {})
            if info.get("type") == "legacy_metadata":
                return info.get("xml")
            item = item.parent()
        return None

    def _can_show_metadata(self, item) -> bool:
        from ...msr.io import read_zarr_attrs
        info = self.nodeinfo.get(self._item_id(item), {})
        if info.get("type") in _IMAGE_NODE_TYPES:
            return True
        if self._legacy_metadata_of(item) is not None:
            return True
        zroot, path = self._node_metadata_target(item)
        if not zroot:
            return False
        try:
            return bool(read_zarr_attrs(zroot, path or ""))
        except Exception:
            return False

    def _ctx_show_metadata(self):
        from minflux_viewer.ui.tools.metadata_viewer import MetadataViewDialog

        from ...core.tiff_source import MetadataDocument
        from ...msr.io import read_zarr_attrs

        shown = 0
        legacy_shown = False
        for item in self._selected_items():
            info = self.nodeinfo.get(self._item_id(item), {})
            if info.get("type") in _IMAGE_NODE_TYPES:
                try:
                    from ...core.obf_image_source import ObfImageSource

                    source = ObfImageSource(
                        self.parsed.get("msr", ""), raw_stack_index=int(info["index"])
                    )
                    try:
                        documents = source.metadata.documents
                        title = source.metadata.image_name or item.text(0)
                    finally:
                        source.close()
                    self._show_child_dialog(
                        MetadataViewDialog(f"Metadata: {title}", documents, self)
                    )
                    shown += 1
                except Exception as exc:
                    self.log(f"[metadata] failed for {item.text(0)}: {exc}")
                continue
            legacy_meta = self._legacy_metadata_of(item)
            if legacy_meta is not None:
                if not legacy_shown:
                    name = Path(self.parsed.get("msr", "")).name or "legacy file"
                    xml = self._legacy_metadata_xml_of(item)
                    if xml:
                        dialog = MetadataViewDialog(
                            f"Metadata: {name}",
                            [MetadataDocument("OME-XML", "xml", xml)],
                            self,
                        )
                    else:
                        dialog = JsonViewDialog(f"Metadata: {name}", legacy_meta, self)
                    self._show_child_dialog(dialog)
                    legacy_shown = True
                    shown += 1
                continue
            zroot, path = self._node_metadata_target(item)
            if not zroot:
                continue
            try:
                attrs = read_zarr_attrs(zroot, path or "")
            except Exception as exc:
                self.log(f"[metadata] failed for {item.text(0)}: {exc}")
                continue
            if not attrs:
                continue
            self._show_child_dialog(JsonViewDialog(f"Metadata: {path or '<dataset root>'}", attrs, self))
            shown += 1
        if shown == 0:
            QMessageBox.information(self, "Metadata", "No Zarr metadata is available for the selected node(s).")

    def _ctx_open_as_image(self, raw_indices):
        """Open the selected OBF series (by raw stack index) in the standalone
        TIFF/image viewer."""
        if not self._open_obf_series_group(raw_indices):
            QMessageBox.information(
                self, "Open as image",
                "The selected series could not be opened as an image.")

    def _selected_image_series_indices(self) -> list[int]:
        return [
            int(info["index"])
            for it in self._selected_items()
            if (info := self.nodeinfo.get(self._item_id(it), {})).get("type") in _IMAGE_NODE_TYPES
        ]

    def _ctx_preview(self):
        # Previewing an image series as a table is useless — open the image viewer.
        image_series = self._selected_image_series_indices()
        if image_series:
            self._open_obf_series_group(image_series)
        shown = 0
        for item in self._selected_array_items():
            path, arr, err = self._resolve_context_array(item)
            if err:
                self.log(f"[preview] {err}")
                continue
            label = self._plot_label_for_path(item, path)
            self._show_child_dialog(ArrayPreviewDialog(f"Preview: {label}", arr, self._preview_rows_limit(), self))
            shown += 1
        if shown == 0 and not image_series:
            QMessageBox.information(self, "Preview", "Choose one or more field or array nodes.")

    def _ctx_plot(self):
        entries = []
        for item in self._selected_array_items():
            path, arr, err = self._resolve_context_array(item)
            if err:
                self.log(f"[plot] {err}")
                continue
            data = np.asarray(self._preview_limited_array(arr))
            if data.size == 0:
                self.log(f"[plot] skip empty selection: {path}")
                continue
            if not _is_numeric_dtype(data.dtype):
                self.log(f"[plot] skip non-numeric selection: {path}")
                continue
            entries.append((item, path, data, self._plot_label_for_path(item, path)))
        if not entries:
            QMessageBox.information(self, "Plot", "Choose one or more numeric field or array nodes.")
            return
        if self.plot_new_window_check.isChecked() or len(entries) == 1:
            for _item, path, data, label in entries:
                self._plot_data(data, f"Plot: {label}", label, self._default_plot_mode(path, data))
            return
        groups: dict[str, list[tuple[Any, str, np.ndarray, str]]] = {}
        for entry in entries:
            _item, path, _data, _label = entry
            groups.setdefault(path, []).append(entry)
        for path, group in groups.items():
            win = None
            for _item, _path, data, label in group:
                if win is None:
                    win = self._show_plot_window(data, f"Plot: {path}", label, self._default_plot_mode(path, data))
                else:
                    can_overlay, reason = win.can_overlay(data)
                    if can_overlay:
                        win.add_data(data, label, redraw=True)
                    else:
                        self.log(f"[warn] plot range for '{label}' does not fit grouped plot ({reason}); opening a new plot window.")
                        self._show_plot_window(data, f"Plot: {label}", label, self._default_plot_mode(path, data))

    def _ctx_export(self, fmt: str):
        items = self._selected_array_items()
        if not items:
            QMessageBox.information(self, "Export", "Choose one or more field or array nodes.")
            return
        if len(items) == 1:
            item = items[0]
            path = self.fullpath_by_item.get(self._item_id(item), item.text(0))
            filters = {"csv": "CSV (*.csv);;All files (*.*)", "npy": "NumPy (*.npy);;All files (*.*)", "json": "JSON (*.json);;All files (*.*)"}
            out, _ = QFileDialog.getSaveFileName(self, f"Export to {fmt.upper()}", f"{self._safe_export_stem(item, path)}.{fmt}", filters[fmt])
            if not out:
                return
            targets = [(item, out)]
        else:
            folder = QFileDialog.getExistingDirectory(self, f"Choose folder for {len(items)} {fmt.upper()} exports", self.out_dir_edit.text().strip() or str(Path.home()))
            if not folder:
                return
            targets = []
            for item in items:
                path = self.fullpath_by_item.get(self._item_id(item), item.text(0))
                targets.append((item, str(Path(folder) / f"{self._safe_export_stem(item, path)}.{fmt}")))
        for item, out in targets:
            path, data, err = self._resolve_context_array(item)
            if err:
                self.log(f"[export] {err}")
                continue
            try:
                if fmt == "csv":
                    np.savetxt(out, self._to_2d(data), delimiter=",")
                elif fmt == "npy":
                    np.save(out, data, allow_pickle=False)
                else:
                    self._save_array_json(out, data)
                self.log(f"[export] wrote {out}")
            except Exception as exc:
                self.log(f"[export] failed {path}: {exc}")

    def _save_array_json(self, fname: str, arr):
        a = np.asarray(arr)
        names = getattr(getattr(a, "dtype", None), "names", None)
        if names:
            out = []
            for row in a:
                d = {}
                for k in names:
                    v = row[k]
                    if isinstance(v, np.ndarray):
                        d[k] = v.tolist()
                    elif isinstance(v, np.generic):
                        d[k] = v.item()
                    else:
                        d[k] = v
                out.append(d)
        else:
            out = a.tolist()
        with open(fname, "w", encoding="utf-8") as f:
            json.dump(out, f, ensure_ascii=False)

    def _to_2d(self, a):
        a = np.asarray(a)
        if a.ndim == 1:
            return a.reshape(-1, 1)
        if a.ndim > 2:
            return a.reshape(a.shape[0], -1)
        return a

    # ------------------------------------------------------------------
    # Fields / export
    # ------------------------------------------------------------------

    def _gather_datasets_for_dialog(self) -> list[dict]:
        MFSTATE = self._msr_state()
        datasets = []
        if self.parsed.get("mode") == "modern":
            for ds in self.parsed.get("datasets") or []:
                key = ds.get("display_name") or ds.get("did") or "dataset"
                mfx_arr = MFSTATE.mfx_map.get(key)
                mbm_arr = MFSTATE.mbm_map.get(key)
                datasets.append({
                    "key": key,
                    "name": key,
                    "mfx_fields": list(getattr(getattr(mfx_arr, "dtype", None), "names", []) or []),
                    "mbm_fields": list(getattr(getattr(mbm_arr, "dtype", None), "names", []) or []),
                })
        else:
            key = "legacy"
            mfx_arr = MFSTATE.mfx_map.get(key) or MFSTATE.mfx
            mbm_arr = MFSTATE.mbm_map.get(key) or MFSTATE.mbm
            datasets.append({
                "key": key,
                "name": key,
                "mfx_fields": list(getattr(getattr(mfx_arr, "dtype", None), "names", []) or []),
                "mbm_fields": list(getattr(getattr(mbm_arr, "dtype", None), "names", []) or []),
            })
        return datasets

    def _selected_export_components(self) -> list[tuple[str, np.ndarray]]:
        """Current field selection as the structured arrays writers will see."""
        mfstate = self._msr_state()
        if self.parsed.get("mode") == "modern":
            entries = [
                ((entry.get("display_name") or entry.get("did") or "dataset"), entry)
                for entry in (self.parsed.get("datasets") or [])
            ]
        else:
            keys = mfstate.mfx_map.keys() | mfstate.mbm_map.keys()
            entries = [(key, None) for key in sorted(keys)] or [("legacy", None)]

        selection = self.field_selection or {}
        components: list[tuple[str, np.ndarray]] = []
        for key, entry in entries:
            selected = selection.get(
                key, {"checked": True, "mfx": None, "mbm": None}
            )
            if not selected.get("checked", True):
                continue
            mfx = (entry or {}).get("_mfx")
            mbm = (entry or {}).get("_mbm")
            if mfx is None:
                mfx = mfstate.mfx_map.get(key)
            if mbm is None:
                mbm = mfstate.mbm_map.get(key)
            mfx = subset_struct_fields(mfx, selected.get("mfx"))
            mbm = subset_struct_fields(mbm, selected.get("mbm"))
            if mfx is not None and np.asarray(mfx).size:
                components.append(("mfx", np.asarray(mfx)))
            if mbm is not None and np.asarray(mbm).size:
                components.append(("mbm", np.asarray(mbm)))
        return components

    def _start_export_estimate(self) -> None:
        self._estimate_generation += 1
        generation = self._estimate_generation
        if self._estimate_task is not None:
            self._estimate_task.cancel()
        components = self._selected_export_components()
        self._export_estimates = {}
        self._update_export_estimate_label(calculating=bool(components))
        if not components:
            self._estimate_task = None
            return
        task = _ExportEstimateTask(generation, components)
        task.signals.done.connect(self._on_export_estimate_ready)
        self._estimate_task = task
        from minflux_viewer.ui.common.background_tasks import shared_thread_pool

        shared_thread_pool("msr-work").start(task)

    def _on_export_estimate_ready(self, generation: int, result: object) -> None:
        if generation != self._estimate_generation:
            return
        self._estimate_task = None
        self._export_estimates = result if isinstance(result, dict) else {}
        self._update_export_estimate_label()

    def _selected_format_names(self) -> list[str]:
        return [
            name
            for name, checkbox in (
                ("mat", self.fmt_mat),
                ("npy", self.fmt_npy),
                ("json", self.fmt_json),
                ("csv", self.fmt_csv),
                ("zarr", self.fmt_zarr),
            )
            if checkbox.isChecked()
        ]

    def _selected_image_uncompressed_bytes(self) -> tuple[int, int]:
        """Selected image count and raw pixel bytes (a conservative size bound)."""
        import re

        series = select_image_series(
            self._gather_image_series_for_dialog(),
            indices=getattr(self, "image_field_selection", set()),
        )
        total = 0
        for entry in series:
            dims = [int(value) for value in re.findall(r"\d+", str(entry.get("shape_str", "")))]
            try:
                itemsize = np.dtype(str(entry.get("dtype") or "uint8")).itemsize
            except (TypeError, ValueError):
                itemsize = 1
            total += int(np.prod(dims, dtype=np.int64)) * itemsize if dims else 0
        return len(series), total

    def _update_export_estimate_label(
        self, _checked: bool = False, *, calculating: bool = False,
    ) -> None:
        if not hasattr(self, "_export_size_label"):
            return
        selected = self._selected_format_names()
        if not self.parsed:
            self._export_size_label.setText(
                "Estimated output: parse a file to calculate from its actual rows and fields."
            )
            return
        if not selected:
            self._export_size_label.setText("Estimated output: select a format.")
            return
        if calculating or not self._export_estimates:
            self._export_size_label.setText(
                "Estimated output: calculating from sampled values…"
            )
            return
        if "error" in self._export_estimates:
            self._export_size_label.setText(
                "Estimated output unavailable; export is still available."
            )
            self._export_size_label.setToolTip(str(self._export_estimates["error"]))
            return
        from ...core.export_size import format_file_size

        labels = {
            "mat": "MAT",
            "npy": "NumPy",
            "json": "JSON",
            "csv": "CSV",
            "zarr": "Zarr",
        }
        parts = []
        image_count, image_bytes = (
            self._selected_image_uncompressed_bytes()
            if "zarr" in selected else (0, 0)
        )
        for fmt in selected:
            estimate = self._export_estimates.get(fmt)
            if estimate is None:
                continue
            prefix = "" if estimate.confidence == "exact" else "≈"
            suffix = ""
            if fmt == "zarr":
                if image_count:
                    suffix = (
                        f" numeric data + up to {format_file_size(image_bytes)} "
                        f"from {image_count} selected image series + search/viewer state"
                    )
                else:
                    suffix = " numeric data + search/viewer state"
            parts.append(
                f"{labels[fmt]} {prefix}{format_file_size(estimate.bytes)}{suffix}"
            )
        self._export_size_label.setText(
            "Estimated output (selected datasets/fields): " + " · ".join(parts)
        )

    def on_fields_dialog(self):
        # Drop empty placeholder datasets (e.g. the "legacy" entry of an image-only
        # file has no mfx/mbm fields) so only real datasets + image series show.
        ds_list = [d for d in self._gather_datasets_for_dialog()
                   if d.get("mfx_fields") or d.get("mbm_fields")]
        image_series = self._gather_image_series_for_dialog()
        if not ds_list and not image_series:
            QMessageBox.information(self, "No datasets", "No datasets loaded to select fields from.")
            return
        dlg = FieldDialog(ds_list, self.field_selection or None, self,
                          image_series=image_series,
                          prechecked_images=self.image_field_selection)
        if dlg.exec() == QDialog.DialogCode.Accepted:
            self.field_selection = dlg.result_payload()
            self.image_field_selection = dlg.image_payload()
            self._save_settings()
            self.log("[select] updated datasets/fields selection.")
            self._start_export_estimate()

    def _selected_image_names(self) -> set[str] | None:
        """Names of the image series the user ticked, or ``None`` when names
        cannot express the choice (all images, or none).

        The Fields dialog records *raw stack indices*, which address one file's
        stack table and mean nothing in another.  Batch export therefore carries
        the **names** across files instead, matching the way dataset field
        filters are already applied globally.  "All images" and "no images" are
        both carried by :attr:`image_field_selection` itself (``None`` / empty),
        since neither is a name list.
        """
        selected = self.image_field_selection
        if not selected:                      # None (all) or empty (none)
            return None
        wanted = {int(i) for i in selected}
        names = {entry["name"] for entry in self._gather_image_series_for_dialog()
                 if int(entry.get("raw_index", -1)) in wanted}
        return names or None

    def _image_filter_summary(self, image_names: set[str] | None) -> str:
        """One-word/one-list description of the image selection, for the Log."""
        if image_names is not None:
            return str(sorted(image_names))
        return "ALL" if self.image_field_selection is None else "none"

    def _image_series_to_export(self, image_names: set[str] | None) -> list[dict]:
        """Image series of the currently parsed file that should be exported."""
        return select_image_series(self._gather_image_series_for_dialog(),
                                   names=image_names,
                                   indices=self.image_field_selection)

    def _gather_image_series_for_dialog(self) -> list[dict]:
        """The viewable OBF image series of the parsed .msr (both legacy and modern
        files may carry confocal images), for the Fields dialog + Open-in-viewer."""
        msr_path = self.parsed.get("msr")
        if not msr_path:
            return []
        try:
            from ...core.obf_image_source import list_obf_image_series
            return list_obf_image_series(msr_path)
        except Exception as exc:
            self.log(f"[image] could not list image series: {exc}")
            return []

    def _open_obf_series_group(self, raw_indices) -> bool:
        """Open one file-backed image viewer for the selected OBF series.

        ``ObfImageSource`` exposes every viewable stack through its Series
        dropdown, so opening N selected stacks must not create N windows. The
        first selected stack becomes the current series; the rest remain
        available in that same dropdown.
        """
        msr_path = self.parsed.get("msr")
        if not msr_path:
            return False
        requested = []
        for raw in raw_indices or ():
            try:
                value = int(raw)
            except (TypeError, ValueError):
                continue
            if value not in requested:
                requested.append(value)
        if not requested:
            return False

        source = None
        for raw in requested:
            try:
                from ...core.obf_image_source import ObfImageSource
                source = ObfImageSource(msr_path, raw_stack_index=raw)
                break
            except Exception as exc:
                self.log(f"[image] stack {raw}: {exc}")
        if source is None:
            return False
        try:
            from minflux_viewer.ui.tools.tiff_viewer_window import TiffViewerWindow
        except Exception as exc:
            source.close()
            self.log(f"[image] {exc}")
            return False
        owner = self._owner
        key = f"{Path(msr_path).resolve()}#obf-images"
        if owner is not None and hasattr(owner, "_open_image_viewer"):
            try:
                owner._open_image_viewer(
                    source,
                    key,
                    initial_series_index=int(source.metadata.series_index),
                )
                return True
            except Exception as exc:
                self.log(f"[image] {exc}")
        self._show_child_dialog(TiffViewerWindow(source))
        return True

    def _open_obf_series(self, raw_index) -> bool:
        """Compatibility wrapper for one selected series."""
        return self._open_obf_series_group([raw_index])

    def _derive_global_field_filters(self):
        sel = self.field_selection or {}
        if not sel:
            return None, None
        active = [v for v in sel.values() if v.get("checked", True)]
        if not active:
            return set(), set()
        any_all_mfx = any(v.get("mfx") is None for v in active)
        mfx_sel = None if any_all_mfx else set().union(*(set(v.get("mfx") or []) for v in active))
        any_all_mbm = any(v.get("mbm") is None for v in active)
        mbm_sel = None if any_all_mbm else set().union(*(set(v.get("mbm") or []) for v in active))
        return mfx_sel, mbm_sel

    def _export_current_parsed(self, out_dir: str, formats: list[str],
                               mfx_sel_global=None, mbm_sel_global=None,
                               image_names_global=None):
        export_parsed_result(
            self.parsed, self._msr_state(), out_dir, formats,
            mfx_sel_global=mfx_sel_global, mbm_sel_global=mbm_sel_global,
            image_names=image_names_global,
            image_indices=self.image_field_selection,
            field_selection=self.field_selection,
            log=self.log,
            on_conflict=getattr(self, "_export_conflict", "error") or "error",
        )

    @staticmethod
    def _safe_export_dataset_stem(value: str) -> str:
        """Convert a source dataset label to a safe single-folder stem."""
        import re

        text = str(value or "").strip()
        text = re.sub(r'[<>:"/\\|?*\x00-\x1f]+', "_", text).strip(" .")
        if not text:
            raise ValueError(f"Dataset label {value!r} has no usable export filename.")
        return text

    def _batch_root(self, path_str: str) -> Path:
        """Folder a batch run searches from — the parent when given a file."""
        p = Path(path_str)
        return p.parent if p.is_file() else p

    def _find_msr_files(self, path_str: str, recursive: bool | None = None):
        """``.msr`` files under *path_str*, optionally including sub-folders.

        Case-insensitive on the suffix rather than globbing both cases, so a
        case-insensitive filesystem cannot yield the same file twice.
        """
        if recursive is None:
            recursive = bool(getattr(self, "recursive_check", None)
                             and self.recursive_check.isChecked())
        folder = self._batch_root(path_str)
        if not folder.is_dir():
            return []
        entries = folder.rglob("*") if recursive else folder.glob("*")
        return sorted(
            entry for entry in entries
            if entry.is_file() and entry.suffix.lower() == ".msr")

    def _batch_output_dir(self, out_root: Path, msr_path: Path,
                          in_root: Path, reproduce_tree: bool) -> Path:
        """Where one input file's exports go.

        **Every input file always gets its own folder, named exactly after the
        ``.msr``.**  One ``.msr`` holds several datasets and may hold image
        series too, and the exported files are named after those *datasets*, not
        after the source file — so without a per-file folder two inputs writing
        into one directory could overwrite each other, and the outputs would in
        any case not be attributable to their source.

        *reproduce_tree* additionally mirrors the input's directory path below
        the batch root::

            off:  root/a/b/test.msr  ->  out/test/
            on:   root/a/b/test.msr  ->  out/a/b/test/
        """
        if not reproduce_tree:
            return out_root / msr_path.stem
        try:
            relative = msr_path.parent.relative_to(in_root)
        except ValueError:
            # Outside the batch root (should not happen); keep it addressable
            # rather than writing somewhere unexpected.
            return out_root / msr_path.stem
        return out_root / relative / msr_path.stem

    def _viewer_import_plan(self) -> dict[str, set[str] | None]:
        """Return dataset -> selected mfx fields, or None for all fields."""
        datasets_info = self.parsed.get("datasets", [])
        all_keys = [(ds.get("display_name") or ds.get("did") or "dataset") for ds in datasets_info]
        selected = self._selected_items()
        plan: dict[str, set[str] | None] = {}
        used_tree_selection = False
        for item in selected:
            ds_item = self._dataset_item_of(item)
            if ds_item is None:
                continue
            key = ds_item.text(0)
            path = self.fullpath_by_item.get(self._item_id(item), "")
            if item.text(1) == "dataset":
                plan[key] = None
                used_tree_selection = True
            elif path == "mfx":
                plan[key] = None
                used_tree_selection = True
            elif path.startswith("mfx/"):
                field = path.split("/", 1)[1].split("/", 1)[0]
                if key in plan and plan[key] is None:
                    pass
                else:
                    plan.setdefault(key, set()).add(field)
                used_tree_selection = True
        if used_tree_selection:
            return plan
        if self.field_selection:
            for key in all_keys:
                sel = self.field_selection.get(key, {"checked": True, "mfx": None})
                if not sel.get("checked", True):
                    continue
                fields = sel.get("mfx")
                plan[key] = None if fields is None else set(fields)
            return plan
        return {key: None for key in all_keys}

    def _viewer_selected_dataset_names(self, import_plan: dict[str, set[str] | None]) -> list[str]:
        from ...msr.session_alignment import AlignmentSession
        session = AlignmentSession(self.parsed, self._msr_state(), log=self.log)
        return session._viewer_selected_dataset_names(import_plan)

    def _viewer_zroot_for_dataset_name(self, name: str) -> str | None:
        from ...msr.session_alignment import AlignmentSession
        session = AlignmentSession(self.parsed, self._msr_state(), log=self.log)
        return session._viewer_zroot_for_dataset_name(name)

    def _ask_viewer_channel_alignment(self, import_plan: dict[str, set[str] | None]):
        """Ask for confocal mapping plus reference dataset/alignment mode.

        Returns ``(mode, ref_name)``, ``None`` when no dataset-overlay alignment
        applies (the dialog can still have collected confocal mapping options), or
        ``_ALIGN_CANCELLED`` when the user cancelled — which must abort the
        whole open, not just skip the transforms.
        """
        self._pending_confocal_options = None
        if self.parsed.get("mode") != "modern":
            return None
        selected = self._viewer_selected_dataset_names(import_plan)
        if not selected:
            return None

        confocal_candidates = []
        msr_path = Path(self.parsed.get("msr", ""))
        selected_info = []
        for ds in self.parsed.get("datasets", []) or []:
            key = ds.get("display_name") or ds.get("did") or "dataset"
            if key in selected:
                selected_info.append({"dataset_key": key, "did": ds.get("did", "")})
        if msr_path.is_file():
            try:
                from ...core.confocal_mapping import discover_confocal_candidates

                confocal_candidates = discover_confocal_candidates(msr_path, selected_info)
                if confocal_candidates:
                    self.log(
                        f"[viewer] detected {len(confocal_candidates)} potential fluorescent "
                        "image channel(s) by calibrated XY/ROI match (1% tolerance)"
                    )
            except Exception as exc:
                self.log(f"[warn] confocal candidate detection failed: {exc}")

        if len(selected) < 2 and not confocal_candidates:
            return None

        reserved_names: set[str] = set()
        MFSTATE = self._msr_state()
        for name in selected:
            mfx = MFSTATE.mfx_map.get(name)
            reserved_names.update(
                getattr(getattr(mfx, "dtype", None), "names", ()) or ()
            )
        from ...core.attributes import DERIVED_ATTRIBUTE_NAMES

        reserved_names.update(DERIVED_ATTRIBUTE_NAMES)
        reserved_names.update({"loc_x", "loc_y", "loc_z", "xnm", "ynm", "znm"})

        def mbm_check(ref_name: str) -> tuple[bool, str]:
            ordered = [ref_name] + [n for n in selected if n != ref_name]
            return self._viewer_mbm_alignment_available(ordered)

        dlg = ViewerAlignmentDialog(
            selected,
            mbm_check,
            self,
            confocal_candidates=confocal_candidates,
            reserved_names=reserved_names,
        )
        if dlg.exec() != QDialog.DialogCode.Accepted:
            self.log("[viewer align] cancelled; open aborted")
            return _ALIGN_CANCELLED
        self._pending_confocal_options = dlg.confocal_options()
        if len(selected) < 2:
            return None
        if dlg.open_individual():
            self.log("[viewer] opening each selected dataset as an individual dataset")
            return _OPEN_INDIVIDUAL
        mode = dlg.alignment_mode()
        ref_name = dlg.reference_dataset()
        self.log(f"[viewer align] selected mode: {mode}; reference: {ref_name}")
        return mode, ref_name

    def _viewer_mbm_alignment_available(self, selected: list[str]) -> tuple[bool, str]:
        from ...msr.session_alignment import AlignmentSession
        session = AlignmentSession(self.parsed, self._msr_state(), log=self.log)
        return session._viewer_mbm_alignment_available(selected)

    def _viewer_used_bead_gri_for_dataset(self, name: str, *, log_prefix: str='[viewer align]') -> set[int] | None:
        from ...msr.session_alignment import AlignmentSession
        session = AlignmentSession(self.parsed, self._msr_state(), log=self.log)
        return session._viewer_used_bead_gri_for_dataset(name, log_prefix=log_prefix)

    def _read_zarr_attrs_safe(self, zroot: str, path: str) -> dict:
        from ...msr.session_alignment import AlignmentSession
        session = AlignmentSession(self.parsed, self._msr_state(), log=self.log)
        return session._read_zarr_attrs_safe(zroot, path)

    def _extract_used_bead_names(self, attrs: dict) -> set[str]:
        from ...msr.session_alignment import AlignmentSession
        session = AlignmentSession(self.parsed, self._msr_state(), log=self.log)
        return session._extract_used_bead_names(attrs)

    def _extract_point_metadata_beads(self, attrs: dict) -> tuple[dict[str, int], set[int]]:
        from ...msr.session_alignment import AlignmentSession
        session = AlignmentSession(self.parsed, self._msr_state(), log=self.log)
        return session._extract_point_metadata_beads(attrs)

    def _filter_mbm_points_to_used_beads(self, name: str, points, *, log_prefix: str='[viewer align]') -> np.ndarray:
        from ...msr.session_alignment import AlignmentSession
        session = AlignmentSession(self.parsed, self._msr_state(), log=self.log)
        return session._filter_mbm_points_to_used_beads(name, points, log_prefix=log_prefix)

    def _viewer_dataset_looks_3d(self, name: str) -> bool:
        from ...msr.session_alignment import AlignmentSession
        session = AlignmentSession(self.parsed, self._msr_state(), log=self.log)
        return session._viewer_dataset_looks_3d(name)

    def _viewer_channel_alignment_transforms(self, import_plan: dict[str, set[str] | None], mode: str, ref_name: str | None=None) -> dict[str, dict]:
        from ...msr.session_alignment import AlignmentSession
        session = AlignmentSession(self.parsed, self._msr_state(), log=self.log)
        return session._viewer_channel_alignment_transforms(import_plan, mode, ref_name)

    def _warn_if_poor_viewer_alignment(self, mode: str, transforms: dict) -> None:
        """Warn (Log + a non-blocking box) when a bead-based ("mbm info") overlay
        alignment has a poor fiducial fit, so a large registration error is not
        applied silently. No-op for the fallback modes (they carry no bead RMSE)."""
        if mode != "mbm info" or not transforms:
            return
        from ...core.bead_alignment import RMSE_WARN_NM

        poor = []
        for name, tf in transforms.items():
            rmse = float(tf.get("rmse_xy_nm", 0.0) or 0.0)
            if int(tf.get("matched_bead_count", 0) or 0) > 0 and rmse > RMSE_WARN_NM:
                poor.append((name, rmse, int(tf.get("matched_bead_count", 0))))
        if not poor:
            return
        detail = ", ".join(f"{n} ({r:.1f} nm, {b} beads)" for n, r, b in poor)
        msg = (
            f"Bead (mbm info) channel alignment has a poor fiducial fit "
            f"(RMSE XY > {RMSE_WARN_NM:.0f} nm) for: {detail}. The beads disagree among "
            f"themselves, so the overlaid channels may be misregistered. If your sample "
            f"contains a common feature/fiducial, re-open with 'data centroid'; or exclude "
            f"suspect beads in 'Show beads drift'."
        )
        self.log("[viewer align] WARNING: " + msg)
        QMessageBox.warning(self, "Open in MINFLUX viewer", msg)

    def _ask_viewer_alignment_fallback(self, reason: str) -> str | None:
        box = QMessageBox(self)
        box.setWindowTitle("Open in MINFLUX viewer")
        box.setIcon(QMessageBox.Icon.Warning)
        box.setText("Bead-based channel alignment is not available for this MSR file.")
        box.setInformativeText(
            f"{reason}\n\n"
            "Choose a fallback alignment method for the viewer overlay:"
        )
        stage_button = box.addButton("Stage origin", QMessageBox.ButtonRole.AcceptRole)
        centroid_button = box.addButton("Data centroid", QMessageBox.ButtonRole.AcceptRole)
        box.addButton(QMessageBox.StandardButton.Cancel)
        box.exec()
        clicked = box.clickedButton()
        if clicked is stage_button:
            self.log("[viewer align] using fallback: stage origin")
            return "stage origin"
        if clicked is centroid_button:
            self.log("[viewer align] using fallback: data centroid")
            return "data centroid"
        self.log("[viewer align] fallback alignment cancelled; opening without transforms")
        return None

    def _viewer_fallback_alignment_transforms(self, selected: list[str], mode: str) -> dict[str, dict]:
        from ...msr.session_alignment import AlignmentSession
        session = AlignmentSession(self.parsed, self._msr_state(), log=self.log)
        return session._viewer_fallback_alignment_transforms(selected, mode)

    def _viewer_transform_record(self, ref_name: str, moving_name: str, matrix_4x4: np.ndarray, *, method: str) -> dict:
        from ...msr.session_alignment import AlignmentSession
        session = AlignmentSession(self.parsed, self._msr_state(), log=self.log)
        return session._viewer_transform_record(ref_name, moving_name, matrix_4x4, method=method)

    def _mfx_centroid_nm(self, mfx) -> np.ndarray:
        from ...msr.session_alignment import AlignmentSession
        session = AlignmentSession(self.parsed, self._msr_state(), log=self.log)
        return session._mfx_centroid_nm(mfx)

    # ------------------------------------------------------------------
    # Alignment
    # ------------------------------------------------------------------

    def _ensure_single_file_parsed(self) -> bool:
        from ...msr import parse_msr_general, pick_one_msr
        ipath = self.input_path_edit.text().strip()
        tmp = self._temp_dir()
        if self.mode_folder.isChecked():
            QMessageBox.information(self, "Align channel", "Channel alignment is currently available only in single-file mode.")
            return False
        if self.parsed:
            return True
        msr = pick_one_msr(ipath)
        if not msr:
            self.log("[align] no msr file found.")
            QMessageBox.critical(self, "Align channel", "Choose a valid .msr file first.")
            return False
        self.log(f"[align] parsing before alignment: {msr}")
        self._store_parse_result(parse_msr_general(msr, tmp, log=self.log))
        return True

    def _channel_alignment_defaults(self):
        from ...msr import pick_one_msr
        out_dir = Path(self.out_dir_edit.text().strip() or self._temp_dir())
        msr = pick_one_msr(self.input_path_edit.text().strip())
        stem = Path(msr).stem if msr else "channel_alignment"
        return {
            "beads": str(out_dir / f"{stem}_matched_bead_summary.csv"),
            "transform": str(out_dir / f"{stem}_alignment_transform.csv"),
        }

    def _write_alignment_beads_csv(self, out_path: str, results):
        Path(out_path).parent.mkdir(parents=True, exist_ok=True)
        with open(out_path, "w", newline="", encoding="utf-8") as fh:
            writer = csv.writer(fh)
            writer.writerow([
                "bead_id", "reference_channel_name", "reference_count_used", "reference_x_nm", "reference_y_nm", "reference_z_nm",
                "moving_channel_name", "moving_count_used", "moving_x_nm", "moving_y_nm", "moving_z_nm",
                "moving_aligned_x_nm", "moving_aligned_y_nm", "moving_aligned_z_nm",
            ])
            for result in results:
                for idx, bead_id in enumerate(result.bead_ids):
                    writer.writerow([
                        int(bead_id), result.channel1_name, int(result.channel1_counts[idx]), *result.channel1_xyz[idx].tolist(),
                        result.channel2_name, int(result.channel2_counts[idx]), *result.channel2_xyz[idx].tolist(),
                        *result.channel2_aligned_xyz[idx].tolist(),
                    ])

    def _write_alignment_transform_csv(self, out_path: str, results):
        Path(out_path).parent.mkdir(parents=True, exist_ok=True)
        matrix_fields = [
            f"matrix_4x4_{row}{col}{'_nm' if col == 3 and row < 3 else ''}"
            for row in range(4)
            for col in range(4)
        ]
        with open(out_path, "w", newline="", encoding="utf-8") as fh:
            writer = csv.writer(fh)
            writer.writerow([
                "reference_channel_name", "moving_channel_name", "transform_type",
                "matched_bead_count", "rmse_xy_nm",
                "translation_x_nm", "translation_y_nm", "translation_z_nm",
                *matrix_fields,
            ])
            for result in results:
                matrix = np.asarray(result.matrix_4x4, dtype=np.float64)
                writer.writerow([
                    result.channel1_name, result.channel2_name, result.transform_type,
                    int(result.bead_ids.size), float(result.rmse),
                    float(result.translation[0]), float(result.translation[1]),
                    float(result.z_translation),
                    *[float(matrix[row, col]) for row in range(4) for col in range(4)],
                ])

    def _aligned_mbm_points(self, name):
        from ...msr.session_alignment import AlignmentSession
        session = AlignmentSession(self.parsed, self._msr_state(), log=self.log)
        return session._aligned_mbm_points(name)

    def _bead_gri_to_rid(self, names) -> dict[int, str]:
        from ...msr.session_alignment import AlignmentSession
        session = AlignmentSession(self.parsed, self._msr_state(), log=self.log)
        return session._bead_gri_to_rid(names)

    @staticmethod
    def _format_bead_ids(gris, gri_to_rid: dict[int, str] | None=None) -> str:
        from ...msr.session_alignment import AlignmentSession
        return AlignmentSession._format_bead_ids(gris, gri_to_rid)

    def _on_show_beads_drift(self):
        MFSTATE = self._msr_state()
        from .beads_drift import gather_msr_bead_drift
        from .beads_drift_dialog import BeadsDriftDialog

        datasets = (self.parsed or {}).get("datasets") or []
        bead_data = gather_msr_bead_drift(
            datasets, getattr(MFSTATE, "mbm_map", {}),
            meta_map=getattr(MFSTATE, "mbm_meta_map", {}))
        if not bead_data:
            QMessageBox.information(
                self, "Show beads drift",
                "The currently parsed .msr file has no MBM/bead data to display.\n\n"
                "Parse a modern multi-channel .msr that contains 'grd/mbm/points'.")
            return
        dlg = BeadsDriftDialog(bead_data, unchecked_gris=self._bead_unchecked_gris, parent=self,
                               drift_correction=self._run_bead_drift_correction,
                               prefs=getattr(self._state, "prefs", None))
        dlg.accepted.connect(lambda d=dlg, data=bead_data: self._apply_beads_drift_selection(d, data))
        self._show_child_dialog(dlg)

    def _drift_correction_entries(self) -> list[dict]:
        """``{"name", "mfx", "points"}`` per parsed dataset that has bead data.

        Both arrays are the **live** parsed objects, not copies: ``mfx_map``
        values and the dataset entries' ``_mfx`` are the same array, so an
        in-place correction reaches everything downstream, including
        "Open in MINFLUX viewer".
        """
        MFSTATE = self._msr_state()
        mfx_map = getattr(MFSTATE, "mfx_map", {}) or {}
        mbm_map = getattr(MFSTATE, "mbm_map", {}) or {}
        entries = []
        for d in (self.parsed or {}).get("datasets") or []:
            name = d.get("display_name") or d.get("did") or "dataset"
            mfx, points = mfx_map.get(name), mbm_map.get(name)
            if mfx is None or points is None or not getattr(points, "size", 0):
                continue
            entries.append({"name": name, "mfx": mfx, "points": points})
        return entries

    def _run_bead_drift_correction(self, gris) -> None:
        """Preview, and on confirmation apply, a drift correction from *gris*.

        Rewrites ``loc`` from the preserved ``lnc``, so it is idempotent and can
        be re-run with a different bead selection without reloading the file.
        """
        from .drift_correction_dialog import DriftCorrectionDialog

        entries = self._drift_correction_entries()
        if not entries:
            QMessageBox.information(
                self, "Drift correction",
                "No parsed dataset has both localization and MBM bead data, so "
                "there is nothing to correct.")
            return
        dlg = DriftCorrectionDialog(entries, gris, parent=self)
        try:
            accepted = dlg.exec() == QDialog.DialogCode.Accepted
            applied = list(dlg.applied)
        finally:
            # Parented to the reader, so without this it would linger (with its
            # plots) for as long as the reader is open.
            dlg.deleteLater()
        if not accepted:
            self.log("[drift] correction cancelled — no coordinates changed.")
            return
        ids = ", ".join(str(int(g)) for g in sorted(gris))
        self.log(f"[drift] correction applied from {len(list(gris))} bead(s): {ids}")
        for res in applied:
            if res.get("error"):
                self.log(f"[drift] {res.get('name')}: FAILED — {res['error']}")
            else:
                self.log(
                    f"[drift] {res.get('name')}: 'loc' rewritten for "
                    f"{res['n_rows']:,} localizations "
                    f"(median shift {res['median_shift_nm']:.2f} nm, "
                    f"max {res['max_shift_nm']:.2f} nm)")
        self.log("[drift] 'Open in MINFLUX viewer' will now load the corrected coordinates.")

    def _apply_beads_drift_selection(self, dlg, bead_data) -> None:
        try:
            self._bead_unchecked_gris = dlg.unchecked_gris()
            gri_to_rid = {int(b["gri"]): str(b.get("rid", "") or "")
                          for d in bead_data for b in d.get("beads", [])}
            gri_to_rid = {g: r for g, r in gri_to_rid.items() if r}
            kept = sorted(int(g) for g in dlg.selected_gris())
            excluded = sorted(int(g) for g in self._bead_unchecked_gris)
            self.log(f"[beads] selection applied: {len(kept)} bead(s) kept: "
                     f"{self._format_bead_ids(kept, gri_to_rid)}")
            self.log(f"[beads] {len(excluded)} bead(s) excluded from alignment: "
                     f"{self._format_bead_ids(excluded, gri_to_rid)}")
        except Exception as exc:
            self.log(f"[beads] selection apply failed: {exc}")

    def _mbm_transform_type(self) -> str:
        """The MBM-handling transform mode from Preferences (Preferences > MBM
        Handling > transform_type); falls back to the rigid-XY + translational-Z
        default when no app state / preference is available."""
        from ...core.bead_alignment import TRANSFORM_RIGID_XY_Z

        default = TRANSFORM_RIGID_XY_Z
        if self._state is None:
            return default
        return (self._state.prefs.get("mbm_handling", {}) or {}).get(
            "transform_type", default) or default

    def _mbm_naming_metadata(self, key: str, zroot) -> tuple[dict, list]:
        from ...msr.session_alignment import AlignmentSession
        session = AlignmentSession(self.parsed, self._msr_state(), log=self.log)
        return session._mbm_naming_metadata(key, zroot)

    def _combined_loc_bounds_nm(self, names) -> tuple[np.ndarray, np.ndarray] | None:
        from ...msr.session_alignment import AlignmentSession
        session = AlignmentSession(self.parsed, self._msr_state(), log=self.log)
        return session._combined_loc_bounds_nm(names)

    def on_align_channel(self):
        MFSTATE = self._msr_state()
        from ...core.bead_alignment import MIN_BEADS_FOR_ALIGNMENT, align_channels_from_arrays

        if not self._ensure_single_file_parsed():
            return
        if self.parsed.get("mode") != "modern":
            QMessageBox.information(self, "Align channel", "Channel alignment is available only for modern MINFLUX datasets.")
            return
        datasets = self.parsed.get("datasets") or []
        if not datasets:
            QMessageBox.critical(self, "Align channel", "No datasets found.")
            return
        if len(datasets) == 1:
            # Single channel → no alignment to compute; just show the beads +
            # data region and their drift (skip the save-outputs dialog).
            self._show_single_channel_beads(datasets[0])
            return
        try:
            ref_name = datasets[0].get("display_name") or datasets[0].get("did") or "channel_1"
            ref_mbm = self._aligned_mbm_points(ref_name)
            if ref_mbm is None:
                QMessageBox.critical(self, "Align channel", "Could not find bead points (`grd/mbm/points`) for the reference channel.")
                return
            ref_mbm_for_alignment = self._filter_mbm_points_to_used_beads(
                ref_name, ref_mbm, log_prefix="[align]"
            )
            transform_type = self._mbm_transform_type()
            self.log(f"[align] MBM-handling transform: {transform_type}")
            results = []
            loc_bound_names = [ref_name]
            for ds in datasets[1:]:
                moving_name = ds.get("display_name") or ds.get("did") or "channel"
                loc_bound_names.append(moving_name)
                moving_mbm = self._aligned_mbm_points(moving_name)
                moving_mfx = MFSTATE.mfx_map.get(moving_name)
                if moving_mbm is None:
                    raise ValueError(f"Could not find bead points (`grd/mbm/points`) for {moving_name}.")
                if moving_mfx is None:
                    raise ValueError(f"Could not find `mfx` localizations for {moving_name}.")
                moving_mbm_for_alignment = self._filter_mbm_points_to_used_beads(
                    moving_name, moving_mbm, log_prefix="[align]"
                )
                result = align_channels_from_arrays(
                    ref_name, ref_mbm_for_alignment, moving_name, moving_mbm_for_alignment,
                    transform_type=transform_type)
                if int(result.bead_ids.size) < MIN_BEADS_FOR_ALIGNMENT:
                    raise ValueError(
                        f"Only {int(result.bead_ids.size)} matched bead(s) between "
                        f"'{ref_name}' and '{moving_name}'; at least {MIN_BEADS_FOR_ALIGNMENT} "
                        f"are required for alignment "
                        f"(check the bead selection in 'Show beads drift').")
                results.append(result)
        except Exception as exc:
            self.log(f"[align] failed: {exc}")
            QMessageBox.critical(self, "Align channel", f"Alignment failed:\n{exc}")
            return
        self.log(f"[align] reference channel: {ref_name}")
        align_names = {ref_name} | {r.channel2_name for r in results}
        gri_to_rid = self._bead_gri_to_rid(align_names)
        excluded = sorted(int(g) for g in (self._bead_unchecked_gris or set()))
        self.log(
            f"[align] user-excluded {len(excluded)} bead(s): "
            f"{self._format_bead_ids(excluded, gri_to_rid)}")
        for result in results:
            used = [int(b) for b in result.bead_ids]
            self.log(
                f"[align] {result.channel2_name} -> {result.channel1_name}; used "
                f"{len(used)} bead(s): {self._format_bead_ids(used, gri_to_rid)}; "
                f"transform={result.transform_type}; "
                f"translation_nm=({result.translation[0]:.3f}, {result.translation[1]:.3f}); "
                f"rmse_xy_nm={result.rmse:.3f}")

        dlg = AlignmentSaveDialog(
            self._channel_alignment_defaults(), self._alignment_save_options, self)
        if dlg.exec() != QDialog.DialogCode.Accepted:
            self.log("[align] save cancelled.")
            return
        payload = dlg.result_payload()
        # Remember the checkbox states for next time.
        self._alignment_save_options = {
            k: bool(payload[k])
            for k in ("save_beads", "save_transform", "show_plot")
        }
        self._save_settings()
        try:
            if payload["save_beads"]:
                self._write_alignment_beads_csv(payload["path_beads"], results)
                self.log(f"[align] wrote beads CSV: {payload['path_beads']}")
            if payload["save_transform"]:
                self._write_alignment_transform_csv(payload["path_transform"], results)
                self.log(f"[align] wrote transform CSV: {payload['path_transform']}")
        except Exception as exc:
            self.log(f"[align] save failed: {exc}")
            QMessageBox.critical(self, "Align channel", f"Failed to save alignment outputs:\n{exc}")
            return
        if payload["show_plot"]:
            data_bounds = self._combined_loc_bounds_nm(loc_bound_names)
            self._show_child_dialog(
                AlignmentPlotWindow(results, self, data_bounds_nm=data_bounds,
                                    requested_transform_type=transform_type,
                                    prefs=getattr(self._state, "prefs", None))
            )
        from ...core.bead_alignment import RMSE_WARN_NM, alignment_quality_report

        reports = [alignment_quality_report(r) for r in results]
        summary = [f"Reference channel: {ref_name}"] + [
            f"{result.channel2_name}: matched {result.bead_ids.size}, translation ({result.translation[0]:.3f}, {result.translation[1]:.3f}) nm, RMSE XY {result.rmse:.3f} nm"
            for result in results
        ]
        poor = [rep for rep in reports if rep["warn"]]
        if poor:
            warn_text = (
                f"Poor fiducial registration (RMSE XY > {RMSE_WARN_NM:.0f} nm) for: "
                + ", ".join(f"{rep['moving']} ({rep['rmse_xy_nm']:.1f} nm)" for rep in poor)
                + ". The beads disagree among themselves; a single transform cannot register "
                "the channels accurately. Consider 'data centroid' alignment if the sample has a "
                "common feature, or exclude suspect beads in 'Show beads drift'."
            )
            summary.append("WARNING: " + warn_text)
            self.log("[align] WARNING: " + warn_text)
        self.log("[align] completed. " + " | ".join(summary))
        if not payload["show_plot"]:
            box = QMessageBox.warning if poor else QMessageBox.information
            box(self, "Align channel", "Alignment completed.\n\n" + "\n".join(summary))

    def _show_single_channel_beads(self, ds):
        """Single-channel 'Align channel': no alignment to compute, so show the
        channel's beads + the data region as a scatter and a per-bead **drift**
        table (peak-to-peak per axis; resid columns are N/A). Skips the
        save-outputs dialog."""

        from .beads_drift import gather_msr_bead_drift, single_channel_bead_summary
        MFSTATE = self._msr_state()
        name = ds.get("display_name") or ds.get("did") or "channel_1"
        gathered = gather_msr_bead_drift(
            [ds], MFSTATE.mbm_map, getattr(MFSTATE, "mbm_meta_map", None))
        beads = gathered[0]["beads"] if gathered else []
        data = single_channel_bead_summary(name, beads)
        if data is None:
            QMessageBox.information(
                self, "Align channel",
                f"No bead (grd/mbm/points) data to show for the single channel '{name}'.")
            return
        bounds = self._combined_loc_bounds_nm([name])
        excluded = sorted(int(g) for g in (self._bead_unchecked_gris or set()))
        self.log(
            f"[align] single channel '{name}': no alignment needed — showing "
            f"{len(beads)} bead(s) + the data region (per-bead drift). "
            f"user-excluded {len(excluded)} bead(s).")
        self._show_child_dialog(
            AlignmentPlotWindow(
                [], self, data_bounds_nm=bounds, single_channel=data,
                prefs=getattr(self._state, "prefs", None)))

    # ------------------------------------------------------------------
    # Export
    # ------------------------------------------------------------------

    def _run_batch_export(self, ipath: str, out_dir: str, formats: list[str],
                          tmp: str) -> None:
        """Parse and export every ``.msr`` under the input folder.

        Failures are reported, not swallowed: an empty search and a run in which
        some files failed both end in a dialog, because the alternative is a
        button that visibly does nothing.
        """

        recursive = bool(self.recursive_check.isChecked())
        reproduce_tree = bool(self.reproduce_tree_check.isChecked())
        in_root = self._batch_root(ipath)
        files = self._find_msr_files(ipath, recursive)
        if not files:
            where = "folder and its sub-folders" if recursive else "folder"
            hint = "" if recursive else (
                "\n\nThe search is not recursive. Tick “recursive” next to the "
                "input field to include sub-folders.")
            message = (f"No .msr files were found in the {where}:\n{in_root}{hint}")
            self.log(f"[batch] no .msr files found in the {where}: {in_root}")
            QMessageBox.warning(self, "Nothing to export", message)
            return

        mfx_sel_global, mbm_sel_global = self._derive_global_field_filters()
        # Captured from the previewed file: raw stack indices do not transfer
        # between files, names do.
        image_names_global = self._selected_image_names()
        self.log(
            f"[batch] exporting {len(files)} file(s) from {in_root} "
            f"({'recursive' if recursive else 'top level only'}, "
            f"{'mirroring the input tree' if reproduce_tree else 'one sub-folder per file'}) "
            f"as {', '.join(formats)}; global filters: "
            f"mfx={'ALL' if mfx_sel_global is None else sorted(mfx_sel_global)}; "
            f"mbm={'ALL' if mbm_sel_global is None else sorted(mbm_sel_global)}; "
            f"images={self._image_filter_summary(image_names_global)}")

        targets = {msr: self._batch_output_dir(Path(out_dir), msr, in_root,
                                               reproduce_tree)
                   for msr in files}
        # Each file owns a folder named after it, so the only way two inputs can
        # share one is a recursive search finding the same file name in
        # different sub-folders while the input tree is *not* being mirrored.
        # Refuse up front rather than let the second file overwrite the first.
        collisions: dict[Path, list[Path]] = {}
        for msr, target in targets.items():
            collisions.setdefault(target, []).append(msr)
        clashing = {t: srcs for t, srcs in collisions.items() if len(srcs) > 1}
        if clashing:
            detail = "\n".join(
                f"• {target.name}/  ←  "
                + ", ".join(str(s.relative_to(in_root)) for s in srcs)
                for target, srcs in sorted(clashing.items()))
            self.log(f"[batch] aborted: {len(clashing)} output folder(s) would "
                     f"receive more than one input file.")
            QMessageBox.warning(
                self, "Output folders would collide",
                "These input files share a name and would export into the same "
                f"folder, overwriting each other:\n\n{detail}\n\n"
                "Tick “reproduce input folder structure” to keep them apart.")
            return

        # Off the UI thread: a batch is minutes of parsing and writing, and
        # running it inline froze both this dialog and the main window.
        self._batch_context = {
            "out_dir": out_dir,
            "total": len(files),
            "kind": "batch",
            "status_header": _BATCH_STATUS_HEADER,
        }
        task = _BatchExportTask(
            files, targets, formats, tmp,
            mfx_sel=mfx_sel_global, mbm_sel=mbm_sel_global,
            image_names=image_names_global,
            image_indices=self.image_field_selection,
            field_selection=self.field_selection,
            on_conflict=getattr(self, "_export_conflict", "error") or "error",
        )
        task.signals.message.connect(self.log)
        task.signals.progress.connect(self._on_batch_progress)
        task.signals.finished.connect(self._on_batch_finished)
        self._batch_task = task
        self._set_batch_running(True, len(files))
        from minflux_viewer.ui.common.background_tasks import shared_thread_pool

        shared_thread_pool("msr-work").start(task)

    def _run_single_export(self, out_dir: str, formats: list[str]) -> None:
        """Start a single parsed export off-thread, like the existing batch path."""
        task = _ParsedExportTask(
            self.parsed,
            self._msr_state(),
            out_dir,
            formats,
            field_selection=self.field_selection,
            image_indices=self.image_field_selection,
            on_conflict=getattr(self, "_export_conflict", "error") or "error",
        )
        task.signals.message.connect(self.log)
        task.signals.progress.connect(self._on_batch_progress)
        task.signals.finished.connect(self._on_batch_finished)
        self._batch_context = {
            "out_dir": out_dir,
            "total": 1,
            "kind": "single",
            "status_header": "MSR export",
        }
        self._batch_task = task
        self._set_batch_running(True, 1)
        from minflux_viewer.ui.common.background_tasks import shared_thread_pool

        shared_thread_pool("msr-work").start(task)

    # -- batch progress -------------------------------------------------

    def _set_batch_running(self, running: bool, total: int = 0) -> None:
        self._progress_row.setVisible(running)
        self._export_button.setEnabled(not running)
        self._parse_button.setEnabled(not running and self.mode_folder.isChecked())
        if running:
            self._progress_bar.setValue(0)
            kind = str((self._batch_context or {}).get("kind", "batch"))
            self._progress_label.setText("" if kind == "single" else f"0 / {total}")
            self._last_batch_pct = -1
            if self._state is not None:
                from ...core.app_state import format_progress_bar
                self._state.log_progress(format_progress_bar(0.0))
                header = str(
                    (self._batch_context or {}).get(
                        "status_header", _BATCH_STATUS_HEADER
                    )
                )
                self._state.status_progress(header, 0.0)
        else:
            self.setWindowTitle(_READER_TITLE)

    def _on_batch_progress(self, fraction: float, current: str) -> None:
        fraction = max(0.0, min(1.0, float(fraction)))
        self._progress_bar.setValue(int(round(fraction * 1000)))
        total = int((self._batch_context or {}).get("total", 0))
        done = int(fraction * total)
        kind = str((self._batch_context or {}).get("kind", "batch"))
        if kind == "single":
            # A single file has no "n of m" to count: the progress bar already
            # shows the percentage, so the line names what is being written.
            self._progress_label.setText(current)
        else:
            self._progress_label.setText(
                f"{done} / {total}" + (f"  ·  {current}" if current else ""))
        # The title carries the percentage so it is legible even when the
        # dialog is behind another window, as requested.
        operation = "batch export" if kind == "batch" else "export"
        self.setWindowTitle(
            f"{_READER_TITLE} ({operation} {fraction * 100:.1f}%…)"
        )
        pct = int(fraction * 1000)                    # throttle at 0.1 %
        if self._state is not None and pct != self._last_batch_pct:
            self._last_batch_pct = pct
            from ...core.app_state import format_progress_bar
            self._state.log_progress(format_progress_bar(fraction))
            header = str(
                (self._batch_context or {}).get(
                    "status_header", _BATCH_STATUS_HEADER
                )
            )
            # Name the file being written in the status line: on a long export
            # the percentage alone does not say what is taking the time.
            self._state.status_progress(
                f"{header} — {current}" if current else header, fraction)

    def _cancel_batch_export(self) -> None:
        if self._batch_task is not None:
            self._batch_task.cancel()
            self._cancel_batch_btn.setEnabled(False)
            kind = str((self._batch_context or {}).get("kind", "batch"))
            unit = "file" if kind == "batch" else "dataset"
            self.log(f"[{kind}] stopping after the current {unit}…")

    def _on_batch_finished(self, exported: int, failures: list, cancelled: bool) -> None:
        context = self._batch_context or {}
        out_dir = str(context.get("out_dir", ""))
        total = int(context.get("total", 0))
        kind = str(context.get("kind", "batch"))
        self._batch_task = None
        self._batch_context = None
        self._cancel_batch_btn.setEnabled(True)
        self._set_batch_running(False)
        state = self._state
        if state is not None:
            from ...core.app_state import format_progress_bar
            state.log_progress(format_progress_bar(1.0, done=True), final=True)
            label = "MSR batch export" if kind == "batch" else "MSR export"
            unit = "file(s)" if kind == "batch" else "dataset(s)"
            state.status_message.emit(
                f"{label}: {'stopped' if cancelled else 'done'} "
                f"({exported}/{total} {unit}).")
        if kind == "single":
            if failures:
                reason = failures[0][1]
                self.log(f"[error] export failed: {reason}")
                QMessageBox.critical(self, "Export failed", reason)
            elif cancelled:
                self.log(f"[done] Export stopped; output is in {out_dir}.")
            else:
                self.log(f"[done] Export complete: {out_dir}")
            return
        verb = "stopped after" if cancelled else "complete:"
        self.log(f"[done] Batch export {verb} {exported}/{total} file(s) "
                 f"exported to {out_dir}"
                 + (f", {len(failures)} failed." if failures else "."))
        if failures:
            detail = "\n".join(f"• {Path(path).name}: {reason}"
                               for path, reason in failures[:10])
            if len(failures) > 10:
                detail += f"\n… and {len(failures) - 10} more (see the Log)."
            QMessageBox.warning(
                self, "Batch export finished with errors",
                f"Exported {exported} of {total} file(s) to:\n{out_dir}\n\n"
                f"Failed:\n{detail}")
        elif cancelled:
            QMessageBox.information(
                self, "Batch export stopped",
                f"Stopped after {exported} of {total} file(s).\n"
                f"What was exported is in:\n{out_dir}")
        else:
            QMessageBox.information(
                self, "Batch export complete",
                f"Exported {exported} file(s) to:\n{out_dir}")

    def on_ok(self):
        out_dir_text = self.out_dir_edit.text().strip() or self._temp_dir()
        formats = []
        if self.fmt_mat.isChecked():
            formats.append("mat")
        if self.fmt_npy.isChecked():
            formats.append("npy")
        if self.fmt_json.isChecked():
            formats.append("json")
        if self.fmt_csv.isChecked():
            formats.append("csv")
        if self.fmt_zarr.isChecked():
            formats.append("zarr")
        if not formats:
            QMessageBox.critical(self, "No formats", "Pick at least one export format.")
            return
        if self._export_estimates:
            from ...core.export_size import text_export_warning

            warning = text_export_warning(self._export_estimates, formats)
            if warning:
                choice = QMessageBox.warning(
                    self,
                    "Very large text export",
                    warning + "\n\nContinue with this export?",
                    QMessageBox.StandardButton.Yes
                    | QMessageBox.StandardButton.Cancel,
                    QMessageBox.StandardButton.Cancel,
                )
                if choice != QMessageBox.StandardButton.Yes:
                    return
        try:
            out_dir = str(prepare_export_output_dir(out_dir_text))
        except ValueError as exc:
            self.log(f"[error] export preflight failed: {exc}")
            QMessageBox.critical(self, "Invalid output folder", str(exc))
            return
        self._save_settings()
        tmp = self._temp_dir()
        ipath = self.input_path_edit.text().strip()
        self._export_conflict = self._ask_export_conflict(out_dir, formats)
        if self._export_conflict is None:
            return
        if self.mode_folder.isChecked():
            self._run_batch_export(ipath, out_dir, formats, tmp)
            return
        if not self.parsed:
            from ...msr import parse_msr_general, pick_one_msr
            msr = pick_one_msr(ipath)
            if not msr:
                self.log("no msr file found!")
                return
            self._store_parse_result(parse_msr_general(msr, tmp, log=self.log))
        # Formatting tens or hundreds of millions of text values is inherently
        # slow, but it need not freeze the whole application.  The worker keeps
        # the UI responsive; errors still return through the normal dialog/log.
        self._run_single_export(out_dir, formats)

    def _ask_export_conflict(self, out_dir: str, formats: list[str]) -> str | None:
        """How to treat export targets that already exist; ``None`` cancels.

        The writers refuse to clobber anything, so before this the only way to
        re-export into a used folder was to read a FileExistsError out of the
        log and delete the files by hand.
        """
        if self.mode_folder.isChecked():
            # A batch spans many files with different conflicts, so one prompt
            # cannot answer for them all: the row's dropdown is the standing
            # policy instead.
            return str(self.conflict_combo.currentData() or "skip")
        if not self.parsed:
            return "error"
        try:
            existing = existing_export_targets(
                self.parsed, self._msr_state(), out_dir, formats,
                field_selection=self.field_selection,
            )
        except Exception as exc:                                # noqa: BLE001
            self.log(f"[warn] could not check existing outputs: {exc}")
            return "error"
        if not existing:
            return "error"

        listed = "\n".join(f"  {path.name}" for path in existing[:8])
        if len(existing) > 8:
            listed += f"\n  ... and {len(existing) - 8} more"
        message = QMessageBox(self)
        message.setIcon(QMessageBox.Icon.Question)
        message.setWindowTitle("Output already exists")
        message.setText(
            f"{len(existing)} export target(s) already exist in\n{out_dir}")
        message.setInformativeText(
            f"{listed}\n\nOverwrite replaces them. Skip existing "
            "keeps them and writes only what is missing.")
        overwrite = message.addButton("Overwrite",
                                      QMessageBox.ButtonRole.DestructiveRole)
        skip = message.addButton("Skip existing", QMessageBox.ButtonRole.AcceptRole)
        cancel = message.addButton(QMessageBox.StandardButton.Cancel)
        message.setDefaultButton(skip)
        message.setEscapeButton(cancel)
        message.exec()
        clicked = message.clickedButton()
        if clicked is overwrite:
            return "overwrite"
        if clicked is skip:
            return "skip"
        return None

    # ------------------------------------------------------------------
    # Open in viewer (viewer-specific addition)
    # ------------------------------------------------------------------

    def _on_open_in_viewer(self):
        MFSTATE = self._msr_state()
        from ...core.overlay import overlay_color_cycle

        if self._state is None:
            QMessageBox.warning(self, "Not connected", "Not connected to a viewer instance.")
            return
        if self._viewer_open_task is not None:
            self.log("[viewer] datasets from this file are still being opened.")
            return

        # Image series selected in "Datasets / Fields included…" open in the
        # standalone image viewer (they are not MINFLUX datasets).
        selected_images = select_image_series(
            self._gather_image_series_for_dialog(),
            indices=self.image_field_selection,
        )
        image_raws = [int(s["raw_index"]) for s in selected_images]
        opened_images = bool(self._open_obf_series_group(image_raws))
        if opened_images:
            first_name = selected_images[0].get("name", f"stack {image_raws[0]}")
            self.log(
                f"[viewer] opened 1 image viewer for {len(image_raws)} selected "
                f"image series (showing '{first_name}')."
            )

        datasets_info = self.parsed.get("datasets", [])
        if not datasets_info:
            if opened_images == 0:
                QMessageBox.warning(
                    self, "No datasets",
                    "No MINFLUX datasets found. Select image series in "
                    "'Datasets / Fields included…' to open them as images.")
            return
        import_plan = self._viewer_import_plan()
        if not import_plan:
            QMessageBox.information(self, "Open in MINFLUX viewer", "No selected datasets or fields are available to open.")
            return

        msr_path = Path(self.parsed.get("msr", ""))
        imported = []
        imported_indices = []
        errors = []
        render_group_id = f"msr:{msr_path.resolve() if str(msr_path) else 'memory'}:{uuid.uuid4().hex}"
        overlay_cycle = overlay_color_cycle(self._state.prefs)
        overlay_index = 1
        parent = self._owner
        if parent is not None:
            try:
                overlay_index = int(getattr(parent, "_next_overlay_index", 1))
                setattr(parent, "_next_overlay_index", overlay_index + 1)
            except Exception:
                overlay_index = 1
        viewer_transforms: dict[str, dict] = {}
        align_choice = self._ask_viewer_channel_alignment(import_plan)
        if align_choice is _ALIGN_CANCELLED:
            self.log("[viewer] open in viewer aborted by user")
            return
        individual = align_choice is _OPEN_INDIVIDUAL
        confocal_options = getattr(self, "_pending_confocal_options", None)
        if align_choice and not individual:
            alignment_mode, reference_name = align_choice
            try:
                viewer_transforms = self._viewer_channel_alignment_transforms(
                    import_plan, alignment_mode, reference_name,
                )
                self._warn_if_poor_viewer_alignment(alignment_mode, viewer_transforms)
            except Exception as exc:
                self.log(f"[viewer align] failed: {exc}")
                QMessageBox.warning(
                    self,
                    "Open in MINFLUX viewer",
                    f"Channel alignment could not be computed and will be skipped:\n\n{exc}",
                )

        # ------------------------------------------------------------------
        # Everything the build needs is gathered here, on the GUI thread, so
        # the worker below reads nothing from this dialog. The build itself
        # (field copy, normalisation, derived attributes, provenance) is
        # Qt-free and runs off the UI thread -- the same split as opening a
        # .mat / .npy / .zarr from the main window. Only the steps that touch
        # widgets or may ask a question (confocal mapping, add_dataset and
        # the windows it opens) run back here when the build delivers.
        # ------------------------------------------------------------------
        jobs = []
        for ds in datasets_info:
            key = ds.get("display_name") or ds.get("did") or "dataset"
            if key not in import_plan:
                continue
            pbg, used = (self._mbm_naming_metadata(key, ds.get("zroot"))
                         if key in MFSTATE.mbm_map else ({}, []))
            jobs.append({
                "key": key,
                "entry": ds,
                "mfx": MFSTATE.mfx_map.get(key),
                "field_sel": import_plan.get(key),
                "display_name": f"{msr_path.name} | {key}" if msr_path.name else key,
                "mbm_points": MFSTATE.mbm_map.get(key),
                "mbm_points_by_gri": pbg,
                "mbm_used": used,
                "transform": viewer_transforms.get(key),
            })

        prefs = copy.deepcopy(self._state.prefs)
        n_jobs = len(jobs)
        n_datasets_total = len(datasets_info)
        signals = _ViewerOpenSignals()
        # Resolved here so the worker never reaches into this (deletable) dialog.
        build = self._build_viewer_dataset

        def work(report):
            built = []
            n_built = 0
            for i, job in enumerate(jobs):
                key = job["key"]
                logs: list[str] = []
                report(f"Opening dataset '{key}' ({i + 1}/{n_jobs})")
                last_pct = [-1]

                def on_progress(fraction, _i=i, _key=key):
                    overall = (_i + max(0.0, min(1.0, fraction))) / max(n_jobs, 1)
                    pct = int(overall * 100)
                    if pct != last_pct[0]:
                        last_pct[0] = pct
                        signals.progress.emit(overall, _key)
                    report(f"Opening dataset '{_key}'")       # cancellation checkpoint

                on_progress(0.0)
                try:
                    dataset = build(
                        job, msr_path=msr_path, prefs=prefs, logs=logs,
                        progress=on_progress,
                        overlay=None if individual else {
                            "id": render_group_id,
                            "index": overlay_index,
                            "order": n_built + 1,
                            "lut": overlay_cycle[n_built % len(overlay_cycle)],
                        },
                        n_datasets_total=n_datasets_total,
                    )
                except BackgroundTaskCancelled:
                    raise
                except Exception as exc:                     # noqa: BLE001
                    built.append({"key": key, "dataset": None, "logs": logs,
                                  "error": str(exc)})
                    continue
                if dataset is None:
                    built.append({"key": key, "dataset": None, "logs": logs,
                                  "error": None})
                    continue
                n_built += 1
                built.append({"key": key, "dataset": dataset, "logs": logs,
                              "error": None})
            signals.progress.emit(1.0, "")
            return built

        from minflux_viewer.ui.common.background_tasks import BackgroundTask, shared_thread_pool

        source_name = msr_path.name or "MSR data"
        task = BackgroundTask(
            work, description=f"Opening {n_jobs} dataset(s) from {source_name}",
            category="load")
        task.signals = signals
        context = {
            "msr_path": msr_path,
            "individual": individual,
            "viewer_transforms": viewer_transforms,
            "align_choice": align_choice,
            "confocal_options": confocal_options,
            "n_jobs": n_jobs,
            "last_pct": -1,
        }
        signals.progress.connect(self._on_viewer_open_progress)
        signals.done.connect(self._on_viewer_open_built)
        signals.failed.connect(self._on_viewer_open_failed)
        signals.cancelled.connect(self._on_viewer_open_cancelled)
        signals.finished.connect(self._on_viewer_open_finished)
        self._viewer_open_task = task
        self._viewer_open_context = context
        self._set_viewer_open_running(True)
        self.log(f"[viewer] opening {n_jobs} dataset(s) from '{source_name}' "
                 "in the background…")
        shared_thread_pool("msr-open").start(task)

    # -- the per-dataset build (worker thread; Qt-free) --------------------

    @staticmethod
    def _build_viewer_dataset(job, *, msr_path, prefs, logs, progress, overlay, n_datasets_total):
        from ...msr.viewer_import import build_viewer_dataset
        return build_viewer_dataset(job, msr_path=msr_path, prefs=prefs, logs=logs, progress=progress, overlay=overlay, n_datasets_total=n_datasets_total)

    # -- delivery (GUI thread) ----------------------------------------------

    _VIEWER_OPEN_HEADER = "Opening dataset"

    def _set_viewer_open_running(self, running: bool) -> None:
        # The whole reader is inert while its arrays are being read on the
        # worker (a drift correction rewrites them in place). The window can
        # still be moved or closed, which abandons the open.
        for widget in self.findChildren(QPushButton):
            if running:
                widget.setProperty("_open_was_enabled", widget.isEnabled())
                widget.setEnabled(False)
            else:
                was = widget.property("_open_was_enabled")
                widget.setEnabled(True if was is None else bool(was))
        if self._state is None:
            return
        if running:
            self._state.log_progress(f"{self._VIEWER_OPEN_HEADER}…  0%")  # same as below
            self._state.status_progress(self._VIEWER_OPEN_HEADER, 0.0)

    def _on_viewer_open_progress(self, fraction: float, key: str) -> None:
        context = self._viewer_open_context
        if context is None or self._state is None:
            return
        pct = int(round(max(0.0, min(1.0, fraction)) * 100))
        if pct == context.get("last_pct"):
            return
        context["last_pct"] = pct
        if key:
            context["last_key"] = key
        name = key or context.get("last_key") or ""
        header = f"{self._VIEWER_OPEN_HEADER}: '{name}'" if name else self._VIEWER_OPEN_HEADER
        # Same wording in the Log's refreshing line and the status bar.
        self._state.log_progress(f"{header}…  {pct}%")
        self._state.status_progress(header, fraction)

    def _on_viewer_open_failed(self, message: str) -> None:
        if self._state is not None:
            self._state.log_progress(f"{self._VIEWER_OPEN_HEADER}: failed.", final=True)
            self._state.status_message.emit(f"{self._VIEWER_OPEN_HEADER}: failed.")
        self.log(f"[error] open in viewer failed: {message}")
        self._viewer_open_context = None
        QMessageBox.warning(self, "Open in MINFLUX viewer", f"Opening failed:\n\n{message}")

    def _on_viewer_open_cancelled(self) -> None:
        if self._state is not None:
            self._state.log_progress(f"{self._VIEWER_OPEN_HEADER}: cancelled.", final=True)
            self._state.status_message.emit(f"{self._VIEWER_OPEN_HEADER}: cancelled.")
        self._viewer_open_context = None

    def _on_viewer_open_finished(self) -> None:
        self._viewer_open_task = None
        try:
            self._set_viewer_open_running(False)
        except RuntimeError:          # the dialog is already being torn down
            pass

    def _on_viewer_open_built(self, built) -> None:
        """Install the built datasets, on the GUI thread."""
        context = self._viewer_open_context
        self._viewer_open_context = None
        if context is None or self._state is None:
            return
        parent = self._owner
        if parent is not None and getattr(parent, "_is_shutting_down", False):
            return
        msr_path = context["msr_path"]
        individual = context["individual"]
        viewer_transforms = context["viewer_transforms"]
        align_choice = context["align_choice"]
        confocal_options = context["confocal_options"]
        confocal_transform_cache: dict = {}
        imported = []
        imported_indices = []
        errors = []

        self._state.log_progress(f"{self._VIEWER_OPEN_HEADER}: added to the viewer.", final=True)
        previous_suspend_auto_render = getattr(self._state, "suspend_auto_render", False)
        # Overlay mode groups channels into one render view, so suppress the
        # per-dataset auto-render and open one grouped view at the end. Individual
        # mode wants each dataset opened on its own per Preferences > Data, so it
        # leaves the auto-open behaviour untouched.
        self._state.suspend_auto_render = previous_suspend_auto_render if individual else True
        try:
            for item in built:
                key = item["key"]
                for line in item["logs"]:
                    self.log(line)
                if item["error"] is not None:
                    errors.append(f"Dataset '{key}': {item['error']}")
                    self.log(f"[error] {key}: {item['error']}")
                    continue
                dataset = item["dataset"]
                if dataset is None:
                    continue
                try:
                    # Confocal mapping stays on this thread: its manual
                    # alignment is a modal dialog parented to this reader.
                    if confocal_options is not None and confocal_options.choices:
                        from minflux_viewer.ui.analysis.confocal_mapping_dialog import (
                            apply_confocal_mapping_options,
                        )

                        mapping_results = apply_confocal_mapping_options(
                            dataset,
                            msr_path,
                            confocal_options,
                            self,
                            transform_cache=confocal_transform_cache,
                        )
                        for result in mapping_results:
                            self.log(
                                f"[viewer confocal] '{result.attribute_name}' on '{key}': "
                                f"{result.finite_count:,}/{result.total_count:,} localizations in bounds"
                            )
                    # Everything above was written onto `dataset` itself, and
                    # add_dataset appends that same object -- so nothing may be
                    # re-stamped here. A second pass used to repeat the overlay
                    # assignment, the provenance and the MBM points, which
                    # (a) referenced a name this module no longer imports and
                    # (b) overwrote whatever the embedded viewer state or the
                    # recipe sidecar had just restored, with the defaults the
                    # import invents.
                    idx = self._state.add_dataset(dataset)
                    loaded = self._state.datasets[idx]
                    imported_indices.append(idx)
                    imported.append(key)
                    self.log(f"[viewer] '{key}' → {loaded.prop.num_loc:,} loc")
                except Exception as exc:
                    if "Manual confocal alignment was cancelled" in str(exc):
                        self.log("[viewer confocal] manual alignment cancelled; open aborted")
                        errors.append("Open aborted during manual confocal alignment.")
                        break
                    errors.append(f"Dataset '{key}': {exc}")
                    self.log(f"[error] {key}: {exc}")
        finally:
            self._state.suspend_auto_render = previous_suspend_auto_render

        self._state.status_message.emit(
            f"Opened {len(imported)} dataset(s) from {msr_path.name or 'MSR data'}."
            if imported else f"{self._VIEWER_OPEN_HEADER}: nothing opened.")

        # A single imported dataset is standalone, not an overlay channel — drop
        # the provisional overlay assignment as well as the channel color. The
        # grouped path assigns these fields before all imports have succeeded;
        # a one-channel result must remain a standalone dataset.
        if len(imported_indices) == 1:
            sole = self._state.datasets[imported_indices[0]]
            from ...core.overlay import clear_overlay_assignment
            clear_overlay_assignment(sole)

        # Record an overlay-level summary so the Method-text generator can
        # describe the whole multi-channel acquisition (source .msr, channels and
        # their properties, reference, bead selection, alignment matrix) — tagged
        # to the first channel so it auto-checks with the overlay. A single
        # imported dataset is standalone, not an overlay, so skip it.
        if len(imported_indices) >= 2 and not individual:
            self._record_overlay_load_event(
                msr_path, imported, imported_indices, viewer_transforms,
                align_choice if (align_choice and align_choice is not _ALIGN_CANCELLED) else None,
            )

        if errors:
            QMessageBox.warning(
                self, "Import warnings",
                "Some datasets could not be imported:\n\n" + "\n".join(errors),
            )
        if imported:
            self.log(f"[viewer] imported {len(imported)} dataset(s): {', '.join(imported)}")
            parent = self._owner
            # Overlay mode opens one grouped render view here. Individual mode has
            # already opened each dataset's viewers per Preferences > Data (auto-
            # render was not suspended), so don't force a render here.
            if not individual and parent is not None and hasattr(parent, "_show_render") and imported_indices:
                existing = getattr(parent, "_render_windows", {}).get(imported_indices[0])
                if existing is not None and hasattr(existing, "_refresh_from_dataset"):
                    existing._refresh_from_dataset()
                parent._show_render(imported_indices[0])
            # Give the windows the order this reader lists the file's contents
            # in -- mfx, then mbm, then image series -- and do it *after* this
            # dialog is gone, because closing a window hands activation to
            # whatever the window manager picks next. On Windows that was often
            # the image viewer, which is why the render view appeared and then
            # disappeared behind it.
            if parent is not None and hasattr(parent, "restack_import_windows"):
                QTimer.singleShot(
                    RESTACK_DELAY_MS,
                    lambda p=parent, i=list(imported_indices): p.restack_import_windows(i),
                )
            self.close()

    def _record_overlay_load_event(self, msr_path, imported, imported_indices,
                                   viewer_transforms, align_choice):
        """Persist an overlay summary on each channel's metadata and emit one
        parseable, overlay-tagged Log event the Method-text generator can expand
        into a full description of the multi-channel load + alignment."""
        if self._state is None:
            return
        alignment_mode = None
        reference_key = imported[0] if imported else None
        if align_choice:
            alignment_mode, ref_name = align_choice
            if ref_name:
                reference_key = ref_name
        for transform in (viewer_transforms or {}).values():
            rc = transform.get("reference_channel")
            if rc:
                reference_key = rc
                break
        excluded = (sorted(int(g) for g in self._bead_unchecked_gris)
                    if self._bead_unchecked_gris else [])

        for idx in imported_indices:
            try:
                md = self._state.datasets[idx].metadata
            except Exception:
                continue
            md["overlay_channels"] = list(imported)
            md["overlay_reference"] = reference_key
            md["overlay_alignment_mode"] = alignment_mode or "none"
            md["overlay_bead_excluded"] = list(excluded)

        msg = (f"MSR overlay loaded from '{Path(msr_path).name}': {len(imported)} channel(s) "
               f"[{', '.join(imported)}]; reference channel '{reference_key}'; "
               f"channel alignment: {alignment_mode or 'none'}")
        if alignment_mode == "mbm info":
            if excluded:
                msg += f"; beads: all but {len(excluded)} user-excluded ({excluded})"
            else:
                msg += "; beads: all available beads used"
        msg += "."
        try:
            self._state.log(msg, dataset_idx=imported_indices[0])
        except TypeError:       # older AppState.log without dataset_idx
            self._state.log(msg)


# ---------------------------------------------------------------------------
# Availability helpers
# ---------------------------------------------------------------------------

def msr_available() -> bool:
    try:
        import msr_reader  # noqa: F401
        return True
    except Exception:
        return False


def msr_unavailable_message() -> str:
    return (
        "The <b>msr-reader</b> library is not installed.<br><br>"
        "Opening <tt>.msr</tt> files requires it. It is a pure-Python, "
        "cross-platform dependency normally installed automatically:<br><br>"
        "<tt>poetry install</tt>  (or)  <tt>pip install msr-reader</tt>"
    )


def register_msr_dialog(owner, dlg) -> None:
    """Retain *dlg* on *owner* so it is not garbage-collected (an MSR reader is an
    unowned, ``WA_DeleteOnClose`` top-level window that needs a live Python
    reference). **Multiple** dialogs are supported — they are kept in a list and
    dropped when destroyed — so opening the plugin again, or dropping several
    ``.msr`` files at once, spawns independent readers instead of replacing the
    previous one. ``owner._plugin_msr_reader_dialog`` is kept pointing at the most
    recent one for back-compat."""
    if owner is None:
        return
    dialogs = getattr(owner, "_plugin_msr_reader_dialogs", None)
    if dialogs is None:
        dialogs = []
        owner._plugin_msr_reader_dialogs = dialogs
    dialogs.append(dlg)
    owner._plugin_msr_reader_dialog = dlg          # back-compat single-attr alias

    def _forget(_obj=None, d=dlg, lst=dialogs):
        try:
            lst.remove(d)
        except ValueError:
            pass
    dlg.destroyed.connect(_forget)


def open_msr(
    msr_path: str,
    state=None,
    parent: QWidget | None = None,
) -> None:
    """Open a new MSR reader dialog for *msr_path* (auto-parses a single file)."""
    if not msr_available():
        QMessageBox.warning(parent, "MSR reader not available", msr_unavailable_message())
        return
    dlg = MsrReaderDialog(state=state, msr_path=msr_path, parent=parent)
    register_msr_dialog(parent, dlg)
    dlg.show()
    dlg.raise_()
    dlg.activateWindow()


# ---------------------------------------------------------------------------
# Standalone entry point
# ---------------------------------------------------------------------------

def main():
    app = QApplication.instance() or QApplication([])
    window = MsrReaderDialog()
    window.show()
    app.exec()


if __name__ == "__main__":
    main()
