"""
minflux_viewer.ui.preferences_dialog
=====================================
Preferences dialog — Edit → Preferences (always the last item).

The dialog uses a left category browser with search and a stacked content
panel on the right. This is easier to extend than a growing row of tabs while
keeping the existing preference storage unchanged.

Changes are applied to ``AppState.prefs`` in memory when the user presses
**OK**; ``AppState.save_prefs()`` persists the result to ``QSettings``
just like any other pref write. The dialog never touches the filesystem
until OK.

Buttons
-------
* **Reset**   — restore defaults for the currently visible tab only
* **Cancel**  — discard edits
* **OK**      — apply edits, save, close
"""

from __future__ import annotations

import copy
from pathlib import Path

from PyQt6.QtCore import Qt
from PyQt6.QtGui import QKeySequence, QShortcut
from PyQt6.QtWidgets import (
    QCheckBox,
    QComboBox,
    QDialog,
    QDoubleSpinBox,
    QFileDialog,
    QFormLayout,
    QFrame,
    QGridLayout,
    QGroupBox,
    QHBoxLayout,
    QLabel,
    QLineEdit,
    QListWidget,
    QListWidgetItem,
    QPushButton,
    QKeySequenceEdit,
    QScrollArea,
    QSpinBox,
    QStackedWidget,
    QTableWidget,
    QHeaderView,
    QVBoxLayout,
    QWidget,
)

from ..core.app_state import AppState, DEFAULT_PREFS


# ---------------------------------------------------------------------------
# Static option lists — match the Plot tab screenshot
# ---------------------------------------------------------------------------

_ITER_OPTIONS = ["last", "all"]

_RENDER_CMAPS = [
    "Hot", "Viridis", "Inferno", "Magma", "Plasma", "Cividis",
    "Turbo", "Gray",
]

_ATTR_CMAPS = [
    "single color", "Viridis", "Inferno", "Magma", "Plasma",
    "Hot", "Gray",
]

_SCATTER_CMAPS = ["glasbey", "jet", "HiLo", "parula", "turbo", "hot"]

_SCATTER_COLOR_BY_OPTIONS = ["tid", "efo", "cfr", "dcr", "den", "vld", "itr", "eco", "ecc"]

_ROI_COLORS = [
    "Yellow", "Red", "Green", "Cyan", "Magenta", "White", "Black",
]

_MBM_TRANSFORM_TYPES = [
    "rigid XY + translational Z",
    "translational XYZ",
    "rigid XYZ",
]

_ATTRIBUTE_ORDER = [
    "vld", "itr", "tid", "loc",
    "efo", "cfr", "dcr", "tim",
    "sta", "fnl", "bot", "eot",
    "gri", "thi", "sqi", "lnc",
    "eco", "ecc", "efc", "fbg",
]

_COMPUTED_ATTRIBUTE_INFO = [
    ("idx", "index of localization data points"),
    ("siz", "the number of localizations within a track"),
    ("dst", "the distance between two adjacent localizations"),
    ("dur", "the time between the first and last points of a track"),
    ("len", "full sum of distance traversed, along the full length of the track"),
    ("spd", "the distance between two points (in meters) divided by time between the two points (in seconds)"),
    ("dt", "time interval from the previous localization in the same track"),
    ("tim_trace", "time stamp zeroed at each track start"),
    ("den", "local density at data point"),
]

_SHORTCUT_LABELS = {
    "focus_main_window": "Focus main window",
    "close_window": "Close current window",
    "show_info": "Show info",
    "dataset_manager": "Dataset Manager",
    "duplicate": "Duplicate",
    "filter": "Filter",
    "next_window": "Next window",
    "previous_window": "Previous window",
    "next_dataset": "Next dataset",
    "previous_dataset": "Previous dataset",
    "open": "Open",
    "open_spreadsheet": "Open spreadsheet data",
    "open_tiff": "Open .tif file",
    "save": "Save processed data",
    "render": "Render",
    "brightness_contrast": "Brightness / Contrast",
    "attribute_plot": "Attribute Plot",
    "attribute_histogram": "Attribute Histogram",
    "scatter_plot": "Loc Scatter Plot",
    "log": "Log",
    "console": "Console",
    "preferences": "Preferences",
}


class _NoWheelComboBox(QComboBox):
    """Combo box that ignores mouse-wheel scrolling.

    Prevents the wheel from silently changing the selection while the user is
    scrolling a list or page (e.g. the Shortcuts command column).
    """

    def wheelEvent(self, event) -> None:  # noqa: N802 - Qt API
        event.ignore()


class PreferencesDialog(QDialog):
    """Category-based editor for :data:`AppState.prefs`."""

    def __init__(self, state: AppState, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self._state = state
        # Work on a deep copy so Cancel really cancels
        self._draft: dict = copy.deepcopy(state.prefs)
        self._page_keys: list[list[str]] = []

        self.setWindowTitle("Preferences")
        self.setModal(True)
        # Tall enough that the longest page (Shortcuts) fits without resizing.
        self.resize(820, 760)
        self.setMinimumSize(720, 560)

        self._build_ui()
        self._load_draft_into_widgets()

    # ------------------------------------------------------------------
    # UI
    # ------------------------------------------------------------------

    def _build_ui(self) -> None:
        root = QVBoxLayout(self)
        root.setContentsMargins(10, 10, 10, 10)
        root.setSpacing(10)

        content = QHBoxLayout()
        content.setSpacing(10)

        left = QVBoxLayout()
        left.setSpacing(0)
        self._search_edit = QLineEdit()
        self._search_edit.setPlaceholderText("Search (Ctrl + F)")
        self._search_edit.setClearButtonEnabled(True)
        self._search_edit.textChanged.connect(self._filter_pages)
        left.addWidget(self._search_edit)

        self._page_list = QListWidget()
        self._page_list.setFixedWidth(168)
        self._page_list.setFrameShape(QFrame.Shape.Box)
        self._page_list.setHorizontalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAlwaysOff)
        self._page_list.setStyleSheet(
            "QListWidget { background: white; border: 1px solid #9aa4b2; }"
            "QListWidget::item { padding: 4px 8px; }"
            "QListWidget::item:selected { background: #e7e7e7; color: black; }"
        )
        self._page_list.currentRowChanged.connect(self._stack_page_changed)
        left.addWidget(self._page_list, stretch=1)
        content.addLayout(left)

        self._stack_frame = QFrame()
        self._stack_frame.setFrameShape(QFrame.Shape.Box)
        self._stack_frame.setStyleSheet(
            "QFrame { background: white; border: 1px solid #9aa4b2; }"
            "QFrame > QWidget { border: none; }"
            "QGroupBox { margin-top: 10px; padding: 10px 8px 8px 8px; }"
            "QGroupBox::title { subcontrol-origin: margin; left: 8px; padding: 0 3px; }"
        )
        stack_layout = QVBoxLayout(self._stack_frame)
        stack_layout.setContentsMargins(12, 12, 12, 12)
        self._stack = QStackedWidget()
        stack_layout.addWidget(self._stack)
        content.addWidget(self._stack_frame, stretch=1)

        self._add_page("File", self._build_file_tab(), ["file"])
        self._add_page("Data", self._build_data_tab(), ["data"])
        self._add_page("Appearance", self._build_plot_tab(), ["plot"])
        self._add_page("Plugin", self._build_plugin_tab(), ["plugin", "file"])
        self._add_page("Shortcuts", self._build_shortcuts_tab(), ["shortcuts"])
        self._add_page("Attributes", self._build_attributes_tab(), ["attributes"])
        self._add_page("MBM Handling", self._build_mbm_tab(), ["mbm_handling"])
        self._page_list.setCurrentRow(0)

        QShortcut(QKeySequence("Ctrl+F"), self, activated=self._search_edit.setFocus)
        root.addLayout(content, stretch=1)

        # ── Buttons ─────────────────────────────────────────────────
        btns = QHBoxLayout()
        reset_btn = QPushButton("Reset")
        reset_btn.setToolTip("Restore defaults for the currently visible tab.")
        reset_btn.clicked.connect(self._reset_current_tab)
        btns.addWidget(reset_btn)

        btns.addStretch()

        cancel_btn = QPushButton("Cancel")
        cancel_btn.clicked.connect(self.reject)
        btns.addWidget(cancel_btn)

        ok_btn = QPushButton("OK")
        ok_btn.setDefault(True)
        ok_btn.clicked.connect(self._accept)
        btns.addWidget(ok_btn)
        root.addLayout(btns)

    def _add_page(self, name: str, widget: QWidget, keys: list[str]) -> None:
        item = QListWidgetItem(name)
        item.setData(Qt.ItemDataRole.UserRole, self._stack.count())
        item.setData(Qt.ItemDataRole.UserRole + 1, name.lower())
        self._page_list.addItem(item)
        scroll = QScrollArea()
        scroll.setWidgetResizable(True)
        scroll.setFrameShape(QFrame.Shape.NoFrame)
        scroll.setWidget(widget)
        self._stack.addWidget(scroll)
        self._page_keys.append(keys)

    def _stack_page_changed(self, row: int) -> None:
        item = self._page_list.item(row)
        if item is None or item.isHidden():
            return
        idx = int(item.data(Qt.ItemDataRole.UserRole))
        self._stack.setCurrentIndex(idx)

    def _filter_pages(self, text: str) -> None:
        query = text.strip().lower()
        first_visible = -1
        for row in range(self._page_list.count()):
            item = self._page_list.item(row)
            visible = query in str(item.data(Qt.ItemDataRole.UserRole + 1))
            item.setHidden(not visible)
            if visible and first_visible < 0:
                first_visible = row
        current = self._page_list.currentItem()
        if current is None or current.isHidden():
            self._page_list.setCurrentRow(first_visible)

    # -- File tab ----------------------------------------------------

    def _build_file_tab(self) -> QWidget:
        w = QWidget()
        form = QFormLayout(w)
        form.setContentsMargins(20, 30, 20, 20)
        form.setHorizontalSpacing(14)
        form.setVerticalSpacing(10)

        self._files_in_history = QSpinBox()
        self._files_in_history.setRange(0, 100)
        self._files_in_history.setMinimumWidth(70)
        form.addRow("Files in history", self._files_in_history)

        self._keep_last_folder = QCheckBox("current data folder as the default folder")
        form.addRow("", self._keep_last_folder)

        self._confirm_overwrite = QCheckBox("confirm before overwriting an existing file")
        form.addRow("", self._confirm_overwrite)

        self._close_paraview = QCheckBox("close ParaView when exiting the application")
        form.addRow("", self._close_paraview)

        self._temp_folder_edit, temp_row = self._browse_row(
            self._browse_temp_folder, file_mode=False,
        )
        self._temp_folder_edit.setPlaceholderText("(empty = system temp folder)")
        self._temp_folder_edit.setToolTip(
            "App-wide folder for any temporary files the application may write "
            "(empty = system temp). This is the single place to set a temp folder."
        )
        form.addRow("Temporary folder", temp_row)

        return w

    # -- Data tab ----------------------------------------------------

    def _build_data_tab(self) -> QWidget:
        w = QWidget()
        root = QVBoxLayout(w)
        root.setContentsMargins(16, 12, 16, 12)
        root.setSpacing(6)

        # "When opening data file:" section
        root.addWidget(self._section_label("When opening data file:"))

        load_row = QHBoxLayout()
        load_row.addWidget(QLabel("Load"))
        self._iter_load = QComboBox()
        self._iter_load.addItems(_ITER_OPTIONS)
        self._iter_load.setMinimumWidth(190)
        load_row.addWidget(self._iter_load)
        load_row.addWidget(QLabel("iteration"))
        load_row.addStretch()
        root.addLayout(load_row)

        self._only_valid = QCheckBox("only valid localizations (vld check)")
        root.addLayout(self._indent(self._only_valid))

        self._load_efc_cfr = QCheckBox("effective CFR and EFC iteration")
        self._load_all_dcr = QCheckBox("all DCR iteration for channel separation")
        # The screenshot shows the two EFC/DCR checkboxes indented under Load
        root.addLayout(self._indent(self._load_efc_cfr))
        root.addLayout(self._indent(self._load_all_dcr))

        # NEW 2D/3D threshold row
        z_row = QHBoxLayout()
        z_row.addSpacing(18)
        self._enforce_z = QCheckBox("min Z range to be recognized as 3D data")
        z_row.addWidget(self._enforce_z)
        self._min_z_spin = QDoubleSpinBox()
        self._min_z_spin.setRange(0.0, 10000.0)
        self._min_z_spin.setDecimals(2)
        self._min_z_spin.setMinimumWidth(88)
        z_row.addWidget(self._min_z_spin)
        z_row.addWidget(QLabel("nm"))
        z_row.addStretch()
        root.addLayout(z_row)
        # Enable the spin box only when the checkbox is on
        self._enforce_z.toggled.connect(self._min_z_spin.setEnabled)

        root.addSpacing(6)

        # Compute section
        root.addWidget(self._section_label("Compute:"))

        rimf_row = QHBoxLayout()
        rimf_row.addSpacing(18)
        self._compute_rimf = QCheckBox("RIMF (refractive index mismatch factor)")
        self._compute_rimf.setToolTip(
            "Estimate the RIMF z-scaling factor on load (from the raw last-valid z "
            "via anisotropy estimation), or tick 'use fixed value' to apply a set value."
        )
        rimf_row.addWidget(self._compute_rimf)
        self._use_fixed_rimf = QCheckBox("use fixed value")
        rimf_row.addWidget(self._use_fixed_rimf)
        self._rimf_spin = QDoubleSpinBox()
        self._rimf_spin.setRange(0.0, 10.0)
        self._rimf_spin.setDecimals(4)
        self._rimf_spin.setSingleStep(0.01)
        self._rimf_spin.setMaximumWidth(90)
        rimf_row.addWidget(self._rimf_spin)
        rimf_row.addStretch()
        root.addLayout(rimf_row)

        self._compute_rimf.toggled.connect(self._on_compute_rimf_toggled)
        self._use_fixed_rimf.toggled.connect(self._on_use_fixed_rimf_toggled)

        self._compute_loc_prec = QCheckBox("Localization Precision")
        root.addLayout(self._indent(self._compute_loc_prec))

        density_row = QHBoxLayout()
        density_row.addSpacing(18)
        self._compute_density = QCheckBox("local density within radius")
        density_row.addWidget(self._compute_density)
        self._density_radius = QDoubleSpinBox()
        self._density_radius.setRange(0.0, 100000.0)
        self._density_radius.setDecimals(0)
        self._density_radius.setMinimumWidth(90)
        density_row.addWidget(self._density_radius)
        density_row.addWidget(QLabel("nm"))
        density_row.addStretch()
        root.addLayout(density_row)

        density_method_row = QHBoxLayout()
        density_method_row.addSpacing(36)
        density_method_row.addWidget(QLabel("method"))
        self._density_method = QComboBox()
        self._density_method.addItem("KD-tree range search", "kdtree")
        self._density_method.addItem("2D histogram", "histogram_2d")
        self._density_method.setMinimumWidth(170)
        density_method_row.addWidget(self._density_method)
        density_method_row.addStretch()
        root.addLayout(density_method_row)

        # Histogram-method parameters on their own row so nothing is clipped.
        density_hist_row = QHBoxLayout()
        density_hist_row.addSpacing(36)
        density_hist_row.addWidget(QLabel("histogram voxel"))
        self._density_voxel = QDoubleSpinBox()
        self._density_voxel.setRange(0.1, 100000.0)
        self._density_voxel.setDecimals(1)
        self._density_voxel.setMinimumWidth(90)
        density_hist_row.addWidget(self._density_voxel)
        density_hist_row.addWidget(QLabel("nm"))
        density_hist_row.addSpacing(18)
        density_hist_row.addWidget(QLabel("smoothing sigma"))
        self._density_sigma = QDoubleSpinBox()
        self._density_sigma.setRange(0.0, 100.0)
        self._density_sigma.setDecimals(2)
        self._density_sigma.setMinimumWidth(88)
        density_hist_row.addWidget(self._density_sigma)
        density_hist_row.addStretch()
        root.addLayout(density_hist_row)

        root.addSpacing(6)

        # Show section (2×2 grid)
        root.addWidget(self._section_label("Show:"))
        show_row1 = QHBoxLayout()
        show_row1.addSpacing(18)
        self._show_data_info = QCheckBox("data info... (by Data Selector)")
        show_row1.addWidget(self._show_data_info); show_row1.addStretch()
        root.addLayout(show_row1)

        show_row2 = QHBoxLayout()
        show_row2.addSpacing(18)
        self._show_attr = QCheckBox("attribute plot")
        self._show_scatter = QCheckBox("scatter plot")
        show_row2.addWidget(self._show_attr)
        show_row2.addSpacing(40)
        show_row2.addWidget(self._show_scatter)
        show_row2.addStretch()
        root.addLayout(show_row2)

        show_row3 = QHBoxLayout()
        show_row3.addSpacing(18)
        self._show_hist = QCheckBox("histogram plot")
        self._show_render = QCheckBox("render view")
        show_row3.addWidget(self._show_hist)
        show_row3.addSpacing(40)
        show_row3.addWidget(self._show_render)
        show_row3.addStretch()
        root.addLayout(show_row3)

        root.addStretch()
        return w

    # -- Plot tab ----------------------------------------------------

    def _build_plot_tab(self) -> QWidget:
        w = QWidget()
        root = QVBoxLayout(w)
        root.setContentsMargins(12, 12, 12, 12)
        root.setSpacing(8)

        # ── Render View ──────────────────────────────────────────────
        grp_render = QGroupBox("Render View")
        form_render = QFormLayout(grp_render)
        form_render.setHorizontalSpacing(12)
        form_render.setVerticalSpacing(6)

        pixel_row = QHBoxLayout()
        self._px_spin = QDoubleSpinBox()
        self._px_spin.setRange(0.1, 10000.0)
        self._px_spin.setDecimals(2)
        self._px_spin.setSingleStep(0.5)
        self._px_spin.setMaximumWidth(120)
        pixel_row.addWidget(self._px_spin)
        pixel_row.addWidget(QLabel("nm"))
        pixel_row.addStretch()
        form_render.addRow("Pixel size", pixel_row)

        self._render_cmap_combo = QComboBox()
        self._render_cmap_combo.addItems(_RENDER_CMAPS)
        form_render.addRow("Colormap", self._render_cmap_combo)

        root.addWidget(grp_render)

        # ── Scatter Plot ─────────────────────────────────────────────
        grp_scatter = QGroupBox("Scatter Plot")
        form_scatter = QFormLayout(grp_scatter)
        form_scatter.setHorizontalSpacing(12)
        form_scatter.setVerticalSpacing(6)

        self._scatter_color_by_combo = QComboBox()
        self._scatter_color_by_combo.addItems(_SCATTER_COLOR_BY_OPTIONS)
        form_scatter.addRow("Color by", self._scatter_color_by_combo)

        self._scatter_cmap_combo = QComboBox()
        self._scatter_cmap_combo.addItems(_SCATTER_CMAPS)
        form_scatter.addRow("Colormap", self._scatter_cmap_combo)

        root.addWidget(grp_scatter)

        # ── Attribute Plot ───────────────────────────────────────────
        grp_attr = QGroupBox("Attribute Plot")
        form_attr = QFormLayout(grp_attr)
        form_attr.setHorizontalSpacing(12)
        form_attr.setVerticalSpacing(6)

        self._plot_cmap_combo = QComboBox()
        self._plot_cmap_combo.addItems(_ATTR_CMAPS)
        form_attr.addRow("Colormap", self._plot_cmap_combo)

        root.addWidget(grp_attr)

        # ── Histogram Plot ───────────────────────────────────────────
        grp_hist = QGroupBox("Histogram Plot")
        hist_layout = QVBoxLayout(grp_hist)
        hist_layout.setSpacing(4)

        hist_values = QHBoxLayout()
        self._hist_trace_mean = QCheckBox("trace mean")
        self._hist_trace_median = QCheckBox("trace median")
        self._hist_trace_min = QCheckBox("trace min")
        self._hist_trace_max = QCheckBox("trace max")
        self._hist_trace_stdev = QCheckBox("trace stdev")
        self._hist_trace_range = QCheckBox("trace range")
        for cb in (
            self._hist_trace_mean, self._hist_trace_median, self._hist_trace_min,
            self._hist_trace_max, self._hist_trace_stdev, self._hist_trace_range,
        ):
            hist_values.addWidget(cb)
        hist_values.addStretch()
        hist_layout.addLayout(hist_values)

        root.addWidget(grp_hist)

        # ── Filter ───────────────────────────────────────────────────
        grp_filter = QGroupBox("Filter")
        form_filter = QFormLayout(grp_filter)
        form_filter.setHorizontalSpacing(12)
        form_filter.setVerticalSpacing(6)

        range_row = QHBoxLayout()
        self._filter_range_color_combo = QComboBox()
        self._filter_range_color_combo.addItems(_ROI_COLORS)
        range_row.addWidget(self._filter_range_color_combo)
        range_row.addWidget(QLabel("transparency"))
        self._filter_range_alpha = QSpinBox()
        self._filter_range_alpha.setRange(0, 100)
        self._filter_range_alpha.setToolTip("Filter fill opacity as percentage (0 = transparent, 100 = opaque).")
        self._filter_range_alpha.setMinimumWidth(74)
        self._filter_range_alpha.setSuffix(" %")
        range_row.addWidget(self._filter_range_alpha)
        range_row.addStretch()
        form_filter.addRow("Range color", range_row)

        bounds_row = QHBoxLayout()
        self._filter_bounds_color_combo = QComboBox()
        self._filter_bounds_color_combo.addItems(_ROI_COLORS)
        bounds_row.addWidget(self._filter_bounds_color_combo)
        bounds_row.addWidget(QLabel("size"))
        self._filter_bounds_size = QSpinBox()
        self._filter_bounds_size.setRange(1, 10)
        self._filter_bounds_size.setMinimumWidth(62)
        bounds_row.addWidget(self._filter_bounds_size)
        bounds_row.addWidget(QLabel("px"))
        bounds_row.addStretch()
        form_filter.addRow("Bounds color", bounds_row)

        root.addWidget(grp_filter)

        # ── ROI ──────────────────────────────────────────────────────
        grp_roi = QGroupBox("ROI")
        form_roi = QFormLayout(grp_roi)
        form_roi.setHorizontalSpacing(12)
        form_roi.setVerticalSpacing(6)

        roi_color_row = QHBoxLayout()
        self._roi_color_combo = QComboBox()
        self._roi_color_combo.addItems(_ROI_COLORS)
        roi_color_row.addWidget(self._roi_color_combo)
        roi_color_row.addWidget(QLabel("transparency"))
        self._roi_transparency = QSpinBox()
        self._roi_transparency.setRange(0, 100)
        self._roi_transparency.setMinimumWidth(74)
        self._roi_transparency.setSuffix(" %")
        roi_color_row.addWidget(self._roi_transparency)
        roi_color_row.addStretch()
        form_roi.addRow("Color", roi_color_row)

        edge_row = QHBoxLayout()
        self._roi_edge_size = QSpinBox()
        self._roi_edge_size.setRange(1, 10)
        self._roi_edge_size.setMinimumWidth(62)
        edge_row.addWidget(self._roi_edge_size)
        edge_row.addWidget(QLabel("px"))
        edge_row.addStretch()
        form_roi.addRow("Edge size", edge_row)

        widget_row = QHBoxLayout()
        self._roi_edit_widget_size = QSpinBox()
        self._roi_edit_widget_size.setRange(4, 32)
        self._roi_edit_widget_size.setMinimumWidth(62)
        widget_row.addWidget(self._roi_edit_widget_size)
        widget_row.addWidget(QLabel("px"))
        widget_row.addStretch()
        form_roi.addRow("Edit widget size", widget_row)

        root.addWidget(grp_roi)

        root.addStretch()
        return w

    # -- Plugin tab --------------------------------------------------

    def _build_plugin_tab(self) -> QWidget:
        w = QWidget()
        root = QVBoxLayout(w)
        root.setContentsMargins(16, 14, 16, 14)
        root.setSpacing(8)

        root.addWidget(self._section_label("ParaView:"))
        self._paraview_edit, pv_row = self._browse_row(
            self._browse_paraview_exe, file_mode=True,
        )
        root.addLayout(self._form_row("Executable path", pv_row))

        root.addSpacing(8)
        root.addWidget(self._section_label("MSR Reader plugin:"))

        self._msr_remember = QCheckBox(
            "remember last used folders (export and file-open)"
        )
        msr_row = QHBoxLayout()
        msr_row.addSpacing(12)
        msr_row.addWidget(self._msr_remember)
        msr_row.addStretch()
        root.addLayout(msr_row)

        root.addStretch()
        return w

    def _build_shortcuts_tab(self) -> QWidget:
        w = QWidget()
        root = QVBoxLayout(w)
        root.setContentsMargins(16, 14, 16, 14)
        root.setSpacing(8)

        self._shortcut_table = QTableWidget(0, 2)
        self._shortcut_table.setHorizontalHeaderLabels(["Command", "Shortcut"])
        self._shortcut_table.verticalHeader().setVisible(False)
        header = self._shortcut_table.horizontalHeader()
        header.setSectionResizeMode(0, QHeaderView.ResizeMode.Stretch)
        header.setSectionResizeMode(1, QHeaderView.ResizeMode.ResizeToContents)
        root.addWidget(self._shortcut_table, stretch=1)

        buttons = QHBoxLayout()
        add_btn = QPushButton("Add")
        add_btn.clicked.connect(self._add_shortcut_row)
        buttons.addWidget(add_btn)
        delete_btn = QPushButton("Delete selected")
        delete_btn.clicked.connect(self._delete_selected_shortcut_rows)
        buttons.addWidget(delete_btn)
        buttons.addStretch()
        root.addLayout(buttons)
        return w

    def _add_shortcut_row(self, command: str | None = None, sequence: str = "") -> None:
        row = self._shortcut_table.rowCount()
        self._shortcut_table.insertRow(row)

        command_combo = _NoWheelComboBox()
        for key, label in _SHORTCUT_LABELS.items():
            command_combo.addItem(label, key)
        if command:
            idx = command_combo.findData(command)
            if idx >= 0:
                command_combo.setCurrentIndex(idx)
        self._shortcut_table.setCellWidget(row, 0, command_combo)

        key_edit = QKeySequenceEdit()
        key_edit.setKeySequence(QKeySequence(sequence))
        key_edit.setMaximumWidth(220)
        self._shortcut_table.setCellWidget(row, 1, key_edit)

    def _delete_selected_shortcut_rows(self) -> None:
        selected = sorted({idx.row() for idx in self._shortcut_table.selectedIndexes()}, reverse=True)
        if not selected and self._shortcut_table.currentRow() >= 0:
            selected = [self._shortcut_table.currentRow()]
        for row in selected:
            self._shortcut_table.removeRow(row)

    def _build_attributes_tab(self) -> QWidget:
        outer = QWidget()
        root = QVBoxLayout(outer)
        root.setContentsMargins(12, 12, 12, 12)
        root.setSpacing(10)
        self._attribute_checks: dict[str, QCheckBox] = {}
        self._computed_attribute_checks: dict[str, QCheckBox] = {}

        root.addWidget(self._section_label("Load raw MINFLUX attributes:"))

        scroll = QScrollArea()
        scroll.setWidgetResizable(True)
        panel = QWidget()
        grid = QGridLayout(panel)
        grid.setHorizontalSpacing(18)
        grid.setVerticalSpacing(6)
        cols = 4
        for i, name in enumerate(_ATTRIBUTE_ORDER):
            cb = QCheckBox(name)
            self._attribute_checks[name] = cb
            grid.addWidget(cb, i // cols, i % cols)
        scroll.setWidget(panel)
        root.addWidget(scroll)

        root.addWidget(self._section_label("Compute additional attributes:"))
        computed_panel = QWidget()
        computed_layout = QVBoxLayout(computed_panel)
        computed_layout.setContentsMargins(4, 0, 4, 0)
        computed_layout.setSpacing(4)
        for name, description in _COMPUTED_ATTRIBUTE_INFO:
            cb = QCheckBox(f"{name}: {description}")
            self._computed_attribute_checks[name] = cb
            computed_layout.addWidget(cb)
        root.addWidget(computed_panel)
        return outer

    def _build_mbm_tab(self) -> QWidget:
        w = QWidget()
        root = QVBoxLayout(w)
        root.setContentsMargins(16, 14, 16, 14)
        root.setSpacing(10)

        self._mbm_only_used = QCheckBox("only take beads that used by minflux measurement for drift correction")
        root.addWidget(self._mbm_only_used)

        min_row = QHBoxLayout()
        min_row.addWidget(QLabel("minimum"))
        self._mbm_min_locs = QSpinBox()
        self._mbm_min_locs.setRange(1, 1_000_000)
        self._mbm_min_locs.setValue(10)
        min_row.addWidget(self._mbm_min_locs)
        min_row.addWidget(QLabel("localization as a valid bead"))
        min_row.addStretch()
        root.addLayout(min_row)

        avg_row = QHBoxLayout()
        avg_row.addWidget(QLabel("use"))
        self._mbm_average_method = QComboBox()
        self._mbm_average_method.addItems(["mean", "median"])
        avg_row.addWidget(self._mbm_average_method)
        avg_row.addWidget(QLabel("value of the first"))
        self._mbm_average_count = QSpinBox()
        self._mbm_average_count.setRange(1, 1_000_000)
        self._mbm_average_count.setValue(10)
        avg_row.addWidget(self._mbm_average_count)
        avg_row.addWidget(QLabel("occurrence as average beads coordinates for a minflux measurement"))
        avg_row.addStretch()
        root.addLayout(avg_row)

        transform_row = QHBoxLayout()
        transform_row.addWidget(QLabel("expect"))
        self._mbm_transform_type = QComboBox()
        self._mbm_transform_type.addItems(_MBM_TRANSFORM_TYPES)
        transform_row.addWidget(self._mbm_transform_type)
        transform_row.addWidget(QLabel("transform for channel alignment"))
        transform_row.addStretch()
        root.addLayout(transform_row)

        align_row = QHBoxLayout()
        align_row.addWidget(QLabel("align to"))
        self._mbm_align_to = QComboBox()
        self._mbm_align_to.addItems(["first", "last"])
        align_row.addWidget(self._mbm_align_to)
        align_row.addWidget(QLabel("channel (dataset)"))
        align_row.addStretch()
        root.addLayout(align_row)

        root.addStretch()
        return w

    # ------------------------------------------------------------------
    # Layout helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _section_label(text: str) -> QLabel:
        lbl = QLabel(f"<b>{text}</b>")
        return lbl

    @staticmethod
    def _indent(widget: QWidget, px: int = 18) -> QHBoxLayout:
        row = QHBoxLayout()
        row.addSpacing(px)
        row.addWidget(widget)
        row.addStretch()
        return row

    @staticmethod
    def _form_row(label_text: str, inner: QHBoxLayout) -> QHBoxLayout:
        row = QHBoxLayout()
        lbl = QLabel(label_text)
        lbl.setMinimumWidth(110)
        row.addWidget(lbl)
        row.addLayout(inner, stretch=1)
        return row

    def _browse_row(
        self, browse_cb, *, file_mode: bool,
    ) -> tuple[QLineEdit, QHBoxLayout]:
        row = QHBoxLayout()
        edit = QLineEdit()
        row.addWidget(edit, stretch=1)
        btn = QPushButton("Browse…")
        btn.clicked.connect(browse_cb)
        row.addWidget(btn)
        return edit, row

    def _browse_paraview_exe(self) -> None:
        start = self._paraview_edit.text().strip() or str(Path.home())
        path, _ = QFileDialog.getOpenFileName(
            self, "Locate ParaView executable", start,
            "ParaView executable (paraview paraview.exe);;All files (*)",
        )
        if path:
            self._paraview_edit.setText(path)

    def _browse_temp_folder(self) -> None:
        start = self._temp_folder_edit.text().strip() or str(Path.home())
        path = QFileDialog.getExistingDirectory(
            self, "Choose temporary folder", start,
        )
        if path:
            self._temp_folder_edit.setText(path)

    def _on_compute_rimf_toggled(self, checked: bool) -> None:
        if checked:
            self._use_fixed_rimf.blockSignals(True)
            self._use_fixed_rimf.setChecked(False)
            self._use_fixed_rimf.blockSignals(False)
            self._rimf_spin.setEnabled(False)

    def _on_use_fixed_rimf_toggled(self, checked: bool) -> None:
        if checked:
            self._compute_rimf.blockSignals(True)
            self._compute_rimf.setChecked(False)
            self._compute_rimf.blockSignals(False)
        self._rimf_spin.setEnabled(checked)

    # ------------------------------------------------------------------
    # Data flow — draft ↔ widgets ↔ prefs
    # ------------------------------------------------------------------

    def _load_draft_into_widgets(self) -> None:
        f = self._draft["file"]
        d = self._draft["data"]
        p = self._draft["plot"]
        g = self._draft.get("plugin", {})
        s = self._draft.get("shortcuts", {})
        a = self._draft.get("attributes", {})
        m = self._draft.get("mbm_handling", {})

        # File
        self._files_in_history.setValue(int(f.get("num_file_history", 5)))
        self._keep_last_folder.setChecked(bool(f.get("keep_last_folder", True)))
        self._confirm_overwrite.setChecked(bool(f.get("confirm_overwrite", True)))
        self._close_paraview.setChecked(bool(f.get("close_paraview_on_exit", True)))

        # Data
        self._iter_load.setCurrentText(str(d.get("iter_load", "last")))
        self._only_valid.setChecked(bool(d.get("only_valid_locs", True)))
        self._load_efc_cfr.setChecked(bool(d.get("load_efc_cfr", True)))
        self._load_all_dcr.setChecked(bool(d.get("load_all_dcr", False)))
        self._enforce_z.setChecked(bool(d.get("enforce_min_z_range", True)))
        self._min_z_spin.setValue(float(d.get("min_z_range_nm", 5.0)))
        self._min_z_spin.setEnabled(self._enforce_z.isChecked())
        compute_rimf = bool(d.get("compute_rimf", False))
        use_fixed = bool(p.get("use_fixed_rimf", False))
        self._compute_rimf.setChecked(compute_rimf)
        self._use_fixed_rimf.setChecked(use_fixed)
        self._rimf_spin.setValue(float(p.get("rimf_value", 0.67)))
        self._rimf_spin.setEnabled(use_fixed)
        self._compute_loc_prec.setChecked(bool(d.get("compute_loc_prec", False)))
        self._compute_density.setChecked(bool(d.get("compute_local_density", False)))
        self._density_radius.setValue(float(d.get("local_density_radius", 100)))
        self._density_dimensions = int(d.get("local_density_dimensions", 2))
        self._set_combo_data(self._density_method, d.get("local_density_method", "kdtree"))
        self._density_voxel.setValue(float(d.get("local_density_voxel_size", 100)))
        self._density_sigma.setValue(float(d.get("local_density_smooth_sigma", 1.0)))
        self._show_data_info.setChecked(bool(d.get("show_data_info", True)))
        self._show_attr.setChecked(bool(d.get("show_attr_plot", True)))
        self._show_scatter.setChecked(bool(d.get("show_scatter", False)))
        self._show_hist.setChecked(bool(d.get("show_histogram", False)))
        self._show_render.setChecked(bool(d.get("show_render", False)))

        # Plot
        self._px_spin.setValue(float(p.get("render_pixel_size", 2)))
        self._set_combo(self._render_cmap_combo, p.get("render_cmap", "Hot"), _RENDER_CMAPS)
        self._set_combo(self._scatter_color_by_combo, p.get("scatter_color_by", "tid"),
                        _SCATTER_COLOR_BY_OPTIONS)
        self._set_combo(self._scatter_cmap_combo, p.get("scatter_cmap", "jet"), _SCATTER_CMAPS)
        self._set_combo(self._plot_cmap_combo, p.get("attr_cmap", "single color"), _ATTR_CMAPS)
        self._set_combo(self._roi_color_combo, p.get("roi_color", "Yellow"), _ROI_COLORS)
        self._roi_transparency.setValue(int(p.get("roi_transparency", 50)))
        self._roi_edge_size.setValue(int(p.get("roi_edge_size", 1)))
        self._roi_edit_widget_size.setValue(int(p.get("roi_edit_widget_size", 8)))
        self._set_combo(self._filter_range_color_combo, p.get("filter_range_color", "Green"),
                        _ROI_COLORS)
        self._filter_range_alpha.setValue(int(p.get("filter_range_alpha", 45)))
        self._set_combo(self._filter_bounds_color_combo, p.get("filter_bounds_color", "Green"),
                        _ROI_COLORS)
        self._filter_bounds_size.setValue(int(p.get("filter_bounds_size", 1)))
        hist_values = set(p.get("histogram_values", ["trace mean"]))
        self._hist_trace_mean.setChecked("trace mean" in hist_values)
        self._hist_trace_median.setChecked("trace median" in hist_values)
        self._hist_trace_min.setChecked("trace min" in hist_values)
        self._hist_trace_max.setChecked("trace max" in hist_values)
        self._hist_trace_stdev.setChecked("trace stdev" in hist_values)
        self._hist_trace_range.setChecked("trace range" in hist_values)

        self._temp_folder_edit.setText(f.get("temp_folder", ""))

        # Plugin
        self._paraview_edit.setText(f.get("paraview_path", ""))
        self._msr_remember.setChecked(bool(g.get("msr_remember_last", True)))

        self._shortcut_table.setRowCount(0)
        for key, label in _SHORTCUT_LABELS.items():
            self._add_shortcut_row(key, str(s.get(key, "") or ""))

        enabled_attrs = set(a.get("enabled", []))
        for name, cb in self._attribute_checks.items():
            cb.setChecked(name in enabled_attrs)
        computed_attrs = {"siz" if name == "nLoc" else name for name in a.get("computed", [name for name, _ in _COMPUTED_ATTRIBUTE_INFO])}
        for name, cb in self._computed_attribute_checks.items():
            cb.setChecked(name in computed_attrs)

        self._mbm_only_used.setChecked(bool(m.get("only_used_for_drift_correction", False)))
        self._mbm_min_locs.setValue(int(m.get("minimum_localizations_per_bead", 10)))
        self._set_combo(self._mbm_average_method, m.get("average_method", "median"), ["mean", "median"])
        self._mbm_average_count.setValue(int(m.get("average_occurrence_count", 10)))
        self._set_combo(self._mbm_transform_type, m.get("transform_type", _MBM_TRANSFORM_TYPES[0]), _MBM_TRANSFORM_TYPES)
        self._set_combo(self._mbm_align_to, m.get("align_to_channel", "first"), ["first", "last"])

    def _apply_widgets_to_draft(self) -> None:
        f = self._draft["file"]
        d = self._draft["data"]
        p = self._draft["plot"]
        g = self._draft.setdefault("plugin", {})
        s = self._draft.setdefault("shortcuts", {})
        a = self._draft.setdefault("attributes", {})
        m = self._draft.setdefault("mbm_handling", {})

        # File
        f["num_file_history"] = int(self._files_in_history.value())
        f["keep_last_folder"] = bool(self._keep_last_folder.isChecked())
        f["confirm_overwrite"] = bool(self._confirm_overwrite.isChecked())
        f["close_paraview_on_exit"] = bool(self._close_paraview.isChecked())

        # Data
        d["iter_load"] = self._iter_load.currentText()
        d["only_valid_locs"] = bool(self._only_valid.isChecked())
        d["load_efc_cfr"] = bool(self._load_efc_cfr.isChecked())
        d["load_all_dcr"] = bool(self._load_all_dcr.isChecked())
        d["enforce_min_z_range"] = bool(self._enforce_z.isChecked())
        d["min_z_range_nm"] = float(self._min_z_spin.value())
        d["compute_rimf"] = bool(self._compute_rimf.isChecked())
        p["use_fixed_rimf"] = bool(self._use_fixed_rimf.isChecked())
        p["rimf_value"] = float(self._rimf_spin.value())
        d["compute_loc_prec"] = bool(self._compute_loc_prec.isChecked())
        d["compute_local_density"] = bool(self._compute_density.isChecked())
        d["local_density_radius"] = float(self._density_radius.value())
        d["local_density_dimensions"] = int(getattr(self, "_density_dimensions", 2))
        d["local_density_method"] = str(self._density_method.currentData() or "kdtree")
        d["local_density_voxel_size"] = float(self._density_voxel.value())
        d["local_density_smooth_sigma"] = float(self._density_sigma.value())
        d["show_data_info"] = bool(self._show_data_info.isChecked())
        d["show_attr_plot"] = bool(self._show_attr.isChecked())
        d["show_scatter"] = bool(self._show_scatter.isChecked())
        d["show_histogram"] = bool(self._show_hist.isChecked())
        d["show_render"] = bool(self._show_render.isChecked())

        # Plot
        p["render_pixel_size"] = float(self._px_spin.value())
        p["render_cmap"] = self._render_cmap_combo.currentText()
        p["scatter_color_by"] = self._scatter_color_by_combo.currentText()
        p["scatter_cmap"] = self._scatter_cmap_combo.currentText()
        p["attr_cmap"] = self._plot_cmap_combo.currentText()
        p["roi_color"] = self._roi_color_combo.currentText()
        p["roi_transparency"] = int(self._roi_transparency.value())
        p["roi_edge_size"] = int(self._roi_edge_size.value())
        p["roi_edit_widget_size"] = int(self._roi_edit_widget_size.value())
        p["filter_range_color"] = self._filter_range_color_combo.currentText()
        p["filter_range_alpha"] = int(self._filter_range_alpha.value())
        p["filter_bounds_color"] = self._filter_bounds_color_combo.currentText()
        p["filter_bounds_size"] = int(self._filter_bounds_size.value())
        hist_values = []
        for name, cb in (
            ("trace mean", self._hist_trace_mean),
            ("trace median", self._hist_trace_median),
            ("trace min", self._hist_trace_min),
            ("trace max", self._hist_trace_max),
            ("trace stdev", self._hist_trace_stdev),
            ("trace range", self._hist_trace_range),
        ):
            if cb.isChecked():
                hist_values.append(name)
        p["histogram_values"] = hist_values or ["trace mean"]

        f["temp_folder"] = self._temp_folder_edit.text().strip()

        # Plugin
        f["paraview_path"] = self._paraview_edit.text().strip()
        g["msr_remember_last"] = bool(self._msr_remember.isChecked())

        s.clear()
        for command in _SHORTCUT_LABELS:
            s[command] = ""
        for row in range(self._shortcut_table.rowCount()):
            command_widget = self._shortcut_table.cellWidget(row, 0)
            key_widget = self._shortcut_table.cellWidget(row, 1)
            if not isinstance(command_widget, QComboBox) or not isinstance(key_widget, QKeySequenceEdit):
                continue
            command = command_widget.currentData()
            sequence = key_widget.keySequence().toString(QKeySequence.SequenceFormat.PortableText)
            if command and sequence:
                s[str(command)] = sequence

        a["enabled"] = [
            name for name in _ATTRIBUTE_ORDER
            if self._attribute_checks[name].isChecked()
        ]
        a["computed"] = [
            name for name, _description in _COMPUTED_ATTRIBUTE_INFO
            if self._computed_attribute_checks[name].isChecked()
        ]

        m["only_used_for_drift_correction"] = bool(self._mbm_only_used.isChecked())
        m["minimum_localizations_per_bead"] = int(self._mbm_min_locs.value())
        m["average_method"] = self._mbm_average_method.currentText()
        m["average_occurrence_count"] = int(self._mbm_average_count.value())
        m["transform_type"] = self._mbm_transform_type.currentText()
        m["align_to_channel"] = self._mbm_align_to.currentText()

    @staticmethod
    def _set_combo(combo: QComboBox, value: str, options: list[str]) -> None:
        """Case-insensitive match; falls back to the first option."""
        for i, opt in enumerate(options):
            if opt.lower() == str(value).lower():
                combo.setCurrentIndex(i)
                return
        combo.setCurrentIndex(0)

    @staticmethod
    def _set_combo_data(combo: QComboBox, value: str) -> None:
        for i in range(combo.count()):
            if str(combo.itemData(i)).lower() == str(value).lower():
                combo.setCurrentIndex(i)
                return
        combo.setCurrentIndex(0)

    # ------------------------------------------------------------------
    # Accept / Reset
    # ------------------------------------------------------------------

    def _accept(self) -> None:
        self._apply_widgets_to_draft()
        self._state.prefs = self._draft
        self._state.save_prefs()
        self._state.log("Preferences saved.")
        self.accept()

    def _reset_current_tab(self) -> None:
        idx = self._stack.currentIndex()
        keys = self._page_keys[idx] if 0 <= idx < len(self._page_keys) else []
        for k in keys:
            if k in DEFAULT_PREFS:
                self._draft[k] = copy.deepcopy(DEFAULT_PREFS[k])
                # Preserve recent files when resetting File tab
                if k == "file":
                    self._draft["file"]["recent_files"] = self._state.prefs.get(
                        "file", {}
                    ).get("recent_files", [])
        self._load_draft_into_widgets()
