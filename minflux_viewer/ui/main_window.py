"""
minflux_viewer.ui.main_window
==============================
Main application window — Fiji-style QMainWindow.

Menu structure
--------------
File    — Open / Open recent / Save / Quit
Edit    — Dataset manager / Filter
View    — Scatter plot / Histogram / Attribute plot / Log
Process — Render image  (Phase 3)
Analysis — Loc precision / Local density  (Phase 4)
Help    — About
"""

from __future__ import annotations

from datetime import datetime
from pathlib import Path

from PyQt6.QtCore import QEvent, QObject, QPoint, QRunnable, QThreadPool, QTimer, Qt, pyqtSignal
from PyQt6.QtGui import (
    QAction,
    QActionGroup,
    QColor,
    QDragEnterEvent,
    QDropEvent,
    QIcon,
    QKeySequence,
    QShortcut,
    QPalette,
)
from PyQt6.QtWidgets import (
    QApplication,
    QComboBox,
    QDialog,
    QDoubleSpinBox,
    QFileDialog,
    QKeySequenceEdit,
    QLabel,
    QLineEdit,
    QMainWindow,
    QMenu,
    QMessageBox,
    QPlainTextEdit,
    QProxyStyle,
    QPushButton,
    QSizePolicy,
    QStyle,
    QStyleOptionMenuItem,
    QSpinBox,
    QTextEdit,
    QVBoxLayout,
    QWidget,
)

from .. import resource_path
from ..core.app_state import AppState
from .data_window import DataWindow

# ---------------------------------------------------------------------------
# Supported file extensions for drag-and-drop and open dialogs
# ---------------------------------------------------------------------------

_SUPPORTED_EXTS: tuple[str, ...] = (
    ".mat", ".npy", ".csv", ".tsv", ".txt", ".xlsx", ".xlsm", ".msr",
    ".tif", ".tiff", ".json", ".roi", ".zip",
)
#: ROI-set files (loaded into the ROI Manager, not as datasets).
_ROI_FILE_EXTS: frozenset[str] = frozenset({".roi", ".zip"})

#: Canonical loader format (see core.format_sniff.resolve_format) → method name.
_FMT_LOADERS: dict[str, str] = {
    "mat": "_load_mat", "npy": "_load_npy", "spreadsheet": "_load_spreadsheet",
    "msr": "_open_msr_dialog", "tiff": "_load_tiff", "json": "_load_json",
}

_ROI_TOOL_DEFS: tuple[tuple[str, str, str], ...] = (
    ("Rectangle", "rectangle", "toolRect"),
    ("Oval", "oval", "toolOval"),
    ("Polygon", "polygon", "toolPolygon"),
    ("Freehand", "freehand", "toolFreehand"),
    ("Line", "line", "toolLine"),
    ("Point", "point", "toolPoint"),
)

#: Fiji-style line family hosted on the single Line toolbar button
#: (label, tool, icon file). Right-click the Line button to switch variant.
_LINE_FAMILY: tuple[tuple[str, str, str], ...] = (
    ("Straight Line", "line", "line.png"),
    ("Polyline", "polyline", "polyline.png"),
    ("Freehand Line", "freehand_line", "freeline.png"),
)


_AI_UNAPPROVED_MENU_TEXT = QColor("#404040")


class _AiMenuStyle(QProxyStyle):
    """Draw unverified QAction text grey without replacing native menu rows."""

    def drawControl(self, element, option, painter, widget=None):  # noqa: N802 - Qt API
        if (
            element == QStyle.ControlElement.CE_MenuItem
            and isinstance(option, QStyleOptionMenuItem)
            and isinstance(widget, QMenu)
        ):
            action = widget.actionAt(option.rect.center())
            if action is not None and action.property("ai_unapproved"):
                opt = QStyleOptionMenuItem(option)
                for role in (
                    QPalette.ColorRole.Text,
                    QPalette.ColorRole.WindowText,
                    QPalette.ColorRole.ButtonText,
                ):
                    opt.palette.setColor(role, _AI_UNAPPROVED_MENU_TEXT)
                return super().drawControl(element, opt, painter, widget)
        return super().drawControl(element, option, painter, widget)


class _ScrollableMenuStyle(QProxyStyle):
    """Make an over-tall menu scroll (arrows at the top/bottom edges, which
    auto-scroll on hover) instead of wrapping into multiple columns. Used for
    the recent-files submenu so a long history stays navigable."""

    def styleHint(self, hint, option=None, widget=None, returnData=None):  # noqa: N802 - Qt API
        if hint == QStyle.StyleHint.SH_Menu_Scrollable:
            return 1
        return super().styleHint(hint, option, widget, returnData)


class _UpdateCheckSignals(QObject):
    done = pyqtSignal(object)   # core.updater.UpdateCheckResult


class _UpdateCheckTask(QRunnable):
    """Run the GitHub update check off the UI thread (Tier-A in-app updater)."""

    def __init__(self, current_version: str) -> None:
        super().__init__()
        self.signals = _UpdateCheckSignals()
        self._current = current_version

    def run(self) -> None:  # noqa: N802 - Qt API
        from ..core.updater import UpdateCheckResult, check_for_update
        try:
            result = check_for_update(self._current)
        except Exception as exc:  # never let a worker exception escape
            result = UpdateCheckResult("error", self._current, None, str(exc))
        self.signals.done.emit(result)


class MainWindow(QMainWindow):
    """Top-level application window."""

    APP_NAME = "MINFLUX Data Viewer"

    def __init__(self, state: AppState) -> None:
        super().__init__()
        self._state = state
        self._data_windows: dict[int, DataWindow] = {}

        # One reusable plot window per dataset.
        self._scatter_windows: dict[int, QWidget] = {}
        self._histogram_windows: dict[int, QWidget] = {}
        self._attr_windows: dict[int, QWidget] = {}
        self._scatter_win   = None       # compatibility alias: most recently raised
        self._histogram_win = None       # compatibility alias: most recently raised
        self._attr_win      = None       # compatibility alias: most recently raised
        self._attr_3d_mpl_win = None
        self._filter_dlg    = None
        self._filter_dlgs: dict[int | None, QWidget] = {}
        self._ds_manager    = None
        self._log_win       = None
        self._console_win   = None
        self._memory_win    = None
        self._roi_manager_win = None
        self._script_editor_win = None
        self._shortcut_actions: dict[str, QAction] = {}
        self._roi_tool_actions: dict[str, QAction] = {}
        self._ai_menu_styles: list[_AiMenuStyle] = []
        self._window_cycle_index = -1
        # One render window per dataset: {dataset_idx: RenderWindow}
        self._render_windows: dict[int, "RenderWindow"] = {}
        # Standalone TIFF viewers are not MINFLUX datasets and never appear in
        # the dataset manager.
        self._tiff_windows: dict[str, QWidget] = {}
        self._last_channel_combine_settings: dict | None = None
        self._next_overlay_index = 1
        # ParaView subprocesses spawned by this session — terminated on exit
        self._paraview_procs: list = []

        # Apply the generated UI (menus, toolbar, status bar, central widget)
        from .generated.main_window_ui import Ui_MainWindow
        self._ui = Ui_MainWindow()
        self._ui.setupUi(self)
        self.setWindowIcon(QIcon(str(resource_path("icons", "minflux_viewer_logo.png"))))
        self._ui.toolbar.setWindowTitle("Main Toolbar")
        self._ui.toolbar.setContextMenuPolicy(Qt.ContextMenuPolicy.NoContextMenu)
        self._ui.toolbar.setStyleSheet(
            "QToolButton:checked {"
            "  background: rgba(0, 120, 215, 0.18);"
            "  border: 1px solid rgba(0, 120, 215, 0.95);"
            "  border-radius: 4px;"
            "  padding: 2px;"
            "}"
            "QToolButton:checked:hover {"
            "  background: rgba(0, 120, 215, 0.26);"
            "}"
        )

        # Override the central widget with the live drop-target WelcomeWidget.
        # Designer cannot declare custom widget subclasses without a plugin;
        # the hint label stays where the designer placed it.
        self.setCentralWidget(_WelcomeWidget())

        # Status label lives in the status bar
        self._status_label = QLabel("No data loaded.")
        self.statusBar().addWidget(self._status_label)

        # The generated UI defines menus and actions but leaves the recent-files
        # submenu empty — we expose it as self._recent_menu for compatibility
        # with _populate_recent_menu().
        self._recent_menu = self._ui.menuOpenRecent
        # Scroll (don't wrap into columns) when the history is long.
        # setStyle() doesn't take ownership, so keep a reference alive.
        self._recent_menu_style = _ScrollableMenuStyle(self._recent_menu.style())
        self._recent_menu.setStyle(self._recent_menu_style)

        # Wire actions to handlers (previously spread across _build_menu/_toolbar)
        self._connect_actions()
        self._populate_recent_menu()
        self._populate_plugins_menu()
        self._bind_scripting_api()

        # React to application-state changes
        state.dataset_added.connect(self._on_dataset_added)
        state.dataset_removed.connect(self._on_dataset_removed)
        state.active_changed.connect(self._on_active_changed)
        state.status_message.connect(self._status_label.setText)
        state.log_message.connect(self._on_log_message)

        # Remembered ROI duplicate/crop options, per dataset (session-only,
        # keyed by dataset identity — "use the same setup and stop asking").
        self._roi_crop_setup: dict = {}

        # Tier-A in-app update check (opt-out via Preferences > File). Delayed so
        # it never slows startup; silent unless a newer release exists.
        self._update_tasks: set = set()
        if state.prefs.get("file", {}).get("check_updates_on_startup", True):
            QTimer.singleShot(3000, lambda: self._check_for_updates(silent=True))

    def createPopupMenu(self):  # noqa: N802 - Qt override
        """Suppress Qt's default toolbar visibility popup on right-click."""
        return None

    # ------------------------------------------------------------------
    # Action wiring
    # ------------------------------------------------------------------

    def _connect_actions(self) -> None:
        """Connect every QAction from the generated UI to its handler."""
        u = self._ui

        # File menu  (.msr opens via drag-drop / the Plugins > MSR Reader entry,
        # so there is no dedicated File > "Open .msr" item.)
        u.actionOpen.triggered.connect(self._open_dialog)
        self.actionOpenSpreadsheet = QAction("Open spreadsheet data...", self)
        self.actionOpenSpreadsheet.triggered.connect(self._open_spreadsheet_dialog)
        self.actionOpenTiff = QAction("Open .tif file...", self)
        self.actionOpenTiff.triggered.connect(self._open_tiff_dialog)
        u.actionSave.triggered.connect(self._save_data)
        u.actionQuit.triggered.connect(self.close)
        self.actionClose = QAction("Close", self)
        self.actionClose.setShortcut(QKeySequence("Ctrl+W"))
        self.actionClose.setShortcutContext(Qt.ShortcutContext.ApplicationShortcut)
        self.actionClose.triggered.connect(self._close_active_dataset)
        self.actionCloseAll = QAction("Close All", self)
        self.actionCloseAll.setShortcut(QKeySequence("Ctrl+Shift+W"))
        self.actionCloseAll.setShortcutContext(Qt.ShortcutContext.ApplicationShortcut)
        self.actionCloseAll.triggered.connect(self._close_all_datasets)

        # Edit menu
        u.actionDatasetManager.triggered.connect(self._show_dataset_manager)
        u.actionFilter.triggered.connect(self._show_filter)
        u.actionDuplicate.triggered.connect(self._duplicate_active_dataset)
        u.actionPreferences.triggered.connect(self._show_preferences)

        # View menu
        u.actionScatter.triggered.connect(self._show_scatter)
        u.actionHistogram.triggered.connect(self._show_histogram)
        u.actionAttributePlot.triggered.connect(self._show_attr_plot)
        self.actionAttributePlot3DMatplotlib = QAction("Attribute Plot 3D (Matplotlib)", self)
        self.actionAttributePlot3DMatplotlib.triggered.connect(self._show_attr_plot_3d_matplotlib)
        u.actionRender.triggered.connect(self._show_render)        # was Process > Render image
        u.actionShowInfo.triggered.connect(self._show_info_for_active)
        u.actionLog.triggered.connect(self._show_log)
        u.actionConsole.triggered.connect(self._show_console)
        # Brightness & Contrast: removed from menu but the action object is
        # still defined so the render window's own toolbar button works.
        u.actionBrightnessContrast.triggered.connect(self._show_brightness_contrast)

        # Analysis menu
        self.menuMeasure = QMenu("Measure", self)
        self.actionScaleBar = QAction("Scale Bar", self)
        self.actionScaleBar.triggered.connect(self._show_scale_bar)
        self.actionPlotProfile = QAction("Plot Profile", self)
        self.actionPlotProfile.triggered.connect(
            lambda: self._placeholder("Plot Profile", "a later implementation")
        )
        self.menuMeasure.addAction(self.actionScaleBar)
        self.menuMeasure.addAction(self.actionPlotProfile)
        self.actionSetMeasurements = QAction("Set Measurements...", self)
        self.actionSetMeasurements.triggered.connect(self._show_set_measurements)
        u.actionLocPrecisionFrc.triggered.connect(self._loc_precision_frc)
        u.actionLocPrecisionCrlb.triggered.connect(self._loc_precision_crlb)
        u.actionLocPrecisionStdDev.triggered.connect(self._loc_precision_stddev)
        u.actionLocalDensity.triggered.connect(self._run_local_density)
        self.menuAnalyzeClustering = QMenu("Clustering", self)
        self.actionDbscan = QAction("DBSCAN", self)
        self.actionDbscan.triggered.connect(
            lambda: self._placeholder("DBSCAN clustering", "a later implementation")
        )
        self.actionKNearestNeighbour = QAction("K Nearest Neighbour", self)
        self.actionKNearestNeighbour.triggered.connect(
            lambda: self._placeholder("K nearest neighbour", "a later implementation")
        )
        self.menuHlyBPair = QMenu("HlyB subunit pair analysis", self)
        self.actionHlyB2D = QAction("2D", self)
        self.actionHlyB2D.triggered.connect(lambda: self._run_hlyb_pair_analysis(mode="2D"))
        self.actionHlyB3D = QAction("3D", self)
        self.actionHlyB3D.triggered.connect(lambda: self._run_hlyb_pair_analysis(mode="3D"))
        self.actionHlyBTemplate3D = QAction("Template matching (3D)", self)
        self.actionHlyBTemplate3D.triggered.connect(
            lambda: self._run_hlyb_pair_analysis(mode="TEMPLATE3D")
        )
        self.menuHlyBPair.addAction(self.actionHlyB2D)
        self.menuHlyBPair.addAction(self.actionHlyB3D)
        self.menuHlyBPair.addAction(self.actionHlyBTemplate3D)
        self.menuAnalyzeClustering.addAction(self.actionDbscan)
        self.menuAnalyzeClustering.addAction(self.actionKNearestNeighbour)
        self.menuAnalyzeClustering.addSeparator()
        self.menuAnalyzeClustering.addMenu(self.menuHlyBPair)

        # Trace submenu — size estimation, anisotropy, trace viewer
        self.menuAnalyzeTrace = QMenu("Trace", self)
        self.actionTraceSize = QAction("Estimate Average Trace Size", self)
        self.actionTraceSize.triggered.connect(self._run_trace_size)
        self.actionTraceAnisotropy = QAction("Estimate Anisotropy", self)
        self.actionTraceAnisotropy.triggered.connect(self._run_trace_anisotropy)
        self.actionTraceViewerAnalyze = QAction("Trace Viewer", self)
        self.actionTraceViewerAnalyze.triggered.connect(self._show_trace_viewer)
        self.menuAnalyzeTrace.addAction(self.actionTraceSize)
        self.menuAnalyzeTrace.addAction(self.actionTraceAnisotropy)
        self.menuAnalyzeTrace.addSeparator()
        self.menuAnalyzeTrace.addAction(self.actionTraceViewerAnalyze)

        # Segmentation submenu — structure segmentation (NPC; more to come)
        self.menuAnalyzeSegmentation = QMenu("Segmentation", self)
        self.menuSegNPC = QMenu("NPC", self)
        self.actionSegNpc2D = QAction("2D", self)
        self.actionSegNpc2D.triggered.connect(self._segment_npc_2d)
        self.actionSegNpc3D = QAction("3D", self)
        self.actionSegNpc3D.triggered.connect(
            lambda: self._placeholder("NPC segmentation (3D)", "a later implementation"))
        self.menuSegNPC.addAction(self.actionSegNpc2D)
        self.menuSegNPC.addAction(self.actionSegNpc3D)
        self.menuAnalyzeSegmentation.addMenu(self.menuSegNPC)
        self.actionSegConvolution = QAction("Convolution…", self)
        self.actionSegConvolution.triggered.connect(self._show_conv_segmentation)
        self.menuAnalyzeSegmentation.addAction(self.actionSegConvolution)
        self.actionSegConvolution3D = QAction("Convolution (3D)…", self)
        self.actionSegConvolution3D.triggered.connect(self._show_conv_segmentation_3d)
        self.menuAnalyzeSegmentation.addAction(self.actionSegConvolution3D)
        self.actionSegCurvilinear = QAction("Curvilinear Structures…", self)
        self.actionSegCurvilinear.triggered.connect(self._show_curvilinear_segmentation)
        self.menuAnalyzeSegmentation.addAction(self.actionSegCurvilinear)
        self.actionSegParticleAverage = QAction("Particle Average…", self)
        self.actionSegParticleAverage.triggered.connect(self._show_particle_average)
        self.menuAnalyzeSegmentation.addAction(self.actionSegParticleAverage)
        self.menuAnalyzeSegmentation.addSeparator()
        self.actionNpcDetectWanlu = QAction("NPC detection wanlu", self)
        self.actionNpcDetectWanlu.triggered.connect(self._npc_detect_wanlu)
        self.menuAnalyzeSegmentation.addAction(self.actionNpcDetectWanlu)
        self.actionNpcAverageWanlu = QAction("NPC average wanlu", self)
        self.actionNpcAverageWanlu.triggered.connect(self._npc_average_wanlu)
        self.menuAnalyzeSegmentation.addAction(self.actionNpcAverageWanlu)

        # Tracking submenu — Phase 5 placeholders
        u.actionParticleTracking.triggered.connect(
            lambda: self._placeholder("Particle tracking", "Phase 5")
        )
        u.actionTraceViewer.triggered.connect(self._show_trace_viewer)
        u.actionMsdAnalysis.triggered.connect(
            lambda: self._placeholder("MSD analysis", "Phase 5")
        )

        # Process menu — Batch Processing submenu (Phase 5 placeholders)
        u.actionBatchRender.triggered.connect(
            lambda: self._placeholder("Batch render", "Phase 5")
        )
        u.actionBatchExport.triggered.connect(
            lambda: self._placeholder("Batch export", "Phase 5")
        )
        u.actionBatchFilter.triggered.connect(
            lambda: self._placeholder("Batch filter", "Phase 5")
        )
        self.menuProcessChannel = QMenu("Channel...", self)
        self.actionChannelTool = QAction("Channel Tool", self)
        self.actionChannelTool.triggered.connect(
            lambda: self._placeholder("Channel Tool", "a later implementation")
        )
        self.actionChannelCombine = QAction("Combine...", self)
        self.actionChannelCombine.triggered.connect(self._show_channel_combine)
        self.actionChannelSplit = QAction("Split...", self)
        self.actionChannelSplit.triggered.connect(self._split_active_channel_group)
        self.actionChannelOverlay = QAction("Overlay", self)
        self.actionChannelOverlay.triggered.connect(
            lambda: self._placeholder("Channel overlay", "a later implementation")
        )
        self.actionChannelSeparateDcr = QAction("Separate Channel by DCR", self)
        self.actionChannelSeparateDcr.triggered.connect(self._show_channel_separation)
        self.menuProcessChannel.addAction(self.actionChannelTool)
        self.menuProcessChannel.addAction(self.actionChannelCombine)
        self.menuProcessChannel.addAction(self.actionChannelSplit)
        self.menuProcessChannel.addAction(self.actionChannelOverlay)
        self.menuProcessChannel.addAction(self.actionChannelSeparateDcr)

        self.menuProcessRoi = QMenu("ROI", self)
        self.actionRoiManager = QAction("ROI Manager", self)
        self.actionRoiManager.triggered.connect(self._show_roi_manager)
        self.menuProcessRoi.addAction(self.actionRoiManager)
        self.menuProcessRoi.addSeparator()
        # Convert submenu: only the target type is listed; the source is the
        # active ROI at call time (selected ROI, else the active draft).
        self.menuRoiConvert = QMenu("Convert", self)
        self._roi_convert_actions = {}
        for label, target in (
            ("to Rectangle", "rectangle"),
            ("to Oval", "oval"),
            ("Line to Region", "region"),
            ("Region to Line", "line"),
            ("to Point", "point"),
        ):
            act = QAction(label, self)
            act.triggered.connect(lambda _checked=False, t=target: self._convert_active_roi(t))
            self.menuRoiConvert.addAction(act)
            self._roi_convert_actions[target] = act
        self.menuProcessRoi.addMenu(self.menuRoiConvert)
        self.actionRoiResize = QAction("Enlarge / Shrink…", self)
        self.actionRoiResize.triggered.connect(self._resize_active_roi)
        self.menuProcessRoi.addAction(self.actionRoiResize)
        self.actionRoiSkeletonize = QAction("Skeletonize", self)
        self.actionRoiSkeletonize.triggered.connect(self._skeletonize_active_roi)
        self.menuProcessRoi.addAction(self.actionRoiSkeletonize)
        self.actionRoiConvexHull = QAction("Convex Hull", self)
        self.actionRoiConvexHull.triggered.connect(self._convex_hull_active_roi)
        self.menuProcessRoi.addAction(self.actionRoiConvexHull)

        # Help menu
        u.actionAbout.triggered.connect(self._show_about)
        u.actionMemoryMonitor.triggered.connect(self._show_memory_monitor)
        self.actionCheckUpdates = QAction("Check for Updates…", self)
        self.actionCheckUpdates.triggered.connect(
            lambda: self._check_for_updates(silent=False)
        )
        u.menuHelp.insertAction(u.actionAbout, self.actionCheckUpdates)
        u.menuHelp.insertSeparator(u.actionAbout)

        # Toolbar tools — LUT and Color (the silent-failure bug fix)
        u.toolLut.triggered.connect(self._show_lut)
        u.toolColor.triggered.connect(self._show_color_picker)
        self._setup_toolbar_widgets()
        self._roi_tool_group = QActionGroup(self)
        try:
            self._roi_tool_group.setExclusionPolicy(
                QActionGroup.ExclusionPolicy.ExclusiveOptional
            )
        except Exception:
            self._roi_tool_group.setExclusive(False)
        # Fiji-style: the Line toolbar button hosts a *family* (straight / poly /
        # freehand line); right-click it to switch, left-click activates the
        # current variant. ``_line_variant`` tracks the active member.
        self._line_variant = "line"
        for _label, tool, attr in _ROI_TOOL_DEFS:
            action = getattr(u, attr)
            self._roi_tool_actions[tool] = action
            self._roi_tool_group.addAction(action)
            if tool == "line":
                action.triggered.connect(lambda checked: self._on_roi_tool(self._line_variant, checked))
            else:
                action.triggered.connect(lambda checked, t=tool: self._on_roi_tool(t, checked))
        # Angle tool (ImageJ-style; not in the generated .ui). It's a measurement
        # tool — three points A·B·C reporting the angle ABC — placed between the
        # Line and Point tools on the toolbar.
        self.toolAngle = QAction("Angle", self)
        self.toolAngle.setCheckable(True)
        self.toolAngle.setToolTip("Angle — click A, B (vertex), C to measure the angle ABC")
        u.toolbar.insertAction(u.toolPoint, self.toolAngle)
        self._roi_tool_actions["angle"] = self.toolAngle
        self._roi_tool_group.addAction(self.toolAngle)
        self.toolAngle.triggered.connect(lambda checked: self._on_roi_tool("angle", checked))
        # Magnetic lasso tool (snaps to the rendered density centre), placed right
        # after the Point tool.
        self.toolMagneticLasso = QAction("Magnetic Lasso", self)
        self.toolMagneticLasso.setCheckable(True)
        self.toolMagneticLasso.setToolTip(
            "Magnetic lasso — click to trace; each vertex snaps to the high-density "
            "centre of the structure under the cursor. Right-click / double-click / "
            "Enter to finish.")
        anchor = getattr(u, "toolLut", None)
        if anchor is not None:
            u.toolbar.insertAction(anchor, self.toolMagneticLasso)
            # Separator sits to the right of the Magnetic Lasso (left of LUT);
            # no separator between Point and Magnetic Lasso.
            u.toolbar.insertSeparator(anchor)
        else:
            u.toolbar.addAction(self.toolMagneticLasso)
        self._roi_tool_actions["magnetic_lasso"] = self.toolMagneticLasso
        self._roi_tool_group.addAction(self.toolMagneticLasso)
        self.toolMagneticLasso.triggered.connect(lambda checked: self._on_roi_tool("magnetic_lasso", checked))
        self._state.rois.tool_changed.connect(self._sync_roi_tool_actions)

        # Wire icon images for toolbar buttons
        self._install_toolbar_icons()
        self._install_roi_tool_menus()
        self._configure_menus()
        self._mark_ai_unapproved_actions()
        self._apply_shortcuts()

    # ------------------------------------------------------------------
    # Menu and shortcuts
    # ------------------------------------------------------------------

    def _configure_menus(self) -> None:
        u = self._ui

        u.actionOpen.setText("Open...")
        u.actionSave.setText("Save Processed Data...")
        u.actionQuit.setText("Quit")
        u.actionDatasetManager.setText("Dataset Manager")
        u.actionFilter.setText("Filter...")
        u.actionDuplicate.setText("Duplicate")
        u.actionPreferences.setText("Preferences...")
        u.actionHistogram.setText("Attribute Histogram")
        u.actionScatter.setText("Loc Scatter Plot")
        u.actionAttributePlot.setText("Attribute Plot")
        self.actionAttributePlot3DMatplotlib.setText("Attribute Plot 3D (Matplotlib)")
        u.actionShowInfo.setText("Show Info...")
        u.actionRender.setText("Render")
        u.actionLog.setText("Log (Events)")
        u.actionConsole.setText("Console (stdout / stderr)")
        u.actionMemoryMonitor.setText("Monitor Memory...")
        u.actionLocPrecisionFrc.setText("FRC (Fourier Ring Correlation)")
        u.actionLocPrecisionCrlb.setText("CRLB (Cramer-Rao Lower Bound)")
        u.actionLocPrecisionStdDev.setText("StdDev per Trace")
        u.actionLocalDensity.setText("Local Density")
        u.actionParticleTracking.setText("Particle Tracking")
        u.actionTraceViewer.setText("Trace Viewer")
        u.actionMsdAnalysis.setText("MSD Analysis")
        u.actionBatchRender.setText("Batch Render...")
        u.actionBatchExport.setText("Batch Export...")
        u.actionBatchFilter.setText("Batch Filter...")
        u.actionAbout.setText("About")
        self.menuMeasure.setTitle("Measure")
        self.actionSetMeasurements.setText("Set Measurements...")
        self.menuAnalyzeClustering.setTitle("Clustering")
        self.actionDbscan.setText("DBSCAN")
        self.actionKNearestNeighbour.setText("K Nearest Neighbour")
        self.menuProcessChannel.setTitle("Channel...")
        self.actionChannelTool.setText("Channel Tool")
        self.actionChannelCombine.setText("Combine...")
        self.actionChannelSplit.setText("Split...")
        self.actionChannelOverlay.setText("Overlay")
        self.actionChannelSeparateDcr.setText("Separate Channel by DCR")
        self.menuProcessRoi.setTitle("ROI")
        self.actionRoiManager.setText("ROI Manager")
        self.actionOpenSpreadsheet.setText("Open Spreadsheet Data...")
        self.actionOpenTiff.setText("Open .tif File...")
        u.menuOpenRecent.setTitle("Open Recent")
        u.menuBatchProcessing.setTitle("Batch Processing")
        u.menuAnalysis.setTitle("Analyze")
        u.menuLocPrecision.setTitle("Localization Precision")
        u.menuTracking.setTitle("Tracking")

        u.menuFile.clear()
        u.menuFile.addAction(u.actionOpen)
        u.menuFile.addAction(self.actionOpenSpreadsheet)
        u.menuFile.addAction(self.actionOpenTiff)
        u.menuFile.addAction(u.menuOpenRecent.menuAction())
        u.menuFile.addSeparator()
        u.menuFile.addAction(u.actionSave)
        u.menuFile.addSeparator()
        u.menuFile.addAction(self.actionClose)
        u.menuFile.addAction(self.actionCloseAll)
        u.menuFile.addSeparator()
        u.menuFile.addAction(u.actionPreferences)
        u.menuFile.addSeparator()
        u.menuFile.addAction(u.actionQuit)

        u.menuView.clear()
        u.menuView.addAction(u.actionShowInfo)
        u.menuView.addAction(u.actionDatasetManager)
        u.menuView.addSeparator()
        u.menuView.addAction(u.actionAttributePlot)
        u.menuView.addAction(self.actionAttributePlot3DMatplotlib)
        u.menuView.addAction(u.actionHistogram)
        u.menuView.addAction(u.actionScatter)
        u.menuView.addAction(u.actionRender)
        u.menuView.addAction(u.actionTraceViewer)
        u.menuView.addSeparator()
        u.menuView.addAction(u.actionLog)

        u.menuProcess.clear()
        u.menuProcess.addAction(self.menuProcessChannel.menuAction())
        u.menuProcess.addAction(self.menuProcessRoi.menuAction())
        u.menuProcess.addSeparator()
        u.menuProcess.addAction(u.menuBatchProcessing.menuAction())

        u.menuAnalysis.clear()
        u.menuAnalysis.addAction(self.menuMeasure.menuAction())
        u.menuAnalysis.addAction(self.actionSetMeasurements)
        u.menuAnalysis.addSeparator()
        u.menuAnalysis.addAction(u.menuLocPrecision.menuAction())
        u.menuAnalysis.addAction(u.actionLocalDensity)
        u.menuAnalysis.addAction(self.menuAnalyzeClustering.menuAction())
        u.menuAnalysis.addAction(self.menuAnalyzeTrace.menuAction())
        u.menuAnalysis.addAction(self.menuAnalyzeSegmentation.menuAction())
        u.menuAnalysis.addAction(u.menuTracking.menuAction())

        self._shortcut_actions = {
            "open": u.actionOpen,
            "open_spreadsheet": self.actionOpenSpreadsheet,
            "open_tiff": self.actionOpenTiff,
            "save": u.actionSave,
            "filter": u.actionFilter,
            "duplicate": u.actionDuplicate,
            "show_info": u.actionShowInfo,
            "render": u.actionRender,
            "brightness_contrast": u.actionBrightnessContrast,
            "attribute_plot": u.actionAttributePlot,
            "attribute_histogram": u.actionHistogram,
            "scatter_plot": u.actionScatter,
            "log": u.actionLog,
            "console": u.actionConsole,
            "preferences": u.actionPreferences,
            "dataset_manager": u.actionDatasetManager,
        }

        # ApplicationShortcut so Shift+V reaches the main window from every
        # window in the app (filter dialogs, dataset manager, child UIs, etc.)
        # without needing per-window installation.
        self._app_shortcut_focus = QShortcut(self)
        self._app_shortcut_focus.setContext(Qt.ShortcutContext.ApplicationShortcut)
        self._app_shortcut_focus.activated.connect(self._focus_main_window)

    def _mark_ai_unapproved_actions(self) -> None:
        """Visually separate AI-generated/unapproved actions from approved UI."""
        u = self._ui
        actions = [
            u.actionSave,
            self.actionOpenSpreadsheet,
            self.actionOpenTiff,
            self.menuProcessChannel.menuAction(),
            self.actionChannelTool,
            self.actionChannelCombine,
            self.actionChannelSplit,
            self.actionChannelOverlay,
            self.actionChannelSeparateDcr,
            self.menuProcessRoi.menuAction(),
            self.actionRoiManager,
            self.menuRoiConvert.menuAction(),
            *self._roi_convert_actions.values(),
            self.actionRoiResize,
            self.actionRoiSkeletonize,
            self.actionRoiConvexHull,
            u.menuBatchProcessing.menuAction(),
            u.actionBatchRender,
            u.actionBatchExport,
            u.actionBatchFilter,
            self.menuMeasure.menuAction(),
            self.actionPlotProfile,
            self.actionSetMeasurements,
            self.actionDbscan,
            self.actionKNearestNeighbour,
            self.menuHlyBPair.menuAction(),
            self.actionHlyB2D,
            self.actionHlyB3D,
            self.actionHlyBTemplate3D,
            self.menuAnalyzeSegmentation.menuAction(),
            self.actionSegNpc2D,
            self.actionSegNpc3D,
            self.actionSegConvolution,
            self.actionSegCurvilinear,
            self.actionSegParticleAverage,
            self.actionNpcDetectWanlu,
            self.actionNpcAverageWanlu,
            u.menuTracking.menuAction(),
            u.actionParticleTracking,
            u.actionMsdAnalysis,
            u.actionMemoryMonitor,
            self.actionAttributePlot3DMatplotlib,
        ]
        for action in actions:
            self._mark_action_ai_unapproved(action)

        for menu in (
            u.menuFile,
            u.menuView,
            self.menuProcessChannel,
            self.menuProcessRoi,
            self.menuRoiConvert,
            u.menuBatchProcessing,
            u.menuAnalysis,
            self.menuMeasure,
            self.menuAnalyzeClustering,
            self.menuHlyBPair,
            self.menuAnalyzeSegmentation,
            self.menuSegNPC,
            u.menuTracking,
            u.menuHelp,
        ):
            self._install_ai_menu_style(menu)

    def _mark_action_ai_unapproved(self, action: QAction) -> None:
        action.setProperty("ai_unapproved", True)
        tip = action.statusTip() or action.toolTip()
        note = "AI-generated; pending human approval."
        action.setStatusTip(f"{tip} {note}".strip() if tip else note)
        self.addAction(action)

    def _install_ai_menu_style(self, menu: QMenu) -> None:
        """Keep menu rows native-aligned while painting unapproved actions grey."""
        if menu.property("ai_menu_style_installed"):
            return
        style = _AiMenuStyle(menu.style())
        style.setParent(menu)
        menu.setStyle(style)
        menu.setProperty("ai_menu_style_installed", True)
        self._ai_menu_styles.append(style)

    def _apply_shortcuts(self) -> None:
        shortcuts = self._state.prefs.get("shortcuts", {})
        for key, action in self._shortcut_actions.items():
            seq = str(shortcuts.get(key, "") or "")
            action.setShortcut(QKeySequence(seq) if seq else QKeySequence())
            # WindowShortcut: fires when the main window is the active window
            # (WidgetShortcut required the widget itself to hold focus, which
            # menu actions almost never do — so the shortcuts never triggered).
            action.setShortcutContext(Qt.ShortcutContext.WindowShortcut)
        self._ui.actionQuit.setShortcut(QKeySequence("Ctrl+Q"))
        self._ui.actionQuit.setShortcutContext(Qt.ShortcutContext.ApplicationShortcut)
        seq = str(shortcuts.get("focus_main_window", "") or "")
        self._app_shortcut_focus.setKey(QKeySequence(seq) if seq else QKeySequence())
        self._app_shortcut_focus.setEnabled(bool(seq))
        self._install_window_shortcuts(self, include_action_commands=False)
        self._refresh_owned_window_shortcuts()
        conflicts = self._shortcut_conflicts()
        if conflicts:
            for seq, commands in conflicts.items():
                self._state.log(
                    f"Keyboard shortcut conflict: {seq} is assigned to {', '.join(commands)}.",
                    "WARN",
                )

    def _shortcut_command_keys(self, *, include_action_commands: bool = True) -> tuple[str, ...]:
        keys = (
            "next_window", "previous_window",
            "next_dataset", "previous_dataset", "close_window",
        )
        if include_action_commands:
            keys = (*keys, *self._shortcut_actions.keys())
        return keys

    def _install_window_shortcuts(
        self,
        widget: QWidget | None,
        *,
        include_action_commands: bool = True,
    ) -> None:
        """Install/update menu shortcuts on a top-level viewer window."""
        if widget is None:
            return
        shortcuts = getattr(widget, "_mfv_window_shortcuts", None)
        if shortcuts is None:
            shortcuts = {}
            setattr(widget, "_mfv_window_shortcuts", shortcuts)
        prefs = self._state.prefs.get("shortcuts", {})
        active_commands = set(self._shortcut_command_keys(
            include_action_commands=include_action_commands,
        ))
        for command, shortcut in list(shortcuts.items()):
            if command not in active_commands:
                shortcut.setEnabled(False)
        for command in active_commands:
            shortcut = shortcuts.get(command)
            if shortcut is None:
                shortcut = QShortcut(widget)
                shortcut.setContext(Qt.ShortcutContext.WidgetWithChildrenShortcut)
                shortcut.activated.connect(lambda cmd=command: self._trigger_shortcut_command(cmd))
                shortcuts[command] = shortcut
            seq = str(prefs.get(command, "") or "")
            shortcut.setKey(QKeySequence(seq) if seq else QKeySequence())
            shortcut.setEnabled(bool(seq))

    def _refresh_owned_window_shortcuts(self) -> None:
        for widget in QApplication.topLevelWidgets():
            if widget is self:
                self._install_window_shortcuts(widget, include_action_commands=False)
            elif getattr(widget, "TAG", None) in {
                "render_window", "attribute_window", "histogram_window",
                "scatter_window", "attribute_3d_matplotlib_window",
            }:
                self._install_window_shortcuts(widget)

    def _trigger_shortcut_command(self, command: str) -> None:
        focus = QApplication.focusWidget()
        editing_text = isinstance(
            focus,
            (QLineEdit, QTextEdit, QPlainTextEdit, QSpinBox, QDoubleSpinBox, QComboBox, QKeySequenceEdit),
        )
        has_modifier = bool(QApplication.keyboardModifiers().value & (
            Qt.KeyboardModifier.ControlModifier.value
            | Qt.KeyboardModifier.AltModifier.value
            | Qt.KeyboardModifier.MetaModifier.value
        ))
        if command == "focus_main_window":
            self._focus_main_window()
            return
        if command == "next_window":
            self._cycle_windows(1)
            return
        if command == "previous_window":
            self._cycle_windows(-1)
            return
        if command == "next_dataset":
            self._cycle_dataset(1)
            return
        if command == "previous_dataset":
            self._cycle_dataset(-1)
            return
        if command == "close_window":
            if not editing_text:
                self._close_current_child_window()
            return
        action = self._shortcut_actions.get(command)
        if action is None:
            return
        if editing_text and not has_modifier and not self._shortcut_allowed_in_editing(command):
            return
        if command == "filter":
            self._set_active_from_focused_dataset_window()
        if action.isEnabled():
            action.trigger()

    def _shortcut_text(self, key: str) -> str:
        return str(self._state.prefs.get("shortcuts", {}).get(key, "") or "")

    def _event_sequence_text(self, event) -> str:
        key = int(event.key())
        if key in (
            int(Qt.Key.Key_Shift), int(Qt.Key.Key_Control),
            int(Qt.Key.Key_Alt), int(Qt.Key.Key_Meta),
        ):
            return ""
        mods = event.modifiers().value & (
            Qt.KeyboardModifier.ShiftModifier.value
            | Qt.KeyboardModifier.ControlModifier.value
            | Qt.KeyboardModifier.AltModifier.value
            | Qt.KeyboardModifier.MetaModifier.value
        )
        return QKeySequence(mods | key).toString(QKeySequence.SequenceFormat.PortableText)

    def eventFilter(self, obj, event) -> bool:
        # Right-click on a ROI toolbar button → tool/variant switch menu.
        if (event.type() == QEvent.Type.MouseButtonPress
                and event.button() == Qt.MouseButton.RightButton
                and obj in getattr(self, "_roi_tool_buttons", {})):
            pos = event.position().toPoint() if hasattr(event, "position") else event.pos()
            if self._roi_tool_buttons[obj] == "line":
                self._show_line_family_menu(obj, pos)
            else:
                self._show_roi_tool_menu(obj, pos)
            return True
        if event.type() not in (QEvent.Type.ShortcutOverride, QEvent.Type.KeyPress):
            return super().eventFilter(obj, event)
        seq = self._event_sequence_text(event)
        if not seq:
            return super().eventFilter(obj, event)
        if event.type() == QEvent.Type.ShortcutOverride:
            command = self._shortcut_command_for_sequence(seq)
            if command is not None:
                focus = QApplication.focusWidget()
                editing_text = isinstance(
                    focus,
                    (QLineEdit, QTextEdit, QPlainTextEdit, QSpinBox, QDoubleSpinBox, QComboBox, QKeySequenceEdit),
                )
                has_modifier = bool(event.modifiers().value & (
                    Qt.KeyboardModifier.ControlModifier.value
                    | Qt.KeyboardModifier.AltModifier.value
                    | Qt.KeyboardModifier.MetaModifier.value
                ))
                if not (editing_text and not has_modifier) or self._shortcut_allowed_in_editing(command):
                    event.accept()
                    return True
            return super().eventFilter(obj, event)

        return self._handle_shortcut_keypress(seq, event)

    def _handle_shortcut_keypress(self, seq: str, event) -> bool:
        shortcuts = self._state.prefs.get("shortcuts", {})
        focus = QApplication.focusWidget()
        editing_text = isinstance(
            focus,
            (QLineEdit, QTextEdit, QPlainTextEdit, QSpinBox, QDoubleSpinBox, QComboBox, QKeySequenceEdit),
        )
        has_modifier = bool(event.modifiers().value & (
            Qt.KeyboardModifier.ControlModifier.value
            | Qt.KeyboardModifier.AltModifier.value
            | Qt.KeyboardModifier.MetaModifier.value
        ))
        if seq == shortcuts.get("next_window", ""):
            self._cycle_windows(1)
            return True
        if seq == shortcuts.get("focus_main_window", ""):
            self._focus_main_window()
            return True
        if seq == shortcuts.get("previous_window", ""):
            self._cycle_windows(-1)
            return True
        if seq == shortcuts.get("next_dataset", ""):
            self._cycle_dataset(1)
            return True
        if seq == shortcuts.get("previous_dataset", ""):
            self._cycle_dataset(-1)
            return True
        if seq == shortcuts.get("close_window", ""):
            if editing_text:
                return False
            self._close_current_child_window()
            return True
        for command, action in self._shortcut_actions.items():
            if seq and seq == shortcuts.get(command, ""):
                if editing_text and not has_modifier and not self._shortcut_allowed_in_editing(command):
                    return False
                if command == "filter":
                    self._set_active_from_focused_dataset_window()
                if action.isEnabled():
                    action.trigger()
                    return True
        return False

    def _shortcut_command_for_sequence(self, seq: str) -> str | None:
        shortcuts = self._state.prefs.get("shortcuts", {})
        for command in (
            "focus_main_window", "next_window", "previous_window",
            "next_dataset", "previous_dataset", "close_window",
        ):
            if seq and seq == shortcuts.get(command, ""):
                return command
        for command in self._shortcut_actions:
            if seq and seq == shortcuts.get(command, ""):
                return command
        return None

    def _shortcut_conflicts(self) -> dict[str, list[str]]:
        shortcuts = self._state.prefs.get("shortcuts", {})
        labels = {
            "focus_main_window": "Focus main window",
            "next_window": "Next window",
            "previous_window": "Previous window",
            "next_dataset": "Next dataset",
            "previous_dataset": "Previous dataset",
            "close_window": "Close current window",
            **{key: action.text().replace("&", "") for key, action in self._shortcut_actions.items()},
        }
        by_seq: dict[str, list[str]] = {}
        for key, label in labels.items():
            seq = str(shortcuts.get(key, "") or "")
            if not seq:
                continue
            by_seq.setdefault(seq, []).append(label)
        return {seq: commands for seq, commands in by_seq.items() if len(commands) > 1}

    def _shortcut_allowed_in_editing(self, command: str) -> bool:
        active = QApplication.activeWindow()
        tag = getattr(active, "TAG", None)
        if command == "filter":
            return tag in {"histogram_window", "attribute_window"}
        if command == "brightness_contrast":
            return tag == "render_window"
        return False

    def _set_active_from_focused_dataset_window(self) -> None:
        """Use the focused dataset-owned plot/view when a shortcut targets data."""
        focus = QApplication.focusWidget()
        candidates: list[QWidget | None] = [QApplication.activeWindow(), focus]
        if focus is not None:
            parent = focus.parentWidget()
            while parent is not None:
                candidates.append(parent)
                parent = parent.parentWidget()
        for widget in candidates:
            idx = getattr(widget, "dataset_idx", None)
            if idx is None:
                idx = getattr(widget, "_dataset_idx", None)
            if idx is None:
                idx = getattr(widget, "_idx", None)
            if type(idx) is int and 0 <= idx < len(self._state.datasets):
                self._state.set_active(idx)
                return



    # ------------------------------------------------------------------
    # File loading
    # ------------------------------------------------------------------

    def _open_dialog(self) -> None:
        default = self._state.prefs["file"].get("default_folder", str(Path.home()))
        # Open… is for recognized MINFLUX data formats. Spreadsheets and TIFFs
        # have their own File-menu entries; drag-drop still accepts everything.
        paths, _ = QFileDialog.getOpenFileNames(
            self,
            "Open MINFLUX data",
            default,
            "MINFLUX data (*.mat *.npy *.msr *.json);;"
            "MATLAB (*.mat);;"
            "NumPy (*.npy);;"
            "Imspector .msr (*.msr);;"
            "MINFLUX JSON (*.json);;"
            "All files (*)",
        )
        for p in paths:
            self._route_file(p)

    def _open_msr_dialog(self, msr_path: str) -> None:
        """Hand an ``.msr`` file to the MSR Reader plugin (drag-drop entry point)."""
        if not msr_path:
            return
        from .msr_import_dialog import open_msr
        open_msr(msr_path, self._state, parent=self)

    def _open_spreadsheet_dialog(self) -> None:
        try:
            from .spreadsheet_import_dialog import choose_spreadsheet_and_import
            dataset = choose_spreadsheet_and_import(self)
            if dataset is not None:
                self._state.add_dataset(dataset)
        except Exception as exc:
            self._state.log(f"Failed to open spreadsheet data: {exc}", "ERROR")
            QMessageBox.critical(self, "Open spreadsheet data", str(exc))

    def _open_tiff_dialog(self) -> None:
        default = self._state.prefs["file"].get("default_folder", str(Path.home()))
        path, _ = QFileDialog.getOpenFileName(
            self,
            "Open TIFF image",
            default,
            "TIFF image (*.tif *.tiff);;All files (*)",
        )
        if path:
            self._load_tiff(path)

    def _route_file(self, path: str) -> None:
        """
        Route a single file path to the correct loader based on extension.
        Unsupported types are reported to the log.
        """
        p   = Path(path)
        ext = p.suffix.lower()

        # ImageJ ROI / RoiSet files load into the ROI Manager, not as datasets.
        if ext in _ROI_FILE_EXTS:
            self._load_roi_json(path)
            return

        from ..core.format_sniff import resolve_format
        fmt, note = resolve_format(path)
        if note:
            self._state.log(f"'{p.name}': {note}.", "WARN")

        loader = _FMT_LOADERS.get(fmt) if fmt else None
        if loader is None:
            msg = (
                f"Unsupported file type: '{p.name}'  "
                f"(extension '{ext or '(none)'}' unrecognised and content could "
                f"not be identified).  Supported: .mat, .npy, .csv, .tsv, .txt, "
                f".xlsx, .xlsm, .msr, .tif, .tiff, .json"
            )
            self._state.log(msg, "WARN")
            self._status_label.setText(f"Skipped: {p.name} (unsupported type)")
            return
        getattr(self, loader)(path)

    def _route_path(self, path: str) -> None:
        """
        Route a path that may be a file OR a directory.
        Directories are scanned for all supported files (non-recursive).
        """
        p = Path(path)
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
                for f in found:
                    self._route_file(str(f))
            else:
                msg = (
                    f"Folder '{p.name}' contains no supported files "
                    f"(.mat, .npy, .csv, .tsv, .xlsx, .xlsm, .msr, .tif, .tiff, .json)."
                )
                self._state.log(msg, "WARN")
                self._status_label.setText(f"No supported files in folder: {p.name}")
        else:
            self._route_file(path)

    # -- individual loaders ----------------------------------------

    def _load_mat(self, path: str) -> None:
        self._status_label.setText(f"Loading {Path(path).name}…")
        self._state.log(f"Opening .mat: {path}", "INFO")
        try:
            from ..core.loader import load_dataset
            dataset = load_dataset(path, prefs=self._state.prefs)
            self._state.add_dataset(dataset)
        except Exception as exc:
            self._state.log(f"Failed to load '{Path(path).name}': {exc}", "ERROR")
            QMessageBox.critical(self, "Load error", str(exc))
            self._status_label.setText("Load failed.")

    def _load_npy(self, path: str) -> None:
        self._status_label.setText(f"Loading {Path(path).name}…")
        self._state.log(f"Opening .npy: {path}", "INFO")
        try:
            from ..core.loader import load_npy
            dataset = load_npy(path, prefs=self._state.prefs)
            self._state.add_dataset(dataset)
        except Exception as exc:
            self._state.log(f"Failed to load '{Path(path).name}': {exc}", "ERROR")
            QMessageBox.critical(self, "Load error", str(exc))
            self._status_label.setText("Load failed.")

    def _load_csv(self, path: str) -> None:
        # CSV/TSV/TXT all go through the smart spreadsheet importer.
        self._load_spreadsheet(path)

    def _load_spreadsheet(self, path: str) -> None:
        self._status_label.setText(f"Loading {Path(path).name}…")
        self._state.log(f"Opening spreadsheet: {path}", "INFO")
        try:
            from ..core.spreadsheet_loader import auto_import
            dataset, ambiguity = auto_import(path, prefs=self._state.prefs)
            if dataset is not None:
                self._state.log(
                    f"Auto-mapped '{Path(path).name}' "
                    f"({dataset.prop.num_loc:,} localizations).", "INFO")
                self._state.add_dataset(dataset)
                return
            # Ambiguous columns → let the user map them manually.
            self._state.log(
                f"'{Path(path).name}': {ambiguity.reason} Opening the column "
                "mapping dialog.", "INFO")
            from .spreadsheet_import_dialog import open_mapping_dialog
            dataset = open_mapping_dialog(ambiguity, parent=self)
            if dataset is None:
                self._status_label.setText("Spreadsheet import cancelled.")
                return
            self._state.add_dataset(dataset)
        except Exception as exc:
            self._state.log(f"Failed to load '{Path(path).name}': {exc}", "ERROR")
            QMessageBox.critical(self, "Load error", str(exc))
            self._status_label.setText("Load failed.")

    def _load_tiff(self, path: str) -> None:
        self._status_label.setText(f"Opening TIFF {Path(path).name}…")
        self._state.log(f"Opening TIFF: {path}", "INFO")
        key = str(Path(path).resolve())
        existing = self._tiff_windows.get(key)
        if existing is not None:
            try:
                existing.show()
                existing.raise_()
                existing.activateWindow()
                self._status_label.setText(f"TIFF already open: {Path(path).name}")
                return
            except RuntimeError:
                self._tiff_windows.pop(key, None)
        try:
            from ..core.tiff_source import TiffImageSource
            from .tiff_viewer_window import TiffViewerWindow

            source = TiffImageSource(path)
            win = TiffViewerWindow(source)
            win.destroyed.connect(lambda _=None, k=key: self._tiff_windows.pop(k, None))
            self._tiff_windows[key] = win
            win.show()
            win.raise_()
            win.activateWindow()
            try:
                self._state._record_recent(key)
                self._populate_recent_menu()
            except Exception:
                pass
            meta = source.metadata
            self._status_label.setText(f"TIFF opened: {Path(path).name}")
            self._state.log(
                f"Opened TIFF image viewer: {Path(path).name}  |  "
                f"axes={meta.axes}  |  shape={meta.shape}  |  dtype={meta.dtype}",
                "INFO",
            )
        except Exception as exc:
            self._state.log(f"Failed to load '{Path(path).name}': {exc}", "ERROR")
            QMessageBox.critical(self, "Load error", str(exc))
            self._status_label.setText("Load failed.")

    def _load_json(self, path: str) -> None:
        self._status_label.setText(f"Loading {Path(path).name}…")
        self._state.log(f"Opening JSON: {path}", "INFO")
        # A native ROI-set JSON (a dict with a "rois" list) loads ROIs, not data.
        try:
            from ..core.roi import is_roi_json_file
            if is_roi_json_file(path):
                self._load_roi_json(path)
                return
        except Exception:
            pass
        try:
            from ..core.loader import load_json
            dataset = load_json(path, prefs=self._state.prefs)
            self._state.add_dataset(dataset)
            return
        except Exception as data_exc:
            try:
                from ..core.filter_io import is_filter_json_file
                if is_filter_json_file(path):
                    self._load_filter_json(path)
                    return
            except Exception:
                pass
            try:
                from ..core.save import is_metadata_json_file
                if is_metadata_json_file(path):
                    self._state.log(
                        f"'{Path(path).name}' is a metadata sidecar, not loadable data.",
                        "WARN",
                    )
                    QMessageBox.information(
                        self, "Metadata file",
                        "This is a MINFLUX-viewer metadata sidecar (RIMF, transform, "
                        "filter specs). It documents an exported dataset and is not "
                        "itself loadable as data.",
                    )
                    self._status_label.setText("Ready.")
                    return
            except Exception:
                pass
            self._state.log(f"Failed to load JSON '{Path(path).name}': {data_exc}", "ERROR")
            QMessageBox.critical(self, "Load error", str(data_exc))
            self._status_label.setText("Load failed.")

    def _load_roi_json(self, path: str) -> None:
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
        for r in records:
            ctx = dict(r.context) if isinstance(r.context, dict) else {}
            if isinstance(idx, int):
                ctx["dataset_idx"] = idx
            r.context = ctx
            r.selection_dirty = True
        self._state.rois.set_show_all(True)
        if records:
            self._state.rois.select([r.id for r in records])
        self._show_roi_manager()
        self._state.log(f"Loaded {len(records)} ROI(s) from {Path(path).name}.")
        self._status_label.setText("Ready.")

    def _populate_recent_menu(self) -> None:
        self._recent_menu.clear()
        recent = self._state.prefs["file"].get("recent_files", [])
        limit = int(self._state.prefs["file"].get("num_file_history", 5) or 5)
        # The store keeps up to MAX_RECENT_REMEMBERED; show only the user's
        # configured count, skipping entries that no longer exist on disk (kept
        # in the store in case the file reappears, e.g. a remounted drive).
        shown: list[str] = []
        for path in recent:
            try:
                if Path(path).is_file():
                    shown.append(path)
            except (TypeError, ValueError):
                continue
            if len(shown) >= limit:
                break
        if not shown:
            a = self._recent_menu.addAction("(none)")
            a.setEnabled(False)
            return
        for path in shown:
            act = QAction(path, self)
            act.triggered.connect(lambda checked, p=path: self._route_file(p))
            self._recent_menu.addAction(act)

    def _populate_plugins_menu(self) -> None:
        """
        Fill the Plugins menu from the plugin registry.

        Built-in plugins register themselves at import time; third-party
        plugins that have been imported before this method runs appear here
        too, in insertion order.
        """
        from .. import plugins

        plugins.ensure_loaded()
        menu = self._ui.menuPlugins
        menu.clear()
        self._install_ai_menu_style(menu)

        entries = plugins.available()
        if not entries:
            placeholder = menu.addAction("(no plugins registered)")
            placeholder.setEnabled(False)
            return

        for entry in entries:
            act = QAction(entry.name, self)
            if entry.tooltip:
                act.setToolTip(entry.tooltip)
                act.setStatusTip(entry.tooltip)
            # Capture `entry` per-iteration with a default argument so the
            # lambda doesn't all point at the last one.
            act.triggered.connect(
                lambda _=False, e=entry: e.launch(self._state, self)
            )
            if entry.name == "Generate Method Text":
                self._mark_action_ai_unapproved(act)
            menu.addAction(act)

    def _bind_scripting_api(self) -> None:
        """Expose this viewer instance through the runtime ``mfv`` module."""
        try:
            from ..scripting import install_runtime_module

            self._state.mfv.bind_main_window(self)
            install_runtime_module(self._state.mfv)
        except Exception as exc:
            self._state.log(f"Could not initialize scripting API: {exc}", "WARN")

    # ------------------------------------------------------------------
    # Tool windows  — open once, raise if already open
    # ------------------------------------------------------------------

    def _show_script_editor(self) -> None:
        from .script_editor_window import ScriptEditorWindow

        if self._script_editor_win is None:
            self._script_editor_win = ScriptEditorWindow(self._state, parent=self)
        self._script_editor_win.show()
        self._script_editor_win.raise_()
        self._script_editor_win.activateWindow()

    def _show_scatter(self, dataset_idx: int | None = None):
        if self._state.active_dataset is None:
            self._no_data_warning(); return
        from .scatter_window import ScatterWindow
        idx = dataset_idx if type(dataset_idx) is int else self._state.active_idx
        if idx is None:
            return None
        win = self._scatter_windows.get(idx)
        if win is None:
            win = ScatterWindow(self._state, dataset_idx=idx)
            win.destroyed.connect(lambda _=None, i=idx: self._scatter_windows.pop(i, None))
            self._scatter_windows[idx] = win
        self._scatter_win = win
        self._install_window_shortcuts(win)
        win.show(); win.raise_(); win.activateWindow()
        self._notify_view_state_changed()
        return win

    def _show_histogram(self, dataset_idx: int | None = None):
        if self._state.active_dataset is None:
            self._no_data_warning(); return
        from .histogram_window import HistogramWindow
        idx = dataset_idx if type(dataset_idx) is int else self._state.active_idx
        if idx is None:
            return None
        win = self._histogram_windows.get(idx)
        if win is None:
            win = HistogramWindow(self._state, dataset_idx=idx)
            win.destroyed.connect(lambda _=None, i=idx: self._histogram_windows.pop(i, None))
            self._histogram_windows[idx] = win
        self._histogram_win = win
        self._install_window_shortcuts(win)
        win.show(); win.raise_(); win.activateWindow()
        self._notify_view_state_changed()
        return win

    def _show_attr_plot(self, dataset_idx: int | None = None):
        if self._state.active_dataset is None:
            self._no_data_warning(); return
        from .attribute_window import AttributeWindow
        idx = dataset_idx if type(dataset_idx) is int else self._state.active_idx
        if idx is None:
            return None
        win = self._attr_windows.get(idx)
        if win is None:
            win = AttributeWindow(self._state, dataset_idx=idx)
            win.destroyed.connect(lambda _=None, i=idx: self._attr_windows.pop(i, None))
            self._attr_windows[idx] = win
        self._attr_win = win
        self._install_window_shortcuts(win)
        win.show(); win.raise_(); win.activateWindow()
        self._notify_view_state_changed()
        return win

    def _show_attr_plot_3d_matplotlib(self) -> None:
        if self._state.active_dataset is None:
            self._no_data_warning(); return
        from .attribute_3d_matplotlib_window import Attribute3DMatplotlibWindow
        self._attr_3d_mpl_win = _raise_or_create(
            self._attr_3d_mpl_win,
            Attribute3DMatplotlibWindow,
            self._state,
        )
        self._install_window_shortcuts(self._attr_3d_mpl_win)

    def _show_filter(self) -> None:
        from .filter_dialog import FilterDialog
        idx = self._state.active_idx
        win = self._filter_dlgs.get(idx)
        if win is not None:
            try:
                if not win.isHidden():
                    win.show()
                    win.raise_()
                    win.activateWindow()
                    self._filter_dlg = win
                    self._notify_view_state_changed()
                    return win
            except RuntimeError:
                self._filter_dlgs.pop(idx, None)
                win = None
        win = FilterDialog(self._state, parent=self, dataset_idx=idx)
        win.destroyed.connect(lambda _=None, key=idx: self._filter_dlgs.pop(key, None))
        self._filter_dlgs[idx] = win
        self._filter_dlg = win
        win.show()
        win.raise_()
        win.activateWindow()
        self._notify_view_state_changed()
        return win

    def _load_filter_json(self, path: str) -> None:
        """Open (or raise) the filter dialog and append filters from a JSON file."""
        self._show_filter()
        self._filter_dlg.load_filter_json(path)

    def _show_dataset_manager(self) -> None:
        from .dataset_manager import DatasetManager
        if self._ds_manager is None:
            self._ds_manager = DatasetManager(self._state, parent=self)
            self._ds_manager.destroyed.connect(lambda _=None: setattr(self, "_ds_manager", None))
        self._ds_manager.show()
        self._ds_manager.raise_()
        self._ds_manager.activateWindow()
        self._notify_view_state_changed()

    def _show_roi_manager(self) -> None:
        from .roi_manager import RoiManagerWindow
        self._roi_manager_win = _raise_or_create(self._roi_manager_win, RoiManagerWindow, self._state)

    # ------------------------------------------------------------------
    # ROI convert / enlarge-shrink / skeletonize (Process › ROI)
    # ------------------------------------------------------------------

    def _active_roi(self):
        """The active ROI as ``(record, kind, adapter)``: the single selected
        stored ROI, else the active overlay's draft. ``kind`` is
        ``"stored"`` / ``"draft"`` / ``None``."""
        store = self._state.rois
        selected = store.selected_records()
        if len(selected) == 1:
            return selected[0], "stored", store.active_adapter
        adapter = store.active_adapter
        if adapter is not None:
            try:
                draft = adapter.current_record()
            except Exception:
                draft = None
            if draft is not None:
                return draft, "draft", adapter
        return None, None, None

    def _clear_roi_cached_mask(self, record) -> None:
        """Drop a stored ROI's cached selection mask (state + derived) so a
        converted/edited ROI doesn't leave a stale highlight behind."""
        from ..core.roi_selection import ROI_MASKS_STATE_KEY

        idx = record.context.get("dataset_idx") if isinstance(record.context, dict) else None
        datasets = self._state.datasets
        ds = datasets[idx] if isinstance(idx, int) and 0 <= idx < len(datasets) else None
        if ds is None:
            ds = self._state.active_dataset
        if ds is None:
            return
        masks = ds.state.get(ROI_MASKS_STATE_KEY, {})
        meta = masks.pop(record.id, None)
        key = getattr(record, "mask_key", "") or (meta or {}).get("key")
        if key:
            ds.derived.pop(key, None)

    def _commit_roi_replacement(self, record, kind, adapter, new_record) -> None:
        """Replace a stored ROI in place; a draft stays a draft of the new type
        in the view (not filed into the Manager)."""
        store = self._state.rois
        if kind == "stored":
            self._clear_roi_cached_mask(record)  # drop the old shape's stale highlight
            new_record.name = (
                f"{new_record.type}-{store.next_type_index(new_record.type)}"
                if store._is_auto_name(record) else record.name)
            store.update(record.id, new_record)
            store.select([record.id])
        else:  # draft — keep it editable in the view, do not open/file the Manager
            replace_draft = getattr(adapter, "replace_draft", None)
            if callable(replace_draft):
                replace_draft(new_record)
            else:
                if adapter is not None:
                    try:
                        adapter.consume_draft()
                    except Exception:
                        pass
                store.add(new_record)
        idx = record.context.get("dataset_idx") if isinstance(record.context, dict) else None
        self._state.notify_roi_selection_changed(idx if isinstance(idx, int) else None)

    def _convert_active_roi(self, target: str) -> None:
        from ..core.roi_convert import available_conversions, convert_roi

        record, kind, adapter = self._active_roi()
        if record is None:
            QMessageBox.information(self, "Convert ROI", "Select an ROI (or draw one) first.")
            return
        if target not in available_conversions(record):
            QMessageBox.information(
                self, "Convert ROI",
                f"Cannot convert a {record.type} ROI ('{record.name}') to {target}.")
            return
        width = height = None
        from .roi_convert_dialog import RoiSizeDialog
        if record.type == "point" and target in {"rectangle", "oval"}:
            dlg = RoiSizeDialog(self, title=f"Point → {target}", need_height=True)
            if dlg.exec() != QDialog.DialogCode.Accepted:
                return
            width, height = dlg.values()
        elif target == "region":
            dlg = RoiSizeDialog(self, title="Line → region", need_height=False, width_label="Width")
            if dlg.exec() != QDialog.DialogCode.Accepted:
                return
            width, _ = dlg.values()
        try:
            new_record = convert_roi(record, target, width=width, height=height)
        except Exception as exc:
            QMessageBox.warning(self, "Convert ROI", str(exc))
            return
        old_type = record.type
        self._commit_roi_replacement(record, kind, adapter, new_record)
        self._state.log(f"Converted ROI '{record.name}' ({old_type}) → {new_record.type}.")

    def _resize_active_roi(self) -> None:
        from ..core.roi_convert import LINE_TYPES, can_resize, enlarge_shrink_roi
        from .roi_convert_dialog import RoiResizeDialog

        record, kind, adapter = self._active_roi()
        if record is None:
            QMessageBox.information(self, "Enlarge / Shrink ROI", "Select an ROI (or draw one) first.")
            return
        if not can_resize(record):
            QMessageBox.information(self, "Enlarge / Shrink ROI", "Angle ROIs cannot be enlarged or shrunk.")
            return
        allow_shrink = record.type not in ({"point"} | LINE_TYPES)
        dlg = RoiResizeDialog(self, allow_shrink=allow_shrink)
        if dlg.exec() != QDialog.DialogCode.Accepted:
            return
        mode, value = dlg.values()
        try:
            new_record = enlarge_shrink_roi(record, value, mode=mode)
        except Exception as exc:
            QMessageBox.warning(self, "Enlarge / Shrink ROI", str(exc))
            return
        old_type = record.type
        self._commit_roi_replacement(record, kind, adapter, new_record)
        self._state.log(
            f"{mode.capitalize()}d ROI '{record.name}' ({old_type}) by {value:g} nm → {new_record.type}.")

    def _skeletonize_active_roi(self) -> None:
        from ..core.roi_convert import REGION_TYPES, skeletonize_roi

        record, kind, adapter = self._active_roi()
        if record is None:
            QMessageBox.information(self, "Skeletonize ROI", "Select a region ROI (or draw one) first.")
            return
        if record.type not in REGION_TYPES:
            QMessageBox.information(
                self, "Skeletonize ROI",
                "Skeletonize applies to region ROIs (rectangle / oval / polygon / freehand).")
            return
        try:
            new_record = skeletonize_roi(record)
        except Exception as exc:
            QMessageBox.warning(self, "Skeletonize ROI", str(exc))
            return
        old_type = record.type
        self._commit_roi_replacement(record, kind, adapter, new_record)
        self._state.log(f"Skeletonized ROI '{record.name}' ({old_type}) → line.")

    def _convex_hull_active_roi(self) -> None:
        """Process › ROI › Convex Hull — convert the active ROI to its convex
        hull when that conversion is available (polygon / freehand). Any other
        case (incompatible type, no active ROI, no active dataset) is ignored."""
        from ..core.roi_convert import available_conversions, convert_roi

        record, kind, adapter = self._active_roi()
        if record is None or "convex_hull" not in available_conversions(record):
            return  # silently ignore
        try:
            new_record = convert_roi(record, "convex_hull")
        except Exception:
            return
        old_type = record.type
        self._commit_roi_replacement(record, kind, adapter, new_record)
        self._state.log(f"Converted ROI '{record.name}' ({old_type}) → convex hull.")

    def _run_hlyb_pair_analysis(self, mode: str = "3D") -> None:
        """Analyze › Clustering › HlyB subunit pair analysis › 2D/3D/template — detect
        protein sub-units from traces, cluster them into HlyB structures and
        report the sub-unit pair distances in a scatter + histogram window. The
        2-D path first builds a per-E.coli mask and drops traces in the eroded
        border margin (where 2-D distances are unreliable). The template path
        accepts partial 3-D matches to the six-site HlyB distance model."""
        import numpy as np
        from ..core.loader import mfx_get

        mode_text = str(mode).upper()
        if "TEMPLATE" in mode_text:
            mode = "TEMPLATE3D"
        else:
            mode = "2D" if mode_text == "2D" else "3D"
        idx = self._state.active_idx
        if idx is None or self._state.active_dataset is None:
            self._no_data_warning()
            return
        ds = self._state.datasets[idx]

        def _col(attr):
            v = mfx_get(ds, attr, itr="last", vld_only=True)
            return None if v is None else np.asarray(v, dtype=float).ravel()

        lx, ly, lz, tid = _col("loc_x"), _col("loc_y"), _col("loc_z"), _col("tid")
        if lx is None or ly is None or tid is None or lx.size < 3:
            QMessageBox.information(
                self, "HlyB Subunit Pair Analysis",
                "The active dataset has no localizations with trace IDs to analyze.")
            return
        if lz is None or lz.size != lx.size:
            lz = np.zeros_like(lx)
        loc_m = np.column_stack([lx, ly, lz])  # metres, raw z (z-scaling applied in analysis)

        from .hlyb_clustering_dialog import HlyBClusteringDialog, HlyBResultWindow
        from ..analysis.hlyb_clustering import (
            HlyBConfig,
            analyze_hlyb,
            analyze_hlyb_2d,
            analyze_hlyb_template3d,
        )

        defaults = getattr(self, "_hlyb_cfg", None)
        if defaults is None:
            defaults = HlyBConfig(z_scaling_factor=float(getattr(ds.cali, "RIMF", 0.67) or 0.67))
        dlg = HlyBClusteringDialog(self, defaults=defaults, mode=mode)
        if dlg.exec() != QDialog.DialogCode.Accepted:
            return
        cfg = dlg.config()
        self._hlyb_cfg = cfg

        QApplication.setOverrideCursor(Qt.CursorShape.WaitCursor)
        try:
            if mode == "2D":
                result = analyze_hlyb_2d(loc_m, tid, cfg)
            elif mode == "TEMPLATE3D":
                result = analyze_hlyb_template3d(loc_m, tid, cfg)
            else:
                result = analyze_hlyb(loc_m, tid, cfg)
        except Exception as exc:
            QApplication.restoreOverrideCursor()
            QMessageBox.warning(self, "HlyB Subunit Pair Analysis", f"Analysis failed: {exc}")
            return
        finally:
            if QApplication.overrideCursor() is not None:
                QApplication.restoreOverrideCursor()

        from .modeless import show_modeless
        title_mode = "Template matching (3D)" if mode == "TEMPLATE3D" else mode
        win = HlyBResultWindow(result, cfg, title=f"{ds.name} ({title_mode})", owner=self,
                               prefer_2d=(mode == "2D"))
        show_modeless(win, self)

        pd = result["all_pair_distances"]
        med = float(np.median(pd)) if pd.size else float("nan")
        if mode == "2D":
            prefix = (
                f"HlyB subunit pair analysis (2D) on '{ds.name}': "
                f"{result['n_total_traces']} trace(s), {result['n_border_traces']} border-excluded, "
                f"{result['n_traces']} interior → ")
            extra = (f"border {cfg.border_size_nm:g} nm, mask px {cfg.mask_pixel_size_nm:g} nm")
        elif mode == "TEMPLATE3D":
            qc = result.get("match_qc", {})
            prefix = (
                f"HlyB subunit pair analysis (template matching 3D) on '{ds.name}': "
                f"{result['n_traces']} trace(s) → ")
            extra = (
                f"z-scale {cfg.z_scaling_factor:g}, template tol {result['model_pair_tolerance_nm']:.1f} nm, "
                f"tested {qc.get('n_candidates_tested', 0)} candidate(s), "
                f"passed {qc.get('n_candidates_passed_thresholds', 0)}, "
                f"overlap-rejected {qc.get('n_overlap_rejected', 0)}"
            )
        else:
            prefix = f"HlyB subunit pair analysis (3D) on '{ds.name}': {result['n_traces']} trace(s) → "
            extra = f"z-scale {cfg.z_scaling_factor:g}"
        radius_label = (
            f"candidate edge {result['candidate_edge_radius_nm']:.1f} nm"
            if mode == "TEMPLATE3D" else f"HlyB radius {result['hlyb_diameter_nm']:.1f} nm"
        )
        self._state.log(
            prefix
            + f"{result['n_subunits']} subunit(s) → {result['n_structures']} HlyB structure(s); "
            f"{pd.size} pair(s), median distance {med:.2f} nm "
            f"(unit Ø {result['dunit_nm']:.1f} nm, {radius_label}, "
            f"min loc/trace {cfg.min_loc_per_trace}, {extra}).",
            dataset_idx=idx)

    def _segment_npc_2d(self) -> None:
        """Analyze › Segmentation › NPC › 2D — detect NPC centres by ring-kernel
        convolution and add a rectangle ROI around each into the ROI Manager."""
        import numpy as np

        idx = self._state.active_idx
        if idx is None or self._state.active_dataset is None:
            self._no_data_warning()
            return
        ds = self._state.datasets[idx]
        try:
            loc = np.asarray(ds.loc_nm, dtype=float)
        except Exception:
            loc = np.empty((0, 3))
        if loc.ndim != 2 or loc.shape[0] < 3 or loc.shape[1] < 2:
            QMessageBox.information(self, "NPC Segmentation", "The active dataset has no localizations to segment.")
            return
        mask = np.asarray(ds.filter_mask, dtype=bool)
        if mask.shape[0] == loc.shape[0]:
            loc = loc[mask]
        xy = loc[:, :2]

        from .npc_segmentation_dialog import NpcSegmentationDialog
        dlg = NpcSegmentationDialog(self, defaults=getattr(self, "_npc_seg_defaults", None))
        if dlg.exec() != QDialog.DialogCode.Accepted:
            return
        p = dlg.params()
        self._npc_seg_defaults = p

        from ..analysis.npc_segmentation import segment_npc_2d
        try:
            res = segment_npc_2d(
                xy, diameter_nm=p["diameter_nm"], rim_nm=p["rim_nm"],
                pixel_size_nm=p["pixel_size_nm"], min_support=p["min_support"])
        except Exception as exc:
            QMessageBox.warning(self, "NPC Segmentation", f"Segmentation failed: {exc}")
            return
        centers = res["centers"]
        if centers.shape[0] == 0:
            QMessageBox.information(
                self, "NPC Segmentation",
                "No NPCs detected with these parameters. Try adjusting the diameter, "
                "rim size, pixel size, or lowering the min support score.")
            return

        from ..core.roi import RoiRecord
        side = float(p["diameter_nm"])
        for i, (cx, cy) in enumerate(centers, start=1):
            rec = RoiRecord.create(
                "rectangle",
                {"bounds": [float(cx) - side / 2.0, float(cy) - side / 2.0, side, side]},
                name=f"NPC {i}", coordinate_space="plot", stroke_color="#ffcc00")
            rec.context = {"dataset_idx": idx, "source": "npc_segmentation_2d"}
            self._state.rois.add(rec)
        self._state.rois.set_show_all(True)     # reveal every detected NPC at once
        self._show_render(idx)
        self._show_roi_manager()
        self._state.log(
            f"NPC segmentation (2D): detected {centers.shape[0]} NPC(s) on '{ds.name}' "
            f"(diameter={p['diameter_nm']:.0f} nm, rim={p['rim_nm']:.0f} nm, "
            f"pixel={p['pixel_size_nm']:.1f} nm, min support={p['min_support']:.2f}); "
            f"added rectangle ROIs to the ROI Manager.")

    def _show_conv_segmentation(self) -> None:
        """Analyze › Segmentation › Convolution… — open the interactive
        geometry-kernel convolution segmentation tool for the active dataset."""
        idx = self._state.active_idx
        if idx is None or not (0 <= idx < len(self._state.datasets)):
            self._no_data_warning()
            return
        from .conv_segmentation_dialog import ConvSegmentationWindow
        from .modeless import show_modeless
        win = ConvSegmentationWindow(self._state, idx, owner=self)
        show_modeless(win, self)

    def _show_conv_segmentation_3d(self) -> None:
        """Analyze › Segmentation › Convolution (3D)… — open the interactive
        3-D geometry-kernel convolution segmentation tool for the active dataset.

        Requires a 3-D dataset; the data and kernel are convolved in full 3-D and
        viewed as an orthogonal (XY/XZ/YZ) slice viewer."""
        import numpy as np

        idx = self._state.active_idx
        if idx is None or not (0 <= idx < len(self._state.datasets)):
            self._no_data_warning()
            return
        ds = self._state.datasets[idx]
        try:
            loc = np.asarray(ds.loc_nm, dtype=float)
        except Exception:
            loc = np.empty((0, 2))
        if loc.ndim != 2 or loc.shape[1] < 3:
            QMessageBox.information(
                self, "Convolution Segmentation (3D)",
                "The active dataset is 2-D. Use 'Convolution…' for 2-D data; the 3-D "
                "tool needs localizations with a Z coordinate.")
            return
        from .conv_segmentation_3d_dialog import ConvSegmentation3DWindow
        from .modeless import show_modeless
        win = ConvSegmentation3DWindow(self._state, idx, owner=self)
        show_modeless(win, self)

    def _show_curvilinear_segmentation(self) -> None:
        """Analyze › Segmentation › Curvilinear Structures… — open the
        interactive Hessian/Frangi filament-tracing tool for the active dataset."""
        idx = self._state.active_idx
        if idx is None or not (0 <= idx < len(self._state.datasets)):
            self._no_data_warning()
            return
        from .curvilinear_segmentation_dialog import CurvilinearSegmentationWindow
        from .modeless import show_modeless
        win = CurvilinearSegmentationWindow(self._state, idx, owner=self)
        show_modeless(win, self)

    def add_segmentation_rois(self, idx: int, centers, *, side_nm: float,
                              name_prefix: str, source: str, stroke_color: str,
                              names: list[str] | None = None,
                              log_message: str | None = None) -> int:
        """Add a rectangle ROI of side ``side_nm`` centred on each ``(x, y)`` in
        *centers* to the ROI Manager, reveal them, and show render + manager.

        ``names`` (if given, one per centre) sets each ROI's name; otherwise they
        are auto-numbered ``"<name_prefix> <i>"``. Shared by the convolution
        segmentation tool; returns the count added."""
        import numpy as np

        from ..core.roi import RoiRecord

        centers = np.asarray(centers, dtype=float).reshape(-1, 2)
        if centers.shape[0] == 0 or not (0 <= idx < len(self._state.datasets)):
            return 0
        side = max(float(side_nm), 1.0)
        for i, (cx, cy) in enumerate(centers, start=1):
            name = names[i - 1] if names is not None and i - 1 < len(names) else f"{name_prefix} {i}"
            rec = RoiRecord.create(
                "rectangle",
                {"bounds": [float(cx) - side / 2.0, float(cy) - side / 2.0, side, side]},
                name=name, coordinate_space="plot",
                stroke_color=stroke_color)
            rec.context = {"dataset_idx": idx, "source": source}
            self._state.rois.add(rec)
        self._state.rois.set_show_all(True)
        self._show_render(idx)
        self._show_roi_manager()
        if log_message:
            self._state.log(log_message)
        return int(centers.shape[0])

    def add_point_rois_3d(self, idx: int, centers, *, name_prefix: str, source: str,
                          stroke_color: str, names: list[str] | None = None,
                          log_message: str | None = None) -> int:
        """Add a 3-D point ROI (full ``[x, y, z]`` nm geometry) for each centre in
        *centers* ``(K, 3)`` to the ROI Manager, reveal them, and show render +
        manager. Used by the 3-D convolution segmentation tool; returns the count
        added."""
        import numpy as np

        from ..core.roi import RoiRecord

        centers = np.asarray(centers, dtype=float).reshape(-1, 3)
        if centers.shape[0] == 0 or not (0 <= idx < len(self._state.datasets)):
            return 0
        for i, (cx, cy, cz) in enumerate(centers, start=1):
            name = names[i - 1] if names is not None and i - 1 < len(names) else f"{name_prefix} {i}"
            rec = RoiRecord.create(
                "point", {"point": [float(cx), float(cy), float(cz)]},
                name=name, coordinate_space="plot", stroke_color=stroke_color)
            rec.context = {"dataset_idx": idx, "source": source}
            self._state.rois.add(rec)
        self._state.rois.set_show_all(True)
        self._show_render(idx)
        self._show_roi_manager()
        if log_message:
            self._state.log(log_message)
        return int(centers.shape[0])

    def add_polyline_rois(self, idx: int, paths, *, name_prefix: str, source: str,
                          stroke_color: str, names: list[str] | None = None,
                          log_message: str | None = None) -> int:
        """Add an open poly-line (``freehand_line``) ROI tracing each path in
        *paths* (each an ``(M, 2)`` array of ``(x, y)`` nm vertices) to the ROI
        Manager, reveal them, and show render + manager. Returns the count added.

        Used by the curvilinear segmentation tool for traced centre lines."""
        import numpy as np

        from ..core.roi import RoiRecord

        if not (0 <= idx < len(self._state.datasets)):
            return 0
        added = 0
        for i, path in enumerate(paths, start=1):
            pts = np.asarray(path, dtype=float).reshape(-1, 2)
            if pts.shape[0] < 2:
                continue
            name = names[i - 1] if names is not None and i - 1 < len(names) else f"{name_prefix} {i}"
            rec = RoiRecord.create(
                "freehand_line",
                {"points": pts.tolist(), "closed": False},
                name=name, coordinate_space="plot", stroke_color=stroke_color)
            rec.context = {"dataset_idx": idx, "source": source}
            self._state.rois.add(rec)
            added += 1
        if added:
            self._state.rois.set_show_all(True)
            self._show_render(idx)
            self._show_roi_manager()
            if log_message:
                self._state.log(log_message)
        return added

    def add_particle_average_dataset(self, points_xyz_nm, *, name: str,
                                     log_message: str | None = None) -> int | None:
        """Build a dataset from an averaged particle's pooled localizations
        (``(N, 3)`` nm, already centred/aligned), register it, and open its render
        + scatter views (3-D-ready). Returns the new dataset index."""
        import uuid

        import numpy as np

        from ..core.dataset import build_localization_dataset

        pts = np.asarray(points_xyz_nm, dtype=float).reshape(-1, 3)
        pts = pts[np.all(np.isfinite(pts), axis=1)]
        if pts.shape[0] == 0:
            return None
        # Re-zero to the averaged/template frame: subtract the (robust) median so
        # the averaged particle is centred on the origin regardless of the source
        # coordinates. For the symmetric NPC ring the median is the ring centre,
        # so this is ~a no-op when the aligner already centred — and corrective if
        # any residual offset remains.
        pts = pts - np.median(pts, axis=0)
        ds = build_localization_dataset(
            name=name, x_nm=pts[:, 0], y_nm=pts[:, 1], z_nm=pts[:, 2],
            source_version="particle_average", prefs=self._state.prefs)
        # Unique synthetic path so repeated averages don't dedup onto each other.
        ds.file.folder = f"<particle-average>/{uuid.uuid4().hex}"
        ds.metadata["particle_average"] = True
        # Coordinates are final/aligned — pin RIMF to 1.0 and suppress auto-z.
        try:
            ds.set_rimf(1.0, source="2D (no z correction)")
            ds.derived["rimf"] = 1.0
        except Exception:
            pass
        idx = self._state.add_dataset(ds)
        if log_message:
            self._state.log(log_message)
        # Deselect the (source) detection ROIs: otherwise the selected box ROIs
        # would highlight in the averaged particle's fresh render/scatter at their
        # original — now meaningless — coordinates, which is confusing.
        try:
            self._state.rois.deselect()
        except Exception:
            pass
        # A freshly-appended dataset gets a new index, so its render/scatter are
        # created fresh (stale windows from closed datasets are handled by the
        # re-index path). Do NOT close/recreate the auto-opened render here — its
        # async tile workers would deliver into a deleted window and crash.
        self._show_render(idx)
        self._show_scatter(idx)
        return idx

    def _npc_detect_wanlu(self) -> None:
        """Analyze › Segmentation › NPC detection wanlu — ring-convolution NPC
        detection + sub-unit (trace) clustering; stores the per-NPC table on the
        dataset and adds NPC-centre ROIs to the Manager."""
        import numpy as np

        idx = self._state.active_idx
        if idx is None or self._state.active_dataset is None:
            self._no_data_warning()
            return
        ds = self._state.datasets[idx]
        from ..core.loader import attr_values_1d
        tid = attr_values_1d(ds, "tid")
        try:
            loc = np.asarray(ds.loc_nm, dtype=float)
        except Exception:
            loc = np.empty((0, 3))
        if loc.ndim != 2 or loc.shape[0] < 10 or tid is None:
            QMessageBox.information(self, "NPC detection wanlu",
                                    "Needs a MINFLUX dataset with trace ids and localizations.")
            return
        mask = np.asarray(ds.filter_mask, dtype=bool)
        tid = np.asarray(tid).ravel()
        if mask.shape[0] == loc.shape[0]:
            loc, tid = loc[mask], tid[mask]

        from .npc_wanlu_dialogs import NpcDetectWanluDialog
        dlg = NpcDetectWanluDialog(self, defaults=getattr(self, "_npc_wanlu_detect_defaults", None))
        if dlg.exec() != QDialog.DialogCode.Accepted:
            return
        p = dlg.params()
        self._npc_wanlu_detect_defaults = p

        from ..analysis import npc_wanlu as nw
        try:
            res = nw.detect_npc_wanlu(loc, tid, pixel_size=p["pixel_size"],
                                      diameter=p["diameter"], rim=p["rim"],
                                      min_units=p["min_units"], z_scale=p["z_scale"])
        except Exception as exc:
            QMessageBox.warning(self, "NPC detection wanlu", f"Detection failed: {exc}")
            return
        if res["n_npc"] == 0:
            QMessageBox.information(self, "NPC detection wanlu",
                                    "No NPCs accepted. Try adjusting diameter / rim / min sub-units.")
            return
        ds.derived["npc_wanlu"] = {"raw6": res["raw6"], "centers": res["centers"], "params": p}
        self.add_segmentation_rois(
            idx, res["centers"], side_nm=float(p["diameter"]),
            name_prefix="NPC", source="npc_wanlu_detection", stroke_color="#ff8c00",
            log_message=(f"NPC detection (Wanlu): {res['n_npc']} NPC(s), "
                         f"{res['n_units']} sub-unit(s) on '{ds.name}' "
                         f"(diameter={p['diameter']:.0f} nm, rim={p['rim']:.0f} nm, "
                         f"min units={p['min_units']}, Z scale={p['z_scale']:.2f}); "
                         f"stored {res['raw6'].shape[0]:,} localizations for averaging."))
        QMessageBox.information(
            self, "NPC detection wanlu",
            f"Detected {res['n_npc']} NPC(s) with {res['n_units']} sub-unit(s).\n\n"
            "Run 'NPC average wanlu' to build the two-ring average.")

    def _npc_average_wanlu(self) -> None:
        """Analyze › Segmentation › NPC average wanlu — two-ring averaging of the
        NPCs detected by 'NPC detection wanlu' on the active dataset."""
        import numpy as np

        idx = self._state.active_idx
        if idx is None or self._state.active_dataset is None:
            self._no_data_warning()
            return
        ds = self._state.datasets[idx]
        stored = ds.derived.get("npc_wanlu")
        if not stored or np.asarray(stored.get("raw6", [])).size == 0:
            QMessageBox.information(self, "NPC average wanlu",
                                    "No NPC detection found on this dataset. "
                                    "Run 'NPC detection wanlu' first.")
            return
        raw6 = np.asarray(stored["raw6"], dtype=float)
        n_npc = int(np.unique(raw6[:, 3]).size)

        from .npc_wanlu_dialogs import NpcAverageWanluDialog
        dlg = NpcAverageWanluDialog(self, defaults=getattr(self, "_npc_wanlu_avg_defaults", None),
                                    n_npc=n_npc)
        if dlg.exec() != QDialog.DialogCode.Accepted:
            return
        p = dlg.params()
        self._npc_wanlu_avg_defaults = {
            "diam_min": p["diameter_bounds"][0], "diam_max": p["diameter_bounds"][1],
            "inter_min": p["inter_ring_bounds"][0], "inter_max": p["inter_ring_bounds"][1],
            "inter_exp": p["expected_inter_ring"] or 0.0, "symmetry": p["symmetry"]}

        # Run the (heavy) two-ring averaging off the UI thread, with a refreshing
        # progress bar in the Log — same pattern as the Particle Average tool.
        if getattr(self, "_wanlu_avg_task", None) is not None:
            return                                           # already averaging
        from ..analysis import npc_wanlu as nw
        from ..core.app_state import format_progress_bar
        from .particle_average_dialog import _AverageTask

        ds_name = ds.name
        holder: dict = {}

        def compute(progress):
            res = nw.average_npc_wanlu(
                raw6, diameter_bounds=p["diameter_bounds"],
                inter_ring_bounds=p["inter_ring_bounds"],
                expected_inter_ring=p["expected_inter_ring"], symmetry=p["symmetry"],
                progress=progress)
            avg = res["average_loc"]
            zlo, zhi = res["z_peaks"]
            holder["name"] = f"{ds_name} — NPC avg [{res['n_accepted']}]"
            holder["log"] = (
                f"NPC average (Wanlu): pooled {res['n_accepted']}/{res['n_npc']} NPC(s) "
                f"({avg.shape[0]:,} localizations) from '{ds_name}'; "
                f"Z rings {zlo:.1f}/{zhi:.1f} nm (sep {zhi - zlo:.1f} nm"
                f"{', fallback' if res['z_fallback'] else ''}); 8-fold aligned.")
            return avg[:, :3], ""

        self._wanlu_avg_holder = holder
        self._wanlu_avg_last_pct = -1
        self._state.log(f"NPC average (Wanlu): averaging {n_npc} NPC(s)…")
        self._state.log_progress(format_progress_bar(0.0))
        task = _AverageTask(compute)
        task.signals.progress.connect(self._on_wanlu_avg_progress)
        task.signals.done.connect(self._on_wanlu_avg_done)
        task.signals.failed.connect(self._on_wanlu_avg_failed)
        self._wanlu_avg_task = task                          # keep a ref (auto-deletes)
        QThreadPool.globalInstance().start(task)

    def _on_wanlu_avg_progress(self, done: int, total: int) -> None:
        from ..core.app_state import format_progress_bar
        if total <= 0:
            return
        pct = int(done / total * 100)
        if pct != getattr(self, "_wanlu_avg_last_pct", -1):  # throttle to whole percent
            self._wanlu_avg_last_pct = pct
            self._state.log_progress(format_progress_bar(done / total))

    def _on_wanlu_avg_done(self, pts, _desc: str) -> None:
        import numpy as np

        from ..core.app_state import format_progress_bar
        self._wanlu_avg_task = None
        self._state.log_progress(format_progress_bar(1.0, done=True), final=True)
        pts = np.asarray(pts)
        holder = getattr(self, "_wanlu_avg_holder", {}) or {}
        if pts.ndim != 2 or pts.shape[0] == 0:
            QMessageBox.information(self, "NPC average wanlu",
                                    "No NPCs passed the two-ring fit. Try widening the "
                                    "diameter / inter-ring bounds.")
            return
        self.add_particle_average_dataset(
            pts, name=holder.get("name", "NPC avg"), log_message=holder.get("log"))

    def _on_wanlu_avg_failed(self, msg: str) -> None:
        self._wanlu_avg_task = None
        self._state.log_progress("=" * 10 + "  FAILED  " + "=" * 10, final=True)
        QMessageBox.warning(self, "NPC average wanlu", f"Averaging failed: {msg}")

    def _show_particle_average(self) -> None:
        """Analyze › Segmentation › Particle Average… — open the particle-averaging
        tool for the active dataset + ROI Manager box ROIs."""
        idx = self._state.active_idx
        if idx is None or not (0 <= idx < len(self._state.datasets)):
            self._no_data_warning()
            return
        from .modeless import show_modeless
        from .particle_average_dialog import ParticleAverageWindow
        win = ParticleAverageWindow(self._state, idx, owner=self)
        show_modeless(win, self)

    def _notify_view_state_changed(self) -> None:
        mgr = getattr(self, "_ds_manager", None)
        if mgr is not None and hasattr(mgr, "refresh_views"):
            try:
                mgr.refresh_views()
            except RuntimeError:
                self._ds_manager = None

    def dataset_view_status(self, idx: int) -> str:
        if not (0 <= idx < len(self._state.datasets)):
            return "None"
        ds = self._state.datasets[idx]
        group_id = ds.state.get("overlay_id") or ds.state.get("render_group_id")
        if group_id:
            overlay_idx = ds.state.get("overlay_index")
            visible_overlay = False
            for win in list(self._render_windows.values()) + list(self._scatter_windows.values()):
                try:
                    channels = getattr(win, "_channels", None)
                    if channels and any(ch.get("dataset_idx") == idx and ch.get("visible", True) for ch in channels):
                        visible_overlay = True
                        break
                except RuntimeError:
                    continue
            if visible_overlay or any(
                getattr(other, "state", {}).get("overlay_id") == group_id
                for other in self._state.datasets
            ):
                return f"Overlay {overlay_idx}" if overlay_idx else "Overlay"
        own_maps = (self._render_windows, self._scatter_windows, self._histogram_windows, self._attr_windows, self._filter_dlgs)
        for mapping in own_maps:
            win = mapping.get(idx)
            if win is not None:
                try:
                    if win.isVisible():
                        return "Own"
                except RuntimeError:
                    pass
        return "None"

    def _on_roi_tool(self, tool: str, checked: bool) -> None:
        if checked:
            self._activate_roi_tool(tool)
        elif self._state.rois.active_tool == tool:
            self._state.rois.set_tool(None)

    def _activate_roi_tool(self, tool: str) -> None:
        self._state.rois.set_tool(tool)
        self._sync_roi_tool_actions(tool)
        if self._state.rois.active_adapter is None:
            self._state.log("Select a render, histogram, scatter, or attribute plot window before drawing ROIs.", "WARN")

    def _sync_roi_tool_actions(self, tool: str = "") -> None:
        active_tool = tool or self._state.rois.active_tool or ""
        family = {t for _l, t, _ic in _LINE_FAMILY}
        if active_tool in family:
            self._line_variant = active_tool
            self._update_line_button_icon()
        for name, action in self._roi_tool_actions.items():
            blocked = action.blockSignals(True)
            # The single Line button stays checked for any of its family members.
            action.setChecked(active_tool in family if name == "line" else name == active_tool)
            action.blockSignals(blocked)

    def _update_line_button_icon(self) -> None:
        from .. import resource_path
        icon_file = {t: ic for _l, t, ic in _LINE_FAMILY}.get(self._line_variant, "line.png")
        path = resource_path("icons", icon_file)
        if path.exists():
            self._ui.toolLine.setIcon(QIcon(str(path)))
            self._ui.toolLine.setToolTip(
                {t: lbl for lbl, t, _ic in _LINE_FAMILY}.get(self._line_variant, "Line")
                + " — right-click to switch (straight / poly / freehand line)")

    def _install_roi_tool_menus(self) -> None:
        # Only the Line button gets a right-click menu — the Fiji-style switch
        # between straight / poly / freehand line. The other ROI buttons have no
        # right-click menu. A QToolButton in a QToolBar swallows
        # ``customContextMenuRequested``, so we intercept the right mouse press
        # with an event filter instead (reliable).
        self._roi_tool_buttons: dict = {}
        line_action = getattr(self._ui, "toolLine", None)
        button = self._ui.toolbar.widgetForAction(line_action) if line_action is not None else None
        if button is not None:
            self._roi_tool_buttons[button] = "line"
            button.installEventFilter(self)

    def _show_line_family_menu(self, widget: QWidget, pos) -> None:
        """Fiji-style right-click switcher on the Line button."""
        menu = QMenu(widget)
        active = self._state.rois.active_tool
        for label, tool, _ic in _LINE_FAMILY:
            action = menu.addAction(label)
            action.setCheckable(True)
            action.setChecked(tool == active or (active is None and tool == self._line_variant))
            action.triggered.connect(lambda _checked=False, t=tool: self._select_line_variant(t))
        menu.exec(widget.mapToGlobal(pos))

    def _select_line_variant(self, tool: str) -> None:
        self._line_variant = tool
        self._update_line_button_icon()
        self._activate_roi_tool(tool)

    def _show_roi_tool_menu(self, widget: QWidget, pos) -> None:
        menu = QMenu(widget)
        active_tool = self._state.rois.active_tool
        for label, tool, _attr in _ROI_TOOL_DEFS:
            action = menu.addAction(label)
            action.setCheckable(True)
            action.setChecked(tool == active_tool)
            action.triggered.connect(lambda _checked=False, t=tool: self._activate_roi_tool(t))
        menu.addSeparator()
        clear_action = menu.addAction("No ROI tool")
        clear_action.triggered.connect(lambda: self._state.rois.set_tool(None))
        menu.exec(widget.mapToGlobal(pos))

    def _show_log(self) -> None:
        """Open (or raise) the Log window — structured application events."""
        self._ensure_log_window(show=True, raise_window=True)

    def _ensure_log_window(self, *, show: bool = True, raise_window: bool = False):
        from .log_window import LogWindow
        if self._log_win is None or not self._log_win.isVisible():
            if self._log_win is None:
                self._log_win = LogWindow(self._state)
            if show:
                self._log_win.show()
        elif show and not self._log_win.isVisible():
            self._log_win.show()
        if raise_window and self._log_win is not None:
            self._log_win.raise_()
            self._log_win.activateWindow()
        return self._log_win

    def _on_log_message(self, message: str, level: str) -> None:
        was_missing = self._log_win is None
        win = self._ensure_log_window(show=True, raise_window=False)
        if was_missing and win is not None:
            try:
                win._append(message, level)
            except Exception:
                pass

    def _show_console(self) -> None:
        """Open (or raise) the Console window — raw stdout / stderr stream."""
        from .console_window import ConsoleWindow
        if self._console_win is None or not self._console_win.isVisible():
            if self._console_win is None:
                self._console_win = ConsoleWindow()
            self._console_win.show()
        self._console_win.raise_()
        self._console_win.activateWindow()

    def _show_brightness_contrast(self) -> None:
        """
        Forward to the render window of the currently active dataset.
        The active dataset follows whichever render window is focused (#2).
        """
        if self._state.active_dataset is None:
            self._no_data_warning()
            return
        idx = self._state.active_idx
        rwin = self._render_windows.get(idx)
        if rwin is None:
            # Ensure a render window exists, then open B&C on it
            self._show_render(idx)
            rwin = self._render_windows.get(idx)
        if rwin is not None:
            rwin._show_brightness_contrast()
            rwin.raise_()

    def _focus_main_window(self) -> None:
        """Restore, show, and raise the main data-viewer window."""
        if self.isMinimized():
            self.showNormal()
        else:
            self.show()
        self.raise_()
        self.activateWindow()

    def _viewer_windows(self) -> list[QWidget]:
        app = QApplication.instance()
        if app is None:
            return []
        windows = []
        for win in app.topLevelWidgets():
            if win.isVisible() and win.windowTitle():
                windows.append(win)
        return windows

    def _cycle_windows(self, direction: int) -> None:
        windows = self._viewer_windows()
        if not windows:
            return
        active = QApplication.activeWindow()
        try:
            idx = windows.index(active)
        except ValueError:
            idx = self._window_cycle_index if 0 <= self._window_cycle_index < len(windows) else 0
        next_idx = (idx + direction) % len(windows)
        self._window_cycle_index = next_idx
        win = windows[next_idx]
        win.show()
        win.raise_()
        win.activateWindow()

    def _cycle_dataset(self, direction: int) -> None:
        if not self._state.datasets:
            self._show_dataset_manager()
            return
        current = self._state.active_idx if self._state.active_idx is not None else 0
        idx = (current + direction) % len(self._state.datasets)
        self._state.set_active(idx)
        self._raise_dataset_window(idx)

    def _raise_dataset_window(self, idx: int) -> None:
        win = self._render_windows.get(idx) or self._data_windows.get(idx)
        if win is None or win.isHidden():
            self._show_dataset_manager()
            return
        win.show()
        win.raise_()
        win.activateWindow()

    def _close_current_child_window(self) -> None:
        active = QApplication.activeWindow()
        if active is not None and active is not self:
            active.close()
            return
        # Fallback: use the top-level ancestor of the current focus widget.
        # QApplication.activeWindow() can lag on Windows when a window is shown
        # programmatically without an explicit user click.
        focus = QApplication.focusWidget()
        if focus is not None:
            top = focus.window()
            if top is not None and top is not self:
                top.close()

    # ------------------------------------------------------------------
    # Preferences, memory monitor, duplicate
    # ------------------------------------------------------------------

    def _show_preferences(self) -> None:
        """Open the modal Preferences dialog (Edit → Preferences…)."""
        from .preferences_dialog import PreferencesDialog
        dlg = PreferencesDialog(self._state, parent=self)
        dlg.exec()
        # After OK the dialog has already written prefs + saved; rebuild
        # any UI bits that depend on them.
        self._apply_shortcuts()
        self._populate_recent_menu()

    def _show_set_measurements(self) -> None:
        from .set_measurements_dialog import SetMeasurementsDialog
        SetMeasurementsDialog(self._state, parent=self).exec()

    # ------------------------------------------------------------------
    # Measure › Scale Bar
    # ------------------------------------------------------------------

    def _active_coordinate_view(self):
        """A 2-D coordinate view (render/scatter) for the active dataset, or None.

        Prefers the currently focused window when it is the active dataset's
        render/scatter and is in a 2-D (XY/XZ/YZ) view."""
        idx = self._state.active_idx
        if idx is None:
            return None
        rw = self._render_windows.get(idx)
        sw = self._scatter_windows.get(idx)

        def is_2d(win):
            try:
                return win is not None and win.coordinate_view_box() is not None
            except Exception:
                return False

        active = QApplication.activeWindow()
        for win in (active, rw, sw):
            if win in (rw, sw) and is_2d(win):
                return win
        return None

    def _window_active_rectangle_bounds(self, win):
        """Bounds (x0,x1,y0,y1) of the active rectangle ROI in *win*'s current
        view plane (draft or selected), or None."""
        from ..core.roi_selection import rectangle_bounds

        plane = win.roi_view_plane()
        candidates = []
        overlay = getattr(win, "_roi_overlay", None)
        if overlay is not None:
            try:
                rec = overlay.current_record()
            except Exception:
                rec = None
            if rec is not None:
                candidates.append(rec)
        try:
            sel = set(self._state.rois.selected_ids)
            candidates.extend(r for r in self._state.rois.records if r.id in sel)
        except Exception:
            pass
        for rec in candidates:
            if getattr(rec, "type", None) != "rectangle":
                continue
            rec_plane = (getattr(rec, "context", {}) or {}).get("view_plane") or plane
            if rec_plane != plane:
                continue
            b = rectangle_bounds(rec)
            if b is not None:
                return b
        return None

    @staticmethod
    def _nice_scalebar_width(span: float) -> float:
        if span <= 0:
            return 100.0
        target = span / 5.0
        best = 1.0
        for n in (1, 2, 5, 10, 20, 50, 100, 200, 500, 1000, 2000, 5000,
                  10000, 20000, 50000, 100000, 200000, 500000):
            if n <= target:
                best = float(n)
            else:
                break
        return best

    def _show_scale_bar(self) -> None:
        if self._state.active_dataset is None:
            self._state.log("Scale bar: open a dataset first.", "WARN")
            QMessageBox.information(self, "Scale Bar", "Open a dataset first.")
            return
        win = self._active_coordinate_view()
        if win is None:
            self._state.log("Scale bar: no 2-D coordinate view (render/scatter) for the "
                            "active dataset.", "WARN")
            QMessageBox.information(
                self, "Scale Bar",
                "Open a render or scatter view of the active dataset in a 2-D "
                "orientation (XY / XZ / YZ) first.")
            return
        vb = win.coordinate_view_box()
        try:
            (xmin, xmax), _ = vb.viewRange()
            span = abs(xmax - xmin)
        except Exception:
            span = 0.0
        default_w = self._nice_scalebar_width(span)
        default_h = max(default_w / 20.0, (span / 400.0) if span else 1.0)

        from .scale_bar_dialog import ScaleBarDialog
        dlg = ScaleBarDialog(self, default_width_nm=default_w, default_height_nm=default_h,
                             has_selection=self._window_active_rectangle_bounds(win) is not None)
        if dlg.exec() != QDialog.DialogCode.Accepted:
            return
        params = dlg.values()

        if params["location"] != "At Selection":
            self._add_scale_bar(win, params, roi_center=None)
            return

        bounds = self._window_active_rectangle_bounds(win)
        if bounds is not None:
            self._add_scale_bar(win, params, roi_center=self._rect_center(bounds))
            return
        # No rectangle ROI yet → let the user draw one, then place at its centre.
        overlay = getattr(win, "_roi_overlay", None)
        if overlay is None or not hasattr(overlay, "request_rectangle"):
            self._add_scale_bar(win, params, roi_center=None)  # fall back to a corner
            return
        self._state.log("Scale bar (At Selection): draw a rectangle on the view to "
                        "position the scale bar.", "INFO")

        def _on_rect(record, _win=win, _params=params):
            from ..core.roi_selection import rectangle_bounds
            b = rectangle_bounds(record)
            if b is not None:
                self._add_scale_bar(_win, _params, roi_center=self._rect_center(b))

        overlay.request_rectangle(_on_rect)

    @staticmethod
    def _rect_center(bounds) -> tuple[float, float]:
        x0, x1, y0, y1 = bounds
        return (0.5 * (x0 + x1), 0.5 * (y0 + y1))

    def _add_scale_bar(self, win, params, *, roi_center) -> None:
        vb = win.coordinate_view_box()
        if vb is None:
            self._state.log("Scale bar: the view is no longer a 2-D coordinate view.", "WARN")
            return
        from .scale_bar import ScaleBarItem, format_nm, initial_scale_bar_center
        center = initial_scale_bar_center(vb, params, roi_center=roi_center)
        try:
            sb = ScaleBarItem(
                vb, width_nm=params["width_nm"], height_nm=params["height_nm"],
                font_size=params["font_size"], color=params["color"],
                bg_color=params["bg_color"], horizontal=params["horizontal"],
                center=center)
        except Exception as exc:
            self._state.log(f"Scale bar failed: {exc}", "ERROR")
            return
        sb.sigEditRequested.connect(lambda item, _w=win: self._edit_scale_bar(_w, item))
        sb.sigDeleteRequested.connect(lambda item, _w=win: self._delete_scale_bar(_w, item))
        bars = getattr(win, "_scale_bars", None)
        if bars is None:
            bars = []
            win._scale_bars = bars
        bars.append(sb)
        where = "at selection" if roi_center is not None else params["location"].lower()
        self._state.log(f"Added {format_nm(params['width_nm'])} scale bar ({where}). "
                        "Drag to move; right-click for Property / Delete.")

    def _edit_scale_bar(self, win, item) -> None:
        from .scale_bar_dialog import ScaleBarDialog
        dlg = ScaleBarDialog(self, initial=item.params(), edit_mode=True)
        if dlg.exec() != QDialog.DialogCode.Accepted:
            return
        item.set_params(dlg.values())
        self._state.log("Updated scale bar properties.")

    def _delete_scale_bar(self, win, item) -> None:
        try:
            item.remove()
        except Exception:
            pass
        bars = getattr(win, "_scale_bars", None)
        if bars and item in bars:
            bars.remove(item)
        self._state.log("Deleted scale bar.")

    def _split_active_channel_group(self) -> None:
        """
        Split the active multi-channel render overlay into one render per dataset.

        The dataset objects, transforms, metadata, dataset manager entries, and
        data-info behaviour are left intact. Only the shared render-group key is
        removed so future Render windows no longer auto-compose the channels.
        """
        active_idx = self._state.active_idx
        if active_idx is None or not (0 <= active_idx < len(self._state.datasets)):
            self._no_data_warning()
            return

        active = self._state.datasets[active_idx]
        group_id = active.state.get("overlay_id") or active.state.get("render_group_id")
        if not group_id:
            QMessageBox.information(
                self,
                "Split channels",
                "The active dataset is not part of a multi-channel render overlay.",
            )
            return

        group_indices = [
            idx for idx, ds in enumerate(self._state.datasets)
            if (ds.state.get("overlay_id") or ds.state.get("render_group_id")) == group_id
        ]
        if len(group_indices) < 2:
            QMessageBox.information(
                self,
                "Split channels",
                "The active render group contains only one dataset.",
            )
            return

        lut_by_idx = self._current_group_luts(group_indices)
        fallback_luts = ["Red", "Green", "Blue", "Cyan", "Magenta", "Yellow", "Gray"]
        for pos, idx in enumerate(group_indices):
            ds = self._state.datasets[idx]
            ds.state["render_channel_lut"] = lut_by_idx.get(idx, fallback_luts[pos % len(fallback_luts)])
            ds.state.pop("overlay_id", None)
            ds.state.pop("overlay_index", None)
            ds.state.pop("overlay_order", None)
            ds.state.pop("overlay_transform", None)
            ds.state.pop("render_group_id", None)
            ds.state.pop("render_transform_2d", None)
            ds.metadata["render_split_from_group"] = group_id

        for idx in group_indices:
            win = self._render_windows.get(idx)
            if win is not None:
                try:
                    win._refresh_from_dataset()
                except Exception:
                    pass

        for idx in group_indices:
            self._show_render(idx)
        self._state.set_active(active_idx)
        self._state.log(
            f"Split channel overlay into {len(group_indices)} separate render view(s)."
        )

    def _current_group_luts(self, group_indices: list[int]) -> dict[int, str]:
        """Return current render LUTs for group members, if a render is open."""
        luts: dict[int, str] = {}
        wanted = set(group_indices)
        for win in list(self._render_windows.values()):
            try:
                channels = getattr(win, "_channels", [])
            except RuntimeError:
                continue
            for ch in channels:
                idx = ch.get("dataset_idx")
                if idx in wanted and idx not in luts:
                    luts[idx] = str(ch.get("lut") or "")
        return {idx: lut for idx, lut in luts.items() if lut}

    def _show_memory_monitor(self) -> None:
        """Open (or raise) the memory monitor (Help → Monitor memory…)."""
        if getattr(self, "_memory_win", None) is None:
            from .memory_monitor import MemoryMonitor
            self._memory_win = MemoryMonitor(self._state, parent=self)
        self._memory_win.show()
        self._memory_win.raise_()
        self._memory_win.activateWindow()

    def _close_active_dataset(self) -> None:
        from ..core.overlay import dataset_group_id

        # For multi-channel overlay views (render/scatter with multiple channels),
        # honour the active dataset that the user selected in the channel panel.
        # For standalone single-dataset windows, resolve active from the focused window.
        active_win = QApplication.activeWindow()
        is_multichannel = len(getattr(active_win, "_channels", None) or []) > 1
        if not is_multichannel:
            self._set_active_from_focused_dataset_window()

        idx = self._state.active_idx
        if idx is None:
            return

        ds = self._state.datasets[idx]
        group_id = dataset_group_id(ds)

        if group_id:
            remaining = [
                i for i, d in enumerate(self._state.datasets)
                if i != idx and dataset_group_id(d) == group_id
            ]
            # Last surviving member: strip overlay state so it renders standalone
            if len(remaining) == 1:
                survivor = self._state.datasets[remaining[0]]
                for key in (
                    "overlay_id", "render_group_id", "overlay_index",
                    "overlay_order", "overlay_lut", "overlay_transform",
                    "render_transform_2d",
                ):
                    survivor.state.pop(key, None)

        self._state.remove_dataset(idx)

        if self._state.active_idx is None:
            self.setWindowTitle(self.APP_NAME)

        # Refresh open render/scatter overlay windows so their channel lists
        # reflect the removal (indices have shifted after remove_dataset).
        if group_id:
            self._refresh_overlay_windows()

    def _refresh_overlay_windows(self) -> None:
        """Rebuild channel lists and re-render open render/scatter overlay windows."""
        for win in list(self._render_windows.values()):
            try:
                win._build_channels()
                win._rebuild_channel_ui()
                win._rebuild_all_grids()
                win._schedule_render()
            except Exception:
                pass
        for win in list(self._scatter_windows.values()):
            try:
                win._refresh()
            except Exception:
                pass

    def _close_all_datasets(self) -> None:
        n = len(self._state.datasets)
        if n == 0:
            return
        for idx in range(n - 1, -1, -1):
            self._state.remove_dataset(idx)
        self.setWindowTitle(self.APP_NAME)

    def _duplicate_active_dataset(self) -> None:
        """Edit → Duplicate / ``Shift+D``.

        With an active rectangle ROI on the dataset, opens the crop options
        dialog (first time per session, or whenever "stop asking" is off / a
        specific Z range is involved) and produces a cropped duplicate;
        otherwise a plain duplicate.
        """
        src = self._state.active_dataset
        if src is None:
            self._no_data_warning()
            return
        src_idx = self._state.active_idx
        record = self._active_region_record(src, src_idx)
        if record is None:
            self._plain_duplicate(src)     # non-region ROI or no ROI → whole dataset
            return

        from ..core import roi_crop as RC
        from ..core.overlay import overlay_members
        try:
            members = overlay_members(self._state, src_idx) or [(src_idx, src)]
        except Exception:
            members = [(src_idx, src)]
        channel_names = [ds.name for _i, ds in members]

        setup = self._roi_crop_setup.get(id(src))
        # Silent reuse only when stop-asking AND All-Z (a specific Z range can't
        # be reused — it varies by XY region); otherwise (re)show the dialog.
        show_dialog = setup is None or not setup.stop_asking or not setup.z_all
        opts = setup
        if show_dialog:
            from .crop_dialog import CropDialog
            z_vals = None
            if int(getattr(src.prop, "num_dim", 2) or 2) >= 3:
                coords = RC.display_coords(src)
                if coords.size:
                    xy_inside = RC.compute_crop_mask(src, record, trace_complete=False)
                    z_vals = coords[:, 2][xy_inside]
            dlg = CropDialog(
                src.name, has_roi=True,
                channels=channel_names if len(channel_names) > 1 else None,
                z_values=z_vals, initial=setup, parent=self,
            )
            if dlg.exec() != QDialog.DialogCode.Accepted:
                return
            opts = dlg.options()
            if opts.stop_asking:
                self._roi_crop_setup[id(src)] = opts
            else:
                self._roi_crop_setup.pop(id(src), None)

        if opts is None:
            self._plain_duplicate(src)
            return
        self._execute_crop(src_idx, src, record, members, opts)

    # ------------------------------------------------------------------
    def _active_region_record(self, ds, ds_idx):
        """The active **region** ROI (rectangle/oval/polygon/freehand) for *ds*.

        A freshly drawn ROI is a *draft* on the render/scatter overlay controller
        (not yet in ``state.rois.records``), so we look there as well as among the
        selected persistent ROIs. Non-region ROIs (line/point/angle) → ``None``
        (the duplicate then ignores the ROI and copies the whole dataset).
        """
        from ..core.roi_convert import REGION_TYPES

        # 1) a selected, persisted region ROI
        try:
            wanted = set(self._state.rois.selected_ids or [])
        except Exception:
            wanted = set()
        try:
            for record in self._state.rois.records:
                if record.id in wanted and record.type in REGION_TYPES:
                    return record
        except Exception:
            pass
        # 2) a freshly drawn draft on this dataset's render/scatter overlay
        for win in (self._render_windows.get(ds_idx), self._scatter_windows.get(ds_idx)):
            ctrl = getattr(win, "_roi_overlay", None)
            if ctrl is None:
                continue
            try:
                draft = ctrl.current_record()
            except Exception:
                draft = getattr(ctrl, "draft", None)
            if draft is not None and getattr(draft, "type", None) in REGION_TYPES:
                return draft
        return None

    def _copy_dataset(self, src, new_name: str):
        """Deep-copy a dataset as a fresh standalone (no overlay/source links)."""
        import copy

        from ..core.dataset import FileInfo, MinfluxDataset

        timestamp = datetime.now().strftime("%Y-%b-%d, %H:%M:%S")
        new_file = FileInfo(
            name=new_name, folder=src.file.folder, datetime=timestamp,
            raw_data=None, recent_path=src.file.recent_path,
        )
        kwargs = dict(
            file=new_file,
            prop=copy.deepcopy(src.prop),
            attr=copy.deepcopy(src.attr),
        )
        cali = copy.deepcopy(getattr(src, "cali", None))
        channel = copy.deepcopy(getattr(src, "channel", None))
        if cali is not None:
            kwargs["cali"] = cali
        if channel is not None:
            kwargs["channel"] = channel
        dup = MinfluxDataset(**kwargs)
        dup.metadata.update(copy.deepcopy(src.metadata))
        dup.state.update(copy.deepcopy(src.state))
        for key in ("msr_source_path", "msr_dataset_key", "msr_dataset_name"):
            dup.metadata.pop(key, None)
        for key in ("overlay_id", "overlay_index", "overlay_order",
                    "overlay_transform", "render_group_id", "render_transform_2d"):
            dup.state.pop(key, None)
            dup.metadata.pop(key, None)
        dup.metadata["source_dataset_name"] = src.file.name
        dup.metadata["created_note"] = f"{timestamp}, derived from dataset: {src.file.name}"
        return dup

    def _plain_duplicate(self, src) -> None:
        try:
            dup = self._copy_dataset(src, self._next_duplicate_name(src.file.name))
            dup.metadata["duplicated_from_dataset"] = src.file.name
            try:
                dup.filter_mask = src.filter_mask.copy()
            except Exception:
                pass
        except Exception as exc:
            self._state.log(f"Duplicate failed: {exc}", "ERROR")
            QMessageBox.critical(self, "Duplicate", str(exc))
            return
        self._state.add_dataset(dup)
        self._state.log(f"Duplicated dataset as '{dup.file.name}'.")
        try:
            self._state.journal.add(
                "transform",
                f"Duplicated dataset '{src.file.name}' as '{dup.file.name}' "
                f"(filter mask preserved).",
            )
        except Exception:
            pass

    def _execute_crop(self, src_idx, src, record, members, opts) -> None:
        """Produce cropped duplicate(s) per the chosen options (Model A or B).

        A multi-channel crop is kept grouped as a new overlay (each channel's
        LUT + transform carried over); a single-channel crop stays standalone,
        so it renders with the default LUT (hot) like any lone dataset.
        """
        import copy
        import uuid

        import numpy as np

        from ..core import roi_crop as RC

        if len(members) > 1 and opts.channels:
            sel = [(i, ds) for pos, (i, ds) in enumerate(members, start=1)
                   if pos in opts.channels]
        else:
            sel = [(src_idx, src)]
        if not sel:
            sel = [(src_idx, src)]

        z_range = None if opts.z_all else opts.z_range
        trace_complete = not opts.clip
        multi = len(sel) > 1

        overlay_id = overlay_index = None
        if multi:
            overlay_index = self._next_overlay_index
            self._next_overlay_index += 1
            overlay_id = f"overlay:{overlay_index}:{uuid.uuid4().hex}"

        made: list[int] = []
        prev_suspend = getattr(self._state, "suspend_auto_render", False)
        if multi:
            self._state.suspend_auto_render = True   # group before rendering
        try:
            exact = bool(getattr(opts, "exact_shape", False))
            for order, (_idx, ds) in enumerate(sel, start=1):
                try:
                    mask = RC.compute_crop_mask(
                        ds, record, z_range=z_range, trace_complete=trace_complete,
                        exact_shape=exact)
                    name = self._unique_name("CROP_", ds.file.name)
                    if opts.spatial_filter:                       # Model A
                        dup = self._copy_dataset(ds, name)
                        base = np.asarray(ds.filter_mask, dtype=bool)
                        if base.shape[0] != mask.shape[0]:
                            base = np.ones(mask.shape[0], dtype=bool)
                        dup.filter_mask = base & mask
                        if RC.crop_is_axis_aligned(ds, record, exact_shape=exact):
                            dup.state["filter_specs"] = list(ds.state.get("filter_specs") or []) \
                                + RC.crop_filter_specs(record, trace_complete=trace_complete,
                                                       z_range=z_range, exact_shape=exact)
                        else:
                            dup.metadata["crop_mask_only"] = True   # opaque mask (not re-evaluable)
                        dup.metadata["cropped_from_dataset"] = ds.file.name
                        kept = int(np.asarray(dup.filter_mask, dtype=bool).sum())
                    else:                                          # Model B (subset)
                        dup = RC.subset_dataset(ds, mask, name=name, prefs=self._state.prefs)
                        kept = int(dup.prop.num_loc)
                    if multi:
                        dup.state["overlay_id"] = overlay_id
                        dup.state["render_group_id"] = overlay_id
                        dup.state["overlay_index"] = overlay_index
                        dup.state["overlay_order"] = order
                        for key in ("overlay_lut", "render_channel_lut",
                                    "overlay_transform", "render_transform_2d"):
                            val = ds.state.get(key)
                            if val is not None:
                                dup.state[key] = copy.deepcopy(val)
                    made.append(self._state.add_dataset(dup))
                    model = "filter" if opts.spatial_filter else "subset"
                    self._state.log(
                        f"Cropped '{ds.file.name}' → '{name}' ({model}, {kept:,} locs).")
                except Exception as exc:
                    self._state.log(f"Crop of '{ds.file.name}' failed: {exc}", "ERROR")
        finally:
            self._state.suspend_auto_render = prev_suspend

        if not made:
            QMessageBox.warning(self, "Duplicate / crop",
                                "No cropped dataset was produced (check the ROI).")
            return
        # Move the ROI itself into the cropped view (active there), per the request.
        self._carry_roi_to_cropped(record, made[0])
        self._show_render(made[0])                  # open the cropped (grouped) view

    def _carry_roi_to_cropped(self, record, new_idx: int) -> None:
        """Add a copy of the source ROI to the store, retargeted to the cropped
        dataset and selected, so it shows as the active ROI in the new view."""
        import copy
        import uuid

        try:
            clone = copy.deepcopy(record)
        except Exception:
            return
        clone.id = uuid.uuid4().hex
        ctx = dict(clone.context) if isinstance(clone.context, dict) else {}
        ctx["dataset_idx"] = new_idx
        clone.context = ctx
        clone.mask_key = ""
        clone.selected_count = None
        clone.selection_dirty = True
        self._state.rois.add(clone)
        self._state.rois.select([clone.id])

    def _unique_name(self, prefix: str, base: str) -> str:
        """``<prefix><base>`` with numbering only when needed."""
        existing = {ds.file.name for ds in self._state.datasets}
        name = f"{prefix}{base}"
        if name not in existing:
            return name
        n = 2
        while f"{prefix}{n}_{base}" in existing:
            n += 1
        return f"{prefix}{n}_{base}"

    def _next_duplicate_name(self, source_name: str) -> str:
        """Return DUP_<source_name>, with numbering only when needed."""
        return self._unique_name("DUP_", source_name)

    def _duplicate_dataset_for_overlay(self, src_idx: int, timestamp: str) -> int:
        import copy

        from ..core.dataset import FileInfo, MinfluxDataset

        src = self._state.datasets[src_idx]
        new_attrs = copy.deepcopy(src.attr)
        new_components = copy.deepcopy(src.components)
        if getattr(new_components, "mfx", None) is not None:
            new_components.mfx.attrs = new_attrs
        dup = MinfluxDataset(
            file=FileInfo(
                name=self._next_duplicate_name(src.file.name),
                folder=src.file.folder,
                datetime=timestamp,
                raw_data=None,
                recent_path=src.file.recent_path,
            ),
            prop=copy.deepcopy(src.prop),
            attr=new_attrs,
            cali=copy.deepcopy(src.cali),
            channel=copy.deepcopy(src.channel),
            components=new_components,
        )
        dup.file.datetime = timestamp
        dup.metadata.update(copy.deepcopy(src.metadata))
        dup.state.update(copy.deepcopy(src.state))
        for key in (
            "overlay_id", "overlay_index", "overlay_order", "overlay_transform",
            "render_group_id", "render_transform_2d",
        ):
            dup.state.pop(key, None)
            dup.metadata.pop(key, None)
        dup.metadata["duplicated_from_dataset"] = src.file.name
        dup.metadata["created_note"] = f"{timestamp}, duplicated for channel overlay."
        return self._state.add_dataset(dup)

    def _show_channel_combine(self) -> None:
        if len(self._state.datasets) < 2:
            QMessageBox.information(self, "Combine datasets", "Load at least two datasets before combining channels.")
            return
        from .channel_combine_dialog import ChannelCombineDialog
        from ..core.overlay import OverlayMemberSpec, build_overlay_transforms
        import uuid

        dlg = ChannelCombineDialog(
            self._state,
            previous=self._last_channel_combine_settings,
            parent=self,
        )
        result = dlg.exec()
        self._last_channel_combine_settings = dlg.session_state()
        if result != QDialog.DialogCode.Accepted:
            return
        rows = dlg.selected_rows()
        if len(rows) < 2:
            QMessageBox.warning(self, "Combine datasets", "Select at least two datasets to combine.")
            return

        timestamp = datetime.now().strftime("%Y-%b-%d, %H:%M:%S")
        keep_source = dlg.keep_source_check.isChecked()
        previous_suspend = getattr(self._state, "suspend_auto_render", False)
        self._state.suspend_auto_render = True
        try:
            overlay_rows = []
            for row in rows:
                idx = int(row["dataset_idx"])
                if keep_source:
                    idx = self._duplicate_dataset_for_overlay(idx, timestamp)
                overlay_rows.append({"dataset_idx": idx, "order": int(row["order"]), "lut": row["lut"]})
        finally:
            self._state.suspend_auto_render = previous_suspend

        overlay_index = self._next_overlay_index
        self._next_overlay_index += 1
        overlay_id = f"overlay:{overlay_index}:{uuid.uuid4().hex}"
        specs = [
            OverlayMemberSpec(dataset_idx=row["dataset_idx"], order=pos + 1, lut=row["lut"])
            for pos, row in enumerate(sorted(overlay_rows, key=lambda item: (item["order"], item["dataset_idx"])))
        ]
        transforms = build_overlay_transforms(
            state=self._state,
            members=specs,
            overlay_id=overlay_id,
            overlay_index=overlay_index,
            alignment_mode=dlg.align_combo.currentText(),
        )
        for spec in specs:
            ds = self._state.datasets[spec.dataset_idx]
            transform = transforms[spec.dataset_idx]
            ds.state["overlay_id"] = overlay_id
            ds.state["render_group_id"] = overlay_id
            ds.state["overlay_index"] = overlay_index
            ds.state["overlay_order"] = spec.order
            ds.state["overlay_lut"] = spec.lut
            ds.state["render_channel_lut"] = spec.lut
            ds.state["overlay_transform"] = transform
            ds.state["render_transform_2d"] = transform
            ds.metadata["overlay_id"] = overlay_id
            ds.metadata["overlay_index"] = overlay_index
            ds.metadata["overlay_alignment_mode"] = dlg.align_combo.currentText()
        anchor_idx = specs[0].dataset_idx
        self._state.set_active(anchor_idx)
        self._show_render(anchor_idx)
        self._notify_view_state_changed()
        self._state.log(f"Created overlay {overlay_index} with {len(specs)} dataset(s).")

    def _show_channel_separation(self) -> None:
        """Process › Channel… › Separate Channel by DCR — open the DCR two-colour
        separation tool for the active dataset."""
        idx = self._state.active_idx
        if idx is None or not (0 <= idx < len(self._state.datasets)):
            self._no_data_warning()
            return
        ds = self._state.datasets[idx]
        from ..core.loader import attr_values_1d
        if attr_values_1d(ds, "dcr") is None:
            QMessageBox.information(
                self, "Separate Channel by DCR",
                "The active dataset has no DCR attribute, so it cannot be separated by DCR.")
            return
        from .channel_separation_window import ChannelSeparationWindow
        from .modeless import show_modeless
        win = ChannelSeparationWindow(self._state, idx, owner=self)
        show_modeless(win, self)

    def apply_dcr_channel_separation(self, src_idx: int, labels) -> bool:
        """Build red / green / (unassigned) truncated copies from per-localization
        channel *labels* (0/1/-1) and combine them as a render overlay. Returns
        True on success. Called by ``ChannelSeparationWindow`` on Apply."""
        import uuid

        import numpy as np

        from ..core.overlay import display_transform_record, identity_matrix4
        from ..core.roi_crop import subset_dataset

        if not (0 <= src_idx < len(self._state.datasets)):
            return False
        src = self._state.datasets[src_idx]
        labels = np.asarray(labels).ravel()
        n = int(getattr(src.prop, "num_loc", labels.size))
        if labels.size != n:
            return False
        base_name = src.name
        # (mask, LUT, name suffix, hidden-by-default)
        plan = [
            (labels == 0, "Red", "ch1 red", False),
            (labels == 1, "Green", "ch2 green", False),
            (labels == -1, "Gray", "unassigned", True),
        ]
        overlay_index = self._next_overlay_index
        self._next_overlay_index += 1
        overlay_id = f"overlay:{overlay_index}:{uuid.uuid4().hex}"
        prefs = self._state.prefs

        new_indices: list[int] = []
        order = 0
        previous = getattr(self._state, "suspend_auto_render", False)
        self._state.suspend_auto_render = True
        try:
            for mask, lut, suffix, hidden in plan:
                keep = np.asarray(mask, dtype=bool)
                if not keep.any():
                    continue
                order += 1
                new_ds = subset_dataset(src, keep, name=f"{base_name} [{suffix}]", prefs=prefs)
                idx = self._state.add_dataset(new_ds)
                ds = self._state.datasets[idx]
                transform = display_transform_record(
                    overlay_id=overlay_id, overlay_index=overlay_index, order=order,
                    lut=lut, source_dataset_idx=idx, alignment_mode="stage origin",
                    matrix_4x4=identity_matrix4(),
                    provenance={"method": "dcr channel separation", "source_dataset": base_name})
                ds.state["overlay_id"] = overlay_id
                ds.state["render_group_id"] = overlay_id
                ds.state["overlay_index"] = overlay_index
                ds.state["overlay_order"] = order
                ds.state["overlay_lut"] = lut
                ds.state["render_channel_lut"] = lut
                ds.state["overlay_transform"] = transform
                ds.state["render_transform_2d"] = transform
                if hidden:
                    ds.state["overlay_default_hidden"] = True
                ds.metadata["overlay_id"] = overlay_id
                ds.metadata["overlay_index"] = overlay_index
                ds.metadata["dcr_separated_from"] = base_name
                new_indices.append(idx)
        finally:
            self._state.suspend_auto_render = previous

        if not new_indices:
            QMessageBox.warning(self, "Separate Channel by DCR",
                                "No localizations were assigned to a channel.")
            return False
        anchor = new_indices[0]
        self._state.set_active(anchor)
        self._show_render(anchor)
        self._notify_view_state_changed()
        self._state.log(
            f"Separated '{base_name}' into {len(new_indices)} DCR channel(s) (overlay {overlay_index}).")
        return True

    def _show_render(self, dataset_idx: int | None = None):
        """
        Open (or raise) the render window for a dataset.
        Defaults to the active dataset.
        """
        if self._state.active_dataset is None:
            self._no_data_warning()
            return
        idx = dataset_idx if type(dataset_idx) is int else self._state.active_idx
        if idx is None:
            return

        win = self._render_windows.get(idx)
        if win is not None:
            self._install_window_shortcuts(win)
            try:
                win._refresh_from_dataset()
            except Exception:
                pass
            win.show()
            win.raise_()
            win.activateWindow()
            self._notify_view_state_changed()
            return win

        from .render_window import RenderWindow
        win = RenderWindow(self._state, dataset_idx=idx)
        self._install_window_shortcuts(win)
        win.destroyed.connect(
            lambda _=None, i=idx: self._render_windows.pop(i, None)
        )
        self._render_windows[idx] = win
        win.show()
        self._notify_view_state_changed()
        return win

    # ------------------------------------------------------------------
    # ParaView
    # ------------------------------------------------------------------

    def _save_data(self) -> None:
        ds = self._state.active_dataset
        if ds is None:
            QMessageBox.information(
                self, "Save processed data", "No active dataset to save."
            )
            return
        from .save_dialog import SaveProcessedDataDialog

        # A dataset is "file-backed" when it has a physical data file that, when
        # re-opened, reproduces it (.mat/.npy/.json). For those we only write the
        # metadata sidecar; otherwise (.msr extract, duplicate) we save the data.
        src = Path(getattr(ds.file, "folder", "") or "") / (getattr(ds.file, "name", "") or "")
        file_backed = src.is_file() and src.suffix.lower() in (".mat", ".npy", ".json")
        default_dir = self._state.prefs["file"].get("default_folder", str(Path.home()))

        dlg = SaveProcessedDataDialog(
            getattr(ds, "name", ""),
            file_backed=file_backed,
            source_path=src if file_backed else None,
            default_dir=default_dir,
            parent=self,
        )
        if dlg.exec() != QDialog.DialogCode.Accepted:
            return
        opts = dlg.options()
        from ..core.save import save_processed

        try:
            written = save_processed(
                ds,
                data_path=opts["data_path"],
                fmt=opts["fmt"],
                metadata_dir=src.parent if file_backed else None,
            )
        except Exception as exc:
            QMessageBox.critical(
                self, "Save failed", f"Could not save processed data:\n{exc}"
            )
            return
        self._state.log(
            f"Saved processed data: {', '.join(p.name for p in written)}", "INFO"
        )
        QMessageBox.information(
            self, "Save processed data",
            "Saved:\n" + "\n".join(str(p) for p in written),
        )

    # ------------------------------------------------------------------
    # AppState signal handlers
    # ------------------------------------------------------------------

    def _on_dataset_added(self, idx: int) -> None:
        ds  = self._state.datasets[idx]
        win = DataWindow(ds, idx, self._state)
        offset = idx * 22
        win.move(120 + offset, 600 - offset)
        self._data_windows[idx] = win
        self._populate_recent_menu()

        data_prefs = self._state.prefs.get("data", {})
        if data_prefs.get("show_data_info", True):
            win.show()

        if data_prefs.get("show_attr_plot", False):
            self._show_attr_plot(idx)
        if data_prefs.get("show_histogram", False):
            self._show_histogram(idx)
        if data_prefs.get("show_scatter", False):
            self._show_scatter(idx)

        # Auto-open a render window for this dataset only when requested,
        # unless a batch importer will open one grouped render view.
        if data_prefs.get("show_render", True) and not getattr(self._state, "suspend_auto_render", False):
            self._show_render(idx)
        self._schedule_post_load_computations(idx)

    def _schedule_post_load_computations(self, idx: int) -> None:
        """Run optional one-time computed attributes after initial windows show."""
        delay_ms = 450 + max(0, idx) * 100
        QTimer.singleShot(delay_ms, lambda i=idx: self._run_post_load_computations(i))

    def _run_post_load_computations(self, idx: int) -> None:
        if not (0 <= idx < len(self._state.datasets)):
            return
        ds = self._state.datasets[idx]
        data_prefs = self._state.prefs.get("data", {})

        if data_prefs.get("compute_rimf", True) and "rimf" not in ds.derived:
            import numpy as np
            plot_prefs = self._state.prefs.get("plot", {})
            if ds.prop.num_dim < 3:
                # 2D data: Z is all zero, so anisotropy estimation is moot.
                # RIMF = 1.0 keeps the Z scaling a no-op and stays consistent
                # with the 3D pipeline without spending the computation.
                ds.set_rimf(1.0, source="2D (no z correction)")
                ds.derived["rimf"] = np.asarray([1.0], dtype=float)
                self._state.log(f"RIMF for '{ds.name}': 1.0 (2D dataset, computation skipped).")
            elif plot_prefs.get("use_fixed_rimf", False):
                # Manual override: apply the fixed preference value instead of
                # estimating. Tracked as a 'fixed' provenance source.
                fixed = float(plot_prefs.get("rimf_value", 1.0))
                ds.set_rimf(fixed, source="fixed (preference)")
                ds.derived["rimf"] = np.asarray([fixed], dtype=float)
                self._state.log(f"RIMF for '{ds.name}': {fixed:.4g} (fixed preference value).")
            else:
                # compute_rimf checked: run the same anisotropy estimation as the
                # plugin (raw last-valid z, never the RIMF-corrected loc_nm).
                value = None
                try:
                    self._state.log(f"Estimating anisotropy / RIMF of '{ds.name}'…")
                    from ..analysis.trace_analysis import estimate_anisotropy_for_dataset
                    res = estimate_anisotropy_for_dataset(ds)
                    if res is not None and np.isfinite(res["rimf"]):
                        value = float(res["rimf"])
                        # Apply only when in the physically expected window;
                        # otherwise treat as failed and reset to 1.0 below.
                        if 0.5 <= value <= 1.0:
                            ds.set_rimf(value, source="auto (estimate anisotropy)")
                            ds.derived["rimf"] = np.asarray([value], dtype=float)
                            ds.derived["rimf_sizes_x"] = res["x"].sizes
                            ds.derived["rimf_sizes_y"] = res["y"].sizes
                            ds.derived["rimf_sizes_z"] = res["z"].sizes
                            self._state.log(
                                f"Computed RIMF for '{ds.name}': {value:.4g} (raw last-valid z)"
                            )
                except Exception as exc:
                    self._state.log(f"RIMF estimation failed for '{ds.name}': {exc}", "WARN")
                if "rimf" not in ds.derived:
                    # Out of [0.5, 1.0] or estimation failed → reset to 1.0.
                    ds.set_rimf(1.0, source="auto (out of range → 1.0)")
                    ds.derived["rimf"] = np.asarray([1.0], dtype=float)
                    shown = f"{value:.4g}" if value is not None else "failed"
                    self._state.log(
                        f"RIMF for '{ds.name}': estimate {shown} outside [0.5, 1.0] — reset to 1.0.",
                        "WARN",
                    )
            # This runs after the dataset's windows are shown, so notify the
            # views to reflect the freshly-applied RIMF (data-info RIMF text,
            # render/scatter z-scaling).
            self._state.notify_calibration_changed(idx)

        if data_prefs.get("compute_loc_prec", True) and "sigma_per_trace_nm" not in ds.derived:
            try:
                self._state.log(f"Computing localization precision of '{ds.name}'…")
                from ..analysis.localization_precision import stddev_per_trace
                from ..core.loader import attr_values_1d
                import numpy as np
                loc_x = np.asarray(attr_values_1d(ds, "loc_x"))
                loc_y = np.asarray(attr_values_1d(ds, "loc_y"))
                _z = attr_values_1d(ds, "loc_z")
                loc_z = np.zeros_like(loc_x) if _z is None else np.asarray(_z)
                _tid = attr_values_1d(ds, "tid")
                tid = np.arange(loc_x.size) if _tid is None else np.asarray(_tid)
                result = stddev_per_trace(np.column_stack([loc_x, loc_y, loc_z]), tid)
                ds.attr["sigma_per_trace_nm"] = result["per_trace_sigma_xyz"]
                ds.attr["sigma_trace_ids"] = result["trace_ids"]
                ds.derived["sigma_per_trace_nm"] = result["per_trace_sigma_xyz"]
                ds.derived["sigma_trace_ids"] = result["trace_ids"]
                ds.cali.loc_precision = np.asarray(result["median_sigma_xyz"], dtype=float)
                self._state.log(
                    f"Computed localization precision for '{ds.name}' using StdDev per trace: "
                    f"median sigma=({result['median_sigma_xyz'][0]:.3g}, "
                    f"{result['median_sigma_xyz'][1]:.3g}, {result['median_sigma_xyz'][2]:.3g}) nm."
                )
            except Exception as exc:
                self._state.log(f"Localization precision computation skipped for '{ds.name}': {exc}", "WARN")

        if data_prefs.get("compute_local_density", True) and "den" not in ds.attr:
            try:
                self._state.log(f"Computing local density of '{ds.name}'…")
                from ..analysis.local_density import compute_local_density_for_dataset
                density, method, detail = compute_local_density_for_dataset(ds, self._state.prefs)
                ds.attr["den"] = density
                ds.attr["local_density"] = density
                ds.derived["den"] = density
                ds.derived["local_density"] = density
                for name in ("den", "local_density"):
                    if name not in ds.prop.attr_names:
                        ds.prop.attr_names.append(name)
                self._state.log(
                    f"Computed local density for '{ds.name}' using {method} ({detail})."
                )
            except Exception as exc:
                self._state.log(f"Local density computation skipped for '{ds.name}': {exc}", "WARN")

        # For all-iteration loads ds.attr is not the last-valid materialization;
        # compute density at the last-valid selection too so mfx_get can
        # broadcast `den` across iteration views (matches an iter_load="last"
        # load of the same file).
        if (
            data_prefs.get("compute_local_density", True)
            and len(ds.components.derived_last)
            and "den" not in ds.components.derived_last
        ):
            try:
                import numpy as np

                from ..analysis.local_density import compute_local_density_for_points
                from ..core.loader import mfx_get
                x = mfx_get(ds, "loc_x", itr="last")
                y = mfx_get(ds, "loc_y", itr="last")
                z = mfx_get(ds, "loc_z", itr="last")
                if x is not None and y is not None:
                    if z is None:
                        z = np.zeros_like(np.asarray(x, dtype=float))
                    points_nm = np.column_stack([
                        np.asarray(x, dtype=float) * 1e9,
                        np.asarray(y, dtype=float) * 1e9,
                        np.asarray(z, dtype=float) * 1e9 * ds.cali.RIMF,
                    ])
                    dims = int(data_prefs.get(
                        "local_density_dimensions", 3 if ds.prop.num_dim == 3 else 2,
                    ))
                    density, method, detail = compute_local_density_for_points(
                        points_nm, self._state.prefs, dimensions=dims,
                    )
                    ds.components.derived_last["den"] = density
                    self._state.log(
                        f"Computed last-valid local density for '{ds.name}' using {method} ({detail})."
                    )
            except Exception as exc:
                self._state.log(
                    f"Last-valid local density computation skipped for '{ds.name}': {exc}", "WARN",
                )

        # Restored-from-metadata filters re-evaluate now that derived attributes
        # (den, …) exist, so a re-opened processed dataset shows its saved filter
        # state. (filter_specs is only populated here via a metadata sidecar.)
        if ds.state.get("filter_specs"):
            try:
                from ..core.loader import apply_saved_filters
                if apply_saved_filters(ds):
                    self._state.notify_filter_changed(idx)
            except Exception as exc:
                self._state.log(f"Restoring filters for '{ds.name}' failed: {exc}", "WARN")

    def _on_dataset_removed(self, idx: int) -> None:
        for mapping in (
            self._data_windows,
            self._render_windows,
            self._scatter_windows,
            self._histogram_windows,
            self._attr_windows,
            self._filter_dlgs,
        ):
            win = mapping.pop(idx, None)
            if win is not None:
                try:
                    win.close()
                    win.deleteLater()
                except Exception:
                    pass
            self._reindex_window_map_after_remove(mapping, idx)
        self._notify_view_state_changed()

    def _reindex_window_map_after_remove(self, mapping: dict[int, QWidget], removed_idx: int) -> None:
        moved = {}
        for key, win in list(mapping.items()):
            new_key = key - 1 if key > removed_idx else key
            if new_key != key:
                mapping.pop(key, None)
                moved[new_key] = win
                if hasattr(win, "_idx"):
                    setattr(win, "_idx", new_key)
                if hasattr(win, "_dataset_idx"):
                    setattr(win, "_dataset_idx", new_key)
                # The window now points to a different dataset slot — drop any
                # per-window coordinate cache and redraw so it shows the correct
                # (e.g. averaged, re-zeroed) localizations, not the previous ones.
                if hasattr(win, "_cached_dataset_idx"):
                    win._cached_dataset_idx = None
                    win._cached_locs_nm = None
                for refresh in ("_refresh", "_refresh_from_dataset"):
                    fn = getattr(win, refresh, None)
                    if callable(fn):
                        try:
                            fn()
                        except Exception:
                            pass
                        break
        mapping.update(moved)

    def _on_active_changed(self, idx: int) -> None:
        if not (0 <= idx < len(self._state.datasets)):
            return
        ds = self._state.datasets[idx]
        self.setWindowTitle(f"{self.APP_NAME}  —  {ds.name}")
        # Report the active dataset to the Log (skip consecutive duplicates from
        # window-focus re-emits).
        if getattr(self, "_last_active_log_idx", None) != idx:
            self._last_active_log_idx = idx
            self._state.log(f"Active dataset: '{ds.name}'")

    # ------------------------------------------------------------------
    # Drag-and-drop  (files AND folders; any supported extension)
    # ------------------------------------------------------------------

    def dragEnterEvent(self, event: QDragEnterEvent) -> None:
        if event.mimeData().hasUrls():
            # Accept if at least one URL is a supported file or a directory
            for url in event.mimeData().urls():
                p = Path(url.toLocalFile())
                if p.is_dir() or p.suffix.lower() in _SUPPORTED_EXTS:
                    event.acceptProposedAction()
                    return
        event.ignore()

    def dragMoveEvent(self, event) -> None:
        # Must accept dragMoveEvent too, or Qt cancels the drop mid-drag
        if event.mimeData().hasUrls():
            for url in event.mimeData().urls():
                p = Path(url.toLocalFile())
                if p.is_dir() or p.suffix.lower() in _SUPPORTED_EXTS:
                    event.acceptProposedAction()
                    return
        event.ignore()

    def dropEvent(self, event: QDropEvent) -> None:
        for url in event.mimeData().urls():
            self._route_path(url.toLocalFile())
        event.acceptProposedAction()

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    # ------------------------------------------------------------------
    # LUT / Color / Show Info / Localization Precision / Toolbar icons
    # ------------------------------------------------------------------

    def _show_lut(self) -> None:
        """Toolbar LUT button — open the Fiji-style LUT editor (#1 fix)."""
        active = QApplication.activeWindow()
        if active is not None and hasattr(active, "open_lut_dialog"):
            try:
                active.open_lut_dialog()
                return
            except Exception as exc:
                self._state.log(f"LUT failed for active window: {exc}", "ERROR")
        if self._state.active_dataset is None:
            self._no_data_warning(); return
        idx = self._state.active_idx
        rwin = self._render_windows.get(idx)
        if rwin is None:
            self._show_render(idx)
            rwin = self._render_windows.get(idx)
        if rwin is None or not hasattr(rwin, "open_lut_dialog"):
            QMessageBox.information(
                self, "LUT",
                "Open the render view first, then use the LUT button.",
            )
            return
        rwin.open_lut_dialog()

    def _show_color_picker(self) -> None:
        """Toolbar Color button — pick a single colour for ROIs/render."""
        from PyQt6.QtGui import QColor
        from PyQt6.QtWidgets import QColorDialog
        current = self._state.prefs.get("plot", {}).get("roi_color", "Yellow")
        # Accept either a name like "Yellow" or a hex string
        c0 = QColor(current) if QColor(current).isValid() else QColor("yellow")
        c = QColorDialog.getColor(c0, self, "Choose colour")
        if not c.isValid():
            return
        self._state.prefs.setdefault("plot", {})["roi_color"] = c.name()
        self._state.save_prefs()
        self._state.log(f"ROI / single colour set to {c.name()}.")

    def _show_info_for_active(self) -> None:
        """View → Show Info — open data-info window(s) for the active dataset."""
        if self._state.active_dataset is None:
            self._no_data_warning(); return
        active_idx = self._state.active_idx
        if active_idx is None:
            return
        indices = self._info_indices_for_active(active_idx)
        active_win = None
        for offset, idx in enumerate(indices):
            win = self._data_windows.get(idx)
            if win is None:
                from .data_window import DataWindow
                win = DataWindow(self._state.datasets[idx], idx, self._state)
                self._data_windows[idx] = win
            if offset:
                win.move(self.x() + 120 + offset * 28, self.y() + 160 + offset * 28)
            win.show()
            win.raise_()
            if idx == active_idx:
                active_win = win
        if active_win is not None:
            active_win.raise_()
            active_win.activateWindow()

    def _info_indices_for_active(self, active_idx: int) -> list[int]:
        """Show all members of an imported multi-channel MSR group together."""
        try:
            active = self._state.datasets[active_idx]
        except Exception:
            return []
        group_id = active.state.get("overlay_id") or active.state.get("render_group_id")
        msr_source = active.metadata.get("msr_source_path")
        if not group_id and not msr_source:
            return [active_idx]
        indices: list[int] = []
        for idx, ds in enumerate(self._state.datasets):
            if group_id and (ds.state.get("overlay_id") or ds.state.get("render_group_id")) == group_id:
                indices.append(idx)
            elif msr_source and ds.metadata.get("msr_source_path") == msr_source:
                indices.append(idx)
        return indices or [active_idx]

    def _loc_precision_frc(self) -> None:
        if self._state.active_dataset is None:
            self._no_data_warning(); return
        from .. import analysis
        analysis.run_frc(self, self._state)

    def _loc_precision_crlb(self) -> None:
        if self._state.active_dataset is None:
            self._no_data_warning(); return
        from .. import analysis
        analysis.run_crlb(self, self._state)

    def _loc_precision_stddev(self) -> None:
        if self._state.active_dataset is None:
            self._no_data_warning(); return
        from .. import analysis
        analysis.run_stddev_per_trace(self, self._state)

    def _run_local_density(self) -> None:
        if self._state.active_dataset is None:
            self._no_data_warning(); return
        from .. import analysis
        analysis.run_local_density(self, self._state)

    # ------------------------------------------------------------------
    # Trace analysis handlers
    # ------------------------------------------------------------------

    def _run_trace_size(self) -> None:
        if self._state.active_dataset is None:
            self._no_data_warning(); return
        from ..analysis.trace_analysis import show_trace_size_dialog
        show_trace_size_dialog(self, self._state.active_dataset, self._state)

    def _run_trace_anisotropy(self) -> None:
        if self._state.active_dataset is None:
            self._no_data_warning(); return
        from ..analysis.trace_analysis import show_anisotropy_dialog
        show_anisotropy_dialog(self, self._state.active_dataset, self._state)

    def _show_trace_viewer(self) -> None:
        if self._state.active_dataset is None:
            self._no_data_warning(); return
        from .trace_viewer import TraceViewer
        idx = self._state.active_idx
        win = TraceViewer(self._state, dataset_idx=idx)
        win.show()
        win.raise_()
        win.activateWindow()

    def _install_toolbar_icons(self) -> None:
        """Attach PNG icons to the toolbar QActions (item #5)."""
        from PyQt6.QtGui import QIcon

        from .. import resource_path
        icon_dir = resource_path("icons")
        mapping = {
            "toolRect":     "rec.png",
            "toolOval":     "ellipse.png",
            "toolPolygon":  "polygon.png",
            "toolFreehand": "free.png",
            "toolLine":     "line.png",
            "toolPoint":    "point.png",
            "toolLut":      "lut.png",
            "toolColor":    "color.png",
        }
        for attr, fname in mapping.items():
            action = getattr(self._ui, attr, None)
            if action is None:
                continue
            ipath = icon_dir / fname
            if ipath.exists():
                action.setIcon(QIcon(str(ipath)))
        # Angle tool lives on self (not the generated UI).
        angle_icon = icon_dir / "angle.png"
        if hasattr(self, "toolAngle") and angle_icon.exists():
            self.toolAngle.setIcon(QIcon(str(angle_icon)))
        lasso_icon = icon_dir / "lasso.png"
        if hasattr(self, "toolMagneticLasso") and lasso_icon.exists():
            self.toolMagneticLasso.setIcon(QIcon(str(lasso_icon)))

    def _no_data_warning(self) -> None:        QMessageBox.information(self, "No data", "Please load a dataset first.")

    def _placeholder(self, feature: str, phase: str) -> None:
        QMessageBox.information(
            self, feature,
            f"<b>{feature}</b> will be implemented in {phase}.",
        )

    def _check_for_updates(self, *, silent: bool = False) -> None:
        """Tier-A update check: query GitHub Releases off-thread, then report.

        ``silent`` (the on-startup check) only surfaces an available update.
        """
        from .. import __version__
        if not silent:
            self._status_label.setText("Checking for updates…")
        task = _UpdateCheckTask(__version__)
        # Keep a reference: QRunnable auto-deletes after run(), which would race
        # the queued cross-thread signal and drop the result.
        self._update_tasks.add(task)
        task.signals.done.connect(
            lambda result, t=task: self._on_update_check_done(result, silent, t)
        )
        QThreadPool.globalInstance().start(task)

    def _on_update_check_done(self, result, silent: bool, task=None) -> None:
        from .update_dialog import show_update_result
        self._update_tasks.discard(task)
        if not silent:
            self._status_label.setText("Ready.")
        if result.status == "update_available":
            self._state.log(
                f"Update available: {result.latest.tag} "
                f"(installed v{result.current_version}).", "INFO",
            )
        show_update_result(result, self, silent=silent)

    def _show_about(self) -> None:
        from .. import __version__
        QMessageBox.about(
            self, f"About {self.APP_NAME}",
            f"<h3>MINFLUX Data Viewer v{__version__}</h3>"
            "<p>A Python/Qt GUI tool for reading, filtering, and visualization of"
            " Abberior MINFLUX nanoscopy data.</p>"
            "<p>The viewer was originally developed in MATLAB to support user "
            "projects at "
            "<a href='https://www.embl.org/about/info/imaging-centre/super-resolution-imaging/#vf-tabs__section-c46b219a-947a-467f-a169-838b85a0d06b'>" \
            "the Advanced Light Microscopy service team of the "
            "EMBL Imaging Centre.</a></p>"
            "<p>It has been rewritten and redesigned in Python with assistance "
            "from AI agents, with the goals of openness, flexibility, and "
            "extensibility.</p>"
            "<p>The interface deliberately follows an ImageJ-like design, in hope that users "
            "can more easily adapt to it.</p>"
            "<p>To support user judgment, functions generated by AI agents that "
            "have not yet been fully approved by a human are shown in grey. Human-approved "
            "or human-scripted functions remain in black.</p>"
            "<p>EMBL Imaging Centre<br>"
            "<a href='mailto:ziqiang.huang@embl.de'>ziqiang.huang@embl.de</a></p>",
        )

    def _setup_toolbar_widgets(self) -> None:
        """Add a leading spacer (to shift the tools right so the first ROI button
        roughly aligns with the 'File' menu) and a right-aligned search field."""
        tb = self._ui.toolbar
        self._toolbar_aligned = False

        # Leading spacer — its width is set in _align_toolbar_to_menu() once the
        # toolbar/menubar have a valid layout (after the first show).
        self._toolbar_lead_spacer = QWidget()
        self._toolbar_lead_spacer.setFixedWidth(0)
        first = tb.actions()[0] if tb.actions() else None
        if first is not None:
            tb.insertWidget(first, self._toolbar_lead_spacer)
        else:
            tb.addWidget(self._toolbar_lead_spacer)

        # Expanding spacer pushes the search field to the toolbar's right edge,
        # regardless of window width.
        expander = QWidget()
        expander.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Preferred)
        tb.addWidget(expander)

        self._toolbar_search = QLineEdit()
        self._toolbar_search.setPlaceholderText("type here to search")
        self._toolbar_search.setClearButtonEnabled(True)
        fm = self._toolbar_search.fontMetrics()
        width = fm.horizontalAdvance("x" * 25) + 24   # ~25 letters + padding/clear button
        self._toolbar_search.setFixedWidth(int(width))
        tb.addWidget(self._toolbar_search)

    def _align_toolbar_to_menu(self) -> None:
        """Size the leading spacer so the first ROI button's left edge roughly
        aligns vertically with the 'File' menu's left edge."""
        try:
            tb = self._ui.toolbar
            mb = self.menuBar()
            menu_actions = mb.actions()
            btn = tb.widgetForAction(self._ui.toolRect)
            if not menu_actions or btn is None:
                return
            file_left = mb.mapTo(self, mb.actionGeometry(menu_actions[0]).topLeft()).x()
            tool_left = btn.mapTo(self, QPoint(0, 0)).x()
            spacer_w = self._toolbar_lead_spacer.width()
            # Align the first ROI button with the 'File' menu, then nudge a bit
            # further right by ~1/4 of a button width.
            extra = 0.25 * btn.width()
            new_w = file_left - (tool_left - spacer_w) + extra
            self._toolbar_lead_spacer.setFixedWidth(max(0, int(round(new_w))))
        except Exception:
            pass

    def showEvent(self, event) -> None:
        super().showEvent(event)
        if not getattr(self, "_toolbar_aligned", True):
            self._toolbar_aligned = True
            QTimer.singleShot(0, self._align_toolbar_to_menu)

    def closeEvent(self, event) -> None:
        """
        On shutdown, save prefs and close every child window this viewer
        owns. Also optionally terminate any ParaView subprocesses we spawned.

        Behaviour is controlled by the pref ``file.close_paraview_on_exit``
        (default ``True``). Set it to ``False`` to let ParaView survive.
        """
        app = QApplication.instance()
        if app is not None:
            try:
                app.removeEventFilter(self)
            except Exception:
                pass
        self._state.save_prefs()
        self._close_all_child_windows()
        close_paraview = bool(
            self._state.prefs.get("file", {}).get("close_paraview_on_exit", True)
        )
        if close_paraview:
            self._terminate_paraview_processes()
        try:
            from .console_window import restore_redirection
            restore_redirection()
        except Exception:
            pass
        super().closeEvent(event)

    # ------------------------------------------------------------------
    # Cleanup helpers
    # ------------------------------------------------------------------

    def _close_all_child_windows(self) -> None:
        """
        Close every floating / tool window this main window has created.

        Safe to call repeatedly. Uses ``close()`` (not ``deleteLater``) so
        each child runs its own ``closeEvent`` (saves its settings, etc.)
        before being torn down by Qt on the main window's destruction.
        """
        # Per-dataset windows
        for win in list(self._data_windows.values()):
            try: win.close()
            except Exception: pass
        for win in list(self._render_windows.values()):
            try: win.close()
            except Exception: pass
        for win in list(self._tiff_windows.values()):
            try: win.close()
            except Exception: pass
        for mapping in (self._scatter_windows, self._histogram_windows, self._attr_windows):
            for win in list(mapping.values()):
                try: win.close()
                except Exception: pass
        for win in list(self._filter_dlgs.values()):
            try: win.close()
            except Exception: pass
        try:
            self._state.mfv.close_windows()
        except Exception:
            pass

        # Singleton tool windows
        for attr in (
            "_attr_3d_mpl_win",
            "_filter_dlg",  "_ds_manager",
            "_log_win",     "_console_win", "_memory_win", "_roi_manager_win",
            "_script_editor_win",
        ):
            w = getattr(self, attr, None)
            if w is not None:
                try:
                    if hasattr(w, "force_close"):
                        w.force_close()
                    else:
                        w.close()
                except Exception: pass

        # Plugin dialogs that stashed themselves on the main window
        plugin_attr = "_plugin_msr_reader_dialog"
        w = getattr(self, plugin_attr, None)
        if w is not None:
            try: w.close()
            except Exception: pass

        # Modeless analysis/plugin windows retained via ui/modeless.py
        try:
            from .modeless import close_modeless
            close_modeless(self)
        except Exception:
            pass

    def _terminate_paraview_processes(self) -> None:
        """
        Politely stop every ParaView subprocess spawned during this session.

        ParaView is launched in *detached* mode so it keeps running after
        the viewer normally. When the user wants the viewer to act as the
        owner of its children, we clean those subprocesses up here.
        """
        import subprocess  # stdlib, no deps
        if not self._paraview_procs:
            return

        still_alive = [p for p in self._paraview_procs if p.poll() is None]
        if not still_alive:
            return

        self._state.log(
            f"Terminating {len(still_alive)} ParaView process(es)…",
            "INFO",
        )
        # Graceful shutdown first
        for p in still_alive:
            try: p.terminate()
            except Exception: pass
        # Give each a couple of seconds, then kill anything still running
        for p in still_alive:
            try:
                p.wait(timeout=2)
            except subprocess.TimeoutExpired:
                try:
                    p.kill()
                except Exception: pass
            except Exception:
                pass
        self._paraview_procs.clear()



# ---------------------------------------------------------------------------
# Helper: raise existing window or create a new one
# ---------------------------------------------------------------------------

def _raise_or_create(existing, cls, state: AppState):
    """Return *existing* (raised) if still alive, else create a new instance."""
    if existing is not None:
        try:
            if not existing.isHidden():
                existing.raise_()
                existing.activateWindow()
                return existing
        except RuntimeError:
            existing = None
    win = cls(state)
    win.show()
    return win


# ---------------------------------------------------------------------------
# Welcome / drop-target widget
# ---------------------------------------------------------------------------

class _WelcomeWidget(QWidget):
    """
    Central drop-target widget.

    Drop events are forwarded to the parent MainWindow, which is the only
    widget that calls setAcceptDrops(True).  Child widgets must NOT call
    setAcceptDrops(True) — doing so causes them to silently absorb the drop
    without forwarding it, so nothing ever loads.
    """

    def __init__(self, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self.setMinimumSize(500, 80)

        layout = QVBoxLayout(self)
        layout.setContentsMargins(20, 16, 20, 16)

        hint = QLabel(
            "Drag files or folders here  ·  supported: .mat  .npy  .csv  .msr  ·  or use  File › Open"
        )
        hint.setAlignment(Qt.AlignmentFlag.AlignCenter)
        hint.setStyleSheet("color: gray; font-size: 12px;")
        # Prevent the label from intercepting mouse/drag events
        hint.setAttribute(Qt.WidgetAttribute.WA_TransparentForMouseEvents)
        layout.addWidget(hint)

    def dragEnterEvent(self, event: "QDragEnterEvent") -> None:  # type: ignore[override]
        if self.parent() is not None:
            self.parent().dragEnterEvent(event)
        else:
            event.ignore()

    def dragMoveEvent(self, event) -> None:  # type: ignore[override]
        if self.parent() is not None:
            self.parent().dragMoveEvent(event)
        else:
            event.ignore()

    def dropEvent(self, event: "QDropEvent") -> None:  # type: ignore[override]
        if self.parent() is not None:
            self.parent().dropEvent(event)
        else:
            event.ignore()
