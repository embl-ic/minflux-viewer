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
    QImage,
    QKeySequence,
    QPainter,
    QPalette,
    QPixmap,
    QShortcut,
    QTransform,
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

from ..core import formats as _formats

#: Everything the application will attempt to open, from the one format
#: registry (:mod:`minflux_viewer.core.formats`).
_SUPPORTED_EXTS: tuple[str, ...] = _formats.supported_extensions()
#: ROI-set files (loaded into the ROI Manager, not as datasets).
_ROI_FILE_EXTS: frozenset[str] = _formats.roi_extensions()

#: Canonical loader format (see core.format_sniff.resolve_format) → method name.
_FMT_LOADERS: dict[str, str] = {
    "mat": "_load_mat", "npy": "_load_npy", "npz": "_load_npz",
    "spreadsheet": "_load_spreadsheet",
    "msr": "_open_msr_dialog", "tiff": "_load_tiff", "json": "_load_json",
    "zarr": "_load_zarr",
}


def startup_paths_from_argv(args) -> list[str]:
    """Filesystem paths to open from the command line, in order.

    Accepts any supported file **and directories** (the caller routes them
    through ``_route_path``, so a folder is scanned like a drop), and drops
    arguments that are not paths to open — option-like arguments, which would
    otherwise be probed as files, and notably the ``-psn_0_…``
    process-serial-number argument macOS passes to a bundled ``.app``.

    Before this, the startup loop matched only ``.mat/.npy/.csv/.msr`` and
    passed each straight to ``_route_file``, so ``minflux-viewer data.json``
    and ``minflux-viewer some_folder/`` silently opened nothing.
    """
    paths: list[str] = []
    for arg in args:
        text = str(arg)
        if not text or text.startswith("-"):
            continue                    # -psn_0_…, -style, -platform, …
        candidate = Path(text)
        suffix = candidate.suffix.lower()
        # A ``.zarr`` store is a directory, so check the suffix before is_dir().
        if suffix in _SUPPORTED_EXTS or candidate.is_dir():
            paths.append(text)
    return paths

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

#: Rectangle / oval / polygon families hosted on their toolbar button (same
#: Fiji-style right-click switch as the Line family). The **rotated** variant reuses
#: the base icon turned 45° (no new asset). (label, tool, icon file, icon rotation °).
#: The 2-D members draw a ``rectangle`` / ``oval`` / ``polygon`` record carrying
#: ``geometry["variant"]``; the **3-D** members (``cuboid`` / ``sphere`` /
#: ``polyhedron``) are placeholders — their icons are wired here now, their drawing
#: actions are implemented later (they are not yet in ``core.roi.ROI_TOOLS``).
_RECT_FAMILY: tuple[tuple[str, str, str, float], ...] = (
    ("Rectangle", "rectangle", "rec.png", 0.0),
    ("Rotated Rectangle", "rotated_rectangle", "rec.png", 45.0),
    ("Cuboid (3D)", "cuboid", "cube.png", 0.0),
)
_OVAL_FAMILY: tuple[tuple[str, str, str, float], ...] = (
    ("Oval", "oval", "ellipse.png", 0.0),
    ("Ellipse", "ellipse", "ellipse.png", 45.0),
    ("Sphere (3D)", "sphere", "sphere.png", 0.0),
)
_POLY_FAMILY: tuple[tuple[str, str, str, float], ...] = (
    ("Polygon", "polygon", "polygon.png", 0.0),
    ("Polyhedron (3D)", "polyhedron", "polyhedron.png", 0.0),
)

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
        self._cancelled = False

    def cancel(self) -> None:
        self._cancelled = True

    def run(self) -> None:  # noqa: N802 - Qt API
        from ..core.updater import UpdateCheckResult, check_for_update
        try:
            result = check_for_update(self._current)
        except Exception as exc:  # never let a worker exception escape
            result = UpdateCheckResult("error", self._current, None, str(exc))
        if not self._cancelled:
            self.signals.done.emit(result)


class _ZarrIoSignals(QObject):
    """Progress/result signals for a Zarr load or save running off the UI thread."""

    stage = pyqtSignal(str)
    done = pyqtSignal(object)
    failed = pyqtSignal(str)


class _ZarrIoTask(QRunnable):
    """Run one Zarr load or save on a worker thread.

    Reading and writing a MINFLUX Zarr store is **CPU**-bound, not I/O-bound:
    on a 20.6 M-row acquisition a load spends 2.0 s decompressing but 8.2 s
    building the structured array and normalizing it, and a save spends 9.6 s
    compressing plus 2.8 s hashing. A thread therefore does not make it faster
    -- it stops it freezing the window, which it can do because numpy, blosc
    and hashlib all release the GIL for the length of those operations.
    """

    def __init__(self, fn, *, description: str) -> None:
        super().__init__()
        self._fn = fn
        self.description = str(description)
        self.signals = _ZarrIoSignals()

    def run(self) -> None:  # noqa: N802 - Qt API
        try:
            result = self._fn(self.signals.stage.emit)
        except Exception as exc:                                # noqa: BLE001
            self.signals.failed.emit(str(exc))
        else:
            self.signals.done.emit(result)


class _OmeZarrCancelled(RuntimeError):
    pass


class _OmeZarrSignals(QObject):
    progress = pyqtSignal(float, str)
    done = pyqtSignal(object)
    failed = pyqtSignal(str)
    cancelled = pyqtSignal()


class _OmeZarrTask(QRunnable):
    """Run an OME-Zarr export off the UI thread with cancellable progress."""

    def __init__(self, fn) -> None:
        super().__init__()
        self._fn = fn
        self._cancelled = False
        self.signals = _OmeZarrSignals()
        self.last_percent = -1
        self.last_stage = ""

    def cancel(self) -> None:
        self._cancelled = True

    def run(self) -> None:  # noqa: N802 - Qt API
        def report(fraction: float, stage: str) -> None:
            if self._cancelled:
                raise _OmeZarrCancelled()
            self.signals.progress.emit(float(fraction), str(stage))

        try:
            result = self._fn(report)
        except _OmeZarrCancelled:
            self.signals.cancelled.emit()
        except Exception as exc:
            if self._cancelled:
                self.signals.cancelled.emit()
            else:
                self.signals.failed.emit(str(exc))
        else:
            if not self._cancelled:
                self.signals.done.emit(result)


def _human_bytes(value: int) -> str:
    amount = float(max(0, int(value)))
    for unit in ("B", "KiB", "MiB", "GiB", "TiB", "PiB"):
        if amount < 1024.0 or unit == "PiB":
            return f"{amount:.1f} {unit}" if unit != "B" else f"{int(amount)} B"
        amount /= 1024.0
    return f"{amount:.1f} PiB"


def _human_duration(seconds: float) -> str:
    seconds = max(0.0, float(seconds))
    if seconds < 60.0:
        return f"{seconds:.0f} s"
    minutes = seconds / 60.0
    if minutes < 60.0:
        return f"{minutes:.1f} min"
    return f"{minutes / 60.0:.1f} h"


def _adaptive_toolbar_pixmap(path: str) -> "QPixmap | None":
    """Load a toolbar icon without its baked-in white matte.

    The small ROI/LUT PNGs are opaque black-on-white images. Treating the white
    pixels as an alpha mask removes the visible white square on dark palettes.
    Monochrome artwork is recolored with the current button-text color so the
    same source remains legible in both light and dark modes; the colored
    LUT-picker icon keeps its original colors.
    """
    image = QImage(path)
    if image.isNull():
        return None
    image = image.convertToFormat(QImage.Format.Format_ARGB32)

    monochrome = True
    for y in range(image.height()):
        for x in range(image.width()):
            color = image.pixelColor(x, y)
            if max(color.red(), color.green(), color.blue()) - min(
                color.red(), color.green(), color.blue()
            ) > 3:
                monochrome = False
                break
        if not monochrome:
            break

    palette = QApplication.palette()
    foreground = palette.color(
        QPalette.ColorGroup.Active, QPalette.ColorRole.ButtonText
    )
    if not foreground.isValid():
        foreground = palette.color(
            QPalette.ColorGroup.Active, QPalette.ColorRole.WindowText
        )
    # Keep dark-mode strokes slightly softer than a fully opaque pure white.
    foreground_alpha = 230

    for y in range(image.height()):
        for x in range(image.width()):
            color = image.pixelColor(x, y)
            red, green, blue = color.red(), color.green(), color.blue()
            if monochrome:
                luminance = round(0.2126 * red + 0.7152 * green + 0.0722 * blue)
                coverage = max(0, min(255, 255 - luminance))
                alpha = round(color.alpha() * coverage * foreground_alpha / (255 * 255))
                image.setPixelColor(
                    x,
                    y,
                    QColor(foreground.red(), foreground.green(), foreground.blue(), alpha),
                )
            else:
                # For colored artwork, only remove the white matte while
                # preserving the original wheel colors and antialiasing.
                coverage = max(0, min(255, 255 - min(red, green, blue)))
                alpha = round(color.alpha() * coverage / 255)
                image.setPixelColor(x, y, QColor(red, green, blue, alpha))

    return QPixmap.fromImage(image)


def _adaptive_toolbar_icon(path: str) -> QIcon:
    pixmap = _adaptive_toolbar_pixmap(path)
    return QIcon(pixmap) if pixmap is not None else QIcon(path)


class MainWindow(QMainWindow):
    """Top-level application window."""

    APP_NAME = "MINFLUX Data Viewer"

    def __init__(self, state: AppState) -> None:
        super().__init__()
        self._state = state
        self._apply_menu_separator_style()
        self._data_windows: dict[int, DataWindow] = {}

        # One reusable plot window per dataset.
        self._scatter_windows: dict[int, QWidget] = {}
        self._histogram_windows: dict[int, QWidget] = {}
        self._attr_windows: dict[int, QWidget] = {}
        self._attr_cpu_windows: dict[int, QWidget] = {}
        self._scatter_win   = None       # compatibility alias: most recently raised
        self._histogram_win = None       # compatibility alias: most recently raised
        self._attr_win      = None       # compatibility alias: most recently raised
        self._attr_cpu_win  = None
        self._filter_dlg    = None
        self._filter_dlgs: dict[int | None, QWidget] = {}
        self._ds_manager    = None
        self._log_win       = None
        self._console_win   = None
        self._memory_win    = None
        self._roi_manager_win = None
        self._script_editor_win = None
        self._color_dialog = None
        self._shortcut_actions: dict[str, QAction] = {}
        self._roi_tool_actions: dict[str, QAction] = {}
        self._window_cycle_index = -1
        # One render window per dataset: {dataset_idx: RenderWindow}
        self._render_windows: dict[int, QWidget] = {}
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
        try:
            self.menuBar().setNativeMenuBar(False)
        except Exception:
            pass
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
        # Its entries are file paths, not commands — keep them out of the toolbar
        # command finder (command_finder.collect_commands skips flagged menus).
        self._recent_menu.setProperty("command_finder_exclude", True)
        # Scroll (don't wrap into columns) when the history is long. A
        # QProxyStyle takes ownership of an explicitly supplied base style, so
        # never pass the QApplication-owned QWidget.style() here.
        self._recent_menu_style = _ScrollableMenuStyle()
        self._recent_menu_style.setParent(self._recent_menu)
        self._recent_menu.setStyle(self._recent_menu_style)
        # Right-click a recent file → "Open file location" (a QMenu doesn't emit
        # customContextMenuRequested for its items, so use an event filter).
        self._recent_menu.installEventFilter(self)

        # Remember the last plot window (render/scatter) the user focused, so the
        # LUT toolbar button targets it — clicking the toolbar activates the MAIN
        # window, so QApplication.activeWindow() no longer identifies it. Using the
        # global focusChanged signal (not an event filter installed on the plot
        # windows, which perturbs pyqtgraph's window teardown).
        self._last_active_plot_window = None
        app = QApplication.instance()
        if app is not None and hasattr(app, "focusChanged"):
            app.focusChanged.connect(self._on_focus_changed)

        # Keyboard commands from Preferences are application-wide. Installing the
        # existing shortcut event filter on QApplication is what lets it see key
        # events delivered to modeless windows and dialogs (the previous
        # installation only covered the recent-menu / ROI toolbar buttons). The
        # filter still lets ordinary text-entry keys pass through when a command
        # is not allowed in an editor, and never fires while a QKeySequenceEdit
        # (the Preferences shortcut recorder) has focus.
        if app is not None:
            app.installEventFilter(self)

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
        state.colors_changed.connect(self._on_colors_changed)
        state.overlay_manual_alignment_requested.connect(
            self._on_overlay_manual_alignment_requested
        )
        self._sync_attribute_gpu_action()

        # Remembered ROI duplicate/crop options, per dataset (session-only,
        # keyed by dataset identity — "use the same setup and stop asking").
        self._roi_crop_setup: dict = {}
        # Last active ROI draft per dataset index, for Process › ROI › Restore ROI
        # (restore across views / after an accidental delete).
        self._roi_last_active: dict = {}

        # Tier-A in-app update check (opt-out via Preferences > File). Delayed so
        # it never slows startup; silent unless a newer release exists.
        self._is_shutting_down = False
        self._update_tasks: set = set()
        self._ome_zarr_tasks: set = set()
        if state.prefs.get("file", {}).get("check_updates_on_startup", False):
            QTimer.singleShot(3000, self._maybe_startup_update_check)

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
        # Spreadsheet (.csv/.tsv/.txt/.xlsx/.xlsm) and TIFF (.tif/.tiff) files open
        # via drag-and-drop (routed through _load_spreadsheet / _load_tiff); there
        # are deliberately no dedicated File-menu entries for them.
        # Sample data is a submenu of user-editable presets (Data Simulator plugin);
        # exclude from the command finder like the recent-files list.
        self.menuOpenSample = QMenu("Open Sample Data", self)
        self.menuOpenSample.setProperty("command_finder_exclude", True)
        u.actionSave.triggered.connect(self._save_data)
        self.menuSaveAs = QMenu("Save As", self)
        self.actionSaveAsMinflux = QAction("MINFLUX data formats (.mat; .npy; .json)", self)
        self.actionSaveAsMinflux.triggered.connect(self._save_as_minflux_formats)
        self.actionSaveAsMsr = QAction("MINFLUX .msr file (experimental)", self)
        self.actionSaveAsMsr.triggered.connect(
            lambda _checked=False: self._save_as_format("msr", "MINFLUX .msr file")
        )
        self.actionSaveAsSpreadsheet = QAction("Custom table (.csv)...", self)
        self.actionSaveAsSpreadsheet.triggered.connect(self._save_as_spreadsheet)
        self.actionSaveAsZarr = QAction("Zarr (.zarr v2) format", self)
        self.actionSaveAsZarr.setToolTip(
            "Save a self-contained MINFLUX Viewer Zarr v2 dataset. If the active "
            "dataset is an overlay, every channel, transform and LUT is bundled; "
            "viewer ROIs and linked MSR images are included."
        )
        self.actionSaveAsZarr.triggered.connect(
            lambda _checked=False: self._save_as_format("zarr", "Zarr v2")
        )
        self.actionSaveAsZarrZip = QAction("Zarr (.zarr.zip v2) single file", self)
        self.actionSaveAsZarrZip.setToolTip(
            "The same self-contained Zarr v2 content sealed into ONE file: raw "
            "data plus processing state, ROIs, overlay channels and linked "
            "images. It opens directly, without unpacking.\n\n"
            "Unlike the .zarr directory it cannot be updated in place — a zip "
            "cannot replace a member — so saving over a package rewrites it. "
            "Use the .zarr directory while a dataset is still being worked on."
        )
        self.actionSaveAsZarrZip.triggered.connect(
            lambda _checked=False: self._save_as_format("zarr_zip", "Zarr v2 single file")
        )
        # Picasso HDF5 is application-specific export. The writer and this
        # action are kept (scripting, and it works), but it is deliberately NOT
        # added to File > Save As: the offered set is this application's own
        # format plus the MINFLUX defaults. See BACKLOG.md > Nice to have.
        self.actionSaveAsHdf5 = QAction("HDF5...", self)
        self.actionSaveAsHdf5.triggered.connect(self._save_as_picasso_hdf5)
        self.actionSaveAsOmeTiff = QAction("OME-TIFF...", self)
        self.actionSaveAsOmeTiff.triggered.connect(self._save_as_ome_tiff)
        self.actionSaveAsOmeZarr = QAction("OME-NGFF 0.5 / Zarr v3...", self)
        self.actionSaveAsOmeZarr.triggered.connect(self._save_as_ome_zarr)
        u.actionQuit.triggered.connect(self.close)
        self.actionClose = QAction("Close Dataset", self)
        self.actionClose.setShortcut(QKeySequence("Ctrl+W"))
        self.actionClose.setShortcutContext(Qt.ShortcutContext.ApplicationShortcut)
        self.actionClose.triggered.connect(self._close_active_dataset)
        # Shift+W — close every dataset and all its windows.
        self.actionCloseAll = QAction("Close All Datasets", self)
        self.actionCloseAll.setShortcut(QKeySequence("Shift+W"))
        self.actionCloseAll.setShortcutContext(Qt.ShortcutContext.ApplicationShortcut)
        self.actionCloseAll.triggered.connect(self._close_all_datasets)
        # Ctrl+Shift+W — close everything (datasets, viewers, plugin/analysis
        # dialogs) and keep only the Log and Console windows.
        self.actionCloseAllWindows = QAction("Close All Windows", self)
        self.actionCloseAllWindows.setShortcut(QKeySequence("Ctrl+Shift+W"))
        self.actionCloseAllWindows.setShortcutContext(Qt.ShortcutContext.ApplicationShortcut)
        self.actionCloseAllWindows.triggered.connect(self._close_all_windows)

        # Edit menu
        u.actionDatasetManager.triggered.connect(self._show_dataset_manager)
        u.actionFilter.triggered.connect(self._show_filter)
        u.actionDuplicate.triggered.connect(self._duplicate_active_dataset)
        u.actionPreferences.triggered.connect(self._show_preferences)

        # View menu
        u.actionScatter.triggered.connect(self._show_scatter)
        u.actionHistogram.triggered.connect(self._show_histogram)
        u.actionAttributePlot.triggered.connect(self._show_attr_plot)
        u.actionRender.triggered.connect(self._show_render)        # was Process > Render image
        # The unified render view exposes methods from its right-click
        # View › Render Method menu.
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
        self.actionPlotProfile.triggered.connect(self._show_plot_profile)
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
        # The HlyB/D subunit pair analysis moved out of Clustering and is now a
        # single entry under *Plugins* (it is one project-specific workflow, not
        # a family of general clustering tools).  Its earlier menu entries —
        # "2D"/"3D", "Pair-distance model fit (2D/3D)" and "Template matching
        # (2D/3D)" — are retired, but every one of those analysis modules is
        # deliberately KEPT (analysis/hlyb_clustering.py, analysis/hlyb_pairwise.py,
        # ui/hlyb_clustering_dialog.py, ui/hlyb_pairwise_dialog.py) together with
        # the E. coli cell delineation, so a 2-D workflow or a template
        # comparison can be rebuilt on them.  See CLAUDE.md for why each was
        # retired.  The handlers below them (_run_hlyb_pair_analysis,
        # _run_hlyb_pairwise_analysis) are likewise kept and remain callable.
        self.menuAnalyzeClustering.addAction(self.actionDbscan)
        self.menuAnalyzeClustering.addAction(self.actionKNearestNeighbour)

        # Trace submenu — trace-specific measurements.
        self.menuAnalyzeTrace = QMenu("Trace", self)
        self.actionTraceSize = QAction("Estimate Average Trace Size", self)
        self.actionTraceSize.triggered.connect(self._run_trace_size)
        self.actionTraceAnisotropy = QAction("Estimate Z Scaling Factor", self)
        self.actionTraceAnisotropy.triggered.connect(self._run_trace_anisotropy)
        self.menuAnalyzeTrace.addAction(self.actionTraceSize)
        self.menuAnalyzeTrace.addAction(self.actionTraceAnisotropy)

        # Segmentation submenu — structure segmentation.
        # The former NPC › 2D / 3D entries were removed: 2D was the same ring
        # kernel + peak finding as Convolution's `ring` model (its one distinct
        # criterion, the ring support score, now lives in that tool's ring
        # validation), and 3D was never more than a placeholder.
        self.menuAnalyzeSegmentation = QMenu("Segmentation", self)
        self.actionSegConvolution = QAction("Convolution…", self)
        self.actionSegConvolution.triggered.connect(self._show_conv_segmentation)
        self.menuAnalyzeSegmentation.addAction(self.actionSegConvolution)
        self.actionSegConvolution3D = QAction("Convolution (3D)…", self)
        self.actionSegConvolution3D.triggered.connect(self._show_conv_segmentation_3d)
        self.menuAnalyzeSegmentation.addAction(self.actionSegConvolution3D)
        # Named for the method (a known shape model is fitted), not for one
        # geometry, so new geometries arrive as entries in the tool's Shape
        # dropdown rather than as new menu items.
        self.actionSegShapeModel = QAction("Shape Model…", self)
        self.actionSegShapeModel.triggered.connect(self._show_shape_segmentation)
        self.menuAnalyzeSegmentation.addAction(self.actionSegShapeModel)
        self.actionSegCurvilinear = QAction("Curvilinear Structures…", self)
        self.actionSegCurvilinear.triggered.connect(self._show_curvilinear_segmentation)
        self.menuAnalyzeSegmentation.addAction(self.actionSegCurvilinear)
        self.actionSegStraightenedVolume = QAction("Straightened Volume along Skeleton...", self)
        self.actionSegStraightenedVolume.triggered.connect(self._show_straightened_volume_skeleton)
        self.menuAnalyzeSegmentation.addAction(self.actionSegStraightenedVolume)
        self.actionSegParticleAverage = QAction("Particle Average…", self)
        self.actionSegParticleAverage.triggered.connect(self._show_particle_average)
        self.menuAnalyzeSegmentation.addAction(self.actionSegParticleAverage)

        # Tracking submenu — Phase 5 placeholders
        u.actionParticleTracking.triggered.connect(
            lambda: self._placeholder("Particle tracking", "Phase 5")
        )
        # actionTraceViewer (legacy View/Tracking entry) retired. The retained
        # trace inspection workflow lives in Plugins > Trace Viewer.
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
        self.actionChannelFlatten = QAction("Flatten", self)
        self.actionChannelFlatten.triggered.connect(self._flatten_active_channel_group)
        # Convert Dataset to Multi-Channel Overlay — the unified home for the
        # separation tools (by DCR / by Time Window / by any attribute) plus the
        # inverse (Revert). DCR / Time keep their existing behaviour; only their
        # menu home + labels moved here.
        self.menuConvertOverlay = QMenu("Convert Dataset to Multi-Channel Overlay", self)
        self.actionChannelSeparateDcr = QAction("by DCR: detection channel ratio", self)
        self.actionChannelSeparateDcr.triggered.connect(self._show_channel_separation)
        self.actionChannelSeparateTime = QAction("by Time Window", self)
        self.actionChannelSeparateTime.triggered.connect(
            self._show_time_channel_separation
        )
        self.actionChannelSeparateAttribute = QAction("by MINFLUX data attribute...", self)
        self.actionChannelSeparateAttribute.triggered.connect(self._show_attribute_separation)
        self.actionRevertOverlay = QAction("Revert Overlay to Original Dataset", self)
        self.actionRevertOverlay.triggered.connect(self._revert_overlay_to_original)
        self.menuConvertOverlay.addAction(self.actionChannelSeparateDcr)
        self.menuConvertOverlay.addAction(self.actionChannelSeparateTime)
        self.menuConvertOverlay.addAction(self.actionChannelSeparateAttribute)
        self.menuConvertOverlay.addSeparator()
        self.menuConvertOverlay.addAction(self.actionRevertOverlay)

        self.menuProcessChannel.addAction(self.actionChannelTool)
        self.menuProcessChannel.addAction(self.actionChannelCombine)
        self.menuProcessChannel.addAction(self.actionChannelSplit)
        self.menuProcessChannel.addAction(self.actionChannelFlatten)
        self.menuProcessChannel.addSeparator()
        self.menuProcessChannel.addMenu(self.menuConvertOverlay)

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
            ("to Bounding Box", "bounding_box"),
            ("to Rectangle", "rectangle"),
            ("to Oval", "oval"),
            ("to Ellipse", "ellipse"),
            ("Line to Region", "region"),
            ("Region to Line", "line"),
            ("to Point", "point"),
        ):
            act = QAction(label, self)
            act.triggered.connect(lambda _checked=False, t=target: self._convert_active_roi(t))
            self.menuRoiConvert.addAction(act)
            self._roi_convert_actions[target] = act
        self.menuProcessRoi.addMenu(self.menuRoiConvert)
        # Fit submenu (right under Convert): fit a shape to the localizations the
        # active region ROI highlights; Spline/Interpolate act on the ROI outline.
        self.menuRoiFit = QMenu("Fit", self)
        self._roi_fit_actions = {}
        for label, target in (
            ("Fit Rectangle", "rectangle"),
            ("Fit Circle", "circle"),
            ("Fit Ellipse", "ellipse"),
            ("Fit Polygon", "polygon"),
            ("Fit Convex Hull", "convex_hull"),
        ):
            act = QAction(label, self)
            act.triggered.connect(lambda _checked=False, t=target: self._fit_active_roi(t))
            self.menuRoiFit.addAction(act)
            self._roi_fit_actions[target] = act
        self.menuRoiFit.addSeparator()
        self.actionRoiFitSpline = QAction("Fit Spline", self)
        self.actionRoiFitSpline.triggered.connect(self._spline_fit_active_roi)
        self.menuRoiFit.addAction(self.actionRoiFitSpline)
        self.actionRoiInterpolate = QAction("Interpolate", self)
        self.actionRoiInterpolate.triggered.connect(self._interpolate_active_roi)
        self.menuRoiFit.addAction(self.actionRoiInterpolate)
        self.menuProcessRoi.addMenu(self.menuRoiFit)

        self.actionRoiResize = QAction("Enlarge / Shrink…", self)
        self.actionRoiResize.triggered.connect(self._resize_active_roi)
        self.menuProcessRoi.addAction(self.actionRoiResize)
        self.actionRoiSkeletonize = QAction("Skeletonize", self)
        self.actionRoiSkeletonize.triggered.connect(self._skeletonize_active_roi)
        self.menuProcessRoi.addAction(self.actionRoiSkeletonize)
        self.actionRoiConvexHull = QAction("Convex Hull", self)
        self.actionRoiConvexHull.triggered.connect(self._convex_hull_active_roi)
        self.menuProcessRoi.addAction(self.actionRoiConvexHull)
        self.menuProcessRoi.addSeparator()
        # Restore ROI: bring the active ROI back on another view / after a delete.
        self.actionRoiRestore = QAction("Restore ROI", self)
        self.actionRoiRestore.triggered.connect(self._restore_roi)
        self.menuProcessRoi.addAction(self.actionRoiRestore)
        self.menuProcessRoi.addSeparator()
        self.actionRoi3D = QAction("3D ROI", self)
        self.actionRoi3D.triggered.connect(self._show_roi_3d)
        self.menuProcessRoi.addAction(self.actionRoi3D)

        # Aggregate localizations (Imspector-style per-trace photon binning).
        self.actionAggregate = QAction("Aggregate Localizations…", self)
        self.actionAggregate.triggered.connect(self._aggregate_active_dataset)

        # Attribute Plot renderer switch. Lives in the app View menu as well as
        # the plot's own right-click menu, so the two renderers can be compared
        # without hunting for the context menu.
        self.actionAttributeGpu = QAction(
            "Attribute Plot: GPU rendering (OpenGL, experimental)", self
        )
        self.actionAttributeGpu.setCheckable(True)
        self.actionAttributeGpu.setToolTip(
            "Draw exact Attribute Plot markers with OpenGL when startup probing\n"
            "and the current memory-derived upload budget permit it. Axes, ROI,\n"
            "zoom and connecting lines remain in the pyqtgraph overlay."
        )
        self.actionAttributeGpu.triggered.connect(self._toggle_attribute_gpu)
        self.actionAttributeCpu = QAction("Attribute Plot (CPU fix)", self)
        self.actionAttributeCpu.setObjectName("actionAttributeCpu")
        self.actionAttributeCpu.setToolTip(
            "Open a separate, non-OpenGL Attribute Plot. Sparse views use Qt "
            "bulk point painting; dense views aggregate every visible row into "
            "a display-sized count grid (and mean C per cell)."
        )
        self.actionAttributeCpu.triggered.connect(self._show_attr_plot_cpu)

        # Help menu
        u.actionAbout.triggered.connect(self._show_about)
        u.actionMemoryMonitor.triggered.connect(self._show_memory_monitor)
        self.actionCommandFinder = QAction("Command Finder…", self)
        self.actionCommandFinder.setStatusTip("Search all menu commands (Fiji-style)")
        self.actionCommandFinder.triggered.connect(lambda: self._open_command_finder(""))
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
        # Fiji-style: the Line / Rectangle / Oval / Polygon toolbar buttons each
        # host a family. Right-click switches variants; left-click activates the
        # current variant, tracked by ``_*_variant``.
        self._line_variant = "line"
        self._rect_variant = "rectangle"
        self._oval_variant = "oval"
        self._poly_variant = "polygon"
        _variant_getters = {
            "line": lambda: self._line_variant,
            "rectangle": lambda: self._rect_variant,
            "oval": lambda: self._oval_variant,
            "polygon": lambda: self._poly_variant,
        }
        for _label, tool, attr in _ROI_TOOL_DEFS:
            action = getattr(u, attr)
            self._roi_tool_actions[tool] = action
            self._roi_tool_group.addAction(action)
            getter = _variant_getters.get(tool)
            if getter is not None:
                action.triggered.connect(lambda checked, g=getter: self._on_roi_tool(g(), checked))
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
        u.actionSave.setText("Save...")
        u.actionQuit.setText("Quit")
        u.actionDatasetManager.setText("Dataset Manager")
        u.actionFilter.setText("Filter...")
        u.actionDuplicate.setText("Duplicate")
        u.actionPreferences.setText("Preferences...")
        u.actionHistogram.setText("Attribute Histogram")
        u.actionScatter.setText("Loc Scatter Plot")
        u.actionAttributePlot.setText("Attribute Plot")
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
        self.menuConvertOverlay.setTitle("Convert Dataset to Multi-Channel Overlay")
        self.actionChannelSeparateDcr.setText("by DCR: detection channel ratio")
        self.actionChannelSeparateTime.setText("by Time Window")
        self.actionChannelSeparateAttribute.setText("by MINFLUX data attribute...")
        self.actionRevertOverlay.setText("Revert Overlay to Original Dataset")
        self.menuProcessRoi.setTitle("ROI")
        self.actionRoiManager.setText("ROI Manager")
        u.menuOpenRecent.setTitle("Open Recent")
        self.menuSaveAs.setTitle("Save As")
        self.actionSaveAsMinflux.setText("MINFLUX data formats (.mat; .npy; .json)")
        self.actionSaveAsMsr.setText("MINFLUX .msr file (experimental)")
        self.actionSaveAsSpreadsheet.setText("Custom table (.csv)...")
        self.actionSaveAsZarr.setText("Zarr (.zarr v2) format")
        self.actionSaveAsZarrZip.setText("Zarr (.zarr.zip v2) single file")
        self.actionSaveAsHdf5.setText("HDF5...")
        self.actionSaveAsOmeTiff.setText("OME-TIFF...")
        self.actionSaveAsOmeZarr.setText("OME-NGFF 0.5 / Zarr v3...")
        u.menuBatchProcessing.setTitle("Batch Processing")
        u.menuAnalysis.setTitle("Analyze")
        u.menuLocPrecision.setTitle("Localization Precision")
        u.menuTracking.setTitle("Tracking")
        self._keep_standard_actions_in_declared_menus()

        u.menuFile.clear()
        u.menuFile.addAction(u.actionOpen)
        u.menuFile.addAction(self.menuOpenSample.menuAction())
        u.menuFile.addAction(u.menuOpenRecent.menuAction())
        u.menuFile.addSeparator()
        u.menuFile.addAction(self.actionClose)
        u.menuFile.addAction(self.actionCloseAll)
        u.menuFile.addAction(self.actionCloseAllWindows)
        u.menuFile.addSeparator()
        u.menuFile.addAction(u.actionSave)
        self.menuSaveAs.clear()
        self.menuSaveAs.addAction(self.actionSaveAsMinflux)
        self.menuSaveAs.addAction(self.actionSaveAsMsr)
        self.menuSaveAs.addAction(self.actionSaveAsSpreadsheet)
        self.menuSaveAs.addAction(self.actionSaveAsZarr)
        self.menuSaveAs.addAction(self.actionSaveAsZarrZip)
        self.menuSaveAs.addAction(self.actionSaveAsOmeTiff)
        self.menuSaveAs.addAction(self.actionSaveAsOmeZarr)
        u.menuFile.addAction(self.menuSaveAs.menuAction())
        u.menuFile.addSeparator()
        u.menuFile.addAction(u.actionQuit)
        self._rebuild_sample_menu()          # populate File › Open Sample Data presets

        u.menuEdit.clear()
        u.menuEdit.addAction(u.actionDatasetManager)
        u.menuEdit.addAction(u.actionFilter)
        u.menuEdit.addSeparator()
        u.menuEdit.addAction(u.actionDuplicate)
        u.menuEdit.addSeparator()
        u.menuEdit.addAction(u.actionPreferences)

        u.menuView.clear()
        u.menuView.addAction(u.actionShowInfo)
        u.menuView.addSeparator()
        u.menuView.addAction(u.actionAttributePlot)
        u.menuView.addAction(self.actionAttributeCpu)
        u.menuView.addAction(u.actionHistogram)
        u.menuView.addAction(u.actionScatter)
        u.menuView.addAction(u.actionRender)
        u.menuView.addSeparator()
        u.menuView.addAction(self.actionAttributeGpu)
        u.menuView.addSeparator()
        u.menuView.addAction(u.actionLog)
        u.menuView.setToolTipsVisible(True)
        u.menuView.aboutToShow.connect(self._sync_attribute_gpu_action)

        u.menuProcess.clear()
        u.menuProcess.addAction(self.menuProcessChannel.menuAction())
        u.menuProcess.addAction(self.menuProcessRoi.menuAction())
        u.menuProcess.addAction(self.actionAggregate)
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

        u.menuHelp.clear()
        u.menuHelp.addAction(u.actionConsole)
        u.menuHelp.addAction(u.actionMemoryMonitor)
        cf = getattr(self, "actionCommandFinder", None)
        if cf is not None:                              # between Monitor Memory & Updates
            u.menuHelp.addAction(cf)
        check_updates = getattr(self, "actionCheckUpdates", None)
        if check_updates is not None:
            u.menuHelp.addAction(check_updates)
        u.menuHelp.addSeparator()
        u.menuHelp.addAction(u.actionAbout)

        self._shortcut_actions = {
            "open": u.actionOpen,
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

    def _keep_standard_actions_in_declared_menus(self) -> None:
        """Do not let macOS TextHeuristicRole relocate these actions.

        Qt's default TextHeuristicRole recognises labels such as About,
        Preferences and Quit and moves them into the macOS application menu.
        For this Fiji-style viewer we keep the actions in our declared menus so
        File/Edit/Help match across platforms.
        """
        try:
            role = QAction.MenuRole.NoRole
        except Exception:
            return
        for action in (
            self._ui.actionAbout,
            self._ui.actionPreferences,
            self._ui.actionQuit,
        ):
            try:
                action.setMenuRole(role)
            except Exception:
                pass

    def _mark_ai_unapproved_actions(self) -> None:
        """Visually separate AI-generated/unapproved actions from approved UI."""
        u = self._ui
        actions = [
            #u.actionSave,
            #self.menuProcessChannel.menuAction(),
            self.actionChannelTool,
            #self.actionChannelCombine,
            #self.actionChannelSplit,
            #self.actionChannelSeparateDcr,
            #self.menuProcessRoi.menuAction(),
            #self.actionRoiManager,
            #self.menuRoiConvert.menuAction(),
            #self._roi_convert_actions.values(),
            #self.actionRoiResize,
            #self.actionRoiSkeletonize,
            #self.actionRoiConvexHull,
            self.actionRoi3D,
            u.menuBatchProcessing.menuAction(),
            u.actionBatchRender,
            u.actionBatchExport,
            u.actionBatchFilter,
            #self.menuMeasure.menuAction(),
            #self.actionPlotProfile,
            #self.actionSetMeasurements,
            self.actionDbscan,
            self.actionKNearestNeighbour,
            #self.menuAnalyzeSegmentation.menuAction(),
            #self.actionSegConvolution,
            #self.actionSegCurvilinear,
            #self.actionSegParticleAverage,
            u.menuTracking.menuAction(),
            #u.actionParticleTracking,
            u.actionMsdAnalysis,
            u.actionMemoryMonitor,
        ]
        for action in actions:
            self._mark_action_ai_unapproved(action)

    def _mark_action_ai_unapproved(self, action: QAction) -> None:
        """Flag an AI-generated/unapproved action with a status-bar hover note.

        The action's appearance is left untouched — approved and unapproved
        menu entries render identically; only the status-bar tip differs.
        """
        tip = action.statusTip() or action.toolTip()
        note = "AI-generated; pending human approval."
        action.setStatusTip(f"{tip} {note}".strip() if tip else note)
        self.addAction(action)

    def _apply_shortcuts(self) -> None:
        shortcuts = self._state.prefs.get("shortcuts", {})
        for key, action in self._shortcut_actions.items():
            seq = str(shortcuts.get(key, "") or "")
            action.setShortcut(QKeySequence(seq) if seq else QKeySequence())
            # Keep the shortcut visible in menus and available through the
            # normal main-window action path. The QApplication event filter
            # dispatches the same command when another window or dialog owns
            # focus.
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
                "scatter_window",
            }:
                self._install_window_shortcuts(widget)

    def _trigger_shortcut_command(self, command: str) -> None:
        # QKeySequenceEdit must receive the key sequence it is recording; it is
        # the one editor in which a preference shortcut is intentionally being
        # entered rather than invoked.
        if self._focus_is_shortcut_editor():
            return
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
        # Right-click on a recent-file entry → "Open file location".
        if (event.type() == QEvent.Type.MouseButtonPress
                and event.button() == Qt.MouseButton.RightButton
                and obj is getattr(self, "_recent_menu", None)):
            pos = event.position().toPoint() if hasattr(event, "position") else event.pos()
            act = self._recent_menu.actionAt(pos)
            if act is not None and act.isEnabled() and act.data():
                gpos = (event.globalPosition().toPoint()
                        if hasattr(event, "globalPosition") else event.globalPos())
                self._show_recent_file_menu(str(act.data()), gpos)
                return True
        # Right-click on a ROI toolbar button → tool/variant switch menu.
        if (event.type() == QEvent.Type.MouseButtonPress
                and event.button() == Qt.MouseButton.RightButton
                and obj in getattr(self, "_roi_tool_buttons", {})):
            pos = event.position().toPoint() if hasattr(event, "position") else event.pos()
            key = self._roi_tool_buttons[obj]
            if key == "line":
                self._show_line_family_menu(obj, pos)
            elif key in ("rectangle", "oval", "polygon"):
                self._show_shape_family_menu(obj, pos, key)
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
                if self._focus_is_shortcut_editor():
                    return super().eventFilter(obj, event)
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
        if self._focus_is_shortcut_editor():
            return False
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

    @staticmethod
    def _focus_is_shortcut_editor() -> bool:
        focus = QApplication.focusWidget()
        while focus is not None:
            if isinstance(focus, QKeySequenceEdit):
                return True
            focus = focus.parentWidget()
        return False

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

    def _open_msr_dialog(self, msr_path: str) -> None:
        """Hand an ``.msr`` file to the MSR Reader plugin (drag-drop entry point).

        **Every** ``.msr`` opens in the MSR reader — including image-only files
        with no MINFLUX data. Their OBF image series are viewable from the reader
        (right-click a Series node › *Preview* / *Open as image*, or select them in
        *Datasets / Fields included…* and *Open in MINFLUX viewer*)."""
        if not msr_path:
            return
        from .msr_import_dialog import open_msr
        open_msr(msr_path, self._state, parent=self)

    # ------------------------------------------------------------------
    # Sample data (Data Simulator) — File › Open Sample Data presets
    # ------------------------------------------------------------------
    def sample_presets(self) -> list:
        from ..core.sample_presets import load_presets
        return load_presets(self._state.prefs)

    def set_sample_presets(self, presets: list) -> None:
        """Persist the Sample-Data presets and rebuild the File submenu."""
        from ..core.sample_presets import save_presets
        save_presets(self._state.prefs, presets)
        self._state.save_prefs()
        self._rebuild_sample_menu()

    def _rebuild_sample_menu(self) -> None:
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

    def _open_data_simulator(self) -> None:
        from .data_simulator_window import DataSimulatorWindow
        from .modeless import show_modeless
        show_modeless(DataSimulatorWindow(self._state, owner=self), self)

    def _finalize_sim_dataset(self, name, coords, tid, attrs):
        """Build one simulated dataset from arrays (version = 'simulation', Z scaling factor pinned)."""
        import uuid

        from ..core.dataset import build_localization_dataset
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

    def _build_simulated_dataset(self, name: str, kwargs: dict):
        from ..core.simulate import simulate_localizations
        coords, tid, attrs = simulate_localizations(**kwargs)
        return self._finalize_sim_dataset(name, coords, tid, attrs)

    def generate_simulated(self, config: dict, *, include_beads: bool = False,
                           bead_count: int = 4) -> None:
        """Generate the dataset(s) for a Data-Simulator config / Sample-Data preset.
        ``channels > 1`` builds a multi-channel overlay; the NPC overlay / DCR
        multi-sims build a co-registered overlay / a single DCR-mixed dataset.

        When ``include_beads`` is set, one shared set of synthetic fiducial beads
        is generated for this call and attached to every produced dataset (so
        saving to .msr writes ``grd/mbm/points``)."""
        from ..core.sample_presets import normalize_preset
        from ..core.simulate import sim_kind
        p = normalize_preset(config)
        self._sim_bead_spec = None
        if include_beads:
            try:
                from ..core.simulate import simulate_beads
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
            else:
                self._generate_sim_single(p)
        finally:
            self._sim_bead_spec = None

    def _generate_sim_overlay(self, p: dict) -> None:
        """NPC 3-channel overlay: co-registered channels on one shared scaffold."""
        import uuid

        from ..core.overlay import overlay_color_cycle
        from ..core.simulate import simulate_npc_overlay
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

    def _generate_sim_dcr(self, p: dict) -> None:
        """NPC 2-channel by DCR: one dataset, two reporters via a bimodal ``dcr``."""
        from ..core.simulate import simulate_npc_dcr
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

    def _generate_sim_ecoli(self, p: dict) -> None:
        """One rod cell of labelled HlyB dimers — a known-distance control."""
        from ..core.simulate import simulate_ecoli_hlyb
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

    def _generate_sim_single(self, p: dict) -> None:
        """Generate a single-structure sample (``channels > 1`` = independent overlay)."""
        import uuid

        from ..core.overlay import overlay_color_cycle
        from ..core.sample_presets import simulate_kwargs
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

    def _route_file(self, path: str) -> None:
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
                # A recipe needs a dataset to act on; dropped on the main window
                # there is no target. Dropping it on a dataset row applies it.
                self._state.log(
                    f"'{p.name}' is a processing metadata sidecar, not loadable "
                    f"data — drop it on a dataset row to apply it.", "WARN")
                self._status_label.setText(f"Skipped: {p.name} (metadata sidecar)")
                return
            if action is _formats.OpenAction.MSR_READER:
                self._open_msr_dialog(path)
                return
            if action is _formats.OpenAction.IMAGE_VIEWER:
                self._load_tiff(path)
                return

        # Dataset and spreadsheet paths still go through content sniffing, which
        # resolves an extension that disagrees with the bytes (a .npz that is
        # really an .xlsx, and so on) and reports the mismatch.
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
                f".xlsx, .xlsm, .msr, .tif, .tiff, .json, .zarr"
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
                for f in found:
                    self._route_file(str(f))
            else:
                msg = (
                    f"Folder '{p.name}' contains no supported files "
                    f"(.mat, .npy, .csv, .tsv, .xlsx, .xlsm, .msr, .tif, .tiff, .json, .zarr)."
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

    def _load_npz(self, path: str) -> None:
        self._status_label.setText(f"Loading {Path(path).name}…")
        self._state.log(f"Opening .npz: {path}", "INFO")
        try:
            from ..core.loader import load_npz
            dataset = load_npz(path, prefs=self._state.prefs)
            self._state.add_dataset(dataset)
        except Exception as exc:
            self._state.log(f"Failed to load '{Path(path).name}': {exc}", "ERROR")
            QMessageBox.critical(self, "Load error", str(exc))
            self._status_label.setText("Load failed.")

    def _load_zarr(self, path: str) -> None:
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
            from ..core.minflux_zarr import load_minflux_zarr_project

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

    def _begin_zarr_io(self, task) -> None:
        """Start a Zarr load/save task and keep it alive until it finishes."""
        from PyQt6.QtCore import QThreadPool

        self._status_label.setText(f"{task.description}…")
        self._state.status_progress(task.description)
        tasks = getattr(self, "_zarr_io_tasks", None)
        if tasks is None:
            tasks = self._zarr_io_tasks = []
        tasks.append(task)
        for signal in (task.signals.done, task.signals.failed):
            signal.connect(lambda *_a, _t=task: self._finish_zarr_io(_t))
        QThreadPool.globalInstance().start(task)

    def _finish_zarr_io(self, task) -> None:
        try:
            getattr(self, "_zarr_io_tasks", []).remove(task)
        except ValueError:
            pass

    def _on_zarr_io_failed(self, name: str, message: str, title: str) -> None:
        self._state.log(f"Failed: '{name}': {message}", "ERROR")
        if not self._is_shutting_down:
            QMessageBox.critical(self, title, message)
        self._status_label.setText(f"{title}.")

    def _on_zarr_loaded(self, project, name: str) -> None:
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

    def _restore_zarr_rois(self, records, manifest: dict, dataset_indices) -> None:
        """Restore portable project ROI records into the session-level store."""
        if not records:
            return
        from ..core.roi import RoiRecord

        members = list(manifest.get("datasets") or [])
        id_to_idx = {
            str(spec.get("id")): int(idx)
            for spec, idx in zip(members, dataset_indices)
        }
        existing = {record.id for record in self._state.rois.records}
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

    def _load_csv(self, path: str) -> None:
        # CSV/TSV/TXT all go through the smart spreadsheet importer.
        self._load_spreadsheet(path)

    def _load_spreadsheet(self, path: str) -> None:
        # Always open the column-mapping dialog (pre-filled from a value-based
        # best guess of x/y/z/tid/tim + units) so the user confirms/corrects the
        # mapping before importing — never a silent auto-import.
        self._status_label.setText(f"Loading {Path(path).name}…")
        self._state.log(f"Opening spreadsheet: {path}", "INFO")
        try:
            from .spreadsheet_import_dialog import import_spreadsheet
            dataset = import_spreadsheet(path, prefs=self._state.prefs, parent=self)
            if dataset is None:
                self._status_label.setText("Spreadsheet import cancelled.")
                return
            self._state.log(
                f"Imported '{Path(path).name}' "
                f"({dataset.prop.num_loc:,} localizations).", "INFO")
            self._state.add_dataset(dataset)
        except Exception as exc:
            self._state.log(f"Failed to load '{Path(path).name}': {exc}", "ERROR")
            QMessageBox.critical(self, "Load error", str(exc))
            self._status_label.setText("Load failed.")

    def _open_image_viewer(self, source, key: str, *, initial_series_index: int | None = None) -> None:
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
        from .tiff_viewer_window import TiffViewerWindow
        win = TiffViewerWindow(source)
        win.destroyed.connect(lambda _=None, k=key: self._tiff_windows.pop(k, None))
        self._tiff_windows[key] = win
        win.show()
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

    def _load_tiff(self, path: str) -> None:
        self._status_label.setText(f"Opening TIFF {Path(path).name}…")
        self._state.log(f"Opening TIFF: {path}", "INFO")
        resolved = str(Path(path).resolve())
        try:
            from ..core.tiff_source import TiffImageSource

            # One window per file; multi-series files switch via the in-window
            # Series dropdown (no separate chooser dialog).
            self._open_image_viewer(TiffImageSource(path), f"{resolved}#img")
            try:
                self._state._record_recent(resolved)
                self._populate_recent_menu()
            except Exception:
                pass
            self._status_label.setText(f"TIFF opened: {Path(path).name}")
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
        source = Path(path)
        if source.stat().st_size >= 1 << 30:
            from ..core.export_size import format_file_size

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
                        "This is a MINFLUX-viewer metadata sidecar (Z scaling factor, transform, "
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
                candidate = Path(path)
                if candidate.is_file() or (
                    candidate.is_dir() and candidate.suffix.lower() == ".zarr"
                ):
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
            act.setData(path)                        # for the right-click location menu
            act.triggered.connect(lambda checked, p=path: self._route_path(p))
            self._recent_menu.addAction(act)

    def _show_recent_file_menu(self, path: str, global_pos) -> None:
        """Right-click context menu for a File › Open Recent entry."""
        menu = QMenu(self)
        act_open = menu.addAction("Open")
        act_loc = menu.addAction("Open file location")
        chosen = menu.exec(global_pos)
        if chosen is act_open:
            self._route_file(path)
        elif chosen is act_loc:
            self.open_file_location(path)

    def open_file_location(self, path: str) -> None:
        """Reveal a file in the OS file manager (falls back to opening its
        containing folder). Cross-platform."""
        import subprocess
        import sys

        from PyQt6.QtCore import QUrl
        from PyQt6.QtGui import QDesktopServices

        p = Path(path)
        folder = p.parent
        try:
            if sys.platform.startswith("win"):
                # Keep the switch separate from the path.  If they are one
                # argument, subprocess quotes the whole string when the path
                # contains spaces ("/select,C:\\some path\\file"), and
                # Explorer treats it as a location instead of as /select.
                subprocess.Popen(["explorer.exe", "/select,", str(p)])
                return
            if sys.platform == "darwin":
                subprocess.Popen(["open", "-R", str(p)])
                return
        except Exception:
            pass
        QDesktopServices.openUrl(QUrl.fromLocalFile(str(folder)))

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

        entries = plugins.available()
        if not entries:
            placeholder = menu.addAction("(no plugins registered)")
            placeholder.setEnabled(False)
            return

        from .command_finder import source_of
        for entry in entries:
            act = QAction(entry.name, self)
            if entry.tooltip:
                act.setToolTip(entry.tooltip)
                act.setStatusTip(entry.tooltip)
            # Record the plugin's implementing file so the Command Finder can show
            # it in the Source column, plus any extra search tags it declares.
            act.setProperty("command_source", source_of(entry.launch))
            if getattr(entry, "keywords", ()):
                act.setProperty("command_keywords", " ".join(entry.keywords))
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

    def _on_overlay_manual_alignment_requested(self, dataset_idx: int) -> None:
        """Enter the existing channel-row Manual align mode for a dataset."""
        from ..core.overlay import is_multichannel_overlay

        if not (0 <= dataset_idx < len(self._state.datasets)):
            return
        if not is_multichannel_overlay(self._state, dataset_idx):
            self._state.log(
                "Manual alignment requires at least two datasets in the same overlay.",
                "WARN",
                dataset_idx=dataset_idx,
            )
            return

        self._state.set_active(dataset_idx)

        def contains_dataset(window) -> bool:
            return any(
                channel.get("dataset_idx") == dataset_idx
                for channel in getattr(window, "_channels", [])
            )

        candidates = []
        focused = getattr(self, "_last_active_plot_window", None)
        if focused is not None and contains_dataset(focused):
            candidates.append(focused)
        candidates.extend(
            window
            for window in self._render_windows.values()
            if contains_dataset(window)
        )
        candidates.extend(
            window
            for window in self._scatter_windows.values()
            if contains_dataset(window)
        )

        seen: set[int] = set()
        for window in candidates:
            if id(window) in seen:
                continue
            seen.add(id(window))
            start = getattr(window, "start_manual_alignment_for_dataset", None)
            if callable(start) and start(dataset_idx):
                window.show()
                window.raise_()
                window.activateWindow()
                return

        # No overlay coordinate view is open. Open the standard render view,
        # then enter the same handler used by channel-row right-click.
        window = self._show_render(dataset_idx)
        start = getattr(window, "start_manual_alignment_for_dataset", None)
        if callable(start) and start(dataset_idx):
            window.raise_()
            window.activateWindow()
            return
        self._state.log(
            "Could not start manual alignment for this overlay.",
            "WARN",
            dataset_idx=dataset_idx,
        )

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

    def _show_attr_plot_cpu(self, dataset_idx: int | None = None):
        """Open the independent CPU bulk/aggregation attribute plot."""

        if self._state.active_dataset is None:
            self._no_data_warning()
            return None
        from .attribute_window import AttributeWindow

        idx = dataset_idx if type(dataset_idx) is int else self._state.active_idx
        if idx is None:
            return None
        win = self._attr_cpu_windows.get(idx)
        if win is None:
            win = AttributeWindow(self._state, dataset_idx=idx, cpu_fix=True)
            win.destroyed.connect(
                lambda _=None, i=idx: self._attr_cpu_windows.pop(i, None)
            )
            self._attr_cpu_windows[idx] = win
        self._attr_cpu_win = win
        self._install_window_shortcuts(win)
        win.show()
        win.raise_()
        win.activateWindow()
        self._notify_view_state_changed()
        return win

    def _attribute_gpu_state(self) -> bool:
        """Whether the active dataset's Attribute Plot uses the GPU renderer.

        Falls back to the dataset's saved view state, so the menu is right even
        before the plot has been opened.
        """
        idx = self._state.active_idx
        if idx is None:
            return False
        win = self._attr_windows.get(idx)
        if win is not None:
            return bool(getattr(win, "gpu_2d", False))
        if 0 <= idx < len(self._state.datasets):
            saved = self._state.datasets[idx].state.get("attribute_plot_state")
            if isinstance(saved, dict):
                return bool(saved.get("gl_2d", False))
        return False

    def _sync_attribute_gpu_action(self) -> None:
        action = getattr(self, "actionAttributeGpu", None)
        if action is None:
            return
        capabilities = getattr(self._state, "gpu_capabilities", None)
        gpu_available = (
            capabilities is None
            or bool(getattr(capabilities, "available", False))
        )
        action.setEnabled(self._state.active_idx is not None and gpu_available)
        if capabilities is not None:
            if gpu_available:
                action.setToolTip(
                    "Draw exact 2-D markers with OpenGL when they fit the current "
                    f"memory budget ({int(capabilities.point_limit):,} points).\n"
                    f"Renderer: {capabilities.renderer or 'unknown'}; "
                    f"{capabilities.memory_summary}."
                )
            else:
                action.setToolTip(
                    "GPU rendering is unavailable on this display: "
                    f"{capabilities.reason}"
                )
        action.blockSignals(True)
        action.setChecked(gpu_available and self._attribute_gpu_state())
        action.blockSignals(False)

    def _toggle_attribute_gpu(self, enabled: bool) -> None:
        """Switch the Attribute Plot renderer, opening the plot if needed."""
        capabilities = getattr(self._state, "gpu_capabilities", None)
        if (
            enabled and capabilities is not None
            and not bool(getattr(capabilities, "available", False))
        ):
            self._state.log(
                f"Attribute Plot GPU unavailable: {capabilities.reason}",
                level="WARN",
            )
            self._sync_attribute_gpu_action()
            return
        win = self._show_attr_plot()
        if win is None:
            self._sync_attribute_gpu_action()
            return
        win.set_gpu_2d(bool(enabled))
        # The switch is a property of the 2-D projections; say so rather than
        # letting a click on a 3-D plot look like it did nothing.
        pending = (
            "  Applies to XY/XZ/YZ; this plot is showing 3D."
            if getattr(win, "view_mode", "") == "3D" else ""
        )
        self._state.log(
            "Attribute Plot renderer: "
            f"{'GPU (OpenGL)' if enabled else 'pyqtgraph'}.{pending}"
        )

    def _show_filter(self) -> None:
        from .filter_dialog import FilterDialog
        from .modeless import show_modeless
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
        # Filter dialogs are top-level modeless windows, like the normal
        # viewer windows.  Giving a top-level Qt.Window the main window as a
        # QWidget parent makes Windows keep it above that owner, so it can
        # obscure the main window whenever the two overlap.
        win = FilterDialog(self._state, parent=None, dataset_idx=idx)
        win.destroyed.connect(lambda _=None, key=idx: self._filter_dlgs.pop(key, None))
        self._filter_dlgs[idx] = win
        self._filter_dlg = win
        show_modeless(win, self)
        self._notify_view_state_changed()
        return win

    def _load_filter_json(self, path: str) -> None:
        """Open (or raise) the filter dialog and append filters from a JSON file."""
        self._show_filter()
        self._filter_dlg.load_filter_json(path)

    def _show_dataset_manager(self) -> None:
        manager = self._get_dataset_manager()
        manager.show()
        manager.raise_()
        manager.activateWindow()
        self._notify_view_state_changed()

    def _get_dataset_manager(self):
        """Return the existing Dataset Manager, creating it if necessary."""
        from .dataset_manager import DatasetManager
        if self._ds_manager is None:
            self._ds_manager = DatasetManager(self._state, parent=self)
            self._ds_manager.destroyed.connect(
                lambda _=None: setattr(self, "_ds_manager", None)
            )
        return self._ds_manager

    def _ensure_dataset_manager_visible(self) -> None:
        """Create/show the Dataset Manager without changing window focus."""
        manager = self._get_dataset_manager()
        try:
            if not manager.isVisible():
                manager.show()
        except RuntimeError:
            # The QObject may have been deleted but the destroyed signal has
            # not run yet; retry through the normal creation path.
            self._ds_manager = None
            manager = self._get_dataset_manager()
            manager.show()
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
        from ..core.roi_convert import _SHAPE_TARGETS
        from .roi_convert_dialog import RoiSizeDialog
        if record.type == "point" and target in _SHAPE_TARGETS:
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

    # ------------------------------------------------------------------
    # Process › ROI › Fit  (data-point fits + spline/interpolate)
    # ------------------------------------------------------------------
    _REGION_ROI_TYPES = {"rectangle", "oval", "polygon", "freehand"}

    def _roi_highlighted_points(self, record, ds):
        """(M, 2) localizations the region *record* highlights, in the ROI's plane
        (display nm, filter applied)."""
        import numpy as np
        from ..core.roi_crop import display_xyz_filtered
        from ..core.roi_selection import roi_region_mask

        plane = "XY"
        if isinstance(record.context, dict):
            plane = record.context.get("view_plane") or "XY"
        axes = {"XY": (0, 1), "XZ": (0, 2), "YZ": (1, 2)}.get(plane, (0, 1))
        xyz = display_xyz_filtered(ds)
        if xyz.shape[0] == 0:
            return np.empty((0, 2), dtype=float)
        x, y = xyz[:, axes[0]], xyz[:, axes[1]]
        mask = np.asarray(roi_region_mask(x, y, record), dtype=bool)
        return np.column_stack([x[mask], y[mask]])

    def _fit_active_roi(self, target: str) -> None:
        """Process › ROI › Fit ▸ Rectangle/Circle/Ellipse/Polygon/Convex Hull —
        fit the shape to the localizations the active region ROI highlights."""
        from ..core.roi_fit import data_fit

        record, kind, adapter = self._active_roi()
        if record is None or record.type not in self._REGION_ROI_TYPES:
            return  # region ROI only; silently ignore otherwise
        ds = self._state.active_dataset
        if ds is None:
            self._no_data_warning()
            return
        pts = self._roi_highlighted_points(record, ds)
        if pts.shape[0] < 3:
            self._state.log(
                "Fit: fewer than 3 localizations highlighted by the ROI.", "WARN")
            return
        try:
            new_record = data_fit(record, pts, target)
        except Exception as exc:
            self._state.log(f"Fit {target} failed: {exc}", "ERROR")
            return
        old_type = record.type
        self._commit_roi_replacement(record, kind, adapter, new_record)
        self._state.log(
            f"Fit {target.replace('_', ' ')} on ROI '{record.name}' ({old_type}) "
            f"using {pts.shape[0]} localization(s).")

    def _spline_fit_active_roi(self) -> None:
        """Process › ROI › Fit ▸ Fit Spline — smooth the active ROI outline."""
        from ..core.roi_fit import spline_fit

        record, kind, adapter = self._active_roi()
        if record is None:
            return
        try:
            new_record = spline_fit(record)
        except Exception as exc:
            self._state.log(f"Spline fit failed: {exc}", "ERROR")
            return
        self._commit_roi_replacement(record, kind, adapter, new_record)
        self._state.log(f"Spline-fit ROI '{record.name}'.")

    def _interpolate_active_roi(self) -> None:
        """Process › ROI › Fit ▸ Interpolate — resample the active ROI outline at
        an even arc-length interval (nm), ImageJ-style."""
        from PyQt6.QtWidgets import QInputDialog
        from ..core.roi_fit import interpolate_outline

        record, kind, adapter = self._active_roi()
        if record is None:
            return
        interval, ok = QInputDialog.getDouble(
            self, "Interpolate ROI", "Interval (nm):", 20.0, 0.1, 1e6, 1)
        if not ok:
            return
        try:
            new_record = interpolate_outline(record, interval_nm=float(interval))
        except Exception as exc:
            self._state.log(f"Interpolate failed: {exc}", "ERROR")
            return
        self._commit_roi_replacement(record, kind, adapter, new_record)
        self._state.log(f"Interpolated ROI '{record.name}' at {interval:g} nm.")

    # ------------------------------------------------------------------
    # Process › ROI › Restore ROI
    # ------------------------------------------------------------------
    def _coordinate_views_for_dataset(self, idx):
        """Open render / scatter windows for dataset *idx*."""
        wins = []
        for reg in (self._render_windows, self._scatter_windows):
            w = reg.get(idx)
            if w is not None:
                wins.append(w)
        return wins

    def _remember_roi_for_restore(self, window, record) -> None:
        """Cache the active ROI draft (called from the overlay controller) so
        Restore ROI can bring it back after a delete."""
        import copy
        idx = getattr(window, "_idx", None)
        if isinstance(idx, int) and record is not None:
            self._roi_last_active[idx] = copy.deepcopy(record)

    def _restore_roi(self) -> None:
        """Process › ROI › Restore ROI — put the active ROI draft onto the other
        open coordinate views of the dataset (render ↔ scatter), or, if none is
        active, restore the last active draft (e.g. after an accidental delete)."""
        import copy
        idx = self._state.active_idx
        ds = self._state.active_dataset
        if idx is None or ds is None:
            self._no_data_warning()
            return
        views = self._coordinate_views_for_dataset(idx)
        source = None
        source_ctrl = None
        for win in views:
            ctrl = getattr(win, "_roi_overlay", None)
            if ctrl is None:
                continue
            try:
                rec = ctrl.current_record()
            except Exception:
                rec = None
            if rec is not None:
                source, source_ctrl = rec, ctrl
                break
        if source is None:
            source = self._roi_last_active.get(idx)   # deleted / last → restore
        if source is None:
            self._state.log(
                "Restore ROI: no active or remembered ROI for this dataset.", "WARN")
            return
        self._roi_last_active[idx] = copy.deepcopy(source)
        applied = 0
        for win in views:
            ctrl = getattr(win, "_roi_overlay", None)
            if ctrl is None or ctrl is source_ctrl:
                continue   # skip the view the source draft already lives on
            try:
                ctrl.replace_draft(copy.deepcopy(source))
                applied += 1
            except Exception:
                pass
        if applied:
            self._state.log(f"Restored ROI '{source.name}' to {applied} view(s).")
        else:
            self._state.log(
                "Restore ROI: no other coordinate view open to restore onto.", "WARN")

    def _run_hlyb_pairwise_analysis(self, dimensions: int = 3) -> None:
        """Analyze › Clustering › HlyB subunit pair analysis › Pair-distance model
        fit — measure the trace-centroid pair-distance distribution against an
        envelope-preserving null and fit the distribution of inter-subunit
        distances to it.

        No merge radius is applied anywhere, so no distance range is removed and
        the short-range same-site population is modelled rather than deleted.
        The 2-D variant additionally delineates each E.coli, shrinks it inward to
        drop the edge-on rim, and models the residual foreshortening from the
        measured membrane tilt.
        """
        dimensions = 2 if int(dimensions) == 2 else 3
        import numpy as np
        from ..core.loader import mfx_get

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
                self, "HlyB Pair-Distance Model Fit",
                "The active dataset has no localizations with trace IDs to analyze.")
            return
        if lz is None or lz.size != lx.size:
            lz = np.zeros_like(lx)
        loc_m = np.column_stack([lx, ly, lz])
        # The time column calibrates the same-site short-range kernel; without
        # it the kernel falls back to an assumed width, which is reported.
        tim = _col("tim")
        if tim is not None and tim.size != lx.size:
            tim = None

        from ..analysis.hlyb_pairwise import PairFitConfig, analyze_hlyb_pairwise
        from .hlyb_pairwise_dialog import HlyBPairwiseDialog, HlyBPairwiseWindow

        current_z_scaling_factor = float(getattr(ds.cali, "z_scaling_factor", 0.67) or 0.67)
        defaults = getattr(self, "_hlyb_pair_cfg", None)
        overrides = {"z_scaling_factor": current_z_scaling_factor, "dimensions": dimensions}
        if defaults is None:
            defaults = PairFitConfig(**overrides)
        else:
            defaults = PairFitConfig(**{**vars(defaults), **overrides})
        dlg = HlyBPairwiseDialog(self, defaults=defaults, dimensions=dimensions)
        if dlg.exec() != QDialog.DialogCode.Accepted:
            return
        cfg = dlg.config()
        self._hlyb_pair_cfg = cfg

        QApplication.setOverrideCursor(Qt.CursorShape.WaitCursor)
        try:
            result = analyze_hlyb_pairwise(loc_m, tid, tim, cfg)
        except Exception as exc:
            QMessageBox.warning(self, "HlyB Pair-Distance Model Fit",
                                f"Analysis failed: {exc}")
            return
        finally:
            if QApplication.overrideCursor() is not None:
                QApplication.restoreOverrideCursor()

        from .modeless import show_modeless
        win = HlyBPairwiseWindow(
            result, title=f"{ds.name} ({dimensions}D)", owner=self,
            prefs=self._state.prefs)
        show_modeless(win, self)

        best = result.get("best_hypothesis", "")
        fits = result.get("fits", {})
        relaxed = result.get("fits_relaxed_kernel", {})
        margin = min((f.get("delta_aic", 0.0) for name, f in fits.items()
                      if name != best), default=float("nan"))
        margin_relaxed = min((f.get("delta_aic", 0.0) for name, f in relaxed.items()
                              if name != result.get("best_hypothesis_relaxed", "")),
                             default=float("nan"))
        kernel = result.get("repeat_kernel", {})
        best_fit = result.get("best_fit", {})

        provenance = ds.z_scaling_factor_provenance or {}
        z_source = str(provenance.get("source") or "").strip()
        if abs(float(cfg.z_scaling_factor) - float(getattr(ds.cali, "z_scaling_factor", 0) or 0)) > 1e-9:
            z_source = "a value entered in the analysis dialog"
        elif not z_source:
            z_source = "the dataset's recorded Z scaling factor"
        else:
            z_source = f"the dataset's recorded Z scaling factor, provenance '{z_source}'"

        def _fit_summary(store):
            return {
                name: {
                    "delta_aic": float(f.get("delta_aic", float("nan"))),
                    "n_repeat_pairs": float(f.get("n_repeat_pairs", float("nan"))),
                    "n_structure_pairs": float(f.get("n_structure_pairs", float("nan"))),
                    "background_scale": float(f.get("background_scale", float("nan"))),
                    "sigma_nm": float(f.get("sigma_nm", float("nan"))),
                    "repeat_scale": float(f.get("repeat_scale", float("nan"))),
                    "structure_label": str(f.get("structure_label", "")),
                    "structure_description": str(f.get("structure_description", "")),
                    "distance_summary": {
                        k: float(v) for k, v in (f.get("distance_summary") or {}).items()
                    },
                    "parameters_at_bounds": list(f.get("parameters_at_bounds", [])),
                }
                for name, f in (store or {}).items()
            }

        sem = np.asarray(result.get("centroid_sem_nm", []), dtype=float).ravel()
        method_data = {
            "schema": "hlyb_pair_distance_fit_3d/v1",
            "dimensions": dimensions,
            "projection": {
                "is_2d": bool(result.get("is_2d")),
                "cell_mask_stats": {
                    k: (float(v) if isinstance(v, (int, float)) else str(v))
                    for k, v in (result.get("cell_mask_stats") or {}).items()
                },
                "median_tilt_deg": float(result.get("median_tilt_deg", float("nan"))),
                "median_foreshortening": float(
                    result.get("median_foreshortening", float("nan"))),
            },
            "input": {
                "dataset_name": ds.name,
                "source_path": str(
                    ds.metadata.get("msr_source_path")
                    or getattr(ds.file, "recent_path", None)
                    or getattr(ds.file, "path", "") or ""),
                "source_format": str(ds.metadata.get("source_format", "") or ""),
                "source_version": str(ds.metadata.get("source_version", "") or ""),
                "n_localizations": int(lx.size),
                "n_traces_total": int(result["n_traces_total"]),
                "n_traces_used": int(result["n_traces_used"]),
                "iteration_selector": "last",
                "valid_only": True,
                "filter_mask_applied": False,
                "time_column_available": bool(tim is not None),
                "z_scaling_source": z_source,
            },
            "parameters": {
                "min_loc_per_trace": int(cfg.min_loc_per_trace),
                "z_scaling_factor": float(cfg.z_scaling_factor),
                "r_max_nm": float(cfg.r_max_nm),
                "bin_nm": float(cfg.bin_nm),
                "fit_r_min_nm": float(cfg.fit_r_min_nm),
                "fit_r_max_nm": float(cfg.fit_r_max_nm),
                "null_cell_nm": float(cfg.null_cell_nm),
                "null_replicates": int(cfg.null_replicates),
                "repeat_gap_s": float(cfg.repeat_gap_s),
                "repeat_max_nm": float(cfg.repeat_max_nm),
                "label_offset_bounds_nm": [float(v) for v in cfg.label_offset_bounds_nm],
                "fit_label_offset": bool(cfg.fit_label_offset),
                "dimer_distance_bounds_nm": [float(v) for v in cfg.dimer_distance_bounds_nm],
                "hypotheses": [str(h) for h in cfg.hypotheses],
            },
            "observable": {
                "centroid_sem_nm": [float(v) for v in sem] if sem.size == 3 else [],
                "sigma_floor_nm": float(result.get("sigma_floor_nm", float("nan"))),
                "excess_outer_nm": float(result.get("excess_outer_nm", float("nan"))),
                "null_replicates": int(result.get("null_replicates", 0)),
            },
            "repeat_kernel": {
                "source": str(kernel.get("source", "")),
                "n_pairs": int(kernel.get("n_pairs", 0)),
                "median_nm": float(kernel.get("median_nm", float("nan"))),
                "sigma_nm": float(kernel.get("sigma_nm", float("nan"))),
                "rejected_far_fraction": float(
                    kernel.get("rejected_far_fraction", float("nan"))),
            },
            "model": {
                "class_names": [str(x) for x in result.get("class_names", [])],
                "class_distances_nm": [float(x) for x in result.get("class_distances_nm", [])],
                "class_weight": 1.0 / max(len(result.get("class_distances_nm", [])) or 1, 1),
                "reference_dimer_nm": float(result.get("reference_dimer_nm", float("nan"))),
                "structure_labels": {
                    str(k): str(v) for k, v in (result.get("structure_labels") or {}).items()
                },
            },
            "distance_scan": {
                key: (float(scan_value) if isinstance(scan_value, (int, float))
                      else [float(v) for v in scan_value]
                      if key in ("ci68_nm", "ci95_nm") else scan_value)
                for key, scan_value in (result.get("distance_scan") or {}).items()
                if key in ("available", "best_nm", "ci68_nm", "ci95_nm", "step_nm",
                           "constrained", "ci68_below_scan_step", "parameter")
            },
            "fits": _fit_summary(fits),
            "fits_relaxed_kernel": _fit_summary(relaxed),
            "best_hypothesis": str(best),
            "best_hypothesis_relaxed": str(result.get("best_hypothesis_relaxed", "")),
        }

        self._state.log(
            f"HlyB pair-distance model fit ({dimensions}D) on '{ds.name}': "
            f"{result['n_traces_used']:,} of {result['n_traces_total']:,} trace(s); "
            f"excess above null out to {result['excess_outer_nm']:.1f} nm; "
            f"best model '{best}' (next by dAIC {margin:.1f}, "
            f"{margin_relaxed:.1f} with the short-range kernel released); "
            f"median inter-subunit distance "
            f"{(best_fit.get('distance_summary') or {}).get('median_nm', float('nan')):.2f} nm "
            f"(68% "
            f"{(best_fit.get('distance_summary') or {}).get('p16_nm', float('nan')):.2f}"
            f"-{(best_fit.get('distance_summary') or {}).get('p84_nm', float('nan')):.2f} nm), "
            f"blur {best_fit.get('sigma_nm', float('nan')):.2f} nm "
            f"(min loc/trace {cfg.min_loc_per_trace}, z-scale {cfg.z_scaling_factor}, "
            f"kernel {kernel.get('source', 'n/a')} from {kernel.get('n_pairs', 0)} pair(s)).",
            dataset_idx=idx, method_data=method_data)

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
            mode = "TEMPLATE2D" if "2D" in mode_text else "TEMPLATE3D"
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
        z_was_synthesized = lz is None or lz.size != lx.size
        if z_was_synthesized:
            lz = np.zeros_like(lx)
        loc_m = np.column_stack([lx, ly, lz])  # metres, raw z (z-scaling applied in analysis)

        from .hlyb_clustering_dialog import HlyBClusteringDialog, HlyBResultWindow
        from ..analysis.hlyb_clustering import (
            HlyBConfig,
            analyze_hlyb,
            analyze_hlyb_2d,
            analyze_hlyb_template2d,
            analyze_hlyb_template3d,
        )

        # The analysis reads RAW z (never Z-scaling-baked) and applies this z-scaling
        # factor itself, so the dialog default must track the dataset's CURRENT
        # Z scaling factor — otherwise re-running after changing Z scaling factor reused a stale cached
        # z-scale and left the result unchanged. Other tweaked parameters persist.
        current_z_scaling_factor = float(getattr(ds.cali, "z_scaling_factor", 0.67) or 0.67)
        defaults = getattr(self, "_hlyb_cfg", None)
        if defaults is None:
            defaults = HlyBConfig(z_scaling_factor=current_z_scaling_factor)
        else:
            defaults = HlyBConfig(**{**vars(defaults), "z_scaling_factor": current_z_scaling_factor})
        dlg = HlyBClusteringDialog(self, defaults=defaults, mode=mode)
        if dlg.exec() != QDialog.DialogCode.Accepted:
            return
        cfg = dlg.config()
        self._hlyb_cfg = cfg

        QApplication.setOverrideCursor(Qt.CursorShape.WaitCursor)
        try:
            if mode == "2D":
                result = analyze_hlyb_2d(loc_m, tid, cfg)
            elif mode == "TEMPLATE2D":
                result = analyze_hlyb_template2d(loc_m, tid, cfg)
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
        title_mode = {"TEMPLATE3D": "Template matching (3D)",
                      "TEMPLATE2D": "Template matching (2D)"}.get(mode, mode)
        win = HlyBResultWindow(
            result, cfg, title=f"{ds.name} ({title_mode})", owner=self,
            prefer_2d=(mode in {"2D", "TEMPLATE2D"}), source_dataset=ds,
            prefs=self._state.prefs,
        )
        show_modeless(win, self)

        pd = result["all_pair_distances"]
        med = float(np.median(pd)) if pd.size else float("nan")
        if mode == "2D":
            prefix = (
                f"HlyB subunit pair analysis (2D) on '{ds.name}': "
                f"{result['n_total_traces']} trace(s), {result['n_border_traces']} border-excluded, "
                f"{result['n_traces']} interior → ")
            extra = (f"border {cfg.border_size_nm:g} nm, mask px {cfg.mask_pixel_size_nm:g} nm")
        elif mode in {"TEMPLATE2D", "TEMPLATE3D"}:
            qc = result.get("match_qc", {})
            dims = "2D" if mode == "TEMPLATE2D" else "3D"
            prefix = (
                f"HlyB subunit pair analysis (template matching {dims}) on '{ds.name}': "
                + (f"{result.get('n_total_traces', result['n_traces'])} trace(s), "
                   f"{result.get('n_border_traces', 0)} border-excluded, "
                   f"{result['n_traces']} interior → " if mode == "TEMPLATE2D"
                   else f"{result['n_traces']} trace(s) → "))
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
            if mode in {"TEMPLATE2D", "TEMPLATE3D"}
            else f"HlyB radius {result['hlyb_diameter_nm']:.1f} nm"
        )
        method_data = None
        if mode in {"TEMPLATE2D", "TEMPLATE3D"}:
            model = result.get("model", {})
            structures = result.get("structures", [])
            structure_sizes: dict[str, int] = {}
            for structure in structures:
                size = str(int(np.asarray(structure.get("unit_indices", [])).size))
                structure_sizes[size] = structure_sizes.get(size, 0) + 1
            residuals = [
                np.asarray(structure.get("pair_residuals", []), dtype=float).ravel()
                for structure in structures
            ]
            residuals = np.concatenate(residuals) if residuals else np.empty(0)
            rms_values = np.asarray(
                [structure.get("rms_residual_nm", np.nan) for structure in structures],
                dtype=float,
            )
            match_fractions = np.asarray(
                [structure.get("match_fraction", np.nan) for structure in structures],
                dtype=float,
            )
            source_path = (
                ds.metadata.get("msr_source_path")
                or getattr(ds.file, "recent_path", None)
                or getattr(ds.file, "path", "")
            )
            qc = result.get("match_qc", {})
            # Where the z scale came from matters scientifically: a value of 1.0
            # can mean "2-D data", "user-fixed" or "the anisotropy estimate was
            # rejected", and those are not the same claim.
            provenance = ds.z_scaling_factor_provenance or {}
            z_source = str(provenance.get("source") or "").strip()
            if abs(float(cfg.z_scaling_factor) - float(getattr(ds.cali, "z_scaling_factor", 0) or 0)) > 1e-9:
                z_source = "a value entered in the analysis dialog"
            elif not z_source:
                z_source = "the dataset's recorded Z scaling factor"
            else:
                z_source = f"the dataset's recorded Z scaling factor, provenance '{z_source}'"
            method_data = {
                "schema": "hlyb_template_matching_3d/v1",
                "input": {
                    "dataset_name": ds.name,
                    "source_path": str(source_path or ""),
                    "source_format": str(ds.metadata.get("source_format", "") or ""),
                    "source_version": str(ds.metadata.get("source_version", "") or ""),
                    "n_dimensions": int(getattr(ds.prop, "num_dim", 3) or 3),
                    "n_localizations": int(lx.size),
                    "n_traces": int(result["n_traces"]),
                    "iteration_selector": "last",
                    "valid_only": True,
                    "filter_mask_applied": False,
                    "coordinate_unit": "metres",
                    "coordinate_fields": (
                        ["loc_x", "loc_y"] if z_was_synthesized
                        else ["loc_x", "loc_y", "loc_z"]
                    ),
                    "trace_id_field": "tid",
                    "z_was_synthesized": bool(z_was_synthesized),
                    "z_scaling_source": z_source,
                },
                "parameters": {
                    "min_loc_per_trace": int(cfg.min_loc_per_trace),
                    "z_scaling_factor": float(cfg.z_scaling_factor),
                    "unit_render_pixel_size_nm": float(cfg.unit_render_pixel_size),
                    "basic_unit_size_nm": float(cfg.basic_unit_size_nm),
                    "min_observed_subunits": int(cfg.min_observed_subunits_per_HlyB),
                    "core_a_ring_side_nm": float(cfg.template_core_a_ring_side_nm),
                    "core_b_ring_side_nm": float(cfg.template_core_b_ring_side_nm),
                    "core_twist_deg": float(cfg.template_core_twist_deg),
                    "core_axial_offset_nm": float(cfg.template_core_axial_offset_nm),
                    "label_offset_nm": float(cfg.template_label_offset_nm),
                    "pair_tolerance_nm": float(cfg.model_pair_tolerance_nm),
                    "rms_threshold_nm": float(cfg.model_rms_threshold_nm),
                    "max_pair_residual_nm": float(cfg.model_max_residual_nm),
                    "min_pair_match_fraction": float(cfg.min_pair_match_fraction),
                },
                "effective_parameters": {
                    "basic_unit_size_nm": float(result["dunit_nm"]),
                    "pair_tolerance_nm": float(result["model_pair_tolerance_nm"]),
                    "rms_threshold_nm": float(result["model_rms_threshold_nm"]),
                    "max_pair_residual_nm": float(result["model_max_residual_nm"]),
                    "candidate_edge_radius_nm": float(result["candidate_edge_radius_nm"]),
                    "max_observed_subunits": int(cfg.max_observed_subunits_per_HlyB),
                    "max_candidate_subsets_per_component": int(
                        cfg.max_candidate_subsets_per_component),
                },
                "template": {
                    "site_labels": [str(x) for x in model.get("labels", [])],
                    "class_distances_nm": {
                        str(key): float(value)
                        for key, value in model.get("class_distances_nm", {}).items()
                    },
                },
                "projection": {
                    "is_2d": bool(result.get("is_2d", False)),
                    "tilt_deg": float(result.get("projection_tilt_deg", 0.0) or 0.0),
                    "max_shortening": float(
                        result.get("projection_max_shortening", 0.0) or 0.0),
                    "n_border_traces": int(result.get("n_border_traces", 0)),
                    "n_total_traces": int(result.get("n_total_traces", 0)),
                    "cell_mask_stats": {
                        k: (float(v) if isinstance(v, (int, float)) else str(v))
                        for k, v in (result.get("cell_mask_stats") or {}).items()
                    },
                },
                "screening": {
                    "n_after_trace_density": int(result.get("n_pass1", 0)),
                    "n_after_log": int(result.get("n_pass2", 0)),
                    "n_components": int(qc.get("n_components", 0)),
                    "n_candidates_tested": int(qc.get("n_candidates_tested", 0)),
                    "n_candidates_passed_thresholds": int(
                        qc.get("n_candidates_passed_thresholds", 0)),
                    "n_overlap_rejected": int(qc.get("n_overlap_rejected", 0)),
                    "n_skipped_large_subsets": int(qc.get("n_skipped_large_subsets", 0)),
                },
                "result": {
                    "n_subunits": int(result["n_subunits"]),
                    "n_structures": int(result["n_structures"]),
                    "structure_size_counts": structure_sizes,
                    "n_pairs": int(pd.size),
                    "pair_distance_median_nm": med,
                    "pair_distance_min_nm": float(np.min(pd)) if pd.size else float("nan"),
                    "pair_distance_max_nm": float(np.max(pd)) if pd.size else float("nan"),
                    "residual_median_abs_nm": (
                        float(np.median(np.abs(residuals))) if residuals.size else float("nan")
                    ),
                    "residual_max_abs_nm": (
                        float(np.max(np.abs(residuals))) if residuals.size else float("nan")
                    ),
                    "structure_rms_median_nm": (
                        float(np.nanmedian(rms_values)) if rms_values.size else float("nan")
                    ),
                    "match_fraction_median": (
                        float(np.nanmedian(match_fractions))
                        if match_fractions.size else float("nan")
                    ),
                },
            }
        self._state.log(
            prefix
            + f"{result['n_subunits']} subunit(s) → {result['n_structures']} HlyB structure(s); "
            f"{pd.size} pair(s), median distance {med:.2f} nm "
            f"(unit Ø {result['dunit_nm']:.1f} nm, {radius_label}, "
            f"min loc/trace {cfg.min_loc_per_trace}, {extra}).",
            dataset_idx=idx,
            method_data=method_data)

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

    def _show_shape_segmentation(self) -> None:
        """Analyze › Segmentation › Shape Model… — open the known-geometry
        (shape-model) segmentation tool for the active dataset."""
        idx = self._state.active_idx
        if idx is None or not (0 <= idx < len(self._state.datasets)):
            self._no_data_warning()
            return
        from .modeless import show_modeless
        from .shape_segmentation_dialog import ShapeSegmentationWindow
        win = ShapeSegmentationWindow(self._state, idx, owner=self)
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

    def _show_roi_3d(self) -> None:
        """Process › ROI › 3D ROI — draw a 3-D ROI by intersecting 2-D shapes
        extruded from the XY / XZ / YZ ortho views, and crop the active dataset
        to the selected localizations."""
        idx = self._state.active_idx
        if idx is None or not (0 <= idx < len(self._state.datasets)):
            self._no_data_warning()
            return
        ds = self._state.datasets[idx]
        from ..core.dataset_kind import is_3d
        if not is_3d(ds):
            QMessageBox.information(
                self, "3-D ROI",
                "The active dataset is 2-D and has no non-degenerate Z coordinate.",
            )
            return
        from .modeless import show_modeless
        from .roi_3d_dialog import Roi3DWindow
        win = Roi3DWindow(self._state, idx, owner=self)
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

    def _show_straightened_volume_skeleton(self) -> None:
        """Analyze › Segmentation › Straightened Volume along Skeleton… —
        experimental centerline-straightened projection of a 3-D point cloud.

        The skeleton is supplied by selected 3-D point/polyline ROI records in
        the ROI Manager.
        """
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
        z = loc[:, 2] if loc.ndim == 2 and loc.shape[1] >= 3 else np.empty(0)
        z = z[np.isfinite(z)]
        if loc.ndim != 2 or loc.shape[1] < 3 or z.size == 0 or float(np.ptp(z)) <= 0:
            QMessageBox.information(
                self, "Straightened Volume along Skeleton",
                "The active dataset does not provide a non-degenerate Z coordinate. "
                "This experimental tool needs a 3-D localization dataset.")
            return
        self._show_render(idx)
        self._show_roi_manager()
        from .modeless import show_modeless
        from .straightened_volume_dialog import StraightenedVolumeAlongSkeletonWindow
        win = StraightenedVolumeAlongSkeletonWindow(self._state, idx, owner=self)
        show_modeless(win, self)

    def _system_roi_color(self) -> str:
        """The user's configured default ROI color (Preferences ▸ Appearance ▸ ROI,
        default ``Yellow``). Auto-generated ROIs (segmentation / detection) use this
        as their base stroke so they match manually drawn ROIs; the bright cyan
        ``RoiOverlayController.MANAGER_SELECT_COLOR`` is reserved for the ROI-Manager
        selection highlight only (so a selected ROI stays distinguishable)."""
        # rgba_hex (#RRGGBBAA), not rgba_qt_hex: this string is handed to
        # pyqtgraph, which reads Qt's #AARRGGBB as RGBA and turns an opaque
        # color into a fully transparent one.
        from ..colors import pg_safe_hex, viewer_color
        return pg_safe_hex(viewer_color(self._state.prefs, "roi_edge"))

    def add_segmentation_rois(self, idx: int, centers, *, side_nm: float,
                              name_prefix: str, source: str, stroke_color: str | None = None,
                              names: list[str] | None = None,
                              log_message: str | None = None) -> int:
        """Add a rectangle ROI of side ``side_nm`` centred on each ``(x, y)`` in
        *centers* to the ROI Manager, reveal them, and show render + manager.

        ``names`` (if given, one per centre) sets each ROI's name; otherwise they
        are auto-numbered ``"<name_prefix> <i>"``. ``stroke_color`` defaults to the
        system ROI color. Shared by the convolution segmentation tool; returns the
        count added."""
        import numpy as np

        from ..core.roi import RoiRecord

        centers = np.asarray(centers, dtype=float).reshape(-1, 2)
        if centers.shape[0] == 0 or not (0 <= idx < len(self._state.datasets)):
            return 0
        stroke = stroke_color or self._system_roi_color()
        side = max(float(side_nm), 1.0)
        for i, (cx, cy) in enumerate(centers, start=1):
            name = names[i - 1] if names is not None and i - 1 < len(names) else f"{name_prefix} {i}"
            rec = RoiRecord.create(
                "rectangle",
                {"bounds": [float(cx) - side / 2.0, float(cy) - side / 2.0, side, side]},
                name=name, coordinate_space="plot",
                stroke_color=stroke)
            rec.context = {"dataset_idx": idx, "source": source}
            self._state.rois.add(rec)
        self._state.rois.set_show_all(True)
        self._show_render(idx)
        self._show_roi_manager()
        if log_message:
            self._state.log(log_message)
        return int(centers.shape[0])

    def add_point_rois_3d(self, idx: int, centers, *, name_prefix: str, source: str,
                          stroke_color: str | None = None, names: list[str] | None = None,
                          log_message: str | None = None) -> int:
        """Add a 3-D point ROI (full ``[x, y, z]`` nm geometry) for each centre in
        *centers* ``(K, 3)`` to the ROI Manager, reveal them, and show render +
        manager. ``stroke_color`` defaults to the system ROI color. Used by the 3-D
        convolution segmentation tool; returns the count added."""
        import numpy as np

        from ..core.roi import RoiRecord

        centers = np.asarray(centers, dtype=float).reshape(-1, 3)
        if centers.shape[0] == 0 or not (0 <= idx < len(self._state.datasets)):
            return 0
        stroke = stroke_color or self._system_roi_color()
        for i, (cx, cy, cz) in enumerate(centers, start=1):
            name = names[i - 1] if names is not None and i - 1 < len(names) else f"{name_prefix} {i}"
            rec = RoiRecord.create(
                "point", {"point": [float(cx), float(cy), float(cz)]},
                name=name, coordinate_space="plot", stroke_color=stroke)
            rec.context = {"dataset_idx": idx, "source": source}
            self._state.rois.add(rec)
        self._state.rois.set_show_all(True)
        self._show_render(idx)
        self._show_roi_manager()
        if log_message:
            self._state.log(log_message)
        return int(centers.shape[0])

    def add_polyline_rois(self, idx: int, paths, *, name_prefix: str, source: str,
                          stroke_color: str | None = None, names: list[str] | None = None,
                          log_message: str | None = None,
                          roi_type: str = "freehand_line") -> int:
        """Add an open poly-line (``freehand_line``) ROI tracing each path in
        *paths* (each an ``(M, 2)`` array of ``(x, y)`` nm vertices) to the ROI
        Manager, reveal them, and show render + manager. ``stroke_color`` defaults to
        the system ROI color. Returns the count added.

        Used by the curvilinear segmentation tool for traced centre lines."""
        import numpy as np

        from ..core.roi import RoiRecord

        if not (0 <= idx < len(self._state.datasets)):
            return 0
        stroke = stroke_color or self._system_roi_color()
        requested = str(roi_type or "freehand_line").strip().lower()
        if requested not in {"freehand_line", "polyline", "line"}:
            requested = "freehand_line"
        added = 0
        for i, path in enumerate(paths, start=1):
            pts = np.asarray(path, dtype=float).reshape(-1, 2)
            if pts.shape[0] < 2:
                continue
            name = names[i - 1] if names is not None and i - 1 < len(names) else f"{name_prefix} {i}"
            record_type = requested
            if requested == "line":
                record_type = "line" if pts.shape[0] == 2 else "polyline"
            rec = RoiRecord.create(
                record_type,
                {"points": pts.tolist(), "closed": False},
                name=name, coordinate_space="plot", stroke_color=stroke)
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

    def add_polygon_rois(self, idx: int, polygons, *, name_prefix: str, source: str,
                         stroke_color: str | None = None,
                         names: list[str] | None = None,
                         log_message: str | None = None) -> int:
        """Add a closed ``polygon`` ROI tracing each contour in *polygons* (each an
        ``(M, 2)`` array of ``(x, y)`` nm vertices) to the ROI Manager, reveal them,
        and show render + manager. ``stroke_color`` defaults to the system ROI
        color. Returns the count added.

        Used by the shape-model segmentation tool for fitted object contours. A
        ``polygon`` is deliberately the output type rather than the parametric
        shape: it is both a region (so masks/crop/highlighting work) and a
        vertex-editable ROI, so the fitted contour can be corrected by hand and
        then behaves like any hand-drawn ROI."""
        import numpy as np

        from ..core.roi import RoiRecord

        if not (0 <= idx < len(self._state.datasets)):
            return 0
        stroke = stroke_color or self._system_roi_color()
        added = 0
        for i, polygon in enumerate(polygons, start=1):
            pts = np.asarray(polygon, dtype=float).reshape(-1, 2)
            # A closed polygon repeats its first vertex for drawing; the record
            # stores the ring once and is marked closed.
            if pts.shape[0] > 2 and np.allclose(pts[0], pts[-1]):
                pts = pts[:-1]
            if pts.shape[0] < 3:
                continue
            name = names[i - 1] if names is not None and i - 1 < len(names) else f"{name_prefix} {i}"
            rec = RoiRecord.create(
                "polygon", {"points": pts.tolist(), "closed": True},
                name=name, coordinate_space="plot", stroke_color=stroke)
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
        view (3-D-ready). Returns the new dataset index."""
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
        # Coordinates are final/aligned — pin Z scaling factor to 1.0 and suppress auto-z.
        try:
            ds.set_z_scaling_factor(1.0, source="2D (no z correction)")
            ds.derived["z_scaling_factor"] = 1.0
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
        # A freshly-appended dataset gets a new index, so its render is created
        # fresh (stale windows from closed datasets are handled by the re-index
        # path). Do NOT close/recreate the auto-opened render here — its async tile
        # workers would deliver into a deleted window and crash. Only the render
        # view is opened by default; the user can open a scatter plot on demand.
        self._show_render(idx)
        return idx

    def add_particle_average_overlay(self, channels: dict, *, luts: dict, name: str,
                                     log_message: str | None = None) -> int | None:
        """Add a **multi-channel** averaged overlay: ``channels`` = ``{name: (N,3) nm}``
        (reference channel first). The reference is centred on its median and the
        **same** offset is applied to every channel so they stay aligned; each
        channel becomes an overlay dataset (shared group, per-channel LUT)."""
        import uuid

        import numpy as np

        from ..core.dataset import build_localization_dataset
        from ..core.overlay import overlay_color_cycle

        items = []
        for cname, pts in channels.items():
            arr = np.asarray(pts, dtype=float).reshape(-1, 3)
            arr = arr[np.all(np.isfinite(arr), axis=1)]
            if arr.shape[0]:
                items.append((str(cname), arr))
        if not items:
            return None
        ref_off = np.median(items[0][1], axis=0)          # centre by the reference channel
        overlay_id = f"pa:{uuid.uuid4().hex}"
        overlay_index = int(getattr(self, "_next_overlay_index", 1))
        self._next_overlay_index = overlay_index + 1
        cycle = overlay_color_cycle(self._state.prefs)
        prev_suspend = getattr(self._state, "suspend_auto_render", False)
        self._state.suspend_auto_render = True
        made: list[int] = []
        try:
            for order, (cname, pts) in enumerate(items, start=1):
                p = pts - ref_off
                ds = build_localization_dataset(
                    name=f"{name} · {cname}", x_nm=p[:, 0], y_nm=p[:, 1], z_nm=p[:, 2],
                    source_version="particle_average", prefs=self._state.prefs)
                ds.file.folder = f"<particle-average>/{uuid.uuid4().hex}"
                ds.metadata["particle_average"] = True
                try:
                    ds.set_z_scaling_factor(1.0, source="2D (no z correction)")
                    ds.derived["z_scaling_factor"] = 1.0
                except Exception:
                    pass
                lut = (luts or {}).get(cname) or cycle[(order - 1) % len(cycle)]
                ds.state.update({
                    "overlay_id": overlay_id, "render_group_id": overlay_id,
                    "overlay_index": overlay_index, "overlay_order": order,
                    "overlay_lut": lut, "render_channel_lut": lut})
                ds.metadata["overlay_id"] = overlay_id
                made.append(self._state.add_dataset(ds))
        finally:
            self._state.suspend_auto_render = prev_suspend
        if log_message:
            self._state.log(log_message)
        try:
            self._state.rois.deselect()
        except Exception:
            pass
        if made:
            self._show_render(made[0])
        return made[0] if made else None

    def _show_particle_average(self) -> None:
        """Analyze › Segmentation › Particle Average… — open the particle-averaging
        tool. No active dataset is required: particles can be collected from the
        active dataset + ROI Manager box ROIs, or loaded from saved particle sets /
        data+ROI pairs."""
        from .modeless import show_modeless
        from .particle_average_dialog import ParticleAverageWindow
        win = ParticleAverageWindow(self._state, self._state.active_idx, owner=self)
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
        from ..core.overlay import is_multichannel_overlay
        is_overlay = bool(group_id and is_multichannel_overlay(self._state, idx))
        if is_overlay:
            overlay_idx = ds.state.get("overlay_index")
            visible_overlay = False
            for win in (
                list(self._render_windows.values())
                + list(self._scatter_windows.values())
            ):
                try:
                    channels = getattr(win, "_channels", None)
                    if channels and any(ch.get("dataset_idx") == idx and ch.get("visible", True) for ch in channels):
                        visible_overlay = True
                        break
                except RuntimeError:
                    continue
            if visible_overlay or is_overlay:
                return f"Overlay {overlay_idx}" if overlay_idx else "Overlay"
        own_maps = (
            self._render_windows,
            self._scatter_windows,
            self._histogram_windows,
            self._attr_windows,
            self._attr_cpu_windows,
            self._filter_dlgs,
        )
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
        rect_family = {t for _l, t, _ic, _r in _RECT_FAMILY}
        oval_family = {t for _l, t, _ic, _r in _OVAL_FAMILY}
        poly_family = {t for _l, t, _ic, _r in _POLY_FAMILY}
        line_family = {t for _l, t, _ic in _LINE_FAMILY}
        # Track the active family member + refresh that button's variant icon.
        if active_tool in line_family:
            self._line_variant = active_tool
            self._update_line_button_icon()
        elif active_tool in rect_family:
            self._rect_variant = active_tool
            self._update_shape_button_icon("rectangle")
        elif active_tool in oval_family:
            self._oval_variant = active_tool
            self._update_shape_button_icon("oval")
        elif active_tool in poly_family:
            self._poly_variant = active_tool
            self._update_shape_button_icon("polygon")
        for name, action in self._roi_tool_actions.items():
            blocked = action.blockSignals(True)
            # A family button stays checked for any of its variants.
            if name == "line":
                checked = active_tool in line_family
            elif name == "rectangle":
                checked = active_tool in rect_family
            elif name == "oval":
                checked = active_tool in oval_family
            elif name == "polygon":
                checked = active_tool in poly_family
            else:
                checked = name == active_tool
            action.setChecked(checked)
            action.blockSignals(blocked)

    def _update_line_button_icon(self) -> None:
        from .. import resource_path
        icon_file = {t: ic for _l, t, ic in _LINE_FAMILY}.get(self._line_variant, "line.png")
        path = resource_path("icons", icon_file)
        if path.exists():
            self._ui.toolLine.setIcon(_adaptive_toolbar_icon(str(path)))
            self._ui.toolLine.setToolTip(
                {t: lbl for lbl, t, _ic in _LINE_FAMILY}.get(self._line_variant, "Line")
                + " — right-click to switch (straight / poly / freehand line)")

    def _install_roi_tool_menus(self) -> None:
        # The Line / Rectangle / Oval / Polygon buttons get a right-click family
        # switcher (Fiji-style). A QToolButton in a QToolBar swallows
        # ``customContextMenuRequested``, so we intercept the right mouse press
        # with an event filter instead (reliable).
        self._roi_tool_buttons: dict = {}
        for attr, key in (
            ("toolLine", "line"),
            ("toolRect", "rectangle"),
            ("toolOval", "oval"),
            ("toolPolygon", "polygon"),
        ):
            action = getattr(self._ui, attr, None)
            button = self._ui.toolbar.widgetForAction(action) if action is not None else None
            if button is not None:
                self._roi_tool_buttons[button] = key
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

    # --- Rectangle / Oval / Polygon family switchers ---
    def _shape_family(self, kind: str):
        if kind == "rectangle":
            return _RECT_FAMILY
        if kind == "oval":
            return _OVAL_FAMILY
        if kind == "polygon":
            return _POLY_FAMILY
        return ()

    def _shape_variant(self, kind: str) -> str:
        if kind == "rectangle":
            return self._rect_variant
        if kind == "oval":
            return self._oval_variant
        if kind == "polygon":
            return self._poly_variant
        return kind

    def _show_shape_family_menu(self, widget: QWidget, pos, kind: str) -> None:
        """Fiji-style right-click switcher on a shape-family toolbar button."""
        menu = QMenu(widget)
        active = self._state.rois.active_tool
        current = self._shape_variant(kind)
        for label, tool, _ic, _rot in self._shape_family(kind):
            action = menu.addAction(label)
            action.setCheckable(True)
            action.setChecked(tool == active or (active is None and tool == current))
            action.triggered.connect(
                lambda _checked=False, k=kind, t=tool: self._select_shape_variant(k, t))
        menu.exec(widget.mapToGlobal(pos))

    def _select_shape_variant(self, kind: str, tool: str) -> None:
        if kind == "rectangle":
            self._rect_variant = tool
        elif kind == "oval":
            self._oval_variant = tool
        elif kind == "polygon":
            self._poly_variant = tool
        self._update_shape_button_icon(kind)
        self._activate_roi_tool(tool)

    def _rotated_icon(self, path: str, rotate_deg: float) -> "QIcon":
        """A QIcon from *path*, optionally turned *rotate_deg*° (the rotated rectangle
        / ellipse variants reuse the base icon at 45°, no separate asset).

        The source artwork has a baked-in white matte. Normalize it (matte →
        transparent, monochrome linework tinted for the palette) before rotating
        so the toolbar background remains visible, including on dark palettes."""
        if abs(rotate_deg) < 1e-6:
            return _adaptive_toolbar_icon(path)
        pix = _adaptive_toolbar_pixmap(path)
        if pix is None:
            return QIcon(path)
        rotated = pix.transformed(
            QTransform().rotate(rotate_deg), Qt.TransformationMode.SmoothTransformation)
        canvas = QPixmap(rotated.size())
        canvas.fill(Qt.GlobalColor.transparent)
        painter = QPainter(canvas)
        painter.drawPixmap(0, 0, rotated)
        painter.end()
        return QIcon(canvas)

    def _update_shape_button_icon(self, kind: str) -> None:
        from .. import resource_path
        action_name = {
            "rectangle": "toolRect",
            "oval": "toolOval",
            "polygon": "toolPolygon",
        }.get(kind)
        action = getattr(self._ui, action_name, None) if action_name else None
        if action is None:
            return
        current = self._shape_variant(kind)
        entry = {t: (ic, rot) for _l, t, ic, rot in self._shape_family(kind)}.get(current)
        label = {t: lbl for lbl, t, _ic, _rot in self._shape_family(kind)}.get(current, kind.title())
        if entry is None:
            return
        icon_file, rot = entry
        path = resource_path("icons", icon_file)
        if not path.exists():
            return
        action.setIcon(self._rotated_icon(str(path), rot))
        switch = {
            "rectangle": "rectangle / rotated / cuboid",
            "oval": "oval / ellipse / sphere",
            "polygon": "polygon / polyhedron",
        }.get(kind, kind)
        action.setToolTip(f"{label} — right-click to switch ({switch})")

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
        """Auto-open the Log window the first time anything is logged — but placed
        so it does NOT cover the main window's menu, and without stealing focus.

        Only fires once (subsequent messages append via the window's own signal
        connection). If the user has closed the window, it is not forced back."""
        if self._log_win is not None:
            return
        from .log_window import LogWindow
        from .modeless import place_beside
        # LogWindow.__init__ replays state.log_history, so the just-logged message
        # (already in history before the signal fired) appears without a manual add.
        self._log_win = LogWindow(self._state)
        # Dock below the main window (falling back to right/above/left, then
        # clamped) so the menu bar at the top stays uncovered on any screen size.
        place_beside(self._log_win, self, prefer=("below", "right", "above", "left"))
        self._log_win.show()
        # Keep the main window in front/active so its menus remain accessible; the
        # log sits beside/below it, visible but not on top of the menu.
        self.raise_()
        self.activateWindow()

    def _show_console(self) -> None:
        """Open (or raise) the Console window — raw stdout / stderr stream."""
        from .console_window import ConsoleWindow
        if self._console_win is None or not self._console_win.isVisible():
            if self._console_win is None:
                self._console_win = ConsoleWindow()
            self._console_win.show()
        self._console_win.raise_()
        self._console_win.activateWindow()

    def _render_window_for_dataset(self, idx: int | None):
        """The open render window showing dataset *idx* — either keyed by it, or
        (in an **overlay**) one whose channels include it. None if none is open.

        An overlay render window is stored in ``_render_windows`` under only its
        anchor/primary dataset index, but displays every channel. Keying strictly
        by the active index therefore misses the window when a non-primary channel
        is active — the cause of a duplicate overlay view being spawned."""
        if idx is None:
            return None
        win = self._render_windows.get(idx)
        if win is not None:
            return win
        for win in self._render_windows.values():
            try:
                if any(ch.get("dataset_idx") == idx for ch in getattr(win, "_channels", [])):
                    return win
            except Exception:
                continue
        return None

    def _show_brightness_contrast(self) -> None:
        """
        Forward to the render window of the currently active dataset.
        The active dataset follows whichever render window is focused (#2).
        """
        # The Shift+C shortcut is application-wide. When a window that owns its own
        # brightness/contrast is focused (a render view, the standalone TIFF/image
        # viewer), route to it rather than the active dataset's render — so the key
        # means "adjust what I'm looking at". This is why those windows no longer
        # need their own Shift+C QShortcut (which would be an ambiguous overload).
        active = QApplication.activeWindow()
        own_bc = getattr(active, "_show_brightness_contrast", None) if (
            active is not None and active is not self) else None
        if active is not None and callable(own_bc):
            try:
                own_bc()
                active.raise_()
                return
            except Exception as exc:
                self._state.log(f"Brightness/Contrast failed for focused window: {exc}", "ERROR")
        if self._state.active_dataset is None:
            self._no_data_warning()
            return
        idx = self._state.active_idx
        # Find the window that already displays this dataset (incl. as an overlay
        # channel) before opening a new one — otherwise pressing Shift+C on a
        # non-primary overlay channel spawns a duplicate overlay view.
        active = QApplication.activeWindow()
        if (
            getattr(active, "TAG", None) == "render_window"
            and any(
                ch.get("dataset_idx") == idx
                for ch in getattr(active, "_channels", [])
            )
        ):
            rwin = active
        else:
            rwin = self._render_window_for_dataset(idx)
        if rwin is None:
            self._show_render(idx)
            rwin = self._render_window_for_dataset(idx)
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
        if dlg.exec() == QDialog.DialogCode.Accepted:
            # After OK the dialog has already written prefs + saved; rebuild
            # any UI bits that depend on them.
            self._apply_shortcuts()
            self._populate_recent_menu()
            self._refresh_plot_preferences()

    def _refresh_plot_preferences(self) -> None:
        windows = (
            list(self._render_windows.values())
            + list(self._scatter_windows.values())
            + list(self._histogram_windows.values())
            + list(self._attr_windows.values())
            + list(self._attr_cpu_windows.values())
        )
        for win in windows:
            refresh = getattr(win, "refresh_preferences", None)
            if callable(refresh):
                refresh()

    def _on_colors_changed(self, payload) -> None:
        """Repaint only consumers touched by a global color change."""
        from ..colors import normalize_rgba, pg_safe_hex, rgba_hex
        from ..core.overlay import dataset_group_id, overlay_color_cycle

        payload = payload if isinstance(payload, dict) else {}
        paths = set(payload.get("paths", ()))
        if not paths:
            return

        solid_changed = any(path.startswith("solid.") for path in paths)
        overlay_changed = any(path.startswith("viewer.overlay.") for path in paths)
        roi_changed = any(p.startswith("viewer.roi") for p in paths)
        attribute_changed = (
            any(path.startswith("viewer.attribute_") for path in paths)
            or any(path.startswith("functions.Iteration series.") for path in paths)
        )
        histogram_changed = (
            any(path.startswith("viewer.histogram_") for path in paths)
            or any(path.startswith("viewer.filter_") for path in paths)
            or any(path.startswith("functions.Iteration series.") for path in paths)
        )

        def migrate_solid_lut(value):
            if not solid_changed or not isinstance(value, str):
                return value
            text = value.strip()
            if text.startswith("solid:custom:"):
                return value
            prefixed = text.startswith("solid:")
            name = text[6:] if prefixed else text
            renames = payload.get("solid_renames", {})
            if isinstance(renames, dict) and name in renames:
                renamed = str(renames[name])
                return f"solid:{renamed}" if prefixed else renamed
            previous_solid = payload.get("previous", {}).get("solid", {})
            current_solid = payload.get("current", {}).get("solid", {})
            if (
                isinstance(previous_solid, dict)
                and isinstance(current_solid, dict)
                and name in previous_solid
                and name not in current_solid
            ):
                return f"solid:custom:{rgba_hex(previous_solid[name])}"
            return value

        if solid_changed:
            for ds in self._state.datasets:
                for key in ("overlay_lut", "render_channel_lut"):
                    if key in ds.state:
                        ds.state[key] = migrate_solid_lut(ds.state[key])
            for win in (*self._render_windows.values(), *self._scatter_windows.values()):
                for channel in getattr(win, "_channels", ()):
                    if isinstance(channel, dict) and "lut" in channel:
                        channel["lut"] = migrate_solid_lut(channel["lut"])
                combo = getattr(win, "_cmap_combo", None)
                if combo is not None:
                    current_lut = combo.currentText()
                    migrated_lut = migrate_solid_lut(current_lut)
                    if migrated_lut != current_lut:
                        blocked = combo.blockSignals(True)
                        try:
                            combo.setCurrentText(migrated_lut)
                        finally:
                            combo.blockSignals(blocked)

        if overlay_changed:
            cycle = overlay_color_cycle(self._state.prefs)
            for ds in self._state.datasets:
                if not dataset_group_id(ds):
                    continue
                try:
                    position = max(0, int(ds.state.get("overlay_order", 1)) - 1)
                except (TypeError, ValueError):
                    position = 0
                lut = cycle[position % len(cycle)]
                ds.state["overlay_lut"] = lut
                ds.state["render_channel_lut"] = lut

        if roi_changed:
            previous = payload.get("previous", {}).get("viewer", {}).get("roi_edge")
            current = payload.get("current", {}).get("viewer", {}).get("roi_edge")
            old_rgba = normalize_rgba(previous)
            new_color = pg_safe_hex(current)

            def _tracks_system_color(stroke) -> bool:
                """Whether this ROI still carries the outgoing system color.

                Compared through ``pg_safe_hex`` so a record written before the
                #AARRGGBB/#RRGGBBAA mix-up was fixed is still recognised — those
                would otherwise never match and could never be recolored back.
                """
                return normalize_rgba(pg_safe_hex(stroke)) == old_rgba

            records_changed = False
            for record in self._state.rois.records:
                if _tracks_system_color(record.stroke_color):
                    record.stroke_color = new_color
                    records_changed = True
            adapters = []
            for win in (
                *self._render_windows.values(),
                *self._scatter_windows.values(),
                *self._histogram_windows.values(),
                *self._attr_windows.values(),
                *self._attr_cpu_windows.values(),
            ):
                adapter = getattr(win, "_roi_overlay", None)
                if adapter is not None and adapter not in adapters:
                    adapters.append(adapter)
            for adapter in adapters:
                draft = getattr(adapter, "draft", None)
                if draft is not None and _tracks_system_color(draft.stroke_color):
                    draft.stroke_color = new_color
                for record in getattr(adapter, "_session_points", ()):
                    if _tracks_system_color(record.stroke_color):
                        record.stroke_color = new_color
                try:
                    adapter.refresh()
                except Exception:
                    pass
            if records_changed:
                self._state.rois.changed.emit()

        if attribute_changed:
            for win in (
                *self._attr_windows.values(),
                *self._attr_cpu_windows.values(),
            ):
                refresh = getattr(win, "refresh_colors", None)
                if callable(refresh):
                    refresh()
        if histogram_changed:
            for win in self._histogram_windows.values():
                refresh = getattr(win, "refresh_colors", None)
                if callable(refresh):
                    refresh()

        if solid_changed or overlay_changed:
            for win in self._render_windows.values():
                refresh = getattr(win, "refresh_global_colors", None)
                if callable(refresh):
                    refresh(reset_overlay=overlay_changed)
            for win in self._scatter_windows.values():
                refresh = getattr(win, "refresh_global_colors", None)
                if callable(refresh):
                    refresh(reset_overlay=overlay_changed)

        if solid_changed or any(
            path.startswith(("functions.", "plugins.")) for path in paths
        ):
            for win in list(getattr(self, "_modeless_windows", ())):
                refresh = getattr(win, "refresh_colors", None)
                if callable(refresh):
                    try:
                        refresh()
                    except RuntimeError:
                        pass

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

    def _show_plot_profile(self) -> None:
        """ImageJ-style Plot Profile: localization density along a line / polyline /
        freehand-line ROI in a **render or scatter** view, live-updating with the ROI
        and a tunable width band. Samples the localizations directly, so it is
        independent of the render viewport/zoom."""
        idx = self._state.active_idx
        if self._state.active_dataset is None or idx is None:
            QMessageBox.information(self, "Plot Profile", "Open a dataset first.")
            return
        # Prefer the focused 2-D coordinate view (render or scatter); fall back to the
        # active dataset's render window.
        win = self._active_coordinate_view()
        if win is None or not hasattr(win, "profile_localizations"):
            win = self._render_windows.get(idx) or self._render_window_for_dataset(idx)
        if win is None or win.coordinate_view_box() is None or not hasattr(win, "profile_localizations"):
            self._state.log("Plot Profile: no 2-D localization view for the active dataset.", "WARN")
            QMessageBox.information(
                self, "Plot Profile",
                "Open a render or scatter view of the active dataset in a 2-D "
                "orientation (XY / XZ / YZ) first.")
            return
        # No open-line ROI yet → activate the line tool so the user can draw one and
        # watch the profile appear live.
        overlay = getattr(win, "_roi_overlay", None)
        if overlay is not None and overlay.active_open_line_record() is None:
            try:
                overlay.activate()
                self._state.rois.set_tool("line")
                self._state.log("Plot Profile: draw a line on the view to profile it.", "INFO")
            except Exception:
                pass
        from . import modeless
        from .plot_profile_dialog import PlotProfileDialog
        modeless.show_modeless(PlotProfileDialog(win, owner=self), self)

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
        from ..colors import solid_color_names

        fallback_luts = list(solid_color_names())
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

    def _flatten_active_channel_group(self) -> None:
        """Process › Channel › Flatten — combine the active multi-channel overlay
        into a single non-overlay dataset.

        Each channel's localizations are concatenated **in place** (overlay
        transform baked into display nm), trace ids are remapped to stay distinct,
        and combinable per-loc attributes are merged; conflicting / stale ones are
        dropped and reported to the Log. The result renders with the ``hot`` LUT
        and can feed convolution / particle analysis on the combined cloud."""
        import uuid

        from ..core.channel_flatten import flatten_overlay
        from ..core.dataset import build_localization_dataset
        from ..core.overlay import dataset_group_id, overlay_members

        active_idx = self._state.active_idx
        if active_idx is None or not (0 <= active_idx < len(self._state.datasets)):
            self._no_data_warning()
            return
        active = self._state.datasets[active_idx]
        if not dataset_group_id(active):
            QMessageBox.information(
                self, "Flatten channels",
                "The active dataset is not part of a multi-channel overlay.")
            return
        members = overlay_members(self._state, active_idx)
        if len(members) < 2:
            QMessageBox.information(
                self, "Flatten channels",
                "The active overlay contains only one dataset.")
            return

        datasets = [ds for _, ds in members]
        res = flatten_overlay(datasets)
        if res.coords_nm.shape[0] == 0:
            QMessageBox.information(
                self, "Flatten channels",
                "No localizations to flatten (all filtered out?).")
            return

        base = getattr(active, "name", "overlay")
        ds = build_localization_dataset(
            name=f"{base} (flattened {len(members)}ch)",
            x_nm=res.coords_nm[:, 0], y_nm=res.coords_nm[:, 1], z_nm=res.coords_nm[:, 2],
            tid=res.tid, attrs=res.attrs, source_version="flattened",
            prefs=self._state.prefs)
        ds.file.folder = f"<flattened>/{uuid.uuid4().hex}"
        ds.metadata["flattened_overlay"] = True
        ds.metadata["flattened_channels"] = [getattr(d, "name", "") for d in datasets]
        ds.state["render_channel_lut"] = "hot"     # single-channel render LUT
        # Coordinates are final display nm (each channel's Z scaling factor already baked) —
        # pin Z scaling factor to 1.0 and suppress the post-load auto-z estimate.
        try:
            ds.set_z_scaling_factor(1.0, source="flattened (z baked)")
            ds.derived["z_scaling_factor"] = 1.0
        except Exception:
            pass

        r = res.report
        self._state.log(
            f"Flatten channels: combined {r.n_channels} channel(s) → {r.n_kept:,} of "
            f"{r.n_total:,} localization(s), {r.n_traces:,} trace(s). "
            f"Kept attributes: {', '.join(r.kept_attrs) or 'none'}.")
        for note in r.notes:
            self._state.log(f"Flatten: {note}.")
        for attr, reason in r.dropped:
            self._state.log(f"Flatten: dropped '{attr}' — {reason}.", "WARN")

        idx = self._state.add_dataset(ds)
        self._show_render(idx)

    def _revert_overlay_to_original(self) -> None:
        """Process › Channel › Convert Dataset to Multi-Channel Overlay › Revert
        Overlay to Original Dataset — the inverse of a channel separation.

        Removes the active overlay's channel datasets and restores the single
        source dataset. When the source is still open (separation keeps it) it is
        re-activated exactly; otherwise it is reconstructed by concatenating the
        channels (:func:`analysis.attribute_channels.reconstruct_from_channels`).
        Only offered for separation overlays (DCR / time / by-attribute); an MSR
        multi-channel overlay has no single original — use Flatten instead."""
        from ..core.overlay import dataset_group_id, overlay_members

        idx = self._state.active_idx
        if idx is None or not (0 <= idx < len(self._state.datasets)):
            self._no_data_warning()
            return
        active = self._state.datasets[idx]
        group_id = dataset_group_id(active)
        if not group_id:
            QMessageBox.information(
                self, "Revert Overlay",
                "The active dataset is not part of a multi-channel overlay.")
            return
        members = overlay_members(self._state, idx)
        member_objs = [ds for _, ds in members]
        if len(member_objs) < 2:
            QMessageBox.information(
                self, "Revert Overlay", "The active overlay contains only one dataset.")
            return
        if not all(m.metadata.get("separated_from") for m in member_objs):
            QMessageBox.information(
                self, "Revert Overlay",
                "This overlay was not created by a channel separation, so it has no "
                "single original dataset to revert to. Use Flatten to combine it.")
            return

        base_name = member_objs[0].metadata.get("separated_from") or "dataset"

        def index_of(obj):
            return next((i for i, d in enumerate(self._state.datasets) if d is obj), None)

        # Prefer the still-present source (exact restore); else reconstruct.
        source = None
        for d in self._state.datasets:
            if d in member_objs:
                continue
            if group_id in (d.metadata.get("produced_overlays") or []):
                source = d
                break
        if source is None:                          # name-match fallback
            for d in self._state.datasets:
                if d in member_objs or dataset_group_id(d):
                    continue
                if getattr(d, "name", None) == base_name:
                    source = d
                    break

        reconstructed = None
        if source is None:
            from ..analysis.attribute_channels import reconstruct_from_channels
            from ..core.dataset import build_localization_dataset
            rec = reconstruct_from_channels(member_objs)
            if not rec or rec.get("x_nm") is None or rec["x_nm"].size == 0:
                QMessageBox.warning(
                    self, "Revert Overlay",
                    "Could not reconstruct the original dataset from the channels.")
                return
            reconstructed = build_localization_dataset(
                name=base_name, x_nm=rec["x_nm"], y_nm=rec["y_nm"], z_nm=rec["z_nm"],
                tid=rec["tid"], attrs=rec["attrs"], source_version="reverted",
                prefs=self._state.prefs)
            try:
                reconstructed.set_z_scaling_factor(1.0, source="reverted (z baked)")
                reconstructed.derived["z_scaling_factor"] = 1.0
            except Exception:
                pass
            reconstructed.metadata["reverted_from_overlay"] = group_id

        # Remove the channel datasets (descending index so lower ones stay valid).
        member_idxs = sorted(
            (i for i in (index_of(m) for m in member_objs) if i is not None), reverse=True)
        previous = getattr(self._state, "suspend_auto_render", False)
        self._state.suspend_auto_render = True
        try:
            for i in member_idxs:
                self._state.remove_dataset(i)
        finally:
            self._state.suspend_auto_render = previous

        if source is not None and source in self._state.datasets:
            try:
                (source.metadata.get("produced_overlays") or []).remove(group_id)
            except ValueError:
                pass
            si = index_of(source)
            if si is not None:
                self._state.set_active(si)
                self._show_render(si)
            self._state.log(
                f"Reverted overlay to original dataset '{source.name}' "
                f"({len(member_objs)} channel(s) removed).")
        else:
            new_idx = self._state.add_dataset(reconstructed)
            self._state.set_active(new_idx)
            self._show_render(new_idx)
            self._state.log(
                f"Reverted overlay by reconstructing '{base_name}' from "
                f"{len(member_objs)} channel(s).")
        self._notify_view_state_changed()

    def _aggregate_active_dataset(self) -> None:
        """Process › Aggregate Localizations — Imspector-style per-trace photon
        binning of the active dataset into a new aggregated dataset.

        Within each trace, valid final localizations are accumulated in time order
        until their combined photon count (``eco`` over the final-scale iterations)
        reaches the threshold, then emitted as one averaged localization. The
        original dataset is kept; the result is a new ``(aggregated)`` dataset.
        Reproduces Abberior Imspector's *Aggregation* to ≈99 %."""
        import uuid

        from ..core.dataset_kind import is_minflux
        from ..core.mfx_sequence import photon_iterations_for_dataset
        from ..core.overlay import dataset_group_id, overlay_members
        from ..analysis.aggregation import raw_dict_from_dataset

        active_idx = self._state.active_idx
        if active_idx is None or not (0 <= active_idx < len(self._state.datasets)):
            self._no_data_warning()
            return
        ds = self._state.datasets[active_idx]
        if not is_minflux(ds):
            QMessageBox.information(
                self, "Aggregate Localizations",
                "The active dataset is not a MINFLUX dataset (no trace ids).")
            return
        if raw_dict_from_dataset(ds) is None:
            QMessageBox.information(
                self, "Aggregate Localizations",
                "Aggregation needs raw all-iteration MINFLUX data with trace, "
                "time, photon and coordinate fields. Load the original .msr / "
                "raw .mat and try again.")
            return

        # Overlay → aggregate every channel with the same threshold and keep them
        # linked as a new overlay; single dataset → just that one.
        group_id = dataset_group_id(ds)
        if group_id:
            members = [m for _, m in overlay_members(self._state, active_idx)]
        else:
            members = [ds]
        aggregatable = [m for m in members
                        if is_minflux(m) and raw_dict_from_dataset(m) is not None]
        if not aggregatable:
            QMessageBox.information(
                self, "Aggregate Localizations",
                "No channel in this overlay has the raw data aggregation needs.")
            return

        pit = photon_iterations_for_dataset(ds)
        thr = self._ask_aggregation_threshold(ds, pit, n_channels=len(aggregatable))
        if thr is None:
            return

        self._state.status_progress("Aggregating localizations")
        if not group_id:
            new = self._build_aggregated_dataset(ds, thr)
            if new is None:
                self._state.status_progress("Aggregation produced no localizations")
                QMessageBox.information(self, "Aggregate Localizations",
                                        "No valid localizations to aggregate.")
                return
            idx = self._state.add_dataset(new)
            self._log_aggregated_dataset(new, idx)
            self._state.status_progress("Aggregation done", 1.0)
            self._show_render(idx)
            return

        # Multi-channel overlay: aggregate each channel, re-link as a new overlay
        # (preserving each channel's LUT + alignment transform).
        new_overlay_id = f"agg:{uuid.uuid4().hex}"
        overlay_index = int(getattr(self, "_next_overlay_index", 1))
        self._next_overlay_index = overlay_index + 1
        prev_suspend = getattr(self._state, "suspend_auto_render", False)
        self._state.suspend_auto_render = True
        made: list[int] = []
        skipped: list[str] = []
        try:
            for order, m in enumerate(members, start=1):
                if m not in aggregatable:
                    skipped.append(getattr(m, "name", "?"))
                    continue
                new = self._build_aggregated_dataset(m, thr)
                if new is None:
                    skipped.append(getattr(m, "name", "?"))
                    continue
                lut = m.state.get("overlay_lut") or m.state.get("render_channel_lut")
                new.state.update({
                    "overlay_id": new_overlay_id, "render_group_id": new_overlay_id,
                    "overlay_index": overlay_index,
                    "overlay_order": int(m.state.get("overlay_order", order)),
                })
                if lut:
                    new.state["overlay_lut"] = lut
                    new.state["render_channel_lut"] = lut
                # Carry the channel's alignment (applied at display time) so the
                # aggregated channels stay registered exactly as before.
                for tkey in ("overlay_transform", "render_transform_2d"):
                    tval = m.state.get(tkey)
                    if tval is not None:
                        new.state[tkey] = tval
                        new.metadata[tkey] = tval
                new.metadata["overlay_id"] = new_overlay_id
                new_idx = self._state.add_dataset(new)
                made.append(new_idx)
                self._log_aggregated_dataset(new, new_idx)
        finally:
            self._state.suspend_auto_render = prev_suspend

        if not made:
            self._state.status_progress("Aggregation produced no localizations")
            QMessageBox.information(self, "Aggregate Localizations",
                                    "No valid localizations to aggregate.")
            return
        if skipped:
            self._state.log(
                f"Aggregation: skipped {len(skipped)} channel(s) without usable "
                f"raw data: {', '.join(skipped)}.", "WARN")
        self._state.log(
            f"Aggregated overlay: {len(made)} channel(s) at photon threshold "
            f"{int(thr)} → new overlay.")
        self._state.status_progress("Aggregation done", 1.0)
        self._show_render(made[0])

    def _build_aggregated_dataset(self, ds, thr):
        """Build (not add) an aggregated dataset from *ds* at photon threshold
        *thr*, with the Z scaling factor + aggregation provenance but no overlay state.
        Returns the new dataset, or ``None`` when nothing aggregates."""
        import uuid

        import numpy as np

        from ..core.dataset import build_localization_dataset
        from ..core.mfx_sequence import photon_iterations_for_dataset
        from ..analysis.aggregation import aggregate_dataset, aggregation_time_mode

        pit = photon_iterations_for_dataset(ds)
        time_mode = aggregation_time_mode(ds)
        res = aggregate_dataset(ds, photon_threshold=float(thr),
                                photon_iters=pit.photon_iters,
                                time_mode=time_mode)
        if res is None or res["tid"].size == 0:
            return None

        loc = res["loc"]                       # metres, (M, 3) — raw (un-Z scaling factor) z
        n_contributing = int(np.asarray(res["n"], dtype=np.int64).sum())
        attrs = {"eco": res["eco"], "n_agg": res["n"].astype(float)}
        for key in ("ecc", "efo", "efc", "fbg", "dcr"):
            if key in res:
                attrs[key] = res[key]
        source_z_scaling_factor = float(getattr(ds.cali, "z_scaling_factor", 1.0) or 1.0)

        new = build_localization_dataset(
            name=f"{ds.name} (aggregated {int(thr)})",
            x_nm=loc[:, 0] * 1e9, y_nm=loc[:, 1] * 1e9, z_nm=loc[:, 2] * 1e9,
            tid=res["tid"], tim=res["tim"], attrs=attrs,
            source_version="aggregated", prefs=self._state.prefs)
        try:
            new.set_z_scaling_factor(source_z_scaling_factor, source="aggregated (from raw z)")
            new.derived["z_scaling_factor"] = np.asarray([source_z_scaling_factor], dtype=float)
        except Exception:
            pass
        new.file.folder = f"<aggregated>/{uuid.uuid4().hex}"
        new.metadata["aggregated"] = True
        new.metadata["aggregation"] = {
            "photon_threshold": float(thr),
            "photon_iterations": list(pit.photon_iters),
            "photon_iter_source": pit.source,
            "photon_attribute": "eco",
            "coordinate_mode": "photon_weighted_centroid",
            "timestamp_mode": time_mode,
            "grouping": "per_trace_time_order",
            "valid_final_only": True,
            "trailing_remainder": "retained",
            "source": ds.name,
            "n_in": n_contributing,
            "n_out": int(res["tid"].size),
        }
        return new

    def _log_aggregated_dataset(self, ds, dataset_idx: int) -> None:
        """Record a parseable, result-tagged aggregation event in the Log."""
        agg = ds.metadata.get("aggregation", {})
        threshold = float(agg.get("photon_threshold", 0.0))
        threshold_text = f"{threshold:g}"
        self._state.log(
            f"Aggregated '{agg.get('source', 'the source dataset')}' into '{ds.name}': "
            f"{int(agg.get('n_in', 0)):,} -> {int(agg.get('n_out', 0)):,} "
            f"localizations; photon threshold = {threshold_text} photons per "
            f"aggregated localization; photon iterations = "
            f"{list(agg.get('photon_iterations', []))} "
            f"({agg.get('photon_iter_source', 'unknown')}); position = "
            f"photon-weighted centroid; timestamp mode = "
            f"{agg.get('timestamp_mode', 'photon_weighted')}; valid final "
            f"localizations grouped per trace in time order; trailing remainder retained.",
            dataset_idx=dataset_idx,
        )

    def _ask_aggregation_threshold(self, ds, pit, n_channels: int = 1) -> "int | None":
        """Modal photon-threshold picker for aggregation (default 3000)."""
        from PyQt6.QtWidgets import (
            QDialog, QDialogButtonBox, QLabel, QSpinBox, QVBoxLayout)
        from ..analysis.aggregation import aggregation_time_mode

        dlg = QDialog(self)
        dlg.setWindowTitle("Aggregate Localizations")
        dlg.setMinimumWidth(620)
        lay = QVBoxLayout(dlg)
        ax = "" if pit.axial_iter is None else f", axial itr {pit.axial_iter}"
        time_mode = aggregation_time_mode(ds)
        if time_mode == "first":
            time_text = (
                "The aggregate timestamp is taken from the first contributing "
                "localization (legacy nested-record convention)."
            )
        else:
            time_text = (
                "The aggregate timestamp is the photon-weighted mean of the "
                "contributing localization timestamps (modern flat-record convention)."
            )
        overlay_note = (f"<i>Overlay: all {n_channels} channels will be aggregated "
                        f"and kept as an overlay.</i><br>" if n_channels > 1 else "")
        info = QLabel(
            f"<b>{ds.name}</b><br>{overlay_note}{ds.prop.num_loc:,} localizations · "
            f"{ds.prop.num_traces:,} traces · {ds.prop.num_dim}-D<br>"
            f"Per-localization photon count <i>P</i> is the sum of background-corrected "
            f"effective counts (<code>eco</code>) over final-scale iterations "
            f"{pit.photon_iters} (lateral itr {pit.lateral_iter}{ax}; "
            f"{pit.source}; 0-based raw iteration indices).<br><br>"
            "Valid final localizations are processed separately for each trace and "
            "ordered by time. Consecutive localizations are accumulated until their "
            "combined photon count reaches or exceeds the threshold below. Complete "
            "localizations are not split, so a bin can exceed the threshold; the final "
            "sub-threshold remainder of each trace is retained.<br><br>"
            "The resulting coordinate is the photon-weighted centroid "
            "Σ(<i>P</i><sub>i</sub> · <i>r</i><sub>i</sub>) / "
            "Σ<i>P</i><sub>i</sub>, and its photon count is Σ<i>P</i><sub>i</sub>. "
            f"{time_text} The source dataset remains unchanged.")
        info.setWordWrap(True)
        lay.addWidget(info)
        row = QLabel("Photon threshold per aggregated localization:")
        lay.addWidget(row)
        spin = QSpinBox()
        spin.setRange(1, 10_000_000)
        spin.setSingleStep(500)
        spin.setSuffix(" photons")
        # Remember the last-used threshold across sessions (default 3000).
        try:
            last = int(self._state.prefs.get("aggregation", {}).get(
                "photon_threshold", 3000))
        except Exception:
            last = 3000
        spin.setValue(max(1, min(10_000_000, last)))
        lay.addWidget(spin)
        bb = QDialogButtonBox(
            QDialogButtonBox.StandardButton.Ok | QDialogButtonBox.StandardButton.Cancel)
        bb.accepted.connect(dlg.accept)
        bb.rejected.connect(dlg.reject)
        lay.addWidget(bb)
        if dlg.exec() != QDialog.DialogCode.Accepted:
            return None
        value = int(spin.value())
        try:
            self._state.prefs.setdefault("aggregation", {})["photon_threshold"] = value
            self._state.save_prefs()
        except Exception:
            pass
        return value

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
        self.close_datasets([idx])

    def close_datasets(self, indices) -> None:
        """Close (remove) every dataset in *indices*, and its windows.

        Shared by ``Ctrl+W`` (the active dataset) and the Dataset Manager's
        *Close* button / multi-selection *Close all*.  Removal runs highest
        index first so the remaining indices stay valid while iterating.
        """
        from ..core.overlay import dataset_group_id

        wanted = sorted(
            {int(i) for i in indices if 0 <= int(i) < len(self._state.datasets)},
            reverse=True,
        )
        if not wanted:
            return

        touched_group = False
        for idx in wanted:
            group_id = dataset_group_id(self._state.datasets[idx])
            if group_id:
                touched_group = True
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
        if touched_group:
            self._refresh_overlay_windows()

    def duplicate_datasets(self, datasets) -> None:
        """Plain-duplicate every dataset in *datasets* (Dataset Manager
        multi-selection *Duplicate all*).

        Whole-dataset copies: the ROI-crop duplicate is a single-dataset command
        (``Shift+D``), since a crop is defined by one dataset's active ROI.
        Datasets are passed by identity, not index — each ``add_dataset`` grows
        the list, so an index captured before the loop would not survive it.
        """
        for ds in list(datasets):
            self._plain_duplicate(ds)

    def combine_datasets_as_overlay(self, indices) -> None:
        """Open *Process › Channel… › Combine* restricted to *indices*."""
        self._show_channel_combine(list(indices))

    def reset_dataset(self, idx: int) -> None:
        """Dataset Manager *Reset* — put the dataset back to its as-loaded state.

        Filters, ROI selection masks, Z scaling factor and the live view layer (LUT /
        transform) all revert; overlay *membership* is kept, because a per-dataset
        reset must not dissolve a group of channels.  See
        ``core/dataset_reset.py`` for the rule.
        """
        if not (0 <= idx < len(self._state.datasets)):
            return
        from ..core.dataset_reset import reset_dataset as _reset

        ds = self._state.datasets[idx]
        try:
            changes = _reset(ds)
        except Exception as exc:
            self._state.log(f"Reset '{ds.name}' failed: {exc}", "ERROR")
            QMessageBox.critical(self, "Reset dataset", str(exc))
            return

        if not changes:
            self._state.log(f"Reset '{ds.name}': already in its as-loaded state.",
                            dataset_idx=idx)
            return
        self._state.log(f"Reset '{ds.name}': " + "; ".join(changes) + ".", dataset_idx=idx)
        # Filters / ROI / Z scaling factor each drive a different set of views.
        self._state.notify_filter_changed(idx)
        self._state.notify_calibration_changed(idx)
        self._state.notify_roi_selection_changed(idx)
        self._refresh_overlay_windows()

    def view_dataset_mbm(self, idx: int) -> None:
        """Dataset Manager *View mbm info…* — the dataset's beam-monitoring beads.

        Opens the combined MBM window (drift traces + beads vs the data region).
        The beads travel with the dataset (``ds.mbm`` / ``metadata["mbm_points"]``
        from the MSR import or an ``.msr`` round trip), so no source file is
        re-parsed.
        """
        if not (0 <= idx < len(self._state.datasets)):
            return
        from .mbm_info_window import open_mbm_info

        ds = self._state.datasets[idx]
        win, reason = open_mbm_info(self, ds)
        if win is None:
            self._state.log(f"View mbm info: {reason.splitlines()[0]}", "WARN")
            QMessageBox.information(self, "MBM info", reason)
            return
        self._state.log(
            f"MBM info for '{ds.name}': {len(win._beads)} bead(s).", dataset_idx=idx)

    def view_dataset_image_series(self, idx: int) -> None:
        """Dataset Manager *View image series* — open this dataset's source ``.msr``
        images in the shared image viewer.

        One viewer per file (every series is in its Series dropdown); the series
        rendered *from this dataset* — matched on the OBF footer's ``source`` did
        — is the one selected on open.
        """
        if not (0 <= idx < len(self._state.datasets)):
            return
        ds = self._state.datasets[idx]
        # A sealed .zarr.zip holds its images inside the archive, so the record's
        # absolute_path does not exist on disk; materialize_image extracts the
        # member on demand. Without it the empty list fell through to reopening
        # the source .msr, which looked like it worked while that file was still
        # where it was imported from -- and showed nothing once it moved.
        from ..core.minflux_zarr import materialize_image

        embedded: list[Path] = []
        labels: list[str] = []
        did = str(ds.metadata.get("msr_dataset_did") or "")
        selected = 0
        for item in (ds.metadata.get("minflux_viewer_images") or []):
            if not isinstance(item, dict):
                continue
            resolved = materialize_image(item)
            if resolved is None:
                continue
            # Prefer the series rendered from THIS dataset, exactly as the .msr
            # path does; it becomes the entry shown when the viewer opens.
            if did and str(item.get("source_did") or "") == did:
                selected = len(embedded)
            embedded.append(resolved)
            labels.append(str(item.get("name") or resolved.stem))
        if embedded:
            from ..core.tiff_source import EmbeddedImageSource

            # ONE viewer listing every embedded image in its Series dropdown --
            # the same shape as the .msr path. Opening a bare TiffImageSource on
            # embedded[0] showed only the first image, and named it by the temp
            # file it had been extracted to.
            try:
                source = EmbeddedImageSource(embedded, labels, series_index=selected)
                store = str(ds.metadata.get("minflux_viewer_zarr_path")
                            or ds.metadata.get("minflux_viewer_project_path")
                            or embedded[0].parent)
                self._open_image_viewer(
                    source, f"{store}#embedded-images",
                    initial_series_index=selected,
                )
            except Exception as exc:
                self._state.log(f"View embedded image failed for '{ds.name}': {exc}", "ERROR")
                QMessageBox.critical(self, "View image series", str(exc))
                return
            self._state.log(
                f"Embedded images of '{ds.name}': {len(embedded)} image(s), "
                f"showing '{labels[selected]}'.", dataset_idx=idx,
            )
            return
        # A dataset restored from a .zarr store carries its own images as
        # OME-TIFFs inside that store. It must NOT reach for the source .msr:
        # the store is meant to stand alone, and the .msr may have moved, been
        # renamed, or never travelled with it. Reporting the empty store is
        # honest; borrowing images from a neighbouring file is not.
        store_path = str(ds.metadata.get("minflux_viewer_zarr_path")
                         or ds.metadata.get("minflux_viewer_project_path") or "")
        if store_path:
            missing = len(ds.metadata.get("minflux_viewer_images") or [])
            detail = (
                f"Its {missing} image record(s) could not be read from the store."
                if missing else "It contains no image series."
            )
            self._state.log(
                f"View image series: '{ds.name}' came from {Path(store_path).name}; "
                f"{detail.lower()}", "WARN", dataset_idx=idx)
            QMessageBox.information(
                self, "View image series",
                f"'{ds.name}' was opened from\n{store_path}\n\n{detail}\n\n"
                "A MINFLUX Viewer Zarr store carries its own images, so no "
                "source .msr is consulted.",
            )
            return
        source_path = Path(str(ds.metadata.get("msr_source_path", "") or ""))
        if not source_path.is_file():
            self._state.log(
                f"View image series: no source .msr for '{ds.name}'.", "WARN")
            QMessageBox.information(
                self, "View image series",
                f"'{ds.name}' has no available source .msr file, so it has no "
                "image series.\n\nImage series come from the .msr the dataset was "
                "imported from.",
            )
            return

        from ..core.obf_image_source import ObfImageSource, list_obf_image_series

        try:
            series = list_obf_image_series(source_path)
        except Exception as exc:
            self._state.log(f"View image series failed for '{ds.name}': {exc}", "ERROR")
            QMessageBox.critical(self, "View image series", str(exc))
            return
        if not series:
            self._state.log(
                f"View image series: '{source_path.name}' has no image series.", "WARN")
            QMessageBox.information(
                self, "View image series",
                f"'{source_path.name}' contains no viewable image series.",
            )
            return

        did = str(ds.metadata.get("msr_dataset_did", "") or "")
        wanted = next(
            (s for s in series if did and str(s.get("source_did") or "") == did),
            series[0],
        )
        try:
            src = ObfImageSource(source_path, raw_stack_index=int(wanted["raw_index"]))
        except Exception as exc:
            self._state.log(f"View image series failed for '{ds.name}': {exc}", "ERROR")
            QMessageBox.critical(self, "View image series", str(exc))
            return
        self._open_image_viewer(
            src,
            f"{source_path.resolve()}#obf-images",
            initial_series_index=int(src.metadata.series_index),
        )
        self._state.log(
            f"Image series of '{source_path.name}': {len(series)} series, showing "
            f"'{wanted.get('name')}'.", dataset_idx=idx)

    # ------------------------------------------------------------------
    # Files dropped ONTO a dataset (Dataset Manager row drop)
    # ------------------------------------------------------------------

    #: Extensions a Dataset-Manager row accepts, and what each does to that
    #: dataset.  A ``.json`` is ambiguous on extension alone, so it is resolved
    #: by content (ROI set / filter preset / metadata sidecar).
    DROP_ON_DATASET_EXTS = _formats.drop_on_dataset_extensions()

    def drop_file_on_dataset(self, idx: int, path: str) -> bool:
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
        if ext in (".roi", ".zip"):
            return self._drop_roi_file(idx, p)
        if ext == ".json":
            # Resolve by content — the three JSON kinds share an extension.
            from ..core.filter_io import is_filter_json_file
            from ..core.roi import is_roi_json_file
            from ..core.save import is_metadata_json_file

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
            self._state.log(
                f"Drop on '{ds.name}': '{p.name}' is not a ROI set, filter preset "
                "or metadata sidecar.", "WARN")
            return False

        self._state.log(
            f"Drop on '{ds.name}': '{p.name}' cannot be applied to a dataset "
            "(drop it on the main window to open it).", "WARN")
        return False

    def _drop_filter_file(self, idx: int, path: Path) -> bool:
        """Append a dropped filter preset to the row's dataset."""
        # The filter dialog is dataset-owned and keyed by the ACTIVE index, so
        # the dropped-on dataset has to become active for its rows to land there.
        self._state.set_active(idx)
        self._load_filter_json(str(path))
        self._state.log(
            f"Filters from '{path.name}' added to "
            f"'{self._state.datasets[idx].name}'.", dataset_idx=idx)
        return True

    def _drop_roi_file(self, idx: int, path: Path) -> bool:
        """Load a dropped ROI set into the ROI Manager, targeting the row's dataset."""
        self._state.set_active(idx)       # _load_roi_json retargets to the active one
        self._load_roi_json(str(path))
        return True

    def _drop_metadata_sidecar(self, idx: int, path: Path) -> bool:
        """Apply a dropped processing-recipe sidecar to the row's dataset."""
        from ..core.loader import apply_metadata_sidecar_file

        ds = self._state.datasets[idx]
        try:
            applied = apply_metadata_sidecar_file(ds, path)
        except Exception as exc:
            self._state.log(f"Metadata sidecar '{path.name}' on '{ds.name}': {exc}",
                            "ERROR")
            QMessageBox.critical(self, "Apply processing metadata", str(exc))
            return False
        if not applied:
            self._state.log(
                f"Metadata sidecar '{path.name}' on '{ds.name}': nothing to apply "
                "(empty recipe).", "WARN")
            return True
        self._state.log(
            f"Applied '{path.name}' to '{ds.name}': " + ", ".join(applied) + ".",
            dataset_idx=idx)
        self._state.notify_filter_changed(idx)
        self._state.notify_calibration_changed(idx)
        self._refresh_overlay_windows()
        return True

    def _drop_confocal_tiff(self, idx: int, path: Path) -> bool:
        """Map a dropped TIFF onto the row's dataset as a confocal signal attribute."""
        from ..core.confocal_mapping import candidates_from_tiff
        from .confocal_mapping_dialog import (
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

    def _close_all_windows(self) -> None:
        """``Ctrl+Shift+W`` — close **everything**: all datasets, their viewer
        windows, and every plugin / analysis / tool dialog (particle average, MSR
        reader, ROI manager, dataset manager, …). Only the **Log** and **Console**
        windows are kept."""
        self._close_all_datasets()
        self._close_all_child_windows(keep_log_console=True)
        self.setWindowTitle(self.APP_NAME)
        self._status_label.setText("Closed all windows (kept Log / Console).")

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
        # 2) a freshly drawn draft on this dataset's render/scatter overlay.
        for win in (
            self._render_windows.get(ds_idx),
            self._scatter_windows.get(ds_idx),
        ):
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

    def _show_channel_combine(self, dataset_indices=None) -> None:
        # QAction.triggered passes checked=False; only a real sequence restricts
        # the dialog to a subset (the Dataset Manager's multi-selection).
        restrict = (
            list(dataset_indices)
            if isinstance(dataset_indices, (list, tuple, set))
            else None
        )
        if restrict is not None and len(restrict) < 2:
            QMessageBox.information(self, "Combine datasets", "Select at least two datasets to combine.")
            return
        if len(self._state.datasets) < 2:
            QMessageBox.information(self, "Combine datasets", "Load at least two datasets before combining channels.")
            return
        from .channel_combine_dialog import ChannelCombineDialog
        from ..core.overlay import OverlayMemberSpec, build_overlay_transforms
        import uuid

        dlg = ChannelCombineDialog(
            self._state,
            previous=self._last_channel_combine_settings,
            dataset_indices=restrict,
            parent=self,
        )
        result = dlg.exec()
        # A restricted run sees only part of the list, so its per-dataset
        # order/LUT choices would be a partial memory — don't overwrite the
        # settings the full dialog remembers.
        if restrict is None:
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
        had_scatter = self._collapse_member_coordinate_views(
            [spec.dataset_idx for spec in specs], anchor_idx)
        self._state.set_active(anchor_idx)
        self._show_render(anchor_idx)
        if had_scatter:
            self._show_scatter(anchor_idx)
        # Any other open overlay view (e.g. one keyed by a dataset that was
        # already a channel) re-reads its channel list.
        self._refresh_overlay_windows()
        self._notify_view_state_changed()
        self._state.log(f"Created overlay {overlay_index} with {len(specs)} dataset(s).")

    def _collapse_member_coordinate_views(self, member_indices, anchor_idx: int) -> bool:
        """Fold the members' standalone render/scatter windows into the anchor's.

        Render and scatter are **overlay-aware**: one window, keyed by the anchor,
        draws every channel. So once datasets are combined in place (*keep source
        dataset* unchecked) the other members' own windows are stale duplicates —
        each still titled and drawn as a standalone dataset, i.e. exactly the
        "Own" views the Dataset Manager no longer lists. They are closed here;
        nothing folds them by itself, which is why one was always left behind.

        Only the coordinate views are folded. A histogram, attribute plot or data
        window of a single channel stays meaningful and is left open, and **no
        dataset is closed** — combining changes how they are viewed, not what
        exists. Returns whether any member had a scatter window, so the caller
        can reopen it on the anchor rather than leaving the user with none.
        """
        wanted_scatter = anchor_idx in self._scatter_windows
        for idx in member_indices:
            if idx == anchor_idx:
                continue
            for registry in (self._render_windows, self._scatter_windows):
                win = registry.pop(idx, None)
                if win is None:
                    continue
                if registry is self._scatter_windows:
                    wanted_scatter = True
                try:
                    win.close()
                except Exception:
                    pass
        return wanted_scatter

    def _show_channel_separation(self) -> None:
        """Process › Channel… › Separate Channel by DCR — open the DCR two-color
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
        from .attribute_separation_dialog import AttributeSeparationDialog
        from .modeless import show_modeless
        win = AttributeSeparationDialog(
            self._state, idx, attribute="dcr",
            title="Separate Channel by DCR", allow_photon_weight=True, owner=self)
        show_modeless(win, self)

    def _show_attribute_separation(self) -> None:
        """Process › Channel › Convert Dataset to Multi-Channel Overlay › by MINFLUX
        data attribute — the generic attribute-based channel separation. Same
        dialog as by-DCR, but with an attribute picker (DCR is one instance)."""
        idx = self._state.active_idx
        if idx is None or not (0 <= idx < len(self._state.datasets)):
            self._no_data_warning()
            return
        ds = self._state.datasets[idx]
        from ..core.loader import attr_values_1d
        default = "dcr" if attr_values_1d(ds, "dcr") is not None else "efo"
        from .attribute_separation_dialog import AttributeSeparationDialog
        from .modeless import show_modeless
        win = AttributeSeparationDialog(
            self._state, idx, attribute=default,
            title="Convert to Multi-Channel Overlay (by attribute)",
            allow_photon_weight=True, pick_attribute=True, owner=self)
        show_modeless(win, self)

    def _show_time_channel_separation(self) -> None:
        """Open the time-window editor for the active dataset."""
        import numpy as np

        from ..core.loader import attr_values_1d
        from ..core.overlay import overlay_color_cycle
        from .time_channel_dialog import TimeChannelDialog

        idx = self._state.active_idx
        if idx is None or not (0 <= idx < len(self._state.datasets)):
            self._no_data_warning()
            return
        source = self._state.datasets[idx]
        tim = attr_values_1d(source, "tim")
        if tim is None:
            QMessageBox.information(
                self,
                "Separate Channels from Time Windows",
                "The active dataset has no tim attribute.",
            )
            return
        tim = np.asarray(tim, dtype=float).ravel()
        base_mask = np.asarray(source.filter_mask, dtype=bool).ravel()
        if tim.size != int(source.prop.num_loc) or base_mask.size != tim.size:
            QMessageBox.warning(
                self,
                "Separate Channels from Time Windows",
                "The tim attribute does not align with the active localization table.",
            )
            return
        active_tim = tim[base_mask & np.isfinite(tim)]
        if active_tim.size < 2 or float(np.max(active_tim)) <= float(np.min(active_tim)):
            QMessageBox.information(
                self,
                "Separate Channels from Time Windows",
                "The active localizations do not span a usable acquisition-time range.",
            )
            return

        try:
            dialog = TimeChannelDialog(
                tim,
                source_name=source.name,
                base_mask=base_mask,
                color_cycle=overlay_color_cycle(self._state.prefs),
                parent=self,
            )
        except ValueError as exc:
            QMessageBox.warning(self, "Separate Channels from Time Windows", str(exc))
            return
        if dialog.exec() != QDialog.DialogCode.Accepted:
            return
        self.apply_time_channel_separation(idx, dialog.windows())

    def apply_time_channel_separation(self, source_idx: int, windows) -> bool:
        """Create filtered full-data clones and group them as a channel overlay."""
        import uuid

        import numpy as np

        from ..core.loader import attr_values_1d
        from ..core.overlay import display_transform_record, identity_matrix4
        from ..core.time_channels import (
            clone_time_channel_dataset,
            time_channel_selections,
        )

        if not (0 <= source_idx < len(self._state.datasets)):
            return False
        source = self._state.datasets[source_idx]
        tim = attr_values_1d(source, "tim")
        if tim is None:
            return False
        try:
            selections = time_channel_selections(
                tim,
                windows,
                base_mask=source.filter_mask,
            )
        except ValueError as exc:
            self._state.log(f"Time-window channel separation failed: {exc}", "ERROR")
            return False
        if any(not np.any(selection.mask) for selection in selections):
            self._state.log(
                "Time-window channel separation failed: at least one channel is empty.",
                "ERROR",
            )
            return False

        reserved = {dataset.name.casefold() for dataset in self._state.datasets}

        def unique_name(requested: str) -> str:
            base = requested.strip()
            candidate = base
            number = 2
            while candidate.casefold() in reserved:
                candidate = f"{base} ({number})"
                number += 1
            reserved.add(candidate.casefold())
            return candidate

        timestamp = datetime.now().strftime("%Y-%b-%d, %H:%M:%S")
        try:
            new_datasets = [
                clone_time_channel_dataset(
                    source,
                    selection,
                    name=unique_name(selection.window.name),
                    timestamp=timestamp,
                )
                for selection in selections
            ]
        except Exception as exc:
            self._state.log(f"Time-window channel separation failed: {exc}", "ERROR")
            QMessageBox.critical(
                self, "Separate Channels from Time Windows", str(exc)
            )
            return False

        overlay_index = self._next_overlay_index
        self._next_overlay_index += 1
        overlay_id = f"overlay:{overlay_index}:{uuid.uuid4().hex}"
        new_indices: list[int] = []
        previous = getattr(self._state, "suspend_auto_render", False)
        self._state.suspend_auto_render = True
        try:
            for order, (selection, dataset) in enumerate(
                zip(selections, new_datasets), start=1
            ):
                idx = self._state.add_dataset(dataset)
                channel = self._state.datasets[idx]
                lut = selection.window.lut
                transform = display_transform_record(
                    overlay_id=overlay_id,
                    overlay_index=overlay_index,
                    order=order,
                    lut=lut,
                    source_dataset_idx=idx,
                    alignment_mode="stage origin",
                    matrix_4x4=identity_matrix4(),
                    provenance={
                        "method": "time-window channel separation",
                        "source_dataset": source.name,
                    },
                )
                channel.state["overlay_id"] = overlay_id
                channel.state["render_group_id"] = overlay_id
                channel.state["overlay_index"] = overlay_index
                channel.state["overlay_order"] = order
                channel.state["overlay_lut"] = lut
                channel.state["render_channel_lut"] = lut
                channel.state["overlay_transform"] = transform
                channel.state["render_transform_2d"] = transform
                channel.metadata["overlay_id"] = overlay_id
                channel.metadata["overlay_index"] = overlay_index
                channel.metadata["overlay_alignment_mode"] = "stage origin"
                channel.metadata["separated_from"] = source.name
                channel.metadata["separated_by"] = "tim"
                new_indices.append(idx)
        finally:
            self._state.suspend_auto_render = previous

        if not new_indices:
            return False
        # Breadcrumb so "Revert Overlay to Original Dataset" can find this source.
        source.metadata.setdefault("produced_overlays", []).append(overlay_id)
        anchor = new_indices[0]
        self._state.set_active(anchor)
        self._show_render(anchor)
        self._notify_view_state_changed()
        summary = ", ".join(
            f"{selection.window.name}: {int(selection.mask.sum()):,}"
            for selection in selections
        )
        self._state.log(
            f"Separated '{source.name}' into {len(new_indices)} time-window "
            f"channels (overlay {overlay_index}): {summary}."
        )
        try:
            self._state.journal.add(
                "transform",
                f"Separated dataset '{source.name}' into {len(new_indices)} "
                "channels using acquisition-time windows.",
                windows=[
                    {
                        "name": selection.window.name,
                        "start_s": float(selection.window.start_s),
                        "end_s": float(selection.window.end_s),
                    }
                    for selection in selections
                ],
            )
        except Exception:
            pass
        return True

    def apply_channel_separation(self, src_idx: int, labels, channels, *,
                                 attribute: str = "", method_label: str = "channel separation") -> bool:
        """Build one truncated dataset per channel (+ a hidden *unassigned*) from
        per-localization channel *labels* (0..N-1, or -1 = unassigned) and combine
        them as a render overlay. *channels* is a list of carriers exposing
        ``.name`` and ``.lut`` (see :class:`analysis.attribute_channels.Channel`).
        Attribute-agnostic — DCR is one instance. Returns True on success."""
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
        # (mask, LUT, dataset name, hidden-by-default), one per channel + unassigned.
        plan = [
            (labels == k, getattr(ch, "lut", "Gray") or "Gray",
             getattr(ch, "name", None) or f"{base_name} [ch {k + 1}]", False)
            for k, ch in enumerate(channels)
        ]
        plan.append((labels == -1, "Gray", f"{base_name} [unassigned]", True))

        overlay_index = self._next_overlay_index
        self._next_overlay_index += 1
        overlay_id = f"overlay:{overlay_index}:{uuid.uuid4().hex}"
        prefs = self._state.prefs

        new_indices: list[int] = []
        order = 0
        previous = getattr(self._state, "suspend_auto_render", False)
        self._state.suspend_auto_render = True
        try:
            for mask, lut, name, hidden in plan:
                keep = np.asarray(mask, dtype=bool)
                if not keep.any():
                    continue
                order += 1
                new_ds = subset_dataset(src, keep, name=name, prefs=prefs)
                idx = self._state.add_dataset(new_ds)
                ds = self._state.datasets[idx]
                transform = display_transform_record(
                    overlay_id=overlay_id, overlay_index=overlay_index, order=order,
                    lut=lut, source_dataset_idx=idx, alignment_mode="stage origin",
                    matrix_4x4=identity_matrix4(),
                    provenance={"method": method_label, "source_dataset": base_name,
                                "attribute": attribute})
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
                ds.metadata["separated_from"] = base_name
                ds.metadata["separated_by"] = attribute
                new_indices.append(idx)
        finally:
            self._state.suspend_auto_render = previous

        if not new_indices:
            QMessageBox.warning(self, "Separate Channels",
                                "No localizations were assigned to a channel.")
            return False
        # Breadcrumb so "Revert Overlay to Original Dataset" can find this source.
        src.metadata.setdefault("produced_overlays", []).append(overlay_id)
        anchor = new_indices[0]
        self._state.set_active(anchor)
        self._show_render(anchor)
        self._notify_view_state_changed()
        by = attribute or method_label
        self._state.log(
            f"Separated '{base_name}' into {len(new_indices)} channel(s) by {by} (overlay {overlay_index}).")
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

        # An overlay render window is keyed only by its anchor index but displays
        # every channel; reuse it (raise, don't spawn a duplicate) when idx is one
        # of its non-anchor channels — e.g. segmentation adding ROIs on channel 2.
        win = self._render_windows.get(idx) or self._render_window_for_dataset(idx)
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

        from .precision_render_window import PrecisionRenderWindow
        win = PrecisionRenderWindow(self._state, dataset_idx=idx)
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

    def _save_data(self, *args) -> None:
        """File › Save — save/export the **active** dataset."""
        ds = self._state.active_dataset
        if ds is None:
            QMessageBox.information(
                self, "Save", "No active dataset to save."
            )
            return
        self.save_dataset(ds)

    def _active_dataset_for_save(self):
        ds = self._state.active_dataset
        if ds is None:
            QMessageBox.information(self, "Save", "No active dataset to save.")
            return None
        return ds

    def _default_save_path(self, ds, suffix: str) -> Path:
        default_dir = Path(self._state.prefs["file"].get("default_folder", str(Path.home())))
        folder = Path(str(getattr(getattr(ds, "file", None), "folder", "") or default_dir))
        if not folder.exists():
            folder = default_dir
        stem = Path(str(getattr(ds, "name", "") or "dataset")).stem or "dataset"
        return folder / f"{stem}{suffix}"

    def _save_as_minflux_formats(self) -> None:
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

    def _zarr_overwrite_mode(self, path: str | Path) -> str | None:
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

    def _save_as_format(self, fmt: str, title: str, *, path: str | None = None) -> None:
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

        zarr_context = (self._zarr_save_context(ds)
                        if fmt in {"zarr", "zarr_zip"} else {})
        # Only the directory store can be updated in place; a sealed package is
        # always rewritten, and QFileDialog already confirmed the overwrite.
        zarr_overwrite = self._zarr_overwrite_mode(path) if fmt == "zarr" else "replace"
        if zarr_overwrite is None:
            return
        kwargs = dict(
            data_path=path, fmt=fmt, content="raw",
            include={"attrs": True, "derived": False, "recipe": True},
            filter_mode="flag", zarr_overwrite=zarr_overwrite, **zarr_context,
        )
        action = "Updated processing in " if zarr_overwrite == "viewer" else "Saved "
        if fmt in {"zarr", "zarr_zip"}:
            # Writing a store is CPU-bound (compress + hash), ~15 s on a large
            # acquisition. Off the UI thread so the window stays usable; the
            # dataset is only read here, so nothing touches widgets.
            name = Path(path).name

            def work(report, _ds=ds, _kwargs=kwargs, _name=name):
                report(f"Writing {_name}")
                written = save_processed(_ds, **_kwargs)
                report(f"Wrote {_name}")
                return written

            task = _ZarrIoTask(work, description=f"Saving {name}")
            task.signals.stage.connect(lambda text: self._state.status_progress(text))
            task.signals.done.connect(
                lambda written, _a=action, _n=name: self._on_zarr_saved(written, _a, _n))
            task.signals.failed.connect(
                lambda message, _n=name: self._on_zarr_io_failed(
                    _n, message, "Save failed"))
            self._begin_zarr_io(task)
            return
        try:
            written = save_processed(ds, **kwargs)
        except Exception as exc:
            QMessageBox.critical(
                self, "Save failed", f"Could not save {title}:\n{exc}")
            return
        self._state.log(
            action + ", ".join(str(p) for p in written),
            dataset_idx=self._state.active_idx,
        )

    def _on_zarr_saved(self, written, action: str, name: str) -> None:
        if self._is_shutting_down:
            return
        self._state.log(
            action + ", ".join(str(path) for path in written),
            dataset_idx=self._state.active_idx,
        )
        self._status_label.setText(f"{action.strip()} {name}.")

    def _zarr_save_context(self, ds) -> dict:
        """Overlay members, portable ROI geometry and linked MSR images."""
        from dataclasses import asdict

        from ..core.overlay import overlay_members
        from ..core.roi_selection import ROI_MASKS_STATE_KEY

        try:
            anchor_idx = next(i for i, item in enumerate(self._state.datasets) if item is ds)
        except StopIteration:
            return {}
        pairs = overlay_members(self._state, anchor_idx)
        if len(pairs) < 2:
            pairs = [(anchor_idx, ds)]
        member_indices = {idx for idx, _member in pairs}
        index_to_id = {idx: f"d{position:06d}" for position, (idx, _ds) in enumerate(pairs)}
        mask_owner = {
            roi_id: idx
            for idx, member in pairs
            for roi_id in dict(member.state.get(ROI_MASKS_STATE_KEY) or {})
        }

        candidates = list(self._state.rois.records)
        adapter = self._state.rois.active_adapter
        if adapter is not None:
            try:
                draft = adapter.current_record()
            except Exception:
                draft = None
            if draft is not None and all(record.id != draft.id for record in candidates):
                candidates.append(draft)
        roi_records = []
        for record in candidates:
            context = dict(getattr(record, "context", {}) or {})
            owner = context.get("dataset_idx")
            if owner not in member_indices:
                owner = mask_owner.get(record.id)
            if owner not in member_indices:
                continue
            payload = asdict(record)
            payload["dataset_id"] = index_to_id[int(owner)]
            payload["context"] = {
                key: value for key, value in context.items() if key != "dataset_idx"
            }
            roi_records.append(payload)

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

    def _save_as_spreadsheet(self) -> None:
        ds = self._active_dataset_for_save()
        if ds is None:
            return
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
            self._default_save_path(ds, ".csv"),
            self,
        )
        if dialog.exec() != QDialog.DialogCode.Accepted:
            return
        try:
            path = write_spreadsheet_csv(
                ds,
                dialog.path_edit.text().strip(),
                column_headers=dialog.selected_columns(),
                separator=dialog.separator_edit.text(),
            )
        except Exception as exc:
            QMessageBox.critical(
                self, "Spreadsheet export failed", f"Could not save spreadsheet:\n{exc}"
            )
            return
        self._state.log(
            f"Saved spreadsheet: {path}",
            dataset_idx=self._state.active_idx,
        )

    def _save_as_picasso_hdf5(self) -> None:
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

        try:
            written = write_picasso_hdf5(ds, path, pixel_size_nm=pixel_spin.value())
        except Exception as exc:
            QMessageBox.critical(self, "Save failed", f"Could not save Picasso HDF5:\n{exc}")
            return
        self._state.log(
            "Saved Picasso HDF5: " + ", ".join(str(p) for p in written),
            dataset_idx=self._state.active_idx,
        )

    def _save_as_ome_tiff(self) -> None:
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

    def _save_as_ome_zarr(self) -> None:
        ds = self._active_dataset_for_save()
        if ds is None:
            return
        idx = self._state.active_idx
        from ..core.ome_zarr import (
            estimate_ome_zarr_export,
            normalize_ome_zarr_path,
            write_ome_zarr,
        )
        from .export_dialogs import OmeZarrExportDialog

        pixel_size_nm = float(getattr(ds.cali, "pixel_size", 4.0) or 4.0)
        is_3d = int(getattr(ds.prop, "num_dim", 2)) >= 3
        loc_precision = getattr(ds.cali, "loc_precision", None)
        try:
            z_default = float(loc_precision[2])
        except (TypeError, IndexError, ValueError):
            z_default = 0.0
        if not z_default > 0.0:
            z_default = max(10.0, pixel_size_nm * 2.0)
        dialog = OmeZarrExportDialog(
            self._default_save_path(ds, ".ome.zarr"),
            pixel_size_nm=pixel_size_nm,
            z_voxel_nm=z_default,
            is_3d=is_3d,
            parent=self,
        )
        if dialog.exec() != QDialog.DialogCode.Accepted:
            return
        output = normalize_ome_zarr_path(dialog.path_edit.text().strip())
        pixel_size_nm = dialog.pixel_size_spin.value()
        z_voxel_nm = dialog.z_voxel_spin.value() if is_3d else None
        max_levels = dialog.levels_spin.value()
        overwrite = False
        if output.exists():
            answer = QMessageBox.question(
                self,
                "Replace OME-Zarr package?",
                f"{output}\nalready exists. Replace the complete package?",
                QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No,
                QMessageBox.StandardButton.No,
            )
            if answer != QMessageBox.StandardButton.Yes:
                return
            overwrite = True

        self._state.status_progress(
            f"Estimating OME-Zarr export of '{ds.name}'", 0.0
        )
        QApplication.processEvents()
        QApplication.setOverrideCursor(Qt.CursorShape.WaitCursor)
        preflight_error = None
        try:
            estimate = estimate_ome_zarr_export(
                ds,
                output,
                pixel_size_nm=pixel_size_nm,
                z_voxel_nm=z_voxel_nm,
                max_levels=max_levels,
            )
        except Exception as exc:
            preflight_error = exc
        finally:
            QApplication.restoreOverrideCursor()
        if preflight_error is not None:
            self._state.status_message.emit("OME-Zarr preflight: failed.")
            QMessageBox.critical(
                self,
                "OME-Zarr preflight failed",
                f"Could not estimate OME-Zarr resources:\n{preflight_error}",
            )
            return

        estimate_text = self._ome_zarr_estimate_text(estimate)
        self._state.log(
            f"OME-Zarr preflight for '{ds.name}': "
            f"shape {' x '.join(str(value) for value in estimate.image_shape)}, "
            f"estimated {_human_bytes(estimate.estimated_output_bytes)}, "
            f"working RAM {_human_bytes(estimate.peak_working_ram_bytes)}, "
            f"time {_human_duration(estimate.estimated_seconds)}.",
            dataset_idx=idx,
        )
        if estimate.blockers:
            QMessageBox.critical(
                self,
                "OME-Zarr export exceeds system capacity",
                estimate_text
                + "\n\nExport cannot start:\n"
                + "\n".join(f"- {reason}" for reason in estimate.blockers),
            )
            return
        if estimate.warnings:
            answer = QMessageBox.warning(
                self,
                "Large OME-Zarr export",
                estimate_text
                + "\n\n"
                + "\n".join(f"- {warning}" for warning in estimate.warnings)
                + "\n\nContinue with the export?",
                QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No,
                QMessageBox.StandardButton.No,
            )
            if answer != QMessageBox.StandardButton.Yes:
                return

        roi_records = []
        for record in self._state.rois.records:
            context = getattr(record, "context", {}) or {}
            record_idx = context.get("dataset_idx")
            if record_idx is not None:
                if idx is not None and int(record_idx) == int(idx):
                    roi_records.append(record)
                continue
            target = str(getattr(record, "target_hint", "") or "")
            if len(self._state.datasets) == 1 or (target and ds.name in target):
                roi_records.append(record)

        journal_entries = []
        source_path = str(getattr(getattr(ds, "file", None), "path", "") or "")
        for entry in self._state.journal.entries:
            if len(self._state.datasets) == 1:
                journal_entries.append(entry)
                continue
            details = getattr(entry, "details", {}) or {}
            haystack = f"{getattr(entry, 'summary', '')} {details}"
            if ds.name in haystack or (source_path and source_path in haystack):
                journal_entries.append(entry)

        def export(progress):
            return write_ome_zarr(
                ds,
                output,
                pixel_size_nm=pixel_size_nm,
                z_voxel_nm=z_voxel_nm,
                max_levels=max_levels,
                dataset_idx=idx,
                roi_records=roi_records,
                journal_entries=journal_entries,
                overwrite=overwrite,
                progress=progress,
            )

        from ..core.app_state import format_progress_bar

        task = _OmeZarrTask(export)
        task.signals.progress.connect(
            lambda fraction, stage, t=task: self._on_ome_zarr_progress(
                fraction, stage, t
            )
        )
        task.signals.done.connect(
            lambda result, t=task, dataset=ds: self._on_ome_zarr_done(
                result, t, dataset
            )
        )
        task.signals.failed.connect(
            lambda message, t=task, dataset=ds: self._on_ome_zarr_failed(
                message, t, dataset
            )
        )
        task.signals.cancelled.connect(
            lambda t=task: self._on_ome_zarr_cancelled(t)
        )
        self._ome_zarr_tasks.add(task)
        self._state.log(
            f"OME-Zarr export started for '{ds.name}' -> {output}",
            dataset_idx=idx,
        )
        self._state.log_progress(format_progress_bar(0.0))
        self._state.status_progress(f"OME-Zarr export of '{ds.name}'", 0.0)
        QThreadPool.globalInstance().start(task)

    @staticmethod
    def _ome_zarr_estimate_text(estimate) -> str:
        axes = "Z x Y x X" if estimate.is_3d else "Y x X"
        voxel = " x ".join(f"{value:g}" for value in estimate.voxel_size_nm)
        return (
            f"Level-0 shape ({axes}): "
            f"{' x '.join(str(value) for value in estimate.image_shape)}\n"
            f"Voxel/pixel size: {voxel} nm\n"
            f"Pyramid levels: {len(estimate.level_shapes)}\n"
            f"Filtered localizations: {estimate.filtered_localizations:,}\n"
            f"Estimated compressed package: "
            f"{_human_bytes(estimate.estimated_output_bytes)}\n"
            f"Conservative upper bound: {_human_bytes(estimate.upper_output_bytes)}\n"
            f"Exporter working RAM: {_human_bytes(estimate.peak_working_ram_bytes)} "
            f"of {_human_bytes(estimate.available_ram_bytes)} available\n"
            f"Dense level-0 stack in a reader: "
            f"{_human_bytes(estimate.dense_level_zero_bytes)}\n"
            f"Free target-disk space: {_human_bytes(estimate.free_disk_bytes)}\n"
            f"Estimated conversion time: {_human_duration(estimate.estimated_seconds)}"
        )

    def _on_ome_zarr_progress(self, fraction: float, stage: str, task) -> None:
        if task not in self._ome_zarr_tasks or self._is_shutting_down:
            return
        from ..core.app_state import format_progress_bar

        fraction = max(0.0, min(1.0, float(fraction)))
        percent = int(round(fraction * 100.0))
        if percent != task.last_percent:
            task.last_percent = percent
            self._state.log_progress(format_progress_bar(fraction))
        task.last_stage = str(stage)
        self._state.status_progress(f"OME-Zarr: {stage}", fraction)

    def _on_ome_zarr_done(self, result, task, ds) -> None:
        self._ome_zarr_tasks.discard(task)
        if self._is_shutting_down:
            return
        from ..core.app_state import format_progress_bar

        self._state.log_progress(format_progress_bar(1.0, done=True), final=True)
        self._state.status_message.emit("OME-Zarr export: done.")
        idx = self._post_load_index(ds)
        details = (
            f"Saved OME-NGFF 0.5 / Zarr v3: {result.path} "
            f"({result.levels} pyramid level(s), "
            f"shape {' x '.join(str(value) for value in result.image_shape)}, "
            f"voxel/pixel size "
            f"{' x '.join(f'{value:g}' for value in result.voxel_size_nm)} nm)"
        )
        self._state.log(details, dataset_idx=idx)
        for warning in result.warnings:
            self._state.log(f"OME-Zarr export warning: {warning}", dataset_idx=idx)

    def _on_ome_zarr_failed(self, message: str, task, ds) -> None:
        self._ome_zarr_tasks.discard(task)
        if self._is_shutting_down:
            return
        self._state.log_progress("=" * 10 + "  FAILED  " + "=" * 10, final=True)
        self._state.status_message.emit("OME-Zarr export: failed.")
        self._state.log(
            f"OME-Zarr export failed: {message}",
            level="ERROR",
            dataset_idx=self._post_load_index(ds),
        )
        QMessageBox.critical(
            self,
            "OME-Zarr export failed",
            f"Could not save OME-NGFF 0.5 / Zarr v3:\n{message}",
        )

    def _on_ome_zarr_cancelled(self, task) -> None:
        self._ome_zarr_tasks.discard(task)
        if self._is_shutting_down:
            return
        self._state.log_progress("=" * 10 + "  CANCELLED  " + "=" * 10, final=True)
        self._state.status_message.emit("OME-Zarr export: cancelled.")

    def save_dataset(self, ds) -> None:
        """Open the Save / Export dialog for *ds* (the active dataset from File ›
        Save Processed Data, or a right-clicked one from the Dataset Manager)."""
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
            prefs=self._state.prefs,
            parent=self,
        )
        if dlg.exec() != QDialog.DialogCode.Accepted:
            return
        opts = dlg.options()
        from ..core.save import save_processed

        # Overlay members, ROI geometry and linked images travel with BOTH Zarr
        # forms; only the directory store can be updated in place.
        zarr_context = (self._zarr_save_context(ds)
                        if opts.get("fmt") in {"zarr", "zarr_zip"} else {})
        zarr_overwrite = (
            self._zarr_overwrite_mode(opts["data_path"])
            if opts.get("fmt") == "zarr"
            else "replace"
        )
        if zarr_overwrite is None:
            return
        try:
            written = save_processed(
                ds,
                data_path=opts["data_path"],
                fmt=opts["fmt"],
                metadata_dir=src.parent if file_backed else None,
                content=opts.get("content", "raw"),
                include=opts.get("include"),
                filter_mode=opts.get("filter_mode", "flag"),
                zarr_overwrite=zarr_overwrite,
                **zarr_context,
            )
        except Exception as exc:
            QMessageBox.critical(
                self, "Save failed", f"Could not save processed data:\n{exc}"
            )
            return
        action = (
            "Updated processing in" if zarr_overwrite == "viewer"
            else "Saved processed data"
        )
        self._state.log(f"{action}: {', '.join(p.name for p in written)}", "INFO")
        QMessageBox.information(
            self, "Save processed data",
            f"{action}:\n" + "\n".join(str(p) for p in written),
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
        if data_prefs.get("show_dataset_manager", False):
            self._ensure_dataset_manager_visible()

        # Open the render view first (when requested, unless a batch importer will
        # open a grouped render view) so the data-info window can be placed *beside*
        # it rather than on top of it.
        if data_prefs.get("show_render", True) and not getattr(self._state, "suspend_auto_render", False):
            self._show_render(idx)

        if data_prefs.get("show_data_info", True):
            from .modeless import place_beside
            # Anchor to this dataset's render view if it opened, else the main
            # window; try right → below → left → above, staying on-screen.
            anchor = self._render_windows.get(idx) or self
            win.adjustSize()                              # realise its fixed size
            place_beside(win, anchor, prefer=("right", "below", "left", "above"))
            win.show()

        if data_prefs.get("show_attr_plot", False):
            self._show_attr_plot(idx)
        if data_prefs.get("show_histogram", False):
            self._show_histogram(idx)
        if data_prefs.get("show_scatter", False):
            self._show_scatter(idx)

        self._schedule_post_load_computations(idx)

    def _schedule_post_load_computations(self, idx: int) -> None:
        """Run optional one-time computed attributes after initial windows show."""
        delay_ms = 450 + max(0, idx) * 100
        QTimer.singleShot(delay_ms, lambda i=idx: self._run_post_load_computations(i))

    def _run_post_load_computations(self, idx: int) -> None:
        """Kick off the one-time computed attributes (Z scaling factor, localization
        precision, local density) as a chain of single-shot steps.

        Each step returns to the event loop before the next runs, so the Log
        updates live and the UI stays responsive — the heavy estimators are now
        vectorised, so every step is sub-second instead of the old ~20 s block.
        Steps re-resolve the dataset by identity, so closing it mid-chain (or
        removing another dataset, which shifts indices) aborts/retargets safely.
        """
        if not (0 <= idx < len(self._state.datasets)):
            return
        ds = self._state.datasets[idx]
        data_prefs = self._state.prefs.get("data", {})
        plot_prefs = self._state.prefs.get("plot", {})
        z_scaling_factor_requested = (
            data_prefs.get("estimate_z_scaling_factor", False)
            or plot_prefs.get("use_fixed_z_scaling_factor", False)
        )
        needs = (
            (z_scaling_factor_requested and "z_scaling_factor" not in ds.derived)
            or (data_prefs.get("compute_loc_prec", True) and "sigma_per_trace_nm" not in ds.derived)
            or (data_prefs.get("compute_local_density", True) and "den" not in ds.attr)
            or bool(ds.state.get("filter_specs"))
        )
        if needs:
            self._state.log(f"Post-load processing of '{ds.name}' (Z scaling factor, precision, density)…")
            self._state.status_progress(f"Processing '{ds.name}'")
        # Schedule (don't call) the first step so the line above paints first.
        self._post_load_next(ds, self._post_load_z_scaling_factor)

    def _post_load_index(self, ds) -> "int | None":
        """Current index of *ds* by identity, or None if it was removed."""
        for i, d in enumerate(self._state.datasets):
            if d is ds:
                return i
        return None

    def _post_load_next(self, ds, step) -> None:
        """Run the next post-load step from the event loop (keeps the UI live)."""
        QTimer.singleShot(0, lambda: step(ds))

    def _post_load_z_scaling_factor(self, ds) -> None:
        idx = self._post_load_index(ds)
        if idx is None:
            return
        data_prefs = self._state.prefs.get("data", {})
        plot_prefs = self._state.prefs.get("plot", {})
        estimate_z_scaling_factor = data_prefs.get("estimate_z_scaling_factor", False)
        use_fixed_z_scaling_factor = plot_prefs.get("use_fixed_z_scaling_factor", False)
        if (estimate_z_scaling_factor or use_fixed_z_scaling_factor) and "z_scaling_factor" not in ds.derived:
            import numpy as np
            if ds.prop.num_dim < 3:
                # 2D data: Z is all zero, so anisotropy estimation is moot.
                ds.set_z_scaling_factor(1.0, source="2D (no z correction)")
                ds.derived["z_scaling_factor"] = np.asarray([1.0], dtype=float)
                self._state.log(f"Z scaling factor for '{ds.name}': 1.0 (2D dataset, computation skipped).")
                self._state.notify_calibration_changed(idx)
            elif use_fixed_z_scaling_factor:
                fixed = float(plot_prefs.get("z_scaling_factor", 0.67))
                ds.set_z_scaling_factor(fixed, source="fixed (preference)")
                ds.derived["z_scaling_factor"] = np.asarray([fixed], dtype=float)
                self._state.log(f"Z scaling factor for '{ds.name}': {fixed:.4g} (fixed preference value).")
                self._state.notify_calibration_changed(idx)
            elif estimate_z_scaling_factor:
                # Heavy estimate: announce now and run it on the next event-loop
                # turn so this line is painted before the (sub-second) compute.
                self._state.log(
                    f"Estimating Z scaling factor from trace anisotropy for '{ds.name}'…"
                )
                self._state.status_progress(
                    f"Estimating Z scaling factor for '{ds.name}'"
                )
                self._post_load_next(ds, self._post_load_z_scaling_factor_estimate)
                return
        self._post_load_next(ds, self._post_load_loc_prec)

    def _post_load_z_scaling_factor_estimate(self, ds) -> None:
        idx = self._post_load_index(ds)
        if idx is None:
            return
        import time

        import numpy as np
        value = None
        try:
            t0 = time.perf_counter()
            from ..analysis.trace_analysis import estimate_anisotropy_for_dataset
            res = estimate_anisotropy_for_dataset(ds)
            dt = time.perf_counter() - t0
            if res is not None and np.isfinite(res["z_scaling_factor"]):
                value = float(res["z_scaling_factor"])
                if 0.5 <= value <= 1.0:
                    ds.set_z_scaling_factor(
                        value, source="estimated (trace anisotropy)"
                    )
                    ds.derived["z_scaling_factor"] = np.asarray([value], dtype=float)
                    ds.derived["z_scaling_factor_sizes_x"] = res["x"].sizes
                    ds.derived["z_scaling_factor_sizes_y"] = res["y"].sizes
                    ds.derived["z_scaling_factor_sizes_z"] = res["z"].sizes
                    self._state.log(
                        f"Estimated Z scaling factor for '{ds.name}': {value:.4g} "
                        f"(trace anisotropy, raw last-valid z, {dt:.1f}s)"
                    )
        except Exception as exc:
            self._state.log(f"Z scaling factor estimation failed for '{ds.name}': {exc}", "WARN")
        if "z_scaling_factor" not in ds.derived:
            ds.set_z_scaling_factor(
                1.0, source="estimate out of range (reset to 1.0)"
            )
            ds.derived["z_scaling_factor"] = np.asarray([1.0], dtype=float)
            shown = f"{value:.4g}" if value is not None else "failed"
            self._state.log(
                f"Z scaling factor for '{ds.name}': estimate {shown} outside [0.5, 1.0] — reset to 1.0.",
                "WARN",
            )
        self._state.notify_calibration_changed(idx)
        self._post_load_next(ds, self._post_load_loc_prec)

    def _post_load_loc_prec(self, ds) -> None:
        if self._post_load_index(ds) is None:
            return
        # A pooled averaged particle has no real traces — its ``tid`` is a per-loc
        # placeholder (each localization its own "trace"), so per-trace precision is
        # meaningless and iterating the N single-point traces is pure wasted work.
        if ds.metadata.get("particle_average"):
            self._post_load_next(ds, self._post_load_density)
            return
        data_prefs = self._state.prefs.get("data", {})
        if data_prefs.get("compute_loc_prec", True) and "sigma_per_trace_nm" not in ds.derived:
            try:
                import time

                import numpy as np
                self._state.log(f"Computing localization precision of '{ds.name}'…")
                self._state.status_progress(f"Computing localization precision of '{ds.name}'")
                t0 = time.perf_counter()
                from ..analysis.localization_precision import stddev_per_trace
                from ..core.loader import attr_values_1d
                loc_x = np.asarray(attr_values_1d(ds, "loc_x"))
                loc_y = np.asarray(attr_values_1d(ds, "loc_y"))
                _z = attr_values_1d(ds, "loc_z")
                loc_z = np.zeros_like(loc_x) if _z is None else np.asarray(_z)
                _tid = attr_values_1d(ds, "tid")
                tid = np.arange(loc_x.size) if _tid is None else np.asarray(_tid)
                result = stddev_per_trace(np.column_stack([loc_x, loc_y, loc_z]), tid)
                dt = time.perf_counter() - t0
                ds.attr["sigma_per_trace_nm"] = result["per_trace_sigma_xyz"]
                ds.attr["sigma_trace_ids"] = result["trace_ids"]
                ds.derived["sigma_per_trace_nm"] = result["per_trace_sigma_xyz"]
                ds.derived["sigma_trace_ids"] = result["trace_ids"]
                ds.cali.loc_precision = np.asarray(result["median_sigma_xyz"], dtype=float)
                self._state.log(
                    f"Computed localization precision for '{ds.name}' using StdDev per trace "
                    f"({dt:.1f}s): median sigma=({result['median_sigma_xyz'][0]:.3g}, "
                    f"{result['median_sigma_xyz'][1]:.3g}, {result['median_sigma_xyz'][2]:.3g}) nm."
                )
            except Exception as exc:
                self._state.log(f"Localization precision computation skipped for '{ds.name}': {exc}", "WARN")
        self._post_load_next(ds, self._post_load_density)

    def _post_load_density(self, ds) -> None:
        if self._post_load_index(ds) is None:
            return
        # A pooled averaged particle is a dense superposition centred at the origin.
        # The KD-tree range-count auto→voxel heuristic estimates work from the
        # bounding-box *volume* assuming uniform density; the averaged cloud is the
        # opposite (dense core, box-sized extent), so the estimate can undershoot and
        # run the exact all-core KD-tree at ≈O(N²) for minutes (the reported freeze).
        # Local density of a synthetic super-particle is not a standard measurement
        # anyway — skip on load; the user can compute it on demand.
        if ds.metadata.get("particle_average"):
            self._post_load_next(ds, self._post_load_finalize)
            return
        data_prefs = self._state.prefs.get("data", {})
        if data_prefs.get("compute_local_density", True) and "den" not in ds.attr:
            try:
                import time
                self._state.log(f"Computing local density of '{ds.name}'…")
                self._state.status_progress(f"Computing local density of '{ds.name}'")
                t0 = time.perf_counter()
                from ..analysis.local_density import compute_local_density_for_dataset
                density, method, detail = compute_local_density_for_dataset(ds, self._state.prefs)
                dt = time.perf_counter() - t0
                if "auto fallback" in detail:
                    fallback_msg = f"Local density auto fallback for '{ds.name}': {detail}."
                    print(fallback_msg)
                    self._state.log(fallback_msg, "WARN")
                ds.attr["den"] = density
                ds.attr["local_density"] = density
                ds.derived["den"] = density
                ds.derived["local_density"] = density
                for name in ("den", "local_density"):
                    if name not in ds.prop.attr_names:
                        ds.prop.attr_names.append(name)
                self._state.log(
                    f"Computed local density for '{ds.name}' using {method} ({detail}, {dt:.1f}s)."
                )
            except Exception as exc:
                self._state.log(f"Local density computation skipped for '{ds.name}': {exc}", "WARN")

        # For all-iteration loads ds.attr is not the last-valid materialization;
        # compute density at the last-valid selection too so mfx_get can
        # broadcast `den` across iteration views (matches a last-valid
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
                        np.asarray(z, dtype=float) * 1e9 * ds.cali.z_scaling_factor,
                    ])
                    dims = 3 if ds.prop.num_dim == 3 else 2
                    density, method, detail = compute_local_density_for_points(
                        points_nm, self._state.prefs, dimensions=dims,
                    )
                    if "auto fallback" in detail:
                        fallback_msg = (
                            f"Last-valid local density auto fallback for '{ds.name}': {detail}."
                        )
                        print(fallback_msg)
                        self._state.log(fallback_msg, "WARN")
                    ds.components.derived_last["den"] = density
                    self._state.log(
                        f"Computed last-valid local density for '{ds.name}' using {method} ({detail})."
                    )
            except Exception as exc:
                self._state.log(
                    f"Last-valid local density computation skipped for '{ds.name}': {exc}", "WARN",
                )
        self._post_load_next(ds, self._post_load_finalize)

    def _post_load_finalize(self, ds) -> None:
        idx = self._post_load_index(ds)
        if idx is None:
            return
        # Restored filters (metadata sidecar or Zarr viewer/state) re-evaluate
        # now that derived attributes (den, …) exist, so a re-opened processed
        # dataset shows its saved filter state.
        if ds.state.get("filter_specs"):
            try:
                from ..core.loader import apply_saved_filters
                if apply_saved_filters(ds):
                    self._state.notify_filter_changed(idx)
            except Exception as exc:
                self._state.log(f"Restoring filters for '{ds.name}' failed: {exc}", "WARN")
        self._state.status_message.emit(f"Finished processing '{ds.name}'.")

    def _on_dataset_removed(self, idx: int) -> None:
        for mapping in (
            self._data_windows,
            self._render_windows,
            self._scatter_windows,
            self._histogram_windows,
            self._attr_windows,
            self._attr_cpu_windows,
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
        self._sync_attribute_gpu_action()
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

    def _on_focus_changed(self, _old, new) -> None:
        """Record the last render/scatter window the user focused (for the LUT
        button). Guarded — focus can shift to a window being torn down."""
        if new is None:
            return
        try:
            win = new.window()
            if win is not None and win is not self and hasattr(win, "open_lut_dialog"):
                self._last_active_plot_window = win
                # Focusing a plot window refreshes its (already-open) LUT dialog,
                # so the dialog reflects whichever view the user is working in.
                sync = getattr(win, "sync_lut_dialog", None)
                if callable(sync):
                    sync()
                self._retarget_lut_dialog(win)
        except (RuntimeError, AttributeError):
            pass

    @staticmethod
    def _lut_view_shows_dataset(view, idx: "int | None") -> bool:
        """True when *view* displays dataset *idx* (directly or as an overlay
        channel), so the LUT button targets the view of the active dataset."""
        if view is None or idx is None:
            return view is not None
        try:
            for attr in ("_idx", "_dataset_idx", "dataset_idx"):
                value = getattr(view, attr, None)
                if callable(value):
                    value = value()
                if value == idx:
                    return True
            return any(
                ch.get("dataset_idx") == idx
                for ch in getattr(view, "_channels", [])
            )
        except (RuntimeError, TypeError, AttributeError):
            return False

    def _retarget_lut_dialog(self, view) -> None:
        """Move the one open LUT dialog onto the newly focused *view*.

        The dialog is per-view, so 'following focus' means closing the other
        view's and opening this one's; the geometry is carried over so it stays
        put instead of jumping.  Does nothing when no LUT dialog is open — this
        must never *summon* one just because the user clicked a plot.
        """
        own = getattr(view, "_lut_dialog", None)
        try:
            if own is not None and own.isVisible():
                return                      # already showing this view's
        except RuntimeError:
            own = None

        dialog = getattr(self._state, "_shared_lut_dialog", None)
        try:
            if dialog is None or not dialog.isVisible():
                return                      # nothing open; never summon one
            geometry = dialog.geometry()
        except RuntimeError:
            return

        # Rebinding keeps the same window, so it neither blinks nor moves; the
        # geometry is restored anyway in case the view re-lays it out.
        try:
            view.open_lut_dialog()
        except Exception as exc:
            self._state.log(f"LUT retarget failed: {exc}", "ERROR")
            return
        try:
            dialog.setGeometry(geometry)
        except RuntimeError:
            pass

    def _lut_capable_views(self) -> list:
        """Every view that can own a LUT dialog."""
        views: list = []
        registries = [
            self._render_windows,
            self._scatter_windows,
            self._attr_windows,
            self._attr_cpu_windows,
        ]
        # Present only when the advanced renderer has been opened.
        advanced = getattr(self, "_advanced_render_windows", None)
        if isinstance(advanced, dict):
            registries.insert(1, advanced)
        for registry in registries:
            for view in registry.values():
                views.append(view)
                volume = getattr(view, "_volume_window", None)
                if volume is not None:
                    views.append(volume)
        return views

    def _close_other_lut_dialogs(self, keep) -> None:
        """No-op kept for callers: there is only ever one LUT dialog now.

        ``ui/lut_dialog.py::shared_lut_dialog`` hands every view the same
        instance and rebinds its callbacks, so nothing else can be on screen to
        close.  This used to hide the other views' dialogs, which could not
        actually guarantee the invariant — ``close()`` may be refused, and every
        view that had ever opened one left a hidden instance behind.
        """
        del keep

    def _open_lut_on_view(self, view, idx: "int | None") -> bool:
        """Open *view*'s LUT dialog if it is a live, visible LUT-capable view of
        dataset *idx*. Returns True once handled (or on failure, to stop trying)."""
        if view is None:
            return False
        try:
            opener = getattr(view, "open_lut_dialog", None)
        except RuntimeError:
            return False
        if not callable(opener):
            return False
        try:
            if not view.isVisible():
                return False
        except (RuntimeError, AttributeError):
            return False
        if not self._lut_view_shows_dataset(view, idx):
            return False
        try:
            # Open first, then retire the others: a view whose data is not ready
            # declines to show (scatter does), and closing first would leave the
            # user with no LUT dialog at all.
            opener()
            self._close_other_lut_dialogs(view)
            self._last_active_plot_window = view
            return True
        except Exception as exc:
            self._state.log(f"LUT failed for active view: {exc}", "ERROR")
            return True

    def _show_lut(self) -> None:
        """Toolbar LUT button — open the LUT / colormap editor for the active
        view: render, scatter, Attribute Plot C dimension, or 3-D volume.

        Clicking the toolbar activates the main window, so ``activeWindow()`` no
        longer identifies the plot window; try the focused view and the last-used
        view, then scan the plot registries (and each render's 3-D volume
        window) for one showing the active dataset.
        """
        active = QApplication.activeWindow()
        if self._state.active_dataset is None:
            self._no_data_warning(); return
        idx = self._state.active_idx
        candidates = [active, getattr(self, "_last_active_plot_window", None)]
        for registry in (
            self._render_windows,
            self._scatter_windows,
            self._attr_windows,
            self._attr_cpu_windows,
        ):
            candidates.extend(registry.values())
            for view in registry.values():
                volume = getattr(view, "_volume_window", None)
                if volume is not None:
                    candidates.append(volume)
        seen: set[int] = set()
        for view in candidates:
            if id(view) in seen:
                continue
            seen.add(id(view))
            if self._open_lut_on_view(view, idx):
                return

        # No suitable view yet — open the active dataset's render and target it.
        rwin = self._render_window_for_dataset(idx)
        if rwin is None:
            self._show_render(idx)
            rwin = self._render_window_for_dataset(idx)
        if rwin is not None and self._open_lut_on_view(rwin, idx):
            return
        QMessageBox.information(
            self, "LUT",
            "Open a render, scatter, or four-dimensional Attribute Plot first, "
            "then use the LUT button.",
        )

    def _show_color_picker(self) -> None:
        """Open or raise the one modeless application-wide COLOR editor."""
        from .global_color_dialog import GlobalColorDialog
        from .modeless import show_modeless

        dialog = self._color_dialog
        if dialog is not None:
            try:
                dialog.show()
                dialog.raise_()
                dialog.activateWindow()
                return
            except RuntimeError:
                self._color_dialog = None
        dialog = GlobalColorDialog(self._state)
        dialog.destroyed.connect(lambda _=None: setattr(self, "_color_dialog", None))
        self._color_dialog = dialog
        show_modeless(dialog, self)

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

    def _install_toolbar_icons(self) -> None:
        """Attach PNG icons to the toolbar QActions (item #5).

        Icons are normalized (white matte stripped, monochrome linework tinted to
        the button-text color) so the same source PNGs read correctly on both
        light and dark palettes."""
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
                action.setIcon(_adaptive_toolbar_icon(str(ipath)))
        # Angle tool lives on self (not the generated UI).
        angle_icon = icon_dir / "angle.png"
        if hasattr(self, "toolAngle") and angle_icon.exists():
            self.toolAngle.setIcon(_adaptive_toolbar_icon(str(angle_icon)))
        lasso_icon = icon_dir / "lasso.png"
        if hasattr(self, "toolMagneticLasso") and lasso_icon.exists():
            self.toolMagneticLasso.setIcon(_adaptive_toolbar_icon(str(lasso_icon)))

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
        if getattr(self, "_is_shutting_down", False):
            return
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

    def _maybe_startup_update_check(self) -> None:
        if getattr(self, "_is_shutting_down", False):
            return
        self._check_for_updates(silent=True)

    def _on_update_check_done(self, result, silent: bool, task=None) -> None:
        from .update_dialog import show_update_result
        self._update_tasks.discard(task)
        if getattr(self, "_is_shutting_down", False):
            return
        if not silent:
            self._status_label.setText("Ready.")
        if result.status == "update_available":
            self._state.log(
                f"Update available: {result.latest.tag} "
                f"(installed v{result.current_version}).", "INFO",
            )
        show_update_result(result, self, silent=silent, state=self._state)

    def _show_about(self) -> None:
        from .. import __version__
        QMessageBox.about(
            self, f"About {self.APP_NAME}",
            f"<h3>MINFLUX Data Viewer v{__version__}</h3>"
            "<p>A Python/Qt GUI tool for reading, filtering, and visualization of"
            " Abberior MINFLUX nanoscopy data.</p>"

            "<p>Under active development, Source code on GitHub:<br>"
            "<a href='https://github.com/embl-ic/minflux-viewer'>https://github.com/embl-ic/minflux-viewer</a></p>"

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
            "have not yet been fully approved by a human show a note in the status "
            "bar (\"AI-generated; pending human approval.\") when you hover over "
            "them.</p>"
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

        from .command_finder import SearchLineEdit
        self._toolbar_search = SearchLineEdit()
        self._toolbar_search.setPlaceholderText("type here to search")
        self._toolbar_search.setClearButtonEnabled(True)
        self._toolbar_search.setToolTip(
            "Search commands (Fiji-style). Type to filter; double-click for the "
            "full Command Finder.")
        fm = self._toolbar_search.fontMetrics()
        width = fm.horizontalAdvance("x" * 25) + 24   # ~25 letters + padding/clear button
        self._toolbar_search.setFixedWidth(int(width))
        #tb.addWidget(self._toolbar_search)
        self._toolbar_search_action = tb.addWidget(self._toolbar_search)
        self._toolbar_search_action.setVisible(False)          # <-- for the moment (2026.07.22) the search is disabled until the command finder is fully implemented, so hide the field to avoid confusion.
        self._setup_command_search()

    def _setup_command_search(self) -> None:
        """Wire the toolbar search field: a live completer of menu commands + a
        double-click full Command Finder dialog (Fiji-style)."""
        from PyQt6.QtWidgets import QCompleter

        from .command_finder import WideCompleterPopup

        self._command_entries: list = []
        self._command_finder = None
        self._search_display_map: dict = {}

        completer = QCompleter(self)
        completer.setCaseSensitivity(Qt.CaseSensitivity.CaseInsensitive)
        completer.setFilterMode(Qt.MatchFlag.MatchContains)
        completer.setCompletionMode(QCompleter.CompletionMode.PopupCompletion)
        completer.setMaxVisibleItems(15)
        # A wider, self-widening dropdown so full "Command · Menu path" text shows.
        completer.setPopup(WideCompleterPopup())
        completer.popup().setTextElideMode(Qt.TextElideMode.ElideNone)
        completer.activated[str].connect(self._on_search_activated)
        self._search_completer = completer
        self._toolbar_search.setCompleter(completer)

        self._toolbar_search.focusedIn.connect(self._refresh_command_index)
        self._toolbar_search.doubleClicked.connect(
            lambda: self._open_command_finder(self._toolbar_search.text()))
        self._toolbar_search.returnPressed.connect(self._on_search_return)
        self._apply_command_meta()

    def _apply_command_meta(self) -> None:
        """Stamp every menu command with its metadata from ``ui.command_meta`` so the
        Command Finder can show a Source and match on keywords/tags — e.g. the leaf
        *Segmentation › NPC › 2D* is findable by ``npc``, *Convolution* by
        ``matched filter``. The registry is keyed by QAction attribute name (resolved
        on either this window or the generated UI); plugin actions are tagged
        separately in ``_populate_plugins_menu`` via ``source_of``. Adding a new
        command means adding one ``COMMAND_META`` entry (missing ⇒ blank Source)."""
        from .command_meta import COMMAND_META
        u = self._ui
        for key, meta in COMMAND_META.items():
            if key.startswith("_"):
                continue                                   # group meta (e.g. _roi_convert)
            act = getattr(self, key, None) or getattr(u, key, None)
            if act is None:
                continue
            if meta.source:
                act.setProperty("command_source", meta.source)
            if meta.keywords:
                act.setProperty("command_keywords", " ".join(meta.keywords))
        # ROI Convert sub-actions are created in a loop (no per-action attribute),
        # so tag the whole submenu's leaves as one group.
        for group_key, menu_attr in (("_roi_convert", "menuRoiConvert"),
                                     ("_roi_fit", "menuRoiFit")):
            grp = COMMAND_META.get(group_key)
            menu = getattr(self, menu_attr, None)
            if grp is None or menu is None:
                continue
            for act in menu.actions():
                if act.isSeparator() or act.menu() is not None:
                    continue
                act.setProperty("command_source", grp.source)
                act.setProperty("command_keywords", " ".join(grp.keywords))

    def _refresh_command_index(self) -> None:
        from PyQt6.QtCore import QStringListModel

        from .command_finder import collect_commands
        self._command_entries = collect_commands(self.menuBar())
        self._search_display_map = {}
        displays = []
        for e in self._command_entries:
            disp = f"{e.text}   ·   {e.path}" if e.path else e.text
            base, k = disp, 2
            while disp in self._search_display_map:   # keep displays unique for the map
                disp = f"{base} ({k})"; k += 1
            self._search_display_map[disp] = e
            displays.append(disp)
        self._search_completer.setModel(QStringListModel(displays, self._search_completer))
        # Widen the dropdown to the longest command so full text is readable (the
        # popup is a top-level widget, so it may extend past the window; capped to
        # the screen, and WideCompleterPopup nudges it back on-screen).
        popup = self._search_completer.popup()
        fm = popup.fontMetrics()
        longest = max((fm.horizontalAdvance(d) for d in displays), default=0)
        try:
            from PyQt6.QtGui import QGuiApplication
            cap = QGuiApplication.primaryScreen().availableGeometry().width() - 40
        except Exception:
            cap = 2000
        popup.setMinimumWidth(min(longest + 40, max(cap, 200)))

    def _on_search_activated(self, text: str) -> None:
        entry = self._search_display_map.get(text)
        if entry is None:
            return
        self._toolbar_search.clear()
        if entry.enabled:
            entry.action.trigger()
        else:
            self._state.log(f"'{entry.text}' is currently unavailable (needs a dataset?).", "WARN")

    def _on_search_return(self) -> None:
        # Enter with the completer popup open is handled by the completer; otherwise
        # open the full Command Finder on the current text.
        try:
            if self._search_completer.popup().isVisible():
                return
        except Exception:
            pass
        text = self._toolbar_search.text().strip()
        if text:
            self._open_command_finder(text)

    def _on_command_finder_destroyed(self, *_args) -> None:
        # The dialog is WA_DeleteOnClose; drop the stale reference so a later open
        # doesn't touch a deleted C++ object (RuntimeError).
        self._command_finder = None

    def _command_finder_alive(self) -> bool:
        dlg = self._command_finder
        if dlg is None:
            return False
        try:
            dlg.isVisible()          # touches the C++ object; raises if deleted
            return True
        except RuntimeError:
            self._command_finder = None
            return False

    def _open_command_finder(self, initial: str = "") -> None:
        from .command_finder import CommandFinderDialog, collect_commands
        from .modeless import show_modeless
        provider = lambda: collect_commands(self.menuBar())
        if not self._command_finder_alive():
            self._command_finder = CommandFinderDialog(provider, owner=self, initial=initial)
            self._command_finder.destroyed.connect(self._on_command_finder_destroyed)
            show_modeless(self._command_finder, self)
        else:
            self._command_finder.refresh()
            self._command_finder.set_query(initial)   # overwrite the dialog's filter
            self._command_finder.show()
            self._command_finder.raise_()
            self._command_finder.activateWindow()
        # The dialog is now the active finder — reset the toolbar field to its
        # placeholder so it's clearly out of the way (it still works if re-typed).
        self._toolbar_search.clear()

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

    def _apply_menu_separator_style(self) -> None:
        """Make menu **section separators** clearly visible. The native 1-px etch
        is almost invisible on some themes / RDP sessions; this draws a solid line
        with an inset margin (Fiji-style), theme-adaptive (darker on a light theme,
        lighter on a dark one). App-wide so menu-bar and right-click menus match.
        Styling only `QMenu::separator` leaves the items' native rendering intact."""
        from PyQt6.QtGui import QPalette
        app = QApplication.instance()
        if not isinstance(app, QApplication):
            return
        try:
            dark = app.palette().color(QPalette.ColorRole.Window).lightness() < 128
        except Exception:
            dark = False
        color = "#9a9a9a" if dark else "#6f6f6f"
        rule = (f"QMenu::separator {{ height: 1px; background: {color}; "
                f"margin: 4px 8px; }}")
        existing = app.styleSheet() or ""
        if "QMenu::separator" not in existing:
            app.setStyleSheet((existing + "\n" + rule).strip())

    def showEvent(self, event) -> None:
        super().showEvent(event)
        if not getattr(self, "_positioned", False):
            self._positioned = True
            # Open on the active monitor (the screen under the cursor) in the
            # upper-right region but nudged toward centre — 75 % toward the right
            # edge and 25 % down (75 % up from the bottom) — so child windows
            # (render, Log, MSR reader) still have room to the left and below
            # without the main window hugging the extreme corner.
            from .modeless import ensure_on_screen
            ensure_on_screen(self, align=(0.75, 0.25))
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
        self._is_shutting_down = True
        app = QApplication.instance()
        if app is not None:
            try:
                app.removeEventFilter(self)
            except Exception:
                pass
        self._state.save_prefs()
        self._close_all_child_windows()
        self._drain_background_tasks()
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

    def _drain_background_tasks(self) -> None:
        """Cancel queued Qt background work before QApplication teardown."""
        for task in list(getattr(self, "_update_tasks", set())):
            try:
                task.cancel()
            except Exception:
                pass
            try:
                task.signals.done.disconnect()
            except Exception:
                pass
        self._update_tasks.clear()
        for task in list(getattr(self, "_ome_zarr_tasks", set())):
            try:
                task.cancel()
            except Exception:
                pass
            for signal_name in ("progress", "done", "failed", "cancelled"):
                try:
                    getattr(task.signals, signal_name).disconnect()
                except Exception:
                    pass
        self._ome_zarr_tasks.clear()
        # A Zarr load/save in flight must not deliver into a closing window.
        for task in list(getattr(self, "_zarr_io_tasks", [])):
            for signal_name in ("stage", "done", "failed"):
                try:
                    getattr(task.signals, signal_name).disconnect()
                except Exception:
                    pass
        if hasattr(self, "_zarr_io_tasks"):
            self._zarr_io_tasks.clear()

        try:
            pool = QThreadPool.globalInstance()
            pool.clear()
            pool.waitForDone(2500)
        except Exception:
            pass

    def _close_all_child_windows(self, *, keep_log_console: bool = False) -> None:
        """
        Close every floating / tool window this main window has created.

        Safe to call repeatedly. Uses ``close()`` (not ``deleteLater``) so
        each child runs its own ``closeEvent`` (saves its settings, etc.)
        before being torn down by Qt on the main window's destruction.

        The shared LUT dialog is parentless (so it survives the view that
        opened it) and is therefore closed here explicitly — otherwise it
        outlives the main window.

        With ``keep_log_console=True`` the Log and Console windows are left open
        (used by the *Close All Windows* command, which clears everything else).
        """
        from .lut_dialog import close_shared_lut_dialog
        close_shared_lut_dialog(self._state)

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
        for mapping in (
            self._scatter_windows,
            self._histogram_windows,
            self._attr_windows,
            self._attr_cpu_windows,
        ):
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
        singletons = [
            "_filter_dlg",  "_ds_manager",
            "_log_win",     "_console_win", "_memory_win", "_roi_manager_win",
            "_script_editor_win",
        ]
        if keep_log_console:
            singletons = [a for a in singletons if a not in ("_log_win", "_console_win")]
        for attr in singletons:
            w = getattr(self, attr, None)
            if w is not None:
                try:
                    if hasattr(w, "force_close"):
                        w.force_close()
                    else:
                        w.close()
                except Exception: pass

        # Plugin dialogs that stashed themselves on the main window. Several MSR
        # reader dialogs can be open at once — close every one in the registry
        # (plus the legacy single-attr alias, for safety).
        msr_dialogs = list(getattr(self, "_plugin_msr_reader_dialogs", None) or [])
        legacy = getattr(self, "_plugin_msr_reader_dialog", None)
        if legacy is not None and legacy not in msr_dialogs:
            msr_dialogs.append(legacy)
        for w in msr_dialogs:
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
            "Drag and Drop data files here <supported .msr / .npy / .mat / .json / .zarr /...>   or use  File › Open"
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
