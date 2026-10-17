"""Application shell, Qt ownership and stable command entry points.

MainWindow owns window/task state and native Qt events. Capability controllers
implement file opening, saving, post-load work, ROI commands, overlays, window
registries and menus; forwarding methods preserve existing callers and signals.
"""

from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING
from weakref import ReferenceType

import numpy as np
from PyQt6.QtCore import QEvent, Qt, QTimer
from PyQt6.QtGui import (
    QAction,
    QActionGroup,
    QDragEnterEvent,
    QDropEvent,
    QIcon,
    QShortcut,
)
from PyQt6.QtWidgets import (
    QApplication,
    QComboBox,
    QCompleter,
    QDoubleSpinBox,
    QKeySequenceEdit,
    QLabel,
    QLineEdit,
    QMainWindow,
    QMenu,
    QPlainTextEdit,
    QProxyStyle,
    QSpinBox,
    QStyle,
    QTextEdit,
    QVBoxLayout,
    QWidget,
)

from .. import resource_path
from ..core import formats as _formats
from ..core.app_state import AppState
from ..core.dataset_key import DatasetKey
from .background_tasks import BackgroundTask
from .dataset_binding import (
    DatasetWindowMap,
)
from .window_support import _FMT_LOADERS as _FMT_LOADERS
from .window_support import _LASSO_FAMILY as _LASSO_FAMILY
from .window_support import _LINE_FAMILY as _LINE_FAMILY
from .window_support import _OVAL_FAMILY as _OVAL_FAMILY
from .window_support import _POINT_FAMILY as _POINT_FAMILY
from .window_support import _POLY_FAMILY as _POLY_FAMILY
from .window_support import _RECT_FAMILY as _RECT_FAMILY
from .window_support import _ROI_TOOL_DEFS as _ROI_TOOL_DEFS
from .window_support import _ROI_TOOL_EXTRA_LABELS as _ROI_TOOL_EXTRA_LABELS
from .window_support import _ROI_TOOL_VERBS as _ROI_TOOL_VERBS
from .window_support import _SUPPORTED_EXTS as _SUPPORTED_EXTS
from .window_support import _adaptive_toolbar_icon as _adaptive_toolbar_icon
from .window_support import _adaptive_toolbar_pixmap as _adaptive_toolbar_pixmap
from .window_support import _raise_or_create as _raise_or_create
from .window_support import _UpdateCheckSignals as _UpdateCheckSignals
from .window_support import _UpdateCheckTask as _UpdateCheckTask
from .window_support import _ZarrIoTask as _ZarrIoTask
from .window_support import roi_tool_labels as roi_tool_labels
from .window_support import roi_tool_status as roi_tool_status

if TYPE_CHECKING:
    from ..core.dataset import MinfluxDataset
    from .command_finder import SearchLineEdit
    from .preferences_dialog import PreferencesDialog
    from .wand_dialog import MagicWandDialog


_ROI_FILE_EXTS: frozenset[str] = _formats.roi_extensions()


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


class _ScrollableMenuStyle(QProxyStyle):
    """Make an over-tall menu scroll (arrows at the top/bottom edges, which
    auto-scroll on hover) instead of wrapping into multiple columns. Used for
    the recent-files submenu so a long history stays navigable."""

    def styleHint(self, hint, option=None, widget=None, returnData=None):  # noqa: N802 - Qt API
        if hint == QStyle.StyleHint.SH_Menu_Scrollable:
            return 1
        return super().styleHint(hint, option, widget, returnData)


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


class MainWindow(QMainWindow):
    """Top-level application window."""

    APP_NAME = "MINFLUX Data Viewer"

    menuOpenSample: QMenu
    menuSaveAs: QMenu
    actionSaveAsMinflux: QAction
    actionSaveAsMsr: QAction
    actionSaveAsMetadata: QAction
    actionSaveAsSpreadsheet: QAction
    actionSaveAsZarr: QAction
    actionSaveAsZarrZip: QAction
    actionSaveAsHdf5: QAction
    actionSaveAsOmeTiff: QAction
    actionClose: QAction
    actionCloseAll: QAction
    actionCloseAllWindows: QAction
    actionTrackingView: QAction
    menuMeasure: QMenu
    actionScaleBar: QAction
    actionPlotProfile: QAction
    actionSetMeasurements: QAction
    menuAnalyzeClustering: QMenu
    actionDbscan: QAction
    actionKNearestNeighbour: QAction
    menuAnalyzeTrace: QMenu
    actionTraceSize: QAction
    actionTraceAnisotropy: QAction
    menuAnalyzeSegmentation: QMenu
    actionSegConvolution: QAction
    actionSegShapeModel: QAction
    actionSegCurvilinear: QAction
    actionSegStraightenedVolume: QAction
    actionSegParticleAverage: QAction
    menuProcessChannel: QMenu
    actionChannelTool: QAction
    actionChannelCombine: QAction
    actionChannelSplit: QAction
    actionChannelFlatten: QAction
    menuConvertOverlay: QMenu
    actionChannelSeparateDcr: QAction
    actionChannelSeparateTime: QAction
    actionChannelSeparateAttribute: QAction
    actionRevertOverlay: QAction
    menuProcessRoi: QMenu
    actionRoiManager: QAction
    menuRoiConvert: QMenu
    _roi_convert_actions: dict[str, QAction]
    menuRoiFit: QMenu
    _roi_fit_actions: dict[str, QAction]
    actionRoiFitSpline: QAction
    actionRoiInterpolate: QAction
    actionRoiResize: QAction
    actionRoiSkeletonize: QAction
    actionRoiConvexHull: QAction
    actionRoiAddSlice: QAction
    actionRoiRestore: QAction
    actionAggregate: QAction
    actionCommandFinder: QAction
    actionTaskMonitor: QAction
    actionCheckUpdates: QAction
    _roi_tool_group: QActionGroup
    _line_variant: str
    _rect_variant: str
    _oval_variant: str
    _poly_variant: str
    _point_variant: str
    _lasso_variant: str
    toolAngle: QAction
    toolMagneticLasso: QAction
    _roi_tool_status_text: str
    _app_shortcut_focus: QShortcut
    _prefs_dialog: PreferencesDialog | None
    _toolbar_lead_spacer: QWidget
    _toolbar_search: SearchLineEdit
    _toolbar_search_action: QAction
    _command_entries: list
    _command_finder: None
    _search_display_map: dict
    _search_completer: QCompleter

    _post_load_tasks: dict[DatasetKey, BackgroundTask]
    _regroup_timer: QTimer
    _regroup_declined: set[str]
    _file_io_tasks: list[BackgroundTask]
    _zarr_io_tasks: list[BackgroundTask]
    _sim_bead_spec: tuple[np.ndarray, dict, list] | None
    _roi_tool_buttons: dict
    _wand_dialog: MagicWandDialog | None
    _last_active_log_ds: ReferenceType[MinfluxDataset]

    def __init__(self, state: AppState) -> None:
        super().__init__()
        self._state = state
        self._apply_menu_separator_style()
        self._data_windows = DatasetWindowMap(state)

        # One reusable plot window per dataset.
        self._scatter_windows = DatasetWindowMap(state)
        self._histogram_windows = DatasetWindowMap(state)
        self._attr_windows = DatasetWindowMap(state)
        self._scatter_win   = None       # compatibility alias: most recently raised
        self._histogram_win = None       # compatibility alias: most recently raised
        self._attr_win      = None       # compatibility alias: most recently raised
        self._filter_dlg    = None
        self._filter_dlgs = DatasetWindowMap(state)
        self._ds_manager    = None
        self._log_win       = None
        self._console_win   = None
        self._task_monitor_win = None
        self._roi_manager_win = None
        self._script_editor_win = None
        self._macro_recorder_win = None
        self._color_dialog = None
        self._shortcut_actions: dict[str, QAction] = {}
        self._roi_tool_actions: dict[str, QAction] = {}
        self._window_cycle_index = -1
        # One render window per dataset. Every per-dataset registry is keyed by
        # the dataset's key and read by position (ui/dataset_binding.py), so an
        # earlier dataset closing leaves the others' windows where they are.
        self._render_windows = DatasetWindowMap(state)
        # ...and one tracking view per dataset, the same contract.
        self._tracking_windows = DatasetWindowMap(state)
        # One modeless MSD workbench per dataset; results stay beside the views.
        self._msd_windows = DatasetWindowMap(state)
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
        # Rebuilt only when it is about to be shown: checking which recent
        # files still exist stats every entry, and on a missing network or
        # removable drive each stat can take tens of ms -- measured 2.2 s on
        # the GUI thread, once per opened dataset, for a menu nobody had open.
        self._recent_menu_stale = False
        self._recent_menu.aboutToShow.connect(self._refresh_recent_menu_if_stale)

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
        state.dataset_events.added.connect(self._on_dataset_added)
        state.dataset_events.removed.connect(self._on_dataset_removed)
        state.dataset_events.active_changed.connect(self._on_active_changed)
        state.status_message.connect(self._status_label.setText)
        state.log_message.connect(self._on_log_message)
        state.colors_changed.connect(self._on_colors_changed)
        state.colormap_order_changed.connect(self._on_colormap_order_changed)
        state.dataset_events.overlay_manual_alignment_requested.connect(
            self._on_overlay_manual_alignment_requested
        )

        # Remembered ROI duplicate/crop options, per dataset (session-only,
        # keyed by dataset identity — "use the same setup and stop asking").
        self._roi_crop_setup: dict = {}
        # Last active ROI draft per dataset index, for Process › ROI › Restore ROI
        # (restore across views / after an accidental delete).
        # Restore ROI's last draft per dataset, keyed by dataset key.
        self._roi_last_active = DatasetWindowMap(state)

        # Tier-A in-app update check (opt-out via Preferences > File). Delayed so
        # it never slows startup; silent unless a newer release exists.
        self._is_shutting_down = False
        self._update_tasks: set = set()
        #: At most ONE processing recipe held for a dataset not yet
        #: loaded, as ``(path, payload)``. A second one replaces it, and
        #: it is dropped as soon as a dataset is added that it does not
        #: match -- see ``_take_pending_metadata``. Session-only: a stale
        #: recipe applying itself days later would be worse than none.
        self._pending_metadata: tuple = ()
        #: Sidecar paths named explicitly in the drop/argv batch being
        #: routed, so a data file dropped WITH its recipe applies it
        #: without asking (the user already said so by dropping both).
        self._batch_sidecars: dict = {}
        #: Batch sidecars already consumed by their data file, so the recipe is
        #: not routed a second time on its own.
        self._batch_sidecars_used: set = set()
        if state.prefs.get("file", {}).get("check_updates_on_startup", False):
            QTimer.singleShot(3000, self._maybe_startup_update_check)

    def createPopupMenu(self):  # noqa: N802 - Qt override
        """Suppress Qt's default toolbar visibility popup on right-click."""
        return None

    # ------------------------------------------------------------------
    # Action wiring
    # ------------------------------------------------------------------

    def _connect_actions(self) -> None:
        'Connect every QAction from the generated UI to its handler.'
        from . import menu_controller as controller
        return controller._connect_actions(self)

    # ------------------------------------------------------------------
    # Menu and shortcuts
    # ------------------------------------------------------------------

    def _configure_menus(self) -> None:
        from . import menu_controller as controller
        return controller._configure_menus(self)

    def _keep_standard_actions_in_declared_menus(self) -> None:
        'Do not let macOS TextHeuristicRole relocate these actions.'
        from . import menu_controller as controller
        return controller._keep_standard_actions_in_declared_menus(self)

    def _mark_ai_unapproved_actions(self) -> None:
        'Visually separate AI-generated/unapproved actions from approved UI.'
        from . import menu_controller as controller
        return controller._mark_ai_unapproved_actions(self)

    def _mark_action_ai_unapproved(self, action: QAction) -> None:
        'Flag an AI-generated/unapproved action with a status-bar hover note.'
        from . import menu_controller as controller
        return controller._mark_action_ai_unapproved(self, action)

    def _apply_shortcuts(self) -> None:
        from . import menu_controller as controller
        return controller._apply_shortcuts(self)

    def _shortcut_command_keys(self, *, include_action_commands: bool=True) -> tuple[str, ...]:
        from . import menu_controller as controller
        return controller._shortcut_command_keys(self, include_action_commands=include_action_commands)

    def _install_window_shortcuts(self, widget: QWidget | None, *, include_action_commands: bool=True) -> None:
        'Install/update menu shortcuts on a top-level viewer window.'
        from . import menu_controller as controller
        return controller._install_window_shortcuts(self, widget, include_action_commands=include_action_commands)

    def _refresh_owned_window_shortcuts(self) -> None:
        from . import menu_controller as controller
        return controller._refresh_owned_window_shortcuts(self)

    def _trigger_shortcut_command(self, command: str) -> None:
        from . import menu_controller as controller
        return controller._trigger_shortcut_command(self, command)

    def _shortcut_text(self, key: str) -> str:
        from . import menu_controller as controller
        return controller._shortcut_text(self, key)

    def _event_sequence_text(self, event) -> str:
        from . import menu_controller as controller
        return controller._event_sequence_text(self, event)

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
            elif key in ("rectangle", "oval", "polygon", "point", "lasso"):
                # ⚠ A family button must reach its OWN switcher. "point" was
                # missing here, so right-clicking Point fell through to
                # _show_roi_tool_menu and listed every ROI tool in the
                # application instead of the two point variants.
                self._show_shape_family_menu(obj, pos, key)
            else:
                self._show_roi_tool_menu(obj, pos)
            return True
        # Double-click a ROI toolbar button → that tool's options, the gesture
        # Fiji uses. Only the magic wand has any, so every other button falls
        # through to its normal behaviour rather than opening an empty dialog.
        if (event.type() == QEvent.Type.MouseButtonDblClick
                and obj in getattr(self, "_roi_tool_buttons", {})):
            key = self._roi_tool_buttons[obj]
            if self._shape_variant(key) == "magic_wand":
                self._show_wand_dialog()
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
        from . import menu_controller as controller
        return controller._handle_shortcut_keypress(self, seq, event)

    def _shortcut_command_for_sequence(self, seq: str) -> str | None:
        from . import menu_controller as controller
        return controller._shortcut_command_for_sequence(self, seq)

    def _shortcut_conflicts(self) -> dict[str, list[str]]:
        from . import menu_controller as controller
        return controller._shortcut_conflicts(self)

    @staticmethod
    def _focus_is_shortcut_editor() -> bool:
        from . import menu_controller as controller
        return controller._focus_is_shortcut_editor()

    def _shortcut_allowed_in_editing(self, command: str) -> bool:
        from . import menu_controller as controller
        return controller._shortcut_allowed_in_editing(self, command)

    def _set_active_from_focused_dataset_window(self) -> None:
        'Use the focused dataset-owned plot/view when a shortcut targets data.'
        from . import menu_controller as controller
        return controller._set_active_from_focused_dataset_window(self)


    # ------------------------------------------------------------------
    # File loading
    # ------------------------------------------------------------------

    def _open_dialog(self) -> None:
        from . import file_open_controller as controller
        return controller._open_dialog(self)

    def _open_msr_dialog(self, msr_path: str) -> None:
        'Hand an ``.msr`` file to the MSR Reader plugin (drag-drop entry point).'
        from . import file_open_controller as controller
        return controller._open_msr_dialog(self, msr_path)

    def _open_viewer_msr(self, msr_path: str) -> None:
        'Open a ``.msr`` this application wrote, restoring its session.'
        from . import file_open_controller as controller
        return controller._open_viewer_msr(self, msr_path)

    def _on_viewer_msr_loaded(self, datasets, name: str, path: str, *, messages=None) -> None:
        from . import file_open_controller as controller
        return controller._on_viewer_msr_loaded(self, datasets, name, path, messages=messages)

    def _on_msr_open_failed(self, name: str, message: str) -> None:
        from . import file_open_controller as controller
        return controller._on_msr_open_failed(self, name, message)

    # ------------------------------------------------------------------
    # Sample data (Data Simulator) — File › Open Sample Data presets
    # ------------------------------------------------------------------
    def sample_presets(self) -> list:
        from . import file_open_controller as controller
        return controller.sample_presets(self)

    def set_sample_presets(self, presets: list) -> None:
        'Persist the Sample-Data presets and rebuild the File submenu.'
        from . import file_open_controller as controller
        return controller.set_sample_presets(self, presets)

    def _rebuild_sample_menu(self) -> None:
        '(Re)build File › Open Sample Data from the persisted presets.'
        from . import file_open_controller as controller
        return controller._rebuild_sample_menu(self)

    def _open_data_simulator(self) -> None:
        from . import file_open_controller as controller
        return controller._open_data_simulator(self)

    def _finalize_sim_dataset(self, name, coords, tid, attrs):
        "Build one simulated dataset from arrays (version = 'simulation', Z scaling factor pinned)."
        from . import file_open_controller as controller
        return controller._finalize_sim_dataset(self, name, coords, tid, attrs)

    def _build_simulated_dataset(self, name: str, kwargs: dict):
        from . import file_open_controller as controller
        return controller._build_simulated_dataset(self, name, kwargs)

    def generate_simulated(self, config: dict, *, include_beads: bool=False, bead_count: int=4) -> None:
        'Generate the dataset(s) for a Data-Simulator config / Sample-Data preset.'
        from . import file_open_controller as controller
        return controller.generate_simulated(self, config, include_beads=include_beads, bead_count=bead_count)

    def _generate_sim_overlay(self, p: dict) -> None:
        'NPC 3-channel overlay: co-registered channels on one shared scaffold.'
        from . import file_open_controller as controller
        return controller._generate_sim_overlay(self, p)

    def _generate_sim_dcr(self, p: dict) -> None:
        'NPC 2-channel by DCR: one dataset, two reporters via a bimodal ``dcr``.'
        from . import file_open_controller as controller
        return controller._generate_sim_dcr(self, p)

    def _generate_sim_tracking(self, p: dict) -> None:
        'A pure tracking dataset: molecules moving, with real timestamps.'
        from . import file_open_controller as controller
        return controller._generate_sim_tracking(self, p)

    def _generate_sim_track_overlay(self, p: dict) -> None:
        'A structure channel and a tracking channel, as one overlay.'
        from . import file_open_controller as controller
        return controller._generate_sim_track_overlay(self, p)

    def _generate_sim_ecoli(self, p: dict) -> None:
        'One rod cell of labelled HlyB dimers — a known-distance control.'
        from . import file_open_controller as controller
        return controller._generate_sim_ecoli(self, p)

    def _generate_sim_single(self, p: dict) -> None:
        'Generate a single-structure sample (``channels > 1`` = independent overlay).'
        from . import file_open_controller as controller
        return controller._generate_sim_single(self, p)

    def _route_file(self, path: str) -> None:
        'Route a single file path to the correct loader based on extension.'
        from . import file_open_controller as controller
        return controller._route_file(self, path)

    def _route_path(self, path: str) -> None:
        'Route a path that may be a file OR a directory.'
        from . import file_open_controller as controller
        return controller._route_path(self, path)

    # -- individual loaders ----------------------------------------

    def _start_dataset_load(self, path: str, loader, *, kind: str) -> None:
        "Load one ordinary dataset without running its parser on Qt's thread."
        from . import file_open_controller as controller
        return controller._start_dataset_load(self, path, loader, kind=kind)

    def _on_dataset_loaded(self, dataset, name: str, kind: str) -> None:
        from . import file_open_controller as controller
        return controller._on_dataset_loaded(self, dataset, name, kind)

    def _on_file_load_failed(self, name: str, message: str) -> None:
        from . import file_open_controller as controller
        return controller._on_file_load_failed(self, name, message)

    def _load_mat(self, path: str) -> None:
        from . import file_open_controller as controller
        return controller._load_mat(self, path)

    def _load_npy(self, path: str) -> None:
        from . import file_open_controller as controller
        return controller._load_npy(self, path)

    def _load_npz(self, path: str) -> None:
        from . import file_open_controller as controller
        return controller._load_npz(self, path)

    def _load_zarr(self, path: str) -> None:
        'Open a ``.zarr`` store off the UI thread.'
        from . import file_open_controller as controller
        return controller._load_zarr(self, path)

    def _begin_file_io(self, task, *, pool_name: str='file-io') -> None:
        'Start heavy non-Qt work and retain it until its terminal signal.'
        from . import file_open_controller as controller
        return controller._begin_file_io(self, task, pool_name=pool_name)

    def _begin_zarr_io(self, task) -> None:
        'Compatibility wrapper for the former Zarr-only task runner.'
        from . import file_open_controller as controller
        return controller._begin_zarr_io(self, task)

    def _finish_file_io(self, task) -> None:
        from . import file_open_controller as controller
        return controller._finish_file_io(self, task)

    def _finish_zarr_io(self, task) -> None:
        'Compatibility wrapper for the former Zarr-only task runner.'
        from . import file_open_controller as controller
        return controller._finish_zarr_io(self, task)

    def _on_zarr_io_failed(self, name: str, message: str, title: str) -> None:
        from . import file_open_controller as controller
        return controller._on_zarr_io_failed(self, name, message, title)

    def _on_zarr_loaded(self, project, name: str) -> None:
        'Install a loaded project — on the UI thread, where widgets live.'
        from . import file_open_controller as controller
        return controller._on_zarr_loaded(self, project, name)

    def _restore_zarr_rois(self, records, manifest: dict, dataset_indices) -> None:
        'Restore portable project ROI records into the session-level store.'
        from . import file_open_controller as controller
        return controller._restore_zarr_rois(self, records, manifest, dataset_indices)

    def _load_mbm_companion(self, path: str) -> None:
        'Open a standalone ``_mbm`` bead table in the MBM info window.'
        from . import file_open_controller as controller
        return controller._load_mbm_companion(self, path)

    def _on_mbm_companion_loaded(self, name: str, result) -> None:
        from . import file_open_controller as controller
        return controller._on_mbm_companion_loaded(self, name, result)

    def _load_csv(self, path: str) -> None:
        from . import file_open_controller as controller
        return controller._load_csv(self, path)

    def _load_spreadsheet(self, path: str) -> None:
        from . import file_open_controller as controller
        return controller._load_spreadsheet(self, path)

    def _on_spreadsheet_preview(self, table, prefs: dict, apply_sidecar: bool) -> None:
        from . import file_open_controller as controller
        return controller._on_spreadsheet_preview(self, table, prefs, apply_sidecar)

    # ------------------------------------------------------------------
    # Where a new window opens, and what covers what
    # ------------------------------------------------------------------
    #: Z-order for the windows one import produces, **bottom to top**. It is the
    #: order the MSR reader lists a file's contents in -- mfx, then mbm, then the
    #: image series -- so the MINFLUX data ends up in front of the images it was
    #: acquired on.
    IMPORT_STACK_ORDER = ("image", "mbm", "data")

    #: What answers the "mbm" tier. These come from three different modules and
    #: share no base class, so they are recognised by name; none of them is
    #: opened by an import, so the tier is usually empty.
    MBM_WINDOW_CLASSES = ("MbmInfoWindow", "AlignmentPlotWindow", "BeadsDriftDialog")

    def _place_new_window(self, win, dataset_idx: int | None=None) -> None:
        'Open *win* clear of the windows it would otherwise hide.'
        from . import window_controller as controller
        return controller._place_new_window(self, win, dataset_idx)

    def _mbm_windows(self) -> list:
        'Every visible bead-info window, whichever module it came from.'
        from . import window_controller as controller
        return controller._mbm_windows(self)

    def _dataset_windows(self, dataset_indices) -> list:
        'The plot/info windows of *dataset_indices*, render view last.'
        from . import window_controller as controller
        return controller._dataset_windows(self, dataset_indices)

    def restack_import_windows(self, dataset_indices=()) -> None:
        "Stack an import's windows the way the MSR reader lists its contents."
        from . import window_controller as controller
        return controller.restack_import_windows(self, dataset_indices)

    def _open_image_viewer(self, source, key: str, *, initial_series_index: int | None=None) -> None:
        'Show *source* (a TIFF or OBF image source) in a TIFF viewer window,'
        from . import file_open_controller as controller
        return controller._open_image_viewer(self, source, key, initial_series_index=initial_series_index)

    def _load_tiff(self, path: str) -> None:
        from . import file_open_controller as controller
        return controller._load_tiff(self, path)

    def _on_tiff_source_loaded(self, source, path: str, resolved: str) -> None:
        from . import file_open_controller as controller
        return controller._on_tiff_source_loaded(self, source, path, resolved)

    def _load_json(self, path: str) -> None:
        from . import file_open_controller as controller
        return controller._load_json(self, path)

    def _load_roi_json(self, path: str) -> None:
        'Load a native ROI-set file (.json / .roi / .zip) into the ROI Manager,'
        from . import file_open_controller as controller
        return controller._load_roi_json(self, path)

    def _mark_recent_menu_stale(self) -> None:
        'The recent-file list changed; rebuild the menu when it is next shown.'
        from . import menu_controller as controller
        return controller._mark_recent_menu_stale(self)

    def _refresh_recent_menu_if_stale(self) -> None:
        from . import menu_controller as controller
        return controller._refresh_recent_menu_if_stale(self)

    def _populate_recent_menu(self) -> None:
        from . import menu_controller as controller
        return controller._populate_recent_menu(self)

    def _show_recent_file_menu(self, path: str, global_pos) -> None:
        'Right-click context menu for a File › Open Recent entry.'
        from . import menu_controller as controller
        return controller._show_recent_file_menu(self, path, global_pos)

    def open_file_location(self, path: str) -> None:
        'Reveal a file in the OS file manager (falls back to opening its'
        from . import menu_controller as controller
        return controller.open_file_location(self, path)

    def _populate_plugins_menu(self) -> None:
        'Fill the Plugins menu from the plugin registry.'
        from . import menu_controller as controller
        return controller._populate_plugins_menu(self)

    # ------------------------------------------------------------------
    # Public window API  — the surface scripts and plugins may rely on
    # ------------------------------------------------------------------
    #
    # These are the ONLY window entry points published through ``mfv``. They
    # exist because external code cannot be refactored alongside a private
    # name: ``minflux_viewer/api/`` is a contract with code this project does
    # not own, and a leading underscore in that contract is a promise nobody
    # made. Same reasoning — and the same shape — as the Dataset Manager's
    # public ``close_datasets`` / ``duplicate_datasets`` /
    # ``combine_datasets_as_overlay`` batch entry points.
    #
    # Deliberately thin: each one delegates, unchanged, to the private
    # implementation. Do not move logic up here, and do not let the private
    # bodies start calling these.
    #
    # ⚠ Every ``_show_*`` below guards against ``QAction.triggered``'s
    # ``checked=False`` with ``type(x) is int`` — a bool would otherwise be
    # read as dataset index 0. So these wrappers must pass an ``int`` or
    # ``None`` through and nothing else; ``_dataset_idx_arg`` enforces that
    # rather than trusting the caller, because the caller is a script.

    @staticmethod
    def _dataset_idx_arg(idx) -> int | None:
        'Return the only two argument shapes the private openers accept.'
        from . import window_controller as controller
        return controller._dataset_idx_arg(idx)

    def show_render(self, idx: int | None=None):
        'Open (or raise) the Render View for a dataset (default: active).'
        from . import window_controller as controller
        return controller.show_render(self, idx)

    def show_scatter(self, idx: int | None=None):
        'Open (or raise) the Localization Scatter Plot for a dataset.'
        from . import window_controller as controller
        return controller.show_scatter(self, idx)

    def show_histogram(self, idx: int | None=None):
        'Open (or raise) the Attribute Histogram for a dataset.'
        from . import window_controller as controller
        return controller.show_histogram(self, idx)

    def show_attribute_plot(self, idx: int | None=None):
        'Open (or raise) the Attribute Plot for a dataset.'
        from . import window_controller as controller
        return controller.show_attribute_plot(self, idx)

    def show_console(self) -> None:
        'Open (or raise) the Console window.'
        from . import window_controller as controller
        return controller.show_console(self)

    def show_script_editor(self):
        'Open (or raise) the Script Editor.'
        from . import window_controller as controller
        return controller.show_script_editor(self)

    def show_macro_recorder(self):
        'Open (or raise) the modeless Macro Recorder plugin window.'
        from . import window_controller as controller
        return controller.show_macro_recorder(self)

    def show_hlyb_cell_collection(self, defaults=None):
        'Open or raise the persistent HlyB/D multi-dataset ROI pool.'
        from . import window_controller as controller
        return controller.show_hlyb_cell_collection(self, defaults)

    def _on_macro_recorder_destroyed(self, *_args) -> None:
        from . import window_controller as controller
        return controller._on_macro_recorder_destroyed(self, *_args)

    def _bind_scripting_api(self) -> None:
        'Expose this viewer instance through the runtime ``mfv`` module.'
        from . import window_controller as controller
        return controller._bind_scripting_api(self)

    # ------------------------------------------------------------------
    # Tool windows  — open once, raise if already open
    # ------------------------------------------------------------------

    def _on_overlay_manual_alignment_requested(self, dataset_idx) -> None:
        'Enter the existing channel-row Manual align mode for a dataset.'
        from . import overlay_controller as controller
        return controller._on_overlay_manual_alignment_requested(self, dataset_idx)

    def _show_script_editor(self):
        from . import window_controller as controller
        return controller._show_script_editor(self)

    def _show_scatter(self, dataset_idx: int | None=None):
        from . import window_controller as controller
        return controller._show_scatter(self, dataset_idx)

    def _show_tracking_view(self, dataset_idx: int | None=None):
        "Open (or raise) this dataset's tracking view."
        from . import window_controller as controller
        return controller._show_tracking_view(self, dataset_idx)

    def _show_msd_analysis(self, dataset_idx: int | None=None):
        "Open (or raise) the active dataset's modeless MSD workbench."
        from . import window_controller as controller
        return controller._show_msd_analysis(self, dataset_idx)

    def _show_histogram(self, dataset_idx: int | None=None):
        from . import window_controller as controller
        return controller._show_histogram(self, dataset_idx)

    def _show_attr_plot(self, dataset_idx: int | None=None):
        from . import window_controller as controller
        return controller._show_attr_plot(self, dataset_idx)


    def _show_filter(self) -> None:
        from . import window_controller as controller
        return controller._show_filter(self)

    def _load_filter_json(self, path: str) -> None:
        'Open (or raise) the filter dialog and append filters from a JSON file.'
        from . import window_controller as controller
        return controller._load_filter_json(self, path)

    def _show_dataset_manager(self) -> None:
        from . import window_controller as controller
        return controller._show_dataset_manager(self)

    def _get_dataset_manager(self):
        'Return the existing Dataset Manager, creating it if necessary.'
        from . import window_controller as controller
        return controller._get_dataset_manager(self)

    def _ensure_dataset_manager_visible(self) -> None:
        'Create/show the Dataset Manager without changing window focus.'
        from . import window_controller as controller
        return controller._ensure_dataset_manager_visible(self)

    def _show_roi_manager(self) -> None:
        from . import window_controller as controller
        return controller._show_roi_manager(self)

    # ------------------------------------------------------------------
    # ROI convert / enlarge-shrink / skeletonize (Process › ROI)
    # ------------------------------------------------------------------

    def _active_roi(self):
        'The active ROI as ``(record, kind, adapter)``: the single selected'
        from . import roi_command_controller as controller
        return controller._active_roi(self)

    def _clear_roi_cached_mask(self, record) -> None:
        "Drop a stored ROI's cached selection mask (state + derived) so a"
        from . import roi_command_controller as controller
        return controller._clear_roi_cached_mask(self, record)

    def _commit_roi_replacement(self, record, kind, adapter, new_record) -> None:
        'Replace a stored ROI in place; a draft stays a draft of the new type'
        from . import roi_command_controller as controller
        return controller._commit_roi_replacement(self, record, kind, adapter, new_record)

    def _convert_active_roi(self, target: str) -> None:
        from . import roi_command_controller as controller
        return controller._convert_active_roi(self, target)

    def _resize_active_roi(self) -> None:
        from . import roi_command_controller as controller
        return controller._resize_active_roi(self)

    def _skeletonize_active_roi(self) -> None:
        from . import roi_command_controller as controller
        return controller._skeletonize_active_roi(self)

    def _convex_hull_active_roi(self) -> None:
        'Process › ROI › Convex Hull — convert the active ROI to its convex'
        from . import roi_command_controller as controller
        return controller._convex_hull_active_roi(self)

    # ------------------------------------------------------------------
    # Process › ROI › Fit  (data-point fits + spline/interpolate)
    # ------------------------------------------------------------------
    _REGION_ROI_TYPES = {"rectangle", "oval", "polygon", "freehand", "composite"}

    def _roi_highlighted_points(self, record, ds):
        "(M, 2) localizations the region *record* highlights, in the ROI's plane"
        from . import roi_command_controller as controller
        return controller._roi_highlighted_points(self, record, ds)

    def _fit_active_roi(self, target: str) -> None:
        'Process › ROI › Fit ▸ Rectangle/Circle/Ellipse/Polygon/Convex Hull —'
        from . import roi_command_controller as controller
        return controller._fit_active_roi(self, target)

    def _spline_fit_active_roi(self) -> None:
        'Process › ROI › Fit ▸ Fit Spline — smooth the active ROI outline.'
        from . import roi_command_controller as controller
        return controller._spline_fit_active_roi(self)

    def _interpolate_active_roi(self) -> None:
        'Process › ROI › Fit ▸ Interpolate — resample the active ROI outline at'
        from . import roi_command_controller as controller
        return controller._interpolate_active_roi(self)

    # ------------------------------------------------------------------
    # Process › ROI › Restore ROI
    # ------------------------------------------------------------------
    def _coordinate_views_for_dataset(self, idx):
        'Open render / scatter windows for dataset *idx*.'
        from . import roi_command_controller as controller
        return controller._coordinate_views_for_dataset(self, idx)

    def _remember_roi_for_restore(self, window, record) -> None:
        'Cache the active ROI draft (called from the overlay controller) so'
        from . import roi_command_controller as controller
        return controller._remember_roi_for_restore(self, window, record)

    def _restore_roi(self) -> None:
        'Process › ROI › Restore ROI — put the active ROI draft onto the other'
        from . import roi_command_controller as controller
        return controller._restore_roi(self)



    def _show_conv_segmentation(self) -> None:
        'Analyze › Segmentation › Convolution… — open the interactive'
        from . import roi_command_controller as controller
        return controller._show_conv_segmentation(self)

    def _show_shape_segmentation(self) -> None:
        'Analyze › Segmentation › Shape Model… — open the known-geometry'
        from . import roi_command_controller as controller
        return controller._show_shape_segmentation(self)

    def _show_curvilinear_segmentation(self) -> None:
        'Analyze › Segmentation › Curvilinear Structures… — open the'
        from . import roi_command_controller as controller
        return controller._show_curvilinear_segmentation(self)

    def _show_straightened_volume_skeleton(self) -> None:
        'Analyze › Segmentation › Straightened Volume along Skeleton… —'
        from . import roi_command_controller as controller
        return controller._show_straightened_volume_skeleton(self)

    def _system_roi_color(self) -> str:
        "The user's configured default ROI color (Preferences ▸ Appearance ▸ ROI,"
        from . import roi_command_controller as controller
        return controller._system_roi_color(self)

    def add_segmentation_rois(self, idx: int, centers, *, side_nm: float, name_prefix: str, source: str, stroke_color: str | None=None, names: list[str] | None=None, log_message: str | None=None) -> int:
        'Add a rectangle ROI of side ``side_nm`` centred on each ``(x, y)`` in'
        from . import roi_command_controller as controller
        return controller.add_segmentation_rois(self, idx, centers, side_nm=side_nm, name_prefix=name_prefix, source=source, stroke_color=stroke_color, names=names, log_message=log_message)

    def add_point_rois_3d(self, idx: int, centers, *, name_prefix: str, source: str, stroke_color: str | None=None, names: list[str] | None=None, log_message: str | None=None) -> int:
        'Add a 3-D point ROI (full ``[x, y, z]`` nm geometry) for each centre in'
        from . import roi_command_controller as controller
        return controller.add_point_rois_3d(self, idx, centers, name_prefix=name_prefix, source=source, stroke_color=stroke_color, names=names, log_message=log_message)

    def add_polyline_rois(self, idx: int, paths, *, name_prefix: str, source: str, stroke_color: str | None=None, names: list[str] | None=None, log_message: str | None=None, roi_type: str='freehand_line') -> int:
        'Add an open poly-line (``freehand_line``) ROI tracing each path in'
        from . import roi_command_controller as controller
        return controller.add_polyline_rois(self, idx, paths, name_prefix=name_prefix, source=source, stroke_color=stroke_color, names=names, log_message=log_message, roi_type=roi_type)

    def add_polygon_rois(self, idx: int, polygons, *, name_prefix: str, source: str, stroke_color: str | None=None, names: list[str] | None=None, log_message: str | None=None) -> int:
        'Add a closed ``polygon`` ROI tracing each contour in *polygons* (each an'
        from . import roi_command_controller as controller
        return controller.add_polygon_rois(self, idx, polygons, name_prefix=name_prefix, source=source, stroke_color=stroke_color, names=names, log_message=log_message)

    def add_particle_average_dataset(self, points_xyz_nm, *, name: str, log_message: str | None=None) -> int | None:
        "Build a dataset from an averaged particle's pooled localizations"
        from . import roi_command_controller as controller
        return controller.add_particle_average_dataset(self, points_xyz_nm, name=name, log_message=log_message)

    def add_particle_average_overlay(self, channels: dict, *, luts: dict, name: str, log_message: str | None=None) -> int | None:
        'Add a **multi-channel** averaged overlay: ``channels`` = ``{name: (N,3) nm}``'
        from . import roi_command_controller as controller
        return controller.add_particle_average_overlay(self, channels, luts=luts, name=name, log_message=log_message)

    def _show_particle_average(self) -> None:
        'Analyze › Segmentation › Particle Average… — open the particle-averaging'
        from . import window_controller as controller
        return controller._show_particle_average(self)

    def _notify_view_state_changed(self) -> None:
        from . import window_controller as controller
        return controller._notify_view_state_changed(self)

    def dataset_view_status(self, idx: int) -> str:
        from . import window_controller as controller
        return controller.dataset_view_status(self, idx)

    def _on_roi_tool(self, tool: str, checked: bool) -> None:
        from . import roi_command_controller as controller
        return controller._on_roi_tool(self, tool, checked)

    def _activate_roi_tool(self, tool: str) -> None:
        from . import roi_command_controller as controller
        return controller._activate_roi_tool(self, tool)

    def _enter_ortho_for_volume_tool(self, tool: str) -> None:
        'Selecting a 3-D ROI tool turns the orthogonal view on.'
        from . import roi_command_controller as controller
        return controller._enter_ortho_for_volume_tool(self, tool)

    def _volume_tool_allowed(self, tool: str) -> bool:
        'Refuse a 3-D drawing tool on a 2-D dataset, before anything is drawn.'
        from . import roi_command_controller as controller
        return controller._volume_tool_allowed(self, tool)

    def _announce_roi_tool(self, tool: str) -> None:
        'Report in the status bar that *tool* is armed and waiting for a draw.'
        from . import roi_command_controller as controller
        return controller._announce_roi_tool(self, tool)

    def _sync_roi_tool_actions(self, tool: str='') -> None:
        from . import roi_command_controller as controller
        return controller._sync_roi_tool_actions(self, tool)

    def _update_line_button_icon(self) -> None:
        from . import roi_command_controller as controller
        return controller._update_line_button_icon(self)

    def _install_roi_tool_menus(self) -> None:
        from . import roi_command_controller as controller
        return controller._install_roi_tool_menus(self)

    def _show_line_family_menu(self, widget: QWidget, pos) -> None:
        'Fiji-style right-click switcher on the Line button.'
        from . import roi_command_controller as controller
        return controller._show_line_family_menu(self, widget, pos)

    def _select_line_variant(self, tool: str) -> None:
        from . import roi_command_controller as controller
        return controller._select_line_variant(self, tool)

    # --- Rectangle / Oval / Polygon family switchers ---
    def _shape_family(self, kind: str):
        from . import roi_command_controller as controller
        return controller._shape_family(self, kind)

    def _shape_variant(self, kind: str) -> str:
        from . import roi_command_controller as controller
        return controller._shape_variant(self, kind)

    def _show_shape_family_menu(self, widget: QWidget, pos, kind: str) -> None:
        'Fiji-style right-click switcher on a shape-family toolbar button.'
        from . import roi_command_controller as controller
        return controller._show_shape_family_menu(self, widget, pos, kind)

    def _select_shape_variant(self, kind: str, tool: str) -> None:
        from . import roi_command_controller as controller
        return controller._select_shape_variant(self, kind, tool)

    def _show_wand_dialog(self) -> None:
        'Open the Magic Wand Tool options, arming the wand with them.'
        from . import roi_command_controller as controller
        return controller._show_wand_dialog(self)

    def _on_wand_dialog_destroyed(self, *_args) -> None:
        from . import roi_command_controller as controller
        return controller._on_wand_dialog_destroyed(self, *_args)

    def _on_wand_parameters_changed(self) -> None:
        'Re-apply the new tolerances to the last wand click.'
        from . import roi_command_controller as controller
        return controller._on_wand_parameters_changed(self)

    def _rotated_icon(self, path: str, rotate_deg: float) -> QIcon:
        'A QIcon from *path*, optionally turned *rotate_deg*° (the rotated rectangle'
        from . import roi_command_controller as controller
        return controller._rotated_icon(self, path, rotate_deg)

    def _update_shape_button_icon(self, kind: str) -> None:
        from . import roi_command_controller as controller
        return controller._update_shape_button_icon(self, kind)

    def _show_roi_tool_menu(self, widget: QWidget, pos) -> None:
        from . import roi_command_controller as controller
        return controller._show_roi_tool_menu(self, widget, pos)

    def _show_log(self) -> None:
        'Open (or raise) the Log window — structured application events.'
        from . import window_controller as controller
        return controller._show_log(self)

    def _ensure_log_window(self, *, show: bool=True, raise_window: bool=False):
        from . import window_controller as controller
        return controller._ensure_log_window(self, show=show, raise_window=raise_window)

    def _on_log_message(self, message: str, level: str) -> None:
        'Auto-open the Log window the first time anything is logged — but placed'
        from . import window_controller as controller
        return controller._on_log_message(self, message, level)

    def _show_console(self) -> None:
        'Open (or raise) the Console window — raw stdout / stderr stream.'
        from . import window_controller as controller
        return controller._show_console(self)

    def _render_window_for_dataset(self, idx: int | None):
        'The open render window showing dataset *idx* — either keyed by it, or'
        from . import window_controller as controller
        return controller._render_window_for_dataset(self, idx)

    def _show_brightness_contrast(self) -> None:
        'Forward to the render window of the currently active dataset.'
        from . import window_controller as controller
        return controller._show_brightness_contrast(self)

    def _focus_main_window(self) -> None:
        'Restore, show, and raise the main data-viewer window.'
        from . import window_controller as controller
        return controller._focus_main_window(self)

    def _viewer_windows(self) -> list[QWidget]:
        from . import window_controller as controller
        return controller._viewer_windows(self)

    def _cycle_windows(self, direction: int) -> None:
        from . import window_controller as controller
        return controller._cycle_windows(self, direction)

    def _cycle_dataset(self, direction: int) -> None:
        from . import window_controller as controller
        return controller._cycle_dataset(self, direction)

    def _raise_dataset_window(self, idx: int) -> None:
        from . import window_controller as controller
        return controller._raise_dataset_window(self, idx)

    def _close_current_child_window(self) -> None:
        from . import window_controller as controller
        return controller._close_current_child_window(self)

    # ------------------------------------------------------------------
    # Preferences, memory monitor, duplicate
    # ------------------------------------------------------------------

    def _show_preferences(self) -> None:
        'Open the modeless Preferences dialog (Edit → Preferences…).'
        from . import menu_controller as controller
        return controller._show_preferences(self)

    def _on_preferences_accepted(self) -> None:
        'Rebuild the UI bits that depend on preferences, after OK.'
        from . import menu_controller as controller
        return controller._on_preferences_accepted(self)

    def _refresh_plot_preferences(self) -> None:
        from . import menu_controller as controller
        return controller._refresh_plot_preferences(self)

    def _on_colormap_order_changed(self) -> None:
        'Rebuild the colormap selectors that are built once and kept.'
        from . import menu_controller as controller
        return controller._on_colormap_order_changed(self)

    def _on_colors_changed(self, payload) -> None:
        'Repaint only consumers touched by a global color change.'
        from . import menu_controller as controller
        return controller._on_colors_changed(self, payload)

    def _show_set_measurements(self) -> None:
        from . import window_controller as controller
        return controller._show_set_measurements(self)

    # ------------------------------------------------------------------
    # Measure › Scale Bar
    # ------------------------------------------------------------------

    def _active_coordinate_view(self):
        'A 2-D coordinate view (render/scatter) for the active dataset, or None.'
        from . import roi_command_controller as controller
        return controller._active_coordinate_view(self)

    def _window_active_rectangle_bounds(self, win):
        "Bounds (x0,x1,y0,y1) of the active rectangle ROI in *win*'s current"
        from . import roi_command_controller as controller
        return controller._window_active_rectangle_bounds(self, win)

    @staticmethod
    def _nice_scalebar_width(span: float) -> float:
        from . import roi_command_controller as controller
        return controller._nice_scalebar_width(span)

    def _show_plot_profile(self) -> None:
        'ImageJ-style Plot Profile: localization density along a line / polyline /'
        from . import roi_command_controller as controller
        return controller._show_plot_profile(self)

    def _show_scale_bar(self) -> None:
        from . import roi_command_controller as controller
        return controller._show_scale_bar(self)

    @staticmethod
    def _rect_center(bounds) -> tuple[float, float]:
        from . import roi_command_controller as controller
        return controller._rect_center(bounds)

    def _add_scale_bar(self, win, params, *, roi_center) -> None:
        from . import roi_command_controller as controller
        return controller._add_scale_bar(self, win, params, roi_center=roi_center)

    def _edit_scale_bar(self, win, item) -> None:
        from . import roi_command_controller as controller
        return controller._edit_scale_bar(self, win, item)

    def _delete_scale_bar(self, win, item) -> None:
        from . import roi_command_controller as controller
        return controller._delete_scale_bar(self, win, item)

    def _split_active_channel_group(self) -> None:
        'Split the active multi-channel render overlay into one render per dataset.'
        from . import overlay_controller as controller
        return controller._split_active_channel_group(self)

    def _flatten_active_channel_group(self) -> None:
        'Process › Channel › Flatten — combine the active multi-channel overlay'
        from . import overlay_controller as controller
        return controller._flatten_active_channel_group(self)

    def _revert_overlay_to_original(self) -> None:
        'Process › Channel › Convert Dataset to Multi-Channel Overlay › Revert'
        from . import overlay_controller as controller
        return controller._revert_overlay_to_original(self)

    def _aggregate_active_dataset(self) -> None:
        'Process › Aggregate Localizations — Imspector-style per-trace photon'
        from . import overlay_controller as controller
        return controller._aggregate_active_dataset(self)

    def _build_aggregated_dataset(self, ds, thr):
        'Build (not add) an aggregated dataset from *ds* at photon threshold'
        from . import overlay_controller as controller
        return controller._build_aggregated_dataset(self, ds, thr)

    def _log_aggregated_dataset(self, ds, dataset_idx: int) -> None:
        'Record a parseable, result-tagged aggregation event in the Log.'
        from . import overlay_controller as controller
        return controller._log_aggregated_dataset(self, ds, dataset_idx)

    def _ask_aggregation_threshold(self, ds, pit, n_channels: int=1) -> int | None:
        'Modal photon-threshold picker for aggregation (default 3000).'
        from . import overlay_controller as controller
        return controller._ask_aggregation_threshold(self, ds, pit, n_channels)

    def _current_group_luts(self, group_indices: list[int]) -> dict[int, str]:
        'Return current render LUTs for group members, if a render is open.'
        from . import overlay_controller as controller
        return controller._current_group_luts(self, group_indices)

    def _show_task_monitor(self) -> None:
        'Open (or raise) the Task Monitor (Help → Task Monitor).'
        from . import window_controller as controller
        return controller._show_task_monitor(self)

    def _close_active_dataset(self) -> None:
        from . import window_controller as controller
        return controller._close_active_dataset(self)

    def close_datasets(self, indices) -> None:
        'Close (remove) every dataset in *indices*, and its windows.'
        from . import window_controller as controller
        return controller.close_datasets(self, indices)

    def duplicate_datasets(self, datasets) -> None:
        'Plain-duplicate every dataset in *datasets* (Dataset Manager'
        from . import window_controller as controller
        return controller.duplicate_datasets(self, datasets)

    def combine_datasets_as_overlay(self, indices) -> None:
        'Open *Process › Channel… › Combine* restricted to *indices*.'
        from . import window_controller as controller
        return controller.combine_datasets_as_overlay(self, indices)

    def reset_dataset(self, idx: int) -> None:
        'Dataset Manager *Reset* — put the dataset back to its as-loaded state.'
        from . import window_controller as controller
        return controller.reset_dataset(self, idx)

    def view_dataset_mbm(self, idx: int) -> None:
        "Dataset Manager *View mbm info…* — the dataset's beam-monitoring beads."
        from . import window_controller as controller
        return controller.view_dataset_mbm(self, idx)

    def view_dataset_image_series(self, idx: int) -> None:
        "Dataset Manager *View image series* — open this dataset's source ``.msr``"
        from . import window_controller as controller
        return controller.view_dataset_image_series(self, idx)

    # ------------------------------------------------------------------
    # Files dropped ONTO a dataset (Dataset Manager row drop)
    # ------------------------------------------------------------------

    #: Extensions a Dataset-Manager row accepts.  Two different verbs share it:
    #: the *apply to this dataset* kinds (a ROI set, a filter preset, a
    #: processing recipe, a confocal TIFF -- a ``.json`` is ambiguous on
    #: extension alone and is resolved by content), and every openable data
    #: format, which the row cannot meaningfully receive and is therefore opened
    #: as a NEW dataset.  Both are accepted because refusing a data file dropped
    #: on the manager -- the window that lists datasets -- reads as a bug.
    DROP_ON_DATASET_EXTS = tuple(dict.fromkeys(
        _formats.drop_on_dataset_extensions() + _formats.supported_extensions()))

    # ------------------------------------------------------------------
    # Processing metadata (``<stem>_viewer_metadata.json``)
    # ------------------------------------------------------------------
    def apply_metadata_to_dataset(self, idx: int, meta: dict, source: str) -> bool:
        'Apply a parsed recipe to dataset *idx* and bring the UI in line.'
        from . import file_open_controller as controller
        return controller.apply_metadata_to_dataset(self, idx, meta, source)

    def _remember_pending_metadata(self, path, meta: dict) -> None:
        'Hold one recipe for a dataset not yet loaded (points 3.1 / 4).'
        from . import file_open_controller as controller
        return controller._remember_pending_metadata(self, path, meta)

    def _take_pending_metadata(self, ds) -> tuple:
        'Consume the pending recipe for a just-added *ds*, or drop it.'
        from . import file_open_controller as controller
        return controller._take_pending_metadata(self, ds)

    def _sidecar_for_load(self, path) -> bool:
        'Whether the recipe beside *path* may be applied — asks if needed.'
        from . import file_open_controller as controller
        return controller._sidecar_for_load(self, path)

    def _open_metadata_sidecar(self, path: str) -> None:
        'A recipe dropped on the main window: ask which dataset it acts on.'
        from . import file_open_controller as controller
        return controller._open_metadata_sidecar(self, path)

    def drop_file_on_dataset(self, idx: int, path: str) -> bool:
        'Apply a dropped file to the dataset at *idx*.  Returns True if handled.'
        from . import file_open_controller as controller
        return controller.drop_file_on_dataset(self, idx, path)

    def _drop_filter_file(self, idx: int, path: Path) -> bool:
        "Append a dropped filter preset to the row's dataset."
        from . import file_open_controller as controller
        return controller._drop_filter_file(self, idx, path)

    def _drop_roi_file(self, idx: int, path: Path) -> bool:
        "Load a dropped ROI set into the ROI Manager, targeting the row's dataset."
        from . import file_open_controller as controller
        return controller._drop_roi_file(self, idx, path)

    def _drop_metadata_sidecar(self, idx: int, path: Path) -> bool:
        "Apply a dropped processing recipe to the row's dataset, once confirmed."
        from . import file_open_controller as controller
        return controller._drop_metadata_sidecar(self, idx, path)

    def _drop_confocal_tiff(self, idx: int, path: Path) -> bool:
        "Map a dropped TIFF onto the row's dataset as a confocal signal attribute."
        from . import file_open_controller as controller
        return controller._drop_confocal_tiff(self, idx, path)

    def _refresh_overlay_windows(self) -> None:
        'Rebuild channel lists and re-render open render/scatter overlay windows.'
        from . import window_controller as controller
        return controller._refresh_overlay_windows(self)

    def _close_all_datasets(self) -> None:
        from . import window_controller as controller
        return controller._close_all_datasets(self)

    def _close_all_windows(self) -> None:
        '``Ctrl+Shift+W`` — close **everything**: all datasets, their viewer'
        from . import window_controller as controller
        return controller._close_all_windows(self)

    def _duplicate_active_dataset(self) -> None:
        'Edit → Duplicate / ``Shift+D``.'
        from . import roi_command_controller as controller
        return controller._duplicate_active_dataset(self)

    # ------------------------------------------------------------------
    def _add_cross_section_to_roi(self) -> None:
        'Process › ROI › Add Cross-Section… — make a polyhedron multi-slice.'
        from . import roi_command_controller as controller
        return controller._add_cross_section_to_roi(self)

    def _active_region_record(self, ds, ds_idx):
        'The active **region** ROI (rectangle/oval/polygon/freehand) for *ds*.'
        from . import roi_command_controller as controller
        return controller._active_region_record(self, ds, ds_idx)

    def _copy_dataset(self, src, new_name: str):
        'Deep-copy a dataset as a fresh standalone (no overlay/source links).'
        from . import roi_command_controller as controller
        return controller._copy_dataset(self, src, new_name)

    def _plain_duplicate(self, src) -> None:
        from . import roi_command_controller as controller
        return controller._plain_duplicate(self, src)

    def _execute_crop(self, src_idx, src, record, members, opts) -> None:
        'Produce cropped duplicate(s) per the chosen options (Model A or B).'
        from . import roi_command_controller as controller
        return controller._execute_crop(self, src_idx, src, record, members, opts)

    def _carry_roi_to_cropped(self, record, new_idx: int) -> None:
        'Add a copy of the source ROI to the store, retargeted to the cropped'
        from . import roi_command_controller as controller
        return controller._carry_roi_to_cropped(self, record, new_idx)

    def _unique_name(self, prefix: str, base: str) -> str:
        '``<prefix><base>`` with numbering only when needed.'
        from . import roi_command_controller as controller
        return controller._unique_name(self, prefix, base)

    def _next_duplicate_name(self, source_name: str) -> str:
        'Return DUP_<source_name>, with numbering only when needed.'
        from . import roi_command_controller as controller
        return controller._next_duplicate_name(self, source_name)

    def _duplicate_dataset_for_overlay(self, src_idx: int, timestamp: str) -> int:
        from . import overlay_controller as controller
        return controller._duplicate_dataset_for_overlay(self, src_idx, timestamp)

    def _show_channel_combine(self, dataset_indices=None) -> None:
        from . import overlay_controller as controller
        return controller._show_channel_combine(self, dataset_indices)

    def _collapse_member_coordinate_views(self, member_indices, anchor_idx: int) -> bool:
        "Fold the members' standalone render/scatter windows into the anchor's."
        from . import overlay_controller as controller
        return controller._collapse_member_coordinate_views(self, member_indices, anchor_idx)

    def _show_channel_separation(self) -> None:
        'Process › Channel… › Separate Channel by DCR — open the DCR two-color'
        from . import overlay_controller as controller
        return controller._show_channel_separation(self)

    def _show_attribute_separation(self) -> None:
        'Process › Channel › Convert Dataset to Multi-Channel Overlay › by MINFLUX'
        from . import overlay_controller as controller
        return controller._show_attribute_separation(self)

    def _show_time_channel_separation(self) -> None:
        'Open the time-window editor for the active dataset.'
        from . import overlay_controller as controller
        return controller._show_time_channel_separation(self)

    def apply_time_channel_separation(self, source_idx: int, windows) -> bool:
        'Create filtered full-data clones and group them as a channel overlay.'
        from . import overlay_controller as controller
        return controller.apply_time_channel_separation(self, source_idx, windows)

    def apply_channel_separation(self, src_idx: int, labels, channels, *, attribute: str='', method_label: str='channel separation', masks=None, include_unassigned: bool=True, basis: str='') -> bool:
        'Build one truncated dataset per channel (+ a hidden *unassigned*) and'
        from . import overlay_controller as controller
        return controller.apply_channel_separation(self, src_idx, labels, channels, attribute=attribute, method_label=method_label, masks=masks, include_unassigned=include_unassigned, basis=basis)

    def _show_render(self, dataset_idx: int | None=None):
        'Open (or raise) the render window for a dataset.'
        from . import window_controller as controller
        return controller._show_render(self, dataset_idx)

    # ------------------------------------------------------------------
    # ParaView
    # ------------------------------------------------------------------

    def _save_data(self, *args) -> None:
        'File › Save (Ctrl+S) — save the active dataset as MINFLUX Viewer Zarr v2.'
        from . import save_export_controller as controller
        return controller._save_data(self, *args)

    def _zarr_save_suggestion(self, ds) -> Path:
        'Where File › Save proposes to write *ds*.'
        from . import save_export_controller as controller
        return controller._zarr_save_suggestion(self, ds)

    def _active_dataset_for_save(self):
        from . import save_export_controller as controller
        return controller._active_dataset_for_save(self)

    def _start_file_save(self, work, *, name: str, on_done) -> None:
        "Run a prepared writer without blocking Qt's event loop."
        from . import save_export_controller as controller
        return controller._start_file_save(self, work, name=name, on_done=on_done)

    def _on_file_save_failed(self, name: str, message: str) -> None:
        from . import save_export_controller as controller
        return controller._on_file_save_failed(self, name, message)

    def _default_save_path(self, ds, suffix: str) -> Path:
        from . import save_export_controller as controller
        return controller._default_save_path(self, ds, suffix)

    def _save_as_minflux_formats(self) -> None:
        from . import save_export_controller as controller
        return controller._save_as_minflux_formats(self)

    def _zarr_overwrite_mode(self, path: str | Path) -> str | None:
        'Ask how an existing application Zarr store should be updated.'
        from . import save_export_controller as controller
        return controller._zarr_overwrite_mode(self, path)

    def _save_as_format(self, fmt: str, title: str, *, path: str | None=None) -> None:
        from . import save_export_controller as controller
        return controller._save_as_format(self, fmt, title, path=path)

    def _on_zarr_saved(self, written, action: str, name: str) -> None:
        from . import save_export_controller as controller
        return controller._on_zarr_saved(self, written, action, name)

    def _on_files_saved(self, written, action: str, name: str, ds) -> None:
        from . import save_export_controller as controller
        return controller._on_files_saved(self, written, action, name, ds)

    def save_roi_records(self, ds, member_indices=None, index_to_id=None) -> list[dict]:
        'Every ROI displayed on *ds* right now, as portable dicts.'
        from . import save_export_controller as controller
        return controller.save_roi_records(self, ds, member_indices, index_to_id)

    def _overlay_members_for(self, ds) -> list:
        "Every channel of *ds*'s overlay, in order, or just *ds*."
        from . import save_export_controller as controller
        return controller._overlay_members_for(self, ds)

    def _zarr_save_context(self, ds) -> dict:
        'Overlay members, portable ROI geometry and linked MSR images.'
        from . import save_export_controller as controller
        return controller._zarr_save_context(self, ds)

    def _save_as_spreadsheet(self) -> None:
        from . import save_export_controller as controller
        return controller._save_as_spreadsheet(self)

    def export_custom_csv(self, ds, suggested: str | Path | None=None) -> None:
        'The custom column table: pick the attributes, name the columns.'
        from . import save_export_controller as controller
        return controller.export_custom_csv(self, ds, suggested)

    def save_metadata_only(self, ds) -> None:
        'Write just the processing recipe, to a path the user chooses.'
        from . import save_export_controller as controller
        return controller.save_metadata_only(self, ds)

    def _save_as_picasso_hdf5(self) -> None:
        from . import save_export_controller as controller
        return controller._save_as_picasso_hdf5(self)

    def _save_as_ome_tiff(self) -> None:
        from . import save_export_controller as controller
        return controller._save_as_ome_tiff(self)

    def save_dataset(self, ds) -> None:
        'Open the Save / Export dialog for *ds* (the active dataset from File ›'
        from . import save_export_controller as controller
        return controller.save_dataset(self, ds)

    # ------------------------------------------------------------------
    # Recognising a saved overlay again
    # ------------------------------------------------------------------
    def _schedule_overlay_regroup(self) -> None:
        'Ask about regrouping once the batch has settled.'
        from . import save_export_controller as controller
        return controller._schedule_overlay_regroup(self)

    def _offer_overlay_regroup(self) -> None:
        'Offer to rebuild an overlay whose channels are all loaded.'
        from . import save_export_controller as controller
        return controller._offer_overlay_regroup(self)

    def _regroup_saved_overlay(self, found) -> None:
        'Put the loaded members back into one overlay, in their saved order.'
        from . import save_export_controller as controller
        return controller._regroup_saved_overlay(self, found)

    def _save_overlay_group(self, members, opts) -> None:
        'Write every channel of an overlay into one folder, off the UI thread.'
        from . import save_export_controller as controller
        return controller._save_overlay_group(self, members, opts)

    def _on_group_saved(self, written, folder, channels: int) -> None:
        from . import save_export_controller as controller
        return controller._on_group_saved(self, written, folder, channels)

    def _on_processed_saved(self, written, action: str, name: str, ds) -> None:
        from . import save_export_controller as controller
        return controller._on_processed_saved(self, written, action, name, ds)

    # ------------------------------------------------------------------
    # AppState signal handlers
    # ------------------------------------------------------------------

    def _maybe_offer_regroup(self, idx: int) -> None:
        from . import save_export_controller as controller
        return controller._maybe_offer_regroup(self, idx)

    def _on_dataset_added(self, event) -> None:
        from . import post_load_controller as controller
        return controller._on_dataset_added(self, event)

    def _schedule_post_load_computations(self, idx: int) -> None:
        'Run optional one-time computed attributes after initial windows show.'
        from . import post_load_controller as controller
        return controller._schedule_post_load_computations(self, idx)

    def _run_post_load_for_key(self, key) -> None:
        'Start post-load for the dataset with *key*, wherever it is now.'
        from . import post_load_controller as controller
        return controller._run_post_load_for_key(self, key)

    def _run_post_load_computations(self, idx: int) -> None:
        "Compute derived attributes in a worker and apply them on Qt's thread."
        from . import post_load_controller as controller
        return controller._run_post_load_computations(self, idx)

    def _finish_post_load_task(self, key, task) -> None:
        from . import post_load_controller as controller
        return controller._finish_post_load_task(self, key, task)

    def _retire_dataset_tasks(self, event) -> None:
        'Cancel and detach the post-load work of a dataset that has closed.'
        from . import post_load_controller as controller
        return controller._retire_dataset_tasks(self, event)

    def _on_post_load_done(self, key, ds, result: dict) -> None:
        'Deliver a post-load result only while the dataset it started for is open.'
        from . import post_load_controller as controller
        return controller._on_post_load_done(self, key, ds, result)

    def _on_post_load_error(self, key, ds, message: str) -> None:
        from . import post_load_controller as controller
        return controller._on_post_load_error(self, key, ds, message)

    def _on_post_load_failed(self, ds, message: str) -> None:
        from . import post_load_controller as controller
        return controller._on_post_load_failed(self, ds, message)

    def _apply_post_load_results(self, ds, result: dict) -> None:
        from . import post_load_controller as controller
        return controller._apply_post_load_results(self, ds, result)

    def _start_saved_filter_restore(self, ds) -> None:
        from . import post_load_controller as controller
        return controller._start_saved_filter_restore(self, ds)

    def _on_saved_filters_done(self, key, ds, mask) -> None:
        from . import post_load_controller as controller
        return controller._on_saved_filters_done(self, key, ds, mask)

    def _on_saved_filters_error(self, key, ds, message: str) -> None:
        from . import post_load_controller as controller
        return controller._on_saved_filters_error(self, key, ds, message)

    def _on_saved_filters_evaluated(self, ds, mask) -> None:
        from . import post_load_controller as controller
        return controller._on_saved_filters_evaluated(self, ds, mask)

    def _on_saved_filter_failed(self, ds, message: str) -> None:
        from . import post_load_controller as controller
        return controller._on_saved_filter_failed(self, ds, message)

    def _post_load_index(self, ds) -> int | None:
        'Current index of *ds* (through its key), or None if it was removed.'
        from . import post_load_controller as controller
        return controller._post_load_index(self, ds)

    def _post_load_next(self, ds, step) -> None:
        'Run the next post-load step from the event loop (keeps the UI live).'
        from . import post_load_controller as controller
        return controller._post_load_next(self, ds, step)

    def _restart_post_load_in_background(self, ds) -> None:
        'Compatibility entry point for extensions calling the former steps.'
        from . import post_load_controller as controller
        return controller._restart_post_load_in_background(self, ds)

    def _post_load_z_scaling_factor(self, ds) -> None:
        from . import post_load_controller as controller
        return controller._post_load_z_scaling_factor(self, ds)

    def _legacy_post_load_z_scaling_factor(self, ds) -> None:
        from . import post_load_controller as controller
        return controller._legacy_post_load_z_scaling_factor(self, ds)

    def _post_load_z_scaling_factor_estimate(self, ds) -> None:
        from . import post_load_controller as controller
        return controller._post_load_z_scaling_factor_estimate(self, ds)

    def _legacy_post_load_z_scaling_factor_estimate(self, ds) -> None:
        from . import post_load_controller as controller
        return controller._legacy_post_load_z_scaling_factor_estimate(self, ds)

    def _post_load_loc_prec(self, ds) -> None:
        from . import post_load_controller as controller
        return controller._post_load_loc_prec(self, ds)

    def _legacy_post_load_loc_prec(self, ds) -> None:
        from . import post_load_controller as controller
        return controller._legacy_post_load_loc_prec(self, ds)

    def _post_load_density(self, ds) -> None:
        from . import post_load_controller as controller
        return controller._post_load_density(self, ds)

    def _legacy_post_load_density(self, ds) -> None:
        from . import post_load_controller as controller
        return controller._legacy_post_load_density(self, ds)

    def _post_load_finalize(self, ds) -> None:
        from . import post_load_controller as controller
        return controller._post_load_finalize(self, ds)

    def _restore_saved_rois(self, ds, idx: int) -> None:
        "Put a reopened dataset's saved ROIs into the ROI Manager."
        from . import post_load_controller as controller
        return controller._restore_saved_rois(self, ds, idx)

    def _show_saved_filters(self, ds, idx: int) -> None:
        'Open the Filter dialog on a dataset that was saved with filters.'
        from . import post_load_controller as controller
        return controller._show_saved_filters(self, ds, idx)

    def _on_dataset_removed(self, event) -> None:
        'Retire what belonged to a dataset that has closed (typed removal).'
        from . import window_controller as controller
        return controller._on_dataset_removed(self, event)

    def _unregister_when_destroyed(self, registry: dict | DatasetWindowMap, window) -> None:
        'Remove *window* from *registry* when Qt destroys it -- by identity.'
        from . import window_controller as controller
        return controller._unregister_when_destroyed(self, registry, window)

    def _on_active_changed(self, event) -> None:
        from . import window_controller as controller
        return controller._on_active_changed(self, event)

    # ------------------------------------------------------------------
    # Drag-and-drop  (files AND folders; any supported extension)
    # ------------------------------------------------------------------

    def dragEnterEvent(self, event: QDragEnterEvent) -> None:
        if event.mimeData().hasUrls():
            # Accept every local path, not only a known suffix. The router can
            # content-sniff mislabeled files and, importantly, reports an
            # unsupported/unreadable drop instead of making the gesture vanish.
            for url in event.mimeData().urls():
                if url.toLocalFile():
                    event.acceptProposedAction()
                    return
        event.ignore()

    def dragMoveEvent(self, event) -> None:
        # Must accept dragMoveEvent too, or Qt cancels the drop mid-drag
        if event.mimeData().hasUrls():
            for url in event.mimeData().urls():
                if url.toLocalFile():
                    event.acceptProposedAction()
                    return
        event.ignore()

    def dropEvent(self, event: QDropEvent) -> None:
        paths = []
        for url in event.mimeData().urls():
            path = url.toLocalFile()
            if path:
                paths.append(path)
        if not paths:
            event.ignore()
            return
        self._record_file_drop(paths)
        self.route_paths(paths)
        event.acceptProposedAction()

    def _record_file_drop(self, paths) -> None:
        'Record the OS drop attempt before parsing/routing can fail.'
        from . import window_controller as controller
        return controller._record_file_drop(self, paths)

    def route_paths(self, paths) -> None:
        'Open a whole batch of paths — a multi-file drop, or the command line.'
        from . import window_controller as controller
        return controller.route_paths(self, paths)

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    # ------------------------------------------------------------------
    # LUT / Color / Show Info / Localization Precision / Toolbar icons
    # ------------------------------------------------------------------

    def _on_focus_changed(self, _old, new) -> None:
        'Record the last render/scatter window the user focused (for the LUT'
        from . import window_controller as controller
        return controller._on_focus_changed(self, _old, new)

    @staticmethod
    def _lut_view_shows_dataset(view, idx: int | None) -> bool:
        'True when *view* displays dataset *idx* (directly or as an overlay'
        from . import window_controller as controller
        return controller._lut_view_shows_dataset(view, idx)

    def _retarget_lut_dialog(self, view) -> None:
        'Move the one open LUT dialog onto the newly focused *view*.'
        from . import window_controller as controller
        return controller._retarget_lut_dialog(self, view)

    def _lut_capable_views(self) -> list:
        'Every view that can own a LUT dialog.'
        from . import window_controller as controller
        return controller._lut_capable_views(self)

    def _close_other_lut_dialogs(self, keep) -> None:
        'No-op kept for callers: there is only ever one LUT dialog now.'
        from . import window_controller as controller
        return controller._close_other_lut_dialogs(self, keep)

    def _open_lut_on_view(self, view, idx: int | None) -> bool:
        "Open *view*'s LUT dialog if it is a live, visible LUT-capable view of"
        from . import window_controller as controller
        return controller._open_lut_on_view(self, view, idx)

    def _image_viewer_for_lut(self, active) -> object | None:
        'The image viewer the LUT button should act on, or ``None``.'
        from . import window_controller as controller
        return controller._image_viewer_for_lut(self, active)

    def _show_lut(self) -> None:
        'Toolbar LUT button — open the LUT / colormap editor for the active'
        from . import window_controller as controller
        return controller._show_lut(self)

    def _show_color_picker(self) -> None:
        'Open or raise the one modeless application-wide COLOR editor.'
        from . import window_controller as controller
        return controller._show_color_picker(self)

    def _show_info_for_active(self) -> None:
        'View → Show Info — open data-info window(s) for the active dataset.'
        from . import window_controller as controller
        return controller._show_info_for_active(self)

    def _info_indices_for_active(self, active_idx: int) -> list[int]:
        'Show all members of an imported multi-channel MSR group together.'
        from . import window_controller as controller
        return controller._info_indices_for_active(self, active_idx)

    def _loc_precision_frc(self) -> None:
        from . import window_controller as controller
        return controller._loc_precision_frc(self)

    def _loc_precision_crlb(self) -> None:
        from . import window_controller as controller
        return controller._loc_precision_crlb(self)

    def _loc_precision_stddev(self) -> None:
        from . import window_controller as controller
        return controller._loc_precision_stddev(self)

    def _run_local_density(self) -> None:
        from . import window_controller as controller
        return controller._run_local_density(self)

    # ------------------------------------------------------------------
    # Trace analysis handlers
    # ------------------------------------------------------------------

    def _run_trace_size(self) -> None:
        from . import window_controller as controller
        return controller._run_trace_size(self)

    def _run_trace_anisotropy(self) -> None:
        from . import window_controller as controller
        return controller._run_trace_anisotropy(self)

    def _install_toolbar_icons(self) -> None:
        'Attach PNG icons to the toolbar QActions (item #5).'
        from . import menu_controller as controller
        return controller._install_toolbar_icons(self)

    def _no_data_warning(self) -> None:
        from . import menu_controller as controller
        return controller._no_data_warning(self)

    def _placeholder(self, feature: str, phase: str) -> None:
        from . import menu_controller as controller
        return controller._placeholder(self, feature, phase)

    def _check_for_updates(self, *, silent: bool=False) -> None:
        'Tier-A update check: query GitHub Releases off-thread, then report.'
        from . import menu_controller as controller
        return controller._check_for_updates(self, silent=silent)

    def _maybe_startup_update_check(self) -> None:
        from . import menu_controller as controller
        return controller._maybe_startup_update_check(self)

    def _on_update_check_done(self, result, silent: bool, task=None) -> None:
        from . import menu_controller as controller
        return controller._on_update_check_done(self, result, silent, task)

    def _show_about(self) -> None:
        from . import menu_controller as controller
        return controller._show_about(self)

    def _setup_toolbar_widgets(self) -> None:
        'Add a leading spacer (to shift the tools right so the first ROI button'
        from . import menu_controller as controller
        return controller._setup_toolbar_widgets(self)

    def _setup_command_search(self) -> None:
        'Wire the toolbar search field: a live completer of menu commands + a'
        from . import menu_controller as controller
        return controller._setup_command_search(self)

    def _apply_command_meta(self) -> None:
        'Stamp every menu command with its metadata from ``ui.command_meta`` so the'
        from . import menu_controller as controller
        return controller._apply_command_meta(self)

    def _connect_command_recorder(self, action, key: str, meta) -> None:
        'Connect one QAction to the recorder exactly once.'
        from . import menu_controller as controller
        return controller._connect_command_recorder(self, action, key, meta)

    def _record_command_action(self, action, key: str, meta) -> None:
        from . import menu_controller as controller
        return controller._record_command_action(self, action, key, meta)

    def _refresh_command_index(self) -> None:
        from . import menu_controller as controller
        return controller._refresh_command_index(self)

    def _on_search_activated(self, text: str) -> None:
        from . import menu_controller as controller
        return controller._on_search_activated(self, text)

    def _on_search_return(self) -> None:
        from . import menu_controller as controller
        return controller._on_search_return(self)

    def _on_command_finder_destroyed(self, *_args) -> None:
        from . import menu_controller as controller
        return controller._on_command_finder_destroyed(self, *_args)

    def _command_finder_alive(self) -> bool:
        from . import menu_controller as controller
        return controller._command_finder_alive(self)

    def _open_command_finder(self, initial: str='') -> None:
        from . import menu_controller as controller
        return controller._open_command_finder(self, initial)

    def _align_toolbar_to_menu(self) -> None:
        "Size the leading spacer so the first ROI button's left edge roughly"
        from . import menu_controller as controller
        return controller._align_toolbar_to_menu(self)

    def _apply_menu_separator_style(self) -> None:
        'Make menu **section separators** clearly visible. The native 1-px etch'
        from . import menu_controller as controller
        return controller._apply_menu_separator_style(self)

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
        # A pending regroup question would fire into a closing window.
        regroup = getattr(self, "_regroup_timer", None)
        if regroup is not None:
            regroup.stop()
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
        'Detach background work without ever blocking the GUI close path.'
        from . import window_controller as controller
        return controller._drain_background_tasks(self)

    def _close_all_child_windows(self, *, keep_log_console: bool=False) -> None:
        'Close every floating / tool window this main window has created.'
        from . import window_controller as controller
        return controller._close_all_child_windows(self, keep_log_console=keep_log_console)

    def _terminate_paraview_processes(self) -> None:
        'Politely stop every ParaView subprocess spawned during this session.'
        from . import window_controller as controller
        return controller._terminate_paraview_processes(self)


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

    def dragEnterEvent(self, event: QDragEnterEvent) -> None:  # type: ignore[override]
        if self.parent() is not None:
            self.parent().dragEnterEvent(event)
        else:
            event.ignore()

    def dragMoveEvent(self, event) -> None:  # type: ignore[override]
        if self.parent() is not None:
            self.parent().dragMoveEvent(event)
        else:
            event.ignore()

    def dropEvent(self, event: QDropEvent) -> None:  # type: ignore[override]
        if self.parent() is not None:
            self.parent().dropEvent(event)
        else:
            event.ignore()
