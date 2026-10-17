"""Menu controller delegated by MainWindow."""

from __future__ import annotations

import weakref
from pathlib import Path
from typing import TYPE_CHECKING

from PyQt6.QtCore import QPoint, Qt
from PyQt6.QtGui import (
    QAction,
    QActionGroup,
    QKeySequence,
    QShortcut,
)
from PyQt6.QtWidgets import (
    QApplication,
    QComboBox,
    QDoubleSpinBox,
    QKeySequenceEdit,
    QLineEdit,
    QMenu,
    QMessageBox,
    QPlainTextEdit,
    QSizePolicy,
    QSpinBox,
    QTextEdit,
    QWidget,
)

from .dataset_binding import window_dataset_index
from .window_support import _ROI_TOOL_DEFS, _adaptive_toolbar_icon, _UpdateCheckTask

if TYPE_CHECKING:
    from .main_window import MainWindow


def _connect_actions(self: MainWindow) -> None:
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
    self.actionSaveAsMsr = QAction("Imspector file (.msr)", self)
    self.actionSaveAsMsr.setToolTip(
        "Save as an Abberior .msr.\n\n"
            "When this data came from a .msr, that file's container is reused, "
            "so the result opens in Imspector and carries this application's "
            "processing inside it.\n\n"
            "Data with no .msr origin is still written, but only this "
            "application can open it — the Log says which was produced."
    )
    self.actionSaveAsMsr.triggered.connect(
        lambda _checked=False: self._save_as_format("msr", "MINFLUX .msr file")
    )
    self.actionSaveAsMetadata = QAction("Viewer metadata only (.json)...", self)
    self.actionSaveAsMetadata.setToolTip(
        "Write only the processing metadata: Z scaling factor, transform, "
            "filters, ROIs and acquisition time.\n\n"
            "Useful to inspect what the current processing amounts to before "
            "committing it to a data format, or to get hold of it when the "
            "format in use keeps it inside the file (.zarr and .msr do)."
    )
    self.actionSaveAsMetadata.triggered.connect(
        lambda _checked=False: self.save_metadata_only(
            self._active_dataset_for_save()))
    self.actionSaveAsSpreadsheet = QAction("Custom table (.csv)...", self)
    self.actionSaveAsSpreadsheet.setToolTip(
        "Choose which attributes to write and what to call each column. "
            "The table is the current view: filtered, last valid iteration, "
            "with xnm/ynm/znm as displayed.\n\n"
            "The all-iteration canonical table (which reloads without the "
            "column-mapping dialog) is written by the MSR reader's .csv export "
            "and by Save / export data > More options."
    )
    self.actionSaveAsSpreadsheet.triggered.connect(self._save_as_spreadsheet)
    self.actionSaveAsZarr = QAction("Viewer format (.zarr)", self)
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
    # format plus the MINFLUX defaults.
    self.actionSaveAsHdf5 = QAction("HDF5...", self)
    self.actionSaveAsHdf5.triggered.connect(self._save_as_picasso_hdf5)
    self.actionSaveAsOmeTiff = QAction("OME-TIFF...", self)
    self.actionSaveAsOmeTiff.triggered.connect(self._save_as_ome_tiff)
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
    # Trajectories as comets over a de-emphasised structure. Its own viewer
    # rather than a mode of Render or Scatter: it answers *when*, and it
    # needs the structure to step back to a context layer for that to read,
    # which is not something to do to a view opened to look at localizations.
    self.actionTrackingView = QAction("Tracking View", self)
    self.actionTrackingView.setStatusTip(
        "Play localization trajectories as comets over the structure")
    self.actionTrackingView.triggered.connect(self._show_tracking_view)
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
    # The live HlyB/D workflow is offered by its built-in plugin.
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
    # Tracking analysis
    # MINFLUX already supplies authoritative ``tid`` values. A generic
    # camera-style linker would overwrite that acquisition fact, while
    # merge/split repair needs experiment-specific priors. Retire the old
    # placeholder; repair algorithms belong behind the tracking method /
    # plugin contracts when a validated model is available.
    u.menuTracking.removeAction(u.actionParticleTracking)
    u.actionParticleTracking.setVisible(False)
    # actionTraceViewer (legacy View/Tracking entry) retired. The retained
    # trace inspection workflow lives in Plugins > Trace Viewer.
    u.actionMsdAnalysis.triggered.connect(self._show_msd_analysis)
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
    # Legacy contour-stack polyhedron: draw another level at Z and
    # interpolate between them. Projection hulls edit their panes directly.
    self.actionRoiAddSlice = QAction("Add Cross-Section…", self)
    self.actionRoiAddSlice.setToolTip(
        "Add the drawn polygon to a legacy contour-stack polyhedron at "
            "another Z (projection-hull polygons are edited in their panes)")
    self.actionRoiAddSlice.triggered.connect(self._add_cross_section_to_roi)
    self.menuProcessRoi.addAction(self.actionRoiAddSlice)
    self.menuProcessRoi.addSeparator()
    # Restore ROI: bring the active ROI back on another view / after a delete.
    self.actionRoiRestore = QAction("Restore ROI", self)
    self.actionRoiRestore.triggered.connect(self._restore_roi)
    self.menuProcessRoi.addAction(self.actionRoiRestore)
    # Aggregate localizations (Imspector-style per-trace photon binning).
    self.actionAggregate = QAction("Aggregate Localizations…", self)
    self.actionAggregate.triggered.connect(self._aggregate_active_dataset)
    # Help menu
    u.actionAbout.triggered.connect(self._show_about)
    self.actionCommandFinder = QAction("Command Finder…", self)
    self.actionCommandFinder.setStatusTip("Search all menu commands (Fiji-style)")
    self.actionCommandFinder.triggered.connect(lambda: self._open_command_finder(""))
    self.actionTaskMonitor = QAction("Task Monitor", self)
    self.actionTaskMonitor.setStatusTip(
        "Background tasks, threads and memory; ask a task to stop")
    self.actionTaskMonitor.triggered.connect(self._show_task_monitor)
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
    self._point_variant = "point"
    self._lasso_variant = "magnetic_lasso"
    _variant_getters = {
        "line": lambda: self._line_variant,
        "rectangle": lambda: self._rect_variant,
        "oval": lambda: self._oval_variant,
        "polygon": lambda: self._poly_variant,
        "point": lambda: self._point_variant,
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
    # ⚠ Through the variant getter, exactly like the .ui family buttons: wired
    # to the literal "magnetic_lasso" instead, a left-click on the shared
    # button re-activated the lasso and silently discarded a chosen Magic
    # Wand, so the variant would not stay put.
    self.toolMagneticLasso.triggered.connect(
        lambda checked: self._on_roi_tool(self._lasso_variant, checked))
    self._state.rois.tool_changed.connect(self._sync_roi_tool_actions)
    # ⚠ A separate connection, deliberately NOT folded into
    # _sync_roi_tool_actions: that method is also called directly to re-sync
    # the buttons after a refusal (see _volume_tool_allowed), which has just
    # put its reason in the status bar -- announcing the tool there too would
    # overwrite the explanation with an unrelated line.
    self._state.rois.tool_changed.connect(self._announce_roi_tool)
    self._roi_tool_status_text = ""
    # Wire icon images for toolbar buttons
    self._install_toolbar_icons()
    self._install_roi_tool_menus()
    self._configure_menus()
    self._mark_ai_unapproved_actions()
    self._apply_shortcuts()


def _configure_menus(self: MainWindow) -> None:
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
    u.actionLocPrecisionFrc.setText("FRC (Fourier Ring Correlation)")
    u.actionLocPrecisionCrlb.setText("CRLB (Cramer-Rao Lower Bound)")
    u.actionLocPrecisionStdDev.setText("StdDev per Trace")
    u.actionLocalDensity.setText("Local Density")
    u.actionParticleTracking.setText("Particle Tracking")  # retired; not in menu
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
    self.actionSaveAsMsr.setText("Imspector file (.msr)")
    self.actionSaveAsSpreadsheet.setText("Custom table (.csv)...")
    self.actionSaveAsMetadata.setText("Viewer metadata only (.json)...")
    self.actionSaveAsZarr.setText("Viewer format (.zarr)")
    self.actionSaveAsZarrZip.setText("Zarr (.zarr.zip v2) single file")
    self.actionSaveAsHdf5.setText("HDF5...")
    self.actionSaveAsOmeTiff.setText("OME-TIFF...")
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
    # Order = the format registry's own order (formats.offered_save_formats).
    # actionSaveAsZarrZip / actionSaveAsHdf5 / actionSaveAsOmeTiff are built
    # and still work, but are deliberately NOT listed: one Zarr form is
    # enough in the menus, Picasso HDF5 is application-specific, and the
    # OME-TIFF entry only forwarded to the render view's own
    # "Export to TIFF...", which is where that export belongs.
    self.menuSaveAs.clear()
    self.menuSaveAs.addAction(self.actionSaveAsZarr)
    self.menuSaveAs.addAction(self.actionSaveAsMinflux)
    self.menuSaveAs.addAction(self.actionSaveAsSpreadsheet)
    self.menuSaveAs.addAction(self.actionSaveAsMsr)
    self.menuSaveAs.addSeparator()
    # Not a data format, so it sits below the separator: it saves the
    # processing *about* a dataset, not the dataset.
    self.menuSaveAs.addAction(self.actionSaveAsMetadata)
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
    u.menuView.addAction(u.actionHistogram)
    u.menuView.addAction(u.actionScatter)
    u.menuView.addAction(u.actionRender)
    u.menuView.addAction(self.actionTrackingView)
    u.menuView.addSeparator()
    u.menuView.addAction(u.actionLog)
    u.menuView.setToolTipsVisible(True)
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
    task_monitor = getattr(self, "actionTaskMonitor", None)
    if task_monitor is not None:      # memory is its third tab, not an entry
        u.menuHelp.addAction(task_monitor)
    cf = getattr(self, "actionCommandFinder", None)
    if cf is not None:                              # between Tasks & Updates
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
        "tracking_view": self.actionTrackingView,
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


def _keep_standard_actions_in_declared_menus(self: MainWindow) -> None:
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


def _mark_ai_unapproved_actions(self: MainWindow) -> None:
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
        #u.menuTracking.menuAction(),
        #u.actionParticleTracking,
        #u.actionMsdAnalysis,
    ]
    for action in actions:
        self._mark_action_ai_unapproved(action)


def _mark_action_ai_unapproved(self: MainWindow, action: QAction) -> None:
    """Flag an AI-generated/unapproved action with a status-bar hover note.

        The action's appearance is left untouched — approved and unapproved
        menu entries render identically; only the status-bar tip differs.
        """
    tip = action.statusTip() or action.toolTip()
    note = "AI-generated; pending human approval."
    action.setStatusTip(f"{tip} {note}".strip() if tip else note)
    self.addAction(action)


def _apply_shortcuts(self: MainWindow) -> None:
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


def _shortcut_command_keys(self: MainWindow, *, include_action_commands: bool = True) -> tuple[str, ...]:
    keys = (
        "next_window", "previous_window",
        "next_dataset", "previous_dataset", "close_window",
    )
    if include_action_commands:
        keys = (*keys, *self._shortcut_actions.keys())
    return keys


def _install_window_shortcuts(
    self: MainWindow,
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


def _refresh_owned_window_shortcuts(self: MainWindow) -> None:
    for widget in QApplication.topLevelWidgets():
        try:
            if widget is self:
                self._install_window_shortcuts(
                    widget, include_action_commands=False
                )
            elif getattr(widget, "TAG", None) in {
                "render_window", "attribute_window", "histogram_window",
                "scatter_window",
            }:
                self._install_window_shortcuts(widget)
        except RuntimeError:
            # topLevelWidgets can briefly contain a Python wrapper whose
            # deferred C++ deletion was processed earlier in this turn.
            continue


def _trigger_shortcut_command(self: MainWindow, command: str) -> None:
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


def _shortcut_text(self: MainWindow, key: str) -> str:
    return str(self._state.prefs.get("shortcuts", {}).get(key, "") or "")


def _event_sequence_text(self: MainWindow, event) -> str:
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


def _handle_shortcut_keypress(self: MainWindow, seq: str, event) -> bool:
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


def _shortcut_command_for_sequence(self: MainWindow, seq: str) -> str | None:
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


def _shortcut_conflicts(self: MainWindow) -> dict[str, list[str]]:
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


def _focus_is_shortcut_editor() -> bool:
    focus = QApplication.focusWidget()
    while focus is not None:
        if isinstance(focus, QKeySequenceEdit):
            return True
        focus = focus.parentWidget()
    return False


def _shortcut_allowed_in_editing(self: MainWindow, command: str) -> bool:
    active = QApplication.activeWindow()
    tag = getattr(active, "TAG", None)
    if command == "filter":
        return tag in {"histogram_window", "attribute_window"}
    if command == "brightness_contrast":
        return tag == "render_window"
    return False


def _set_active_from_focused_dataset_window(self: MainWindow) -> None:
    """Use the focused dataset-owned plot/view when a shortcut targets data."""
    focus = QApplication.focusWidget()
    candidates: list[QWidget | None] = [QApplication.activeWindow(), focus]
    if focus is not None:
        parent = focus.parentWidget()
        while parent is not None:
            candidates.append(parent)
            parent = parent.parentWidget()
    for widget in candidates:
        idx = window_dataset_index(widget)
        if idx is not None and 0 <= idx < len(self._state.datasets):
            self._state.set_active(idx)
            return


def _mark_recent_menu_stale(self: MainWindow) -> None:
    "The recent-file list changed; rebuild the menu when it is next shown."
    self._recent_menu_stale = True


def _refresh_recent_menu_if_stale(self: MainWindow) -> None:
    if getattr(self, "_recent_menu_stale", False):
        self._populate_recent_menu()


def _populate_recent_menu(self: MainWindow) -> None:
    self._recent_menu_stale = False
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


def _show_recent_file_menu(self: MainWindow, path: str, global_pos) -> None:
    """Right-click context menu for a File › Open Recent entry."""
    menu = QMenu(self)
    act_open = menu.addAction("Open")
    act_loc = menu.addAction("Open file location")
    chosen = menu.exec(global_pos)
    if chosen is act_open:
        self._route_file(path)
    elif chosen is act_loc:
        self.open_file_location(path)


def open_file_location(self: MainWindow, path: str) -> None:
    """Reveal a file in the OS file manager (falls back to opening its
        containing folder). Cross-platform; see :mod:`.file_manager`."""
    from .file_manager import reveal_in_file_manager
    reveal_in_file_manager(path)


def _populate_plugins_menu(self: MainWindow) -> None:
    """
        Fill the Plugins menu from the plugin registry.

        Built-in plugins register themselves at import time; third-party
        plugins that have been imported before this method runs appear here
        too, in insertion order.
        """
    from .. import plugins
    from ..core import user_libs
    # External library folders must be on sys.path BEFORE any plugin is
    # imported, so a plugin can rely on them at import time. Appended,
    # never prepended -- see core/user_libs.py for the measured reason.
    try:
        added = user_libs.install_paths(self._state.prefs)
    except Exception as exc:                      # never block startup
        self._state.log(f"Could not add external library folders: {exc}", "WARN")
    else:
        if added:
            self._state.log(
                f"Added {len(added)} external library folder(s) to the Python path.",
                "INFO",
            )
    plugins.ensure_loaded()
    # User plugins from folders outside the package. A failure here must
    # never stop the built-in plugins from reaching the menu.
    try:
        plugins.discover(self._state.prefs)
    except Exception as exc:
        self._state.log(f"Plugin discovery failed: {exc}", "WARN")
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
        act.setProperty("command_source", entry.source or source_of(entry.launch))
        if getattr(entry, "keywords", ()):
            act.setProperty("command_keywords", " ".join(entry.keywords))
        # Capture `entry` per-iteration with a default argument so the
        # lambda doesn't all point at the last one.
        act.triggered.connect(
            lambda _=False, e=entry: e.launch(self._state, self)
        )
        if entry.name == "Generate Method Text":
            self._mark_action_ai_unapproved(act)
        if entry.error:
            act.setEnabled(False)
        # Reuse an existing submenu at every level so independently
        # discovered plugins can share paths such as Plugins > HlyB/D.
        target = menu
        for level in entry.menu_path:
            existing = next(
                (
                    action.menu()
                    for action in target.actions()
                    if action.menu() is not None
                    and action.menu().title() == level
                ),
                None,
            )
            target = existing or target.addMenu(level)
        target.addAction(act)


def _show_preferences(self: MainWindow) -> None:
    """Open the modeless Preferences dialog (Edit → Preferences…).

        Modeless so a preference can be chosen while looking at the plot or
        dataset it applies to. That is only safe because ``_accept`` merges the
        edited leaves into the live preferences instead of assigning its opening
        snapshot wholesale — see ``core/app_state.py::merge_pref_changes``.

        Single-instance: two dialogs would each snapshot, and the second OK
        would decide every field the first had just set.
        """
    from .modeless import show_modeless
    from .preferences_dialog import PreferencesDialog
    from .qt_lifecycle import qobject_alive
    existing = getattr(self, "_prefs_dialog", None)
    if existing is not None and qobject_alive(existing):
        existing.show()
        existing.raise_()
        existing.activateWindow()
        return
    # Non-owned, per the modeless window convention: a parented top-level
    # is an *owned* window the OS pins above its owner.
    dlg = PreferencesDialog(self._state)
    dlg.accepted.connect(self._on_preferences_accepted)
    dlg.destroyed.connect(
        lambda _=None: setattr(self, "_prefs_dialog", None)
    )
    self._prefs_dialog = dlg
    show_modeless(dlg, self)


def _on_preferences_accepted(self: MainWindow) -> None:
    """Rebuild the UI bits that depend on preferences, after OK."""
    if self._is_shutting_down:
        return
    self._apply_shortcuts()
    self._mark_recent_menu_stale()
    self._refresh_plot_preferences()


def _refresh_plot_preferences(self: MainWindow) -> None:
    windows = (
        list(self._render_windows.values())
        + list(self._scatter_windows.values())
        + list(self._tracking_windows.values())
        + list(self._msd_windows.values())
        + list(self._histogram_windows.values())
        + list(self._attr_windows.values())
    )
    for win in windows:
        refresh = getattr(win, "refresh_preferences", None)
        if callable(refresh):
            refresh()


def _on_colormap_order_changed(self: MainWindow) -> None:
    """Rebuild the colormap selectors that are built once and kept.

        A menu is rebuilt every time it opens and needs nothing; a ``QComboBox``
        populated in a window's constructor does.
        """
    for registry in (self._scatter_windows, self._render_windows,
                     self._attr_windows):
        for window in list(registry.values()):
            refresh = getattr(window, "refresh_colormap_list", None)
            if callable(refresh):
                try:
                    refresh()
                except RuntimeError:
                    continue


def _on_colors_changed(self: MainWindow, payload) -> None:
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


def _install_toolbar_icons(self: MainWindow) -> None:
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


def _no_data_warning(self: MainWindow) -> None:        QMessageBox.information(self, "No data", "Please load a dataset first.")


def _placeholder(self: MainWindow, feature: str, phase: str) -> None:
    QMessageBox.information(
        self, feature,
        f"<b>{feature}</b> will be implemented in {phase}.",
    )


def _check_for_updates(self: MainWindow, *, silent: bool = False) -> None:
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
    from .background_tasks import shared_thread_pool
    shared_thread_pool("update-check").start(task)


def _maybe_startup_update_check(self: MainWindow) -> None:
    if getattr(self, "_is_shutting_down", False):
        return
    self._check_for_updates(silent=True)


def _on_update_check_done(self: MainWindow, result, silent: bool, task=None) -> None:
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


def _show_about(self: MainWindow) -> None:
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


def _setup_toolbar_widgets(self: MainWindow) -> None:
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


def _setup_command_search(self: MainWindow) -> None:
    """Wire the toolbar search field: a live completer of menu commands + a
        double-click full Command Finder dialog (Fiji-style)."""
    from PyQt6.QtWidgets import QCompleter

    from .command_finder import WideCompleterPopup
    self._command_entries = []
    self._command_finder = None
    self._search_display_map = {}
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


def _apply_command_meta(self: MainWindow) -> None:
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
        if meta.gui_class is not None:
            act.setProperty("command_gui_class", meta.gui_class.value)
        if meta.record:
            act.setProperty("command_record", meta.record)
        self._connect_command_recorder(act, key, meta)
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
            if grp.gui_class is not None:
                act.setProperty("command_gui_class", grp.gui_class.value)
            if grp.record:
                act.setProperty("command_record", grp.record)
            label = str(act.text()).replace("&", "").strip()
            self._connect_command_recorder(act, f"{group_key}:{label}", grp)


def _connect_command_recorder(self: MainWindow, action, key: str, meta) -> None:
    """Connect one QAction to the recorder exactly once.

        The connection is intentionally inert while recording is disabled.
        Command handlers will progressively add accepted parameters/runnable
        calls; until then the identity event renders as an honest TODO.

        ⚠ The slot holds **weak** references to this window and to the action.
        A lambda capturing ``self`` and ``action`` strongly builds the cycle
        ``window → action (its child) → connection → lambda → window``, on every
        one of the ~68 command actions. Python's GC then breaks those cycles at
        an arbitrary later moment, and collecting a QObject graph while Qt is
        delivering queued events is the documented native-abort hazard in this
        project -- Track C hit exactly this shape with its destroyed-signal
        callbacks. Measured here: the strong form passed every test in
        isolation and reproducibly aborted the mixed suite inside an unrelated
        LUT-dialog test.
        """
    if bool(action.property("_mfv_macro_recorder_connected")):
        return
    window_ref = weakref.ref(self)
    action_ref = weakref.ref(action)
    def _record(_checked: bool = False, k: str = key, m=meta) -> None:
        window = window_ref()
        act = action_ref()
        if window is None or act is None:
            return
        window._record_command_action(act, k, m)
    action.triggered.connect(_record)
    action.setProperty("_mfv_macro_recorder_connected", True)


def _record_command_action(self: MainWindow, action, key: str, meta) -> None:
    recorder = getattr(self._state, "recorder", None)
    if recorder is None or not recorder.enabled:
        return
    label = str(action.text()).replace("&", "").strip()
    summary = meta.summary or label or key
    recorder.record_command(
        key,
        summary=summary,
        code=meta.record or None,
        gui_class=meta.gui_class,
        command_text=label,
        command_category=meta.category,
    )


def _refresh_command_index(self: MainWindow) -> None:
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


def _on_search_activated(self: MainWindow, text: str) -> None:
    entry = self._search_display_map.get(text)
    if entry is None:
        return
    self._toolbar_search.clear()
    if entry.enabled:
        entry.action.trigger()
    else:
        self._state.log(f"'{entry.text}' is currently unavailable (needs a dataset?).", "WARN")


def _on_search_return(self: MainWindow) -> None:
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


def _on_command_finder_destroyed(self: MainWindow, *_args) -> None:
    # The dialog is WA_DeleteOnClose; drop the stale reference so a later open
    # doesn't touch a deleted C++ object (RuntimeError).
    self._command_finder = None


def _command_finder_alive(self: MainWindow) -> bool:
    dlg = self._command_finder
    if dlg is None:
        return False
    try:
        dlg.isVisible()          # touches the C++ object; raises if deleted
        return True
    except RuntimeError:
        self._command_finder = None
        return False


def _open_command_finder(self: MainWindow, initial: str = "") -> None:
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


def _align_toolbar_to_menu(self: MainWindow) -> None:
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


def _apply_menu_separator_style(self: MainWindow) -> None:
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
