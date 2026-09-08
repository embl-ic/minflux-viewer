"""
minflux_viewer.ui.command_meta
==============================
Declarative **metadata registry** for menu commands — one coherent source of truth
consumed by the Command Finder (Source column + keyword search), and designed to
also feed the method-text generator and future headless / batch / scripting use.

Each entry (:class:`CommandMeta`, keyed by the command's action attribute name):

* ``source``   — implementing python file (``minflux_viewer/…``); every non-trivial
  command should be traceable to one (recent-file entries etc. are excluded).
* ``keywords`` — extra search terms **not** in the command name or menu path, so a
  leaf like *Segmentation › NPC › 2D* is findable by ``npc``/``nuclear pore`` and
  *Convolution* by ``matched filter`` etc.
* ``summary``  — one-line scientific description (tooltip / method text).
* ``category`` — ``file`` | ``view`` | ``edit`` | ``process`` | ``analysis`` | ``help`` | ``plugin``.
* ``params`` / ``inputs`` / ``outputs`` — operational metadata for method-text and
  headless/scripting/batch. **Populated incrementally**; empty ⇒ not yet described.

``main_window._apply_command_meta()`` stamps ``command_source`` / ``command_keywords``
on each QAction from this registry; `command_finder` reads those back.
"""

from __future__ import annotations

from dataclasses import dataclass

from ..core.recorder import GuiClass


@dataclass(frozen=True)
class ParamMeta:
    """One operational parameter of a command (for method text / headless calls)."""
    name: str
    kind: str = "float"          # float | int | bool | nm | choice | str
    default: object = None
    unit: str = ""
    description: str = ""


#: How a command relates to the GUI, for the macro recorder. The rule, applied
#: consistently so a silent script means something:
#:
#: ``GUI_ONLY``    presentation with no data consequence -- opening a window,
#:                 a monitor, a settings dialog. Dropped in silent mode.
#: ``GUI_RESULT``  the command collected something from a human: a dialog
#:                 value, a file path, a drawn shape, a level picked by eye.
#:                 The recorded call embeds that decision, so it replays
#:                 without a user -- but the numbers are the ones chosen that
#:                 day, which is what makes a recording a macro and not an
#:                 analysis.
#: ``GUI_FREE``    no interactive input at all; nothing a human chose is needed.
#:
#: Both GUI_RESULT and GUI_FREE are kept in silent mode; the distinction is
#: documentary, and it is what tells a reader which numbers were somebody's
#: judgement.


@dataclass(frozen=True)
class CommandMeta:
    source: str = ""
    keywords: tuple[str, ...] = ()
    summary: str = ""
    category: str = ""
    params: tuple[ParamMeta, ...] = ()
    inputs: tuple[str, ...] = ()
    outputs: tuple[str, ...] = ()
    gui_class: GuiClass | None = None
    record: str = ""


A = "minflux_viewer/analysis/"
U = "minflux_viewer/ui/"
C = "minflux_viewer/core/"

# Keyed by the QAction attribute name (on MainWindow or the generated UI).
COMMAND_META: dict[str, CommandMeta] = {
    # ---- File -----------------------------------------------------------------
    "actionOpen": CommandMeta(C + "loader.py", ("import", "load", "mat", "npy", "csv", "json"),
                              "Open a MINFLUX/localization dataset.", "file", gui_class=GuiClass.GUI_RESULT),
    # Spreadsheet (.csv/.xlsx) and TIFF (.tif) files open by drag-and-drop only —
    # no dedicated File-menu commands, so no command-finder entries.
    "actionSave": CommandMeta(C + "minflux_zarr.py",
                              ("save", "write", "zarr", "store", "self-contained"),
                              "Save the active dataset as a MINFLUX Viewer Zarr v2 store "
                              "(the format is fixed; the dialog asks only where).",
                              "file", gui_class=GuiClass.GUI_RESULT),
    "actionSaveAsMinflux": CommandMeta(C + "save.py", ("save as", "mat", "npy", "json", "minflux"),
                              "Save raw canonical MINFLUX data as .mat, .npy, or .json.", "file", gui_class=GuiClass.GUI_RESULT),
    "actionSaveAsMsr": CommandMeta(C + "save.py", ("save as", "msr", "experimental", "minflux"),
                              "Save raw canonical MINFLUX data as an experimental .msr file.", "file", gui_class=GuiClass.GUI_RESULT),
    "actionSaveAsSpreadsheet": CommandMeta(C + "save.py",
                              ("save as", "spreadsheet", "csv", "delimiter", "columns", "table"),
                              "Export chosen attributes as a custom CSV table for another tool "
                              "(the all-iteration canonical table is written by the MSR reader "
                              "and by Save / export data > More options).",
                              "file", gui_class=GuiClass.GUI_RESULT),
    "actionSaveAsZarrZip": CommandMeta(C + "minflux_zarr.py",
                              ("save as", "zarr", "zip", "single file", "sealed",
                               "package", "archive", "portable"),
                              "Save the self-contained Zarr v2 store sealed into one "
                              ".zarr.zip file (no in-place processing updates).", "file", gui_class=GuiClass.GUI_RESULT),
    "actionSaveAsZarr": CommandMeta(C + "save.py", ("save as", "zarr", "v2"),
                              "Save raw canonical MINFLUX data as a Zarr v2 directory.", "file", gui_class=GuiClass.GUI_RESULT),
    "actionSaveAsHdf5": CommandMeta(C + "save.py", ("save as", "hdf5", "picasso", "render"),
                              "Export the active dataset as Picasso-compatible HDF5 + YAML.", "file", gui_class=GuiClass.GUI_RESULT),
    "actionSaveAsOmeTiff": CommandMeta(C + "tiff_export.py", ("save as", "ome", "tiff", "render", "imagej"),
                              "Export the active render as an OME-TIFF stack.", "file", gui_class=GuiClass.GUI_RESULT),
    "actionClose": CommandMeta(U + "main_window.py", ("close", "remove", "dataset"),
                               "Close the active dataset and its windows.", "file", gui_class=GuiClass.GUI_FREE),
    "actionCloseAll": CommandMeta(U + "main_window.py", ("close", "remove", "all", "datasets"),
                                  "Close all datasets and their windows.", "file", gui_class=GuiClass.GUI_FREE),
    "actionCloseAllWindows": CommandMeta(U + "main_window.py",
                                  ("close", "all", "windows", "dialogs", "particle", "everything"),
                                  "Close all datasets and every plugin/analysis dialog, "
                                  "keeping only the Log and Console.", "file", gui_class=GuiClass.GUI_ONLY),
    "actionQuit": CommandMeta(U + "main_window.py", ("exit", "quit"), "Quit the application.", "file", gui_class=GuiClass.GUI_ONLY),

    # ---- Edit -----------------------------------------------------------------
    "actionDatasetManager": CommandMeta(U + "dataset_manager.py", ("datasets", "layers", "manager", "list"),
                              "Open the Dataset Manager (list / activate / organize datasets).", "edit", gui_class=GuiClass.GUI_ONLY),
    "actionFilter": CommandMeta(U + "filter_dialog.py", ("threshold", "gate", "select", "efo", "cfr", "dcr"),
                              "Filter localizations by attribute ranges.", "edit", gui_class=GuiClass.GUI_RESULT),
    "actionDuplicate": CommandMeta(C + "roi_crop.py", ("crop", "copy", "subset", "roi"),
                              "Duplicate / crop the dataset (optionally to an ROI).", "edit", gui_class=GuiClass.GUI_RESULT),
    "actionPreferences": CommandMeta(U + "preferences_dialog.py", ("settings", "options", "config"),
                              "Application preferences.", "edit", gui_class=GuiClass.GUI_ONLY),

    # ---- View -----------------------------------------------------------------
    "actionShowInfo": CommandMeta(U + "data_window.py", ("dataset", "information", "metadata"),
                              "Show the Dataset Information window.", "view", gui_class=GuiClass.GUI_ONLY),
    "actionAttributePlot": CommandMeta(U + "attribute_window.py",
                              ("scatter", "attribute", "color", "opengl", "gpu", "cpu",
                               "aggregation", "density", "large data", "millions"),
                              "Attribute plot (color localizations by an attribute). The 2-D "
                              "renderer is chosen automatically: GPU for round markers when "
                              "OpenGL is available, otherwise exact bulk painting or a "
                              "screen-space count/mean aggregation for dense views.", "view",
                              gui_class=GuiClass.GUI_ONLY,
                              record="mfv.view.attribute_plot()"),
    "actionHistogram": CommandMeta(U + "histogram_window.py", ("distribution", "attribute", "bins"),
                              "Attribute histogram.", "view",
                              gui_class=GuiClass.GUI_ONLY,
                              record="mfv.view.histogram()"),
    "actionScatter": CommandMeta(U + "scatter_window.py", ("points", "localizations", "xy", "xz", "yz", "3d"),
                              "Localization scatter plot.", "view",
                              gui_class=GuiClass.GUI_ONLY,
                              record="mfv.view.scatter()"),
    "actionRender": CommandMeta(U + "render_window.py",
                              ("image", "reconstruction", "histogram", "gaussian", "advanced", "precision", "bilinear"),
                              "Rendered localization image (right-click › View › Render Method "
                              "selects the reconstruction method).", "view",
                              gui_class=GuiClass.GUI_ONLY,
                              record="mfv.view.render()"),
    "actionLog": CommandMeta(U + "log_window.py", ("events", "messages"), "Event log window.", "view", gui_class=GuiClass.GUI_ONLY),

    # ---- Process › Channel ----------------------------------------------------
    "actionChannelTool": CommandMeta(U + "channel_combine_dialog.py", ("channel", "multicolor"),
                              "Channel tools.", "process", gui_class=GuiClass.GUI_RESULT),
    "actionChannelCombine": CommandMeta(U + "channel_combine_dialog.py", ("channel", "merge", "overlay", "multicolor"),
                              "Combine datasets into a multi-channel overlay.", "process", gui_class=GuiClass.GUI_RESULT),
    "actionChannelSplit": CommandMeta(U + "channel_combine_dialog.py", ("channel", "separate", "split"),
                              "Split an overlay back into per-channel datasets.", "process", gui_class=GuiClass.GUI_RESULT),
    "actionChannelFlatten": CommandMeta(C + "channel_flatten.py",
                              ("channel", "flatten", "merge", "combine", "overlay", "single", "pool", "collapse"),
                              "Flatten a multi-channel overlay into one non-overlay dataset "
                              "(transforms baked, trace ids remapped) for combined analysis.", "process",
                              inputs=("active multi-channel overlay",),
                              outputs=("single flattened dataset (hot LUT)",), gui_class=GuiClass.GUI_FREE),
    "actionChannelSeparateDcr": CommandMeta(U + "attribute_separation_dialog.py",
                              ("dcr", "two color", "2 color", "spectral", "em", "gaussian mixture", "unmix",
                               "photon weighted", "majority vote", "channel"),
                              "Separate colors by a mixture fit of the DCR distribution into "
                              "value-window channels; each trace is assigned by mean/median/majority vote "
                              "(optionally photon-weighted DCR).", "process",
                              params=(ParamMeta("n_components", "int", 2, "", "mixture components"),),
                              inputs=("active dataset with dcr",),
                              outputs=("per-channel (+unassigned) overlay datasets",), gui_class=GuiClass.GUI_RESULT),
    "actionChannelSeparateTime": CommandMeta(U + "time_channel_dialog.py",
                              ("time", "window", "split", "exchange paint", "multiplex", "channel",
                               "convert overlay"),
                              "Separate the active dataset into an overlay of "
                              "filtered acquisition-time channels.", "process",
                              inputs=("active dataset with tim",),
                              outputs=("time-window overlay datasets",), gui_class=GuiClass.GUI_RESULT),
    "actionChannelSeparateAttribute": CommandMeta(U + "attribute_separation_dialog.py",
                              ("attribute", "distribution", "mixture", "gaussian", "log-normal",
                               "gamma", "poisson", "convert overlay", "channel", "unmix", "efo", "cfr"),
                              "Convert a dataset to a multi-channel overlay by any MINFLUX "
                              "attribute's distribution (mixture fit → value-window channels).",
                              "process",
                              params=(ParamMeta("n_components", "int", 2, "", "mixture components"),),
                              inputs=("active MINFLUX dataset",),
                              outputs=("per-channel (+unassigned) overlay datasets",), gui_class=GuiClass.GUI_RESULT),
    "actionRevertOverlay": CommandMeta(U + "main_window.py",
                              ("revert", "undo", "overlay", "original", "recombine", "unsplit",
                               "convert overlay", "channel"),
                              "Revert a separation overlay back to its single original dataset "
                              "(removes the channels; reconstructs if the source was closed).",
                              "process",
                              inputs=("active separation overlay",),
                              outputs=("original single dataset",), gui_class=GuiClass.GUI_FREE),

    # ---- Process › ROI --------------------------------------------------------
    "actionRoiManager": CommandMeta(U + "roi_manager.py", ("roi", "regions", "manager"),
                              "ROI Manager (add/edit/convert/save ROIs).", "process", gui_class=GuiClass.GUI_ONLY),
    "actionRoiResize": CommandMeta(C + "roi_convert.py", ("roi", "enlarge", "shrink", "grow", "band", "buffer"),
                              "Enlarge / shrink an ROI.", "process", gui_class=GuiClass.GUI_RESULT),
    "actionRoiSkeletonize": CommandMeta(C + "roi_convert.py", ("roi", "skeleton", "centerline", "medial axis"),
                              "Skeletonize a region ROI to its centreline.", "process", gui_class=GuiClass.GUI_FREE),
    "actionRoiConvexHull": CommandMeta(C + "roi_convert.py", ("roi", "hull", "convex"),
                              "Convex hull of a polygon/freehand ROI.", "process", gui_class=GuiClass.GUI_FREE),
    "actionRoi3D": CommandMeta(U + "roi_3d_dialog.py",
                              ("roi", "3d", "volume", "extrude", "orthogonal", "xy", "xz", "yz",
                               "crop", "select", "region", "intersection"),
                              "Draw a 3-D ROI by intersecting 2-D shapes extruded from the "
                              "XY/XZ/YZ ortho views; crop the active dataset to it.",
                              "process", inputs=("active dataset (loc)",),
                              outputs=("cropped dataset (localizations inside the 3-D ROI)",), gui_class=GuiClass.GUI_RESULT),
    "actionRoiRestore": CommandMeta(U + "main_window.py",
                              ("roi", "restore", "recover", "undo delete", "bring back", "sync", "views"),
                              "Restore the active ROI onto another view (render ↔ scatter), "
                              "or bring back the last active ROI after an accidental delete.",
                              "process", gui_class=GuiClass.GUI_FREE),
    # Convert / Fit sub-menu actions are tagged as a group (see _apply_command_meta).
    "_roi_convert": CommandMeta(C + "roi_convert.py", ("roi", "convert", "rectangle", "oval", "point", "line", "region"),
                              "Convert an ROI to another type.", "process", gui_class=GuiClass.GUI_FREE),
    "_roi_fit": CommandMeta(C + "roi_fit.py",
                              ("roi", "fit", "rectangle", "circle", "ellipse", "polygon", "convex hull",
                               "spline", "interpolate", "minimum enclosing", "circumscribed", "moment"),
                              "Fit a shape to the localizations a region ROI highlights "
                              "(or spline-fit / interpolate its outline).", "process", gui_class=GuiClass.GUI_FREE),
    "actionAggregate": CommandMeta(A + "aggregation.py",
                              ("aggregate", "localizations", "imspector", "trace", "photon", "binning"),
                              "Aggregate valid final MINFLUX localizations per trace using a "
                              "photon threshold and photon-weighted spatial centroids.",
                              "process", inputs=("active MINFLUX dataset",),
                              outputs=("aggregated dataset",), gui_class=GuiClass.GUI_RESULT),

    # ---- Process › Batch ------------------------------------------------------
    "actionBatchRender": CommandMeta(U + "main_window.py", ("batch", "render", "export"),
                              "Batch render (placeholder).", "process", gui_class=GuiClass.GUI_RESULT),
    "actionBatchExport": CommandMeta(U + "main_window.py", ("batch", "export"), "Batch export (placeholder).", "process", gui_class=GuiClass.GUI_RESULT),
    "actionBatchFilter": CommandMeta(U + "main_window.py", ("batch", "filter"), "Batch filter (placeholder).", "process", gui_class=GuiClass.GUI_RESULT),

    # ---- Analyze › Measure ----------------------------------------------------
    "actionScaleBar": CommandMeta(U + "scale_bar.py", ("scale bar", "ruler", "measure", "nm"),
                              "Add a draggable scale bar to a 2-D view.", "analysis", gui_class=GuiClass.GUI_ONLY),
    "actionPlotProfile": CommandMeta(U + "plot_profile_dialog.py",
                              ("profile", "line", "intensity", "measure", "imagej", "fiji", "width"),
                              "Plot the localization-density profile along a line/polyline/freehand-line "
                              "ROI in a render or scatter view (live, ImageJ-style, tunable width).", "analysis", gui_class=GuiClass.GUI_RESULT),
    "actionSetMeasurements": CommandMeta(U + "set_measurements_dialog.py", ("measure", "settings"),
                              "Configure which measurements are reported.", "analysis", gui_class=GuiClass.GUI_ONLY),

    # ---- Analyze › Localization precision -------------------------------------
    "actionLocPrecisionStdDev": CommandMeta(A + "localization_precision.py",
                              ("precision", "sigma", "std dev", "per trace", "resolution", "ostersehlt"),
                              "Localization precision as the per-trace standard deviation "
                              "(traces with ≥ MIN_LOCS locs; raw z). Ref: Ostersehlt 2022.", "analysis",
                              params=(ParamMeta("min_locs", "int", 5, "", "min localizations per trace"),),
                              inputs=("active dataset (loc, tid)",),
                              outputs=("per-trace σ (x,y,z) nm", "median σ"), gui_class=GuiClass.GUI_FREE),
    "actionLocPrecisionCrlb": CommandMeta(A + "localization_precision.py",
                              ("precision", "crlb", "cramer rao", "photons", "eco", "mortensen"),
                              "Cramér-Rao lower bound from eco/efo/fbg at the last valid iteration. "
                              "Ref: Mortensen 2010.", "analysis",
                              inputs=("active dataset (eco, efo, fbg)",), outputs=("CRLB precision (nm)",), gui_class=GuiClass.GUI_RESULT),
    "actionLocPrecisionFrc": CommandMeta(A + "localization_precision.py",
                              ("precision", "frc", "fourier ring correlation", "resolution", "banterle", "nieuwenhuizen"),
                              "Fourier ring correlation resolution (1/7 threshold). "
                              "Refs: Banterle 2013, Nieuwenhuizen 2013.", "analysis",
                              inputs=("active dataset (loc)",), outputs=("FRC resolution (nm)",), gui_class=GuiClass.GUI_FREE),

    # ---- Analyze › Local density ---------------------------------------------
    "actionLocalDensity": CommandMeta(A + "local_density.py",
                              ("density", "neighbours", "kd-tree", "ripley", "crowding", "den"),
                              "Local density per localization (neighbour count within a radius; centre included).",
                              "analysis",
                              params=(ParamMeta("radius_nm", "nm", 100.0, "nm", "neighbourhood radius"),
                                      ParamMeta("dimensions", "choice", 2, "", "2 or 3"),
                                      ParamMeta("method", "choice", "kdtree", "",
                                                "kdtree | voxel_histogram | voxel_radius")),
                              inputs=("active dataset (loc)",), outputs=("den attribute (per loc)",), gui_class=GuiClass.GUI_RESULT),

    # ---- Analyze › Clustering -------------------------------------------------
    "actionDbscan": CommandMeta(U + "main_window.py", ("cluster", "dbscan", "density"),
                              "DBSCAN clustering (placeholder).", "analysis", gui_class=GuiClass.GUI_RESULT),
    "actionKNearestNeighbour": CommandMeta(U + "main_window.py", ("cluster", "knn", "nearest neighbour"),
                              "K-nearest-neighbour analysis (placeholder).", "analysis", gui_class=GuiClass.GUI_RESULT),
    # The HlyB/D subunit pair analysis is now a *plugin*, so its metadata lives
    # on the PluginEntry (name/tooltip/keywords) rather than here — this
    # registry is keyed by QAction attribute name, and a plugin action has
    # none.  The customer workflow now lives in plugins/hlyb_pair_analysis/.

    # ---- Analyze › Trace ------------------------------------------------------
    "actionTraceSize": CommandMeta(A + "trace_analysis.py",
                              ("trace", "size", "spread", "cluster size", "localization spread"),
                              "Estimate the average per-trace size (log-distance Gaussian fit).", "analysis",
                              inputs=("active dataset (raw loc, tid)",), outputs=("trace size (nm)",), gui_class=GuiClass.GUI_FREE),
    "actionTraceAnisotropy": CommandMeta(A + "trace_analysis.py",
                              ("z scaling factor", "axial calibration", "anisotropy", "aspect ratio"),
                              "Estimate the Z scaling factor from raw last-valid trace sizes.",
                              "analysis", inputs=("active dataset (raw loc, tid)",),
                              outputs=("Z scaling factor",), gui_class=GuiClass.GUI_RESULT),

    # ---- Analyze › Segmentation ----------------------------------------------
    # The former actionSegNpc2D / actionSegNpc3D entries were removed with their
    # commands; NPC detection is the `ring` model of the Convolution tool, hence
    # the npc/nuclear-pore keywords below.
    "actionSegConvolution": CommandMeta(U + "conv_segmentation_dialog.py",
                              ("convolution", "matched filter", "ring", "disk", "blob", "detection",
                               "segmentation", "2d", "npc", "nuclear pore"),
                              "Geometry-kernel matched-filter detection (ring/disk/gaussian/LoG) → rectangle ROIs.",
                              "analysis",
                              params=(ParamMeta("geometry", "choice", "ring", "", "ring|disk|gaussian|log"),
                                      ParamMeta("pixel_nm", "nm", 5.0, "nm", "render pixel"),
                                      ParamMeta("min_response", "float", 0.5, "", "matched-filter response threshold"),
                                      ParamMeta("separation_nm", "nm", 0.0, "nm", "min peak separation (NMS)"),
                                      ParamMeta("min_support", "float", 0.0, "", "ring support threshold (0 = off)")),
                              inputs=("active dataset (loc)",), outputs=("detection rectangle ROIs",), gui_class=GuiClass.GUI_RESULT),
    "actionSegConvolution3D": CommandMeta(U + "conv_segmentation_3d_dialog.py",
                              ("convolution", "matched filter", "shell", "ball", "3d", "detection",
                               "segmentation", "npc", "nuclear pore"),
                              "Genuinely-3-D matched-filter detection (shell/ball/gaussian/LoG) → 3-D point ROIs.",
                              "analysis", inputs=("active dataset (loc, 3-D)",), outputs=("3-D point ROIs",), gui_class=GuiClass.GUI_RESULT),
    "actionSegShapeModel": CommandMeta(U + "shape_segmentation_dialog.py",
                              ("shape model", "shape prior", "known geometry", "capsule",
                               "obround", "stadium", "rod", "bacteria", "bacterium",
                               "ecoli", "e. coli", "cell", "outline", "contour",
                               "polygon", "instance segmentation", "touching", "clipped"),
                              "Fit objects of a known geometry (capsule/curved capsule/"
                              "ellipse/rectangle/disk) to the XY density → editable "
                              "polygon contour ROIs; separates touching objects and "
                              "flags ones clipped by the acquisition frame.",
                              "analysis",
                              params=(ParamMeta("shape", "choice", "capsule", "",
                                                "capsule|arc_capsule|ellipse|rectangle|disk"),
                                      ParamMeta("length_nm", "range", (1400.0, 4000.0), "nm",
                                                "expected object length"),
                                      ParamMeta("width_nm", "range", (600.0, 1200.0), "nm",
                                                "expected object width"),
                                      ParamMeta("pixel_nm", "nm", 20.0, "nm", "detection pixel"),
                                      ParamMeta("instance_cost", "float", 0.25, "",
                                                "price of one extra object, in object areas"),
                                      ParamMeta("vertices", "int", 48, "",
                                                "polygon contour vertices")),
                              inputs=("active dataset (loc)",
                                      "acquisition ROI from the source .msr (optional)"),
                              outputs=("polygon contour ROIs",), gui_class=GuiClass.GUI_RESULT),
    "actionSegCurvilinear": CommandMeta(U + "curvilinear_segmentation_dialog.py",
                              ("curvilinear", "filament", "fiber", "skeleton", "ridge", "structure"),
                              "Detect curvilinear structures.", "analysis", gui_class=GuiClass.GUI_RESULT),
    "actionSegStraightenedVolume": CommandMeta(U + "straightened_volume_dialog.py",
                              ("straighten", "skeleton", "centerline", "volume", "reslice"),
                              "Straighten a volume along a skeleton/line.", "analysis", gui_class=GuiClass.GUI_RESULT),
    "actionSegParticleAverage": CommandMeta(U + "particle_average_dialog.py",
                              ("particle average", "averaging", "template", "npc", "super particle", "locmofit", "fusion"),
                              "Particle averaging (template-free / template / NPC model fit) of collected particles.",
                              "analysis",
                              params=(ParamMeta("method", "choice", "free", "", "free | template | geomfit"),
                                      ParamMeta("box_nm", "nm", 150.0, "nm", "average box"),
                                      ParamMeta("pixel_nm", "nm", 3.0, "nm", "alignment pixel"),
                                      ParamMeta("n_angles", "int", 36, "", "rotation search steps"),
                                      ParamMeta("correct_tilt", "bool", False, "", "PCA axial-tilt correction")),
                              inputs=("collected particles",), outputs=("averaged super-particle dataset",), gui_class=GuiClass.GUI_RESULT),

    # ---- Analyze › Tracking ---------------------------------------------------
    "actionParticleTracking": CommandMeta(U + "main_window.py", ("tracking", "linking", "trajectory"),
                              "Particle tracking (placeholder).", "analysis", gui_class=GuiClass.GUI_RESULT),
    "actionMsdAnalysis": CommandMeta(U + "main_window.py", ("msd", "diffusion", "mean square displacement"),
                              "MSD / diffusion analysis (placeholder).", "analysis", gui_class=GuiClass.GUI_RESULT),

    # ---- Help -----------------------------------------------------------------
    "actionConsole": CommandMeta(U + "console_window.py", ("stdout", "stderr", "console", "output"),
                              "Console (raw stdout/stderr).", "help", gui_class=GuiClass.GUI_ONLY, record='mfv.view.console()'),
    "actionTaskMonitor": CommandMeta(U + "task_monitor.py",
                              ("thread", "task", "background", "job", "worker",
                               "cancel", "stop", "kill", "monitor",
                               "memory", "ram", "rss", "usage"),
                              "Background tasks, threads and memory usage; ask a "
                              "running task to stop.", "help", gui_class=GuiClass.GUI_ONLY),
    "actionCommandFinder": CommandMeta(U + "command_finder.py", ("search", "commands", "finder", "palette"),
                              "Search all menu commands (Fiji-style).", "help", gui_class=GuiClass.GUI_ONLY),
    "actionCheckUpdates": CommandMeta(C + "updater.py", ("update", "version", "release"),
                              "Check for a newer release.", "help", gui_class=GuiClass.GUI_ONLY),
    "actionAbout": CommandMeta(U + "main_window.py", ("about", "version", "credits"), "About this application.", "help", gui_class=GuiClass.GUI_ONLY),
}


def meta_for(key: str) -> CommandMeta | None:
    return COMMAND_META.get(key)
