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

from dataclasses import dataclass, field


@dataclass(frozen=True)
class ParamMeta:
    """One operational parameter of a command (for method text / headless calls)."""
    name: str
    kind: str = "float"          # float | int | bool | nm | choice | str
    default: object = None
    unit: str = ""
    description: str = ""


@dataclass(frozen=True)
class CommandMeta:
    source: str = ""
    keywords: tuple[str, ...] = ()
    summary: str = ""
    category: str = ""
    params: tuple[ParamMeta, ...] = ()
    inputs: tuple[str, ...] = ()
    outputs: tuple[str, ...] = ()


A = "minflux_viewer/analysis/"
U = "minflux_viewer/ui/"
C = "minflux_viewer/core/"

# Keyed by the QAction attribute name (on MainWindow or the generated UI).
COMMAND_META: dict[str, CommandMeta] = {
    # ---- File -----------------------------------------------------------------
    "actionOpen": CommandMeta(C + "loader.py", ("import", "load", "mat", "npy", "csv", "json"),
                              "Open a MINFLUX/localization dataset.", "file"),
    # Spreadsheet (.csv/.xlsx) and TIFF (.tif) files open by drag-and-drop only —
    # no dedicated File-menu commands, so no command-finder entries.
    "actionSave": CommandMeta(C + "save.py",
                              ("export", "write", "processed", "snapshot", "mat", "npy", "csv", "zarr", "msr"),
                              "Save/export the active dataset (raw canonical or snapshot; any "
                              "enabled format incl. a custom .msr writer).", "file"),
    "actionClose": CommandMeta(U + "main_window.py", ("close", "remove", "dataset"),
                               "Close the active dataset and its windows.", "file"),
    "actionCloseAll": CommandMeta(U + "main_window.py", ("close", "remove", "all", "datasets"),
                                  "Close all datasets and their windows.", "file"),
    "actionCloseAllWindows": CommandMeta(U + "main_window.py",
                                  ("close", "all", "windows", "dialogs", "particle", "everything"),
                                  "Close all datasets and every plugin/analysis dialog, "
                                  "keeping only the Log and Console.", "file"),
    "actionQuit": CommandMeta(U + "main_window.py", ("exit", "quit"), "Quit the application.", "file"),

    # ---- Edit -----------------------------------------------------------------
    "actionDatasetManager": CommandMeta(U + "dataset_manager.py", ("datasets", "layers", "manager", "list"),
                              "Open the Dataset Manager (list / activate / organize datasets).", "edit"),
    "actionFilter": CommandMeta(U + "filter_dialog.py", ("threshold", "gate", "select", "efo", "cfr", "dcr"),
                              "Filter localizations by attribute ranges.", "edit"),
    "actionDuplicate": CommandMeta(C + "roi_crop.py", ("crop", "copy", "subset", "roi"),
                              "Duplicate / crop the dataset (optionally to an ROI).", "edit"),
    "actionPreferences": CommandMeta(U + "preferences_dialog.py", ("settings", "options", "config"),
                              "Application preferences.", "edit"),

    # ---- View -----------------------------------------------------------------
    "actionShowInfo": CommandMeta(U + "data_window.py", ("dataset", "information", "metadata"),
                              "Show the Dataset Information window.", "view"),
    "actionAttributePlot": CommandMeta(U + "attribute_window.py", ("scatter", "attribute", "color"),
                              "Attribute plot (color localizations by an attribute).", "view"),
    "actionAttributePlot3DMatplotlib": CommandMeta(U + "attribute_3d_matplotlib_window.py",
                              ("3d", "matplotlib", "attribute"), "3-D attribute plot (matplotlib).", "view"),
    "actionHistogram": CommandMeta(U + "histogram_window.py", ("distribution", "attribute", "bins"),
                              "Attribute histogram.", "view"),
    "actionScatter": CommandMeta(U + "scatter_window.py", ("points", "localizations", "xy", "xz", "yz", "3d"),
                              "Localization scatter plot.", "view"),
    "actionRender": CommandMeta(U + "render_window.py", ("image", "reconstruction", "histogram", "gaussian"),
                              "Rendered localization image.", "view"),
    "actionLog": CommandMeta(U + "log_window.py", ("events", "messages"), "Event log window.", "view"),

    # ---- Process › Channel ----------------------------------------------------
    "actionChannelTool": CommandMeta(U + "channel_combine_dialog.py", ("channel", "multicolor"),
                              "Channel tools.", "process"),
    "actionChannelCombine": CommandMeta(U + "channel_combine_dialog.py", ("channel", "merge", "overlay", "multicolor"),
                              "Combine datasets into a multi-channel overlay.", "process"),
    "actionChannelSplit": CommandMeta(U + "channel_combine_dialog.py", ("channel", "separate", "split"),
                              "Split an overlay back into per-channel datasets.", "process"),
    "actionChannelFlatten": CommandMeta(C + "channel_flatten.py",
                              ("channel", "flatten", "merge", "combine", "overlay", "single", "pool", "collapse"),
                              "Flatten a multi-channel overlay into one non-overlay dataset "
                              "(transforms baked, trace ids remapped) for combined analysis.", "process",
                              inputs=("active multi-channel overlay",),
                              outputs=("single flattened dataset (hot LUT)",)),
    "actionChannelOverlay": CommandMeta(U + "channel_combine_dialog.py", ("channel", "overlay", "multicolor"),
                              "Overlay datasets as channels.", "process"),
    "actionChannelSeparateDcr": CommandMeta(A + "dcr_em.py",
                              ("dcr", "two color", "2 color", "spectral", "em", "gaussian mixture", "unmix"),
                              "Separate two colours by a Gaussian-mixture (EM) fit of the DCR distribution; "
                              "each trace is assigned by its mean DCR.", "process",
                              params=(ParamMeta("n_components", "int", 2, "", "mixture components"),),
                              inputs=("active dataset with dcr",),
                              outputs=("red/green/(unassigned) overlay datasets",)),

    # ---- Process › ROI --------------------------------------------------------
    "actionRoiManager": CommandMeta(U + "roi_manager.py", ("roi", "regions", "manager"),
                              "ROI Manager (add/edit/convert/save ROIs).", "process"),
    "actionRoiResize": CommandMeta(C + "roi_convert.py", ("roi", "enlarge", "shrink", "grow", "band", "buffer"),
                              "Enlarge / shrink an ROI.", "process"),
    "actionRoiSkeletonize": CommandMeta(C + "roi_convert.py", ("roi", "skeleton", "centerline", "medial axis"),
                              "Skeletonize a region ROI to its centreline.", "process"),
    "actionRoiConvexHull": CommandMeta(C + "roi_convert.py", ("roi", "hull", "convex"),
                              "Convex hull of a polygon/freehand ROI.", "process"),
    "actionRoi3D": CommandMeta(U + "roi_3d_dialog.py",
                              ("roi", "3d", "volume", "extrude", "orthogonal", "xy", "xz", "yz",
                               "crop", "select", "region", "intersection"),
                              "Draw a 3-D ROI by intersecting 2-D shapes extruded from the "
                              "XY/XZ/YZ ortho views; crop the active dataset to it.",
                              "process", inputs=("active dataset (loc)",),
                              outputs=("cropped dataset (localizations inside the 3-D ROI)",)),
    # Convert sub-menu actions are tagged as a group (see _apply_command_meta).
    "_roi_convert": CommandMeta(C + "roi_convert.py", ("roi", "convert", "rectangle", "oval", "point", "line", "region"),
                              "Convert an ROI to another type.", "process"),

    # ---- Process › Batch ------------------------------------------------------
    "actionBatchRender": CommandMeta(U + "main_window.py", ("batch", "render", "export"),
                              "Batch render (placeholder).", "process"),
    "actionBatchExport": CommandMeta(U + "main_window.py", ("batch", "export"), "Batch export (placeholder).", "process"),
    "actionBatchFilter": CommandMeta(U + "main_window.py", ("batch", "filter"), "Batch filter (placeholder).", "process"),

    # ---- Analyze › Measure ----------------------------------------------------
    "actionScaleBar": CommandMeta(U + "scale_bar.py", ("scale bar", "ruler", "measure", "nm"),
                              "Add a draggable scale bar to a 2-D view.", "analysis"),
    "actionPlotProfile": CommandMeta(U + "plot_profile_dialog.py",
                              ("profile", "line", "intensity", "measure", "imagej", "fiji", "width"),
                              "Plot the localization-density profile along a line/polyline/freehand-line "
                              "ROI in a render or scatter view (live, ImageJ-style, tunable width).", "analysis"),
    "actionSetMeasurements": CommandMeta(U + "set_measurements_dialog.py", ("measure", "settings"),
                              "Configure which measurements are reported.", "analysis"),

    # ---- Analyze › Localization precision -------------------------------------
    "actionLocPrecisionStdDev": CommandMeta(A + "localization_precision.py",
                              ("precision", "sigma", "std dev", "per trace", "resolution", "ostersehlt"),
                              "Localization precision as the per-trace standard deviation "
                              "(traces with ≥ MIN_LOCS locs; raw z). Ref: Ostersehlt 2022.", "analysis",
                              params=(ParamMeta("min_locs", "int", 5, "", "min localizations per trace"),),
                              inputs=("active dataset (loc, tid)",),
                              outputs=("per-trace σ (x,y,z) nm", "median σ")),
    "actionLocPrecisionCrlb": CommandMeta(A + "localization_precision.py",
                              ("precision", "crlb", "cramer rao", "photons", "eco", "mortensen"),
                              "Cramér-Rao lower bound from eco/efo/fbg at the last valid iteration. "
                              "Ref: Mortensen 2010.", "analysis",
                              inputs=("active dataset (eco, efo, fbg)",), outputs=("CRLB precision (nm)",)),
    "actionLocPrecisionFrc": CommandMeta(A + "localization_precision.py",
                              ("precision", "frc", "fourier ring correlation", "resolution", "banterle", "nieuwenhuizen"),
                              "Fourier ring correlation resolution (1/7 threshold). "
                              "Refs: Banterle 2013, Nieuwenhuizen 2013.", "analysis",
                              inputs=("active dataset (loc)",), outputs=("FRC resolution (nm)",)),

    # ---- Analyze › Local density ---------------------------------------------
    "actionLocalDensity": CommandMeta(A + "local_density.py",
                              ("density", "neighbours", "kd-tree", "ripley", "crowding", "den"),
                              "Local density per localization (neighbour count within a radius; centre included).",
                              "analysis",
                              params=(ParamMeta("radius_nm", "nm", 100.0, "nm", "neighbourhood radius"),
                                      ParamMeta("dimensions", "choice", 2, "", "2 or 3"),
                                      ParamMeta("method", "choice", "kdtree", "",
                                                "kdtree | voxel_histogram | voxel_radius")),
                              inputs=("active dataset (loc)",), outputs=("den attribute (per loc)",)),

    # ---- Analyze › Clustering -------------------------------------------------
    "actionDbscan": CommandMeta(U + "main_window.py", ("cluster", "dbscan", "density"),
                              "DBSCAN clustering (placeholder).", "analysis"),
    "actionKNearestNeighbour": CommandMeta(U + "main_window.py", ("cluster", "knn", "nearest neighbour"),
                              "K-nearest-neighbour analysis (placeholder).", "analysis"),
    "actionHlyB2D": CommandMeta(A + "hlyb_clustering.py",
                              ("hlyb", "subunit", "dimer", "pair", "cluster", "ecoli", "2d"),
                              "HlyB sub-unit pair analysis in 2-D (E. coli border removal + pair distances).",
                              "analysis", inputs=("active dataset (loc, tid)",),
                              outputs=("sub-unit centres", "pair-distance histogram")),
    "actionHlyB3D": CommandMeta(A + "hlyb_clustering.py",
                              ("hlyb", "subunit", "dimer", "pair", "cluster", "3d"),
                              "HlyB sub-unit pair analysis in 3-D.", "analysis",
                              inputs=("active dataset (loc, tid)",),
                              outputs=("sub-unit centres", "pair-distance histogram")),
    "actionHlyBTemplate3D": CommandMeta(A + "hlyb_clustering.py", ("hlyb", "template", "3d"),
                              "HlyB template matching (3-D).", "analysis"),

    # ---- Analyze › Trace ------------------------------------------------------
    "actionTraceSize": CommandMeta(A + "trace_analysis.py",
                              ("trace", "size", "spread", "cluster size", "localization spread"),
                              "Estimate the average per-trace size (log-distance Gaussian fit).", "analysis",
                              inputs=("active dataset (raw loc, tid)",), outputs=("trace size (nm)",)),
    "actionTraceAnisotropy": CommandMeta(A + "trace_analysis.py",
                              ("anisotropy", "rimf", "z scaling", "refractive index", "aspect ratio"),
                              "Estimate anisotropy / RIMF (z-scaling) from raw last-valid trace sizes.",
                              "analysis", inputs=("active dataset (raw loc, tid)",),
                              outputs=("anisotropy factor / RIMF",)),

    # ---- Analyze › Segmentation ----------------------------------------------
    "actionSegNpc2D": CommandMeta(A + "npc_segmentation.py",
                              ("npc", "nuclear pore", "ring", "segmentation", "convolution", "2d"),
                              "Detect NPCs by ring-kernel convolution → rectangle ROIs.", "analysis",
                              params=(ParamMeta("diameter_nm", "nm", 100.0, "nm", "NPC diameter"),
                                      ParamMeta("rim_nm", "nm", 20.0, "nm", "ring rim width"),
                                      ParamMeta("pixel_nm", "nm", 4.0, "nm", "histogram pixel"),
                                      ParamMeta("min_support", "float", 0.25, "", "ring support threshold")),
                              inputs=("active dataset (loc)",), outputs=("NPC rectangle ROIs",)),
    "actionSegNpc3D": CommandMeta(U + "main_window.py", ("npc", "nuclear pore", "3d", "segmentation"),
                              "NPC segmentation in 3-D (placeholder).", "analysis"),
    "actionSegConvolution": CommandMeta(U + "conv_segmentation_dialog.py",
                              ("convolution", "matched filter", "ring", "disk", "blob", "detection", "segmentation", "2d"),
                              "Geometry-kernel matched-filter detection (ring/disk/gaussian/LoG) → rectangle ROIs.",
                              "analysis",
                              params=(ParamMeta("geometry", "choice", "ring", "", "ring|disk|gaussian|log"),
                                      ParamMeta("pixel_nm", "nm", 5.0, "nm", "render pixel"),
                                      ParamMeta("min_response", "float", 0.5, "", "matched-filter response threshold"),
                                      ParamMeta("separation_nm", "nm", 0.0, "nm", "min peak separation (NMS)")),
                              inputs=("active dataset (loc)",), outputs=("detection rectangle ROIs",)),
    "actionSegConvolution3D": CommandMeta(U + "conv_segmentation_3d_dialog.py",
                              ("convolution", "matched filter", "shell", "ball", "3d", "detection", "segmentation"),
                              "Genuinely-3-D matched-filter detection (shell/ball/gaussian/LoG) → 3-D point ROIs.",
                              "analysis", inputs=("active dataset (loc, 3-D)",), outputs=("3-D point ROIs",)),
    "actionSegCurvilinear": CommandMeta(U + "curvilinear_segmentation_dialog.py",
                              ("curvilinear", "filament", "fiber", "skeleton", "ridge", "structure"),
                              "Detect curvilinear structures.", "analysis"),
    "actionSegStraightenedVolume": CommandMeta(U + "straightened_volume_dialog.py",
                              ("straighten", "skeleton", "centerline", "volume", "reslice"),
                              "Straighten a volume along a skeleton/line.", "analysis"),
    "actionSegParticleAverage": CommandMeta(U + "particle_average_dialog.py",
                              ("particle average", "averaging", "template", "npc", "super particle", "locmofit", "fusion"),
                              "Particle averaging (template-free / template / NPC model fit) of collected particles.",
                              "analysis",
                              params=(ParamMeta("method", "choice", "free", "", "free | template | geomfit"),
                                      ParamMeta("box_nm", "nm", 150.0, "nm", "average box"),
                                      ParamMeta("pixel_nm", "nm", 3.0, "nm", "alignment pixel"),
                                      ParamMeta("n_angles", "int", 36, "", "rotation search steps"),
                                      ParamMeta("correct_tilt", "bool", False, "", "PCA axial-tilt correction")),
                              inputs=("collected particles",), outputs=("averaged super-particle dataset",)),

    # ---- Analyze › Tracking ---------------------------------------------------
    "actionParticleTracking": CommandMeta(U + "main_window.py", ("tracking", "linking", "trajectory"),
                              "Particle tracking (placeholder).", "analysis"),
    "actionMsdAnalysis": CommandMeta(U + "main_window.py", ("msd", "diffusion", "mean square displacement"),
                              "MSD / diffusion analysis (placeholder).", "analysis"),

    # ---- Help -----------------------------------------------------------------
    "actionConsole": CommandMeta(U + "console_window.py", ("stdout", "stderr", "console", "output"),
                              "Console (raw stdout/stderr).", "help"),
    "actionMemoryMonitor": CommandMeta(U + "main_window.py", ("memory", "monitor", "ram"),
                              "Monitor memory usage.", "help"),
    "actionCommandFinder": CommandMeta(U + "command_finder.py", ("search", "commands", "finder", "palette"),
                              "Search all menu commands (Fiji-style).", "help"),
    "actionCheckUpdates": CommandMeta(C + "updater.py", ("update", "version", "release"),
                              "Check for a newer release.", "help"),
    "actionAbout": CommandMeta(U + "main_window.py", ("about", "version", "credits"), "About this application.", "help"),
}


def meta_for(key: str) -> CommandMeta | None:
    return COMMAND_META.get(key)
