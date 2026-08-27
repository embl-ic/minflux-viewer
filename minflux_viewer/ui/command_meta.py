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
    "actionSaveAsMinflux": CommandMeta(C + "save.py", ("save as", "mat", "npy", "json", "minflux"),
                              "Save raw canonical MINFLUX data as .mat, .npy, or .json.", "file"),
    "actionSaveAsMsr": CommandMeta(C + "save.py", ("save as", "msr", "experimental", "minflux"),
                              "Save raw canonical MINFLUX data as an experimental .msr file.", "file"),
    "actionSaveAsSpreadsheet": CommandMeta(C + "save.py",
                              ("save as", "spreadsheet", "csv", "delimiter", "columns"),
                              "Export chosen attributes as a custom CSV table for another tool "
                              "(the canonical round-tripping table is File > Save).", "file"),
    "actionSaveAsZarrZip": CommandMeta(C + "minflux_zarr.py",
                              ("save as", "zarr", "zip", "single file", "sealed",
                               "package", "archive", "portable"),
                              "Save the self-contained Zarr v2 store sealed into one "
                              ".zarr.zip file (no in-place processing updates).", "file"),
    "actionSaveAsZarr": CommandMeta(C + "save.py", ("save as", "zarr", "v2"),
                              "Save raw canonical MINFLUX data as a Zarr v2 directory.", "file"),
    "actionSaveAsHdf5": CommandMeta(C + "save.py", ("save as", "hdf5", "picasso", "render"),
                              "Export the active dataset as Picasso-compatible HDF5 + YAML.", "file"),
    "actionSaveAsOmeTiff": CommandMeta(C + "tiff_export.py", ("save as", "ome", "tiff", "render", "imagej"),
                              "Export the active render as an OME-TIFF stack.", "file"),
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
    "actionAttributeCpu": CommandMeta(U + "attribute_cpu.py",
                              ("cpu", "bulk", "painter", "aggregation", "density", "large data", "lod"),
                              "Non-GPU Attribute Plot using exact bulk painting for sparse views and "
                              "complete screen-space count/mean aggregation for dense views.", "view"),
    "actionHistogram": CommandMeta(U + "histogram_window.py", ("distribution", "attribute", "bins"),
                              "Attribute histogram.", "view"),
    "actionScatter": CommandMeta(U + "scatter_window.py", ("points", "localizations", "xy", "xz", "yz", "3d"),
                              "Localization scatter plot.", "view"),
    "actionRender": CommandMeta(U + "render_window.py",
                              ("image", "reconstruction", "histogram", "gaussian", "advanced", "precision", "bilinear"),
                              "Rendered localization image (right-click › View › Render Method "
                              "selects the reconstruction method).", "view"),
    "actionAttributeGpu": CommandMeta(U + "attribute_window.py",
                              ("opengl", "gpu", "graphics card", "fast", "millions", "performance",
                               "renderer", "experimental"),
                              "Draw the Attribute Plot's 2-D projection on the GPU: every point of a "
                              "multi-million-row selection at interactive speed, without thinning "
                              "(no ROI selection or Lines in this mode).", "view"),
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
    "actionChannelSeparateDcr": CommandMeta(U + "attribute_separation_dialog.py",
                              ("dcr", "two color", "2 color", "spectral", "em", "gaussian mixture", "unmix",
                               "photon weighted", "majority vote", "channel"),
                              "Separate colors by a mixture fit of the DCR distribution into "
                              "value-window channels; each trace is assigned by mean/median/majority vote "
                              "(optionally photon-weighted DCR).", "process",
                              params=(ParamMeta("n_components", "int", 2, "", "mixture components"),),
                              inputs=("active dataset with dcr",),
                              outputs=("per-channel (+unassigned) overlay datasets",)),
    "actionChannelSeparateTime": CommandMeta(U + "time_channel_dialog.py",
                              ("time", "window", "split", "exchange paint", "multiplex", "channel",
                               "convert overlay"),
                              "Separate the active dataset into an overlay of "
                              "filtered acquisition-time channels.", "process",
                              inputs=("active dataset with tim",),
                              outputs=("time-window overlay datasets",)),
    "actionChannelSeparateAttribute": CommandMeta(U + "attribute_separation_dialog.py",
                              ("attribute", "distribution", "mixture", "gaussian", "log-normal",
                               "gamma", "poisson", "convert overlay", "channel", "unmix", "efo", "cfr"),
                              "Convert a dataset to a multi-channel overlay by any MINFLUX "
                              "attribute's distribution (mixture fit → value-window channels).",
                              "process",
                              params=(ParamMeta("n_components", "int", 2, "", "mixture components"),),
                              inputs=("active MINFLUX dataset",),
                              outputs=("per-channel (+unassigned) overlay datasets",)),
    "actionRevertOverlay": CommandMeta(U + "main_window.py",
                              ("revert", "undo", "overlay", "original", "recombine", "unsplit",
                               "convert overlay", "channel"),
                              "Revert a separation overlay back to its single original dataset "
                              "(removes the channels; reconstructs if the source was closed).",
                              "process",
                              inputs=("active separation overlay",),
                              outputs=("original single dataset",)),

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
    "actionRoiRestore": CommandMeta(U + "main_window.py",
                              ("roi", "restore", "recover", "undo delete", "bring back", "sync", "views"),
                              "Restore the active ROI onto another view (render ↔ scatter), "
                              "or bring back the last active ROI after an accidental delete.",
                              "process"),
    # Convert / Fit sub-menu actions are tagged as a group (see _apply_command_meta).
    "_roi_convert": CommandMeta(C + "roi_convert.py", ("roi", "convert", "rectangle", "oval", "point", "line", "region"),
                              "Convert an ROI to another type.", "process"),
    "_roi_fit": CommandMeta(C + "roi_fit.py",
                              ("roi", "fit", "rectangle", "circle", "ellipse", "polygon", "convex hull",
                               "spline", "interpolate", "minimum enclosing", "circumscribed", "moment"),
                              "Fit a shape to the localizations a region ROI highlights "
                              "(or spline-fit / interpolate its outline).", "process"),
    "actionAggregate": CommandMeta(A + "aggregation.py",
                              ("aggregate", "localizations", "imspector", "trace", "photon", "binning"),
                              "Aggregate valid final MINFLUX localizations per trace using a "
                              "photon threshold and photon-weighted spatial centroids.",
                              "process", inputs=("active MINFLUX dataset",),
                              outputs=("aggregated dataset",)),

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
    # The HlyB/D subunit pair analysis is now a *plugin*, so its metadata lives
    # on the PluginEntry (name/tooltip/keywords) rather than here — this
    # registry is keyed by QAction attribute name, and a plugin action has
    # none.  See plugins/hlyb_pair_analysis/.

    # ---- Analyze › Trace ------------------------------------------------------
    "actionTraceSize": CommandMeta(A + "trace_analysis.py",
                              ("trace", "size", "spread", "cluster size", "localization spread"),
                              "Estimate the average per-trace size (log-distance Gaussian fit).", "analysis",
                              inputs=("active dataset (raw loc, tid)",), outputs=("trace size (nm)",)),
    "actionTraceAnisotropy": CommandMeta(A + "trace_analysis.py",
                              ("z scaling factor", "axial calibration", "anisotropy", "aspect ratio"),
                              "Estimate the Z scaling factor from raw last-valid trace sizes.",
                              "analysis", inputs=("active dataset (raw loc, tid)",),
                              outputs=("Z scaling factor",)),

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
                              inputs=("active dataset (loc)",), outputs=("detection rectangle ROIs",)),
    "actionSegConvolution3D": CommandMeta(U + "conv_segmentation_3d_dialog.py",
                              ("convolution", "matched filter", "shell", "ball", "3d", "detection",
                               "segmentation", "npc", "nuclear pore"),
                              "Genuinely-3-D matched-filter detection (shell/ball/gaussian/LoG) → 3-D point ROIs.",
                              "analysis", inputs=("active dataset (loc, 3-D)",), outputs=("3-D point ROIs",)),
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
                              outputs=("polygon contour ROIs",)),
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
    "actionTaskMonitor": CommandMeta(U + "task_monitor.py",
                              ("thread", "task", "background", "job", "worker",
                               "cancel", "stop", "kill", "monitor"),
                              "Background tasks and threads in flight; ask a "
                              "running task to stop.", "help"),
    "actionCommandFinder": CommandMeta(U + "command_finder.py", ("search", "commands", "finder", "palette"),
                              "Search all menu commands (Fiji-style).", "help"),
    "actionCheckUpdates": CommandMeta(C + "updater.py", ("update", "version", "release"),
                              "Check for a newer release.", "help"),
    "actionAbout": CommandMeta(U + "main_window.py", ("about", "version", "credits"), "About this application.", "help"),
}


def meta_for(key: str) -> CommandMeta | None:
    return COMMAND_META.get(key)
