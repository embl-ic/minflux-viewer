# Getting started

## Install

Download the archive for your operating system from the
[Releases](https://github.com/embl-ic/minflux-viewer/releases) page, unpack it
and start the application — the [README](../../README.md#installation-and-running)
has the per-platform details, and how to run from source.

## Open data

- *File › Open* (**Ctrl+O**), or drag files or folders onto the main window.
- *File › Open Recent* lists recently opened files.
- A Python script (`.py`) dropped on the main window opens in the Script Editor.
  Opening never runs it; press *Run* when you want to. A dropped folder opens
  the data in it, not its scripts.
- *File › Open Sample Data* opens a synthetic dataset — useful to try the
  application without your own data. The presets are managed in
  *Plugins › Data Simulator*.

Supported inputs include Imspector `.msr`, `.mat` (m2205 and m2410 layouts),
`.npy`, `.json`, `.csv` and other tables, and MINFLUX Viewer `.zarr` stores. See
the [README](../../README.md#input-formats) for the full list.

### Opening an Imspector `.msr` file

An `.msr` file opens in the **MSR Reader**, because one file can hold several
acquisitions, channels, images and bead (MBM) data. In the reader:

- *Datasets / Fields included…* chooses what to import;
- *Align channel* aligns the channels of a multi-colour acquisition with the
  fiducial beads, and *Show beads drift* shows how the beads moved;
- *Open in MINFLUX viewer* opens the chosen datasets — several channels either
  as one aligned multi-channel overlay or each as its own dataset.

An `.msr` file saved by MINFLUX Viewer itself skips the reader and reopens with
its saved processing.

### What happens after a dataset opens

By default the **Render** view and the **Dataset Information** window open,
and two quantities are computed in the background: the per-trace localization
precision (StdDev per trace) and the local density (neighbours within 100 nm).
Which windows open and what is computed are set in *Edit › Preferences ›
Data*. Long computations appear in *Help › Task Monitor*, where they can be
stopped.

## The main window

The main window holds the menus, the toolbar and a status bar; data is shown in
separate windows.

- **Menus.** *File* (open, save, export), *Edit* (Dataset Manager, Duplicate,
  Preferences), *View* (the data windows), *Process* (filters, channels, ROIs,
  aggregation), *Analyze* (measurement, precision and resolution, density,
  segmentation, tracking), *Plugins*, *Help*.
- **Toolbar.** Region-of-interest tools (rectangle, oval, polygon, freehand,
  line, angle, point, magnetic lasso), the lookup table (*LUT*) and colour
  pickers.
- **Search field** (right of the toolbar). Type a few letters of a command, its
  menu or what it does — for example `npc`, `drift` or `outlier` — and pick the
  command. *Help › Command Finder* shows the full searchable list.

The [command reference](../reference/commands.md) lists every command.

## Data windows

| Window | Opens with | Shows |
|---|---|---|
| Render | *View › Render*, **Ctrl+R** | the localization image (density reconstruction), with depth slider for 3-D data, scale bar and export |
| Loc Scatter Plot | **Ctrl+3** | the localizations as points (XY, XZ, YZ or 3-D), coloured by an attribute |
| Attribute Histogram | **Ctrl+2** | the distribution of one attribute; the place to set filter bounds by eye |
| Attribute Plot | **Ctrl+1** | any attribute against any other |
| Tracking View | **Ctrl+T** | trajectories played back as comets |
| Show Info | **Ctrl+I** | the dataset's metadata, acquisition time and summary numbers |
| Log | **Ctrl+L** | every processing step with its parameters |

Several datasets can be open at once. *Edit › Dataset Manager* (**Ctrl+D**)
lists them and sets the **active** dataset — the one that menu commands act on;
**Ctrl+Tab** / **Ctrl+Shift+Tab** cycle through datasets. Right-clicking a
dataset in the manager offers duplicating, combining into a multi-channel
overlay, mapping a confocal image signal, viewing bead (MBM) data, saving and
closing.

Most views have a right-click menu, including *Export figure…* to save the view
as a pixel image (PNG, TIFF, JPEG at up to 8× screen resolution) or a vector
figure (SVG, PDF).

## Save your work

- *File › Save* (**Ctrl+S**) writes a MINFLUX Viewer `.zarr` store: the raw
  data together with the processing (filters, ROIs, channel alignment, Z
  scaling), all in one folder.
- *File › Save As* writes the raw MINFLUX data as `.mat`, `.npy` or `.json`, a
  custom `.csv` table, or an experimental `.msr`. Processing that the format
  cannot hold is written next to it as `<name>_viewer_metadata.json`; when the
  data file is opened again, the viewer offers to apply it (it is never applied
  silently).

## Preferences and shortcuts

*Edit › Preferences* (**Shift+P**) has pages for files, data loading,
attributes, processing (including the Auto filter rules), tracking, appearance,
bead handling, keyboard shortcuts and plugins. Shortcuts can be changed on the
*Shortcuts* page; frequently used defaults:

| Action | Shortcut |
|---|---|
| Open / Save | Ctrl+O / Ctrl+S |
| Filter | Shift+F |
| Render · Attribute Plot · Histogram · Scatter | Ctrl+R · Ctrl+1 · Ctrl+2 · Ctrl+3 |
| Tracking View · Show Info · Log | Ctrl+T · Ctrl+I · Ctrl+L |
| Dataset Manager · Duplicate | Ctrl+D · Shift+D |
| Next / previous window | Tab / Shift+Tab |
| Next / previous dataset | Ctrl+Tab / Ctrl+Shift+Tab |
| Back to the main window | Shift+V |
| Close the current window | W |

## Next

- [Concepts](concepts.md) explains how the viewer represents MINFLUX data.
- [Filtering](filtering.md) covers cleaning a dataset.
