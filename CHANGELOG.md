# Changelog

## v0.4.1

This release makes colormaps an application-owned feature, removes Matplotlib and
its packaging chain, adds a single application-wide COLOR dialog, and brings the
Render and Localization Scatter context menus into the same Fiji-style layout.

### COLOR dialog

- **New application-wide COLOR dialog** (toolbar Color button) owning every
  configurable RGBA in `prefs["colors"]`, organized as four tabs — Solid Color
  List, Viewer / Plots, Components, Plugins — each with its own Reset, over a
  shared Custom Color Palette (basic/custom swatches, gradient, alpha bar, Pick
  Screen Color, preview block, HSV / HEX / RGBA fields).
- Fixed width set by the palette; a tab with more entries than fit scrolls
  instead of squeezing its rows.
- **Component and plugin color groups are wired, not decorative**: ROI Manager,
  Iteration series, Localization precision, Spatial Line Pattern, Drift
  Correction, Trace Viewer. `colors.runtime_component_colors()` lets a module
  read its group without threading `prefs` through the call chain.
- A group may nest one level (`{row: {item: color}}`) and is rendered as
  labelled rows; ROI Manager uses this for *ROI entries* / *ROI selected*.

### ROI colors

- The single ROI color is split into **face, edge, corner and highlight**, plus
  the ROI Manager group's per-state face/edge/corner/label.
- **Fixed: ROIs and ROI Manager labels could be invisible and could not be
  recolored.** Colors were serialized as Qt `#AARRGGBB` but read by PyQtGraph as
  `#RRGGBBAA`, so the blue byte became alpha and an opaque color arrived fully
  transparent. Writers now emit the PyQtGraph-compatible form, legacy values are
  repaired on draw, and the recolor pass compares through the same parser.
- Preferences ▸ Appearance ▸ ROI: *Edit widget size* renamed **Corner size**
  with an explanatory tooltip; the highlight checkbox no longer claims to follow
  the ROI color.

### LUT dialog

- **One LUT dialog application-wide**, rebound to the focused view and closed
  with the application. Previously each view created its own, the render window
  closed other views' dialogs on focus, and hidden instances outlived the main
  window.
- **Invert LUT now works on the localization render**, flipping the colormap and
  the background together (and ticking View ▸ White background). It previously
  applied only in TIFF/image mode.
- A custom colormap's **alpha now dims its channel**, matching solid colors; the
  render composite is opaque, so alpha scales intensity.
- The one-off **Custom...** entry is gone from the render and scatter Solid color
  menus — that list is the COLOR dialog's solid list. Saved `solid:custom:` LUTs
  still render.

### Spelling

- `colour` → `color` across user-facing text, identifiers and comments (the
  `v041_global_rgba_colours` migration key is deliberately unchanged, so the
  migration does not re-run against preferences that already have it).

### Colormaps and LUTs

- **Matplotlib is no longer a dependency.** Render, scatter, TIFF, volume,
  density, segmentation, straightened-volume and plugin views all use one local
  PyQtGraph colormap registry. The PyInstaller spec explicitly excludes
  Matplotlib and its otherwise-unused support packages.
- **Focused built-in map set:** `hot`, `jet`, `HiLo`, `glasbey`, `viridis`,
  `inferno`, and `gray`, plus the solid channel colors. Previously exposed
  `parula`, `turbo`, `magma`, `plasma`, and `cividis` remain resolvable for saved
  views but are hidden from new menus.
- **Persistent custom gradients** can be created, edited, and deleted from the
  LUT dialog's new **Custom** menu. They are stored in application preferences
  and become available to render, scatter, volume, and channel-LUT selectors.

### Render and scatter views

- **Localization Scatter gains Plot style** for marker shape, size,
  transparency, and custom color; the settings persist per dataset and apply in
  2-D and 3-D.
- **View controls are organized consistently.** Scatter puts XY/XZ/YZ/3D,
  background, axis, grid lines, and Plot style under **View**. Render puts axis
  and grid lines between background and Render Method.
- **Render's Colormap menu now mirrors Scatter's:** named maps, a separator, a
  **Solid color** submenu, and a separated **Custom...** color picker. Custom
  render colors keep the tonal intensity ramp instead of flattening the image.
- **The experimental Matplotlib 3-D Attribute Plot is removed.** Localization
  Scatter remains the supported spatial 3-D point viewer; Attribute Plot remains
  the supported arbitrary-attribute plot.

### Analysis and defaults

- **NPC ring detection is consolidated into Convolution segmentation.** The
  duplicate NPC 2-D command and unused 3-D placeholder are removed. The
  convolution ring model now includes the former detector's optional angular
  ring-support score, while legacy processing logs still generate method text.
- **Fresh installs apply no RIMF correction by default.** Both estimate and
  fixed-value modes start off, leaving Z as recorded until the user opts in.
  Existing saved preferences remain unchanged. Fresh defaults are now protected
  from accidentally running old one-shot migrations.

### Packaging

- The lock file no longer contains Matplotlib, contourpy, cycler, fonttools,
  kiwisolver, Pillow, pyparsing, python-dateutil, or six through the former
  plotting dependency.
- Windows builds use a native `.ico`, avoiding a build-time Pillow dependency.
- **The Windows executable carries a version resource** (FileVersion,
  ProductVersion, ProductName …), built from the same `__version__` the spec
  already parses. It previously had none: the version was applied only to the
  macOS bundle, so Explorer's Properties dialog showed nothing.
- The PyInstaller spec stops immediately with the build interpreter and missing
  package names if it is accidentally launched from an incomplete global Python
  environment. Its build instructions explicitly install and invoke PyInstaller
  through the project `.venv` interpreter.

See `RELEASE_NOTES_v0.4.1.md` for the user-facing release notes and compatibility
details.

---

## v0.4.0

Fluorescent image channels can now be mapped onto localizations as an attribute,
multi-channel overlays can be aligned by hand in the render *and* scatter views,
and the Dataset Manager grew into the place you manage datasets from — multi-
selection, per-dataset actions, and drag-and-drop of files onto a dataset.

### Fluorescent image channels mapped onto localizations
- **A confocal / fluorescent image channel can be sampled at every localization and kept as an attribute.** It then behaves like any other attribute: colour a render or scatter by it, histogram it, filter on it. Detection is deliberately conservative and always user-confirmed — MINFLUX-generated stacks (data / trace / density / …) are excluded, and an image is only offered when its complete calibrated X *and* Y bounds match the dataset's acquisition ROI within 1% on each axis independently. The viewer never guesses which channel is biologically interesting; it offers every geometric match.
- Reachable from **two places**: the MSR reader's *Open in MINFLUX viewer*, and the Dataset Manager's per-row *Map confocal signal…*. Both use the same dialog — candidate checkboxes with editable attribute names, 2-D or 3-D sampling, nearest / bilinear / bicubic (nearest / trilinear in 3-D), and automatic or manual alignment.
- **Manual alignment shows the composite you are aligning**: localizations plus every selected image, each independently shown/hidden and recoloured, moved by drag or arrow keys with comma/period to rotate. Channels sharing one calibrated grid share one transform, so aligning once aligns them all.
- **The mapped value is genuinely per iteration.** The image is sampled at each row's own coordinates in the raw store as well as the materialized table, so browsing iterations can change the value — because that iteration's position changed, not because the image did. Localizations outside the image are `NaN`, never zero.

### Manual overlay alignment in the view (render and scatter)
- **Aligning overlay channels no longer happens in a separate modal dialog.** Right-click a channel row › *Manual align* and the channel list is replaced in place by an alignment panel: pick a channel, drag it with the mouse or nudge it with the arrow keys, comma/period to rotate, then *Apply* or *Cancel*. Cancel restores every channel's transform and visibility exactly.
- **Now available in the scatter plot too**, not only the render view, with the same controls and wording.
- **Steps are physical, not zoom-dependent** — 1.0 nm translation and 0.1° rotation by default, remembered separately per view. A keyboard nudge means the same distance whatever the zoom.
- **Render alignment now uses a fast interactive preview.** The temporary overlay is capped at 512 px on its longest side, keeps stationary channels as cached 8-bit colour contributions, and coalesces rapid drag/key events to one pending frame; Apply/Cancel returns to the exact full-resolution render. On a three-channel 2500² synthetic view, a changed-channel preview averaged ~28 ms (p95 ~31 ms), versus ~1.37 s for the former full-frame recomposition.

### Canonical MSR export and round-trip loading
- **MSR Reader exports now use the same writers as *File → Save*.** The reader previously had its own export path that flattened the nested `mfx.itr` structure into names like `itr_itr` and `itr_loc` — files the viewer could not read back. Every localization export is now the canonical flat m2410 representation with a top-level `itr` field, so **anything you export can be re-opened**.
- **`.zarr` is now a loadable format,** not just a write target. **Drag a canonical Zarr store onto the window** to open it, with strict validation so an MBM (bead) companion store is not mistaken for a localization dataset. (A Zarr store is a *directory*, so drag-and-drop is the working route — the *File → Open* dialog cannot select one.)
- **The MSR Reader export gained `.npz` and `.msr` checkboxes**, matching the formats available elsewhere.
- **Large JSON and CSV exports stream in bounded memory** instead of materializing the whole table — a multi-gigabyte export no longer scales its peak memory with the file size.
- Round trips verified for `.mat`, `.npy`, `.json`, `.csv` and `.zarr` across 11 MINFLUX datasets from the recursive sample-data set. Legacy image-only OBF `.msr` files remain non-exportable as MINFLUX datasets (they contain no localizations).

### MSR Reader — batch export
- **Batch export runs in the background with a progress read-out.** It previously ran on the UI thread, so a multi-file export froze both the reader and the whole viewer until it finished. It now reports progress in four places — a progress bar with a `done / total · current file` label in the reader, the reader's **title** (`MINFLUX .msr Reader & Converter (batch export 11.5%…)`, legible when the window is behind another), the **main-window status row**, and the **Log** progress bar — the same way *Particle Average* does. A **Stop** button ends the run after the file in flight, and closing the reader cancels it. Measured on a 3-file export: the export call returns in 0.12 s instead of blocking for 56 s, and the UI keeps servicing events throughout.
- **Fixed: batch export silently did nothing when the `.msr` files were in sub-folders.** The search was top level only, so pointing it at a project root found no files, wrote one line to the Log and returned — with no visible response from the *OK (export)* button. It now reports an empty search in a dialog (naming the folder and pointing at the new *recursive* option), and finishes with a summary dialog listing any files that failed.
- **New `recursive` checkbox** beside the input field: search the folder and all sub-folders. Off by default, so existing behaviour is unchanged.
- **New `reproduce input folder structure` checkbox** beside the output field: mirror each input's path below the output folder. Off by default.
- **Every input file now exports into its own folder, named exactly after the `.msr`** — `root/a/b/test.msr` → `output/test/…`, or `output/a/b/test/…` when mirroring. Exported files are named after the *datasets inside* the file, so without this the outputs were not attributable to their source, and a recursive search could have two files overwrite each other. If two inputs would still land in one folder (same file name in different sub-folders, mirroring off), the run stops before writing anything and says so.
- **Image series are exported.** Series ticked in *Datasets / Fields included…* are written as OME-TIFF into the same per-file folder, through the same writer the render view's *Export to TIFF…* uses — so pixel calibration round-trips into the viewer's TIFF reader. Previously the image selection was only used for *Open in MINFLUX viewer* and **no image was ever exported**. In batch the selection is carried by series **name**, since raw stack indices address one file's stack table and mean nothing in another.
- **Exported images keep the name the `.msr` uses.** `Ch1 {1}` is written as `Ch1 {1}.tif`, not `Ch1__1_img.tif` — only characters no filesystem accepts are replaced (so `MF(<run>)/density/loc` becomes `MF(<run>)_density_loc.tif`). The old sanitizer rewrote spaces and braces too, which made an exported image impossible to match back to the channel Imspector shows.
- **New "Include all images" master checkbox** in *Datasets / Fields included…*, above the individual series (laid out like a dataset panel). It is the selection most exports want, and — unlike the per-series ticks, which address one file's stack list — it carries across a **batch of files whose channels differ**: each file contributes whatever image series it has. The individual ticks remain for single-file work.

### Dataset Manager — multi-selection and a repeatable Close
- **Rows are now multi-selectable** (Ctrl / Shift click). Selecting is still not activating: the active dataset only changes on double-click or *Set active*, so picking several rows never retargets the rest of the app.
- **Right-click inside a multi-selection for batch actions** — *Close all*, *Duplicate all* (one plain copy per selected dataset), and *Combine as multi-channel overlay*, which opens the usual *Process › Channel… › Combine* dialog **listing only the selected datasets**, all pre-checked. Right-clicking any other row (or with a single row selected) keeps the per-dataset *Save as… / Map confocal signal…* menu.
- **The bottom button is now *Close* (was *Remove*), and it advances the highlight**, so datasets can be closed one after another without going back to the table to pick the next one. Closing the top row keeps the highlight at the top and walks *down* the list; closing any other row moves it to the entry **above** the closed one — so closing the bottom row walks *up*. It closes the whole selection when several rows are selected.
- Closing from the manager now goes through the same path as `Ctrl+W`, so closing a channel of an overlay leaves a lone survivor rendering as a standalone dataset instead of keeping stale overlay state.

### Dataset Manager — per-dataset actions and drop-on-a-row
- **The single-row right-click menu now covers the whole per-dataset workflow**: *Reset · Save as… · Close · Duplicate*, then *View mbm info… · View image series*, then *Map confocal signal…*. Entries that depend on the source are **greyed out with a tooltip explaining why**, rather than disappearing — so it is visible that, say, *View mbm info…* needs a dataset that carries bead data.
- **Reset** puts a dataset back to how it was opened, data and view: filters cleared, its ROI selection masks dropped, RIMF restored to the value established when it loaded, and the live view layer (LUT, manual channel alignment) reverted to what the import recorded. Its **overlay membership is kept** — resetting one channel must not dissolve the group; use *Close* or *Combine* for that. If nothing had changed, it says so instead of claiming a reset.
- **View mbm info…** opens the beam-monitoring beads of one dataset without re-opening its `.msr` — the beads travel with the dataset. The two bead views that previously existed only inside the MSR reader, and only while a file was parsed, are combined here as **two tabs of one window**: *Drift* (per-bead time-coloured drift traces) and *Beads & data region* (absolute bead positions against the localization extent, with a per-bead total-drift table). Both are read-only — the alignment controls are hidden, since there is no alignment for a bead selection to feed.
- **Fixed: *View mbm info…* failed on essentially every dataset** with "has an MBM points array but no bead trace could be reconstructed from it". Bead traces were only reconstructed from the bead *name map*, which the MSR import never carried into the dataset — but the bead ids are in the points array itself, and are now the last-resort source (such beads are named by their id). The import also carries the name map and the used-bead marking through now, so real R-IDs survive.
- **View image series** opens that dataset's source `.msr` images in the shared image viewer, pre-selected to the series rendered *from this dataset*.
- **Drop a file onto a dataset row and it is applied to that dataset** — a different action from dropping on the main window, which opens a file as a new dataset. Four kinds are understood, with `.json` resolved by content rather than by extension: a **filter preset** appends its rows to that dataset's filter, a **ROI set** (native JSON, ImageJ `.roi`, or a RoiSet `.zip`) loads into the ROI Manager attached to that dataset, a **processing-metadata sidecar** re-applies its recipe (RIMF, transform, filters), and a **TIFF** is mapped as a confocal signal. The hovered target is named in the manager's info line while you drag. Anything else is refused and reported, rather than quietly opening as a new dataset and ignoring the row you aimed at.
- **A dropped TIFF can be used as a confocal channel.** Note the honest limitation: a TIFF carries a pixel size but *no stage position*, so — unlike an `.msr` image, which is matched to the acquisition ROI geometrically — the image is **centred on the dataset's own extent** and the mapping dialog's manual alignment is there to correct it. An uncalibrated TIFF is refused rather than assumed to be 1 nm/pixel.
- The processing-metadata sidecar is being used as the **prospective** format for a saved processing recipe and is expected to change; it currently records only what *File > Save* needed. Tracked at the top of `BACKLOG.md` ▸ *Essential*.

### Process › Channel… › Combine
- **Fixed: a dataset was left behind in its own standalone view after combining.** With *keep source dataset* unchecked the selected datasets become channels of one overlay in place, but each one's existing render/scatter window stayed open, still drawn and titled as a lone dataset — so after combining two datasets you were left looking at the overlay *and* at one of its channels as if it were still separate. Those windows are now folded into the overlay view. No dataset is closed: combining changes how they are viewed, not what exists, and a histogram or attribute plot of a single channel stays open because it is still meaningful. If only a member had a coordinate view, the anchor is given one rather than leaving you with none. *Keep source dataset* is unaffected — the overlay is built from copies there, so the originals and their windows stay standalone.

### Image viewer — an active ROI that lives in the image file
- **The image viewer now shows, edits and stores one active ROI, ImageJ-style.** ImageJ keeps an image's active ROI inside the TIFF (the `IJMetadata` tag) and restores it on open; we now do the same, in both directions — a TIFF written here opens in ImageJ/Fiji with its ROI, and a TIFF ImageJ wrote shows its ROI here. Deliberately **one ROI per image**, which is what the format holds: drawing replaces it, and there is no ROI Manager involved. Right-click the image for **ROI › Draw Rectangle / Draw Oval / Delete ROI** and **Save As TIFF…**; the ROI is listed in the info line and in the Info window. ROI types the viewer cannot draw (polygon, line, …) still display, move, and survive a save.
- **Exported `.msr` image series carry the MINFLUX acquisition rectangle.** Imspector stores the rectangles the operator drew to place each MINFLUX run — in the measurement, not on any image — and redraws them on whichever image covers them. We now do the same: each exported image gets its run's acquisition area as the active ROI, named after the run, and opening an OBF series in the image viewer shows it without exporting anything. The mapping from stage metres to image pixels comes from the stack's own scan geometry and was checked against the data: projecting a run's localizations onto its own confocal image puts them on **3.3–10.7× background** signal, while a flipped Y axis puts them on background.
- **One run per image, one ROI per image.** A MINFLUX run is spread over 3–4 *overlapping* rectangles, so the active ROI is the box they span, matching the run's localization extent to ~50 nm and named after the run. An image wide enough to cover **several** runs gets **no** ROI: merging them produced one box spanning mostly empty field, which marked nothing.
- **Not to be confused with Imspector's other rectangle.** Imspector also draws the *scan region of the next, zoomed image* on its parent overview — a different object, 3–10× larger, unrelated to where the localizations are. It is another image's `ExpControl/scan/range`, not a ROI record. If that is the box you are comparing against, it will not match.
- **Caveat:** the ROI is read by ImageJ's own TIFF reader. If Fiji routes the file to Bio-Formats instead (which reads ROIs from OME-XML, where tifffile cannot write them), the image opens without it.

### MSR Reader — image series in the parsed tree
- **Confocal-channel candidates can be mapped directly onto MINFLUX localizations.** The reader and Dataset Manager exclude known data/trace/density/generated stacks, then offer every scalar image whose full calibrated X/Y bounds match the run's acquisition ROI within 1% on both axes. The user chooses channels and editable attribute names; 2-D nearest/bilinear/bicubic mapping uses a float64 Z sum, while calibrated 3-D nearest/trilinear mapping samples ZYX directly. Out-of-bounds values are `NaN`, and source geometry/interpolation/alignment provenance is retained on each attribute.
- **Manual confocal alignment has keyboard sub-pixel and rotation controls.** Arrow keys translate the localization overlay by 0.5 pixel; Alt+Left/Right rotates it 0.1° counter-clockwise/clockwise about the image centre before signal sampling.
- **The parsed-contents tree now lists the file's image series.** Confocal/overview channels appear in one `Image series (N)` node beside the datasets, last; a density or trace render names the MINFLUX dataset it was computed from, so it is nested **under that dataset**, parallel to `mfx`/`grd`. Both groups start collapsed. Right-click → *Open as image* / *Preview* works on them as it already did for image-only files.
- **Fixed: the MINFLUX data blob was listed and exported as an image.** Imspector stores the MFXDTA localization container as a plain `uint8` OBF stack, and in real files that stack is declared **2-D** — a near-square block of bytes (e.g. `7301 x 7301`) that shape alone cannot tell from a real image. Every such blob was offered as an image series and, if ticked, written out as a multi-megabyte TIFF of raw container bytes (12 MB of them in one exported acquisition). The stack's `minflux` footer tag is now the discriminator: `type: data` is localization payload, `density`/`trace` are genuine images. Same fix makes `msr_is_image_only` correct for these files.
- **`.zarr` export now also writes the measurement's source store**, as `<name>_mfxdta.zarr` beside the canonical `<name>_mfx.zarr`. The embedded MFXDTA container holds several things the flat localization table cannot: the **acquisition ROIs**, the **search grid** (148 hexagonal targets at 180.6 nm pitch in the reference file), **which beads were used**, and the **acquisition date, MINFLUX sequence and scan geometry**. Unpacking it costs one decode layer and gives a plain zarr v2 store any tool can open, so none of that is lost. Only `.zarr` does this — a byte container has no meaningful `.mat`/`.csv` form — and the containers are otherwise neither listed in the parsed tree nor exported separately.
- **Fixed: a `.msr` listing the same dataset label twice aborted the export and lost data.** The export looked datasets up in a name-keyed map, so of two entries sharing a label only one survived — and the duplicate then tripped a "names collide" error that stopped the file part-way, dropping every dataset after it. On a real acquisition that silently cost the file's largest dataset (3.25 M localizations) and exported a 316-row array in place of the intended 279 k one. Datasets now carry their own arrays through the export, and a repeated name is written with a numeric suffix and a Log line rather than aborting.
- **The input and output fields accept dropped files and folders.** Dropping a `.msr` selects single-file mode and parses it; dropping a folder switches to Folder (batch) mode. A dropped file of any other type is rejected outright rather than being pasted in as a `file:///…` URL.
- **Removed `MINFLUX (.msr)` from the export formats** — saving `.msr` from the reader is available through *File → Save* like every other format.
- **`.npz` is labelled write-only.** It has no loader, so a `.npz` export cannot be re-opened in the viewer; it remains useful as a compact, dtype-preserving archive for `numpy.load()` elsewhere. Use `.mat`, `.npy`, `.json`, `.csv` or `.zarr` for a round trip.

### Image viewer
- **One image viewer per file.** Opening several series of one `.msr` reuses a single window and switches its Series dropdown, instead of stacking one window per stack. The first selected series is the one shown.
- **The acquisition ROI can be toggled off** with a checkbox next to the series selector, and the series selector no longer drifts to the right of the row on a 2-D image.

### Rod-shaped cell segmentation (E. coli)
- **New `rod` component mode in the HlyB/D pair analysis**, replacing generic spatial linkage with actual cell delineation. Cells are found by their *width*, which for E. coli is tightly constrained (~800–1100 nm) while length is not: the mask's Euclidean distance transform encodes local half-width directly, so the method is rotation- and length-invariant with no oriented template bank. Touching cells can be split at thin bridges, and each accepted cell is fitted with a minimum-area oriented rectangle whose long axis becomes the cell frame. The sensitivity audit varies the width prior (×0.9 / ×1.0 / ×1.1) alongside the existing knobs, and the run's method text reports how many regions were accepted.

### Trace read-outs and missing data
- **An all-NaN trace is treated as missing data, not as an error.** This matters now that an attribute can be a mapped fluorescent signal, which is `NaN` wherever a localization falls outside the image: a trace entirely outside the image has no value, and a partly-mapped trace should summarise the rows it does have. Every trace read-out (mean / median / min / max / stdev / range) returns `NaN` for empty or all-NaN input without emitting a warning, and a fully unmapped trace simply fails a finite filter bound and is excluded. Histogram aggregation, filter preview/apply and persisted-filter re-evaluation all share this one implementation.

### Other
- **Pure-colour LUTs are one list everywhere** — Red, Green, Blue, Cyan, Magenta, Yellow, **Orange**, **White**, Gray, **Black** — shared by the LUT dialog, overlay channels, DCR channel separation and the scale-bar colour pickers, which previously offered three different subsets in three different orders. Gray is now a true mid-grey rather than white.
- **Mapped image attributes appear in attribute dropdowns** without needing to be enabled in Preferences ▸ Attributes: an attribute created by the user is user-visible by construction.
- **Fixed: on macOS, a packaged build could show a second MINFLUX Viewer instance on startup.** Zarr's compression layer creates a multiprocessing lock on first import, which spawns a helper process through the app executable; without a `freeze_support()` dispatcher ahead of the application import, that helper ran the normal GUI entry point.

## v0.3.9

### Voronoi density rendering (2-D and 3-D)
- **New "Voronoi density" renderer** in the advanced render view. Each localization is assigned the density of its own Voronoi cell (multiplicity ÷ cell area), so the image **adapts to the local sampling density** instead of to a fixed pixel or blur width — no bin size and no sigma to choose. It recomputes for the current filters and depth selection.
- **True 3-D Voronoi volumes.** Opening the 3-D view from a Voronoi render builds a genuine XYZ Voronoi field (multiplicity ÷ cell *volume*), not a stack of 2-D projections. Works for multi-channel overlays.

### Histogram
- **Right-click Zoom tool** — *horizontal*, *vertical* or *unconstrained* drag-to-zoom, plus *Reset View*. It is one-shot (the next drag pans again), and the guide rides at the cursor so you can line it up against the bars. Horizontal and unconstrained zooms **re-bin and re-fit the height**, so a zoomed peak fills the plot instead of staying squashed.
- **Fixed: the floating "A" button zoomed further out on every click** (x +26 %, y +42 % over six clicks, compounding). It now performs the same deterministic fit as *Reset*.
- **Choose which read-outs you see** in *Preferences → Appearance → Histogram Plot*: which trace values (*mean median min max 1st last stdev range*) appear in the **As** dropdown, and which pooled-iteration values (*flatten stacked sum average*) appear in **Iter**. Both apply live.
- **New `trace 1st` / `trace last` read-outs**, and the Filter dialog now offers the same eight trace read-outs as the histogram.

### Removed: photon (`eco`) weighting in histogram and filter
Validated against a real two-colour ratiometric `.msr`, the option only had a statistical basis for `dcr`. Localization positions did **not** follow the expected 1/√N precision scaling (measured +0.17…+0.49 against a predicted −0.5 — the scatter is drift-dominated), and the effect was 0.09 nm median anyway; `cfr` is measured at a single iteration; and weighting a photon count by itself inflated it by 14.7 %. The checkbox, the *Weighted* filter column and the saved-preset key are gone; a stale key in an old preset is ignored, not honoured. **The ECO-weighted DCR channel separation is unaffected.**

### Plugins
- **Implemented for a user project a custom plugin as "HlyB/D subunit pair analysis"** (*Plugins* menu). Running it against a dataset produces a full Methods account in *Plugins → Generate Method Text*, including every parameter and a definition of each reported term.
- **The HlyB/D analysis can now delineate cells by rod detection** instead of by neighbour linking. Choose *Spatial components → Rod cell detection* and state the cell width (E. coli: ~800–1100 nm). Cells are found in the XY projection through the distance transform of the density mask — whose ridge is the local half-width — so detection is independent of cell length and orientation, with no template bank. This addresses two things neighbour linking could not: the conditional null now uses each cell's **measured long axis** rather than a per-component principal axis (which is the part that goes wrong on a fragment or a clump), and objects of the wrong size or shape are **rejected and reported** rather than silently analysed as one cell. Cells joined only by a thin bridge in the mask are separated; cells that overlap in projection while running parallel cannot be separated by any 2-D method and are rejected by the width gate instead. Detected cells are drawn in the result view, every region's measured width is listed — rejected ones included — and the sensitivity audit varies the width window in place of the link distance.
- **The cell-delineation bridging length now follows the labelling sparsity.** In MINFLUX data the scale that decides whether one cell is found as one cell is set by how far apart the labelled positions are, not by the optics — and it is estimated from the spacing of the inferred sites rather than from the localizations, which pile up tens deep at each molecule and badly understate it. On a reference single-cell acquisition (~320 sites on a 1020 × 3477 nm cell) the previous fixed length recovered only three disconnected patches ~450 nm across; the adaptive length recovers the whole cell. It is reported in the result and can be overridden.
- **Plugins can now declare search keywords**, so the Command Finder finds them by synonyms that are not in their menu label.

### Fixes

- **macOS MSR parsing no longer launches phantom viewer instances.** The frozen entry point now diverts PyInstaller multiprocessing helper processes before importing the GUI. This prevents the resource tracker started by `numcodecs.blosc` during `.msr` parsing from entering the normal application startup path, including the apparent relaunch during shutdown.
- **A single-channel `.msr` import no longer shows as "Overlay"** in the Dataset Manager. It is reported as "Own", and the provisional channel grouping the importer assigns before it knows the channel count no longer survives.
- **Editing Preferences no longer mutates the built-in defaults** for the rest of the session (a shallow copy meant the preference dictionary *was* the defaults).
- **Unknown trace aggregation modes now raise instead of silently returning wrong-length data** — this had been quietly breaking `trace 1st` / `trace last` and `trace median` in some paths.

---

## v0.3.8

*Wires the v0.3.7 modules into the menus, plus this release's rendering / ROI / MSR work.*

### Rendering
- **Switch render engine from inside the window** — right-click *View → Render Mode → Basic / Advanced*. The window swaps engines in place, keeping your zoom, orientation, window geometry **and any ROI draft you were drawing**. The standalone *Render View (advanced)* menu entry is gone; exactly one render window (one engine) exists per dataset.
- **Advanced rendering is now interactive.** The localization-precision Gaussian is vectorized (≈10–16× faster, identical output), a view change paints an instant coarse preview and then **fills in tiles progressively** instead of blocking until the whole frame is ready. Each reconstruction method (*Histogram · Bilinear · Bicubic · Basic · Fixed Gaussian · Localization-precision Gaussian*) carries hover help explaining what it does.
- **Fixed: a wide dataset opened with its left and right edges cut off.** The initial fit ran before the window was on screen, so it used a placeholder size. It now re-runs once at the real size — one-shot, so re-raising a window keeps your zoom.
- **3-D volume:** voxelization follows the 2-D render method; true per-localization precision in 3-D; an anti-alias blur floor that fixes the blocky look when the voxel cap forces a coarse grid; a GPU-aware *Max dim* control; and *Black % / White %* brightness-contrast.

### ROI
- **New *Process → ROI → Fit*** — fit a *Rectangle, Circle, Ellipse, Polygon, Convex Hull* or *Spline* to **the localizations a region ROI highlights** (distinct from Convert, which fits the ROI's own outline), plus *Interpolate* to resample an outline at a given spacing.
- **New *Process → ROI → Restore ROI*** — put the active draft onto the dataset's other open views, or bring back the last draft after an accidental delete.
- **Duplicate / Crop (Shift+D)** defaults tuned, now works on the advanced render view, and its Z-distribution plot is vertically expandable with a hover read-out giving the exact bin and count.

### Segmentation & analysis
- **Curvilinear structure segmentation** (*Analyze → Segmentation → Curvilinear Structures*): Frangi/Sato ridge filtering or a point-spline, then skeletonization into traced centre-line ROIs, with a live preview and filament-width profiling.
- **Menu entries for the v0.3.7 modules** — OME-Zarr export, time-window channels and ROI fitting became reachable here.

### MSR reader
- **Single-channel *Align channel* no longer asks about save outputs** — there is nothing to register. It shows the beads against the data region plus a per-bead drift table instead.
- **The data-region box is drawn as a translucent, labelled region**, so it stays visible when the fiducial beads are spread far wider than the measured data.

### UX
- **Menu section separators are visibly drawn** (Fiji-style) — the native etch was near-invisible on some themes and over RDP.
- **The main window opens toward the top-right of the active monitor**, and the MSR reader opens in the opposite corner, so it no longer lands on top of the main window and Log.

---

## v0.3.7

*Aggregation and drift correction, plus four self-contained modules whose menu entries arrive in v0.3.8.*

### Aggregate Localizations
- **New *Process → Aggregate Localizations…*** — reproduces Abberior Imspector's post-processing. The `aggN` in an Imspector filename is a **photon threshold, not a point count**: each trace is walked in time order, accumulating photons until the threshold is reached, then emitting one mean-position point. Validated end-to-end against reference raw/aggregated file pairs (98.99 % exact photon match, 0.27 nm median position error, 100 % begin/end flags).
- Non-destructive — the result is a new `(aggregated N)` dataset. **Multi-channel overlays aggregate every channel** at the same threshold and re-link the results into a new overlay. The threshold is remembered across sessions.

### Drift Correction (plugin)
- **New *Plugins → Drift Correction*** — time-window auto-correlation drift correction (2-D and 3-D), a faithful reimplementation of pyMINFLUX (Ostersehlt et al. 2022). Estimate the trajectory, inspect it, and if satisfied create a **new corrected dataset** so you can compare before and after. The time window can be set or left on *Auto*.

### Localization precision
- **CRLB now uses a per-dimension photon budget.** The materialized `eco` is the photon count at the final *axial* iteration — correct for σ_z but too few for σ_xy, whose photons come from the final *lateral* iteration. The lateral count is now detected and used, lowering median σ_xy from ≈2.47 to ≈2.03 nm on the reference file (σ_z unchanged). Single-`eco` datasets behave exactly as before.

### Other
- **Multi-channel 3-D volumes** — an overlay now composites every channel into one RGBA volume on a shared grid, each in its own LUT colour, instead of rendering only the anchor.
- **New *Plugins → Spatial Pattern Analysis along Line Profile*** — directed repeating-pattern analysis along a line or curved-centreline ROI.
- Groundwork landed for **OME-NGFF 0.5 / Zarr v3 pyramid export**, **time-window channels** and **ROI shape fitting**; their menu entries arrive in v0.3.8.

---

## v0.3.6

### Two-channel bead alignment — quality feedback & interactive bead selection (MSR reader)
The **"Beads and alignment result"** window (from *Align channel* / *Show beads drift*) now tells you how good the fiducial-bead registration actually is, and lets you tune it:
- **Fit-quality banner** — shows the alignment **RMSE** (XY and Z) and matched-bead count per channel, and turns red with a ⚠ warning when the registration is poor (the beads disagree with each other and no single transform can register the channels well). Opening an overlay or running *Align channel* with a poor bead fit now **warns you** instead of silently applying it.
- **Per-bead residual table** — one row per bead showing the raw offset (Δ) and the leftover error after the fit (residual), worst-first, with a plain-language **Comment** column that evaluates each bead to help you decide whether to keep or exclude it.
- **Interactive include/exclude** — tick/untick a bead's checkbox to add or remove it from the fit; the transform, the RMSE, and the aligned bead positions all **update live**. Your selection carries over to the next align/open.
- **Bead IDs are labelled** next to each bead (2-D and 3-D), and **excluded beads are shown faint** so you can see at a glance which beads are out of the computation.
- The plot / table split is **draggable**, so you can give the table more room to show all beads.

### ROI conversion — enclosing-shape family
- **Convert any ROI to a fitted shape:** axis-aligned **bounding box**, **minimum-area oriented rectangle**, **axis-aligned enclosing ellipse**, **minimum-area oriented ellipse**, **convex hull**, **region ↔ line**, **skeletonize**, and **to point** — each guaranteed to enclose the source outline.
- **Fixed:** converting a *stored* ROI (e.g. freehand → oval) now updates the shape shown on screen instead of leaving the old outline behind.

### Particle averaging
- **Template-free averaging now stops automatically when it converges** (new *convergence tolerance* control), rather than always running a fixed iteration count — faster, with the max-iterations spin now acting as a safety cap. The result label reports the actual iterations and whether it converged.

---

## v0.3.4

### Particle averaging — major overhaul
- **New unbiased "NPC two-ring model fit" method.** Fits a canonical 8-corner two-ring model directly to each particle's raw 3-D localizations (maximum-likelihood; diameter, inter-ring, rim, tilt, phase and centre fit jointly) — no sub-unit pre-detection, no Z-peak ring assignment, no phase heuristic. Reports **per-particle diameter / inter-ring / rim / tilt** so you can study how a parameter varies across conditions (e.g. ring diameter vs. wild type).
- **Per-particle fit table for every method** (template-free / template-provided / model fit), **sortable** by any column.
- **Histogram + interactive range filter** on any fit column (GoF, diameter, rim, …), combinable across columns, with a live "N of M selected" count.
- **Rebuild the average from a filtered sub-selection** — re-pools only the selected particles into a new view; near-instant (reuses the already-computed transforms, no re-fit).
- **Per-particle inspector** — open any particle to see its point cloud (2-D or 3-D) with the fitted model overlaid, plus toggleable **axis** and **bounding box** matching the 3-D scatter view.
- **Trace grouping (default on):** collapse each MINFLUX trace (`tid`) to one point per molecule-blink before averaging — de-biases long/bright molecules and speeds the fit.
- **LoG sub-unit fit seeding (default on):** the geometry fit is initialized from Laplacian-of-Gaussian sub-unit detection for faster, more robust convergence.

### Auto-update
- **One-click download → in-place install → restart** in the update dialog, now on **Windows *and* Linux** (macOS shows a download link — in-place update there needs code-signing).
- **Restart is user-confirmed** ("Restart now / Later"), Fiji-style; a deferred download is reused without re-downloading.
- **On-startup update check is now opt-in (off by default)** — enable it in the update dialog or *Preferences → File*. No network request on startup unless you choose it.

### Builds
- **One cross-platform PyInstaller spec** — the same build command produces a Windows folder, a macOS `.app`, or a Linux folder.

### Cleanup
- Removed the redundant **NPC detection / NPC average (Wanlu)** menu commands and the Wanlu two-ring averaging method — superseded by the unbiased model fit above (the sub-unit clustering it relied on is kept and reused for fit seeding).

---

Earlier releases: see the [Releases page](https://github.com/embl-ic/minflux-viewer/releases).
