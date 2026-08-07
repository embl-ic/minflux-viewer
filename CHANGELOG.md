# Changelog

## v0.3.9

### Canonical MSR export and round-trip loading
- **MSR Reader exports now use the same writers as *File → Save*.** The reader previously had its own export path that flattened the nested `mfx.itr` structure into names like `itr_itr` and `itr_loc` — files the viewer could not read back. Every localization export is now the canonical flat m2410 representation with a top-level `itr` field, so **anything you export can be re-opened**.
- **`.zarr` is now a loadable format,** not just a write target. **Drag a canonical Zarr store onto the window** to open it, with strict validation so an MBM (bead) companion store is not mistaken for a localization dataset. (A Zarr store is a *directory*, so drag-and-drop is the working route — the *File → Open* dialog cannot select one.)
- **The MSR Reader export gained `.npz` and `.msr` checkboxes**, matching the formats available elsewhere.
- **Large JSON and CSV exports stream in bounded memory** instead of materializing the whole table — a multi-gigabyte export no longer scales its peak memory with the file size.
- Round trips verified for `.mat`, `.npy`, `.json`, `.csv` and `.zarr` across 11 MINFLUX datasets from the recursive sample-data set. Legacy image-only OBF `.msr` files remain non-exportable as MINFLUX datasets (they contain no localizations).

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
- **Plugins can now declare search keywords**, so the Command Finder finds them by synonyms that are not in their menu label.

### macOS
- **Fixed: dropping a file on the app icon opened a second copy of the viewer.** The bundle never told macOS which documents it opens, so the system had no running-instance handler and launched a new one; and nothing in the app handled the system's open-document request even when it did arrive. Both are fixed, and on top of that **only one viewer now runs per user** — a duplicate launch hands its files to the running window and exits immediately, so a second window cannot appear even if macOS ignores the request to prevent one (which it does for a relocated, unsigned, or stale-registered bundle). Set `MINFLUX_VIEWER_ALLOW_MULTIPLE=1` if you *want* several copies.
- Opening a file from Finder — drop, double-click, or *Open With* — reuses the window you already have. We register as an *alternative* handler for `.msr`, so an Imspector file association is left alone.
- Each open request is logged with the process id and how it arrived, so it is clear which window handled it.
- **New `scripts/check_macos_bundle.py`** checks a built `.app`: whether it actually contains the fix, whether the bundle keys landed, and whether stale copies of the app are confusing macOS.

### Command line
- **`minflux-viewer <path>` accepts every supported format and folders.** It previously recognised only `.mat`, `.npy`, `.csv` and `.msr` and silently ignored anything else, so `minflux-viewer data.json` opened nothing. Folders are now scanned the same way a dropped folder is.

### Fixes
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
