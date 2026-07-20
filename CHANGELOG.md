# Changelog

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

Earlier releases: see the [Releases page](https://github.com/EMBL-ICLM/minflux-viewer/releases).
