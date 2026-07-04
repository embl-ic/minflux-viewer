# Changelog

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
