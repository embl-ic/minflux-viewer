# MINFLUX Viewer v0.4.0

Fluorescent image channels can now be **mapped onto localizations as an
attribute**, multi-channel overlays can be **aligned by hand inside the view**
(render *and* scatter), and the **Dataset Manager** has grown into the place you
manage datasets from.

---

## Fluorescent image channels on your localizations

Sample a confocal / fluorescent image at every localization and keep it as an
ordinary attribute — colour a render or scatter by it, histogram it, filter on
it.

- **Detection is conservative and always confirmed by you.** MINFLUX-generated
  stacks (data / trace / density / …) are excluded, and a channel is only offered
  when its complete calibrated X *and* Y bounds match the dataset's acquisition
  ROI to within 1 % on each axis independently. The viewer never guesses which
  channel is biologically interesting — it offers every geometric match and lets
  you choose.
- **Two entry points, one dialog:** the MSR reader's *Open in MINFLUX viewer*,
  and the Dataset Manager's per-row *Map confocal signal…*. Pick channels, name
  the attributes, choose 2-D or 3-D sampling and the interpolation
  (nearest / bilinear / bicubic, or nearest / trilinear in 3-D).
- **Manual alignment shows the composite you are aligning** — localizations plus
  every selected image, each independently shown, hidden and recoloured. Drag or
  use the arrow keys; comma and period rotate. Channels on the same calibrated
  grid share one transform, so aligning once aligns them all.
- **The value is genuinely per iteration.** The image is sampled at each row's
  own coordinates in the raw store as well as at the materialized table, so
  browsing iterations can change the value — because that iteration's position
  changed, not because the image did. Localizations outside the image are `NaN`,
  never zero.

## Manual overlay alignment, in the view

- **No more separate modal dialog.** Right-click a channel row › *Manual align*
  and the channel list is replaced in place by an alignment panel. Drag the
  channel with the mouse or nudge it with the arrow keys, comma / period to
  rotate, then *Apply* or *Cancel* — Cancel restores every channel's transform
  and visibility exactly.
- **Now in the scatter plot too**, with the same controls and wording.
- **Steps are physical, not zoom-dependent**: 1.0 nm translation and 0.1°
  rotation by default, remembered separately per view. A keyboard nudge means
  the same distance whatever the zoom.
- **The render preview is fast.** Alignment uses a dedicated preview path capped
  at 512 px on its longest side, with each channel's colour contribution cached
  so moving one channel does not re-colourize the rest, and rapid input coalesced
  into a single pending frame. On a three-channel 2500² view a preview frame
  averages ~28 ms, against ~1.37 s for the previous full-frame recomposition;
  Apply / Cancel returns to the exact full-resolution render.

## Dataset Manager

- **Multi-row selection** (Ctrl / Shift click). Selecting still never activates —
  the active dataset only changes on double-click or *Set active*.
- **Batch actions** on a multi-selection: *Close all*, *Duplicate all*, and
  *Combine as multi-channel overlay*, which opens the Combine dialog listing only
  the datasets you selected.
- **A per-dataset menu**: *Reset · Save as… · Close · Duplicate*, then
  *View mbm info… · View image series*, then *Map confocal signal…*. Entries that
  need a particular source are greyed out with a tooltip saying why, rather than
  vanishing.
- **Reset** puts a dataset back to how it was opened — filters, ROI masks, RIMF
  and the live view layer (LUT, manual alignment) — while keeping its overlay
  membership, because resetting one channel must not dissolve the group.
- **The bottom button is now *Close* and it advances the highlight**, so datasets
  can be closed one after another without going back to the table each time.
- **Drop a file onto a dataset row and it is applied to that dataset** — a
  different action from dropping on the main window, which opens a new dataset.
  A filter preset appends its rows to that dataset's filter, a ROI set attaches
  to it, a processing-metadata sidecar re-applies its recipe, and a TIFF is
  mapped as a confocal signal.
- **View mbm info…** shows a dataset's beam-monitoring beads without re-opening
  its `.msr`: drift traces and beads-vs-data-region, combined in one window.

## Canonical MSR export and round-trip loading

- **MSR Reader exports now use the same writers as *File → Save*** — every
  localization export is the canonical flat m2410 layout, so **anything you
  export can be re-opened**.
- **`.zarr` is loadable**, not just a write target (drag the store onto the
  window). `.zarr` export additionally writes the measurement's source store, so
  acquisition ROIs, the search grid, the used-bead list and the acquisition
  date / sequence / scan geometry are preserved.
- **Large JSON and CSV exports stream in bounded memory.**
- **Batch export runs in the background** with progress in the reader, its title,
  the status row and the Log, plus a *Stop* button; it can search sub-folders,
  mirror the input tree, and it exports image series as OME-TIFF.

## Also in this release

- **An all-NaN trace is missing data, not an error** — which matters now that an
  attribute can be a mapped signal that is `NaN` outside the image. Trace
  read-outs return `NaN` quietly, partly-mapped traces summarise the rows they
  have, and a fully unmapped trace simply fails a finite filter bound.
- **Rod-shaped cell segmentation (E. coli)** as a component mode of the HlyB/D
  pair analysis: cells are found by their tightly-constrained *width* via the
  mask's distance transform, so the method is rotation- and length-invariant,
  touching cells can be split, and each cell gets an oriented rectangle frame.
- **One image viewer per file** — opening several series of one `.msr` reuses a
  single window and switches its Series dropdown. The acquisition ROI can be
  toggled off.
- **One shared pure-colour LUT list everywhere** — Red, Green, Blue, Cyan,
  Magenta, Yellow, **Orange**, **White**, Gray, **Black** — instead of three
  different subsets in three different orders.
- **Fixed: on macOS a packaged build could show a second MINFLUX Viewer
  instance on startup**, because Zarr's compression layer spawns a
  multiprocessing helper through the app executable.

---

## Known limitations

- A dropped **TIFF** carries a pixel size but no stage position, so — unlike an
  `.msr` image, which is matched to the acquisition ROI geometrically — it is
  centred on the dataset's extent and the mapping dialog's manual alignment is
  there to correct it. An uncalibrated TIFF is refused rather than assumed to be
  1 nm/pixel.
- The **processing-metadata sidecar** accepted on a dataset row is a
  *prospective* format: it currently records only what *File → Save* needed
  (RIMF, transform, filters) and is expected to change.
- `.npz` remains **write-only** — use `.mat`, `.npy`, `.json`, `.csv` or `.zarr`
  for a round trip.

## Install

Download the build for your platform from the release assets.

- **Windows** — one-folder build; run `MINFLUX Viewer.exe`.
- **macOS** — unsigned, so the first launch needs right-click → *Open*, or:
  ```
  xattr -dr com.apple.quarantine "/Applications/MINFLUX Viewer.app"
  ```

From source (Python 3.10–3.12):

```
poetry install
poetry run minflux-viewer
```
