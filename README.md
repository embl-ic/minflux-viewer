# MINFLUX Viewer

MINFLUX Viewer is a desktop application for viewing, filtering, processing, and
analysing MINFLUX microscopy data. It reads common MINFLUX data
formats (for example `.mat`, `.npy` and `.json`), including Abberior Imspector
`.msr`, on Windows, macOS and Linux. Its Qt interface supports data exploration,
attribute-based filtering and processing. Processed data can be exported to
other applications, including an installed ParaView. The interaction model is
intended to feel familiar to ImageJ/Fiji users.

Developed at the [EMBL Imaging Centre](https://www.embl.org/about/info/imaging-centre/)
as a Python/Qt reimplementation of a previous in-house built MATLAB toolset.

---

## Input formats

- Abberior Imspector `.msr` — modern multi-dataset and also legacy
  formats, including any embedded confocal/STED images and
  bead/fiducial reference data (mbm). Read on Windows, macOS, and Linux.
- `.mat` (Imspector `m2205` and `m2410`), `.npy`, `.csv`, `.json`, `.npz`,
  `.zarr`, and spreadsheet tables.
- Files are opened from the menu or by drag-and-drop. Multiple datasets as multi-channel data are supported as overlay, but need to be imported as the original .msr raw data.

MINFLUX data exists in several structural versions (`m2410`, `m2205`, and older
*legacy* layouts) and container formats that have changed across Imspector/MINFLUX releases. The common ones are supported. A file that does not open correctly can be reported on the [issue tracker](https://github.com/embl-ic/minflux-viewer/issues) with a minimum sample, so the parser can be extended to cover it.

## Visualization

- Render: 2-D localization-density reconstruction with colormaps, a depth slider
  for 3-D data, selectable background, a scale bar, and a 3-D volume preview.
- Scatter: 2-D (XY/XZ/YZ) and 3-D point views, coloured by an attribute or by
  computed features of data.
- Histogram and attribute plots for any attribute or pair of attributes. It also serve as the main UI for user-interactive filter.
- The data information can be displayed against each loaded MINFLUX dataset (by keyboard shortcut 'I').
- Multiple data can be opened and viewed, and processed at the same time. A dataset manager would keep track of which dataset is being the active one.

## Filtering

- Multi-attribute filtering (for example by `efo`, `cfr`, `dcr`, localization
  precision, or trace length), applied across all views.
- Iteration/validity browsing; filters are re-evaluated across iterations. Filter
  definitions can be saved and reloaded.

## Regions of interest

- Rectangle, oval, polygon, freehand, line, point, and angle ROIs. Regions can be
  cropped, duplicated, converted, resized, or skeletonized, and imported/exported as native JSON or as ImageJ `.roi`/`.zip` (physical coordinates of localization will be lost when save to ImageJ ROI).

## Analysis

- Local density; localization precision (StdDev, CRLB, FRC); Z scaling factor estimation from trace anisotropy
  estimation; DCR channel separation; structure segmentation (NPC ring detection
  and general 2-D/3-D convolution matched-filter); sub-unit clustering; and
  particle averaging, including a canonical NPC two-ring model fit that reports
  per-particle diameter, inter-ring distance, and tilt.
- Export to `.mat`/`.npy`/`.json`/`.csv`/`.npz`/`.zarr`, OME-TIFF, and ROI sets.
- Generation of a methods-text paragraph describing the applied processing, with
  citations.
- A data simulator generates synthetic MINFLUX datasets (NPC, microtubule,
  cluster, sphere, multi-channel, DCR) for testing.

## Plugins

- ParaView can transfer the active dataset, with its current status (e.g.: filtered data) to ParaView. The ParaView application path need to be provided in Preference setup.
- more ad-hoc function or workflow, or test functionality are dumped into plugins.

---

## Installation and running

Prebuilt applications for each operating system are provided on the
[Releases](https://github.com/embl-ic/minflux-viewer/releases) page.

| OS | File | Run |
|----|------|-----|
| Windows | `MINFLUX-Viewer-<version>-win.zip` | Unzip, then run `minflux_viewer.exe`. |
| macOS | `MINFLUX-Viewer-<version>-macOS.zip` | Unzip, then open `MINFLUX Viewer.app`. The application is not code-signed, so on first launch use right-click → Open, or run `xattr -dr com.apple.quarantine "MINFLUX Viewer.app"` once. |
| Linux | `MINFLUX-Viewer-<version>-linux.tar.gz` | Extract, then run `./minflux_viewer/minflux_viewer`. If it does not start, install the Qt/OpenGL runtime libraries: `sudo apt install libgl1 libegl1 libxcb-cursor0`. |

The on-startup update check is disabled by default. It can be enabled in the
update dialog or in Preferences → File. `Help → Check for Updates` reports whether
a newer release exists. Frozen Windows/Linux builds can install a signed update
only when a trusted publisher key is configured; current unconfigured builds
offer manual release links. macOS uses manual downloads. See
[Release authentication](#release-authentication) for the trust requirements.

## Running from source

Python 3.10, 3.11, or 3.12 is required.

```bash
# Clone
git clone https://github.com/embl-ic/minflux-viewer.git
cd minflux-viewer

# Create and activate a virtual environment
python3 -m venv .venv
#   Windows (PowerShell):  .venv\Scripts\Activate.ps1
#   macOS / Linux:         source .venv/bin/activate

# Install the application and its dependencies
pip install -e .

# Run
minflux-viewer                    # or:  python -m minflux_viewer
minflux-viewer /path/to/data.msr  # open a file on launch
```

On Linux, the Qt/OpenGL system libraries listed above may also be required.

### Convolution previews and the 3D convolution plugin

**Analyze › Segmentation › Convolution…** is the existing CPU (SciPy) 2D tool.
Its scientific histogram, FFT and detections use the exact **Compute pixel**
size. The independent Data inspection raster adapts to the viewport (at most
1024 pixels per axis), including when scientific computation exceeds its budget.
Changing display zoom or B/C never changes detection resolution. Response previews
may be max-pooled for display only. **Use required budgets** explicitly raises
the CPU estimate limit without silently coarsening scientific bins.

Data copies the open render view's dataset-channel LUT, gamma/inversion/background
and B/C clipping percentiles on opening. Percentiles preserve contrast intent
across different bin sizes; raw count levels are not interchangeable. With no
open render, the dataset/render-default LUT is used. **Copy render appearance**
re-copies later edits; local B/C changes do not change the render view.

**Plugins › 3D Convolution…** is separate. It performs true XYZ matched
filtering on an explicit region and exact X/Y/Z voxels, then physical-distance
3D peak suppression. Templates: 3D Gaussian, spherical shell and NPC two rings
(normal along Z; no tilt search). Inspect XY/XZ/YZ projections and explicitly
add selected XYZ point ROIs to the ROI Manager. The initial small region is near
the data centre; use the render viewport, whole extent or edit XYZ bounds.
Localizations up to one template radius outside the region are included, so a
structure at the region's edge is not cut; detections are reported inside the
region only. Scores are relative to the strongest response in the region.

Press **Run** after setting parameters; threshold-only changes reuse the volume
on the next Run. The convolution runs on the CPU (SciPy, float64) in the
background. Its memory estimate is checked against the background-work budget in
**Preferences › Data**; the voxel limit (default 8 million, adjustable) bounds
the grid. Refusals ask for a smaller region or a larger voxel size and never
downsample the data silently.

---

## For developers

A short technical overview, sufficient to clone or fork the project.

Stack: Python 3.10–3.12, PyQt6 (GUI), pyqtgraph with PyOpenGL (2-D/3-D plotting),
NumPy/SciPy (computation), zarr, h5py, and tifffile (I/O), and msr-reader
(pure-Python OBF/`.msr` reading). The project is managed with Poetry
(`poetry sync`) and tested with `pytest`. Releases use the locked PyInstaller
build dependency group.

Build the Windows one-folder executable from the Poetry environment—not from a
global Python installation. PyInstaller is pinned in the optional `build`
dependency group and locked in `poetry.lock`, so the build is reproducible:

```powershell
poetry sync --with build
.\.venv\Scripts\python.exe tools\vendor_pip.py    # generated build input
.\.venv\Scripts\python.exe -m PyInstaller --clean --noconfirm minflux_viewer.spec
```

The executable is written to `dist\minflux_viewer\minflux_viewer.exe`.

Layout:

```
minflux_viewer/
  core/       data model, format loaders, AppState (Qt signal bus), save/export
  ui/         capability packages: controllers, rendering, plots, roi, dialogs, ...
  analysis/   algorithms (density, precision, anisotropy, segmentation, particle averaging, ...)
  msr/        the .msr parser (OBF -> MFXDTA -> zarr -> mfx) and .msr writer
  plugins/    MSR reader, Data Simulator, Script Editor, Trace Viewer, ParaView, method-text
resources/    Qt .ui files, icons
tests/        pytest suite
```

Design principles:

- Canonical import. Every source format is normalized to one internal
  representation, so the UI and analyses do not branch on file type.
- One fact, one storage location. Normalized coordinates are metre-valued
  `loc_x/y/z` columns; calibrated/transformed arrays are views or explicit copies,
  not additional persistent coordinate fields.
- Dataset and viewer are separate. Datasets hold data; viewer windows reference
  them as layers. Multi-colour channels are separate datasets, combined only at
  the viewer as an overlay.
- Raw, derived, and view state are kept apart. Canonical `mfx`, computed
  `derived`, and persistent `state` (filters, ROIs, LUTs) do not overlap.

Data model. A dataset carries `attr`/`mfx` (the materialized localizations used
for plotting), `mfx_raw` (the all-iteration, all-validity flat store), an optional
`mbm` (bead/fiducial reference), `derived` (computed attributes), `state`
(persistent view/filter preferences), and `metadata`. Common source `mfx` fields are
`loc`, `itr`, `tid`, `tim`, `vld`, `efo`, `cfr`, `dcr`, plus optional `eco`,
`ecc`, `fbg`, and others. The MINFLUX iteration formats (`m2410` flat, `m2205`
nested, legacy) are collapsed to this flat model, with selectors for `last`,
`all`, and per-iteration views.

Runtime registrations use stable internal dataset keys, not list positions.
Shared read-only coordinate views have an adjustable cache-owned-array budget
(25% of physical RAM by default), not a total application memory cap. Public
coordinate access returns independent arrays; filters have one canonical mask.

Filtering. A filter is a specification rather than a baked mask: each row is
`{attribute, mode, lo, hi, lo_inc, hi_inc}`, stored in
`dataset.state["filter_specs"]`. `core/loader.py::mfx_filter_mask(ds, itr=,
vld_only=)` re-applies those specifications against the raw store on demand, so a
filter extends to any iteration and to invalid localizations (NaN is excluded),
and to per-localization or trace-aggregated (mean/median/min/max/...) values.
Toggling a filter does not modify the raw arrays; views recompute from the mask.
ROIs are selection/highlight overlays, not filters — they mark localizations
without changing `filter_mask` or the raw data.

The `.msr` reader is a custom, pure-Python parser. It does not use Abberior's
specpy SDK or an Imspector installation; it decodes the nested container directly
(OBF stack -> custom `MFXDTA` archive -> embedded zarr store -> structured `mfx`
array) via the cross-platform [`msr-reader`](https://pypi.org/project/msr-reader/)
dependency. It targets the layouts that have been validated (modern multi-channel
and legacy single-channel) and is not affiliated with or endorsed by Abberior
Instruments.

Tests:

```bash
poetry sync --with build
poetry run python -m pytest -m "not qt" -q
poetry run python tools/run_qt_tests.py -j 4 --strict
poetry run python tools/run_qt_tests.py -j 4 --strict
poetry run python tools/lint_ratchet.py
poetry run python tools/exception_ratchet.py
```

Pure tests run in one process; Qt modules run in separate processes. A native
crash fails the strict gate even if a diagnostic retry passes. CI-style pure
checks use `PYTHONUTF8=1` and `TZ=UTC0`; also test a source-only copy in local time.
The suite redirects preferences and plugin folders away from real user settings.
Lint/type/exception baselines measure existing debt, not permission to add more.

The entry points are `run_app.py` and `minflux_viewer/__main__.py`.

---

## Release authentication

Automatic installation on frozen Windows/Linux builds requires an Ed25519-signed
release manifest and a trusted public key pinned in the application. No key is
configured by default; manual release links and update checks remain available.
GitHub's asset checksums alone are not publisher authentication.

The maintainer can create manifests locally with `tools/sign_release.py --help`.
Use an encrypted private key outside the checkout and outside GitHub/CI; publish
only the manifest, detached signature and exact archives. Distribute public-key
pins separately in a reviewed build. Expiry relies on the system clock, rollback
protection compares the installed version, and key rotation is manual: this is
not a full TUF implementation or operating-system code signing/notarization.

## License and contact

MIT License. Ziqiang Huang, [ziqiang.huang@embl.de](mailto:ziqiang.huang@embl.de),
EMBL Imaging Centre, Heidelberg.
