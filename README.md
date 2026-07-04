# MINFLUX Viewer

MINFLUX Viewer is a desktop application for viewing, filtering, processing, and
analysing MINFLUX microscopy data. It reads most if not all MINFLUX data
formats (e.g.: .mat, .npy, .json etc), including Abberior Imspector `.msr`, and operate on Windows, macOS, and Linux. It use Qt backend GUI to support regular data exploration, processing especially filter on data attributes. It also capable of export processed data to 3rd party application, including directly in-app call to ParaView if it already installed and accessible. It is intended for users who is familar with using ImageJ/Fiji, and need to inspect and process MINFLUX data.

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
*legacy* layouts) and container formats that have changed across Imspector/MINFLUX releases. The common ones are supported. A file that does not open correctly can be reported on the [issue tracker](https://github.com/EMBL-ICLM/minflux-viewer/issues) with a minimum sample, so the parser can be extended to cover it.

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

- Local density; localization precision (StdDev, CRLB, FRC); anisotropy / RIMF
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
[Releases](https://github.com/EMBL-ICLM/minflux-viewer/releases) page.

| OS | File | Run |
|----|------|-----|
| Windows | `MINFLUX-Viewer-<version>-win.zip` | Unzip, then run `minflux_viewer.exe`. |
| macOS | `MINFLUX-Viewer-<version>-macOS.zip` | Unzip, then open `MINFLUX Viewer.app`. The application is not code-signed, so on first launch use right-click → Open, or run `xattr -dr com.apple.quarantine "MINFLUX Viewer.app"` once. |
| Linux | `MINFLUX-Viewer-<version>-linux.tar.gz` | Extract, then run `./minflux_viewer/minflux_viewer`. If it does not start, install the Qt/OpenGL runtime libraries: `sudo apt install libgl1 libegl1 libxcb-cursor0`. |

The on-startup update check is disabled by default. It can be enabled in the
update dialog or in Preferences → File. `Help → Check for Updates` reports whether
a newer release exists; on Windows and Linux it can download and install the new
version in place and restart after confirmation. On macOS the dialog provides a
download link (in-place update requires code-signing).

## Running from source

Python 3.10, 3.11, or 3.12 is required.

```bash
# Clone
git clone https://github.com/EMBL-ICLM/minflux-viewer.git
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

---

## For developers

A short technical overview, sufficient to clone or fork the project.

Stack: Python 3.10–3.12, PyQt6 (GUI), pyqtgraph with PyOpenGL (2-D/3-D plotting),
NumPy/SciPy (computation), zarr, h5py, and tifffile (I/O), and msr-reader
(pure-Python OBF/`.msr` reading). The project is managed with Poetry
(`poetry install`) and tested with `pytest`. Releases are built with `pyinstaller`.

Layout:

```
minflux_viewer/
  core/       data model, format loaders, AppState (Qt signal bus), save/export
  ui/         windows and dialogs (render, scatter, histogram, ROI, preferences, ...)
  analysis/   algorithms (density, precision, anisotropy, segmentation, particle averaging, ...)
  msr/        the .msr parser (OBF -> MFXDTA -> zarr -> mfx) and .msr writer
  plugins/    MSR reader, Data Simulator, Script Editor, Trace Viewer, ParaView, method-text
resources/    Qt .ui files, icons
tests/        pytest suite
```

Design principles:

- Canonical import. Every source format is normalized to one internal
  representation, so the UI and analyses do not branch on file type.
- One fact, one storage location. Coordinates are stored once (canonical `loc`
  in metres); other quantities are views or derived values, not duplicate fields.
- Dataset and viewer are separate. Datasets hold data; viewer windows reference
  them as layers. Multi-colour channels are separate datasets, combined only at
  the viewer as an overlay.
- Raw, derived, and view state are kept apart. Canonical `mfx`, computed
  `derived`, and persistent `state` (filters, ROIs, LUTs) do not overlap.

Data model. A dataset carries `attr`/`mfx` (the materialized localizations used
for plotting), `mfx_raw` (the all-iteration, all-validity flat store), an optional
`mbm` (bead/fiducial reference), `derived` (computed attributes), `state`
(persistent view/filter preferences), and `metadata`. Canonical `mfx` keys are
`loc`, `itr`, `tid`, `tim`, `vld`, `efo`, `cfr`, `dcr`, plus optional `eco`,
`ecc`, `fbg`, and others. The MINFLUX iteration formats (`m2410` flat, `m2205`
nested, legacy) are collapsed to this flat model, with selectors for `last`,
`all`, and per-iteration views.

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
pip install pytest pytest-qt   # or: poetry install  (installs the dev group)
pytest -q
```

The entry points are `run_app.py` and `minflux_viewer/__main__.py`.

---

## License and contact

MIT License. Ziqiang Huang, [ziqiang.huang@embl.de](mailto:ziqiang.huang@embl.de),
EMBL Imaging Centre, Heidelberg.
