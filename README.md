# MINFLUX Data Viewer

A Fiji-style desktop application for viewing, filtering, and analysing
[MINFLUX](https://www.embl.org/about/info/imaging-centre/super-resolution-imaging/#vf-tabs__section-c46b219a-947a-467f-a169-838b85a0d06b) nanoscopy data.

Python/Qt port of the original MATLAB application developed at the
[EMBL Imaging Centre](https://www.embl.org/about/info/imaging-centre/).

---

## Features (v0.1 — Phase 1 + 2 + MSR import)

- Load Abberior Imspector `.mat` files (formats **m2205** and **m2410**)
- **Load `.msr` raw data files** directly (Windows, requires specpy) via File → Open .msr or drag-and-drop
- Automatic parsing of both nested and flat iteration structures
- Per-dataset floating info windows — click to activate (Fiji-style)
- Multi-dataset management via `AppState` Qt signal bus
- Drag-and-drop `.mat` and `.msr` files onto the main window
- **Histogram window** — any attribute, 6 aggregation modes, draggable filter lines
- **Scatter plot** — XY/XZ/YZ view, colour by density or any attribute
- **Attribute plot** — plot any two attributes against each other
- **Filter dialog** — multi-row filter table with JSON save/load
- **Dataset manager** — table view of all loaded datasets
- **Render window** — fast 2D Gaussian-blur render with pan/zoom and Z slider for 3D data
- **ParaView integration** — View → ParaView launches ParaView with the active dataset
- **Log / Console window** — timestamped event log accessible via View → Log / Console (`Ctrl+L`)
- **Qt Designer workflow** — main-window layout lives in `resources/ui/*.ui`; see `UI_DESIGNER.md`
- Persistent preferences via `QSettings`

---

## Requirements

| Dependency   | Version  |
|--------------|----------|
| Python       | ≥ 3.10   |
| PyQt6        | ≥ 6.5    |
| numpy        | ≥ 1.24   |
| scipy        | ≥ 1.11   |
| pyqtgraph    | ≥ 0.13   |
| h5py         | ≥ 3.9    |

---

## Installation

### Using Poetry (recommended)

```bash
# 1. Clone the repository
git clone https://github.com/EMBL-ICLM/minflux-viewer.git
cd minflux-viewer

# 2. Install Poetry if you don't have it
#    https://python-poetry.org/docs/#installation
curl -sSL https://install.python-poetry.org | python3 -

# 3. Install all dependencies and create the virtual environment
poetry install

# 4. Launch
poetry run minflux-viewer
```

### Using pip (without Poetry)

```bash
pip install PyQt6 numpy scipy pyqtgraph h5py
python -m minflux_viewer
```

---

## Running

```bash
# Open the viewer
poetry run minflux-viewer

# Open a specific file on launch
poetry run minflux-viewer /path/to/data.mat

# Run without poetry (from the repo root)
python -m minflux_viewer

# Run tests
poetry run pytest
poetry run pytest -v          # verbose
poetry run pytest --tb=short  # short tracebacks
```

---

## Project structure

```
minflux_viewer/
├── __init__.py             version, author
├── __main__.py             entry point  (python -m minflux_viewer)
│
├── core/
│   ├── dataset.py          MinfluxDataset dataclass + AttrStore
│   ├── loader.py           .mat parser + load_from_mfx_array()
│   └── app_state.py        AppState — Qt signal bus, preferences
│
├── ui/
│   ├── main_window.py      Fiji-style QMainWindow
│   ├── data_window.py      Per-dataset floating info window
│   ├── histogram_window.py Attribute histogram + live filter lines
│   ├── scatter_window.py   Scatter plot (XY/XZ/YZ, colour-by)
│   ├── attribute_window.py Attribute vs attribute line/scatter plot
│   ├── filter_dialog.py    Multi-row filter table + JSON I/O
│   ├── dataset_manager.py  Dataset list table
│   └── msr_import_dialog.py  .msr import dialog (Qt, wraps msr backend)
│
├── msr/                    MSR backend (vendored from minflux_msr)
│   ├── __init__.py         Public API: parse_msr_general, export_arrays
│   ├── msr_parser.py       Modern + legacy .msr parser via specpy
│   ├── io.py               Zarr field collection, parse routing
│   ├── export.py           .mat / .npy / .npz / .json exporters
│   ├── state.py            mfx_map / mbm_map global arrays
│   └── ...                 parser_model, utils, cli
│
└── utils/
    └── filters.py          Pure-numpy filter masks + aggregation

tests/
├── test_dataset.py         Core data model and loader
└── test_phase2.py          Filter/aggregation/density + mfx adapter
```

---

## Architecture

```
AppState  ──signals──▶  MainWindow
    │                       │
    │                   DataWindow(s)   ← one per dataset
    │                       │
    └── datasets[]          └── on focus → state.set_active(idx)
    └── active_idx
    └── prefs
```

`AppState` is the single source of truth. No UI component calls another
directly — they all react to signals (`dataset_added`, `active_changed`,
`filter_changed`, `status_message`).

---

## MATLAB → Python correspondence

| MATLAB | Python |
|--------|--------|
| `app.data(idx)` struct | `MinfluxDataset` dataclass |
| `app.active_idx` | `AppState.active_idx` |
| `set_active_data.m` | `AppState.add_dataset()` |
| `get_active_data_index.m` | `AppState.active_idx` property |
| `arrange_MINFLUX_data_structure.m` | `loader._parse_m2205_unwrapped()` |
| `data_parser_m2410.m` | `loader._parse_m2410()` |
| `compute_extra_property.m` | `loader._add_derived_attributes()` |
| `app.Prefs` | `AppState.prefs` dict + `QSettings` |
| `open_data_window()` | `DataWindow` widget |
| `dialog_info.mlapp` | `DataWindow` + future dataset-manager dialog |

---

## Development roadmap

| Phase | Content | Status |
|-------|---------|--------|
| **1** | Foundation: data model, loader, main window, data windows | ✅ Done |
| **2** | Analysis tools: histogram, scatter plot, attribute plot, filter dialog | ✅ Done |
| **2b** | `.msr` import: Qt dialog wrapping the `minflux_msr` backend | ✅ Done |
| **3** | Rendering: 2D Gaussian render, LUT dialog, ROI drawing, TIFF export | 🔲 Next |
| **4** | Polish: preferences dialog, 3D views, loc precision, local density | 🔲 Planned |

---

## Contributing

```bash
# Install dev dependencies
poetry install

# Lint
poetry run ruff check .

# Type-check
poetry run mypy minflux_viewer

# Tests
poetry run pytest
```

---

## Licence

MIT — see `LICENSE`.

## Contact

Ziqiang Huang — [ziqiang.huang@embl.de](mailto:ziqiang.huang@embl.de)  
EMBL Imaging Centre, Heidelberg

---

## Opening .msr files

`.msr` files are Abberior Imspector's raw acquisition format. Opening them
requires **specpy**, Abberior's own Python SDK — a Windows-only `.pyd` binary.

### Install specpy (Windows only, Python 3.12)

The wheel is already in the repository at `vendor/`:

```powershell
# Step 1 — base install (includes zarr + tifffile, required for .msr parsing):
poetry install

# Step 2 — install the vendored specpy wheel (Windows, Python 3.12 only):
poetry run pip install vendor\specpy-1.2.3-cp312-cp312-win_amd64.whl
```

See **`INSTALL_MSR.md`** for requirements, verification, and troubleshooting.

### Usage

Once specpy is installed, `.msr` files can be opened two ways:

**Menu:** File → Open .msr file… (`Ctrl+Shift+O`)

**Drag-and-drop:** Drop one or more `.msr` files onto the main window.

Both routes open the **MSR Import Dialog**, which parses the file using
`specpy`, shows a tree of detected datasets and fields, then on **Import**
converts the in-memory `mfx` structured array directly into a
`MinfluxDataset` — no intermediate `.mat` file is written.

### On non-Windows machines

If specpy is not available (Linux, macOS) the dialog shows an informative
message and suggests exporting from Imspector as `.mat` instead. All other
viewer functionality is unaffected.


---

## Developing the UI with Qt Designer

Several windows have their layout defined in Qt Designer `.ui` files
instead of hand-built Python. This lets you tweak menus, toolbars, and
static layouts visually.

**Edit loop:**

```bash
# 1. Open a .ui file in Qt Designer
poetry run pyqt6-tools designer resources/ui/main_window.ui

# 2. Save your changes (Ctrl+S)

# 3. Regenerate the Python bindings
poetry run python tools/compile_ui.py

# 4. Run the app — your changes are live
poetry run minflux-viewer
```

Auto-rebuild on save:

```bash
poetry run python tools/compile_ui.py --watch
```

See **[UI_DESIGNER.md](UI_DESIGNER.md)** for the complete workflow,
what's in vs. out of the `.ui` files, and troubleshooting.
