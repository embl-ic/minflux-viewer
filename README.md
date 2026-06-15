# MINFLUX Data Viewer

A Fiji-style desktop application for viewing, filtering, and analysing
[MINFLUX](https://www.embl.org/about/info/imaging-centre/super-resolution-imaging/#vf-tabs__section-c46b219a-947a-467f-a169-838b85a0d06b) nanoscopy data.

Python/Qt port of the original MATLAB application developed at the
[EMBL Imaging Centre](https://www.embl.org/about/info/imaging-centre/).

---

## Features (v0.1 — Phase 1 + 2 + MSR import)

- Load Abberior Imspector `.mat` files (formats **m2205** and **m2410**)
- **Load `.msr` raw data files** directly (cross-platform, no specpy/Imspector needed) via File → Open or drag-and-drop
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
| msr-reader   | ≥ 0.2.1  |

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
pip install PyQt6 numpy scipy pyqtgraph h5py msr-reader
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

## Contact

Ziqiang Huang — [ziqiang.huang@embl.de](mailto:ziqiang.huang@embl.de)  
EMBL Imaging Centre, Heidelberg
