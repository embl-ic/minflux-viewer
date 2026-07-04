# MINFLUX Viewer

**A desktop app to view, filter, process, and analyse [MINFLUX](https://www.embl.org/about/info/imaging-centre/) single-molecule localization data — in one place, on Windows, macOS, and Linux.**

If you have MINFLUX data (Abberior `.msr`, exported `.mat`/`.csv`/`.npy`, or plain localization tables) and want to *look at it, clean it up, and measure things* without writing code or wrestling with a Python environment, this tool is for you. It reads your files natively — including Abberior `.msr` on **Windows, macOS, and Linux** — and gives you Fiji-style windows for rendering, scatter/histogram exploration, region selection, and a growing set of quantitative analyses.

Developed at the [EMBL Imaging Centre](https://www.embl.org/about/info/imaging-centre/) (a Python/Qt successor to an in-house MATLAB toolset).

---

## What you can do with it

**📂 Open almost any MINFLUX data**
- Abberior Imspector **`.msr`** — read natively on **Windows, macOS, and Linux** (not Windows-only), both **modern multi-channel** and older **legacy single-channel** files, including any embedded confocal/STED images and bead/fiducial data.
- `.mat` (Imspector `m2205` and `m2410`), `.npy`, `.csv`, `.json`, `.npz`, `.zarr`, spreadsheets — any localization table.
- TIFF images and OBF image series (Fiji-style image viewer).
- Just **drag-and-drop** files onto the window. Multiple datasets and multi-colour **overlays** are handled together.

> **A file or version not opening?** MINFLUX data has evolved through several structural versions (`m2410`, `m2205`, and older *legacy* layouts) and container formats across Imspector/MINFLUX releases. We've validated the common ones, but if a file of yours doesn't open correctly we'd genuinely like to see it — please [open an issue](https://github.com/EMBL-ICLM/minflux-viewer/issues) with a sample so we can extend the parser to cover it.

**🔬 Visualize**
- **Render** — fast 2-D super-resolution reconstruction with colormaps, a depth slider for 3-D data, white/black background, scale bar, and a **3-D volume** preview.
- **Scatter** — 2-D (XY/XZ/YZ) and interactive **3-D** point views, coloured by any attribute or density.
- **Histogram** & **Attribute plot** — explore any attribute or plot two against each other.

**🎚 Filter**
- Interactive, **multi-attribute filtering** (e.g. keep localizations by `efo`, `cfr`, `dcr`, precision, trace length…), applied live across all views.
- Browse by **iteration / validity**; filters re-evaluate correctly across iterations. Save and reload filter sets.

**✏️ Select regions (ROIs)**
- Rectangle, oval, polygon, freehand, line, point, angle. Crop/duplicate a region, convert/resize/skeletonize shapes, and import/export **ImageJ** `.roi`/`.zip` or native JSON.

**📈 Process & analyze**
- Local density, **localization precision** (StdDev / CRLB / FRC), **anisotropy / RIMF** estimation, **DCR channel separation**, structure **segmentation** (NPC ring detection, general 2-D/3-D convolution matched-filter), sub-unit clustering, and **particle averaging** — including an unbiased **canonical NPC model fit** that reports per-particle diameter / inter-ring / tilt for studying structural variation across conditions.
- **Save / export** to `.mat`/`.npy`/`.json`/`.csv`/`.npz`/`.zarr`, OME-**TIFF**, and ROI sets.
- Generate a **methods paragraph** describing the processing you applied (with citations).

**🧪 No data yet?** Built-in **Data Simulator** generates synthetic MINFLUX datasets (NPCs, microtubules, clusters, spheres, multi-channel, DCR) to try things out.

### Why this viewer
- **One tool for the whole workflow** — open → filter → process → analyse → export, without switching software.
- **Reads Abberior `.msr` natively on Windows, macOS, and Linux** — not limited to Windows.
- **Everything normalizes to one internal model**, so the same tools work regardless of the source format.
- **Cross-platform** builds with **in-app auto-update** (Windows & Linux).

---

## Install & run (recommended: download a build)

Grab the build for your OS from the [**Releases** page](https://github.com/EMBL-ICLM/minflux-viewer/releases), then:

| OS | File | Run |
|----|------|-----|
| **Windows** | `MINFLUX-Viewer-<version>-win.zip` | Unzip anywhere → double-click **`minflux_viewer.exe`** |
| **macOS** | `MINFLUX-Viewer-<version>-macOS.zip` | Unzip → open **`MINFLUX Viewer.app`**. First launch: **right-click → Open** (or run `xattr -dr com.apple.quarantine "MINFLUX Viewer.app"` once — the app is not code-signed) |
| **Linux** | `MINFLUX-Viewer-<version>-linux.tar.gz` | Extract → run `./minflux_viewer/minflux_viewer`. If it doesn't start, install Qt/GL runtime libs: `sudo apt install libgl1 libegl1 libxcb-cursor0` |

**Staying up to date:** *Help → Check for Updates*. On Windows and Linux it can **download and install the new version in place and restart** (with your confirmation). The on-startup update check is **off by default** (opt-in) and can be enabled in the update dialog or *Preferences → File*. macOS shows a download link (in-place auto-update there needs code-signing).

---

## Run from source (Python, no build)

If you'd rather run it from the repository (any OS). **Requires Python 3.10, 3.11, or 3.12.**

```bash
# 1. Clone
git clone https://github.com/EMBL-ICLM/minflux-viewer.git
cd minflux-viewer

# 2. Create + activate a virtual environment
python3 -m venv .venv
#   Windows (PowerShell):  .venv\Scripts\Activate.ps1
#   macOS / Linux:         source .venv/bin/activate

# 3. Install the app and its dependencies
pip install -e .

# 4. Launch
minflux-viewer                    # or:  python -m minflux_viewer
minflux-viewer /path/to/data.msr  # open a file on launch
```

On Linux you may also need the Qt/GL system libraries listed in the table above.

---

## For developers

A short technical overview — enough to clone/fork and customize. Contributions aren't actively solicited, but the code is MIT-licensed and self-contained.

**Stack.** Python 3.10–3.12 · **PyQt6** (GUI) · **pyqtgraph** + PyOpenGL (2-D/3-D plotting) · **NumPy/SciPy** (compute) · **zarr** + **h5py** + **tifffile** (I/O) · **msr-reader** (pure-Python OBF/`.msr`). Managed with Poetry (`poetry install`); tests with `pytest`.

**Layout.**
```
minflux_viewer/
  core/       data model, format loaders, AppState (Qt signal bus), save/export
  ui/         all windows & dialogs (render, scatter, histogram, ROI, preferences, …)
  analysis/   algorithms (density, precision, anisotropy, segmentation, particle averaging, …)
  msr/        the .msr parser (OBF → MFXDTA → zarr → mfx) and .msr writer
  plugins/    MSR reader, Data Simulator, Script Editor, Trace Viewer, ParaView, method-text
resources/    Qt .ui files, icons
tests/        pytest suite
```

**Design principles.**
- **Canonical import** — every source format is normalized to *one* internal representation, so the UI and analyses never branch on file type.
- **One fact, one storage location** — coordinates live once (canonical `loc` in metres); everything else is a *view* or a *derived* value, not a duplicate field.
- **Dataset ≠ viewer** — datasets hold data; viewer windows reference them as layers. Multi-colour channels are **separate datasets**, combined only at the viewer (overlay).
- **Raw / derived / view-state are separate** — canonical `mfx`, computed `derived`, and persistent `state` (filters, ROIs, LUTs) don't leak into each other.

**Data model.** A dataset carries `attr`/`mfx` (the materialized localizations the UI plots), `mfx_raw` (the all-iteration/all-validity flat store), optional `mbm` (bead/fiducial reference), `derived` (computed attributes), `state` (persistent view/filter prefs), and `metadata`. Canonical `mfx` keys: `loc`, `itr`, `tid`, `tim`, `vld`, `efo`, `cfr`, `dcr` (+ optional `eco`, `ecc`, `fbg`, …). MINFLUX iteration formats (`m2410` flat, `m2205` nested, legacy) all collapse to this flat model with selectors for `last` / `all` / per-iteration views.

**Filtering logic (the important part).** A filter is a *specification*, not a baked mask: each row is `{attribute, mode, lo, hi, lo_inc, hi_inc}` saved in `dataset.state["filter_specs"]`. `core/loader.py::mfx_filter_mask(ds, itr=, vld_only=)` re-applies those specs against the raw store on demand, so **the same filter extends to any iteration and to invalid localizations** (NaN → excluded), and to per-localization *or* trace-aggregated (mean/median/min/max/…) values. Toggling a filter never mutates the raw arrays — views recompute from the mask. **ROIs are selection/highlight overlays, not filters** — they mark localizations without changing `filter_mask` or the raw data.

**About `.msr` reading (disclaimer).** The `.msr` reader here is our **own custom, pure-Python parser** — it does **not** use Abberior's **specpy** SDK or an Imspector installation. It decodes the nested container ourselves (OBF stack → custom `MFXDTA` archive → embedded zarr store → structured `mfx` array), via the cross-platform [`msr-reader`](https://pypi.org/project/msr-reader/) dependency. This is what makes `.msr` support work on any OS with no vendor software. It targets the layouts we've validated (modern multi-channel + legacy single-channel); it is not affiliated with or endorsed by Abberior Instruments.

**Run the tests.**
```bash
pip install pytest pytest-qt   # or: poetry install  (installs the dev group)
pytest -q
```

Fork it, open `run_app.py` / `minflux_viewer/__main__.py` as entry points, and customize freely.

---

## License & contact

MIT License. Ziqiang Huang — [ziqiang.huang@embl.de](mailto:ziqiang.huang@embl.de), EMBL Imaging Centre, Heidelberg.
