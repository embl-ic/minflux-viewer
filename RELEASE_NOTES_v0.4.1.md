# MINFLUX Viewer v0.4.1

v0.4.1 focuses on color. Colormaps now belong to the application rather than an
external plotting library, a new application-wide **COLOR** dialog owns every
configurable color in the program, ROI colors are split into the parts that are
actually drawn, and the LUT editor is a single window that follows the view you
are working in. Two long-standing color bugs are fixed along the way: ROIs that
rendered invisible, and Invert LUT doing nothing on the render view.

---

## Application-owned colormaps

MINFLUX Viewer no longer depends on Matplotlib or Colorcet to provide colors.
Every supported view now resolves maps through one application-owned PyQtGraph
registry, including Render, Localization Scatter, TIFF, 3-D Volume, Local
Density, segmentation response images, straightened volumes, and analysis
plugins.

- The focused built-in set is **hot, jet, HiLo, glasbey, viridis, inferno, and
  gray**, plus the solid channel colors.
- Previously exposed **parula, turbo, magma, plasma, and cividis** remain
  available internally so an existing saved view does not fail or silently
  change just because its map is no longer offered to new users.
- Unknown colormap names now raise at the shared registry boundary instead of
  depending on whichever optional external collection happened to be installed.

### Create your own gradient

Open the main **LUT** dialog and use the **Custom** button beside the Colormap
dropdown:

- **Create custom colormap…** opens a gradient editor. Right-click the gradient
  to add a stop, drag stops to position them, click a stop to change its color,
  and right-click a stop to remove it.
- A saved map is immediately selectable in Render, Localization Scatter, 3-D
  Volume, Local Density, and channel-LUT selectors.
- Existing custom maps can be edited or deleted from the same menu.
- Custom maps are stored in the application's preferences and survive restarts.

## The COLOR dialog

The toolbar **Color** button opens a single, application-wide **COLOR** dialog
that owns every configurable color in the program. Colors are stored as RGBA in
`prefs["colors"]` and every view reads them through one registry, so a color
changed here takes effect everywhere at once.

The dialog is organized as four tabs, each with its own **Reset**:

- **Solid Color List** — the named solid colors (Red … Black) offered by the
  render, scatter and channel color menus. Entries can be renamed, added and
  deleted; the list is what those menus show.
- **Viewer / Plots** — Attribute Plot, Histogram, Filter, Overlay channel
  colors, and ROI.
- **Components** — built-in features that own a set of colors: ROI Manager,
  Iteration series, Localization precision.
- **Plugins** — Spatial Line Pattern, Drift Correction, Trace Viewer.

Below the tabs, the **Custom Color Palette** provides Qt's basic/custom swatch
grids, the gradient and alpha bar, *Pick Screen Color*, a live preview block,
and HSV / HEX / RGBA fields. **Update Selected Feature Color** applies the
picked color to whichever entry is selected above.

The dialog opens at a fixed width (the palette decides it) and a compact height;
a tab holding more entries than fit scrolls rather than squeezing its rows.

### ROI colors

The single ROI color is now four, so the parts that are drawn separately can be
set separately:

```
ROI:  face   edge   corner   highlight data in ROI
```

Stored ROIs are styled by the **ROI Manager** component group, which
distinguishes listed entries from the selected one:

```
ROI entries:   face  edge  corner  label
ROI selected:  face  edge  corner  label
```

A ROI that carries its own color — segmentation output, or one set in ROI
Properties — keeps it; only ROIs still on the system color follow the palette.
An existing single `roi` preference is carried over into the new keys rather
than reverting to the default.

**Fixed: ROIs and their labels could become invisible.** ROI colors were being
written in Qt's `#AARRGGBB` form, which PyQtGraph reads as `#RRGGBBAA` — taking
the blue byte as alpha, so an opaque yellow arrived fully transparent. Edges,
fills and ROI Manager labels disappeared, and changing the color could not
restore them because the recolor pass compared with the wrong parser. Colors are
now written in the form PyQtGraph reads, and an ROI still holding a legacy value
is repaired when it is drawn.

## LUT dialog

- **One LUT dialog, application-wide.** Opening a LUT from a second view used to
  create a second dialog, and focusing a render view closed whichever dialog
  another view had open. There is now a single instance that is rebound to the
  view you focus, keeping its position; it is closed with the application.
- **Invert LUT works on the render view.** It previously applied only in
  TIFF/image mode. Inverting now flips the colormap and the page together, which
  also ticks **View ▸ White background**, so zero-signal still matches the
  background.
- **A custom colormap's alpha now dims its channel**, the same treatment solid
  colors already had. The render composite is opaque, so alpha scales intensity
  rather than making pixels see-through.

## Consistent spelling

User-facing text, identifiers and comments now use **color** throughout, rather
than mixing `color` and `colour`. Section and tab names use Title Case.

## Render and Localization Scatter menus

### Localization Scatter

The right-click **View** submenu is now organized as:

```
XY
XZ
YZ
3D
-----------------------
Black background
Axis
Grid lines
Plot style
```

**Plot style** uses the same marker editor as the MSR Reader attribute plot. It
controls marker shape, size, transparency, and color. These settings persist for
the dataset and apply to both the 2-D and 3-D scatter views. The existing subtle
PyQtGraph grid style is retained; enabling grid lines does not replace it with
heavy solid lines.

### Render

- **Axis** and the new **Grid lines** toggle now live under **View**, between
  background and Render Method.
- The **Colormap** menu now has the same structure as Localization Scatter:
  named maps, a separator, a **Solid color** submenu, and a separated
  **Custom...** picker at the end.
- A custom solid render color still maps image intensity through a tonal ramp;
  it does not turn every non-zero pixel into one flat color.

## Matplotlib removal

The experimental **Attribute Plot 3D (Matplotlib)** command has been removed.
The regular Attribute Plot remains the supported tool for plotting arbitrary
attributes against one another, while **Localization Scatter → 3D** remains the
supported spatial XYZ point viewer.

Removing the experimental window also removes the last runtime need for
Matplotlib. The Poetry dependency and lock entries are gone, and the PyInstaller
spec explicitly excludes Matplotlib plus its stale support chain so an old build
environment cannot accidentally bundle it.

## Convolution segmentation replaces the duplicate NPC command

The former **Segmentation → NPC → 2D** command used the same ring kernel and peak
finding as **Segmentation → Convolution… → Ring**. The duplicate command and its
unused 3-D placeholder are removed.

Convolution's optional ring validation now also includes the former NPC tool's
**minimum ring support** score. It measures angular coverage and radial fit, so
it can reject a partial arc even when the inside/outside localization ratios
look acceptable. A value of `0` keeps this criterion off. Command Finder keeps
the **NPC** and **nuclear pore** keywords on the Convolution tools, and older Log
entries can still be turned into method text.

## Z scaling factor default for new installations

A fresh installation now starts with both **estimate Z scaling factor from trace
anisotropy** and **use fixed Z scaling factor** switched off. Z coordinates are
therefore left exactly as recorded until the user chooses a calibration policy.

This does **not** overwrite an existing user's saved preference. The preference
migration path was also corrected so one-shot migrations apply to genuinely old
saved settings, not to a brand-new copy of the current defaults.

## Packaging

- The direct Matplotlib dependency is removed from `pyproject.toml` and its
  transitive plotting packages are removed from `poetry.lock`.
- The build spec now rejects an incomplete global Python environment before
  creating a broken executable and tells the builder how to install and invoke
  PyInstaller through the project `.venv` interpreter.
- The Windows PyInstaller build uses a native application `.ico`, avoiding the
  former build-time Pillow requirement for converting a PNG icon.
- The application and Poetry package versions are both **0.4.1**; the shared
  PyInstaller spec reads that application version for packaged metadata.
- **The Windows executable now carries a version resource.** Explorer's
  Properties dialog and inventory tools previously showed nothing at all: the
  spec parsed the version but only applied it to the macOS bundle. It is built
  from the same `__version__`, so there is still one place to bump.

## Compatibility notes

- Existing saved views using legacy colormap names remain supported.
- A saved single ROI color is migrated into the new face/edge/highlight keys;
  ROIs stored with the old, unreadable color string are repaired on display.
- Solid-color menus no longer offer a one-off **Custom...** entry — the solid
  list in the COLOR dialog is that list now. Views already saved with a custom
  solid color still render it.
- Custom gradients are application preferences, not embedded into exported
  localization datasets. Installations that need the same custom map must create
  it in their own preferences.
- Existing saved Z scaling factor choices are preserved; the both-off behavior is the fresh
  install default.

## Install

Download the build for your platform from the release assets.

From source (Python 3.10–3.12):

```
poetry sync
poetry run minflux-viewer
```

Build the Windows one-folder executable from that same environment:

```powershell
.\.venv\Scripts\python.exe -m pip install pyinstaller==6.19.0
.\.venv\Scripts\python.exe -m PyInstaller --clean --noconfirm minflux_viewer.spec
```

Do not use a bare global `pyinstaller` command; it cannot see packages installed
only in the project environment.
