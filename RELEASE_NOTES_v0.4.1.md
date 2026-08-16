# MINFLUX Viewer v0.4.1

v0.4.1 focuses on the plotting and LUT experience: colormaps now belong to the
application instead of an external plotting library, custom gradients can be
created from the LUT dialog, and the Render and Localization Scatter menus now
follow the same layout.

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

## RIMF default for new installations

A fresh installation now starts with both **estimate RIMF from anisotropy** and
**use fixed RIMF value** switched off. Z coordinates are therefore left exactly
as recorded until the user chooses a correction policy.

This does **not** overwrite an existing user's saved preference. The preference
migration path was also corrected so one-shot migrations apply to genuinely old
saved settings, not to a brand-new copy of the current defaults.

## Packaging

- The direct Matplotlib dependency is removed from `pyproject.toml` and its
  transitive plotting packages are removed from `poetry.lock`.
- The Windows PyInstaller build uses a native application `.ico`, avoiding the
  former build-time Pillow requirement for converting a PNG icon.
- The application and Poetry package versions are both **0.4.1**; the shared
  PyInstaller spec reads that application version for packaged metadata.

## Compatibility notes

- Existing saved views using legacy colormap names remain supported.
- Custom gradients are application preferences, not embedded into exported
  localization datasets. Installations that need the same custom map must create
  it in their own preferences.
- Existing saved RIMF choices are preserved; the both-off behavior is the fresh
  install default.

## Install

Download the build for your platform from the release assets.

From source (Python 3.10–3.12):

```
poetry sync
poetry run minflux-viewer
```
