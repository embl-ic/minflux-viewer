# MINFLUX Viewer scripting & plugin API

`mfv` is the published Python surface for automating the running viewer. A
script in the Script Editor and a plugin from your own folder use the same
object: the script imports it as `mfv`, the plugin receives it as `ctx`.

**API version 1.3.** Adding a namespace, a function or a defaulted keyword is a
minor bump; removing or renaming anything is a major one. A plugin declares
what it was written against and the loader refuses to run it against an API
that has moved.

```python
import mfv

ds  = mfv.data.active()
loc = mfv.data.loc(unit="nm", filtered=True)
mfv.ui.log(f"{ds.name}: {len(loc)} localizations")
```

---

## Contents

- [Namespaces](#namespaces) · [`data`](#mfvdata) · [`roi`](#mfvroi) ·
  [`results`](#mfvresults) · [`plot`](#mfvplot) · [`view`](#mfvview) ·
  [`ui`](#mfvui) · [`run`](#mfvrun) · [`journal`](#mfvjournal) ·
  [`record`](#mfvrecord) · [`plugins`](#mfvplugins)
- [Writing a plugin](#writing-a-plugin)
- [Using libraries the viewer does not ship](#using-libraries-the-viewer-does-not-ship)
- [Compatibility and limits](#compatibility-and-limits)

---

## Namespaces

| | |
|---|---|
| `mfv.data` | datasets, attributes, coordinates, filters — read *and* write |
| `mfv.roi` | regions of interest, masks, crops, what falls inside them |
| `mfv.results` | the shared results table |
| `mfv.plot` | line / scatter / histogram / image windows |
| `mfv.view` | the dataset-owned viewer windows |
| `mfv.ui` | log, status, parameter dialogs, file pickers, message boxes |
| `mfv.run` | background execution, progress, cancellation |
| `mfv.journal` | record a step so it reaches *Generate Method Text* |
| `mfv.record` | record viewer work as a reusable Python script |
| `mfv.plugins` | list plugins, write up or re-run their last run (since 1.3) |

Statistics are deliberately **not** wrapped. SciPy is bundled; use
`scipy.stats` directly.

---

## `mfv.data`

```python
mfv.data.datasets()                 # every loaded dataset, in viewer order
mfv.data.active()                   # the active dataset, or None
mfv.data.get(ref=None)              # None | index | name | dataset object
mfv.data.index(dataset=None)        # its viewer index
mfv.data.properties()               # num_loc, num_traces, dimensions, ...

mfv.data.attr_names()
mfv.data.attr(name, itr="auto", vld_only=True, filtered=True)
mfv.data.loc(unit="nm", filtered=True, transformed=False)

mfv.data.add_attr(name, values)     # a computed per-localization attribute
mfv.data.create(xyz, tid=None, attrs=None, name="...")

mfv.data.filter_mask()
mfv.data.filter_specs()
mfv.data.set_filter(specs, replace=True)
mfv.data.clear_filter()
```

### Choosing an iteration

`itr` uses the application's own tokens, so a script and the UI agree on what a
value *is*:

| `itr` | meaning |
|---|---|
| `"auto"` *(default)* | what the viewer shows — `cfr`/`efc` at their effective measurement iteration, everything else the last valid value |
| `"last"` | the value at the global-maximum iteration. For `cfr`/`efc` this is the raw ≈0 value, which is usually not what you want |
| `"effective"` | `cfr`/`efc` at the iteration they were actually measured at |
| an `int` | that specific 0-based iteration |
| `"sum"` / `"average"` | pooled over all of a localization's iterations |

`vld_only=False` reads through the raw store including invalid rows. Those do
not align to the materialized rows, so `filtered` no longer applies and there
is no longer one value per localization.

### Coordinates

`loc()` returns `(N, 3)`, and the two units are **not** a conversion of one
another.

`unit="nm"` is the display view: the Z scaling factor applied, which is what
the viewer draws. `transformed=True` additionally applies the dataset's overlay
transform — **that is the frame ROI geometry is expressed in**, so use it when
comparing against a ROI.

`unit="m"` is the canonical raw store, `loc_x`/`loc_y`/`loc_z` exactly as the
file recorded them. Z scaling is a *view*, so dividing the nm values by 1e9
would give calibrated metres, off by the Z factor from what you asked for.
`transformed=True` is refused with `unit="m"`: an overlay transform is defined
in display space, so there is no raw-metre answer.

Returned coordinate arrays are independent writable copies. Internal views can
share read-only cached arrays, but editing a `loc()` result does not edit the
dataset. Runtime dataset keys and cache objects are not part of the public API.

### Adding an attribute

```python
loc = mfv.data.loc(filtered=False)  # add_attr needs every materialized row
mfv.data.add_attr("radius_nm", (loc[:, 0] ** 2 + loc[:, 1] ** 2) ** 0.5)
```

One value per localization of the **unfiltered** dataset. Afterwards it is
selectable in the histogram, filter, colour-by and attribute-plot lists like
any built-in attribute.

### Creating a dataset

```python
ds = mfv.data.create(points_nm, tid=trace_ids, name="drift corrected")
```

Coordinates are treated as final: the Z scaling factor is pinned to 1.0 so the
post-load chain does not estimate one and scale Z a second time.

### Filters

A spec is `{attribute, mode, lo, hi}` plus optional `itr`, `lo_inc`, `hi_inc`.
`mode` is `"per loc"` or one of the trace read-outs (`"trace mean"`,
`"trace median"`, …).

```python
efo = mfv.data.attr("efo", filtered=False)
mfv.data.set_filter([
    {"attribute": "efo", "mode": "per loc",
     "lo": float(np.median(efo)), "hi": float(efo.max())},
])
```

`replace=False` appends to the user's existing filter instead of replacing it.
Specs are persisted in the same re-evaluable form the Filter dialog writes, so
they survive save and reload.

---

## `mfv.roi`

```python
mfv.roi.list(region_only=False)     # stored ROIs for this dataset
mfv.roi.active()                    # what a command would act on
mfv.roi.selected()
mfv.roi.add(type, geometry, view="render", name="", color=None)
mfv.roi.remove(roi) · mfv.roi.select(rois)

mfv.roi.mask(roi)                   # boolean, one entry per localization
mfv.roi.points_in(roi)              # (N, 3) inside a region
mfv.roi.attr_in(roi, name)          # one attribute, inside a region
mfv.roi.crop(roi, exact_shape=True) # a new dataset restricted to the region

mfv.roi.geometry(roi) · mfv.roi.bounds(roi)
```

Coordinates are **display nm**. Geometry uses the store's own keys:
`{"bounds": [x, y, w, h], "angle": deg}` for a rectangle or oval,
`{"points": [...]}` for a polygon or line, `{"point": [x, y, z]}` for a point.

`mask`, `points_in`, `attr_in` and `crop` need a *region* type — `rectangle`,
`oval`, `polygon`, `freehand`. A line or point encloses no area and yields an
all-False mask rather than an error.

Coordinates, masks and attributes are row-aligned, which is what makes this
work:

```python
roi   = mfv.roi.active()
efo   = mfv.roi.attr_in(roi, "efo")
pts   = mfv.roi.points_in(roi)
assert len(efo) == len(pts)
```

A ROI created here is scoped to its dataset and its view family at creation. An
unscoped ROI is drawn in every view of every dataset, which is why `view=`
exists — `"render"`/`"scatter"` are one family, `"histogram"` and `"attribute"`
are their own.

---

## `mfv.results`

The shared table. **The same name always returns the same table**, so a plugin
run twice appends to the table the user already has rather than stacking
windows.

```python
t = mfv.results.table("Sites")
t.add_row(id=1, radius_nm=52.4, note="rim")
t.add_rows([{...}, {...}])          # far cheaper than a loop
t.from_arrays(x=xs, y=ys)           # replace contents; lengths must match
t.show()
t.save_csv(path)

t.columns() · t.column(name) · t.rows() · t.to_dict() · len(t)
t.clear() · t.hide() · t.close()
mfv.results.tables() · mfv.results.close(name) · mfv.results.close_all()
```

Rows may introduce columns as they go; a cell never written reads back as
`None` and saves as empty. `to_dict()` is ready for `pandas.DataFrame`.

Closing the window discards only the display — the named table survives, so a
script can keep appending and call `show()` again.

---

## `mfv.plot`

Every constructor returns a chainable handle.

```python
mfv.plot.line(x, y, label=None, title=None)
mfv.plot.scatter(x, y, color_by=None, symbol="o", size=5.0)
mfv.plot.hist(values, bins=None, bin_width=None)
mfv.plot.image(array, extent=(x0, y0, x1, y1), colormap="hot")
```

```python
(mfv.plot.line(r, excess, label="observed")
    .series(r, null, label="null", style="--")
    .labels(x="pair distance (nm)", y="observed / null")
    .legend(True)
    .grid(True)
    .show())
```

Handle methods: `series`, `labels`, `legend`, `log(x=, y=)`, `grid`, `range`,
`clear`, `show`, `close`, `save_png(path)`. Line styles are `-`, `--`, `:`,
`-.`; symbols are `o s t d + x p h star`.

`image()` takes `extent` in nm so the image lands in data coordinates rather
than pixel indices.

---

## `mfv.view`

```python
mfv.view.render(orientation="XY")
mfv.view.scatter(color_by="tid")
mfv.view.histogram(attribute="efo")
mfv.view.attribute_plot(x="idx", y="efo", z=None, c=None)
mfv.view.console() · mfv.view.script_editor()
mfv.view.snapshot(path, view="render")
mfv.view.refresh()
```

Viewer windows are dataset-owned: one per dataset, reused. Calling `render()`
twice raises the existing window rather than opening a second.

`snapshot()` captures what is on screen — the exact zoom, filters and channels
the user is looking at — not a re-render.

`mfv.view.volume()` and `mfv.view.lut()` are published but **deferred in 1.0**;
they raise with the reason. Use the render window's own View menu and the LUT
dialog.

---

## `mfv.ui`

```python
mfv.ui.log(message, level="INFO", dataset=None)
mfv.ui.status(text, fraction=None)

params = mfv.ui.ask({"radius_nm": 25.0, "method": ["mean", "median"],
                     "normalise": True})
if params is None:
    return                          # the user cancelled

mfv.ui.info(msg) · warn(msg) · error(msg) · confirm(msg) -> bool
mfv.ui.choose_file() · choose_files() · choose_dir()
mfv.ui.drop_files(paths)             # replay an interactive OS drag-and-drop
```

`ask()` builds the dialog from the defaults you give: `bool` → checkbox,
`int`/`float` → spin box, `str` → line edit, a list → dropdown. It returns a
dict of the same keys, or **`None` if cancelled** — check for that before
proceeding. Use it rather than building your own dialog: it keeps a plugin free
of Qt.

---

## `mfv.run`

Long work belongs here, not on the GUI thread.

```python
def work(task):
    total = len(items)
    for i, item in enumerate(items):
        task.check_cancelled()              # cooperative stop
        task.progress(i, total, "measuring")
        ...
    return result

def done(result):
    mfv.results.table("Out").add_rows(result.rows).show()

mfv.run.background(work, on_done=done, name="my analysis")
```

The worker **touches no Qt widget and no viewer state**: capture what you need
first, and apply the result in `on_done`, which runs on the GUI thread. A
failure goes to `on_error` if you give one, and to the Log otherwise — never
silently.

The returned handle's `is_running()` is true only while queued, running or
stopping, and false after success, failure or cancellation. Tasks are visible in
the monitor while queued. Error callbacks receive a `RuntimeError` subclass with
`exception_type`, `worker_traceback` and `code` diagnostics; these are plain values,
not retained worker frames. A stop request does not undo an already committed file.

The task appears in **Help ▸ Task Monitor** with a working *Request stop*
button. Cancellation is cooperative: a thread cannot be killed, so the work
stops when it next calls `task.check_cancelled()` (or `task.progress()`, which
is also a checkpoint). Work with no checkpoint runs to completion, and the
monitor says so.

---

## `mfv.journal`

```python
mfv.journal.record("analysis", "Measured dimer distances",
                   dataset=ds, radius_nm=25, replicates=99)
```

Records the run so it reaches **Plugins ▸ Generate Method Text**, and writes a
Log line so it is visible while the session runs. Pass the parameters actually
used, not the defaults. Categories: `load`, `filter`, `transform`, `analysis`,
`export`, `plugin`, `other`.

---

## `mfv.record`

**Plugins ▸ Macro Recorder** captures known viewer actions as ordinary `mfv`
Python. Display-only calls can be removed with *Silent mode*; actions whose
parameters have not yet been instrumented remain visible as `TODO` comments
rather than becoming misleading code.

```python
mfv.record.start()                     # clears and begins a new recording
mfv.record.start(clear=False)          # resume the visible recording

# Plugins can contribute an exact outcome while recording is active.
mfv.record.step(
    "mfv.data.add_attr('radius_nm', radius_nm)",
    gui_class="gui_free",
    summary="Added the radius attribute",
)

steps = mfv.record.steps()             # immutable RecordedStep snapshots
code = mfv.record.script(silent=True)  # omit GUI_ONLY presentation calls
mfv.record.stop()                      # pause; captured steps remain
```

The GUI classes are `gui_free` (no interaction), `gui_result` (the recorded
data outcome of an interaction), and `gui_only` (presentation). Record the
accepted value or resulting geometry/matrix, never mouse coordinates or
keystrokes. Calling `step()` while paused is a no-op, so plugin code can call it
unconditionally.

---

## Writing a plugin

Plugins live in folders **outside** the application, so they are yours and are
not shared with other users of the same machine:

| Root | Path |
|---|---|
| Per-user *(the default)* | `%APPDATA%\MINFLUX Viewer\plugins`<br>`~/Library/Application Support/MINFLUX Viewer/plugins`<br>`~/.local/share/minflux-viewer/plugins` |
| Portable | `plugins/` beside the application |
| Extra | anything added in **Preferences ▸ Plugin ▸ Plugin folders** |

### Tier 1 — a single file

Drop `My_Project_Tool.py` in the folder. It becomes one menu entry, *My Project
Tool* (underscores become spaces, Fiji-style). Define `run(ctx)`, or just write
a script — importing it runs it.

### Tier 2 — a folder with a manifest

```toml
# plugin.toml
[plugin]
id          = "embl.hlyb_dimer"      # reverse-DNS, unique, stable
name        = "HlyB/D Dimer Distance"
version     = "1.2.0"
author      = "…"
description = "Short sentence shown as the menu tooltip."

[menu]
path     = "Plugins > HlyB"          # optional nesting
entry     = "Dimer Distance..."
keywords = ["dimer", "ecoli"]        # extra Command Finder search terms

[run]
module   = "main"                    # main.py in this folder
callable = "run"                     # def run(ctx): ...

[requires]
mfv_api = ">=1.0,<2.0"
app     = ">=0.4.2"
python  = ["scikit-image>=0.24"]
```

#### `[method]` — the write-up, declared once

A plugin that declares a `[method]` block gets two things for free: *Generate
Method Text* renders a complete method section for every run of it, and the
Macro Recorder emits a line that re-runs it with the same answers.

Write the prose **once**, with parameter values left as `{slots}`. The values a
run actually used are substituted at write-up time, so the text cannot drift
out of step with the numbers it quotes.

```toml
[method]
title       = "Short-range pair analysis of inferred label sites"
description = "One or more paragraphs: what the method is and what it claims."
workflow    = [
  "Traces with at least {min_locs} localization(s) contributed a centroid.",
  "Repeats were consolidated within {merge_nm}.",
]
limitations = "What the result is not. Optional but strongly encouraged."
citations   = [{ text = "Author et al. 2026", url = "https://doi.org/..." }]

[[method.inputs]]
name        = "merge_nm"      # what a {merge_nm} slot refers to
label       = "same-site diameter"
unit        = "nm"            # appended automatically: {merge_nm} -> "4 nm"
description = "Shown beside the value in the Parameters block."

[[method.inputs]]
name   = "null_model"
replay = false                # configured by the run, not asked for

[[method.outputs]]
name  = "band_ratio"
label = "observed/null ratio"
```

Then record what the run used, once, at the end of it:

```python
ctx.journal.record("analysis", "Ran the pair analysis", dataset=ds,
                   min_locs=10, merge_nm=4.0, band_ratio=2.04)
```

That is all. The runner knows which plugin is executing, so the entry is
attributed to it automatically; nothing else has to be wired up.

**Rules worth knowing.**

- A slot with no recorded value renders as `<name not recorded>` and is listed
  at the end of the section. It is never silently dropped — a paragraph that
  quietly loses a number it promised to state cannot be checked.
- Declared **inputs** are what a replay feeds back to your `ctx.ui.ask()`
  dialog. Mark a setting the run *derived* rather than asked for with
  `replay = false`, so a recorded script does not appear to set it.
- Declared **outputs** are reported but never replayed: they are results.
- `{slots}` are a plain scan, not `str.format`, so a brace in your prose is
  safe and no attribute access is reachable from a template.
- A malformed block is a manifest error: the plugin is listed with the reason
  rather than silently producing no method text.
- The block is optional. Without one, a run still appears in the generated
  text as its plain log line, exactly as before.

### `mfv.plugins`

Reading it back, or writing up without opening the menu (since 1.3):

```python
mfv.plugins.list()                      # id, name, menu path, has_method
mfv.plugins.find("embl.my_tool")        # its entry; raises if not installed
mfv.plugins.method("embl.my_tool")      # its [method] block, or None
mfv.plugins.method_text("embl.my_tool") # the last run of it, written up
mfv.plugins.last_run("embl.my_tool")    # the values that run used
mfv.plugins.run("embl.my_tool", answers={"min_locs": 10})
```

`run(answers=…)` stages the answers for the **next** `ctx.ui.ask()` the plugin
makes, so the dialog is answered from the recording instead of by hand. Keys
the plugin does not ask for are ignored, and keys it asks for but the recording
lacks keep the plugin's own defaults — so a recording still replays after the
plugin gains a parameter, with the new one at its default.

### An example `main.py`

```python
# main.py
def run(ctx):
    roi = ctx.roi.active()
    if roi is None:
        ctx.ui.error("Draw a region first, then run this again.")
        return

    p = ctx.ui.ask({"radius_nm": 25.0})
    if p is None:
        return

    def work(task):
        return analyse(ctx.roi.points_in(roi), **p, progress=task.progress)

    def done(result):
        ctx.results.table("Sites").add_rows(result.rows).show()
        ctx.journal.record("analysis", "Measured sites", **p)

    ctx.run.background(work, on_done=done)
```

### Rules worth knowing

- **Import your own helper modules relatively**: `from .utils import thing`. An
  absolute `import utils` is a single global name and cannot be isolated, so it
  is refused rather than silently sharing another plugin's module.
- The plugin body is imported the first time its menu entry is used, so a
  plugin that imports a heavy library costs nothing at startup.
- A plugin that cannot run still appears in the menu and explains why when
  clicked — a broken manifest, a missing module, an API mismatch, a missing
  package.
- The first time you run a plugin from a new folder you are asked to accept
  that folder. Plugins are ordinary Python with full access to your files and
  the application, exactly like a Fiji plugin. There is no sandbox.
- After editing a plugin, use **Preferences ▸ Plugin ▸ Rescan now**.

---

## Using libraries the viewer does not ship

**Preferences ▸ Plugin ▸ Python packages.** Two routes:

**Point at an environment you already have.** Add its `site-packages` folder.
Nothing is downloaded. The folder must be built for the same Python
(CPython 3.12 on the same platform); a mismatch is refused with the reason.

**Let the application install a package.** Type a requirement (`scikit-image`,
`pandas==2.2.0`) and it is fetched into a managed per-user folder.

Both are **appended** to the import path, so the application's own NumPy,
SciPy, Qt and HDF5 keep import priority. Managed installation protects `numpy`,
`scipy`, `scikit-learn` (including `sklearn`), `PyQt5/6`, `PySide2/6`, their Qt/SIP/
Shiboken runtime distributions, `pyqtgraph`, `h5py`, `zarr` and `tifffile`.
External roots containing a conflicting Qt binding are refused; other protected
packages produce a warning. Appending paths is not a native-library sandbox or
a guarantee that arbitrary packages are ABI-compatible. Named index requirements
are supported; direct URLs/local wheels are not accepted by the managed installer.

Only pre-built wheels are installed, and dependencies are not pulled in
automatically; add any you need explicitly. Deep-learning stacks (PyTorch,
TensorFlow) are out of scope here: they are very large, pin their own NumPy and
ship their own OpenMP runtime, which collides at the DLL level. Run those in a
separate environment.

---

## Compatibility and limits

Every name the original scripting API published still works:
`mfv.get_active_dataset()`, `get_datasets()`, `get_dataset()`, `get_attr()`,
`get_loc()`, `log()`, `show_console()`, `mfv.viewer.*` and `mfv.ScriptError`.
They are aliases onto the namespaces, so there is one implementation of each
behaviour.

One behaviour change: `get_attr(name)` now resolves `cfr`/`efc` to their
effective iteration, so it agrees with the Filter dialog where the old direct
read did not.

- Scripts are trusted local Python, not sandboxed.
- A script runs on the GUI thread. Use `mfv.run.background` for the slow part
  rather than expecting the whole script to run off-thread — only you know
  which part of it is free of Qt.
- `mfv.ScriptError` (an alias of `mfv.ApiError`) is what the API raises for
  misuse, with a message written for whoever wrote the script.
