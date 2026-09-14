# Orthogonal view mode — implementation handoff

**Status:** working, tested, uncommitted. **Date:** 2026-09-13.
**Audience:** an independent agent session doing review, independent validation, and possible expansion.

This document is the full context for the orthogonal ("Ortho-View") mode added to the
MINFLUX Viewer's **Scatter** and **Render** windows. It records what was decided and
why, what was built, what is verified with numbers, and what is knowingly left open.

> Read `CLAUDE.md` ▸ *Viewer rules* ▸ **Orthogonal view mode** first — it is the
> canonical project record (≈90 lines) and is deliberately more prescriptive than this
> file. This document adds the planning narrative, the measurements, and the traps.

---

## 1. What it does

Shows XY, YZ and XZ of a 3-D dataset at once, in the arrangement of the legacy MATLAB
`interactive_render_MINFLUX.m` and of ImageJ's `Orthogonal_Views`:

```
        col 0            col 1
row 0   [ XY ]           [ YZ ]        YZ shares XY's vertical Y axis
row 1   [ XZ ]           (empty)       XZ shares XY's horizontal X axis
```

* **Scatter** — embedded: one window, 2×2 grid, side panes hidden until the mode is on.
* **Render** — floating: the two side panes move into their own top-level windows,
  fitted onto the current monitor.

Each pane is a **projection** over the axis it does not show (not a slice).

### How to reach it

| View | Entry |
|---|---|
| Scatter | right-click ▸ **View ▸ Ortho-View** (5th projection, after `3D`) |
| Render | right-click ▸ **View ▸ Ortho-View** (below `3D`), then **Keep aligned** / **Crosshair** |

Offered only for 3-D localization data (`_ortho_available`); disabled with a tooltip
otherwise, and a saved ortho state falls back to XY on a 2-D dataset.

---

## 2. File map

| File | Lines | Role |
|---|---|---|
| `minflux_viewer/ui/ortho_view.py` | 1478 | **All shared plumbing.** Pure geometry + `OrthoPanes`, `OrthoCrosshair`, `FloatingPaneWindow`. View-agnostic: imports no render/scatter widgets. |
| `minflux_viewer/ui/render_window.py` | — | Render integration (~40 methods, grep `_ortho`), plus the shared `render_scalar`. |
| `minflux_viewer/ui/scatter_window.py` | — | Scatter integration (~16 methods, grep `_ortho`/`_pane_`). |
| `tests/test_ortho_view.py` | 956 | 38 tests — pure geometry + scatter integration. |
| `tests/test_render_ortho.py` | 688 | 26 tests — render integration, crosshair, axis direction, reconstruction identity and decorated-frame geometry. |
| `tests/test_render_ortho_floating.py` | 662 | 23 tests — screen fitting, initial frame separation, crosshair-centred 3-D zoom coupling, display linking, focus, overlay, grids and depth navigation. |

**87 dedicated ortho tests**, plus advanced-render integration in
`tests/test_precision_render.py`. Two adjacent suites sit on top of this mode rather
than inside it: `tests/test_ortho_roi.py` (29 — volume ROIs drawn and edited in the
side panes) and `tests/test_ortho_rotation.py` (7 — the fourth cell, §7.8).

### Public surface of `ortho_view.py`

Pure (no Qt widgets needed to test):
`ORTHO_AXIS` · `AXIS_COLUMNS` · `ORTHO_AXIS_COLUMNS` · `PANE_PLANES` · `PANE_CELLS` ·
`SIDE_PLANES` · `PLACEMENTS` · `axis_columns()` · `axis_labels()` ·
`ortho_pane_labels()` · `grid_stretch()` · `floating_geometry()` ·
`fit_ortho_plot_rects()` · `plot_item_of()`

Qt: `OrthoPanes` · `OrthoCrosshair` · `FloatingPaneWindow`

---

## 3. Planning — decisions, and what was rejected

### 3.1 Embedded grid, not three `RenderWindow`s

The original request was to copy ImageJ: open two more render windows. **Rejected**,
with reasons measured in this codebase:

* Each `RenderWindow` owns a **512-entry `PhysicalTileCache`** and a **4-thread
  `RenderScheduler`** → 1536 tiles and 12 threads for one dataset.
* **32 call sites** in `main_window.py` key on `_render_windows` assuming *one window
  per dataset*; `_render_window_for_dataset` would become ambiguous, breaking Shift+C,
  `_show_render`, `_swap_render_mode` and the LUT dialog.
* Three `RoiOverlayController`s would share `source_view="render"` for one dataset, and
  `roi_visible_in` gates on **family + dataset, not plane** — so every ROI would draw in
  all three panes with geometry meaningless in two. *(Per-pane controllers were added
  later — `ui/ortho_roi.py::OrthoPaneOwner` — and they answer this objection without a
  second `RenderWindow`: each pane states its own `roi_view_columns()`, so the geometry
  is read through the axes that pane actually shows.)*

ImageJ's own implementation shows the cost of the architecture: `arrangeWindows()`
**polls up to 2.5 s** for the windows to exist, and `updateMagnification()` loops
`zoomIn()`/`zoomOut()` until magnifications agree. A `QGridLayout` gets the alignment
*by construction* — same row ⇒ same height, same column ⇒ same width.

The floating arrangement was later added anyway (user asked to test it), but as a
**placement of the same panes**, not as extra render windows. The panes are *moved*,
never duplicated, so both arrangements share one projection path, one set of links and
one isotropy rule.

### 3.2 Side panes are lightweight, not a second render engine

A side pane is a `PlotWidget` + `ImageItem`. The whole projection is one call to
`render_scalar` over the localizations already inside the XY viewport.

**One `SpatialGrid` serves all three panes.** The channel grid is built on `(x, y)` for
the XY orientation, and an XZ projection wants the locs whose X **and** Y are on screen
— which is exactly the XY viewport query `_render_direct` already performs. So **no
per-orientation grid is built**, which is why the mode did not require the render
engine's singleton view state (`_channel_locs_xyz`, `_tile_grid_x0/y0`, `_bounds_xy`,
`_last_scalar_tile`) to be de-singletonised.

### 3.3 The primary pane keeps the window's existing plot object

`_image_view` / `_plot_2d` are never reparented out of their cell. Everything keyed to
them — ROI controller, scale bar, plot profile, colorbar docking, B&C, volume window,
TIFF export, manual alignment — is untouched. `_active_plane()` resolves
**`Ortho-View` → `XY`** at every site that asks "which projection is this" for an
interactive purpose.

---

## 4. Invariants — do not break these

Each was a real defect found by measurement. The numbers are the evidence.

### 4.1 The YZ pane is **transposed** against the standalone YZ projection

Standalone `YZ` plots Y horizontally, Z vertically (`AXIS_COLUMNS["YZ"] == (1, 2)`).
Beside XY it must share XY's **vertical** Y, so `ORTHO_AXIS_COLUMNS["YZ"] == (2, 1)`.

⚠ Getting this wrong draws **Z data inside a Y range**. The link forces the ranges to
agree, so nothing errors and no range assertion fails — the pane's content is merely
squeezed into a narrow band. Found by looking at a screenshot, not by a passing test.
`axis_columns(plane, ortho=)` is the single resolver; `_pane_targets()` the single
caller.

### 4.2 Pinned axis metrics, or linked axes silently drift

pyqtgraph maps a linked range through the **ratio of the two panes' plot rectangles**.
Panes whose tick labels differ in width (Z's `200` vs Y's `12000`) get different
rectangles and therefore different data ranges at the same nm/px.

**Measured: a stable 36–74 nm of drift** — one feature at two screen positions.
`AXIS_WIDTH_PX = 62` / `AXIS_HEIGHT_PX = 30` on every pane fixes it to **0.00 nm**.

⚠ **Panes must also agree on axis _visibility_.** A hidden axis occupies no space
whatever `setWidth` says. The render view hides axes by default while a plain
`PlotWidget` shows them — measured **XZ 528 px wide against XY 590**.

### 4.3 Full isotropy, computed — not pyqtgraph's aspect lock

Z is placed at `primary_scale_nm_per_px()`, the XY pane's own nm/px.
**Measured: XY 16.8924 / XZ 16.8924 / YZ 16.8924.**

⚠ **Equal *ranges* are not equal *scales*.** XZ shows Z down its height and YZ across
its width; measured **210 px vs 162 px**, so one Z range rendered at **7.36 vs
9.55 nm/px** — a feature **30 % taller in one pane than it was wide in the other**.

⚠ Not delegated to `setAspectLocked`: the lock resolves a violated aspect by expanding
the *other* axis, and on a side pane that axis is **linked to XY**, so it could push the
primary pane's range around.

**This is what makes the Z scaling factor visible.** `ds.loc_nm` always carried the
calibrated Z and the cache was always invalidated — but a pane that auto-fitted Z made
halving `cali.z_scaling_factor` change *nothing on screen*. Price: Z can be **clipped**
at deep zoom; `OrthoPanes.depth_clipped` reports it in the status line.

### 4.4 Across windows the shared axis is **copied, not linked**

Inside one layout the plot-rect ratio is exact by construction. Across top-level windows
the window manager gets a vote: a **2 px** height difference on the YZ pane remapped the
Y range by 0.6 % — **27.9 nm of silent data misalignment**.

`_copy_shared_ranges` assigns the primary's range verbatim when floating. Data agreement
is then exact (**0.00000 nm**) whatever the pixels do; the cost is that a side pane's own
scale can differ by that pixel error, so it is very slightly anisotropic instead of
slightly wrong about where things are.

### 4.5 A shared axis has a **direction**, and neither link nor copy carries it

YZ mirrors XY's `invertY`; XZ never inverts (its vertical is Z, nobody else's axis).
The render view shipped without this and its **YZ pane was vertically flipped** — the
same point sat **51 px apart** while the Y ranges agreed to 0.000 nm the whole time.

Direction is asserted on **global screen position** (`_screen_pos` in
`test_render_ortho.py`), for both origin preferences and both placements, in render and
scatter. ⚠ Must be re-applied after `_ensure_ortho_panes` — the panes do not exist when
`_apply_y_axis_direction` first runs.

### 4.6 ONE reconstruction, dispatched by the concrete render window

For the base `RenderWindow`, the XY pane and both side panes call
`RenderWindow.render_scalar`.  The production UI, however, opens
`PrecisionRenderWindow`, whose XY pane uses `render_advanced_tile` for the selected
Histogram/Bilinear/Smoothed/Fixed Gaussian/Localization-precision Gaussian/Voronoi
method.  Side panes therefore dispatch through `_render_ortho_channel_scalar`:
the base implementation calls `render_scalar`, while `PrecisionRenderWindow`
projects the selected rows and calls `render_advanced_tile` with plane-correct
precision and fixed-sigma axes.

This distinction matters.  The first unification attempt only changed the base
class, so the real selectable-method window continued to draw both side panes with
the inherited smoothed reconstruction.  Its `_compose_from_cache` override also
omitted the base orthoview refresh hook, leaving side panes stale after display-only
changes.  `test_precision_render_ortho_side_panes_follow_selected_method` now changes
every selectable method and inspects both side-pane scalar and displayed RGBA images.

The panes can no longer take different methods.
**Two look-alike implementations drifted three times** before this:

* XY switches to a **per-localization Gaussian below `PER_LOC_SWITCH_COUNT` (500
  visible locs)** while the side panes always histogrammed — so zooming past 500 made
  the primary go smooth and the side panes stay blocky.
* The side panes sized from `_last_px_nm` — whatever XY rendered at *last*, on a
  **separate debounce** — so they were a render behind at a resolution belonging to a
  different view.

Each pane now derives pixel size by the same rule from its **own** view
(`max(span) / _RENDER_SIZE`, `_RENDER_SIZE = 800`) and sigma by the same rule
(`_sigma_for_plane`). Regression asserts the two paths produce the same plane **bit for
bit** (`max|diff| = 0`) — anything looser is what let them drift.

### 4.7 Leaving the mode must zero the grid stretch

Hiding the side panes is not enough: a hidden widget still holds its row/column open
while a stretch factor is set. Measured — the primary kept **720 px of a 900 px**
window. `grid_stretch(active)` is pure and asserted without a window.

### 4.8 Never auto-range a side pane

Both axes are driven (shared from the primary, Z from `apply_depth_range`). With
interaction on, an auto-range triggered merely by **drawing the projection** pushed a
new range back onto the primary — *rendering the side panes moved the XY view*.

### 4.9 The crosshair never moves by itself

Verified against the ImageJ source: `Orthogonal_Views` assigns `crossLoc` in exactly
four places — `run()` (initial), `mouseDragged`, `mouseWheelMoved`, `keyPressed` — plus
a `setCrossLoc` setter, and **nothing repositions it** on zoom, pan, scroll,
magnification change or resize, nor is there any code that recentres a view on it.

`_centre_crosshair_on_view` only *seeds* it when it has no position. *(An earlier
version had it follow the XY view centre. That was wrong and is gone.)* It does still
**anchor the side panes' Z**, which is what makes a sub-area zoom close in on the marked
point rather than the middle of the data.

An explicit **side-pane Z pan/zoom is navigation**, however: it now sets the durable
depth anchor and the crosshair's Z coordinate. Previously the visible side range moved,
but `_render_ortho_sides_now()` re-anchored it to the unchanged marker on the next
debounce, making the interaction appear broken. The callback is guarded while its
InfiniteLines move, and the status QLabel is not changed mid-gesture — relayout there
would trigger `sigResized`, reapply the old range, and undo the pan.

### 4.10 Expensive scientific side renders run in cancellable background jobs

`PrecisionRenderWindow` dispatches XZ and YZ through dedicated
`PrecisionRenderScheduler` / `VoronoiFieldScheduler` instances. Their pools have
different names from XY, so cancelling a side gesture cannot clear queued primary
tiles. Side schedulers deliberately do not clear their shared pool on cancel either,
because that would cancel another render window's jobs; stale queued tasks exit at
their generation check before computation. Leaving Ortho-View and closing the window
invalidate the side batch. Precision Gaussian checks cancellation during stamping.
Qhull itself is non-interruptible once entered, so Voronoi cancellation discards an
in-flight field when it returns rather than blocking teardown.

Side scalar rasters have a 64 MiB/48-item LRU. Voronoi fields have a separate LRU whose
capacity grows to retain every channel in the current two-plane batch; otherwise an
overlay with more than four channels could evict its early fields before rasterization.

---

## 5. Behaviour reference

### Scatter (embedded)

| Aspect | Behaviour |
|---|---|
| Proportions | 0.6 / 0.4 both axes (`PRIMARY_STRETCH=3`, `SIDE_STRETCH=2`) |
| Side panes | Interactive; only `sigRangeChangedManually` couples panes, while programmatic link updates are ignored (**see §7.1**) |
| Projection | Cropped to the XY viewport (`_ortho_row_filter`), **before** decimation |
| Crop lifts | While the view auto-ranges (`_ortho_view_rect()` → `None`) |
| Reset View | Lifts the crop **before** fitting, or the fit re-fits to the cropped subset and can never escape |
| Colorbar | Hidden in the mode, gutter handed to the panes (92 → 184 px at the default window); user's choice while in the mode is preserved on exit |
| Labels | X named once at the bottom, Y once on the left; both Z labels kept |
| Status | `XY · YZ · XZ (visible region, Z clipped) | Z scaling 0.50` |

### Render (floating)

| Aspect | Behaviour |
|---|---|
| Placement | Floating only; `OrthoPanes` still implements embedded for scatter |
| Screen fit | `fit_ortho_plot_rects` scales the primary **down only** until the set fits the monitor (was overflowing by 214 px); the final sticky alignment preserves the chrome-aware 8 px frame gap |
| View carry-over | Range captured/restored around the resize — **0.27 nm** drift, centre exact |
| From XZ/YZ | Carried as X/Y/Z: X kept exactly, the depth gate becomes Y, Z keeps its centre |
| Side panes | **Interactive** (`interactive_sides=True`); a zoom couples all 3 axes, anchoring the missing X/Y axis on the visible crosshair (or its current centre), while Z interaction updates the depth anchor/crosshair |
| Depth slider | **Hidden**; "All" forced so no gate survives invisibly |
| Focus | Activating any of the three raises all three, primary last (wired on **both** `FloatingPaneWindow.changeEvent` and the render window's own) |
| Closing a side window | **Leaves the mode**, back to XY (Fiji behaviour) |
| Display state | B&C, colormap, invert, per-channel LUT, visibility, white background and grid visibility all share |
| Crosshair | On by default; status line carries one 3-D coordinate |
| Overlay | Works; channel list stays in the primary window |

**Display-range linking is automatic and hidden.** Manual levels are identical numeric
limits in all panes. Auto B/C transfers XY's black/white **clipping percentiles** to
each projection: raw count limits are not comparable after collapsing a different axis
and made side panes saturate flat white. This gives one exposure policy without adding
a user-facing toggle.

---

## 6. Verified measurements

| Property | Result |
|---|---|
| Embedded alignment | XZ width == XY width (669/669); YZ height == XY height (653/653) |
| Floating alignment | dx = dy = dw = dh = **0 px** |
| Linked axis agreement | **0.000 nm** (X and Y), both placements |
| Isotropy | XY / XZ / YZ all **16.8924 nm/px** |
| Side zoom coupling | 0.5× zoom-in and 1.8× zoom-out from XZ and YZ retain one scale; previous opposite pane error was exactly **2×** after a 0.5× zoom |
| Axis direction | **0 px** on both shared axes, both origins, both placements |
| Reconstruction identity | `max|diff| = 0` at three zoom levels |
| Screen fit | 964 × 947 inside 1920 × 1032 |
| Window overlap on entry | **0 px**; XZ/YZ frames clear XY by the requested 8 px (was 50 px overlap at XZ after the final alignment pass discarded chrome) |
| Per-pane render cost | 6.9 ms @ 30 k · 56 ms @ 500 k · 2.1 s @ 20 M visible locs |
| `histogram2d` vs `bincount` | 4.8× (bincount faster; **not** used — see §7.3) |

---

## 7. Known limitations and open items

### 7.1 Scatter side interaction and the crop guard
Scatter side panes are interactive. The earlier attempt inferred "showing everything"
from pyqtgraph auto-range, but a side pane's programmatic push disables auto-range; that
engaged the crop, moved Z, fed back into the side pane and ran ranges to **±100 µm**.
`_ortho_show_all` is now explicit and changes only on `sigRangeChangedManually`, Reset,
or mode entry. `OrthoPanes` likewise couples a side range only from that manual signal,
so native-link/programmatic echoes cannot masquerade as a second user gesture.

### 7.2 The physical tiled LOD regime is intentionally not shared yet
The production `PrecisionRenderWindow` already uses tiled LOD for XY at every zoom;
XZ/YZ now use cancellable, cached viewport jobs capped at 800 px, but do not reuse
physical tiles across pans. A side tile has an extra dependency that XY's `TileKey`
does not express: XZ changes when the **collapsed Y range** changes, and YZ changes
when the **collapsed X range** changes. Merely adding an orientation/grid would return
stale projections after a primary pan. A correct extension needs either the collapsed
axis range in every side key (limited reuse and potentially explosive cache churn) or,
preferably, a 3-D voxel/pyramid source from which all three projections are derived.
The new background jobs remove the UI-blocking failure mode while that larger data
structure remains deferred.

### 7.3 `bincount` fast path was removed
The side panes once used a `bincount` linear-index histogram (4.8× faster). It was
dropped so both paths share `render_scalar`, which keeps XY's `np.histogram2d` — the
bin-edge semantics differ and changing XY's long-standing binning to chase speed was
judged not worth it. Reinstating it would have to change **both** paths together and
re-prove the bit-identity test.

### 7.4 The screen fit is best-effort
Three windows have three sets of minimum sizes, so on a monitor too small for them the
fit shrinks as far as it can and still overflows — a cost the embedded grid does not
pay. Tests skip the containment assertion when the reported screen is < 1200 × 900.

### 7.5 Crosshair is an indicator, not a slice navigator
The side panes are projections, so moving the crosshair changes nothing in them; it
reports *where* along the collapsed axis a feature sits. Making it a true navigator
(ImageJ semantics) means **adding a slab mode first**, after which `OrthoCrosshair`
gains the callback that re-renders on move. This is the most natural expansion.

### 7.6 Renaming `ORTHO_AXIS` broke old saved scatter states
`ORTHO_AXIS` doubles as the scatter combo's persisted value. It is now `"Ortho-View"`,
so a state saved as `"Ortho"` falls back to XY — handled silently by `findText`.

### 7.7 Not carried to scatter
The crosshair, the status-line coordinate and the floating placement are render-only.
Wiring the crosshair into scatter is small (`OrthoCrosshair` is already shared).

### 7.8 The bottom-right cell
0.4 × 0.4 of the embedded grid. It now holds a **rotating projection**
(`OrthoPanes.set_extra_pane`, `EXTRA_CELL`, `ui/ortho_rotation.py`): one matrix multiply
on the localizations already in hand, re-projected into the same nm/px as its
neighbours, so at 0° it reproduces the XY view. It answers what three fixed projections
cannot — orthogonal silhouettes are ambiguous about depth ordering. **Embedded only**,
so scatter has it and render (floating-only) does not; floating hands the whole page
back to the primary pane instead. The colorbar therefore no longer has this cell to
move into.

### 7.9 Image-stack orthogonal slicing is out of scope
ImageJ/Fiji already handles ordinary image stacks well. MINFLUX Viewer keeps this mode
focused on localization projections and deliberately does not duplicate stack-slice,
reslice or slab tooling.

---

## 8. Traps for the next agent

**⚠ The full suite crashes natively at a moving target on this tree, and it is not
caused by any logic in this feature.** Measured: adding the mode took
`test_qt_lifecycle_regressions.py::test_qt_heavy_module_exits_cleanly[test_lut_dialog.py]`
from **0/14 failures on the clean tree to ~7/18**. Eight bisection experiments removed
each suspected cause in turn — extra panes (built lazily), the layout wrapper, the new
imports, the whole `closeEvent` addition, every `_active_plane()` substitution — and
**each still failed at ~50 %**, while the new imports alone and a deliberate null
perturbation both gave **0/6**. The decisive run: **16 kB of method definitions that
nothing ever calls reproduced it at 4/6.** The trigger is the *size* of the module,
presumably via GC or class-dict timing. Do not bisect a crash here against a change that
merely grew a Qt module; measure both trees over ≥6 runs first. That one test is
deselected in the suite runs quoted above.

**⚠ Teardown tier matters.** Side panes use the **safe** tier (`dispose_plot_widgets`),
not `close_plot_widgets`. The aggressive tier reparents and `deleteLater()`s, leaving an
orphan that outlives its page — it reproduced as a Qt abort inside a later, unrelated
test.

**⚠ Range-based assertions are not enough.** Three separate defects passed every
range-based test: the transposed YZ pane, the flipped Y direction, and the diverged
reconstruction. All three needed a *screen position* or *pixel* assertion, or simply
looking at a rendered PNG. **Render a screenshot and look at it.**

**⚠ Measuring a pane mid-layout gives stale numbers.** Bitten four times: the fit had to
be deferred a turn (measured the pane at 577 px where it was about to be 966); the
depth-range apply needed a `sigResized` re-apply; the crosshair test needed the view to
settle; and the reconstruction test had to read `viewRange()` with **no settling between
`_render()` and the read**.

**⚠ Do not measure a gate or a filter by summing the pane's composed RGB.**
`_compose_rgba_for(auto=True)` transfers the primary clipping percentiles to each
projection, so removing data can brighten what remains and the total is not a count.
Count rows.

**⚠ A patch script that asserts mid-way can leave nothing written.** One multi-edit
script aborted on its last assertion before `write_text`, silently dropping five earlier
edits that had "succeeded". Apply edits individually, or write before asserting.

---

## 9. Suggested validation for an independent session

1. **Re-derive the invariants independently** — especially §4.1 (transposition), §4.5
   (direction) and §4.6 (reconstruction identity). Write the assertions from scratch
   rather than reading the existing ones; all three shipped broken past tests that
   looked adequate.
2. **Look at rendered output.** `win.grab()` on each window composited at
   `frameGeometry()` gives a faithful picture of the real arrangement.
3. **Exercise real data**, not just synthetic Gaussians — an `.msr` with a genuine 3-D
   MINFLUX acquisition, and a multi-channel overlay.
4. **Check the regimes this skips**: physical side-tile reuse (§7.2), 2-D datasets,
   and a dataset with `z_scaling_factor ≠ 1`. Image/TIFF orthoview is deliberately out
   of scope (§7.9).
5. **Run the ortho files repeatedly** (`pytest tests/test_ortho_view.py
   tests/test_render_ortho.py tests/test_render_ortho_floating.py`) and the full suite
   several times, reading §8 before attributing any native crash.

---

## 10. Repository state

Uncommitted. Ortho-related paths:

```
new:      minflux_viewer/ui/ortho_view.py
new:      tests/test_ortho_view.py
new:      tests/test_render_ortho.py
new:      tests/test_render_ortho_floating.py
modified: minflux_viewer/ui/render_window.py
modified: minflux_viewer/ui/precision_render_window.py
modified: minflux_viewer/ui/scatter_window.py
modified: tests/test_precision_render.py              (advanced-method linkage)
modified: tests/test_view_context_menus.py        (View menu now lists "Ortho-View")
modified: tests/test_overlay_alignment_ui.py      (stub needs _active_plane)
```

`CLAUDE.md` carries the canonical record of this feature (its *Orthogonal view
mode* section), but it is **local-only and gitignored** by project convention — it
never appears in `git status` and must not be staged or committed. Same for
`AGENTS.md` and everything under `docs/`.

⚠ The working tree used to carry **unrelated** in-progress work from earlier sessions
(`attribute_separation_dialog.py`, `peak_channels.py`, `channel_labels.py`,
`roi_highlight.py` and their tests). That is now committed separately, as
`feat(channel): rework attribute separation around channels of two kinds` — but the
lesson stands: name the paths explicitly when committing, because anything left in the
tree is otherwise swept into whichever commit runs next.

Per project convention: **no `Co-Authored-By:` trailers**, on any machine, in any
session, whatever the tool.
