# 3-D ROI — handoff for independent verification

**Feature:** volume (3-D) regions of interest for MINFLUX Viewer, drawn through the
orthogonal view.
**Status:** core complete and in use; projection-hull drawing and cylinder integration
completed 2026-09-15; remaining extensions are in §10.
**Companion document:** `ORTHO_VIEW_HANDOFF.md` — the view this feature is drawn in.
Read it first if the orthogonal mode is unfamiliar; this document assumes it.
**Canonical record:** `CLAUDE.md` ▸ *Volume (3-D) ROIs* and ▸ *Multi-point ROI*.
That file is local-only and gitignored; this one is tracked.

The reader of this document is expected to **disagree with it where it is wrong**.
Every number below was measured, and §11 says how to measure each one again.

---

## 1. What was asked for

Six ROI types/families that are genuinely three-dimensional:

| # | Asked for | Became | Family it is drawn from |
|---|---|---|---|
| 1 | cuboid | `cuboid` | rectangle |
| 2 | sphere | `sphere` — an **axis-aligned ellipsoid** | oval |
| 3 | cylinder | `cylinder` — an axis-aligned elliptic cylinder | oval |
| 4 | polyhedron | `polyhedron` — three editable projection constraints (legacy cross-section stacks still load) | polygon / freehand |
| 5 | 3-D line | the existing `line` / `polyline` / `freehand_line`, whose vertices already carry a per-vertex depth | line family |
| 6 | 3-D point | the existing `point`, plus a new `points` (ImageJ multi-point) | point |

Types 1–4 are the volume records and are collectively `VOLUME_ROI_TYPES`. Types 5–6 needed
no new record type — a line vertex and a point were already `[x, y, z]` — but they did
need the side panes to be drawable in, which is what made them 3-D in practice.

**"Sphere" is deliberately the user-facing name for an ellipsoid.** A drag almost never
produces equal radii, and *sphere* is what a user looks for in a toolbar. `cylinder` is
nested under the same oval family: its drawing-plane silhouette is the oval and its two
side silhouettes are rectangles.

Free ROI orientation remains unmodelled after an explicit burden review. It is not just
three 2-D angle fields: the record needs a 3×3 orthonormal frame, masks must inverse-
transform points, bounds/silhouettes/meshes must project the rotated solid, and editing
needs a 3-D orientation gizmo with defined resize semantics. A rotated cuboid may project
to a hexagon and a tilted cylinder no longer has the current oval/rectangle projections.
That is a separate feature rather than a safe extension of the axis-aligned pane handles.

### 1.1 Projection-hull update (2026-09-15)

A newly drawn polyhedron no longer starts as one polygon plus rectangular side bands. It
stores an editable polygon in all three canonical projections:

```python
{"representation": "projection_hull", "primary_plane": "XY",
 "projections": {
   "XY": {"points": [[x, y], ...], "source": "manual"},
   "XZ": {"points": [[x, z], ...], "source": "auto_hull"},
   "YZ": {"points": [[y, z], ...], "source": "auto_hull"}}}
```

The drawn primary polygon is kept exactly. With at least ten contained localizations,
the other two panes receive independently computed convex hulls of those same XYZ rows,
expanded by a cheap margin derived from the primary polygon's mean nonnegative bounding-
box clearance (at least 1 nm). With no points **or too few points**, both side panes use
the same existing four-corner fallback from the primary extent and the crosshair/visible-
depth seed. Every projection is then an ordinary editable polygon: vertices can be moved,
added or deleted, and an edited generated outline changes its source to `manual`.

Membership is the exact intersection of the XY, XZ and YZ polygon prisms. Convex
projection constraints also yield an exact half-space-intersection mesh in Scatter 3-D
and the rotating pane. Concave constraints retain exact membership but intentionally do
not receive a fake convex mesh.

---

## 2. The idea

A volume ROI is **drawn in one plane and bounded in the other two**. The user draws the
familiar 2-D shape; the third dimension is taken from the localizations the shape already
covers. Nothing asks the user to type a Z range.

That has one consequence that drove the whole design: a single projection cannot show
what is being made — the seeded depth would be invisible until the ROI was filed. So
**selecting a 3-D tool turns the orthogonal view on**, and the shape is visible in all
three panes as it is made.

### Retired to make room

`60bd445` removed *Convolution (3D)* (`analysis/conv_segmentation_3d.py`,
`ui/conv_segmentation_3d_dialog.py`) and the old *Process ▸ ROI ▸ 3D ROI* dialog
(`core/roi_3d.py`, `ui/roi_3d_dialog.py`), 2,265 lines including their tests.

The 3-D ROI dialog **created no `RoiRecord`** — it only cropped — so the ROI Manager, the
save formats, Convert/Fit and the highlight path never saw its output, and it was
unconnected to the toolbar. Conv-3D was a real 3-D matched filter, but the 8 M
total-voxel cap forces a voxel far coarser than the structures on any realistic field of
view. Conv-3D is in `BACKLOG.md` with the condition for restoring it (region-focused
convolution, so the voxel can stay fine); the 2×2 viewer layout is written up in
`docs/ortho-2x2-viewer-retired.md`.

⚠ **One fact outlives both, and it is the foundation of this feature.** The retired
`roi_3d.PLANE_AXES["YZ"] == (2, 1)` disagreed with `scatter_window`'s `{"YZ": (1, 2)}`.
Both were correct *for their own view*. That is why a 3-D ROI stores geometry in **named
data axes**, never as a 2-D shape plus a plane string.

---

## 3. Design decisions, and what was rejected

### 3.1 Geometry lives in named data axes

```python
cuboid      {"x": [lo, hi], "y": [lo, hi], "z": [lo, hi]}
sphere      {"center": [x, y, z], "radii": [rx, ry, rz]}
cylinder    {"axis": "Z", "center": [x, y, z], "radii": [ru, rv], "height": h}
polyhedron  {"representation": "projection_hull", "projections": {"XY": …, "XZ": …, "YZ": …}}
```

Projection polygons use canonical ascending axis order: XY → (X,Y), XZ → (X,Z),
YZ → (Y,Z). `projection_polygon` / `set_projection_polygon` reverse YZ at the ortho
pane boundary. The earlier contour-stack schema remains supported for native JSON and
the legacy Add Cross-Section workflow; its polygon vertices use `cross_axes(axis)`.

The tree carries two incompatible readings of the string `"YZ"`:

```
ORTHO_AXIS_COLUMNS = {'XY': (0, 1), 'YZ': (2, 1), 'XZ': (0, 2)}   # ortho pane: Z is horizontal
AXIS_COLUMNS       = {'XY': (0, 1), 'XZ': (0, 2), 'YZ': (1, 2)}   # standalone: Y is horizontal
```

Four numbers plus a plane name cannot say which. So every query takes **axis columns**:
`volume_silhouette(record, h_axis, v_axis)`, `project_flat_record(record, h_axis, v_axis, …)`,
`set_volume_extent(record, columns, bounds)`, `translate_volume(record, {column: delta})`.
A view states what it shows; there is no convention left to pick wrongly.

### 3.2 Legacy interpolation between cross-sections is angular resampling

`radial_profile` describes a cross-section as a centroid plus a radius at
`DEFAULT_ANGLES = 64` fixed angles. Intermediate sections interpolate centroid and radii.

Rejected, with reasons:

| Alternative | Why not |
|---|---|
| Pair vertices between levels | Fails on unequal vertex counts and self-intersects. |
| Rasterised signed-distance field (ImageJ `RoiInterpolator`, the RSOM plugin) | Robust, but it **imports a pixel size**, and a localization dataset has no native resolution — membership would then depend on a grid chosen for smoothing. It also needs marching squares, and `scikit-image` is not a dependency. |

The price: a cross-section must be **star-shaped about its centroid**. `radial_profile`
raises `NotStarShaped` rather than quietly dropping a lobe (`strict=False` takes the
outer boundary, filling concavities).

The payoff is why it was chosen for legacy contour stacks: **membership is closed-form and fully vectorised**.
`_polyhedron_mask` interpolates the centroid and radius profile at each point's own
stacking coordinate and compares the point's own radius against the boundary — no
per-point polygon is ever constructed.

### 3.3 One fallback seeding rule for all volume shapes

The derived dimension begins as an *interval* — a cuboid extent, ellipsoid diameter,
cylinder height, or projection-hull fallback depth. `seed_interval` is shared:

```python
MIN_SEED_LOCS = 10; MIN_SEED_THICKNESS_NM = 20.0; EMPTY_SEED_FRACTION = 0.5
```

* ≥ 10 contained localizations → their **full extent**, so nothing drawn over is left out.
* Fewer → `EMPTY_SEED_FRACTION` of the **visible** range about the crosshair value.
  Drawing in a void beside a structure is normal; three localizations are noise.
* Never thinner than `MIN_SEED_THICKNESS_NM` — a zero-thickness region selects nothing.
* Clamped into the visible range.

⚠ A near-edge centre therefore has its seed clamped and its centre moved. What is
guaranteed is that the interval still **covers** the requested centre.

### 3.4 The seed reuses `compute_roi_selection`

"Which localizations are under this 2-D shape" is exactly what that method answers. A
private reimplementation would drift about filtering, channel visibility and the viewport
crop. `_seed_depth_interval` takes its depths from that mask and its bounds from the
owner hook `roi_depth_range()`.

### 3.5 Z is frozen at draw time

A volume ROI's Z does **not** track `cali.z_scaling_factor`. Rescaling is an explicit
multi-select ROI Manager command, *Scale Z*. The draw-time factor is recorded in
`context["z_scaling_factor"]` — **recorded, never auto-applied** — because without it the
old value cannot be recovered, so "update with the Z scaling factor" would not be
computable at all and the command could not pre-fill the exact ratio.

⚠ `scale_z` scales about **z = 0**, not the ROI centre, because the data rescales about
the origin (`z_calibrated = loc_z * 1e9 * factor`).

### 3.6 A 3-D tool turns the orthogonal view ON, never off

Leaving the mode when the tool is deselected would yank the layout out from under someone
who had arranged it. `MainWindow._enter_ortho_for_volume_tool` prefers the **focused**
coordinate view, then this dataset's render, then its scatter; with neither open it opens
a **scatter**, because scatter's ortho is an embedded 2×2 grid while render's opens two
more top-level windows.

### 3.7 A 3-D tool is refused on a 2-D dataset

`MainWindow._volume_tool_allowed` refuses before anything is drawn, with the reason in
the status bar and the Log. Refusing the gesture after the fact — draw, then revert — is
the worse experience.

---

## 4. File map

| File | Lines | Role |
|---|---|---|
| `minflux_viewer/core/roi_volume.py` | — | **The geometry core.** Qt-free. Schemas, membership, seeding, silhouettes, meshes, interpolation, edits. |
| `minflux_viewer/core/roi_projection.py` | 149 | How a **flat** ROI appears in a plane it was not drawn in. |
| `minflux_viewer/ui/ortho_roi.py` | 307 | The side panes: outlines, the per-pane controller adapter, the ownership rule. |
| `minflux_viewer/ui/ortho_rotation.py` | 60 | The grid's fourth cell: a rotating projection. |
| `minflux_viewer/ui/roi_overlay.py` | — | Drawing, hit-testing, editing (grep `volume`, `_view_axes`, `_SELECTING_TYPES`). |
| `minflux_viewer/ui/render_window.py` / `scatter_window.py` | — | Integration (grep `_pane_roi_controllers`, `roi_pane_coords`, `compute_roi_selection`). |
| `minflux_viewer/ui/main_window.py` | — | Tool gating, auto-ortho, *Add Cross-Section…* (grep `_volume_tool_allowed`, `_enter_ortho_for_volume_tool`, `_add_cross_section_to_roi`). |

### Tests — 183 dedicated

| File | Lines | Tests | Covers |
|---|---|---|---|
| `tests/test_roi_volume.py` | — | 93 | Geometry, membership, seeding, projection hulls, meshes, edits, the property read-out. |
| `tests/test_ortho_roi.py` | — | 54 | The side panes, end to end, **driving real mouse events**. |
| `tests/test_multi_point_roi.py` | 212 | 16 | The `points` record type. |
| `tests/test_roi_projection.py` | 123 | 13 | Flat ROIs seen edge-on. |
| `tests/test_ortho_rotation.py` | 59 | 7 | The rotating fourth pane. |

### Public surface

```
roi_volume     VOLUME_ROI_TYPES · AXIS_INDEX · AXIS_NAMES · PROJECTION_AXES · DEFAULT_ANGLES ·
               MIN_SEED_LOCS · MIN_SEED_THICKNESS_NM · EMPTY_SEED_FRACTION ·
               PLANE_NORMAL_AXIS · FLAT_TO_VOLUME · NotStarShaped ·
               cross_axes · plane_in_plane_axes · seed_interval · radial_profile ·
               cross_section_at · roi_volume_mask · volume_bounds ·
               volume_silhouette · volume_mesh · volume_from_flat · projection_hull_from_flat ·
               projection_polygon · set_projection_polygon · scale_z · translate_volume ·
               set_volume_extent · add_cross_section · convex_hull_polyhedron ·
               volume_geometry_text
roi_projection FULL · DEGENERATE_LINE · DEGENERATE_POINT · PLANE_COLUMNS ·
               flat_extent · project_flat_record
ortho_roi      pane_outline · pane_owns_record · OrthoRoiOutlines ·
               OrthoPaneOwner · attach_pane_controllers
```

---

## 5. How drawing works

A volume tool **draws its 2-D counterpart**, and the finished shape is lifted:

```python
FLAT_TO_VOLUME = {'rectangle': 'cuboid', 'oval': 'sphere',
                  'polygon': 'polyhedron', 'freehand': 'polyhedron'}
```

`cylinder` deliberately is not another value in this one-to-one fallback table: it uses
the same oval gesture as `sphere`, and the active tool is passed explicitly to
`volume_from_flat(..., volume_type="cylinder")`. New polyhedron drawings take the
separate `projection_hull_from_flat` path described in §1.1.

During the drag the rubber band, the handles and the status read-out are the existing
ones. The third dimension is seeded only on the **finishing gesture** (release for
cuboid/sphere/cylinder, the polygon's right-click for polyhedron), when there is a finished shape
to measure the data against. `volume_from_flat(flat_type, geometry, plane, interval)`
does the lift.

The derived axis is the one **normal to the drawing plane** —
`PLANE_NORMAL_AXIS = {'XY': 'Z', 'XZ': 'Y', 'YZ': 'X'}` — so the rule is plane-agnostic
and there are no per-view special cases. Asserted by a test:
`plane_in_plane_axes(plane) == cross_axes(PLANE_NORMAL_AXIS[plane])`.

### The item kind matches the shape, not merely the outline

An axis-aligned box and ellipsoid **are** a rectangle and an oval in every plane. A
cylinder is an oval down its named axis and a rectangle from either side. `_make_item`
therefore gives all three analytic shapes bounding-box handles, while a projection hull
uses a polyline whose vertices edit that pane's stored constraint. `set_volume_extent`
writes an analytic resize back on the **two visible axes only**; a projection has nothing
to say about the third.

### A legacy prism is a polyhedron with one level

`{"levels": [one], "thickness": t}` — same record type, mask and editor, so it upgrades to
a multi-level contour stack with no migration. *Process ▸ ROI ▸ Add Cross-Section…* draws the
outline again at another Z; a **second level supersedes `thickness`** (with one level the
thickness *is* the extent; with several the outermost levels are, and keeping both would
be two answers to one question). Re-adding at an existing level replaces it. The command
refuses projection-hull records, whose three polygons are edited directly in the panes.

⚠ The Z is **asked for**, not taken from the view: render has a crosshair to read and
scatter does not, and a level at a depth the user never chose is worse than one typed in.

---

## 6. Where it is wired

| Consumer | State |
|---|---|
| `compute_roi_selection` (render + scatter) | ✅ volume branch → `roi_volume_mask`, and takes `columns=` so a side pane asks about its own plane |
| `roi_crop.compute_crop_mask` (Shift+D) | ✅ intersects the ROI's own Z with an explicit `z_range` |
| `channel_labels.roi_mask_for_record` | ✅ channel separation accepts a volume ROI |
| `main_window._active_region_record` | ✅ `REGION_TYPES │ VOLUME_ROI_TYPES` |
| ROI Manager — list, Update, Delete, Property, **Scale Z** | ✅ |
| Native JSON ROI set | ✅ round-trips exactly |
| `record_to_imagej` | ⚠ **raises by name** — ImageJ has no volume ROI type, and no projection of one *is* the ROI. See §9.7 |
| `roi_region_mask` (the 2-D entry point) | ⚠ **raises** for a volume type rather than returning all-False — all-False is right for a line and a trap for a volume |
| Convert | offers **to Point** only, giving the real 3-D centre |
| Resize | refuses up front, naming the type |
| Fit | silently ignores a volume ROI — §10.4 |
| Particle extraction | rectangle-only — §10.5 |
| Plot Profile | out of scope (open-line ROIs only) |

### The side panes

`ortho_roi.attach_pane_controllers` gives each side pane its own `RoiOverlayController`,
sharing the window's `RoiStore`, so a ROI drawn in any pane is the same record everywhere
and the panes differ only in the axes they read it through. `OrthoPaneOwner` is the
adapter: a **`QObject`** (the controller parents itself to its owner), parented to the
real window so the chain dies with the view, delegating everything through `__getattr__`
except the handful of answers that are about *which plane this pane shows*.

**`pane_owns_record(record, plane)` is the one rule** for what a pane's own controller
draws and edits: a **volume** ROI (geometry in named data axes, so any pane can read it
through its own columns), or a **flat** ROI drawn in that pane. Everything else gets the
display-only **dashed** outline, which `project_flat_record` places honestly.

An unfiled volume ROI is still a single draft rather than a store record. Pointing into
another pane transfers that draft before PyQtGraph caches the hover/drag target, replacing
its decorative projection with the correct editable rectangle/oval/polygon item. A direct
click is also a fallback hand-off. The decorative copy is removed from the pane that now
owns the editable draft but retained in the third pane. A stored ROI needs an activation
step: ROI Manager *Update* reads the active adapter, so every edit-mode left press activates
its pane before the item moves.

⚠ Dashed is load-bearing. A flat ROI has no extent on the axis it was drawn against, so
edge-on it is a segment at its recorded depth. Solid would make it indistinguishable from
a volume ROI of zero thickness, and a user would reasonably read it as constraining Z
when it does not.

---

## 7. Measured

On the reference file
`1_sample_A_1-100_seqTrk-3D-Seb_Octahedron_100_pho_exc_10.mat` (45,105 localizations, 3-D):

| Measurement | Value |
|---|---|
| Cuboid drawn in XY | seeds Z −607 → 398 nm, selects **4,680** |
| Polyhedron drawn in XY | seeds thickness **1914.5 nm**, selects **9,327** |
| Cuboid drawn in the **XZ** pane | takes X/Z from the drag, seeds Y 2543 → 6695 nm, selects **296** |
| Rotating pane at 45° | draws all **45,105** points; at 0° reproduces the XY view |

On a 5,000-point synthetic Gaussian cloud (σ = 300 nm), one cuboid
`x ±200, y ±150, z ±80`:

| Consumer | Rows |
|---|---|
| `compute_roi_selection` | **206** |
| `roi_crop.compute_crop_mask` | **206** |
| `channel_labels.roi_mask_for_record` | **206** |
| crop ∩ explicit `z_range=(−20, 20)` | **52** |
| ROI Properties "Localizations within" | **206** |

Three independent consumers agreeing on 206 is the check worth repeating.

Other measurements:

| Measurement | Value |
|---|---|
| Multi-slice interpolation, levels at r = 600 and r = 150 | midway radius **370.7 nm** (centroid + radii interpolated, so not exactly the 375 a naive linear midpoint gives) |
| Two-level polyhedron selection | 2 levels, `thickness` correctly gone, selects **1,348** |
| `convex_hull_polyhedron` (9 levels) | encloses **~99 %** of its own points |
| Point drawn in the XZ pane | `[-247.0, -11.6, 42.2]` — X and Z from the click, Y derived |
| Volume ROI geometry read-out | `X=-200, Y=-150, Z=-80, W=400, H=300, D=160` |
| `scale_z(0.5)` on z = [−80, 80] | `[-40, 40]` — about z = 0 |

---

## 8. Defects found, and how

Every one of these was found by **driving the application**, not by reading the code.
That is the method worth reusing: three of the five were invisible to tests that called
the controller's own methods.

### 8.1 A drawn volume ROI was unselectable, and invisible in the side panes — `63a7e29`

Two causes. `_bounds` knows only `bounds` / `point` / `points`, so it returned
`(0, 0, 0, 0)` for a named-axis geometry — a degenerate box at the origin, so every
hit-test missed and the shape could not be selected, right-clicked or deleted.
`_volume_view_bounds` takes the extent from the ROI's own **silhouette in the current
plane**. Separately, a **draft is not in the store**, so `rois.changed` never fires for it
and the side panes were refreshed only by store signals — a freshly drawn shape appeared
there only when something unrelated happened next. `_notify_overlay_changed()` calls the
owner's `on_roi_overlay_changed` from `_finalize_draft_selection` **and** `_clear_draft`.

**Reported by the user**, not by the suite.

### 8.2 A cuboid deformed into a polygon when a corner was dragged — `0c3150e`

`_make_item` returned `FilledPolyLineROI` for *all* volume types, so a cuboid *was* a
polygon on screen: its handles moved single vertices. Fixed as §5. Also fixed here:
`volume_silhouette` ran every polyhedron through `radial_profile`, which rounded the
corners off a single-level prism (a square came back as 64 radial samples); with one
level the polygon is now returned verbatim.

**Reported by the user.**

### 8.3 A volume ROI converted to a point landed at the origin — `58aae2a`

`available_conversions` offered *to Point* — by falling through every branch, not by
intent — and `convert_roi` then took the 2-D centroid path, found no `bounds`/`points`
key and returned `(0, 0)`. A **silent wrong answer**. Now the real 3-D centre, with the
volume branch explicit. `can_resize` likewise said yes while `enlarge_shrink_roi` said no,
so the dialog opened, took a number, then refused — with a message about angle ROIs.

**Found by probing the deferred surfaces** rather than trusting that "deferred" meant
"refused".

### 8.4 A volume ROI read as empty wherever `_REGION_TYPES` was the gate — `823e4f7`

Three sites spelled "this shape selects localizations" as the literal four 2-D region
types, so a volume ROI fell through all of them: ROI Properties never showed
*Localizations within* and never recomputed a stale selection, and replacing a volume
draft with a line left the old highlight on screen. `_SELECTING_TYPES` is the set those
three mean. The geometry read-out was the same class of defect as 8.1 — both property
dialogs ran a volume ROI through the 2-D `_bounds` and reported
`0 vertices, bbox X=0, Y=0, W=0, H=0`, describing the shape as *empty* rather than as
*undescribed*. `volume_geometry_text` serves both dialogs, so they cannot drift.

### 8.5 Drawing in the XZ/YZ panes did nothing in the render view — `8e7f707`

Three defects in one report.

1. **The render view had no pane controllers at all.** Only scatter attached them, while
   the volume tools prefer the focused view and then **render before scatter** — and a
   render view is what opens when a dataset loads. So the common path armed a 3-D tool,
   turned the mode on, announced *"draw in any pane"*, and ignored every drag in XZ
   and YZ.
2. **A shape drawn in a side pane was filed as an XY shape.** `normalize_roi_record` reads
   `self.roi_view_plane()`, and reaching it through `OrthoPaneOwner.__getattr__` binds it
   to the *window*, which answers `"XY"` for the whole mode. Binding the same function to
   the adapter makes every `self.` in it resolve to the pane.
3. **Every ROI was drawn twice in each side pane**, once by the pane's controller and once
   by the outline layer — and worse, `roi_visible_in` gates on family + dataset, **not
   plane**, so the XZ controller drew a rectangle drawn in XY verbatim. Measured: a
   rectangle whose bounds were `[-200, 5000, 400, 300]` was drawn at `v = 5000` on the
   XZ pane's **Z** axis, where the data spans ±1200 — and it was *draggable* there, so
   the next drag would have written that back into the record. `pane_owns_record` is the
   fix, consulted from `_record_in_scope`, which both the draw loop and the hit-test come
   through.

**Reported by the user.** ⚠ The tests had asserted the geometry by calling `_set_draft`
and `_promote_draft_to_volume` directly, so they proved the lift and never that a **drag**
does anything. The replacements send real press/move/release to the pane viewport,
parametrised over both views.

### 8.6 Test-side lessons

* Two older tests asserted the *mechanism* (an outline curve exists) rather than the
  behaviour (the ROI is visible in the pane), and broke when the controllers took over the
  volume shapes. `_shown_in_pane` now asks "by whichever layer draws it".
* `RoiOverlayController._fmt` is a `@staticmethod`; assigning it into a plain class body
  re-binds it as an instance method, so a stand-in must use `staticmethod(...)`.
* Pane controllers are now disposed in both windows' `closeEvent` — each holds a viewport
  event filter and three store connections, and a store change arriving at a controller
  whose plot is half torn down is the documented route to a Qt abort.

### 8.7 XZ/YZ looked editable but a pending ROI was still owned by XY

`18d2a9e` attached side-pane controllers only in Scatter; `8e7f707` added the missing
Render controllers and proved that a real side drag creates a draft. Neither covered the
next gesture. A draft is deliberately absent from `RoiStore`, so it remained owned by the
controller that created it and its other two projections were `PlotCurveItem` decorations
with no handles. Also, an edit-mode left press did not call `activate()`: a stored side-pane
item could move visually while ROI Manager *Update* still read the untouched active XY
adapter.

Controller peers now identify their common real window. Hovering into another pane transfers
the one pending **volume** draft within that group before PyQtGraph selects a drag target;
edit-mode left press also activates the pane before PyQtGraph handles the drag. Flat ROIs do
not transfer across planes because a projection cannot express a change on the collapsed
axis. The regression drives both side planes in Scatter and Render and verifies both a
pending cuboid edit and the adapter used by Manager Update.

### 8.8 Render highlighted ROI data only in XY

Scatter had a highlight `ScatterPlotItem` in every orthogonal pane, but Render had only
the primary pane's item. Its XZ/YZ windows therefore reconstructed the selected rows into
their raster and drew the ROI silhouette, but had no layer on which to mark those rows.

Each Render side pane now owns a highlight item above its image and below the ROI overlay.
The existing boolean ROI masks are resolved once, then painted as XZ `(X, Z)` and ortho YZ
`(Z, Y)` using `ORTHO_AXIS_COLUMNS`. Side highlights use the same XY-viewport crop as the
side reconstructions, are cleared when ortho is left or highlighting is disabled, and a
draft owned by a side controller counts as a draft from this Render view for the
`roi_highlight_in_roi` preference. The regression checks both side-axis mappings and the
side-draft preference path.

### 8.9 Volume geometry was absent from 3-D and rotating views

The fixed orthogonal panes displayed volume silhouettes, but Scatter's actual OpenGL 3-D
view contained only localization points/highlights, and the rotating fourth pane projected
only localization points. `core.roi_volume.volume_mesh` now supplies one tested
`(vertices, triangle faces, wire edges)` representation for both consumers: a cuboid is 8
corners / 12 real edges / 12 triangles, `sphere` is its stored axis-aligned ellipsoid,
`cylinder` is a watertight elliptic-cylinder mesh with outward winding for all three named
axes, and a convex projection hull is the exact half-space intersection of its projected
prisms. Legacy contour stacks and concave projection constraints remain deliberately
unmeshed until they have a tested CSG/tessellation contract.

Scatter draws all visible mesh-capable volume records as a translucent `GLMeshItem` plus opaque
wireframe in 3-D, and projects the same wire edges into the lower-right pane. Both follow
the ROI Manager's visibility, Show-all and selection gates and include a pending draft.
The rotation pane now has Play/Pause (1 degree per 100 ms), an **About X/Y/Z** selector,
and explicit labels: the vertical label names the held rotation axis while the horizontal
label states the two-axis mixture (for example `X cos θ + Z sin θ`). Play stops when the
user leaves Ortho or closes the window, and animation reuses the already displayed,
filtered and decimated points rather than scanning the full dataset on every timer tick.

---

## 9. Known limitations

**9.1 Legacy contour stacks require star-shaped cross-sections.** `radial_profile(strict=True)` raises
`NotStarShaped` for an outline a ray from the centroid crosses twice (a crescent, a U).
Between levels the interpolation is exact and vectorised; for such an outline there is
none.

**9.2 The legacy `convex_hull_polyhedron` command is sampled cross-sections, not a face list.**
`convex_hull_polyhedron(points, levels=9)` slices the hull at nine heights, because this
application's volume ROI *is* a stack of cross-sections and reusing that keeps the mask,
silhouette, Scale Z and the editor working unchanged. The reconstruction is slightly
tighter than the true hull between levels — measured ~99 % enclosure of its own points.

**9.3 The empty/sparse-region seed is clamped to the visible depth range.** When the
orthogonal crosshair is visible, its component on the derived axis is the requested
centre; otherwise the visible/data-range midpoint is used. Near an edge, clamping can
move the interval's midpoint, but the interval still contains the crosshair.

**9.4 A line or point drawn in a *side* pane gets a constant third coordinate.**
Measured: both vertices of a line drawn in XZ came out at Y = −11.6.
`OrthoPaneOwner.roi_depths_at` returns `None` per vertex by design — "a side pane falls
back to the view centre rather than inventing a second rule" — so such a line is planar in
the third axis. In the **XY** pane the same line gets a per-vertex, data-aware depth.

**9.5 A flat ROI is display-only in the side panes.** By design (§6), so a rectangle drawn
in XY cannot be edited from XZ.

**9.6 Projection-hull polyhedra are reshapeable; legacy contour stacks are not.**
Each projection-hull pane exposes its own polygon handles plus right-click Add/Delete
point. A body move translates all three constraints. The older multi-level silhouette is
the union of several levels, so there is still no unique level to receive a dragged
vertex; those records remain translation-only.

**9.7 ImageJ export refuses a volume ROI, and the refusal aborts the whole save.** A set
holding one cuboid cannot be written to `.zip` at all, rather than writing the 2-D ones
and naming the skipped. The ROI Manager catches the exception and reports it. Native JSON
is unaffected.

**9.8 The full test suite carries a pre-existing native flake**, unrelated to this work: a
`Fatal Python error: Aborted` inside pytest-qt's `_process_events`, most often in
`tests/test_lut_dialog.py`, at roughly a third to a half of runs on **every** tree
including untouched ones. A single green run proves little, and a bisect built on one run
per configuration will indict whatever it looked at first. See `CLAUDE.md` ▸ *Qt lifecycle*.

**9.9 Only convex projection-hull polyhedra have a 3-D/rotating mesh.** Concave projection
constraints still have exact point membership and editable fixed-pane outlines, but a
watertight Boolean-prism tessellation needs a separate CSG contract. Legacy interpolated
cross-sections are likewise left unmeshed rather than approximated as a box.

---

## 10. Not implemented

Ordered by my estimate of value; the next agent is welcome to disagree.

1. **Per-vertex depth for a line/point drawn in a side pane** (§9.4). Small, and it is the
   limitation a user meets first.
2. **Level editing** — `Add Cross-Section…` is the only level operation; no list, delete,
   reorder, or jump-to-Z. `add_cross_section` already replaces at the same `at` and keeps
   levels sorted, so the core half is small; the weight is the dialog and where it lives.
3. **Legacy contour-stack cross-section reshaping** (§9.6). Projection-hull polygons are
   already editable in every pane; the old level representation needs 2 first so the
   editor can name which level receives the change.
4. **Free 3-D orientation for cuboid/ellipsoid/cylinder.** This needs a shared 3×3 frame,
   rotated mask/silhouette/mesh maths, migration and a 3-D orientation editor; independent
   pane angles are not a valid orientation model.
5. **Fit for volume types** — silently ignored, gated on `_REGION_ROI_TYPES`.
   `min_enclosing.min_enclosing_ellipsoid` is already N-D, so this is record plumbing plus
   a decision about what "fit a cuboid to these points" should mean.
6. **Particle extraction is rectangle-only** — `extract_particles` skips anything else. A
   cuboid is the natural 3-D collection box; `ParticleData` would need to carry a Z crop.
7. **The rotating pane exists only in the embedded layout**, i.e. scatter. Render offers
   floating ortho only, which has no 2×2 grid and so no fourth cell.
8. **Concave projection-hull / legacy contour-stack surface tessellation** (§9.9) for the
   OpenGL and rotating displays.

---

## 11. How to verify

### 11.1 Run the suites

```bash
.venv/Scripts/python.exe -m pytest tests/test_roi_volume.py tests/test_roi_projection.py \
    tests/test_ortho_roi.py tests/test_ortho_rotation.py tests/test_multi_point_roi.py -q
```
All of these should pass. Then run the surrounding surface:
```bash
.venv/Scripts/python.exe -m pytest tests/test_ortho_view.py tests/test_render_ortho.py \
    tests/test_render_ortho_floating.py tests/test_roi_manager.py tests/test_roi_convert.py \
    tests/test_roi_crop.py tests/test_roi_keyboard.py -q
```
Expect 192 passed.
Full suite, excluding the two files documented as fragile (§9.8):
```bash
.venv/Scripts/python.exe -m pytest -q --ignore=tests/test_lut_dialog.py \
    --ignore=tests/test_particle_set.py
```
Last measured: **2840 passed, 7 skipped**.

### 11.2 Drive the application, do not call the handler

This is the check that found §8.5, and the one most worth repeating. Build a window, enter
ortho, arm a tool, and send **real** `QTest.mousePress` / `mouseMove` / `mouseRelease` to
`pane.viewport()` — not `_set_draft`. Assert on the resulting record. The pattern is in
`tests/test_ortho_roi.py::_drag`, parametrised over `["scatter", "render"]`.

⚠ **`QTest.mouseMove` does not deliver a hover.** Offscreen it produced only `Enter` on
the pane viewport, never the `MouseMove` with `NoButton` that the draft hand-off (§8.7)
keys on — so a probe built on it reports the feature dead when it is fine. Every pane
viewport has `hasMouseTracking() is True`, so a real pointer move does deliver it;
construct the event explicitly instead (`QMouseEvent(QEvent.Type.MouseMove, …,
Qt.MouseButton.NoButton, …)` sent with `app.sendEvent(viewport, event)`), as
`tests/test_ortho_roi.py::_move_to_view_point` does.

A click test should compare against **where the click landed**, not against a tolerance:
a widget click is an integer pixel, several nm wide at these zooms. Map the pixel back
through `mapSceneToView` and assert exactly — see `tests/test_ortho_view.py::_click_pane`.

Things worth driving by hand that no test covers yet:

* Draw a cuboid in XZ, press `t`, select it in the ROI Manager, check it appears in all
  three panes and that *Property* reports a sane extent and count.
* Draw a non-rectangular polyhedron in XY over a populated structure. Confirm XY stays
  exactly as drawn, XZ/YZ start as fitted hulls, and Add/Delete point works in both sides.
  Repeat in an empty area and confirm both sides use editable four-corner fallbacks.
* With **Show all** on and several ROIs, confirm no ROI is drawn twice in a side pane.
* Draw a flat rectangle in XY and confirm it is **dashed** in XZ/YZ and cannot be grabbed
  there.
* Select a region ROI in Render and confirm the same localization rows are highlighted
  over the XY, XZ and YZ images.
* With a cuboid, sphere, cylinder and convex projection hull visible, switch Scatter
  between Ortho and 3D. Confirm the
  lower-right wireframes and the translucent 3-D meshes follow Manager selection/Show-all;
  play each About-X/Y/Z rotation and compare the held axis with its label.

### 11.3 Re-measure the agreement

The 206-row agreement in §7 is a one-file probe: build a synthetic 3-D dataset, make one
cuboid, and compare `compute_roi_selection`, `roi_crop.compute_crop_mask` and
`channel_labels.roi_mask_for_record`. If those three ever disagree, the bug is in whichever
one stopped routing through `roi_volume_mask`.

### 11.4 Questions I would put to a verifier

* Is `pane_owns_record` the right rule, or should a flat ROI drawn in XY be editable from
  a side pane by projecting the edit back? I judged not — a projection cannot express a
  change on the collapsed axis — but the alternative is arguable.
* Is `EMPTY_SEED_FRACTION = 0.5` of the visible range a defensible default, given §9.3?
* Does `volume_silhouette` return the right thing for a **non-convex legacy multi-level**
  polyhedron? The radial union is exercised by tests, but not against a deliberately
  awkward shape. (A non-convex projection hull follows a different exact mask path.)
* `MIN_SEED_LOCS = 10` is a judgement, not a measurement. Is there a dataset where it is
  wrong?

---

## 12. Traps

1. **Never store a volume geometry as a 2-D shape plus a plane name** (§3.1). This is the
   one mistake that produces silently wrong data rather than an error.
2. **`roi_visible_in` gates on family + dataset, not plane.** Anything that draws or
   hit-tests in a side pane must narrow further — go through `_record_in_scope`, which is
   the single place both the draw loop and the hit-test consult.
3. **A draft is not in the store**, so `rois.changed` never fires for it. Anything that
   must react to a draft needs `on_roi_overlay_changed`.
4. **`OrthoPaneOwner.__getattr__` binds the owner's methods to the *window*.** If a
   delegated method reads `self.roi_view_plane()` or any other pane-specific answer, it
   must be overridden on the adapter and bound to it (see `normalize_roi_record`).
5. **`scale_z` is about z = 0**, not the ROI centre (§3.5).
6. **`_bounds` cannot measure a volume geometry** — it returns `(0, 0, 0, 0)`. Two
   separate defects came from this (§8.1, §8.4). Use `volume_bounds` or
   `_volume_view_bounds`.
7. **Do not add a stdlib-style "all-False" fallback for a volume type.** `roi_region_mask`
   raises deliberately: an empty selection is indistinguishable from a correct one.
8. `CLAUDE.md`, `AGENTS.md`, `BACKLOG.md` and everything under `docs/` are **local-only and
   gitignored**. Do not stage or commit them. No `Co-Authored-By:` trailers, on any
   machine, in any session, whatever the tool.

---

## 13. Repo state

Merge base `1009bf8`. Commits, oldest first:

```
16995e0  feat(viewer): orthogonal view mode, and one highlight rule for every view
72a1050  feat(channel): rework attribute separation around channels of two kinds
60bd445  refactor: retire Convolution (3D) and the old 3D ROI dialog
266574b  feat(roi): volume ROI geometry, membership and seeding
911321d  fix(scatter): make "showing everything" explicit so the side panes can take the mouse
ec650bb  fix(ortho): share Z between the side panes as a scale, not as a range
5ee6811  feat(roi): multi-point ROI — N markers as one entry
e7b9512  feat(roi): draw volume ROIs, and teach the consumers to read them
cf87ca3  feat(roi): 3-D tools open the orthogonal view, and ROIs show in all three panes
63a7e29  fix(roi): a drawn volume ROI was unselectable, and invisible in the side panes
0c3150e  fix(roi): a cuboid is a rectangle in every view, not a polygon
18d2a9e  feat(roi): draw in the side panes, multi-slice polyhedra, and a rotating 4th pane
a778603  fix(build): mark images binary so autocrlf cannot corrupt them
58aae2a  fix(roi): a volume ROI converted to a point landed at the origin
823e4f7  fix(roi): a volume ROI read as empty wherever _REGION_TYPES was the gate
90adc65  docs(ortho): bring the handoff up to the state that shipped
f5fa700  docs: MINFLUX tracking research report, 2021-2026
8e7f707  fix(roi): drawing in the XZ/YZ panes did nothing in the render view
```

`f5fa700` and earlier are pushed; `8e7f707` is local at the time of writing.

⚠ Several of these commits contain more than their message describes. The working tree
was shared with a parallel session working on the orthogonal view, and its changes were
swept into whichever commit ran next — `side_zoom_anchor` landed in `cf87ca3` and
`owner_chrome` in `18d2a9e`. Nothing was lost and the authorship is unaffected (every
commit in the repository is authored by the repository owner, with no attribution
trailers anywhere), but do not read those three messages as a complete inventory of their
diffs.
