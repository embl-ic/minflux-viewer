# Attribute Plot performance — briefing for an independent review

**Purpose.** Hand this to a fresh agent session so it can independently re-derive
(a) the original slowness diagnosis, (b) the "Thinning" artefact and whether its
rationale holds, (c) whether the implemented solution is sound, and (d) whether
the conclusion *"the Qt/CPU path cannot carry this"* is justified.
**Re-measure; do not trust the numbers below.**

## Environment and reference data

| | |
|---|---|
| Repo | `d:\Git\minflux-viewer` |
| Platform | Windows, PyQt6 6.11.0, pyqtgraph 0.14.0 |
| Interpreter | `.venv\Scripts\python.exe` (pytest available) |
| Reference dataset | `D:\Workspace\Users\Cigdem\2026\20260625\4_BD_MRED_75pM_MINFLUX_3D.mat` (also `.msr`) |
| Its size | 246,437 materialized localizations · 20,627,153 raw all-iteration rows · 10 iterations |

---

## 1. The original user-visible symptom (the artefact)

Plotting `X=idx` vs `Y=efo`, coloured by `C=vld`, zoomed to `idx ∈ [0, 3×10⁴]`:
**unchecking "Valid only" showed FEWER points than checking it**, and some points
marked valid appeared/disappeared as the checkbox was toggled. That is backwards —
the unchecked set is a strict superset.

### It is not a filtering bug

`core/loader.py::mfx_row_mask` builds `base = vld_col` when `vld_only` else
all-True, then intersects with the last iteration, so valid ⊂ all by construction.
Counts on the reference file:

| selection | rows |
|---|---|
| raw store | 20,627,153 |
| valid rows anywhere | 1,159,272 (5.62 %) |
| Valid only **on** → `ds.attr` | 246,437 |
| Valid only **off** → `itr == last`, any validity | 2,196,618 |

The 1,950,181 extra rows are **trace-termination records, not measurements**:
`vld=0`, `eot=1`, `sta ∈ 8..11`, `eco` exactly 0, `loc_x/y/z` + `efo` + `fbg` +
`lnc` all NaN; `dcr`/`tim`/`tid` finite. Across *all* iterations only 44,839 of
19,467,881 invalid rows carry a finite `efo`, and they sit only at itr 4 and itr 6.

### The actual cause: display thinning

`ui/attribute_window.py` had, since the repo's first commit (`648fa36`,
2026-05-11 — no rationale recorded, never measured):

```python
if n > _MAX_DISPLAY_POINTS:                 # 50_000
    step = ceil(n / _MAX_DISPLAY_POINTS)
    values = {dim: v[::step] for ...}
```

computed over the **whole selection**, before and independent of zoom:

| Iter | Valid only ✔ | Valid only ✘ |
|---|---|---|
| last (10th) | n = 246,437 → stride 5 | n = 2,196,618 → stride **44** |
| all [flatten] | n = 1,159,272 → stride 24 | n = 20,627,153 → stride **413** |

In the fixed window `idx ∈ [0, 3×10⁴]` that drew **287 vs 34** points (last) and
**269 vs 23** (flatten) — the visible density depended on the size of the part you
*cannot see*. It also changed *which* points survived (`::5` vs `::44`), so points
flickered on an unrelated toggle, and zooming in never revealed more. The status
line reported the **pre-thinning** `n`, which is why it stayed invisible.

The same defect had a second form in 3-D and on the first 2-D draw: **NaN rows
consumed the budget.** Unchecking Valid only in 3-D drew **82,162** points where
checking drew **246,437** — 88.8 % of that selection is NaN-coordinate probe rows
that paint nothing.

---

## 2. Profiling — where the time actually goes

Un-thinned draw, 1,159,272 points, C dimension active, 900×700 window:

| phase | time | share |
|---|---|---|
| fetch + align (`_series_data`, reads the raw store) | 0.13 s | 6 % |
| colour mapping (`_mapped_colors`) | 0.10 s | 5 % |
| everything else in `_draw` | 0.01 s | — |
| **build items (`ScatterPlotItem.setData`)** | **1.89 s** | **89 %** |
| first paint | 0.74 s | |

Self-time inside `setData` (cProfile): `_style` 0.65 s · `SymbolAtlas._keys`
0.49 s · `updateSpots` 0.29 s · `_mkBrush` 1,159,273 calls 0.21 s · `getId`
**2,318,544 calls** (2 per point, *even with a uniform brush*) · **`renderSymbol`
— the actual Qt rasterization — 256 calls = 0.006 s.**

⇒ ≈ **1 µs of Python per point**, with no flag to disable it: 246 k ≈ 0.2 s,
1.16 M ≈ 1.9 s, 20.6 M ≈ 19 s. **Qt is not the bottleneck.**

Second finding: pyqtgraph's symbol atlas keys on **object identity**, not colour
value (`getId` stamps `obj._id` per instance). 246,437 `QBrush` objects carrying
only 256 distinct colours built a **246,437-entry atlas** costing 7.18 s; sharing
one brush per LUT entry collapses it to 256 entries and 0.68 s (**10.7×**).

### Alternatives measured — all drawing every point, per-point colour, offscreen

| approach | 246,437 | 1,000,000 |
|---|---|---|
| pyqtgraph, brush per point (old) | 7.32 s | skipped |
| pyqtgraph, shared LUT brushes | 0.68 s | 2.83 s |
| pyqtgraph, grouped into 256 uniform items | 0.52 s | 2.18 s |
| `QPainter.drawPoints`, 256 colour groups | 0.26 s | ~0.30 s |
| `QPainter.drawPoints`, one colour | 0.12 s | 0.12 s |
| **OpenGL (`GLScatterPlotItem`)** | **0.08 s** | **0.13 s** |
| matplotlib 3.11 Agg `scatter` | 0.90 s | 3.47 s |

⚠ Trap for any colour-grouped CPU path: `np.argsort(bins, kind="stable")` on
20.6 M **int64** takes 15.0 s; casting `bins` to **uint8** (radix) takes 0.15 s — 100×.

---

## 3. Why the Qt/CPU path cannot carry this — the deduced bottlenecks

1. **pyqtgraph's `ScatterPlotItem` is O(N) in Python**, ≈1 µs/point, in `_style`,
   `SymbolAtlas._keys`, `updateSpots` and `_mkBrush`. No option switches it off;
   it runs even for a single uniform brush.
2. **Qt's rasterization is negligible** (256 `renderSymbol` calls = 6 ms), and a
   bare `QPainter.drawPoints` does 1.16 M points in 0.12 s — the ceiling is
   pyqtgraph's bookkeeping, not the toolkit.
3. **Per-point colour costs ≈16 µs/point inside `setData`** even with pre-built,
   shared brushes (4.06 s vs 0.24 s uniform at 246 k — a 17× penalty). Brush
   caching cannot remove it; it is inherent to per-spot style keys.
4. **The cost is paid on every rebuild**, so it multiplies with interaction.
5. **matplotlib is ~1.3× slower** than the fixed pyqtgraph path and puts its cost
   in the Agg paint, repeated on every view change, with no partial redraw and no
   GPU path. Not an escape route.
6. **A custom `QPainter` item is fast** (0.12–0.26 s at 1.16 M) but abandons
   pyqtgraph's item ecosystem, and still costs ~2.9–3.4 s at 20.6 M versus 2.07 s
   on the GPU with 0.03 s repaints.

⇒ Any CPU renderer must hide data at multi-million scale. Only the GPU makes
"draw everything" affordable.

---

## 4. What was implemented

All in `ui/attribute_window.py` unless noted.

**S1 — brush cache.** `_mapped_colors` builds ≤257 brushes (one per LUT entry plus
a transparent one) and indexes them. ⚠ `bins` is `uint8`, so `np.where(..., 256)`
wraps to 0 and paints missing values opaque — cast to `int32`.

**S2 — budgets follow the renderer, not a checkbox.** `_MAX_DISPLAY_POINTS`
1,000,000 (uniform colour, CPU) · `_MAX_COLOR_DISPLAY_POINTS` 50,000 (per-point
brushes, CPU) · `_MAX_GPU_DISPLAY_POINTS` 25,000,000 (GPU: a graphics-memory
guard, not thinning).

**S3 — thinning became view-aware.** `_thin_for_view` + `_visible_row_mask`
(+25 % margin) restrict to rows inside the current range, re-thinning on
`ViewBox.sigRangeChanged` with a 120 ms trailing debounce. `_drawable_row_mask`
drops rows that are non-finite in a plotted coordinate first, so NaN rows never
consume the budget. `_visible_row_mask` returns `None` while an axis still
auto-ranges — checking the placeholder range would draw nothing and then fit the
view to that remnant.

**S4 — honest read-out** (`_point_count_text`): `N points` · `D of N points
(visible range)` · `(1 in k of the visible range)` · `(1 in k of the M with a
finite value)`.

**S5 — one-entry pre-thinning cache** (`_series_cache`), used only by the
pan/zoom re-thin, so a pan does not re-read 20 M rows.

**S6 — GPU renderer, now the default.** A `GLViewWidget` canvas placed **behind**
the pyqtgraph plot (which is made transparent), **not replacing it**. Consequence:
axes, grid, zoom modes, Reset View, Lines, legend, colorbar and ROI
drawing/selection all keep working unchanged; only the markers move.

- Positions uploaded **once** (centred and divided by span so float32 keeps
  precision at `idx ≈ 2×10⁷`); each pan/zoom only sets a per-axis affine.
- The canvas geometry is driven to the ViewBox rect and is deliberately **not** in
  a layout: pyqtgraph 0.14's `GLViewWidget.getViewport()` is hard-coded to the
  full widget, so the widget itself must *be* the viewport, or points spill over
  the axis strips.
- `_add_series(markers=False)` gives the scatter **no data**. Handing pyqtgraph
  the points and merely hiding the item still pays ≈1 µs/point (1.77 s vs 0.79 s
  at 1.16 M).
- `_DataBoundsItem`: ⚠ `ViewBox.childrenBounds` **skips invisible items**, so with
  an empty hidden scatter the view had *no data bounds at all*. Auto-range then
  latched onto the zoom rubber band (range jumped at drag start, collapsed during
  the drag, overrode the applied zoom) and Reset View / the `A` button could never
  find the data again. ⚠ A `PlotDataItem` carrying the corner points does **not**
  fix it — with no pen and no symbol it hides both of its children.
- Three fallbacks, each switching GPU off, forcing thinning on, logging at WARN
  and stating the reason in the status line: (i) `pyqtgraph.opengl` not
  importable; (ii) the widget never gets a context (`isValid()` False);
  (iii) a context that reports itself valid but paints nothing — reproduced
  exactly with `QT_OPENGL=software`, detected by grabbing the framebuffer once and
  testing whether every pixel is identical, gated on `QOpenGLWidget.frameSwapped`
  plus 150 ms with one retry at 400 ms. (Checking at `singleShot(0)` reports an
  unpainted widget as broken and disables a working GPU.)

**S7 — deployment.** The GPU path adds no new dependency (the same stack the 3-D
views already use), and the shipped one-folder build already contains
`pyqtgraph.opengl`, `OpenGL.*`, `PyQt6/QtOpenGL(.Widgets).pyd`, the freeglut DLLs
and Qt's software rasterizer `opengl32sw.dll` — verified inside
`dist/MINFLUX-Viewer-0.4.1-win.zip`.

---

## 5. Current status

Reference file, `C=vld`, thinning off, **first draw / repaint**:

| points | CPU (pyqtgraph) | GPU |
|---|---|---|
| 246,437 | 0.66 s / 0.10 s | 0.64 s / **0.02 s** |
| 1,159,272 | 2.40 s / 0.21 s | **0.31 s / 0.03 s** |
| 20,627,153 | ~19 s (not attempted) | **2.07 s / 0.03 s** |

The reported case now draws **1,433 in-window rows with Valid only either way**
(was 287 vs 34), and unchecking correctly *adds* rows in flatten (6,421 → 6,522).

Marker **symbols** are still circles on the GPU: `GLScatterPlotItem`'s fragment
shader hard-codes a disc (`discard` outside `dot(xy, xy) <= 1.0`). Line styles do
work there, because the Lines curve is still a pyqtgraph item.

**Tests:** `tests/test_attribute_multidim.py` (34 passed) covers thinning
zoom-awareness, the drawable-row rule, the read-out wording, GPU-behind-plot, the
zoom/Reset regression and both GPU fallbacks. ⚠ Running the **whole** pytest suite
in one process crashes in pyqtgraph GC, and a few Qt-heavy files exit 127 — both
reproduce on a clean checkout, so run per-file.

---

## 6. What I would like challenged

- Is ≈1 µs/point really irreducible in pyqtgraph 0.14, or is there a supported
  fast path I missed (`setData` kwargs, `useCache`, a different item class)?
- Is the 1 M / 50 k / 25 M budget split defensible, or should the CPU path simply
  refuse above a threshold and say so?
- Is "GPU canvas behind a transparent plot" robust across drivers and remote
  sessions, or should the fallback be more aggressive?
- Is the blank-canvas detection (one framebuffer grab, uniform-colour test,
  `frameSwapped` + 150 ms, one retry) safe against false positives on slow GPUs?
- Does the drawable-row rule ever hide something a user wanted — rows that are
  non-finite in a plotted axis but meaningful elsewhere?

---

## 7. Independent review and CPU implementation (2026-08-23)

### Conclusion

The original diagnosis is half right: pyqtgraph's ordinary `ScatterPlotItem`
bookkeeping is the slow CPU path, while Qt bulk raster painting itself is fast.
The conclusion that only a GPU can solve the product problem is too strong.
OpenGL is the fastest option when the requirement is to animate every literal
marker, but an attribute plot has only `viewport width × height` independent
display cells. At high density, complete screen-space aggregation is both more
informative (it preserves count/density) and much cheaper than drawing millions
of mutually occluded markers.

This is also the approach documented by Datashader: map every point to a
pixel-sized aggregate and apply an explicit reduction such as count or mean
([Datashader pipeline](https://datashader.org/getting_started/Pipeline.html)).
Pyqtgraph itself recommends clipping and pixel-aware downsampling for dense
plots, although its automatic downsampling assumes ordered/uniform X and is not
a general scatter-density solution
([PlotDataItem performance options](https://pyqtgraph.readthedocs.io/en/latest/api_reference/graphicsItems/plotdataitem.html)).
Qt exposes array/bulk primitives such as `QPainter.drawPoints`, and its raster
backend is a supported high-performance engine
([QPainter documentation](https://doc.qt.io/qt-6/qpainter.html)).

Therefore the implemented policy is:

- keep the existing GPU window for literal all-marker rendering and 3-D;
- add a separate, guaranteed non-OpenGL **View ▸ Attribute Plot (CPU fix)**;
- bulk-paint exact markers while sparse;
- when overplotted, chunk **every visible drawable row** into a display-sized
  count grid, plus a separately named mean-C grid when C is active;
- recompute after settled pan/zoom; never fixed-stride sample the CPU-fix data.

### Implemented changes

- `ui/attribute_cpu.py`: `BulkScatterItem`, chunked screen count/mean reduction,
  joint-finite extents, and deterministic spatial representative selection.
- `ui/attribute_window.py`: independent CPU mode/state, sparse bulk markers,
  dense `ImageItem` aggregation, screen-aware line LOD, and removal of fixed-k
  sampling from the legacy/GPU memory guard.
- `ui/main_window.py`: separate dataset-owned CPU action/window integrated with
  lifecycle, colours, ROI overlays, LUT routing, dataset deletion and shutdown.
- `ui/gpu_capabilities.py` + `__main__.py`: one GUI-thread startup capability
  probe using `QOffscreenSurface`/`QOpenGLContext`; Qt explicitly says to check
  `create()`/`isValid()` ([QOpenGLContext](https://doc.qt.io/qt-6/qopenglcontext.html)).
  The fixed 25 M cap is gone. The point limit uses half reported free VRAM and
  one eighth available system RAM, with a conservative shared/unknown-GPU path.
- The startup probe's cleanup destroys the context before its surface. The
  reverse order reproduced a Windows heap-error exit and was corrected.

### Verification

900×700 screen reduction, all iterations, invalid rows included unless noted:

| corpus/case | input rows | occupied cells (`idx/efo`) | CPU reduction |
|---|---:|---:|---:|
| Cigdem reference MAT | 20,627,153 | 106,717 | 0.40–0.61 s |
| Cigdem reference MSR | 20,627,153 | 106,717 | 0.47 s |
| largest sample MAT | 4,278,814 | 111,958 | 0.31–0.44 s |
| sample ratiometric MSR | 6,558,076 | 126,898 | 0.33–0.36 s |
| synthetic, 95% memory edge | 123,617,067 | 3,079 | 4.62 s |

The memory-derived edge on the test machine was about 130.1 M points; the 95%
synthetic input used two float32 arrays (943 MiB) and peaked at about 1.0 GiB
RSS. On the exact reported zoom (`idx=0..30000`, `efo`, all iterations), the
reference MAT produced 6,556 visible all rows versus 6,443 valid rows and
**zero screen cells where valid exceeded all**; count + mean-C took 0.48 s.
Thus the inverted-density
artefact is structurally impossible in the CPU-fix renderer.

Corpus coverage:

- all 5 MAT and all 18 MSR files under `D:\Workspace\Users\Cigdem\2026`;
- all 64 MAT/MSR/NPY candidates under the sample-data tree: 29 MINFLUX datasets
  reduced successfully; 35 legacy/image-only MSR containers were correctly
  classified as containing no MINFLUX dataset;
- focused Qt/unit suite: CPU renderer, multidimensional Attribute Plot and
  command finder/menu integration.

Raw benchmark JSON is under `output/attribute_cpu_*`; the repeatable harness is
`scripts/benchmark_attribute_cpu.py`.
