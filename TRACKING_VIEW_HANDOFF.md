# Tracking View — handoff

`View › Tracking View` (**Ctrl+T**), directly under Render View. A third viewer
that plays trajectories as comets over a de-emphasised structure. Render and Loc
Scatter Plot are **unchanged** — the earlier attempt to give them a shared time
row was reverted in full (see *What was reverted*).

## The picture

Modelled on the MATLAB NPC-trafficking workflow this is meant to serve
(`npctat2021/MINFLUX_NPC_Tracking`, Nature 2025), whose own UI draws the merged
NPC as a density backdrop with cargo trajectories as connected lines and *"a
playable star head in magenta colour that mimics the displacement of the cargo
over time"*. Here:

- the **structure** is a grey, sub-dominant backdrop — a render raster or faint
  points, at three brightness levels;
- each **trajectory** is a polyline with a fading tail;
- the **head** is a magenta star at the leading edge of each live track;
- the **time row** (the same widget the depth row's slider comes from) scrubs and
  plays; its window *is* the comet — the end is the head, the width is the tail.

## Why a separate viewer

It answers *when*, and for that the structure has to step back to a context
layer — which is not something to do to a view someone opened to look at
localizations. It also keeps the one-window-per-dataset invariant intact:
`_tracking_windows` is a peer of `_render_windows`, not a mode inside it.

## Time is zeroed to each trace by default

Measured on the reference file
(`1_sample_A_1-100_seqTrk-3D-Seb_Octahedron_100_pho_exc_10.mat`): **230 traces,
45,105 localizations, 0.9 ms between localizations, 12.7 nm median step, 104 nm
median end-to-end** — and each trace lasts a median of **60 ms inside a 1,446 s
acquisition**. Played on absolute time you would watch an empty field for
minutes and almost never see two traces at once. Zeroed, all 230 walk together
and the comparison is the point; the axis then spans the longest *trace*
(2.34 s) rather than the run (24 min). Absolute time stays on the mode combo for
the question it does answer.

Trace-relative time is computed by `core/tracking_time.py::time_axis_values`, the
same function the axis itself uses, so there is one definition of it.

## Structure and tracking channels

`core/tracks.py::looks_like_tracking` separates them by how far a trace travels
end to end, and **returns the reason, not just the verdict**:

| | verdict | reason |
|---|---|---|
| reference tracking file | Tracking | *traces move 104 nm end to end over 77 localizations* |
| `.msr` sample | Tracking | *traces move 146 nm end to end over 152 localizations* |
| simulated NPC scaffold | Structure | *traces move 8.7 nm end to end, within localization precision of standing still* |
| simulated cargo | Tracking | *traces move 177 nm end to end over 140 localizations* |

The threshold is 25 nm and ≥5 localizations per trace — well above any
localization precision and well below a real excursion, so the two populations
are nowhere near it. The role is a **combo on each channel row** with that reason
as its tooltip: a default that cannot be corrected is a guess in disguise.

With no structure channel, the backdrop is the tracking channel's own
localizations — the everywhere-it-has-been context a single-channel run has
instead of a scaffold.

## Performance — measured, not assumed

The one thing that has to be fast is *which localizations are in the tail right
now*, answered many times a second. `core/tracks.py` answers it with **no Python
loop over traces**: rows are sorted once by `(trace, t)`, a monotone
`key = trace_code * scale + t` is built over that order, and a **single**
`np.searchsorted` pair resolves the window's span *in every trace at once*; the
rows are then gathered by a vectorised ragged-range expansion. A frame is
`O(T log N)` to locate plus `O(k)` to gather — never `O(N)`.

Windowing (6 bands per frame):

| | traces | points | ms/frame |
|---|---|---|---|
| reference file | 230 | 45,105 | **0.20** |
| synthetic | 2,000 | 1,000,000 | 1.67 |
| synthetic | 2,000 | 10,000,000 | 5.39 |
| synthetic | 10,000 | 5,000,000 | 16.58 |

It scales with **trace count and drawn points**, not with total points — 10 M
points in 2,000 traces costs a third of 5 M points in 10,000 traces.

Drawing, in pyqtgraph:

| tail points | heads | ms/frame |
|---|---|---|
| 5,000 | 230 | 3.3 (300 fps) |
| 40,000 | 2,000 | 8.3 (121 fps) |
| 200,000 | 10,000 | 33.1 (30 fps) |
| 500,000 | 10,000 | 63.8 (16 fps) |

⚠ **The tail is drawn as a handful of constant-opacity polylines, not with
per-point alpha.** Measured at the same 40,000 points: **8.3 ms against
89.7 ms**, an 11× difference. pyqtgraph pays a per-point cost for a varying
brush (the same effect the Attribute Plot notes), while a curve with a `connect`
array is one painter path. So the comet fades in `DEFAULT_TAIL_BANDS = 6` age
bands, each one polyline at its own opacity.

⚠ **The backdrop is cached, and that was most of the frame time.** It does not
depend on time, but it was being re-rastered every frame: **49.7 ms/frame before,
5.1 ms/frame after** on the reference file. `_backdrop_key` lists what it *does*
depend on (mode, brightness, projection, background, bounds, channel roles and
visibility, track identity) and time is deliberately not in it.

End to end, playback:

| | ms/frame | fps |
|---|---|---|
| reference `.mat`, 230 tracks | 5.1 | 194 |
| reference `.msr`, 122 tracks | 4.4 | 225 |
| simulated scaffold + cargo | 3.0 | 336 |
| simulated shells, 80 tracks | 3.3 | 307 |

## Two readability decisions that came from looking at it

- **The head is 7 px, not 11.** A MINFLUX track is 100–600 nm end to end inside a
  field of ~10 µm, so at full-field zoom a whole track is a few pixels and an
  11 px star covered every one of them. *Zoom to live tracks* (right-click) fits
  the view to what is currently drawn, which is the real answer to wanting more.
- **The backdrop raster is normalised with a square root.** A structure render is
  sparse — most non-empty pixels hold one or two localizations while the 99th
  percentile is several — so a linear ramp left the scaffold at under a fifth of
  the available opacity and it disappeared. *Structure brightness* offers
  Faint / Normal / Strong.

## ⚠ A bug this found in the colour handling

`overlay_color_cycle` encodes channel colours as `solid:custom:#rrggbbaa`, and
**`solid_color_rgb` does not understand that form**: `is_solid_color` returns
False for it and the accessor hands back a grey **without raising**, so a caller
that guards with `try` still gets grey and never learns why. That is why the
first comet tails came out grey instead of the channel colour.

`core/overlay.py::channel_rgb` is the resolver that handles all three spellings
(bare name, `solid:<Name>`, encoded literal). Two other places decode this by
hand — `render_window._render_solid_rgba` (correct) and
`attribute_separation_dialog._rgb_for_lut` (correct) — and
**`volume_window.lut_rgb` has the same defect**: given an encoded value it falls
through to `representative_rgb`, prints *"Unknown volume colormap"* and returns
white, so a multi-channel volume colours every channel white. Not touched here;
worth a one-line fix on its own.

## Simulated tracking data

Two new entries in the Data Simulator and in *File › Open Sample Data*, with step
size and sampling taken from the reference acquisition so a simulated run plays
at the same rate as a real one:

- **Tracking on shells 3D** (`tracking_shells`, kind `tracking`) — molecules
  diffusing on the surface of spherical shells, the geometry of the reference
  file. Each step is taken in the tangent plane and projected back onto the
  shell, so the walk stays on the surface. One dataset, with `tim`.
- **Tracking: NPC scaffold + cargo** (`npc_tracking_2ch`, kind `track_overlay`) —
  the two-colour trafficking design: an 8-fold double-ring scaffold channel that
  does not move, and a cargo channel translocating along the pore axis under
  lateral confinement, a settable fraction of which aborts and returns. Arrives
  as one overlay, so the Tracking View opens with a structure and a tracking
  channel already grouped.

Both start each trace at a random time in the acquisition, which is what makes
absolute-time playback mostly empty and is the honest reason the default is
trace-relative.

## Files

New: `core/tracks.py`, `ui/tracking_window.py`, `tests/test_tracks.py`,
`tests/test_tracking_view.py`.
Changed: `core/simulate.py` (two generators), `core/sample_presets.py`,
`core/overlay.py` (`channel_rgb`), `core/app_state.py` (`Ctrl+T`),
`ui/main_window.py` (action, menu, registry, two sim builders),
`ui/command_meta.py`, `ui/preferences_dialog.py` (shortcut label),
`ui/time_slider.py` (`set_window`).
Kept from the earlier work: `core/tracking_time.py` and `ui/time_slider.py` — the
time axis and the playback row are exactly what this window needed.

## What was reverted

The time row added to Render and Loc Scatter Plot, and everything it touched:
`render_window.py`, `scatter_window.py`, `tile_cache.py` (`TileKey` is back to 8
fields), `render_scheduler.py`, `precision_render.py`,
`precision_render_window.py`. Verified by diffing each against `HEAD` for any
mention of the time gate — the only matches left are the pre-existing
`setMouseTracking` / `setKeyboardTracking` calls.

## Subsequent milestones completed

1. **Time-coloured tails:** selectable Channel/Time style; time style runs
   oldest-to-newest while star heads retain the channel LUT.
2. **Track selection and hover:** clicking the nearest visible comet isolates
   its actual `tid`; hover reports count, duration, net displacement and median
   speed. Selection survives redraws and is invalidated honestly if filtering
   removes the trace.
3. **3-D mode:** lazy PyOpenGL trajectory/backdrop rendering uses the same
   gap-safe windows and band colours. Missing PyOpenGL is an explicit message.
4. **Movie export:** deterministic event-loop capture writes an RGB TIFF stack
   plus JSON timing/projection sidecar through a partial path, so cancellation or
   window close never presents an incomplete target as a completed movie.
5. **Trace Viewer synchronization:** selected `tid` and trace-relative time are
   shared bidirectionally through `AppState.tracking_playhead_changed`; source
   tokens prevent feedback loops.
6. **Off-thread indexing:** coordinate/time preparation and `build_track_set`
   run in a process-owned serial pool, appear in Task Monitor, and use
   cancellation, generation and dataset-identity guards.
7. **Analysis:** the parentless modeless MSD workbench and advanced pure methods
   are described in `TRACKING_PLAN.md`.

Remaining work is intentionally method/experiment-specific: concrete state
inference adapters, a motor stepping/dwell plugin, trace merge/split repair with
validated priors, and dynamic-error correction when acquisition-loop metadata
is available.

## Verification

```text
.venv\Scripts\python.exe -m pytest tests/test_tracks.py tests/test_tracking_view.py -q
28 passed

.venv\Scripts\python.exe -m pytest tests/ -q
3104 passed, 7 skipped   (after updating the two lists this change adds to:
                          tests/test_sample_presets.py pins the preset names and
                          tests/test_window_stacking.py hand-builds the window
                          registries, so a new viewer has to be added to both)
```

⚠ `tests/test_ortho_view.py` and `tests/test_render_ortho_floating.py` contain
geometry assertions that fail intermittently under full-suite load on this
machine and pass in isolation; the failure set moved across six runs during this
work. That is the behaviour `CLAUDE.md` already records for this suite and is
unrelated to the Tracking View.
