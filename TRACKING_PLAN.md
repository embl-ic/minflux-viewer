# MINFLUX tracking in this viewer — plan

**Status (2026-09-27):** the generic tracking foundation, Tracking View, and
modeless MSD workbench are implemented and tested in the working tree. The
foundation includes discontinuity-safe 2-D/3-D playback, persistent channel
roles, selectable channel/time tail colour, click selection and hover summaries,
Trace Viewer playhead synchronization, deterministic TIFF movie export, the
pure tracking input/result/method contracts, kinematics, irregular-time MSD,
empirical jumps, fixed-size Brownian jump mixtures, the Simson confinement
index, a formal finite-time ergodicity-breaking statistic, and an optional
state-adapter registry. The generic Particle Tracking placeholder was retired:
MINFLUX `tid` remains authoritative and experiment-specific merge/split repair
belongs in a plugin. A motor stepping/dwell workbench and concrete external
state-inference plugins remain separate future projects, not unfinished parts of
the generic core.
This work is included in the v0.4.4 release.

---

## 0. ⚠ Git authorship and contribution — mandatory before every commit

**Ziqiang Huang is the sole developer, author, and contributor to this
repository.** The only permitted author/committer email identities are:

- `ziqiang.huang@embl.de`
- `hzq.fox@gmail.com`

No coding agent, AI system, automation, tool vendor, other person, or
organization may be named or implied as an author, co-author, committer, or
contributor. In particular, **never add `Co-Authored-By`, `Signed-off-by`, or
similar attribution for a tool or agent**, whether in commit trailers, release
notes, generated documentation, or source comments.

**A commit is never implicitly authorized. Before each individual git commit,
stop and obtain Ziqiang Huang's explicit confirmation.** At that point also
confirm which of the two permitted email addresses should be used. Do not alter
global git configuration; if an identity must be supplied, scope it to that
confirmed commit. A request to edit, test, finish, or prepare a change is not a
request to commit it.

- If a harness appends attribution **by default**, disable it before committing.
- The repository owner performs all pushes. Never push on the owner's behalf.
- **Check before pushing**, every time — the cure after a push is a history
  rewrite:

  ```bash
  git log --format='%H %s' <range> | while read -r sha rest; do
    git log -1 --format='%B' "$sha" | grep -qi 'co-authored' && echo "TRAILER: $sha $rest"
  done
  ```

  Silence means clean. Verified clean over the last 5 commits as of this plan.
- Found and **not yet pushed** → `git commit --amend` or reword in a rebase.
  Found and **already pushed** → the surgical repair only: rebuild that one
  commit with `git commit-tree` reusing its **tree, parent, author and committer
  dates**, rebuild any merge above it the same way, move the refs, and confirm
  `git diff <old-head> <new-head>` is **empty** before force-pushing. That is
  what was done for `9617e31` → `06d3603`, and it cost every other clone a
  `git fetch && git reset --hard origin/main`.
Also standing: `CLAUDE.md` and `AGENTS.md` are local, gitignored, and must not be
staged or committed.

---

## 1. What already exists (verify each before building on it)

### 1.1 Shipped and working

| | where | what it does |
|---|---|---|
| **Trace Viewer** | `plugins/trace_viewer/` (836 lines) | Per-trace inspection, **one trace at a time**. Table of every trace (N loc, efo, dcr, dt, velocity, bbox size), four linked time-series (dt, RMS velocity, eco, efo), a **play button + time slider** scrubbing a window across all four, and an animated **star head** (averaged over ±head points) with a **tail** over the last *N* points — already using `symbol="star"` and the colour-registry entries `Trace Viewer ▸ Head marker` / `Tail line`. Tile View, Overlay, tail removal, export. Reached from Plugins, not from a menu. |
| **Tracking View** | `ui/tracking_window.py`, `core/tracks.py` | `View › Tracking View`, **Ctrl+T**. All traces at once as comets over a de-emphasised structure. See *2*. |
| **Derived per-localization attributes** | `core/attributes.py` | `dst` (step distance), `dt` (interval from the previous localization in the trace), `spd` (`dst/dt`), `tim_trace` (time zeroed at each track start), and the trace-wise `siz` / `dur` / `len`. All already computed at load, filterable and plottable. |
| **Localization precision** | `analysis/localization_precision.py` | StdDev-per-trace (Ostersehlt 2022, ≥5 locs), MINFLUX targeted-donut CRLB (Balzarotti 2017; Marin & Ries 2024) with per-dimension photon counts, FRC. Plus the σ_fl excess-precision budget, `STD² = σ_fl² + σ_CRB²`. |
| **Z scaling factor from trace anisotropy** | `analysis/trace_analysis.py` | The ≈0.67 axial correction the NPC workflow also applies. |

### 1.2 Tracking analysis menu — resolved

`Analyze › Tracking › MSD Analysis` now opens the real modeless workbench.
`actionParticleTracking` is removed from that menu and hidden. MINFLUX already
delivers `tid`, so camera-SMLM-style linking is not a generic viewer operation.
Merging traces split by a blinking gap or splitting an apparent molecule jump
is trace repair, whose distance/time/state priors depend on the experiment; it
must arrive as a validated method/plugin rather than an attractive generic
button that silently rewrites trace identity.

`Analyze › Trace` holds two *real* actions: Estimate Average Trace Size, Estimate
Z Scaling Factor.

### 1.3 Backlog entry this plan replaces

`BACKLOG.md` ▸ Essential, one line:

> - Implement tracking-related functions — linking localizations into traces,
>   diffusion analysis, MSD, confinement detection

and, two lines below it, the movie-player entry, which the Tracking View has now
substantially delivered:

> - Make an advanced playable viewer (movie player) — beyond the trace viewer's
>   play button + time slider …: a simulated / reconstructed playable
>   **trajectory in 2D and 3D**, plus a playable vertical time indicator riding
>   along the attribute-vs-time plots, kept in sync with the trajectory playback.

**Done in this working tree:** both were replaced by the structured tracking
entry summarized in *§8*.

---

## 2. The Tracking View — motivation and current implementation

### 2.1 Why it exists, and why it is its own window

A first attempt added a shared **time row** to Render and Loc Scatter Plot, as a
sibling of the depth slider. It worked and was fast, and it was **rejected on
review**: gating the established viewers turns a view someone opened to look at
localizations into something else, and it broke the reading of those windows.
That work was **fully reverted** (see *§7*).

The Tracking View is a third viewer because it answers a different question —
*when* — and for that the structure has to step back to a context layer. It
keeps the one-window-per-dataset invariant: `_tracking_windows` is a peer of
`_render_windows`.

It is modelled on the MATLAB workflow of the two-colour NPC-trafficking study
this repository's owner co-authored
(`npctat2021/MINFLUX_NPC_Tracking`; Nature 2025, `s41586-025-08738-0`), whose
`NPC_trafficking_visualizationUI.m` draws the merged NPC as a density backdrop
with cargo trajectories as connected lines and *"a playable star head in magenta
colour that mimics the displacement of the cargo over time"*.

### 2.2 What it does now

- **Comets.** Each live trace is a polyline with a tail fading over
  `DEFAULT_TAIL_BANDS = 6` age bands and a star head at its leading edge. Heads
  and tails use the channel's overlay LUT colour, so a multi-channel view keeps
  channel identity.
- **Backdrop.** The structure channel as a grey render raster or faint points, at
  three brightness levels; with no structure channel, the tracking channel's own
  localizations serve as the where-it-has-been context.
- **Time zeroed per trace by default.** On the reference file 230 traces of ~60 ms
  sit inside a 1,446 s run, so absolute time is an empty field; zeroed, the axis
  spans the longest trace (2.34 s) and all 230 walk together. Absolute time
  remains selectable.
- **Roles.** `core/tracks.py::looks_like_tracking` separates a structure channel
  from a tracking channel by median end-to-end trace displacement (threshold
  25 nm, ≥5 locs/trace) and **returns the reason**, which becomes the tooltip on
  a per-channel role combo the user can override. Manual roles persist as
  dataset state and survive filter, calibration and preference rebuilds.
- **Projections** XY / XZ / YZ, *Zoom to live tracks*, head size, black
  background.
- **Responsive indexing.** Coordinate calibration/overlay transformation,
  time-axis preparation, `build_track_set`, interval diagnostics and automatic
  role classification run through a process-owned serial background pool and
  appear in the Task Monitor. Every rebuild has a generation; filter, mode or
  preference changes cancel older work, and stale generations/source identities
  cannot publish. Closing the window requests cancellation, disconnects the UI
  and never waits for the worker.

### 2.3 Measured behaviour (do not re-derive; re-measure only if the code changes)

Windowing is `O(T log N) + O(k)` per frame — one `searchsorted` pair over a
monotone `key = trace_code * scale + t` resolves the tail span in every trace at
once, with no Python loop over traces.

| | traces | points | window ms/frame |
|---|---|---|---|
| reference `.mat` | 230 | 45,105 | **0.20** |
| synthetic | 2,000 | 1,000,000 | 1.67 |
| synthetic | 2,000 | 10,000,000 | 5.39 |
| synthetic | 10,000 | 5,000,000 | 16.58 |

Drawing, and two decisions it forced:

- 6 constant-opacity polyline bands vs per-point alpha at 40,000 points:
  **8.3 ms vs 89.7 ms** (11×). pyqtgraph pays per point for a varying brush.
- Backdrop caching: **49.7 → 5.1 ms/frame**. It does not depend on time.

Playback end to end: `.mat` 194 fps · `.msr` 225 fps · simulated two-channel
336 fps · simulated shells 307 fps.

### 2.4 Simulated tracking data added for it

`Tracking on shells 3D` (pure tracking, tangent-plane walk projected back onto a
shell) and `Tracking: NPC scaffold + cargo` (static 8-fold double ring + axial
translocation with a settable abort fraction, delivered as an overlay). Step size
and sampling are the reference file's measured medians (0.9 ms, 12 nm).

### 2.5 Encoded custom LUT defect — fixed in the current working tree

`volume_window.lut_rgb` now resolves solid/custom LUT encodings through the
central colour helpers. The Tracking View resolves channel colours with
`core/overlay.py::channel_rgb`. Regression coverage is in
`tests/test_volume_multichannel.py` and `tests/test_tracking_view.py`.

---

## 3. Localization precision for a tracking experiment

This is a real gap and deserves its own workstream. The precision estimators in
the viewer were all written for **static** emitters, and none of their
assumptions survive a moving one.

### 3.1 Why the existing three do not transfer

- **StdDev per trace** measures the scatter of a trace about its own mean. For a
  static emitter that is the localization precision; **for a moving one it is
  dominated by the motion** and reports the excursion, not the precision. Used
  unchanged on a tracking dataset it will silently overstate σ by an order of
  magnitude.
- **CRLB** (`crlb_precision`) is the right *form* — it is photon-limited and
  per-localization — but the MINFLUX tracking sequence differs from the imaging
  one: fewer photons per localization, a different `L`, and the beam is actively
  following the molecule. The per-dimension photon logic
  (`mfx_sequence.photon_iterations_for_dataset`) needs re-checking against a
  tracking sequence's `Itr` table rather than assumed from the imaging case.
- **FRC** needs two independent reconstructions of *the same static structure*.
  A trajectory is not that. FRC should be **refused, with a reason**, on a
  channel whose role is Tracking.

### 3.2 What to implement instead

1. **Motion-corrected static precision.** Estimate σ from the residual about a
   *local* fit rather than about the trace mean — a sliding-window polynomial or
   a short-lag MSD intercept. The standard SPT result is that the MSD intercept
   at zero lag is `2σ² − (4/3)·D·Δt` in 1-D (Michalet 2010), so σ and D come out
   of the same fit and neither contaminates the other. **This is the single most
   useful thing to add** and it is also the entry point to MSD (*§5.2*).
2. **Report the localization-vs-motion split explicitly**, in the style of the
   existing σ_fl budget: how much of the observed scatter is photon-limited
   precision, how much is motion during the sampling interval, and how much is
   drift.
3. **Dynamic/localization error from the tracking loop itself.** MINFLUX tracking
   has a lag between the molecule moving and the beam following; at high speed
   this biases the reported position. Whether the viewer can estimate it from the
   data alone is an open question for the literature search (*§6*).
4. **Gate the existing estimators by role.** StdDev and FRC should say why they
   are inappropriate on a tracking channel rather than returning a number.

---

## 4. Proposal: enrich `Preferences › Tracking`

The page now contains timestamp precision plus the following two implemented
sub-sections.

### 4.1 Sub-section A — "Computed on tracking" (what to derive)

Attributes computed **when a dataset is opened in the Tracking View, or when
tracking mode is invoked**, in the manner of the existing on-load Compute block
(`compute_local_density`, `compute_loc_precision`) — a checkbox plus its
parameters, each with hover help, each honest about its cost.

Candidates, roughly in order of value:

| attribute | per | what it is |
|---|---|---|
| `msd_d` | trace | Diffusion coefficient from a short-lag MSD fit |
| `msd_alpha` | trace | Anomalous exponent α (α<1 sub-diffusive/confined, ≈1 Brownian, >1 directed) |
| `loc_prec_track` | loc | Motion-corrected localization precision (*§3.2*) |
| `conf_radius` | trace | Confinement radius, where α indicates confinement |
| `step_angle` | loc | Turning angle between consecutive steps — the directional-persistence signature |
| `jump_dist` | loc | Already `dst`; keep, and document it as the jump-distance histogram input |
| `inst_speed` | loc | Already `spd`; ditto |
| `state_label` | loc | Motion-state segmentation (immobile / diffusive / directed) — the largest item, and the one most in need of a literature decision first |
| `track_straightness` | trace | End-to-end distance over contour length (`dst` sum) — cheap, and a good first-pass motion classifier |

**Design constraints, all following existing project rules:**

- These are **derived**, so they belong in `dataset.derived` / `attr` with
  `user_visible` metadata and an entry in `prop.attr_names`, exactly as
  `confocal_mapping` does — writing only to `derived` leaves them invisible in
  every dropdown.
- Every one must go into `core/attributes.py`'s description table so the shared
  hover help covers them, and into `Preferences › Attributes`' computed list.
- Anything whose cost scales badly must run through `ui/background_tasks.py` and
  appear in the Task Monitor, not on the GUI thread.
- ⚠ **A derived attribute is a claim.** Each needs a stated fit range, a minimum
  trace length, and a documented failure mode — an MSD `D` from 6 points is
  noise, and it must not be delivered as a number without that caveat. Follow the
  `looks_like_tracking` pattern: return the reason as well as the value.

### 4.2 Sub-section B — "Tracking View" (UI configuration)

Move the view's hard-coded constants into preferences, keeping the current values
as defaults:

- Default tail length (currently 4% of the axis span) and whether it opens in
  sliding or **grow from start** mode
- Tail band count (6), tail width (2.0), head/tip opacity (255/70)
- Head symbol and size (7 px). Head/tail colour is **not** a separate Tracking
  preference: both now use the dataset/channel overlay LUT, the existing source
  of channel identity. The Trace Viewer's `plugins > Trace Viewer` annotation
  colours remain appropriate for its single-trace, attribute-annotation UI and
  are not reused as channel colours.
- Backdrop default mode (Render / Scatter / None) and brightness (Faint / Normal
  / Strong)
- Default axis mode (`trace` / `absolute` / `index`) and default projection
- Playback rate default, loop on/off
- The role threshold: `TRACKING_DISPLACEMENT_NM` (25 nm) and
  `TRACKING_MIN_MEDIAN_LOCS` (5) — exposed because a user with an unusual
  experiment will need to move them, and hiding them would make a wrong
  auto-role uncorrectable at the source

**Both sub-sections must follow the modeless-Preferences rule**: the dialog
merges only changed leaves (`core/app_state.py::merge_pref_changes`), so add
defaults to `DEFAULT_PREFS` **and** the corresponding key to `_MIGRATION_KEYS` if
any migration is written — otherwise a fresh install runs it.

---

## 5. Suggested next steps, in order

### 5.1 Small, do first

1. **Done:** fix `volume_window.lut_rgb` (*§2.5*).
2. **Done, with a deliberate design change:** comet heads and tails use each
   channel's central overlay LUT colour. A multi-channel tracking view therefore
   retains channel identity instead of using one hard-coded magenta head or an
   average tail colour. The Trace Viewer keeps its own single-trace annotation
   colours; these are not the same semantic fact.
3. **Done:** rewrite the backlog entries into *§8*'s structure.
4. **Done:** FRC and StdDev-per-trace refuse a channel recorded as Tracking and
   state the violated static-emitter assumption. Manual role choices persist in
   `dataset.state["tracking_role"]`; the heuristic and its reason are stored
  separately as automatic state.
5. **Done:** trajectory indexing is off the GUI thread. The worker receives
   frozen, non-Qt snapshots, runs coordinate/time preparation and
   `build_track_set` on the process-owned `tracking-index` pool, and reports to
   the Task Monitor. Cooperative cancellation plus generation and dataset-
   identity checks prevent an older filter/mode/source result from publishing.
   Window teardown retires and disconnects work without waiting. The pool is
   deliberately serial to cap peak memory when a user changes settings during a
   many-million-row build.

### 5.2 The analysis core — foundation implemented

`analysis/tracking_stats.py` is pure and Qt-free. It now provides:

- `PreparedTrajectories`: calibrated-nm coordinates, original row indices,
  trace IDs, rounded absolute/trace-relative time, explicit segment boundaries,
  source scope and preprocessing provenance;
- `prepare_trajectories(...)` and `trajectories_from_dataset(...)`: explicit
  timestamp/index mode and filtered/materialized-unfiltered scope. Rounding and
  onset calculation precede filtering. Removed middle rows, duplicate/reversed
  time, and long gaps become boundaries. **No position is interpolated**;
- a common `TrackingResult` envelope with level-specific tables, units,
  diagnostics, exclusions, provenance and citations;
- `TrackingMethodSpec` plus collision-safe register/get/list/run functions, the
  extension seam for project-specific plugins;
- localization/segment **kinematics**: `dt`, jump distance, speed, turning
  angle, duration, path length, net displacement and straightness;
- measured-lag, irregular-time **MSD**: per-segment curves, pooled time-averaged
  MSD, ensemble MSD, short-lag Brownian `D`, descriptive anomalous exponent
  `α`, intercept, apparent σ, pair counts and fit diagnostics.

The intercept-derived sigma is deliberately named **apparent** rather than
motion-corrected: exposure-time blur and feedback-loop dynamic error cannot be
removed honestly without acquisition metadata. Index mode reports `nm²/index`,
never a physical `nm²/s` value.

Added behind the same contract after method-specific validation:

- `jump_distance_distribution`: gap-safe adjacent measured jumps plus an
  empirical radial histogram, with no imposed model;
- `fit_jump_distance_mixture`: explicit one-to-six-component isotropic
  Brownian candidates using every measured interval in
  `2 D_k Δt + 2 σ²`; AIC/BIC are reported but no component count or biological
  state is selected automatically;
- `confinement_index`: the two-dimensional Simson residence-probability index,
  requiring an explicit free-diffusion coefficient so the test is not circular;
- `ergodicity_breaking`: per-segment TAMSD amplitudes, normalized `ξ`, and the
  formal finite-time statistic `<TAMSD²>/<TAMSD>² − 1`, with unequal-duration
  and population-heterogeneity warnings;
- `TrackingStateAdapterSpec`: collision-safe, versioned registration,
  availability and execution contracts for ExTrack/saSPT/DeepSPT-style plugin
  backends. No universal backend or `state_label` is installed by default.

### 5.3 Then the UI on top

6. **Done:** `Analyze › Tracking › MSD Analysis` is a parentless modeless
   workbench. It prepares and computes off the GUI thread; plots pooled,
   ensemble and per-segment curves; exposes sortable fits, exclusions,
   provenance/citations; exports a ZIP; and can opt-in to provenance-bearing
   derived attributes.
7. **Done by retirement:** `Particle Tracking` was removed from the menu.
   MINFLUX `tid` is authoritative; a generic linker would overwrite acquisition
   identity, while merge/split repair requires experiment-specific priors.
8. **Done:** tail colour is selectable as Channel or oldest-to-newest Time.
   Time colour keeps the channel LUT on the star heads.
9. **Done:** left-click isolates the nearest visible comet; clicking empty space
   or the context action clears it. The unselected population remains in the
   backdrop.
10. **Done:** lazy OpenGL 3-D mode uses the same gap-safe time windows, bands,
    roles, selection, colours and backdrop. PyOpenGL absence is reported rather
    than silently falling back.
11. **Done:** deterministic event-loop-driven export writes a multi-page RGB
    TIFF and JSON sidecar over the complete current axis, preserving the current
    tail, projection, colours, backdrop and selection. It writes to a partial
    path and replaces the target only on completion; cancel/close removes the
    partial file.
12. **Done:** `AppState.tracking_playhead_changed` synchronizes dataset object,
    selected `tid`, and trace-relative time bidirectionally between Tracking View
    and Trace Viewer, with a source token preventing feedback loops.
13. **Done:** hover reports trace ID, localization count, duration, net
    displacement and median speed; this stays descriptive rather than assigning
    a motion state.

### 5.4 Relationship between the two viewers — keep it explicit

**Trace Viewer = one trace, deep** (time-series, quality attributes, tail
removal). **Tracking View = all traces, spatial, over the structure.** They
should share the time model (`core/tracking_time.py`) and eventually the
playhead, but they answer different questions and should not be merged.

---

## 6. Literature review and decisions

The first review is complete. Its most important outcome is that MINFLUX's
timestamp digits, localization cadence, localization uncertainty, and dynamic
tracking error are four different quantities and must not be collapsed into one
"time resolution" preference.

### 6.1 Decisions adopted in the foundation

1. **Keep the default rounding precision at 1 µs, but describe it as numerical
   timestamp quantization, not instrument resolution.** Schmidt et al. reported
   effective steps down to about 80.5 µs and mean sampling around 117 µs with
   <20 nm conservative lateral precision; the acquisition itself varied with
   photons and sequence settings
   ([Nature Communications 2021, DOI 10.1038/s41467-021-21652-z](https://doi.org/10.1038/s41467-021-21652-z)).
   The viewer therefore preserves raw `tim`, rounds only a derived working axis,
   and measures the modal interval from each dataset.
2. **Do not interpolate by default.** Recent two-colour MINFLUX SPT work states
   explicitly that intervals are irregular and treats step lags accordingly
   ([Nature Communications 2025, DOI 10.1038/s41467-025-61489-4](https://doi.org/10.1038/s41467-025-61489-4)).
   `tracking_stats.py` uses actual pairwise lags and bins them; missing or
   filtered rows become boundaries. Interpolation may later be offered only as
   an explicitly labelled visualization/derived-data operation.
3. **Treat the MSD intercept as apparent precision until acquisition metadata is
   available.** Classical MSD work establishes joint sensitivity to diffusion,
   localization error and motion blur
   ([Michalet 2010](https://doi.org/10.1103/PhysRevE.82.041914);
   [Berglund 2010](https://doi.org/10.1103/PhysRevE.82.011917)). MINFLUX adds a
   feedback/search sequence: simulation shows that timing/dead time and loss of
   fast molecules can bend MSD and underestimate diffusion
   ([Nature Communications 2026, DOI 10.1038/s41467-025-66952-w](https://doi.org/10.1038/s41467-025-66952-w)).
   A future dynamic-error correction must consume sequence/exposure/dead-time
   metadata or a validated instrument model; it must not infer them from decimal
   precision.
4. **There is no honest universal state classifier.** ExTrack jointly estimates
   diffusion states, transitions and localization error under its model
   ([JCB 2023](https://doi.org/10.1083/jcb.202208059)); saSPT estimates occupancy
   over diffusion/error state grids and is especially useful for populations of
   short tracks ([eLife 2022](https://doi.org/10.7554/eLife.70169)); DeepSPT
   supplies learned temporal motion labels. Their assumptions and dependencies
   differ. The viewer should expose them as adapters behind `TrackingMethodSpec`,
   with method/version/model provenance, rather than bake one into `state_label`.
5. **Stepping is a separate workbench from diffusion.** MINFLUX kinesin analysis
   uses axis alignment, iterative change-point fitting, spike filtering and an
   HMM for bound/unbound substep sequences
   ([Communications Biology 2024, DOI 10.1038/s42003-024-06358-4](https://doi.org/10.1038/s42003-024-06358-4)).
   Its priors (for example expected 8/16 nm steps) are experiment-specific and
   must live in a dedicated command/plugin, not in generic MSD.
6. **Visual defaults should preserve scientific context and time.** Published
   MINFLUX figures use time-coloured trajectories, position-versus-time plus
   fitted steps, and structure/cargo overlays. The current channel-coloured
   comets preserve overlay identity; time colour remains a selectable future
   style. Two-colour NPC tracking supports retaining the static scaffold as a
   separate context layer
   ([Nature 2025, DOI 10.1038/s41586-025-08738-0](https://doi.org/10.1038/s41586-025-08738-0)).
7. **Show empirical jumps before fitting states.** Probability distributions of
   squared displacements can resolve mobility components
   ([Schütz, Schindler & Schmidt 1997, DOI 10.1016/S0006-3495(97)78139-6](https://doi.org/10.1016/S0006-3495(97)78139-6)),
   but the number and physical meaning of components remain model choices. The
   viewer therefore separates the raw jump table/histogram from an explicitly
   requested fixed-size Brownian mixture, uses each irregular measured `Δt`, and
   reports AIC/BIC without automatically selecting a component count or writing
   a `state_label`.
8. **Confinement needs a stated free-diffusion null model.** The Simson method
   asks how improbable it is for 2-D free Brownian motion with known `D` to
   remain within radius `R` for time `t`
   ([Biophysical Journal 1995, DOI 10.1016/S0006-3495(95)79972-6](https://doi.org/10.1016/S0006-3495(95)79972-6)).
   `confinement_index` requires `D` as explicit caller input and records window
   length and the probability formula; it refuses 3-D rather than silently
   applying the 2-D coefficients. Its output is an index, not a state label.
9. **Ergodicity is an amplitude-distribution question, not an MSD ratio.** The
   implemented statistic follows the distribution of per-trajectory TAMSDs and
   `EB = <TAMSD²>/<TAMSD>² − 1`
   ([Burov et al. 2011, DOI 10.1039/C0CP01879A](https://doi.org/10.1039/C0CP01879A)).
   It reports normalized `ξ` and observation duration for every contributing
   segment and warns that finite/unequal tracks, localization error and a
   heterogeneous population can broaden the statistic without proving a
   non-ergodic stochastic process.
10. **Generic integration means contracts, not vendoring every classifier.**
    ExTrack, saSPT and learned approaches answer different questions and bring
    different model/dependency burdens. `TrackingStateAdapterSpec` lets a plugin
    publish availability, method and model versions, citations, parameters and
    a common `TrackingResult`; no backend is privileged by the core viewer.

### 6.2 Seeds given by the repository owner

- Kinesin stepping, MINFLUX — https://pubmed.ncbi.nlm.nih.gov/39704940/
- Science, MINFLUX tracking — https://www.science.org/doi/10.1126/science.ade2650
- Commun. Biol. 2024 — https://www.nature.com/articles/s42003-024-06358-4
- The owner's own two-colour NPC work — https://www.nature.com/articles/s41586-025-08738-0
  and https://github.com/npctat2021/MINFLUX_NPC_Tracking

### 6.3 Questions for later, method-specific reviews

1. **What sampling and precision does MINFLUX tracking actually achieve**, per
   published system and per experiment? (Already in hand: Balzarotti 2017 at
   ~µs-scale; Schmidt 2021 80.5 µs and 117 µs/localization; Eilers 2018 400 µs;
   Wolff dynein 0.28 ms / 150 µs dwell; this repository's reference file 0.9 ms.)
   Needed to justify defaults, not to hard-code them.
2. **How is localization precision reported for a *moving* emitter** in the
   MINFLUX literature? Specifically: is the MSD-intercept decomposition
   (σ and D from one fit) what the field uses, or something MINFLUX-specific
   arising from the beam-following loop?
3. **How is the tracking-loop lag / dynamic error treated**, and can it be
   estimated from the data?
4. **Which motion-state segmentation method** is standard for MINFLUX-rate data?
   Methods built for camera SPT at 10–50 ms may not transfer to sub-ms sampling
   with very few photons per point.
5. **Stepping analysis** — kinesin/dynein/myosin: step-finding on MINFLUX traces
   (change-point detection, t-test/Kerssemakers/Chi-square-style step fitting),
   dwell-time distributions, and how substeps are established rather than
   asserted. This is a distinct workstream from diffusion analysis and may
   deserve its own command.
6. **Anomalous diffusion and confinement** at these rates; ensemble- vs
   time-averaged MSD and ergodicity breaking.
7. **What visualisation conventions** the field uses for tracking figures —
   time-coloured tracks, comet tails, structure overlays — so the view's defaults
   match what a reader expects to see.
8. **Two-colour structure + cargo** beyond NPC: which other systems, and what
   alignment/assignment they use (the NPC repo uses bead fiducials for the affine
   transform between channels, then per-cluster rotation).

### 6.4 Search practicalities

- Sources: PubMed, bioRxiv, arXiv (q-bio.QM, physics.optics), Nature/Science/
  Nat. Methods/Nat. Commun./Commun. Biol., and the Ries, Hell, Jungmann and
  Lasker groups' software repositories.
- Terms: `MINFLUX tracking`, `MINFLUX single-particle tracking`, `MINFLUX
  stepping kinesin`, `sub-millisecond single-molecule tracking`, `localization
  precision moving emitter`, `MSD localization error intercept`, `dynamic
  localization error`, `motion blur SPT`, `anomalous diffusion exponent SMLM`,
  `two-colour tracking nuclear pore transport`.
- Also survey **what other tools already do** — pyMINFLUX, SMAP, TrackMate,
  Picasso, `swift`, `ExTrack`, `SPTAnalyser` — so this viewer's tracking analysis
  is comparable rather than idiosyncratic. Note explicitly where we deliberately
  differ.
- **Output for each added method:** a short annotated bibliography with verified
  DOIs, a recommendation against alternatives, synthetic/real-data validation,
  and citation entries ready for `analysis/method_text.py`. Paper-backed methods
  carry a verified DOI; in-house methods carry an inline methodology note and
  `url=None`.

---

## 7. What was reverted, and must stay reverted

The time row in Render and Loc Scatter Plot, and everything it touched:
`render_window.py`, `scatter_window.py`, `tile_cache.py` (`TileKey` is back to 8
fields), `render_scheduler.py`, `precision_render.py`,
`precision_render_window.py`. Verified by diffing each against `HEAD`.

**Do not reintroduce a time gate into those two viewers.** The lesson is
recorded, not the code: gating an established viewer changes what it is. If a
future need arises for time gating *inside* Render, it should be argued
separately and not as a side effect of tracking work.

Kept from that work because the Tracking View needed them:
`core/tracking_time.py` (the time-axis model, the three axis modes, the interval
diagnostics) and `ui/time_slider.py` (the playback row).

---

## 8. BACKLOG.md is updated

The two obsolete one-line entries were replaced with one structured tracking
entry. It distinguishes what is now done (playback plus pure analysis
foundation) from the UI, validated advanced methods, stepping workbench,
and 3-D/export/synchronization work that remains. Off-thread indexing is now
also complete.

---

## 9. Verification the next session should repeat

```text
.venv\Scripts\python.exe -m pytest \
  tests/test_tracking_stats.py tests/test_tracking_advanced.py \
  tests/test_tracking_attributes.py tests/test_tracking_analysis.py \
  tests/test_tracks.py tests/test_tracking_time.py \
  tests/test_tracking_preferences.py tests/test_tracking_view.py \
  tests/test_tracking_precision_guard.py tests/test_volume_multichannel.py \
  tests/test_precision_histograms.py tests/test_background_tasks.py \
  tests/test_window_stacking.py tests/test_qt_lifecycle_regressions.py -q \
  --basetemp=.tmp/pytest-tracking-final -p no:cacheprovider
155 passed (2026-09-27; workspace-local basetemp avoids the machine's locked
global pytest temp/cache directories)

.venv\Scripts\python.exe -m ruff check \
  minflux_viewer/analysis/tracking_stats.py \
  minflux_viewer/analysis/tracking_advanced.py \
  minflux_viewer/analysis/tracking_adapters.py \
  minflux_viewer/analysis/tracking_attributes.py \
  minflux_viewer/analysis/tracking_export.py minflux_viewer/core/tracks.py \
  minflux_viewer/core/tracking_time.py minflux_viewer/ui/tracking_window.py \
  minflux_viewer/ui/tracking_analysis_window.py \
  minflux_viewer/plugins/trace_viewer/trace_viewer_window.py \
  tests/test_tracking_stats.py tests/test_tracking_advanced.py \
  tests/test_tracking_attributes.py tests/test_tracking_analysis.py \
  tests/test_tracks.py tests/test_tracking_preferences.py \
  tests/test_tracking_precision_guard.py tests/test_tracking_view.py \
  tests/test_qt_lifecycle_regressions.py
All checks passed (2026-09-27)
```

The full suite was attempted on 2026-09-26 and hit the repository's documented
nondeterministic Qt/pyqtgraph native abort at 2%, inside
`pytestqt.plugin._process_events`; pytest produced no assertion failure or final
summary. The focused suite above completed cleanly. `CLAUDE.md` records this
teardown behaviour; **re-run a suspected regression in isolation before
attributing it.**

### Files that define the new foundation

- `analysis/tracking_stats.py` — input/result/method contracts, preprocessing,
  kinematics and MSD;
- `analysis/tracking_advanced.py` — empirical jumps, explicit Brownian-mixture
  candidates, Simson confinement and finite-time EB;
- `analysis/tracking_adapters.py` — optional state-inference backend contract;
- `analysis/tracking_attributes.py`, `analysis/tracking_export.py` — opt-in
  auditable dataset attributes and portable table/metadata ZIP export;
- `core/tracks.py` — playable index plus filter-discontinuity preservation and
  recorded channel-role lookup;
- `core/tracking_time.py` — rounding, per-trace/index axes and interval
  diagnostics shared by playback and analysis;
- `ui/tracking_window.py` — persistent roles, channel/time-coloured comets,
  2-D/3-D playback, selection/hover, TIFF export, playhead synchronization,
  all-channel absolute-time bounds and cancellable generation-safe indexing on
  the process-owned `tracking-index` pool;
- `ui/tracking_analysis_window.py` — modeless cancellable MSD workbench;
- `plugins/trace_viewer/trace_viewer_window.py` — the other endpoint of shared
  `tid`/playhead synchronization;
- `analysis/localization_precision.py` — role-aware FRC/StdDev refusal;
- `tests/test_tracking_stats.py`, `tests/test_tracks.py`,
  `tests/test_tracking_view.py`, `tests/test_tracking_precision_guard.py` — the
  corresponding regression contract; `tests/test_qt_lifecycle_regressions.py`
  additionally proves that closing during a running index does not block or
  deliver into deleted widgets.

Reference data used, and worth reusing:

- `D:\Workspace\Microscopes\MINFLUX\sample data\1_sample_A_1-100_seqTrk-3D-Seb_Octahedron_100_pho_exc_10.mat`
  — 45,105 locs, 230 traces, 0.9 ms sampling, 12.7 nm median step, 104 nm median
  end-to-end, 1,446 s run. The new foundation read 45,105 rows into 230
  segments, measured a 0.896 ms modal interval, and completed preparation plus
  3-D MSD in **1.35 s**. The initial bin implementation took 16.49 s on the same
  data; replacing repeated per-bin scans with grouped reductions preserved the
  result and reduced this by about 12×. The pooled short-lag diagnostic was
  `D = 46,151 nm²/s` (`0.0462 µm²/s`) and `α = 0.990`; these are smoke-test
  outputs, not yet a reviewed biological result.
- `…\1_sample_A_1-100_seq_3D_ori_exc_5.msr` — 58,577 locs, 122 traces, 146 nm
  median end-to-end, 921 s run.
- The two simulated presets, for a known ground truth.

Full detail of the implementation and its measurements is in
`TRACKING_VIEW_HANDOFF.md`.

---

## 10. Deliberate boundaries and plugin handoff

These are not silent TODOs; they are the points at which the generic viewer
must stop making assumptions:

- **No feedback-loop dynamic-error correction yet.** Timestamp decimal digits
  do not identify exposure, dead time, tracking-loop latency or loss bias. The
  MSD intercept remains `msd_sigma_apparent` until sequence/loop metadata and a
  validated instrument model are supplied.
- **No universal `state_label`.** Optional inference backends register through
  `TrackingStateAdapterSpec` and must return method/model version, parameters,
  diagnostics, citations and row mapping in a `TrackingResult`. ExTrack, saSPT
  and DeepSPT-style plugins may coexist.
- **No generic trace relinking.** A plugin may propose merge/split edits, but it
  must preserve original `tid`, publish the repair rule and parameters, preview
  the mapping, and write a derived/repaired identity rather than silently
  replacing the acquisition field.
- **Stepping/dwell analysis is a plugin/workbench.** Axis definition, expected
  step family, spike handling, change-point method and HMM topology are motor-
  and experiment-specific. The prepared-trajectory/result/export/task contracts
  are ready for it, but the generic viewer does not assert those priors.
- **Advanced methods currently expose pure Python/registry/export contracts.**
  The built-in GUI specializes in MSD, the recurrent model-independent entry
  point. A plugin can build experiment-specific controls over jump mixtures,
  confinement, EB or a state adapter without duplicating preprocessing,
  cancellation, provenance or ZIP serialization.
- **Movie output is TIFF + JSON, not MP4.** TIFF keeps exact frames and avoids a
  bundled codec/runtime dependency; an MP4 plugin can consume the deterministic
  frame sequence. Track picking/hover is provided in 2-D; a selected track
  remains isolated in 3-D, where OpenGL picking is not claimed.
