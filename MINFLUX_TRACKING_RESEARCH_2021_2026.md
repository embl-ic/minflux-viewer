# MINFLUX tracking, 2021–2026: methods, software, precision, and a roadmap for `minflux-viewer`

**Document type:** technical research report  
**Research cut-off:** 12 September 2026  
**Scope:** peer-reviewed work and identifiable preprints from 2021 onward, with emphasis on genuine trajectory acquisition or analysis. Static MINFLUX studies are included only when they introduce a tracking-relevant method such as drift correction, calibration, sequence simulation, multiplexing, or uncertainty estimation.  
**Audience:** developers and scientific users of `minflux-viewer`.

## Executive conclusions

1. **MINFLUX “tracking” spans two different problems.** During acquisition, feedback moves the excitation-coordinate pattern around the latest estimated emitter position. After acquisition, most studies do not link camera detections into tracks: they accept the instrument-assigned `tid`, select the final valid localization from each iteration cycle, then validate, trim, split, transform, classify, and model the resulting trajectory. `minflux-viewer` should preserve that distinction.

2. **No single localization-precision number is valid for every track.** The literature uses at least five estimands: a photon/model-based CRLB; the standard deviation of repeated localizations of a static emitter; consecutive-difference precision; residual precision around a stepwise or constrained model; and dynamic localization uncertainty jointly fitted with diffusion. For freely moving tracks, a raw within-track standard deviation is a measure of spatial exploration, not localization precision.

3. **The most reusable workflow is not an application-specific algorithm.** It is a non-destructive pipeline:

   `select final-valid observations → validate order/units → QC → propose segments → inspect/edit → estimate uncertainty → fit motion models → compare diagnostics → export results with provenance`.

4. **The current viewer already has a strong front half.** It imports raw and materialized MINFLUX data, exposes filters and iteration data, computes basic trace observables, contains drift-correction and localization-precision tools, and has a useful MATLAB-derived Trace Viewer. The highest-value next step is a shared trajectory-analysis core plus a general Trajectory Analysis plugin—not another one-off viewer.

5. **Three project-specific workflows should remain plugins over that core.** The NPC workflow needs reference-structure fitting, bead/channel registration, pore assignment, and pose normalization. The nanoparticle workflow needs trace-to-particle grouping and surface geometry. Motor work needs axis projection, change-point/step inference, and manual correction. These models should not become assumptions in the base data model.

6. **Acquisition censoring must be visible.** MINFLUX loses fast or poorly centered emitters preferentially. SimuFLUX and parameter-optimization work show that apparently clean surviving tracks can underestimate diffusion. Track survival, termination reason when available, sequence parameters, and completeness diagnostics therefore belong beside every diffusion result.[17,22]

## 1. How this review was conducted

The search began with the two supplied projects: the published NPC transport repository and the locally cloned nanoparticle tracking project. Their associated papers, code-availability statements, and cited MINFLUX methods were followed outward. Searches then covered combinations of *MINFLUX*, *tracking*, *trajectory*, *diffusion*, *stepping*, *live cell*, *software*, *GitHub*, *bioRxiv*, and the years 2021–2026. Version-of-record articles were paired with their preprints so that the same study is not double-counted. Primary papers, repositories, and author-provided archives were preferred over reviews.

This is a broad, targeted technical review rather than a registered systematic review. It should be close to complete for prominent MINFLUX tracking applications discoverable by the cut-off date, but negative statements such as “no public code identified” mean that no linked public analysis package was found—not that private code does not exist.

## 2. What a MINFLUX trajectory contains

MINFLUX repeatedly localizes the same emitter by placing an excitation minimum at several positions in a targeted coordinate pattern (TCP). Later localization iterations use a smaller pattern and the estimated position becomes the center for the next tracking cycle. In commercial files this normally produces repeated iteration records plus a materialized localization stream grouped by `tid`.[1,2]

That produces four analytically distinct layers:

| Layer | Typical content | Appropriate responsibility |
|---|---|---|
| Acquisition attempts | TCP iteration, photons, validity, CFR, background, timing | Preserve as raw evidence; inspect sequence performance |
| Final-valid observations | `tid`, time, x/y/z, efo/eco/dcr and channel values | Canonical trajectory input |
| Segments and annotations | gaps, approach/tail, ROI membership, motion state, accepted/rejected intervals | Non-destructive derived state with provenance |
| Scientific models | diffusion, confinement, steps, NPC pose, sphere surface, channel transform | Versioned results, usually plugin-specific |

The viewer should not overwrite the first two layers when producing the latter two. A `tid` is also not always the final biological unit: the nanoparticle study groups multiple `tid` traces on one silica particle, while the NPC study assigns cargo tracks to fitted pores.[13,14]

## 3. Literature map: application studies

### 3.1 Core tracking applications

| Year and status | Project and biological/physical question | Acquisition and tracking analysis | Software and availability | Transferable lesson |
|---|---|---|---|---|
| **2021, peer reviewed** | Fast 2D diffusion of ATTO 647N–DPPE in a supported lipid bilayer on a standard microscope stand | Repeated final MINFLUX iteration with feedback recentering; about 30 photons/localization and effective steps down to about 80.5 µs. Consecutive lateral displacement spread was reported as a conservative precision bound. | Imspector v16 plus custom microscope drivers and supplementary Python software.[1] | Preserve iteration/sequence metadata and describe consecutive-difference estimates as dynamic upper bounds unless motion is modeled. |
| **2021, peer reviewed** | Pulsed-interleaved MINFLUX (pMINFLUX): lifetime-capable, rapid localization and nanoscopic motion | Excitation positions are encoded by interleaved pulses, allowing position and lifetime to be inferred from photon arrival channels. | Open Python control package **pyflux** and analysis/MLE package **p-minflux**.[2,3] | Make photon timing/lifetime and detector-channel attributes extensible; do not assume every MINFLUX trajectory has the commercial schema. |
| **2022 preprints → 2023 peer reviewed** | Kinesin-1 stepping in vitro and in living cells | Trajectories are projected onto the microtubule axis; iterative change-point/step fitting and HMM-style analyses recover 8-nm substeps or 16-nm center-of-mass steps. Precision is assessed from plateau residuals or consecutive differences, not whole-track spread. | One study released MATLAB curvature, step-finder, and HMM scripts on Zenodo; the live-cell study used **SMAP** with custom motor-PAINT and `stepsMINFLUX` plugins plus manual review.[5,6] | Axis projection, change-point proposals, model residuals, and editable boundaries are reusable, but belong in a motor-analysis plugin. |
| **2024, peer reviewed** | 3D bead and fluorophore diffusion with 4Pi-MINFLUX | Interferometric opposed objectives improve axial information; bead trajectories reach sub-nanometre 3D uncertainty at kilohertz rate. MSD/free- and hop-diffusion analyses are used. | Custom acquisition/analysis reported with the article; no general viewer package identified.[9] | Store anisotropic uncertainty and coordinate-frame/calibration metadata; never collapse 3D precision to one lateral value by default. |
| **2024, peer reviewed** | Super-resolved FRET and co-tracking in pMINFLUX | Track a donor while estimating the acceptor by FRET multilateration, or co-track lifetime-separated fluorophores of similar spectra. | Study code available from authors on request; built on pMINFLUX concepts.[10] | A future trajectory model must support multiple synchronized positions/identities and uncertainty, not just one `tid → xyz` stream. |
| **2024, peer reviewed** | Kinesin dynamics and microtubule switching in neurites | Light fixation preserves trajectories; axis projection, iterative step fitting and HMM analysis identify stepping and track changes. | Custom LabVIEW acquisition/screening, MATLAB R2022b analysis, Fiji; code and data on Zenodo.[11] | Separate acquisition screening from reproducible offline filters, and retain manual start/end edits as annotations. |
| **2024, peer reviewed** | Endogenous dynein stepping in live neurons | 2D tracks up to micrometres; a bias-reduced step algorithm infers plateaus, steps, and dwell times without fixing step count or size. | Imspector plus custom analysis; no public reusable package was identified from the article.[12] | Step inference needs BIC/likelihood diagnostics and uncertainty per event, not only a filtered trace. |
| **2025, peer reviewed** | Import and export paths through individual nuclear pore complexes | First reconstruct the NPC scaffold, then use bead-based two-colour registration, assign cargo tracks to pores, transform each pore to a common pose, classify complete/abortive events, and fit displacement distributions. Dynamic uncertainty is jointly inferred with diffusion-like motion. | Imspector; MATLAB analysis; ParaView, Fiji, Origin, Kaleidagraph and Excel for ancillary work. Full MATLAB workflow at `npctat2021/MINFLUX_NPC_Tracking`.[13] | This is a reference-frame and registration workflow. Its filtering/segmentation can be generic; pore fitting, eightfold alignment and transport classification should be an NPC plugin. |
| **2025, peer reviewed** | 3D lipid mobility on silica-supported lipid-bilayer nanoparticles | Trim track ends, split temporal gaps, group `tid` centroids into particles with DBSCAN, fit sphere/sphericity, compute raw and regularized all-pair MSD and diffusion summaries, and inspect them interactively. | Custom MATLAB workflow, locally supplied and public as `EMBL-ICLM/MINFLUX-particle-tracking-and-diffusion-analysis`.[14] | Excellent source for particle grouping and geometry UI. Reimplement the statistics with robust gap inference, pair-count weighting, motion/noise models, and bounded-memory computation. |
| **2025, peer reviewed** | Concurrent 2D diffusion of nicotinic acetylcholine receptors and fluorescent cholesterol | DCR-based two-colour separation; irregular-time turning angles and time-averaged MSD; nonlinear free/anomalous/confined/hop fits with motion blur and localization error; BIC model selection; spatial overlap analysis and CONDOR classification. | Custom Python using NumPy/SciPy, `roipoly`, SciPy/Shapely spatial tools, and CONDOR; public analysis repository and Zenodo archive.[15] | This is the strongest general template for irregular-time diffusion/model comparison, though deep-learning classification should remain optional. |
| **2025, peer reviewed** | Dense-sample live tracking using gradual fluorogenic labeling (GLF-MINFLUX) | Sequential label–locate–bleach cycles reduce simultaneous emitters. Live membrane proteins were tracked at about 200 µs and reported 7.8-nm precision. Consecutive x/y differences were used for tracking precision. | Imspector 16.3.15636, MATLAB 2022b custom scripts, ImageJ, ParaView.[16] | Store labeling/acquisition recipe metadata. The analysis again illustrates why “difference spread” must state its normalization and motion assumptions. |
| **2025, peer reviewed** | Systematic optimization of MINFLUX single-particle-tracking parameters | Varies TCP cycles/repeats, dwell, photon limit, laser power and pattern diameter `L`; compares speed, bias and trackable diffusion. | Imspector; TIRF reference trajectories linked with `trackpy` 0.6.4; raw data and sequences public, but the custom MINFLUX analysis code was not publicly released at publication.[17] | Sequence configuration and track-loss statistics are first-class provenance. A shorter sequence can improve temporal fidelity while changing precision and censoring. |
| **2025, preprint** | *Clustering within a single-component biomolecular condensate*: slow 3D FUS motion and internal heterogeneity | Gold-particle stabilization, axial scaling, track QC, jump distributions, MSD, multi-mobility diffusion and confinement simulations. | Imspector 16.3.13924; custom analysis; no linked public code identified.[18] | Add mixture/state diagnostics and simulation comparison, but report the preprint status and avoid presenting fitted components as uniquely identified. |
| **2026, peer reviewed** | Chromatin dynamics from 200 µs to hours across cell types | Combines 5-kHz MINFLUX, camera SPT and slow imaging. MINFLUX tracks are QC-filtered and modeled with Bayesian MSD including localization error and motion blur; simulations quantify TCP loss bias. | Imspector 16.3.15645, Python/Jupyter, **bayesmsd**, TrackMate 7.14; public code and Zenodo data.[19] | A viewer should allow external estimators through a stable trajectory/result API and show time-scale coverage plus censoring diagnostics. |
| **2026, peer reviewed** | Event-triggered MINFLUX at rare caveola, endocytosis and HIV-budding events | Real-time confocal analysis triggers 2D/3D MINFLUX. Offline analysis filters and splits tracks, jointly estimates dynamic precision and diffusion from irregular displacements, computes sliding-window diffusion, and detects confinement with a local convex-hull packing coefficient. | Python control via `specpy`; public control, analysis, and napari viewer repositories using Qt/PyQtGraph, NumPy/SciPy, scikit-image, OpenCV and trackpy.[20] | Segment-by-ROI, rolling analysis, event metadata and packing-coefficient plots are highly reusable. Hardware control should remain external to `minflux-viewer`. |
| **2026, peer reviewed** | PIEZO2 membrane dynamics alongside structural studies | Final-valid 3D tracks are axially scaled, efo-filtered, truncated at the first long gap and restricted by trace length. Short- and long-lag MSD regimes are fit for diffusion. | Imspector 16.3.15645; public custom MATLAB repository and Zenodo data.[21] | Adopt explicit gap segmentation and pair-count weighting. Do not copy a zero-intercept MSD fit as the generic default because it hides localization error. |
| **2025 preprint → 2026 peer reviewed** | SimuFLUX: in-silico evaluation and optimization of imaging and tracking | Simulates emitter paths, photophysics, TCP measurements, localization and feedback. Quantifies CRB, RMSE, bias, track loss, and the downward bias of diffusion among surviving fast tracks. | Open GPL MATLAB/Python, Jupyter/Colab, SMAP-format outputs and Zenodo release.[22] | Use SimuFLUX as the synthetic ground-truth generator and regression benchmark for the viewer. |
| **2024 preprint → 2026 peer reviewed** | Productive dynein stepping, including two-colour and sub-millisecond measurements | Step inference in 2D, ATP-dependent dwell/kinetic fits, bead and trajectory-based channel registration; fast acquisition reveals brief backward dips. | Custom MINFLUX analysis; source data are public, but no standalone general analysis package was identified.[23] | The result model must support channel transforms, step-event uncertainty and alternative kinetic fits. |
| **2026, peer reviewed** | ER–lipid-droplet membrane bridges and nanodomain partitioning | Spatial masks classify track regions; local consecutive-difference precision, rolling MSD/diffusion, whole-track anomalous fits, KDE nanodomain states, and DeepSPT on camera trajectories. | Public Python repository using pandas/OpenCV/SciPy; DeepSPT for the companion HILO data.[24] | Region masks, rolling windows and per-localization state annotations are reusable; learned models should be adapters with explicit model/version metadata. |

### 3.2 Tracking-relevant methods and software

| Work/tool | Contribution | Relevance to the viewer |
|---|---|---|
| DNA-PAINT MINFLUX (2022) | Reference structures, final-iteration processing and drift correction; public MATLAB code/data.[4] | The viewer already contains a pyMINFLUX-derived drift workflow. Retain bead/reference diagnostics and distinguish corrected coordinates from raw coordinates. |
| Blob-B-Gone (2023) | Unsupervised detection of compact/blob-like trajectory artefacts from geometric features in 2D/3D, implemented with scikit-learn.[7] | Port features and scores as optional QC flags; do not auto-delete tracks or require an ML dependency for the base workflow. |
| ISM-FLUX (2023) | Array-detector extension that changes information collection and potential estimator inputs.[8] | Keep detector/channel schemas open and capability-based. |
| Variable-phase-plate 3D MINFLUX (2024) | A simplified route to 3D excitation patterns.[25] | Another reason not to hard-code one TCP geometry in CRLB or metadata logic. |
| DNA-origami performance preprint (2026) | Long-acquisition drift and site-precision benchmarking with known structures.[26] | Useful validation dataset for drift and static precision, but not a dynamic-track precision benchmark. |
| Live-cell kinesin tracking protocol (2025) | Practical sample preparation, acquisition and analysis workflow rather than a new biological result.[27] | Useful documentation reference for a guided recipe/preset, while keeping instrument operation outside the viewer. |
| pyMINFLUX | Maintained Python reader, processor and viewer for NPY/MAT/Zarr/PMX, trace statistics, filtering, time inspection, precision and FRC.[28] | Compare file semantics and analysis results; interoperate rather than clone the whole UI. |
| SMAP | Extensible MATLAB super-resolution platform used by motor-MINFLUX work.[29] | Its plugin separation and editable analysis are a useful precedent. |

### 3.3 Cross-study patterns

Across these projects, the recurring operations are:

- choose final valid iteration records and document the selection;
- remove invalid/multi-emitter/low-quality records using CFR, efo, dcr, photons, background or trace-length criteria;
- correct drift and apply spatial calibration without mutating raw coordinates;
- sort by time, remove duplicates if scientifically justified, and split long gaps;
- inspect trace start/end and acquisition-lock artefacts;
- map observations into a coordinate frame: microtubule axis, fitted sphere, NPC frame, channel transform, or segmented cellular ROI;
- estimate localization uncertainty appropriate to the motion model;
- compute displacements, turning angles, MSD/covariance and rolling statistics;
- compare free, directed, anomalous, confined, hop, mixture, or stepwise models;
- save inclusion decisions, parameters, diagnostics and software versions.

These operations should define the common API. The scientific frame and model determine the plugin.

## 4. How to quantify localization precision for tracking data

### 4.1 First define the estimand

“Localization precision” can mean:

- **single-observation random uncertainty** in x, y and z;
- **uncertainty of a trace centroid**, which decreases roughly as `σ/√n` only for independent stationary observations;
- **dynamic localization uncertainty** inferred jointly with a motion model;
- **step/plateau positional uncertainty** after fitting a stepwise model;
- **ensemble spatial resolution**, such as FRC, which is not a per-track uncertainty.

The UI and exports should use these full names. A bare “precision = 4 nm” is insufficient.

### 4.2 Recommended estimators

#### A. CRLB or estimator-reported uncertainty

Use the photon counts, background/SBR, TCP geometry, doughnut width and estimator model to obtain a per-localization lower bound. This is valuable for comparing sequence configurations and diagnosing photon-limited observations. It is conditional on the optical/statistical model and generally excludes sample motion, drift, calibration error, model mismatch and feedback loss. Therefore report it as **CRLB**, not measured precision.[17,22]

#### B. Repeated-localization spread for static emitters

For an immobilized emitter or stable structural site, compute robust or sample standard deviations per axis for each group with enough observations:

`σx = SD(x)`, `σy = SD(y)`, `σz = SD(z)`, and `σr = sqrt((σx² + σy²)/2)`.

Report the median and distribution across groups, with the group count and minimum localizations. This is appropriate for beads, DNA-origami sites and repeated localizations of a static label. It is **not** appropriate for a freely moving `tid` because diffusion and confinement contribute to the spread.[4,26]

The centroid standard error `σ/√n` is a different quantity. It may be useful for fitted NPC/scaffold centers, but should be labeled **centroid precision** and should account for temporal correlation where present.

#### C. Consecutive-difference estimate

For a static emitter with independent, equal-variance errors,

`σloc,x = SD(x[i+1] − x[i]) / sqrt(2)`

and similarly for y and z. A robust MAD-based scale can reduce sensitivity to jumps. This suppresses slow drift, but for a Brownian coordinate (ignoring blur),

`Var(Δx) = 2σloc,x² + 2DΔt`.

Consequently, applying the static formula to a moving track gives an upper bound that includes motion. Positive temporal error correlation can instead reduce the difference variance. Report the lag distribution, normalization, motion assumption and whether drift correction preceded the calculation. Several MINFLUX studies use consecutive differences, sometimes explicitly as a conservative bound.[1,6,16,24]

#### D. Joint dynamic-noise and motion fit

For freely diffusing trajectories, estimate localization uncertainty together with motion. In the simplest ideal d-dimensional model,

`MSD(τ) = 2 d D τ + 2 d σloc²`,

but finite exposure/TCP acquisition introduces motion blur and MINFLUX timestamps are often irregular. A direct likelihood, covariance estimator, or displacement fit that uses the actual intervals and blur model is preferable to ordinary least-squares fitting of many correlated MSD points. Established SPT work provides optimal and covariance-based estimators, and recent MINFLUX studies explicitly fit dynamic precision with diffusion.[15,19,20,30,31]

At minimum, the result should include `D`, `σx/y/z` (or an explicitly isotropic `σ`), confidence intervals, the lag range, blur/exposure assumption, number of tracks, number of displacement pairs per lag, residuals and model-selection score.

#### E. Model-residual precision for constrained or stepwise motion

For motors, first project onto a fitted/known filament axis, infer plateau boundaries, and estimate noise from residuals around plateaus while allowing uncertainty in the change points. For surface or pore transport, fit the geometric frame and separate registration/frame uncertainty from localization noise. Report both observation uncertainty and uncertainty of fitted steps, centers, radii or transforms.[5,6,12,13,23]

#### F. Simulation-based accuracy, bias and completeness

When a known ground truth is available, report RMSE and bias, not only standard deviation. For feedback tracking, also report the fraction of paths acquired, survival versus speed/displacement, and the difference between the true population and surviving tracks. SimuFLUX is the clearest open benchmark for this purpose.[22]

### 4.3 What `minflux-viewer` should display

For every precision result, show:

| Required field | Example |
|---|---|
| Estimand | dynamic single-localization uncertainty |
| Method | irregular-time Brownian likelihood with blur |
| Axes | σx, σy, σz; plus σr only as a convenience |
| Population summary | median [IQR], bootstrap 95% CI, 73 accepted segments |
| Acquisition context | 2D/3D, sequence/TCP, `L`, photons/SBR, temporal interval distribution |
| Preprocessing | final-valid rule, filters, drift correction, z scale, gap rule |
| Diagnostics | residuals, lag counts, convergence, track survival/completeness |

FRC should remain available for static ensemble resolution, but must not be offered as interchangeable with per-localization dynamic precision.

## 5. Audit of the supplied and existing code

### 5.1 Current `minflux-viewer`

The repository already provides:

- a raw/materialized data separation and `mfx_get` access to iteration-level fields;
- final-valid localization materialization and filters;
- per-localization `dt`, displacement and speed plus repeated per-trace size, duration and path length;
- a Trace Viewer with sortable trace summaries, linked x/y/z, `dt`, RMS speed, eco/efo plots, head/tail playback and export;
- localization-precision tools for static per-trace spread, CRLB and FRC;
- drift correction and reference/bead tooling;
- a versioned extension layer suitable for project plugins.

Four correctness issues should be addressed before adding model fitting:

1. **Final-iteration semantics.** The known global-maximum-iteration path can omit tracking traces whose valid final iteration is lower than the dataset maximum. Trajectory extraction needs a tested per-event/per-cycle final-valid rule across m2410 and other supported layouts.
2. **Ordering assumptions.** Derived `dt`, speed, duration and length assume each `tid` occupies a contiguous, time-ordered block. Validate this invariant or build an index/sorted view without changing stored row order.
3. **Precision naming.** `stddev_per_trace` is appropriate for static localization clouds. On dynamic tracks it mixes motion with error. Its `mean(σ/√n)` output is centroid uncertainty, not single-localization precision; rename it and stop selecting static standard deviation automatically for tracking datasets.
4. **Tail removal.** The current velocity/distance heuristic is useful exploratory code, but setting working `tid` values to zero obscures the decision. It should emit a proposed excluded interval with reason, score and parameters, which a user can accept, edit or undo.

### 5.2 NPC transport workflow

The published MATLAB repository is a ten-stage analysis:

1. load final MINFLUX data and filter CFR, efo, dcr and trace length;
2. DBSCAN plus editable rectangular ROIs to identify NPC localization clusters;
3. fit a double-ring/cylindrical NPC model;
4. filter candidate pores by geometry, count and z position;
5. fit/clean ring points and normalize centers;
6. exploit eightfold symmetry to estimate and normalize in-plane rotation;
7. merge pores and render a composite scaffold;
8. derive a bead-based inter-channel transform for cargo coordinates;
9. assign cargo trajectories to pores and apply each pore's center/rotation;
10. inspect and play 3D NPC/cargo trajectories.[13,32]

The reusable parts are filter recipes, manual ROI editing, reference/channel transforms, nearest-structure assignment, coordinate-frame objects, an aligned-overlay viewer, and provenance. NPC cylinder/ring fitting, eightfold rotation and transport-event classification should remain in an **NPC Transport** plugin.

Hard-coded publication defaults should become named presets whose individual filters remain visible. Transform residuals and uncertainty should be shown; applying a transform must create a derived coordinate frame rather than replace raw x/y/z.

### 5.3 Nanoparticle tracking and diffusion workflow

The local MATLAB project:

- supports two MAT layouts and takes the final localization coordinates;
- excludes traces near the outer 1% x/y boundary;
- trims a requested count from both ends;
- defines a long gap as `2.5 × min(diff(global_time))`;
- applies a user-supplied axial scale;
- clusters trace centroids with DBSCAN (`eps = 200 nm`, `minPts = 1`) to group traces by particle;
- computes convex-hull sphericity and a least-squares sphere fit;
- splits time gaps, computes all-pair irregular-lag MSD, linearly interpolated regular-time MSD, and diffusion values;
- provides an interactive 3D/spatial, displacement, speed, `dt`, MSD and diffusion-distribution viewer.[14,33]

The project is scientifically useful and should be treated as the reference behavior for a future **Nanoparticle Surface Diffusion** plugin. Several implementation choices should not become generic defaults:

- `min(diff(t))` is unstable in the presence of duplicate or exceptional timestamps; infer the nominal interval robustly per sequence/trace and let the user set an absolute or relative gap threshold;
- interpolating coordinates before estimation can smooth fast motion and create correlated synthetic observations; retain regularized curves as an optional visualization, not the default estimator;
- all-pairs lag construction is quadratic in time and memory; use chunked lag accumulation or estimator-specific algorithms;
- average MSD should be weighted by valid displacement-pair counts at each lag, not only by segment length;
- `D = MSD/(2dτ)` at every lag is a diagnostic, not a fitted diffusion coefficient, and ignores the localization-error intercept and motion blur;
- sphere/sphericity and centroid DBSCAN describe the nanoparticle application, not trajectory linking.

### 5.4 Reuse decision

| Capability | Reuse now | Adapt | Keep plugin-specific | Do not port literally |
|---|:---:|:---:|:---:|:---:|
| Trace table, playback, linked diagnostics | ✓ |  |  |  |
| Final-valid extraction and raw iteration inspection |  | ✓ |  |  |
| Gap/start/end segment proposals |  | ✓ |  |  |
| Irregular-time displacement/MSD accumulation |  | ✓ |  |  |
| Static, difference and dynamic precision estimators |  | ✓ |  |  |
| NPC registration, pose and assignment |  |  | ✓ |  |
| Sphere fit, sphericity and trace-to-particle grouping |  |  | ✓ |  |
| Motor-axis projection and step/HMM analysis |  |  | ✓ |  |
| Tail-to-`tid=0` mutation |  |  |  | ✓ |
| `2.5 × min(dt)` gap rule |  |  |  | ✓ |
| Interpolation as primary diffusion estimator |  |  |  | ✓ |
| Unbounded all-pairs matrices |  |  |  | ✓ |

## 6. Proposed architecture

### 6.1 A small common trajectory model

Add pure analysis objects, independent of Qt:

- `TrajectorySet`: read-only references to observation row IDs, `tid`, time, raw coordinates, active coordinate frame, quality fields and sequence metadata;
- `TrajectoryIndex`: stable mappings among rows, traces and time-sorted views, with validation findings;
- `SegmentProposal`: trace ID, start/end row/time, proposal type, reason, score, parameters and source algorithm;
- `TrajectorySelection`: accepted observations/segments plus manual edits; never encoded by changing `tid`;
- `CoordinateFrame`: named transform chain (raw, drift-corrected, z-scaled, channel-registered, NPC-aligned, sphere-centered) with matrices/models, units and residual diagnostics;
- `AnalysisResult`: estimator/version, parameters, input-selection digest, results, confidence intervals, diagnostics, warnings, software version and random seed.

Derived coordinates should remain views/transforms. Only genuinely per-localization outputs, such as accepted-segment membership or local state probability, should optionally become namespaced derived attributes. Per-trace and per-segment tables should not be duplicated over every localization merely to make them displayable.

### 6.2 Modules and UI

Suggested implementation split:

```text
minflux_viewer/analysis/trajectory.py       validation, indexing, coordinate views
minflux_viewer/analysis/segmentation.py     gaps, start/end proposals, manual edits
minflux_viewer/analysis/trajectory_qc.py    quality and geometric features
minflux_viewer/analysis/diffusion.py        irregular-time statistics and models
minflux_viewer/analysis/precision.py        tracking-specific estimators/results
minflux_viewer/plugins/trajectory_analysis/ general UI and export
plugins/npc_transport/                      pore model and channel registration
plugins/nanoparticle_diffusion/             particle grouping and sphere surface
plugins/motor_steps/                        projection, change points and kinetics
```

The existing Trace Viewer should become the visual inspection surface used by the general plugin. Its plots need not be rewritten; they should consume `TrajectorySet` and display accepted/excluded intervals, model predictions and uncertainty.

The first implementation can use the already bundled NumPy/SciPy stack. Optional ecosystems such as `bayesmsd`, scikit-learn, CONDOR or DeepSPT should be external-plugin adapters, not mandatory application dependencies.

### 6.3 General analysis workflow

1. **Choose observations:** final-valid, explicit iteration, or a diagnostic raw-iteration view.
2. **Validate:** units, dimensionality, finite values, `tid` grouping, time order, duplicates and sequence metadata.
3. **Apply coordinate views:** drift, z calibration and optional registration; show the transform chain.
4. **Apply reproducible QC recipe:** CFR/efo/dcr/photon/background/length filters with before/after counts.
5. **Propose segments:** long gaps, initial acquisition lock, terminal artefact and optional Blob-B-Gone-like scores.
6. **Inspect/edit:** linked trajectory and time-series viewer, undoable accept/split/merge/exclude operations.
7. **Estimate precision:** static, difference-bound, dynamic joint fit, step residual or CRLB, selected explicitly.
8. **Analyze motion:** displacement/turning-angle, irregular-time MSD/covariance, rolling metrics, and candidate models.
9. **Diagnose:** pair counts, residuals, confidence intervals, convergence, survival/censoring and sensitivity to filters/lags.
10. **Export:** observations with stable row IDs, segment table, trace table, model results and a JSON provenance record.

## 7. Implementation roadmap

### Phase 0 — correctness and contracts (highest priority)

**Deliverables**

- tested per-localization-cycle final-valid extraction for all supported MINFLUX layouts;
- `TrajectoryIndex` that supports non-contiguous/interleaved `tid` rows and stable time-sorted views;
- explicit SI units and 2D/3D capability checks;
- classification of datasets as static localization groups, tracking trajectories, or user-selected/unknown;
- rename current `σ/√n` output to centroid precision and prevent automatic static-spread reporting as tracking precision.

**Acceptance tests**

- mixed final-iteration traces are retained correctly;
- interleaved rows produce the same `dt`, distance, duration and length as contiguous rows;
- raw row order and raw coordinates never change;
- every result records observation mode, selection and coordinate frame.

### Phase 1 — non-destructive trajectory QC and segmentation

**Deliverables**

- trace/segment table integrated with the existing Trace Viewer;
- robust nominal-interval estimator and absolute/relative gap proposals;
- editable start/end, split, merge, accept and exclude operations with undo;
- quality summaries for photons/efo, CFR, dcr, background, `dt`, speed and boundary proximity;
- optional geometric artefact scores inspired by Blob-B-Gone;
- CSV/NPZ result export plus JSON provenance.

**Acceptance tests**

- accepted/excluded intervals survive save/reload and remain reversible;
- threshold changes update proposals without overwriting manual decisions;
- exports reproduce exactly the displayed selection.

### Phase 2 — localization uncertainty and diffusion

**Deliverables**

- consecutive-difference estimator clearly labeled static-assumption/dynamic-bound;
- irregular-time displacement and lag-bin accumulator with pair counts and bounded memory;
- Brownian dynamic precision + diffusion fit with optional blur correction;
- time-averaged and ensemble-averaged MSD, turning angles and rolling displacement/diffusion;
- free, directed, anomalous and confined models, followed by hop/mixture models only when validation is solid;
- bootstrap/profile confidence intervals, residual plots and AIC/BIC where applicable.

**Acceptance tests**

- recover known `D` and `σx/y/z` from synthetic irregular-time Brownian data within predefined tolerances;
- stable estimates under row permutation, provided `tid`/time are unchanged;
- correct pair counts and no interpolation unless explicitly selected;
- warnings for insufficient lag coverage, correlated residuals, non-identifiability and likely censoring.

### Phase 3 — three application plugins

1. **NPC Transport:** port filter presets, editable pore clustering, cylinder/ring fit, channel registration with residuals, track-to-pore assignment, pose normalization, event classification and the existing playback idiom. Validate against the public example/model data.[13,32]
2. **Nanoparticle Surface Diffusion:** port centroid grouping, convex-hull/sphere metrics and surface views; use the Phase 2 estimators. Reproduce the local MATLAB output on a frozen reference case, then document intentional statistical differences.[14,33]
3. **Motor Steps:** projection-axis editor, change-point proposals, plateau residual precision, manual boundary correction, step/dwell distributions, BIC comparison and optional HMM adapter. Validate on the public kinesin archives.[5,6,11]

These can initially be in-tree plugins, but should depend only on the public dataset/trajectory/result interface so they can later move to external packages.

### Phase 4 — simulation, censoring and advanced integrations

- import SimuFLUX truth/estimate pairs and add automated accuracy, bias and track-survival reports;
- compare sequence/TCP metadata and CRLB predictions with measured residual/dynamic uncertainty;
- add ROI/event metadata compatible with event-triggered datasets and napari exports;
- expose a stable extension API for external estimators and learned state classifiers;
- consider synchronized multi-emitter/lifetime trajectories after the single-emitter result schema is mature.

## 8. Validation strategy

### Synthetic fixtures

- static anisotropic points with drift and correlated noise;
- Brownian motion with irregular times, known localization error and configurable blur;
- directed, anomalous, confined and hop trajectories;
- steps with known heights, dwell times, missed short states and axis error;
- gaps, duplicate timestamps, interleaved `tid`, truncated tracks and feedback loss;
- two channels with a known rigid/affine transform and registration noise.

### Public-data regression cases

- SimuFLUX for truth, bias and censoring;
- nanoparticle repository for grouping, segmentation and geometry;
- NPC repository/model data for channel transform, pore pose and assignment;
- kinesin archives for projection and step recovery;
- DNA-PAINT/origami datasets for static precision and drift, not dynamic precision.

### Scientific acceptance rules

- never report a dynamic precision without the assumed motion/blur model;
- never call `σ/√n` single-localization precision;
- never compare `D` values without showing time/lag range and dimensionality;
- never hide the number of rejected tracks or displacement pairs;
- preserve raw coordinates and `tid` throughout;
- include algorithm version, parameters and selection/transform digests in every export.

## 9. Recommended product decisions

| Priority | Decision | Rationale |
|---:|---|---|
| 1 | Build a validated `TrajectorySet`/segment/result contract | Every downstream workflow currently re-solves indexing, filtering and provenance. |
| 2 | Correct the tracking-precision vocabulary and defaults | Prevents a scientifically serious interpretation error in dynamic data. |
| 3 | Turn tail/gap operations into reversible proposals | Immediate improvement to the translated Trace Viewer and foundation for all projects. |
| 4 | Implement irregular-time Brownian `D + σ` with diagnostics | Broadest useful quantitative analysis across published MINFLUX tracking. |
| 5 | Port the nanoparticle workflow as the first application plugin | Source is local, bounded, and exercises grouping, geometry, segmentation, MSD and UI. |
| 6 | Port NPC transport next | High scientific value, but depends on a mature transform/frame/result model. |
| 7 | Add motor steps and advanced state models | Requires careful model validation and editable inference. |
| 8 | Add SimuFLUX regression and censoring reports before ML | Ground-truth validation is more valuable than adding opaque classifiers early. |

## 10. Sources

1. Schmidt R, Weihs T, Wurm CA, et al. “MINFLUX nanometer-scale 3D imaging and microsecond-range tracking on a common fluorescence microscope.” *Nature Communications* 12, 1478 (2021). [Article and methods](https://www.nature.com/articles/s41467-021-21652-z). DOI: 10.1038/s41467-021-21652-z.
2. Masullo LA, Steiner F, Zähringer J, et al. “Pulsed Interleaved MINFLUX.” *Nano Letters* 21, 840–846 (2021). [PubMed record](https://pubmed.ncbi.nlm.nih.gov/33336573/). DOI: 10.1021/acs.nanolett.0c04600.
3. Masullo laboratory. [pyflux acquisition/control](https://github.com/lumasullo/pyflux) and [p-minflux analysis](https://github.com/lumasullo/p-minflux), GitHub repositories.
4. Ostersehlt LM, Jans DC, Wittek A, et al. “DNA-PAINT MINFLUX nanoscopy.” *Nature Methods* 19, 1072–1075 (2022). [Article](https://www.nature.com/articles/s41592-022-01577-1) and [analysis repository](https://github.com/jkfindeisen/dna_paint_minflux_nanoscopy_2021). DOI: 10.1038/s41592-022-01577-1.
5. Wolff JO, Wirth M, et al. “MINFLUX dissects the unimpeded walking of kinesin-1.” *Science* 379, 1004–1010 (2023). [PubMed](https://pubmed.ncbi.nlm.nih.gov/36893244/), [2022 bioRxiv preprint](https://www.biorxiv.org/content/10.1101/2022.07.25.501426v1.full), and [code archive](https://zenodo.org/records/7442902). DOI: 10.1126/science.ade2650.
6. Deguchi T, Iwanski MK, Schentarra E-M, et al. “Direct observation of motor protein stepping in living cells using MINFLUX.” *Science* 379, 1010–1015 (2023). [PMC article](https://pmc.ncbi.nlm.nih.gov/articles/PMC7614483/) and [2022 preprint](https://doi.org/10.1101/2022.07.25.500391). DOI: 10.1126/science.ade2676.
7. Vogler BTL, Reina F, et al. “Blob-B-Gone: unsupervised trajectory artefact detection for single-particle tracking.” *Frontiers in Bioinformatics* (2023). [Article](https://www.frontiersin.org/journals/bioinformatics/articles/10.3389/fbinf.2023.1268899/full) and [repository](https://github.com/Eggeling-Lab-Microscope-Software/blob-B-gone). DOI: 10.3389/fbinf.2023.1268899.
8. Slenders E, Vicidomini G. “ISM-FLUX: MINFLUX with an array detector.” *Physical Review Research* 5, 023033 (2023). [Article](https://journals.aps.org/prresearch/abstract/10.1103/PhysRevResearch.5.023033). DOI: 10.1103/PhysRevResearch.5.023033.
9. Rickert C, et al. “4Pi MINFLUX arrangement maximizes spatio-temporal localization precision of fluorescence emitter.” *PNAS* 121, e2318870121 (2024). [PMC article](https://pmc.ncbi.nlm.nih.gov/articles/PMC10945813/). DOI: 10.1073/pnas.2318870121.
10. Cole F, Zähringer J, Bohlen J, et al. “Super-resolved FRET and co-tracking in pMINFLUX.” *Nature Photonics* 18, 478–484 (2024). [Article](https://www.nature.com/articles/s41566-024-01384-4). DOI: 10.1038/s41566-024-01384-4.
11. Wirth M, et al. “Uncovering kinesin dynamics in neurites with MINFLUX.” *Communications Biology* 7, 661 (2024). [Article](https://www.nature.com/articles/s42003-024-06358-4) and [code/data](https://zenodo.org/records/10718784). DOI: 10.1038/s42003-024-06358-4.
12. Schleske JM, et al. “MINFLUX reveals dynein stepping in live neurons.” *PNAS* 121, e2412241121 (2024). [PMC article](https://pmc.ncbi.nlm.nih.gov/articles/PMC11420169/). DOI: 10.1073/pnas.2412241121.
13. Sau M, et al. “Overlapping nuclear import and export paths unveiled by two-colour MINFLUX.” *Nature* 640, 821–827 (2025). [Article](https://www.nature.com/articles/s41586-025-08738-0). DOI: 10.1038/s41586-025-08738-0.
14. Fitzpatrick V, et al. “MINFLUX: µs and nm precision 3D tracking of dynamic lipid mobility on nanoparticles.” *Nanoscale* (2025). [PMC article](https://pmc.ncbi.nlm.nih.gov/articles/PMC12224226/) and [analysis repository](https://github.com/EMBL-ICLM/MINFLUX-particle-tracking-and-diffusion-analysis). DOI: 10.1039/D5NR00948K.
15. Reina F, et al. “Concurrent diffusion of nicotinic acetylcholine receptors and fluorescent cholesterol disclosed by two-colour sub-millisecond MINFLUX-based single-molecule tracking.” *Nature Communications* 16, 6336 (2025). [Article](https://www.nature.com/articles/s41467-025-61489-4) and [analysis repository](https://github.com/lucasSaavedra123/minflux_analysis). DOI: 10.1038/s41467-025-61489-4.
16. Yao L, Si D, et al. “Gradual labeling with fluorogenic probes: A general method for MINFLUX imaging and tracking.” *Science Advances* 11, eadv5971 (2025). [PMC article and methods](https://pmc.ncbi.nlm.nih.gov/articles/PMC12094232/). DOI: 10.1126/sciadv.adv5971.
17. “Parameter optimization for MINFLUX microscopy enabled single particle tracking.” *Communications Biology* (2025). [PMC article](https://pmc.ncbi.nlm.nih.gov/articles/PMC12618476/). DOI: 10.1038/s42003-025-09060-1.
18. Sharma A, Sau A, Dave S, et al. “Clustering within a single-component biomolecular condensate.” bioRxiv preprint (2025). [PMC preprint record](https://pmc.ncbi.nlm.nih.gov/articles/PMC12393318/). DOI: 10.1101/2025.08.18.670948.
19. Mazzocca M, et al. “Integrated MINFLUX tracking reveals two distinct chromatin dynamics classes across cell types.” *Nature Structural & Molecular Biology* (2026). [PMC article](https://pmc.ncbi.nlm.nih.gov/articles/PMC13419685/) and [analysis repository](https://github.com/ahansenlab/chromatin_dynamics). DOI: 10.1038/s41594-026-01807-6.
20. Alvelid J, et al. “Smart event-triggered MINFLUX microscopy to catch and follow rare events.” *Nature Communications* (2026). [Article](https://www.nature.com/articles/s41467-026-73176-z), [control software](https://github.com/jonatanalvelid/etMINFLUX), [analysis](https://github.com/jonatanalvelid/etMINFLUX-analysis-public), and [napari viewer](https://github.com/jonatanalvelid/napari-etminflux-data-viewer). DOI: 10.1038/s41467-026-73176-z.
21. “The molecular basis of force selectivity by PIEZO2.” *Nature* (2026), MINFLUX tracking methods and public analysis. [PMC article](https://pmc.ncbi.nlm.nih.gov/articles/PMC13149025/) and [repository](https://github.com/PatapoutianLab/MINFLUX_Localization_and_Tracking_Analysis). DOI: 10.1038/s41586-026-10182-7.
22. Marin Z, Ries J. “Evaluating MINFLUX experimental performance in silico.” *Nature Communications* 17, 246 (2026). [Article](https://www.nature.com/articles/s41467-025-66952-w), [2025 preprint](https://www.biorxiv.org/content/10.1101/2025.04.08.647786v1), and [SimuFLUX repository](https://github.com/ries-lab/SimuFLUX). DOI: 10.1038/s41467-025-66952-w.
23. “Characterizing dynamics of productive dynein stepping by MINFLUX.” *Nature Structural & Molecular Biology* (published 26 August 2026). [Article](https://www.nature.com/articles/s41594-026-01873-w) and [earlier bioRxiv version](https://www.biorxiv.org/content/10.1101/2024.07.16.603667v2.full). DOI: 10.1038/s41594-026-01873-w.
24. “Membrane bridges and nanodomain partitioning govern membrane protein targeting to lipid droplets.” *Nature Cell Biology* (2026). [Article](https://www.nature.com/articles/s41556-026-01963-3) and [analysis repository](https://github.com/amizrak/MINFLUX_analysis). DOI: 10.1038/s41556-026-01963-3.
25. Deguchi T, Ries J. “A variable phase plate for 3D MINFLUX nanoscopy.” *Light: Science & Applications* (2024). [Article](https://www.nature.com/articles/s41377-024-01487-1). DOI: 10.1038/s41377-024-01487-1.
26. Clowsley AH, Bokhobza AFE, Janicek R, et al. “Characterizing MINFLUX imaging performance with DNA origami.” bioRxiv preprint (2026). [Preprint](https://www.biorxiv.org/content/10.64898/2026.02.24.707670v1.full). DOI: 10.64898/2026.02.24.707670.
27. Deguchi T, et al. “Tracking Single Kinesin in Live Cells Using MINFLUX.” *Methods in Molecular Biology* (2025). [PubMed](https://pubmed.ncbi.nlm.nih.gov/39704940/). DOI: 10.1007/978-1-0716-4280-1_5.
28. Scientific Center for Optical and Electron Microscopy, ETH Zurich. [pyMINFLUX](https://github.com/bsse-scf/pyMINFLUX), Python reader/analyzer/viewer.
29. Ries laboratory. [SMAP](https://github.com/jries/SMAP), extensible MATLAB super-resolution microscopy analysis platform.
30. Michalet X, Berglund AJ. “Optimal diffusion coefficient estimation in single-particle tracking.” *Physical Review E* 85, 061916 (2012). [NIST record](https://www.nist.gov/publications/optimal-diffusion-coefficient-estimation-single-particle-tracking). DOI: 10.1103/PhysRevE.85.061916.
31. Vestergaard CL, Blainey PC, Flyvbjerg H. “Optimal estimation of diffusion coefficients from single-particle trajectories.” *Physical Review E* 89, 022726 (2014). [Article](https://journals.aps.org/pre/abstract/10.1103/PhysRevE.89.022726). DOI: 10.1103/PhysRevE.89.022726.
32. Huang Z, Sau M, Zhang H, et al. [MINFLUX NPC Tracking](https://github.com/npctat2021/MINFLUX_NPC_Tracking), publication analysis code.
33. EMBL Imaging Centre Light Microscopy. [MINFLUX particle tracking and diffusion analysis](https://github.com/EMBL-ICLM/MINFLUX-particle-tracking-and-diffusion-analysis), MATLAB analysis code.
