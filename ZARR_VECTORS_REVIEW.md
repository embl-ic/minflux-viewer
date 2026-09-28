# Zarr Vectors review for MINFLUX Viewer

**Review date:** 2026-09-22
**Repository:** `embl-ic/minflux-viewer`
**Status:** Research and architecture recommendation; no implementation is proposed by this document.

## Purpose

This report reviews the current MINFLUX Viewer Zarr v2, Zarr v3, and
OME-NGFF plans against the recent Zarr Vectors work:

- [Zarr Vectors specification repository](https://github.com/AllenInstitute/zarr_vectors)
- [Zarr Vectors Python implementation](https://github.com/AllenInstitute/zarr-vectors-py)
- [Python API documentation](https://zarr-vectors-py.readthedocs.io/en/latest/api/zarr_vectors.html)
- [image.sc announcement and discussion](https://forum.image.sc/t/zarr-based-vector-data-storage-points-meshes-skeletons-streamlines-tracks/120997)

It is intended to be self-contained enough for another agent to verify the
research, challenge the conclusions, and inspect the relevant repository code.

## Executive conclusion

Zarr Vectors is relevant and potentially valuable for MINFLUX Viewer, but it
should currently be treated as an experimental **spatial representation of
materialized localization geometry**, not as the authoritative MINFLUX raw-data
or application-project format.

The recommended division of responsibilities is:

| Payload | Recommended representation |
|---|---|
| Lossless application project, processing state, overlays, and embedded images | Existing MINFLUX Viewer Zarr v2 profile |
| Canonical all-iteration `mfx`, including invalid and non-finite rows | Existing columnar MINFLUX schema |
| Materialized last-valid localization point cloud | Zarr Vectors pilot |
| Density images, volumes, and conventional image pyramids | OME-NGFF |
| Future tracks, skeletons, meshes, and geometric analysis results | Potential Zarr Vectors nodes |

Zarr Vectors could replace or standardize the custom
`processed/current/position` portion of the earlier OME-Zarr experiment. It
would not replace canonical `raw/mfx`, provenance, filters, application state,
or an OME-compatible density image.

The recommended decision is therefore:

> Build a constrained, isolated proof of concept and benchmark it. Do not yet
> make Zarr Vectors the primary save/open format or a required runtime
> dependency.

## Response to the row-order argument

The objection to the earlier limitation is substantially correct.

Original row order is not, by itself, critical scientific information for a
MINFLUX point cloud. Zarr Vectors may partition and reorder vertices spatially
without loss, provided that:

1. every per-point attribute is reordered together with its vertex;
2. object/trace membership remains correct; and
3. a durable identifier is retained when a point must be mapped back to an
   authoritative source dataset.

The earlier wording conflated three different concepts:

- **Storage order:** where a point happens to be stored. This is not semantic
  and Zarr Vectors is free to change it.
- **Point identity:** how the same localization is recognized after reordering,
  filtering, or joining it to derived data.
- **Source mapping:** how a materialized last-valid point maps back to the
  corresponding row in canonical all-iteration `mfx`.

Only the latter two require a contract. Preserving physical row order is not
required.

### What `idx` currently means

Repository inspection shows two related `idx` behaviours:

1. `ds.attr["idx"]` is normally regenerated as `1..N` after materialization.
   See `minflux_viewer/core/loader.py:2479-2481`. In this form it is a display
   or current-view ordinal. If points are spatially reordered and the dataset
   is reloaded, this value can be regenerated in the new order; it is not by
   itself a durable source identity.

2. `mfx_get(ds, "idx", itr=..., vld_only=...)`, when `mfx_raw` exists, returns
   the one-based flattened raw-row ordinals selected by the requested mask.
   See `minflux_viewer/core/loader.py:1558-1564`. For the last-valid selection,
   this is effectively `source_row_id + 1` and can reconstruct the source order
   by sorting.

This supports the argument that source order is at least partially—and for the
normal source-backed last-valid view, effectively exactly—recoverable through
the current `idx` machinery.

There are two qualifications:

- `idx` is excluded from canonical raw and snapshot exports as a recomputed
  attribute (`minflux_viewer/core/save.py:71-75`), so a Zarr Vectors writer must
  deliberately materialize it rather than assume it will already be exported.
- An ordinal is only globally meaningful when paired with its source dataset
  identity or canonical raw fingerprint. Two channels, two acquisitions, or a
  derived/combined dataset can all contain the same numeric `idx`.

### The previous experiment already handled this correctly

The `ome-zarr-export-experiment` branch defines `_source_row_ids` by applying
the raw last-valid mask and storing `np.flatnonzero(mask)` as a zero-based
`uint64` array. See:

```text
git show ome-zarr-export-experiment:minflux_viewer/core/ome_zarr.py
lines 375-393
```

It writes that array alongside processed positions and explicitly references
the canonical raw measurement rows. That is the right idea for a Zarr Vectors
pilot.

### Revised limitation

The earlier “Zarr Vectors does not preserve canonical row order” limitation
should be replaced with:

> Spatial reordering is acceptable and is one of Zarr Vectors' advantages.
> A Zarr Vectors localization must nevertheless carry a durable source-scoped
> identifier when it needs to map back to canonical raw data. The existing
> raw-selection form of `idx` already supplies the row ordinal; the export
> schema should freeze it as `source_row_id` (or define an immutable exported
> `idx`) and pair it with a dataset ID/raw fingerprint.

Using the explicit name `source_row_id` remains preferable in an interchange
schema because it distinguishes the durable source ordinal from the current
view's regenerated, one-based `idx`. This is a clarity and provenance issue,
not a requirement to preserve storage order.

## Current repository state

### Adopted application format

The main branch uses a custom, self-contained Zarr v2 profile:

- Format IDs are `org.minflux-viewer.dataset` and
  `org.minflux-viewer.project`.
- Canonical `mfx` columns, MBM/search data, metadata, derived values, filters,
  ROIs, project membership, overlays, and embedded images are preserved.
- Canonical raw content is fingerprinted with SHA-256.
- Writes and viewer-only updates are transactional.
- Directory and sealed ZIP forms use the same logical schema.

Primary implementation:

```text
minflux_viewer/core/minflux_zarr.py
docs/data-model.md
file_format_zarr_v2_schema.md
```

The full-scale reference contains 192,334 materialized localizations and
20,134,823 all-iteration rows across 26 canonical columns. Recorded results are
approximately 14.0 seconds to write, 9.6 seconds to read, and 199 MiB output
from a 318 MiB MSR, with all raw columns round-tripping bit-identically. See
`file_format_zarr_v2_schema.md:220-235`.

### Zarr v3 dependency boundary

The application currently declares:

```toml
zarr = ">=2.17,<3.0"
```

See `pyproject.toml:30`. Zarr Vectors requires Zarr 3 and does not support Zarr
v2 stores, according to its
[installation documentation](https://zarr-vectors-py.readthedocs.io/en/latest/getting_started/installation.html).

The private vendor/MSR Zarr v2 parser can remain independent, but using
`zarr-vectors-py` in the main process still requires the planned Zarr-Python 3
runtime and build migration.

### OME-NGFF / Zarr v3 experiment

The branch `ome-zarr-export-experiment` contains the previous writer. It:

- emits an OME-NGFF 0.5 density pyramid at the root;
- writes canonical raw arrays and application-specific data under `minflux/`;
- writes processed positions, attributes, and `source_row_id`;
- manually emits Zarr v3 metadata so it does not depend on the application's
  pinned Zarr v2 package.

Commit `74f1e1d` removed this path from main on 2026-08-27. The practical
reason was not that Zarr v3 was unsuitable: the feature was exposed as a save
format but had no reader or registered `.ome.zarr` open path, so the result was
handed to the Zarr v2 reader and failed. It also had a Windows compound-suffix
dialog bug.

`file_format_zarr_v2_schema.md:3-5` still describes this exporter as an active
forward-looking format even though it was removed from main. That documentation
is stale and should be corrected independently of any Zarr Vectors work.

## What Zarr Vectors provides

Zarr Vectors is a draft Zarr v3 convention for large N-dimensional sparse
geometry. It covers point clouds, polylines/streamlines, skeletons, graphs, and
meshes. The specification emphasizes:

- spatial chunking;
- bounding-box and selective attribute reads;
- per-vertex, per-object, and per-group attributes;
- object identity across chunks;
- multiresolution geometry;
- cloud/object-store access; and
- independent writes to different spatial chunks.

Useful primary references:

- [Introduction and goals](https://alleninstitute.github.io/zarr_vectors/01-introduction.html)
- [Data model](https://alleninstitute.github.io/zarr_vectors/04-data-model.html)
- [Store structure](https://alleninstitute.github.io/zarr_vectors/05-zarr-store-structure.html)
- [Spatial indexing](https://alleninstitute.github.io/zarr_vectors/06-spatial-indexing.html)
- [Metadata](https://alleninstitute.github.io/zarr_vectors/08-metadata.html)
- [Geometry types](https://alleninstitute.github.io/zarr_vectors/12-geometry-types.html)

The Python package exposes `Schema`, `Layout`, `Dataset`, selection/query
objects, and `create`/`open` entry points. Its quickstart writes point positions
with attributes and object IDs, builds a pyramid, and reads only a bounding box:

- [Package README and quickstart](https://github.com/AllenInstitute/zarr-vectors-py)
- [Top-level API](https://zarr-vectors-py.readthedocs.io/en/latest/api/zarr_vectors.html)

## Benefits for MINFLUX Viewer

### 1. Natural representation of the displayed localization layer

The materialized last-valid localization view is genuinely a 2-D or 3-D point
cloud. Zarr Vectors can represent its coordinates as vertices and its numeric
scientific values as row-aligned vertex attributes.

Suggested initial mapping:

| MINFLUX concept | Zarr Vectors representation |
|---|---|
| Registered/display XYZ position | Level-0 vertices |
| Raw-row ordinal | `source_row_id` vertex attribute |
| Trace ID | Original `tid` vertex attribute |
| Dense ZV object ID, if trace lookup is useful | Remapped from `tid` with the original value retained |
| `tim`, `itr`, `vld`, `efo`, `cfr`, `dcr_*`, etc. | Vertex attributes |
| Channel/dataset identity | Separate vector node/store plus project reference |
| Filter result | `ftr` vertex attribute or application state, depending export intent |
| Track geometry | Later polyline/streamline profile |
| Mesh/skeleton analysis results | Matching ZV geometry type |

The canonical raw table remains authoritative. The vector layer is a spatial
view over a selected/materialized subset.

### 2. Viewport and out-of-core access

Zarr Vectors computes an N-dimensional spatial chunk from position and resolves
a bounding-box query to the intersecting chunks. This can support future
viewport loading, remote access, and larger-than-memory datasets.

This benefit is not automatic. The current application materializes complete
NumPy datasets and its renderer expects complete point arrays. Real gains
require a deferred/bounding-box-aware loader and renderer.

### 3. Progressive visualization

Optional coarser levels can support overview rendering followed by progressive
loading of full-resolution points. These levels must be treated as derived
caches, because point-cloud coarsening can replace points with centroids or
sparsify objects. Level 0 should remain the only lossless vector level.

### 4. Geometry beyond points

The same family could later describe traces, polylines, skeletons, surfaces,
and mesh-based analysis outputs more naturally than arbitrary application JSON.
That is an additional benefit, not a reason to force current ROI or project
state into the format prematurely.

## OME-Zarr relationship

Zarr Vectors complements OME-Zarr but is not an OME-Zarr localization-table
standard.

It borrows NGFF-style `multiscales`, axes, units, and coordinate transforms, so
metadata registries and geometry-aware viewers can register vector and image
data in the same physical coordinate system. It does not use the dense image
array model expected by normal OME-Zarr image viewers.

The Zarr Vectors documentation states that OME metadata tooling may discover
the store, while ordinary image viewers may reject or meaninglessly interpret
the vector arrays:

- [Zarr Vectors and OME-Zarr comparison](https://zarr-vectors-py.readthedocs.io/en/latest/spec/comparisons/ome_zarr.html)

Recommended packaging model:

```text
experiment/
├── density.ome.zarr/             # conventional OME-NGFF image pyramid
├── localizations.zarrvectors/    # Zarr Vectors point geometry
└── project/application manifest  # MINFLUX state and references
```

An OME collection/RFC node or application manifest may relate the stores. The
whole MINFLUX project should not be labelled `.ome.zarr` merely because it
contains NGFF-compatible vector metadata.

## Remaining limitations and risks

### 1. Draft status and adoption

The specification repository calls itself a draft released to explore the
design and spark discussion. It is not an adopted Zarr-core or OME-NGFF
standard:

- [Specification statement of support](https://github.com/AllenInstitute/zarr_vectors)

### 2. Package and specification churn

PyPI lists only `zarr-vectors` 0.2.0, released 2026-05-17 and classified Alpha:

- [PyPI project and release history](https://pypi.org/project/zarr-vectors/)

The repository and live specification describe subsequent changes such as
fragment-index revisions, renamed metadata fields, native Zarr v3 sharding,
and v0.6-v0.8 capability/layout changes. A prototype must pin an exact package
version or Git commit and record the Zarr Vectors schema version it writes.

### 3. Non-finite raw measurements

Zarr Vectors assigns a spatial chunk using a floor operation on position.
Non-finite coordinates therefore cannot participate in its spatial index. The
current MINFLUX loader deliberately retains all raw iteration rows in
`mfx_raw`, while the materialized valid view filters non-finite positions.

This reinforces the two-layer design:

- canonical raw table for every measurement row;
- Zarr Vectors for finite materialized geometry.

### 4. Source-scoped identity, not storage order

Spatial reordering itself is not a limitation. The actual requirement is to
retain a durable mapping when vector points refer back to canonical raw rows.
The existing raw-selection `idx` semantics and experimental `source_row_id`
already provide the row ordinal. A source dataset ID or raw fingerprint is
also needed where multiple sources can coexist.

### 5. Trace/object mapping

Zarr Vectors object IDs are dense integer slots. MINFLUX `tid` values should
remain scientific attributes and, if used as ZV objects, be remapped through an
explicit table rather than assumed to be dense or globally unique.

### 6. Multiresolution scientific meaning

ZV's generic point-cloud coarsening is appropriate for visualization but may
not preserve localization statistics, trace structure, or density semantics.
Coarse levels must carry their derivation method and be disposable.

### 7. Dependency and packaging migration

The main application still uses Zarr-Python 2. A production integration must
validate Zarr-Python 3, codecs, the Windows/PyInstaller build, existing Zarr v2
stores, and the private MSR shim. Until then, the pilot should run in an
isolated environment or helper process.

## Why Zarr Vectors was missed in the previous research

It existed before the earlier report and therefore was genuinely overlooked.

Timeline:

- 2026-05-16: public image.sc announcement describing a prototype and asking
  for community/NGFF guidance.
- 2026-05-17: `zarr-vectors` 0.2.0 published to PyPI.
- May-July 2026: substantial implementation development continued.
- 2026-08-23: MINFLUX Viewer format research recorded in
  `file_format_research_result.md`.

The likely research gap was scope and vocabulary:

1. The prior investigation focused on Zarr core, OME-NGFF, general
   table/dataframe interchange, packaging, Parquet/Arrow, and lossless project
   preservation.
2. Its source list (`file_format_research_result.md:580-591`) did not include
   vector geometry, point-cloud Zarr extensions, Neuroglancer, or the Allen
   Institute projects.
3. The new work was advertised through neuroscience/vector channels, while the
   spec repository's short description referred to “a specification for
   skeletons.” A table-oriented search could easily miss it.
4. It is a third-party draft convention, not functionality supplied by Zarr v3
   core or an adopted OME-NGFF specification.

The previous statement remains accurate in its narrow sense:

> Zarr v3 core defines N-dimensional typed arrays, not mature general
> table/dataframe semantics.

See `file_format_research_result.md:380-385`. The missing qualification was
that an early, domain-specific spatial-vector convention had appeared outside
Zarr core and OME-NGFF.

Had Zarr Vectors been found, it should have been recorded as an emerging
candidate. Its draft/alpha status and the absence of a complete MINFLUX reader
mean it would not have changed the decisions to keep the tested Zarr v2 project
format or remove the write-only OME exporter from main.

## Recommended implementation plan

### Phase 0: documentation correction

1. Update `file_format_zarr_v2_schema.md:3-5` to say the OME-NGFF/Zarr v3 work
   is preserved on `ome-zarr-export-experiment`, not active on main.
2. Amend the earlier research conclusion with Zarr Vectors as an emerging
   point/vector convention.
3. Replace any “row order must be preserved” language with the source-scoped
   identity requirement described above.

### Phase 1: isolated proof of concept

Use an exact upstream version or commit in a separate environment. Do not
change the main application's Zarr dependency for the first benchmark.

For one dataset, write:

- finite last-valid XYZ positions at level 0;
- `source_row_id`, derived from the raw mask, as `uint64`;
- source dataset ID and canonical raw SHA-256 fingerprint;
- original `tid` and a documented dense-object-ID remapping if object lookup is
  tested;
- representative attributes including `tim`, `itr`, `efo`, `cfr`, and both DCR
  channels;
- axes, units, bounds, transform provenance, writer version, package commit,
  and ZV schema version;
- no coarse level initially.

The existing `_source_row_ids` implementation on the experiment branch can be
reused conceptually. The writer should not rely on ordinary `ds.attr["idx"]`
because that is a regenerated current-view ordinal.

### Phase 2: correctness tests

Verify:

1. Vertex and attribute values are exact at level 0, including dtype and NaN
   policy.
2. Sorting by `source_row_id` reconstructs the source last-valid selection.
3. `(source fingerprint, source_row_id)` is unique for every exported point.
4. Duplicate/non-dense `tid` values survive and object remapping is reversible.
5. Filtered, cropped, simulated, derived, and multi-channel datasets have
   explicit identity semantics.
6. Bounding-box reads return the same set of source IDs as an in-memory spatial
   predicate, independent of order.
7. Empty spatial regions and negative coordinates behave correctly.
8. The store passes the upstream validator for the pinned schema revision.

### Phase 3: benchmarks

Measure on the existing full-scale reference:

- level-0 write time;
- full read time;
- representative 2-D and 3-D viewport query latency;
- peak writer and reader memory;
- number of physical objects/chunks;
- compressed size;
- overhead when the canonical raw table and vector layer coexist;
- local directory, ZIP/package, and optionally object-store behaviour;
- float64 coordinate support and exactness;
- optional pyramid generation time and error/aggregation semantics.

Compare against the existing Zarr v2 baseline, but do not compare the small
materialized vector layer directly with the 20-million-row raw store without
stating that they represent different payloads.

### Phase 4: adoption gates

Do not expose Zarr Vectors as a supported save/open format until:

- the chosen schema revision is tagged and documented;
- the package/API version and schema version are aligned;
- a reader exists in MINFLUX Viewer;
- source identity and all supported dataset transformations round-trip;
- Zarr-Python 3 and PyInstaller builds pass on supported platforms;
- ordinary Zarr v2 application stores and vendor/MSR data remain readable;
- error messages distinguish Zarr v2, Zarr v3, OME-Zarr, and Zarr Vectors;
- the UI identifies a one-way operation as **Export**, not **Save As**, unless
  the result can be reopened faithfully.

## Questions for the verifying agent

The next reviewer should explicitly confirm or challenge:

1. Whether the current Zarr Vectors release preserves float64 positions and
   arbitrary numeric vertex-attribute dtypes without coercion.
2. Whether the current package release actually implements the live spec's
   v0.6-v0.8 layout, or whether GitHub main and PyPI 0.2.0 are incompatible.
3. Whether a ZV group can be cleanly nested as a node inside an OME-NGFF 0.5
   collection, or should remain a sibling store referenced by an RFC/application
   manifest.
4. Whether the current upstream API can stream/add tens of millions of points
   without materializing all chunk partitions in memory.
5. Whether `source_row_id` should be zero- or one-based. The earlier experiment
   used zero-based values; user-facing `idx` is one-based.
6. Whether the canonical source identity should be the existing raw SHA-256,
   dataset UUID, DID, or a tuple of these.
7. Whether `tid`-as-object provides enough practical value to justify object
   manifests for the first point-cloud pilot.
8. Whether time should remain a vertex attribute or participate in a separate
   temporal/chunk dimension.
9. Whether ZV multiresolution adds practical value beyond the existing density
   pyramid for the current desktop renderer.

## Final recommendation

Zarr Vectors fills a real gap in the earlier investigation: it provides an
emerging spatial convention for the point-cloud aspect of MINFLUX data. The
best immediate use is a level-0, spatially indexed representation of finite
materialized localizations, with all row-aligned attributes and explicit
source-scoped IDs.

The user correction about row order should be accepted. Row order itself is
not an invariant that needs to constrain the vector layout. Existing `idx`
semantics already recover raw-row ordinals for source-backed selections, and
the old experiment already materialized the same information as
`source_row_id`. The architectural invariant is referential identity, not
physical order.

The present draft status, schema churn, Zarr 3 dependency, incomplete OME image
compatibility, and lack of an integrated reader still argue for a pilot rather
than adoption as the main format.
