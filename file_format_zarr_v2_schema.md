# MINFLUX Viewer Zarr v2 dataset format

**Status:** adopted as the MINFLUX Viewer application format. The separate
OME-NGFF / Zarr v3 export (`core/ome_zarr.py`) is the forward-looking,
interoperability-facing format and evolves independently of this one.  
**Schema:** `1.0.0`  
**Physical storage:** Zarr v2 directory store  
**Format ids:** `org.minflux-viewer.dataset` (single dataset) and
`org.minflux-viewer.project` (multi-dataset acquisition/overlay)

This document supersedes the ZIP64/`.mfv` implementation direction in
`file_format_plan.md`. There is no previously saved MINFLUX Viewer project data to
migrate, so the former unmarked flat-column Zarr export is not a compatibility
target. The first supported application format is a marked, self-contained Zarr v2
store.

## Goals

- Preserve canonical raw localizations, including every iteration and both DCR
  columns.
- Preserve MBM points, bead names/selection, the MBM search grid and their native
  metadata without retaining a duplicate `mfxdta.zarr` store.
- Keep raw acquisition facts, derived results and viewer/process state separate.
- Reopen a saved dataset without the original `.msr` or a metadata sidecar.
- Preserve a sequential multi-channel acquisition as one reopenable overlay,
  including channel order, LUTs and per-channel transforms.
- Preserve viewer ROI geometry as well as its cached selection masks, and keep
  DID-linked image series inside the dataset that owns them.
- Remain readable with ordinary Zarr v2 tooling: arrays are ordinary NumPy/Zarr
  arrays and attributes are JSON-compatible; no pickle or object arrays are used.

## Identity and versioning

The root `.zattrs` contains:

```json
{
  "_minflux_viewer_format": "org.minflux-viewer.dataset",
  "_minflux_viewer_schema_version": "1.0.0",
  "_minflux_viewer_created": "<ISO-8601 timestamp>",
  "_minflux_viewer_app_version": "<application version>",
  "_minflux_viewer_raw_sha256": "<canonical raw fingerprint>"
}
```

The schema version and physical Zarr version are independent. Unknown schema major
versions fail explicitly. Unmarked flat-column and Imspector stores are not guessed
as application save files.

A project root uses `org.minflux-viewer.project` with its own 1.x logical schema.
Each child below `datasets/` is still a complete marked
`org.minflux-viewer.dataset` store, so dataset boundaries remain explicit.
The fingerprint is stored on dataset roots, not the project root; pre-fingerprint
stores are compared by loading and hashing their canonical raw components once.

## Store tree

```text
dataset.zarr/
├── .zgroup / .zattrs
├── mfx/
│   ├── .zattrs                 native MFX attrs + component/layout metadata
│   ├── loc_x, loc_y, loc_z     metres; canonical raw coordinates
│   ├── dcr_0, dcr_1            both native DCR channels
│   ├── itr, vld, tid, tim, ... all other canonical raw columns
│   └── ...
├── grd/
│   ├── mbm/
│   │   ├── .zattrs             native MBM attrs and `used`
│   │   └── points              structured native MBM point array
│   └── search_0/
│       ├── .zattrs             native search-grid attrs
│       └── points              structured native search point array
└── viewer/
    ├── dataset/                file/calibration/channel description
    ├── metadata/               acquisition/app provenance
    ├── state/                  filters, transforms and persistent view state
    ├── rois/                   portable viewer ROI geometry (when present)
    ├── images/                 embedded-image manifest (when present)
    ├── derived/                computed arrays
    └── derived_last/           last-valid derived arrays, when required
```

Native acquisition root attributes such as `rois` and `version` remain directly on
the root. Native MFX attributes such as `did`, `acquisition_date`, `measurement` and
`scan_range` remain directly on `mfx`. This makes the relationship with the vendor
store visible while reserved `_minflux_viewer_*` keys identify the extension.

## Data rules

### Raw MFX

`mfx` uses one Zarr array per field. Vector fields are split into columns:

- `loc` → `loc_x`, `loc_y`, `loc_z`
- `dcr` → `dcr_0`, `dcr_1`
- other supported spatial vector fields use the same `_x/_y/_z` convention

Columns are recomposed into a canonical flat m2410 structured array during load and
then pass through the normal loader. Raw coordinates remain in metres and Z scaling factor is a
view/calibration value, so Z is not corrected twice.

### MBM and search grid

The structured point arrays remain structured arrays because they are small and this
retains the native relationship among `gri`, `xyz`, `tim` and `str`. Bead naming
(`points_by_gri`) and the used-bead list remain attached to the MBM component.

### Processing state

`viewer/metadata` and `viewer/state` are recursive JSON-compatible payloads. NumPy
arrays are stored below their payload's `arrays/` group and referenced from the JSON
tree. `viewer/derived` uses named ordinary arrays. On reload, a derived array aligned
to materialized localization rows is restored both to `dataset.derived` and the
materialized attribute table.

Active filters have one canonical persistent location:
`viewer/state` → `filter_specs`. Each ordered specification records
`attribute`, aggregation `mode`, iteration selector `itr`, numeric `lo`/`hi`,
and `lo_inc`/`hi_inc`. The accompanying `filter_mask` is a cached result stored
as a payload array, not the filter definition. On load the mask is available
immediately, the specifications are re-evaluated after derived attributes are
ready, and opening the Filter dialog reconstructs the enabled rows with their
saved modes, iterations, bounds and inclusivity. Keeping this in `viewer/state`
instead of duplicating it under another node follows the one-fact/one-location
rule and lets stores written before dialog reconstruction was fixed reopen
without migration.

### Multi-dataset project, images and ROIs

An overlay/acquisition is stored as:

```text
acquisition.zarr/
├── viewer/
│   ├── project/                 member/order/overlay manifest
│   ├── rois/                    geometry + owning dataset id
│   └── images/                  image path/DID/owner manifest
├── datasets/
│   ├── d000000/                 complete dataset store
│   │   └── images/*.tif         DID-linked calibrated OME-TIFFs
│   ├── d000001/
│   │   └── images/*.tif
│   └── d000002/
│       └── images/*.tif
└── images/unassigned/*.tif      selected images with no source DID
```

OME-TIFF is deliberately used for embedded images: it retains the existing image
calibration/metadata round trip and is a normal file inside the directory store.
Zarr places no restriction on such additional payload files. The project manifest,
not folder-name guessing, is authoritative for association. Images carrying the
vendor `source_did` go below their owning dataset; an image with no DID remains
explicitly unassigned rather than being attached to an arbitrary channel.

Acquisition ROIs from the native store remain native root attributes on their
dataset. Viewer-created ROIs are a different object and live under `viewer/rois`.
Their cached masks remain derived/state arrays; their geometry record is stored once
and is reattached to the session ROI store with the new dataset index on load.

## Write behavior

- Zarr saves are raw-canonical plus separate processing state; a baked-only snapshot
  is rejected.
- No sibling `_metadata.json` is written.
- The former `_mfx.zarr + _mbm.zarr + _mfxdta.zarr` MSR export is retired. A
  multi-dataset MSR export writes one `<msr-name>.zarr` project; a single selected
  dataset remains a single dataset store.
- File > Save As automatically expands the active dataset to every member of its
  overlay. Its live transform, order and LUT are saved per channel. DID-linked MSR
  images are embedded automatically.
- A complete replacement is transactional at directory level: write and validate a
  same-filesystem temporary store, move an existing target aside, replace it, then
  remove the backup. A failed replacement restores the previous store.
- A processing-only overwrite first recomputes SHA-256 fingerprints from the actual
  stored and in-memory canonical MFX/MBM/search components and native metadata. The
  stored digest is also checked for external raw-chunk changes. The update is refused
  if the store type, dataset count, or any raw fingerprint differs. Matching is by DID
  plus fingerprint, so changing overlay order does not swap channel ownership.
- After that verification, only the root/child `viewer/` directories are staged and
  transactionally replaced. Raw Zarr chunks and embedded OME-TIFF files remain in
  place. Existing image manifests and unknown optional viewer extensions are carried
  forward; adding/replacing an embedded image requires a complete replacement.
- Arrays use Blosc/LZ4 with byte shuffle and approximately 1 MiB row chunks.

## Current UI

Use **File > Save As > Zarr (.zarr v2) format**. The ordinary Save/Export dialog also
labels the same backend as **MINFLUX Viewer Zarr v2 (.zarr)** and pins it to raw
canonical content because processing is stored separately inside the package.
When the target already exists, the application offers **Update processing only**,
**Replace complete store**, or **Cancel**. Processing-only is the default and fails
safely rather than modifying raw data when the fingerprints do not match.
The Zarr-specific non-native path chooser treats an existing `.zarr` directory as
the package being saved: pressing **Save** returns that path instead of navigating
into the directory, after which the three-way overwrite prompt is shown.

## Acceptance coverage implemented

`tests/test_minflux_zarr.py` verifies:

- format/schema markers and group layout;
- exact raw structured-field round trip, including both non-complementary DCR
  columns;
- MBM/search points and native attributes;
- Z scaling factor, filter mask, transform, custom metadata and derived arrays;
- exact filter specifications and Filter-dialog row reconstruction on reopen;
- multi-channel project membership, transforms, embedded-image association and ROI
  geometry restoration;
- replacement cleanup/transaction behavior;
- processing-only overwrite without changing raw chunk timestamps, raw-mismatch
  refusal, unknown-viewer-extension preservation, and reordered project channels;
- accepting an existing `.zarr` directory from the save chooser without entering it;
- explicit rejection of unmarked legacy stores and baked snapshot mode;
- the File-menu entry targets the new backend.

The related save/MSR integration suites verify that the old three-store MSR export is
retired and the enriched store reopens through `core.loader.load_zarr`.

## Measured at full scale

Reference acquisition `3_BD+A_MRED_75pM_MINFLUX_3D.msr` (318 MiB `.msr`,
192,334 localizations, **20,134,823 raw iteration rows**, 26 canonical columns),
on a Windows workstation:

| operation | time | note |
|---|---|---|
| parse + load the `.msr` | 9.8 s | for comparison |
| **write the store** | **14.0 s** | 1.4 M raw rows/s |
| **read the store back** | **9.6 s** | on par with reading the `.msr` |
| **processing-only update** | **8.1 s** | 1.7x faster than a full write |
| output size | **199 MiB** | 1.6x smaller than the `.msr` |

All 26 raw columns round-trip bit-identically at this scale, and a
processing-only update leaves every raw chunk file untouched (verified by
mtime).

On a small single-channel acquisition (15,886 localizations / 1.39 M raw rows)
the store is 15.8 MiB against a 72.5 MiB `.msr` — 4.6x smaller. The compression
ratio is data-dependent; native dtypes (`float16` cfr/dcr, `uint32` counts) are
preserved rather than widened, which is where most of the saving comes from.

### Why the processing-only update costs what it does

It is dominated by two integrity hashes, not by writing: the in-memory dataset's
canonical digest (~5.7 s) and the store's own (~2.4 s). The store side is hashed
by reading its arrays directly rather than by loading a full dataset — doing the
latter cost 16.5 s and made the "fast" path *slower than replacing the whole
store*, defeating its purpose. `_store_raw_fingerprint` and
`_dataset_raw_fingerprint` therefore share one `_raw_parts`/`_raw_digest`
routine so they cannot drift; `tests/test_minflux_zarr.py` pins their equality
and pins that the direct read still detects an externally rewritten chunk.

Of the remaining in-memory cost, ~3 s is `dataset_to_mfx_array` building a
20 M-row structured array only for `flatten_mfx_array` to split it back into
columns. Removing that round trip would need a canonical-column builder that is
provably identical to the current pair; not attempted.

## Single-file packaging (`.zarr.zip`)

**Both forms are offered in File ▸ Save As** — *Zarr (.zarr v2) format* writes
the directory store, *Zarr (.zarr.zip v2) single file* writes the sealed package
(`save_processed(fmt="zarr_zip")` → `write_minflux_zarr_package`, which builds
the store in a temporary directory, validates it, packs it and renames into
place). The content is identical: raw canonical data, MBM/search, viewer state,
ROIs, overlay channels and embedded images all live inside, and neither writes a
metadata sidecar. The **only** difference is that the package cannot take a
processing-only update, so saving over one always rewrites it; the overwrite
prompt (Update / Replace / Cancel) is therefore offered for `.zarr` alone.

`pack_minflux_zarr(store)` seals a directory store into one `.zarr.zip`;
`unpack_minflux_zarr(package)` reverses it; and `load_minflux_zarr` opens a
sealed package **directly**, without unpacking. Measured on the 199 MiB / 2,004
member reference store: pack 1.1 s, open 10.0 s (vs 9.6 s for the directory
form), unpack 3.2 s, byte-identical either way, no size penalty.

Members are stored with `ZIP_STORED`, because the chunks are already
Blosc-compressed and deflating them again costs time for nothing. ZIP64 is
enabled, lifting the 4 GiB and 65,535-member limits; the reference store needs
2,004 members, so the ceiling is remote.

**The zip is a sealed distribution copy, not the working format.** A
processing-only update replaces members of `viewer/`, and a zip cannot replace a
member — writing one again *appends a second entry with the same name*. Readers
then disagree (`zipfile` and Zarr take the last, some archive tools take the
first) and the file grows on every save. So the directory store stays the
editable form, and `unpack_minflux_zarr` refuses a package containing duplicate
members rather than silently picking one.

This mirrors how Zarr v3 `ZipStore` is used, so the two formats package the same
way.

## Processing state beside untouched raw data

The self-contained store is one of two routes. The other is the **metadata
sidecar** (`<stem>_metadata.json`, `METADATA_JSON_MARKER`), which carries the
same application state *next to* a raw file this application never rewrites —
`.mat`, `.npy`, `.csv`, `.json`, `.npz` and `.msr` alike. It is deliberately
application-specific rather than format-specific: one schema, one reader, no
per-format variants. It records acquisition provenance, gating, calibration
(Z scaling factor + provenance), the overlay transform, `filter_specs`, and **ROI records**
in the same JSON shape the Zarr v2 store and the OME-Zarr v3 package use, so a
ROI means the same thing wherever it is written. `apply_metadata_recipe`
restores all of it, putting ROIs on `metadata["minflux_viewer_roi_records"]` —
the same key the Zarr loader uses.

## Images

Images live **inside the store** as OME-TIFF files, written by
`_export_embedded_images`: `datasets/<id>/images/<name>.tif` for a series whose
`source_did` matches that dataset, `images/unassigned/<name>.tif` for the rest
(confocal channels, overviews — these carry no DID). `viewer/images` holds the
manifest. Zarr ignores keys that are not `.zgroup`/`.zarray`/chunks, so a plain
`zarr.open()` on the store is unaffected; verified on a 33-image store where the
TIFFs are 31.6 MiB of 65.9 MiB.

**The store is self-sufficient and must never consult the source `.msr`.** A
`.msr` may have moved, been renamed, or never travelled with the store, so
borrowing from it makes an incomplete store look complete. Two rules keep that
honest:

- **File › Save As embeds every image series of the source `.msr`**, not only
  the DID-linked ones. It used to filter on `source_did`, which dropped 10 of 11
  series on a single-run file and 30 of 33 on a three-channel one.
- **A dataset restored from a store reports what it has**;
  `view_dataset_image_series` returns early for it rather than falling through
  to `metadata["msr_source_path"]`. That fallback is retained only for a dataset
  imported directly from a `.msr`, which legitimately has no embedded copies.

⚠ A sealed `.zarr.zip` stores its images as archive members, so their recorded
`absolute_path` does not exist on disk. `minflux_zarr.materialize_image(record)`
extracts the member once to a temp file and returns that path; every consumer
must go through it rather than testing `Path(absolute_path).is_file()`. Skipping
it is what made a package silently fall back to the `.msr` and *appear* to hold
every image.

## Remaining work

1. Add save progress/cancellation for multi-hundred-megabyte datasets; the current UI
   call is synchronous, and a 14 s write blocks the UI.
2. Add a project-level image browser for unassigned image series. Dataset-linked
   images already reopen from Dataset Manager; unassigned images remain accessible
   as files inside `images/unassigned`.

## Application-specific formats

Deferred — see `BACKLOG.md` ▸ Nice to have. The supported set is this
application's own Zarr v2 store plus the MINFLUX default formats
(`.msr`, `.npy`, `.mat`, `.json`), with the generic table (`.csv`) kept as the
spreadsheet interchange path.
