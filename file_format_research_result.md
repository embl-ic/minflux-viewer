# Processed MINFLUX project format research result

**Status:** implementation handoff / proposed architecture  
**Research date:** 2026-08-23  
**Input:** [`file_write_format.md`](file_write_format.md) plus independent review of the
repository, the MSR/MFXDTA reader, and current upstream specifications.

## 1. Executive decision

Use a **custom MINFLUX Viewer project package with a standard ZIP64 envelope**.
The proposed portable extension is `.mfxp` (provisional). The package should:

1. Preserve every original `.msr` source **byte-for-byte** as an immutable payload.
2. Never rewrite or augment the vendor's embedded Zarr store.
3. Store MINFLUX Viewer-owned processed data, metadata, provenance, and persistent
   state in a **separate application-owned Zarr store** inside the package.
4. Bind the source and project store with a small, strict, versioned JSON manifest.
5. Add SHA-256 integrity information.
6. Use ZIP64 with `ZIP_STORED` at the archive layer; let Zarr codecs compress Zarr
   chunks.
7. Rebuild the package transactionally on save: write a temporary package, validate
   it, close it, and atomically replace the destination.

The central model is:

```text
MINFLUX Viewer project (.mfxp / ZIP64)
├── immutable acquisition source(s): original .msr
└── mutable application payload: project.zarr
```

This preserves all vendor information while permitting the embedded source store to
remain Zarr v2 and the new project store to use Zarr v3. They are separate stores, not
a mixed-version hierarchy.

## 2. Why the complete MSR, not only its extracted Zarr, is the source boundary

An MSR file is not simply a Zarr directory with another suffix. In the files supported
by this project, the relevant nesting is approximately:

```text
MSR / OBF container
└── one or more stacks
    └── MFXDTA entry archive
        └── embedded Zarr v2 key/value store
            └── mfx array and related nodes
```

The repository confirms this:

- [`minflux_viewer/msr/mfxdta.py`](minflux_viewer/msr/mfxdta.py) describes MFXDTA as a
  flat archive of a Zarr v2 DirectoryStore.
- `extract_zarr_store()` deliberately keeps only entries under the MFXDTA `zarr/`
  prefix and discards other MFXDTA content.
- `extract_did_label_map()` and stack-tag parsing recover dataset identifiers and
  labels from MSR-level information because `mfx/.zattrs` may omit them.
- A single MSR may hold multiple MFXDTA datasets/channels plus images and other OBF
  stacks that are not part of any one embedded Zarr store.
- [`minflux_viewer/msr/msr_parser.py`](minflux_viewer/msr/msr_parser.py) combines the
  embedded store with MSR-level stack identity and label information.

Consequently, exporting only the embedded Zarr can preserve the localization array but
cannot yet be claimed to preserve the complete MSR. It would also discard vendor data
that the current application does not understand but a future reader might.

The safe initial rule is therefore:

> The original `.msr` is the immutable raw/provenance authority. Parsed MSR metadata
> may be cached for convenience, but the cache is derived and is not a lossless
> replacement for the original file.

Do not include both the complete MSR and an extracted copy of its raw Zarr by default.
That duplicates the largest raw data. An extracted raw store may be added later as an
explicitly optional, regenerable cache whose source checksum is recorded.

## 3. Proposed ZIP64 layout

```text
experiment.mfxp                         # standard ZIP64 file
├── minflux-project.json                # required; first ZIP member
├── manifest-sha256.txt                 # required for portable snapshots
├── ro-crate-metadata.json              # optional publication metadata
└── data/
    ├── sources/
    │   ├── <source-uuid-1>.msr         # original bytes, ZIP_STORED
    │   └── <source-uuid-2>.msr         # optional additional source
    ├── project.zarr/                   # app-owned Zarr store
    │   ├── zarr.json                   # for Zarr v3
    │   ├── datasets/
    │   │   └── <dataset-uuid>/
    │   │       ├── mfx/
    │   │       ├── derived/
    │   │       ├── mbm/
    │   │       └── state/provenance metadata
    │   └── groups/                     # overlay/group relationships
    └── previews/                       # optional, non-authoritative
```

Internal paths should use `/`, ASCII-safe fixed names, and generated UUIDs. Preserve the
user-facing original filename in the manifest instead of placing an unchecked source
filename directly into the archive path.

The ZIP root is a MINFLUX Viewer package, **not** a Zarr hierarchy. `data/project.zarr`
is a Zarr hierarchy at a known prefix. Avoid a nested `project.zarr.zip`; nested ZIPs
create two indexes and make random access and recovery worse.

The same logical tree may later be supported as an unpacked working directory. That is
useful for incremental saves, debugging, `rsync`, and direct `LocalStore` access, but it
does not need to block the initial `.mfxp` implementation.

## 4. Required manifest

`minflux-project.json` is the authoritative and intentionally small application
manifest. It should be JSON-Schema validated. A provisional example is:

```json
{
  "format": "org.minflux-viewer.project",
  "format_version": "0.1.0",
  "project_uuid": "43f1c29a-8d77-4dc3-9d52-3f9e034ec3bc",
  "created_at": "2026-08-23T18:00:00+02:00",
  "modified_at": "2026-08-23T18:00:00+02:00",
  "writer": {
    "application": "MINFLUX Viewer",
    "version": "<application-version>"
  },
  "project_store": {
    "path": "data/project.zarr",
    "kind": "minflux-viewer-project-zarr",
    "schema_version": "0.1.0",
    "zarr_format": 3
  },
  "sources": [
    {
      "source_id": "d77d7605-24bf-4d9c-a437-f52571a18475",
      "path": "data/sources/d77d7605-24bf-4d9c-a437-f52571a18475.msr",
      "original_name": "experiment.msr",
      "media_type": "application/x-abberior-msr",
      "size": 123456789,
      "sha256": "<64 lowercase hex characters>",
      "embedded_zarr_format": 2
    }
  ],
  "required_features": [],
  "optional_features": []
}
```

The exact format identifier and extension are still provisional, but they must be
frozen before public files are written.

### Separate version axes

Do not use one version number for unrelated compatibility boundaries. Record at least:

1. **Package format version** — ZIP layout and manifest contract.
2. **MINFLUX project schema version** — meaning and organization of datasets,
   processing, provenance, and state.
3. **Physical Zarr format** — `2` or `3` for the app-owned store.
4. **OME-NGFF version**, only when an actual OME-Zarr image payload is present.
5. Source container and embedded-store versions, when known, as source facts rather
   than package requirements.

Suggested compatibility behavior:

- Unknown package major version: fail with an explicit unsupported-version error.
- Newer minor version: open only if every `required_feature` is supported.
- Unknown optional feature: ignore with a visible warning, without corrupting it.
- Never silently reinterpret an unknown layout as a flat `.zarr` export.

## 5. Project Zarr profile

Zarr defines storage for N-dimensional typed arrays. It does not define MINFLUX table
semantics, processing lineage, filters, or viewer state. A small MINFLUX Viewer Zarr
profile is required whether the physical store is v2 or v3.

The profile should follow the repository's canonical data model:

- `mfx` / `attr`: materialized localizations used by the UI.
- `mfx_raw`: all-iteration raw data when a dataset must be materialized independently
  of a source MSR.
- `derived`: computed values aligned to the materialized localizations.
- `mbm`: bead/beam-reference data.
- `metadata`: immutable import decisions and provenance facts.
- `state`: persistent user-editable filters, transforms, display choices, and ROI
  references.
- Groups/overlays should reference separate datasets; channels remain separate in
  storage.

### Array layout recommendation

Use groups of ordinary arrays rather than one deeply nested NumPy structured dtype.
The current MSR Zarr-v3 compatibility problem is specifically associated with the
vendor structured dtype and nested subarray fields. A column/group layout is easier to
validate and implement across languages.

Follow the project's “one fact, one storage location” rule:

- Store canonical `loc` once, preferably as an `(N, 3)` array in metres.
- Do not also persist redundant `loc_x`, `xnm`, `loc_nm`, and similar views.
- Store scalar-per-localization and per-iteration arrays in their natural shapes.
- Put large numeric metadata in arrays, not enormous JSON attributes.
- Record units and coordinate-frame meaning explicitly.

The precise Zarr tree and JSON Schema are the principal design work still required
before the writer is considered stable.

### Materialized results versus recipes

For an exact project reopen, store the materialized processed result as well as its
processing provenance. Do not automatically rerun old algorithms when opening a file;
software updates could produce different numbers.

Authority should be explicit:

- Original MSR: authoritative immutable acquisition source.
- Materialized project arrays: authoritative saved processed result.
- Processing recipe/history: provenance explaining how the result was produced.
- Viewer state: presentation and editable workflow state, not scientific raw data.

For source-backed datasets that have not been materially changed, the project may store
only a source reference plus metadata/state. Derived datasets, cropped snapshots, and
results that cannot be reconstructed exactly must be materialized in `project.zarr`.

## 6. Stable source identity and row mapping

This is a non-negotiable part of the schema. A processed mask, transformation, or
derived column is useless if it cannot be mapped back to the exact source dataset and
row ordering.

For each source-backed dataset, record at least:

- Source file SHA-256 and `source_id`.
- MSR stack index.
- Vendor dataset `did`, when available.
- MFXDTA container version.
- Embedded node path, normally `mfx`.
- Raw localization-table checksum or canonical content fingerprint.
- Declared raw row-ordering version.
- A stable `source_row_id` for each materialized localization.

Do not use `tid` alone as a row identity; it represents trajectory relationships and
is not guaranteed to be unique for every localization. A source row index is reliable
only when coupled to the exact source dataset fingerprint and ordering contract.

Multi-channel files require one source-dataset locator per channel. Overlay membership,
order, LUT, and alignment belong in project groups/state, not in duplicated channel
arrays.

## 7. ZIP64 engineering rules

### Compression and entry ordering

- Enable ZIP64 unconditionally.
- Use archive method `ZIP_STORED` for the MSR and Zarr members.
- Compress Zarr chunks using the codec declared by the Zarr array.
- Do not double-compress already compressed MSR/Zarr bytes at the ZIP layer.
- Write `minflux-project.json` as the first member for quick recognition.
- Prefer Zarr v3 sharding or sensible chunk sizes if member count becomes excessive.
- Write members in deterministic order where practical.

Zarr-Python provides a `ZipStore` and supports ZIP64, but `ZipStore` does not support
deletion. The Zarr documentation also labels the formal Zip Store specification as
draft. The project package should therefore rely on documented ZIP64 plus its own
manifest, rather than claim that the whole `.mfxp` file is a generic Zarr ZipStore.

OME RFC-9's proposed `.ozx` format is useful engineering precedent for ZIP64, stored
members, early identifying metadata, and sharding. It is not the correct identity or
schema for a MINFLUX project.

### Transactional save

Never append changed nodes repeatedly to an existing package. ZIP permits duplicate
member names and Zarr ZipStore cannot delete individual members; append-only updates
would accumulate stale entries and make recovery ambiguous.

Use this save sequence:

1. Resolve and validate the exact destination.
2. Create a temporary file on the same filesystem.
3. Stream source MSR bytes and project arrays into a new ZIP64 archive.
4. Compute SHA-256 values while streaming rather than rereading when possible.
5. Write/finish manifests and close the archive.
6. Reopen it read-only.
7. Validate paths, manifest schema, required members, checksums, and the project Zarr.
8. Load a small representative slice from every required array.
9. Atomically replace the destination.
10. Leave the previous valid project untouched if any step fails.

Large MSRs make rewrite-on-save an important performance cost. Measure it with real
files. If frequent saves are too slow, add an unpacked working-directory representation
with the identical logical tree and reserve `.mfxp` for portable snapshots. Do not solve
this with unsafe ZIP appends.

### Opening the embedded source MSR

The current MSR path is filename-oriented. The simplest first implementation is:

1. Validate the package and source checksum.
2. Extract the `ZIP_STORED` MSR member to a managed temporary/cache path.
3. Open it through the existing MSR parser.
4. Match source datasets using the manifest locators.
5. Overlay/load the saved project materialization and state.

The extraction must be streamed and must not load the complete MSR into memory. A later
optimization may adapt the pure-Python OBF/MSR reader to a bounded seekable view of an
uncompressed ZIP member, if the parser architecture permits it.

### Reader security

Treat `.mfxp` as untrusted input. Before extraction or allocation:

- Reject absolute member paths, drive-qualified paths, `..`, NULs, and normalized paths
  escaping the destination.
- Reject duplicate normalized member names.
- Reject encrypted members and unsupported compression methods.
- Reject archive symlinks and special files.
- Limit manifest size, member count, individual uncompressed size, and total
  uncompressed size.
- Check declared sizes before allocating arrays.
- Validate JSON types and lengths, not just key presence.
- Verify checksums before treating the source or processed result as authoritative.
- Never execute scripts, pickles, object arrays, or plugin code from a project.

Using `ZIP_STORED` greatly reduces decompression-bomb risk but does not remove path,
duplicate-name, allocation, or malformed-metadata risks.

## 8. Integrity, preservation, and research metadata

ZIP CRC32 detects many accidental errors but is not a cryptographic identity. Portable
snapshots should include SHA-256 for every authoritative payload member, or a documented
Merkle/aggregate alternative if the Zarr member count becomes impractical.

BagIt is a useful optional archival profile. RFC 8493 defines a directory with an opaque
`data/` payload plus cryptographic manifests. A `.mfxp` package could expand to a
BagIt-compatible directory, but BagIt does not define MINFLUX semantics and does not
replace `minflux-project.json`.

RO-Crate 1.2 is useful for publication metadata: people, instruments, software,
licenses, citations, and relationships between source and derived datasets. Keep it
optional initially. The strict application loader should depend on
`minflux-project.json`, not require JSON-LD processing.

## 9. Effect on Zarr v2, Zarr v3, and OME-Zarr decisions

### Zarr v2

- The original embedded vendor store remains v2 and is read through the repository's
  private v2-compatible MSR path.
- Directly augmenting the extracted vendor store would force the project hierarchy to
  remain v2. The outer package avoids that constraint.

### Zarr v3

- Zarr v3 is appropriate for the separate application-owned `project.zarr` once the
  zarr-python 3 migration and PyInstaller build are validated.
- The MSR reader does not need zarr-python 3 to understand the vendor structured array;
  it can continue using `minflux_viewer/msr/zarr2.py`.
- Do not hand-write another canonical Zarr-v3 encoder. Use a maintained Zarr library for
  the new store.
- If implementation must precede the runtime migration, the same project schema may be
  prototyped in Zarr v2, provided the physical Zarr version is explicit. Do not publish
  both encodings under an indistinguishable profile.

### Tables

As of the research date, the Zarr v3.1 core specification defines N-dimensional typed
arrays, not a mature general table/dataframe interchange standard. Future table support
is a reason to keep the semantic schema storage-independent; it is not a reason to wait
or to omit a MINFLUX-specific schema now.

For v0.1, use ordinary named arrays/groups. If later interoperability with analytics
tools becomes more important, the ZIP64 envelope can also carry Parquet or Arrow tables
without changing the original MSR or the envelope. Do not introduce this additional
payload format in v0.1 without a measured need.

### OME-Zarr

- OME-NGFF describes bioimaging data conventions. It should not define the whole
  MINFLUX Viewer project or the localization table.
- Use an OME-Zarr payload only for actual images, density volumes, or multiscale image
  products where OME semantics apply.
- Record the OME-NGFF version only on that payload.
- Do not name the project `.ome.zarr` or `.ozx` unless it actually conforms to the
  corresponding OME specification.

## 10. Findings against `file_write_format.md`

The original document contains a valuable repository audit, benchmarks, a working
Zarr-v2 compatibility shim, and a practical migration analysis. The independent
research changes the package boundary more than it changes the choice of Zarr.

| Original finding or direction | Result | Revised conclusion |
|---|---|---|
| Zarr is the right main scientific-data substrate. | **Confirm.** | Use Zarr for the application-owned numerical payload, with a MINFLUX-specific schema. |
| Existing MSR files already contain genuine Zarr v2 data. | **Confirm.** | This strongly supports reuse of the existing reader, but does not mean the embedded Zarr is the complete MSR. |
| Adopt Zarr v3 for the future/default writer. | **Confirm with boundary.** | Use v3 for the separate `project.zarr`; preserve the vendor v2 store inside the immutable MSR. |
| Fix the zarr-python 3/MSR structured-dtype blocker first. | **Partly revise.** | Keep the private v2 MSR shim. The separate project store may use zarr-python 3 without asking it to decode the vendor dtype. The normal zarr-python 3 migration/build still needs validation. |
| Directory and ZIP forms can share one logical layout. | **Confirm and extend.** | Define one project tree, but make the ZIP root a custom package containing a nested Zarr store rather than claiming the whole archive is one Zarr dataset. |
| `.zarr.zip` is a suitable single-file expression. | **Revise.** | Use a project-specific extension such as `.mfxp` to avoid loader ambiguity and to signal that the archive also contains MSR, manifest, and provenance payloads. |
| ZIP-backed Zarr is compact and benchmark results are competitive. | **Confirm.** | Use `ZIP_STORED` and chunk-level compression. The dominant remaining concern is full-archive rewrite cost. |
| ZIP stores are rewrite-on-save. | **Confirm and elevate.** | Make temp-write, validate, atomic-replace mandatory. Consider a directory working form if measurements show unacceptable save latency. |
| The extracted MSR Zarr could become the project root with app metadata added beside it. | **Reject as the lossless default.** | Some identity/labels and other MSR content lie outside that store. Preserve the complete MSR and create a separate project store. |
| OME-NGFF 0.5 / Zarr v3 can serve as the principal new format. | **Reject for the project as a whole.** | OME-Zarr is appropriate only for image/volume payloads. Localization/project semantics require their own profile. |
| Future Zarr table work strengthens the Zarr decision. | **Qualify.** | It supports keeping Zarr in the design, but current core Zarr still does not supply the required table semantics. Define the MINFLUX schema now. |
| Sidecars can carry processing metadata for otherwise flat exports. | **Supersede for project saves.** | Bundle authoritative processing metadata and state in `.mfxp`; retain sidecars only for interchange formats that cannot contain project metadata. |
| A project archive is a separate higher tier from ordinary scientific export. | **Confirm and promote.** | `.mfxp` should be the lossless project/save format; `.csv`, `.npz`, flat `.zarr`, OME-Zarr, and similar outputs remain explicit interchange/export formats. |
| Current flat `load_zarr()` represents generic Zarr import. | **Reject as a universal dispatcher.** | It currently expects canonical root columns such as `loc_x`, `loc_y`, `itr`, and `vld`. Project packages and vendor stores need explicit detection and dedicated loaders. |

The largest new conclusion is that **format choice and container choice should be
separated**:

- ZIP64 is the portable project container.
- MSR is the preserved acquisition source.
- Zarr is the application numerical storage substrate.
- MINFLUX Viewer defines the semantic project profile.
- OME, BagIt, RO-Crate, and possibly Parquet are optional payload/profile standards for
  their specific purposes.

## 11. Alternatives considered

| Candidate | What it can do | Why it is not the primary choice |
|---|---|---|
| ZIP64 | Preserve arbitrary files, indexed member access, ubiquitous recovery tools, single-file transport. | Rewrite-on-save; solved with transactional rebuild and possibly a directory working form. |
| Directory package | Direct Zarr access, incremental updates, simple debugging and recovery. | Not one file; useful as a working representation. |
| HDF5 | Single hierarchical file with metadata and arrays. | A contained Zarr becomes opaque bytes or must be transcoded, losing direct Zarr interoperability. |
| SQLite | One file with transactions and BLOB/key-value storage. | Would require a custom Zarr store adapter and reduces scientific-tool interoperability. |
| TAR / TAR+Zstandard / 7z | Wrap arbitrary trees efficiently for cold storage. | Poor random member access; usually requires extraction before Zarr use. |
| BagIt | Standard preservation manifests and opaque payload organization. | A packaging profile rather than the application schema; optional archival layer. |
| RO-Crate | Rich research-object and provenance description. | A metadata/catalog standard rather than an array store or binary container. |
| Icechunk | Transactional, versioned, concurrent Zarr storage. | Valuable future cloud/collaboration backend, but substantially more complex than a portable desktop project. |
| Custom binary container | Could duplicate the MFXDTA/MSR approach exactly. | Creates a new index, parser, recovery, security, and interoperability burden that ZIP64 already solves. |

## 12. Proposed implementation sequence

### Phase 0 — freeze the profile

Before producing user data:

1. Choose the final extension and format identifier.
2. Write JSON Schema for `minflux-project.json`.
3. Freeze the v0.1 project-Zarr tree, array names, units, and source-row identity.
4. Document required versus optional features and version behavior.
5. Decide the v0.1 scope for source-backed, derived, simulated, and imported non-MSR
   datasets.

Do not publish a writer before its reader and validator exist.

### Phase 1 — ZIP64 envelope and validator

Suggested pure, Qt-free responsibilities, possibly in
`minflux_viewer/core/project_format.py`:

- Detect the package by ZIP signature plus manifest content, not extension alone.
- Parse and JSON-Schema validate the manifest.
- Normalize and validate member paths.
- Reject duplicates, symlinks, encryption, unsafe compression, and unsupported
  required features.
- Stream SHA-256 calculation and source copying.
- Read a Zarr store rooted at the `data/project.zarr` prefix.
- Provide explicit, typed errors suitable for UI reporting.

Start with a CLI/test helper that can list and validate a package without launching Qt.

### Phase 2 — project Zarr serializer

- Serialize one canonical dataset using the raw/derived/state separation.
- Use ordinary arrays and JSON-compatible metadata; no pickle/object payloads.
- Add source locators and stable row IDs.
- Round-trip materialized processed arrays exactly.
- Preserve unknown optional metadata where feasible.

### Phase 3 — source integration and project loader

- Package one original MSR without modifying it.
- Extract/cache it safely on open and load it through the current parser.
- Match source datasets by checksum + stack index + `did`.
- Apply project materialization, derived arrays, transforms, groups, and state.
- Support multi-channel MSRs before declaring the format production-ready.

### Phase 4 — transactional writer and UI

- Add `Save Project` / `Save Project As`, distinct from scientific `Export`.
- Rebuild to a temporary archive and atomically replace only after validation.
- Report progress for hashing, copying, writing Zarr, and verification.
- Make cancellation leave the previous project intact.
- Add recent-file and open-file dispatch for `.mfxp`.

### Phase 5 — performance and archival additions

- Benchmark realistic 1 GB, 10 GB, and largest-available MSRs.
- Measure first open, repeated open with cache, save, verification, and memory use.
- Decide whether an unpacked working form is necessary.
- Add optional RO-Crate export and/or BagIt-compatible archival output.
- Evaluate Parquet only if measured table-interoperability requirements justify a
  second processed-data encoding.

## 13. Minimum acceptance tests

### Correctness and preservation

- Source MSR extracted after a package round trip is byte-identical and SHA-256
  identical to the input.
- Vendor embedded Zarr v2 and project Zarr v3 coexist and both load through their
  designated readers.
- Single- and multi-channel MSRs map to the correct project datasets.
- Processed coordinates, all canonical attributes, derived arrays, metadata, groups,
  filters, transforms, ROI references, and persistent state round-trip.
- A materialized result reopens identically without rerunning its processing recipe.
- Unknown optional manifest fields do not destroy data.
- Unknown required features and incompatible major versions fail clearly.

### Integrity and failure behavior

- Altering any authoritative payload is detected by SHA-256 verification.
- A truncated ZIP, missing manifest, duplicate name, missing source, or invalid Zarr
  fails without partial import.
- Injected failure/cancellation at every save stage leaves the previous project valid.
- The writer never appends duplicate logical members to an existing archive.

### Security

- Reject `../`, absolute, drive-qualified, backslash-confused, and duplicate-normalized
  paths.
- Reject symlink, encrypted, and unsupported-compression members.
- Enforce configured member-count, manifest-size, total-size, and allocation limits.
- Reject object arrays, pickle-bearing NumPy data, and executable content.

### Performance

- Source copy and extraction are streamed with bounded memory.
- ZIP-level compression is disabled for MSR and compressed Zarr chunks.
- Save/open time and temporary-disk requirements are reported for realistic data.
- ZIP64 paths are tested explicitly rather than relying only on small ordinary-ZIP
  fixtures.

## 14. Explicit non-goals for v0.1

- Writing processed data back into vendor `.msr`.
- Modifying the embedded vendor Zarr store.
- Claiming a complete independent replacement for all unknown MSR content.
- Treating the project as OME-Zarr.
- Concurrent multi-writer access.
- Append-only mutation of a ZIP archive.
- Cloud-native transactional storage.
- Executable workflows or embedded Python objects.
- Requiring RO-Crate, BagIt, Parquet, or Arrow in the core reader.

## 15. Open decisions that must be resolved before implementation is final

1. Final extension and media type (`.mfxp` is provisional).
2. Whether the first release supports only ZIP64 or both ZIP64 and directory forms.
3. Exact `project.zarr` node layout and schema URI.
4. Whether v0.1 requires Zarr v3 or permits a temporary v2 encoding.
5. Stable row-ID construction for every current loader and derived-dataset workflow.
6. Which `metadata`, `derived`, and `state` fields are required, optional, or transient.
7. How source-less simulated and application-derived datasets are materialized.
8. Temporary extraction/cache lifecycle and maximum disk-use policy.
9. Whether SHA-256 is per ZIP member, per logical array, or represented by a scalable
   tree for highly sharded stores.
10. Forward-compatibility behavior for unknown optional project-Zarr nodes.

## 16. Upstream sources

Primary/current technical sources consulted:

- [Zarr v3.1 core specification](https://zarr-specs.readthedocs.io/en/latest/v3/core/)
- [Zarr-Python storage guide](https://zarr.readthedocs.io/en/latest/user-guide/storage/)
- [Zarr-Python `ZipStore` API](https://zarr.readthedocs.io/en/stable/api/zarr/storage/)
- [OME-NGFF RFC-9: Zipped OME-Zarr](https://ngff.openmicroscopy.org/rfc/9/)
- [RFC 8493: BagIt File Packaging Format](https://www.rfc-editor.org/info/rfc8493/)
- [RO-Crate Metadata Specification 1.2](https://w3id.org/ro/crate/1.2)
- [Apache Parquet overview](https://parquet.apache.org/docs/overview/)
- [Apache Parquet file-format specification](https://parquet.apache.org/docs/file-format/)
- [Apache Arrow columnar and IPC format](https://arrow.apache.org/docs/format/Columnar.html)
- [HDF5 virtual datasets through h5py](https://docs.h5py.org/en/stable/vds.html)
- [Icechunk overview](https://icechunk.io/en/latest/overview/)

## 17. Final recommendation for the next implementation session

Start with the envelope and validator, not the complete dataset serializer:

1. Freeze a minimal `minflux-project.json` schema.
2. Write a package containing one unmodified MSR and an empty/minimal Zarr v3 project
   root.
3. Prove byte-identical MSR recovery, safe validation, and opening the nested Zarr
   prefix.
4. Then add one dataset round trip with stable source identity.
5. Expand only after single-source and multi-channel fixtures pass.

This sequence tests the architectural risk—ZIP64 packaging, source preservation,
version separation, and safe reopen—before committing to the full processing/state
schema.

## 18. Empirical export-size model and text-format performance (2026-08-25)

The reference acquisition
`4_BD_MRED_75pM_MINFLUX_3D.msr` is 380,215,735 bytes (362.60 MiB), but its parsed
MFX component contains 20,627,153 all-iteration rows with a 101-byte structured
dtype: 2,083,342,453 bytes (1.94 GiB) of uncompressed numeric payload. Therefore
the MSR size is not the baseline for predicting an export. MSR is a compressed OBF
container and also carries metadata/images; export estimates must start from parsed
array shapes, dtypes and values.

Measured exports were:

| component | MAT | NumPy | JSON | CSV |
|---|---:|---:|---:|---:|
| MFX | 216,921,615 B | 2,083,342,901 B | 5,444,912,421 B | 2,507,171,687 B |
| MBM | 1,344,819 B | 1,977,960 B | 7,056,689 B | 5,643,804 B |

For MFX, NumPy is essentially the exact uncompressed binary payload; MAT compresses
to 10.4% of that payload, CSV decimal text expands to 120.3%, and JSON record text
expands to 261.3%. The MSR-reader Zarr was 296,052,936 bytes (282.34 MiB), including
images and ancillary content. These ratios are dataset-dependent, especially MAT and
Zarr compression, but the direction and order are structural.

The implemented estimator (`core/export_size.py`) therefore uses:

- exact dtype payload + NumPy header for `.npy`;
- actual compact-JSON and CSV serialization of 8,192 stratified rows, extrapolated
  to the complete row count;
- real MATLAB compression over eight distributed contiguous blocks;
- the actual Zarr v2 Blosc/LZ4 codec over distributed blocks, explicitly labelling
  images/search/viewer metadata as additional content.

On the reference MFX+MBM exports, one 1.6-second estimation pass predicted NumPy
exactly, JSON within -0.16%, CSV within -0.34%, and MAT within +3.8%. This is precise
enough for a pre-export UI estimate while remaining far cheaper than performing the
export.

### Performance conclusion

The large text files and their basic conversion cost are inherent: hundreds of
millions of numeric values must be formatted as decimal/JSON tokens and parsed back.
Streaming avoids a giant Python record list but cannot remove that CPU and I/O work.
JSON also repeats every field name for every row, explaining why it is about twice the
CSV size here. CSV and JSON should remain interoperability/inspection formats, not the
recommended working formats for complete all-iteration acquisitions.

Application unresponsiveness was partly avoidable and is handled separately:

- single-file MSR export now runs off the Qt UI thread;
- the spreadsheet mapping dialog samples at most 100 displayed rows from the start,
  distributed byte offsets and the tail instead of parsing the entire CSV first;
- its row count is explicitly approximate (estimated from 512 byte-offset samples);
- a canonical MINFLUX CSV accepted with its default mapping uses the dedicated raw
  loader, preserving every iteration field rather than reducing the table to generic
  x/y/z roles;
- canonical CSV parsing now constructs its structured dtype directly, avoiding the
  previous multi-gigabyte all-float matrix before canonical recomposition;
- top-level JSON arrays are rejected as ROI files from a 4 KiB prefix, avoiding a
  complete 5.07 GiB classification parse before the streaming data loader;
- large text export/load warnings recommend Zarr, MAT or NumPy and state that the full
  conversion can still take minutes.

The remaining limit is fundamental full import time. After the user confirms the CSV
mapping—or confirms a large JSON warning—the complete text file still has to be read
and converted. Binary Zarr/MAT/NumPy should be used when fast repeated reopening is a
requirement.
