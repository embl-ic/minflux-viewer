# MINFLUX Viewer project format — decision and implementation plan

**Status:** decided; ready to implement
**Date:** 2026-08-23
**Supersedes:** the format direction in `file_write_format.md` §7.3
**Builds on:** `file_format_research_result.md`, with the corrections in §2 below.

---

## 1. Decision

**A ZIP64 package is the MINFLUX Viewer default save format.**

```text
project.mfv        (ZIP64)  |  project.mfv/  (directory)
├── minflux-project.json      # first member; the authoritative manifest
├── manifest-sha256.txt
└── data/
    ├── sources/<uuid>.msr    # ZIP_STORED, byte-for-byte, seekable in place
    └── project.zarr/         # Zarr v3, app-owned; derived + state ONLY, no raw copy
```

### Why this is not "inventing a file format"

The stated preference was to adopt community standards rather than invent a format.
This design honours that: **every layer that holds data is a standard**, and only the
binding between them is ours.

| Layer | Standard | Ours? |
|---|---|---|
| Container | ZIP64 (PKWARE APPNOTE; universal tooling) | no |
| Acquisition source | Abberior `.msr` / OBF, unmodified | no |
| Numeric payload | Zarr v3 + Blosc/zstd | no |
| Image payloads, when present | OME-NGFF | no |
| Publication metadata, optional | RO-Crate 1.2 | no |
| **Manifest + arrangement** | **`minflux-project.json`** | **yes** |

This is the same pattern OME-Zarr uses (Zarr + JSON conventions), that pyMINFLUX's
`.pmx` uses (HDF5 + a layout), and that SpatialData uses (Zarr + Parquet +
conventions). No existing community format does this job — verified: OME-NGFF
explicitly declined to standardize localization tables, and nothing in the ecosystem
preserves a vendor MSR alongside derived analysis.

A consequence worth stating plainly: **anyone can recover the science from a
`.mfv` with `unzip` plus any Zarr reader**, without our software. That is the
property that matters for a scientific archive, and a bespoke binary container would
not have it.

### Evidence base

Every number below was measured on real sample files on this machine.

| Property | Result |
|---|---|
| Members in a real package | 249 - 1,274 |
| Central directory at 100,000 members | 0.30 s to parse |
| Random member read | **0.013 s, independent of member count** |
| >4 GiB member (true ZIP64) | works; ZIP64 EOCD present; **seek past 4 GiB = 0.0 ms** |
| Embedded MSR openable in place | **yes** - `ZIP_STORED` member is seekable |
| Full save, 370 MB source, 2 channels | 7.9 s (1.7 s of it packing) |
| Re-save, unchanged 370 MB source | **0.45 - 0.73 s** (plain file copy = 0.17 s) |
| Reopen + read manifest | **1.3 - 3.5 ms** |
| Package overhead, no-duplication rule | **1.005x - 1.021x** of source |
| SHA-256 of a whole source | 0.01 - 0.22 s |

Member count, ZIP64 limits, save latency and integrity hashing are all non-issues at
our scale. The container is not the risk; the schema is.

---

## 2. Corrections to the research document

These are empirical, and the plan below reflects them.

| Claim in `file_format_research_result.md` | Verdict | Correction |
|---|---|---|
| "`extract_zarr_store()` discards other MFXDTA content", used to justify preserving the whole MSR | **Right conclusion, wrong reason** | Discarded MFXDTA content measures **0.000 MB** (the `sync` manifest is empty). The real argument is the **rest of the OBF container** - 23%-69% of every sample file: confocal images, overview stacks, OME-XML, footers, acquisition ROIs. |
| "Do not double-compress already compressed MSR/Zarr bytes at the ZIP layer" | **False** | Real MSRs deflate to **0.610x on average** (range 0.131-0.830 over 1.2 GB). They are *not* already-compressed - only the MFXDTA chunks are. `ZIP_STORED` costs +64% on average. |
| `ZIP_STORED` for the MSR | **Keep, for a different reason** | Justified by **in-place seekability** (verified: 0.0 ms seeks, including past 4 GiB), which removes extraction entirely. Not by compression. |
| Phase 3: "extract the MSR member to a managed temporary/cache path" | **Drop** | Unnecessary. Patch `OBFFile` to accept a file object (we already monkeypatch msr-reader in `msr/obf_compat.py`). This also deletes open decision #8, cache lifecycle and disk policy. |
| Every dataset gets `raw/mfx` in `project.zarr` | **Reject outright** | Duplicates data preserved byte-for-byte in the MSR, violating design rule 1 (*one fact, one storage location*). A source-backed dataset must store **no** raw columns and **no** coordinates at all - only derived, masks and state. Measured: **1.376x-1.684x** for a raw copy vs **1.005x-1.021x** for no duplication, i.e. the project store shrinks 28x-110x. |
| "Prefer Zarr v3 sharding if member count becomes excessive" | **Not needed** | Real packages are 250-1,300 members; 100,000 is still 0.30 s. Do not add sharding for member-count reasons. |
| Rewrite-on-save may force a directory working form | **Not supported by measurement** | Re-save of a 370 MB source is 0.45-0.73 s. Offer the directory form for rsync/debugging, not latency. |
| `ZipStore` cannot delete, so transactional rebuild is mandatory | **Confirmed** | `supports_deletes == False`, raises `NotImplementedError`. |
| Extension `.mfxp` | **Change** | Already used by Native Instruments (Maschine Effect Preset). Proposed: **`.mfxproj`**. |

---

## 3. Format specification v0.1

### 3.1 Identity

- Extension **`.mfv`** · format id `org.minflux-viewer.project` · `format_version` `0.1.0`
  (`.mfv` collides only with obscure formats - MobileFrame Device Pack, a genomics
  "Multiple File Viewer" - none plausible on a microscopy workstation. `.mfxp` was
  rejected: Native Instruments Maschine Effect Preset.)
- Detection: ZIP signature **and** a parsable `minflux-project.json` first member.
  Never by extension alone.
- Media type (provisional): `application/vnd.minflux-viewer.project+zip`

### 3.2 The no-duplication rule (central)

> `project.zarr` stores **nothing that the embedded MSR already contains.**

The raw localizations live in exactly one place: the preserved `.msr`, byte-for-byte,
in the vendor's own encoding. `project.zarr` holds only what the viewer added.

**Stored in `project.zarr`:**

| Group | Contents | Cost |
|---|---|---|
| `identity/` | `source_row_id` (int64) mapping each materialized row to its `mfx_raw` row | ~0.05 MB / 180k locs (highly compressible; a strided sequence) |
| `derived/` | expensive or externally-dependent values: `den` (radius, method), `loc_precision_*`, mapped `confocal_signal_*` | ~4 bytes/loc each |
| `state/` | `ftr` filter mask, `roi_<id>_mask` selection masks | ~1 byte/loc each |
| attrs (JSON) | RIMF + provenance, `filter_specs`, `transform_4x4`, overlay id/order/LUT, view state, lineage | KB |

**Never stored — read from the MSR:** `loc_x/y/z`, `tid`, `tim`, `itr`, `vld`, `efo`,
`cfr`, `dcr`, `eco`, `ecc`, `efc`, `fbg`, `lnc`, `gri`, `sta`, `thi`, `sqi`, `fnl`,
`bot`, `eot`.

**Never stored — recomputed on load:** `dst`, `spd`, `dt`, `tim_trace`, `siz`, `dur`,
`len`. These are O(n) from `tid`/`tim`/`loc` and are already recomputed on every load
today.

**Never stored — a computed view:** `xnm`, `ynm`, `znm`. `Dataset.loc_nm` is
`loc_x/y/z x 1e9`, with `z x RIMF`, then the display transform. The viewer keeps
exposing nm coordinates; the archive keeps `mfx.loc` in metres. Storing both would be
the exact duplication this rule forbids, and would reintroduce the
"RIMF applied twice" hazard that `loc_nm` exists to prevent.

**Measured cost of each policy** (real files, project.zarr size / package overhead):

| policy | 2_3C (78.6 MB) | 3_Bonly (240.4 MB) | 4_3D_PAINT (369.9 MB) |
|---|---|---|---|
| copy all raw rows | 29.6 MB / 1.376x | 140.3 MB / 1.584x | 252.9 MB / 1.684x |
| copy materialized view | 7.2 MB / 1.092x | 10.3 MB / 1.043x | 58.9 MB / 1.159x |
| **no duplication** | **1.05 MB / 1.013x** | **1.28 MB / 1.005x** | **7.44 MB / 1.020x** |

No-duplication is 28x-110x smaller than copying the raw and 7x-8x smaller than copying
the materialized view. Package overhead becomes **1.005x-1.021x** - the MSR plus a
rounding error.

An explicit `source_row_id` costs only ~4% more than storing the selection rule alone
(1.33 vs 1.28 MB), so **store the index**: it removes all ambiguity about how
materialized rows map back to raw rows, and settles the row-identity question without
relying on a rule staying expressible.

**Full materialization is the exception, not the default:**

| Dataset kind | Coordinates |
|---|---|
| Source-backed (any amount of processing) | **never stored** - re-read from the MSR, view recomputed from RIMF + transform |
| Derived: crop, aggregate, drift-correct, flatten, DCR/time separation | **materialized** - genuinely new data, with a lineage edge to the parent |
| Simulated / imported non-MSR | **materialized** - no source to reference |
| Archival snapshot that must survive without the MSR | **materialized** - explicit `full_materialization: true` |

### 3.3 Source identity contract (non-negotiable)

Per source-backed dataset, recorded in the manifest: `source_id` · source file
`sha256` · MSR stack index · vendor `did` · MFXDTA container version · embedded node
path (normally `mfx`) · raw table fingerprint · row-ordering version · and a stable
`source_row_id` per materialized localization.

**`tid` must never be used as row identity** - it is a trace id shared by many
localizations.

`source_row_id` is stored explicitly rather than inferred from a selection rule: it
compresses to ~4% overhead (measured 1.33 vs 1.28 MB) and removes any dependence on a
rule remaining expressible as the materialization logic evolves. This settles open
question 5 of the research document.

### 3.4 Version axes (independent)

1. package format version · 2. project schema version · 3. physical Zarr format ·
4. OME-NGFF version (only on an actual image payload) · 5. source container and
embedded-store versions, recorded as facts, not requirements.

Behaviour: unknown major -> explicit failure. Newer minor -> open only if every
`required_feature` is supported. Unknown optional feature -> warn, preserve, never
corrupt. **Never** silently reinterpret an unknown layout as a flat `.zarr` export.

### 3.5 Engineering rules

- ZIP64 unconditionally; `ZIP_STORED` for source MSRs and Zarr chunks; compression
  happens in the Zarr codec - `BloscCodec(cname="zstd", clevel=3, shuffle="shuffle")`,
  because a plain `ZstdCodec` has no byte-shuffle and is **28% worse**.
- `minflux-project.json` first member; deterministic member order.
- Root group attributes passed at `create_group(attributes=...)`, never assigned after
  array creation - a `ZipStore` cannot rewrite an entry and would append a **duplicate
  `zarr.json`**.
- Transactional save: temp file on the same filesystem -> stream + hash -> close ->
  reopen read-only -> validate -> atomic replace. Never append to an existing archive.
- Untrusted input: reject `..`, absolute/drive-qualified paths, NULs, duplicate
  normalized names, symlinks, encrypted members, unsupported compression; enforce
  member-count, manifest-size and total-size limits; verify checksums before treating a
  payload as authoritative; never execute pickles or object arrays.

### 3.6 Two forms, one tree

The directory form is the same tree unzipped. `.mfv` (ZIP64) is the portable snapshot; the
`.mfv/` directory is the working form (rsync, debugging, direct `LocalStore`). One reader
serves both, selected by path type.

---

## 4. Implementation phases

Each phase ends green and is independently reviewable. Module names are proposals.

### Phase 0 — freeze (0.5 d)

Freeze the extension, format id, JSON Schema, the v0.1 `project.zarr` tree, and the
row-identity contract. Write the schema **before** the writer. No user data is
produced until the reader and validator exist.

Deliverable: `docs/project-format-v0.1.md` + `minflux_viewer/core/project_schema.json`.

### Phase 1 — envelope, validator, in-place source open (2.5 - 3 d)

`minflux_viewer/core/project_package.py`, Qt-free:

- `detect(path) -> PackageKind` — signature + manifest, not extension
- `read_manifest(path)` — JSON-Schema validated, typed errors
- `validate(path, *, deep=False)` — paths, duplicates, limits, checksums, Zarr root
- `open_source(path, source_id) -> BinaryIO` — **seekable member, no extraction**
- `write_package(...)` — transactional temp -> validate -> atomic replace
- `PackageError` hierarchy suitable for UI reporting

Plus `msr/obf_compat.py`: patch `OBFFile.__init__` to accept a file object, so the
embedded MSR opens in place. **This is the phase that retires the architectural risk**
— ZIP64 packaging, source preservation, safe reopen — before any schema work.

CLI: `python -m minflux_viewer.core.project_package validate <path>`.

Tests: `tests/test_project_package.py` — byte-identical source recovery, ZIP64 >4 GiB
(synthetic), path-traversal and duplicate-name rejection, truncated archive, atomic
replace under injected failure, in-place `OBFFile` open against a real `.msr`.

### Phase 2 — project Zarr serializer (3 - 4 d)

`minflux_viewer/core/project_store.py`:

- Write / read one dataset under the raw / derived / state separation
- **No-duplication rule** (§3.2): source-backed datasets store no raw columns and no
  coordinates; `full_materialization` is an explicit opt-in for derived/source-less data
- Source locators and `source_row_id`
- Ordinary named arrays; JSON-compatible metadata; no pickles or object dtypes
- Preserve unknown optional nodes on rewrite

Tests: exact round trip of processed coordinates, all canonical attributes, derived
arrays, filters, transforms, ROI references, overlay/group membership, and state.

### Phase 3 — project loader (2 - 3 d)

Open a package -> match source datasets by `sha256` + stack index + `did` -> re-read
raw from the embedded MSR -> apply materialized processed arrays, derived, transforms,
groups, state. Multi-channel MSRs before the format is called production-ready.

### Phase 4 — transactional writer + UI (2 - 2.5 d)

`File > Save Project` / `Save Project As`, distinct from `Export`. Progress for
hashing, copying, Zarr write and verification. Cancellation leaves the previous
project intact. Recent-files and drag-drop dispatch for `.mfv`, including the
`.zip`-suffix disambiguation for the directory/zip forms.

### Phase 5 — performance and archival (1.5 - 2 d)

Benchmark the largest available MSRs; decide whether the directory working form ships
in v0.1; optional RO-Crate export.

**Total: 12 - 15 days.**

### Sequencing against the audit bugs

F1-F4 from `file_write_format.md` (silent 100 nm alignment loss, corrupt CSV raw
reload, unreadable `.npz` / `.zarr` snapshots, ignored `.msr` sidecar) are **2 - 3 days
and independent of all of this**. They should land first or in parallel: a new project
format does not help anyone whose current saves are silently wrong.

---

## 5. Acceptance tests

**Preservation** — extracted source is byte- and SHA-256-identical; vendor Zarr v2 and
project Zarr v3 coexist and load through their designated readers; single- and
multi-channel MSRs map to the correct datasets; a materialized result reopens
identically **without rerunning its recipe**; unknown optional fields survive.

**Integrity** — any payload alteration is detected; truncated ZIP, missing manifest,
duplicate name, missing source or invalid Zarr all fail without partial import;
injected failure at every save stage leaves the previous project valid; the writer
never appends duplicate logical members.

**Security** — reject `../`, absolute, drive-qualified, backslash-confused and
duplicate-normalized paths; reject symlink, encrypted and unsupported-compression
members; enforce member-count, manifest-size, total-size and allocation limits; reject
object arrays and pickle-bearing NumPy data.

**Performance** — streamed source copy with bounded memory; ZIP-level compression off
for stored members; save/open time and temp-disk use reported for realistic data;
ZIP64 paths tested explicitly, not only small fixtures.

---

## 6. Non-goals for v0.1

Writing processed data back into a vendor `.msr` · modifying the embedded vendor Zarr
store · claiming a complete independent replacement for all unknown MSR content ·
treating the project as OME-Zarr · concurrent multi-writer access · append-only ZIP
mutation · cloud-native transactional storage · executable workflows or embedded
Python objects · requiring RO-Crate, BagIt, Parquet or Arrow in the core reader.

---

## 7. Decisions taken

| # | Decision | Note |
|---|---|---|
| 1 | **Extension `.mfv`** | Collides only with obscure formats; `.mfxp` rejected (Native Instruments) |
| 2 | **Embed the MSR by default** | Self-contained project; overhead 1.005x-1.021x under the no-duplication rule. Reference-by-checksum remains a manifest-expressible option, not the default |
| 3 | **Directory form is the primitive; ZIP64 is a packaging step over it** | See below |
| 4 | **F1-F4 correctness fixes land first**, then Phase 1 | They are independent and users are losing data today |
| 5 | Branch **`zarr-format-test`**; `main` untouched | |

### Why directory-first makes ZIP64 nearly free

The directory form *is* the tree; the ZIP form is `zipfile.write()` over that tree.
Concretely:

- **Write:** build the package as a directory (`LocalStore` for `project.zarr`), then
  either leave it or zip it. The prototype did exactly this.
- **Read:** a directory opens with `LocalStore`. A ZIP opens by slurping the
  `data/project.zarr/` members into an in-memory store.

That second point is a direct consequence of the no-duplication rule: `project.zarr` is
now **1-8 MB**, so reading it wholesale into memory is trivial and **no custom Zarr
store adapter is needed for the ZIP form**. Only the MSR needs streaming, and that is
already solved by the seekable `ZIP_STORED` member.

Note zarr's own `ZipStore` is *not* the mechanism: it makes the whole archive one Zarr
store, whereas the project needs Zarr at a prefix alongside the MSR and manifest.

## 8. Reproducing the measurements

```bash
# per-file MSR composition, deflate ratios, ZIP timings
./.venv/Scripts/python.exe scripts/zarr3_roundtrip_check.py --help

# prototype package build (scratchpad scripts from the research session):
#   proto_pkg.py   - MSR + project.zarr -> .mfv, timed
#   proto_lean.py  - full vs lean project store, size comparison
```

The prototype scripts live in the session scratchpad and should be promoted into
`scripts/` alongside the Phase 1 validator so the numbers stay reproducible.
