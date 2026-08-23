# Data save / export format — state, conclusion, decisions, challenges

**Status:** research + prototype complete; main migration not started.
**Date:** 2026-08-19  ·  **Baseline commit:** `78a9b90` (working tree dirty, see §6)
**Purpose of this file:** a hand-off brief for an independent review session. Every
factual claim below was *measured on this machine*, not inferred from documentation.
Claims are written so they can be re-run and falsified — see §8 for the checklist.

> **Reviewer note.** Treat the numbers as reproducible observations, not conclusions.
> The two judgements most worth attacking are (a) that Zarr is the right target
> format, and (b) that the zarr-python 3 blocker justifies a local shim rather than
> waiting for upstream. Both are argued in §5 and challenged in §7.

---

## 1. Scope

Localization **save / export / re-open** for MINFLUX Viewer:

- what the app can write today and whether it can read it back,
- which processing survives a round trip,
- what the default "save my work" format should be,
- what it costs to get there.

Out of scope: image (TIFF/OME-TIFF) export, ROI-set JSON as a standalone feature.

---

## 2. Current state — what ships

### 2.1 Eight entry points, three dialogs, no shared contract

| Menu command | Handler | Writes | Recipe? | Reopens? |
|---|---|---|---|---|
| `File > Save…` | `save_dataset` | raw or snapshot, 7 formats, prefs-driven | yes | format-dependent |
| `Save As > MINFLUX (.mat/.npy/.json)` | `_save_as_format` | raw canonical only, options hardcoded | yes | yes |
| `Save As > .msr` | `_save_as_format` | raw canonical, custom OBF/MFXDTA | written, **never read** | via MSR reader |
| `Save As > Spreadsheet (.csv)` | `_save_as_spreadsheet` | processed snapshot, chosen columns | **none** | as generic table |
| `Save As > Zarr v2` | `_save_as_format` | raw canonical, flat columns | yes | yes |
| `Save As > HDF5…` | `_save_as_picasso_hdf5` | Picasso `/locs` + YAML | **none** | **no** |
| `Save As > OME-TIFF…` | `_save_as_ome_tiff` | rendered image voxels — not localizations | n/a | as image |
| `Save As > OME-NGFF 0.5 / Zarr v3` | `_save_as_ome_zarr` | density pyramid + tables + ROIs + journal | embedded | **no** (§2.4) |

Two of the eight do not write localizations at all. Two write processed data with no
provenance. `File > Save…` honours the Preferences export defaults; the seven
`Save As` items hardcode their own and bypass both the prefs and the unified dialog.

Five unrelated file species already exist: ROI-set JSON, filter-preset JSON,
particle-set HDF5 (`core/particle_set.py`), Picasso HDF5, and the OME-Zarr package.

### 2.2 Round-trip matrix (measured)

One synthetic dataset (50 localizations × 4 iterations) carrying realistic
processing — RIMF 0.67, a +100/−50 nm channel transform, an `efo` filter, a derived
`den`, a precision calibration, overlay membership, an ROI mask — saved and reloaded
through every format × content mode.

| Target | Locs | Raw rows | Itr | RIMF | Filter | Transform | Overlay | ROI | Reopens |
|---|---|---|---|---|---|---|---|---|---|
| *original (in app)* | 50 | 200 | 4 | 0.67 | 33/50 | yes | yes | 1 | — |
| mat / raw | 50 | 200 | 4 | 0.67 | 33/50 | **LOST** | no | 0 | yes |
| mat / snapshot | 50 | 0 | — | 1.0 (baked) | 33/50 | baked | no | 0 | yes |
| npy / raw | 50 | 200 | 4 | 0.67 | 33/50 | **LOST** | no | 0 | yes |
| npy / snapshot | 50 | 0 | — | 1.0 | 33/50 | baked | no | 0 | yes |
| json / raw | 50 | 200 | 4 | 0.67 | 33/50 | **LOST** | no | 0 | yes |
| json / snapshot | 50 | 50 | 1 | 1.0 | **50/50 — ftr lost** | baked | no | 0 | yes |
| csv / raw | **200** | 0 | — | **1.0** | **lost** | **LOST** | no | 0 | **corrupt** |
| csv / snapshot | 50 | 0 | — | 1.0 | **50/50 — ftr lost** | baked | no | 0 | yes |
| zarr / raw | 50 | 200 | 4 | 0.67 | 33/50 | **LOST** | no | 0 | yes |
| zarr / snapshot | — | — | — | — | — | — | — | — | **NO** — reload rejects it |
| npz / raw + snapshot | — | — | — | — | — | — | — | — | **NO** — no loader exists |
| msr / raw | — | — | — | — | — | — | — | — | partial — sidecar ignored |
| msr / snapshot | — | — | — | — | — | — | — | — | correctly refused |

**Baseline:** all 61 existing save/export tests pass. None of the defects below are
covered, because no test round-trips a dataset that has been processed.

### 2.3 Verified defects, ranked

| ID | Sev | Finding |
|---|---|---|
| **F1** | critical | **Channel alignment destroyed by every raw round trip.** Measured displacement **100.000 nm**, silent. `save.build_metadata` writes the transform correctly (a `display_transform_record` **dict**); `loader.apply_metadata_recipe` reloads it with `np.asarray(tf, dtype=float)`, which raises `TypeError` on a dict, swallowed by a bare `except Exception: pass`. Had it not raised, the ndarray it stored is ignored by `apply_display_transform_nm` (dict-only) but honoured by `transform_to_matrix4` — so render and crop/save would then disagree about where the data is. |
| **F2** | critical | **CSV "raw canonical" reloads as structurally different data.** 200 raw iteration rows return as *200 localizations* instead of 50 × 4 iterations; iteration axis gone, RIMF reset to 1.0, filters lost, metre-scale `loc_x` re-read by a path assuming processed nm. Cause: `load_zarr` reassembles via `save.columns_to_mfx_array`; `load_csv` does not, despite the same writer. |
| **F3** | high | **Three format combinations write files that can never be reopened.** `.npz` is enabled by default in Preferences but absent from `_SUPPORTED_EXTS`. `.zarr` + snapshot writes a store `load_zarr` then rejects (`missing required column(s): itr, loc_x, loc_y, vld`). All report success. |
| **F4** | high | **`.msr` recipe sidecar written but never read.** `apply_metadata_sidecar` is wired into `load_dataset`, `load_npy`, `load_zarr`, `load_csv`, `load_json` only; `.msr` opens through the MSR reader plugin. RIMF, filters, transform evaporate. |
| **F5** | medium | **The "processing recipe" records 3 of ~20 operations** — RIMF, transform, `filter_specs`. Not recorded: ROI crop, duplication, aggregation, drift correction, channel flatten, DCR separation, time-window separation, confocal mapping, local-density params, precision method, overlay membership/LUT, ROI masks, any analysis result. The dataset already carries in-memory provenance keys for most (`aggregation`, `drift_correction`, `cropped_from_dataset`, `flattened_overlay`, `separated_by`, `particle_average`, `time_channels_*`); all dropped at save. |
| **F6** | medium | **"Raw canonical" is not the file's raw data.** `dataset_to_mfx_array` re-serializes `mfx_raw`, which import already truncated. A two-channel `dcr` (N,2) exports as (N,) — **the second DCR channel is gone**, and DCR separation is a shipped feature. m2205/JSON raw builders keep `dcr_0`/`dcr_1`; the flat m2410 path keeps only column 0 (`attrs[key] = arr[vld, 0].ravel()`). |
| **F7** | medium | **Internal cache columns leak into saved files.** `_raw_loc_id` lazily caches a `loc_id` column into `mfx_raw`; `dataset_to_mfx_array` has no exclusion, so it is written as acquisition data. File contents therefore depend on what the user clicked before saving. |
| **F8** | medium | **`ftr` filter-state restore is format-dependent.** Snapshot+flag: `.mat`/`.npy` restore the mask (33/50); `.json`/`.csv` do not (50/50). |
| **F9** | medium | **Two incompatible filter serializations.** Recipe: `{attribute, mode, itr, lo, hi, lo_inc, hi_inc}`. Filter preset JSON: `{apply, attribute, value_as, min, max, iteration}` — drops bound inclusivity. |
| **F10** | medium | **Units implicit and vary within one extension.** `loc_*` metres vs `xnm/*` nm; a CSV may be either, distinguished only by an optional sidecar. |
| **F11** | low | **`Save As` bypasses the `Save` design** (hardcoded `content="raw"`, `include={attrs:True, derived:False, recipe:True}`). |
| **F12** | low | **Three container schemas invented independently** (particle-set HDF5, Picasso HDF5, OME-Zarr package). |

### 2.4 `File > Save As > OME-NGFF 0.5 / Zarr v3…` — what it actually is

`core/ome_zarr.py`, 1270 lines. Its docstring states it is *"intentionally
independent of the application's pinned Zarr v2 dependency"* and emits *"the small
Zarr v3 core needed by this profile"* — i.e. **someone hand-rolled a zarr v3 writer
specifically to dodge the zarr-2 pin.**

**It works, and is better than the audit first credited.** Verified on a real `.msr`
(19,954 locs, 3-D), read back with real zarr-python 3.3.0:

```
ome.version      : 0.5
axes             : ['z', 'y', 'x']
pyramid levels   : ['0', '1', '2']
level 0: shape=(29, 682, 636) uint32 chunks=(16, 256, 256) codec=GzipCodec
level 0 sum = 19954   level 1 sum = 19954   pyramid consistent: True
```

Structure written:

```
<name>.ome.zarr/
├── zarr.json                  # NGFF 0.5: multiscales, axes, coordinateTransformations
├── 0/ 1/ 2/                   # density pyramid, count-preserving 2x2 XY downsample
├── minflux/
│   ├── manifest.json
│   └── datasets/<uuid>/
│       ├── raw/mfx/           # loc, lnc, dcr, itr, vld, efo, ... (20 arrays)
│       ├── processed/current/ # position, ftr, derived attrs, source_row_id
│       ├── state/             # filters.json, rois.json, view.json
│       ├── provenance/        # recipe.json, events.json
│       └── metadata/source.json
├── ro-crate-metadata.json     # RO-Crate 1.1 provenance
└── _SUCCESS
```

`minflux/raw/mfx/loc` is **byte-identical to the source `.msr`**, NaN positions
included (363,338 rows; compare with `equal_nan=True` — a plain `array_equal`
reports False purely because of the m2410 invalid-row NaNs).

**But it is write-only, and the failure mode is worse than "no reader":**

```
>>> load_zarr("ome_test.ome.zarr")
PathNotFoundError: nothing found at path ''
```

That error comes from **zarr-python 2** — the app writes zarr **v3**, which our
pinned zarr cannot open at all. So *the app emits a format its own dependency
cannot read*. And because `Path("x.ome.zarr").suffix == ".zarr"`, dragging it back
in routes to `load_zarr` and dies there. There is no `.ome.zarr` reader anywhere.

`tests/test_ome_zarr_export.py` has 7 tests and **none import zarr** — they read the
emitted `zarr.json` as plain JSON, so they verify metadata shape but could never
catch the read-back gap.

Easy win: the pyramid uses `GzipCodec`. Same volume, same speed:

| codec | size | time |
|---|---|---|
| GzipCodec (current) | 0.08 MB | 0.11 s |
| Blosc zstd-3 + shuffle | **0.02 MB** | 0.11 s |

---

## 3. Measurements

### 3.1 Format benchmark

250,000 localizations × 10 iterations = 2.5M rows, 18 columns, 302.5 MB
uncompressed, clustered field. Our own `.msr` writer is the reference.

| Format | Size MB | vs .msr | Write | Read |
|---|---|---|---|---|
| zarr · blosc/zstd-9 | 193.9 | 0.99× | 16.31 s | 0.35 s |
| **.msr (our writer)** | **195.8** | **1.00×** | 2.81 s | — |
| **zarr · zstd-3 · ZIP [1 file]** | **203.9** | **1.04×** | **0.80 s** | **0.21 s** |
| **zarr · zstd-3 · directory** | **203.9** | **1.04×** | 1.34 s | 0.21 s |
| zarr · blosc/lz4 (.msr codec) | 214.0 | 1.09× | 1.24 s | 0.32 s |
| HDF5 · gzip-9 | 224.2 | 1.14× | 22.10 s | 1.06 s |
| npz (compressed) | 224.4 | 1.15× | 9.38 s | 1.09 s |
| mat (scipy, compressed) | 224.4 | 1.15× | 10.52 s | — |
| HDF5 · gzip-4 | 225.1 | 1.15× | 6.92 s | 1.07 s |
| HDF5 · lzf | 254.3 | 1.30× | 1.07 s | 0.23 s |
| npy (uncompressed) | 302.5 | 1.54× | 0.25 s | 0.09 s |
| csv | 729.0 | 3.72× | 14.52 s | — |
| json | 933.5 | 4.77× | 31.84 s | — |

The ZIP store writes **faster** than the directory form at identical size (fewer
filesystem ops), so "one file" costs nothing.

### 3.2 Scaling — 64× range, constant advantage

| Localizations | Rows | Raw MB | zarr.zip MB | write | read | HDF5 MB | write | read | ratio |
|---|---|---|---|---|---|---|---|---|---|
| 25,000 | 250,000 | 30.2 | 20.4 | 0.11 s | 0.03 s | 22.5 | 0.70 s | 0.11 s | 1.11× |
| 100,000 | 1,000,000 | 121.0 | 81.6 | 0.35 s | 0.10 s | 90.1 | 2.78 s | 0.42 s | 1.10× |
| 400,000 | 4,000,000 | 484.0 | 326.2 | 1.27 s | 0.36 s | 360.0 | 11.03 s | 1.71 s | 1.10× |
| 1,600,000 | 16,000,000 | 1936.0 | 1304.5 | 4.83 s | 1.40 s | 1439.5 | 44.55 s | 6.68 s | 1.10× |

Both scale linearly. At 16M rows Zarr writes **9.2× faster**, reads **4.8× faster**,
in a **10 % smaller** file.

> **Fairness caveat (do not drop this).** HDF5 *can* reach similar compression via
> `hdf5plugin`, so much of the gap is compressor, not container. The default is
> reported because that plugin is an extra compiled dependency **and files written
> with it fail to open in HDF5 readers lacking it**. Zarr reaches zstd through
> numcodecs with nothing added. The container difference that survives is
> chunk-per-object layout, which is what gives the write advantage.

### 3.3 Dependency cost of zarr-python 3

| Package | zarr 2 stack | zarr 3 stack | Note |
|---|---|---|---|
| zarr | 3.28 MB | 3.39 MB | |
| numcodecs | 4.14 MB | 2.19 MB | −1.95 MB |
| asciitree, fasteners | 0.26 MB | — | dropped |
| packaging | — | 1.00 MB | already in venv |
| PyYAML | — | 0.73 MB | **new** |
| typing_extensions | — | 0.18 MB | already in venv |
| donfig | — | 0.13 MB | **new** |
| google-crc32c | — | 0.10 MB | **new** |
| **Total** | **7.69 MB** | **7.72 MB** | **+0.03 MB** |

---

## 4. The blocker: zarr-python 3 cannot read a single MINFLUX file

### 4.1 Symptom

Every `.msr` embeds a zarr v2 store whose `mfx` array is a **structured dtype with
subarray fields**. zarr-python 3.3.0 (newest release) cannot represent it:

```
ValueError: No Zarr data type found that matches
  {'name': [['vld','|b1'], ['tid','<i4'], ['dcr','<f8',[2]], ['loc','<f8',[3]]], ...}
```

Confirmed on **6 of 6** real sample files:

```
1.msr                                    n=   197,120  fields= 9  subarrays=['itr']
10_Nup62_215TCO_LD655_50nM_2mM_THP_anti  n=   363,338  fields=20  subarrays=['loc','lnc','dcr']
1_sample_A_1-100_seq_3D_ori_exc_5.msr    n=   152,434  fields= 9  subarrays=['itr']
2_3C_measurement.msr                     n=   570,844  fields=20  subarrays=['loc','lnc','dcr']
3_Bonly_MRED_75pM_MINFLUX_3D.msr         n=13,992,124  fields=20  subarrays=['loc','lnc','dcr']
4_3D_PAINT_2C_ratiometric_ALFA-25pM      n=   183,619  fields=20  subarrays=['loc','lnc','dcr']
```

The m2205 files (`1.msr`, `1_sample_A…`) nest a *structured* `itr`, which is harder
still. Scalar-only structured dtypes work fine; only subarray fields fail.

### 4.2 Root cause — an upstream bug, not a design decision

`parse_data_type` resolves the dtype correctly to
`Struct(fields=(('tid', Int32), ('loc', RawBytes(length=24))))`. The failure is
downstream, computing a default fill value:

```python
# zarr/core/dtype/npy/structured.py
def default_scalar(self):
    return self._cast_scalar_unchecked(0)        # passes int 0 ...

def _cast_scalar_unchecked(self, data):
    return np.array([data], dtype=na_dtype)[0]   # ... into a RawBytes field

TypeError: a bytes-like object is required, not 'int'
```

Present in 3.3.0. Worth filing upstream — but the `.msr` reader is the most-used
path in the app and must not depend on someone else's release schedule.

### 4.3 Fix: `minflux_viewer/msr/zarr2.py` (implemented, 536 lines)

A self-contained zarr-v2 reader/writer depending only on **numpy + numcodecs**, not
on zarr-python. Mirrors the small zarr API the `.msr` path uses (`open` → `Group`,
nested `group[path]`, `in`, `.attrs`, `.visititems`, `array[...]`, `np.asarray`,
plus `group[path] = arr` / `require_group` for writing).

Validation:

- **Byte-identical (md5) to zarr-python 2 on all six real files**, including 14M rows
  in 2.1–2.7 s; `visititems` finds the identical node set.
- **What it writes is genuine zarr v2** — verified by reading it back with the real
  zarr 2 library (attrs included), so `.msr` files we produce stay interoperable.
- Rejects zarr v3 stores and malformed dtypes rather than guessing (project rule 8).

**Rewiring is 13 lines across 5 files** — `msr/mfxdta.py`, `msr/writer.py`,
`msr/io.py`, `msr/msr_parser.py`, `plugins/msr_reader/msr_reader_dialog.py`. The
`.msr` path now contains zero `import zarr`.

**Result: all 6 files parse end-to-end under zarr-python 3.3.0**, multi-channel, MBM
beads, both m2410 and m2205, correct `source_version` detection.

---

## 5. Conclusion

1. **Saving is lossy and the losses are silent.** The worst case is a 100 nm channel
   displacement with no error, no warning, no log line, in a technique whose value
   proposition is 1–5 nm precision. Three format combinations write files that can
   never be reopened, and the save reports success in all of them.

2. **The recipe is not a recipe.** It records 3 of ~20 operations while claiming in
   its own schema to be a processing recipe. The data needed for the rest already
   exists in memory and is discarded at save.

3. **Zarr is the right target, and it is not an outside choice.** The `.msr` you open
   already *contains* a zarr v2 store; Imspector m2410 exports `.zarr` directly;
   pyMINFLUX 0.6 imports it. On top sits OME-NGFF, the bioimaging standard, on the
   same substrate, with RFC-5 (coordinate transformations, v0.6) and RFC-9 (zipped
   OME-Zarr, `.ozx`) addressing our two hardest problems.

4. **Honest limit:** OME-NGFF does **not** standardize localization tables. A 2022
   proposal was deliberately declined — the maintainers judged tabular specs belong
   to AnnData. So *no* format gives a community-blessed schema for MINFLUX
   localizations; that part stays ours to document. This applies equally to every
   alternative considered.

5. **We are closer than the audit suggested.** `core/ome_zarr.py` already writes a
   valid NGFF 0.5 / zarr v3 package carrying raw arrays, processed table, ROIs,
   filters, view state, recipe and journal. What is missing is **the reader** — which
   is also what would have exposed the gap.

### Rejected alternatives

| Candidate | Strength | Why not |
|---|---|---|
| pyMINFLUX `.pmx` | HDF5, MINFLUX-native, stores analysis settings | Their format for their tool; inheriting a schema we don't control, no broader community than inventing our own |
| `.smlm` | Designed for localization data | Effectively dormant — no active development, no maintained library, no growing user base |
| SpatialData (scverse) | Genuinely the right model (points, shapes, transformations first-class); very active | Wrong community (spatial omics); pulls dask + geopandas + anndata + pyarrow into a Qt desktop app. **Track it** — points-as-Parquet is the model to borrow if it stabilizes |
| Picasso HDF5 | Real adoption in DNA-PAINT/SMLM | Flat `/locs` + YAML; no iterations, ROIs, transformations, or extension mechanism. Keep as an export target |
| Parquet alone | Best-in-class columnar compression | A table format, not a container — no place for ROIs, images, nested results |
| Plain HDF5, own layout | One file, mature, already a dependency | Loses on size, write and read simultaneously; needs a plugin for competitive compression which then breaks portability; no community spec to inherit |

---

## 6. Decisions taken

| # | Decision | Rationale |
|---|---|---|
| D1 | **Adopt Zarr** as the default save format | §5.3; benchmark §3.1–3.2 |
| D2 | **Adopt zarr v3 with zarr-python 3** | OME-NGFF 0.5+ requires v3; sharding; RFC-9 |
| D3 | **Two interchangeable forms, one layout** — `.zarr.zip` for storage/sharing, `.zarr` **directory** for working | ZIP writes faster at identical size; directory rsyncs incrementally and survives partial writes |
| D4 | **Fix the `.msr` zarr-3 blocker first** | It gates everything else; solution validated |
| D5 | Prefer adopting a community format over inventing one | User requirement |

### Already changed in this repo (prototype, D4)

| Path | Status |
|---|---|
| `minflux_viewer/msr/zarr2.py` | **new**, 536 lines — the shim |
| `tests/test_zarr2_shim.py` | **new**, 209 lines, 16 tests |
| `scripts/zarr3_roundtrip_check.py` | **new**, 305 lines — `.msr` → zarr → fresh-process verify |
| `msr/{io,mfxdta,msr_parser,writer}.py`, `plugins/msr_reader/msr_reader_dialog.py` | rewired to the shim (13 lines) |

**Not changed:** `pyproject.toml` still pins `zarr = ">=2.17,<3.0"`. The shim is a
no-op improvement under zarr 2 today and unblocks zarr 3 whenever the pin moves.

> ⚠ **The working tree was already dirty before this work.** `msr/export.py`,
> `plugins/msr_reader/beads_drift*.py` and parts of `msr/writer.py` /
> `msr_reader_dialog.py` carry unrelated in-progress changes (acquisition-timestamp
> stamping, `iter_load` removal). Do not attribute those to the zarr work. The zarr
> rewiring is exactly the `import zarr` → `from . import zarr2` hunks.

### Test status

- `tests/test_zarr2_shim.py`: **16 passed** under zarr 2; **12 passed, 4 skipped**
  under zarr 3 (skips are the zarr-2 parity references).
- msr + save suites: **131 passed**.
- Full suite, per-file isolation: **1367 passed, 0 failed**.
- ⚠ Two files crash with a Windows access violation — `test_line_end_marker.py`,
  `test_point_roi_session.py`. **Verified identical on the unmodified tree** (via
  `git stash`), so pre-existing Qt/GL teardown, not from this work. It does mean the
  full suite cannot currently run in one process.

---

## 7. Challenges and open questions

### 7.1 Committed but unresolved

| # | Challenge | Detail |
|---|---|---|
| C1 | **Python 3.10 must be dropped** | zarr 3.0–3.1.6 → Python ≥3.11; zarr 3.2–3.3 → ≥3.12. `pyproject.toml` declares `>=3.10,<3.13`, mypy targets 3.10. Venv is 3.12.10 so dev is fine. Pin `zarr>=3.1,<3.2` to keep 3.11, or take 3.3 and require ≥3.12. **Confirm who runs what before committing.** |
| C2 | **PyInstaller will likely break silently** | zarr 3 discovers codecs via `importlib.metadata` entry points (verified in `zarr.registry`). Entry points are lost in frozen bundles; symptom is a *runtime* missing-codec error, not a build failure. Needs `copy_metadata('zarr')`, `copy_metadata('numcodecs')`, hidden imports for `donfig`/`google_crc32c`, and a frozen smoke test that opens a real store. |
| C3 | **`.zarr.zip` collides with the ROI loader** | `Path("x.zarr.zip").suffix == ".zip"`, which `_route_file` sends to the ROI Manager as an ImageJ RoiSet. Needs content sniffing — cheap, since RFC-9 mandates root `zarr.json` be the first ZIP entry. |
| C4 | **Localization tables remain ours to define** | OME-NGFF will not bless the layout. Interoperability for the localization part comes from documenting it and shipping a reader, not from a badge. |
| C5 | **ZIP stores are rewrite-on-save** | Cannot update one block in place; and assigning `root.attrs` *after* creating arrays appends a **duplicate `zarr.json`** (warning, and a file RFC-9 readers may reject). Metadata must be written once, in order — pass `attributes=` at `create_group`. Discovered by running it. |
| C6 | **Codec choice is load-bearing** | zarr 3's `ZstdCodec` has **no byte-shuffle** and compressed the same columns **28 % worse**. Use `BloscCodec(cname="zstd", clevel=3, shuffle="shuffle")` — restores 1.00× parity with zarr 2. Same issue applies to `ome_zarr.py`'s `GzipCodec` (4× larger than Blosc at equal speed). |
| C7 | **RFC-9 is not final** | "Waiting on reviewers"; prototype implementation; recommends `.ozx`. `ZipStore` works today regardless, so we are not blocked — but do not claim `.ozx` conformance until it lands; follow its ordering rules now so we can. |

### 7.2 Genuinely open — worth an independent opinion

1. **Is the shim the right call, or should we wait / patch upstream?**
   For: 536 lines, validated byte-exact, and it makes the `.msr` path immune to
   further zarr-python churn — exactly what bit us. Against: a second implementation
   to maintain, and a clean upstream fix (`default_scalar` for structured dtypes)
   might be a handful of lines. *File the upstream bug regardless.*

2. **Promote `core/ome_zarr.py` to the default format, or rebuild?**
   It already writes a valid NGFF 0.5 / v3 package with most of the content the new
   format needs, but hand-rolls the v3 encoder (1270 lines) that zarr-python 3 would
   make redundant. Promote-and-slim looks right, but the reader must be written
   either way — and it is the reader that defines the format.

3. **How much of the recipe is replayable vs must-be-frozen?**
   Proposed taxonomy — the distinction is *not* "common vs project-specific":

   | Tier | Operations | Storage rule | Scope |
   |---|---|---|---|
   | T0 import | format detection, iteration materialization, validity gating, unit conversion | parameters only; replay by re-import | common |
   | T1 view/calibration | RIMF, overlay transform, filter specs, LUT, overlay membership, depth range | recipe, never baked (except a deliberate snapshot) | common |
   | T2 derived attrs | `den`, localization precision, confocal mapping, `dst/spd/dt/siz/dur/len` | parameters always; values optionally frozen | common |
   | T3 dataset-producing | ROI crop, duplicate, aggregation, drift correction, channel flatten, DCR/time separation | lineage edge + frozen result | common |
   | T4 analysis products | conv/NPC/curvilinear/rod segmentation, particle averaging, HlyB, plot profiles, precision estimates | **must be frozen** — not reproducible from parameters | **project-specific** |

   T4 involves iterative alignment to convergence, conditional randomization against
   a null, and manual acceptance — re-running does not reliably reproduce the
   artefact a figure was made from. So the format needs **two mechanisms**, not one:
   a replayable recipe for T0–T3, and frozen result objects for T4 in a **versioned
   plugin namespace** (`/analysis/<plugin-id>/<schema>/vN`) so adding a field never
   forces a core version bump and unknown blocks are preserved untouched.

4. **Do plain formats keep the raw/processed choice, or go processed-only?**
   The original instinct was "plain formats carry untouched raw attributes only". The
   evidence complicates it: raw CSV is the *broken* case (F2) while processed CSV is
   what researchers actually want for external analysis. Current recommendation: keep
   both, but make the payload kind explicit and validated (`loc_x/loc_y/loc_z` metres
   = raw; `xnm/ynm/znm` nm = processed) rather than implied by an optional sidecar.

5. **How aggressively to restructure the menus?** Splitting `Save` (project) from
   `Export` (one-way interop) expresses the tier model but moves commands users know.

### 7.3 Proposed tier model (not yet decided)

- **Tier 1 — interchange, plain:** `.csv .json .mat .npy`. Exactly one payload kind,
  never mixed, declared by column naming. No internal columns. **Recipe sidecar
  mandatory.** Explicitly does not preserve ROIs, overlay grouping, analysis results.
- **Tier 2 — self-contained interchange:** `.npz .zarr`. Same payloads, recipe travels
  *inside*. Must have a working loader for both kinds (fixes F3).
- **Tier 3 — project archive:** the Zarr store (D1–D3). Everything, multi-dataset.
  This is what *Save* means.
- **Export (one-way, clearly labelled):** `.msr`, Picasso `.hdf5`, OME-TIFF, plain
  OME-NGFF. Either embed the recipe where we control both ends, or state plainly that
  provenance does not travel (fixes F4).

**One recipe schema, three carriers.** JSON, versioned (`minflux-recipe/v2`), stable
top-level shape covering T0–T3 + a lineage graph; carried as a sidecar beside Tier 1,
embedded in Tier 2, and as a group inside the archive. One schema, one validator, one
set of tests. The filter-preset schema (F9) folds in as a subset.

### 7.4 Revised effort

| Workstream | Days |
|---|---|
| A — zarr-python 3 migration (shim done; 2 call sites + pin + PyInstaller left) | 2 – 3 |
| B — audit correctness fixes F1–F4, F7, F8, F10 + round-trip parity tests | 2 – 3 |
| C — the new default format (revised down after the `ome_zarr.py` discovery: reader, Blosc, routing, recipe v2, lineage) | 5 – 7 |
| D — UI: menu restructure, Preferences as capability declaration | 3 – 3.5 |
| E — tests + format specification document | 3 – 5 |
| **Total** | **15 – 21 d ≈ 3 – 4 weeks** |

**Suggested sequencing.** B first and standalone — independent of the format work,
and users are losing alignment today. Then A. Lock md5 fixtures for the six sample
`.msr` files *before* touching anything: that path carries the most risk and the
least coverage relative to its importance.

---

## 8. Verification checklist for an independent session

Re-run these; do not take them on trust.

```bash
cd /d/Git/minflux-viewer

# 0. baseline - the shim's own tests, under the pinned zarr 2
./.venv/Scripts/python.exe -m pytest tests/test_zarr2_shim.py -q     # expect 16 passed

# 1. reproduce the zarr-3 blocker (throwaway venv, leaves the project alone)
python -m venv /c/temp/z3venv
/c/temp/z3venv/Scripts/python.exe -m pip install "zarr>=3.0" msr-reader pytest
/c/temp/z3venv/Scripts/python.exe -c "
import numpy as np, zarr
from zarr.storage import MemoryStore
dt = np.dtype([('tid', np.int32), ('loc', np.float64, (3,))])
g = zarr.create_group(store=MemoryStore(), overwrite=True)
g['x'] = np.zeros(20, dt)"          # expect TypeError: a bytes-like object is required

# 2. the shim under zarr 3
/c/temp/z3venv/Scripts/python.exe -m pytest tests/test_zarr2_shim.py -q  # 12 passed, 4 skipped

# 3. full .msr round trip, both interpreters
MSR="D:/Workspace/Microscopes/MINFLUX/sample data/2_3C_measurement.msr"
./.venv/Scripts/python.exe        scripts/zarr3_roundtrip_check.py parse "$MSR" --out out_v2
/c/temp/z3venv/Scripts/python.exe scripts/zarr3_roundtrip_check.py parse "$MSR" --out out_v3
/c/temp/z3venv/Scripts/python.exe scripts/zarr3_roundtrip_check.py verify out_v3/data.zarr
/c/temp/z3venv/Scripts/python.exe scripts/zarr3_roundtrip_check.py verify out_v3/data.zarr.zip
/c/temp/z3venv/Scripts/python.exe scripts/zarr3_roundtrip_check.py compare \
    out_v2/roundtrip_manifest.json out_v3/roundtrip_manifest.json
```

**Independently worth re-deriving:**

- **F1** — build a dataset with an `overlay_transform` record, `save_processed(content="raw")`,
  reload, compare `roi_crop.display_xy_filtered` before/after. Expect a shift equal to
  the transform.
- **F2** — save raw to `.csv`, reload, compare `prop.num_loc` against the source.
- **F6** — check whether `dataset_to_mfx_array(ds)["dcr"].ndim == 2` for a 2-colour file.
- **§2.4** — export OME-NGFF from the GUI, then try to drag the result back in.
- **§3.1** — the benchmark is synthetic; re-run against a real large `.msr` if the
  size argument is load-bearing for you.

**Known weak points in this analysis, stated plainly:**

- The benchmark uses synthetic data with a plausible but invented clustering
  structure; real compression ratios will differ.
- The HDF5 comparison uses h5py defaults, not `hdf5plugin` (see §3.2 caveat).
- Community-adoption claims (§5.3) come from web research in Aug 2026, not from
  surveying actual MINFLUX users — **worth checking against your own network.**
- The `ome_zarr.py` assessment is based on one export of one 3-D dataset.
- Effort estimates assume one developer already familiar with the codebase.
- Sample `.msr` files live outside the repo
  (`D:/Workspace/Microscopes/MINFLUX/sample data/`) and are not on CI.

---

## 9. References

External:

- OME-Zarr 0.5 spec — https://ngff.openmicroscopy.org/0.5/
- RFC-5 coordinate systems & transformations — https://ngff.openmicroscopy.org/rfc/5/
- RFC-9 zipped OME-Zarr — https://ngff.openmicroscopy.org/rfc/9/index.html
- OME-Zarr paper — https://link.springer.com/article/10.1007/s00418-023-02209-1
- NGFF table proposal, declined — https://forum.image.sc/t/proposal-for-ome-ngff-table-specification/68908
- Zarr-Python 3 release — https://zarr.dev/blog/zarr-python-3-release/
- pyMINFLUX releases — https://github.com/bsse-scf/pyMINFLUX/releases
- SpatialData design doc — https://spatialdata.scverse.org/en/latest/design_doc.html

In-repo:

- `docs/save-export-design.md` — the previous design note. **Partly superseded:** it
  predates the defects in §2.3 and the Zarr decision, and still presents the
  raw/snapshot × recipe model as sound.
- `minflux_viewer/msr/zarr2.py` · `tests/test_zarr2_shim.py` · `scripts/zarr3_roundtrip_check.py`
- `minflux_viewer/core/save.py` · `minflux_viewer/core/loader.py` · `minflux_viewer/core/ome_zarr.py`

Detailed reports produced during this work (private Claude artifacts — open in a
browser, not fetchable by an agent):

- Save/Export audit — https://claude.ai/code/artifact/6194b2df-7aea-435d-bf85-6b04f4ac2b34
- Format decision record — https://claude.ai/code/artifact/336f2090-631a-487e-bcd2-9724ea534696
- Migration work estimate — https://claude.ai/code/artifact/24124fd4-1336-41b9-b1db-7994f6356eef
