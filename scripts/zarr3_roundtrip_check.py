#!/usr/bin/env python
"""
Round-trip check:  .msr  ->  zarr store  ->  fresh-process reopen.

Validates the zarr-v3 migration end to end *before* the application's ``zarr``
pin is moved, by running the same script under two interpreters:

* the project venv   (zarr-python 2 -> writes a **zarr v2** store)
* a zarr-3 venv      (zarr-python 3 -> writes a **zarr v3** store)

Both read the ``.msr`` through :mod:`minflux_viewer.msr.zarr2`, the shim that
exists because zarr-python 3 cannot represent the structured ``mfx`` dtype
(subarray fields ``loc``/``lnc``/``dcr``). The store that gets *written* uses the
canonical **flat column** layout (``loc_x``/``loc_y``/``loc_z``, ``dcr_0``/
``dcr_1``, ...), which is plain 1-D arrays and therefore sidesteps that
limitation entirely - this is the layout the new default format is built on.

Only the **first** MINFLUX channel of a multi-channel file is checked; the
manifest records how many the file contains.

The manifest records a per-column md5, so ``verify`` in a *separate* process
proves the data survived the round trip rather than merely that a file opened.

Usage
-----
    python scripts/zarr3_roundtrip_check.py parse  <file.msr> --out <dir>
    python scripts/zarr3_roundtrip_check.py verify <store>            # dir or .zip
    python scripts/zarr3_roundtrip_check.py compare <manifest_a> <manifest_b>

Qt-free by design, so it runs in a bare venv with only numpy + numcodecs + zarr.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import shutil
import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from minflux_viewer.core.save import flatten_mfx_array  # noqa: E402
from minflux_viewer.msr import zarr2  # noqa: E402
from minflux_viewer.msr.mfxdta import (  # noqa: E402
    extract_zarr_store,
    read_obf_mfxdta_stacks,
)

MANIFEST = "roundtrip_manifest.json"


# ---------------------------------------------------------------------------
def zarr_info() -> tuple[str, int]:
    """Installed zarr-python version and the store format it will write."""
    import zarr

    version = zarr.__version__
    return version, 3 if version.startswith("3.") else 2


def _md5(a: np.ndarray) -> str:
    return hashlib.md5(np.ascontiguousarray(a).tobytes()).hexdigest()


def _columns_from_msr(msr_path: Path) -> tuple[dict[str, np.ndarray], dict]:
    """Read every MFXDTA channel of *msr_path* into canonical flat columns."""
    stacks = read_obf_mfxdta_stacks(msr_path)
    if not stacks:
        raise SystemExit(f"{msr_path.name}: no MINFLUX data (image-only .msr)")

    stack_idx, desc, blob = stacks[0]
    store = extract_zarr_store(blob)
    t0 = time.perf_counter()
    mfx = np.asarray(zarr2.open(store, mode="r")["mfx"])
    read_s = time.perf_counter() - t0

    attrs = zarr2.open(store, mode="r")["mfx"].attrs.asdict()
    subarray_fields = [n for n in mfx.dtype.names if mfx.dtype[n].subdtype is not None]
    info = {
        "source": msr_path.name,
        "channels_in_file": len(stacks),
        "stack_index": stack_idx,
        "stack_description": str(desc),
        "did": attrs.get("did", ""),
        "n_rows": int(len(mfx)),
        "n_fields": len(mfx.dtype.names),
        "subarray_fields": subarray_fields,
        "msr_read_seconds": round(read_s, 3),
    }
    return flatten_mfx_array(mfx), info


# ---------------------------------------------------------------------------
def _write_store(columns: dict[str, np.ndarray], path: Path, *, zipped: bool) -> float:
    """Write *columns* as a zarr store; v3 on zarr-python 3, v2 on zarr-python 2."""
    import zarr

    _version, fmt = zarr_info()
    if path.exists():
        shutil.rmtree(path) if path.is_dir() else path.unlink()

    t0 = time.perf_counter()
    if fmt == 3:
        from zarr.codecs import BloscCodec
        from zarr.storage import LocalStore, ZipStore

        # Blosc *with byte-shuffle*, matching what the .msr itself uses. A plain
        # ZstdCodec has no shuffle filter and compresses these numeric columns
        # ~28 % worse - measured, not assumed.
        codec = BloscCodec(cname="zstd", clevel=3, shuffle="shuffle")
        store = ZipStore(str(path), mode="w") if zipped else LocalStore(str(path))
        # Root attributes are passed at creation, not assigned afterwards: a
        # ZipStore cannot rewrite an entry, so a later `root.attrs[...] = ...`
        # appends a *second* `zarr.json` (a duplicate-name warning, and a file
        # RFC-9 readers may reject). Write each metadata key exactly once.
        root = zarr.create_group(
            store=store, overwrite=True,
            attributes={"minflux_viewer": {"format_version": 2,
                                           "layout": "flat_columns"}})
        for name, values in columns.items():
            values = np.ascontiguousarray(values)
            root.create_array(name, shape=values.shape, dtype=values.dtype,
                              chunks=(min(len(values), 1 << 18),),
                              compressors=codec)
            root[name][:] = values
        if zipped:
            store.close()
    else:
        from numcodecs import Blosc

        compressor = Blosc(cname="zstd", clevel=3, shuffle=Blosc.SHUFFLE)
        store = zarr.ZipStore(str(path), mode="w") if zipped else zarr.DirectoryStore(str(path))
        root = zarr.group(store=store, overwrite=True)
        for name, values in columns.items():
            root.create_dataset(name, data=np.ascontiguousarray(values),
                                compressor=compressor)
        root.attrs["minflux_viewer"] = {"format_version": 2, "layout": "flat_columns"}
        if zipped:
            store.close()
    return time.perf_counter() - t0


def _open_store(path: Path):
    """Open a zarr store (directory or .zip) for reading, under either library."""
    import zarr

    _version, fmt = zarr_info()
    if path.suffix.lower() == ".zip":
        if fmt == 3:
            from zarr.storage import ZipStore

            store = ZipStore(str(path), mode="r")
        else:
            store = zarr.ZipStore(str(path), mode="r")
        return zarr.open(store=store, mode="r"), store
    return zarr.open(str(path), mode="r"), None


def _store_size(path: Path) -> int:
    if path.is_file():
        return path.stat().st_size
    return sum(f.stat().st_size for f in path.rglob("*") if f.is_file())


# ---------------------------------------------------------------------------
def cmd_parse(args) -> int:
    msr = Path(args.msr)
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    version, fmt = zarr_info()

    print(f"zarr-python {version}  ->  will write zarr format v{fmt}")
    print(f"reading {msr.name} ...")
    columns, info = _columns_from_msr(msr)

    print(f"  {info['n_rows']:,} rows | {info['n_fields']} fields | "
          f"subarray fields {info['subarray_fields']}")
    print(f"  read via the zarr2 shim in {info['msr_read_seconds']} s")
    print(f"  flattened to {len(columns)} plain 1-D columns")

    digests = {name: _md5(values) for name, values in sorted(columns.items())}
    manifest = {
        "zarr_python": version,
        "zarr_format_written": fmt,
        "source_info": info,
        "n_columns": len(columns),
        "column_md5": digests,
        "dtypes": {k: str(np.asarray(v).dtype) for k, v in sorted(columns.items())},
        "stores": {},
    }

    for label, zipped in (("directory", False), ("zip", True)):
        target = out / ("data.zarr" if not zipped else "data.zarr.zip")
        elapsed = _write_store(columns, target, zipped=zipped)
        size = _store_size(target)
        manifest["stores"][label] = {
            "path": str(target), "bytes": size, "write_seconds": round(elapsed, 3),
        }
        print(f"  wrote {label:9s} {target.name:16s} {size/1e6:8.1f} MB  in {elapsed:5.2f} s")

    (out / MANIFEST).write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    print(f"\nmanifest -> {out / MANIFEST}")
    print("now verify in a FRESH process:")
    print(f"  python {Path(__file__).name} verify {out / 'data.zarr'}")
    print(f"  python {Path(__file__).name} verify {out / 'data.zarr.zip'}")
    return 0


def cmd_verify(args) -> int:
    path = Path(args.store)
    manifest_path = Path(args.manifest) if args.manifest else path.parent / MANIFEST
    if not manifest_path.is_file():
        raise SystemExit(f"manifest not found: {manifest_path}")
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    version, fmt = zarr_info()

    print(f"zarr-python {version}  reading {path.name}")
    print(f"written by zarr-python {manifest['zarr_python']} "
          f"as zarr format v{manifest['zarr_format_written']}")

    t0 = time.perf_counter()
    root, store = _open_store(path)
    names = sorted(root.array_keys())
    values = {name: np.asarray(root[name][:]) for name in names}
    elapsed = time.perf_counter() - t0
    marker = dict(root.attrs).get("minflux_viewer")
    if store is not None:
        store.close()

    expected = manifest["column_md5"]
    missing = sorted(set(expected) - set(names))
    extra = sorted(set(names) - set(expected))
    bad = [n for n in sorted(set(names) & set(expected)) if _md5(values[n]) != expected[n]]

    n_rows = len(next(iter(values.values()))) if values else 0
    print(f"  {len(names)} columns | {n_rows:,} rows | read in {elapsed:.2f} s")
    print(f"  root attrs: {marker}")

    if missing:
        print(f"  MISSING columns: {missing}")
    if extra:
        print(f"  UNEXPECTED columns: {extra}")
    if bad:
        print(f"  CHECKSUM MISMATCH in: {bad}")
    ok = not (missing or extra or bad)
    print("\nPASS - every column byte-identical to the source .msr" if ok
          else "\nFAIL - data changed in the round trip")
    return 0 if ok else 1


def cmd_compare(args) -> int:
    a = json.loads(Path(args.a).read_text(encoding="utf-8"))
    b = json.loads(Path(args.b).read_text(encoding="utf-8"))
    print(f"A: zarr-python {a['zarr_python']} (format v{a['zarr_format_written']})")
    print(f"B: zarr-python {b['zarr_python']} (format v{b['zarr_format_written']})")

    same_source = a["source_info"]["n_rows"] == b["source_info"]["n_rows"]
    diff = [k for k in set(a["column_md5"]) | set(b["column_md5"])
            if a["column_md5"].get(k) != b["column_md5"].get(k)]
    print(f"  same row count : {same_source}")
    print(f"  columns        : {len(a['column_md5'])} vs {len(b['column_md5'])}")
    for label in ("directory", "zip"):
        sa = a["stores"].get(label, {}).get("bytes")
        sb = b["stores"].get(label, {}).get("bytes")
        if sa and sb:
            print(f"  {label:9s} size : {sa/1e6:8.1f} MB  vs {sb/1e6:8.1f} MB "
                  f"({sb/sa:.2f}x)")
    if diff:
        print(f"  DIFFERING COLUMNS: {sorted(diff)}")
    ok = same_source and not diff
    print(f"\n{'PASS' if ok else 'FAIL'} - zarr 2 and zarr 3 stores carry identical data")
    return 0 if ok else 1


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = parser.add_subparsers(dest="cmd", required=True)

    p = sub.add_parser("parse", help="read a .msr and write zarr stores")
    p.add_argument("msr")
    p.add_argument("--out", required=True)
    p.set_defaults(func=cmd_parse)

    v = sub.add_parser("verify", help="reopen a store and checksum it")
    v.add_argument("store")
    v.add_argument("--manifest", default=None)
    v.set_defaults(func=cmd_verify)

    c = sub.add_parser("compare", help="compare two manifests")
    c.add_argument("a")
    c.add_argument("b")
    c.set_defaults(func=cmd_compare)

    args = parser.parse_args()
    return args.func(args)


if __name__ == "__main__":
    raise SystemExit(main())
