"""
minflux_viewer.msr.mfxdta
=========================
Decoder for the Imspector ``.msr`` MINFLUX storage format — internally named
``"obf / mfxdta"`` in this project.

In every ``.msr`` file the MINFLUX data lives inside a regular 1-D ``uint8`` OBF
stack whose byte payload is a custom container with the magic ``MFXDTA``. That
container is a flat archive of a **zarr v2 DirectoryStore** (blosc/lz4-compressed
chunks) holding the standard structured ``mfx`` array. Early single-channel files
carry one such stack; modern multi-channel files carry one per channel.

Two layers, decoded here:

    OBF stack (uint8, OMAS_BF_STACK — read by msr-reader)
      └─ MFXDTA container  (this module)
           └─ zarr v2 store (blosc/lz4 chunks)
                └─ structured mfx array  → localizations

Container byte layout (little-endian)::

    [0:7]    magic     "MFXDTA\\0"
    [7:11]   uint32    container version (observed: 2)
    [11:15]  uint32    acquisition unix timestamp
    [15:..]  reserved  zero padding up to the entry table
    entry table, each record:
        uint64  name_len
        bytes   name            (backslash-separated store key)
        uint8   flag            (1 = group node / no payload, 0 = leaf)
        if leaf:
            uint64 data_len
            bytes  data

The archive contains two key subtrees: a zero-length ``sync`` listing (an
Imspector bookkeeping manifest, ignored) and the real ``zarr`` store. The key
separator is version-dependent — ``\\`` in container v2 (early files), ``/`` in
v3 (modern files) — so keys are normalised before use.

The decode is pure Python: the stack bytes come from the cross-platform
``msr-reader`` (``OBFFile.read_stack``); no Abberior ``specpy`` SDK is needed.

This module only decodes the container and reconstructs the zarr store; the
decoded ``mfx`` array is consumed by the existing
``minflux_viewer.core.loader.load_from_mfx_array`` path. MBM/bead data and
channel alignment are intentionally **not** handled here.
"""

from __future__ import annotations

import mmap
import re
import struct
from pathlib import Path

import numpy as np

#: Magic at the start of the OBF stack payload.
MFXDTA_MAGIC = b"MFXDTA"
#: ``source_version`` label written onto datasets loaded from this format.
SOURCE_FORMAT = "obf / mfxdta"


def stack_payload_is_mfxdta(data) -> bool:
    """True when *data* (a stack's pixels) is an MFXDTA container payload."""
    arr = np.ascontiguousarray(data).ravel()
    return (
        arr.dtype == np.uint8
        and arr.size >= 16
        and arr[:6].tobytes() == MFXDTA_MAGIC
    )


def container_version(blob: bytes) -> int:
    """Return the MFXDTA container version (uint32 at offset 7)."""
    if blob[:6] != MFXDTA_MAGIC:
        raise ValueError("not an MFXDTA container")
    return int(struct.unpack_from("<I", blob, 7)[0])


def container_timestamp(blob: bytes) -> int:
    """Return the embedded acquisition unix timestamp (uint32 at offset 11)."""
    if blob[:6] != MFXDTA_MAGIC:
        raise ValueError("not an MFXDTA container")
    return int(struct.unpack_from("<I", blob, 11)[0])


def _looks_like_record(blob: bytes, off: int) -> bool:
    """Cheap validity test that *off* is the start of an entry record."""
    n = len(blob)
    if off + 8 > n:
        return False
    nlen = struct.unpack_from("<Q", blob, off)[0]
    if not (0 < nlen <= 256) or off + 8 + nlen + 1 > n:
        return False
    name = blob[off + 8: off + 8 + nlen]
    if not all(32 <= b < 127 for b in name):
        return False
    return blob[off + 8 + nlen] in (0, 1)


def _first_record_offset(blob: bytes) -> int:
    """Locate the first entry record after the fixed header.

    The header (magic + version + timestamp) is followed by zero padding; the
    first record's ``name_len`` begins at the first non-zero byte after it.
    A short brute-force scan is the fallback if that heuristic misses.
    """
    n = len(blob)
    candidates: list[int] = []
    i = 15                                   # past magic(7) + version(4) + ts(4)
    while i < n and blob[i] == 0:
        i += 1
    candidates.append(i)
    candidates.extend(range(8, min(512, n)))
    for off in candidates:
        if _looks_like_record(blob, off):
            return off
    raise ValueError("MFXDTA: could not locate the entry table")


def parse_mfxdta_entries(blob: bytes) -> dict[str, bytes]:
    """Parse the container into ``{key: payload}`` for every leaf entry.

    Keys keep their original backslash separators. Group nodes carry no
    payload and are omitted.
    """
    if blob[:6] != MFXDTA_MAGIC:
        raise ValueError("not an MFXDTA container")
    n = len(blob)
    pos = _first_record_offset(blob)
    store: dict[str, bytes] = {}
    while pos < n - 8:
        nlen = struct.unpack_from("<Q", blob, pos)[0]
        pos += 8
        if not (0 < nlen <= 1024) or pos + nlen > n:
            break
        name = blob[pos:pos + nlen].decode("latin1")
        pos += nlen
        if pos >= n:
            break
        flag = blob[pos]
        pos += 1
        if flag == 1:                        # group node — no payload
            continue
        if pos + 8 > n:
            break
        dlen = struct.unpack_from("<Q", blob, pos)[0]
        pos += 8
        if pos + dlen > n:
            break
        store[name] = blob[pos:pos + dlen]
        pos += dlen
    return store


def extract_zarr_store(blob: bytes) -> dict[str, bytes]:
    """Return the embedded zarr store as ``{key: bytes}`` with ``/`` keys.

    Only the real ``zarr`` subtree is kept; the ``sync`` manifest is dropped.
    The key separator differs by container version — ``\\`` in v2 (early
    files), ``/`` in v3 (modern files) — so keys are normalised to ``/`` and
    the result is a standard zarr v2 key/value store either way.
    """
    entries = parse_mfxdta_entries(blob)
    store: dict[str, bytes] = {}
    for key, val in entries.items():
        norm = key.replace("\\", "/")
        if norm.startswith("zarr/"):
            store[norm[len("zarr/"):]] = val
    if not any(k.startswith("mfx/") for k in store):
        raise ValueError("MFXDTA: no 'mfx' array found in embedded zarr store")
    return store


def unpack_zarr_store_to_dir(store, dest_dir) -> Path:
    """Write an in-memory zarr key/value store to *dest_dir* as an on-disk
    DirectoryStore. Used for the ``.zarr`` export option (parsing itself keeps
    the store in memory). *store* is a ``{key: bytes}`` mapping."""
    dest = Path(dest_dir)
    dest.mkdir(parents=True, exist_ok=True)
    for key, data in store.items():
        target = dest / key                  # key uses '/'
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_bytes(data)
    return dest


def read_mfxdta_mfx(blob: bytes) -> np.ndarray:
    """Decode an MFXDTA blob straight to the structured ``mfx`` numpy array."""
    import zarr

    store = extract_zarr_store(blob)
    return zarr.open(zarr.storage.KVStore(store), mode="r")["mfx"][:]


#: Matches the embedded ``{"did":"<uuid>","label":"<…>"}`` dataset descriptors.
_DID_LABEL_RE = re.compile(
    rb'"did"\s*:\s*"([0-9a-fA-F-]{36})"\s*,\s*"label"\s*:\s*"([^"]*)"'
)


def extract_did_label_map(msr_path) -> dict[str, str]:
    """Map each dataset ``did`` (UUID) → its human label.

    Modern multi-channel ``.msr`` files embed ``{"did":…,"label":…}`` descriptors
    (e.g. ``1_anti_ELYS_F2_Atto655_100pM``); each MFXDTA store records its ``did``
    in ``mfx/.zattrs``, so the two can be joined to name channels without specpy.
    Returns ``{}`` for early single-channel files (which carry no such labels).
    """
    out: dict[str, str] = {}
    try:
        with open(msr_path, "rb") as fh:
            mm = mmap.mmap(fh.fileno(), 0, access=mmap.ACCESS_READ)
            try:
                for m in _DID_LABEL_RE.finditer(mm):
                    out.setdefault(
                        m.group(1).decode("ascii"),
                        m.group(2).decode("utf-8", "replace"),
                    )
            finally:
                mm.close()
    except Exception:
        pass
    return out


def read_obf_mfxdta_stacks(msr_path) -> list[tuple[int, str, bytes]]:
    """Specpy-free MFXDTA scan using the pure-Python ``msr-reader``.

    Returns ``[(stack_index, description, blob_bytes), …]`` for every uint8
    stack whose payload begins with ``MFXDTA``. The description comes from the
    OBF stack header (``220601-132142_minflux`` in early files; empty in modern
    multi-channel files, where a synthetic ``minflux_<idx>`` is used instead).
    """
    from msr_reader import OBFFile

    out: list[tuple[int, str, bytes]] = []
    with OBFFile(str(msr_path)) as f:
        for i in range(f.num_stacks):
            arr = np.ascontiguousarray(f.read_stack(i)).ravel()
            if arr.dtype != np.uint8 or arr.size < 16:
                continue
            if arr[:6].tobytes() != MFXDTA_MAGIC:
                continue
            try:
                desc = f.stack_headers[i].description or f"minflux_{i}"
            except Exception:
                desc = f"minflux_{i}"
            out.append((i, str(desc), arr.tobytes()))
    return out
