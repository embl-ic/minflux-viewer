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

The decode is pure Python: the stack headers come from the cross-platform
``msr-reader`` (``OBFFile``); no Abberior ``specpy`` SDK is needed. Only the
``uint8`` stacks are read, each with one large read, and their zlib streams
(newer Imspector versions compress every stack) are inflated on a thread pool
while the next stack is read -- ``zlib`` releases the GIL.

The record primitives (``parse_mfxdta_entries`` and friends) live in
:mod:`minflux_viewer.core.mfxdta_container` so the ``core`` metadata readers
can use them; they are re-exported here unchanged.

This module only decodes the container and reconstructs the zarr store; the
decoded ``mfx`` array is consumed by the existing
``minflux_viewer.core.loader.load_from_mfx_array`` path. MBM/bead data and
channel alignment are intentionally **not** handled here.
"""

from __future__ import annotations

import json
import math
import struct
import zlib
from collections import deque
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import numpy as np

from ..core.acquisition_metadata import extract_did_label_map as extract_did_label_map
from ..core.mfxdta_container import MFXDTA_MAGIC as MFXDTA_MAGIC
from ..core.mfxdta_container import _first_record_offset as _first_record_offset
from ..core.mfxdta_container import _looks_like_record as _looks_like_record
from ..core.mfxdta_container import parse_mfxdta_entries as parse_mfxdta_entries

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
    from ..utils import zarr2

    store = extract_zarr_store(blob)
    return np.asarray(zarr2.open(store, mode="r")["mfx"])


def minflux_stack_tags_from_obf(obf) -> dict[int, dict]:
    """Parsed per-stack ``minflux`` JSON tags of an open OBF file, keyed by
    stack index (see :func:`extract_minflux_stack_tags`)."""
    out: dict[int, dict] = {}
    for index, footer in enumerate(obf.stack_footers):
        raw = (getattr(footer, "tag_dictionary", None) or {}).get("minflux")
        if not raw:
            continue
        try:
            tag = json.loads(raw)
        except (TypeError, ValueError, RecursionError):   # incl. JSONDecodeError
            continue
        if isinstance(tag, dict):
            out[index] = tag
    return out


def extract_minflux_stack_tags(msr_path) -> dict[int, dict]:
    """Return parsed per-stack ``minflux`` JSON tags keyed by OBF index.

    The tag is the authoritative identity record for older m2205 MFXDTA
    stores, whose ``mfx/.zattrs`` may omit ``did`` even though the surrounding
    OBF stack records it.  Non-JSON placeholders used by ordinary confocal
    stacks are ignored.
    """
    from msr_reader import OBFFile

    try:
        with OBFFile(str(msr_path)) as obf:
            return minflux_stack_tags_from_obf(obf)
    except (OSError, ValueError, TypeError, IndexError, KeyError, EOFError):
        return {}


def _payload_bytes(header) -> int | None:
    """Declared payload size of a ``uint8`` stack, or None for another dtype.

    ``samples_per_pixel`` is part of it: RGB/RGBA byte stacks are ``uint8``
    too. An unknown dtype flag is simply not a candidate payload."""
    from msr_reader.obffile import _parse_dtype

    try:
        dtype, samples = _parse_dtype(int(header.dtype))
    except (TypeError, ValueError, OverflowError):
        return None
    if np.dtype(dtype) != np.uint8:
        return None
    # Python integers: a corrupt header cannot wrap the product around.
    return math.prod(int(s) for s in header.size) * int(samples)


def _inflate(index: int, raw: bytes, compressed: bool, expected: int) -> bytes:
    blob = zlib.decompress(raw) if compressed else raw
    if len(blob) != expected:
        # What msr-reader's reshape to the declared stack shape rejected.
        raise ValueError(f"OBF stack {index}: payload holds {len(blob)} bytes, "
                         f"its header declares {expected}")
    return blob


def read_mfxdta_stacks_from_obf(obf, *, lease=None) -> list[tuple[int, str, bytes]]:
    """``[(stack_index, description, blob), …]`` for every MFXDTA stack of an
    open OBF file, in file order.

    Only ``uint8`` stacks are read -- image stacks never are -- each with one
    large sequential read. With a thread *lease* from the process budget
    (``core.execution.acquire_cpu_threads``, held by the caller) the zlib
    streams are inflated on its threads while the following stacks are read,
    which overlaps the file transfer (a network share) with decompression. At
    most two stacks per thread are in flight, so the compressed copies do not
    all pile up in memory. Any read or decode failure raises, as a single
    unreadable stack always did.
    """
    candidates = []
    for i, header in enumerate(obf.stack_headers):
        expected = _payload_bytes(header)
        if expected is not None and expected >= 16:
            candidates.append((i, header, expected))

    def description(i, header):
        return str(getattr(header, "description", "") or f"minflux_{i}")

    def read_raw(header):
        obf.fd.seek(header.data_position)
        return obf.fd.read(header.data_length)

    blobs: list[tuple[int, str, bytes]] = []
    if lease is None or int(lease.threads) <= 1 or len(candidates) < 2:
        for i, header, expected in candidates:
            blobs.append((i, description(i, header),
                          _inflate(i, read_raw(header), bool(header.compressed), expected)))
    else:
        with ThreadPoolExecutor(max_workers=int(lease.threads),
                                thread_name_prefix="mfxdta-inflate") as pool:
            in_flight: deque = deque()
            for i, header, expected in candidates:
                if len(in_flight) >= 2 * int(lease.threads):
                    j, header_j, fut = in_flight.popleft()
                    blobs.append((j, description(j, header_j), fut.result()))
                fut = pool.submit(_inflate, i, read_raw(header),
                                  bool(header.compressed), expected)
                in_flight.append((i, header, fut))
            while in_flight:
                j, header_j, fut = in_flight.popleft()
                blobs.append((j, description(j, header_j), fut.result()))
    return [(i, desc, blob) for i, desc, blob in blobs
            if blob[:len(MFXDTA_MAGIC)] == MFXDTA_MAGIC]


def read_obf_mfxdta_stacks(msr_path, *, lease=None) -> list[tuple[int, str, bytes]]:
    """Specpy-free MFXDTA scan using the pure-Python ``msr-reader``.

    Returns ``[(stack_index, description, blob_bytes), …]`` for every uint8
    stack whose payload begins with ``MFXDTA``. The description comes from the
    OBF stack header (``220601-132142_minflux`` in early files; empty in modern
    multi-channel files, where a synthetic ``minflux_<idx>`` is used instead).
    *lease*: see :func:`read_mfxdta_stacks_from_obf`.
    """
    from msr_reader import OBFFile

    with OBFFile(str(msr_path)) as f:
        return read_mfxdta_stacks_from_obf(f, lease=lease)
