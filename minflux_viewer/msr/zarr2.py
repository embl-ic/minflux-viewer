"""
minflux_viewer.msr.zarr2
========================
A small, self-contained **zarr v2** reader/writer for the stores embedded in
``.msr`` files — deliberately independent of ``zarr-python``.

Why this exists
---------------
Every ``.msr`` embeds its MINFLUX localizations as a zarr v2 store whose ``mfx``
array is a **structured dtype with subarray fields** (``loc`` ``(N,3)``, ``dcr``
``(N,2)``, ``lnc`` ``(N,3)``; older m2205 files also nest a structured ``itr``).
``zarr-python`` 3.x cannot represent that dtype — reading raises

    ValueError: No Zarr data type found that matches
        {'name': [['vld','|b1'], ['dcr','<f8',[2]], ['loc','<f8',[3]]], ...}

and writing fails in ``Structured.default_scalar`` (it feeds the integer ``0``
into a ``RawBytes`` field). Verified against zarr-python 3.3.0 on six real
``.msr`` files: **none** of them can be opened. It is an upstream bug rather
than a design decision — ``parse_data_type`` resolves the dtype correctly and
only the default-fill-value path is broken — but the ``.msr`` reader is the most
important path in the application and must not depend on someone else's release
schedule.

A zarr v2 array is a ``.zarray`` JSON header plus compressed chunk blobs, and
``numcodecs`` (already a dependency) does the codec work. Implementing the small
subset we need is therefore cheap, and it makes the ``.msr`` path immune to
further ``zarr-python`` API churn.

What it covers
--------------
Only what the ``.msr`` path uses, with a deliberately ``zarr``-shaped API so the
call sites read the same as before:

* :func:`open` → :class:`Group` over a ``{key: bytes}`` mapping or a directory.
* ``group[path]`` (nested), ``path in group``, ``group.attrs``,
  ``group.visititems``, ``group.array_keys`` / ``group_keys``.
* ``array[...]``, ``np.asarray(array)``, ``array.shape``, ``array.dtype``.
* Writing: ``group[path] = ndarray``, ``group.require_group``, attribute
  assignment — producing genuine, interoperable zarr v2.

Not covered (raises rather than guessing, per the project's no-silent-fallback
rule): zarr v3 stores, resizing, partial chunk writes, object dtypes.
"""

from __future__ import annotations

import base64
import json
from collections.abc import Callable, Iterator, Mapping
from pathlib import Path
from typing import Any

import numpy as np

__all__ = ["open", "Group", "Array", "Attrs", "ZarrV2Error"]

_ZARRAY = ".zarray"
_ZGROUP = ".zgroup"
_ZATTRS = ".zattrs"

#: Default chunk length (in elements) for arrays written by :meth:`Group.__setitem__`.
DEFAULT_CHUNK = 1 << 18


class ZarrV2Error(ValueError):
    """Raised for malformed or unsupported zarr v2 content."""


# ---------------------------------------------------------------------------
# dtype  <->  zarr v2 JSON
# ---------------------------------------------------------------------------
def dtype_from_json(spec: Any) -> np.dtype:
    """Convert a zarr v2 JSON dtype spec to a :class:`numpy.dtype`.

    Handles the nested structured form Abberior uses, including subarray fields
    (``['loc', '<f8', [3]]``) and structured fields nested inside a field
    (the m2205 ``itr`` layout).
    """
    if isinstance(spec, str):
        return np.dtype(spec)
    if not isinstance(spec, (list, tuple)):
        raise ZarrV2Error(f"unsupported dtype spec: {spec!r}")
    fields: list[tuple] = []
    for field in spec:
        if not isinstance(field, (list, tuple)) or len(field) < 2:
            raise ZarrV2Error(f"unsupported dtype field: {field!r}")
        name, base = str(field[0]), dtype_from_json(field[1])
        if len(field) > 2:                      # subarray, e.g. ['loc','<f8',[3]]
            fields.append((name, base, tuple(int(v) for v in field[2])))
        else:
            fields.append((name, base))
    return np.dtype(fields)


def dtype_to_json(dt: np.dtype) -> Any:
    """Inverse of :func:`dtype_from_json`."""
    if dt.names is None:
        return dt.str
    out: list[Any] = []
    for name in dt.names:
        base = dt.fields[name][0]
        if base.subdtype is not None:
            sub, shape = base.subdtype
            out.append([name, dtype_to_json(sub), list(shape)])
        else:
            out.append([name, dtype_to_json(base)])
    return out


def _decode_fill_value(raw: Any, dt: np.dtype) -> Any:
    """zarr v2 ``fill_value`` → a scalar usable as ``np.full`` fill.

    Structured/void fills are stored base64-encoded; numeric fills may be the
    JSON strings ``NaN`` / ``Infinity`` / ``-Infinity``.
    """
    if raw is None:
        return None
    if dt.names is not None or dt.kind == "V":
        if isinstance(raw, str):
            return np.frombuffer(base64.b64decode(raw), dtype=dt, count=1)[0]
        return None
    if isinstance(raw, str):
        special = {"NaN": np.nan, "Infinity": np.inf, "-Infinity": -np.inf}
        if raw in special:
            return special[raw]
        return np.frombuffer(base64.b64decode(raw), dtype=dt, count=1)[0]
    return raw


# ---------------------------------------------------------------------------
# store adapters
# ---------------------------------------------------------------------------
class _MappingStore:
    """A ``{key: bytes}`` store (what the MFXDTA container decodes to)."""

    def __init__(self, mapping: Mapping[str, bytes] | dict) -> None:
        self._m = mapping

    def get(self, key: str) -> bytes | None:
        value = self._m.get(key)
        return bytes(value) if value is not None else None

    def set(self, key: str, data: bytes) -> None:
        self._m[key] = data

    def keys(self) -> Iterator[str]:
        return iter(list(self._m.keys()))


class _DirectoryStore:
    """A zarr v2 directory on disk."""

    def __init__(self, root: str | Path) -> None:
        self._root = Path(root)

    def get(self, key: str) -> bytes | None:
        path = self._root / key
        try:
            return path.read_bytes()
        except (FileNotFoundError, NotADirectoryError, IsADirectoryError, PermissionError):
            return None

    def set(self, key: str, data: bytes) -> None:
        path = self._root / key
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(data)

    def keys(self) -> Iterator[str]:
        if not self._root.is_dir():
            return iter(())
        return (str(p.relative_to(self._root)).replace("\\", "/")
                for p in self._root.rglob("*") if p.is_file())


def _as_store(store: Any):
    if hasattr(store, "get") and hasattr(store, "keys") and not isinstance(store, (str, Path)):
        if isinstance(store, (dict, Mapping)):
            return _MappingStore(store)
        return store                                    # already a store adapter
    return _DirectoryStore(store)


# ---------------------------------------------------------------------------
# attributes
# ---------------------------------------------------------------------------
class Attrs:
    """A node's ``.zattrs``; mutations are written straight back to the store."""

    def __init__(self, store, path: str) -> None:
        self._store, self._path = store, path

    def _key(self) -> str:
        return f"{self._path}/{_ZATTRS}" if self._path else _ZATTRS

    def asdict(self) -> dict:
        raw = self._store.get(self._key())
        if not raw:
            return {}
        try:
            return json.loads(raw.decode("utf-8"))
        except (ValueError, UnicodeDecodeError) as exc:
            raise ZarrV2Error(f"{self._key()}: invalid .zattrs JSON ({exc})") from exc

    # -- mapping-ish read API (mirrors zarr's Attributes) ------------------
    def __getitem__(self, key: str) -> Any:
        return self.asdict()[key]

    def __contains__(self, key: object) -> bool:
        return key in self.asdict()

    def __iter__(self) -> Iterator[str]:
        return iter(self.asdict())

    def __len__(self) -> int:
        return len(self.asdict())

    def get(self, key: str, default: Any = None) -> Any:
        return self.asdict().get(key, default)

    def keys(self):
        return self.asdict().keys()

    def items(self):
        return self.asdict().items()

    def values(self):
        return self.asdict().values()

    # -- write -------------------------------------------------------------
    def __setitem__(self, key: str, value: Any) -> None:
        data = self.asdict()
        data[str(key)] = value
        self.update(data)

    def update(self, data: dict) -> None:
        self._store.set(self._key(), json.dumps(data).encode("utf-8"))

    def __repr__(self) -> str:            # pragma: no cover - debugging aid
        return f"Attrs({self._path!r}, {self.asdict()!r})"


# ---------------------------------------------------------------------------
# array
# ---------------------------------------------------------------------------
class Array:
    """A zarr v2 array. Metadata is parsed eagerly; chunks are read on demand."""

    def __init__(self, store, path: str, meta: dict) -> None:
        self._store, self._path = store, path
        if int(meta.get("zarr_format", 2)) != 2:
            raise ZarrV2Error(f"{path}: only zarr v2 is supported "
                              f"(got zarr_format={meta.get('zarr_format')!r})")
        self._meta = meta
        self.dtype = dtype_from_json(meta["dtype"])
        self.shape = tuple(int(v) for v in meta["shape"])
        self.chunks = tuple(int(v) for v in meta["chunks"])
        if len(self.chunks) != len(self.shape):
            raise ZarrV2Error(f"{path}: chunk rank {len(self.chunks)} "
                              f"does not match shape rank {len(self.shape)}")
        self._order = meta.get("order", "C")
        self._sep = meta.get("dimension_separator", ".")
        self._fill = _decode_fill_value(meta.get("fill_value"), self.dtype)
        self.attrs = Attrs(store, path)

    # -- introspection ------------------------------------------------------
    @property
    def ndim(self) -> int:
        return len(self.shape)

    @property
    def size(self) -> int:
        return int(np.prod(self.shape)) if self.shape else 0

    @property
    def nchunks(self) -> int:
        return int(np.prod(self._chunk_grid())) if self.shape else 0

    def _chunk_grid(self) -> tuple[int, ...]:
        return tuple(-(-s // c) for s, c in zip(self.shape, self.chunks))

    # -- codecs -------------------------------------------------------------
    def _codecs(self):
        import numcodecs

        compressor = self._meta.get("compressor")
        codec = numcodecs.get_codec(compressor) if compressor else None
        filters = [numcodecs.get_codec(f) for f in (self._meta.get("filters") or [])]
        return codec, filters

    def _decode_chunk(self, raw: bytes, codec, filters, n_items: int) -> np.ndarray:
        buf: Any = raw
        if codec is not None:
            buf = codec.decode(buf)
        for filt in reversed(filters):
            buf = filt.decode(buf)
        if isinstance(buf, np.ndarray):
            buf = buf.tobytes()
        block = np.frombuffer(buf, dtype=self.dtype, count=n_items)
        return block

    # -- data ---------------------------------------------------------------
    def __array__(self, dtype=None, copy=None) -> np.ndarray:
        out = self._read_all()
        if dtype is not None:
            out = out.astype(dtype, copy=False)
        return out

    def __getitem__(self, item) -> np.ndarray:
        return self._read_all()[item]

    def __len__(self) -> int:
        return self.shape[0] if self.shape else 0

    def _read_all(self) -> np.ndarray:
        if not self.shape:                              # 0-d
            raw = self._store.get(f"{self._path}/0")
            if raw is None:
                return np.zeros((), self.dtype)
            codec, filters = self._codecs()
            return self._decode_chunk(raw, codec, filters, 1).reshape(())

        out = np.empty(self.shape, self.dtype)
        if self._fill is not None:
            out[...] = self._fill
        else:
            out[...] = np.zeros((), self.dtype)
        codec, filters = self._codecs()
        grid = self._chunk_grid()
        for flat in range(int(np.prod(grid)) if grid else 0):
            idx = np.unravel_index(flat, grid) if len(grid) > 1 else (flat,)
            key = f"{self._path}/{self._sep.join(str(int(i)) for i in idx)}"
            raw = self._store.get(key)
            if raw is None:                             # missing chunk = fill value
                continue
            n_items = int(np.prod(self.chunks))
            block = self._decode_chunk(raw, codec, filters, n_items)
            block = block.reshape(self.chunks, order=self._order)
            sel = tuple(slice(int(i) * c, min(int(i) * c + c, s))
                        for i, c, s in zip(idx, self.chunks, self.shape))
            trimmed = tuple(slice(0, s.stop - s.start) for s in sel)
            out[sel] = block[trimmed]
        return out

    def __repr__(self) -> str:            # pragma: no cover - debugging aid
        return f"<zarr2.Array {self._path!r} shape={self.shape} dtype={self.dtype}>"


# ---------------------------------------------------------------------------
# group
# ---------------------------------------------------------------------------
class Group:
    """A zarr v2 group, addressable by nested ``a/b/c`` paths."""

    def __init__(self, store, path: str = "") -> None:
        self._store, self._path = store, path.strip("/")
        self.attrs = Attrs(store, self._path)

    # -- helpers ------------------------------------------------------------
    def _abs(self, path: str) -> str:
        path = str(path).strip("/")
        return f"{self._path}/{path}" if self._path else path

    def _meta_at(self, abs_path: str, kind: str) -> dict | None:
        key = f"{abs_path}/{kind}" if abs_path else kind
        raw = self._store.get(key)
        if raw is None:
            return None
        try:
            return json.loads(raw.decode("utf-8"))
        except (ValueError, UnicodeDecodeError) as exc:
            raise ZarrV2Error(f"{key}: invalid JSON ({exc})") from exc

    def _node(self, abs_path: str):
        meta = self._meta_at(abs_path, _ZARRAY)
        if meta is not None:
            return Array(self._store, abs_path, meta)
        if self._meta_at(abs_path, _ZGROUP) is not None:
            return Group(self._store, abs_path)
        return None

    # -- read ---------------------------------------------------------------
    def __getitem__(self, path: str):
        node = self._node(self._abs(path))
        if node is None:
            raise KeyError(path)
        return node

    def __contains__(self, path: object) -> bool:
        try:
            return self._node(self._abs(str(path))) is not None
        except ZarrV2Error:
            return False

    def get(self, path: str, default: Any = None):
        try:
            return self[path]
        except KeyError:
            return default

    def _child_names(self) -> list[str]:
        prefix = f"{self._path}/" if self._path else ""
        names: set[str] = set()
        for key in self._store.keys():
            if prefix and not key.startswith(prefix):
                continue
            rest = key[len(prefix):]
            if "/" not in rest:
                continue
            head, tail = rest.split("/", 1)
            if tail.split("/")[-1] in (_ZARRAY, _ZGROUP, _ZATTRS):
                names.add(head)
        return sorted(names)

    def keys(self) -> list[str]:
        return [n for n in self._child_names() if self._node(self._abs(n)) is not None]

    def array_keys(self) -> list[str]:
        return [n for n in self._child_names()
                if self._meta_at(self._abs(n), _ZARRAY) is not None]

    def group_keys(self) -> list[str]:
        return [n for n in self._child_names()
                if self._meta_at(self._abs(n), _ZARRAY) is None
                and self._meta_at(self._abs(n), _ZGROUP) is not None]

    def visititems(self, func: Callable[[str, Any], Any]) -> Any:
        """Walk every descendant, calling ``func(relative_path, node)``.

        Mirrors ``zarr.Group.visititems``: a non-``None`` return stops the walk.
        """
        base = f"{self._path}/" if self._path else ""

        def walk(group: Group) -> Any:
            for name in group._child_names():
                abs_path = group._abs(name)
                node = group._node(abs_path)
                if node is None:
                    continue
                result = func(abs_path[len(base):], node)
                if result is not None:
                    return result
                if isinstance(node, Group):
                    result = walk(node)
                    if result is not None:
                        return result
            return None

        return walk(self)

    def tree(self) -> list[str]:
        """Flat list of descendant paths — a small debugging convenience."""
        out: list[str] = []
        self.visititems(lambda path, _node: out.append(path) and None)
        return out

    # -- write --------------------------------------------------------------
    def require_group(self, path: str) -> Group:
        abs_path = self._abs(path)
        parts = abs_path.split("/")
        for i in range(len(parts)):
            sub = "/".join(parts[: i + 1])
            if self._meta_at(sub, _ZGROUP) is None and self._meta_at(sub, _ZARRAY) is None:
                self._store.set(f"{sub}/{_ZGROUP}",
                                json.dumps({"zarr_format": 2}).encode("utf-8"))
        return Group(self._store, abs_path)

    create_group = require_group

    def __setitem__(self, path: str, value: np.ndarray) -> None:
        self.create_array(path, np.asarray(value))

    def create_array(self, path: str, data: np.ndarray, *,
                     chunk: int | None = None, compressor: Any = None) -> Array:
        """Write *data* as a zarr v2 array at *path* (creating parent groups)."""
        import numcodecs

        arr = np.ascontiguousarray(np.asarray(data))
        if arr.dtype.hasobject:
            raise ZarrV2Error(f"{path}: object dtypes are not supported")
        abs_path = self._abs(path)
        parent = abs_path.rsplit("/", 1)[0] if "/" in abs_path else ""
        if parent:
            self.require_group(parent[len(self._path):].strip("/")
                               if self._path else parent)

        codec = compressor if compressor is not None else numcodecs.Blosc(
            cname="lz4", clevel=5, shuffle=numcodecs.Blosc.SHUFFLE)
        if arr.ndim == 0:
            arr = arr.reshape(1)
        n = arr.shape[0]
        chunk_len = min(int(chunk or DEFAULT_CHUNK), n) or 1
        chunks = (chunk_len,) + arr.shape[1:]

        self._store.set(f"{abs_path}/{_ZARRAY}", json.dumps({
            "chunks": list(chunks),
            "compressor": codec.get_config(),
            "dtype": dtype_to_json(arr.dtype),
            "fill_value": None,
            "filters": None,
            "order": "C",
            "shape": list(arr.shape),
            "zarr_format": 2,
        }).encode("utf-8"))

        for i in range(-(-n // chunk_len)):
            block = np.ascontiguousarray(arr[i * chunk_len:(i + 1) * chunk_len])
            if block.shape[0] < chunk_len:          # pad the trailing chunk
                pad = np.zeros((chunk_len,) + arr.shape[1:], arr.dtype)
                pad[: block.shape[0]] = block
                block = pad
            # Hand Blosc the ARRAY, not raw bytes: numcodecs then sets
            # typesize = dtype.itemsize, which is what its SHUFFLE filter needs.
            # With .tobytes() the filter saw typesize 1 and had nothing to
            # reorder, making every .msr we write ~1.8x larger than necessary
            # (439 -> 238 MiB of chunks on a 20.6 M-row acquisition, against
            # Imspector's own 292 MiB for the same data).
            self._store.set(f"{abs_path}/{i}", codec.encode(block))
        return Array(self._store, abs_path, self._meta_at(abs_path, _ZARRAY))

    def __repr__(self) -> str:            # pragma: no cover - debugging aid
        return f"<zarr2.Group {self._path or '/'!r} keys={self.keys()}>"


# ---------------------------------------------------------------------------
# entry point
# ---------------------------------------------------------------------------
def open(store: Any, mode: str = "r") -> Group:  # noqa: A001 - mirrors zarr.open
    """Open *store* (a ``{key: bytes}`` mapping or a directory path) as a group.

    ``mode="w"`` initialises an empty root group; ``"r"``/``"a"`` open in place.
    Mirrors the ``zarr.open`` call signature the ``.msr`` code already uses.
    """
    adapter = _as_store(store)
    if mode == "w":
        adapter.set(_ZGROUP, json.dumps({"zarr_format": 2}).encode("utf-8"))
    elif adapter.get(_ZGROUP) is None and adapter.get(_ZARRAY) is None:
        # Tolerated: some embedded stores omit the root .zgroup marker but do
        # carry their arrays. Only reject a store with nothing zarr-like at all.
        if not any(k.endswith((_ZARRAY, _ZGROUP)) for k in adapter.keys()):
            raise ZarrV2Error("not a zarr v2 store (no .zgroup/.zarray found)")
    return Group(adapter, "")
