"""The ``MFXDTA`` container inside an Imspector ``.msr``, and reading only its start.

Every ``.msr`` carries its MINFLUX data as ``uint8`` OBF stacks whose payload is
an ``MFXDTA`` container: a flat archive of records wrapping a zarr v2 store (see
:mod:`minflux_viewer.msr.mfxdta` for the layout and the full decode). The record
primitives live here, in ``core``, because metadata readers in this layer need
them too, and ``core`` may not import ``msr``.

**Why a streaming record walk.** Newer Imspector versions zlib-compress *every*
OBF stack, so the container's small JSON records -- the acquisition ROIs in the
zarr root ``.zattrs`` among them -- are not visible as plain bytes in the file.
A byte scan of the file finds nothing there, and decoding the whole payload to
reach a few hundred bytes would mean inflating every localization blob. On every
vendor file examined, with and without compression, the root ``.zattrs`` record
sits within the first few kilobytes of the payload, so :func:`read_root_attrs`
walks the records from the start and stops there. A container that stores it
late -- or not at all, like the files this application writes -- is still walked
correctly: the data of every other record is skipped (a seek when the stack is
not compressed, inflated and dropped when it is).

**Record lengths are untrusted.** A damaged container can advertise any 64-bit
length. No read or inflate step is sized by it (each is at most ``_MAX_STEP``),
a length that cannot fit in what is left of the payload is refused before
anything is read, and a record cut short by the end of the payload is absent --
where :func:`parse_mfxdta_entries` stops too.
"""

from __future__ import annotations

import struct
import zlib

#: Magic at the start of the OBF stack payload.
MFXDTA_MAGIC = b"MFXDTA"

#: The zarr root attributes record, after separator normalisation.
ROOT_ATTRS_KEY = "zarr/.zattrs"

_READ_STEP = 64 * 1024
#: The most one raw read or one inflate call produces, whatever size is asked for.
_MAX_STEP = 4 * 1024 * 1024
#: Deflate's maximum expansion (zlib FAQ: 1032:1), to bound what a compressed
#: remainder can still decode to.
_MAX_INFLATE_RATIO = 1032
# Enough of the payload start for _first_record_offset's heuristic.
_HEAD_PROBE = 4096


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
    payload and are omitted. A record cut short by the end of *blob* ends the
    walk, so a *prefix* of a container yields exactly its complete records.
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


def normalise_key(name: str) -> str:
    """A container key with ``/`` separators (v2 containers use ``\\``)."""
    return name.replace("\\", "/")


class StackPayloadStream:
    """One OBF stack's payload as a forward-only byte stream.

    Reads the stack's on-disk bytes in steps and inflates them when the stack
    is zlib-compressed. Only what is asked for is decoded, and :meth:`skip`
    moves past data without keeping it -- by seeking when the stack is not
    compressed. *fd* is a seekable binary file; it is repositioned on every
    read, so it may be shared with other sequential readers on one thread.
    A corrupt compressed stream raises :class:`zlib.error`.
    """

    def __init__(self, fd, data_position: int, data_length: int, compressed: bool) -> None:
        self._fd = fd
        self._next = int(data_position)
        self._end = int(data_position) + int(data_length)
        self._inflater = zlib.decompressobj() if compressed else None
        self._pending = b""                  # compressed input not yet inflated
        self._out = bytearray()              # decoded, not yet consumed
        self._done = False

    def _read_raw(self, size: int) -> bytes:
        size = min(int(size), self._end - self._next)
        if size <= 0:
            return b""
        self._fd.seek(self._next)
        chunk = self._fd.read(size)
        self._next += len(chunk)
        return chunk

    def _fill(self, size: int) -> None:
        """Decode until *size* bytes are held or the payload ends, in steps of
        at most ``_MAX_STEP`` whatever *size* is."""
        while len(self._out) < size and not self._done:
            missing = min(max(_READ_STEP, size - len(self._out)), _MAX_STEP)
            if self._inflater is None:
                chunk = self._read_raw(missing)
                if chunk:
                    self._out += chunk
                else:
                    self._done = True
                continue
            if not self._pending:
                self._pending = self._read_raw(_READ_STEP)
                if not self._pending:
                    self._out += self._inflater.flush()
                    self._done = True
                    continue
            # Bounded output: a highly compressible stretch must not inflate
            # far past what was asked for.
            self._out += self._inflater.decompress(self._pending, missing)
            self._pending = self._inflater.unconsumed_tail
            if self._inflater.eof:
                self._done = True

    def peek(self, size: int) -> bytes:
        """The next *size* bytes (fewer at the end), without consuming them."""
        self._fill(size)
        return bytes(self._out[:size])

    def _at_most_left(self) -> int:
        """An upper bound on the payload bytes not yet consumed."""
        raw_left = max(0, self._end - self._next)
        if self._inflater is None:
            return len(self._out) + raw_left
        if self._done:
            return len(self._out)
        # The slack covers output the inflater may still be holding back.
        return (len(self._out) + _READ_STEP
                + _MAX_INFLATE_RATIO * (len(self._pending) + raw_left))

    def read_exactly(self, size: int) -> bytes | None:
        """The next *size* bytes, or ``None`` when the payload ends first.

        A size that cannot fit in what is left of the payload is refused
        without reading anything; any other is decoded in bounded steps, so
        an untrusted length never sizes an allocation or an inflate call."""
        size = int(size)
        if size > self._at_most_left():
            return None
        self._fill(size)
        if len(self._out) < size:
            return None
        chunk = bytes(self._out[:size])
        del self._out[:size]
        return chunk

    def skip(self, size: int) -> None:
        """Move *size* bytes forward without keeping them."""
        held = min(int(size), len(self._out))
        del self._out[:held]
        size = int(size) - held
        if size <= 0:
            return
        if self._inflater is None:
            self._next = min(self._end, self._next + size)
            return
        while size > 0 and not self._done:
            self._fill(min(size, 4 * _READ_STEP))
            held = min(size, len(self._out))
            del self._out[:held]
            size -= held


def read_root_attrs(fd, data_position: int, data_length: int,
                    compressed: bool) -> bytes | None:
    """The raw zarr root ``.zattrs`` record of one MFXDTA payload stack.

    ``None`` when the payload is not an MFXDTA container or holds no root
    attributes -- including a root record cut short or advertising a length
    the payload cannot hold. Walks the records from the start of the payload,
    reading only their names and lengths and skipping their data, and stops at
    the record; the walk ends where :func:`parse_mfxdta_entries` would stop.
    Raises :class:`zlib.error` for a corrupt compressed stack.
    """
    stream = StackPayloadStream(fd, data_position, data_length, compressed)
    head = stream.peek(_HEAD_PROBE)
    if head[:len(MFXDTA_MAGIC)] != MFXDTA_MAGIC:
        return None
    try:
        stream.skip(_first_record_offset(head))
    except ValueError:                       # no entry table
        return None
    while True:
        field = stream.read_exactly(8)
        if field is None:
            return None
        (nlen,) = struct.unpack("<Q", field)
        if not 0 < nlen <= 1024:
            return None
        name_and_flag = stream.read_exactly(nlen + 1)
        if name_and_flag is None:
            return None
        if name_and_flag[nlen] == 1:         # group node — no payload
            continue
        field = stream.read_exactly(8)
        if field is None:
            return None
        (dlen,) = struct.unpack("<Q", field)
        if normalise_key(name_and_flag[:nlen].decode("latin1")) == ROOT_ATTRS_KEY:
            return stream.read_exactly(dlen)
        stream.skip(dlen)
