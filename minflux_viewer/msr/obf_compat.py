"""
minflux_viewer.msr.obf_compat
=============================
Compatibility patches for the third-party, read-only ``msr-reader`` OBF reader.

``msr_reader.obffile._read_file_header`` unconditionally reads the file-level
metadata tag dictionary whenever ``format_version > 1``, even when the header's
``meta_data_position`` is **0** — the OBF spec's "no metadata" sentinel. It then
seeks to offset 0 and parses the file's magic header as a tag dict, which reads
past EOF and raises ``struct.error: unpack requires a buffer of 4 bytes``. That
aborts opening the file entirely — blocking both the embedded MINFLUX (MFXDTA)
stacks and the legacy image-series tree.

We monkeypatch ``_read_file_header`` to treat a 0 / out-of-range
``meta_data_position`` as "no metadata" (and to tolerate a malformed metadata
block) so such files open normally. Idempotent and safe if upstream is fixed.
"""

from __future__ import annotations

import struct

_PATCHED = False


def apply_obf_reader_patches() -> None:
    """Patch ``msr_reader`` so OBF files with ``meta_data_position == 0`` parse.
    Idempotent; a no-op if ``msr-reader`` is unavailable."""
    global _PATCHED
    if _PATCHED:
        return
    try:
        import msr_reader.obffile as obf
    except Exception:
        return

    def _read_file_header(fd):
        header_fmt = "<10sLQL"
        magic, fmtver, first_stack_pos, descr_len = struct.unpack(
            header_fmt, fd.read(struct.calcsize(header_fmt)))
        if magic != obf.MAGIC_FILE_HEADER:
            raise ValueError("Can not parse file, no magic header found")
        descr = fd.read(descr_len).decode("utf-8")
        meta_pos, metadata = None, None
        if fmtver > 1:
            (meta_pos,) = struct.unpack("<Q", fd.read(struct.calcsize("<Q")))
            # 0 is the "no metadata" sentinel; only read a positive position, and
            # never let a malformed metadata block abort the whole open.
            if meta_pos and meta_pos > 0:
                fd.seek(meta_pos)
                try:
                    metadata = obf._read_tag_dict(fd)
                except Exception:
                    metadata = None
        return obf.OBFHeader(fmtver, descr, first_stack_pos, meta_pos, metadata)

    obf._read_file_header = _read_file_header
    _PATCHED = True
