"""
minflux_viewer.core.format_sniff
=================================
Identify a file's real format from its **content** (magic bytes / structure),
independent of its extension. Used to recover from mislabelled files — e.g. a
``.json`` that is actually a NumPy ``.npy`` array, or a data file dropped with
no extension at all.

``sniff_format(path)`` returns one of:

    "npy" | "mat" | "xlsx" | "tiff"      — binary, identified by magic bytes
    "json" | "delimited"                 — text, identified by structure
    None                                 — unrecognised

Binary results are *authoritative* (magic bytes don't lie); the caller may use
them to override a wrong extension. Text results are advisory — used only when
the extension is otherwise unknown.
"""

from __future__ import annotations

from pathlib import Path

#: Formats identified by an unambiguous binary magic number.
MAGIC_FORMATS = frozenset({"npy", "mat", "xlsx", "tiff"})


def _looks_text(head: bytes) -> str | None:
    """Decode a header as UTF-8 text if it is plausibly text (few control bytes)."""
    if b"\x00" in head:
        return None
    ctrl = sum(1 for b in head if b < 9 or (13 < b < 32))
    if ctrl > 0.05 * max(1, len(head)):
        return None
    try:
        return head.decode("utf-8-sig", errors="strict")
    except UnicodeDecodeError:
        try:
            return head.decode("utf-8", errors="ignore")
        except Exception:
            return None


def sniff_format(path: str | Path) -> str | None:
    """Best-guess format of *path* from its leading bytes. See module docstring."""
    try:
        with open(path, "rb") as fh:
            head = fh.read(2048)
    except OSError:
        return None
    if not head:
        return None

    # --- binary magic numbers (authoritative) ---
    if head[:6] == b"\x93NUMPY":
        return "npy"
    if head[:8] == b"\x89HDF\r\n\x1a\n":
        return "mat"                       # MATLAB v7.3 (HDF5 container)
    if head[:6] == b"MATLAB":
        return "mat"                       # classic "MATLAB 5.0 MAT-file" header
    if head[:4] == b"PK\x03\x04":
        return "xlsx"                      # zip container (xlsx/xlsm)
    if head[:4] in (b"II*\x00", b"MM\x00*"):
        return "tiff"

    # --- text structure (advisory) ---
    text = _looks_text(head)
    if text is None:
        return None
    stripped = text.lstrip()
    if stripped[:1] in ("{", "["):
        return "json"
    first_line = stripped.splitlines()[0] if stripped.splitlines() else ""
    if first_line and any(d in first_line for d in (",", "\t", ";")):
        return "delimited"
    return None
