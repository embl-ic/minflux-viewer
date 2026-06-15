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

#: File extension → canonical loader format.
EXT_TO_FMT: dict[str, str] = {
    ".mat": "mat", ".npy": "npy",
    ".csv": "spreadsheet", ".tsv": "spreadsheet", ".txt": "spreadsheet",
    ".xlsx": "spreadsheet", ".xlsm": "spreadsheet",
    ".msr": "msr", ".tif": "tiff", ".tiff": "tiff", ".json": "json",
}
#: Content-sniff result → canonical loader format (xlsx/delimited → spreadsheet).
_SNIFF_TO_FMT: dict[str, str] = {"xlsx": "spreadsheet", "delimited": "spreadsheet"}


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


def resolve_format(path: str | Path) -> tuple[str | None, str]:
    """Decide the canonical loader format for *path*, returning ``(fmt, note)``.

    A binary magic number that disagrees with the extension wins (mislabelled
    files, e.g. a ``.json`` that is really a ``.npy``); a known extension is
    otherwise trusted; an unknown/missing extension falls back to the sniffed
    content. ``note`` is a human-readable explanation when the extension was
    overridden or absent, else ``""``. ``fmt`` is ``None`` when unidentifiable.
    """
    ext = Path(path).suffix.lower()
    ext_fmt = EXT_TO_FMT.get(ext)
    raw = sniff_format(path)
    sniffed = _SNIFF_TO_FMT.get(raw, raw)

    if raw in MAGIC_FORMATS and sniffed != ext_fmt:
        return sniffed, (f"extension '{ext or '(none)'}' but the content is "
                         f"{raw} — loading as {sniffed}")
    if ext_fmt is not None:
        return ext_fmt, ""
    if sniffed is not None:
        return sniffed, (f"no usable extension — content looks like {raw}, "
                         f"loading as {sniffed}")
    return None, ""
