"""
minflux_viewer.core.format_sniff
=================================
Identify a file's real format from its **content** (magic bytes / structure),
independent of its extension. Used to recover from mislabelled files — e.g. a
``.json`` that is actually a NumPy ``.npy`` array, or a data file dropped with
no extension at all.

``sniff_format(path)`` returns one of:

    "npy" | "npz" | "mat" | "xlsx" | "tiff"  — binary, identified by magic bytes
    "zarr"                                  — canonical Zarr directory
    "json" | "delimited"                 — text, identified by structure
    None                                 — unrecognised

Binary results are *authoritative* (magic bytes don't lie); the caller may use
them to override a wrong extension. Text results are advisory — used only when
the extension is otherwise unknown.
"""

from __future__ import annotations

from pathlib import Path

#: Formats identified by an unambiguous binary magic number.
MAGIC_FORMATS = frozenset({"npy", "npz", "mat", "xlsx", "tiff"})

#: File extension → canonical loader format.
EXT_TO_FMT: dict[str, str] = {
    ".mat": "mat", ".npy": "npy", ".npz": "npz",
    ".csv": "spreadsheet", ".tsv": "spreadsheet", ".txt": "spreadsheet",
    ".xlsx": "spreadsheet", ".xlsm": "spreadsheet",
    ".msr": "msr", ".tif": "tiff", ".tiff": "tiff", ".json": "json",
    ".zarr": "zarr",
    # A sealed Zarr store is a zip, and so is an .xlsx — the magic bytes cannot
    # tell them apart. The compound extension is the discriminator, so it is
    # matched (longest-first, see resolve_format) before the plain suffix.
    ".zarr.zip": "zarr",
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
        # Both .npz and .xlsx are zip containers. An .npz stores one ".npy"
        # member per array, so its first local file header names a .npy; an
        # .xlsx names "[Content_Types].xml". Without this an .npz was sniffed
        # as a spreadsheet and routed to the column-mapping importer.
        if b".npy" in head[:512]:
            return "npz"
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
    # Longest-first so a compound extension wins: ".zarr.zip" is a sealed Zarr
    # store, not the ".zip" that a zip magic number would suggest.
    suffixes = [s.lower() for s in Path(path).suffixes]
    ext = suffixes[-1] if suffixes else ""
    ext_fmt = None
    compound = False
    for count in range(len(suffixes), 0, -1):
        tail = "".join(suffixes[-count:])
        if tail in EXT_TO_FMT:
            ext, ext_fmt, compound = tail, EXT_TO_FMT[tail], count > 1
            break
    raw = sniff_format(path)
    sniffed = _SNIFF_TO_FMT.get(raw, raw)

    # A compound extension is an explicit declaration and outranks the magic
    # number, because several unrelated formats share one container: .xlsx,
    # .zarr.zip and a RoiSet .zip are all zips, so "the content is xlsx" really
    # only means "this is a zip" and must not override ".zarr.zip".
    if compound:
        return ext_fmt, ""
    if raw in MAGIC_FORMATS and sniffed != ext_fmt:
        return sniffed, (f"extension '{ext or '(none)'}' but the content is "
                         f"{raw} — loading as {sniffed}")
    if ext_fmt is not None:
        return ext_fmt, ""
    if sniffed is not None:
        return sniffed, (f"no usable extension — content looks like {raw}, "
                         f"loading as {sniffed}")
    return None, ""
