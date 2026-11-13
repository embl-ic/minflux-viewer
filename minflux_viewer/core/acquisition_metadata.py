"""MINFLUX **acquisition ROIs** stored in an Imspector ``.msr``.

When the operator places a MINFLUX run they draw one or more rectangles on an
overview image; Imspector keeps those rectangles with the *measurement*, and
redraws them on any image whose scan area covers them.  They live in the
MFXDTA container's zarr **root** ``.zattrs``::

    {"rois": [{"corners": [[x0, y0], [x1, y1]],
               "type": "ROI", "linked_dta": "<dataset did>",
               "linked_cfg": "acquiring mfx", …}, …],
     "version": "2.1"}

``corners`` are **metres** in the same sample/stage frame as ``mfx.loc`` —
verified on real files: the union of a run's ROIs matches that run's
localization extent to ~50 nm.

Nothing in the file binds a ROI to an *image* stack; the association is purely
geometric, which is why :func:`rois_within` takes the image's physical extent.

**Reading.** The ROIs come from each MINFLUX payload stack's root ``.zattrs``
record, read through :func:`~minflux_viewer.core.mfxdta_container.read_root_attrs`,
which inflates only the first kilobytes of the payload. A byte scan of the whole
file used to be the reader; it read every byte of a multi-hundred-MB file (slow
over a network share) and found nothing at all in files whose stacks Imspector
zlib-compresses -- the JSON is not plain text there.

The dataset labels come from the OBF stack headers and footers that opening the
file parses anyway: Imspector keeps the ``{"did":…,"label":…}`` descriptor in each
payload stack's ``minflux`` footer tag, and this application's minimal writer in
the file description.
"""

from __future__ import annotations

import json
import re
import struct
import zlib
from dataclasses import dataclass

import numpy as np

from .mfxdta_container import read_root_attrs

#: Expected failures of reading an OBF file that is missing, truncated or not
#: an OBF file at all -- metadata that cannot be read is absent, not an error.
#: Resource and programming errors (MemoryError, AssertionError, …) propagate.
_UNREADABLE_OBF = (OSError, ValueError, struct.error, EOFError, zlib.error)


@dataclass(frozen=True)
class AcquisitionRoi:
    """One acquisition rectangle, in metres, plus the dataset it belongs to."""

    did: str
    x0: float
    y0: float
    x1: float
    y1: float

    @property
    def bounds(self) -> tuple[float, float, float, float]:
        """``(x, y, width, height)`` in metres, normalised to positive extents."""
        x0, x1 = sorted((self.x0, self.x1))
        y0, y1 = sorted((self.y0, self.y1))
        return x0, y0, x1 - x0, y1 - y0


def _open_obf(path):
    """An ``msr_reader.OBFFile`` for *path*, with the compatibility patches on."""
    from msr_reader import OBFFile

    from ..utils.obf_compat import apply_obf_reader_patches

    apply_obf_reader_patches()
    return OBFFile(str(path))


def _is_byte_stack(header) -> bool:
    """True for a ``uint8`` OBF stack -- the only kind that carries MFXDTA."""
    from msr_reader.obffile import _parse_dtype

    try:
        return np.dtype(_parse_dtype(int(header.dtype))[0]) == np.uint8
    except (TypeError, ValueError, OverflowError):
        return False


def _rois_from_block(block: dict) -> list[AcquisitionRoi]:
    """The well-formed rectangles of one ``{"rois": [...]}`` object.

    The ROIs are optional metadata read during the parse, so a malformed value
    -- not a list, entries that are not objects, corners that are not two
    coordinate pairs -- contributes nothing rather than failing the import of
    the localizations beside it."""
    out: list[AcquisitionRoi] = []
    entries = block.get("rois")
    if not isinstance(entries, list):
        return out
    for entry in entries:
        if not isinstance(entry, dict):
            continue
        corners = entry.get("corners")
        if not (isinstance(corners, list) and len(corners) >= 2):
            continue
        try:
            (x0, y0), (x1, y1) = (corners[0][:2], corners[1][:2])
            out.append(AcquisitionRoi(
                did=str(entry.get("linked_dta") or ""),
                x0=float(x0), y0=float(y0), x1=float(x1), y1=float(y1)))
        except (TypeError, ValueError, IndexError):
            continue
    return out


def rois_from_attrs(attrs) -> list[AcquisitionRoi]:
    """The acquisition rectangles in one zarr ``.zattrs`` document.

    *attrs* is the record's raw JSON (``bytes``/``str``) or the parsed value.
    Every JSON object holding a ``"rois"`` list contributes; unreadable JSON
    (including nesting too deep for the decoder) contributes nothing."""
    if isinstance(attrs, (bytes, bytearray, str)):
        try:
            attrs = json.loads(attrs)
        except (ValueError, RecursionError):     # JSONDecodeError, UnicodeDecodeError
            return []
    out: list[AcquisitionRoi] = []
    pending = [attrs]                            # preorder: document order
    while pending:
        node = pending.pop()
        if isinstance(node, dict):
            if "rois" in node:
                out.extend(_rois_from_block(node))
            pending.extend(reversed(list(node.values())))
        elif isinstance(node, list):
            pending.extend(reversed(node))
    return out


def unique_rois(rois) -> list[AcquisitionRoi]:
    """*rois* in order, each rectangle once.

    A ``.msr`` can embed the same measurement's attributes more than once (a
    run and its aggregated companion share a ``did`` and repeat the ROI list)."""
    found: list[AcquisitionRoi] = []
    seen: set[tuple] = set()
    for roi in rois:
        key = (roi.did, roi.x0, roi.y0, roi.x1, roi.y1)
        if key not in seen:
            seen.add(key)
            found.append(roi)
    return found


def read_acquisition_rois(msr_path) -> list[AcquisitionRoi]:
    """Every MINFLUX acquisition rectangle in *msr_path*, in file order, once.

    Reads the OBF headers, then the first kilobytes of each ``uint8`` stack's
    payload -- inflated when the stack is compressed -- up to the MFXDTA zarr
    root ``.zattrs`` record. A missing or non-OBF file has no ROIs; a stack
    that cannot be decoded contributes none and the others are still read."""
    rois: list[AcquisitionRoi] = []
    try:
        with _open_obf(msr_path) as obf:
            for header in obf.stack_headers:
                if not _is_byte_stack(header):
                    continue
                try:
                    raw = read_root_attrs(obf.fd, header.data_position,
                                          header.data_length, header.compressed)
                except (zlib.error, ValueError, EOFError):     # one corrupt stack
                    continue
                if raw is not None:
                    rois.extend(rois_from_attrs(raw))
    except _UNREADABLE_OBF:
        return []
    return unique_rois(rois)


def rois_within(rois, extent_m) -> list[AcquisitionRoi]:
    """The ROIs of *rois* that lie inside the physical box *extent_m*.

    *extent_m* is ``((x_lo, x_hi), (y_lo, y_hi))`` in metres. Containment is
    total, not overlap: a rectangle only half inside the image would be drawn
    clipped and misread as the acquisition area."""
    if not rois or not extent_m:
        return []
    (x_lo, x_hi), (y_lo, y_hi) = extent_m
    inside = []
    for roi in rois:
        x, y, w, h = roi.bounds
        if x >= x_lo and y >= y_lo and x + w <= x_hi and y + h <= y_hi:
            inside.append(roi)
    return inside


def group_by_dataset(rois) -> dict[str, list[AcquisitionRoi]]:
    """The ROIs of *rois* keyed by the MINFLUX run (``did``) that drew them.

    Only rectangles of the **same** run may be merged: an image wide enough to
    contain several runs would otherwise get one box spanning the empty field
    between them, which marks nothing."""
    out: dict[str, list[AcquisitionRoi]] = {}
    for roi in rois:
        out.setdefault(roi.did, []).append(roi)
    return out


def union_bounds(rois) -> tuple[float, float, float, float] | None:
    """``(x, y, width, height)`` in metres spanning every ROI, or ``None``.

    A MINFLUX run is usually **tiled** over several overlapping rectangles, and
    an image carries a single active ROI, so the acquisition *area* is their
    union box — which on real files matches the run's localization extent."""
    boxes = [roi.bounds for roi in rois]
    if not boxes:
        return None
    x0 = min(b[0] for b in boxes)
    y0 = min(b[1] for b in boxes)
    x1 = max(b[0] + b[2] for b in boxes)
    y1 = max(b[1] + b[3] for b in boxes)
    return x0, y0, x1 - x0, y1 - y0


_DID_LABEL_RE = re.compile(
    rb'"did"\s*:\s*"([0-9a-fA-F-]{36})"\s*,\s*"label"\s*:\s*"([^"]*)"'
)


def _obf_header_texts(obf):
    """Every metadata text an opened OBF file's headers and footers carry, in
    file order: the file description and metadata, then per stack its name,
    description, footer metadata and footer tags."""
    yield obf.main_header.description or ""
    yield from ((obf.main_header.metadata or {}).values())
    for header, footer in zip(obf.stack_headers, obf.stack_footers):
        yield header.name or ""
        yield header.description or ""
        yield getattr(footer, "metadata", "") or ""
        yield from ((getattr(footer, "tag_dictionary", None) or {}).values())


def did_label_map_from_obf(obf) -> dict[str, str]:
    """Map each dataset ``did`` (UUID) → its human label, from an open OBF file.

    Imspector keeps the ``{"did":…,"label":…}`` descriptor in each payload
    stack's ``minflux`` footer tag; this application's minimal writer puts the
    list in the file description. Both are parsed when the file is opened, so
    no payload byte is read. Checked against the former whole-file byte scan
    on every sample file: the same map each time. The first label seen for a
    ``did`` wins."""
    out: dict[str, str] = {}
    for text in _obf_header_texts(obf):
        if not text or '"did"' not in text:
            continue
        for m in _DID_LABEL_RE.finditer(str(text).encode("utf-8")):
            out.setdefault(
                m.group(1).decode("ascii"),
                m.group(2).decode("utf-8", "replace"),
            )
    return out


def extract_did_label_map(msr_path) -> dict[str, str]:
    """Map each dataset ``did`` (UUID) → its human label.

    Modern multi-channel ``.msr`` files embed ``{"did":…,"label":…}`` descriptors
    (e.g. ``1_anti_ELYS_F2_Atto655_100pM``). Newer stores may repeat ``did`` in
    ``mfx/.zattrs``; m2205 stores often keep it only in the enclosing OBF stack
    tag (handled by :func:`extract_minflux_stack_tags`). Returns ``{}`` for early
    single-channel files that carry no labels, and for a missing or non-OBF
    file. Reads headers only (see :func:`did_label_map_from_obf`).
    """
    try:
        with _open_obf(msr_path) as obf:
            return did_label_map_from_obf(obf)
    except _UNREADABLE_OBF:
        return {}
