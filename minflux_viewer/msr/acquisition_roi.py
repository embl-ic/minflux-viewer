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

Reading is a byte scan of the file (mmap + a bracket-matched JSON slice), the
same cheap approach ``mfxdta.extract_did_label_map`` uses.  Decoding the MFXDTA
container instead would mean reading every localization blob — tens of
megabytes — just to learn where a rectangle is.
"""

from __future__ import annotations

import json
import mmap
from dataclasses import dataclass

#: Marker of the attribute block that holds the acquisition rectangles.
_ROIS_KEY = b'"rois"'
#: A ``.zattrs`` block is small; refuse to bracket-match beyond this.
_MAX_BLOCK_BYTES = 1 << 20


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


def _json_object_at(data, start: int) -> dict | None:
    """Parse the JSON object that encloses *start* (the offset of a key).

    Scans back to the opening brace, then forward with a brace counter that
    ignores braces inside strings, and hands the slice to ``json.loads``.
    """
    begin = data.rfind(b"{", max(0, start - _MAX_BLOCK_BYTES), start)
    if begin < 0:
        return None
    depth = 0
    in_string = False
    escaped = False
    limit = min(len(data), begin + _MAX_BLOCK_BYTES)
    for pos in range(begin, limit):
        ch = data[pos]
        if in_string:
            if escaped:
                escaped = False
            elif ch == 0x5C:                       # backslash
                escaped = True
            elif ch == 0x22:                       # closing quote
                in_string = False
            continue
        if ch == 0x22:
            in_string = True
        elif ch == 0x7B:                           # {
            depth += 1
        elif ch == 0x7D:                           # }
            depth -= 1
            if depth == 0:
                try:
                    return json.loads(data[begin:pos + 1].decode("utf-8"))
                except Exception:
                    return None
    return None


def _rois_from_block(block: dict) -> list[AcquisitionRoi]:
    out: list[AcquisitionRoi] = []
    for entry in block.get("rois") or []:
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
        except Exception:
            continue
    return out


def read_acquisition_rois(msr_path) -> list[AcquisitionRoi]:
    """Every MINFLUX acquisition rectangle in *msr_path*, in file order.

    Duplicates are dropped: a ``.msr`` can embed the same measurement's
    attributes more than once (a run and its aggregated companion share a
    ``did`` and repeat the ROI list)."""
    found: list[AcquisitionRoi] = []
    seen: set[tuple] = set()
    try:
        with open(msr_path, "rb") as fh:
            data = mmap.mmap(fh.fileno(), 0, access=mmap.ACCESS_READ)
            try:
                pos = data.find(_ROIS_KEY)
                while pos >= 0:
                    block = _json_object_at(data, pos)
                    if block is not None:
                        for roi in _rois_from_block(block):
                            key = (roi.did, roi.x0, roi.y0, roi.x1, roi.y1)
                            if key not in seen:
                                seen.add(key)
                                found.append(roi)
                    pos = data.find(_ROIS_KEY, pos + len(_ROIS_KEY))
            finally:
                data.close()
    except Exception:
        return []
    return found


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
