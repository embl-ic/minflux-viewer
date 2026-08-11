"""ImageJ-compatible **active ROI** stored inside a TIFF file.

ImageJ/Fiji keeps the image's single active ROI in the TIFF itself, as an
``IJMetadata`` tag (code 50839) holding the ROI in ImageJ's own binary ``.roi``
format, alongside an ``IJMetadataByteCounts`` tag (50838).  Re-opening the file
restores the ROI onto the image.  This module is that mechanism for our image
viewer and image writer, so a TIFF we write carries its ROI into ImageJ and a
TIFF ImageJ wrote shows its ROI here.

Scope is deliberately **one ROI per image** — the active ROI, exactly what
ImageJ stores in this tag.  Multi-ROI sets belong in the ROI Manager
(``core/roi.py``, native JSON / ImageJ RoiSet ``.zip``), not in the image file.

Coordinates are ImageJ's: **pixels**, origin at the top-left of the image,
y increasing downwards.  The viewer displays images in nm, so it scales by the
pixel size on the way in and out; :func:`rectangle_roi_from_nm` does that for a
physical rectangle.

Qt-free (only ``numpy``/``roifile``/``tifffile``), so the export path stays
importable without a GUI; the ``RoiRecord`` conversion lives in ``core/roi.py``.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

#: TIFF tag codes ImageJ uses for its metadata block.
IJ_METADATA_TAG = 50839
IJ_METADATA_BYTE_COUNTS_TAG = 50838

#: We always write little-endian TIFFs, so the ImageJ metadata block — whose
#: header and counts are read with the *file's* byte order — must match.
TIFF_BYTEORDER = "<"


@dataclass(frozen=True)
class TiffRoi:
    """An image's active ROI, in ImageJ pixel coordinates.

    *blob* is the raw ImageJ ``.roi`` record — the authoritative form, written
    to and read from the file unchanged, so a shape we cannot model (a composite
    ROI, say) still survives a round trip.  The other fields are the decoded
    summary the viewer draws and reports.
    """

    blob: bytes
    roi_type: str                     # "rectangle" / "oval" / "polygon" / "line" / …
    name: str
    bounds: tuple[float, float, float, float]      # x, y, w, h in pixels
    points: np.ndarray | None = None               # (N, 2) pixel vertices, else None

    def bounds_nm(self, pixel_size_x_nm: float, pixel_size_y_nm: float
                  ) -> tuple[float, float, float, float]:
        x, y, w, h = self.bounds
        return (x * pixel_size_x_nm, y * pixel_size_y_nm,
                w * pixel_size_x_nm, h * pixel_size_y_nm)

    def summary(self) -> str:
        x, y, w, h = self.bounds
        label = f"{self.roi_type} x={x:g}, y={y:g}, w={w:g}, h={h:g} px"
        return f"{self.name}: {label}" if self.name else label


#: roifile ROI_TYPE name → our vocabulary (matches ``core/roi.py``'s types).
_TYPE_FROM_IMAGEJ = {
    "rect": "rectangle",
    "oval": "oval",
    "polygon": "polygon",
    "freehand": "freehand",
    "traced": "polygon",
    "line": "line",
    "polyline": "polyline",
    "freeline": "freehand_line",
    "angle": "angle",
    "point": "point",
}


def _imagej_roi_summary(roi) -> tuple[str, str, tuple[float, float, float, float], np.ndarray | None]:
    """(type, name, pixel bounds, vertices) for a decoded ``roifile.ImagejRoi``."""
    type_name = str(getattr(getattr(roi, "roitype", None), "name", "")).lower()
    roi_type = _TYPE_FROM_IMAGEJ.get(type_name, "polygon")
    name = str(getattr(roi, "name", "") or "")

    points: np.ndarray | None = None
    if roi_type not in {"rectangle", "oval"}:
        try:
            coords = np.asarray(roi.coordinates(), dtype=float)
            if coords.ndim == 2 and coords.shape[0] and coords.shape[1] >= 2:
                points = coords[:, :2]
        except Exception:
            points = None

    left, top = float(roi.left), float(roi.top)
    bounds = (left, top, float(roi.right) - left, float(roi.bottom) - top)
    if points is not None and (bounds[2] <= 0 or bounds[3] <= 0):
        lo = points.min(axis=0)
        hi = points.max(axis=0)
        bounds = (float(lo[0]), float(lo[1]), float(hi[0] - lo[0]), float(hi[1] - lo[1]))
    return roi_type, name, bounds, points


def decode_roi(blob: bytes | None) -> TiffRoi | None:
    """Decode an ImageJ ``.roi`` byte record, or ``None`` when absent/unreadable."""
    if not blob:
        return None
    try:
        from roifile import ImagejRoi
        roi = ImagejRoi.frombytes(bytes(blob))
    except Exception:
        return None
    roi_type, name, bounds, points = _imagej_roi_summary(roi)
    return TiffRoi(blob=bytes(blob), roi_type=roi_type, name=name,
                   bounds=bounds, points=points)


def rectangle_roi(x: float, y: float, width: float, height: float,
                  *, name: str = "") -> TiffRoi:
    """An axis-aligned rectangle ROI at pixel ``(x, y)`` of size ``width × height``.

    ImageJ rectangles are integer pixel bounds, so the box is rounded outward —
    the stored ROI never excludes a pixel the requested box touched."""
    from roifile import ROI_TYPE, ImagejRoi

    left = int(np.floor(float(x)))
    top = int(np.floor(float(y)))
    right = int(np.ceil(float(x) + float(width)))
    bottom = int(np.ceil(float(y) + float(height)))
    roi = ImagejRoi(roitype=ROI_TYPE.RECT, left=left, top=top,
                    right=max(right, left + 1), bottom=max(bottom, top + 1),
                    name=str(name))
    return decode_roi(roi.tobytes())


#: Our vocabulary → roifile ROI_TYPE attribute name (inverse of _TYPE_FROM_IMAGEJ).
_TYPE_TO_IMAGEJ = {
    "rectangle": "RECT",
    "oval": "OVAL",
    "polygon": "POLYGON",
    "freehand": "FREEHAND",
    "line": "LINE",
    "polyline": "POLYLINE",
    "freehand_line": "FREELINE",
    "angle": "ANGLE",
    "point": "POINT",
}


def roi_from_shape(roi_type: str, *, bounds=None, points=None, name: str = "") -> TiffRoi | None:
    """Build a :class:`TiffRoi` from a shape in **pixel** coordinates.

    ``rectangle``/``oval`` take *bounds* ``(x, y, w, h)``; every other type takes
    *points*, an ``(N, 2)`` vertex array.  ``None`` when the shape is degenerate
    (no area, or too few vertices to draw)."""
    from roifile import ROI_TYPE, ImagejRoi

    kind = str(roi_type or "").lower()
    if kind in {"rectangle", "oval"}:
        if bounds is None:
            return None
        x, y, w, h = (float(v) for v in bounds)
        if not (w > 0.0 and h > 0.0):
            return None
        if kind == "rectangle":
            return rectangle_roi(x, y, w, h, name=name)
        left, top = int(np.floor(x)), int(np.floor(y))
        roi = ImagejRoi(roitype=ROI_TYPE.OVAL, left=left, top=top,
                        right=max(int(np.ceil(x + w)), left + 1),
                        bottom=max(int(np.ceil(y + h)), top + 1), name=str(name))
        return decode_roi(roi.tobytes())

    pts = np.asarray(points, dtype=float) if points is not None else np.empty((0, 2))
    if pts.ndim != 2 or pts.shape[0] < (1 if kind == "point" else 2):
        return None
    roi = ImagejRoi.frompoints(pts[:, :2], name=str(name))
    roi.roitype = getattr(ROI_TYPE, _TYPE_TO_IMAGEJ.get(kind, "POLYGON"))
    return decode_roi(roi.tobytes())


def rectangle_roi_from_nm(x_nm: float, y_nm: float, width_nm: float, height_nm: float,
                          *, pixel_size_x_nm: float, pixel_size_y_nm: float,
                          name: str = "") -> TiffRoi | None:
    """A rectangle ROI from a physical (nm) box, converted to image pixels.

    Returns ``None`` when the image has no usable pixel calibration — inventing
    a 1 nm/px fallback would silently place the ROI in the wrong spot."""
    try:
        sx = float(pixel_size_x_nm)
        sy = float(pixel_size_y_nm)
    except (TypeError, ValueError):
        return None
    if not (np.isfinite(sx) and np.isfinite(sy)) or sx <= 0.0 or sy <= 0.0:
        return None
    return rectangle_roi(float(x_nm) / sx, float(y_nm) / sy,
                         float(width_nm) / sx, float(height_nm) / sy, name=name)


def roi_extratags(roi: TiffRoi | bytes | None, byteorder: str = TIFF_BYTEORDER):
    """``tifffile`` ``extratags`` carrying *roi* as the image's ImageJ active ROI.

    ``()`` when there is no ROI, so a caller can splat it unconditionally.
    *byteorder* must match the byte order of the TIFF being written — ImageJ's
    metadata header and byte counts are read with the file's order, and a
    mismatch makes the whole block unparseable (silently, on ImageJ's side)."""
    blob = roi.blob if isinstance(roi, TiffRoi) else roi
    if not blob:
        return ()
    import tifffile
    return tifffile.imagej_metadata_tag({"ROI": bytes(blob)}, byteorder)


def read_roi_from_tiff(tiff) -> TiffRoi | None:
    """The active ROI of an open ``tifffile.TiffFile``, or ``None``.

    Reads the ``IJMetadata`` tag directly rather than ``TiffFile.imagej_metadata``:
    that property is gated on the file being an *ImageJ* TIFF, and our writer
    produces **OME**-TIFFs that also carry the tag."""
    try:
        tag = tiff.pages[0].tags.get(IJ_METADATA_TAG)
    except Exception:
        return None
    value = getattr(tag, "value", None)
    if not isinstance(value, dict):
        return None
    return decode_roi(value.get("ROI"))
