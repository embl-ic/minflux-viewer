"""
OBF (``.msr``) image source — a lazy, series-aware image reader that presents
Abberior OBF image stacks through the **same interface as** :class:`~minflux_viewer.core.tiff_source.TiffImageSource`,
so the standalone TIFF viewer can display image series parsed from a ``.msr`` file
(the ones Fiji's Bio-Formats importer lists as loadable series).

An ``.msr`` embeds one OBF stack per "series". Some stacks are genuine 2-D/3-D
images (confocal/overview channels); others are 1-D histograms/populations, and
MINFLUX data is a ``uint8`` MFXDTA blob. Only genuine image stacks are exposed
here as viewable image series (see :func:`is_image_stack`).

Pure helpers (``axes_for_sizes``, ``is_minflux_data_stack``, ``is_image_stack``,
``classify_image_only``) are unit-tested without a file; :class:`ObfImageSource`
wraps ``msr_reader.OBFFile`` and reuses ``TiffMetadata`` so no viewer-side
metadata changes are needed.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np

# Importing the msr package applies the OBF reader compat monkeypatches
# (meta_data_position == 0 sentinel), needed for some legacy files to open.
import minflux_viewer.msr  # noqa: F401

from .tiff_source import (
    MetadataDocument,
    PhysicalSize,
    TiffMetadata,
    _parse_ome_metadata,
    extract_ome_image_xml,
)


def axes_for_sizes(sizes: tuple[int, ...]) -> str:
    """Viewer axis string for an OBF stack of the given (numpy-order) ``sizes``.

    ``read_stack`` returns arrays whose shape already equals ``sizes``. OBF image
    stacks are ``YX`` (2-D) or ``ZYX`` (3-D); a trailing 3/4 dim is treated as RGB
    samples (``S``)."""
    n = len(sizes)
    if n <= 1:
        return "X" if n == 1 else ""
    if n == 2:
        return "YX"
    if n == 3:
        return "YXS" if int(sizes[-1]) in (3, 4) else "ZYX"
    if n == 4:
        return "ZYXS" if int(sizes[-1]) in (3, 4) else "CZYX"
    return "".join("QTCZYX"[-(n):])  # best-effort for exotic stacks


def _nm(pixel_m: float | None) -> PhysicalSize:
    """OBF pixel size (metres) → a ``PhysicalSize`` in nm, ignoring non-spatial
    (huge) values reported for 1-D histogram stacks."""
    try:
        v = float(pixel_m)
    except (TypeError, ValueError):
        return PhysicalSize(None, "nm")
    if not np.isfinite(v) or v <= 0.0 or v > 1e-2:   # > 1 cm/px ⇒ not spatial
        return PhysicalSize(None, "nm")
    return PhysicalSize(v * 1e9, "nm")


def is_minflux_data_stack(stack: dict) -> bool:
    """Does this OBF stack hold MINFLUX localizations (an MFXDTA blob)?

    Imspector stores the MFXDTA container as a plain ``uint8`` OBF stack. It is
    often declared 1-D, but real files also declare it **2-D** — a near-square
    block of bytes (e.g. ``7301 x 7301``) that shape alone cannot tell apart
    from a genuine image, and which was therefore listed and exported as one.
    The stack footer's ``minflux`` tag is authoritative: MINFLUX payload stacks
    carry ``{"type": "data", …}`` (density/trace renders carry other types and
    *are* images). Falls back to the classic 1-D ``uint8`` signature when a file
    carries no such tag."""
    if str(stack.get("minflux_type", "")) == "data":
        return True
    if str(stack.get("minflux_type", "")):
        return False                                   # tagged, but not payload
    return int(stack.get("ndim", 0)) <= 1 and str(stack.get("dtype", "")) == "uint8"


def is_image_stack(stack: dict) -> bool:
    """Is this OBF stack a viewable image series?

    The one predicate shared by the series listing, the viewer source and the
    export, so all three agree on what an image is."""
    return int(stack.get("ndim", 0)) >= 2 and not is_minflux_data_stack(stack)


def classify_image_only(stacks: list[dict]) -> bool:
    """Is a ``.msr`` **image-only** (no MINFLUX data), given per-stack header info?

    *stacks* is ``[{"ndim": int, "dtype": str, "minflux_type": str}, …]``: a file
    with at least one image stack and no MINFLUX payload stack is image-only.
    Header/footer-only (no pixel reads), so it stays cheap even for large
    MINFLUX files."""
    has_image = any(is_image_stack(s) for s in stacks)
    has_mfx = any(is_minflux_data_stack(s) for s in stacks)
    return has_image and not has_mfx


def _stack_dtype_name(header) -> str:
    try:
        from msr_reader.obffile import _parse_dtype
        return np.dtype(_parse_dtype(int(getattr(header, "dtype", 0)))[0]).name
    except Exception:
        return ""


def _open_obf(path):
    from msr_reader import OBFFile
    return OBFFile(str(path))


def _scan_extent(footer) -> tuple[tuple[float, float], tuple[float, float]] | None:
    """The stack's physical XY extent in metres, ``((x_lo, x_hi), (y_lo, y_hi))``.

    Imspector records the scan geometry in the footer's ``imspector`` XML at
    ``ExpControl/scan/range/{x,y}`` as ``off`` (the **centre**), ``len``, ``psz``
    and ``res``.  The frame is the same sample/stage frame as ``mfx.loc``, with
    array row 0 at the low-``y`` edge and column 0 at the low-``x`` edge —
    verified by projecting a run's localizations onto its confocal image, where
    the direct mapping puts them on 3.3x background signal and a flipped ``y``
    puts them on background.

    ``None`` when the stack carries no scan range (derived stacks do not)."""
    xml = (getattr(footer, "tag_dictionary", None) or {}).get("imspector")
    if not xml:
        return None
    try:
        import xml.etree.ElementTree as ET

        root = ET.fromstring(xml)
        spans = []
        for axis in ("x", "y"):
            node = root.find(f".//ExpControl/scan/range/{axis}")
            if node is None:
                return None
            centre = float(node.findtext("off", "nan"))
            length = float(node.findtext("len", "nan"))
            if not (np.isfinite(centre) and np.isfinite(length)) or length <= 0.0:
                return None
            spans.append((centre - length / 2.0, centre + length / 2.0))
        return spans[0], spans[1]
    except Exception:
        return None


def _minflux_tag(footer) -> dict:
    """The stack footer's ``minflux`` tag parsed as a dict (``{}`` when absent).

    Imspector writes it as JSON for MINFLUX-related stacks (``type`` =
    ``data`` / ``density`` / ``trace`` …) and as a small XML placeholder for
    plain confocal channels, which is simply not a dict."""
    raw = (getattr(footer, "tag_dictionary", None) or {}).get("minflux")
    if not raw:
        return {}
    try:
        data = json.loads(raw)
    except Exception:
        return {}
    return data if isinstance(data, dict) else {}


def source_did_from_minflux_tag(tag: dict) -> str:
    """Dataset DID referenced by an image tag across Imspector generations."""
    return str((tag or {}).get("source") or (tag or {}).get("did") or "")


def _scan_stacks(path) -> list[dict]:
    """Header/footer-only scan of every OBF stack: ``sizes``/``ndim``/``dtype``/
    ``name`` plus the ``minflux`` tag's ``type`` and source dataset. No pixel
    data is read.

    ``name`` is the stack name **exactly as the ``.msr`` records it** — it is
    what Imspector shows (``Ch1 {1}``) and what image exports are named after,
    so it must not be reworded here."""
    out: list[dict] = []
    with _open_obf(path) as obf:
        for i in range(obf.num_stacks):
            sizes = tuple(int(s) for s in (obf.shapes[i].sizes or []))
            try:
                pixel = tuple(float(p) for p in (obf.pixel_sizes[i].sizes or []))
            except Exception:
                pixel = ()
            header = obf.stack_headers[i]
            try:
                footer = obf.stack_footers[i]
                tag = _minflux_tag(footer)
                extent = _scan_extent(footer)
                dimension_labels = tuple(getattr(footer, "dimension_labels", ()) or ())
            except Exception:
                tag, extent, dimension_labels = {}, None, ()
            name = (getattr(header, "description", "") or getattr(header, "name", "")
                    or f"Series {i + 1}")
            out.append({
                "raw_index": i,
                "name": str(name).strip() or f"Series {i + 1}",
                "sizes": sizes,
                "ndim": len(sizes),
                "pixel_m": pixel,
                "dtype": _stack_dtype_name(header),
                "minflux_type": str(tag.get("type", "") or ""),
                # m2410 density/value renders use ``source``; older m2205
                # trace/tid renders put the source dataset directly in ``did``.
                # Plain confocal stacks have neither.
                "source_did": source_did_from_minflux_tag(tag),
                "minflux_tag": tag,
                "extent_m": extent,
                "stack_version": int(getattr(header, "stack_version", 0) or 0),
                "data_length": int(getattr(header, "data_length", 0) or 0),
                "data_position": int(getattr(header, "data_position", 0) or 0),
                "compressed": bool(getattr(header, "compressed", False)),
                "offset": tuple(getattr(header, "offset", ()) or ()),
                "length": tuple(getattr(header, "length", ()) or ()),
                "dimension_labels": dimension_labels,
            })
    return out


def _image_stacks(path) -> list[dict]:
    """Every scanned stack of *path* that is a viewable image series."""
    return [s for s in _scan_stacks(path) if is_image_stack(s)]


def msr_is_image_only(path) -> bool:
    """True when *path* is an image-only ``.msr`` (route it to the image viewer)."""
    try:
        return classify_image_only(_scan_stacks(path))
    except Exception:
        return False


def list_obf_image_series(path) -> list[dict]:
    """The viewable image series of a ``.msr``, in file order:
    ``[{"raw_index", "name", "shape_str", "dtype", "source_did"}, …]``.

    ``source_did`` is the ``did`` of the MINFLUX dataset a density/trace render
    was computed from, or ``""`` for a standalone image (a confocal channel)."""
    return [
        {
            "raw_index": s["raw_index"],
            "name": s["name"],
            "shape_str": " x ".join(str(v) for v in s["sizes"]),
            "dtype": s["dtype"],
            "source_did": s["source_did"],
            "minflux_type": s["minflux_type"],
        }
        for s in _image_stacks(path)
    ]


class ObfImageSource:
    """Lazy, series-aware image reader for a ``.msr`` OBF file.

    Mirrors :class:`~minflux_viewer.core.tiff_source.TiffImageSource`: exposes a
    ``TiffMetadata``, ``axis_size``, ``read_plane`` and ``close``, plus series
    helpers (``series_names`` / ``series_summaries`` / ``set_series``). Series are
    the stacks :func:`is_image_stack` accepts — the same set
    :func:`list_obf_image_series` reports, so ``series_index`` addresses that one
    filtered list; ``raw_stack_index`` addresses the underlying OBF stack."""

    def __init__(self, path, *, series_index: int = 0, raw_stack_index: int | None = None) -> None:
        self.path = Path(path)
        self._obf = _open_obf(self.path)
        self._stacks = _image_stacks(self.path)
        if not self._stacks:
            self.close()
            raise ValueError(f"'{self.path.name}' contains no readable OBF image series")
        if raw_stack_index is not None:
            series_index = next(
                (i for i, s in enumerate(self._stacks) if s["raw_index"] == int(raw_stack_index)),
                None,
            )
            if series_index is None:
                self.close()
                raise IndexError(f"OBF stack {raw_stack_index} is not a viewable image series")
        self.series_index = int(np.clip(series_index, 0, len(self._stacks) - 1))
        self._array: np.ndarray | None = None
        self._acq_rois: list | None = None      # lazily scanned once per file
        self._did_labels: dict[str, str] = {}   # MINFLUX run did -> its label
        self.metadata = self._build_metadata()

    # -- series ---------------------------------------------------------------
    @property
    def series_count(self) -> int:
        return len(self._stacks)

    def series_names(self) -> list[str]:
        return [s["name"] for s in self._stacks]

    def series_summaries(self) -> list[dict]:
        return [
            {
                "index": i,
                "raw_index": s["raw_index"],
                "name": s["name"],
                "shape_str": " x ".join(str(v) for v in s["sizes"]),
                "dtype": s["dtype"],
            }
            for i, s in enumerate(self._stacks)
        ]

    def set_series(self, index: int) -> None:
        index = int(index)
        if not (0 <= index < len(self._stacks)):
            raise IndexError(f"OBF series index {index} is out of range")
        self.series_index = index
        self._array = None
        self.metadata = self._build_metadata()

    # -- reading --------------------------------------------------------------
    def axis_size(self, axis: str) -> int:
        return self.metadata.axis_size(axis)

    def _single_run_rois(self):
        """``(label, [AcquisitionRoi, …])`` for the one MINFLUX run this series
        covers, or ``None``.

        An ``.msr`` has no per-image ROI; it has the rectangles the operator drew
        to place each MINFLUX run (see :mod:`minflux_viewer.msr.acquisition_roi`),
        which Imspector redraws on whichever image covers them.  A wide overview
        can cover **several** runs, and merging those would produce one box
        spanning the empty field between them — marking nothing — so an image
        that is not specific to a single run gets no ROI at all."""
        from ..msr.acquisition_roi import (
            group_by_dataset,
            read_acquisition_rois,
            rois_within,
        )
        from ..msr.mfxdta import extract_did_label_map

        extent = self._stacks[self.series_index].get("extent_m")
        if not extent:
            return None
        if self._acq_rois is None:
            self._acq_rois = read_acquisition_rois(self.path)
            self._did_labels = extract_did_label_map(self.path)
        by_run = group_by_dataset(rois_within(self._acq_rois, extent))
        if len(by_run) != 1:
            return None
        did, rois = next(iter(by_run.items()))
        return (self._did_labels.get(did) or "MINFLUX acquisition"), rois

    def _roi_from_box(self, box_m, name: str):
        """A pixel-space rectangle ROI from a stage-coordinate box in metres."""
        from .tiff_roi import rectangle_roi_from_nm

        (x_lo, _x_hi), (y_lo, _y_hi) = self._stacks[self.series_index]["extent_m"]
        x_m, y_m, w_m, h_m = box_m
        return rectangle_roi_from_nm(
            (x_m - x_lo) * 1e9, (y_m - y_lo) * 1e9, w_m * 1e9, h_m * 1e9,
            pixel_size_x_nm=self.metadata.pixel_size_x.nm,
            pixel_size_y_nm=self.metadata.pixel_size_y.nm,
            name=name,
        )

    def active_roi(self):
        """The MINFLUX acquisition area over this series, as a rectangle ROI.

        A run is usually **tiled** over several overlapping rectangles and an
        image carries one active ROI, so this is the box they span — which on
        real files matches the run's localization extent to ~50 nm.  The tiles
        themselves are not exported: individually they say only where the beam
        dwelt within an area the single box already delimits."""
        from ..msr.acquisition_roi import union_bounds

        found = self._single_run_rois()
        if found is None:
            return None
        label, rois = found
        box = union_bounds(rois)
        return None if box is None else self._roi_from_box(box, label)

    def _load_array(self) -> np.ndarray:
        if self._array is None:
            raw = self._stacks[self.series_index]["raw_index"]
            self._array = np.asarray(self._obf.read_stack(raw))
        return self._array

    def read_plane(self, *, t: int = 0, c: int = 0, z: int = 0) -> np.ndarray:
        arr = self._load_array()
        axes = self.metadata.axes
        requested = {"T": int(t), "C": int(c), "Z": int(z)}
        selection: list[int | slice] = []
        for axis in axes:
            if axis in {"Y", "X", "S"}:
                selection.append(slice(None))
            else:
                value = requested.get(axis, 0)
                selection.append(int(np.clip(value, 0, self.axis_size(axis) - 1)))
        return np.squeeze(np.asarray(arr[tuple(selection)]))

    def close(self) -> None:
        try:
            self._obf.close()
        except Exception:
            pass

    # -- metadata -------------------------------------------------------------
    def _build_metadata(self) -> TiffMetadata:
        s = self._stacks[self.series_index]
        sizes = tuple(int(v) for v in s["sizes"])
        whole_ome_xml = self._ome_xml()
        ome_xml = extract_ome_image_xml(whole_ome_xml, int(s["raw_index"]))
        ome_info = _parse_ome_metadata(ome_xml, 0)
        axes = _axes_from_ome_sizes(sizes, ome_info.get("sizes")) or axes_for_sizes(sizes)
        pixel = list(s["pixel_m"]) + [None] * 3
        # pixel[k] aligns to axis k of the (numpy-order) sizes / array.
        pz = (ome_info.get("pixel_size_z")
              or (_nm(pixel[axes.index("Z")]) if "Z" in axes else PhysicalSize(None, "nm")))
        py = (ome_info.get("pixel_size_y")
              or (_nm(pixel[axes.index("Y")]) if "Y" in axes else PhysicalSize(None, "nm")))
        px = (ome_info.get("pixel_size_x")
              or (_nm(pixel[axes.index("X")]) if "X" in axes else PhysicalSize(None, "nm")))
        dtype = s["dtype"] or "unknown"
        acquisition_date = str(ome_info.get("acquisition_date") or "")
        time_interval = ome_info.get("time_interval") or PhysicalSize(None, "s")
        documents: list[MetadataDocument] = []
        if ome_xml:
            documents.append(MetadataDocument("OME-XML", "xml", ome_xml))
        raw = int(s["raw_index"])
        try:
            footer = self._obf.stack_footers[raw]
            imspector_xml = str(
                (getattr(footer, "tag_dictionary", None) or {}).get("imspector") or ""
            )
        except Exception:
            imspector_xml = ""
        if imspector_xml:
            documents.append(MetadataDocument("Imspector XML", "xml", imspector_xml))
        if s.get("minflux_tag"):
            documents.append(MetadataDocument(
                "MINFLUX stack tag", "json",
                json.dumps(s["minflux_tag"], indent=2, ensure_ascii=False, default=str),
            ))
        stack_header = {
            "raw_stack_index": raw,
            "stack_version": s.get("stack_version"),
            "data_length": s.get("data_length"),
            "data_position": s.get("data_position"),
            "compressed": s.get("compressed"),
            "shape": sizes,
            "dtype": dtype,
            "pixel_sizes_m": s.get("pixel_m"),
            "offset": s.get("offset"),
            "length": s.get("length"),
            "dimension_labels": s.get("dimension_labels"),
            "source_did": s.get("source_did"),
        }
        documents.append(MetadataDocument(
            "OBF stack metadata", "json",
            json.dumps(stack_header, indent=2, ensure_ascii=False, default=str),
        ))
        summary = [
            ("File", str(self.path)),
            ("Series", f"{self.series_index + 1} / {len(self._stacks)}  (OBF stack {s['raw_index']})"),
            ("Image name", s["name"]),
            ("Axes", axes),
            ("Shape", " x ".join(str(v) for v in sizes)),
            ("dtype", dtype),
            ("Source", "OBF / .msr"),
            ("OBF stack version", str(s.get("stack_version") or "unknown")),
            ("Pixel size X", px.display()),
            ("Pixel size Y", py.display()),
            ("Pixel size Z", pz.display()),
        ]
        if acquisition_date:
            summary.append(("Acquisition date", acquisition_date))
        if time_interval.value is not None:
            summary.append(("Frame interval", time_interval.display()))
        return TiffMetadata(
            path=self.path,
            series_index=self.series_index,
            series_count=len(self._stacks),
            axes=axes,
            shape=sizes,
            dtype=dtype,
            is_ome=bool(ome_xml),
            is_imagej=False,
            pixel_size_x=px,
            pixel_size_y=py,
            pixel_size_z=pz,
            channel_names=tuple(ome_info.get("channel_names") or ()),
            image_name=str(ome_info.get("image_name") or s["name"]),
            raw_summary=tuple(summary),
            ome_xml=ome_xml,
            acquisition_date=acquisition_date,
            description=str(ome_info.get("description") or ""),
            time_interval=time_interval,
            documents=tuple(documents),
        )

    def _ome_xml(self) -> str | None:
        try:
            xml = self._obf.get_ome_xml_metadata()
            return str(xml) if xml else None
        except Exception:
            return None


def _axes_from_ome_sizes(shape: tuple[int, ...], ome_sizes) -> str:
    """OME's TCZYX sizes expressed in the numpy order used by the viewer."""
    if not isinstance(ome_sizes, dict):
        return ""
    axes = "".join(axis for axis in "TCZYX" if int(ome_sizes.get(axis, 1)) > 1)
    ome_shape = tuple(int(ome_sizes[axis]) for axis in axes)
    return axes if ome_shape == tuple(shape) else ""
